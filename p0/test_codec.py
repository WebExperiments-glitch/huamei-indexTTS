"""P3a：Semantic Codec 验证 —— 自洽 + 官方 PyTorch 对拍"""
import sys
import os
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/codec.safetensors"

from codec_impl import SemanticCodec, load_codec_weights  # noqa: E402

print("== [1] 加载权重 ==")
w = load_codec_weights(MODEL)
print(f"    张量 {len(w)} 个")
need = ["quantizer.quantizers.0.codebook.weight", "decoder.0.embed.weight",
        "decoder.1.weight", "up.weight"]
missing = [k for k in need if k not in w]
print(f"    关键 key 齐全: {'✓' if not missing else '✗ ' + str(missing)}")
print(f"    codebook {tuple(w['quantizer.quantizers.0.codebook.weight'].shape)}  "
      f"embed {tuple(w['decoder.0.embed.weight'].shape)}")

codec = SemanticCodec(w)
print("\n== [2] 前向自洽 ==")
codes = torch.randint(0, 8192, (1, 32))
out = codec.decode(codes)
print(f"    codes {tuple(codes.shape)} → S_infer {tuple(out.shape)}")
print(f"    期望 [1,64,1024]（2× 上采样）: {'✓' if tuple(out.shape) == (1, 64, 1024) else '✗'}")
print(f"    finite={bool(torch.isfinite(out).all())}  std={out.std():.4f}")
out2 = codec.decode(codes)
print(f"    确定性: {bool(torch.equal(out, out2))}")
# 敏感性：改一个 code 输出必须变
codes3 = codes.clone(); codes3[0, 5] = (codes3[0, 5] + 1234) % 8192
print(f"    扰动敏感: {not bool(torch.equal(out, codec.decode(codes3)))}")

# ---------------- 官方对拍 ----------------
print("\n== [3] 官方 EnhancedCodec 对拍 ==")
REF = r"D:/indexTTS 2.5/reference/index-tts-main"
sys.path.insert(0, REF)
# stub 掉绘图/音频依赖（官方推理路径不用，仅 import 期需要）
for name in ("scipy", "scipy.signal", "matplotlib"):
    if name not in sys.modules:
        m = types.ModuleType(name)
        if name == "scipy.signal":
            m.cosine = lambda n: None
        sys.modules[name] = m
ta = types.ModuleType("torchaudio")
taf = types.ModuleType("torchaudio.functional")
taff = types.ModuleType("torchaudio.functional.functional")
taff._hz_to_mel = lambda x: x
taff._mel_to_hz = lambda x: x
taf.functional = taff
ta.functional = taff
sys.modules["torchaudio"] = ta
sys.modules["torchaudio.functional"] = taf
sys.modules["torchaudio.functional.functional"] = taff

try:
    from indextts.codec.models import EnhancedCodec
    from torch.nn.utils import remove_weight_norm
except Exception as e:
    print(f"    官方 import 失败（跳过对拍）: {type(e).__name__}: {str(e)[:160]}")
    raise SystemExit(0)

off = EnhancedCodec(codebook_size=8192, hidden_size=1024, codebook_dim=8,
                    vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12,
                    num_quantizers=1, downsample_scale=2).eval()
# 剥掉 weight_norm：转换后权重已合并成裸 weight
n_rm = 0
for m in off.modules():
    if hasattr(m, "weight_g"):
        remove_weight_norm(m); n_rm += 1
print(f"    剥离 weight_norm 的层: {n_rm}")

state = {}
for k, v in w.items():
    # conv 类权重 (O,K,I) → torch (O,I,K)；Linear/Embedding/LN 保持
    if k.endswith(".weight") and len(v.shape) == 3:
        v = v.permute(0, 2, 1).contiguous()
    state[k] = v
miss, unexp = off.load_state_dict(state, strict=False)
print(f"    load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("      missing 示例:", miss[:6])
if unexp:
    print("      unexpected 示例:", unexp[:6])

with torch.no_grad():
    mine = codec.decode(codes)
    theirs = off.decode(codes)
print(f"    自研 {tuple(mine.shape)}   官方 {tuple(theirs.shape)}")
if mine.shape == theirs.shape:
    d = (mine - theirs).abs()
    cos = F.cosine_similarity(mine.flatten(), theirs.flatten(), dim=0).item()
    print(f"    cosine = {cos:.6f}   maxΔ = {d.max().item():.3e}   meanΔ = {d.mean().item():.3e}")
    print(f"    {'✅ 逐算子一致' if cos > 0.9999 else '❌ 存在差异'}")
