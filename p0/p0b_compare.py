"""P0b：自研 BigVGAN vs 官方 PyTorch BigVGAN —— 数值对拍

方法：
  1. stub 掉官方 utils/meldataset 的重依赖（matplotlib/scipy/librosa —— 推理用不到）
  2. 真实 import 官方 bigvgan 模块（含 alias_free torch 抗混叠）
  3. 同一份 MLX safetensors 权重，分别喂给官方模型与自研模型
  4. 同一 mel 输入，比较输出（预期差异仅来自官方激活的抗混叠滤波）

跑： python p0/p0b_compare.py
"""
import sys
import os
import types

P0 = r"D:/indexTTS 2.5/p0"
REF = r"D:/indexTTS 2.5/reference/index-tts-main"
MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/bigvgan.safetensors"
sys.path.insert(0, P0)
sys.path.insert(0, REF)

# ---- 1) stub 重依赖（官方仅 import，不用于推理）----
def _stub_module(name: str, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

mpl = _stub_module("matplotlib", use=lambda *a, **k: None)
_stub_module("matplotlib.pylab")
scipy = _stub_module("scipy")
_stub_module("scipy.io", wavfile=_stub_module("scipy.io.wavfile", write=lambda *a, **k: None))
scipy.signal = _stub_module("scipy.signal")   # 若 alias-free 用到（实际不用，纯 torch）
meld = _stub_module("indextts.s2mel.modules.bigvgan.meldataset", MAX_WAV_VALUE=32767.0)

import torch
import torch.nn.functional as F

# ---- 2) import 官方 ----
from indextts.s2mel.modules.bigvgan import bigvgan as official
from indextts.s2mel.modules.bigvgan.env import AttrDict
from safetensors_loader import load_tensors
from bigvgan_torch import BigVGAN as MineBigVGAN

# ---- 3) 超参（与权重/官方结构一致）----
h = AttrDict(
    num_mels=80,
    upsample_initial_channel=1536,
    upsample_rates=[4, 4, 2, 2, 2, 2],
    upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    activation="snakebeta",
    snake_logscale=True,
    use_bias_at_final=False,
    use_tanh_at_final=True,
    use_cuda_kernel=False,
)

print("== 加载权重 ==")
w = load_tensors(MODEL, fp16_to_fp32=True)

print("== 构造自研模型 ==")
mine = MineBigVGAN(w).eval()

print("== 构造官方模型（含抗混叠）==")
off = official.BigVGAN(h, use_cuda_kernel=False)
off.remove_weight_norm()

# key 映射：官方 ups 是 ModuleList 包 ModuleList → "ups.{i}.0.weight"；safetensors 无中间 .0
# 布局映射：safetensors 是 MLX 布局 (O,K,I)；官方模型是 torch 布局 → 加载前 permute
state = {}
for k, v in w.items():
    kk = k
    vv = v
    if kk.startswith("ups."):
        # 官方 ups 是 ModuleList 包 ModuleList → key 加一层 .0（weight 和 bias 都要）
        parts = kk.split(".")
        kk = f"ups.{parts[1]}.0.{parts[2]}"
        if kk.endswith(".weight"):
            # ConvTranspose1d 需 (I,O,K)：safetensors (O,K,I) → permute(2,0,1)
            vv = v.permute(2, 0, 1).contiguous()
    elif kk.endswith(".weight") and any(s in kk for s in
          (".convs1.", ".convs2.", "conv_pre.", "conv_post.")):
        # Conv1d 需 (O,I,K)：safetensors (O,K,I) → permute(0,2,1)
        vv = v.permute(0, 2, 1).contiguous()
    state[kk] = vv
missing, unexpected = off.load_state_dict(state, strict=False)
print(f"  load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing 示例(应为抗混叠滤波器等):", missing[:5])
if unexpected:
    print("  unexpected 示例:", unexpected[:5])
off.eval()

# ---- 4) 同一输入前向（三组对照）----
def mx(a, b):
    d = (a - b).abs()
    c = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return c, d.max().item(), d.mean().item()

def dealias(module):
    """把官方模型的 Activation1d 抗混叠 patch 掉，只留裸 SnakeBeta（用于隔离差异源）。"""
    def strip(act):
        act._orig_forward = act.forward
        act.forward = lambda x, a=act: a.act(x)
    for blk in module.resblocks:
        for act in blk.activations:
            strip(act)
    strip(module.activation_post)

torch.manual_seed(0)
mel = torch.randn(1, 80, 64)
with torch.no_grad():
    mine_wav = mine(mel)
    off_wav = off(mel)          # 官方：完整抗混叠
    dealias(off)
    off_da_wav = off(mel)       # 官方：去掉抗混叠

print(f"\n== 三组对照 ==")
print(f"  自研: {tuple(mine_wav.shape)}")
for name, ref in (("官方(含抗混叠)", off_wav), ("官方(去抗混叠)", off_da_wav)):
    c, mx_, mn = mx(mine_wav, ref)
    print(f"  自研 vs {name:<16} cos={c:.6f}  maxΔ={mx_:.5f}  meanΔ={mn:.6f}")
c, mx_, mn = mx(off_wav, off_da_wav)
print(f"  官方含vs去抗混叠             cos={c:.6f}  maxΔ={mx_:.5f}  meanΔ={mn:.6f}")
print(f"\n  判定: 若『自研 vs 官方(去抗混叠)』cos≈1.0000 → 自研与官方核心逻辑逐算子一致；")
print(f"        剩余差异仅来自官方激活前的抗混叠滤波（MLX 转换时已按设计丢弃）。")
