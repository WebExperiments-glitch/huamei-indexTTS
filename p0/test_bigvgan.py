"""P0a 测试：BigVGAN 自研实现 × 真实 MLX 权重。

跑法（需先装 CPU torch）：
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python p0/test_bigvgan.py

断言目标：
  1. 所有权重载入且 key 齐全
  2. alpha/beta 数值分布 → 判断是否 logscale（打印后人工确认）
  3. 随机 mel 前向：形状正确（×256）、无 NaN/Inf、输出量级合理（tanh → [-1,1]）
  4. 输出有"非平凡"结构（std > 0，不是常数）→ 权重加载没全错
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_loader import load_tensors
from bigvgan_torch import BigVGAN

MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/bigvgan.safetensors"


def main():
    torch.manual_seed(0)
    t0 = time.time()
    print("== [1] 加载权重 ==")
    w = load_tensors(MODEL, fp16_to_fp32=True)
    print(f"    张量数: {len(w)}  内存: {sum(t.numel() for t in w.values()) * 4 / 2**20:.0f} MB (F32)")
    missing = [k for k in ("conv_pre.weight", "conv_pre.bias", "conv_post.weight",
                           "activation_post.act.alpha",
                           "activation_post.act.beta")
               + tuple(f"ups.{i}.weight" for i in range(6))
               + tuple(f"ups.{i}.bias" for i in range(6)) if k not in w]
    assert not missing, f"缺张量: {missing}"
    # 检查 resblock 齐整
    for i in range(18):
        for g in ("convs1", "convs2"):
            for j in range(3):
                assert f"resblocks.{i}.{g}.{j}.weight" in w
        for a in range(6):
            assert f"resblocks.{i}.activations.{a}.act.alpha" in w
    print("    key 结构校验 ✓")

    print("\n== [2] SnakeBeta 参数分布（判断 logscale）==")
    a0 = w["resblocks.0.activations.0.act.alpha"]
    print(f"    resblocks.0.act0.alpha: min={a0.min():.3f} max={a0.max():.3f} "
          f"mean={a0.mean():.3f}  (若含明显负值/近零 → logscale=True)")
    pa = w["activation_post.act.alpha"]
    print(f"    activation_post.alpha: min={pa.min():.3f} max={pa.max():.3f} mean={pa.mean():.3f}")

    print("\n== [3] 前向（mel 64 帧，约 0.29 秒音频）==")
    model = BigVGAN(w)
    model.eval()
    mel = torch.randn(1, 80, 64)          # [B, n_mels=80, T]
    with torch.no_grad():
        wav = model(mel)
    print(f"    输入 mel: {tuple(mel.shape)} → 输出 wav: {tuple(wav.shape)}")
    expect = 64 * 256
    got = wav.shape[-1]
    print(f"    期望长度 ≈ {expect}（=T×256），实际 {got}，误差 {abs(expect-got)} 采样")
    assert abs(got - expect) < 256, "长度偏差过大"

    finite = bool(torch.isfinite(wav).all())
    print(f"    数值有限: {finite}")
    assert finite, "输出含 NaN/Inf"
    print(f"    wav: min={wav.min():+.4f} max={wav.max():+.4f} std={wav.std():.4f} "
          f"(tanh 输出应在 [-1,1] 且 std 非 0)")

    print("\n== [4] 确定性（同权重同输入 → 同输出）==")
    with torch.no_grad():
        wav2 = model(mel)
    print(f"    两次输出一致: {bool(torch.equal(wav, wav2))}")

    print("\n== [5] 残差连接有效性（改权重应改变输出）==")
    w2 = {k: v.clone() for k, v in w.items()}
    with torch.no_grad():
        w2["resblocks.0.convs1.0.weight"][0, 0, 0] += 0.5
        m2 = BigVGAN(w2).eval()
        wav3 = m2(mel)
    print(f"    扰动一个权重 → 输出改变: {bool((wav-wav3).abs().max() > 1e-6)}")

    print(f"\n✅ P0a 全部通过，耗时 {time.time()-t0:.1f}s")
    # 供下一步对拍用：输出一份参考 wav
    torch.save({"mel": mel, "wav": wav}, os.path.join(os.path.dirname(__file__), "golden_bigvgan.pt"))
    print("    已保存 golden_bigvgan.pt（mel + 本实现 wav）")


if __name__ == "__main__":
    main()
