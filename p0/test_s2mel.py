"""P2 测试：s2mel（DiT/LR/CFM）自洽
1) LR 前向 shape 2) DiT 单步 shape/有限 3) CFM 小步数跑通 + 确定性
跑：python p0/test_s2mel.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/s2mel.safetensors"

from s2mel_impl import load_s2mel_weights, CFM, DiT, LengthRegulator  # noqa: E402


def main():
    t0 = time.time()
    print("== [1] 加载权重 ==")
    w = load_s2mel_weights(MODEL)
    print(f"    {len(w)} 张量, {sum(v.numel() for v in w.values())*4/2**20:.0f} MB (F32), "
          f"{time.time()-t0:.0f}s")

    print("\n== [2] LengthRegulator ==")
    lr = LengthRegulator(w)
    B, Tin = 1, 12
    content = torch.randn(B, Tin, 1024)
    ylens = torch.tensor([20])                      # 12 → 20（×1.72 附近）
    out = lr(content, ylens)
    print(f"    content {tuple(content.shape)} → {tuple(out.shape)} "
          f"finite={bool(torch.isfinite(out).all())}")

    print("\n== [3] DiT 单步 ==")
    dit = DiT(w)
    T = 20
    x = torch.randn(1, 80, T)
    px = torch.randn(1, 80, T)
    t = torch.tensor([0.5])
    style = torch.randn(1, 192)
    cond = torch.randn(1, T, 512)
    d = dit(x, px, torch.full((1,), T, dtype=torch.long), t, style, cond)
    print(f"    {tuple(d.shape)}  finite={bool(torch.isfinite(d).all())} std={d.std():.4f}")

    print("\n== [4] CFM 采样（8 步自测）==")
    cfm = CFM(w)
    P, Tg = 8, 12
    mu = torch.randn(1, P + Tg, 512)
    prompt = torch.randn(1, 80, P)
    style2 = torch.randn(1, 192)
    mel = cfm.inference(mu, P + Tg, prompt, style2, n_steps=8, cfg_rate=0.7)
    print(f"    → mel {tuple(mel.shape)}  finite={bool(torch.isfinite(mel).all())} "
          f"std={mel.std():.4f}")

    print("\n== [5] 确定性（固定 seed）+ prompt 区保持 ==")
    torch.manual_seed(0)
    mel2 = cfm.inference(mu, P + Tg, prompt, style2, n_steps=8)
    print(f"    seed=0 两次一致: {bool(torch.equal(mel, mel2))}   (randn 采样 → 需固定 seed)")
    torch.manual_seed(0)
    mel3 = cfm.inference(mu, P + Tg, prompt, style2, n_steps=8)
    print(f"    同 seed 重跑一致: {bool(torch.equal(mel2, mel3))} → "
          f"{'✓' if torch.equal(mel2, mel3) else '✗'}")
    print(f"    prompt 区(前{P})应清零: max={mel2[..., :P].abs().max():.8f} → "
          f"{'✓' if mel2[..., :P].abs().max() < 1e-6 else '✗'}")
    print(f"\n✅ P2 s2mel 自洽完成，耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
