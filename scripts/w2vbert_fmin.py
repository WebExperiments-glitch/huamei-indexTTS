#!/usr/bin/env python3
"""锁定左/右半 80 通道的确切 fmin/fmax（在 0-200Hz 内扫 fmin）"""
import numpy as np

npz = r"d:\DeepSeek前端代码\indexTTS 2.5\scripts\golden\w2vbert_golden.npz"
SR = 16000
z = np.load(npz)
for key in ("chirp", "tone", "noise"):
    sig = z[key + "_sig"].astype(np.float64)
    f = z[key + "_feat"].astype(np.float64)
    L, R = f[:, :80], f[:, 80:]
    best, bs = [], []
    for fmin in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0):
        for fmax in (7600.0, 8000.0):
            h2m = lambda h: 2595 * np.log10(1 + h / 700)
            m2h = lambda m: 700 * (10 ** (m / 2595) - 1)
            nf, nwin, hop, nm = 400, 400, 160, 80
            fftf = np.linspace(0, SR / 2, nf // 2 + 1)
            pts = m2h(h2m(fmin) + (h2m(fmax) - h2m(fmin)) * np.arange(nm + 2) / (nm + 1))
            Fb = np.zeros((nm, nf // 2 + 1))
            for i in range(nm):
                lo, ct, hi = pts[i], pts[i + 1], pts[i + 2]
                up = np.where(fftf <= ct, (fftf - lo) / max(ct - lo, 1e-9), 0.0)
                dn = np.where((fftf > ct) & (fftf <= hi), (hi - fftf) / max(hi - ct, 1e-9), 0.0)
                Fb[i] = up + dn
            win = np.hanning(nwin)
            n = len(sig)
            pad = nwin // 2
            sp = np.pad(sig, (pad, pad))  # 先试 zero-pad 版本
            nfr = int(np.ceil(n / hop)) + 1
            frames = np.stack([sp[s:s + nwin] for s in range(0, nfr * hop, hop)])
            S = np.abs(np.fft.rfft(np.ascontiguousarray(frames[:nfr]) * win, n=nf)) ** 2
            m = S @ Fb.T
            lm = np.log1p(np.maximum(m, 0))
            lmz = (lm - lm.mean(0)) / (lm.std(0) + 1e-5)
            ma = min(lmz.shape[0], L.shape[0])
            cL = np.corrcoef(lmz[:ma].ravel(), L[:ma].ravel())[0, 1]
            cR = np.corrcoef(lmz[:ma].ravel(), R[:ma].ravel())[0, 1]
            # 也试偶数帧（stride2）
            lme = lm[::2]; lmez = (lme - lme.mean(0)) / (lme.std(0) + 1e-5)
            ma2 = min(lmez.shape[0], L.shape[0])
            cLe = np.corrcoef(lmez[:ma2].ravel(), L[:ma2].ravel())[0, 1]
            cRe = np.corrcoef(lmez[:ma2].ravel(), R[:ma2].ravel())[0, 1]
            best.append((max(cL, cLe), cL, cLe, cR, cRe, fmin, fmax))
    best.sort(reverse=True)
    print(f"== {key} ==")
    for b in best[:3]:
        print(f"   maxcorr={b[0]:.4f} (L={b[1]:.4f} Le={b[2]:.4f} R={b[3]:.4f} Re={b[4]:.4f}) fmin={b[5]} fmax={b[6]}")