#!/usr/bin/env python3
"""决胜测试：右半 = 不同 n_fft / 窗的双分支？全扫描一遍"""
import numpy as np

npz = r"d:\DeepSeek前端代码\indexTTS 2.5\scripts\golden\w2vbert_golden.npz"
SR = 16000
sig = np.load(npz)["chirp_sig"].astype(np.float64)
f = np.load(npz)["chirp_feat"].astype(np.float64)
L, R = f[:, :80], f[:, 80:]

def make_fb(nf, nm, fmin, fmax):
    fftf = np.linspace(0, SR / 2, nf // 2 + 1)
    h2m = lambda h: 2595 * np.log10(1 + h / 700)
    m2h = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = m2h(h2m(fmin) + (h2m(fmax) - h2m(fmin)) * np.arange(nm + 2) / (nm + 1))
    Fb = np.zeros((nm, nf // 2 + 1))
    for i in range(nm):
        lo, ct, hi = pts[i], pts[i + 1], pts[i + 2]
        up = np.where(fftf <= ct, (fftf - lo) / max(ct - lo, 1e-9), 0.0)
        dn = np.where((fftf > ct) & (fftf <= hi), (hi - fftf) / max(hi - ct, 1e-9), 0.0)
        Fb[i] = up + dn
    return Fb

def logmel(nf, nwin, hop, nm, fmin, fmax, winname, zop, padmode):
    win = {"hann": np.hanning, "hamming": np.hamming, "bartlett": np.bartlett, "blackman": np.blackman}[winname](nwin)
    pad = nwin // 2
    sp = np.pad(sig, (pad, pad), mode=padmode)
    nfr = int(np.ceil(len(sig) / hop)) + 1
    frames = np.stack([sp[s:s + nwin] for s in range(0, nfr * hop, hop)][:nfr])
    S = np.abs(np.fft.rfft(frames * win, n=nf)) ** 2
    m = S @ make_fb(nf, nm, fmin, fmax).T
    lm = np.log1p(np.maximum(m, 0))
    if zop == "time":
        return (lm - lm.mean(0)) / (lm.std(0) + 1e-5)
    return lm

def bestcorr(cand, tgt):
    out = []
    for off in range(0, 3):
        ma = min(cand.shape[0] - off, tgt.shape[0])
        a, b = cand[off:off + ma].ravel(), tgt[:ma].ravel()
        out.append(np.corrcoef(a, b)[0, 1])
    for sel in (slice(0, None, 2), slice(1, None, 2)):
        c2 = cand[sel]
        ma = min(c2.shape[0], tgt.shape[0])
        out.append(np.corrcoef(c2[:ma].ravel(), tgt[:ma].ravel())[0, 1])
    return max(out)

res = []
for nf in (256, 400, 512, 800):
    for nwin in (256, 320, 400, 512, 800):
        if nwin > nf:
            continue
        for hop in (160, 320):
            for nm in (80, 160):
                for fmax in (8000.0, 7600.0):
                    for winname in ("hann", "hamming", "bartlett", "blackman"):
                        for zop in ("time",):
                            c = logmel(nf, nwin, hop, nm, 0.0, fmax, winname, zop, "constant")
                            cc = c
                            if cc.shape[1] == 80:
                                cL = bestcorr(cc, L)
                                cR = bestcorr(cc, R)
                                res.append((max(cL, cR), cL, cR, nf, nwin, hop, nm, fmax, winname))
                            elif cc.shape[1] == 160:
                                cL = bestcorr(cc[:, :80], L)
                                cR = bestcorr(cc[:, 80:], R)
                                res.append((max(cL, cR), cL, cR, nf, nwin, hop, nm, fmax, winname))
res.sort(key=lambda x: -x[0])
print("TOP")
for r in res[:10]:
    print(f"corr={r[0]:.4f} L={r[1]:.4f} R={r[2]:.4f} nf={r[3]} win={r[4]} hop={r[5]} nm={r[6]} fmax={r[7]} w={r[8]}")