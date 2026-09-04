#!/usr/bin/env python3
"""w2vbert_probe.py —— w2v-bert-2.0 输入特征配方黑盒对拍工具

用途（0 到 1 自研路线下的"官方黑盒标定"）：
  1. 用官方 SeamlessM4TFeatureExtractor 跑已知刺激信号，得到精确的 [B,T,C] 数值参照；
  2. 逆向比对"手工 mel 滤组配方"，锁定 n_fft / hop / n_mels / 对数形式 / 时间 stride；
  3. 落盘 golden npz（scripts/golden/w2vbert_golden.npz），供 iOS Swift 实现逐值对拍。

不读取任何第三方实现源码，仅观测输入输出数值（黑盒标定，与 p0 官方对拍同方法论）。
"""
import os, sys, json
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor

W2V_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "hf_cache", "w2v-bert-2.0")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
SR = 16000


# ---------------------------------------------------------------- 黑盒取样
def probe_extractor():
    fex = SeamlessM4TFeatureExtractor.from_pretrained(W2V_DIR, local_files_only=True)
    print("== extractor config ==")
    print(json.dumps(fex.to_dict(), indent=1, ensure_ascii=False))
    return fex


def make_signal(kind, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    if kind == "tone":
        return 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 1200 * t)
    if kind == "high":
        return np.sin(2 * np.pi * 7000 * t)
    if kind == "low":
        return np.sin(2 * np.pi * 200 * t)
    if kind == "chirp":
        return 0.8 * np.sin(2 * np.pi * (200 + (7000 - 200) / 2 * t) * t)
    if kind == "noise":
        rng = np.random.default_rng(0)
        return 0.3 * rng.standard_normal(len(t))
    raise ValueError(kind)


# ---------------------------------------------------------------- 手工 mel 候选配方
def hz2mel(h):  return 2595.0 * np.log10(1.0 + h / 700.0)
def mel2hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(nf, nm, fmin, fmax):
    """librosa 同款三角滤组（hz 轴线性取点）。返回 [nm, nf//2+1]"""
    fftf = np.linspace(0, SR / 2, nf // 2 + 1)
    pts = mel2hz(hz2mel(fmin) + (hz2mel(fmax) - hz2mel(fmin)) * np.arange(nm + 2) / (nm + 1))
    out = np.zeros((nm, nf // 2 + 1))
    for i in range(nm):
        lo, ct, hi = pts[i], pts[i + 1], pts[i + 2]
        up = np.where(fftf <= ct, (fftf - lo) / max(ct - lo, 1e-9), 0.0)
        dn = np.where((fftf > ct) & (fftf <= hi), (hi - fftf) / max(hi - ct, 1e-9), 0.0)
        out[i] = up + dn
    return out


def manual_mel(sig, nf, nm, nwin, hop, fmax, fmin=0.0, window="hann",
               logfn="log1p", center=True, stride=1):
    if window == "hann":
        win = np.hanning(nwin)
    elif window == "rect":
        win = np.ones(nwin)
    # 逐帧 FFT（center=True → 前后补零到整帧数）
    n = len(sig)
    if center:
        nfr = int(np.ceil(n / hop)) + 1
        padded = np.pad(sig, (nwin // 2, nwin // 2))
        frames = np.lib.stride_tricks.sliding_window_view(padded, nwin)[::hop][:nfr]
    else:
        frames = np.stack([sig[i * hop:i * hop + nwin] for i in range((n - nwin) // hop + 1)])
    S = np.abs(np.fft.rfft(frames * win, n=nf)) ** 2
    fb = mel_filterbank(nf if nf >= 2 * (nwin) else nf, nm, fmin, fmax)
    m = S @ fb.T                                   # [T, nm]
    if logfn == "log1p":
        lm = np.log1p(m)
    elif logfn == "log+1e-6":
        lm = np.log(m + 1e-6)
    elif logfn == "log+eps":
        lm = np.log(m + np.finfo(np.float32).eps)
    elif logfn == "power2db":
        lm = 10 * np.log10(m + 1e-10)
    elif logfn == "log1p+zscore":
        lm = np.log1p(m)
        lm = (lm - lm.mean(0)) / (lm.std(0) + 1e-5)
    elif logfn == "log1p+zscore_db":
        lm = 10 * np.log10(m + 1e-10)
        lm = (lm - lm.mean(0)) / (lm.std(0) + 1e-5)
    if stride > 1:
        lm = lm[::stride]
    return lm


def align_t(cand, target):
    """返回对齐后 (cand, target) 与相关性（允许候选时间轴 slice 偏移）"""
    best, best_sl = -9, None
    tc = cand.shape[0]
    for start in range(0, 4):
        cc = cand[start:start + tc - start]
        ma = min(cc.shape[0], target.shape[0])
        x, y = cc[:ma].ravel().astype(np.float64), target[:ma].ravel().astype(np.float64)
        c = np.corrcoef(x, y)[0, 1]
        if c > best:
            best, best_sl = c, (start, ma)
    return best, best_sl


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    fex = probe_extractor()

    kinds = ["tone", "high", "low", "chirp", "noise"]
    golds = {}
    print("\n== 黑盒取样 ==")
    for k in kinds:
        sig = make_signal(k)
        inp = fex(sig, sampling_rate=SR, return_tensors="pt")
        f = inp["input_features"].numpy()          # [B, T, C]
        m = inp["attention_mask"].numpy()
        print(f"{k:7s}  features {f.shape}  mask {m.shape}  range [{f.min():.3f},{f.max():.3f}]")
        golds[k] = (sig.astype(np.float32), f[0].astype(np.float32), m[0].astype(np.int32))
    np.savez(os.path.join(GOLDEN_DIR, "w2vbert_golden.npz"),
             tone_sig=golds["tone"][0], tone_feat=golds["tone"][1], tone_mask=golds["tone"][2],
             high_sig=golds["high"][0], high_feat=golds["high"][1],
             low_sig=golds["low"][0], low_feat=golds["low"][1],
             chirp_sig=golds["chirp"][0], chirp_feat=golds["chirp"][1],
             noise_sig=golds["noise"][0], noise_feat=golds["noise"][1],
             sr=SR)
    print("golden saved ->", os.path.join(GOLDEN_DIR, "w2vbert_golden.npz"))

    # ---- 配方匹配：对 tone + chirp 两个信号求平均相关 ----
    C = fex(make_signal("tone"), sampling_rate=SR, return_tensors="pt")["input_features"][0].numpy()
    C2 = fex(make_signal("chirp"), sampling_rate=SR, return_tensors="pt")["input_features"][0].numpy()
    NM = C.shape[1]
    print(f"\n== 通道数 {NM}，开始配方匹配 ==")
    cands = []
    for nf in (400, 512, 800):
        for nm in (80, 160):
            if nm != NM and nm != NM // 2:
                continue
            for nwin in (400, 512, 800):
                if nwin > nf:
                    continue
                for hop in (160, 320):
                    for fmax in (8000.0, 7600.0, 0.0):
                        if fmax == 0.0:
                            fm = SR / 2
                        else:
                            fm = fmax
                        for logfn in ("log1p", "log+1e-6", "log1p+zscore", "log1p+zscore_db"):
                            for stride in (1, 2):
                                try:
                                    m1 = manual_mel(make_signal("tone"), nf, nm, nwin, hop, fm,
                                                    logfn=logfn, stride=stride)
                                except Exception:
                                    continue
                                if m1.shape[1] != NM:
                                    continue
                                # 两模型都对齐后取平均相关
                                t1 = C.shape[0]
                                a1 = m1.shape[0]
                                a2 = None
                                m2 = None
                                corr_pairs = []
                                for tgt in (C, C2):
                                    if a1 >= tgt.shape[0]:
                                        best, _ = align_t(m1, tgt)
                                    else:
                                        best, _ = -9, None
                                        for start in range(0, 4):
                                            tt = tgt[start:start + tgt.shape[0] - start]
                                            ma = min(a1, tt.shape[0])
                                            corr = np.corrcoef(m1[:ma].ravel(), tt[:ma].ravel())[0, 1]
                                            best = max(best, corr)
                                    corr_pairs.append(best)
                                c0 = min(corr_pairs)
                                cands.append((c0, nf, nm, nwin, hop, fm, logfn, stride))
    cands.sort(key=lambda x: -x[0])
    print("\n== TOP 候选配方 ==")
    for r in cands[:10]:
        print(f"corr={r[0]:.6f}  n_fft={r[1]}  n_mels={r[2]}  win={r[3]}  hop={r[4]}  fmax={r[5]}  log={r[6]}  stride={r[7]}")


if __name__ == "__main__":
    main()