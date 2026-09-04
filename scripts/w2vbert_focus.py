#!/usr/bin/env python3
"""焦点评测：锁定 w2v-bert 特征 160 通道的真实生成方式（最小化、直接对比）"""
import os
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import scipy.signal as sg

SR = 16000
d = r"d:\DeepSeek前端代码\indexTTS 2.5\models\hf_cache\w2v-bert-2.0"
npz = r"d:\DeepSeek前端代码\indexTTS 2.5\scripts\golden\w2vbert_golden.npz"

fex = SeamlessM4TFeatureExtractor.from_pretrained(d, local_files_only=True)
z = np.load(npz)
f = z["chirp_feat"]  # [49,160]

def fb(nf, nm, fmin, fmax):
    fftf = np.linspace(0, SR/2, nf//2+1)
    h2m = lambda h: 2595*np.log10(1+h/700)
    m2h = lambda m: 700*(10**(m/2595)-1)
    pts = m2h(h2m(fmin)+(h2m(fmax)-h2m(fmin))*np.arange(nm+2)/(nm+1))
    F = np.zeros((nm, nf//2+1))
    for i in range(nm):
        lo, ct, hi = pts[i], pts[i+1], pts[i+2]
        up = np.where(fftf<=ct, (fftf-lo)/max(ct-lo,1e-9), 0.0)
        dn = np.where((fftf>ct)&(fftf<=hi), (hi-fftf)/max(hi-ct,1e-9), 0.0)
        F[i] = up+dn
    return F

sig = z["chirp_sig"]
def mel_of(nf, nwin, hop, nm, fmax, center, window):
    win = np.hanning(nwin) if window=="hann" else np.ones(nwin)
    n = len(sig)
    # 与 librosa stft(center=True, pad_mode=reflect) 对齐！librosa stft center 用 reflect padding
    if center:
        pad = nwin//2
        sp = np.pad(sig, (pad, pad), mode="reflect")
        nfr = int(np.ceil(n/hop))+1
    else:
        sp = sig
        nfr = (n-nwin)//hop+1
    # 手动滑窗
    cols = []
    segs = sp
    Np = len(sp)
    starts = np.arange(nfr)*hop
    frames = np.stack([sp[s:s+nwin] for s in starts if s+nwin<=Np])
    if frames.shape[0] < nfr:
        frames2 = np.zeros((nfr, nwin))
        frames2[:frames.shape[0]] = frames
        frames = frames2
    S = np.abs(np.fft.rfft(frames*win, n=nf))**2
    Fb = fb(nf, nm, 0.0, fmax)
    return S@Fb.T

def corr(a, b):
    a=a.ravel().astype(float); b=b.ravel().astype(float)
    return np.corrcoef(a,b)[0,1]

# 尝试一批 n_fft/win/hop/nm/fmax/center/window
for nf,nwin,hop,nm,fmax,center,winname in [
    (400,400,160,80,8000,True,"hann"),
    (400,400,160,80,7600,True,"hann"),
    (512,512,160,80,8000,True,"hann"),
    (800,800,160,80,8000,True,"hann"),
    (400,400,160,160,8000,True,"hann"),
    (400,400,320,80,8000,True,"hann"),
    (400,400,160,80,8000,False,"hann"),
]:
    try:
        m = mel_of(nf,nwin,hop,nm,fmax,center,winname)
    except Exception as e:
        print("fail", (nf,nwin,hop,nm,fmax,center,winname), e); continue
    logm = np.log1p(m)
    logm_z = (logm-logm.mean(0))/(logm.std(0)+1e-5)
    print(f"... nf={nf} win={nwin} hop={hop} nm={nm} fmax={fmax} center={center} shape={logm.shape}")
    if logm_z.shape[1]==160:
        print("   corr L", round(corr(logm_z[:,:80], f[:,:80]),4), " R", round(corr(logm_z[:,80:], f[:,80:]),4))
    elif logm_z.shape[1]==80:
        # 时间轴可能 2x，试偶数/奇数帧 + 偏移
        for sl,lab in [(slice(None,None,2),'even'),(slice(1,None,2),'odd')]:
            mm=logm_z[sl]; mmz=(mm-mm.mean(0))/(mm.std(0)+1e-5)
            ma=min(mmz.shape[0], f.shape[0])
            cL=corr(mmz[:ma], f[:ma,:80]); cR=corr(mmz[:ma], f[:ma,80:])
            print(f"   {lab} ma={ma} corr L", round(cL,4), " R", round(cR,4))