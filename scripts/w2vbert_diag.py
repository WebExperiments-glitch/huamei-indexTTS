#!/usr/bin/env python3
"""聚焦诊断：官方 160 通道特征的内部统计特征"""
import numpy as np

npz = r"d:\DeepSeek前端代码\indexTTS 2.5\scripts\golden\w2vbert_golden.npz"
z = np.load(npz)
f = z["chirp_feat"].astype(np.float64)   # [49,160]

L, R = f[:, :80], f[:, 80:]
print("整体 shape", f.shape)
print("corr(左半,右半) 逐行均值:", np.mean([np.corrcoef(L[i], R[i])[0,1] for i in range(L.shape[0])] if L.shape[0]==R.shape[0] else [0]))
print("corr(左半.ravel, 右半.ravel):", round(float(np.corrcoef(L.ravel(), R.ravel())[0,1]),4))
print("左半 逐通道 mean 范围: [%.4f, %.4f]" % (L.mean(0).min(), L.mean(0).max()))
print("左半 逐通道 std  范围: [%.4f, %.4f]" % (L.std(0).min(), L.std(0).max()))
print("右半 逐通道 mean 范围: [%.4f, %.4f]" % (R.mean(0).min(), R.mean(0).max()))
print("右半 逐通道 std  范围: [%.4f, %.4f]" % (R.std(0).min(), R.std(0).max()))
# 逐帧能量与通道排序
row = f[10]
print("第10帧 左半峰值 idx:", np.argmax(L[10]), "值", L[10].max(), " 右半峰值 idx:", np.argmax(R[10]), "值", R[10].max())
print("第10帧 左半 [0:10]", L[10,:10].round(2).tolist())
print("第10帧 右半 [0:10]", R[10,:10].round(2).tolist())
# 峰值通道随帧移动是否同步（左右半同一物理通道？）
import numpy as np
lpk = np.argmax(L, axis=1); rpk = np.argmax(R, axis=1)
print("左右半峰值通道 变换差异 (max|lpk-rpk|):", int(np.abs(lpk-rpk).max()))