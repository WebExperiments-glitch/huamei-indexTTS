#!/usr/bin/env python3
"""w2vbert_export.py —— 导出 w2v-bert-2.0 权重清单 + 转 campplus_cn_common.bin → safetensors

产出：
  scripts/golden/w2vbert_keys.json        全部 tensor 名 + shape（Swift 端按图索骥）
  models/hf_cache/campplus_cn_common.safetensors   26.7MB torch 权重转标准格式
"""
import json, os, struct
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2V = os.path.join(ROOT, "models", "hf_cache", "w2v-bert-2.0", "model.safetensors")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
CAMP_BIN = os.path.join(ROOT, "models", "hf_cache", "campplus_cn_common.bin")

os.makedirs(GOLDEN_DIR, exist_ok=True)

# 1) w2v-bert 权重清单
with open(W2V, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
keys = sorted(k for k in hdr if k != "__metadata__")
summary = {}
for k in keys:
    head = k.split(".")
    grp = ".".join(head[:2])
    summary.setdefault(grp, []).append(k)
out = {k: (hdr[k].get("shape"), hdr[k].get("dtype")) for k in keys}
with open(os.path.join(GOLDEN_DIR, "w2vbert_keys.json"), "w") as f:
    json.dump(out, f, indent=0)
print("w2v-bert keys:", len(keys))
for grp, ks in sorted(summary.items()):
    print(f"  {grp:42s} {len(ks):3d} 例: {ks[0]}")

# 3) wav2vec2bert_stats.pt → JSON（layer17 特征归一化 mean/var）
STATS_PT = os.path.join(ROOT, "models", "mlx-indextts2-2.5-8bit", "wav2vec2bert_stats.pt")
if os.path.isfile(STATS_PT):
    st = torch.load(STATS_PT, map_location="cpu", weights_only=False)
    json.dump({k: v.tolist() for k, v in st.items()},
              open(os.path.join(GOLDEN_DIR, "wav2vec2bert_stats.json"), "w"))
    print("\nwav2vec2bert_stats exported:", {k: list(v.shape) for k, v in st.items()})

# 2) campplus 转换（torch state_dict → safetensors）
if os.path.isfile(CAMP_BIN):
    sd = torch.load(CAMP_BIN, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise RuntimeError(f"campplus 不是 state_dict: {type(sd)}")
    from safetensors.torch import save_file
    dst = os.path.join(ROOT, "models", "hf_cache", "campplus_cn_common.safetensors")
    save_file({k: v.contiguous().float() for k, v in sd.items() if isinstance(v, torch.Tensor)}, dst)
    print("\ncampplus keys:", len(sd))
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:40s} {tuple(v.shape)}")
    print("saved ->", dst)
else:
    print("campplus bin missing:", CAMP_BIN)