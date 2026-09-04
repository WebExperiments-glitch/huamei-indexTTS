#!/usr/bin/env python3
"""prepare_voicepack.py —— A1 说话人条件束（voicepack）预计算（Mac 端运行）

把一段"干净人声参考音频"变成 iOS 端可直接使用的条件束，等价于官方
IndexTTS-2.5 推理里的"参考音频缓存段"（infer_v2_5.py 618-668 行），
产出 3 个 json：style（GPT 说话人条件）+ prompt（s2mel 扩散锚定）+ refmel。

    iOS 侧对接：
      style.json            → [192]           与 feat1[73,192] 一行同构（GPT 条件）
      prompt_<id>.json      → [P,512]         reshape [1,P,512] → PromptBundle.promptCondition
      refmel_<id>.json      → [P,80]          reshape [1,P,80] → transpose → [1,80,P] = refMel

计算链（逐行对应官方源码）：
    wav ─┬→ 22.05k ─ mel_spectrogram ──────────────→ ref_mel            (L640)
         ├→ 16k ─ SeamlessM4TFeatureExtractor(fbank80)
         │     └→ w2v-bert-2.0 hidden_states[17]                       (L631-639, L288)
         │          → (x - mean)/std  (wav2vec2bert_stats.pt)          (L289)
         │          → length_regulator(S_ref, ylens=mel_T) ──→ prompt  (L651-656)
         └→ 16k ─ kaldi fbank80 -mean ─ CAMPPlus ──────→ style         (L643-649)

运行环境（Mac）：
    cd 项目根
    pip install torch torchaudio librosa scipy transformers pyyaml
    PYTHONPATH=reference/index-tts-main  # 复用官方 mel_spectrogram / CAMPPlus（纯 torch）

必需权重（下载一次，默认放 ./models/hf_cache/）：
    w2v-bert-2.0        HF facebook/w2v-bert-2.0（或魔搭镜像；官方是 hf_cache 本地目录）
    campplus_cn_common.bin  官方 IndexTTS-2.5 模型包 aux（魔搭 IndexTeam/IndexTTS-2.5）
    wav2vec2bert_stats.pt    同上（本项目已在 models/mlx-indextts2-2.5-8bit/ 下）

用法：
    python scripts/prepare_voicepack.py --wav 参考.wav --id my_voice \
        --out 输出目录 \
        --model-dir models/mlx-indextts2-2.5-8bit \
        --hf-cache models/hf_cache

注意：Python 数值孪生（p0/）已在 Windows 全链对拍；本脚本的 LR 部分与 p0 同实现。
"""
import argparse, json, os, sys, wave
import numpy as np
import torch
import torch.nn.functional as F
try:
    import torchaudio            # Mac 运行必需；Windows dry-run 时缺失可跳过
except ImportError:
    torchaudio = None
import yaml

def err(msg): print(f"[prepare_voicepack] ❌ {msg}", file=sys.stderr); sys.exit(1)

# ---------------------------------------------------------------- safetensors 读取（F16 权重用）
def load_safetensors_f16(path, keys):
    """只读所需 key（safetensors: 8B header_len + json + data）。返回 {key: torch.tensor}。"""
    import struct
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k in keys:
            if k not in hdr: continue
            s0, e0 = hdr[k]["data_offsets"]
            f.seek(base + s0)
            raw = f.read(e0 - s0)
            if hdr[k]["dtype"] == "F16":
                out[k] = torch.frombuffer(raw, dtype=torch.float16).clone().float()
            else:
                out[k] = torch.frombuffer(raw, dtype=torch.float32).clone()
            out[k] = out[k].reshape(hdr[k]["shape"])
    return out

# ---------------------------------------------------------------- LengthRegulator（与 p0 同实现，权重来自 MLX s2mel）
class LengthRegulator:
    """s2mel LengthRegulator（is_discrete=False 分支）：
    content_in_proj(1024→512) → 最近邻插值到 ylens → 4×(Conv1d k3+GN(1)+Mish) → Conv1d k1
    官方源码 length_regulator.py InterpolateRegulator；权重 key 实测见日志。"""
    def __init__(self, w: dict):
        self.w = w
        pre = "length_regulator"
        # Linear: MLX [out,in]=[512,1024] == torch F.linear 布局，无需转置
        self.cp = (w[f"{pre}.content_in_proj.weight"], w[f"{pre}.content_in_proj.bias"])
        self.convs = []   # (weight[O,I,K] torch 布局, bias, is_gn, gn_w, gn_b)
        idxs = [0, 3, 6, 9]
        for i, ci in enumerate(idxs):
            cw = w[f"{pre}.model.{ci}.weight"].permute(0, 2, 1).contiguous()  # (O,K,I)->(O,I,K)
            cb = w[f"{pre}.model.{ci}.bias"]
            gw = w[f"{pre}.model.{ci+1}.weight"]
            gb = w[f"{pre}.model.{ci+1}.bias"]
            self.convs.append((cw, cb, gw, gb))
        cw = w[f"{pre}.model.12.weight"].permute(0, 2, 1).contiguous()        # k1
        self.final = (cw, w[f"{pre}.model.12.bias"])

    def __call__(self, x, ylens):
        # x [B,T,1024] → [B,T,512]
        h = F.linear(x, self.cp[0], self.cp[1])
        h = h.transpose(1, 2)                                   # [B,512,T]
        h = F.interpolate(h, size=int(ylens), mode="nearest")   # → [B,512,ylens]
        for (cw, cb, gw, gb) in self.convs:
            h = F.conv1d(h, cw, cb, padding=1)
            h = F.group_norm(h, 1, gw, gb)
            h = F.mish(h)
        h = F.conv1d(h, self.final[0], self.final[1])
        return h.transpose(1, 2)                                # [B,ylens,512]

# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="参考音频（干净人声，建议 5-15s）")
    ap.add_argument("--id", required=True, help="输出 id（如 my_voice / zh_01），作为文件名后缀")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--model-dir", default=r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit")
    ap.add_argument("--hf-cache", default="hf_cache", help="w2v-bert-2.0 / campplus 权重目录")
    ap.add_argument("--max-sec", type=float, default=15.0)
    a = ap.parse_args()
    if torchaudio is None:
        err("缺 torchaudio：Mac 上执行 `pip install torch torchaudio librosa scipy transformers pyyaml`")

    os.makedirs(a.out, exist_ok=True)
    dev = "cpu"

    # ---- 0) 读参考音频并重采样（官方 _load_and_cut_audio，≤15s）----
    wav, sr = torchaudio.load(a.wav)
    if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)          # 混单声道
    max_n = int(a.max_sec * sr)
    wav = wav[:, :max_n]
    print(f"[0] wav {tuple(wav.shape)} sr={sr}")
    a16 = torchaudio.functional.resample(wav, sr, 16000)
    a22 = torchaudio.functional.resample(wav, sr, 22050)

    # ---- 1) ref_mel：22.05k mel（官方 L640 + audio.py mel_spectrogram）----
    cfg = yaml.safe_load(open(os.path.join(a.model_dir, "config.yaml"), encoding="utf-8"))
    sp = cfg["s2mel"]["preprocess_params"]["spect_params"]
    sys.path.insert(0, os.path.abspath("reference/index-tts-main"))
    from indextts.s2mel.modules.audio import mel_spectrogram
    fmax = None if str(sp.get("fmax", "None")) == "None" else float(sp["fmax"])
    mel_args = dict(n_fft=sp["n_fft"], win_size=sp["win_length"], hop_size=sp["hop_length"],
                    num_mels=sp["n_mels"], sampling_rate=sp["sr"],
                    fmin=float(sp.get("fmin", 0)), fmax=fmax, center=False)
    ref_mel = mel_spectrogram(a22.float(), **mel_args)          # [1,80,T]
    P = ref_mel.size(2)
    print(f"[1] ref_mel {tuple(ref_mel.shape)}  (P={P})")

    # ---- 2) prompt：w2v-bert 第 17 层 → std → LR(spk_cond_emb, ylens=P) ----
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    w2v_dir = a.hf_cache
    if not os.path.isdir(os.path.join(w2v_dir, "w2v-bert-2.0")):
        # 兜底：直接用 HF 仓库名（联网拉取）
        w2v_dir = "facebook/w2v-bert-2.0"
    fex = SeamlessM4TFeatureExtractor.from_pretrained(w2v_dir, local_files_only=os.path.isdir(a.hf_cache))
    model = Wav2Vec2BertModel.from_pretrained(w2v_dir, local_files_only=os.path.isdir(a.hf_cache))
    model.eval().to(dev)
    inputs = fex(a16.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = model(input_features=inputs["input_features"].to(dev),
                    attention_mask=inputs["attention_mask"].to(dev),
                    output_hidden_states=True)
    S_ref = out.hidden_states[17]                               # [1,T16,1024]（官方 L288）
    stats = torch.load(os.path.join(a.model_dir, "wav2vec2bert_stats.pt"),
                       map_location="cpu", weights_only=False)
    S_ref = (S_ref - stats["mean"]) / torch.sqrt(stats["var"])  # 官方 L289
    print(f"[2] w2v-bert layer17 {tuple(S_ref.shape)}")

    lr_w = load_safetensors_f16(os.path.join(a.model_dir, "s2mel.safetensors"),
        [k for k in ("length_regulator.content_in_proj.weight",
                     "length_regulator.content_in_proj.bias")] +
        [f"length_regulator.model.{i}.{s}" for i in (0,1,3,4,6,7,9,10,12) for s in ("weight","bias")])
    lr = LengthRegulator(lr_w)
    with torch.no_grad():
        prompt_cond = lr(S_ref, ylens=P)[0]                     # [P,512]
    print(f"[3] prompt_condition {tuple(prompt_cond.shape)}")

    # ---- 3) style：16k kaldi fbank80 → CAMPPlus（官方 L643-649 + campplus/DTDNN.py）----
    fbank = torchaudio.compliance.kaldi.fbank(
        a16.to(dev), num_mel_bins=80, dither=0, sample_frequency=16000)   # [T16,80]
    feat = fbank - fbank.mean(0, keepdim=True)
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    ckpt = os.path.join(a.hf_cache, "campplus_cn_common.bin")
    if not os.path.isfile(ckpt):
        err("缺 campplus_cn_common.bin：魔搭 IndexTeam/IndexTTS-2.5 模型包 aux 内，放入 --hf-cache")
    camp = CAMPPlus(feat_dim=80, embedding_size=192)
    camp.load_state_dict(torch.load(ckpt, map_location="cpu"))
    camp.eval().to(dev)
    with torch.no_grad():
        style = camp(feat.unsqueeze(0))                         # [1,192]
    print(f"[4] style {tuple(style.shape)}")

    # ---- 4) 落盘（iOS 读取格式见文件头 docstring）----
    def dumps(name, arr2d):
        p = os.path.join(a.out, name)
        json.dump(arr2d.tolist(), open(p, "w"), ensure_ascii=False)
        print(f"    {name}  {tuple(arr2d.shape)}  {os.path.getsize(p)/1024:.0f}KB")
    dumps(f"prompt_{a.id}.json", prompt_cond)                   # [P,512]
    dumps(f"refmel_{a.id}.json", ref_mel.squeeze(0).transpose(0, 1))  # [P,80]
    dumps(f"style_{a.id}.json", style.squeeze(0))               # [192]
    meta = {"id": a.id, "source": os.path.basename(a.wav), "mel_frames": int(P),
            "prompt": f"prompt_{a.id}.json", "refmel": f"refmel_{a.id}.json",
            "style": f"style_{a.id}.json", "note": "GPT 用 style[192]；s2mel 用 prompt/refmel（PromptBundle）"}
    json.dump(meta, open(os.path.join(a.out, f"meta_{a.id}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n✅ voicepack '{a.id}' 完成 → {a.out}/")
    print("   iOS 放置：拷 3 个 json 到 Documents/huamei-models/，按 id 挂载即可")

if __name__ == "__main__":
    main()
