"""IndexTTS-2.5 端到端推理管线（P3b）—— 全部自研模块组装

数据流（对齐官方 infer_v2_5.infer 主循环）：
    text → tiktoken(60,510) → [start]ids[stop]
         → GPT(8bit 反量化, 24 层, KV cache) → mel codes [T]        @ 25 fps
         → Codec.decode → S_infer [2T, 1024]                        @ 50 fps
         → LengthRegulator(×1.72) → cond [T', 512]                  @ 86 fps
         → cat([prompt_condition, cond]) → CFM(25 步欧拉 + CFG 0.7) → mel [80, T']
         → BigVGAN → wav                                            @ 22050 Hz

说话人条件（SpkCondition）官方来自 w2v-bert + campplus。本模块支持两种来源：
  - from_preset()  : style 取 spk_matrix 的某一行（真实 campplus 向量，192 维）
  - from_npz()     : Mac 端预计算后导入（A1 方案，iOS 运行时走的路径）
prompt_condition / ref_mel 在无 w2v-bert 环境下用占位张量（管线集成验证用）。
"""
from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tiktoken_bpe import Tokenizer
from gpt_core import load_gpt_weights, D_MODEL
from gpt_gen import GPTGenerator
from codec_impl import SemanticCodec, load_codec_weights
from s2mel_impl import CFM, LengthRegulator, load_s2mel_weights
from safetensors_loader import load_tensors

MODELS = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit"
EMO_NUM = [3, 17, 2, 8, 4, 5, 10, 24]          # 8 类情绪的样本数（config.emo_num）
EMO_NAMES = ["happy", "angry", "sad", "afraid",
             "disgusted", "melancholic", "surprised", "calm"]
SR = 22050


def _language_dict(reference_root: str) -> dict:
    """从 reference 源码用 AST 提取 LANGUAGES，构造 lang → lang_embedding 索引。"""
    p = os.path.join(reference_root, "indextts/utils/tokenizer.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "LANGUAGES":
                    langs = list(ast.literal_eval(node.value).keys())
                    return {lang: i for i, lang in enumerate(langs)}
    raise RuntimeError("未找到 LANGUAGES")


# ---------------- 说话人条件 ----------------
@dataclass
class SpkCondition:
    """一次参考音频预计算得到的条件束（可缓存复用，iOS 端直接读 npz）。"""
    style: torch.Tensor                 # [1,192]  campplus 全局音色向量
    prompt_condition: torch.Tensor      # [1,P,512]  s2mel 参考声学条件
    ref_mel: torch.Tensor               # [1,80,P]   参考 mel（CFM prompt 锚）
    name: str = "preset"

    @property
    def prompt_len(self) -> int:
        return int(self.ref_mel.shape[-1])

    def save(self, path: str) -> None:
        torch.save({k: v.cpu() for k, v in
                    (("style", self.style), ("prompt_condition", self.prompt_condition),
                     ("ref_mel", self.ref_mel),
                     ("name", torch.tensor(0)))}, path)
        print(f"SpkCondition 已保存 → {path}")


class EmotionBank:
    """73 组预设的说话人/情绪矩阵（feat1.pt [73,192]、feat2.pt [73,1280]）。

    官方公式（infer_v2_5.py:669-680）：
        idx_i   = argmax cosine(style, spk_matrix_group_i)
        mat     = cat([emo_matrix_group_i[idx_i]])            # [8,1280]
        emovec  = sum_k w_k · mat_k                            # [1280]
        emovec  = emovec + (1 - Σw) · emovec_w2v               # Σw=1 时 w2v 项归零
    """

    def __init__(self, model_dir: str = MODELS):
        self.spk = torch.load(os.path.join(model_dir, "feat1.pt"), weights_only=True).float()
        self.emo = torch.load(os.path.join(model_dir, "feat2.pt"), weights_only=True).float()
        assert self.spk.shape[0] == sum(EMO_NUM) == self.emo.shape[0], \
            f"矩阵行数 {self.spk.shape[0]}/{self.emo.shape[0]} ≠ Σemo_num {sum(EMO_NUM)}"
        self.spk_groups = list(torch.split(self.spk, EMO_NUM))
        self.emo_groups = list(torch.split(self.emo, EMO_NUM))

    @torch.no_grad()
    def emovec(self, style: torch.Tensor, weight: list[float]) -> torch.Tensor:
        """style [1,192] + 8 维权向量 → emovec [1280]"""
        idx = [int(torch.argmax(F.cosine_similarity(style.float(), g, dim=1)))
               for g in self.spk_groups]
        mat = torch.cat([g[i].unsqueeze(0) for i, g in zip(idx, self.emo_groups)], 0)
        w = torch.tensor(weight, dtype=torch.float32).unsqueeze(1)
        return (w * mat).sum(0)                      # [1280]

    def preset_style(self, row: int = 0) -> torch.Tensor:
        """取一行真实 campplus 向量当作音色（192 维）。"""
        return self.spk[row].unsqueeze(0)            # [1,192]


# ---------------- 主管线 ----------------
class IndexTTS25:
    def __init__(self, model_dir: str = MODELS,
                 reference_root: str = r"D:/indexTTS 2.5/reference/index-tts-main",
                 verbose: bool = True):
        self.dir = model_dir
        self.verbose = verbose
        t0 = time.time()

        def log(msg, t):
            if verbose:
                print(f"  {msg:<28} {time.time() - t:6.1f}s")

        t = time.time()
        self.tok = Tokenizer(os.path.join(model_dir, "multilingual_zh_ja_yue_char_del.tiktoken"),
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "specials.json"))
        self.lang_dict = _language_dict(reference_root)
        log("tokenizer + 语言表", t)

        t = time.time()
        self.w_gpt = load_gpt_weights(os.path.join(model_dir, "gpt.safetensors"))
        self.gpt = GPTGenerator(self.w_gpt)
        log(f"GPT 权重({len(self.w_gpt)})", t)

        t = time.time()
        self.codec = SemanticCodec(load_codec_weights(os.path.join(model_dir, "codec.safetensors")))
        log("Codec 权重", t)

        t = time.time()
        self.w_s2mel = load_s2mel_weights(os.path.join(model_dir, "s2mel.safetensors"))
        self.lr = LengthRegulator(self.w_s2mel)
        self.cfm = CFM(self.w_s2mel)
        log(f"s2mel 权重({len(self.w_s2mel)})", t)

        t = time.time()
        from bigvgan_torch import BigVGAN
        self.w_bv = load_tensors(os.path.join(model_dir, "bigvgan.safetensors"), fp16_to_fp32=True)
        self.bigvgan = BigVGAN(self.w_bv).eval()
        log(f"BigVGAN 权重({len(self.w_bv)})", t)

        self.emobank = EmotionBank(model_dir)
        if verbose:
            print(f"  合计加载 {time.time() - t0:.1f}s")

    # ---- 条件束构造 ----
    def make_preset_condition(self, row: int = 0, prompt_len: int = 86) -> SpkCondition:
        """无 w2v-bert 时的占位条件束：style 用真实预设向量，prompt/ref_mel 用占位。
        （数值上等价于"未提供参考音频"，仅用于管线集成验证）"""
        style = self.emobank.preset_style(row)
        return SpkCondition(
            style=style,
            prompt_condition=torch.zeros(1, prompt_len, 512),
            ref_mel=torch.zeros(1, 80, prompt_len),
            name=f"preset#{row}",
        )

    @torch.no_grad()
    def synth(self, text: str, spk: SpkCondition, lang: str = "zh",
              emo_vector: list[float] | None = None, seed: int | None = 42,
              max_mel_tokens: int = 400, diffusion_steps: int = 25,
              duration_factor: float = 1.0) -> tuple[torch.Tensor, dict]:
        """返回 (wav [1,1,N] @22050Hz, 统计信息)"""
        stats: dict = {}
        # 1) 文本 → ids（官方：lang 前缀 + 小写 + 尾部补 stop）
        lang_key = lang.lower()
        ids = self.tok.encode(f"<|{lang_key}|> " + text.lower()) + [1]
        lang_id = self.lang_dict.get(lang_key, self.lang_dict.get("common", 0))
        stats["text_tokens"] = len(ids)

        # 2) 情绪向量（Σw=1 时官方的 w2v-bert 项被完全抵消）
        w = emo_vector if emo_vector is not None else [1.0] + [0.0] * 7
        emo_vec = self.emobank.emovec(spk.style, w)
        stats["emo_sum"] = float(sum(w))

        # 3) GPT 自回归 → mel codes
        t = time.time()
        prefix = self.gpt.prepare_prefix(ids, spk.style, emo_vec, lang_id=lang_id)
        codes = self.gpt.generate(prefix, seed=seed, max_tokens=max_mel_tokens)
        stats["mel_codes"] = len(codes)
        stats["gpt_sec"] = time.time() - t
        if not codes:
            raise RuntimeError("GPT 未生成任何 mel code")
        stats["audio_sec_est"] = len(codes) / 25.0

        # 4) Codec：25fps → 50fps
        t = time.time()
        S = self.codec.decode(torch.tensor([codes], dtype=torch.long))     # [1,2T,1024]
        stats["codec_sec"] = time.time() - t

        # 5) LengthRegulator：50fps → 86fps（×1.72）
        t = time.time()
        target_len = int(S.shape[1] * 1.72 * duration_factor)
        cond = self.lr(S, torch.tensor([target_len]))                       # [1,T',512]
        cat_cond = torch.cat([spk.prompt_condition, cond], dim=1)           # [1,P+T',512]
        stats["lr_sec"] = time.time() - t

        # 6) CFM 扩散 → mel
        t = time.time()
        mel = self.cfm.inference(cat_cond, cat_cond.shape[1], spk.ref_mel,
                                 spk.style, n_steps=diffusion_steps, cfg_rate=0.7)
        mel = mel[:, :, spk.prompt_len:]                                    # 裁掉参考段
        stats["cfm_sec"] = time.time() - t
        stats["mel_frames"] = int(mel.shape[-1])

        # 7) BigVGAN → wav
        t = time.time()
        wav = self.bigvgan(mel.float())
        stats["vocoder_sec"] = time.time() - t
        stats["wav_sec"] = wav.shape[-1] / SR
        return wav, stats
