"""GPT 自回归生成（P1c）—— KV cache 版，自研

流程（对齐官方 inference_speech 理解）：
  1. 条件：conds = [spk_proj(192→1280) + emo_vec(1280)] ∥ 2×零 → 3 个 token
  2. 文本：clean → [start]<text>[stop]，text_emb = wte + text_pos[:L] + lang_emb
  3. 前缀 emb = conds ∥ text_emb（transformer 内部 wpe=null，位置全由外部表管）
  4. 首 token：start_mel(8192) 位置 0 → 前缀+start 因果前向（记 KV）
  5. 循环：单 token（mel 位置 = 已生成序号）KV 增量 → mel_head logits → 采样
  6. 遇 stop_mel_token(8193) 或超长停止
"""
import torch
import torch.nn.functional as F

from gpt_core import (D_MODEL, GPT_LAYERS, N_HEADS, HEAD_DIM, gelu_new,
                      causal_attn)

START_TEXT, STOP_TEXT = 0, 1
START_MEL, STOP_MEL = 8192, 8193
TOP_K, TOP_P, TEMP = 30, 0.8, 0.8
MAX_MEL = 200


def sample(logits: torch.Tensor, top_k=TOP_K, top_p=TOP_P, temperature=TEMP,
           rng: torch.Generator | None = None) -> int:
    """单 token 采样：top_k → top_p → 温度 → categorical。logits [V]"""
    logits = logits.float() / temperature
    v = logits.shape[-1]
    k = min(top_k, v)
    kth = torch.topk(logits, k).values[..., -1].unsqueeze(-1)
    logits = torch.where(logits < kth, float("-inf"), logits)
    sorted_l, idx = logits.sort(descending=True)
    cum = sorted_l.softmax(-1).cumsum(-1)
    rm = cum - sorted_l.softmax(-1) > top_p
    sorted_l[rm] = float("-inf")
    logits = torch.zeros_like(logits).scatter(-1, idx, sorted_l)
    p = logits.softmax(-1)
    return int(torch.multinomial(p, 1, generator=rng).item())


class KVCache:
    def __init__(self, w: dict, device="cpu"):
        self.w = w
        # k/v: [B, H, T, Dh]，逐层
        self.k: list[torch.Tensor | None] = [None] * GPT_LAYERS
        self.v: list[torch.Tensor | None] = [None] * GPT_LAYERS

    def reset(self):
        self.k = [None] * GPT_LAYERS
        self.v = [None] * GPT_LAYERS

    def _attn_qkv(self, x: torch.Tensor, layer: int):
        """q/k/v 计算。x [B,1,D] 或全序列；返回 heads 版 q/k/v [B,H,T,Dh]"""
        w = self.w
        p = f"gpt.h.{layer}"
        qkv = F.linear(x, w[f"{p}.attn.c_attn.weight"], w[f"{p}.attn.c_attn.bias"])
        b, t, _ = qkv.shape
        qkv = qkv.view(b, t, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _ln(self, x, layer, which):
        w = self.w
        return F.layer_norm(x, (D_MODEL,), w[f"gpt.h.{layer}.ln_{which}.weight"],
                            w[f"gpt.h.{layer}.ln_{which}.bias"], 1e-5)

    def _mlp(self, x, layer):
        w = self.w
        p = f"gpt.h.{layer}.mlp"
        h = gelu_new(F.linear(x, w[f"{p}.c_fc.weight"], w[f"{p}.c_fc.bias"]))
        return F.linear(h, w[f"{p}.c_proj.weight"], w[f"{p}.c_proj.bias"])

    def _proj(self, x, layer):
        w = self.w
        p = f"gpt.h.{layer}.attn.c_proj"
        return F.linear(x, w[f"{p}.weight"], w[f"{p}.bias"])

    def _final_ln(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        return F.layer_norm(x, (D_MODEL,), w["gpt.ln_f.weight"], w["gpt.ln_f.bias"], 1e-5)

    def full_forward(self, emb: torch.Tensor):
        """首调用：完整序列（前缀+首个 mel），因果掩码，返回 ln_f 后 hidden。"""
        x = emb
        for i in range(GPT_LAYERS):
            nx = self._ln(x, i, 1)
            q, k, v = self._attn_qkv(nx, i)
            a = causal_attn(q, k, v, HEAD_DIM ** -0.5)
            b, t, _ = nx.shape
            a = a.transpose(1, 2).reshape(b, t, D_MODEL)
            x = x + self._proj(a, i)
            nx = self._ln(x, i, 2)
            x = x + self._mlp(nx, i)
            # 缓存 k/v（全量）
            self.k[i] = k
            self.v[i] = v
        return self._final_ln(x)   # [B, T, D]

    def step(self, x: torch.Tensor):
        """KV 增量步：x [B,1,D]，返回 ln_f 后 hidden [B,1,D]。"""
        for i in range(GPT_LAYERS):
            nx = self._ln(x, i, 1)
            q, k, v = self._attn_qkv(nx, i)
            kk = k if self.k[i] is None else torch.cat([self.k[i], k], 2)
            vv = v if self.v[i] is None else torch.cat([self.v[i], v], 2)
            self.k[i], self.v[i] = kk, vv
            attn = (q @ kk.transpose(-2, -1) * (HEAD_DIM ** -0.5)).softmax(-1)
            a = (attn @ vv).transpose(1, 2).reshape(1, 1, D_MODEL)
            x = x + self._proj(a, i)
            nx = self._ln(x, i, 2)
            x = x + self._mlp(nx, i)
        return self._final_ln(x)


class GPTGenerator:
    def __init__(self, w: dict, device="cpu"):
        self.w = w
        self.kvc = KVCache(w, device)
        self.device = device

    # ---- 前缀构造（对齐官方 prepare_gpt_inputs 的语义）----
    def prepare_prefix(self, text_ids: list[int], spk_vec: torch.Tensor,
                       emo_vec: torch.Tensor, lang_id: int = 1) -> torch.Tensor:
        w = self.w
        # 1) 文本 ids：滤掉 start/stop → [start] text [stop]
        clean = [i for i in text_ids if i not in (START_TEXT, STOP_TEXT)]
        seq = [START_TEXT] + clean + [STOP_TEXT]
        ids = torch.tensor(seq, dtype=torch.long)
        L = len(seq)
        text_emb = w["text_embedding.weight"][ids] \
            + w["text_pos_embedding.emb.weight"][:L] \
            + w["lang_embedding.weight"][lang_id]
        # 2) 条件：spk_proj(192→1280) + emo_vec[1280]
        spk = F.linear(spk_vec.view(1, -1).float(), w["spk_emb_proj.weight"],
                       w["spk_emb_proj.bias"])                     # [1,1280]
        cond0 = spk + emo_vec.view(1, -1).float()                   # [1,1280]
        zeros = torch.zeros(2, D_MODEL)
        conds = torch.cat([cond0, zeros], 0)                        # [3,D]
        return torch.cat([conds, text_emb], 0)                      # [3+L, D]

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden 末 token → mel logits [8194]"""
        w = self.w
        x = F.layer_norm(hidden, (D_MODEL,), w["final_norm.weight"],
                         w["final_norm.bias"], 1e-5)
        return F.linear(x, w["mel_head.weight"], w["mel_head.bias"]).squeeze(0)

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, seed: int | None = None,
                 max_tokens: int = MAX_MEL, verbose: bool = False) -> list[int]:
        """prefix [3+L,D]（text 侧；无 mel 位置）→ codes 列表。"""
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        self.kvc.reset()
        mel_pos = self.w["mel_pos_embedding.emb.weight"]
        # 首 token = start_mel，位置 0
        start = self.w["mel_embedding.weight"][START_MEL] + mel_pos[0]
        seq = torch.cat([prefix, start.unsqueeze(0)])               # [P+1,D]
        hidden = self.kvc.full_forward(seq.unsqueeze(0))            # [1,P+1,D]
        codes: list[int] = []
        for j in range(1, max_tokens + 1):
            logits = self.head(hidden[:, -1]).reshape(-1)
            tok = sample(logits, rng=rng)
            if verbose:
                print(f"  step{j}: token={tok} logits_std={logits.std():.2f} "
                      f"top={logits.topk(3).values.tolist()}")
            if tok == STOP_MEL:
                break
            codes.append(tok)
            # 下一 mel token：位置 = j（start 是 0，这里首个新 token 是位置 1）
            emb = self.w["mel_embedding.weight"][tok] + mel_pos[j]
            hidden = self.kvc.step(emb.unsqueeze(0).unsqueeze(0))
        return codes
