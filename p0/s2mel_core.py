"""s2mel DiT 的 Transformer 核心 —— 自研（P2）

结构（依据 gpt_fast/model.py 理解 + s2mel.safetensors 实测）：
  - Llama 风格：RMSNorm + RoPE + SwiGLU
  - adaLN 调制：AdaptiveLayerNorm = project_layer(embedding)→2D 拆 shift/scale，
    对 RMSNorm 输出做 scale*x+bias（embedding = t_embedder 的时间向量）
  - U-ViT 长跳跃：前 6 层 emit、后 6 层 receive（skip_in_linear 合并）
  - 非因果（mel 全序列双向可见）
"""
import math

import torch
import torch.nn.functional as F

DIM = 512          # DiT hidden
N_LAYER = 13
N_HEAD = 8
HEAD_DIM = 64
FFN = 1536         # w1/w3 输出（权重实测 [1536,512]）
BASE = 10000.0


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def adaln(norm_out: torch.Tensor, embed: torch.Tensor, project_w: torch.Tensor,
          project_b: torch.Tensor) -> torch.Tensor:
    """AdaptiveLayerNorm.forward：weight * norm(x) + bias（embed 过 project_layer）。

    ⚠️ 官方 split 顺序：前半 = weight(scale, 直接乘), 后半 = bias(shift)。
    与 FinalLayer 的 modulate(x*(1+scale)+shift) 约定相反，勿混用。"""
    scale, shift = F.linear(embed, project_w, project_b).chunk(2, dim=-1)
    return scale.unsqueeze(1) * norm_out + shift.unsqueeze(1)


def precompute_freqs(seq_len: int, head_dim: int = HEAD_DIM, base: float = BASE):
    """RoPE 频率 [T, head_dim/2, 2]（cos/sin 表，与官方 polar 等价）。"""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2)[:half].float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    ang = torch.outer(t, freqs)                     # [T, half]
    return torch.stack([ang.cos(), ang.sin()], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x [B,T,H,D] → 旋转位置编码。freqs [T, half, 2]。"""
    bsz, seqlen, h, d = x.shape
    half = d // 2
    xs = x.float().view(bsz, seqlen, h, half, 2)
    fc = freqs.view(1, seqlen, 1, half, 2)          # [., cos, sin]
    out = torch.stack([
        xs[..., 0] * fc[..., 0] - xs[..., 1] * fc[..., 1],
        xs[..., 1] * fc[..., 0] + xs[..., 0] * fc[..., 1],
    ], -1).flatten(3)
    return out.type_as(x)


class Attention:
    def __init__(self, w: dict, prefix: str):
        self.wqkv_w = w[f"{prefix}.wqkv.weight"]          # [1536,512] 无 bias
        self.wo_w = w[f"{prefix}.wo.weight"]

    def __call__(self, x: torch.Tensor, freqs: torch.Tensor,
                 mask: torch.Tensor | None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q, k, v = F.linear(x, self.wqkv_w).split(HEAD_DIM * N_HEAD, -1)
        q = q.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        k = k.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        v = v.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, HEAD_DIM * N_HEAD)
        return F.linear(y, self.wo_w)


class FeedForward:
    def __init__(self, w: dict, prefix: str):
        self.w1, self.w2, self.w3 = (
            w[f"{prefix}.w1.weight"], w[f"{prefix}.w2.weight"], w[f"{prefix}.w3.weight"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.w1)) * F.linear(x, self.w3), self.w2)


class TransformerBlock:
    def __init__(self, layer: int, w: dict):
        self.w = w
        self.p = f"cfm.estimator.transformer.layers.{layer}"
        self.attn = Attention(w, f"{self.p}.attention")
        self.ffn = FeedForward(w, f"{self.p}.feed_forward")
        self.has_skip = f"{self.p}.skip_in_linear.weight" in w

    def __call__(self, x: torch.Tensor, c: torch.Tensor, freqs: torch.Tensor,
                 mask: torch.Tensor | None, skip_in_x: torch.Tensor | None):
        w, p = self.w, self.p
        if skip_in_x is not None:
            x = F.linear(torch.cat([x, skip_in_x], -1), w[f"{p}.skip_in_linear.weight"],
                         w[f"{p}.skip_in_linear.bias"])
        # attention 分支
        nx = rmsnorm(x, w[f"{p}.attention_norm.norm.weight"])
        nx = adaln(nx, c, w[f"{p}.attention_norm.project_layer.weight"],
                   w[f"{p}.attention_norm.project_layer.bias"])
        h = x + self.attn(nx, freqs, mask)
        # FFN 分支
        nh = rmsnorm(h, w[f"{p}.ffn_norm.norm.weight"])
        nh = adaln(nh, c, w[f"{p}.ffn_norm.project_layer.weight"],
                   w[f"{p}.ffn_norm.project_layer.bias"])
        return h + self.ffn(nh)


class Transformer:
    """DiT 主体。forward: x[B,T,D] + c[B,D] → [B,T,D]（含 uvit skip + 尾 norm）。"""

    def __init__(self, w: dict):
        self.w = w
        self.blocks = [TransformerBlock(i, w) for i in range(N_LAYER)]
        # U-ViT skip：前一半 emit，后一半 receive（第 6 层除外）
        self.emit = list(range(N_LAYER // 2))                    # 0..5
        self.receive = list(range(N_LAYER // 2 + 1, N_LAYER))    # 7..12

    def forward(self, x: torch.Tensor, c: torch.Tensor, freqs: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        stack = []
        for i, blk in enumerate(self.blocks):
            skip = stack.pop() if i in self.receive else None
            x = blk(x, c, freqs, mask, skip)
            if i in self.emit:
                stack.append(x)
        w = self.w
        p = "cfm.estimator.transformer.norm"
        nx = rmsnorm(x, w[f"{p}.norm.weight"])
        return adaln(nx, c, w[f"{p}.project_layer.weight"], w[f"{p}.project_layer.bias"])


def load_transformer_weights(path: str) -> dict:
    """从 s2mel.safetensors 提取 transformer 相关权重（F16→F32）。"""
    from safetensors_loader import load_tensors
    w = load_tensors(path, fp16_to_fp32=True)
    keys = ["cfm.estimator.transformer.norm.norm.weight",
            "cfm.estimator.transformer.norm.project_layer.weight",
            "cfm.estimator.transformer.norm.project_layer.bias"]
    for i in range(N_LAYER):
        p = f"cfm.estimator.transformer.layers.{i}"
        for k in ("attention.wqkv.weight", "attention.wo.weight",
                  "attention_norm.norm.weight",
                  "attention_norm.project_layer.weight", "attention_norm.project_layer.bias",
                  "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight",
                  "ffn_norm.norm.weight",
                  "ffn_norm.project_layer.weight", "ffn_norm.project_layer.bias",
                  "skip_in_linear.weight", "skip_in_linear.bias"):
            keys.append(f"{p}.{k}")
    return {k: w[k] for k in keys if k in w}
