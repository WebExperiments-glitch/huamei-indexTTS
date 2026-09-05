"""GPT（UnifiedVoice）自研核心 —— P1b

只实现"transformer 主体 + 嵌入/投影层"的加载与单次前向（自回归采样后续做）。
结构依据（自研，对照公开 GPT-2 架构）：
  - 24 层，d_model=1280，20 头（head 64），MLP 中间 5120
  - c_attn 融合 QKV（3840=3×1280），GPT2 风格；激活 gelu_new
  - 无内部 wte/wpe（位置嵌入由外部 text/mel 位置表显式加）
8bit 权重：U32 打包 int8 × scales + biases（group 64），公式已实测验证。
"""
import torch
import torch.nn.functional as F

GROUP = 64
GPT_LAYERS = 24
D_MODEL = 1280
N_HEADS = 20
HEAD_DIM = D_MODEL // N_HEADS
MLP_DIM = 5120


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2 的 tanh 近似 GELU（HF 默认 activation_function）。"""
    c = 0.7978845608028654          # sqrt(2/pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


def dequant_affine(w_u8: torch.Tensor, scales: torch.Tensor,
                   biases: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """w_u8[O,I](uint8) × scales + biases → F32[O,I]。公式已实测验证。"""
    o, i = w_u8.shape
    wf = w_u8.float().view(o, i // group, group)
    return (wf * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(o, i)


def load_gpt_weights(path: str, layers: int = GPT_LAYERS) -> dict[str, torch.Tensor]:
    """读 gpt.safetensors，把量化 Linear 反量化成 F32（便于对拍/调试）。

    返回 key → tensor（仅 transformer 相关：gpt.h.*、gpt.ln_f、final_norm、head）。
    """
    from safetensors_loader import load_tensors
    w = load_tensors(path, fp16_to_fp32=True)
    out: dict[str, torch.Tensor] = {}
    # 各 Linear 的 (weight, scales, biases, bias) → dequant
    lin = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    for i in range(layers):
        for m in lin:
            p = f"gpt.h.{i}.{m}"
            out[f"{p}.weight"] = dequant_affine(w[f"{p}.weight"], w[f"{p}.scales"],
                                                w[f"{p}.biases"])
            out[f"{p}.bias"] = w[f"{p}.bias"]
        for n in ("ln_1", "ln_2"):
            out[f"gpt.h.{i}.{n}.weight"] = w[f"gpt.h.{i}.{n}.weight"]
            out[f"gpt.h.{i}.{n}.bias"] = w[f"gpt.h.{i}.{n}.bias"]
    for n in ("weight", "bias"):
        out[f"gpt.ln_f.{n}"] = w[f"gpt.ln_f.{n}"]
        out[f"final_norm.{n}"] = w[f"final_norm.{n}"]
    out["mel_head.weight"] = w["mel_head.weight"]
    out["mel_head.bias"] = w["mel_head.bias"]
    # 嵌入表 / 投影（P1c 自回归用）
    for key in ("text_embedding.weight", "mel_embedding.weight", "lang_embedding.weight",
                "text_pos_embedding.emb.weight", "mel_pos_embedding.emb.weight",
                "spk_emb_proj.weight", "spk_emb_proj.bias",
                "emo_layer.weight", "emo_layer.bias",
                "emovec_layer.weight", "emovec_layer.bias",
                "text_head.weight", "text_head.bias"):
        if key in w:
            out[key] = w[key]
    return out


def causal_attn(q, k, v, scale) -> torch.Tensor:
    """简单因果注意力（全序列，非 KV-cache 版）。q/k/v: [B, H, T, D]"""
    attn = q @ k.transpose(-2, -1) * scale          # [B,H,T,T]
    T = attn.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attn.device), 1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = attn.softmax(dim=-1)
    return attn @ v


class GPT2Block:
    """带权重的 transformer 层：ln_1→attn→残差→ln_2→mlp→残差。"""

    def __init__(self, layer: int, w: dict):
        self.w = w
        self.p = f"gpt.h.{layer}"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        p = self.p
        # LayerNorm
        nx = F.layer_norm(x, (D_MODEL,), w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"], 1e-5)
        # fused QKV
        qkv = F.linear(nx, w[f"{p}.attn.c_attn.weight"], w[f"{p}.attn.c_attn.bias"])   # [B,T,3D]
        b, t, _ = qkv.shape
        qkv = qkv.view(b, t, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)  # [3,B,H,T,D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = causal_attn(q, k, v, HEAD_DIM ** -0.5).transpose(1, 2).reshape(b, t, D_MODEL)
        a = F.linear(a, w[f"{p}.attn.c_proj.weight"], w[f"{p}.attn.c_proj.bias"])
        x = x + a
        # MLP
        nx = F.layer_norm(x, (D_MODEL,), w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"], 1e-5)
        h = gelu_new(F.linear(nx, w[f"{p}.mlp.c_fc.weight"], w[f"{p}.mlp.c_fc.bias"]))
        h = F.linear(h, w[f"{p}.mlp.c_proj.weight"], w[f"{p}.mlp.c_proj.bias"])
        return x + h


class GPTCore:
    """transformer 主体 + ln_f（输入已是 embedding，无位置逻辑）。"""

    def __init__(self, w: dict):
        self.w = w
        self.blocks = [GPT2Block(i, w) for i in range(GPT_LAYERS)]

    def __call__(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb
        for blk in self.blocks:
            x = blk(x)
        x = F.layer_norm(x, (D_MODEL,), self.w["gpt.ln_f.weight"],
                         self.w["gpt.ln_f.bias"], 1e-5)
        return x

    def mel_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden[B,T,D] → mel_head → logits[B,T,8194]"""
        w = self.w
        x = F.layer_norm(hidden, (D_MODEL,), w["final_norm.weight"],
                         w["final_norm.bias"], 1e-5)
        return F.linear(x, w["mel_head.weight"], w["mel_head.bias"])
