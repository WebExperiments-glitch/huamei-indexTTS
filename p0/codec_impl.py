"""Semantic Codec（P3）—— Vocos 系 ConvNeXt + FVQ，自研

官方 decode 路径（indextts/codec/models.py EnhancedCodec.decode）：
    codes [B,T] 或 [N,B,T]
      → quantizer.vq2emb(codes)          # FVQ: embedding(codebook) → out_project(Conv1d 8→1024 k1)
      → decoder(VocosBackbone → Linear)  # [B,1024,T] → [B,T,384] → [B,T,1024]
      → upsample ×2 (nearest) + up conv  # [B,2T,1024]
25fps 的 GPT codes → 50fps 的 S_infer（再经 LengthRegulator ×1.72 → 86fps mel）

布局：safetensors 用 MLX (O,K,I)，torch Conv1d 需 (O,I,K) → permute(0,2,1)。
激活默认 GELU(exact)，LayerNorm eps=1e-6（与官方 VocosBackbone 一致）。
"""
import torch
import torch.nn.functional as F

CODEBOOK_SIZE = 8192
CODEBOOK_DIM = 8
HIDDEN = 1024
VOCOS_DIM = 384
VOCOS_LAYERS = 12
INTER_DIM = 2048


def _conv(wt: torch.Tensor, bs: torch.Tensor, pad: int, groups: int = 1):
    """MLX (O,K,I) → torch (O,I,K)，打包 (w, b, pad, groups)。"""
    return wt.permute(0, 2, 1).contiguous(), bs, pad, groups


def _c(x, conv):
    cw, cb, pad, groups = conv
    return F.conv1d(x, cw, cb, padding=pad, groups=groups)


class ConvNeXtBlock:
    """dwconv(k7,depthwise) → LN → pwconv1 → GELU → pwconv2 → gamma· → 残差"""

    def __init__(self, w: dict, pre: str):
        self.dwconv = _conv(w[f"{pre}.dwconv.weight"], w[f"{pre}.dwconv.bias"],
                            3, groups=VOCOS_DIM)          # depthwise: groups=C
        self.nw, self.nb = w[f"{pre}.norm.weight"], w[f"{pre}.norm.bias"]
        self.p1w, self.p1b = w[f"{pre}.pwconv1.weight"], w[f"{pre}.pwconv1.bias"]
        self.p2w, self.p2b = w[f"{pre}.pwconv2.weight"], w[f"{pre}.pwconv2.bias"]
        self.gamma = w[f"{pre}.gamma"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,C,T] → [B,C,T]"""
        res = x
        x = _c(x, self.dwconv)
        x = x.transpose(1, 2)                                   # [B,T,C]
        x = F.layer_norm(x, (x.shape[-1],), self.nw, self.nb, 1e-6)
        x = F.linear(x, self.p1w, self.p1b)
        x = F.gelu(x)
        x = F.linear(x, self.p2w, self.p2b)
        x = self.gamma * x
        x = x.transpose(1, 2)                                   # [B,C,T]
        return res + x


class VocosBackbone:
    """embed(Conv1d k7) → LN → 12×ConvNeXtBlock → final LN；输入/输出均为 [B,C,T]→[B,T,C]"""

    def __init__(self, w: dict, pre: str):
        self.embed = _conv(w[f"{pre}.embed.weight"], w[f"{pre}.embed.bias"], 3)
        self.nw, self.nb = w[f"{pre}.norm.weight"], w[f"{pre}.norm.bias"]
        self.blocks = [ConvNeXtBlock(w, f"{pre}.convnext.{i}") for i in range(VOCOS_LAYERS)]
        self.fw, self.fb = (w[f"{pre}.final_layer_norm.weight"],
                            w[f"{pre}.final_layer_norm.bias"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = _c(x, self.embed)                                   # [B,384,T]
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (x.shape[-1],), self.nw, self.nb, 1e-6)
        x = x.transpose(1, 2)                                   # [B,384,T]
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)
        return F.layer_norm(x, (x.shape[-1],), self.fw, self.fb, 1e-6)   # [B,T,384]


class SemanticCodec:
    """只实现推理需要的 decode（encoder / quantize 不需要）。"""

    def __init__(self, w: dict):
        self.w = w
        self.codebook = w["quantizer.quantizers.0.codebook.weight"]       # [8192,8]
        self.out_proj = _conv(w["quantizer.quantizers.0.out_project.weight"],
                              w["quantizer.quantizers.0.out_project.bias"], 0)
        self.backbone = VocosBackbone(w, "decoder.0")
        self.head_w, self.head_b = w["decoder.1.weight"], w["decoder.1.bias"]
        self.up = _conv(w["up.weight"], w["up.bias"], 1)                  # k3 pad1

    @torch.no_grad()
    def vq2emb(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B,T] → [B,1024,T]（单量化器：embedding + out_project）"""
        emb = F.embedding(codes, self.codebook).transpose(1, 2)           # [B,8,T]
        return _c(emb, self.out_proj)                                     # [B,1024,T]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B,T]（或 [N,B,T]，取第 0 个量化器）→ S_infer [B,2T,1024]"""
        if codes.dim() == 3:
            codes = codes[0]
        x = self.vq2emb(codes)                                            # [B,1024,T]
        x = self.backbone(x)                                              # [B,T,384]
        x = F.linear(x, self.head_w, self.head_b)                         # [B,T,1024]
        x = x.transpose(1, 2)                                             # [B,1024,T]
        x = F.interpolate(x, scale_factor=2, mode="nearest")              # [B,1024,2T]
        return _c(x, self.up).transpose(1, 2)                             # [B,2T,1024]


def load_codec_weights(path: str) -> dict:
    from safetensors_loader import load_tensors
    return load_tensors(path, fp16_to_fp32=True)
