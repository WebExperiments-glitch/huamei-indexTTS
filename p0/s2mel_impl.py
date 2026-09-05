"""s2mel 完整实现（P2）：DiT + WaveNet 尾 + LengthRegulator + CFM 采样

自研，基于官方源码理解 + s2mel.safetensors 实测。
注意：safetensors conv 权重 MLX 布局 (O,K,I) → 用前 permute(0,2,1)；Linear 权重 (out,in) 直接可用。
"""
import torch
import torch.nn.functional as F

from s2mel_core import Transformer, precompute_freqs, load_transformer_weights, DIM

N_MELS = 80          # DiT in_channels = mel 通道数
STYLE_DIM = 192


# ---------------- 小工具 ----------------
def sequence_mask(length: torch.Tensor, max_len: int) -> torch.Tensor:
    x = torch.arange(max_len, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)          # [B,T]


class TimestepEmbedder:
    """scale=1000 正弦嵌入 + MLP(256→D, SiLU, D→D)。freqs 用权重表（buffer 已存）。"""

    def __init__(self, w: dict, prefix: str):
        self.scale = 1000.0
        self.freqs = w[f"{prefix}.freqs"].float()
        self.mlp0_w = w[f"{prefix}.linear1.weight"]; self.mlp0_b = w[f"{prefix}.linear1.bias"]
        self.mlp2_w = w[f"{prefix}.linear2.weight"]; self.mlp2_b = w[f"{prefix}.linear2.bias"]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:   # t [B]
        args = self.scale * t[:, None].float() * self.freqs[None]     # [B,128]
        emb = torch.cat([torch.cos(args), torch.sin(args)], -1)      # [B,256]
        h = F.linear(emb, self.mlp0_w, self.mlp0_b)
        h = F.silu(h)
        return F.linear(h, self.mlp2_w, self.mlp2_b)                  # [B,512]


# ---------------- WaveNet 尾 ----------------
def fused_gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """tanh(前半+gate) * sigmoid(后半+gate)；x/g [B,2C,T] / [B,2C,T]"""
    a = x + g
    half = a.shape[1] // 2
    return torch.tanh(a[:, :half]) * torch.sigmoid(a[:, half:])


class WaveNetTail:
    """WN(512, k5, dil=1, 8 层, cond 512) —— DiT 的最终层。"""

    def __init__(self, w: dict, prefix: str):
        self.w = w
        self.p = prefix
        self.ch = 512
        self.cond = _conv1d(w[f"{prefix}.cond_layer.conv.weight"],
                            w[f"{prefix}.cond_layer.conv.bias"], pad=0)   # 512→8192 k1
        self.in_layers = [_conv1d(w[f"{prefix}.in_layers.{i}.conv.weight"],
                                  w[f"{prefix}.in_layers.{i}.conv.bias"],
                                  pad=2, mode="reflect") for i in range(8)]  # k5 reflect!
        self.res_skip = [_conv1d(w[f"{prefix}.res_skip_layers.{i}.conv.weight"],
                                 w[f"{prefix}.res_skip_layers.{i}.conv.bias"],
                                 pad=0) for i in range(8)]

    def __call__(self, x: torch.Tensor, mask: torch.Tensor,
                 g: torch.Tensor) -> torch.Tensor:
        """x [B,512,T], mask [B,1,T] bool, g [B,512,1] → [B,512,T]"""
        g_all = _conv_apply(g, self.cond)                      # [B,8192,1]
        output = torch.zeros_like(x)
        xm = x
        for i in range(8):
            x_in = _conv_apply(xm, self.in_layers[i])
            gl = g_all[:, i * 2 * self.ch:(i + 1) * 2 * self.ch, :].expand(-1, -1, x.shape[2])
            acts = fused_gate(x_in, gl)           # [B,512,T]
            rsa = _conv_apply(acts, self.res_skip[i])
            if i < 7:
                xm = (xm + rsa[:, :self.ch]) * mask
                output = output + rsa[:, self.ch:]
            else:
                output = output + rsa
        return output * mask


# ---------------- 组装辅助 ----------------
def _linear(wt: torch.Tensor, bs: torch.Tensor):
    return wt, bs


def _conv1d(wt: torch.Tensor, bs: torch.Tensor, pad: int, mode: str = "zeros"):
    """MLX conv (O,K,I) → (permute后, bias, pad, mode)；调用统一走 _conv_apply。"""
    return wt.permute(0, 2, 1).contiguous(), bs, pad, mode


def _conv_apply(x, conv):
    cw, cb, pad, mode = conv
    if mode == "reflect":
        # 官方 SConv1d（encodec.py:212-228）：reflect 填充 pad+pad，conv 本身无 padding
        xp = F.pad(x, (pad, pad), mode="reflect")
        return F.conv1d(xp, cw, cb)
    return F.conv1d(x, cw, cb, padding=pad)


class FinalLayer:
    """LN(no-affine,1e-6) + adaLN(t1) + Linear(512→512)。"""

    def __init__(self, w: dict, prefix: str):
        self.linear_w = w[f"{prefix}.linear.weight"]; self.linear_b = w[f"{prefix}.linear.bias"]
        self.ada_w = w[f"{prefix}.adaLN_modulation.layers.1.weight"]
        self.ada_b = w[f"{prefix}.adaLN_modulation.layers.1.bias"]

    def __call__(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        nx = F.layer_norm(x, (x.shape[-1],), None, None, 1e-6)
        # 官方 adaLN_modulation = Sequential(SiLU, Linear)
        shift, scale = F.linear(F.silu(c), self.ada_w, self.ada_b).chunk(2, -1)
        nx = nx * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return F.linear(nx, self.linear_w, self.linear_b)


# ---------------- DiT（estimator）----------------
class DiT:
    """CFM 的 estimator。forward(x[B,80,T], prompt_x, x_lens, t[B], style[B,192], cond[B,T,512])"""

    def __init__(self, w: dict, pre: str = "cfm.estimator"):
        self.w = w
        self.pre = pre
        # 各 merge/proj 线性（weight (out,in)）
        def lin(k):
            return w[f"{pre}.{k}.weight"], w[f"{pre}.{k}.bias"]
        self.cond_proj = lin("cond_projection")                 # 512→512
        self.cond_x_merge = lin("cond_x_merge_linear")          # 80+80+512+192=864→512
        self.skip_lin = lin("skip_linear")                      # 512+80→512
        self.conv1 = lin("conv1")                               # Linear 512→512
        self.res_proj = lin("res_projection")                   # Linear 512→512
        # conv2: Conv1d 512→80 k1（MLX (80,1,512)）
        self.conv2 = _conv1d(w[f"{pre}.conv2.weight"], w[f"{pre}.conv2.bias"], 0)
        self.t_emb = TimestepEmbedder(w, f"{pre}.t_embedder")
        self.t_emb2 = TimestepEmbedder(w, f"{pre}.t_embedder2")
        self.transformer = Transformer(w)
        self.wn = WaveNetTail(w, f"{pre}.wavenet")
        self.final = FinalLayer(w, f"{pre}.final_layer")
        # 权重 shape 校验（对拍时用）
        self.cond_x_in = 80 * 2 + 512 + STYLE_DIM

    def __call__(self, x, prompt_x, x_lens, t, style, cond):
        w = self.w
        B, _, T = x.shape
        t1 = self.t_emb(t)                                        # [B,512]
        cond = F.linear(cond, *self.cond_proj)                    # [B,T,512]
        x = x.transpose(1, 2); px = prompt_x.transpose(1, 2)      # [B,T,80]
        x_in = torch.cat([x, px, cond], -1)                       # [B,T,672]
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], -1)   # [B,T,864]
        x_in = F.linear(x_in, *self.cond_x_merge)                 # [B,T,512]
        mask2 = sequence_mask(x_lens, x_in.size(1)).to(x.device).unsqueeze(1)  # [B,1,T]
        mask4 = mask2[:, None].repeat(1, 1, T, 1)                 # [B,1,T,T]（非因果）
        freqs = precompute_freqs(T)
        x_res = self.transformer.forward(x_in, t1, freqs.to(x.dtype), mask4)
        if f"{self.pre}.skip_linear.weight" in w:
            x_res = F.linear(torch.cat([x_res, x], -1), *self.skip_lin)  # 512+80→512
        # wavenet 尾
        h = F.linear(x_res, *self.conv1).transpose(1, 2)          # [B,512,T]
        t2 = self.t_emb2(t)
        xw = self.wn(h, mask2, t2.unsqueeze(2)).transpose(1, 2)   # [B,T,512]
        xw = xw + F.linear(x_res, *self.res_proj)
        xf = self.final(xw, t1).transpose(1, 2)                   # [B,512,T]
        return _conv_apply(xf, self.conv2)                          # [B,80,T]


# ---------------- LengthRegulator（复用已确认逻辑）----------------
class LengthRegulator:
    """content [B,T,1024] → interpolate(nearest) → 4×(Conv+GroupNorm(1)+Mish)+Conv1 → [B,T',512]"""

    def __init__(self, w: dict, pre: str = "length_regulator"):
        self.pre = pre
        self.content_proj = (w[f"{pre}.content_in_proj.weight"],
                             w[f"{pre}.content_in_proj.bias"])    # Linear 1024→512
        # model: conv idx 0,3,6,9 (k3 pad1) + norm 1,4,7,10 + 收尾 12 (k1)
        import torch.nn as nn
        self.convs = [_conv1d(w[f"{pre}.model.{i}.weight"], w[f"{pre}.model.{i}.bias"], pad=1)
                      for i in (0, 3, 6, 9)]
        self.norms = [(w[f"{pre}.model.{i}.weight"], w[f"{pre}.model.{i}.bias"])
                      for i in (1, 4, 7, 10)]
        self.final1x1 = _conv1d(w[f"{pre}.model.12.weight"], w[f"{pre}.model.12.bias"], 0)

    def __call__(self, x: torch.Tensor, ylens: torch.Tensor) -> torch.Tensor:
        """x [B,T,1024] → [B,T',512]（T' = ylens）"""
        x = F.linear(x, *self.content_proj)                        # [B,T,512]
        x = x.transpose(1, 2)                                      # [B,512,T]
        x = F.interpolate(x, size=int(ylens.item()), mode="nearest") if ylens.numel() else x
        for i in range(len(self.convs)):
            y = _conv_apply(x, self.convs[i])
            gw, gb = self.norms[i]
            y = F.group_norm(y, 1, gw, gb)
            x = F.mish(y)
        return _conv_apply(x, self.final1x1).transpose(1, 2)         # [B,T',512]


# ---------------- CFM 采样 ----------------
class CFM:
    """流匹配：25 步欧拉 + CFG(0.7)。prompt 区作锚并保持清零。"""

    def __init__(self, w: dict):
        self.est = DiT(w)

    @torch.no_grad()
    def inference(self, mu: torch.Tensor, x_len: int, prompt: torch.Tensor,
                  style: torch.Tensor, n_steps: int = 25,
                  cfg_rate: float = 0.7) -> torch.Tensor:
        """mu [B,T,512]（prompt 段+目标段）；prompt [B,80,P] 参考 mel；→ mel [B,80,T]"""
        B, T = mu.shape[0], mu.shape[1]
        prompt_len = prompt.shape[-1]
        x = torch.randn(B, N_MELS, T)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu2 = mu.clone()
        x_lens = torch.full((B,), T, dtype=torch.long)
        t_span = torch.linspace(0, 1, n_steps + 1)
        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t
            if cfg_rate > 0:
                px = torch.cat([prompt_x, torch.zeros_like(prompt_x)])
                st = torch.cat([style, torch.zeros_like(style)])
                m2 = torch.cat([mu2, torch.zeros_like(mu2)])
                xx = torch.cat([x, x]); tt = torch.stack([t.expand(B), t.expand(B)]).flatten()
                d = self.est(xx, px, x_lens.repeat(2), tt, st, m2)
                d, d0 = d.chunk(2, 0)
                d = (1 + cfg_rate) * d - cfg_rate * d0
            else:
                d = self.est(x, prompt_x, x_lens, t.expand(B), style, mu2)
            x = x + dt * d
            x[..., :prompt_len] = 0
            t = t + dt
        return x


def load_s2mel_weights(path: str) -> dict:
    """读 s2mel.safetensors 全部权重（F16→F32）。"""
    from safetensors_loader import load_tensors
    return load_tensors(path, fp16_to_fp32=True)
