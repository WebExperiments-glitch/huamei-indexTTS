"""BigVGAN 声码器 —— torch 自研实现（P0a）

基于对官方推理版结构（reference/.../s2mel/modules/bigvgan/bigvgan.py）的理解，从零编写。
关键约定（与 MLX 转换版一致）：
  1. 激活 = 裸 SnakeBeta，无 AntiAlias 滤波（MLX 转换时已丢弃抗混叠滤波器，见 manifest ignored）。
  2. safetensors 卷积权重是 MLX 布局 (O, K, I)；torch 需要 (O, I, K) / 转置卷积 (I, O, K)，
     使用时 permute，不改存储。
  3. SnakeBeta 的 alpha/beta 存 log 空间（alpha_logscale=True）→ 前向需 exp()。
     （是否 logscale 由 test 里打印数值分布确认，默认 True）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def snake_beta(x: torch.Tensor, alpha_log: torch.Tensor, beta_log: torch.Tensor,
               logscale: bool = True) -> torch.Tensor:
    """SnakeBeta ∶= x + 1/(beta+eps) * sin²(x * alpha)

    alpha_log/beta_log: [C]；logscale=True 时存的是 log(alpha)/log(beta)。
    """
    a = torch.exp(alpha_log) if logscale else alpha_log
    b = torch.exp(beta_log) if logscale else beta_log
    a = a.view(1, -1, 1)
    b = b.view(1, -1, 1)
    return x + torch.sin(x * a).pow(2) / (b + 1e-9)


class AMPBlock1(nn.Module):
    """3 对 (Snake→conv1→Snake→conv2) + 残差。

    关键（实测发现）：每个 stage 的 3 个 resblock 卷积核大小不同 ——
    官方 config resblock_kernel_sizes=[3,7,11]，即 blocks{3i}.k=3 / {3i+1}.k=7 / {3i+2}.k=11。
    卷积核 K 直接从权重 shape 第 1 维读取，padding 按 (K-1)*dilation//2 计算。
    convs1: dilation=(1,3,5)；convs2: dilation=1（紧随 convs1 之后）。
    激活配对：activations[2i] 用于 conv1 前，activations[2i+1] 用于 conv2 前。
    """

    def __init__(self, block_idx: int, w: dict):
        super().__init__()
        self.w = w
        self.prefix = f"resblocks.{block_idx}"

    def _conv_wb(self, group: str, j: int):
        """返回 (weight_torch, bias, kernel_K)；weight [O,K,I] → [O,I,K]。"""
        w = self.w[f"{self.prefix}.{group}.{j}.weight"]     # [O,K,I] (MLX 布局)
        b = self.w[f"{self.prefix}.{group}.{j}.bias"]
        return w.permute(0, 2, 1), b, w.shape[1]

    def forward(self, x):
        for i, dil in enumerate((1, 3, 5)):
            al = self.w[f"{self.prefix}.activations.{2*i}.act.alpha"]
            be = self.w[f"{self.prefix}.activations.{2*i}.act.beta"]
            xt = snake_beta(x, al, be)
            cw, cb, k = self._conv_wb("convs1", i)
            xt = F.conv1d(xt, cw, cb, padding=(k - 1) * dil // 2, dilation=dil)

            al = self.w[f"{self.prefix}.activations.{2*i+1}.act.alpha"]
            be = self.w[f"{self.prefix}.activations.{2*i+1}.act.beta"]
            xt = snake_beta(xt, al, be)
            cw, cb, k = self._conv_wb("convs2", i)
            xt = F.conv1d(xt, cw, cb, padding=(k - 1) // 2, dilation=1)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    """推理版 BigVGAN：mel(80) → 6 级上采样(×256) → 18 AMPBlock1 → 波形。

    forward 全程用 F.conv1d / F.conv_transpose1d + 显式权重，逐层可断点、可对拍。
    """

    def __init__(self, w: dict):
        super().__init__()
        self.w = w
        # conv_pre: [1536,7,80] (O,K,I)
        self.conv_pre_w = w["conv_pre.weight"].permute(0, 2, 1)
        self.conv_pre_b = w["conv_pre.bias"]
        # conv_post: [1,7,24] —— 官方 use_bias_at_final=False → 无 bias
        self.conv_post_w = w["conv_post.weight"].permute(0, 2, 1)
        self.conv_post_b = w.get("conv_post.bias")
        # 18 个 resblock（6 阶段 × 3），索引 0..17
        self.blocks = nn.ModuleList(AMPBlock1(i, w) for i in range(18))
        # activation_post 的 alpha/beta（末级 24 通道）
        self.post_alpha = w["activation_post.act.alpha"]
        self.post_beta = w["activation_post.act.beta"]
        self.upsample_rates = (4, 4, 2, 2, 2, 2)

    def _upsample_stage(self, x, stage: int):
        """ConvTranspose1d：w=[O,K,I](safetensors) → permute→[I,O,K]，stride/ kernel 逐段已知。"""
        kernels = (8, 8, 4, 4, 4, 4)
        u, k = self.upsample_rates[stage], kernels[stage]
        wt = self.w[f"ups.{stage}.weight"]          # [O,K,I]
        w = wt.permute(2, 0, 1).contiguous()         # [I,O,K]
        b = self.w[f"ups.{stage}.bias"]
        return F.conv_transpose1d(x, w, b, stride=u, padding=(k - u) // 2)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 80, T] → wav: [B, 1, T*256]（约）。"""
        x = F.conv1d(mel, self.conv_pre_w, self.conv_pre_b, padding=3)
        for stage in range(6):
            x = self._upsample_stage(x, stage)
            # 该阶段 3 个 AMPBlock 求和取平均
            acc = None
            for j in range(3):
                out = self.blocks[stage * 3 + j](x)
                acc = out if acc is None else acc + out
            x = acc / 3.0
        x = snake_beta(x, self.post_alpha, self.post_beta)
        x = F.conv1d(x, self.conv_post_w, self.conv_post_b, padding=3)
        return torch.tanh(x)


def build_from_safetensors(path: str) -> BigVGAN:
    """从 MLX safetensors 加载并构造模型（权重原地转 F32 + permute 在 forward 内做）。"""
    from safetensors_loader import load_tensors
    w = load_tensors(path, fp16_to_fp32=True)
    return BigVGAN(w)
