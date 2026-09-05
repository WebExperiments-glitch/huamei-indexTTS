"""P2b：s2mel（LengthRegulator / DiT / CFM）与官方 PyTorch 逐算子对拍

官方构造走 MyModel(cfg.s2mel)（配置来自 models/config.yaml）。
两处适配：
  1) 剥 weight_norm —— 转换后权重已合并成裸 weight
  2) key 版本漂移 —— 权重是 t_embedder.linear1/linear2，reference 源码是 mlp.0/mlp.2
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REF = r"D:/indexTTS 2.5/reference/index-tts-main"
MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit"
sys.path.insert(0, HERE)
sys.path.insert(0, REF)

from s2mel_impl import CFM, LengthRegulator, load_s2mel_weights  # noqa: E402


def mx(a, b):
    d = (a - b).abs()
    return (f"cos={F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.6f}  "
            f"maxΔ={d.max().item():.3e}  meanΔ={d.mean().item():.3e}")


print("== 加载权重 ==")
w = load_s2mel_weights(os.path.join(MODEL, "s2mel.safetensors"))
print(f"   {len(w)} 张量")

print("== 构造官方 MyModel(cfg.s2mel) ==")
import yaml
from munch import Munch
from torch.nn.utils import remove_weight_norm

# stub dac 子包：length_regulator 只从它取 VectorQuantize，
# 而本模型 vector_quantize=False，该分支根本不执行。
# （真实 dac 链会拉 audiotools / argbind / torchaudio，与推理无关）
for _n in ("indextts.s2mel.dac", "indextts.s2mel.dac.nn"):
    _m = types.ModuleType(_n)
    _m.__path__ = []
    sys.modules[_n] = _m
_q = types.ModuleType("indextts.s2mel.dac.nn.quantize")


class VectorQuantize:             # 占位；vector_quantize=False 时不会被实例化
    def __init__(self, *a, **k):
        raise RuntimeError("不应实例化：本配置 vector_quantize=False")


_q.VectorQuantize = VectorQuantize
sys.modules["indextts.s2mel.dac.nn.quantize"] = _q

cfg = Munch.fromDict(yaml.safe_load(open(os.path.join(MODEL, "config.yaml"), encoding="utf-8")))
from indextts.s2mel.modules.commons import MyModel

off = MyModel(cfg.s2mel).eval()
n_rm = 0
for m in off.modules():
    if hasattr(m, "weight_g"):
        remove_weight_norm(m); n_rm += 1
print(f"   剥离 weight_norm 层数: {n_rm}")

# ---- key 适配 + 布局转换 ----
CONV3D_KEYS = (".conv.weight", "conv1.weight", "conv2.weight", "cond_layer.conv.weight",
               ".in_layers.", ".res_skip_layers.", "embed.weight", "dwconv.weight")


def adapt(k: str, v: torch.Tensor):
    # 版本漂移：mlp.0/mlp.2 ← linear1/linear2
    for pre in ("cfm.estimator.t_embedder.", "cfm.estimator.t_embedder2."):
        if k.startswith(pre):
            rest = k[len(pre):]
            if rest.startswith("linear1."):
                return pre + "mlp.0." + rest.split(".", 1)[1], v
            if rest.startswith("linear2."):
                return pre + "mlp.2." + rest.split(".", 1)[1], v
            return None            # freqs 等 buffer 由代码生成
    # 版本漂移：FinalLayer 的 adaLN_modulation 权重在转换版里带 layers. 前缀
    if "final_layer.adaLN_modulation.layers.1." in k:
        k = k.replace("final_layer.adaLN_modulation.layers.1.",
                      "final_layer.adaLN_modulation.1.")
    # wavenet 的 SConv1d 多一层包装：X.conv.weight → X.conv.conv.weight
    if ".wavenet." in k and k.endswith(".conv.weight"):
        k = k[: -len(".conv.weight")] + ".conv.conv.weight"
        if v.dim() == 3:
            v = v.permute(0, 2, 1).contiguous()
        return k, v
    if ".wavenet." in k and k.endswith(".conv.bias"):
        return k[: -len(".conv.bias")] + ".conv.conv.bias", v
    # s2mel 里所有 3 维权重都是 Conv1d，MLX (O,K,I) → torch (O,I,K)
    # （transformer 全是 1/2 维；t_embedder.freqs 已在上面跳过）
    if v.dim() == 3:
        v = v.permute(0, 2, 1).contiguous()
    return k, v


# MyModel 把模块放在 self.models(ModuleDict) 下 → 官方 key 带 "models." 前缀
off.models["cfm"].estimator.setup_caches(max_batch_size=2, max_seq_length=8192)

state = {}
for k, v in w.items():
    r = adapt(k, v)
    if r is not None:
        state["models." + r[0]] = r[1]
miss, unexp = off.load_state_dict(state, strict=False)
print(f"   load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("     missing 示例:", miss[:8])
if unexp:
    print("     unexpected 示例:", unexp[:8])

mine_lr = LengthRegulator(w)
mine_cfm = CFM(w)
off_lr = off.models["length_regulator"]
off_cfm = off.models["cfm"]

# ---------------- [1] LengthRegulator ----------------
print("\n== [1] LengthRegulator 对拍 ==")
torch.manual_seed(0)
S = torch.randn(1, 12, 1024)
ylens = torch.tensor([20])
with torch.no_grad():
    a = mine_lr(S, ylens)
    b = off_lr(S, ylens=ylens, n_quantizers=3, f0=None)[0]
print(f"   自研 {tuple(a.shape)}  官方 {tuple(b.shape)}")
print(f"   {mx(a, b)}   {'✅' if F.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.9999 else '❌'}")

# ---------------- [2] DiT 单步 ----------------
print("\n== [2] DiT 单步对拍 ==")
torch.manual_seed(1)
B, T = 1, 20
x = torch.randn(B, 80, T)
prompt_x = torch.zeros(B, 80, T)
mu = torch.randn(B, T, 512)
style = torch.randn(B, 192)
t = torch.tensor([0.3])
with torch.no_grad():
    a = mine_cfm.est(x, prompt_x, torch.tensor([T]), t, style, mu)
    b = off_cfm.estimator(x, prompt_x, torch.tensor([T]).long(), t, style, mu)
print(f"   自研 {tuple(a.shape)}  官方 {tuple(b.shape)}")
print(f"   {mx(a, b)}   {'✅' if F.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.9999 else '❌'}")

# ---------------- [3] CFM 完整采样 ----------------
print("\n== [3] CFM 采样对拍（8 步，固定 seed）==")
torch.manual_seed(2)
mu2 = torch.randn(1, 30, 512)
prompt = torch.randn(1, 80, 10)
style2 = torch.randn(1, 192)
torch.manual_seed(1234)
a = mine_cfm.inference(mu2, 30, prompt, style2, n_steps=8, cfg_rate=0.7)
torch.manual_seed(1234)
b = off_cfm.inference(mu2, torch.tensor([30]).long(), prompt, style2, None, 8, 1.0, 0.7)
print(f"   自研 {tuple(a.shape)}  官方 {tuple(b.shape)}")
c = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
print(f"   {mx(a, b)}   {'✅' if c > 0.999 else '❌'}")
