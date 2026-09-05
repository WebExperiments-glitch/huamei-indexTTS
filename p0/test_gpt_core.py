"""P1b 测试：自研 GPT transformer vs 官方 transformers.GPT2Model —— 主体对拍

用官方 transformers GPT2Model（IndexTTS 推理用的同款结构）加载同一 gpt.h 权重，
wpe 置零（位置由外部管理），相同 inputs_embeds → 对比 last_hidden_state。
跑：python p0/test_gpt_core.py（首次需加载 1.18GB 权重并反量化，约 1-3 分钟）
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/gpt.safetensors"

from gpt_core import load_gpt_weights, GPTCore, D_MODEL, GPT_LAYERS, N_HEADS, MLP_DIM  # noqa: E402


def build_official(w: dict, n_layers: int = GPT_LAYERS):
    from transformers import GPT2Config, GPT2Model
    cfg = GPT2Config(n_layer=n_layers, n_embd=D_MODEL, n_head=N_HEADS, n_inner=MLP_DIM,
                     activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0,
                     attn_pdrop=0.0, layer_norm_epsilon=1e-5, use_cache=False)
    m = GPT2Model(cfg)
    # wpe 与 wte 由外部管理/置零 → wpe 置 0，wte 权重从 h 不包含（inputs_embeds 路径不查 wte）
    with torch.no_grad():
        m.wpe.weight.zero_()
    sd = {}
    for k, v in w.items():
        if k.startswith("gpt.h.") or k.startswith("gpt.ln_f."):
            kk = k[len("gpt."):]                    # 剥 UnifiedVoice 前缀
            # HF GPT2 的 attn/mlp Linear 是 Conv1D(in,out)；我们是 (out,in) → 转置
            if kk.endswith(".weight") and any(s in kk for s in
                  ("attn.c_attn.", "attn.c_proj.", "mlp.c_fc.", "mlp.c_proj.")):
                v = v.t().contiguous()
            sd[kk] = v
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # missing 应为 wte/wpe
    m.eval()
    return m, missing


def main():
    t0 = time.time()
    print("== [1] 加载 + 反量化 gpt.safetensors ==")
    w = load_gpt_weights(MODEL)
    gb = sum(v.numel() * 4 for v in w.values()) / 2**30
    print(f"    反量化后权重 {gb:.1f} GB (F32), {time.time()-t0:.0f}s")

    print("\n== [2] 自研 GPTCore ==")
    mine = GPTCore(w)

    print("== [3] 官方 transformers.GPT2Model ==")
    off, missing = build_official(w)
    print(f"    load_state_dict missing(应为 wte/wpe): {missing[:4]}")

    print("\n== [4] 同输入前向对拍 ==")
    torch.manual_seed(1)
    seq = 48
    emb = torch.randn(1, seq, D_MODEL)
    attn_mask = torch.ones(1, seq, dtype=torch.long)
    pos = torch.zeros(1, seq, dtype=torch.long)     # wpe 已置零 → 位置无关

    with torch.no_grad():
        m_out = mine(emb)
        o_out = off(inputs_embeds=emb, attention_mask=attn_mask,
                    position_ids=pos).last_hidden_state
    print(f"    自研: {tuple(m_out.shape)}  官方: {tuple(o_out.shape)}")
    diff = (m_out - o_out).abs()
    cos = torch.nn.functional.cosine_similarity(m_out.flatten(), o_out.flatten(), dim=0)
    print(f"    cosine = {cos.item():.8f}")
    print(f"    maxΔ   = {diff.max().item():.8f}")
    print(f"    meanΔ  = {diff.mean().item():.8f}")

    print("\n== [5] mel_logits 合理性 ==")
    with torch.no_grad():
        lg = mine.mel_logits(emb)
    print(f"    logits {tuple(lg.shape)}  finite={bool(torch.isfinite(lg).all())}  "
          f"std={lg.std():.4f}  (8194 = 码本+start+stop)")

    assert torch.allclose(m_out, o_out, atol=1e-3, rtol=1e-3), "主体对拍不一致！"
    print(f"\n✅ P1b 主体通过（与官方 GPT2Model 一致），总耗时 {time.time()-t0:.0f}s")
    # 释放 1.18GB 权重
    del w, off, mine
    import gc; gc.collect()


if __name__ == "__main__":
    main()
