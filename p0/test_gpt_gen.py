"""P1c 测试：KV cache 自回归（自洽三测）
1) 前缀拼接形状/数值手算一致
2) KV 单步 == 全量重算末位置（数学等价验证）
3) 采样跑通：生成 mel codes、stop 机制、分布不退化
跑：python p0/test_gpt_gen.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit/gpt.safetensors"

from gpt_core import load_gpt_weights, D_MODEL, GPTCore  # noqa: E402
from gpt_gen import GPTGenerator, KVCache, START_MEL, STOP_MEL, STOP_TEXT  # noqa: E402
from tiktoken_bpe import Tokenizer  # noqa: E402


def main():
    t0 = time.time()
    print("== [0] 加载权重 + 分词器 ==")
    w = load_gpt_weights(MODEL)
    tok = Tokenizer(os.path.join(os.path.dirname(MODEL),
                                 "multilingual_zh_ja_yue_char_del.tiktoken"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "specials.json"))

    gen = GPTGenerator(w)
    spk = torch.randn(192)
    emo = torch.randn(1280)
    text_ids = tok.encode("<|zh|> hello world 你好")
    text_ids.append(STOP_TEXT)          # 模拟 infer 的 F.pad(尾 stop)

    print("\n== [1] 前缀构造 ==")
    prefix = gen.prepare_prefix(text_ids, spk, emo, lang_id=1)
    L_clean = len([i for i in text_ids if i not in (0, 1)])
    expect_len = 3 + (L_clean + 2)
    print(f"    prefix {tuple(prefix.shape)}，期望 3+文本(L+2)={expect_len} → "
          f"{'✓' if prefix.shape[0] == expect_len else '✗'}")
    # 手算校验 cond0 = spk_proj+emo、后 2 行为零
    cond0 = torch.nn.functional.linear(spk, w["spk_emb_proj.weight"],
                                       w["spk_emb_proj.bias"]) + emo
    assert torch.allclose(prefix[0], cond0, atol=1e-5)
    assert prefix[1].abs().max() == 0 and prefix[2].abs().max() == 0
    print("    cond0=spk_proj+emo ✓  cond1/2=零 ✓")

    print("\n== [2] KV 单步 == 全量重算（数学等价）==")
    torch.manual_seed(0)
    # 全量参考：GPTCore 对整个序列（前缀 + start_mel(位置0) + mel_toks(位置1..)）无 KV 因果前向
    full = GPTCore(w)
    mel_pos = w["mel_pos_embedding.emb.weight"]
    mel_toks = [112, 34, 256, 8191, 77]
    seqs = [prefix]
    seqs.append((w["mel_embedding.weight"][START_MEL] + mel_pos[0]).unsqueeze(0))
    for j, tk in enumerate(mel_toks, 1):
        seqs.append((w["mel_embedding.weight"][tk] + mel_pos[j]).unsqueeze(0))
    full_seq = torch.cat(seqs, 0)
    ref = full(full_seq.unsqueeze(0))[:, -1]        # 全量末位置 hidden
    # KV：首(前缀+start) + 逐 token
    gen.kvc.reset()
    start = w["mel_embedding.weight"][START_MEL] + mel_pos[0]
    gen.kvc.full_forward(torch.cat([prefix, start.unsqueeze(0)]).unsqueeze(0))
    for j, tk in enumerate(mel_toks, 1):
        h = gen.kvc.step((w["mel_embedding.weight"][tk] + mel_pos[j])
                         .unsqueeze(0).unsqueeze(0))
    diff = (h.squeeze(0) - ref).abs()
    print(f"    cos={torch.nn.functional.cosine_similarity(h.flatten(), ref.flatten(), dim=0).item():.8f}  "
          f"maxΔ={diff.max().item():.8f}  → {'✓ KV 正确' if diff.max() < 1e-4 else '✗'}")

    print("\n== [3] 采样生成（seed 固定）==")
    t1 = time.time()
    codes = gen.generate(prefix, seed=11, max_tokens=60)
    dt = time.time() - t1
    print(f"    生成 {len(codes)} 个 mel codes（耗时 {dt:.1f}s，"
          f"{dt/max(len(codes),1)*1000:.0f} ms/token）")
    print(f"    前 12 个: {codes[:12]}")
    import collections
    c = collections.Counter(codes)
    print(f"    唯一 token {len(c)}/{len(codes)}，范围 [{min(codes)}, {max(codes)}] "
          f"(应 <8192，越界即错) → {'✓ 合理' if max(codes) < 8192 and len(codes) > 0 else '✗'}")
    print(f"    是否到 stop 自然停: {'达 60 上限' if len(codes) >= 60 else '提前停'}")
    print(f"\n✅ P1c 自洽三测完成，总耗时 {time.time()-t0:.0f}s")
    del w


if __name__ == "__main__":
    main()
