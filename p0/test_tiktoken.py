"""P1a 测试：自研 tiktoken BPE vs 官方 tiktoken.Encoding —— golden 对照

跑：python p0/test_tiktoken.py
首次运行会从 reference 源码提取语言/事件/情绪表，生成 p0/data/specials.json。
断言：多语言/中英/标点/数字句子的编码结果与官方完全一致（逐 id）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = r"D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit"
SPECIALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specials.json")

from tiktoken_bpe import Tokenizer, dump_specials_to_json, _PAT  # noqa: E402

# ---- 0) 生成/确认 specials.json ----
if not os.path.exists(SPECIALS):
    dump_specials_to_json(SPECIALS)

# ---- 1) 官方 encoding（真值 = tiktoken 库标准行为，同一份词表数据）----
import base64
import json
import tiktoken

ranks = {}
with open(os.path.join(BASE, "multilingual_zh_ja_yue_char_del.tiktoken"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            tok, r = line.split()
            ranks[base64.b64decode(tok)] = int(r)

with open(SPECIALS, encoding="utf-8") as f:
    sp_ordered = list(json.load(f).keys())          # JSON 保序
official = tiktoken.Encoding(
    name="golden",
    pat_str=_PAT,
    mergeable_ranks=ranks,
    special_tokens={t: len(ranks) + i for i, t in enumerate(sp_ordered)},
    explicit_n_vocab=len(ranks) + len(sp_ordered),
)

# ---- 2) 自研 tokenizer ----
mine = Tokenizer(os.path.join(BASE, "multilingual_zh_ja_yue_char_del.tiktoken"), SPECIALS)
assert len(mine.ranks) == 58836, f"base vocab {len(mine.ranks)} != 58836"
assert len(mine.special) == 1673, f"specials {len(mine.special)} != 1673 (官方 num_languages=99)"
print(f"自研词表: base={len(mine.ranks)} + special={len(mine.special)} = {len(mine.ranks)+len(mine.special)}  (官方 vocab=60509)")

# ---- 3) 对照 ----
cases = [
    "hello world",
    "Hello, IndexTTS! How are you?",
    "I'm fine. It's OK — 3.5% done.",
    "你好世界，今天天气很好。",
    "<|zh|> 你好，欢迎使用语音合成。",
    "<|ja|> こんにちは、元気ですか",
    "IndexTTS 2024年3月5日 test@email.com",
    "The quick brown fox jumps over the lazy dog.",
    "……《》「」『』·—",
]
ok = True
for text in cases:
    ref = official.encode(text, allowed_special="all")
    got = mine.encode(text)
    match = ref == got
    ok &= match
    print(f"{'✅' if match else '❌'} {text[:36]!r:<40} 官方={len(ref)} id  自研={len(got)} id"
          + ("" if match else f"\n     官方: {ref[:20]}...\n     自研: {got[:20]}..."))

# ---- 4) 语言前缀 + 特殊 token 抽验 ----
for t in ["<|zh|>", "<|en|>", "<|0.00|>", "<|30.00|>"]:
    r = official.encode(t, allowed_special="all")
    g = mine.encode(t)
    print(f"{'✅' if r == g else '❌'} 特殊token {t}: 官方={r} 自研={g}")
    ok &= r == g

print("\n" + ("🎉 P1a 通过：自研 BPE 与官方逐 id 一致" if ok else "❌ 存在不一致，见上"))
