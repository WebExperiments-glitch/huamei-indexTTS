"""自研 tiktoken BPE 分词器（IndexTTS2 词表 60,509 + 1 行 embedding padding）

设计：
- mergeable ranks 从 `*.tiktoken` 文件加载（58,836 个 byte→rank，rank 0..58,835）
- 特殊 token（1,673 个）从 `specials.json` 加载（id 58,836..60,508）
- 词表总量 = 58,836 + 1,673 = 60,509 = config.number_text_tokens = manifest.vocab_size
  （text_embedding 60,510 行 = 60,509×1+1，末行是未被使用过的 padding，保持随机初始值）
- 官方 get_encoding 默认 num_languages=99（语言取 LANGUAGES 前 99 个，yue 恰为第 100 个 → 不在表内）
- 编码：regex Unicode 切分（GPT-2 pat_str 公开正则）→ 每段做 byte-level BPE 贪心合并
- BPE 合并用每轮最小 rank pair（同 rank 最左），与 tiktoken 语义一致
"""
import heapq
import json
import os
import regex

# GPT-2 / tiktoken 公开的切分正则（IndexTTS 官方同款）
_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, tiktoken_path: str, specials_path: str):
        self.ranks: dict[bytes, int] = {}
        with open(tiktoken_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok, rank = line.split()
                self.ranks[base64_decode(tok)] = int(rank)
        self.n_base = len(self.ranks)
        with open(specials_path, encoding="utf-8") as f:
            self.special: dict[str, int] = json.load(f)     # {token: id}
        self.special = {k: int(v) for k, v in self.special.items()}
        # 反查表（不含特殊 token）
        self.inverse = {r: b for b, r in self.ranks.items()}
        self.pat = regex.compile(_PAT)
        # 特殊 token 交替正则（按长度降序，避免短前缀抢先）
        if self.special:
            alt = "|".join(regex.escape(t) for t in sorted(self.special, key=len, reverse=True))
            self._special_re = regex.compile(f"({alt})")
        else:
            self._special_re = None

    # ---- 编码 ----
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self._special_re is not None:
            for seg in self._special_re.split(text):
                if not seg:
                    continue
                sid = self.special.get(seg)
                if sid is not None:
                    ids.append(sid)
                else:
                    ids.extend(self._encode_regular(seg))
        else:
            ids.extend(self._encode_regular(text))
        return ids

    def _encode_regular(self, piece: str) -> list[int]:
        ids: list[int] = []
        for m in self.pat.finditer(piece):
            tok = m.group(0)
            if not tok:
                continue
            ids.extend(self._bpe_merge(tok.encode("utf-8")))
        return ids

    def _bpe_merge(self, data: bytes) -> list[int]:
        """byte-level BPE：迭代合并当前 rank 最小的相邻 pair（同 rank 取最左）。

        语义与 tiktoken 一致（贪心最小 rank）；片段短，O(n²) 可接受。
        """
        if len(data) == 1:
            return [self.ranks[data]]
        parts: list[bytes] = [data[i:i + 1] for i in range(len(data))]
        while len(parts) > 1:
            best_rank = None
            best_i = -1
            best_key: bytes | None = None
            for i in range(len(parts) - 1):
                key = parts[i] + parts[i + 1]
                r = self.ranks.get(key)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
                    best_key = key
            if best_key is None:            # 无可合并 pair
                break
            parts[best_i] = best_key
            del parts[best_i + 1]
        return [self.ranks[p] for p in parts]

    # ---- 解码 ----
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        chunks: list[bytes] = []
        for i in ids:
            b = self.inverse.get(i)
            if b is not None:
                chunks.append(b)
        return b"".join(chunks).decode("utf-8", errors="replace")


def base64_decode(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def make_specials(languages: list[str], audio_events: list[str], emotions: list[str],
                  tts_tokens: list[str], num_languages: int = 99,
                  base_id: int = 58836) -> dict[str, int]:
    """按官方 get_encoding 的固定顺序生成 {token: id}。num_languages 默认 99（官方同款）。"""
    specials: list[str] = []
    specials += ["<|endoftext|>", "<|startoftranscript|>"]
    specials += [f"<|{l}|>" for l in languages[:num_languages]]
    specials += [f"<|{a}|>" for a in audio_events]
    specials += [f"<|{e}|>" for e in emotions]
    specials += ["<|translate|>", "<|transcribe|>", "<|startoflm|>",
                 "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]
    specials += [f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)]
    specials += [f"<|{t}|>" for t in tts_tokens]
    specials += [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    return {t: base_id + i for i, t in enumerate(specials)}


def dump_specials_to_json(path: str) -> None:
    """用 AST 从 reference 源码提取数据字典（纯字面量，不执行模块），生成 specials.json。"""
    import ast
    src = open(r"D:/indexTTS 2.5/reference/index-tts-main/indextts/utils/tokenizer.py",
               encoding="utf-8").read()
    tree = ast.parse(src)
    data: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("LANGUAGES", "AUDIO_EVENT", "EMOTION"):
                    data[t.id] = list(ast.literal_eval(node.value).keys())
    tts = ["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"] \
        + [f"TTS/SP{i:02d}" for i in range(1, 14)]
    sp = make_specials(data["LANGUAGES"], data["AUDIO_EVENT"], data["EMOTION"],
                       tts, num_languages=99)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=0)
    print(f"specials.json 已生成: {len(sp)} 个 token → {path}")
