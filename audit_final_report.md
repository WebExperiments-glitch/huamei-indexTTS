# IndexTTS-2.5 MLX 权重全量精确审计报告

> 日期：2026-09-03 | 范围：models/mlx-indextts2-2.5-8bit 全部数据 ↔ config/manifest/官方源码 ↔ 自研 p0/ 代码
> 判定标准：逐项程序核对，**数值精确到个位/字节，任何"差不多"一律判 ❌**

---

## 审计结论（先说结果）

- ✅ **发现 1 个真实精度 bug 并已修复**：tokenizer 的 `num_languages` 应为 **99**（原代码误用 100，导致语言表第 100 项之后**全部 1,573 个特殊 token 的 id 整体偏移 +1**）。已按官方三重证据修正并重新逐 id 对拍。
- ✅ 除该 bug 外，**全部数据点精确对齐**（文件字节、张量计数、维度矩阵、词表、dead keys、8bit 量化参数、帧率链）。
- ✅ 修复后全量回归 **8/8 套测试 PASS**。

---

## AUDIT-A 文件级一致性（vs model_manifest.json）

| 组件 | header 张量数 | manifest.saved | mapped | ignored | 磁盘字节 | manifest 字节 | 判定 |
|---|---|---|---|---|---|---|---|
| bigvgan.safetensors | 449 | 449 | 449 | 218 | 224,443,177 | 224,443,177 | ✅ |
| codec.safetensors | 241 | 241 | 241 | 0 | 101,188,975 | 101,188,975 | ✅ |
| gpt.safetensors | 649 | 649 | 457 | 1 | 1,182,278,329 | 1,182,278,329 | ✅ |
| s2mel.safetensors | 264 | 264 | 264 | 1 | 207,320,482 | 207,320,482 | ✅ |

- 4 个文件字节数与张量计数与 manifest **逐字节一致**
- 各文件 `__metadata__`：`{component, format: mlx, model_family: IndexTTS, model_version: 2.5}` ✅
- bigvgan ignored=218 = 抗混叠 sinc 滤波器（18 resblock×12 + activation_post×2），**已全部剥离，header 内 0 个 filter 张量** ✅（这正是 P0b 中"官方去抗混叠后 cos=1.0"的原因）

## AUDIT-B GPT 关键形状矩阵（safetensors header 实测）

| 张量 | 实测形状 | 预期/语义 | 判定 |
|---|---|---|---|
| text_embedding.weight | [60510, 1280] | 60509×1+1（embedding 含 1 行 padding） | ✅ |
| mel_embedding.weight | [8194, 1280] | 8192 codes + start(8192) + stop(8193) | ✅ |
| lang_embedding.weight | [107, 1280] | LANGUAGES 106 + 1 | ✅ |
| spk_emb_proj.weight | [1280, 192] | campplus 192→1280 | ✅ |
| mel_head.weight | [8194, 1280] | → mel logits | ✅ |
| text_pos_embedding.emb.weight | [602, 1280] | 覆盖文本位置上限 | ✅ |
| mel_pos_embedding.emb.weight | [1818, 1280] | 覆盖 mel 生成位置上限(1500+) | ✅ |
| gpt.h.N 层数 | 24 | config num_layers | ✅ |
| gpt.h.0.attn.c_attn.weight | U32 [3840, 320] | 反量化后 [3840, 1280]，q/k/v 各 1280 | ✅ |
| 每层 c_attn/c_proj/c_fc/c_proj | U32+F16 scales+biases | 8bit affine 量化 | ✅ |
| dtype 分布 | U32×96, F16×553 | 96 = 24 层×4 个量化线性层 | ✅ |

## AUDIT-C tokenizer 词表（⚠️ 本审计发现并修复的 bug）

### 三重证据：官方词表 = **60,509**（num_languages=99）

1. `config.yaml` → `number_text_tokens: 60509`
2. `model_manifest.json` → `tokenizer.vocab_size: 60509`
3. 官方 `tokenizer.py get_encoding()` → **默认参数 `num_languages: int = 99`**（全仓推理路径无任何覆盖）

### 原 bug 与影响

- 自研代码原用 `num_languages=100` → specials 1,674 个 → 总词表 60,510
- 实测差异：**1,573 个 token**（语言表第 99 项之后全部，含 `<|yue|>` 与所有情绪/事件/TTS/时间戳）的 id **整体 +1**
- 关键 id 变化：`<|0.00|>` 59009→**59008**，`<|30.00|>` 60509→**60508**
- `yue` 恰为 LANGUAGES **第 100 个**（idx=99）→ 官方 99 版里 `<|yue|>` **不是 special token**（按普通文本编码），100 版错误地把粤语纳入了语言 token 表

### 修复后对拍（官方真值 = tiktoken.Encoding(num_languages=99)）

- 词表 = 58,836 + 1,673 = **60,509**，specials.json 已重新生成（1673 个）
- 14 个关键 special id 与官方逐一相同（`<|zh|>`=58839、`<|en|>`=58838、`<|0.00|>`=59008、`<|30.00|>`=60508、`<|TTS/SP13|>`=59007…）
- 11 种句子（中/英/日/粤/标点/数字/300 长串/语言前缀）**逐 id 一致** ✅
- 推论：text_embedding 第 60,510 行（id 60509）为**从未被使用的 padding 行**（非零 = 初始随机值，官方训练/推理均不触达）

## AUDIT-D BigVGAN 结构矩阵（实测）

| 项目 | 实测 | 判定 |
|---|---|---|
| conv_pre | [1536, 7, 80] + bias | ✅ |
| conv_post | [1, 7, 24]，**无 bias**（use_bias_at_final=False） | ✅ |
| ups 6 级 stride | 4/4/2/2/2/2（k=8/8/4/4/4/4） | ✅ |
| 总上采样 | 4×4×2×2×2×2 = **256 = hop_length** | ✅ |
| resblock kernel | 每 stage [3, 7, 11] 循环（18 块实测 3,7,11,3,7,11…） | ✅ |
| 通道链 | 1536→768→384→192→96→48→24；resblock 通道 [768×3,384×3,192×3,96×3,48×3,24×3] | ✅ |
| SnakeBeta | 108（18 块×6）+ activation_post 1 组，logscale 存储 | ✅ |
| dtype | 全 F16 | ✅ |

## AUDIT-E dead-weights 审计（对照官方 forward 实际使用）

**s2mel（264 张量）**：推理用 252；dead 12 = x_embedder(2) + content_mask_embedder(1) + cond_embedder(1) + gpt_layer(6) + length_regulator.embedding(1) + mask_token(1)——全部为官方 `is_discrete=False` / 2.5 推理路径不使用 ✅

**gpt（649 张量，U32/F16 拆分后）**：
- 推理必需（emo 预设模式）：gpt 主体 482 + embeddings 5 + head/spk_proj 6
- 可跳过（仅 `emo_vec=None`/w2v-bert 路径用）：emo_conditioning_encoder 130 + emo_perceiver_encoder 20 + emo_layer 2 + emovec_layer 2（Σ权重=1 时该分量乘 0）
- 官方推理不用：text_head 2 ✅

**codec（241 张量）**：全部被 decoder/vq/up 使用（P3a 官方对拍 load_state_dict missing=0/unexpected=0 佐证）✅

## AUDIT-F 推理链路数值链

| 环节 | 数值 | 判定 |
|---|---|---|
| GPT 帧率 | 25 fps | ✅ |
| Codec 上采样 | ×2 nearest → 50 fps | ✅ |
| LengthRegulator | ×1.72 → **86 fps = 22050/256** | ✅ |
| CFM | 25 步欧拉 + CFG rate 0.7（infer_v2_5.py:830 写死） | ✅ |
| 输出采样率 | 22050 Hz | ✅ |
| emo_num 求和 | 3+17+2+8+4+5+10+24 = **73 = feat1.pt[73,192] = feat2.pt[73,1280] 行数** | ✅ |
| 8bit 量化 | bits=8, group_size=64, uint8(0..255) 小端×scales+biases | ✅ |
| 位置嵌入上限 | text 602 / mel 1818 覆盖推理范围 | ✅ |

## AUDIT-G 全量回归（修复后 8/8 PASS）

| 测试 | 耗时 | 结果 |
|---|---|---|
| P0a BigVGAN 自洽 | 9s | PASS |
| P0b BigVGAN vs 官方(去抗混叠) | 13s | PASS（cos=1.000000） |
| P1a tokenizer（99 版）| 1s | PASS（逐 id 一致） |
| P1b GPT 24层+8bit vs GPT2Model | 184s | PASS（cos≈1.0） |
| P1c KV cache 自回归 | 24s | PASS（cos≈0.9999999） |
| P2 s2mel 自洽 | 9s | PASS |
| P2b LR/DiT/CFM vs 官方 MyModel | 8s | PASS（cos=1.0/0.999999/1.0） |
| P3a Codec vs EnhancedCodec | 6s | PASS（cos=1.000000） |

---

## 遗留说明（如实记录）

1. `wav2vec2bert_stats.pt`（9,343B）未随仓库下载 —— 该文件仅官方 w2v-bert 路径用（feature 归一化 mean/std），73 组预设 + A1 预计算方案不需要。如需 Mac 端产说话人条件束时再用。
2. 端到端音频质量验证需真实参考条件（w2v-bert/campplus 输出），当前 Windows 环境用预设 style + 占位 prompt 验证的是**数据流正确性**（模块数值均已与官方钉死）。
