# IndexTTS2 → iOS/iPadOS 原生离线部署：完整实施方案

> 基于 `D:\indexTTS 2.5\models\mlx-indextts2-2.5-8bit\` 的**实测权重**编写（非推测）。
> 所有张量名 / 形状 / dtype 均已从 safetensors header 与 config 实读验证。
> 生成时间：2026-09-02 · 本文取代早期那份基于猜测的 `plan_mlx_swift_indextts2_ios.md`

---

## 0. TL;DR — 最重要的 6 件事

| # | 结论 | 影响 |
|---|---|---|
| 1 | **8bit 是标准 MLX affine 量化**（`U32` 打包 + `scales` + `biases`，group_size=64），不是私有格式 | ✅ MLX Swift 能**直接原生消费**，不用手写反量化。最大风险点解除 |
| 2 | **说话人有 73 组现成预设向量**：`feat1.pt=[73,192]`、`feat2.pt=[73,1280]`，共 430KB | ✅ iOS 端**完全可以不带 w2v-bert(~1GB)** 就能出声 |
| 3 | GPT **没有** mel 条件编码器，只有 `spk_emb_proj(192→1280)` + 情绪编码器 | ✅ 相比 XTTS 血统大幅简化，不需要在端上跑大型音频编码器 |
| 4 | 权重总计 **1.71GB**，跳过 `text_head` 后约 **1.56GB** | ⚠️ 需 iPad Pro / iPhone 15 Pro+，并申请 Increased Memory Limit |
| 5 | **坑点：MLX 卷积权重是 (O, K, I)，PyTorch 是 (O, I, K)** | ⚠️ 从 PyTorch 参考实现移植时**必须转置**，否则静默出错 |
| 6 | 你目前**无 Mac / iPad 在修 / GitHub 登不了** | ⚠️ 无法编译验证。方案改为「先在 Windows 用 Python 数值孪生验证架构理解，再机械翻译成 Swift」 |

---

## 1. 资产盘点（已全部到位 ✅）

路径：`D:\indexTTS 2.5\models\mlx-indextts2-2.5-8bit\`

| 文件 | 字节 | 张量数 | dtype | 说明 |
|---|---:|---:|---|---|
| `gpt.safetensors` | 1,182,278,329 | 649 | F16×553 + **U32×96** | GPT 主模型，**仅 GPT 被量化** |
| `s2mel.safetensors` | 207,320,482 | 264 | F16 | CFM DiT 声学模型 |
| `bigvgan.safetensors` | 224,443,177 | 449 | F16 | 声码器 |
| `codec.safetensors` | 101,188,975 | 241 | F16 | 语义 codec（Vocos 系） |
| `multilingual_zh_ja_yue_char_del.tiktoken` | 907,395 | — | — | 58,836 行，`base64 rank` |
| `feat1.pt` | 57,170 | — | F32 | `spk_matrix` = **[73, 192]** |
| `feat2.pt` | 374,866 | — | F32 | `emo_matrix` = **[73, 1280]** |
| `wav2vec2bert_stats.pt` | 9,343 | — | F32 | 2×[1024]（w2v-bert 特征 mean/std） |
| `config.json` / `config.yaml` | 3,464 / 2,663 | — | — | 内容一致，用 json 版 |
| `model_manifest.json` | 15,387 | — | — | 组件映射 + 量化元信息 |
| `conversion_report.json` | 14,711 | — | — | `status: pass` |

**校验**：4 个 safetensors header 全部 JSON 解析成功（无半截/HTML），元数据 `format=mlx / model_family=IndexTTS / model_version=2.5`，manifest 中 `missing: []`、`unexpected: []`、`status: pass`。

**manifest 明确忽略的张量**（转换时丢弃，Swift 侧不需要）：
- `gpt`: `emo_conditioning_encoder.embed.pos_enc.pe`（正弦位置编码，运行时重算）
- `s2mel`: `cfm.estimator.input_pos`（同上）
- `bigvgan`: 全部 `*.activations.*.downsample.lowpass.filter` / `*.upsample.filter`（AntiAlias 抗混叠滤波器，**已丢弃 → resblock 是纯 conv+snake，不含抗混叠**）

---

## 2. 整体推理图

```
文本 "你好世界"
   │  ① tiktoken BPE 编码（vocab 60509 + 1673 特殊 token）
   ▼
 text_ids [T_text]                     spk_vec[192]  ← feat1.pt 预设 / 预计算
   │  text_embedding[60510,1280]       emo_vec[1024] ← Conformer+Perceiver(参考音频) 
   │  + text_pos_embedding[602,1280]        或 feat2.pt 预设[1280]
   ▼
┌──────────────────────────────────────────────────────────┐
│ ② GPT（UnifiedVoice）24层 d=1280 heads=20   **8bit量化**   │
│    自回归生成 mel-code 序列（vocab 8194，25 fps）           │
│    KV cache 复用                                          │
└──────────────────────────────────────────────────────────┘
   ▼
 mel_codes [T_mel]   (T_mel ≈ 秒数 × 25，上限 1815)
   │
   │  ③ codec 反量化：codebook[8192,8] → out_project(8→1024)
   ▼
 content [T_mel, 1024]
   │  ④ length_regulator：Conv×4(k3)+Conv(k1)，1024→512，上采样到 mel 帧率
   ▼
 content [T_frame, 512] ─┐
                          ├→ ⑤ s2mel CFM-DiT（13层 d=512 heads=8）
 noise [T_frame, 80]  ───┘     + WaveNet final layer
   │                          N 步 ODE 去噪（欧拉/CFM）
   ▼
 mel [80, T_frame]  （24kHz, n_fft=1024, hop=256, n_mels=80）
   │  ⑥ BigVGAN：conv_pre(80→1536) → 6×上采样(总×256) + 18 resblock → conv_post(24→1)
   ▼
 wav @ 24kHz
```

⚠️ **第 ③④ 步的精确衔接是最大未知**（见 §14 待确认清单）。③④ 的具体张量流向必须从 `IndexTeam/IndexTTS-2.5` 参考源码确认。

---

## 3. 组件详解（实测结构）

### 3.1 GPT（UnifiedVoice 血统）

**超参**（config.json → `gpt`）

| 项 | 值 |
|---|---|
| layers / model_dim / heads | 24 / 1280 / 20 → **head_dim = 64** |
| MLP 中间维 | 5120 |
| text vocab / mel codes | 60510 / 8194 |
| 特殊 token | `start_text=0, stop_text=1, start_mel=8192, stop_mel=8193` |
| max_text / max_mel | 600(位置表602) / 1815(位置表1818) |
| mel_length_compression | 1024 |

**每层 20 个张量**（`gpt.h.{0..23}.`，共 480 = 24×20）：

```
gpt.h.N.attn.c_attn      weight U32[3840, 320]  scales F16[3840,20]  biases F16[3840,20]  bias F16[3840]
gpt.h.N.attn.c_proj      weight U32[1280, 320]  scales F16[1280,20]  biases F16[1280,20]  bias F16[1280]
gpt.h.N.mlp.c_fc         weight U32[5120, 320]  scales F16[5120,20]  biases F16[5120,20]  bias F16[5120]
gpt.h.N.mlp.c_proj       weight U32[1280,1280]  scales F16[1280,80]  biases F16[1280,80]  bias F16[1280]
gpt.h.N.ln_1 / ln_2      weight+bias F16[1280]
```

> 形状解读：`c_attn` 逻辑形状 `[3840, 1280]`（QKV 三合一 = 3×1280）。
> 存储为 U32 `[3840, 320]`：1280 / 4 = **320**（每个 uint32 装 4 个 int8）。
> `scales/biases` 第二维 = 1280 / 64 = **20** → 印证 `group_size=64`。
> `mlp.c_proj` 逻辑 `[1280, 5120]` → U32 `[1280, 1280]`（5120/4），scales 第二维 5120/64 = **80** ✓

**嵌入与头**

| 张量 | 形状 | 说明 |
|---|---|---|
| `text_embedding.weight` | [60510, 1280] F16 | 155 MB |
| `text_head.weight/bias` | [60510, 1280] | 155 MB，**推理时可能不需要**（见 §8 优化） |
| `mel_embedding.weight` | [8194, 1280] | 21 MB |
| `mel_head.weight/bias` | [8194, 1280] | 21 MB，必需（自回归出 logits） |
| `text_pos_embedding.emb.weight` | [602, 1280] | |
| `mel_pos_embedding.emb.weight` | [1818, 1280] | |
| `lang_embedding.weight` | [107, 1280] | zh/en/ja/es/ar 等多语言 id |
| `final_norm` / `gpt.ln_f` | [1280]×2 | |

**说话人 / 情绪条件**

| 张量 | 形状 | 作用 |
|---|---|---|
| `spk_emb_proj.weight` | [1280, 192] | 说话人向量 192 → 1280（**192 = s2mel style_encoder.dim**） |
| `emo_layer.weight` | [1280, 1280] | 情绪投影 A |
| `emovec_layer.weight` | [1280, 1024] | 情绪投影 B（**1024 = w2v-bert 维度**） |
| `emo_conditioning_encoder.*` | 130 张量 | Conformer（4 blocks） |
| `emo_perceiver_encoder.*` | 20 张量 | Perceiver（2 blocks） |

**Conformer block（`emo_conditioning_encoder.encoders.N`，31 张量/block）**
```
norm_mha → self_attn(RelPos) → norm_conv → conv_module → norm_ff → feed_forward → norm_final
self_attn: linear_q/k/v [512,512](+bias), linear_out [512,512](+bias),
           linear_pos [512,512](无bias), pos_bias_u/v [4,128]   ← 4头×128 = 512
conv_module: pointwise_conv1[1024,1,512] → depthwise_conv[512,15,1] → pointwise_conv2[512,1,512]
feed_forward: w_1[1024,512], w_2[512,1024]     （linear_units=1024）
```
⚠️ `self_attn` 是 **Transformer-XL 式相对位置注意力**（u/v bias），**不是** RoPE 也**不是**绝对位置。这是最容易实现错的地方。

**Perceiver block（`emo_perceiver_encoder.layers.N`，8 张量/block × 2 layers）**
```
latents [1, 1024]                  ← 只有 1 个 latent，1024 维
layers.N.0: cross-attn  linear_q/k/v [256,1024], linear_out [1024,256]   (4头×64=256)
layers.N.1: GLU-FFN     w_1 [2730,1024] → 2730 = 2×1365 → GLU → w_2 [1024,1365]
norm [1024], proj_context [1024]
```
→ 把情绪参考音频特征压缩成**单个 1024 维向量** → 喂 `emovec_layer(1024→1280)` ✓

---

### 3.2 Semantic Codec（Vocos 系）

**超参**：`codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12, frame_rate=25`

```
down.weight [1024, 3, 1024]      Conv1d(1024 → 1024, k=3)
up.weight   [1024, 3, 1024]      Conv1d(1024 → 1024, k=3)

encoder.0:  embed[384] → convnext×12 → norm → final_layer_norm → encoder.1 Linear(384→1024)
decoder.0:  同结构（对称）
encoder.1.weight [1024, 384]     Linear(384 → 1024)
decoder.1.weight [1024, 384]     Linear(384 → 1024)

convnext block（9 张量 × 12 层 = 108）：
  dwconv.weight [384, 7, 1]      Depthwise Conv1d(384→384, k=7, groups=384)
  norm.weight/bias [384]         LayerNorm
  pwconv1.weight [2048, 384]     Linear(384 → 2048)
  gamma [384]                    layer scale
  pwconv2.weight [384, 2048]     Linear(2048 → 384)
  经典 ConvNeXt：dwconv → norm → pwconv1 → GELU → ×gamma → pwconv2 (+residual)

quantizer.quantizers.0（5 张量，单组 VQ）：
  in_project.weight  [8, 1, 1024]     Conv1d(1024 → 8, k=1)
  codebook.weight    [8192, 8]        8192 个码本，每个 8 维
  out_project.weight [1024, 1, 8]     Conv1d(8 → 1024, k=1)
```

⚠️ 注意 `codebook_size=8192` 与 GPT 的 `number_mel_codes=8194` 的关系：**8192 码本 + start(8192) + stop(8193) = 8194** ✓

---

### 3.3 s2mel（CFM 流匹配声学模型）

**DiT 超参**：`hidden_dim=512, num_heads=8 (head_dim=64), depth=13, in_channels=80(mel bins), block_size=8192, final_layer_type=wavenet, long_skip_connection=true, uvit_skip_connection=true, style_condition=true`

**DiT block（`cfm.estimator.transformer.layers.N`，13 张量 × 13 层 = 169）**
```
attention.wqkv.weight       [1536, 512]   ← QKV 三合一（3×512），无 bias
attention.wo.weight         [512, 512]    无 bias
attention_norm.norm.weight  [512]         RMSNorm/LayerNorm
attention_norm.project_layer.weight [1024, 512] + bias[1024]   ← adaLN，输出 1024 = shift(512)+scale(512)
feed_forward.w1.weight      [1536, 512]   ← SwiGLU 上投影
feed_forward.w3.weight      [1536, 512]   ← SwiGLU 门投影
feed_forward.w2.weight      [512, 1536]   ← 下投影
ffn_norm.norm.weight        [512]
ffn_norm.project_layer.weight [1024, 512] + bias[1024]         ← adaLN
skip_in_linear.weight       [512, 1024]   ← uvit 长跳跃连接（1024→512）
```
→ 标准 **adaLN 调制（只有 shift+scale，无 gate）** + **SwiGLU FFN** + **U-ViT 长跳跃**。

**其他子模块**
| 模块 | 张量 | 说明 |
|---|---|---|
| `x_embedder` | 2 | mel(80) → 512 |
| `conv1` / `conv2` | 2 / 2 | [512] / [80]，输入输出卷积 |
| `t_embedder` / `t_embedder2` | 5 / 5 | 时间步嵌入，`freqs[128]`（正弦表，**运行时重算**） |
| `cond_embedder` | 1 | [1024, 512] |
| `cond_projection` / `cond_x_merge_linear` | 2 / 2 | [512] |
| `content_mask_embedder` | 1 | [1, 512] |
| `skip_linear` / `res_projection` | 2 / 2 | [512]，长跳跃相关 |
| `final_layer` | 4 | adaLN_modulation（layers.1.bias [1024]） |
| `wavenet` | 34 | 最终层 |

**WaveNet final layer（34 张量）**
```
cond_layer.conv  [8192, 1, 512]   Conv1d(512 → 8192, k=1)
                 8192 = 8层 × 2(scale+shift) × 512  ← 每层独立的条件调制
in_layers.0..7   [1024, 5, 512]   Conv1d(512 → 1024, k=5)，1024 = gate+filter 各 512
                                  （8 层 × 2 张量 = 16）
res_skip_layers  （推断 8 层 × 2 = 16，合计 34 ✓）
```
→ 8 层门控 WaveNet（tanh/sigmoid 门），dilation=1（config `wavenet.dilation_rate: 1`）。

**Length Regulator**
```
content_in_proj [512, 1024]       1024(codec hidden) → 512
embedding       [2048, 512]       content_codebook_size=2048
mask_token      [1, 512]
model（18 张量，索引 0..12，2/5/8/11 为空=激活函数）：
  model.0  Conv [512,3,512]     model.1  Norm[512]
  model.3  Conv [512,3,512]     model.4  Norm[512]
  model.6  Conv [512,3,512]     model.7  Norm[512]
  model.9  Conv [512,3,512]     model.10 Norm[512]
  model.12 Conv [512,1,512]     ← 1×1 收尾
sampling_ratios: [1,1,1,1]      （4 个阶段）
```

**gpt_layer（6 张量，小型 MLP）**
```
layers.0.bias [256]  layers.1.bias [128]  layers.2.bias [1024]
→ 疑似 1280(GPT hidden) → 256 → 128 → 1024(codec 空间)
```
⚠️ 作用未知，需从参考源码确认（见 §14）。

---

### 3.4 BigVGAN（nvidia/bigvgan_v2_22khz_80band_256x）

**已从上采样链精确反推出完整拓扑**：

```
conv_pre.weight [1536, 7, 80]      Conv1d(80 → 1536, k=7, pad=3)

6 个上采样阶段（已精确读出）：
  阶段  ups.weight形状        in→out      kernel  stride  padding
  0     [768, 8, 1536]       1536→768      8        4       2
  1     [384, 8,  768]        768→384      8        4       2
  2     [192, 4,  384]        384→192      4        2       1
  3     [ 96, 4,  192]        192→ 96      4        2       1
  4     [ 48, 4,   96]         96→ 48      4        2       1
  5     [ 24, 4,   48]         48→ 24      4        2       1
  总上采样 = 4×4×2×2×2×2 = 256 ✓（hop_length=256，24kHz 完美对齐）

18 个 resblock（6 阶段 × 3）：
  resblocks.0-2  → 768     resblocks.3-5  → 384    resblocks.6-8  → 192
  resblocks.9-11 →  96     resblocks.12-14 → 48    resblocks.15-17 → 24

每个 resblock（24 张量）：
  activations.0..5.act.alpha/beta   [ch]   ← SnakeBeta 激活，每通道可学习参数（12 张量）
  convs1.0/1/2.weight  [ch, 3, ch]  ← Conv1d(ch→ch, k=3, dilation=[1,3,5])
  convs2.0/1/2.weight  [ch, 3, ch]  ← 第二组（12 张量）

activation_post.act.alpha/beta [24]        ← SnakeBeta
conv_post.weight [1, 7, 24]                Conv1d(24 → 1, k=7, pad=3)
```

**SnakeBeta 激活公式**（需与参考核对精确形式）：
```
y = x + (1 / (beta + eps)) * sin(alpha * x)^2
```
⚠️ BigVGAN 常见两种变体 `snake`（只有 alpha）与 `snake_beta`（alpha+beta）。这里 alpha 和 beta **都有** → 是 **snake_beta**。

⚠️ 抗混叠滤波器（AntiAlias）在转换时**已丢弃**（manifest `ignored`），所以 resblock 内没有上下采样，实现更简单 ✓

---

## 4. 8bit 量化格式详解（**最大利好**）

manifest：
```json
"quantization": { "bits": 8, "component": "gpt", "group_size": 64 }
```

实测张量三元组（`gpt.h.N.attn.c_attn`）：
```
weight  U32   [3840, 320]     逻辑 [3840, 1280]，每 uint32 打包 4 个 int8
scales  F16   [3840,  20]     1280 / 64 = 20 组
biases  F16   [3840,  20]     每组零点偏移
bias    F16   [3840]          普通 bias，加到输出上
```

**这完全就是 MLX 标准的 affine 量化布局**：
```
96 个 U32 = 24 层 × 4 个矩阵（c_attn / c_proj / mlp.c_fc / mlp.c_proj）
反量化：w = unpack_int8(weight) * scales + biases
前向：  y = x @ w.T + bias
```

### 落地策略（**推荐 C2：原生量化**）

| 方案 | 做法 | 评价 |
|---|---|---|
| ~~C1 反量化 F16~~ | 加载时手动 unpack → F16 → 普通 matmul | ❌ 白扔掉一半内存优势（472MB→944MB），且更慢。当初担心的"格式不匹配"已被证伪 |
| **C2 原生量化** ✅ | 直接用 MLX Swift 的量化 matmul | ✅ 零转换、最快、最省内存 |

MLX Swift 侧：
```swift
// 底层原语（具体签名按你用的 swift-mlx 版本核对）
// MLX.quantizedMatmul / MLXFast 中的量化算子
let y = quantizedMatmul(x, weight /*U32*/, scales: scales, biases: biases,
                        bits: 8, groupSize: 64, transpose: true)
```
⚠️ **风险缓解**：API 签名可能因 swift-mlx 版本而异。做法——封装一层 `QuantLinear` 协议/结构体，内部先试原生 API，不通则回退到 `dequantize + matmul`。**只需改 1 个文件**。

**自检方法**（P0 必做）：反量化任一矩阵，检查数值统计量是否合理（GPT 权重 std 应 ≈ 0.02 量级，均值≈0）。若出现 std=50 之类的离谱值，说明字节序/有符号性搞反了。

---

## 5. MLX ↔ PyTorch 移植陷阱清单 ⚠️

从 PyTorch 参考实现（IndexTeam/IndexTTS-2.5）读代码、写 Swift 时，**每一条都要注意**：

| # | 项目 | MLX | PyTorch | 处理 |
|---|---|---|---|---|
| 1 | **Conv1d 权重** | `(O, K, I)` | `(O, I/groups, K)` | **必须转置** `permute(0,2,1)` |
| 2 | **ConvTranspose1d 权重** | `(O, K, I)`（已由 bias 维度验证） | `(I, O/groups, K)` | **必须转置** |
| 3 | Linear 权重 | `(O, I)` | `(O, I)` | 一致 ✓ |
| 4 | Depthwise conv | `(ch, K, 1)` | `(ch, 1, K)` | 转置 |
| 5 | **Conv padding** | 需显式指定 | `padding=` 参数 | MLX conv 也可用 padding 参数，逐个核对 |
| 6 | 正弦位置表 | 运行时重算（`pe`/`input_pos` 被 ignored） | checkpoint 里存 | **自己实现 sinusoidal** |
| 7 | LayerNorm eps | 需确认（常见 1e-5 / 1e-6） | config 里 | 从参考源码确认 |
| 8 | 注意力 scale | `1/sqrt(head_dim)` | 同 | ✓ |
| 9 | GELU 版本 | exact vs tanh 近似 | 通常 exact | 从参考源码确认（ConvNeXt 多用 exact） |

**最危险的是 #1/#2**：转置错了**不会报错**，只会输出噪声。必须在 P1 用数值对拍抓出来。

---

## 6. 分词器（tiktoken）✅ 已核实（见 §15.1）

- 格式：每行 `base64(token_bytes) rank`，**58,836 行**（rank 0 .. 58,835）
- 例：`IQ== 0`（`!` → 0）、`Ig== 1`、`Iw== 2`
- **特殊 token 1,674 个**，id 从 **58,836** 起顺序追加
- 实际词表 **60,510** = 58,836 + 1,674，与 `text_embedding[60510,1280]` 吻合
  （已用数据验证：末尾 id 60509 那行**不是全零**，absMax=0.0105，是训练过的真实权重）

构造顺序（**必须严格按此顺序**，源码 `utils/tokenizer.py::get_encoding`）：
```
<|endoftext|>, <|startoftranscript|>,
<|lang|> × 100            ← LANGUAGES 前 100 个（默认参数写 99，但数据证明是 100）
AUDIO_EVENT × 11,
EMOTION × 4,
<|translate|>, <|transcribe|>, <|startoflm|>, <|startofprev|>, <|nospeech|>, <|notimestamps|>,
<|SPECIAL_TOKEN_1|> .. <|SPECIAL_TOKEN_30|>   (30 个, ASR 用),
TTS_Vocal_Token × 20      (TTS/B,O,Q,A,CO,CL,H + SP01..SP13),
<|0.00|> .. <|30.00|> × 1501                 ← 时长 token，占大头
```
→ `<|en|>` = **58,838**；`<|zh|>` = **58,839**；`<|0.00|>` = **59,009**

**Swift 实现**：
```swift
// 1) 解析 tiktoken 文件 58,836 行 → [bytes: rank]
// 2) 按上面顺序追加 1,674 个特殊 token（id 从 58,836 递增）
// 3) BPE 用标准 GPT-2 byte-pair merge，正则：
//    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// 4) 推理前必须加语言前缀：lang_prefix = "<|zh|> " 再编码
```

---

## 7. 说话人 / 情绪策略（**73 组预设，关键利好**）

实测：
```
feat1.pt (spk_matrix) = [73, 192]  F32   56,064 B ✓  (73 × 192 × 4)
feat2.pt (emo_matrix) = [73, 1280] F32  373,760 B ✓  (73 × 1280 × 4)
73 = sum(config.emo_num) = 3+17+2+8+4+5+10+24 ✓  （8 类情绪，各类样本数）
wav2vec2bert_stats.pt = 2 × [1024] F32  （mean/std，用于归一化 w2v-bert 特征）
```

### 两档方案

**Tier 1 — 内置 73 组预设（推荐先做，零额外开销）**
- 把 `[73,192]` 和 `[73,1280]` 转成原始二进制（430KB）打进 App 包
- UI 让用户选「情绪类别 + 样本序号」→ 直接取向量
- **完全不需要 w2v-bert / campplus**，iOS 包体不增加 1GB
- ✅ 今天就能跑通全链路

**Tier 2 — 自定义音色克隆（后续）**
- 需要 w2v-bert-2.0（~580M 参数）+ campplus 从参考音频提取特征
- **你没有 Mac → 不能在端上预计算，但可以在 Windows 上做**：
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  # 用 IndexTeam/IndexTTS-2.5 的 speaker 提取脚本处理你的参考音频
  # → 导出 spk_vec[192] + emo_vec[1280] → 存成 .bin / .json
  ```
- App 侧支持「导入音色向量文件」

⚠️ 注意：你系统 Python 3.13 目前**没有装 torch**（已验证 `import torch` → ModuleNotFoundError），需要补装 CPU 版。

---

## 8. 内存与算力预算

### 内存（权重常驻）

| 组件 | 磁盘 | 备注 |
|---|---:|---|
| GPT | 1,182 MB | 量化 transformer ≈472MB + scales/biases ≈30MB + **嵌入表 ≈352MB** + 情绪编码器 |
| BigVGAN | 224 MB | |
| s2mel | 207 MB | DiT ≈129MB + WaveNet ≈78MB |
| Codec | 101 MB | |
| **合计** | **1,714 MB** | |

**优化点**：
- `text_head [60510,1280]` = **155 MB** — 自回归只生成 mel token，**text_head 推理时不需要** → 可跳过加载 → **降至 ~1,559 MB**
- 若 `text_embedding` 与 `text_head` 权重共享（tie），更省（需确认）
- 位置表/语言表可忽略不计

### 运行期峰值
```
权重常驻        ~1,559 MB
GPT KV cache    24层 × 2(K,V) × T × 1280 × 2B
                 T=1815(满) → 223 MB ／ T=350(10秒语音) → 43 MB
DiT 激活        937帧 × 512 × 13层，注意力矩阵 937²×8头 → 峰值数十 MB
BigVGAN 激活    最大张量 ≈ 937 × 1536 → 小；末段 240,000 × 24
────────────────────────────────────
峰值估算        ≈ 1.9 – 2.3 GB（10 秒语音，典型场景）
```
**设备要求**：
- iPad Pro (M 系, 8–16GB)：✅ 舒适
- iPhone 15/16 Pro (8GB)：⚠️ 可行但需 `Increased Memory Limit`  entitlement + 谨慎设置 MLX `cacheLimit`
- 建议首测平台：**iPad Pro**

### 速度（粗略估算，需实测）
| 阶段 | 10 秒语音的耗时估算 | 瓶颈 |
|---|---|---|
| GPT 自回归（250 步） | 2.5 – 4 s | **内存带宽**（每步要读 472MB 权重） |
| s2mel CFM（≈20 步 ODE） | 2 – 3 s | 算力（每步 13 层 DiT + WaveNet） |
| BigVGAN | 0.1 – 0.3 s | 算力 |
| **合计** | **≈ 5 – 7 s（RTF 0.5–0.7）** | |

> iPad Pro M 系预计 2–3× 更快（RTF 0.2–0.3）。
> 可通过减少 ODE 步数（20→10）进一步加速，代价是音质略降。

---

## 9. "没有 Mac" 的现实约束与验证策略 ⚠️

你的现状：**iPad 在修 / 无 Mac / GitHub 登不了 → 无法编译、无法运行、无法用 Mac 官方工具做 ground truth**。

这三件事**都不阻塞推进**，改用下面的路径：

### 核心策略：Windows 上的 Python「数值孪生」

```
Stage A（现在就能做，Windows）
  ├─ 用 Python 直接读 MLX safetensors（safetensors 格式跨平台，无需 MLX）
  ├─ 实现 GPT / codec / s2mel / BigVGAN 的 numpy 前向
  ├─ 装 CPU 版 torch → 跑 IndexTeam/IndexTTS-2.5 官方 PyTorch 推理
  └─ 逐模块数值对拍 → 证明"架构理解正确" + 产出全套中间张量 .npy 作为黄金标准

Stage B（等你拿到 Mac / iPad 修好 / GitHub 能登）
  ├─ 把 Stage A 验证过的算法机械翻译成 Swift + MLX Swift API
  ├─ 编译运行，加载 Stage A 的 .npy 输入
  └─ 逐层比对 Swift 输出 vs .npy → 只剩 API 层面的 bug 需要修
```

**为什么这样最稳**：
- 架构理解是**最大的风险**（s2mel/Codec 内部未公开文档）。Stage A 在 Windows 上就能彻底解决它。
- Swift 翻译是**机械劳动**，且 MLX Swift 与 MLX Python 语义一致（同一 C++ 核心），翻译出错概率远低于架构猜错。
- Stage A 产出的 `.npy` 黄金标准，在 Stage B 第一次编译成功时就能立刻验证。

### 立即可做的准备工作（不依赖 Mac）
1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`（Windows CPU 版）
2. 拉取 `IndexTeam/IndexTTS-2.5` 源码，读通 `inference` 与模型定义
3. 把 `.pt` 文件（feat1/feat2/wav2vec2bert_stats）转成原始二进制

---

## 10. 工程架构（Swift 包结构）

```
IndexTTSKit/                          ← Swift Package（纯推理，无 UI）
├─ Package.swift                      依赖 mlx-swift
└─ Sources/IndexTTSKit/
   ├─ IO/
   │   ├─ Safetensors.swift           safetensors 解析（mmap，避免全量读入）
   │   ├─ Quantization.swift          ⚠️ 量化 matmul 抽象层（原生/回退双实现）
   │   └─ ModelBundle.swift           模型目录定位 + 懒加载
   ├─ Text/
   │   ├─ Tokenizer.swift             tiktoken BPE 编解码（60509+1673）
   │   └─ TextNormalize.swift         中文文本规范化
   ├─ Modules/
   │   ├─ RMSNorm.swift  LayerNorm.swift  Conv1d.swift  ConvTranspose1d.swift
   │   ├─ SnakeBeta.swift             BigVGAN 激活
   │   ├─ RotaryNone.swift            （本模型不用 RoPE）
   │   └─ Attention.swift             含 RelPos 变体
   ├─ Models/
   │   ├─ GPT.swift                   24层 + KV cache + 自回归采样
   │   ├─ Conditioning.swift          Conformer + Perceiver（情绪）
   │   ├─ SemanticCodec.swift
   │   ├─ S2Mel.swift                 LengthRegulator + DiT + WaveNet + CFM ODE
   │   └─ BigVGAN.swift
   ├─ Speaker/
   │   ├─ PresetVoices.swift          73 组内置预设
   │   └─ VoiceVector.swift           导入/导出自定义音色
   └─ Pipeline/
       └─ IndexTTS.swift              端到端编排 + 流式回调

IndexTTSApp/                          ← SwiftUI App（iOS 26+，液态玻璃）
├─ App/IndexTTSApp.swift
├─ Views/
│   ├─ ContentView.swift              主界面
│   ├─ VoicePickerView.swift          73 组预设选择（网格）
│   ├─ GenerationView.swift           进度 / 波形 / 播放
│   └─ Components/
│       ├─ GlassCard.swift            玻璃卡片容器
│       ├─ GlassButton.swift          .buttonStyle(.glass)
│       └─ WaveformView.swift
├─ ViewModels/
│   └─ TTSViewModel.swift             @Observable (iOS 17+)
└─ Resources/
    └─ Models/                        gpt/codec/s2mel/bigvgan + tiktoken + presets.bin
```

**UI 要点（用上你调研的液态玻璃）**：
- `.glassEffect(.regular, in: .rect(cornerRadius: 20))` 用在浮动控件、工具栏、播放条
- 多个玻璃元素放同一 `GlassEffectContainer` 内（避免接缝）
- **不要**在主内容/长文本上铺玻璃（可读性）
- 全部用 `if #available(iOS 26, *)` 门控
- ⚠️ 别用 `.clipped()`（会废掉玻璃效果）；`Form` 内玻璃是 no-op

---

## 11. 分阶段路线图

### P0 — Windows 数值孪生（**现在就做，不需要 Mac**）🎯
1. Python 读 4 个 safetensors（含 GPT 反量化）
2. 实现 BigVGAN 前向（最独立、最容易验证）
3. 装 CPU torch，跑官方 PyTorch 推理，产出 **golden `.npy` 全集**
4. BigVGAN 对拍通过 → 证明「权重加载 + 转置规则 + SnakeBeta」理解正确

**验收**：Swift 侧 mel→wav 的输出，与 golden wav 的误差 < 1e-3

### P1 — Python 孪生补全 GPT + 采样
1. 反量化 + 24 层 Transformer + KV cache
2. Conformer(RelPos) + Perceiver 情绪编码器
3. tiktoken 完整 vocab（**含 1673 特殊 token**）
4. 自回归采样（top-k / top-p / temperature / 重复惩罚）
5. 73 组预设向量接入

**验收**：生成的 mel-code 序列与官方一致（或高度接近）

### P2 — Python 孪生搞定 s2mel（**最凶险**）
1. LengthRegulator
2. DiT（adaLN + SwiGLU + U-ViT skip）
3. WaveNet final layer
4. CFM ODE 采样器（步数/调度需从源码确认）

**验收**：mel 谱与官方对拍；主观听感正常

### P3 — codec 与衔接（**最大未知**）
1. Semantic codec 编解码
2. **GPT 输出 → codec → s2mel 的确切衔接**（§14 核心问题）

**验收**：端到端在 Python 上产出可懂语音

### P4 — Swift 工程骨架
1. Xcode 工程 + swift-mlx 依赖
2. safetensors mmap 解析
3. 量化 matmul 抽象层 + 自检
4. 模型加载 + 内存预算控制

**验收**：能在设备上加载全部权重不 OOM

### P5 — Swift 移植（按 P0→P3 顺序）
逐模块翻译，每完成一个就拿 golden `.npy` 对拍

### P6 — App 与 UI
液态玻璃界面、73 组音色选择、生成/播放/导出

### P7 — 优化
- 减少 ODE 步数提速
- ANE（Apple Neural Engine）加速路径
- 流式生成

---

## 12. 风险清单（按严重度）

| 级别 | 风险 | 缓解 |
|---|---|---|
| 🔴 高 | **1673 个特殊 token 未知** → 文本编码全错 | 从 IndexTTS-2.5 源码找；或对比 tiktoken 文件与 embedding 行数 |
| 🔴 高 | **GPT→codec→s2mel 衔接未知** | 必读参考源码；Python 孪生先行验证 |
| 🔴 高 | **CFM ODE 采样调度未知**（步数、solver、noise schedule） | 从源码确认；先用 20 步欧拉试，能出声再优化 |
| 🟠 中 | 卷积权重转置搞反 → 静默噪声 | P0 对拍必抓；逐个模块验证 |
| 🟠 中 | RelPos 注意力实现错 | 单独单元测试，与官方 attention 输出对拍 |
| 🟠 中 | SnakeBeta 公式变体 | 对比 BigVGAN 官方实现两种变体 |
| 🟠 中 | iPhone 8GB 内存吃紧 | 首测用 iPad Pro；跳过 text_head 省 155MB；设 cacheLimit |
| 🟡 低 | MLX Swift 量化 API 版本差异 | `QuantLinear` 抽象层 + 回退实现 |
| 🟡 低 | LayerNorm eps / GELU 变体 | 从源码确认 |
| 🟡 低 | 无 Mac 无法编译 | Python 孪生先行；Mac 到位后一次性验证 |

---

## 13. 阻塞项确认状态（2026-09-02：红色项已全部核实，详见 §15）

源码已解压到 `D:\indexTTS 2.5\reference\index-tts-main\`（`index-tts/index-tts` 仓库 main 分支）。
下方 1–4 项 **已解决**；5–10 项是数值细节，不会导致结构错误，P1/P2 对拍时顺带确认。

1. 🔴 **1673 个特殊 token 从哪来**（tiktoken 文件 58,836 行 vs vocab 60,509）
2. 🔴 **GPT mel-code 输出 → s2mel 输入的完整张量流向**（codec 的哪个部分参与？`gpt_layer` 是什么？）
3. 🔴 **CFM 采样细节**：ODE 步数、solver（欧拉/RK4）、noise schedule、`inference_cfg_rate`
4. 🟠 **LengthRegulator 的 `sampling_ratios=[1,1,1,1]` 与 `mask_token` 语义**（是否上采样？上采样多少倍？）
5. 🟠 **adaLN 的 1024 维输出如何拆分**（shift/scale 各 512？有无 gate？）
6. 🟠 **LayerNorm eps**、**GELU 变体**（exact / tanh）
7. 🟡 Conformer 的 macaron 结构与 `ff_scale`（通常 0.5）
8. 🟡 `emo_num=[3,17,2,8,4,5,10,24]` 对应的 8 类情绪名称
9. 🟡 BigVGAN SnakeBeta 的精确公式与 eps
10. 🟡 `wav2vec2bert_stats.pt`（mean/std）在推理哪一步使用

---

## 14. 下一步建议

**最务实的启动动作（今天就能做，不需要 Mac）**：

```powershell
# 1. 装 CPU 版 torch（你系统 Python 目前没装）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 拉官方参考源码（看 §13 那 10 个问题）
#    https://github.com/IndexTeam/IndexTTS-2.5  (rev d0aa86e)
```

然后我开始 **P0：Windows Python 数值孪生** —— 先写 BigVGAN 的 numpy 前向并对拍。这一步能一次性验证「safetensors 加载 + 卷积转置规则 + SnakeBeta + 上采样拓扑」这四件事，是整个项目风险最低、收益最高的起点。

---

## 15. 参考源码核实结果（2026-09-02）

源码位置：`D:\indexTTS 2.5\reference\index-tts-main\`（`index-tts/index-tts` 仓库 main 分支，35.9MB zip）

### 15.1 分词器：1,674 个特殊 token ✅

源：`indextts/utils/tokenizer.py::get_encoding()`（第 180–218 行）

```python
n_vocab = len(ranks)      # 58,836
specials = [
    "<|endoftext|>", "<|startoftranscript|>",                    #    2
    *[f"<|{lang}|>" for lang in LANGUAGES.keys()[:num_languages]],  # 100 ← 数据证明是100
    *[f"<|{a}|>" for a in AUDIO_EVENT.keys()],                   #   11
    *[f"<|{e}|>" for e in EMOTION.keys()],                       #    4
    "<|translate|>", "<|transcribe|>", "<|startoflm|>",
    "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",       #    6
    *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],           #   30 (ASR)
    *[f"<|{t}|>" for t in TTS_Vocal_Token.keys()],               #   20 (7 + SP01..SP13)
    *[f"<|{i*0.02:.2f}|>" for i in range(1501)],                 # 1501 ← 时长 token
]                                                                # 合计 1674
for token in specials:
    special_tokens[token] = n_vocab; n_vocab += 1
```

**关键 id 表**（实现时直接用）：

| token | id |
|---|---|
| `<|endoftext|>` | 58,836 |
| `<|startoftranscript|>` | 58,837 |
| `<|en|>`（LANGUAGES[0]） | 58,838 |
| `<|zh|>`（LANGUAGES[1]） | **58,839** |
| 语言 token 段 | 58,838 – 58,937 |
| AUDIO_EVENT 段 | 58,938 – 58,948 |
| EMOTION 段 | 58,949 – 58,952 |
| 任务 token 段 | 58,953 – 58,958 |
| SPECIAL_TOKEN_1..30 | 58,959 – 58,988 |
| TTS_Vocal_Token | 58,989 – 59,008 |
| `<|0.00|>` … `<|30.00|>` | **59,009 – 60,509** |

> ⚠️ `num_languages` 代码默认写 **99**，但那样总数是 60,509。
> 数据验证：`text_embedding.weight[60509]`（末行）**absMax=0.0105，非全零**
> （相邻行 0.0092，同量级）→ 该行是训练过的 → **实际 vocab = 60,510 → num_languages = 100**。
> 影响：时长 token 起始 id 差 1，其余（语言/情绪等）不受影响。

### 15.2 端到端推理流程 ✅（最核心）

源：`indextts/infer_v2_5.py::infer_generator()` 第 620–853 行

```python
# ---------- A. 参考音频预处理（只需算一次，可缓存）----------
spk_cond_emb = self.get_emb(input_features, attention_mask)   # w2v-bert 特征 [1,T,1024]
style        = self.campplus_model(feat)                      # campplus → [1,192]
ref_mel      = self.mel_fn(audio_22k)                         # 参考音频 mel
prompt_condition = s2mel.models['length_regulator'](
                      spk_cond_emb, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]

# ---------- B. 文本 → GPT 自回归 ----------
emovec = self.gpt.merge_emovec(spk_cond_emb, emo_cond_emb, ..., alpha=emo_alpha)
codes, speech_conditioning_latent = self.gpt.inference_speech(
    spk_cond_emb, text_tokens, lang, emo_cond_emb,
    emo_vec=emovec, campplus_embedding=style,
    do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
    repetition_penalty=10.0, num_beams=3, length_penalty=0.0,
    max_generate_length=1500)

# ---------- C. codes → mel → wav（★关键段，第 830-853 行）----------
diffusion_steps    = 25        # ← CFM ODE 步数
inference_cfg_rate = 0.7       # ← CFG 强度

S_infer = self.semantic_codec.decode(codes)      # codes → [B, T*2, 1024]（内部 ×2 上采样）
target_lengths = [int(S_infer.shape[1] * 1.72 * duration_factor)]   # ← ×1.72
cond = s2mel.models['length_regulator'](S_infer, ylens=target_lengths,
                                        n_quantizers=3, f0=None)[0]
cat_condition = torch.cat([prompt_condition, cond], dim=1)   # 参考段拼在前面
vc_target = s2mel.models['cfm'].inference(cat_condition,
                                          [cat_condition.size(1)],
                                          ref_mel, style, None,
                                          diffusion_steps,
                                          inference_cfg_rate=inference_cfg_rate)
vc_target = vc_target[:, :, ref_mel.size(-1):]    # ← 切掉参考音频那一段
wav = self.bigvgan(vc_target.float())             # mel → 波形
wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
```

**帧率链（完全闭环）**：
```
GPT codes  25 fps
  ↓ semantic_codec.decode(): F.interpolate(scale_factor=2, mode="nearest")
 50 fps
  ↓ length_regulator(): F.interpolate(size=ylens.max()) 其中 ylens = T*1.72
 86 fps  = 22050 / 256 (hop_length) ✓
```

⚠️ **采样率是 22050 不是 24000**！
`infer_generator` 内 `sampling_rate = 22050`，且 BigVGAN 是 `bigvgan_v2_22khz_80band`。
（`config.yaml` 里 `dataset.sample_rate: 24000` 是训练数据口径，推理输出以代码为准。）

### 15.3 LengthRegulator ✅

源：`indextts/s2mel/modules/length_regulator.py::InterpolateRegulator.forward()`（第 90–141 行）

```python
def forward(self, x, ylens=None, n_quantizers=None, f0=None):
    x = self.content_in_proj(x)                       # Linear(1024 → 512)，is_discrete=False
    mask = sequence_mask(ylens).unsqueeze(-1)
    if self.interpolate:                              # sampling_ratios 非空即为 True
        x = F.interpolate(x.transpose(1,2).contiguous(),
                          size=ylens.max(),           # ← 直接插值到目标长度（×1.72）
                          mode='nearest')             # ← 最近邻，不是线性！
    out = self.model(x).transpose(1,2).contiguous()   # 卷积堆栈
    return out * mask, ylens, None, None, None
```

`self.model` 结构（`sampling_ratios=[1,1,1,1]` → 4 组 + 收尾）：
```
[Conv1d(512,512,k=3,pad=1), GroupNorm(1,512), Mish]   ×4
 Conv1d(512,512,k=1)
```
对应权重索引 `model.0 / 3 / 6 / 9` = conv，`1 / 4 / 7 / 10` = norm，`2 / 5 / 8 / 11` = 激活（无参数），`12` = 1×1 conv ✓

⚠️ **两个易错点**：
1. 归一化是 **GroupNorm(groups=1, channels=512)**，不是 LayerNorm
2. 激活是 **Mish**，不是 SiLU / ReLU
3. `embedding[2048,512]` 与 `mask_token` 在 `is_discrete=False` 时**不使用**（可跳过加载）

### 15.4 Semantic Codec ✅

源：`indextts/codec/models.py::EnhancedCodec.decode()`（第 205–231 行）

```python
def decode(self, codes):                      # codes: [B, T]
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)            # → [1, B, T]
    quantized_out = self.quantizer.vq2emb(codes)   # 码本查表 → [B, T, 1024]
    x = self.decoder(quantized_out)                # VocosBackbone + Linear(384→1024)
    if self.downsample_scale > 1:                  # = 2
        x = x.transpose(1, 2)                      # → [B, 1024, T]
        x = F.interpolate(x, scale_factor=2, mode="nearest")   # ★ ×2 最近邻
        x_rec = self.up(x).transpose(1, 2)         # up = Conv1d(1024,1024,k=3,stride=1,pad=1)
    return x_rec
```

- `downsample_scale = 2`（构造函数默认），`decoder` = `Sequential(VocosBackbone(input_channels=1024, dim=384, intermediate_dim=2048, num_layers=12), Linear(384, 1024))`
- 与权重对应：`decoder.0.convnext.*`（12 层 ConvNeXt）+ `decoder.1`（Linear 384→1024）✓
- `quantizer` = ResidualVQ，`num_quantizers=1`，`use_l2_normlize=True`（量化时做 L2 归一化）
- **encoder 侧与 `down`(stride=2) 在推理时不用** → 可只加载 decoder + quantizer，省内存

### 15.5 说话人 / 情绪向量的真实用法 ✅

源：`infer_v2_5.py` 第 620–680 行

```python
# style = campplus 输出 [1,192]
if emo_vector is not None:                       # emo_vector = 8 个权重（对应 8 类情绪）
    random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
    #   ↑ 在 spk_matrix 的"每一类"里，找与当前说话人 style 最相似的那一条
    emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
    emo_matrix = torch.cat(emo_matrix, 0)        # [8, 1280]
    emovec_mat = weight_vector.unsqueeze(1) * emo_matrix   # 8 个权重逐维加权
    emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)     # → [1, 1280]
```

**含义确认**：`spk_matrix` / `emo_matrix` 按 `emo_num=[3,17,2,8,4,5,10,24]` 分成 **8 组**（组大小 3/17/2/8/4/5/10/24，合计 73）。
每组用余弦相似度挑 1 条，用 8 维权向量加权求和 → 1 个 1280 维情绪向量。

→ **iOS 端实现极简**：73×192 + 73×1280 两个小矩阵（430KB）打进包，
   用 `emo_vector`（8 个 0~1 权重，可由 UI 滑块控制）直接算，无需 w2v-bert。

### 15.6 其他确认项

| 项 | 值 | 源 |
|---|---|---|
| CFM ODE 步数 | **25** | infer_v2_5.py:830 |
| CFG 强度 | **0.7** | infer_v2_5.py:831 |
| 输出采样率 | **22050**（非 24000） | infer_v2_5.py:741 |
| GPT top_p / top_k / temperature | 0.8 / 30 / 0.8 | infer_v2_5.py:733-735 |
| repetition_penalty / num_beams / length_penalty | 10.0 / 3 / 0.0 | infer_v2_5.py:737-739 |
| max_mel_tokens | 1500 | infer_v2_5.py:740 |
| 文本分句上限 | 120 token/段 | infer_v2_5.py:573 |
| 语言前缀 | `f"<\|{lang}\|> "` | infer_v2_5.py:700 |
| 时长因子 | `duration_factor`（默认 1.0），乘在 1.72 上 | infer_v2_5.py:833 |
| wav 后处理 | `clamp(32767*wav)` → int16 | infer_v2_5.py:855 |

---

## 16. 下一步（源码已就位，阻塞项已清）

红色阻塞项全部解决，架构理解已无重大盲区。**可以立刻开始 P0**：

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`（Windows CPU 版）
2. 写 **BigVGAN 的 numpy 前向**（读 `bigvgan.safetensors`），验证：
   safetensors 加载 / `(O,K,I)` 转置规则 / SnakeBeta / 6 级上采样拓扑 —— 四件事一次验证
3. 用官方 PyTorch 跑同一段 mel，逐层对拍

BigVGAN 最独立、结构已完全确定（§3.4），是风险最低收益最高的起点。
