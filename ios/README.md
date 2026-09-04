# Huamei IndexTTS · iOS

> Native on-device voice cloning · **Internal Beta 1** · **iOS 17+**（推理内核要求）· Liquid Glass (iOS 26+)

仓库: https://github.com/WebExperiments-glitch/huamei-indexTTS

## ⚠️ 平台变更记录（2026-09-03）

- 最低兼容 **iOS 16 → iOS 17**：MLX Swift（mlx-swift）从 macOS 14 / **iOS 17** 起才可用，
  且 Metal 需真机（Simulator 不支持）。原 iOS 16 目标会导致 Xcode 解析/编译报错。

## 演示栈

- **UI**: SwiftUI · Liquid Glass modifier（iOS 17+，iOS 26 自动增强）
- **Audio**: AVFoundation（录制/播放）+ AVAssetExportSession（MP4 → M4A 提音轨）
- **Engine (Stage 1)**: `MLXIndexTTS2Core`（MLX Swift · 自研算子 · 逻辑逐模块对照 `p0/` Python 数值孪生）

## 主色（白 + 橙）

| Token | Hex | 用途 |
|---|---|---|
| `Theme.Colors.accent`       | `#FF7319` | 主品牌 / 主按钮 |
| `Theme.Colors.accentStrong` | `#FF610D` | 强调文本 / 进度 |
| `Theme.Colors.surfaceAlt`   | `#FFF5EE` | 橙色高亮卡 |
| `Theme.Colors.canvas`       | `#FCFCFD` | 整体背景 |

## 主要交互（三卡 + 一钮）

1. **Voice Reference**（音源 4 入口：①App 内录制 ②音频文件 ③视频文件自动提音轨 ④文件 App）
2. **Text**（多行 + 计数）
3. **Synthesize**（6 段进度 → 完成播放）
   - 设置：Language（zh/en/ja/es/ar）· Duration Factor · 采样超参（与官方 demo 一致）

## 工程结构

```
ios/
├── Package.swift
├── Sources/
│   ├── MLXIndexTTS2Core/            # 推理内核（纯 MLX，无 UI）
│   │   ├── Config.swift             # 常量（与 audit 对齐）
│   │   ├── Ops.swift / OpsCPU.swift # 算子层（含 RoPE CPU 精确实现）
│   │   ├── IO/Safetensors.swift     # safetensors 读取器
│   │   ├── Text/TiktokenBPE.swift   # 60,509 词表（num_languages=99）
│   │   ├── GPT2.swift               # 24 层量化 GPT
│   │   ├── GPTGenerator.swift       # 自回归生成 + 73 组情绪
│   │   ├── Codec.swift              # FVQ→Vocos decode
│   │   ├── S2MelWeights.swift       # s2mel 权重加载
│   │   ├── S2MelInfer.swift         # LR / DiT / WN / CFM
│   │   ├── BigVGAN.swift            # vocoder
│   │   ├── TTSPipeline.swift        # 编排（阶段独占 + 用完即卸）
│   │   └── Voice/                   # P5 设备端克隆骨架（Voiceprint + VoiceExtractor）
│   └── HuameiIndexTTS/              # SwiftUI App（UI / Theme / Audio / Model）
├── Resources/
└── README.md
```

## 内存架构（峰值 ≈ 2GB，为什么）

**设计原则：峰值 = max(各阶段)，不是求和。** 合成前只常驻小资源（tokenizer <10MB + 73 组情绪矩阵 0.5MB）；
每段模型用完即 `releaseX()` 置 nil，让 MLX/Metal 统一内存可回收：

```
① GPT 自回归   载 gpt(1.13GB 8bit) → codes([Int] 仅几KB) → 卸     峰值 ≈ 1.9GB
② Codec        载 codec(0.10GB)    → S_infer            → 卸     峰值 ≈ +0.1GB
③ s2mel 扩散   载 s2mel(0.20GB)    → mel                → 卸     峰值 ≈ +0.3GB
④ BigVGAN      载 bigvgan(0.21GB)  → wav                → 卸     峰值 ≈ +0.3GB
```

对比同类 fp16 全载实现（resident 4.46GB + 瞬时 5GB 叠加 → run peak 9.25GB）：
本作 8bit 原生量化 + 73 组预设免 w2v-bert（省 ~1.4GB）+ 分段卸载 → **理论峰值 ≈ 2GB**。
设备端"随机音频即时克隆"（P5，w2v-bert 组件）同样用**独占会话**：提取完即卸，峰值仍 ≈ 3GB。

## 🛠 first-compile 修正清单（在 Mac + Xcode 上必看）

本工程在 Windows 上编写，**无法本地编译**；MLX Swift API 已尽力贴官方，以下为高概率需要按你锁定的 mlx-swift 版本修正的点。建议 `File → Open Package.swift` 后按 Xcode 报错逐条对照：

| # | 位置 | 内容 | 修正建议 |
|---|---|---|---|
| 1 | `Package.swift` | mlx-swift 版本 | `from: "0.30.0"` → 若 resolve 失败，按最新 tag（如 0.31.x）上调 |
| 2 | `Ops.swift quantizedLinear` | `quantizedMatmul` 签名 | 0.29 起含 `mode:`；若报参错，删 `mode:` 或调整 `transpose:`（python 语义 `x @ wᵀ`） |
| 3 | `OpsCPU.swift asFloatArray` | `self.asArray()` | 若不存在 → 尝试 `Array(self)` 或 `MLXArray.data` 桥接；只在 1 处改 |
| 4 | `Safetensors.uint32Array` | `MLXArray(data:dtype:shape:)` | 若该 init 不存在 → 用 `[UInt32]` init 或 Cmlx 桥接；影响 GPT 量化权重加载 |
| 5 | 所有 `x[0..., i..<j, 0...]` 切片 | MLX Swift 索引 | 已尽量收敛为 Range 切片 / CPU 行读；若个别语法不支持 → 按报错改 reshape 方式 |
| 6 | `Ops.conv1d` | `MLX.conv1d` 参数 | 若为 `conv1d(x, w, stride:, padding:, dilation:)` 形式则逐一对齐 |
| 7 | `GPT2` qkv 分段 | reshape 后 Range 取段 | 已用 [B,T,3*D]→Range 拆；若 shape 广播不符 → 按报错微调 |
| 8 | ~~t_embedder 频率表~~ | ✅ 已关闭 | RoPE 公式已在 p2b 官方对拍验证（cos=1.0），Swift 运行时按标准公式生成，无需 buffer 文件 |
| 9 | ~~finalLayer 占位~~ | ✅ 已修复 | 已实现真实 modulate：LN(no-affine)→SiLU→Linear(512→1024)→(shift,scale)→x*(1+scale)+shift→Linear |
| 10 | `BigVGAN.upsample` | 反卷积用 nearest+conv 近似 | 需换真正 `conv_transpose1d`（MLX 若有）或逐点展开；当前会降低音质 |
| 11 | `CFM` 高斯噪声 | 用均匀近似 | 换 Box-Muller 或 `MLXRandom.normal`（影响扩散种子一致性）|
| 12 | ~~语言表 json~~ | ✅ 已完成 | `langs.json`（106 项）已在 Windows 导出；pipeline 优先读取，ar=13 等语言前缀 bug 同步修复 |

### 已修 bug 记录（静态自查 2026-09-03）

- q/k/v attention 维度：补 `[B,H,T,D]` 转置，修复 scores matmul 维度错误
- KV 增量步 RoPE 用绝对位置（offset），不再用局部位置 0
- `SynthesizeConfig` 移入 core（消除 UI/core 重复定义导致的编译失败）
- GPT 嵌入行采集改 `SafetensorsFile.row` CPU 路径（消除 2D 表 3 切片索引 bug）
- GPTHead 末位置改用 Range 切片（避免 int 索引不确定性）
- s2mel `finalLayer` 由占位改为真实 modulate 实现；attn q/k/v 改 Range 分段
- LiquidGlass `Glass26Enhancer` 形状补 `AnyShape` 包装（类型不匹配）
- BigVGAN resblock kernel 改为从权重形状推断（不依赖 magic 3/7/11）
- TiktokenBPE 反查表一次性缓存（避免每次 decode 重建）
- 模型加载移到后台任务（避免阻塞主线程）

## 模型资源（拷贝到 App 的 Documents/huamei-models/）

✅ 已在 Windows 导出完成（`models/mlx-indextts2-2.5-8bit/`）：
```
feat1.json · feat2.json    # 73 组情绪矩阵（283KB / 1.9MB，feat1.pt/feat2.pt → JSON）
langs.json                 # 完整语言表（106 项，含 lang_dict）—— pipeline 优先读取
```
随 4 个 safetensors + tiktoken + specials.json 一起拷贝即完整。

⏳ 仍待准备（Mac，跑 `scripts/prepare_voicepack.py`）：
```
prompt_<id>.json · refmel_<id>.json · style_<id>.json  # voicepack（A1 条件束）
```
```bash
# Mac 上（脚本每步对应官方源码行号；LR 部分已在 Windows 自检通过）
pip install torch torchaudio librosa scipy transformers pyyaml
PYTHONPATH=reference/index-tts-main \
python scripts/prepare_voicepack.py --wav 参考音频.wav --id my_voice \
    --out voicepacks --model-dir models/mlx-indextts2-2.5-8bit \
    --hf-cache models/hf_cache    # 需先放 w2v-bert-2.0 + campplus_cn_common.bin
# 产出拷到 Documents/huamei-models/ 即可（style 喂 GPT，prompt/refmel 作 PromptBundle）
```
t_freqs.json 不再需要：RoPE 频率表公式已在 p2b 对拍验证（cos=1.0），Swift 运行时生成。

## 已知限制（Stage 1 如实记录）

- 预设音色模式（speakerRow=0），参考音频 UI 入口暂不驱动推理（A1 未接）
- CFM prompt/ref_mel 为零占位时输出的音频为"管线验证级"，非可用音色 —— 接入 A1 条件束后为正式路径
- Simulator 无 Metal → 全部推理验证需真机（≥A17 Pro 推荐 / ≥8GB 内存）
- 冷启动 Metal 首次生成偏慢（shader 编译），随后正常

## 许可

- 引擎权重：IndexTeam IndexTTS-2.5 · **Bilibili Model Use License Agreement**（商用 ≤100M MAU / RMB 1B 营收以下允许；衍生需保留声明）
- 权重转换：vanch007/mlx-indextts2-2.5-8bit · MIT
- 框架：Apple MLX Swift · Apache-2.0
## P5 路线：设备端"随机音频即时克隆"（骨架已就位）

> **A2 主体已落代码（2026-09-04）**：全新自研模块，纯 0 到 1、无第三方实现参照。

### 新增模块（`Sources/MLXIndexTTS2Core/Voice/`）

| 文件 | 职责 |
|---|---|
| `AudioFeatures.swift` | 自研复数 FFT/STFT/mel 滤组/logMel z-score + Seamless 前端([T,160]) + kaldi fbank80 + 22.05k refmel + 线性重采样 |
| `W2VBert.swift` | w2v-bert-2.0 24 层 Conformer 前向（relative-key 注意力、深度卷积模块），返回第 17 索引隐藏态 |
| `Campplus.swift` | CAM++ DTDNN 全链（ResNet head → TDNN → 3×denseCAM 块 → stats pooling → 192 嵌入），含 BatchNorm1d/2d |
| `VoiceExtractor.swift` | 完整提取链（style + prompt[P,512] + refmel[P,80]）+ 二进制声纹束缓存（`voiceprints/*.vpbin`）+ 独占会话卸载 |
| `TTSPipeline.swift`（改） | `synthesize` 新增 `styleOverride`：克隆 style 直接进 GPT 条件 |

### 运行前提（模型组件，按需下载，App 首包不含）

```
Documents/huamei-models/
├── w2v-bert-2.0/model.safetensors   # HF facebook/w2v-bert-2.0（MLX 加载，2.16GB 全精度）
├── campplus_cn_common.safetensors   # 魔搭 iic/speech_campplus_sv_zh-cn_16k-common 转换
├── s2mel.safetensors                # 已有（LR 权重复用）
└── wav2vec2bert_stats.json          # 由 stats.pt 导出（scripts/w2vbert_export.py）
```
Windows 侧一次性准备：`scripts/w2vbert_export.py`（campplus .bin→safetensors + keys 清单 + stats json）、
`scripts/w2vbert_probe.py / _focus / _diag / _fmin / _sweep`（Seamless 前端配方黑盒标定）。

### ⚠️ 两处数值对拍校准点（golden npz 已存 `scripts/golden/`，Mac 端对拍收敛后移除标记）

1. **Seamless 前端双分支 fmin 偏移**：实测 160 通道 = 同一滤组两份近邻副本（corr≈0.96，峰值差 1-2 bin）。
   当前副分支 `fminB=40` 为占位，需对 golden 收敛。
2. **w2v-bert relative_key 相对项**：实现为 query-side 点积（`distance_embedding` [73,64]），
   公式方向/桶距需对官网 by 层输出做数值对拍。

### A2 使用流程（UI 层对接中）

```
UI 解码 → 16k mono PCM → VoiceExtractor.voiceprint(pcm16k:) → VoiceprintBundle（缓存）
合成  → TTSPipeline.synthesize(styleOverride: bundle.style, prompt: extractor.promptBundle(bundle))
```

用户拿任意音频就克隆，**不需要电脑**。内存：提取阶段独占 w2v(~1.2GB F32)/campplus/s2mel，用完即卸，峰值 ≈ max(提取, 合成) ≈ 2-3GB。

```
① 提取（首次，~秒级）   独占载 w2v-bert + campplus + s2mel(LR) → style[192] + prompt[P,512] + refmel[P,80]
② 缓存                voiceprints/<id>.vpbin（二进制，MB 级）→ 同一音频二次克隆 0 等待
③ 合成                进 TTSPipeline（≈2GB）→ 卸载提取组件
总峰值 = max(①,③) ≈ 2-3GB（8GB 机型安全）
```

> 上表为 A2 已实现形态；原"P5 骨架（extractorNotImplemented）"已被真实前向替换。
