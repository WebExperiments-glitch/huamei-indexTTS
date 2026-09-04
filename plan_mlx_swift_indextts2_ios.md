# 方案：IndexTTS2 原生离线 MLX Swift 部署（iPhone / iPad）

> 状态：规划稿（未开始写代码）
> 时间：2026-09-01
> 主设备坏、不编译；本方案先定架构与边界，待你确认决策 + 补齐模型后我动手写 Swift。

---

## 0. 模型完整性核查（先答你的问题）

**结论：不够。** 目前 workspace `D:\indexTTS 2.5` 里只有 `bigvgan.safetensors`（224MB，已验证为正确 MLX/F16/v2.5 格式）。缺其余全部权重与配置。

| 文件 | 状态 | 用途 |
|---|---|---|
| bigvgan.safetensors | ✅ 已就位 | 声码器 mel→波形 |
| gpt.safetensors | ❌ 缺失 | 自回归语言模型（声学/mel token 生成） |
| codec.safetensors | ❌ 缺失 | 声学 codec（VQ / 语义↔声学） |
| s2mel.safetensors | ❌ 缺失 | 扩散解码器（token→mel） |
| multilingual_zh_ja_yue_char_del.tiktoken | ❌ 缺失 | 多语言分词器 |
| config.yaml / config.json | ❌ 缺失 | 模型超参（层数/维度/头数） |
| model_manifest.json | ❌ 缺失 | 张量名↔原 torch 名映射（**重建推理图必需**） |
| conversion_report.json | ❌ 缺失 | 量化/转换校验信息 |
| reference preprocessing statistics/features | ❌ 缺失 | 说话人预处理统计量 |
| LICENSE | ❌ 缺失 | bilibili 衍生授权（商用阈值） |

**你需要下载的（手动，放 `D:\indexTTS 2.5\models\mlx-indextts2-2.5-8bit\`）：**
```bash
# HuggingFace（你习惯手动下，这里只给命令，我不执行）
hf download vanch007/mlx-indextts2-2.5-8bit --local-dir "D:/indexTTS 2.5/models/mlx-indextts2-2.5-8bit"
# 若 ModelScope 有镜像更快；但这是社区 HF 模型，优先确认 HF 是否齐全
```
⚠️ **不要只挑几个文件下**——`model_manifest.json` 和 `config.yaml` 是重建 Swift 推理图的命门，必须全量下。

**说话人预处理器（w2v-bert-2.0 / campplus）**：iOS 端要不要带，取决于下面决策 A。若走「预计算」，这两个模型只在 Mac 端用，不进 iOS 包。

---

## 1. 总体推理图（要重写什么）

```
文本 ──► [Tokenizer] ──► token ids
                                  │
参考音频 ──► [Speaker Enc] ──► speaker embedding (.npz 预计算)
                                  │
        token ids + speaker ──► [GPT 自回归 + KV cache] ──► 中间 token 序列
                                          │
                          ──► [Codec + s2mel 扩散解码] ──► mel 谱
                                          │
                                  ──► [BigVGAN] ──► 波形 WAV
                                          │
                                  ──► [AVFoundation] 播放 / 存盘
```

- **要在 Swift 重写**：Tokenizer 前端、GPT（RoPE+KV cache）、Codec、s2mel（扩散）、BigVGAN。
- **可绕开（决策 A）**：w2v-bert + campplus 说话人编码器 → 改为 Mac 端一次性预计算成 `.npz`，iOS 直接读。
- **BigVGAN 已确认格式可用**，是最小风险模块，作为第一个打通目标。

---

## 2. 三个要你拍板的决策

### A. 说话人策略（最影响工作量）
- **A1（推荐）预计算 .npz**：说话人音色在 Mac 端用官方 Python 工具跑一次 `mlx-indextts speaker -r ref.wav -o speaker.npz`，iOS 直接加载 .npz。iOS 包**不塞 w2v-bert（~1GB）**，内存与复杂度骤降，且 w2v-bert 在 MLX Swift 上重建风险极高。
  - 代价：换一个新声音要先在 Mac 预处理一次。
- **A2 全量 on-device**：iOS 内跑 w2v-bert + campplus，任意参考音频即时克隆。
  - 代价：w2v-bert 是巨型 transformer，MLX Swift 重写 + 内存 + 速度都凶险，很可能 iPad Pro 才扛得住。

### B. 目标设备 RAM 档位
- iPhone 15 Pro / 16 系列（8GB）：权重常驻 ~1.5–1.8GB，扩散 16 步峰值可能贴边，**需 Increased Memory Limit 授权 + 谨慎 cacheLimit**。
- iPad Pro（M 系，8–16GB）：宽裕很多，推荐首测平台。
- 请告诉我具体机型，我按 RAM 定 cacheLimit 与是否允许全量 on-device。

### C. 8bit GPT 权重在 MLX Swift 的落地方式
- **C1（推荐，稳）**：加载时把 8bit 分组权重在 Swift 里**反量化为 F16**，跑普通 matmul。慢一点但格式零风险，且能用 Mac Python 参考对拍数值。
- **C2 原生量化**：用 MLX Swift `QuantizedLinear` 直接吃量化张量。更快，但要求 vanch007 的转换格式与 MLX Swift 期望完全一致，不匹配会静默出错。先 C1，跑通后再换 C2 提速。

---

## 3. Xcode 工程结构

```
IndexTTS2iOS/
├── Package.swift              # 依赖 mlx-swift / mlx-swift-examples
├── Sources/
│   ├── ModelLoader.swift      # 读 safetensors + model_manifest 映射 + config 解析
│   ├── Tokenizer.swift        # tiktoken 移植 + 多语言注音
│   ├── GPT.swift              # 自回归 + RoPE + KV cache + (反量化)
│   ├── Codec.swift            # VQ codec
│   ├── S2MelDiffusion.swift   # 扩散解码 token→mel
│   ├── BigVGAN.swift          # Snake/resblock/上采样（最独立）
│   ├── SpeakerStore.swift     # 读预计算 .npz
│   ├── AudioIO.swift          # mel→WAV、AVAudioEngine 播放、存盘
│   └── App.swift + Views/     # SwiftUI（液态玻璃风）
└── Resources/models/          # 放补齐的权重 + config + tokenizer
```

- 依赖：Apple 官方 `ml-explore/mlx-swift`（含 `MLX`、`MLXNN`、`MLXRandom`、`MLXOptimizers`）。不引第三方推理框架。
- 授权：`com.apple.developer.kernel.increased-memory-limit`（iOS 18+ / A17 Pro+），Sideloadly 注入 entitlements 侧载。

---

## 4. 各模块实现要点

### 4.1 权重加载
- `MLX.Safetensors` 读张量；用 `model_manifest.json` 把转换后的 MLX 名映射回模块参数名。
- 8bit 走决策 C1：读 scales+ints → `dequantize(groupSize)` → `MLXArray(F16)`。
- config.yaml 定 n_layers / d_model / n_heads / vocab / mel bins，等模型补齐后我读它定结构。

### 4.2 文本前端（Tokenzier.swift）
- 移植 tiktoken：base64 词表 + BPE 合并规则，纯 Swift 实现（无官方 Swift 版）。
- 多语言：中/英/日/西/阿；拼音/CMU/Kana 注音需用分隔符标记，逻辑需**字节级一致**否则生成崩坏。
- 风险：分词必须和 Python 参考逐 token 对齐 → 用 Mac 参考输出做对拍。

### 4.3 GPT（GPT.swift）
- Decoder-only transformer：RMSNorm + RoPE（旋转位置编码）+ 自注意力（KV cache 增量）+ MLP。
- 流式：已完成安全文本段产出即送下游（对应 `--stream`）。
- 自回归循环在 Swift actor 里跑，避免阻塞 UI；后台卸载模型。

### 4.4 Codec + s2mel 扩散（Codec.swift / S2MelDiffusion.swift）
- **最高风险模块**：IndexTTS2 内部扩散解码器架构未完全公开，只能从权重 + config 反推 op 顺序。
- 策略：严格对照 Mac Python 参考的层名与张量形状逐层重建；先在小 batch 对拍单步去噪输出。
- 扩散 16 步（`--diffusion-steps 16`）在 A17 上可能数秒，需进度反馈。

### 4.5 BigVGAN（BigVGAN.swift）
- Snake 激活：`x + (1/γ)·sin²(γx)` 的 anti-aliased 变体 + α/β 参数（已见于权重头）。
- HiFi-GAN resblock： dilated conv + LReLU；上采样转置/亚像素卷积。
- 输入 mel[80/... bins] → 输出波形。最独立，**先做单测对拍**。

### 4.6 音频输出（AudioIO.swift）
- 拼 WAV 头写文件；`AVAudioPCMBuffer` + `AVAudioEngine` 播放；采样率按 config（通常 24k/32k/48k）。

---

## 5. 内存与性能预算

- 权重常驻：8bit gpt(~1.1GB) + codec + s2mel + f16 bigvgan(224MB) ≈ **1.5–1.8GB**。
- 峰值：激活 + KV cache + 扩散 16 步中间量，可能再 +0.3–0.8GB。
- 规则：iOS 不超过总 RAM 60%；`MLX.GPU.set(cacheLimit: 512MB–1GB)`；进后台卸载。
- 延迟参考（Mac 8bit RTF≈0.9）：A17 可能 2–5×，一句约 3–10s；iPad Pro M 系接近 Mac。
- 必须真机（模拟器无 Metal GPU）。

---

## 6. UI（正好用上你调研的液态玻璃）

- SwiftUI + iOS 26 `.glassEffect()`：参考音频选择卡、文本输入浮层、情绪控制（happy 等 + emo-alpha 滑杆）、生成进度、播放/存盘。
- 可用性门控 `#available(iOS 26, *)`，老系统降级 `.thinMaterial`。
- 参考音频从「文件」App 选，或直接录一段。

---

## 7. 验证策略（你不编译，所以靠对拍）

以 Mac 端官方 Python 工具（`vanch007/mlx-indextts2`）为 **ground truth**，逐模块数值对拍：
1. **BigVGAN**：同 mel 输入 → 比对我方波形 vs Python 波形（MSE 阈值）。
2. **GPT**：同 token + speaker → 对拍自回归输出的 token 序列一致。
3. **Tokenizer**：同文本 → 对拍 token id 列表字节级一致。
4. **端到端**：同文本 + 同参考 → 听感 + ASR 校验（mlx-whisper）无明显错句。
- 这样即使你主设备坏、我无法上机，也能在 Mac 侧（你或我）验证正确性后再上真机。

---

## 8. 凶险风险清单（你要心里有数）

1. **s2mel 扩散解码器**：内部架构不公开，从权重反推 op 顺序最容易出错 → 产出噪声。
2. **8bit 分组格式**：C1 反量化最稳，C2 原生量化可能静默不匹配。
3. **Tokenizer 字节级一致**：多语言注音分隔稍有偏差 → 生成崩。
4. **数值 parity**：哪怕一层归一化顺序错 → 可听伪影，只能靠对拍发现。
5. **iOS 内存上限**：扩散 16 步峰值可能 OOM，需 entitlement + cacheLimit 调优。
6. **免费签名 Increased Memory Limit**：免费开发者账号能否注入该授权需实测（Sideloadly 可加 entitlements）。
7. **无法在你主设备编译**：我写全量代码，真机验证需你另找 Mac 或用我（你）的 Mac 跑。

---

## 9. 我写代码的阶段顺序（P0→P7）

- **P0** 工程骨架 + 权重加载 + config/manifest 解析（模型补齐后）
- **P1** BigVGAN 单通 + mel→wav 对拍（最小风险，先立信心）
- **P2** Tokenizer 多语言前端 + 对拍
- **P3** GPT 自回归 + KV cache + 对拍
- **P4** Codec + s2mel 扩散（最高难）
- **P5** 全链路端到端 + 对拍
- **P6** SwiftUI 液态玻璃 UI + 音频播放/存盘
- **P7** 内存/性能调优 + 真机 sideload 指引

---

## 10. 我现在需要你确认 / 提供

1. **补齐模型**：把 `vanch007/mlx-indextts2-2.5-8bit` 全量下到 `D:\indexTTS 2.5\models\mlx-indextts2-2.5-8bit\`（含 manifest/config/tokenizer）。
2. **拍板决策 A / B / C**（见第 2 节）。
3. **目标机型**（RAM 档位），决定内存预算与能否全量 on-device。
4. 若走 A1：Mac 端是否已有 `vanch007/mlx-indextts2` Python 工具 + w2v-bert/campplus，用来产 .npz。

确认后我从 P0 开始写，先不碰真机编译。
