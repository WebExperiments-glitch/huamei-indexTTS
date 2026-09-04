# Huamei IndexTTS · iOS

原生端侧（on-device）语音克隆 App：iOS 上纯本地跑通 IndexTTS-2.5 的全链路 TTS 与"任意音频即时声纹克隆"（A2）。

- 内核：自研 MLX Swift 推理（量化 GPT / 语义编解码 / s2mel 扩散 / BigVGAN 声码），p0 Python 数值孪生逐模块对拍
- 克隆：0 到 1 自研 w2v-bert Conformer + CAM++ DTDNN + Seamless 特征化（无第三方实现参照）
- UI：SwiftUI · Liquid Glass · 白+橙
- 构建：SwiftPM 单包，Windows 写码 + GitHub Actions（xtool builder，Xcode 26）云编译 → zsign/爱思重签侧载

## 目录

| 路径 | 内容 |
|---|---|
| [ios/](ios/README.md) | iOS 工程（**入口文档**：架构 / 内存 / 模型资源 / A2 状态 / first-compile 清单） |
| [ios/Sources/MLXIndexTTS2Core/](ios/Sources/MLXIndexTTS2Core) | 推理内核（GPT / Codec / s2mel / BigVGAN / Voice 克隆链） |
| [scripts/](scripts/) | 模型转换与特征配方黑盒标定工具（w2vbert_export / probe / golden） |
| [p0/](p0/)（本地） | Python 数值孪生 + golden（不进仓库，见 .gitignore） |
| [models/](models/)（本地） | MLX 权重 + 克隆组件（按 README 单独下载，不进仓库） |

## 模型组件（按需下载，见 ios/README）

- 合成四件套：`vanch007/mlx-indextts2-2.5-8bit`（gpt/codec/s2mel/bigvgan safetensors + tiktoken）
- 克隆组件：`facebook/w2v-bert-2.0`（magpod 镜像亦可）+ `iic/speech_campplus_sv_zh-cn_16k-common`（campplus_cn_common.bin）

## 许可

- 引擎权重 IndexTeam IndexTTS-2.5 · Bilibili Model Use License（商用 ≤100M MAU / RMB 1B 营收）
- 权重转换 vanch007 · MIT · 框架 Apple MLX Swift · Apache-2.0