# Huamei IndexTTS · iOS

原生端侧（on-device）语音克隆 App：iOS 上纯本地跑通 IndexTTS-2.5 的合成与"任意音频即时声纹克隆"（A2）。

- 内核：自研 MLX Swift 推理（量化 GPT / 语义编解码 / s2mel 扩散 / BigVGAN 声码）
- 克隆：0 到 1 自研 w2v-bert Conformer + CAM++ 语音嵌入 + Seamless 特征化
- 模型分发：App 内一键从魔搭下载（进度 / sha256 校验 / 断点重试），零手工配置
- UI：SwiftUI · Liquid Glass · 白+橙
- 构建：SwiftPM 单包 + XcodeGen；GitHub Actions（Xcode 26）云编译出免签名 IPA，重签侧载

## 目录

| 路径 | 内容 |
|---|---|
| [ios/](ios/README.md) | iOS 工程（架构 / 内存 / 模型 / A2 状态 / first-compile 清单） |
| [ios/Sources/MLXIndexTTS2Core/](ios/Sources/MLXIndexTTS2Core) | 推理内核（GPT / Codec / s2mel / BigVGAN / Voice 克隆链） |
| [ios/Sources/HuameiIndexTTS/](ios/Sources/HuameiIndexTTS) | SwiftUI App（下载 / 合成 / 克隆 UI） |

## 模型

App 首次启动自动从魔搭仓库按清单下载并校验（清单随包发布：`ModelManifest.json`）：

- 托管：https://modelscope.cn/models/jfjijiogkijijg/huamei-TTS
- 合成四件套 gpt/codec/s2mel/bigvgan + 克隆组件 w2v-bert-2.0 / campplus / stats

## 许可

- 引擎权重 IndexTeam IndexTTS-2.5 · Bilibili Model Use License（商用 ≤100M MAU / RMB 1B 营收）
- 权重转换 vanch007 · MIT · 框架 Apple MLX Swift · Apache-2.0