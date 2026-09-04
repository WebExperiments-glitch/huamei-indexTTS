# 调研报告：Apple 液态玻璃 / Swift 整洁写法 / MLX IndexTTS2 在 iPhone·iPad 部署

> 调研时间：2026-09-01
> 来源：Apple Developer Documentation、bleepingswift.com、swiftcrafted.dev、lobehub skill、vanch007/mlx-indextts2（GitHub + HuggingFace）、mlx-swift-examples、Microsoft PhiCookBook、WWDC 2025 MLX 资料

---

## 一、Apple 液态玻璃（Liquid Glass）怎么写

### 1.1 是什么
Liquid Glass 是 iOS 26 / iPadOS 26 / macOS 26 / tvOS 26 / watchOS 26 引入的半透明动态材质。它会**模糊背后的内容、反射周围颜色与光线、随设备姿态产生高光、随触摸形变**。系统组件（tab bar、toolbar、navigation bar、sheet）用 iOS 26 SDK 编译后自动带上玻璃效果；自定义视图用 `.glassEffect()` 自己加。

### 1.2 最小可用代码
```swift
import SwiftUI

struct FloatingButton: View {
    var body: some View {
        Button("Continue") { /* action */ }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .glassEffect()                       // 默认 .regular 变体 + 胶囊形状
    }
}
```

### 1.3 三个变体
```swift
.glassEffect(.regular)                          // 标准磨砂玻璃，最常用
.glassEffect(.clear)                            // 纯折射、几乎不磨砂，适合照片/视频上方（文字上会看不清）
.glassEffect(.identity)                         // 无效果，用于可访问性 / 条件关闭
```

### 1.4 形状（第二个参数）
```swift
.glassEffect(.regular, in: .circle)                          // 圆形图标按钮
.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 16)) // 大卡片
.glassEffect(.regular, in: .rect(cornerRadius: .containerConcentric)) // 嵌套时自动对齐父容器圆角
```
默认形状是 `Capsule`。

### 1.5 着色
```swift
.glassEffect(.regular.tint(.red))                // 破坏性操作
.glassEffect(.regular.tint(.green.opacity(0.8))) // 确认
```
注意：`.regular.tint(color)` 在浅色模式约混合 30%，深色模式约 50%。品牌色本身偏暗时，深色模式结果会接近不透明，务必两端都测。

### 1.6 合并多个玻璃面 / 变形动画
相邻两个 `.glassEffect` 各自采样背景会留**接缝**。用 `GlassEffectContainer` 共享一次采样并解锁“变形(morph)”：
```swift
GlassEffectContainer(spacing: 12) {              // spacing 控制多近才开始融合
    Image(systemName: "play.fill").glassEffect()
    Image(systemName: "pause.fill").glassEffect()
    Image(systemName: "forward.fill").glassEffect()
}
```
跨视图层级变化时的变形身份（系统媒体控件用的就是这招）：
```swift
.glassEffectID(myID, in: namespace)             // 给玻璃一个稳定 ID，插入/移除时 morph
.glassEffectTransition(.materialize)            // 出现/消失的过渡：.matchedGeometry / .materialize / .identity
.glassEffectUnion(id: myID, namespace: namespace) // 多个视图合并成一个玻璃形状
```

### 1.7 按钮样式 / 背景延伸 / 滚动边缘 / 工具栏
```swift
Button("Action") { }.buttonStyle(.glass)          // 标准玻璃按钮
Button("Primary") { }.buttonStyle(.glassProminent) // 强调玻璃按钮

// 让内容镜像+模糊延伸进安全区、垫在玻璃工具栏后面
content.backgroundExtensionEffect()

// 滚动边界柔化
ScrollView { content }
    .scrollEdgeEffectStyle(.soft, for: .top)

// 工具栏里给玻璃项之间留视觉断口
.toolbar {
    ToolbarItem { Button("Edit") {} }
    ToolbarItem { ToolbarSpacer() }
    ToolbarItem { Button("Share") {} }
}
```

### 1.8 可用性门控（必须，否则老系统崩溃）
```swift
if #available(iOS 26, *) {
    Text("Status")
        .padding()
        .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 16))
} else {
    Text("Status").padding().background(.thinMaterial)
}
```

### 1.9 坑（实测踩过的）
- `.clipped()` / `.mask()` 会**直接干掉玻璃效果**（渲染器采不到背景），降级成扁平材质且无警告。要逃出滚动容器就把玻璃 `.overlay` / `.containerRelativeFrame` 抬出去。
- `Form` 强制自己的 grouped 材质，`.glassEffect` 在 Form 行里是 no-op；行要玻璃用 `List`。
- `GlassEffectContainer` **不跨 NavigationStack**，跨屏 morph 会降级成淡入淡出；跨屏用 `.navigationTransition(.zoom(...))`。
- `.interactive()`（触摸实时形变）**只在 A17 Pro 及以上生效**，老设备静默降级成静态玻璃。
- 玻璃压在密集文字上是可读性灾难（Mail 用实心底放正文、只把工具栏留给玻璃）。
- XCUITest 截图非确定性（高光依赖设备姿态），要么跳过玻璃要么固定朝向。
- 性能：`.interactive()` 每帧多一次全屏采样。iPhone 15 Pro Max 上 12 个可见玻璃 cell 会掉帧；改成非 interactive + 每组一个 `GlassEffectContainer` 即可。

### 1.10 何时用
放在**导航层**（浮动按钮、自定义工具栏/tab bar、媒体播放浮层、sheet 头部）。主内容流里的列表行、卡片一般**不要**用玻璃——效果最好时有真实内容在背后折射。

---

## 二、Swift / SwiftUI 怎么写好看又整洁

### 2.1 架构
- 简单 App：**MVVM**（View 只管展示，ViewModel 管逻辑与状态）。
- 复杂 App / 大团队 / 需要强测试：考虑 **TCA（The Composable Architecture）** 单向数据流。
- iOS 17+ 用 **`@Observable` macro** 取代 `ObservableObject` + `@Published`，代码更干净：
```swift
@Observable
class UserStore {
    var currentUser: User?
    var isLoggedIn: Bool { currentUser != nil }
    func login(email: String, password: String) async throws { /* ... */ }
}
```

### 2.2 属性包装器正确分工
| 包装器 | 用途 |
|---|---|
| `@State` | 视图内局部的简单值（Bool/String/Int） |
| `@Binding` | 把可变状态传给子视图 |
| `@StateObject` / `@Observable` 持有 | 创建并拥有可观察模型 |
| `@ObservedObject` | 引用别人拥有的模型 |
| `@EnvironmentObject` | 跨视图层级共享状态 |

常见错误：`@ObservedObject var vm = ViewModel()` 写在视图里会在每次刷新重建——应用该用 `@StateObject`（或 `@State` 持有 `@Observable`）。

### 2.3 视图拆分 + @ViewBuilder 复用容器
单一职责，每个视图只做一件事；用 `@ViewBuilder` 做可复用卡片/容器：
```swift
struct Card<Content: View>: View {
    @ViewBuilder let content: () -> Content
    var body: some View {
        content()
            .padding()
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
```

### 2.4 现代 API 写法（别用废弃的）
```swift
.foregroundStyle(.blue)                       // 不是 .foregroundColor()
.clipShape(.rect(cornerRadius: 12))           // 不是 RoundedRectangle(cornerRadius:) 当简单圆角
.background(.white)
.overlay(RoundedRectangle(cornerRadius: 12).stroke(.gray.opacity(0.2), lineWidth: 1))
```

### 2.5 性能
- 长列表/网格用 `LazyVStack` / `LazyHStack` / `LazyVGrid`（只渲染可见项）。
- 让自定义视图 `conform Equatable` 或用 `EquatableView` 减少无谓重绘。
- **body 里绝不跑重计算**（过滤/网络/图像处理），移到 ViewModel 或后台任务。
- 保持 state 小而聚焦，把静态子视图抽成独立 struct，避免过深层级。
- 动画克制、有目的（Apple 建议保持 60fps）。

### 2.6 设计系统（好看的关键）
- 间距体系：`padding(.horizontal, 20)`（页面边距）、`VStack(spacing: 24)`（大段）、`12`（相关）、`8`（紧凑）。
- 字号层级：页面标题 20 semibold / 区块头 16 semibold / 正文 16 / 次要 14 / 注释 12。
- 触控目标 ≥ 44pt。
- 语义色：`.primary` / `.secondary` / `.blue` 等；状态色 `.green`/`.red`/`.orange`/`.blue`。
- 现代几何模糊背景提升质感：4+ 个重叠渐变形状 + `.blur(radius: 12~25)` + `.offset()`。

---

## 三、vanch007/mlx-indextts2-2.5-8bit 在 iPhone / iPad 部署

### 3.1 先说清楚现实
- 这个模型的**官方运行仓库** `vanch007/mlx-indextts2` 明确要求：
  ```
  Requirements:
  - macOS with Apple Silicon (M1/M2/M3/M4)
  - Python 3.10+
  - uv package manager
  ```
- 通读 README：**没有任何 iOS / iPadOS / iPhone / MLX Swift 的内容**。runtime 是 **Python 版 MLX**，不是 Swift。
- 模型本体：8-bit 量化，约 **1.6GB**，转换自 `IndexTeam/IndexTTS-2.5`；GPT 用持久 8-bit 分组量化，其余组件 float16。支持中/英/日/西/阿，含跨语言音色克隆、独立说话人/情绪参考、拼音/CMU/Kana 注音。
- **结论：这个特定工具链不能在 iPhone/iPad 上原生直接跑。** 下面两条是现实可行的路。

### 3.2 路径 A（今天就能用）：Mac 推理 + iPad/iPhone 浏览器访问
在 Mac 上起 API Server / WebUI，iPad 用 Safari 同局域网打开。仓库自带：
```bash
# 安装带 WebUI / API 的完整依赖
uv sync --extra v25 --extra qwen --extra api --extra webui

# 起 WebUI（浏览器访问，含语言选择/情绪/注音/进度）
uv run mlx-indextts-webui

# 或只起 API
uv run mlx-indextts-api
# 端点：/health /profiles /generate /generate/stream /speaker /batch /plan /audio
```
- 命令行生成示例：
```bash
uv run mlx-indextts generate --profile v25 -r reference.wav \
    --language zh -t "你好，这是 IndexTTS 2.5 的 MLX 推理测试。" -o output.wav
```
- 远程访问：Mac 在家跑服务，iPad 在外用 **Tailscale / 内网穿透** 连回（WireGuard 加密、点对点）。这就是“iPad 上用 IndexTTS2”的务实方案——Mac 出算力，移动端出界面。
- 缺点：依赖 Mac 开机 + 联网，不是“纯口袋离线”。

### 3.3 路径 B（真·原生 on-device）：用 MLX Swift 重写推理
- **MLX Swift** 是 Apple 官方 Swift 端口，能在 **iPhone / iPad（iOS 18+，至少 8GB RAM，如 iPhone 15 Pro+ / 近年 iPad）** 上跑模型，走统一内存 + Metal GPU。已有成熟范式：Microsoft PhiCookBook 的 “iOS Integration with MLX”、`ml-explore/mlx-swift-examples` 的 LLMEval（克隆→Xcode 签名→真机运行→下载 mlx-community 模型→设备内推理）。
- 但 **IndexTTS2 不是 LLM**，它是「自回归 GPT + 扩散声码器 + BigVGAN + w2v-bert 预处理」的复合 TTS 推理图。vanch007 的 Python 实现里这一整套要在 **MLX Swift 里重写**（加载 safetensors、实现 GPT 自回归循环、扩散采样、BigVGAN 波形合成、speaker 条件预处理）。**目前没有现成的 IndexTTS2 iOS App**。
- 设备门槛（iOS 端跑大模型通用规则）：
  - 设备 RAM ≥ 8GB（iPhone 15 Pro+/16 系列、M 系 iPad）。
  - 模型占内存不超过总 RAM 的 ~60%；需 **Increased Memory Limit** 授权（iOS 18+ / A17 Pro+）。
  - 设 GPU 缓存上限：`MLX.GPU.set(cacheLimit: 512 * 1024 * 1024)`；App 进后台要卸载模型。
  - 必须真机（模拟器不支持 Metal GPU）。
  - 你之前用 Sideloadly 侧载 IPA 的流程，理论上可以用来装一个自己/社区用 mlx-swift 编译出的 TTS App——前提是有人先把推理层用 Swift 写出来。

### 3.4 不推荐的路径
- 转 Core ML（`.mlpackage`）：TTS 这种带循环/扩散的模型转 CoreML 极麻烦，基本不现实。
- 直接把 Python 包塞 iOS：不可能，iOS 没有 Python 运行时也没法跑 uv。

### 3.5 给你的建议
1. **想马上在 iPad 上用** → 走路径 A：Mac 跑 `mlx-indextts-webui`，iPad Safari 同网访问；外出加 Tailscale。零移植成本。
2. **想真口袋离线** → 等社区出 MLX Swift 版 IndexTTS2，或自己基于 `mlx-swift-examples` 把 TTS 推理图用 Swift 实现（工作量不小，但架构可行）。可盯着 `vanch007/mlx-indextts2` 仓库和 `ml-explore/mlx-swift-examples` 的后续。
3. 模型授权注意：权重是 **bilibili Model Use License** 的衍生作品，商用有阈值和义务限制，下载前看 LICENSE。

---

## 附：一句话总览
- 液态玻璃 = `.glassEffect()`，导航层专用，注意 `.clipped()`/`.interactive()`/可用性门控三个坑。
- 好 Swift = MVVM + `@Observable` + 视图拆小 + `@ViewBuilder` + 现代 API + 懒加载。
- IndexTTS2 上 iPad/iPhone = 官方只给 Mac 的 Python 工具链；今天用「Mac 跑 WebUI + iPad 浏览器」，原生离线需 MLX Swift 移植（暂无现成 App）。
