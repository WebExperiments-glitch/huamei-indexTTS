import SwiftUI

/// "液态玻璃" 视觉修饰 —— 最低 iOS 17（项目约束）；iOS 26 之上叠原生增强
///
/// 实现策略：
///   · iOS 17-25：用 SwiftUI 内置 `.regularMaterial` + 圆角 + 细描边 + 阴影，模拟 ~90% 视觉接近
///   · iOS 26+：在 material 之上再叠 `.glassEffect(.regular.interactive(), in: shape)`
///
/// 调用方式：`.glassCard(cornerRadius: 24)`（默认交互型）或 `.glassCard(.subtle)`
extension View {

    /// 玻璃卡片（圆角矩形 + 模糊 + 细描边 + 阴影）
    func glassCard(cornerRadius: CGFloat = 22,
                   style: GlassStyle = .regular,
                   interactive: Bool = true) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius,
                                   style: style,
                                   interactive: interactive))
    }

    /// 玻璃按钮（圆角胶囊，主操作色为橙色）
    func glassPrimaryButton(disabled: Bool = false) -> some View {
        modifier(GlassPrimaryButtonModifier(disabled: disabled))
    }

    /// 玻璃进度条底槽（半透明胶囊）
    func glassTrack() -> some View {
        modifier(GlassTrackModifier())
    }
}

/// 玻璃风格预设
enum GlassStyle: Hashable {
    case regular     // 主卡片
    case subtle      // 次要背景层
    case elevated    // 弹出层（更强光晕）
}

/// ─────────────────────────── Modifiers ───────────────────────────

private struct GlassCardModifier: ViewModifier {
    let cornerRadius: CGFloat
    let style: GlassStyle
    let interactive: Bool

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        let material: Material = {
            switch style {
            case .regular: return .regularMaterial
            case .subtle:  return .ultraThinMaterial
            case .elevated: return .thickMaterial
            }
        }()

        content
            .background {
                shape
                    .fill(material)
                    .overlay(
                        shape.stroke(Theme.Colors.glassStroke,
                                     lineWidth: 0.5)
                    )
                    .shadow(color: Theme.Colors.glassShadow,
                            radius: 16, x: 0, y: 6)
            }
            // iOS 26 叠加原生液态玻璃；旧版本自动忽略
            .modifier(Glass26Enhancer(shape: AnyShape(shape), interactive: interactive))
    }
}

private struct Glass26Enhancer: ViewModifier {
    let shape: AnyShape
    let interactive: Bool

    func body(content: Content) -> some View {
        if #available(iOS 26, *) {
            content.glassEffect((interactive ? .regular.interactive()
                                          : .regular),
                                in: shape)
        } else {
            content
        }
    }
}

/// 绕开 iOS 26 API 的 type-erase：把 shape 包成 AnyShape
private struct AnyShape: Shape {
    private let _path: (CGRect) -> Path
    init<S: Shape>(_ s: S) { self._path = { rect in s.path(in: rect) } }
    func path(in rect: CGRect) -> Path { _path(rect) }
}

private struct GlassPrimaryButtonModifier: ViewModifier {
    let disabled: Bool

    func body(content: Content) -> some View {
        let shape = Capsule(style: .continuous)
        return content
            .font(Theme.Fonts.heading)
            .foregroundStyle(Theme.Colors.labelInverse)
            .padding(.horizontal, 22)
            .padding(.vertical, 14)
            .background {
                shape
                    .fill(disabled
                          ? Theme.Colors.accentSoft.opacity(0.6)
                          : Theme.Colors.accent)
                    .overlay(shape.stroke(Theme.Colors.glassStroke, lineWidth: 0.5))
                    .shadow(color: Theme.Colors.accent.opacity(0.25),
                            radius: 14, x: 0, y: 6)
            }
            .opacity(disabled ? 0.55 : 1.0)
            .animation(.spring(response: 0.28, dampingFraction: 0.8),
                       value: disabled)
    }
}

private struct GlassTrackModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .frame(height: 8)
            .background(
                Capsule(style: .continuous)
                    .fill(Theme.Colors.divider)
            )
    }
}