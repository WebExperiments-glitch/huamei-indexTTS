import SwiftUI

/// 主配色 —— 白 + 橙（Internal Beta 1）
///
/// 原则：主背景接近纯白（#FCFCFD），主操作与品牌强调统一使用一种橙色 accent；
/// 灰阶用于次要文字 / 边框 / 卡片底；错误用偏红橙，警告偏琥珀；保持三原色克制使用。
///
/// 访问方式：`Theme.Colors.accent`、`Theme.Fonts.title`（避免与 SwiftUI.Color / Font 同名冲突）
enum Theme {

    enum Colors {
        // 背景层级（从底层到顶层）
        static let canvas       = Color(white: 0.988)
        static let surface      = Color(red: 0.99, green: 0.99, blue: 0.995)
        static let surfaceAlt   = Color(red: 1.00, green: 0.96, blue: 0.95)
        static let divider      = Color(white: 0.90).opacity(0.7)

        // 文字
        static let labelPrimary   = Color(white: 0.10)
        static let labelSecondary = Color(white: 0.35)
        static let labelTertiary  = Color(white: 0.55)
        static let labelInverse   = Color.white

        // 主品牌色 —— 橙
        static let accent         = Color(red: 1.00, green: 0.45, blue: 0.10)   // #FF7319
        static let accentStrong   = Color(red: 1.00, green: 0.38, blue: 0.05)   // #FF610D
        static let accentSoft     = Color(red: 1.00, green: 0.62, blue: 0.36)
        static let accentTint     = Color(red: 1.00, green: 0.86, blue: 0.74).opacity(0.35)

        // 状态
        static let success = Color(red: 0.13, green: 0.70, blue: 0.36)
        static let warning = Color(red: 0.96, green: 0.62, blue: 0.10)
        static let danger  = Color(red: 0.92, green: 0.27, blue: 0.21)

        // 玻璃效果
        static let glassStroke = Color.white.opacity(0.55)
        static let glassShadow = Color.black.opacity(0.07)
    }

    enum Fonts {
        static let titleHero = Font.system(size: 28, weight: .bold,    design: .rounded)
        static let title     = Font.system(size: 22, weight: .semibold, design: .rounded)
        static let heading   = Font.system(size: 18, weight: .semibold, design: .default)
        static let body      = Font.system(size: 15, weight: .regular, design: .default)
        static let callout   = Font.system(size: 14, weight: .medium,  design: .default)
        static let mono      = Font.system(size: 13, weight: .regular, design: .monospaced)
        static let caption   = Font.system(size: 12, weight: .regular, design: .default)
    }
}