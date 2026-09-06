import SwiftUI

/// 克隆音频参考卡（点击 → AudioSourceSheet 4 入口）
struct VoiceReferenceCard: View {

    @EnvironmentObject private var s: SessionStore
    let onPick: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text("Voice Reference")
                    .font(Theme.Fonts.heading)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                Spacer()
            }
            Button(action: onPick) {
                HStack(spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(s.referenceURL == nil ? Theme.colors_canvas_fill : Theme.Colors.accentTint)
                            .frame(width: 44, height: 44)
                        Image(systemName: s.referenceURL == nil ? "waveform.slash" : "waveform")
                            .font(.system(size: 22, weight: .medium))
                            .foregroundStyle(s.referenceURL == nil
                                             ? Theme.Colors.labelTertiary
                                             : Theme.Colors.accentStrong)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(s.referenceURL?.lastPathComponent ?? "添加参考声音")
                            .font(Theme.Fonts.callout)
                            .foregroundStyle(Theme.Colors.labelPrimary)
                            .lineLimit(1)
                        Text(s.referenceURL == nil
                             ? "点击选择：录音 · 音频 · 视频 · 文件"
                             : "点击更换")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Colors.labelTertiary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.Colors.labelTertiary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 14)
            }
            .buttonStyle(.borderless)
            .glassCard(cornerRadius: 18, interactive: false)
        }
    }
}

// Color 缺失主题快捷
private extension Theme {
    static let colors_canvas_fill = Theme.Colors.surfaceAlt
}