import SwiftUI

/// 关于页（版本号 + 仓库地址 + 许可声明）
struct AboutSheet: View {

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Huamei IndexTTS")
                                .font(Theme.Fonts.titleHero)
                                .foregroundStyle(Theme.Colors.labelPrimary)
                            Text("Internal Beta 1")
                                .font(Theme.Fonts.callout)
                                .foregroundStyle(Theme.Colors.accentStrong)
                            Text("Native on-device voice cloning · iOS 17+")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Colors.labelSecondary)
                        }
                        Spacer()
                        Image(systemName: "waveform.circle.fill")
                            .font(.system(size: 56))
                            .foregroundStyle(Theme.Colors.accent)
                    }

                    Group {
                        Text("功能亮点").font(Theme.Fonts.heading)
                        Bullet("零云端，全部数据留在设备本地")
                        Bullet("多语言：中文 · 英文 · 日语 · 西语 · 阿语")
                        Bullet("8 种情感预设")
                        Bullet("支持 Liquid Glass 毛玻璃（iOS 26+）")
                    }
                    .foregroundStyle(Theme.Colors.labelSecondary)

                    Group {
                        Text("致谢与许可").font(Theme.Fonts.heading)
                        Text("推理引擎 · IndexTeam IndexTTS-2.5 · Bilibili 模型使用许可")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("模型权重 · vanch007/mlx-indextts2-2.5-8bit · MIT 转换")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("框架 · Apple MLX Swift · Apache-2.0")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("开源仓库 · github.com/WebExperiments-glitch/huamei-indexTTS")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                    }

                    Text("© 2026 · 为开源社区而作")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 12)
                }
                .padding(20)
            }
            .background(Theme.Colors.canvas.ignoresSafeArea())
            .navigationTitle("关于")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}

private struct Bullet: View {
    let text: String
    init(_ t: String) { text = t }
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(Theme.Colors.accent)
                .font(.system(size: 14, weight: .semibold))
                .padding(.top, 2)
            Text(text)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Colors.labelSecondary)
        }
    }
}