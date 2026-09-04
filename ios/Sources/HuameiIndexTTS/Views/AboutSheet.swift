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
                        Text("What's inside").font(Theme.Fonts.heading)
                        Bullet("Zero-cloud, all on your device")
                        Bullet("Multilingual: ZH · EN · JA · YUE · ES")
                        Bullet("Emotion presets (8 categories)")
                        Bullet("Liquid Glass where supported (iOS 26+)")
                    }
                    .foregroundStyle(Theme.Colors.labelSecondary)

                    Group {
                        Text("Credits & License").font(Theme.Fonts.heading)
                        Text("Engine · IndexTeam IndexTTS-2.5 · Bilibili Model Use License")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("Weights · vanch007/mlx-indextts2-2.5-8bit · MIT conversion")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("Framework · Apple MLX Swift · Apache-2.0")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("Repository · github.com/WebExperiments-glitch/huamei-indexTTS")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                    }

                    Text("© 2026 · Made for the open-source community")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 12)
                }
                .padding(20)
            }
            .background(Theme.Colors.canvas.ignoresSafeArea())
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
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