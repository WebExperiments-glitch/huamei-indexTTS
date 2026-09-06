import SwiftUI
import UIKit

/// 合成结果播放卡
struct ResultCard: View {

    @EnvironmentObject private var s: SessionStore
    @State private var player = AudioPlayer()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("合成结果")
                    .font(Theme.Fonts.heading)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                Spacer()
            }

            HStack(spacing: 12) {
                Button(action: togglePlay) {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44, weight: .regular))
                        .foregroundStyle(Theme.Colors.accent)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .frame(minWidth: 56, minHeight: 56)

                VStack(alignment: .leading, spacing: 4) {
                    Text(player.isPlaying ? "播放中…" : "点击播放")
                        .font(Theme.Fonts.callout)
                        .foregroundStyle(Theme.Colors.labelPrimary)
                    Text(s.resultURL?.lastPathComponent ?? "")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Colors.labelTertiary)
                        .lineLimit(1)
                }
                Spacer()
                Button {
                    if let url = s.resultURL { share(url: url) }
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundStyle(Theme.Colors.accentSoft)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .frame(minWidth: 44, minHeight: 44)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .glassCard(cornerRadius: 18, interactive: false)
        }
        .onChange(of: s.resultURL) { _, new in
            if let url = new { try? player.load(url: url) }
        }
        .onDisappear { player.stop() }
    }

    private func togglePlay() {
        if player.isPlaying { player.pause() }
        else { player.play() }
    }

    /// 真正的系统分享面板（此前是空占位 → 点分享无反应）
    private func share(url: URL) {
        let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = scene.windows.first?.rootViewController else { return }
        // iPad popover 必须指定锚点
        if let pop = av.popoverPresentationController {
            pop.sourceView = root.view
            pop.sourceRect = CGRect(x: root.view.bounds.midX, y: root.view.bounds.midY,
                                    width: 0, height: 0)
            pop.permittedArrowDirections = []
        }
        root.present(av, animated: true)
    }
}