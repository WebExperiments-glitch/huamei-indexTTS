import SwiftUI

/// 合成结果播放卡
struct ResultCard: View {

    @EnvironmentObject private var s: SessionStore
    @State private var player = AudioPlayer()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Synthesis Result")
                    .font(Theme.Fonts.heading)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                Spacer()
            }

            HStack(spacing: 12) {
                Button(action: togglePlay) {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44, weight: .regular))
                        .foregroundStyle(Theme.Colors.accent)
                }
                .buttonStyle(.plain)

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
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .glassCard(cornerRadius: 18)
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

    private func share(url: URL) {
        // 占位：实际可接 UIActivityViewController 分享
        #if canImport(UIKit)
        // TODO: UI share sheet
        _ = url
        #endif
    }
}