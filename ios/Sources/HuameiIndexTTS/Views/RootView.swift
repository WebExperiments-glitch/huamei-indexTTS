import SwiftUI

/// 主屏（Root）—— 三个区域：克隆音频 + 文本 + 合成/播放
/// Internal Beta 1
struct RootView: View {

    @Environment(SessionStore.self) private var s
    @State private var showAudioSheet: Bool  = false
    @State private var showSettingsSheet: Bool = false
    @State private var showAbout: Bool       = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    ModelStatusView()
                    VoiceReferenceCard(onPick: { showAudioSheet = true })
                    TextInputCard()
                    SynthesizeControl()
                    if s.resultURL != nil { ResultCard() }
                    Spacer(minLength: 12)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 28)
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Theme.Colors.canvas.ignoresSafeArea())
            .navigationTitle("Huamei IndexTTS")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettingsSheet = true } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button { showAbout = true } label: {
                        Image(systemName: "info.circle")
                    }
                }
            }
        }
        .sheet(isPresented: $showAudioSheet) {
            AudioSourceSheet()
        }
        .sheet(isPresented: $showSettingsSheet) {
            SettingsSheet()
        }
        .sheet(isPresented: $showAbout) {
            AboutSheet()
        }
    }
}

#Preview {
    RootView()
        .environment(SessionStore())
}