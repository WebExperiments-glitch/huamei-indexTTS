import SwiftUI

/// 主屏（Root）—— 三个区域：克隆音频 + 文本 + 合成/播放
/// Internal Beta 1
struct RootView: View {

    @EnvironmentObject private var s: SessionStore
    @EnvironmentObject private var engine: InferenceEngine
    @State private var showAudioSheet: Bool  = false
    @State private var showSettingsSheet: Bool = false
    @State private var showAbout: Bool       = false
    @State private var showImportFolder: Bool = false
    @State private var showImportZip: Bool = false
    @State private var isImporting: Bool     = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    ModelStatusView(onImportFolder: { showImportFolder = true },
                                    isImporting: isImporting)
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
            .navigationTitle("声库克隆")
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
        // 模型导入：选择整个模型文件夹（推荐，一次搞定）
        .fileImporter(isPresented: $showImportFolder,
                      allowedContentTypes: [.folder],
                      allowsMultipleSelection: false) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            isImporting = true
            Task {
                try? await ModelImporter.importFolder(from: url,
                                                      manifest: ModelDownloadManager.loadManifest())
                await engine.refreshAfterImport()
                isImporting = false
            }
        }
        // 模型导入：选择一个模型 ZIP，自动解压（最省事）
        .fileImporter(isPresented: $showImportZip,
                      allowedContentTypes: [.zip],
                      allowsMultipleSelection: false) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            isImporting = true
            Task {
                try? await ModelImporter.importZip(from: url)
                await engine.refreshAfterImport()
                isImporting = false
            }
        }
    }
}

#Preview {
    RootView()
        .environmentObject(SessionStore())
        .environmentObject(InferenceEngine())
}