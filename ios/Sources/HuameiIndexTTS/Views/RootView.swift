import SwiftUI

/// 主屏（Root）—— 三个区域：克隆音频 + 文本 + 合成/播放
/// Internal Beta 1
struct RootView: View {

    /// 统一 Sheet 路由（SwiftUI 多 sheet 同视图挂载有冲突，全部走这一个）
    private enum SheetKind: String, Identifiable {
        case audio, settings, about, importFolder, importZip
        var id: String { rawValue }
    }

    @EnvironmentObject private var s: SessionStore
    @EnvironmentObject private var engine: InferenceEngine
    @State private var activeSheet: SheetKind?
    @State private var isImporting: Bool     = false
    @State private var importError: String? = nil

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    ModelStatusView(onImportFolder: { activeSheet = .importFolder },
                                    onImportZip: { activeSheet = .importZip },
                                    isImporting: isImporting)
                    VoiceReferenceCard(onPick: { activeSheet = .audio })
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
                    Button { activeSheet = .settings } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button { activeSheet = .about } label: {
                        Image(systemName: "info.circle")
                    }
                }
            }
        }
        // 单一 Sheet 路由：audio / settings / about / 模型导入
        .sheet(item: $activeSheet) { kind in
            switch kind {
            case .audio:
                AudioSourceSheet()
            case .settings:
                SettingsSheet()
            case .about:
                AboutSheet()
            case .importFolder:
                DocumentPicker(allowFolders: true) { urls in
                    guard let url = urls.first else { return }
                    handleImport { try await ModelImporter.importFolder(from: url,
                                                                        manifest: ModelDownloadManager.loadManifest()) }
                }
                .ignoresSafeArea()
            case .importZip:
                DocumentPicker(allowFolders: false) { urls in
                    guard let url = urls.first else { return }
                    handleImport { try await ModelImporter.importZip(from: url) }
                }
                .ignoresSafeArea()
            }
        }
        .alert("导入模型出错", isPresented: .init(get: { importError != nil },
                                               set: { if !$0 { importError = nil } })) {
            Button("好") { importError = nil }
        } message: {
            Text(importError ?? "")
        }
    }

    /// 统一的模型导入处理（解压/复制 → 刷新引擎）
    private func handleImport(_ work: @escaping () async throws -> Void) {
        isImporting = true
        Task {
            defer { isImporting = false }
            do {
                try await work()
                await engine.refreshAfterImport()
            } catch {
                importError = "模型导入失败：\(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    RootView()
        .environmentObject(SessionStore())
        .environmentObject(InferenceEngine())
}