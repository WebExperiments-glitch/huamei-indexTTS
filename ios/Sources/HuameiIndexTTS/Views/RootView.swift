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
    @State private var importError: String? = nil

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
        // 模型导入：选择整个模型文件夹（推荐，一次搞定）——用 UIKit 选择器（fileImporter 在部分系统上弹不出）
        .sheet(isPresented: $showImportFolder) {
            DocumentPicker(allowFolders: true) { urls in
                guard let url = urls.first else { return }
                isImporting = true
                Task {
                    defer { isImporting = false }
                    do {
                        try await ModelImporter.importFolder(from: url,
                                                             manifest: ModelDownloadManager.loadManifest())
                        await engine.refreshAfterImport()
                    } catch {
                        importError = "模型导入失败：\(error.localizedDescription)"
                    }
                }
            }
            .ignoresSafeArea()
        }
        // 模型导入：选择一个模型 ZIP，自动解压（最省事）
        .sheet(isPresented: $showImportZip) {
            DocumentPicker(allowFolders: false) { urls in
                guard let url = urls.first else { return }
                isImporting = true
                Task {
                    defer { isImporting = false }
                    do {
                        try await ModelImporter.importZip(from: url)
                        await engine.refreshAfterImport()
                    } catch {
                        importError = "模型导入失败：\(error.localizedDescription)"
                    }
                }
            }
            .ignoresSafeArea()
        }
        .alert("导入模型出错", isPresented: .init(get: { importError != nil },
                                               set: { if !$0 { importError = nil } })) {
            Button("好") { importError = nil }
        } message: {
            Text(importError ?? "")
        }
    }
}

#Preview {
    RootView()
        .environmentObject(SessionStore())
        .environmentObject(InferenceEngine())
}