import SwiftUI
import UniformTypeIdentifiers

/// 4 选项音源选择 Sheet（核心交互）
struct AudioSourceSheet: View {

    /// 文件选择路由（统一走 UIKit DocumentPicker，规避 fileImporter 弹不出的系统问题）
    private enum FilePick: String, Identifiable {
        case audio, video, files
        var id: String { rawValue }
    }

    @EnvironmentObject private var s: SessionStore
    @Environment(\.dismiss)        private var dismiss
    @State private var picker: FilePick?
    @State private var recorder = AudioRecorder()

    var body: some View {
        VStack(spacing: 12) {
            Capsule().fill(Theme.Colors.divider).frame(width: 38, height: 4)
                .padding(.top, 8)
            Text("声音来源")
                .font(Theme.Fonts.title)
                .foregroundStyle(Theme.Colors.labelPrimary)
                .padding(.bottom, 4)

            VStack(spacing: 10) {
                sourceRow(icon: "mic.fill",
                          title: "应用内录音",
                          subtitle: recorder.isRecording ? "录音中…" : (recorder.status ?? "可直接录制您的嗓音"),
                          tint: Theme.Colors.accent,
                          disabled: recorder.isRecording || recorder.status != nil) {
                    startRecording()
                }
                sourceRow(icon: "music.note",
                          title: "上传音频文件",
                          subtitle: "MP3 · WAV · M4A · FLAC",
                          tint: Theme.Colors.accentSoft) {
                    picker = .audio
                }
                sourceRow(icon: "film",
                          title: "上传视频文件",
                          subtitle: "MP4 · MOV · 自动提取音轨",
                          tint: Theme.Colors.accentSoft) {
                    picker = .video
                }
                sourceRow(icon: "tray.and.arrow.down",
                          title: "从文件 / iCloud 选择",
                          subtitle: "可浏览任意位置",
                          tint: Theme.Colors.accentSoft) {
                    picker = .files
                }
            }
            .padding(.horizontal, 16)

            if recorder.isRecording {
                recordingBar
                    .padding(.horizontal, 16)
                    .padding(.top, 4)
            }

            Button(role: .cancel, action: { dismiss() }) {
                Text("取消").font(Theme.Fonts.callout)
            }
            .padding(.top, 4)
        }
        .padding(.bottom, 28)
        .background(Theme.Colors.canvas.ignoresSafeArea())
        // 统一 UIKit 文档选择器（音频 / 视频 / 任意文件）
        .sheet(item: $picker) { kind in
            DocumentPicker(allowFolders: false) { urls in
                handlePick(kind, urls)
            }
            .ignoresSafeArea()
        }
        .alert("录音失败",
               isPresented: .init(get: { recorder.error != nil },
                                  set: { if !$0 { recorder.error = nil } })) {
            Button("好") { recorder.error = nil }
        } message: { Text(recorder.error ?? "") }
    }

    // ---- UI 块 ----

    private func sourceRow(icon: String, title: String, subtitle: String,
                           tint: Color, disabled: Bool = false,
                           action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(tint.opacity(0.18))
                        .frame(width: 46, height: 46)
                    Image(systemName: icon)
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(tint)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(Theme.Fonts.callout)
                        .foregroundStyle(Theme.Colors.labelPrimary)
                    Text(subtitle).font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Colors.labelTertiary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Colors.labelTertiary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .opacity(disabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .glassCard(cornerRadius: 16)
        .disabled(disabled)
    }

    // 选择完成：导入到沙盒并按类型处理
    private func handlePick(_ kind: FilePick, _ urls: [URL]) {
        picker = nil
        guard let url = urls.first else { return }
        let result: Result<[URL], Error> = .success([url])
        switch kind {
        case .audio, .files:
            AudioImporter.handle(result: result) { s.referenceURL = $0 }
        case .video:
            handleVideo(result)
        }
    }

    private var recordingBar: some View {
        HStack(spacing: 12) {
            Circle().fill(Theme.Colors.danger).frame(width: 10, height: 10)
                .opacity(0.9)
                .scaleEffect(recorder.isRecording ? 1.2 : 1.0)
                .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                           value: recorder.isRecording)
            Text(formatTime(recorder.elapsed))
                .font(Theme.Fonts.mono)
                .foregroundStyle(Theme.Colors.danger)
            Spacer()
            Button {
                recorder.stop()
                if let url = recorder.lastURL { s.referenceURL = url; dismiss() }
            } label: {
                Text("停止")
                    .font(Theme.Fonts.callout)
                    .foregroundStyle(Theme.Colors.labelInverse)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(Theme.Colors.danger, in: Capsule())
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .glassCard(cornerRadius: 14, style: .subtle)
    }

    private func startRecording() {
        Task {
            do { try await recorder.start() }
            catch { recorder.error = error.localizedDescription }
        }
    }

    private func handleVideo(_ result: Result<[URL], Error>) {
        AudioImporter.handle(result: result) { url in
            // MP4 → 自动提取音频
            Task {
                if let extracted = try? await VideoAudioExtractor.extractAudio(from: url) {
                    s.referenceURL = extracted
                } else {
                    s.referenceURL = url   // 退化：保留原文件
                }
                dismiss()
            }
        }
    }

    private func formatTime(_ t: TimeInterval) -> String {
        let m = Int(t) / 60, s = Int(t) % 60
        return String(format: "%02d:%02d", m, s)
    }
}