import SwiftUI
import UniformTypeIdentifiers

/// 4 选项音源选择 Sheet（核心交互）
struct AudioSourceSheet: View {

    @Environment(SessionStore.self) private var s
    @Environment(\.dismiss)        private var dismiss
    @State private var showFilePicker = false
    @State private var showVideoPicker = false
    @State private var showFilesPicker = false
    @State private var recorder = AudioRecorder()

    var body: some View {
        VStack(spacing: 12) {
            Capsule().fill(Theme.Colors.divider).frame(width: 38, height: 4)
                .padding(.top, 8)
            Text("Voice Source")
                .font(Theme.Fonts.title)
                .foregroundStyle(Theme.Colors.labelPrimary)
                .padding(.bottom, 4)

            VStack(spacing: 10) {
                sourceRow(icon: "mic.fill",
                          title: "Record in app",
                          subtitle: "Save as M4A / AAC  · compatible with any MP3 player",
                          tint: Theme.Colors.accent,
                          disabled: recorder.isRecording) {
                    startRecording()
                }
                sourceRow(icon: "music.note",
                          title: "Upload audio file",
                          subtitle: "MP3 · WAV · M4A · FLAC",
                          tint: Theme.Colors.accentSoft) {
                    showFilePicker = true
                }
                sourceRow(icon: "film",
                          title: "Upload video file",
                          subtitle: "MP4 · MOV · audio track auto-extracted",
                          tint: Theme.Colors.accentSoft) {
                    showVideoPicker = true
                }
                sourceRow(icon: "folder",
                          title: "Pick from Files / iCloud Drive",
                          subtitle: "Browse any location",
                          tint: Theme.Colors.accentSoft) {
                    showFilesPicker = true
                }
            }
            .padding(.horizontal, 16)

            if recorder.isRecording {
                recordingBar
                    .padding(.horizontal, 16)
                    .padding(.top, 4)
            }

            Button(role: .cancel, action: { dismiss() }) {
                Text("Cancel").font(Theme.Fonts.callout)
            }
            .padding(.top, 4)
        }
        .padding(.bottom, 28)
        .background(Theme.Colors.canvas.ignoresSafeArea())
        .fileImporter(isPresented: $showFilePicker,
                      allowedContentTypes: AudioImporter.audioTypes,
                      allowsMultipleSelection: false) { result in
            AudioImporter.handle(result: result) { s.referenceURL = $0 }
        }
        .fileImporter(isPresented: $showVideoPicker,
                      allowedContentTypes: AudioImporter.videoTypes,
                      allowsMultipleSelection: false) { result in
            handleVideo(result)
        }
        .fileImporter(isPresented: $showFilesPicker,
                      allowedContentTypes: [.audio, .video, .data],
                      allowsMultipleSelection: false) { result in
            AudioImporter.handle(result: result) { s.referenceURL = $0 }
        }
        .alert("Recording failed",
               isPresented: .init(get: { recorder.error != nil },
                                  set: { if !$0 { recorder.error = nil } })) {
            Button("OK") { recorder.error = nil }
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
                Text("Stop")
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