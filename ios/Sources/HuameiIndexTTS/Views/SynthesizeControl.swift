import SwiftUI
import MLXIndexTTS2Core

/// 合成按钮 + 进度条（合二为一）
struct SynthesizeControl: View {

    @EnvironmentObject private var s: SessionStore
    @EnvironmentObject private var engine: InferenceEngine
    @State private var task: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 12) {
            progressBar

            Button(action: tap) {
                HStack(spacing: 10) {
                    if isWorking {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(Theme.Colors.labelInverse)
                            .scaleEffect(0.9)
                    } else {
                        Image(systemName: "waveform")
                            .font(.system(size: 18, weight: .semibold))
                    }
                    Text(buttonTitle)
                        .font(Theme.Fonts.heading)
                }
                .frame(maxWidth: .infinity)
                .glassPrimaryButton(disabled: !s.canSynthesize)
            }
            .disabled(!s.canSynthesize)
            .buttonStyle(.plain)

            if let err = s.lastError {
                Text(err)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Colors.danger)
                    .multilineTextAlignment(.center)
            }
        }
    }

    // MARK: - 进度条

    private var progressBar: some View {
        Group {
            if isWorking || s.phase == .done {
                VStack(spacing: 6) {
                    ProgressView(value: s.progress)
                        .progressViewStyle(.linear)
                        .tint(Theme.Colors.accent)
                        .frame(height: 6)
                    HStack {
                        Text(phaseLabel)
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Colors.labelSecondary)
                        Spacer()
                        Text("\(Int(s.progress * 100))%")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Colors.labelSecondary)
                            .monospacedDigit()
                    }
                }
                .transition(.opacity)
            }
        }
    }

    // MARK: - 状态文案

    private var phaseLabel: String {
        switch s.phase {
        case .idle:    return "就绪"
        case .loading: return "加载模型…"
        case .encoding:return "编码文本…"
        case .gpt:     return "生成 Token…"
        case .codec:   return "解码 Codec…"
        case .s2mel:   return "扩散声谱…"
        case .vocoder: return "合成音频…"
        case .done:    return "完成"
        case .failed:  return "失败"
        }
    }

    private var buttonTitle: String {
        switch s.phase {
        case .idle, .done, .failed: return "开始合成"
        default:                    return "停止"
        }
    }

    private var isWorking: Bool {
        switch s.phase {
        case .loading, .encoding, .gpt, .codec, .s2mel, .vocoder: return true
        default: return false
        }
    }

    // MARK: - 行为

    private func tap() {
        if isWorking { cancel(); return }
        s.reset()
        s.phase = .loading
        s.progress = 0.01
        let cfg = SynthesizeConfig(
            durationFactor: s.durationFactor,
            temperature: s.temperature,
            topP: s.topP,
            topK: s.topK,
            repetitionPenalty: s.repetitionPenalty,
            maxMelTokens: s.maxMelTokens
        )
        task = Task {
            await runSynthesize(text: s.text,
                                languageId: s.language.langId,
                                config: cfg)
        }
    }

    private func cancel() {
        task?.cancel(); task = nil
        s.phase = .idle; s.progress = 0
    }

    private func runSynthesize(text: String, languageId: Int, config: SynthesizeConfig) async {
        do {
            let url = try await engine.synthesize(
                text: text,
                languageId: languageId,
                speakerRow: 0,
                emotionWeight: [1, 0, 0, 0, 0, 0, 0, 0],
                config: config,
                seed: 42,
                referenceURL: s.referenceURL,   // A2：有参考音频 → 克隆；否则预设
                onProgress: { label, frac in
                    Task { @MainActor in
                        s.progress = frac
                        s.phase = mapStage(label)
                    }
                }
            )
            await MainActor.run {
                s.resultURL = url
                s.phase = .done
            }
        } catch is CancellationError {
            await MainActor.run { s.phase = .idle; s.progress = 0 }
        } catch {
            let msg = error.localizedDescription
            await MainActor.run {
                s.phase = .failed
                s.lastError = msg
            }
            SystemMonitor.shared.appendLog("合成失败：\(msg)")
        }
    }

    private func mapStage(_ label: String) -> SessionStore.Phase {
        switch label {
        case "loading":   return .loading
        case "encoding":  return .encoding
        case "gpt":       return .gpt
        case "codec":     return .codec
        case "s2mel":     return .s2mel
        case "vocoder":   return .vocoder
        default:          return .idle
        }
    }
}