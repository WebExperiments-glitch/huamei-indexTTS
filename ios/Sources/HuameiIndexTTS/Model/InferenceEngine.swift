import Foundation
import AVFoundation
import MLXIndexTTS2Core

/// 推断引擎（Stage 1：接 MLXIndexTTS2Core 真实推理）
///
/// 模型目录约定（Documents/huamei-models/，首次启动前由用户在 Mac 上准备好）：
///   gpt.safetensors · codec.safetensors · s2mel.safetensors · bigvgan.safetensors
///   multilingual_zh_ja_yue_char_del.tiktoken · specials.json
///   feat1.json · feat2.json（73×192 / 73×1280）
///   prompt_<row>.json · refmel_<row>.json（A1 预计算说话人条件束，可选）
@Observable
@MainActor
final class InferenceEngine {

    enum State: Equatable {
        case uninitialized
        case missingModel
        case loading
        case ready
        case failed(String)
    }

    private(set) var state: State = .uninitialized
    private var pipeline: TTSPipeline?

    static var modelDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("huamei-models", isDirectory: true)
    }

    static var modelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir.path)
    }

    func prepareIfNeeded() async {
        guard state == .uninitialized || state == .missingModel else { return }
        guard Self.modelAvailable else {
            state = .missingModel
            return
        }
        state = .loading
        do {
            let dir = Self.modelDir
            // 大 I/O（4 个 safetensors + 词典）放到后台执行，避免卡主线程
            let pipe = try await Task.detached(priority: .userInitiated) {
                try TTSPipeline(modelDir: dir)
            }.value
            pipeline = pipe
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    // MARK: - 合成

    /// 异步合成（preset 音色模式：feat1 行 speakerRow；prompt 条件可选 bundle）
    /// - Returns: wav 文件 URL（Documents/huamei-output/）
    func synthesize(text: String,
                    languageId: Int,
                    speakerRow: Int,
                    emotionWeight: [Float],
                    config: SynthesizeConfig,
                    seed: UInt64,
                    onProgress: @escaping (String, Double) -> Void) async throws -> URL {

        guard state == .ready, let pipe = pipeline else {
            throw NSError(domain: "InferenceEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "model not ready"])
        }
        // 运行在后台线程，避免卡 UI
        return try await Task.detached(priority: .userInitiated) {
            let floats = try pipe.synthesize(
                text: text,
                langId: languageId,
                speakerRow: speakerRow,
                emoWeight: emotionWeight,
                config: config,
                seed: seed,
                prompt: nil,            // TODO: 从模型目录加载 A1 条件束（见 README）
                onStage: { stage, frac in
                    Task { @MainActor in onProgress(stage, frac) }
                }
            )
            let url = try Self.writeWav(floats, sampleRate: TTSConfig.sampleRate)
            return url
        }.value
    }

    // MARK: - WAV 写出

    private static func writeWav(_ samples: [Float], sampleRate: Int) throws -> URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huamei-output", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("out-\(UUID().uuidString.prefix(8)).wav")

        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(sampleRate),
                                channels: 1, interleaved: false)!
        guard let file = try? AVAudioFile(forWriting: url, settings: fmt.settings) else {
            throw NSError(domain: "WAV", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot create wav"])
        }
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count))!
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buf.floatChannelData?[0] {
            for (i, s) in samples.enumerated() { ch[i] = s }
        }
        try file.write(from: buf)
        return url
    }
}