import Foundation
import AVFoundation
import Combine
import MLX
import MLXIndexTTS2Core

/// 推断引擎（Stage 1：接 MLXIndexTTS2Core 真实推理）
///
/// 模型目录约定（Documents/huamei-models/，首次启动前由用户在 Mac 上准备好）：
///   gpt.safetensors · codec.safetensors · s2mel.safetensors · bigvgan.safetensors
///   multilingual_zh_ja_yue_char_del.tiktoken · specials.json
///   feat1.json · feat2.json（73×192 / 73×1280）
///   prompt_<row>.json · refmel_<row>.json（A1 预计算说话人条件束，可选）
final class InferenceEngine: ObservableObject {

    private var cancellables = Set<AnyCancellable>()

    init() {
        // 上次（可能崩溃的）运行日志尾部 → 日志面板（重启后可见，flash 前已落盘）
        let tail = DLog.tail(40)
        if !tail.isEmpty {
            SystemMonitor.shared.appendLog("=== 上次运行记录（崩溃日志落盘） ===")
            tail.forEach { SystemMonitor.shared.appendLog($0) }
        }
        // 下载器状态变化 → 转发主对象变化（UI 观察 downloadState 刷新）+ 写入开发者日志
        downloader.$state
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                self?.objectWillChange.send()
                switch state {
                case .downloading(_, let file):
                    SystemMonitor.shared.appendLog("下载中：\((file as NSString).lastPathComponent)")
                case .done:
                    SystemMonitor.shared.appendLog("模型下载完成")
                case .failed(let msg):
                    SystemMonitor.shared.appendLog("下载失败：\(msg)")
                case .idle:
                    break
                }
            }
            .store(in: &cancellables)
    }

    enum State: Equatable {
        case uninitialized
        case missingModel
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var state: State = .uninitialized
    private var pipeline: TTSPipeline?

    /// 傻瓜式模型下载器（缺模型 → 一键下载 → 自动就绪）
    let downloader = ModelDownloadManager()

    /// 半透明给 UI：下载状态
    var downloadState: ModelDownloadManager.State { downloader.state }

    static var modelDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("huamei-models", isDirectory: true)
    }

    static var modelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir.path)
    }

    /// 自检关键小文件（tokenizer/feat/langs/config），输出实际大小与 sha256 是否匹配清单
    @MainActor
    static func verifySmallFiles() {
        guard let m = ModelDownloadManager.loadManifest() else {
            SystemMonitor.shared.appendLog("自检：模型清单读取失败")
            return
        }
        let small = m.files.filter { $0.size <= 5_000_000 }
        for entry in small {
            let f = modelDir.appendingPathComponent(entry.path)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: f.path),
                  let size = attrs[.size] as? Int else {
                SystemMonitor.shared.appendLog("自检缺失：\(entry.path)")
                continue
            }
            let hash = ModelDownloadManager.sha256(url: f)
            let ok = (size == entry.size) && (hash == entry.sha256)
            SystemMonitor.shared.appendLog("自检\(ok ? "✔" : "✘")：\(entry.path)  \(size)B \(ok ? "" : "sha=\(hash.prefix(16))…")")
        }
    }

    /// 一键下载（main-actor）：缺模型时的总入口。下载完成自动 prepare。
    func startModelDownload() {
        guard state == .missingModel else { return }
        downloader.download(group: "synthesis")
        Task { @MainActor [weak self] in
            guard let self else { return }
            while true {
                if case .done = downloader.state {
                    self.state = .loading
                    await self.prepareIfNeeded()
                    return
                }
                if case .failed = downloader.state {
                    self.state = .missingModel
                    return
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }

    /// 用户手动导入模型后刷新状态（替代网络下载）
    @MainActor
    func refreshAfterImport() async {
        SystemMonitor.shared.appendLog("导入完成，开始自检关键文件…")
        Self.verifySmallFiles()
        SystemMonitor.shared.appendLog("导入完成，开始加载模型…")
        if Self.modelAvailable {
            state = .loading
            do {
                let dir = Self.modelDir
                let pipe = try await Task.detached(priority: .userInitiated) {
                    try TTSPipeline(modelDir: dir)
                }.value
                pipeline = pipe
                state = .ready
                SystemMonitor.shared.appendLog("模型加载成功，可以开始合成了")
            } catch {
                // 文件不完整/缺关键件 → 保持缺失，提示继续导入
                state = .missingModel
                SystemMonitor.shared.appendLog("模型加载失败：\(error.localizedDescription)")
            }
        } else {
            state = .missingModel
            SystemMonitor.shared.appendLog("未找到模型目录，请先导入模型")
        }
    }

    func prepareIfNeeded() async {
        guard state == .uninitialized || state == .missingModel else { return }
        guard Self.modelAvailable else {
            state = .missingModel
            return
        }
        state = .loading
        SystemMonitor.shared.appendLog("开始加载模型…")
        do {
            let dir = Self.modelDir
            // 大 I/O（4 个 safetensors + 词典）放到后台执行，避免卡主线程
            let pipe = try await Task.detached(priority: .userInitiated) {
                try TTSPipeline(modelDir: dir)
            }.value
            pipeline = pipe
            state = .ready
            SystemMonitor.shared.appendLog("模型就绪")
        } catch {
            state = .failed(error.localizedDescription)
            SystemMonitor.shared.appendLog("模型加载失败：\(error.localizedDescription)")
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
                    referenceURL: URL?,
                    onProgress: @escaping (String, Double) -> Void) async throws -> URL {

        guard state == .ready, let pipe = pipeline else {
            throw NSError(domain: "InferenceEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "model not ready"])
        }
        // 运行在后台线程，避免卡 UI
        return try await Task.detached(priority: .userInitiated) {
            DLog.reset()                       // 清空上次记录，开始新一轮
            DLog.write("ENGINE synthesize start")
            SystemMonitor.shared.appendLog("开始合成推理…")
            // A2：参考音频 → 声纹（style + prompt 束）；缺参考音频/克隆组件时回退预设音色
            var styleOverride: [Float]?
            var promptBundle: PromptBundle?
            if let ref = referenceURL {
                do {
                    let pcm = try Self.decodeToPCM16k(url: ref)
                    let extr = VoiceExtractor(paths: Self.clonePaths)
                    let vb = try extr.voiceprint(audioURL: ref, pcm16k: pcm)
                    styleOverride = vb.style
                    promptBundle = extr.promptBundle(vb)
                    DLog.write("A2 voiceprint style=\(vb.style.count) promptFrames=\(vb.melFrames) src=\(vb.sourceName)")
                    SystemMonitor.shared.appendLog("已提取克隆声纹：\(vb.sourceName)（prompt \(vb.melFrames) 帧）")
                } catch {
                    SystemMonitor.shared.appendLog("克隆声纹提取失败，回退预设：\(error.localizedDescription)")
                    DLog.write("A2 voiceprint FAIL: \(String(describing: error))")
                    styleOverride = nil; promptBundle = nil
                }
            } else {
                DLog.write("A2 无参考音频，走预设音色")
            }
            // withError：捕获 MLX 内部错误，避免 _mlx_error → fatalError 崩溃；错误可抛给 UI
            let floats = try MLX.withError {
                try pipe.synthesize(
                    text: text,
                    langId: languageId,
                    speakerRow: speakerRow,
                    emoWeight: emotionWeight,
                    config: config,
                    seed: seed,
                    prompt: promptBundle,            // A2：参考音频条件束
                    styleOverride: styleOverride,    // A2：克隆声纹（192）
                    onStage: { stage, frac in
                        // 阶段进度写入日志面板（崩溃前用户可看到最后到达的阶段）
                        let pct = Int(frac * 100)
                        if stage == "s2mel" {
                            SystemMonitor.shared.appendLog("s2mel 扩散 \(pct)%…")
                        } else if pct % 25 == 0 || stage == "done" {
                            SystemMonitor.shared.appendLog("阶段 \(stage) \(pct)%…")
                        }
                        Task { @MainActor in onProgress(stage, frac) }
                    }
                )
            }
            let url = try Self.writeWav(floats, sampleRate: TTSConfig.sampleRate)
            SystemMonitor.shared.appendLog("合成完成，已生成音频")
            return url
        }.value
    }

    // MARK: - A2 克隆辅助

    /// A2 克隆组件路径（模型目录内 clone 组；s2mel 复用 synthesis 组）
    private static var clonePaths: VoiceExtractor.ComponentPaths {
        let dir = modelDir
        return VoiceExtractor.ComponentPaths(
            w2vBert: dir.appendingPathComponent("w2v-bert-2.0/model.safetensors").path,
            campplus: dir.appendingPathComponent("campplus_cn_common.safetensors").path,
            s2mel: dir.appendingPathComponent("s2mel.safetensors").path,
            statsJSON: dir.appendingPathComponent("wav2vec2bert_stats.json").path
        )
    }

    /// URL → 16kHz mono PCM（AVFoundation 解码 → 通道平均 → 重采样）
    nonisolated private static func decodeToPCM16k(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let fmt = file.processingFormat
        let cap = AVAudioFrameCount(file.length)
        guard cap > 0, let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap) else {
            throw NSError(domain: "InferenceEngine", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "无法解码参考音频"])
        }
        try file.read(into: buf)
        let frames = Int(buf.frameLength)
        let ch = Int(fmt.channelCount)
        guard frames > 0, ch > 0, let data = buf.floatChannelData else {
            throw NSError(domain: "InferenceEngine", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "参考音频为空或解码失败"])
        }
        var mono = [Float](repeating: 0, count: frames)
        for i in 0..<frames {
            var s: Float = 0
            for c in 0..<ch { s += data[c][i] }
            mono[i] = s / Float(ch)
        }
        let src = Int(fmt.sampleRate)
        return AudioFeatures.resample(mono, from: src, to: 16000)
    }

    // MARK: - WAV 写出

    nonisolated private static func writeWav(_ samples: [Float], sampleRate: Int) throws -> URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huamei-output", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("out-\(UUID().uuidString.prefix(8)).wav")

        // ⚠️ 手写 16-bit PCM WAV：不依赖 AVAudioFormat/AVAudioFile（真机偶发返回 nil/失败 → 崩溃），
        //    标准 RIFF 头 + 数据，AVAudioPlayer/系统播放器 100% 兼容。
        let count = samples.count
        let dataSize = count * 2
        var d = Data(capacity: 44 + dataSize)
        func str(_ s: String) { d.append(contentsOf: Array(s.utf8)) }
        func le16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        str("RIFF"); le32(UInt32(36 + dataSize)); str("WAVE")
        str("fmt "); le32(16)
        le16(1)                              // PCM
        le16(1)                              // mono
        le32(UInt32(sampleRate))             // sample rate
        le32(UInt32(sampleRate) * 2)         // byte rate
        le16(2)                              // block align
        le16(16)                             // bits per sample
        str("data"); le32(UInt32(dataSize))
        for s in samples {
            let v: Double = s.isFinite ? Double(s) : 0
            let clamped = max(-1.0, min(1.0, v))
            le16(UInt16(bitPattern: Int16(clamping: Int(clamped * 32767.0))))
        }
        try d.write(to: url)
        return url
    }
}