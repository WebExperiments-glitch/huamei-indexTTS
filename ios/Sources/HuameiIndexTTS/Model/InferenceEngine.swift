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
            // withError：捕获 MLX 内部错误，避免 _mlx_error → fatalError 崩溃；错误可抛给 UI
            let floats = try MLX.withError {
                try pipe.synthesize(
                    text: text,
                    langId: languageId,
                    speakerRow: speakerRow,
                    emoWeight: emotionWeight,
                    config: config,
                    seed: seed,
                    prompt: nil,            // TODO: 从模型目录加载 A1 条件束（见 README）
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

    // MARK: - WAV 写出

    nonisolated private static func writeWav(_ samples: [Float], sampleRate: Int) throws -> URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huamei-output", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("out-\(UUID().uuidString.prefix(8)).wav")

        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(sampleRate),
                                channels: 1, interleaved: false)!
        guard let file = try? AVAudioFile(forWriting: url, settings: fmt.settings) else {
            throw NSError(domain: "WAV", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "无法创建音频文件"])
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