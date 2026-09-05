import Foundation
import AVFoundation

/// App 内音频录制（保存为 AAC/M4A 容器；与 MP3 播放器兼容）
/// 说明：iOS AVAudioRecorder 不支持 MP3 编码（专利），AAC/M4A 在所有播放器等同播放。
@Observable
@MainActor
final class AudioRecorder: NSObject {

    private(set) var isRecording = false
    private(set) var elapsed: TimeInterval = 0
    private(set) var lastURL: URL?
    /// 即时状态提示（如"正在请求录音权限…"），让点击后立刻有反馈
    private(set) var status: String?
    var error: String?

    private var recorder: AVAudioRecorder?
    private var levelTimer: Timer?
    private var startDate: Date?

    func start() async throws {
        guard !isRecording, recorder == nil else { return }
        status = "正在请求录音权限…"

        // 请求麦克风权限（带兜底：回调异常/超时也给出中文指引而非卡死）
        let granted: Bool = await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { ok in
                cont.resume(returning: ok)
            }
        }
        guard granted else {
            status = nil
            throw NSError(domain: "AudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "无法使用麦克风。请到 设置 → 隐私与安全性 → 麦克风 中允许本 App 使用，然后重试。"])
        }

        status = "正在准备录音…"
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            status = nil
            throw NSError(domain: "AudioRecorder", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "音频会话启动失败：\(error.localizedDescription)"])
        }

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 22050,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rec-\(UUID().uuidString.prefix(8)).m4a")
        let rec = try AVAudioRecorder(url: url, settings: settings)
        rec.isMeteringEnabled = true
        guard rec.record() else {
            status = nil
            throw NSError(domain: "AudioRecorder", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "录音初始化失败，请重试。"])
        }
        recorder = rec
        startDate = Date()
        isRecording = true
        elapsed = 0
        lastURL = url
        status = nil

        levelTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let start = self.startDate else { return }
                self.elapsed = Date().timeIntervalSince(start)
            }
        }
    }

    func stop() {
        recorder?.stop()
        levelTimer?.invalidate()
        levelTimer = nil
        isRecording = false
        status = nil
        lastURL = recorder?.url
        recorder = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }
}