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
    var error: String?

    private var recorder: AVAudioRecorder?
    private var levelTimer: Timer?
    private var startDate: Date?

    func start() async throws {
        // 请求麦克风权限
        let granted: Bool = await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { ok in cont.resume(returning: ok) }
        }
        guard granted else {
            throw NSError(domain: "AudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

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
        rec.record()
        recorder = rec
        startDate = Date()
        isRecording = true
        elapsed = 0
        lastURL = url

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
        lastURL = recorder?.url
        recorder = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }
}