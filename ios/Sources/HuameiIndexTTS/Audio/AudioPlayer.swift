import Foundation
import AVFoundation

/// 简单播放外壳（AVAudioPlayer）
@Observable
@MainActor
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {

    private(set) var isPlaying = false
    private var player: AVAudioPlayer?

    func load(url: URL) throws {
        let p = try AVAudioPlayer(contentsOf: url)
        p.delegate = self
        p.prepareToPlay()
        player = p
        isPlaying = false
    }

    func play() {
        guard let p = player else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        if p.play() { isPlaying = true }
    }

    func pause() {
        player?.pause()
        isPlaying = false
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.isPlaying = false }
    }
}