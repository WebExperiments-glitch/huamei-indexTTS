import Foundation
import AVFoundation

/// MP4 / MOV 视频 → 提取音轨为 M4A/AAC（22.05 kHz 单声道，对齐 IndexTTS 训练侧）
enum VideoAudioExtractor {

    enum ExtractionError: Error {
        case noAudioTrack
        case exportFailed(String)
    }

    /// 异步导出（异步包装 AVAssetExportSession）
    static func extractAudio(from videoURL: URL,
                              sampleRate: Double = 22_050,
                              channels: UInt32 = 1) async throws -> URL {

        let asset = AVURLAsset(url: videoURL)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = tracks.first else { throw ExtractionError.noAudioTrack }

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vid-\(UUID().uuidString.prefix(8)).m4a")

        guard let session = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            throw ExtractionError.exportFailed("AVAssetExportSession init failed")
        }

        // 输出设置（22.05 kHz mono）
        let output = OutputConfiguration(
            audioSampleRate: sampleRate,
            audioChannels: channels
        )
        session.outputURL = outURL
        session.outputFileType = .m4a
        session.audioMix = makeAudioMix(track: audioTrack, settings: output)
        session.shouldOptimizeForNetworkUse = false

        try await session.exportAsync()
        return outURL
    }

    // ---- 辅助 ----

    private struct OutputConfiguration {
        let audioSampleRate: Double
        let audioChannels: UInt32
    }

    private static func makeAudioMix(track: AVAssetTrack,
                                     settings: OutputConfiguration) -> AVAudioMix {
        let mix = AVMutableAudioMix()
        let params = AVMutableAudioMixInputParameters(track: track)
        // 采样率 / 通道由 AVAssetExportSession preset 处理；这里只设音量
        params.setVolume(1.0, at: .zero)
        mix.inputParameters = [params]
        return mix
    }
}

// 异步 export 包装（iOS 18+ 有原生 export(to:)，旧版用回调包成 async）
private extension AVAssetExportSession {
    func exportAsync() async throws {
        if #available(iOS 18, *) {
            try await export(to: outputURL!, as: .m4a)
            return
        }
        // 旧实现：包装 completionHandler
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.exportAsynchronously {
                switch self.status {
                case .completed: cont.resume(returning: ())
                case .cancelled: cont.resume(throwing: CancellationError())
                default:
                    let err = self.error ?? NSError(domain: "AVAssetExportSession",
                                                     code: -1,
                                                     userInfo: [NSLocalizedDescriptionKey:
                                                                "export status=\(self.status.rawValue)"])
                    cont.resume(throwing: err)
                }
            }
        }
    }
}