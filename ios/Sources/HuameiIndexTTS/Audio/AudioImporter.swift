import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// 文件导入统一封装（处理 fileImporter 结果 + 安全作用域）
enum AudioImporter {

    static var audioTypes: [UTType] {
        var t: [UTType] = [.audio]
        t.append(contentsOf: [.mp3, .wav, .mpeg4Audio, .aiff].compactMap { $0 })
        return t
    }

    static var videoTypes: [UTType] {
        [.movie, .video, .mpeg4Movie, .quickTimeMovie].compactMap { $0 }
    }

    /// 处理 fileImporter Result → 用户所选 URL
    static func handle(result: Result<[URL], Error>,
                       onPicked: @escaping (URL) -> Void) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            // 开始访问安全作用域
            _ = url.startAccessingSecurityScopedResource()
            onPicked(url)
        case .failure(let err):
            // 主进程不展示：仅日志
            print("[AudioImporter] picker error: \(err.localizedDescription)")
        }
    }
}