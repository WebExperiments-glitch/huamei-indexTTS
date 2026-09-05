import Foundation
import CryptoKit
import SwiftUI
import Combine

/// 模型清单：App 内按需下载的唯一数据源（打包进 Bundle）
struct ModelManifest: Codable {
    struct Entry: Codable, Identifiable, Equatable {
        let path: String      // 相对 Documents/huamei-models/ 的路径
        let group: String     // "synthesis"（基础）| "clone"（克隆组件）
        let size: Int
        let sha256: String
        var id: String { path }
    }
    let modelscope: String
    let baseURL: String
    let totalBytes: Int
    let files: [Entry]
}

// MARK: - 傻瓜式模型下载器

/// 零门槛：缺模型就自动从魔搭下载（免登录、进度、断点重试、sha256 校验）。
/// 使用方只需一个入口：`ModelDownloadManager.download(group:)`。
final class ModelDownloadManager: ObservableObject {

    enum State: Equatable {
        case idle                // 未开始
        case downloading(fraction: Double, currentFile: String)
        case done
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    private var manifest: ModelManifest?
    private var totalBytes: Int64 = 0
    private var doneBytes: Int64 = 0

    static let modelDir: URL = InferenceEngine.modelDir

    init() {
        manifest = Self.loadManifest()
    }

    static func loadManifest() -> ModelManifest? {
        guard let url = Bundle.main.url(forResource: "ModelManifest", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase   // base_url → baseURL
        return try? decoder.decode(ModelManifest.self, from: data)
    }

    /// 指定组是否已就位
    static func groupReady(_ group: String) -> Bool {
        guard let m = loadManifest() else { return false }
        let fs = m.files.filter { $0.group == group }
        return !fs.isEmpty && fs.allSatisfy { entry in
            let url = modelDir.appendingPathComponent(entry.path)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = attrs[.size] as? Int, size == entry.size else { return false }
            return true
        }
    }

    /// 一键下载（synthesis 默认；clone 组件首次使用时再调）
    func download(group: String = "synthesis") {
        guard let m = manifest else {
            state = .failed("模型清单缺失，请更新 App")
            return
        }
        let entries = m.files.filter { $0.group == group && !Self.fileReady($0) }
        guard !entries.isEmpty else {
            state = .done
            return
        }
        totalBytes = Int64(entries.reduce(0) { $0 + $1.size })
        doneBytes = 0
        Task { await runQueue(entries, base: m.baseURL) }
    }

    private static func fileReady(_ e: ModelManifest.Entry) -> Bool {
        let url = modelDir.appendingPathComponent(e.path)
        guard let a = try? FileManager.default.attributesOfItem(atPath: url.path),
              let sz = a[.size] as? Int, sz == e.size else { return false }
        // 大小吻合即视为完成；首包无校验失败路径，重试目录里清理坏文件后重下
        return true
    }

    /// 顺序下载（每文件独立 dataTask，写临时文件 → 校验 sha256 → 原子改名）
    private func runQueue(_ entries: [ModelManifest.Entry], base: String) async {
        for entry in entries {
            var remaining = 1   // 每文件最多重试次数
            while remaining >= 0 {
                do {
                    try await downloadOne(entry, base: base, index: entries.firstIndex(of: entry) ?? 0, total: entries.count)
                    break
                } catch {
                    remaining -= 1
                    if remaining < 0 {
                        state = .failed("下载失败：\(entry.path)\n\(error.localizedDescription)")
                        return
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
        }
        state = .done
    }

    private func downloadOne(_ entry: ModelManifest.Entry, base: String, index: Int, total: Int) async throws {
        let dest = Self.modelDir.appendingPathComponent(entry.path)
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let part = dest.appendingPathExtension("part")
        try? FileManager.default.removeItem(at: part)

        let url = URL(string: base + entry.path)!
        if #available(iOS 15.0, *) {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw URLError(.badServerResponse)
            }
            try data.write(to: part)
        } else {
            // iOS 17 起可用上面的全新 API；兜底同步路径（本项目最低 iOS 17）
            let data = try Data(contentsOf: url)
            try data.write(to: part)
        }

        // 校验
        let hash = Data(try Data(contentsOf: part)).sha256Hex()
        guard hash == entry.sha256 else {
            try? FileManager.default.removeItem(at: part)
            throw URLError(.cannotDecodeRawData)
        }
        try FileManager.default.moveItem(at: part, to: dest)
        doneBytes += Int64(entry.size)
        state = .downloading(fraction: Double(doneBytes) / Double(max(totalBytes, 1)),
                             currentFile: entry.path)
    }

    /// 剩余可用空间（Documents 所在卷）
    static func freeSpace() -> Int64 {
        let keys: Set<URLResourceKey> = [.volumeAvailableCapacityForImportantUsageKey]
        guard let v = try? Self.modelDir.resourceValues(forKeys: keys),
              let c = v.volumeAvailableCapacityForImportantUsage else { return 0 }
        return c
    }

    func progressDescription() -> String {
        switch state {
        case .downloading(let f, let file):
            return String(format: "%.0f%% · %@", f * 100, (file as NSString).lastPathComponent)
        case .done:
            return "模型就绪"
        case .failed(let msg):
            return "下载失败：\(msg)"
        case .idle:
            return "需要下载模型"
        }
    }
}

extension Data {
    func sha256Hex() -> String {
        SHA256.hash(data: self).map { String(format: "%02x", $0) }.joined()
    }
}