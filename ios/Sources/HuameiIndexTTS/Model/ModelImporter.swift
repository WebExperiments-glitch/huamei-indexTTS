import Foundation
import UniformTypeIdentifiers

/// 模型导入器：让用户从「文件」App 直接提供模型（替代网络下载）
///
/// 两种方式：
///   1. 选择整个模型文件夹（推荐）——把模型文件按目录结构整理好，
///      放进 iPhone 的「文件」App（例如从电脑传过来），一键导入。
///   2. 多选模型文件 —— 按文件名自动归类放置（支持嵌套子目录条目）。
final class ModelImporter {

    static var modelDir: URL { InferenceEngine.modelDir }

    /// 文件选择器可选的类型（safetensors 无系统类型，用 .data 兜底）
    static let fileTypes: [UTType] = [.data, .json, .yaml, .text, .item]

    /// 导入文件夹：递归复制整个目录结构到 Documents/huamei-models/
    static func importFolder(from url: URL, manifest: ModelManifest?) async throws {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try copyRecursively(from: url, to: modelDir)
    }

    /// 多选文件导入：按 Manifest 里 path 的最后一段匹配放置
    static func importFiles(_ urls: [URL], manifest: ModelManifest?) async throws {
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }

            let name = url.lastPathComponent
            if let m = manifest {
                let matches = m.files.filter { ($0.path as NSString).lastPathComponent == name }
                if matches.count == 1, let entry = matches.first {
                    let dest = modelDir.appendingPathComponent(entry.path)
                    try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                            withIntermediateDirectories: true)
                    try? FileManager.default.removeItem(at: dest)
                    try FileManager.default.copyItem(at: url, to: dest)
                    continue
                }
            }
            // 未匹配（或名字重复）→ 放根目录
            let dest = modelDir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: url, to: dest)
        }
    }

    // MARK: - 递归复制

    private static func copyRecursively(from src: URL, to dst: URL) throws {
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let items = try FileManager.default.contentsOfDirectory(at: src, includingPropertiesForKeys: nil)
        for item in items {
            if item.lastPathComponent.hasPrefix(".") { continue }   // 跳过隐藏文件
            let isDir = (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            if isDir {
                let sub = dst.appendingPathComponent(item.lastPathComponent, isDirectory: true)
                try copyRecursively(from: item, to: sub)
            } else {
                let dest = dst.appendingPathComponent(item.lastPathComponent)
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.copyItem(at: item, to: dest)
            }
        }
    }
}