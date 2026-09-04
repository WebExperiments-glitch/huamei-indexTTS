import Foundation
import CryptoKit

/// 声纹（一次"随机音频即时克隆"的全部条件数据）
///
/// 与 73 组预设的关系：`style` 与 `feat1[speakerRow]`（[73,192] 的一行）**同构**——
/// 设备端提取出的声纹可以完全替代预设行进入 GPT 条件；prompt 束（A1 同格式）供 s2mel 锚定。
///
/// 体积：style 192 个 Float ≈ 几 KB → 缓存成本可忽略，同一段音频二次克隆 0 等待。
public struct Voiceprint: Codable, Sendable {

    /// 稳定指纹（缓存键）：sha256(文件名|字节数|修改时间) 前 16 hex。
    /// 同一文件内容不变 → 同一 id → 缓存命中；改动/换文件 → 新 id 自动重提。
    public let id: String

    /// 192 维说话人嵌入（w2v-bert 特征，与 feat1 行同构）
    public let style: [Float]

    public let sourceName: String
    public let createdAt: Date

    public init(id: String, style: [Float], sourceName: String, createdAt: Date = Date()) {
        self.id = id
        self.style = style
        self.sourceName = sourceName
        self.createdAt = createdAt
    }

    /// 从音频文件算稳定 id（不读内容，只读文件元数据 → 快）
    public static func stableID(audioURL: URL) throws -> String {
        let attrs = try FileManager.default.attributesOfItem(atPath: audioURL.path)
        let size = (attrs[.size] as? Int) ?? 0
        let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let seed = "\(audioURL.lastPathComponent)|\(size)|\(mtime)"
        let digest = SHA256.hash(data: Data(seed.utf8))
        return digest.map { String(format: "%02x", $0) }.prefix(16).joined()
    }
}
