import Foundation

/// 崩溃安全的同步落盘日志（"日志面板来不及刷新就闪退"的兜底）
/// 写 Documents/huamei-subsys.log（每次写都 seek 末尾+flush）；崩溃后 UI 重启读取尾部展示。
public enum DLog {

    private static let lock = NSLock()

    private static var url: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huamei-subsys.log")
    }

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    public static func write(_ s: String) {
        lock.lock(); defer { lock.unlock() }
        let line = "[\(formatter.string(from: Date()))] \(s)\n"
        guard let data = line.data(using: .utf8) else { return }
        if FileManager.default.fileExists(atPath: url.path),
           let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write(data)
            try? h.close()
        } else {
            try? data.write(to: url)
        }
    }

    public static func reset() {
        lock.lock(); defer { lock.unlock() }
        try? FileManager.default.removeItem(at: url)
    }

    public static func tail(_ n: Int = 40) -> [String] {
        lock.lock(); defer { lock.unlock() }
        guard let d = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return d.split(separator: "\n").suffix(n).map(String.init)
    }
}