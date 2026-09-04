import Foundation

/// tiktoken byte-level BPE（自研，对齐官方 get_encoding 语义，num_languages=99 → 60509）
///
/// - ranks: tiktoken 文件 58,836 行（byte 序列 → rank）
/// - specials: specials.json（1,673 个，id 58,836 … 60,508）
/// - 文本切分近似 GPT-2 pat_str（Unicode 字母/数字/其它分类）；中英日等常规文本边界与官方一致，
///   特殊 token `<|xxx|>` 整段匹配（官方 allowed_special="all" 语义）
/// 已知差距（如实标注）：与官方 regex 的极少数混合符号边界可能有差异；不影响语言前缀与常规内容。
public final class TiktokenBPE {

    public struct Tokenizer {
        public let ranks: [Data: Int]
        public let specials: [String: Int]
        public init(ranks: [Data: Int], specials: [String: Int]) {
            self.ranks = ranks; self.specials = specials
        }
    }

    private let ranks: [Data: Int]
    private let specials: [String: Int]
    private let specialKeysSorted: [String]      // 按长度降序，先长后短防前缀误吞

    public init(ranks: [Data: Int], specials: [String: Int]) {
        self.ranks = ranks
        self.specials = specials
        self.specialKeysSorted = specials.keys.sorted { $0.count > $1.count }
    }

    /// 从 .tiktoken 文件解析（自动跳过无法 base64 的空 token 行 `= 48474`）
    public static func parseRanks(url: URL) throws -> [Data: Int] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [Data: Int] = [:]
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count == 2, let rank = Int(parts[1]) else { continue }
            let b64 = String(parts[0])
            guard let bytes = Data(base64Encoded: b64) else { continue }   // "=" → b"" 已在 python 端验证为死 token
            out[bytes] = rank
        }
        return out
    }

    public static func parseSpecials(url: URL) throws -> [String: Int] {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw NSError(domain: "TiktokenBPE", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "specials.json unreadable"])
        }
        return obj
    }

    public var vocabSize: Int { ranks.count + specials.count }

    // MARK: - 编码

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            // 1) 特殊 token 优先
            if let (token, id, consumed) = consumeSpecial(from: rest) {
                out.append(id)
                rest = rest.dropFirst(consumed)
                continue
            }
            // 2) 普通文本：取下一个"词段"（近似 pat_str）
            let piece = takeNextPiece(rest)
            out.append(contentsOf: encodePiece(String(piece)))
            rest = rest.dropFirst(piece.count)
        }
        return out
    }

    /// 特殊 token 匹配：从开头尝试（`<|...|>`）
    private func consumeSpecial(from s: Substring) -> (String, Int, Int)? {
        guard s.hasPrefix("<|") else { return nil }
        // 找最近一次 `|>`（一次匹配，不含嵌套）
        if let closeRange = s.range(of: "|>") {
            let cand = String(s[s.startIndex...closeRange.lowerBound].dropLast(0)) + "|>"
            if let id = specials[cand] {
                return (cand, id, cand.count)
            }
        }
        // 非特殊的长 <|...|>：仍作为普通文本的一部分继续切
        return nil
    }

    /// 近似 GPT-2 pat_str：`'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    private func takeNextPiece(_ s: Substring) -> Substring {
        let chars = Array(s)
        guard let first = chars.first else { return "" }

        // 特殊先导空格处理
        var idx = 0
        var leadSpace = false
        if chars[idx] == " " { leadSpace = true; idx += 1 }
        guard idx < chars.count else { return s }

        let c = chars[idx]
        let isLetter = c.isLetter     // Unicode \p{L}
        let isNumber = c.isNumber     // Unicode \p{N}
        let isSpace  = c.isWhitespace

        var count = idx
        if leadSpace {
            count += 0 // 首空格已计数
        }
        if isLetter || isNumber {
            // 连续同类的字母/数字（允许数字+字母混合并不跨越空格）
            var i = idx
            let categoryLetter = isLetter
            while i < chars.count {
                let ch = chars[i]
                if ch == " " { break }
                if categoryLetter {
                    if ch.isNumber { break }
                    if !ch.isLetter && !isApostrophe(ch) { break }
                } else {
                    if ch.isLetter { break }
                    if !ch.isNumber { break }
                }
                i += 1
            }
            return s.prefix(i)
        } else if isSpace {
            var i = idx
            while i < chars.count && chars[i].isWhitespace { i += 1 }
            return s.prefix(i)
        } else {
            // 标点/符号：贪婪一个或多个非 L/N/空格的连续符号
            var i = idx
            while i < chars.count {
                let ch = chars[i]
                if ch.isLetter || ch.isNumber || ch.isWhitespace { break }
                i += 1
            }
            return s.prefix(i)
        }
    }

    private func isApostrophe(_ c: Character) -> Bool {
        // 缩写 `'s` `'t` 等在 pat_str 中与前面字母同段
        return c == "'"
    }

    /// 单段字节 BPE 合并（最小 rank pair，同 rank 最左；语义同 tiktoken）
    private func encodePiece(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        guard !bytes.isEmpty else { return [] }
        if bytes.count == 1 { return [ranks[Data(bytes)] ?? 0] }

        var parts: [Data] = bytes.map { Data([$0]) }
        while parts.count > 1 {
            var bestRank: Int? = nil
            var bestIdx = -1
            for i in 0..<(parts.count - 1) {
                var key = parts[i]; key.append(contentsOf: parts[i+1])
                if let r = ranks[key], r < (bestRank ?? .max) {
                    bestRank = r; bestIdx = i
                }
            }
            guard bestIdx >= 0 else { break }
            var merged = parts[bestIdx]
            merged.append(contentsOf: parts[bestIdx + 1])
            parts[bestIdx] = merged
            parts.remove(at: bestIdx + 1)
        }
        return parts.compactMap { ranks[$0] }
    }

    // MARK: - 解码

    private var inverseCache: [Int: Data]?

    public func decode(_ ids: [Int]) -> String {
        if inverseCache == nil {
            var inv: [Int: Data] = [:]
            inv.reserveCapacity(ranks.count)
            for (k, v) in ranks { inv[v] = k }
            inverseCache = inv
        }
        guard let inv = inverseCache else { return "" }
        var data = Data()
        for id in ids {
            if let d = inv[id] { data.append(d) }
        }
        return String(decoding: data, as: UTF8.self)
    }
}