import Foundation

/// safetensors 读取器（轻量自研）
/// 支持 dtype：F16 / F32 / U32（打包量化权重）
///
/// 说明：权重按需 lazy 加载；读取时直接解码为 MLXArray。
/// API 存疑点（待 Xcode 首编确认）已在函数注释标注。
import MLX

public final class SafetensorsFile {

    public let name: String
    private let handle: FileHandle
    private let header: [String: Any]
    private let dataOffset: UInt64

    public init(path: String) throws {
        name = (path as NSString).lastPathComponent
        guard let fh = FileHandle(forReadingAtPath: path) else {
            throw SafetensorsError.cannotOpen(path)
        }
        handle = fh

        // 8 字节小端 header 长度
        let lenData = try fh.read(upToCount: 8) ?? Data()
        guard lenData.count == 8 else { throw SafetensorsError.badFormat }
        let headerLen = UInt64(littleEndian: lenData.withUnsafeBytes { $0.load(as: UInt64.self) })
        let headerData = try fh.read(upToCount: Int(headerLen)) ?? Data()
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SafetensorsError.badJSON
        }
        header = obj
        dataOffset = 8 + headerLen
    }

    public var tensorNames: [String] {
        header.keys.sorted()
    }

    public func shape(of name: String) -> [Int]? {
        guard let info = header[name] as? [String: Any],
              let shape = info["shape"] as? [Int] else { return nil }
        return shape
    }

    /// 解码单个张量 → MLXArray
    public func tensor(_ name: String) throws -> MLXArray {
        guard let info = header[name] as? [String: Any],
              let dtype = info["dtype"] as? String,
              let shape = info["shape"] as? [Int],
              let offsets = info["data_offsets"] as? [Int], offsets.count == 2 else {
            throw SafetensorsError.missing(name)
        }
        let start = dataOffset + UInt64(offsets[0])
        let len = UInt64(offsets[1] - offsets[0])
        try handle.seek(toOffset: start)
        let data = try handle.read(upToCount: Int(len)) ?? Data()
        return try decode(data: data, dtype: dtype, shape: shape, name: name)
    }

    /// 读取单行（embedding 查表路径，避免整表驻留）→ [Float]
    public func row(_ name: String, index: Int) throws -> [Float] {
        guard let info = header[name] as? [String: Any],
              let dtype = info["dtype"] as? String,
              let shape = info["shape"] as? [Int],
              let offsets = info["data_offsets"] as? [Int], offsets.count == 2 else {
            throw SafetensorsError.missing(name)
        }
        guard index >= 0 && index < shape[0] else { throw SafetensorsError.missing(name) }
        let rowBytes = shape.dropFirst().reduce(1, *) * (dtype == "F32" ? 4 : 2)
        let start = dataOffset + UInt64(offsets[0]) + UInt64(index * rowBytes)
        try handle.seek(toOffset: start)
        let data = try handle.read(upToCount: rowBytes) ?? Data()
        if dtype == "F32" {
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        // F16
        return data.withUnsafeBytes { raw in
            raw.bindMemory(to: UInt16.self).map { SafetensorsFile.halfToFloat($0) }
        }
    }

    public func close() { try? handle.close() }
    deinit { try? handle.close() }

    // ---- 解码（f16/f32/u32 → MLXArray）----
    fileprivate static func halfToFloat(_ h: UInt16) -> Float {
        let sign: Float = (h & 0x8000) != 0 ? -1 : 1
        let exp = Int((h >> 10) & 0x1F)
        let mant = Float(h & 0x3FF)
        switch exp {
        case 0:  return sign * mant * powf(2.0, -24)
        case 31: return sign * (mant == 0 ? .infinity : .nan)
        default: return sign * (1.0 + mant / 1024.0) * powf(2.0, Float(exp - 15))
        }
    }

    private func decode(data: Data, dtype: String, shape: [Int], name: String) throws -> MLXArray {
        switch dtype {
        case "F16":
            return MLXArray.fromHalfBytes(data, shape: shape)
        case "F32":
            return MLXArray.fromFloatBytes(data, shape: shape)
        case "U32":
            // 打包量化权重：保留 uint32 语义（由 QuantizedLinear/quantizedMatmul 消费）
            let words: [UInt32] = data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            return MLXArray(words, shape)
        case "I32", "I64":
            // 低频使用（如 index 类），按需转 Int32
            throw SafetensorsError.unsupported(dtype)
        case "BF16":
            throw SafetensorsError.unsupported(dtype)
        default:
            throw SafetensorsError.unsupported(dtype)
        }
    }}

public enum SafetensorsError: Error, LocalizedError {
    case cannotOpen(String)
    case badFormat
    case badJSON(String)
    case missing(String)
    case unsupported(String)
    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let p): return "cannot open \(p)"
        case .badFormat: return "bad safetensors header"
        case .badJSON(let n): return "bad safetensors JSON: \(n)"
        case .missing(let n): return "tensor missing: \(n)"
        case .unsupported(let d): return "unsupported dtype: \(d)"
        }
    }
}

// MARK: - MLXArray 构造 helpers
// ⚠️ 首次在 Xcode 编译时若以下 init 不存在，参考 mlx-swift 实际 API 改一处即可。
extension MLXArray {

    /// F16 小端字节 → MLXArray（F32 表达；内部由 MLX 处理 half→float）
    static func fromHalfBytes(_ data: Data, shape: [Int]) -> MLXArray {
        let floats: [Float] = data.withUnsafeBytes { raw in
            let halfs = raw.bindMemory(to: UInt16.self)
            return halfs.map { SafetensorsFile.halfToFloat($0) }
        }
        return MLXArray(floats, shape)
    }

    static func fromFloatBytes(_ data: Data, shape: [Int]) -> MLXArray {
        let floats: [Float] = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return MLXArray(floats, shape)
    }
}