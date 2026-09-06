import Foundation
import MLX

/// EnhancedCodec decode（p0/codec_impl 移植）：FVQ(vq2emb) → VocosBackbone → Linear → ×2 → up
/// 输入 codes [T]；输出 S_infer [2T, 1024]
public final class SemanticCodecDecoder {

    private let codebook: MLXArray            // [8192, 8]
    private let outProjW: MLXArray            // (1024, 1, 8)
    private let outProjB: MLXArray?
    // decoder.0 = VocosBackbone
    private let embedW: MLXArray, embedB: MLXArray          // (384,7,1024)
    private let normW: MLXArray, normB: MLXArray
    private let finalNormW: MLXArray, finalNormB: MLXArray
    private let blocks: [ConvNextBlock]
    private let headW: MLXArray, headB: MLXArray            // Linear 384→1024
    private let upW: MLXArray, upB: MLXArray                // (1024,3,1024)

    private let file: SafetensorsFile

    public init(path: String) throws {
        let f = try SafetensorsFile(path: path)
        self.file = f
        func load(_ k: String) throws -> MLXArray { try f.tensor(k) }
        codebook  = try load("quantizer.quantizers.0.codebook.weight")
        outProjW  = try load("quantizer.quantizers.0.out_project.weight")
        outProjB  = try load("quantizer.quantizers.0.out_project.bias")
        embedW    = try load("decoder.0.embed.weight")
        embedB    = try load("decoder.0.embed.bias")
        normW     = try load("decoder.0.norm.weight")
        normB     = try load("decoder.0.norm.bias")
        finalNormW = try load("decoder.0.final_layer_norm.weight")
        finalNormB = try load("decoder.0.final_layer_norm.bias")
        headW     = try load("decoder.1.weight")
        headB     = try load("decoder.1.bias")
        upW       = try load("up.weight")
        upB       = try load("up.bias")
        var blks: [ConvNextBlock] = []
        for i in 0..<TTSConfig.vocosLayers {
            let p = "decoder.0.convnext.\(i)"
            blks.append(ConvNextBlock(
                dwW: try load("\(p).dwconv.weight"),
                dwB: try load("\(p).dwconv.bias"),
                nW: try load("\(p).norm.weight"), nB: try load("\(p).norm.bias"),
                p1W: try load("\(p).pwconv1.weight"), p1B: try load("\(p).pwconv1.bias"),
                p2W: try load("\(p).pwconv2.weight"), p2B: try load("\(p).pwconv2.bias"),
                gamma: try load("\(p).gamma")))
        }
        blocks = blks
    }

    struct ConvNextBlock {
        let dwW, dwB: MLXArray
        let nW, nB: MLXArray
        let p1W, p1B, p2W, p2B: MLXArray
        let gamma: MLXArray
        func call(_ x: MLXArray) -> MLXArray {
            var h = Ops.conv1d(x, w: dwW, b: dwB)          // depthwise 由 groups 实现；kernel(384,7,1)
            h = h.transposed(0, 2, 1)
            h = Ops.layerNorm(h, weight: nW, bias: nB, eps: 1e-6)
            var y = Ops.linear(h, w: p1W, b: p1B)
            y = Ops.gelu(y)
            y = Ops.linear(y, w: p2W, b: p2B)
            y = y * gamma
            y = y.transposed(0, 2, 1)
            return x + y
        }
    }

    private func vocosBackbone(_ x: MLXArray) -> MLXArray {
        // x [B,1024,T]
        var h = Ops.conv1d(x, w: embedW, b: embedB)         // [B,384,T]
        h = h.transposed(0, 2, 1)
        h = Ops.layerNorm(h, weight: normW, bias: normB, eps: 1e-6)
        h = h.transposed(0, 2, 1)
        for blk in blocks { h = blk.call(h) }
        h = h.transposed(0, 2, 1)
        return Ops.layerNorm(h, weight: finalNormW, bias: finalNormB, eps: 1e-6)   // [B,T,384]
    }

    /// decode：codes [T] → S_infer [2T, 1024]
    public func decode(codes: [Int]) -> MLXArray {
        let T = codes.count
        // vq2emb：embedding 行（文件直读 → CPU 数组，8×T 很小）
        var embRows = [Float](repeating: 0, count: T * 8)
        for t in 0..<T {
            guard let row = try? file.row("quantizer.quantizers.0.codebook.weight",
                                          index: codes[t]) else { continue }
            for k in 0..<8 { embRows[t * 8 + k] = row[k] }
        }
        var x = MLXArray(embRows, [T, 8]).transposed(0, 1)  // [8, T]
        x = x.reshaped([1, 8, T])
        // 权重为 F16：输入必须转成同等 dtype（F32 输入 + F16 权重在 MLX conv1d 断言崩溃）
        if outProjW.dtype == .float16 {
            x = x.asType(.float16)
        }
        var q = Ops.conv1d(x, w: outProjW, b: outProjB)     // [1,1024,T]
        // decoder
        var feat = vocosBackbone(q)                          // [1,T,384]
        feat = Ops.linear(feat, w: headW, b: headB)          // [1,T,1024]
        // ×2 nearest → up conv
        feat = feat.transposed(0, 2, 1)                      // [1,1024,T]
        feat = nearest2x(feat)                               // [1,1024,2T]
        let out = Ops.conv1d(feat, w: upW, b: upB)           // [1,1024,2T]
        return out.transposed(0, 2, 1)                       // [1,2T,1024]
    }

    private func nearest2x(_ x: MLXArray) -> MLXArray {
        // [B,C,T] → 每元素重复（nearest ×2）
        let shape = x.shape
        let (b, c, t) = (shape[0], shape[1], shape[2])
        var floats = x.asFloatArray()
        var out = [Float](repeating: 0, count: b * c * t * 2)
        for i in 0..<floats.count { out[i * 2] = floats[i]; out[i * 2 + 1] = floats[i] }
        return MLXArray(out, [b, c, t * 2])
    }
}