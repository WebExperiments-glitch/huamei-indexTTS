import Foundation
import MLX

/// GPT-2 主体（24 层，d=1280，20 头）—— 自研，权重按 vanch007 safetensors 布局直取
///
/// 8bit affine 量化（group 64）由 `quantizedMatmul` 内核消费（weight 打包 uint32，
/// scales/biases 每组 1 个）。布局：c_attn/c_proj/c_fc/c_proj 4 组量化线性。
public final class GPT2 {

    public struct Block {
        let ln1w, ln1b: MLXArray?
        let cAttnWq: MLXArray, cAttnScales: MLXArray, cAttnBiases: MLXArray?, cAttnB: MLXArray?
        let cProjWq: MLXArray, cProjScales: MLXArray, cProjBiases: MLXArray?, cProjB: MLXArray?
        let ln2w, ln2b: MLXArray?
        let cFcWq: MLXArray, cFcScales: MLXArray, cFcBiases: MLXArray?, cFcB: MLXArray?
        let cProj2Wq: MLXArray, cProj2Scales: MLXArray, cProj2Biases: MLXArray?, cProj2B: MLXArray?
    }

    public let blocks: [Block]
    public let lnFw: MLXArray, lnFb: MLXArray

    public init(store: GPTStore, layers: Int = TTSConfig.gptLayers) throws {
        var blks: [Block] = []
        for i in 0..<layers {
            let p = "gpt.h.\(i)"
            let quant = try store.quantPair("\(p).attn.c_attn")
            let cProj = try store.quantPair("\(p).attn.c_proj")
            let cFc   = try store.quantPair("\(p).mlp.c_fc")
            let cP2   = try store.quantPair("\(p).mlp.c_proj")
            blks.append(Block(
                ln1w: try store.optional("\(p).ln_1.weight"), ln1b: try store.optional("\(p).ln_1.bias"),
                cAttnWq: quant.0, cAttnScales: quant.1, cAttnBiases: quant.2, cAttnB: try store.optional("\(p).attn.c_attn.bias"),
                cProjWq: cProj.0, cProjScales: cProj.1, cProjBiases: cProj.2, cProjB: try store.optional("\(p).attn.c_proj.bias"),
                ln2w: try store.optional("\(p).ln_2.weight"), ln2b: try store.optional("\(p).ln_2.bias"),
                cFcWq: cFc.0, cFcScales: cFc.1, cFcBiases: cFc.2, cFcB: try store.optional("\(p).mlp.c_fc.bias"),
                cProj2Wq: cP2.0, cProj2Scales: cP2.1, cProj2Biases: cP2.2, cProj2B: try store.optional("\(p).mlp.c_proj.bias")))
        }
        blocks = blks
        lnFw = try store.required("gpt.ln_f.weight")
        lnFb = try store.required("gpt.ln_f.bias")
    }

    // MARK: - 前向

    /// 输入已是 embedding 序列 [B,T,D]，全序列因果（含 KV cache 更新）
    /// cache: 每层 [k,v] 之已缓存；startPos：当前序列在全局位置中的起点
    public func forward(_ x: MLXArray, cacheK: inout [MLXArray?], cacheV: inout [MLXArray?],
                        startPos: Int, ropeCosSin: MLXArray,
                        causalFull: Bool) -> MLXArray {
        var h = x
        let scale = 1.0 / Float(TTSConfig.gptHeadDim).squareRoot()
        for (i, blk) in blocks.enumerated() {
            let ln1 = Ops.layerNorm(h, weight: blk.ln1w, bias: blk.ln1b, eps: 1e-5)
            var qkv = Ops.quantizedLinear(ln1, wq: blk.cAttnWq, scales: blk.cAttnScales, biases: blk.cAttnBiases)
            if let b = blk.cAttnB { qkv = qkv + b }
            // [B,T,3*D] → q/k/v（reshape 后按 int 索引取组；若 MLX 索引不支持，
            // 备选：split 三段再 reshape）
            let d = TTSConfig.gptDim
            let qkvShape = qkv.shape
            let bsz = qkvShape[0], t = qkvShape[1]
            qkv = qkv.reshaped([bsz, t, 3, TTSConfig.gptHeads, TTSConfig.gptHeadDim])
            // 取出 [B,T,H,Dh]
            var q = qkv[0..., 0..., 0, 0..., 0...]
            var k = qkv[0..., 0..., 1, 0..., 0...]
            let v = qkv[0..., 0..., 2, 0..., 0...]
            // RoPE（CPU 精确；offset 保证增量步用绝对位置）
            q = Rotary.applyCPU(q, cs: ropeCosSin, offset: startPos)
            k = Rotary.applyCPU(k, cs: ropeCosSin, offset: startPos)
            // [B,T,H,Dh] → [B,H,T,Dh]（attention 布局）
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            let vT = v.transposed(0, 2, 1, 3)
            // 缓存（存 [B,H,T,Dh]）
            let fullK: MLXArray, fullV: MLXArray
            if causalFull {
                fullK = k; fullV = vT
            } else if let oldK = cacheK[i], let oldV = cacheV[i] {
                fullK = oldK.concatenated([k], axis: 2)
                fullV = oldV.concatenated([vT], axis: 2)
            } else {
                fullK = k; fullV = vT
            }
            cacheK[i] = fullK
            cacheV[i] = fullV
            // 因果注意力
            // q [B,H,T,Dh] × fullKᵀ [B,H,Dh,fullT]
            var scores = q.matmul(fullK.transposed(0, 1, 3, 2)) * scale
            let fullT = fullK.shape[2]
            if causalFull {
                // 上三角 -inf（对每个 (i,j) i<j 屏蔽；本批次首步 T 全序列）
                let mask = makeCausalMask(t: t, fullT: fullT)
                scores = scores + mask
            }
            let probs = Ops.softmaxLast(scores)
            var ctx = probs.matmul(fullV)                    // [B,H,T,Dh]
            ctx = ctx.transposed(0, 2, 1, 3).reshaped([bsz, t, d])
            var attnOut = Ops.quantizedLinear(ctx, wq: blk.cProjWq, scales: blk.cProjScales, biases: blk.cProjBiases)
            if let b = blk.cProjB { attnOut = attnOut + b }
            h = h + attnOut

            let ln2 = Ops.layerNorm(h, weight: blk.ln2w, bias: blk.ln2b, eps: 1e-5)
            var fc = Ops.quantizedLinear(ln2, wq: blk.cFcWq, scales: blk.cFcScales, biases: blk.cFcBiases)
            if let b = blk.cFcB { fc = fc + b }
            fc = Ops.geluNew(fc)
            var proj = Ops.quantizedLinear(fc, wq: blk.cProj2Wq, scales: blk.cProj2Scales, biases: blk.cProj2Biases)
            if let b = blk.cProj2B { proj = proj + b }
            h = h + proj
        }
        return Ops.layerNorm(h, weight: lnFw, bias: lnFb, eps: 1e-5)
    }

    private func makeCausalMask(t: Int, fullT: Int) -> MLXArray {
        var rows: [Float] = []
        rows.reserveCapacity(t * fullT)
        for i in 0..<t {
            for j in 0..<fullT {
                rows.append((fullT - t + i) >= j ? 0 : -Float.infinity)
            }
        }
        // [1,1,t,fullT] 广播到 [B,H,t,fullT]
        return MLXArray(rows, [1, 1, t, fullT])
    }
}

// MARK: - 顶部 head（final_norm → mel_head）

public final class GPTHead {
    let finalNw: MLXArray, finalNb: MLXArray
    let melHeadW: MLXArray, melHeadB: MLXArray
    public init(store: GPTStore) throws {
        finalNw = try store.required("final_norm.weight")
        finalNb = try store.required("final_norm.bias")
        melHeadW = try store.required("mel_head.weight")
        melHeadB = try store.required("mel_head.bias")
    }
    public func logits(_ hidden: MLXArray) -> MLXArray {
        // hidden [B,T,D] → 末位置 [B,1,D]（Range 切片，避免 int 索引不确定性）
        let t = hidden.shape[1]
        let last = hidden[0..., (t - 1)..<t, 0...]
        let n = Ops.layerNorm(last, weight: finalNw, bias: finalNb, eps: 1e-5)
        return Ops.linear(n, w: melHeadW, b: melHeadB)     // [B,1,8194]
    }
}

// MARK: - 权重源

public final class GPTStore {
    private let file: SafetensorsFile
    private var cache: [String: MLXArray] = [:]
    public init(path: String) throws {
        file = try SafetensorsFile(path: path)
    }
    public func required(_ key: String) throws -> MLXArray {
        if let v = cache[key] { return v }
        guard let v = try? file.tensor(key) else {
            throw SafetensorsError.missing(key)
        }
        cache[key] = v
        return v
    }
    public func optional(_ key: String) throws -> MLXArray? {
        guard file.tensorNames.contains(key) else { return nil }
        return try required(key)
    }
    /// 量化线性权重三元组 (weightU32, scales, biases?)
    public func quantPair(_ key: String) throws -> (MLXArray, MLXArray, MLXArray?) {
        let w = try required(key + ".weight")
        let s = try required(key + ".scales")
        let b = try optional(key + ".biases")
        return (w, s, b)
    }

    /// 行级读取（embedding 查表；CPU 安全路径）
    public func row(_ key: String, index: Int) throws -> [Float] {
        try file.row(key, index: index)
    }

    public func close() { file.close() }
}