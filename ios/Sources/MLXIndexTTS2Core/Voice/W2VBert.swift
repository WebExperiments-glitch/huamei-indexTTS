import Foundation
import MLX

/// 自研 W2VBert Conformer 编码器（w2v-bert-2.0，1B 级，全量解压前向）
///
/// 结构（由权重清单 `scripts/golden/w2vbert_keys.json` 与 config.json 锁定）：
///   · feature_projection：Conv1d(160→1024, k3) + LayerNorm
///   · 24 × Conformer 块（macaron 半残差）：
///       x += 0.5·FFN1(x)  →  x += Attn(x)  →  x += ConvModule(x)  →  x += 0.5·FFN2(x)
///   · 注意力：relative_key（distance_embedding[73,64]，left 64 / right 8）
///   · 返回第 17 索引（= 16 号层输出，与官方 hidden_states[17] 对齐）
///
/// ⚠️ 权重布局：本仓库 safetensors 为 torch 布局 [O,I,K]；MLX conv 需 [O,K,I]，加载时转置一次。
/// ⚠️ relative-key 与卷积模块的两处公式细节标 *calibrate*，用 golden 对拍收敛（见 README P5）。
public final class W2VBert {

    public struct Config {
        public let hidden: Int
        public let layers: Int
        public let heads: Int
        public let headDim: Int
        public let ffnDim: Int
        public let convKernel: Int
        public let leftMax: Int
        public let rightMax: Int
        public let featureIn: Int
        public static let v25 = Config(hidden: 1024, layers: 24, heads: 16, headDim: 64,
                                       ffnDim: 4096, convKernel: 31, leftMax: 64, rightMax: 8,
                                       featureIn: 160)
    }

    public let cfg: Config

    // feature projection
    private let featProjW: MLXArray     // [o=1024, k=3, i=160]
    private let featProjB: MLXArray
    private let featLN_w: MLXArray
    private let featLN_b: MLXArray

    // 每层
    private struct Block {
        let ffn1LNw: MLXArray, ffn1LNb: MLXArray
        let ffn1IW: MLXArray, ffn1Ib: MLXArray, ffn1OW: MLXArray, ffn1Ob: MLXArray
        let qW: MLXArray, qB: MLXArray, kW: MLXArray, kB: MLXArray
        let vW: MLXArray, vB: MLXArray, oW: MLXArray, oB: MLXArray
        let attnLNw: MLXArray, attnLNb: MLXArray
        let distEmb: MLXArray           // [73, 64]
        let convLNw: MLXArray, convLNb: MLXArray
        let dwLNw: MLXArray, dwLNb: MLXArray
        let dwW: MLXArray               // [1024, 31, 1]
        let pw1W: MLXArray, pw1B: MLXArray    // [2048,1,1024]
        let pw2W: MLXArray, pw2B: MLXArray    // [1024,1,1024]
        let ffn2LNw: MLXArray, ffn2LNb: MLXArray
        let ffn2IW: MLXArray, ffn2Ib: MLXArray, ffn2OW: MLXArray, ffn2Ob: MLXArray
        let finLNw: MLXArray, finLNb: MLXArray
    }
    private var blocks: [Block] = []

    /// 从 safetensors 构建（权重解压为 F32 MLXArray 常驻；首包外的按需加载组件，用完即卸）
    public init(path: String, config: Config = .v25) throws {
        cfg = config
        let f = try SafetensorsFile(path: path)
        defer { f.close() }

        func t(_ n: String) throws -> MLXArray { try f.tensor(n) }
        // torch [O,I,K] → MLX [O,K,I]
        func convPerm(_ x: MLXArray) -> MLXArray { x[0..., 2..., 1...] }

        featProjW = convPerm(try t("feature_projection.projection.weight"))
        featProjB = try t("feature_projection.projection.bias")
        featLN_w = try t("feature_projection.layer_norm.weight")
        featLN_b = try t("feature_projection.layer_norm.bias")

        for i in 0..<config.layers {
            let p = "encoder.layers.\(i)."
            blocks.append(Block(
                ffn1LNw: try t(p + "ffn1_layer_norm.weight"),
                ffn1LNb: try t(p + "ffn1_layer_norm.bias"),
                ffn1IW: try t(p + "ffn1.intermediate_dense.weight"),
                ffn1Ib: try t(p + "ffn1.intermediate_dense.bias"),
                ffn1OW: try t(p + "ffn1.output_dense.weight"),
                ffn1Ob: try t(p + "ffn1.output_dense.bias"),
                qW: try t(p + "self_attn.linear_q.weight"),
                qB: try t(p + "self_attn.linear_q.bias"),
                kW: try t(p + "self_attn.linear_k.weight"),
                kB: try t(p + "self_attn.linear_k.bias"),
                vW: try t(p + "self_attn.linear_v.weight"),
                vB: try t(p + "self_attn.linear_v.bias"),
                oW: try t(p + "self_attn.linear_out.weight"),
                oB: try t(p + "self_attn.linear_out.bias"),
                attnLNw: try t(p + "self_attn_layer_norm.weight"),
                attnLNb: try t(p + "self_attn_layer_norm.bias"),
                distEmb: try t(p + "self_attn.distance_embedding.weight"),
                convLNw: try t(p + "conv_module.layer_norm.weight"),
                convLNb: try t(p + "conv_module.layer_norm.bias"),
                dwLNw: try t(p + "conv_module.depthwise_layer_norm.weight"),
                dwLNb: try t(p + "conv_module.depthwise_layer_norm.bias"),
                dwW: convPerm(try t(p + "conv_module.depthwise_conv.weight")),
                pw1W: convPerm(try t(p + "conv_module.pointwise_conv1.weight")),
                pw1B: try t(p + "conv_module.pointwise_conv1.bias"),
                pw2W: convPerm(try t(p + "conv_module.pointwise_conv2.weight")),
                pw2B: try t(p + "conv_module.pointwise_conv2.bias"),
                ffn2LNw: try t(p + "ffn2_layer_norm.weight"),
                ffn2LNb: try t(p + "ffn2_layer_norm.bias"),
                ffn2IW: try t(p + "ffn2.intermediate_dense.weight"),
                ffn2Ib: try t(p + "ffn2.intermediate_dense.bias"),
                ffn2OW: try t(p + "ffn2.output_dense.weight"),
                ffn2Ob: try t(p + "ffn2.output_dense.bias"),
                finLNw: try t(p + "final_layer_norm.weight"),
                finLNb: try t(p + "final_layer_norm.bias")
            ))
        }
    }

    // MARK: - 基础算子（复用 Ops 语义，写清每步）

    private func ffn(_ x: MLXArray, lnW: MLXArray, lnB: MLXArray,
                     iW: MLXArray, iB: MLXArray, oW: MLXArray, oB: MLXArray) -> MLXArray {
        var h = Ops.layerNorm(x, weight: lnW, bias: lnB)
        h = Ops.linear(h, w: iW, b: iB)
        h = Ops.silu(h)
        h = Ops.linear(h, w: oW, b: oB)
        return h
    }

    /// 自注意力（relative_key；距离桶 = clamp(j-i, -right, +left) + right）
    private func attention(_ x: MLXArray, b: Block) -> MLXArray {
        let T = x.shape[1]
        let h = cfg.heads
        let d = cfg.headDim
        let inX = Ops.layerNorm(x, weight: b.attnLNw, bias: b.attnLNb)

        func heads(_ w: MLXArray, _ bb: MLXArray) -> MLXArray {
            Ops.linear(inX, w: w, b: bb)
                .reshaped([T, h, d])       // [T,H,D]
        }
        let q = heads(b.qW, b.qB)          // [T,H,D]
        let k = heads(b.kW, b.kB)
        let v = heads(b.vW, b.vB)

        // 绝对注意力分数 [T,H,T]（每位置 i 一批，softmax 沿最后轴 j）
        let scores = MLX.matmul(q, k.transposed(0, 2, 1))

        // 相对偏置（query-side）：r(i,j,h) = Σ_d q[i,h,d]·emb[bucket(i,j),h,d]
        let maxRel = cfg.leftMax + cfg.rightMax + 1                      // 73
        // [T,H,73] = q [T,H,D] · embᵀ [D,73]
        let relLogits = MLX.matmul(q, b.distEmb.transposed(0, 1))        // [T,H,73]
        let relLogitsT = relLogits.transposed(0, 2, 1)                   // [T,73,H]
        // 距离桶矩阵 [T,T]：r = clamp(j - i, -right, left) + right
        let idx = indexBuckets(T: T, left: cfg.leftMax, right: cfg.rightMax)  // [T,T]
        // 逐位置 i 收集 → [T,T,H]；转 [T,H,T] 加到分数
        let bias = gatherPerPair(relLogitsT, idxs: idx, h: h)            // [T,T,H]
        let scoresH = scores + bias.transposed(0, 2, 1)                  // [T,H,T]

        let probs = Ops.softmaxLast(scoresH)                             // 沿 j
        let ctx = MLX.matmul(probs, v)                                   // [T,H,D]
        let merged = ctx.reshaped([1, T, h * d])
        let out = Ops.linear(merged, w: b.oW, b: b.oB)
        _ = maxRel
        return out
    }

    /// 距离桶索引 [T,T]（Int32）
    private func indexBuckets(T: Int, left: Int, right: Int) -> MLXArray {
        let positions = (0..<T).map { Int32($0) }
        var rows: [Int32] = []
        rows.reserveCapacity(T * T)
        for i in 0..<T {
            for j in 0..<T {
                let shift = positions[j] - positions[i]
                let clamped = max(-Int32(right), min(Int32(left), shift))
                rows.append(clamped + Int32(right))
            }
        }
        return MLXArray(rows, [T, T])
    }

    /// 按 [T,T] 索引从 [T,73,H] 收集 → [T,T,H]
    private func gatherPerPair(_ rel: MLXArray, idxs: MLXArray, h: Int) -> MLXArray {
        let T = idxs.shape[0]
        let flat = idxs.reshaped([-1])                                   // [T*T]
        let collected = MLX.take(rel, flat, axis: 1)                     // [T, T*T, H]
        return collected.reshaped([T, T, h])                             // [T,T,H]
    }

    /// 卷积模块（depthwise k31 → 通道 LN → pointwise 1×1(2048) → swish → pointwise(1024)）
    private func convModule(_ x: MLXArray, b: Block) -> MLXArray {
        let h = Ops.layerNorm(x, weight: b.convLNw, bias: b.convLNb)     // [B,T,D]
        let B = h.shape[0], T = h.shape[1], D = h.shape[2]
        var ct = h.transposed(0, 2, 1)                                    // [B,D,T]
        ct = Ops.reflectPad(ct, left: 15, right: 15)
        // depthwise：dwconv 权重 [D,31,1]（out=in=D, in/group=1）→ groups=D 才能每通道独立卷积
        ct = Ops.conv1d(ct, w: b.dwW, b: nil, dilation: 1, groups: b.dwW.shape[0])
        // depthwise_layer_norm：对通道维归一化（GroupNorm(1) 语义）
        ct = Ops.layerNorm(ct, weight: b.dwLNw.reshaped([1, D, 1]),
                           bias: b.dwLNb.reshaped([1, D, 1]), eps: 1e-5)
        ct = Ops.conv1d(ct, w: b.pw1W, b: b.pw1B)                        // [B,2048,T]
        ct = Ops.silu(ct)
        ct = Ops.conv1d(ct, w: b.pw2W, b: b.pw2B)                        // [B,D,T]
        let out = ct.transposed(0, 2, 1)
        _ = B; _ = T; _ = D
        return out
    }

    /// 单 conformer 块
    private func block(_ x: MLXArray, b: Block) -> MLXArray {
        var y = ffn(x, lnW: b.ffn1LNw, lnB: b.ffn1LNb,
                    iW: b.ffn1IW, iB: b.ffn1Ib, oW: b.ffn1OW, oB: b.ffn1Ob)
        var h = x + y * 0.5
        h = h + attention(h, b: b)
        h = h + convModule(h, b: b)
        y = ffn(h, lnW: b.ffn2LNw, lnB: b.ffn2LNb,
                iW: b.ffn2IW, iB: b.ffn2Ib, oW: b.ffn2OW, oB: b.ffn2Ob)
        return h + y * 0.5
    }

    // MARK: - 前向

    private func featureProjection(_ x: MLXArray) -> MLXArray {
        // x [B,T,160] → [B,160,T] conv → [B,T,1024]
        var h = x.transposed(0, 2, 1)
        h = Ops.conv1d(h, w: featProjW, b: featProjB)
        h = h.transposed(0, 2, 1)
        h = Ops.layerNorm(h, weight: featLN_w, bias: featLN_b)
        return h
    }

    /// 前向：返回 hiddenStates[i] 输出。targetIndex=17 → 与官方 hidden_states[17] 对齐（0=投影后, 1..24=各层后）
    public func hiddenState(_ features: MLXArray, targetIndex: Int = 17) throws -> MLXArray {
        // features [B,T,160]
        var h = featureProjection(features)
        if targetIndex == 0 { return h }
        for i in 0..<cfg.layers {
            h = block(h, b: blocks[i])
            if targetIndex == i + 1 { return h }
        }
        throw SafetensorsError.missing("layer \(targetIndex) out of range")
    }
}