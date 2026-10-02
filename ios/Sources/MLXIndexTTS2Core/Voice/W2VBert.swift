import Foundation
import MLX

/// 自研 W2VBert Conformer 编码器（w2v-bert-2.0，1B 级，全量解压前向）
///
/// 结构（由权重清单 `scripts/golden/w2vbert_keys.json` 与 config.json 锁定）：
///   · feature_projection：LayerNorm(160) → Linear(160→1024)（官方 Wav2Vec2BertFeatureProjection）
///   · 24 × Conformer 块（macaron 半残差）：
///       x += 0.5·FFN1(x)  →  x += Attn(x)  →  x += ConvModule(x)  →  x += 0.5·FFN2(x)
///   · 注意力：relative_key（distance_embedding[73,64]，left 64 / right 8）
///   · 返回第 17 索引（= 16 号层输出，与官方 hidden_states[17] 对齐）
///
/// ⚠️ 权重布局：conv 权重为 torch [O,I,K]，MLX conv 需 [O,K,I]，加载时转置一次；
///    Linear 权重（feature_projection/FFN/attn）本就是 [out,in] 2D，**不可转置**。
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

    /// 数值对拍开关：true 时导出各层输出（日志统计量 + Documents/w2v_dump/*.bin）。
    /// 与 scripts/w2vbert_layers_golden.py 的产物逐层比对，校准完成后置 false。
    public var dumpForCalibration = true

    // feature projection（官方：LayerNorm(160) → Linear(160→1024)）
    private let featProjW: MLXArray     // [o=1024, i=160] 2D Linear（不可转置）
    private let featProjB: MLXArray
    private let featLN_w: MLXArray      // [160]
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
        let dwW: MLXArray               // [1024, 31, 1]（groups=1024，无 bias）
        let pw1W: MLXArray              // [2048, 1, 1024]（无 bias）
        let pw2W: MLXArray              // [1024, 1, 1024]（无 bias）
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
        // ⚠️ 轴置换必须用 transposed —— `x[0..., 2..., 1...]` 在 mlx-swift 里是轴切片
        //    （"从索引N切到末尾"）：对 depthwise [1024,1,7] / pw [·,·,1] 会切出空维，
        //    触发 mlx_array_dim 报错崩溃（与 Campplus p1/p2 同源问题）。
        // torch [O,I,K] → MLX [O,K,I]
        func convPerm(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 1) }

        // ⚠️ feature_projection 是 Linear（权重 2D [1024,160]），绝不走 convPerm/转置
        featProjW = try t("feature_projection.projection.weight")
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
                pw2W: convPerm(try t(p + "conv_module.pointwise_conv2.weight")),
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

    /// 自注意力 —— 严格对齐官方 Wav2Vec2BertSelfAttention.forward（relative_key 分支）：
    ///   scores = q·kᵀ / √D
    ///   distance = clamp(j-i, -left(64), +right(8))，桶号 = distance + left ∈ [0,72]
    ///   posemb = distance_embedding(桶号) ∈ R^D
    ///   scores += (Σ_d q[i,h,d]·posemb[i,j,d]) / √D
    /// ⚠️ 此前三处写错：① 完全没有 √D 缩放；② clamp 上下界写成 (-right,+left) 方向正好反；
    ///    ③ 桶偏移用 +right 而非 +left。另 gather 元素数差 T 倍导致返回空数组。
    private func attention(_ x: MLXArray, b: Block) -> MLXArray {
        let T = x.shape[1]
        let H = cfg.heads
        let D = cfg.headDim
        let scale = 1.0 / Float(D).squareRoot()                      // 官方 / math.sqrt(head_size)
        let inX = Ops.layerNorm(x, weight: b.attnLNw, bias: b.attnLNb)

        func heads(_ w: MLXArray, _ bb: MLXArray) -> MLXArray {
            Ops.linear(inX, w: w, b: bb).reshaped([T, H, D])          // [T,H,D]
        }
        let q = heads(b.qW, b.qB)
        let k = heads(b.kW, b.kB)
        let v = heads(b.vW, b.vB)

        // ① scores = q·kᵀ / √D
        let scores = MLX.matmul(q, k.transposed(0, 2, 1)) * scale      // [T,H,T]

        // ② relative_key 偏置：rel[i,j,h] = Σ_d q[i,h,d] · emb[bucket(i,j),d]
        let nBucket = cfg.leftMax + cfg.rightMax + 1                   // 73
        let relLogits = MLX.matmul(q, b.distEmb.transposed(0, 1))      // [T,H,73] = q·embᵀ
        let relFlat = relLogits.transposed(0, 2, 1).reshaped([T * nBucket, H])   // [T*73,H]
        let idx = pairIndices(T: T, left: cfg.leftMax, right: cfg.rightMax, buckets: nBucket)
        let gathered = MLX.take(relFlat, idx.reshaped([-1]), axis: 0)  // [T*T,H]
        let bias = gathered.reshaped([T, T, H])                        // [T,T,H]

        // ③ scores += bias / √D
        let scoresH = scores + bias.transposed(0, 2, 1) * scale        // [T,H,T]

        let probs = Ops.softmaxLast(scoresH)                           // 沿 key 维
        let ctx = MLX.matmul(probs, v)                                 // [T,H,D]
        let merged = ctx.reshaped([1, T, H * D])
        return Ops.linear(merged, w: b.oW, b: b.oB)
    }

    /// 相对位置的展平行索引 [T,T]（Int32）= i*73 + (clamp(j-i, -left, +right) + left)
    /// 官方：distance = clamp(j-i, -left_max, +right_max)，桶号 = distance + left_max
    private func pairIndices(T: Int, left: Int, right: Int, buckets: Int) -> MLXArray {
        var rows: [Int32] = []
        rows.reserveCapacity(T * T)
        for i in 0..<T {
            let base = Int32(i * buckets)
            for j in 0..<T {
                let shift = Int32(j - i)
                let clamped = max(-Int32(left), min(Int32(right), shift))
                rows.append(base + clamped + Int32(left))
            }
        }
        return MLXArray(rows, [T, T])
    }

    /// 卷积模块 —— 严格对齐官方 Wav2Vec2BertConvolutionModule.forward：
    ///   LN(1024) → [B,C,T] → pointwise_conv1(1×1, C→2C) → GLU(→C)
    ///   → 左侧零 pad (k-1)=30（causal）→ depthwise_conv(k31, groups=C, padding=0)
    ///   → [B,T,C] → depthwise_layer_norm(LN over C) → activation(swish) → pointwise_conv2(1×1) → [B,T,C]
    /// ⚠️ 三个 conv 均无 bias；pad 是 causal 左 30，不是对称 15。
    private func convModule(_ x: MLXArray, b: Block) -> MLXArray {
        DLog.write("W2VCONV in x=\(x.shape) ndim=\(x.ndim) ln=\(b.convLNw.shape) dw=\(b.dwW.shape) pw1=\(b.pw1W.shape) pw2=\(b.pw2W.shape)")
        // 兜底：形状异常时不让 Swift 下标越界成崩溃，打印后跳过该子层（便于拿到完整日志定位上游）
        guard x.ndim == 3, b.dwW.ndim == 3 else {
            DLog.write("W2VCONV SKIP bad x.ndim=\(x.ndim) dw.ndim=\(b.dwW.ndim)")
            return x
        }
        let h = Ops.layerNorm(x, weight: b.convLNw, bias: b.convLNb)     // ① LN(1024) on [B,T,C]
        let D = h.shape[2]
        var ct = h.transposed(0, 2, 1)                                    // [B,C,T]
        ct = Ops.conv1d(ct, w: b.pw1W, b: nil)                            // ② [B,2C,T]
        ct = glu(ct)                                                      // ③ GLU → [B,C,T]
        ct = Ops.zeroPad(ct, left: b.dwW.shape[1] - 1, right: 0)          // ④ causal pad 30
        ct = Ops.conv1d(ct, w: b.dwW, b: nil, groups: D)                  // ⑤ depthwise k31
        ct = ct.transposed(0, 2, 1)                                       // [B,T,C]
        ct = Ops.layerNorm(ct, weight: b.dwLNw, bias: b.dwLNb)            // ⑥ LN over C（非通道 GN）
        ct = Ops.silu(ct)                                                 // ⑦ swish
        ct = ct.transposed(0, 2, 1)                                       // [B,C,T]
        ct = Ops.conv1d(ct, w: b.pw2W, b: nil)                            // ⑧ [B,C,T]
        return ct.transposed(0, 2, 1)
    }

    /// nn.GLU(dim=1)：前半 a 乘以后半的 sigmoid；x [B,2C,T]
    private func glu(_ x: MLXArray) -> MLXArray {
        let c = x.shape[1] / 2
        let a = x[0..., 0..<c, 0...]
        let b = x[0..., c..<(2 * c), 0...]
        return a * Ops.sigmoid(b)
    }

    /// 单 conformer 块（官方 Wav2Vec2BertEncoderLayer：
    ///   Macaron-FFN1 → Self-Attn → ConvModule → Macaron-FFN2 → final_layer_norm）
    private func block(_ x: MLXArray, b: Block) -> MLXArray {
        var y = ffn(x, lnW: b.ffn1LNw, lnB: b.ffn1LNb,
                    iW: b.ffn1IW, iB: b.ffn1Ib, oW: b.ffn1OW, oB: b.ffn1Ob)
        var h = x + y * 0.5
        let attn = attention(h, b: b)
        DLog.write("W2VBLK x=\(x.shape) ffn=\(y.shape) attn=\(attn.shape)")
        h = h + attn
        DLog.write("W2VBLK h_after_attn=\(h.shape) ndim=\(h.ndim)")
        h = h + convModule(h, b: b)
        y = ffn(h, lnW: b.ffn2LNw, lnB: b.ffn2LNb,
                iW: b.ffn2IW, iB: b.ffn2Ib, oW: b.ffn2OW, oB: b.ffn2Ob)
        h = h + y * 0.5
        // ⚠️ 官方 Conformer 块末尾还有 final_layer_norm（finLNw/finLNb 此前定义了却从未使用）
        return Ops.layerNorm(h, weight: b.finLNw, bias: b.finLNb)
    }

    // MARK: - 前向

    private func featureProjection(_ x: MLXArray) -> MLXArray {
        // 官方 Wav2Vec2BertFeatureProjection：LayerNorm(160) → Linear(160→1024)
        // x [B,T,160] → LN(160) → [B,T,1024]
        let h = Ops.layerNorm(x, weight: featLN_w, bias: featLN_b)
        return Ops.linear(h, w: featProjW, b: featProjB)
    }

    /// 前向：返回 hiddenStates[i] 输出。targetIndex=17 → 与官方 hidden_states[17] 对齐（0=投影后, 1..24=各层后）
    public func hiddenState(_ features: MLXArray, targetIndex: Int = 17) throws -> MLXArray {
        // features [B,T,160]
        DLog.write("W2V hiddenState in=\(features.shape) target=\(targetIndex)")
        let dump = dumpForCalibration
        return try MLX.withError {
            // 必须先导出「本次实际输入」：Windows 侧要用同一份 input_feat 跑官方模型生成 golden，
            // 否则两端输入不同，逐层比对无意义（golden 里的 tone_feat 仅用于自检）。
            if dump { dumpLayer(features, index: 0, kind: "input") }
            var h = featureProjection(features)
            if dump { dumpLayer(h, index: 0, kind: "proj") }
            if targetIndex == 0 { return h }
            for i in 0..<cfg.layers {
                h = block(h, b: blocks[i])
                if dump { dumpLayer(h, index: i + 1, kind: "layer") }
                if targetIndex == i + 1 {
                    DLog.write("W2V hiddenState out=\(h.shape) layer=\(i + 1)")
                    return h
                }
            }
            throw SafetensorsError.missing("layer \(targetIndex) out of range")
        }
    }

    /// 数值对拍导出：日志写 shape/mean/absmax/前 8 值（与 Python golden 的统计量直接可比），
    /// 完整张量写 Documents/w2v_dump/<kind>_<index>.bin（float32 小端；
    /// Python 侧 `np.fromfile(p, '<f4').reshape(T,1024)` 即可还原做逐元素比对）。
    private func dumpLayer(_ h: MLXArray, index: Int, kind: String) {
        let f = h.asType(.float32).asFloatArray()
        guard !f.isEmpty else {
            DLog.write("W2VDUMP \(kind)_\(index) EMPTY shape=\(h.shape)")
            return
        }
        let n = Float(f.count)
        let mean = f.reduce(0, +) / n
        var acc: Float = 0
        for v in f { let d = v - mean; acc += d * d }
        let std = (acc / n).squareRoot()
        let absmax = f.map { abs($0) }.max() ?? 0
        let head = f.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ",")
        DLog.write(String(format: "W2VDUMP %@_%d shape=%@ mean=%.6f std=%.6f absmax=%.6f head=[%@]",
                          kind, index, "\(h.shape)", mean, std, absmax, head))
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("w2v_dump")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = f.withUnsafeBufferPointer { Data(buffer: $0) }
        try? data.write(to: dir.appendingPathComponent("\(kind)_\(index).bin"))
    }
}