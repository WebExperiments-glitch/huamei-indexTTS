import Foundation
import MLX

/// BigVGAN v2 vocoder（p0/bigvgan_torch 移植）
/// mel [1,80,T] → wav [1,1,T*256]
public final class BigVGAN {

    private let convPreW: MLXArray, convPreB: MLXArray
    private let convPostW: MLXArray            // 无 bias！
    private let postAlpha: MLXArray, postBeta: MLXArray
    private let ups: [(MLXArray, MLXArray)]    // 6 级
    private let blocks: [AMPBlock]

    public init(path: String) throws {
        let f = try SafetensorsFile(path: path)
        func load(_ k: String) throws -> MLXArray { try f.tensor(k) }
        convPreW = try load("conv_pre.weight"); convPreB = try load("conv_pre.bias")
        convPostW = try load("conv_post.weight")
        postAlpha = try load("activation_post.act.alpha")
        postBeta = try load("activation_post.act.beta")
        var ups: [(MLXArray, MLXArray)] = []
        for i in 0..<6 {
            ups.append((try load("ups.\(i).weight"), try load("ups.\(i).bias")))
        }
        self.ups = ups
        var blocks: [AMPBlock] = []
        for i in 0..<18 { blocks.append(try AMPBlock(load: load, idx: i)) }
        self.blocks = blocks
        f.close()
    }

    struct AMPBlock {
        let kernels: Int                       // 3/7/11
        let convs1: [ConvItem], convs2: [ConvItem]
        let alphas: [MLXArray], betas: [MLXArray]

        struct ConvItem { let w: MLXArray, b: MLXArray }

        init(load: (String) throws -> MLXArray, idx: Int) throws {
            let p = "resblocks.\(idx)."
            // kernel 从权重第 1 维精确推断（3/7/11 循环，但以权重为准）
            let first = try load("\(p)convs1.0.weight")
            kernels = first.shape[1]
            var c1: [ConvItem] = []; var c2: [ConvItem] = []
            var al: [MLXArray] = []; var be: [MLXArray] = []
            for j in 0..<3 {
                c1.append(ConvItem(w: try load("\(p)convs1.\(j).weight"),
                                   b: try load("\(p)convs1.\(j).bias")))
                c2.append(ConvItem(w: try load("\(p)convs2.\(j).weight"),
                                   b: try load("\(p)convs2.\(j).bias")))
                al.append(try load("\(p)activations.\(2*j).act.alpha"))
                be.append(try load("\(p)activations.\(2*j).act.beta"))
                al.append(try load("\(p)activations.\(2*j+1).act.alpha"))
                be.append(try load("\(p)activations.\(2*j+1).act.beta"))
            }
            convs1 = c1; convs2 = c2; alphas = al; betas = be
        }

        func call(_ x: MLXArray) -> MLXArray {
            var h = x
            let dil = [1, 3, 5]
            for j in 0..<3 {
                // snake(alpha,beta) → convs1[dil] → snake → convs2[1]
                var y = snakeBetaOp(h, alpha: alphas[2*j], beta: betas[2*j])
                // ⚠️ padding=kernel 对齐见 README：BigVGAN 用零 pad、dilation pad 规则 (k-1)*d/2
                y = padConv(y, k: kernels, dil: dil[j], conv: convs1[j])
                y = snakeBetaOp(y, alpha: alphas[2*j+1], beta: betas[2*j+1])
                y = padConv(y, k: kernels, dil: 1, conv: convs2[j])
                h = h + y
            }
            return h
        }

        private func padConv(_ x: MLXArray, k: Int, dil: Int, conv: ConvItem) -> MLXArray {
            let pad = (k * dil - dil) / 2
            let xp = Ops.zeroPad(x, left: pad, right: pad)
            return Ops.conv1d(xp, w: conv.w, b: conv.b, dilation: dil)
        }
    }

    /// 上采样：conv1d 插值（ConvTranspose 语义）—— im2col 反卷积兜底
    private func upsample(_ x: MLXArray, stage: Int, targetLen: Int) -> MLXArray {
        let (w, b) = ups[stage]
        // ⚠️ 反卷积：先用 nearest ×rate 再 conv（近似，需修正见 README）
        let rate = TTSConfig.upsRates[stage]
        let up = nearest(x, times: rate)
        let k = w.shape[1]
        let pad = (k - rate) / 2
        var o = Ops.conv1d(up, w: w, b: b)
        // 裁剪到 targetLen（反卷积导致的边缘差）
        if o.shape[2] > targetLen {
            o = o[0..., 0..., 0..<targetLen]
        }
        return o
    }

    private func nearest(_ x: MLXArray, times: Int) -> MLXArray {
        // 纯 MLX 最近邻上采样：无 CPU 数组拷贝（asFloatArray 对 F16 会断言崩溃）
        let (b, c, t) = (x.shape[0], x.shape[1], x.shape[2])
        let e = x.reshaped([b, c, t, 1])
        let cc = MLX.concatenated([MLXArray](repeating: e, count: times), axis: 3)
        return cc.reshaped([b, c, t * times])
    }

    /// mel [1,80,T] → wav [1,1, T*256]
    public func synthesize(mel: MLXArray) -> MLXArray {
        var x = Ops.zeroPad(mel, left: 3, right: 3)
        x = Ops.conv1d(x, w: convPreW, b: convPreB)          // [1,1536,T]
        var curLen = x.shape[2]
        var stageOut: [MLXArray] = []
        for stage in 0..<6 {
            let targetLen = curLen * TTSConfig.upsRates[stage]
            var h = upsample(x, stage: stage, targetLen: targetLen)
            curLen = h.shape[2]
            // 3 个 AMPBlock 求和 /3
            var acc: MLXArray?
            for j in 0..<3 {
                let blk = blocks[stage * 3 + j]
                let o = blk.call(h)
                acc = (acc == nil) ? o : acc! + o
            }
            h = acc! / 3.0
            x = h
        }
        // post snake → conv_post(1,7,24) → tanh
        x = snakeBetaOp(x, alpha: postAlpha, beta: postBeta)
        x = Ops.zeroPad(x, left: 3, right: 3)
        x = Ops.conv1d(x, w: convPostW, b: nil)
        x = MLX.tanh(x)
        return x
    }
}

/// snake beta（全局函数，避免嵌套类型访问外层实例成员）：y = x + sin²(x·α)/β
private func snakeBetaOp(_ x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray {
    let a = alpha.exp()
    let b = beta.exp()
    let sin = MLX.sin(x * a)
    return x + (sin * sin) / (b + 1e-9)
}