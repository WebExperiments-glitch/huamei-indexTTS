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

    /// 转置卷积上采样（官方 ConvTranspose1d 语义）
    /// 配置：stride=upsRates[4,4,2,2,2,2]，kernel=upsKernels[8,8,4,4,4,4]，
    ///      padding=(k-stride)/2 → 输出长度精确 = L×stride（无需裁剪）。
    /// 真实权重 ups.\(i).weight [C_out,k,C_in] 即 MLX 布局，原样直传。
    /// ⚠️ 旧实现「nearest×rate+conv」是错误近似，已替换为真实转置卷积。
    private func upsample(_ x: MLXArray, stage: Int) -> MLXArray {
        let (w, b) = ups[stage]
        let rate = TTSConfig.upsRates[stage]
        let k = w.shape[1]
        let pad = (k - rate) / 2
        let xt = x.transposed(0, 2, 1)                         // [B,T,C_in]
        var o = MLX.convTransposed1d(xt, w, stride: rate, padding: pad)  // [B,L*rate,C_out]
        o = o + b.reshaped([1, 1, -1])
        return o.transposed(0, 2, 1)                           // [B,C_out,L*rate]
    }

    /// mel [1,80,T] → wav [1,1, T*256]
    public func synthesize(mel: MLXArray) throws -> MLXArray {
        DLog.write("BigVGAN begin mel=\(mel.shape) dtype=\(mel.dtype)")
        do {
            try MLX.withError {
                var x = Ops.zeroPad(mel, left: 3, right: 3)
                x = Ops.conv1d(x, w: convPreW, b: convPreB)          // [1,1536,T]
                for stage in 0..<6 {
                    x = upsample(x, stage: stage)                    // ×rate 逐级上采样
                    DLog.write("BigVGAN stage\(stage) up x=\(x.shape)")
                    var acc: MLXArray?
                    for j in 0..<3 {
                        let blk = blocks[stage * 3 + j]
                        let o = blk.call(x)
                        acc = (acc == nil) ? o : acc! + o
                    }
                    x = acc! / 3.0
                    MLX.eval(x)
                    DLog.write("BigVGAN stage\(stage) ok x=\(x.shape)")
                }
                // post snake → conv_post(1,7,24) → tanh
                x = snakeBetaOp(x, alpha: postAlpha, beta: postBeta)
                x = Ops.zeroPad(x, left: 3, right: 3)
                x = Ops.conv1d(x, w: convPostW, b: nil)
                x = MLX.tanh(x)
                MLX.eval(x)
                DLog.write("BigVGAN done x=\(x.shape)")
                return x
            }
        } catch {
            DLog.write("BigVGAN ERROR: \(String(describing: error))")
            throw error
        }
    }
}

/// snake beta（全局函数，避免嵌套类型访问外层实例成员）：y = x + sin²(x·α)/β
private func snakeBetaOp(_ x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray {
    let a = alpha.exp()
    let b = beta.exp()
    let sin = MLX.sin(x * a)
    return x + (sin * sin) / (b + 1e-9)
}