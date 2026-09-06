import Foundation
import MLX

/// 自研 CAMPPlus（CAM++，speaker embedding [192]）—— 权重来自官方 campplus_cn_common.bin
///
/// 结构（由官方 DTDNN.py 架构 + 权重清单锁定，见 reference/.../campplus/ 与 scripts 转换产物）：
///   input [T,80 kaldi fbank] → head(FCM: StrConv 2D ResNet，80→320)
///   → tdnn(k5 stride2) → block1(12 denseCAM) → transit1 → block2(24) → transit2
///   → block3(16) → transit3 → BN+ReLU → stats pooling(mean+std) → dense(→192, affine-0 BN)
///
/// 数值对拍：与 A1 脚本同一计算链，golden 复用 Mac 端对拍结果（README P5）。
public final class Campplus {

    // head
    private let conv1: (w: MLXArray, b: MLXArray?)   // [32,1,3,3]
    private let bn1: BatchNorm2d
    private let resBlocks: [(b1: MLXArray, bn1: BatchNorm2d, b2: MLXArray, bn2: BatchNorm2d, sc: MLXArray?, scbn: BatchNorm2d?)]
    private let conv2: (w: MLXArray, b: MLXArray?)
    private let bn2: BatchNorm2d

    // xvector
    private struct TDNNBlock { var layers: [CAMDenseLayer] = [] }
    private let tdnn0: (w: MLXArray, bn: BatchNorm1d)
    private var blocks: [TDNNBlock] = []
    private let bnInits: [Int]                   // 每块初始通道（linear1 输入）
    private var transits: [(bn: BatchNorm1d, w: MLXArray)] = []
    private let outBN: BatchNorm1d
    private let denseW: MLXArray                 // [192,1024,1]
    private let denseBN: BatchNorm1d             // affine=false

    public init(path: String) throws {
        let f = try SafetensorsFile(path: path)
        defer { f.close() }
        func t(_ n: String) throws -> MLXArray { try f.tensor(n) }
        // torch 2D 卷积权重 [O,I,Kh,Kw] → MLX [O,Kh,Kw,I]
        func p2(_ x: MLXArray) -> MLXArray { x[0..., 2..., 3..., 1...] }
        // torch 1D 卷积权重 [O,I,K] → MLX [O,K,I]
        func p1(_ x: MLXArray) -> MLXArray { x[0..., 2..., 1...] }

        // ---- head ----
        conv1 = (p2(try t("head.conv1.weight")), nil)
        bn1 = try BatchNorm2d(prefix: "head.bn1", file: f)
        var b1d: [(b1: MLXArray, bn1: BatchNorm2d, b2: MLXArray, bn2: BatchNorm2d,
                   sc: MLXArray?, scbn: BatchNorm2d?)] = []
        for stage in 1...2 {
            let nBlocks = 2
            for bi in 0..<nBlocks {
                let p = "head.layer\(stage).\(bi)."
                let scW = (try? t(p + "shortcut.0.weight")).map(p2)
                let scBN: BatchNorm2d? = (try? BatchNorm2d(prefix: p + "shortcut.1", file: f))
                b1d.append((
                    p2(try t(p + "conv1.weight")), try BatchNorm2d(prefix: p + "bn1", file: f),
                    p2(try t(p + "conv2.weight")), try BatchNorm2d(prefix: p + "bn2", file: f),
                    scW, scBN
                ))
            }
        }
        resBlocks = b1d
        conv2 = (p2(try t("head.conv2.weight")), nil)
        bn2 = try BatchNorm2d(prefix: "head.bn2", file: f)

        // ---- xvector ----
        tdnn0 = (p1(try t("xvector.tdnn.linear.weight")),
                 try BatchNorm1d(prefix: "xvector.tdnn.nonlinear.batchnorm", file: f))
        // 块参数：12/24/16 层，初始通道由前级推出
        let bns = [128, 128, 128]
        let grows = [32, 32, 32]
        let dils = [1, 2, 2]
        var inCh = 128
        for bi in 0..<3 {
            var block = TDNNBlock()
            for li in 0..<[12, 24, 16][bi] {
                let p = "xvector.block\(bi + 1).tdnnd\(li + 1)."
                block.layers.append(CAMDenseLayer(
                    nonlinear1: try BatchNorm1d(prefix: p + "nonlinear1.batchnorm", file: f),
                    linear1W: p1(try t(p + "linear1.weight")),
                    nonlinear2: try BatchNorm1d(prefix: p + "nonlinear2.batchnorm", file: f),
                    camLocalW: p1(try t(p + "cam_layer.linear_local.weight")),
                    camL1W: p1(try t(p + "cam_layer.linear1.weight")),
                    camL1B: try t(p + "cam_layer.linear1.bias"),
                    camL2W: p1(try t(p + "cam_layer.linear2.weight")),
                    camL2B: try t(p + "cam_layer.linear2.bias"),
                    bnIn: bns[bi],
                    out: grows[bi], dil: dils[bi]
                ))
            }
            blocks.append(block)
            inCh += [12, 24, 16][bi] * grows[bi]
            let tp = "xvector.transit\(bi + 1)."
            transits.append((
                try BatchNorm1d(prefix: tp + "nonlinear.batchnorm", file: f),
                p1(try t(tp + "linear.weight"))
            ))
            inCh /= 2
        }
        bnInits = [128]
        outBN = try BatchNorm1d(prefix: "xvector.out_nonlinear.batchnorm", file: f)
        denseW = p1(try t("xvector.dense.linear.weight"))
        denseBN = try BatchNorm1d(prefix: "xvector.dense.nonlinear.batchnorm", file: f,
                                  affine: false)
    }

    // MARK: - 前向

    public func embed(_ fbank: MLXArray) throws -> MLXArray {
        return try MLX.withError {
            // fbank [B,T,80] → [B,80,T] → head 需 4D [B,1,80,T]（官方 head.forward 里 x.unsqueeze(1)）
            var x = fbank.transposed(0, 2, 1)
            x = x.reshaped([x.shape[0], 1, x.shape[1], x.shape[2]])   // [B,1,H=80,W=T]
            DLog.write("CAMP embed fbank=\(fbank.shape) x=\(x.shape) conv1.w=\(conv1.w.shape)")
            x = reluBN2d(bn1, conv2d(x, conv1.w, padding: 1))
            DLog.write("CAMP head.conv1 out=\(x.shape)")
            for (stageIdx, _) in resBlocks.enumerated() {
                // stride(2,1) 作用于每 stage 第一块
                let first = stageIdx % 2 == 0
                x = res(x, r: resBlocks[stageIdx], stride2: first)
            }
            DLog.write("CAMP head resBlocks out=\(x.shape)")
            x = reluBN2d(bn2, downsampleH(conv2d(x, conv2.w, padding: 1)))
            DLog.write("CAMP head.conv2 out=\(x.shape)")
            // reshape [B, C*H, W] → 320
            let (B, C, H, W) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            x = x.reshaped([B, C * H, W])
            DLog.write("CAMP head reshape out=\(x.shape)")

            // xvector
            x = conv1dPadded(x, w: tdnn0.w, dil: 1, pad: 2, stride: 2)   // 128
            x = tdnn0.bn.run(x)
            x = relu(x)
            DLog.write("CAMP tdnn out=\(x.shape)")
            for bi in 0..<blocks.count {
                for layer in blocks[bi].layers {
                    let y = layer.forward(x)
                    x = MLX.concatenated([x, y], axis: 1)
                }
                x = transits[bi].bn.run(x)
                x = relu(x)
                x = conv1dPadded(x, w: transits[bi].w, dil: 1, pad: 0, stride: 1)
            }
            x = outBN.run(x)
            x = relu(x)
            DLog.write("CAMP xvector out=\(x.shape)")
            // stats pooling: [B,C,T] → mean/std → [B,2C]
            let meanX = x.mean(axis: 2)
            let tN = Float(x.shape[2])
            // 无偏 std（除以 T-1）
            let centered = (x - x.mean(axis: 2, keepDims: true))
            let sq = (centered * centered).sum(axis: 2)
            let stdX = MLX.sqrt(sq / (tN - 1))
            x = MLX.concatenated([meanX, stdX], axis: 1)                 // [B,1024]

            // dense 1x1 → [B,1024,1] → [B,192,1] → squeeze
            x = conv1dPadded(x.reshaped([B, 1024, 1]), w: denseW, dil: 1, pad: 0, stride: 1)
            x = x[0..., 0..., 0...].reshaped([B, 192])
            x = denseBN.run(x)
            DLog.write("CAMP dense out=\(x.shape)")
            return x
        }
    }

    // MARK: - 算子

    private func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }

    /// 2D conv：x [N,C,H,W] → channels-last → conv → 翻回 [N,O,H,W]。
    /// ⚠️ padding 按 kernel 传：主分支 k3 用 1；shortcut 是 1×1（官方 padding=0），不能统一写死 1。
    private func conv2d(_ x: MLXArray, _ w: MLXArray, padding: Int = 1) -> MLXArray {
        // x [N,C,H,W] w [O,Kh,Kw,I]；mlx conv2d 为 channels-last：input (N,H,W,C_in)、output (N,H,W,O)。
        let xt = x.transposed(0, 2, 3, 1)                       // [N,H,W,C]
        let o = MLX.conv2d(xt, w, stride: 1, padding: padding)  // [N,H,W,O]
        return o.transposed(0, 3, 1, 2)                          // [N,O,H,W]
    }

    /// 等效 stride(2,1)：H 维隔行取样（reshape 分块取偶，零新 API）
    private func downsampleH(_ x: MLXArray) -> MLXArray {
        let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let r = x.reshaped([n, c, h / 2, 2, w])
        return r[0..., 0..., 0..., 0, 0...]
    }

    private func reluBN2d(_ bn: BatchNorm2d, _ x: MLXArray) -> MLXArray {
        relu(bn.run(x))
    }

    private func conv1dPadded(_ x: MLXArray, w: MLXArray,
                              dil: Int, pad: Int, stride: Int) -> MLXArray {
        var inp = x
        if pad > 0 {
            let B = x.shape[0], C = x.shape[1]
            inp = MLX.concatenated([MLXArray.zeros([B, C, pad]), inp], axis: 2)
            inp = MLX.concatenated([inp, MLXArray.zeros([B, C, pad])], axis: 2)
        }
        // channels-first [B,C,T] → channels-last [B,T,C]（mlx conv1d 布局，权重 [O,K,I] 原样）
        let xt = inp.transposed(0, 2, 1)
        let o = MLX.conv1d(xt, w, stride: stride, padding: 0, dilation: dil)  // [B,T',O]
        return o.transposed(0, 2, 1)                            // [B,O,T']
    }

    private func res(_ x: MLXArray, r: (b1: MLXArray, bn1: BatchNorm2d, b2: MLXArray,
                                        bn2: BatchNorm2d, sc: MLXArray?, scbn: BatchNorm2d?),
                     stride2: Bool) -> MLXArray {
        // ⚠️ shortcut 是 1×1 卷积（官方 layers.py: k=1, stride=(s,1), padding=0）——不能用 k3 的 padding=1
        var out = reluBN2d(r.bn1, stride2 ? downsampleH(conv2d(x, r.b1, padding: 1)) : conv2d(x, r.b1, padding: 1))
        out = r.bn2.run(conv2d(out, r.b2, padding: 1))
        var sc = x
        if let sw = r.sc {
            sc = stride2 ? downsampleH(conv2d(x, sw, padding: 0)) : conv2d(x, sw, padding: 0)
            sc = (r.scbn)!.run(sc)
        }
        return relu(out + sc)
    }
}

// MARK: - BatchNorm（eval：使用训练统计，推理模式见 reference get_nonlinear）

final class BatchNorm1d {
    private let rm: MLXArray, rv: MLXArray
    private let w: MLXArray?, b: MLXArray?
    private let eps: Float = 1e-5

    init(prefix: String, file f: SafetensorsFile, affine: Bool = true) throws {
        rm = try f.tensor(prefix + ".running_mean")
        rv = try f.tensor(prefix + ".running_var")
        if affine {
            w = try f.tensor(prefix + ".weight")
            b = try f.tensor(prefix + ".bias")
        } else {
            w = nil; b = nil
        }
    }

    func run(_ x: MLXArray) -> MLXArray {
        // x [B,C,T]；按通道（轴1）广播
        var out = (x - rm.reshaped([1, -1, 1])) * (rv.reshaped([1, -1, 1]) + eps).rsqrt()
        if let w { out = out * w.reshaped([1, -1, 1]) }
        if let b { out = out + b.reshaped([1, -1, 1]) }
        return out
    }
}

final class BatchNorm2d {
    private let rm: MLXArray, rv: MLXArray
    private let w: MLXArray?, b: MLXArray?
    private let eps: Float = 1e-5

    init(prefix: String, file f: SafetensorsFile, affine: Bool = true) throws {
        rm = try f.tensor(prefix + ".running_mean")
        rv = try f.tensor(prefix + ".running_var")
        if affine {
            w = try f.tensor(prefix + ".weight")
            b = try f.tensor(prefix + ".bias")
        } else {
            w = nil; b = nil
        }
    }

    func run(_ x: MLXArray) -> MLXArray {
        // x [N,C,H,W]；按通道（轴1）
        var out = (x - rm.reshaped([1, -1, 1, 1])) * (rv.reshaped([1, -1, 1, 1]) + eps).rsqrt()
        if let w { out = out * w.reshaped([1, -1, 1, 1]) }
        if let b { out = out + b.reshaped([1, -1, 1, 1]) }
        return out
    }
}

// MARK: - CAM 密集 TDNN 层

struct CAMDenseLayer {
    let nonlinear1: BatchNorm1d
    let linear1W: MLXArray                 // [bn,1,in]
    let nonlinear2: BatchNorm1d
    let camLocalW: MLXArray                // [out,k,bn]
    let camL1W: MLXArray, camL1B: MLXArray  // [bn/2,1,bn]
    let camL2W: MLXArray, camL2B: MLXArray  // [out,1,bn/2]
    let bnIn: Int, out: Int, dil: Int

    func forward(_ x: MLXArray) -> MLXArray {
        // linear1(nonlinear1(x))：1x1 conv 提维到 bn
        var h = nonlinear1.run(x)
        h = conv1x1(h, w: linear1W)
        // CAM：对 bn 通道做通道注意力门
        let gate = attention(h)
        // cam_local：带 dilation 的时间卷积提取帧级特征
        var y = camLocal(h)
        y = y * gate
        return y
    }

    private func conv1x1(_ x: MLXArray, w: MLXArray) -> MLXArray {
        let xt = x.transposed(0, 2, 1)                          // [B,T,C]
        let o = MLX.conv1d(xt, w, stride: 1, padding: 0, dilation: 1)  // [B,T,out]
        return o.transposed(0, 2, 1)                            // [B,out,T]
    }

    private func camLocal(_ x: MLXArray) -> MLXArray {
        // w [out, k, bn]，pad = (k-1)/2*dil
        let pad = dil  // k=3
        var inp = x
        let B = x.shape[0], C = x.shape[1]
        if pad > 0 {
            inp = MLX.concatenated([MLXArray.zeros([B, C, pad]), inp], axis: 2)
            inp = MLX.concatenated([inp, MLXArray.zeros([B, C, pad])], axis: 2)
        }
        let xt = inp.transposed(0, 2, 1)
        let o = MLX.conv1d(xt, camLocalW, stride: 1, padding: 0, dilation: dil)
        return o.transposed(0, 2, 1)
    }

    private func attention(_ x: MLXArray) -> MLXArray {
        // context = mean(轴时间) + seg_pooling(x)  → [B,bn,1]
        let B = x.shape[0], C = x.shape[1], T = x.shape[2]
        let globalMean = x.mean(axis: -1, keepDims: true)          // [B,bn,1]
        let seg = segPooling(x)                                    // [B,bn,1]
        var ctx = globalMean + seg
        // 1x1 conv：channels-last 翻转后运算再翻回 [B,C,1]
        ctx = conv1x1(ctx, w: camL1W) + camL1B.reshaped([1, -1, 1])  // bn→bn/2
        ctx = MLX.maximum(ctx, 0)
        ctx = conv1x1(ctx, w: camL2W) + camL2B.reshaped([1, -1, 1])  // →out
        let gate = 1 / (1 + MLX.exp(-ctx))                          // sigmoid [B,out,1]
        // 广播到时间维
        // gate [B,out,1] 广播到时间维：reshape + 乘法广播（mlx 无 public broadcast）
        return gate.reshaped([B, out, 1]) * MLXArray.ones([1, 1, T])
    }

    private func segPooling(_ x: MLXArray) -> MLXArray {
        let segLen = 100
        let B = x.shape[0], C = x.shape[1], T = x.shape[2]
        let S = (T + segLen - 1) / segLen
        // pad → [B,C,S,segLen] → 均值
        var padded = x
        let padTail = S * segLen - T
        if padTail > 0 {
            padded = MLX.concatenated([padded, MLXArray.zeros([B, C, padTail])], axis: 2)
        }
        let pooled = padded.reshaped([B, C, S, segLen]).mean(axis: 3)   // [B,C,S]
        // 每段扩展 segLen 份，截断回 T
        let expanded = pooled.reshaped([B, C, S, 1]) * MLXArray.ones([1, 1, 1, segLen])
        var out = expanded.reshaped([B, C, S * segLen])
        if out.shape[2] > T {
            out = out[0..., 0..., 0..<T]
        }
        return out
    }
}