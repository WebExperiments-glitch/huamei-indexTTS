import Foundation
import MLX

/// 自研算子层 —— 只依赖 MLX core（matmul/add/元素级）
///
/// 与 p0 Python 数值孪生逐模块语义一致（见 audit_final_report.md）：
///   · Linear 权重 [out, in]（MLX 原生布局，无转置）
///   · Conv1d 权重 (out, k, in)
///   · QuantizedLinear：打包 uint32 weight + scales/biases
/// RoPE 统一走 OpsCPU.Rotary（CPU 精确 interleave，见 OpsCPU.swift）
///
/// ⚠️ 若下列 API 与当前 mlx-swift 版本签名不符，见 README「first-compile 修正清单」逐条对照。

enum Ops {

    // MARK: - 归一化

    /// LayerNorm（最后轴）
    static func layerNorm(_ x: MLXArray, weight: MLXArray?, bias: MLXArray?,
                          eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let diff = x - mean
        let varx = (diff * diff).mean(axis: -1, keepDims: true)
        var out = diff * (varx + eps).rsqrt()
        if let w = weight { out = out * w }
        if let b = bias   { out = out + b }
        return out
    }

    /// RMSNorm（E[x²] 精确）
    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let meanSq = (x * x).mean(axis: -1, keepDims: true)
        return (x * (meanSq + eps).rsqrt()) * weight
    }

    // MARK: - 激活

    /// GPT-2 精确 GELU（erf；Abramowitz-Stegun 7.1.26，max err 1.5e-7）
    static func geluNew(_ x: MLXArray) -> MLXArray {
        x * 0.5 * (1 + erfOp(x / 2.0.squareRoot()))
    }

    static func gelu(_ x: MLXArray) -> MLXArray { geluNew(x) }   // Vocos 用精确 erf 同款

    static func erfOp(_ x: MLXArray) -> MLXArray {
        let a1: Float = 0.254_829_592
        let a2: Float = -0.284_496_736
        let a3: Float = 1.421_413_741
        let a4: Float = -1.453_152_027
        let a5: Float = 1.061_405_429
        let p: Float = 0.327_591_1
        let sign = MLX.sign(x)
        let ax = MLX.abs(x)
        let t = 1 / (1 + p * ax)
        let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        return sign * (1 - MLX.exp(-ax * ax) * poly)
    }

    static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
    static func sigmoid(_ x: MLXArray) -> MLXArray { 1 / (1 + MLX.exp(-x)) }
    static func mish(_ x: MLXArray) -> MLXArray { x * MLX.tanh(softplus(x)) }

    static func softplus(_ x: MLXArray) -> MLXArray {
        let ax = MLX.abs(x)
        return MLX.maximum(x, 0) + MLX.log(1 + MLX.exp(-ax))
    }

    // MARK: - 线性

    static func linear(_ x: MLXArray, w: MLXArray, b: MLXArray?) -> MLXArray {
        // 计算用权重 dtype（F16/量化），输出还原输入 dtype，避免下游 F32/F16 拼接崩溃
        let xd = x.dtype
        let xw = x.dtype == w.dtype ? x : x.asType(w.dtype)
        var out = matmul(xw, w.T)
        if let b {
            let bb = b.dtype == out.dtype ? b : b.asType(out.dtype)
            out = out + bb
        }
        return out.dtype == xd ? out : out.asType(xd)
    }

    /// 量化 Linear（affine, group 64, 8bit）
    static func quantizedLinear(_ x: MLXArray,
                                wq: MLXArray, scales: MLXArray,
                                biases: MLXArray?) -> MLXArray {
        let out = MLX.quantizedMatmul(
            x, wq,
            scales: scales,
            biases: biases ?? MLXArray.zeros(like: scales),
            transpose: true,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        return out
    }

    // MARK: - 卷积

    /// conv1d：x [B,C,T]（channels-first 内部语义）；w [out,k,in]。
    /// ⚠️ 真实权重核对（models/mlx-indextts2-2.5-8bit/*.safetensors）：
    ///   全部 conv 权重即 MLX 原生布局 (C_out,K,C_in) —— 如 dwconv [384,7,1]、
    ///   out_project [1024,1,8]、depthwise [512,15,1] —— **不能再转置**！
    /// MLX 的 conv1d 是 channels-last：input (N,L,C_in)、weight (C_out,K,C_in)、output (N,L,C_out)。
    /// 此前崩溃「input (1,8,41) vs weight (1024,1,8)」即 input 布局错误：
    /// input 实为 (N=1,L=8,C=41) 与权重 C_in=8 不匹配 → 修复 = 输入/输出做布局翻转，权重原样。
    /// 输出 [B,out,T]；groups：depthwise（in==1）传 C。
    static func conv1d(_ x: MLXArray, w: MLXArray, b: MLXArray?,
                       dilation: Int = 1, groups: Int = 1,
                       padding: Int = 0, stride: Int = 1) -> MLXArray {
        // 计算用权重 dtype，输出还原输入 dtype（避免下游 F32/F16 拼接崩溃）
        let xd = x.dtype
        let xw = x.dtype == w.dtype ? x : x.asType(w.dtype)
        let xt = xw.transposed(0, 2, 1)                        // [B,T,C]
        var out = MLX.conv1d(xt, w, stride: stride, padding: padding,
                             dilation: dilation, groups: groups)  // [B,T',out]
        if let b {
            let bb = b.dtype == out.dtype ? b : b.asType(out.dtype)
            out = out + bb.reshaped([1, 1, -1])
        }
        out = out.transposed(0, 2, 1)                        // [B,out,T']
        return out.dtype == xd ? out : out.asType(xd)
    }

    /// 零 pad（尾轴）
    static func zeroPad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        guard left > 0 || right > 0 else { return x }
        let (b, c, _) = (x.shape[0], x.shape[1], x.shape[2])
        func zeros(_ n: Int) -> MLXArray {
            // 与 x 同 dtype，避免 F16/F32 混合拼接
            MLXArray.zeros([b, c, n]).asType(x.dtype)
        }
        var out = x
        if left > 0 { out = MLX.concatenated([zeros(left), out], axis: 2) }
        if right > 0 { out = MLX.concatenated([out, zeros(right)], axis: 2) }
        return out
    }

    /// 反射 pad（尾轴；k5 → 2/2）
    /// mlx-swift 无 flip → 对 slice 取行到内存再反转（仅 pad 小段，成本可忽略）
    static func reflectPad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        guard left > 0 || right > 0 else { return x }
        let t = x.shape[2]
        var out = x
        func reversedSlice(_ r: Range<Int>) -> MLXArray {
            let sl = x[0..., 0..., r]
            let b = sl.shape[0], c = sl.shape[1], l = sl.shape[2]
            let f = sl.asFloatArray()
            var rf = [Float](repeating: 0, count: f.count)
            for i in 0..<(b * c) {
                for j in 0..<l { rf[i * l + j] = f[i * l + (l - 1 - j)] }
            }
            return MLXArray(rf, sl.shape)
        }
        if left > 0 { out = MLX.concatenated([reversedSlice(0..<left), out], axis: 2) }
        if right > 0 { out = MLX.concatenated([out, reversedSlice((t - right)..<t)], axis: 2) }
        // reversedSlice 经 CPU F32 兜底后转回原 dtype
        return out.asType(x.dtype)
    }

    // MARK: - 杂项

    static func softmaxLast(_ x: MLXArray) -> MLXArray {
        let maxV = x.max(axis: -1, keepDims: true)
        let e = MLX.exp(x - maxV)
        let s = e.sum(axis: -1, keepDims: true)
        return e / s
    }
}

func matmul(_ a: MLXArray, _ b: MLXArray) -> MLXArray { MLX.matmul(a, b) }
