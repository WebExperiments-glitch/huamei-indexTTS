import Foundation
import MLX

// MARK: - MLXArray 便捷桥接（CPU 兜底路径）
// ⚠️ asFloatArray：mlx-swift 提供若干取数 API，以下列出最可能者；在 Mac 上按报错选其一。
extension MLXArray {

    /// 取回 [Float]（数组必需路径：RoPE / WAV 输出 / 采样 logits）
    func asFloatArray() -> [Float] {
        // mlx-swift 0.30：asArray(_ type:) 带类型参数（throws）；失败回退空数组
        return (try? self.asArray(Float.self)) ?? []
    }
}

// MARK: - RoPE（CPU 精确 interleave，θ_j = base^(-j/32)，与官方 gpt_fast 一致）
// precompute：对任意 T、head_dim=64，产出 [T,32,2]（cos,sin）
enum Rotary {

    static func precompute(seqLen: Int, headDim: Int = 64,
                           base: Float = 10_000) -> MLXArray {
        let half = headDim / 2
        var freqs = [Float](repeating: 0, count: half)
        for j in 0..<half {
            freqs[j] = powf(base, -Float(2 * j) / Float(headDim))
        }
        var cs = [Float](repeating: 0, count: seqLen * half * 2)
        for t in 0..<seqLen {
            for j in 0..<half {
                let angle = Float(t) * freqs[j]
                cs[(t * half + j) * 2 + 0] = cos(angle)
                cs[(t * half + j) * 2 + 1] = sin(angle)
            }
        }
        return MLXArray(cs, [seqLen, half, 2])
    }

    /// interleave-pair 旋转：x [B,T,H,64]；cs [T,32,2]
    /// 每 pair (x[2j], x[2j+1]) ← (x0·cos−x1·sin, x0·sin+x1·cos)
    /// offset：KV 增量步传绝对起始位置（否则增量步会用局部位置 0）
    static func applyCPU(_ x: MLXArray, cs: MLXArray, offset: Int = 0) -> MLXArray {
        let shape = x.shape
        let (b, t, h, d) = (shape[0], shape[1], shape[2], shape[3])
        let half = d / 2
        let xf = x.asFloatArray()
        let cf = cs.asFloatArray()
        var out = [Float](repeating: 0, count: xf.count)
        for bi in 0..<b {
            for ti in 0..<t {
                let pos = offset + ti
                for hi in 0..<h {
                    let base = ((bi * t + ti) * h + hi) * d
                    for j in 0..<half {
                        let cosv = cf[(pos * half + j) * 2 + 0]
                        let sinv = cf[(pos * half + j) * 2 + 1]
                        let x0 = xf[base + 2 * j]
                        let x1 = xf[base + 2 * j + 1]
                        out[base + 2 * j]     = x0 * cosv - x1 * sinv
                        out[base + 2 * j + 1] = x0 * sinv + x1 * cosv
                    }
                }
            }
        }
        return MLXArray(out, shape)
    }
}

// MARK: - 采样小工具（logits 处理全 CPU [Float]，稳妥）
enum SamplerUtil {

    /// top_k → top_p → temperature → multinomial（单 token）
    /// logits [V]
    static func sampleCPU(logits: [Float], topK: Int, topP: Float,
                          temperature: Float, rng: inout some RandomNumberGenerator) -> Int {
        var logits = logits.map { $0 / max(temperature, 1e-6) }
        let v = logits.count
        // top_k
        if topK > 0 && topK < v {
            let kth = logits.sorted(by: >)[topK - 1]
            for i in 0..<v where logits[i] < kth { logits[i] = -Float.infinity }
        }
        // softmax + top_p
        let maxL = logits.max() ?? 0
        var p = logits.map { exp($0 - maxL) }
        let sum = p.reduce(0, +)
        p = p.map { $0 / sum }
        // top_p 截断
        var order = Array(0..<v).sorted { p[$0] > p[$1] }
        var cum: Float = 0
        var cut = v
        for (i, idx) in order.enumerated() {
            cum += p[idx]
            if cum > topP { cut = i + 1; break }
        }
        if cut < v { order = Array(order[0..<cut]) }
        var cum2: Float = 0
        for idx in order {
            cum2 += p[idx]
            p[idx] = cum2
        }
        let r = Float.random(in: 0..<1, using: &rng)
        for idx in order where p[idx] >= r { return idx }
        return order.last ?? 0
    }

    /// repetition penalty（HF 语义：对出现过 token 的 logits 惩罚）
    static func applyRepetitionPenalty(logits: inout [Float],
                                       previous: Set<Int>,
                                       penalty: Float) {
        guard penalty > 1.0 else { return }
        for idx in previous where idx < logits.count {
            logits[idx] = logits[idx] < 0 ? logits[idx] * penalty : logits[idx] / penalty
        }
    }
}