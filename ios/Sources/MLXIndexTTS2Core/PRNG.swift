import Foundation

/// SplitMix64 PRNG（64 位状态，multiplicative mix）
/// 作为 `RandomNumberGenerator` 供给种子化采样（CFM 噪声 / GPT 解码等）。
/// 算法：splitmix64（状态 +⨯ 常数黄金比例 → xorshift-mix），确定性、低开销。
struct SplitMix64: RandomNumberGenerator {

    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}