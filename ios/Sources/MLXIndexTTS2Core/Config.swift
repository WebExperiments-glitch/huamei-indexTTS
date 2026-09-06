import Foundation

/// IndexTTS-2.5 全局常量（与 config.yaml / safetensors 实测一致，见 audit_final_report.md）
public enum TTSConfig {

    // GPT（gpt.safetensors）
    public static let gptLayers  = 24
    public static let gptDim     = 1280
    public static let gptHeads   = 20
    public static let gptHeadDim = 64          // 1280 / 20
    public static let gptMLPDim  = 5120        // c_fc 输出
    public static let textVocab  = 60509       // 58836 base + 1673 specials（embedding 60510 行含 padding）
    public static let melVocab   = 8194        // 8192 codes + start(8192) + stop(8193)
    public static let startMel   = 8192
    public static let stopMel    = 8193
    public static let startText  = 0
    public static let stopText   = 1
    public static let langCount  = 107         // lang_embedding 行数（LANGUAGES 106 + 1）
    public static let spkDim     = 192         // campplus
    public static let emoCount   = 8           // 8 类情绪预设组
    public static let emoNum     = [3, 17, 2, 8, 4, 5, 10, 24]   // 求和 = 73

    // Codec（codec.safetensors）
    public static let codebookSize   = 8192
    public static let codebookDim    = 8
    public static let codecHidden    = 1024
    public static let vocosDim       = 384
    public static let vocosLayers    = 12
    public static let vocosFFN       = 2048

    // S2Mel（s2mel.safetensors）
    public static let melChannels    = 80     // DiT in_channels / mel 频带
    public static let cfmSteps       = 25
    // ⚠️ 诊断：CFG(batch=2) 与单 batch 二分定位崩溃用；定为 0.0 走单 batch（质量略降）。
    //    崩溃定位后再恢复 0.7。
    public static let cfmCfgRate     = 0.0
    public static let ditDim         = 512
    public static let ditLayers      = 13
    public static let ditHeads       = 8
    public static let ditHeadDim     = 64
    public static let ditFFN         = 1536   // w1/w3 输出
    public static let lengthRatio    = 1.72   // LengthRegulator 目标 = S_infer ×1.72 × durationFactor
    public static let wnChannels     = 512
    public static let wnLayers       = 8
    public static let wnKernel       = 5

    // BigVGAN（bigvgan.safetensors）
    public static let sampleRate     = 22_050
    public static let hopLength      = 256
    public static let nFFT           = 1024
    public static let winLength      = 1024
    public static let upsRates       = [4, 4, 2, 2, 2, 2]
    public static let upsKernels     = [8, 8, 4, 4, 4, 4]
    public static let resblockKernels = [3, 7, 11]      // 每 stage 3 个 AMPBlock
    public static let resblockDilations = [1, 3, 5]

    // 生成默认（官方 infer_v2_5 默认）
    public static let topK: Int      = 30
    public static let topP: Float    = 0.8
    public static let temperature: Float = 0.8
    public static let repetitionPenalty: Float = 10.0
    public static let maxMelTokens: Int = 1500
}

/// 合成超参（core 与 UI 共用；对齐官方 demo 默认）
public struct SynthesizeConfig: Equatable {
    public var durationFactor: Double        = 1.0
    public var temperature: Double           = 0.8
    public var topP: Double                  = 0.8
    public var topK: Int                     = 30
    public var repetitionPenalty: Double     = 10.0
    public var maxMelTokens: Int             = 1500
    public init(durationFactor: Double = 1.0,
                temperature: Double = 0.8,
                topP: Double = 0.8,
                topK: Int = 30,
                repetitionPenalty: Double = 10.0,
                maxMelTokens: Int = 1500) {
        self.durationFactor = durationFactor
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.maxMelTokens = maxMelTokens
    }
}
