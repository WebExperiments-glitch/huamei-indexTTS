import Foundation
import MLX

/// 端到端合成编排（Stage 1）—— text → wav
///
/// 内存架构：**阶段独占 + 用完即卸（峰值 = max(阶段)，而非求和）**
///
/// ```
/// 合成前               常驻：tokenizer(<10MB) + feat1/feat2(0.5MB)      ≈  0.01GB
/// ① GPT 自回归         载 gpt(1.13GB) → codes([Int] 仅几 KB) → 卸 gpt  峰值 ≈ 1.5-1.9GB
/// ② Codec decode       载 codec(0.10GB) → S_infer → 卸 codec         峰值 ≈ +0.1GB
/// ③ s2mel 扩散         载 s2mel(0.20GB) → mel → 卸 s2mel              峰值 ≈ +0.3GB
/// ④ BigVGAN 声码       载 bigvgan(0.21GB) → wav → 卸 bigvgan         峰值 ≈ +0.3GB
/// 任一时刻至多一个模型驻留 → 理论峰值 ≈ 2GB（8GB 设备余量充足）
/// ```
///
/// 73 组预设音色 = feat1 行做 style + PromptBundle（Mac A1 预计算）做 prompt 条件。
/// A1 进阶：参考音频路径需 w2v-bert/CAMPPlus 输出（独立 VoiceExtractor 会话，同样独占/即卸）。
public final class TTSPipeline {

    // 常驻（小资源）
    private let tokenizer: TiktokenBPE
    private let feat1: [[Float]]      // [73,192]
    private let feat2: [[Float]]      // [73,1280]

    // 完整语言表（langs.json，106 项；缺失时回退前 8 项硬编码）
    private var langNames: [String] = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja"]

    // 模型路径（不持有权重；按阶段惰性载入）
    private let gptPath: String
    private let codecPath: String
    private let s2melPath: String
    private let bigvganPath: String

    // 阶段会话：同一时刻至多一个非 nil
    private var gptGen: GPTGenerator?
    private var codec: SemanticCodecDecoder?
    private var s2mel: S2Mel?
    private var s2melInfer: S2MelInfer?
    private var vocoder: BigVGAN?

    public init(modelDir: URL) throws {
        let dir = modelDir.path
        let tiktokenURL = modelDir.appendingPathComponent("multilingual_zh_ja_yue_char_del.tiktoken")
        let specialsURL = modelDir.appendingPathComponent("specials.json")
        tokenizer = TiktokenBPE(ranks: try TiktokenBPE.parseRanks(url: tiktokenURL),
                                specials: try TiktokenBPE.parseSpecials(url: specialsURL))
        feat1 = try Self.loadJson(dir + "/feat1.json", rows: 73, cols: 192)
        feat2 = try Self.loadJson(dir + "/feat2.json", rows: 73, cols: 1280)
        gptPath = dir + "/gpt.safetensors"
        codecPath = dir + "/codec.safetensors"
        s2melPath = dir + "/s2mel.safetensors"
        bigvganPath = dir + "/bigvgan.safetensors"
        // 注意：init 不载任何大模型 —— 构造轻量，首段（GPT）才真正占内存
        // 语言表：优先读 langs.json（全 106 语言）；缺失/损坏回退前 8 项硬编码
        if let data = try? Data(contentsOf: modelDir.appendingPathComponent("langs.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let all = obj["languages"] as? [String], !all.isEmpty {
            langNames = all
        }
    }

    static func loadJson(_ path: String, rows: Int, cols: Int) throws -> [[Float]] {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let rowsArr = obj as? [[Any]] else {
            var detail = "文件不存在"
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
               let s = attrs[.size] as? Int {
                detail = "大小=\(s)B"
                if let d = try? Data(contentsOf: url) {
                    let head = String(data: d.prefix(64), encoding: .utf8) ?? "非文本"
                    detail += " 头部=\(head)"
                }
            }
            throw NSError(domain: "TTSPipeline", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bad JSON: \((path as NSString).lastPathComponent)（\(detail)）"])
        }
        // 手动 NSNumber → Float 映射（[[Any]] 不能直接用 as? [[Float]] 桥接）
        let arr = rowsArr.map { row in row.map { ($0 as? NSNumber)?.floatValue ?? 0 } }
        guard arr.count == rows, arr.allSatisfy({ $0.count == cols }) else {
            throw NSError(domain: "TTSPipeline", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bad JSON: \((path as NSString).lastPathComponent)（shape 不符 \(arr.count)×\(arr.first?.count ?? 0)）"])
        }
        return arr
    }

    // MARK: - 阶段装载 / 卸载

    private func ensureGPT() throws -> GPTGenerator {
        if let g = gptGen { return g }
        let g = try GPTGenerator(gptPath: gptPath, feat1: feat1, feat2: feat2)
        gptGen = g
        return g
    }

    private func releaseGPT() {
        gptGen = nil
        // 引用归零 → MLX/Metal 统一内存可被系统回收。
        // 若真机实测峰值未如期回落，可在每段末补一次 MLX 缓存清理调用（见 README first-compile）。
    }

    private func ensureCodec() throws -> SemanticCodecDecoder {
        if let c = codec { return c }
        let c = try SemanticCodecDecoder(path: codecPath)
        codec = c
        return c
    }

    private func releaseCodec() { codec = nil }

    private func ensureS2Mel() throws -> S2Mel {
        if let s = s2mel { return s }
        let s = try S2Mel(path: s2melPath)
        s2mel = s
        s2melInfer = S2MelInfer(weights: s)
        return s
    }

    private func releaseS2Mel() { s2mel = nil; s2melInfer = nil }

    private func ensureVocoder() throws -> BigVGAN {
        if let v = vocoder { return v }
        let v = try BigVGAN(path: bigvganPath)
        vocoder = v
        return v
    }

    private func releaseVocoder() { vocoder = nil }

    /// 释放全部模型权重（合成结束/出错时兜底调用）
    public func unloadAll() {
        releaseGPT(); releaseCodec(); releaseS2Mel(); releaseVocoder()
    }

    deinit { unloadAll() }

    // MARK: - 主合成入口

    /// 主合成入口（内存安全：任何路径离开前 unloadAll）
    /// - Parameters:
    ///   - text: 要合成的文本（不含语言前缀；内部处理）
    ///   - langId: LANGUAGE_DICT[lang]（en=0, zh=1, …）
    ///   - speakerRow: feat1 预设行（0..72）
    ///   - emoWeight: 8 维权向量（默认 happy=1）
    ///   - prompt: 预计算条件束（speakerRow 对应的 prompt_condition/ref_mel），nil 时用零占位
    public func synthesize(text: String,
                           langId: Int,
                           speakerRow: Int,
                           emoWeight: [Float],
                           config: SynthesizeConfig,
                           seed: UInt64,
                           prompt: PromptBundle?,
                           styleOverride: [Float]? = nil,
                           onStage: (String, Double) -> Void) throws -> [Float] {

        defer { unloadAll() }                       // 全路径兜底（含抛错）

        onStage("encoding", 0.1)
        // 前缀语言：num_languages=99 → 仅 langId < 99 存在 <|lang|> 特殊 token。
        // ⚠️ ar=13 等 ≥8 的语言此前被错误回退到不存在的 <|common|>（common idx=105）→ 现查全表
        let langName: String? = (langId >= 0 && langId < 99 && langId < langNames.count)
            ? langNames[langId] : nil
        let prefix = langName.map { "<|\($0)|> " } ?? ""
        let lower = (langName == "en" || langName == "zh" || langName == "ja")
        let prefixed = prefix + (lower ? text.lowercased() : text)
        let ids = tokenizer.encode(prefixed)
        _ = ids

        // A2：克隆声纹欢迎 styleOverride（192），否则走 73 组预设行
        let style = styleOverride ?? feat1[speakerRow]

        // ① GPT 自回归 —— codes 落内存后立即卸载 gpt（1.13GB 不跨阶段）
        let codes: [Int]
        do {
            let g = try ensureGPT()
            onStage("gpt", 0.35)
            codes = try g.generate(tokenIds: ids,
                                   langId: langId,
                                   style: style,
                                   emoWeight: emoWeight,
                                   config: config,
                                   seed: seed,
                                   onToken: nil)
            releaseGPT()
        }

        // ② Codec —— S_infer 落内存后卸载 codec
        let sInfer: MLXArray
        do {
            let c = try ensureCodec()
            onStage("codec", 0.55)
            sInfer = try c.decode(codes: codes)              // [1,2T,1024]
            releaseCodec()
        }

        // ③ s2mel（LengthRegulator + CFM 扩散）—— mel 落内存后卸载 s2mel
        let mel: MLXArray
        do {
            let sm = try ensureS2Mel()
            guard let infer = s2melInfer else { throw SafetensorsError.badJSON("s2mel session unloaded") }
            onStage("s2mel", 0.75)
            let targetLen = Int(Double(sInfer.shape[1]) * TTSConfig.lengthRatio * config.durationFactor)
            let cond = infer.lengthRegulate(sInfer, targetLen: targetLen)   // [1,T',512]

            let promptLen = prompt?.refMel.shape[2] ?? 86
            let catMu: MLXArray
            let promptMel: MLXArray
            if let pb = prompt {
                catMu = MLX.concatenated([pb.promptCondition.asType(cond.dtype), cond], axis: 1)
                promptMel = pb.refMel
            } else {
                // 零占位 prompt（管线验证路径）
                let zeros = MLXArray.zeros([1, promptLen, 512]).asType(cond.dtype)   // ⚠️ 与 cond 同 dtype
                catMu = MLX.concatenated([zeros, cond], axis: 1)
                promptMel = MLXArray.zeros([1, 80, promptLen])
            }
            let styleArr = MLXArray(style, [1, TTSConfig.spkDim])
            let melFull = infer.cfm(mu: catMu, prompt: promptMel,
                                    style: styleArr, seed: seed,
                                    onStep: { s, n in
                                        onStage("s2mel", 0.75 + 0.2 * Double(s) / Double(n))
                                    })
            // 裁掉 prompt 段
            let total = melFull.shape[2]
            mel = total > promptLen ? melFull[0..., 0..., promptLen..<total] : melFull
            releaseS2Mel()
        }

        // ④ BigVGAN —— 波形落内存后卸载 bigvgan
        let floats: [Float]
        do {
            let v = try ensureVocoder()
            onStage("vocoder", 0.95)
            let wav = v.synthesize(mel: mel)               // [1,1,N]
            floats = wav[0..., 0, 0...].reshaped([-1]).asFloatArray()
            releaseVocoder()
        }

        onStage("done", 1.0)
        return floats
    }
}

/// 预计算说话人条件束（A1：Mac 端产出，iOS 只读）
public struct PromptBundle {
    public let promptCondition: MLXArray   // [1,P,512]
    public let refMel: MLXArray            // [1,80,P]
    public init(prompt: MLXArray, ref: MLXArray) { promptCondition = prompt; refMel = ref }
}
