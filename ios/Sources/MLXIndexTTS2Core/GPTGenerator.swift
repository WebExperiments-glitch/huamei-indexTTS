import Foundation
import MLX

/// GPT 自回归生成器（对齐 infer_v2_5.inference_speech + p0/gpt_gen）
/// 流程：条件(3行) ∥ [start]<text>[stop] → +start_mel(位置0) 全量前向 → 逐 token KV 增量采样
///
/// 嵌入行一律走 SafetensorsFile.row（CPU 行读取，避开 MLX 高维切片不确定性）。
public final class GPTGenerator {

    private let store: GPTStore
    private let model: GPT2
    private let head: GPTHead
    private let feat1: [[Float]]        // [73, 192]
    private let feat2: [[Float]]        // [73, 1280]

    public init(gptPath: String, feat1: [[Float]], feat2: [[Float]]) throws {
        store = try GPTStore(path: gptPath)
        model = try GPT2(store: store)
        head = try GPTHead(store: store)
        self.feat1 = feat1
        self.feat2 = feat2
    }

    public func close() { store.close() }

    /// 行 → MLXArray [1,D]（file 直读）
    private func rowVec(_ name: String, _ i: Int) throws -> MLXArray {
        let floats = try store.row(name, index: i)
        return MLXArray(floats, [1, floats.count])
    }

    // MARK: - 73 组情绪（官方 emovec_mat 公式）

    public func emoVector(style: [Float], weight: [Float]) -> [Float] {
        var acc = [Float](repeating: 0, count: TTSConfig.gptDim)
        var off = 0
        for g in 0..<TTSConfig.emoNum.count {
            let n = TTSConfig.emoNum[g]
            var best = 0; var bestCos: Float = -2
            for j in 0..<n {
                let c = cosine(feat1[off + j], style)
                if c > bestCos { bestCos = c; best = j }
            }
            let w = weight[g]
            if w != 0 {
                for k in 0..<TTSConfig.gptDim { acc[k] += w * feat2[off + best][k] }
            }
            off += n
        }
        return acc
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        return dot / max(1e-9, (na * nb).squareRoot())
    }

    // MARK: - 生成

    /// tokenIds：tokenizer 输出（含 `<|lang|>` 前缀）；stop(1) 由内部追加
    public func generate(tokenIds ids: [Int],
                         langId: Int,
                         style: [Float],
                         emoWeight: [Float],
                         config: SynthesizeConfig,
                         seed: UInt64,
                         onToken: ((Int, Int) -> Void)?) throws -> [Int] {

        var rng = SplitMix64(seed: seed)
        let d = TTSConfig.gptDim

        // ---- 前缀：conds ∥ [start]text[stop] ----
        let emo = emoVector(style: style, weight: emoWeight)          // [1280]
        var cond0 = Ops.linear(MLXArray(style, [1, TTSConfig.spkDim]),
                               w: try store.required("spk_emb_proj.weight"),
                               b: try store.required("spk_emb_proj.bias"))
        cond0 = cond0 + MLXArray(emo, [1, d])                         // [1,1280]

        // 文本序列：过滤已在 tokenizer 加入的 0/1（保留语言前缀 token）
        let clean = ids.filter { $0 != TTSConfig.startText && $0 != TTSConfig.stopText }
        let textSeq = [TTSConfig.startText] + clean + [TTSConfig.stopText]
        let L = textSeq.count

        // conds[3,D] + 文本行
        var prefixEmb = cond0
        prefixEmb = MLX.concatenated([prefixEmb, MLXArray.zeros([1, d])], axis: 0)
        prefixEmb = MLX.concatenated([prefixEmb, MLXArray.zeros([1, d])], axis: 0)   // [3,1280]

        let langRow = try rowVec("lang_embedding.weight", min(langId, TTSConfig.langCount - 1))
        for (pos, tok) in textSeq.enumerated() {
            var row = try rowVec("text_embedding.weight", tok)
            let tpos = try rowVec("text_pos_embedding.emb.weight", pos)
            row = row + tpos + langRow
            prefixEmb = MLX.concatenated([prefixEmb, row], axis: 0)
        }
        // [3+L, D]

        // ---- 自回归 ----
        let maxPos = 1 + config.maxMelTokens + L + 3
        let ropeCS = Rotary.precompute(seqLen: maxPos)
        var kvK = [MLXArray?](repeating: nil, count: TTSConfig.gptLayers)
        var kvV = [MLXArray?](repeating: nil, count: TTSConfig.gptLayers)

        // 首 token：start_mel(8192) 位置 0
        let startRow = try rowVec("mel_embedding.weight", TTSConfig.startMel)
        let startPos0 = try rowVec("mel_pos_embedding.emb.weight", 0)
        var fullSeq = MLX.concatenated([prefixEmb, startRow + startPos0], axis: 0)   // [4+L,D]
        fullSeq = fullSeq.reshaped([1, fullSeq.shape[0], d])

        var hidden = model.forward(fullSeq, cacheK: &kvK, cacheV: &kvV,
                                   startPos: 0, ropeCosSin: ropeCS,
                                   causalFull: true)

        var codes: [Int] = []
        var prev = Set<Int>()
        var step = 1
        while codes.count < config.maxMelTokens {
            var logits = head.logits(hidden).reshaped([-1]).asFloatArray()
            SamplerUtil.applyRepetitionPenalty(logits: &logits,
                                               previous: prev,
                                               penalty: Float(config.repetitionPenalty))
            let tok = SamplerUtil.sampleCPU(logits: logits,
                                            topK: config.topK,
                                            topP: Float(config.topP),
                                            temperature: Float(config.temperature),
                                            rng: &rng)
            onToken?(tok, step)
            if tok == TTSConfig.stopMel { break }
            codes.append(tok)
            prev.insert(tok)
            // 下一 mel token：位置 = step（start=0，首个生成 token 位置 1）
            let row = try rowVec("mel_embedding.weight", tok)
            let pos = try rowVec("mel_pos_embedding.emb.weight", step)
            let emb = (row + pos).reshaped([1, 1, d])
            hidden = model.forward(emb, cacheK: &kvK, cacheV: &kvV,
                                   startPos: step, ropeCosSin: ropeCS,
                                   causalFull: false)
            step += 1
        }
        return codes
    }
}