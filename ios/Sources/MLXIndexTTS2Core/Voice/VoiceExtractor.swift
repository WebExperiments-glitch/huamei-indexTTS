import Foundation
import MLX

/// P5/A2 实现：设备端"随机音频即时克隆"完整链路
///
/// 计算链（与官方 infer_v2_5.py 参考缓存段 + scripts/prepare_voicepack.py 同构）：
/// ```
/// wav ─┬→ 16k ─ seamless 前端(80×2 log-mel z-score) ─ w2v-bert layer17
///      │      → (x-mean)/std ─ LengthRegulator(ylens=P) ──→ prompt[P,512]
///      ├→ 16k ─ kaldi fbank80 ─ CAMPPlus ──────────────────→ style[192]
///      └→ 22.05k ─ mel80(1024/256) ────────────────────────→ refmel[P,80]
/// ```
/// 内存策略：任意时刻至多一个重模型驻留（提取时 w2v/campplus/s2mel 独占会话）。
///
/// ⚠️ 两处数值对拍校准点（.npz golden 已存 scripts/golden/，Mac 端对拍收敛）：
///     1) seamless 前端 160 通道双分支的 fmin 偏移（SeamlessFrontendConfig.fminB）；
///     2) w2v-bert relative_key 注意力的相对项公式（W2VBert.Attention）。
public struct VoiceprintBundle: Codable {
    /// 稳定指纹（缓存键）
    public let id: String
    /// 192 维说话人嵌入（与 feat1 行同构；GPT 条件）
    public let style: [Float]
    /// s2mel prompt 条件 [P,512]（行主序展开）
    public let promptCondition: [Float]
    /// 参考 mel [P,80]（行主序展开；s2mel 端 refMel 需 [1,80,P]）
    public let refMel: [Float]
    public let melFrames: Int
    public let sourceName: String

    public init(id: String, style: [Float], promptCondition: [Float],
                refMel: [Float], melFrames: Int, sourceName: String) {
        self.id = id; self.style = style
        self.promptCondition = promptCondition
        self.refMel = refMel
        self.melFrames = melFrames
        self.sourceName = sourceName
    }
}

public final class VoiceExtractor {

    public enum VoiceError: Error, Equatable {
        case componentNotInstalled
        case badAudio
        case cacheIO(String)
    }

    /// 模型目录约定（Documents/huamei-models/ 内追加组件）
    public struct ComponentPaths {
        public var w2vBert: String      // w2v-bert-2.0/model.safetensors（按需下载组件）
        public var campplus: String     // campplus_cn_common.safetensors
        public var s2mel: String        // s2mel.safetensors（LR 权重）
        public var statsJSON: String    // wav2vec2bert_stats.json
        public init(w2vBert: String, campplus: String, s2mel: String, statsJSON: String) {
            self.w2vBert = w2vBert; self.campplus = campplus
            self.s2mel = s2mel; self.statsJSON = statsJSON
        }
    }

    private let paths: ComponentPaths

    // 独占会话句柄（用完即卸）
    private var w2v: W2VBert?
    private var camp: Campplus?
    private var s2mel: S2MelInfer?

    public init(paths: ComponentPaths) {
        self.paths = paths
    }

    /// 组件是否已安装（有全部必需文件才算 ready）
    public static func hasComponent(_ paths: ComponentPaths) -> Bool {
        FileManager.default.fileExists(atPath: paths.w2vBert)
            && FileManager.default.fileExists(atPath: paths.campplus)
            && FileManager.default.fileExists(atPath: paths.s2mel)
            && FileManager.default.fileExists(atPath: paths.statsJSON)
    }

    // MARK: - 主入口

    /// 提取/克隆一段音频的声纹束。缓存优先；miss 才跑模型。
    /// - Parameters:
    ///   - audioURL: 参考音频（缓存键来源）
    ///   - pcm16k: 16kHz mono PCM（UI 层 AVFoundation 解码产物；nil 时 0 到 1 解码留待 UI）
    ///   - frontend: 特征化配置（含对拍校准参数）
    public func voiceprint(audioURL: URL,
                           pcm16k: [Float]?,
                           frontend: AudioFeatures.SeamlessFrontendConfig = .init()) throws -> VoiceprintBundle {
        let id = try Voiceprint.stableID(audioURL: audioURL)
        // 1) 缓存优先
        if let cached = try loadBundle(id: id) { return cached }
        guard let pcm = pcm16k, !pcm.isEmpty else { throw VoiceError.badAudio }
        guard Self.hasComponent(paths) else { throw VoiceError.componentNotInstalled }

        // 2) 独占会话：style + prompt + refmel
        let style = try extractStyle(pcm16k: pcm)
        let (prompt, melFrames) = try extractPrompt(pcm16k: pcm, fminB: frontend.fminB)
        let refMel = extractRefMel(pcm16k: pcm)
        let bundle = VoiceprintBundle(id: id, style: style,
                                      promptCondition: prompt,
                                      refMel: refMel,
                                      melFrames: melFrames,
                                      sourceName: audioURL.lastPathComponent)
        unloadAll()
        try saveBundle(bundle)
        return bundle
    }

    // MARK: - 三段式提取

    private func extractStyle(pcm16k: [Float]) throws -> [Float] {
        if camp == nil { camp = try Campplus(path: paths.campplus) }
        let fbank = AudioFeatures.kaldiFbank80(pcm16k)          // [T,80]
        let B = fbank.count == 0 ? 0 : fbank.count
        guard B > 0 else { throw VoiceError.badAudio }
        let C = fbank[0].count
        let x = MLXArray(fbank.flatMap { $0 }, [1, B, C])
        let emb = try camp!.embed(x)                            // [B,192]
        let arr = emb[0..., 0...].reshaped([-1]).asFloatArray()
        return arr
    }

    private func extractPrompt(pcm16k: [Float], fminB: Float) throws -> ([Float], Int) {
        // 160 通道前端
        var feCfg = AudioFeatures.SeamlessFrontendConfig()
        feCfg.fminB = fminB
        let feat = AudioFeatures.seamlessFrontend(pcm16k, cfg: feCfg)   // [T,160]
        guard !feat.isEmpty else { throw VoiceError.badAudio }
        let T = feat.count
        let x = MLXArray(feat.flatMap { $0 }, [1, T, 160])
        if w2v == nil { w2v = try W2VBert(path: paths.w2vBert) }
        var h = try w2v!.hiddenState(x, targetIndex: 17)        // [1,T16,1024]

        // 归一化（stats）
        let stats = try Self.loadStats(path: paths.statsJSON)   // mean/var [1024]
        let mean = MLXArray(stats.0, [1, 1, 1024])
        let std = MLX.sqrt(MLXArray(stats.1, [1, 1, 1024]))
        h = (h - mean) / std

        // ref_mel 帧数 P（22.05k path 同产）
        let pcm22 = AudioFeatures.resample(pcm16k, from: 16000, to: 22050)
        let refmel = AudioFeatures.refMel80(pcm22, sr: 22050)   // [P,80]
        let P = refmel.count

        // LengthRegulator → prompt [1,P,512]
        if s2mel == nil {
            let w = try S2Mel(path: paths.s2mel)
            s2mel = S2MelInfer(weights: w)
        }
        let prompt = s2mel!.lengthRegulate(h, targetLen: P) // [1,P,512]
        let arr = prompt[0..., 0...].reshaped([-1]).asFloatArray()
        return (arr, P)
    }

    private func extractRefMel(pcm16k: [Float]) -> [Float] {
        let pcm22 = AudioFeatures.resample(pcm16k, from: 16000, to: 22050)
        return AudioFeatures.refMel80(pcm22, sr: 22050).flatMap { $0 }   // [P*80] row-major
    }

    // MARK: - 卸载

    /// 缓存束 → 合成可用 PromptBundle（prompt [1,P,512] / refMel [1,80,P]）
    public func promptBundle(_ b: VoiceprintBundle) -> PromptBundle {
        let pc = MLXArray(b.promptCondition, [1, b.melFrames, 512])
        let rm = MLXArray(b.refMel, [1, b.melFrames, 80]).transposed(0, 2, 1)
        return PromptBundle(prompt: pc, ref: rm)
    }

    public func unloadAll() {
        w2v = nil; camp = nil; s2mel = nil
    }
    deinit { unloadAll() }

    // MARK: - 缓存（binary v1）

    static var cacheDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("voiceprints", isDirectory: true)
    }

    private func bundleURL(id: String) -> URL {
        VoiceExtractor.cacheDir.appendingPathComponent(id + ".vpbin")
    }

    public func loadBundle(id: String) throws -> VoiceprintBundle? {
        let url = bundleURL(id: id)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            // 魔数 + id + 各段
            let magic = data.prefix(4)
            guard magic == Data("VPB1".utf8) else { return nil }
            var off = 4
            func readLen() -> Int {
                let v = data.subdata(in: off..<(off + 4)); off += 4
                return v.withUnsafeBytes { $0.load(as: UInt32.self) }.hashValue & 0x7fffffff
            }
            // 简化：直接读 UInt32 LE
            func readU32() -> Int {
                let v = data.subdata(in: off..<(off + 4)); off += 4
                return Int(v.withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian)
            }
            let idLen = readU32()
            let idStr = String(data: data.subdata(in: off..<(off + idLen)), encoding: .utf8) ?? ""
            off += idLen
            let styleN = readU32()
            let style = readFloats(data, &off, count: styleN)
            let pN = readU32()
            let prompt = readFloats(data, &off, count: pN)
            let rN = readU32()
            let refmel = readFloats(data, &off, count: rN)
            let frames = readU32()
            let srcLen = readU32()
            let src = String(data: data.subdata(in: off..<(off + srcLen)), encoding: .utf8) ?? ""
            return VoiceprintBundle(id: idStr, style: style, promptCondition: prompt,
                                    refMel: refmel, melFrames: frames, sourceName: src)
        } catch {
            throw VoiceError.cacheIO(error.localizedDescription)
        }
    }

    private func readFloats(_ data: Data, _ off: inout Int, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let bytes = data.subdata(in: off..<(off + count * 4))
        off += count * 4
        return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    public func saveBundle(_ b: VoiceprintBundle) throws {
        let dir = VoiceExtractor.cacheDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var out = Data("VPB1".utf8)
        func u32(_ v: Int) { var x = UInt32(v); out.append(Data(bytes: &x, count: 4)) }
        func floats(_ a: [Float]) { a.withUnsafeBytes { out.append(Data($0)) } }
        let idD = Data(b.id.utf8); let srcD = Data(b.sourceName.utf8)
        u32(idD.count); out.append(idD)
        u32(b.style.count); floats(b.style)
        u32(b.promptCondition.count); floats(b.promptCondition)
        u32(b.refMel.count); floats(b.refMel)
        u32(b.melFrames)
        u32(srcD.count); out.append(srcD)
        do {
            try out.write(to: bundleURL(id: b.id), options: .atomic)
        } catch {
            throw VoiceError.cacheIO(error.localizedDescription)
        }
    }

    /// 清空全部声纹缓存（用户"管理声音"入口）
    public func purgeCache() throws {
        let dir = VoiceExtractor.cacheDir
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        let files = (try? FileManager.default.contentsOfDirectory(at: dir,
                                                                  includingPropertiesForKeys: nil)) ?? []
        for f in files where f.pathExtension == "vpbin" {
            try? FileManager.default.removeItem(at: f)
        }
    }

    // MARK: - stats loader

    private struct StatsBox { static var cached: (mean: [Float], std: [Float])? }

    private static func loadStats(path: String) throws -> ([Float], [Float]) {
        if let s = StatsBox.cached { return s }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: [Float]] else {
            throw VoiceError.cacheIO("stats missing")
        }
        let mean = obj["mean"] ?? []
        let varx = obj["var"] ?? []
        let std = varx.map { sqrt(max($0, 1e-6)) }
        let s = (mean, std)
        StatsBox.cached = s
        return s
    }
}