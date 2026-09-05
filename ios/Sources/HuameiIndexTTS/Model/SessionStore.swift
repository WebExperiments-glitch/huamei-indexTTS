import SwiftUI
import Combine

/// App 全局状态（用户输入、合成进度、播放状态、版本号）
/// Internal Beta 1
final class SessionStore: ObservableObject {

    init() {}

    // MARK: - 用户输入
    @Published var referenceURL: URL?       // 克隆参考音频
    @Published var text: String = "" // 要克隆的文字
    @Published var language: Language       = .zh
    @Published var durationFactor: Double   = 1.0   // 0.5 … 2.0（demo 同款滑条）

    // MARK: - 高级（折叠）
    @Published var showExperimental: Bool   = false
    @Published var developerMode: Bool      = false   // 开发者模式（Beta）
    @Published var doSample: Bool           = true
    @Published var temperature: Double      = 0.8
    @Published var topP: Double             = 0.8
    @Published var topK: Int                = 30
    @Published var numBeams: Int            = 3      // 与官方 demo 默认对齐
    @Published var repetitionPenalty: Double = 10.0 // 官方默认
    @Published var lengthPenalty: Double    = 0.0
    @Published var maxMelTokens: Int        = 1500

    // MARK: - 合成状态
    @Published var phase: Phase = .idle
    @Published var progress: Double         = 0.0   // 0.0 … 1.0
    @Published var lastError: String?

    // MARK: - 播放状态
    @Published var resultURL: URL?

    enum Phase: Equatable {
        case idle
        case loading      // 模型加载中
        case encoding     // 文本 + tokens
        case gpt          // GPT 自回归
        case codec        // codec decode
        case s2mel        // CFM diffusion
        case vocoder      // BigVGAN
        case done
        case failed
    }

    enum Language: String, CaseIterable, Identifiable {
        case zh = "ZH", en = "EN", ja = "JA", es = "ES", ar = "AR"
        var id: String { rawValue }
        var label: String {
            switch self {
            case .zh:  return "中文（简体）"
            case .en:  return "英语（English）"
            case .ja:  return "日语（日本語）"
            case .es:  return "西班牙语（Español）"
            case .ar:  return "阿拉伯语（العربية）"
            }
        }
        /// LANGUAGE_DICT 索引（en=0, zh=1, de=2, es=3, ru=4, ko=5, fr=6, ja=7, …, ar=13）
        /// 与官方 tokenizer.py LANGUAGES 顺序一致（AST 提取验证）
        var langId: Int {
            switch self {
            case .en: return 0
            case .zh: return 1
            case .ja: return 7
            case .es: return 3
            case .ar: return 13
            }
        }
    }

    func reset() {
        phase = .idle
        progress = 0
        lastError = nil
        resultURL = nil
    }

    var canSynthesize: Bool {
        // Stage1：预设音色模式（speakerRow=0），无需参考音频
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && phase != .loading && phase != .encoding
        && phase != .gpt && phase != .codec
        && phase != .s2mel && phase != .vocoder
    }
}