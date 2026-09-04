import Foundation

/// 自研音频特征化（V2A：参考音频 → 模型输入特征），纯 Swift 实现、零三方依赖。
///
/// 设计目标（0 到 1）：
///   1. 复数 FFT（基-2 迭代）自行实现 —— 不依赖 vDSP / Accelerate / MLXFFT；
///   2. STFT 功率谱 → mel 滤组 → log1p → 逐通道时间 z-score，整链自定义；
///   3. w2v-bert 前置"Seamless 前端"：输出 [T, 160] 特征（80×2 双分支，见探测记录）；
///   4. campplus 前置 kaldi fbank80 前端（16k，与 torchaudio.compliance.kaldi.fbank 语义对齐）。
///
/// 数值对拍：`scripts/golden/w2vbert_golden.npz`（官方黑盒导出）为 Swift 实现的
/// 校准基准；fbank 双分支的 ±1 通道微差在 Mac 上用 golden 逐值对拍收敛（见 README P5）。
public enum AudioFeatures {

    // MARK: - 复数 FFT（基-2 迭代，原位蝶形）

    public struct Complex {
        public var re: Float
        public var im: Float
        public init(_ re: Float, _ im: Float) { self.re = re; self.im = im }
    }

    /// 迭代基-2 FFT（N 须为 2 的幂；非幂先补零）。
    public static func fft(_ input: [Complex]) -> [Complex] {
        let n = input.count
        guard n > 1 else { return input }
        var a = input
        // 位反转排列
        var j = 0
        for i in 1..<(n - 1) {
            var bit = n >> 1
            while j & bit != 0 {
                j ^= bit
                bit >>= 1
            }
            j ^= bit
            if i < j { a.swapAt(i, j) }
        }
        // 蝶形
        var len = 2
        while len <= n {
            let ang = -2.0 * Float.pi / Float(len)
            let wRe = cos(ang)
            let wIm = sin(ang)
            var half = len >> 1
            var wi = 0
            while wi < n {
                var curRe: Float = 1
                var curIm: Float = 0
                for k in 0..<half {
                    let i0 = wi + k
                    let i1 = i0 + half
                    let xr = a[i1].re * curRe - a[i1].im * curIm
                    let xi = a[i1].re * curIm + a[i1].im * curRe
                    a[i1].re = a[i0].re - xr
                    a[i1].im = a[i0].im - xi
                    a[i0].re += xr
                    a[i0].im += xi
                    let nr = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nr
                }
                wi += len
            }
            len <<= 1
        }
        return a
    }

    /// 补零到 2 的幂
    public static func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }

    /// 单帧功率谱 [nBins]（nFFT/2+1 根谱线；无窗缩放，仅供相对能量比较 + 卷积于 mel 滤组）
    public static func powerSpectrum(_ frame: [Float], nFFT: Int) -> [Float] {
        let n = nextPow2(nFFT)
        var input = [Complex](repeating: .init(0, 0), count: n)
        for (i, s) in frame.enumerated() where i < n { input[i] = Complex(s, 0) }
        let out = fft(input)
        let nBins = min(nFFT / 2 + 1, n / 2 + 1)
        var spec = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            spec[i] = out[i].re * out[i].re + out[i].im * out[i].im
        }
        return spec
    }

    /// Hann 窗
    public static func hann(_ n: Int) -> [Float] {
        (0..<n).map { i in
            0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(n - 1)))
        }
    }

    // MARK: - 帧与 STFT

    public struct STFTConfig {
        public var nFFT: Int
        public var winLength: Int
        public var hop: Int
        public init(nFFT: Int = 400, winLength: Int = 400, hop: Int = 160) {
            self.nFFT = nFFT; self.winLength = winLength; self.hop = hop
        }
    }

    /// 滑动分帧（中心补零模式，与 librosa center=True 对齐：前后补 winLength/2）
    public static func frames(_ samples: [Float], winLength: Int, hop: Int) -> [[Float]] {
        let pad = winLength / 2
        let n = samples.count
        let padded = [Float](repeating: 0, count: pad) + samples + [Float](repeating: 0, count: pad)
        let nFrames = Int(ceil(Float(n) / Float(hop))) + 1
        var out: [[Float]] = []
        out.reserveCapacity(nFrames)
        for i in 0..<nFrames {
            let start = i * hop
            if start + winLength <= padded.count {
                out.append(Array(padded[start..<(start + winLength)]))
            } else {
                // 尾部不足帧补零对齐
                var f = [Float](repeating: 0, count: winLength)
                if start < padded.count {
                    f.replaceSubrange(0..<(padded.count - start), with: padded[start...])
                }
                out.append(f)
            }
        }
        return out
    }

    /// 帧序列 → 功率谱帧 [T, nBins]
    public static func powerSpectra(_ samples: [Float], cfg: STFTConfig, window: [Float]) -> [[Float]] {
        frames(samples, winLength: cfg.winLength, hop: cfg.hop).map { frame in
            let win = frame.enumerated().map { $0.element * window[$0.offset] }
            return powerSpectrum(win, nFFT: cfg.nFFT)
        }
    }

    // MARK: - Mel 滤组

    public static func hzToMel(_ hz: Float) -> Float {
        2595.0 * log10(1.0 + hz / 700.0)
    }

    public static func melToHz(_ mel: Float) -> Float {
        700.0 * (powf(10.0, mel / 2595.0) - 1.0)
    }

    /// 三角 mel 滤波器组（librosa 公式），返回 [nMel, nBins]
    public static func melFilterbank(nFFT: Int, nMel: Int, sr: Int,
                                     fmin: Float = 0, fmax: Float = 8000) -> [[Float]] {
        let nBins = nFFT / 2 + 1
        let fftFreqs = (0..<nBins).map { Float($0) * Float(sr) / Float(nFFT) }
        let melMin = hzToMel(fmin)
        let melMax = hzToMel(fmax)
        let pts = (0...(nMel + 1)).map { i -> Float in
            let frac = Float(i) / Float(nMel + 1)
            return melToHz(melMin + (melMax - melMin) * frac)
        }
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nMel)
        for m in 0..<nMel {
            let lo = pts[m], ct = pts[m + 1], hi = pts[m + 2]
            for b in 0..<nBins {
                let f = fftFreqs[b]
                let up = f <= ct ? (f - lo) / max(ct - lo, 1e-9) : 0
                let dn = (f > ct && f <= hi) ? (hi - f) / max(hi - ct, 1e-9) : 0
                fb[m][b] = max(0, up + dn)
            }
        }
        return fb
    }

    // MARK: - log-mel + z-score

    /// 特征 [T,nMel] → log1p → 逐通道时间 z-score
    public static func logMelZScore(_ mel: [[Float]], eps: Float = 1e-5) -> [[Float]] {
        let t = mel.count
        let c = mel.isEmpty ? 0 : mel[0].count
        var lm = [[Float]](repeating: [Float](repeating: 0, count: c), count: t)
        // log1p
        for i in 0..<t {
            for j in 0..<c { lm[i][j] = log1p(max(0, mel[i][j])) }
        }
        // 每通道 mean/std
        var mean = [Float](repeating: 0, count: c)
        for j in 0..<c {
            var s: Float = 0
            for i in 0..<t { s += lm[i][j] }
            mean[j] = s / Float(t)
        }
        var varx = [Float](repeating: 0, count: c)
        for j in 0..<c {
            var s: Float = 0
            for i in 0..<t { let d = lm[i][j] - mean[j]; s += d * d }
            varx[j] = s / Float(t)
        }
        let std = varx.map { sqrt($0 + eps) }
        for i in 0..<t {
            for j in 0..<c { lm[i][j] = (lm[i][j] - mean[j]) / std[j] }
        }
        return lm
    }

    // MARK: - Seamless 前端（w2v-bert 输入，160 通道双分支）

    /// w2v-bert 前置特征：80 折 log-mel(z-score) 的双分支拼接 → [T,160]。
    /// 探测结论（scripts/*probe*.py）：
    ///   · 基础 = 16k nFFT400/hop160/win400 log1p 80mel[0,8000]，逐通道时间 z-score；
    ///   · 第二分支与主分支同刻高度相关（corr≈0.96，峰值通道差 1-2）—— 是同一滤组的
    ///     近邻频率副本（fmin 微移）。fmin 偏移量留 golden 对拍校准（见 README P5）。
    public struct SeamlessFrontendConfig {
        public var sr: Int = 16000
        public var nFFT: Int = 400
        public var winLength: Int = 400
        public var hop: Int = 160
        public var nMel: Int = 80
        public var fminA: Float = 0       // 主分支
        public var fmaxA: Float = 8000
        public var fminB: Float = 40      // 副分支（对拍校准点）
        public var fmaxB: Float = 8000
        public init() {}
    }

    /// 计算 w2v-bert 输入特征 [T,160]（行序 = 主分支 80 + 副分支 80）
    public static func seamlessFrontend(_ samples: [Float],
                                        cfg: SeamlessFrontendConfig = .init()) -> [[Float]] {
        let window = hann(cfg.winLength)
        let spectra = powerSpectra(samples, cfg: .init(nFFT: cfg.nFFT,
                                                       winLength: cfg.winLength,
                                                       hop: cfg.hop), window: window)
        let fbA = melFilterbank(nFFT: cfg.nFFT, nMel: cfg.nMel, sr: cfg.sr,
                                fmin: cfg.fminA, fmax: cfg.fmaxA)
        let fbB = melFilterbank(nFFT: cfg.nFFT, nMel: cfg.nMel, sr: cfg.sr,
                                fmin: cfg.fminB, fmax: cfg.fmaxB)
        func toMel(_ fb: [[Float]]) -> [[Float]] {
            spectra.map { spec in
                (0..<cfg.nMel).map { m in
                    var acc: Float = 0
                    let row = fb[m]
                    for b in 0..<spec.count { acc += spec[b] * row[b] }
                    return acc
                }
            }
        }
        let melA = logMelZScore(toMel(fbA))   // [T,80]
        let melB = logMelZScore(toMel(fbB))   // [T,80]
        return zip(melA, melB).map { a, b in a + b }
    }

    // MARK: - kaldi fbank80（campplus 前端）

    /// 16k fbank80（对比 torchaudio.compliance.kaldi.fbank(dither=0) 语义）：
    /// nFFT 512 / hop 160 / win 400 / mel[0,8000] / 平方功率 → log → 逐行减均值。
    public static func kaldiFbank80(_ samples: [Float], sr: Int = 16000) -> [[Float]] {
        let nFFT = 512
        let winLength = 400
        let hop = 160
        let nMel = 80
        let window = hann(winLength)
        let spectra = powerSpectra(samples, cfg: .init(nFFT: nFFT, winLength: winLength, hop: hop),
                                   window: window)
        let fb = melFilterbank(nFFT: nFFT, nMel: nMel, sr: sr, fmin: 0, fmax: 8000)
        var out = [[Float]]()
        out.reserveCapacity(spectra.count)
        for spec in spectra {
            var row = [Float](repeating: 0, count: nMel)
            for m in 0..<nMel {
                var acc: Float = 0
                let fbRow = fb[m]
                for b in 0..<spec.count { acc += spec[b] * fbRow[b] }
                row[m] = log(max(acc, 1e-10))
            }
            let mean = row.reduce(0, +) / Float(nMel)
            out.append(row.map { $0 - mean })
        }
        return out
    }

    // MARK: - 重采样（线性插值）

    /// 线性重采样到目标采样率（UI 解码不一定给 16k → 统一到 16k 供特征化）
    public static func resample(_ samples: [Float], from src: Int, to dst: Int) -> [Float] {
        guard src != dst, !samples.isEmpty else { return samples }
        let nOut = Int(floor(Float(samples.count) * Float(dst) / Float(src)))
        var out = [Float](repeating: 0, count: max(nOut, 0))
        guard nOut > 0 else { return samples }
        let ratio = Float(src) / Float(dst)
        for i in 0..<nOut {
            let pos = Float(i) * ratio
            let i0 = Int(floor(pos))
            let frac = pos - Float(i0)
            let i1 = min(i0 + 1, samples.count - 1)
            out[i] = samples[i0] * (1 - frac) + samples[i1] * frac
        }
        return out
    }

    // MARK: - 22.05k mel（ref_mel 用，与 BigVGAN 参数一致）

    /// 参考音频 ref_mel [P,80]（22.05k，nFFT1024/hop256/win1024，与官方 mel_spectrogram 对齐）
    public static func refMel80(_ samples: [Float], sr: Int) -> [[Float]] {
        let nFFT = 1024
        let winLength = 1024
        let hop = 256
        let nMel = 80
        let window = hann(winLength)
        let spectra = powerSpectra(samples, cfg: .init(nFFT: nFFT, winLength: winLength, hop: hop),
                                   window: window)
        let fb = melFilterbank(nFFT: nFFT, nMel: nMel, sr: sr, fmin: 0, fmax: 8000)
        var out = [[Float]]()
        out.reserveCapacity(spectra.count)
        for spec in spectra {
            var row = [Float](repeating: 0, count: nMel)
            for m in 0..<nMel {
                var acc: Float = 0
                let fbRow = fb[m]
                for b in 0..<spec.count { acc += spec[b] * fbRow[b] }
                row[m] = log(max(acc, 1e-6))
            }
            out.append(row)
        }
        return out
    }
}