import Foundation
import MLX
import MLXRandom

/// s2mel 推理（用 S2MelWeights）+ CFM 采样（p0/s2mel_impl 移植）
public struct S2MelInfer {

    public let w: S2Mel

    public init(weights: S2Mel) { w = weights }

    // MARK: - TimestepEmbedder（transformer 用 t_embedder；wavenet 用 t_embedder2，两套独立权重）
    /// 官方：freqs = exp(-log(max_period)·j/128)（对数间隔）、scale=1000、
    ///      args = scale·t[:,None]·freqs[None] → [B,128]、cat[cos,sin] → [B,256]
    /// ⚠️ t 必须扩成 [B,1]：1D×1D 广播在 B≠128 时报错（CFG batch=2 的崩溃根因）
    private func timeEmb(_ t: MLXArray, w: [MLXArray]) -> MLXArray {
        let freqs: [Float] = (0..<128).map { Float(exp(-log(10_000.0) * Double($0) / 128.0)) }
        let scale: Float = 1000.0
        let tb = t.reshaped([t.shape[0], 1])                           // [B]→[B,1]
        let args = scale * tb * MLXArray(freqs, [1, 128])              // [B,128]
        let emb = MLX.concatenated([args.cos(), args.sin()], axis: 1)  // [B,256]
        var h = Ops.linear(emb, w: w[0], b: w[1])                      // [B,512]
        h = Ops.silu(h)
        return Ops.linear(h, w: w[2], b: w[3])                         // [B,512]
    }
    private func tEmb1(_ t: MLXArray) -> MLXArray { timeEmb(t, w: w.tEmbW) }
    private func tEmb2(_ t: MLXArray) -> MLXArray { timeEmb(t, w: w.t2EmbW) }

    // MARK: - adaLN（transformer 版：split 前 scale 后 shift，直接乘；官方 AdaptiveLayerNorm）
    private func adaln(_ x: MLXArray, c: MLXArray, pw: MLXArray, pb: MLXArray) -> MLXArray {
        // c [B,512] → Linear → [B,1024] → split(scale, shift) [B,512]；x [B,T,512]
        let m = Ops.linear(c, w: pw, b: pb)                  // [B,1024]
        let dim = TTSConfig.ditDim
        let b = m.shape[0]
        let scale = m[0..., 0..<dim].reshaped([b, 1, dim])   // 前 = weight(scale)
        let shift = m[0..., dim..<(dim * 2)].reshaped([b, 1, dim])
        return x * scale + shift
    }

    // MARK: - DiT forward（x/prompt_x/cond/style → velocity）
    public func diTForward(x: MLXArray, promptX: MLXArray, cond: MLXArray,
                           t: MLXArray, style: MLXArray) throws -> MLXArray {
        // x/promptX [1,80,T]；cond [1,T,512]；style [1,192]
        let B = x.shape[0], T = x.shape[2]
        let t1 = tEmb1(t)                                      // [B,512]
        // cond → 512
        var c = cond
        c = Ops.linear(c, w: w.condProjW, b: w.condProjB)                // [1,T,512]
        let xt = x.transposed(0, 2, 1)                                   // [1,T,80]
        let pxt = promptX.transposed(0, 2, 1)
        // cat [x,prompt,cond(512),style(192)] = 864（⚠️ 全部统一 xIn 的 dtype 再拼接）
        var xIn = MLX.concatenated([xt, pxt], axis: 2)
        let cT = c.dtype == xIn.dtype ? c : c.asType(xIn.dtype)
        xIn = MLX.concatenated([xIn, cT], axis: 2)
        let styleT = (style.reshaped([B, 1, TTSConfig.spkDim]) * MLXArray.ones([1, T, 1]))
        xIn = MLX.concatenated([xIn, styleT.asType(xIn.dtype)], axis: 2)
        xIn = Ops.linear(xIn, w: w.condMergeW, b: w.condMergeB)          // [1,T,512]

        // Transformer（13 层，非因果，uvit skip 0-5 → 7-12）
        var h = xIn
        var skipStack: [MLXArray] = []
        let freqT = Rotary.precompute(seqLen: T, headDim: 64)            // 复用 rope 频率
        for (i, blk) in w.tfBlocks.enumerated() {
            var skipIn: MLXArray?
            if i > TTSConfig.ditLayers / 2 { skipIn = skipStack.popLast() }
            if let s = skipIn, let sw = blk.skipW, let sb = blk.skipB {
                h = Ops.linear(MLX.concatenated([h, s], axis: 2), w: sw, b: sb)
            }
            // attn
            let lnA = Ops.rmsNorm(h, weight: blk.anNormW, eps: 1e-5)
            let ca = adaln(lnA, c: t1, pw: blk.anProjW, pb: blk.anProjB)
            // qkv（fused）[B,T,1536] → q|k|v 三段（各 512）
            let qkv = Ops.linear(ca, w: blk.attnWqkv, b: nil)
            let D = TTSConfig.ditDim, HD = TTSConfig.ditHeadDim, H = TTSConfig.ditHeads
            var q = qkv[0..., 0..., 0..<D].reshaped([B, T, H, HD])
            var k = qkv[0..., 0..., D..<(2 * D)].reshaped([B, T, H, HD])
            let v = qkv[0..., 0..., (2 * D)..<(3 * D)].reshaped([B, T, H, HD])
            q = try Rotary.applyCPU(q, cs: freqT)            // [B,T,H,HD]
            k = try Rotary.applyCPU(k, cs: freqT)
            let qB = q.transposed(0, 2, 1, 3)            // [B,H,T,HD]
            let kB = k.transposed(0, 2, 1, 3)
            let vB = v.transposed(0, 2, 1, 3)
            var sc = qB.matmul(kB.transposed(0, 1, 3, 2)) * (1.0 / Float(HD).squareRoot())
            sc = Ops.softmaxLast(sc)
            let ctxOut = sc.matmul(vB)                                    // [B,H,T,HD]
            let ctxFlat = ctxOut.transposed(0, 2, 1, 3).reshaped([B, T, H * HD])  // [B,T,D]
            let aOut = Ops.linear(ctxFlat, w: blk.attnWo, b: nil)
            h = h + aOut
            // ffn
            let lnF = Ops.rmsNorm(h, weight: blk.fnNormW, eps: 1e-5)
            let cf = adaln(lnF, c: t1, pw: blk.fnProjW, pb: blk.fnProjB)
            let g = Ops.silu(Ops.linear(cf, w: blk.ffnW1, b: nil)) * Ops.linear(cf, w: blk.ffnW3, b: nil)
            var fOut = Ops.linear(g, w: blk.ffnW2, b: nil)
            _ = fOut.shape
            h = h + fOut
            if i < TTSConfig.ditLayers / 2 { skipStack.append(h) }
        }
        // tail adaLN
        let lnT = Ops.rmsNorm(h, weight: w.tfNormW, eps: 1e-5)
        h = adaln(lnT, c: t1, pw: w.tfNormProjW, pb: w.tfNormProjB)

        // skip_linear：cat(h, 原始 x80) → 512
        h = Ops.linear(MLX.concatenated([h, xt], axis: 2), w: w.skipLinW, b: w.skipLinB)   // [1,T,512]

        // wavenet 尾
        var wv = Ops.linear(h, w: w.conv1W, b: w.conv1B)                  // [1,T,512]
        wv = wv.transposed(0, 2, 1)                                       // [1,512,T]
        let t2 = tEmb2(t)                                                 // [B,512]
        let g2 = t2.reshaped([B, 512, 1])
        let wav = wavenet(x: wv, g: g2)                                   // [1,512,T]
        var out = wav.transposed(0, 2, 1)                                 // [1,T,512]
        let resAdd = Ops.linear(h, w: w.resProjW, b: w.resProjB)
        out = out + resAdd
        // final layer
        out = finalLayer(out, c: t1)                                      // [1,T,512]
        out = out.transposed(0, 2, 1)                                     // [1,512,T]
        // conv2 (80,1,512) k1
        return Ops.conv1d(out, w: w.conv2W, b: w.conv2B)                  // [1,80,T]
    }

    // MARK: - WaveNet（reflect pad）
    private func wavenet(x: MLXArray, g: MLXArray) -> MLXArray {
        // x [1,512,T] g [1,512,1]
        var gAll = Ops.conv1d(g, w: w.wnCondW, b: w.wnCondB)              // [1,8192,1]
        var output: MLXArray?
        var xm = x
        let ch = TTSConfig.wnChannels
        for i in 0..<TTSConfig.wnLayers {
            // reflect pad 2+2
            let padded = Ops.reflectPad(xm, left: 2, right: 2)
            let xIn = Ops.conv1d(padded, w: w.wnIn[i].w, b: w.wnIn[i].b)  // [1,1024,T]
            let gslice = gAll[0..., (i*2*ch)..<((i+1)*2*ch), 0...]
            let gs = gslice * MLXArray.ones([1, 1, xIn.shape[2]])
            let acts = fusedGate(xIn, gs)
            let (rw, rb) = w.wnResSkip[i]
            var rsa = Ops.conv1d(acts, w: rw, b: rb)                      // [1,1024,T]
            if i < TTSConfig.wnLayers - 1 {
                let res = rsa[0..., 0..<ch, 0...]
                let skip = rsa[0..., ch..<(2*ch), 0...]
                xm = xm + res
                if output == nil { output = skip } else { output = output! + skip }
            } else {
                output = (output ?? rsa) + rsa
            }
        }
        return output ?? x
    }

    private func fusedGate(_ a: MLXArray, _ g: MLXArray) -> MLXArray {
        let t = a + g
        let ch2 = t.shape[1]
        let half = ch2 / 2
        let h1 = t[0..., 0..<half, 0...]
        let h2 = t[0..., half..<ch2, 0...]
        return MLX.tanh(h1) * Ops.sigmoid(h2)
    }

    // MARK: - FinalLayer（官方：LN(no-affine,1e-6) → SiLU→Linear(512→1024) → (shift,scale) → x*(1+scale)+shift → Linear）
    private func finalLayer(_ x: MLXArray, c: MLXArray) -> MLXArray {
        // x [B,T,512]；LN 作用在最后维（512）
        let xN = Ops.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
        // adaLN_modulation = SiLU → Linear(512→1024)；c [B,512]
        let s = Ops.silu(c)                                        // [B,512]
        let m = Ops.linear(s, w: w.finalAdaW, b: w.finalAdaB)      // [B,1024]
        // FinalLayer 约定：前 shift，后 scale，且用 (1+scale)
        let dim = TTSConfig.ditDim
        let b = m.shape[0]
        let shift = m[0..., 0..<dim].reshaped([b, 1, dim])
        let scale = m[0..., dim..<(dim * 2)].reshaped([b, 1, dim])
        let y = xN * (1 + scale) + shift
        return Ops.linear(y, w: w.finalLinW, b: w.finalLinB)       // [B,T,512]
    }

    // MARK: - LengthRegulator
    public func lengthRegulate(_ sInfer: MLXArray, targetLen: Int) -> MLXArray {
        // sInfer [1,T,1024]
        var x = Ops.linear(sInfer, w: w.lrContentProjW, b: w.lrContentProjB)  // [1,T,512]
        x = x.transposed(0, 2, 1)
        x = nearestSize(x, size: targetLen)
        for i in 0..<4 {
            // lrConvs k=3 → padding=1（MLX conv 原生 padding，避免 concat 路径）
            var y = Ops.conv1d(x, w: w.lrConvs[i].0, b: w.lrConvs[i].1, padding: 1)
            y = Ops.layerNorm(y, weight: w.lrNorms[i].0, bias: w.lrNorms[i].1, eps: 1e-5)
            x = Ops.mish(y)
        }
        x = Ops.conv1d(x, w: w.lrFinal.0, b: w.lrFinal.1)
        return x.transposed(0, 2, 1)                                    // [1,T',512]
    }

    private func nearestSize(_ x: MLXArray, size: Int) -> MLXArray {
        // [B,C,T] → 最近邻插值
        let (b, c, t) = (x.shape[0], x.shape[1], x.shape[2])
        let floats = x.asFloatArray()
        var out = [Float](repeating: 0, count: b * c * size)
        for bi in 0..<b {
            for ci in 0..<c {
                for ti in 0..<size {
                    let src = min(t - 1, ti * t / max(1, size))
                    out[(bi * c + ci) * size + ti] = floats[(bi * c + ci) * t + src]
                }
            }
        }
        return MLXArray(out, [b, c, size])
    }

    // MARK: - CFM（欧拉 + CFG）
    public func cfm(mu: MLXArray, prompt: MLXArray, style: MLXArray,
                    steps: Int = TTSConfig.cfmSteps,
                    seed: UInt64) throws -> MLXArray {
        // mu [1,T,512]（prompt 段+目标段）；prompt [1,80,P]
        let B = mu.shape[0], T = mu.shape[1]
        let promptLen = prompt.shape[2]
        // z = randn[1,80,T]（官方随机：MLXRandom.normal，等价 torch.randn_like）
        MLXRandom.seed(seed)
        var x = MLXRandom.normal([1, 80, T])
        var promptX = MLXArray.zeros([1, 80, T])
        // prompt 区锚定：prompt(P) 左对齐 + 零填充（MLX 原生）
        if promptLen > 0 && promptLen <= T {
            let padZ = MLXArray.zeros([1, 80, T - promptLen]).asType(prompt.dtype)
            promptX = MLX.concatenated([prompt[0..., 0..., 0..<promptLen], padZ], axis: 2)
        }
        // x[..., :P] = 0
        x = zeroPrefix(x, count: promptLen)

        let tSpan = (0...steps).map { Float($0) / Float(steps) }
        let cfg = TTSConfig.cfmCfgRate
        var xt = x
        let tt0 = MLXArray([tSpan[0]], [1])
        _ = tt0
        for s in 1...steps {
            let dt = tSpan[s] - tSpan[s - 1]
            let tVal = MLXArray([tSpan[s - 1]], [1])
            if cfg > 0 {
                // stack [prompt,0] × [style,0] × [mu,0] × [x,x]（⚠️ zeros 与拼接对象同 dtype）
                let px2 = MLX.concatenated([promptX, MLXArray.zeros([1, 80, T]).asType(promptX.dtype)], axis: 0)
                let st2 = MLX.concatenated([style, MLXArray.zeros([1, TTSConfig.spkDim]).asType(style.dtype)], axis: 0)
                let mu2 = MLX.concatenated([mu, MLXArray.zeros([1, T, 512]).asType(mu.dtype)], axis: 0)
                let xx = MLX.concatenated([xt, xt], axis: 0)
                let tt = MLXArray([tSpan[s - 1], tSpan[s - 1]], [2])
                var d = try diTForward(x: xx, promptX: px2, cond: mu2, t: tt, style: st2)
                let dHalf = d.shape[0] / 2
                let d1 = d[0..., 0..<dHalf, 0...]
                let d0 = d[0..., dHalf..<d.shape[0], 0...]
                d = (d1 * (1 + cfg)) - (d0 * cfg)
                xt = xt + d * dt
            } else {
                let d = try diTForward(x: xt, promptX: promptX, cond: mu, t: tVal, style: style)
                xt = xt + d * dt
            }
            xt = zeroPrefix(xt, count: promptLen)
            // ⚠️ 每步 eager 求值：让任何 MLX 形状/广播/dtype 错误在循环内立即暴露，
            //    由外层（InferenceEngine）withError 转成可读 throw；不可再嵌套 withError
            //    （嵌套会破坏错误 handler 状态 → Swift 空结果 trap = 此前 brk#1 崩溃）
            MLX.eval(xt)
        }
        return xt
    }

    private func zeroPrefix(_ x: MLXArray, count: Int) -> MLXArray {
        // 前 count 帧置零（MLX 原生，避免手写 CPU 循环/越界）
        let t = x.shape[2]
        guard count > 0, count < t else { return x }
        let zeroPart = x[0..., 0..., 0..<count] * 0
        let rest = x[0..., 0..., count..<t]
        return MLX.concatenated([zeroPart, rest], axis: 2)
    }
}