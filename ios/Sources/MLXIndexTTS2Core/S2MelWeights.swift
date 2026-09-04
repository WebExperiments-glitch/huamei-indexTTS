import Foundation
import MLX

/// s2mel：LengthRegulator + DiT(含 gpt_fast Transformer + WaveNet 尾) + CFM 采样（p0 移植）
public final class S2Mel {

    // ---------- LengthRegulator ----------
    public let lrContentProjW: MLXArray, lrContentProjB: MLXArray
    public let lrConvs: [(MLXArray, MLXArray)]        // 4×k3 conv (512,3,512)
    public let lrNorms: [(MLXArray, MLXArray)]        // 4×GroupNorm(1) w/b
    public let lrFinal: (MLXArray, MLXArray)          // k1 (512,1,512)

    // ---------- DiT ----------
    public let tEmbW: [MLXArray]                      // t_embedder linear1/2
    public let condProjW: MLXArray, condProjB: MLXArray
    public let condMergeW: MLXArray, condMergeB: MLXArray   // 864→512
    public let skipLinW: MLXArray, skipLinB: MLXArray       // 512+80→512
    public let conv1W: MLXArray, conv1B: MLXArray           // Linear 512→512
    public let conv2W: MLXArray, conv2B: MLXArray           // conv (80,1,512)
    public let resProjW: MLXArray, resProjB: MLXArray
    public let finalAdaW: MLXArray, finalAdaB: MLXArray
    public let finalLinW: MLXArray, finalLinB: MLXArray
    // Transformer
    public let tfBlocks: [TFBlock]
    public let tfNormW: MLXArray, tfNormProjW: MLXArray, tfNormProjB: MLXArray
    // WaveNet
    public let wnCondW: MLXArray, wnCondB: MLXArray
    public let wnIn: [ConvRefl]
    public let wnResSkip: [(MLXArray, MLXArray)]

    public struct TFBlock {
        let attnWqkv: MLXArray
        let attnWo: MLXArray
        let anNormW: MLXArray, anProjW: MLXArray, anProjB: MLXArray
        let ffnW1: MLXArray, ffnW2: MLXArray, ffnW3: MLXArray
        let fnNormW: MLXArray, fnProjW: MLXArray, fnProjB: MLXArray
        let skipW: MLXArray?, skipB: MLXArray?
    }
    public struct ConvRefl { let w: MLXArray, b: MLXArray }

    // DiT 结构参数（推理固定）
    public let ditDim = TTSConfig.ditDim

    public init(path: String) throws {
        let f = try SafetensorsFile(path: path)
        func load(_ k: String) throws -> MLXArray { try f.tensor(k) }
        func opt(_ k: String) throws -> MLXArray? {
            f.tensorNames.contains(k) ? try load(k) : nil
        }
        let pre = "cfm.estimator."
        // LR
        lrContentProjW = try load("length_regulator.content_in_proj.weight")
        lrContentProjB = try load("length_regulator.content_in_proj.bias")
        var cv: [(MLXArray, MLXArray)] = []; var nr: [(MLXArray, MLXArray)] = []
        for i in [0, 3, 6, 9] {
            cv.append((try load("length_regulator.model.\(i).weight"),
                       try load("length_regulator.model.\(i).bias")))
            nr.append((try load("length_regulator.model.\(i+1).weight"),
                       try load("length_regulator.model.\(i+1).bias")))
        }
        lrConvs = cv; lrNorms = nr
        lrFinal = (try load("length_regulator.model.12.weight"),
                   try load("length_regulator.model.12.bias"))

        // t_embedder / t_embedder2（linear1/linear2）
        tEmbW = try [
            load(pre + "t_embedder.linear1.weight"), load(pre + "t_embedder.linear1.bias"),
            load(pre + "t_embedder.linear2.weight"), load(pre + "t_embedder.linear2.bias"),
        ]
        // cond merge
        condProjW = try load(pre + "cond_projection.weight")
        condProjB = try load(pre + "cond_projection.bias")
        condMergeW = try load(pre + "cond_x_merge_linear.weight")
        condMergeB = try load(pre + "cond_x_merge_linear.bias")
        skipLinW = try load(pre + "skip_linear.weight")
        skipLinB = try load(pre + "skip_linear.bias")
        conv1W = try load(pre + "conv1.weight"); conv1B = try load(pre + "conv1.bias")
        conv2W = try load(pre + "conv2.weight"); conv2B = try load(pre + "conv2.bias")
        resProjW = try load(pre + "res_projection.weight"); resProjB = try load(pre + "res_projection.bias")
        finalAdaW = try load(pre + "final_layer.adaLN_modulation.1.weight")
        finalAdaB = try load(pre + "final_layer.adaLN_modulation.1.bias")
        finalLinW = try load(pre + "final_layer.linear.weight")
        finalLinB = try load(pre + "final_layer.linear.bias")

        // Transformer（13 层）
        var blks: [TFBlock] = []
        for i in 0..<TTSConfig.ditLayers {
            let p = pre + "transformer.layers.\(i)."
            let hasSkip = f.tensorNames.contains(p + "skip_in_linear.weight")
            blks.append(TFBlock(
                attnWqkv: try load(p + "attention.wqkv.weight"),
                attnWo: try load(p + "attention.wo.weight"),
                anNormW: try load(p + "attention_norm.norm.weight"),
                anProjW: try load(p + "attention_norm.project_layer.weight"),
                anProjB: try load(p + "attention_norm.project_layer.bias"),
                ffnW1: try load(p + "feed_forward.w1.weight"),
                ffnW2: try load(p + "feed_forward.w2.weight"),
                ffnW3: try load(p + "feed_forward.w3.weight"),
                fnNormW: try load(p + "ffn_norm.norm.weight"),
                fnProjW: try load(p + "ffn_norm.project_layer.weight"),
                fnProjB: try load(p + "ffn_norm.project_layer.bias"),
                skipW: hasSkip ? try opt(p + "skip_in_linear.weight") : nil,
                skipB: hasSkip ? try opt(p + "skip_in_linear.bias") : nil))
        }
        tfBlocks = blks
        tfNormW = try load(pre + "transformer.norm.norm.weight")
        tfNormProjW = try load(pre + "transformer.norm.project_layer.weight")
        tfNormProjB = try load(pre + "transformer.norm.project_layer.bias")

        // WaveNet
        wnCondW = try load(pre + "wavenet.cond_layer.conv.weight")
        wnCondB = try load(pre + "wavenet.cond_layer.conv.bias")
        var wi: [ConvRefl] = []
        var wr: [(MLXArray, MLXArray)] = []
        for i in 0..<TTSConfig.wnLayers {
            wi.append(ConvRefl(w: try load(pre + "wavenet.in_layers.\(i).conv.weight"),
                               b: try load(pre + "wavenet.in_layers.\(i).conv.bias")))
            wr.append((try load(pre + "wavenet.res_skip_layers.\(i).conv.weight"),
                       try load(pre + "wavenet.res_skip_layers.\(i).conv.bias")))
        }
        wnIn = wi; wnResSkip = wr
        f.close()
    }
}