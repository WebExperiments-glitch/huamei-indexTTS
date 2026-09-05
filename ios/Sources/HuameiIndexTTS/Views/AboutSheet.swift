import SwiftUI

/// 关于页（版本号 + 功能 + 使用说明 + 许可声明）
struct AboutSheet: View {

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Huamei IndexTTS")
                                .font(Theme.Fonts.titleHero)
                                .foregroundStyle(Theme.Colors.labelPrimary)
                            Text("内部测试版（Internal Beta 1）")
                                .font(Theme.Fonts.callout)
                                .foregroundStyle(Theme.Colors.accentStrong)
                            Text("纯本地方言克隆 · 支持 iOS 17+ / iPadOS")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Colors.labelSecondary)
                        }
                        Spacer()
                        Image(systemName: "waveform.circle.fill")
                            .font(.system(size: 56))
                            .foregroundStyle(Theme.Colors.accent)
                    }

                    Group {
                        Text("这是什么").font(Theme.Fonts.heading)
                        Text("Huamei IndexTTS 是一款完全在设备端运行的语音克隆与合成应用：录音、推理、合成全部在本地完成，不依赖云端服务器，数据不会离开你的设备。")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Colors.labelSecondary)
                    }

                    Group {
                        Text("功能亮点").font(Theme.Fonts.heading)
                        Bullet("零云端：所有推理都在设备本地完成")
                        Bullet("多语言合成：中文 · 英文 · 日语 · 西语 · 阿语")
                        Bullet("音色克隆：录制参考音频即可克隆人声")
                        Bullet("8 种情感预设，合成更自然")
                        Bullet("GPU 加速（Apple MLX + Metal），实时合成")
                        Bullet("Liquid Glass 毛玻璃界面（iOS 26+）")
                    }
                    .foregroundStyle(Theme.Colors.labelSecondary)

                    Group {
                        Text("三步上手").font(Theme.Fonts.heading)
                        Bullet("1 · 导入模型：点「导入模型 ZIP」，选择模型压缩包自动安装")
                        Bullet("2 · 添加参考声音：录音 / 选音频 / 选视频 / 从文件选")
                        Bullet("3 · 输入文字，点「开始合成」，等待结果即可播放、分享")
                    }
                    .foregroundStyle(Theme.Colors.labelSecondary)

                    Group {
                        Text("工作原理").font(Theme.Fonts.heading)
                        Text("内置 IndexTTS-2.5 声学管线：文本经 GPT 自回归生成语义 Token → Codec 解码 → S2Mel 扩散声谱图 → BigVGAN 声码器合成波形；克隆组件使用 W2V-BERT-2.0 与 CAM++ 提取声纹特征。全部权重以 8-bit 量化压缩，可在 iPad / iPhone 上实时运行。")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Colors.labelSecondary)
                    }

                    Group {
                        Text("常见问题").font(Theme.Fonts.heading)
                        Text("Q：模型从哪来？\nA：模型文件打包在 huamei-models.zip，可从魔搭社区（jfjijiogkijijg/huamei-TTS）下载，或由开发者直接提供。放入「文件」App 后一键导入即可。\n\nQ：为什么首次使用要导入模型？\nA：模型权重约 4GB，不随 App 安装包分发，按需导入可显著减小安装包体积。\n\nQ：合成的音频会上传吗？\nA：不会。所有处理均在设备本地完成。")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(.secondary)
                    }

                    Group {
                        Text("致谢与许可").font(Theme.Fonts.heading)
                        Text("推理引擎 · IndexTeam IndexTTS-2.5 · Bilibili 模型使用许可")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("模型权重转换 · vanch007/mlx-indextts2-2.5-8bit · MIT")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("推理框架 · Apple MLX Swift · Apache-2.0")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("模型托管 · ModelScope 魔搭社区（huamei-TTS）")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("开源仓库 · github.com/WebExperiments-glitch/huamei-indexTTS")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                    }

                    Group {
                        Text("版本信息").font(Theme.Fonts.heading)
                        Text("当前版本 · 0.1.0（Internal Beta 1）")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("构建 · Release · Xcode 26 · iOS 26.5 SDK")
                            .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                        Text("发布状态 · 内部测试版本，仅限开发与体验使用，请勿对外公开传播。")
                            .font(Theme.Fonts.caption).foregroundStyle(Theme.Colors.warning)
                    }

                    Text("© 2026 · 为开源社区而作")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 12)
                }
                .padding(20)
            }
            .background(Theme.Colors.canvas.ignoresSafeArea())
            .navigationTitle("关于")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}

private struct Bullet: View {
    let text: String
    init(_ t: String) { text = t }
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(Theme.Colors.accent)
                .font(.system(size: 14, weight: .semibold))
                .padding(.top, 2)
            Text(text)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Colors.labelSecondary)
        }
    }
}