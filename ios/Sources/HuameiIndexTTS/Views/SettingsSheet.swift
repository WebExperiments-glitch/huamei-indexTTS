import SwiftUI

/// 高级设置 Sheet（语言 / 时长因子 / 采样超参 / 开发者模式（Beta））
struct SettingsSheet: View {

    @EnvironmentObject private var s: SessionStore
    @EnvironmentObject private var engine: InferenceEngine
    @Environment(\.dismiss)        private var dismiss
    @ObservedObject private var monitor = SystemMonitor.shared

    var body: some View {
        NavigationStack {
            Form {
                Section("语言") {
                    Picker("语言", selection: languageBinding) {
                        ForEach(SessionStore.Language.allCases) { lang in
                            Text(lang.label).tag(lang)
                        }
                    }
                    .pickerStyle(.menu)
                }

                Section {
                    HStack {
                        Text("语速")
                        Spacer()
                        Text(String(format: "%.2f", s.durationFactor))
                            .monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $s.durationFactor, in: 0.5...2.0, step: 0.05) {
                        Text("语速")
                    } minimumValueLabel: { Text("0.5") }
                      maximumValueLabel: { Text("2.0") }
                    Text("快 ← 正常 → 慢")
                        .font(Theme.Fonts.caption).foregroundStyle(.secondary)
                } header: { Text("语速调节") }
                  footer: { Text("数值越大语速越慢，越小越快。") }

                Section("实验功能（Beta）") {
                    Toggle("显示实验功能（Beta）", isOn: $s.showExperimental)
                    if s.showExperimental {
                        advancedBlock
                    }
                }

                Section("开发者模式（Beta）") {
                    Toggle("开发者模式（Beta）", isOn: devModeBinding)
                    Text("开启后显示实时系统资源（CPU / 内存 / GPU 显存）与 App 日志，并放宽下载校验（跳过 sha256、提高重试次数），仅用于测试排查。")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(.secondary)
                    if s.developerMode {
                        devModeBlock
                    }
                }
            }
            .navigationTitle("设置")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }

    // MARK: - 开发者模式（Beta）

    private var devModeBinding: Binding<Bool> {
        Binding(get: { s.developerMode }, set: { v in
            s.developerMode = v
            ModelDownloadManager.relaxedChecks = v
            if v { monitor.start() } else { monitor.stop() }
        })
    }

    @ViewBuilder
    private var devModeBlock: some View {
        Section("系统资源（Beta）") {
            LabeledContent("CPU 占用") {
                Text(String(format: "%.1f %%", monitor.cpu))
                    .monospacedDigit()
            }
            LabeledContent("内存（当前）") {
                Text(String(format: "%.0f MB", monitor.memoryMB))
                    .monospacedDigit()
            }
            LabeledContent("内存涨幅") {
                Text(String(format: "%+.0f MB", monitor.memoryDeltaMB))
                    .monospacedDigit()
                    .foregroundStyle(monitor.memoryDeltaMB > 0 ? Theme.Colors.danger : .secondary)
            }
            LabeledContent("GPU 显存（近似）") {
                Text(String(format: "%.0f MB", monitor.gpuMB))
                    .monospacedDigit()
            }
            Text("iOS 不提供 GPU 利用率公开 API，此处以显存占用近似。")
                .font(Theme.Fonts.caption).foregroundStyle(.secondary)
        }

        Section("放宽限制（Beta）") {
            Toggle("跳过下载 sha256 校验", isOn: relaxedBinding)
            Text("放宽时每文件自动重试 3 次（默认 1 次），便于弱网/测试环境。")
                .font(Theme.Fonts.caption).foregroundStyle(.secondary)
        }

        Section("App 信息") {
            LabeledContent("版本", value: appVersion)
            LabeledContent("Bundle ID", value: "com.huamei.indextts.app")
            LabeledContent("模型状态", value: engine.state == .ready ? "已就绪" : "未就绪")
            LabeledContent("模型目录", value: "Documents/huamei-models")
            LabeledContent("模型根目录", value: InferenceEngine.modelDir.path)
                .lineLimit(1)
                .truncationMode(.middle)
            Button("清空模型数据（重新导入）", role: .destructive) { clearModels() }
        }
    }

    /// 删除模型目录（模型文件损坏/残留时可重置重导）
    private func clearModels() {
        let dir = InferenceEngine.modelDir
        try? FileManager.default.removeItem(at: dir)
        SystemMonitor.shared.appendLog("已清空模型目录，请重新导入模型")
        Task { @MainActor in await engine.refreshAfterImport() }
    }

    private var relaxedBinding: Binding<Bool> {
        Binding(get: { ModelDownloadManager.relaxedChecks },
                set: { v in ModelDownloadManager.relaxedChecks = v })
    }

    private var appVersion: String {
        let short = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(short)（\(build)）Internal Beta"
    }

    @ViewBuilder
    private var advancedBlock: some View {
        Section("GPT-2 采样") {
            Toggle("启用采样", isOn: $s.doSample)
            if s.doSample {
                sliderRow("温度", value: $s.temperature, range: 0.1...2.0, step: 0.05)
                sliderRow("Top-P", value: $s.topP, range: 0.0...1.0, step: 0.05)
                Stepper("Top-K   \(s.topK)", value: $s.topK, in: 1...100)
                Stepper("束宽   \(s.numBeams)", value: $s.numBeams, in: 1...10)
            }
        }
        Section("惩罚与长度") {
            Stepper("重复惩罚  \(String(format: "%.1f", s.repetitionPenalty))",
                    value: $s.repetitionPenalty, in: 1.0...20.0, step: 0.5)
            Stepper("长度惩罚  \(String(format: "%.1f", s.lengthPenalty))",
                    value: $s.lengthPenalty, in: -5.0...5.0, step: 0.5)
            Stepper("最大谱图长度  \(s.maxMelTokens)",
                    value: $s.maxMelTokens, in: 50...1815, step: 10)
        }
    }

    private func sliderRow(_ name: String, value: Binding<Double>,
                           range: ClosedRange<Double>, step: Double) -> some View {
        VStack(alignment: .leading) {
            HStack {
                Text(name)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
        }
    }

    private var languageBinding: Binding<SessionStore.Language> {
        Binding(get: { s.language }, set: { s.language = $0 })
    }
}