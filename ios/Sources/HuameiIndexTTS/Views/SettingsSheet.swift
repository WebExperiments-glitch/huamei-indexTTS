import SwiftUI

/// 高级设置 Sheet（语言 / 时长因子 / 采样超参）—— 对齐 demo
struct SettingsSheet: View {

    @EnvironmentObject private var s: SessionStore
    @Environment(\.dismiss)        private var dismiss

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

                Section("实验功能") {
                    Toggle("显示实验功能", isOn: $s.showExperimental)
                    if s.showExperimental {
                        advancedBlock
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

    @ViewBuilder
    private var advancedBlock: some View {
        Section("GPT-2 Sampling") {
            Toggle("Enable sampling", isOn: $s.doSample)
            if s.doSample {
                sliderRow("temperature", value: $s.temperature, range: 0.1...2.0, step: 0.05)
                sliderRow("top_p", value: $s.topP, range: 0.0...1.0, step: 0.05)
                Stepper("top_k   \(s.topK)", value: $s.topK, in: 1...100)
                Stepper("num_beams   \(s.numBeams)", value: $s.numBeams, in: 1...10)
            }
        }
        Section("惩罚与长度") {
            Stepper("repetition_penalty  \(String(format: "%.1f", s.repetitionPenalty))",
                    value: $s.repetitionPenalty, in: 1.0...20.0, step: 0.5)
            Stepper("length_penalty  \(String(format: "%.1f", s.lengthPenalty))",
                    value: $s.lengthPenalty, in: -5.0...5.0, step: 0.5)
            Stepper("max_mel_tokens  \(s.maxMelTokens)",
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