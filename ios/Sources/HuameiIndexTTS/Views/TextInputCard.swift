import SwiftUI

/// 文本输入卡（带字符计数 + 多行自适应）
struct TextInputCard: View {

    @EnvironmentObject private var s: SessionStore
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Text")
                    .font(Theme.Fonts.heading)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                Spacer()
                Text("\(s.text.count)")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Colors.labelTertiary)
                    .monospacedDigit()
            }

            ZStack(alignment: .topLeading) {
                if s.text.isEmpty {
                    Text("Enter the text to synthesize...")
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Colors.labelTertiary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 14)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $s.text)
                    .focused($focused)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .frame(minHeight: 120, maxHeight: 220)
            }
            .glassCard(cornerRadius: 18)

            Text("提示：当前模型版本 2.5")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Colors.labelTertiary)
                .padding(.horizontal, 4)
        }
    }
}