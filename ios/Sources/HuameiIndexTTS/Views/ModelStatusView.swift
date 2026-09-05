import SwiftUI

/// 极简模型状态卡：缺模型时展示"一键下载"，下载中显示进度条，其余情况隐藏。
/// 目标：用户什么都不用懂，按一个按钮等它自己好。
struct ModelStatusView: View {

    @Environment(InferenceEngine.self) private var engine

    var body: some View {
        switch engine.state {
        case .missingModel:
            missingModelView
        case .loading:
            ProgressView("模型加载中…")
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 8)
        default:
            EmptyView()
        }
    }

    @ViewBuilder
    private var missingModelView: some View {
        VStack(spacing: 12) {
            switch engine.downloadState {
            case .downloading(let fraction, let file):
                VStack(spacing: 8) {
                    ProgressView(value: fraction)
                    HStack {
                        Text(String(format: "正在下载模型 %.0f%%", fraction * 100))
                            .font(.subheadline.weight(.medium))
                        Spacer()
                        Text((file as NSString).lastPathComponent)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            case .failed(let msg):
                Text("下载失败：\(msg)")
                    .font(.footnote)
                    .foregroundStyle(.red)
                Button("重试") { engine.startModelDownload() }
            case .idle, .done:
                VStack(spacing: 10) {
                    Label("首次使用需要下载模型", systemImage: "arrow.down.circle.fill")
                        .font(.subheadline.weight(.semibold))
                    Text("约 3.7GB，下载完成后即可正常使用（一次性）")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button {
                        engine.startModelDownload()
                    } label: {
                        Text("一键下载")
                            .font(.headline)
                            .padding(.horizontal, 28)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Theme.Colors.surfaceAlt, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Theme.Colors.accent.opacity(0.15), lineWidth: 1)
        )
    }
}