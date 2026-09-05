import SwiftUI

/// 极简模型状态卡：缺模型时展示"一键下载"，下载中显示进度条，其余情况隐藏。
/// 目标：用户什么都不用懂，按一个按钮等它自己好。
struct ModelStatusView: View {

    @EnvironmentObject private var engine: InferenceEngine

    /// 由 RootView 注入：打开文件夹选择器
    var onImportFolder: () -> Void = {}
    /// 导入进行中（RootView 维护）
    var isImporting: Bool = false

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
                    Label("还没有模型", systemImage: "externaldrive.badge.plus")
                        .font(.subheadline.weight(.semibold))
                    Text("把模型压缩包（ZIP）或文件夹放进「文件」App，\n选择导入后即可使用，一键搞定")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    if isImporting {
                        ProgressView("正在导入模型…")
                            .padding(.vertical, 6)
                    }
                    Button(action: onImportZip) {
                        Text("导入模型 ZIP（推荐）")
                            .font(.headline)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isImporting)
                    Button("导入模型文件夹") { onImportFolder() }
                        .font(.subheadline)
                        .disabled(isImporting)
                    Button("联网自动下载") { engine.startModelDownload() }
                        .font(.subheadline)
                        .disabled(isImporting)
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