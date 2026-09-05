import SwiftUI

/// 极简模型状态卡：缺模型时展示"一键下载"，下载中显示进度条，其余情况隐藏。
/// 目标：用户什么都不用懂，按一个按钮等它自己好。
struct ModelStatusView: View {

    @EnvironmentObject private var engine: InferenceEngine
    @ObservedObject private var monitor = SystemMonitor.shared

    /// 由 RootView 注入：打开模型目录选择器
    var onImportFolder: () -> Void = {}
    /// 由 RootView 注入：打开 ZIP 选择器（推荐，一键导入）
    var onImportZip: () -> Void = {}
    /// 导入进行中（RootView 维护）
    var isImporting: Bool = false

    var body: some View {
        VStack(spacing: 12) {
            statusContent
            logPanel
        }
    }

    @ViewBuilder
    private var statusContent: some View {
        switch engine.state {
        case .missingModel:
            missingModelView
        case .loading:
            ProgressView("模型加载中…")
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 8)
        case .ready:
            readyView
        case .failed(let msg):
            failedView(msg)
        default:
            EmptyView()
        }
    }

    /// 运行日志面板（始终显示在主界面，便于排查）
    @ViewBuilder
    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("运行日志")
                    .font(Theme.Fonts.heading)
                    .foregroundStyle(Theme.Colors.labelPrimary)
                Spacer()
                Button("复制") { copyLogs() }
                    .font(Theme.Fonts.caption)
                Button("清空") { monitor.clearLogs() }
                    .font(Theme.Fonts.caption)
            }
            if monitor.logs.isEmpty {
                Text("暂无日志。导入模型、下载或合成时这里会实时显示进度与错误原因。")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Colors.labelTertiary)
            } else {
                ForEach(Array(monitor.logs.suffix(12).reversed()), id: \.self) { line in
                    Text(line)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.Colors.labelSecondary)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.Colors.surfaceAlt, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Theme.Colors.divider, lineWidth: 0.5)
        )
    }

    /// 复制全部日志到剪贴板
    private func copyLogs() {
        let text = monitor.logs.joined(separator: "\n")
        guard !text.isEmpty else { return }
        UIPasteboard.general.string = text
    }

    /// 模型已就绪：展示状态 + 管理入口（重新导入/换模型）
    @ViewBuilder
    private var readyView: some View {
        VStack(spacing: 10) {
            Label("模型已就绪", systemImage: "checkmark.seal.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.Colors.success)
            Text("可以开始添加参考声音并合成。需要更换模型时，点下方按钮重新导入。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            HStack(spacing: 12) {
                Button("重新导入模型") { onImportZip() }
                    .font(.subheadline)
                    .buttonStyle(.bordered)
                Button("导入文件夹") { onImportFolder() }
                    .font(.subheadline)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Theme.Colors.surfaceAlt, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Theme.Colors.success.opacity(0.25), lineWidth: 1)
        )
    }

    /// 加载失败：显示原因 + 重导入口
    @ViewBuilder
    private func failedView(_ msg: String) -> some View {
        VStack(spacing: 10) {
            Label("模型加载失败", systemImage: "exclamationmark.triangle.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.Colors.danger)
            Text(msg)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            HStack(spacing: 12) {
                Button("重新导入模型") { onImportZip() }
                    .font(.subheadline)
                    .buttonStyle(.bordered)
                Button("联网下载") { engine.startModelDownload() }
                    .font(.subheadline)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Theme.Colors.surfaceAlt, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Theme.Colors.danger.opacity(0.25), lineWidth: 1)
        )
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