import SwiftUI
import UniformTypeIdentifiers

/// UIKit 文件/文件夹选择器包装
/// （SwiftUI 的 fileImporter 在部分 iPadOS 26/27 上不弹窗，用 UIKit 保底）
struct DocumentPicker: UIViewControllerRepresentable {

    /// true = 选整个文件夹；false = 选任意文件
    var allowFolders: Bool
    var onPick: ([URL]) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let types: [UTType] = allowFolders ? [.folder] : [.item]
        let vc = UIDocumentPickerViewController(forOpeningContentTypes: types)
        vc.allowsMultipleSelection = false
        vc.delegate = context.coordinator
        return vc
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController,
                                context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: DocumentPicker
        init(_ parent: DocumentPicker) { self.parent = parent }

        func documentPicker(_ controller: UIDocumentPickerViewController,
                            didPickDocumentsAt urls: [URL]) {
            parent.onPick(urls)
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {}
    }
}