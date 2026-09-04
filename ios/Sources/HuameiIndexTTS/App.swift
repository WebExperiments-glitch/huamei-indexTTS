import SwiftUI

/// App 入口（最小化；所有 UI 在 RootView）
/// Internal Beta 1
@main
struct HuameiIndexTTSApp: App {

    @State private var session = SessionStore()
    @State private var engine  = InferenceEngine()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
                .environment(engine)
                .preferredColorScheme(.light)             // 主色白橙
                .tint(Theme.Color.accent)
                .task { await engine.prepareIfNeeded() }
        }
    }
}