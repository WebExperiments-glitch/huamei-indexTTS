// swift-tools-version: 6.0
//
// Huamei IndexTTS — iOS 原生端侧语音克隆 App
// ================================================
// Internal Beta 1 · 主色：白 + 橙色 accent
// 仓库：https://github.com/WebExperiments-glitch/huamei-indexTTS
//
// 平台说明：
//   · 最低 iOS 17（推理内核 MLXIndexTTS2Core 依赖 mlx-swift：iOS 17+ / macOS 14+）
//   · 液态玻璃（iOS 26+）自动增强；旧版用 material 模拟
//
import PackageDescription

let package = Package(
    name: "HuameiIndexTTS",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "MLXIndexTTS2Core", targets: ["MLXIndexTTS2Core"]),
        .executable(name: "HuameiIndexTTS", targets: ["HuameiIndexTTS"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0")
    ],
    targets: [
        // ---------- 推理内核（纯 MLX，无 SwiftUI；与 UI 解耦） ----------
        .target(
            name: "MLXIndexTTS2Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Sources/MLXIndexTTS2Core",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // ---------- SwiftUI App ----------
        .executableTarget(
            name: "HuameiIndexTTS",
            dependencies: [
                .target(name: "MLXIndexTTS2Core"),
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Sources/HuameiIndexTTS",
            resources: [.process("Resources")],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("UniformTypeIdentifiers")
            ]
        )
    ]
)