import Foundation
import Combine
import SwiftUI
import Metal

/// 系统资源监控（开发者模式用）：
///   - CPU 占用率（Mach host_processor_info）
///   - 内存占用 + 涨幅（task_vm_info.phys_footprint）
///   - GPU 显存占用（MTLDevice.currentAllocatedSize，iOS 无 GPU 利用率公开 API）
///   - App 内日志缓冲（环形，保留最近 200 条）
final class SystemMonitor: ObservableObject {

    /// 全局共享（App 运行期持续收集日志，设置页展示）
    static let shared = SystemMonitor()

    @Published private(set) var cpu: Double = 0          // %
    @Published private(set) var memoryMB: Double = 0     // 当前物理占用
    @Published private(set) var memoryDeltaMB: Double = 0 // 相对基准的涨幅
    @Published private(set) var gpuMB: Double = 0        // 显存占用（近似）
    @Published private(set) var logs: [String] = []

    // 供 SettingsSheet 订阅一个实例；baseline 在 shared 创建时即建立
    init() {
        baselineMB = currentMemoryMB()
    }

    private var baselineMB: Double = 0
    private var timer: Timer?
    private let device = MTLCreateSystemDefaultDevice()
    private let maxLogs = 200

    func start() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func appendLog(_ line: String) {
        let ts = Self.formatter.string(from: Date())
        logs.append("[\(ts)] \(line)")
        if logs.count > maxLogs { logs.removeFirst(logs.count - maxLogs) }
    }

    func clearLogs() {
        logs.removeAll()
    }

    private func tick() {
        cpu = Self.cpuUsage()
        let m = currentMemoryMB()
        memoryMB = m
        memoryDeltaMB = m - baselineMB
        if let device { gpuMB = Double(device.currentAllocatedSize) / (1024 * 1024) }
    }

    private func currentMemoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    /// 全系统 CPU 占用（单位 %）
    private static func cpuUsage() -> Double {
        var cpuCount: natural_t = 0
        var cpuInfo: processor_info_array_t?
        var cpuInfoCount: mach_msg_type_number_t = 0
        let kr = host_processor_info(mach_host_self(),
                                     PROCESSOR_CPU_LOAD_INFO,
                                     &cpuCount,
                                     &cpuInfo,
                                     &cpuInfoCount)
        guard kr == KERN_SUCCESS, let info = cpuInfo else { return 0 }
        defer {
            vm_deallocate(mach_task_self_,
                          vm_address_t(UInt(bitPattern: info)),
                          vm_size_t(cpuInfoCount) * vm_size_t(MemoryLayout<integer_t>.size))
        }
        let ptr = info
        var total = 0.0
        for i in 0..<Int(cpuCount) {
            let offset = Int(CPU_STATE_MAX) * i
            let user = Double(ptr[offset + Int(CPU_STATE_USER)])
            let system = Double(ptr[offset + Int(CPU_STATE_SYSTEM)])
            let idle = Double(ptr[offset + Int(CPU_STATE_IDLE)])
            let nice = Double(ptr[offset + Int(CPU_STATE_NICE)])
            let all = user + system + idle + nice
            guard all > 0 else { continue }
            total += (user + system + nice) / all
        }
        return total / Double(max(1, Int(cpuCount))) * 100
    }

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}