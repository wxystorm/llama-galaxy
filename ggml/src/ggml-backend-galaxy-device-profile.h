#pragma once

// Opt-in PC-only Fig.3 device activity sidecars. Each backend writes a
// separate file to avoid DLL-local mutexes interleaving writes.
// All timestamps are PC std::chrono::steady_clock nanoseconds.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

namespace ggml_galaxy_device_profile {

using clock = std::chrono::steady_clock;
inline int64_t ns(clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
inline bool enabled() {
    const char * p = std::getenv("LLAMA_GALAXY_PROFILE_DIR");
    return p && *p;
}
inline void append(const char * filename, const char * category,
                   int64_t start_ns, int64_t end_ns,
                   const char * source, int device, uint64_t bytes,
                   double device_ms = 0.0) {
    const char * dir = std::getenv("LLAMA_GALAXY_PROFILE_DIR");
    if (!dir || !*dir) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    try {
        const auto path = std::filesystem::path(dir) / filename;
        std::filesystem::create_directories(path.parent_path());
        std::error_code ec;
        const bool header = !std::filesystem::exists(path, ec) ||
                            std::filesystem::file_size(path, ec) == 0;
        std::ofstream out(path, std::ios::app);
        if (!out) return;
        if (header) out << "category,start_steady_ns,end_steady_ns,source,device,bytes,device_ms\n";
        if (end_ns > start_ns) {
            out << category << ',' << start_ns << ',' << end_ns << ','
                << source << ',' << device << ',' << bytes << ','
                << std::fixed << std::setprecision(6) << device_ms << '\n';
        }
    } catch (...) {
        // Profiling is best-effort and may not break successful inference.
    }
}
inline void cpu_graph(clock::time_point start, clock::time_point end, int nodes) {
    append("pc_cpu_events.csv", "CPU Compute", ns(start), ns(end),
           "CPU_BACKEND_GRAPH", -1, static_cast<uint64_t>(nodes),
           std::chrono::duration<double, std::milli>(end - start).count());
}
inline void cpu_probe() {
    // Ensure an empty trace is distinguishable from an uninstrumented binary.
    append("pc_cpu_events.csv", "CPU Compute", 0, 0, "CPU_PROFILE_READY", -1, 0);
}

} // namespace ggml_galaxy_device_profile
