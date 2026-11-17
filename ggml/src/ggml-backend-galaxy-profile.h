#pragma once

// Lightweight opt-in CSV profiling for Galaxy TP, Galaxy and Galaxy2.
// Set LLAMA_GALAXY_PROFILE_DIR to an existing output directory on the PC.
// Profiles are synchronized observations. RPC dispatch_ms includes BOTH
// the network roundtrip and server execution; only phone-side native CSV
// separates them. CUDA observed_ms is a launch-to-post-RPC-fence window,
// NOT an isolated GPU kernel time.
// This file is header-only to keep instrumentation isolated from inference.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace ggml_galaxy_profile {

using clock = std::chrono::steady_clock;
using instant = clock::time_point;

inline bool enabled() {
    const char * dir = std::getenv("LLAMA_GALAXY_PROFILE_DIR");
    return dir != nullptr && dir[0] != '\0';
}

inline double ms(instant a, instant b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline uint64_t new_graph() {
    static std::atomic<uint64_t> next {0};
    return ++next;
}

inline uint64_t & current_graph() {
    static thread_local uint64_t graph = 0;
    return graph;
}

inline std::string quote(const char * value) {
    std::string result = "\"";
    const char * s = value != nullptr ? value : "";
    while (*s) {
        if (*s == '\"') result.push_back('\"');
        result.push_back(*s++);
    }
    result += '\"';
    return result;
}

inline void append(const char * filename, const char * header, const std::string & record) {
    const char * dir = std::getenv("LLAMA_GALAXY_PROFILE_DIR");
    if (dir == nullptr || dir[0] == '\0') return;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    try {
        const std::filesystem::path path = std::filesystem::path(dir) / filename;
        std::filesystem::create_directories(path.parent_path());
        std::error_code ec;
        const bool create_header = !std::filesystem::exists(path, ec) ||
                std::filesystem::file_size(path, ec) == 0;
        std::ofstream out(path, std::ios::app);
        if (!out) return;
        if (create_header) out << header << '\n';
        out << record << '\n';
    } catch (...) {
        // Profiling must never alter the outcome of model inference.
    }
}

inline void communication(
        const char * kind, const char * tensor, int64_t tokens,
        uint64_t read_bytes, uint64_t write_bytes,
        double read_ms, double reduce_ms, double write_ms,
        double sync_ms, double wall_ms,
        uint64_t read_calls = 0, uint64_t write_calls = 0,
        uint64_t sync_calls = 0, size_t tile_tokens = 0, bool rs_fused_read = false) {
    if (!enabled()) return;
    std::ostringstream row;
    row << std::fixed << std::setprecision(5)
        << current_graph() << ',' << quote(kind) << ',' << quote(tensor) << ','
        << tokens << ',' << read_bytes << ',' << write_bytes << ','
        << read_ms << ',' << reduce_ms << ',' << write_ms << ','
        << sync_ms << ',' << wall_ms << ','
        << read_calls << ',' << write_calls << ','
        << sync_calls << ',' << tile_tokens << ',' << int(rs_fused_read);
    append("communication_timeline.csv",
            "graph,kind,tensor,tokens,read_bytes,write_bytes,read_ms,host_reduce_ms,write_ms,sync_wait_ms,wall_ms,read_calls,write_calls,sync_calls,tile_tokens,rs_fused_read",
            row.str());
}

inline void compute(
        uint64_t graph, size_t subgraph, size_t rank, const char * backend,
        const char * terminal, int nodes,
        double dispatch_ms, double wait_ms, double elapsed_ms) {
    if (!enabled()) return;
    std::ostringstream row;
    row << std::fixed << std::setprecision(5)
        << graph << ',' << subgraph << ',' << rank << ','
        << quote(backend) << ',' << quote(terminal) << ',' << nodes << ','
        << dispatch_ms << ',' << wait_ms << ',' << elapsed_ms << ",1";
    append("compute_timeline.csv",
            "graph,subgraph,rank,backend,terminal,nodes,dispatch_ms,sync_wait_ms,observed_ms,profile_forced_sync",
            row.str());
}


// Diagnostics for the executed TP AllReduce path. Copy-call time can include
// RPC/network + backend staging; add-dispatch time can include Phone RPC
// execution. Neither is a pure wire-time or pure CPU compute measurement.
// Completion fences are intentionally enabled only for -Profile runs.
struct allreduce_breakdown {
    double native_call_ms = 0.0;
    double copy_call_ms = 0.0;
    double add_dispatch_ms = 0.0;
    double completion_wait_ms = 0.0;
    uint64_t copy_bytes = 0;
    size_t copy_calls = 0;
    size_t add_dispatches = 0;
    bool used_fallback = false;
};

inline int64_t steady_ns(instant t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

inline void allreduce_detail(
        uint64_t graph, size_t subgraph, const char * tensor,
        int64_t begin_ns, int64_t end_ns, double wall_ms,
        const allreduce_breakdown & d) {
    if (!enabled()) return;
    double accounted_ms = d.native_call_ms + d.copy_call_ms +
                          d.add_dispatch_ms + d.completion_wait_ms;
    double remainder_ms = wall_ms - accounted_ms;
    // Includes backend enqueue/fence gaps and local bookkeeping.
    // Never attribute this remainder to Wi-Fi or CPU computation.
    if (remainder_ms < 0.0) remainder_ms = 0.0;
    std::ostringstream row;
    row << std::fixed << std::setprecision(5)
        << graph << ',' << subgraph << ',' << quote(tensor) << ','
        << int(d.used_fallback) << ',' << d.copy_calls << ',' << d.copy_bytes << ','
        << d.add_dispatches << ',' << d.native_call_ms << ','
        << d.copy_call_ms << ',' << d.add_dispatch_ms << ','
        << d.completion_wait_ms << ',' << remainder_ms << ','
        << wall_ms << ',' << begin_ns << ',' << end_ns;
    append("allreduce_breakdown.csv",
           "graph,subgraph,tensor,used_fallback,copy_calls,copy_bytes,add_dispatches,"
           "native_call_ms,copy_call_ms,add_dispatch_ms,completion_wait_ms,"
           "other_ms,wall_ms,start_steady_ns,end_steady_ns",
           row.str());
}

inline void summary(
        uint64_t graph, const char * mode, int nodes, size_t subgraphs,
        size_t reduce_scatter, size_t residual_scatter, size_t all_gather,
        size_t all_reduce, double wall_ms) {
    if (!enabled()) return;
    std::ostringstream row;
    row << std::fixed << std::setprecision(5)
        << graph << ',' << quote(mode) << ',' << nodes << ',' << subgraphs << ','
        << reduce_scatter << ',' << residual_scatter << ',' << all_gather << ','
        << all_reduce << ',' << wall_ms << ",1";
    append("summary.csv",
            "graph,mode,graph_nodes,subgraphs,reduce_scatter,residual_scatter,all_gather,all_reduce,graph_wall_ms,profile_forced_sync",
            row.str());
}

} // namespace ggml_galaxy_profile
