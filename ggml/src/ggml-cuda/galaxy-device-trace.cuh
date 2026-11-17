#pragma once

// PC CUDA device time-series instrumentation (enabled only by
// LLAMA_GALAXY_PROFILE_DIR). CUDA timestamp events are anchored to the
// process monotonic clock once per device. The recorded device interval
// tracks stream work, including graph execution and kernel launches; a
// graph-level interval is NOT a sum of each individual kernel's active time.
//
// Deferred collection at existing ggml backend synchronize boundaries
// avoids adding an extra GPU fence to each CUDA graph dispatch.

#include "../ggml-backend-galaxy-device-profile.h"
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

namespace ggml_galaxy_cuda_trace {

struct ticket {
    cudaEvent_t start = nullptr;
    int device = -1;
    const char * category = nullptr;
    const char * source = nullptr;
    uint64_t bytes = 0;
};

struct pending {
    ticket begin;
    cudaEvent_t end = nullptr;
};

struct device_state {
    cudaEvent_t anchor = nullptr;
    int64_t pc_anchor_ns = 0;
    std::vector<pending> intervals;
};

inline std::mutex & mutex() {
    static std::mutex m;
    return m;
}
inline std::map<int, device_state> & devices() {
    static std::map<int, device_state> data;
    return data;
}
inline ticket begin(int dev, cudaStream_t stream, const char * category,
                    const char * source, uint64_t bytes = 0) {
    if (!ggml_galaxy_device_profile::enabled()) return {};
    std::lock_guard<std::mutex> guard(mutex());
    device_state & state = devices()[dev];
    if (state.anchor == nullptr) {
        cudaEvent_t anchor = nullptr;
        if (cudaEventCreate(&anchor) != cudaSuccess) return {};
        if (cudaEventRecord(anchor, stream) != cudaSuccess ||
            cudaEventSynchronize(anchor) != cudaSuccess) {
            cudaEventDestroy(anchor);
            return {};
        }
        state.anchor = anchor;
        state.pc_anchor_ns = ggml_galaxy_device_profile::ns(
            ggml_galaxy_device_profile::clock::now());
    }
    cudaEvent_t start = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess) return {};
    if (cudaEventRecord(start, stream) != cudaSuccess) {
        cudaEventDestroy(start);
        return {};
    }
    return {start, dev, category, source, bytes};
}

inline void end(ticket t, cudaStream_t stream) {
    if (t.start == nullptr) return;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(t.start);
        return;
    }
    if (cudaEventRecord(stop, stream) != cudaSuccess) {
        cudaEventDestroy(t.start);
        cudaEventDestroy(stop);
        return;
    }
    std::lock_guard<std::mutex> guard(mutex());
    devices()[t.device].intervals.push_back({t, stop});
}

inline void flush(int dev) {
    if (!ggml_galaxy_device_profile::enabled()) return;
    struct row {
        const char * category;
        const char * source;
        int64_t start_ns, end_ns;
        int dev;
        uint64_t bytes;
        double elapsed_ms;
    };
    std::vector<row> ready;
    {
        std::lock_guard<std::mutex> guard(mutex());
        auto found = devices().find(dev);
        if (found == devices().end() || found->second.anchor == nullptr) return;
        device_state & state = found->second;
        std::vector<pending> remaining;
        for (auto & item : state.intervals) {
            if (cudaEventQuery(item.end) != cudaSuccess) {
                remaining.push_back(item);
                continue;
            }
            float start_ms = 0.f, end_ms = 0.f;
            const bool ok =
                cudaEventElapsedTime(&start_ms, state.anchor, item.begin.start) == cudaSuccess &&
                cudaEventElapsedTime(&end_ms, state.anchor, item.end) == cudaSuccess;
            if (ok && end_ms > start_ms) {
                const auto absolute_start = state.pc_anchor_ns +
                    static_cast<int64_t>(std::llround(static_cast<double>(start_ms) * 1.0e6));
                const auto absolute_end = state.pc_anchor_ns +
                    static_cast<int64_t>(std::llround(static_cast<double>(end_ms) * 1.0e6));
                ready.push_back({item.begin.category, item.begin.source,
                                 absolute_start, absolute_end, item.begin.device,
                                 item.begin.bytes, end_ms - start_ms});
            }
            cudaEventDestroy(item.begin.start);
            cudaEventDestroy(item.end);
        }
        state.intervals.swap(remaining);
    }
    for (const auto & item : ready) {
        ggml_galaxy_device_profile::append(
            "pc_cuda_events.csv", item.category,
            item.start_ns, item.end_ns, item.source, item.dev, item.bytes,
            item.elapsed_ms);
    }
}

} // namespace ggml_galaxy_cuda_trace
