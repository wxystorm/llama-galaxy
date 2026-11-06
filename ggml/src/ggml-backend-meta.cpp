#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"
#include "ggml-backend-galaxy-collectives.h"
#include "ggml-backend-galaxy-causal-live.h"
#include "ggml-backend-galaxy-causal-layers.h"
#include "ggml-backend-galaxy-profile.h"
#include "ggml-backend-galaxy-markers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <exception>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cinttypes>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

// Named DUP tensors are graph-only transitions. Their identity is encoded
// in op_params, not the name: llama callbacks can rename a tensor in place.
using galaxy_sp_transition = ggml_galaxy::sp_transition;

static galaxy_sp_transition galaxy_sp_get_transition(const ggml_tensor * tensor) {
    return ggml_galaxy::get_transition(tensor);
}

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;   //这个是一个映射表，key是原始张量，value是切分后的张量列表

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

// Optional model-buffer-owned SP-QKV full-weight replicas.
// Constructed when model buffers are allocated, populated by the same
// GGUF set_tensor callbacks, and destroyed with the model Meta buffer.
// No static/global pointers or cross-model caches.
struct ggml_backend_meta_resident_qkv {
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<ggml_tensor *> tensors;
    std::vector<std::pair<size_t,size_t>> loaded_ranges;
    size_t total_bytes = 0;

    void record_write(size_t offset, size_t size) {
        if (offset > total_bytes || size > total_bytes-offset) {
            throw std::invalid_argument("resident SP QKV GGUF write exceeds tensor");
        }
        if (size==0) return;
        loaded_ranges.emplace_back(offset,offset+size);
        std::sort(loaded_ranges.begin(),loaded_ranges.end());
        std::vector<std::pair<size_t,size_t>> merged;
        for (const auto & interval:loaded_ranges) {
            if (merged.empty() || merged.back().second<interval.first) {
                merged.push_back(interval);
            } else {
                merged.back().second=std::max(merged.back().second,interval.second);
            }
        }
        loaded_ranges=std::move(merged);
    }
    bool ready() const {
        return loaded_ranges.size()==1 &&
               loaded_ranges.front().first==0 &&
               loaded_ranges.front().second==total_bytes &&
               tensors.size()==2;
    }
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::map<const ggml_tensor *, ggml_backend_meta_resident_qkv> resident_qkv;
    std::mutex resident_qkv_mutex;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}
// This is intentionally a model-buffer lookup: the mirror lifetime is
// attached to the model, not to a temporary graph or a backend pointer.
static ggml_tensor * ggml_backend_meta_resident_qkv_tensor(
        const ggml_tensor * model_weight, size_t rank) {
    if (!model_weight || !model_weight->buffer ||
            !ggml_backend_buffer_is_meta(model_weight->buffer)) return nullptr;
    auto * ctx=static_cast<ggml_backend_meta_buffer_context *>(model_weight->buffer->context);
    const auto it=ctx->resident_qkv.find(model_weight);
    if (it==ctx->resident_qkv.end() || !it->second.ready() ||
            rank>=it->second.tensors.size()) return nullptr;
    return it->second.tensors[rank];
}

// 判断一个张量是否是静态的，即是否在静态简单张量容器中存在
static bool ggml_backend_meta_tensor_is_static(
        const ggml_tensor * tensor) {

    if (tensor == nullptr ||
        tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return false;
    }

    auto * buf_ctx =
        static_cast<ggml_backend_meta_buffer_context *>(
            tensor->buffer->context);

    return buf_ctx->stc_static.simple_tensors.find(tensor) !=
           buf_ctx->stc_static.simple_tensors.end();
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // This branch is cold: never log for successful tensor-layout
            // inference. A print of the *actual* node and source states is
            // essential because both incompatible split inputs and a
            // scalar_only restriction lead to the same assertion here.
            GGML_LOG_ERROR("[GALAXY_SP_LAYOUT_FAIL] node=%s op=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] scalar_only=%d n_backends=%zu\n",
                    tensor->name, ggml_op_name(tensor->op),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (int) scalar_only, n_bufs);
            for (size_t i = 0; i < GGML_MAX_SRC; ++i) {
                const ggml_tensor * src = tensor->src[i];
                if (src == nullptr || src == tensor) {
                    continue;
                }
                const ggml_backend_meta_split_state & state = src_ss[i];
                GGML_LOG_ERROR("[GALAXY_SP_LAYOUT_SRC] index=%zu name=%s op=%s axis=%s ne=[%" PRId64 ",%" PRId64 "] split=[%" PRId64 ",%" PRId64 "] segments=%u\n",
                        i, src->name, ggml_op_name(src->op),
                        ggml_backend_meta_split_axis_name(state.axis),
                        src->ne[0], src->ne[1], state.ne[0], state.ne[1], state.n_segments);
            }
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        GGML_ABORT("fatal error");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        // A transition is represented by a GGML_OP_DUP only in the host
        // dependency graph; Meta physically materializes the changed layout.
        // Do not run the ordinary DUP split inference, which would inherit
        // the original layout and make the SP residual shape incorrect.
        const galaxy_sp_transition transition = galaxy_sp_get_transition(tensor);
        if (transition != galaxy_sp_transition::none) {
            GGML_ASSERT(n_bufs == 2);
            GGML_ASSERT(tensor->src[0] != nullptr);
            GGML_ASSERT(tensor->ne[0] > 0 && tensor->ne[1] >= 2);
            GGML_ASSERT(tensor->ne[2] == 1 && tensor->ne[3] == 1);
            GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
            if (transition == galaxy_sp_transition::all_gather) {
                return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            }
            const auto seq = ggml_galaxy::configured_sequence_split(tensor->ne[1]);
            return {GGML_BACKEND_SPLIT_AXIS_1, {seq[0], seq[1]}, {1}, 1};
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get(); //第j个张量后端目录，初始里面没有张量
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static enum ggml_backend_local_file_result
ggml_backend_meta_try_set_local_file_2d(
        ggml_tensor * tensor,
        uint64_t src_offset,
        uint64_t dst_offset,
        uint64_t copy_size,
        uint64_t n_copies,
        uint64_t src_stride,
        uint64_t dst_stride) {

    if (tensor == nullptr || tensor->buffer == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(tensor->buffer);

    if (buft == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_dev_t dev =
        ggml_backend_buft_get_device(buft);

    if (dev == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_reg_t reg =
        ggml_backend_dev_backend_reg(dev);

    if (reg == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    void * proc =
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_SET_LOCAL_2D_PROC);

    if (proc == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    auto fn =
        reinterpret_cast<ggml_backend_rpc_set_local_2d_t>(
            proc);

    return fn(
        tensor,
        src_offset,
        dst_offset,
        copy_size,
        n_copies,
        src_stride,
        dst_stride);
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // When loading the model, copy the ORIGINAL full quantized GGUF bytes
    // to both persistent SP-QKV buffers. Never reconstruct from TP shards
    // at inference time. Track all write ranges, not only final chunk size.
    auto * meta_ctx=static_cast<ggml_backend_meta_buffer_context *>(buffer->context);
    auto native=meta_ctx->resident_qkv.find(tensor);
    if (native!=meta_ctx->resident_qkv.end()) {
        // Model loaders may issue subrange writes: serialize coverage updates
        // and copying to the same long-lived physical weight tensor.
        std::lock_guard<std::mutex> guard(meta_ctx->resident_qkv_mutex);
        auto & mirror=native->second;
        const bool was_ready=mirror.ready();
        if (offset>mirror.total_bytes || size>mirror.total_bytes-offset)
            throw std::invalid_argument("SP QKV load-time mirror offset/size out of range");
        if (size>0) {
            for (auto * full:mirror.tensors) {
                ggml_backend_tensor_set(full,data,offset,size);
            }
        }
        mirror.record_write(offset,size);
        if (!was_ready && mirror.ready()) {
            GGML_LOG_INFO("[GALAXY_SP_QKV_RESIDENT_LOAD] tensor=%s"
                    " bytes=%zu ranks=2 phase=MODEL_LOAD status=PASS\n",
                    tensor->name,mirror.total_bytes);
        }
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }
    const bool is_static_tensor =
                    ggml_backend_meta_tensor_is_static(tensor);

                /*
                * 再增加几项防御性限制：
                * 本地文件快速路径只处理原始、非 view 的静态张量。
                */
                const bool can_use_local_file =
                    is_static_tensor &&
                    tensor->op == GGML_OP_NONE &&
                    tensor->view_src == nullptr &&
                    tensor->name[0] != '\0';

    switch (split_state.axis) {
    case GGML_BACKEND_SPLIT_AXIS_0:
    case GGML_BACKEND_SPLIT_AXIS_1:
    case GGML_BACKEND_SPLIT_AXIS_2: {
        // 完整 Meta tensor 中，一个完整 chunk 的字节数
        const size_t chunk_size_full =
            tensor->nb[split_state.axis + 1];

        GGML_ASSERT(chunk_size_full != 0);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);

        const int64_t i_start =
            offset / chunk_size_full;

        const int64_t i_stop =
            (offset + size) / chunk_size_full;

        GGML_ASSERT(i_start >= 0);
        GGML_ASSERT(i_stop >= i_start);

        size_t offset_j = 0;

        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor =
                ggml_backend_meta_buffer_simple_tensor(
                    tensor,
                    j);

            GGML_ASSERT(simple_tensor != nullptr);

            const size_t chunk_size_j =
                simple_tensor->nb[split_state.axis + 1];

            if (chunk_size_j == 0) {
                continue;
            }

            /*
             * 本次写入在局部目标张量中的起始偏移。
             */
            const size_t simple_offset =
                static_cast<size_t>(i_start) *
                chunk_size_j;

            /*
             * 本地文件源偏移必须是相对于“完整源张量起点”的偏移。
             *
             * offset：
             *   当前写入区域在完整 Meta tensor 中的起始偏移。
             *
             * offset_j：
             *   当前设备分片在每个完整 chunk 内的偏移。
             */
            const uint64_t src_offset =
                static_cast<uint64_t>(offset) +
                static_cast<uint64_t>(offset_j);

            const uint64_t dst_offset =
                static_cast<uint64_t>(simple_offset);

            const uint64_t copy_size =
                static_cast<uint64_t>(chunk_size_j);

            const uint64_t n_copies =
                static_cast<uint64_t>(i_stop - i_start);

            const uint64_t src_stride =
                static_cast<uint64_t>(chunk_size_full);

            const uint64_t dst_stride =
                static_cast<uint64_t>(chunk_size_j);

            ggml_backend_local_file_result result =
    GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
            // 只有是静态才走
            if (can_use_local_file) {
                result =
                    ggml_backend_meta_try_set_local_file_2d(
                        simple_tensor,
                        src_offset,
                        dst_offset,
                        copy_size,
                        n_copies,
                        src_stride,
                        dst_stride);
            }

            switch (result) {
                case GGML_BACKEND_LOCAL_FILE_HANDLED: {
                    /*
                     * RPC 服务端已经从本地 GGUF 读取并写入，
                     * 不再发送 data 中的权重。
                     */
                } break;

                case GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED: {
                    /*
                     * CPU、CUDA、旧版 RPC，或者服务端没有本地模型，
                     * 使用 llama.cpp 原来的内存复制路径。
                     */
                    ggml_backend_tensor_set_2d(
                        simple_tensor,
                        static_cast<const char *>(data) +
                            offset_j,
                        simple_offset,
                        chunk_size_j,
                        static_cast<size_t>(
                            i_stop - i_start),
                        chunk_size_j,
                        chunk_size_full);
                } break;

                case GGML_BACKEND_LOCAL_FILE_ERROR: {
                    GGML_ABORT(
                        "failed to load tensor '%s' "
                        "from RPC local file: "
                        "src_offset=%" PRIu64
                        ", dst_offset=%" PRIu64
                        ", copy_size=%" PRIu64
                        ", n_copies=%" PRIu64,
                        simple_tensor->name,
                        src_offset,
                        dst_offset,
                        copy_size,
                        n_copies);
                } break;

                default: {
                    GGML_ABORT(
                        "invalid RPC local-file result");
                }
            }

            /*
             * 注意：无论本地加载还是普通复制，
             * 都要继续累计下一个设备的源分片偏移。
             */
            offset_j += chunk_size_j;
        }

        /*
         * 检查所有设备的局部分片是否刚好组成完整 chunk。
         */
        GGML_ASSERT(offset_j == chunk_size_full);
    } break;

    case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    for (size_t j = 0; j < n_bufs; ++j) {
        ggml_tensor * simple_tensor =
            ggml_backend_meta_buffer_simple_tensor(
                tensor,
                j);

        GGML_ASSERT(simple_tensor != nullptr);

        ggml_backend_local_file_result result =
            GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;

        /*
         * 只有静态模型权重才尝试 RPC 本地 GGUF。
         *
         * 运行时 embd、tokens、mask 等直接保持
         * NOT_SUPPORTED，然后进入普通 set_tensor。
         */
        if (can_use_local_file) {
            result =
                ggml_backend_meta_try_set_local_file_2d(
                    simple_tensor,
                    /* src_offset = */ static_cast<uint64_t>(offset),
                    /* dst_offset = */ static_cast<uint64_t>(offset),
                    /* copy_size  = */ static_cast<uint64_t>(size),
                    /* n_copies   = */ 1,
                    /* src_stride = */ static_cast<uint64_t>(size),
                    /* dst_stride = */ static_cast<uint64_t>(size));
        }

        switch (result) {
            case GGML_BACKEND_LOCAL_FILE_HANDLED: {
                // RPC服务端已经从本地GGUF加载完成
            } break;

            case GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED: {
                /*
                 * 包括：
                 * 1. CPU/CUDA后端；
                 * 2. 运行时计算张量；
                 * 3. RPC不支持本地文件扩展；
                 * 4. 非静态张量。
                 */
                ggml_backend_tensor_set(
                    simple_tensor,
                    data,
                    offset,
                    size);
            } break;

            case GGML_BACKEND_LOCAL_FILE_ERROR: {
                GGML_ABORT(
                    "failed to load mirrored tensor '%s' "
                    "from RPC local file",
                    simple_tensor->name);
            } break;

            default: {
                GGML_ABORT(
                    "invalid RPC local-file result");
            }
        }
    }
} break;

    case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
        /*
         * 这里不能直接从本地 GGUF 加载。
         *
         * 写入的数据不是文件中的原始数据，
         * 而是原始 F32 数据除以 n_bufs 后产生的新数据。
         */
        GGML_ASSERT(tensor->type == GGML_TYPE_F32);

        const int64_t ne =
            ggml_nelements(tensor);

        std::vector<float> tmp;
        tmp.reserve(ne);

        for (int64_t i = 0; i < ne; ++i) {
            tmp.push_back(
                static_cast<const float *>(data)[i] /
                n_bufs);
        }

        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor =
                ggml_backend_meta_buffer_simple_tensor(
                    tensor,
                    j);

            ggml_backend_tensor_set(
                simple_tensor,
                tmp.data(),
                offset,
                size);
        }
    } break;

    default: {
        GGML_ABORT("fatal error");
    }
}
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

// Extra full Q/K/V weights for native token-sharded projector, prepared
// alongside model tensor allocation. TP model weights stay split for Decode.
static void ggml_backend_meta_prepare_resident_qkv(
        ggml_context * model_ctx, ggml_backend_meta_buffer_context * meta_ctx,
        ggml_backend_buffer_type_t meta_buft) {
    const char * raw=std::getenv("LLAMA_GALAXY_NATIVE_SP_QKV_RESIDENT_WEIGHTS");
    if (!raw || std::strcmp(raw,"1")!=0) return;
    const auto plan=ggml_galaxy_causal_layers::native_qkv_from_environment();
    if (!plan.valid() || plan.layers.empty() || meta_ctx->bufs.size()!=2) {
        throw std::invalid_argument("SP QKV resident weights require a valid two-rank native prefix");
    }
    size_t total_bytes=0, selected_tensors=0;
    for (ggml_tensor * t=ggml_get_first_tensor(model_ctx);
         t!=nullptr; t=ggml_get_next_tensor(model_ctx,t)) {
        int layer=-1;
        const std::string name(t->name);
        for (const int candidate:plan.layers) {
            const std::string stem="blk."+std::to_string(candidate)+".attn_";
            if (name==stem+"q.weight" || name==stem+"k.weight" ||
                    name==stem+"v.weight") {
                layer=candidate;
                break;
            }
        }
        if (layer<0) continue;
        if (t->op!=GGML_OP_NONE || t->view_src || t->ne[2]!=1 ||
                t->ne[3]!=1 || !ggml_is_contiguous(t) ||
                ggml_backend_meta_get_split_state(t,false).axis!=GGML_BACKEND_SPLIT_AXIS_1) {
            throw std::invalid_argument("SP QKV resident model weight is not a static TP head shard: "+name);
        }
        ggml_backend_meta_resident_qkv mirror;
        mirror.total_bytes=ggml_nbytes(t);
        for (size_t rank=0;rank<2;++rank) {
            const ggml_init_params params={size_t(64)*1024,nullptr,true};
            ggml_context * raw_ctx=ggml_init(params);
            if (!raw_ctx) throw std::runtime_error("SP QKV resident GGML context allocation failed");
            ggml_context_ptr own_ctx(raw_ctx);
            ggml_tensor * full=ggml_new_tensor_2d(raw_ctx,t->type,t->ne[0],t->ne[1]);
            ggml_set_name(full,t->name);
            const auto physical_buft=ggml_backend_meta_buft_simple_buft(meta_buft,rank);
            ggml_backend_buffer_ptr physical(
                    ggml_backend_alloc_ctx_tensors_from_buft(raw_ctx,physical_buft));
            if (!physical || !full->buffer || ggml_nbytes(full)!=mirror.total_bytes)
                throw std::runtime_error("SP QKV resident physical weight allocation failed: "+name);
            ggml_backend_buffer_set_usage(physical.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            mirror.tensors.push_back(full);
            mirror.contexts.emplace_back(std::move(own_ctx));
            mirror.buffers.emplace_back(std::move(physical));
        }
        if (!meta_ctx->resident_qkv.emplace(t,std::move(mirror)).second) {
            throw std::runtime_error("duplicate SP QKV resident weight "+name);
        }
        ++selected_tensors;
        total_bytes+=ggml_nbytes(t);
        GGML_LOG_INFO("[GALAXY_SP_QKV_RESIDENT_ALLOC] layer=%d tensor=%s"
                " bytes_per_rank=%zu ranks=2 phase=MODEL_ALLOC status=PASS\n",
                layer,name.c_str(),ggml_nbytes(t));
    }
    // Called for both MODEL and COMPUTE Meta buffer contexts; not every
    // context contains model weights. Never require global coverage within
    // one buffer. At inference each selected Q/K/V must resolve to a fully
    // loaded mirror or the native projector rejects that layer.
    if (selected_tensors) {
        GGML_LOG_INFO("[GALAXY_SP_QKV_RESIDENT_ALLOC_SUM] requested_layers=%zu"
                " weights_in_this_buffer=%zu bytes_per_rank=%zu status=PASS\n",
                plan.layers.size(),selected_tensors,total_bytes);
    }
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    // If optional resident allocations fail, release all Meta and physical
    // buffers instead of leaking model memory during a failed model load.
    ggml_backend_buffer_ptr meta_guard(meta_buf);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {    //这里是张量切分主要逻辑
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t); //后端切分张量
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    // Both physical model buffers now exist; allocate mirrors BEFORE
    // llama_model_loader starts writing their GGUF weights.
    ggml_backend_meta_prepare_resident_qkv(ctx,meta_buf_ctx,buft);
    return meta_guard.release();    //里面的context指向的就是meta_buf_ctx
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    const bool profiling = ggml_galaxy_profile::enabled();
    const auto graph_start = ggml_galaxy_profile::clock::now();
    const uint64_t graph_id = profiling ? ggml_galaxy_profile::new_graph() : 0;
    ggml_galaxy_profile::current_graph() = graph_id;
    bool has_galaxy_markers = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        has_galaxy_markers |= galaxy_sp_get_transition(cgraph->nodes[i]) != galaxy_sp_transition::none;
    }
    // Debug only: capture the exact Meta subgraph or collective where PC
    // waits. Enable explicitly with LLAMA_GALAXY_SP_TRACE=1. Also trace
    // the final decode graph (which contains no SP markers).
    const char * trace_value = std::getenv("LLAMA_GALAXY_SP_TRACE");
    const bool trace = trace_value != nullptr && trace_value[0] == '1' && trace_value[1] == '\0';
    // QKV snapshots must be captured at their producing Meta subgraph's
    // completion. Reading graph-temporary tensors after the whole model
    // finishes is unsafe even when diagnostic OUTPUT flags are present.
    const char * qkv_probe_dir = std::getenv("LLAMA_GALAXY_QKV_PROBE_DIR");
    const bool qkv_probe_enabled = qkv_probe_dir != nullptr && qkv_probe_dir[0] != '\0';
    const auto live_plan = ggml_galaxy_causal_layers::from_environment();
    if (!live_plan.valid()) {
        std::fprintf(stderr,
                "[GALAXY_CAUSAL_LIVE] phase=CONFIG status=FAIL reason=%s\n",
                live_plan.error.c_str());
        return GGML_STATUS_FAILED;
    }
    const bool live_shadow_requested = !live_plan.layers.empty();
    const char * live_mode_raw = std::getenv("LLAMA_GALAXY_CAUSAL_LIVE_MODE");
    const std::string live_mode = live_mode_raw && live_mode_raw[0] ?
            std::string(live_mode_raw) : "SHADOW";
    if ((live_shadow_requested && live_mode != "SHADOW" && live_mode != "REPLACE") ||
            (!live_shadow_requested && live_mode_raw && live_mode_raw[0])) {
        GGML_LOG_ERROR("[GALAXY_CAUSAL_LIVE] invalid mode/selector combination\n");
        return GGML_STATUS_FAILED;
    }
    const bool live_replace_requested = live_shadow_requested && live_mode == "REPLACE";
    // Explicit first-prefill hybrid: producer-time SP Attention REPLACE,
    // then the existing Galaxy2 token-sharded FFN graph transitions.
    // This does not alter one-token Decode or enable SP KV history.
    const char * hybrid_raw = std::getenv("LLAMA_GALAXY_CAUSAL_HYBRID_FFN");
    const bool live_hybrid_ffn_requested = hybrid_raw &&
            hybrid_raw[0] == '1' && hybrid_raw[1] == '\0';
    const char * final_sp_raw = std::getenv("LLAMA_GALAXY_SP_FINAL_LAYER");
    const bool live_hybrid_final_sp_requested = live_hybrid_ffn_requested &&
            final_sp_raw && std::strcmp(final_sp_raw, "1") == 0;
    const auto native_qkv_plan = ggml_galaxy_causal_layers::native_qkv_from_environment();
    if (!native_qkv_plan.valid()) {
        GGML_LOG_ERROR("[GALAXY_SP_QKV] invalid selector: %s\n",
                native_qkv_plan.error.c_str());
        return GGML_STATUS_FAILED;
    }
    const bool native_sp_qkv_requested = !native_qkv_plan.layers.empty();
    const char * resident_raw=std::getenv("LLAMA_GALAXY_NATIVE_SP_QKV_RESIDENT_WEIGHTS");
    const bool resident_qkv_requested=resident_raw && std::strcmp(resident_raw,"1")==0;
    const char * no_tp_raw=std::getenv("LLAMA_GALAXY_NATIVE_SP_QKV_SKIP_TP_PREFILL");
    const bool no_tp_qkv_requested=no_tp_raw && std::strcmp(no_tp_raw,"1")==0;
    const auto selected_layer = [&](const char * name, const char * stem) -> int {
        return live_plan.match_boundary(name,stem);
    };
    const bool qkv_capture_enabled = qkv_probe_enabled || live_shadow_requested;
    const auto is_qkv_probe_node = [qkv_capture_enabled,&native_qkv_plan](const ggml_tensor * node) -> bool {
        if (!qkv_capture_enabled || node == nullptr ||
                (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 ||
                node->ne[1] <= 0) {
            return false;
        }
        return native_qkv_plan.match_boundary(node->name,"attn_norm")>=0 ||
               std::strncmp(node->name, "Qcur-", 5) == 0 ||
               std::strncmp(node->name, "Kcur-", 5) == 0 ||
               std::strncmp(node->name, "Vcur-", 5) == 0 ||
               std::strncmp(node->name, "fattn_raw-", 10) == 0 ||
               std::strncmp(node->name, "kqv_out-", 8) == 0;
    };
    // Per invocation only. Once captured, the independent host copy
    // survives subsequent backend buffer reuse and next-layer execution.
    std::map<const ggml_tensor *, std::vector<std::vector<uint8_t>>> qkv_live_snapshots;
    // Each selected layer owns a separate injection lifecycle. Never allow
    // an unfinished Layer N raw Flash output to be consumed/acknowledged
    // by Layer M's kqv_out, even when the two have the same byte extent.
    struct live_layer_state {
        std::array<std::vector<uint8_t>,2> tp_heads;
        bool native_sp_qkv = false;
        bool pending = false;
        bool verified = false;
        int64_t tokens = 0;
    };
    std::map<int,live_layer_state> live_states;
    std::set<int> live_shadow_verified;
    std::set<int> native_qkv_completed;
    struct native_tp_cache_bridge {
        ggml_galaxy_qkv_sp::projected sp;
        ggml_galaxy_causal_live::request geometry;
        int64_t tokens=0;
        unsigned injected_mask=0;
        bool prepared=false;
        bool consumed=false;
        double projection_wall_ms=0.0;
    };
    std::map<int,native_tp_cache_bridge> native_tp_cache_bridges;
    std::set<int> bypass_tp_cache_completed;

    // Name alone is not identity: projection, reshape and RoPE can all
    // share "Qcur-0". Restrict selection to the actual OUTPUT node that
    // was captured when its producing subgraph completed. Q/K must be
    // post-RoPE, V must be the final rank-sharded 3D tensor.
    const auto find_live_stage = [&](int layer, const char * stage,
                                      bool must_be_rope,
                                      bool require_capture = true) -> const ggml_tensor * {
        const std::string expected=std::string(stage)+"-"+std::to_string(layer);
        const ggml_tensor * match=nullptr;
        int named=0,eligible=0;
        for(int id=0;id<cgraph->n_nodes;++id) {
            const ggml_tensor * t=cgraph->nodes[id];
            if (!t || expected != t->name) continue;
            ++named;
            if ((t->flags & GGML_TENSOR_FLAG_OUTPUT)==0 ||
                    (require_capture && qkv_live_snapshots.find(t)==qkv_live_snapshots.end()) ||
                    (must_be_rope && t->op != GGML_OP_ROPE)) continue;
            ++eligible;
            if(match) throw std::invalid_argument("ambiguous captured model stage "+expected);
            match=t;
        }
        std::fprintf(stderr,
                "[GALAXY_CAUSAL_LIVE_NODE] name=%s candidates=%d"
                " captured_eligible=%d selected_op=%s"
                " selection=%s status=%s\n",
                expected.c_str(),named,eligible,
                match?ggml_op_name(match->op):"NONE",
                require_capture?"PRODUCER_CAPTURED_OUTPUT":"PLANNED_PRODUCER_OUTPUT",
                match?"PASS":"FAIL");
        std::fflush(stderr);
        if(!match) throw std::invalid_argument("missing captured model stage "+expected);
        return match;
    };

    const auto build_live_request = [&](int layer, const ggml_tensor * reference)
            -> ggml_galaxy_causal_live::request {
        const ggml_tensor * q=find_live_stage(layer,"Qcur",true);
        const ggml_tensor * k=find_live_stage(layer,"Kcur",true);
        const ggml_tensor * v=find_live_stage(layer,"Vcur",false);
        const int64_t dim=q->ne[0],tokens=q->ne[2];
        if(q->ne[3]!=1 || dim<=0 || tokens<8 || tokens>512 ||
                k->ne[0]!=dim || v->ne[0]!=dim ||
                k->ne[2]!=tokens || v->ne[2]!=tokens ||
                k->ne[3]!=1 || v->ne[3]!=1 ||
                (reference && (reference->ne[1]!=tokens ||
                        reference->ne[2]!=1 || reference->ne[3]!=1 ||
                        reference->ne[0]!=q->ne[0]*q->ne[1]))) {
            throw std::invalid_argument(
                    "unsupported live causal QKV/reference shape or token count");
        }
        // A token count >= 8 is NOT sufficient proof of a fresh first
        // Prefill. The native SP path deliberately sees only the current
        // token interval, so reject continuation/chunked prefill here.
        // The post-RoPE Q's src[1] is the actual model position tensor.
        const ggml_tensor * rope_pos=q->src[1];
        if(q->op!=GGML_OP_ROPE || !rope_pos ||
                rope_pos->type!=GGML_TYPE_I32 ||
                !ggml_is_contiguous(rope_pos) ||
                ggml_nelements(rope_pos)!=tokens) {
            throw std::invalid_argument(
                    "cannot prove zero-based contiguous RoPE positions for first prefill");
        }
        const auto check_pos=[&](const ggml_tensor * t,int rank) {
            if(!t || t->type!=GGML_TYPE_I32 ||
                    !ggml_is_contiguous(t) ||
                    ggml_nbytes(t)!=size_t(tokens)*sizeof(int32_t)) {
                throw std::invalid_argument("unavailable first-prefill position buffer");
            }
            std::vector<int32_t> pos(static_cast<size_t>(tokens),-1);
            if(t->buffer) {
                if(ggml_backend_buffer_is_meta(t->buffer)) {
                    throw std::invalid_argument("position backend view still points to Meta");
                }
                // Works for CPU input buffers and RPC/CUDA physical
                // views; get_tensor imposes the appropriate read fence.
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                ggml_backend_tensor_get(t,pos.data(),0,ggml_nbytes(t));
            } else if(t->data) {
                std::memcpy(pos.data(),t->data,ggml_nbytes(t));
            } else {
                throw std::invalid_argument("first-prefill RoPE positions not readable");
            }
            for(int64_t i=0;i<tokens;++i) {
                if(pos[static_cast<size_t>(i)]!=i) {
                    throw std::invalid_argument(
                            "historical KV or nonzero/noncontiguous positions are unsupported");
                }
            }
        };
        if(ggml_backend_buffer_is_meta(rope_pos->buffer)) {
            for(int rank=0;rank<2;++rank) {
                const ggml_tensor * local_pos=
                        ggml_backend_meta_buffer_simple_tensor(rope_pos,rank);
                check_pos(local_pos,rank);
            }
        } else {
            check_pos(rope_pos,0);
        }
        std::fprintf(stderr,
                "[GALAXY_CAUSAL_FRESH_PREFIX] layer=%d"
                " tokens=%" PRId64 " positions=0..%" PRId64
                " history=UNSUPPORTED status=PASS\n",
                layer,tokens,tokens-1);
        std::fflush(stderr);

        const auto take=[&](const ggml_tensor * t,bool headed,const char * label)
                -> ggml_galaxy_causal_live::head_shards {
            ggml_galaxy_causal_live::head_shards h;
            h.type=t->type;
            if (t->type!=GGML_TYPE_F32 ||
                    !ggml_backend_buffer_is_meta(t->buffer) ||
                    !ggml_is_contiguous(t))
                throw std::invalid_argument(std::string(label)+" is not dense F32 Meta");
            const auto found=qkv_live_snapshots.find(t);
            if(found==qkv_live_snapshots.end() || found->second.size()!=2)
                throw std::invalid_argument(std::string(label)+" has no producer snapshot");
            const auto state=ggml_backend_meta_get_split_state(t,false);
            const auto expected=headed?GGML_BACKEND_SPLIT_AXIS_1:
                                       GGML_BACKEND_SPLIT_AXIS_0;
            if(state.axis!=expected)
                throw std::invalid_argument(std::string(label)+" is not head-sharded");
            for(int rank=0;rank<2;++rank) {
                const ggml_tensor * local=ggml_backend_meta_buffer_simple_tensor(t,rank);
                if(!local || !local->buffer || local->type!=GGML_TYPE_F32 ||
                        !ggml_is_contiguous(local) ||
                        (headed ? (local->ne[0]!=dim || local->ne[2]!=tokens ||
                                   local->ne[3]!=1)
                                 : (local->ne[1]!=tokens || local->ne[2]!=1 ||
                                    local->ne[3]!=1 || local->ne[0]%dim!=0))) {
                    throw std::invalid_argument(std::string(label)+" rank shape mismatch");
                }
                h.heads[rank]=headed?local->ne[1]:local->ne[0]/dim;
                h.bytes[rank]=&found->second[rank];
                if(h.heads[rank]<=0 || h.bytes[rank]->size()!=ggml_nbytes(local))
                    throw std::invalid_argument(std::string(label)+" rank byte size mismatch");
            }
            return h;
        };
        ggml_galaxy_causal_live::request work;
        work.q=take(q,true,"Qcur");
        work.k=take(k,true,"Kcur");
        work.v=take(v,true,"Vcur");
        if(reference) work.reference=take(reference,false,"kqv_out");
        work.tokens=tokens;
        work.head_dim=dim;
        work.layer=layer;
        work.tile_tokens=ggml_galaxy_collectives::configured_tile_tokens();
        work.max_rmse=0.001;
        return work;
    };

    // First native SP QKV step: MIRRORED attention-norm activation is
    // scatter-by-token. Reassemble full quantized GGUF rows from the live
    // TP weight shards, preserving exact row byte order and quant type.
    // Decode / original TP KV writes are intentionally retained in v1.
    const auto build_native_sp_qkv = [&](int layer, const ggml_galaxy_causal_live::request & work)
            -> ggml_galaxy_qkv_sp::source {
        if (!native_qkv_plan.contains(layer))
            throw std::invalid_argument("SP QKV projection layer is not selected");
        const ggml_tensor * norm=find_live_stage(layer,"attn_norm",false);
        const auto snap=qkv_live_snapshots.find(norm);
        if ((norm->op!=GGML_OP_MUL && norm->op!=GGML_OP_RMS_NORM) ||
                norm->type!=GGML_TYPE_F32 ||
                !ggml_is_contiguous(norm) || norm->ne[1]!=work.tokens ||
                norm->ne[2]!=1 ||
                ggml_backend_meta_get_split_state(norm,false).axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                snap==qkv_live_snapshots.end() || snap->second.size()!=2 ||
                snap->second[0].size()!=ggml_nbytes(norm) ||
                snap->second[1].size()!=ggml_nbytes(norm)) {
            throw std::invalid_argument("layer "+std::to_string(layer)+
                    " normalized F32 mirrored producer snapshot missing");
        }
        ggml_galaxy_qkv_sp::source src;
        src.hidden=norm->ne[0];
        src.tokens=work.tokens;
        src.head_dim=work.head_dim;
        src.q_heads=ggml_galaxy_causal_live::total_heads(work.q);
        src.kv_heads=ggml_galaxy_causal_live::total_heads(work.k);
        src.normalized=snap->second[0];
        src.resident_weights=resident_qkv_requested;
        src.positions.resize(size_t(src.tokens));
        for (int64_t t=0;t<src.tokens;++t) src.positions[size_t(t)]=static_cast<int32_t>(t);

        // Follow the real producer nodes, NOT cgraph->leafs. Backend-created
        // graphs can omit model weights from their leaf enumeration despite
        // their presence as MatMul src[0] and as live Meta static tensors.
        const ggml_tensor * q=find_live_stage(layer,"Qcur",true,!no_tp_qkv_requested);
        const ggml_tensor * k=find_live_stage(layer,"Kcur",true,!no_tp_qkv_requested);
        const ggml_tensor * v=find_live_stage(layer,"Vcur",false,!no_tp_qkv_requested);
        const std::string prefix="blk."+std::to_string(layer)+".";
        const auto full_weight=[&](const ggml_tensor * producer,
                                   const char * suffix,
                                   const std::array<int64_t,2> & rank_heads,
                                   int weight_kind)
                -> ggml_galaxy_qkv_sp::weight {
            const std::string name=prefix+suffix;
            const ggml_tensor * global=ggml_galaxy_qkv_sp::projection_weight(producer,norm,name);
            if (!global->buffer || !ggml_backend_buffer_is_meta(global->buffer)) {
                throw std::invalid_argument("QKV weight "+name+" is not in a Meta buffer");
            }
            const auto split=ggml_backend_meta_get_split_state(global,false);
            if (split.axis!=GGML_BACKEND_SPLIT_AXIS_1 ||
                    global->ne[2]!=1 || global->ne[3]!=1 ||
                    global->ne[1]!=(rank_heads[0]+rank_heads[1])*work.head_dim) {
                throw std::invalid_argument("QKV weight "+name+" does not match TP head-shard layout");
            }
            ggml_galaxy_qkv_sp::weight full;
            full.type=global->type;
            full.in=global->ne[0];
            full.out=global->ne[1];
            const size_t row_bytes=ggml_row_size(full.type,full.in);
            if (resident_qkv_requested) {
                // Model buffer owns the replicas and their lifetime.
                // No inference-time TP shard readback or weight uploads.
                for (int rank=0;rank<2;++rank) {
                    ggml_tensor * resident=ggml_backend_meta_resident_qkv_tensor(global,rank);
                    if (!resident || !resident->buffer ||
                            resident->type!=full.type ||
                            resident->ne[0]!=full.in || resident->ne[1]!=full.out ||
                            resident->ne[2]!=1 || resident->ne[3]!=1 ||
                            ggml_nbytes(resident)!=row_bytes*size_t(full.out)) {
                        throw std::invalid_argument("resident QKV weight not fully loaded on rank "+
                                std::to_string(rank)+": "+name);
                    }
                    if (!ggml_backend_supports_buft(
                            backend_ctx->backend_configs[rank].backend,
                            ggml_backend_buffer_get_type(resident->buffer))) {
                        throw std::invalid_argument("resident QKV weight on unsupported physical backend: "+name);
                    }
                    src.physical_weights[rank][weight_kind]=resident;
                }
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_WEIGHT] layer=%d tensor=%s"
                        " source=RESIDENT_MODEL_LOAD meta_split=AXIS_1"
                        " rows=%" PRId64 " bytes_per_rank=%zu"
                        " runtime_read_bytes=0 runtime_upload_bytes=0 status=PASS\n",
                        layer,name.c_str(),full.out,row_bytes*size_t(full.out));
                std::fflush(stderr);
                return full;
            }
            full.bytes.resize(row_bytes*size_t(full.out));
            int64_t rows=0;
            for (int rank=0;rank<2;++rank) {
                const ggml_tensor * local=ggml_backend_meta_buffer_simple_tensor(global,rank);
                const int64_t expected=rank_heads[rank]*work.head_dim;
                if (!local || !local->buffer || local->type!=full.type ||
                        !ggml_is_contiguous(local) || local->ne[0]!=full.in ||
                        local->ne[1]!=expected || local->ne[2]!=1 || local->ne[3]!=1 ||
                        ggml_nbytes(local)!=row_bytes*size_t(expected) ||
                        rows+expected>full.out) {
                    throw std::invalid_argument("TP QKV weight shard dimensions mismatch "+name+
                            " rank="+std::to_string(rank));
                }
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                ggml_backend_tensor_get(local,full.bytes.data()+row_bytes*size_t(rows),
                        0,ggml_nbytes(local));
                rows+=expected;
            }
            if (rows!=full.out) throw std::invalid_argument("incomplete QKV weight rows "+name);
            full.validate();
            std::fprintf(stderr,
                    "[GALAXY_SP_QKV_WEIGHT] layer=%d tensor=%s"
                    " source=MATMUL_SRC0 meta_split=AXIS_1"
                    " rows=%" PRId64 " rank_rows=%" PRId64 "+%" PRId64
                    " bytes=%zu status=PASS\n",
                    layer,name.c_str(),full.out,
                    rank_heads[0]*work.head_dim,rank_heads[1]*work.head_dim,
                    full.bytes.size());
            std::fflush(stderr);
            return full;
        };
        src.wq=full_weight(q,"attn_q.weight",work.q.heads,0);
        src.wk=full_weight(k,"attn_k.weight",work.k.heads,1);
        src.wv=full_weight(v,"attn_v.weight",work.v.heads,2);
        if (q->src[2]!=k->src[2])
            throw std::invalid_argument("Q/K RoPE factors disagree");
        std::memcpy(src.q_rope_params.data(),q->op_params,sizeof(q->op_params));
        std::memcpy(src.k_rope_params.data(),k->op_params,sizeof(k->op_params));
        const ggml_tensor * factors=q->src[2];
        if (factors) {
            if (factors->type!=GGML_TYPE_F32 || !ggml_is_contiguous(factors) ||
                    ggml_nelements(factors)!=work.head_dim/2)
                throw std::invalid_argument("unsupported Llama RoPE freq factors");
            src.rope_factors.resize(ggml_nbytes(factors));
            if (ggml_backend_buffer_is_meta(factors->buffer)) {
                const ggml_tensor * physical=ggml_backend_meta_buffer_simple_tensor(factors,0);
                if (!physical || !physical->buffer ||
                        ggml_nbytes(physical)!=src.rope_factors.size())
                    throw std::invalid_argument("missing physical RoPE factors");
                ggml_backend_tensor_get(physical,src.rope_factors.data(),0,src.rope_factors.size());
            } else if (factors->buffer) {
                ggml_backend_tensor_get(factors,src.rope_factors.data(),0,src.rope_factors.size());
            } else if (factors->data) {
                std::memcpy(src.rope_factors.data(),factors->data,src.rope_factors.size());
            } else {
                throw std::invalid_argument("original RoPE factors unreadable");
            }
        }
        ggml_galaxy_qkv_sp::validate(src);
        return src;
    };

    // Build first-Prefill geometry from pinned *planned* TP output nodes.
    // The Q/K/V producers have not run yet; this helper deliberately does
    // not require their snapshots. The model graph remains unchanged.
    const auto build_bypass_geometry = [&](int layer)
            -> ggml_galaxy_causal_live::request {
        const ggml_tensor * q=find_live_stage(layer,"Qcur",true,false);
        const ggml_tensor * k=find_live_stage(layer,"Kcur",true,false);
        const ggml_tensor * v=find_live_stage(layer,"Vcur",false,false);
        ggml_galaxy_causal_live::request work;
        // An early projection runs after attn_norm but before the original
        // Q/K/V producers. The old live bridge checks the Meta split axes
        // AFTER their producer snapshots. Here, validate the symbolic
        // tensors and two physical rank views; defer final axis checking
        // to the same build_live_request() used by the proven late path.
        const auto q_axis=ggml_backend_meta_get_split_state(q,false).axis;
        const auto k_axis=ggml_backend_meta_get_split_state(k,false).axis;
        const auto v_axis=ggml_backend_meta_get_split_state(v,false).axis;
        const bool global_geometry_ok =
                q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F32 &&
                v->type==GGML_TYPE_F32 &&
                ggml_backend_buffer_is_meta(q->buffer) &&
                ggml_backend_buffer_is_meta(k->buffer) &&
                ggml_backend_buffer_is_meta(v->buffer) &&
                ggml_is_contiguous(q) && ggml_is_contiguous(k) &&
                ggml_is_contiguous(v) &&
                q->ne[0]>0 && q->ne[2]>=8 && q->ne[2]<=512 &&
                q->ne[3]==1 && k->ne[0]==q->ne[0] &&
                k->ne[2]==q->ne[2] && k->ne[3]==1 &&
                v->ne[0]==q->ne[0] && v->ne[2]==q->ne[2] &&
                v->ne[3]==1 &&
                // Dense Llama Q/K/V are ALL split by attention Head axis 1.
                // The original V producer is 3D [head_dim,kv_heads,tokens],
                // not a flattened [kv_heads*head_dim,tokens] view.
                q_axis==GGML_BACKEND_SPLIT_AXIS_1 &&
                k_axis==GGML_BACKEND_SPLIT_AXIS_1 &&
                v_axis==GGML_BACKEND_SPLIT_AXIS_1;
        std::fprintf(stderr,
                "[GALAXY_SP_QKV_EARLY_LAYOUT] layer=%d phase=PRE_PRODUCER"
                " Q=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " type=%s"
                " K=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " type=%s"
                " V=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " type=%s"
                " meta_axes=%s,%s,%s final_axis_check=DEFERRED status=%s\n",
                layer,
                q->ne[0],q->ne[1],q->ne[2],q->ne[3],ggml_type_name(q->type),
                k->ne[0],k->ne[1],k->ne[2],k->ne[3],ggml_type_name(k->type),
                v->ne[0],v->ne[1],v->ne[2],v->ne[3],ggml_type_name(v->type),
                ggml_backend_meta_split_axis_name(q_axis),
                ggml_backend_meta_split_axis_name(k_axis),
                ggml_backend_meta_split_axis_name(v_axis),
                global_geometry_ok?"PASS":"FAIL");
        std::fflush(stderr);
        if (!global_geometry_ok)
            throw std::invalid_argument("early Q/K/V symbolic geometry/type incompatible with TP cache bridge");
        work.tokens=q->ne[2];
        work.head_dim=q->ne[0];
        work.layer=layer;
        for (int rank=0;rank<2;++rank) {
            const ggml_tensor * qp=ggml_backend_meta_buffer_simple_tensor(q,rank);
            const ggml_tensor * kp=ggml_backend_meta_buffer_simple_tensor(k,rank);
            const ggml_tensor * vp=ggml_backend_meta_buffer_simple_tensor(v,rank);
            const bool physical_ok =
                    qp && kp && vp &&
                    qp->buffer && kp->buffer && vp->buffer &&
                    qp->type==GGML_TYPE_F32 && kp->type==GGML_TYPE_F32 &&
                    vp->type==GGML_TYPE_F32 &&
                    ggml_is_contiguous(qp) && ggml_is_contiguous(kp) &&
                    ggml_is_contiguous(vp) &&
                    qp->ne[0]==work.head_dim && kp->ne[0]==work.head_dim &&
                    qp->ne[1]>0 && kp->ne[1]>0 &&
                    qp->ne[2]==work.tokens && kp->ne[2]==work.tokens &&
                    qp->ne[3]==1 && kp->ne[3]==1 &&
                    vp->ne[0]==work.head_dim &&
                    vp->ne[1]==kp->ne[1] &&
                    vp->ne[2]==work.tokens && vp->ne[3]==1;
            std::fprintf(stderr,
                    "[GALAXY_SP_QKV_EARLY_RANK_LAYOUT] layer=%d rank=%d"
                    " Q=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64
                    " K=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64
                    " V=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64
                    " physical_ready=%d status=%s\n",
                    layer,rank,
                    qp?qp->ne[0]:-1,qp?qp->ne[1]:-1,
                    qp?qp->ne[2]:-1,qp?qp->ne[3]:-1,
                    kp?kp->ne[0]:-1,kp?kp->ne[1]:-1,
                    kp?kp->ne[2]:-1,kp?kp->ne[3]:-1,
                    vp?vp->ne[0]:-1,vp?vp->ne[1]:-1,
                    vp?vp->ne[2]:-1,vp?vp->ne[3]:-1,
                    int(qp && kp && vp && qp->buffer && kp->buffer && vp->buffer),
                    physical_ok?"PASS":"FAIL");
            std::fflush(stderr);
            if (!physical_ok)
                throw std::invalid_argument("physical TP QKV cache-producer rank shape/type mismatch");
            work.q.heads[rank]=qp->ne[1];
            work.k.heads[rank]=kp->ne[1];
            work.v.heads[rank]=vp->ne[1];
        }
        // Check the ACTUAL model positions *before* skipping TP nodes.
        // A continuation prefill must not be treated as positions 0..N-1.
        const ggml_tensor * rope_pos=q->src[1];
        if (!rope_pos || rope_pos->type!=GGML_TYPE_I32 ||
                !ggml_is_contiguous(rope_pos) ||
                ggml_nelements(rope_pos)!=work.tokens) {
            throw std::invalid_argument("SP KV bridge missing original RoPE position tensor");
        }
        for (int rank=0;rank<2;++rank) {
            const ggml_tensor * p=ggml_backend_buffer_is_meta(rope_pos->buffer) ?
                    ggml_backend_meta_buffer_simple_tensor(rope_pos,rank) : rope_pos;
            if (!p || p->type!=GGML_TYPE_I32 ||
                    !ggml_is_contiguous(p) ||
                    ggml_nbytes(p)!=size_t(work.tokens)*sizeof(int32_t)) {
                throw std::invalid_argument("SP KV bridge lacks readable original positions");
            }
            std::vector<int32_t> values(size_t(work.tokens),-1);
            if(p->buffer) {
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                ggml_backend_tensor_get(p,values.data(),0,ggml_nbytes(p));
            } else if(p->data) {
                std::memcpy(values.data(),p->data,ggml_nbytes(p));
            } else {
                throw std::invalid_argument("SP KV bridge position tensor has no backing data");
            }
            for(int64_t t=0;t<work.tokens;++t) {
                if(values[size_t(t)]!=t)
                    throw std::invalid_argument("SP KV cache bypass only supports zero-based fresh prefill");
            }
        }
        return work;
    };

    // Ensure the cache writes are *actually* part of this original Llama
    // graph before skipping its TP projection producers. We preserve the
    // original GGML SET_ROWS nodes, their index tensors and KV allocation.
    const auto verify_cache_stores = [&](int layer, size_t start, size_t stop) {
        const ggml_tensor * k=find_live_stage(layer,"Kcur",true,false);
        const ggml_tensor * v=find_live_stage(layer,"Vcur",false,false);
        int k_stores=0, v_stores=0;
        const auto follows_stage=[](const ggml_tensor * src,
                                    const ggml_tensor * stage)->bool {
            for(int depth=0;src && depth<12;++depth) {
                if(src==stage) return true;
                if(src->op!=GGML_OP_VIEW && src->op!=GGML_OP_RESHAPE &&
                        src->op!=GGML_OP_CONT && src->op!=GGML_OP_PERMUTE)
                    return false;
                src=src->src[0];
            }
            return false;
        };
        for(size_t j=start;j<stop;++j) {
            const ggml_tensor * op=cgraph->nodes[j];
            if(op->op!=GGML_OP_SET_ROWS) continue;
            if(follows_stage(op->src[1],k)) ++k_stores;
            if(follows_stage(op->src[1],v)) ++v_stores;
        }
        if(k_stores!=1 || v_stores!=1)
            throw std::invalid_argument("SP-to-TP K/V cache bridge requires one original K and one V SET_ROWS");
        std::fprintf(stderr,
                "[GALAXY_SP_QKV_CACHE_STORES] layer=%d"
                " source=SP_KV_TO_TP_HEAD cache_k_stores=1 cache_v_stores=1"
                " cache_indices=ORIGINAL_LLAMA phase=PREFLIGHT status=READY\n",layer);
        std::fflush(stderr);
    };

    if (trace) {
        GGML_LOG_INFO("[GALAXY_META_TRACE] event=GRAPH_BEGIN nodes=%d backends=%zu sp=%d uid=%" PRIu64 "\n",
                cgraph->n_nodes, n_backends, int(has_galaxy_markers), cgraph->uid);
    }
    if (has_galaxy_markers && n_backends != 2) {
        GGML_LOG_ERROR("[GALAXY_SP] graph transitions need exactly two Meta backends\n");
        return GGML_STATUS_FAILED;
    }
    if (no_tp_qkv_requested && (!resident_qkv_requested ||
            !native_sp_qkv_requested || !live_replace_requested ||
            !live_hybrid_ffn_requested || n_backends!=2)) {
        GGML_LOG_ERROR("[GALAXY_SP_QKV_TP_BYPASS] requires two-rank resident native QKV, Hybrid REPLACE\n");
        return GGML_STATUS_FAILED;
    }
    if (resident_qkv_requested && !native_sp_qkv_requested) {
        GGML_LOG_ERROR("[GALAXY_SP_QKV_RESIDENT] requires native SP QKV prefix\n");
        return GGML_STATUS_FAILED;
    }
    if (native_sp_qkv_requested) {
        if (!live_hybrid_ffn_requested || !live_replace_requested || n_backends!=2) {
            GGML_LOG_ERROR("[GALAXY_SP_QKV] requires two-rank Hybrid REPLACE\n");
            return GGML_STATUS_FAILED;
        }
        for (const int layer : native_qkv_plan.layers) {
            if (!live_plan.contains(layer)) {
                GGML_LOG_ERROR("[GALAXY_SP_QKV] layer=%d not in selected Attention prefix\n",
                        layer);
                return GGML_STATUS_FAILED;
            }
        }
    }
    if (live_hybrid_ffn_requested) {
        const char * galaxy2_raw = std::getenv("LLAMA_GALAXY2");
        const bool galaxy2 = galaxy2_raw && std::strcmp(galaxy2_raw, "1") == 0;
        const char * sp_raw = std::getenv("LLAMA_GALAXY_SP");
        const bool galaxy_sp = sp_raw && std::strcmp(sp_raw, "1") == 0;
        if (!live_replace_requested || !galaxy2 || !galaxy_sp || n_backends != 2) {
            GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN] invalid config: requires REPLACE+Galaxy2 SP on two ranks\n");
            return GGML_STATUS_FAILED;
        }
        if (live_hybrid_final_sp_requested && live_plan.layers.size() != 28) {
            GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN] final-layer SP requires the 28-layer Llama3.2 prefix\n");
            return GGML_STATUS_FAILED;
        }
    }
    if (live_shadow_requested && has_galaxy_markers && !live_hybrid_ffn_requested) {
        GGML_LOG_ERROR("[GALAXY_CAUSAL_LIVE] SP markers require explicit hybrid FFN opt-in\n");
        return GGML_STATUS_FAILED;
    }
    if (qkv_capture_enabled && n_backends != 2) {
        GGML_LOG_ERROR("[GALAXY_QKV_CAPTURE] producer-time capture requires two Meta backends\n");
        return GGML_STATUS_FAILED;
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                // A Galaxy layout transition must immediately follow the
                // original PARTIAL output. Delaying the AllReduce would
                // incorrectly consume/overwrite one of its partial inputs.
                if (has_galaxy_markers || qkv_capture_enabled) {
                    // Never delay a reduction across an opt-in diagnostic
                    // boundary: the producing tensor must remain last.
                    return i;
                }
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes ||
                        split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                        galaxy_sp_get_transition(node) != galaxy_sp_transition::none ||
                        is_qkv_probe_node(node);
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = 0;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    // The marker is allocated as a real ggml tensor but Meta
                    // fills it with the appropriate collective. Running the
                    // DUP op would copy a full tensor into a sequence shard.
                    if (galaxy_sp_get_transition(cgraph->nodes[i_node]) != galaxy_sp_transition::none) {
                        continue;
                    }
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[cgraph_ij->n_nodes++] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                if (cgraph->uid != 0) {
                    uint64_t uid = cgraph->uid;
                    uid ^= (uint64_t) (j + 1)       * 0x9e3779b97f4a7c15ULL;
                    uid ^= (uint64_t) (i_graph + 1) * 0xbf58476d1ce4e5b9ULL;
                    uid ^= uid >> 30;
                    uid *= 0xbf58476d1ce4e5b9ULL;
                    uid ^= uid >> 27;
                    uid *= 0x94d049bb133111ebULL;
                    uid ^= uid >> 31;
                    cgraph_ij->uid = uid != 0 ? uid : ggml_graph_next_uid();
                } else {
                    cgraph_ij->uid = ggml_graph_next_uid();
                }
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    size_t galaxy_rs_count = 0;
    size_t galaxy_ag_count = 0;
    size_t galaxy_scatter_count = 0;
    size_t galaxy_ar_count = 0;
    // Count successful (executed) Galaxy markers, not just graph nodes.
    std::set<int> hybrid_attention_rs;
    std::set<int> hybrid_residual_scatter;
    std::set<int> hybrid_ffn_ag;
    std::set<int> hybrid_attn_ag;
    std::set<int> hybrid_ffn_rs;
    size_t bypass_actual_tp_nodes_skipped=0;
    size_t bypass_actual_kv_store_ops=0;
    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        const size_t i_start = backend_ctx->backend_configs[0].cgraphs[i].offset;
        const size_t i_stop = i + 1 < backend_ctx->n_subgraphs
                ? backend_ctx->backend_configs[0].cgraphs[i + 1].offset
                : size_t(cgraph->n_nodes);
        GGML_ASSERT(i_stop > i_start);
        ggml_tensor * original_last = cgraph->nodes[i_stop - 1];
        const int raw_layer = live_replace_requested ?
                selected_layer(original_last->name,"fattn_raw") : -1;
        const bool is_replace_raw_boundary = raw_layer >= 0 &&
                original_last->ne[2]>=8;
        const int q_bypass_layer = no_tp_qkv_requested ?
                selected_layer(original_last->name,"Qcur") : -1;
        const int k_bypass_layer = no_tp_qkv_requested ?
                selected_layer(original_last->name,"Kcur") : -1;
        const int v_bypass_layer = no_tp_qkv_requested ?
                selected_layer(original_last->name,"Vcur") : -1;
        const int producer_layer = q_bypass_layer>=0 ? q_bypass_layer :
                                   k_bypass_layer>=0 ? k_bypass_layer :
                                   v_bypass_layer;
        const char producer_kind = q_bypass_layer>=0?'q':
                                   k_bypass_layer>=0?'k':'v';
        const bool is_bypass_producer = producer_layer>=0 &&
                native_qkv_plan.contains(producer_layer) &&
                original_last->ne[2]>=8;
        if (is_replace_raw_boundary) {
            const auto split=ggml_backend_meta_get_split_state(original_last,false);
            if (original_last->op!=GGML_OP_FLASH_ATTN_EXT ||
                    original_last->type!=GGML_TYPE_F32 ||
                    (original_last->flags & GGML_TENSOR_FLAG_OUTPUT)==0 ||
                    split.axis!=GGML_BACKEND_SPLIT_AXIS_1 ||
                    !ggml_is_contiguous(original_last) ||
                    original_last->ne[3]!=1) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] raw Flash boundary contract failed\n");
                return GGML_STATUS_FAILED;
            }
            auto & state=live_states[raw_layer];
            if (state.pending || state.verified) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] layer=%d duplicate or unconsumed Flash output\n",raw_layer);
                return GGML_STATUS_FAILED;
            }
            // Prefill layers are visited in ascending order. A later
            // layer must not begin before every preceding selected layer
            // proved its SP->TP kqv_out consumption.
            for(const int preceding:live_plan.layers) {
                if(preceding>=raw_layer) break;
                if(!live_states[preceding].verified) {
                    GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] layer=%d previous layer %d not verified\n",
                            raw_layer,preceding);
                    return GGML_STATUS_FAILED;
                }
            }
        }
        if (is_bypass_producer) {
            try {
                const auto state=native_tp_cache_bridges.find(producer_layer);
                const int expected_nodes=producer_kind=='v'?2:3;
                const ggml_op terminal=producer_kind=='v' ?
                        GGML_OP_RESHAPE : GGML_OP_ROPE;
                if (state==native_tp_cache_bridges.end() ||
                        !state->second.prepared ||
                        state->second.tokens!=original_last->ne[2] ||
                        state->second.injected_mask !=
                            (producer_kind=='q'?0u:producer_kind=='v'?1u:3u) ||
                        i_stop-i_start!=size_t(expected_nodes) ||
                        original_last->op!=terminal ||
                        cgraph->nodes[i_start]->op!=GGML_OP_MUL_MAT ||
                        cgraph->nodes[i_start+1]->op!=GGML_OP_RESHAPE ||
                        (expected_nodes==3 && cgraph->nodes[i_start+2]->op!=GGML_OP_ROPE)) {
                    throw std::invalid_argument(
                            "unexpected TP-QKV graph topology/order; refusing to skip any cache producer");
                }
                const ggml_tensor * expected=find_live_stage(producer_layer,
                        producer_kind=='q'?"Qcur":producer_kind=='k'?"Kcur":"Vcur",
                        producer_kind!='v',false);
                if (expected!=original_last)
                    throw std::invalid_argument("TP-QKV terminal identity mismatch");
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_TP_SKIP] layer=%d tensor=%s status=FAIL reason=%s\n",
                        producer_layer,original_last->name,e.what());
                return GGML_STATUS_FAILED;
            }
        }
        if (no_tp_qkv_requested && is_replace_raw_boundary &&
                native_qkv_plan.contains(raw_layer)) {
            try {
                const auto state=native_tp_cache_bridges.find(raw_layer);
                if (state==native_tp_cache_bridges.end() ||
                        !state->second.prepared ||
                        state->second.injected_mask!=7u ||
                        state->second.consumed) {
                    throw std::invalid_argument("K/V cache not prepared by all three SP-QKV projections");
                }
                verify_cache_stores(raw_layer,i_start,i_stop);
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_CACHE_STORES] layer=%d status=FAIL reason=%s\n",
                        raw_layer,e.what());
                return GGML_STATUS_FAILED;
            }
        }
        std::vector<double> dispatch_ms(n_backends, 0);
        std::vector<ggml_galaxy_profile::instant> dispatch_start(n_backends);

        if (trace) {
            GGML_LOG_INFO("[GALAXY_META_TRACE] event=SUBGRAPH_BEGIN idx=%zu total=%zu range=%zu..%zu last=%s op=%s transition=%d\n",
                    i, backend_ctx->n_subgraphs, i_start, i_stop,
                    original_last->name, ggml_op_name(original_last->op),
                    int(galaxy_sp_get_transition(original_last)));
        }
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * simple_graph = bcj.cgraphs[i].cgraph_main;
            if (simple_graph->n_nodes == 0) {
                continue;  // graph-only Galaxy collective marker
            }
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=DISPATCH_BEGIN idx=%zu rank=%zu backend=%s nodes=%d\n",
                        i, j, ggml_backend_name(bcj.backend), simple_graph->n_nodes);
            }
            dispatch_start[j] = ggml_galaxy_profile::clock::now();
            // REPLACE executes every predecessor node (including the
            // persistent KV-cache writes) but deliberately omits the
            // original TP FLASH_ATTN_EXT kernel. The output will be
            // supplied by native PC+Phone SP Attention before any
            // downstream kqv_out reshape / W_o graph can execute.
            //
            // Use a shallow graph copy, never mutate the stored Meta
            // cgraph. Force uid=0: RPC graph reuse MUST hash the actual
            // trimmed node sequence, not use the old complete graph UID.
            ggml_cgraph without_flash = *simple_graph;
            ggml_cgraph without_tp = *simple_graph;
            ggml_cgraph * run_graph = simple_graph;
            if (is_bypass_producer) {
                // The exact MatMul, Reshape and optional RoPE were proven
                // to be the *only* nodes in this producer subgraph. Their
                // physical output is populated from native SP bytes below.
                without_tp.n_nodes=0;
                without_tp.uid=0;
                run_graph=&without_tp;
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_TP_SKIP] layer=%d rank=%zu"
                        " tensor=%s skipped_nodes=%d"
                        " native_sp_projected=1 status=PASS\n",
                        producer_layer,j,original_last->name,simple_graph->n_nodes);
                std::fflush(stderr);
            }
            if (is_replace_raw_boundary) {
                ggml_tensor * source_last =
                        ggml_backend_meta_buffer_simple_tensor(original_last,j);
                if (simple_graph->n_nodes<1 ||
                        simple_graph->nodes[simple_graph->n_nodes-1]!=source_last ||
                        source_last->op!=GGML_OP_FLASH_ATTN_EXT) {
                    GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] rank=%zu final kernel identity mismatch\n",j);
                    return GGML_STATUS_FAILED;
                }
                for(int k=0;k<simple_graph->n_nodes-1;++k) {
                    if (simple_graph->nodes[k]->op==GGML_OP_FLASH_ATTN_EXT) {
                        GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] unexpected preceding Flash op\n");
                        return GGML_STATUS_FAILED;
                    }
                }
                --without_flash.n_nodes;
                without_flash.uid=0;
                run_graph=&without_flash;
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_REPLACE_DISPATCH] layer=%d rank=%zu"
                        " subgraph=%zu nodes_total=%d nodes_run=%d"
                        " skipped_node=%s skipped_op=%s"
                        " tp_flash_skipped=1 status=PASS\n",
                        raw_layer,j,i,simple_graph->n_nodes,without_flash.n_nodes,
                        original_last->name,ggml_op_name(source_last->op));
                std::fflush(stderr);
            }
            const ggml_status status=run_graph->n_nodes>0 ?
                    ggml_backend_graph_compute_async(bcj.backend,run_graph) :
                    GGML_STATUS_SUCCESS;
            if (profiling) {
                dispatch_ms[j] = ggml_galaxy_profile::ms(dispatch_start[j],
                        ggml_galaxy_profile::clock::now());
            }
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=DISPATCH_RETURN idx=%zu rank=%zu status=%d\n",
                        i, j, int(status));
            }
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        if (profiling) {
            // Synchronize AFTER dispatching both ranks. CUDA compute can
            // overlap RPC processing, but RPC graph_compute itself blocks
            // for server completion and RPC synchronize is a no-op. Thus
            // rank-1 dispatch_ms includes the entire RPC round trip, and
            // rank-0 observed_ms is a launch-to-fence window, NOT exclusive
            // GPU kernel time. The extra CUDA fence changes the schedule.
            for (size_t j = 0; j < n_backends; ++j) {
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_cgraph * simple = bcj.cgraphs[i].cgraph_main;
                if (simple->n_nodes == 0) continue;
                const auto sync_start = ggml_galaxy_profile::clock::now();
                ggml_backend_synchronize(bcj.backend);
                const auto sync_end = ggml_galaxy_profile::clock::now();
                ggml_galaxy_profile::compute(graph_id, i, j,
                        ggml_backend_name(bcj.backend), original_last->name,
                        simple->n_nodes, dispatch_ms[j],
                        ggml_galaxy_profile::ms(sync_start, sync_end),
                        ggml_galaxy_profile::ms(dispatch_start[j], sync_end));
            }
        }

        if (is_bypass_producer) {
            try {
                auto & bridge=native_tp_cache_bridges.at(producer_layer);
                const auto & local_sp=producer_kind=='q' ? bridge.sp.q :
                                      producer_kind=='k' ? bridge.sp.k :
                                                           bridge.sp.v;
                const auto & heads=producer_kind=='q' ? bridge.geometry.q.heads :
                                    bridge.geometry.k.heads;
                const auto repacked=ggml_galaxy_qkv_sp::token_to_head_f32(
                        local_sp,bridge.tokens,bridge.geometry.head_dim,heads);
                for(int rank=0;rank<2;++rank) {
                    ggml_tensor * output=ggml_backend_meta_buffer_simple_tensor(original_last,rank);
                    if (!output || !output->buffer || output->type!=GGML_TYPE_F32 ||
                            !ggml_is_contiguous(output) ||
                            ggml_nbytes(output)!=repacked[rank].size()) {
                        throw std::invalid_argument("SP-to-TP cache producer physical tensor differs in shape/type");
                    }
                    ggml_backend_tensor_set(output,repacked[rank].data(),0,repacked[rank].size());
                    ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                    // Verify the *actual backend* destination, not only
                    // the host reshuffle result, before original cache
                    // SET_ROWS consumes it in the next subgraph.
                    uint32_t observed=0,expected=0;
                    ggml_backend_tensor_get(output,&observed,0,sizeof(observed));
                    std::memcpy(&expected,repacked[rank].data(),sizeof(expected));
                    if(observed!=expected)
                        throw std::invalid_argument("SP QKV TP-head output readback mismatch");
                }
                bridge.injected_mask|=producer_kind=='q'?1u:
                                      producer_kind=='v'?2u:4u;
                bypass_actual_tp_nodes_skipped+=i_stop-i_start;
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_TP_INJECT] layer=%d tensor=%s"
                        " mode=SP_TOKEN_TO_TP_HEAD ranks=2"
                        " cache_producer=ORIGINAL_LLAMA bytes_pc=%zu"
                        " bytes_phone=%zu identity=F32_BIT_EXACT status=PASS\n",
                        producer_layer,original_last->name,
                        repacked[0].size(),repacked[1].size());
                std::fflush(stderr);
            } catch(const std::exception & e) {
                std::fprintf(stderr,
                        "[GALAXY_SP_QKV_TP_INJECT] layer=%d tensor=%s"
                        " status=FAIL reason=%s\n",
                        producer_layer,original_last->name,e.what());
                return GGML_STATUS_FAILED;
            }
        }

        if (no_tp_qkv_requested && is_replace_raw_boundary &&
                native_qkv_plan.contains(raw_layer)) {
            // Backend dispatch returned successfully for BOTH original
            // SET_ROWS writers. Fence them before consuming any new
            // Attention output or advancing to another Transformer layer.
            for (int rank=0;rank<2;++rank)
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
            bypass_actual_kv_store_ops+=2;
            std::fprintf(stderr,
                    "[GALAXY_SP_QKV_CACHE_STORES] layer=%d"
                    " source=SP_KV_TO_TP_HEAD cache_k_stores=1 cache_v_stores=1"
                    " cache_indices=ORIGINAL_LLAMA phase=EXECUTED status=PASS\n",
                    raw_layer);
            std::fflush(stderr);
        }

        // In REPLACE, the original TP Flash node was intentionally
        // removed from the submitted graph. Its exact output tensor must
        // now be filled with the native SP result, *before* the next
        // reshape and W_o nodes can run. The original KV-cache store
        // predecessor nodes have already been dispatched normally.
        if (is_replace_raw_boundary) {
            try {
                auto work=build_live_request(raw_layer,nullptr);
                if (work.tokens!=original_last->ne[2] ||
                        work.head_dim!=original_last->ne[0] ||
                        work.q.heads[0]+work.q.heads[1]!=original_last->ne[1]) {
                    throw std::invalid_argument(
                            "native SP replacement does not match Flash raw tensor shape");
                }
                const ggml_galaxy_collectives::devices_t devs={
                        backend_ctx->backend_configs[0].backend,
                        backend_ctx->backend_configs[1].backend
                };
                std::array<std::vector<uint8_t>,2> outputs;
                ggml_galaxy_qkv_sp::projected local_sp;
                double sp_projection_ms = 0.0;
                size_t native_qkv_weight_bytes=0;
                const bool native_selected = native_qkv_plan.contains(raw_layer);
                if (native_selected) {
                    if (no_tp_qkv_requested) {
                        auto & bridge=native_tp_cache_bridges.at(raw_layer);
                        if (!bridge.prepared || bridge.consumed ||
                                bridge.injected_mask!=7u || bridge.tokens!=work.tokens)
                            throw std::invalid_argument("native SP early QKV not ready for cache/Attention");
                        work.sp_qkv=&bridge.sp;
                        sp_projection_ms=bridge.projection_wall_ms;
                        // Measured at ATTN_NORM rather than at Flash boundary.
                        // Full matrices are already model resident; weights
                        // are not read back or uploaded during Prefill.
                    } else {
                        const auto started=ggml_galaxy_profile::clock::now();
                        const auto sp_source=build_native_sp_qkv(raw_layer,work);
                        native_qkv_weight_bytes=
                                ggml_row_size(sp_source.wq.type,sp_source.wq.in)*size_t(sp_source.wq.out)+
                                ggml_row_size(sp_source.wk.type,sp_source.wk.in)*size_t(sp_source.wk.out)+
                                ggml_row_size(sp_source.wv.type,sp_source.wv.in)*size_t(sp_source.wv.out);
                        local_sp=ggml_galaxy_qkv_sp::run(devs,sp_source);
                        work.sp_qkv=&local_sp;
                        sp_projection_ms=ggml_galaxy_profile::ms(started,
                                ggml_galaxy_profile::clock::now());
                    }
                }
                const auto measured=ggml_galaxy_causal_live::replace(devs,work,outputs);
                if (native_selected) {
                    if (!measured.direct_sp_qkv) {
                        throw std::invalid_argument(
                                "SP QKV was not consumed directly by native Flash");
                    }
                    std::fprintf(stderr,
                            "[GALAXY_SP_QKV] layer=%d mode=SP_NATIVE"
                            " producer=FULL_WEIGHT_LOCAL_TOKENS local_tokens=%" PRId64 "+%" PRId64
                            " projection_wall_ms=%.3f"
                            " tp_projection_kept_for_kv=%d direct_native_flash=1"
                            " resident_weights=%d runtime_weight_read_bytes=%zu"
                            " runtime_weight_upload_bytes=%zu status=PASS\n",
                            raw_layer,(work.tokens+1)/2,work.tokens/2,sp_projection_ms,
                            int(!no_tp_qkv_requested),int(resident_qkv_requested),
                            resident_qkv_requested ? size_t(0) : native_qkv_weight_bytes,
                            resident_qkv_requested ? size_t(0) : 2*native_qkv_weight_bytes);
                    std::fflush(stderr);
                }
                for (int rank=0;rank<2;++rank) {
                    const ggml_tensor * local=
                            ggml_backend_meta_buffer_simple_tensor(original_last,rank);
                    if (!local || !local->buffer ||
                            local->op!=GGML_OP_FLASH_ATTN_EXT ||
                            !ggml_is_contiguous(local) ||
                            local->type!=GGML_TYPE_F32 ||
                            local->ne[0]!=work.head_dim ||
                            local->ne[1]!=work.q.heads[rank] ||
                            local->ne[2]!=work.tokens || local->ne[3]!=1 ||
                            outputs[rank].size()!=ggml_nbytes(local)) {
                        throw std::invalid_argument(
                                "native SP TP shard does not match destination Flash buffer");
                    }
                    // The raw Flash output and following 2D kqv_out
                    // share the production GGML allocation. Do not
                    // modify the symbolic graph or allocate fake W_o
                    // input nodes: set bytes on the ACTUAL backend buffer.
                    ggml_backend_tensor_set(const_cast<ggml_tensor *>(local),
                            outputs[rank].data(),0,outputs[rank].size());
                    ggml_backend_synchronize(devs[rank]);
                    uint32_t observed=0,expected=0;
                    ggml_backend_tensor_get(local,&observed,0,sizeof(observed));
                    std::memcpy(&expected,outputs[rank].data(),sizeof(expected));
                    if(observed!=expected) {
                        throw std::invalid_argument(
                                "native SP raw Flash destination readback mismatch");
                    }
                }
                auto & state=live_states.at(raw_layer);
                state.native_sp_qkv=native_selected;
                if (no_tp_qkv_requested && native_selected)
                    native_tp_cache_bridges.at(raw_layer).consumed=true;
                state.tp_heads=std::move(outputs);
                state.pending=true;
                state.tokens=work.tokens;
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_LIVE] layer=%d tokens=%" PRId64
                        " mode=REPLACE tp_heads=%" PRId64 "+%" PRId64
                        " kv_heads=%" PRId64 "+%" PRId64
                        " first_token_rmse=%.9g"
                        " tp_sp_ms=%.3f kv_gather_ms=%.3f"
                        " native_ready_wall_ms=%.3f sp_tp_ms=%.3f"
                        " tp_flash_skipped=1 kv_store_preserved=1"
                        " injected_into=FATTN_RAW"
                        " reference=SEPARATE_BASELINE_REQUIRED"
                        " status=INJECTED\n",
                        raw_layer,work.tokens,
                        work.q.heads[0],work.q.heads[1],
                        work.k.heads[0],work.k.heads[1],
                        measured.first_token_rmse,
                        measured.head_to_token_ms,measured.kv_gather_ms,
                        measured.native_ready_wall_ms,measured.token_to_head_ms);
                std::fflush(stderr);
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_LIVE] layer=%d mode=REPLACE"
                        " status=FAIL reason=%s\n",
                        raw_layer,e.what());
                std::fflush(stderr);
                return GGML_STATUS_FAILED;
            }
        }

        // Capture the producing node NOW, immediately after both backend
        // submissions have completed. The normal end-of-model buffer may
        // be reused by later graph nodes, especially for V and reshape
        // views; never read it late to create a numerical reference.
        if (is_qkv_probe_node(original_last)) {
            if (n_backends != 2 || !ggml_backend_buffer_is_meta(original_last->buffer)) {
                GGML_LOG_ERROR("[GALAXY_QKV_CAPTURE] expected a two-backend Meta tensor\n");
                return GGML_STATUS_FAILED;
            }
            std::vector<std::vector<uint8_t>> rank_copies;
            rank_copies.reserve(n_backends);
            for (size_t rank = 0; rank < n_backends; ++rank) {
                const ggml_tensor * local =
                        ggml_backend_meta_buffer_simple_tensor(original_last, rank);
                if (local == nullptr || local->buffer == nullptr ||
                        ggml_nbytes(local) == 0) {
                    GGML_LOG_ERROR("[GALAXY_QKV_CAPTURE] name=%s rank=%zu missing source buffer\n",
                            original_last->name, rank);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                rank_copies.emplace_back(ggml_nbytes(local));
                std::vector<uint8_t> & bytes = rank_copies.back();
                ggml_backend_tensor_get(local, bytes.data(), 0, bytes.size());
                uint64_t hash = UINT64_C(14695981039346656037);
                for (uint8_t byte : bytes) {
                    hash ^= static_cast<uint64_t>(byte);
                    hash *= UINT64_C(1099511628211);
                }
                std::fprintf(stderr,
                        "[GALAXY_QKV_CAPTURE] name=%s rank=%zu subgraph=%zu"
                        " backend=%s bytes=%zu fnv1a64=%016" PRIx64
                        " phase=PRODUCER_DONE status=PASS\n",
                        original_last->name, rank, i,
                        ggml_backend_name(backend_ctx->backend_configs[rank].backend),
                        bytes.size(), hash);
            }
            std::fflush(stderr);
            if (!qkv_live_snapshots.emplace(original_last, std::move(rank_copies)).second) {
                GGML_LOG_ERROR("[GALAXY_QKV_CAPTURE] name=%s duplicate producer\n",
                        original_last->name);
                return GGML_STATUS_FAILED;
            }

            const int norm_layer = no_tp_qkv_requested ?
                    selected_layer(original_last->name,"attn_norm") : -1;
            if (norm_layer>=0 && original_last->ne[1]>=8 &&
                    native_qkv_plan.contains(norm_layer)) {
                try {
                    const auto geometry=build_bypass_geometry(norm_layer);
                    auto & bridge=native_tp_cache_bridges[norm_layer];
                    if(bridge.prepared || bridge.tokens!=0)
                        throw std::invalid_argument("duplicate native SP QKV projection at normalized producer");
                    const ggml_galaxy_collectives::devices_t devices={
                            backend_ctx->backend_configs[0].backend,
                            backend_ctx->backend_configs[1].backend
                    };
                    const auto started=ggml_galaxy_profile::clock::now();
                    const auto source=build_native_sp_qkv(norm_layer,geometry);
                    bridge.sp=ggml_galaxy_qkv_sp::run(devices,source);
                    bridge.projection_wall_ms=ggml_galaxy_profile::ms(started,
                            ggml_galaxy_profile::clock::now());
                    bridge.tokens=geometry.tokens;
                    bridge.geometry=geometry;
                    bridge.prepared=true;
                    std::fprintf(stderr,
                            "[GALAXY_SP_QKV_EARLY] layer=%d"
                            " boundary=ATTN_NORM token_shards=%" PRId64 "+%" PRId64
                            " full_qkv_weights=RESIDENT"
                            " projection_wall_ms=%.3f status=PASS\n",
                            norm_layer,(geometry.tokens+1)/2,geometry.tokens/2,
                            bridge.projection_wall_ms);
                    std::fflush(stderr);
                } catch (const std::exception & e) {
                    std::fprintf(stderr,
                            "[GALAXY_SP_QKV_EARLY] layer=%d status=FAIL reason=%s\n",
                            norm_layer,e.what());
                    return GGML_STATUS_FAILED;
                }
            }
        }

        // Boundary names are non-unique inside llama-graph; only the
        // pinned, producer-captured terminal is eligible. Each selected
        // Layer N owns a distinct pending/output/verified state.
        const int kqv_layer = live_shadow_requested ?
                selected_layer(original_last->name,"kqv_out") : -1;
        if (live_replace_requested && kqv_layer>=0 && original_last->ne[1]>=8) {
            auto it=live_states.find(kqv_layer);
            if(it==live_states.end() || !it->second.pending ||
                    it->second.verified ||
                    original_last->op!=GGML_OP_RESHAPE ||
                    original_last->type!=GGML_TYPE_F32 ||
                    !ggml_is_contiguous(original_last) ||
                    ggml_backend_meta_get_split_state(original_last,false).axis!=
                            GGML_BACKEND_SPLIT_AXIS_0) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] layer=%d missing matching pending Flash output\n",
                        kqv_layer);
                return GGML_STATUS_FAILED;
            }
            auto & state=it->second;
            if(state.tokens!=original_last->ne[1]) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] layer=%d token count changed between Flash and Wo\n",
                        kqv_layer);
                return GGML_STATUS_FAILED;
            }
            const auto captured=qkv_live_snapshots.find(original_last);
            if(captured==qkv_live_snapshots.end() || captured->second.size()!=2) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_REPLACE] layer=%d missing reshaped producer snapshot\n",
                        kqv_layer);
                return GGML_STATUS_FAILED;
            }
            for(int rank=0;rank<2;++rank) {
                if(captured->second[rank].size()!=state.tp_heads[rank].size() ||
                        std::memcmp(captured->second[rank].data(),
                                    state.tp_heads[rank].data(),
                                    state.tp_heads[rank].size())!=0) {
                    std::fprintf(stderr,
                            "[GALAXY_CAUSAL_REPLACE_VERIFY] layer=%d rank=%d"
                            " pre_wo_identity=FAIL status=FAIL\n",
                            kqv_layer,rank);
                    return GGML_STATUS_FAILED;
                }
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_REPLACE_VERIFY] layer=%d rank=%d"
                        " bytes=%zu pre_wo_identity=BYTE_EXACT status=PASS\n",
                        kqv_layer,rank,state.tp_heads[rank].size());
            }
            state.pending=false;
            state.verified=true;
            if (no_tp_qkv_requested && state.native_sp_qkv) {
                const auto b=native_tp_cache_bridges.find(kqv_layer);
                if (b==native_tp_cache_bridges.end() || !b->second.consumed ||
                        !b->second.prepared || b->second.injected_mask!=7u ||
                        !bypass_tp_cache_completed.insert(kqv_layer).second) {
                    GGML_LOG_ERROR("[GALAXY_SP_QKV_TP_BYPASS] layer=%d missing complete cache bridge\n",
                            kqv_layer);
                    return GGML_STATUS_FAILED;
                }
                // Release all host QKV F32 shards after the existing
                // SP Attention output is verified at the real W_o input.
                native_tp_cache_bridges.erase(b);
            }
            if (state.native_sp_qkv) {
                if (!native_qkv_plan.contains(kqv_layer) ||
                        !native_qkv_completed.insert(kqv_layer).second) {
                    GGML_LOG_ERROR("[GALAXY_SP_QKV] layer=%d unexpected/duplicate verification\n",
                            kqv_layer);
                    return GGML_STATUS_FAILED;
                }
            } else if (native_qkv_plan.contains(kqv_layer)) {
                GGML_LOG_ERROR("[GALAXY_SP_QKV] layer=%d selected but no SP producer\n",
                        kqv_layer);
                return GGML_STATUS_FAILED;
            }
            for(auto & bytes:state.tp_heads) bytes.clear();
            std::fprintf(stderr,
                    "[GALAXY_CAUSAL_LIVE] layer=%d tokens=%" PRId64
                    " mode=REPLACE tp_flash_skipped=1"
                    " production_wo_input=SP_ATTENTION"
                    " pre_wo_identity=BYTE_EXACT status=PASS\n",
                    kqv_layer,original_last->ne[1]);
            std::fflush(stderr);
        }

        // SHADOW runs original model Attention, reconstructs identical
        // per-layer native SP output, and leaves the production W_o input
        // untouched. Never mix snapshots from separate model layers.
        if (live_shadow_requested && !live_replace_requested && kqv_layer>=0) {
            try {
                const int64_t tokens=original_last->ne[1];
                if(tokens<8) {
                    std::fprintf(stderr,
                            "[GALAXY_CAUSAL_LIVE] layer=%d tokens=%" PRId64
                            " mode=SHADOW status=SKIP reason=WARMUP_BELOW_8\n",
                            kqv_layer,tokens);
                    std::fflush(stderr);
                } else {
                    if(live_shadow_verified.count(kqv_layer)) {
                        throw std::invalid_argument("duplicate in-graph layer Shadow verification");
                    }
                    for(const int preceding:live_plan.layers) {
                        if(preceding>=kqv_layer) break;
                        if(!live_shadow_verified.count(preceding)) {
                            throw std::invalid_argument(
                                    "previous selected layer has no Shadow verification");
                        }
                    }
                    const auto work=build_live_request(kqv_layer,original_last);
                    const ggml_galaxy_collectives::devices_t devs={
                            backend_ctx->backend_configs[0].backend,
                            backend_ctx->backend_configs[1].backend
                    };
                    const auto measured=ggml_galaxy_causal_live::shadow(devs,work);
                    live_shadow_verified.insert(kqv_layer);
                    std::fprintf(stderr,
                            "[GALAXY_CAUSAL_LIVE] layer=%d tokens=%" PRId64
                            " mode=SHADOW tp_heads=%" PRId64 "+%" PRId64
                            " kv_heads=%" PRId64 "+%" PRId64
                            " first_token_rmse=%.9g rmse=%.9g max_abs=%.9g"
                            " cosine=%.9g"
                            " tp_sp_ms=%.3f kv_gather_ms=%.3f"
                            " native_ready_wall_ms=%.3f sp_tp_ms=%.3f"
                            " threshold=%.3g production_output_unchanged=1"
                            " status=PASS\n",
                            kqv_layer,tokens,
                            work.q.heads[0],work.q.heads[1],
                            work.k.heads[0],work.k.heads[1],
                            measured.first_token_rmse,measured.rmse,
                            measured.max_abs,measured.cosine,
                            measured.head_to_token_ms,measured.kv_gather_ms,
                            measured.native_ready_wall_ms,measured.token_to_head_ms,
                            work.max_rmse);
                    std::fflush(stderr);
                }
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_LIVE] layer=%d mode=SHADOW"
                        " status=FAIL reason=%s\n",
                        kqv_layer,e.what());
                std::fflush(stderr);
                return GGML_STATUS_FAILED;
            }
        }

        // Once a selected layer's kqv_out has been fully consumed,
        // its producer-time Q/K/V and Attention host snapshots are no
        // longer needed by the NEXT layer. This is important for 16/28
        // layer experiments: otherwise every layer's multi-megabyte
        // F32 snapshots remain live until the whole Meta graph ends.
        //
        // Explicit GLG1 probe export is the exception: keep snapshots
        // when requested, because that diagnostic writes at graph end.
        if (live_shadow_requested && kqv_layer >= 0 &&
                original_last->ne[1] >= 8 && !qkv_probe_enabled) {
            const auto state = live_states.find(kqv_layer);
            const bool verified = live_replace_requested ?
                    (state != live_states.end() && state->second.verified) :
                    live_shadow_verified.count(kqv_layer) != 0;
            if (verified) {
                size_t released_bytes=0, released_tensors=0;
                auto it=qkv_live_snapshots.begin();
                while(it != qkv_live_snapshots.end()) {
                    const ggml_tensor * captured=it->first;
                    const bool belongs_to_layer =
                            selected_layer(captured->name,"attn_norm")==kqv_layer ||
                            selected_layer(captured->name,"Qcur")==kqv_layer ||
                            selected_layer(captured->name,"Kcur")==kqv_layer ||
                            selected_layer(captured->name,"Vcur")==kqv_layer ||
                            selected_layer(captured->name,"fattn_raw")==kqv_layer ||
                            selected_layer(captured->name,"kqv_out")==kqv_layer;
                    if (!belongs_to_layer) {
                        ++it;
                        continue;
                    }
                    for (const auto & copy : it->second) released_bytes+=copy.size();
                    ++released_tensors;
                    it=qkv_live_snapshots.erase(it);
                }
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_CAPTURE_RELEASE] layer=%d"
                        " host_bytes=%zu tensors=%zu"
                        " deferred_probe_export=0 status=PASS\n",
                        kqv_layer,released_bytes,released_tensors);
                std::fflush(stderr);
            }
        }

        const galaxy_sp_transition transition = galaxy_sp_get_transition(original_last);
        if (transition != galaxy_sp_transition::none) {
            if (original_last->src[0] == nullptr ||
                    !ggml_backend_buffer_is_meta(original_last->buffer) ||
                    !ggml_backend_buffer_is_meta(original_last->src[0]->buffer)) {
                GGML_LOG_ERROR("[GALAXY_SP] marker requires Meta source/destination: %s\n", original_last->name);
                return GGML_STATUS_FAILED;
            }
            const auto input_state = ggml_backend_meta_get_split_state(original_last->src[0], false);
            const bool transition_valid =
                    (transition == galaxy_sp_transition::reduce_scatter &&
                         input_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) ||
                    (transition == galaxy_sp_transition::scatter_mirrored &&
                         input_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) ||
                    (transition == galaxy_sp_transition::all_gather &&
                         input_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
            if (!transition_valid) {
                GGML_LOG_ERROR("[GALAXY_SP] unexpected source layout for %s: %s\n",
                        original_last->name, ggml_backend_meta_split_axis_name(input_state.axis));
                return GGML_STATUS_FAILED;
            }
            // src[0] is the source activation, the marker itself is the
            // destination. Both are Meta tensors with distinct per-rank
            // backend allocations. Synchronization is inside the primitive.
            const ggml_galaxy_collectives::devices_t devs = {
                backend_ctx->backend_configs[0].backend,
                backend_ctx->backend_configs[1].backend,
            };
            const ggml_galaxy_collectives::input_t in = {
                ggml_backend_meta_buffer_simple_tensor(original_last->src[0], 0),
                ggml_backend_meta_buffer_simple_tensor(original_last->src[0], 1),
            };
            const ggml_galaxy_collectives::output_t out = {
                ggml_backend_meta_buffer_simple_tensor(original_last, 0),
                ggml_backend_meta_buffer_simple_tensor(original_last, 1),
            };
            const ggml_galaxy_collectives::counts_t counts = transition == galaxy_sp_transition::all_gather
                    ? ggml_galaxy_collectives::counts_t{in[0]->ne[1], in[1]->ne[1]}
                    : ggml_galaxy_collectives::counts_t{out[0]->ne[1], out[1]->ne[1]};
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=COLLECTIVE_BEGIN idx=%zu kind=%d tensor=%s tokens=%" PRId64 "+%" PRId64 "\n",
                        i, int(transition), original_last->name, counts[0], counts[1]);
            }
            try {
                const size_t tile_tokens = ggml_galaxy_collectives::configured_tile_tokens();
                switch (transition) {
                    case galaxy_sp_transition::reduce_scatter:
                        ggml_galaxy_collectives::reduce_scatter(devs, in, out, counts, tile_tokens,
                                ggml_galaxy_collectives::configured_rs_fused_read());
                        galaxy_rs_count++;
                        break;
                    case galaxy_sp_transition::scatter_mirrored:
                        ggml_galaxy_collectives::scatter_mirrored(devs, in, out, counts, tile_tokens);
                        galaxy_scatter_count++;
                        break;
                    case galaxy_sp_transition::all_gather:
                        ggml_galaxy_collectives::all_gather(devs, in, out, counts, tile_tokens);
                        galaxy_ag_count++;
                        break;
                    case galaxy_sp_transition::none:
                        break;
                }
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("[GALAXY_SP] %s failed: %s\n", original_last->name, e.what());
                return GGML_STATUS_FAILED;
            }
            if (live_hybrid_ffn_requested) {
                // The callback can rename GALAXY_AG_FFN_N to ffn_out-N or
                // l_out-N. Stage and layer are encoded in stable op_params.
                const auto stage = ggml_galaxy::get_stage(original_last);
                const int layer = ggml_galaxy::get_stage_layer(original_last);
                std::set<int> * executed = nullptr;
                switch (stage) {
                    case ggml_galaxy::sp_stage::attention_rs:     executed = &hybrid_attention_rs;       break;
                    case ggml_galaxy::sp_stage::residual_scatter: executed = &hybrid_residual_scatter;   break;
                    case ggml_galaxy::sp_stage::ffn_input_ag:    executed = &hybrid_attn_ag;           break;
                    case ggml_galaxy::sp_stage::ffn_output_rs:   executed = &hybrid_ffn_rs;            break;
                    case ggml_galaxy::sp_stage::ffn_output_ag:   executed = &hybrid_ffn_ag;            break;
                    default: break;
                }
                if (!executed || layer < 0 || !executed->insert(layer).second) {
                    GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN] status=FAIL"
                            " reason=INVALID_OR_DUPLICATE_STAGE marker=%s stage=%d layer=%d\n",
                            original_last->name, int(stage), layer);
                    return GGML_STATUS_FAILED;
                }
            }
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=COLLECTIVE_END idx=%zu kind=%d tensor=%s\n",
                        i, int(transition), original_last->name);
            }
            continue; // marker has already finished its collective
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            // Q/K/V and Flash Attention tensor-parallel shards are already
            // owned by their rank (axis 0/1), not PARTIAL contributions.
            // Forced diagnostic cuts must not invoke a spurious AllReduce.
            if (is_qkv_probe_node(original_last)) {
                const auto probe_state = ggml_backend_meta_get_split_state(original_last, false);
                if (probe_state.axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    continue;
                }
            }
            if (has_galaxy_markers) {
                // Only an actual PARTIAL terminal needs AllReduce. When the
                // next node is GALAXY_RS, skip the old AllReduce entirely;
                // ReduceScatter will consume both unreduced rank outputs.
                const auto state = ggml_backend_meta_get_split_state(original_last, false);
                if (state.axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    continue;
                }
                if (i_stop < size_t(cgraph->n_nodes) &&
                        galaxy_sp_get_transition(cgraph->nodes[i_stop]) ==
                                galaxy_sp_transition::reduce_scatter) {
                    if (cgraph->nodes[i_stop]->src[0] != original_last) {
                        GGML_LOG_ERROR("[GALAXY_SP] a ReduceScatter marker must immediately consume the PARTIAL output\n");
                        return GGML_STATUS_FAILED;
                    }
                    continue;
                }
            }
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=ALLREDUCE_BEGIN idx=%zu tensor=%s\n",
                        i, original_last->name);
            }
            const auto reduce_start = ggml_galaxy_profile::clock::now();
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            galaxy_ar_count++;
            if (profiling) {
                // The fallback may queue asynchronous copies/ADD operations.
                // Wait for both ranks before recording the end-to-end time.
                for (size_t j = 0; j < n_backends; ++j) {
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }
                ggml_galaxy_profile::communication("ALLREDUCE", original_last->name,
                        original_last->ne[1], 0, 0, 0, 0, 0, 0,
                        ggml_galaxy_profile::ms(reduce_start, ggml_galaxy_profile::clock::now()));
            }
            if (trace) {
                GGML_LOG_INFO("[GALAXY_META_TRACE] event=ALLREDUCE_END idx=%zu tensor=%s native=%d\n",
                        i, original_last->name, int(backend_allreduce_success));
            }
        }
    }
    if (trace) {
        GGML_LOG_INFO("[GALAXY_META_TRACE] event=GRAPH_END subgraphs=%zu sp=%d\n",
                backend_ctx->n_subgraphs, int(has_galaxy_markers));
    }
    if (profiling) {
        const bool galaxy2 = std::getenv("LLAMA_GALAXY2") &&
                             std::strcmp(std::getenv("LLAMA_GALAXY2"), "1") == 0;
        const char * galaxy2_limit = std::getenv("LLAMA_GALAXY2_FFN_SP_LAYERS");
        const bool partial_galaxy2 = galaxy2 && galaxy2_limit != nullptr &&
                std::strcmp(galaxy2_limit, "all") != 0;
        ggml_galaxy_profile::summary(graph_id,
                has_galaxy_markers
                    ? (galaxy2 ? (partial_galaxy2 ? "GALAXY2_PARTIAL" : "GALAXY2_FULL") : "GALAXY")
                    : "TP",
                cgraph->n_nodes, backend_ctx->n_subgraphs,
                galaxy_rs_count, galaxy_scatter_count, galaxy_ag_count,
                galaxy_ar_count,
                ggml_galaxy_profile::ms(graph_start, ggml_galaxy_profile::clock::now()));
    }
    // Optional per-layer activation snapshots (selected output tensors only).
    // Read rank 0 after the full Meta graph has finished: TP PARTIAL
    // terminals have already gone through the normal AllReduce; SP
    // l_out markers have already been materialized by AllGather.
    // Identical GLG1/F32 framing to the existing logits comparator allows
    // the same numerical tool to compare hidden-state vectors.
    const char * dump_dir = std::getenv("LLAMA_GALAXY_LAYER_DUMP_DIR");
    if (dump_dir != nullptr && dump_dir[0] != '\0') {
        const std::filesystem::path folder(dump_dir);
        std::error_code directory_error;
        std::filesystem::create_directories(folder, directory_error);
        for (int node_id = 0; node_id < cgraph->n_nodes; ++node_id) {
            const ggml_tensor * node = cgraph->nodes[node_id];
            if (node == nullptr ||
                    std::strncmp(node->name, "l_out-", 6) != 0 ||
                    (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 ||
                    !ggml_backend_buffer_is_meta(node->buffer) ||
                    (node->type != GGML_TYPE_F16 && node->type != GGML_TYPE_F32) ||
                    !ggml_is_contiguous(node) || node->ne[1] <= 1) {
                continue;
            }
            const auto state = ggml_backend_meta_get_split_state(node, false);
            if (state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                    state.axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_LOG_ERROR("[GALAXY_LAYER_DUMP] name=%s unexpected_layout=%s\n",
                        node->name, ggml_backend_meta_split_axis_name(state.axis));
                continue;
            }
            const ggml_tensor * source = ggml_backend_meta_buffer_simple_tensor(node, 0);
            const int64_t elements = ggml_nelements(node);
            if (source == nullptr || source->buffer == nullptr ||
                    source->ne[0] != node->ne[0] || source->ne[1] != node->ne[1] ||
                    elements <= 0 || elements > std::numeric_limits<int32_t>::max() ||
                    ggml_nbytes(source) != ggml_nbytes(node)) {
                GGML_LOG_ERROR("[GALAXY_LAYER_DUMP] name=%s rank0_tensor_not_full\n", node->name);
                continue;
            }
            ggml_backend_synchronize(backend_ctx->backend_configs[0].backend);
            std::vector<uint8_t> raw(ggml_nbytes(source));
            ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
            std::vector<float> values(static_cast<size_t>(elements));
            if (source->type == GGML_TYPE_F32) {
                std::memcpy(values.data(), raw.data(), raw.size());
            } else {
                for (int64_t e = 0; e < elements; ++e) {
                    ggml_fp16_t half;
                    std::memcpy(&half, raw.data() + size_t(e)*sizeof(ggml_fp16_t), sizeof(half));
                    values[size_t(e)] = ggml_fp16_to_fp32(half);
                }
            }
            const std::filesystem::path outpath = folder / (std::string(node->name) + ".f32");
            std::ofstream out(outpath, std::ios::binary | std::ios::trunc);
            if (!out) {
                GGML_LOG_ERROR("[GALAXY_LAYER_DUMP] name=%s could_not_open_output\n", node->name);
                continue;
            }
            const int32_t n_values = int32_t(elements);
            out.write("GLG1", 4);
            out.write(reinterpret_cast<const char *>(&n_values), sizeof(n_values));
            out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            out.flush();
            std::fprintf(stderr, "[GALAXY_LAYER_DUMP] name=%s elements=%d tokens=%" PRId64 " status=%s\n",
                    node->name, n_values, node->ne[1], out.good() ? "PASS" : "FAIL");
        }
        std::fflush(stderr);
    }
    // Diagnostic-only intermediate activations for dense Llama. Unlike the
    // legacy final-layer snapshots, these may be sequence-sharded (AXIS_1).
    // Reconstruct the logical [hidden, tokens] matrix in rank/token order.
    // Every requested tensor has GGML_TENSOR_FLAG_OUTPUT set when the graph
    // is built, so its backing storage survives graph execution.
    const char * stage_dir = std::getenv("LLAMA_GALAXY_STAGE_DUMP_DIR");
    if (stage_dir != nullptr && stage_dir[0] != '\0') {
        const std::filesystem::path folder(stage_dir);
        std::error_code directory_error;
        std::filesystem::create_directories(folder, directory_error);
        if (directory_error) {
            GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] cannot create directory: %s\n", stage_dir);
            return GGML_STATUS_FAILED;
        }
        for (int node_id = 0; node_id < cgraph->n_nodes; ++node_id) {
            const ggml_tensor * node = cgraph->nodes[node_id];
            if (node == nullptr || (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 ||
                    !ggml_backend_buffer_is_meta(node->buffer)) {
                continue;
            }
            const bool selected = std::strncmp(node->name, "ffn_inp-", 8) == 0 ||
                    std::strncmp(node->name, "ffn_norm-", 9) == 0;
            if (!selected) {
                continue;
            }
            if ((node->type != GGML_TYPE_F16 && node->type != GGML_TYPE_F32) ||
                    !ggml_is_contiguous(node) || node->ne[0] <= 0 || node->ne[1] <= 0 ||
                    node->ne[2] != 1 || node->ne[3] != 1 ||
                    ggml_nelements(node) > std::numeric_limits<int32_t>::max()) {
                GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s invalid_type_or_shape\n", node->name);
                return GGML_STATUS_FAILED;
            }
            const auto state = ggml_backend_meta_get_split_state(node, false);
            if (state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                    state.axis != GGML_BACKEND_SPLIT_AXIS_1) {
                GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s unsupported_layout=%s\n",
                        node->name, ggml_backend_meta_split_axis_name(state.axis));
                return GGML_STATUS_FAILED;
            }
            const size_t element_bytes = node->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
            const size_t row_bytes = static_cast<size_t>(node->ne[0]) * element_bytes;
            const size_t expected_bytes = static_cast<size_t>(ggml_nelements(node)) * element_bytes;
            std::vector<uint8_t> raw(expected_bytes);
            const int n_sources = state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ? 1 : 2;
            int64_t copied_tokens = 0;
            for (int rank = 0; rank < n_sources; ++rank) {
                const ggml_tensor * source = ggml_backend_meta_buffer_simple_tensor(node, rank);
                if (source == nullptr || source->buffer == nullptr ||
                        source->type != node->type || !ggml_is_contiguous(source) ||
                        source->ne[0] != node->ne[0] || source->ne[1] <= 0 ||
                        source->ne[2] != 1 || source->ne[3] != 1 ||
                        source->ne[1] > node->ne[1] - copied_tokens ||
                        ggml_nbytes(source) != row_bytes * static_cast<size_t>(source->ne[1])) {
                    GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s rank=%d invalid_source_shape\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                ggml_backend_tensor_get(source, raw.data() + static_cast<size_t>(copied_tokens)*row_bytes,
                        0, ggml_nbytes(source));
                copied_tokens += source->ne[1];
            }
            if (copied_tokens != node->ne[1]) {
                GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s tokens=%" PRId64 " copied=%" PRId64 "\n",
                        node->name, node->ne[1], copied_tokens);
                return GGML_STATUS_FAILED;
            }
            const size_t elements = static_cast<size_t>(ggml_nelements(node));
            std::vector<float> values(elements);
            if (node->type == GGML_TYPE_F32) {
                std::memcpy(values.data(), raw.data(), raw.size());
            } else {
                for (size_t i = 0; i < elements; ++i) {
                    ggml_fp16_t half;
                    std::memcpy(&half, raw.data() + i*sizeof(ggml_fp16_t), sizeof(half));
                    values[i] = ggml_fp16_to_fp32(half);
                }
            }
            const std::filesystem::path path = folder / (std::string(node->name) + ".f32");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            const int32_t count = static_cast<int32_t>(elements);
            if (!out) {
                GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s open_failed\n", node->name);
                return GGML_STATUS_FAILED;
            }
            out.write("GLG1", 4);
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
            out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            out.flush();
            if (!out.good()) {
                GGML_LOG_ERROR("[GALAXY_STAGE_DUMP] name=%s write_failed\n", node->name);
                return GGML_STATUS_FAILED;
            }
            std::fprintf(stderr,
                    "[GALAXY_STAGE_DUMP] name=%s layout=%s hidden=%" PRId64 " tokens=%" PRId64 " status=PASS\n",
                    node->name, ggml_backend_meta_split_axis_name(state.axis),
                    node->ne[0], node->ne[1]);
        }
        std::fflush(stderr);
    }
    // Opt-in FFN Up/Gate/activated-output snapshots. The TP outputs are
    // sharded along feature axis 0, while the Galaxy2 FFN SP outputs are
    // sharded along token axis 1. Reconstruct the same logical matrix
    // [feature, token] before writing the GLG1/F32 binary snapshot.
    const char * ffn_op_dir = std::getenv("LLAMA_GALAXY_FFN_OP_DUMP_DIR");
    if (ffn_op_dir != nullptr && ffn_op_dir[0] != '\0') {
        if (n_backends != 2) {
            GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] requires exactly 2 backends\n");
            return GGML_STATUS_FAILED;
        }
        const std::filesystem::path folder(ffn_op_dir);
        std::error_code directory_error;
        std::filesystem::create_directories(folder, directory_error);
        if (directory_error) {
            GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] cannot create directory: %s\n", ffn_op_dir);
            return GGML_STATUS_FAILED;
        }
        for (int node_id = 0; node_id < cgraph->n_nodes; ++node_id) {
            const ggml_tensor * node = cgraph->nodes[node_id];
            if (node == nullptr || (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 ||
                    !ggml_backend_buffer_is_meta(node->buffer)) {
                continue;
            }
            const bool selected = std::strncmp(node->name, "ffn_up-", 7) == 0 ||
                    std::strncmp(node->name, "ffn_gate-", 9) == 0 ||
                    std::strncmp(node->name, "ffn_swiglu-", 11) == 0;
            if (!selected) {
                continue;
            }
            if ((node->type != GGML_TYPE_F16 && node->type != GGML_TYPE_F32) ||
                    !ggml_is_contiguous(node) || node->ne[0] <= 0 || node->ne[1] <= 0 ||
                    node->ne[2] != 1 || node->ne[3] != 1 ||
                    ggml_nelements(node) > std::numeric_limits<int32_t>::max()) {
                GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s invalid_type_or_shape\n", node->name);
                return GGML_STATUS_FAILED;
            }
            const auto state = ggml_backend_meta_get_split_state(node, false);
            if (state.axis != GGML_BACKEND_SPLIT_AXIS_0 &&
                    state.axis != GGML_BACKEND_SPLIT_AXIS_1) {
                GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s unexpected_layout=%s\n",
                        node->name, ggml_backend_meta_split_axis_name(state.axis));
                return GGML_STATUS_FAILED;
            }
            const size_t features = static_cast<size_t>(node->ne[0]);
            const size_t tokens = static_cast<size_t>(node->ne[1]);
            std::vector<float> values(features * tokens);
            size_t consumed = 0;
            for (int rank = 0; rank < 2; ++rank) {
                const ggml_tensor * source = ggml_backend_meta_buffer_simple_tensor(node, rank);
                if (source == nullptr || source->buffer == nullptr ||
                        source->type != node->type || !ggml_is_contiguous(source) ||
                        source->ne[0] <= 0 || source->ne[1] <= 0 ||
                        source->ne[2] != 1 || source->ne[3] != 1) {
                    GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s rank=%d invalid_source\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                const size_t src_features = static_cast<size_t>(source->ne[0]);
                const size_t src_tokens = static_cast<size_t>(source->ne[1]);
                const bool feature_shard = state.axis == GGML_BACKEND_SPLIT_AXIS_0;
                const bool shape_ok = feature_shard
                        ? src_tokens == tokens && src_features <= features - consumed
                        : src_features == features && src_tokens <= tokens - consumed;
                const size_t item_bytes = node->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
                if (!shape_ok || ggml_nbytes(source) != src_features * src_tokens * item_bytes) {
                    GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s rank=%d invalid_rank_shape\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_synchronize(backend_ctx->backend_configs[rank].backend);
                std::vector<uint8_t> raw(ggml_nbytes(source));
                ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
                const auto read_float = [&](size_t index) -> float {
                    if (node->type == GGML_TYPE_F32) {
                        float value;
                        std::memcpy(&value, raw.data() + index*sizeof(float), sizeof(value));
                        return value;
                    }
                    ggml_fp16_t half;
                    std::memcpy(&half, raw.data() + index*sizeof(ggml_fp16_t), sizeof(half));
                    return ggml_fp16_to_fp32(half);
                };
                for (size_t t = 0; t < src_tokens; ++t) {
                    for (size_t f = 0; f < src_features; ++f) {
                        const size_t dst_t = feature_shard ? t : consumed + t;
                        const size_t dst_f = feature_shard ? consumed + f : f;
                        values[dst_t*features + dst_f] = read_float(t*src_features + f);
                    }
                }
                consumed += feature_shard ? src_features : src_tokens;
            }
            if (consumed != (state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? features : tokens)) {
                GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s incomplete_shards=%zu\n",
                        node->name, consumed);
                return GGML_STATUS_FAILED;
            }
            const std::filesystem::path filepath = folder / (std::string(node->name) + ".f32");
            std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
            if (!out) {
                GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s open_failed\n", node->name);
                return GGML_STATUS_FAILED;
            }
            const int32_t count = static_cast<int32_t>(values.size());
            out.write("GLG1", 4);
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
            out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            out.flush();
            if (!out.good()) {
                GGML_LOG_ERROR("[GALAXY_FFN_OP_DUMP] name=%s write_failed\n", node->name);
                return GGML_STATUS_FAILED;
            }
            std::fprintf(stderr,
                    "[GALAXY_FFN_OP_DUMP] name=%s layout=%s features=%zu tokens=%zu status=PASS\n",
                    node->name, ggml_backend_meta_split_axis_name(state.axis),
                    features, tokens);
        }
        std::fflush(stderr);
    }
    // Opt-in real-GGUF Attention diagnostic. Reconstruct head-sharded Q/K/V
    // after RoPE (3D [head_dim, heads, tokens]) and the pre-Wo kqv_out
    // (2D [heads*head_dim, tokens]) as logical F32 [width, tokens].
    // The production Attention and cache path are untouched. Each token
    // count gets its own directory so warmup graphs cannot overwrite prefill.
    if (qkv_probe_enabled) {
        if (n_backends != 2) {
            GGML_LOG_ERROR("[GALAXY_REAL_QKV] requires exactly 2 backends\n");
            return GGML_STATUS_FAILED;
        }
        for (int node_id = 0; node_id < cgraph->n_nodes; ++node_id) {
            const ggml_tensor * node = cgraph->nodes[node_id];
            if (node == nullptr || (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 ||
                    !ggml_backend_buffer_is_meta(node->buffer)) {
                continue;
            }
            const bool is_q = std::strncmp(node->name, "Qcur-", 5) == 0;
            const bool is_k = std::strncmp(node->name, "Kcur-", 5) == 0;
            const bool is_v = std::strncmp(node->name, "Vcur-", 5) == 0;
            const bool is_out = std::strncmp(node->name, "kqv_out-", 8) == 0;
            const bool is_flash_raw = std::strncmp(node->name, "fattn_raw-", 10) == 0;
            if (!is_q && !is_k && !is_v && !is_out && !is_flash_raw) {
                continue;
            }
            const bool three_dimensional = !is_out;
            const int64_t width = three_dimensional
                    ? node->ne[0] * node->ne[1] : node->ne[0];
            const int64_t tokens = three_dimensional ? node->ne[2] : node->ne[1];
            if (tokens <= 1) {
                continue; // prefill-only, do not dump decode or one-token graphs
            }
            if ((node->type != GGML_TYPE_F16 && node->type != GGML_TYPE_F32) ||
                    !ggml_is_contiguous(node) || node->ne[0] <= 0 ||
                    node->ne[1] <= 0 || node->ne[3] != 1 ||
                    (three_dimensional ? node->ne[2] <= 0 : node->ne[2] != 1) ||
                    width <= 0 || width > std::numeric_limits<int32_t>::max()/tokens ||
                    ggml_nelements(node) != width*tokens) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s unsupported shape/type\n", node->name);
                return GGML_STATUS_FAILED;
            }
            const auto split = ggml_backend_meta_get_split_state(node, false);
            const auto expected_axis = three_dimensional
                    ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
            if (split.axis != expected_axis) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s axis=%s expected=%s\n",
                        node->name, ggml_backend_meta_split_axis_name(split.axis),
                        ggml_backend_meta_split_axis_name(expected_axis));
                return GGML_STATUS_FAILED;
            }
            const size_t bytes_per_item = node->type == GGML_TYPE_F32
                    ? sizeof(float) : sizeof(ggml_fp16_t);
            // These are host-owned copies read when each node finished,
            // NOT views of memory that might have been reused later.
            const auto captured = qkv_live_snapshots.find(node);
            if (captured == qkv_live_snapshots.end() ||
                    captured->second.size() != n_backends) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s no producer-time snapshot\n",
                        node->name);
                return GGML_STATUS_FAILED;
            }
            std::vector<float> values(static_cast<size_t>(width*tokens));
            int64_t done_features = 0;
            for (int rank = 0; rank < 2; ++rank) {
                const ggml_tensor * local = ggml_backend_meta_buffer_simple_tensor(node, rank);
                if (local == nullptr || local->buffer == nullptr ||
                        local->type != node->type || !ggml_is_contiguous(local)) {
                    GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s rank=%d unavailable\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                const int64_t local_width = three_dimensional
                        ? local->ne[0]*local->ne[1] : local->ne[0];
                const int64_t local_tokens = three_dimensional ? local->ne[2] : local->ne[1];
                if (local_width <= 0 || local_tokens != tokens ||
                        local_width > width - done_features ||
                        (three_dimensional && local->ne[0] != node->ne[0]) ||
                        local->ne[3] != 1 ||
                        (!three_dimensional && local->ne[2] != 1) ||
                        ggml_nbytes(local) != static_cast<size_t>(local_width*tokens)*bytes_per_item) {
                    GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s rank=%d invalid_shard\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                const std::vector<uint8_t> & raw = captured->second[rank];
                if (raw.size() != ggml_nbytes(local)) {
                    GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s rank=%d capture length mismatch\n",
                            node->name, rank);
                    return GGML_STATUS_FAILED;
                }
                if (is_out || is_flash_raw) {
                    // Hash the *actual rank-local source buffer* before
                    // reconstructing any global output. This separates
                    // duplicated device results from a faulty 2D exporter.
                    uint64_t hash = UINT64_C(14695981039346656037);
                    for (const uint8_t byte : raw) {
                        hash ^= static_cast<uint64_t>(byte);
                        hash *= UINT64_C(1099511628211);
                    }
                    // This is explicitly opted-in diagnostic evidence and
                    // must not disappear when the normal GGML INFO verbosity
                    // is filtered by the CLI. Always write to stderr.
                    std::fprintf(stderr,
                            "[GALAXY_REAL_QKV_RANK] name=%s op=%s rank=%d"
                            " backend=%s local_features=%" PRId64
                            " tokens=%" PRId64 " bytes=%zu fnv1a64=%016" PRIx64
                            " status=PRODUCER_SNAPSHOT\n",
                            node->name, ggml_op_name(node->op), rank,
                            ggml_backend_name(backend_ctx->backend_configs[rank].backend),
                            local_width, local_tokens, raw.size(), hash);
                    std::fflush(stderr);
                }
                for (int64_t t = 0; t < tokens; ++t) {
                    for (int64_t f = 0; f < local_width; ++f) {
                        const size_t src_index = static_cast<size_t>(t*local_width+f);
                        float value = 0.0f;
                        if (node->type == GGML_TYPE_F32) {
                            std::memcpy(&value, raw.data()+src_index*sizeof(float), sizeof(value));
                        } else {
                            ggml_fp16_t half;
                            std::memcpy(&half, raw.data()+src_index*sizeof(ggml_fp16_t), sizeof(half));
                            value = ggml_fp16_to_fp32(half);
                        }
                        values[static_cast<size_t>(t*width+done_features+f)] = value;
                    }
                }
                done_features += local_width;
            }
            if (done_features != width) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] name=%s incomplete_features=%" PRId64 "\n",
                        node->name, done_features);
                return GGML_STATUS_FAILED;
            }
            const std::filesystem::path folder = std::filesystem::path(qkv_probe_dir) /
                    ("tokens-" + std::to_string(tokens));
            std::error_code ec;
            std::filesystem::create_directories(folder, ec);
            if (ec) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] mkdir_failed=%s\n", folder.string().c_str());
                return GGML_STATUS_FAILED;
            }
            const std::filesystem::path filepath = folder / (std::string(node->name)+".f32");
            std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
            const int32_t count = static_cast<int32_t>(values.size());
            if (!out) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] open_failed=%s\n", filepath.string().c_str());
                return GGML_STATUS_FAILED;
            }
            out.write("GLG1", 4);
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
            out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            out.flush();
            if (!out.good()) {
                GGML_LOG_ERROR("[GALAXY_REAL_QKV] write_failed=%s\n", filepath.string().c_str());
                return GGML_STATUS_FAILED;
            }
            GGML_LOG_INFO("[GALAXY_REAL_QKV] name=%s width=%" PRId64 " tokens=%" PRId64
                    " axis=%s dtype=%s rank_features=%" PRId64 " status=PASS\n",
                    node->name, width, tokens,
                    ggml_backend_meta_split_axis_name(split.axis),
                    ggml_type_name(node->type), done_features);
        }
        if (!qkv_live_snapshots.empty()) {
            std::fprintf(stderr,
                    "[GALAXY_QKV_CAPTURE_SUM] tensors=%zu phase=PRODUCER_DONE"
                    " late_backend_reads=0 status=PASS\n",
                    qkv_live_snapshots.size());
            std::fflush(stderr);
        }
    }
    // Graph-invocation-wide prefix contract. 2-token warmup graphs may
    // legitimately complete zero selected layers; the 98-token Prefill
    // must complete ALL selected layers, in order, within the same graph.
    // The PowerShell smoke script independently requires the full prefix
    // witness, so a silent skipped layer can never pass.
    if (live_shadow_requested) {
        size_t completed=0;
        for(const int layer:live_plan.layers) {
            const auto found=live_states.find(layer);
            const bool pending=found!=live_states.end() && found->second.pending;
            if(pending) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_PREFIX_SUM] layer=%d has unconsumed Flash output status=FAIL\n",
                        layer);
                return GGML_STATUS_FAILED;
            }
            const bool verified=live_replace_requested ?
                    (found!=live_states.end() && found->second.verified) :
                    live_shadow_verified.count(layer)!=0;
            if(!verified) continue;
            ++completed;
            if(live_replace_requested) {
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_REPLACE_SUM] layer=%d"
                        " tp_flash_skipped=1 pre_wo_input=SP"
                        " tp_head_restore=BYTE_EXACT"
                        " mode=REPLACE status=PASS\n",layer);
            }
        }
        if(completed>0 && completed!=live_plan.layers.size()) {
            std::fprintf(stderr,
                    "[GALAXY_CAUSAL_PREFIX_SUM] mode=%s layers=%s"
                    " completed=%zu expected=%zu status=FAIL"
                    " reason=PARTIAL_PREFIX\n",
                    live_mode.c_str(),live_plan.description().c_str(),
                    completed,live_plan.layers.size());
            return GGML_STATUS_FAILED;
        }
        if(completed==live_plan.layers.size()) {
            std::fprintf(stderr,
                    "[GALAXY_CAUSAL_PREFIX_SUM] mode=%s layers=%s"
                    " completed=%zu expected=%zu"
                    " all_layers_verified=1 status=PASS\n",
                    live_mode.c_str(),live_plan.description().c_str(),
                    completed,live_plan.layers.size());
        }
        if(completed>0) std::fflush(stderr);
    }
    if (no_tp_qkv_requested && !bypass_tp_cache_completed.empty()) {
        const std::set<int> expected(native_qkv_plan.layers.begin(),native_qkv_plan.layers.end());
        if(bypass_tp_cache_completed!=expected ||
                bypass_actual_tp_nodes_skipped!=8*expected.size() ||
                bypass_actual_kv_store_ops!=2*expected.size()) {
            GGML_LOG_ERROR("[GALAXY_SP_QKV_TP_BYPASS_SUM] completed=%zu expected=%zu"
                    " status=FAIL reason=PRODUCER_OR_CACHE_COVERAGE\n",
                    bypass_tp_cache_completed.size(),expected.size());
            return GGML_STATUS_FAILED;
        }
        std::fprintf(stderr,
                "[GALAXY_SP_QKV_TP_BYPASS_SUM] layers=%s"
                " completed=%zu expected=%zu tp_qkv_nodes_skipped=%zu"
                " cache_kv_store_ops=%zu kv_source=SP_TOKEN_TO_TP_HEAD"
                " decode=TP_ORIGINAL status=PASS\n",
                native_qkv_plan.description().c_str(),
                expected.size(),expected.size(),
                bypass_actual_tp_nodes_skipped,bypass_actual_kv_store_ops);
        std::fflush(stderr);
    }
    if (native_sp_qkv_requested) {
        size_t attention_completed=0;
        for (const int layer: live_plan.layers) {
            const auto found=live_states.find(layer);
            if (found != live_states.end() && found->second.verified) ++attention_completed;
        }
        // A 2-token warmup may complete none; real Prefill must cover all
        // selected Attention and SP-QKV layers within one graph invocation.
        if (attention_completed == live_plan.layers.size()) {
            const std::set<int> expected(native_qkv_plan.layers.begin(),
                                         native_qkv_plan.layers.end());
            if (native_qkv_completed != expected) {
                GGML_LOG_ERROR("[GALAXY_SP_QKV_PREFIX_SUM] completed=%zu expected=%zu"
                        " status=FAIL reason=NATIVE_LAYER_COVERAGE\n",
                        native_qkv_completed.size(),expected.size());
                return GGML_STATUS_FAILED;
            }
            std::fprintf(stderr,
                    "[GALAXY_SP_QKV_PREFIX_SUM] mode=SP_NATIVE layers=%s"
                    " completed=%zu expected=%zu all_layers_verified=1"
                    " tp_cache_preserved=1 resident_weights=%d status=PASS\n",
                    native_qkv_plan.description().c_str(),
                    native_qkv_completed.size(),expected.size(),
                    int(resident_qkv_requested));
            std::fflush(stderr);
        }
    }
    if (live_hybrid_ffn_requested && !hybrid_attention_rs.empty()) {
        size_t attention_replaced = 0;
        for (const int layer : live_plan.layers) {
            const auto it = live_states.find(layer);
            if (it != live_states.end() && it->second.verified) ++attention_replaced;
        }
        // This 2-token warmup may carry FFN markers but has no native SP
        // Attention replacements; do not emit the prefill witness for it.
        if (attention_replaced != 0) {
            if (attention_replaced != live_plan.layers.size() ||
                    hybrid_attention_rs != hybrid_residual_scatter ||
                    hybrid_attention_rs != hybrid_ffn_ag ||
                    hybrid_attn_ag != hybrid_ffn_rs) {
                // Emit the actual executed layer sets to distinguish a
                // broken collective path from a bad stage identity.
                const auto list_layers = [](const std::set<int> & values) {
                    std::string result;
                    for (const int layer : values) {
                        if (!result.empty()) result += ',';
                        result += std::to_string(layer);
                    }
                    return result.empty() ? std::string("NONE") : result;
                };
                std::fprintf(stderr,
                        "[GALAXY_CAUSAL_HYBRID_FFN_COVERAGE]"
                        " replaced=%zu expected=%zu"
                        " attention_rs=%s residual_scatter=%s"
                        " ffn_output_ag=%s ffn_input_ag=%s ffn_output_rs=%s\n",
                        attention_replaced, live_plan.layers.size(),
                        list_layers(hybrid_attention_rs).c_str(),
                        list_layers(hybrid_residual_scatter).c_str(),
                        list_layers(hybrid_ffn_ag).c_str(),
                        list_layers(hybrid_attn_ag).c_str(),
                        list_layers(hybrid_ffn_rs).c_str());
                std::fflush(stderr);
                GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN_SUM] status=FAIL reason=MARKER_OR_REPLACE_COVERAGE\n");
                return GGML_STATUS_FAILED;
            }
            size_t sp_ffn = 0;
            for (const int layer : hybrid_attention_rs) {
                if (!hybrid_attn_ag.count(layer)) ++sp_ffn;
            }
            if (sp_ffn == 0) {
                GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN_SUM] status=FAIL reason=NO_SP_FFN_EXECUTED\n");
                return GGML_STATUS_FAILED;
            }
            const bool last_ffn_sp = hybrid_attention_rs.count(27) &&
                    hybrid_ffn_ag.count(27) && !hybrid_attn_ag.count(27) &&
                    !hybrid_ffn_rs.count(27);
            // In full-layer mode, prove that the output IDs are applied
            // AFTER the final FFN's AllGather. A flag alone is not evidence.
            bool deferred_output_select = false;
            if (live_hybrid_final_sp_requested) {
                int last_gather_node = -1;
                int output_select_node = -1;
                for (int node_id = 0; node_id < cgraph->n_nodes; ++node_id) {
                    const ggml_tensor * node = cgraph->nodes[node_id];
                    if (ggml_galaxy::get_stage(node) == ggml_galaxy::sp_stage::ffn_output_ag &&
                            ggml_galaxy::get_stage_layer(node) == 27) {
                        last_gather_node = node_id;
                    }
                    if (node->op == GGML_OP_GET_ROWS &&
                            std::strcmp(node->name, "galaxy_final_out_ids") == 0) {
                        output_select_node = node_id;
                    }
                }
                deferred_output_select = last_gather_node >= 0 &&
                        output_select_node > last_gather_node;
                if (!last_ffn_sp || !deferred_output_select ||
                        !hybrid_residual_scatter.count(27) ||
                        !hybrid_attention_rs.count(27)) {
                    GGML_LOG_ERROR("[GALAXY_CAUSAL_HYBRID_FFN_SUM] status=FAIL"
                            " reason=FINAL_LAYER_CONTRACT"
                            " last_ffn_sp=%d gather_node=%d output_select_node=%d\n",
                            int(last_ffn_sp), last_gather_node, output_select_node);
                    return GGML_STATUS_FAILED;
                }
            }
            std::fprintf(stderr,
                    "[GALAXY_CAUSAL_HYBRID_FFN_SUM] mode=REPLACE"
                    " attention_replaced=%zu attention_rs=%zu"
                    " residual_scatter=%zu ffn_sp_layers=%zu ffn_tp_layers=%zu"
                    " final_all_gather=%zu final_layer_sp=%d final_ffn_sp=%d"
                    " deferred_output_select=%d status=PASS\n",
                    attention_replaced, hybrid_attention_rs.size(),
                    hybrid_residual_scatter.size(), sp_ffn,
                    hybrid_attn_ag.size(), hybrid_ffn_ag.size(),
                    int(live_hybrid_final_sp_requested), int(last_ffn_sp),
                    int(deferred_output_select));
            std::fflush(stderr);
        }
    }
    if (has_galaxy_markers) {
        // A required correctness witness, not a verbose debug message.
        // The common CLI logger maps GGML_LOG_INFO to verbosity 4 (TRACE),
        // so it silently disappears under the normal -lv 3 smoke runner.
        // Emit exactly once per executed multi-token Meta graph, regardless
        // of the user's verbosity filter; the counts are POST-execution.
        std::fprintf(stderr,
                "[GALAXY_SP_SUM] reduced_scatter=%zu residual_scatter=%zu all_gather=%zu phase=PREFILL tile_tokens=%zu rs_fused_requested=%d\n",
                galaxy_rs_count, galaxy_scatter_count, galaxy_ag_count,
                ggml_galaxy_collectives::configured_tile_tokens(),
                int(ggml_galaxy_collectives::configured_rs_fused_read()));
        std::fflush(stderr);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
