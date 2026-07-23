#include "fe_absorption_metal.hpp"

#ifdef HDFE_USE_METAL

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace hdfe {
namespace detail {
namespace {

constexpr uint32_t kThreadgroupSize = 256;

struct ResetParams {
    uint32_t count = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};

struct AccumulateParams {
    uint32_t n = 0;
    uint32_t cols = 0;
    uint32_t ld = 0;
    uint32_t groups = 0;
    uint32_t unit_weights = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};

struct MeanParams {
    uint32_t groups = 0;
    uint32_t cols = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
};

struct ApplyParams {
    uint32_t n = 0;
    uint32_t cols = 0;
    uint32_t ld = 0;
    uint32_t groups = 0;
};

struct SumSqParams {
    uint32_t n = 0;
    uint32_t cols = 0;
    uint32_t ld = 0;
    uint32_t pad0 = 0;
};

struct JacobiParams {
    uint32_t n = 0;
    uint32_t cols = 0;
    uint32_t ld = 0;
    uint32_t dims = 0;
    double relaxation = 0.0;
    double pad0 = 0.0;
};

static const char kMetalSource[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct ResetParams {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct AccumulateParams {
    uint n;
    uint cols;
    uint ld;
    uint groups;
    uint unit_weights;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct MeanParams {
    uint groups;
    uint cols;
    uint pad0;
    uint pad1;
};

struct ApplyParams {
    uint n;
    uint cols;
    uint ld;
    uint groups;
};

struct SumSqParams {
    uint n;
    uint cols;
    uint ld;
    uint pad0;
};

struct JacobiParams {
    uint n;
    uint cols;
    uint ld;
    uint dims;
    double relaxation;
    double pad0;
};

inline void atomic_add_double(device atomic_ulong* addr, double val) {
    ulong old_bits = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        double old_val = as_type<double>(old_bits);
        double new_val = old_val + val;
        ulong new_bits = as_type<ulong>(new_val);
        if (atomic_compare_exchange_weak_explicit(addr, &old_bits, new_bits,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            break;
        }
    }
}

kernel void reset_u64(device atomic_ulong* data [[buffer(0)]],
                      constant ResetParams& params [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= params.count) {
        return;
    }
    atomic_store_explicit(&data[tid], 0, memory_order_relaxed);
}

kernel void accumulate_sums(const device double* y [[buffer(0)]],
                            const device double* X [[buffer(1)]],
                            const device int* gid [[buffer(2)]],
                            const device double* weights [[buffer(3)]],
                            device atomic_ulong* sum_y [[buffer(4)]],
                            device atomic_ulong* sum_x [[buffer(5)]],
                            constant AccumulateParams& params [[buffer(6)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= params.n) {
        return;
    }
    int g = gid[tid];
    double w = 1.0;
    if (params.unit_weights == 0) {
        w = weights[tid];
    }
    atomic_add_double(&sum_y[g], w * y[tid]);
    if (params.cols > 0) {
        for (uint j = 0; j < params.cols; ++j) {
            uint idx = j * params.groups + static_cast<uint>(g);
            atomic_add_double(&sum_x[idx], w * X[j * params.ld + tid]);
        }
    }
}

kernel void compute_means(const device atomic_ulong* sum_y [[buffer(0)]],
                          const device atomic_ulong* sum_x [[buffer(1)]],
                          const device double* weight_sums [[buffer(2)]],
                          device double* mean_y [[buffer(3)]],
                          device double* mean_x [[buffer(4)]],
                          constant MeanParams& params [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= params.groups) {
        return;
    }
    double denom = weight_sums[tid];
    double inv = denom > 0.0 ? 1.0 / denom : 0.0;
    ulong sumy_bits = atomic_load_explicit(&sum_y[tid], memory_order_relaxed);
    double sumy = as_type<double>(sumy_bits);
    mean_y[tid] = sumy * inv;
    if (params.cols > 0) {
        for (uint j = 0; j < params.cols; ++j) {
            uint idx = j * params.groups + tid;
            ulong sumx_bits = atomic_load_explicit(&sum_x[idx], memory_order_relaxed);
            double sumx = as_type<double>(sumx_bits);
            mean_x[idx] = sumx * inv;
        }
    }
}

kernel void apply_means(device double* y [[buffer(0)]],
                        device double* X [[buffer(1)]],
                        const device int* gid [[buffer(2)]],
                        const device double* mean_y [[buffer(3)]],
                        const device double* mean_x [[buffer(4)]],
                        constant ApplyParams& params [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= params.n) {
        return;
    }
    int g = gid[tid];
    y[tid] -= mean_y[g];
    if (params.cols > 0) {
        for (uint j = 0; j < params.cols; ++j) {
            uint idx = j * params.groups + static_cast<uint>(g);
            X[j * params.ld + tid] -= mean_x[idx];
        }
    }
}

kernel void sumsq(const device double* y [[buffer(0)]],
                  const device double* X [[buffer(1)]],
                  device atomic_ulong* sumsq_out [[buffer(2)]],
                  constant SumSqParams& params [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
    if (tid >= params.n) {
        return;
    }
    double local = y[tid] * y[tid];
    if (params.cols > 0) {
        for (uint j = 0; j < params.cols; ++j) {
            double xi = X[j * params.ld + tid];
            local += xi * xi;
        }
    }
    atomic_add_double(&sumsq_out[0], local);
}

kernel void jacobi_update(device double* y [[buffer(0)]],
                          device double* X [[buffer(1)]],
                          const device int* gid_all [[buffer(2)]],
                          const device double* mean_y_all [[buffer(3)]],
                          const device double* mean_x_all [[buffer(4)]],
                          const device int* group_offsets [[buffer(5)]],
                          const device int* x_offsets [[buffer(6)]],
                          const device int* groups [[buffer(7)]],
                          constant JacobiParams& params [[buffer(8)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= params.n) {
        return;
    }
    double delta_y = 0.0;
    for (uint dim = 0; dim < params.dims; ++dim) {
        uint base_gid = dim * params.n;
        int g = gid_all[base_gid + tid];
        int group_offset = group_offsets[dim];
        delta_y += mean_y_all[group_offset + g];
        if (params.cols > 0) {
            int groups_dim = groups[dim];
            int x_offset = x_offsets[dim];
            for (uint j = 0; j < params.cols; ++j) {
                uint idx = static_cast<uint>(x_offset + j * groups_dim + g);
                X[j * params.ld + tid] -= params.relaxation * mean_x_all[idx];
            }
        }
    }
    y[tid] -= params.relaxation * delta_y;
}
)metal";

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> reset_u64 = nil;
    id<MTLComputePipelineState> accumulate_sums = nil;
    id<MTLComputePipelineState> compute_means = nil;
    id<MTLComputePipelineState> apply_means = nil;
    id<MTLComputePipelineState> sumsq = nil;
    id<MTLComputePipelineState> jacobi_update = nil;
    bool available = false;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name) {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:fn error:&error];
    return pipeline;
}

MetalContext& metal_context() {
    static MetalContext ctx;
    static std::once_flag once;
    std::call_once(once, []() {
        @autoreleasepool {
            ctx.device = MTLCreateSystemDefaultDevice();
            if (!ctx.device) {
                return;
            }
            ctx.queue = [ctx.device newCommandQueue];
            if (!ctx.queue) {
                return;
            }

            NSString* source = [NSString stringWithUTF8String:kMetalSource];
            if (!source) {
                return;
            }
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            NSError* error = nil;
            ctx.library = [ctx.device newLibraryWithSource:source options:options error:&error];
            if (!ctx.library) {
                return;
            }

            ctx.reset_u64 = make_pipeline(ctx.device, ctx.library, @"reset_u64");
            ctx.accumulate_sums = make_pipeline(ctx.device, ctx.library, @"accumulate_sums");
            ctx.compute_means = make_pipeline(ctx.device, ctx.library, @"compute_means");
            ctx.apply_means = make_pipeline(ctx.device, ctx.library, @"apply_means");
            ctx.sumsq = make_pipeline(ctx.device, ctx.library, @"sumsq");
            ctx.jacobi_update = make_pipeline(ctx.device, ctx.library, @"jacobi_update");

            if (!ctx.reset_u64 || !ctx.accumulate_sums || !ctx.compute_means || !ctx.apply_means ||
                !ctx.sumsq || !ctx.jacobi_update) {
                return;
            }

            ctx.available = true;
        }
    });
    return ctx;
}

MTLSize threadgroup_size(id<MTLComputePipelineState> pipeline) {
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger size = std::min<NSUInteger>(kThreadgroupSize, max_threads);
    return MTLSizeMake(size, 1, 1);
}

double bits_to_double(uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

bool metal_backend_available() {
    return metal_context().available;
}

bool absorb_fixed_effects_metal(const Eigen::Ref<const Eigen::VectorXd>& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const std::vector<GpuFeInput>& fe_inputs,
                                const Eigen::VectorXd* weights,
                                const std::vector<std::size_t>& sweep_order,
                                const HdfeOptions& options,
                                AbsorptionMethod method,
                                AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    if (ld != n) {
        return false;
    }
    if (fe_inputs.empty()) {
        result.y_tilde = y;
        result.X_tilde = X;
        result.iterations = 0;
        result.converged = true;
        return true;
    }

    MetalContext& ctx = metal_context();
    if (!ctx.available) {
        return false;
    }

    const bool unit_weights = (weights == nullptr);
    const int dims = static_cast<int>(fe_inputs.size());

    std::vector<int> groups(static_cast<std::size_t>(dims));
    std::vector<int> group_offsets(static_cast<std::size_t>(dims));
    std::vector<int> x_offsets(static_cast<std::size_t>(dims));
    std::size_t total_groups = 0;
    std::size_t total_x = 0;

    result.fe_levels.clear();
    result.fe_levels.reserve(fe_inputs.size());
    for (int d = 0; d < dims; ++d) {
        groups[static_cast<std::size_t>(d)] = fe_inputs[d].num_groups;
        group_offsets[static_cast<std::size_t>(d)] = static_cast<int>(total_groups);
        x_offsets[static_cast<std::size_t>(d)] = static_cast<int>(total_x);
        total_groups += static_cast<std::size_t>(fe_inputs[d].num_groups);
        total_x += static_cast<std::size_t>(fe_inputs[d].num_groups) * static_cast<std::size_t>(cols);
        result.fe_levels.push_back(fe_inputs[d].num_levels_present);
    }

    std::vector<int> gid_all(static_cast<std::size_t>(dims) * static_cast<std::size_t>(n));
    for (int d = 0; d < dims; ++d) {
        std::memcpy(gid_all.data() + static_cast<std::size_t>(d) * n,
                    fe_inputs[d].group_ids, sizeof(int) * n);
    }

    std::vector<double> weight_sums_all(total_groups);
    for (int d = 0; d < dims; ++d) {
        const int offset = group_offsets[static_cast<std::size_t>(d)];
        std::memcpy(weight_sums_all.data() + offset, fe_inputs[d].weight_sums,
                    sizeof(double) * fe_inputs[d].num_groups);
    }

    auto make_buffer = [&](std::size_t bytes) -> id<MTLBuffer> {
        if (bytes == 0) {
            return nil;
        }
        id<MTLBuffer> buffer =
            [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        return buffer;
    };

    id<MTLBuffer> dummy_buffer = make_buffer(sizeof(double));
    if (!dummy_buffer) {
        return false;
    }

    id<MTLBuffer> y_buf = make_buffer(sizeof(double) * static_cast<std::size_t>(n));
    id<MTLBuffer> x_buf = cols > 0 ? make_buffer(sizeof(double) * static_cast<std::size_t>(ld) *
                                                    static_cast<std::size_t>(cols))
                                   : dummy_buffer;
    id<MTLBuffer> weights_buf = unit_weights
                                    ? dummy_buffer
                                    : make_buffer(sizeof(double) * static_cast<std::size_t>(n));
    id<MTLBuffer> gid_buf = make_buffer(sizeof(int) * gid_all.size());
    id<MTLBuffer> weight_sums_buf = make_buffer(sizeof(double) * weight_sums_all.size());
    id<MTLBuffer> sum_y_buf = make_buffer(sizeof(uint64_t) * total_groups);
    id<MTLBuffer> sum_x_buf =
        total_x > 0 ? make_buffer(sizeof(uint64_t) * total_x) : dummy_buffer;
    id<MTLBuffer> mean_y_buf = make_buffer(sizeof(double) * weight_sums_all.size());
    id<MTLBuffer> mean_x_buf =
        total_x > 0 ? make_buffer(sizeof(double) * total_x) : dummy_buffer;
    id<MTLBuffer> sumsq_buf = make_buffer(sizeof(uint64_t));
    id<MTLBuffer> group_offsets_buf = make_buffer(sizeof(int) * group_offsets.size());
    id<MTLBuffer> x_offsets_buf = make_buffer(sizeof(int) * x_offsets.size());
    id<MTLBuffer> groups_buf = make_buffer(sizeof(int) * groups.size());

    if (!y_buf || !x_buf || !weights_buf || !gid_buf || !weight_sums_buf || !sum_y_buf ||
        !mean_y_buf || !sumsq_buf || !group_offsets_buf || !x_offsets_buf || !groups_buf) {
        return false;
    }

    std::memcpy([y_buf contents], y.data(), sizeof(double) * static_cast<std::size_t>(n));
    if (cols > 0) {
        std::memcpy([x_buf contents], X.data(),
                    sizeof(double) * static_cast<std::size_t>(ld) * cols);
    }
    if (!unit_weights) {
        std::memcpy([weights_buf contents], weights->data(),
                    sizeof(double) * static_cast<std::size_t>(n));
    }
    std::memcpy([gid_buf contents], gid_all.data(), sizeof(int) * gid_all.size());
    std::memcpy([weight_sums_buf contents], weight_sums_all.data(),
                sizeof(double) * weight_sums_all.size());
    std::memcpy([group_offsets_buf contents], group_offsets.data(),
                sizeof(int) * group_offsets.size());
    std::memcpy([x_offsets_buf contents], x_offsets.data(), sizeof(int) * x_offsets.size());
    std::memcpy([groups_buf contents], groups.data(), sizeof(int) * groups.size());

    auto dispatch_1d = [&](id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pipeline,
                           uint32_t count) {
        if (count == 0) {
            return;
        }
        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize tg = threadgroup_size(pipeline);
        [enc setComputePipelineState:pipeline];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    };

    auto encode_reset = [&](id<MTLComputeCommandEncoder> enc,
                            id<MTLBuffer> buffer,
                            uint32_t count) {
        if (count == 0) {
            return;
        }
        ResetParams params;
        params.count = count;
        [enc setComputePipelineState:ctx.reset_u64];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc setBytes:&params length:sizeof(params) atIndex:1];
        dispatch_1d(enc, ctx.reset_u64, count);
    };

    auto encode_sumsq = [&](id<MTLComputeCommandEncoder> enc) {
        SumSqParams params;
        params.n = static_cast<uint32_t>(n);
        params.cols = static_cast<uint32_t>(cols);
        params.ld = static_cast<uint32_t>(ld);
        [enc setComputePipelineState:ctx.sumsq];
        [enc setBuffer:y_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:sumsq_buf offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(enc, ctx.sumsq, static_cast<uint32_t>(n));
    };

    auto encode_dim_pass = [&](id<MTLComputeCommandEncoder> enc,
                               std::size_t dim,
                               bool apply_means) {
        const int groups_dim = groups[dim];
        const NSUInteger gid_offset = static_cast<NSUInteger>(dim) * n * sizeof(int);
        const NSUInteger group_offset_bytes =
            static_cast<NSUInteger>(group_offsets[dim]) * sizeof(uint64_t);
        const NSUInteger x_offset_bytes =
            static_cast<NSUInteger>(x_offsets[dim]) * sizeof(uint64_t);
        const NSUInteger group_offset_f64 =
            static_cast<NSUInteger>(group_offsets[dim]) * sizeof(double);
        const NSUInteger x_offset_f64 =
            static_cast<NSUInteger>(x_offsets[dim]) * sizeof(double);

        AccumulateParams acc_params;
        acc_params.n = static_cast<uint32_t>(n);
        acc_params.cols = static_cast<uint32_t>(cols);
        acc_params.ld = static_cast<uint32_t>(ld);
        acc_params.groups = static_cast<uint32_t>(groups_dim);
        acc_params.unit_weights = unit_weights ? 1U : 0U;
        [enc setComputePipelineState:ctx.accumulate_sums];
        [enc setBuffer:y_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:gid_buf offset:gid_offset atIndex:2];
        [enc setBuffer:weights_buf offset:0 atIndex:3];
        [enc setBuffer:sum_y_buf offset:group_offset_bytes atIndex:4];
        [enc setBuffer:sum_x_buf offset:x_offset_bytes atIndex:5];
        [enc setBytes:&acc_params length:sizeof(acc_params) atIndex:6];
        dispatch_1d(enc, ctx.accumulate_sums, static_cast<uint32_t>(n));

        MeanParams mean_params;
        mean_params.groups = static_cast<uint32_t>(groups_dim);
        mean_params.cols = static_cast<uint32_t>(cols);
        [enc setComputePipelineState:ctx.compute_means];
        [enc setBuffer:sum_y_buf offset:group_offset_bytes atIndex:0];
        [enc setBuffer:sum_x_buf offset:x_offset_bytes atIndex:1];
        [enc setBuffer:weight_sums_buf offset:group_offset_f64 atIndex:2];
        [enc setBuffer:mean_y_buf offset:group_offset_f64 atIndex:3];
        [enc setBuffer:mean_x_buf offset:x_offset_f64 atIndex:4];
        [enc setBytes:&mean_params length:sizeof(mean_params) atIndex:5];
        dispatch_1d(enc, ctx.compute_means, static_cast<uint32_t>(groups_dim));

        if (apply_means) {
            ApplyParams apply_params;
            apply_params.n = static_cast<uint32_t>(n);
            apply_params.cols = static_cast<uint32_t>(cols);
            apply_params.ld = static_cast<uint32_t>(ld);
            apply_params.groups = static_cast<uint32_t>(groups_dim);
            [enc setComputePipelineState:ctx.apply_means];
            [enc setBuffer:y_buf offset:0 atIndex:0];
            [enc setBuffer:x_buf offset:0 atIndex:1];
            [enc setBuffer:gid_buf offset:gid_offset atIndex:2];
            [enc setBuffer:mean_y_buf offset:group_offset_f64 atIndex:3];
            [enc setBuffer:mean_x_buf offset:x_offset_f64 atIndex:4];
            [enc setBytes:&apply_params length:sizeof(apply_params) atIndex:5];
            dispatch_1d(enc, ctx.apply_means, static_cast<uint32_t>(n));
        }
    };

    auto read_sumsq = [&]() -> double {
        uint64_t bits = 0;
        std::memcpy(&bits, [sumsq_buf contents], sizeof(bits));
        return bits_to_double(bits);
    };

    auto compute_initial_norm = [&]() -> bool {
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        if (!cmd) {
            return false;
        }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) {
            return false;
        }
        encode_reset(enc, sumsq_buf, 1);
        encode_sumsq(enc);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            return false;
        }
        return true;
    };

    double prev_norm = 0.0;
    if (!compute_initial_norm()) {
        return false;
    }
    prev_norm = std::sqrt(read_sumsq());

    const int check_interval = std::max(1, options.convergence_check_interval);
    int last_check_iter = -1;
    bool converged = false;

    std::vector<std::size_t> order = sweep_order;
    if (order.empty()) {
        order.resize(static_cast<std::size_t>(dims));
        for (int d = 0; d < dims; ++d) {
            order[static_cast<std::size_t>(d)] = static_cast<std::size_t>(d);
        }
    }

    if (method == AbsorptionMethod::Jacobi) {
        double relaxation = options.jacobi_relaxation;
        if (relaxation <= 0.0) {
            relaxation = 2.0 / (static_cast<double>(dims) + 1.0);
        }
        relaxation = std::min(relaxation, 1.0);

        for (int iter = 0; iter < options.max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            if (!cmd) {
                return false;
            }
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!enc) {
                return false;
            }

            encode_reset(enc, sum_y_buf, static_cast<uint32_t>(total_groups));
            if (total_x > 0) {
                encode_reset(enc, sum_x_buf, static_cast<uint32_t>(total_x));
            }

            for (int d = 0; d < dims; ++d) {
                encode_dim_pass(enc, static_cast<std::size_t>(d), false);
            }

            JacobiParams jacobi_params;
            jacobi_params.n = static_cast<uint32_t>(n);
            jacobi_params.cols = static_cast<uint32_t>(cols);
            jacobi_params.ld = static_cast<uint32_t>(ld);
            jacobi_params.dims = static_cast<uint32_t>(dims);
            jacobi_params.relaxation = relaxation;
            [enc setComputePipelineState:ctx.jacobi_update];
            [enc setBuffer:y_buf offset:0 atIndex:0];
            [enc setBuffer:x_buf offset:0 atIndex:1];
            [enc setBuffer:gid_buf offset:0 atIndex:2];
            [enc setBuffer:mean_y_buf offset:0 atIndex:3];
            [enc setBuffer:mean_x_buf offset:0 atIndex:4];
            [enc setBuffer:group_offsets_buf offset:0 atIndex:5];
            [enc setBuffer:x_offsets_buf offset:0 atIndex:6];
            [enc setBuffer:groups_buf offset:0 atIndex:7];
            [enc setBytes:&jacobi_params length:sizeof(jacobi_params) atIndex:8];
            dispatch_1d(enc, ctx.jacobi_update, static_cast<uint32_t>(n));

            if (do_check) {
                encode_reset(enc, sumsq_buf, 1);
                encode_sumsq(enc);
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                return false;
            }

            if (do_check) {
                const double curr_norm = std::sqrt(read_sumsq());
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (step > 0 && (rel_change / static_cast<double>(step)) < options.tol) {
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
        }
    } else {
        const bool use_symmetric = options.symmetric_sweep && order.size() > 1;
        for (int iter = 0; iter < options.max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            if (!cmd) {
                return false;
            }
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!enc) {
                return false;
            }

            encode_reset(enc, sum_y_buf, static_cast<uint32_t>(total_groups));
            if (total_x > 0) {
                encode_reset(enc, sum_x_buf, static_cast<uint32_t>(total_x));
            }
            for (std::size_t pos = 0; pos < order.size(); ++pos) {
                encode_dim_pass(enc, order[pos], true);
            }
            if (use_symmetric) {
                encode_reset(enc, sum_y_buf, static_cast<uint32_t>(total_groups));
                if (total_x > 0) {
                    encode_reset(enc, sum_x_buf, static_cast<uint32_t>(total_x));
                }
                for (std::size_t idx = order.size(); idx-- > 0;) {
                    encode_dim_pass(enc, order[idx], true);
                }
            }

            if (do_check) {
                encode_reset(enc, sumsq_buf, 1);
                encode_sumsq(enc);
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                return false;
            }

            if (do_check) {
                const double curr_norm = std::sqrt(read_sumsq());
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (step > 0 && (rel_change / static_cast<double>(step)) < options.tol) {
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
        }
    }

    result.converged = converged;
    if (!converged) {
        result.iterations = options.max_iter;
    } else if (result.iterations == 0) {
        result.iterations = 1;
    }

    result.y_tilde.resize(n);
    std::memcpy(result.y_tilde.data(), [y_buf contents],
                sizeof(double) * static_cast<std::size_t>(n));
    if (cols > 0) {
        result.X_tilde.resize(ld, cols);
        std::memcpy(result.X_tilde.data(), [x_buf contents],
                    sizeof(double) * static_cast<std::size_t>(ld) * cols);
    } else {
        result.X_tilde.resize(ld, 0);
    }

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(fe_inputs.size());
        result.fe_means.resize(fe_inputs.size());
        result.fe_weight_sums.resize(fe_inputs.size());
        const double* mean_y_ptr = static_cast<const double*>([mean_y_buf contents]);
        for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
            const int groups_dim = fe_inputs[dim].num_groups;
            const int offset = group_offsets[dim];
            result.fe_group_ids[dim].assign(fe_inputs[dim].group_ids,
                                            fe_inputs[dim].group_ids + n);
            result.fe_means[dim].resize(groups_dim);
            result.fe_weight_sums[dim].resize(groups_dim);
            std::memcpy(result.fe_weight_sums[dim].data(), fe_inputs[dim].weight_sums,
                        sizeof(double) * groups_dim);
            std::memcpy(result.fe_means[dim].data(), mean_y_ptr + offset,
                        sizeof(double) * groups_dim);
        }
    }

    result.sweep_order_used.clear();
    if (method == AbsorptionMethod::Jacobi || order.empty()) {
        result.sweep_order_used.reserve(fe_inputs.size());
        for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }
    } else {
        result.sweep_order_used.reserve(order.size());
        for (const std::size_t dim : order) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }
    }

    return true;
}

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_USE_METAL
