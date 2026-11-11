#include "n1_graph_preconditioner.hpp"
#include "fe_absorption_cuda.hpp"
#include "group_refinement.hpp"
#include "n1_continuation.hpp"

#include <cuda_runtime.h>
#include "weighted_two_by_two.hpp"
#include "additive_cell_projection.hpp"
#include <cub/cub.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hdfe {
namespace detail {
namespace {

constexpr int kBlockSize = 256;
constexpr int kForwardProbeMaxP = kCudaForwardProbeMaxP;
constexpr int kForwardProbeGrid = 512;
static_assert(kForwardProbeMaxP * kForwardProbeMaxP <= kBlockSize,
              "forward-probe Gram finish requires one CUDA block");

inline bool finite_double_bits(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "xhdfe requires 64-bit IEEE-754 doubles");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

bool strict_residual_tolerance_mode(const HdfeOptions& options) {
    return options.tolerance_mode == ToleranceMode::StrictResidual;
}

double effective_absorption_tolerance(const HdfeOptions& options) {
    if (options.tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(options.tol, 1.0e-9);
    }
    return options.tol;
}

double group_individual_absorption_tolerance(const HdfeOptions& options) {
    const double tol = effective_absorption_tolerance(options);
    if (tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(tol, 1.0e-12);
    }
    return tol;
}

// Prototype knob (measurement phase): see fe_absorption.cpp counterpart.
bool honest_tol_trigger_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("XHDFE_TOL_TRIGGER");
        return e != nullptr && (e[0] == 'c' || e[0] == 'C' || e[0] == '1');
    }();
    return enabled;
}

// See the fe_absorption.cpp counterpart for the full rationale: the Irons-Tuck
// divergence safeguard must not be gated on the stopping rule. The CUDA solver
// carried the identical defect — on `github` at tol=1e-12 the GPU path returned
// converged=true with coefficients [37.44, 83.27, -17.21] against the correct
// [0.962444, 0.955465, 23.854767].
// Set XHDFE_ACCEL_GUARD_ALWAYS=0 to restore the honest_mode-only gating.
bool accel_guard_always_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("XHDFE_ACCEL_GUARD_ALWAYS");
        return !(e != nullptr && e[0] == '0');
    }();
    return enabled;
}

inline void cuda_check(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        const char* err = cudaGetErrorString(status);
        throw std::runtime_error(std::string(msg) + ": " + (err ? err : "unknown"));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count == 0) {
            reset();
            return;
        }
        if (ptr_ && count <= size_) {
            return;
        }
        reset();
        size_ = count;
        cuda_check(cudaMalloc(&ptr_, sizeof(T) * count), "cudaMalloc failed");
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void move_from(DeviceBuffer& other) {
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Pinned host staging for large host<->device copies. Pageable cudaMemcpy of
// multi-GB buffers runs far below PCIe rate (the CUDA driver pipelines them
// through a small internal bounce buffer); a double-buffered pinned pipeline
// recovers most of the link bandwidth. Below kStagedCopyMinBytes the plain
// cudaMemcpy path is kept so small transfers are byte-for-byte unaffected.
// The staged copies move exactly the same bytes; results are bit-identical.
class PinnedStage {
public:
    ~PinnedStage() { reset(); }

    PinnedStage() = default;
    PinnedStage(const PinnedStage&) = delete;
    PinnedStage& operator=(const PinnedStage&) = delete;

    bool ensure(std::size_t chunk_bytes) {
        if (ptr_ && chunk_ >= chunk_bytes) {
            return true;
        }
        reset();
        void* p = nullptr;
        if (cudaMallocHost(&p, chunk_bytes * 2) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        ptr_ = static_cast<unsigned char*>(p);
        chunk_ = chunk_bytes;
        return true;
    }

    unsigned char* data(int idx) {
        return ptr_ + static_cast<std::size_t>(idx) * chunk_;
    }

    void reset() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            chunk_ = 0;
        }
    }

private:
    unsigned char* ptr_ = nullptr;
    std::size_t chunk_ = 0;
};

constexpr std::size_t kStagedCopyMinBytes = 64ull << 20;
constexpr std::size_t kStagedChunkBytes = 32ull << 20;

// WP1 telemetry (XHDFE_PROFILE_CUDA=1): phase timings and transfer counts for
// the absorption entry point. Counters are process-wide; the per-call report
// diffs them so concurrent fits merely add noise to counts, never to timings.
bool cuda_profile_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("XHDFE_PROFILE_CUDA");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return enabled;
}

std::atomic<long long> g_cuda_h2d_calls{0};
std::atomic<long long> g_cuda_d2h_calls{0};

// WP3 shadow-mode device certificate (default OFF). When on, the verifier TU
// computes the certificate quantities on the device and the result is logged
// for comparison against the host certificate, which stays authoritative.
bool cuda_cert_shadow_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("XHDFE_GPU_CERT_SHADOW");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return enabled;
}

// Diagnostic-only convergence trace for the accelerated CUDA sweeps.  The
// solver already copies the three Irons-Tuck sums below to the host on every
// accelerated iteration.  Keeping their raw values here adds no device work,
// synchronization, allocation, or public result field.  All derived
// arithmetic lives in a noinline emitter so the CUDA host compiler cannot
// reassociate it with the solver's fast-math expressions.
bool cuda_accel_trace_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("XHDFE_CUDA_ACCEL_TRACE");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return enabled;
}

bool cuda_forward_probe_environment_enabled() {
    static const bool environment_enabled = [] {
        const char* raw = std::getenv("XHDFE_CUDA_FORWARD_PROBE");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return environment_enabled;
}

bool cuda_forward_probe_enabled() {
    return cuda_forward_probe_environment_enabled() ||
           thread_cuda_forward_probe_requested();
}

bool cuda_forward_probe_diag_enabled() {
    static const bool enabled = [] {
        const char* own = std::getenv("XHDFE_CUDA_FORWARD_PROBE_DIAG");
        const char* retry = std::getenv(
            "XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_RETRY_DIAG");
        return (own != nullptr && *own != '\0' && *own != '0') ||
               (retry != nullptr && *retry != '\0' && *retry != '0');
    }();
    return enabled;
}

int cuda_forward_probe_sweeps() {
    static const int sweeps = [] {
        const char* raw = std::getenv("XHDFE_CUDA_FORWARD_PROBE_SWEEPS");
        const int parsed = raw != nullptr ? std::atoi(raw) : 4;
        return parsed > 0 ? std::min(parsed, 64) : 4;
    }();
    return sweeps;
}

int cuda_forward_probe_forced_status() {
    static const int status = [] {
        const char* raw = std::getenv(
            "XHDFE_CUDA_FORWARD_PROBE_FORCE_STATUS");
        const int parsed = raw != nullptr ? std::atoi(raw) : 0;
        return parsed >= 2 && parsed <= 4 ? parsed : 0;
    }();
    return status;
}

std::atomic<long long> g_cuda_forward_probe_sequence{0};

#if defined(_MSC_VER)
#define XHDFE_CUDA_HOST_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define XHDFE_CUDA_HOST_NOINLINE __attribute__((noinline))
#else
#define XHDFE_CUDA_HOST_NOINLINE
#endif

struct CudaAccelTraceRecord {
    double vprod;
    double ssq;
    double d1sq;
    double prev_norm;
    int phase;
    int iter;
};

struct CudaAccelTraceBuffer {
    static constexpr int kRingSize = 16;
    bool enabled = false;
    bool use_accel = false;
    bool two_stage = false;
    bool honest_mode = false;
    int n = 0;
    int cols = 0;
    int dims = 0;
    int phase = 0;
    int ring_count = 0;
    int ring_next = 0;
    long long records = 0;
    double honest_tol = 0.0;
    double convergence_tol = 0.0;
    CudaAccelTraceRecord ring[kRingSize];

    void begin_phase() {
        if (enabled) {
            ++phase;
        }
    }

    void record(double vprod_value,
                double ssq_value,
                double d1sq_value,
                double prev_norm_value,
                int iter_value) {
        if (!enabled) {
            return;
        }
        CudaAccelTraceRecord& out = ring[ring_next];
        out.vprod = vprod_value;
        out.ssq = ssq_value;
        out.d1sq = d1sq_value;
        out.prev_norm = prev_norm_value;
        out.phase = phase;
        out.iter = iter_value;
        ring_next = (ring_next + 1) % kRingSize;
        ring_count = std::min(ring_count + 1, kRingSize);
        ++records;
    }
};

XHDFE_CUDA_HOST_NOINLINE void emit_cuda_accel_trace(
    const CudaAccelTraceBuffer& trace,
    int iterations,
    bool converged) {
    if (!trace.enabled) {
        return;
    }

    double ratios[CudaAccelTraceBuffer::kRingSize];
    int ratio_count = 0;
    double r_exit = std::numeric_limits<double>::quiet_NaN();
    double cos_exit = std::numeric_limits<double>::quiet_NaN();
    double displacement_rel_exit = std::numeric_limits<double>::quiet_NaN();
    double error_rel_exit = std::numeric_limits<double>::quiet_NaN();
    int exit_phase = 0;
    int exit_iter = -1;

    for (int k = 0; k < trace.ring_count; ++k) {
        const int index =
            (trace.ring_next - trace.ring_count + k +
             2 * CudaAccelTraceBuffer::kRingSize) %
            CudaAccelTraceBuffer::kRingSize;
        const CudaAccelTraceRecord& rec = trace.ring[index];
        const double d0sq = rec.ssq + rec.d1sq - 2.0 * rec.vprod;
        if (!(d0sq > 0.0) || !(rec.d1sq >= 0.0) ||
            !finite_double_bits(d0sq) || !finite_double_bits(rec.d1sq)) {
            continue;
        }
        const double ratio = std::sqrt(rec.d1sq / d0sq);
        if (!finite_double_bits(ratio)) {
            continue;
        }
        ratios[ratio_count++] = ratio;
        if (k + 1 == trace.ring_count) {
            const double d0d1 = rec.d1sq - rec.vprod;
            r_exit = ratio;
            cos_exit = rec.d1sq > 0.0
                           ? d0d1 / std::sqrt(d0sq * rec.d1sq)
                           : std::numeric_limits<double>::quiet_NaN();
            displacement_rel_exit =
                std::sqrt(rec.d1sq) / std::max(1.0, rec.prev_norm);
            error_rel_exit = ratio < 1.0
                                 ? displacement_rel_exit / (1.0 - ratio)
                                 : std::numeric_limits<double>::infinity();
            exit_phase = rec.phase;
            exit_iter = rec.iter;
        }
    }

    std::sort(ratios, ratios + ratio_count);
    const double r_tail_median =
        ratio_count > 0
            ? ratios[ratio_count / 2]
            : std::numeric_limits<double>::quiet_NaN();
    const double r_tail_max =
        ratio_count > 0
            ? ratios[ratio_count - 1]
            : std::numeric_limits<double>::quiet_NaN();
    std::fprintf(
        stderr,
        "cuda_accel_trace n=%d cols=%d dims=%d accel=%d two_stage=%d "
        "honest=%d tol=%.17g eff_tol=%.17g phases=%d records=%lld "
        "iterations=%d converged=%d exit_phase=%d exit_iter=%d "
        "r_exit=%.17g r_tail_median=%.17g r_tail_max=%.17g "
        "cos_exit=%.17g displacement_rel_exit=%.17g "
        "error_rel_exit=%.17g ring=%d\n",
        trace.n, trace.cols, trace.dims, trace.use_accel ? 1 : 0,
        trace.two_stage ? 1 : 0, trace.honest_mode ? 1 : 0,
        trace.honest_tol, trace.convergence_tol, trace.phase, trace.records,
        iterations, converged ? 1 : 0, exit_phase, exit_iter, r_exit,
        r_tail_median, r_tail_max, cos_exit, displacement_rel_exit,
        error_rel_exit, trace.ring_count);
}

struct CudaForwardProbeGram {
    int p = 0;
    bool centered = false;
    double values[kForwardProbeMaxP * kForwardProbeMaxP] = {0.0};
};

struct CudaForwardProbeFit {
    bool ok = false;
    int columns = 0;
    unsigned int mask = 0;
    double yy = 0.0;
    double rss = 0.0;
    double intercept = 0.0;
    double condition_proxy = 0.0;
    double coefficients[kForwardProbeMaxP] = {0.0};
    double standard_errors[kForwardProbeMaxP] = {0.0};
    double covariance[kForwardProbeMaxP * kForwardProbeMaxP] = {0.0};
};

XHDFE_CUDA_HOST_NOINLINE bool fit_cuda_forward_probe_gram(
    const CudaForwardProbeGram& gram,
    long long observations,
    CudaForwardProbeFit& out) {
    out = CudaForwardProbeFit{};
    const int p = gram.p;
    if (p < 2 || p > kForwardProbeMaxP || observations <= 0) {
        return false;
    }
    for (int index = 0; index < p * p; ++index) {
        if (!finite_double_bits(gram.values[index])) {
            return false;
        }
    }
    auto raw_value = [&](int row, int column) {
        return gram.values[row * p + column];
    };
    const int regressor_count = gram.centered ? p - 2 : p - 1;
    if (regressor_count < 1) {
        return false;
    }
    const int constant_column = p - 1;
    const double centering_denominator =
        gram.centered
            ? raw_value(constant_column, constant_column)
            : 1.0;
    if (!(centering_denominator > 0.0)) {
        return false;
    }
    auto value = [&](int row, int column) {
        const double raw = raw_value(row, column);
        if (!gram.centered) {
            return raw;
        }
        return raw -
               raw_value(row, constant_column) *
                   raw_value(column, constant_column) /
                   centering_denominator;
    };

    out.yy = value(0, 0);
    double max_diagonal = 0.0;
    for (int column = 1; column <= regressor_count; ++column) {
        max_diagonal = std::max(max_diagonal, value(column, column));
    }
    if (!(max_diagonal > 0.0)) {
        return false;
    }

    constexpr double kAbsoluteFloor = 1.0e-14;
    constexpr double kPivotTolerance = 1.0e-12;
    int selected[kForwardProbeMaxP] = {0};
    int selected_count = 0;
    double lower[kForwardProbeMaxP * kForwardProbeMaxP] = {0.0};
    double new_row[kForwardProbeMaxP] = {0.0};
    for (int column = 1; column <= regressor_count; ++column) {
        const double diagonal = value(column, column);
        if (!(diagonal > kAbsoluteFloor * max_diagonal)) {
            continue;
        }
        double pivot = diagonal;
        for (int prior = 0; prior < selected_count; ++prior) {
            double cross = value(column, selected[prior]);
            for (int inner = 0; inner < prior; ++inner) {
                cross -= lower[prior * kForwardProbeMaxP + inner] *
                         new_row[inner];
            }
            new_row[prior] =
                cross / lower[prior * kForwardProbeMaxP + prior];
            pivot -= new_row[prior] * new_row[prior];
        }
        if (!(pivot > kPivotTolerance * diagonal)) {
            continue;
        }
        for (int prior = 0; prior < selected_count; ++prior) {
            lower[selected_count * kForwardProbeMaxP + prior] =
                new_row[prior];
        }
        lower[selected_count * kForwardProbeMaxP + selected_count] =
            std::sqrt(pivot);
        selected[selected_count] = column;
        out.mask |= 1U << static_cast<unsigned int>(column - 1);
        ++selected_count;
    }
    if (selected_count == 0) {
        return false;
    }

    double work[kForwardProbeMaxP] = {0.0};
    double beta[kForwardProbeMaxP] = {0.0};
    for (int row = 0; row < selected_count; ++row) {
        double rhs = value(selected[row], 0);
        for (int prior = 0; prior < row; ++prior) {
            rhs -= lower[row * kForwardProbeMaxP + prior] * work[prior];
        }
        work[row] = rhs / lower[row * kForwardProbeMaxP + row];
    }
    for (int row = selected_count; row-- > 0;) {
        double rhs = work[row];
        for (int following = row + 1; following < selected_count; ++following) {
            rhs -= lower[following * kForwardProbeMaxP + row] *
                   beta[following];
        }
        beta[row] = rhs / lower[row * kForwardProbeMaxP + row];
    }

    double rss = out.yy;
    for (int row = 0; row < selected_count; ++row) {
        rss -= beta[row] * value(selected[row], 0);
        out.coefficients[selected[row] - 1] = beta[row];
    }
    if (gram.centered) {
        double intercept =
            raw_value(0, constant_column) / centering_denominator;
        for (int row = 0; row < selected_count; ++row) {
            intercept -= beta[row] *
                         raw_value(selected[row], constant_column) /
                         centering_denominator;
        }
        out.intercept = intercept;
    }
    if (!finite_double_bits(rss) || !(rss > 0.0)) {
        return false;
    }

    double inverse[kForwardProbeMaxP * kForwardProbeMaxP] = {0.0};
    for (int column = 0; column < selected_count; ++column) {
        for (int row = 0; row < selected_count; ++row) {
            double rhs = row == column ? 1.0 : 0.0;
            for (int prior = 0; prior < row; ++prior) {
                rhs -= lower[row * kForwardProbeMaxP + prior] * work[prior];
            }
            work[row] = rhs / lower[row * kForwardProbeMaxP + row];
        }
        for (int row = selected_count; row-- > 0;) {
            double rhs = work[row];
            for (int following = row + 1; following < selected_count;
                 ++following) {
                rhs -= lower[following * kForwardProbeMaxP + row] *
                       inverse[following * kForwardProbeMaxP + column];
            }
            inverse[row * kForwardProbeMaxP + column] =
                rhs / lower[row * kForwardProbeMaxP + row];
        }
    }

    const double degrees =
        static_cast<double>(observations - selected_count -
                            (gram.centered ? 1 : 0));
    if (!(degrees > 0.0)) {
        return false;
    }
    const double sigma2 = rss / degrees;
    double minimum_cholesky =
        lower[0 * kForwardProbeMaxP + 0];
    double maximum_cholesky = minimum_cholesky;
    for (int row = 0; row < selected_count; ++row) {
        const double diagonal = lower[row * kForwardProbeMaxP + row];
        minimum_cholesky = std::min(minimum_cholesky, diagonal);
        maximum_cholesky = std::max(maximum_cholesky, diagonal);
        const int public_row = selected[row] - 1;
        const double variance =
            sigma2 * inverse[row * kForwardProbeMaxP + row];
        if (!(variance >= 0.0) || !finite_double_bits(variance)) {
            return false;
        }
        out.standard_errors[public_row] = std::sqrt(variance);
        for (int column = 0; column < selected_count; ++column) {
            const int public_column = selected[column] - 1;
            out.covariance[public_row * kForwardProbeMaxP + public_column] =
                sigma2 * inverse[row * kForwardProbeMaxP + column];
        }
    }

    out.columns = selected_count;
    out.rss = rss;
    out.condition_proxy =
        minimum_cholesky > 0.0
            ? (maximum_cholesky / minimum_cholesky) *
                  (maximum_cholesky / minimum_cholesky)
            : std::numeric_limits<double>::infinity();
    out.ok = finite_double_bits(out.condition_proxy);
    return out.ok;
}

XHDFE_CUDA_HOST_NOINLINE CudaForwardProbeSummary emit_cuda_forward_probe(
    long long sequence,
    int n,
    int cols,
    int dims,
    bool honest_mode,
    bool use_accel,
    int iterations,
    int polish_sweeps,
    int continuation_sweeps,
    double exit_gram_ms,
    double continuation_ms,
    bool diagnostics_enabled,
    const CudaForwardProbeGram& gram_public,
    const CudaForwardProbeGram& gram_continuation,
    const CudaForwardProbeFit& exit_fit,
    const CudaForwardProbeFit& public_fit,
    const CudaForwardProbeFit& continuation_fit,
    const CudaForwardProbeFit& second_continuation_fit) {
    CudaForwardProbeSummary summary;
    summary.continuation_sweeps = continuation_sweeps;
    const bool masks_match =
        exit_fit.mask == public_fit.mask &&
        public_fit.mask == continuation_fit.mask &&
        continuation_fit.mask == second_continuation_fit.mask;
    if (!exit_fit.ok || !public_fit.ok || !continuation_fit.ok ||
        !second_continuation_fit.ok || !masks_match) {
        if (diagnostics_enabled) {
            std::fprintf(
                stderr,
                "cuda_forward_probe seq=%lld status=fit_error n=%d cols=%d "
                "dims=%d exit_ok=%d public_ok=%d k_ok=%d k2_ok=%d "
                "mask_exit=%x mask_public=%x mask_k=%x mask_k2=%x\n",
                sequence, n, cols, dims, exit_fit.ok ? 1 : 0,
                public_fit.ok ? 1 : 0, continuation_fit.ok ? 1 : 0,
                second_continuation_fit.ok ? 1 : 0, exit_fit.mask,
                public_fit.mask, continuation_fit.mask,
                second_continuation_fit.mask);
        }
        summary.status = 2;
        return summary;
    }

    double coefficient_polish_max = 0.0;
    double coefficient_k_max = 0.0;
    double coefficient_k2_max = 0.0;
    double first_step_squared = 0.0;
    double second_step_squared = 0.0;
    double second_step_max = 0.0;
    double standard_error_k2_max = 0.0;
    double covariance_k2_max = 0.0;
    for (int column = 0; column < cols; ++column) {
        if ((public_fit.mask & (1U << static_cast<unsigned int>(column))) == 0U) {
            continue;
        }
        coefficient_polish_max = std::max(
            coefficient_polish_max,
            std::abs(public_fit.coefficients[column] -
                     exit_fit.coefficients[column]));
        const double first_step =
            continuation_fit.coefficients[column] -
            public_fit.coefficients[column];
        const double second_step =
            second_continuation_fit.coefficients[column] -
            continuation_fit.coefficients[column];
        coefficient_k_max =
            std::max(coefficient_k_max, std::abs(first_step));
        coefficient_k2_max = std::max(
            coefficient_k2_max,
            std::abs(second_continuation_fit.coefficients[column] -
                     public_fit.coefficients[column]));
        first_step_squared += first_step * first_step;
        second_step_squared += second_step * second_step;
        second_step_max = std::max(second_step_max, std::abs(second_step));
        standard_error_k2_max = std::max(
            standard_error_k2_max,
            std::abs(second_continuation_fit.standard_errors[column] -
                     public_fit.standard_errors[column]));
        for (int other = 0; other < cols; ++other) {
            if ((public_fit.mask &
                 (1U << static_cast<unsigned int>(other))) == 0U) {
                continue;
            }
            covariance_k2_max = std::max(
                covariance_k2_max,
                std::abs(second_continuation_fit.covariance[
                             column * kForwardProbeMaxP + other] -
                         public_fit.covariance[
                             column * kForwardProbeMaxP + other]));
        }
    }
    const double first_norm = std::sqrt(first_step_squared);
    const double second_norm = std::sqrt(second_step_squared);
    const double rho_k =
        first_norm > 0.0 ? second_norm / first_norm : 0.0;
    const double tail_max =
        rho_k < 1.0
            ? second_step_max * rho_k / (1.0 - rho_k)
            : std::numeric_limits<double>::infinity();
    const double total_drift_estimate = coefficient_k2_max + tail_max;
    const double rss_scale = std::max(std::abs(public_fit.rss), 1.0e-300);
    const double rss_k_relative =
        std::abs(continuation_fit.rss - public_fit.rss) / rss_scale;
    const double rss_k2_relative =
        std::abs(second_continuation_fit.rss - public_fit.rss) / rss_scale;

    double gram_difference_squared = 0.0;
    double gram_public_squared = 0.0;
    for (int row = 1; row <= cols; ++row) {
        for (int column = 1; column <= cols; ++column) {
            const double public_value =
                gram_public.values[row * gram_public.p + column];
            const double difference =
                gram_continuation.values[
                    row * gram_continuation.p + column] -
                public_value;
            gram_difference_squared += difference * difference;
            gram_public_squared += public_value * public_value;
        }
    }
    const double gram_relative =
        gram_public_squared > 0.0
            ? std::sqrt(gram_difference_squared / gram_public_squared)
            : std::numeric_limits<double>::infinity();
    const double rss_cancellation =
        public_fit.rss > 0.0
            ? public_fit.yy / public_fit.rss
            : std::numeric_limits<double>::infinity();

    summary.status = 1;
    summary.coefficient_drift_abs = coefficient_k2_max;
    summary.rss_drift_rel = rss_k2_relative;
    summary.standard_error_drift_abs = standard_error_k2_max;
    summary.covariance_drift_abs = covariance_k2_max;
    summary.contraction = rho_k;
    summary.coefficient_tail_abs = tail_max;
    summary.condition_proxy = public_fit.condition_proxy;

    if (diagnostics_enabled) {
        std::fprintf(
            stderr,
        "cuda_forward_probe seq=%lld status=ok n=%d cols=%d dims=%d "
        "honest=%d accel=%d iterations=%d polish=%d k=%d mask=%x "
        "db_polish_max=%.17g db_k_max=%.17g db_k2_max=%.17g "
        "rho_k=%.17g tail_b_max=%.17g drift_total_est=%.17g "
        "rss_public=%.17g rss_k_rel=%.17g rss_k2_rel=%.17g "
        "dse_k2_max=%.17g dcov_k2_max=%.17g dgram_k2_rel=%.17g "
        "condition_proxy=%.17g rss_cancel=%.17g "
        "exit_gram_ms=%.3f continuation_ms=%.3f\n",
        sequence, n, cols, dims, honest_mode ? 1 : 0,
        use_accel ? 1 : 0, iterations, polish_sweeps,
        continuation_sweeps, public_fit.mask, coefficient_polish_max,
        coefficient_k_max, coefficient_k2_max, rho_k, tail_max,
        total_drift_estimate, public_fit.rss, rss_k_relative,
        rss_k2_relative, standard_error_k2_max, covariance_k2_max,
            gram_relative, public_fit.condition_proxy, rss_cancellation,
            exit_gram_ms, continuation_ms);
    }
    return summary;
}

#undef XHDFE_CUDA_HOST_NOINLINE

void staged_memcpy_h2d(void* dst_dev,
                       const void* src_host,
                       std::size_t bytes,
                       PinnedStage& stage,
                       const char* msg) {
    g_cuda_h2d_calls.fetch_add(1, std::memory_order_relaxed);
    if (bytes < kStagedCopyMinBytes || !stage.ensure(kStagedChunkBytes)) {
        cuda_check(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice), msg);
        return;
    }
    cudaEvent_t ev[2];
    cuda_check(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming),
               "staged h2d event create failed");
    if (cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaEventDestroy(ev[0]);
        cuda_check(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice), msg);
        return;
    }
    const unsigned char* src = static_cast<const unsigned char*>(src_host);
    unsigned char* dst = static_cast<unsigned char*>(dst_dev);
    std::size_t off = 0;
    int buf = 0;
    while (off < bytes) {
        const std::size_t len = std::min(kStagedChunkBytes, bytes - off);
        // Wait for this stage buffer's previous async upload to drain, then
        // refill it while the other buffer's upload is still in flight.
        cuda_check(cudaEventSynchronize(ev[buf]), "staged h2d event sync failed");
        std::memcpy(stage.data(buf), src + off, len);
        cuda_check(cudaMemcpyAsync(dst + off, stage.data(buf), len,
                                   cudaMemcpyHostToDevice),
                   msg);
        cuda_check(cudaEventRecord(ev[buf]), "staged h2d event record failed");
        off += len;
        buf ^= 1;
    }
    cuda_check(cudaEventSynchronize(ev[0]), "staged h2d final sync failed");
    cuda_check(cudaEventSynchronize(ev[1]), "staged h2d final sync failed");
    (void)cudaEventDestroy(ev[0]);
    (void)cudaEventDestroy(ev[1]);
}

void staged_memcpy_d2h(void* dst_host,
                       const void* src_dev,
                       std::size_t bytes,
                       PinnedStage& stage,
                       const char* msg) {
    g_cuda_d2h_calls.fetch_add(1, std::memory_order_relaxed);
    if (bytes < kStagedCopyMinBytes || !stage.ensure(kStagedChunkBytes)) {
        cuda_check(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost), msg);
        return;
    }
    cudaEvent_t ev[2];
    cuda_check(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming),
               "staged d2h event create failed");
    if (cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaEventDestroy(ev[0]);
        cuda_check(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost), msg);
        return;
    }
    const unsigned char* src = static_cast<const unsigned char*>(src_dev);
    unsigned char* dst = static_cast<unsigned char*>(dst_host);
    std::size_t off = 0;
    int buf = 0;
    bool have_prev = false;
    int prev_buf = 0;
    std::size_t prev_off = 0;
    std::size_t prev_len = 0;
    while (off < bytes) {
        const std::size_t len = std::min(kStagedChunkBytes, bytes - off);
        cuda_check(cudaMemcpyAsync(stage.data(buf), src + off, len,
                                   cudaMemcpyDeviceToHost),
                   msg);
        cuda_check(cudaEventRecord(ev[buf]), "staged d2h event record failed");
        if (have_prev) {
            cuda_check(cudaEventSynchronize(ev[prev_buf]),
                       "staged d2h event sync failed");
            // Intentionally single-threaded: the destination arrays are
            // re-read by the whole post-absorption pipeline, and a parallel
            // first-touch here would fragment their transparent huge pages
            // (see PERF_PREP_APPLIED.md for the regression this caused).
            std::memcpy(dst + prev_off, stage.data(prev_buf), prev_len);
        }
        have_prev = true;
        prev_buf = buf;
        prev_off = off;
        prev_len = len;
        off += len;
        buf ^= 1;
    }
    cuda_check(cudaEventSynchronize(ev[prev_buf]), "staged d2h event sync failed");
    std::memcpy(dst + prev_off, stage.data(prev_buf), prev_len);
    (void)cudaEventDestroy(ev[0]);
    (void)cudaEventDestroy(ev[1]);
}

__global__ __launch_bounds__(256, 4) void accumulate_sums_atomic_kernel(const double* y,
                                              const double* X,
                                              int n,
                                              int cols,
                                              int ld,
                                              const int* gid,
                                              const double* weights,
                                              bool unit_weights,
                                              double* sum_y,
                                              double* sum_x,
                                              int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    const double w = unit_weights ? 1.0 : weights[i];
    atomicAdd(&sum_y[g], w * y[i]);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g],
                  w * X[static_cast<std::size_t>(j) * ld + i]);
    }
}

// Deterministic small-path accumulator.  Each lane owns one whole RHS column
// (lane 0 owns y), so no two CUDA threads update the same output array.  This
// removes atomic-order variation from the small, nearly saturated 2-FE CG
// route while still executing the absorption on the GPU.
__global__ __launch_bounds__(256, 2) void accumulate_sums_deterministic_columns_kernel(
    const double* y,
    const double* X,
    int n,
    int cols,
    int ld,
    const int* gid,
    const double* weights,
    bool unit_weights,
    double* sum_y,
    double* sum_x,
    int groups) {
    const int component = blockIdx.x * blockDim.x + threadIdx.x;
    if (component > cols) {
        return;
    }
    if (component == 0) {
        for (int i = 0; i < n; ++i) {
            const int g = gid[i];
            const double w = unit_weights ? 1.0 : weights[i];
            sum_y[g] += w * y[i];
        }
        return;
    }
    const int column = component - 1;
    double* out =
        sum_x + static_cast<std::size_t>(column) * groups;
    const double* in =
        X + static_cast<std::size_t>(column) * ld;
    for (int i = 0; i < n; ++i) {
        const int g = gid[i];
        const double w = unit_weights ? 1.0 : weights[i];
        out[g] += w * in[i];
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_slope_sums_atomic_kernel(
    const double* y,
    const double* X,
    int n,
    int cols,
    int ld,
    const int* gid,
    const double* z,
    const double* weights,
    bool unit_weights,
    double* sum_y,
    double* sum_zy,
    double* sum_x,
    double* sum_zx,
    int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    if (g < 0 || g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[i];
    const double zi = z[i];
    const double yi = y[i];
    atomicAdd(&sum_y[g], w * yi);
    atomicAdd(&sum_zy[g], w * zi * yi);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const double xi = X[static_cast<std::size_t>(j) * ld + i];
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g], w * xi);
        atomicAdd(&sum_zx[static_cast<std::size_t>(j) * groups + g], w * zi * xi);
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_edge_sums_kernel(const double* y,
                                                                      const double* X,
                                                                      int n,
                                                                      int cols,
                                                                      int ld,
                                                                      int edge_stride,
                                                                      const int* edge_group,
                                                                      double* sum_y,
                                                                      double* sum_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = edge_group[i];
    atomicAdd(&sum_y[e], y[i]);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * edge_stride + e],
                  X[static_cast<std::size_t>(j) * ld + i]);
    }
}

__global__ __launch_bounds__(256, 4) void apply_edge_residuals_kernel(double* y,
                                                                      double* X,
                                                                      int n,
                                                                      int cols,
                                                                      int ld,
                                                                      int edge_stride,
                                                                      const int* edge_group,
                                                                      const int* edge_counts,
                                                                      const double* sum_y,
                                                                      const double* sum_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = edge_group[i];
    const double inv = 1.0 / static_cast<double>(edge_counts[e]);
    y[i] -= sum_y[e] * inv;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        X[static_cast<std::size_t>(j) * ld + i] -=
            sum_x[static_cast<std::size_t>(j) * edge_stride + e] * inv;
    }
}

// Accumulate group sums using warp-segmented reductions to cut down on atomics
// when observations are locally sorted by gid (common for panel data).
// Correctness does not rely on global sorting: repeated gid values that are not
// contiguous within a warp simply form multiple segments whose sums accumulate.
__global__ __launch_bounds__(256, 2) void accumulate_sums_segmented_kernel(const double* y,
                                                 const double* X,
                                                 int n,
                                                 int cols,
                                                 int ld,
                                                 const int* gid,
                                                 const double* weights,
                                                 bool unit_weights,
                                                 double* sum_y,
                                                 double* sum_x,
                                                 int groups) {
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
    static_assert(kBlockSize % kWarpSize == 0, "kBlockSize must be a multiple of warp size");

    using WarpReduce = cub::WarpReduce<double, kWarpSize>;
    __shared__ typename WarpReduce::TempStorage temp_storage[kWarpsPerBlock];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in_range = (i < n);
    const int g = in_range ? gid[i] : -1;
    const double w = (!in_range) ? 0.0 : (unit_weights ? 1.0 : weights[i]);
    const double vy = w * (in_range ? y[i] : 0.0);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    const int prev_g = __shfl_up_sync(0xFFFFFFFFu, g, 1);
    const int head = (lane == 0) || (g != prev_g);

    const double agg_y = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(vy, head);
    if (in_range && head) {
        atomicAdd(&sum_y[g], agg_y);
    }

#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const double vx = w * (in_range ? X[static_cast<std::size_t>(j) * ld + i] : 0.0);
        const double agg_x = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(vx, head);
        if (in_range && head) {
            atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g], agg_x);
        }
    }
}

// Privatized accumulation for low-cardinality FEs: each block holds a
// per-group shared-memory sum, with fast shared atomics, then flushes to
// global once per (block, group). Reduces global atomic contention by a
// factor of `blocks_n` compared with a plain global-atomic kernel.
// Required shared memory per block: groups * (1 + cols) * 8 bytes.
// Caller must ensure this fits in the dynamic shared-memory budget.
__global__ __launch_bounds__(256, 2) void accumulate_sums_privatized_kernel(
    const double* y,
    const double* X,
    int n,
    int cols,
    int ld,
    const int* gid,
    const double* weights,
    bool unit_weights,
    double* sum_y,
    double* sum_x,
    int groups) {
    extern __shared__ double shared_sums[];
    double* shared_y = shared_sums;
    double* shared_x = shared_sums + groups;
    const int total_x = groups * cols;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    for (int g = tid; g < groups; g += block_size) {
        shared_y[g] = 0.0;
    }
    for (int k = tid; k < total_x; k += block_size) {
        shared_x[k] = 0.0;
    }
    __syncthreads();

    const int i = blockIdx.x * block_size + tid;
    if (i < n) {
        const int g = gid[i];
        const double w = unit_weights ? 1.0 : weights[i];
        atomicAdd(&shared_y[g], w * y[i]);
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            atomicAdd(&shared_x[static_cast<std::size_t>(j) * groups + g],
                      w * X[static_cast<std::size_t>(j) * ld + i]);
        }
    }
    __syncthreads();

    for (int g = tid; g < groups; g += block_size) {
        const double v = shared_y[g];
        if (v != 0.0) {
            atomicAdd(&sum_y[g], v);
        }
    }
    for (int k = tid; k < total_x; k += block_size) {
        const double v = shared_x[k];
        if (v != 0.0) {
            atomicAdd(&sum_x[k], v);
        }
    }
}

__global__ __launch_bounds__(256, 4) void compute_means_kernel(double* sum_y,
                                     double* sum_x,
                                     const double* weight_sums,
                                     int groups,
                                     int cols) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double denom = weight_sums[g];
    const double inv = denom > 0.0 ? 1.0 / denom : 0.0;
    sum_y[g] *= inv;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        sum_x[static_cast<std::size_t>(j) * groups + g] *= inv;
    }
}

__global__ __launch_bounds__(256, 4) void compute_slope_coefficients_kernel(
    double* sum_y,
    double* sum_zy,
    double* sum_x,
    double* sum_zx,
    const double* weight_sums,
    const double* sum_z,
    const double* sum_zz,
    int groups,
    int cols,
    bool include_intercept) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    constexpr double kDetTol = 1e-12;
    const double sw = weight_sums[g];
    const double sz = sum_z[g];
    const double szz = sum_zz[g];
    const double sy = sum_y[g];
    const double szy = sum_zy[g];
    double alpha = 0.0;
    double gamma = 0.0;
    if (include_intercept) {
        if (sw > 0.0) {
            const double det = sw * szz - sz * sz;
            const double scale = sw * szz;
            if (det > kDetTol * scale) {
                alpha = (szz * sy - sz * szy) / det;
                gamma = (-sz * sy + sw * szy) / det;
            } else {
                alpha = sy / sw;
            }
        }
    } else {
        if (szz > 0.0) {
            gamma = szy / szz;
        }
    }
    sum_y[g] = alpha;
    sum_zy[g] = gamma;

#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
        const double sx = sum_x[idx];
        const double szx = sum_zx[idx];
        double ax = 0.0;
        double gx = 0.0;
        if (include_intercept) {
            if (sw > 0.0) {
                const double det = sw * szz - sz * sz;
                const double scale = sw * szz;
                if (det > kDetTol * scale) {
                    ax = (szz * sx - sz * szx) / det;
                    gx = (-sz * sx + sw * szx) / det;
                } else {
                    ax = sx / sw;
                }
            }
        } else {
            if (szz > 0.0) {
                gx = szx / szz;
            }
        }
        sum_x[idx] = ax;
        sum_zx[idx] = gx;
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_alpha_kernel(const double* mean_y,
                                        const double* mean_x,
                                        double* alpha_y,
                                        double* alpha_x,
                                        int groups,
                                        int cols,
                                        double scale) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    if (alpha_y) {
        alpha_y[g] += scale * mean_y[g];
    }
    if (alpha_x) {
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
            alpha_x[idx] += scale * mean_x[idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void apply_means_kernel(double* y,
                                   double* X,
                                   int n,
                                   int cols,
                                   int ld,
                                   const int* gid,
                                   const double* mean_y,
                                   const double* mean_x,
                                   int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    y[i] -= mean_y[g];
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        X[static_cast<std::size_t>(j) * ld + i] -=
            mean_x[static_cast<std::size_t>(j) * groups + g];
    }
}

__global__ __launch_bounds__(256, 4) void apply_slope_kernel(double* y,
                                   double* X,
                                   int n,
                                   int cols,
                                   int ld,
                                   const int* gid,
                                   const double* z,
                                   const double* alpha_y,
                                   const double* gamma_y,
                                   const double* alpha_x,
                                   const double* gamma_x,
                                   int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    const double zi = z[i];
    y[i] -= alpha_y[g] + gamma_y[g] * zi;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
        X[static_cast<std::size_t>(j) * ld + i] -= alpha_x[idx] + gamma_x[idx] * zi;
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void apply_means_sumsq_kernel(double* y,
                                         double* X,
                                         int n,
                                         int cols,
                                         int ld,
                                         const int* gid,
                                         const double* mean_y,
                                         const double* mean_x,
                                         int groups,
                                         double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const int g = gid[i];
        double yi = y[i] - mean_y[g];
        y[i] = yi;
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            double xi =
                X[static_cast<std::size_t>(j) * ld + i] -
                mean_x[static_cast<std::size_t>(j) * groups + g];
            X[static_cast<std::size_t>(j) * ld + i] = xi;
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void apply_slope_sumsq_kernel(double* y,
                                         double* X,
                                         int n,
                                         int cols,
                                         int ld,
                                         const int* gid,
                                         const double* z,
                                         const double* alpha_y,
                                         const double* gamma_y,
                                         const double* alpha_x,
                                         const double* gamma_x,
                                         int groups,
                                         double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const int g = gid[i];
        const double zi = z[i];
        double yi = y[i] - alpha_y[g] - gamma_y[g] * zi;
        y[i] = yi;
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
            double xi =
                X[static_cast<std::size_t>(j) * ld + i] - alpha_x[idx] - gamma_x[idx] * zi;
            X[static_cast<std::size_t>(j) * ld + i] = xi;
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void sumsq_kernel(const double* y,
	                             const double* X,
	                             int n,
                             int cols,
                             int ld,
                             double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const double yi = y[i];
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const double xi = X[static_cast<std::size_t>(j) * ld + i];
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

__device__ double reldif_device(double newer, double older) {
    return fabs(newer - older) / (fabs(older) + 1.0);
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void reldif_update_sums_kernel(
    const double* y_new,
    const double* y_old,
    const double* X_new,
    const double* X_old,
    const double* weights,
    bool unit_weights,
    int n,
    int cols,
    int ld,
    double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int width = cols + 1;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        const double w = unit_weights ? 1.0 : weights[i];
        if (component == width) {
            local += w;
        } else if (component == 0) {
            local += w * reldif_device(y_new[i], y_old[i]);
        } else {
            const int j = component - 1;
            const std::size_t idx = static_cast<std::size_t>(j) * ld + i;
            local += w * reldif_device(X_new[idx], X_old[idx]);
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void irons_tuck_stats_kernel(const double* x,
                                        const double* gx,
                                        const double* ggx,
                                        int n,
                                        double* stats_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_vprod;
    __shared__ typename BlockReduce::TempStorage temp_ssq;
    __shared__ typename BlockReduce::TempStorage temp_d1sq;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local_vprod = 0.0;
    double local_ssq = 0.0;
    double local_d1sq = 0.0;
#pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        const double gx_val = gx[i];
        const double delta_gx = ggx[i] - gx_val;
        const double delta2 = delta_gx - gx_val + x[i];
        local_vprod += delta_gx * delta2;
        local_ssq += delta2 * delta2;
        local_d1sq += delta_gx * delta_gx;
    }
    const double block_vprod = BlockReduce(temp_vprod).Sum(local_vprod);
    const double block_ssq = BlockReduce(temp_ssq).Sum(local_ssq);
    const double block_d1sq = BlockReduce(temp_d1sq).Sum(local_d1sq);
    if (threadIdx.x == 0) {
        atomicAdd(&stats_out[0], block_vprod);
        atomicAdd(&stats_out[1], block_ssq);
        atomicAdd(&stats_out[2], block_d1sq);
    }
}

__global__ __launch_bounds__(256, 4) void irons_tuck_update_kernel(double* x,
                                         const double* gx,
                                         const double* ggx,
                                         int n,
                                         double coef) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
#pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        const double delta_gx = ggx[i] - gx[i];
        x[i] = ggx[i] - coef * delta_gx;
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void component_sumsq_kernel(const double* y,
                                                                   const double* X,
                                                                   int n,
                                                                   int cols,
                                                                   int ld,
                                                                   double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        double v = 0.0;
        if (component == 0) {
            v = y[i];
        } else {
            const int j = component - 1;
            if (j < cols) {
                v = X[static_cast<std::size_t>(j) * ld + i];
            }
        }
        local += v * v;
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void component_dot_kernel(const double* y_a,
                                                                const double* y_b,
                                                                const double* X_a,
                                                                const double* X_b,
                                                                int n,
                                                                int cols,
                                                                int ld,
                                                                double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        double a = 0.0;
        double b = 0.0;
        if (component == 0) {
            a = y_a[i];
            b = y_b[i];
        } else {
            const int j = component - 1;
            if (j < cols) {
                const std::size_t idx = static_cast<std::size_t>(j) * ld + i;
                a = X_a[idx];
                b = X_b[idx];
            }
        }
        local += a * b;
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

__global__ __launch_bounds__(kBlockSize, 2) void forward_probe_gram_partials_kernel(
    const double* y,
    const double* X,
    int n,
    int ld,
    int cols,
    int p,
    double* partials) {
    const int first = static_cast<int>(blockIdx.y);
    const int second = static_cast<int>(blockIdx.z);
    if (second < first) {
        return;
    }
    const double* first_values = nullptr;
    if (first == 0) {
        first_values = y;
    } else if (first <= cols) {
        first_values = X + static_cast<std::size_t>(first - 1) * ld;
    }
    const double* second_values = nullptr;
    if (second == 0) {
        second_values = y;
    } else if (second <= cols) {
        second_values = X + static_cast<std::size_t>(second - 1) * ld;
    }
    __shared__ double scratch[kBlockSize];
    double local = 0.0;
    for (int row = static_cast<int>(blockIdx.x) * kBlockSize + threadIdx.x;
         row < n;
         row += kBlockSize * static_cast<int>(gridDim.x)) {
        const double first_value =
            first_values != nullptr ? first_values[row] : 1.0;
        const double second_value =
            second_values != nullptr ? second_values[row] : 1.0;
        local = __dadd_rn(
            local, __dmul_rn(first_value, second_value));
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = __dadd_rn(
                scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const std::size_t pair =
            static_cast<std::size_t>(first) * p + second;
        partials[pair * gridDim.x + blockIdx.x] = scratch[0];
    }
}

__global__ __launch_bounds__(kBlockSize, 2) void forward_probe_gram_finish_kernel(
    const double* partials,
    int p,
    int grid,
    double* gram) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= p * p) {
        return;
    }
    const int first = pair / p;
    const int second = pair % p;
    if (second < first) {
        return;
    }
    const double* values =
        partials + static_cast<std::size_t>(pair) * grid;
    double total = 0.0;
    for (int block = 0; block < grid; ++block) {
        total = __dadd_rn(total, values[block]);
    }
    gram[pair] = total;
    gram[second * p + first] = total;
}

__global__ __launch_bounds__(kBlockSize, 2) void forward_probe_rss_partials_kernel(
    const double* y,
    const double* X,
    const double* coefficients,
    double intercept,
    int n,
    int ld,
    int cols,
    double* partials) {
    __shared__ double scratch[kBlockSize];
    double local = 0.0;
    for (int row = static_cast<int>(blockIdx.x) * kBlockSize + threadIdx.x;
         row < n;
         row += kBlockSize * static_cast<int>(gridDim.x)) {
        double residual = __dadd_rn(y[row], -intercept);
        for (int column = 0; column < cols; ++column) {
            residual = __dadd_rn(
                residual,
                -__dmul_rn(
                    X[static_cast<std::size_t>(column) * ld + row],
                    coefficients[column]));
        }
        local = __dadd_rn(local, __dmul_rn(residual, residual));
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = __dadd_rn(
                scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = scratch[0];
    }
}

__global__ void forward_probe_rss_finish_kernel(
    const double* partials,
    int count,
    double* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double total = 0.0;
        for (int index = 0; index < count; ++index) {
            total = __dadd_rn(total, partials[index]);
        }
        output[0] = total;
    }
}

__global__ __launch_bounds__(256, 4) void projection_residual_kernel(double* proj_y,
                                                                     const double* src_y,
                                                                     double* proj_x,
                                                                     const double* src_x,
                                                                     std::size_t total,
                                                                     int n,
                                                                     int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            proj_y[idx] = src_y[idx] - proj_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            proj_x[x_idx] = src_x[x_idx] - proj_x[x_idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void scaled_subtract_columns_kernel(double* target_y,
                                                                        double* target_x,
                                                                        const double* direction_y,
                                                                        const double* direction_x,
                                                                        const double* alpha,
                                                                        std::size_t total,
                                                                        int n,
                                                                        int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            target_y[idx] -= alpha[0] * direction_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            const int j = static_cast<int>(x_idx / static_cast<std::size_t>(ld));
            target_x[x_idx] -= alpha[j + 1] * direction_x[x_idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void scale_columns_kernel(
    double* y,
    double* X,
    const double* scale,
    std::size_t total,
    int n,
    int ld) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            y[idx] *= scale[0];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            const int j =
                static_cast<int>(x_idx / static_cast<std::size_t>(ld));
            X[x_idx] *= scale[j + 1];
        }
    }
}

__global__ __launch_bounds__(256, 4) void cg_direction_update_kernel(double* u_y,
                                                                     double* u_x,
                                                                     const double* r_y,
                                                                     const double* r_x,
                                                                     const double* beta,
                                                                     std::size_t total,
                                                                     int n,
                                                                     int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            u_y[idx] = r_y[idx] + beta[0] * u_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            const int j = static_cast<int>(x_idx / static_cast<std::size_t>(ld));
            u_x[x_idx] = r_x[x_idx] + beta[j + 1] * u_x[x_idx];
        }
    }
}

__global__ void any_nonzero_bits_kernel(const double* values,
                                        int n,
                                        unsigned int* any_nonzero) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        const unsigned long long bits =
            static_cast<unsigned long long>(__double_as_longlong(values[i]));
        if ((bits & UINT64_C(0x7fffffffffffffff)) != 0U) {
            atomicExch(any_nonzero, 1U);
            return;
        }
    }
}

// Reduce max(|values[i]|) over [0, n). IEEE-754 non-negative doubles preserve
// their ordering when reinterpreted as unsigned 64-bit, so atomicMax on the
// bit representation is correct for |x|.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void max_abs_kernel(const double* values,
                                                            int n,
                                                            double* max_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local_max = 0.0;
    for (int i = tid; i < n; i += stride) {
        const double v = fabs(values[i]);
        if (v > local_max) {
            local_max = v;
        }
    }
    const double block_max = BlockReduce(temp).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0 && block_max > 0.0) {
        unsigned long long* addr_ull =
            reinterpret_cast<unsigned long long*>(max_out);
        unsigned long long new_ull = __double_as_longlong(block_max);
        atomicMax(addr_ull, new_ull);
    }
}

__global__ void jacobi_update_kernel(double* y,
                                     double* X,
                                     int n,
                                     int cols,
                                     int ld,
                                     const int* const* gid_ptrs,
                                     const double* const* mean_y_ptrs,
                                     const double* const* mean_x_ptrs,
                                     const int* groups,
                                     int dims,
                                     double relaxation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double delta_y = 0.0;
    for (int dim = 0; dim < dims; ++dim) {
        const int g = gid_ptrs[dim][i];
        delta_y += mean_y_ptrs[dim][g];
        for (int j = 0; j < cols; ++j) {
            X[static_cast<std::size_t>(j) * ld + i] -=
                relaxation * mean_x_ptrs[dim][static_cast<std::size_t>(j) * groups[dim] + g];
        }
    }
    y[i] -= relaxation * delta_y;
}

__global__ void gi_denom_kernel(const int* group_ptr,
                                const int* group_individual,
                                const double* group_scale,
                                const double* weights,
                                bool unit_weights,
                                int groups,
                                double* denom) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[g];
    const double scale = group_scale[g];
    const double contrib = w * scale * scale;
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];
    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        atomicAdd(&denom[i], contrib);
    }
}

__global__ void gi_denom_by_individual_kernel(const int* individual_ptr,
                                              const int* individual_group,
                                              const double* group_scale,
                                              const double* weights,
                                              bool unit_weights,
                                              int groups,
                                              int individuals,
                                              double* denom) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= individuals) {
        return;
    }
    const int begin = individual_ptr[i];
    const int end = individual_ptr[i + 1];
    double acc = 0.0;
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        const double w = unit_weights ? 1.0 : weights[g];
        acc += w * scale * scale;
    }
    denom[i] = acc > 0.0 ? acc : 1.0;
}

// Gauss-Seidel sweep over one color class of individuals. Individuals within
// a color share no observation row, so the reads (numer) and the atomicAdd
// updates touch disjoint rows and the launch is race-free and deterministic.
// Colors are processed sequentially by the host loop, which makes the overall
// sweep a valid Gauss-Seidel pass (same fixed point as the CPU sweep). A
// single launch over ALL individuals (the previous scheme) is NOT convergent:
// two individuals sharing a row read the same residual and double-subtract,
// which oscillates instead of contracting.
__global__ void gi_color_sweep_kernel(double* y,
                                      double* X,
                                      int groups,
                                      int cols,
                                      int ld,
                                      const int* individual_ptr,
                                      const int* individual_group,
                                      const double* group_scale,
                                      const double* weights,
                                      bool unit_weights,
                                      const double* denom,
                                      const int* color_individuals,
                                      int color_begin,
                                      int color_count,
                                      double relaxation) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= color_count) {
        return;
    }
    const int i = color_individuals[color_begin + t];
    const int begin = individual_ptr[i];
    const int end = individual_ptr[i + 1];
    const double inv_denom = 1.0 / (denom[i] > 0.0 ? denom[i] : 1.0);

    double numer_y = 0.0;
    if (cols > 0 && cols <= 8) {
        double numer_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            const double w = unit_weights ? 1.0 : weights[g];
            const double weight_scale = w * scale;
            numer_y += weight_scale * y[g];
            for (int j = 0; j < cols; ++j) {
                numer_x[j] += weight_scale * X[static_cast<std::size_t>(j) * ld + g];
            }
        }
        const double alpha_y = numer_y * inv_denom;
        double alpha_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int j = 0; j < cols; ++j) {
            alpha_x[j] = numer_x[j] * inv_denom;
        }
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            atomicAdd(&y[g], -relaxation * scale * alpha_y);
            for (int j = 0; j < cols; ++j) {
                atomicAdd(&X[static_cast<std::size_t>(j) * ld + g],
                          -relaxation * scale * alpha_x[j]);
            }
        }
        return;
    }

    // Generic fallback for cols > 8: loop over columns to avoid large per-thread buffers.
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        const double w = unit_weights ? 1.0 : weights[g];
        const double weight_scale = w * scale;
        numer_y += weight_scale * y[g];
    }
    const double alpha_y = numer_y * inv_denom;
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        atomicAdd(&y[g], -relaxation * scale * alpha_y);
    }
    for (int j = 0; j < cols; ++j) {
        double numer = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            const double w = unit_weights ? 1.0 : weights[g];
            const double weight_scale = w * scale;
            numer += weight_scale * X[static_cast<std::size_t>(j) * ld + g];
        }
        const double alpha = numer * inv_denom;
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            atomicAdd(&X[static_cast<std::size_t>(j) * ld + g],
                      -relaxation * scale * alpha);
        }
    }
}

__global__ void gi_accumulate_numer_kernel(const double* y,
                                          const double* X,
                                          int groups,
                                          int cols,
                                          int ld,
                                          const int* group_ptr,
                                          const int* group_individual,
                                          const double* group_scale,
                                          const double* weights,
                                          bool unit_weights,
                                          int individuals,
                                          double* numer_y,
                                          double* numer_x) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[g];
    const double scale = group_scale[g];
    const double coeff = w * scale;
    const double y_contrib = coeff * y[g];
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];

    if (cols > 0 && cols <= 8) {
        double x_contrib[8];
        for (int j = 0; j < cols; ++j) {
            x_contrib[j] = coeff * X[static_cast<std::size_t>(j) * ld + g];
        }
        for (int pos = begin; pos < end; ++pos) {
            const int i = group_individual[pos];
            if (i < 0 || i >= individuals) {
                continue;
            }
            atomicAdd(&numer_y[i], y_contrib);
            for (int j = 0; j < cols; ++j) {
                atomicAdd(&numer_x[static_cast<std::size_t>(j) * individuals + i], x_contrib[j]);
            }
        }
        return;
    }

    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        if (i < 0 || i >= individuals) {
            continue;
        }
        atomicAdd(&numer_y[i], y_contrib);
        for (int j = 0; j < cols; ++j) {
            const double x_contrib = coeff * X[static_cast<std::size_t>(j) * ld + g];
            atomicAdd(&numer_x[static_cast<std::size_t>(j) * individuals + i], x_contrib);
        }
    }
}

__global__ void gi_compute_alpha_kernel(const double* numer_y,
                                       const double* numer_x,
                                       const double* denom,
                                       int individuals,
                                       int cols,
                                       double* alpha_y,
                                       double* alpha_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= individuals) {
        return;
    }
    const double d = denom[i] > 0.0 ? denom[i] : 1.0;
    const double inv = 1.0 / d;
    alpha_y[i] = numer_y[i] * inv;
    for (int j = 0; j < cols; ++j) {
        alpha_x[static_cast<std::size_t>(j) * individuals + i] =
            numer_x[static_cast<std::size_t>(j) * individuals + i] * inv;
    }
}

__global__ void gi_apply_update_kernel(double* y,
                                      double* X,
                                      int groups,
                                      int cols,
                                      int ld,
                                      const int* group_ptr,
                                      const int* group_individual,
                                      const double* group_scale,
                                      const double* alpha_y,
                                      const double* alpha_x,
                                      int individuals,
                                      double relaxation) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double scale = group_scale[g];
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];

    double sum_y = 0.0;
    if (cols > 0) {
        if (cols <= 8) {
            double sum_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_individual[pos];
                if (i < 0 || i >= individuals) {
                    continue;
                }
                sum_y += alpha_y[i];
                for (int j = 0; j < cols; ++j) {
                    sum_x[j] += alpha_x[static_cast<std::size_t>(j) * individuals + i];
                }
            }
            y[g] -= relaxation * scale * sum_y;
            for (int j = 0; j < cols; ++j) {
                X[static_cast<std::size_t>(j) * ld + g] -= relaxation * scale * sum_x[j];
            }
            return;
        }
        // Generic fallback: scan the small group neighborhood once per column.
        for (int pos = begin; pos < end; ++pos) {
            const int i = group_individual[pos];
            if (i < 0 || i >= individuals) {
                continue;
            }
            sum_y += alpha_y[i];
        }
        y[g] -= relaxation * scale * sum_y;
        for (int j = 0; j < cols; ++j) {
            double sum_x = 0.0;
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_individual[pos];
                if (i < 0 || i >= individuals) {
                    continue;
                }
                sum_x += alpha_x[static_cast<std::size_t>(j) * individuals + i];
            }
            X[static_cast<std::size_t>(j) * ld + g] -= relaxation * scale * sum_x;
        }
        return;
    }

    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        if (i < 0 || i >= individuals) {
            continue;
        }
        sum_y += alpha_y[i];
    }
    y[g] -= relaxation * scale * sum_y;
}

// Matrix-free joint group()/individual() LSMR primitives. Every A' output has
// one stable CSR owner and all norm reductions use a fixed two-stage tree; no
// atomic accumulation can affect a Krylov iterate or stopping decision.
__global__ void gi_lsmr_inverse_sqrt_kernel(const double* diagonal,
                                            int count,
                                            int output_offset,
                                            double* column_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const double value = diagonal[i];
        column_scale[output_offset + i] =
            value > 0.0 ? 1.0 / sqrt(value) : 0.0;
    }
}

__global__ void gi_lsmr_build_rhs_kernel(const double* raw,
                                         const double* weights,
                                         bool unit_weights,
                                         int n,
                                         double* rhs) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        rhs[row] = (unit_weights ? 1.0 : sqrt(weights[row])) * raw[row];
    }
}

__global__ void gi_lsmr_scale_coefficients_kernel(const double* values,
                                                  const double* column_scale,
                                                  int count,
                                                  double* scaled) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < count) {
        scaled[column] = column_scale[column] * values[column];
    }
}

__global__ void gi_lsmr_apply_a_kernel(
    const double* scaled_coefficients,
    const int* const* standard_gid,
    const int* offsets,
    int dims,
    const int* group_ptr,
    const int* group_individual,
    const double* group_scale,
    int individual_offset,
    const double* weights,
    bool unit_weights,
    bool weighted_output,
    int groups,
    double* output) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) {
        return;
    }
    double fitted = 0.0;
    for (int dim = 0; dim < dims; ++dim) {
        fitted += scaled_coefficients[
            offsets[dim] + standard_gid[dim][group]];
    }
    double individual_sum = 0.0;
    for (int pos = group_ptr[group]; pos < group_ptr[group + 1]; ++pos) {
        individual_sum += scaled_coefficients[
            individual_offset + group_individual[pos]];
    }
    fitted += group_scale[group] * individual_sum;
    if (weighted_output && !unit_weights) {
        fitted *= sqrt(weights[group]);
    }
    output[group] = fitted;
}

__global__ void gi_lsmr_apply_at_standard_kernel(
    const double* input,
    const int* group_ptr,
    const int* observation_index,
    const double* weights,
    bool unit_weights,
    const double* column_scale,
    int output_offset,
    int levels,
    double* output) {
    const int level = blockIdx.x * blockDim.x + threadIdx.x;
    if (level >= levels) {
        return;
    }
    double value = 0.0;
    for (int pos = group_ptr[level]; pos < group_ptr[level + 1]; ++pos) {
        const int row = observation_index ? observation_index[pos] : pos;
        value += (unit_weights ? 1.0 : sqrt(weights[row])) * input[row];
    }
    output[output_offset + level] =
        column_scale[output_offset + level] * value;
}

__global__ void gi_lsmr_apply_at_individual_kernel(
    const double* input,
    const int* individual_ptr,
    const int* individual_group,
    const double* group_scale,
    const double* weights,
    bool unit_weights,
    const double* column_scale,
    int output_offset,
    int individuals,
    double* output) {
    const int individual = blockIdx.x * blockDim.x + threadIdx.x;
    if (individual >= individuals) {
        return;
    }
    double value = 0.0;
    for (int pos = individual_ptr[individual];
         pos < individual_ptr[individual + 1]; ++pos) {
        const int group = individual_group[pos];
        value += (unit_weights ? 1.0 : sqrt(weights[group])) *
                 group_scale[group] * input[group];
    }
    output[output_offset + individual] =
        column_scale[output_offset + individual] * value;
}

__global__ void gi_lsmr_scale_copy_kernel(const double* source,
                                          double scale,
                                          int count,
                                          double* target) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        target[i] = scale * source[i];
    }
}

// Ordinary, unweighted incidence columns contain literal ones. Large groups
// need cooperative sums rather than a serial walk by one CUDA thread.
__global__ void ordinary_incidence_diagonal_kernel(
    const int* ptr, int levels, double* diagonal) {
    const int level = blockIdx.x * blockDim.x + threadIdx.x;
    if (level < levels) diagonal[level] = static_cast<double>(ptr[level + 1] - ptr[level]);
}

template <int BLOCK>
__global__ void ordinary_at_block_kernel(
    const double* input, const int* ptr, const int* observation,
    const double* scale, int offset, int levels, double* output) {
    const int level = blockIdx.x;
    if (level >= levels) return;
    using Reduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename Reduce::TempStorage temp;
    double sum = 0.0;
    const int end = ptr[level + 1];
    for (std::int64_t pos = static_cast<std::int64_t>(ptr[level]) + threadIdx.x;
         pos < end; pos += BLOCK) {
        const int row = observation ? observation[pos] : static_cast<int>(pos);
        sum += input[row];
    }
    const double total = Reduce(temp).Sum(sum);
    if (threadIdx.x == 0) output[offset + level] = scale[offset + level] * total;
}

template <int BLOCK>
__global__ void ordinary_at_warp_kernel(
    const double* input, const int* ptr, const int* observation,
    const double* scale, int offset, int levels, double* output) {
    constexpr int WARP = 32;
    constexpr int WARPS = BLOCK / WARP;
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int level = blockIdx.x * WARPS + warp;
    if (level >= levels) return;  // A whole warp takes this branch.
    using Reduce = cub::WarpReduce<double, WARP>;
    __shared__ typename Reduce::TempStorage temp[WARPS];
    double sum = 0.0;
    const int end = ptr[level + 1];
    for (std::int64_t pos = static_cast<std::int64_t>(ptr[level]) + lane;
         pos < end; pos += WARP) {
        const int row = observation ? observation[pos] : static_cast<int>(pos);
        sum += input[row];
    }
    const double total = Reduce(temp[warp]).Sum(sum);
    if (lane == 0) output[offset + level] = scale[offset + level] * total;
}

__global__ void gi_lsmr_affine_subtract_kernel(const double* first,
                                               const double* second,
                                               double second_scale,
                                               int count,
                                               double* target) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const double a = first[i];
        const double b = second[i];
        target[i] = a - second_scale * b;
    }
}

__global__ void gi_lsmr_update_solution_kernel(double* solution,
                                               double* hbar,
                                               const double* h,
                                               double hbar_scale,
                                               double solution_scale,
                                               int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const double next_hbar = hbar_scale * hbar[i] + h[i];
        hbar[i] = next_hbar;
        solution[i] += solution_scale * next_hbar;
    }
}

__global__ void gi_lsmr_update_h_kernel(double* h,
                                        const double* v,
                                        double h_scale,
                                        int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        h[i] = h_scale * h[i] + v[i];
    }
}

__global__ void gi_lsmr_make_residual_kernel(const double* original,
                                             const double* fitted,
                                             int count,
                                             double* residual) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        residual[i] = original[i] - fitted[i];
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void gi_lsmr_sumsq_partials_kernel(
    const double* values,
    int count,
    double* partials) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < count; i += stride) {
        const double value = values[i];
        local += value * value;
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_sum;
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 1) void gi_lsmr_sumsq_finish_kernel(
    const double* partials,
    int count,
    double* output) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    double local = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        local += partials[i];
    }
    const double total = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        output[0] = total;
    }
}

__global__ void n1_graph_gather_kernel(
    const double* coefficients, int levels, const int* g0, const int* g1,
    const int* g2, int o1, int o2, int n, int ld, double* y, double* X) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = blockIdx.y;
    if (row >= n) return;
    const double* c = coefficients + static_cast<std::size_t>(lane) * levels;
    const double value = (c[g0[row]] + c[o1 + g1[row]]) + c[o2 + g2[row]];
    if (lane == 0) y[row] = value;
    else X[static_cast<std::size_t>(lane - 1) * ld + row] = value;
}

__global__ void n1_graph_pack_sums_kernel(
    const double* y, const double* X, const double* counts, int groups,
    int offset, int levels, double* out) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = blockIdx.y;
    if (g >= groups) return;
    const double mean = lane == 0 ? y[g] :
        X[static_cast<std::size_t>(lane - 1) * groups + g];
    out[static_cast<std::size_t>(lane) * levels + offset + g] = mean * counts[g];
}

struct CudaFeDevice {
    int num_groups = 0;
    int num_levels_present = 0;
    bool is_slope = false;
    bool slope_has_intercept = false;
    bool use_segmented_sums = false;
    // Privatized accumulation: each CUDA block keeps per-group partial sums
    // in shared memory, then flushes to global once per (block,group).
    // Drastically reduces atomic contention when num_groups is small compared
    // to n (e.g., year, a handful of industries).
    bool use_privatized_sums = false;
    std::size_t privatized_shmem_bytes = 0;
    DeviceBuffer<int> gid;
    DeviceBuffer<double> weight_sums;
    DeviceBuffer<double> slope_z;
    DeviceBuffer<double> slope_sum_z;
    DeviceBuffer<double> slope_sum_zz;
    DeviceBuffer<double> sum_y;
    DeviceBuffer<double> sum_zy;
    DeviceBuffer<double> sum_x;
    DeviceBuffer<double> sum_zx;
    DeviceBuffer<double> alpha_y;
    DeviceBuffer<double> alpha_x;
    DeviceBuffer<double> alpha_y_gx;
    DeviceBuffer<double> alpha_x_gx;
    DeviceBuffer<double> alpha_y_ggx;
    DeviceBuffer<double> alpha_x_ggx;
    DeviceBuffer<double> alpha_y_acc_a;
    DeviceBuffer<double> alpha_x_acc_a;
    DeviceBuffer<double> alpha_y_acc_b;
    DeviceBuffer<double> alpha_x_acc_b;
    DeviceBuffer<double> alpha_y_acc_c;
    DeviceBuffer<double> alpha_x_acc_c;
};

struct CudaWorkspace {
    std::vector<CudaFeDevice> fe_dev;
    DeviceBuffer<double> d_y;
    DeviceBuffer<double> d_x;
	DeviceBuffer<double> d_weights;
	DeviceBuffer<double> d_sumsq;
    DeviceBuffer<double> d_y_prev_update;
    DeviceBuffer<double> d_x_prev_update;
    DeviceBuffer<double> d_update_sums;
    DeviceBuffer<double> d_cg_alpha;
    DeviceBuffer<double> d_cg_beta;
	DeviceBuffer<double> d_y_gx;
    DeviceBuffer<double> d_y_ggx;
    DeviceBuffer<double> d_x_gx;
    DeviceBuffer<double> d_x_ggx;
    DeviceBuffer<double> d_stats;
    DeviceBuffer<double> d_y_acc_a;
    DeviceBuffer<double> d_y_acc_b;
    DeviceBuffer<double> d_y_acc_c;
    DeviceBuffer<double> d_x_acc_a;
    DeviceBuffer<double> d_x_acc_b;
    DeviceBuffer<double> d_x_acc_c;

    // group()/individual() CUDA buffers.
    DeviceBuffer<int> gi_group_ptr;
    DeviceBuffer<int> gi_group_individual;
    DeviceBuffer<int> gi_individual_ptr;
    DeviceBuffer<int> gi_individual_group;
    DeviceBuffer<int> gi_color_individuals;
    DeviceBuffer<double> gi_group_scale;
    DeviceBuffer<double> gi_denom;
    DeviceBuffer<double> gi_numer_y;
    DeviceBuffer<double> gi_numer_x;
    DeviceBuffer<double> gi_alpha_y;
    DeviceBuffer<double> gi_alpha_x;

    // Reusable pinned staging for large host<->device copies.
    PinnedStage h_stage;
};

// The joint LSMR vectors are intentionally call-local. They can be much
// larger than the legacy grouped-sweep workspace, and retaining a failed or
// completed solve here would make unrelated later CUDA fits inherit its peak
// allocation. Stack unwinding releases every buffer on all return paths.
struct GiLsmrLocalWorkspace {
    std::vector<DeviceBuffer<int>> standard_group_ptr;
    std::vector<DeviceBuffer<int>> standard_observation_index;
    DeviceBuffer<const int*> gid_ptrs;
    DeviceBuffer<int> offsets;
    DeviceBuffer<double> column_scale;
    DeviceBuffer<double> b;
    DeviceBuffer<double> u;
    DeviceBuffer<double> v;
    DeviceBuffer<double> h;
    DeviceBuffer<double> hbar;
    DeviceBuffer<double> solution;
    DeviceBuffer<double> av;
    DeviceBuffer<double> atu;
    DeviceBuffer<double> scaled;
    DeviceBuffer<double> residual_y;
    DeviceBuffer<double> residual_x;
    DeviceBuffer<double> norm_partials;
    DeviceBuffer<double> norm_scalar;
};

struct AlphaStatePtrs {
    std::vector<double*> y;
    std::vector<double*> x;
};

static thread_local CudaWorkspace cuda_workspace;

bool needs_reghdfe_update_check(const HdfeOptions& options) {
    return options.convergence_criterion == ConvergenceCriterion::Reghdfe ||
           options.convergence_criterion == ConvergenceCriterion::Both;
}

bool convergence_reached(const HdfeOptions& options,
                         double norm_error,
                         double update_error,
                         double tol) {
    const bool norm_ok = norm_error < tol;
    const bool reghdfe_ok = update_error <= tol;
    switch (options.convergence_criterion) {
        case ConvergenceCriterion::Auto:
        case ConvergenceCriterion::NormChange:
            return norm_ok;
        case ConvergenceCriterion::Reghdfe:
            return reghdfe_ok;
        case ConvergenceCriterion::Both:
            return norm_ok && reghdfe_ok;
    }
    return norm_ok;
}

double auto_reghdfe_cg_dominant_share_threshold_cuda() {
    static const double threshold = []() {
        const char* e = std::getenv("XHDFE_AUTO_CG_DOM_SHARE");
        if (e != nullptr) {
            const double v = std::atof(e);
            return v > 0.0 ? v : 0.0;
        }
        return 0.37;
    }();
    return threshold;
}

// ---- GPU-CG adaptive gate ----------------------------------------------------------------
// Route ill-conditioned (high projected MAP-iters) comparable jobs to the EXISTING block-PCG
// even when the dominant-FE-share gate (should_use_cuda_auto_default_reghdfe_cg) declines.
// On such graphs (e.g. the difficult worker+firm+year DGP) the Irons-Tuck MAP comparable path
// converges to a point ~2e-7 off reghdfe and needs ~530 sweeps, whereas the block-PCG matches
// reghdfe to ~1e-9 in fewer iterations. This mirrors the CPU Schwarz auto-gate: it reuses the
// SAME XHDFE_SCHWARZ_NMIN / XHDFE_SCHWARZ_MSWITCH knobs and the SAME plain-GS difficulty probe
// so CPU and GPU switch on exactly the same datasets. It is comparable-only and from_auto-only
// BY DESIGN: fast mode keeps the already-optimal IT-MAP (where unpreconditioned CG would need
// MORE iterations, i.e. a timing regression), and explicit absorption-method choices are
// honored. Disable independently with XHDFE_GPU_CG_AUTO=0.
// OPT-IN, default OFF. The GPU-CG substitution is a net win only in a narrow regime: ~1M-scale
// pathological 3-FE graphs, where the block-PCG converges in fewer iterations than IT-MAP AND
// closes the ~2e-7 comparable-mode coefficient gap (IT-MAP comparable stalls there; the CPU
// comparable path reaches ~1e-13 via Schwarz). It REGRESSES at larger scale -- the native 10M
// difficult graph needs ~1989 unpreconditioned-CG iters vs IT-MAP's ~530, i.e. ~4x slower -- and
// the host-side difficulty probe adds O(n) cost even to in-band datasets that end up declining.
// Per the project's strict non-regression mandate it is therefore disabled by default and must
// be requested explicitly with XHDFE_GPU_CG_AUTO=1. When enabled it is still bounded by an
// upper-n guard (XHDFE_GPU_CG_NMAX) so opted-in users never hit the large-scale regression.
constexpr int kCudaSmallCgMaxRows = 20000;

double cuda_accel_occupancy_threshold() {
    static const double value = []() {
        const char* e = std::getenv("XHDFE_ACCEL_OCC");
        return e != nullptr ? std::atof(e) : 12.0;
    }();
    return value;
}

bool should_use_cuda_sparse_four_fe_cg(const std::vector<GpuFeInput>& fe_inputs,
                                      int n, const Eigen::VectorXd* weights,
                                      const HdfeOptions& options, bool store_alphas) {
    if (!options.from_auto || options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable ||
        store_alphas || options.retain_fixed_effects || weights != nullptr ||
        fe_inputs.size() != 4 || n <= kCudaSmallCgMaxRows || n > 2000000 ||
        !(options.tol > 0.0 && options.tol <= 1.0e-10)) {
        return false;
    }
    bool sparse = false;
    for (const auto& fe : fe_inputs) {
        if (fe.is_slope) return false;
        const int levels = fe.num_levels_present > 0 ? fe.num_levels_present : fe.num_groups;
        if (levels > 0 && static_cast<double>(n) / levels <= cuda_accel_occupancy_threshold())
            sparse = true;
    }
    return sparse;
}

bool gpu_cg_auto_enabled() {
    const char* e = std::getenv("XHDFE_GPU_CG_AUTO");
    return e != nullptr && e[0] != '0';
}
int64_t gpu_cg_nmax() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_GPU_CG_NMAX");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(2000000);
    }();
    return v;
}
int64_t schwarz_auto_nmin_cuda() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_NMIN");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(750000);
    }();
    return v;
}
double schwarz_auto_mswitch_cuda() {
    static const double v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_MSWITCH");
        return e ? std::atof(e) : 50000.0;
    }();
    return v;
}
bool gpu_cg_diag_enabled() { return std::getenv("XHDFE_GPU_CG_DIAG") != nullptr; }
// Bound the (serial, host-side) difficulty probe to a stride-sampled subset so it never adds
// meaningful wall-clock on very large datasets that the gate ends up declining (e.g. a 46M-row
// real panel: the full probe costs ~5.5s, the 2M-row sample ~0.25s). n at or below the cap runs
// the full probe unchanged. The switch/no-switch margin is large (~50x: easy panels project a
// few thousand iters, pathological graphs hundreds of thousands), so a uniform edge sub-sample
// does not flip the decision; re-validated on ready (->MAP) and difficult 1M/10M (->CG).
int64_t gpu_cg_probe_cap() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_GPU_CG_PROBE_CAP");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(2000000);
    }();
    return v;
}

// Plain-Gauss-Seidel MAP-difficulty probe on a COPY of host y. Identical in form to the CPU
// schwarz_probe_projected_iters so the projected-iters scale and the mswitch calibration carry
// over unchanged. Returns +inf when not contracting. Does NOT mutate the caller's y.
double schwarz_probe_projected_iters_cuda(const std::vector<const int*>& g,
                                          const std::vector<int>& ng,
                                          const double* y, int n, double tol) {
    const int K0 = 5, W = 4, NS = K0 + W;
    const int D = static_cast<int>(g.size());
    std::vector<std::vector<double>> cnt(D), sum(D);
    for (int d = 0; d < D; ++d) {
        cnt[d].assign(ng[d], 0.0);
        sum[d].assign(ng[d], 0.0);
        for (int i = 0; i < n; ++i) cnt[d][g[d][i]] += 1.0;
    }
    std::vector<double> yp(y, y + n);
    auto norm2 = [&]() {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += yp[i] * yp[i];
        return std::sqrt(s);
    };
    double prevn = std::max(norm2(), 1e-300);
    double logprod = 0.0;
    int nr = 0;
    for (int it = 0; it < NS; ++it) {
        for (int d = 0; d < D; ++d) {
            auto& s = sum[d];
            std::fill(s.begin(), s.end(), 0.0);
            const int* gd = g[d];
            for (int i = 0; i < n; ++i) s[gd[i]] += yp[i];
            for (size_t j = 0; j < s.size(); ++j) s[j] = cnt[d][j] > 0.0 ? s[j] / cnt[d][j] : 0.0;
            for (int i = 0; i < n; ++i) yp[i] -= s[gd[i]];
        }
        double nn = norm2();
        if (it >= K0 && prevn > 1e-300) {
            double r = nn / prevn;
            if (r > 0.0 && r < 1.0) { logprod += std::log(r); nr++; }
        }
        prevn = nn;
    }
    if (nr == 0) return 1e18;
    const double rho_bar = std::exp(logprod / nr);
    if (!(rho_bar > 0.0 && rho_bar < 1.0)) return 1e18;
    const double proj = std::log(tol) / std::log(rho_bar);
    return proj > 0.0 ? proj : 1e18;
}

bool should_use_cuda_adaptive_cg(const std::vector<GpuFeInput>& fe_inputs,
                                 int n,
                                 const Eigen::VectorXd* weights,
                                 const HdfeOptions& options,
                                 bool store_alphas,
                                 const Eigen::Ref<const Eigen::VectorXd>& y) {
    if (!gpu_cg_auto_enabled()) {
        return false;  // opt-in only (default OFF) -- see gpu_cg_auto_enabled() rationale
    }
    // Comparable + auto-default only (mirror should_use_cuda_auto_default_reghdfe_cg), and only
    // when the absorption method was left on Auto (from_auto) so explicit choices are honored.
    if (options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable ||
        !options.from_auto) {
        return false;
    }
    if (store_alphas || options.retain_fixed_effects || weights != nullptr || n <= 0) {
        return false;
    }
    const std::size_t dims = fe_inputs.size();
    if (dims < 2 || dims > 3) {
        return false;
    }
    if (static_cast<int64_t>(n) < schwarz_auto_nmin_cuda()) {
        return false;
    }
    if (static_cast<int64_t>(n) > gpu_cg_nmax()) {
        return false;  // above the validated-win regime; IT-MAP avoids the CG-at-scale regression
    }
    std::vector<const int*> g;
    std::vector<int> ng;
    g.reserve(dims);
    ng.reserve(dims);
    for (const auto& fi : fe_inputs) {
        if (!fi.group_ids || fi.num_groups <= 0) {
            return false;
        }
        g.push_back(fi.group_ids);
        ng.push_back(fi.num_groups);
    }
    const double eff_tol = effective_absorption_tolerance(options);
    // Stride-sample the probe input when n exceeds the cap so the serial host probe stays cheap.
    const int64_t cap = gpu_cg_probe_cap();
    int m = n;
    std::vector<const int*> gp = g;
    const double* yptr = y.data();
    std::vector<int> gs_storage;
    std::vector<const int*> gs_ptrs;
    std::vector<double> ys;
    if (cap > 0 && static_cast<int64_t>(n) > cap) {
        const int stride = static_cast<int>(static_cast<int64_t>(n) / cap);
        m = (n + stride - 1) / stride;
        ys.resize(static_cast<size_t>(m));
        gs_storage.resize(static_cast<size_t>(dims) * static_cast<size_t>(m));
        for (size_t d = 0; d < dims; ++d) {
            int* dst = gs_storage.data() + d * static_cast<size_t>(m);
            const int* src = g[d];
            int k = 0;
            for (int i = 0; i < n && k < m; i += stride) dst[k++] = src[i];
        }
        int k = 0;
        for (int i = 0; i < n && k < m; i += stride) ys[k++] = yptr[i];
        gs_ptrs.resize(dims);
        for (size_t d = 0; d < dims; ++d) gs_ptrs[d] = gs_storage.data() + d * static_cast<size_t>(m);
        gp = gs_ptrs;
        yptr = ys.data();
    }
    const double proj = schwarz_probe_projected_iters_cuda(gp, ng, yptr, m, eff_tol);
    const bool go = proj > schwarz_auto_mswitch_cuda();
    if (gpu_cg_diag_enabled()) {
        std::fprintf(stderr,
                     "[gpu-cg-gate] n=%d probe_rows=%d ndim=%zu proj_iters=%.0f mswitch=%.0f nmin=%lld -> %s\n",
                     n, m, dims, proj, schwarz_auto_mswitch_cuda(),
                     static_cast<long long>(schwarz_auto_nmin_cuda()), go ? "CG" : "MAP");
    }
    return go;
}

bool should_use_cuda_auto_default_reghdfe_cg(const std::vector<GpuFeInput>& fe_inputs,
                                             int n,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options,
                                             bool store_alphas) {
    if (options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable) {
        return false;
    }
    if (store_alphas || options.retain_fixed_effects || weights != nullptr ||
        fe_inputs.size() < 3 || n <= 0) {
        return false;
    }

    const double threshold = auto_reghdfe_cg_dominant_share_threshold_cuda();
    if (!(threshold > 0.0 && threshold <= 1.0)) {
        return false;
    }

    double max_share = 0.0;
    const double denom = static_cast<double>(n);
    for (const auto& fi : fe_inputs) {
        if (!fi.weight_sums || fi.num_groups <= 0) {
            return false;
        }
        double max_weight = 0.0;
        for (int g = 0; g < fi.num_groups; ++g) {
            max_weight = std::max(max_weight, fi.weight_sums[g]);
        }
        max_share = std::max(max_share, max_weight / denom);
    }
    return max_share >= threshold;
}

bool should_use_cuda_pathlike_cg(const std::vector<GpuFeInput>& fe_inputs,
                                 int n,
                                 const Eigen::VectorXd* weights,
                                 const HdfeOptions& options,
                                 bool store_alphas) {
    if (store_alphas || options.retain_fixed_effects ||
        fe_inputs.size() != 2 || n <= 0 || n > 20000) {
        return false;
    }
    std::int64_t total_levels = 0;
    for (const auto& fe : fe_inputs) {
        if (fe.is_slope || fe.num_groups <= 0) {
            return false;
        }
        total_levels += static_cast<std::int64_t>(fe.num_groups);
    }
    if (total_levels < static_cast<std::int64_t>(n) - 2 ||
        total_levels > static_cast<std::int64_t>(n) + 2) {
        return false;
    }
    if (!weights) {
        return true;
    }
    if (weights->size() != n) {
        return false;
    }
    // This gate is deliberately bit-exact.  Treating nearly-unit weights as
    // unitary would silently change a weighted estimator; the small-n
    // path-like cap keeps the one-time validation outside every large hot path.
    constexpr std::uint64_t kOneBits = UINT64_C(0x3ff0000000000000);
    for (Eigen::Index i = 0; i < weights->size(); ++i) {
        std::uint64_t bits = 0;
        const double value = (*weights)[i];
        std::memcpy(&bits, &value, sizeof(bits));
        if (bits != kOneBits) {
            return false;
        }
    }
    return true;
}

bool should_use_cuda_reghdfe_cg(const std::vector<GpuFeInput>& fe_inputs,
                                int n,
                                const Eigen::VectorXd* weights,
                                const HdfeOptions& options,
                                bool store_alphas) {
    const bool explicit_reghdfe =
        options.convergence_criterion != ConvergenceCriterion::Auto &&
        options.convergence_criterion != ConvergenceCriterion::NormChange;
    const bool auto_default =
        should_use_cuda_auto_default_reghdfe_cg(fe_inputs, n, weights, options,
                                                store_alphas);
    const bool pathlike =
        should_use_cuda_pathlike_cg(fe_inputs, n, weights, options,
                                    store_alphas);
    const bool sparse_four_fe = should_use_cuda_sparse_four_fe_cg(
        fe_inputs, n, weights, options, store_alphas);
    if (!explicit_reghdfe && !auto_default && !pathlike && !sparse_four_fe) {
        return false;
    }
    const std::size_t dims = fe_inputs.size();
    if (store_alphas || options.retain_fixed_effects || dims < 2) {
        return false;
    }
    const char* disabled = std::getenv("XHDFE_CUDA_REGHDFE_CG");
    if (disabled != nullptr && disabled[0] == '0') {
        return false;
    }
    return weights == nullptr || pathlike;
}

struct ExactEdgeUnionFind {
    std::vector<int> parent;
    std::vector<unsigned char> rank;

    explicit ExactEdgeUnionFind(int n)
        : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return;
        }
        const std::size_t ia = static_cast<std::size_t>(ra);
        const std::size_t ib = static_cast<std::size_t>(rb);
        if (rank[ia] < rank[ib]) {
            parent[ia] = rb;
        } else if (rank[ia] > rank[ib]) {
            parent[ib] = ra;
        } else {
            parent[ib] = ra;
            ++rank[ia];
        }
    }
};

struct TreeDuplicateEdgeGroups {
    std::vector<int> obs_edge;
    std::vector<int> edge_counts;
    int total_levels = 0;
    int components = 0;
};

bool build_tree_duplicate_edge_groups(const std::vector<GpuFeInput>& fe_inputs,
                                      int n,
                                      TreeDuplicateEdgeGroups& out) {
    out = TreeDuplicateEdgeGroups{};
    if (fe_inputs.size() != 2 || n <= 0 || n > 20000) {
        return false;
    }
    const int g0 = fe_inputs[0].num_groups;
    const int g1 = fe_inputs[1].num_groups;
    if (g0 <= 0 || g1 <= 0 || fe_inputs[0].group_ids == nullptr ||
        fe_inputs[1].group_ids == nullptr) {
        return false;
    }
    const int total_levels = g0 + g1;
    if (total_levels < n - 2 || total_levels > n + 2) {
        return false;
    }

    ExactEdgeUnionFind uf(total_levels);
    std::unordered_map<std::uint64_t, int> edge_to_group;
    edge_to_group.reserve(static_cast<std::size_t>(n));
    out.obs_edge.resize(static_cast<std::size_t>(n));
    out.edge_counts.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const int a = fe_inputs[0].group_ids[i];
        const int b = fe_inputs[1].group_ids[i];
        if (a < 0 || a >= g0 || b < 0 || b >= g1) {
            out = TreeDuplicateEdgeGroups{};
            return false;
        }
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
            static_cast<std::uint32_t>(b);
        auto it = edge_to_group.find(key);
        if (it == edge_to_group.end()) {
            const int e = static_cast<int>(out.edge_counts.size());
            edge_to_group.emplace(key, e);
            out.obs_edge[static_cast<std::size_t>(i)] = e;
            out.edge_counts.push_back(1);
            uf.unite(a, g0 + b);
        } else {
            out.obs_edge[static_cast<std::size_t>(i)] = it->second;
            ++out.edge_counts[static_cast<std::size_t>(it->second)];
        }
    }

    std::unordered_map<int, int> roots;
    roots.reserve(static_cast<std::size_t>(total_levels));
    for (int node = 0; node < total_levels; ++node) {
        roots.emplace(uf.find(node), 1);
    }
    out.components = static_cast<int>(roots.size());
    out.total_levels = total_levels;

    const int unique_edges = static_cast<int>(out.edge_counts.size());
    const int duplicate_rows = n - unique_edges;
    if (duplicate_rows <= 0) {
        out = TreeDuplicateEdgeGroups{};
        return false;
    }
    if (unique_edges != total_levels - out.components) {
        out = TreeDuplicateEdgeGroups{};
        return false;
    }
    return true;
}

__global__ void weighted_two_by_two_apply_kernel(
    double* y,double* X,int n,int cols,const unsigned char* cells,
    const weighted_two_by_two::Cell* values,int* bad_output) {
    const std::size_t row=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=static_cast<std::size_t>(n)) return;const int i=static_cast<int>(row);
    const int c=cells[i];
    const double within=weighted_two_by_two::apply(y[i],values[c]);y[i]=within;
    if(!weighted_two_by_two::finite(within)) atomicExch(bad_output,1);
    for(int j=0;j<cols;++j) {
        const std::size_t at=static_cast<std::size_t>(j)*n+i;
        const double value=weighted_two_by_two::apply(X[at],values[static_cast<std::size_t>(j+1)*4+c]);
        X[at]=value;if(!weighted_two_by_two::finite(value)) atomicExch(bad_output,1);
    }
}

bool absorb_weighted_two_by_two_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const weighted_two_by_two::Plan& plan,
    const HdfeOptions& options,AbsorptionResult& result) {
    DeviceBuffer<double> device_y(static_cast<std::size_t>(plan.n));
    DeviceBuffer<double> device_X(static_cast<std::size_t>(plan.n)*plan.cols);
    DeviceBuffer<unsigned char> device_cell(plan.cell.size());
    DeviceBuffer<weighted_two_by_two::Cell> device_values(plan.values.size());
    DeviceBuffer<int> device_bad(1);
    cuda_check(cudaMemcpy(device_y.data(),y.data(),sizeof(double)*plan.n,cudaMemcpyHostToDevice),
        "cudaMemcpy 2x2 outcome failed");
    if(plan.cols) cuda_check(cudaMemcpy(device_X.data(),X.data(),sizeof(double)*static_cast<std::size_t>(plan.n)*plan.cols,cudaMemcpyHostToDevice),
        "cudaMemcpy 2x2 regressors failed");
    cuda_check(cudaMemcpy(device_cell.data(),plan.cell.data(),plan.cell.size(),cudaMemcpyHostToDevice),
        "cudaMemcpy 2x2 cell ids failed");
    cuda_check(cudaMemcpy(device_values.data(),plan.values.data(),sizeof(weighted_two_by_two::Cell)*plan.values.size(),cudaMemcpyHostToDevice),
        "cudaMemcpy 2x2 cell parameters failed");
    cuda_check(cudaMemset(device_bad.data(),0,sizeof(int)),"cudaMemset 2x2 status failed");
    const int blocks=plan.n/kBlockSize+(plan.n%kBlockSize!=0);
    weighted_two_by_two_apply_kernel<<<blocks,kBlockSize>>>(device_y.data(),device_X.data(),plan.n,plan.cols,
        device_cell.data(),device_values.data(),device_bad.data());
    cuda_check(cudaGetLastError(),"2x2 weighted projection kernel launch failed");
    int bad=0;
    cuda_check(cudaMemcpy(&bad,device_bad.data(),sizeof(int),cudaMemcpyDeviceToHost),
        "cudaMemcpy 2x2 projection status failed");
    if(bad) throw std::runtime_error("2x2 weighted CUDA projection exceeds the finite output range; no estimates returned");
    AbsorptionResult projected;projected.y_tilde.resize(plan.n);projected.X_tilde.resize(plan.n,plan.cols);
    cuda_check(cudaMemcpy(projected.y_tilde.data(),device_y.data(),sizeof(double)*plan.n,cudaMemcpyDeviceToHost),
        "cudaMemcpy 2x2 within outcome failed");
    if(plan.cols) cuda_check(cudaMemcpy(projected.X_tilde.data(),device_X.data(),sizeof(double)*static_cast<std::size_t>(plan.n)*plan.cols,cudaMemcpyDeviceToHost),
        "cudaMemcpy 2x2 within regressors failed");
    weighted_two_by_two::metadata(plan,options,projected);
    projected.gpu_used=true;projected.gpu_attempted=true;projected.gpu_status_code=1;
    projected.gpu_absorption_converged=true;projected.gpu_absorption_iterations=projected.iterations;
    result=std::move(projected);return true;
}

__global__ void additive_cell_apply_kernel(double* y,double* X,int n,int cols,int cells,
    const unsigned char* cell,const double* means) {
    const std::size_t row=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=static_cast<std::size_t>(n)) return;
    const int c=cell[row];y[row]=__dsub_rn(y[row],means[c]);
    for(int j=0;j<cols;++j) {
        const auto at=static_cast<std::size_t>(j)*n+row;
        X[at]=__dsub_rn(X[at],means[static_cast<std::size_t>(j+1)*cells+c]);
    }
}

bool absorb_additive_cells_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const additive_cell_projection::Plan& plan,
    const HdfeOptions& options,AbsorptionResult& result) {
    DeviceBuffer<double> device_y(static_cast<std::size_t>(plan.n));
    DeviceBuffer<double> device_X(static_cast<std::size_t>(plan.n)*plan.cols);
    DeviceBuffer<unsigned char> device_cell(plan.cell.size());
    DeviceBuffer<double> device_means(plan.means.size());
    cuda_check(cudaMemcpy(device_y.data(),y.data(),sizeof(double)*plan.n,cudaMemcpyHostToDevice),
        "cudaMemcpy additive-cell outcome failed");
    if(plan.cols) cuda_check(cudaMemcpy(device_X.data(),X.data(),sizeof(double)*static_cast<std::size_t>(plan.n)*plan.cols,cudaMemcpyHostToDevice),
        "cudaMemcpy additive-cell regressors failed");
    cuda_check(cudaMemcpy(device_cell.data(),plan.cell.data(),plan.cell.size(),cudaMemcpyHostToDevice),
        "cudaMemcpy additive-cell ids failed");
    cuda_check(cudaMemcpy(device_means.data(),plan.means.data(),sizeof(double)*plan.means.size(),cudaMemcpyHostToDevice),
        "cudaMemcpy additive-cell means failed");
    const int blocks=plan.n/kBlockSize+(plan.n%kBlockSize!=0);
    additive_cell_apply_kernel<<<blocks,kBlockSize>>>(device_y.data(),device_X.data(),plan.n,plan.cols,
        plan.cells,device_cell.data(),device_means.data());
    cuda_check(cudaGetLastError(),"additive-cell projection kernel launch failed");
    AbsorptionResult projected;projected.y_tilde.resize(plan.n);projected.X_tilde.resize(plan.n,plan.cols);
    cuda_check(cudaMemcpy(projected.y_tilde.data(),device_y.data(),sizeof(double)*plan.n,cudaMemcpyDeviceToHost),
        "cudaMemcpy additive-cell within outcome failed");
    if(plan.cols) cuda_check(cudaMemcpy(projected.X_tilde.data(),device_X.data(),sizeof(double)*static_cast<std::size_t>(plan.n)*plan.cols,cudaMemcpyDeviceToHost),
        "cudaMemcpy additive-cell within regressors failed");
    additive_cell_projection::metadata(plan,options,projected);
    projected.gpu_used=true;projected.gpu_attempted=true;projected.gpu_status_code=1;
    projected.gpu_absorption_converged=true;projected.gpu_absorption_iterations=projected.iterations;
    result=std::move(projected);return true;
}

bool absorb_tree_duplicate_edges_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                                      const Eigen::Ref<const Eigen::MatrixXd>& X,
                                      const TreeDuplicateEdgeGroups& edge_groups,
                                      const std::vector<GpuFeInput>& fe_inputs,
                                      const HdfeOptions& options,
                                      AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    const int unique_edges = static_cast<int>(edge_groups.edge_counts.size());
    if (n <= 0 || unique_edges <= 0 || static_cast<int>(edge_groups.obs_edge.size()) != n ||
        ld != n) {
        result.converged = false;
        return false;
    }

    try {
        cudaGetLastError();
        DeviceBuffer<double> d_y;
        DeviceBuffer<double> d_x;
        DeviceBuffer<int> d_edge_group;
        DeviceBuffer<int> d_edge_counts;
        DeviceBuffer<double> d_sum_y;
        DeviceBuffer<double> d_sum_x;

        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), y.data(), sizeof(double) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree y failed");
        if (cols > 0) {
            d_x.allocate(static_cast<std::size_t>(ld) * cols);
            cuda_check(cudaMemcpy(d_x.data(), X.data(), sizeof(double) * ld * cols,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy exact tree X failed");
        }

        d_edge_group.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_edge_group.data(), edge_groups.obs_edge.data(),
                              sizeof(int) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree edge ids failed");
        d_edge_counts.allocate(static_cast<std::size_t>(unique_edges));
        cuda_check(cudaMemcpy(d_edge_counts.data(), edge_groups.edge_counts.data(),
                              sizeof(int) * unique_edges, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree edge counts failed");

        d_sum_y.allocate(static_cast<std::size_t>(unique_edges));
        cuda_check(cudaMemset(d_sum_y.data(), 0, sizeof(double) * unique_edges),
                   "cudaMemset exact tree sum_y failed");
        if (cols > 0) {
            d_sum_x.allocate(static_cast<std::size_t>(unique_edges) * cols);
            cuda_check(cudaMemset(d_sum_x.data(), 0,
                                  sizeof(double) * unique_edges * cols),
                       "cudaMemset exact tree sum_x failed");
        }

        const int blocks = (n + kBlockSize - 1) / kBlockSize;
        accumulate_edge_sums_kernel<<<blocks, kBlockSize>>>(
            d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld, unique_edges,
            d_edge_group.data(), d_sum_y.data(), cols > 0 ? d_sum_x.data() : nullptr);
        cuda_check(cudaGetLastError(), "exact tree edge-sum kernel launch failed");

        // WP3 shadow certificate on the exact-tree early path. This path
        // never uploads per-dimension group ids (it works on fused edge ids),
        // so in shadow mode — and only then — the dims' gids and weight_sums
        // are uploaded here just for the verifier. Orig/Op capture must
        // happen BEFORE the residual kernel mutates d_y/d_x in place.
        struct CertShadowGuard {
            CudaCertShadowHandle* h = nullptr;
            ~CertShadowGuard() {
                if (h != nullptr) {
                    cuda_cert_shadow_abort(h);
                }
            }
        } cert_shadow;
        std::vector<DeviceBuffer<int>> shadow_gid_bufs;
        std::vector<DeviceBuffer<double>> shadow_wsum_bufs;
        if (cuda_cert_shadow_enabled()) {
            bool shadow_ok = true;
            std::vector<const int*> gids;
            std::vector<const double*> wsums;
            std::vector<int> groups;
            shadow_gid_bufs.resize(fe_inputs.size());
            shadow_wsum_bufs.resize(fe_inputs.size());
            for (std::size_t dim = 0; shadow_ok && dim < fe_inputs.size(); ++dim) {
                const auto& fe = fe_inputs[dim];
                if (fe.group_ids == nullptr || fe.num_groups <= 0) {
                    shadow_ok = false;
                    break;
                }
                shadow_gid_bufs[dim].allocate(static_cast<std::size_t>(n));
                cuda_check(cudaMemcpy(shadow_gid_bufs[dim].data(), fe.group_ids,
                                      sizeof(int) * n, cudaMemcpyHostToDevice),
                           "cudaMemcpy shadow gid failed");
                shadow_wsum_bufs[dim].allocate(
                    static_cast<std::size_t>(fe.num_groups));
                cuda_check(cudaMemcpy(shadow_wsum_bufs[dim].data(),
                                      fe.weight_sums,
                                      sizeof(double) * fe.num_groups,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy shadow weight_sums failed");
                gids.push_back(shadow_gid_bufs[dim].data());
                wsums.push_back(shadow_wsum_bufs[dim].data());
                groups.push_back(fe.num_groups);
            }
            if (shadow_ok) {
                cert_shadow.h = cuda_cert_shadow_begin(
                    d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld,
                    nullptr, gids.data(), wsums.data(), groups.data(),
                    static_cast<int>(fe_inputs.size()));
            }
        }

        apply_edge_residuals_kernel<<<blocks, kBlockSize>>>(
            d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld, unique_edges,
            d_edge_group.data(), d_edge_counts.data(), d_sum_y.data(),
            cols > 0 ? d_sum_x.data() : nullptr);
        cuda_check(cudaGetLastError(), "exact tree residual kernel launch failed");

        if (cert_shadow.h != nullptr) {
            CudaCertShadowResult shadow{};
            const double shadow_eff = effective_absorption_tolerance(options);
            const double shadow_limit =
                std::max(8.0 * std::max(0.0, shadow_eff),
                         64.0 * std::numeric_limits<double>::epsilon());
            const bool shadow_ok = cuda_cert_shadow_finish(
                cert_shadow.h, d_y.data(), cols > 0 ? d_x.data() : nullptr,
                shadow_limit, &shadow);
            cert_shadow.h = nullptr;
            static const char* kVerdictNames[4] = {"PASS", "FAIL", "INDET",
                                                   "ERROR"};
            std::fprintf(
                stderr,
                "cuda_cert_shadow ok=%d verdict=%s host_like_pass=%d "
                "rel=%.17e rel_perdim=%.17e r_lo=%.17e r_up=%.17e L=%.17e "
                "wsum_maxdiff=%.3e wsum_tol=%.3e wsum_ok=%d "
                "empty_groups=%lld dims=%d rhs=%d "
                "begin_ms=%.3f finish_ms=%.3f\n",
                shadow_ok ? 1 : 0,
                kVerdictNames[shadow.verdict & 3], shadow.host_like_pass,
                shadow.rel_max, shadow.rel_max_perdim, shadow.r_lower,
                shadow.r_upper, shadow_limit, shadow.wsum_maxdiff,
                shadow.wsum_tol, shadow.wsum_ok,
                shadow.empty_groups, shadow.dims, shadow.rhs,
                shadow.shadow_begin_ms, shadow.shadow_finish_ms);
        }

        result.y_tilde.resize(n);
        cuda_check(cudaMemcpy(result.y_tilde.data(), d_y.data(), sizeof(double) * n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy exact tree y_tilde failed");
        if (cols > 0) {
            result.X_tilde.resize(ld, cols);
            cuda_check(cudaMemcpy(result.X_tilde.data(), d_x.data(),
                                  sizeof(double) * ld * cols, cudaMemcpyDeviceToHost),
                       "cudaMemcpy exact tree X_tilde failed");
        } else {
            result.X_tilde.resize(ld, 0);
        }

        result.fe_levels.clear();
        result.fe_levels.reserve(fe_inputs.size());
        for (const auto& fe : fe_inputs) {
            result.fe_levels.push_back(fe.num_levels_present);
        }
        result.sweep_order_used = {0, 1};
        result.iterations = 1;
        result.converged = true;
        return true;
    } catch (const std::exception& e) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] exact tree absorption failed: %s\n", e.what());
        }
        cudaGetLastError();
        result.converged = false;
        result.iterations = 0;
        return false;
    } catch (...) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] exact tree absorption failed: unknown\n");
        }
        cudaGetLastError();
        result.converged = false;
        result.iterations = 0;
        return false;
    }
}

struct FeProfileStats {
    int num_groups = 0;
    int num_levels_present = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double cv = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double singleton_ratio = 0.0;
};

FeProfileStats profile_fe(const GpuFeInput& fe, bool unit_weights) {
    FeProfileStats stats;
    const int groups = fe.num_groups;
    stats.num_groups = groups;
    stats.num_levels_present = fe.num_levels_present;
    if (groups <= 0 || fe.weight_sums == nullptr) {
        return stats;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    int singletons = 0;
    constexpr double kSingletonTol = 1e-12;
    for (int g = 0; g < groups; ++g) {
        const double v = fe.weight_sums[g];
        sum += v;
        sumsq += v * v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        if (v <= 1.0 + kSingletonTol) {
            ++singletons;
        }
    }
    stats.mean = sum / static_cast<double>(groups);
    const double var =
        std::max(0.0, sumsq / static_cast<double>(groups) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(var);
    stats.cv = stats.mean > 0.0 ? stats.stddev / stats.mean : 0.0;
    stats.min_val = std::isfinite(min_val) ? min_val : 0.0;
    stats.max_val = std::isfinite(max_val) ? max_val : 0.0;
    stats.singleton_ratio = unit_weights
                                ? static_cast<double>(singletons) / static_cast<double>(groups)
                                : 0.0;
    return stats;
}

enum class FeDifficulty { Easy, Medium, Hard, Extreme };

FeDifficulty classify_problem(const std::vector<FeProfileStats>& profiles) {
    const int dims = static_cast<int>(profiles.size());
    double max_singleton = 0.0;
    double max_cv = 0.0;
    for (const auto& p : profiles) {
        max_singleton = std::max(max_singleton, p.singleton_ratio);
        max_cv = std::max(max_cv, p.cv);
    }
    if (dims >= 4 && (max_singleton >= 0.30 || max_cv >= 2.0)) {
        return FeDifficulty::Extreme;
    }
    if (dims >= 4 || max_singleton >= 0.20 || max_cv >= 1.5) {
        return FeDifficulty::Hard;
    }
    if (dims >= 3 || max_singleton >= 0.10 || max_cv >= 1.0) {
        return FeDifficulty::Medium;
    }
    return FeDifficulty::Easy;
}

std::vector<std::size_t> select_two_largest_dims(const std::vector<FeProfileStats>& profiles) {
    if (profiles.size() < 2) {
        return {};
    }
    int best = -1;
    int second = -1;
    int best_groups = -1;
    int second_groups = -1;
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        const int groups = profiles[i].num_groups;
        if (groups > best_groups) {
            second = best;
            second_groups = best_groups;
            best = static_cast<int>(i);
            best_groups = groups;
        } else if (groups > second_groups) {
            second = static_cast<int>(i);
            second_groups = groups;
        }
    }
    if (best < 0 || second < 0) {
        return {};
    }
    std::vector<std::size_t> dims;
    dims.reserve(2);
    dims.push_back(static_cast<std::size_t>(best));
    dims.push_back(static_cast<std::size_t>(second));
    return dims;
}

std::vector<std::size_t> filter_order(const std::vector<std::size_t>& order,
                                      const std::vector<std::size_t>& dims) {
    if (order.empty() || dims.empty()) {
        return {};
    }
    std::vector<uint8_t> wanted(order.size(), 0);
    for (const std::size_t dim : dims) {
        if (dim < wanted.size()) {
            wanted[dim] = 1;
        }
    }
    std::vector<std::size_t> filtered;
    filtered.reserve(dims.size());
    for (const std::size_t dim : order) {
        if (dim < wanted.size() && wanted[dim]) {
            filtered.push_back(dim);
        }
    }
    return filtered;
}

// ---- F2-A: hand-over of the three-FE Comparable completion ("tail") ----
// The device LSMR that completes the projection after the IT sweeps has only a
// diagonal preconditioner. Where the graph mixes slowly it needs thousands of
// iterations per right-hand side (pf_difficult_10m_3fe: about 7 000 x 11, 86 s),
// while the host MLSMR with its graph preconditioner needs tens. A probe of the
// device solve predicts its remaining work; when that work is long and the
// CPU's own route selector would use MLSMR on this fit, the host MLSMR finishes
// the projection from the same IT state. Calibration (phase 0, 07nov2026):
// benchmarks/_out/perf_fereg_gap_20261106/f2_phase0/.
//
// F2-B (08nov2026): the cold device LSMR that restarts a three-FE Comparable
// fit after a broken CG stage (absorb_fixed_effects_cuda) is probed by the same
// rule; on a long prediction and an MLSMR vote, the host MLSMR solves the fit
// from the original data (never from the CG iterate), within the budget left
// by the sweeps, the CG stage and the probe.
//
// XHDFE_CUDA_TAIL_HANDOVER, read once at the tail of each absorption call and
// once at a restart: unset or any other value = adaptive (default); a value
// beginning with '0' = the 2.30.1 device tail and restart, without probe and
// with the 2.30.1 iteration accounting; "force" = hand over after the first
// device iteration, whatever the prediction and the CPU selector. "force" is
// for tests and diagnostics only; it is reported as host-mlsmr like any
// hand-over.
enum class CudaTailMode { Adaptive, Off, Force };

CudaTailMode cuda_tail_handover_mode() {
    const char* e = std::getenv("XHDFE_CUDA_TAIL_HANDOVER");
    if (e != nullptr && e[0] == '0') return CudaTailMode::Off;
    if (e != nullptr && std::strcmp(e, "force") == 0) return CudaTailMode::Force;
    return CudaTailMode::Adaptive;
}

// P8 (A1; design note section 6): the Comparable three-FE completion by the
// device LSMR is open to a weights vector when the public request is
// reghdfe-comparable with the Auto method and the Auto criterion (provenance:
// options.comparable_intent, options.auto_intent; the weighted native repairs
// run under NormChange and stay out). With weights the F2 hand-over is never
// reached, not even under XHDFE_CUDA_TAIL_HANDOVER=force: the completion runs
// without its probe (the host MLSMR and the F2-B restart solve the unweighted
// problem). Opt-in in 2.31.0 (its cost on difficult weighted data exceeded the
// release's cost limit): XHDFE_P8_CUDA_WEIGHTED_COMPLETION=1 enables it; unset or
// any other value keeps the unweighted-only completion (read once per process).
bool cuda_weighted_completion_applies(const HdfeOptions& options) {
    static const bool enabled = [] {
        const char* e = std::getenv("XHDFE_P8_CUDA_WEIGHTED_COMPLETION");
        return e != nullptr && e[0] == '1';
    }();
    return enabled && options.comparable_intent && options.auto_intent &&
           options.convergence_criterion == ConvergenceCriterion::Auto;
}

// The rule. On the first right-hand side that reaches k = 64 device iterations
// without converging, at k = 64, 128, 256, ... of that right-hand side:
//   q     = (norm_ar_k / norm_ar_{k/2})^(2/k)   (norm_ar = |zetabar|, monotone)
//   m0    = ln(tol / test2_k) / ln q
//   m1    = (ln(tol / test2_k) + 0.5 ln((k + m0) / k)) / ln q
//           (test2 divides by norm_a, the Frobenius estimate growing ~sqrt(k))
//   work  = m_rem + (rhs_left - 1) (k + m_rem),   m_rem = max(1, ceil(m1))
// The tail is long when work >= 8192 device RHS-iterations, about twice the
// measured host/device break-even (ready 3 400, pf_difficult_10m_3fe 3 900);
// q >= 1 counts as long. A long tail is handed over only if the CPU selector
// says MLSMR; otherwise, or when the deciding right-hand side converges first,
// the device keeps the tail for the whole fit.
constexpr int kCudaTailFirstCheckpoint = 64;
constexpr double kCudaTailWorkThreshold = 8192.0;

// F2's own size guard (not a property of the CPU selector, whose minimum is
// 200 000 rows): below 750 000 rows the CUDA tail is never handed over.
constexpr int kCudaTailMinRows = 750000;

// F2-B: the family of the probed cold restart and of its cumulative budget:
// reghdfe-comparable, ordinary FEs (no slope, no saved FEs), exactly three of
// them, unweighted, with regressors (the family of the F2-A tail).
bool cuda_three_fe_family(const HdfeOptions& options,
                          const std::vector<GpuFeInput>& fe_inputs,
                          const Eigen::VectorXd* weights,
                          int cols) {
    return options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
           !options.retain_fixed_effects && fe_inputs.size() == 3 &&
           weights == nullptr && cols > 0 &&
           std::none_of(fe_inputs.begin(), fe_inputs.end(),
                        [](const GpuFeInput& fe) { return fe.is_slope; });
}

// The CPU's own route for this fit. On the CPU, a three-FE Auto fit takes its
// route from the automatic MLSMR selector of the regressor
// (select_auto_mlsmr_method: a sampled Gauss-Seidel contraction probe on the
// design, confirmed on all rows when the sample is unrepresentative), which
// resolves to MLSMR or to its sweep fallback before the absorption starts. On
// a CUDA fit the regressor lends that same selector, with the same arguments,
// to the primary absorption through options.cuda_tail_route, and only when the
// public request was Auto (options.auto_intent; from_auto keeps its own,
// narrower, CUDA meaning). It is evaluated at most once per fit, after a long
// prediction. No selector, a sweep answer or an exception keeps the device tail.
struct CudaTailHostSelection {
    bool applicable = false;  // selector lent, public Auto, n >= the F2 guard
    bool evaluated = false;   // the selector answered (in this fit, at most once)
    bool host_mlsmr = false;  // the CPU selector picks MLSMR
    double rho = std::numeric_limits<double>::quiet_NaN();  // its sampled contraction
    double seconds = 0.0;     // its evaluation's wall time in this fit (trace only)
    int evaluations = 0;      // selector evaluations so far in this fit
    // "not-asked" before the decision, then "auto-mlsmr", "auto-sweep" (the
    // selector's vote), "n/a" (not applicable) or "exception".
    const char* selector = "not-asked";
    // "-", or why: "voted", "not-auto", "no-selector", "below-nmin", "exception",
    // "forced".
    const char* reason = "-";
};

CudaTailHostSelection select_cuda_tail_host_route(int n, const HdfeOptions& options) {
    CudaTailHostSelection out;
    out.selector = "n/a";
    if (!options.auto_intent) {
        out.reason = "not-auto";
        return out;
    }
    if (options.cuda_tail_route == nullptr || !*options.cuda_tail_route) {
        out.reason = "no-selector";
        return out;
    }
    if (n < kCudaTailMinRows) {
        out.reason = "below-nmin";
        return out;
    }
    out.applicable = true;
    const AutoRouteAnswer answer = (*options.cuda_tail_route)();
    out.evaluated = true;
    out.host_mlsmr = answer.mlsmr;
    out.rho = answer.rho;
    out.seconds = answer.seconds;
    out.evaluations = answer.evaluations;
    out.selector = answer.mlsmr ? "auto-mlsmr" : "auto-sweep";
    out.reason = "voted";
    return out;
}

struct CudaTailProbe {
    CudaTailMode mode = CudaTailMode::Adaptive;
    bool trace = false;
    // "tail": the completion after converged IT sweeps (F2-A); "restart": the
    // cold device LSMR after a broken CG stage (F2-B). Same rule, same selector.
    const char* stage = "tail";
    int rhs_count = 0;
    int budget = 0;  // device iterations available to each right-hand side
    std::function<CudaTailHostSelection()> select;  // the CPU selector, run once
    int rhs = -1;    // the deciding right-hand side
    bool decided = false;
    bool handover = false;
    bool selected = false;
    CudaTailHostSelection selection;
    int k = 0;                  // iterations of the deciding RHS at the decision
    int device_iterations = 0;  // max over the device solves at the hand-over
    double q = 0.0;
    double m1 = 0.0;
    double work = 0.0;
    // Inputs of the last checkpoint, printed with the decision (trace only).
    double norm_ar_k = 0.0;
    double norm_ar_half = 0.0;
    double test2_k = 0.0;
    double tol_k = 0.0;
    double m0 = 0.0;
    double m_rem = 0.0;
    double norm_ar_at[32] = {};  // norm_ar at iteration 2^j of the current RHS

    // End of iteration kk (1-based) of right-hand side rhs_index, after the
    // stop tests. Reads host scalars only. True to leave the device solve.
    bool observe(int rhs_index, int kk, double norm_ar, double test2, double tol) {
        if (decided) return false;
        if (mode == CudaTailMode::Force) {
            rhs = rhs_index;
            k = kk;
            decided = true;
            handover = budget - kk >= 1;
            selection.selector = "forced";
            selection.reason = "forced";
            report(rhs_index, kk, handover ? "handover" : "keep-budget");
            return handover;
        }
        const bool power = kk > 0 && (kk & (kk - 1)) == 0;
        if (!power) return false;
        int j = 0;
        for (unsigned v = static_cast<unsigned>(kk); v > 1U; v >>= 1U) ++j;
        if (j >= 32) return false;
        norm_ar_at[j] = norm_ar;  // overwritten by each right-hand side in turn
        if (kk < kCudaTailFirstCheckpoint) return false;
        if (rhs < 0) rhs = rhs_index;
        if (rhs_index != rhs) return false;
        const double before = norm_ar_at[j - 1];
        if (!finite_double_bits(norm_ar) || !finite_double_bits(before) ||
            !finite_double_bits(test2) || !(before > 0.0) || !(norm_ar > 0.0) ||
            !(tol > 0.0) || !(test2 > tol)) {
            return false;  // no decision at this checkpoint
        }
        k = kk;
        norm_ar_k = norm_ar;
        norm_ar_half = before;
        test2_k = test2;
        tol_k = tol;
        q = std::pow(norm_ar / before, 2.0 / static_cast<double>(kk));
        bool finite_m = false;
        m0 = 0.0;
        m1 = 0.0;
        if (q > 0.0 && q < 1.0) {
            const double lq = std::log(q);
            const double gap = std::log(tol / test2);
            m0 = gap / lq;
            m1 = (gap + 0.5 * std::log((kk + m0) / static_cast<double>(kk))) / lq;
            finite_m = finite_double_bits(m1);
        }
        const double rhs_left = static_cast<double>(rhs_count - rhs_index);
        m_rem = 0.0;
        if (finite_m) {
            m_rem = std::max(1.0, std::ceil(m1));
            work = m_rem + (rhs_left - 1.0) * (static_cast<double>(kk) + m_rem);
        } else {
            work = std::numeric_limits<double>::infinity();
        }
        const char* decision = "continue";
        if (!(work < kCudaTailWorkThreshold)) {
            decided = true;
            if (budget - kk < 1) {
                decision = "keep-budget";
            } else {
                if (!selected) {
                    selected = true;
                    try {
                        if (select) selection = select();
                    } catch (...) {
                        selection = CudaTailHostSelection{};  // keep the device tail
                        selection.selector = "exception";
                        selection.reason = "exception";
                    }
                }
                handover = selection.host_mlsmr;
                decision = handover ? "handover" : "keep-selector";
            }
        }
        report(rhs_index, kk, decision);
        return handover;
    }

    // The device solve of right-hand side rhs_index ended without a hand-over.
    void finish(int rhs_index) {
        if (!decided && rhs == rhs_index) decided = true;
    }

    void report(int rhs_index, int kk, const char* decision) const {
        if (!trace) return;
        std::fprintf(stderr,
                     "cuda_tail_probe rhs=%d k=%d rhs_count=%d norm_ar=%.17g "
                     "norm_ar_half=%.17g test2=%.17g tol=%.17g q=%.17g m0=%.17g "
                     "m1=%.17g m_rem=%.17g work=%.17g threshold=%.17g budget=%d "
                     "budget_left=%d decision=%s selector=%s reason=%s applicable=%d "
                     "evaluated=%d evaluations=%d rho=%.17g select_seconds=%.6f "
                     "stage=%s\n",
                     rhs_index, kk, rhs_count, norm_ar_k, norm_ar_half, test2_k, tol_k,
                     q, m0, m1, m_rem, work, kCudaTailWorkThreshold, budget,
                     budget - kk, decision, selection.selector, selection.reason,
                     selection.applicable ? 1 : 0, selection.evaluated ? 1 : 0,
                     selection.evaluations, selection.rho, selection.seconds, stage);
    }
};

// Host MLSMR on the IT state (F2-A). Absorption is a projection: the IT state
// differs from the original data by a vector in the span of the fixed effects,
// so its projection is the projection of the original data; the device probe's
// own partial work is simply discarded. Exceptions are contained: a host
// failure never turns the CUDA fit into a GPU failure, the caller finishes on
// the device with what is left of the budget. The thread team is the fit's own
// (options.num_threads, the absorption lease), as for any host absorption.
bool complete_cuda_tail_on_host(const Eigen::VectorXd& warm_y,
                                const Eigen::MatrixXd& warm_x,
                                const std::vector<GpuFeInput>& fe_inputs,
                                int n,
                                const HdfeOptions& base,
                                int budget,
                                AbsorptionResult& host,
                                int& host_iterations,
                                double& host_seconds) {
    host_iterations = 0;
    host_seconds = 0.0;
    if (budget <= 0) return false;
    const auto start = std::chrono::steady_clock::now();
    bool ok = false;
    try {
        std::vector<Eigen::VectorXi> fes;
        fes.reserve(fe_inputs.size());
        for (const auto& fe : fe_inputs) {
            fes.emplace_back(Eigen::Map<const Eigen::VectorXi>(fe.group_ids, n));
        }
        HdfeOptions mo = base;
        mo.absorption_method = AbsorptionMethod::Mlsmr;
        mo.max_iter = budget;
        mo.early_routing = false;
        host = absorb_fixed_effects_mlsmr(warm_y, warm_x, fes, nullptr, mo);
        host_iterations = std::max(0, host.iterations);
        ok = host.converged;
    } catch (...) {
        host = AbsorptionResult{};
        ok = false;
    }
    host_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return ok;
}

// ---- P3-fix: the xhdfe-fast restart of a broken CUDA CG stage ----
// Brief Verifications/perf_fereg_gap_20261106/briefs/P3_FIX_REINICIO_FAST_CUDA.md;
// investigation RELATORIO_P3_20261108.md; the restart is
// restart_fast_cg_after_breakdown, below absorb_fixed_effects_cuda_impl.
//
// Test hook, tests and diagnostics only (inert unless set, read once per
// process): XHDFE_TEST_CUDA_CG_BREAKDOWN=<k> (k >= 1) breaks down the first
// CUDA block-CG stage of the process that reaches its iteration k, before that
// iteration's step, as a real breakdown would; "<k>:all" breaks down every CG
// stage of the process at its iteration k. Any other value leaves the hook
// off. It lets tests/analytic_precision/cuda_fast_restart_contract.py exercise
// the restart on demand; nothing else reads it.
struct CudaCgTestBreakdown {
    int iteration = 0;  // 0: off
    bool every_stage = false;
};

const CudaCgTestBreakdown& cuda_cg_test_breakdown() {
    static const CudaCgTestBreakdown hook = []() {
        CudaCgTestBreakdown out;
        const char* e = std::getenv("XHDFE_TEST_CUDA_CG_BREAKDOWN");
        if (e == nullptr || *e == '\0') return out;
        char* end = nullptr;
        const long k = std::strtol(e, &end, 10);
        if (end == e || k < 1 || k > std::numeric_limits<int>::max()) return out;
        if (*end == '\0') {
            out.iteration = static_cast<int>(k);
        } else if (std::strcmp(end, ":all") == 0) {
            out.iteration = static_cast<int>(k);
            out.every_stage = true;
        }
        return out;
    }();
    return hook;
}

std::atomic<int> g_cuda_cg_test_breakdowns{0};

// Test hook, tests and diagnostics only (inert unless set, read once per process):
// XHDFE_TEST_CUDA_STRICT_LSMR_REJECT=1 rejects the state that the strict device
// LSMR restart of a broken CG stage approved (restart_strict_after_breakdown), so
// tests/analytic_precision exercises the strict repair that must follow it and the
// end of the sequence (flow control, not a frequency). Any other value: off.
bool cuda_strict_lsmr_test_reject() {
    static const bool reject = []() {
        const char* e = std::getenv("XHDFE_TEST_CUDA_STRICT_LSMR_REJECT");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return reject;
}

// True when the hook breaks down the current CG stage at iteration `iter`.
bool cuda_cg_test_breakdown_due(int iter) {
    const CudaCgTestBreakdown& hook = cuda_cg_test_breakdown();
    if (hook.iteration <= 0 || iter != hook.iteration) return false;
    if (hook.every_stage) {
        g_cuda_cg_test_breakdowns.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    int none = 0;
    return g_cuda_cg_test_breakdowns.compare_exchange_strong(none, 1);
}

// Cold restarts of the CG stage after its first attempt broke down: six
// attempts in all at most. On the P3 reproducer 9 of 52 first attempts broke
// down (0.17; Wilson 95 % upper bound 0.30). If consecutive attempts were
// independent - an inference, measured per attempt index by the permanent test,
// not a property this code relies on - six consecutive breakdowns would have
// probability 0.17^6 = 2.4e-5 per fit (7.3e-4 at the upper bound). Each attempt
// there costs 45 or 46 iterations, and every attempt runs within what the
// earlier ones left of max_iter.
constexpr int kFastCgColdRestarts = 5;

// The device LSMR after every cold restart broke down (restart_fast_cg_after_breakdown,
// step 2). It stays only if every forced execution of the permanent test ends in an
// accepted LSMR fit on the GPU inside the envelope (decision rule written before the
// measurement, RELATORIO_P3FIX_20261109.md section 4); otherwise it is false, and an
// exhausted restart is refused.
constexpr bool kFastCgRestartLsmrTier = true;

// The shape of restart_fast_cg_after_breakdown: an xhdfe-fast fit with ordinary
// FEs (no slope, no saved FE), unweighted, whose broken CG stage was entered at
// the gate (cuda_it_handoff_reason is 0 exactly then: every hand-off from the
// IT sweeps sets 1 or 2) and ran the non-deterministic reductions
// (n > kCudaSmallCgMaxRows). The public request must be xhdfe-fast without saved
// FEs (options.fast_intent, set once by the regressor): a strict-residual fit
// repaired natively as xhdfe-fast restarts with the device LSMR and the strict
// check instead (HdfeOptions::strict_provenance, restart_strict_after_breakdown),
// and the
// projection continuation of a savefe fit reaches this point with xhdfe-fast
// options but keeps its 2.30.1 restart, as does every caller that sets neither
// flag. Every other broken CG stage keeps its 2.30.1 restart
// (absorb_fixed_effects_cuda).
bool cuda_fast_cg_restart_applies(const HdfeOptions& options,
                                  const std::vector<GpuFeInput>& fe_inputs,
                                  const Eigen::VectorXd* weights,
                                  int n,
                                  const AbsorptionResult& broken) {
    return options.fast_intent && options.tolerance_mode == ToleranceMode::XhdfeFast &&
           !options.retain_fixed_effects && weights == nullptr &&
           n > kCudaSmallCgMaxRows && broken.cuda_it_handoff_reason == 0 &&
           std::none_of(fe_inputs.begin(), fe_inputs.end(),
                        [](const GpuFeInput& fe) { return fe.is_slope; });
}

// F2-B: the ledger of the cold restart of a three-FE Comparable fit after a
// broken CG stage, one trace line (XHDFE_GI_LSMR_TRACE) with the iterations and
// wall seconds of every stage, device and host apart, and the outcome.
// it_iterations: the IT sweeps before the hand-off to the CG stage that the CG
// stage did not count; cg_iterations: what the CG stage counted (also the
// sweeps under the N1 continuation); counted_before: what the restart charged
// to the budget before its own work (it + cg; cg alone under
// XHDFE_CUDA_TAIL_HANDOVER=0, the 2.30.1 accounting); budget: the device
// iterations left to each right-hand side. device_*: the probed (or, with the
// switch at 0, the plain) cold device LSMR, max over right-hand sides; host_*:
// the host MLSMR after a hand-over; after_*: the cold device LSMR after a host
// failure. Seconds are trace only.
struct CudaRestartLedger {
    const char* mode = "adaptive";
    int it_iterations = 0;
    int it_reason = 0;
    double it_seconds = 0.0;
    int cg_iterations = 0;
    double cg_seconds = 0.0;
    int counted_before = 0;
    int max_iter = 0;
    int budget = 0;
    int device_iterations = 0;
    double device_seconds = 0.0;
    int host_budget = 0;
    int host_iterations = 0;
    bool host_converged = false;
    double host_seconds = 0.0;
    int after_iterations = 0;
    double after_seconds = 0.0;
    int threads = 0;
    int iterations = 0;
    const char* outcome = "restart->device-lsmr";

    void print(const CudaTailProbe* probe) const {
        std::fprintf(stderr,
                     "cuda_tail_restart mode=%s it_iterations=%d it_reason=%s "
                     "it_seconds=%.6f cg_iterations=%d cg_seconds=%.6f counted_before=%d "
                     "max_iter=%d budget=%d probe_decided=%d probe_rhs=%d probe_k=%d "
                     "device_iterations=%d device_seconds=%.6f selector=%s reason=%s "
                     "rho=%.17g select_seconds=%.6f evaluations=%d host_method=mlsmr "
                     "host_budget=%d host_iterations=%d host_converged=%d "
                     "host_seconds=%.6f after_iterations=%d after_seconds=%.6f threads=%d "
                     "iterations=%d outcome=%s\n",
                     mode, it_iterations,
                     it_reason == 1 ? "divergence" : (it_reason == 2 ? "cap" : "none"),
                     it_seconds, cg_iterations, cg_seconds, counted_before, max_iter,
                     budget, probe != nullptr && probe->decided ? 1 : 0,
                     probe != nullptr ? probe->rhs : -1, probe != nullptr ? probe->k : 0,
                     device_iterations, device_seconds,
                     probe != nullptr ? probe->selection.selector : "n/a",
                     probe != nullptr ? probe->selection.reason : "off",
                     probe != nullptr ? probe->selection.rho
                                      : std::numeric_limits<double>::quiet_NaN(),
                     probe != nullptr ? probe->selection.seconds : 0.0,
                     probe != nullptr ? probe->selection.evaluations : 0, host_budget,
                     host_iterations, host_converged ? 1 : 0, host_seconds,
                     after_iterations, after_seconds, threads, iterations, outcome);
    }
};

}  // namespace

bool cuda_backend_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (std::getenv("XHDFE_CUDA_AVAIL_DIAG") != nullptr) {
        std::fprintf(stderr, "[xhdfe-cuda] cudaGetDeviceCount status=%d (%s) count=%d\n",
                     static_cast<int>(status), cudaGetErrorString(status), count);
    }
    if (status != cudaSuccess) {
        return false;
    }
    return count > 0;
}

static bool absorb_ordinary_lsmr_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& fe_inputs,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionResult& result,
    CudaTailProbe* probe = nullptr);

bool absorb_fixed_effects_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<GpuFeInput>& fe_inputs,
                               const Eigen::VectorXd* weights,
                               const std::vector<std::size_t>& sweep_order,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               AbsorptionResult& result) {
    return absorb_fixed_effects_cuda(y, X, fe_inputs, weights, sweep_order,
                                      options, method, result, 0.0);
}

static bool absorb_fixed_effects_cuda_impl(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<GpuFeInput>& fe_inputs,
                               const Eigen::VectorXd* weights,
                               const std::vector<std::size_t>& sweep_order,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               AbsorptionResult& result,
                               double solver_fe_tolerance,
                               bool allow_cg) {
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

    bool any_slope = false;
    for (const auto& fe : fe_inputs) {
        any_slope = any_slope || fe.is_slope;
        if (fe.is_slope &&
            (!fe.slope_values || !fe.slope_sum_z || !fe.slope_sum_zz)) {
            return false;
        }
    }
    if (any_slope && method == AbsorptionMethod::Jacobi) {
        return false;
    }
    if (!any_slope && weights != nullptr && fe_inputs.size() == 2 &&
        method != AbsorptionMethod::Lsmr && method != AbsorptionMethod::Mlsmr &&
        options.max_iter > 0 &&
        (fe_inputs[0].num_levels_present <= 0 || fe_inputs[0].num_levels_present == 2) &&
        (fe_inputs[1].num_levels_present <= 0 || fe_inputs[1].num_levels_present == 2)) {
        weighted_two_by_two::Plan plan;
        if (weighted_two_by_two::prepare(y, X, fe_inputs[0].group_ids,
                fe_inputs[1].group_ids, weights, options, plan)) {
            return absorb_weighted_two_by_two_cuda(y, X, plan, options, result);
        }
    }
    if (!any_slope && weights != nullptr && fe_inputs.size() > 2 &&
        method != AbsorptionMethod::Lsmr && method != AbsorptionMethod::Mlsmr &&
        options.max_iter > 0) {
        std::vector<weighted_two_by_two::FeView> views;
        views.reserve(fe_inputs.size());
        for (const auto& input : fe_inputs)
            views.push_back({input.group_ids, input.num_groups, input.num_levels_present});
        weighted_two_by_two::Plan plan;
        if (weighted_two_by_two::prepare_indexed(y, X, views, weights, options, plan)) {
            return absorb_weighted_two_by_two_cuda(y, X, plan, options, result);
        }
    }

    if (!any_slope && fe_inputs.size() == 2) {
        const auto& a=fe_inputs[0];const auto& b=fe_inputs[1];
        additive_cell_projection::Plan plan;
        if (additive_cell_projection::prepare(y,X,
                {a.group_ids,a.num_groups,a.num_levels_present},
                {b.group_ids,b.num_groups,b.num_levels_present},weights,options,plan)) {
            return absorb_additive_cells_cuda(y,X,plan,options,result);
        }
    }

    // Default ON (10jun2026): restores the exact 1-iteration solve for
    // saturated tree/path graphs (zigzag class) that the feature-branch
    // kernel restructure had regressed to tens of thousands of accelerated
    // iterations with degraded SEs. Shape-gated by
    // build_tree_duplicate_edge_groups; disable with XHDFE_CUDA_EXACT_TREE=0.
    const char* exact_tree_env = std::getenv("XHDFE_CUDA_EXACT_TREE");
    const bool use_exact_tree_shortcut =
        exact_tree_env == nullptr || exact_tree_env[0] != '0';
    if (use_exact_tree_shortcut && !any_slope && weights == nullptr &&
        !options.retain_fixed_effects) {
        TreeDuplicateEdgeGroups edge_groups;
        if (build_tree_duplicate_edge_groups(fe_inputs, n, edge_groups)) {
            return absorb_tree_duplicate_edges_cuda(y, X, edge_groups, fe_inputs,
                                                    options, result);
        }
    }

    const bool store_alphas =
        !any_slope &&
        options.retain_fixed_effects &&
        options.fe_recovery_method == FeRecoveryMethod::Hybrid &&
        cols <= options.savefe_fastpath_max_cols;
    // Mirror the CPU acceleration gate: large n always, plus ill-conditioned
    // sparse multi-FE graphs detected via low mean group occupancy
    // (observations per FE level). Tree-like citation / board-membership /
    // matched-pair panels (e.g. patents, directors) otherwise run unaccelerated
    // and need thousands of Gauss-Seidel sweeps; the Irons-Tuck path converges
    // in tens. Threshold tunable via XHDFE_ACCEL_OCC (benchmarking only).
    double min_fe_occupancy_gpu = -1.0;
    for (const auto& fi : fe_inputs) {
        const int levels = fi.num_levels_present > 0 ? fi.num_levels_present : fi.num_groups;
        if (levels > 0) {
            const double occ = static_cast<double>(n) / static_cast<double>(levels);
            if (min_fe_occupancy_gpu < 0.0 || occ < min_fe_occupancy_gpu) {
                min_fe_occupancy_gpu = occ;
            }
        }
    }
    const double accel_occ_threshold_gpu = cuda_accel_occupancy_threshold();
    const bool ill_conditioned_graph_gpu =
        fe_inputs.size() >= 2 && min_fe_occupancy_gpu >= 0.0 &&
        min_fe_occupancy_gpu <= accel_occ_threshold_gpu;
    // Sparse ordinary two-FE graphs can meet MAP's step test with unfinished weak
    // directions (CUDA04). Preserve the exact shortcuts above, and otherwise
    // select device LSMR before solving, with the ordinary Krylov contract.
    if (fe_inputs.size() == 2 && weights == nullptr &&
        ill_conditioned_graph_gpu && options.from_auto && !any_slope &&
        !options.retain_fixed_effects && !options.symmetric_sweep &&
        !options.use_sparse_solver && !options.use_krylov &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
        options.convergence_criterion == ConvergenceCriterion::Auto) {
        return absorb_ordinary_lsmr_cuda(y, X, fe_inputs, weights, options, result);
    }
    const bool n1_graph_correction = n1_continuation_predicate() != nullptr &&
        fe_inputs.size() == 3 && !any_slope && weights == nullptr &&
        !store_alphas && !options.retain_fixed_effects &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable;
    bool use_reghdfe_cg = allow_cg && (n1_graph_correction ||
        should_use_cuda_reghdfe_cg(fe_inputs, n, weights, options, store_alphas));
    if (allow_cg && !use_reghdfe_cg) {
        // GPU-CG adaptive gate: hand ill-conditioned high-iteration comparable jobs to the
        // existing block-PCG. On these graphs the IT-MAP comparable path stalls ~2e-7 off
        // reghdfe in ~530 sweeps; the PCG matches reghdfe to ~1e-9 in fewer iterations. Only
        // runs the probe when the dominant-share gate already declined. Fast mode untouched.
        use_reghdfe_cg =
            should_use_cuda_adaptive_cg(fe_inputs, n, weights, options, store_alphas, y);
    }
    // Small CG problems are latency-bound.  Use deterministic column-owned
    // accumulation and one-block scalar reductions throughout this regime;
    // this also makes the CG stopping path invariant to global-atomic order.
    const bool deterministic_small_cg =
        use_reghdfe_cg && n <= kCudaSmallCgMaxRows && !any_slope;
    const bool use_accel =
        use_reghdfe_cg ||
        (fe_inputs.size() >= 2 && (n >= 200000 || ill_conditioned_graph_gpu));
    // Match the CPU work-bound guarantee: for occupancy-triggered acceleration
    // (small, possibly pathological graphs) cap accelerated iterations so the
    // accelerated path never performs more sweeps than the plain solver would.
    const bool accel_from_occupancy_gpu = use_accel && (n < 200000);
    const int accel_iter_budget =
        accel_from_occupancy_gpu ? std::max(1, (options.max_iter * 2) / 5) : options.max_iter;
    constexpr int kGrandAccelInterval = 4;
    const bool accel_store_alphas = store_alphas && use_accel;
    const bool accel_grand_alphas = accel_store_alphas && (kGrandAccelInterval > 0);
    const double fe_tol =
        options.fe_tolerance > 0.0 ? options.fe_tolerance : options.tol;
    double convergence_tol = effective_absorption_tolerance(options);
    // See fe_absorption.cpp: reghdfe-comparable mode switches the accelerated
    // loop to the norm-of-change criterion at the NOMINAL tolerance.
    const bool honest_mode =
        honest_tol_trigger_enabled() ||
        options.tolerance_mode == ToleranceMode::ReghdfeComparable;
    const double honest_tol = options.tol;
    // Heterogeneous-slope designs: the CPU mixed absorber resolves the Auto
    // criterion under reghdfe-comparable to the Reghdfe mean-reldif update
    // check (resolved_mixed_convergence_criterion in fe_absorption.cpp). The
    // honest norm-of-change exit is a different, stricter stopping metric and
    // made GPU comparable run ~2x the CPU iterations on slope designs. Mirror
    // the CPU criterion here so both backends stop on the same validated
    // rule. Kill switch: XHDFE_GPU_MIXED_REGHDFE_CRIT=0.
    static const bool mixed_crit_enabled = []() {
        const char* e = std::getenv("XHDFE_GPU_MIXED_REGHDFE_CRIT");
        return !(e != nullptr && e[0] == '0');
    }();
    const bool mixed_update_criterion =
        mixed_crit_enabled && any_slope &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
        options.convergence_criterion == ConvergenceCriterion::Auto &&
        !honest_tol_trigger_enabled();
    // IT->CG divergence hand-off (mirror of fe_absorption.cpp). When the
    // Irons-Tuck safeguard detects divergence (norm rises past cg_bail_factor x
    // its running minimum) or the accelerator grinds past cg_iter_cap honest
    // iterations on an ill-conditioned graph (github-class), hand the current
    // post-sweep iterate to the stable block-CG instead of plain Gauss-Seidel.
    // No-op on well-conditioned data: the iterate norm is monotone and every
    // supported dataset converges well under the cap, so the 16 others stay
    // bit-identical. Tunable via XHDFE_PACKED_CG_FALLBACK / XHDFE_CG_BAIL_FACTOR
    // / XHDFE_CG_ITER_CAP (kept in sync with the CPU constants).
    static const bool cuda_cg_fallback_enabled = []() {
        const char* e = std::getenv("XHDFE_PACKED_CG_FALLBACK");
        return !(e != nullptr && e[0] == '0');
    }();
    static const double cg_bail_factor = []() {
        const char* e = std::getenv("XHDFE_CG_BAIL_FACTOR");
        const double v = (e != nullptr) ? std::atof(e) : 1.5;
        return v > 1.0 ? v : 1.5;
    }();
    static const int cg_iter_cap = []() {
        const char* e = std::getenv("XHDFE_CG_ITER_CAP");
        if (e != nullptr) {
            const int v = std::atoi(e);
            return v > 0 ? v : 0;
        }
        return 1000;
    }();
    // The block-CG handles only the no-alpha, unit-weight, multi-FE case (same
    // preconditions as should_use_cuda_reghdfe_cg minus the criterion gate).
    const bool cuda_cg_capable =
        cuda_cg_fallback_enabled && !store_alphas &&
        !options.retain_fixed_effects && fe_inputs.size() >= 2 && weights == nullptr;
    // Divergence-triggered hand-off mirrors the CPU packed absorber: available
    // whenever the safeguard itself is active, not only in honest mode. After
    // the safeguard decoupling, the fast-mode guard would otherwise suspend
    // extrapolation and grind plain sweeps to the finish (github, tol<=1e-11:
    // ~13k GPU iterations vs ~450 total on the CPU, which does hand off). The
    // CG stops on update_error <= options.tol — the rigorous norm-of-change
    // criterion at the nominal tolerance — so the hand-off cannot weaken the
    // fast-mode result; it replaces a diverging trajectory with a convergent
    // one. The iteration-cap hand-off below keeps its honest_mode gate: the
    // cap can trigger on healthy-but-slow fast runs (workers legitimately
    // needs 1020 accelerated iterations at tol=1e-12) and altering those is
    // not authorized. Kill switch: XHDFE_CUDA_GUARD_CG_BAIL=0 restores the
    // suspend-only fast-mode behaviour.
    static const bool cuda_guard_cg_bail_enabled = []() {
        const char* e = std::getenv("XHDFE_CUDA_GUARD_CG_BAIL");
        return !(e != nullptr && e[0] == '0');
    }();
    const bool cuda_cg_fallback_ok =
        allow_cg && cuda_cg_capable &&
        (honest_mode ||
         (accel_guard_always_enabled() && cuda_guard_cg_bail_enabled));
    bool accel_diverged = false;
    // Why the IT sweeps handed their iterate to the CG stage (F2-B ledger and
    // trace): 1 divergence (cg_bail_factor), 2 the cg_iter_cap iteration cap.
    int accel_handoff_reason = 0;
    if (store_alphas && fe_tol > 0.0 && convergence_tol > 0.0) {
        constexpr double kFeTolScale = 1e-8;
        // Transporting an absolute max-update tolerance must not relax this
        // existing relative norm stopping rule.
        const double work_fe_tol = solver_fe_tolerance > 0.0 ? solver_fe_tolerance : fe_tol;
        convergence_tol = std::min(convergence_tol, work_fe_tol * kFeTolScale);
    }

    try {
        const auto* necessary_continuation = n1_continuation_predicate();
        int next_necessary_check = 0;
        auto necessary_candidate_ready = [&](const double* device_y,
                                              const double* device_X, int iteration,
                                              const std::vector<double>* restore = nullptr) {
            if (!necessary_continuation) return true;
            if (iteration < next_necessary_check) return false;
            // Graph-preconditioned corrections need only tens of steps; do
            // not force 64 additional steps after a failed necessary check.
            const int kNecessaryCheckInterval = n1_graph_correction ? 1 : 64;
            next_necessary_check = iteration + kNecessaryCheckInterval;
            // These buffers also receive the final D2H result; do not allocate
            // another N-by-p matrix merely to check a failed fit's continuation.
            result.y_tilde.resize(n);
            result.X_tilde.resize(n, cols);
            cuda_check(cudaMemcpy(result.y_tilde.data(), device_y, sizeof(double) * n,
                                  cudaMemcpyDeviceToHost), "N1 continuation y copy failed");
            if (cols > 0)
                cuda_check(cudaMemcpy(result.X_tilde.data(), device_X,
                    sizeof(double) * static_cast<std::size_t>(n) * cols,
                    cudaMemcpyDeviceToHost), "N1 continuation X copy failed");
            if (restore) {
                result.y_tilde *= (*restore)[0];
                for (int j = 0; j < cols; ++j)
                    result.X_tilde.col(j) *= (*restore)[static_cast<std::size_t>(j + 1)];
            }
            return (*necessary_continuation)(result.y_tilde, result.X_tilde);
        };
        // Consume any stale, sticky CUDA error left by a prior failed or
        // non-converged call in this process. CUDA runtime errors are sticky and
        // are otherwise surfaced by the next CUDA API call (here the first
        // cudaMalloc/cudaMemcpy), which would spuriously fail this absorption and
        // (under gpubackend(cuda)) be reported as rc=498 even though the GPU is
        // healthy. This was the cause of the transient "CUDA absorption failed"
        // results observed when many estimations run back-to-back in one session.
        cudaGetLastError();
        // WP1 phase telemetry. The reporter destructs on every exit of this
        // try block (returns and exceptions alike), so a partial record is
        // still emitted if the solve throws. Host steady_clock is accurate
        // here because each phase boundary coincides with a synchronizing
        // CUDA call (staged copies and the per-check D2H of sumsq); per-kernel
        // CUDA events belong to the WP3 shadow kernels where genuinely async
        // regions appear. setup_ms is the in-function remainder.
        struct CudaProfileReport {
            bool enabled = false;
            std::chrono::steady_clock::time_point t0;
            double h2d_ms = 0.0;
            double solve_ms = 0.0;
            double d2h_ms = 0.0;
            long long sweeps = 0;
            long long h2d_calls0 = 0;
            long long d2h_calls0 = 0;
            ~CudaProfileReport() {
                if (!enabled) {
                    return;
                }
                const double total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
                std::fprintf(
                    stderr,
                    "cuda_profile total_ms=%.3f h2d_ms=%.3f solve_ms=%.3f "
                    "d2h_ms=%.3f setup_ms=%.3f sweeps=%lld h2d_calls=%lld "
                    "d2h_calls=%lld\n",
                    total_ms, h2d_ms, solve_ms, d2h_ms,
                    std::max(0.0, total_ms - h2d_ms - solve_ms - d2h_ms),
                    sweeps, g_cuda_h2d_calls.load() - h2d_calls0,
                    g_cuda_d2h_calls.load() - d2h_calls0);
            }
        } cuda_prof;
        cuda_prof.enabled = cuda_profile_enabled();
        cuda_prof.t0 = std::chrono::steady_clock::now();
        cuda_prof.h2d_calls0 = g_cuda_h2d_calls.load();
        cuda_prof.d2h_calls0 = g_cuda_d2h_calls.load();
        CudaAccelTraceBuffer accel_trace;
        accel_trace.enabled = cuda_accel_trace_enabled();
        accel_trace.use_accel = use_accel;
        accel_trace.honest_mode = honest_mode;
        accel_trace.n = n;
        accel_trace.cols = cols;
        accel_trace.dims = static_cast<int>(fe_inputs.size());
        accel_trace.honest_tol = honest_tol;
        accel_trace.convergence_tol = convergence_tol;
        const bool unit_weights = (weights == nullptr);
        CudaWorkspace& workspace = cuda_workspace;

        const auto cuda_prof_h2d_t0 = std::chrono::steady_clock::now();
        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        staged_memcpy_h2d(d_y.data(), y.data(), sizeof(double) * n, workspace.h_stage,
                          "cudaMemcpy y failed");

        const std::size_t x_size = static_cast<std::size_t>(ld) * cols;
        DeviceBuffer<double>& d_x = workspace.d_x;
        if (cols > 0) {
            d_x.allocate(x_size);
            staged_memcpy_h2d(d_x.data(), X.data(), sizeof(double) * ld * cols,
                              workspace.h_stage, "cudaMemcpy X failed");
        }

        DeviceBuffer<double>& d_weights = workspace.d_weights;
        if (!unit_weights) {
            d_weights.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(d_weights.data(), weights->data(), sizeof(double) * n,
                              workspace.h_stage, "cudaMemcpy weights failed");
        }

        std::vector<CudaFeDevice>& fe_dev = workspace.fe_dev;
        // Keep the persistent workspace sized to EXACTLY the current number of
        // fixed effects. Previously this only grew (`<`), so after a model with
        // more FEs, a model with fewer FEs left stale extra CudaFeDevice entries
        // behind. Several solve loops iterate over fe_dev.size() (the polish
        // sweeps, the grand-acceleration alpha handling, the final demean), so
        // those stale dimensions were demeaned with a buffer (gid/sum) sized for
        // the *previous* dataset's larger n -> an out-of-bounds GPU read (the
        // rc=498 "illegal memory access" cascade) and corrupted residuals.
        // Truncating to fe_inputs.size() makes every dimension loop correct.
        if (fe_dev.size() != fe_inputs.size()) {
            fe_dev.resize(fe_inputs.size());
        }
        const std::size_t active_fes = fe_inputs.size();
        for (std::size_t idx = 0; idx < fe_inputs.size(); ++idx) {
            const auto& fe = fe_inputs[idx];
            CudaFeDevice& dev = fe_dev[idx];
            dev.num_groups = fe.num_groups;
            dev.num_levels_present = fe.num_levels_present;
            dev.is_slope = fe.is_slope;
            dev.slope_has_intercept = fe.slope_has_intercept;
            // Dispatch strategy for the accumulation kernel:
            //   1) Privatized (shared-memory partial sums) if num_groups *
            //      (1 + cols) fits in ~40 KB shared memory per block. Best for
            //      low-cardinality FEs (year, industry, etc.) where global
            //      atomic contention otherwise dominates.
            //   2) Warp-segmented otherwise: never slower than a plain atomic
            //      kernel and collapses same-group threads into one atomic.
            // The atomic-only branch is no longer selected directly; segmented
            // subsumes it on modern hardware (H100+).
            constexpr std::size_t kPrivatizedShmemBudget = 40 * 1024;  // bytes
            const std::size_t shmem_needed =
                static_cast<std::size_t>(fe.num_groups) *
                (1 + static_cast<std::size_t>(cols)) * sizeof(double);
            dev.use_privatized_sums =
                (fe.num_groups > 0 && shmem_needed <= kPrivatizedShmemBudget);
            dev.privatized_shmem_bytes = dev.use_privatized_sums ? shmem_needed : 0;
            dev.use_segmented_sums = !dev.use_privatized_sums;
            dev.gid.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(dev.gid.data(), fe.group_ids, sizeof(int) * n,
                              workspace.h_stage, "cudaMemcpy group ids failed");
            dev.weight_sums.allocate(static_cast<std::size_t>(fe.num_groups));
            staged_memcpy_h2d(dev.weight_sums.data(), fe.weight_sums,
                              sizeof(double) * fe.num_groups, workspace.h_stage,
                              "cudaMemcpy weight sums failed");
            if (dev.is_slope) {
                dev.slope_z.allocate(static_cast<std::size_t>(n));
                staged_memcpy_h2d(dev.slope_z.data(), fe.slope_values, sizeof(double) * n,
                                  workspace.h_stage, "cudaMemcpy slope values failed");
                dev.slope_sum_z.allocate(static_cast<std::size_t>(fe.num_groups));
                dev.slope_sum_zz.allocate(static_cast<std::size_t>(fe.num_groups));
                staged_memcpy_h2d(dev.slope_sum_z.data(), fe.slope_sum_z,
                                  sizeof(double) * fe.num_groups, workspace.h_stage,
                                  "cudaMemcpy slope sum_z failed");
                staged_memcpy_h2d(dev.slope_sum_zz.data(), fe.slope_sum_zz,
                                  sizeof(double) * fe.num_groups, workspace.h_stage,
                                  "cudaMemcpy slope sum_zz failed");
                dev.sum_zy.allocate(static_cast<std::size_t>(fe.num_groups));
                if (cols > 0) {
                    dev.sum_zx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                } else {
                    dev.sum_zx.reset();
                }
            } else {
                dev.slope_z.reset();
                dev.slope_sum_z.reset();
                dev.slope_sum_zz.reset();
                dev.sum_zy.reset();
                dev.sum_zx.reset();
            }
            dev.sum_y.allocate(static_cast<std::size_t>(fe.num_groups));
            if (cols > 0) {
                dev.sum_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
            }
            if (store_alphas) {
                dev.alpha_y.allocate(static_cast<std::size_t>(fe.num_groups));
                cuda_check(cudaMemset(dev.alpha_y.data(), 0,
                                      sizeof(double) * fe.num_groups),
                           "cudaMemset alpha_y failed");
                if (cols > 0) {
                    dev.alpha_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                    cuda_check(cudaMemset(dev.alpha_x.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset alpha_x failed");
                } else {
                    dev.alpha_x.reset();
                }
                if (accel_store_alphas) {
                    dev.alpha_y_gx.allocate(static_cast<std::size_t>(fe.num_groups));
                    cuda_check(cudaMemset(dev.alpha_y_gx.data(), 0,
                                          sizeof(double) * fe.num_groups),
                               "cudaMemset alpha_y_gx failed");
                    dev.alpha_y_ggx.allocate(static_cast<std::size_t>(fe.num_groups));
                    cuda_check(cudaMemset(dev.alpha_y_ggx.data(), 0,
                                          sizeof(double) * fe.num_groups),
                               "cudaMemset alpha_y_ggx failed");
                    if (cols > 0) {
                        dev.alpha_x_gx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                        cuda_check(cudaMemset(dev.alpha_x_gx.data(), 0,
                                              sizeof(double) * fe.num_groups * cols),
                                   "cudaMemset alpha_x_gx failed");
                        dev.alpha_x_ggx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                        cuda_check(cudaMemset(dev.alpha_x_ggx.data(), 0,
                                              sizeof(double) * fe.num_groups * cols),
                                   "cudaMemset alpha_x_ggx failed");
                    } else {
                        dev.alpha_x_gx.reset();
                        dev.alpha_x_ggx.reset();
                    }
                    if (accel_grand_alphas) {
                        dev.alpha_y_acc_a.allocate(static_cast<std::size_t>(fe.num_groups));
                        dev.alpha_y_acc_b.allocate(static_cast<std::size_t>(fe.num_groups));
                        dev.alpha_y_acc_c.allocate(static_cast<std::size_t>(fe.num_groups));
                        cuda_check(cudaMemset(dev.alpha_y_acc_a.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_a failed");
                        cuda_check(cudaMemset(dev.alpha_y_acc_b.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_b failed");
                        cuda_check(cudaMemset(dev.alpha_y_acc_c.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_c failed");
                        if (cols > 0) {
                            dev.alpha_x_acc_a.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            dev.alpha_x_acc_b.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            dev.alpha_x_acc_c.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            cuda_check(cudaMemset(dev.alpha_x_acc_a.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_a failed");
                            cuda_check(cudaMemset(dev.alpha_x_acc_b.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_b failed");
                            cuda_check(cudaMemset(dev.alpha_x_acc_c.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_c failed");
                        } else {
                            dev.alpha_x_acc_a.reset();
                            dev.alpha_x_acc_b.reset();
                            dev.alpha_x_acc_c.reset();
                        }
                    } else {
                        dev.alpha_y_acc_a.reset();
                        dev.alpha_y_acc_b.reset();
                        dev.alpha_y_acc_c.reset();
                        dev.alpha_x_acc_a.reset();
                        dev.alpha_x_acc_b.reset();
                        dev.alpha_x_acc_c.reset();
                    }
                } else {
                    dev.alpha_y_gx.reset();
                    dev.alpha_x_gx.reset();
                    dev.alpha_y_ggx.reset();
                    dev.alpha_x_ggx.reset();
                    dev.alpha_y_acc_a.reset();
                    dev.alpha_y_acc_b.reset();
                    dev.alpha_y_acc_c.reset();
                    dev.alpha_x_acc_a.reset();
                    dev.alpha_x_acc_b.reset();
                    dev.alpha_x_acc_c.reset();
                }
            } else {
                dev.alpha_y.reset();
                dev.alpha_x.reset();
                dev.alpha_y_gx.reset();
                dev.alpha_x_gx.reset();
                dev.alpha_y_ggx.reset();
                dev.alpha_x_ggx.reset();
                dev.alpha_y_acc_a.reset();
                dev.alpha_y_acc_b.reset();
                dev.alpha_y_acc_c.reset();
                dev.alpha_x_acc_a.reset();
                dev.alpha_x_acc_b.reset();
                dev.alpha_x_acc_c.reset();
            }
        }

        result.fe_levels.clear();
        result.fe_levels.reserve(fe_inputs.size());
        for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
            result.fe_levels.push_back(fe_dev[dim].num_levels_present);
        }

        AlphaStatePtrs alpha_state_ptrs;
        AlphaStatePtrs alpha_gx_ptrs;
        AlphaStatePtrs alpha_ggx_ptrs;
        AlphaStatePtrs alpha_acc_a_ptrs;
        AlphaStatePtrs alpha_acc_b_ptrs;
        AlphaStatePtrs alpha_acc_c_ptrs;
        if (store_alphas) {
            const std::size_t dims = fe_inputs.size();
            alpha_state_ptrs.y.assign(dims, nullptr);
            alpha_state_ptrs.x.assign(dims, nullptr);
            for (std::size_t dim = 0; dim < dims; ++dim) {
                alpha_state_ptrs.y[dim] = fe_dev[dim].alpha_y.data();
                alpha_state_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x.data() : nullptr;
            }
            if (accel_store_alphas) {
                alpha_gx_ptrs.y.assign(dims, nullptr);
                alpha_gx_ptrs.x.assign(dims, nullptr);
                alpha_ggx_ptrs.y.assign(dims, nullptr);
                alpha_ggx_ptrs.x.assign(dims, nullptr);
                for (std::size_t dim = 0; dim < dims; ++dim) {
                    alpha_gx_ptrs.y[dim] = fe_dev[dim].alpha_y_gx.data();
                    alpha_gx_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_gx.data() : nullptr;
                    alpha_ggx_ptrs.y[dim] = fe_dev[dim].alpha_y_ggx.data();
                    alpha_ggx_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_ggx.data() : nullptr;
                }
                if (accel_grand_alphas) {
                    alpha_acc_a_ptrs.y.assign(dims, nullptr);
                    alpha_acc_a_ptrs.x.assign(dims, nullptr);
                    alpha_acc_b_ptrs.y.assign(dims, nullptr);
                    alpha_acc_b_ptrs.x.assign(dims, nullptr);
                    alpha_acc_c_ptrs.y.assign(dims, nullptr);
                    alpha_acc_c_ptrs.x.assign(dims, nullptr);
                    for (std::size_t dim = 0; dim < dims; ++dim) {
                        alpha_acc_a_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_a.data();
                        alpha_acc_a_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_a.data()
                                                           : nullptr;
                        alpha_acc_b_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_b.data();
                        alpha_acc_b_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_b.data()
                                                           : nullptr;
                        alpha_acc_c_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_c.data();
                        alpha_acc_c_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_c.data()
                                                           : nullptr;
                    }
                }
            }
        }

	        cuda_prof.h2d_ms = std::chrono::duration<double, std::milli>(
	            std::chrono::steady_clock::now() - cuda_prof_h2d_t0).count();

	        // WP3 shadow certificate: capture the solver-invariant quantities
	        // (Orig_sq, Op_sq) from the freshly-uploaded originals, before the
	        // solver overwrites d_y/d_x in place. Standard surface only in this
	        // increment (heterogeneous slopes are WP4).
	        struct CertShadowGuard {
	            CudaCertShadowHandle* h = nullptr;
	            ~CertShadowGuard() {
	                if (h != nullptr) {
	                    cuda_cert_shadow_abort(h);
	                }
	            }
	        } cert_shadow;
	        if (cuda_cert_shadow_enabled()) {
	            std::vector<const int*> shadow_gids;
	            std::vector<const double*> shadow_wsums;
	            std::vector<int> shadow_groups;
	            std::vector<const double*> shadow_slopes;
	            std::vector<unsigned char> shadow_slope_int;
	            shadow_gids.reserve(fe_inputs.size());
	            shadow_wsums.reserve(fe_inputs.size());
	            shadow_groups.reserve(fe_inputs.size());
	            shadow_slopes.reserve(fe_inputs.size());
	            shadow_slope_int.reserve(fe_inputs.size());
	            for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
	                shadow_gids.push_back(fe_dev[dim].gid.data());
	                shadow_wsums.push_back(fe_dev[dim].weight_sums.data());
	                shadow_groups.push_back(fe_inputs[dim].num_groups);
	                shadow_slopes.push_back(
	                    fe_inputs[dim].is_slope ? fe_dev[dim].slope_z.data()
	                                            : nullptr);
	                shadow_slope_int.push_back(
	                    fe_inputs[dim].slope_has_intercept ? 1 : 0);
	            }
	            cert_shadow.h = cuda_cert_shadow_begin(
	                d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld,
	                unit_weights ? nullptr : d_weights.data(),
	                shadow_gids.data(), shadow_wsums.data(),
	                shadow_groups.data(), static_cast<int>(fe_inputs.size()),
	                shadow_slopes.data(), shadow_slope_int.data());
	        }

	        DeviceBuffer<double>& d_sumsq = workspace.d_sumsq;
	        d_sumsq.allocate(1);
	        const bool use_update_error =
	            needs_reghdfe_update_check(options) || mixed_update_criterion;
	        DeviceBuffer<double>& d_y_prev_update = workspace.d_y_prev_update;
	        DeviceBuffer<double>& d_x_prev_update = workspace.d_x_prev_update;
	        DeviceBuffer<double>& d_update_sums = workspace.d_update_sums;
	        if (use_update_error) {
	            d_y_prev_update.allocate(static_cast<std::size_t>(n));
	            if (cols > 0) {
	                d_x_prev_update.allocate(x_size);
	            }
	        }
	        if (use_update_error || use_reghdfe_cg || cuda_cg_fallback_ok) {
	            // Explicit/adaptive CG and the divergence fallback all use this
	            // workspace for their per-column reductions.  Allocate it even
	            // when the reghdfe update check itself is inactive.
	            d_update_sums.allocate(static_cast<std::size_t>(cols + 2));
	        }

	        const auto cuda_prof_solve_t0 = std::chrono::steady_clock::now();
	        auto compute_sumsq = [&](const double* y_ptr, const double* x_ptr, double& out) {
            cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)), "cudaMemset sumsq failed");
            const int blocks = (n + kBlockSize - 1) / kBlockSize;
            sumsq_kernel<kBlockSize><<<blocks, kBlockSize>>>(
                y_ptr, x_ptr, n, cols, ld, d_sumsq.data());
            cuda_check(cudaGetLastError(), "sumsq kernel launch failed");
            cuda_check(cudaMemcpy(&out, d_sumsq.data(), sizeof(double), cudaMemcpyDeviceToHost),
	                       "cudaMemcpy sumsq failed");
	        };

	        auto copy_update_snapshot = [&](const double* y_ptr, const double* x_ptr) {
	            if (!use_update_error) {
	                return;
	            }
	            cuda_check(cudaMemcpy(d_y_prev_update.data(), y_ptr, sizeof(double) * n,
	                                  cudaMemcpyDeviceToDevice),
	                       "cudaMemcpy y convergence snapshot failed");
	            if (cols > 0) {
	                cuda_check(cudaMemcpy(d_x_prev_update.data(), x_ptr, sizeof(double) * x_size,
	                                      cudaMemcpyDeviceToDevice),
	                           "cudaMemcpy X convergence snapshot failed");
	            }
	        };

	        auto compute_update_error = [&](const double* y_new,
	                                        const double* x_new,
	                                        double& out) {
	            out = std::numeric_limits<double>::max();
	            if (!use_update_error) {
	                return;
	            }
	            const int width = cols + 1;
	            cuda_check(cudaMemset(d_update_sums.data(), 0,
	                                  sizeof(double) * static_cast<std::size_t>(width + 1)),
	                       "cudaMemset update sums failed");
	            const int blocks = (n + kBlockSize - 1) / kBlockSize;
	            const dim3 grid(blocks, width + 1);
	            reldif_update_sums_kernel<kBlockSize><<<grid, kBlockSize>>>(
	                y_new,
	                d_y_prev_update.data(),
	                cols > 0 ? x_new : nullptr,
	                cols > 0 ? d_x_prev_update.data() : nullptr,
	                unit_weights ? nullptr : d_weights.data(),
	                unit_weights,
	                n,
	                cols,
	                ld,
	                d_update_sums.data());
	            cuda_check(cudaGetLastError(), "reldif update kernel launch failed");
	            std::vector<double> host(static_cast<std::size_t>(width + 1), 0.0);
	            cuda_check(cudaMemcpy(host.data(), d_update_sums.data(),
	                                  sizeof(double) * static_cast<std::size_t>(width + 1),
	                                  cudaMemcpyDeviceToHost),
	                       "cudaMemcpy update sums failed");
	            const double denom = host[static_cast<std::size_t>(width)];
	            if (!(denom > 0.0)) {
	                throw std::runtime_error("Weights must sum to a positive value");
	            }
	            double max_mean = 0.0;
	            for (int j = 0; j < width; ++j) {
	                max_mean = std::max(max_mean, host[static_cast<std::size_t>(j)] / denom);
	            }
	            out = max_mean;
	        };

	        auto accumulate_alpha = [&](CudaFeDevice& fe,
                                    double scale,
                                    double* alpha_y_ptr,
                                    double* alpha_x_ptr) {
            if (!alpha_y_ptr && !alpha_x_ptr) {
                return;
            }
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            accumulate_alpha_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                alpha_y_ptr,
                alpha_x_ptr,
                fe.num_groups,
                cols,
                scale);
            cuda_check(cudaGetLastError(), "accumulate alpha kernel launch failed");
        };

        auto compute_means_only = [&](CudaFeDevice& fe,
                                      double* y_ptr,
                                      double* x_ptr,
                                      double alpha_scale,
                                      double* alpha_y_ptr,
                                      double* alpha_x_ptr) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }

            if (fe.is_slope) {
                cuda_check(cudaMemset(fe.sum_zy.data(), 0, sizeof(double) * fe.num_groups),
                           "cudaMemset sum_zy failed");
                if (cols > 0) {
                    cuda_check(cudaMemset(fe.sum_zx.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset sum_zx failed");
                }
                accumulate_slope_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.slope_z.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_slope_sums kernel launch failed");

                compute_slope_coefficients_kernel<<<blocks_g, kBlockSize>>>(
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.weight_sums.data(),
                    fe.slope_sum_z.data(),
                    fe.slope_sum_zz.data(),
                    fe.num_groups,
                    cols,
                    fe.slope_has_intercept);
                cuda_check(cudaGetLastError(),
                           "compute_slope_coefficients kernel launch failed");
                return;
            }

            if (deterministic_small_cg) {
                const int component_blocks =
                    (cols + 1 + kBlockSize - 1) / kBlockSize;
                accumulate_sums_deterministic_columns_kernel
                    <<<component_blocks, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        unit_weights ? nullptr : d_weights.data(),
                        unit_weights,
                        fe.sum_y.data(),
                        fe.sum_x.data(),
                        fe.num_groups);
                cuda_check(
                    cudaGetLastError(),
                    "accumulate_sums (deterministic small-CG) kernel launch failed");
            } else if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            accumulate_alpha(fe, alpha_scale, alpha_y_ptr, alpha_x_ptr);
        };

        bool recovery_delta_requested = false;
        DeviceBuffer<double> d_recovery_max_delta;
        auto run_demean = [&](CudaFeDevice& fe,
                              double* y_ptr,
                              double* x_ptr,
                              bool compute_check,
                              double& sumsq_out,
                              double alpha_scale,
                              double* alpha_y_ptr,
                              double* alpha_x_ptr) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }

            if (fe.is_slope) {
                cuda_check(cudaMemset(fe.sum_zy.data(), 0, sizeof(double) * fe.num_groups),
                           "cudaMemset sum_zy failed");
                if (cols > 0) {
                    cuda_check(cudaMemset(fe.sum_zx.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset sum_zx failed");
                }
                accumulate_slope_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.slope_z.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_slope_sums kernel launch failed");

                compute_slope_coefficients_kernel<<<blocks_g, kBlockSize>>>(
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.weight_sums.data(),
                    fe.slope_sum_z.data(),
                    fe.slope_sum_zz.data(),
                    fe.num_groups,
                    cols,
                    fe.slope_has_intercept);
                cuda_check(cudaGetLastError(),
                           "compute_slope_coefficients kernel launch failed");

                if (compute_check) {
                    cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                               "cudaMemset sumsq failed");
                    apply_slope_sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        fe.slope_z.data(),
                        fe.sum_y.data(),
                        fe.sum_zy.data(),
                        fe.sum_x.data(),
                        fe.sum_zx.data(),
                        fe.num_groups,
                        d_sumsq.data());
                    cuda_check(cudaGetLastError(),
                               "apply_slope_sumsq kernel launch failed");
                    cuda_check(cudaMemcpy(&sumsq_out, d_sumsq.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy sumsq failed");
                } else {
                    apply_slope_kernel<<<blocks_n, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        fe.slope_z.data(),
                        fe.sum_y.data(),
                        fe.sum_zy.data(),
                        fe.sum_x.data(),
                        fe.sum_zx.data(),
                        fe.num_groups);
                    cuda_check(cudaGetLastError(), "apply_slope kernel launch failed");
                }
                return;
            }

            if (deterministic_small_cg) {
                const int component_blocks =
                    (cols + 1 + kBlockSize - 1) / kBlockSize;
                accumulate_sums_deterministic_columns_kernel
                    <<<component_blocks, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        unit_weights ? nullptr : d_weights.data(),
                        unit_weights,
                        fe.sum_y.data(),
                        fe.sum_x.data(),
                        fe.num_groups);
                cuda_check(
                    cudaGetLastError(),
                    "accumulate_sums (deterministic small-CG) kernel launch failed");
            } else if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            accumulate_alpha(fe, alpha_scale, alpha_y_ptr, alpha_x_ptr);

            if (recovery_delta_requested) {
                max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                    fe.sum_y.data(), fe.num_groups, d_recovery_max_delta.data());
                cuda_check(cudaGetLastError(), "recovery delta kernel launch failed");
            }

            if (compute_check) {
                cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                           "cudaMemset sumsq failed");
                apply_means_sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups,
                    d_sumsq.data());
                cuda_check(cudaGetLastError(), "apply_means_sumsq kernel launch failed");
                cuda_check(cudaMemcpy(&sumsq_out, d_sumsq.data(), sizeof(double),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy sumsq failed");
            } else {
                apply_means_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "apply_means kernel launch failed");
            }
        };

        const int dims = static_cast<int>(active_fes);
        if (method == AbsorptionMethod::Jacobi) {
            std::vector<const int*> gid_ptrs(static_cast<std::size_t>(dims));
            std::vector<const double*> mean_y_ptrs(static_cast<std::size_t>(dims));
            std::vector<const double*> mean_x_ptrs(static_cast<std::size_t>(dims));
            std::vector<int> groups(static_cast<std::size_t>(dims));
            for (int d = 0; d < dims; ++d) {
                gid_ptrs[static_cast<std::size_t>(d)] = fe_dev[d].gid.data();
                mean_y_ptrs[static_cast<std::size_t>(d)] = fe_dev[d].sum_y.data();
                mean_x_ptrs[static_cast<std::size_t>(d)] =
                    cols > 0 ? fe_dev[d].sum_x.data() : nullptr;
                groups[static_cast<std::size_t>(d)] = fe_dev[d].num_groups;
            }

            DeviceBuffer<const int*> d_gid_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<const double*> d_mean_y_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<const double*> d_mean_x_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<int> d_groups(static_cast<std::size_t>(dims));

            cuda_check(cudaMemcpy(d_gid_ptrs.data(), gid_ptrs.data(),
                                  sizeof(const int*) * dims, cudaMemcpyHostToDevice),
                       "cudaMemcpy gid ptrs failed");
            cuda_check(cudaMemcpy(d_mean_y_ptrs.data(), mean_y_ptrs.data(),
                                  sizeof(const double*) * dims, cudaMemcpyHostToDevice),
                       "cudaMemcpy mean y ptrs failed");
            if (cols > 0) {
                cuda_check(cudaMemcpy(d_mean_x_ptrs.data(), mean_x_ptrs.data(),
                                      sizeof(const double*) * dims, cudaMemcpyHostToDevice),
                           "cudaMemcpy mean x ptrs failed");
            }
            cuda_check(cudaMemcpy(d_groups.data(), groups.data(), sizeof(int) * dims,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy groups failed");

            double relaxation = options.jacobi_relaxation;
            if (relaxation <= 0.0) {
                relaxation = 2.0 / (static_cast<double>(dims) + 1.0);
            }
            relaxation = std::min(relaxation, 1.0);

	            double prev_norm = 0.0;
	            compute_sumsq(d_y.data(), d_x.data(), prev_norm);
	            prev_norm = std::sqrt(prev_norm);
	            copy_update_snapshot(d_y.data(), d_x.data());
	            const int check_interval = std::max(1, options.convergence_check_interval);
            int last_check_iter = -1;
            bool converged = false;

            for (int iter = 0; iter < options.max_iter; ++iter) {
                for (int d = 0; d < dims; ++d) {
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas) {
                        alpha_y_ptr = alpha_state_ptrs.y[static_cast<std::size_t>(d)];
                        alpha_x_ptr = alpha_state_ptrs.x[static_cast<std::size_t>(d)];
                    }
                    compute_means_only(fe_dev[d], d_y.data(), d_x.data(), relaxation,
                                       alpha_y_ptr, alpha_x_ptr);
                }

                const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
                jacobi_update_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    d_gid_ptrs.data(),
                    d_mean_y_ptrs.data(),
                    cols > 0 ? d_mean_x_ptrs.data() : nullptr,
                    d_groups.data(),
                    dims,
                    relaxation);
                cuda_check(cudaGetLastError(), "jacobi update kernel launch failed");

                const bool do_check =
                    (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
                if (do_check) {
                    double curr_norm = 0.0;
                    compute_sumsq(d_y.data(), d_x.data(), curr_norm);
                    curr_norm = std::sqrt(curr_norm);
                    const double denom = std::max(1.0, prev_norm);
                    const double rel_change = std::abs(curr_norm - prev_norm) / denom;
	                    const int step = iter - last_check_iter;
	                    last_check_iter = iter;
	                    prev_norm = curr_norm;
	                    double update_error = std::numeric_limits<double>::max();
	                    compute_update_error(d_y.data(), d_x.data(), update_error);
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    const double norm_error = rel_change / static_cast<double>(step);
	                    if (step > 0 &&
	                        convergence_reached(options, norm_error, update_error, convergence_tol) &&
                            necessary_candidate_ready(d_y.data(), d_x.data(), iter)) {
	                        result.iterations = iter + 1;
                        converged = true;
                        break;
                    }
                }
            }

            result.converged = converged;
            if (!converged) {
                result.iterations = options.max_iter;
            } else if (result.iterations == 0) {
                result.iterations = 1;
            }
        } else {
            bool use_symmetric = options.symmetric_sweep && sweep_order.size() > 1;
            bool have_cg_iteration_count = false;
            if (use_reghdfe_cg) {
                use_symmetric = true;
            }

            std::vector<FeProfileStats> fe_profiles;
            fe_profiles.reserve(fe_inputs.size());
            for (const auto& fe : fe_inputs) {
                fe_profiles.push_back(profile_fe(fe, unit_weights));
            }

            const FeDifficulty difficulty = classify_problem(fe_profiles);
            bool enable_two_stage = false;
            std::vector<std::size_t> two_fe_order;
            int iter_warmup = 0;
            int iter_two_fe = 0;
            if (fe_profiles.size() >= 3 && n >= 200000) {
                int first = -1;
                int second = -1;
                int third = -1;
                for (const auto& p : fe_profiles) {
                    const int groups = p.num_groups;
                    if (groups > first) {
                        third = second;
                        second = first;
                        first = groups;
                    } else if (groups > second) {
                        third = second;
                        second = groups;
                    } else if (groups > third) {
                        third = groups;
                    }
                }
                if (first > 0 && third > 0) {
                    constexpr double kTwoStageThirdRatioMax = 0.05;
                    if (static_cast<double>(third) <=
                        kTwoStageThirdRatioMax * static_cast<double>(first)) {
                        enable_two_stage = true;
                        if (difficulty == FeDifficulty::Extreme) {
                            iter_warmup = 5;
                            iter_two_fe = 5;
                        } else if (difficulty >= FeDifficulty::Hard) {
                            iter_warmup = 8;
                            iter_two_fe = 8;
                        } else {
                            iter_warmup = 5;
                            iter_two_fe = 8;
                        }
                    }
                }
            }
            if (enable_two_stage) {
                const std::vector<std::size_t> two_dims = select_two_largest_dims(fe_profiles);
                two_fe_order = filter_order(sweep_order, two_dims);
                if (two_fe_order.size() != 2) {
                    enable_two_stage = false;
                    iter_warmup = 0;
                    iter_two_fe = 0;
                }
            }
            accel_trace.two_stage = enable_two_stage;

            bool converged = false;

            auto compute_norm = [&]() {
                double norm = 0.0;
                compute_sumsq(d_y.data(), d_x.data(), norm);
                return std::sqrt(norm);
            };

            auto run_sweep = [&](double* y_ptr,
                                 double* x_ptr,
                                 const std::vector<std::size_t>& order,
                                 bool compute_check,
                                 double& sumsq_out,
                                 const AlphaStatePtrs* alpha_ptrs) {
                ++cuda_prof.sweeps;
                if (order.empty()) {
                    if (compute_check) {
                        sumsq_out = 0.0;
                    }
                    return;
                }
                double sumsq = 0.0;
                const bool do_symmetric = use_symmetric && order.size() > 1;
                for (std::size_t pos = 0; pos < order.size(); ++pos) {
                    const std::size_t dim = order[pos];
                    const bool last_call =
                        compute_check && !do_symmetric && (pos + 1 == order.size());
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (alpha_ptrs) {
                        alpha_y_ptr = alpha_ptrs->y[dim];
                        alpha_x_ptr = alpha_ptrs->x[dim];
                    }
                    run_demean(fe_dev[dim], y_ptr, x_ptr, last_call, sumsq, 1.0,
                               alpha_y_ptr, alpha_x_ptr);
                }
                if (do_symmetric) {
                    for (std::size_t idx = order.size(); idx-- > 0;) {
                        const std::size_t dim = order[idx];
                        const bool last_call = compute_check && (idx == 0);
                        double* alpha_y_ptr = nullptr;
                        double* alpha_x_ptr = nullptr;
                        if (alpha_ptrs) {
                            alpha_y_ptr = alpha_ptrs->y[dim];
                            alpha_x_ptr = alpha_ptrs->x[dim];
                        }
                        run_demean(fe_dev[dim], y_ptr, x_ptr, last_call, sumsq, 1.0,
                                   alpha_y_ptr, alpha_x_ptr);
                    }
                }
                if (compute_check) {
                    sumsq_out = sumsq;
                }
            };

            const int check_interval = std::max(1, options.convergence_check_interval);

            if (use_accel) {
                DeviceBuffer<double>& d_y_gx = workspace.d_y_gx;
                DeviceBuffer<double>& d_y_ggx = workspace.d_y_ggx;
                DeviceBuffer<double>& d_x_gx = workspace.d_x_gx;
                DeviceBuffer<double>& d_x_ggx = workspace.d_x_ggx;
                d_y_gx.allocate(static_cast<std::size_t>(n));
                d_y_ggx.allocate(static_cast<std::size_t>(n));
                if (cols > 0) {
                    d_x_gx.allocate(x_size);
                    d_x_ggx.allocate(x_size);
                }

                DeviceBuffer<double>& d_stats = workspace.d_stats;
                d_stats.allocate(3);
                DeviceBuffer<double>& d_cg_alpha = workspace.d_cg_alpha;
                DeviceBuffer<double>& d_cg_beta = workspace.d_cg_beta;
                const int cg_width = cols + 1;
                if (use_reghdfe_cg || cuda_cg_fallback_ok) {
                    // Allocate CG scalar buffers when CG can run either as the
                    // explicit accelerator or as the IT divergence fallback.
                    d_cg_alpha.allocate(static_cast<std::size_t>(cg_width));
                    d_cg_beta.allocate(static_cast<std::size_t>(cg_width));
                }
                const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
                const int blocks_x = (static_cast<int>(x_size) + kBlockSize - 1) / kBlockSize;
                const std::size_t total_values = static_cast<std::size_t>(n) + x_size;
                const int blocks_values =
                    static_cast<int>((total_values + kBlockSize - 1) / kBlockSize);
                const int iter_grand_acc = kGrandAccelInterval;

                DeviceBuffer<double>& d_y_acc_a = workspace.d_y_acc_a;
                DeviceBuffer<double>& d_y_acc_b = workspace.d_y_acc_b;
                DeviceBuffer<double>& d_y_acc_c = workspace.d_y_acc_c;
                DeviceBuffer<double>& d_x_acc_a = workspace.d_x_acc_a;
                DeviceBuffer<double>& d_x_acc_b = workspace.d_x_acc_b;
                DeviceBuffer<double>& d_x_acc_c = workspace.d_x_acc_c;
                if (iter_grand_acc > 0) {
                    d_y_acc_a.allocate(static_cast<std::size_t>(n));
                    d_y_acc_b.allocate(static_cast<std::size_t>(n));
                    d_y_acc_c.allocate(static_cast<std::size_t>(n));
                    if (cols > 0) {
                        d_x_acc_a.allocate(x_size);
                        d_x_acc_b.allocate(x_size);
                        d_x_acc_c.allocate(x_size);
                    }
                }

                auto copy_alpha_state = [&](const AlphaStatePtrs& dst,
                                            const AlphaStatePtrs& src) {
                    for (std::size_t dim = 0; dim < active_fes; ++dim) {
                        if (!dst.y[dim] || !src.y[dim]) {
                            continue;
                        }
                        const int groups = fe_dev[dim].num_groups;
                        cuda_check(cudaMemcpy(dst.y[dim], src.y[dim],
                                              sizeof(double) * groups,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy alpha_y state failed");
                        if (cols > 0 && dst.x[dim] && src.x[dim]) {
                            const std::size_t count =
                                static_cast<std::size_t>(groups) * static_cast<std::size_t>(cols);
                            cuda_check(cudaMemcpy(dst.x[dim], src.x[dim],
                                                  sizeof(double) * count,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy alpha_x state failed");
                        }
                    }
                };

                auto lincomb_alpha_state = [&](const AlphaStatePtrs& out,
                                               const AlphaStatePtrs& a,
                                               const AlphaStatePtrs& b,
                                               double coef) {
                    for (std::size_t dim = 0; dim < active_fes; ++dim) {
                        if (!out.y[dim] || !a.y[dim] || !b.y[dim]) {
                            continue;
                        }
                        const int groups = fe_dev[dim].num_groups;
                        const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;
                        irons_tuck_update_kernel<<<blocks_g, kBlockSize>>>(
                            out.y[dim], a.y[dim], b.y[dim], groups, coef);
                        cuda_check(cudaGetLastError(),
                                   "alpha lincomb kernel launch failed");
                        if (cols > 0 && out.x[dim] && a.x[dim] && b.x[dim]) {
                            const std::size_t count =
                                static_cast<std::size_t>(groups) * static_cast<std::size_t>(cols);
                            const int blocks_x =
                                (static_cast<int>(count) + kBlockSize - 1) / kBlockSize;
                            irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                out.x[dim], a.x[dim], b.x[dim],
                                static_cast<int>(count), coef);
                            cuda_check(cudaGetLastError(),
                                       "alpha lincomb X kernel launch failed");
                        }
                    }
                };

                auto run_accel_phase = [&](const std::vector<std::size_t>& order,
                                           int max_iter,
                                           int& iter_used) -> bool {
                    iter_used = 0;
                    next_necessary_check = 0;
                    if (max_iter <= 0 || order.empty()) {
                        return false;
	                    }
	                    accel_trace.begin_phase();
	                    double prev_norm = compute_norm();
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    int last_check_iter = -1;
	                    int grand_acc = 0;
                    // Adaptive-restart safeguard against Irons-Tuck divergence
                    // on ill-conditioned FE graphs (mirrors the CPU packed/SoA
                    // accel loops in fe_absorption.cpp). No-op on
                    // well-conditioned data (monotone iterate norm).
                    constexpr double kAccelDivergeFactor = 4.0;
                    double best_resid = prev_norm;
                    bool accel_suspended = false;
                    int accel_restart_count = 0;
                    double max_ratio_seen = 1.0;

                    cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy y_gx failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy x_gx failed");
                    }
                    if (store_alphas) {
                        copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                    }
                    double ignored = 0.0;
                    run_sweep(d_y_gx.data(), d_x_gx.data(), order, false, ignored,
                              store_alphas ? &alpha_gx_ptrs : nullptr);

                    for (int iter = 0; iter < max_iter; ++iter) {
                        const bool do_check = (check_interval == 1 || iter < check_interval ||
                                               iter % check_interval == 0);
                        cuda_check(cudaMemcpy(d_y_ggx.data(), d_y_gx.data(), sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy y_ggx failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(d_x_ggx.data(), d_x_gx.data(),
                                                  sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy x_ggx failed");
                        }
                        if (store_alphas) {
                            copy_alpha_state(alpha_ggx_ptrs, alpha_gx_ptrs);
                        }
                        run_sweep(d_y_ggx.data(), d_x_ggx.data(), order, false, ignored,
                                  store_alphas ? &alpha_ggx_ptrs : nullptr);

                        cuda_check(cudaMemset(d_stats.data(), 0, sizeof(double) * 3),
                                   "cudaMemset stats failed");
                        irons_tuck_stats_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                            d_y.data(), d_y_gx.data(), d_y_ggx.data(), n, d_stats.data());
                        cuda_check(cudaGetLastError(), "irons tuck stats kernel launch failed");
                        if (cols > 0) {
                            irons_tuck_stats_kernel<kBlockSize><<<blocks_x, kBlockSize>>>(
                                d_x.data(), d_x_gx.data(), d_x_ggx.data(),
                                static_cast<int>(x_size), d_stats.data());
                            cuda_check(cudaGetLastError(),
                                       "irons tuck stats X kernel launch failed");
                        }

                        double stats_host[3] = {0.0, 0.0, 0.0};
                        cuda_check(cudaMemcpy(stats_host, d_stats.data(), sizeof(double) * 3,
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy stats failed");
                        const double vprod = stats_host[0];
                        const double ssq = stats_host[1];
                        accel_trace.record(vprod, ssq, stats_host[2], prev_norm,
                                           iter);
                        if (honest_mode && !mixed_update_criterion &&
                            std::sqrt(stats_host[2]) <=
                                honest_tol * std::max(1.0, prev_norm) &&
                            necessary_candidate_ready(d_y_ggx.data(), d_x_ggx.data(), iter)) {
                            // One full sweep moved the iterate by less than
                            // tol: return ggx, the closest available iterate.
                            cuda_check(cudaMemcpy(d_y.data(), d_y_ggx.data(),
                                                  sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y honest-exit failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_ggx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x honest-exit failed");
                            }
                            if (store_alphas) {
                                copy_alpha_state(alpha_state_ptrs, alpha_ggx_ptrs);
                            }
                            iter_used = iter + 1;
                            if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr) {
                                std::fprintf(stderr,
                                             "[ACCEL cuda CONVERGED iter_used=%d restarts=%d]\n",
                                             iter_used, accel_restart_count);
                            }
                            return true;
                        }
                        if (ssq == 0.0) {
                            cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(), sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y fallback failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x fallback failed");
                            }
                            if (store_alphas) {
                                copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                            }
                            iter_used = iter + 1;
                            return necessary_candidate_ready(d_y.data(), d_x.data(),
                                std::max(iter, next_necessary_check));
                        }

                        double coef = vprod / ssq;
                        if (accel_suspended) {
                            // Diverging: take the plain (firmly non-expansive)
                            // sweep step instead of the IT extrapolation.
                            coef = 0.0;
                        }
                        irons_tuck_update_kernel<<<blocks_n, kBlockSize>>>(
                            d_y.data(), d_y_gx.data(), d_y_ggx.data(), n, coef);
                        cuda_check(cudaGetLastError(), "irons tuck update kernel launch failed");
                        if (cols > 0) {
                            irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                d_x.data(), d_x_gx.data(), d_x_ggx.data(),
                                static_cast<int>(x_size), coef);
                            cuda_check(cudaGetLastError(),
                                       "irons tuck update X kernel launch failed");
                        }
                        if (store_alphas) {
                            lincomb_alpha_state(alpha_state_ptrs, alpha_gx_ptrs, alpha_ggx_ptrs,
                                                coef);
                        }

                        cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(), sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy y_gx update failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                                  sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy x_gx update failed");
                        }
                        if (store_alphas) {
                            copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                        }

                        double sumsq = 0.0;
                        run_sweep(d_y_gx.data(), d_x_gx.data(), order, do_check, sumsq,
                                  store_alphas ? &alpha_gx_ptrs : nullptr);
                        if (do_check) {
                            const double curr_norm = std::sqrt(sumsq);
                            const double denom = std::max(1.0, prev_norm);
                            const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                            const int step = iter - last_check_iter;
                            last_check_iter = iter;
                            prev_norm = curr_norm;
                            // Deliberately NOT gated on honest_mode — see
                            // accel_guard_always_enabled().
                            if (honest_mode || accel_guard_always_enabled()) {
                                const double ratio =
                                    curr_norm / std::max(best_resid, 1e-300);
                                if (ratio > max_ratio_seen) {
                                    max_ratio_seen = ratio;
                                }
                                if (curr_norm <= best_resid) {
                                    best_resid = curr_norm;
                                    accel_suspended = false;
                                } else {
                                    if (cuda_cg_fallback_ok &&
                                        curr_norm > cg_bail_factor * best_resid) {
                                        // Diverging: hand the post-sweep iterate
                                        // (d_*_gx) to the stable block-CG.
                                        cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                              sizeof(double) * n,
                                                              cudaMemcpyDeviceToDevice),
                                                   "cudaMemcpy y cg-bail failed");
                                        if (cols > 0) {
                                            cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                                  sizeof(double) * x_size,
                                                                  cudaMemcpyDeviceToDevice),
                                                       "cudaMemcpy x cg-bail failed");
                                        }
                                        accel_diverged = true;
                                        accel_handoff_reason = 1;
                                        iter_used = iter + 1;
                                        return false;
                                    }
                                    if (curr_norm > kAccelDivergeFactor * best_resid) {
                                        if (!accel_suspended) {
                                            ++accel_restart_count;
                                            grand_acc = 0;
                                        }
                                        accel_suspended = true;
                                    }
                                }
                            }
                            if (mixed_update_criterion && step > 0) {
                                // Mirror of the CPU mixed absorber comparable
                                // exit: weighted mean reldif of the post-sweep
                                // iterate vs the previous check <= nominal tol.
                                double update_error =
                                    std::numeric_limits<double>::max();
                                compute_update_error(d_y_gx.data(), d_x_gx.data(),
                                                     update_error);
                                copy_update_snapshot(d_y_gx.data(), d_x_gx.data());
                                if (update_error <= options.tol &&
                                    necessary_candidate_ready(d_y_gx.data(), d_x_gx.data(), iter)) {
                                    cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                          sizeof(double) * n,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy y converged failed");
                                    if (cols > 0) {
                                        cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                              sizeof(double) * x_size,
                                                              cudaMemcpyDeviceToDevice),
                                                   "cudaMemcpy x converged failed");
                                    }
                                    if (store_alphas) {
                                        copy_alpha_state(alpha_state_ptrs,
                                                         alpha_gx_ptrs);
                                    }
                                    iter_used = iter + 1;
                                    return true;
                                }
                            }
                            if (!honest_mode && step > 0 &&
                                (rel_change / static_cast<double>(step)) < convergence_tol &&
                                necessary_candidate_ready(d_y_gx.data(), d_x_gx.data(), iter)) {
                                cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy y converged failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy x converged failed");
                                }
                                if (store_alphas) {
                                    copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                                }
                                iter_used = iter + 1;
                                return true;
                            }
                        }

                        if (cuda_cg_fallback_ok && honest_mode && cg_iter_cap > 0 &&
                            !accel_diverged && (iter + 1) >= cg_iter_cap) {
                            // Still grinding far past where any well-conditioned
                            // graph converges: hand the post-sweep iterate to CG.
                            cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                  sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y cg-cap failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x cg-cap failed");
                            }
                            accel_diverged = true;
                            accel_handoff_reason = 2;
                            iter_used = iter + 1;
                            return false;
                        }

                        if (!accel_suspended && iter_grand_acc > 0 &&
                            ((iter + 1) % iter_grand_acc == 0)) {
                            ++grand_acc;
                            if (grand_acc == 1) {
                                cuda_check(cudaMemcpy(d_y_acc_a.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc a failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_a.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc a X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_a_ptrs, alpha_gx_ptrs);
                                }
                            } else if (grand_acc == 2) {
                                cuda_check(cudaMemcpy(d_y_acc_b.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc b failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_b.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc b X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_b_ptrs, alpha_gx_ptrs);
                                }
                            } else {
                                cuda_check(cudaMemcpy(d_y_acc_c.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc c failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_c.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc c X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_c_ptrs, alpha_gx_ptrs);
                                }

                                cuda_check(cudaMemset(d_stats.data(), 0, sizeof(double) * 3),
                                           "cudaMemset stats failed");
                                irons_tuck_stats_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                                    d_y_acc_a.data(), d_y_acc_b.data(), d_y_acc_c.data(),
                                    n, d_stats.data());
                                cuda_check(cudaGetLastError(),
                                           "grand irons tuck stats kernel launch failed");
                                if (cols > 0) {
                                    irons_tuck_stats_kernel<kBlockSize><<<blocks_x, kBlockSize>>>(
                                        d_x_acc_a.data(), d_x_acc_b.data(), d_x_acc_c.data(),
                                        static_cast<int>(x_size), d_stats.data());
                                    cuda_check(cudaGetLastError(),
                                               "grand irons tuck stats X kernel launch failed");
                                }

                                double acc_stats[2] = {0.0, 0.0};
                                cuda_check(cudaMemcpy(acc_stats, d_stats.data(),
                                                      sizeof(double) * 2,
                                                      cudaMemcpyDeviceToHost),
                                           "cudaMemcpy acc stats failed");
                                if (acc_stats[1] != 0.0) {
                                    const double acc_coef = acc_stats[0] / acc_stats[1];
                                    irons_tuck_update_kernel<<<blocks_n, kBlockSize>>>(
                                        d_y_acc_a.data(), d_y_acc_b.data(), d_y_acc_c.data(),
                                        n, acc_coef);
                                    cuda_check(cudaGetLastError(),
                                               "grand irons tuck update kernel launch failed");
                                    if (cols > 0) {
                                        irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                            d_x_acc_a.data(), d_x_acc_b.data(), d_x_acc_c.data(),
                                            static_cast<int>(x_size), acc_coef);
                                        cuda_check(cudaGetLastError(),
                                                   "grand irons tuck update X kernel launch failed");
                                    }
                                    if (store_alphas && accel_grand_alphas) {
                                        lincomb_alpha_state(alpha_acc_a_ptrs, alpha_acc_b_ptrs,
                                                            alpha_acc_c_ptrs, acc_coef);
                                    }
                                }

                                cuda_check(cudaMemcpy(d_y.data(), d_y_acc_a.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc result failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x.data(), d_x_acc_a.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc result X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_state_ptrs, alpha_acc_a_ptrs);
                                }
                                cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc y_gx failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc x_gx failed");
                                }
                                if (store_alphas) {
                                    copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                                }
                                run_sweep(d_y_gx.data(), d_x_gx.data(), order, false, ignored,
                                          store_alphas ? &alpha_gx_ptrs : nullptr);
                                grand_acc = 0;
                            }
                        }
                    }

                    cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy y fallback final failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy x fallback final failed");
                    }
                    if (store_alphas) {
                        copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                    }
                    iter_used = max_iter;
                    return false;
                };

                // Stable block-CG over the symmetric (I - G) demean operator,
                // reused both as the explicit reghdfe CG accelerator (gate below)
                // and as the IT divergence fallback. CG re-derives its residual
                // from d_y/d_x, so it is safe to run on any handed-off iterate.
                auto run_cuda_cg = [&](const std::vector<std::size_t>& cg_order,
                                       int preceding_iterations,
                                       int budget_charge) {
                    next_necessary_check = 0;
                    const int counted_before = necessary_continuation ? preceding_iterations : 0;
                    // F2-B: budget_charge is what the stages before this one already
                    // spent of max_iter (the IT sweeps, in the three-FE family with
                    // the hand-over switch not at 0; 0 otherwise, the 2.30.1 budget).
                    // The CG stage never runs past what is left; its `iterations`
                    // keep the 2.30.1 accounting (counted_before).
                    const int cg_budget = std::max(
                        0, options.max_iter - std::max(counted_before, budget_charge));
                    // CG needs the symmetric (forward+backward) sweep so that
                    // (I - G) is self-adjoint. The gate path already forces this
                    // (use_reghdfe_cg => use_symmetric=true at setup); for the
                    // divergence fallback run_accel_phase has finished, so setting
                    // it here only affects the CG sweeps, matching the gate.
                    use_symmetric = true;
                    converged = false;
                    auto compute_component_sumsq = [&](const double* y_ptr,
                                                        const double* x_ptr,
                                                        std::vector<double>& out) {
                        out.assign(static_cast<std::size_t>(cg_width), 0.0);
                        cuda_check(cudaMemset(d_update_sums.data(), 0,
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width)),
                                   "cudaMemset CG sumsq failed");
                        const int reduction_blocks =
                            deterministic_small_cg ? 1 : blocks_n;
                        const dim3 grid(reduction_blocks, cg_width);
                        component_sumsq_kernel<kBlockSize><<<grid, kBlockSize>>>(
                            y_ptr, x_ptr, n, cols, ld, d_update_sums.data());
                        cuda_check(cudaGetLastError(), "CG sumsq kernel launch failed");
                        cuda_check(cudaMemcpy(out.data(), d_update_sums.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy CG sumsq failed");
                    };
                    auto component_exact_zero = [&](const double* y_ptr,
                                                    const double* x_ptr,
                                                    int component) {
                        const double* values =
                            component == 0
                                ? y_ptr
                                : x_ptr +
                                      static_cast<std::size_t>(component - 1) *
                                          static_cast<std::size_t>(ld);
                        cuda_check(cudaMemset(d_update_sums.data(), 0,
                                              sizeof(unsigned int)),
                                   "cudaMemset CG exact-zero flag failed");
                        any_nonzero_bits_kernel<<<blocks_n, kBlockSize>>>(
                            values, n,
                            reinterpret_cast<unsigned int*>(
                                d_update_sums.data()));
                        cuda_check(cudaGetLastError(),
                                   "CG exact-zero kernel launch failed");
                        unsigned int any_nonzero = 0;
                        cuda_check(cudaMemcpy(
                                       &any_nonzero, d_update_sums.data(),
                                       sizeof(any_nonzero),
                                       cudaMemcpyDeviceToHost),
                                   "cudaMemcpy CG exact-zero flag failed");
                        return any_nonzero == 0U;
                    };
                    auto compute_component_dot = [&](const double* y_a,
                                                     const double* y_b,
                                                     const double* x_a,
                                                     const double* x_b,
                                                     std::vector<double>& out) {
                        out.assign(static_cast<std::size_t>(cg_width), 0.0);
                        cuda_check(cudaMemset(d_update_sums.data(), 0,
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width)),
                                   "cudaMemset CG dot failed");
                        const int reduction_blocks =
                            deterministic_small_cg ? 1 : blocks_n;
                        const dim3 grid(reduction_blocks, cg_width);
                        component_dot_kernel<kBlockSize><<<grid, kBlockSize>>>(
                            y_a, y_b, x_a, x_b, n, cols, ld, d_update_sums.data());
                        cuda_check(cudaGetLastError(), "CG dot kernel launch failed");
                        cuda_check(cudaMemcpy(out.data(), d_update_sums.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy CG dot failed");
                    };
                    auto projection_sweep = [&](const double* src_y,
                                                const double* src_x,
                                                double* proj_y,
                                                double* proj_x) {
                        cuda_check(cudaMemcpy(proj_y, src_y, sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy CG projection y failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(proj_x, src_x, sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy CG projection X failed");
                        }
                        double ignored = 0.0;
                        run_sweep(proj_y, proj_x, cg_order, false, ignored, nullptr);
                        projection_residual_kernel<<<blocks_values, kBlockSize>>>(
                            proj_y, src_y, proj_x, src_x, total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG projection residual kernel failed");
                    };
                    auto subtract_columns = [&](double* target_y,
                                                double* target_x,
                                                const double* direction_y,
                                                const double* direction_x,
                                                const std::vector<double>& alpha) {
                        cuda_check(cudaMemcpy(d_cg_alpha.data(), alpha.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyHostToDevice),
                                   "cudaMemcpy CG alpha failed");
                        scaled_subtract_columns_kernel<<<blocks_values, kBlockSize>>>(
                            target_y, target_x, direction_y, direction_x, d_cg_alpha.data(),
                            total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG scaled subtract kernel failed");
                    };
                    auto update_direction = [&](double* u_y,
                                                double* u_x,
                                                const double* r_y,
                                                const double* r_x,
                                                const std::vector<double>& beta) {
                        cuda_check(cudaMemcpy(d_cg_beta.data(), beta.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyHostToDevice),
                                   "cudaMemcpy CG beta failed");
                        cg_direction_update_kernel<<<blocks_values, kBlockSize>>>(
                            u_y, u_x, r_y, r_x, d_cg_beta.data(), total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG direction update kernel failed");
                    };

                    if (n1_graph_correction) {
                        const std::int64_t total_levels =
                            static_cast<std::int64_t>(fe_inputs[0].num_groups) +
                            fe_inputs[1].num_groups + fe_inputs[2].num_groups;
                        if (total_levels > std::numeric_limits<int>::max())
                            throw std::runtime_error("N1 graph index range exceeded; no estimates returned");
                        const int o1 = fe_inputs[0].num_groups;
                        const int o2 = o1 + fe_inputs[1].num_groups;
                        const int levels = static_cast<int>(total_levels);
                        auto precondition = make_n1_graph_preconditioner(fe_inputs, n, options);
                        Eigen::MatrixXd r(levels, cg_width), z, p;
                        DeviceBuffer<double> device_levels;
                        const std::size_t level_count =
                            static_cast<std::size_t>(levels) * cg_width;
                        device_levels.allocate(level_count);
                        auto transpose_residual = [&]() {
                            int offset = 0;
                            for (std::size_t d = 0; d < fe_inputs.size(); ++d) {
                                auto& fe = fe_dev[d];
                                compute_means_only(fe, d_y.data(), d_x.data(), 0.0, nullptr, nullptr);
                                const dim3 grid((fe.num_groups + kBlockSize - 1) / kBlockSize,
                                                cg_width);
                                n1_graph_pack_sums_kernel<<<grid, kBlockSize>>>(
                                    fe.sum_y.data(), fe.sum_x.data(), fe.weight_sums.data(),
                                    fe.num_groups, offset, levels, device_levels.data());
                                cuda_check(cudaGetLastError(), "N1 graph transpose packing failed");
                                offset += fe.num_groups;
                            }
                            cuda_check(cudaMemcpy(r.data(), device_levels.data(),
                                sizeof(double) * level_count, cudaMemcpyDeviceToHost),
                                "N1 graph transpose copy failed");
                        };
                        auto products = [&]() {
                            std::vector<double> rz(cg_width, 0.0);
                            for (int k = 0; k < cg_width; ++k) {
                                long double sum = 0.0L;
                                for (int g = 0; g < levels; ++g)
                                    sum += static_cast<long double>(r(g,k)) * z(g,k);
                                rz[k] = static_cast<double>(sum);
                            }
                            return rz;
                        };
                        transpose_residual();
                        precondition(r, z);
                        p = z;
                        auto rz = products();
                        std::vector<double> steps(cg_width), beta(cg_width), denominator, state;
                        int iterations = 0;
                        for (int iter = 1; iter <= cg_budget; ++iter) {
                            cuda_check(cudaMemcpy(device_levels.data(), p.data(),
                                sizeof(double) * level_count, cudaMemcpyHostToDevice),
                                "N1 graph direction copy failed");
                            n1_graph_gather_kernel<<<dim3(blocks_n, cg_width), kBlockSize>>>(
                                device_levels.data(), levels, fe_dev[0].gid.data(),
                                fe_dev[1].gid.data(), fe_dev[2].gid.data(), o1, o2,
                                n, ld, d_y_acc_a.data(), d_x_acc_a.data());
                            cuda_check(cudaGetLastError(), "N1 graph gather failed");
                            compute_component_sumsq(d_y_acc_a.data(), d_x_acc_a.data(), denominator);
                            for (int k = 0; k < cg_width; ++k) {
                                if (!finite_double_bits(rz[k]) || rz[k] < 0.0 ||
                                    !finite_double_bits(denominator[k]) || denominator[k] < 0.0 ||
                                    (rz[k] > 0.0 && denominator[k] == 0.0))
                                    throw std::runtime_error("N1 graph CG breakdown; no estimates returned");
                                steps[k] = rz[k] > 0.0 ? rz[k] / denominator[k] : 0.0;
                            }
                            subtract_columns(d_y.data(), d_x.data(),
                                d_y_acc_a.data(), d_x_acc_a.data(), steps);
                            compute_component_sumsq(d_y.data(), d_x.data(), state);
                            double step_error = 0.0;
                            for (int k = 0; k < cg_width; ++k) {
                                const double step = std::abs(steps[k]) * std::sqrt(denominator[k]);
                                const double scale = std::sqrt(state[k]);
                                step_error = std::max(step_error, scale > 0.0 ? step/scale :
                                    (step == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()));
                            }
                            iterations = iter;
                            if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr)
                                std::fprintf(stderr, "[ACCEL cuda N1 graph iter=%d step=%.9g]\n", iter, step_error);
                            if (step_error <= options.tol &&
                                necessary_candidate_ready(d_y.data(), d_x.data(), iter)) {
                                converged = true;
                                break;
                            }
                            const auto old_rz = rz;
                            transpose_residual();
                            precondition(r, z);
                            rz = products();
                            for (int k = 0; k < cg_width; ++k) {
                                beta[k] = old_rz[k] > 0.0 ? rz[k]/old_rz[k] : 0.0;
                                p.col(k) = z.col(k) + beta[k] * p.col(k);
                            }
                        }
                        result.iterations = counted_before + iterations;
                        have_cg_iteration_count = true;
                        return;
                    }

                    std::vector<double> improvement;
                    std::vector<double> ssr;
                    std::vector<double> ssr_old;
                    std::vector<double> denom;
                    std::vector<double> alpha(static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<double> beta(static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<double> recent(static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<double> next_improvement(
                        static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<uint8_t> lane_solved(
                        static_cast<std::size_t>(cg_width), 0);
                    const double max_cg_step =
                        1.0 / std::sqrt(std::numeric_limits<double>::epsilon());
                    std::vector<double> restore_scale(
                        static_cast<std::size_t>(cg_width), 1.0);
                    bool normalized_small_cg = false;

                    compute_component_sumsq(d_y.data(), d_x.data(), improvement);
                    if (deterministic_small_cg) {
                        std::vector<double> normalize_scale(
                            static_cast<std::size_t>(cg_width), 1.0);
                        for (int k = 0; k < cg_width; ++k) {
                            const std::size_t lane =
                                static_cast<std::size_t>(k);
                            const double ss = improvement[lane];
                            if (!finite_double_bits(ss) || ss <= 0.0) {
                                continue;
                            }
                            const double norm = std::sqrt(ss);
                            const double inv = 1.0 / norm;
                            if (!finite_double_bits(norm) || norm <= 0.0 ||
                                !finite_double_bits(inv) || inv <= 0.0) {
                                continue;
                            }
                            normalize_scale[lane] = inv;
                            restore_scale[lane] = norm;
                            normalized_small_cg = true;
                        }
                        if (normalized_small_cg) {
                            cuda_check(
                                cudaMemcpy(
                                    d_cg_alpha.data(),
                                    normalize_scale.data(),
                                    sizeof(double) *
                                        static_cast<std::size_t>(cg_width),
                                    cudaMemcpyHostToDevice),
                                "cudaMemcpy CG normalization scale failed");
                            scale_columns_kernel<<<blocks_values, kBlockSize>>>(
                                d_y.data(), d_x.data(), d_cg_alpha.data(),
                                total_values, n, ld);
                            cuda_check(
                                cudaGetLastError(),
                                "CG normalization kernel launch failed");
                            compute_component_sumsq(
                                d_y.data(), d_x.data(), improvement);
                        }
                    }
                    projection_sweep(d_y.data(), d_x.data(), d_y_gx.data(), d_x_gx.data());
                    compute_component_sumsq(d_y_gx.data(), d_x_gx.data(), ssr);
                    cuda_check(cudaMemcpy(d_y_ggx.data(), d_y_gx.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy CG direction y failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x_ggx.data(), d_x_gx.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy CG direction X failed");
                    }
                    auto report_cg_breakdown =
                        [&](const char* stage, int iter, int lane,
                            double a, double b, double c) {
                            if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr) {
                                std::fprintf(
                                    stderr,
                                    "[ACCEL cuda CG breakdown stage=%s "
                                    "iter=%d lane=%d a=%.17g b=%.17g c=%.17g]\n",
                                    stage, iter, lane, a, b, c);
                            }
                        };
                    // CG-02 slow-mode continuation (F1). Kill switch
                    // XHDFE_CG_SLOW_MODE_CONTINUE=0 restores the 2.30.1 rule (every
                    // step > tol at an energy-predicted stop breaks down).
                    const bool slow_mode_continue = []() {
                        const char* e = std::getenv("XHDFE_CG_SLOW_MODE_CONTINUE");
                        return !(e != nullptr && e[0] == '0');
                    }();
                    const double slow_mode_floor =
                        100.0 * std::numeric_limits<double>::epsilon();
                    std::vector<double> deferred_step(
                        static_cast<std::size_t>(cg_width), 0.0);
                    bool any_deferred = false;
                    int slow_mode_deferrals = 0;
                    // F1 rev. 2: a genuine slow mode keeps shrinking the step geometrically
                    // (CG), so progress, not a count, stops the deferrals: a lane breaks down
                    // when its step has not fallen by 10 % in kSlowModeStallWindow consecutive
                    // deferrals. kSlowModeMaxDeferrals (cumulative per solve, never reset) is
                    // a hard backstop so a pathological lane still reaches the restart with
                    // budget left. CG-02's direction is rejected by the lambda_p floor first.
                    constexpr int kSlowModeStallWindow = 32;
                    constexpr int kSlowModeMaxDeferrals = 2048;
                    std::vector<int> lane_deferrals(
                        static_cast<std::size_t>(cg_width), 0);
                    std::vector<int> deferrals_since_progress(
                        static_cast<std::size_t>(cg_width), 0);
                    std::vector<double> best_step(
                        static_cast<std::size_t>(cg_width),
                        std::numeric_limits<double>::infinity());
                    const bool cg_accel_debug =
                        std::getenv("XHDFE_ACCEL_DEBUG") != nullptr;
                    auto report_cg_step_guard =
                        [&](const char* event, int iter, int lane, double step,
                            double alpha_lane, double lambda_p, double energy_ratio,
                            int since_progress, double lane_best_step) {
                            if (cg_accel_debug) {
                                std::fprintf(
                                    stderr,
                                    "[ACCEL cuda CG %s iter=%d lane=%d a=%.17g b=%.17g "
                                    "c=%.17g lambda_p=%.17g energy_ratio=%.17g "
                                    "since_progress=%d best_step=%.17g]\n",
                                    event, iter, lane, step, options.tol, alpha_lane,
                                    lambda_p, energy_ratio, since_progress, lane_best_step);
                            }
                        };

                    // P3-fix test hook (XHDFE_TEST_CUDA_CG_BREAKDOWN); 0 unless set.
                    const int test_breakdown_iteration = cuda_cg_test_breakdown().iteration;
                    int cg_iters = 0;
                    for (int iter = 1; iter <= cg_budget; ++iter) {
                        projection_sweep(d_y_ggx.data(), d_x_ggx.data(),
                                         d_y_acc_a.data(), d_x_acc_a.data());
                        compute_component_dot(d_y_ggx.data(), d_y_acc_a.data(),
                                              d_x_ggx.data(), d_x_acc_a.data(), denom);
                        std::fill(alpha.begin(), alpha.end(), 0.0);
                        next_improvement = improvement;
                        bool cg_breakdown = false;
                        for (int k = 0; k < cg_width; ++k) {
                            const std::size_t lane = static_cast<std::size_t>(k);
                            if (!finite_double_bits(improvement[lane]) ||
                                !finite_double_bits(ssr[lane]) ||
                                ssr[lane] < 0.0) {
                                report_cg_breakdown(
                                    "initial", iter, k, improvement[lane],
                                    ssr[lane], 0.0);
                                cg_breakdown = true;
                                break;
                            }
                            if (lane_solved[lane]) {
                                if (ssr[lane] != 0.0) {
                                    report_cg_breakdown(
                                        "solved-alpha", iter, k, ssr[lane],
                                        0.0, 0.0);
                                    cg_breakdown = true;
                                    break;
                                }
                                recent[lane] = 0.0;
                                continue;
                            }
                            if (ssr[lane] == 0.0) {
                                if (!component_exact_zero(
                                        d_y_gx.data(), d_x_gx.data(), k)) {
                                    report_cg_breakdown(
                                        "false-zero-alpha", iter, k,
                                        ssr[lane], 0.0, 0.0);
                                    cg_breakdown = true;
                                    break;
                                }
                                lane_solved[lane] = 1;
                                recent[lane] = 0.0;
                                continue;
                            }
                            if (!finite_double_bits(denom[lane]) ||
                                denom[lane] <= 0.0) {
                                report_cg_breakdown(
                                    "denom", iter, k, denom[lane],
                                    ssr[lane], improvement[lane]);
                                cg_breakdown = true;
                                break;
                            }
                            alpha[lane] = ssr[lane] / denom[lane];
                            recent[lane] = alpha[lane] * ssr[lane];
                            next_improvement[lane] =
                                improvement[lane] - recent[lane];
                            if (!finite_double_bits(alpha[lane]) ||
                                alpha[lane] <= 0.0 ||
                                alpha[lane] > max_cg_step ||
                                !finite_double_bits(recent[lane]) ||
                                recent[lane] <= 0.0 ||
                                !finite_double_bits(next_improvement[lane])) {
                                report_cg_breakdown(
                                    "alpha", iter, k, alpha[lane],
                                    recent[lane],
                                    next_improvement[lane]);
                                cg_breakdown = true;
                                break;
                            }
                        }
                        cg_iters = iter;
                        if (!cg_breakdown && iter == test_breakdown_iteration &&
                            cuda_cg_test_breakdown_due(iter)) {
                            report_cg_breakdown("test-forced", iter, -1, 0.0, 0.0, 0.0);
                            cg_breakdown = true;
                        }
                        if (cg_breakdown) {
                            break;
                        }
                        if (any_deferred) {
                            std::fill(deferred_step.begin(), deferred_step.end(), 0.0);
                            any_deferred = false;
                        }
                        bool candidate_lane = false;
                        for (int k = 0; k < cg_width; ++k) {
                            const std::size_t lane = static_cast<std::size_t>(k);
                            if (!lane_solved[lane] &&
                                std::sqrt(recent[lane] / std::abs(next_improvement[lane])) <= options.tol) {
                                candidate_lane = true;
                            }
                        }
                        // CG02 is a Comparable/Strict accuracy guard. Fast
                        // keeps its published energy-based stopping rule.
                        if (candidate_lane &&
                            options.tolerance_mode != ToleranceMode::XhdfeFast) {
                            std::vector<double> direction_sq(
                                static_cast<std::size_t>(cg_width), 0.0);
                            compute_component_sumsq(d_y_ggx.data(), d_x_ggx.data(), direction_sq);
                            for (int k = 0; k < cg_width; ++k) {
                                const std::size_t lane = static_cast<std::size_t>(k);
                                const double scale = std::abs(next_improvement[lane]);
                                if (!lane_solved[lane] && scale > 0.0 &&
                                    std::sqrt(recent[lane] / scale) <= options.tol) {
                                    const double step = std::abs(alpha[lane]) *
                                        std::sqrt(direction_sq[lane] / scale);
                                    if (finite_double_bits(step) && step <= options.tol) {
                                        continue;
                                    }
                                    // Lesson CG-02 (F1): lambda_p = p'Ap/p'p (Rayleigh quotient of p)
                                    // amplifies sweep roundoff by ~1/lambda_p. While eps/lambda_p is two
                                    // orders below tol, apply the step but defer acceptance; else break down.
                                    const double energy_ratio = std::sqrt(recent[lane] / scale);
                                    const double lambda_p = denom[lane] / direction_sq[lane];
                                    bool defer = slow_mode_continue &&
                                        finite_double_bits(step) &&
                                        finite_double_bits(lambda_p) &&
                                        lambda_p * options.tol >= slow_mode_floor;
                                    const char* event =
                                        defer ? "slow-mode-continue" : "breakdown stage=step-norm";
                                    if (defer) {
                                        if (step < 0.9 * best_step[lane]) {
                                            best_step[lane] = step;
                                            deferrals_since_progress[lane] = 0;
                                        } else {
                                            ++deferrals_since_progress[lane];
                                        }
                                        if (deferrals_since_progress[lane] >= kSlowModeStallWindow) {
                                            defer = false;
                                            event = "breakdown stage=slow-mode-stalled";
                                        } else if (lane_deferrals[lane] >= kSlowModeMaxDeferrals) {
                                            defer = false;
                                            event = "breakdown stage=slow-mode-exhausted";
                                        }
                                    }
                                    report_cg_step_guard(
                                        event, iter, k, step, alpha[lane], lambda_p, energy_ratio,
                                        deferrals_since_progress[lane], best_step[lane]);
                                    if (!defer) {
                                        cg_breakdown = true;
                                        break;
                                    }
                                    deferred_step[lane] = step;
                                    any_deferred = true;
                                    ++lane_deferrals[lane];
                                    ++slow_mode_deferrals;
                                }
                            }
                        }
                        if (cg_breakdown) break;
                        improvement = next_improvement;

                        subtract_columns(d_y.data(), d_x.data(),
                                         d_y_ggx.data(), d_x_ggx.data(), alpha);
                        subtract_columns(d_y_gx.data(), d_x_gx.data(),
                                         d_y_acc_a.data(), d_x_acc_a.data(), alpha);

                        ssr_old = ssr;
                        compute_component_sumsq(d_y_gx.data(), d_x_gx.data(), ssr);
                        std::fill(beta.begin(), beta.end(), 0.0);
                        for (int k = 0; k < cg_width; ++k) {
                            const std::size_t lane = static_cast<std::size_t>(k);
                            if (!finite_double_bits(ssr[lane]) || ssr[lane] < 0.0 ||
                                !finite_double_bits(ssr_old[lane]) ||
                                ssr_old[lane] < 0.0) {
                                report_cg_breakdown(
                                    "residual", iter, k, ssr[lane],
                                    ssr_old[lane], 0.0);
                                cg_breakdown = true;
                                break;
                            }
                            if (lane_solved[lane]) {
                                if (ssr[lane] != 0.0) {
                                    report_cg_breakdown(
                                        "solved-beta", iter, k, ssr[lane],
                                        0.0, 0.0);
                                    cg_breakdown = true;
                                    break;
                                }
                                beta[lane] = 0.0;
                                continue;
                            }
                            if (ssr[lane] == 0.0) {
                                if (!component_exact_zero(
                                        d_y_gx.data(), d_x_gx.data(), k)) {
                                    report_cg_breakdown(
                                        "false-zero-beta", iter, k,
                                        ssr[lane], 0.0, 0.0);
                                    cg_breakdown = true;
                                    break;
                                }
                                lane_solved[lane] = 1;
                                beta[lane] = 0.0;
                                continue;
                            }
                            if (ssr_old[lane] <= 0.0) {
                                report_cg_breakdown(
                                    "ssr-old", iter, k, ssr[lane],
                                    ssr_old[lane], 0.0);
                                cg_breakdown = true;
                                break;
                            }
                            beta[lane] = ssr[lane] / ssr_old[lane];
                            if (!finite_double_bits(beta[lane]) ||
                                beta[lane] <= 0.0) {
                                report_cg_breakdown(
                                    "beta", iter, k, beta[lane],
                                    ssr[lane], ssr_old[lane]);
                                cg_breakdown = true;
                                break;
                            }
                        }
                        if (cg_breakdown) {
                            break;
                        }
                        bool newly_solved = false;
                        double update_error = 0.0;
                        for (int k = 0; k < cg_width; ++k) {
                            const std::size_t lane = static_cast<std::size_t>(k);
                            if (lane_solved[lane]) {
                                continue;
                            }
                            const double den = std::abs(improvement[lane]);
                            if (!finite_double_bits(recent[lane]) ||
                                recent[lane] <= 0.0 ||
                                !finite_double_bits(den) || den <= 0.0) {
                                report_cg_breakdown(
                                    "update", iter, k, recent[lane], den,
                                    improvement[lane]);
                                cg_breakdown = true;
                                break;
                            }
                            const double ratio = recent[lane] / den;
                            if (!finite_double_bits(ratio) || ratio <= 0.0) {
                                report_cg_breakdown(
                                    "ratio", iter, k, ratio, recent[lane],
                                    den);
                                cg_breakdown = true;
                                break;
                            }
                            double err = std::sqrt(ratio);
                            if (any_deferred) {
                                // A lane deferred by the CG-02 slow-mode rule reports the
                                // step it just applied (> tol): not accepted, no stop now.
                                err = std::max(err, deferred_step[lane]);
                            }
                            if (!finite_double_bits(err)) {
                                report_cg_breakdown(
                                    "error", iter, k, err, ratio, 0.0);
                                cg_breakdown = true;
                                break;
                            }
                            update_error = std::max(update_error, err);
                            if (err <= options.tol) {
                                lane_solved[lane] = 1;
                                ssr[lane] = 0.0;
                                newly_solved = true;
                            }
                        }
                        if (cg_breakdown) {
                            break;
                        }
                        if (update_error <= options.tol) {
                            converged = necessary_candidate_ready(
                                d_y.data(), d_x.data(), iter,
                                normalized_small_cg ? &restore_scale : nullptr);
                            // A rejected candidate must be recomputed from
                            // original inputs, never kept alive with zero or
                            // roundoff-sized search directions.
                            break;
                        }
                        if (newly_solved) {
                            for (int k = 0; k < cg_width; ++k) {
                                alpha[static_cast<std::size_t>(k)] =
                                    lane_solved[static_cast<std::size_t>(k)] ? 0.0 : 1.0;
                            }
                            cuda_check(cudaMemcpy(d_cg_alpha.data(), alpha.data(),
                                sizeof(double) * static_cast<std::size_t>(cg_width),
                                cudaMemcpyHostToDevice), "cudaMemcpy CG freeze mask failed");
                            scale_columns_kernel<<<blocks_values, kBlockSize>>>(
                                d_y_gx.data(), d_x_gx.data(), d_cg_alpha.data(),
                                total_values, n, ld);
                            scale_columns_kernel<<<blocks_values, kBlockSize>>>(
                                d_y_ggx.data(), d_x_ggx.data(), d_cg_alpha.data(),
                                total_values, n, ld);
                            cuda_check(cudaGetLastError(), "CG freeze kernel launch failed");
                        }
                        update_direction(d_y_ggx.data(), d_x_ggx.data(),
                                         d_y_gx.data(), d_x_gx.data(), beta);
                    }
                    result.iterations = counted_before + cg_iters;
                    result.cg_restart_required = !converged;
                    have_cg_iteration_count = true;
                    if (normalized_small_cg) {
                        cuda_check(
                            cudaMemcpy(
                                d_cg_alpha.data(), restore_scale.data(),
                                sizeof(double) *
                                    static_cast<std::size_t>(cg_width),
                                cudaMemcpyHostToDevice),
                            "cudaMemcpy CG restore scale failed");
                        scale_columns_kernel<<<blocks_values, kBlockSize>>>(
                            d_y.data(), d_x.data(), d_cg_alpha.data(),
                            total_values, n, ld);
                        cuda_check(
                            cudaGetLastError(),
                            "CG restore-scale kernel launch failed");
                    }
                    if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr) {
                        std::fprintf(stderr, "[ACCEL cuda CG iters=%d converged=%d deferred=%d]\n",
                                     result.iterations, converged ? 1 : 0, slow_mode_deferrals);
                    }
                };

                // F2-B ledger of the CG stage. The stage itself runs as in
                // 2.30.1: its budget and `iterations` exclude the IT sweeps
                // that preceded a hand-off (counted only under the N1
                // continuation). The ledger keeps those uncounted sweeps, the
                // hand-off reason and the wall time of the sweeps and of the
                // CG stage for the cold restart that follows a broken CG stage
                // (absorb_fixed_effects_cuda) and for the trace. With
                // XHDFE_GI_LSMR_TRACE the stage boundaries are synchronised,
                // so the seconds are each stage's device work; no decision
                // reads them, and without the trace nothing is synchronised.
                const bool f2_stage_trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
                if (f2_stage_trace) {
                    cuda_check(cudaDeviceSynchronize(), "IT stage trace synchronisation failed");
                }
                const auto accel_stage_start = std::chrono::steady_clock::now();
                auto run_cg_stage = [&](int preceding_iterations, bool after_sweeps) {
                    if (f2_stage_trace) {
                        cuda_check(cudaDeviceSynchronize(),
                                   "CG stage trace synchronisation failed");
                    }
                    const auto cg_start = std::chrono::steady_clock::now();
                    const int uncounted = necessary_continuation ? 0 : preceding_iterations;
                    const double it_seconds = after_sweeps
                        ? std::chrono::duration<double>(cg_start - accel_stage_start).count()
                        : 0.0;
                    // F2-B: in the three-FE family (switch not at 0) the budget is
                    // cumulative in every stage: the CG stage receives max_iter minus
                    // the IT sweeps before it, so the sweeps, the CG stage and the
                    // restart that may follow never exceed max_iter together.
                    const int budget_charge =
                        after_sweeps && cuda_three_fe_family(options, fe_inputs, weights, cols) &&
                                cuda_tail_handover_mode() != CudaTailMode::Off
                            ? preceding_iterations
                            : 0;
                    const int cg_budget = std::max(
                        0, options.max_iter -
                               std::max(preceding_iterations - uncounted, budget_charge));
                    if (f2_stage_trace && after_sweeps) {
                        std::fprintf(stderr,
                                     "cuda_it_to_cg reason=%s it_iterations=%d it_seconds=%.6f "
                                     "uncounted=%d max_iter=%d cg_budget=%d\n",
                                     accel_handoff_reason == 1 ? "divergence"
                                         : (accel_handoff_reason == 2 ? "cap" : "other"),
                                     preceding_iterations, it_seconds, uncounted,
                                     options.max_iter, cg_budget);
                    }
                    run_cuda_cg(sweep_order, preceding_iterations, budget_charge);
                    if (f2_stage_trace) {
                        cuda_check(cudaDeviceSynchronize(),
                                   "CG stage trace synchronisation failed");
                    }
                    const double cg_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - cg_start).count();
                    result.cuda_cg_uncounted_it_iterations = uncounted;
                    result.cuda_it_handoff_reason = after_sweeps ? accel_handoff_reason : 0;
                    result.cuda_it_seconds = it_seconds;
                    result.cuda_cg_seconds = cg_seconds;
                    if (f2_stage_trace) {
                        std::fprintf(stderr,
                                     "cuda_cg_end entry=%s cg_iterations=%d iterations=%d "
                                     "converged=%d restart_required=%d cg_budget=%d "
                                     "seconds=%.6f\n",
                                     after_sweeps ? "sweeps" : "gate",
                                     result.iterations - (preceding_iterations - uncounted),
                                     result.iterations, converged ? 1 : 0,
                                     result.cg_restart_required ? 1 : 0, cg_budget,
                                     cg_seconds);
                    }
                };

                const bool cg_fallback_possible =
                    cuda_cg_fallback_ok && !sweep_order.empty();
                if (use_reghdfe_cg && !sweep_order.empty()) {
                    run_cg_stage(0, false);
                } else if (!enable_two_stage) {
                    int phase_iters = 0;
                    converged = run_accel_phase(sweep_order, accel_iter_budget, phase_iters);
                    if (!converged && accel_diverged && cg_fallback_possible) {
                        run_cg_stage(phase_iters, true);
                    } else if (converged) {
                        result.iterations = phase_iters;
                    }
                } else {
                    int total_iters = 0;
                    int remaining = options.max_iter;
                    if (iter_warmup > 0 && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_accel_phase(sweep_order,
                                                    std::min(iter_warmup, remaining),
                                                    phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !accel_diverged && !two_fe_order.empty() &&
                        iter_two_fe > 0 && remaining > 0) {
                        int phase_iters = 0;
                        run_accel_phase(two_fe_order,
                                        std::min(iter_two_fe, remaining),
                                        phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !accel_diverged && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_accel_phase(sweep_order, remaining, phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && accel_diverged && cg_fallback_possible) {
                        run_cg_stage(total_iters, true);
                    } else if (converged) {
                        result.iterations = total_iters;
                    }
                }
                // Complete the unfinished FE component from the successful
                // IT state with the existing device Krylov solve. The original
                // ordinary tolerance and remaining iteration budget apply.
                //
                // F2-A: the device solve is probed (CudaTailProbe). When it is
                // predicted long and the CPU's own selector would use MLSMR on
                // this fit, the host MLSMR completes the projection from the
                // same IT state, at the same 1e-11 Krylov floor, within the
                // remaining budget: completed + device probe + host iterations.
                // The host never runs a sweep here. The IT state has already
                // satisfied the sweep's Comparable stop (the norm-of-change rule
                // at the nominal tolerance) while weak FE components were left
                // unfinished; a sweep started from it would accept at its first
                // check and return that incomplete projection, which is the
                // demonstrated covariance error this completion exists to fix
                // (docs/LICOES_CUDA_COVARIANCIA_20261020.md). When the selector
                // would sweep, or does not apply, the device keeps the tail.
                if (allow_cg && converged && !use_reghdfe_cg && !have_cg_iteration_count &&
                    options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
                    fe_inputs.size() == 3 &&
                    (weights == nullptr || cuda_weighted_completion_applies(options)) &&
                    !any_slope &&
                    !store_alphas && !options.retain_fixed_effects && cols > 0) {
                    const int completed = result.iterations;
                    const int remaining = std::max(0, options.max_iter - completed);
                    if (remaining > 0) {
                        Eigen::VectorXd warm_y(n);
                        Eigen::MatrixXd warm_x(ld, cols);
                        cuda_check(cudaMemcpy(warm_y.data(), d_y.data(),
                                              sizeof(double) * n, cudaMemcpyDeviceToHost),
                                   "warm ordinary LSMR y copy failed");
                        cuda_check(cudaMemcpy(warm_x.data(), d_x.data(),
                                              sizeof(double) * x_size, cudaMemcpyDeviceToHost),
                                   "warm ordinary LSMR X copy failed");
                        // P8 (A1), test hook only (inert unless set, read once per
                        // process): XHDFE_TEST_CUDA_COMPLETION_COLD=1 starts the
                        // completion from the original data instead of the IT state,
                        // for the cold/warm comparison with variable weights that
                        // the design requires before trusting the warm start.
                        static const bool completion_cold_hook = [] {
                            const char* e = std::getenv("XHDFE_TEST_CUDA_COMPLETION_COLD");
                            return e != nullptr && std::strcmp(e, "1") == 0;
                        }();
                        if (completion_cold_hook) {
                            warm_y = y;
                            warm_x = X;
                        }
                        HdfeOptions tail_options = options;
                        tail_options.max_iter = remaining;
                        const CudaTailMode tail_mode = cuda_tail_handover_mode();
                        const bool tail_trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
                        CudaTailProbe probe;
                        probe.mode = tail_mode;
                        probe.trace = tail_trace;
                        probe.rhs_count = cols + 1;
                        probe.budget = remaining;
                        probe.select = [&]() {
                            return select_cuda_tail_host_route(n, options);
                        };
                        AbsorptionResult repaired;
                        bool tail_ok = absorb_ordinary_lsmr_cuda(
                            warm_y, warm_x, fe_inputs, weights, tail_options, repaired,
                            // P8 (A1): never a probe with weights, so no F2 hand-over.
                            (tail_mode == CudaTailMode::Off || weights != nullptr) ? nullptr
                                                                                   : &probe);
                        int spent = 0;
                        if (tail_ok && probe.handover) {
                            spent = probe.device_iterations;
                            AbsorptionResult host;
                            int host_iterations = 0;
                            double host_seconds = 0.0;
                            const bool host_ok = complete_cuda_tail_on_host(
                                warm_y, warm_x, fe_inputs, n, tail_options,
                                remaining - spent, host, host_iterations, host_seconds);
                            if (tail_trace) {
                                std::fprintf(stderr,
                                             "cuda_tail_handover completed=%d probe_rhs=%d "
                                             "probe_k=%d device_iterations=%d selector=%s "
                                             "rho=%.17g select_seconds=%.6f "
                                             "host_method=mlsmr host_budget=%d "
                                             "host_iterations=%d host_converged=%d "
                                             "host_seconds=%.6f threads=%d outcome=%s\n",
                                             completed, probe.rhs, probe.k, spent,
                                             probe.selection.selector,
                                             probe.selection.rho,
                                             probe.selection.seconds,
                                             remaining - spent, host_iterations,
                                             host_ok ? 1 : 0, host_seconds,
                                             tail_options.num_threads,
                                             host_ok ? "host-mlsmr"
                                                     : "device-lsmr-after-host");
                            }
                            if (host_ok) {
                                host.iterations = completed + spent + host_iterations;
                                host.mlsmr_used = true;
                                host.precision_certified = false;  // certified by the caller
                                host.sweep_order_used.clear();
                                host.cuda_tail = kCudaTailHostMlsmr;
                                result = std::move(host);
                                return true;
                            }
                            // The host stopped short of its budget or failed: the
                            // device tail finishes with what is left, no probe.
                            spent += host_iterations;
                            tail_options.max_iter = remaining - spent;
                            repaired = AbsorptionResult{};
                            if (tail_options.max_iter > 0) {
                                tail_ok = absorb_ordinary_lsmr_cuda(
                                    warm_y, warm_x, fe_inputs, weights, tail_options,
                                    repaired);
                            } else {
                                repaired.converged = false;
                                repaired.precision_certified = false;
                                repaired.mlsmr_used = true;
                            }
                        }
                        repaired.iterations += completed + spent;
                        repaired.cuda_tail = kCudaTailDeviceLsmr;
                        result = std::move(repaired);
                        return tail_ok;
                    }
                    converged = false;
                }
            } else {
                auto run_simple_phase = [&](const std::vector<std::size_t>& order,
                                            int max_iter,
                                            int& iter_used) -> bool {
                    iter_used = 0;
                    next_necessary_check = 0;
                    if (max_iter <= 0 || order.empty()) {
                        return false;
	                    }
	                    double prev_norm = compute_norm();
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    int last_check_iter = -1;
	                    for (int iter = 0; iter < max_iter; ++iter) {
                        const bool do_check = (check_interval == 1 || iter < check_interval ||
                                               iter % check_interval == 0);
                        double sumsq = 0.0;
                        run_sweep(d_y.data(), d_x.data(), order, do_check, sumsq,
                                  store_alphas ? &alpha_state_ptrs : nullptr);
                        if (do_check) {
                            const double curr_norm = std::sqrt(sumsq);
                            const double denom = std::max(1.0, prev_norm);
                            const double rel_change = std::abs(curr_norm - prev_norm) / denom;
	                            const int step = iter - last_check_iter;
	                            last_check_iter = iter;
	                            prev_norm = curr_norm;
	                            double update_error = std::numeric_limits<double>::max();
	                            compute_update_error(d_y.data(), d_x.data(), update_error);
	                            copy_update_snapshot(d_y.data(), d_x.data());
	                            const double norm_error = rel_change / static_cast<double>(step);
	                            if (step > 0 &&
	                                convergence_reached(options, norm_error, update_error,
	                                                    convergence_tol) &&
                                    necessary_candidate_ready(d_y.data(), d_x.data(), iter)) {
	                                iter_used = iter + 1;
                                return true;
                            }
                        }
                    }
                    iter_used = max_iter;
                    return false;
                };

                if (!enable_two_stage) {
                    int phase_iters = 0;
                    converged = run_simple_phase(sweep_order, options.max_iter, phase_iters);
                    if (converged) {
                        result.iterations = phase_iters;
                    }
                } else {
                    int total_iters = 0;
                    int remaining = options.max_iter;
                    if (iter_warmup > 0 && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_simple_phase(sweep_order,
                                                     std::min(iter_warmup, remaining),
                                                     phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                        int phase_iters = 0;
                        run_simple_phase(two_fe_order,
                                         std::min(iter_two_fe, remaining),
                                         phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_simple_phase(sweep_order, remaining, phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (converged) {
                        result.iterations = total_iters;
                    }
                }
            }

            if (converged && store_alphas && cols == 0 &&
                fe_tol > 0.0 && result.iterations < options.max_iter) {
                // Complete the existing necessary FE-update check on-device.
                // Norm convergence alone does not imply the requested max delta.
                d_recovery_max_delta.allocate(1);
                recovery_delta_requested = true;
                converged = false;
                while (result.iterations < options.max_iter) {
                    cuda_check(cudaMemset(d_recovery_max_delta.data(), 0, sizeof(double)),
                               "recovery delta reset failed");
                    double ignored = 0.0;
                    run_sweep(d_y.data(), d_x.data(), sweep_order, false, ignored,
                              &alpha_state_ptrs);
                    ++result.iterations;
                    double max_delta = 0.0;
                    cuda_check(cudaMemcpy(&max_delta, d_recovery_max_delta.data(),
                                          sizeof(double), cudaMemcpyDeviceToHost),
                               "recovery delta read failed");
                    if (finite_double_bits(max_delta) && max_delta <= fe_tol) {
                        converged = true;
                        break;
                    }
                }
                recovery_delta_requested = false;
            }
            result.converged = converged;
            if (!converged && !have_cg_iteration_count) {
                result.iterations = options.max_iter;
            } else if (converged && result.iterations == 0) {
                result.iterations = 1;
            }
        }

        struct ForwardProbeState {
            long long sequence = 0;
            int continuation_sweeps = 4;
            double exit_gram_ms = 0.0;
            DeviceBuffer<double> partials;
            DeviceBuffer<double> device_gram;
            DeviceBuffer<double> rss_partials;
            DeviceBuffer<double> rss_scalar;
            DeviceBuffer<double> coefficients;
            DeviceBuffer<double> scratch_y;
            DeviceBuffer<double> scratch_x;
            CudaForwardProbeGram gram_exit;
            CudaForwardProbeGram gram_public;
            CudaForwardProbeGram gram_continuation;
            CudaForwardProbeGram gram_second_continuation;
            CudaForwardProbeFit fit_exit;
            CudaForwardProbeFit fit_public;
            CudaForwardProbeFit fit_continuation;
            CudaForwardProbeFit fit_second_continuation;
        };
        std::unique_ptr<ForwardProbeState> forward_probe;
        const bool forward_probe_requested = cuda_forward_probe_enabled();
        const bool forward_probe_thread_requested =
            thread_cuda_forward_probe_requested();
        bool forward_probe_selector_shape =
            forward_probe_thread_requested;
        if (forward_probe_requested && !forward_probe_thread_requested) {
            auto env_int = [](const char* name, int fallback, int minimum) {
                const char* raw = std::getenv(name);
                const int parsed = raw != nullptr ? std::atoi(raw) : fallback;
                return std::max(minimum, parsed);
            };
            const int min_rows = env_int(
                "XHDFE_AUTO_MLSMR_MIN_ROWS", 200000, 1);
            const int min_rhs = env_int(
                "XHDFE_AUTO_MLSMR_MIN_RHS", 2, 1);
            const int max_fes = env_int(
                "XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES", 3, 1);
            const int four_fe_max_rows = env_int(
                "XHDFE_AUTO_MLSMR_DEFAULT_4FE_MAX_ROWS", 2000000, 1);
            const bool bounded_four_fe =
                max_fes >= 3 && active_fes == 4 &&
                n <= four_fe_max_rows;
            forward_probe_selector_shape =
                n >= min_rows && cols >= min_rhs &&
                (static_cast<int>(active_fes) <= max_fes ||
                 bounded_four_fe);
        }
        const bool forward_probe_eligible =
            forward_probe_requested && result.converged &&
            forward_probe_selector_shape &&
            method != AbsorptionMethod::Jacobi && !any_slope && unit_weights &&
            !store_alphas && !options.retain_fixed_effects && cols >= 1 &&
            cols + 1 + (options.fit_intercept ? 1 : 0) <=
                kForwardProbeMaxP &&
            active_fes > 0;
        auto capture_forward_probe_gram = [&](ForwardProbeState& state,
                                              const double* y_values,
                                              const double* x_values,
                                              CudaForwardProbeGram& output) {
            const int p = cols + 1 + (options.fit_intercept ? 1 : 0);
            const std::size_t pairs =
                static_cast<std::size_t>(p) * static_cast<std::size_t>(p);
            state.partials.allocate(
                pairs * static_cast<std::size_t>(kForwardProbeGrid));
            state.device_gram.allocate(pairs);
            const dim3 grid(kForwardProbeGrid, p, p);
            forward_probe_gram_partials_kernel<<<grid, kBlockSize>>>(
                y_values, x_values, n, ld, cols, p,
                state.partials.data());
            cuda_check(cudaGetLastError(),
                       "forward probe Gram partials kernel launch failed");
            forward_probe_gram_finish_kernel<<<1, kBlockSize>>>(
                state.partials.data(), p, kForwardProbeGrid,
                state.device_gram.data());
            cuda_check(cudaGetLastError(),
                       "forward probe Gram finish kernel launch failed");
            cuda_check(cudaMemcpy(output.values, state.device_gram.data(),
                                  sizeof(double) * pairs,
                                  cudaMemcpyDeviceToHost),
                       "forward probe Gram copy failed");
            output.p = p;
            output.centered = options.fit_intercept;
        };
        auto capture_forward_probe_rss = [&](ForwardProbeState& state,
                                             const double* y_values,
                                             const double* x_values,
                                             CudaForwardProbeFit& fit) {
            if (!fit.ok) {
                throw std::runtime_error(
                    "forward probe cannot compute RSS from an invalid fit");
            }
            state.rss_partials.allocate(kForwardProbeGrid);
            state.rss_scalar.allocate(1);
            state.coefficients.allocate(static_cast<std::size_t>(cols));
            cuda_check(cudaMemcpy(
                           state.coefficients.data(), fit.coefficients,
                           sizeof(double) * static_cast<std::size_t>(cols),
                           cudaMemcpyHostToDevice),
                       "forward probe coefficient copy failed");
            forward_probe_rss_partials_kernel
                <<<kForwardProbeGrid, kBlockSize>>>(
                    y_values, x_values, state.coefficients.data(),
                    fit.intercept, n, ld, cols,
                    state.rss_partials.data());
            cuda_check(cudaGetLastError(),
                       "forward probe RSS partials kernel launch failed");
            forward_probe_rss_finish_kernel<<<1, 1>>>(
                state.rss_partials.data(), kForwardProbeGrid,
                state.rss_scalar.data());
            cuda_check(cudaGetLastError(),
                       "forward probe RSS finish kernel launch failed");
            double stable_rss = 0.0;
            cuda_check(cudaMemcpy(&stable_rss, state.rss_scalar.data(),
                                  sizeof(double), cudaMemcpyDeviceToHost),
                       "forward probe RSS copy failed");
            if (!(stable_rss > 0.0) || !finite_double_bits(stable_rss)) {
                throw std::runtime_error(
                    "forward probe RSS is non-positive or non-finite");
            }
            const double scale = stable_rss / fit.rss;
            const double se_scale = std::sqrt(scale);
            for (int column = 0; column < cols; ++column) {
                fit.standard_errors[column] *= se_scale;
                for (int other = 0; other < cols; ++other) {
                    fit.covariance[
                        column * kForwardProbeMaxP + other] *= scale;
                }
            }
            fit.rss = stable_rss;
        };
        if (forward_probe_eligible) {
            forward_probe = std::make_unique<ForwardProbeState>();
            forward_probe->sequence =
                g_cuda_forward_probe_sequence.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            forward_probe->continuation_sweeps =
                cuda_forward_probe_sweeps();
            const auto probe_exit_start = std::chrono::steady_clock::now();
            try {
                capture_forward_probe_gram(
                    *forward_probe, d_y.data(), d_x.data(),
                    forward_probe->gram_exit);
                forward_probe->exit_gram_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - probe_exit_start)
                        .count();
            } catch (const std::exception& error) {
                if (cuda_forward_probe_diag_enabled()) {
                    std::fprintf(
                        stderr,
                        "cuda_forward_probe seq=%lld status=exit_error n=%d "
                        "cols=%d dims=%zu error=%s\n",
                        forward_probe->sequence, n, cols, active_fes,
                        error.what());
                }
                result.cuda_forward_probe.status = 3;
                result.cuda_forward_probe.continuation_sweeps =
                    forward_probe->continuation_sweeps;
                (void)cudaGetLastError();
                forward_probe.reset();
            }
        }

        int published_polish_sweeps = 0;
        if (result.converged && active_fes > 0) {
            constexpr int kPolishSweeps = 6;
            const bool strict_tolerance = strict_residual_tolerance_mode(options);
            const int max_polish_sweeps =
                strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                                 : kPolishSweeps;
            const double polish_tol = strict_tolerance ? options.tol : 0.0;
            const bool polish_symmetric =
                options.symmetric_sweep && active_fes > 1;
            double final_max = strict_tolerance ? std::numeric_limits<double>::infinity() : 0.0;
            int polish_done = 0;
            for (int polish = 0; polish < max_polish_sweeps; ++polish) {
                if (strict_tolerance) {
                    cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                               "cudaMemset polish max_abs failed");
                }
                for (std::size_t dim = 0; dim < active_fes; ++dim) {
                    double ignored = 0.0;
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas) {
                        alpha_y_ptr = alpha_state_ptrs.y[dim];
                        alpha_x_ptr = alpha_state_ptrs.x[dim];
                    }
                    run_demean(fe_dev[dim], d_y.data(), d_x.data(), false, ignored, 1.0,
                               alpha_y_ptr, alpha_x_ptr);
                    if (strict_tolerance) {
                        const int blocks_g =
                            (fe_dev[dim].num_groups + kBlockSize - 1) / kBlockSize;
                        max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                            fe_dev[dim].sum_y.data(), fe_dev[dim].num_groups,
                            d_sumsq.data());
                        cuda_check(cudaGetLastError(),
                                   "polish max_abs kernel launch failed");
                    }
                }
                if (polish_symmetric) {
                    for (std::size_t idx = active_fes; idx-- > 0;) {
                        double ignored = 0.0;
                        double* alpha_y_ptr = nullptr;
                        double* alpha_x_ptr = nullptr;
                        if (store_alphas) {
                            alpha_y_ptr = alpha_state_ptrs.y[idx];
                            alpha_x_ptr = alpha_state_ptrs.x[idx];
                        }
                        run_demean(fe_dev[idx], d_y.data(), d_x.data(), false, ignored, 1.0,
                                   alpha_y_ptr, alpha_x_ptr);
                        if (strict_tolerance) {
                            const int blocks_g =
                                (fe_dev[idx].num_groups + kBlockSize - 1) / kBlockSize;
                            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                                fe_dev[idx].sum_y.data(), fe_dev[idx].num_groups,
                                d_sumsq.data());
                            cuda_check(cudaGetLastError(),
                                       "polish max_abs kernel launch failed");
                        }
                    }
                }
                ++polish_done;
                if (strict_tolerance) {
                    cuda_check(cudaMemcpy(&final_max, d_sumsq.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy polish max_abs failed");
                    if (final_max <= polish_tol) {
                        break;
                    }
                }
            }
            published_polish_sweeps = polish_done;
            if (strict_tolerance) {
                result.iterations += polish_done;
                if (final_max > polish_tol) {
                    result.converged = false;
                }
            }
        }

        cuda_prof.solve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cuda_prof_solve_t0).count();
        emit_cuda_accel_trace(accel_trace, result.iterations, result.converged);

        // WP3 shadow certificate: the residuals are still device-resident;
        // run the device verifier before any D2H. The host certificate that
        // runs later remains authoritative; this result is only logged for
        // the Gate-3 decision-identity comparison.
        if (cert_shadow.h != nullptr) {
            CudaCertShadowResult shadow{};
            const double shadow_eff = effective_absorption_tolerance(options);
            const double shadow_limit =
                std::max(8.0 * std::max(0.0, shadow_eff),
                         64.0 * std::numeric_limits<double>::epsilon());
            const bool shadow_ok = cuda_cert_shadow_finish(
                cert_shadow.h, d_y.data(), cols > 0 ? d_x.data() : nullptr,
                shadow_limit, &shadow);
            cert_shadow.h = nullptr;  // finish always frees the handle
            static const char* kVerdictNames[4] = {"PASS", "FAIL", "INDET",
                                                   "ERROR"};
            std::fprintf(
                stderr,
                "cuda_cert_shadow ok=%d verdict=%s host_like_pass=%d "
                "rel=%.17e rel_perdim=%.17e r_lo=%.17e r_up=%.17e L=%.17e "
                "wsum_maxdiff=%.3e wsum_tol=%.3e wsum_ok=%d "
                "empty_groups=%lld dims=%d rhs=%d "
                "begin_ms=%.3f finish_ms=%.3f\n",
                shadow_ok ? 1 : 0,
                kVerdictNames[shadow.verdict & 3], shadow.host_like_pass,
                shadow.rel_max, shadow.rel_max_perdim, shadow.r_lower,
                shadow.r_upper, shadow_limit, shadow.wsum_maxdiff,
                shadow.wsum_tol, shadow.wsum_ok,
                shadow.empty_groups, shadow.dims, shadow.rhs,
                shadow.shadow_begin_ms, shadow.shadow_finish_ms);
        }

        if (forward_probe) {
            const auto continuation_start = std::chrono::steady_clock::now();
            try {
                capture_forward_probe_gram(
                    *forward_probe, d_y.data(), d_x.data(),
                    forward_probe->gram_public);
                (void)fit_cuda_forward_probe_gram(
                    forward_probe->gram_public, n,
                    forward_probe->fit_public);
                capture_forward_probe_rss(
                    *forward_probe, d_y.data(), d_x.data(),
                    forward_probe->fit_public);
                forward_probe->scratch_y.allocate(static_cast<std::size_t>(n));
                forward_probe->scratch_x.allocate(x_size);
                cuda_check(cudaMemcpy(forward_probe->scratch_y.data(),
                                      d_y.data(), sizeof(double) * n,
                                      cudaMemcpyDeviceToDevice),
                           "forward probe y snapshot failed");
                cuda_check(cudaMemcpy(forward_probe->scratch_x.data(),
                                      d_x.data(), sizeof(double) * x_size,
                                      cudaMemcpyDeviceToDevice),
                           "forward probe X snapshot failed");

                const bool probe_symmetric =
                    options.symmetric_sweep && active_fes > 1;
                auto run_probe_sweep = [&]() {
                    for (std::size_t dim = 0; dim < active_fes; ++dim) {
                        double ignored = 0.0;
                        run_demean(
                            fe_dev[dim], forward_probe->scratch_y.data(),
                            forward_probe->scratch_x.data(), false, ignored,
                            1.0, nullptr, nullptr);
                    }
                    if (probe_symmetric) {
                        for (std::size_t dim = active_fes; dim-- > 0;) {
                            double ignored = 0.0;
                            run_demean(
                                fe_dev[dim],
                                forward_probe->scratch_y.data(),
                                forward_probe->scratch_x.data(), false,
                                ignored, 1.0, nullptr, nullptr);
                        }
                    }
                };

                for (int sweep = 0;
                     sweep < forward_probe->continuation_sweeps; ++sweep) {
                    run_probe_sweep();
                }
                capture_forward_probe_gram(
                    *forward_probe, forward_probe->scratch_y.data(),
                    forward_probe->scratch_x.data(),
                    forward_probe->gram_continuation);
                (void)fit_cuda_forward_probe_gram(
                    forward_probe->gram_continuation, n,
                    forward_probe->fit_continuation);
                capture_forward_probe_rss(
                    *forward_probe, forward_probe->scratch_y.data(),
                    forward_probe->scratch_x.data(),
                    forward_probe->fit_continuation);
                for (int sweep = 0;
                     sweep < forward_probe->continuation_sweeps; ++sweep) {
                    run_probe_sweep();
                }
                capture_forward_probe_gram(
                    *forward_probe, forward_probe->scratch_y.data(),
                    forward_probe->scratch_x.data(),
                    forward_probe->gram_second_continuation);
                (void)fit_cuda_forward_probe_gram(
                    forward_probe->gram_second_continuation, n,
                    forward_probe->fit_second_continuation);
                capture_forward_probe_rss(
                    *forward_probe, forward_probe->scratch_y.data(),
                    forward_probe->scratch_x.data(),
                    forward_probe->fit_second_continuation);

                (void)fit_cuda_forward_probe_gram(
                    forward_probe->gram_exit, n,
                    forward_probe->fit_exit);
                const double continuation_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        continuation_start)
                        .count();
                result.cuda_forward_probe = emit_cuda_forward_probe(
                    forward_probe->sequence, n, cols,
                    static_cast<int>(active_fes), honest_mode, use_accel,
                    result.iterations, published_polish_sweeps,
                    forward_probe->continuation_sweeps,
                    forward_probe->exit_gram_ms, continuation_ms,
                    cuda_forward_probe_diag_enabled(),
                    forward_probe->gram_public,
                    forward_probe->gram_second_continuation,
                    forward_probe->fit_exit,
                    forward_probe->fit_public,
                    forward_probe->fit_continuation,
                    forward_probe->fit_second_continuation);
                const int forced_status =
                    cuda_forward_probe_forced_status();
                if (forced_status != 0) {
                    result.cuda_forward_probe.status = forced_status;
                    if (cuda_forward_probe_diag_enabled()) {
                        std::fprintf(
                            stderr,
                            "cuda_forward_probe seq=%lld status=forced%d\n",
                            forward_probe->sequence, forced_status);
                    }
                }
            } catch (const std::exception& error) {
                if (cuda_forward_probe_diag_enabled()) {
                    std::fprintf(
                        stderr,
                        "cuda_forward_probe seq=%lld status=error n=%d cols=%d "
                        "dims=%zu error=%s\n",
                        forward_probe->sequence, n, cols, active_fes,
                        error.what());
                }
                result.cuda_forward_probe.status = 3;
                result.cuda_forward_probe.continuation_sweeps =
                    forward_probe->continuation_sweeps;
                (void)cudaGetLastError();
            }
        } else if (forward_probe_requested) {
            if (result.cuda_forward_probe.status == 0) {
                result.cuda_forward_probe.status = 4;
                result.cuda_forward_probe.continuation_sweeps =
                    cuda_forward_probe_sweeps();
            }
            if (cuda_forward_probe_diag_enabled()) {
                std::fprintf(
                    stderr,
                    "cuda_forward_probe seq=0 status=skipped n=%d cols=%d "
                    "dims=%zu converged=%d slopes=%d weights=%d savefe=%d "
                    "method=%d\n",
                    n, cols, active_fes, result.converged ? 1 : 0,
                    any_slope ? 1 : 0, unit_weights ? 0 : 1,
                    options.retain_fixed_effects ? 1 : 0,
                    static_cast<int>(method));
            }
        }

        const auto cuda_prof_d2h_t0 = std::chrono::steady_clock::now();
        result.y_tilde.resize(n);
        staged_memcpy_d2h(result.y_tilde.data(), d_y.data(), sizeof(double) * n,
                          workspace.h_stage, "cudaMemcpy y_tilde failed");
        if (cols > 0) {
            result.X_tilde.resize(ld, cols);
            staged_memcpy_d2h(result.X_tilde.data(), d_x.data(),
                              sizeof(double) * ld * cols, workspace.h_stage,
                              "cudaMemcpy X_tilde failed");
        } else {
            result.X_tilde.resize(ld, 0);
        }

        if (options.retain_fixed_effects) {
            result.fe_group_ids.resize(fe_inputs.size());
            result.fe_weight_sums.resize(fe_inputs.size());
            if (store_alphas) {
                result.fe_alpha_y.resize(fe_inputs.size());
                result.fe_alpha_X.resize(fe_inputs.size());
                for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                    const int groups = fe_inputs[dim].num_groups;
                    result.fe_group_ids[dim].assign(fe_inputs[dim].group_ids,
                                                    fe_inputs[dim].group_ids + n);
                    result.fe_weight_sums[dim].resize(groups);
                    std::memcpy(result.fe_weight_sums[dim].data(), fe_inputs[dim].weight_sums,
                                sizeof(double) * groups);
                    result.fe_alpha_y[dim].resize(groups);
                    cuda_check(cudaMemcpy(result.fe_alpha_y[dim].data(),
                                          fe_dev[dim].alpha_y.data(),
                                          sizeof(double) * groups, cudaMemcpyDeviceToHost),
                               "cudaMemcpy fe alpha_y failed");
                    result.fe_alpha_X[dim].resize(groups, cols);
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(result.fe_alpha_X[dim].data(),
                                              fe_dev[dim].alpha_x.data(),
                                              sizeof(double) * groups * cols,
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy fe alpha_X failed");
                    }
                }
                result.fe_means.clear();
            } else {
                result.fe_means.resize(fe_inputs.size());
                for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                    const int groups = fe_inputs[dim].num_groups;
                    result.fe_group_ids[dim].assign(fe_inputs[dim].group_ids,
                                                    fe_inputs[dim].group_ids + n);
                    result.fe_weight_sums[dim].resize(groups);
                    std::memcpy(result.fe_weight_sums[dim].data(), fe_inputs[dim].weight_sums,
                                sizeof(double) * groups);
                    result.fe_means[dim].resize(groups);
                    cuda_check(cudaMemcpy(result.fe_means[dim].data(), fe_dev[dim].sum_y.data(),
                                          sizeof(double) * groups, cudaMemcpyDeviceToHost),
                               "cudaMemcpy fe means failed");
                }
            }
        }

        cuda_prof.d2h_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cuda_prof_d2h_t0).count();

        result.sweep_order_used.clear();
        if (method == AbsorptionMethod::Jacobi || sweep_order.empty()) {
            result.sweep_order_used.reserve(fe_inputs.size());
            for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                result.sweep_order_used.push_back(static_cast<int>(dim));
            }
        } else {
            result.sweep_order_used.reserve(sweep_order.size());
            for (const std::size_t dim : sweep_order) {
                result.sweep_order_used.push_back(static_cast<int>(dim));
            }
        }

        return true;
    } catch (const std::exception& e) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] absorb_fixed_effects_cuda failed: %s\n", e.what());
        }
        cudaGetLastError();  // clear the error so the next call starts clean
        return false;
    } catch (...) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] absorb_fixed_effects_cuda failed: unknown\n");
        }
        cudaGetLastError();
        return false;
    }
}

// F2-B: the cold restart of a three-FE Comparable fit (unweighted, ordinary
// FEs, regressors present) after a broken CG stage, probed like the F2-A tail.
// The budget accumulates every stage of the fit: the IT sweeps before the
// hand-off to the CG stage (which the CG stage does not count in its own
// `iterations`, but which already limited its budget: run_cg_stage), the CG
// stage, the device probe and the host; each stage receives what the stages
// before it left of options.max_iter, which is never reset, and `iterations`
// counts all of it, never more than max_iter. The device LSMR starts cold from the original data, as the 2.30.1
// restart. When the probe predicts a long solve (work >= 8192 RHS-iterations
// with budget left) and the CPU selector, evaluated at most once per fit,
// votes MLSMR (or under "force"), the host MLSMR solves the fit cold from the
// same original data: never from the CG iterate, whose broken recurrence may
// have lost null-space components. A sweep vote, a selector that does not
// apply or a short prediction keep the 2.30.1 device restart. A host failure
// is contained: the device restarts cold with what is left, without probe.
static bool restart_three_fe_after_cg(const Eigen::Ref<const Eigen::VectorXd>& y,
                                      const Eigen::Ref<const Eigen::MatrixXd>& X,
                                      const std::vector<GpuFeInput>& fe_inputs,
                                      const HdfeOptions& options,
                                      CudaTailMode mode,
                                      AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const bool trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
    CudaRestartLedger ledger;
    ledger.mode = mode == CudaTailMode::Force ? "force" : "adaptive";
    ledger.it_iterations = std::max(0, result.cuda_cg_uncounted_it_iterations);
    ledger.it_reason = result.cuda_it_handoff_reason;
    ledger.it_seconds = result.cuda_it_seconds;
    ledger.cg_iterations = std::max(0, result.iterations);
    ledger.cg_seconds = result.cuda_cg_seconds;
    ledger.max_iter = options.max_iter;
    ledger.threads = options.num_threads;
    const std::int64_t counted =
        static_cast<std::int64_t>(ledger.it_iterations) + ledger.cg_iterations;
    ledger.counted_before = static_cast<int>(
        std::min<std::int64_t>(counted, std::numeric_limits<int>::max()));
    ledger.budget = static_cast<int>(
        std::max<std::int64_t>(0, static_cast<std::int64_t>(options.max_iter) - counted));
    CudaTailProbe probe;
    probe.mode = mode;
    probe.trace = trace;
    probe.stage = "restart";
    probe.rhs_count = cols + 1;
    probe.budget = ledger.budget;
    probe.select = [&]() { return select_cuda_tail_host_route(n, options); };
    if (ledger.budget <= 0) {
        // The sweeps and the CG stage left nothing (the CG stage stopped at
        // max_iter - IT): refuse with every stage counted (IT + CG = max_iter),
        // as 2.30.1 does when its CG stage alone exhausts the budget.
        result.iterations = ledger.counted_before;
        result.converged = false;
        result.precision_certified = false;
        ledger.iterations = result.iterations;
        ledger.outcome = "restart->budget-exhausted";
        if (trace) ledger.print(&probe);
        return true;
    }
    HdfeOptions retry_options = options;
    retry_options.max_iter = ledger.budget;
    AbsorptionResult repaired;
    auto start = std::chrono::steady_clock::now();
    bool repaired_ok = absorb_ordinary_lsmr_cuda(y, X, fe_inputs, nullptr, retry_options,
                                                 repaired, &probe);
    ledger.device_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ledger.device_iterations = std::max(0, repaired.iterations);
    int spent = 0;
    if (repaired_ok && probe.handover) {
        spent = probe.device_iterations;
        ledger.device_iterations = spent;
        ledger.host_budget = ledger.budget - spent;
        AbsorptionResult host;
        int host_iterations = 0;
        double host_seconds = 0.0;
        bool host_ok = false;
        {
            // The host absorber takes its inputs by value: one explicit copy of
            // the original data (never the CG iterate).
            const Eigen::VectorXd y_original = y;
            const Eigen::MatrixXd X_original = X;
            host_ok = complete_cuda_tail_on_host(y_original, X_original, fe_inputs, n,
                                                 retry_options, ledger.host_budget, host,
                                                 host_iterations, host_seconds);
        }
        ledger.host_iterations = host_iterations;
        ledger.host_converged = host_ok;
        ledger.host_seconds = host_seconds;
        if (host_ok) {
            host.iterations = ledger.counted_before + spent + host_iterations;
            host.mlsmr_used = true;
            host.precision_certified = false;  // certified by the caller
            host.sweep_order_used.clear();
            host.cuda_tail = kCudaTailHostMlsmr;
            result = std::move(host);
            ledger.iterations = result.iterations;
            ledger.outcome = "restart->host-mlsmr";
            if (trace) ledger.print(&probe);
            return true;
        }
        // The host stopped short of its budget or failed: the device restarts
        // cold with what is left, without probe.
        spent += host_iterations;
        retry_options.max_iter = ledger.budget - spent;
        repaired = AbsorptionResult{};
        ledger.outcome = "restart->device-lsmr-after-host";
        if (retry_options.max_iter > 0) {
            start = std::chrono::steady_clock::now();
            repaired_ok = absorb_ordinary_lsmr_cuda(y, X, fe_inputs, nullptr,
                                                    retry_options, repaired);
            ledger.after_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            ledger.after_iterations = std::max(0, repaired.iterations);
        } else {
            repaired.converged = false;
            repaired.precision_certified = false;
            repaired.mlsmr_used = true;
        }
    }
    repaired.iterations += ledger.counted_before + spent;
    result = std::move(repaired);
    ledger.iterations = result.iterations;
    if (trace) ledger.print(&probe);
    return repaired_ok;
}

// P3-fix (B): the restart of a broken xhdfe-fast CG stage in the shape of
// cuda_fast_cg_restart_applies. That is the route of the P3 reproducer (the
// public cov240k fixture: 240 000 rows, three FEs, tol 1e-8), where the 2.30.1
// restart solved the fit again with a cold MAP under the Comparable stopping
// rule (eff_tol 1e-9, norm of change; contraction 0.996 at its stop): against
// an independent oracle that state was 43x (SE), 82x (V) and about 790x
// (residuals) worse than the worst 2.28.1 fit, refused at tol 1e-8 by the
// 0.5*tol repair gate and returned certified above 1e-8.
//
// 1. Up to kFastCgColdRestarts cold restarts of the same CG stage: the fit's
//    own xhdfe-fast options (public tolerance, the CG's energy stopping rule,
//    no Comparable floor), from the original data, each within what the
//    earlier attempts left of max_iter. Why a cold restart can finish where the
//    first attempt broke down: the stage combines its reductions with atomics
//    in an order that changes between launches (n > kCudaSmallCgMaxRows), and
//    on the reproducer the breakdown is a rounding event at the stop decision,
//    not a property of the data: with the same inputs, 43 of 52 fresh
//    processes converged at iteration 45 and 9 broke down at iteration 46 (a
//    lane already at roundoff, alpha above the 1/sqrt(eps) ceiling). A
//    deterministic stage would replay its breakdown and is outside this shape.
//    Whether consecutive attempts in one process are independent is measured
//    by the permanent test, not assumed by any decision here.
// 2. When every attempt broke down and budget is left (and kFastCgRestartLsmrTier;
//    otherwise the fit is refused), the device LSMR at the public xhdfe-fast
//    tolerance (no 1e-11 floor, no refinement: the 2.26.2 LSMR semantics) solves
//    the fit from the original data with what is left:
//    the device counterpart of the CPU restart of a broken CG stage, which runs
//    MLSMR with the caller's options (hdfe_regressor_v11.cpp,
//    primary.cg_restart_required). The caller certifies the result like any
//    CUDA absorption and applies its acceptance gates unchanged.
// A broken attempt is never reused (AbsorptionResult::cg_restart_required),
// and the 2.30.1 cold MAP never runs in this shape. `iterations` counts the
// broken stage, every attempt and the LSMR, never more than max_iter; with no
// budget left the fit is refused with all of it counted. One trace line
// (XHDFE_GI_LSMR_TRACE) records the ledger.
static bool restart_fast_cg_after_breakdown(const Eigen::Ref<const Eigen::VectorXd>& y,
                                            const Eigen::Ref<const Eigen::MatrixXd>& X,
                                            const std::vector<GpuFeInput>& fe_inputs,
                                            const Eigen::VectorXd* weights,
                                            const std::vector<std::size_t>& sweep_order,
                                            const HdfeOptions& options,
                                            AbsorptionMethod method,
                                            double solver_fe_tolerance,
                                            AbsorptionResult& result) {
    const bool trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
    const auto start = std::chrono::steady_clock::now();
    const int broken_iterations = std::max(0, result.iterations);
    int spent = std::min(options.max_iter, broken_iterations);
    int restarts = 0;
    int lsmr_iterations = 0;
    bool lsmr_used = false;
    bool finished = false;
    bool ok = true;
    const char* outcome = "budget-exhausted";
    std::string attempt_iterations;  // iterations of every cold restart (trace only)
    while (!finished && restarts < kFastCgColdRestarts && spent < options.max_iter) {
        HdfeOptions retry_options = options;
        retry_options.max_iter = options.max_iter - spent;
        AbsorptionResult attempt;
        const bool attempt_ok = absorb_fixed_effects_cuda_impl(
            y, X, fe_inputs, weights, sweep_order, retry_options, method, attempt,
            solver_fe_tolerance, true);
        ++restarts;
        const int used = std::max(0, attempt.iterations) +
                         std::max(0, attempt.cuda_cg_uncounted_it_iterations);
        spent = std::min(options.max_iter, spent + used);
        if (trace) {
            if (!attempt_iterations.empty()) attempt_iterations += ',';
            attempt_iterations += std::to_string(used);
        }
        if (!attempt_ok || !attempt.cg_restart_required) {
            // A CUDA failure ends the restart, as in 2.30.1; otherwise the
            // stage finished without a breakdown.
            outcome = !attempt_ok ? "cg-restart-failed"
                    : (attempt.converged ? "cg-restart-converged" : "cg-restart-stopped");
            ok = attempt_ok;
            result = std::move(attempt);
            finished = true;
        } else {
            outcome = "cg-restarts-broken";
        }
    }
    if (!finished && kFastCgRestartLsmrTier && spent < options.max_iter) {
        HdfeOptions retry_options = options;
        retry_options.max_iter = options.max_iter - spent;
        AbsorptionResult repaired;
        ok = absorb_ordinary_lsmr_cuda(y, X, fe_inputs, weights, retry_options, repaired);
        lsmr_used = true;
        lsmr_iterations = std::max(0, repaired.iterations);
        spent = std::min(options.max_iter, spent + lsmr_iterations);
        outcome = !ok ? "lsmr-failed" : (repaired.converged ? "lsmr-converged" : "lsmr-stopped");
        result = std::move(repaired);
        finished = true;
    }
    if (!finished) {
        // No budget left, or every attempt broke down and the LSMR tier is off:
        // refuse with every stage counted.
        outcome = spent < options.max_iter ? "cg-restarts-exhausted" : "budget-exhausted";
        result.converged = false;
        result.precision_certified = false;
    }
    result.iterations = spent;
    result.cuda_fast_cg_restarts = restarts;
    result.cuda_fast_lsmr_restart = lsmr_used;
    if (trace) {
        std::fprintf(stderr,
                     "cuda_fast_restart broken_iterations=%d cold_restarts=%d "
                     "restart_iterations=%s lsmr=%d lsmr_iterations=%d max_iter=%d "
                     "iterations=%d outcome=%s seconds=%.6f\n",
                     broken_iterations, restarts,
                     attempt_iterations.empty() ? "-" : attempt_iterations.c_str(),
                     lsmr_used ? 1 : 0, lsmr_iterations, options.max_iter, spent, outcome,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                         .count());
    }
    return ok;
}

// Largest absolute group mean of v over the ordinary FEs, with the FEs' own
// group weight sums (unit weights: the group sizes). Host pass, long double
// sums; infinity on a non-finite value. The strict condition of
// restart_strict_after_breakdown.
static double strict_max_abs_group_mean(const Eigen::Ref<const Eigen::VectorXd>& v,
                                        const std::vector<GpuFeInput>& fe_inputs,
                                        const Eigen::VectorXd* weights) {
    double out = 0.0;
    const Eigen::Index n = v.size();
    for (const GpuFeInput& fe : fe_inputs) {
        if (fe.is_slope || fe.group_ids == nullptr || fe.weight_sums == nullptr ||
            fe.num_groups <= 0) {
            return std::numeric_limits<double>::infinity();
        }
        std::vector<long double> sums(static_cast<std::size_t>(fe.num_groups), 0.0L);
        for (Eigen::Index i = 0; i < n; ++i) {
            const long double w =
                weights != nullptr ? static_cast<long double>((*weights)[i]) : 1.0L;
            sums[static_cast<std::size_t>(fe.group_ids[i])] +=
                w * static_cast<long double>(v[i]);
        }
        for (int g = 0; g < fe.num_groups; ++g) {
            const double denominator = fe.weight_sums[g];
            if (!(denominator > 0.0)) continue;
            const double mean = static_cast<double>(
                sums[static_cast<std::size_t>(g)] / static_cast<long double>(denominator));
            if (!finite_double_bits(mean)) return std::numeric_limits<double>::infinity();
            out = std::max(out, std::fabs(mean));
        }
    }
    return out;
}

// P3-fix (3'), hypothesis P3-S and its scope addendum (RELATORIO_P3FIX_20261109.md
// sections 11.2 and 11.3): the restart of a broken CG stage of a stage with the strict
// provenance (HdfeOptions::strict_provenance): the native CUDA attempt of a
// strict-residual fit, which configure_native_cuda_repair runs as xhdfe-fast with the
// update criterion, or that fit's strict repair (run_strict_cuda_repair), whose
// path-like two-FE CG gate ignores the mode. Up to 2.30.1 such a stage restarted with
// the cold MAP under the Comparable stopping rule (norm of change), whose state,
// outside the 2.28.1 envelope, the strict certificate accepted (sections 7.4 and 10).
// The fit's recovery is a finite sequence: one device LSMR restart (here), then at
// most one strict repair (the regressor), never the cold MAP. Here, within what the
// broken stage left of max_iter:
// 1. when the sequence has not used it (!strict_lsmr_consumed), a cold device LSMR
//    from the original y and X (never from the CG iterate), the core of the
//    Comparable restart without its host hand-over (no probe), under the existing
//    ordinary strict policy: the stage's options in the strict-residual mode, so its
//    Krylov target is min(tol, 1e-11) under the caller's ordinary parity floor;
// 2. the strict condition - the largest absolute group mean of y~ over the FEs at
//    most polish_tol = min(tol, 1e-10), that of the existing strict polish - judged on
//    the LSMR state itself: when it holds, that state is delivered with no sweep;
//    otherwise the existing strict polish (the MAP in the strict-residual mode at
//    polish_tol, no CG) runs on the LSMR state, and its own exit (converged only when
//    the condition is met) decides.
// An approved state returns converged; the caller certifies it. Otherwise (LSMR
// stopped, condition not met, budget exhausted, or the LSMR already used: a second
// breakdown ends the sequence) the stage returns "not converged" (status 3) with
// cuda_strict_lsmr_restart set, which the regressor reads to give its one strict
// repair the state of the sequence; a CUDA failure returns false (status 4).
// `iterations` counts the broken stage, the IT sweeps before it that it did not
// count, the LSMR and the polish, never truncated, never more than max_iter. One
// trace line (XHDFE_GI_LSMR_TRACE), cuda_strict_restart. Test hook, tests only:
// XHDFE_TEST_CUDA_STRICT_LSMR_REJECT=1 rejects an approved LSMR state (flow control
// of the sequence, not a frequency).
static bool restart_strict_after_breakdown(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& fe_inputs,
    const Eigen::VectorXd* weights,
    const std::vector<std::size_t>& sweep_order,
    const HdfeOptions& options,
    AbsorptionMethod method,
    double solver_fe_tolerance,
    AbsorptionResult& result) {
    const bool trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
    const auto start = std::chrono::steady_clock::now();
    const int cg_iterations = std::max(0, result.iterations);
    const int uncounted_it = std::max(0, result.cuda_cg_uncounted_it_iterations);
    int spent = std::min(options.max_iter, cg_iterations + uncounted_it);
    const bool lsmr_available = !options.strict_lsmr_consumed;
    const double polish_tol =
        options.tol > 0.0 ? std::min(options.tol, 1.0e-10) : options.tol;
    int lsmr_iterations = 0;
    int polish_iterations = 0;
    bool lsmr_converged = false;
    bool polished = false;
    bool test_rejected = false;
    double krylov_target = 0.0;
    double lsmr_max_mean = std::numeric_limits<double>::quiet_NaN();
    double final_max_mean_all = std::numeric_limits<double>::quiet_NaN();
    double lsmr_seconds = 0.0;
    double polish_seconds = 0.0;
    bool ok = true;
    const char* outcome = lsmr_available ? "budget-exhausted" : "lsmr-already-consumed";
    bool converged = false;
    if (lsmr_available && spent < options.max_iter) {
        AbsorptionResult delivered;
        HdfeOptions lsmr_options = options;
        lsmr_options.max_iter = options.max_iter - spent;
        lsmr_options.tolerance_mode = ToleranceMode::StrictResidual;
        lsmr_options.convergence_criterion = ConvergenceCriterion::Auto;
        lsmr_options.cuda_tail_route = nullptr;
        const auto lsmr_start = std::chrono::steady_clock::now();
        ok = absorb_ordinary_lsmr_cuda(y, X, fe_inputs, weights, lsmr_options, delivered);
        lsmr_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - lsmr_start).count();
        lsmr_iterations = std::max(0, delivered.iterations);
        lsmr_converged = ok && delivered.converged;
        krylov_target = delivered.krylov_internal_tolerance;
        spent = std::min(options.max_iter, spent + lsmr_iterations);
        if (!ok) {
            outcome = "lsmr-failed";
        } else if (!lsmr_converged) {
            outcome = "lsmr-stopped";
        } else {
            lsmr_max_mean = strict_max_abs_group_mean(delivered.y_tilde, fe_inputs, weights);
            if (lsmr_max_mean <= polish_tol) {
                outcome = "lsmr-strict";
                converged = true;
            } else if (spent >= options.max_iter) {
                outcome = "polish-budget-exhausted";
            } else {
                HdfeOptions polish_options = options;
                polish_options.max_iter = options.max_iter - spent;
                polish_options.tol = polish_tol;
                polish_options.tolerance_mode = ToleranceMode::StrictResidual;
                polish_options.convergence_criterion = ConvergenceCriterion::Auto;
                polish_options.cuda_tail_route = nullptr;
                AbsorptionResult polish;
                const auto polish_start = std::chrono::steady_clock::now();
                ok = absorb_fixed_effects_cuda_impl(
                    delivered.y_tilde, delivered.X_tilde, fe_inputs, weights, sweep_order,
                    polish_options, method, polish, solver_fe_tolerance, false);
                polish_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - polish_start).count();
                polished = true;
                polish_iterations = std::max(0, polish.iterations);
                spent = std::min(options.max_iter, spent + polish_iterations);
                if (!ok) {
                    outcome = "polish-failed";
                } else {
                    converged = polish.converged;
                    outcome = converged ? "lsmr-polished" : "polish-stopped";
                    // The LSMR result keeps its Krylov diagnostics; the state
                    // delivered is the polished one.
                    delivered.y_tilde = std::move(polish.y_tilde);
                    delivered.X_tilde = std::move(polish.X_tilde);
                }
            }
        }
        if (ok && converged && cuda_strict_lsmr_test_reject()) {
            converged = false;
            test_rejected = true;
            outcome = "lsmr-rejected-by-test-hook";
        }
        if (trace && ok && delivered.y_tilde.size() == y.size()) {
            final_max_mean_all = strict_max_abs_group_mean(delivered.y_tilde, fe_inputs, weights);
            for (Eigen::Index c = 0; c < delivered.X_tilde.cols(); ++c) {
                final_max_mean_all = std::max(
                    final_max_mean_all,
                    strict_max_abs_group_mean(delivered.X_tilde.col(c), fe_inputs, weights));
            }
        }
        result = std::move(delivered);
    }
    result.converged = ok && converged;
    result.precision_certified = false;
    result.iterations = spent;
    result.cuda_strict_lsmr_restart = true;
    if (trace) {
        std::fprintf(stderr,
                     "cuda_strict_restart lsmr_available=%d cg_iterations=%d uncounted_it=%d "
                     "lsmr_iterations=%d lsmr_converged=%d krylov_target=%.3e "
                     "lsmr_max_mean=%.17g polish_tol=%.3e polished=%d polish_iterations=%d "
                     "final_max_mean_all_columns=%.17g test_rejected=%d max_iter=%d "
                     "iterations=%d outcome=%s seconds=%.6f lsmr_seconds=%.6f "
                     "polish_seconds=%.6f\n",
                     lsmr_available ? 1 : 0, cg_iterations, uncounted_it, lsmr_iterations,
                     lsmr_converged ? 1 : 0, krylov_target, lsmr_max_mean, polish_tol,
                     polished ? 1 : 0, polish_iterations, final_max_mean_all,
                     test_rejected ? 1 : 0, options.max_iter, spent, outcome,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                         .count(),
                     lsmr_seconds, polish_seconds);
    }
    return ok;
}

bool absorb_fixed_effects_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<GpuFeInput>& fe_inputs,
                               const Eigen::VectorXd* weights,
                               const std::vector<std::size_t>& sweep_order,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               AbsorptionResult& result,
                               double solver_fe_tolerance) {
    const bool ok = absorb_fixed_effects_cuda_impl(
        y, X, fe_inputs, weights, sweep_order, options, method, result,
        solver_fe_tolerance, true);
    if (!ok || !result.cg_restart_required) return ok;

    const bool ordinary_comparable =
        options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
        !options.retain_fixed_effects &&
        std::none_of(fe_inputs.begin(), fe_inputs.end(),
                     [](const GpuFeInput& fe) { return fe.is_slope; });
    // F2-B: the three-FE family of the F2-A tail (unweighted, regressors
    // present) restarts with the probe unless XHDFE_CUDA_TAIL_HANDOVER=0.
    const bool three_fe_restart =
        cuda_three_fe_family(options, fe_inputs, weights, static_cast<int>(X.cols()));
    if (three_fe_restart) {
        const CudaTailMode restart_mode = cuda_tail_handover_mode();
        if (restart_mode != CudaTailMode::Off) {
            return restart_three_fe_after_cg(y, X, fe_inputs, options, restart_mode, result);
        }
    }
    // P3-fix (B): a broken xhdfe-fast stage in the shape of
    // cuda_fast_cg_restart_applies restarts cold (restart_fast_cg_after_breakdown),
    // never with the cold MAP below.
    if (cuda_fast_cg_restart_applies(options, fe_inputs, weights,
                                     static_cast<int>(y.size()), result)) {
        return restart_fast_cg_after_breakdown(y, X, fe_inputs, weights, sweep_order, options,
                                               method, solver_fe_tolerance, result);
    }
    // P3-fix (3'), strict scope: a broken CG stage of a stage with the strict
    // provenance (the native CUDA attempt of a strict-residual fit or its strict
    // repair; HdfeOptions::strict_provenance, set by the regressor) restarts cold
    // with the device LSMR and the strict check unless the fit already used that
    // restart, and is refused otherwise (restart_strict_after_breakdown); never
    // the cold MAP below. A weighted native attempt runs under the Comparable rule
    // and keeps the device LSMR restart below.
    if (options.strict_provenance && !ordinary_comparable) {
        return restart_strict_after_breakdown(y, X, fe_inputs, weights, sweep_order, options,
                                              method, solver_fe_tolerance, result);
    }

    // Restart on the GPU from the original RHS: a broken CG recurrence can
    // lose null-space components which a warm repair cannot recover.
    // (2.30.1, unchanged; for the three-FE family also under the switch at 0.)
    const int cg_iterations = result.iterations;
    if (cg_iterations >= options.max_iter) {
        result.converged = false;
        result.precision_certified = false;
        return ok;
    }
    HdfeOptions retry_options = options;
    retry_options.max_iter -= cg_iterations;
    if (ordinary_comparable) {
        // Cold MAP still left CUDA04's weak graph modes unfinished. Use the
        // device counterpart of the CPU's Krylov restart, at the existing
        // ordinary 1e-11 floor and strictly within the remaining budget.
        CudaRestartLedger ledger;
        const bool trace = three_fe_restart && std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
        if (trace) {
            ledger.mode = "off";
            ledger.it_iterations = std::max(0, result.cuda_cg_uncounted_it_iterations);
            ledger.it_reason = result.cuda_it_handoff_reason;
            ledger.it_seconds = result.cuda_it_seconds;
            ledger.cg_iterations = cg_iterations;
            ledger.cg_seconds = result.cuda_cg_seconds;
            ledger.counted_before = cg_iterations;
            ledger.max_iter = options.max_iter;
            ledger.budget = retry_options.max_iter;
            ledger.threads = options.num_threads;
        }
        const auto start = std::chrono::steady_clock::now();
        AbsorptionResult repaired;
        const bool repaired_ok = absorb_ordinary_lsmr_cuda(
            y, X, fe_inputs, weights, retry_options, repaired);
        if (trace) {
            ledger.device_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            ledger.device_iterations = std::max(0, repaired.iterations);
        }
        repaired.iterations += cg_iterations;
        result = std::move(repaired);
        if (trace) {
            ledger.iterations = result.iterations;
            ledger.print(nullptr);
        }
        return repaired_ok;
    }
    // The accelerated cold MAP exit uses the nominal tolerance. Honour the
    // original Comparable floor here too. An xhdfe-fast stage reaches this cold
    // MAP only outside the P3-fix shape and outside a stage with the strict
    // provenance (both above), and then still under the Comparable stopping rule
    // (RELATORIO_P3_20261108.md, section 7a).
    retry_options.tol = effective_absorption_tolerance(options);
    retry_options.tolerance_mode = ToleranceMode::ReghdfeComparable;
    retry_options.convergence_criterion = ConvergenceCriterion::NormChange;
    // Trace only (diagnostic build cand_p3_trace_20261110, P4): with
    // XHDFE_SLOPE_STAGE_TRACE=1 this cold MAP restart prints the broken stage
    // it replaces and its own options, budget and outcome. Off (the default),
    // nothing is timed or printed.
    static const bool cold_map_trace = [] {
        const char* e = std::getenv("XHDFE_SLOPE_STAGE_TRACE");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    const int trace_uncounted_it = result.cuda_cg_uncounted_it_iterations;
    const double trace_cg_seconds = result.cuda_cg_seconds;
    const auto cold_start = cold_map_trace ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    AbsorptionResult repaired;
    const bool repaired_ok = absorb_fixed_effects_cuda_impl(
        y, X, fe_inputs, weights, sweep_order, retry_options, method, repaired,
        solver_fe_tolerance, false);
    if (cold_map_trace) {
        const bool slopes = std::any_of(fe_inputs.begin(), fe_inputs.end(),
                                        [](const GpuFeInput& fe) { return fe.is_slope; });
        std::fprintf(stderr,
                     "cuda_cold_map_restart_trace slopes=%d weights=%d regressors=%d "
                     "stage_mode=%d stage_tol=%.6e retry_tol=%.6e retry_mode=comparable "
                     "retry_criterion=norm-change cg_iterations=%d uncounted_it=%d "
                     "cg_seconds=%.6f budget=%d map_iterations=%d converged=%d ok=%d "
                     "seconds=%.6f\n",
                     slopes ? 1 : 0, weights != nullptr ? 1 : 0, static_cast<int>(X.cols()),
                     static_cast<int>(options.tolerance_mode), options.tol, retry_options.tol,
                     cg_iterations, trace_uncounted_it, trace_cg_seconds, retry_options.max_iter,
                     repaired.iterations, repaired.converged ? 1 : 0, repaired_ok ? 1 : 0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - cold_start)
                         .count());
    }
    repaired.iterations += cg_iterations;
    result = std::move(repaired);
    return repaired_ok;
}

bool valid_group_individual_lsmr_structure(
    const GroupIndividualStructure& gi,
    int groups,
    const Eigen::VectorXd* weights) {
    if (groups <= 0 || gi.num_groups != groups || gi.num_individuals <= 0 ||
        gi.group_ptr.size() != static_cast<std::size_t>(groups) + 1U ||
        gi.individual_ptr.size() !=
            static_cast<std::size_t>(gi.num_individuals) + 1U ||
        gi.group_scale.size() != static_cast<std::size_t>(groups) ||
        gi.group_individual.size() != gi.individual_group.size() ||
        (weights && weights->size() != groups)) {
        return false;
    }

    auto valid_ptr = [](const std::vector<int>& ptr,
                        std::size_t rows,
                        std::size_t edges) {
        if (ptr.size() != rows + 1U || ptr.empty() || ptr.front() != 0 ||
            ptr.back() < 0 ||
            static_cast<std::size_t>(ptr.back()) != edges) {
            return false;
        }
        int previous = 0;
        for (const int value : ptr) {
            if (value < previous || value < 0 ||
                static_cast<std::size_t>(value) > edges) {
                return false;
            }
            previous = value;
        }
        return true;
    };
    if (!valid_ptr(gi.group_ptr, static_cast<std::size_t>(groups),
                   gi.group_individual.size()) ||
        !valid_ptr(gi.individual_ptr,
                   static_cast<std::size_t>(gi.num_individuals),
                   gi.individual_group.size())) {
        return false;
    }
    for (const int individual : gi.group_individual) {
        if (individual < 0 || individual >= gi.num_individuals) {
            return false;
        }
    }
    for (const int group : gi.individual_group) {
        if (group < 0 || group >= groups) {
            return false;
        }
    }
    for (const double scale : gi.group_scale) {
        if (!finite_double_bits(scale)) {
            return false;
        }
    }
    if (weights) {
        for (Eigen::Index row = 0; row < weights->size(); ++row) {
            const double weight = (*weights)[row];
            if (!finite_double_bits(weight) || weight < 0.0) {
                return false;
            }
        }
    }
    return true;
}

static bool absorb_group_individual_lsmr_cuda_impl(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& standard_fe_inputs,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionResult& result,
    bool ordinary_projection = false,
    CudaTailProbe* probe = nullptr) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    const int dims = static_cast<int>(standard_fe_inputs.size());
    const int individuals = gi.num_individuals;
    const bool unit_weights = weights == nullptr;
    const bool trace = std::getenv("XHDFE_GI_LSMR_TRACE") != nullptr;
    // The F2-A probe concerns the ordinary three-FE completion only.
    if (!ordinary_projection) probe = nullptr;
    // Ordinary FEs reuse only the device operator and primary LSMR solve.
    // Group refinement and its forward guard do not define ordinary work.
    const bool capture_projection = !ordinary_projection;
    const double effective_tol = effective_absorption_tolerance(options);
    const bool ordinary_parity = ordinary_projection &&
        options.ordinary_krylov_parity_floor && effective_tol > 0.0 &&
        (options.tolerance_mode == ToleranceMode::ReghdfeComparable ||
         options.tolerance_mode == ToleranceMode::StrictResidual);
    const double solve_tolerance = ordinary_projection
        ? (ordinary_parity ? std::min(effective_tol, 1.0e-11) : effective_tol)
        : group_individual_absorption_tolerance(options);
    if (ordinary_projection) result.krylov_internal_tolerance = solve_tolerance;
    const auto solve_start = std::chrono::steady_clock::now();
    if (!valid_group_individual_lsmr_structure(gi, n, weights)) {
        return false;
    }

    try {
        GiLsmrLocalWorkspace lsmr;
        lsmr.standard_group_ptr.resize(static_cast<std::size_t>(dims));
        lsmr.standard_observation_index.resize(
            static_cast<std::size_t>(dims));
        std::vector<int> offsets(static_cast<std::size_t>(dims) + 1U, 0);
        for (int dim = 0; dim < dims; ++dim) {
            const GpuFeInput& fe = standard_fe_inputs[static_cast<std::size_t>(dim)];
            if (fe.group_ids == nullptr || fe.weight_sums == nullptr ||
                fe.num_groups <= 0 ||
                offsets[static_cast<std::size_t>(dim)] >
                    std::numeric_limits<int>::max() - fe.num_groups) {
                return false;
            }
            offsets[static_cast<std::size_t>(dim + 1)] =
                offsets[static_cast<std::size_t>(dim)] + fe.num_groups;
        }
        const int individual_offset = offsets.back();
        if (individuals <= 0 ||
            individual_offset > std::numeric_limits<int>::max() - individuals) {
            return false;
        }
        const int total_fe = individual_offset + individuals;

        CudaWorkspace& workspace = cuda_workspace;
        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        staged_memcpy_h2d(d_y.data(), y.data(), sizeof(double) * n,
                          workspace.h_stage, "cudaMemcpy GI LSMR y failed");
        DeviceBuffer<double>& d_x = workspace.d_x;
        if (cols > 0) {
            d_x.allocate(static_cast<std::size_t>(ld) * cols);
            staged_memcpy_h2d(d_x.data(), X.data(),
                              sizeof(double) * static_cast<std::size_t>(ld) * cols,
                              workspace.h_stage, "cudaMemcpy GI LSMR X failed");
        } else {
            d_x.reset();
        }
        DeviceBuffer<double>& d_weights = workspace.d_weights;
        if (!unit_weights) {
            d_weights.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(d_weights.data(), weights->data(),
                              sizeof(double) * n, workspace.h_stage,
                              "cudaMemcpy GI LSMR weights failed");
        } else {
            d_weights.reset();
        }

        std::vector<CudaFeDevice>& fe_dev = workspace.fe_dev;
        if (fe_dev.size() != standard_fe_inputs.size()) {
            fe_dev.resize(standard_fe_inputs.size());
        }
        std::vector<const int*> gid_ptrs(static_cast<std::size_t>(dims));
        result.fe_levels.clear();
        result.fe_levels.reserve(static_cast<std::size_t>(dims) + 1U);
        for (int dim = 0; dim < dims; ++dim) {
            const GpuFeInput& fe = standard_fe_inputs[static_cast<std::size_t>(dim)];
            CudaFeDevice& dev = fe_dev[static_cast<std::size_t>(dim)];
            dev.num_groups = fe.num_groups;
            dev.num_levels_present = fe.num_levels_present;
            dev.gid.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(dev.gid.data(), fe.group_ids, sizeof(int) * n,
                              workspace.h_stage,
                              "cudaMemcpy GI LSMR group ids failed");

            std::vector<int> group_ptr(
                static_cast<std::size_t>(fe.num_groups) + 1U, 0);
            for (int row = 0; row < n; ++row) {
                const int group = fe.group_ids[row];
                if (group < 0 || group >= fe.num_groups) {
                    return false;
                }
                ++group_ptr[static_cast<std::size_t>(group + 1)];
            }
            for (int group = 0; group < fe.num_groups; ++group) {
                group_ptr[static_cast<std::size_t>(group + 1)] +=
                    group_ptr[static_cast<std::size_t>(group)];
            }
            std::vector<int> observation_index(static_cast<std::size_t>(n));
            std::vector<int> cursor = group_ptr;
            for (int row = 0; row < n; ++row) {
                const int group = fe.group_ids[row];
                observation_index[static_cast<std::size_t>(
                    cursor[static_cast<std::size_t>(group)]++)] = row;
            }
            DeviceBuffer<int>& standard_group_ptr =
                lsmr.standard_group_ptr[static_cast<std::size_t>(dim)];
            standard_group_ptr.allocate(group_ptr.size());
            staged_memcpy_h2d(standard_group_ptr.data(), group_ptr.data(),
                              sizeof(int) * group_ptr.size(), workspace.h_stage,
                              "cudaMemcpy GI LSMR FE ptr failed");
            DeviceBuffer<int>& standard_observation_index =
                lsmr.standard_observation_index[static_cast<std::size_t>(dim)];
            standard_observation_index.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(standard_observation_index.data(),
                              observation_index.data(), sizeof(int) * n,
                              workspace.h_stage,
                              "cudaMemcpy GI LSMR FE observations failed");
            dev.weight_sums.allocate(static_cast<std::size_t>(fe.num_groups));
            staged_memcpy_h2d(dev.weight_sums.data(), fe.weight_sums,
                              sizeof(double) * fe.num_groups,
                              workspace.h_stage,
                              "cudaMemcpy GI LSMR weight sums failed");
            gid_ptrs[static_cast<std::size_t>(dim)] = dev.gid.data();
            result.fe_levels.push_back(fe.num_levels_present);
        }
        result.fe_levels.push_back(individuals);

        lsmr.gid_ptrs.allocate(static_cast<std::size_t>(dims));
        if (dims > 0) {
            cuda_check(cudaMemcpy(lsmr.gid_ptrs.data(),
                                  gid_ptrs.data(),
                                  sizeof(const int*) * static_cast<std::size_t>(dims),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy GI LSMR gid pointers failed");
        }
        lsmr.offsets.allocate(offsets.size());
        cuda_check(cudaMemcpy(lsmr.offsets.data(), offsets.data(),
                              sizeof(int) * offsets.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy GI LSMR offsets failed");

        DeviceBuffer<int>& d_group_ptr = workspace.gi_group_ptr;
        d_group_ptr.allocate(static_cast<std::size_t>(n) + 1U);
        staged_memcpy_h2d(d_group_ptr.data(), gi.group_ptr.data(),
                          sizeof(int) * (static_cast<std::size_t>(n) + 1U),
                          workspace.h_stage,
                          "cudaMemcpy GI LSMR group ptr failed");
        DeviceBuffer<int>& d_group_individual = workspace.gi_group_individual;
        d_group_individual.allocate(gi.group_individual.size());
        staged_memcpy_h2d(d_group_individual.data(), gi.group_individual.data(),
                          sizeof(int) * gi.group_individual.size(),
                          workspace.h_stage,
                          "cudaMemcpy GI LSMR group individuals failed");
        DeviceBuffer<int>& d_individual_ptr = workspace.gi_individual_ptr;
        d_individual_ptr.allocate(static_cast<std::size_t>(individuals) + 1U);
        staged_memcpy_h2d(d_individual_ptr.data(), gi.individual_ptr.data(),
                          sizeof(int) *
                              (static_cast<std::size_t>(individuals) + 1U),
                          workspace.h_stage,
                          "cudaMemcpy GI LSMR individual ptr failed");
        DeviceBuffer<int>& d_individual_group = workspace.gi_individual_group;
        d_individual_group.allocate(gi.individual_group.size());
        staged_memcpy_h2d(d_individual_group.data(), gi.individual_group.data(),
                          sizeof(int) * gi.individual_group.size(),
                          workspace.h_stage,
                          "cudaMemcpy GI LSMR individual groups failed");
        DeviceBuffer<double>& d_group_scale = workspace.gi_group_scale;
        d_group_scale.allocate(static_cast<std::size_t>(n));
        staged_memcpy_h2d(d_group_scale.data(), gi.group_scale.data(),
                          sizeof(double) * n, workspace.h_stage,
                          "cudaMemcpy GI LSMR group scale failed");

        const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
        const int blocks_i = (individuals + kBlockSize - 1) / kBlockSize;
        const int blocks_fe = (total_fe + kBlockSize - 1) / kBlockSize;
        DeviceBuffer<double>& d_denom = workspace.gi_denom;
        d_denom.allocate(static_cast<std::size_t>(individuals));
        if (ordinary_projection && unit_weights) {
            ordinary_incidence_diagonal_kernel<<<blocks_i, kBlockSize>>>(
                d_individual_ptr.data(), individuals, d_denom.data());
        } else {
            gi_denom_by_individual_kernel<<<blocks_i, kBlockSize>>>(
                d_individual_ptr.data(), d_individual_group.data(),
                d_group_scale.data(), unit_weights ? nullptr : d_weights.data(),
                unit_weights, n, individuals, d_denom.data());
        }
        cuda_check(cudaGetLastError(),
                   "GI LSMR individual diagonal kernel launch failed");

        DeviceBuffer<double>& d_column_scale = lsmr.column_scale;
        d_column_scale.allocate(static_cast<std::size_t>(total_fe));
        for (int dim = 0; dim < dims; ++dim) {
            const CudaFeDevice& dev = fe_dev[static_cast<std::size_t>(dim)];
            const int blocks = (dev.num_groups + kBlockSize - 1) / kBlockSize;
            gi_lsmr_inverse_sqrt_kernel<<<blocks, kBlockSize>>>(
                dev.weight_sums.data(), dev.num_groups,
                offsets[static_cast<std::size_t>(dim)], d_column_scale.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR standard column-scale kernel launch failed");
        }
        gi_lsmr_inverse_sqrt_kernel<<<blocks_i, kBlockSize>>>(
            d_denom.data(), individuals, individual_offset,
            d_column_scale.data());
        cuda_check(cudaGetLastError(),
                   "GI LSMR individual column-scale kernel launch failed");

        lsmr.b.allocate(static_cast<std::size_t>(n));
        lsmr.u.allocate(static_cast<std::size_t>(n));
        lsmr.av.allocate(static_cast<std::size_t>(n));
        lsmr.v.allocate(static_cast<std::size_t>(total_fe));
        lsmr.h.allocate(static_cast<std::size_t>(total_fe));
        lsmr.hbar.allocate(static_cast<std::size_t>(total_fe));
        lsmr.solution.allocate(static_cast<std::size_t>(total_fe));
        lsmr.atu.allocate(static_cast<std::size_t>(total_fe));
        lsmr.scaled.allocate(static_cast<std::size_t>(total_fe));
        lsmr.residual_y.allocate(static_cast<std::size_t>(n));
        if (cols > 0) {
            lsmr.residual_x.allocate(
                static_cast<std::size_t>(ld) * cols);
        }
        constexpr int kNormBlocks = 256;
        lsmr.norm_partials.allocate(kNormBlocks);
        lsmr.norm_scalar.allocate(1);

        auto device_norm = [&](const double* values, int count) {
            const int blocks = std::max(
                1, std::min(kNormBlocks,
                            (count + kBlockSize - 1) / kBlockSize));
            gi_lsmr_sumsq_partials_kernel<kBlockSize>
                <<<blocks, kBlockSize>>>(
                    values, count, lsmr.norm_partials.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR norm partial kernel launch failed");
            gi_lsmr_sumsq_finish_kernel<kBlockSize><<<1, kBlockSize>>>(
                lsmr.norm_partials.data(), blocks,
                lsmr.norm_scalar.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR norm finish kernel launch failed");
            double sumsq = 0.0;
            cuda_check(cudaMemcpy(&sumsq,
                                  lsmr.norm_scalar.data(),
                                  sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy GI LSMR norm failed");
            if (!finite_double_bits(sumsq) || sumsq < 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::sqrt(sumsq);
        };

        auto apply_a = [&](const double* coefficients,
                           double* output,
                           bool weighted_output) {
            gi_lsmr_scale_coefficients_kernel<<<blocks_fe, kBlockSize>>>(
                coefficients, d_column_scale.data(), total_fe,
                lsmr.scaled.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR coefficient-scale kernel launch failed");
            gi_lsmr_apply_a_kernel<<<blocks_n, kBlockSize>>>(
                lsmr.scaled.data(),
                dims > 0 ? lsmr.gid_ptrs.data() : nullptr,
                lsmr.offsets.data(), dims,
                d_group_ptr.data(), d_group_individual.data(),
                d_group_scale.data(), individual_offset,
                unit_weights ? nullptr : d_weights.data(), unit_weights,
                weighted_output, n, output);
            cuda_check(cudaGetLastError(), "GI LSMR A kernel launch failed");
        };

        auto apply_at = [&](const double* input, double* output) {
            auto cooperative = [&](const int* ptr, const int* observation,
                                   int levels, int offset) {
                if (!ordinary_projection || !unit_weights) return false;
                const std::int64_t rows = n;
                if (rows >= static_cast<std::int64_t>(levels) * kBlockSize) {
                    ordinary_at_block_kernel<kBlockSize><<<levels, kBlockSize>>>(
                        input, ptr, observation, d_column_scale.data(),
                        offset, levels, output);
                    return true;
                }
                constexpr int warp_size = 32;
                if (rows >= static_cast<std::int64_t>(levels) * warp_size) {
                    constexpr int warps = kBlockSize / warp_size;
                    const int blocks = (levels + warps - 1) / warps;
                    ordinary_at_warp_kernel<kBlockSize><<<blocks, kBlockSize>>>(
                        input, ptr, observation, d_column_scale.data(),
                        offset, levels, output);
                    return true;
                }
                return false;
            };
            for (int dim = 0; dim < dims; ++dim) {
                const CudaFeDevice& dev = fe_dev[static_cast<std::size_t>(dim)];
                const int blocks =
                    (dev.num_groups + kBlockSize - 1) / kBlockSize;
                if (!cooperative(
                        lsmr.standard_group_ptr[static_cast<std::size_t>(dim)].data(),
                        lsmr.standard_observation_index[static_cast<std::size_t>(dim)].data(),
                        dev.num_groups, offsets[static_cast<std::size_t>(dim)])) {
                    gi_lsmr_apply_at_standard_kernel<<<blocks, kBlockSize>>>(
                        input,
                        lsmr.standard_group_ptr[static_cast<std::size_t>(dim)].data(),
                        lsmr.standard_observation_index[
                            static_cast<std::size_t>(dim)].data(),
                        unit_weights ? nullptr : d_weights.data(), unit_weights,
                        d_column_scale.data(),
                        offsets[static_cast<std::size_t>(dim)], dev.num_groups,
                        output);
                }
                cuda_check(cudaGetLastError(),
                           "GI LSMR standard A-transpose kernel launch failed");
            }
            if (!cooperative(d_individual_ptr.data(), d_individual_group.data(),
                             individuals, individual_offset)) {
                gi_lsmr_apply_at_individual_kernel<<<blocks_i, kBlockSize>>>(
                    input, d_individual_ptr.data(), d_individual_group.data(),
                    d_group_scale.data(),
                    unit_weights ? nullptr : d_weights.data(), unit_weights,
                    d_column_scale.data(), individual_offset, individuals, output);
            }
            cuda_check(cudaGetLastError(),
                       "GI LSMR individual A-transpose kernel launch failed");
        };

        struct Rotation {
            double c = 1.0;
            double s = 0.0;
            double r = 0.0;
        };
        auto sym_ortho = [](double a, double b) {
            Rotation rotation;
            if (b == 0.0) {
                rotation.c = a < 0.0 ? -1.0 : 1.0;
                rotation.r = std::abs(a);
            } else if (a == 0.0) {
                rotation.c = 0.0;
                rotation.s = b < 0.0 ? -1.0 : 1.0;
                rotation.r = std::abs(b);
            } else if (std::abs(b) > std::abs(a)) {
                const double tau = a / b;
                rotation.s = (b < 0.0 ? -1.0 : 1.0) /
                             std::sqrt(1.0 + tau * tau);
                rotation.c = rotation.s * tau;
                rotation.r = b / rotation.s;
            } else {
                const double tau = b / a;
                rotation.c = (a < 0.0 ? -1.0 : 1.0) /
                             std::sqrt(1.0 + tau * tau);
                rotation.s = rotation.c * tau;
                rotation.r = a / rotation.c;
            }
            return rotation;
        };

        auto solve_one = [&](const double* raw,
                             int rhs_index,
                             int& iterations,
                             bool& converged,
                             int iteration_limit) {
            DeviceBuffer<double>& d_b = lsmr.b;
            DeviceBuffer<double>& d_u = lsmr.u;
            DeviceBuffer<double>& d_v = lsmr.v;
            DeviceBuffer<double>& d_h = lsmr.h;
            DeviceBuffer<double>& d_hbar = lsmr.hbar;
            DeviceBuffer<double>& d_solution = lsmr.solution;
            DeviceBuffer<double>& d_av = lsmr.av;
            DeviceBuffer<double>& d_atu = lsmr.atu;

            gi_lsmr_build_rhs_kernel<<<blocks_n, kBlockSize>>>(
                raw, unit_weights ? nullptr : d_weights.data(), unit_weights,
                n, d_b.data());
            cuda_check(cudaGetLastError(), "GI LSMR RHS kernel launch failed");
            cuda_check(cudaMemset(d_solution.data(), 0,
                                  sizeof(double) * total_fe),
                       "cudaMemset GI LSMR solution failed");
            cuda_check(cudaMemset(d_hbar.data(), 0,
                                  sizeof(double) * total_fe),
                       "cudaMemset GI LSMR hbar failed");

            const double norm_b = device_norm(d_b.data(), n);
            if (!finite_double_bits(norm_b)) {
                iterations = 0;
                converged = false;
                return;
            }
            if (norm_b == 0.0) {
                iterations = 0;
                converged = true;
                return;
            }
            gi_lsmr_scale_copy_kernel<<<blocks_n, kBlockSize>>>(
                d_b.data(), 1.0 / norm_b, n, d_u.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR initial u kernel launch failed");
            apply_at(d_u.data(), d_v.data());
            double alpha = device_norm(d_v.data(), total_fe);
            if (!finite_double_bits(alpha)) {
                iterations = 0;
                converged = false;
                return;
            }
            if (alpha == 0.0) {
                iterations = 0;
                converged = true;
                return;
            }
            gi_lsmr_scale_copy_kernel<<<blocks_fe, kBlockSize>>>(
                d_v.data(), 1.0 / alpha, total_fe, d_v.data());
            cuda_check(cudaGetLastError(),
                       "GI LSMR initial v kernel launch failed");
            cuda_check(cudaMemcpy(d_h.data(), d_v.data(),
                                  sizeof(double) * total_fe,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy GI LSMR initial h failed");

            double beta = norm_b;
            double zetabar = alpha * beta;
            double alphabar = alpha;
            double rho = 1.0;
            double rhobar = 1.0;
            double cbar = 1.0;
            double sbar = 0.0;
            double betadd = beta;
            double betad = 0.0;
            double rhodold = 1.0;
            double tautildeold = 0.0;
            double thetatilde = 0.0;
            double zeta = 0.0;
            double dnorm = 0.0;
            double norm_a_sq = alpha * alpha;
            double max_rbar = 0.0;
            double min_rbar = 1.0e100;
            double norm_a = std::sqrt(norm_a_sq);
            double condition = 1.0;
            double norm_r = beta;
            const double tolerance = solve_tolerance;
            const double condition_limit = 1.0e12;
            const double forward_target = group_forward_tolerance(options);
            double final_test1 = std::numeric_limits<double>::infinity();
            double final_test2 = std::numeric_limits<double>::infinity();
            const char* stop_reason = "maxiter";

            converged = false;
            int k = 0;
            for (; k < iteration_limit; ++k) {
                apply_a(d_v.data(), d_av.data(), true);
                gi_lsmr_affine_subtract_kernel<<<blocks_n, kBlockSize>>>(
                    d_av.data(), d_u.data(), alpha, n, d_u.data());
                cuda_check(cudaGetLastError(),
                           "GI LSMR u recurrence kernel launch failed");
                beta = device_norm(d_u.data(), n);
                if (!finite_double_bits(beta)) {
                    stop_reason = "nonfinite_beta";
                    break;
                }
                if (beta > 0.0) {
                    gi_lsmr_scale_copy_kernel<<<blocks_n, kBlockSize>>>(
                        d_u.data(), 1.0 / beta, n, d_u.data());
                    cuda_check(cudaGetLastError(),
                               "GI LSMR u normalization kernel launch failed");
                    apply_at(d_u.data(), d_atu.data());
                    gi_lsmr_affine_subtract_kernel<<<blocks_fe, kBlockSize>>>(
                        d_atu.data(), d_v.data(), beta, total_fe, d_v.data());
                    cuda_check(cudaGetLastError(),
                               "GI LSMR v recurrence kernel launch failed");
                    alpha = device_norm(d_v.data(), total_fe);
                    if (!finite_double_bits(alpha)) {
                        stop_reason = "nonfinite_alpha";
                        break;
                    }
                    if (alpha > 0.0) {
                        gi_lsmr_scale_copy_kernel<<<blocks_fe, kBlockSize>>>(
                            d_v.data(), 1.0 / alpha, total_fe, d_v.data());
                        cuda_check(cudaGetLastError(),
                                   "GI LSMR v normalization kernel launch failed");
                    } else {
                        cuda_check(cudaMemset(d_v.data(), 0,
                                              sizeof(double) * total_fe),
                                   "cudaMemset GI LSMR v failed");
                    }
                } else {
                    cuda_check(cudaMemset(d_v.data(), 0,
                                          sizeof(double) * total_fe),
                               "cudaMemset GI LSMR exhausted v failed");
                    alpha = 0.0;
                }

                const Rotation qhat = sym_ortho(alphabar, 0.0);
                const double alphahat = qhat.r;
                const double rho_old = rho;
                const Rotation q = sym_ortho(alphahat, beta);
                rho = q.r;
                if (rho == 0.0) {
                    stop_reason = "rho_zero";
                    break;
                }
                const double theta_new = q.s * alpha;
                alphabar = q.c * alpha;
                const double rhobar_old = rhobar;
                const double zeta_old = zeta;
                const double theta_bar = sbar * rho;
                const double rho_temp = cbar * rho;
                const Rotation qbar = sym_ortho(rho_temp, theta_new);
                cbar = qbar.c;
                sbar = qbar.s;
                rhobar = qbar.r;
                if (rhobar == 0.0 || rhobar_old == 0.0 || rho_old == 0.0) {
                    stop_reason = "rhobar_zero";
                    break;
                }
                zeta = cbar * zetabar;
                zetabar = -sbar * zetabar;

                gi_lsmr_update_solution_kernel<<<blocks_fe, kBlockSize>>>(
                    d_solution.data(), d_hbar.data(), d_h.data(),
                    -(theta_bar * rho / (rho_old * rhobar_old)),
                    zeta / (rho * rhobar), total_fe);
                cuda_check(cudaGetLastError(),
                           "GI LSMR solution kernel launch failed");
                gi_lsmr_update_h_kernel<<<blocks_fe, kBlockSize>>>(
                    d_h.data(), d_v.data(), -(theta_new / rho), total_fe);
                cuda_check(cudaGetLastError(),
                           "GI LSMR h kernel launch failed");

                const double beta_acute = qhat.c * betadd;
                const double beta_check = -qhat.s * betadd;
                const double beta_hat = q.c * beta_acute;
                betadd = -q.s * beta_acute;
                const double theta_tilde_old = thetatilde;
                const Rotation qtilde = sym_ortho(rhodold, theta_bar);
                if (qtilde.r == 0.0) {
                    stop_reason = "qtilde_zero";
                    break;
                }
                thetatilde = qtilde.s * rhobar;
                rhodold = qtilde.c * rhobar;
                if (rhodold == 0.0) {
                    stop_reason = "rhodold_zero";
                    break;
                }
                betad = -qtilde.s * betad + qtilde.c * beta_hat;
                tautildeold =
                    (zeta_old - theta_tilde_old * tautildeold) / qtilde.r;
                const double taud =
                    (zeta - thetatilde * tautildeold) / rhodold;
                dnorm += beta_check * beta_check;
                norm_r = std::sqrt(std::max(
                    0.0, dnorm + (betad - taud) * (betad - taud) +
                             betadd * betadd));
                norm_a_sq += beta * beta;
                norm_a = std::sqrt(norm_a_sq);
                norm_a_sq += alpha * alpha;
                max_rbar = std::max(max_rbar, rhobar_old);
                if (k > 0) {
                    min_rbar = std::min(min_rbar, rhobar_old);
                }
                const double condition_denom = std::min(min_rbar, rho_temp);
                condition = condition_denom > 0.0
                                ? std::max(max_rbar, rho_temp) /
                                      condition_denom
                                : std::numeric_limits<double>::infinity();

                const double norm_ar = std::abs(zetabar);
                const double norm_x = device_norm(d_solution.data(), total_fe);
                if (!finite_double_bits(norm_x)) {
                    stop_reason = "nonfinite_solution";
                    break;
                }
                final_test1 = norm_r / norm_b;
                final_test2 = norm_a * norm_r != 0.0
                                  ? norm_ar / (norm_a * norm_r)
                                  : std::numeric_limits<double>::infinity();
                const double residual_tolerance =
                    tolerance + tolerance * norm_a * norm_x / norm_b;
                // Forward-error guard (same rule as the CPU solver): a dual
                // stop is accepted only when cond(A) * test2 meets the public
                // forward target; nearly null incidence directions otherwise
                // pass the dual test with a large projection error.
                const bool dual_stop = final_test2 <= tolerance;
                const bool forward_ok = ordinary_projection ||
                    condition * final_test2 <= forward_target;
                if (final_test1 <= residual_tolerance ||
                    (dual_stop && forward_ok) || alpha == 0.0) {
                    converged = true;
                    stop_reason = final_test1 <= residual_tolerance
                                      ? "test1"
                                      : (dual_stop ? "test2" : "alpha_zero");
                    ++k;
                    break;
                }
                if (dual_stop && !forward_ok &&
                    final_test2 <= 64.0 * std::numeric_limits<double>::epsilon()) {
                    // FP64 floor: the dual stop stands (an inconclusive bound
                    // never refuses); the unresolved guard is a diagnostic.
                    converged = true;
                    stop_reason = "forward_guard_floor";
                    ++k;
                    break;
                }
                if (condition >= condition_limit) {
                    stop_reason = "condition";
                    ++k;
                    break;
                }
                // F2-A probe of the three-FE completion: it reads host scalars
                // only, so a fit that keeps the device tail is unchanged.
                if (probe != nullptr &&
                    probe->observe(rhs_index, k + 1, norm_ar, final_test2,
                                   tolerance)) {
                    stop_reason = "handover";
                    ++k;
                    break;
                }
            }
            iterations = k;
            if (probe != nullptr && !probe->handover) probe->finish(rhs_index);
            if (ordinary_projection && finite_double_bits(final_test2)) {
                result.krylov_max_final_backward_error = std::max(
                    result.krylov_max_final_backward_error, final_test2);
            }
            result.krylov_max_condition =
                std::max(result.krylov_max_condition, condition);
            result.krylov_max_condition_times_backward_error =
                std::max(result.krylov_max_condition_times_backward_error,
                         finite_double_bits(final_test2) ? condition * final_test2
                                                         : condition);
            if (trace) {
                std::fprintf(stderr,
                             "gi_lsmr_rhs rhs=%d tol=%.3e iterations=%d "
                             "converged=%d reason=%s test1=%.3e test2=%.3e "
                             "condition=%.3e\n",
                             rhs_index, tolerance, iterations,
                             converged ? 1 : 0, stop_reason,
                             final_test1, final_test2, condition);
            }
        };

        int max_iterations = 0;
        bool all_converged = true;
        std::vector<double> captured_coefficients;
        if (capture_projection) {
            captured_coefficients.resize(total_fe);
            result.fe_alpha_y.resize(dims + 1);
            result.fe_alpha_X.resize(dims + 1);
            for (int d = 0; d <= dims; ++d) {
                const int count = d < dims ? standard_fe_inputs[d].num_groups : individuals;
                result.fe_alpha_y[d].resize(count);
                result.fe_alpha_X[d].resize(count, cols);
            }
        }
        // Applies A*solution to `target` (raw - A*solution) and folds the
        // scaled coefficients into the captured alphas.
        auto project_solution = [&](const double* raw, double* target,
                                    int rhs_index, bool accumulate) {
            apply_a(lsmr.solution.data(), lsmr.av.data(), false);
            if (capture_projection) {
                cuda_check(cudaMemcpy(captured_coefficients.data(), lsmr.scaled.data(),
                                      sizeof(double) * total_fe, cudaMemcpyDeviceToHost),
                           "GI LSMR projection coefficients copy failed");
                for (int d = 0; d <= dims; ++d) {
                    const int count = d < dims ? standard_fe_inputs[d].num_groups : individuals;
                    const int offset = d < dims ? offsets[d] : individual_offset;
                    Eigen::Map<const Eigen::VectorXd> coefficients(
                        captured_coefficients.data() + offset, count);
                    if (rhs_index == 0) {
                        if (accumulate) result.fe_alpha_y[d] += coefficients;
                        else result.fe_alpha_y[d] = coefficients;
                    } else if (accumulate) {
                        result.fe_alpha_X[d].col(rhs_index-1) += coefficients;
                    } else {
                        result.fe_alpha_X[d].col(rhs_index-1) = coefficients;
                    }
                }
            }
            const double correction = device_norm(lsmr.av.data(), n);
            gi_lsmr_make_residual_kernel<<<blocks_n, kBlockSize>>>(
                raw, lsmr.av.data(), n, target);
            cuda_check(cudaGetLastError(),
                       "GI LSMR residual kernel launch failed");
            return correction;
        };
        // Primary solve of one right-hand side (2.26.2 semantics).
        std::vector<int> rhs_iterations(static_cast<std::size_t>(cols) + 1U, 0);
        auto solve_and_project = [&](const double* raw,
                                     double* residual,
                                     int rhs_index) {
            int& iterations = rhs_iterations[static_cast<std::size_t>(rhs_index)];
            bool converged = false;
            solve_one(raw, rhs_index, iterations, converged, options.max_iter);
            max_iterations = std::max(max_iterations, iterations);
            if (probe != nullptr && probe->handover) {
                return;  // the host completes the tail; nothing to project
            }
            project_solution(raw, residual, rhs_index, false);
            all_converged = all_converged && converged;
        };

        solve_and_project(d_y.data(), lsmr.residual_y.data(), 0);
        for (int column = 0; column < cols; ++column) {
            if (probe != nullptr && probe->handover) break;
            solve_and_project(
                d_x.data() + static_cast<std::size_t>(column) * ld,
                lsmr.residual_x.data() +
                    static_cast<std::size_t>(column) * ld,
                column + 1);
        }
        if (probe != nullptr && probe->handover) {
            // F2-A: leave before any projection or copy-back. The caller
            // discards this device work and completes from the IT state;
            // every device iteration stays counted.
            probe->device_iterations = max_iterations;
            result.iterations = max_iterations;
            result.converged = false;
            result.precision_certified = false;
            result.mlsmr_used = true;
            result.sweep_order_used.clear();
            if (trace) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - solve_start).count();
                std::fprintf(stderr,
                             "gi_lsmr_handover rhs=%d k=%d iterations=%d "
                             "seconds=%.6f\n",
                             probe->rhs, probe->k, max_iterations, elapsed);
            }
            return true;
        }

        auto copy_residuals_to_host = [&]() {
            result.y_tilde.resize(n);
            staged_memcpy_d2h(result.y_tilde.data(),
                              lsmr.residual_y.data(),
                              sizeof(double) * n, workspace.h_stage,
                              "cudaMemcpy GI LSMR y_tilde failed");
            result.X_tilde.resize(ld, cols);
            if (cols > 0) {
                staged_memcpy_d2h(result.X_tilde.data(),
                                  lsmr.residual_x.data(),
                                  sizeof(double) * static_cast<std::size_t>(ld) * cols,
                                  workspace.h_stage,
                                  "cudaMemcpy GI LSMR X_tilde failed");
            }
        };

        // Residual refinement (reghdfe-comparable and strict-residual modes;
        // same rules as the CPU solver, see absorb_group_individual_lsmr_cpu):
        // in comparable mode applied to every right-hand side or to none;
        // when it changes the projection beyond the contract's resolution the
        // primary solution is certified first and the refinement is kept only
        // if it passes; never in xhdfe-fast. A correction below the forward
        // target of the mode leaves the projection within the contract;
        // strict-residual refines to the limit of its maximum-FE-mean gate
        // with a larger budget and keeps every stage (the gate adjudicates).
        if (!ordinary_projection && all_converged &&
            options.tolerance_mode != ToleranceMode::XhdfeFast) {
            const bool strict_mode =
                options.tolerance_mode == ToleranceMode::StrictResidual;
            const double refinement_target = strict_mode
                    ? std::max(std::max(0.0, options.tol),
                               64.0 * std::numeric_limits<double>::epsilon())
                    : group_forward_tolerance(options);
            const int max_stages = strict_mode ? 4 : 3;
            const std::size_t x_count = static_cast<std::size_t>(ld) * cols;
            auto device_copy = [&](double* dst, const double* src, std::size_t count,
                                   const char* what) {
                if (count == 0) return;
                cuda_check(cudaMemcpy(dst, src, sizeof(double) * count,
                                      cudaMemcpyDeviceToDevice), what);
            };
            DeviceBuffer<double> primary_y(static_cast<std::size_t>(n));
            DeviceBuffer<double> primary_x;
            if (cols > 0) primary_x.allocate(x_count);
            device_copy(primary_y.data(), lsmr.residual_y.data(), n,
                        "GI LSMR residual backup failed");
            device_copy(primary_x.data(), lsmr.residual_x.data(), x_count,
                        "GI LSMR residual backup failed");
            const std::vector<Eigen::VectorXd> primary_alpha_y = result.fe_alpha_y;
            const std::vector<Eigen::MatrixXd> primary_alpha_x = result.fe_alpha_X;
            bool significant = false;
            auto refine_rhs = [&](double* residual, int rhs_index) {
                int& iterations = rhs_iterations[static_cast<std::size_t>(rhs_index)];
                int budget = strict_mode ? std::max(32, iterations / 2)
                                         : std::max(16, iterations / 4);
                for (int stage = 0; stage < max_stages && budget > 0 &&
                                    iterations < options.max_iter; ++stage) {
                    const double residual_norm = device_norm(residual, n);
                    if (!finite_double_bits(residual_norm)) return false;
                    if (!(residual_norm > 0.0)) return true;
                    int extra = 0;
                    bool extra_converged = false;
                    solve_one(residual, rhs_index, extra, extra_converged,
                              std::min(budget, options.max_iter - iterations));
                    iterations += extra;
                    budget -= extra;
                    if (extra == 0) return false;
                    const double correction =
                        project_solution(residual, residual, rhs_index, true);
                    if (trace) {
                        std::fprintf(stderr,
                                     "gi_lsmr_refine rhs=%d stage=%d iterations=%d "
                                     "correction=%.3e residual=%.3e\n",
                                     rhs_index, stage + 1, extra, correction,
                                     residual_norm);
                    }
                    if (!finite_double_bits(correction)) return false;
                    if (correction <= refinement_target * residual_norm) return true;
                    significant = true;
                }
                return false;
            };
            bool complete = refine_rhs(lsmr.residual_y.data(), 0);
            for (int column = 0; (complete || strict_mode) && column < cols; ++column) {
                complete = refine_rhs(
                    lsmr.residual_x.data() + static_cast<std::size_t>(column) * ld,
                    column + 1) && complete;
            }
            bool keep = complete || strict_mode;
            bool gate = true;
            bool gate_checked = false;
            if (keep && significant) {
                // Certify the primary solution (host copy) before keeping a
                // refinement that changed the projection materially.
                DeviceBuffer<double> refined_y(static_cast<std::size_t>(n));
                DeviceBuffer<double> refined_x;
                if (cols > 0) refined_x.allocate(x_count);
                device_copy(refined_y.data(), lsmr.residual_y.data(), n,
                            "GI LSMR refined backup failed");
                device_copy(refined_x.data(), lsmr.residual_x.data(), x_count,
                            "GI LSMR refined backup failed");
                const std::vector<Eigen::VectorXd> refined_alpha_y = result.fe_alpha_y;
                const std::vector<Eigen::MatrixXd> refined_alpha_x = result.fe_alpha_X;
                device_copy(lsmr.residual_y.data(), primary_y.data(), n,
                            "GI LSMR residual restore failed");
                device_copy(lsmr.residual_x.data(), primary_x.data(), x_count,
                            "GI LSMR residual restore failed");
                result.fe_alpha_y = primary_alpha_y;
                result.fe_alpha_X = primary_alpha_x;
                copy_residuals_to_host();
                std::vector<Eigen::VectorXi> fes;
                fes.reserve(static_cast<std::size_t>(dims));
                for (int d = 0; d < dims; ++d) {
                    fes.emplace_back(Eigen::Map<const Eigen::VectorXi>(
                        standard_fe_inputs[static_cast<std::size_t>(d)].group_ids, n));
                }
                result.converged = true;
                certify_absorption_result(y, X, fes, weights, options, {}, result, &gi);
                gate = result.precision_certified;
                gate_checked = true;
                if (gate) {
                    device_copy(lsmr.residual_y.data(), refined_y.data(), n,
                                "GI LSMR refined restore failed");
                    device_copy(lsmr.residual_x.data(), refined_x.data(), x_count,
                                "GI LSMR refined restore failed");
                    result.fe_alpha_y = refined_alpha_y;
                    result.fe_alpha_X = refined_alpha_x;
                } else {
                    keep = false;
                }
            } else if (!keep) {
                device_copy(lsmr.residual_y.data(), primary_y.data(), n,
                            "GI LSMR residual restore failed");
                device_copy(lsmr.residual_x.data(), primary_x.data(), x_count,
                            "GI LSMR residual restore failed");
                result.fe_alpha_y = primary_alpha_y;
                result.fe_alpha_X = primary_alpha_x;
            }
            for (int rhs = 0; rhs <= cols; ++rhs) {
                max_iterations = std::max(
                    max_iterations, rhs_iterations[static_cast<std::size_t>(rhs)]);
            }
            if (trace) {
                std::fprintf(stderr,
                             "gi_lsmr_refine_outcome significant=%d gate_checked=%d gate=%d "
                             "complete=%d kept=%d iterations=%d\n",
                             significant ? 1 : 0, gate_checked ? 1 : 0, gate ? 1 : 0,
                             complete ? 1 : 0, keep ? 1 : 0, max_iterations);
            }
        }
        copy_residuals_to_host();

        result.iterations = max_iterations;
        result.converged = all_converged;
        result.precision_certified = false;
        result.mlsmr_used = true;
        result.sweep_order_used.clear();
        if (trace) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start).count();
            std::fprintf(stderr,
                         "gi_lsmr_solve tol=%.3e rhs=%d iterations=%d "
                         "converged=%d seconds=%.6f\n",
                         solve_tolerance, cols + 1,
                         max_iterations, all_converged ? 1 : 0, elapsed);
        }
        return true;
    } catch (const std::exception& e) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr || trace) {
            std::fprintf(stderr, "[gpu] group/individual LSMR failed: %s\n",
                         e.what());
        }
        cudaGetLastError();
        result.converged = false;
        result.precision_certified = false;
        result.mlsmr_used = true;
        return false;
    } catch (...) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr || trace) {
            std::fprintf(stderr,
                         "[gpu] group/individual LSMR failed: unknown\n");
        }
        cudaGetLastError();
        result.converged = false;
        result.precision_certified = false;
        result.mlsmr_used = true;
        return false;
    }
}

static bool absorb_ordinary_lsmr_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& fe_inputs,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionResult& result,
    CudaTailProbe* probe) {
    if (y.size() <= 0 || y.size() > std::numeric_limits<int>::max() ||
        X.rows() != y.size() || fe_inputs.empty() ||
        options.retain_fixed_effects) {
        return false;
    }
    const int n = static_cast<int>(y.size());
    const GpuFeInput& last = fe_inputs.back();
    if (last.group_ids == nullptr || last.num_groups <= 0) return false;

    // An ordinary FE is a one-membership incidence operator. Express the
    // final dimension that way, without changing observations or statistics.
    GroupIndividualStructure gi;
    gi.num_groups = n;
    gi.num_individuals = last.num_groups;
    gi.group_ptr.resize(static_cast<std::size_t>(n) + 1U);
    std::iota(gi.group_ptr.begin(), gi.group_ptr.end(), 0);
    gi.group_individual.assign(last.group_ids, last.group_ids + n);
    gi.group_scale.assign(static_cast<std::size_t>(n), 1.0);
    gi.individual_ptr.assign(static_cast<std::size_t>(last.num_groups) + 1U, 0);
    for (const int group : gi.group_individual) {
        if (group < 0 || group >= last.num_groups) return false;
        ++gi.individual_ptr[static_cast<std::size_t>(group) + 1U];
    }
    std::partial_sum(gi.individual_ptr.begin(), gi.individual_ptr.end(),
                     gi.individual_ptr.begin());
    std::vector<int> cursor = gi.individual_ptr;
    gi.individual_group.resize(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        gi.individual_group[static_cast<std::size_t>(
            cursor[static_cast<std::size_t>(last.group_ids[row])]++)] = row;
    }
    const std::vector<GpuFeInput> standard(fe_inputs.begin(), fe_inputs.end() - 1);
    const bool ok = absorb_group_individual_lsmr_cuda_impl(
        y, X, standard, gi, weights, options, result, true, probe);
    // The fit layer reports this device Krylov diversion as LSMR.
    result.mlsmr_used = true;
    if (ok) result.fe_levels.back() = last.num_levels_present;
    return ok;
}

bool absorb_fixed_effects_group_individual_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& standard_fe_inputs,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const std::vector<std::size_t>& sweep_order,
    const HdfeOptions& options,
    AbsorptionMethod method,
    AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    if (n <= 0 || ld != n) {
        return false;
    }
    if (gi.num_groups != n || gi.num_individuals <= 0) {
        return false;
    }
    if (static_cast<int>(gi.group_ptr.size()) != n + 1) {
        return false;
    }
    if (static_cast<int>(gi.group_scale.size()) != n) {
        return false;
    }
    if (gi.group_ptr.empty()) {
        return false;
    }
    if (static_cast<std::size_t>(gi.group_ptr.back()) != gi.group_individual.size()) {
        return false;
    }
    if (static_cast<int>(gi.individual_ptr.size()) != gi.num_individuals + 1 ||
        gi.individual_ptr.empty() ||
        static_cast<std::size_t>(gi.individual_ptr.back()) !=
            gi.individual_group.size() ||
        gi.group_individual.size() != gi.individual_group.size()) {
        return false;
    }
    if (method == AbsorptionMethod::Lsmr || options.from_auto) {
        return absorb_group_individual_lsmr_cuda_impl(
            y, X, standard_fe_inputs, gi, weights, options, result);
    }
    if (method == AbsorptionMethod::Jacobi) {
        return false;
    }

    std::vector<std::size_t> order = sweep_order;
    const std::size_t k = standard_fe_inputs.size();
    auto default_order = [&]() {
        order.clear();
        order.reserve(k);
        for (std::size_t d = 0; d < k; ++d) {
            order.push_back(d);
        }
    };
    if (k > 0) {
        if (order.empty()) {
            default_order();
        } else {
            std::vector<uint8_t> seen(k, 0);
            bool ok = true;
            for (const std::size_t d : order) {
                if (d >= k || seen[d]) {
                    ok = false;
                    break;
                }
                seen[d] = 1;
            }
            if (!ok || order.size() != k) {
                default_order();
            }
        }
    } else {
        order.clear();
    }

    double relaxation = options.jacobi_relaxation;
    if (!(relaxation > 0.0 && relaxation <= 1.0)) {
        relaxation = 1.0;
    }
    relaxation = std::min(relaxation, 1.0);

    try {
        const bool unit_weights = (weights == nullptr);
        const bool use_symmetric =
            (method == AbsorptionMethod::SymmetricGaussSeidel) && ((order.size() + 1) > 1);

        CudaWorkspace& workspace = cuda_workspace;

        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), y.data(), sizeof(double) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy y failed");

        DeviceBuffer<double>& d_x = workspace.d_x;
        if (cols > 0) {
            d_x.allocate(static_cast<std::size_t>(ld) * cols);
            cuda_check(cudaMemcpy(d_x.data(), X.data(), sizeof(double) * ld * cols,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy X failed");
        } else {
            d_x.reset();
        }

        DeviceBuffer<double>& d_weights = workspace.d_weights;
        if (!unit_weights) {
            d_weights.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_weights.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy weights failed");
        } else {
            d_weights.reset();
        }

        std::vector<CudaFeDevice>& fe_dev = workspace.fe_dev;
        if (fe_dev.size() != k) {
            fe_dev.resize(k);
        }
        for (std::size_t idx = 0; idx < k; ++idx) {
            const auto& fe = standard_fe_inputs[idx];
            CudaFeDevice& dev = fe_dev[idx];
            dev.num_groups = fe.num_groups;
            dev.num_levels_present = fe.num_levels_present;
            // Same privatized-vs-segmented dispatch as the main path.
            constexpr std::size_t kPrivatizedShmemBudget = 40 * 1024;
            const std::size_t shmem_needed =
                static_cast<std::size_t>(fe.num_groups) *
                (1 + static_cast<std::size_t>(cols)) * sizeof(double);
            dev.use_privatized_sums =
                (fe.num_groups > 0 && shmem_needed <= kPrivatizedShmemBudget);
            dev.privatized_shmem_bytes = dev.use_privatized_sums ? shmem_needed : 0;
            dev.use_segmented_sums = !dev.use_privatized_sums;
            dev.gid.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(dev.gid.data(), fe.group_ids, sizeof(int) * n,
                              workspace.h_stage, "cudaMemcpy group ids failed");
            dev.weight_sums.allocate(static_cast<std::size_t>(fe.num_groups));
            staged_memcpy_h2d(dev.weight_sums.data(), fe.weight_sums,
                              sizeof(double) * fe.num_groups, workspace.h_stage,
                              "cudaMemcpy weight sums failed");
            dev.sum_y.allocate(static_cast<std::size_t>(fe.num_groups));
            if (cols > 0) {
                dev.sum_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
            } else {
                dev.sum_x.reset();
            }
        }

        // Copy group/individual structure (individual -> groups adjacency + group scale).
        DeviceBuffer<double>& d_group_scale = workspace.gi_group_scale;
        d_group_scale.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_group_scale.data(), gi.group_scale.data(),
                              sizeof(double) * static_cast<std::size_t>(n),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.group_scale failed");

        const int individuals = gi.num_individuals;
        if (static_cast<int>(gi.individual_ptr.size()) != individuals + 1) {
            return false;
        }
        if (gi.individual_ptr.empty()) {
            return false;
        }
        if (static_cast<std::size_t>(gi.individual_ptr.back()) != gi.individual_group.size()) {
            return false;
        }

        DeviceBuffer<int>& d_ind_ptr = workspace.gi_individual_ptr;
        d_ind_ptr.allocate(static_cast<std::size_t>(individuals) + 1);
        cuda_check(cudaMemcpy(d_ind_ptr.data(), gi.individual_ptr.data(),
                              sizeof(int) * (static_cast<std::size_t>(individuals) + 1),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.individual_ptr failed");

        const std::size_t edges = gi.individual_group.size();
        DeviceBuffer<int>& d_ind_group = workspace.gi_individual_group;
        d_ind_group.allocate(edges);
        cuda_check(cudaMemcpy(d_ind_group.data(), gi.individual_group.data(),
                              sizeof(int) * edges, cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.individual_group failed");

        // Greedy-color the individuals so individuals sharing an observation
        // row land in different colors: each color is then one race-free
        // Gauss-Seidel kernel launch (see gi_color_sweep_kernel). The fixed
        // visit order makes the coloring (and the sweep) deterministic.
        std::vector<int> color(static_cast<std::size_t>(individuals), 0);
        int num_colors = 1;
        {
            std::vector<int> used;  // used[c] == i marks color c taken by a neighbor of i
            for (int i = 0; i < individuals; ++i) {
                const int beg = gi.individual_ptr[static_cast<std::size_t>(i)];
                const int end = gi.individual_ptr[static_cast<std::size_t>(i) + 1];
                for (int pos = beg; pos < end; ++pos) {
                    const int g = gi.individual_group[static_cast<std::size_t>(pos)];
                    if (g < 0 || g >= n) {
                        continue;
                    }
                    const int gbeg = gi.group_ptr[static_cast<std::size_t>(g)];
                    const int gend = gi.group_ptr[static_cast<std::size_t>(g) + 1];
                    for (int pos2 = gbeg; pos2 < gend; ++pos2) {
                        const int j = gi.group_individual[static_cast<std::size_t>(pos2)];
                        if (j < 0 || j >= i) {
                            continue;  // only already-colored neighbors matter
                        }
                        const int cj = color[static_cast<std::size_t>(j)];
                        if (cj >= static_cast<int>(used.size())) {
                            used.resize(static_cast<std::size_t>(cj) + 1, -1);
                        }
                        used[static_cast<std::size_t>(cj)] = i;
                    }
                }
                int c = 0;
                while (c < static_cast<int>(used.size()) &&
                       used[static_cast<std::size_t>(c)] == i) {
                    ++c;
                }
                color[static_cast<std::size_t>(i)] = c;
                num_colors = std::max(num_colors, c + 1);
            }
        }
        std::vector<int> color_offsets(static_cast<std::size_t>(num_colors) + 1, 0);
        for (int i = 0; i < individuals; ++i) {
            ++color_offsets[static_cast<std::size_t>(color[static_cast<std::size_t>(i)]) + 1];
        }
        for (int c = 0; c < num_colors; ++c) {
            color_offsets[static_cast<std::size_t>(c) + 1] +=
                color_offsets[static_cast<std::size_t>(c)];
        }
        std::vector<int> individuals_by_color(static_cast<std::size_t>(individuals), 0);
        {
            std::vector<int> cursor(color_offsets.begin(), color_offsets.end() - 1);
            for (int i = 0; i < individuals; ++i) {
                const int c = color[static_cast<std::size_t>(i)];
                individuals_by_color[static_cast<std::size_t>(
                    cursor[static_cast<std::size_t>(c)]++)] = i;
            }
        }
        DeviceBuffer<int>& d_color_individuals = workspace.gi_color_individuals;
        d_color_individuals.allocate(static_cast<std::size_t>(individuals));
        cuda_check(cudaMemcpy(d_color_individuals.data(), individuals_by_color.data(),
                              sizeof(int) * static_cast<std::size_t>(individuals),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi color order failed");

        DeviceBuffer<double>& d_denom = workspace.gi_denom;
        d_denom.allocate(static_cast<std::size_t>(individuals));
        const int blocks_i = (individuals + kBlockSize - 1) / kBlockSize;
        gi_denom_by_individual_kernel<<<blocks_i, kBlockSize>>>(
            d_ind_ptr.data(),
            d_ind_group.data(),
            d_group_scale.data(),
            unit_weights ? nullptr : d_weights.data(),
            unit_weights,
            n,
            individuals,
            d_denom.data());
        cuda_check(cudaGetLastError(), "gi denom (by individual) kernel launch failed");

        // Keep the unused buffers around for workspace reuse, but avoid forcing allocations here.
        workspace.gi_numer_y.reset();
        workspace.gi_numer_x.reset();
        workspace.gi_alpha_y.reset();
        workspace.gi_alpha_x.reset();

        DeviceBuffer<double>& d_sumsq = workspace.d_sumsq;
        d_sumsq.allocate(1);

        auto compute_norm = [&](double& out_norm) {
            cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)), "cudaMemset sumsq failed");
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                d_y.data(), d_x.data(), n, cols, ld, d_sumsq.data());
            cuda_check(cudaGetLastError(), "sumsq kernel launch failed");
            double sumsq = 0.0;
            cuda_check(cudaMemcpy(&sumsq, d_sumsq.data(), sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy sumsq failed");
            out_norm = std::sqrt(sumsq);
        };

        auto run_standard_demean = [&](CudaFeDevice& fe) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_h = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }
            if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }
            compute_means_kernel<<<blocks_h, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(),
                d_x.data(),
                n,
                cols,
                ld,
                fe.gid.data(),
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.num_groups);
            cuda_check(cudaGetLastError(), "apply_means kernel launch failed");
        };

        auto run_individual_sweep = [&]() {
            for (int c = 0; c < num_colors; ++c) {
                const int color_begin = color_offsets[static_cast<std::size_t>(c)];
                const int color_count =
                    color_offsets[static_cast<std::size_t>(c) + 1] - color_begin;
                if (color_count <= 0) {
                    continue;
                }
                const int blocks_c = (color_count + kBlockSize - 1) / kBlockSize;
                gi_color_sweep_kernel<<<blocks_c, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    d_ind_ptr.data(),
                    d_ind_group.data(),
                    d_group_scale.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    d_denom.data(),
                    d_color_individuals.data(),
                    color_begin,
                    color_count,
                    relaxation);
                cuda_check(cudaGetLastError(), "gi color sweep kernel launch failed");
            }
        };

        auto run_full_sweep = [&]() {
            for (const std::size_t dim : order) {
                run_standard_demean(fe_dev[dim]);
            }
            run_individual_sweep();
            if (use_symmetric) {
                run_individual_sweep();
                for (std::size_t idx = order.size(); idx-- > 0;) {
                    run_standard_demean(fe_dev[order[idx]]);
                }
            }
        };

        double prev_norm = 0.0;
        compute_norm(prev_norm);
        const int check_interval = std::max(1, options.convergence_check_interval);
        int last_check_iter = -1;
        bool converged = false;

        for (int iter = 0; iter < options.max_iter; ++iter) {
            run_full_sweep();

            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            if (do_check) {
                double curr_norm = 0.0;
                compute_norm(curr_norm);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (step > 0 &&
                    (rel_change / static_cast<double>(step)) <
                        group_individual_absorption_tolerance(options)) {
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
        }

        result.iterations = converged ? std::max(1, result.iterations)
                                      : options.max_iter;

        result.sweep_order_used.clear();
        result.sweep_order_used.reserve(order.size());
        for (const std::size_t dim : order) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }

        // Match the CPU warm start exactly and count every on-device sweep.
        constexpr int kInitialPolishSweeps = 16;
        if (converged && result.iterations < options.max_iter) {
            const int initial_sweeps = std::min(
                kInitialPolishSweeps, options.max_iter - result.iterations);
            for (int sweep = 0; sweep < initial_sweeps; ++sweep) {
                run_full_sweep();
                ++result.iterations;
            }
        }

        auto copy_candidate_to_host = [&]() {
            result.y_tilde.resize(n);
            cuda_check(cudaMemcpy(result.y_tilde.data(), d_y.data(),
                                  sizeof(double) * n, cudaMemcpyDeviceToHost),
                       "cudaMemcpy y_tilde failed");
            result.X_tilde.resize(ld, cols);
            if (cols > 0) {
                cuda_check(cudaMemcpy(result.X_tilde.data(), d_x.data(),
                                      sizeof(double) * ld * cols,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy X_tilde failed");
            }
        };

        std::vector<Eigen::VectorXi> certificate_fes;
        certificate_fes.reserve(standard_fe_inputs.size());
        for (const auto& input : standard_fe_inputs) {
            Eigen::VectorXi ids(n);
            for (int row = 0; row < n; ++row) {
                ids(row) = input.group_ids[row];
            }
            certificate_fes.push_back(std::move(ids));
        }

        copy_candidate_to_host();
        bool certified = certify_group_individual_candidate(
            y, X, certificate_fes, gi, weights, options, result);
        constexpr int kCertificateBatchSweeps = 64;
        while (!certified && result.iterations < options.max_iter) {
            const int batch = std::min(
                kCertificateBatchSweeps, options.max_iter - result.iterations);
            for (int sweep = 0; sweep < batch; ++sweep) {
                run_full_sweep();
                ++result.iterations;
            }
            copy_candidate_to_host();
            certified = certify_group_individual_candidate(
                y, X, certificate_fes, gi, weights, options, result);
        }
        result.converged = certified;
        result.precision_certified = certified;
        return true;
    } catch (...) {
        return false;
    }
}

// GPU port of fe_recovery_max_delta (see fe_absorption.cpp:5996). For each FE
// dimension in sweep_order: (a) accumulate weighted group sums of `residual`,
// (b) convert to group means, (c) track the maximum |mean_g| across dims and
// groups, (d) subtract the means from `residual` in-place (Gauss-Seidel).
// Returns the final max |mean| across all dims via out_max_delta. On any CUDA
// error returns false; caller falls back to CPU implementation.
bool fe_recovery_max_delta_cuda(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                const std::vector<GpuFeInput>& fe_inputs,
                                const Eigen::VectorXd* weights,
                                const std::vector<std::size_t>& sweep_order,
                                double& out_max_delta) {
    try {
        const int n = static_cast<int>(residual.size());
        const std::size_t dims = fe_inputs.size();
        if (n == 0 || dims == 0) {
            out_max_delta = 0.0;
            return true;
        }
        if (weights && weights->size() != n) {
            return false;
        }

        std::vector<std::size_t> order = sweep_order;
        if (order.empty()) {
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
        }
        for (const std::size_t d : order) {
            if (d >= dims) {
                return false;
            }
        }

        const bool unit_weights = (weights == nullptr);
        const int blocks_n = (n + kBlockSize - 1) / kBlockSize;

        DeviceBuffer<double> d_y;
        DeviceBuffer<double> d_w;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), residual.data(), sizeof(double) * n,
                              cudaMemcpyHostToDevice),
                   "fe_recovery_max_delta: H2D residual failed");
        if (!unit_weights) {
            d_w.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_w.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D weights failed");
        }

        std::vector<DeviceBuffer<int>> d_gids(dims);
        std::vector<DeviceBuffer<double>> d_weight_sums(dims);
        std::vector<DeviceBuffer<double>> d_sum_y(dims);
        for (std::size_t d = 0; d < dims; ++d) {
            const int groups = fe_inputs[d].num_groups;
            if (groups <= 0 || fe_inputs[d].group_ids == nullptr ||
                fe_inputs[d].weight_sums == nullptr) {
                return false;
            }
            d_gids[d].allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_gids[d].data(), fe_inputs[d].group_ids,
                                  sizeof(int) * n, cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D group_ids failed");
            d_weight_sums[d].allocate(static_cast<std::size_t>(groups));
            cuda_check(cudaMemcpy(d_weight_sums[d].data(), fe_inputs[d].weight_sums,
                                  sizeof(double) * groups, cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D weight_sums failed");
            d_sum_y[d].allocate(static_cast<std::size_t>(groups));
        }

        DeviceBuffer<double> d_max_abs;
        d_max_abs.allocate(1);
        cuda_check(cudaMemset(d_max_abs.data(), 0, sizeof(double)),
                   "fe_recovery_max_delta: memset max_abs failed");

        for (const std::size_t d : order) {
            const int groups = fe_inputs[d].num_groups;
            const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;

            cuda_check(cudaMemset(d_sum_y[d].data(), 0, sizeof(double) * groups),
                       "fe_recovery_max_delta: memset sum_y failed");

            // Accumulate weighted group sums of y. cols=0 → X/sum_x branch is
            // skipped inside the kernel, so nullptrs are safe.
            accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                d_gids[d].data(),
                unit_weights ? nullptr : d_w.data(), unit_weights,
                d_sum_y[d].data(), /*sum_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: accumulate kernel launch");

            // Convert sums to means (divide by weight_sums; zero-safe inside).
            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                d_sum_y[d].data(), /*sum_x=*/nullptr, d_weight_sums[d].data(),
                groups, /*cols=*/0);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: compute_means kernel launch");

            // Reduce max|mean_g| into d_max_abs (monotone in the |.|).
            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                d_sum_y[d].data(), groups, d_max_abs.data());
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: max_abs kernel launch");

            // Subtract means from residual in-place (Gauss-Seidel).
            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                d_gids[d].data(), d_sum_y[d].data(),
                /*mean_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: apply_means kernel launch");
        }

        double host_max = 0.0;
        cuda_check(cudaMemcpy(&host_max, d_max_abs.data(), sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "fe_recovery_max_delta: D2H max_abs failed");
        out_max_delta = host_max;
        return true;
    } catch (...) {
        return false;
    }
}

bool fe_recovery_max_delta_cuda_cached(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                       const std::vector<GpuFeInput>& fe_inputs,
                                       const Eigen::VectorXd* weights,
                                       const std::vector<std::size_t>& sweep_order,
                                       double& out_max_delta) {
    try {
        const int n = static_cast<int>(residual.size());
        const std::size_t dims = fe_inputs.size();
        if (n == 0 || dims == 0) {
            out_max_delta = 0.0;
            return true;
        }
        if (weights && weights->size() != n) {
            return false;
        }

        std::vector<std::size_t> order = sweep_order;
        if (order.empty()) {
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
        }
        for (const std::size_t d : order) {
            if (d >= dims) {
                return false;
            }
        }

        CudaWorkspace& workspace = cuda_workspace;
        if (workspace.fe_dev.size() != dims) {
            return false;
        }
        for (std::size_t d = 0; d < dims; ++d) {
            const int groups = fe_inputs[d].num_groups;
            if (groups <= 0 || fe_inputs[d].group_ids == nullptr ||
                fe_inputs[d].weight_sums == nullptr) {
                return false;
            }
            CudaFeDevice& dev = workspace.fe_dev[d];
            if (dev.num_groups != groups ||
                dev.num_levels_present != fe_inputs[d].num_levels_present ||
                dev.gid.size() < static_cast<std::size_t>(n) ||
                dev.weight_sums.size() < static_cast<std::size_t>(groups) ||
                dev.sum_y.size() < static_cast<std::size_t>(groups)) {
                return false;
            }
            // Reuse allocated buffers, not FE contents identified only by shape.
            cuda_check(cudaMemcpy(dev.gid.data(), fe_inputs[d].group_ids, sizeof(int) * n,
                                  cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta_cached: refresh group_ids failed");
            cuda_check(cudaMemcpy(dev.weight_sums.data(), fe_inputs[d].weight_sums,
                                  sizeof(double) * groups, cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta_cached: refresh weight_sums failed");
        }

        const bool unit_weights = (weights == nullptr);
        const int blocks_n = (n + kBlockSize - 1) / kBlockSize;

        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), residual.data(), sizeof(double) * n,
                              cudaMemcpyHostToDevice),
                   "fe_recovery_max_delta_cached: H2D residual failed");

        DeviceBuffer<double>& d_w = workspace.d_weights;
        if (!unit_weights) {
            d_w.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_w.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta_cached: H2D weights failed");
        }

        DeviceBuffer<double>& d_max_abs = workspace.d_sumsq;
        d_max_abs.allocate(1);
        cuda_check(cudaMemset(d_max_abs.data(), 0, sizeof(double)),
                   "fe_recovery_max_delta_cached: memset max_abs failed");

        for (const std::size_t d : order) {
            CudaFeDevice& dev = workspace.fe_dev[d];
            const int groups = fe_inputs[d].num_groups;
            const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;

            cuda_check(cudaMemset(dev.sum_y.data(), 0, sizeof(double) * groups),
                       "fe_recovery_max_delta_cached: memset sum_y failed");

            accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                dev.gid.data(),
                unit_weights ? nullptr : d_w.data(), unit_weights,
                dev.sum_y.data(), /*sum_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: accumulate kernel launch");

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                dev.sum_y.data(), /*sum_x=*/nullptr, dev.weight_sums.data(),
                groups, /*cols=*/0);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: compute_means kernel launch");

            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                dev.sum_y.data(), groups, d_max_abs.data());
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: max_abs kernel launch");

            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                dev.gid.data(), dev.sum_y.data(),
                /*mean_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: apply_means kernel launch");
        }

        double host_max = 0.0;
        cuda_check(cudaMemcpy(&host_max, d_max_abs.data(), sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "fe_recovery_max_delta_cached: D2H max_abs failed");
        out_max_delta = host_max;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace detail
}  // namespace hdfe
