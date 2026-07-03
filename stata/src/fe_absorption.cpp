#include "fe_absorption.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>
#include <random>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include "fe_absorption_cuda.hpp"
#include "fe_absorption_metal.hpp"
#include "schwarz_demean.hpp"

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe {
namespace detail {
namespace {

#ifndef HDFE_GPU_BACKEND_DEFAULT
#define HDFE_GPU_BACKEND_DEFAULT "cpu"
#endif

// Mean group occupancy (observations per FE level) at or below which a
// multi-FE problem is treated as ill-conditioned for plain Gauss-Seidel and is
// routed to the Irons-Tuck accelerated absorber regardless of n. Sparse
// "tree-like" graphs (citation / board-membership / matched-pair panels such
// as the patents and directors data) sit far below this; well-connected panels
// sit far above it and keep the previous behaviour. Tunable at runtime via the
// XHDFE_ACCEL_OCC environment variable (for benchmarking only).
constexpr double kAccelOccupancyThreshold = 12.0;

GpuBackend parse_gpu_backend(const char* raw) {
    if (!raw) {
        return GpuBackend::Cpu;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "cuda") {
        return GpuBackend::Cuda;
    }
    if (value == "metal") {
        return GpuBackend::Metal;
    }
    return GpuBackend::Cpu;
}

GpuBackend resolve_gpu_backend() {
    const char* env = std::getenv("XHDFE_GPU_BACKEND");
    if (env && *env) {
        return parse_gpu_backend(env);
    }
    return parse_gpu_backend(HDFE_GPU_BACKEND_DEFAULT);
}

bool gpu_backend_requested(GpuBackend backend) {
    return backend != GpuBackend::Cpu;
}

bool strict_residual_tolerance_mode(const HdfeOptions& options) {
    return options.tolerance_mode == ToleranceMode::StrictResidual;
}

// Prototype knob (measurement phase): XHDFE_TOL_TRIGGER=change switches the
// accelerated solvers' stopping rule to the reghdfe-comparable norm-of-change
// criterion  ‖G(x)−x‖ / max(1,‖x‖) < tol  (one full sweep moves the iterate
// by less than tol, relatively). Off by default; the historical
// change-of-norm trigger is unaffected when unset.
bool honest_tol_trigger_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("XHDFE_TOL_TRIGGER");
        return e != nullptr && (e[0] == 'c' || e[0] == 'C' || e[0] == '1');
    }();
    return enabled;
}

double effective_absorption_tolerance(const HdfeOptions& options) {
    if (options.tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(options.tol, 1.0e-9);
    }
    return options.tol;
}

double limited_polish_tolerance(const HdfeOptions& options) {
    const double tol = effective_absorption_tolerance(options);
    return tol > 0.0 ? tol * 0.01 : tol;
}

double group_individual_absorption_tolerance(const HdfeOptions& options) {
    const double tol = effective_absorption_tolerance(options);
    if (tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(tol, 1.0e-12);
    }
    return tol;
}

void mark_gpu_unavailable(AbsorptionResult& result) {
    result.gpu_used = false;
    result.gpu_status_code = 2;
    result.gpu_attempted = false;
    result.gpu_absorption_converged = false;
    result.gpu_absorption_iterations = 0;
}

void mark_gpu_failure_fallback(AbsorptionResult& result,
                               const AbsorptionResult& gpu_result,
                               bool gpu_call_ok) {
    result.gpu_used = false;
    result.gpu_status_code = gpu_call_ok ? 3 : 4;
    result.gpu_attempted = true;
    const bool have_gpu_diag =
        gpu_result.gpu_status_code != 0 || gpu_result.gpu_absorption_iterations > 0;
    result.gpu_absorption_converged =
        have_gpu_diag ? gpu_result.gpu_absorption_converged : gpu_result.converged;
    result.gpu_absorption_iterations =
        have_gpu_diag ? gpu_result.gpu_absorption_iterations : gpu_result.iterations;
}

bool savefe_profile_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char* raw = std::getenv("XHDFE_PROFILE_SAVEFE");
    if (!raw || *raw == '\0' || *raw == '0') {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

void savefe_profile_log(const std::string& msg) {
    if (!savefe_profile_enabled()) {
        return;
    }
    std::cerr << "savefe_profile " << msg << '\n';
}

bool cpu_profile_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char* raw = std::getenv("XHDFE_PROFILE_CPU");
    if (!raw || *raw == '\0' || *raw == '0') {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

void cpu_profile_log_elapsed(
    const char* label,
    const std::chrono::steady_clock::time_point& start) {
    if (!cpu_profile_enabled()) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cerr << "cpu_profile label=" << label << " ms=" << elapsed.count() << '\n';
}

class SavefeTimer {
public:
    explicit SavefeTimer(const char* label)
        : label_(label), enabled_(savefe_profile_enabled()) {
        if (enabled_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~SavefeTimer() {
        if (!enabled_) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start_;
        std::cerr << "savefe_profile label=" << label_
                  << " ms=" << elapsed.count() << '\n';
    }

private:
    const char* label_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};

struct FeIndexer {
    std::vector<int> group_ids;
    std::vector<int> group_ptr;
    int num_groups = 0;
    int num_levels_present = 0;
    bool groups_contiguous = false;
};

struct ExactSparseUnionFind {
    std::vector<int> parent;
    std::vector<uint8_t> rank;

    explicit ExactSparseUnionFind(int n)
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

void finalize_contiguous_groups(FeIndexer& idx) {
    idx.groups_contiguous = false;
    idx.group_ptr.clear();

    const int n = static_cast<int>(idx.group_ids.size());
    if (n <= 0 || idx.num_groups <= 0) {
        return;
    }

    int prev = idx.group_ids[0];
    if (prev < 0 || prev >= idx.num_groups) {
        return;
    }
    for (int i = 1; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        if (g < prev) {
            return;
        }
        prev = g;
    }

    idx.groups_contiguous = true;
    idx.group_ptr.assign(static_cast<std::size_t>(idx.num_groups) + 1U, n);
    idx.group_ptr[0] = 0;

    const int first_group = idx.group_ids[0];
    for (int g = 1; g <= first_group; ++g) {
        idx.group_ptr[static_cast<std::size_t>(g)] = 0;
    }

    int last_group = first_group;
    for (int i = 1; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        if (g != last_group) {
            for (int fill = last_group + 1; fill <= g; ++fill) {
                idx.group_ptr[static_cast<std::size_t>(fill)] = i;
            }
            last_group = g;
        }
    }
    for (int fill = last_group + 1; fill <= idx.num_groups; ++fill) {
        idx.group_ptr[static_cast<std::size_t>(fill)] = n;
    }
}

inline bool indexer_has_contiguous_groups(const FeIndexer& idx, int n) {
    return idx.groups_contiguous &&
           static_cast<int>(idx.group_ids.size()) == n &&
           static_cast<int>(idx.group_ptr.size()) == idx.num_groups + 1;
}

FeIndexer build_indexer_from_groups(const std::vector<int>& group_ids, int expected_levels) {
    FeIndexer idx;
    if (group_ids.empty()) {
        return idx;
    }
    idx.group_ids = group_ids;

    // Vectorized min/max scan via Eigen (SIMD).
    Eigen::Map<const Eigen::VectorXi> ids_view(
        group_ids.data(), static_cast<Eigen::Index>(group_ids.size()));
    const int min_id = ids_view.minCoeff();
    const int max_id = ids_view.maxCoeff();
    if (min_id < 0) {
        throw std::runtime_error("Fixed-effect group ids must be non-negative");
    }
    const int derived_groups = max_id + 1;
    if (expected_levels > 0 && derived_groups <= expected_levels) {
        idx.num_groups = expected_levels;
    } else {
        idx.num_groups = derived_groups;
    }
    idx.num_levels_present = idx.num_groups;
    finalize_contiguous_groups(idx);
    return idx;
}

struct FeWorkspace {
    Eigen::VectorXd weight_sums_const;
    Eigen::VectorXd weight_sums_inv;
    Eigen::VectorXd y_sums;
    Eigen::MatrixXd x_sums;
    std::vector<uint8_t> touched_mask;
    std::vector<int> touched_groups;
    bool sparse_reset = false;
#ifdef HDFE_USE_OPENMP
    std::vector<Eigen::VectorXd> y_tls;
    std::vector<Eigen::MatrixXd> x_tls;
    int tls_threads = 0;
    int tls_cols = -1;
    // Optional sparse TLS bookkeeping to avoid clearing dense per-thread buffers
    // when each thread touches only a subset of groups (common when data is
    // roughly sorted by FE ids).
    bool tls_sparse = false;
    int tls_epoch = 0;
    std::vector<int> tls_epoch_tags;                   // [threads * groups] last epoch touched
    std::vector<std::vector<int>> tls_touched_groups;  // per-thread touched group ids
#endif

    FeWorkspace(int groups, int cols)
        : weight_sums_const(Eigen::VectorXd::Zero(groups)),
          weight_sums_inv(Eigen::VectorXd::Zero(groups)),
          y_sums(Eigen::VectorXd::Zero(groups)),
          x_sums(Eigen::MatrixXd::Zero(groups, cols)),
          touched_mask(static_cast<std::size_t>(groups), 0) {}

    int num_groups() const { return static_cast<int>(weight_sums_const.size()); }

    void refresh_weight_sums_inv() {
        const int groups = num_groups();
        if (groups <= 0) {
            weight_sums_inv.resize(0);
            return;
        }
        if (weight_sums_inv.size() != weight_sums_const.size()) {
            weight_sums_inv.resize(weight_sums_const.size());
        }
        const double* src = weight_sums_const.data();
        double* dst = weight_sums_inv.data();
#ifdef HDFE_USE_OPENMP
        constexpr int kRefreshParallelThreshold = 65536;
        if (groups >= kRefreshParallelThreshold) {
#pragma omp parallel for schedule(static)
            for (int g = 0; g < groups; ++g) {
                const double denom = src[g];
                dst[g] = denom > 0.0 ? 1.0 / denom : 0.0;
            }
            return;
        }
#endif
        for (int g = 0; g < groups; ++g) {
            const double denom = src[g];
            dst[g] = denom > 0.0 ? 1.0 / denom : 0.0;
        }
    }

    void reset() {
        if (!sparse_reset || touched_groups.empty()) {
            y_sums.setZero();
            if (x_sums.size() > 0) {
                x_sums.setZero();
            }
            if (!touched_groups.empty()) {
                std::fill(touched_mask.begin(), touched_mask.end(), 0);
                touched_groups.clear();
            }
            return;
        }
        const int groups = static_cast<int>(touched_groups.size());
        const int cols = static_cast<int>(x_sums.cols());
        for (int idx = 0; idx < groups; ++idx) {
            const int g = touched_groups[static_cast<std::size_t>(idx)];
            y_sums[g] = 0.0;
            if (cols > 0) {
                for (int c = 0; c < cols; ++c) {
                    x_sums(g, c) = 0.0;
                }
            }
            touched_mask[static_cast<std::size_t>(g)] = 0;
        }
        touched_groups.clear();
    }

#ifdef HDFE_USE_OPENMP
    void prepare_tls(int threads, int cols) {
        if (threads <= 1) {
            return;
        }
        const bool need_resize = threads != tls_threads || cols != tls_cols ||
                                 static_cast<int>(y_tls.size()) != threads;
        if (need_resize) {
            y_tls.assign(static_cast<std::size_t>(threads), Eigen::VectorXd::Zero(num_groups()));
            if (cols > 0) {
                x_tls.assign(static_cast<std::size_t>(threads),
                             Eigen::MatrixXd::Zero(num_groups(), cols));
            } else {
                x_tls.clear();
            }
            tls_threads = threads;
            tls_cols = cols;
        }

        const int groups = num_groups();
        // Heuristic: enable sparse TLS only when dense clearing/reduction is expensive.
        // For small thread counts, dense setZero() + dense reduction is typically faster.
        constexpr std::size_t kSparseTlsMinBytes = 64U * 1024U * 1024U;  // 64 MiB
        const std::size_t cols_entries =
            static_cast<std::size_t>(std::max(0, cols)) + 1U;  // y + X columns
        const std::size_t dense_bytes =
            static_cast<std::size_t>(threads) * static_cast<std::size_t>(groups) * cols_entries *
            sizeof(double);
        tls_sparse = (dense_bytes >= kSparseTlsMinBytes);
        if (tls_sparse) {
            const std::size_t needed =
                static_cast<std::size_t>(threads) * static_cast<std::size_t>(groups);
            if (tls_epoch_tags.size() != needed) {
                tls_epoch_tags.assign(needed, 0);
            }
            if (static_cast<int>(tls_touched_groups.size()) != threads) {
                tls_touched_groups.assign(static_cast<std::size_t>(threads), {});
            }
            for (auto& v : tls_touched_groups) {
                v.clear();
            }
            ++tls_epoch;
            if (tls_epoch == std::numeric_limits<int>::max()) {
                std::fill(tls_epoch_tags.begin(), tls_epoch_tags.end(), 0);
                tls_epoch = 1;
            }
            return;
        }

        for (auto& v : y_tls) {
            v.setZero();
        }
        if (cols > 0) {
            for (auto& m : x_tls) {
                m.setZero();
            }
        }
    }
#endif

    inline void touch(int g) {
        if (!sparse_reset) {
            return;
        }
        uint8_t& flag = touched_mask[static_cast<std::size_t>(g)];
        if (!flag) {
            flag = 1;
            touched_groups.push_back(g);
        }
    }
};

struct RecoveryWorkspace {
    Eigen::VectorXd sums;
    Eigen::VectorXd counts;
    Eigen::VectorXd values;

    RecoveryWorkspace() = default;
    RecoveryWorkspace(int groups)
        : sums(Eigen::VectorXd::Zero(groups)),
          counts(Eigen::VectorXd::Zero(groups)),
          values(Eigen::VectorXd::Zero(groups)) {}

    void resize(int groups) {
        if (sums.size() != groups) {
            sums = Eigen::VectorXd::Zero(groups);
            counts = Eigen::VectorXd::Zero(groups);
            values = Eigen::VectorXd::Zero(groups);
        } else {
            reset();
        }
    }

    void reset() {
        sums.setZero();
        counts.setZero();
        values.setZero();
    }
};

FeIndexer build_indexer(const Eigen::VectorXi& raw_ids) {
    FeIndexer idx;
    idx.group_ids.resize(raw_ids.size());
    if (raw_ids.size() == 0) {
        idx.num_groups = 0;
        idx.num_levels_present = 0;
        return idx;
    }

    int min_id = raw_ids(0);
    int max_id = raw_ids(0);
    for (int i = 1; i < raw_ids.size(); ++i) {
        const int v = raw_ids(i);
        if (v < min_id) {
            min_id = v;
        } else if (v > max_id) {
            max_id = v;
        }
    }

    const long long range =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    constexpr long long kDenseMinDensityNumerator = 9;
    constexpr long long kDenseMinDensityDenominator = 10;

    // Dense fast path: range is compact enough to fit a [0, range) lookup table.
    // Works regardless of min_id sign because mapped = raw - min_id is in [0, range).
    if (range > 0 && range <= kDenseRangeCap &&
        range <= static_cast<long long>(raw_ids.size()) * 2LL) {
        const int groups = static_cast<int>(range);
        idx.num_groups = groups;
        std::vector<uint8_t> seen(static_cast<std::size_t>(groups), 0);
        int unique = 0;
        for (int i = 0; i < raw_ids.size(); ++i) {
            const int mapped = raw_ids(i) - min_id;
            idx.group_ids[static_cast<std::size_t>(i)] = mapped;
            uint8_t& flag = seen[static_cast<std::size_t>(mapped)];
            if (!flag) {
                flag = 1;
                ++unique;
            }
        }
        idx.num_levels_present = unique;
        if (static_cast<long long>(unique) * kDenseMinDensityDenominator <
            range * kDenseMinDensityNumerator) {
            std::vector<int> remap(static_cast<std::size_t>(groups), -1);
            int next_id = 0;
            for (int g = 0; g < groups; ++g) {
                if (seen[static_cast<std::size_t>(g)]) {
                    remap[static_cast<std::size_t>(g)] = next_id++;
                }
            }
            for (int i = 0; i < raw_ids.size(); ++i) {
                idx.group_ids[static_cast<std::size_t>(i)] =
                    remap[static_cast<std::size_t>(idx.group_ids[static_cast<std::size_t>(i)])];
            }
            idx.num_groups = next_id;
            idx.num_levels_present = next_id;
        }
        finalize_contiguous_groups(idx);
        return idx;
    }

    // Sparse-range fallback: sort (index, key) pairs so we can assign normalized
    // IDs in a single sequential pass. O(n log n) with much better cache behavior
    // than an unordered_map when n is large. Preserves semantics: each unique key
    // gets a unique normalized ID in [0, num_groups); the absolute mapping differs
    // but downstream code only relies on uniqueness within an indexer.
    const std::size_t n_raw = static_cast<std::size_t>(raw_ids.size());
    std::vector<std::pair<int, int>> keyed(n_raw);
    for (std::size_t i = 0; i < n_raw; ++i) {
        keyed[i] = {raw_ids(static_cast<Eigen::Index>(i)), static_cast<int>(i)};
    }
    std::sort(keyed.begin(), keyed.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first < b.first;
              });

    int next_id = -1;
    int prev_key = 0;
    for (std::size_t k = 0; k < n_raw; ++k) {
        const int key = keyed[k].first;
        const int pos = keyed[k].second;
        if (next_id < 0 || key != prev_key) {
            ++next_id;
            prev_key = key;
        }
        idx.group_ids[static_cast<std::size_t>(pos)] = next_id;
    }
    const int num_groups_final = next_id + 1;
    idx.num_groups = num_groups_final;
    idx.num_levels_present = num_groups_final;
    finalize_contiguous_groups(idx);
    return idx;
}

Eigen::VectorXd compute_weight_sums(const FeIndexer& idx,
                                    const double* weight_ptr,
                                    bool unit_weights,
                                    int n,
                                    int threads = 1) {
    if (indexer_has_contiguous_groups(idx, n)) {
        Eigen::VectorXd sums(idx.num_groups);
        const int* group_ptr = idx.group_ptr.data();
        if (unit_weights) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
#endif
            for (int g = 0; g < idx.num_groups; ++g) {
                sums[g] = static_cast<double>(group_ptr[g + 1] - group_ptr[g]);
            }
            return sums;
        }
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
#endif
        for (int g = 0; g < idx.num_groups; ++g) {
            const int start = group_ptr[g];
            const int len = group_ptr[g + 1] - start;
            // Eigen::Map + .sum() gives SIMD vectorization and a tree-reduction
            // that is both faster and more numerically stable than a naive
            // left-fold loop over weight_ptr.
            sums[g] = len > 0
                        ? Eigen::Map<const Eigen::VectorXd>(weight_ptr + start, len).sum()
                        : 0.0;
        }
        return sums;
    }

    Eigen::VectorXd sums = Eigen::VectorXd::Zero(idx.num_groups);
#ifdef HDFE_USE_OPENMP
    if (threads > 1) {
        std::vector<Eigen::VectorXd> tls(static_cast<std::size_t>(threads),
                                         Eigen::VectorXd::Zero(idx.num_groups));
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::VectorXd& local = tls[static_cast<std::size_t>(tid)];
            const int* gid = idx.group_ids.data();
            if (unit_weights) {
#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    local[gid[i]] += 1.0;
                }
            } else {
#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    local[gid[i]] += weight_ptr[i];
                }
            }
        }
        for (const auto& local : tls) {
            sums.noalias() += local;
        }
        return sums;
    }
#endif
    if (unit_weights) {
        for (int i = 0; i < n; ++i) {
            const int g = idx.group_ids[static_cast<std::size_t>(i)];
            sums[g] += 1.0;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            const int g = idx.group_ids[static_cast<std::size_t>(i)];
            sums[g] += weight_ptr[i];
        }
    }
    return sums;
}

Eigen::VectorXd compute_weight_sums_from_groups(const int* group_ids,
                                                int n,
                                                int num_groups,
                                                const double* weight_ptr,
                                                bool unit_weights,
                                                int threads = 1) {
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(num_groups);
#ifdef HDFE_USE_OPENMP
    if (threads > 1) {
        std::vector<Eigen::VectorXd> tls(static_cast<std::size_t>(threads),
                                         Eigen::VectorXd::Zero(num_groups));
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::VectorXd& local = tls[static_cast<std::size_t>(tid)];
            if (unit_weights) {
#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    local[group_ids[i]] += 1.0;
                }
            } else {
#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    local[group_ids[i]] += weight_ptr[i];
                }
            }
        }
        for (const auto& local : tls) {
            sums.noalias() += local;
        }
        return sums;
    }
#endif
    if (unit_weights) {
        for (int i = 0; i < n; ++i) {
            sums[group_ids[i]] += 1.0;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            sums[group_ids[i]] += weight_ptr[i];
        }
    }
    return sums;
}

struct GroupCSR {
    std::vector<int> group_ptr;
    std::vector<int> obs_index;
};

GroupCSR build_group_csr(const int* group_ids, int n, int num_groups) {
    GroupCSR csr;
    csr.group_ptr.assign(static_cast<std::size_t>(num_groups) + 1U, 0);
    for (int i = 0; i < n; ++i) {
        const int g = group_ids[i];
        if (g < 0 || g >= num_groups) {
            throw std::runtime_error("Fixed-effect group ids must be in [0, num_groups)");
        }
        ++csr.group_ptr[static_cast<std::size_t>(g) + 1U];
    }
    for (int g = 0; g < num_groups; ++g) {
        csr.group_ptr[static_cast<std::size_t>(g) + 1U] +=
            csr.group_ptr[static_cast<std::size_t>(g)];
    }
    csr.obs_index.resize(static_cast<std::size_t>(n));
    std::vector<int> cursor = csr.group_ptr;
    for (int i = 0; i < n; ++i) {
        const int g = group_ids[i];
        const int pos = cursor[static_cast<std::size_t>(g)]++;
        csr.obs_index[static_cast<std::size_t>(pos)] = i;
    }
    return csr;
}

double demean_y_only_groups(Eigen::VectorXd& y,
                            const int* group_ids,
                            FeWorkspace& workspace,
                            const double* weight_ptr,
                            bool unit_weights,
                            int threads,
                            bool subtract,
                            double alpha_scale,
                            double* alpha_y,
                            double* out_max_abs) {
    const int n = static_cast<int>(y.size());
    const int groups = workspace.num_groups();
    double* y_ptr = y.data();

    int local_threads = std::max(1, threads);
#ifndef HDFE_USE_OPENMP
    local_threads = 1;
#else
    workspace.prepare_tls(local_threads, 0);
#endif

    workspace.reset();
    const double* weight_sums_inv = workspace.weight_sums_inv.data();

#ifdef HDFE_USE_OPENMP
    if (local_threads > 1) {
        if (!workspace.tls_sparse) {
            if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        y_sum[group_ids[i]] += y_ptr[i];
                    }
                }
            } else {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = group_ids[i];
                        y_sum[g] += weight_ptr[i] * y_ptr[i];
                    }
                }
            }
            for (const auto& local : workspace.y_tls) {
                workspace.y_sums.noalias() += local;
            }
        } else {
            const int epoch = workspace.tls_epoch;
            const int groups = workspace.num_groups();
            if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    int* tags = workspace.tls_epoch_tags.data() +
                                static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                    std::vector<int>& touched =
                        workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = group_ids[i];
                        if (tags[g] != epoch) {
                            tags[g] = epoch;
                            touched.push_back(g);
                            y_sum[g] = 0.0;
                        }
                        y_sum[g] += y_ptr[i];
                    }
                }
            } else {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    int* tags = workspace.tls_epoch_tags.data() +
                                static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                    std::vector<int>& touched =
                        workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = group_ids[i];
                        if (tags[g] != epoch) {
                            tags[g] = epoch;
                            touched.push_back(g);
                            y_sum[g] = 0.0;
                        }
                        y_sum[g] += weight_ptr[i] * y_ptr[i];
                    }
                }
            }
            for (int tid = 0; tid < local_threads; ++tid) {
                const auto& touched =
                    workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
                const Eigen::VectorXd& local = workspace.y_tls[static_cast<std::size_t>(tid)];
                for (int idx = 0; idx < static_cast<int>(touched.size()); ++idx) {
                    const int g = touched[static_cast<std::size_t>(idx)];
                    workspace.y_sums[g] += local[g];
                }
            }
        }
    } else
#endif
    {
        double* y_sums = workspace.y_sums.data();
        if (unit_weights) {
            for (int i = 0; i < n; ++i) {
                const int g = group_ids[i];
                workspace.touch(g);
                y_sums[g] += y_ptr[i];
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const int g = group_ids[i];
                const double w = weight_ptr[i];
                workspace.touch(g);
                y_sums[g] += w * y_ptr[i];
            }
        }
    }

    double* y_sums = workspace.y_sums.data();
    const bool track_max = (out_max_abs != nullptr);
    double local_max = 0.0;
    if (workspace.sparse_reset && !workspace.touched_groups.empty()) {
        for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size()); ++idx) {
            const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
            const double inv = weight_sums_inv[g];
            y_sums[g] *= inv;
            if (alpha_y) {
                alpha_y[g] += alpha_scale * y_sums[g];
            }
            if (track_max) {
                const double abs_val = std::abs(y_sums[g]);
                if (abs_val > local_max) {
                    local_max = abs_val;
                }
            }
        }
    } else {
#ifdef HDFE_USE_OPENMP
#pragma omp simd
#endif
        for (int g = 0; g < groups; ++g) {
            const double inv = weight_sums_inv[g];
            y_sums[g] *= inv;
            if (alpha_y) {
                alpha_y[g] += alpha_scale * y_sums[g];
            }
            if (track_max) {
                const double abs_val = std::abs(y_sums[g]);
                if (abs_val > local_max) {
                    local_max = abs_val;
                }
            }
        }
    }

    if (track_max) {
        *out_max_abs = local_max;
    }

    if (!subtract) {
        return track_max ? local_max : 0.0;
    }

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
    for (int i = 0; i < n; ++i) {
        const int g = group_ids[i];
        y_ptr[i] -= y_sums[g];
    }

    return track_max ? local_max : 0.0;
}

double demean_y_only_csr(Eigen::VectorXd& y,
                         const GroupCSR& csr,
                         const double* weight_ptr,
                         bool unit_weights,
                         const double* weight_sums_inv,
                         bool subtract,
                         double alpha_scale,
                         double* alpha_y,
                         double* out_max_abs,
                         int threads) {
    const int groups = static_cast<int>(csr.group_ptr.size()) - 1;
    double* y_ptr = y.data();
    const int* obs_index = csr.obs_index.data();
    const int* group_ptr = csr.group_ptr.data();
    double max_abs = 0.0;
    const bool track_max = (out_max_abs != nullptr);

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads)) reduction(max : max_abs)
#endif
    for (int g = 0; g < groups; ++g) {
        double sum = 0.0;
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        if (unit_weights) {
            for (int idx = start; idx < end; ++idx) {
                const int i = obs_index[idx];
                sum += y_ptr[i];
            }
        } else {
            for (int idx = start; idx < end; ++idx) {
                const int i = obs_index[idx];
                sum += weight_ptr[i] * y_ptr[i];
            }
        }
        const double mean = sum * weight_sums_inv[g];
        if (alpha_y) {
            alpha_y[g] += alpha_scale * mean;
        }
        if (track_max) {
            const double abs_val = std::abs(mean);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        if (subtract) {
            for (int idx = start; idx < end; ++idx) {
                const int i = obs_index[idx];
                y_ptr[i] -= mean;
            }
        }
    }

    if (track_max) {
        *out_max_abs = max_abs;
    }
    return track_max ? max_abs : 0.0;
}

void demean_inplace_csr_smallcols(Eigen::VectorXd& y,
                                  Eigen::MatrixXd& X,
                                  const GroupCSR& csr,
                                  FeWorkspace& workspace,
                                  const double* weight_ptr,
                                  bool unit_weights,
                                  int threads,
                                  bool subtract = true,
                                  double* out_sum_squares = nullptr,
                                  double alpha_scale = 1.0,
                                  double* alpha_y = nullptr,
                                  double* alpha_X = nullptr,
                                  double* out_max_abs = nullptr) {
    const int cols = static_cast<int>(X.cols());
    if (cols < 0 || cols > 4) {
        throw std::runtime_error("CSR demeaning kernel only supports 0-4 columns");
    }

    const int groups = workspace.num_groups();
    const int rows = static_cast<int>(X.rows());
    double* y_ptr = y.data();
    double* x_ptr = X.data();
    double* y_sums = workspace.y_sums.data();
    double* x_sums = workspace.x_sums.data();
    const double* weight_sums_inv = workspace.weight_sums_inv.data();
    const int* obs_index = csr.obs_index.data();
    const int* group_ptr = csr.group_ptr.data();
    const int local_threads = std::max(1, threads);
    const bool track_max = (out_max_abs != nullptr);
    double max_abs = 0.0;

    // Software prefetch distance for the indirect (gather/scatter) accesses.
    // This kernel is only used for non-contiguous fixed effects (CSR path),
    // where obs_index[idx] dereferences y/X at effectively random rows and the
    // working set (>L3) makes the loop DRAM-bandwidth/latency bound. Because
    // obs_index is contiguous and partitioned into contiguous group ranges per
    // thread, prefetching obs_index[idx + kPfDist] (valid whenever the linear
    // position is < rows) issues the next cache lines ahead of use. It only
    // touches addresses that real observations occupy, so it is a pure
    // performance hint with zero effect on the computed result.
    constexpr int kPfDist = 24;
    (void)kPfDist;

    double* x0 = cols > 0 ? x_ptr : nullptr;
    double* x1 = cols > 1 ? (x_ptr + static_cast<Eigen::Index>(rows)) : nullptr;
    double* x2 = cols > 2 ? (x_ptr + static_cast<Eigen::Index>(2) * rows) : nullptr;
    double* x3 = cols > 3 ? (x_ptr + static_cast<Eigen::Index>(3) * rows) : nullptr;

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(max : max_abs)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        double sum_y = 0.0;
        double sum_x0 = 0.0;
        double sum_x1 = 0.0;
        double sum_x2 = 0.0;
        double sum_x3 = 0.0;

        if (unit_weights) {
            switch (cols) {
                case 0:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        sum_y += y_ptr[i];
                    }
                    break;
                case 1:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        sum_y += y_ptr[i];
                        sum_x0 += x0[i];
                    }
                    break;
                case 2:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        sum_y += y_ptr[i];
                        sum_x0 += x0[i];
                        sum_x1 += x1[i];
                    }
                    break;
                case 3:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        sum_y += y_ptr[i];
                        sum_x0 += x0[i];
                        sum_x1 += x1[i];
                        sum_x2 += x2[i];
                    }
                    break;
                case 4:
                    for (int idx = start; idx < end; ++idx) {
                        const int pf = idx + kPfDist;
                        if (pf < rows) {
                            const int ip = obs_index[pf];
                            __builtin_prefetch(y_ptr + ip, 0, 1);
                            __builtin_prefetch(x0 + ip, 0, 1);
                            __builtin_prefetch(x1 + ip, 0, 1);
                            __builtin_prefetch(x2 + ip, 0, 1);
                            __builtin_prefetch(x3 + ip, 0, 1);
                        }
                        const int i = obs_index[idx];
                        sum_y += y_ptr[i];
                        sum_x0 += x0[i];
                        sum_x1 += x1[i];
                        sum_x2 += x2[i];
                        sum_x3 += x3[i];
                    }
                    break;
            }
        } else {
            switch (cols) {
                case 0:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        const double w = weight_ptr[i];
                        sum_y += w * y_ptr[i];
                    }
                    break;
                case 1:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        const double w = weight_ptr[i];
                        sum_y += w * y_ptr[i];
                        sum_x0 += w * x0[i];
                    }
                    break;
                case 2:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        const double w = weight_ptr[i];
                        sum_y += w * y_ptr[i];
                        sum_x0 += w * x0[i];
                        sum_x1 += w * x1[i];
                    }
                    break;
                case 3:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        const double w = weight_ptr[i];
                        sum_y += w * y_ptr[i];
                        sum_x0 += w * x0[i];
                        sum_x1 += w * x1[i];
                        sum_x2 += w * x2[i];
                    }
                    break;
                case 4:
                    for (int idx = start; idx < end; ++idx) {
                        const int pf = idx + kPfDist;
                        if (pf < rows) {
                            const int ip = obs_index[pf];
                            __builtin_prefetch(y_ptr + ip, 0, 1);
                            __builtin_prefetch(x0 + ip, 0, 1);
                            __builtin_prefetch(x1 + ip, 0, 1);
                            __builtin_prefetch(x2 + ip, 0, 1);
                            __builtin_prefetch(x3 + ip, 0, 1);
                            __builtin_prefetch(weight_ptr + ip, 0, 1);
                        }
                        const int i = obs_index[idx];
                        const double w = weight_ptr[i];
                        sum_y += w * y_ptr[i];
                        sum_x0 += w * x0[i];
                        sum_x1 += w * x1[i];
                        sum_x2 += w * x2[i];
                        sum_x3 += w * x3[i];
                    }
                    break;
            }
        }

        const double inv = weight_sums_inv[g];
        const double mean_y = sum_y * inv;
        y_sums[g] = mean_y;
        if (alpha_y) {
            alpha_y[g] += alpha_scale * mean_y;
        }
        if (track_max) {
            const double abs_val = std::abs(mean_y);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        switch (cols) {
            case 0:
                break;
            case 1: {
                const double mean_x0 = sum_x0 * inv;
                x_sums[g] = mean_x0;
                if (alpha_X) {
                    alpha_X[g] += alpha_scale * mean_x0;
                }
                break;
            }
            case 2: {
                const double mean_x0 = sum_x0 * inv;
                const double mean_x1 = sum_x1 * inv;
                x_sums[g] = mean_x0;
                x_sums[groups + g] = mean_x1;
                if (alpha_X) {
                    alpha_X[g] += alpha_scale * mean_x0;
                    alpha_X[groups + g] += alpha_scale * mean_x1;
                }
                break;
            }
            case 3: {
                const double mean_x0 = sum_x0 * inv;
                const double mean_x1 = sum_x1 * inv;
                const double mean_x2 = sum_x2 * inv;
                x_sums[g] = mean_x0;
                x_sums[groups + g] = mean_x1;
                x_sums[2 * groups + g] = mean_x2;
                if (alpha_X) {
                    alpha_X[g] += alpha_scale * mean_x0;
                    alpha_X[groups + g] += alpha_scale * mean_x1;
                    alpha_X[2 * groups + g] += alpha_scale * mean_x2;
                }
                break;
            }
            case 4: {
                const double mean_x0 = sum_x0 * inv;
                const double mean_x1 = sum_x1 * inv;
                const double mean_x2 = sum_x2 * inv;
                const double mean_x3 = sum_x3 * inv;
                x_sums[g] = mean_x0;
                x_sums[groups + g] = mean_x1;
                x_sums[2 * groups + g] = mean_x2;
                x_sums[3 * groups + g] = mean_x3;
                if (alpha_X) {
                    alpha_X[g] += alpha_scale * mean_x0;
                    alpha_X[groups + g] += alpha_scale * mean_x1;
                    alpha_X[2 * groups + g] += alpha_scale * mean_x2;
                    alpha_X[3 * groups + g] += alpha_scale * mean_x3;
                }
                break;
            }
        }
    }

    if (track_max) {
        *out_max_abs = max_abs;
    }

    if (!subtract) {
        return;
    }

    if (!out_sum_squares) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double mean_y = y_sums[g];
            switch (cols) {
                case 0:
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        y_ptr[i] -= mean_y;
                    }
                    break;
                case 1: {
                    const double mean_x0 = x_sums[g];
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        y_ptr[i] -= mean_y;
                        x0[i] -= mean_x0;
                    }
                    break;
                }
                case 2: {
                    const double mean_x0 = x_sums[g];
                    const double mean_x1 = x_sums[groups + g];
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        y_ptr[i] -= mean_y;
                        x0[i] -= mean_x0;
                        x1[i] -= mean_x1;
                    }
                    break;
                }
                case 3: {
                    const double mean_x0 = x_sums[g];
                    const double mean_x1 = x_sums[groups + g];
                    const double mean_x2 = x_sums[2 * groups + g];
                    for (int idx = start; idx < end; ++idx) {
                        const int i = obs_index[idx];
                        y_ptr[i] -= mean_y;
                        x0[i] -= mean_x0;
                        x1[i] -= mean_x1;
                        x2[i] -= mean_x2;
                    }
                    break;
                }
                case 4: {
                    const double mean_x0 = x_sums[g];
                    const double mean_x1 = x_sums[groups + g];
                    const double mean_x2 = x_sums[2 * groups + g];
                    const double mean_x3 = x_sums[3 * groups + g];
                    for (int idx = start; idx < end; ++idx) {
                        const int pf = idx + kPfDist;
                        if (pf < rows) {
                            const int ip = obs_index[pf];
                            __builtin_prefetch(y_ptr + ip, 1, 1);
                            __builtin_prefetch(x0 + ip, 1, 1);
                            __builtin_prefetch(x1 + ip, 1, 1);
                            __builtin_prefetch(x2 + ip, 1, 1);
                            __builtin_prefetch(x3 + ip, 1, 1);
                        }
                        const int i = obs_index[idx];
                        y_ptr[i] -= mean_y;
                        x0[i] -= mean_x0;
                        x1[i] -= mean_x1;
                        x2[i] -= mean_x2;
                        x3[i] -= mean_x3;
                    }
                    break;
                }
            }
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(+ : sumsq)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        const double mean_y = y_sums[g];
        switch (cols) {
            case 0:
                for (int idx = start; idx < end; ++idx) {
                    const int i = obs_index[idx];
                    y_ptr[i] -= mean_y;
                    const double yi = y_ptr[i];
                    sumsq += yi * yi;
                }
                break;
            case 1: {
                const double mean_x0 = x_sums[g];
                for (int idx = start; idx < end; ++idx) {
                    const int i = obs_index[idx];
                    y_ptr[i] -= mean_y;
                    const double yi = y_ptr[i];
                    sumsq += yi * yi;
                    x0[i] -= mean_x0;
                    const double x0v = x0[i];
                    sumsq += x0v * x0v;
                }
                break;
            }
            case 2: {
                const double mean_x0 = x_sums[g];
                const double mean_x1 = x_sums[groups + g];
                for (int idx = start; idx < end; ++idx) {
                    const int i = obs_index[idx];
                    y_ptr[i] -= mean_y;
                    const double yi = y_ptr[i];
                    sumsq += yi * yi;
                    x0[i] -= mean_x0;
                    const double x0v = x0[i];
                    sumsq += x0v * x0v;
                    x1[i] -= mean_x1;
                    const double x1v = x1[i];
                    sumsq += x1v * x1v;
                }
                break;
            }
            case 3: {
                const double mean_x0 = x_sums[g];
                const double mean_x1 = x_sums[groups + g];
                const double mean_x2 = x_sums[2 * groups + g];
                for (int idx = start; idx < end; ++idx) {
                    const int i = obs_index[idx];
                    y_ptr[i] -= mean_y;
                    const double yi = y_ptr[i];
                    sumsq += yi * yi;
                    x0[i] -= mean_x0;
                    const double x0v = x0[i];
                    sumsq += x0v * x0v;
                    x1[i] -= mean_x1;
                    const double x1v = x1[i];
                    sumsq += x1v * x1v;
                    x2[i] -= mean_x2;
                    const double x2v = x2[i];
                    sumsq += x2v * x2v;
                }
                break;
            }
            case 4: {
                const double mean_x0 = x_sums[g];
                const double mean_x1 = x_sums[groups + g];
                const double mean_x2 = x_sums[2 * groups + g];
                const double mean_x3 = x_sums[3 * groups + g];
                for (int idx = start; idx < end; ++idx) {
                    const int pf = idx + kPfDist;
                    if (pf < rows) {
                        const int ip = obs_index[pf];
                        __builtin_prefetch(y_ptr + ip, 1, 1);
                        __builtin_prefetch(x0 + ip, 1, 1);
                        __builtin_prefetch(x1 + ip, 1, 1);
                        __builtin_prefetch(x2 + ip, 1, 1);
                        __builtin_prefetch(x3 + ip, 1, 1);
                    }
                    const int i = obs_index[idx];
                    y_ptr[i] -= mean_y;
                    const double yi = y_ptr[i];
                    sumsq += yi * yi;
                    x0[i] -= mean_x0;
                    const double x0v = x0[i];
                    sumsq += x0v * x0v;
                    x1[i] -= mean_x1;
                    const double x1v = x1[i];
                    sumsq += x1v * x1v;
                    x2[i] -= mean_x2;
                    const double x2v = x2[i];
                    sumsq += x2v * x2v;
                    x3[i] -= mean_x3;
                    const double x3v = x3[i];
                    sumsq += x3v * x3v;
                }
                break;
            }
        }
    }
    *out_sum_squares = sumsq;
}

bool should_use_reference_sparse_absorber(const std::vector<FeIndexer>& indexers,
                                          int n,
                                          int total_levels,
                                          const Eigen::VectorXd* weights,
                                          const HdfeOptions& options) {
    if (weights != nullptr || options.retain_fixed_effects || indexers.size() != 2) {
        return false;
    }
    if (n <= 0 || n > 20000 || total_levels <= 0) {
        return false;
    }
    // Surgical path for small, path-like two-way FE graphs such as Sergio's
    // synthetic-zigzag. Other Sergio datasets keep the existing MAP route.
    // Integration note (10jun2026): the feature branch disabled this exact
    // sparse route citing a normalization-target difference vs reghdfe's MAP
    // path on nearly saturated graphs. Disabling it regressed the zigzag
    // class (1 -> 5001 CPU / 36481 CUDA iterations, SEs 1e-13 -> 7.2/76.5,
    // CPU/CUDA divergence), so the route is restored; the pathlike-CG
    // accelerator remains as the fallback when the sparse solve declines or
    // fails to converge. The normalization question is tracked for follow-up.
    return total_levels >= n - 2 && total_levels <= n + 2;
}

bool should_use_pathlike_cg_accelerator(const std::vector<FeIndexer>& indexers,
                                        int n,
                                        int total_levels,
                                        const Eigen::VectorXd* weights,
                                        const HdfeOptions& options) {
    if (options.retain_fixed_effects || indexers.size() != 2) {
        return false;
    }
    if (n <= 0 || n > 20000 || total_levels <= 0) {
        return false;
    }
    if (weights != nullptr) {
        for (Eigen::Index i = 0; i < weights->size(); ++i) {
            if (std::abs((*weights)[i] - 1.0) > 1e-12) {
                return false;
            }
        }
    }
    return total_levels >= n - 2 && total_levels <= n + 2;
}

bool reghdfe_cg_env_disabled();

bool should_use_reghdfe_cg_accelerator(const std::vector<FeIndexer>& indexers,
                                       const Eigen::VectorXd* weights,
                                       const HdfeOptions& options) {
    // Only an EXPLICIT reghdfe/both criterion opts into the CG accelerator;
    // the Auto default follows tolerance_mode through the standard path.
    if (options.convergence_criterion == ConvergenceCriterion::Auto ||
        options.convergence_criterion == ConvergenceCriterion::NormChange) {
        return false;
    }
    if (options.retain_fixed_effects || indexers.size() < 2) {
        return false;
    }
    if (reghdfe_cg_env_disabled()) {
        return false;
    }
    if (weights != nullptr) {
        for (Eigen::Index i = 0; i < weights->size(); ++i) {
            if (std::abs((*weights)[i] - 1.0) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

std::vector<int> build_reference_sparse_columns(const std::vector<FeIndexer>& indexers,
                                                const std::vector<int>& offsets,
                                                int n,
                                                int& kept_cols) {
    const int dims = static_cast<int>(indexers.size());
    const int total_levels = offsets.back();
    std::vector<int> column_for_level(static_cast<std::size_t>(total_levels), -1);
    kept_cols = 0;
    if (dims <= 0 || total_levels <= 0) {
        return column_for_level;
    }
    if (dims == 1) {
        for (int g = 0; g < total_levels; ++g) {
            column_for_level[static_cast<std::size_t>(g)] = kept_cols++;
        }
        return column_for_level;
    }

    ExactSparseUnionFind uf(total_levels);
    for (int i = 0; i < n; ++i) {
        const int anchor = offsets[0] + indexers[0].group_ids[static_cast<std::size_t>(i)];
        for (int d = 1; d < dims; ++d) {
            const int node = offsets[static_cast<std::size_t>(d)] +
                             indexers[static_cast<std::size_t>(d)]
                                 .group_ids[static_cast<std::size_t>(i)];
            uf.unite(anchor, node);
        }
    }

    std::vector<uint8_t> drop(static_cast<std::size_t>(total_levels), 0);
    for (int d = 1; d < dims; ++d) {
        std::unordered_map<int, int> reference_by_component;
        reference_by_component.reserve(
            static_cast<std::size_t>(indexers[static_cast<std::size_t>(d)].num_groups));
        const int offset = offsets[static_cast<std::size_t>(d)];
        const int groups = indexers[static_cast<std::size_t>(d)].num_groups;
        for (int g = 0; g < groups; ++g) {
            const int global = offset + g;
            const int root = uf.find(global);
            auto inserted = reference_by_component.emplace(root, global);
            if (inserted.second) {
                drop[static_cast<std::size_t>(global)] = 1;
            }
        }
    }

    for (int global = 0; global < total_levels; ++global) {
        if (!drop[static_cast<std::size_t>(global)]) {
            column_for_level[static_cast<std::size_t>(global)] = kept_cols++;
        }
    }
    return column_for_level;
}

AbsorptionResult absorb_fixed_effects_sparse(const Eigen::VectorXd& y,
                                             const Eigen::MatrixXd& X,
                                             const std::vector<FeIndexer>& indexers,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options,
                                             bool reference_gauge) {
    AbsorptionResult result;
    result.y_tilde = y;
    result.X_tilde = X;
    result.converged = false;

    const int n = static_cast<int>(y.size());
    const int dims = static_cast<int>(indexers.size());
    if (dims == 0) {
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    std::vector<int> offsets(static_cast<std::size_t>(dims) + 1, 0);
    for (int d = 0; d < dims; ++d) {
        offsets[static_cast<std::size_t>(d + 1)] =
            offsets[static_cast<std::size_t>(d)] + indexers[static_cast<std::size_t>(d)].num_groups;
        result.fe_levels.push_back(indexers[static_cast<std::size_t>(d)].num_levels_present);
    }
    const int total_levels = offsets.back();
    int solver_levels = total_levels;
    std::vector<int> column_for_level;
    if (reference_gauge) {
        column_for_level =
            build_reference_sparse_columns(indexers, offsets, n, solver_levels);
        if (solver_levels <= 0) {
            result.converged = false;
            return result;
        }
    } else {
        column_for_level.resize(static_cast<std::size_t>(total_levels));
        for (int g = 0; g < total_levels; ++g) {
            column_for_level[static_cast<std::size_t>(g)] = g;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    const std::size_t nnz =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(dims);
    triplets.reserve(nnz);

    const bool use_weights = (weights != nullptr);
    const double* w_ptr = use_weights ? weights->data() : nullptr;

    for (int d = 0; d < dims; ++d) {
        const int offset = offsets[static_cast<std::size_t>(d)];
        const int* gid = indexers[static_cast<std::size_t>(d)].group_ids.data();
        if (use_weights) {
            for (int i = 0; i < n; ++i) {
                const int col =
                    column_for_level[static_cast<std::size_t>(offset + gid[i])];
                if (col >= 0) {
                    triplets.emplace_back(i, col, std::sqrt(w_ptr[i]));
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const int col =
                    column_for_level[static_cast<std::size_t>(offset + gid[i])];
                if (col >= 0) {
                    triplets.emplace_back(i, col, 1.0);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> D(n, solver_levels);
    D.setFromTriplets(triplets.begin(), triplets.end());

    const Eigen::VectorXd* y_used = &y;
    const Eigen::MatrixXd* X_used = &X;
    Eigen::VectorXd y_w;
    Eigen::MatrixXd X_w;
    if (use_weights) {
        y_w = y;
        X_w = X;
        const Eigen::VectorXd sqrt_w = weights->array().sqrt();
        y_w.array() *= sqrt_w.array();
        for (int j = 0; j < X_w.cols(); ++j) {
            X_w.col(j).array() *= sqrt_w.array();
        }
        y_used = &y_w;
        X_used = &X_w;
    }

    Eigen::SparseMatrix<double> A = D.transpose() * D;
    if (!reference_gauge) {
        const double lambda = 1e-11;
        for (int i = 0; i < solver_levels; ++i) {
            A.coeffRef(i, i) += lambda;
        }
    }
    A.makeCompressed();

    bool solved = false;
    bool converged = true;
    Eigen::VectorXd alpha_y_solved;
    Eigen::MatrixXd alpha_X_solved;
    if (X.cols() > 0) {
        alpha_X_solved.resize(solver_levels, X.cols());
    }

    const bool try_direct = (solver_levels < 50000);
    if (try_direct) {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() == Eigen::Success) {
            const Eigen::VectorXd rhs_y = D.transpose() * (*y_used);
            alpha_y_solved = solver.solve(rhs_y);
            if (X.cols() > 0) {
                const Eigen::MatrixXd rhs_X = D.transpose() * (*X_used);
                alpha_X_solved = solver.solve(rhs_X);
            }
            solved = (solver.info() == Eigen::Success);
            converged = solved;
            if (solved) {
                result.iterations = 1;
            }
        } else {
            converged = false;
        }
    }

    if (!solved) {
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
                                 Eigen::IncompleteCholesky<double>>
            cg;
        cg.setTolerance(effective_absorption_tolerance(options));
        cg.setMaxIterations(options.max_iter);
        cg.compute(A);

        const Eigen::VectorXd rhs_y = D.transpose() * (*y_used);
        alpha_y_solved = cg.solve(rhs_y);
        int max_iters = static_cast<int>(cg.iterations());
        converged = (cg.info() == Eigen::Success);

        if (X.cols() > 0) {
            const Eigen::MatrixXd rhs_X = D.transpose() * (*X_used);
            for (int j = 0; j < X.cols(); ++j) {
                alpha_X_solved.col(j) = cg.solve(rhs_X.col(j));
                max_iters = std::max(max_iters, static_cast<int>(cg.iterations()));
                converged = converged && (cg.info() == Eigen::Success);
            }
        }
        result.iterations = max_iters;
        solved = converged;
    }

    if (!solved) {
        result.converged = false;
        return result;
    }

    Eigen::VectorXd alpha_y = Eigen::VectorXd::Zero(total_levels);
    Eigen::MatrixXd alpha_X;
    if (X.cols() > 0) {
        alpha_X = Eigen::MatrixXd::Zero(total_levels, X.cols());
    }
    for (int global = 0; global < total_levels; ++global) {
        const int col = column_for_level[static_cast<std::size_t>(global)];
        if (col >= 0) {
            alpha_y[global] = alpha_y_solved[col];
            if (X.cols() > 0) {
                alpha_X.row(global) = alpha_X_solved.row(col);
            }
        }
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    }
    threads = std::max(1, threads);
#endif
    threads = std::max(1, threads);

    const double* alpha_y_ptr = alpha_y.data();
    double* y_ptr = result.y_tilde.data();

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int i = 0; i < n; ++i) {
        double fe_sum = 0.0;
        for (int d = 0; d < dims; ++d) {
            const int gid = indexers[static_cast<std::size_t>(d)]
                                .group_ids[static_cast<std::size_t>(i)];
            fe_sum += alpha_y_ptr[offsets[static_cast<std::size_t>(d)] + gid];
        }
        y_ptr[i] -= fe_sum;
    }

    if (X.cols() > 0) {
        for (int j = 0; j < X.cols(); ++j) {
            const double* alpha_x_ptr = alpha_X.col(j).data();
            double* x_ptr = result.X_tilde.col(j).data();
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                double fe_sum = 0.0;
                for (int d = 0; d < dims; ++d) {
                    const int gid = indexers[static_cast<std::size_t>(d)]
                                        .group_ids[static_cast<std::size_t>(i)];
                    fe_sum += alpha_x_ptr[offsets[static_cast<std::size_t>(d)] + gid];
                }
                x_ptr[i] -= fe_sum;
            }
        }
    }

    result.converged = true;

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        result.fe_means.resize(indexers.size());
        result.fe_weight_sums.resize(indexers.size());
        for (int d = 0; d < dims; ++d) {
            result.fe_group_ids[static_cast<std::size_t>(d)] =
                indexers[static_cast<std::size_t>(d)].group_ids;
            result.fe_means[static_cast<std::size_t>(d)] =
                alpha_y.segment(offsets[static_cast<std::size_t>(d)],
                                indexers[static_cast<std::size_t>(d)].num_groups);
            result.fe_weight_sums[static_cast<std::size_t>(d)] =
                compute_weight_sums(indexers[static_cast<std::size_t>(d)],
                                    w_ptr, !use_weights, n, threads);
        }
    }

    return result;
}

void demean_inplace_contiguous(Eigen::VectorXd& y,
                               Eigen::MatrixXd& X,
                               const FeIndexer& idx,
                               FeWorkspace& workspace,
                               const double* weight_ptr,
                               bool unit_weights,
                               int threads,
                               bool subtract,
                               double* out_sum_squares,
                               double alpha_scale,
                               double* alpha_y,
                               double* alpha_X,
                               double* out_max_abs) {
    constexpr int kInlineCols = 8;

    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int rows = static_cast<int>(X.rows());
    const int groups = workspace.num_groups();
    const int* group_ptr = idx.group_ptr.data();
    const double* weight_sums_inv = workspace.weight_sums_inv.data();

    double* y_ptr = y.data();
    double* X_ptr = X.data();
    double* y_sums = workspace.y_sums.data();
    double* x_sums_ptr = cols > 0 ? workspace.x_sums.data() : nullptr;
    const bool track_max = (out_max_abs != nullptr);
    const int local_threads = std::max(1, threads);

    double* x_cols[kInlineCols] = {};
    for (int j = 0; j < cols; ++j) {
        x_cols[j] = X_ptr + static_cast<Eigen::Index>(j) * rows;
    }

    double local_max = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(max : local_max)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        const double inv = weight_sums_inv[g];

        double y_acc = 0.0;
        double x_acc[kInlineCols] = {};
        if (unit_weights) {
            for (int i = start; i < end; ++i) {
                y_acc += y_ptr[i];
                for (int j = 0; j < cols; ++j) {
                    x_acc[j] += x_cols[j][i];
                }
            }
        } else {
            for (int i = start; i < end; ++i) {
                const double w = weight_ptr[i];
                y_acc += w * y_ptr[i];
                for (int j = 0; j < cols; ++j) {
                    x_acc[j] += w * x_cols[j][i];
                }
            }
        }

        const double y_mean = y_acc * inv;
        y_sums[g] = y_mean;
        if (alpha_y) {
            alpha_y[g] += alpha_scale * y_mean;
        }
        if (track_max) {
            local_max = std::max(local_max, std::abs(y_mean));
        }

        for (int j = 0; j < cols; ++j) {
            const Eigen::Index offset = static_cast<Eigen::Index>(j) * groups + g;
            const double x_mean = x_acc[j] * inv;
            x_sums_ptr[offset] = x_mean;
            if (alpha_X) {
                alpha_X[offset] += alpha_scale * x_mean;
            }
        }
    }

    if (track_max) {
        *out_max_abs = local_max;
    }
    if (!subtract) {
        return;
    }

    if (!out_sum_squares) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double y_mean = y_sums[g];
            for (int i = start; i < end; ++i) {
                y_ptr[i] -= y_mean;
                for (int j = 0; j < cols; ++j) {
                    x_cols[j][i] -=
                        x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                }
            }
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for reduction(+ : sumsq) schedule(static) num_threads(local_threads)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        const double y_mean = y_sums[g];
        for (int i = start; i < end; ++i) {
            y_ptr[i] -= y_mean;
            const double yi = y_ptr[i];
            sumsq += yi * yi;
            for (int j = 0; j < cols; ++j) {
                x_cols[j][i] -=
                    x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                const double xj = x_cols[j][i];
                sumsq += xj * xj;
            }
        }
    }
    *out_sum_squares = sumsq;
}

void demean_inplace_csr(Eigen::VectorXd& y,
                        Eigen::MatrixXd& X,
                        const GroupCSR& csr,
                        FeWorkspace& workspace,
                        const double* weight_ptr,
                        bool unit_weights,
                        int threads,
                        bool subtract,
                        double* out_sum_squares,
                        double alpha_scale,
                        double* alpha_y,
                        double* alpha_X,
                        double* out_max_abs) {
    constexpr int kInlineCols = 8;

    const int cols = static_cast<int>(X.cols());
    if (cols <= 4) {
        demean_inplace_csr_smallcols(y, X, csr, workspace, weight_ptr, unit_weights, threads,
                                     subtract, out_sum_squares, alpha_scale, alpha_y, alpha_X,
                                     out_max_abs);
        return;
    }
    const int rows = static_cast<int>(X.rows());
    const int groups = workspace.num_groups();
    const int* group_ptr = csr.group_ptr.data();
    const int* obs_index = csr.obs_index.data();
    const double* weight_sums_inv = workspace.weight_sums_inv.data();
    const int local_threads = std::max(1, threads);

    double* y_ptr = y.data();
    double* X_ptr = X.data();
    double* y_sums = workspace.y_sums.data();
    double* x_sums_ptr = cols > 0 ? workspace.x_sums.data() : nullptr;
    const bool track_max = (out_max_abs != nullptr);

    double* x_cols_inline[kInlineCols];
    std::vector<double*> x_cols_storage;
    double** x_cols = nullptr;
    if (cols > 0) {
        if (cols <= kInlineCols) {
            x_cols = x_cols_inline;
            for (int j = 0; j < cols; ++j) {
                x_cols_inline[j] = X_ptr + static_cast<Eigen::Index>(j) * rows;
            }
        } else {
            x_cols_storage.resize(static_cast<std::size_t>(cols));
            for (int j = 0; j < cols; ++j) {
                x_cols_storage[static_cast<std::size_t>(j)] =
                    X_ptr + static_cast<Eigen::Index>(j) * rows;
            }
            x_cols = x_cols_storage.data();
        }
    }

    double local_max = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(max : local_max)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        const double inv = weight_sums_inv[g];

        double y_acc = 0.0;
        double x_acc[kInlineCols] = {};
        for (int idx = start; idx < end; ++idx) {
            const int i = obs_index[idx];
            if (unit_weights) {
                y_acc += y_ptr[i];
                for (int j = 0; j < cols; ++j) {
                    x_acc[j] += x_cols[j][i];
                }
            } else {
                const double w = weight_ptr[i];
                y_acc += w * y_ptr[i];
                for (int j = 0; j < cols; ++j) {
                    x_acc[j] += w * x_cols[j][i];
                }
            }
        }

        const double y_mean = y_acc * inv;
        y_sums[g] = y_mean;
        if (alpha_y) {
            alpha_y[g] += alpha_scale * y_mean;
        }
        if (track_max) {
            local_max = std::max(local_max, std::abs(y_mean));
        }

        for (int j = 0; j < cols; ++j) {
            const Eigen::Index offset = static_cast<Eigen::Index>(j) * groups + g;
            const double x_mean = x_acc[j] * inv;
            x_sums_ptr[offset] = x_mean;
            if (alpha_X) {
                alpha_X[offset] += alpha_scale * x_mean;
            }
        }
    }

    if (track_max) {
        *out_max_abs = local_max;
    }
    if (!subtract) {
        return;
    }

    if (!out_sum_squares) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double y_mean = y_sums[g];
            for (int idx = start; idx < end; ++idx) {
                const int i = obs_index[idx];
                y_ptr[i] -= y_mean;
                for (int j = 0; j < cols; ++j) {
                    x_cols[j][i] -=
                        x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                }
            }
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for reduction(+ : sumsq) schedule(static) num_threads(local_threads)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        const double y_mean = y_sums[g];
        for (int idx = start; idx < end; ++idx) {
            const int i = obs_index[idx];
            y_ptr[i] -= y_mean;
            const double yi = y_ptr[i];
            sumsq += yi * yi;
            for (int j = 0; j < cols; ++j) {
                x_cols[j][i] -=
                    x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                const double xj = x_cols[j][i];
                sumsq += xj * xj;
            }
        }
    }
    *out_sum_squares = sumsq;
}

void demean_inplace(Eigen::VectorXd& y,
                    Eigen::MatrixXd& X,
                    const FeIndexer& idx,
                    FeWorkspace& workspace,
                    const double* weight_ptr,
                    bool unit_weights,
                    int threads,
                    bool subtract = true,
                    double* out_sum_squares = nullptr,
                    double alpha_scale = 1.0,
                    double* alpha_y = nullptr,
                    double* alpha_X = nullptr,
                    double* out_max_abs = nullptr) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const bool accumulate = (alpha_y != nullptr || alpha_X != nullptr);
    if (cols <= 8 && indexer_has_contiguous_groups(idx, n)) {
        demean_inplace_contiguous(y, X, idx, workspace, weight_ptr, unit_weights, threads,
                                  subtract, out_sum_squares, alpha_scale,
                                  alpha_y, alpha_X, out_max_abs);
        return;
    }
    if (cols == 1) {
        const int x_rows = static_cast<int>(X.rows());
        const int groups = workspace.num_groups();
        double* y_ptr = y.data();
        double* x0 = X.data();
        const int* gid = idx.group_ids.data();

        int local_threads = std::max(1, threads);
#ifndef HDFE_USE_OPENMP
        local_threads = 1;
#else
        workspace.prepare_tls(local_threads, 1);
#endif

        workspace.reset();
        const double* weight_sums_inv = workspace.weight_sums_inv.data();

#ifdef HDFE_USE_OPENMP
        if (local_threads > 1) {
            if (!workspace.tls_sparse) {
                if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                    {
                        const int tid = omp_get_thread_num();
                        Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                        Eigen::MatrixXd& x_sum = workspace.x_tls[static_cast<std::size_t>(tid)];
                        double* xs0 = x_sum.data();
#pragma omp for schedule(static)
                        for (int i = 0; i < n; ++i) {
                            const int g = gid[i];
                            y_sum[g] += y_ptr[i];
                            xs0[g] += x0[i];
                        }
                    }
                } else {
#pragma omp parallel num_threads(local_threads)
                    {
                        const int tid = omp_get_thread_num();
                        Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                        Eigen::MatrixXd& x_sum = workspace.x_tls[static_cast<std::size_t>(tid)];
                        double* xs0 = x_sum.data();
#pragma omp for schedule(static)
                        for (int i = 0; i < n; ++i) {
                            const int g = gid[i];
                            const double w = weight_ptr[i];
                            y_sum[g] += w * y_ptr[i];
                            xs0[g] += w * x0[i];
                        }
                    }
                }
                for (const auto& local : workspace.y_tls) {
                    workspace.y_sums.noalias() += local;
                }
                for (const auto& local : workspace.x_tls) {
                    workspace.x_sums.noalias() += local;
                }
            } else {
                const int epoch = workspace.tls_epoch;
                if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                    {
                        const int tid = omp_get_thread_num();
                        Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                        Eigen::MatrixXd& x_sum = workspace.x_tls[static_cast<std::size_t>(tid)];
                        double* xs0 = x_sum.data();
                        int* tags =
                            workspace.tls_epoch_tags.data() +
                            static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                        std::vector<int>& touched =
                            workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                        for (int i = 0; i < n; ++i) {
                            const int g = gid[i];
                            if (tags[g] != epoch) {
                                tags[g] = epoch;
                                touched.push_back(g);
                                y_sum[g] = 0.0;
                                xs0[g] = 0.0;
                            }
                            y_sum[g] += y_ptr[i];
                            xs0[g] += x0[i];
                        }
                    }
                } else {
#pragma omp parallel num_threads(local_threads)
                    {
                        const int tid = omp_get_thread_num();
                        Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                        Eigen::MatrixXd& x_sum = workspace.x_tls[static_cast<std::size_t>(tid)];
                        double* xs0 = x_sum.data();
                        int* tags =
                            workspace.tls_epoch_tags.data() +
                            static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                        std::vector<int>& touched =
                            workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                        for (int i = 0; i < n; ++i) {
                            const int g = gid[i];
                            if (tags[g] != epoch) {
                                tags[g] = epoch;
                                touched.push_back(g);
                                y_sum[g] = 0.0;
                                xs0[g] = 0.0;
                            }
                            const double w = weight_ptr[i];
                            y_sum[g] += w * y_ptr[i];
                            xs0[g] += w * x0[i];
                        }
                    }
                }

                double* y_sums_out = workspace.y_sums.data();
                double* x_sums_out = workspace.x_sums.data();
                for (int tid = 0; tid < local_threads; ++tid) {
                    const auto& touched =
                        workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
                    const Eigen::VectorXd& y_local =
                        workspace.y_tls[static_cast<std::size_t>(tid)];
                    const double* x_local =
                        workspace.x_tls[static_cast<std::size_t>(tid)].data();
                    for (int idx = 0; idx < static_cast<int>(touched.size()); ++idx) {
                        const int g = touched[static_cast<std::size_t>(idx)];
                        y_sums_out[g] += y_local[g];
                        x_sums_out[g] += x_local[g];
                    }
                }
            }
        } else
#endif
        {
            double* y_sums = workspace.y_sums.data();
            double* x_sums = workspace.x_sums.data();
            if (unit_weights) {
                for (int i = 0; i < n; ++i) {
                    const int g = gid[i];
                    workspace.touch(g);
                    y_sums[g] += y_ptr[i];
                    x_sums[g] += x0[i];
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    const int g = gid[i];
                    workspace.touch(g);
                    const double w = weight_ptr[i];
                    y_sums[g] += w * y_ptr[i];
                    x_sums[g] += w * x0[i];
                }
            }
        }

        double* y_sums = workspace.y_sums.data();
        double* x_sums_ptr = workspace.x_sums.data();
        const bool track_max = (out_max_abs != nullptr);
        double local_max = 0.0;
        if (workspace.sparse_reset && !workspace.touched_groups.empty()) {
            if (!accumulate) {
                for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size());
                     ++idx) {
                    const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
                    const double inv = weight_sums_inv[g];
                    y_sums[g] *= inv;
                    if (track_max) {
                        const double abs_val = std::abs(y_sums[g]);
                        if (abs_val > local_max) {
                            local_max = abs_val;
                        }
                    }
                    x_sums_ptr[g] *= inv;
                }
            } else {
                for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size());
                     ++idx) {
                    const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
                    const double inv = weight_sums_inv[g];
                    y_sums[g] *= inv;
                    if (alpha_y) {
                        alpha_y[g] += alpha_scale * y_sums[g];
                    }
                    if (track_max) {
                        const double abs_val = std::abs(y_sums[g]);
                        if (abs_val > local_max) {
                            local_max = abs_val;
                        }
                    }
                    x_sums_ptr[g] *= inv;
                    if (alpha_X) {
                        alpha_X[g] += alpha_scale * x_sums_ptr[g];
                    }
                }
            }
        } else {
            if (!accumulate) {
#ifdef HDFE_USE_OPENMP
#pragma omp simd
#endif
                for (int g = 0; g < groups; ++g) {
                    const double inv = weight_sums_inv[g];
                    y_sums[g] *= inv;
                    if (track_max) {
                        const double abs_val = std::abs(y_sums[g]);
                        if (abs_val > local_max) {
                            local_max = abs_val;
                        }
                    }
                    x_sums_ptr[g] *= inv;
                }
            } else {
#ifdef HDFE_USE_OPENMP
#pragma omp simd
#endif
                for (int g = 0; g < groups; ++g) {
                    const double inv = weight_sums_inv[g];
                    y_sums[g] *= inv;
                    if (alpha_y) {
                        alpha_y[g] += alpha_scale * y_sums[g];
                    }
                    if (track_max) {
                        const double abs_val = std::abs(y_sums[g]);
                        if (abs_val > local_max) {
                            local_max = abs_val;
                        }
                    }
                    x_sums_ptr[g] *= inv;
                    if (alpha_X) {
                        alpha_X[g] += alpha_scale * x_sums_ptr[g];
                    }
                }
            }
        }

        if (track_max) {
            *out_max_abs = local_max;
        }

        if (!subtract) {
            return;
        }

#ifdef HDFE_USE_OPENMP
        if (!out_sum_squares) {
#pragma omp parallel for schedule(static) num_threads(local_threads)
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                y_ptr[i] -= y_sums[g];
                x0[i] -= x_sums_ptr[g];
            }
        } else {
            double sumsq = 0.0;
#pragma omp parallel for reduction(+ : sumsq) schedule(static) num_threads(local_threads)
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                y_ptr[i] -= y_sums[g];
                const double yi = y_ptr[i];
                sumsq += yi * yi;
                x0[i] -= x_sums_ptr[g];
                const double xi = x0[i];
                sumsq += xi * xi;
            }
            *out_sum_squares = sumsq;
        }
#else
        if (!out_sum_squares) {
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                y_ptr[i] -= y_sums[g];
                x0[i] -= x_sums_ptr[g];
            }
        } else {
            double sumsq = 0.0;
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                y_ptr[i] -= y_sums[g];
                const double yi = y_ptr[i];
                sumsq += yi * yi;
                x0[i] -= x_sums_ptr[g];
                const double xi = x0[i];
                sumsq += xi * xi;
            }
            *out_sum_squares = sumsq;
        }
#endif
        return;
    }
    const int x_rows = static_cast<int>(X.rows());
    const int groups = workspace.num_groups();

    double* y_ptr = y.data();
    double* X_ptr = X.data();

    int local_threads = std::max(1, threads);
#ifndef HDFE_USE_OPENMP
    local_threads = 1;
#else
    workspace.prepare_tls(local_threads, cols);
#endif

    workspace.reset();
    const double* weight_sums_inv = workspace.weight_sums_inv.data();
    const int* gid = idx.group_ids.data();

    constexpr int kInlineCols = 8;
    double* x_cols_inline[kInlineCols];
    double* x_sum_inline[kInlineCols];
    std::vector<double*> x_cols_storage;
    std::vector<double*> x_sum_storage;
    double** x_cols = nullptr;
    double** x_sum_cols = nullptr;
    if (cols > 0) {
        if (cols <= kInlineCols) {
            x_cols = x_cols_inline;
            x_sum_cols = x_sum_inline;
            for (int j = 0; j < cols; ++j) {
                x_cols_inline[j] = X_ptr + static_cast<Eigen::Index>(j) * x_rows;
                x_sum_inline[j] = workspace.x_sums.data() + static_cast<Eigen::Index>(j) * groups;
            }
        } else {
            x_cols_storage.resize(static_cast<std::size_t>(cols));
            x_sum_storage.resize(static_cast<std::size_t>(cols));
            for (int j = 0; j < cols; ++j) {
                x_cols_storage[static_cast<std::size_t>(j)] =
                    X_ptr + static_cast<Eigen::Index>(j) * x_rows;
                x_sum_storage[static_cast<std::size_t>(j)] =
                    workspace.x_sums.data() + static_cast<Eigen::Index>(j) * groups;
            }
            x_cols = x_cols_storage.data();
            x_sum_cols = x_sum_storage.data();
        }
    }

#ifdef HDFE_USE_OPENMP
    if (local_threads > 1) {
        if (!workspace.tls_sparse) {
            if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    Eigen::MatrixXd* x_sum =
                        cols > 0 ? &workspace.x_tls[static_cast<std::size_t>(tid)] : nullptr;
                    double* x_sum_ptr = cols > 0 ? x_sum->data() : nullptr;
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        y_sum[g] += y_ptr[i];
                        switch (cols) {
                            case 0:
                                break;
                            case 1:
                                x_sum_ptr[g] += x_cols[0][i];
                                break;
                            case 2:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                break;
                            case 3:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += x_cols[2][i];
                                break;
                            case 4:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += x_cols[2][i];
                                x_sum_ptr[3 * groups + g] += x_cols[3][i];
                                break;
                            default:
                                for (int j = 0; j < cols; ++j) {
                                    x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] +=
                                        x_cols[j][i];
                                }
                                break;
                        }
                    }
                }
            } else {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    Eigen::MatrixXd* x_sum =
                        cols > 0 ? &workspace.x_tls[static_cast<std::size_t>(tid)] : nullptr;
                    double* x_sum_ptr = cols > 0 ? x_sum->data() : nullptr;
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        const double w = weight_ptr[i];
                        y_sum[g] += w * y_ptr[i];
                        switch (cols) {
                            case 0:
                                break;
                            case 1:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                break;
                            case 2:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                break;
                            case 3:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += w * x_cols[2][i];
                                break;
                            case 4:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += w * x_cols[2][i];
                                x_sum_ptr[3 * groups + g] += w * x_cols[3][i];
                                break;
                            default:
                                for (int j = 0; j < cols; ++j) {
                                    x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] +=
                                        w * x_cols[j][i];
                                }
                                break;
                        }
                    }
                }
            }
            // Reduce
            for (const auto& local : workspace.y_tls) {
                workspace.y_sums.noalias() += local;
            }
            if (cols > 0) {
                for (const auto& local : workspace.x_tls) {
                    workspace.x_sums.noalias() += local;
                }
            }
        } else {
            const int epoch = workspace.tls_epoch;
            if (unit_weights) {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    Eigen::MatrixXd* x_sum =
                        cols > 0 ? &workspace.x_tls[static_cast<std::size_t>(tid)] : nullptr;
                    double* x_sum_ptr = cols > 0 ? x_sum->data() : nullptr;
                    int* tags =
                        workspace.tls_epoch_tags.data() +
                        static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                    std::vector<int>& touched =
                        workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        if (tags[g] != epoch) {
                            tags[g] = epoch;
                            touched.push_back(g);
                            y_sum[g] = 0.0;
                            switch (cols) {
                                case 0:
                                    break;
                                case 1:
                                    x_sum_ptr[g] = 0.0;
                                    break;
                                case 2:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    break;
                                case 3:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    x_sum_ptr[2 * groups + g] = 0.0;
                                    break;
                                case 4:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    x_sum_ptr[2 * groups + g] = 0.0;
                                    x_sum_ptr[3 * groups + g] = 0.0;
                                    break;
                                default:
                                    for (int j = 0; j < cols; ++j) {
                                        x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] =
                                            0.0;
                                    }
                                    break;
                            }
                        }
                        y_sum[g] += y_ptr[i];
                        switch (cols) {
                            case 0:
                                break;
                            case 1:
                                x_sum_ptr[g] += x_cols[0][i];
                                break;
                            case 2:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                break;
                            case 3:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += x_cols[2][i];
                                break;
                            case 4:
                                x_sum_ptr[g] += x_cols[0][i];
                                x_sum_ptr[groups + g] += x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += x_cols[2][i];
                                x_sum_ptr[3 * groups + g] += x_cols[3][i];
                                break;
                            default:
                                for (int j = 0; j < cols; ++j) {
                                    x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] +=
                                        x_cols[j][i];
                                }
                                break;
                        }
                    }
                }
            } else {
#pragma omp parallel num_threads(local_threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& y_sum = workspace.y_tls[static_cast<std::size_t>(tid)];
                    Eigen::MatrixXd* x_sum =
                        cols > 0 ? &workspace.x_tls[static_cast<std::size_t>(tid)] : nullptr;
                    double* x_sum_ptr = cols > 0 ? x_sum->data() : nullptr;
                    int* tags =
                        workspace.tls_epoch_tags.data() +
                        static_cast<std::size_t>(tid) * static_cast<std::size_t>(groups);
                    std::vector<int>& touched =
                        workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        if (tags[g] != epoch) {
                            tags[g] = epoch;
                            touched.push_back(g);
                            y_sum[g] = 0.0;
                            switch (cols) {
                                case 0:
                                    break;
                                case 1:
                                    x_sum_ptr[g] = 0.0;
                                    break;
                                case 2:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    break;
                                case 3:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    x_sum_ptr[2 * groups + g] = 0.0;
                                    break;
                                case 4:
                                    x_sum_ptr[g] = 0.0;
                                    x_sum_ptr[groups + g] = 0.0;
                                    x_sum_ptr[2 * groups + g] = 0.0;
                                    x_sum_ptr[3 * groups + g] = 0.0;
                                    break;
                                default:
                                    for (int j = 0; j < cols; ++j) {
                                        x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] =
                                            0.0;
                                    }
                                    break;
                            }
                        }
                        const double w = weight_ptr[i];
                        y_sum[g] += w * y_ptr[i];
                        switch (cols) {
                            case 0:
                                break;
                            case 1:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                break;
                            case 2:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                break;
                            case 3:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += w * x_cols[2][i];
                                break;
                            case 4:
                                x_sum_ptr[g] += w * x_cols[0][i];
                                x_sum_ptr[groups + g] += w * x_cols[1][i];
                                x_sum_ptr[2 * groups + g] += w * x_cols[2][i];
                                x_sum_ptr[3 * groups + g] += w * x_cols[3][i];
                                break;
                            default:
                                for (int j = 0; j < cols; ++j) {
                                    x_sum_ptr[static_cast<Eigen::Index>(j) * groups + g] +=
                                        w * x_cols[j][i];
                                }
                                break;
                        }
                    }
                }
            }

            double* y_sums_out = workspace.y_sums.data();
            double* x_sums_out = workspace.x_sums.data();
            for (int tid = 0; tid < local_threads; ++tid) {
                const auto& touched =
                    workspace.tls_touched_groups[static_cast<std::size_t>(tid)];
                const Eigen::VectorXd& y_local =
                    workspace.y_tls[static_cast<std::size_t>(tid)];
                const double* x_local =
                    cols > 0 ? workspace.x_tls[static_cast<std::size_t>(tid)].data()
                             : nullptr;
                for (int idx = 0; idx < static_cast<int>(touched.size()); ++idx) {
                    const int g = touched[static_cast<std::size_t>(idx)];
                    y_sums_out[g] += y_local[g];
                    switch (cols) {
                        case 0:
                            break;
                        case 1:
                            x_sums_out[g] += x_local[g];
                            break;
                        case 2:
                            x_sums_out[g] += x_local[g];
                            x_sums_out[groups + g] += x_local[groups + g];
                            break;
                        case 3:
                            x_sums_out[g] += x_local[g];
                            x_sums_out[groups + g] += x_local[groups + g];
                            x_sums_out[2 * groups + g] += x_local[2 * groups + g];
                            break;
                        case 4:
                            x_sums_out[g] += x_local[g];
                            x_sums_out[groups + g] += x_local[groups + g];
                            x_sums_out[2 * groups + g] += x_local[2 * groups + g];
                            x_sums_out[3 * groups + g] += x_local[3 * groups + g];
                            break;
                        default:
                            for (int j = 0; j < cols; ++j) {
                                const Eigen::Index offset =
                                    static_cast<Eigen::Index>(j) * groups + g;
                                x_sums_out[offset] += x_local[offset];
                            }
                            break;
                    }
                }
            }
        }
    } else
#endif
    {
        double* y_sums = workspace.y_sums.data();
        if (unit_weights) {
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                workspace.touch(g);
                y_sums[g] += y_ptr[i];
                switch (cols) {
                    case 0:
                        break;
                    case 1:
                        x_sum_cols[0][g] += x_cols[0][i];
                        break;
                    case 2:
                        x_sum_cols[0][g] += x_cols[0][i];
                        x_sum_cols[1][g] += x_cols[1][i];
                        break;
                    case 3:
                        x_sum_cols[0][g] += x_cols[0][i];
                        x_sum_cols[1][g] += x_cols[1][i];
                        x_sum_cols[2][g] += x_cols[2][i];
                        break;
                    case 4:
                        x_sum_cols[0][g] += x_cols[0][i];
                        x_sum_cols[1][g] += x_cols[1][i];
                        x_sum_cols[2][g] += x_cols[2][i];
                        x_sum_cols[3][g] += x_cols[3][i];
                        break;
                    default:
                        for (int j = 0; j < cols; ++j) {
                            x_sum_cols[j][g] += x_cols[j][i];
                        }
                        break;
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const int g = gid[i];
                const double w = weight_ptr[i];
                workspace.touch(g);
                y_sums[g] += w * y_ptr[i];
                switch (cols) {
                    case 0:
                        break;
                    case 1:
                        x_sum_cols[0][g] += w * x_cols[0][i];
                        break;
                    case 2:
                        x_sum_cols[0][g] += w * x_cols[0][i];
                        x_sum_cols[1][g] += w * x_cols[1][i];
                        break;
                    case 3:
                        x_sum_cols[0][g] += w * x_cols[0][i];
                        x_sum_cols[1][g] += w * x_cols[1][i];
                        x_sum_cols[2][g] += w * x_cols[2][i];
                        break;
                    case 4:
                        x_sum_cols[0][g] += w * x_cols[0][i];
                        x_sum_cols[1][g] += w * x_cols[1][i];
                        x_sum_cols[2][g] += w * x_cols[2][i];
                        x_sum_cols[3][g] += w * x_cols[3][i];
                        break;
                    default:
                        for (int j = 0; j < cols; ++j) {
                            x_sum_cols[j][g] += w * x_cols[j][i];
                        }
                        break;
                }
            }
        }
    }

    // Convert to means
    double* y_sums = workspace.y_sums.data();
    double* x_sums_ptr = workspace.x_sums.data();
    const bool track_max = (out_max_abs != nullptr);
    double local_max = 0.0;
    if (workspace.sparse_reset && !workspace.touched_groups.empty()) {
        if (!accumulate) {
            for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size()); ++idx) {
                const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
                const double inv = weight_sums_inv[g];
                y_sums[g] *= inv;
                if (track_max) {
                    const double abs_val = std::abs(y_sums[g]);
                    if (abs_val > local_max) {
                        local_max = abs_val;
                    }
                }
                if (cols > 0) {
                    for (int j = 0; j < cols; ++j) {
                        x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g] *= inv;
                    }
                }
            }
        } else {
            for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size()); ++idx) {
                const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
                const double inv = weight_sums_inv[g];
                y_sums[g] *= inv;
                if (alpha_y) {
                    alpha_y[g] += alpha_scale * y_sums[g];
                }
                if (track_max) {
                    const double abs_val = std::abs(y_sums[g]);
                    if (abs_val > local_max) {
                        local_max = abs_val;
                    }
                }
                if (cols > 0) {
                    for (int j = 0; j < cols; ++j) {
                        const Eigen::Index offset = static_cast<Eigen::Index>(j) * groups + g;
                        x_sums_ptr[offset] *= inv;
                        if (alpha_X) {
                            alpha_X[offset] += alpha_scale * x_sums_ptr[offset];
                        }
                    }
                }
            }
        }
    } else {
        if (!accumulate) {
#ifdef HDFE_USE_OPENMP
#pragma omp simd
#endif
            for (int g = 0; g < groups; ++g) {
                const double inv = weight_sums_inv[g];
                y_sums[g] *= inv;
                if (track_max) {
                    const double abs_val = std::abs(y_sums[g]);
                    if (abs_val > local_max) {
                        local_max = abs_val;
                    }
                }
                if (cols > 0) {
                    for (int j = 0; j < cols; ++j) {
                        x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g] *= inv;
                    }
                }
            }
        } else {
#ifdef HDFE_USE_OPENMP
#pragma omp simd
#endif
            for (int g = 0; g < groups; ++g) {
                const double inv = weight_sums_inv[g];
                y_sums[g] *= inv;
                if (alpha_y) {
                    alpha_y[g] += alpha_scale * y_sums[g];
                }
                if (track_max) {
                    const double abs_val = std::abs(y_sums[g]);
                    if (abs_val > local_max) {
                        local_max = abs_val;
                    }
                }
                if (cols > 0) {
                    for (int j = 0; j < cols; ++j) {
                        const Eigen::Index offset = static_cast<Eigen::Index>(j) * groups + g;
                        x_sums_ptr[offset] *= inv;
                        if (alpha_X) {
                            alpha_X[offset] += alpha_scale * x_sums_ptr[offset];
                        }
                    }
                }
            }
        }
    }

    if (track_max) {
        *out_max_abs = local_max;
    }

    if (!subtract) {
        return;
    }

    // Subtract means
    if (!out_sum_squares) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
        for (int i = 0; i < n; ++i) {
            const int g = gid[i];
            y_ptr[i] -= y_sums[g];
            switch (cols) {
                case 0:
                    break;
                case 1:
                    x_cols[0][i] -= x_sums_ptr[g];
                    break;
                case 2:
                    x_cols[0][i] -= x_sums_ptr[g];
                    x_cols[1][i] -= x_sums_ptr[groups + g];
                    break;
                case 3:
                    x_cols[0][i] -= x_sums_ptr[g];
                    x_cols[1][i] -= x_sums_ptr[groups + g];
                    x_cols[2][i] -= x_sums_ptr[2 * groups + g];
                    break;
                case 4:
                    x_cols[0][i] -= x_sums_ptr[g];
                    x_cols[1][i] -= x_sums_ptr[groups + g];
                    x_cols[2][i] -= x_sums_ptr[2 * groups + g];
                    x_cols[3][i] -= x_sums_ptr[3 * groups + g];
                    break;
                default:
                    for (int j = 0; j < cols; ++j) {
                        x_cols[j][i] -= x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                    }
                    break;
            }
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for reduction(+ : sumsq) schedule(static) num_threads(local_threads)
#endif
    for (int i = 0; i < n; ++i) {
        const int g = gid[i];
        y_ptr[i] -= y_sums[g];
        const double yi = y_ptr[i];
        sumsq += yi * yi;
        switch (cols) {
            case 0:
                break;
            case 1: {
                x_cols[0][i] -= x_sums_ptr[g];
                const double xi = x_cols[0][i];
                sumsq += xi * xi;
                break;
            }
            case 2: {
                x_cols[0][i] -= x_sums_ptr[g];
                const double x0v = x_cols[0][i];
                sumsq += x0v * x0v;
                x_cols[1][i] -= x_sums_ptr[groups + g];
                const double x1v = x_cols[1][i];
                sumsq += x1v * x1v;
                break;
            }
            case 3: {
                x_cols[0][i] -= x_sums_ptr[g];
                const double x0v = x_cols[0][i];
                sumsq += x0v * x0v;
                x_cols[1][i] -= x_sums_ptr[groups + g];
                const double x1v = x_cols[1][i];
                sumsq += x1v * x1v;
                x_cols[2][i] -= x_sums_ptr[2 * groups + g];
                const double x2v = x_cols[2][i];
                sumsq += x2v * x2v;
                break;
            }
            case 4: {
                x_cols[0][i] -= x_sums_ptr[g];
                const double x0v = x_cols[0][i];
                sumsq += x0v * x0v;
                x_cols[1][i] -= x_sums_ptr[groups + g];
                const double x1v = x_cols[1][i];
                sumsq += x1v * x1v;
                x_cols[2][i] -= x_sums_ptr[2 * groups + g];
                const double x2v = x_cols[2][i];
                sumsq += x2v * x2v;
                x_cols[3][i] -= x_sums_ptr[3 * groups + g];
                const double x3v = x_cols[3][i];
                sumsq += x3v * x3v;
                break;
            }
            default:
                for (int j = 0; j < cols; ++j) {
                    x_cols[j][i] -= x_sums_ptr[static_cast<Eigen::Index>(j) * groups + g];
                    const double xj = x_cols[j][i];
                    sumsq += xj * xj;
                }
                break;
        }
    }
    *out_sum_squares = sumsq;
}

double combined_norm(const Eigen::VectorXd& y, const Eigen::MatrixXd& X) {
#ifdef HDFE_USE_OPENMP
    double total = 0.0;
    const Eigen::Index y_size = y.size();
    const double* y_data = y.data();
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (Eigen::Index i = 0; i < y_size; ++i) {
        const double v = y_data[i];
        total += v * v;
    }
    const Eigen::Index rows = X.rows();
    const Eigen::Index cols = X.cols();
    const double* X_data = X.data();
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (Eigen::Index j = 0; j < cols; ++j) {
        const double* __restrict__ col = X_data + j * rows;
        double local = 0.0;
#pragma omp simd reduction(+ : local)
        for (Eigen::Index i = 0; i < rows; ++i) {
            const double v = col[i];
            local += v * v;
        }
        total += local;
    }
    return std::sqrt(total);
#else
    const double y_norm = y.squaredNorm();
    const double x_norm = X.squaredNorm();
    return std::sqrt(y_norm + x_norm);
#endif
}

// The heterogeneous-slope (mixed) absorber resolves convergence_criterion =
// Auto through tolerance_mode so that absorb(fe#c.x) follows the same default
// semantics as the rest of xhdfe: reghdfe-comparable maps to the
// reghdfe-style update criterion at the nominal tolerance, xhdfe-fast keeps
// the historical norm-change trigger. Explicit convergence() settings
// override the mapping. Standard (slope-free) absorption never consults the
// criterion; it is governed by tolerance_mode alone.
ConvergenceCriterion resolved_mixed_convergence_criterion(const HdfeOptions& options) {
    if (options.convergence_criterion != ConvergenceCriterion::Auto) {
        return options.convergence_criterion;
    }
    return options.tolerance_mode == ToleranceMode::ReghdfeComparable
               ? ConvergenceCriterion::Reghdfe
               : ConvergenceCriterion::NormChange;
}

bool needs_reghdfe_update_check(ConvergenceCriterion criterion) {
    return criterion == ConvergenceCriterion::Reghdfe ||
           criterion == ConvergenceCriterion::Both;
}

bool needs_reghdfe_update_check(const HdfeOptions& options) {
    return needs_reghdfe_update_check(options.convergence_criterion);
}

double reldif_value(double newer, double older) {
    return std::abs(newer - older) / (std::abs(older) + 1.0);
}

bool convergence_reached(ConvergenceCriterion criterion,
                         double tol,
                         double norm_error,
                         double update_error) {
    const bool norm_ok = norm_error < tol;
    const bool reghdfe_ok = update_error <= tol;
    switch (criterion) {
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

bool convergence_reached(const HdfeOptions& options,
                         double norm_error,
                         double update_error) {
    return convergence_reached(options.convergence_criterion, options.tol, norm_error,
                               update_error);
}

double mean_reldif_update_error(const Eigen::VectorXd& y_new,
                                const Eigen::VectorXd& y_old,
                                const Eigen::MatrixXd& X_new,
                                const Eigen::MatrixXd& X_old,
                                const Eigen::VectorXd* weights,
                                int threads) {
    const int n = static_cast<int>(y_new.size());
    if (n == 0) {
        return 0.0;
    }
    if (y_old.size() != y_new.size() || X_new.rows() != n || X_old.rows() != n ||
        X_new.cols() != X_old.cols()) {
        throw std::runtime_error("Invalid convergence snapshot dimensions");
    }
    const int cols = static_cast<int>(X_new.cols());
    const int width = cols + 1;
    std::vector<long double> sums(static_cast<std::size_t>(width), 0.0L);
    long double weight_sum = 0.0L;
    const double* yp_new = y_new.data();
    const double* yp_old = y_old.data();
    const double* Xp_new = X_new.data();
    const double* Xp_old = X_old.data();
    const double* wp = weights ? weights->data() : nullptr;

#ifdef HDFE_USE_OPENMP
    const int use_threads = std::max(1, threads);
    std::vector<std::vector<long double>> partial(
        static_cast<std::size_t>(use_threads),
        std::vector<long double>(static_cast<std::size_t>(width), 0.0L));
    std::vector<long double> partial_w(static_cast<std::size_t>(use_threads), 0.0L);
#pragma omp parallel num_threads(use_threads)
    {
        const int tid = omp_get_thread_num();
        std::vector<long double>& local = partial[static_cast<std::size_t>(tid)];
        long double local_w = 0.0L;
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const long double w = wp ? static_cast<long double>(wp[i]) : 1.0L;
            local_w += w;
            local[0] += w * static_cast<long double>(reldif_value(yp_new[i], yp_old[i]));
            for (int j = 0; j < cols; ++j) {
                const Eigen::Index idx =
                    static_cast<Eigen::Index>(j) * static_cast<Eigen::Index>(n) + i;
                local[static_cast<std::size_t>(j + 1)] +=
                    w * static_cast<long double>(reldif_value(Xp_new[idx], Xp_old[idx]));
            }
        }
        partial_w[static_cast<std::size_t>(tid)] = local_w;
    }
    for (int t = 0; t < use_threads; ++t) {
        weight_sum += partial_w[static_cast<std::size_t>(t)];
        for (int j = 0; j < width; ++j) {
            sums[static_cast<std::size_t>(j)] +=
                partial[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)];
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        const long double w = wp ? static_cast<long double>(wp[i]) : 1.0L;
        weight_sum += w;
        sums[0] += w * static_cast<long double>(reldif_value(yp_new[i], yp_old[i]));
        for (int j = 0; j < cols; ++j) {
            const Eigen::Index idx =
                static_cast<Eigen::Index>(j) * static_cast<Eigen::Index>(n) + i;
            sums[static_cast<std::size_t>(j + 1)] +=
                w * static_cast<long double>(reldif_value(Xp_new[idx], Xp_old[idx]));
        }
    }
#endif

    if (!(weight_sum > 0.0L)) {
        throw std::runtime_error("Weights must sum to a positive value");
    }
    long double max_mean = 0.0L;
    for (const long double sum : sums) {
        max_mean = std::max(max_mean, sum / weight_sum);
    }
    return static_cast<double>(max_mean);
}

double packed_mean_reldif_update_error(const Eigen::VectorXd& newer,
                                       const Eigen::VectorXd& older,
                                       int n,
                                       int stride,
                                       const Eigen::VectorXd* weights,
                                       int threads) {
    if (n <= 0 || stride <= 0) {
        return 0.0;
    }
    if (newer.size() != older.size() ||
        newer.size() != static_cast<Eigen::Index>(n) * static_cast<Eigen::Index>(stride)) {
        throw std::runtime_error("Invalid packed convergence snapshot dimensions");
    }
    std::vector<long double> sums(static_cast<std::size_t>(stride), 0.0L);
    long double weight_sum = 0.0L;
    const double* np = newer.data();
    const double* op = older.data();
    const double* wp = weights ? weights->data() : nullptr;

#ifdef HDFE_USE_OPENMP
    const int use_threads = std::max(1, threads);
    std::vector<std::vector<long double>> partial(
        static_cast<std::size_t>(use_threads),
        std::vector<long double>(static_cast<std::size_t>(stride), 0.0L));
    std::vector<long double> partial_w(static_cast<std::size_t>(use_threads), 0.0L);
#pragma omp parallel num_threads(use_threads)
    {
        const int tid = omp_get_thread_num();
        std::vector<long double>& local = partial[static_cast<std::size_t>(tid)];
        long double local_w = 0.0L;
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const long double w = wp ? static_cast<long double>(wp[i]) : 1.0L;
            local_w += w;
            const std::size_t base =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(stride);
            for (int j = 0; j < stride; ++j) {
                const std::size_t idx = base + static_cast<std::size_t>(j);
                local[static_cast<std::size_t>(j)] +=
                    w * static_cast<long double>(reldif_value(np[idx], op[idx]));
            }
        }
        partial_w[static_cast<std::size_t>(tid)] = local_w;
    }
    for (int t = 0; t < use_threads; ++t) {
        weight_sum += partial_w[static_cast<std::size_t>(t)];
        for (int j = 0; j < stride; ++j) {
            sums[static_cast<std::size_t>(j)] +=
                partial[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)];
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        const long double w = wp ? static_cast<long double>(wp[i]) : 1.0L;
        weight_sum += w;
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(stride);
        for (int j = 0; j < stride; ++j) {
            const std::size_t idx = base + static_cast<std::size_t>(j);
            sums[static_cast<std::size_t>(j)] +=
                w * static_cast<long double>(reldif_value(np[idx], op[idx]));
        }
    }
#endif

    if (!(weight_sum > 0.0L)) {
        throw std::runtime_error("Weights must sum to a positive value");
    }
    long double max_mean = 0.0L;
    for (const long double sum : sums) {
        max_mean = std::max(max_mean, sum / weight_sum);
    }
    return static_cast<double>(max_mean);
}

double max_abs_means(const FeWorkspace& workspace) {
    double max_val = 0.0;
    const int groups = workspace.num_groups();
    const int cols = static_cast<int>(workspace.x_sums.cols());
    if (workspace.sparse_reset && !workspace.touched_groups.empty()) {
        for (int idx = 0; idx < static_cast<int>(workspace.touched_groups.size()); ++idx) {
            const int g = workspace.touched_groups[static_cast<std::size_t>(idx)];
            max_val = std::max(max_val, std::abs(workspace.y_sums[g]));
            if (cols > 0) {
                const double* x_ptr = workspace.x_sums.data();
                for (int c = 0; c < cols; ++c) {
                    const double v = x_ptr[g + static_cast<Eigen::Index>(c) * groups];
                    max_val = std::max(max_val, std::abs(v));
                }
            }
        }
        return max_val;
    }

    for (int g = 0; g < groups; ++g) {
        max_val = std::max(max_val, std::abs(workspace.y_sums[g]));
    }
    if (workspace.x_sums.size() > 0) {
        const double* x_ptr = workspace.x_sums.data();
        for (Eigen::Index i = 0; i < workspace.x_sums.size(); ++i) {
            max_val = std::max(max_val, std::abs(x_ptr[i]));
        }
    }
    return max_val;
}

struct HeteroSlopeWorkspace {
    GroupCSR csr;
    Eigen::VectorXd sum_z;
    Eigen::VectorXd sum_zz;
    Eigen::VectorXi rank;
    Eigen::VectorXd alpha_y;
    Eigen::VectorXd gamma_y;
    Eigen::MatrixXd alpha_X;
    Eigen::MatrixXd gamma_X;
    bool include_intercept = false;

    int num_groups() const { return static_cast<int>(sum_z.size()); }
};

std::vector<const HeterogeneousSlopeTerm*> build_slope_lookup(
    const std::vector<HeterogeneousSlopeTerm>& slopes,
    std::size_t dims,
    int n) {
    std::vector<const HeterogeneousSlopeTerm*> lookup(dims, nullptr);
    for (const auto& slope : slopes) {
        if (slope.fe_index < 0 || slope.fe_index >= static_cast<int>(dims)) {
            throw std::runtime_error("Heterogeneous slope FE index out of range");
        }
        if (slope.values.size() != n) {
            throw std::runtime_error("Heterogeneous slope variable must match the length of y");
        }
        auto& slot = lookup[static_cast<std::size_t>(slope.fe_index)];
        if (slot != nullptr) {
            throw std::runtime_error("Only one heterogeneous slope variable per absorbed term is supported");
        }
        slot = &slope;
    }
    return lookup;
}

HeteroSlopeWorkspace prepare_slope_workspace(const FeIndexer& idx,
                                             const HeterogeneousSlopeTerm& slope,
                                             const double* weight_ptr,
                                             bool unit_weights,
                                             int n,
                                             int cols,
                                             int threads) {
    HeteroSlopeWorkspace ws;
    ws.include_intercept = slope.include_intercept;
    ws.csr = build_group_csr(idx.group_ids.data(), n, idx.num_groups);
    ws.sum_z = Eigen::VectorXd::Zero(idx.num_groups);
    ws.sum_zz = Eigen::VectorXd::Zero(idx.num_groups);
    ws.rank = Eigen::VectorXi::Zero(idx.num_groups);
    ws.alpha_y = Eigen::VectorXd::Zero(idx.num_groups);
    ws.gamma_y = Eigen::VectorXd::Zero(idx.num_groups);
    ws.alpha_X = Eigen::MatrixXd::Zero(idx.num_groups, cols);
    ws.gamma_X = Eigen::MatrixXd::Zero(idx.num_groups, cols);

    const double* z = slope.values.data();
    const int* group_ptr = ws.csr.group_ptr.data();
    const int* obs_index = ws.csr.obs_index.data();
    const double det_tol_base = 1e-12;

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
#endif
    for (int g = 0; g < idx.num_groups; ++g) {
        double sw = 0.0;
        double sz = 0.0;
        double szz = 0.0;
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        if (unit_weights) {
            sw = static_cast<double>(end - start);
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                const double zi = z[i];
                sz += zi;
                szz += zi * zi;
            }
        } else {
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                const double w = weight_ptr[i];
                const double zi = z[i];
                sw += w;
                sz += w * zi;
                szz += w * zi * zi;
            }
        }
        ws.sum_z[g] = sz;
        ws.sum_zz[g] = szz;
        if (slope.include_intercept) {
            if (sw > 0.0) {
                const double det = sw * szz - sz * sz;
                const double scale = std::max(1.0, sw * szz);
                ws.rank[g] = det > det_tol_base * scale ? 2 : 1;
            }
        } else {
            const double scale = std::max(1.0, szz);
            ws.rank[g] = szz > det_tol_base * scale ? 1 : 0;
        }
    }

    return ws;
}

void slope_project_inplace(Eigen::VectorXd& y,
                           Eigen::MatrixXd& X,
                           const HeterogeneousSlopeTerm& slope,
                           HeteroSlopeWorkspace& ws,
                           const double* weight_ptr,
                           bool unit_weights,
                           int threads,
                           bool subtract,
                           double* out_sum_squares = nullptr) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int rows = static_cast<int>(X.rows());
    const int groups = ws.num_groups();
    const double* z = slope.values.data();
    const double* y_ptr_const = y.data();
    double* y_ptr = y.data();
    double* X_ptr = X.data();
    const int* group_ptr = ws.csr.group_ptr.data();
    const int* obs_index = ws.csr.obs_index.data();
    const double det_tol_base = 1e-12;

    ws.alpha_y.setZero();
    ws.gamma_y.setZero();
    if (cols > 0) {
        ws.alpha_X.setZero();
        ws.gamma_X.setZero();
    }

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(std::max(1, threads))
#endif
    {
        // Fused multi-column accumulation: one walk over each group's rows
        // gathers y and every X column together instead of re-walking
        // obs_index/z once per column. Each per-column sum still accumulates
        // in increasing-p order, so the results match the per-column walk.
        std::vector<double> col_acc(static_cast<std::size_t>(2 * std::max(cols, 1)), 0.0);
        double* sx = col_acc.data();
        double* szx = col_acc.data() + cols;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int g = 0; g < groups; ++g) {
            double sw = 0.0;
            double sy = 0.0;
            double szy = 0.0;
            for (int j = 0; j < cols; ++j) {
                sx[j] = 0.0;
                szx[j] = 0.0;
            }
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];

            if (unit_weights) {
                sw = static_cast<double>(end - start);
                for (int p = start; p < end; ++p) {
                    const int i = obs_index[p];
                    const double yi = y_ptr_const[i];
                    const double zi = z[i];
                    sy += yi;
                    szy += zi * yi;
                    for (int j = 0; j < cols; ++j) {
                        const double xi = X_ptr[static_cast<Eigen::Index>(j) * rows + i];
                        sx[j] += xi;
                        szx[j] += zi * xi;
                    }
                }
            } else {
                for (int p = start; p < end; ++p) {
                    const int i = obs_index[p];
                    const double w = weight_ptr[i];
                    const double yi = y_ptr_const[i];
                    const double zi = z[i];
                    sw += w;
                    sy += w * yi;
                    szy += w * zi * yi;
                    for (int j = 0; j < cols; ++j) {
                        const double xi = X_ptr[static_cast<Eigen::Index>(j) * rows + i];
                        sx[j] += w * xi;
                        szx[j] += w * zi * xi;
                    }
                }
            }

            const double sz = ws.sum_z[g];
            const double szz = ws.sum_zz[g];
            // The det/scale test and 2x2 inverse depend only on the group,
            // not on the column being solved.
            double det = 0.0;
            bool full_rank = false;
            if (slope.include_intercept) {
                if (sw > 0.0) {
                    det = sw * szz - sz * sz;
                    const double scale = std::max(1.0, sw * szz);
                    full_rank = det > det_tol_base * scale;
                }
            } else {
                full_rank = szz > det_tol_base * std::max(1.0, szz);
            }

            double alpha = 0.0;
            double gamma = 0.0;
            if (slope.include_intercept) {
                if (sw > 0.0) {
                    if (full_rank) {
                        alpha = (szz * sy - sz * szy) / det;
                        gamma = (-sz * sy + sw * szy) / det;
                    } else {
                        alpha = sy / sw;
                    }
                }
            } else if (full_rank) {
                gamma = szy / szz;
            }
            ws.alpha_y[g] = alpha;
            ws.gamma_y[g] = gamma;

            for (int j = 0; j < cols; ++j) {
                double ax = 0.0;
                double gx = 0.0;
                if (slope.include_intercept) {
                    if (sw > 0.0) {
                        if (full_rank) {
                            ax = (szz * sx[j] - sz * szx[j]) / det;
                            gx = (-sz * sx[j] + sw * szx[j]) / det;
                        } else {
                            ax = sx[j] / sw;
                        }
                    }
                } else if (full_rank) {
                    gx = szx[j] / szz;
                }
                ws.alpha_X(g, j) = ax;
                ws.gamma_X(g, j) = gx;
            }
        }
    }

    if (!subtract) {
        if (out_sum_squares) {
            *out_sum_squares = combined_norm(y, X);
            *out_sum_squares *= *out_sum_squares;
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(std::max(1, threads)) reduction(+ : sumsq)
#endif
    {
        // Per-group alpha/gamma hoisted out of the row loop: (g, j) indexing
        // into the groups x cols matrices strides by `groups` per column.
        std::vector<double> coef_acc(static_cast<std::size_t>(2 * std::max(cols, 1)), 0.0);
        double* ax = coef_acc.data();
        double* gx = coef_acc.data() + cols;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double ay = ws.alpha_y[g];
            const double gy = ws.gamma_y[g];
            for (int j = 0; j < cols; ++j) {
                ax[j] = ws.alpha_X(g, j);
                gx[j] = ws.gamma_X(g, j);
            }
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                const double zi = z[i];
                const double y_new = y_ptr[i] - ay - gy * zi;
                y_ptr[i] = y_new;
                if (out_sum_squares) {
                    sumsq += y_new * y_new;
                }
                for (int j = 0; j < cols; ++j) {
                    double* x_col = X_ptr + static_cast<Eigen::Index>(j) * rows;
                    const double x_new = x_col[i] - ax[j] - gx[j] * zi;
                    x_col[i] = x_new;
                    if (out_sum_squares) {
                        sumsq += x_new * x_new;
                    }
                }
            }
        }
    }
    if (out_sum_squares) {
        *out_sum_squares = sumsq;
    }
}

int limit_threads_by_tls(int requested, int groups, int cols) {
#ifdef HDFE_USE_OPENMP
    if (requested <= 1 || groups <= 0) {
        return std::max(1, requested);
    }
    constexpr double kMaxTlsBytes = 1024.0 * 1024.0 * 1024.0;  // 1 GiB cap per FE.
    const double per_thread =
        static_cast<double>(groups) * static_cast<double>(cols + 1) * sizeof(double);
    if (per_thread <= 0.0) {
        return std::max(1, requested);
    }
    int max_threads = static_cast<int>(kMaxTlsBytes / per_thread);
    if (max_threads < 1) {
        max_threads = 1;
    }
    return std::min(requested, max_threads);
#else
    (void)requested;
    (void)groups;
    (void)cols;
    return 1;
#endif
}

int cap_threads_by_fe_shape(int requested,
                            int n,
                            const std::vector<FeIndexer>& indexers) {
#ifdef HDFE_USE_OPENMP
    int capped = std::max(1, requested);
    if (capped <= 1 || n <= 0 || indexers.empty()) {
        return capped;
    }
    if (n < 100000) {
        return 1;
    }

    int max_groups = 0;
    int high_cardinality_dims = 0;
    double min_occupancy = -1.0;
    for (const auto& idx : indexers) {
        const int groups = std::max(1, idx.num_groups);
        max_groups = std::max(max_groups, groups);
        if (groups >= 100000) {
            ++high_cardinality_dims;
        }
        const double occupancy = static_cast<double>(n) / static_cast<double>(groups);
        if (min_occupancy < 0.0 || occupancy < min_occupancy) {
            min_occupancy = occupancy;
        }
    }

    int shape_cap = 8;
    if (max_groups <= 4096 && min_occupancy >= 64.0) {
        shape_cap = 4;
    } else if (n >= 300000 && high_cardinality_dims >= 2 &&
               min_occupancy >= 0.0 && min_occupancy <= 3.0) {
        shape_cap = 16;
    }
    return std::min(capped, shape_cap);
#else
    (void)n;
    (void)indexers;
    return std::max(1, requested);
#endif
}

// The shape cap above is tuned for sub-million-row panels, where per-sweep
// synchronization overhead dominates and 4-16 threads win. On very large
// panels (tens of millions of rows) the demean sweeps are memory-bandwidth
// bound and the cap strands most of a multi-socket machine's bandwidth, so
// honor the requested thread count there. Every dataset below the gate keeps
// the capped behavior byte-for-byte; per-dim TLS and >=10M-group protections
// still apply downstream. Disable with XHDFE_UNCAP_LARGE_N=0 to restore the
// cap unconditionally.
int cap_threads_by_fe_shape_gated(int requested,
                                  int n,
                                  const std::vector<FeIndexer>& indexers) {
    const int capped = cap_threads_by_fe_shape(requested, n, indexers);
#ifdef HDFE_USE_OPENMP
    static const bool uncap_disabled = []() {
        const char* e = std::getenv("XHDFE_UNCAP_LARGE_N");
        return e != nullptr && e[0] == '0';
    }();
    if (!uncap_disabled && n >= 4194304) {
        return std::max(capped, std::min(std::max(1, requested), omp_get_max_threads()));
    }
#endif
    return capped;
}

// ---------------------------------------------------------------------------
// Fused same-FE heterogeneous-slope projection.
//
// When several slope terms are attached to the SAME absorbed FE (identical
// group ids), the alternating-projection sweep applies them as separate
// sequential rank-1/rank-2 projections. Projecting once onto the joint
// per-group span {1 (if any term has an intercept), z_1, ..., z_k} reaches the
// same fixed point (the joint span equals the sum of the per-term spans) with
// one gather/scatter pass instead of one per term, and typically in fewer
// sweeps. The per-group Gram matrix is iteration-invariant, so its
// rank-revealing Cholesky factor is computed once up front.
// Kill switch: XHDFE_MIXED_FUSE_SLOPES=0 restores the per-term sweeps.
struct FusedSlopeWorkspace {
    std::vector<std::size_t> member_dims;  // absorbed dims folded into this unit
    std::vector<const double*> zs;         // slope value arrays, member order
    bool include_intercept = false;
    int basis = 0;                         // (intercept ? 1 : 0) + zs.size()
    const GroupCSR* csr = nullptr;         // shared CSR (first member's)
    int num_groups = 0;
    std::vector<double> chol;              // packed lower-tri per group
    std::vector<uint8_t> keep;             // per group per direction
    std::vector<double> beta;              // per group: basis x (1 + cols)
    int beta_stride = 0;                   // basis * (1 + cols)
};

bool mixed_fuse_slopes_enabled() {
    const char* e = std::getenv("XHDFE_MIXED_FUSE_SLOPES");
    if (e && *e && (*e == '0' || *e == 'n' || *e == 'N' || *e == 'f' || *e == 'F')) {
        return false;
    }
    return true;
}

FusedSlopeWorkspace prepare_fused_slope_workspace(
    const std::vector<std::size_t>& member_dims,
    const std::vector<const HeterogeneousSlopeTerm*>& slope_lookup,
    const GroupCSR& csr,
    int num_groups,
    const double* weight_ptr,
    bool unit_weights,
    int cols,
    int threads) {
    FusedSlopeWorkspace ws;
    ws.member_dims = member_dims;
    ws.csr = &csr;
    ws.num_groups = num_groups;
    for (const std::size_t dim : member_dims) {
        ws.zs.push_back(slope_lookup[dim]->values.data());
        ws.include_intercept = ws.include_intercept || slope_lookup[dim]->include_intercept;
    }
    const int k = static_cast<int>(ws.zs.size());
    const int m = (ws.include_intercept ? 1 : 0) + k;
    ws.basis = m;
    const int packed = m * (m + 1) / 2;
    ws.chol.assign(static_cast<std::size_t>(num_groups) * packed, 0.0);
    ws.keep.assign(static_cast<std::size_t>(num_groups) * m, 0);
    ws.beta_stride = m * (cols + 1);
    ws.beta.assign(static_cast<std::size_t>(num_groups) * ws.beta_stride, 0.0);

    const int* group_ptr = csr.group_ptr.data();
    const int* obs_index = csr.obs_index.data();
    const double det_tol_base = 1e-12;
    const int intercept_off = ws.include_intercept ? 1 : 0;

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(std::max(1, threads))
#endif
    {
        // Per-thread Gram scratch (m x m, lower triangle used).
        std::vector<double> gram(static_cast<std::size_t>(m) * m, 0.0);
        std::vector<double> zv(static_cast<std::size_t>(std::max(k, 1)), 0.0);
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int g = 0; g < num_groups; ++g) {
            std::fill(gram.begin(), gram.end(), 0.0);
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            double sw = 0.0;
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                const double w = unit_weights ? 1.0 : weight_ptr[i];
                sw += w;
                for (int a = 0; a < k; ++a) {
                    zv[static_cast<std::size_t>(a)] = ws.zs[static_cast<std::size_t>(a)][i];
                }
                if (intercept_off) {
                    for (int a = 0; a < k; ++a) {
                        gram[static_cast<std::size_t>(a + 1) * m] +=
                            w * zv[static_cast<std::size_t>(a)];
                    }
                }
                for (int a = 0; a < k; ++a) {
                    const double za = w * zv[static_cast<std::size_t>(a)];
                    for (int b = 0; b <= a; ++b) {
                        gram[static_cast<std::size_t>(a + intercept_off) * m +
                             (b + intercept_off)] += za * zv[static_cast<std::size_t>(b)];
                    }
                }
            }
            if (intercept_off) {
                gram[0] = sw;
            }

            // Fixed-order rank-revealing Cholesky: intercept direction first,
            // then slope vars in member order (mirrors the sequential sweep
            // priority). A direction is dropped when its residual diagonal
            // falls below the same relative tolerance the per-term projections
            // use for their det/szz tests.
            double* L = ws.chol.data() + static_cast<std::size_t>(g) * packed;
            uint8_t* keep = ws.keep.data() + static_cast<std::size_t>(g) * m;
            const auto packed_at = [](int r, int c) {
                return r * (r + 1) / 2 + c;  // r >= c
            };
            for (int d = 0; d < m; ++d) {
                double resid = gram[static_cast<std::size_t>(d) * m + d];
                for (int t = 0; t < d; ++t) {
                    if (!keep[t]) continue;
                    const double ldt = L[packed_at(d, t)];
                    resid -= ldt * ldt;
                }
                const double diag_scale =
                    std::max(1.0, gram[static_cast<std::size_t>(d) * m + d]);
                if (!(resid > det_tol_base * diag_scale)) {
                    keep[d] = 0;
                    continue;
                }
                keep[d] = 1;
                const double ldd = std::sqrt(resid);
                L[packed_at(d, d)] = ldd;
                for (int r = d + 1; r < m; ++r) {
                    double v = gram[static_cast<std::size_t>(r) * m + d];
                    for (int t = 0; t < d; ++t) {
                        if (!keep[t]) continue;
                        v -= L[packed_at(r, t)] * L[packed_at(d, t)];
                    }
                    L[packed_at(r, d)] = v / ldd;
                }
            }
        }
    }
    return ws;
}

void fused_slope_project_inplace(Eigen::VectorXd& y,
                                 Eigen::MatrixXd& X,
                                 FusedSlopeWorkspace& ws,
                                 const double* weight_ptr,
                                 bool unit_weights,
                                 int threads,
                                 bool subtract,
                                 double* out_sum_squares = nullptr) {
    const int cols = static_cast<int>(X.cols());
    const int rows = static_cast<int>(X.rows());
    const int groups = ws.num_groups;
    const int m = ws.basis;
    const int k = static_cast<int>(ws.zs.size());
    const int intercept_off = ws.include_intercept ? 1 : 0;
    const int packed = m * (m + 1) / 2;
    const int ncol_all = cols + 1;  // y first, then the X columns
    double* y_ptr = y.data();
    double* X_ptr = X.data();
    const int* group_ptr = ws.csr->group_ptr.data();
    const int* obs_index = ws.csr->obs_index.data();
    const auto packed_at = [](int r, int c) { return r * (r + 1) / 2 + c; };

    // Specialized hot shape: two slope vars sharing one FE with an intercept
    // (m == 3). Scalar accumulators + __restrict keep the row loop in
    // registers; the generic path below covers every other shape.
    const bool fast_shape3 = (k == 2 && intercept_off == 1);

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(std::max(1, threads))
#endif
    {
        // Per-thread moment scratch: b[c * m + d] for c over {y, X cols}.
        std::vector<double> mom(static_cast<std::size_t>(ncol_all) * m, 0.0);
        std::vector<double> zv(static_cast<std::size_t>(std::max(k, 1)), 0.0);
        std::vector<double> tvec(static_cast<std::size_t>(m), 0.0);
        double* __restrict mom_ptr = mom.data();
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            if (fast_shape3) {
                const double* __restrict zA = ws.zs[0];
                const double* __restrict zB = ws.zs[1];
                const double* __restrict y_c = y_ptr;
                double my0 = 0.0, my1 = 0.0, my2 = 0.0;
                for (int j = 0; j < 3 * cols; ++j) {
                    mom_ptr[j] = 0.0;
                }
                if (unit_weights) {
                    for (int p = start; p < end; ++p) {
                        const int i = obs_index[p];
                        const double za = zA[i];
                        const double zb = zB[i];
                        const double yi = y_c[i];
                        my0 += yi;
                        my1 += yi * za;
                        my2 += yi * zb;
                        for (int j = 0; j < cols; ++j) {
                            const double xi =
                                X_ptr[static_cast<Eigen::Index>(j) * rows + i];
                            mom_ptr[3 * j] += xi;
                            mom_ptr[3 * j + 1] += xi * za;
                            mom_ptr[3 * j + 2] += xi * zb;
                        }
                    }
                } else {
                    for (int p = start; p < end; ++p) {
                        const int i = obs_index[p];
                        const double w = weight_ptr[i];
                        const double za = zA[i];
                        const double zb = zB[i];
                        const double yi = w * y_c[i];
                        my0 += yi;
                        my1 += yi * za;
                        my2 += yi * zb;
                        for (int j = 0; j < cols; ++j) {
                            const double xi =
                                w * X_ptr[static_cast<Eigen::Index>(j) * rows + i];
                            mom_ptr[3 * j] += xi;
                            mom_ptr[3 * j + 1] += xi * za;
                            mom_ptr[3 * j + 2] += xi * zb;
                        }
                    }
                }

                const double* __restrict L =
                    ws.chol.data() + static_cast<std::size_t>(g) * 6;
                const uint8_t* __restrict keep =
                    ws.keep.data() + static_cast<std::size_t>(g) * 3;
                double* __restrict beta =
                    ws.beta.data() + static_cast<std::size_t>(g) * ws.beta_stride;
                const auto solve3 = [&](double b0, double b1, double b2,
                                        double* __restrict out) {
                    // Forward substitution (packed lower tri: L00 L10 L11 L20 L21 L22).
                    double t0 = 0.0, t1 = 0.0, t2 = 0.0;
                    if (keep[0]) t0 = b0 / L[0];
                    if (keep[1]) {
                        double v = b1;
                        if (keep[0]) v -= L[1] * t0;
                        t1 = v / L[2];
                    }
                    if (keep[2]) {
                        double v = b2;
                        if (keep[0]) v -= L[3] * t0;
                        if (keep[1]) v -= L[4] * t1;
                        t2 = v / L[5];
                    }
                    double o2 = 0.0, o1 = 0.0, o0 = 0.0;
                    if (keep[2]) o2 = t2 / L[5];
                    if (keep[1]) {
                        double v = t1;
                        if (keep[2]) v -= L[4] * o2;
                        o1 = v / L[2];
                    }
                    if (keep[0]) {
                        double v = t0;
                        if (keep[1]) v -= L[1] * o1;
                        if (keep[2]) v -= L[3] * o2;
                        o0 = v / L[0];
                    }
                    out[0] = o0;
                    out[1] = o1;
                    out[2] = o2;
                };
                solve3(my0, my1, my2, beta);
                for (int j = 0; j < cols; ++j) {
                    solve3(mom_ptr[3 * j], mom_ptr[3 * j + 1], mom_ptr[3 * j + 2],
                           beta + static_cast<std::size_t>(j + 1) * 3);
                }
                continue;
            }

            std::fill(mom.begin(), mom.end(), 0.0);
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                const double w = unit_weights ? 1.0 : weight_ptr[i];
                for (int a = 0; a < k; ++a) {
                    zv[static_cast<std::size_t>(a)] = ws.zs[static_cast<std::size_t>(a)][i];
                }
                const double yi = w * y_ptr[i];
                double* my = mom.data();
                if (intercept_off) {
                    my[0] += yi;
                }
                for (int a = 0; a < k; ++a) {
                    my[a + intercept_off] += yi * zv[static_cast<std::size_t>(a)];
                }
                for (int j = 0; j < cols; ++j) {
                    const double xi =
                        w * X_ptr[static_cast<Eigen::Index>(j) * rows + i];
                    double* mx = mom.data() + static_cast<std::size_t>(j + 1) * m;
                    if (intercept_off) {
                        mx[0] += xi;
                    }
                    for (int a = 0; a < k; ++a) {
                        mx[a + intercept_off] += xi * zv[static_cast<std::size_t>(a)];
                    }
                }
            }

            const double* L = ws.chol.data() + static_cast<std::size_t>(g) * packed;
            const uint8_t* keep = ws.keep.data() + static_cast<std::size_t>(g) * m;
            double* beta =
                ws.beta.data() + static_cast<std::size_t>(g) * ws.beta_stride;
            for (int c = 0; c < ncol_all; ++c) {
                const double* b = mom.data() + static_cast<std::size_t>(c) * m;
                double* bc = beta + static_cast<std::size_t>(c) * m;
                // Forward substitution over kept directions.
                for (int d = 0; d < m; ++d) {
                    if (!keep[d]) {
                        tvec[static_cast<std::size_t>(d)] = 0.0;
                        continue;
                    }
                    double v = b[d];
                    for (int t = 0; t < d; ++t) {
                        if (!keep[t]) continue;
                        v -= L[packed_at(d, t)] * tvec[static_cast<std::size_t>(t)];
                    }
                    tvec[static_cast<std::size_t>(d)] = v / L[packed_at(d, d)];
                }
                // Backward substitution.
                for (int d = m - 1; d >= 0; --d) {
                    if (!keep[d]) {
                        bc[d] = 0.0;
                        continue;
                    }
                    double v = tvec[static_cast<std::size_t>(d)];
                    for (int r = d + 1; r < m; ++r) {
                        if (!keep[r]) continue;
                        v -= L[packed_at(r, d)] * bc[r];
                    }
                    bc[d] = v / L[packed_at(d, d)];
                }
            }
        }
    }

    if (!subtract) {
        if (out_sum_squares) {
            *out_sum_squares = combined_norm(y, X);
            *out_sum_squares *= *out_sum_squares;
        }
        return;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(std::max(1, threads)) reduction(+ : sumsq)
#endif
    {
        std::vector<double> zv(static_cast<std::size_t>(std::max(k, 1)), 0.0);
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double* __restrict beta =
                ws.beta.data() + static_cast<std::size_t>(g) * ws.beta_stride;
            if (fast_shape3) {
                const double* __restrict zA = ws.zs[0];
                const double* __restrict zB = ws.zs[1];
                for (int p = start; p < end; ++p) {
                    const int i = obs_index[p];
                    const double za = zA[i];
                    const double zb = zB[i];
                    const double y_new =
                        y_ptr[i] - (beta[0] + beta[1] * za + beta[2] * zb);
                    y_ptr[i] = y_new;
                    if (out_sum_squares) {
                        sumsq += y_new * y_new;
                    }
                    for (int j = 0; j < cols; ++j) {
                        const double* __restrict bc =
                            beta + static_cast<std::size_t>(j + 1) * 3;
                        double* __restrict x_col =
                            X_ptr + static_cast<Eigen::Index>(j) * rows;
                        const double x_new =
                            x_col[i] - (bc[0] + bc[1] * za + bc[2] * zb);
                        x_col[i] = x_new;
                        if (out_sum_squares) {
                            sumsq += x_new * x_new;
                        }
                    }
                }
                continue;
            }
            for (int p = start; p < end; ++p) {
                const int i = obs_index[p];
                for (int a = 0; a < k; ++a) {
                    zv[static_cast<std::size_t>(a)] = ws.zs[static_cast<std::size_t>(a)][i];
                }
                double fit_y = intercept_off ? beta[0] : 0.0;
                for (int a = 0; a < k; ++a) {
                    fit_y += beta[a + intercept_off] * zv[static_cast<std::size_t>(a)];
                }
                const double y_new = y_ptr[i] - fit_y;
                y_ptr[i] = y_new;
                if (out_sum_squares) {
                    sumsq += y_new * y_new;
                }
                for (int j = 0; j < cols; ++j) {
                    const double* bc = beta + static_cast<std::size_t>(j + 1) * m;
                    double fit_x = intercept_off ? bc[0] : 0.0;
                    for (int a = 0; a < k; ++a) {
                        fit_x += bc[a + intercept_off] * zv[static_cast<std::size_t>(a)];
                    }
                    double* x_col = X_ptr + static_cast<Eigen::Index>(j) * rows;
                    const double x_new = x_col[i] - fit_x;
                    x_col[i] = x_new;
                    if (out_sum_squares) {
                        sumsq += x_new * x_new;
                    }
                }
            }
        }
    }
    if (out_sum_squares) {
        *out_sum_squares = sumsq;
    }
}

struct IronsTuckStats {
    double vprod = 0.0;
    double ssq = 0.0;
    // Squared norm of the first difference ‖G(gx)−gx‖² (the change produced
    // by one full sweep). Used by the opt-in norm-of-change stopping
    // criterion; accumulated unconditionally because it costs one extra FMA
    // in a memory-bound loop.
    double d1sq = 0.0;
};

// TEMP INSTRUMENTATION (env-gated, no-op unless XHDFE_ACCEL_DEBUG set). Used to
// capture the Irons-Tuck divergence signature on ill-conditioned designs.
inline bool xhdfe_accel_debug() {
    static const bool on = []() {
        const char* e = std::getenv("XHDFE_ACCEL_DEBUG");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

struct FeAlphaState {
    std::vector<Eigen::VectorXd> y;
    std::vector<Eigen::MatrixXd> X;
    bool enabled = false;
    int cols = 0;
};

void init_alpha_state(FeAlphaState& state, const std::vector<FeIndexer>& indexers, int cols) {
    state.enabled = true;
    state.cols = cols;
    state.y.resize(indexers.size());
    state.X.resize(indexers.size());
    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        const int groups = indexers[dim].num_groups;
        state.y[dim] = Eigen::VectorXd::Zero(groups);
        state.X[dim] = Eigen::MatrixXd::Zero(groups, cols);
    }
}

void copy_alpha(FeAlphaState& dst, const FeAlphaState& src) {
    if (!src.enabled) {
        dst.enabled = false;
        dst.cols = 0;
        dst.y.clear();
        dst.X.clear();
        return;
    }
    dst.enabled = true;
    dst.cols = src.cols;
    dst.y.resize(src.y.size());
    dst.X.resize(src.X.size());
    for (std::size_t dim = 0; dim < src.y.size(); ++dim) {
        dst.y[dim] = src.y[dim];
        dst.X[dim] = src.X[dim];
    }
}

void lincomb_alpha(FeAlphaState& out,
                   const FeAlphaState& a,
                   const FeAlphaState& b,
                   double coef) {
    if (!a.enabled || !b.enabled) {
        out.enabled = false;
        out.cols = 0;
        out.y.clear();
        out.X.clear();
        return;
    }
    const double inv = 1.0 - coef;
    out.enabled = true;
    out.cols = a.cols;
    out.y.resize(a.y.size());
    out.X.resize(a.X.size());
    for (std::size_t dim = 0; dim < a.y.size(); ++dim) {
        out.y[dim].resize(a.y[dim].size());
        out.y[dim].noalias() = inv * b.y[dim];
        out.y[dim].noalias() += coef * a.y[dim];
        out.X[dim].resize(a.X[dim].rows(), a.X[dim].cols());
        out.X[dim].noalias() = inv * b.X[dim];
        out.X[dim].noalias() += coef * a.X[dim];
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

FeProfileStats profile_fe(const FeIndexer& idx,
                          const Eigen::VectorXd& weight_sums,
                          bool unit_weights) {
    FeProfileStats stats;
    const int groups = idx.num_groups;
    stats.num_groups = groups;
    stats.num_levels_present = idx.num_levels_present;
    if (groups <= 0 || weight_sums.size() == 0) {
        return stats;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    int singletons = 0;
    constexpr double kSingletonTol = 1e-12;
    for (int g = 0; g < groups; ++g) {
        const double v = weight_sums[g];
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

FeProfileStats profile_fe_from_weights(int num_groups,
                                       int num_levels_present,
                                       const Eigen::VectorXd& weight_sums,
                                       bool unit_weights) {
    FeProfileStats stats;
    stats.num_groups = num_groups;
    stats.num_levels_present = num_levels_present;
    if (num_groups <= 0 || weight_sums.size() == 0) {
        return stats;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    int singletons = 0;
    constexpr double kSingletonTol = 1e-12;
    for (int g = 0; g < num_groups; ++g) {
        const double v = weight_sums[g];
        sum += v;
        sumsq += v * v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        if (v <= 1.0 + kSingletonTol) {
            ++singletons;
        }
    }
    stats.mean = sum / static_cast<double>(num_groups);
    const double var =
        std::max(0.0, sumsq / static_cast<double>(num_groups) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(var);
    stats.cv = stats.mean > 0.0 ? stats.stddev / stats.mean : 0.0;
    stats.min_val = std::isfinite(min_val) ? min_val : 0.0;
    stats.max_val = std::isfinite(max_val) ? max_val : 0.0;
    stats.singleton_ratio = unit_weights
                                ? static_cast<double>(singletons) / static_cast<double>(num_groups)
                                : 0.0;
    return stats;
}

double auto_reghdfe_cg_dominant_share_threshold() {
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

bool reghdfe_cg_env_disabled() {
    const char* disabled = std::getenv("XHDFE_REGHDFE_CG");
    if (disabled == nullptr) {
        disabled = std::getenv("XHDFE_REHDFE_CG");
    }
    return disabled != nullptr && disabled[0] == '0';
}

// ---- Schwarz adaptive auto-gate: env knobs + cheap MAP-difficulty probe ----
bool schwarz_env_disabled() {
    const char* e = std::getenv("XHDFE_SCHWARZ");
    return e != nullptr && e[0] == '0';
}
int64_t schwarz_auto_nmin() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_NMIN");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(750000);
    }();
    return v;
}
double schwarz_auto_mswitch() {
    // Threshold on the plain-Gauss-Seidel probe's projected iters-to-tol. The probe runs
    // UNACCELERATED GS, which overestimates difficulty for accel-friendly datasets (e.g.
    // ready: probe proj~6600 but Irons-Tuck MAP needs only ~77 iters). Only genuinely
    // pathological graphs (difficult_1m: probe proj~330000, MAP-accel 470 it) project this
    // high. 50000 cleanly separates "switch" (difficult_1m-class) from "MAP handles it"
    // (akm ~2400, ready ~6600) on the 19-dataset suite, avoiding regressions on the latter.
    static const double v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_MSWITCH");
        return e ? std::atof(e) : 50000.0;
    }();
    return v;
}
bool schwarz_diag_enabled() { return std::getenv("XHDFE_SCHWARZ_DIAG") != nullptr; }

// ---- Adaptive MLSMR auto-promotion (default-auto path) --------------------
// When the Schwarz gate's plain-GS probe projects a MODERATELY hard graph --
// projected iters in [min, mswitch] -- the design is large and slow-converging
// (AKM/ready worker-firm-year class) but not pathological enough for Schwarz.
// On exactly this band the matrix-free MLSMR absorber beats accelerated MAP
// (AKM ~175->109s, ready ~81->75s; coef bit-identical). The band is gated to
// reghdfe-comparable/strict (the looser fast-mode tol needs far fewer MAP iters
// so MLSMR is not worth it) and reuses the gate's already-computed projection,
// so it adds NO probe cost. proj < min -> MAP handles it cheaply; proj > mswitch
// -> Schwarz/Jacobi-PCG (untouched); 4-way / n<nmin / weights / savefe / gpu
// never reach the gate. Disable with XHDFE_MLSMR_AUTO_GATE=0.
bool mlsmr_auto_gate_enabled() {
    static const bool v = []() {
        const char* e = std::getenv("XHDFE_MLSMR_AUTO_GATE");
        return !(e != nullptr && e[0] == '0');
    }();
    return v;
}
double mlsmr_auto_gate_min_proj() {
    static const double v = []() {
        const char* e = std::getenv("XHDFE_MLSMR_AUTO_MIN_PROJ");
        return e ? std::atof(e) : 2000.0;
    }();
    return v;
}

// Predict MAP difficulty WITHOUT touching the real iterate: run a few plain Gauss-Seidel
// sweeps on a COPY of y (the within projection is invariant to prior partial demeaning, so
// the contraction rate is a faithful proxy), estimate the geometric-mean contraction factor
// over a window, and return projected iterations-to-tol. Returns +inf when not contracting.
double schwarz_probe_projected_iters(const std::vector<const int*>& g,
                                     const std::vector<int>& ng,
                                     const double* y, int n, double tol) {
    const int K0 = 5, W = 4, NS = K0 + W;
    const int D = static_cast<int>(g.size());
    // Parallel group scatter. The serial scatter over n (run once for counts and NS*D times
    // for the sweep sums) dominated the probe cost on the large 2-3-way graphs the gate is
    // evaluated on (ready/akm ~46-56M rows): it added several sweeps of serial random-write
    // work ahead of MAP. For high-cardinality dims (many buckets -> low per-bucket
    // contention) an atomic parallel scatter is much faster; low-cardinality dims (few
    // buckets, cache-resident, high contention) stay serial where they are already fast.
    // The probe only feeds the coarse proj>mswitch gate threshold, so the low-bit
    // nondeterminism of the parallel reduction cannot change the routing decision, and it
    // never touches the real solver, residuals, or coefficients.
    constexpr int kProbeAtomicMinLevels = 4096;
    auto scatter_add = [&](const int* gd, int ngd, const double* val, double* out) {
#ifdef HDFE_USE_OPENMP
        if (ngd >= kProbeAtomicMinLevels) {
            if (val) {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
#pragma omp atomic
                    out[gd[i]] += val[i];
                }
            } else {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
#pragma omp atomic
                    out[gd[i]] += 1.0;
                }
            }
            return;
        }
#endif
        if (val) {
            for (int i = 0; i < n; ++i) out[gd[i]] += val[i];
        } else {
            for (int i = 0; i < n; ++i) out[gd[i]] += 1.0;
        }
    };
    std::vector<std::vector<double>> cnt(D), sum(D);
    for (int d = 0; d < D; ++d) {
        cnt[d].assign(ng[d], 0.0);
        sum[d].assign(ng[d], 0.0);
        scatter_add(g[d], ng[d], nullptr, cnt[d].data());
    }
    std::vector<double> yp(y, y + n);
    auto norm2 = [&]() { double s = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for reduction(+ : s) schedule(static)
#endif
        for (int i = 0; i < n; ++i) s += yp[i] * yp[i]; return std::sqrt(s); };
    double prevn = std::max(norm2(), 1e-300);
    double logprod = 0.0; int nr = 0;
    for (int it = 0; it < NS; ++it) {
        for (int d = 0; d < D; ++d) {
            auto& s = sum[d]; std::fill(s.begin(), s.end(), 0.0);
            const int* gd = g[d];
            scatter_add(gd, ng[d], yp.data(), s.data());
            for (size_t j = 0; j < s.size(); ++j) s[j] = cnt[d][j] > 0.0 ? s[j] / cnt[d][j] : 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < n; ++i) yp[i] -= s[gd[i]];
        }
        double nn = norm2();
        if (it >= K0 && prevn > 1e-300) { double r = nn / prevn; if (r > 0.0 && r < 1.0) { logprod += std::log(r); nr++; } }
        prevn = nn;
    }
    if (nr == 0) return 1e18;
    const double rho_bar = std::exp(logprod / nr);
    if (!(rho_bar > 0.0 && rho_bar < 1.0)) return 1e18;
    const double proj = std::log(tol) / std::log(rho_bar);
    return proj > 0.0 ? proj : 1e18;
}

bool run_schwarz_absorption_from_indexers(const Eigen::VectorXd& y,
                                          const Eigen::MatrixXd& X,
                                          const std::vector<FeIndexer>& indexers,
                                          const HdfeOptions& options,
                                          double convergence_tol,
                                          AbsorptionResult& result) {
    static_assert(sizeof(int) == sizeof(int32_t), "FeIndexer group_ids must be 32-bit");
    const int n = static_cast<int>(y.size());
    const int ncols = static_cast<int>(X.cols());
    if (indexers.size() < 2 || (ncols + 1) > 16) {
        return false;
    }

    result.y_tilde = y;
    result.X_tilde = X;
    std::vector<const int32_t*> fep;
    std::vector<int> nlev;
    fep.reserve(indexers.size());
    nlev.reserve(indexers.size());
    for (const auto& idx : indexers) {
        fep.push_back(reinterpret_cast<const int32_t*>(idx.group_ids.data()));
        nlev.push_back(idx.num_groups);
    }

    int sw_iters = 0;
    const bool ok = xhdfe::schwarz_demean_raw(
        result.y_tilde.data(), result.X_tilde.data(), n, ncols, fep, nlev,
        options.num_threads, convergence_tol, options.max_iter, &sw_iters, -1);
    if (!ok) {
        result.iterations = sw_iters;
        result.converged = false;
        result.schwarz_used = true;
        return false;
    }
    result.iterations = sw_iters;
    result.converged = true;
    result.schwarz_used = true;
    return true;
}

bool should_use_auto_default_reghdfe_cg_accelerator(
    const std::vector<FeProfileStats>& profiles,
    int n,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    bool store_alphas) {
    if (options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable) {
        return false;
    }
    if (store_alphas || options.retain_fixed_effects || weights != nullptr ||
        profiles.size() < 3 || n <= 0 || reghdfe_cg_env_disabled()) {
        return false;
    }

    const double threshold = auto_reghdfe_cg_dominant_share_threshold();
    if (!(threshold > 0.0 && threshold <= 1.0)) {
        return false;
    }

    double max_share = 0.0;
    const double denom = static_cast<double>(n);
    for (const auto& p : profiles) {
        max_share = std::max(max_share, p.max_val / denom);
    }
    return max_share >= threshold;
}

inline void irons_tuck_accumulate(const double* x,
                                  const double* gx,
                                  const double* ggx,
                                  Eigen::Index n,
                                  IronsTuckStats& stats) {
    double vprod = 0.0;
    double ssq = 0.0;
    double d1sq = 0.0;
#ifdef HDFE_USE_OPENMP
    constexpr Eigen::Index kParallelMinN = 1 << 18;
    if (n >= kParallelMinN) {
#pragma omp parallel for reduction(+ : vprod, ssq, d1sq) schedule(static)
        for (Eigen::Index i = 0; i < n; ++i) {
            const double gx_val = gx[i];
            const double delta_gx = ggx[i] - gx_val;
            const double delta2 = delta_gx - gx_val + x[i];
            vprod += delta_gx * delta2;
            ssq += delta2 * delta2;
            d1sq += delta_gx * delta_gx;
        }
        stats.vprod += vprod;
        stats.ssq += ssq;
        stats.d1sq += d1sq;
        return;
    }
#pragma omp simd reduction(+ : vprod, ssq, d1sq)
#endif
    for (Eigen::Index i = 0; i < n; ++i) {
        const double gx_val = gx[i];
        const double delta_gx = ggx[i] - gx_val;
        const double delta2 = delta_gx - gx_val + x[i];
        vprod += delta_gx * delta2;
        ssq += delta2 * delta2;
        d1sq += delta_gx * delta_gx;
    }
    stats.vprod += vprod;
    stats.ssq += ssq;
    stats.d1sq += d1sq;
}

inline void irons_tuck_update(double* x,
                              const double* gx,
                              const double* ggx,
                              Eigen::Index n,
                              double coef) {
#ifdef HDFE_USE_OPENMP
    constexpr Eigen::Index kParallelMinN = 1 << 18;
    if (n >= kParallelMinN) {
#pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < n; ++i) {
            const double delta_gx = ggx[i] - gx[i];
            x[i] = ggx[i] - coef * delta_gx;
        }
        return;
    }
#pragma omp simd
#endif
    for (Eigen::Index i = 0; i < n; ++i) {
        const double delta_gx = ggx[i] - gx[i];
        x[i] = ggx[i] - coef * delta_gx;
    }
}

// Parallel memcpy for double arrays. Splits the copy across threads using
// plain memcpy on per-thread contiguous chunks. Falls back to memcpy when
// below the size threshold. Used to speed up the per-iteration buffer copies
// in the accelerator loop (y_ggx = y_gx, y_gx = result.y_tilde, etc.).
inline void parallel_copy_doubles(double* __restrict__ dst,
                                  const double* __restrict__ src,
                                  std::size_t n,
                                  int threads) {
    if (dst == src || n == 0) {
        return;
    }
#ifdef HDFE_USE_OPENMP
    // Threshold: only parallelize when the copy is large enough that the
    // OpenMP fork/join overhead is amortized. 1 MiB (= 131072 doubles) works
    // well empirically on modern x86 servers.
    constexpr std::size_t kParallelMinDoubles = 1U << 17;
    if (threads > 1 && n >= kParallelMinDoubles) {
        const int T = std::max(1, threads);
        const std::size_t chunk =
            (n + static_cast<std::size_t>(T) - 1U) / static_cast<std::size_t>(T);
#pragma omp parallel num_threads(T)
        {
            const int tid = omp_get_thread_num();
            const std::size_t start = static_cast<std::size_t>(tid) * chunk;
            if (start < n) {
                const std::size_t len = std::min(chunk, n - start);
                std::memcpy(dst + start, src + start, len * sizeof(double));
            }
        }
        return;
    }
#else
    (void)threads;
#endif
    std::memcpy(dst, src, n * sizeof(double));
}

inline void parallel_copy_vec(Eigen::VectorXd& dst,
                              const Eigen::VectorXd& src,
                              int threads) {
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }
    parallel_copy_doubles(dst.data(), src.data(),
                          static_cast<std::size_t>(src.size()), threads);
}

inline void parallel_copy_mat(Eigen::MatrixXd& dst,
                              const Eigen::MatrixXd& src,
                              int threads) {
    if (dst.rows() != src.rows() || dst.cols() != src.cols()) {
        dst.resize(src.rows(), src.cols());
    }
    parallel_copy_doubles(dst.data(), src.data(),
                          static_cast<std::size_t>(src.size()), threads);
}

// Fused Irons-Tuck update + broadcast into the old "gx" buffer. Replicates
// exactly the same formula used by irons_tuck_update:
//   new_x[i] = ggx[i] - coef * (ggx[i] - gx[i])
// and writes new_x to both x[i] and gx_and_out[i] in a single pass. This
// collapses the subsequent "y_gx = result.y_tilde" copy that otherwise follows
// the update inside the accelerator loop, saving one full read per element.
// Safe in-place: gx_and_out[i] is read once (as old gx) before being written.
inline void irons_tuck_update_broadcast(double* x,
                                        double* gx_and_out,
                                        const double* ggx,
                                        Eigen::Index n,
                                        double coef) {
#ifdef HDFE_USE_OPENMP
    constexpr Eigen::Index kParallelMinN = 1 << 18;
    if (n >= kParallelMinN) {
#pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < n; ++i) {
            const double ggx_val = ggx[i];
            const double gx_val = gx_and_out[i];
            const double delta_gx = ggx_val - gx_val;
            const double new_x = ggx_val - coef * delta_gx;
            x[i] = new_x;
            gx_and_out[i] = new_x;
        }
        return;
    }
#pragma omp simd
#endif
    for (Eigen::Index i = 0; i < n; ++i) {
        const double ggx_val = ggx[i];
        const double gx_val = gx_and_out[i];
        const double delta_gx = ggx_val - gx_val;
        const double new_x = ggx_val - coef * delta_gx;
        x[i] = new_x;
        gx_and_out[i] = new_x;
    }
}

std::vector<std::size_t> select_sweep_order(const Eigen::Ref<const Eigen::VectorXd>& y,
                                            const std::vector<FeIndexer>& indexers,
                                            const double* weight_ptr,
                                            bool unit_weights,
                                            const std::vector<int>* override_order) {
    std::vector<std::size_t> order(indexers.size());
    std::iota(order.begin(), order.end(), 0);
    const std::size_t dims = indexers.size();
    if (override_order && !override_order->empty()) {
        if (override_order->size() == dims) {
            std::vector<uint8_t> seen(dims, 0);
            bool valid = true;
            for (const int v : *override_order) {
                if (v < 0 || v >= static_cast<int>(dims)) {
                    valid = false;
                    break;
                }
                const std::size_t idx = static_cast<std::size_t>(v);
                if (seen[idx]) {
                    valid = false;
                    break;
                }
                seen[idx] = 1;
            }
            if (valid) {
                std::vector<std::size_t> override_cast;
                override_cast.reserve(dims);
                for (const int v : *override_order) {
                    override_cast.push_back(static_cast<std::size_t>(v));
                }
                return override_cast;
            }
        }
    }
    if (dims <= 1 || dims > 4) {
        return order;
    }

    const int n_probe =
        static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(y.size()), 200000ULL));
    if (n_probe <= 0) {
        return order;
    }
    Eigen::VectorXd probe_y = y.head(n_probe);
    Eigen::MatrixXd probe_X(n_probe, 0);

    std::vector<FeWorkspace> trial_ws;
    trial_ws.reserve(dims);
    for (const auto& idx : indexers) {
        trial_ws.emplace_back(idx.num_groups, 0);
        trial_ws.back().weight_sums_const =
            compute_weight_sums(idx, weight_ptr, unit_weights, n_probe, 1);
        trial_ws.back().refresh_weight_sums_inv();
    }

    std::vector<std::size_t> perm = order;
    double best_norm = std::numeric_limits<double>::infinity();
    do {
        Eigen::VectorXd tmp_y = probe_y;
        for (int iter = 0; iter < 2; ++iter) {
            for (std::size_t pos : perm) {
                trial_ws[pos].reset();
                demean_inplace(tmp_y, probe_X, indexers[pos], trial_ws[pos], weight_ptr,
                               unit_weights, 1);
            }
        }
        const double norm = tmp_y.norm();
        if (norm < best_norm) {
            best_norm = norm;
            order = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return order;
}

std::vector<std::size_t> order_by_profile(const std::vector<FeProfileStats>& profiles) {
    std::vector<std::size_t> order(profiles.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         const auto& pa = profiles[a];
                         const auto& pb = profiles[b];
                         if (pa.num_groups != pb.num_groups) {
                             return pa.num_groups > pb.num_groups;
                         }
                         if (pa.singleton_ratio != pb.singleton_ratio) {
                             return pa.singleton_ratio < pb.singleton_ratio;
                         }
                         if (pa.cv != pb.cv) {
                             return pa.cv < pb.cv;
                         }
                         return a < b;
                     });
    return order;
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

AbsorptionMethod choose_absorption_method(std::size_t dims,
                                          int threads,
                                          const HdfeOptions& options,
                                          AbsorptionMethod requested) {
    if (requested != AbsorptionMethod::Auto) {
        return requested;
    }
    if (options.absorption_method != AbsorptionMethod::Auto) {
        return options.absorption_method;
    }
    if (dims <= 1) {
        return AbsorptionMethod::GaussSeidel;
    }
    if (dims == 2) {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    if (threads >= static_cast<int>(2 * dims)) {
        return AbsorptionMethod::Jacobi;
    }
    if (options.symmetric_sweep && dims > 1) {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    return AbsorptionMethod::GaussSeidel;
}

void apply_jacobi_update(Eigen::VectorXd& y,
                         Eigen::MatrixXd& X,
                         const std::vector<FeIndexer>& indexers,
                         const std::vector<FeWorkspace>& workspaces,
                         double relaxation,
                         int threads) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int rows = static_cast<int>(X.rows());
    const int dims = static_cast<int>(indexers.size());
    if (n == 0 || dims == 0) {
        return;
    }

    std::vector<const int*> gid_ptrs;
    gid_ptrs.reserve(static_cast<std::size_t>(dims));
    for (const auto& idx : indexers) {
        gid_ptrs.push_back(idx.group_ids.data());
    }

    std::vector<const double*> y_means;
    y_means.reserve(static_cast<std::size_t>(dims));
    std::vector<std::vector<const double*>> x_means;
    x_means.reserve(static_cast<std::size_t>(dims));
    for (const auto& ws : workspaces) {
        y_means.push_back(ws.y_sums.data());
        if (cols > 0) {
            std::vector<const double*> cols_ptr(static_cast<std::size_t>(cols));
            const double* base = ws.x_sums.data();
            const int groups = ws.num_groups();
            for (int j = 0; j < cols; ++j) {
                cols_ptr[static_cast<std::size_t>(j)] =
                    base + static_cast<Eigen::Index>(j) * groups;
            }
            x_means.push_back(std::move(cols_ptr));
        } else {
            x_means.emplace_back();
        }
    }

    double* y_ptr = y.data();
    constexpr int kInlineCols = 8;
    double* x_cols_inline[kInlineCols];
    std::vector<double*> x_cols_storage;
    double** x_cols = nullptr;
    if (cols > 0) {
        if (cols <= kInlineCols) {
            x_cols = x_cols_inline;
            for (int j = 0; j < cols; ++j) {
                x_cols_inline[j] = X.data() + static_cast<Eigen::Index>(j) * rows;
            }
        } else {
            x_cols_storage.resize(static_cast<std::size_t>(cols));
            for (int j = 0; j < cols; ++j) {
                x_cols_storage[static_cast<std::size_t>(j)] =
                    X.data() + static_cast<Eigen::Index>(j) * rows;
            }
            x_cols = x_cols_storage.data();
        }
    }

    const int local_threads = std::max(1, threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
    for (int i = 0; i < n; ++i) {
        double delta_y = 0.0;
        for (int dim = 0; dim < dims; ++dim) {
            const int g = gid_ptrs[static_cast<std::size_t>(dim)][i];
            delta_y += y_means[static_cast<std::size_t>(dim)][g];
            switch (cols) {
                case 0:
                    break;
                case 1:
                    x_cols[0][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][0][g];
                    break;
                case 2:
                    x_cols[0][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][0][g];
                    x_cols[1][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][1][g];
                    break;
                case 3:
                    x_cols[0][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][0][g];
                    x_cols[1][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][1][g];
                    x_cols[2][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][2][g];
                    break;
                case 4:
                    x_cols[0][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][0][g];
                    x_cols[1][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][1][g];
                    x_cols[2][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][2][g];
                    x_cols[3][i] -= relaxation * x_means[static_cast<std::size_t>(dim)][3][g];
                    break;
                default:
                    for (int j = 0; j < cols; ++j) {
                        x_cols[j][i] -= relaxation *
                                        x_means[static_cast<std::size_t>(dim)]
                                                [static_cast<std::size_t>(j)][g];
                    }
                    break;
            }
        }
        y_ptr[i] -= relaxation * delta_y;
    }
}

// ===================================================================
// AoS (packed / interleaved) fast-path demean kernels.
//
// Layout: a single flat row-major buffer `cur` of size n*S, where element
// (obs i, lane l) lives at cur[i*S + l]; lane 0 = y, lanes 1..S-1 = X columns
// (S = num_cols + 1). Co-locating y and all X columns of an observation in a
// single cache line slashes the number of distinct cache lines touched per
// random (non-contiguous) gather/scatter from S down to ~1, which is the
// dominant cost for high-cardinality non-contiguous FEs (e.g. firm id).
//
// These kernels intentionally support ONLY the unit-weight, no-stored-alpha
// case: weighted and savefe (FE-recovery) regressions keep using the proven
// Struct-of-Arrays path. The per-group reduction order matches the SoA kernels
// (sum observations of a group in the same order, divide, subtract), so the
// absorbed result agrees with the SoA path to floating-point round-off.
// ===================================================================
constexpr int kPackedPfDist = 24;

// Grouped gather/scatter. Random == true uses csr.obs_index (non-contiguous
// FE); Random == false treats observations of a group as the contiguous range
// [group_ptr[g], group_ptr[g+1]) (contiguous FE). `means` is scratch of size
// groups*S. When out_sumsq != null, accumulates the post-subtraction sum of
// squares of every touched element (matches the SoA demean sumsq).
template <int S, bool Random>
void packed_demean_grouped(double* __restrict cur,
                           const int* __restrict group_ptr,
                           const int* __restrict obs_index,
                           int groups,
                           int n,
                           const double* __restrict wsinv,
                           double* __restrict means,
                           int threads,
                           double* out_sumsq) {
    const int local_threads = std::max(1, threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
    for (int g = 0; g < groups; ++g) {
        const int start = group_ptr[g];
        const int end = group_ptr[g + 1];
        double acc[S];
        for (int l = 0; l < S; ++l) acc[l] = 0.0;
        for (int idx = start; idx < end; ++idx) {
            if (Random) {
                const int pf = idx + kPackedPfDist;
                if (pf < n) {
                    __builtin_prefetch(cur + static_cast<std::size_t>(obs_index[pf]) * S, 0, 1);
                }
            }
            const int i = Random ? obs_index[idx] : idx;
            const double* __restrict p = cur + static_cast<std::size_t>(i) * S;
            for (int l = 0; l < S; ++l) acc[l] += p[l];
        }
        const double inv = wsinv[g];
        double* __restrict m = means + static_cast<std::size_t>(g) * S;
        for (int l = 0; l < S; ++l) m[l] = acc[l] * inv;
    }

    if (out_sumsq) {
        double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(+ : sumsq)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double* __restrict m = means + static_cast<std::size_t>(g) * S;
            for (int idx = start; idx < end; ++idx) {
                if (Random) {
                    const int pf = idx + kPackedPfDist;
                    if (pf < n) {
                        __builtin_prefetch(cur + static_cast<std::size_t>(obs_index[pf]) * S, 1, 1);
                    }
                }
                const int i = Random ? obs_index[idx] : idx;
                double* __restrict p = cur + static_cast<std::size_t>(i) * S;
                for (int l = 0; l < S; ++l) {
                    p[l] -= m[l];
                    sumsq += p[l] * p[l];
                }
            }
        }
        *out_sumsq = sumsq;
    } else {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads)
#endif
        for (int g = 0; g < groups; ++g) {
            const int start = group_ptr[g];
            const int end = group_ptr[g + 1];
            const double* __restrict m = means + static_cast<std::size_t>(g) * S;
            for (int idx = start; idx < end; ++idx) {
                if (Random) {
                    const int pf = idx + kPackedPfDist;
                    if (pf < n) {
                        __builtin_prefetch(cur + static_cast<std::size_t>(obs_index[pf]) * S, 1, 1);
                    }
                }
                const int i = Random ? obs_index[idx] : idx;
                double* __restrict p = cur + static_cast<std::size_t>(i) * S;
                for (int l = 0; l < S; ++l) p[l] -= m[l];
            }
        }
    }
}

// Observation-partitioned gather/scatter for low-cardinality non-contiguous
// FEs (e.g. year): each thread accumulates per-group partial sums over its
// contiguous chunk of observations (sequential reads of `cur`), then the
// partials are combined. `tls[t]` is per-thread scratch of size groups*S.
template <int S>
void packed_demean_obspart(double* __restrict cur,
                           const int* __restrict group_ids,
                           int groups,
                           int n,
                           const double* __restrict wsinv,
                           double* __restrict means,
                           std::vector<std::vector<double>>& tls,
                           int threads,
                           double* out_sumsq) {
    const int local_threads = std::max(1, threads);
    const std::size_t msz = static_cast<std::size_t>(groups) * S;
    std::fill(means, means + msz, 0.0);
#ifdef HDFE_USE_OPENMP
    if (local_threads > 1) {
#pragma omp parallel num_threads(local_threads)
        {
            const int tid = omp_get_thread_num();
            double* __restrict t = tls[static_cast<std::size_t>(tid)].data();
            std::fill(t, t + msz, 0.0);
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const double* __restrict p = cur + static_cast<std::size_t>(i) * S;
                double* __restrict tg = t + static_cast<std::size_t>(group_ids[i]) * S;
                for (int l = 0; l < S; ++l) tg[l] += p[l];
            }
        }
        for (int tid = 0; tid < local_threads; ++tid) {
            const double* __restrict t = tls[static_cast<std::size_t>(tid)].data();
            for (std::size_t k = 0; k < msz; ++k) means[k] += t[k];
        }
    } else
#endif
    {
        for (int i = 0; i < n; ++i) {
            const double* __restrict p = cur + static_cast<std::size_t>(i) * S;
            double* __restrict mg = means + static_cast<std::size_t>(group_ids[i]) * S;
            for (int l = 0; l < S; ++l) mg[l] += p[l];
        }
    }
    for (int g = 0; g < groups; ++g) {
        const double inv = wsinv[g];
        double* __restrict m = means + static_cast<std::size_t>(g) * S;
        for (int l = 0; l < S; ++l) m[l] *= inv;
    }

    double sumsq = 0.0;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(local_threads) reduction(+ : sumsq)
#endif
    for (int i = 0; i < n; ++i) {
        double* __restrict p = cur + static_cast<std::size_t>(i) * S;
        const double* __restrict m = means + static_cast<std::size_t>(group_ids[i]) * S;
        for (int l = 0; l < S; ++l) {
            p[l] -= m[l];
            sumsq += p[l] * p[l];
        }
    }
    if (out_sumsq) *out_sumsq = sumsq;
}

// Runtime dispatch on S = num_cols + 1 (supported range [2, 9], i.e. up to 8
// regressors). Returns false if S is out of range so the caller can fall back.
bool packed_demean_grouped_dispatch(int S, bool random, double* cur,
                                    const int* group_ptr, const int* obs_index,
                                    int groups, int n, const double* wsinv,
                                    double* means, int threads, double* out_sumsq) {
#define HDFE_PG_CASE(s)                                                                       \
    case s:                                                                                   \
        if (random)                                                                           \
            packed_demean_grouped<s, true>(cur, group_ptr, obs_index, groups, n, wsinv,       \
                                           means, threads, out_sumsq);                          \
        else                                                                                  \
            packed_demean_grouped<s, false>(cur, group_ptr, obs_index, groups, n, wsinv,      \
                                            means, threads, out_sumsq);                         \
        return true;
    switch (S) {
        HDFE_PG_CASE(2)
        HDFE_PG_CASE(3)
        HDFE_PG_CASE(4)
        HDFE_PG_CASE(5)
        HDFE_PG_CASE(6)
        HDFE_PG_CASE(7)
        HDFE_PG_CASE(8)
        HDFE_PG_CASE(9)
        default:
            return false;
    }
#undef HDFE_PG_CASE
}

bool packed_demean_obspart_dispatch(int S, double* cur, const int* group_ids,
                                    int groups, int n, const double* wsinv, double* means,
                                    std::vector<std::vector<double>>& tls, int threads,
                                    double* out_sumsq) {
#define HDFE_PO_CASE(s)                                                                       \
    case s:                                                                                   \
        packed_demean_obspart<s>(cur, group_ids, groups, n, wsinv, means, tls, threads,       \
                                 out_sumsq);                                                  \
        return true;
    switch (S) {
        HDFE_PO_CASE(2)
        HDFE_PO_CASE(3)
        HDFE_PO_CASE(4)
        HDFE_PO_CASE(5)
        HDFE_PO_CASE(6)
        HDFE_PO_CASE(7)
        HDFE_PO_CASE(8)
        HDFE_PO_CASE(9)
        default:
            return false;
    }
#undef HDFE_PO_CASE
}

}  // namespace

AbsorptionResult absorb_fixed_effects(const Eigen::VectorXd& y,
                                      const Eigen::MatrixXd& X,
                                      const std::vector<Eigen::VectorXi>& fes,
                                      const Eigen::VectorXd* weights,
                                      const HdfeOptions& options) {
    const auto absorb_total_t0 = std::chrono::steady_clock::now();
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }
    const double convergence_tol = effective_absorption_tolerance(options);
    // reghdfe-comparable mode (and the XHDFE_TOL_TRIGGER=change diagnostic
    // knob) replace the historical change-of-norm trigger in the accelerated
    // loops with the norm-of-change criterion ‖G(x)−x‖ ≤ tol·max(1,‖x‖) at
    // the NOMINAL tolerance — the same semantics reghdfe attaches to its
    // tolerance(). Non-accelerated paths keep the calibrated effective
    // tolerance from effective_absorption_tolerance().
    const bool honest_mode =
        honest_tol_trigger_enabled() ||
        options.tolerance_mode == ToleranceMode::ReghdfeComparable;
    const double honest_tol = options.tol;

    AbsorptionResult result;
    const auto copy_t0 = std::chrono::steady_clock::now();
    // NOTE: keep this a single-threaded Eigen assignment. A 32-thread parallel
    // first-touch fill made the copy itself faster (~0.8s) but fragmented the
    // huge-page backing of y_tilde/X_tilde, which the CPU solver re-reads across
    // all MAP iterations -> +16s and highly variable wall-clock. The serial copy
    // preserves transparent-huge-page backing and stable solver bandwidth.
    result.y_tilde = y;
    result.X_tilde = X;
    cpu_profile_log_elapsed("abs_input_copy", copy_t0);

    if (fes.empty()) {
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    const auto indexer_t0 = std::chrono::steady_clock::now();
    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
    }
    std::vector<FeIndexer> indexers(fes.size());
#ifdef HDFE_USE_OPENMP
    {
        int idx_threads = options.num_threads > 0
                              ? std::min(options.num_threads, omp_get_max_threads())
                              : 1;
        idx_threads = std::min<int>(std::max(1, idx_threads),
                                    static_cast<int>(fes.size()));
        if (idx_threads > 1 && n >= 4194304) {
            // Dimensions are independent; each indexer is built and
            // first-touched by a single thread, bit-identical to serial.
#pragma omp parallel for schedule(dynamic, 1) num_threads(idx_threads)
            for (std::ptrdiff_t dim = 0; dim < static_cast<std::ptrdiff_t>(fes.size());
                 ++dim) {
                indexers[static_cast<std::size_t>(dim)] =
                    build_indexer(fes[static_cast<std::size_t>(dim)]);
            }
        } else {
            for (std::size_t dim = 0; dim < fes.size(); ++dim) {
                indexers[dim] = build_indexer(fes[dim]);
            }
        }
    }
#else
    for (std::size_t dim = 0; dim < fes.size(); ++dim) {
        indexers[dim] = build_indexer(fes[dim]);
    }
#endif
    for (const auto& idx : indexers) {
        result.fe_levels.push_back(idx.num_levels_present);
    }
    cpu_profile_log_elapsed("abs_indexers", indexer_t0);

    // ---- Forced Schwarz / approx-Cholesky PCG path (opt-in: absorption_method=schwarz) ----
    // Gated accelerator for large ill-conditioned multi-way FE (e.g. high-MAP-iteration panels).
    // The within projection M_D satisfies M_D*D=0, so running it on the
    // un-demeaned y/X yields the exact within residual when the PCG certificate
    // succeeds. Explicit Schwarz requests fail closed; auto-gated Schwarz may
    // still fall through to MAP if the specialized path is not usable.
    if (options.absorption_method == AbsorptionMethod::Schwarz) {
        const int ncols = static_cast<int>(result.X_tilde.cols());
        const bool supported = (weights == nullptr) && !options.retain_fixed_effects &&
                               indexers.size() >= 2 && (ncols + 1) <= 16;
        if (supported) {
            if (run_schwarz_absorption_from_indexers(y, X, indexers, options,
                                                     convergence_tol, result)) {
                cpu_profile_log_elapsed("abs_total", absorb_total_t0);
                return result;
            }
            if (!options.from_auto) {
                cpu_profile_log_elapsed("abs_total", absorb_total_t0);
                return result;
            }
        } else if (!options.from_auto) {
            result.iterations = 0;
            result.converged = false;
            cpu_profile_log_elapsed("abs_total", absorb_total_t0);
            return result;
        }
        // Unsupported auto-gated feature set or auto-gated solver failure:
        // fall through to the default MAP path.
    }

    int total_levels_for_exact_sparse = 0;
    for (const auto& idx : indexers) {
        total_levels_for_exact_sparse += idx.num_groups;
    }
    if (should_use_reference_sparse_absorber(indexers, n, total_levels_for_exact_sparse,
                                             weights, options)) {
        AbsorptionResult exact_result =
            absorb_fixed_effects_sparse(y, X, indexers, weights, options, true);
        if (exact_result.converged) {
            cpu_profile_log_elapsed("abs_total", absorb_total_t0);
            return exact_result;
        }
    }

    const auto setup_t0 = std::chrono::steady_clock::now();
    std::vector<FeWorkspace> workspaces;
    workspaces.reserve(indexers.size());
    const int num_cols = static_cast<int>(result.X_tilde.cols());
    for (const auto& idx : indexers) {
        workspaces.emplace_back(idx.num_groups, num_cols);
    }
    const bool store_alphas =
        options.retain_fixed_effects &&
        options.fe_recovery_method == FeRecoveryMethod::Hybrid &&
        num_cols <= options.savefe_fastpath_max_cols;

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();
    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    } else {
        threads = 1;  // default to single-thread for best cache efficiency on wide datasets
    }
    threads = std::max(1, threads);
    if (!options.retain_fixed_effects) {
        threads = cap_threads_by_fe_shape_gated(threads, n, indexers);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);
    std::vector<int> threads_by_dim(indexers.size(), threads);
#ifdef HDFE_USE_OPENMP
    if (threads > 2) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            const int groups = indexers[dim].num_groups;
            if (groups >= 10000000) {
                threads_by_dim[dim] = std::min(threads_by_dim[dim], 2);
            }
        }
    }
    if (threads > 1) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            if (!indexer_has_contiguous_groups(indexers[dim], n)) {
                threads_by_dim[dim] =
                    limit_threads_by_tls(threads_by_dim[dim], indexers[dim].num_groups, num_cols);
            }
        }
    }
#endif

    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        workspaces[dim].weight_sums_const =
            compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, threads_by_dim[dim]);
        workspaces[dim].refresh_weight_sums_inv();
        workspaces[dim].sparse_reset =
            (threads_by_dim[dim] == 1 && indexers[dim].num_groups <= 2000000 &&
             indexers[dim].num_groups > (n / 4));
    }

    std::vector<FeProfileStats> fe_profiles;
    fe_profiles.reserve(indexers.size());
    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        fe_profiles.push_back(
            profile_fe(indexers[dim], workspaces[dim].weight_sums_const, unit_weights));
    }
    const std::vector<int>* override_order =
        options.sweep_order_override.empty() ? nullptr : &options.sweep_order_override;
    std::vector<std::size_t> sweep_order;
    if (!override_order && indexers.size() > 2) {
        sweep_order = order_by_profile(fe_profiles);
    } else {
        sweep_order =
            select_sweep_order(result.y_tilde, indexers, weight_ptr, unit_weights, override_order);
    }

    bool use_symmetric = options.symmetric_sweep && sweep_order.size() > 1;
    // Irons-Tuck acceleration is normally reserved for large n, where the extra
    // sweep per iteration is amortised. But sparse, "tree-like" multi-FE graphs
    // (citation / board-membership / matched-pair panels such as the patents and
    // directors data) are ill-conditioned for plain Gauss-Seidel even at modest
    // n: unaccelerated GS needs hundreds-to-thousands of sweeps where the
    // accelerated path converges in tens. Detect them cheaply via the mean group
    // occupancy: when the smallest FE dimension averages only a few observations
    // per level the bipartite graph is near-acyclic and benefits from
    // acceleration. Well-connected panels (many obs/level) stay below the gate
    // and keep the exact previous behaviour, so no previously fast case slows.
    double min_fe_occupancy = -1.0;  // finite sentinel (avoid inf under -ffast-math)
    for (const auto& p : fe_profiles) {
        if (p.num_groups > 0 && (min_fe_occupancy < 0.0 || p.mean < min_fe_occupancy)) {
            min_fe_occupancy = p.mean;
        }
    }
    static const double accel_occ_threshold = []() {
        const char* e = std::getenv("XHDFE_ACCEL_OCC");
        return (e != nullptr) ? std::atof(e) : kAccelOccupancyThreshold;
    }();
    const bool ill_conditioned_graph =
        indexers.size() >= 2 && min_fe_occupancy >= 0.0 &&
        min_fe_occupancy <= accel_occ_threshold;
    const bool use_accel =
        indexers.size() >= 2 && (n >= 200000 || ill_conditioned_graph);
    // The occupancy gate also catches genuinely pathological graphs (e.g. a long
    // path/"zigzag" of degree-2 levels) where neither plain Gauss-Seidel nor the
    // accelerated solver can reach the tolerance. There, acceleration would spend
    // ~2 plain-equivalent sweeps per iteration without converging and could cost
    // more wall-clock than the plain path. To guarantee acceleration is purely
    // opportunistic — never slower than the un-accelerated solver at the same
    // budget — cap occupancy-triggered acceleration at 2/5 of max_iter: since an
    // accelerated symmetric iteration costs ~2 plain-equivalent sweeps, this
    // keeps its total sweep count strictly below the plain solver's (with
    // headroom for per-iteration bookkeeping). Converging cases (directors,
    // patents, ...) finish in far fewer iterations, so the cap is never reached
    // for them; only non-converging pathological cases are bounded. The proven
    // n>=200000 path keeps the full budget and is left exactly as before.
    const bool accel_from_occupancy = use_accel && (n < 200000);
    const int accel_iter_budget =
        accel_from_occupancy ? std::max(1, (options.max_iter * 2) / 5) : options.max_iter;
    if (std::getenv("XHDFE_DEBUG_ACCEL") != nullptr) {
        std::fprintf(stderr,
                     "[accel] n=%d nfe=%zu min_occ=%.3f thr=%.3f use_accel=%d budget=%d\n",
                     n, indexers.size(), min_fe_occupancy, accel_occ_threshold,
                     static_cast<int>(use_accel), accel_iter_budget);
    }
    bool converged = false;

    std::vector<GroupCSR> csr;
    std::vector<uint8_t> use_csr(indexers.size(), 0);
    const bool allow_csr = (n >= 200000 && num_cols <= 8);
    if (allow_csr) {
        csr.resize(indexers.size());
        constexpr int kCsrGroupsThreshold = 100000;
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            if (threads_by_dim[dim] > 1 &&
                indexers[dim].num_groups >= kCsrGroupsThreshold &&
                !indexer_has_contiguous_groups(indexers[dim], n)) {
                use_csr[dim] = 1;
                csr[dim] =
                    build_group_csr(indexers[dim].group_ids.data(), n, indexers[dim].num_groups);
            }
        }
    }

    std::vector<double> profile_dim_ms(indexers.size(), 0.0);
    std::vector<long long> profile_dim_calls(indexers.size(), 0);
    auto apply_demean = [&](Eigen::VectorXd& y_in,
                            Eigen::MatrixXd& X_in,
                            std::size_t dim,
                            double* out_sumsq,
                            double* alpha_y_ptr,
                            double* alpha_x_ptr) {
        const auto demean_t0 = std::chrono::steady_clock::now();
        if (use_csr[dim]) {
            demean_inplace_csr(y_in, X_in, csr[dim], workspaces[dim], weight_ptr, unit_weights,
                               threads_by_dim[dim], true, out_sumsq, 1.0,
                               alpha_y_ptr, alpha_x_ptr, nullptr);
        } else {
            demean_inplace(y_in, X_in, indexers[dim], workspaces[dim], weight_ptr, unit_weights,
                           threads_by_dim[dim], true, out_sumsq, 1.0,
                           alpha_y_ptr, alpha_x_ptr, nullptr);
        }
        if (cpu_profile_enabled()) {
            const auto demean_t1 = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = demean_t1 - demean_t0;
            profile_dim_ms[dim] += elapsed.count();
            ++profile_dim_calls[dim];
        }
    };

    FeAlphaState alpha_result;
    if (store_alphas) {
        init_alpha_state(alpha_result, indexers, num_cols);
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
    cpu_profile_log_elapsed("abs_setup", setup_t0);

    const auto solve_t0 = std::chrono::steady_clock::now();

    // ---------------- AoS (packed) fast path ----------------
    // Eligible only for the standard accelerated absorption with unit weights
    // and no fixed-effect recovery (savefe). Weighted / savefe regressions keep
    // the proven Struct-of-Arrays solve below. The packed solve replicates the
    // same two-stage Irons-Tuck algorithm on an interleaved [y|X] buffer; the
    // final tightening polish below (shared with the SoA path) makes the result
    // equivalent to the SoA path within floating-point tolerance.
    bool packed_done = false;
    bool packed_polish_ran = false;
    int packed_polish_sweeps = 0;
    double packed_polish_final_max = 0.0;
    {
        static const bool packed_disabled = []() {
            const char* e = std::getenv("XHDFE_PACKED");
            return e != nullptr && e[0] == '0';
        }();
        bool any_noncontig = false;
        for (const auto& idx : indexers) {
            if (!idx.groups_contiguous) {
                any_noncontig = true;
                break;
            }
        }
        const bool use_packed = use_accel && unit_weights && !options.retain_fixed_effects &&
                                num_cols >= 1 && num_cols <= 8 && any_noncontig && !packed_disabled;
        if (use_packed) {
            const int Sp = num_cols + 1;
            const std::size_t flat_size = static_cast<std::size_t>(n) * static_cast<std::size_t>(Sp);
            const int check_interval = std::max(1, options.convergence_check_interval);

            // Per-dimension demean strategy: 0 = contiguous, 1 = CSR (random),
            // 2 = observation-partitioned (low-cardinality non-contiguous).
            constexpr int kObsPartMaxGroups = 2048;
            std::vector<int> pmode(indexers.size(), 0);
            std::vector<const GroupCSR*> pcsr(indexers.size(), nullptr);
            std::vector<GroupCSR> plocal_csr(indexers.size());
            std::vector<std::vector<double>> pmeans(indexers.size());
            std::vector<std::vector<std::vector<double>>> ptls(indexers.size());
            for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                const int groups = indexers[dim].num_groups;
                pmeans[dim].assign(static_cast<std::size_t>(groups) * Sp, 0.0);
                if (indexers[dim].groups_contiguous) {
                    pmode[dim] = 0;
                } else if (groups <= kObsPartMaxGroups) {
                    pmode[dim] = 2;
                    const int dth = std::max(1, threads_by_dim[dim]);
                    ptls[dim].assign(static_cast<std::size_t>(dth),
                                     std::vector<double>(static_cast<std::size_t>(groups) * Sp, 0.0));
                } else {
                    pmode[dim] = 1;
                    if (use_csr[dim]) {
                        pcsr[dim] = &csr[dim];
                    } else {
                        plocal_csr[dim] =
                            build_group_csr(indexers[dim].group_ids.data(), n, groups);
                        pcsr[dim] = &plocal_csr[dim];
                    }
                }
            }

            Eigen::VectorXd cur(static_cast<Eigen::Index>(flat_size));
            // Pack [y | X(col-major)] -> interleaved row-major.
            {
                double* __restrict dst = cur.data();
                const double* __restrict yp = result.y_tilde.data();
                const double* __restrict Xp = result.X_tilde.data();
                const Eigen::Index nn = n;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                for (Eigen::Index i = 0; i < nn; ++i) {
                    double* __restrict d = dst + static_cast<std::size_t>(i) * Sp;
                    d[0] = yp[i];
                    for (int j = 0; j < num_cols; ++j) {
                        d[1 + j] = Xp[static_cast<Eigen::Index>(j) * nn + i];
                    }
                }
            }

            auto packed_demean_dim = [&](double* curp, std::size_t dim, double* out_sumsq) {
                const int groups = indexers[dim].num_groups;
                const double* wsinv = workspaces[dim].weight_sums_inv.data();
                double* means = pmeans[dim].data();
                // Contiguous packed demean parallelizes over groups with no
                // thread-local state: each group is reduced by exactly one
                // thread in observation order, so the result is bit-identical
                // for any thread count. The >=10M-group cap in threads_by_dim
                // protects the SoA kernels; it only throttles streaming
                // bandwidth here, so use the solver-wide thread count instead.
                const int dth = std::max(1, pmode[dim] == 0 ? threads
                                                            : threads_by_dim[dim]);
                if (pmode[dim] == 0) {
                    packed_demean_grouped_dispatch(Sp, false, curp,
                                                   indexers[dim].group_ptr.data(), nullptr, groups,
                                                   n, wsinv, means, dth, out_sumsq);
                } else if (pmode[dim] == 1) {
                    const GroupCSR* c = pcsr[dim];
                    packed_demean_grouped_dispatch(Sp, true, curp, c->group_ptr.data(),
                                                   c->obs_index.data(), groups, n, wsinv, means, dth,
                                                   out_sumsq);
                } else {
                    packed_demean_obspart_dispatch(Sp, curp, indexers[dim].group_ids.data(), groups,
                                                   n, wsinv, means, ptls[dim], dth, out_sumsq);
                }
            };

            auto packed_sweep = [&](double* curp, const std::vector<std::size_t>& order,
                                    double* out_sumsq) {
                double sumsq = 0.0;
                for (std::size_t pos = 0; pos < order.size(); ++pos) {
                    const std::size_t dim = order[pos];
                    const bool last_call =
                        out_sumsq && !use_symmetric && (pos + 1 == order.size());
                    packed_demean_dim(curp, dim, last_call ? &sumsq : nullptr);
                }
                if (use_symmetric) {
                    for (std::size_t idx = order.size(); idx-- > 0;) {
                        const std::size_t dim = order[idx];
                        const bool last_call = out_sumsq && (idx == 0);
                        packed_demean_dim(curp, dim, last_call ? &sumsq : nullptr);
                    }
                }
                if (out_sumsq) {
                    *out_sumsq = sumsq;
                }
            };

            Eigen::VectorXd gx(static_cast<Eigen::Index>(flat_size));
            Eigen::VectorXd ggx(static_cast<Eigen::Index>(flat_size));
            const int iter_grand_acc = 4;
            Eigen::VectorXd acc_a, acc_b, acc_c;
            acc_a.resize(static_cast<Eigen::Index>(flat_size));
            acc_b.resize(static_cast<Eigen::Index>(flat_size));
            acc_c.resize(static_cast<Eigen::Index>(flat_size));

            auto packed_col_sumsq = [&](const Eigen::VectorXd& value) {
                Eigen::VectorXd out = Eigen::VectorXd::Zero(Sp);
#ifdef HDFE_USE_OPENMP
                const int nth = std::max(1, threads);
                std::vector<double> tls(static_cast<std::size_t>(nth) *
                                        static_cast<std::size_t>(Sp), 0.0);
#pragma omp parallel num_threads(threads)
                {
                    const int tid = omp_get_thread_num();
                    double* local = tls.data() + static_cast<std::size_t>(tid) *
                                                  static_cast<std::size_t>(Sp);
#pragma omp for schedule(static)
                    for (Eigen::Index i = 0; i < n; ++i) {
                        const double* row =
                            value.data() + static_cast<std::size_t>(i) *
                                               static_cast<std::size_t>(Sp);
                        for (int k = 0; k < Sp; ++k) {
                            local[k] += row[k] * row[k];
                        }
                    }
                }
                for (int t = 0; t < nth; ++t) {
                    const double* local = tls.data() + static_cast<std::size_t>(t) *
                                                        static_cast<std::size_t>(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        out[k] += local[k];
                    }
                }
#else
                for (Eigen::Index i = 0; i < n; ++i) {
                    const double* row =
                        value.data() + static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        out[k] += row[k] * row[k];
                    }
                }
#endif
                return out;
            };

            auto packed_col_dot = [&](const Eigen::VectorXd& lhs,
                                      const Eigen::VectorXd& rhs) {
                Eigen::VectorXd out = Eigen::VectorXd::Zero(Sp);
#ifdef HDFE_USE_OPENMP
                const int nth = std::max(1, threads);
                std::vector<double> tls(static_cast<std::size_t>(nth) *
                                        static_cast<std::size_t>(Sp), 0.0);
#pragma omp parallel num_threads(threads)
                {
                    const int tid = omp_get_thread_num();
                    double* local = tls.data() + static_cast<std::size_t>(tid) *
                                                  static_cast<std::size_t>(Sp);
#pragma omp for schedule(static)
                    for (Eigen::Index i = 0; i < n; ++i) {
                        const std::size_t off =
                            static_cast<std::size_t>(i) * static_cast<std::size_t>(Sp);
                        const double* a = lhs.data() + off;
                        const double* b = rhs.data() + off;
                        for (int k = 0; k < Sp; ++k) {
                            local[k] += a[k] * b[k];
                        }
                    }
                }
                for (int t = 0; t < nth; ++t) {
                    const double* local = tls.data() + static_cast<std::size_t>(t) *
                                                        static_cast<std::size_t>(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        out[k] += local[k];
                    }
                }
#else
                for (Eigen::Index i = 0; i < n; ++i) {
                    const std::size_t off =
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(Sp);
                    const double* a = lhs.data() + off;
                    const double* b = rhs.data() + off;
                    for (int k = 0; k < Sp; ++k) {
                        out[k] += a[k] * b[k];
                    }
                }
#endif
                return out;
            };

            auto packed_axpy_columns = [&](Eigen::VectorXd& target,
                                           const Eigen::VectorXd& direction,
                                           const Eigen::VectorXd& alpha) {
                const double* ap = alpha.data();
                double* tp = target.data();
                const double* dp = direction.data();
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                for (Eigen::Index i = 0; i < n; ++i) {
                    const std::size_t off =
                        static_cast<std::size_t>(i) * static_cast<std::size_t>(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        tp[off + static_cast<std::size_t>(k)] -=
                            ap[k] * dp[off + static_cast<std::size_t>(k)];
                    }
                }
            };

            // IT->CG divergence hand-off (default ON). When the Irons-Tuck
            // adaptive safeguard detects divergence on an ill-conditioned FE
            // graph (e.g. the github 3-FE case), bail out of the accelerator and
            // finish with the stable symmetric-operator CG instead of grinding
            // 10k-20k plain Gauss-Seidel sweeps. Set XHDFE_PACKED_CG_FALLBACK=0
            // to restore the plain-GS fallback (for A/B on the same binary).
            static const bool packed_cg_fallback_enabled = []() {
                const char* e = std::getenv("XHDFE_PACKED_CG_FALLBACK");
                return !(e != nullptr && e[0] == '0');
            }();
            bool packed_accel_diverged = false;

            auto packed_accel_phase = [&](const std::vector<std::size_t>& order, int max_iter,
                                          int& iter_used) -> bool {
                iter_used = 0;
                if (max_iter <= 0 || order.empty()) {
                    return false;
                }
	                int last_check_iter = -1;
	                double prev_norm = std::sqrt(cur.squaredNorm());
	                int grand_acc = 0;
                // Adaptive-restart safeguard against Irons-Tuck instability on
                // ill-conditioned FE graphs. The IT extrapolation can amplify
                // (spectral radius > 1) and make the iterate norm diverge
                // exponentially while still reporting "converged" on a drifted
                // state. Because Delta1 = sweep(gx)-gx lies entirely in the FE
                // span, the IT step only ever injects FE-span components, so the
                // orthogonal projection M_D(iterate) (the actual answer) is
                // invariant; a plain sweep is firmly non-expansive and removes
                // that junk. We track the running-minimum post-sweep iterate
                // norm and, if the norm grows past kAccelDivergeFactor times it,
                // suspend extrapolation (coef=0, a plain non-expansive step)
                // until the norm recovers to a new best. On well-conditioned
                // data the iterate norm is monotone non-increasing, so the
                // trigger never fires and the iterate sequence is unchanged.
                constexpr double kAccelDivergeFactor = 4.0;
                // Hand off to CG as soon as the post-sweep norm rises this far
                // above its running minimum. On well-conditioned data the norm
                // is monotone non-increasing so the ratio never exceeds ~1, and
                // this never fires; only a diverging IT trajectory (github-class)
                // crosses it. Kept well below kAccelDivergeFactor so the hand-off
                // happens early (small pre-bail cost) yet with ample margin over
                // healthy numerical noise. Tunable via XHDFE_CG_BAIL_FACTOR.
                static const double cg_bail_factor = []() {
                    const char* e = std::getenv("XHDFE_CG_BAIL_FACTOR");
                    const double v = (e != nullptr) ? std::atof(e) : 1.5;
                    return v > 1.0 ? v : 1.5;
                }();
                // Iteration-cap hand-off: every well-conditioned dataset in the
                // supported battery converges in well under this many honest-mode
                // accelerated iterations (the slowest, workers, needs ~575); only
                // a github-class ill-conditioned graph keeps grinding past it.
                // Handing such a case to the stable symmetric-operator CG caps the
                // wasted Irons-Tuck sweeps. Tunable via XHDFE_CG_ITER_CAP (0=off).
                static const int cg_iter_cap = []() {
                    const char* e = std::getenv("XHDFE_CG_ITER_CAP");
                    if (e != nullptr) {
                        const int v = std::atoi(e);
                        return v > 0 ? v : 0;
                    }
                    return 1000;
                }();
                double best_resid = prev_norm;
                bool accel_suspended = false;
                int accel_restart_count = 0;
                double max_ratio_seen = 1.0;
                parallel_copy_vec(gx, cur, threads);
                packed_sweep(gx.data(), order, nullptr);
                for (int iter = 0; iter < max_iter; ++iter) {
                    const bool do_check =
                        (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
                    parallel_copy_vec(ggx, gx, threads);
                    packed_sweep(ggx.data(), order, nullptr);

                    IronsTuckStats stats;
                    irons_tuck_accumulate(cur.data(), gx.data(), ggx.data(), cur.size(), stats);
                    if (honest_mode &&
                        std::sqrt(stats.d1sq) <=
                            honest_tol * std::max(1.0, prev_norm)) {
                        // One full sweep moved the iterate by less than tol:
                        // ggx (= one extra sweep past gx) is the closest
                        // available iterate to the fixed point.
                        cur = ggx;
                        iter_used = iter + 1;
                        if (xhdfe_accel_debug()) {
                            std::fprintf(stderr,
                                         "[ACCEL packed CONVERGED iter_used=%d restarts=%d max_ratio=%.4f]\n",
                                         iter_used, accel_restart_count, max_ratio_seen);
                        }
                        return true;
                    }
                    if (stats.ssq == 0.0) {
                        cur = gx;
                        iter_used = iter + 1;
                        return true;
                    }
                    double coef = stats.vprod / stats.ssq;
                    if (accel_suspended) {
                        // Diverging: take the plain (non-accelerated, firmly
                        // non-expansive) sweep step instead of the IT step.
                        coef = 0.0;
                    }
                    irons_tuck_update_broadcast(cur.data(), gx.data(), ggx.data(), cur.size(), coef);

                    double sumsq = 0.0;
                    packed_sweep(gx.data(), order, do_check ? &sumsq : nullptr);
                    if (do_check) {
                        const double curr_norm = std::sqrt(sumsq);
                        const double denom = std::max(1.0, prev_norm);
                        const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                        const int step = iter - last_check_iter;
                        last_check_iter = iter;
                        prev_norm = curr_norm;
                        // Adaptive-restart divergence detection (honest_mode):
                        // resume on a new best, suspend on runaway growth.
                        if (honest_mode) {
                            const double ratio = curr_norm / std::max(best_resid, 1e-300);
                            if (ratio > max_ratio_seen) {
                                max_ratio_seen = ratio;
                            }
                            if (curr_norm <= best_resid) {
                                best_resid = curr_norm;
                                accel_suspended = false;
                            } else {
                                if (packed_cg_fallback_enabled &&
                                    curr_norm > cg_bail_factor * best_resid) {
                                    // Diverging on an ill-conditioned graph. Bail
                                    // out with the current post-sweep iterate (gx,
                                    // whose invariant M_D projection is the answer)
                                    // so the caller finishes with the stable
                                    // symmetric-operator CG instead of grinding
                                    // 10k-20k plain Gauss-Seidel sweeps.
                                    packed_accel_diverged = true;
                                    cur = gx;
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
                        if (!honest_mode && step > 0 &&
                            (rel_change / static_cast<double>(step)) < convergence_tol) {
                            cur = gx;
                            iter_used = iter + 1;
                            return true;
                        }
                    }
                    if (packed_cg_fallback_enabled && honest_mode && cg_iter_cap > 0 &&
                        !packed_accel_diverged && (iter + 1) >= cg_iter_cap) {
                        // Still grinding far past where any well-conditioned graph
                        // converges: hand the current post-sweep iterate to CG.
                        packed_accel_diverged = true;
                        cur = gx;
                        iter_used = iter + 1;
                        return false;
                    }
                    if (!accel_suspended && iter_grand_acc > 0 &&
                        ((iter + 1) % iter_grand_acc == 0)) {
                        ++grand_acc;
                        if (grand_acc == 1) {
                            acc_a = gx;
                        } else if (grand_acc == 2) {
                            acc_b = gx;
                        } else {
                            acc_c = gx;
                            IronsTuckStats acc_stats;
                            irons_tuck_accumulate(acc_a.data(), acc_b.data(), acc_c.data(),
                                                  acc_a.size(), acc_stats);
                            if (acc_stats.ssq != 0.0) {
                                const double acc_coef = acc_stats.vprod / acc_stats.ssq;
                                irons_tuck_update(acc_a.data(), acc_b.data(), acc_c.data(),
                                                  acc_a.size(), acc_coef);
                            }
                            cur = acc_a;
                            parallel_copy_vec(gx, cur, threads);
                            packed_sweep(gx.data(), order, nullptr);
                            grand_acc = 0;
                        }
                    }
                }
                if (xhdfe_accel_debug()) {
                    std::fprintf(stderr,
                                 "[ACCEL packed EXIT not-converged iter_used=%d restarts=%d best_resid=%.6e final_norm=%.6e max_ratio=%.4f]\n",
                                 max_iter, accel_restart_count, best_resid, prev_norm, max_ratio_seen);
                }
                cur = gx;
                iter_used = max_iter;
                return false;
            };

            // Stable per-column conjugate gradient over the symmetric (I - G)
            // demean operator. Used both as the explicit reghdfe/pathlike CG
            // accelerator (gate below) and as the divergence fallback when the
            // Irons-Tuck safeguard fires on an ill-conditioned graph. CG
            // re-derives its residual from `cur`, so it is safe to invoke on any
            // (firmly non-expansive) iterate the accelerator hands off.
            auto run_packed_cg = [&](const std::vector<std::size_t>& cg_order) {
                use_symmetric = true;

                auto projection_sweep = [&](const Eigen::VectorXd& src,
                                            Eigen::VectorXd& proj) {
                    parallel_copy_vec(proj, src, threads);
                    packed_sweep(proj.data(), cg_order, nullptr);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                    for (Eigen::Index i = 0; i < proj.size(); ++i) {
                        proj[i] = src[i] - proj[i];
                    }
                };
                auto safe_ratio = [](double num, double den) {
                    constexpr double eps = std::numeric_limits<double>::epsilon();
                    return std::abs(den) <= eps ? 0.0 : num / den;
                };

                Eigen::VectorXd improvement = packed_col_sumsq(cur);
                Eigen::VectorXd r(static_cast<Eigen::Index>(flat_size));
                projection_sweep(cur, r);
                Eigen::VectorXd ssr = packed_col_sumsq(r);
                Eigen::VectorXd u = r;
                Eigen::VectorXd v(static_cast<Eigen::Index>(flat_size));
                Eigen::VectorXd recent = Eigen::VectorXd::Zero(Sp);

                converged = false;
                int cg_iters = 0;
                for (int iter = 1; iter <= options.max_iter; ++iter) {
                    projection_sweep(u, v);
                    const Eigen::VectorXd denom = packed_col_dot(u, v);
                    Eigen::VectorXd alpha(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        alpha[k] = safe_ratio(ssr[k], denom[k]);
                        recent[k] = alpha[k] * ssr[k];
                        improvement[k] -= recent[k];
                    }

                    packed_axpy_columns(cur, u, alpha);
                    packed_axpy_columns(r, v, alpha);

                    const Eigen::VectorXd ssr_old = ssr;
                    ssr = packed_col_sumsq(r);
                    Eigen::VectorXd beta(Sp);
                    for (int k = 0; k < Sp; ++k) {
                        beta[k] = safe_ratio(ssr[k], ssr_old[k]);
                    }

#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                    for (Eigen::Index i = 0; i < n; ++i) {
                        const std::size_t off =
                            static_cast<std::size_t>(i) * static_cast<std::size_t>(Sp);
                        for (int k = 0; k < Sp; ++k) {
                            u[off + static_cast<std::size_t>(k)] =
                                r[off + static_cast<std::size_t>(k)] +
                                beta[k] * u[off + static_cast<std::size_t>(k)];
                        }
                    }

                    double update_error = 0.0;
                    for (int k = 0; k < Sp; ++k) {
                        constexpr double eps_floor = 1e-15;
                        const double num = std::max(0.0, recent[k]);
                        const double den = std::max(std::abs(improvement[k]),
                                                    std::numeric_limits<double>::epsilon());
                        const double err = std::sqrt(num < eps_floor ? 0.0 : num / den);
                        update_error = std::max(update_error, err);
                    }
                    cg_iters = iter;
                    if (update_error <= options.tol) {
                        converged = true;
                        break;
                    }
                }
                result.iterations = cg_iters > 0 ? cg_iters : options.max_iter;
                if (xhdfe_accel_debug()) {
                    std::fprintf(stderr, "[ACCEL packed CG iters=%d converged=%d]\n",
                                 result.iterations, converged ? 1 : 0);
                }
            };

            const bool use_packed_cg =
                should_use_reghdfe_cg_accelerator(indexers, weights, options) ||
                should_use_auto_default_reghdfe_cg_accelerator(
                    fe_profiles, n, weights, options, store_alphas) ||
                should_use_pathlike_cg_accelerator(indexers, n, total_levels_for_exact_sparse,
                                                   weights, options);
            if (use_packed_cg && !sweep_order.empty()) {
                run_packed_cg(sweep_order);
            } else {
                int remaining = accel_iter_budget;
                int total_iters = 0;
                converged = false;
                if (iter_warmup > 0 && remaining > 0) {
                    int phase_iters = 0;
                    converged = packed_accel_phase(sweep_order, std::min(iter_warmup, remaining),
                                                   phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && !packed_accel_diverged && !two_fe_order.empty() &&
                    iter_two_fe > 0 && remaining > 0) {
                    int phase_iters = 0;
                    packed_accel_phase(two_fe_order, std::min(iter_two_fe, remaining), phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && !packed_accel_diverged && remaining > 0) {
                    int phase_iters = 0;
                    converged = packed_accel_phase(sweep_order, remaining, phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && packed_accel_diverged && !sweep_order.empty()) {
                    // Irons-Tuck diverged (safeguard fired). Finish with the
                    // stable symmetric-operator CG rather than plain Gauss-Seidel.
                    run_packed_cg(sweep_order);
                } else if (converged) {
                    result.iterations = total_iters;
                }
            }

            // Tightening polish executed in the packed layout. The SoA polish
            // block below would otherwise re-run these sweeps through the much
            // slower SoA kernels (the packed path leaves the SoA workspace
            // means stale, so that block is forced for packed solves). The
            // sweeps, per-group reduction order, sweep order, tolerance, and
            // stop rule are identical to the SoA polish; only the memory
            // layout differs. Disable with XHDFE_PACKED_POLISH=0 to restore
            // the SoA polish behavior.
            static const bool packed_polish_disabled = []() {
                const char* e = std::getenv("XHDFE_PACKED_POLISH");
                return e != nullptr && e[0] == '0';
            }();
            // Only worthwhile for large inputs: on sub-second datasets the
            // SoA polish costs a few ms and interleaved A/B showed the packed
            // variant can be marginally slower there; below the threshold the
            // previous SoA polish path is kept byte-identical.
            constexpr int kPackedPolishMinObs = 4194304;
            if (!packed_polish_disabled && converged && n >= kPackedPolishMinObs &&
                !sweep_order.empty()) {
                const auto packed_polish_t0 = std::chrono::steady_clock::now();
                constexpr int kPolishSweeps = 6;
                const bool strict_tolerance = strict_residual_tolerance_mode(options);
                const int max_polish_sweeps =
                    strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                                     : kPolishSweeps;
                const double polish_tol =
                    strict_tolerance ? options.tol
                                     : limited_polish_tolerance(options);
                auto packed_max_abs_means = [&](std::size_t dim) {
                    const std::vector<double>& m = pmeans[dim];
                    double mx = 0.0;
                    for (const double v : m) {
                        mx = std::max(mx, std::abs(v));
                    }
                    return mx;
                };
                double final_max = 0.0;
                int polish_done = 0;
                for (int polish = 0; polish < max_polish_sweeps; ++polish) {
                    double max_abs = 0.0;
                    for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
                        const std::size_t dim = sweep_order[pos];
                        packed_demean_dim(cur.data(), dim, nullptr);
                        max_abs = std::max(max_abs, packed_max_abs_means(dim));
                    }
                    if (use_symmetric) {
                        for (std::size_t idx = sweep_order.size(); idx-- > 0;) {
                            const std::size_t dim = sweep_order[idx];
                            packed_demean_dim(cur.data(), dim, nullptr);
                            max_abs = std::max(max_abs, packed_max_abs_means(dim));
                        }
                    }
                    final_max = max_abs;
                    ++polish_done;
                    if (max_abs <= polish_tol) {
                        break;
                    }
                }
                packed_polish_ran = true;
                packed_polish_sweeps = polish_done;
                packed_polish_final_max = final_max;
                cpu_profile_log_elapsed("abs_polish_packed", packed_polish_t0);
            }

            // Unpack interleaved buffer back to [y | X(col-major)].
            {
                const double* __restrict src = cur.data();
                double* __restrict yp = result.y_tilde.data();
                double* __restrict Xp = result.X_tilde.data();
                const Eigen::Index nn = n;
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                for (Eigen::Index i = 0; i < nn; ++i) {
                    const double* __restrict s = src + static_cast<std::size_t>(i) * Sp;
                    yp[i] = s[0];
                    for (int j = 0; j < num_cols; ++j) {
                        Xp[static_cast<Eigen::Index>(j) * nn + i] = s[1 + j];
                    }
                }
            }
            packed_done = true;
        }
    }

    if (!packed_done) {
        const bool use_pathlike_cg =
            should_use_pathlike_cg_accelerator(indexers, n, total_levels_for_exact_sparse,
                                               weights, options);
        if (!enable_two_stage && use_accel && !use_pathlike_cg) {
        auto run_sweep = [&](Eigen::VectorXd& y_in,
                             Eigen::MatrixXd& X_in,
                             double* out_sumsq,
                             FeAlphaState* alpha_state) {
            double sumsq = 0.0;
            for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
                const std::size_t dim = sweep_order[pos];
                const int dim_threads = threads_by_dim[dim];
                const bool last_call = out_sumsq && !use_symmetric &&
                                       (pos + 1 == sweep_order.size());
                double* out_ptr = last_call ? &sumsq : nullptr;
                double* alpha_y_ptr = nullptr;
                double* alpha_x_ptr = nullptr;
                if (alpha_state && alpha_state->enabled) {
                    alpha_y_ptr = alpha_state->y[dim].data();
                    if (num_cols > 0) {
                        alpha_x_ptr = alpha_state->X[dim].data();
                    }
                }
                apply_demean(y_in, X_in, dim, out_ptr, alpha_y_ptr, alpha_x_ptr);
            }
            if (use_symmetric) {
                for (std::size_t idx = sweep_order.size(); idx-- > 0;) {
                    const std::size_t dim = sweep_order[idx];
                    const bool last_call = out_sumsq && (idx == 0);
                    double* out_ptr = last_call ? &sumsq : nullptr;
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (alpha_state && alpha_state->enabled) {
                        alpha_y_ptr = alpha_state->y[dim].data();
                        if (num_cols > 0) {
                            alpha_x_ptr = alpha_state->X[dim].data();
                        }
                    }
                    apply_demean(y_in, X_in, dim, out_ptr, alpha_y_ptr, alpha_x_ptr);
                }
            }
            if (out_sumsq) {
                *out_sumsq = sumsq;
            }
        };

	        const int check_interval = std::max(1, options.convergence_check_interval);
	        int last_check_iter = -1;
	        double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	        const int iter_grand_acc = 4;
        FeAlphaState alpha_gx;
        FeAlphaState alpha_ggx;
        FeAlphaState alpha_acc_a;
        FeAlphaState alpha_acc_b;
        FeAlphaState alpha_acc_c;
        if (store_alphas) {
            init_alpha_state(alpha_gx, indexers, num_cols);
            init_alpha_state(alpha_ggx, indexers, num_cols);
            if (iter_grand_acc > 0) {
                init_alpha_state(alpha_acc_a, indexers, num_cols);
                init_alpha_state(alpha_acc_b, indexers, num_cols);
                init_alpha_state(alpha_acc_c, indexers, num_cols);
            }
        }
        int grand_acc = 0;
        // Adaptive-restart safeguard against Irons-Tuck divergence on
        // ill-conditioned FE graphs (see packed_accel_phase above for the full
        // rationale). No-op on well-conditioned data (monotone iterate norm).
        constexpr double kAccelDivergeFactor = 4.0;
        double best_resid = prev_norm;
        bool accel_suspended = false;
        int accel_restart_count = 0;
        Eigen::VectorXd y_acc_a;
        Eigen::VectorXd y_acc_b;
        Eigen::VectorXd y_acc_c;
        Eigen::MatrixXd X_acc_a;
        Eigen::MatrixXd X_acc_b;
        Eigen::MatrixXd X_acc_c;
        if (iter_grand_acc > 0) {
            y_acc_a.resize(n);
            y_acc_b.resize(n);
            y_acc_c.resize(n);
            if (result.X_tilde.cols() > 0) {
                X_acc_a.resize(result.X_tilde.rows(), result.X_tilde.cols());
                X_acc_b.resize(result.X_tilde.rows(), result.X_tilde.cols());
                X_acc_c.resize(result.X_tilde.rows(), result.X_tilde.cols());
            }
        }

        Eigen::VectorXd y_gx = result.y_tilde;
        Eigen::MatrixXd X_gx = result.X_tilde;
        if (store_alphas) {
            copy_alpha(alpha_gx, alpha_result);
        }
        run_sweep(y_gx, X_gx, nullptr, store_alphas ? &alpha_gx : nullptr);

        Eigen::VectorXd y_ggx = y_gx;
        Eigen::MatrixXd X_ggx = X_gx;
        for (int iter = 0; iter < accel_iter_budget; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);

            parallel_copy_vec(y_ggx, y_gx, threads);
            parallel_copy_mat(X_ggx, X_gx, threads);
            if (store_alphas) {
                copy_alpha(alpha_ggx, alpha_gx);
            }
            run_sweep(y_ggx, X_ggx, nullptr, store_alphas ? &alpha_ggx : nullptr);

            IronsTuckStats stats;
            irons_tuck_accumulate(result.y_tilde.data(), y_gx.data(), y_ggx.data(), y_gx.size(),
                                  stats);
            if (X_gx.size() > 0) {
                irons_tuck_accumulate(result.X_tilde.data(), X_gx.data(), X_ggx.data(),
                                      X_gx.size(), stats);
            }

            if (honest_mode &&
                std::sqrt(stats.d1sq) <= honest_tol * std::max(1.0, prev_norm)) {
                result.y_tilde = std::move(y_ggx);
                result.X_tilde = std::move(X_ggx);
                if (store_alphas) {
                    copy_alpha(alpha_result, alpha_ggx);
                }
                result.iterations = iter + 1;
                converged = true;
                if (xhdfe_accel_debug()) {
                    std::fprintf(stderr, "[ACCEL soa CONVERGED iter_used=%d restarts=%d]\n",
                                 iter + 1, accel_restart_count);
                }
                break;
            }
            if (stats.ssq == 0.0) {
                result.y_tilde = std::move(y_gx);
                result.X_tilde = std::move(X_gx);
                if (store_alphas) {
                    copy_alpha(alpha_result, alpha_gx);
                }
                result.iterations = iter + 1;
                converged = true;
                break;
            }

            double coef = stats.vprod / stats.ssq;
            if (accel_suspended) {
                // Diverging: take the plain (firmly non-expansive) sweep step.
                coef = 0.0;
            }
            // Fused update + broadcast: writes new x into both result.y_tilde
            // and y_gx in a single pass, eliminating the follow-up copy.
            irons_tuck_update_broadcast(result.y_tilde.data(), y_gx.data(), y_ggx.data(),
                                        y_gx.size(), coef);
            if (X_gx.size() > 0) {
                irons_tuck_update_broadcast(result.X_tilde.data(), X_gx.data(), X_ggx.data(),
                                            X_gx.size(), coef);
            }
            if (store_alphas) {
                lincomb_alpha(alpha_result, alpha_gx, alpha_ggx, coef);
            }

            double sumsq = 0.0;
            if (store_alphas) {
                copy_alpha(alpha_gx, alpha_result);
            }
            run_sweep(y_gx, X_gx, do_check ? &sumsq : nullptr,
                      store_alphas ? &alpha_gx : nullptr);
            if (do_check) {
                const double curr_norm = std::sqrt(sumsq);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (honest_mode) {
                    if (curr_norm <= best_resid) {
                        best_resid = curr_norm;
                        accel_suspended = false;
                    } else if (curr_norm > kAccelDivergeFactor * best_resid) {
                        if (!accel_suspended) {
                            ++accel_restart_count;
                            grand_acc = 0;
                        }
                        accel_suspended = true;
                    }
                }
                if (!honest_mode && step > 0 &&
                    (rel_change / static_cast<double>(step)) < convergence_tol) {
                    result.y_tilde = std::move(y_gx);
                    result.X_tilde = std::move(X_gx);
                    if (store_alphas) {
                        copy_alpha(alpha_result, alpha_gx);
                    }
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
            if (!accel_suspended && iter_grand_acc > 0 && ((iter + 1) % iter_grand_acc == 0)) {
                ++grand_acc;
                if (grand_acc == 1) {
                    y_acc_a = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_a = X_gx;
                    }
                    if (store_alphas) {
                        copy_alpha(alpha_acc_a, alpha_gx);
                    }
                } else if (grand_acc == 2) {
                    y_acc_b = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_b = X_gx;
                    }
                    if (store_alphas) {
                        copy_alpha(alpha_acc_b, alpha_gx);
                    }
                } else {
                    y_acc_c = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_c = X_gx;
                    }
                    if (store_alphas) {
                        copy_alpha(alpha_acc_c, alpha_gx);
                    }
                    IronsTuckStats stats;
                    irons_tuck_accumulate(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                          y_acc_a.size(), stats);
                    if (X_gx.size() > 0) {
                        irons_tuck_accumulate(X_acc_a.data(), X_acc_b.data(), X_acc_c.data(),
                                              X_acc_a.size(), stats);
                    }
                    if (stats.ssq != 0.0) {
                        const double coef = stats.vprod / stats.ssq;
                        irons_tuck_update(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                          y_acc_a.size(), coef);
                        if (X_gx.size() > 0) {
                            irons_tuck_update(X_acc_a.data(), X_acc_b.data(), X_acc_c.data(),
                                              X_acc_a.size(), coef);
                        }
                        if (store_alphas) {
                            lincomb_alpha(alpha_acc_a, alpha_acc_b, alpha_acc_c, coef);
                        }
                    }
                    result.y_tilde = y_acc_a;
                    if (X_gx.size() > 0) {
                        result.X_tilde = X_acc_a;
                    }
                    if (store_alphas) {
                        copy_alpha(alpha_result, alpha_acc_a);
                    }
                    parallel_copy_vec(y_gx, result.y_tilde, threads);
                    parallel_copy_mat(X_gx, result.X_tilde, threads);
                    if (store_alphas) {
                        copy_alpha(alpha_gx, alpha_result);
                    }
                    run_sweep(y_gx, X_gx, nullptr, store_alphas ? &alpha_gx : nullptr);
                    grand_acc = 0;
                }
            }
        }
        if (!converged) {
            result.y_tilde = std::move(y_gx);
            result.X_tilde = std::move(X_gx);
            if (store_alphas) {
                copy_alpha(alpha_result, alpha_gx);
            }
        }
        } else if (!enable_two_stage && !use_pathlike_cg) {
	        double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	        const int check_interval = std::max(1, options.convergence_check_interval);
	        int last_check_iter = -1;
	        for (int iter = 0; iter < options.max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            double sumsq = 0.0;

            for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
                const std::size_t dim = sweep_order[pos];
                const bool last_call =
                    do_check && !use_symmetric && (pos + 1 == sweep_order.size());
                double* out_ptr = last_call ? &sumsq : nullptr;
                double* alpha_y_ptr = nullptr;
                double* alpha_x_ptr = nullptr;
                if (store_alphas && alpha_result.enabled) {
                    alpha_y_ptr = alpha_result.y[dim].data();
                    if (num_cols > 0) {
                        alpha_x_ptr = alpha_result.X[dim].data();
                    }
                }
                apply_demean(result.y_tilde, result.X_tilde, dim, out_ptr, alpha_y_ptr,
                             alpha_x_ptr);
            }
            if (use_symmetric) {
                for (std::size_t idx = sweep_order.size(); idx-- > 0;) {
                    const std::size_t dim = sweep_order[idx];
                    const bool last_call = do_check && (idx == 0);
                    double* out_ptr = last_call ? &sumsq : nullptr;
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas && alpha_result.enabled) {
                        alpha_y_ptr = alpha_result.y[dim].data();
                        if (num_cols > 0) {
                            alpha_x_ptr = alpha_result.X[dim].data();
                        }
                    }
                    apply_demean(result.y_tilde, result.X_tilde, dim, out_ptr, alpha_y_ptr,
                                 alpha_x_ptr);
                }
            }
            if (do_check) {
                const double curr_norm = std::sqrt(sumsq);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (step > 0 &&
                    (rel_change / static_cast<double>(step)) < convergence_tol) {
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
        }
    } else {
        auto run_sweep = [&](Eigen::VectorXd& y_in,
                             Eigen::MatrixXd& X_in,
                             const std::vector<std::size_t>& order,
                             double* out_sumsq,
                             FeAlphaState* alpha_state) {
            if (order.empty()) {
                if (out_sumsq) {
                    *out_sumsq = 0.0;
                }
                return;
            }
            double sumsq = 0.0;
            for (std::size_t pos = 0; pos < order.size(); ++pos) {
                const std::size_t dim = order[pos];
                const bool last_call =
                    out_sumsq && !use_symmetric && (pos + 1 == order.size());
                double* out_ptr = last_call ? &sumsq : nullptr;
                double* alpha_y_ptr = nullptr;
                double* alpha_x_ptr = nullptr;
                if (alpha_state && alpha_state->enabled) {
                    alpha_y_ptr = alpha_state->y[dim].data();
                    if (num_cols > 0) {
                        alpha_x_ptr = alpha_state->X[dim].data();
                    }
                }
                apply_demean(y_in, X_in, dim, out_ptr, alpha_y_ptr, alpha_x_ptr);
            }
            if (use_symmetric) {
                for (std::size_t idx = order.size(); idx-- > 0;) {
                    const std::size_t dim = order[idx];
                    const bool last_call = out_sumsq && (idx == 0);
                    double* out_ptr = last_call ? &sumsq : nullptr;
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (alpha_state && alpha_state->enabled) {
                        alpha_y_ptr = alpha_state->y[dim].data();
                        if (num_cols > 0) {
                            alpha_x_ptr = alpha_state->X[dim].data();
                        }
                    }
                    apply_demean(y_in, X_in, dim, out_ptr, alpha_y_ptr, alpha_x_ptr);
                }
            }
            if (out_sumsq) {
                *out_sumsq = sumsq;
            }
        };

        if (use_pathlike_cg) {
            const std::vector<std::size_t> cg_order{0, 1};
            use_symmetric = true;
            auto projection_sweep = [&](const Eigen::VectorXd& y_src,
                                        const Eigen::MatrixXd& X_src,
                                        Eigen::VectorXd& y_proj,
                                        Eigen::MatrixXd& X_proj) {
                y_proj = y_src;
                X_proj = X_src;
                run_sweep(y_proj, X_proj, cg_order, nullptr, nullptr);
                y_proj = y_src - y_proj;
                X_proj = X_src - X_proj;
            };
            auto col_sumsq = [&](const Eigen::VectorXd& yv,
                                 const Eigen::MatrixXd& Xm) {
                Eigen::VectorXd out(1 + Xm.cols());
                out[0] = yv.squaredNorm();
                for (int j = 0; j < Xm.cols(); ++j) {
                    out[1 + j] = Xm.col(j).squaredNorm();
                }
                return out;
            };
            auto col_dot = [&](const Eigen::VectorXd& ay,
                               const Eigen::MatrixXd& aX,
                               const Eigen::VectorXd& by,
                               const Eigen::MatrixXd& bX) {
                Eigen::VectorXd out(1 + aX.cols());
                out[0] = ay.dot(by);
                for (int j = 0; j < aX.cols(); ++j) {
                    out[1 + j] = aX.col(j).dot(bX.col(j));
                }
                return out;
            };
            auto safe_ratio = [](double num, double den) {
                constexpr double eps = std::numeric_limits<double>::epsilon();
                return std::abs(den) <= eps ? 0.0 : num / den;
            };

            Eigen::VectorXd improvement = col_sumsq(result.y_tilde, result.X_tilde);
            Eigen::VectorXd r_y;
            Eigen::MatrixXd r_X;
            projection_sweep(result.y_tilde, result.X_tilde, r_y, r_X);
            Eigen::VectorXd ssr = col_sumsq(r_y, r_X);
            Eigen::VectorXd u_y = r_y;
            Eigen::MatrixXd u_X = r_X;
            Eigen::VectorXd v_y;
            Eigen::MatrixXd v_X;
            Eigen::VectorXd recent = Eigen::VectorXd::Zero(ssr.size());

            bool cg_converged = false;
            int cg_iters = 0;
            for (int iter = 1; iter <= options.max_iter; ++iter) {
                projection_sweep(u_y, u_X, v_y, v_X);
                const Eigen::VectorXd denom = col_dot(u_y, u_X, v_y, v_X);
                Eigen::VectorXd alpha(ssr.size());
                for (Eigen::Index k = 0; k < ssr.size(); ++k) {
                    alpha[k] = safe_ratio(ssr[k], denom[k]);
                    recent[k] = alpha[k] * ssr[k];
                    improvement[k] -= recent[k];
                }

                result.y_tilde.noalias() -= alpha[0] * u_y;
                r_y.noalias() -= alpha[0] * v_y;
                for (int j = 0; j < result.X_tilde.cols(); ++j) {
                    result.X_tilde.col(j).noalias() -= alpha[1 + j] * u_X.col(j);
                    r_X.col(j).noalias() -= alpha[1 + j] * v_X.col(j);
                }

                const Eigen::VectorXd ssr_old = ssr;
                ssr = col_sumsq(r_y, r_X);
                Eigen::VectorXd beta(ssr.size());
                for (Eigen::Index k = 0; k < ssr.size(); ++k) {
                    beta[k] = safe_ratio(ssr[k], ssr_old[k]);
                }

                u_y = r_y + beta[0] * u_y;
                for (int j = 0; j < u_X.cols(); ++j) {
                    u_X.col(j) = r_X.col(j) + beta[1 + j] * u_X.col(j);
                }

                double update_error = 0.0;
                for (Eigen::Index k = 0; k < recent.size(); ++k) {
                    constexpr double eps_floor = 1e-15;
                    const double num = std::max(0.0, recent[k]);
                    const double den = std::max(std::abs(improvement[k]),
                                                std::numeric_limits<double>::epsilon());
                    const double err = std::sqrt(num < eps_floor ? 0.0 : num / den);
                    update_error = std::max(update_error, err);
                }
                cg_iters = iter;
                if (update_error <= options.tol) {
                    cg_converged = true;
                    break;
                }
            }

            result.iterations = cg_iters > 0 ? cg_iters : options.max_iter;
            result.converged = cg_converged;
            result.sweep_order_used.clear();
            result.sweep_order_used.reserve(cg_order.size());
            for (const std::size_t dim : cg_order) {
                result.sweep_order_used.push_back(static_cast<int>(dim));
            }
            return result;
        }

        const int check_interval = std::max(1, options.convergence_check_interval);
        int total_iters = 0;
        FeAlphaState* alpha_state = store_alphas ? &alpha_result : nullptr;

        if (use_accel) {
            const int iter_grand_acc = 4;
            Eigen::VectorXd y_acc_a;
            Eigen::VectorXd y_acc_b;
            Eigen::VectorXd y_acc_c;
            Eigen::MatrixXd X_acc_a;
            Eigen::MatrixXd X_acc_b;
            Eigen::MatrixXd X_acc_c;
            if (iter_grand_acc > 0) {
                y_acc_a.resize(n);
                y_acc_b.resize(n);
                y_acc_c.resize(n);
                if (result.X_tilde.cols() > 0) {
                    X_acc_a.resize(result.X_tilde.rows(), result.X_tilde.cols());
                    X_acc_b.resize(result.X_tilde.rows(), result.X_tilde.cols());
                    X_acc_c.resize(result.X_tilde.rows(), result.X_tilde.cols());
                }
            }

            Eigen::VectorXd y_gx = result.y_tilde;
            Eigen::VectorXd y_ggx = y_gx;
            Eigen::MatrixXd X_gx = result.X_tilde;
            Eigen::MatrixXd X_ggx = X_gx;

            auto run_accel_phase = [&](const std::vector<std::size_t>& order,
                                       int max_iter,
                                       int& iter_used,
                                       FeAlphaState* alpha_state) -> bool {
                iter_used = 0;
                if (max_iter <= 0 || order.empty()) {
                    return false;
	                }
	                int last_check_iter = -1;
	                double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	                int grand_acc = 0;
                const bool track_alpha = store_alphas && alpha_state && alpha_state->enabled;
                FeAlphaState alpha_gx;
                FeAlphaState alpha_ggx;
                FeAlphaState alpha_acc_a;
                FeAlphaState alpha_acc_b;
                FeAlphaState alpha_acc_c;
                if (track_alpha) {
                    init_alpha_state(alpha_gx, indexers, num_cols);
                    init_alpha_state(alpha_ggx, indexers, num_cols);
                    if (iter_grand_acc > 0) {
                        init_alpha_state(alpha_acc_a, indexers, num_cols);
                        init_alpha_state(alpha_acc_b, indexers, num_cols);
                        init_alpha_state(alpha_acc_c, indexers, num_cols);
                    }
                }

                parallel_copy_vec(y_gx, result.y_tilde, threads);
                parallel_copy_mat(X_gx, result.X_tilde, threads);
                if (track_alpha) {
                    copy_alpha(alpha_gx, *alpha_state);
                }
                run_sweep(y_gx, X_gx, order, nullptr, track_alpha ? &alpha_gx : nullptr);

                for (int iter = 0; iter < max_iter; ++iter) {
                    const bool do_check =
                        (check_interval == 1 || iter < check_interval || iter % check_interval == 0);

                    parallel_copy_vec(y_ggx, y_gx, threads);
                    parallel_copy_mat(X_ggx, X_gx, threads);
                    if (track_alpha) {
                        copy_alpha(alpha_ggx, alpha_gx);
                    }
                    run_sweep(y_ggx, X_ggx, order, nullptr,
                              track_alpha ? &alpha_ggx : nullptr);

                    IronsTuckStats stats;
                    irons_tuck_accumulate(result.y_tilde.data(), y_gx.data(), y_ggx.data(),
                                          y_gx.size(), stats);
                    if (X_gx.size() > 0) {
                        irons_tuck_accumulate(result.X_tilde.data(), X_gx.data(), X_ggx.data(),
                                              X_gx.size(), stats);
                    }

                    if (honest_mode &&
                        std::sqrt(stats.d1sq) <=
                            honest_tol * std::max(1.0, prev_norm)) {
                        result.y_tilde = y_ggx;
                        if (X_ggx.size() > 0) {
                            result.X_tilde = X_ggx;
                        }
                        if (track_alpha) {
                            copy_alpha(*alpha_state, alpha_ggx);
                        }
                        iter_used = iter + 1;
                        return true;
                    }
                    if (stats.ssq == 0.0) {
                        result.y_tilde = y_gx;
                        if (X_gx.size() > 0) {
                            result.X_tilde = X_gx;
                        }
                        if (track_alpha) {
                            copy_alpha(*alpha_state, alpha_gx);
                        }
                        iter_used = iter + 1;
                        return true;
                    }

                    const double coef = stats.vprod / stats.ssq;
                    // Fused update + broadcast: writes new x into both
                    // result.y_tilde and y_gx in a single pass, eliminating the
                    // subsequent copy.
                    irons_tuck_update_broadcast(result.y_tilde.data(), y_gx.data(), y_ggx.data(),
                                                y_gx.size(), coef);
                    if (X_gx.size() > 0) {
                        irons_tuck_update_broadcast(result.X_tilde.data(), X_gx.data(),
                                                    X_ggx.data(), X_gx.size(), coef);
                    }
                    if (track_alpha) {
                        lincomb_alpha(*alpha_state, alpha_gx, alpha_ggx, coef);
                    }

                    double sumsq = 0.0;
                    if (track_alpha) {
                        copy_alpha(alpha_gx, *alpha_state);
                    }
                    run_sweep(y_gx, X_gx, order, do_check ? &sumsq : nullptr,
                              track_alpha ? &alpha_gx : nullptr);
                    if (do_check) {
                        const double curr_norm = std::sqrt(sumsq);
                        const double denom = std::max(1.0, prev_norm);
                        const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                        const int step = iter - last_check_iter;
                        last_check_iter = iter;
                        prev_norm = curr_norm;
                        if (!honest_mode && step > 0 &&
                            (rel_change / static_cast<double>(step)) < convergence_tol) {
                            result.y_tilde = y_gx;
                            if (X_gx.size() > 0) {
                                result.X_tilde = X_gx;
                            }
                            if (track_alpha) {
                                copy_alpha(*alpha_state, alpha_gx);
                            }
                            iter_used = iter + 1;
                            return true;
                        }
                    }

                    if (iter_grand_acc > 0 && ((iter + 1) % iter_grand_acc == 0)) {
                        ++grand_acc;
                        if (grand_acc == 1) {
                            y_acc_a = y_gx;
                            if (X_gx.size() > 0) {
                                X_acc_a = X_gx;
                            }
                            if (track_alpha) {
                                copy_alpha(alpha_acc_a, alpha_gx);
                            }
                        } else if (grand_acc == 2) {
                            y_acc_b = y_gx;
                            if (X_gx.size() > 0) {
                                X_acc_b = X_gx;
                            }
                            if (track_alpha) {
                                copy_alpha(alpha_acc_b, alpha_gx);
                            }
                        } else {
                            y_acc_c = y_gx;
                            if (X_gx.size() > 0) {
                                X_acc_c = X_gx;
                            }
                            if (track_alpha) {
                                copy_alpha(alpha_acc_c, alpha_gx);
                            }
                            IronsTuckStats acc_stats;
                            irons_tuck_accumulate(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                                  y_acc_a.size(), acc_stats);
                            if (X_gx.size() > 0) {
                                irons_tuck_accumulate(X_acc_a.data(), X_acc_b.data(),
                                                      X_acc_c.data(), X_acc_a.size(), acc_stats);
                            }
                            if (acc_stats.ssq != 0.0) {
                                const double acc_coef = acc_stats.vprod / acc_stats.ssq;
                                irons_tuck_update(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                                  y_acc_a.size(), acc_coef);
                                if (X_gx.size() > 0) {
                                    irons_tuck_update(X_acc_a.data(), X_acc_b.data(),
                                                      X_acc_c.data(), X_acc_a.size(), acc_coef);
                                }
                                if (track_alpha) {
                                    lincomb_alpha(alpha_acc_a, alpha_acc_b, alpha_acc_c, acc_coef);
                                }
                            }
                            result.y_tilde = y_acc_a;
                            if (X_gx.size() > 0) {
                                result.X_tilde = X_acc_a;
                            }
                            if (track_alpha) {
                                copy_alpha(*alpha_state, alpha_acc_a);
                            }
                            parallel_copy_vec(y_gx, result.y_tilde, threads);
                            parallel_copy_mat(X_gx, result.X_tilde, threads);
                            if (track_alpha) {
                                copy_alpha(alpha_gx, *alpha_state);
                            }
                            run_sweep(y_gx, X_gx, order, nullptr,
                                      track_alpha ? &alpha_gx : nullptr);
                            grand_acc = 0;
                        }
                    }
                }

                result.y_tilde = y_gx;
                if (X_gx.size() > 0) {
                    result.X_tilde = X_gx;
                }
                if (track_alpha) {
                    copy_alpha(*alpha_state, alpha_gx);
                }
                iter_used = max_iter;
                return false;
            };

            int remaining = options.max_iter;
            if (iter_warmup > 0 && remaining > 0) {
                int phase_iters = 0;
                converged =
                    run_accel_phase(sweep_order, std::min(iter_warmup, remaining), phase_iters,
                                    alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
            if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                int phase_iters = 0;
                run_accel_phase(two_fe_order, std::min(iter_two_fe, remaining), phase_iters,
                                alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
            if (!converged && remaining > 0) {
                int phase_iters = 0;
                converged = run_accel_phase(sweep_order, remaining, phase_iters, alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
        } else {
            auto run_simple_phase = [&](const std::vector<std::size_t>& order,
                                        int max_iter,
                                        int& iter_used,
                                        FeAlphaState* alpha_state) -> bool {
                iter_used = 0;
                if (max_iter <= 0 || order.empty()) {
                    return false;
	                }
	                double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	                int last_check_iter = -1;
	                for (int iter = 0; iter < max_iter; ++iter) {
                    const bool do_check =
                        (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
                    double sumsq = 0.0;
                    run_sweep(result.y_tilde, result.X_tilde, order,
                              do_check ? &sumsq : nullptr, alpha_state);
                    if (do_check) {
                        const double curr_norm = std::sqrt(sumsq);
                        const double denom = std::max(1.0, prev_norm);
                        const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                        const int step = iter - last_check_iter;
                        last_check_iter = iter;
                        prev_norm = curr_norm;
                        if (step > 0 &&
                            (rel_change / static_cast<double>(step)) < convergence_tol) {
                            iter_used = iter + 1;
                            return true;
                        }
                    }
                }
                iter_used = max_iter;
                return false;
            };

            int remaining = options.max_iter;
            if (iter_warmup > 0 && remaining > 0) {
                int phase_iters = 0;
                converged =
                    run_simple_phase(sweep_order, std::min(iter_warmup, remaining), phase_iters,
                                     alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
            if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                int phase_iters = 0;
                run_simple_phase(two_fe_order, std::min(iter_two_fe, remaining), phase_iters,
                                 alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
            if (!converged && remaining > 0) {
                int phase_iters = 0;
                converged = run_simple_phase(sweep_order, remaining, phase_iters, alpha_state);
                total_iters += phase_iters;
                remaining -= phase_iters;
            }
        }

        if (converged) {
            result.iterations = total_iters;
        }
    }
    }  // if (!packed_done)
    cpu_profile_log_elapsed("abs_solve", solve_t0);

    if (!converged) {
        result.iterations = options.max_iter;
        result.converged = false;
    } else {
        result.converged = true;
        if (result.iterations == 0) {
            result.iterations = 1;
        }
    }

    const auto polish_t0 = std::chrono::steady_clock::now();
    if (result.converged && !sweep_order.empty()) {
        constexpr int kPolishSweeps = 6;
        const bool strict_tolerance = strict_residual_tolerance_mode(options);
        const int max_polish_sweeps =
            strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                             : kPolishSweeps;
        const double polish_tol =
            strict_tolerance ? options.tol
                             : limited_polish_tolerance(options);
        double initial_max = 0.0;
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            initial_max = std::max(initial_max, max_abs_means(workspaces[dim]));
        }
        double final_max = initial_max;
        // The packed fast path solves on its own buffers and leaves the SoA
        // workspace group means stale, so initial_max would read as 0. Force the
        // tightening polish to run (as it always does on the SoA path) to keep
        // the absorbed result equivalent to the SoA path within tolerance.
        // When the packed solve already ran its own polish (same sweeps and
        // stop rule, packed layout), only the bookkeeping is applied here.
        if (packed_polish_ran) {
            final_max = packed_polish_final_max;
            if (strict_tolerance) {
                result.iterations += packed_polish_sweeps;
            }
        } else if ((initial_max > polish_tol || packed_done) && max_polish_sweeps > 0) {
            const std::vector<std::size_t>& polish_order = sweep_order;
            int polish_done = 0;
            for (int polish = 0; polish < max_polish_sweeps; ++polish) {
                double max_abs = 0.0;
                for (std::size_t pos = 0; pos < polish_order.size(); ++pos) {
                    const std::size_t dim = polish_order[pos];
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas && alpha_result.enabled) {
                        alpha_y_ptr = alpha_result.y[dim].data();
                        if (num_cols > 0) {
                            alpha_x_ptr = alpha_result.X[dim].data();
                        }
                    }
                    apply_demean(result.y_tilde, result.X_tilde, dim, nullptr, alpha_y_ptr,
                                 alpha_x_ptr);
                    max_abs = std::max(max_abs, max_abs_means(workspaces[dim]));
                }
                if (use_symmetric) {
                    for (std::size_t idx = polish_order.size(); idx-- > 0;) {
                        const std::size_t dim = polish_order[idx];
                        double* alpha_y_ptr = nullptr;
                        double* alpha_x_ptr = nullptr;
                        if (store_alphas && alpha_result.enabled) {
                            alpha_y_ptr = alpha_result.y[dim].data();
                            if (num_cols > 0) {
                                alpha_x_ptr = alpha_result.X[dim].data();
                            }
                        }
                        apply_demean(result.y_tilde, result.X_tilde, dim, nullptr, alpha_y_ptr,
                                     alpha_x_ptr);
                        max_abs = std::max(max_abs, max_abs_means(workspaces[dim]));
                    }
                }
                final_max = max_abs;
                ++polish_done;
                if (max_abs <= polish_tol) {
                    break;
                }
            }
            if (strict_tolerance) {
                result.iterations += polish_done;
            }
        }
        if (strict_tolerance && final_max > polish_tol) {
            result.converged = false;
        }
    }
    cpu_profile_log_elapsed("abs_polish", polish_t0);

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.fe_group_ids[dim] = indexers[dim].group_ids;
        }
        result.fe_weight_sums.resize(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.fe_weight_sums[dim] = workspaces[dim].weight_sums_const;
        }
        result.sweep_order_used.clear();
        result.sweep_order_used.reserve(sweep_order.size());
        for (const std::size_t dim : sweep_order) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }
        if (store_alphas && alpha_result.enabled) {
            result.fe_alpha_y = alpha_result.y;
            result.fe_alpha_X = alpha_result.X;
            result.fe_means.clear();
        } else {
            result.fe_means.resize(indexers.size());
            for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                result.fe_means[dim] = workspaces[dim].y_sums;
            }
        }
    }

    cpu_profile_log_elapsed("abs_total", absorb_total_t0);
    if (cpu_profile_enabled()) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            std::cerr << "cpu_profile label=abs_dim"
                      << " dim=" << dim
                      << " groups=" << indexers[dim].num_groups
                      << " contiguous=" << (indexers[dim].groups_contiguous ? 1 : 0)
                      << " csr=" << static_cast<int>(use_csr[dim])
                      << " threads=" << threads_by_dim[dim]
                      << " calls=" << profile_dim_calls[dim]
                      << " ms=" << profile_dim_ms[dim] << '\n';
        }
    }
    return result;
}

AbsorptionResult absorb_fixed_effects_v6_mixed(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionMethod method,
    const std::vector<HeterogeneousSlopeTerm>& slopes) {
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }

    if (fes.empty()) {
        AbsorptionResult result;
        result.y_tilde = y;
        result.X_tilde = X;
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    AbsorptionResult result;
    result.y_tilde = y;
    result.X_tilde = X;

    std::vector<FeIndexer> indexers;
    indexers.reserve(fes.size());
    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        indexers.push_back(build_indexer(raw));
        result.fe_levels.push_back(indexers.back().num_levels_present);
    }

    const std::vector<const HeterogeneousSlopeTerm*> slope_lookup =
        build_slope_lookup(slopes, indexers.size(), n);

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    }
    threads = std::max(1, threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();
    const int num_cols = static_cast<int>(result.X_tilde.cols());

    std::vector<FeWorkspace> workspaces;
    workspaces.reserve(indexers.size());
    std::vector<HeteroSlopeWorkspace> slope_workspaces(indexers.size());
    std::vector<int> threads_by_dim(indexers.size(), threads);
    std::vector<uint8_t> is_slope(indexers.size(), 0);

    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        const FeIndexer& idx = indexers[dim];
        workspaces.emplace_back(idx.num_groups, num_cols);
        if (slope_lookup[dim]) {
            is_slope[dim] = 1;
            slope_workspaces[dim] =
                prepare_slope_workspace(idx, *slope_lookup[dim], weight_ptr, unit_weights, n,
                                        num_cols, threads_by_dim[dim]);
        } else {
            workspaces[dim].weight_sums_const =
                compute_weight_sums(idx, weight_ptr, unit_weights, n, threads_by_dim[dim]);
            workspaces[dim].refresh_weight_sums_inv();
            workspaces[dim].sparse_reset =
                (threads_by_dim[dim] == 1 && idx.num_groups <= 2000000 &&
                 idx.num_groups > (n / 4));
        }
    }

    std::vector<FeProfileStats> fe_profiles;
    fe_profiles.reserve(indexers.size());
    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        if (is_slope[dim]) {
            Eigen::VectorXd counts =
                compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, 1);
            fe_profiles.push_back(
                profile_fe(indexers[dim], counts, unit_weights));
        } else {
            fe_profiles.push_back(
                profile_fe(indexers[dim], workspaces[dim].weight_sums_const, unit_weights));
        }
    }

    std::vector<std::size_t> sweep_order;
    if (indexers.size() > 2) {
        sweep_order = order_by_profile(fe_profiles);
    } else {
        sweep_order.resize(indexers.size());
        std::iota(sweep_order.begin(), sweep_order.end(), 0);
    }
    if (!options.sweep_order_override.empty() &&
        options.sweep_order_override.size() == sweep_order.size()) {
        std::vector<uint8_t> seen(sweep_order.size(), 0);
        bool valid = true;
        std::vector<std::size_t> override_order;
        override_order.reserve(options.sweep_order_override.size());
        for (const int v : options.sweep_order_override) {
            if (v < 0 || v >= static_cast<int>(sweep_order.size()) ||
                seen[static_cast<std::size_t>(v)]) {
                valid = false;
                break;
            }
            seen[static_cast<std::size_t>(v)] = 1;
            override_order.push_back(static_cast<std::size_t>(v));
        }
        if (valid) {
            sweep_order.swap(override_order);
        }
    }

    AbsorptionMethod selected =
        choose_absorption_method(indexers.size(), threads, options, method);
    if (selected == AbsorptionMethod::Jacobi) {
        selected = indexers.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                       : AbsorptionMethod::GaussSeidel;
    }
    const bool use_symmetric =
        (selected == AbsorptionMethod::SymmetricGaussSeidel || options.symmetric_sweep) &&
        sweep_order.size() > 1;
    const bool use_accel = indexers.size() >= 2 && n >= 200000;
    const int check_interval = std::max(1, options.convergence_check_interval);

    const GpuBackend gpu_backend = resolve_gpu_backend();
    const bool gpu_cuda = (gpu_backend == GpuBackend::Cuda && cuda_backend_available());
    const bool gpu_available = gpu_cuda;
    if (gpu_backend_requested(gpu_backend) && !gpu_available) {
        result.converged = false;
        mark_gpu_unavailable(result);
        return result;
    }

    if (gpu_cuda) {
        std::vector<Eigen::VectorXd> weight_sums;
        weight_sums.reserve(indexers.size());
        for (const auto& idx : indexers) {
            weight_sums.push_back(
                compute_weight_sums(idx, weight_ptr, unit_weights, n, 1));
        }

        std::vector<GpuFeInput> fe_inputs;
        fe_inputs.reserve(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            GpuFeInput input;
            input.group_ids = indexers[dim].group_ids.data();
            input.num_groups = indexers[dim].num_groups;
            input.num_levels_present = indexers[dim].num_levels_present;
            input.weight_sums = weight_sums[dim].data();
            if (is_slope[dim]) {
                input.is_slope = true;
                input.slope_has_intercept = slope_lookup[dim]->include_intercept;
                input.slope_values = slope_lookup[dim]->values.data();
                input.slope_sum_z = slope_workspaces[dim].sum_z.data();
                input.slope_sum_zz = slope_workspaces[dim].sum_zz.data();
            }
            fe_inputs.push_back(input);
        }

        HdfeOptions passthrough = options;
        passthrough.symmetric_sweep = use_symmetric;
        passthrough.absorption_method = selected;

        AbsorptionResult gpu_result;
        const bool ok = absorb_fixed_effects_cuda(result.y_tilde, result.X_tilde, fe_inputs,
                                                  weights, sweep_order, passthrough, selected,
                                                  gpu_result);
        if (ok && gpu_result.converged) {
            gpu_result.gpu_used = true;
            gpu_result.gpu_status_code = 1;
            gpu_result.gpu_attempted = true;
            gpu_result.gpu_absorption_converged = true;
            gpu_result.gpu_absorption_iterations = gpu_result.iterations;
            return gpu_result;
        }
        if (gpu_backend_requested(gpu_backend)) {
            result.gpu_used = false;
            result.gpu_status_code = ok ? 3 : 4;
            result.gpu_attempted = true;
            result.gpu_absorption_converged = ok && gpu_result.converged;
            result.gpu_absorption_iterations = gpu_result.iterations;
            result.converged = false;
            return result;
        }
    }

    // Non-contiguous high-cardinality plain dims take the same CSR demean the
    // plain absorber dispatches to; the generic TLS scatter path pays
    // O(threads x groups x cols) dense accumulators + reduction per apply.
    std::vector<GroupCSR> plain_csr(indexers.size());
    std::vector<uint8_t> plain_use_csr(indexers.size(), 0);
    if (n >= 200000 && num_cols <= 8) {
        constexpr int kCsrGroupsThreshold = 100000;
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            if (!is_slope[dim] && threads_by_dim[dim] > 1 &&
                indexers[dim].num_groups >= kCsrGroupsThreshold &&
                !indexer_has_contiguous_groups(indexers[dim], n)) {
                plain_use_csr[dim] = 1;
                plain_csr[dim] = build_group_csr(indexers[dim].group_ids.data(), n,
                                                 indexers[dim].num_groups);
            }
        }
    }

    // Fuse slope dims that share one FE (identical group ids) into a single
    // joint per-group projection (CPU sweeps only; the GPU path above keeps
    // its per-term kernels and has already returned when it converged).
    std::vector<int> fused_unit_of(indexers.size(), -1);
    std::vector<FusedSlopeWorkspace> fused_units;
    if (mixed_fuse_slopes_enabled()) {
        for (std::size_t d1 = 0; d1 < indexers.size(); ++d1) {
            if (!is_slope[d1] || fused_unit_of[d1] >= 0) {
                continue;
            }
            std::vector<std::size_t> members{d1};
            for (std::size_t d2 = d1 + 1; d2 < indexers.size(); ++d2) {
                if (!is_slope[d2] || fused_unit_of[d2] >= 0) {
                    continue;
                }
                if (indexers[d2].num_groups == indexers[d1].num_groups &&
                    indexers[d2].group_ids == indexers[d1].group_ids) {
                    members.push_back(d2);
                }
            }
            const int fused_basis =
                static_cast<int>(members.size()) +
                (std::any_of(members.begin(), members.end(),
                             [&](std::size_t d) {
                                 return slope_lookup[d]->include_intercept;
                             })
                     ? 1
                     : 0);
            if (members.size() < 2 || fused_basis > 8) {
                continue;
            }
            const int unit = static_cast<int>(fused_units.size());
            for (const std::size_t d : members) {
                fused_unit_of[d] = unit;
            }
            fused_units.push_back(prepare_fused_slope_workspace(
                members, slope_lookup, slope_workspaces[d1].csr,
                indexers[d1].num_groups, weight_ptr, unit_weights, num_cols,
                threads_by_dim[d1]));
        }
    }
    // The fused unit is applied at its first position in the sweep order;
    // remaining member positions are skipped.
    std::vector<uint8_t> sweep_skip(indexers.size(), 0);
    if (!fused_units.empty()) {
        std::vector<uint8_t> unit_seen(fused_units.size(), 0);
        for (const std::size_t dim : sweep_order) {
            const int fu = fused_unit_of[dim];
            if (fu < 0) {
                continue;
            }
            if (unit_seen[static_cast<std::size_t>(fu)]) {
                sweep_skip[dim] = 1;
            } else {
                unit_seen[static_cast<std::size_t>(fu)] = 1;
            }
        }
    }
    std::size_t sweep_first_applied = 0;
    std::size_t sweep_last_applied = sweep_order.empty() ? 0 : sweep_order.size() - 1;
    for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
        if (!sweep_skip[sweep_order[pos]]) {
            sweep_first_applied = pos;
            break;
        }
    }
    for (std::size_t pos = sweep_order.size(); pos-- > 0;) {
        if (!sweep_skip[sweep_order[pos]]) {
            sweep_last_applied = pos;
            break;
        }
    }

    auto apply_dim = [&](Eigen::VectorXd& y_in,
                         Eigen::MatrixXd& X_in,
                         std::size_t dim,
                         double* out_sumsq) {
        if (is_slope[dim]) {
            const int fu = fused_unit_of[dim];
            if (fu >= 0) {
                fused_slope_project_inplace(y_in, X_in,
                                            fused_units[static_cast<std::size_t>(fu)],
                                            weight_ptr, unit_weights,
                                            threads_by_dim[dim], true, out_sumsq);
                return;
            }
            slope_project_inplace(y_in, X_in, *slope_lookup[dim], slope_workspaces[dim],
                                  weight_ptr, unit_weights, threads_by_dim[dim], true,
                                  out_sumsq);
        } else if (plain_use_csr[dim]) {
            demean_inplace_csr(y_in, X_in, plain_csr[dim], workspaces[dim], weight_ptr,
                               unit_weights, threads_by_dim[dim], true, out_sumsq, 1.0,
                               nullptr, nullptr, nullptr);
        } else {
            demean_inplace(y_in, X_in, indexers[dim], workspaces[dim], weight_ptr, unit_weights,
                           threads_by_dim[dim], true, out_sumsq, 1.0, nullptr, nullptr,
                           nullptr);
        }
    };

    auto run_sweep = [&](Eigen::VectorXd& y_in,
                         Eigen::MatrixXd& X_in,
                         double* out_sumsq) {
        double sumsq = 0.0;
        for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
            const std::size_t dim = sweep_order[pos];
            if (sweep_skip[dim]) {
                continue;
            }
            const bool last_call = out_sumsq && !use_symmetric &&
                                   (pos == sweep_last_applied);
            apply_dim(y_in, X_in, dim, last_call ? &sumsq : nullptr);
        }
        if (use_symmetric) {
            for (std::size_t pos = sweep_order.size(); pos-- > 0;) {
                const std::size_t dim = sweep_order[pos];
                if (sweep_skip[dim]) {
                    continue;
                }
                const bool last_call = out_sumsq && (pos == sweep_first_applied);
                apply_dim(y_in, X_in, dim, last_call ? &sumsq : nullptr);
            }
        }
        if (out_sumsq) {
            *out_sumsq = sumsq;
        }
    };

    bool converged = false;
    if (use_accel) {
        double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
        int last_check_iter = -1;
        const ConvergenceCriterion mixed_criterion =
            resolved_mixed_convergence_criterion(options);
        const bool use_update_error = needs_reghdfe_update_check(mixed_criterion);
        Eigen::VectorXd y_prev_update;
        Eigen::MatrixXd X_prev_update;
        if (use_update_error) {
            y_prev_update = result.y_tilde;
            X_prev_update = result.X_tilde;
        }
        constexpr int kMixedGrandAccelInterval = 4;
        int grand_acc = 0;
        Eigen::VectorXd y_acc_a;
        Eigen::VectorXd y_acc_b;
        Eigen::VectorXd y_acc_c;
        Eigen::MatrixXd X_acc_a;
        Eigen::MatrixXd X_acc_b;
        Eigen::MatrixXd X_acc_c;
        if (kMixedGrandAccelInterval > 0) {
            y_acc_a.resize(n);
            y_acc_b.resize(n);
            y_acc_c.resize(n);
            if (result.X_tilde.size() > 0) {
                X_acc_a.resize(result.X_tilde.rows(), result.X_tilde.cols());
                X_acc_b.resize(result.X_tilde.rows(), result.X_tilde.cols());
                X_acc_c.resize(result.X_tilde.rows(), result.X_tilde.cols());
            }
        }
        Eigen::VectorXd y_gx = result.y_tilde;
        Eigen::MatrixXd X_gx = result.X_tilde;
        Eigen::VectorXd y_ggx = y_gx;
        Eigen::MatrixXd X_ggx = X_gx;
        run_sweep(y_gx, X_gx, nullptr);

        for (int iter = 0; iter < options.max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            parallel_copy_vec(y_ggx, y_gx, threads);
            parallel_copy_mat(X_ggx, X_gx, threads);
            run_sweep(y_ggx, X_ggx, nullptr);

            IronsTuckStats stats;
            irons_tuck_accumulate(result.y_tilde.data(), y_gx.data(), y_ggx.data(),
                                  y_gx.size(), stats);
            if (result.X_tilde.size() > 0) {
                irons_tuck_accumulate(result.X_tilde.data(), X_gx.data(), X_ggx.data(),
                                      X_gx.size(), stats);
            }
            if (!(stats.ssq > 0.0) || !std::isfinite(stats.ssq)) {
                result.y_tilde = y_gx;
                result.X_tilde = X_gx;
                result.iterations = iter + 1;
                converged = true;
                break;
            }

            const double coef = stats.vprod / stats.ssq;
            irons_tuck_update_broadcast(result.y_tilde.data(), y_gx.data(), y_ggx.data(),
                                        y_gx.size(), coef);
            if (result.X_tilde.size() > 0) {
                irons_tuck_update_broadcast(result.X_tilde.data(), X_gx.data(), X_ggx.data(),
                                            X_gx.size(), coef);
            }

            double sumsq = 0.0;
            run_sweep(y_gx, X_gx, do_check ? &sumsq : nullptr);
            if (do_check) {
                const double curr_norm = std::sqrt(sumsq);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                double update_error = std::numeric_limits<double>::max();
                if (use_update_error) {
                    update_error = mean_reldif_update_error(y_gx, y_prev_update, X_gx,
                                                            X_prev_update, weights, threads);
                    parallel_copy_vec(y_prev_update, y_gx, threads);
                    parallel_copy_mat(X_prev_update, X_gx, threads);
                }
                const double norm_error = rel_change / static_cast<double>(step);
                if (step > 0 && convergence_reached(mixed_criterion, options.tol, norm_error,
                                                    update_error)) {
                    result.y_tilde = y_gx;
                    result.X_tilde = X_gx;
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
            if (kMixedGrandAccelInterval > 0 &&
                ((iter + 1) % kMixedGrandAccelInterval == 0)) {
                ++grand_acc;
                if (grand_acc == 1) {
                    y_acc_a = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_a = X_gx;
                    }
                } else if (grand_acc == 2) {
                    y_acc_b = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_b = X_gx;
                    }
                } else {
                    y_acc_c = y_gx;
                    if (X_gx.size() > 0) {
                        X_acc_c = X_gx;
                    }
                    IronsTuckStats acc_stats;
                    irons_tuck_accumulate(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                          y_acc_a.size(), acc_stats);
                    if (X_gx.size() > 0) {
                        irons_tuck_accumulate(X_acc_a.data(), X_acc_b.data(), X_acc_c.data(),
                                              X_acc_a.size(), acc_stats);
                    }
                    if (acc_stats.ssq != 0.0 && std::isfinite(acc_stats.ssq)) {
                        const double acc_coef = acc_stats.vprod / acc_stats.ssq;
                        irons_tuck_update(y_acc_a.data(), y_acc_b.data(), y_acc_c.data(),
                                          y_acc_a.size(), acc_coef);
                        if (X_gx.size() > 0) {
                            irons_tuck_update(X_acc_a.data(), X_acc_b.data(), X_acc_c.data(),
                                              X_acc_a.size(), acc_coef);
                        }
                    }
                    result.y_tilde = y_acc_a;
                    if (X_gx.size() > 0) {
                        result.X_tilde = X_acc_a;
                    }
                    parallel_copy_vec(y_gx, result.y_tilde, threads);
                    parallel_copy_mat(X_gx, result.X_tilde, threads);
                    run_sweep(y_gx, X_gx, nullptr);
                    grand_acc = 0;
                }
            }
        }
    } else {
	        double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	        int last_check_iter = -1;
	        const ConvergenceCriterion mixed_criterion =
	            resolved_mixed_convergence_criterion(options);
	        const bool use_update_error = needs_reghdfe_update_check(mixed_criterion);
	        Eigen::VectorXd y_prev_update;
	        Eigen::MatrixXd X_prev_update;
	        if (use_update_error) {
	            y_prev_update = result.y_tilde;
	            X_prev_update = result.X_tilde;
	        }
	        for (int iter = 0; iter < options.max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            double sumsq = 0.0;
            run_sweep(result.y_tilde, result.X_tilde, do_check ? &sumsq : nullptr);
            if (do_check) {
                const double curr_norm = std::sqrt(sumsq);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
	                const int step = iter - last_check_iter;
	                last_check_iter = iter;
	                prev_norm = curr_norm;
	                double update_error = std::numeric_limits<double>::max();
	                if (use_update_error) {
	                    update_error = mean_reldif_update_error(result.y_tilde, y_prev_update,
	                                                            result.X_tilde, X_prev_update,
	                                                            weights, threads);
	                    parallel_copy_vec(y_prev_update, result.y_tilde, threads);
	                    parallel_copy_mat(X_prev_update, result.X_tilde, threads);
	                }
	                const double norm_error = rel_change / static_cast<double>(step);
	                if (step > 0 && convergence_reached(mixed_criterion, options.tol, norm_error,
	                                                    update_error)) {
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

    if (result.converged && !sweep_order.empty()) {
        constexpr int kPolishSweeps = 3;
        for (int polish = 0; polish < kPolishSweeps; ++polish) {
            run_sweep(result.y_tilde, result.X_tilde, nullptr);
        }
    }

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        result.fe_weight_sums.resize(indexers.size());
        result.fe_means.resize(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.fe_group_ids[dim] = indexers[dim].group_ids;
            result.fe_weight_sums[dim] =
                compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, 1);
            if (!is_slope[dim]) {
                result.fe_means[dim] = workspaces[dim].y_sums;
            } else {
                result.fe_means[dim] = Eigen::VectorXd::Zero(indexers[dim].num_groups);
            }
        }
    }
    result.sweep_order_used.clear();
    result.sweep_order_used.reserve(sweep_order.size());
    for (const std::size_t dim : sweep_order) {
        result.sweep_order_used.push_back(static_cast<int>(dim));
    }

    return result;
}

AbsorptionResult absorb_fixed_effects_v6(const Eigen::Ref<const Eigen::VectorXd>& y,
                                         const Eigen::Ref<const Eigen::MatrixXd>& X,
                                         const std::vector<Eigen::VectorXi>& fes,
                                         const Eigen::VectorXd* weights,
                                         const HdfeOptions& options,
                                         AbsorptionMethod method,
                                         const std::vector<HeterogeneousSlopeTerm>& slopes) {
    if ((method == AbsorptionMethod::Lsmr || method == AbsorptionMethod::Mlsmr ||
         options.absorption_method == AbsorptionMethod::Lsmr ||
         options.absorption_method == AbsorptionMethod::Mlsmr) &&
        !slopes.empty()) {
        throw std::runtime_error(
            "absorptionmethod(lsmr/mlsmr) does not support heterogeneous slopes; "
            "use auto, gauss-seidel, or symmetric-gauss-seidel");
    }
    if (!slopes.empty()) {
        if (options.tolerance_mode == ToleranceMode::StrictResidual) {
            throw std::runtime_error(
                "tolerancemode(strict-residual) is not supported with heterogeneous "
                "slopes in absorb() yet; use reghdfe-comparable or xhdfe-fast, or an "
                "explicit convergence() criterion");
        }
        return absorb_fixed_effects_v6_mixed(y, X, fes, weights, options, method, slopes);
    }

    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }
    const double convergence_tol = effective_absorption_tolerance(options);

    if (fes.empty()) {
        AbsorptionResult result;
        result.y_tilde = y;
        result.X_tilde = X;
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    {
        const GpuBackend pre_gpu_backend = resolve_gpu_backend();
        const bool pre_gpu_available =
            (pre_gpu_backend == GpuBackend::Cuda && cuda_backend_available()) ||
            (pre_gpu_backend == GpuBackend::Metal && metal_backend_available());
        if (gpu_backend_requested(pre_gpu_backend) && !pre_gpu_available) {
            AbsorptionResult unavailable;
            unavailable.y_tilde = y;
            unavailable.X_tilde = X;
            unavailable.iterations = 0;
            unavailable.converged = false;
            mark_gpu_unavailable(unavailable);
            return unavailable;
        }

        // ---- Adaptive Schwarz auto-gate (Auto mode, CPU) ----
        // MUST run BEFORE the pre-selected MAP early-return below, otherwise the gate is
        // dead code on the production fit path (where `method`/options.absorption_method
        // arrive concrete, not Auto) and large ill-conditioned multi-way FE graphs
        // (difficult_1m / difficult_10m) silently fall back to MAP. It also runs after the
        // explicit-GPU-unavailable precheck above, so an explicit-but-unavailable GPU
        // request can never be served a CPU Schwarz result. Uses the cheap raw-FE
        // Gauss-Seidel probe (no FeIndexer build), so the MAP fall-through adds no duplicate
        // indexer cost on large graphs like ready/akm. The probe indexes its count/sum
        // arrays by raw FE id, so guard against non-compact ids: a negative id would index
        // below the array and a sparse-high id (e.g. 1e9 on a few levels) would force a
        // huge allocation. Valid compact MAP ids (including post-singleton gaps) satisfy
        // 0 <= id and maxid+1 <= 2n; anything else skips the gate and is handled by the
        // normal indexer-based MAP path, which normalizes arbitrary ids safely.
        if (options.from_auto &&
            weights == nullptr && !options.retain_fixed_effects &&
            fes.size() >= 2 && fes.size() <= 3 &&
            static_cast<int64_t>(n) >= schwarz_auto_nmin() &&
            (static_cast<int>(X.cols()) + 1) <= 16 &&
            !pre_gpu_available && !schwarz_env_disabled()) {
            bool probe_ids_ok = true;
            std::vector<const int*> gptr;
            std::vector<int> ng;
            gptr.reserve(fes.size());
            ng.reserve(fes.size());
            for (const auto& f : fes) {
                if (f.size() == 0) { probe_ids_ok = false; break; }
                const int lo = f.minCoeff();
                const int hi = f.maxCoeff();
                if (lo < 0 ||
                    static_cast<int64_t>(hi) + 1 > 2 * static_cast<int64_t>(n)) {
                    probe_ids_ok = false;
                    break;
                }
                gptr.push_back(f.data());
                ng.push_back(hi + 1);
            }
            if (probe_ids_ok) {
                const double eff_tol = effective_absorption_tolerance(options);
                const double proj =
                    schwarz_probe_projected_iters(gptr, ng, y.data(), n, eff_tol);
                const bool go = proj > schwarz_auto_mswitch();
                if (schwarz_diag_enabled()) {
                    std::fprintf(stderr,
                                 "[schwarz-gate] n=%d ndim=%zu proj_iters=%.0f mswitch=%.0f nmin=%lld -> %s\n",
                                 n, fes.size(), proj, schwarz_auto_mswitch(),
                                 static_cast<long long>(schwarz_auto_nmin()),
                                 go ? "SCHWARZ" : "MAP");
                }
                if (go) {
                    // First try matrix-free Jacobi-PCG. For well-conditioned graphs (simple
                    // DGP; 2-way worker+year) it converges in a handful of iterations and is
                    // far cheaper than the approx-Cholesky Schwarz preconditioner, whose fill
                    // explodes on these (simple 3-way 10M: ~190M factor nnz, ~100s vs ~12s
                    // here) -- and it produces the identical within transform (both solve the
                    // same M_D system to the same tol). The probe caps the leading y-solve;
                    // if PCG stalls (genuine high-mobility AKM-type graphs, e.g. difficult
                    // 3-way) it returns not-converged and we fall back to Schwarz after only
                    // ~probe wasted iterations. Disable with XHDFE_GATE_KRYLOV=0.
                    static const bool krylov_gate_disabled = []() {
                        const char* e = std::getenv("XHDFE_GATE_KRYLOV");
                        return e != nullptr && e[0] == '0';
                    }();
                    static const int krylov_probe = []() {
                        const char* e = std::getenv("XHDFE_KRYLOV_PROBE");
                        const int v = e ? std::atoi(e) : 32;
                        return v > 0 ? v : 32;
                    }();
                    if (!krylov_gate_disabled) {
                        HdfeOptions ok = options;
                        ok.use_krylov = true;
                        ok.krylov_probe_iters = krylov_probe;
                        AbsorptionResult kr =
                            absorb_fixed_effects_krylov(y, X, fes, weights, ok);
                        if (kr.converged) {
                            if (schwarz_diag_enabled()) {
                                std::fprintf(stderr,
                                    "[schwarz-gate] -> JACOBI-PCG (probe ok, it=%d)\n",
                                    kr.iterations);
                            }
                            return kr;
                        }
                    }
                    // Jacobi-PCG bailed: a genuine high-mobility AKM/difficult
                    // 3-way graph. On exactly this band the matrix-free MLSMR
                    // absorber beats the approx-Cholesky Schwarz preconditioner
                    // at every tested scale and mode (interleaved 5-rep medians,
                    // 29jun2026: difficult_1m 3fe -22%/-22%, difficult_10m 3fe
                    // -14%/-18% comparable/fast; coef parity ~1e-14 vs Schwarz --
                    // both solve the same M_D system to tol). This reuses the
                    // FULL-DATA gate (no 200k-sample contraction probe), so it
                    // cannot mis-promote the well-conditioned designs the sample
                    // probe trips on; and it only sees proj>mswitch graphs where
                    // Jacobi-PCG already failed, so PCG's simple-DGP wins (which
                    // return above) are untouched. Falls through to Schwarz if
                    // MLSMR does not converge. Kill with XHDFE_GATE_MLSMR_ON_PCG_BAIL=0.
                    static const bool mlsmr_on_pcg_bail_disabled = []() {
                        const char* e = std::getenv("XHDFE_GATE_MLSMR_ON_PCG_BAIL");
                        return e != nullptr && e[0] == '0';
                    }();
                    if (mlsmr_auto_gate_enabled() && !mlsmr_on_pcg_bail_disabled) {
                        HdfeOptions mo = options;
                        mo.absorption_method = AbsorptionMethod::Mlsmr;
                        AbsorptionResult mr =
                            absorb_fixed_effects_mlsmr(y, X, fes, weights, mo);
                        if (mr.converged) {
                            mr.mlsmr_used = true;
                            if (schwarz_diag_enabled()) {
                                std::fprintf(stderr,
                                    "[schwarz-gate] -> MLSMR (PCG-bail, proj=%.0f, it=%d)\n",
                                    proj, mr.iterations);
                            }
                            return mr;
                        }
                    }
                    HdfeOptions o2 = options;
                    o2.absorption_method = AbsorptionMethod::Schwarz;
                    return absorb_fixed_effects(y, X, fes, weights, o2);
                }
                // Reached only when go==false (proj <= mswitch): the moderate-proj MAP
                // band. On this band the matrix-free MLSMR absorber beats accelerated MAP
                // for large slow-converging worker-firm-year designs (AKM/ready). Gated to
                // reghdfe-comparable/strict and proj >= min; reuses the probe (no extra
                // cost). Falls through to MAP if MLSMR does not converge.
                if (mlsmr_auto_gate_enabled() &&
                    (options.tolerance_mode == ToleranceMode::ReghdfeComparable ||
                     options.tolerance_mode == ToleranceMode::StrictResidual) &&
                    proj >= mlsmr_auto_gate_min_proj()) {
                    HdfeOptions mo = options;
                    mo.absorption_method = AbsorptionMethod::Mlsmr;
                    AbsorptionResult mr =
                        absorb_fixed_effects_mlsmr(y, X, fes, weights, mo);
                    if (mr.converged) {
                        mr.mlsmr_used = true;
                        if (schwarz_diag_enabled()) {
                            std::fprintf(stderr,
                                         "[schwarz-gate] -> MLSMR (proj=%.0f, it=%d)\n",
                                         proj, mr.iterations);
                        }
                        return mr;
                    }
                }
            }
        }

        int pre_threads = options.num_threads > 0 ? options.num_threads : 1;
        pre_threads = std::max(1, pre_threads);
        const AbsorptionMethod pre_selected =
            choose_absorption_method(fes.size(), pre_threads, options, method);
        // Opt-in LSMR/MLSMR must be intercepted HERE, before the pre-selected MAP
        // early-return below routes every concrete sweep method through the standard
        // absorber (which has no Lsmr/Mlsmr handling and would silently run GS).
        if (pre_selected == AbsorptionMethod::Lsmr ||
            pre_selected == AbsorptionMethod::Mlsmr) {
            if (options.retain_fixed_effects) {
                throw std::runtime_error(
                    "absorptionmethod(lsmr/mlsmr) does not yet support savefe/retain_fixed_effects");
            }
            HdfeOptions passthrough = options;
            passthrough.absorption_method = pre_selected;
            AbsorptionResult cpu_result =
                absorb_fixed_effects_mlsmr(y, X, fes, weights, passthrough);
            if (gpu_backend_requested(pre_gpu_backend)) {
                mark_gpu_unavailable(cpu_result);
            }
            return cpu_result;
        }
        const bool allow_sparse_pre =
            options.use_sparse_solver ||
            (method == AbsorptionMethod::Auto &&
             options.absorption_method == AbsorptionMethod::Auto);
        if (pre_selected != AbsorptionMethod::Jacobi &&
            !pre_gpu_available &&
            !allow_sparse_pre &&
            !options.use_krylov) {
            HdfeOptions passthrough = options;
            passthrough.symmetric_sweep =
                (pre_selected == AbsorptionMethod::SymmetricGaussSeidel);
            passthrough.absorption_method = pre_selected;
            AbsorptionResult cpu_result = absorb_fixed_effects(y, X, fes, weights, passthrough);
            if (gpu_backend_requested(pre_gpu_backend)) {
                mark_gpu_unavailable(cpu_result);
            }
            return cpu_result;
        }
    }

    AbsorptionResult result;
    // result.y_tilde / result.X_tilde are filled lazily: the GPU paths only
    // read y/X as upload sources, so the full working copy is materialized
    // only on the paths that mutate it in place (CPU Jacobi) or return it
    // (GPU-requested failure). This avoids a serial multi-GB copy on the
    // GPU fast path.

    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
    }
    std::vector<FeIndexer> indexers(fes.size());
#ifdef HDFE_USE_OPENMP
    {
        int idx_threads = options.num_threads > 0
                              ? std::min(options.num_threads, omp_get_max_threads())
                              : omp_get_max_threads();
        idx_threads = std::min<int>(std::max(1, idx_threads),
                                    static_cast<int>(fes.size()));
        if (idx_threads > 1 && n >= 4194304) {
            // Dimensions are independent; each indexer is built and
            // first-touched by a single thread, bit-identical to serial.
#pragma omp parallel for schedule(dynamic, 1) num_threads(idx_threads)
            for (std::ptrdiff_t dim = 0; dim < static_cast<std::ptrdiff_t>(fes.size());
                 ++dim) {
                indexers[static_cast<std::size_t>(dim)] =
                    build_indexer(fes[static_cast<std::size_t>(dim)]);
            }
        } else {
            for (std::size_t dim = 0; dim < fes.size(); ++dim) {
                indexers[dim] = build_indexer(fes[dim]);
            }
        }
    }
#else
    for (std::size_t dim = 0; dim < fes.size(); ++dim) {
        indexers[dim] = build_indexer(fes[dim]);
    }
#endif
    int total_levels = 0;
    for (const auto& idx : indexers) {
        result.fe_levels.push_back(idx.num_levels_present);
        total_levels += idx.num_groups;
    }

    const GpuBackend gpu_backend = resolve_gpu_backend();
    const bool gpu_cuda = (gpu_backend == GpuBackend::Cuda && cuda_backend_available());
    const bool gpu_metal = (gpu_backend == GpuBackend::Metal && metal_backend_available());
    const bool gpu_available = gpu_cuda || gpu_metal;

    // NOTE: the adaptive Schwarz auto-gate is evaluated earlier, in the pre-dispatch block
    // above (before the pre-selected MAP early-return), so it stays reachable on the
    // production fit path. It is intentionally NOT repeated here.

    const bool allow_sparse =
        options.use_sparse_solver ||
        (method == AbsorptionMethod::Auto && options.absorption_method == AbsorptionMethod::Auto);
    bool use_sparse = false;
    if (allow_sparse) {
        if (options.use_sparse_solver) {
            use_sparse = true;
        } else if (options.sparse_threshold > 0.0) {
            use_sparse = static_cast<double>(total_levels) <= options.sparse_threshold;
        } else if (indexers.size() >= 2) {
            constexpr int kSparseLevelCap = 200000;
            constexpr double kSparseLevelRatio = 0.35;
            const double ratio = n > 0 ? static_cast<double>(total_levels) / n : 1.0;
            if (total_levels < n && total_levels < kSparseLevelCap &&
                ratio <= kSparseLevelRatio) {
                use_sparse = true;
            }
        }
        if (total_levels > 500000) {
            use_sparse = false;
        }
    }

    if (use_sparse && !gpu_available) {
        AbsorptionResult sparse_result =
            absorb_fixed_effects_sparse(y, X, indexers, weights, options, false);
        if (gpu_backend_requested(gpu_backend)) {
            mark_gpu_unavailable(sparse_result);
        }
        if (sparse_result.converged) {
            return sparse_result;
        }
    }

    if (options.use_krylov && !gpu_available) {
        AbsorptionResult krylov_result = absorb_fixed_effects_krylov(y, X, fes, weights, options);
        if (gpu_backend_requested(gpu_backend)) {
            mark_gpu_unavailable(krylov_result);
        }
        return krylov_result;
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    }
    threads = std::max(1, threads);
    if (!options.retain_fixed_effects) {
        // Keep the historical shape cap at the dispatch level: the GPU path's
        // CPU-side prep and the post-absorption phases of GPU rows showed a
        // small but consistent slowdown when uncapped under load (A/B
        // 10jun2026). The large-n uncap applies only inside the CPU solver
        // (absorb_fixed_effects), which re-derives its own thread count.
        threads = cap_threads_by_fe_shape(threads, n, indexers);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();

    const AbsorptionMethod selected =
        choose_absorption_method(fes.size(), threads, options, method);
    // ---- Opt-in LSMR / MLSMR absorber (explicit request only; CPU-only) ----
    // Reached only when the user explicitly selects absorptionmethod(lsmr|mlsmr)
    // (or the auto-mlsmr selector resolved to Mlsmr). choose_absorption_method()
    // never returns Lsmr/Mlsmr for an Auto request, so the default path cannot
    // enter here. Fails closed on savefe; slopes were rejected at function entry.
    if (selected == AbsorptionMethod::Lsmr || selected == AbsorptionMethod::Mlsmr) {
        if (options.retain_fixed_effects) {
            throw std::runtime_error(
                "absorptionmethod(lsmr/mlsmr) does not yet support savefe/retain_fixed_effects");
        }
        HdfeOptions passthrough = options;
        passthrough.absorption_method = selected;
        AbsorptionResult cpu_result =
            absorb_fixed_effects_mlsmr(y, X, fes, weights, passthrough);
        if (gpu_backend_requested(gpu_backend)) {
            mark_gpu_unavailable(cpu_result);
        }
        return cpu_result;
    }
    if (selected != AbsorptionMethod::Jacobi) {
        const bool use_cuda = gpu_cuda;
        const bool use_metal = gpu_metal;
        if (use_cuda || use_metal) {
            std::vector<Eigen::VectorXd> weight_sums(indexers.size());
#ifdef HDFE_USE_OPENMP
            if (indexers.size() > 1 && n >= 4194304 && threads > 1) {
                // Each dimension's weight-sum vector is computed and
                // first-touched by a single thread, so per-dimension results
                // are bit-identical to the serial build.
                const int ws_threads =
                    std::min<int>(threads, static_cast<int>(indexers.size()));
#pragma omp parallel for schedule(dynamic, 1) num_threads(ws_threads)
                for (std::ptrdiff_t dim = 0;
                     dim < static_cast<std::ptrdiff_t>(indexers.size()); ++dim) {
                    weight_sums[static_cast<std::size_t>(dim)] = compute_weight_sums(
                        indexers[static_cast<std::size_t>(dim)], weight_ptr,
                        unit_weights, n, 1);
                }
            } else
#endif
            {
                for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                    weight_sums[dim] =
                        compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, 1);
                }
            }

            std::vector<GpuFeInput> fe_inputs;
            fe_inputs.reserve(indexers.size());
            for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                GpuFeInput input;
                input.group_ids = indexers[dim].group_ids.data();
                input.num_groups = indexers[dim].num_groups;
                input.num_levels_present = indexers[dim].num_levels_present;
                input.weight_sums = weight_sums[dim].data();
                fe_inputs.push_back(input);
            }

            std::vector<FeProfileStats> fe_profiles;
            fe_profiles.reserve(indexers.size());
            for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                fe_profiles.push_back(profile_fe(indexers[dim], weight_sums[dim], unit_weights));
            }
            const std::vector<int>* override_order =
                options.sweep_order_override.empty() ? nullptr : &options.sweep_order_override;
            std::vector<std::size_t> sweep_order;
            if (!override_order && indexers.size() > 2) {
                sweep_order = order_by_profile(fe_profiles);
            } else {
                sweep_order = select_sweep_order(y, indexers, weight_ptr, unit_weights,
                                                 override_order);
            }

            HdfeOptions passthrough = options;
            passthrough.symmetric_sweep = (selected == AbsorptionMethod::SymmetricGaussSeidel);
            passthrough.absorption_method = selected;

            AbsorptionResult gpu_result;
            bool ok = false;
            if (use_cuda) {
                ok = absorb_fixed_effects_cuda(y, X, fe_inputs, weights,
                                               sweep_order, passthrough, selected, gpu_result);
            } else if (use_metal) {
                ok = absorb_fixed_effects_metal(y, X, fe_inputs, weights,
                                                sweep_order, passthrough, selected, gpu_result);
            }
            if (ok && gpu_result.converged) {
                gpu_result.gpu_used = true;
                gpu_result.gpu_status_code = 1;
                gpu_result.gpu_attempted = true;
                gpu_result.gpu_absorption_converged = true;
                gpu_result.gpu_absorption_iterations = gpu_result.iterations;
                return gpu_result;
            }
            result.gpu_status_code = ok ? 3 : 4;
            result.gpu_attempted = true;
            result.gpu_absorption_converged = gpu_result.converged;
            result.gpu_absorption_iterations = gpu_result.iterations;
            if (gpu_backend_requested(gpu_backend)) {
                result.converged = false;
                result.iterations = gpu_result.iterations;
                result.y_tilde = y;
                result.X_tilde = X;
                return result;
            }
        }

        HdfeOptions passthrough = options;
        passthrough.symmetric_sweep = (selected == AbsorptionMethod::SymmetricGaussSeidel);
        passthrough.absorption_method = selected;
        AbsorptionResult cpu_result = absorb_fixed_effects(y, X, fes, weights, passthrough);
        if (result.gpu_attempted) {
            mark_gpu_failure_fallback(cpu_result, result, result.gpu_status_code == 3);
        } else if (gpu_backend_requested(gpu_backend)) {
            mark_gpu_unavailable(cpu_result);
        }
        return cpu_result;
    }

    const bool use_cuda = gpu_cuda;
    const bool use_metal = gpu_metal;
    if (use_cuda || use_metal) {
        std::vector<Eigen::VectorXd> weight_sums;
        weight_sums.reserve(indexers.size());
        for (const auto& idx : indexers) {
            weight_sums.push_back(
                compute_weight_sums(idx, weight_ptr, unit_weights, n, 1));
        }

        std::vector<GpuFeInput> fe_inputs;
        fe_inputs.reserve(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            GpuFeInput input;
            input.group_ids = indexers[dim].group_ids.data();
            input.num_groups = indexers[dim].num_groups;
            input.num_levels_present = indexers[dim].num_levels_present;
            input.weight_sums = weight_sums[dim].data();
            fe_inputs.push_back(input);
        }

        HdfeOptions passthrough = options;
        passthrough.absorption_method = selected;

        AbsorptionResult gpu_result;
        bool ok = false;
        if (use_cuda) {
            ok = absorb_fixed_effects_cuda(y, X, fe_inputs, weights, {},
                                           passthrough, selected, gpu_result);
        } else if (use_metal) {
            ok = absorb_fixed_effects_metal(y, X, fe_inputs, weights, {},
                                            passthrough, selected, gpu_result);
        }
        if (ok && gpu_result.converged) {
            gpu_result.gpu_used = true;
            gpu_result.gpu_status_code = 1;
            gpu_result.gpu_attempted = true;
            gpu_result.gpu_absorption_converged = true;
            gpu_result.gpu_absorption_iterations = gpu_result.iterations;
            return gpu_result;
        }
        result.gpu_status_code = ok ? 3 : 4;
        result.gpu_attempted = true;
        result.gpu_absorption_converged = gpu_result.converged;
        result.gpu_absorption_iterations = gpu_result.iterations;
        if (gpu_backend_requested(gpu_backend)) {
            result.converged = false;
            result.iterations = gpu_result.iterations;
            result.y_tilde = y;
            result.X_tilde = X;
            return result;
        }
    }

    // CPU Jacobi mutates the working copy in place; materialize it here.
    result.y_tilde = y;
    result.X_tilde = X;

    std::vector<FeWorkspace> workspaces;
    workspaces.reserve(indexers.size());
    const int num_cols = static_cast<int>(result.X_tilde.cols());
    for (const auto& idx : indexers) {
        workspaces.emplace_back(idx.num_groups, num_cols);
    }
    const bool store_alphas =
        options.retain_fixed_effects &&
        options.fe_recovery_method == FeRecoveryMethod::Hybrid &&
        num_cols <= options.savefe_fastpath_max_cols;

    const bool sparse_reset = (threads == 1);
    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        workspaces[dim].weight_sums_const =
            compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, threads);
        workspaces[dim].refresh_weight_sums_inv();
        workspaces[dim].sparse_reset = sparse_reset;
    }

    double relaxation = options.jacobi_relaxation;
    if (relaxation <= 0.0) {
        relaxation = 2.0 / (static_cast<double>(indexers.size()) + 1.0);
    }
    relaxation = std::min(relaxation, 1.0);

	const int check_interval = 1;
	double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	int last_check_iter = -1;
	bool converged = false;
    FeAlphaState alpha_result;
    if (store_alphas) {
        init_alpha_state(alpha_result, indexers, num_cols);
    }

    for (int iter = 0; iter < options.max_iter; ++iter) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            double* alpha_y_ptr = nullptr;
            double* alpha_x_ptr = nullptr;
            if (store_alphas && alpha_result.enabled) {
                alpha_y_ptr = alpha_result.y[dim].data();
                if (num_cols > 0) {
                    alpha_x_ptr = alpha_result.X[dim].data();
                }
            }
            demean_inplace(result.y_tilde, result.X_tilde, indexers[dim], workspaces[dim],
                           weight_ptr, unit_weights, threads, false, nullptr, relaxation,
                           alpha_y_ptr, alpha_x_ptr);
        }

        apply_jacobi_update(result.y_tilde, result.X_tilde, indexers, workspaces, relaxation,
                            threads);

        if (check_interval == 1 || iter < check_interval || iter % check_interval == 0) {
            const double curr_norm = combined_norm(result.y_tilde, result.X_tilde);
            const double denom = std::max(1.0, prev_norm);
            const double rel_change = std::abs(curr_norm - prev_norm) / denom;
            const int step = iter - last_check_iter;
            last_check_iter = iter;
            prev_norm = curr_norm;

            if (step > 0 && (rel_change / static_cast<double>(step)) < convergence_tol) {
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

    if (result.converged && !indexers.empty()) {
        constexpr int kPolishSweeps = 6;
        const bool strict_tolerance = strict_residual_tolerance_mode(options);
        const int max_polish_sweeps =
            strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                             : kPolishSweeps;
        const double polish_tol =
            strict_tolerance ? options.tol
                             : limited_polish_tolerance(options);
        double initial_max = 0.0;
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            initial_max = std::max(initial_max, max_abs_means(workspaces[dim]));
        }
        double final_max = initial_max;
        if (initial_max > polish_tol && max_polish_sweeps > 0) {
            int polish_done = 0;
            for (int polish = 0; polish < max_polish_sweeps; ++polish) {
                double max_abs = 0.0;
                for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas && alpha_result.enabled) {
                        alpha_y_ptr = alpha_result.y[dim].data();
                        if (num_cols > 0) {
                            alpha_x_ptr = alpha_result.X[dim].data();
                        }
                    }
                    demean_inplace(result.y_tilde, result.X_tilde, indexers[dim], workspaces[dim],
                                   weight_ptr, unit_weights, threads, true, nullptr, 1.0,
                                   alpha_y_ptr, alpha_x_ptr);
                    max_abs = std::max(max_abs, max_abs_means(workspaces[dim]));
                }
                final_max = max_abs;
                ++polish_done;
                if (max_abs <= polish_tol) {
                    break;
                }
            }
            if (strict_tolerance) {
                result.iterations += polish_done;
            }
        }
        if (strict_tolerance && final_max > polish_tol) {
            result.converged = false;
        }
    }

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.fe_group_ids[dim] = indexers[dim].group_ids;
        }
        result.fe_weight_sums.resize(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.fe_weight_sums[dim] = workspaces[dim].weight_sums_const;
        }
        result.sweep_order_used.clear();
        result.sweep_order_used.reserve(indexers.size());
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }
        if (store_alphas && alpha_result.enabled) {
            result.fe_alpha_y = alpha_result.y;
            result.fe_alpha_X = alpha_result.X;
            result.fe_means.clear();
        } else {
            result.fe_means.resize(indexers.size());
            for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
                result.fe_means[dim] = workspaces[dim].y_sums;
            }
        }
    }

    if (!result.gpu_attempted && gpu_backend_requested(gpu_backend) && !gpu_available) {
        mark_gpu_unavailable(result);
    }

    return result;
}

namespace {

void demean_individual_inplace(Eigen::VectorXd& y,
                               Eigen::MatrixXd& X,
                               const hdfe::detail::GroupIndividualStructure& gi,
                               const Eigen::VectorXd& denom,
                               const double* weight_ptr,
                               bool unit_weights,
                               double* out_max_abs = nullptr) {
    const int individuals = gi.num_individuals;
    if (individuals <= 0) {
        if (out_max_abs) {
            *out_max_abs = 0.0;
        }
        return;
    }
    if (static_cast<int>(gi.individual_ptr.size()) != individuals + 1) {
        throw std::runtime_error("Invalid individual_ptr size for group/individual FE structure");
    }
    if (static_cast<int>(denom.size()) != individuals) {
        throw std::runtime_error("Invalid denom size for group/individual FE structure");
    }

    const int cols = static_cast<int>(X.cols());
    Eigen::RowVectorXd numer_x;
    if (cols > 0) {
        numer_x = Eigen::RowVectorXd::Zero(cols);
    }
    double max_abs = 0.0;

    for (int i = 0; i < individuals; ++i) {
        const double denom_i = denom(i);
        if (denom_i <= 0.0) {
            continue;
        }

        double numer_y = 0.0;
        if (cols > 0) {
            numer_x.setZero();
        }

        const int begin = gi.individual_ptr[static_cast<std::size_t>(i)];
        const int end = gi.individual_ptr[static_cast<std::size_t>(i + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int g = gi.individual_group[static_cast<std::size_t>(pos)];
            const double scale = gi.group_scale[static_cast<std::size_t>(g)];
            const double w = unit_weights ? 1.0 : weight_ptr[g];
            const double weight_scale = w * scale;
            numer_y += weight_scale * y(g);
            if (cols > 0) {
                numer_x.noalias() += weight_scale * X.row(g);
            }
        }

        const double alpha_y = numer_y / denom_i;
        Eigen::RowVectorXd alpha_x;
        if (cols > 0) {
            alpha_x = numer_x / denom_i;
        }
        max_abs = std::max(max_abs, std::abs(alpha_y));
        if (cols > 0 && alpha_x.size() > 0) {
            max_abs = std::max(max_abs, alpha_x.cwiseAbs().maxCoeff());
        }

        for (int pos = begin; pos < end; ++pos) {
            const int g = gi.individual_group[static_cast<std::size_t>(pos)];
            const double scale = gi.group_scale[static_cast<std::size_t>(g)];
            y(g) -= scale * alpha_y;
            if (cols > 0) {
                X.row(g).noalias() -= scale * alpha_x;
            }
        }
    }

    if (out_max_abs) {
        *out_max_abs = max_abs;
    }
}

}  // namespace

AbsorptionResult absorb_fixed_effects_group_individual(const Eigen::VectorXd& y,
                                                       const Eigen::MatrixXd& X,
                                                       const std::vector<Eigen::VectorXi>& standard_fes,
                                                       const GroupIndividualStructure& gi,
                                                       const Eigen::VectorXd* weights,
                                                       const HdfeOptions& options,
                                                       AbsorptionMethod method) {
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }
    if (gi.num_groups != n) {
        throw std::runtime_error("Group/individual structure must match the length of y");
    }
    if (gi.num_individuals <= 0) {
        throw std::runtime_error("Group/individual structure must have at least one individual");
    }
    if (static_cast<int>(gi.group_scale.size()) != n) {
        throw std::runtime_error("Invalid group_scale length for group/individual FE structure");
    }
    const double convergence_tol = group_individual_absorption_tolerance(options);

    AbsorptionResult result;
    result.y_tilde = y;
    result.X_tilde = X;

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    } else {
        threads = 1;
    }
    threads = std::max(1, threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    // Build standard FE indexers.
    std::vector<FeIndexer> indexers;
    indexers.reserve(standard_fes.size());
    for (const auto& raw : standard_fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        indexers.push_back(build_indexer(raw));
        result.fe_levels.push_back(indexers.back().num_levels_present);
    }
    result.fe_levels.push_back(gi.num_individuals);

    // Validate absorption method.
    const AbsorptionMethod selected =
        choose_absorption_method(standard_fes.size() + 1, threads, options, method);
    if (selected == AbsorptionMethod::Jacobi) {
        throw std::runtime_error("Jacobi absorption is not supported with group/individual FEs");
    }

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();

    const GpuBackend gpu_backend = resolve_gpu_backend();
    const bool gpu_cuda = (gpu_backend == GpuBackend::Cuda && cuda_backend_available());
    if (gpu_backend_requested(gpu_backend) && !gpu_cuda) {
        result.converged = false;
        mark_gpu_unavailable(result);
        return result;
    }
    if (gpu_cuda) {
        // For group()/individual() absorption, a CUDA backend can be substantially faster.
        // Fall back silently if the backend fails (OOM, missing device, etc.).
        std::vector<Eigen::VectorXd> weight_sums;
        weight_sums.reserve(indexers.size());
        std::vector<GpuFeInput> fe_inputs;
        fe_inputs.reserve(indexers.size());
        for (const auto& idx : indexers) {
            weight_sums.push_back(
                compute_weight_sums(idx, weight_ptr, unit_weights, n, 1));
            GpuFeInput input;
            input.group_ids = idx.group_ids.data();
            input.num_groups = idx.num_groups;
            input.num_levels_present = idx.num_levels_present;
            input.weight_sums = weight_sums.back().data();
            fe_inputs.push_back(input);
        }

        auto resolve_order = [&]() -> std::vector<std::size_t> {
            const std::size_t dims = indexers.size();
            std::vector<std::size_t> order;
            order.reserve(dims);
            if (options.sweep_order_override.size() == static_cast<int>(dims) && dims > 0) {
                std::vector<uint8_t> seen(dims, 0);
                bool valid = true;
                for (const int v : options.sweep_order_override) {
                    if (v < 0 || v >= static_cast<int>(dims)) {
                        valid = false;
                        break;
                    }
                    const std::size_t idx = static_cast<std::size_t>(v);
                    if (seen[idx]) {
                        valid = false;
                        break;
                    }
                    seen[idx] = 1;
                }
                if (valid) {
                    for (const int v : options.sweep_order_override) {
                        order.push_back(static_cast<std::size_t>(v));
                    }
                    return order;
                }
                order.clear();
            }
            for (std::size_t d = 0; d < dims; ++d) {
                order.push_back(d);
            }
            return order;
        };

        const std::vector<std::size_t> order = resolve_order();
        HdfeOptions gpu_opts = options;
        // Keep all tolerances and iteration caps identical; only the backend changes.
        const bool ok = absorb_fixed_effects_group_individual_cuda(
            y, X, fe_inputs, gi, weights, order, gpu_opts, selected, result);
        if (ok && result.converged) {
            result.gpu_used = true;
            result.gpu_status_code = 1;
            result.gpu_attempted = true;
            result.gpu_absorption_converged = true;
            result.gpu_absorption_iterations = result.iterations;
            return result;
        }
        result.gpu_used = false;
        result.gpu_status_code = ok ? 3 : 4;
        result.gpu_attempted = true;
        result.gpu_absorption_converged = ok && result.converged;
        result.gpu_absorption_iterations = result.iterations;
        if (gpu_backend_requested(gpu_backend)) {
            result.converged = false;
            return result;
        }
    }

    std::vector<FeWorkspace> workspaces;
    workspaces.reserve(indexers.size());
    const int num_cols = static_cast<int>(result.X_tilde.cols());
    for (const auto& idx : indexers) {
        workspaces.emplace_back(idx.num_groups, num_cols);
    }

    std::vector<int> threads_by_dim(indexers.size(), threads);
#ifdef HDFE_USE_OPENMP
    if (threads > 2) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            const int groups = indexers[dim].num_groups;
            if (groups >= 10000000) {
                threads_by_dim[dim] = std::min(threads_by_dim[dim], 2);
            }
        }
    }
    if (threads > 1) {
        for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
            threads_by_dim[dim] =
                limit_threads_by_tls(threads_by_dim[dim], indexers[dim].num_groups, num_cols);
        }
    }
#endif

    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        workspaces[dim].weight_sums_const =
            compute_weight_sums(indexers[dim], weight_ptr, unit_weights, n, threads_by_dim[dim]);
        workspaces[dim].refresh_weight_sums_inv();
        workspaces[dim].sparse_reset =
            (threads_by_dim[dim] == 1 && indexers[dim].num_groups <= 2000000);
    }

    std::vector<FeProfileStats> fe_profiles;
    fe_profiles.reserve(indexers.size());
    for (std::size_t dim = 0; dim < indexers.size(); ++dim) {
        fe_profiles.push_back(
            profile_fe(indexers[dim], workspaces[dim].weight_sums_const, unit_weights));
    }
    std::vector<std::size_t> sweep_order;
    if (indexers.size() > 2) {
        sweep_order = order_by_profile(fe_profiles);
    } else {
        sweep_order =
            select_sweep_order(result.y_tilde, indexers, weight_ptr, unit_weights, nullptr);
    }

    // Precompute denominators for individual updates: sum_g w_g * scale_g^2.
    Eigen::VectorXd denom = Eigen::VectorXd::Zero(gi.num_individuals);
    for (int i = 0; i < gi.num_individuals; ++i) {
        const int begin = gi.individual_ptr[static_cast<std::size_t>(i)];
        const int end = gi.individual_ptr[static_cast<std::size_t>(i + 1)];
        double acc = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int g = gi.individual_group[static_cast<std::size_t>(pos)];
            const double scale = gi.group_scale[static_cast<std::size_t>(g)];
            const double w = unit_weights ? 1.0 : weight_ptr[g];
            acc += w * scale * scale;
        }
        denom(i) = acc > 0.0 ? acc : 1.0;
    }

    const bool use_symmetric = (selected == AbsorptionMethod::SymmetricGaussSeidel) &&
                               ((sweep_order.size() + 1) > 1);
	const int check_interval = std::max(1, options.convergence_check_interval);
	double prev_norm = combined_norm(result.y_tilde, result.X_tilde);
	int last_check_iter = -1;
	bool converged = false;

    for (int iter = 0; iter < options.max_iter; ++iter) {
        for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
            const std::size_t dim = sweep_order[pos];
            const int dim_threads = threads_by_dim[dim];
            demean_inplace(result.y_tilde, result.X_tilde, indexers[dim], workspaces[dim],
                           weight_ptr, unit_weights, dim_threads);
        }
        demean_individual_inplace(result.y_tilde, result.X_tilde, gi, denom, weight_ptr,
                                  unit_weights);

        if (use_symmetric) {
            demean_individual_inplace(result.y_tilde, result.X_tilde, gi, denom, weight_ptr,
                                      unit_weights);
            for (std::size_t idx = sweep_order.size(); idx-- > 0;) {
                const std::size_t dim = sweep_order[idx];
                const int dim_threads = threads_by_dim[dim];
                demean_inplace(result.y_tilde, result.X_tilde, indexers[dim], workspaces[dim],
                               weight_ptr, unit_weights, dim_threads);
            }
        }

        if (check_interval == 1 || iter < check_interval || iter % check_interval == 0) {
            const double curr_norm = combined_norm(result.y_tilde, result.X_tilde);
            const double denom_norm = std::max(1.0, prev_norm);
            const double rel_change = std::abs(curr_norm - prev_norm) / denom_norm;
            const int step = iter - last_check_iter;
            last_check_iter = iter;
            prev_norm = curr_norm;
            if (step > 0 && (rel_change / static_cast<double>(step)) < convergence_tol) {
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

    if (result.converged) {
        constexpr int kPolishSweeps = 16;
        const bool strict_tolerance = strict_residual_tolerance_mode(options);
        const int max_polish_sweeps =
            strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                             : kPolishSweeps;
        const double polish_tol =
            strict_tolerance ? options.tol
                             : ((limited_polish_tolerance(options) > 0.0)
                                    ? std::min(limited_polish_tolerance(options), 1.0e-14)
                                    : 1.0e-14);
        double final_max = strict_tolerance ? std::numeric_limits<double>::infinity() : 0.0;
        int polish_done = 0;
        for (int polish = 0; polish < max_polish_sweeps; ++polish) {
            double max_abs = 0.0;
            for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
                const std::size_t dim = sweep_order[pos];
                const int dim_threads = threads_by_dim[dim];
                demean_inplace(result.y_tilde, result.X_tilde, indexers[dim], workspaces[dim],
                               weight_ptr, unit_weights, dim_threads);
                max_abs = std::max(max_abs, max_abs_means(workspaces[dim]));
            }
            double individual_max = 0.0;
            demean_individual_inplace(result.y_tilde, result.X_tilde, gi, denom, weight_ptr,
                                      unit_weights, &individual_max);
            max_abs = std::max(max_abs, individual_max);

            if (use_symmetric) {
                individual_max = 0.0;
                demean_individual_inplace(result.y_tilde, result.X_tilde, gi, denom, weight_ptr,
                                          unit_weights, &individual_max);
                max_abs = std::max(max_abs, individual_max);
                for (std::size_t idx = sweep_order.size(); idx-- > 0;) {
                    const std::size_t dim = sweep_order[idx];
                    const int dim_threads = threads_by_dim[dim];
                    demean_inplace(result.y_tilde, result.X_tilde, indexers[dim],
                                   workspaces[dim], weight_ptr, unit_weights, dim_threads);
                    max_abs = std::max(max_abs, max_abs_means(workspaces[dim]));
                }
            }

            final_max = max_abs;
            ++polish_done;
            if (max_abs <= polish_tol) {
                break;
            }
        }
        if (strict_tolerance) {
            result.iterations += polish_done;
            if (final_max > polish_tol) {
                result.converged = false;
            }
        }
    }

    result.sweep_order_used.clear();
    result.sweep_order_used.reserve(sweep_order.size());
    for (const std::size_t dim : sweep_order) {
        result.sweep_order_used.push_back(static_cast<int>(dim));
    }

    return result;
}

AbsorptionResult absorb_fixed_effects_krylov(const Eigen::VectorXd& y,
                                             const Eigen::MatrixXd& X,
                                             const std::vector<Eigen::VectorXi>& fes,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options) {
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }

    AbsorptionResult result;
    result.y_tilde = y;
    result.X_tilde = X;

    if (fes.empty()) {
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    }
    threads = std::max(1, threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    std::vector<FeIndexer> indexers;
    indexers.reserve(fes.size());
    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        indexers.push_back(build_indexer(raw));
        result.fe_levels.push_back(indexers.back().num_levels_present);
    }

    const double* weight_ptr = weights ? weights->data() : nullptr;
    const bool unit_weights = (weights == nullptr);

    const int dims = static_cast<int>(indexers.size());
    std::vector<int> offsets(static_cast<std::size_t>(dims) + 1, 0);
    for (int d = 0; d < dims; ++d) {
        offsets[static_cast<std::size_t>(d + 1)] =
            offsets[static_cast<std::size_t>(d)] + indexers[static_cast<std::size_t>(d)].num_groups;
    }
    const int total_fe = offsets.back();

    const double lambda = std::max(0.0, options.krylov_lambda);
    Eigen::VectorXd diagA(total_fe);
    for (int d = 0; d < dims; ++d) {
        const auto& idx = indexers[static_cast<std::size_t>(d)];
        Eigen::VectorXd counts =
            compute_weight_sums(idx, weight_ptr, unit_weights, n, threads);
        counts.array() += lambda;
        for (Eigen::Index g = 0; g < counts.size(); ++g) {
            if (counts[g] <= 0.0) {
                counts[g] = lambda > 0.0 ? lambda : 1.0;
            }
        }
        diagA.segment(offsets[static_cast<std::size_t>(d)], idx.num_groups) = counts;
    }

    std::vector<const int*> gid_ptrs(static_cast<std::size_t>(dims));
    std::vector<int> groups_per_dim(static_cast<std::size_t>(dims));
    for (int d = 0; d < dims; ++d) {
        gid_ptrs[static_cast<std::size_t>(d)] =
            indexers[static_cast<std::size_t>(d)].group_ids.data();
        groups_per_dim[static_cast<std::size_t>(d)] =
            indexers[static_cast<std::size_t>(d)].num_groups;
    }

    std::vector<std::vector<Eigen::VectorXd>> tls_out;
#ifdef HDFE_USE_OPENMP
    std::vector<std::vector<Eigen::VectorXd>> tls_zt;
    std::vector<std::vector<double*>> tls_out_ptrs;
#endif
#ifdef HDFE_USE_OPENMP
    if (threads > 1) {
        tls_out.resize(static_cast<std::size_t>(dims));
        tls_zt.resize(static_cast<std::size_t>(dims));
        for (int d = 0; d < dims; ++d) {
            tls_out[static_cast<std::size_t>(d)].assign(
                static_cast<std::size_t>(threads),
                Eigen::VectorXd::Zero(groups_per_dim[static_cast<std::size_t>(d)]));
            tls_zt[static_cast<std::size_t>(d)].assign(
                static_cast<std::size_t>(threads),
                Eigen::VectorXd::Zero(groups_per_dim[static_cast<std::size_t>(d)]));
        }
        tls_out_ptrs.assign(static_cast<std::size_t>(threads),
                            std::vector<double*>(static_cast<std::size_t>(dims)));
        for (int t = 0; t < threads; ++t) {
            for (int d = 0; d < dims; ++d) {
                tls_out_ptrs[static_cast<std::size_t>(t)][static_cast<std::size_t>(d)] =
                    tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)].data();
            }
        }
    }
#endif
    std::vector<const double*> x_ptrs(static_cast<std::size_t>(dims));
    std::vector<double*> out_ptrs(static_cast<std::size_t>(dims));

    auto compute_Zt = [&](const Eigen::Ref<const Eigen::VectorXd>& v) {
        Eigen::VectorXd b(total_fe);
        b.setZero();
        const double* v_ptr = v.data();
        for (int d = 0; d < dims; ++d) {
            const int groups = groups_per_dim[static_cast<std::size_t>(d)];
            double* b_seg = b.data() + offsets[static_cast<std::size_t>(d)];
#ifdef HDFE_USE_OPENMP
            if (threads > 1) {
#pragma omp parallel num_threads(threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& local =
                        tls_zt[static_cast<std::size_t>(d)][static_cast<std::size_t>(tid)];
                    local.setZero();
                    const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        if (unit_weights) {
                            local[g] += v_ptr[i];
                        } else {
                            local[g] += weight_ptr[i] * v_ptr[i];
                        }
                    }
                }
                Eigen::Map<Eigen::VectorXd> b_seg_vec(b_seg, groups);
                for (const auto& local : tls_zt[static_cast<std::size_t>(d)]) {
                    b_seg_vec.noalias() += local;
                }
            } else
#endif
            {
                const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
                if (unit_weights) {
                    for (int i = 0; i < n; ++i) {
                        b_seg[gid[i]] += v_ptr[i];
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        b_seg[gid[i]] += weight_ptr[i] * v_ptr[i];
                    }
                }
            }
        }
        return b;
    };

    auto apply_A = [&](const Eigen::VectorXd& x, Eigen::VectorXd& out) {
        for (int d = 0; d < dims; ++d) {
            x_ptrs[static_cast<std::size_t>(d)] =
                x.data() + offsets[static_cast<std::size_t>(d)];
        }

#ifdef HDFE_USE_OPENMP
        if (threads > 1) {
            for (int d = 0; d < dims; ++d) {
                for (int t = 0; t < threads; ++t) {
                    tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)].setZero();
                }
            }
#pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                auto& local_out = tls_out_ptrs[static_cast<std::size_t>(tid)];
                if (dims == 1) {
                    const int* gid0 = gid_ptrs[0];
                    double* out0 = local_out[0];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const double sum = x_ptrs[0][g0];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                    }
                } else if (dims == 2) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const double sum = x_ptrs[0][g0] + x_ptrs[1][g1];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                    }
                } else if (dims == 3) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    const int* gid2 = gid_ptrs[2];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
                    double* out2 = local_out[2];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const int g2 = gid2[i];
                        const double sum =
                            x_ptrs[0][g0] + x_ptrs[1][g1] + x_ptrs[2][g2];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                        out2[g2] += ws;
                    }
                } else if (dims == 4) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    const int* gid2 = gid_ptrs[2];
                    const int* gid3 = gid_ptrs[3];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
                    double* out2 = local_out[2];
                    double* out3 = local_out[3];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const int g2 = gid2[i];
                        const int g3 = gid3[i];
                        const double sum = x_ptrs[0][g0] + x_ptrs[1][g1] +
                                           x_ptrs[2][g2] + x_ptrs[3][g3];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                        out2[g2] += ws;
                        out3[g3] += ws;
                    }
                } else {
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        double sum = 0.0;
                        for (int d = 0; d < dims; ++d) {
                            const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                            sum += x_ptrs[static_cast<std::size_t>(d)][g];
                        }
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        for (int d = 0; d < dims; ++d) {
                            const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                            local_out[static_cast<std::size_t>(d)][g] += ws;
                        }
                    }
                }
            }

            for (int d = 0; d < dims; ++d) {
                Eigen::Map<Eigen::VectorXd> out_seg(
                    out.data() + offsets[static_cast<std::size_t>(d)],
                    groups_per_dim[static_cast<std::size_t>(d)]);
                out_seg = tls_out[static_cast<std::size_t>(d)][0];
                for (int t = 1; t < threads; ++t) {
                    out_seg.noalias() +=
                        tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)];
                }
            }
        } else
#endif
        {
            out.setZero(total_fe);
            for (int d = 0; d < dims; ++d) {
                out_ptrs[static_cast<std::size_t>(d)] =
                    out.data() + offsets[static_cast<std::size_t>(d)];
            }
            if (dims == 1) {
                const int* gid0 = gid_ptrs[0];
                double* out0 = out_ptrs[0];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const double sum = x_ptrs[0][g0];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    out0[g0] += w * sum;
                }
            } else if (dims == 2) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const double sum = x_ptrs[0][g0] + x_ptrs[1][g1];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                }
            } else if (dims == 3) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                const int* gid2 = gid_ptrs[2];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                double* out2 = out_ptrs[2];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const int g2 = gid2[i];
                    const double sum =
                        x_ptrs[0][g0] + x_ptrs[1][g1] + x_ptrs[2][g2];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                    out2[g2] += ws;
                }
            } else if (dims == 4) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                const int* gid2 = gid_ptrs[2];
                const int* gid3 = gid_ptrs[3];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                double* out2 = out_ptrs[2];
                double* out3 = out_ptrs[3];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const int g2 = gid2[i];
                    const int g3 = gid3[i];
                    const double sum = x_ptrs[0][g0] + x_ptrs[1][g1] +
                                       x_ptrs[2][g2] + x_ptrs[3][g3];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                    out2[g2] += ws;
                    out3[g3] += ws;
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    double sum = 0.0;
                    for (int d = 0; d < dims; ++d) {
                        const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                        sum += x_ptrs[static_cast<std::size_t>(d)][g];
                    }
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    for (int d = 0; d < dims; ++d) {
                        const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                        out_ptrs[static_cast<std::size_t>(d)][g] += ws;
                    }
                }
            }
        }

        if (lambda > 0.0) {
            out.noalias() += lambda * x;
        }
    };

    auto pcg_solve = [&](const Eigen::VectorXd& b, int& iters, bool& ok, int cap) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(total_fe);
        const double b_norm_sq = b.squaredNorm();
        if (b_norm_sq == 0.0) {
            iters = 0;
            ok = true;
            return x;
        }
        const double convergence_tol = effective_absorption_tolerance(options);
        const double tol_sq = convergence_tol * convergence_tol;

        Eigen::VectorXd r = b;
        Eigen::VectorXd z = r.cwiseQuotient(diagA);
        Eigen::VectorXd p = z;
        Eigen::VectorXd Ap(total_fe);
        double rz_old = r.dot(z);
        ok = false;
        int k = 0;
        const int _maxit = cap > 0 ? cap : options.max_iter;
        for (; k < _maxit; ++k) {
            apply_A(p, Ap);
            const double denom = p.dot(Ap);
            if (denom == 0.0) {
                break;
            }
            const double alpha = rz_old / denom;
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;

            const double r_norm_sq = r.squaredNorm();
            if (r_norm_sq <= tol_sq * b_norm_sq) {
                ok = true;
                ++k;
                break;
            }

            z = r.cwiseQuotient(diagA);
            const double rz_new = r.dot(z);
            const double beta = rz_new / rz_old;
            p = z + beta * p;
            rz_old = rz_new;
        }
        iters = k;
        return x;
    };

    auto subtract_projection = [&](Eigen::Ref<Eigen::VectorXd> v,
                                   const Eigen::VectorXd& alpha) {
        double* v_ptr = v.data();
        if (dims == 1) {
            const int* gid0 = gid_ptrs[0];
            const int offset0 = offsets[0];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                v_ptr[i] -= alpha[offset0 + g0];
            }
        } else if (dims == 2) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const double hat = alpha[offset0 + g0] + alpha[offset1 + g1];
                v_ptr[i] -= hat;
            }
        } else if (dims == 3) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int* gid2 = gid_ptrs[2];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
            const int offset2 = offsets[2];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const int g2 = gid2[i];
                const double hat =
                    alpha[offset0 + g0] + alpha[offset1 + g1] + alpha[offset2 + g2];
                v_ptr[i] -= hat;
            }
        } else if (dims == 4) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int* gid2 = gid_ptrs[2];
            const int* gid3 = gid_ptrs[3];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
            const int offset2 = offsets[2];
            const int offset3 = offsets[3];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const int g2 = gid2[i];
                const int g3 = gid3[i];
                const double hat = alpha[offset0 + g0] + alpha[offset1 + g1] +
                                   alpha[offset2 + g2] + alpha[offset3 + g3];
                v_ptr[i] -= hat;
            }
        } else {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                double hat = 0.0;
                for (int d = 0; d < dims; ++d) {
                    const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                    hat += alpha[offsets[static_cast<std::size_t>(d)] + g];
                }
                v_ptr[i] -= hat;
            }
        }
    };

    int max_iters_used = 0;
    bool all_converged = true;

    // Probe mode (krylov_probe_iters>0, used by the adaptive Schwarz gate): cap the leading
    // y-solve tightly and bail immediately if it does not converge, so a graph on which
    // matrix-free Jacobi-PCG stalls (genuine high-mobility AKM-type) falls back to the
    // approx-Cholesky Schwarz path after only ~probe wasted iterations rather than running
    // the full multi-column solve. Well-conditioned graphs converge in a few iterations, so
    // the cap never bites them; the X columns then get a generous (but bounded) budget.
    const int probe = options.krylov_probe_iters;
    const int y_cap = probe > 0 ? probe : options.max_iter;
    const int x_cap = probe > 0 ? std::max(probe * 8, 64) : options.max_iter;

    int iters_y = 0;
    bool ok_y = true;
    const Eigen::VectorXd b_y = compute_Zt(result.y_tilde);
    const Eigen::VectorXd alpha_y = pcg_solve(b_y, iters_y, ok_y, y_cap);
    subtract_projection(result.y_tilde, alpha_y);
    max_iters_used = std::max(max_iters_used, iters_y);
    all_converged = all_converged && ok_y;

    if (probe > 0 && !ok_y) {
        result.iterations = max_iters_used;
        result.converged = false;   // stalled -> let the caller fall back to Schwarz
        return result;
    }

    const int cols = static_cast<int>(result.X_tilde.cols());
    for (int j = 0; j < cols; ++j) {
        int iters_x = 0;
        bool ok_x = true;
        const Eigen::VectorXd b_x = compute_Zt(result.X_tilde.col(j));
        const Eigen::VectorXd alpha_x = pcg_solve(b_x, iters_x, ok_x, x_cap);
        subtract_projection(result.X_tilde.col(j), alpha_x);
        max_iters_used = std::max(max_iters_used, iters_x);
        all_converged = all_converged && ok_x;
    }

    result.iterations = max_iters_used;
    result.converged = all_converged;

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        result.fe_means.resize(indexers.size());
        result.fe_weight_sums.resize(indexers.size());
        for (int d = 0; d < dims; ++d) {
            result.fe_group_ids[static_cast<std::size_t>(d)] = indexers[static_cast<std::size_t>(d)].group_ids;
            result.fe_means[static_cast<std::size_t>(d)] =
                alpha_y.segment(offsets[static_cast<std::size_t>(d)],
                                groups_per_dim[static_cast<std::size_t>(d)]);
            result.fe_weight_sums[static_cast<std::size_t>(d)] =
                compute_weight_sums(indexers[static_cast<std::size_t>(d)],
                                    weight_ptr, unit_weights, n, threads);
        }
    }

    return result;
}

// ===================================================================
// Opt-in LSMR / MLSMR additive-Schwarz Krylov absorber (ported from
// Tiago's fix/current-xhdfe-issues branch, 24jun2026). Self-contained:
// reached ONLY via explicit absorptionmethod(lsmr|mlsmr) (or the
// auto-mlsmr selector). absorb_fixed_effects_mlsmr is the renamed branch
// krylov entry; the default sweep/Schwarz/Jacobi-PCG path is untouched.
// ===================================================================
int schwarz_max_local_dofs() {
    static const int cap = []() {
        const char* raw = std::getenv("XHDFE_SCHWARZ_MAX_LOCAL");
        if (!raw || *raw == '\0') {
            return 2048;
        }
        const int parsed = std::atoi(raw);
        return parsed > 0 ? parsed : 2048;
    }();
    return cap;
}

double schwarz_ridge_scale() {
    static const double scale = []() {
        const char* raw = std::getenv("XHDFE_SCHWARZ_RIDGE");
        if (!raw || *raw == '\0') {
            return 1.0e-10;
        }
        char* end = nullptr;
        const double parsed = std::strtod(raw, &end);
        return (end != raw && parsed >= 0.0 && std::isfinite(parsed)) ? parsed : 1.0e-10;
    }();
    return scale;
}

struct DenseSchwarzBlock {
    std::vector<int> global_indices;
    std::vector<double> partition_weights;
    Eigen::LDLT<Eigen::MatrixXd> factor;
};

struct DenseSchwarzPreconditioner {
    std::vector<DenseSchwarzBlock> blocks;
    Eigen::VectorXd diagonal_inv;
    std::vector<uint8_t> covered;

    bool enabled() const {
        return !blocks.empty();
    }

    void apply(const Eigen::VectorXd& r, Eigen::VectorXd& z) const {
        z.setZero(r.size());
        if (blocks.empty()) {
            z = diagonal_inv.array() * r.array();
            return;
        }

        for (const DenseSchwarzBlock& block : blocks) {
            const int local_n = static_cast<int>(block.global_indices.size());
            Eigen::VectorXd rhs(local_n);
            for (int i = 0; i < local_n; ++i) {
                rhs[i] = block.partition_weights[static_cast<std::size_t>(i)] *
                         r[block.global_indices[static_cast<std::size_t>(i)]];
            }

            const Eigen::VectorXd sol = block.factor.solve(rhs);
            if (!sol.allFinite()) {
                continue;
            }
            for (int i = 0; i < local_n; ++i) {
                const int g = block.global_indices[static_cast<std::size_t>(i)];
                z[g] += block.partition_weights[static_cast<std::size_t>(i)] * sol[i];
            }
        }

        for (Eigen::Index i = 0; i < r.size(); ++i) {
            if (covered.empty() || !covered[static_cast<std::size_t>(i)]) {
                z[i] = diagonal_inv[i] * r[i];
            }
        }
    }
};

int mlsmr_env_positive_int(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int mlsmr_env_nonnegative_int(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

double mlsmr_env_nonnegative_double(const char* name, double default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    char* end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || parsed < 0.0 || !std::isfinite(parsed)) {
        return default_value;
    }
    return parsed;
}

bool mlsmr_trace_enabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("XHDFE_MLSMR_TRACE");
        return raw && *raw && raw[0] != '0';
    }();
    return enabled;
}

double mlsmr_elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

bool mlsmr_env_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "0" || value == "off" || value == "false" || value == "no") {
        return false;
    }
    if (value == "1" || value == "on" || value == "true" || value == "yes") {
        return true;
    }
    return default_value;
}

struct MlsmrDsu {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit MlsmrDsu(int n = 0) : parent(static_cast<std::size_t>(n)),
                                   rank(static_cast<std::size_t>(n), 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        int root = x;
        while (parent[static_cast<std::size_t>(root)] != root) {
            root = parent[static_cast<std::size_t>(root)];
        }
        while (parent[static_cast<std::size_t>(x)] != x) {
            const int next = parent[static_cast<std::size_t>(x)];
            parent[static_cast<std::size_t>(x)] = root;
            x = next;
        }
        return root;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return;
        }
        if (rank[static_cast<std::size_t>(ra)] < rank[static_cast<std::size_t>(rb)]) {
            std::swap(ra, rb);
        }
        parent[static_cast<std::size_t>(rb)] = ra;
        if (rank[static_cast<std::size_t>(ra)] == rank[static_cast<std::size_t>(rb)]) {
            ++rank[static_cast<std::size_t>(ra)];
        }
    }
};

struct MlsmrPairEdge {
    int q = 0;
    int r = 0;
    double weight = 0.0;
};

struct MlsmrBuildComponent {
    std::vector<int> q_nodes;
    std::vector<int> r_nodes;
    std::vector<MlsmrPairEdge> edges;
};

struct MlsmrSchurEdge {
    int a = 0;
    int b = 0;
    double weight = 0.0;
};

constexpr double kMlsmrApproxCholNearZero = 1.0e-14;

double mlsmr_uniform01(std::mt19937_64& rng) {
    return std::generate_canonical<double, 53>(rng);
}

int mlsmr_sample_weighted_suffix(const std::vector<double>& cumsum,
                                 int start,
                                 int end,
                                 std::mt19937_64& rng) {
    if (start >= end || end > static_cast<int>(cumsum.size())) {
        return -1;
    }
    const double base = start > 0 ? cumsum[static_cast<std::size_t>(start - 1)] : 0.0;
    const double remaining = cumsum[static_cast<std::size_t>(end - 1)] - base;
    if (!(remaining > kMlsmrApproxCholNearZero) || !std::isfinite(remaining)) {
        return -1;
    }
    const double draw = base + mlsmr_uniform01(rng) * remaining;
    const auto begin = cumsum.begin() + start;
    const auto finish = cumsum.begin() + end;
    auto it = std::lower_bound(begin, finish, draw);
    if (it == finish) {
        return end - 1;
    }
    return static_cast<int>(std::distance(cumsum.begin(), it));
}

void mlsmr_clique_tree_sample_edges(std::vector<std::pair<int, double>>& entries,
                                    std::uint64_t seed,
                                    int split,
                                    std::vector<MlsmrSchurEdge>& out) {
    const int n = static_cast<int>(entries.size());
    if (n <= 1) {
        return;
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const auto& e) {
                                     return e.second <= kMlsmrApproxCholNearZero ||
                                            !std::isfinite(e.second);
                                 }),
                  entries.end());
    if (entries.size() <= 1) {
        return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (lhs.second != rhs.second) {
                      return lhs.second < rhs.second;
                  }
                  return lhs.first < rhs.first;
              });

    std::vector<double> cumsum;
    cumsum.reserve(entries.size());
    double total_weight = 0.0;
    for (const auto& entry : entries) {
        total_weight += entry.second;
        cumsum.push_back(total_weight);
    }
    if (!(total_weight > kMlsmrApproxCholNearZero) || !std::isfinite(total_weight)) {
        return;
    }

    auto add_sampled_edge = [&](int u, int v, double weight) {
        if (u == v || !(weight > kMlsmrApproxCholNearZero) ||
            !std::isfinite(weight)) {
            return;
        }
        if (u > v) {
            std::swap(u, v);
        }
        out.push_back({u, v, weight});
    };

    std::mt19937_64 rng(seed);
    double scale = 1.0;
    double capacity = total_weight;
    if (split <= 1) {
        for (int i = 0; i < static_cast<int>(entries.size()) - 1; ++i) {
            if (!(capacity > kMlsmrApproxCholNearZero)) {
                break;
            }
            const int j = entries[static_cast<std::size_t>(i)].first;
            const double w = entries[static_cast<std::size_t>(i)].second;
            const double f = w * scale / capacity;
            const double fill_wt = f * (1.0 - f) * capacity;
            const int koff = mlsmr_sample_weighted_suffix(
                cumsum, i + 1, static_cast<int>(entries.size()), rng);
            if (koff >= 0) {
                add_sampled_edge(j, entries[static_cast<std::size_t>(koff)].first,
                                 fill_wt);
            }
            const double retain = 1.0 - f;
            scale *= retain;
            capacity *= retain * retain;
        }
        return;
    }

    const double split_scale = static_cast<double>(split);
    double remaining = total_weight;
    for (int i = 0; i < static_cast<int>(entries.size()) - 1; ++i) {
        if (!(capacity > kMlsmrApproxCholNearZero)) {
            break;
        }
        const int j = entries[static_cast<std::size_t>(i)].first;
        const double w = entries[static_cast<std::size_t>(i)].second;
        remaining -= w;
        const double f = w * scale / capacity;
        const double fill_wt = w * remaining / (split_scale * total_weight);
        for (int s = 0; s < split; ++s) {
            const int koff = mlsmr_sample_weighted_suffix(
                cumsum, i + 1, static_cast<int>(entries.size()), rng);
            if (koff >= 0) {
                add_sampled_edge(j, entries[static_cast<std::size_t>(koff)].first,
                                 fill_wt);
            }
        }
        const double retain = 1.0 - f;
        scale *= retain;
        capacity *= retain * retain;
    }
}

struct MlsmrAcGraphEdge {
    int to = 0;
    int rev = 0;
    double weight = 0.0;
    int count = 1;
};

struct MlsmrAcNeighbor {
    int to = 0;
    double weight = 0.0;
    int count = 1;
};

struct MlsmrBucketOrdering {
    static constexpr int kSentinel = std::numeric_limits<int>::max();

    struct Elem {
        int prev = kSentinel;
        int next = kSentinel;
        int key = kSentinel;
    };

    std::vector<Elem> elems;
    std::vector<int> lists;
    int min_list = 0;
    int n_items = 0;
    int bucket_base = 1;
    int bucket_upper = 0;

    static int key_map(int degree, int base, int upper) {
        if (degree <= base) {
            return std::max(0, degree);
        }
        return std::min(upper, base + degree / std::max(1, base));
    }

    MlsmrBucketOrdering() = default;

    MlsmrBucketOrdering(int n, const std::vector<int>& degrees, int degree_scale) {
        reset(n, degrees, degree_scale);
    }

    void reset(int n, const std::vector<int>& degrees, int degree_scale) {
        elems.clear();
        lists.clear();
        min_list = 0;
        n_items = 0;
        bucket_base = std::max(1, degree_scale) * std::max(1, n);
        const long long raw_lists =
            2LL * static_cast<long long>(bucket_base) + 1LL;
        const int n_lists =
            raw_lists > static_cast<long long>(std::numeric_limits<int>::max())
                ? std::numeric_limits<int>::max()
                : static_cast<int>(raw_lists);
        bucket_upper = std::max(0, n_lists - 1);
        lists.assign(static_cast<std::size_t>(n_lists), kSentinel);
        elems.reserve(static_cast<std::size_t>(n));
        min_list = n_lists;
        for (int v = 0; v < n; ++v) {
            const int key = std::max(0, degrees[static_cast<std::size_t>(v)]);
            const int list = key_map(key, bucket_base, bucket_upper);
            const int old_head = lists[static_cast<std::size_t>(list)];
            elems.push_back({kSentinel, old_head, key});
            if (old_head != kSentinel) {
                elems[static_cast<std::size_t>(old_head)].prev = v;
            }
            lists[static_cast<std::size_t>(list)] = v;
            if (list < min_list) {
                min_list = list;
            }
            ++n_items;
        }
        if (min_list == n_lists) {
            min_list = 0;
        }
    }

    int next_vertex() {
        if (n_items <= 0) {
            return -1;
        }
        while (min_list < static_cast<int>(lists.size()) &&
               lists[static_cast<std::size_t>(min_list)] == kSentinel) {
            ++min_list;
        }
        if (min_list >= static_cast<int>(lists.size())) {
            return -1;
        }
        const int v = lists[static_cast<std::size_t>(min_list)];
        const int next = elems[static_cast<std::size_t>(v)].next;
        lists[static_cast<std::size_t>(min_list)] = next;
        if (next != kSentinel) {
            elems[static_cast<std::size_t>(next)].prev = kSentinel;
        }
        elems[static_cast<std::size_t>(v)].key = kSentinel;
        --n_items;
        return v;
    }

    void move_to_key(int v, int new_key) {
        if (v < 0 || v >= static_cast<int>(elems.size())) {
            return;
        }
        auto& elem = elems[static_cast<std::size_t>(v)];
        const int old_key = elem.key;
        if (old_key == kSentinel) {
            return;
        }
        new_key = std::max(0, new_key);
        const int old_list = key_map(old_key, bucket_base, bucket_upper);
        const int new_list = key_map(new_key, bucket_base, bucket_upper);
        elem.key = new_key;
        if (old_list == new_list) {
            return;
        }

        const int prev = elem.prev;
        const int next = elem.next;
        if (prev != kSentinel) {
            elems[static_cast<std::size_t>(prev)].next = next;
        } else {
            lists[static_cast<std::size_t>(old_list)] = next;
        }
        if (next != kSentinel) {
            elems[static_cast<std::size_t>(next)].prev = prev;
        }

        const int old_head = lists[static_cast<std::size_t>(new_list)];
        elem.prev = kSentinel;
        elem.next = old_head;
        if (old_head != kSentinel) {
            elems[static_cast<std::size_t>(old_head)].prev = v;
        }
        lists[static_cast<std::size_t>(new_list)] = v;
        if (new_list < min_list) {
            min_list = new_list;
        }
    }

    void notify_neighbor_removed_n(int v, int count) {
        if (count <= 0 || v < 0 || v >= static_cast<int>(elems.size())) {
            return;
        }
        const int key = elems[static_cast<std::size_t>(v)].key;
        if (key == kSentinel) {
            return;
        }
        move_to_key(v, std::max(0, key - count));
    }

    void notify_fill_edge(int u, int v, int count) {
        if (count <= 0) {
            return;
        }
        if (u >= 0 && u < static_cast<int>(elems.size())) {
            const int key = elems[static_cast<std::size_t>(u)].key;
            if (key != kSentinel && key <= std::numeric_limits<int>::max() - count) {
                move_to_key(u, key + count);
            }
        }
        if (v >= 0 && v < static_cast<int>(elems.size())) {
            const int key = elems[static_cast<std::size_t>(v)].key;
            if (key != kSentinel && key <= std::numeric_limits<int>::max() - count) {
                move_to_key(v, key + count);
            }
        }
    }
};

struct MlsmrApproxCholFactor {
    int n = 0;
    std::vector<int> vertices;
    std::vector<int> offsets;
    std::vector<int> neighbor_indices;
    std::vector<double> fractions;
    std::vector<double> inv_diagonal;

    bool valid() const {
        return n >= 0 && offsets.size() == vertices.size() + 1 &&
               inv_diagonal.size() == vertices.size() &&
               neighbor_indices.size() == fractions.size();
    }

    void record_column(int vertex,
                       const std::vector<int>& neighbors,
                       const std::vector<double>& fracs,
                       double diagonal) {
        vertices.push_back(vertex);
        neighbor_indices.insert(neighbor_indices.end(), neighbors.begin(), neighbors.end());
        fractions.insert(fractions.end(), fracs.begin(), fracs.end());
        inv_diagonal.push_back(
            std::abs(diagonal) > kMlsmrApproxCholNearZero ? 1.0 / diagonal : 0.0);
        offsets.push_back(static_cast<int>(neighbor_indices.size()));
    }

    bool solve_in_place(Eigen::Ref<Eigen::VectorXd> y) const {
        if (!valid() || y.size() != n) {
            return false;
        }

        for (int step_idx = 0; step_idx < static_cast<int>(vertices.size()); ++step_idx) {
            const int vertex = vertices[static_cast<std::size_t>(step_idx)];
            const int start = offsets[static_cast<std::size_t>(step_idx)];
            const int end = offsets[static_cast<std::size_t>(step_idx + 1)];
            if (vertex < 0 || vertex >= n || start < 0 || end < start ||
                end > static_cast<int>(neighbor_indices.size())) {
                return false;
            }
            const int deg = end - start;
            if (deg == 0) {
                if (inv_diagonal[static_cast<std::size_t>(step_idx)] != 0.0) {
                    y[vertex] *= inv_diagonal[static_cast<std::size_t>(step_idx)];
                }
                continue;
            }

            double yi = y[vertex];
            for (int pos = start; pos < end - 1; ++pos) {
                const int j = neighbor_indices[static_cast<std::size_t>(pos)];
                if (j < 0 || j >= n) {
                    return false;
                }
                const double f = fractions[static_cast<std::size_t>(pos)];
                y[j] += f * yi;
                yi *= (1.0 - f);
            }
            const int j_last = neighbor_indices[static_cast<std::size_t>(end - 1)];
            if (j_last < 0 || j_last >= n) {
                return false;
            }
            y[j_last] += yi;
            const double invd = inv_diagonal[static_cast<std::size_t>(step_idx)];
            y[vertex] = invd != 0.0 ? yi * invd : yi;
        }

        for (int step_idx = static_cast<int>(vertices.size()) - 1; step_idx >= 0; --step_idx) {
            const int vertex = vertices[static_cast<std::size_t>(step_idx)];
            const int start = offsets[static_cast<std::size_t>(step_idx)];
            const int end = offsets[static_cast<std::size_t>(step_idx + 1)];
            const int deg = end - start;
            if (deg == 0) {
                continue;
            }
            const int j_last = neighbor_indices[static_cast<std::size_t>(end - 1)];
            double yi = y[vertex] + y[j_last];
            for (int pos = end - 2; pos >= start; --pos) {
                const int j = neighbor_indices[static_cast<std::size_t>(pos)];
                const double f = fractions[static_cast<std::size_t>(pos)];
                yi = (1.0 - f) * yi + f * y[j];
            }
            y[vertex] = yi;
        }

        return y.allFinite();
    }

    static void add_edge_pair(std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
                              int u,
                              int v,
                              double weight,
                              int count,
                              std::vector<int>* degree_est,
                              std::priority_queue<std::pair<int, int>,
                                                  std::vector<std::pair<int, int>>,
                                                  std::greater<std::pair<int, int>>>* pq,
                              MlsmrBucketOrdering* bucket_ordering = nullptr) {
        if (u == v || u < 0 || v < 0 || u >= static_cast<int>(adj.size()) ||
            v >= static_cast<int>(adj.size()) ||
            !(weight > kMlsmrApproxCholNearZero) || !std::isfinite(weight) ||
            count <= 0) {
            return;
        }
        const int rev_u = static_cast<int>(adj[static_cast<std::size_t>(v)].size());
        const int rev_v = static_cast<int>(adj[static_cast<std::size_t>(u)].size());
        adj[static_cast<std::size_t>(u)].push_back({v, rev_u, weight, count});
        adj[static_cast<std::size_t>(v)].push_back({u, rev_v, weight, count});
        if (degree_est && pq) {
            (*degree_est)[static_cast<std::size_t>(u)] += count;
            (*degree_est)[static_cast<std::size_t>(v)] += count;
            pq->push({(*degree_est)[static_cast<std::size_t>(u)], u});
            pq->push({(*degree_est)[static_cast<std::size_t>(v)], v});
        }
        if (bucket_ordering) {
            bucket_ordering->notify_fill_edge(u, v, count);
        }
    }

    static void remove_edge_at(std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
                               int u,
                               int idx) {
        auto& edges = adj[static_cast<std::size_t>(u)];
        const int last = static_cast<int>(edges.size()) - 1;
        if (idx < 0 || idx > last) {
            return;
        }
        if (idx != last) {
            edges[static_cast<std::size_t>(idx)] = edges[static_cast<std::size_t>(last)];
            const auto moved = edges[static_cast<std::size_t>(idx)];
            if (moved.to >= 0 && moved.to < static_cast<int>(adj.size()) &&
                moved.rev >= 0 &&
                moved.rev < static_cast<int>(adj[static_cast<std::size_t>(moved.to)].size())) {
                adj[static_cast<std::size_t>(moved.to)]
                   [static_cast<std::size_t>(moved.rev)]
                       .rev = idx;
            }
        }
        edges.pop_back();
    }

    static std::vector<MlsmrAcNeighbor> live_neighbors(
        const std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
        const std::vector<uint8_t>& eliminated,
        int v,
        bool ac2,
        int merge_limit,
        MlsmrBucketOrdering* bucket_ordering = nullptr) {
        std::vector<MlsmrAcNeighbor> raw;
        raw.reserve(adj[static_cast<std::size_t>(v)].size());
        for (const auto& edge : adj[static_cast<std::size_t>(v)]) {
            if (edge.to < 0 || edge.to >= static_cast<int>(adj.size()) ||
                eliminated[static_cast<std::size_t>(edge.to)] ||
                !(edge.weight > kMlsmrApproxCholNearZero) || edge.count <= 0) {
                continue;
            }
            raw.push_back({edge.to, edge.weight * static_cast<double>(edge.count),
                           edge.count});
        }
        if (raw.empty()) {
            return raw;
        }
        std::sort(raw.begin(), raw.end(),
                  [](const MlsmrAcNeighbor& lhs, const MlsmrAcNeighbor& rhs) {
                      return lhs.to < rhs.to;
                  });
        std::vector<MlsmrAcNeighbor> out;
        out.reserve(raw.size());
        for (const auto& nbr : raw) {
            if (!out.empty() && out.back().to == nbr.to) {
                out.back().weight += nbr.weight;
                out.back().count = std::max(1, out.back().count + nbr.count);
            } else {
                out.push_back(nbr);
            }
        }
        for (auto& nbr : out) {
            const int old_count = std::max(1, nbr.count);
            int new_count = 1;
            if (ac2 && merge_limit > 0) {
                new_count = std::max(1, std::min(old_count, merge_limit));
            }
            if (bucket_ordering && old_count > new_count) {
                bucket_ordering->notify_neighbor_removed_n(nbr.to, old_count - new_count);
            }
            nbr.count = new_count;
        }
        std::sort(out.begin(), out.end(),
                  [ac2](const MlsmrAcNeighbor& lhs, const MlsmrAcNeighbor& rhs) {
                      const double lhs_key = ac2
                                                 ? lhs.weight /
                                                       static_cast<double>(std::max(1, lhs.count))
                                                 : lhs.weight;
                      const double rhs_key = ac2
                                                 ? rhs.weight /
                                                       static_cast<double>(std::max(1, rhs.count))
                                                 : rhs.weight;
                      if (lhs_key != rhs_key) {
                          return lhs_key < rhs_key;
                      }
                      return lhs.to < rhs.to;
                  });
        return out;
    }

    static int live_degree(const std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
                           const std::vector<uint8_t>& eliminated,
                           int v) {
        int degree = 0;
        for (const auto& edge : adj[static_cast<std::size_t>(v)]) {
            if (edge.to >= 0 && edge.to < static_cast<int>(adj.size()) &&
                !eliminated[static_cast<std::size_t>(edge.to)] &&
                edge.weight > kMlsmrApproxCholNearZero && edge.count > 0) {
                degree += edge.count;
            }
        }
        return degree;
    }

    static void eliminate_vertex(
        std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
        std::vector<uint8_t>& eliminated,
        std::vector<int>& degree_est,
        std::priority_queue<std::pair<int, int>,
                            std::vector<std::pair<int, int>>,
                            std::greater<std::pair<int, int>>>& pq,
        int v) {
        eliminated[static_cast<std::size_t>(v)] = 1;
        auto& edges = adj[static_cast<std::size_t>(v)];
        while (!edges.empty()) {
            const auto edge = edges.back();
            edges.pop_back();
            const int u = edge.to;
            if (u < 0 || u >= static_cast<int>(adj.size()) ||
                eliminated[static_cast<std::size_t>(u)]) {
                continue;
            }
            remove_edge_at(adj, u, edge.rev);
            degree_est[static_cast<std::size_t>(u)] =
                std::max(0, degree_est[static_cast<std::size_t>(u)] - edge.count);
            pq.push({degree_est[static_cast<std::size_t>(u)], u});
        }
        if (edges.capacity() > 64) {
            std::vector<MlsmrAcGraphEdge>().swap(edges);
        }
    }

    static void eliminate_vertex_bucket(
        std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
        std::vector<uint8_t>& eliminated,
        MlsmrBucketOrdering& ordering,
        int v) {
        eliminated[static_cast<std::size_t>(v)] = 1;
        auto& edges = adj[static_cast<std::size_t>(v)];
        while (!edges.empty()) {
            const auto edge = edges.back();
            edges.pop_back();
            const int u = edge.to;
            if (u < 0 || u >= static_cast<int>(adj.size()) ||
                eliminated[static_cast<std::size_t>(u)]) {
                continue;
            }
            remove_edge_at(adj, u, edge.rev);
            ordering.notify_neighbor_removed_n(u, edge.count);
        }
        if (edges.capacity() > 64) {
            std::vector<MlsmrAcGraphEdge>().swap(edges);
        }
    }

    static int next_vertex(
        const std::vector<std::vector<MlsmrAcGraphEdge>>& adj,
        const std::vector<uint8_t>& eliminated,
        std::vector<int>& degree_est,
        std::priority_queue<std::pair<int, int>,
                            std::vector<std::pair<int, int>>,
                            std::greater<std::pair<int, int>>>& pq) {
        while (!pq.empty()) {
            const auto [key, v] = pq.top();
            pq.pop();
            if (v < 0 || v >= static_cast<int>(adj.size()) ||
                eliminated[static_cast<std::size_t>(v)]) {
                continue;
            }
            if (key != degree_est[static_cast<std::size_t>(v)]) {
                continue;
            }
            const int actual = live_degree(adj, eliminated, v);
            if (actual != key) {
                degree_est[static_cast<std::size_t>(v)] = actual;
                pq.push({actual, v});
                continue;
            }
            return v;
        }
        return -1;
    }

    static std::shared_ptr<MlsmrApproxCholFactor> build(
        int n,
        const std::vector<MlsmrSchurEdge>& edges,
        std::uint64_t seed,
        int split_merge) {
        if (n <= 0) {
            return nullptr;
        }
        auto factor = std::make_shared<MlsmrApproxCholFactor>();
        factor->n = n;
        factor->offsets.reserve(static_cast<std::size_t>(n));
        factor->offsets.push_back(0);
        factor->vertices.reserve(static_cast<std::size_t>(std::max(0, n - 1)));
        factor->inv_diagonal.reserve(static_cast<std::size_t>(std::max(0, n - 1)));

        std::vector<std::vector<MlsmrAcGraphEdge>> adj(static_cast<std::size_t>(n));
        std::vector<double> diag(static_cast<std::size_t>(n), 0.0);
        for (const auto& edge : edges) {
            if (edge.a < 0 || edge.b < 0 || edge.a >= n || edge.b >= n ||
                edge.a == edge.b || !(edge.weight > kMlsmrApproxCholNearZero) ||
                !std::isfinite(edge.weight)) {
                continue;
            }
            const int count = std::max(1, split_merge);
            const double stored_weight =
                split_merge > 1 ? edge.weight / static_cast<double>(count) : edge.weight;
            add_edge_pair(adj, edge.a, edge.b, stored_weight, count, nullptr, nullptr);
            diag[static_cast<std::size_t>(edge.a)] += edge.weight;
            diag[static_cast<std::size_t>(edge.b)] += edge.weight;
        }

        std::vector<uint8_t> eliminated(static_cast<std::size_t>(n), 0);
        std::vector<int> degree_est(static_cast<std::size_t>(n), 0);
        std::priority_queue<std::pair<int, int>,
                            std::vector<std::pair<int, int>>,
                            std::greater<std::pair<int, int>>>
            pq;
        const bool bucket_ordering_enabled =
            mlsmr_env_enabled("XHDFE_MLSMR_AC_BUCKET_ORDERING", true);
        for (int v = 0; v < n; ++v) {
            degree_est[static_cast<std::size_t>(v)] = live_degree(adj, eliminated, v);
            if (!bucket_ordering_enabled) {
                pq.push({degree_est[static_cast<std::size_t>(v)], v});
            }
        }
        MlsmrBucketOrdering bucket_ordering;
        if (bucket_ordering_enabled) {
            bucket_ordering.reset(n, degree_est, std::max(1, split_merge));
        }

        std::mt19937_64 rng(seed);
        const bool ac2 = split_merge > 1;
        const int merge_limit = std::max(1, split_merge);
        const int target_steps = std::max(0, n - 1);
        int steps_done = 0;
        std::vector<double> cumsum;
        while (steps_done < target_steps) {
            const int v = bucket_ordering_enabled
                              ? bucket_ordering.next_vertex()
                              : next_vertex(adj, eliminated, degree_est, pq);
            if (v < 0) {
                break;
            }
            if (eliminated[static_cast<std::size_t>(v)]) {
                continue;
            }
            ++steps_done;

            const bool dedup_degree_updates =
                bucket_ordering_enabled &&
                mlsmr_env_enabled("XHDFE_MLSMR_AC_DEDUP_DEGREE_UPDATES", false);
            std::vector<MlsmrAcNeighbor> entries =
                live_neighbors(adj, eliminated, v, ac2, merge_limit,
                               dedup_degree_updates ? &bucket_ordering : nullptr);
            if (entries.empty()) {
                factor->record_column(v, {}, {}, diag[static_cast<std::size_t>(v)]);
                if (bucket_ordering_enabled) {
                    eliminate_vertex_bucket(adj, eliminated, bucket_ordering, v);
                } else {
                    eliminate_vertex(adj, eliminated, degree_est, pq, v);
                }
                continue;
            }

            std::vector<int> column_neighbors;
            std::vector<double> column_fractions;
            std::vector<MlsmrSchurEdge> fill_edges;
            column_neighbors.reserve(entries.size());
            column_fractions.reserve(entries.size());

            double column_diag = diag[static_cast<std::size_t>(v)];
            if (entries.size() == 1) {
                column_neighbors.push_back(entries.front().to);
                column_fractions.push_back(1.0);
            } else {
                cumsum.clear();
                cumsum.reserve(entries.size());
                double total_weight = 0.0;
                for (const auto& entry : entries) {
                    total_weight += entry.weight;
                    cumsum.push_back(total_weight);
                }
                if (!(total_weight > kMlsmrApproxCholNearZero) ||
                    !std::isfinite(total_weight)) {
                    factor->record_column(v, {}, {}, column_diag);
                    if (bucket_ordering_enabled) {
                        eliminate_vertex_bucket(adj, eliminated, bucket_ordering, v);
                    } else {
                        eliminate_vertex(adj, eliminated, degree_est, pq, v);
                    }
                    continue;
                }

                double scale = 1.0;
                double capacity = total_weight;
                if (!ac2) {
                    for (int i = 0; i < static_cast<int>(entries.size()) - 1; ++i) {
                        if (!(capacity > kMlsmrApproxCholNearZero)) {
                            break;
                        }
                        const auto& entry = entries[static_cast<std::size_t>(i)];
                        const double f = entry.weight * scale / capacity;
                        const double fill_wt = f * (1.0 - f) * capacity;
                        column_neighbors.push_back(entry.to);
                        column_fractions.push_back(f);
                        const int koff = mlsmr_sample_weighted_suffix(
                            cumsum, i + 1, static_cast<int>(entries.size()), rng);
                        if (koff >= 0 && fill_wt > kMlsmrApproxCholNearZero) {
                            int a = entry.to;
                            int b = entries[static_cast<std::size_t>(koff)].to;
                            if (a != b) {
                                if (a > b) {
                                    std::swap(a, b);
                                }
                                fill_edges.push_back({a, b, fill_wt});
                            }
                        }
                        const double retain = 1.0 - f;
                        scale *= retain;
                        capacity *= retain * retain;
                    }
                } else {
                    double remaining = total_weight;
                    for (int i = 0; i < static_cast<int>(entries.size()) - 1; ++i) {
                        if (!(capacity > kMlsmrApproxCholNearZero)) {
                            break;
                        }
                        const auto& entry = entries[static_cast<std::size_t>(i)];
                        remaining -= entry.weight;
                        const double f = entry.weight * scale / capacity;
                        const double fill_wt =
                            entry.weight * remaining /
                            (static_cast<double>(std::max(1, entry.count)) * total_weight);
                        column_neighbors.push_back(entry.to);
                        column_fractions.push_back(f);
                        for (int s = 0; s < std::max(1, entry.count); ++s) {
                            const int koff = mlsmr_sample_weighted_suffix(
                                cumsum, i + 1, static_cast<int>(entries.size()), rng);
                            if (koff >= 0 && fill_wt > kMlsmrApproxCholNearZero) {
                                int a = entry.to;
                                int b = entries[static_cast<std::size_t>(koff)].to;
                                if (a != b) {
                                    if (a > b) {
                                        std::swap(a, b);
                                    }
                                    fill_edges.push_back({a, b, fill_wt});
                                }
                            }
                        }
                        const double retain = 1.0 - f;
                        scale *= retain;
                        capacity *= retain * retain;
                    }
                }

                const auto& last = entries.back();
                column_neighbors.push_back(last.to);
                column_fractions.push_back(1.0);
                column_diag = last.weight * scale;
            }

            factor->record_column(v, column_neighbors, column_fractions, column_diag);
            if (bucket_ordering_enabled) {
                eliminate_vertex_bucket(adj, eliminated, bucket_ordering, v);
            } else {
                eliminate_vertex(adj, eliminated, degree_est, pq, v);
            }

            for (const auto& entry : entries) {
                if (entry.to >= 0 && entry.to < n) {
                    diag[static_cast<std::size_t>(entry.to)] -= entry.weight;
                }
            }
            for (const auto& edge : fill_edges) {
                if (edge.a < 0 || edge.b < 0 || edge.a >= n || edge.b >= n ||
                    eliminated[static_cast<std::size_t>(edge.a)] ||
                    eliminated[static_cast<std::size_t>(edge.b)] ||
                    !(edge.weight > kMlsmrApproxCholNearZero)) {
                    continue;
                }
                if (bucket_ordering_enabled) {
                    add_edge_pair(adj, edge.a, edge.b, edge.weight, 1,
                                  nullptr, nullptr, &bucket_ordering);
                } else {
                    add_edge_pair(adj, edge.a, edge.b, edge.weight, 1,
                                  &degree_est, &pq);
                }
                diag[static_cast<std::size_t>(edge.a)] += edge.weight;
                diag[static_cast<std::size_t>(edge.b)] += edge.weight;
            }
        }

        return factor->valid() ? factor : nullptr;
    }
};

struct MlsmrSchwarzDomain {
    std::vector<int> global_indices;
    std::vector<double> partition_weights;
    std::vector<double> sqrt_diag;
    std::shared_ptr<Eigen::LDLT<Eigen::MatrixXd>> dense_factor;
    std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> sparse_factor;
    std::shared_ptr<Eigen::LDLT<Eigen::MatrixXd>> reduced_dense_factor;
    std::shared_ptr<MlsmrApproxCholFactor> reduced_ac_factor;
    std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> reduced_sparse_factor;
    std::shared_ptr<Eigen::IncompleteCholesky<double>> reduced_ic_factor;
    Eigen::VectorXd reduced_diag_inv;
    std::vector<std::vector<std::pair<int, double>>> elim_to_keep;
    std::vector<int> elim_to_keep_indptr;
    std::vector<int> elim_to_keep_indices;
    std::vector<double> elim_to_keep_data;
    std::vector<double> inv_diag_elim;
    int n_local = 0;
    int nq = 0;
    int nr = 0;
    int n_keep = 0;
    int n_elim = 0;
    bool block_elim = false;
    bool eliminate_q = false;
    bool normalized_operator = true;

    bool valid() const {
        const bool full_valid = dense_factor || sparse_factor;
        const bool has_elim_vectors =
            elim_to_keep.size() == static_cast<std::size_t>(std::max(0, n_elim));
        const bool has_elim_csr =
            elim_to_keep_indptr.size() == static_cast<std::size_t>(std::max(0, n_elim) + 1) &&
            elim_to_keep_indices.size() == elim_to_keep_data.size();
        const bool block_valid =
            block_elim && n_keep > 0 &&
            (has_elim_vectors || has_elim_csr) &&
            (n_keep <= 1 || reduced_dense_factor || reduced_ac_factor ||
             reduced_sparse_factor ||
             reduced_ic_factor || reduced_diag_inv.size() == std::max(0, n_keep - 1));
        return n_local > 0 && (full_valid || block_valid);
    }

    static void subtract_mean(Eigen::Ref<Eigen::VectorXd> x, int n) {
        if (n <= 0) {
            return;
        }
        const double mean = x.head(n).mean();
        x.head(n).array() -= mean;
    }

    bool solve_block_elim(const Eigen::Ref<const Eigen::VectorXd>& rhs,
                          Eigen::Ref<Eigen::VectorXd> sol) const {
        if (!block_elim || rhs.size() != n_local ||
            sqrt_diag.size() != static_cast<std::size_t>(n_local) ||
            inv_diag_elim.size() != static_cast<std::size_t>(n_elim)) {
            return false;
        }
        const bool use_csr =
            elim_to_keep_indptr.size() == static_cast<std::size_t>(n_elim + 1) &&
            elim_to_keep_indices.size() == elim_to_keep_data.size();
        const bool use_vectors =
            elim_to_keep.size() == static_cast<std::size_t>(n_elim);
        if (!use_csr && !use_vectors) {
            return false;
        }

        Eigen::VectorXd signed_rhs(n_local);
        if (normalized_operator) {
            for (int i = 0; i < n_local; ++i) {
                signed_rhs[i] = sqrt_diag[static_cast<std::size_t>(i)] * rhs[i];
            }
        } else {
            signed_rhs = rhs;
        }
        for (int i = nq; i < n_local; ++i) {
            signed_rhs[i] = -signed_rhs[i];
        }
        subtract_mean(signed_rhs, n_local);

        const int keep_offset = eliminate_q ? nq : 0;
        const int elim_offset = eliminate_q ? 0 : nq;
        Eigen::VectorXd reduced_rhs =
            signed_rhs.segment(keep_offset, n_keep);
        for (int k = 0; k < n_elim; ++k) {
            const double scaled =
                inv_diag_elim[static_cast<std::size_t>(k)] *
                signed_rhs[elim_offset + k];
            if (use_csr) {
                const int start = elim_to_keep_indptr[static_cast<std::size_t>(k)];
                const int end = elim_to_keep_indptr[static_cast<std::size_t>(k + 1)];
                for (int idx = start; idx < end; ++idx) {
                    reduced_rhs[elim_to_keep_indices[static_cast<std::size_t>(idx)]] +=
                        elim_to_keep_data[static_cast<std::size_t>(idx)] * scaled;
                }
            } else {
                for (const auto& edge : elim_to_keep[static_cast<std::size_t>(k)]) {
                    reduced_rhs[edge.first] += edge.second * scaled;
                }
            }
        }
        subtract_mean(reduced_rhs, n_keep);

        Eigen::VectorXd keep_solution = Eigen::VectorXd::Zero(n_keep);
        const int factor_dim = std::max(0, n_keep - 1);
        if (reduced_ac_factor) {
            keep_solution = reduced_rhs;
            if (!reduced_ac_factor->solve_in_place(keep_solution) ||
                keep_solution.size() != n_keep || !keep_solution.allFinite()) {
                return false;
            }
        } else if (factor_dim > 0) {
            Eigen::VectorXd anchored_rhs = reduced_rhs.head(factor_dim);
            Eigen::VectorXd anchored_sol;
            if (reduced_dense_factor) {
                anchored_sol = reduced_dense_factor->solve(anchored_rhs);
            } else if (reduced_sparse_factor) {
                anchored_sol = reduced_sparse_factor->solve(anchored_rhs);
            } else if (reduced_ic_factor) {
                anchored_sol = reduced_ic_factor->solve(anchored_rhs);
            } else if (reduced_diag_inv.size() == factor_dim) {
                anchored_sol = reduced_diag_inv.array() * anchored_rhs.array();
            } else {
                return false;
            }
            if (anchored_sol.size() != factor_dim || !anchored_sol.allFinite()) {
                return false;
            }
            keep_solution.head(factor_dim) = anchored_sol;
        }

        sol.setZero();
        sol.segment(keep_offset, n_keep) = keep_solution;
        for (int k = 0; k < n_elim; ++k) {
            double value = signed_rhs[elim_offset + k];
            if (use_csr) {
                const int start = elim_to_keep_indptr[static_cast<std::size_t>(k)];
                const int end = elim_to_keep_indptr[static_cast<std::size_t>(k + 1)];
                for (int idx = start; idx < end; ++idx) {
                    value += elim_to_keep_data[static_cast<std::size_t>(idx)] *
                             keep_solution[elim_to_keep_indices[static_cast<std::size_t>(idx)]];
                }
            } else {
                for (const auto& edge : elim_to_keep[static_cast<std::size_t>(k)]) {
                    value += edge.second * keep_solution[edge.first];
                }
            }
            sol[elim_offset + k] =
                inv_diag_elim[static_cast<std::size_t>(k)] * value;
        }

        subtract_mean(sol, n_local);
        for (int i = nq; i < n_local; ++i) {
            sol[i] = -sol[i];
        }
        if (normalized_operator) {
            for (int i = 0; i < n_local; ++i) {
                sol[i] *= sqrt_diag[static_cast<std::size_t>(i)];
            }
        }
        return sol.allFinite();
    }

    bool solve(const Eigen::Ref<const Eigen::VectorXd>& rhs,
               Eigen::Ref<Eigen::VectorXd> sol) const {
        if (!valid() || rhs.size() != n_local) {
            return false;
        }
        if (block_elim && solve_block_elim(rhs, sol)) {
            return true;
        }
        if (dense_factor) {
            sol = dense_factor->solve(rhs);
        } else if (sparse_factor) {
            sol = sparse_factor->solve(rhs);
        } else {
            return false;
        }
        return sol.size() == n_local && sol.allFinite();
    }
};

struct MlsmrAdditiveSchwarzPreconditioner {
    std::vector<MlsmrSchwarzDomain> domains;
    Eigen::VectorXd diagonal_inv;
    std::vector<uint8_t> covered;
    bool domains_disjoint = false;

    bool enabled() const {
        return !domains.empty();
    }

    void apply(const Eigen::VectorXd& r, Eigen::VectorXd& z) const {
        z.setZero(r.size());
        if (domains.empty()) {
            if (diagonal_inv.size() == r.size()) {
                z = diagonal_inv.array() * r.array();
            } else {
                z = r;
            }
            return;
        }

#ifdef HDFE_USE_OPENMP
        const bool parallel_apply =
            mlsmr_env_enabled("XHDFE_MLSMR_PARALLEL_APPLY", true);
        const bool disjoint_direct_apply =
            domains_disjoint &&
            mlsmr_env_enabled("XHDFE_MLSMR_DISJOINT_DIRECT_APPLY", false);
        const int apply_threads = std::max(1, omp_get_max_threads());
        if (parallel_apply && apply_threads > 1 && domains.size() > 1) {
#pragma omp parallel num_threads(apply_threads)
            {
                Eigen::VectorXd rhs_scratch;
                Eigen::VectorXd sol_scratch;
#pragma omp for schedule(dynamic)
                for (int d = 0; d < static_cast<int>(domains.size()); ++d) {
                    const auto& domain = domains[static_cast<std::size_t>(d)];
                    const int n = domain.n_local;
                    if (rhs_scratch.size() < n) {
                        rhs_scratch.resize(n);
                        sol_scratch.resize(n);
                    }
                    Eigen::Ref<Eigen::VectorXd> rhs = rhs_scratch.head(n);
                    for (int i = 0; i < n; ++i) {
                        const int g = domain.global_indices[static_cast<std::size_t>(i)];
                        rhs[i] = domain.partition_weights[static_cast<std::size_t>(i)] * r[g];
                    }
                    Eigen::Ref<Eigen::VectorXd> sol = sol_scratch.head(n);
                    if (!domain.solve(rhs, sol)) {
                        continue;
                    }
                    for (int i = 0; i < n; ++i) {
                        const int g = domain.global_indices[static_cast<std::size_t>(i)];
                        const double contribution =
                            domain.partition_weights[static_cast<std::size_t>(i)] * sol[i];
                        if (disjoint_direct_apply) {
                            z[g] = contribution;
                        } else {
#pragma omp atomic update
                            z[g] += contribution;
                        }
                    }
                }
            }

            for (Eigen::Index i = 0; i < r.size(); ++i) {
                if (covered.empty() || !covered[static_cast<std::size_t>(i)]) {
                    z[i] = diagonal_inv.size() == r.size() ? diagonal_inv[i] * r[i] : r[i];
                }
            }
            return;
        }
#endif

        int max_local = 0;
        for (const auto& domain : domains) {
            max_local = std::max(max_local, domain.n_local);
        }
        Eigen::VectorXd rhs_scratch(max_local);
        Eigen::VectorXd sol_scratch(max_local);

        for (const auto& domain : domains) {
            const int n = domain.n_local;
            Eigen::Ref<Eigen::VectorXd> rhs = rhs_scratch.head(n);
            for (int i = 0; i < n; ++i) {
                const int g = domain.global_indices[static_cast<std::size_t>(i)];
                rhs[i] = domain.partition_weights[static_cast<std::size_t>(i)] * r[g];
            }
            Eigen::Ref<Eigen::VectorXd> sol = sol_scratch.head(n);
            if (!domain.solve(rhs, sol)) {
                continue;
            }
            for (int i = 0; i < n; ++i) {
                const int g = domain.global_indices[static_cast<std::size_t>(i)];
                z[g] += domain.partition_weights[static_cast<std::size_t>(i)] * sol[i];
            }
        }

        for (Eigen::Index i = 0; i < r.size(); ++i) {
            if (covered.empty() || !covered[static_cast<std::size_t>(i)]) {
                z[i] = diagonal_inv.size() == r.size() ? diagonal_inv[i] * r[i] : r[i];
            }
        }
    }
};

bool mlsmr_pair_token_matches(const std::string& token, int q, int r) {
    if (token.empty()) {
        return false;
    }
    std::string clean = token;
    for (char& c : clean) {
        if (c == ':' || c == '_') {
            c = '-';
        }
    }
    const std::size_t dash = clean.find('-');
    if (dash == std::string::npos) {
        return false;
    }
    char* end_a = nullptr;
    char* end_b = nullptr;
    const long a = std::strtol(clean.c_str(), &end_a, 10);
    const long b = std::strtol(clean.c_str() + dash + 1, &end_b, 10);
    if (end_a != clean.c_str() + dash || end_b == clean.c_str() + dash + 1 ||
        *end_b != '\0' || a < 0 || b < 0 ||
        a > std::numeric_limits<int>::max() ||
        b > std::numeric_limits<int>::max()) {
        return false;
    }
    const int ia = static_cast<int>(a);
    const int ib = static_cast<int>(b);
    return (ia == q && ib == r) || (ia == r && ib == q);
}

bool mlsmr_pair_explicitly_skipped(int q, int r) {
    const char* raw = std::getenv("XHDFE_MLSMR_SKIP_PAIR_DIMS");
    if (!raw || !*raw) {
        return false;
    }
    std::string spec(raw);
    for (char& c : spec) {
        if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
            c = ' ';
        }
    }
    std::istringstream stream(spec);
    std::string token;
    while (stream >> token) {
        if (mlsmr_pair_token_matches(token, q, r)) {
            return true;
        }
    }
    return false;
}

std::uint64_t mlsmr_edge_key(int q, int r) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(q)) << 32) |
           static_cast<std::uint32_t>(r);
}

std::uint64_t mlsmr_local_edge_key(int lo, int hi) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32) |
           static_cast<std::uint32_t>(hi);
}

std::string mlsmr_env_string(const char* name, const char* default_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return std::string(default_value);
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool build_mlsmr_domain_from_component(const MlsmrBuildComponent& comp,
                                       int q_dim,
                                       int r_dim,
                                       const std::vector<int>& offsets,
                                       const Eigen::VectorXd& diagA,
                                       int full_dense_threshold,
                                       int dense_schur_threshold,
                                       int exact_star_threshold,
                                       int full_fallback_max_local,
                                       double ridge_scale,
                                       bool normalized_operator,
                                       MlsmrSchwarzDomain& out) {
    const int nq = static_cast<int>(comp.q_nodes.size());
    const int nr = static_cast<int>(comp.r_nodes.size());
    const int n_local = nq + nr;
    if (n_local <= 0) {
        return false;
    }
    const bool component_trace =
        mlsmr_env_enabled("XHDFE_MLSMR_COMPONENT_TRACE", false);
    const int component_trace_threshold =
        mlsmr_env_nonnegative_int("XHDFE_MLSMR_COMPONENT_TRACE_THRESHOLD", 100000);
    const bool trace_component =
        component_trace && n_local >= component_trace_threshold;
    const auto component_trace_start = std::chrono::steady_clock::now();

    out.global_indices.clear();
    out.partition_weights.clear();
    out.sqrt_diag.clear();
    out.dense_factor.reset();
    out.sparse_factor.reset();
    out.reduced_dense_factor.reset();
    out.reduced_ac_factor.reset();
    out.reduced_sparse_factor.reset();
    out.reduced_ic_factor.reset();
    out.reduced_diag_inv.resize(0);
    out.elim_to_keep.clear();
    out.elim_to_keep_indptr.clear();
    out.elim_to_keep_indices.clear();
    out.elim_to_keep_data.clear();
    out.inv_diag_elim.clear();
    out.n_local = n_local;
    out.nq = nq;
    out.nr = nr;
    out.n_keep = 0;
    out.n_elim = 0;
    out.block_elim = false;
    out.eliminate_q = false;
    out.normalized_operator = normalized_operator;
    out.global_indices.reserve(static_cast<std::size_t>(n_local));
    out.sqrt_diag.reserve(static_cast<std::size_t>(n_local));

    const auto map_start = std::chrono::steady_clock::now();
    std::vector<double> q_diag(static_cast<std::size_t>(nq), 0.0);
    std::vector<double> r_diag(static_cast<std::size_t>(nr), 0.0);
    std::unordered_map<int, int> q_pos_map;
    std::unordered_map<int, int> r_pos_map;
    q_pos_map.reserve(static_cast<std::size_t>(std::max(1, nq)));
    r_pos_map.reserve(static_cast<std::size_t>(std::max(1, nr)));

    auto q_pos = [&](int g) -> int {
        auto it = q_pos_map.find(g);
        return it == q_pos_map.end() ? -1 : it->second;
    };
    auto r_pos = [&](int g) -> int {
        auto it = r_pos_map.find(g);
        return it == r_pos_map.end() ? -1 : it->second;
    };

    for (int i = 0; i < nq; ++i) {
        const int g = comp.q_nodes[static_cast<std::size_t>(i)];
        q_pos_map.emplace(g, i);
        const int global = offsets[static_cast<std::size_t>(q_dim)] + g;
        const double diag = std::max(0.0, diagA[global]);
        q_diag[static_cast<std::size_t>(i)] = diag;
        out.global_indices.push_back(global);
        out.sqrt_diag.push_back(std::sqrt(diag));
    }
    for (int i = 0; i < nr; ++i) {
        const int g = comp.r_nodes[static_cast<std::size_t>(i)];
        r_pos_map.emplace(g, i);
        const int global = offsets[static_cast<std::size_t>(r_dim)] + g;
        const double diag = std::max(0.0, diagA[global]);
        r_diag[static_cast<std::size_t>(i)] = diag;
        out.global_indices.push_back(global);
        out.sqrt_diag.push_back(std::sqrt(diag));
    }
    const double map_seconds =
        trace_component ? mlsmr_elapsed_seconds(map_start) : 0.0;

    const bool block_eliminate_q = nq >= nr;
    const int block_n_elim = block_eliminate_q ? nq : nr;
    const bool csr_domain = mlsmr_env_enabled("XHDFE_MLSMR_CSR_DOMAIN", false);
    std::vector<std::vector<std::pair<int, double>>> block_elim_to_keep;
    std::vector<int> block_elim_indptr;
    std::vector<int> block_elim_indices;
    std::vector<double> block_elim_data;

    auto edge_to_block_positions = [&](const MlsmrPairEdge& edge,
                                       int& elim_local,
                                       int& keep_local) -> bool {
        const int q_local = q_pos(edge.q);
        const int r_local = r_pos(edge.r);
        if (q_local < 0 || r_local < 0 || edge.weight <= 0.0) {
            return false;
        }
        if (block_eliminate_q) {
            elim_local = q_local;
            keep_local = r_local;
        } else {
            elim_local = r_local;
            keep_local = q_local;
        }
        return true;
    };

    const auto block_star_start = std::chrono::steady_clock::now();
    if (csr_domain) {
        block_elim_indptr.assign(static_cast<std::size_t>(block_n_elim + 1), 0);
        int elim_local = -1;
        int keep_local = -1;
        for (const auto& edge : comp.edges) {
            if (edge_to_block_positions(edge, elim_local, keep_local)) {
                ++block_elim_indptr[static_cast<std::size_t>(elim_local + 1)];
            }
        }
        for (int k = 0; k < block_n_elim; ++k) {
            block_elim_indptr[static_cast<std::size_t>(k + 1)] +=
                block_elim_indptr[static_cast<std::size_t>(k)];
        }
        const int nnz = block_elim_indptr.back();
        block_elim_indices.assign(static_cast<std::size_t>(nnz), 0);
        block_elim_data.assign(static_cast<std::size_t>(nnz), 0.0);
        std::vector<int> write_pos = block_elim_indptr;
        for (const auto& edge : comp.edges) {
            if (!edge_to_block_positions(edge, elim_local, keep_local)) {
                continue;
            }
            const int pos = write_pos[static_cast<std::size_t>(elim_local)]++;
            block_elim_indices[static_cast<std::size_t>(pos)] = keep_local;
            block_elim_data[static_cast<std::size_t>(pos)] = edge.weight;
        }
    } else {
        block_elim_to_keep.assign(
            static_cast<std::size_t>(std::max(0, block_n_elim)), {});
        int elim_local = -1;
        int keep_local = -1;
        for (const auto& edge : comp.edges) {
            if (!edge_to_block_positions(edge, elim_local, keep_local)) {
                continue;
            }
            block_elim_to_keep[static_cast<std::size_t>(elim_local)].push_back(
                {keep_local, edge.weight});
        }
    }
    const double block_star_seconds =
        trace_component ? mlsmr_elapsed_seconds(block_star_start) : 0.0;

    auto try_build_block_elim = [&]() -> bool {
        const auto block_start = std::chrono::steady_clock::now();
        const bool eliminate_q = block_eliminate_q;
        const int n_keep = eliminate_q ? nr : nq;
        const int n_elim = eliminate_q ? nq : nr;
        if (n_keep <= 0 || n_elim <= 0) {
            return false;
        }
        const auto& diag_keep = eliminate_q ? r_diag : q_diag;
        const auto& diag_elim = eliminate_q ? q_diag : r_diag;

        const auto inv_start = std::chrono::steady_clock::now();
        std::vector<double> inv_diag_elim(static_cast<std::size_t>(n_elim), 0.0);
        for (int k = 0; k < n_elim; ++k) {
            const double d = diag_elim[static_cast<std::size_t>(k)];
            if (d <= 0.0 || !std::isfinite(d)) {
                return false;
            }
            inv_diag_elim[static_cast<std::size_t>(k)] = 1.0 / d;
        }
        const double inv_seconds =
            trace_component ? mlsmr_elapsed_seconds(inv_start) : 0.0;

        Eigen::VectorXd schur_diag(n_keep);
        for (int i = 0; i < n_keep; ++i) {
            schur_diag[i] = diag_keep[static_cast<std::size_t>(i)];
        }

        const int factor_dim = std::max(0, n_keep - 1);
        const double mean_diag =
            n_keep > 0 ? schur_diag.sum() / static_cast<double>(n_keep) : 1.0;
        const double ridge = ridge_scale * std::max(1.0, mean_diag);

        if (factor_dim > 0 && n_keep <= dense_schur_threshold) {
            Eigen::MatrixXd minor = Eigen::MatrixXd::Zero(factor_dim, factor_dim);
            for (int i = 0; i < factor_dim; ++i) {
                minor(i, i) = schur_diag[i] + ridge;
            }
            for (int k = 0; k < n_elim; ++k) {
                const double inv_d = inv_diag_elim[static_cast<std::size_t>(k)];
                if (csr_domain) {
                    const int start = block_elim_indptr[static_cast<std::size_t>(k)];
                    const int end = block_elim_indptr[static_cast<std::size_t>(k + 1)];
                    for (int ia = start; ia < end; ++ia) {
                        const int a = block_elim_indices[static_cast<std::size_t>(ia)];
                        if (a >= factor_dim) {
                            continue;
                        }
                        const double aw = block_elim_data[static_cast<std::size_t>(ia)];
                        for (int ib = start; ib < end; ++ib) {
                            const int b = block_elim_indices[static_cast<std::size_t>(ib)];
                            if (b >= factor_dim) {
                                continue;
                            }
                            minor(a, b) -=
                                aw * inv_d * block_elim_data[static_cast<std::size_t>(ib)];
                        }
                    }
                } else {
                    const auto& star = block_elim_to_keep[static_cast<std::size_t>(k)];
                    for (const auto& a : star) {
                        if (a.first >= factor_dim) {
                            continue;
                        }
                        for (const auto& b : star) {
                            if (b.first >= factor_dim) {
                                continue;
                            }
                            minor(a.first, b.first) -= a.second * inv_d * b.second;
                        }
                    }
                }
            }
            auto factor = std::make_shared<Eigen::LDLT<Eigen::MatrixXd>>();
            factor->compute(minor);
            if (factor->info() != Eigen::Success) {
                return false;
            }
            out.reduced_dense_factor = factor;
        } else if (factor_dim > 0) {
            const auto emit_start = std::chrono::steady_clock::now();
            std::vector<MlsmrSchurEdge> offdiag_edges;
            offdiag_edges.reserve(comp.edges.size());
            auto add_edge_to = [](std::vector<MlsmrSchurEdge>& target,
                                  int a,
                                  int b,
                                  double weight) {
                if (a == b || weight <= 0.0 || !std::isfinite(weight)) {
                    return;
                }
                if (a > b) {
                    std::swap(a, b);
                }
                target.push_back({a, b, weight});
            };
            const int schur_split = mlsmr_env_positive_int(
                "XHDFE_MLSMR_APPROX_SCHUR_SPLIT", 1);
            const std::uint64_t schur_seed =
                static_cast<std::uint64_t>(mlsmr_env_positive_int(
                    "XHDFE_MLSMR_APPROX_SCHUR_SEED", 0));

            auto emit_star_edges = [&](int k, std::vector<MlsmrSchurEdge>& target) {
                const double inv_d = inv_diag_elim[static_cast<std::size_t>(k)];
                const int csr_start =
                    csr_domain ? block_elim_indptr[static_cast<std::size_t>(k)] : 0;
                const int csr_end =
                    csr_domain ? block_elim_indptr[static_cast<std::size_t>(k + 1)] : 0;
                const int star_size = csr_domain
                                          ? (csr_end - csr_start)
                                          : static_cast<int>(
                                                block_elim_to_keep[static_cast<std::size_t>(k)]
                                                    .size());
                if (star_size <= 0) {
                    return;
                }

                if (star_size <= exact_star_threshold) {
                    if (csr_domain) {
                        for (int ia = csr_start; ia < csr_end; ++ia) {
                            for (int ib = ia + 1; ib < csr_end; ++ib) {
                                add_edge_to(
                                    target,
                                    block_elim_indices[static_cast<std::size_t>(ia)],
                                    block_elim_indices[static_cast<std::size_t>(ib)],
                                    block_elim_data[static_cast<std::size_t>(ia)] *
                                        inv_d *
                                        block_elim_data[static_cast<std::size_t>(ib)]);
                            }
                        }
                    } else {
                        const auto& star = block_elim_to_keep[static_cast<std::size_t>(k)];
                        for (std::size_t a = 0; a < star.size(); ++a) {
                            for (std::size_t b = a + 1; b < star.size(); ++b) {
                                add_edge_to(target, star[a].first, star[b].first,
                                            star[a].second * inv_d * star[b].second);
                            }
                        }
                    }
                    return;
                }

                (void)inv_d;
                std::vector<std::pair<int, double>> sample_entries;
                sample_entries.reserve(static_cast<std::size_t>(star_size));
                if (csr_domain) {
                    for (int idx = csr_start; idx < csr_end; ++idx) {
                        sample_entries.push_back(
                            {block_elim_indices[static_cast<std::size_t>(idx)],
                             block_elim_data[static_cast<std::size_t>(idx)]});
                    }
                } else {
                    const auto& star = block_elim_to_keep[static_cast<std::size_t>(k)];
                    for (const auto& item : star) {
                        sample_entries.push_back({item.first, item.second});
                    }
                }
                mlsmr_clique_tree_sample_edges(
                    sample_entries,
                    schur_seed + static_cast<std::uint64_t>(k),
                    std::max(1, schur_split),
                    target);
            };

#ifdef HDFE_USE_OPENMP
            const bool parallel_schur =
                mlsmr_env_enabled("XHDFE_MLSMR_PARALLEL_SCHUR", false);
            const int parallel_schur_threshold =
                mlsmr_env_positive_int("XHDFE_MLSMR_PARALLEL_SCHUR_THRESHOLD", 50000);
            const int schur_threads = std::max(1, omp_get_max_threads());
            if (parallel_schur && schur_threads > 1 &&
                n_elim >= parallel_schur_threshold) {
                std::vector<std::vector<MlsmrSchurEdge>> local_edges(
                    static_cast<std::size_t>(schur_threads));
#pragma omp parallel num_threads(schur_threads)
                {
                    const int tid = omp_get_thread_num();
                    auto& target = local_edges[static_cast<std::size_t>(tid)];
#pragma omp for schedule(dynamic)
                    for (int k = 0; k < n_elim; ++k) {
                        emit_star_edges(k, target);
                    }
                }
                std::size_t total_edges = 0;
                for (const auto& local : local_edges) {
                    total_edges += local.size();
                }
                offdiag_edges.reserve(total_edges);
                for (auto& local : local_edges) {
                    std::move(local.begin(), local.end(),
                              std::back_inserter(offdiag_edges));
                }
            } else
#endif
            {
                for (int k = 0; k < n_elim; ++k) {
                    emit_star_edges(k, offdiag_edges);
                }
            }
            const double emit_seconds =
                trace_component ? mlsmr_elapsed_seconds(emit_start) : 0.0;

            const auto sort_start = std::chrono::steady_clock::now();
            std::sort(offdiag_edges.begin(), offdiag_edges.end(),
                      [](const MlsmrSchurEdge& lhs, const MlsmrSchurEdge& rhs) {
                          if (lhs.a != rhs.a) {
                              return lhs.a < rhs.a;
                          }
                          if (lhs.b != rhs.b) {
                              return lhs.b < rhs.b;
                          }
                          return lhs.weight < rhs.weight;
                      });
            std::size_t write = 0;
            for (std::size_t read = 0; read < offdiag_edges.size(); ++read) {
                if (write > 0 && offdiag_edges[write - 1].a == offdiag_edges[read].a &&
                    offdiag_edges[write - 1].b == offdiag_edges[read].b) {
                    offdiag_edges[write - 1].weight += offdiag_edges[read].weight;
                } else {
                    offdiag_edges[write++] = offdiag_edges[read];
                }
            }
            offdiag_edges.resize(write);
            const double sort_seconds =
                trace_component ? mlsmr_elapsed_seconds(sort_start) : 0.0;

            const std::string factor_kind =
                mlsmr_env_string("XHDFE_MLSMR_REDUCED_FACTOR", "auto");
            bool factor_built = false;
            Eigen::VectorXd lap_diag;
            Eigen::SparseMatrix<double> schur;
            bool lap_diag_ready = false;
            bool schur_ready = false;
            double lap_schur_seconds = 0.0;
            auto ensure_lap_schur = [&](bool need_schur) {
                if (lap_diag_ready && (!need_schur || schur_ready)) {
                    return;
                }
                const auto lap_start = std::chrono::steady_clock::now();
                lap_diag = Eigen::VectorXd::Zero(factor_dim);
                std::vector<Eigen::Triplet<double>> trips;
                if (need_schur) {
                    trips.reserve(static_cast<std::size_t>(factor_dim) +
                                  2 * offdiag_edges.size());
                }
                for (const auto& edge : offdiag_edges) {
                    const int a = edge.a;
                    const int b = edge.b;
                    const double w = edge.weight;
                    if (a < factor_dim) {
                        lap_diag[a] += w;
                    }
                    if (b < factor_dim) {
                        lap_diag[b] += w;
                    }
                    if (need_schur && a < factor_dim && b < factor_dim && w > 0.0) {
                        trips.emplace_back(a, b, -w);
                        trips.emplace_back(b, a, -w);
                    }
                }
                if (need_schur) {
                    for (int i = 0; i < factor_dim; ++i) {
                        const double d = std::max(0.0, lap_diag[i]) + ridge;
                        trips.emplace_back(i, i, d);
                    }
                    schur.resize(factor_dim, factor_dim);
                    schur.setFromTriplets(trips.begin(), trips.end());
                    schur.makeCompressed();
                    schur_ready = true;
                }
                lap_diag_ready = true;
                lap_schur_seconds +=
                    trace_component ? mlsmr_elapsed_seconds(lap_start) : 0.0;
            };

            const auto factor_start = std::chrono::steady_clock::now();
            if (factor_kind == "auto" || factor_kind == "ac" ||
                factor_kind == "approx" || factor_kind == "approx-chol" ||
                factor_kind == "approxchol") {
                const std::uint64_t ac_seed =
                    static_cast<std::uint64_t>(mlsmr_env_positive_int(
                        "XHDFE_MLSMR_AC_SEED", 0));
                const int ac_split_merge = mlsmr_env_positive_int(
                    "XHDFE_MLSMR_AC_SPLIT_MERGE", 2);
                auto factor = MlsmrApproxCholFactor::build(
                    n_keep, offdiag_edges, ac_seed, std::max(1, ac_split_merge));
                if (factor && factor->valid()) {
                    out.reduced_ac_factor = factor;
                    factor_built = true;
                }
            }
            if (!factor_built && factor_kind != "auto" && factor_kind != "diag" &&
                factor_kind != "diagonal") {
                if (factor_kind == "ic" || factor_kind == "ichol" ||
                    factor_kind == "eigen-ic" || factor_kind == "eigen-ichol") {
                    ensure_lap_schur(true);
                    auto factor = std::make_shared<Eigen::IncompleteCholesky<double>>();
                    factor->compute(schur);
                    if (factor->info() == Eigen::Success) {
                        out.reduced_ic_factor = factor;
                        factor_built = true;
                    }
                }
                if (!factor_built &&
                    (factor_kind == "ldlt" || factor_kind == "simplicial" ||
                     factor_kind == "sparse" || factor_kind == "eigen-ldlt")) {
                    ensure_lap_schur(true);
                    auto factor =
                        std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
                    factor->compute(schur);
                    if (factor->info() == Eigen::Success) {
                        out.reduced_sparse_factor = factor;
                        factor_built = true;
                    }
                }
            }
            if (!factor_built) {
                ensure_lap_schur(false);
                out.reduced_diag_inv.resize(factor_dim);
                for (int i = 0; i < factor_dim; ++i) {
                    const double d = std::max(0.0, lap_diag[i]) + ridge;
                    out.reduced_diag_inv[i] = d > 0.0 ? 1.0 / d : 0.0;
                }
            }
            const double factor_seconds =
                trace_component ? mlsmr_elapsed_seconds(factor_start) : 0.0;
            if (trace_component) {
                std::cerr << "xhdfe_mlsmr_component_trace q=" << q_dim
                          << " r=" << r_dim
                          << " nq=" << nq
                          << " nr=" << nr
                          << " edges=" << comp.edges.size()
                          << " keep=" << n_keep
                          << " elim=" << n_elim
                          << " csr=" << (csr_domain ? 1 : 0)
                          << " map_seconds=" << map_seconds
                          << " star_seconds=" << block_star_seconds
                          << " inv_seconds=" << inv_seconds
                          << " emit_seconds=" << emit_seconds
                          << " sort_seconds=" << sort_seconds
                          << " lap_schur_seconds=" << lap_schur_seconds
                          << " factor_seconds=" << factor_seconds
                          << " block_seconds=" << mlsmr_elapsed_seconds(block_start)
                          << " total_seconds=" << mlsmr_elapsed_seconds(component_trace_start)
                          << std::endl;
            }
        }

        out.block_elim = true;
        out.eliminate_q = eliminate_q;
        out.n_keep = n_keep;
        out.n_elim = n_elim;
        if (csr_domain) {
            out.elim_to_keep_indptr = std::move(block_elim_indptr);
            out.elim_to_keep_indices = std::move(block_elim_indices);
            out.elim_to_keep_data = std::move(block_elim_data);
            out.elim_to_keep.clear();
        } else {
            out.elim_to_keep = std::move(block_elim_to_keep);
        }
        out.inv_diag_elim = std::move(inv_diag_elim);
        return true;
    };

    if (mlsmr_env_enabled("XHDFE_MLSMR_BLOCK_ELIM", true) &&
        try_build_block_elim()) {
        return true;
    }
    if (n_local > full_fallback_max_local) {
        return false;
    }

    double mean_diag = 1.0;
    if (!normalized_operator && n_local > 0) {
        mean_diag = 0.0;
        for (double d : q_diag) {
            mean_diag += d;
        }
        for (double d : r_diag) {
            mean_diag += d;
        }
        mean_diag /= static_cast<double>(n_local);
    }
    const double ridge = ridge_scale > 0.0 ? ridge_scale * std::max(1.0, mean_diag) : 0.0;
    if (n_local <= full_dense_threshold) {
        Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n_local, n_local);
        for (int i = 0; i < n_local; ++i) {
            if (normalized_operator) {
                gram(i, i) = 1.0 + ridge;
            } else {
                const int g = out.global_indices[static_cast<std::size_t>(i)];
                gram(i, i) = std::max(0.0, diagA[g]) + ridge;
            }
        }
        for (const auto& edge : comp.edges) {
            const int qi = q_pos(edge.q);
            const int ri = r_pos(edge.r);
            if (qi < 0 || ri < 0) {
                continue;
            }
            double val = edge.weight;
            if (normalized_operator) {
                const int gq_global = offsets[static_cast<std::size_t>(q_dim)] + edge.q;
                const int gr_global = offsets[static_cast<std::size_t>(r_dim)] + edge.r;
                const double denom = std::sqrt(std::max(0.0, diagA[gq_global]) *
                                               std::max(0.0, diagA[gr_global]));
                if (denom <= 0.0) {
                    continue;
                }
                val /= denom;
            }
            gram(qi, nq + ri) += val;
            gram(nq + ri, qi) += val;
        }
        auto factor = std::make_shared<Eigen::LDLT<Eigen::MatrixXd>>();
        factor->compute(gram);
        if (factor->info() != Eigen::Success) {
            return false;
        }
        out.dense_factor = factor;
        return true;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(n_local) + 2 * comp.edges.size());
    for (int i = 0; i < n_local; ++i) {
        if (normalized_operator) {
            trips.emplace_back(i, i, 1.0 + ridge);
        } else {
            const int g = out.global_indices[static_cast<std::size_t>(i)];
            trips.emplace_back(i, i, std::max(0.0, diagA[g]) + ridge);
        }
    }
    for (const auto& edge : comp.edges) {
        const int qi = q_pos(edge.q);
        const int ri = r_pos(edge.r);
        if (qi < 0 || ri < 0) {
            continue;
        }
        double val = edge.weight;
        if (normalized_operator) {
            const int gq_global = offsets[static_cast<std::size_t>(q_dim)] + edge.q;
            const int gr_global = offsets[static_cast<std::size_t>(r_dim)] + edge.r;
            const double denom = std::sqrt(std::max(0.0, diagA[gq_global]) *
                                           std::max(0.0, diagA[gr_global]));
            if (denom <= 0.0) {
                continue;
            }
            val /= denom;
        }
        trips.emplace_back(qi, nq + ri, val);
        trips.emplace_back(nq + ri, qi, val);
    }

    Eigen::SparseMatrix<double> gram(n_local, n_local);
    gram.setFromTriplets(trips.begin(), trips.end());
    gram.makeCompressed();
    auto factor = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
    factor->compute(gram);
    if (factor->info() != Eigen::Success) {
        return false;
    }
    out.sparse_factor = factor;
    return true;
}

MlsmrAdditiveSchwarzPreconditioner build_mlsmr_additive_preconditioner(
    const std::vector<FeIndexer>& indexers,
    const std::vector<int>& offsets,
    const std::vector<int>& groups_per_dim,
    const Eigen::VectorXd& diagA,
    const double* weight_ptr,
    bool unit_weights,
    bool normalized_operator,
    int n) {
    const bool trace = mlsmr_trace_enabled();
    const auto trace_start = std::chrono::steady_clock::now();
    MlsmrAdditiveSchwarzPreconditioner pre;
    const int dims = static_cast<int>(indexers.size());
    const int total_fe = static_cast<int>(diagA.size());
    if (trace) {
        std::cerr << "xhdfe_mlsmr_trace preconditioner_start dims=" << dims
                  << " total_fe=" << total_fe << " n=" << n << std::endl;
    }
    pre.diagonal_inv.resize(total_fe);
    for (int i = 0; i < total_fe; ++i) {
        if (normalized_operator) {
            pre.diagonal_inv[i] = diagA[i] > 0.0 ? 1.0 : 0.0;
        } else {
            pre.diagonal_inv[i] = diagA[i] > 0.0 ? 1.0 / diagA[i] : 0.0;
        }
    }
    pre.covered.assign(static_cast<std::size_t>(total_fe), 0);

    if (dims < 2) {
        return pre;
    }

    const int max_local = mlsmr_env_positive_int(
        "XHDFE_MLSMR_MAX_LOCAL_DOFS", std::numeric_limits<int>::max());
    const int full_fallback_max_local =
        mlsmr_env_positive_int("XHDFE_MLSMR_FULL_FALLBACK_MAX_LOCAL_DOFS", 20000);
    const int full_dense_threshold =
        mlsmr_env_positive_int("XHDFE_MLSMR_DENSE_THRESHOLD", 192);
    const int dense_schur_threshold =
        mlsmr_env_positive_int("XHDFE_MLSMR_DENSE_SCHUR_THRESHOLD", 24);
    const int exact_star_threshold =
        mlsmr_env_positive_int("XHDFE_MLSMR_EXACT_STAR_THRESHOLD", 16);
    const double ridge_scale = mlsmr_env_nonnegative_double("XHDFE_MLSMR_RIDGE", 1.0e-10);
    const int skip_tiny_keep_threshold =
        mlsmr_env_nonnegative_int("XHDFE_MLSMR_SKIP_TINY_KEEP_THRESHOLD", 64);

    for (int q = 0; q < dims; ++q) {
        for (int r = q + 1; r < dims; ++r) {
            const auto pair_start = std::chrono::steady_clock::now();
            const int nq = groups_per_dim[static_cast<std::size_t>(q)];
            const int nr = groups_per_dim[static_cast<std::size_t>(r)];
            if (nq <= 0 || nr <= 0) {
                continue;
            }
            const bool explicit_skip = mlsmr_pair_explicitly_skipped(q, r);
            const bool tiny_keep_skip =
                skip_tiny_keep_threshold > 0 &&
                std::min(nq, nr) <= skip_tiny_keep_threshold;
            if (explicit_skip || tiny_keep_skip) {
                if (trace) {
                    std::cerr << "xhdfe_mlsmr_trace pair q=" << q << " r=" << r
                              << " nq=" << nq << " nr=" << nr
                              << " skipped=1 reason="
                              << (explicit_skip ? "explicit" : "tiny_keep")
                              << " pair_seconds=" << mlsmr_elapsed_seconds(pair_start)
                              << std::endl;
                }
                continue;
            }

            std::unordered_map<std::uint64_t, double> edge_weights;
            edge_weights.reserve(static_cast<std::size_t>(std::min(n, 1 << 20)));
            const int* gid_q = indexers[static_cast<std::size_t>(q)].group_ids.data();
            const int* gid_r = indexers[static_cast<std::size_t>(r)].group_ids.data();
            for (int i = 0; i < n; ++i) {
                const double w = unit_weights ? 1.0 : weight_ptr[i];
                edge_weights[mlsmr_edge_key(gid_q[i], gid_r[i])] += w;
            }
            const double edge_seconds = mlsmr_elapsed_seconds(pair_start);

            std::vector<MlsmrPairEdge> edges;
            edges.reserve(edge_weights.size());
            MlsmrDsu dsu(nq + nr);
            for (const auto& kv : edge_weights) {
                const int gq = static_cast<int>(kv.first >> 32);
                const int gr = static_cast<int>(kv.first & 0xffffffffu);
                if (gq < 0 || gq >= nq || gr < 0 || gr >= nr || kv.second <= 0.0) {
                    continue;
                }
                edges.push_back({gq, gr, kv.second});
                dsu.unite(gq, nq + gr);
            }

            std::unordered_map<int, int> root_to_component;
            root_to_component.reserve(edges.size());
            std::vector<MlsmrBuildComponent> components;
            std::vector<int> q_owner(static_cast<std::size_t>(nq), -1);
            std::vector<int> r_owner(static_cast<std::size_t>(nr), -1);
            for (const auto& edge : edges) {
                const int root = dsu.find(edge.q);
                auto it = root_to_component.find(root);
                if (it == root_to_component.end()) {
                    const int new_idx = static_cast<int>(components.size());
                    it = root_to_component.emplace(root, new_idx).first;
                    components.emplace_back();
                }
                const int comp_idx = it->second;
                auto& comp = components[static_cast<std::size_t>(comp_idx)];
                if (q_owner[static_cast<std::size_t>(edge.q)] != comp_idx) {
                    q_owner[static_cast<std::size_t>(edge.q)] = comp_idx;
                    comp.q_nodes.push_back(edge.q);
                }
                if (r_owner[static_cast<std::size_t>(edge.r)] != comp_idx) {
                    r_owner[static_cast<std::size_t>(edge.r)] = comp_idx;
                    comp.r_nodes.push_back(edge.r);
                }
                comp.edges.push_back(edge);
            }

            const auto domain_start = std::chrono::steady_clock::now();
            std::vector<MlsmrSchwarzDomain> built_domains;
            auto build_component_domain = [&](std::size_t comp_idx,
                                              std::vector<MlsmrSchwarzDomain>& out_domains) {
                auto& comp = components[comp_idx];
                std::sort(comp.q_nodes.begin(), comp.q_nodes.end());
                std::sort(comp.r_nodes.begin(), comp.r_nodes.end());
                const int local_n =
                    static_cast<int>(comp.q_nodes.size() + comp.r_nodes.size());
                if (local_n <= 0 || local_n > max_local) {
                    return;
                }
                MlsmrSchwarzDomain domain;
                if (build_mlsmr_domain_from_component(comp, q, r, offsets, diagA,
                                                       full_dense_threshold,
                                                       dense_schur_threshold,
                                                       exact_star_threshold,
                                                       full_fallback_max_local,
                                                       ridge_scale,
                                                       normalized_operator,
                                                       domain)) {
                    out_domains.push_back(std::move(domain));
                }
            };

#ifdef HDFE_USE_OPENMP
            const int omp_threads = std::max(1, omp_get_max_threads());
            if (components.size() > 1 && omp_threads > 1) {
                std::vector<std::vector<MlsmrSchwarzDomain>> local_domains(
                    static_cast<std::size_t>(omp_threads));
#pragma omp parallel for schedule(dynamic) num_threads(omp_threads)
                for (int ci = 0; ci < static_cast<int>(components.size()); ++ci) {
                    const int tid = omp_get_thread_num();
                    build_component_domain(
                        static_cast<std::size_t>(ci),
                        local_domains[static_cast<std::size_t>(tid)]);
                }
                std::size_t total_new = 0;
                for (const auto& local : local_domains) {
                    total_new += local.size();
                }
                built_domains.reserve(total_new);
                for (auto& local : local_domains) {
                    std::move(local.begin(), local.end(), std::back_inserter(built_domains));
                }
            } else
#endif
            {
                built_domains.reserve(components.size());
                for (std::size_t ci = 0; ci < components.size(); ++ci) {
                    build_component_domain(ci, built_domains);
                }
            }

            int accepted_domains = 0;
            std::int64_t accepted_local_dofs = 0;
            for (auto& domain : built_domains) {
                ++accepted_domains;
                accepted_local_dofs += domain.n_local;
                pre.domains.push_back(std::move(domain));
            }
            if (trace) {
                std::cerr << "xhdfe_mlsmr_trace pair q=" << q << " r=" << r
                          << " nq=" << nq << " nr=" << nr
                          << " unique_edges=" << edges.size()
                          << " components=" << components.size()
                          << " domains=" << accepted_domains
                          << " local_dofs=" << accepted_local_dofs
                          << " edge_seconds=" << edge_seconds
                          << " domain_seconds=" << mlsmr_elapsed_seconds(domain_start)
                          << " pair_seconds=" << mlsmr_elapsed_seconds(pair_start)
                          << std::endl;
            }
        }
    }

    std::vector<int> cover_count(static_cast<std::size_t>(total_fe), 0);
    for (const auto& domain : pre.domains) {
        for (const int g : domain.global_indices) {
            if (g >= 0 && g < total_fe) {
                ++cover_count[static_cast<std::size_t>(g)];
            }
        }
    }
    pre.domains_disjoint = true;
    for (int count : cover_count) {
        if (count > 1) {
            pre.domains_disjoint = false;
            break;
        }
    }
    for (auto& domain : pre.domains) {
        domain.partition_weights.clear();
        domain.partition_weights.reserve(domain.global_indices.size());
        for (const int g : domain.global_indices) {
            const int count = (g >= 0 && g < total_fe) ? cover_count[static_cast<std::size_t>(g)] : 0;
            domain.partition_weights.push_back(
                count > 0 ? 1.0 / std::sqrt(static_cast<double>(count)) : 1.0);
            if (g >= 0 && g < total_fe) {
                pre.covered[static_cast<std::size_t>(g)] = 1;
            }
        }
    }

    if (trace) {
        std::int64_t total_local_dofs = 0;
        for (const auto& domain : pre.domains) {
            total_local_dofs += domain.n_local;
        }
        std::cerr << "xhdfe_mlsmr_trace preconditioner_done domains="
                  << pre.domains.size() << " total_local_dofs=" << total_local_dofs
                  << " domains_disjoint=" << (pre.domains_disjoint ? 1 : 0)
                  << " seconds=" << mlsmr_elapsed_seconds(trace_start) << std::endl;
    }
    return pre;
}

struct SymOrthoResult {
    double c = 1.0;
    double s = 0.0;
    double r = 0.0;
};

SymOrthoResult sym_ortho(double a, double b) {
    SymOrthoResult out;
    if (b == 0.0) {
        out.c = (a < 0.0) ? -1.0 : 1.0;
        out.s = 0.0;
        out.r = std::abs(a);
    } else if (a == 0.0) {
        out.c = 0.0;
        out.s = (b < 0.0) ? -1.0 : 1.0;
        out.r = std::abs(b);
    } else if (std::abs(b) > std::abs(a)) {
        const double tau = a / b;
        out.s = ((b < 0.0) ? -1.0 : 1.0) / std::sqrt(1.0 + tau * tau);
        out.c = out.s * tau;
        out.r = b / out.s;
    } else {
        const double tau = b / a;
        out.c = ((a < 0.0) ? -1.0 : 1.0) / std::sqrt(1.0 + tau * tau);
        out.s = out.c * tau;
        out.r = a / out.c;
    }
    return out;
}

DenseSchwarzPreconditioner build_dense_schwarz_preconditioner(
    const std::vector<FeIndexer>& indexers,
    const std::vector<int>& offsets,
    const std::vector<int>& groups_per_dim,
    const Eigen::VectorXd& diagA,
    const double* weight_ptr,
    bool unit_weights,
    int n) {
    DenseSchwarzPreconditioner pre;
    const int dims = static_cast<int>(indexers.size());
    const int total_fe = static_cast<int>(diagA.size());
    pre.diagonal_inv.resize(total_fe);
    for (int i = 0; i < total_fe; ++i) {
        pre.diagonal_inv[i] = diagA[i] > 0.0 ? 1.0 / diagA[i] : 0.0;
    }
    pre.covered.assign(static_cast<std::size_t>(total_fe), 0);

    if (dims < 2) {
        return pre;
    }

    struct PendingBlock {
        std::vector<int> global_indices;
        Eigen::LDLT<Eigen::MatrixXd> factor;
    };

    const int max_local = schwarz_max_local_dofs();
    const double ridge_scale = schwarz_ridge_scale();
    std::vector<PendingBlock> pending;
    std::vector<int> cover_count(static_cast<std::size_t>(total_fe), 0);

    for (int q = 0; q < dims; ++q) {
        for (int r = q + 1; r < dims; ++r) {
            const int nq = groups_per_dim[static_cast<std::size_t>(q)];
            const int nr = groups_per_dim[static_cast<std::size_t>(r)];
            const int local_n = nq + nr;
            if (local_n <= 0 || local_n > max_local) {
                continue;
            }

            Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(local_n, local_n);
            const int* gid_q = indexers[static_cast<std::size_t>(q)].group_ids.data();
            const int* gid_r = indexers[static_cast<std::size_t>(r)].group_ids.data();
            for (int i = 0; i < n; ++i) {
                const double w = unit_weights ? 1.0 : weight_ptr[i];
                const int gq = gid_q[i];
                const int gr = gid_r[i];
                const int lr = nq + gr;
                gram(gq, gq) += w;
                gram(lr, lr) += w;
                gram(gq, lr) += w;
                gram(lr, gq) += w;
            }

            const double mean_diag =
                local_n > 0 ? gram.diagonal().sum() / static_cast<double>(local_n) : 1.0;
            const double ridge = ridge_scale * std::max(1.0, mean_diag);
            if (ridge > 0.0) {
                gram.diagonal().array() += ridge;
            }

            Eigen::LDLT<Eigen::MatrixXd> factor;
            factor.compute(gram);
            if (factor.info() != Eigen::Success) {
                continue;
            }

            PendingBlock block;
            block.global_indices.reserve(static_cast<std::size_t>(local_n));
            for (int g = 0; g < nq; ++g) {
                block.global_indices.push_back(offsets[static_cast<std::size_t>(q)] + g);
            }
            for (int g = 0; g < nr; ++g) {
                block.global_indices.push_back(offsets[static_cast<std::size_t>(r)] + g);
            }
            block.factor = std::move(factor);
            for (const int global : block.global_indices) {
                ++cover_count[static_cast<std::size_t>(global)];
            }
            pending.push_back(std::move(block));
        }
    }

    pre.blocks.reserve(pending.size());
    for (PendingBlock& pending_block : pending) {
        DenseSchwarzBlock block;
        block.global_indices = std::move(pending_block.global_indices);
        block.factor = std::move(pending_block.factor);
        block.partition_weights.reserve(block.global_indices.size());
        for (const int global : block.global_indices) {
            const int count = cover_count[static_cast<std::size_t>(global)];
            block.partition_weights.push_back(
                count > 0 ? 1.0 / std::sqrt(static_cast<double>(count)) : 1.0);
            pre.covered[static_cast<std::size_t>(global)] = 1;
        }
        pre.blocks.push_back(std::move(block));
    }

    return pre;
}

AbsorptionResult absorb_fixed_effects_mlsmr(const Eigen::VectorXd& y,
                                             const Eigen::MatrixXd& X,
                                             const std::vector<Eigen::VectorXi>& fes,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options) {
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("Dimension mismatch between y and X");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as y");
    }

    AbsorptionResult result;
    result.y_tilde = y;
    result.X_tilde = X;

    if (fes.empty()) {
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    }
    threads = std::max(1, threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    std::vector<FeIndexer> indexers;
    indexers.reserve(fes.size());
    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        indexers.push_back(build_indexer(raw));
        result.fe_levels.push_back(indexers.back().num_levels_present);
    }

    const double* weight_ptr = weights ? weights->data() : nullptr;
    const bool unit_weights = (weights == nullptr);

    const int dims = static_cast<int>(indexers.size());
    std::vector<int> offsets(static_cast<std::size_t>(dims) + 1, 0);
    for (int d = 0; d < dims; ++d) {
        offsets[static_cast<std::size_t>(d + 1)] =
            offsets[static_cast<std::size_t>(d)] + indexers[static_cast<std::size_t>(d)].num_groups;
    }
    const int total_fe = offsets.back();

    const double lambda = std::max(0.0, options.krylov_lambda);
    Eigen::VectorXd diagA(total_fe);
    for (int d = 0; d < dims; ++d) {
        const auto& idx = indexers[static_cast<std::size_t>(d)];
        Eigen::VectorXd counts =
            compute_weight_sums(idx, weight_ptr, unit_weights, n, threads);
        counts.array() += lambda;
        for (Eigen::Index g = 0; g < counts.size(); ++g) {
            if (counts[g] <= 0.0) {
                counts[g] = lambda > 0.0 ? lambda : 1.0;
            }
        }
        diagA.segment(offsets[static_cast<std::size_t>(d)], idx.num_groups) = counts;
    }

    std::vector<const int*> gid_ptrs(static_cast<std::size_t>(dims));
    std::vector<int> groups_per_dim(static_cast<std::size_t>(dims));
    for (int d = 0; d < dims; ++d) {
        gid_ptrs[static_cast<std::size_t>(d)] =
            indexers[static_cast<std::size_t>(d)].group_ids.data();
        groups_per_dim[static_cast<std::size_t>(d)] =
            indexers[static_cast<std::size_t>(d)].num_groups;
    }

    const bool needs_normal_eq_operator =
        options.absorption_method != AbsorptionMethod::Lsmr &&
        options.absorption_method != AbsorptionMethod::Mlsmr;

    std::vector<std::vector<Eigen::VectorXd>> tls_out;
#ifdef HDFE_USE_OPENMP
    std::vector<std::vector<Eigen::VectorXd>> tls_zt;
    std::vector<std::vector<double*>> tls_out_ptrs;
    std::vector<uint8_t> krylov_atomic_scatter(static_cast<std::size_t>(dims), 0);
    const bool allow_krylov_atomic_scatter =
        mlsmr_env_enabled("XHDFE_KRYLOV_ATOMIC_SCATTER", false);
    const int krylov_atomic_threshold =
        mlsmr_env_nonnegative_int("XHDFE_KRYLOV_ATOMIC_SCATTER_THRESHOLD", 100000);
    if (threads > 1 && allow_krylov_atomic_scatter && krylov_atomic_threshold > 0) {
        for (int d = 0; d < dims; ++d) {
            if (groups_per_dim[static_cast<std::size_t>(d)] >= krylov_atomic_threshold) {
                krylov_atomic_scatter[static_cast<std::size_t>(d)] = 1;
            }
        }
    }
    if (mlsmr_trace_enabled() &&
        (options.absorption_method == AbsorptionMethod::Lsmr ||
         options.absorption_method == AbsorptionMethod::Mlsmr)) {
        std::cerr << "xhdfe_krylov_trace atomic_scatter="
                  << (allow_krylov_atomic_scatter ? 1 : 0)
                  << " threshold=" << krylov_atomic_threshold
                  << " dims=";
        for (int d = 0; d < dims; ++d) {
            if (d > 0) {
                std::cerr << ",";
            }
            std::cerr << static_cast<int>(krylov_atomic_scatter[static_cast<std::size_t>(d)]);
        }
        std::cerr << std::endl;
    }
#endif
#ifdef HDFE_USE_OPENMP
    if (threads > 1) {
        if (needs_normal_eq_operator) {
            tls_out.resize(static_cast<std::size_t>(dims));
        }
        tls_zt.resize(static_cast<std::size_t>(dims));
        for (int d = 0; d < dims; ++d) {
            if (needs_normal_eq_operator) {
                tls_out[static_cast<std::size_t>(d)].assign(
                    static_cast<std::size_t>(threads),
                    Eigen::VectorXd::Zero(groups_per_dim[static_cast<std::size_t>(d)]));
            }
            if (!krylov_atomic_scatter[static_cast<std::size_t>(d)]) {
                tls_zt[static_cast<std::size_t>(d)].assign(
                    static_cast<std::size_t>(threads),
                    Eigen::VectorXd::Zero(groups_per_dim[static_cast<std::size_t>(d)]));
            }
        }
        if (needs_normal_eq_operator) {
            tls_out_ptrs.assign(static_cast<std::size_t>(threads),
                                std::vector<double*>(static_cast<std::size_t>(dims)));
            for (int t = 0; t < threads; ++t) {
                for (int d = 0; d < dims; ++d) {
                    tls_out_ptrs[static_cast<std::size_t>(t)][static_cast<std::size_t>(d)] =
                        tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)].data();
                }
            }
        }
    }
#endif
    std::vector<const double*> x_ptrs(static_cast<std::size_t>(dims));
    std::vector<double*> out_ptrs(static_cast<std::size_t>(dims));

    auto compute_Zt = [&](const Eigen::Ref<const Eigen::VectorXd>& v) {
        Eigen::VectorXd b(total_fe);
        b.setZero();
        const double* v_ptr = v.data();
        for (int d = 0; d < dims; ++d) {
            const int groups = groups_per_dim[static_cast<std::size_t>(d)];
            double* b_seg = b.data() + offsets[static_cast<std::size_t>(d)];
#ifdef HDFE_USE_OPENMP
            if (threads > 1 && krylov_atomic_scatter[static_cast<std::size_t>(d)]) {
                const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
                if (unit_weights) {
#pragma omp parallel for schedule(static) num_threads(threads)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
#pragma omp atomic update
                        b_seg[g] += v_ptr[i];
                    }
                } else {
#pragma omp parallel for schedule(static) num_threads(threads)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        const double value = weight_ptr[i] * v_ptr[i];
#pragma omp atomic update
                        b_seg[g] += value;
                    }
                }
            } else if (threads > 1) {
#pragma omp parallel num_threads(threads)
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& local =
                        tls_zt[static_cast<std::size_t>(d)][static_cast<std::size_t>(tid)];
                    local.setZero();
                    const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        if (unit_weights) {
                            local[g] += v_ptr[i];
                        } else {
                            local[g] += weight_ptr[i] * v_ptr[i];
                        }
                    }
                }
                Eigen::Map<Eigen::VectorXd> b_seg_vec(b_seg, groups);
                for (const auto& local : tls_zt[static_cast<std::size_t>(d)]) {
                    b_seg_vec.noalias() += local;
                }
            } else
#endif
            {
                const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
                if (unit_weights) {
                    for (int i = 0; i < n; ++i) {
                        b_seg[gid[i]] += v_ptr[i];
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        b_seg[gid[i]] += weight_ptr[i] * v_ptr[i];
                    }
                }
            }
        }
        return b;
    };

    auto apply_A = [&](const Eigen::VectorXd& x, Eigen::VectorXd& out) {
        for (int d = 0; d < dims; ++d) {
            x_ptrs[static_cast<std::size_t>(d)] =
                x.data() + offsets[static_cast<std::size_t>(d)];
        }

#ifdef HDFE_USE_OPENMP
        if (threads > 1) {
            for (int d = 0; d < dims; ++d) {
                for (int t = 0; t < threads; ++t) {
                    tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)].setZero();
                }
            }
#pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                auto& local_out = tls_out_ptrs[static_cast<std::size_t>(tid)];
                if (dims == 1) {
                    const int* gid0 = gid_ptrs[0];
                    double* out0 = local_out[0];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const double sum = x_ptrs[0][g0];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                    }
                } else if (dims == 2) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const double sum = x_ptrs[0][g0] + x_ptrs[1][g1];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                    }
                } else if (dims == 3) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    const int* gid2 = gid_ptrs[2];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
                    double* out2 = local_out[2];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const int g2 = gid2[i];
                        const double sum =
                            x_ptrs[0][g0] + x_ptrs[1][g1] + x_ptrs[2][g2];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                        out2[g2] += ws;
                    }
                } else if (dims == 4) {
                    const int* gid0 = gid_ptrs[0];
                    const int* gid1 = gid_ptrs[1];
                    const int* gid2 = gid_ptrs[2];
                    const int* gid3 = gid_ptrs[3];
                    double* out0 = local_out[0];
                    double* out1 = local_out[1];
                    double* out2 = local_out[2];
                    double* out3 = local_out[3];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g0 = gid0[i];
                        const int g1 = gid1[i];
                        const int g2 = gid2[i];
                        const int g3 = gid3[i];
                        const double sum = x_ptrs[0][g0] + x_ptrs[1][g1] +
                                           x_ptrs[2][g2] + x_ptrs[3][g3];
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        out0[g0] += ws;
                        out1[g1] += ws;
                        out2[g2] += ws;
                        out3[g3] += ws;
                    }
                } else {
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        double sum = 0.0;
                        for (int d = 0; d < dims; ++d) {
                            const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                            sum += x_ptrs[static_cast<std::size_t>(d)][g];
                        }
                        const double w = unit_weights ? 1.0 : weight_ptr[i];
                        const double ws = w * sum;
                        for (int d = 0; d < dims; ++d) {
                            const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                            local_out[static_cast<std::size_t>(d)][g] += ws;
                        }
                    }
                }
            }

            for (int d = 0; d < dims; ++d) {
                Eigen::Map<Eigen::VectorXd> out_seg(
                    out.data() + offsets[static_cast<std::size_t>(d)],
                    groups_per_dim[static_cast<std::size_t>(d)]);
                out_seg = tls_out[static_cast<std::size_t>(d)][0];
                for (int t = 1; t < threads; ++t) {
                    out_seg.noalias() +=
                        tls_out[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)];
                }
            }
        } else
#endif
        {
            out.setZero(total_fe);
            for (int d = 0; d < dims; ++d) {
                out_ptrs[static_cast<std::size_t>(d)] =
                    out.data() + offsets[static_cast<std::size_t>(d)];
            }
            if (dims == 1) {
                const int* gid0 = gid_ptrs[0];
                double* out0 = out_ptrs[0];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const double sum = x_ptrs[0][g0];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    out0[g0] += w * sum;
                }
            } else if (dims == 2) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const double sum = x_ptrs[0][g0] + x_ptrs[1][g1];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                }
            } else if (dims == 3) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                const int* gid2 = gid_ptrs[2];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                double* out2 = out_ptrs[2];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const int g2 = gid2[i];
                    const double sum =
                        x_ptrs[0][g0] + x_ptrs[1][g1] + x_ptrs[2][g2];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                    out2[g2] += ws;
                }
            } else if (dims == 4) {
                const int* gid0 = gid_ptrs[0];
                const int* gid1 = gid_ptrs[1];
                const int* gid2 = gid_ptrs[2];
                const int* gid3 = gid_ptrs[3];
                double* out0 = out_ptrs[0];
                double* out1 = out_ptrs[1];
                double* out2 = out_ptrs[2];
                double* out3 = out_ptrs[3];
                for (int i = 0; i < n; ++i) {
                    const int g0 = gid0[i];
                    const int g1 = gid1[i];
                    const int g2 = gid2[i];
                    const int g3 = gid3[i];
                    const double sum = x_ptrs[0][g0] + x_ptrs[1][g1] +
                                       x_ptrs[2][g2] + x_ptrs[3][g3];
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    out0[g0] += ws;
                    out1[g1] += ws;
                    out2[g2] += ws;
                    out3[g3] += ws;
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    double sum = 0.0;
                    for (int d = 0; d < dims; ++d) {
                        const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                        sum += x_ptrs[static_cast<std::size_t>(d)][g];
                    }
                    const double w = unit_weights ? 1.0 : weight_ptr[i];
                    const double ws = w * sum;
                    for (int d = 0; d < dims; ++d) {
                        const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                        out_ptrs[static_cast<std::size_t>(d)][g] += ws;
                    }
                }
            }
        }

        if (lambda > 0.0) {
            out.noalias() += lambda * x;
        }
    };

    const bool use_schwarz_preconditioner =
        options.absorption_method == AbsorptionMethod::Schwarz;
    DenseSchwarzPreconditioner schwarz_preconditioner;
    if (use_schwarz_preconditioner) {
        schwarz_preconditioner = build_dense_schwarz_preconditioner(
            indexers, offsets, groups_per_dim, diagA, weight_ptr, unit_weights, n);
    }

    auto pcg_solve = [&](const Eigen::VectorXd& b, int& iters, bool& ok) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(total_fe);
        const double b_norm_sq = b.squaredNorm();
        if (b_norm_sq == 0.0) {
            iters = 0;
            ok = true;
            return x;
        }
        const double convergence_tol = effective_absorption_tolerance(options);
        const double tol_sq = convergence_tol * convergence_tol;

        Eigen::VectorXd r = b;
        Eigen::VectorXd z(total_fe);
        if (use_schwarz_preconditioner) {
            schwarz_preconditioner.apply(r, z);
            if (!z.allFinite() || r.dot(z) <= 0.0) {
                z = r.cwiseQuotient(diagA);
            }
        } else {
            z = r.cwiseQuotient(diagA);
        }
        Eigen::VectorXd p = z;
        Eigen::VectorXd Ap(total_fe);
        double rz_old = r.dot(z);
        ok = false;
        int k = 0;
        for (; k < options.max_iter; ++k) {
            apply_A(p, Ap);
            const double denom = p.dot(Ap);
            if (denom == 0.0) {
                break;
            }
            const double alpha = rz_old / denom;
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;

            const double r_norm_sq = r.squaredNorm();
            if (r_norm_sq <= tol_sq * b_norm_sq) {
                ok = true;
                ++k;
                break;
            }

            if (use_schwarz_preconditioner) {
                schwarz_preconditioner.apply(r, z);
                if (!z.allFinite() || r.dot(z) <= 0.0) {
                    z = r.cwiseQuotient(diagA);
                }
            } else {
                z = r.cwiseQuotient(diagA);
            }
            const double rz_new = r.dot(z);
            if (!std::isfinite(rz_new) || rz_new <= 0.0) {
                break;
            }
            const double beta = rz_new / rz_old;
            p = z + beta * p;
            rz_old = rz_new;
        }
        iters = k;
        return x;
    };

    auto subtract_projection = [&](Eigen::Ref<Eigen::VectorXd> v,
                                   const Eigen::VectorXd& alpha) {
        double* v_ptr = v.data();
        if (dims == 1) {
            const int* gid0 = gid_ptrs[0];
            const int offset0 = offsets[0];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                v_ptr[i] -= alpha[offset0 + g0];
            }
        } else if (dims == 2) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const double hat = alpha[offset0 + g0] + alpha[offset1 + g1];
                v_ptr[i] -= hat;
            }
        } else if (dims == 3) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int* gid2 = gid_ptrs[2];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
            const int offset2 = offsets[2];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const int g2 = gid2[i];
                const double hat =
                    alpha[offset0 + g0] + alpha[offset1 + g1] + alpha[offset2 + g2];
                v_ptr[i] -= hat;
            }
        } else if (dims == 4) {
            const int* gid0 = gid_ptrs[0];
            const int* gid1 = gid_ptrs[1];
            const int* gid2 = gid_ptrs[2];
            const int* gid3 = gid_ptrs[3];
            const int offset0 = offsets[0];
            const int offset1 = offsets[1];
            const int offset2 = offsets[2];
            const int offset3 = offsets[3];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                const int g0 = gid0[i];
                const int g1 = gid1[i];
                const int g2 = gid2[i];
                const int g3 = gid3[i];
                const double hat = alpha[offset0 + g0] + alpha[offset1 + g1] +
                                   alpha[offset2 + g2] + alpha[offset3 + g3];
                v_ptr[i] -= hat;
            }
        } else {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                double hat = 0.0;
                for (int d = 0; d < dims; ++d) {
                    const int g = gid_ptrs[static_cast<std::size_t>(d)][i];
                    hat += alpha[offsets[static_cast<std::size_t>(d)] + g];
                }
                v_ptr[i] -= hat;
            }
        }
    };

    const bool mlsmr_use_column_scaling =
        options.absorption_method == AbsorptionMethod::Mlsmr &&
        mlsmr_env_enabled("XHDFE_MLSMR_COLUMN_SCALE", false);
    const bool use_lsmr_column_scaling =
        options.absorption_method == AbsorptionMethod::Lsmr || mlsmr_use_column_scaling;
    Eigen::VectorXd column_scale(total_fe);
    for (int i = 0; i < total_fe; ++i) {
        if (use_lsmr_column_scaling) {
            column_scale[i] = diagA[i] > 0.0 ? 1.0 / std::sqrt(diagA[i]) : 0.0;
        } else {
            column_scale[i] = diagA[i] > 0.0 ? 1.0 : 0.0;
        }
    }

    Eigen::VectorXd sqrt_weights;
    if (!unit_weights) {
        sqrt_weights.resize(n);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int i = 0; i < n; ++i) {
            sqrt_weights[i] = std::sqrt(std::max(0.0, weight_ptr[i]));
        }
    }

    auto make_lsmr_rhs = [&](const Eigen::Ref<const Eigen::VectorXd>& v) {
        Eigen::VectorXd b(n);
        if (unit_weights) {
            b = v;
        } else {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int i = 0; i < n; ++i) {
                b[i] = sqrt_weights[i] * v[i];
            }
        }
        return b;
    };

    auto apply_lsmr_A = [&](const Eigen::VectorXd& x, Eigen::VectorXd& out) {
        out.resize(n);
        // Pre-scale x by column_scale once (O(total_fe)): the row loop then
        // gathers a single array instead of two dependent random gathers per
        // row per dim. Same multiply, same accumulation order per row, so the
        // results are bit-identical. thread_local: batch lanes run this lambda
        // concurrently and must not share the scratch.
        static thread_local std::vector<double> sx_buf;
        sx_buf.resize(static_cast<std::size_t>(total_fe));
        double* __restrict sx = sx_buf.data();
        const double* __restrict x_ptr = x.data();
        const double* __restrict cs = column_scale.data();
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int g = 0; g < total_fe; ++g) {
            sx[g] = cs[g] * x_ptr[g];
        }
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int d = 0; d < dims; ++d) {
                sum += sx[offsets[static_cast<std::size_t>(d)] +
                          gid_ptrs[static_cast<std::size_t>(d)][i]];
            }
            out[i] = (unit_weights ? 1.0 : sqrt_weights[i]) * sum;
        }
    };

    auto make_lsmr_At_tls = [&]() {
        std::vector<std::vector<Eigen::VectorXd>> local_tls;
#ifdef HDFE_USE_OPENMP
        // Inside a batch lane (outer parallel-for over RHS columns) nested
        // parallelism is inactive, so the scatter team has exactly one
        // thread: allocate one accumulator per dim instead of `threads`.
        // Removes an O(threads x total_fe) always-zero reduce per apply and
        // the matching RSS spike; the surviving accumulation is unchanged.
        const int tls_threads = omp_in_parallel() ? 1 : threads;
        if (threads > 1) {
            local_tls.resize(static_cast<std::size_t>(dims));
            for (int d = 0; d < dims; ++d) {
                if (!krylov_atomic_scatter[static_cast<std::size_t>(d)]) {
                    local_tls[static_cast<std::size_t>(d)].assign(
                        static_cast<std::size_t>(tls_threads),
                        Eigen::VectorXd::Zero(groups_per_dim[static_cast<std::size_t>(d)]));
                }
            }
        }
#endif
        return local_tls;
    };

    auto apply_lsmr_At_impl =
        [&](const Eigen::VectorXd& u,
            std::vector<std::vector<Eigen::VectorXd>>* tls_override) {
        Eigen::VectorXd out(total_fe);
        out.setZero();
        const double* u_ptr = u.data();
        for (int d = 0; d < dims; ++d) {
            const int groups = groups_per_dim[static_cast<std::size_t>(d)];
            const int offset = offsets[static_cast<std::size_t>(d)];
            double* out_seg = out.data() + offset;
#ifdef HDFE_USE_OPENMP
            if (threads > 1 && krylov_atomic_scatter[static_cast<std::size_t>(d)]) {
                const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
#pragma omp parallel for schedule(static) num_threads(threads)
                for (int i = 0; i < n; ++i) {
                    const double sw = unit_weights ? 1.0 : sqrt_weights[i];
                    const double value = sw * u_ptr[i];
#pragma omp atomic update
                    out_seg[gid[i]] += value;
                }
            } else if (threads > 1) {
                auto& zt_storage = tls_override ? *tls_override : tls_zt;
                const int tls_count = static_cast<int>(
                    zt_storage[static_cast<std::size_t>(d)].size());
#pragma omp parallel num_threads(std::min(threads, tls_count))
                {
                    const int tid = omp_get_thread_num();
                    Eigen::VectorXd& local =
                        zt_storage[static_cast<std::size_t>(d)][static_cast<std::size_t>(tid)];
                    local.setZero();
                    const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const double sw = unit_weights ? 1.0 : sqrt_weights[i];
                        local[gid[i]] += sw * u_ptr[i];
                    }
                }
                Eigen::Map<Eigen::VectorXd> out_seg_vec(out_seg, groups);
                for (const auto& local : zt_storage[static_cast<std::size_t>(d)]) {
                    out_seg_vec.noalias() += local;
                }
            } else
#endif
            {
                const int* gid = gid_ptrs[static_cast<std::size_t>(d)];
                for (int i = 0; i < n; ++i) {
                    const double sw = unit_weights ? 1.0 : sqrt_weights[i];
                    out_seg[gid[i]] += sw * u_ptr[i];
                }
            }
            Eigen::Map<Eigen::VectorXd> scaled(out_seg, groups);
            scaled.array() *= column_scale.segment(offset, groups).array();
        }
        return out;
    };

    auto apply_lsmr_At = [&](const Eigen::VectorXd& u) {
        return apply_lsmr_At_impl(u, nullptr);
    };

    const bool use_mlsmr_preconditioner =
        options.absorption_method == AbsorptionMethod::Mlsmr;
    MlsmrAdditiveSchwarzPreconditioner mlsmr_preconditioner;
    if (use_mlsmr_preconditioner) {
        mlsmr_preconditioner = build_mlsmr_additive_preconditioner(
            indexers, offsets, groups_per_dim, diagA, weight_ptr, unit_weights,
            use_lsmr_column_scaling, n);
    }

    auto lsmr_solve = [&](const Eigen::Ref<const Eigen::VectorXd>& raw_rhs,
                          int& iters,
                          bool& ok) {
        const double atol = effective_absorption_tolerance(options);
        const double btol = atol;
        constexpr double kConlim = 1.0e8;
        constexpr double kHuge = 1.0e100;

        const Eigen::VectorXd b = make_lsmr_rhs(raw_rhs);
        const double normb = b.norm();
        Eigen::VectorXd x = Eigen::VectorXd::Zero(total_fe);
        if (normb == 0.0) {
            iters = 0;
            ok = true;
            return x;
        }

        Eigen::VectorXd u = b;
        double beta = normb;
        u /= beta;

        Eigen::VectorXd v = apply_lsmr_At(u);
        double alpha = v.norm();
        if (alpha > 0.0) {
            v /= alpha;
        } else {
            iters = 0;
            ok = true;
            return x;
        }

        double zetabar = alpha * beta;
        double alphabar = alpha;
        double rho = 1.0;
        double rhobar = 1.0;
        double cbar = 1.0;
        double sbar = 0.0;

        Eigen::VectorXd h = v;
        Eigen::VectorXd hbar = Eigen::VectorXd::Zero(total_fe);
        Eigen::VectorXd Av(n);

        double betadd = beta;
        double betad = 0.0;
        double rhodold = 1.0;
        double tautildeold = 0.0;
        double thetatilde = 0.0;
        double zeta = 0.0;
        double dnorm = 0.0;

        double normA2 = alpha * alpha;
        double maxrbar = 0.0;
        double minrbar = kHuge;
        double normA = std::sqrt(normA2);
        double condA = 1.0;
        double normr = beta;
        double normar = alpha * beta;
        const double ctol = 1.0 / kConlim;

        ok = false;
        int k = 0;
        for (; k < options.max_iter; ++k) {
            apply_lsmr_A(v, Av);
            u = Av - alpha * u;
            beta = u.norm();

            if (beta > 0.0) {
                u /= beta;
                Eigen::VectorXd Atu = apply_lsmr_At(u);
                v = Atu - beta * v;
                alpha = v.norm();
                if (alpha > 0.0) {
                    v /= alpha;
                }
            }

            const SymOrthoResult qhat = sym_ortho(alphabar, 0.0);
            const double alphahat = qhat.r;

            const double rhoold = rho;
            const SymOrthoResult q = sym_ortho(alphahat, beta);
            const double c = q.c;
            const double s = q.s;
            rho = q.r;
            if (rho == 0.0) {
                break;
            }
            const double thetanew = s * alpha;
            alphabar = c * alpha;

            const double rhobarold = rhobar;
            const double zetaold = zeta;
            const double thetabar = sbar * rho;
            const double rhotemp = cbar * rho;
            const SymOrthoResult qbar = sym_ortho(rhotemp, thetanew);
            cbar = qbar.c;
            sbar = qbar.s;
            rhobar = qbar.r;
            if (rhobar == 0.0 || rhobarold == 0.0 || rhoold == 0.0) {
                break;
            }
            zeta = cbar * zetabar;
            zetabar = -sbar * zetabar;

            hbar *= -(thetabar * rho / (rhoold * rhobarold));
            hbar += h;
            x.noalias() += (zeta / (rho * rhobar)) * hbar;
            h *= -(thetanew / rho);
            h += v;

            const double betaacute = qhat.c * betadd;
            const double betacheck = -qhat.s * betadd;
            const double betahat = c * betaacute;
            betadd = -s * betaacute;

            const double thetatildeold = thetatilde;
            const SymOrthoResult qtilde = sym_ortho(rhodold, thetabar);
            if (qtilde.r == 0.0) {
                break;
            }
            thetatilde = qtilde.s * rhobar;
            rhodold = qtilde.c * rhobar;
            if (rhodold == 0.0) {
                break;
            }
            betad = -qtilde.s * betad + qtilde.c * betahat;

            tautildeold = (zetaold - thetatildeold * tautildeold) / qtilde.r;
            const double taud = (zeta - thetatilde * tautildeold) / rhodold;
            dnorm += betacheck * betacheck;
            normr = std::sqrt(std::max(0.0,
                                       dnorm + (betad - taud) * (betad - taud) +
                                           betadd * betadd));

            normA2 += beta * beta;
            normA = std::sqrt(normA2);
            normA2 += alpha * alpha;

            maxrbar = std::max(maxrbar, rhobarold);
            if (k > 0) {
                minrbar = std::min(minrbar, rhobarold);
            }
            const double cond_denom = std::min(minrbar, rhotemp);
            condA = cond_denom > 0.0 ? std::max(maxrbar, rhotemp) / cond_denom
                                     : std::numeric_limits<double>::infinity();

            normar = std::abs(zetabar);
            const double normx = x.norm();
            const double test1 = normr / normb;
            const double test2 =
                (normA * normr) != 0.0 ? normar / (normA * normr)
                                       : std::numeric_limits<double>::infinity();
            const double test3 = condA > 0.0 ? 1.0 / condA : 0.0;
            const double rtol = btol + atol * normA * normx / normb;

            if (test1 <= rtol || test2 <= atol) {
                ok = true;
                ++k;
                break;
            }
            if (test3 <= ctol) {
                ++k;
                break;
            }
        }

        iters = k;
        return x;
    };

    auto mlsmr_solve = [&](const Eigen::Ref<const Eigen::VectorXd>& raw_rhs,
                           int& iters,
                           bool& ok,
                           std::vector<std::vector<Eigen::VectorXd>>* tls_override) {
        const bool trace = mlsmr_trace_enabled();
        const int trace_every =
            trace ? mlsmr_env_positive_int("XHDFE_MLSMR_TRACE_EVERY", 10) : 0;
        const double atol = effective_absorption_tolerance(options);
        const double btol = atol;
        constexpr double kConlim = 1.0e8;
        constexpr double kHuge = 1.0e100;

        const Eigen::VectorXd b = make_lsmr_rhs(raw_rhs);
        const double normb = b.norm();
        Eigen::VectorXd x = Eigen::VectorXd::Zero(total_fe);
        if (normb == 0.0) {
            iters = 0;
            ok = true;
            return x;
        }

        const int local_size = mlsmr_env_nonnegative_int("XHDFE_MLSMR_LOCAL_SIZE", 0);
        std::vector<Eigen::VectorXd> v_window;
        std::vector<Eigen::VectorXd> p_window;
        int window_next = 0;
        int window_count = 0;
        if (local_size > 0) {
            v_window.resize(static_cast<std::size_t>(local_size),
                            Eigen::VectorXd::Zero(total_fe));
            p_window.resize(static_cast<std::size_t>(local_size),
                            Eigen::VectorXd::Zero(total_fe));
        }

        Eigen::VectorXd u = b;
        double beta = normb;
        u /= beta;

        Eigen::VectorXd p_tilde = apply_lsmr_At_impl(u, tls_override);
        Eigen::VectorXd v(total_fe);

        auto reorthogonalize_v = [&]() {
            if (local_size <= 0 || window_count == 0) {
                return;
            }
            for (int i = 0; i < window_count; ++i) {
                const int slot =
                    (window_count < local_size) ? i : ((window_next + i) % local_size);
                const double coeff = v.dot(p_window[static_cast<std::size_t>(slot)]);
                v.noalias() -= coeff * v_window[static_cast<std::size_t>(slot)];
                p_tilde.noalias() -= coeff * p_window[static_cast<std::size_t>(slot)];
            }
        };

        auto push_reorthogonalization_pair = [&](double current_alpha) {
            if (local_size <= 0 || current_alpha <= 0.0) {
                return;
            }
            const std::size_t slot = static_cast<std::size_t>(window_next);
            v_window[slot] = v;
            p_window[slot].noalias() = p_tilde / current_alpha;
            window_next = (window_next + 1) % local_size;
            window_count = std::min(window_count + 1, local_size);
        };

        auto precondition_and_measure = [&]() {
            mlsmr_preconditioner.apply(p_tilde, v);
            if (!v.allFinite()) {
                v = p_tilde;
                return v.squaredNorm();
            }
            reorthogonalize_v();
            double measured = v.dot(p_tilde);
            if (!std::isfinite(measured) || measured < 0.0) {
                v = p_tilde;
                measured = v.squaredNorm();
            }
            return measured;
        };

        double vp = precondition_and_measure();
        if (!std::isfinite(vp) || vp < 0.0) {
            v = p_tilde;
            vp = v.squaredNorm();
        }
        double alpha = std::sqrt(std::max(0.0, vp));
        if (alpha > 0.0) {
            v /= alpha;
            push_reorthogonalization_pair(alpha);
        } else {
            iters = 0;
            ok = true;
            return x;
        }

        double beta_prev_inv = 1.0;
        double zetabar = alpha * beta;
        double alphabar = alpha;
        double rho = 1.0;
        double rhobar = 1.0;
        double cbar = 1.0;
        double sbar = 0.0;

        Eigen::VectorXd h = v;
        Eigen::VectorXd hbar = Eigen::VectorXd::Zero(total_fe);
        Eigen::VectorXd Av(n);

        double betadd = beta;
        double betad = 0.0;
        double rhodold = 1.0;
        double tautildeold = 0.0;
        double thetatilde = 0.0;
        double zeta = 0.0;
        double dnorm = 0.0;

        double normA2 = alpha * alpha;
        double maxrbar = 0.0;
        double minrbar = kHuge;
        double normA = std::sqrt(normA2);
        double condA = 1.0;
        double normr = beta;
        double normar = alpha * beta;
        const double ctol = 1.0 / kConlim;

        ok = false;
        int k = 0;
        for (; k < options.max_iter; ++k) {
            apply_lsmr_A(v, Av);
            u = Av - (alpha * beta_prev_inv) * u;
            beta = u.norm();
            if (beta == 0.0) {
                alpha = 0.0;
                v.setZero();
                p_tilde.setZero();
                beta_prev_inv = 0.0;
            } else {
                const double beta_inv = 1.0 / beta;
                Eigen::VectorXd Atu = apply_lsmr_At_impl(u, tls_override);
                p_tilde = beta_inv * Atu - (beta / alpha) * p_tilde;
                vp = precondition_and_measure();
                alpha = std::sqrt(std::max(0.0, vp));
                if (alpha > 0.0) {
                    v /= alpha;
                    push_reorthogonalization_pair(alpha);
                } else {
                    v.setZero();
                    p_tilde.setZero();
                }
                beta_prev_inv = beta_inv;
            }

            const SymOrthoResult qhat = sym_ortho(alphabar, 0.0);
            const double alphahat = qhat.r;

            const double rhoold = rho;
            const SymOrthoResult q = sym_ortho(alphahat, beta);
            const double c = q.c;
            const double s = q.s;
            rho = q.r;
            if (rho == 0.0) {
                break;
            }
            const double thetanew = s * alpha;
            alphabar = c * alpha;

            const double rhobarold = rhobar;
            const double zetaold = zeta;
            const double thetabar = sbar * rho;
            const double rhotemp = cbar * rho;
            const SymOrthoResult qbar = sym_ortho(rhotemp, thetanew);
            cbar = qbar.c;
            sbar = qbar.s;
            rhobar = qbar.r;
            if (rhobar == 0.0 || rhobarold == 0.0 || rhoold == 0.0) {
                break;
            }
            zeta = cbar * zetabar;
            zetabar = -sbar * zetabar;

            hbar *= -(thetabar * rho / (rhoold * rhobarold));
            hbar += h;
            x.noalias() += (zeta / (rho * rhobar)) * hbar;
            h *= -(thetanew / rho);
            h += v;

            const double betaacute = qhat.c * betadd;
            const double betacheck = -qhat.s * betadd;
            const double betahat = c * betaacute;
            betadd = -s * betaacute;

            const double thetatildeold = thetatilde;
            const SymOrthoResult qtilde = sym_ortho(rhodold, thetabar);
            if (qtilde.r == 0.0) {
                break;
            }
            thetatilde = qtilde.s * rhobar;
            rhodold = qtilde.c * rhobar;
            if (rhodold == 0.0) {
                break;
            }
            betad = -qtilde.s * betad + qtilde.c * betahat;

            tautildeold = (zetaold - thetatildeold * tautildeold) / qtilde.r;
            const double taud = (zeta - thetatilde * tautildeold) / rhodold;
            dnorm += betacheck * betacheck;
            normr = std::sqrt(std::max(0.0,
                                       dnorm + (betad - taud) * (betad - taud) +
                                           betadd * betadd));

            normA2 += beta * beta;
            normA = std::sqrt(normA2);
            normA2 += alpha * alpha;

            maxrbar = std::max(maxrbar, rhobarold);
            if (k > 0) {
                minrbar = std::min(minrbar, rhobarold);
            }
            const double cond_denom = std::min(minrbar, rhotemp);
            condA = cond_denom > 0.0 ? std::max(maxrbar, rhotemp) / cond_denom
                                     : std::numeric_limits<double>::infinity();

            normar = std::abs(zetabar);
            const double normx = x.norm();
            const double test1 = normr / normb;
            const double test2 =
                (normA * normr) != 0.0 ? normar / (normA * normr)
                                       : std::numeric_limits<double>::infinity();
            const double test3 = condA > 0.0 ? 1.0 / condA : 0.0;
            const double rtol = btol + atol * normA * normx / normb;

            if (test1 <= rtol || test2 <= atol) {
                ok = true;
                ++k;
                break;
            }
            if (test3 <= ctol) {
                ++k;
                break;
            }
            if (alpha == 0.0) {
                ok = true;
                ++k;
                break;
            }
            if (trace && trace_every > 0 && ((k + 1) % trace_every == 0)) {
                std::cerr << "xhdfe_mlsmr_trace solve_iter iter=" << (k + 1)
                          << " test1=" << test1 << " test2=" << test2
                          << " condA=" << condA << " normr=" << normr
                          << std::endl;
            }
        }

        iters = k;
        return x;
    };

    auto timed_mlsmr_solve = [&](const Eigen::Ref<const Eigen::VectorXd>& rhs,
                                 const char* label,
                                 int& iters,
                                 bool& ok,
                                 std::vector<std::vector<Eigen::VectorXd>>* tls_override) {
        const bool trace = mlsmr_trace_enabled();
        const auto start = std::chrono::steady_clock::now();
        if (trace) {
            std::cerr << "xhdfe_mlsmr_trace solve_start label=" << label
                      << " rhs_size=" << rhs.size() << std::endl;
        }
        Eigen::VectorXd beta = mlsmr_solve(rhs, iters, ok, tls_override);
        if (trace) {
            std::cerr << "xhdfe_mlsmr_trace solve_done label=" << label
                      << " iters=" << iters << " ok=" << (ok ? 1 : 0)
                      << " seconds=" << mlsmr_elapsed_seconds(start) << std::endl;
        }
        return beta;
    };

    int max_iters_used = 0;
    bool all_converged = true;

    int iters_y = 0;
    bool ok_y = true;
    Eigen::VectorXd alpha_y;
    const int cols = static_cast<int>(result.X_tilde.cols());
    bool mlsmr_batch_rhs_safe = false;
#ifdef HDFE_USE_OPENMP
    if (options.absorption_method == AbsorptionMethod::Mlsmr && threads > 1) {
        mlsmr_batch_rhs_safe = true;
    }
#endif
    const bool use_mlsmr_batch_rhs =
        options.absorption_method == AbsorptionMethod::Mlsmr &&
        cols > 0 &&
        mlsmr_env_enabled("XHDFE_MLSMR_BATCH_RHS", true) &&
        mlsmr_batch_rhs_safe;
    if (use_mlsmr_batch_rhs) {
        const int rhs_count = cols + 1;
        std::vector<Eigen::VectorXd> alphas(static_cast<std::size_t>(rhs_count));
        std::vector<int> rhs_iters(static_cast<std::size_t>(rhs_count), 0);
        std::vector<uint8_t> rhs_ok(static_cast<std::size_t>(rhs_count), 0);
        const bool trace = mlsmr_trace_enabled();
        if (trace) {
            std::cerr << "xhdfe_mlsmr_trace batch_rhs_start rhs_count="
                      << rhs_count << " private_scatter=1" << std::endl;
        }
#ifdef HDFE_USE_OPENMP
        const int batch_threads = std::max(1, std::min(threads, rhs_count));
#pragma omp parallel for schedule(dynamic) num_threads(batch_threads)
#endif
        for (int rhs_idx = 0; rhs_idx < rhs_count; ++rhs_idx) {
            int rhs_it = 0;
            bool rhs_converged = true;
            Eigen::VectorXd beta;
            auto rhs_tls = make_lsmr_At_tls();
            if (rhs_idx == 0) {
                beta = timed_mlsmr_solve(result.y_tilde, "y", rhs_it,
                                         rhs_converged, &rhs_tls);
            } else {
                beta = timed_mlsmr_solve(result.X_tilde.col(rhs_idx - 1),
                                         "x", rhs_it, rhs_converged, &rhs_tls);
            }
            alphas[static_cast<std::size_t>(rhs_idx)] =
                column_scale.array() * beta.array();
            rhs_iters[static_cast<std::size_t>(rhs_idx)] = rhs_it;
            rhs_ok[static_cast<std::size_t>(rhs_idx)] =
                rhs_converged ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        }
        if (trace) {
            std::cerr << "xhdfe_mlsmr_trace batch_rhs_done rhs_count="
                      << rhs_count << std::endl;
        }

        alpha_y = std::move(alphas[0]);
        iters_y = rhs_iters[0];
        ok_y = rhs_ok[0] != 0;
        subtract_projection(result.y_tilde, alpha_y);
        max_iters_used = std::max(max_iters_used, iters_y);
        all_converged = all_converged && ok_y;

        for (int j = 0; j < cols; ++j) {
            const std::size_t rhs_idx = static_cast<std::size_t>(j + 1);
            subtract_projection(result.X_tilde.col(j), alphas[rhs_idx]);
            max_iters_used = std::max(max_iters_used, rhs_iters[rhs_idx]);
            all_converged = all_converged && (rhs_ok[rhs_idx] != 0);
        }
    } else if (options.absorption_method == AbsorptionMethod::Lsmr) {
        const Eigen::VectorXd beta_y = lsmr_solve(result.y_tilde, iters_y, ok_y);
        alpha_y = column_scale.array() * beta_y.array();
    } else if (options.absorption_method == AbsorptionMethod::Mlsmr) {
        const Eigen::VectorXd beta_y =
            timed_mlsmr_solve(result.y_tilde, "y", iters_y, ok_y, nullptr);
        alpha_y = column_scale.array() * beta_y.array();
    } else {
        const Eigen::VectorXd b_y = compute_Zt(result.y_tilde);
        alpha_y = pcg_solve(b_y, iters_y, ok_y);
    }
    if (!use_mlsmr_batch_rhs) {
        subtract_projection(result.y_tilde, alpha_y);
        max_iters_used = std::max(max_iters_used, iters_y);
        all_converged = all_converged && ok_y;

        for (int j = 0; j < cols; ++j) {
            int iters_x = 0;
            bool ok_x = true;
            Eigen::VectorXd alpha_x;
            if (options.absorption_method == AbsorptionMethod::Lsmr) {
                const Eigen::VectorXd beta_x = lsmr_solve(result.X_tilde.col(j), iters_x, ok_x);
                alpha_x = column_scale.array() * beta_x.array();
            } else if (options.absorption_method == AbsorptionMethod::Mlsmr) {
                const Eigen::VectorXd beta_x =
                    timed_mlsmr_solve(result.X_tilde.col(j), "x", iters_x, ok_x,
                                      nullptr);
                alpha_x = column_scale.array() * beta_x.array();
            } else {
                const Eigen::VectorXd b_x = compute_Zt(result.X_tilde.col(j));
                alpha_x = pcg_solve(b_x, iters_x, ok_x);
            }
            subtract_projection(result.X_tilde.col(j), alpha_x);
            max_iters_used = std::max(max_iters_used, iters_x);
            all_converged = all_converged && ok_x;
        }
    }

    result.iterations = max_iters_used;
    result.converged = all_converged;

    if (options.retain_fixed_effects) {
        result.fe_group_ids.resize(indexers.size());
        result.fe_means.resize(indexers.size());
        result.fe_weight_sums.resize(indexers.size());
        for (int d = 0; d < dims; ++d) {
            result.fe_group_ids[static_cast<std::size_t>(d)] = indexers[static_cast<std::size_t>(d)].group_ids;
            result.fe_means[static_cast<std::size_t>(d)] =
                alpha_y.segment(offsets[static_cast<std::size_t>(d)],
                                groups_per_dim[static_cast<std::size_t>(d)]);
            result.fe_weight_sums[static_cast<std::size_t>(d)] =
                compute_weight_sums(indexers[static_cast<std::size_t>(d)],
                                    weight_ptr, unit_weights, n, threads);
        }
    }

    return result;
}

namespace {

struct FeRecoveryIndexer {
    const int* group_ids = nullptr;
    int num_groups = 0;
    int num_levels_present = 0;
};

FeRecoveryResult recover_fixed_effects_impl(const Eigen::VectorXd& partial,
                                            const std::vector<FeRecoveryIndexer>& indexers,
                                            const std::vector<Eigen::VectorXd>* weight_sums_override,
                                            const Eigen::VectorXd* weights,
                                            const HdfeOptions& options) {
    const int n = static_cast<int>(partial.size());
    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();
    const std::size_t dims = indexers.size();

    FeRecoveryResult result;
    if (dims == 0) {
        result.converged = true;
        return result;
    }

    int threads = 1;
    std::vector<int> threads_by_dim;
    std::vector<FeWorkspace> workspaces;
    std::vector<FeProfileStats> fe_profiles;
    std::vector<std::size_t> sweep_order;
    bool use_symmetric = false;
    double fe_tol = options.fe_tolerance > 0.0 ? options.fe_tolerance : options.tol;
    std::vector<GroupCSR> csr;
    std::vector<uint8_t> use_csr(dims, 0);
    std::vector<Eigen::VectorXd> alpha_state;
    Eigen::VectorXd y_state;
    int iter_warmup = 0;
    int iter_two_fe = 0;
    bool enable_two_stage = false;
    std::vector<std::size_t> two_fe_order;
    bool use_accel = false;

    {
        SavefeTimer timer("savefe_recover_setup");
#ifdef HDFE_USE_OPENMP
        if (options.num_threads > 0) {
            threads = options.num_threads;
        } else {
            threads = 1;
        }
        threads = std::max(1, threads);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#endif
        threads = std::max(1, threads);

        threads_by_dim.assign(dims, threads);
#ifdef HDFE_USE_OPENMP
        if (threads > 1) {
            for (std::size_t dim = 0; dim < dims; ++dim) {
                threads_by_dim[dim] =
                    limit_threads_by_tls(threads_by_dim[dim], indexers[dim].num_groups, 0);
            }
        }
#endif

        workspaces.reserve(dims);
        for (const auto& idx : indexers) {
            workspaces.emplace_back(idx.num_groups, 0);
        }
        const bool have_weight_sums =
            (weight_sums_override && weight_sums_override->size() == dims);
        for (std::size_t dim = 0; dim < dims; ++dim) {
            if (have_weight_sums &&
                static_cast<int>((*weight_sums_override)[dim].size()) == indexers[dim].num_groups) {
                workspaces[dim].weight_sums_const = (*weight_sums_override)[dim];
            } else {
                workspaces[dim].weight_sums_const =
                    compute_weight_sums_from_groups(indexers[dim].group_ids, n,
                                                    indexers[dim].num_groups,
                                                    weight_ptr, unit_weights,
                                                    threads_by_dim[dim]);
            }
            workspaces[dim].refresh_weight_sums_inv();
            workspaces[dim].sparse_reset =
                (threads_by_dim[dim] == 1 && indexers[dim].num_groups <= 2000000 &&
                 indexers[dim].num_groups > (n / 4));
        }

        fe_profiles.reserve(dims);
        for (std::size_t dim = 0; dim < dims; ++dim) {
            fe_profiles.push_back(profile_fe_from_weights(indexers[dim].num_groups,
                                                          indexers[dim].num_levels_present,
                                                          workspaces[dim].weight_sums_const,
                                                          unit_weights));
        }

        auto resolve_sweep_order = [&]() -> std::vector<std::size_t> {
            std::vector<std::size_t> order;
            order.reserve(dims);
            if (options.sweep_order_override.size() == dims && dims > 0) {
                std::vector<uint8_t> seen(dims, 0);
                bool valid = true;
                for (const int v : options.sweep_order_override) {
                    if (v < 0 || v >= static_cast<int>(dims)) {
                        valid = false;
                        break;
                    }
                    const std::size_t idx = static_cast<std::size_t>(v);
                    if (seen[idx]) {
                        valid = false;
                        break;
                    }
                    seen[idx] = 1;
                }
                if (valid) {
                    for (const int v : options.sweep_order_override) {
                        order.push_back(static_cast<std::size_t>(v));
                    }
                    return order;
                }
                order.clear();
            }
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
            return order;
        };

        sweep_order = resolve_sweep_order();
        use_symmetric = options.symmetric_sweep && sweep_order.size() > 1;

        const bool allow_csr = (n >= 200000);
        if (allow_csr) {
            csr.resize(dims);
            constexpr int kCsrGroupsThreshold = 200000;
            for (std::size_t dim = 0; dim < dims; ++dim) {
                if (threads_by_dim[dim] > 1 && indexers[dim].num_groups >= kCsrGroupsThreshold) {
                    use_csr[dim] = 1;
                    csr[dim] = build_group_csr(indexers[dim].group_ids, n, indexers[dim].num_groups);
                }
            }
        }

        auto init_alpha = [&]() {
            std::vector<Eigen::VectorXd> alpha;
            alpha.reserve(dims);
            for (const auto& idx : indexers) {
                alpha.emplace_back(Eigen::VectorXd::Zero(idx.num_groups));
            }
            return alpha;
        };
        y_state = partial;
        alpha_state = init_alpha();

        const FeDifficulty difficulty = classify_problem(fe_profiles);
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

        use_accel = dims >= 2 && n >= 200000;
        if (savefe_profile_enabled()) {
            std::ostringstream oss;
            oss << "event=fe_recover_impl_begin n=" << n
                << " dims=" << dims
                << " threads=" << threads
                << " symmetric=" << (use_symmetric ? 1 : 0)
                << " accel=" << (use_accel ? 1 : 0)
                << " two_stage=" << (enable_two_stage ? 1 : 0);
            savefe_profile_log(oss.str());
        }
    }

    auto init_alpha = [&]() {
        std::vector<Eigen::VectorXd> alpha;
        alpha.reserve(dims);
        for (const auto& idx : indexers) {
            alpha.emplace_back(Eigen::VectorXd::Zero(idx.num_groups));
        }
        return alpha;
    };
    auto copy_alpha = [&](std::vector<Eigen::VectorXd>& dst,
                          const std::vector<Eigen::VectorXd>& src) {
        for (std::size_t dim = 0; dim < dims; ++dim) {
            dst[dim] = src[dim];
        }
    };
    auto lincomb_alpha = [&](std::vector<Eigen::VectorXd>& out,
                             const std::vector<Eigen::VectorXd>& a,
                             const std::vector<Eigen::VectorXd>& b,
                             double coef) {
        const double inv = 1.0 - coef;
        for (std::size_t dim = 0; dim < dims; ++dim) {
            out[dim].noalias() = inv * b[dim];
            out[dim].noalias() += coef * a[dim];
        }
    };

    auto run_sweep = [&](Eigen::VectorXd& y_in,
                         const std::vector<std::size_t>& order,
                         std::vector<Eigen::VectorXd>* alpha_state,
                         double* out_max_abs) {
        if (order.empty()) {
            if (out_max_abs) {
                *out_max_abs = 0.0;
            }
            return;
        }
        const bool track_max = (out_max_abs != nullptr);
        double max_abs = 0.0;
        for (std::size_t pos = 0; pos < order.size(); ++pos) {
            const std::size_t dim = order[pos];
            double max_abs_dim = 0.0;
            double* alpha_ptr = nullptr;
            if (alpha_state) {
                alpha_ptr = (*alpha_state)[dim].data();
            }
            if (use_csr[dim]) {
                demean_y_only_csr(y_in,
                                  csr[dim],
                                  weight_ptr,
                                  unit_weights,
                                  workspaces[dim].weight_sums_inv.data(),
                                  true,
                                  1.0,
                                  alpha_ptr,
                                  track_max ? &max_abs_dim : nullptr,
                                  threads_by_dim[dim]);
            } else {
                demean_y_only_groups(y_in,
                                     indexers[dim].group_ids,
                                     workspaces[dim],
                                     weight_ptr,
                                     unit_weights,
                                     threads_by_dim[dim],
                                     true,
                                     1.0,
                                     alpha_ptr,
                                     track_max ? &max_abs_dim : nullptr);
            }
            if (track_max) {
                max_abs = std::max(max_abs, max_abs_dim);
            }
        }
        if (use_symmetric) {
            for (std::size_t idx = order.size(); idx-- > 0;) {
                const std::size_t dim = order[idx];
                double max_abs_dim = 0.0;
                double* alpha_ptr = nullptr;
                if (alpha_state) {
                    alpha_ptr = (*alpha_state)[dim].data();
                }
                if (use_csr[dim]) {
                    demean_y_only_csr(y_in,
                                      csr[dim],
                                      weight_ptr,
                                      unit_weights,
                                      workspaces[dim].weight_sums_inv.data(),
                                      true,
                                      1.0,
                                      alpha_ptr,
                                      track_max ? &max_abs_dim : nullptr,
                                      threads_by_dim[dim]);
                } else {
                    demean_y_only_groups(y_in,
                                         indexers[dim].group_ids,
                                         workspaces[dim],
                                         weight_ptr,
                                         unit_weights,
                                         threads_by_dim[dim],
                                         true,
                                         1.0,
                                         alpha_ptr,
                                         track_max ? &max_abs_dim : nullptr);
                }
                if (track_max) {
                    max_abs = std::max(max_abs, max_abs_dim);
                }
            }
        }
        if (track_max) {
            *out_max_abs = max_abs;
        }
    };

    const int check_interval = 1;
    double last_max_abs = 0.0;

    auto run_simple_phase = [&](const std::vector<std::size_t>& order,
                                int max_iter,
                                int& iter_used) -> bool {
        iter_used = 0;
        if (max_iter <= 0 || order.empty()) {
            return false;
        }
        for (int iter = 0; iter < max_iter; ++iter) {
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            double max_abs = 0.0;
            run_sweep(y_state, order, &alpha_state, do_check ? &max_abs : nullptr);
            if (do_check) {
                last_max_abs = max_abs;
                if (max_abs < fe_tol) {
                    iter_used = iter + 1;
                    return true;
                }
            }
        }
        iter_used = max_iter;
        return false;
    };

    auto run_accel_phase = [&](const std::vector<std::size_t>& order,
                               int max_iter,
                               int& iter_used) -> bool {
        iter_used = 0;
        if (max_iter <= 0 || order.empty()) {
            return false;
        }

        Eigen::VectorXd y_gx = y_state;
        Eigen::VectorXd y_ggx = y_gx;
        std::vector<Eigen::VectorXd> alpha_gx = init_alpha();
        std::vector<Eigen::VectorXd> alpha_ggx = init_alpha();

        copy_alpha(alpha_gx, alpha_state);
        run_sweep(y_gx, order, &alpha_gx, nullptr);

        for (int iter = 0; iter < max_iter; ++iter) {
            y_ggx = y_gx;
            copy_alpha(alpha_ggx, alpha_gx);
            run_sweep(y_ggx, order, &alpha_ggx, nullptr);

            IronsTuckStats stats;
            irons_tuck_accumulate(y_state.data(), y_gx.data(), y_ggx.data(), y_gx.size(), stats);
            if (stats.ssq == 0.0) {
                y_state = y_gx;
                copy_alpha(alpha_state, alpha_gx);
                iter_used = iter + 1;
                return true;
            }

            const double coef = stats.vprod / stats.ssq;
            irons_tuck_update(y_state.data(), y_gx.data(), y_ggx.data(), y_gx.size(), coef);
            lincomb_alpha(alpha_state, alpha_gx, alpha_ggx, coef);

            y_gx = y_state;
            copy_alpha(alpha_gx, alpha_state);
            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            double max_abs = 0.0;
            run_sweep(y_gx, order, &alpha_gx, do_check ? &max_abs : nullptr);
            if (do_check) {
                last_max_abs = max_abs;
                if (max_abs < fe_tol) {
                    y_state = y_gx;
                    copy_alpha(alpha_state, alpha_gx);
                    iter_used = iter + 1;
                    return true;
                }
            }
        }

        y_state = y_gx;
        copy_alpha(alpha_state, alpha_gx);
        iter_used = max_iter;
        return false;
    };

    bool converged = false;
    int total_iters = 0;
    {
        SavefeTimer timer("savefe_recover_loop");
        if (!enable_two_stage && use_accel) {
            converged = run_accel_phase(sweep_order, options.max_iter, total_iters);
        } else if (!enable_two_stage) {
            converged = run_simple_phase(sweep_order, options.max_iter, total_iters);
        } else {
            int remaining = options.max_iter;
            if (use_accel) {
                if (iter_warmup > 0 && remaining > 0) {
                    int phase_iters = 0;
                    converged =
                        run_accel_phase(sweep_order, std::min(iter_warmup, remaining), phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                    int phase_iters = 0;
                    run_accel_phase(two_fe_order, std::min(iter_two_fe, remaining), phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && remaining > 0) {
                    int phase_iters = 0;
                    converged = run_accel_phase(sweep_order, remaining, phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
            } else {
                if (iter_warmup > 0 && remaining > 0) {
                    int phase_iters = 0;
                    converged =
                        run_simple_phase(sweep_order, std::min(iter_warmup, remaining), phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                    int phase_iters = 0;
                    run_simple_phase(two_fe_order, std::min(iter_two_fe, remaining), phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
                if (!converged && remaining > 0) {
                    int phase_iters = 0;
                    converged = run_simple_phase(sweep_order, remaining, phase_iters);
                    total_iters += phase_iters;
                    remaining -= phase_iters;
                }
            }
        }
    }

    result.max_delta = last_max_abs;
    if (converged) {
        result.iterations = total_iters;
        result.converged = true;
    } else {
        result.iterations = options.max_iter;
        result.converged = false;
    }

    if (savefe_profile_enabled()) {
        std::ostringstream oss;
        oss << "event=fe_recover_impl_end iterations=" << result.iterations
            << " converged=" << (result.converged ? 1 : 0)
            << " max_delta=" << result.max_delta;
        savefe_profile_log(oss.str());
    }

    {
        SavefeTimer timer("savefe_recover_expand");
        result.contributions.resize(dims);
        for (std::size_t dim = 0; dim < dims; ++dim) {
            result.contributions[dim].resize(n);
            const int* gid = indexers[dim].group_ids;
            const double* alpha_ptr = alpha_state[dim].data();
            double* out_ptr = result.contributions[dim].data();
            const int dim_threads = threads_by_dim[dim];
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(dim_threads)
#endif
            for (int i = 0; i < n; ++i) {
                out_ptr[i] = alpha_ptr[gid[i]];
            }
        }
    }

    return result;
}

}  // namespace

FeRecoveryResult recover_fixed_effects(const Eigen::VectorXd& partial,
                                       const std::vector<Eigen::VectorXi>& fes,
                                       const Eigen::VectorXd* weights,
                                       const HdfeOptions& options) {
    const int n = static_cast<int>(partial.size());
    if (n == 0) {
        throw std::runtime_error("Partial residual vector must be non-empty");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as the outcome");
    }

    if (fes.empty()) {
        FeRecoveryResult empty;
        empty.converged = true;
        return empty;
    }

    std::vector<FeIndexer> indexers;
    indexers.reserve(fes.size());
    for (const auto& raw : fes) {
        if (raw.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        indexers.push_back(build_indexer(raw));
    }

    std::vector<FeRecoveryIndexer> views;
    views.reserve(indexers.size());
    for (const auto& idx : indexers) {
        FeRecoveryIndexer view;
        view.group_ids = idx.group_ids.data();
        view.num_groups = idx.num_groups;
        view.num_levels_present = idx.num_levels_present;
        views.push_back(view);
    }

    return recover_fixed_effects_impl(partial, views, nullptr, weights, options);
}

FeRecoveryResult recover_fixed_effects_group_ids(const Eigen::VectorXd& partial,
                                                 const std::vector<std::vector<int>>& fe_group_ids,
                                                 const std::vector<int>& fe_levels,
                                                 const Eigen::VectorXd* weights,
                                                 const HdfeOptions& options,
                                                 const std::vector<Eigen::VectorXd>* weight_sums_override) {
    const int n = static_cast<int>(partial.size());
    if (n == 0) {
        throw std::runtime_error("Partial residual vector must be non-empty");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as the outcome");
    }
    if (fe_group_ids.empty()) {
        FeRecoveryResult empty;
        empty.converged = true;
        return empty;
    }

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();

    const std::size_t dims = fe_group_ids.size();
    std::vector<FeRecoveryIndexer> views;
    views.reserve(dims);
    bool override_ok = (weight_sums_override && weight_sums_override->size() == dims);
    for (std::size_t dim = 0; dim < dims; ++dim) {
        const auto& ids = fe_group_ids[dim];
        if (static_cast<int>(ids.size()) != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        // Vectorized min/max scan via Eigen (SIMD).
        Eigen::Map<const Eigen::VectorXi> ids_view(
            ids.data(), static_cast<Eigen::Index>(ids.size()));
        const int min_id = ids_view.minCoeff();
        const int max_id = ids_view.maxCoeff();
        if (min_id < 0) {
            throw std::runtime_error("Fixed-effect group ids must be non-negative");
        }
        const int derived_groups = max_id + 1;
        const int expected = dim < fe_levels.size() ? fe_levels[dim] : -1;
        const int num_groups =
            (expected > 0 && derived_groups <= expected) ? expected : derived_groups;
        if (override_ok &&
            static_cast<int>((*weight_sums_override)[dim].size()) != num_groups) {
            override_ok = false;
        }
        FeRecoveryIndexer view;
        view.group_ids = ids.data();
        view.num_groups = num_groups;
        view.num_levels_present = num_groups;
        views.push_back(view);
    }

    const GpuBackend gpu_backend = resolve_gpu_backend();
    const bool gpu_cuda = (gpu_backend == GpuBackend::Cuda && cuda_backend_available());
    if (gpu_backend_requested(gpu_backend) && !gpu_cuda) {
        throw std::runtime_error("Requested GPU backend is not available for fixed-effect recovery");
    }
    if (gpu_cuda) {
        std::vector<Eigen::VectorXd> weight_sums_local;
        weight_sums_local.reserve(dims);
        std::vector<const Eigen::VectorXd*> weight_sums_ptrs;
        weight_sums_ptrs.reserve(dims);
        std::vector<GpuFeInput> fe_inputs;
        fe_inputs.reserve(dims);

        auto resolve_sweep_order = [&]() -> std::vector<std::size_t> {
            std::vector<std::size_t> order;
            order.reserve(dims);
            if (options.sweep_order_override.size() == dims && dims > 0) {
                std::vector<uint8_t> seen(dims, 0);
                bool valid = true;
                for (const int v : options.sweep_order_override) {
                    if (v < 0 || v >= static_cast<int>(dims)) {
                        valid = false;
                        break;
                    }
                    const std::size_t idx = static_cast<std::size_t>(v);
                    if (seen[idx]) {
                        valid = false;
                        break;
                    }
                    seen[idx] = 1;
                }
                if (valid) {
                    for (const int v : options.sweep_order_override) {
                        order.push_back(static_cast<std::size_t>(v));
                    }
                    return order;
                }
                order.clear();
            }
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
            return order;
        };

        for (std::size_t dim = 0; dim < dims; ++dim) {
            const int num_groups = views[dim].num_groups;
            if (override_ok) {
                weight_sums_ptrs.push_back(&(*weight_sums_override)[dim]);
            } else {
                const auto& ids = fe_group_ids[dim];
                weight_sums_local.push_back(
                    compute_weight_sums_from_groups(ids.data(), n, num_groups,
                                                    weight_ptr, unit_weights, 1));
                weight_sums_ptrs.push_back(&weight_sums_local.back());
            }
            GpuFeInput input;
            input.group_ids = fe_group_ids[dim].data();
            input.num_groups = num_groups;
            input.num_levels_present = num_groups;
            input.weight_sums = weight_sums_ptrs[dim]->data();
            fe_inputs.push_back(input);
        }

        HdfeOptions gpu_opts = options;
        gpu_opts.retain_fixed_effects = true;
        gpu_opts.fe_recovery_method = FeRecoveryMethod::Hybrid;
        const AbsorptionMethod method =
            (gpu_opts.symmetric_sweep && dims > 1)
                ? AbsorptionMethod::SymmetricGaussSeidel
                : AbsorptionMethod::GaussSeidel;

        AbsorptionResult gpu_abs;
        Eigen::MatrixXd empty_X(n, 0);
        const std::vector<std::size_t> sweep_order = resolve_sweep_order();
        const bool ok = absorb_fixed_effects_cuda(partial, empty_X, fe_inputs, weights,
                                                  sweep_order, gpu_opts, method, gpu_abs);
        if (ok && gpu_abs.converged && gpu_abs.fe_alpha_y.size() == dims) {
            FeRecoveryResult gpu_result;
            gpu_result.contributions.resize(dims);

            int threads = 1;
#ifdef HDFE_USE_OPENMP
            if (options.num_threads > 0) {
                threads = options.num_threads;
            }
            threads = std::max(1, threads);
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
#endif
            threads = std::max(1, threads);

            bool sizes_ok = true;
            for (std::size_t dim = 0; dim < dims; ++dim) {
                if (gpu_abs.fe_alpha_y[dim].size() != weight_sums_ptrs[dim]->size()) {
                    sizes_ok = false;
                    break;
                }
                gpu_result.contributions[dim].resize(n);
                const int* gid = fe_group_ids[dim].data();
                const double* alpha_ptr = gpu_abs.fe_alpha_y[dim].data();
                double* out_ptr = gpu_result.contributions[dim].data();
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
                for (int i = 0; i < n; ++i) {
                    out_ptr[i] = alpha_ptr[gid[i]];
                }
            }

            if (sizes_ok) {
                const double fe_tol =
                    options.fe_tolerance > 0.0 ? options.fe_tolerance : options.tol;
                gpu_result.iterations = gpu_abs.iterations;
                gpu_result.max_delta = 0.0;
                gpu_result.converged = true;
                if (fe_tol > 0.0) {
                    Eigen::VectorXd residual = partial;
                    for (const auto& fe_vec : gpu_result.contributions) {
                        residual.noalias() -= fe_vec;
                    }
                    double max_delta = 0.0;
                    bool gpu_check_ok = fe_recovery_max_delta_cuda_cached(
                        residual, fe_inputs, weights, sweep_order, max_delta);
                    if (!gpu_check_ok) {
                        gpu_check_ok = fe_recovery_max_delta_cuda(
                            residual, fe_inputs, weights, sweep_order, max_delta);
                    }
                    if (!gpu_check_ok) {
                        // GPU path failed: fall back to the CPU implementation
                        // on a copy of the residual so state stays consistent.
                        Eigen::VectorXd residual_check = residual;
                        max_delta = fe_recovery_max_delta(
                            residual_check, fe_group_ids, fe_levels, weights,
                            options,
                            override_ok ? weight_sums_override : nullptr);
                    }
                    gpu_result.max_delta = max_delta;
                    gpu_result.converged = (max_delta <= fe_tol);
                }
                if (gpu_result.converged) {
                    return gpu_result;
                }
            }
        }
        if (gpu_backend_requested(gpu_backend)) {
            throw std::runtime_error("Requested GPU backend did not converge during fixed-effect recovery");
        }
    }
    return recover_fixed_effects_impl(partial, views,
                                      override_ok ? weight_sums_override : nullptr,
                                      weights, options);
}

double fe_recovery_max_delta(Eigen::VectorXd& residual,
                             const std::vector<std::vector<int>>& fe_group_ids,
                             const std::vector<int>& fe_levels,
                             const Eigen::VectorXd* weights,
                             const HdfeOptions& options,
                             const std::vector<Eigen::VectorXd>* weight_sums_override) {
    const int n = static_cast<int>(residual.size());
    if (n == 0 || fe_group_ids.empty()) {
        return 0.0;
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as the outcome");
    }

    const std::size_t dims = fe_group_ids.size();
    std::vector<FeRecoveryIndexer> views;
    views.reserve(dims);
    bool override_ok = (weight_sums_override && weight_sums_override->size() == dims);
    for (std::size_t dim = 0; dim < dims; ++dim) {
        const auto& ids = fe_group_ids[dim];
        if (static_cast<int>(ids.size()) != n) {
            throw std::runtime_error("Each fixed-effect vector must match the length of y");
        }
        // Vectorized min/max scan via Eigen (SIMD).
        Eigen::Map<const Eigen::VectorXi> ids_view(
            ids.data(), static_cast<Eigen::Index>(ids.size()));
        const int min_id = ids_view.minCoeff();
        const int max_id = ids_view.maxCoeff();
        if (min_id < 0) {
            throw std::runtime_error("Fixed-effect group ids must be non-negative");
        }
        const int derived_groups = max_id + 1;
        const int expected = dim < fe_levels.size() ? fe_levels[dim] : -1;
        const int num_groups =
            (expected > 0 && derived_groups <= expected) ? expected : derived_groups;
        if (override_ok &&
            static_cast<int>((*weight_sums_override)[dim].size()) != num_groups) {
            override_ok = false;
        }
        FeRecoveryIndexer view;
        view.group_ids = ids.data();
        view.num_groups = num_groups;
        view.num_levels_present = num_groups;
        views.push_back(view);
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (options.num_threads > 0) {
        threads = options.num_threads;
    } else {
        threads = 1;
    }
    threads = std::max(1, threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#endif
    threads = std::max(1, threads);

    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();

    auto resolve_sweep_order = [&]() -> std::vector<std::size_t> {
        std::vector<std::size_t> order;
        order.reserve(dims);
        if (options.sweep_order_override.size() == dims && dims > 0) {
            std::vector<uint8_t> seen(dims, 0);
            bool valid = true;
            for (const int v : options.sweep_order_override) {
                if (v < 0 || v >= static_cast<int>(dims)) {
                    valid = false;
                    break;
                }
                const std::size_t idx = static_cast<std::size_t>(v);
                if (seen[idx]) {
                    valid = false;
                    break;
                }
                seen[idx] = 1;
            }
            if (valid) {
                for (const int v : options.sweep_order_override) {
                    order.push_back(static_cast<std::size_t>(v));
                }
                return order;
            }
            order.clear();
        }
        order.resize(dims);
        std::iota(order.begin(), order.end(), 0);
        return order;
    };
    const std::vector<std::size_t> sweep_order = resolve_sweep_order();
    const bool use_symmetric = options.symmetric_sweep && sweep_order.size() > 1;

    std::vector<int> threads_by_dim(dims, threads);
#ifdef HDFE_USE_OPENMP
    if (threads > 1) {
        for (std::size_t dim = 0; dim < dims; ++dim) {
            threads_by_dim[dim] =
                limit_threads_by_tls(threads_by_dim[dim], views[dim].num_groups, 0);
        }
    }
#endif

    std::vector<FeWorkspace> workspaces;
    workspaces.reserve(dims);
    for (const auto& idx : views) {
        workspaces.emplace_back(idx.num_groups, 0);
    }
    const bool have_weight_sums =
        (override_ok && weight_sums_override && weight_sums_override->size() == dims);
    for (std::size_t dim = 0; dim < dims; ++dim) {
        if (have_weight_sums &&
            static_cast<int>((*weight_sums_override)[dim].size()) == views[dim].num_groups) {
            workspaces[dim].weight_sums_const = (*weight_sums_override)[dim];
        } else {
            workspaces[dim].weight_sums_const =
                compute_weight_sums_from_groups(views[dim].group_ids, n,
                                                views[dim].num_groups,
                                                weight_ptr, unit_weights,
                                                threads_by_dim[dim]);
        }
        workspaces[dim].refresh_weight_sums_inv();
        workspaces[dim].sparse_reset =
            (threads_by_dim[dim] == 1 && views[dim].num_groups <= 2000000 &&
             views[dim].num_groups > (n / 4));
    }

    double max_abs = 0.0;
    const bool allow_csr = (n >= 200000);
    constexpr int kCsrGroupsThreshold = 200000;
    std::vector<uint8_t> use_csr(dims, 0);
    std::vector<GroupCSR> csr;
    if (allow_csr) {
        csr.resize(dims);
        for (std::size_t dim = 0; dim < dims; ++dim) {
            if (threads_by_dim[dim] > 1 &&
                views[dim].num_groups >= kCsrGroupsThreshold) {
                use_csr[dim] = 1;
                csr[dim] = build_group_csr(views[dim].group_ids, n, views[dim].num_groups);
            }
        }
    }
    for (std::size_t pos = 0; pos < sweep_order.size(); ++pos) {
        const std::size_t dim = sweep_order[pos];
        double max_abs_dim = 0.0;
        if (use_csr[dim]) {
            demean_y_only_csr(residual,
                              csr[dim],
                              weight_ptr,
                              unit_weights,
                              workspaces[dim].weight_sums_inv.data(),
                              true,
                              1.0,
                              nullptr,
                              &max_abs_dim,
                              threads_by_dim[dim]);
        } else {
            demean_y_only_groups(residual,
                                 views[dim].group_ids,
                                 workspaces[dim],
                                 weight_ptr,
                                 unit_weights,
                                 threads_by_dim[dim],
                                 true,
                                 1.0,
                                 nullptr,
                                 &max_abs_dim);
        }
        max_abs = std::max(max_abs, max_abs_dim);
    }
    if (use_symmetric) {
        for (std::size_t pos = sweep_order.size(); pos-- > 0;) {
            const std::size_t dim = sweep_order[pos];
            double max_abs_dim = 0.0;
            if (use_csr[dim]) {
                demean_y_only_csr(residual,
                                  csr[dim],
                                  weight_ptr,
                                  unit_weights,
                                  workspaces[dim].weight_sums_inv.data(),
                                  true,
                                  1.0,
                                  nullptr,
                                  &max_abs_dim,
                                  threads_by_dim[dim]);
            } else {
                demean_y_only_groups(residual,
                                     views[dim].group_ids,
                                     workspaces[dim],
                                     weight_ptr,
                                     unit_weights,
                                     threads_by_dim[dim],
                                     true,
                                     1.0,
                                     nullptr,
                                     &max_abs_dim);
            }
            max_abs = std::max(max_abs, max_abs_dim);
        }
    }
    return max_abs;
}

}  // namespace detail
}  // namespace hdfe
