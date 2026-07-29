#include "fe_absorption_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/parallel_work_observer.hpp"

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

#ifdef HDFE_USE_CUDA

namespace hdfe {
namespace detail {
namespace {

// Keep the reduction graph and scratch cap identical to the public
// certificate in fe_absorption.cpp.  These helpers are private to this
// translation unit so changing the CUDA certificate cannot alter CPU solver
// inlining or outlining decisions.
constexpr std::size_t kDeterministicScatterScratchBytes =
    256ULL * 1024ULL * 1024ULL;

inline bool finite_double_bits(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "xhdfe requires 64-bit IEEE-754 doubles");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

template <typename Index>
inline int deterministic_chunk_count(Index n) {
    return deterministic_parallel_chunk_count(n);
}

inline int deterministic_scatter_chunk_count(int n,
                                             int groups,
                                             int vectors_per_chunk = 1) {
    if (n <= 0 || groups <= 0 || vectors_per_chunk <= 0) {
        return 0;
    }
    const std::size_t bytes_per_chunk =
        static_cast<std::size_t>(groups) *
        static_cast<std::size_t>(vectors_per_chunk) * sizeof(double);
    const std::size_t target_chunks =
        static_cast<std::size_t>(deterministic_chunk_target());
    const std::size_t scratch_limited =
        bytes_per_chunk == 0
            ? target_chunks
            : std::max<std::size_t>(
                  1U, kDeterministicScatterScratchBytes / bytes_per_chunk);
    const int max_chunks = static_cast<int>(std::min<std::size_t>(
        target_chunks, scratch_limited));
    return std::max(1, std::min(n, max_chunks));
}

inline ParallelWorkObserver* observer_if_needed(
    ParallelWorkObserver* observer,
    int requested_team,
    int useful_units) {
    if (!observer) {
        return nullptr;
    }
    const int team = std::max(1, requested_team);
    const int active = std::max(1, std::min(team, useful_units));
    if (observer->max_team_size() >= team &&
        observer->active_workers() >= active) {
        return nullptr;
    }
    return observer;
}

template <typename Index>
inline Index deterministic_chunk_begin(Index n, int chunk, int chunks) {
    return deterministic_parallel_chunk_begin(n, chunk, chunks);
}

template <typename Index>
inline Index deterministic_chunk_end(Index n, int chunk, int chunks) {
    return deterministic_parallel_chunk_end(n, chunk, chunks);
}

template <typename Index, typename Body>
double deterministic_chunked_sum(
    Index n,
    int threads,
    const Body& body,
    ParallelWorkObserver* observer = nullptr) {
    const int chunks = deterministic_chunk_count(n);
    if (chunks == 0) {
        return 0.0;
    }
    InlineOrDynamicBuffer<double> partial(
        static_cast<std::size_t>(chunks));
    int local_threads = 1;
#ifdef HDFE_USE_OPENMP
    local_threads =
        std::max(1, threads > 0 ? threads : omp_get_max_threads());
#else
    (void)threads;
#endif
    ParallelWorkObserver* region_observer =
        observer_if_needed(observer, local_threads, chunks);
    if (region_observer) {
        region_observer->begin_region(local_threads);
    }
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(local_threads)
#endif
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (region_observer) {
            region_observer->observe_work();
        }
        const Index begin = deterministic_chunk_begin(n, chunk, chunks);
        const Index end = deterministic_chunk_end(n, chunk, chunks);
        double local = 0.0;
        for (Index i = begin; i < end; ++i) {
            local += body(i);
        }
        partial[static_cast<std::size_t>(chunk)] = local;
    }
    if (region_observer) {
        region_observer->end_region();
    }
    double total = 0.0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        total += partial[static_cast<std::size_t>(chunk)];
    }
    return total;
}

double deterministic_sumsq_raw(
    const double* data,
    Eigen::Index n,
    int threads,
    ParallelWorkObserver* observer) {
    return deterministic_chunked_sum<Eigen::Index>(
        n, threads, [data](Eigen::Index i) {
            const double value = data[i];
            return value * value;
        }, observer);
}

double effective_absorption_tolerance(const HdfeOptions& options) {
    if (options.tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(options.tol, 1.0e-9);
    }
    return options.tol;
}

std::vector<const HeterogeneousSlopeTerm*> build_slope_lookup(
    const std::vector<HeterogeneousSlopeTerm>& slopes,
    std::size_t dims,
    int n) {
    std::vector<const HeterogeneousSlopeTerm*> lookup(dims, nullptr);
    for (const auto& slope : slopes) {
        if (slope.fe_index < 0 ||
            slope.fe_index >= static_cast<int>(dims)) {
            throw std::runtime_error(
                "Heterogeneous slope FE index out of range");
        }
        if (slope.values.size() != n) {
            throw std::runtime_error(
                "Heterogeneous slope variable must match the length of y");
        }
        auto& slot = lookup[static_cast<std::size_t>(slope.fe_index)];
        if (slot != nullptr) {
            throw std::runtime_error(
                "Only one heterogeneous slope variable per absorbed term is supported");
        }
        slot = &slope;
    }
    return lookup;
}

struct CertificateAccumulationPlan {
    enum class Kind {
        OwnerByGroup,
        OrderedPartials,
        DenseChunks,
    };

    Kind kind = Kind::DenseChunks;
    int logical_chunk_count = 1;
    bool groups_contiguous = false;
    std::size_t scratch_bytes = 0;
};

// WP1 telemetry gate. Same channel and env variable as the solver's
// cpu_profile lines (XHDFE_PROFILE_CPU) so the phase balance sees one stream;
// self-contained here because those helpers have internal linkage in their
// own translation units.
inline bool certificate_profile_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("XHDFE_PROFILE_CPU");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return enabled;
}

inline std::size_t certificate_dense_bytes(int groups,
                                           int values_per_group) {
    if (groups <= 0 || values_per_group <= 0) {
        return 0;
    }
    const std::size_t group_count = static_cast<std::size_t>(groups);
    const std::size_t value_count =
        static_cast<std::size_t>(values_per_group);
    if (group_count >
        std::numeric_limits<std::size_t>::max() / value_count /
            sizeof(double)) {
        throw std::overflow_error(
            "CUDA certificate accumulation scratch size overflow");
    }
    return group_count * value_count * sizeof(double);
}

inline CertificateAccumulationPlan choose_certificate_plan(
    int n,
    int groups,
    int values_per_group,
    int moment_count,
    bool groups_contiguous,
    bool unit_weights,
    bool has_slope) {
    (void)unit_weights;
    (void)has_slope;
    CertificateAccumulationPlan plan;
    plan.logical_chunk_count =
        deterministic_scatter_chunk_count(n, groups, values_per_group);
    plan.groups_contiguous = groups_contiguous;
    plan.kind = CertificateAccumulationPlan::Kind::OwnerByGroup;
    // WP1 (plan C7): report the bytes this plan actually allocates instead of
    // the previous hard-coded 0, which made the scratch-budget check vacuous
    // and the telemetry false. Per dimension the certificate allocates:
    //   sums               groups x values_per_group doubles (dominant)
    //   stable order       obs_index (n ints) + group_ptr and its cursor
    //                      copy (2 x (groups+1) ints), only when the input
    //                      groups are not contiguous
    //   operator_partials  chunk_count x moment_count doubles
    plan.scratch_bytes = certificate_dense_bytes(groups, values_per_group);
    if (!groups_contiguous) {
        plan.scratch_bytes +=
            (static_cast<std::size_t>(n) +
             2U * (static_cast<std::size_t>(groups) + 1U)) *
            sizeof(int);
    }
    plan.scratch_bytes +=
        static_cast<std::size_t>(plan.logical_chunk_count) *
        static_cast<std::size_t>(moment_count) * sizeof(double);
    return plan;
}

inline bool input_has_contiguous_groups(
    const GpuFeCertificateView& input,
    int n) {
    return input.groups_contiguous &&
           input.group_ids != nullptr &&
           input.group_ptr != nullptr &&
           input.num_groups > 0 &&
           input.group_ids_size == static_cast<std::size_t>(n) &&
           input.group_ptr_size ==
               static_cast<std::size_t>(input.num_groups) + 1U &&
           input.group_ptr[0] == 0 &&
           input.group_ptr[input.num_groups] == n;
}

struct StableCertificateGroupOrder {
    std::vector<int> group_ptr;
    std::vector<int> obs_index;

    void build(const int* group_ids, int n, int num_groups) {
        group_ptr.assign(
            static_cast<std::size_t>(num_groups) + 1U, 0);
        for (int row = 0; row < n; ++row) {
            const int group = group_ids[row];
            if (group < 0 || group >= num_groups) {
                throw std::runtime_error(
                    "CUDA certificate group ids must be in range");
            }
            ++group_ptr[static_cast<std::size_t>(group) + 1U];
        }
        for (int group = 0; group < num_groups; ++group) {
            group_ptr[static_cast<std::size_t>(group) + 1U] +=
                group_ptr[static_cast<std::size_t>(group)];
        }
        obs_index.resize(static_cast<std::size_t>(n));
        std::vector<int> cursor = group_ptr;
        for (int row = 0; row < n; ++row) {
            const int group = group_ids[row];
            const int position =
                cursor[static_cast<std::size_t>(group)]++;
            obs_index[static_cast<std::size_t>(position)] = row;
        }
    }
};

}  // namespace

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void certify_cuda_absorption_result(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    const std::vector<HeterogeneousSlopeTerm>& slopes,
    const std::vector<GpuFeCertificateView>& fe_views,
    AbsorptionResult& result) {
    result.abs_residual = 0.0;
    result.abs_residual_rel = 0.0;
    result.precision_certified = result.converged;

    const int n = static_cast<int>(y.size());
    if (!result.gpu_used || X.rows() != n ||
        result.y_tilde.size() != n ||
        result.X_tilde.rows() != n ||
        result.X_tilde.cols() != X.cols() ||
        (weights && weights->size() != n)) {
        throw std::runtime_error(
            "Cannot certify CUDA absorption: invalid result dimensions or backend state");
    }
    if (fes.empty()) {
        return;
    }
    if (fe_views.size() != fes.size()) {
        throw std::runtime_error(
            "Cannot certify CUDA absorption: FE/input count differs");
    }
    for (std::size_t dim = 0; dim < fes.size(); ++dim) {
        if (fes[dim].size() != n ||
            fe_views[dim].group_ids == nullptr ||
            fe_views[dim].group_ids_size !=
                static_cast<std::size_t>(n) ||
            fe_views[dim].num_groups <= 0) {
            throw std::runtime_error(
                "Cannot certify CUDA absorption: invalid FE/input dimensions");
        }
    }

    const std::vector<const HeterogeneousSlopeTerm*> slope_lookup =
        build_slope_lookup(slopes, fe_views.size(), n);
    int threads = 1;
#ifdef HDFE_USE_OPENMP
    threads = std::max(1, options.num_threads > 0 ? options.num_threads : 1);
#endif
    const bool unit_weights = (weights == nullptr);
    const double* weight_ptr = unit_weights ? nullptr : weights->data();
    const int rhs_count = static_cast<int>(X.cols()) + 1;
    std::vector<long double> original_norm_sq(
        static_cast<std::size_t>(rhs_count), 0.0L);
    std::vector<long double> residual_norm_sq(
        static_cast<std::size_t>(rhs_count), 0.0L);
    original_norm_sq[0] = static_cast<long double>(
        deterministic_sumsq_raw(
            y.data(), n, threads, options.parallel_observer));
    for (int rhs = 1; rhs < rhs_count; ++rhs) {
        original_norm_sq[static_cast<std::size_t>(rhs)] =
            static_cast<long double>(deterministic_sumsq_raw(
                X.col(rhs - 1).data(), n, threads,
                options.parallel_observer));
    }
    long double operator_norm_sq = 0.0L;

    using CertificateMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // WP1_STANDARD_CERTIFICATE_BEGIN
    for (std::size_t dim = 0; dim < fe_views.size(); ++dim) {
        const GpuFeCertificateView& input = fe_views[dim];
        const HeterogeneousSlopeTerm* slope = slope_lookup[dim];
        const int moment_count =
            slope ? (slope->include_intercept ? 2 : 1) : 1;
        const int values_per_group = rhs_count * moment_count;
        const bool groups_contiguous =
            input_has_contiguous_groups(input, n);
        const CertificateAccumulationPlan plan = choose_certificate_plan(
            n, input.num_groups, values_per_group, moment_count,
            groups_contiguous, unit_weights, slope != nullptr);
        // The budget is the dense accumulation cost plus the bounded index
        // and partial overheads — the same quantities the plan now reports
        // truthfully. Until WP3 introduces an independent device-side budget
        // this check can only catch planner bugs (negative chunk counts,
        // overflow in the byte accounting), not policy violations; the
        // previous form compared a hard-coded 0 against the budget and could
        // not fail at all.
        const std::size_t scratch_budget =
            certificate_dense_bytes(input.num_groups, values_per_group) +
            (static_cast<std::size_t>(n) +
             2U * (static_cast<std::size_t>(input.num_groups) + 1U)) *
                sizeof(int) +
            static_cast<std::size_t>(plan.logical_chunk_count) *
                static_cast<std::size_t>(moment_count) * sizeof(double);
        const bool scratch_budget_ok =
            plan.scratch_bytes <= scratch_budget;
        if (!scratch_budget_ok || plan.logical_chunk_count <= 0) {
            throw std::runtime_error(
                "CUDA certificate accumulation plan exceeds its scratch budget");
        }
        if (certificate_profile_enabled()) {
            // WP1 telemetry: actual scratch and observation passes for this
            // dimension. passes = rows x (operator moments + sums reads per
            // row) + the two stable-order passes when an inverse order must
            // be built. Emitted on the cpu_profile channel so the WP1 phase
            // balance can consume it alongside the existing labels.
            const long long passes =
                static_cast<long long>(n) *
                    (static_cast<long long>(moment_count) +
                     static_cast<long long>(values_per_group)) +
                (groups_contiguous ? 0LL : 2LL * static_cast<long long>(n));
            std::fprintf(stderr,
                         "cpu_profile label=cuda_cert_scratch dim=%zu "
                         "bytes=%zu passes=%lld chunks=%d contiguous=%d\n",
                         dim, plan.scratch_bytes, passes,
                         plan.logical_chunk_count,
                         plan.groups_contiguous ? 1 : 0);
        }
        const int chunk_count = plan.logical_chunk_count;
        const int* group_ids = input.group_ids;
        const double* slope_values =
            slope ? slope->values.data() : nullptr;

        // Preserve the legacy operator-norm loop and combination order
        // literally.  It is small; changing its loop shape under fast-math
        // can move the final relative residual by one ULP.
        std::vector<std::vector<double>> operator_partials(
            static_cast<std::size_t>(chunk_count),
            std::vector<double>(
                static_cast<std::size_t>(moment_count), 0.0));
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(std::min(threads, chunk_count))
#endif
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            std::vector<double>& local_operator =
                operator_partials[static_cast<std::size_t>(chunk)];
            const int begin =
                deterministic_chunk_begin(n, chunk, chunk_count);
            const int end =
                deterministic_chunk_end(n, chunk, chunk_count);
            for (int row = begin; row < end; ++row) {
                const double weight =
                    unit_weights ? 1.0 : weight_ptr[row];
                for (int moment = 0; moment < moment_count; ++moment) {
                    double multiplier = weight;
                    if (slope) {
                        const bool intercept_moment =
                            slope->include_intercept && moment == 0;
                        if (!intercept_moment) {
                            multiplier *= slope_values[row];
                        }
                    }
                    local_operator[static_cast<std::size_t>(moment)] +=
                        multiplier * multiplier;
                }
            }
        }
        for (int moment = 0; moment < moment_count; ++moment) {
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                operator_norm_sq += static_cast<long double>(
                    operator_partials[static_cast<std::size_t>(chunk)]
                                     [static_cast<std::size_t>(moment)]);
            }
        }

        StableCertificateGroupOrder stable_order;
        const int* group_ptr = input.group_ptr;
        const int* obs_index = nullptr;
        if (!groups_contiguous) {
            stable_order.build(group_ids, n, input.num_groups);
            group_ptr = stable_order.group_ptr.data();
            obs_index = stable_order.obs_index.data();
        }

        CertificateMatrix sums(input.num_groups, values_per_group);
        // §10.8 minimal correction, applied because the regression was
        // measured before WP6 replaces this path: the previous task layout
        // (group × value_index) walked every group's rows once PER VALUE —
        // values_per_group full passes over the observations per dimension,
        // which the ab12 §12 cells measured as the dominant candidate-vs-
        // public slowdown (ready +1.1 s, simulated_panel +4.6 s). Hoisting
        // the (moment, rhs) loop inside the per-group task makes it ONE pass.
        // Bit-preservation: for each (group, value_index) the additions into
        // its accumulator happen for the same rows, in the same row order,
        // with the same chunk-sequential merge (chunk==0 assigns, later
        // chunks add) — only the interleaving BETWEEN independent
        // accumulators changes, which no sum observes.
        const std::int64_t group_task_count =
            static_cast<std::int64_t>(input.num_groups);
        const int certificate_threads = std::max(
            1, static_cast<int>(
                   std::min<std::int64_t>(threads, group_task_count)));
        ParallelWorkObserver* region_observer = observer_if_needed(
            options.parallel_observer, certificate_threads,
            static_cast<int>(std::min<std::int64_t>(
                group_task_count, std::numeric_limits<int>::max())));
        if (region_observer) {
            region_observer->begin_region(certificate_threads);
        }
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(certificate_threads)
        {
            if (region_observer) {
                region_observer->observe_work();
            }
            std::vector<double> task_totals(
                static_cast<std::size_t>(values_per_group));
            std::vector<double> task_locals(
                static_cast<std::size_t>(values_per_group));
#pragma omp for schedule(static)
            for (std::int64_t task = 0; task < group_task_count; ++task) {
#else
        if (region_observer) {
            region_observer->observe_work();
        }
        {
            std::vector<double> task_totals(
                static_cast<std::size_t>(values_per_group));
            std::vector<double> task_locals(
                static_cast<std::size_t>(values_per_group));
        for (std::int64_t task = 0; task < group_task_count; ++task) {
#endif
                const int group = static_cast<int>(task);
                int position = group_ptr[group];
                const int group_end = group_ptr[group + 1];
                for (int chunk = 0; chunk < chunk_count; ++chunk) {
                    const int chunk_end =
                        deterministic_chunk_end(n, chunk, chunk_count);
                    for (int vi = 0; vi < values_per_group; ++vi) {
                        task_locals[static_cast<std::size_t>(vi)] = 0.0;
                    }
                    while (position < group_end) {
                        const int row =
                            obs_index ? obs_index[position] : position;
                        if (row >= chunk_end) {
                            break;
                        }
                        const double base_weight =
                            unit_weights ? 1.0 : weight_ptr[row];
                        for (int moment = 0; moment < moment_count; ++moment) {
                            double multiplier = base_weight;
                            if (slope) {
                                const bool intercept_moment =
                                    slope->include_intercept && moment == 0;
                                if (!intercept_moment) {
                                    multiplier *= slope_values[row];
                                }
                            }
                            const int base_vi = moment * rhs_count;
                            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                                const double value =
                                    rhs == 0
                                        ? result.y_tilde[row]
                                        : result.X_tilde(row, rhs - 1);
                                task_locals[static_cast<std::size_t>(
                                    base_vi + rhs)] += multiplier * value;
                            }
                        }
                        ++position;
                    }
                    if (chunk == 0) {
                        for (int vi = 0; vi < values_per_group; ++vi) {
                            task_totals[static_cast<std::size_t>(vi)] =
                                task_locals[static_cast<std::size_t>(vi)];
                        }
                    } else {
                        for (int vi = 0; vi < values_per_group; ++vi) {
                            task_totals[static_cast<std::size_t>(vi)] +=
                                task_locals[static_cast<std::size_t>(vi)];
                        }
                    }
                }
                for (int vi = 0; vi < values_per_group; ++vi) {
                    sums(group, vi) =
                        task_totals[static_cast<std::size_t>(vi)];
                }
            }
#ifdef HDFE_USE_OPENMP
        }
#else
        }
#endif
        if (region_observer) {
            region_observer->end_region();
        }
        for (int moment = 0; moment < moment_count; ++moment) {
            const int base = moment * rhs_count;
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                const double residual_sq = deterministic_chunked_sum<int>(
                    input.num_groups, threads, [&](int group) {
                        const double value = sums(group, base + rhs);
                        return value * value;
                    }, options.parallel_observer);
                residual_norm_sq[static_cast<std::size_t>(rhs)] +=
                    static_cast<long double>(residual_sq);
            }
        }
    }
    // WP1_STANDARD_CERTIFICATE_END

    double max_absolute = 0.0;
    double max_relative = 0.0;
    auto fail_nonfinite_certificate = [&]() {
        result.abs_residual = std::numeric_limits<double>::max();
        result.abs_residual_rel = std::numeric_limits<double>::max();
        result.precision_certified = false;
    };
    const double operator_norm_sq_double =
        static_cast<double>(operator_norm_sq);
    if (!finite_double_bits(operator_norm_sq_double) ||
        operator_norm_sq_double < 0.0) {
        fail_nonfinite_certificate();
        return;
    }
    const double operator_norm = std::sqrt(operator_norm_sq_double);
    if (!finite_double_bits(operator_norm)) {
        fail_nonfinite_certificate();
        return;
    }
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        const double residual_sq = static_cast<double>(
            residual_norm_sq[static_cast<std::size_t>(rhs)]);
        const double original_sq = static_cast<double>(
            original_norm_sq[static_cast<std::size_t>(rhs)]);
        if (!finite_double_bits(residual_sq) ||
            !finite_double_bits(original_sq) ||
            residual_sq < 0.0 || original_sq < 0.0) {
            fail_nonfinite_certificate();
            return;
        }
        const double absolute = std::sqrt(residual_sq);
        const double original_norm = std::sqrt(original_sq);
        const double scale = operator_norm * original_norm;
        if (!finite_double_bits(absolute) ||
            !finite_double_bits(original_norm) ||
            !finite_double_bits(scale)) {
            fail_nonfinite_certificate();
            return;
        }
        double relative = 0.0;
        if (scale > 0.0) {
            relative = absolute / scale;
        } else if (absolute > 0.0) {
            relative = std::numeric_limits<double>::max();
        }
        if (!finite_double_bits(relative)) {
            fail_nonfinite_certificate();
            return;
        }
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
    }
    result.abs_residual = max_absolute;
    result.abs_residual_rel = max_relative;

    const double requested_tol =
        effective_absorption_tolerance(options);
    if (!finite_double_bits(requested_tol)) {
        fail_nonfinite_certificate();
        return;
    }
    const double certificate_limit =
        std::max(8.0 * std::max(0.0, requested_tol),
                 64.0 * std::numeric_limits<double>::epsilon());
    if (!finite_double_bits(certificate_limit)) {
        fail_nonfinite_certificate();
        return;
    }
    result.precision_certified =
        result.converged && max_relative <= certificate_limit;
}

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_USE_CUDA
