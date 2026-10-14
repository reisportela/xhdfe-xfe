#ifndef HDFE_DETERMINISTIC_PARALLEL_HPP
#define HDFE_DETERMINISTIC_PARALLEL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

// Arithmetic exactly as written, whatever the flags of the build: inside a
// function marked XHDFE_EXACT_FP no sum or product is reordered and no
// division becomes a product. GCC takes the attribute; clang takes the
// pragmas that open the body. Neither is asked to keep a product apart from
// the sum it feeds, which clang does not honour under -ffast-math: exact
// kernels write every such pair as std::fma, or as target_mul_add() below
// when they must give the bits of a two-pass kernel on every target.
#if defined(__clang__)
#define XHDFE_EXACT_FP
#define XHDFE_EXACT_FP_BODY \
    _Pragma("float_control(precise, on)") _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define XHDFE_EXACT_FP                                                     \
    __attribute__((optimize("fp-contract=off", "no-associative-math",      \
                            "no-reciprocal-math", "signed-zeros",          \
                            "no-unsafe-math-optimizations")))
#define XHDFE_EXACT_FP_BODY
#else
#define XHDFE_EXACT_FP
#define XHDFE_EXACT_FP_BODY
#endif

namespace hdfe {
namespace detail {

// a * b + c as the two-pass kernels compute it in this translation unit: one
// rounding only when the target has FMA and the build permits contraction,
// two roundings when XHDFE_FP_CONTRACT_OFF matches -ffp-contract=off.
// An exact kernel that must give the bits of a two-pass kernel on every
// target writes its product-sums with this; std::fma alone would round
// once where the two-pass kernel rounds twice. The body depends on the
// target flags of the translation unit: call it only from the .cpp units
// of the core, which one build compiles with one set of flags, never from
// a .cu unit (nvcc's host pass may not see the same -march).
XHDFE_EXACT_FP
inline double target_mul_add(double a, double b, double c) {
    XHDFE_EXACT_FP_BODY
#if (defined(__FP_FAST_FMA) || defined(FP_FAST_FMA)) && !defined(XHDFE_FP_CONTRACT_OFF)
    return std::fma(a, b, c);
#else
    const double product = a * b;
    return product + c;
#endif
}

// Preserve the historical arithmetic graph on machines with at most 48
// runtime-visible processors.  On larger machines the logical chunk count
// grows with runtime capacity, but never with the thread count requested for a
// particular fit.  Consequently all thread requests on the same machine and
// runtime configuration combine the same partials in the same order.
constexpr int kLegacyDeterministicChunkFloor = 48;

inline int runtime_thread_capacity() noexcept {
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
    return std::max(
        1, std::min(omp_get_num_procs(), omp_get_thread_limit()));
#else
    return 1;
#endif
}

inline int deterministic_chunk_target_for_capacity(int capacity) noexcept {
    return std::max(kLegacyDeterministicChunkFloor, std::max(1, capacity));
}

inline int& deterministic_capacity_override_slot() noexcept {
    static thread_local int capacity = 0;
    return capacity;
}

/**
 * Captures the runtime-visible capacity for one fit/command.
 *
 * The override is task-local, so concurrent fits with different requested
 * teams still share a request-independent chunk plan.  A non-positive
 * capacity inherits the enclosing plan.
 */
class ScopedDeterministicParallelCapacity {
public:
    explicit ScopedDeterministicParallelCapacity(int capacity) noexcept
        : previous_(deterministic_capacity_override_slot()) {
        if (capacity > 0) {
            deterministic_capacity_override_slot() = capacity;
        }
    }

    ~ScopedDeterministicParallelCapacity() {
        deterministic_capacity_override_slot() = previous_;
    }

    ScopedDeterministicParallelCapacity(
        const ScopedDeterministicParallelCapacity&) = delete;
    ScopedDeterministicParallelCapacity& operator=(
        const ScopedDeterministicParallelCapacity&) = delete;

private:
    int previous_ = 0;
};

inline int deterministic_chunk_target() noexcept {
    const int captured_capacity = deterministic_capacity_override_slot();
    return deterministic_chunk_target_for_capacity(
        captured_capacity > 0 ? captured_capacity
                              : runtime_thread_capacity());
}

template <typename Index>
inline int deterministic_parallel_chunk_count(Index n) noexcept {
    if (n <= 0) {
        return 0;
    }
    return static_cast<int>(std::min<Index>(
        n, static_cast<Index>(deterministic_chunk_target())));
}

template <typename Index>
inline Index deterministic_parallel_chunk_begin(
    Index n, int chunk, int chunks) noexcept {
    const Index base = n / static_cast<Index>(chunks);
    const Index extra = n % static_cast<Index>(chunks);
    return static_cast<Index>(chunk) * base +
           std::min<Index>(static_cast<Index>(chunk), extra);
}

template <typename Index>
inline Index deterministic_parallel_chunk_end(
    Index n, int chunk, int chunks) noexcept {
    return deterministic_parallel_chunk_begin(n, chunk + 1, chunks);
}

// The level sums of the absorption certificate for a dimension without
// slopes:
//     sums[g * columns + c] = sum over the rows of level g of w[row] column_c[row]
// with the association of a chunked scatter: inside each of `chunk_count`
// fixed row chunks the rows of a level are added in order from zero, every
// product fused into its sum, and the chunks are added in order.
// `operator_sums`, when given, takes per chunk the sum of w w in row order.
//
// With `group_ptr` the rows are grouped by level: each level is summed by
// one member of the team and split only where a chunk boundary cuts it.
// Otherwise every chunk scatters into a block of its own and the blocks are
// added level by level. Both give the numbers of the scatter, whatever the
// team.
XHDFE_EXACT_FP
inline void certificate_add_rows(int begin,
                                 int end,
                                 int columns,
                                 const int* group_ids,
                                 const double* const* column,
                                 const double* weights,
                                 double* sums) {
    XHDFE_EXACT_FP_BODY
    for (int row = begin; row < end; ++row) {
#if defined(__GNUC__) || defined(__clang__)
        if (group_ids && row + 16 < end) {
            __builtin_prefetch(
                sums + static_cast<std::size_t>(group_ids[row + 16]) *
                           static_cast<std::size_t>(columns),
                1, 1);
        }
#endif
        const double multiplier = weights ? weights[row] : 1.0;
        double* const out =
            group_ids ? sums + static_cast<std::size_t>(group_ids[row]) *
                                   static_cast<std::size_t>(columns)
                      : sums;
        for (int c = 0; c < columns; ++c) {
            out[c] = target_mul_add(multiplier, column[c][row], out[c]);
        }
    }
}

XHDFE_EXACT_FP
inline void certificate_add_totals(const double* values, int count, double* totals) {
    XHDFE_EXACT_FP_BODY
    for (int k = 0; k < count; ++k) {
        totals[k] += values[k];
    }
}

XHDFE_EXACT_FP
inline double certificate_weight_squares(int begin, int end, const double* weights) {
    XHDFE_EXACT_FP_BODY
    if (weights == nullptr) {
        // A sum of ones is exact.
        return static_cast<double>(end - begin);
    }
    double total = 0.0;
    for (int row = begin; row < end; ++row) {
        total = target_mul_add(weights[row], weights[row], total);
    }
    return total;
}

inline void certificate_level_sums(int n,
                                   int groups,
                                   const int* group_ids,
                                   const int* group_ptr,
                                   const double* const* column,
                                   int columns,
                                   const double* weights,
                                   int chunk_count,
                                   int threads,
                                   double* sums,
                                   double* operator_sums) {
    const int team = std::max(1, threads);
    if (operator_sums) {
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(team) if(team > 1)
#endif
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            operator_sums[chunk] = certificate_weight_squares(
                deterministic_parallel_chunk_begin(n, chunk, chunk_count),
                deterministic_parallel_chunk_end(n, chunk, chunk_count), weights);
        }
    }

    if (group_ptr != nullptr) {
        const int base = n / chunk_count;
        const int extra = n % chunk_count;
        const int long_rows = extra * (base + 1);
        std::vector<std::vector<double>> local_by_thread(
            static_cast<std::size_t>(team),
            std::vector<double>(static_cast<std::size_t>(columns)));
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#pragma omp parallel num_threads(team) if(team > 1)
#endif
        {
            int tid = 0;
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
            tid = omp_get_thread_num();
#endif
            std::vector<double>& local =
                local_by_thread[static_cast<std::size_t>(tid)];
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int g = 0; g < groups; ++g) {
                double* const total =
                    sums + static_cast<std::size_t>(g) * static_cast<std::size_t>(columns);
                int position = group_ptr[g];
                const int last = group_ptr[g + 1];
                std::fill(total, total + columns, 0.0);
                if (position >= last) {
                    continue;
                }
                int chunk = position < long_rows
                                ? position / (base + 1)
                                : extra + (position - long_rows) / std::max(1, base);
                bool first = true;
                while (position < last) {
                    const int stop = std::min(
                        last,
                        deterministic_parallel_chunk_end(n, chunk, chunk_count));
                    if (first) {
                        certificate_add_rows(position, stop, columns, nullptr, column,
                                             weights, total);
                        first = false;
                    } else {
                        std::fill(local.begin(), local.end(), 0.0);
                        certificate_add_rows(position, stop, columns, nullptr, column,
                                             weights, local.data());
                        certificate_add_totals(local.data(), columns, total);
                    }
                    position = stop;
                    ++chunk;
                }
            }
        }
        return;
    }

    const std::size_t block =
        static_cast<std::size_t>(groups) * static_cast<std::size_t>(columns);
    const std::unique_ptr<double[]> partial(
        new double[std::max<std::size_t>(1, block * static_cast<std::size_t>(chunk_count))]);
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(team) if(team > 1)
#endif
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        double* const mine = partial.get() + static_cast<std::size_t>(chunk) * block;
        std::fill(mine, mine + block, 0.0);
        certificate_add_rows(deterministic_parallel_chunk_begin(n, chunk, chunk_count),
                             deterministic_parallel_chunk_end(n, chunk, chunk_count),
                             columns, group_ids, column, weights, mine);
    }
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(team) if(team > 1)
#endif
    for (int g = 0; g < groups; ++g) {
        const std::size_t offset =
            static_cast<std::size_t>(g) * static_cast<std::size_t>(columns);
        double* const total = sums + offset;
        std::copy(partial.get() + offset, partial.get() + offset + columns, total);
        for (int chunk = 1; chunk < chunk_count; ++chunk) {
            certificate_add_totals(
                partial.get() + static_cast<std::size_t>(chunk) * block + offset,
                columns, total);
        }
    }
}

/**
 * A zero-initialized partial-results buffer with a heap-free historical path.
 *
 * Capacity <= InlineCapacity uses the embedded array.  Larger runtime
 * capacities allocate exactly once for the local reduction.  data_ makes
 * indexed access branch-free in both cases.
 */
template <typename T,
          std::size_t InlineCapacity =
              static_cast<std::size_t>(kLegacyDeterministicChunkFloor)>
class InlineOrDynamicBuffer {
public:
    explicit InlineOrDynamicBuffer(std::size_t size)
        : size_(size) {
        if (size_ > InlineCapacity) {
            dynamic_.resize(size_);
            data_ = dynamic_.data();
        }
    }

    InlineOrDynamicBuffer(const InlineOrDynamicBuffer&) = delete;
    InlineOrDynamicBuffer& operator=(const InlineOrDynamicBuffer&) = delete;
    InlineOrDynamicBuffer(InlineOrDynamicBuffer&&) = delete;
    InlineOrDynamicBuffer& operator=(InlineOrDynamicBuffer&&) = delete;

    T& operator[](std::size_t index) noexcept {
        return data_[index];
    }

    const T& operator[](std::size_t index) const noexcept {
        return data_[index];
    }

    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    bool uses_inline_storage() const noexcept {
        return data_ == inline_.data();
    }

private:
    std::array<T, InlineCapacity> inline_{};
    std::vector<T> dynamic_;
    std::size_t size_ = 0;
    T* data_ = inline_.data();
};

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_DETERMINISTIC_PARALLEL_HPP
