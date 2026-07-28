#ifndef HDFE_DETERMINISTIC_PARALLEL_HPP
#define HDFE_DETERMINISTIC_PARALLEL_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

namespace hdfe {
namespace detail {

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
