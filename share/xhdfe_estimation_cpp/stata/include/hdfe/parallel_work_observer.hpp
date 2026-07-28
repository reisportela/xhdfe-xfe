#ifndef HDFE_PARALLEL_WORK_OBSERVER_HPP
#define HDFE_PARALLEL_WORK_OBSERVER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe {
namespace detail {

/**
 * Records OpenMP workers that execute real xhdfe loop iterations.
 *
 * The hot-path cost is paid once per worker and observation epoch: subsequent
 * calls from the same worker are a thread-local comparison. Worker
 * registrations are backed by dynamically sized epoch slots, so the
 * diagnostic has no compile-time worker ceiling.
 */
class ParallelWorkObserver {
public:
    ParallelWorkObserver()
        : instance_id_(next_instance_id()) {
        ensure_worker_capacity(1);
    }

    void reset() {
        if (region_open_.load(std::memory_order_acquire)) {
            throw std::logic_error(
                "ParallelWorkObserver::reset called during an active region");
        }
        region_active_workers_.store(0, std::memory_order_relaxed);
        max_team_size_.store(1, std::memory_order_relaxed);
        max_active_workers_.store(1, std::memory_order_relaxed);
        capacity_mismatch_.store(false, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
    }

    void begin_region(int expected_team) {
        bool expected_open = false;
        if (!region_open_.compare_exchange_strong(
                expected_open, true, std::memory_order_acq_rel)) {
            throw std::logic_error(
                "ParallelWorkObserver does not support overlapping regions");
        }
        const std::size_t declared_team =
            static_cast<std::size_t>(
                expected_team > 0 ? expected_team : 1);
        try {
            ensure_worker_capacity(declared_team);
        } catch (...) {
            region_open_.store(false, std::memory_order_release);
            throw;
        }
        region_expected_team_ = declared_team;
        region_active_workers_.store(0, std::memory_order_relaxed);
        capacity_mismatch_.store(false, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
    }

    void observe_work() noexcept {
        if (!region_open_.load(std::memory_order_acquire)) {
            capacity_mismatch_.store(true, std::memory_order_release);
            return;
        }
#ifdef HDFE_USE_OPENMP
        const std::uint64_t epoch = epoch_.load(std::memory_order_acquire);
        static thread_local std::uint64_t last_instance_id = 0;
        static thread_local std::uint64_t last_epoch = 0;
        if (last_instance_id == instance_id_ && last_epoch == epoch) {
            return;
        }
        last_instance_id = instance_id_;
        last_epoch = epoch;

        const int team_size = omp_in_parallel() ? omp_get_num_threads() : 1;
        if (team_size > 0 &&
            static_cast<std::size_t>(team_size) > region_expected_team_) {
            capacity_mismatch_.store(true, std::memory_order_release);
        }
        int previous = max_team_size_.load(std::memory_order_relaxed);
        while (previous < team_size &&
               !max_team_size_.compare_exchange_weak(
                   previous, team_size, std::memory_order_relaxed)) {
        }

        const int thread_id = omp_in_parallel() ? omp_get_thread_num() : 0;
        if (thread_id >= 0 &&
            static_cast<std::size_t>(thread_id) < worker_capacity_) {
            const std::uint64_t previous_epoch =
                worker_epoch_[static_cast<std::size_t>(thread_id)].exchange(
                    epoch, std::memory_order_acq_rel);
            if (previous_epoch != epoch) {
                region_active_workers_.fetch_add(
                    1, std::memory_order_acq_rel);
            }
        } else {
            // begin_region(expected_team) is called before every observed
            // OpenMP region. Reaching this branch means a caller formed a team
            // larger than it declared; retain a visible invariant failure
            // rather than silently wrapping or aliasing worker identities.
            capacity_mismatch_.store(true, std::memory_order_release);
        }
#else
        max_team_size_.store(1, std::memory_order_relaxed);
        const std::uint64_t epoch = epoch_.load(std::memory_order_acquire);
        const std::uint64_t previous_epoch =
            worker_epoch_[0].exchange(epoch, std::memory_order_acq_rel);
        if (previous_epoch != epoch) {
            region_active_workers_.fetch_add(
                1, std::memory_order_acq_rel);
        }
#endif
    }

    void end_region() {
        if (!region_open_.exchange(false, std::memory_order_acq_rel)) {
            throw std::logic_error(
                "ParallelWorkObserver::end_region called without begin_region");
        }
        int count =
            region_active_workers_.load(std::memory_order_acquire);
        count = count > 0 ? count : 1;
        int previous = max_active_workers_.load(std::memory_order_relaxed);
        while (previous < count &&
               !max_active_workers_.compare_exchange_weak(
                   previous, count, std::memory_order_relaxed)) {
        }
        if (capacity_mismatch_.load(std::memory_order_acquire)) {
            throw std::logic_error(
                "ParallelWorkObserver observed a team larger than "
                "begin_region(expected_team)");
        }
    }

    int max_team_size() const noexcept {
        return max_team_size_.load(std::memory_order_relaxed);
    }

    int active_workers() const noexcept {
        return max_active_workers_.load(std::memory_order_relaxed);
    }

    bool capacity_mismatch() const noexcept {
        return capacity_mismatch_.load(std::memory_order_acquire);
    }

    std::size_t worker_capacity() const noexcept {
        return worker_capacity_;
    }

private:
    static std::uint64_t next_instance_id() noexcept {
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    void ensure_worker_capacity(std::size_t required) {
        if (required <= worker_capacity_) {
            return;
        }
        std::unique_ptr<std::atomic<std::uint64_t>[]> replacement(
            new std::atomic<std::uint64_t>[required]);
        for (std::size_t i = 0; i < required; ++i) {
            replacement[i].store(0, std::memory_order_relaxed);
        }
        worker_epoch_ = std::move(replacement);
        worker_capacity_ = required;
    }

    std::unique_ptr<std::atomic<std::uint64_t>[]> worker_epoch_;
    std::size_t worker_capacity_ = 0;
    std::size_t region_expected_team_ = 1;
    std::atomic<int> region_active_workers_{0};
    std::atomic<int> max_team_size_{1};
    std::atomic<int> max_active_workers_{1};
    std::atomic<bool> capacity_mismatch_{false};
    std::atomic<bool> region_open_{false};
    std::atomic<std::uint64_t> epoch_{1};
    const std::uint64_t instance_id_;
};

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_PARALLEL_WORK_OBSERVER_HPP
