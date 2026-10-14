#ifndef HDFE_LOCKSTEP_MEMORY_PLAN_HPP
#define HDFE_LOCKSTEP_MEMORY_PLAN_HPP

#include <algorithm>
#include <cstddef>
#include <limits>

#include "hdfe/deterministic_parallel.hpp"

namespace hdfe {
namespace detail {

struct LockstepMemoryPlan {
    std::size_t row_bytes = 0;
    std::size_t budget = 0;
    std::size_t chunk_bytes = 0;
    std::size_t level_bytes = 0;
    std::size_t joint_budget = 0;
    std::size_t max_chunks = 0;
    std::size_t legacy_max_chunks = 0;
    std::size_t max_private_bytes = 0;
    std::size_t decision_private_bytes = 0;
    std::size_t decision_working_bytes = 0;
    int decision_chunks = 0;
    int legacy_decision_chunks = 0;
    bool valid = false;
    bool eligible = false;

    int execution_chunks(int requested_chunks) const noexcept {
        if (!valid || max_chunks == 0 || requested_chunks <= 0) return 0;
        return static_cast<int>(std::min(
            static_cast<std::size_t>(requested_chunks), max_chunks));
    }

    int legacy_execution_chunks(int requested_chunks) const noexcept {
        if (!valid || legacy_max_chunks == 0 || requested_chunks <= 0) return 0;
        return static_cast<int>(std::min(
            static_cast<std::size_t>(requested_chunks), legacy_max_chunks));
    }
};

// Routing prices the historical 48-chunk partition, using only data. Execution
// may grow on larger hosts, but its scatter/carry/partial storage obeys BOTH
// the byte budget and the same 3R-minus-level-storage bound used by routing.
// These are selected scratch requests, not a bound on whole-process RSS.
inline LockstepMemoryPlan lockstep_memory_plan(std::size_t rows,
                                               std::size_t rhs,
                                               std::size_t levels,
                                               std::size_t private_levels,
                                               std::size_t contiguous_dims) noexcept {
    LockstepMemoryPlan plan;
    auto multiply = [](std::size_t a, std::size_t b, std::size_t& out) {
        if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
            return false;
        out = a * b;
        return true;
    };
    auto add = [](std::size_t a, std::size_t b, std::size_t& out) {
        if (a > std::numeric_limits<std::size_t>::max() - b) return false;
        out = a + b;
        return true;
    };
    std::size_t lane_bytes = 0, twice_rows = 0, three_rows = 0;
    std::size_t chunk_entries = 0, level_entries = 0, legacy_chunk_bytes = 0;
    if (rows == 0 || rhs == 0 || private_levels > levels ||
        !multiply(rhs, sizeof(double), lane_bytes) ||
        !multiply(rows, lane_bytes, plan.row_bytes) ||
        !multiply(2, plan.row_bytes, twice_rows) ||
        !multiply(3, plan.row_bytes, three_rows) ||
        !add(private_levels, contiguous_dims, chunk_entries) ||
        !add(chunk_entries, 1, chunk_entries) ||
        !multiply(chunk_entries, lane_bytes, plan.chunk_bytes) ||
        !multiply(private_levels, lane_bytes, legacy_chunk_bytes) ||
        !multiply(7, levels, level_entries) ||
        !multiply(level_entries, lane_bytes, plan.level_bytes)) {
        return plan;
    }
    plan.budget = std::max(std::size_t{1} << 30, twice_rows);
    plan.legacy_max_chunks = legacy_chunk_bytes == 0 ? rows :
        std::min(rows, std::max(std::size_t{1}, plan.budget / legacy_chunk_bytes));
    plan.legacy_decision_chunks = static_cast<int>(std::min(
        static_cast<std::size_t>(kLegacyDeterministicChunkFloor), plan.legacy_max_chunks));
    plan.valid = true;
    plan.decision_working_bytes = plan.level_bytes;
    if (plan.level_bytes > three_rows) return plan;
    plan.joint_budget = three_rows - plan.level_bytes;
    plan.max_chunks = std::min(rows, std::min(plan.budget / plan.chunk_bytes,
                                             plan.joint_budget / plan.chunk_bytes));
    plan.decision_chunks = static_cast<int>(std::min(
        static_cast<std::size_t>(kLegacyDeterministicChunkFloor), plan.max_chunks));
    if (!multiply(plan.max_chunks, plan.chunk_bytes, plan.max_private_bytes) ||
        !multiply(static_cast<std::size_t>(plan.decision_chunks), plan.chunk_bytes,
                  plan.decision_private_bytes) ||
        !add(plan.decision_private_bytes, plan.level_bytes, plan.decision_working_bytes)) {
        plan.valid = false;
        return plan;
    }
    // A reduced partition is a different arithmetic route. At historical
    // capacity, retain lockstep only when its row partition is unchanged.
    plan.eligible = plan.decision_chunks >= 8 &&
                    plan.decision_chunks == plan.legacy_decision_chunks;
    return plan;
}

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_LOCKSTEP_MEMORY_PLAN_HPP
