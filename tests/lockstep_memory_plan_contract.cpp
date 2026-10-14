#include "lockstep_memory_plan.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

using namespace hdfe::detail;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL " << message << '\n';
        std::exit(1);
    }
}

int main() {
    // The independently reproduced 31-RHS failure has no narrow-RHS exemption.
    for (std::size_t rhs : {1U, 5U, 20U, 30U, 31U, 32U, 60U, 250U}) {
        const auto plan = lockstep_memory_plan(282191, rhs, 136674, 56959, 1);
        require(plan.valid && !plan.eligible, "E1 narrow/wide memory gate");
        require(plan.max_chunks == 0 && plan.execution_chunks(48) == 0,
                "no serial fallback when levels exceed joint budget");
    }
    const auto ready = lockstep_memory_plan(46156187, 5, 5684794, 610115, 1);
    const auto akm = lockstep_memory_plan(55947171, 7, 6748094, 799301, 1);
    require(ready.eligible && akm.eligible, "protected ready/AKM dispatch");
    require(ready.execution_chunks(48) == 48 && akm.execution_chunks(48) == 48,
            "protected 48-chunk arithmetic");

    for (std::size_t rhs : {5U, 31U, 32U, 60U, 250U}) {
        for (std::size_t contiguous : {0U, 1U, 2U}) {
            const std::size_t rows = 20000000, levels = 2000, scattered = 1000;
            const auto baseline = lockstep_memory_plan(rows, rhs, levels,
                                                       scattered, contiguous);
            require(baseline.eligible, "ample-row memory geometry");
            for (int capacity : {48, 64, 96}) {
                ScopedDeterministicParallelCapacity override(capacity);
                const auto plan = lockstep_memory_plan(rows, rhs, levels,
                                                       scattered, contiguous);
                require(plan.eligible == baseline.eligible &&
                        plan.decision_chunks == baseline.decision_chunks &&
                        plan.budget == baseline.budget &&
                        plan.max_private_bytes == baseline.max_private_bytes,
                        "routing independent of capacity");
                const int chunks = plan.execution_chunks(
                    deterministic_parallel_chunk_count(rows));
                require(chunks == capacity, "no fixed execution ceiling of 48");
                // Sum the three actually allocated families independently.
                const std::size_t scatter = chunks * scattered * rhs * sizeof(double);
                const std::size_t carries = chunks * contiguous * rhs * sizeof(double);
                const std::size_t partials = chunks * rhs * sizeof(double);
                require(scatter + carries + partials <= plan.budget,
                        "joint allocation bound");
                require(scatter + carries + partials + plan.level_bytes <=
                            3U * plan.row_bytes,
                        "execution obeys routing bound at larger capacities");
            }
        }
    }
    // A large contiguous carry changes the legal partition even with U=0.
    const auto carries = lockstep_memory_plan(300000, 250, 20000, 0, 20000);
    require(carries.valid && carries.execution_chunks(96) == 29,
            "contiguous carries counted in execution budget");
    require(!carries.eligible && carries.legacy_decision_chunks == 48,
            "changed historical partition is diverted");
    require(30U * 20000U * 250U * sizeof(double) <= carries.budget &&
            30U * (20000U + 1U) * 250U * sizeof(double) > carries.budget,
            "partials prevent an otherwise legal carry partition");
    const auto capacity_case = lockstep_memory_plan(300000, 32, 15000, 10000, 1);
    require(capacity_case.eligible && capacity_case.max_chunks == 79,
            "N9 prices the 48-chunk decision rather than filling the budget");
    require(capacity_case.decision_private_bytes ==
                48U * capacity_case.chunk_bytes &&
            capacity_case.decision_private_bytes < capacity_case.max_private_bytes,
            "decision allocation distinguished from capacity bound");
    require(lockstep_memory_plan(300000, 32, 15000, 10000, 0).eligible,
            "original abstract N9 vector intentionally changes to lockstep");
    for (int capacity : {48, 64, 96}) {
        ScopedDeterministicParallelCapacity override(capacity);
        require(lockstep_memory_plan(300000, 32, 15000, 10000, 1).eligible,
                "N9 route independent of host capacity");
        require(capacity_case.execution_chunks(capacity) * capacity_case.chunk_bytes
                    <= capacity_case.budget,
                "N9 joint partition bound");
        require(capacity_case.execution_chunks(capacity) * capacity_case.chunk_bytes +
                    capacity_case.level_bytes <= 3U * capacity_case.row_bytes,
                "N9 execution cannot exceed joint routing bound");
    }
    const auto boundary = lockstep_memory_plan(266000, 250, 21184, 11184, 1);
    require(boundary.valid && boundary.decision_chunks == 47 &&
            boundary.legacy_decision_chunks == 48 && !boundary.eligible,
            "independently reproduced retained-route bit change is diverted");
    const auto historical31 = lockstep_memory_plan(300000, 250, 20000, 19000, 1);
    require(historical31.eligible && historical31.decision_chunks == 31 &&
            historical31.legacy_decision_chunks == 31,
            "eligible reduced partition matches historical cap");
    for (const auto& plan : {ready, akm, capacity_case, historical31}) {
        require(plan.eligible, "retained partition invariant fixture");
        for (int chunks = 1; chunks <= kLegacyDeterministicChunkFloor; ++chunks) {
            require(plan.execution_chunks(chunks) == plan.legacy_execution_chunks(chunks),
                    "every retained request up to 48 preserves historical partition");
        }
    }
    const auto github = lockstep_memory_plan(540674, 3, 44726, 23195, 1);
    require(github.eligible && github.execution_chunks(48) == 48 &&
            github.execution_chunks(96) == 56 && github.legacy_execution_chunks(96) == 96,
            "historical partition retained and larger-capacity class explicit");
    const auto sub512 = lockstep_memory_plan(1000000, 50, 2000, 1000, 1);
    require(sub512.row_bytes < (std::size_t{512} << 20) && sub512.eligible,
            "small row storage does not imply unconditional diversion");
    const auto tiny = lockstep_memory_plan(7, 1, 1, 0, 1);
    require(tiny.valid && !tiny.eligible && tiny.decision_chunks == 7,
            "minimum logical units");
    const auto largest = std::numeric_limits<std::size_t>::max();
    require(!lockstep_memory_plan(largest, 250, 1, 1, 0).valid,
            "row multiplication overflow");
    require(!lockstep_memory_plan(300000, 1, largest, largest, 1).valid,
            "chunk addition overflow");
    require(!lockstep_memory_plan(300000, 1, largest, 0, 1).valid,
            "level multiplication overflow");
    require(!lockstep_memory_plan(0, 5, 1, 1, 0).eligible &&
            !lockstep_memory_plan(300000, 0, 1, 1, 0).eligible,
            "empty geometry");
    require(lockstep_memory_plan(0, 5, 1, 1, 0).execution_chunks(48) == 0,
            "invalid plans have no executable partition");
    std::cout << "PASS lockstep fixed-48 decision, joint execution, partition, 48/64/96, overflow\n";
}
