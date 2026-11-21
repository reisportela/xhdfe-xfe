// Phase 3 of the Stata-path plan: the dense rank of a wide identifier column
// (a value outside exact int32) from its (value, row) pairs sorted by value,
// computed on a team. The code of a row is the number of distinct values below
// its value, which is what the serial pass of read_categorical_numeric
// (xhdfe_plugin.cpp) gives: its counter rises at every break, j == 0 or
// s[j].value != s[j-1].value. Equal values are contiguous after the sort, so
// their order never changes a code. Each of `chunks` fixed chunks counts its
// breaks, the first against its global predecessor; an exclusive prefix of the
// counts gives each chunk its offset; each chunk then writes codes[s[j].row].
// The rows are distinct, so the writes are disjoint. An empty chunk (more
// chunks than rows) counts nothing and reads nothing. The result does not
// depend on the team or the chunks. Shared by xhdfe_plugin.cpp and
// tests/wide_id_rank_unit.cpp.
#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xhdfe_wide_id {

// A trivial type: a buffer of pairs can be allocated without being written, so
// that the first touch follows the read's pretouch policy.
struct Pair {
    double value;
    int row;
};

struct RankResult {
    long long distinct = 0;  // distinct values (the largest code plus one)
    int team = 1;            // team observed in the region
    int workers = 0;         // threads that ranked at least one row
};

// s: n pairs sorted by value (any order among equal values); codes: n slots,
// indexed by the rows of the pairs. chunks >= 1 (values below 1 count as 1).
inline RankResult rank_sorted(const Pair* s, long long n, int* codes, long long chunks,
                              int threads) {
    RankResult result;
    if (n <= 0) {
        return result;
    }
    if (chunks < 1) {
        chunks = 1;
    }
    std::vector<long long> counts(static_cast<std::size_t>(chunks), 0);
    std::vector<long long> offset(static_cast<std::size_t>(chunks) + 1, 0);
    std::atomic<int> active{0};
    int team = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#else
    (void)threads;
#endif
    {
#ifdef _OPENMP
#pragma omp single
        team = omp_get_num_threads();
#endif
        int ranked = 0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long long c = 0; c < chunks; ++c) {
            const long long lo = n * c / chunks;
            const long long hi = n * (c + 1) / chunks;
            long long breaks = 0;
            for (long long j = lo; j < hi; ++j) {
                if (j == 0 || s[j].value != s[j - 1].value) {
                    ++breaks;
                }
            }
            counts[static_cast<std::size_t>(c)] = breaks;
        }
#ifdef _OPENMP
#pragma omp single
#endif
        {
            for (long long c = 0; c < chunks; ++c) {
                offset[static_cast<std::size_t>(c) + 1] =
                    offset[static_cast<std::size_t>(c)] + counts[static_cast<std::size_t>(c)];
            }
        }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long long c = 0; c < chunks; ++c) {
            const long long lo = n * c / chunks;
            const long long hi = n * (c + 1) / chunks;
            long long running = offset[static_cast<std::size_t>(c)];
            for (long long j = lo; j < hi; ++j) {
                if (j == 0 || s[j].value != s[j - 1].value) {
                    ++running;
                }
                codes[s[j].row] = static_cast<int>(running - 1);
                ++ranked;
            }
        }
        if (ranked > 0) {
            active.fetch_add(1, std::memory_order_relaxed);
        }
    }
    result.distinct = offset[static_cast<std::size_t>(chunks)];
    result.team = team;
    result.workers = active.load();
    return result;
}

}  // namespace xhdfe_wide_id
