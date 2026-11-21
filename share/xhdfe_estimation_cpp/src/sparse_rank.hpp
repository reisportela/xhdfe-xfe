// Sparse identifier ranking for the indexers of the core (phase 1 of the
// Stata-path plan; design note Verifications/perf_stata_path_20261114/fase1/
// DESENHO_FASE1.md).
//
// When the raw int32 identifiers of a fixed effect span more than twice the
// rows (or more than the 50 million levels some indexers allow), the indexers
// used to number them in series: a sort of (identifier, row) pairs for the
// ascending numbering, a hash map for the order of first appearance. Here the
// same numbers come from a bitmap over [min, max] with per-word prefix counts,
// on the caller's team: the rank of an identifier is the number of distinct
// identifiers below it, which is the ascending numbering of the sort, and the
// order of first appearance follows from the ranks and the first row of each
// level. Everything is integer and exact (idempotent ORs, exact minima, sums
// over fixed logical chunks), so the numbers do not depend on the team, the
// schedule or the chunking.
//
// The index takes 8 + 4 bytes per 64 identifiers of the range and is used only
// when that fits min(256 MiB, 8 n) bytes (fits()); otherwise the callers keep
// today's serial branch. The choice is of algorithm, never of numbering.
// XHDFE_SPARSE_RANK=0, read once per process, restores today's serial
// branches at every call site. With XHDFE_PROFILE_CPU=1 the call sites print
// one "cpu_profile label=sparse_rank" line each (profile_line()).
// XHDFE_SPARSE_RANK_VERIFY=1 (tests only) makes every call site that took the
// index compare its result with today's serial branch (verify_*()).
#ifndef XHDFE_SPARSE_RANK_HPP
#define XHDFE_SPARSE_RANK_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/parallel_work_observer.hpp"

// The runner uses OpenMP whenever the translation unit is compiled with it
// (the core defines HDFE_USE_OPENMP as well; a test built with -fopenmp alone
// must not run its teams serially).
#if defined(HDFE_USE_OPENMP) || defined(_OPENMP)
#define XHDFE_SPARSE_RANK_OPENMP 1
#include <omp.h>
#endif

namespace hdfe {
namespace detail {
namespace sparse_rank {

// The budget of bitmap plus prefixes (design note, section 2).
constexpr std::uint64_t kBudgetBytes = std::uint64_t{256} << 20;

// XHDFE_SPARSE_RANK: "0" restores today's serial branches; anything else, or
// nothing, keeps the ranking. Read once per process.
inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("XHDFE_SPARSE_RANK");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// The rule of cpu_profile_enabled() in the core.
inline bool profile_enabled() {
    static const bool on = [] {
        const char* raw = std::getenv("XHDFE_PROFILE_CPU");
        return raw != nullptr && *raw != '\0' && *raw != '0';
    }();
    return on;
}

inline int popcount64(std::uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x = x - ((x >> 1U) & UINT64_C(0x5555555555555555));
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2U) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4U)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    return static_cast<int>((x * UINT64_C(0x0101010101010101)) >> 56U);
#endif
}

// Words of the bitmap over [min_id, max_id], the range formed in 64 bits.
inline std::uint64_t words_for(int min_id, int max_id) noexcept {
    const std::int64_t range =
        static_cast<std::int64_t>(max_id) - static_cast<std::int64_t>(min_id) + 1;
    return range > 0 ? (static_cast<std::uint64_t>(range) + 63U) / 64U : 0U;
}

// The algorithm, never the numbering: the index is used when its 12 bytes per
// word fit the budget and never exceed the 8 n bytes of (identifier, row)
// pairs that today's sorted branch allocates. At most 2^26 words for an int32
// range, so the products cannot overflow.
inline bool fits(int min_id, int max_id, int n,
                 std::uint64_t budget = kBudgetBytes) noexcept {
    if (n <= 0 || max_id < min_id) {
        return false;
    }
    const std::uint64_t words = words_for(min_id, max_id);
    const std::uint64_t bytes = words * 12U;
    return words > 0U && bytes <= budget &&
           bytes <= std::uint64_t{8} * static_cast<std::uint64_t>(n);
}

// Why a call site keeps today's serial branch, for the profile line.
inline const char* serial_reason() { return enabled() ? "budget" : "off"; }

// XHDFE_SPARSE_RANK_VERIFY=1 (a test hook, off by default, read once per
// process): every call site that took the index also computes today's serial
// result and compares it field by field. A match prints
// "cpu_profile label=sparse_rank_verify site=<site> ok=1" under
// XHDFE_PROFILE_CPU. A difference always prints the line with ok=0 and the
// field (the tier turns exceptions into declines, so the line is the evidence)
// and throws std::logic_error naming the site; inside an OpenMP region, where
// an exception cannot leave, it is recorded and verify_rethrow() throws it
// after the region.
inline bool verify_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("XHDFE_SPARSE_RANK_VERIFY");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

inline std::atomic<int>& verify_pending() {
    static std::atomic<int> pending{0};
    return pending;
}

inline void verify_ok(const char* site) {
    if (!profile_enabled()) {
        return;
    }
    std::ostringstream line;
    line << "cpu_profile label=sparse_rank_verify site=" << site << " ok=1\n";
    std::cerr << line.str();
}

inline void verify_fail(const char* site, const char* what) {
    std::ostringstream line;
    line << "cpu_profile label=sparse_rank_verify site=" << site << " ok=0 what=" << what
         << '\n';
    std::cerr << line.str();
#if defined(XHDFE_SPARSE_RANK_OPENMP)
    if (omp_in_parallel()) {
        verify_pending().store(1, std::memory_order_relaxed);
        return;
    }
#endif
    throw std::logic_error(std::string("sparse_rank verify: ") + site + ": " + what +
                           " differs from today's serial branch");
}

inline void verify_check(bool same, const char* site, const char* what) {
    if (!same) {
        verify_fail(site, what);
    }
}

// After an OpenMP region whose bodies verify: throw what they recorded.
inline void verify_rethrow() {
    if (verify_pending().exchange(0, std::memory_order_relaxed) != 0) {
        throw std::logic_error(
            "sparse_rank verify: a difference inside a parallel region (see the "
            "sparse_rank_verify line with ok=0)");
    }
}

// The bitmap and its prefix counts. rank(v) is the number of distinct indexed
// identifiers below v, for v among them.
struct Index {
    int min_id = 0;
    int levels = 0;
    std::uint64_t words = 0;
    std::unique_ptr<std::atomic<std::uint64_t>[]> bits;
    std::unique_ptr<std::uint32_t[]> prefix;

    int rank(int v) const noexcept {
        const std::uint64_t u = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(v) - static_cast<std::int64_t>(min_id));
        const std::uint64_t w = u >> 6U;
        const std::uint64_t below = (std::uint64_t{1} << (u & 63U)) - 1U;
        return static_cast<int>(
            prefix[w] + static_cast<std::uint32_t>(popcount64(
                            bits[w].load(std::memory_order_relaxed) & below)));
    }

    void clear() noexcept {
        bits.reset();
        prefix.reset();
        words = 0;
        levels = 0;
    }
};

// Runs body(unit) for every unit in [0, units) on a team of the caller's size
// (never more), reporting to the caller's observer when there is one. Serial
// without OpenMP, with one thread or with one unit. The bodies do not throw.
struct OmpRun {
    int threads = 1;
    ParallelWorkObserver* observer = nullptr;

    template <class Body>
    void operator()(int units, Body&& body) const {
        if (units <= 0) {
            return;
        }
#if defined(XHDFE_SPARSE_RANK_OPENMP)
        const int team = std::max(1, threads);
        if (team > 1 && units > 1) {
            if (observer != nullptr) {
                observer->begin_region(team);
            }
#pragma omp parallel num_threads(team)
            {
                bool observed = false;
#pragma omp for schedule(static)
                for (int unit = 0; unit < units; ++unit) {
                    if (observer != nullptr && !observed) {
                        observer->observe_work();
                        observed = true;
                    }
                    body(unit);
                }
            }
            if (observer != nullptr) {
                observer->end_region();
            }
            return;
        }
#endif
        for (int unit = 0; unit < units; ++unit) {
            body(unit);
        }
    }
};

// Builds the index of raw[0..n), whose minimum and maximum are min_id and
// max_id. False, with nothing allocated, when it does not fit; then the
// caller keeps today's branch. Allocation failures propagate.
template <class Run>
bool build_index(const int* raw, int n, int min_id, int max_id, const Run& run,
                 Index& out, std::uint64_t budget = kBudgetBytes) {
    out.clear();
    if (!fits(min_id, max_id, n, budget)) {
        return false;
    }
    const std::uint64_t words = words_for(min_id, max_id);
    const int word_count = static_cast<int>(words);  // at most 2^26
    std::unique_ptr<std::atomic<std::uint64_t>[]> bits(
        new std::atomic<std::uint64_t>[static_cast<std::size_t>(words)]);
    std::unique_ptr<std::uint32_t[]> prefix(
        new std::uint32_t[static_cast<std::size_t>(words)]);
    const int word_chunks = deterministic_parallel_chunk_count(word_count);
    const int row_chunks = deterministic_parallel_chunk_count(n);
    std::atomic<std::uint64_t>* const bit_words = bits.get();
    std::uint32_t* const prefix_words = prefix.get();
    run(word_chunks, [&](int chunk) {
        const int first = deterministic_parallel_chunk_begin(word_count, chunk, word_chunks);
        const int last = deterministic_parallel_chunk_end(word_count, chunk, word_chunks);
        for (int w = first; w < last; ++w) {
            bit_words[w].store(0U, std::memory_order_relaxed);
        }
    });
    const std::int64_t low = min_id;
    run(row_chunks, [&](int chunk) {
        const int first = deterministic_parallel_chunk_begin(n, chunk, row_chunks);
        const int last = deterministic_parallel_chunk_end(n, chunk, row_chunks);
        for (int i = first; i < last; ++i) {
            const std::uint64_t u =
                static_cast<std::uint64_t>(static_cast<std::int64_t>(raw[i]) - low);
            const std::uint64_t mask = std::uint64_t{1} << (u & 63U);
            std::atomic<std::uint64_t>& word = bit_words[u >> 6U];
            if ((word.load(std::memory_order_relaxed) & mask) == 0U) {
                word.fetch_or(mask, std::memory_order_relaxed);
            }
        }
    });
    std::vector<std::uint64_t> chunk_begin(static_cast<std::size_t>(word_chunks) + 1U, 0U);
    run(word_chunks, [&](int chunk) {
        const int first = deterministic_parallel_chunk_begin(word_count, chunk, word_chunks);
        const int last = deterministic_parallel_chunk_end(word_count, chunk, word_chunks);
        std::uint64_t count = 0U;
        for (int w = first; w < last; ++w) {
            count += static_cast<std::uint64_t>(
                popcount64(bit_words[w].load(std::memory_order_relaxed)));
        }
        chunk_begin[static_cast<std::size_t>(chunk) + 1U] = count;
    });
    for (int chunk = 0; chunk < word_chunks; ++chunk) {
        chunk_begin[static_cast<std::size_t>(chunk) + 1U] +=
            chunk_begin[static_cast<std::size_t>(chunk)];
    }
    run(word_chunks, [&](int chunk) {
        const int first = deterministic_parallel_chunk_begin(word_count, chunk, word_chunks);
        const int last = deterministic_parallel_chunk_end(word_count, chunk, word_chunks);
        std::uint64_t below = chunk_begin[static_cast<std::size_t>(chunk)];
        for (int w = first; w < last; ++w) {
            prefix_words[w] = static_cast<std::uint32_t>(below);
            below += static_cast<std::uint64_t>(
                popcount64(bit_words[w].load(std::memory_order_relaxed)));
        }
    });
    out.min_id = min_id;
    out.levels = static_cast<int>(chunk_begin[static_cast<std::size_t>(word_chunks)]);
    out.words = words;
    out.bits = std::move(bits);
    out.prefix = std::move(prefix);
    return true;
}

// out[i] = the ascending dense rank of raw[i] (the numbering of the sorts).
template <class Run>
void map_ranks(const int* raw, int n, const Index& index, int* out, const Run& run) {
    const int chunks = deterministic_parallel_chunk_count(n);
    run(chunks, [&](int chunk) {
        const int first = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int last = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = first; i < last; ++i) {
            out[i] = index.rank(raw[i]);
        }
    });
}

// Ascending dense codes 0..levels-1 become codes by order of first appearance,
// in place: the first row of every level is an exact minimum, and the code of
// a level is the number of levels whose first row comes before its own,
// counted per block and summed in block order. The scheme of
// build_indexer_lite_team(), with the level given by the rank.
template <class Run>
void first_appearance_in_place(int* codes, int n, int levels, const Run& run) {
    if (n <= 0 || levels <= 0) {
        return;
    }
    std::unique_ptr<std::atomic<int>[]> first_row(
        new std::atomic<int>[static_cast<std::size_t>(levels)]);
    std::vector<int> renumber(static_cast<std::size_t>(levels));
    std::atomic<int>* const first = first_row.get();
    const int level_chunks = deterministic_parallel_chunk_count(levels);
    const int chunks = deterministic_parallel_chunk_count(n);
    run(level_chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(levels, chunk, level_chunks);
        const int hi = deterministic_parallel_chunk_end(levels, chunk, level_chunks);
        for (int level = lo; level < hi; ++level) {
            first[level].store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        }
    });
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = lo; i < hi; ++i) {
            std::atomic<int>& slot = first[codes[i]];
            int seen = slot.load(std::memory_order_relaxed);
            while (i < seen &&
                   !slot.compare_exchange_weak(seen, i, std::memory_order_relaxed)) {
            }
        }
    });
    std::vector<int> block_begin(static_cast<std::size_t>(chunks) + 1U, 0);
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        int firsts = 0;
        for (int i = lo; i < hi; ++i) {
            firsts += first[codes[i]].load(std::memory_order_relaxed) == i ? 1 : 0;
        }
        block_begin[static_cast<std::size_t>(chunk) + 1U] = firsts;
    });
    for (int chunk = 0; chunk < chunks; ++chunk) {
        block_begin[static_cast<std::size_t>(chunk) + 1U] +=
            block_begin[static_cast<std::size_t>(chunk)];
    }
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        int code = block_begin[static_cast<std::size_t>(chunk)];
        for (int i = lo; i < hi; ++i) {
            const int level = codes[i];
            if (first[level].load(std::memory_order_relaxed) == i) {
                renumber[static_cast<std::size_t>(level)] = code++;
            }
        }
    });
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = lo; i < hi; ++i) {
            codes[i] = renumber[static_cast<std::size_t>(codes[i])];
        }
    });
}

// True when values[0..n) never decrease.
template <class Run>
bool non_decreasing(const int* values, int n, const Run& run) {
    if (n <= 1) {
        return true;
    }
    const int chunks = deterministic_parallel_chunk_count(n);
    std::vector<unsigned char> rising(static_cast<std::size_t>(chunks), 1U);
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        int previous = lo > 0 ? values[lo - 1] : values[lo];
        for (int i = lo; i < hi; ++i) {
            if (values[i] < previous) {
                rising[static_cast<std::size_t>(chunk)] = 0U;
                return;
            }
            previous = values[i];
        }
    });
    return std::all_of(rising.begin(), rising.end(),
                       [](unsigned char flag) { return flag != 0U; });
}

// The group_ptr of finalize_contiguous_groups() for non-decreasing ids in
// [0, groups), groups + 1 entries: the levels up to the first one begin at 0,
// every later level at the one row where the ids rise to it or past it, and
// the levels after the last one at n. Every entry is written exactly once.
template <class Run>
void fill_group_ptr(const int* ids, int n, int groups, int* group_ptr, const Run& run) {
    if (n <= 0 || groups <= 0) {
        return;
    }
    const int first_group = ids[0];
    const int last_group = ids[n - 1];
    for (int g = 0; g <= first_group; ++g) {
        group_ptr[g] = 0;
    }
    const int chunks = deterministic_parallel_chunk_count(n);
    run(chunks, [&](int chunk) {
        const int lo = std::max(1, deterministic_parallel_chunk_begin(n, chunk, chunks));
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = lo; i < hi; ++i) {
            const int g = ids[i];
            for (int fill = ids[i - 1] + 1; fill <= g; ++fill) {
                group_ptr[fill] = i;
            }
        }
    });
    for (int fill = last_group + 1; fill <= groups; ++fill) {
        group_ptr[fill] = n;
    }
}

// Phase 2 of the Stata-path plan (design note Verifications/perf_stata_path_20261114/
// fase2/DESENHO_FASE2.md, section 6): does build_indexer() (src/fe_absorption.cpp)
// number an effect of n rows, whose identifiers lie in [min_id, max_id] with
// `levels` distinct values, by their ascending compact ranks, that is with
// num_groups == num_levels_present? Outside its dense branch (a range above 50
// million or above 2n) always; inside it when fewer than 90 % of the range is
// present (it compacts) or all of it is (raw - min is then the rank). With at least
// 90 % present and holes it keeps raw - min, which is not the rank. The constants
// are build_indexer()'s; the tests compare this rule with the builder.
inline bool indexer_numbers_by_rank(int n, int min_id, int max_id, int levels) noexcept {
    const long long range =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    const bool dense = range > 0 && range <= 50000000LL &&
                       range <= static_cast<long long>(n) * 2LL;
    if (!dense) {
        return true;
    }
    const long long present = levels;
    return present * 10LL < range * 9LL || present == range;
}

// groups_contiguous and group_ptr of build_indexer() for an effect numbered by its
// ascending compact ranks `codes` (0..levels-1): the rows are grouped exactly when
// the codes never fall, and group_ptr then has levels + 1 entries; the scheme of
// build_indexer_sparse_rank(). Returns groups_contiguous; group_ptr is left empty
// when it is false.
template <class Run>
bool contiguous_group_ptr(const int* codes, int n, int levels, std::vector<int>& group_ptr,
                          const Run& run) {
    group_ptr.clear();
    if (n <= 0 || levels <= 0 || !non_decreasing(codes, n, run)) {
        return false;
    }
    group_ptr.assign(static_cast<std::size_t>(levels) + 1U, n);
    fill_group_ptr(codes, n, levels, group_ptr.data(), run);
    return true;
}

// The question of fe_nested_within_cluster(): does every level of fe meet a
// single value of clusters? On the index of fe: the first row of every level
// by an exact minimum, then every row against that row, one flag per block.
// The answer does not depend on which row stands for a level. Not used by the
// core: the bench of 14 Nov found it slower than today's hash, which stops at
// the first conflict (Verifications/perf_stata_path_20261114/fase1/
// DECISAO_VISTAS_BENCH_20261114.md); kept for the unit test and the bench.
template <class Run>
bool nested_by_rank(const int* fe, const int* clusters, int n, const Index& index,
                    const Run& run) {
    const int levels = index.levels;
    if (n <= 0 || levels <= 0) {
        return true;
    }
    std::unique_ptr<std::atomic<int>[]> first_row(
        new std::atomic<int>[static_cast<std::size_t>(levels)]);
    std::atomic<int>* const first = first_row.get();
    const int level_chunks = deterministic_parallel_chunk_count(levels);
    const int chunks = deterministic_parallel_chunk_count(n);
    run(level_chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(levels, chunk, level_chunks);
        const int hi = deterministic_parallel_chunk_end(levels, chunk, level_chunks);
        for (int level = lo; level < hi; ++level) {
            first[level].store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        }
    });
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = lo; i < hi; ++i) {
            std::atomic<int>& slot = first[index.rank(fe[i])];
            int seen = slot.load(std::memory_order_relaxed);
            while (i < seen &&
                   !slot.compare_exchange_weak(seen, i, std::memory_order_relaxed)) {
            }
        }
    });
    std::vector<unsigned char> same(static_cast<std::size_t>(chunks), 1U);
    run(chunks, [&](int chunk) {
        const int lo = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const int hi = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (int i = lo; i < hi; ++i) {
            const int row = first[index.rank(fe[i])].load(std::memory_order_relaxed);
            if (clusters[i] != clusters[row]) {
                same[static_cast<std::size_t>(chunk)] = 0U;
                return;
            }
        }
    });
    return std::all_of(same.begin(), same.end(),
                       [](unsigned char flag) { return flag != 0U; });
}

// One line per call site under XHDFE_PROFILE_CPU: the positive control of the
// tests (path=bitmap, or path=serial with reason=off|budget) and the
// measurement of the sites. Each line is written at once.
inline void profile_line(const char* site, const char* path, const char* reason, int n,
                         int min_id, int max_id, int levels, int team,
                         const std::chrono::steady_clock::time_point& start) {
    if (!profile_enabled()) {
        return;
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    const long long range =
        n > 0 ? static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL : 0LL;
    std::ostringstream line;
    line << "cpu_profile label=sparse_rank site=" << site << " path=" << path
         << " reason=" << reason << " n=" << n << " range=" << range << " levels=" << levels
         << " words=" << words_for(min_id, max_id) << " team=" << team
         << " ms=" << elapsed.count() << '\n';
    std::cerr << line.str();
}

// The component counter's own line under XHDFE_PROFILE_CPU (the marking test
// of phase 1): which path formed the unions (marking or serial) and the modes
// of the two lookups. It may be written from inside a parallel region.
inline void components_line(const char* path, const char* base_mode, const char* other_mode,
                            long long rows, long long cells,
                            const std::chrono::steady_clock::time_point& start) {
    if (!profile_enabled()) {
        return;
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    std::ostringstream line;
    line << "cpu_profile label=sparse_rank site=components path=" << path
         << " base=" << base_mode << " other=" << other_mode << " n=" << rows
         << " cells=" << cells << " ms=" << elapsed.count() << '\n';
    std::cerr << line.str();
}

}  // namespace sparse_rank
}  // namespace detail
}  // namespace hdfe

#endif  // XHDFE_SPARSE_RANK_HPP
