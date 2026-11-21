// Unit test of src/sparse_rank.hpp (phase 1 of the Stata-path plan, design
// note Verifications/perf_stata_path_20261114/fase1/DESENHO_FASE1.md, test 1).
//
// The bitmap ranking against verbatim copies of today's sparse branches: the
// sort of build_indexer() and finalize_contiguous_groups() (src/fe_absorption.cpp),
// the hash map of build_indexer_lite() and of the Sparse branch of
// build_lookup_compact() (src/hdfe_regressor_v11.cpp, first appearance), and
// the hash map of fe_nested_within_cluster(). Generated identifiers: random
// ranges around the dispatch thresholds (2n, 2n+1, 8.7n, 64n), the budget and
// the 8n guard at their edges (by parameter), duplicates of one level, sorted,
// reverse-sorted and plateaus, negative identifiers, INT_MIN and INT_MAX, n = 1
// and 2, one level per row; with --large also the 50 million cap (R = 50 M and
// 50 M + 1 with n = 25 M + 1). Teams 1, 2, 3, 8, 16, 24 and 48, with a static
// runner and with a dynamic runner that visits the units backwards.
//
// The builders themselves live in anonymous namespaces of the core files; their
// glue around these functions is checked against the frozen binaries by
// tests/sparse_rank_contract.py and tests/stata/sparse_rank_contract.do.
//
// Every case states which path it expects (index built or not) and checks it;
// the run fails if no case built the index. One digest of all outputs is
// printed; it must not depend on the team or the runner.
//
// Build (no Eigen needed), from the repository root:
//   g++ -std=gnu++17 -O2 -fopenmp -DHDFE_USE_OPENMP -Iinclude -Isrc
//       tests/sparse_rank_unit.cpp -o sparse_rank_unit
// Usage: sparse_rank_unit [--large]

#include "sparse_rank.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sr = hdfe::detail::sparse_rank;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 40) {
            std::fprintf(stderr, "FAIL %s\n", what.c_str());
        }
    }
}

// --- References: today's sparse branches, verbatim in substance ------------

struct Ascending {
    std::vector<int> ids;
    int groups = 0;
    bool contiguous = false;
    std::vector<int> group_ptr;
};

// build_indexer(), sparse branch (fa:1019-1049), then finalize_contiguous_groups()
// (fa:657-700).
Ascending ref_ascending(const std::vector<int>& raw) {
    Ascending out;
    const std::size_t n_raw = raw.size();
    out.ids.assign(n_raw, 0);
    std::vector<std::pair<int, int>> keyed(n_raw);
    for (std::size_t i = 0; i < n_raw; ++i) {
        keyed[i] = {raw[i], static_cast<int>(i)};
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
        out.ids[static_cast<std::size_t>(pos)] = next_id;
    }
    out.groups = next_id + 1;
    // finalize_contiguous_groups()
    const int n = static_cast<int>(out.ids.size());
    out.contiguous = false;
    out.group_ptr.clear();
    if (n <= 0 || out.groups <= 0) {
        return out;
    }
    int prev = out.ids[0];
    if (prev < 0 || prev >= out.groups) {
        return out;
    }
    for (int i = 1; i < n; ++i) {
        const int g = out.ids[static_cast<std::size_t>(i)];
        if (g < prev) {
            return out;
        }
        prev = g;
    }
    out.contiguous = true;
    out.group_ptr.assign(static_cast<std::size_t>(out.groups) + 1U, n);
    out.group_ptr[0] = 0;
    const int first_group = out.ids[0];
    for (int g = 1; g <= first_group; ++g) {
        out.group_ptr[static_cast<std::size_t>(g)] = 0;
    }
    int last_group = first_group;
    for (int i = 1; i < n; ++i) {
        const int g = out.ids[static_cast<std::size_t>(i)];
        if (g != last_group) {
            for (int fill = last_group + 1; fill <= g; ++fill) {
                out.group_ptr[static_cast<std::size_t>(fill)] = i;
            }
            last_group = g;
        }
    }
    for (int fill = last_group + 1; fill <= out.groups; ++fill) {
        out.group_ptr[static_cast<std::size_t>(fill)] = n;
    }
    return out;
}

// build_indexer_lite(), sparse branch (hr:1709-1733); also index() of the Sparse
// branch of build_lookup_compact() (hr:1643-1663) row by row.
std::vector<int> ref_first_appearance(const std::vector<int>& raw, int* groups) {
    std::unordered_map<int, int> mapping;
    std::vector<int> out(raw.size(), 0);
    int next_id = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const int key = raw[i];
        auto it = mapping.find(key);
        if (it == mapping.end()) {
            out[i] = next_id;
            mapping.emplace(key, next_id);
            ++next_id;
        } else {
            out[i] = it->second;
        }
    }
    *groups = next_id;
    return out;
}

// fe_nested_within_cluster(), hash branch (hr:9267-9279).
bool ref_nested(const std::vector<int>& fe, const std::vector<int>& clusters) {
    std::unordered_map<int, int> mapping;
    for (std::size_t i = 0; i < fe.size(); ++i) {
        auto it = mapping.find(fe[i]);
        if (it == mapping.end()) {
            mapping.emplace(fe[i], clusters[i]);
        } else if (it->second != clusters[i]) {
            return false;
        }
    }
    return true;
}

// --- Runners -----------------------------------------------------------------

// The units on a team with a dynamic schedule, visited backwards: the results
// must not depend on the schedule (the tier's Team uses a dynamic one).
struct ReverseDynamicRun {
    int threads = 1;
    template <class Body>
    void operator()(int units, Body&& body) const {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1)
        for (int k = 0; k < units; ++k) {
            body(units - 1 - k);
        }
#else
        for (int k = 0; k < units; ++k) {
            body(units - 1 - k);
        }
#endif
    }
};

// Wraps a runner and records the largest team seen inside the bodies: a run
// with teams above one that never saw two threads would pass in vain.
template <class Run>
struct Observed {
    const Run& inner;
    std::atomic<int>* max_team;
    template <class Body>
    void operator()(int units, Body&& body) const {
        inner(units, [&](int unit) {
#if defined(_OPENMP)
            const int team = omp_get_num_threads();
#else
            const int team = 1;
#endif
            int seen = max_team->load(std::memory_order_relaxed);
            while (team > seen &&
                   !max_team->compare_exchange_weak(seen, team, std::memory_order_relaxed)) {
            }
            body(unit);
        });
    }
};

// --- Digest ------------------------------------------------------------------

struct Digest {
    std::uint64_t h = 1469598103934665603ULL;
    void add(std::uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (8 * b)) & 0xffU;
            h *= 1099511628211ULL;
        }
    }
    void add_ints(const std::vector<int>& v) {
        add(v.size());
        for (int x : v) {
            add(static_cast<std::uint32_t>(x));
        }
    }
};

// --- Cases -------------------------------------------------------------------

struct Case {
    std::string name;
    std::vector<int> raw;
    std::uint64_t budget = sr::kBudgetBytes;
};

int min_of(const std::vector<int>& v) { return *std::min_element(v.begin(), v.end()); }
int max_of(const std::vector<int>& v) { return *std::max_element(v.begin(), v.end()); }

// The guard of sparse_rank::fits(), restated independently.
bool expect_fit(const Case& c) {
    const long long n = static_cast<long long>(c.raw.size());
    if (n <= 0) {
        return false;
    }
    const long long range = static_cast<long long>(max_of(c.raw)) -
                            static_cast<long long>(min_of(c.raw)) + 1LL;
    const unsigned long long words = (static_cast<unsigned long long>(range) + 63ULL) / 64ULL;
    const unsigned long long bytes = 12ULL * words;
    return bytes <= c.budget && bytes <= 8ULL * static_cast<unsigned long long>(n);
}

// Random identifiers in [low, low + range), with both ends present.
std::vector<int> spread(int n, long long low, long long range, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<long long> pick(0, range - 1);
    std::vector<int> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(i)] = static_cast<int>(low + pick(gen));
    }
    if (n >= 1) v[0] = static_cast<int>(low);
    if (n >= 2) v[static_cast<std::size_t>(n) - 1U] = static_cast<int>(low + range - 1);
    return v;
}

std::vector<Case> make_cases(bool large) {
    std::vector<Case> cases;
    const int n = 20000;
    const long long two_n = 2LL * n;
    cases.push_back({"range_2n", spread(n, 0, two_n, 1)});
    cases.push_back({"range_2n_plus_1", spread(n, 0, two_n + 1, 2)});
    cases.push_back({"range_8.7n", spread(n, 1000, 174000, 3)});
    cases.push_back({"range_64n_guard_fails", spread(n, 0, 64LL * n, 4)});
    // Guard edge: n = 3k, R = 128k gives 12 W = 8 n exactly; R + 64 one word more.
    {
        const int k = 7000;
        cases.push_back({"guard_edge_fits", spread(3 * k, 0, 128LL * k, 5)});
        cases.push_back({"guard_edge_plus_one_word", spread(3 * k, 0, 128LL * k + 64, 6)});
    }
    // Budget edge, by parameter: 12 W equal to the budget, and one byte below.
    {
        Case c{"budget_edge_fits", spread(n, 0, 64LL * 1000, 7)};
        c.budget = 12ULL * 1000ULL;
        cases.push_back(c);
        Case d{"budget_edge_one_byte_short", c.raw};
        d.budget = 12ULL * 1000ULL - 1ULL;
        cases.push_back(d);
    }
    // Occupancy near the 90 % rule of build_indexer's dense branch (a compact
    // range, so the builders keep their dense branch; the primitive is checked
    // on it anyway).
    {
        std::vector<int> v = spread(n, 0, static_cast<long long>(n / 0.9), 8);
        cases.push_back({"occupancy_90", v});
        cases.push_back({"occupancy_89", spread(n, 0, static_cast<long long>(n / 0.89), 9)});
    }
    // One level with 90 % of the rows.
    {
        std::vector<int> v = spread(n, 0, 50LL * n / 3, 10);
        for (int i = 0; i < n; ++i) {
            if (i % 10 != 0) v[static_cast<std::size_t>(i)] = 123457;
        }
        cases.push_back({"one_level_90pct", v});
    }
    // Sorted, reverse-sorted, plateaus.
    {
        std::vector<int> v = spread(n, 5, 10LL * n, 11);
        std::sort(v.begin(), v.end());
        cases.push_back({"sorted", v});
        std::vector<int> r(v.rbegin(), v.rend());
        cases.push_back({"reverse_sorted", r});
        std::vector<int> p(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) p[static_cast<std::size_t>(i)] = (i / 37) * 1009;
        cases.push_back({"plateaus", p});
        std::vector<int> q = p;
        q[static_cast<std::size_t>(n) - 1U] = 0;  // the last row falls
        cases.push_back({"plateaus_last_row_falls", q});
    }
    // Negative identifiers, INT_MIN and INT_MAX.
    cases.push_back({"negative_low", spread(n, -1000000, 9LL * n, 12)});
    cases.push_back({"int_min_small_range",
                     spread(n, std::numeric_limits<int>::min(), 8LL * n, 13)});
    cases.push_back({"int_max_small_range",
                     spread(n, static_cast<long long>(std::numeric_limits<int>::max()) - 8LL * n + 1,
                            8LL * n, 14)});
    {
        std::vector<int> v = spread(n, 0, 9LL * n, 15);
        v[3] = std::numeric_limits<int>::min();
        v[4] = std::numeric_limits<int>::max();
        cases.push_back({"int_min_and_int_max", v});
    }
    // n = 1 and 2; a single negative level with n >= 2; one level per row.
    cases.push_back({"n1_negative", {-7}});
    cases.push_back({"n2_spread", {0, 1000}});
    cases.push_back({"single_negative_level_n4", {-3, -3, -3, -3}});
    {
        std::vector<int> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i * 5 + 17;
        std::mt19937_64 gen(16);
        std::shuffle(v.begin(), v.end(), gen);
        cases.push_back({"one_level_per_row", v});
    }
    if (large) {
        const int big = 25000001;
        cases.push_back({"large_range_50M", spread(big, 0, 50000000LL, 17)});
        cases.push_back({"large_range_50M_plus_1", spread(big, 0, 50000001LL, 18)});
    }
    return cases;
}

template <class Run>
void run_case(const Case& c, const Run& run, const std::string& tag, Digest& digest,
              int* bitmap_cases) {
    const int n = static_cast<int>(c.raw.size());
    const int lo = min_of(c.raw);
    const int hi = max_of(c.raw);
    const bool expected = expect_fit(c);
    sr::Index index;
    const bool built = sr::build_index(c.raw.data(), n, lo, hi, run, index, c.budget);
    check(built == expected, c.name + " " + tag + ": path (index built=" +
                                 std::to_string(built) + ", expected " +
                                 std::to_string(expected) + ")");
    digest.add(built ? 1U : 0U);
    if (!built) {
        check(index.bits == nullptr && index.prefix == nullptr,
              c.name + " " + tag + ": nothing allocated without the index");
        return;
    }
    ++*bitmap_cases;
    const Ascending ref = ref_ascending(c.raw);
    check(index.levels == ref.groups, c.name + " " + tag + ": levels");
    std::vector<int> ranks(static_cast<std::size_t>(n), -1);
    sr::map_ranks(c.raw.data(), n, index, ranks.data(), run);
    check(ranks == ref.ids, c.name + " " + tag + ": ascending ranks (build_indexer)");
    // rank() of a value against the reference, at a few rows.
    for (int i = 0; i < n; i += std::max(1, n / 17)) {
        check(index.rank(c.raw[static_cast<std::size_t>(i)]) == ref.ids[static_cast<std::size_t>(i)],
              c.name + " " + tag + ": rank() at row " + std::to_string(i));
    }
    // Contiguity and group_ptr (finalize_contiguous_groups()).
    const bool rising = sr::non_decreasing(ranks.data(), n, run);
    check(rising == ref.contiguous, c.name + " " + tag + ": contiguity");
    if (rising) {
        std::vector<int> ptr(static_cast<std::size_t>(index.levels) + 1U, n);
        sr::fill_group_ptr(ranks.data(), n, index.levels, ptr.data(), run);
        check(ptr == ref.group_ptr, c.name + " " + tag + ": group_ptr");
        digest.add_ints(ptr);
    }
    // Nested test against the hash (both outcomes, and a conflict on the last row).
    {
        std::vector<int> nested_clusters(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            nested_clusters[static_cast<std::size_t>(i)] = ref.ids[static_cast<std::size_t>(i)] / 3;
        }
        const bool want_true = ref_nested(c.raw, nested_clusters);
        const bool got_true = sr::nested_by_rank(c.raw.data(), nested_clusters.data(), n, index, run);
        check(want_true && got_true == want_true, c.name + " " + tag + ": nested (true)");
        std::vector<int> conflict = nested_clusters;
        if (n >= 2) {
            // The last row's level also occurs earlier, unless it is unique.
            conflict[static_cast<std::size_t>(n) - 1U] += 1000003;
        }
        const bool want = ref_nested(c.raw, conflict);
        const bool got = sr::nested_by_rank(c.raw.data(), conflict.data(), n, index, run);
        check(got == want, c.name + " " + tag + ": nested (conflict on the last row)");
        digest.add(got_true ? 1U : 0U);
        digest.add(got ? 1U : 0U);
    }
    // First appearance (build_indexer_lite, and the Rows lookup row by row).
    int ref_groups = 0;
    const std::vector<int> first = ref_first_appearance(c.raw, &ref_groups);
    const int levels = index.levels;
    index.clear();
    check(levels == ref_groups, c.name + " " + tag + ": first-appearance levels");
    sr::first_appearance_in_place(ranks.data(), n, levels, run);
    check(ranks == first, c.name + " " + tag + ": first appearance (lite, Rows)");
    digest.add(static_cast<std::uint64_t>(levels));
    digest.add_ints(ranks);
    digest.add_ints(ref.ids);
}

}  // namespace

int main(int argc, char** argv) {
    const bool large = argc > 1 && std::strcmp(argv[1], "--large") == 0;
    const std::vector<Case> cases = make_cases(large);
    const int teams[] = {1, 2, 3, 8, 16, 24, 48};
    std::uint64_t reference_digest = 0;
    bool have_reference = false;
    int bitmap_cases = 0;
    int largest_team_seen = 1;
    for (const int team : teams) {
        for (int runner = 0; runner < 2; ++runner) {
            Digest digest;
            std::atomic<int> max_team{1};
            const std::string tag =
                std::string(runner == 0 ? "static" : "dynamic-backwards") + " team " +
                std::to_string(team);
            for (const Case& c : cases) {
                if (runner == 0) {
                    const sr::OmpRun inner{team, nullptr};
                    const Observed<sr::OmpRun> run{inner, &max_team};
                    run_case(c, run, tag, digest, &bitmap_cases);
                } else {
                    const ReverseDynamicRun inner{team};
                    const Observed<ReverseDynamicRun> run{inner, &max_team};
                    run_case(c, run, tag, digest, &bitmap_cases);
                }
            }
            if (!have_reference) {
                reference_digest = digest.h;
                have_reference = true;
            }
            check(digest.h == reference_digest, "digest independent of the team: " + tag);
            // Teams above one must have run with at least two threads somewhere
            // (the runtime may grant fewer than asked, never just one).
            check(team == 1 || max_team.load() >= 2,
                  "a team of " + std::to_string(team) + " ran serially: " + tag);
            largest_team_seen = std::max(largest_team_seen, max_team.load());
        }
    }
    check(bitmap_cases > 0, "at least one case built the index");
    // The switch and the guard, as plain functions.
    check(sr::words_for(std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) ==
              (std::uint64_t{1} << 26),
          "words of the full int32 range");
    check(!sr::fits(std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), 1 << 30),
          "the full int32 range is above the budget");
    check(!sr::fits(0, 10, 0), "no index for n = 0");
    for (std::uint64_t x : {0ULL, 1ULL, 0x8000000000000000ULL, 0xffffffffffffffffULL,
                            0x0123456789abcdefULL}) {
        int slow = 0;
        for (int b = 0; b < 64; ++b) slow += static_cast<int>((x >> b) & 1ULL);
        check(sr::popcount64(x) == slow, "popcount64");
    }
    std::printf("%s checks=%d failures=%d cases=%zu bitmap_case_runs=%d largest_team=%d "
                "digest=%016llx%s\n",
                failures == 0 ? "SPARSE_RANK_UNIT_PASS" : "SPARSE_RANK_UNIT_FAIL", checks,
                failures, cases.size(), bitmap_cases, largest_team_seen,
                static_cast<unsigned long long>(reference_digest), large ? " large" : "");
    return failures == 0 ? 0 : 1;
}
