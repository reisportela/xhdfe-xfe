// Direct contract of phase 2 of the Stata-path plan (design note
// Verifications/perf_stata_path_20261114/fase2/DESENHO_FASE2.md, section 6): the
// certificate of the reduced-system tier takes the tier's codes over for every
// effect whose indexer would number the identifiers by their ascending compact
// ranks, and derives the rest of the indexer from them.
//
// The real rule and derivation of src/sparse_rank.hpp (indexer_numbers_by_rank()
// and contiguous_group_ptr()) against verbatim copies of today's builder,
// build_indexer() with finalize_contiguous_groups() (src/fe_absorption.cpp; the
// team builder gives the same indexer), and of the tier's codes
// (sorted_compact_codes(), src/reduced_schur_pcg.cpp). Every field of the indexer
// is compared; the rule must say "shared" exactly when the builder's num_groups
// equals num_levels_present, so an effect with holes is never shared. Generated
// identifiers: occupancy built at exactly 89, 90 and 100 % (and 90/91 of a range
// of 101), holes inside and next to both ends, sparse ranges, sorted and unsorted
// rows, one level, n = 1, the 2n boundary (2n and 2n + 1), negative identifiers,
// INT_MIN and INT_MAX; with --large also compact ranges at and above the 50
// million cap (n = 25 000 001). Teams 1, 2, 3, 8, 16, 24 and 48, with a static
// runner and a dynamic one that visits the units backwards. The builders inside
// the core are compared with the real build_indexer_team() by the contracts
// (tests/shared_fe_coding_contract.py, XHDFE_SPARSE_RANK_VERIFY=1).
//
// Build (no Eigen needed), from the repository root:
//   g++ -std=gnu++17 -O2 -fopenmp -DHDFE_USE_OPENMP -Iinclude -Isrc
//       tests/shared_fe_coding_unit.cpp -o shared_fe_coding_unit
// Usage: shared_fe_coding_unit [--large]

#include "sparse_rank.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
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
        if (failures <= 40) std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

// --- Verbatim copies (src/fe_absorption.cpp, src/reduced_schur_pcg.cpp) -------

struct FeIndexer {
    std::vector<int> group_ids;
    std::vector<int> group_ptr;
    int num_groups = 0;
    int num_levels_present = 0;
    bool groups_contiguous = false;
};

void finalize_contiguous_groups(FeIndexer& idx) {
    idx.groups_contiguous = false;
    idx.group_ptr.clear();
    const int n = static_cast<int>(idx.group_ids.size());
    if (n <= 0 || idx.num_groups <= 0) return;
    int prev = idx.group_ids[0];
    if (prev < 0 || prev >= idx.num_groups) return;
    for (int i = 1; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        if (g < prev) return;
        prev = g;
    }
    idx.groups_contiguous = true;
    idx.group_ptr.assign(static_cast<std::size_t>(idx.num_groups) + 1U, n);
    idx.group_ptr[0] = 0;
    const int first_group = idx.group_ids[0];
    for (int g = 1; g <= first_group; ++g) idx.group_ptr[static_cast<std::size_t>(g)] = 0;
    int last_group = first_group;
    for (int i = 1; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        if (g != last_group) {
            for (int fill = last_group + 1; fill <= g; ++fill) idx.group_ptr[static_cast<std::size_t>(fill)] = i;
            last_group = g;
        }
    }
    for (int fill = last_group + 1; fill <= idx.num_groups; ++fill) idx.group_ptr[static_cast<std::size_t>(fill)] = n;
}

FeIndexer build_indexer(const std::vector<int>& raw_ids) {
    FeIndexer idx;
    const int size = static_cast<int>(raw_ids.size());
    idx.group_ids.resize(raw_ids.size());
    if (size == 0) return idx;
    int min_id = raw_ids[0], max_id = raw_ids[0];
    for (int i = 1; i < size; ++i) {
        const int v = raw_ids[static_cast<std::size_t>(i)];
        if (v < min_id) min_id = v;
        else if (v > max_id) max_id = v;
    }
    const long long range = static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    constexpr long long kDenseMinDensityNumerator = 9;
    constexpr long long kDenseMinDensityDenominator = 10;
    if (range > 0 && range <= kDenseRangeCap && range <= static_cast<long long>(size) * 2LL) {
        const int groups = static_cast<int>(range);
        idx.num_groups = groups;
        std::vector<uint8_t> seen(static_cast<std::size_t>(groups), 0);
        int unique = 0;
        for (int i = 0; i < size; ++i) {
            const int mapped = raw_ids[static_cast<std::size_t>(i)] - min_id;
            idx.group_ids[static_cast<std::size_t>(i)] = mapped;
            uint8_t& flag = seen[static_cast<std::size_t>(mapped)];
            if (!flag) { flag = 1; ++unique; }
        }
        idx.num_levels_present = unique;
        if (static_cast<long long>(unique) * kDenseMinDensityDenominator < range * kDenseMinDensityNumerator) {
            std::vector<int> remap(static_cast<std::size_t>(groups), -1);
            int next_id = 0;
            for (int g = 0; g < groups; ++g)
                if (seen[static_cast<std::size_t>(g)]) remap[static_cast<std::size_t>(g)] = next_id++;
            for (int i = 0; i < size; ++i)
                idx.group_ids[static_cast<std::size_t>(i)] =
                    remap[static_cast<std::size_t>(idx.group_ids[static_cast<std::size_t>(i)])];
            idx.num_groups = next_id;
            idx.num_levels_present = next_id;
        }
        finalize_contiguous_groups(idx);
        return idx;
    }
    std::vector<std::pair<int, int>> keyed(raw_ids.size());
    for (std::size_t i = 0; i < raw_ids.size(); ++i) keyed[i] = {raw_ids[i], static_cast<int>(i)};
    std::sort(keyed.begin(), keyed.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });
    int next_id = -1, prev_key = 0;
    for (std::size_t k = 0; k < keyed.size(); ++k) {
        if (next_id < 0 || keyed[k].first != prev_key) { ++next_id; prev_key = keyed[k].first; }
        idx.group_ids[static_cast<std::size_t>(keyed[k].second)] = next_id;
    }
    idx.num_groups = next_id + 1;
    idx.num_levels_present = next_id + 1;
    finalize_contiguous_groups(idx);
    return idx;
}

int sorted_compact_codes(const int* src, int n, int* codes) {
    std::vector<std::pair<int, int>> keyed(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) keyed[static_cast<std::size_t>(i)] = {src[i], i};
    std::sort(keyed.begin(), keyed.end());
    int next = -1, previous = 0;
    for (std::size_t i = 0; i < keyed.size(); ++i) {
        if (next < 0 || keyed[i].first != previous) { ++next; previous = keyed[i].first; }
        codes[static_cast<std::size_t>(keyed[i].second)] = next;
    }
    return next + 1;
}

// --- Runners -------------------------------------------------------------------

struct ReverseDynamicRun {
    int threads = 1;
    template <class Body>
    void operator()(int units, Body&& body) const {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1)
#endif
        for (int k = 0; k < units; ++k) body(units - 1 - k);
    }
};

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
            while (team > seen && !max_team->compare_exchange_weak(seen, team, std::memory_order_relaxed)) {
            }
            body(unit);
        });
    }
};

// --- Cases -----------------------------------------------------------------------

struct Case {
    std::string name;
    std::vector<int> raw;
    int expect_shared;  // 1 shared, 0 not shared (holes kept), -1 unstated
};

// `present` distinct values of [low, low + range), both ends included, `holes`
// chooses where the missing ones are (interior at random, or next to the ends
// first); every value gets at least two rows; rows sorted or shuffled.
std::vector<int> occupancy_ids(int low, int range, int present, int rows, bool holes_at_ends,
                               bool sorted, std::mt19937_64& rng) {
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(range));
    for (int v = 0; v < range; ++v) values.push_back(low + v);
    const int missing = range - present;
    std::vector<int> drop;
    if (holes_at_ends) {
        for (int k = 0; k < missing; ++k) drop.push_back(k % 2 == 0 ? 1 + k / 2 : range - 2 - k / 2);
    } else {
        std::vector<int> interior(static_cast<std::size_t>(range - 2));
        std::iota(interior.begin(), interior.end(), 1);
        std::shuffle(interior.begin(), interior.end(), rng);
        drop.assign(interior.begin(), interior.begin() + missing);
    }
    std::vector<char> gone(static_cast<std::size_t>(range), 0);
    for (int d : drop) gone[static_cast<std::size_t>(d)] = 1;
    std::vector<int> kept;
    for (int v = 0; v < range; ++v)
        if (!gone[static_cast<std::size_t>(v)]) kept.push_back(values[static_cast<std::size_t>(v)]);
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(rows));
    for (int k = 0; k < rows; ++k) ids.push_back(kept[static_cast<std::size_t>(k) % kept.size()]);
    if (sorted) std::sort(ids.begin(), ids.end());
    else std::shuffle(ids.begin(), ids.end(), rng);
    return ids;
}

std::vector<Case> make_cases(bool large) {
    std::mt19937_64 rng(20261114);
    std::vector<Case> cases;
    for (int range : {100, 1000}) {
        for (int pct : {89, 90, 100}) {
            const int present = range * pct / 100;
            for (int ends = 0; ends < 2; ++ends) {
                if (present == range && ends) continue;
                for (int sorted = 0; sorted < 2; ++sorted) {
                    cases.push_back({"occ" + std::to_string(pct) + "_R" + std::to_string(range) +
                                         (ends ? "_ends" : "_inside") + (sorted ? "_sorted" : "_unsorted"),
                                     occupancy_ids(1000, range, present, 300000, ends != 0, sorted != 0, rng),
                                     pct == 90 ? 0 : 1});
                }
            }
        }
    }
    // 90/101 compacts (900 < 909), 91/101 keeps its holes (910 >= 909).
    cases.push_back({"occ90of101", occupancy_ids(7, 101, 90, 5000, false, false, rng), 1});
    cases.push_back({"occ91of101", occupancy_ids(7, 101, 91, 5000, false, false, rng), 0});
    cases.push_back({"occ91of101_sorted", occupancy_ids(7, 101, 91, 5000, false, true, rng), 0});
    // Sparse ranges (above 2n), sorted and not.
    for (int sorted = 0; sorted < 2; ++sorted) {
        std::vector<int> ids(400000);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>((i % 40000) * 87 + 3);
        if (sorted) std::sort(ids.begin(), ids.end());
        else std::shuffle(ids.begin(), ids.end(), rng);
        cases.push_back({std::string("sparse_") + (sorted ? "sorted" : "unsorted"), ids, 1});
    }
    cases.push_back({"one_level", std::vector<int>(250000, 42), 1});
    cases.push_back({"n1", std::vector<int>(1, 5), 1});
    cases.push_back({"n2_same", std::vector<int>(2, -9), 1});
    // The 2n boundary: range 2n (dense branch, half or less present) and 2n + 1 (sparse).
    for (int extra = 0; extra < 2; ++extra) {
        const int n = 200000;
        std::vector<int> ids(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) ids[static_cast<std::size_t>(i)] = (i % (n / 2)) * 4;  // 0 .. 2n - 4
        ids[0] = 0;
        ids[1] = 2 * n - 1 + extra;  // range 2n or 2n + 1
        std::shuffle(ids.begin(), ids.end(), rng);
        cases.push_back({std::string("boundary_2n") + (extra ? "_plus1" : ""), ids, 1});
    }
    // Negative identifiers, full and with holes; INT_MIN and INT_MAX.
    cases.push_back({"negative_full", occupancy_ids(-500, 1000, 1000, 60000, false, false, rng), 1});
    cases.push_back({"negative_holes95", occupancy_ids(-500, 1000, 950, 60000, false, false, rng), 0});
    {
        std::vector<int> ids(100000);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i % 997) - 400;
        ids[10] = INT_MIN;
        ids[20] = INT_MAX;
        ids[30] = INT_MIN;
        ids[40] = INT_MAX;
        cases.push_back({"int_extremes", ids, 1});
    }
    if (large) {
        // 25 000 001 rows: a compact range at the 50 million cap (dense branch, half
        // present) and one above it (the builder's sparse branch; the tier's dense
        // remap has no cap), sorted and not.
        const int n = 25000001;
        for (int extra = 0; extra < 2; ++extra) {
            for (int sorted = 0; sorted < 2; ++sorted) {
                std::vector<int> ids(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) ids[static_cast<std::size_t>(i)] = 2 * i;  // 0 .. 50 000 000
                ids[static_cast<std::size_t>(n - 1)] = extra ? 50000000 : 49999999;  // range 50 000 001 or 50 000 000
                if (sorted) std::sort(ids.begin(), ids.end());
                else std::shuffle(ids.begin(), ids.end(), rng);
                cases.push_back({std::string(extra ? "cap_plus1" : "cap") + (sorted ? "_sorted" : "_unsorted"), ids,
                                 1});
            }
        }
    }
    return cases;
}

}  // namespace

int main(int argc, char** argv) {
    const bool large = argc > 1 && std::strcmp(argv[1], "--large") == 0;
    const std::vector<Case> cases = make_cases(large);
    const int teams[] = {1, 2, 3, 8, 16, 24, 48};
    int shared_cases = 0, kept_cases = 0, contiguous_shared = 0, largest_team = 1;
    for (const Case& c : cases) {
        const int n = static_cast<int>(c.raw.size());
        const FeIndexer ref = build_indexer(c.raw);
        std::vector<int> codes(static_cast<std::size_t>(n));
        const int levels = sorted_compact_codes(c.raw.data(), n, codes.data());
        const int min_id = *std::min_element(c.raw.begin(), c.raw.end());
        const int max_id = *std::max_element(c.raw.begin(), c.raw.end());
        const bool shared = sr::indexer_numbers_by_rank(n, min_id, max_id, levels);
        const bool same_numbering = ref.num_groups == ref.num_levels_present;
        check(shared == same_numbering, c.name + ": rule against the builder");
        if (c.expect_shared >= 0) check(shared == (c.expect_shared == 1), c.name + ": expected sharing");
        check(ref.num_levels_present == levels, c.name + ": levels");
        if (!shared) {
            ++kept_cases;
            check(ref.group_ids != codes, c.name + ": a kept-holes numbering equal to the codes");
            continue;
        }
        ++shared_cases;
        check(ref.group_ids == codes, c.name + ": group_ids against the codes");
        check(ref.num_groups == levels, c.name + ": num_groups");
        if (ref.groups_contiguous) ++contiguous_shared;
        for (int runner = 0; runner < 2; ++runner) {
            for (int team : teams) {
                std::atomic<int> max_team{1};
                std::vector<int> group_ptr{-7};  // must be replaced
                bool contiguous;
                if (runner == 0) {
                    const sr::OmpRun inner{team, nullptr};
                    const Observed<sr::OmpRun> run{inner, &max_team};
                    contiguous = sr::contiguous_group_ptr(codes.data(), n, levels, group_ptr, run);
                } else {
                    const ReverseDynamicRun inner{team};
                    const Observed<ReverseDynamicRun> run{inner, &max_team};
                    contiguous = sr::contiguous_group_ptr(codes.data(), n, levels, group_ptr, run);
                }
                largest_team = std::max(largest_team, max_team.load());
                const std::string what = c.name + (runner == 0 ? " static" : " dynamic-backwards") +
                                         " team " + std::to_string(team);
                check(contiguous == ref.groups_contiguous, what + ": groups_contiguous");
                check(group_ptr == ref.group_ptr, what + ": group_ptr");
            }
        }
    }
    check(shared_cases > 0 && kept_cases > 0 && contiguous_shared > 0, "coverage: shared, kept and contiguous cases");
    check(largest_team >= 2, "a team of two or more was observed");
    std::printf("%s checks=%d failures=%d cases=%zu shared=%d kept_holes=%d contiguous_shared=%d largest_team=%d%s\n",
                failures == 0 ? "PASS" : "FAIL", checks, failures, cases.size(), shared_cases, kept_cases,
                contiguous_shared, largest_team, large ? " (with --large)" : "");
    return failures == 0 ? 0 : 1;
}
