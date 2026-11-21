// Measurement of the sparse ranking alone (phase 1 of the Stata-path plan,
// design note Verifications/perf_stata_path_20261114/fase1/DESENHO_FASE1.md,
// section 6). Not part of the package.
//
// One process measures one arm, one view, one shape and one team once, so the
// process's peak resident set (ru_maxrss) belongs to that arm; the runner
// (tests/sparse_rank_bench.sh) interleaves arms and repeats. Synthetic data
// only, generated from fixed seeds before the clock starts.
//
//   shape  F      firstreg-like: n = 56 M, 800 k levels in [0, 5e8), rows in
//                 workers of 6 to 10 with a home level and 15 % moves
//          Frand  the same levels in random row order
//          C      compact with 60 M levels: n = 64 M, every value 0..60 M-1
//                 present plus 4 M repeats, random order (range <= 2n, above
//                 the 50 M cap of the builders)
//          E      budget edge: n = 56 M, 800 k levels in [0, 1.43e9) (the index
//                 near 256 MiB)
//          Efull  800 k levels over the whole int32 range (above the budget:
//                 the new arm falls back to today's branch)
//          S      skewed: F with one level on 50 % of the rows and a Zipf tail
//   view   ascending  build_indexer(): ranks, contiguity and group_ptr
//          codes      compact_codes(): ranks
//          lite       build_indexer_lite() and the Rows lookup: first appearance
//          nested     fe_nested_within_cluster() against a cluster vector
//   arm    old        today's serial branch (sort of pairs, or hash map)
//          new        the bitmap index on the team, or today's branch when the
//                     index does not fit (as the builders do)
//
// Output, one CSV line: shape,view,arm,team,observed_team,n,range,levels,path,ms,
// maxrss_kb,digest (the digest must match between the arms; observed_team is the
// largest team seen inside the new arm's bodies, 1 for the old arm).
//
// Build, from the repository root:
//   g++ -std=gnu++17 -O3 -march=native -fopenmp -DHDFE_USE_OPENMP -Iinclude -Isrc
//       tests/sparse_rank_bench.cpp -o sparse_rank_bench
// Usage: sparse_rank_bench <shape> <view> <arm> <team>

#include "sparse_rank.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sr = hdfe::detail::sparse_rank;

namespace {

// Distinct values in [low, low + range), both ends included, sorted.
std::vector<int> distinct_levels(int count, long long low, long long range, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<long long> pick(0, range - 1);
    std::vector<long long> v;
    v.reserve(static_cast<std::size_t>(count) + 2U);
    v.push_back(0);
    v.push_back(range - 1);
    while (static_cast<int>(v.size()) < count) {
        v.push_back(pick(gen));
        if (static_cast<int>(v.size()) == count) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    }
    std::vector<int> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<int>(low + v[i]);
    return out;
}

// Rows in workers of 6 to 10 with a home level and 15 % moves; or random order.
std::vector<int> firm_rows(int n, const std::vector<int>& levels, bool random_order,
                           std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    const int g = static_cast<int>(levels.size());
    std::uniform_int_distribution<int> any(0, g - 1);
    std::uniform_int_distribution<int> spell(6, 10);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<int> rows(static_cast<std::size_t>(n));
    int i = 0;
    while (i < n) {
        const int home = any(gen);
        const int len = random_order ? 1 : spell(gen);
        for (int k = 0; k < len && i < n; ++k, ++i) {
            const bool move = random_order || unit(gen) < 0.15;
            rows[static_cast<std::size_t>(i)] = levels[static_cast<std::size_t>(move ? any(gen) : home)];
        }
    }
    // Every level present at least once.
    for (int k = 0; k < g && k < n; ++k) {
        rows[static_cast<std::size_t>((static_cast<long long>(k) * 7919LL) % n)] =
            levels[static_cast<std::size_t>(k)];
    }
    return rows;
}

std::vector<int> generate_shape(const std::string& shape);

// With SPARSE_RANK_BENCH_DATA_DIR set, a shape is generated once and read back
// by later processes from <dir>/<shape>.bin (same bytes, a shorter process).
std::vector<int> make_shape(const std::string& shape) {
    const char* dir = std::getenv("SPARSE_RANK_BENCH_DATA_DIR");
    if (dir == nullptr || *dir == '\0') {
        return generate_shape(shape);
    }
    const std::string path = std::string(dir) + "/" + shape + ".bin";
    if (std::FILE* in = std::fopen(path.c_str(), "rb")) {
        std::int64_t n = 0;
        if (std::fread(&n, sizeof(n), 1, in) == 1 && n > 0) {
            std::vector<int> rows(static_cast<std::size_t>(n));
            const std::size_t got = std::fread(rows.data(), sizeof(int), rows.size(), in);
            std::fclose(in);
            if (got == rows.size()) {
                return rows;
            }
        } else {
            std::fclose(in);
        }
        std::fprintf(stderr, "unreadable %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<int> rows = generate_shape(shape);
    const std::string tmp = path + ".tmp";
    if (std::FILE* outf = std::fopen(tmp.c_str(), "wb")) {
        const std::int64_t n = static_cast<std::int64_t>(rows.size());
        const bool ok = std::fwrite(&n, sizeof(n), 1, outf) == 1 &&
                        std::fwrite(rows.data(), sizeof(int), rows.size(), outf) == rows.size();
        std::fclose(outf);
        if (ok) {
            std::rename(tmp.c_str(), path.c_str());
        }
    }
    return rows;
}

std::vector<int> generate_shape(const std::string& shape) {
    if (shape == "F" || shape == "Frand" || shape == "S") {
        const int n = 56000000;
        const std::vector<int> levels = distinct_levels(800000, 0, 500000000LL, 11);
        std::vector<int> rows = firm_rows(n, levels, shape == "Frand", 12);
        if (shape == "S") {
            std::mt19937_64 gen(13);
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            const double g = static_cast<double>(levels.size());
            for (int i = 0; i < n; ++i) {
                const double u = unit(gen);
                if (u < 0.5) {
                    rows[static_cast<std::size_t>(i)] = levels[levels.size() / 2];
                } else {
                    // Zipf-like tail: level index ~ g^(2(u - 0.5)) - 1.
                    const double x = std::pow(g, 2.0 * (u - 0.5)) - 1.0;
                    const std::size_t k = std::min(levels.size() - 1U, static_cast<std::size_t>(x));
                    rows[static_cast<std::size_t>(i)] = levels[k];
                }
            }
            rows[0] = levels.front();
            rows[1] = levels.back();
        }
        return rows;
    }
    if (shape == "C") {
        const int levels = 60000000;
        const int n = 64000000;
        std::vector<int> rows(static_cast<std::size_t>(n));
        for (int i = 0; i < levels; ++i) rows[static_cast<std::size_t>(i)] = i;
        std::mt19937_64 gen(14);
        std::uniform_int_distribution<int> any(0, levels - 1);
        for (int i = levels; i < n; ++i) rows[static_cast<std::size_t>(i)] = any(gen);
        std::shuffle(rows.begin(), rows.end(), gen);
        return rows;
    }
    if (shape == "E" || shape == "Efull") {
        const int n = 56000000;
        const long long low = shape == "E" ? 0LL : static_cast<long long>(std::numeric_limits<int>::min());
        const long long range = shape == "E" ? 1430000000LL : (1LL << 32);
        const std::vector<int> levels = distinct_levels(800000, low, range, 15);
        return firm_rows(n, levels, false, 16);
    }
    std::fprintf(stderr, "unknown shape %s\n", shape.c_str());
    std::exit(2);
}

// The runner of the new arm, recording the largest team seen inside the bodies
// (a measurement at 16 threads that ran serially would be worthless).
struct ObservedRun {
    sr::OmpRun inner;
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

std::uint64_t digest_of(const std::vector<int>& v, std::uint64_t extra) {
    std::uint64_t h = 1469598103934665603ULL ^ extra;
    for (int x : v) {
        h ^= static_cast<std::uint32_t>(x);
        h *= 1099511628211ULL;
    }
    return h;
}

// --- Today's branches ---------------------------------------------------------

int old_ascending(const std::vector<int>& raw, std::vector<int>& ids) {
    const std::size_t n = raw.size();
    std::vector<std::pair<int, int>> keyed(n);
    for (std::size_t i = 0; i < n; ++i) keyed[i] = {raw[i], static_cast<int>(i)};
    std::sort(keyed.begin(), keyed.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first < b.first;
              });
    int next = -1;
    int prev = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (next < 0 || keyed[k].first != prev) {
            ++next;
            prev = keyed[k].first;
        }
        ids[static_cast<std::size_t>(keyed[k].second)] = next;
    }
    return next + 1;
}

int old_first_appearance(const std::vector<int>& raw, std::vector<int>& ids) {
    std::unordered_map<int, int> mapping;
    mapping.reserve(std::min<std::size_t>(raw.size(), 5000000U));
    int next = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        auto it = mapping.find(raw[i]);
        if (it == mapping.end()) {
            ids[i] = next;
            mapping.emplace(raw[i], next++);
        } else {
            ids[i] = it->second;
        }
    }
    return next;
}

bool old_nested(const std::vector<int>& fe, const std::vector<int>& clusters) {
    std::unordered_map<int, int> mapping;
    mapping.reserve(fe.size());
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <shape> <view> <arm> <team>\n", argv[0]);
        return 2;
    }
    const std::string shape = argv[1];
    const std::string view = argv[2];
    const std::string arm = argv[3];
    const int team = std::max(1, std::atoi(argv[4]));
    const std::vector<int> raw = make_shape(shape);
    const int n = static_cast<int>(raw.size());
    const int lo = *std::min_element(raw.begin(), raw.end());
    const int hi = *std::max_element(raw.begin(), raw.end());
    std::vector<int> clusters;
    if (view == "nested") {
        // Not nested: a cluster per block of 9 rows, so most levels meet several.
        clusters.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) clusters[static_cast<std::size_t>(i)] = i / 9;
    }
    std::vector<int> out(static_cast<std::size_t>(n), 0);
    std::vector<int> group_ptr;
    int levels = 0;
    bool flag = false;
    std::string path = "serial";
    std::atomic<int> observed_team{1};
    const ObservedRun run{sr::OmpRun{team, nullptr}, &observed_team};

    const auto t0 = std::chrono::steady_clock::now();
    if (arm == "new" && sr::fits(lo, hi, n)) {
        path = "bitmap";
        sr::Index index;
        sr::build_index(raw.data(), n, lo, hi, run, index);
        levels = index.levels;
        if (view == "nested") {
            flag = sr::nested_by_rank(raw.data(), clusters.data(), n, index, run);
        } else {
            sr::map_ranks(raw.data(), n, index, out.data(), run);
            index.clear();
            if (view == "lite") {
                sr::first_appearance_in_place(out.data(), n, levels, run);
            } else if (view == "ascending") {
                flag = sr::non_decreasing(out.data(), n, run);
                if (flag) {
                    group_ptr.assign(static_cast<std::size_t>(levels) + 1U, n);
                    sr::fill_group_ptr(out.data(), n, levels, group_ptr.data(), run);
                }
            }
        }
    } else {
        if (view == "nested") {
            flag = old_nested(raw, clusters);
        } else if (view == "lite") {
            levels = old_first_appearance(raw, out);
        } else {
            levels = old_ascending(raw, out);
            if (view == "ascending") {
                flag = std::is_sorted(out.begin(), out.end());
            }
        }
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    const std::uint64_t digest =
        view == "nested" ? (flag ? 1ULL : 0ULL) : digest_of(out, static_cast<std::uint64_t>(levels));
    std::printf("%s,%s,%s,%d,%d,%d,%lld,%d,%s,%.3f,%ld,%016llx\n", shape.c_str(), view.c_str(),
                arm.c_str(), team, observed_team.load(), n,
                static_cast<long long>(hi) - static_cast<long long>(lo) + 1LL, levels,
                path.c_str(), ms, static_cast<long>(usage.ru_maxrss),
                static_cast<unsigned long long>(digest));
    return 0;
}
