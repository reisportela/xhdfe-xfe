// Phase 3 of the Stata-path plan: the direct contract of the codes of a wide
// identifier column. Today's coding, a literal copy of read_categorical_numeric
// (stata/src/xhdfe_plugin.cpp) applied to the values in row order, is compared
// with the new one: the int32 pass-through test; `raw`, where the rows before
// the first wide value come back from int; pairs, the sort and the serial
// ranks. The new coding takes the pairs of the values themselves, the same
// sort, and xhdfe_wide_id::rank_sorted on 1 to 48 threads with 1 chunk up to
// more chunks than rows. A second sort orders equal values differently, which
// must not change a code. The code vectors must be exactly equal.
//
// Build with the flags of the plugin (stata/tools/build-plugin.sh), from the
// repository root:
//   g++ -std=gnu++17 -O3 -DNDEBUG -ffast-math -fno-finite-math-only -funroll-loops
//       -march=native -mtune=native -fopenmp -DHDFE_USE_OPENMP -pthread
//       tests/wide_id_rank_unit.cpp -o wide_id_rank_unit
#include "../stata/src/xhdfe_wide_id_rank.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(__GNUC__) && defined(_OPENMP) && __has_include(<parallel/algorithm>)
#include <parallel/algorithm>
#define WIDE_SORT(first, last, cmp) __gnu_parallel::sort((first), (last), (cmp))
#else
#define WIDE_SORT(first, last, cmp) std::sort((first), (last), (cmp))
#endif

namespace {

long long checks = 0;
long long failures = 0;
int largest_team = 0;  // the test fails if no region ever ran on more than one thread
std::uint64_t digest = 1469598103934665603ULL;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 20) {
            std::printf("FAIL %s\n", what.c_str());
        }
    }
}

// stata/src/xhdfe_plugin.cpp, exact_int32_id.
bool exact_int32_id(double value, int& out) {
    if (value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int candidate = static_cast<int>(value);
    if (static_cast<double>(candidate) != value) {
        return false;
    }
    out = candidate;
    return true;
}

// Today's read_categorical_numeric from the values in row order (the reads and
// their checks aside). False when every value is exact int32: today passes them
// through, and the new path never runs.
bool old_codes(const std::vector<double>& values, std::vector<int>& ids) {
    const int n = static_cast<int>(values.size());
    ids.assign(static_cast<std::size_t>(n), 0);
    std::vector<double> raw;
    bool all_int = true;
    for (int i = 0; i < n; ++i) {
        const double z = values[static_cast<std::size_t>(i)];
        if (all_int) {
            int id = 0;
            if (exact_int32_id(z, id)) {
                ids[static_cast<std::size_t>(i)] = id;
                continue;
            }
            all_int = false;
            raw.resize(static_cast<std::size_t>(n));
            for (int k = 0; k < i; ++k) {
                raw[static_cast<std::size_t>(k)] = static_cast<double>(ids[static_cast<std::size_t>(k)]);
            }
        }
        raw[static_cast<std::size_t>(i)] = z;
    }
    if (all_int) {
        return false;
    }
    std::vector<std::pair<double, int>> keyed;
    keyed.resize(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        keyed[static_cast<std::size_t>(j)] = std::pair<double, int>(raw[static_cast<std::size_t>(j)], j);
    }
    WIDE_SORT(keyed.begin(), keyed.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;
              });
    int next_id = -1;
    double prev = 0.0;
    for (const auto& kv : keyed) {
        const double value = kv.first;
        if (next_id < 0 || value != prev) {
            ++next_id;
            prev = value;
        }
        ids[static_cast<std::size_t>(kv.second)] = next_id;
    }
    return next_id >= 0;
}

// The new coding; `reverse_ties` orders equal values by descending row.
long long new_codes(const std::vector<double>& values, std::vector<int>& codes, long long chunks,
                    int threads, bool reverse_ties, int& team, int& workers) {
    const int n = static_cast<int>(values.size());
    std::vector<xhdfe_wide_id::Pair> pairs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        pairs[static_cast<std::size_t>(i)].value = values[static_cast<std::size_t>(i)];
        pairs[static_cast<std::size_t>(i)].row = i;
    }
    xhdfe_wide_id::Pair* const first = pairs.data();
    if (reverse_ties) {
        std::sort(first, first + n, [](const xhdfe_wide_id::Pair& a, const xhdfe_wide_id::Pair& b) {
            return a.value < b.value || (!(b.value < a.value) && a.row > b.row);
        });
    } else {
        WIDE_SORT(first, first + n, [](const xhdfe_wide_id::Pair& a, const xhdfe_wide_id::Pair& b) {
            return a.value < b.value;
        });
    }
    codes.assign(static_cast<std::size_t>(n), -7);
    const xhdfe_wide_id::RankResult r =
        xhdfe_wide_id::rank_sorted(first, n, codes.data(), chunks, threads);
    team = r.team;
    workers = r.workers;
    return r.distinct;
}

void run_case(const std::string& name, const std::vector<double>& values) {
    std::vector<int> expected;
    const bool wide = old_codes(values, expected);
    check(wide, name + ": the vector holds a value outside exact int32");
    if (!wide) {
        return;
    }
    const long long n = static_cast<long long>(values.size());
    int max_code = -1;
    for (const int c : expected) {
        max_code = std::max(max_code, c);
        digest = (digest ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(c))) * 1099511628211ULL;
    }
    std::vector<long long> chunk_options = {1, 2, 3, 7, (n + 16383) / 16384, n, n + 3, 2 * n + 1};
    std::sort(chunk_options.begin(), chunk_options.end());
    chunk_options.erase(std::unique(chunk_options.begin(), chunk_options.end()), chunk_options.end());
    for (const int threads : {1, 2, 3, 8, 16, 48}) {
        for (const long long chunks : chunk_options) {
            for (const bool reverse_ties : {false, true}) {
                if (reverse_ties && (threads != 3 || chunks != chunk_options.back())) {
                    continue;
                }
                std::vector<int> codes;
                int team = 0;
                int workers = 0;
                const long long distinct =
                    new_codes(values, codes, chunks, threads, reverse_ties, team, workers);
                const std::string where = name + " threads=" + std::to_string(threads) +
                                          " chunks=" + std::to_string(chunks) +
                                          (reverse_ties ? " reversed ties" : "");
                check(codes == expected, where + ": codes equal today's");
                check(distinct == static_cast<long long>(max_code) + 1, where + ": distinct");
                check(team >= 1 && team <= threads, where + ": observed team within the request");
                largest_team = std::max(largest_team, team);
                check(workers >= 1 && workers <= team, where + ": active workers within the team");
            }
        }
    }
}

}  // namespace

int main() {
    const double two53 = 9007199254740992.0;
    const double imin = static_cast<double>(std::numeric_limits<int>::min());
    const double imax = static_cast<double>(std::numeric_limits<int>::max());
    std::mt19937_64 rng(20261114);
    int cases = 0;

    run_case("n=1", {std::ldexp(1.0, 40)}), ++cases;
    run_case("one distinct value", std::vector<double>(1000, 3e9)), ++cases;
    {
        // Signed zeros among wide values: -0 and +0 compare equal, and today a
        // -0 before the first wide value comes back from int as +0.
        const std::vector<double> pool = {-0.0, 0.0, 3e9, -3e9, 5e9, -0.0, 0.0, -1e10, 1.0, -1.0};
        std::vector<double> v = {-0.0, 0.0, -0.0};
        std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
        for (int i = 0; i < 4000; ++i) v.push_back(pool[pick(rng)]);
        run_case("signed zeros among wide values", v), ++cases;
        run_case("signed zeros, wide first", std::vector<double>{3e9, -0.0, 0.0, -0.0, 3e9, 0.0}), ++cases;
    }
    {
        std::vector<double> v = {imin, imin - 1.0, imax, imax + 1.0, 2147483648.0, -2147483649.0,
                                 0.0, 1.0, -1.0, imin, imax, imax + 1.0, imin - 1.0};
        run_case("int32 boundaries", v), ++cases;
        std::vector<double> late;
        for (int i = 0; i < 5000; ++i) late.push_back(static_cast<double>(static_cast<int>(rng() % 100) - 50));
        late.push_back(imax + 1.0);
        run_case("late wide value", late), ++cases;
    }
    {
        std::vector<double> v;
        for (const double x : {1.5, std::ldexp(1.0, 40) + 0.5, -3.25, 1e-300, 123456789.125}) {
            double y = x;
            for (int k = 0; k < 4; ++k) {
                v.push_back(y);
                v.push_back(-y);
                y = std::nextafter(y, std::numeric_limits<double>::infinity());
            }
        }
        std::shuffle(v.begin(), v.end(), rng);
        run_case("adjacent fractions", v), ++cases;
    }
    run_case("2^53 and 2^53+2",
             {two53, two53 + 2.0, two53 - 1.0, -two53, two53, std::ldexp(1.0, 52) + 0.5, two53 + 2.0}),
        ++cases;
    {
        // Seven wide values over 100 000 rows: long runs that cross every chunk
        // boundary once sorted.
        const std::vector<double> pool = {3e9, 3e9 + 1, 7e10, -5e9, 1e12, 2.5e9 + 0.5, -0.0};
        std::vector<double> v(100000);
        std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
        for (double& x : v) x = pool[pick(rng)];
        run_case("ties across chunk boundaries", v), ++cases;
    }
    {
        std::vector<double> v(300000);
        std::uniform_int_distribution<long long> id(0, 49999);
        std::uniform_int_distribution<int> kind(0, 99);
        for (double& x : v) {
            const int k = kind(rng);
            x = 1e10 + static_cast<double>(id(rng));
            if (k == 0) x += 0.5;
            if (k == 1) x = -x;
        }
        run_case("random wide, many repeats, fractions and negatives", v), ++cases;
    }
    {
        std::vector<double> v(70000);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 1e12 + 3.0 * static_cast<double>(i);
        std::shuffle(v.begin(), v.end(), rng);
        run_case("all distinct wide values", v), ++cases;
    }

    check(largest_team >= 2, "some region ran on a team of two or more threads");
    std::printf("WIDE_ID_RANK_UNIT_%s checks=%lld failures=%lld cases=%d largest_team=%d digest=%016llx\n",
                failures == 0 ? "PASS" : "FAIL", checks, failures, cases, largest_team,
                static_cast<unsigned long long>(digest));
    return failures == 0 ? 0 : 1;
}
