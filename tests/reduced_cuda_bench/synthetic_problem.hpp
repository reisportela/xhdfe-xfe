#ifndef XHDFE_TESTS_REDUCED_CUDA_BENCH_SYNTHETIC_PROBLEM_HPP
#define XHDFE_TESTS_REDUCED_CUDA_BENCH_SYNTHETIC_PROBLEM_HPP
// Synthetic fixed-effect codes for the reduced-tier operator bench (package 2,
// stage 2a, 17 November 2026). Public by construction: every code comes from a
// seeded generator and nothing is read from disk. Deterministic: segment i draws
// from its own stream (seed, i), so the codes do not depend on the number of
// threads.
//
// Effect 0 is the eliminated one (its level is the segment of a row). An effect
// with at most `year_like_max` levels takes consecutive codes inside a segment
// (a calendar); any other effect is firm-like: a segment starts at a level drawn
// from a mixture (with probability `tail` a Zipf law of exponent `skew` over the
// levels, else uniform) and moves to a new draw with probability `move` per row. Segment sizes are the mean +- up to
// `spread`, perturbed in pairs so that the total is exactly `rows`.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace xhdfe_bench {

struct SplitMix64 {
    std::uint64_t state;
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
    std::uint64_t below(std::uint64_t bound) { return bound ? next() % bound : 0; }
};

struct Shape {
    std::string name = "custom";
    long long rows = 0;
    std::vector<int> levels;          // eliminated effect first
    double skew = 1.0;                // Zipf exponent of the level sizes of firm-like effects
    double tail = 1.0;                // share of firm-like draws from the Zipf law; the rest uniform
    double move = 0.1;                // per-row probability that a firm-like code changes
    int year_like_max = 64;           // effects with at most this many levels count consecutively
    int spread = 6;                   // segment sizes: mean +- up to spread (pairs keep the total)
    bool sorted = true;               // rows ordered by segment; else a seeded permutation
    double dup = 0.0;                 // probability that a row repeats the previous row's tuple
    // Adversarial switches (each takes the first free segments, in this order).
    long long long_segment_rows = 0;  // one segment with this many rows
    bool single_cell_segment = false; // one segment with one row
    bool left_out_segment = false;    // one segment of three rows sharing one tuple
    bool stayer_level = false;        // level 1 of the first firm-like effect: two segments that never leave it
    int unused_levels = 0;            // declared levels never used, on every remaining effect
    bool disconnected = false;        // two halves of segments and of every remaining effect's levels
    double long_level_share = 0.0;    // share of the first firm-like effect's draws forced to level 0
    std::uint64_t seed = 1;
    // Segment lengths of the free segments (package 2, K1 iteration, 17 November 2026). MeanSpread, the default,
    // is the mean +- `spread` in pairs above (on the plan shapes it never exceeds 16 cells). With a law, each free
    // segment draws its length from its own stream (seed, segment): Uniform on [len_lo, len_hi]; Mixture: with
    // probability len_p uniform on [len_lo, len_mid], else uniform on [len_mid + 1, len_hi]. The total is then
    // brought to exactly `rows` by +-1 steps at the segments a fixed stride visits, inside [len_lo, len_hi].
    enum class LengthLaw { MeanSpread, Uniform, Mixture };
    LengthLaw length_law = LengthLaw::MeanSpread;
    int len_lo = 2, len_mid = 12, len_hi = 26;
    double len_p = 0.85;
};

struct Codes {
    int n = 0;
    std::vector<std::vector<int>> codes;  // k arrays of n codes
    std::vector<int> levels;              // declared levels per effect
};

namespace detail {

inline std::uint64_t stream_seed(std::uint64_t seed, std::uint64_t index) {
    SplitMix64 mix(seed ^ (0xD1B54A32D192ED03ULL * (index + 1)));
    return mix.next();
}

// Zipf(skew) over [0, levels): inverse CDF by binary search on the cumulative weights.
struct Zipf {
    std::vector<double> cdf;
    int levels = 0;
    Zipf(int count, double skew) : levels(count) {
        if (skew <= 0.0) return;  // uniform
        cdf.resize(static_cast<std::size_t>(count));
        long double total = 0.0L;
        for (int i = 0; i < count; ++i) total += std::pow(static_cast<long double>(i + 1), -static_cast<long double>(skew));
        long double run = 0.0L;
        for (int i = 0; i < count; ++i) {
            run += std::pow(static_cast<long double>(i + 1), -static_cast<long double>(skew));
            cdf[static_cast<std::size_t>(i)] = static_cast<double>(run / total);
        }
        cdf.back() = 1.0;
    }
    int draw(SplitMix64& rng) const {
        const double u = rng.uniform();
        if (cdf.empty()) return std::min(levels - 1, static_cast<int>(u * levels));
        const auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
        return std::min(levels - 1, static_cast<int>(it - cdf.begin()));
    }
};

}  // namespace detail

inline Codes generate(const Shape& s) {
    const int k = static_cast<int>(s.levels.size());
    if (k < 2 || k > 5) throw std::invalid_argument("synthetic shape: 2 to 5 effects");
    if (s.rows < 1 || s.rows > INT_MAX) throw std::invalid_argument("synthetic shape: rows out of range");
    for (int b = 0; b < k; ++b)
        if (s.levels[static_cast<std::size_t>(b)] < 1) throw std::invalid_argument("synthetic shape: a level count below 1");
    const long long n = s.rows;
    const int S = s.levels[0];
    if (n < S) throw std::invalid_argument("synthetic shape: fewer rows than segments");

    // Roles of the first segments, then the sizes of the free ones.
    std::vector<long long> size(static_cast<std::size_t>(S), 0);
    int next = 0;
    int long_seg = -1, single_seg = -1, left_seg = -1, stay0 = -1, stay1 = -1;
    if (s.long_segment_rows > 0) { long_seg = next++; size[static_cast<std::size_t>(long_seg)] = s.long_segment_rows; }
    if (s.single_cell_segment) { single_seg = next++; size[static_cast<std::size_t>(single_seg)] = 1; }
    if (s.left_out_segment) { left_seg = next++; size[static_cast<std::size_t>(left_seg)] = 3; }
    if (s.stayer_level) { stay0 = next++; stay1 = next++; size[static_cast<std::size_t>(stay0)] = 4; size[static_cast<std::size_t>(stay1)] = 5; }
    if (next > S) throw std::invalid_argument("synthetic shape: more special segments than segments");
    long long fixed = 0;
    for (int i = 0; i < next; ++i) fixed += size[static_cast<std::size_t>(i)];
    const long long free_rows = n - fixed;
    const int free_segs = S - next;
    if (free_segs > 0 && s.length_law != Shape::LengthLaw::MeanSpread) {
        const long long lo = s.len_lo, mid = s.len_mid, hi = s.len_hi;
        if (lo < 1 || hi < lo || (s.length_law == Shape::LengthLaw::Mixture && (mid < lo || mid >= hi)) ||
            free_rows < lo * free_segs || free_rows > hi * free_segs)
            throw std::invalid_argument("synthetic shape: the length law cannot reach the rows");
        long long total = 0;
        for (int i = 0; i < free_segs; ++i) {
            SplitMix64 rng(detail::stream_seed(s.seed ^ 0x1E46ULL, static_cast<std::uint64_t>(next + i)));
            long long len;
            if (s.length_law == Shape::LengthLaw::Uniform) {
                len = lo + static_cast<long long>(rng.below(static_cast<std::uint64_t>(hi - lo + 1)));
            } else if (rng.uniform() < s.len_p) {
                len = lo + static_cast<long long>(rng.below(static_cast<std::uint64_t>(mid - lo + 1)));
            } else {
                len = mid + 1 + static_cast<long long>(rng.below(static_cast<std::uint64_t>(hi - mid)));
            }
            size[static_cast<std::size_t>(next + i)] = len;
            total += len;
        }
        // To exactly free_rows: +-1 at the segments visited with a stride coprime with free_segs.
        long long diff = free_rows - total;
        long long stride = 1000003;
        while (std::gcd(stride, static_cast<long long>(free_segs)) != 1) stride += 2;
        long long at = 0, idle = 0;
        while (diff != 0) {
            long long& len = size[static_cast<std::size_t>(next + at)];
            if (diff > 0 && len < hi) { ++len; --diff; idle = 0; }
            else if (diff < 0 && len > lo) { --len; ++diff; idle = 0; }
            else if (++idle > free_segs) throw std::logic_error("synthetic shape: length adjustment stalled");
            at = (at + stride) % free_segs;
        }
    } else if (free_segs > 0) {
        if (free_rows < free_segs) throw std::invalid_argument("synthetic shape: rows too few for the segments");
        const long long base = free_rows / free_segs, extra = free_rows % free_segs;
        for (int i = 0; i < free_segs; ++i)
            size[static_cast<std::size_t>(next + i)] = base + (i < extra ? 1 : 0);
        const long long floor_size = base >= 3 ? 2 : 1;
        for (int i = 0; i + 1 < free_segs; i += 2) {
            const std::size_t a = static_cast<std::size_t>(next + i), b = a + 1;
            SplitMix64 rng(detail::stream_seed(s.seed ^ 0x5151ULL, a));
            const long long room = std::min<long long>({static_cast<long long>(s.spread), size[a] - floor_size, size[b] - floor_size});
            if (room <= 0) continue;
            const long long delta = static_cast<long long>(rng.below(static_cast<std::uint64_t>(2 * room + 1))) - room;
            size[a] += delta;
            size[b] -= delta;
        }
    } else if (free_rows != 0) {
        throw std::invalid_argument("synthetic shape: rows do not match the special segments");
    }
    std::vector<long long> start(static_cast<std::size_t>(S) + 1U, 0);
    for (int i = 0; i < S; ++i) start[static_cast<std::size_t>(i) + 1U] = start[static_cast<std::size_t>(i)] + size[static_cast<std::size_t>(i)];
    if (start[static_cast<std::size_t>(S)] != n) throw std::logic_error("synthetic shape: sizes do not add up");

    // Effects: calendar-like or firm-like; the first firm-like one carries the
    // long level and the stayer level.
    std::vector<char> year(static_cast<std::size_t>(k), 0);
    int first_firm = -1;
    for (int b = 1; b < k; ++b) {
        year[static_cast<std::size_t>(b)] = s.levels[static_cast<std::size_t>(b)] <= s.year_like_max ? 1 : 0;
        if (!year[static_cast<std::size_t>(b)] && first_firm < 0) first_firm = b;
    }
    if (s.stayer_level && (first_firm < 0 || s.levels[static_cast<std::size_t>(first_firm)] < 3))
        throw std::invalid_argument("synthetic shape: the stayer level needs a firm-like effect of 3 or more levels");
    std::vector<int> span(static_cast<std::size_t>(k), 0);  // levels a segment may use
    for (int b = 1; b < k; ++b) {
        span[static_cast<std::size_t>(b)] = s.disconnected ? std::max(1, s.levels[static_cast<std::size_t>(b)] / 2)
                                                           : s.levels[static_cast<std::size_t>(b)];
    }
    std::vector<detail::Zipf> zipf;
    for (int b = 0; b < k; ++b)
        zipf.emplace_back(b == 0 || year[static_cast<std::size_t>(b)] ? 1 : span[static_cast<std::size_t>(b)],
                          b == 0 || year[static_cast<std::size_t>(b)] ? 0.0 : s.skew);

    Codes out;
    out.n = static_cast<int>(n);
    out.codes.assign(static_cast<std::size_t>(k), std::vector<int>(static_cast<std::size_t>(n)));
    out.levels = s.levels;
    for (int b = 1; b < k; ++b) out.levels[static_cast<std::size_t>(b)] += s.unused_levels;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < S; ++i) {
        SplitMix64 rng(detail::stream_seed(s.seed, static_cast<std::uint64_t>(i)));
        const bool second_half = s.disconnected && i >= S / 2;
        const bool stayer = i == stay0 || i == stay1;
        int cur[5] = {0, 0, 0, 0, 0}, first[5] = {0, 0, 0, 0, 0};
        auto draw = [&](int b) {
            if (b == first_firm && s.long_level_share > 0.0 && rng.uniform() < s.long_level_share) return 0;
            const int sp = span[static_cast<std::size_t>(b)];
            int code = s.tail >= 1.0 || rng.uniform() < s.tail
                           ? zipf[static_cast<std::size_t>(b)].draw(rng)
                           : std::min(sp - 1, static_cast<int>(rng.uniform() * sp));
            if (s.stayer_level && b == first_firm && code == 1) code = 2;  // level 1 is the stayers' only
            return code;
        };
        for (int b = 1; b < k; ++b) {
            first[b] = year[static_cast<std::size_t>(b)] ? static_cast<int>(rng.below(static_cast<std::uint64_t>(span[static_cast<std::size_t>(b)])))
                                                        : (stayer && b == first_firm ? 1 : draw(b));
            cur[b] = first[b];
        }
        long long step = 0;
        const long long r0 = start[static_cast<std::size_t>(i)], rows_i = size[static_cast<std::size_t>(i)];
        for (long long t = 0; t < rows_i; ++t) {
            const bool repeat = t > 0 && (i == left_seg || (s.dup > 0.0 && rng.uniform() < s.dup));
            if (t > 0 && !repeat) ++step;
            const std::size_t pos = static_cast<std::size_t>(r0 + t);
            out.codes[0][pos] = i;
            for (int b = 1; b < k; ++b) {
                const int sp = span[static_cast<std::size_t>(b)];
                int code;
                if (year[static_cast<std::size_t>(b)]) {
                    code = static_cast<int>((first[b] + step) % sp);
                } else {
                    if (t > 0 && !repeat && !(stayer && b == first_firm) && s.move > 0.0 && rng.uniform() < s.move)
                        cur[b] = draw(b);
                    code = cur[b];
                }
                // The second half of a disconnected design uses the upper half of the levels.
                if (second_half && sp < s.levels[static_cast<std::size_t>(b)]) code += sp;
                out.codes[static_cast<std::size_t>(b)][pos] = code;
            }
        }
    }

    if (!s.sorted) {
        std::vector<int> perm(static_cast<std::size_t>(n));
        for (int r = 0; r < static_cast<int>(n); ++r) perm[static_cast<std::size_t>(r)] = r;
        SplitMix64 rng(detail::stream_seed(s.seed ^ 0xA5A5ULL, 0));
        for (long long r = n - 1; r > 0; --r)
            std::swap(perm[static_cast<std::size_t>(r)], perm[static_cast<std::size_t>(rng.below(static_cast<std::uint64_t>(r + 1)))]);
        std::vector<int> tmp(static_cast<std::size_t>(n));
        for (int b = 0; b < k; ++b) {
            auto& c = out.codes[static_cast<std::size_t>(b)];
            for (long long r = 0; r < n; ++r) tmp[static_cast<std::size_t>(r)] = c[static_cast<std::size_t>(perm[static_cast<std::size_t>(r)])];
            c.swap(tmp);
        }
    }
    return out;
}

}  // namespace xhdfe_bench
#endif
