// Unit test of D10 in the reduced-system kernel (src/reduced_schur_pcg.hpp):
// with Config::chat_stop > 0 a column stops when its chat, confirmed on the
// replaced residual, is at most chat_stop; tau <= atol (test 2) is no longer
// an exit on any path; Zero, Span and test 1 stay; the stall and projection
// declines read chat / chat_stop. Pre-registration and design note:
// Verifications/perf_stata_path_20261114/d10/ (sections 10 and 11).
//
// Cases: a crossing just before and just after the target; a recurrence
// candidate that the replacement does not confirm; a candidate created at a
// periodic replacement; lanes that stop at different iterations; more than
// eight columns (two groups); rho <= 0 at the start; Zero; Span (and the
// reviewer's counterexample, which publishes above 1e-12); test 1; the entry
// into the formed residual; a decline on the chat measure; the same bits at
// every team size; and, over every run with D10 on, no column stopped by test 2.
// With chat_stop = 0 the kernel is the one before D10 (tests/reduced_schur_pcg_unit.cpp
// keeps its digest).
//
// Build, from the worktree root (the flags of the tier's unit test):
//   c++ -std=gnu++17 -O3 -DNDEBUG -march=native -mtune=native -ffast-math
//     -fno-finite-math-only -funroll-loops -ftree-vectorize -fomit-frame-pointer
//     -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math
//     -fopenmp -I src -isystem r/xhdfe/src/eigen tests/reduced_pcg_chat_stop_unit.cpp
//     -o reduced_pcg_chat_stop_unit

#include "reduced_schur_pcg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace rp = hdfe::reduced_pcg;

namespace {

int g_checks = 0;
int g_failures = 0;
#define CHECK(cond, ...)                                 \
    do {                                                 \
        ++g_checks;                                      \
        if (!(cond)) {                                   \
            ++g_failures;                                \
            std::printf("FAIL line %d: ", __LINE__);     \
            std::printf(__VA_ARGS__);                    \
            std::printf("\n");                           \
        }                                                \
    } while (0)

constexpr double kTarget = 1.0e-12;

struct Design {
    int n = 0;
    std::vector<std::vector<int>> codes;
    std::vector<int> levels;
    int eliminated = 0;
    std::vector<std::vector<double>> cols;
};

// Worker-firm-year panel (as in tests/reduced_schur_pcg_unit.cpp), `extra` more
// regressor columns.
Design mobility_panel(unsigned seed, int workers, int firms, int years, double mobility, int extra) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> rows_dist(2, 8), firm_dist(0, firms - 1), start_dist(0, years - 8);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    Design d;
    d.codes.assign(3, {});
    d.levels = {workers, firms, years};
    std::vector<double> wfe(static_cast<std::size_t>(workers)), ffe(static_cast<std::size_t>(firms));
    for (double& v : wfe) v = normal(rng);
    for (double& v : ffe) v = normal(rng);
    std::vector<double> y;
    std::vector<std::vector<double>> xs(static_cast<std::size_t>(extra));
    for (int u = 0; u < workers; ++u) {
        const int rows = rows_dist(rng), home = firm_dist(rng), start = start_dist(rng);
        for (int r = 0; r < rows; ++r) {
            const int firm = unit(rng) < mobility ? firm_dist(rng) : home;
            d.codes[0].push_back(u);
            d.codes[1].push_back(firm);
            d.codes[2].push_back(start + r);
            double v = wfe[static_cast<std::size_t>(u)] + ffe[static_cast<std::size_t>(firm)] + 0.3 * normal(rng);
            for (int x = 0; x < extra; ++x) {
                const double a = normal(rng) + 0.2 * x;
                xs[static_cast<std::size_t>(x)].push_back(a);
                v += 0.1 * a;
            }
            y.push_back(v);
        }
    }
    d.n = static_cast<int>(y.size());
    d.cols.push_back(y);
    for (auto& x : xs) d.cols.push_back(x);
    return d;
}

std::vector<double> span_column(const Design& d, unsigned seed, double noise) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> col(static_cast<std::size_t>(d.n), 0.0);
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        std::vector<double> a(static_cast<std::size_t>(d.levels[b]));
        for (double& v : a) v = normal(rng);
        for (int i = 0; i < d.n; ++i) col[static_cast<std::size_t>(i)] += a[static_cast<std::size_t>(d.codes[b][static_cast<std::size_t>(i)])];
    }
    for (double& v : col) v += noise * normal(rng);
    return col;
}

// abs_residual_rel as certify_absorption_result defines it for unit weights,
// one column (the copy in tests/reduced_schur_pcg_unit.cpp).
double certificate(const Design& d, const std::vector<double>& v, const std::vector<double>& res) {
    long double num = 0.0L;
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        std::vector<long double> sums(static_cast<std::size_t>(d.levels[b]), 0.0L);
        for (int i = 0; i < d.n; ++i) sums[static_cast<std::size_t>(d.codes[b][static_cast<std::size_t>(i)])] += res[static_cast<std::size_t>(i)];
        for (long double s : sums) num += s * s;
    }
    long double raw = 0.0L, sum = 0.0L;
    for (double x : v) {
        raw += static_cast<long double>(x) * x;
        sum += x;
    }
    long double norm_sq = raw;
    const long double n = d.n;
    if (sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * raw) {
        const long double origin = v[0];
        long double shifted = 0.0L;
        for (double x : v) shifted += static_cast<long double>(x) - origin;
        const long double mean = shifted / n;
        long double centered = 0.0L;
        for (double x : v) {
            const long double delta = static_cast<long double>(x) - origin - mean;
            centered += delta * delta;
        }
        if (centered > 0.0L && centered < raw / 1024.0L) norm_sq = centered;
    }
    const long double scale = std::sqrt(static_cast<long double>(d.codes.size()) * n) * std::sqrt(norm_sq);
    if (scale <= 0.0L) return num > 0.0L ? 1.0e308 : 0.0;
    return static_cast<double>(std::sqrt(num) / scale);
}

// Every (column, iteration, chat) the kernel reports after an iteration.
struct Trace {
    std::vector<std::vector<double>> chat;  // per column, index iteration - 1
    static void record(void* self, int column, int iteration, double, double chat) {
        auto* me = static_cast<Trace*>(self);
        if (iteration <= 0) return;
        if (static_cast<int>(me->chat.size()) <= column) me->chat.resize(static_cast<std::size_t>(column) + 1U);
        auto& c = me->chat[static_cast<std::size_t>(column)];
        if (static_cast<int>(c.size()) < iteration) c.resize(static_cast<std::size_t>(iteration), -1.0);
        c[static_cast<std::size_t>(iteration) - 1U] = chat;
    }
};

struct Run {
    rp::Report rep;
    std::vector<std::vector<double>> res;
    Trace trace;
};

rp::Config config(bool d10, int threads = 1) {
    rp::Config c;
    c.threads = threads;
    c.blocks = 48;
    c.atol = 1.0e-11;
    c.btol = 1.0e-11;
    c.trigger = 1.0e-11;
    c.max_iter = 1500;
    c.chat_stop = d10 ? 0.1 * c.trigger : 0.0;
    return c;
}

int g_d10_runs = 0;
int g_test2_with_d10 = 0;

Run solve(const Design& d, rp::Config cfg) {
    Run run;
    const int k = static_cast<int>(d.codes.size());
    std::vector<const int*> codes(static_cast<std::size_t>(k));
    for (int b = 0; b < k; ++b) codes[static_cast<std::size_t>(b)] = d.codes[static_cast<std::size_t>(b)].data();
    run.res.assign(d.cols.size(), std::vector<double>(static_cast<std::size_t>(d.n), 0.0));
    std::vector<const double*> cols(d.cols.size());
    std::vector<double*> res(d.cols.size());
    for (std::size_t c = 0; c < d.cols.size(); ++c) {
        cols[c] = d.cols[c].data();
        res[c] = run.res[c].data();
    }
    rp::Problem p;
    p.n = d.n;
    p.k = k;
    p.codes = codes.data();
    p.levels = d.levels.data();
    p.eliminated = d.eliminated;
    p.cols = static_cast<int>(d.cols.size());
    p.columns = cols.data();
    p.residuals = res.data();
    cfg.iteration_context = &run.trace;
    cfg.on_iteration = &Trace::record;
    run.rep = rp::solve(p, cfg);
    if (cfg.chat_stop > 0.0) {
        ++g_d10_runs;
        for (const auto& c : run.rep.columns) g_test2_with_d10 += c.stop == rp::Stop::Test2 ? 1 : 0;
    }
    return run;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Every Chat stop: finite confirmed chat at most the target, and the first
// iteration whose reported chat is at most the target (the one before it above).
void check_chat_stops(const Design& d, const Run& run, const char* label) {
    CHECK(run.rep.solved, "%s: declined (%s)", label, rp::decline_name(run.rep.decline));
    for (std::size_t c = 0; c < run.rep.columns.size(); ++c) {
        const auto& col = run.rep.columns[c];
        if (col.stop != rp::Stop::Chat) continue;
        CHECK(std::isfinite(col.chat) && col.chat <= kTarget, "%s col %zu: chat %.3e at a Chat stop", label, c, col.chat);
        if (col.iterations > 0 && c < run.trace.chat.size()) {
            const auto& tr = run.trace.chat[c];
            const int k = col.iterations;
            CHECK(static_cast<int>(tr.size()) >= k && tr[static_cast<std::size_t>(k) - 1U] <= kTarget,
                  "%s col %zu: reported chat at the stop", label, c);
            if (k >= 2 && static_cast<int>(tr.size()) >= k)
                CHECK(tr[static_cast<std::size_t>(k) - 2U] > kTarget,
                      "%s col %zu: chat %.3e already at the target one iteration before the stop (%d)", label, c,
                      tr[static_cast<std::size_t>(k) - 2U], k);
        }
        if (run.rep.solved) {
            const double cert = certificate(d, d.cols[c], run.res[c]);
            std::printf("  %s col %zu: stop chat at %d, chat %.4e, certificate %.4e\n", label, c, col.iterations,
                        col.chat, cert);
        }
    }
}

}  // namespace

int main() {
    // 1. Crossings, lanes at different iterations, team sizes, and the rule
    //    before D10 on the same design.
    {
        const Design d = mobility_panel(11, 600, 60, 20, 0.2, 3);
        const Run on = solve(d, config(true));
        check_chat_stops(d, on, "panel");
        int chat_stops = 0, first = -1, distinct = 0;
        for (const auto& c : on.rep.columns) {
            chat_stops += c.stop == rp::Stop::Chat ? 1 : 0;
            if (first < 0) first = c.iterations;
            else if (c.iterations != first) distinct = 1;
        }
        CHECK(chat_stops == static_cast<int>(d.cols.size()), "panel: %d of %zu columns stop on chat", chat_stops,
              d.cols.size());
        CHECK(distinct == 1, "panel: every column stopped at the same iteration (lanes not exercised)");
        const Run off = solve(d, config(false));
        int test2 = 0;
        for (const auto& c : off.rep.columns) test2 += c.stop == rp::Stop::Test2 ? 1 : 0;
        CHECK(off.rep.solved && test2 > 0, "panel, rule before D10: %d test 2 stops", test2);
        std::printf("  panel: iterations %d with D10, %d before\n", on.rep.iterations, off.rep.iterations);
        for (int t : {2, 8, 16, 48}) {
            const Run other = solve(d, config(true, t));
            bool same = other.rep.solved == on.rep.solved && other.rep.iterations == on.rep.iterations;
            for (std::size_t c = 0; same && c < on.res.size(); ++c) {
                same = other.rep.columns[c].stop == on.rep.columns[c].stop &&
                       other.rep.columns[c].iterations == on.rep.columns[c].iterations;
                for (int i = 0; same && i < d.n; ++i)
                    same = same_bits(other.res[c][static_cast<std::size_t>(i)], on.res[c][static_cast<std::size_t>(i)]);
            }
            CHECK(same, "panel: threads=%d differ from threads=1 with D10", t);
        }

        // 2. A recurrence candidate that the replacement does not confirm: at an
        //    early iteration the recurrence's chat reads 0; the replaced residual
        //    is far above the target, so no column stops there.
        rp::Config forced = config(true);
        forced.test_recurrence_chat_at = 3;
        forced.test_recurrence_chat_value = 0.0;
        const Run unconfirmed = solve(d, forced);
        bool none_at_3 = true;
        for (const auto& c : unconfirmed.rep.columns) none_at_3 = none_at_3 && c.iterations > 3;
        CHECK(unconfirmed.rep.solved && none_at_3, "unconfirmed candidate: a column stopped at iteration 3");
        check_chat_stops(d, unconfirmed, "unconfirmed candidate");

        // 3. A candidate created at a periodic replacement: replacements at every
        //    iteration; K is column 0's stop in that schedule; then the same
        //    schedule with the recurrence's chat made huge at K, so the candidate
        //    can only come from the periodic replacement. Both runs are equal up
        //    to K, so column 0 must still stop on chat at K.
        rp::Config every = config(true);
        every.refresh_period = 1;
        const Run base_every = solve(d, every);
        const int K = base_every.rep.columns[0].iterations;
        rp::Config periodic = every;
        periodic.test_recurrence_chat_at = K;
        periodic.test_recurrence_chat_value = 1.0e300;
        const Run at_periodic = solve(d, periodic);
        CHECK(base_every.rep.solved && base_every.rep.columns[0].stop == rp::Stop::Chat && at_periodic.rep.solved &&
                  at_periodic.rep.columns[0].stop == rp::Stop::Chat && at_periodic.rep.columns[0].iterations == K,
              "periodic candidate: column 0 stop %s at %d (want chat at %d)",
              rp::stop_name(at_periodic.rep.columns[0].stop), at_periodic.rep.columns[0].iterations, K);
        check_chat_stops(d, at_periodic, "periodic candidate");
    }

    // 4. More than eight columns: two groups.
    {
        const Design d = mobility_panel(21, 500, 50, 20, 0.25, 9);
        const Run on = solve(d, config(true));
        CHECK(on.rep.groups == 2, "ten columns: %d groups", on.rep.groups);
        check_chat_stops(d, on, "ten columns");
        int chat_stops = 0;
        for (const auto& c : on.rep.columns) chat_stops += c.stop == rp::Stop::Chat ? 1 : 0;
        CHECK(chat_stops == static_cast<int>(d.cols.size()), "ten columns: %d chat stops", chat_stops);
    }

    // 5. rho <= 0 at the start (every worker in one cell: g = 0 while v keeps a
    //    within-cell residual): chat is 0, a Chat stop at iteration 0 with D10,
    //    test 2 before D10.
    {
        Design d;
        d.codes.assign(3, {});
        std::mt19937_64 rng(61);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::vector<double> y;
        for (int u = 0; u < 80; ++u)
            for (int r = 0; r < 3; ++r) {
                d.codes[0].push_back(u);
                d.codes[1].push_back(u % 7);
                d.codes[2].push_back(u % 3);
                y.push_back(normal(rng));
            }
        d.levels = {80, 7, 3};
        d.n = static_cast<int>(y.size());
        d.cols.push_back(y);
        const Run on = solve(d, config(true));
        CHECK(on.rep.solved && on.rep.columns[0].stop == rp::Stop::Chat && on.rep.columns[0].iterations == 0 &&
                  on.rep.columns[0].chat <= kTarget,
              "rho <= 0: stop %s after %d, chat %.3e", rp::stop_name(on.rep.columns[0].stop),
              on.rep.columns[0].iterations, on.rep.columns[0].chat);
        const Run off = solve(d, config(false));
        CHECK(off.rep.solved && off.rep.columns[0].stop == rp::Stop::Test2, "rho <= 0 before D10: stop %s",
              rp::stop_name(off.rep.columns[0].stop));
        bool same = on.rep.solved && off.rep.solved;
        for (int i = 0; same && i < d.n; ++i) same = same_bits(on.res[0][static_cast<std::size_t>(i)], off.res[0][static_cast<std::size_t>(i)]);
        CHECK(same, "rho <= 0: residuals differ between the two rules");
    }

    // 6. Zero, Span and the reviewer's Span counterexample (round 21): two
    //    effects of two levels, v = (1+δ, 1−δ, −1+δ, −1−δ), δ = 2^-38: Span with
    //    zero iterations, and a certificate near 1.82e-12, above the target.
    {
        Design d = mobility_panel(31, 300, 30, 20, 0.2, 0);
        std::vector<double> zero(static_cast<std::size_t>(d.n), 0.0);
        std::vector<double> in_e(static_cast<std::size_t>(d.n));
        for (int i = 0; i < d.n; ++i) in_e[static_cast<std::size_t>(i)] = 0.01 * d.codes[0][static_cast<std::size_t>(i)];
        d.cols.push_back(zero);
        d.cols.push_back(in_e);
        const Run on = solve(d, config(true));
        CHECK(on.rep.solved && on.rep.columns[1].stop == rp::Stop::Zero && on.rep.columns[2].stop == rp::Stop::Span,
              "zero and span: %s, %s", rp::stop_name(on.rep.columns[1].stop), rp::stop_name(on.rep.columns[2].stop));
        check_chat_stops(d, on, "zero and span");

        Design c;
        c.n = 4;
        c.codes = {{0, 0, 1, 1}, {0, 1, 0, 1}};
        c.levels = {2, 2};
        const double delta = std::ldexp(1.0, -38);
        c.cols = {{1.0 + delta, 1.0 - delta, -1.0 + delta, -1.0 - delta}};
        const Run cr = solve(c, config(true));
        const double cert = cr.rep.solved ? certificate(c, c.cols[0], cr.res[0]) : -1.0;
        CHECK(cr.rep.solved && cr.rep.columns[0].stop == rp::Stop::Span && cr.rep.columns[0].iterations == 0 &&
                  cert > kTarget && cert < 1.0e-11,
              "span counterexample: stop %s, certificate %.4e", rp::stop_name(cr.rep.columns[0].stop), cert);
        std::printf("  span counterexample: stop %s, certificate %.4e (above 1e-12, as the reviewer derived)\n",
                    rp::stop_name(cr.rep.columns[0].stop), cert);
    }

    // 7. Test 1 stays: with a loose btol a column mostly in the span stops on
    //    test 1 above the target.
    {
        Design d = mobility_panel(41, 400, 40, 20, 0.2, 0);
        d.cols = {span_column(d, 42, 0.05)};
        rp::Config cfg = config(true);
        cfg.btol = 0.3;
        const Run on = solve(d, cfg);
        CHECK(on.rep.solved && on.rep.columns[0].stop == rp::Stop::Test1 && on.rep.columns[0].chat > kTarget,
              "test 1: stop %s, chat %.3e", rp::stop_name(on.rep.columns[0].stop), on.rep.columns[0].chat);
        std::printf("  test 1: stop %s after %d, chat %.3e\n", rp::stop_name(on.rep.columns[0].stop),
                    on.rep.columns[0].iterations, on.rep.columns[0].chat);
    }

    // 8. The entry into the formed residual: a column of the span plus a tiny
    //    noise; the column still stops on chat.
    {
        Design d = mobility_panel(51, 400, 40, 20, 0.2, 0);
        d.cols = {span_column(d, 52, 1.0e-9)};
        const Run on = solve(d, config(true));
        CHECK(on.rep.solved && on.rep.columns[0].formed_checks > 0, "formed: %d formed checks, %s",
              on.rep.columns[0].formed_checks, on.rep.solved ? "solved" : rp::decline_name(on.rep.decline));
        CHECK(on.rep.columns[0].stop == rp::Stop::Chat, "formed: stop %s", rp::stop_name(on.rep.columns[0].stop));
        check_chat_stops(d, on, "formed");
    }

    // 9. A target nobody can reach: the chat measure must end the attempt by
    //    stall or projection, before the ceiling.
    {
        const Design d = mobility_panel(61, 300, 30, 20, 0.2, 0);
        rp::Config cfg = config(true);
        cfg.chat_stop = 1.0e-40;
        const Run run = solve(d, cfg);
        CHECK(!run.rep.solved && (run.rep.decline == rp::Decline::Stall || run.rep.decline == rp::Decline::Projected),
              "unreachable target: %s after %d", rp::decline_name(run.rep.decline), run.rep.iterations);
        std::printf("  unreachable target: decline %s after %d iterations\n", rp::decline_name(run.rep.decline),
                    run.rep.iterations);
    }

    CHECK(g_test2_with_d10 == 0, "%d column(s) stopped on test 2 with D10 on", g_test2_with_d10);
    std::printf("%d checks, %d failures (%d runs with D10 on, %d test 2 stops among them)\n", g_checks, g_failures,
                g_d10_runs, g_test2_with_d10);
    return g_failures == 0 ? 0 : 1;
}
