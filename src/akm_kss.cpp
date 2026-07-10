// AKM two-way estimation + leave-out (KSS) variance decomposition.
//
// Numerical semantics follow Saggio's LeaveOutTwoWay (MATLAB), the canonical
// oracle for Kline-Saggio-Soelvsten (2020):
//   - leave-out connected set = largest connected set, then iterative removal
//     of workers that are articulation points of the mover-firm bipartite
//     graph, then dropping workers observed once;
//   - match-level leave-out collapses to match means and solves the two-way
//     system as sqrt(match-length) FGLS (equal to person-year OLS);
//   - quadratic forms are person-year weighted and grand-mean centered with
//     denominator (n_py - 1);
//   - sigma_i = (y_i - mean(y)) * eta_i / (1 - P_ii) on the transformed rows,
//     with the JLA nonlinearity correction, and the person-year stayer
//     formula (P_ii = 1/T_i) at match level;
//   - JLA: per draw, one Rademacher projection for (Pii, Mii) plus one
//     centered person-year Rademacher draw shared by the fe/pe solves.
//
// This module is opt-in and self-contained: it is not referenced by any
// existing estimation path.

#include "hdfe/akm_kss.hpp"
#include "hdfe/akm_kss_am_tabulation.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
#endif
#endif

#include "hdfe/hdfe_regressor_v11.hpp"

#ifdef HDFE_USE_CUDA
#include "hdfe/akm_kss_cuda.hpp"
#endif

namespace hdfe {
namespace akm {
namespace {

// The core is compiled with -ffast-math, so std::isfinite is unreliable;
// use a finite magnitude threshold instead (same convention as the absorber).
constexpr double kFiniteGuard = 1e300;

inline bool guarded_finite(double v) { return v > -kFiniteGuard && v < kFiniteGuard; }

// SplitMix64: deterministic, stateless, cheap per-(draw, unit) streams so the
// JLA draws are reproducible for any thread count.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline std::uint64_t stream_seed(std::uint64_t seed, std::uint64_t draw_tag, std::uint64_t unit) {
    std::uint64_t s = splitmix64(seed ^ (0xA24BAED4963EE407ULL * (draw_tag + 1)));
    return splitmix64(s ^ (0x9FB21C651E98DF25ULL * (unit + 1)));
}

// Sum of `count` Rademacher(+-1) variables drawn from the SplitMix64 stream
// starting at state `s` (64 draws per state advance, via popcount).
inline double rademacher_sum(std::uint64_t s, long long count) {
    double acc = 0.0;
    long long remaining = count;
    while (remaining > 0) {
        s = splitmix64(s);
        const int k = remaining >= 64 ? 64 : static_cast<int>(remaining);
        const std::uint64_t bits =
            k == 64 ? s : (s & ((1ULL << static_cast<unsigned>(k)) - 1ULL));
        acc += 2.0 * static_cast<double>(std::bitset<64>(bits).count()) - static_cast<double>(k);
        remaining -= k;
    }
    return acc;
}

struct UnionFind {
    std::vector<int> parent;

    explicit UnionFind(int n) : parent(static_cast<std::size_t>(n)) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (a > b) {
            std::swap(a, b);
        }
        parent[static_cast<std::size_t>(b)] = a;
    }
};

// Env-gated phase profiler (XHDFE_AKM_PROFILE=1 -> stderr): diagnostic only,
// no effect on results or default behavior.
struct AkmPhaseProfiler {
    bool on = false;
    std::chrono::steady_clock::time_point t0;
    AkmPhaseProfiler() {
        const char* e = std::getenv("XHDFE_AKM_PROFILE");
        on = (e != nullptr && e[0] != '\0' && e[0] != '0');
        t0 = std::chrono::steady_clock::now();
    }
    void mark(const char* phase) {
        if (!on) return;
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        std::fprintf(stderr, "[akm-profile] %-24s %10.1f ms\n", phase, ms);
        t0 = t1;
    }
};

int relabel_dense(const Eigen::VectorXi& ids, std::vector<int>& out) {
    const Eigen::Index n = ids.size();
    out.resize(static_cast<std::size_t>(n));
    // Fast path for near-dense id ranges: a direct-indexed LUT assigns labels
    // in the same first-appearance row order as the hash map, so the output
    // is identical; it just avoids ~n hash lookups.
    if (n > 0) {
        int mn = ids[0], mx = ids[0];
        for (Eigen::Index i = 1; i < n; ++i) {
            mn = std::min(mn, ids[i]);
            mx = std::max(mx, ids[i]);
        }
        const long long range =
            static_cast<long long>(mx) - static_cast<long long>(mn) + 1;
        if (range <= 4LL * static_cast<long long>(n) + 1024) {
            std::vector<int> lut(static_cast<std::size_t>(range), -1);
            int next = 0;
            for (Eigen::Index i = 0; i < n; ++i) {
                int& slot = lut[static_cast<std::size_t>(ids[i] - mn)];
                if (slot < 0) {
                    slot = next++;
                }
                out[static_cast<std::size_t>(i)] = slot;
            }
            return next;
        }
    }
    std::unordered_map<int, int> levels;
    levels.reserve(static_cast<std::size_t>(std::min<Eigen::Index>(n, 1 << 20)));
    int next = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        auto it = levels.find(ids[i]);
        if (it == levels.end()) {
            it = levels.emplace(ids[i], next++).first;
        }
        out[static_cast<std::size_t>(i)] = it->second;
    }
    return next;
}

// Sample structures shared by the leave-out set and the estimator. Matches
// are unique (worker, firm) pairs sorted by (worker, firm) over the FINAL
// kept rows, with dense final labels.
struct SampleBuild {
    LeaveOutSetResult set;
    std::vector<int> rows;        // kept original row indices, input order
    std::vector<int> row_w;       // final dense worker label per kept row
    std::vector<int> row_f;       // final dense firm label per kept row
    std::vector<int> row_match;   // final match id per kept row
    int n_workers = 0;
    int n_firms = 0;
    std::vector<int> m_w;         // per match
    std::vector<int> m_f;
    std::vector<long long> m_cnt; // person-year rows per match
    std::vector<int> worker_orig; // original worker id per final label
    std::vector<int> firm_orig;   // original firm id per final label
};

// Iterative Tarjan articulation points on an undirected graph given in CSR
// form. Parallel edges are irrelevant for articulation vertices, so the
// caller passes unique edges.
void articulation_points(int n_vertices,
                         const std::vector<int>& adj_ptr,
                         const std::vector<int>& adj,
                         std::vector<std::uint8_t>& artic) {
    artic.assign(static_cast<std::size_t>(n_vertices), 0);
    std::vector<int> disc(static_cast<std::size_t>(n_vertices), -1);
    std::vector<int> low(static_cast<std::size_t>(n_vertices), 0);
    std::vector<int> parent(static_cast<std::size_t>(n_vertices), -1);
    std::vector<int> child_count(static_cast<std::size_t>(n_vertices), 0);
    std::vector<int> edge_it(static_cast<std::size_t>(n_vertices), 0);
    std::vector<int> stack;
    stack.reserve(1024);
    int timer = 0;

    for (int root = 0; root < n_vertices; ++root) {
        if (disc[static_cast<std::size_t>(root)] != -1) {
            continue;
        }
        stack.push_back(root);
        disc[static_cast<std::size_t>(root)] = low[static_cast<std::size_t>(root)] = timer++;
        edge_it[static_cast<std::size_t>(root)] = adj_ptr[static_cast<std::size_t>(root)];
        while (!stack.empty()) {
            const int u = stack.back();
            if (edge_it[static_cast<std::size_t>(u)] < adj_ptr[static_cast<std::size_t>(u) + 1]) {
                const int v = adj[static_cast<std::size_t>(edge_it[static_cast<std::size_t>(u)]++)];
                if (disc[static_cast<std::size_t>(v)] == -1) {
                    parent[static_cast<std::size_t>(v)] = u;
                    ++child_count[static_cast<std::size_t>(u)];
                    disc[static_cast<std::size_t>(v)] = low[static_cast<std::size_t>(v)] = timer++;
                    edge_it[static_cast<std::size_t>(v)] = adj_ptr[static_cast<std::size_t>(v)];
                    stack.push_back(v);
                } else if (v != parent[static_cast<std::size_t>(u)]) {
                    low[static_cast<std::size_t>(u)] =
                        std::min(low[static_cast<std::size_t>(u)], disc[static_cast<std::size_t>(v)]);
                }
            } else {
                stack.pop_back();
                const int p = parent[static_cast<std::size_t>(u)];
                if (p != -1) {
                    low[static_cast<std::size_t>(p)] =
                        std::min(low[static_cast<std::size_t>(p)], low[static_cast<std::size_t>(u)]);
                    if (p != root && low[static_cast<std::size_t>(u)] >= disc[static_cast<std::size_t>(p)]) {
                        artic[static_cast<std::size_t>(p)] = 1;
                    }
                }
            }
        }
        if (child_count[static_cast<std::size_t>(root)] >= 2) {
            artic[static_cast<std::size_t>(root)] = 1;
        }
    }
}

SampleBuild build_leave_out_sample(const Eigen::VectorXi& worker_ids,
                                   const Eigen::VectorXi& firm_ids,
                                   bool prune,
                                   const std::vector<long long>* fw = nullptr) {
    if (worker_ids.size() != firm_ids.size()) {
        throw std::runtime_error("worker and firm id vectors must have the same length");
    }
    const Eigen::Index n = worker_ids.size();
    SampleBuild out;
    out.set.n_obs_input = static_cast<long long>(n);
    if (n == 0) {
        return out;
    }
    AkmPhaseProfiler bprof_;

    std::vector<int> w0;
    std::vector<int> f0;
    const int W0 = relabel_dense(worker_ids, w0);
    const int F0 = relabel_dense(firm_ids, f0);
    bprof_.mark("los_relabel");

    // Initial match table: sort row indices by (worker, firm).
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    {
        std::vector<std::uint64_t> key(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            key[static_cast<std::size_t>(i)] =
                static_cast<std::uint64_t>(w0[static_cast<std::size_t>(i)]) *
                    static_cast<std::uint64_t>(F0) +
                static_cast<std::uint64_t>(f0[static_cast<std::size_t>(i)]);
        }
        const auto cmp = [&key](std::int64_t a, std::int64_t b) {
            return key[static_cast<std::size_t>(a)] != key[static_cast<std::size_t>(b)]
                       ? key[static_cast<std::size_t>(a)] < key[static_cast<std::size_t>(b)]
                       : a < b;
        };
        // cmp is a strict total order (index tie-break), so every correct sort
        // yields the identical permutation; use the parallel sort where
        // available (GCC libstdc++ parallel mode) and std::sort elsewhere.
#if defined(_OPENMP) && defined(__GNUC__) && !defined(__clang__)
        __gnu_parallel::sort(order.begin(), order.end(), cmp);
#else
        std::sort(order.begin(), order.end(), cmp);
#endif
    }
    bprof_.mark("los_sort");

    std::vector<int> match_w;
    std::vector<int> match_f;
    std::vector<long long> match_cnt;
    std::vector<int> row_match_all(static_cast<std::size_t>(n));
    for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
        const std::size_t row = static_cast<std::size_t>(order[k]);
        if (match_w.empty() || match_w.back() != w0[row] || match_f.back() != f0[row]) {
            match_w.push_back(w0[row]);
            match_f.push_back(f0[row]);
            match_cnt.push_back(0);
        }
        match_cnt.back() += fw != nullptr ? (*fw)[row] : 1;
        row_match_all[row] = static_cast<int>(match_w.size()) - 1;
    }
    const std::size_t n_matches_all = match_w.size();
    bprof_.mark("los_match_table");
    std::vector<std::uint8_t> active(n_matches_all, 1);

    // Largest connected component over active matches; component size is
    // measured in firms (LeaveOutTwoWay convention), ties broken by
    // person-year rows and then — deterministically — by the smallest
    // ORIGINAL firm id in the component (adversarial audit C2, 09jul2026:
    // the previous final tie-break was the first-encountered union-find
    // root, i.e. raw input ROW ORDER; on tie-heavy small graphs row order
    // could even decide an empty-vs-valid final sample. The min-firm-id key
    // is unique — components partition firms — and invariant to row
    // permutation; non-tied graphs are unaffected. The reference MATLAB has
    // no defined tie-break at all.)
    std::vector<long long> comp_obs(static_cast<std::size_t>(W0 + F0));
    std::vector<int> comp_firms(static_cast<std::size_t>(W0 + F0));
    std::vector<int> comp_min_firm(static_cast<std::size_t>(W0 + F0));
    std::vector<std::uint8_t> firm_seen(static_cast<std::size_t>(F0));
    std::vector<int> firm_orig(static_cast<std::size_t>(F0));
    for (Eigen::Index i = 0; i < n; ++i) {
        firm_orig[static_cast<std::size_t>(f0[static_cast<std::size_t>(i)])] =
            firm_ids[i];
    }
    const auto largest_cc = [&]() {
        UnionFind uf(W0 + F0);
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (active[m]) {
                uf.unite(match_w[m], W0 + match_f[m]);
            }
        }
        std::fill(comp_obs.begin(), comp_obs.end(), 0);
        std::fill(comp_firms.begin(), comp_firms.end(), 0);
        std::fill(comp_min_firm.begin(), comp_min_firm.end(),
                  std::numeric_limits<int>::max());
        std::fill(firm_seen.begin(), firm_seen.end(), 0);
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (!active[m]) {
                continue;
            }
            const int root = uf.find(match_w[m]);
            comp_obs[static_cast<std::size_t>(root)] += match_cnt[m];
            if (!firm_seen[static_cast<std::size_t>(match_f[m])]) {
                firm_seen[static_cast<std::size_t>(match_f[m])] = 1;
                const std::size_t fr =
                    static_cast<std::size_t>(uf.find(W0 + match_f[m]));
                ++comp_firms[fr];
                const int fo = firm_orig[static_cast<std::size_t>(match_f[m])];
                if (fo < comp_min_firm[fr]) {
                    comp_min_firm[fr] = fo;
                }
            }
        }
        int best = -1;
        for (int r = 0; r < W0 + F0; ++r) {
            const std::size_t rs = static_cast<std::size_t>(r);
            if (comp_firms[rs] == 0 && comp_obs[rs] == 0) {
                continue;
            }
            if (best < 0) {
                best = r;
                continue;
            }
            const std::size_t bs = static_cast<std::size_t>(best);
            if (comp_firms[rs] != comp_firms[bs]) {
                if (comp_firms[rs] > comp_firms[bs]) {
                    best = r;
                }
            } else if (comp_obs[rs] != comp_obs[bs]) {
                if (comp_obs[rs] > comp_obs[bs]) {
                    best = r;
                }
            } else if (comp_min_firm[rs] < comp_min_firm[bs]) {
                best = r;
            }
        }
        if (best < 0) {
            return;
        }
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (active[m] && uf.find(match_w[m]) != best) {
                active[m] = 0;
            }
        }
    };

    largest_cc();
    {
        long long obs = 0;
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (active[m]) {
                obs += match_cnt[m];
            }
        }
        out.set.n_obs_connected = obs;
    }
    bprof_.mark("los_largest_cc");

    if (prune) {
        std::vector<int> worker_deg(static_cast<std::size_t>(W0));
        std::vector<int> mover_idx(static_cast<std::size_t>(W0));
        std::vector<int> firm_idx(static_cast<std::size_t>(F0));
        double ms_deg = 0.0, ms_csr = 0.0, ms_artic = 0.0, ms_drop = 0.0;
        auto now_ = std::chrono::steady_clock::now;
        auto acc_ = [](std::chrono::steady_clock::time_point& t, double& sink) {
            const auto t1 = std::chrono::steady_clock::now();
            sink += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t).count() / 1000.0;
            t = t1;
        };
        while (true) {
            auto tp_ = now_();
            // Movers = workers with >= 2 active matches (distinct firms).
            // match_w is non-decreasing (match table built from (worker, firm)
            // sorted rows), so per-worker counts land in contiguous runs and
            // parallel chunks only race on at most one boundary worker each —
            // atomics keep the counts exact.
            std::fill(worker_deg.begin(), worker_deg.end(), 0);
            const std::int64_t m_all = static_cast<std::int64_t>(n_matches_all);
#pragma omp parallel for schedule(static)
            for (std::int64_t m = 0; m < m_all; ++m) {
                const std::size_t ms = static_cast<std::size_t>(m);
                if (active[ms]) {
#pragma omp atomic
                    ++worker_deg[static_cast<std::size_t>(match_w[ms])];
                }
            }
            std::fill(mover_idx.begin(), mover_idx.end(), -1);
            std::fill(firm_idx.begin(), firm_idx.end(), -1);
            int n_mov = 0;
            int n_frm = 0;
            long long n_edges = 0;
            for (std::size_t m = 0; m < n_matches_all; ++m) {
                if (!active[m] || worker_deg[static_cast<std::size_t>(match_w[m])] < 2) {
                    continue;
                }
                ++n_edges;
                if (mover_idx[static_cast<std::size_t>(match_w[m])] < 0) {
                    mover_idx[static_cast<std::size_t>(match_w[m])] = n_mov++;
                }
                if (firm_idx[static_cast<std::size_t>(match_f[m])] < 0) {
                    firm_idx[static_cast<std::size_t>(match_f[m])] = n_frm++;
                }
            }
            if (n_mov == 0) {
                break;
            }
            acc_(tp_, ms_deg);
            const int n_vertices = n_mov + n_frm;
            std::vector<int> adj_ptr(static_cast<std::size_t>(n_vertices) + 1, 0);
            for (std::size_t m = 0; m < n_matches_all; ++m) {
                if (!active[m] || worker_deg[static_cast<std::size_t>(match_w[m])] < 2) {
                    continue;
                }
                ++adj_ptr[static_cast<std::size_t>(mover_idx[static_cast<std::size_t>(match_w[m])]) + 1];
                ++adj_ptr[static_cast<std::size_t>(n_mov + firm_idx[static_cast<std::size_t>(match_f[m])]) + 1];
            }
            for (std::size_t v = 0; v < static_cast<std::size_t>(n_vertices); ++v) {
                adj_ptr[v + 1] += adj_ptr[v];
            }
            std::vector<int> adj(static_cast<std::size_t>(2 * n_edges));
            {
                std::vector<int> fill_pos(adj_ptr.begin(), adj_ptr.end() - 1);
                for (std::size_t m = 0; m < n_matches_all; ++m) {
                    if (!active[m] || worker_deg[static_cast<std::size_t>(match_w[m])] < 2) {
                        continue;
                    }
                    const int mv = mover_idx[static_cast<std::size_t>(match_w[m])];
                    const int fv = n_mov + firm_idx[static_cast<std::size_t>(match_f[m])];
                    adj[static_cast<std::size_t>(fill_pos[static_cast<std::size_t>(mv)]++)] = fv;
                    adj[static_cast<std::size_t>(fill_pos[static_cast<std::size_t>(fv)]++)] = mv;
                }
            }
            acc_(tp_, ms_csr);
            std::vector<std::uint8_t> artic;
            articulation_points(n_vertices, adj_ptr, adj, artic);
            acc_(tp_, ms_artic);
            std::vector<std::uint8_t> bad_worker(static_cast<std::size_t>(W0), 0);
            bool any_bad = false;
            for (int w = 0; w < W0; ++w) {
                const int mv = mover_idx[static_cast<std::size_t>(w)];
                if (mv >= 0 && artic[static_cast<std::size_t>(mv)]) {
                    bad_worker[static_cast<std::size_t>(w)] = 1;
                    any_bad = true;
                }
            }
            if (!any_bad) {
                break;
            }
#pragma omp parallel for schedule(static)
            for (std::int64_t m = 0; m < m_all; ++m) {
                const std::size_t ms = static_cast<std::size_t>(m);
                if (active[ms] && bad_worker[static_cast<std::size_t>(match_w[ms])]) {
                    active[ms] = 0;
                }
            }
            largest_cc();
            ++out.set.prune_iterations;
            acc_(tp_, ms_drop);
        }
        if (bprof_.on) {
            std::fprintf(stderr,
                         "[akm-profile]   prune sub: deg=%.1f csr=%.1f artic=%.1f "
                         "drop+cc=%.1f ms\n",
                         ms_deg, ms_csr, ms_artic, ms_drop);
        }

        // Drop workers observed only once (LeaveOutTwoWay: T > 1 filter).
        std::vector<long long> worker_obs(static_cast<std::size_t>(W0), 0);
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (active[m]) {
                worker_obs[static_cast<std::size_t>(match_w[m])] += match_cnt[m];
            }
        }
        for (std::size_t m = 0; m < n_matches_all; ++m) {
            if (active[m] && worker_obs[static_cast<std::size_t>(match_w[m])] == 1) {
                active[m] = 0;
            }
        }
        bprof_.mark("los_prune_loop");
    }

    // Final sample: rows of active matches, final dense labels in
    // first-appearance order over kept rows.
    out.set.keep.assign(static_cast<std::size_t>(n), 0);
    std::vector<int> w_final(static_cast<std::size_t>(W0), -1);
    std::vector<int> f_final(static_cast<std::size_t>(F0), -1);
    out.rows.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        const std::size_t is = static_cast<std::size_t>(i);
        if (!active[static_cast<std::size_t>(row_match_all[is])]) {
            continue;
        }
        out.set.keep[is] = 1;
        out.rows.push_back(static_cast<int>(i));
        if (w_final[static_cast<std::size_t>(w0[is])] < 0) {
            w_final[static_cast<std::size_t>(w0[is])] = out.n_workers++;
            out.worker_orig.push_back(worker_ids[i]);
        }
        if (f_final[static_cast<std::size_t>(f0[is])] < 0) {
            f_final[static_cast<std::size_t>(f0[is])] = out.n_firms++;
            out.firm_orig.push_back(firm_ids[i]);
        }
    }
    const std::size_t n_kept = out.rows.size();
    out.row_w.resize(n_kept);
    out.row_f.resize(n_kept);
    for (std::size_t k = 0; k < n_kept; ++k) {
        const std::size_t row = static_cast<std::size_t>(out.rows[k]);
        out.row_w[k] = w_final[static_cast<std::size_t>(w0[row])];
        out.row_f[k] = f_final[static_cast<std::size_t>(f0[row])];
    }

    // Final match table sorted by (worker, firm) in final labels.
    out.row_match.resize(n_kept);
    if (n_kept > 0) {
        std::vector<std::int64_t> ord(n_kept);
        std::iota(ord.begin(), ord.end(), 0);
        std::vector<std::uint64_t> key(n_kept);
        for (std::size_t k = 0; k < n_kept; ++k) {
            key[k] = static_cast<std::uint64_t>(out.row_w[k]) *
                         static_cast<std::uint64_t>(std::max(out.n_firms, 1)) +
                     static_cast<std::uint64_t>(out.row_f[k]);
        }
        const auto cmp2 = [&key](std::int64_t a, std::int64_t b) {
            return key[static_cast<std::size_t>(a)] != key[static_cast<std::size_t>(b)]
                       ? key[static_cast<std::size_t>(a)] < key[static_cast<std::size_t>(b)]
                       : a < b;
        };
#if defined(_OPENMP) && defined(__GNUC__) && !defined(__clang__)
        __gnu_parallel::sort(ord.begin(), ord.end(), cmp2);
#else
        std::sort(ord.begin(), ord.end(), cmp2);
#endif
        for (std::size_t k = 0; k < n_kept; ++k) {
            const std::size_t r = static_cast<std::size_t>(ord[k]);
            if (out.m_w.empty() || out.m_w.back() != out.row_w[r] || out.m_f.back() != out.row_f[r]) {
                out.m_w.push_back(out.row_w[r]);
                out.m_f.push_back(out.row_f[r]);
                out.m_cnt.push_back(0);
            }
            out.m_cnt.back() +=
                fw != nullptr ? (*fw)[static_cast<std::size_t>(out.rows[r])] : 1;
            out.row_match[r] = static_cast<int>(out.m_w.size()) - 1;
        }
    }

    // Set summary. n_obs counts person-year observations (= sum of the
    // frequency weights on the kept rows; the number of kept input rows
    // when unweighted).
    {
        long long tot = 0;
        for (std::size_t m = 0; m < out.m_cnt.size(); ++m) tot += out.m_cnt[m];
        out.set.n_obs = tot;
    }
    out.set.n_workers = out.n_workers;
    out.set.n_firms = out.n_firms;
    out.set.n_matches = static_cast<long long>(out.m_w.size());
    if (out.n_workers > 0) {
        std::vector<int> deg(static_cast<std::size_t>(out.n_workers), 0);
        for (std::size_t m = 0; m < out.m_w.size(); ++m) {
            ++deg[static_cast<std::size_t>(out.m_w[m])];
        }
        for (int w = 0; w < out.n_workers; ++w) {
            if (deg[static_cast<std::size_t>(w)] >= 2) {
                ++out.set.n_movers;
            }
        }
        out.set.n_stayers = out.n_workers - out.set.n_movers;
    }
    bprof_.mark("los_finalize");
    return out;
}

// Two-way normal-equation solver on the (weighted) match design:
// K = [[Dw, B], [B', Df]] with the last firm grounded. Direct sparse
// Cholesky of the worker-partialled firm Laplacian S = Df - B' Dw^{-1} B
// when feasible, matrix-free Jacobi-PCG otherwise. All paths deterministic.
struct TwoWaySolver {
    int N = 0;
    int J = 0;
    int Jr = 0;  // grounded firm-block dimension (J - 1)
    int team = 1;  // work-aware OMP team size for the light per-iteration regions
    const std::vector<int>* m_w = nullptr;
    const std::vector<int>* m_f = nullptr;
    std::vector<double> m_c;  // match weights (person-year counts)
    std::vector<double> m_sqrt_c;  // lazy sqrt(match weight), reused by JLA
    Eigen::VectorXd Dw;
    Eigen::VectorXd Df;
    // Worker-major CSR (matches are sorted by worker already).
    std::vector<std::int64_t> w_ptr;
    // Firm-major copy.
    std::vector<std::int64_t> f_ptr;
    std::vector<std::int64_t> f_matches;
    Eigen::VectorXd diagS;
    Eigen::VectorXd inv_diagS;
    bool direct = false;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    double cg_tol = 1e-10;
    int cg_max_iter = 1000;
    bool parallel_products = true;  // outer loops may disable to nest safely
#ifdef HDFE_USE_CUDA
    hdfe::akm::AkmCudaContext* cuda_ctx = nullptr;
#endif
    bool use_cuda = false;
    long long cuda_iters = 0;

    ~TwoWaySolver() {
#ifdef HDFE_USE_CUDA
        if (cuda_ctx != nullptr) {
            akm_cuda_destroy(cuda_ctx);
        }
#endif
    }

    void build(int n_workers, int n_firms,
               const std::vector<int>& mw, const std::vector<int>& mf,
               const std::vector<long long>& mc,
               const AkmOptions& opt, std::string& notes) {
        N = n_workers;
        J = n_firms;
        Jr = std::max(J - 1, 0);
        m_w = &mw;
        m_f = &mf;
        const std::size_t M = mw.size();
        m_c.resize(M);
        Dw = Eigen::VectorXd::Zero(N);
        Df = Eigen::VectorXd::Zero(J);
        for (std::size_t m = 0; m < M; ++m) {
            m_c[m] = static_cast<double>(mc[m]);
            Dw[mw[m]] += m_c[m];
            Df[mf[m]] += m_c[m];
        }
        w_ptr.assign(static_cast<std::size_t>(N) + 1, 0);
        for (std::size_t m = 0; m < M; ++m) {
            ++w_ptr[static_cast<std::size_t>(mw[m]) + 1];
        }
        for (std::size_t w = 0; w < static_cast<std::size_t>(N); ++w) {
            w_ptr[w + 1] += w_ptr[w];
        }
        f_ptr.assign(static_cast<std::size_t>(J) + 1, 0);
        for (std::size_t m = 0; m < M; ++m) {
            ++f_ptr[static_cast<std::size_t>(mf[m]) + 1];
        }
        for (std::size_t f = 0; f < static_cast<std::size_t>(J); ++f) {
            f_ptr[f + 1] += f_ptr[f];
        }
        f_matches.resize(M);
        {
            std::vector<std::int64_t> pos(f_ptr.begin(), f_ptr.end() - 1);
            for (std::size_t m = 0; m < M; ++m) {
                f_matches[static_cast<std::size_t>(pos[static_cast<std::size_t>(mf[m])]++)] =
                    static_cast<std::int64_t>(m);
            }
        }
        diagS = Df;
        for (std::size_t m = 0; m < M; ++m) {
            diagS[mf[m]] -= m_c[m] * m_c[m] / Dw[mw[m]];
        }
        inv_diagS.resize(J);
        for (int f = 0; f < J; ++f) {
            inv_diagS[f] = (f < Jr && diagS[f] > 0.0) ? 1.0 / diagS[f] : 0.0;
        }
        // Work-aware OMP team size (perf audit 09jul2026): the per-iteration
        // parallel regions do O(M) gather work split into FIXED blocks, so a
        // full 48-thread team on a small/medium graph spends more time at the
        // per-region barriers than computing — measured 4-34x slowdowns vs
        // 8-16 threads on every panel below ~10M person-year rows (both CPU
        // and the host side of GPU runs), while at 47.5M the full team is
        // right. Cap the team by the actual edge work; the fixed 4096/64
        // element blocks make every capped region's output partition-
        // invariant (disjoint writes, no cross-thread FP reductions), so the
        // cap CANNOT change any output bit — verified t1/t8/t16/t48 identical.
        // XHDFE_AKM_TEAM=0 restores the uncapped legacy team; =k forces
        // min(k, omp_get_max_threads()).
#ifdef _OPENMP
        {
            const int amb = omp_get_max_threads();
            team = static_cast<int>(std::min<long long>(
                amb,
                std::max<long long>(1, static_cast<long long>(M) / 65536)));
            if (const char* e = std::getenv("XHDFE_AKM_TEAM")) {
                const int v = std::atoi(e);
                if (v == 0) {
                    team = amb;
                } else if (v > 0) {
                    team = std::min(v, amb);
                }
            }
        }
#else
        team = 1;
#endif
        cg_tol = opt.cg_tol;
        cg_max_iter = opt.cg_max_iter > 0
                          ? opt.cg_max_iter
                          : static_cast<int>(std::min<long long>(
                                50000, std::max<long long>(1000, 2LL * Jr)));

        long long fill_est = 0;
        for (std::size_t w = 0; w < static_cast<std::size_t>(N); ++w) {
            const long long deg = static_cast<long long>(w_ptr[w + 1] - w_ptr[w]);
            fill_est += deg * deg;
        }
#ifdef HDFE_USE_CUDA
        if (opt.use_gpu && Jr > 0) {
            cuda_ctx = akm_cuda_create(N, J, mw.data(), mf.data(), m_c.data(),
                                       M, w_ptr.data(), f_ptr.data(),
                                       f_matches.data(), Dw.data(), Df.data(),
                                       inv_diagS.data());
            if (cuda_ctx != nullptr) {
                use_cuda = true;
            } else {
                notes += "CUDA unavailable for the AKM solver; using CPU. ";
            }
        }
#else
        if (opt.use_gpu) {
            notes += "CUDA not compiled in; AKM solver on CPU. ";
        }
#endif
        if (!use_cuda && Jr > 0 && J <= opt.direct_max_firms && fill_est <= opt.direct_max_nnz) {
            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(static_cast<std::size_t>(fill_est) + static_cast<std::size_t>(Jr));
            for (int f = 0; f < Jr; ++f) {
                trips.emplace_back(f, f, Df[f]);
            }
            for (std::size_t w = 0; w < static_cast<std::size_t>(N); ++w) {
                for (std::int64_t a = w_ptr[w]; a < w_ptr[w + 1]; ++a) {
                    const int fa = mf[static_cast<std::size_t>(a)];
                    if (fa >= Jr) {
                        continue;
                    }
                    const double ca = m_c[static_cast<std::size_t>(a)];
                    for (std::int64_t b = w_ptr[w]; b < w_ptr[w + 1]; ++b) {
                        const int fb = mf[static_cast<std::size_t>(b)];
                        if (fb >= Jr) {
                            continue;
                        }
                        trips.emplace_back(fa, fb,
                                           -ca * m_c[static_cast<std::size_t>(b)] /
                                               Dw[mw[static_cast<std::size_t>(a)]]);
                    }
                }
            }
            Eigen::SparseMatrix<double> S(Jr, Jr);
            S.setFromTriplets(trips.begin(), trips.end());
            // Guard the factorization, not just the triplet count: on dense
            // mobility graphs (high average firm degree) sparse Cholesky fill
            // explodes even when nnz(S) is moderate, while Jacobi-PCG on such
            // well-connected Laplacians converges quickly. Only factor when
            // the graph is genuinely sparse.
            const long long s_nnz = static_cast<long long>(S.nonZeros());
            if (s_nnz <= opt.direct_max_nnz && s_nnz <= 32LL * Jr) {
                ldlt.compute(S);
                if (ldlt.info() == Eigen::Success) {
                    direct = true;
                } else {
                    notes += "direct Cholesky of the firm Laplacian failed; using PCG. ";
                }
            }
        }
    }

    void ensure_sqrt_c() {
        if (m_sqrt_c.size() == m_c.size()) return;
        m_sqrt_c.resize(m_c.size());
        for (std::size_t m = 0; m < m_c.size(); ++m) {
            m_sqrt_c[m] = std::sqrt(m_c[m]);
        }
    }

    // t = B x (worker space), x over firms.
    void mult_B(const double* x, double* t, bool parallel) const {
        const std::vector<int>& mf = *m_f;
        parallel = parallel && N >= 16384;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && team > 1) num_threads(team)
#endif
        for (std::int64_t w = 0; w < N; ++w) {
            double acc = 0.0;
            for (std::int64_t a = w_ptr[static_cast<std::size_t>(w)];
                 a < w_ptr[static_cast<std::size_t>(w) + 1]; ++a) {
                acc += m_c[static_cast<std::size_t>(a)] * x[mf[static_cast<std::size_t>(a)]];
            }
            t[w] = acc;
        }
    }

    // y = B' s (firm space), s over workers.
    void mult_Bt(const double* s, double* y, bool parallel) const {
        const std::vector<int>& mw = *m_w;
        parallel = parallel && J >= 16384;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && team > 1) num_threads(team)
#endif
        for (std::int64_t f = 0; f < J; ++f) {
            double acc = 0.0;
            for (std::int64_t a = f_ptr[static_cast<std::size_t>(f)];
                 a < f_ptr[static_cast<std::size_t>(f) + 1]; ++a) {
                const std::size_t m = static_cast<std::size_t>(f_matches[static_cast<std::size_t>(a)]);
                acc += m_c[m] * s[mw[m]];
            }
            y[f] = acc;
        }
    }

    // Solve S z = rhs on the grounded firm block (rhs/z length J, ground
    // entry ignored and zeroed). Returns PCG iterations used via `iters`.
    bool solve_S(Eigen::VectorXd& rhs_full, Eigen::VectorXd& z_full,
                 Eigen::VectorXd& scr_w, long long& iters, bool parallel) const {
        if (Jr == 0) {
            z_full.setZero();
            return true;
        }
#ifdef HDFE_USE_CUDA
        // GPU path only from single-threaded call sites (parallel == true):
        // the per-match exact loop calls with parallel == false and stays on
        // the CPU rather than serializing on the device.
        if (use_cuda && parallel) {
            const int it = akm_cuda_solve_S(cuda_ctx, rhs_full.data(),
                                            z_full.data(), cg_tol, cg_max_iter);
            if (it >= 0) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                const_cast<TwoWaySolver*>(this)->cuda_iters += it;
                return true;
            }
            return false;
        }
#endif
        if (direct) {
            Eigen::VectorXd z = ldlt.solve(rhs_full.head(Jr).eval());
            z_full.setZero();
            z_full.head(Jr) = z;
            return true;
        }
        Eigen::VectorXd x = Eigen::VectorXd::Zero(J);
        Eigen::VectorXd r = rhs_full;
        r[Jr] = 0.0;
        const double rhs_norm2 = r.squaredNorm();
        z_full.setZero();
        if (rhs_norm2 == 0.0) {
            return true;
        }
        Eigen::VectorXd z = inv_diagS.cwiseProduct(r);
        Eigen::VectorXd p = z;
        Eigen::VectorXd Ap(J);
        double rz = r.dot(z);
        const double tol2 = cg_tol * cg_tol * rhs_norm2;
        int it = 0;
        bool ok = false;
        for (; it < cg_max_iter; ++it) {
            // Ap = S p (grounded)
            mult_B(p.data(), scr_w.data(), parallel);
            scr_w.array() /= Dw.array();
            mult_Bt(scr_w.data(), Ap.data(), parallel);
            Ap = Df.cwiseProduct(p) - Ap;
            Ap[Jr] = 0.0;
            const double pAp = p.dot(Ap);
            if (!(pAp > 0.0) || !guarded_finite(pAp)) {
                break;
            }
            const double a = rz / pAp;
            x += a * p;
            r -= a * Ap;
            if (r.squaredNorm() <= tol2) {
                ++it;
                ok = true;
                break;
            }
            z = inv_diagS.cwiseProduct(r);
            const double rz_new = r.dot(z);
            p = z + (rz_new / rz) * p;
            rz = rz_new;
        }
        iters += it;
        if (!ok && r.squaredNorm() > tol2) {
            return false;
        }
        x[Jr] = 0.0;
        z_full = x;
        return true;
    }

    // Solve K z = (tw, tf) with the ground firm pinned to zero.
    bool solve_K(const Eigen::VectorXd& tw, const Eigen::VectorXd& tf,
                 Eigen::VectorXd& zw, Eigen::VectorXd& zf,
                 Eigen::VectorXd& scr_w, Eigen::VectorXd& scr_f,
                 long long& iters, bool parallel) const {
        scr_w = tw.cwiseQuotient(Dw);
        mult_Bt(scr_w.data(), scr_f.data(), parallel);
        scr_f = tf - scr_f;
        if (!solve_S(scr_f, zf, scr_w, iters, parallel)) {
            return false;
        }
        mult_B(zf.data(), scr_w.data(), parallel);
        zw = (tw - scr_w).cwiseQuotient(Dw);
        return true;
    }

    // ---- Batched independent multi-RHS solves --------------------------
    // The B / B' gathers are fused across columns (the graph indices and
    // coefficients are streamed once per tile of up to kMrhsTile columns,
    // and each gathered cache line serves every column in the tile), while
    // every per-column vector/scalar operation reuses exactly the same
    // Eigen expressions as the single-RHS path, applied in the same order.
    // Each column of a batched solve is therefore bit-identical to a
    // sequential solve_S / solve_K call for that RHS: the fused loops
    // accumulate per column in the same edge order as mult_B / mult_Bt,
    // and dot products / axpy updates stay per-column Eigen kernels.
    static constexpr int kMrhsTile = 8;

    // T = B X (worker space); X row-major J x kMrhsTile, T row-major
    // N x kMrhsTile. The tile width is a compile-time constant and the outer
    // loop runs over fixed-size 4096-row blocks, so both the generated code
    // and the work partition are identical for every call and every thread
    // count: a column's arithmetic never depends on how many real columns
    // share the tile (padding lanes carry zeros) nor on the OpenMP split.
    void mult_B_multi(const double* X, double* T, bool parallel) const {
        const std::vector<int>& mf = *m_f;
        parallel = parallel && N >= 16384;
        const std::int64_t nblk = (N + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && team > 1) num_threads(team)
#endif
        for (std::int64_t blk = 0; blk < nblk; ++blk) {
            const std::int64_t w_end = std::min<std::int64_t>((blk + 1) * 4096, N);
            for (std::int64_t w = blk * 4096; w < w_end; ++w) {
                double acc[kMrhsTile];
                for (int k = 0; k < kMrhsTile; ++k) {
                    acc[k] = 0.0;
                }
                for (std::int64_t a = w_ptr[static_cast<std::size_t>(w)];
                     a < w_ptr[static_cast<std::size_t>(w) + 1]; ++a) {
                    const double c = m_c[static_cast<std::size_t>(a)];
                    const double* xf =
                        X + static_cast<std::size_t>(mf[static_cast<std::size_t>(a)]) *
                                static_cast<std::size_t>(kMrhsTile);
                    for (int k = 0; k < kMrhsTile; ++k) {
                        acc[k] += c * xf[k];
                    }
                }
                double* tw =
                    T + static_cast<std::size_t>(w) * static_cast<std::size_t>(kMrhsTile);
                for (int k = 0; k < kMrhsTile; ++k) {
                    tw[k] = acc[k];
                }
            }
        }
    }

    // Y = B' S (firm space); S row-major N x kMrhsTile, Y row-major
    // J x kMrhsTile (fixed tile width + fixed 4096 blocks; see mult_B_multi).
    void mult_Bt_multi(const double* S, double* Y, bool parallel) const {
        const std::vector<int>& mw = *m_w;
        parallel = parallel && J >= 16384;
        const std::int64_t nblk = (J + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && team > 1) num_threads(team)
#endif
        for (std::int64_t blk = 0; blk < nblk; ++blk) {
            const std::int64_t f_end = std::min<std::int64_t>((blk + 1) * 4096, J);
            for (std::int64_t f = blk * 4096; f < f_end; ++f) {
                double acc[kMrhsTile];
                for (int k = 0; k < kMrhsTile; ++k) {
                    acc[k] = 0.0;
                }
                for (std::int64_t a = f_ptr[static_cast<std::size_t>(f)];
                     a < f_ptr[static_cast<std::size_t>(f) + 1]; ++a) {
                    const std::size_t m =
                        static_cast<std::size_t>(f_matches[static_cast<std::size_t>(a)]);
                    const double* sw =
                        S + static_cast<std::size_t>(mw[m]) * static_cast<std::size_t>(kMrhsTile);
                    const double c = m_c[m];
                    for (int k = 0; k < kMrhsTile; ++k) {
                        acc[k] += c * sw[k];
                    }
                }
                double* yf =
                    Y + static_cast<std::size_t>(f) * static_cast<std::size_t>(kMrhsTile);
                for (int k = 0; k < kMrhsTile; ++k) {
                    yf[k] = acc[k];
                }
            }
        }
    }

    // Apply Ap_c = S p_c (grounded) for a set of columns, fusing the two
    // gathers across columns in tiles. Per column the arithmetic sequence
    // matches solve_S exactly: mult_B, elementwise /Dw, mult_Bt, then
    // Ap = Df .* p - Ap with Ap[Jr] = 0.
    // The tile pack/unpack loops here and in solve_K_multi are parallelized
    // (perf audit 09jul2026, P3): they are pure element copies or per-lane
    // Eigen expressions evaluated exactly as in the sequential code, so any
    // work partition yields identical bits; they were a measurable serial
    // host cost per solve batch at 47.5M rows.
    void apply_S_multi(const std::vector<const Eigen::VectorXd*>& p,
                       std::vector<Eigen::VectorXd*>& Ap,
                       std::vector<double>& packP, std::vector<double>& packT,
                       bool parallel) const {
        const int nc = static_cast<int>(p.size());
        packP.assign(static_cast<std::size_t>(J) * kMrhsTile, 0.0);
        packT.resize(static_cast<std::size_t>(N) * kMrhsTile);
        for (int c0 = 0; c0 < nc; c0 += kMrhsTile) {
            const int nb = std::min(static_cast<int>(kMrhsTile), nc - c0);
            {
                const double* srcp[kMrhsTile] = {nullptr};
                for (int k = 0; k < nb; ++k) {
                    srcp[k] = p[static_cast<std::size_t>(c0 + k)]->data();
                }
                const std::int64_t nblkj = (J + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && J >= 16384 && team > 1) num_threads(team)
#endif
                for (std::int64_t blk = 0; blk < nblkj; ++blk) {
                    const std::int64_t j_end = std::min<std::int64_t>((blk + 1) * 4096, J);
                    for (std::int64_t j = blk * 4096; j < j_end; ++j) {
                        double* dst = packP.data() +
                                      static_cast<std::size_t>(j) * kMrhsTile;
                        for (int k = 0; k < nb; ++k) {
                            dst[k] = srcp[k][j];
                        }
                    }
                }
            }
            mult_B_multi(packP.data(), packT.data(), parallel);
            {
                double* T = packT.data();
                const std::int64_t nblkw = (N + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && N >= 16384 && team > 1) num_threads(team)
#endif
                for (std::int64_t blk = 0; blk < nblkw; ++blk) {
                    const std::int64_t w_end = std::min<std::int64_t>((blk + 1) * 4096, N);
                    for (std::int64_t w = blk * 4096; w < w_end; ++w) {
                        const double dw = Dw[w];
                        double* tw = T + static_cast<std::size_t>(w) * kMrhsTile;
                        for (int k = 0; k < kMrhsTile; ++k) {
                            tw[k] /= dw;
                        }
                    }
                }
            }
            mult_Bt_multi(packT.data(), packP.data(), parallel);
            // per-lane unpack + finish: each lane's Eigen expressions are
            // exactly the sequential ones (lane-parallel, disjoint outputs).
            // Gated on J like every other J-length region: this loop runs
            // once per CG step on the CPU path, and spawning a team for
            // small-J lanes costs more than the work (measured +35% on a
            // 330k-row panel at J=8k under load).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && nb > 1 && J >= 16384) num_threads(std::min(team, nb))
#endif
            for (int k = 0; k < nb; ++k) {
                Eigen::VectorXd& apc = *Ap[static_cast<std::size_t>(c0 + k)];
                for (std::int64_t j = 0; j < J; ++j) {
                    apc[j] = packP[static_cast<std::size_t>(j) * kMrhsTile +
                                   static_cast<std::size_t>(k)];
                }
                apc = Df.cwiseProduct(*p[static_cast<std::size_t>(c0 + k)]) - apc;
                apc[Jr] = 0.0;
            }
            if (c0 + kMrhsTile < nc) {
                const std::int64_t nblkj = (J + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && J >= 16384 && team > 1) num_threads(team)
#endif
                for (std::int64_t blk = 0; blk < nblkj; ++blk) {
                    const std::int64_t j_end = std::min<std::int64_t>((blk + 1) * 4096, J);
                    for (std::int64_t j = blk * 4096; j < j_end; ++j) {
                        double* dst = packP.data() +
                                      static_cast<std::size_t>(j) * kMrhsTile;
                        for (int k = 0; k < nb; ++k) {
                            dst[k] = 0.0;
                        }
                    }
                }
            }
        }
    }

    // Batched independent Jacobi-PCG: per-column state and scalar updates
    // are the exact single-RHS sequences; only the S-applies are fused.
    // Returns per-column success; z columns of failed solves stay zero.
    void solve_S_multi(std::vector<Eigen::VectorXd>& rhs,
                       std::vector<Eigen::VectorXd>& z_full,
                       std::vector<char>& success, long long& iters,
                       bool parallel) const {
        const int nc = static_cast<int>(rhs.size());
        success.assign(static_cast<std::size_t>(nc), 0);
        if (Jr == 0) {
            for (int c = 0; c < nc; ++c) {
                z_full[static_cast<std::size_t>(c)].setZero();
                success[static_cast<std::size_t>(c)] = 1;
            }
            return;
        }
#ifdef HDFE_USE_CUDA
        if (use_cuda && parallel) {
            // GPU path: one batched device solve per group of lanes (fewer
            // kernel launches and one host sync per CG step for the whole
            // group, instead of 3-4 per lane per step). The rhs columns are
            // packed straight into the context's pinned staging buffer when
            // available (page-locked DMA transfers; identical values), with
            // the old pageable vectors as fallback. rhs and z share the
            // staging buffer: the solve reads rhs fully before writing z.
            const int max_lanes = akm_cuda_max_lanes();
            std::vector<double> pack_fallback;
            std::vector<int> its;
            for (int c0 = 0; c0 < nc; c0 += max_lanes) {
                const int nb = std::min(max_lanes, nc - c0);
                double* pack = akm_cuda_pack_buffer(cuda_ctx, nb);
                if (pack == nullptr) {
                    pack_fallback.resize(static_cast<std::size_t>(J) * nb);
                    pack = pack_fallback.data();
                }
                its.assign(static_cast<std::size_t>(nb), -1);
                for (int k = 0; k < nb; ++k) {
                    std::memcpy(pack + static_cast<std::size_t>(k) * J,
                                rhs[static_cast<std::size_t>(c0 + k)].data(),
                                sizeof(double) * static_cast<std::size_t>(J));
                }
                const int rc = akm_cuda_solve_S_multi(cuda_ctx, pack, pack, nb,
                                                      cg_tol, cg_max_iter,
                                                      its.data());
                if (rc != 0) {
                    // device error: fall back to per-lane solves
                    for (int k = 0; k < nb; ++k) {
                        Eigen::VectorXd& r = rhs[static_cast<std::size_t>(c0 + k)];
                        Eigen::VectorXd& z = z_full[static_cast<std::size_t>(c0 + k)];
                        const int it = akm_cuda_solve_S(cuda_ctx, r.data(), z.data(),
                                                        cg_tol, cg_max_iter);
                        if (it >= 0) {
                            const_cast<TwoWaySolver*>(this)->cuda_iters += it;
                            success[static_cast<std::size_t>(c0 + k)] = 1;
                        } else {
                            z.setZero();
                        }
                    }
                    continue;
                }
                for (int k = 0; k < nb; ++k) {
                    Eigen::VectorXd& z = z_full[static_cast<std::size_t>(c0 + k)];
                    if (its[static_cast<std::size_t>(k)] >= 0) {
                        std::memcpy(z.data(),
                                    pack + static_cast<std::size_t>(k) * J,
                                    sizeof(double) * static_cast<std::size_t>(J));
                        const_cast<TwoWaySolver*>(this)->cuda_iters +=
                            its[static_cast<std::size_t>(k)];
                        success[static_cast<std::size_t>(c0 + k)] = 1;
                    } else {
                        z.setZero();
                    }
                }
            }
            return;
        }
#endif
        if (direct) {
            // Independent cached-LDLT backsolves into disjoint outputs: lane
            // order is irrelevant and each solve is deterministic, so the
            // parallel loop is bit-identical to the sequential one (perf
            // audit 09jul2026 — the direct multi-RHS branch was serial).
            // Team capped by the work-aware `team` (see the P1 cap above)
            // and the lane count: on small direct panels the sparse
            // backsolves are microseconds each and a full-team fork/barrier
            // per call dominated the SE/CI phases (measured 5-10x at 15k
            // rows, t1 == pre-batching); thread count cannot change bits.
#ifdef _OPENMP
            const int bs_team = std::min(nc, team);
#pragma omp parallel for schedule(static) \
    if (parallel && nc > 1 && bs_team > 1) num_threads(bs_team)
#endif
            for (int c = 0; c < nc; ++c) {
                Eigen::VectorXd z = ldlt.solve(rhs[static_cast<std::size_t>(c)].head(Jr).eval());
                z_full[static_cast<std::size_t>(c)].setZero();
                z_full[static_cast<std::size_t>(c)].head(Jr) = z;
                success[static_cast<std::size_t>(c)] = 1;
            }
            return;
        }

        struct Lane {
            Eigen::VectorXd x, r, z, p, Ap;
            double rz = 0.0, tol2 = 0.0;
            int it = 0;
            bool active = false, ok = false;
        };
        std::vector<Lane> lane(static_cast<std::size_t>(nc));
        for (int c = 0; c < nc; ++c) {
            Lane& L = lane[static_cast<std::size_t>(c)];
            L.x = Eigen::VectorXd::Zero(J);
            L.r = rhs[static_cast<std::size_t>(c)];
            L.r[Jr] = 0.0;
            const double rhs_norm2 = L.r.squaredNorm();
            z_full[static_cast<std::size_t>(c)].setZero();
            if (rhs_norm2 == 0.0) {
                success[static_cast<std::size_t>(c)] = 1;
                continue;
            }
            L.z = inv_diagS.cwiseProduct(L.r);
            L.p = L.z;
            L.Ap.resize(J);
            L.rz = L.r.dot(L.z);
            L.tol2 = cg_tol * cg_tol * rhs_norm2;
            L.active = true;
        }

        std::vector<double> packP, packT;
        std::vector<const Eigen::VectorXd*> pcols;
        std::vector<Eigen::VectorXd*> apcols;
        std::vector<int> idx;
        for (;;) {
            pcols.clear();
            apcols.clear();
            idx.clear();
            for (int c = 0; c < nc; ++c) {
                Lane& L = lane[static_cast<std::size_t>(c)];
                if (L.active && L.it < cg_max_iter) {
                    pcols.push_back(&L.p);
                    apcols.push_back(&L.Ap);
                    idx.push_back(c);
                } else if (L.active) {
                    // iteration budget exhausted without convergence
                    L.active = false;
                }
            }
            if (idx.empty()) {
                break;
            }
            apply_S_multi(pcols, apcols, packP, packT, parallel);
            for (std::size_t q = 0; q < idx.size(); ++q) {
                Lane& L = lane[static_cast<std::size_t>(idx[q])];
                const double pAp = L.p.dot(L.Ap);
                if (!(pAp > 0.0) || !guarded_finite(pAp)) {
                    L.active = false;
                    continue;
                }
                const double a = L.rz / pAp;
                L.x += a * L.p;
                L.r -= a * L.Ap;
                if (L.r.squaredNorm() <= L.tol2) {
                    ++L.it;
                    L.ok = true;
                    L.active = false;
                    continue;
                }
                L.z = inv_diagS.cwiseProduct(L.r);
                const double rz_new = L.r.dot(L.z);
                L.p = L.z + (rz_new / L.rz) * L.p;
                L.rz = rz_new;
                ++L.it;
            }
        }
        for (int c = 0; c < nc; ++c) {
            Lane& L = lane[static_cast<std::size_t>(c)];
            if (success[static_cast<std::size_t>(c)]) {
                continue;  // zero-rhs fast path already succeeded
            }
            iters += L.it;
            if (!L.ok && L.r.size() > 0 && L.r.squaredNorm() > L.tol2) {
                continue;  // failed: z stays zero, success stays 0
            }
            L.x[Jr] = 0.0;
            z_full[static_cast<std::size_t>(c)] = L.x;
            success[static_cast<std::size_t>(c)] = 1;
        }
    }

    // Batched solve_K: tw/tf may be nullptr to mean a zero vector. Per
    // column the sequence matches solve_K exactly.
    void solve_K_multi(const std::vector<const Eigen::VectorXd*>& tw,
                       const std::vector<const Eigen::VectorXd*>& tf,
                       std::vector<Eigen::VectorXd>& zw,
                       std::vector<Eigen::VectorXd>& zf,
                       std::vector<char>& success, long long& iters,
                       bool parallel) const {
        const int nc = static_cast<int>(tw.size());
        std::vector<Eigen::VectorXd> rhs(static_cast<std::size_t>(nc));
        {
            // Fused B' over the nc worker-space columns. Write tw/Dw
            // straight into the tile instead of creating and then copying
            // one N-vector per RHS (about 960 MB at 5M workers, 24 lanes).
            // Each quotient is still rounded once before B' consumes it;
            // nullptr lanes remain exact zero lanes.
            std::vector<double> packS(static_cast<std::size_t>(N) * kMrhsTile, 0.0);
            std::vector<double> packY(static_cast<std::size_t>(J) * kMrhsTile);
            for (int c0 = 0; c0 < nc; c0 += kMrhsTile) {
                const int nb = std::min(static_cast<int>(kMrhsTile), nc - c0);
                {
                    const double* srcp[kMrhsTile] = {nullptr};
                    for (int k = 0; k < nb; ++k) {
                        const Eigen::VectorXd* src =
                            tw[static_cast<std::size_t>(c0 + k)];
                        srcp[k] = src != nullptr ? src->data() : nullptr;
                    }
                    const std::int64_t nblkw = (N + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && N >= 16384 && team > 1) num_threads(team)
#endif
                    for (std::int64_t blk = 0; blk < nblkw; ++blk) {
                        const std::int64_t w_end =
                            std::min<std::int64_t>((blk + 1) * 4096, N);
                        for (std::int64_t w = blk * 4096; w < w_end; ++w) {
                            double* dst = packS.data() +
                                          static_cast<std::size_t>(w) * kMrhsTile;
                            for (int k = 0; k < nb; ++k) {
                                dst[k] = srcp[k] != nullptr
                                             ? srcp[k][w] / Dw[w]
                                             : 0.0;
                            }
                        }
                    }
                }
                mult_Bt_multi(packS.data(), packY.data(), parallel);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && nb > 1 && J >= 16384) num_threads(std::min(team, nb))
#endif
                for (int k = 0; k < nb; ++k) {
                    Eigen::VectorXd& r = rhs[static_cast<std::size_t>(c0 + k)];
                    r.resize(J);
                    for (std::int64_t j = 0; j < J; ++j) {
                        r[j] = packY[static_cast<std::size_t>(j) * kMrhsTile +
                                     static_cast<std::size_t>(k)];
                    }
                    if (tf[static_cast<std::size_t>(c0 + k)] != nullptr) {
                        r = *tf[static_cast<std::size_t>(c0 + k)] - r;
                    } else {
                        r = -r;
                    }
                }
                if (c0 + kMrhsTile < nc) {
                    const std::int64_t nblkw = (N + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && N >= 16384 && team > 1) num_threads(team)
#endif
                    for (std::int64_t blk = 0; blk < nblkw; ++blk) {
                        const std::int64_t w_end =
                            std::min<std::int64_t>((blk + 1) * 4096, N);
                        for (std::int64_t w = blk * 4096; w < w_end; ++w) {
                            double* dst = packS.data() +
                                          static_cast<std::size_t>(w) * kMrhsTile;
                            for (int k = 0; k < nb; ++k) {
                                dst[k] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        solve_S_multi(rhs, zf, success, iters, parallel);
        {
            std::vector<double> packZ(static_cast<std::size_t>(J) * kMrhsTile, 0.0);
            std::vector<double> packT(static_cast<std::size_t>(N) * kMrhsTile);
            for (int c0 = 0; c0 < nc; c0 += kMrhsTile) {
                const int nb = std::min(static_cast<int>(kMrhsTile), nc - c0);
                {
                    const double* srcp[kMrhsTile] = {nullptr};
                    for (int k = 0; k < nb; ++k) {
                        srcp[k] = zf[static_cast<std::size_t>(c0 + k)].data();
                    }
                    const std::int64_t nblkj = (J + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && J >= 16384 && team > 1) num_threads(team)
#endif
                    for (std::int64_t blk = 0; blk < nblkj; ++blk) {
                        const std::int64_t j_end =
                            std::min<std::int64_t>((blk + 1) * 4096, J);
                        for (std::int64_t j = blk * 4096; j < j_end; ++j) {
                            double* dst = packZ.data() +
                                          static_cast<std::size_t>(j) * kMrhsTile;
                            for (int k = 0; k < nb; ++k) {
                                dst[k] = srcp[k][j];
                            }
                        }
                    }
                }
                mult_B_multi(packZ.data(), packT.data(), parallel);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && nb > 1 && N >= 16384) num_threads(std::min(team, nb))
#endif
                for (int k = 0; k < nb; ++k) {
                    Eigen::VectorXd& w = zw[static_cast<std::size_t>(c0 + k)];
                    w.resize(N);
                    for (std::int64_t i = 0; i < N; ++i) {
                        w[i] = packT[static_cast<std::size_t>(i) * kMrhsTile +
                                     static_cast<std::size_t>(k)];
                    }
                    if (tw[static_cast<std::size_t>(c0 + k)] != nullptr) {
                        w = (*tw[static_cast<std::size_t>(c0 + k)] - w).cwiseQuotient(Dw);
                    } else {
                        w = (-w).cwiseQuotient(Dw);
                    }
                }
                if (c0 + kMrhsTile < nc) {
                    const std::int64_t nblkj = (J + 4095) / 4096;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel && J >= 16384 && team > 1) num_threads(team)
#endif
                    for (std::int64_t blk = 0; blk < nblkj; ++blk) {
                        const std::int64_t j_end =
                            std::min<std::int64_t>((blk + 1) * 4096, J);
                        for (std::int64_t j = blk * 4096; j < j_end; ++j) {
                            double* dst = packZ.data() +
                                          static_cast<std::size_t>(j) * kMrhsTile;
                            for (int k = 0; k < nb; ++k) {
                                dst[k] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
};

}  // namespace


// The batched SE/CI code paths below are outlined into noinline lambdas:
// they add a lot of code to an already-huge translation unit, and letting
// the optimizer inline them into the enclosing (fast-math) function bodies
// was observed to flip unrelated inlining/vectorization decisions and shift
// untouched loops (e.g. the sigma_i products) by one ulp between builds.
// Outlining keeps the legacy code's codegen byte-stable while the new
// blocks compile as separate functions.
#if defined(_MSC_VER)
#define XHDFE_AKM_OUTLINE
#else
#define XHDFE_AKM_OUTLINE __attribute__((noinline))
#endif

// Association-pinned scalar island: -fassociative-math lets the
// reassociation pass pick the evaluation order of a*b*(c/d), and its choice
// was observed to flip with translation-unit growth (2.14.x binaries
// realize ((a*b))*(c/d); the grown unit flips to (a*(c/d))*b, shifting
// sigma_i by one ulp between builds). This forces the shipped binaries'
// order contractually: GCC re-enables strict association per-function,
// clang gets optnone (value-safe), MSVC never reassociates parenthesized
// expressions under /fp:precise.
#if defined(_MSC_VER)
#define XHDFE_AKM_STRICT_FN
#elif defined(__clang__)
#define XHDFE_AKM_STRICT_FN __attribute__((noinline, optnone))
#else
#define XHDFE_AKM_STRICT_FN \
    __attribute__((noinline, optimize("-fno-associative-math")))
#endif

namespace {

// Mover-row sigma product in the association the shipped binaries realize
// (see XHDFE_AKM_STRICT_FN above).
XHDFE_AKM_STRICT_FN
double sigma_row_product(double yc, double corr_r, double eta_r, double m) {
    return yc * corr_r * (eta_r / m);
}

// SE/CI-phase batching block size (mirrors XHDFE_AKM_JLA_BLOCK): the
// component-SE simulations, the Hutchinson trace draws, the pencil subspace
// lanes, the A_b/xi solve pair and the lincom columns are routed in blocks
// through the batched multi-RHS solver. Per lane every scalar and vector
// operation keeps its exact sequential sequence (ascending sim/draw/column
// index), so results are identical for ANY block size >= 1 and ANY thread
// count — the same guarantee (and the same ~last-ulp difference vs the
// legacy single-RHS kernels) as the 2.14.0 JLA batching.
// XHDFE_AKM_SE_BLOCK=0 keeps every sequential single-RHS loop verbatim.
int akm_se_block_env() {
    int blk = 8;
    if (const char* sbe = std::getenv("XHDFE_AKM_SE_BLOCK")) {
        const int v = std::atoi(sbe);
        if (v >= 0 && v <= 64) {
            blk = v;
        }
    }
    return blk;
}

// Deterministic standard normal via Box-Muller on SplitMix64 uniforms.
inline double normal_from_stream(std::uint64_t& state) {
    state = splitmix64(state);
    const double u1 = (static_cast<double>(state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    state = splitmix64(state);
    const double u2 = (static_cast<double>(state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586476925286766559 * u2);
}

// group_equally (Altman-Bland percentile groups), replicated from the
// LeaveOutTwoWay helper: quantile cut points p_i at k_i*(N+1)/100 with
// linear interpolation, then sequential strict-below binning.
std::vector<int> group_equally(const std::vector<double>& x, int n_groups) {
    const std::size_t N = x.size();
    std::vector<double> s(x);
    std::sort(s.begin(), s.end());
    const int ng = n_groups;
    std::vector<double> cuts(static_cast<std::size_t>(ng - 1));
    for (int i = 1; i < ng; ++i) {
        const double k = 100.0 * i / ng;
        const double q = k * (static_cast<double>(N) + 1.0) / 100.0;
        long long w = static_cast<long long>(std::floor(q));
        const double f = q - static_cast<double>(w);
        if (w < 1) { w = 1; }
        if (w >= static_cast<long long>(N)) { w = static_cast<long long>(N) - 1; }
        cuts[static_cast<std::size_t>(i - 1)] =
            (1.0 - f) * s[static_cast<std::size_t>(w - 1)] +
            f * s[static_cast<std::size_t>(w)];
    }
    std::vector<int> g(N, ng);
    // The cuts are non-decreasing (interpolated quantiles of the sorted
    // sample) and the legacy O(ng*N) scan assigned row k the FIRST cut
    // strictly above x[k] (later passes could not reassign a binned row).
    // That is exactly std::upper_bound: bit-identical integer bins at
    // O(N log ng) instead of ~1e10 serial scalar ops on a 10M-row panel
    // (perf audit 09jul2026), parallel over rows (independent lookups), and
    // it removes the fast-math-hazardous infinity sentinel of the old loop.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N >= 16384)
#endif
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(N); ++k) {
        const std::size_t idx = static_cast<std::size_t>(
            std::upper_bound(cuts.begin(), cuts.end(),
                             x[static_cast<std::size_t>(k)]) -
            cuts.begin());
        if (idx < cuts.size()) {
            g[static_cast<std::size_t>(k)] = static_cast<int>(idx) + 1;
        }
    }
    return g;
}

}  // namespace


namespace {

// Andrews-Mikusheva q=1 confidence interval (AM_CI.m, literal port):
// nearest-grid critical value from the embedded tabulation, real roots of
// the quartic via the companion matrix, then the min/max bound envelope.
void am_confidence_interval(double NT, double lambda_1_raw, double gamma_sq,
                            double cov11, double cov12, double cov22,
                            double b_1, double theta_2,
                            double& LB, double& UB, double& C_out) {
    const double lambda_1 = lambda_1_raw / NT;
    const double gamma = std::sqrt(std::max(gamma_sq, 0.0));
    const double sigma_sq = cov11;
    const double tau_sq = cov22;
    const double rho = cov12 / (std::sqrt(sigma_sq) * std::sqrt(tau_sq));
    const double one_m_rho2 = std::max(1.0 - rho * rho, 1e-14);
    const double C = 2.0 * std::abs(gamma) / std::sqrt(one_m_rho2);
    C_out = C;
    double z_crit = kAmTabulation[kAmTabulationSize - 1][1];
    double best = 1e300;
    for (int i = 0; i < kAmTabulationSize; ++i) {
        const double d = std::abs(C - kAmTabulation[i][0]);
        if (d < best) {
            best = d;
            z_crit = kAmTabulation[i][1];
        }
    }
    const double sq_s = std::sqrt(sigma_sq);
    const double p1 = 4.0 * gamma_sq / sigma_sq;
    const double p2 = 4.0 * gamma * rho / sq_s - 8.0 * b_1 * gamma_sq / sigma_sq;
    const double p3 = 1.0 - 4.0 * gamma_sq * z_crit * z_crit +
                      4.0 * gamma_sq * b_1 * b_1 / sigma_sq -
                      8.0 * b_1 * rho * gamma / sq_s;
    const double p4 = 4.0 * b_1 * b_1 * gamma * rho / sq_s - 2.0 * b_1 -
                      4.0 * gamma * z_crit * z_crit * rho * sq_s;
    const double p5 = b_1 * b_1 - rho * rho * sigma_sq * z_crit * z_crit;
    // real roots of p1 x^4 + p2 x^3 + p3 x^2 + p4 x + p5
    std::vector<double> roots;
    if (std::abs(p1) > 1e-300) {
        Eigen::Matrix4d comp = Eigen::Matrix4d::Zero();
        comp(0, 3) = -p5 / p1;
        comp(1, 3) = -p4 / p1;
        comp(2, 3) = -p3 / p1;
        comp(3, 3) = -p2 / p1;
        comp(1, 0) = comp(2, 1) = comp(3, 2) = 1.0;
        Eigen::EigenSolver<Eigen::Matrix4d> es(comp);
        for (int i = 0; i < 4; ++i) {
            const std::complex<double> r = es.eigenvalues()[i];
            if (std::abs(r.imag()) <= 1e-8 * (1.0 + std::abs(r.real()))) {
                roots.push_back(r.real());
            }
        }
    }
    // finite sentinels: the module is compiled with -ffast-math, where
    // NaN self-comparisons are folded away
    LB = 1e300;
    UB = -1e300;
    for (double b : roots) {
        const double den = std::sqrt(
            one_m_rho2 + std::pow(2.0 * gamma * b / sq_s + rho, 2.0));
        const double base = lambda_1 * b * b + theta_2 -
                            rho * std::sqrt(tau_sq / sigma_sq) * (b_1 - b);
        const double lo = base - z_crit * std::sqrt(tau_sq) * one_m_rho2 / den;
        const double hi = base + z_crit * std::sqrt(tau_sq) * one_m_rho2 / den;
        if (lo < LB) LB = lo;
        if (hi > UB) UB = hi;
    }
    if (roots.empty()) {
        LB = std::numeric_limits<double>::quiet_NaN();
        UB = std::numeric_limits<double>::quiet_NaN();
    }
}

}  // namespace

namespace {

#ifdef _OPENMP
// Keep the per-call OpenMP override exception-safe. Without this guard an
// exception in AKM could silently change the thread limit seen by a later
// xhdfe/core23 fit in the same process.
struct ScopedOmpThreads {
    int previous = 1;
    explicit ScopedOmpThreads(int requested) : previous(omp_get_max_threads()) {
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }
    ~ScopedOmpThreads() { omp_set_num_threads(previous); }
};
#endif

std::vector<long long> validate_fweights(const Eigen::VectorXd& fwv,
                                         Eigen::Index n) {
    if (fwv.size() != n) {
        throw std::runtime_error("fweights must have the same length as y");
    }
    std::vector<long long> out(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        const double v = fwv[i];
        const long long w = static_cast<long long>(v);
        if (!(v >= 1.0) || static_cast<double>(w) != v) {
            throw std::runtime_error(
                "fweights must be positive integers (frequency weights)");
        }
        out[static_cast<std::size_t>(i)] = w;
    }
    return out;
}

}  // namespace

LeaveOutSetResult leave_out_connected_set(const Eigen::VectorXi& worker_ids,
                                          const Eigen::VectorXi& firm_ids,
                                          const Eigen::VectorXd* fweights) {
    if (fweights != nullptr) {
        const std::vector<long long> fw =
            validate_fweights(*fweights, worker_ids.size());
        return build_leave_out_sample(worker_ids, firm_ids, true, &fw).set;
    }
    return build_leave_out_sample(worker_ids, firm_ids, true).set;
}

AkmKssResult akm_kss_decompose(const Eigen::VectorXd& y,
                               const Eigen::VectorXi& worker_ids,
                               const Eigen::VectorXi& firm_ids,
                               const Eigen::MatrixXd* X,
                               const AkmOptions& options,
                               const Eigen::MatrixXd* Z,
                               const Eigen::VectorXd* fweights) {
    if (y.size() != worker_ids.size() || y.size() != firm_ids.size()) {
        throw std::runtime_error("y, worker and firm vectors must have the same length");
    }
    if (X != nullptr && X->rows() != y.size()) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (options.jla_draws < 1) {
        throw std::runtime_error("jla_draws must be >= 1");
    }

#ifdef _OPENMP
    ScopedOmpThreads omp_threads(options.num_threads);
#endif

    std::vector<long long> fw_rows;
    const bool has_fw = fweights != nullptr;
    if (has_fw) {
        fw_rows = validate_fweights(*fweights, y.size());
        if (options.leave_out_level != LeaveOutLevel::Match) {
            throw std::runtime_error(
                "fweights require leave_out_level=match; expand the data for "
                "observation-level leave-out (identical results)");
        }
        if (options.compute_se || options.eigen_diagnostics) {
            throw std::runtime_error(
                "fweights with compute_se/eigen_diagnostics are not supported "
                "yet; expand the data (identical results by construction)");
        }
        if (Z != nullptr) {
            throw std::runtime_error(
                "fweights with Z (lincom) are not supported yet; expand the "
                "data (identical results by construction)");
        }
    }

    AkmKssResult res;
    AkmPhaseProfiler prof_;
    res.seed_used = options.seed;
    long long cg_iters = 0;
    SampleBuild sb = build_leave_out_sample(worker_ids, firm_ids, options.prune,
                                            has_fw ? &fw_rows : nullptr);
    res.sample = std::move(sb.set);
    prof_.mark("leave_out_sample");

    const std::size_t n_kept = sb.rows.size();
    const std::size_t M = sb.m_w.size();
    const int N = sb.n_workers;
    const int J = sb.n_firms;
    const bool match_level = options.leave_out_level == LeaveOutLevel::Match;
    const std::size_t R = match_level ? M : n_kept;  // working rows
    double n_py_ll = 0.0;
    for (std::size_t m = 0; m < M; ++m) {
        n_py_ll += static_cast<double>(sb.m_cnt[m]);
    }
    const double n_py = n_py_ll;  // person-year count (= n_kept unweighted)
    const bool rademacher_tot_int_exact =
        n_py < 9007199254740992.0;  // strict gate avoids a rounded 2^53+1 total
    // Keep this empty in the common unweighted case. Materialising an
    // all-ones vector costs about 380 MB on a 47.5M-row panel.
    Eigen::VectorXd kw;
    if (has_fw) {
        kw.resize(static_cast<Eigen::Index>(n_kept));
        for (std::size_t k = 0; k < n_kept; ++k) {
            kw[static_cast<Eigen::Index>(k)] =
                static_cast<double>(fw_rows[static_cast<std::size_t>(sb.rows[k])]);
        }
    }

    if (n_kept < 3 || N < 1 || J < 1 || n_py - 1.0 <= 0.0) {
        res.converged = false;
        res.notes = "leave-out sample too small for a variance decomposition. ";
        return res;
    }

    // ---- Person-year arrays on the kept sample ----------------------------
    Eigen::VectorXd ystar(static_cast<Eigen::Index>(n_kept));
    for (std::size_t k = 0; k < n_kept; ++k) {
        ystar[static_cast<Eigen::Index>(k)] = y[sb.rows[k]];
    }
    // var_y (LeaveOutTwoWay's var_den): person-year variance before controls.
    {
        const double mean_y = has_fw ? kw.dot(ystar) / n_py : ystar.mean();
        res.var_y =
            has_fw
                ? (kw.array() * (ystar.array() - mean_y).square()).sum() /
                      (n_py - 1.0)
                : (ystar.array() - mean_y).square().sum() / (n_py - 1.0);
    }

    // FWL: partial controls out at the person-year level via the absorber.
    if (X != nullptr && X->cols() > 0) {
        Eigen::MatrixXd Xk(static_cast<Eigen::Index>(n_kept), X->cols());
        for (std::size_t k = 0; k < n_kept; ++k) {
            Xk.row(static_cast<Eigen::Index>(k)) = X->row(sb.rows[k]);
        }
        std::vector<Eigen::VectorXi> fes(2);
        fes[0].resize(static_cast<Eigen::Index>(n_kept));
        fes[1].resize(static_cast<Eigen::Index>(n_kept));
        for (std::size_t k = 0; k < n_kept; ++k) {
            fes[0][static_cast<Eigen::Index>(k)] = sb.row_w[k];
            fes[1][static_cast<Eigen::Index>(k)] = sb.row_f[k];
        }
        HdfeOptions ho;
        ho.se_type = StandardErrorType::Homoskedastic;
        ho.fit_intercept = false;
        ho.drop_singletons = false;
        ho.tol = options.fwl_tol;
        ho.max_iter = options.fwl_max_iter;
        ho.num_threads = options.num_threads;
        v11::HdfeRegressorV11 reg(ho);
        reg.fit(ystar, Xk, fes, has_fw ? &kw : nullptr);
        if (!reg.results().converged) {
            res.converged = false;
            res.notes += "control partialling (FWL) absorber did not converge. ";
        }
        res.beta = reg.results().coefficients;
        ystar.noalias() -= Xk * res.beta;
        prof_.mark("fwl_controls");
    }

    // ---- Working rows ------------------------------------------------------
    Eigen::VectorXd m_ysum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
    for (std::size_t k = 0; k < n_kept; ++k) {
        m_ysum[sb.row_match[k]] +=
            (has_fw ? kw[static_cast<Eigen::Index>(k)] : 1.0) *
            ystar[static_cast<Eigen::Index>(k)];
    }

    long long n_input_py = static_cast<long long>(y.size());
    if (has_fw) {
        n_input_py = 0;
        for (std::size_t i = 0; i < fw_rows.size(); ++i) n_input_py += fw_rows[i];
    }
    const bool use_exact =
        options.leverage_method == LeverageMethod::Exact ||
        (options.leverage_method == LeverageMethod::Auto &&
         n_input_py <= options.exact_max_rows);
    res.leverages_exact = use_exact;

    TwoWaySolver solver;
    solver.build(N, J, sb.m_w, sb.m_f, sb.m_cnt, options, res.notes);
    if (match_level && !use_exact) solver.ensure_sqrt_c();
    res.solver_direct = solver.direct;
    res.gpu_used = solver.use_cuda;
    res.n_rows = static_cast<long long>(R);
    prof_.mark("solver_build");

    // Point estimates: person-year OLS normal equations.
    Eigen::VectorXd alpha(N);
    Eigen::VectorXd psi(J);
    {
        Eigen::VectorXd tw = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd tf = Eigen::VectorXd::Zero(J);
        for (std::size_t m = 0; m < M; ++m) {
            tw[sb.m_w[m]] += m_ysum[static_cast<Eigen::Index>(m)];
            tf[sb.m_f[m]] += m_ysum[static_cast<Eigen::Index>(m)];
        }
        Eigen::VectorXd scr_w(N);
        Eigen::VectorXd scr_f(J);
        if (!solver.solve_K(tw, tf, alpha, psi, scr_w, scr_f, cg_iters, true)) {
            res.converged = false;
            res.notes += "two-way point-estimation solve did not converge. ";
        }
    }
    prof_.mark("point_solve");

    // Working-row transforms (sqrt-weight FGLS at match level).
    const std::vector<int>& row_w_of = match_level ? sb.m_w : sb.row_w;
    const std::vector<int>& row_f_of = match_level ? sb.m_f : sb.row_f;
    Eigen::VectorXd row_wgt(static_cast<Eigen::Index>(R));
    Eigen::VectorXd yt(static_cast<Eigen::Index>(R));
    Eigen::VectorXd eta(static_cast<Eigen::Index>(R));
    std::vector<std::uint8_t> row_stayer(R, 0);
    {
        std::vector<int> worker_deg(static_cast<std::size_t>(N), 0);
        for (std::size_t m = 0; m < M; ++m) {
            ++worker_deg[static_cast<std::size_t>(sb.m_w[m])];
        }
        if (match_level) {
            for (std::size_t m = 0; m < M; ++m) {
                const double c = static_cast<double>(sb.m_cnt[m]);
                const double sc = !solver.m_sqrt_c.empty()
                                      ? solver.m_sqrt_c[m]
                                      : std::sqrt(c);
                row_wgt[static_cast<Eigen::Index>(m)] = c;
                yt[static_cast<Eigen::Index>(m)] = sc * (m_ysum[static_cast<Eigen::Index>(m)] / c);
                eta[static_cast<Eigen::Index>(m)] =
                    yt[static_cast<Eigen::Index>(m)] - sc * (alpha[sb.m_w[m]] + psi[sb.m_f[m]]);
                row_stayer[m] = worker_deg[static_cast<std::size_t>(sb.m_w[m])] < 2 ? 1 : 0;
            }
        } else {
            for (std::size_t k = 0; k < n_kept; ++k) {
                row_wgt[static_cast<Eigen::Index>(k)] = 1.0;
                yt[static_cast<Eigen::Index>(k)] = ystar[static_cast<Eigen::Index>(k)];
                eta[static_cast<Eigen::Index>(k)] =
                    ystar[static_cast<Eigen::Index>(k)] - alpha[sb.row_w[k]] - psi[sb.row_f[k]];
            }
        }
    }
    const double mean_yt = yt.mean();

    // ---- Leverages (Pii) and quadratic-form weights (Bii) ------------------
    Eigen::VectorXd Pii(static_cast<Eigen::Index>(R));
    Eigen::VectorXd Mii(static_cast<Eigen::Index>(R));
    Eigen::VectorXd corr = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(R));
    Eigen::VectorXd Bfe(static_cast<Eigen::Index>(R));
    Eigen::VectorXd Bpe(static_cast<Eigen::Index>(R));
    Eigen::VectorXd Bcov(static_cast<Eigen::Index>(R));

    bool solve_failed = false;
    if (use_exact) {
        // One unweighted solve per match; all rows of a match share it.
        Eigen::VectorXd base_p(static_cast<Eigen::Index>(M));
        Eigen::VectorXd base_fe(static_cast<Eigen::Index>(M));
        Eigen::VectorXd base_pe(static_cast<Eigen::Index>(M));
        Eigen::VectorXd base_cov(static_cast<Eigen::Index>(M));
        long long iter_sum = 0;
        bool failed_local = false;
#ifdef _OPENMP
#pragma omp parallel reduction(+ : iter_sum) reduction(|| : failed_local)
#endif
        {
            Eigen::VectorXd tw = Eigen::VectorXd::Zero(N);
            Eigen::VectorXd tf = Eigen::VectorXd::Zero(J);
            Eigen::VectorXd zw(N), zf(J), scr_w(N), scr_f(J);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
            for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
                const int w = sb.m_w[static_cast<std::size_t>(m)];
                const int f = sb.m_f[static_cast<std::size_t>(m)];
                tw[w] = 1.0;
                tf[f] = 1.0;
                if (!solver.solve_K(tw, tf, zw, zf, scr_w, scr_f, iter_sum, false)) {
                    failed_local = true;
                }
                tw[w] = 0.0;
                tf[f] = 0.0;
                base_p[m] = zw[w] + zf[f];
                double s1 = 0.0, s2 = 0.0;
                for (int ff = 0; ff < J; ++ff) {
                    s1 += solver.Df[ff] * zf[ff] * zf[ff];
                    s2 += solver.Df[ff] * zf[ff];
                }
                double s3 = 0.0, s4 = 0.0;
                for (int ww = 0; ww < N; ++ww) {
                    s3 += solver.Dw[ww] * zw[ww] * zw[ww];
                    s4 += solver.Dw[ww] * zw[ww];
                }
                double cross = 0.0;
                for (std::size_t mm = 0; mm < M; ++mm) {
                    cross += solver.m_c[mm] * zw[sb.m_w[mm]] * zf[sb.m_f[mm]];
                }
                base_fe[m] = s1 - s2 * s2 / n_py;
                base_pe[m] = s3 - s4 * s4 / n_py;
                base_cov[m] = cross - s4 * s2 / n_py;
            }
        }
        cg_iters += iter_sum;
        solve_failed = solve_failed || failed_local;
        if (match_level) {
            for (std::size_t m = 0; m < M; ++m) {
                const double c = static_cast<double>(sb.m_cnt[m]);
                Pii[static_cast<Eigen::Index>(m)] = c * base_p[static_cast<Eigen::Index>(m)];
                Bfe[static_cast<Eigen::Index>(m)] = c * base_fe[static_cast<Eigen::Index>(m)];
                Bpe[static_cast<Eigen::Index>(m)] = c * base_pe[static_cast<Eigen::Index>(m)];
                Bcov[static_cast<Eigen::Index>(m)] = c * base_cov[static_cast<Eigen::Index>(m)];
            }
        } else {
            for (std::size_t k = 0; k < n_kept; ++k) {
                const int m = sb.row_match[k];
                Pii[static_cast<Eigen::Index>(k)] = base_p[m];
                Bfe[static_cast<Eigen::Index>(k)] = base_fe[m];
                Bpe[static_cast<Eigen::Index>(k)] = base_pe[m];
                Bcov[static_cast<Eigen::Index>(k)] = base_cov[m];
            }
        }
        Mii = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(R)) - Pii;
    } else {
        // JLA path: sequential draws with inner parallelism; deterministic
        // per-(draw, unit) SplitMix64 Rademacher streams, thread-count
        // independent accumulation (per-row bins, sequential scalar passes).
        const int p = options.jla_draws;
        res.jla_draws_used = p;
        Eigen::VectorXd Pacc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(R));
        Eigen::VectorXd Psq = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(R));
        Eigen::VectorXd Macc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(R));
        Eigen::VectorXd Msq = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(R));
        Eigen::VectorXd PM = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(R));
        Bfe.setZero();
        Bpe.setZero();
        Bcov.setZero();

        std::vector<std::int8_t> rvec(R);
        Eigen::VectorXd msum(static_cast<Eigen::Index>(M));
        Eigen::VectorXd tw(N), tf(J), zw(N), zf(J), scr_w(N), scr_f(J);
        Eigen::VectorXd zw_fe(N), zf_fe(J), zw_pe(N), zf_pe(J);
        const double pd = static_cast<double>(p);
        const double scale = 1.0 / std::sqrt(pd);
        const std::int64_t n_blocks = static_cast<std::int64_t>((R + 63) / 64);
        const bool par_rows = R >= 16384;
        (void)par_rows;

        // Draw-block batching: the p draws are processed in blocks of DB and
        // every draw's 3 solves go through the batched multi-RHS solver. Per
        // draw the Rademacher streams, scatters and accumulation order are
        // exactly the sequential sequences (draw-indexed SplitMix64 streams
        // make generation order irrelevant; per-row accumulation keeps
        // ascending-d order), and the fused kernels use fixed tile widths and
        // fixed work blocks, so results are identical for ANY block size >= 1
        // and ANY thread count. XHDFE_AKM_JLA_BLOCK=0 selects the pre-2.14
        // sequential single-RHS loop kept verbatim below (bit-reproduces
        // older builds; its per-edge instruction schedule differs from the
        // fused kernels by ~1 ulp PER SOLVE — note that at draws=1 the
        // Pii/(Pii+Mii) ratio estimator can amplify that last-ulp solver
        // difference to ~1e-10 in the components through the near-zero
        // 1-draw Mii denominator; at draws >= 2 and at the shipped default
        // of 200 the block=0-vs-batched difference is at machine epsilon.
        // The block>=1 bit-invariance guarantee is CPU-scoped: the CUDA
        // batched solver is bit-deterministic run-to-run but its CUB
        // segmented-reduce shapes vary with the lane-group size, so GPU
        // results differ across block sizes at the ~1e-14 level). The
        // direct (cached-LDLT) mode also defaults to the batched path since
        // 2.15.0: fusion gains it nothing per solve, but the batched direct
        // branch runs its independent backsolves in parallel (perf audit
        // 09jul2026 — the legacy sequential loop made small direct-path
        // panels the slowest per row; the ~1-ulp block=0-vs-batched note
        // above applies unchanged, and XHDFE_AKM_JLA_BLOCK=0 still restores
        // the pre-2.14 sequential loop bitwise). The CUDA mode batches the
        // draw solves on the device.
        int jla_block = 8;
        if (const char* jbe = std::getenv("XHDFE_AKM_JLA_BLOCK")) {
            const int v = std::atoi(jbe);
            if (v >= 0 && v <= 64) {
                jla_block = v;
            }
        }
        if (jla_block >= 1) {
            const int DB = jla_block;
            std::vector<Eigen::VectorXd> tw1(static_cast<std::size_t>(DB));
            std::vector<Eigen::VectorXd> tf1(static_cast<std::size_t>(DB));
            std::vector<Eigen::VectorXd> tw2(static_cast<std::size_t>(DB));
            std::vector<Eigen::VectorXd> tf2(static_cast<std::size_t>(DB));
            std::vector<Eigen::VectorXd> zw_all(static_cast<std::size_t>(3 * DB));
            std::vector<Eigen::VectorXd> zf_all(static_cast<std::size_t>(3 * DB));
            std::vector<char> okv;
            std::vector<const Eigen::VectorXd*> tw_ptr(static_cast<std::size_t>(3 * DB));
            std::vector<const Eigen::VectorXd*> tf_ptr(static_cast<std::size_t>(3 * DB));

            // CSR-parallel per-draw scatters (perf audit 09jul2026, P3): the
            // sequential ascending-m scatter loops were the last O(M) serial
            // work in the block prep — at 47.5M rows on the GPU path they
            // left the device idle ~40% of the leverage-phase wall. They are
            // parallelized over the solver's worker/firm CSR structures with
            // the SAME per-slot addition sequences: each tw slot is owned by
            // exactly one worker whose matches are visited ascending in m
            // (matches are worker-major sorted, so w_ptr spans ascend in m),
            // and each tf slot by exactly one firm whose f_matches entries
            // ascend in m (f_matches is a counting sort over ascending m).
            // The scattered values are precomputed elementwise into draw_val
            // (one rounded expression per element, no accumulation, safe to
            // vectorize), and the scatter adds keep the indirect
            // `slot[index[m]] += draw_val[m]` form of the legacy loop so the
            // compiler cannot contract or reassociate them under
            // -ffast-math. Results are therefore bit-identical to the
            // sequential scatters for any thread count (verified vs the
            // pre-P3 module and vs XHDFE_AKM_SCATTER_CSR=0 in-binary).
            // XHDFE_AKM_SCATTER_CSR=0 restores the sequential loops.
            bool csr_scatter = true;
            if (const char* cse = std::getenv("XHDFE_AKM_SCATTER_CSR")) {
                csr_scatter = std::atoi(cse) != 0;
            }
            // Use the CSR-parallel scatters only when the work amortizes the
            // per-draw region spawns (6 per draw): below ~2M working rows
            // the legacy serial scatter is ~1ms/draw while a team wake-up on
            // a contended box can cost several ms (measured +20% on a 330k
            // panel at load 74), and the serial CSR traversal itself pays an
            // extra indirection over the legacy loop. Below the threshold
            // the ORIGINAL sequential loops run unchanged; above it the CSR
            // loops produce the identical bits (same per-slot sequences).
            const bool par_scatter =
                csr_scatter && par_rows && solver.team > 1 && R >= 2097152;
            const bool par_scatter_m =
                csr_scatter && par_rows && solver.team > 1 && M >= 2097152;
            Eigen::VectorXd draw_val;
            if (par_scatter || par_scatter_m) {
                draw_val.resize(static_cast<Eigen::Index>(M));
            }
            // Observation-level rows have no CSR; build one per side once
            // per call (counting sort over ascending row order — the same
            // construction as f_matches) so the obs-level +/-1 scatter gets
            // per-slot ownership too. The +/-1 partial sums are integers
            // (every add is exact), so ownership alone already guarantees
            // bit-identical slots; ascending order is preserved anyway.
            std::vector<std::int64_t> rw_ptr, rf_ptr, rw_rows, rf_rows;
            if (par_scatter && !match_level) {
                rw_ptr.assign(static_cast<std::size_t>(N) + 1, 0);
                rf_ptr.assign(static_cast<std::size_t>(J) + 1, 0);
                for (std::size_t k = 0; k < n_kept; ++k) {
                    ++rw_ptr[static_cast<std::size_t>(sb.row_w[k]) + 1];
                    ++rf_ptr[static_cast<std::size_t>(sb.row_f[k]) + 1];
                }
                for (std::size_t w = 0; w < static_cast<std::size_t>(N); ++w) {
                    rw_ptr[w + 1] += rw_ptr[w];
                }
                for (std::size_t f = 0; f < static_cast<std::size_t>(J); ++f) {
                    rf_ptr[f + 1] += rf_ptr[f];
                }
                rw_rows.resize(n_kept);
                rf_rows.resize(n_kept);
                std::vector<std::int64_t> pw(rw_ptr.begin(), rw_ptr.end() - 1);
                std::vector<std::int64_t> pf(rf_ptr.begin(), rf_ptr.end() - 1);
                for (std::size_t k = 0; k < n_kept; ++k) {
                    rw_rows[static_cast<std::size_t>(
                        pw[static_cast<std::size_t>(sb.row_w[k])]++)] =
                        static_cast<std::int64_t>(k);
                    rf_rows[static_cast<std::size_t>(
                        pf[static_cast<std::size_t>(sb.row_f[k])]++)] =
                        static_cast<std::int64_t>(k);
                }
            }

            for (int d0 = 0; d0 < p; d0 += DB) {
                const int B = std::min(DB, p - d0);
                // --- per-draw scatters (legacy sequences, block-local) -----
                for (int b = 0; b < B; ++b) {
                    const int d = d0 + b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
                    for (std::int64_t blk = 0; blk < n_blocks; ++blk) {
                        const std::uint64_t bits =
                            stream_seed(options.seed, static_cast<std::uint64_t>(3 * d),
                                        static_cast<std::uint64_t>(blk));
                        const std::size_t base = static_cast<std::size_t>(blk) * 64;
                        const std::size_t hi = std::min(base + 64, R);
                        for (std::size_t r = base; r < hi; ++r) {
                            rvec[r] = ((bits >> (r - base)) & 1ULL) ? 1 : -1;
                        }
                    }
                    Eigen::VectorXd& tw_b = tw1[static_cast<std::size_t>(b)];
                    Eigen::VectorXd& tf_b = tf1[static_cast<std::size_t>(b)];
                    tw_b = Eigen::VectorXd::Zero(N);
                    tf_b = Eigen::VectorXd::Zero(J);
                    if (par_scatter && match_level) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter) num_threads(solver.team)
#endif
                        for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
                            draw_val[m] =
                                solver.m_sqrt_c[static_cast<std::size_t>(m)] *
                                static_cast<double>(rvec[static_cast<std::size_t>(m)]);
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter) num_threads(solver.team)
#endif
                        for (std::int64_t w = 0; w < static_cast<std::int64_t>(N); ++w) {
                            for (std::int64_t a = solver.w_ptr[static_cast<std::size_t>(w)];
                                 a < solver.w_ptr[static_cast<std::size_t>(w) + 1]; ++a) {
                                tw_b[sb.m_w[static_cast<std::size_t>(a)]] += draw_val[a];
                            }
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter) num_threads(solver.team)
#endif
                        for (std::int64_t f = 0; f < static_cast<std::int64_t>(J); ++f) {
                            for (std::int64_t a = solver.f_ptr[static_cast<std::size_t>(f)];
                                 a < solver.f_ptr[static_cast<std::size_t>(f) + 1]; ++a) {
                                const std::int64_t m =
                                    solver.f_matches[static_cast<std::size_t>(a)];
                                tf_b[sb.m_f[static_cast<std::size_t>(m)]] += draw_val[m];
                            }
                        }
                    } else if (par_scatter) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter) num_threads(solver.team)
#endif
                        for (std::int64_t w = 0; w < static_cast<std::int64_t>(N); ++w) {
                            for (std::int64_t a = rw_ptr[static_cast<std::size_t>(w)];
                                 a < rw_ptr[static_cast<std::size_t>(w) + 1]; ++a) {
                                const std::size_t k =
                                    static_cast<std::size_t>(rw_rows[static_cast<std::size_t>(a)]);
                                tw_b[sb.row_w[k]] += static_cast<double>(rvec[k]);
                            }
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter) num_threads(solver.team)
#endif
                        for (std::int64_t f = 0; f < static_cast<std::int64_t>(J); ++f) {
                            for (std::int64_t a = rf_ptr[static_cast<std::size_t>(f)];
                                 a < rf_ptr[static_cast<std::size_t>(f) + 1]; ++a) {
                                const std::size_t k =
                                    static_cast<std::size_t>(rf_rows[static_cast<std::size_t>(a)]);
                                tf_b[sb.row_f[k]] += static_cast<double>(rvec[k]);
                            }
                        }
                    } else if (match_level) {
                        for (std::size_t m = 0; m < M; ++m) {
                            const double v =
                                solver.m_sqrt_c[m] * static_cast<double>(rvec[m]);
                            tw_b[sb.m_w[m]] += v;
                            tf_b[sb.m_f[m]] += v;
                        }
                    } else {
                        for (std::size_t k = 0; k < n_kept; ++k) {
                            const double v = static_cast<double>(rvec[k]);
                            tw_b[sb.row_w[k]] += v;
                            tf_b[sb.row_f[k]] += v;
                        }
                    }
                    long long tot_i = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team) reduction(+ : tot_i)
#endif
                    for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
                        msum[m] = rademacher_sum(
                            stream_seed(options.seed,
                                        static_cast<std::uint64_t>(3 * d + 1),
                                        static_cast<std::uint64_t>(m)),
                            sb.m_cnt[static_cast<std::size_t>(m)]);
                        if (rademacher_tot_int_exact) {
                            tot_i += static_cast<long long>(msum[m]);
                        }
                    }
                    // Every Rademacher sum is integral. The integer
                    // reduction is exact and thread-count invariant; its
                    // final double conversion matches the legacy ascending
                    // sum for all representable person-year counts.
                    double tot = static_cast<double>(tot_i);
                    if (!rademacher_tot_int_exact) {
                        // Preserve the legacy FP accumulation for theoretical
                        // frequency-weight totals beyond exact-double range.
                        tot = 0.0;
                        for (std::size_t m = 0; m < M; ++m) {
                            tot += msum[static_cast<Eigen::Index>(m)];
                        }
                    }
                    const double vbar = (tot / n_py) * scale;
                    Eigen::VectorXd& tw_c = tw2[static_cast<std::size_t>(b)];
                    Eigen::VectorXd& tf_c = tf2[static_cast<std::size_t>(b)];
                    tw_c = Eigen::VectorXd::Zero(N);
                    tf_c = Eigen::VectorXd::Zero(J);
                    if (par_scatter_m) {
                        // su is match-indexed in both leave-out levels; the
                        // elementwise expression below is textually the
                        // legacy `su` (same contraction), evaluated once per
                        // match, then scattered through the match CSR.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter_m) num_threads(solver.team)
#endif
                        for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
                            draw_val[m] =
                                msum[static_cast<Eigen::Index>(m)] * scale -
                                vbar * static_cast<double>(
                                           sb.m_cnt[static_cast<std::size_t>(m)]);
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter_m) num_threads(solver.team)
#endif
                        for (std::int64_t w = 0; w < static_cast<std::int64_t>(N); ++w) {
                            for (std::int64_t a = solver.w_ptr[static_cast<std::size_t>(w)];
                                 a < solver.w_ptr[static_cast<std::size_t>(w) + 1]; ++a) {
                                tw_c[sb.m_w[static_cast<std::size_t>(a)]] += draw_val[a];
                            }
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_scatter_m) num_threads(solver.team)
#endif
                        for (std::int64_t f = 0; f < static_cast<std::int64_t>(J); ++f) {
                            for (std::int64_t a = solver.f_ptr[static_cast<std::size_t>(f)];
                                 a < solver.f_ptr[static_cast<std::size_t>(f) + 1]; ++a) {
                                const std::int64_t m =
                                    solver.f_matches[static_cast<std::size_t>(a)];
                                tf_c[sb.m_f[static_cast<std::size_t>(m)]] += draw_val[m];
                            }
                        }
                    } else {
                        for (std::size_t m = 0; m < M; ++m) {
                            const double su = msum[static_cast<Eigen::Index>(m)] * scale -
                                              vbar * static_cast<double>(sb.m_cnt[m]);
                            tw_c[sb.m_w[m]] += su;
                            tf_c[sb.m_f[m]] += su;
                        }
                    }
                }
                // --- batched solves: lanes [0,B)=(tw1,tf1) Pii draw;
                //     [B,2B)=(0,tf2) fe; [2B,3B)=(tw2,0) pe ---------------
                const int nl = 3 * B;
                for (int b = 0; b < B; ++b) {
                    tw_ptr[static_cast<std::size_t>(b)] = &tw1[static_cast<std::size_t>(b)];
                    tf_ptr[static_cast<std::size_t>(b)] = &tf1[static_cast<std::size_t>(b)];
                    tw_ptr[static_cast<std::size_t>(B + b)] = nullptr;
                    tf_ptr[static_cast<std::size_t>(B + b)] = &tf2[static_cast<std::size_t>(b)];
                    tw_ptr[static_cast<std::size_t>(2 * B + b)] = &tw2[static_cast<std::size_t>(b)];
                    tf_ptr[static_cast<std::size_t>(2 * B + b)] = nullptr;
                }
                std::vector<const Eigen::VectorXd*> tws(tw_ptr.begin(), tw_ptr.begin() + nl);
                std::vector<const Eigen::VectorXd*> tfs(tf_ptr.begin(), tf_ptr.begin() + nl);
                for (int c = 0; c < nl; ++c) {
                    zw_all[static_cast<std::size_t>(c)].resize(N);
                    zf_all[static_cast<std::size_t>(c)].resize(J);
                }
                solver.solve_K_multi(tws, tfs, zw_all, zf_all, okv, cg_iters, true);
                for (int c = 0; c < nl; ++c) {
                    if (!okv[static_cast<std::size_t>(c)]) {
                        solve_failed = true;
                    }
                }
                // --- fused accumulation, ascending d inside each row -------
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
                for (std::int64_t blk = 0; blk < n_blocks; ++blk) {
                    const std::size_t base = static_cast<std::size_t>(blk) * 64;
                    const std::size_t hi = std::min(base + 64, R);
                    for (int b = 0; b < B; ++b) {
                        const int d = d0 + b;
                        const std::uint64_t bits =
                            stream_seed(options.seed, static_cast<std::uint64_t>(3 * d),
                                        static_cast<std::uint64_t>(blk));
                        const Eigen::VectorXd& zw_b = zw_all[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd& zf_b = zf_all[static_cast<std::size_t>(b)];
                        for (std::size_t r = base; r < hi; ++r) {
                            const int w = row_w_of[r];
                            const int f = row_f_of[r];
                            const double sw =
                                match_level ? solver.m_sqrt_c[r] : 1.0;
                            const double Z = sw * (zw_b[w] + zf_b[f]);
                            const double g =
                                ((bits >> (r - base)) & 1ULL) ? 1.0 : -1.0;
                            const double diff = g - Z;
                            const double z2 = Z * Z;
                            const double m2 = diff * diff;
                            Pacc[static_cast<Eigen::Index>(r)] += z2 / pd;
                            Psq[static_cast<Eigen::Index>(r)] += z2 * z2 / pd;
                            Macc[static_cast<Eigen::Index>(r)] += m2 / pd;
                            Msq[static_cast<Eigen::Index>(r)] += m2 * m2 / pd;
                            PM[static_cast<Eigen::Index>(r)] += z2 * m2 / pd;
                        }
                    }
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
                for (std::int64_t r = 0; r < static_cast<std::int64_t>(R); ++r) {
                    const int w = row_w_of[static_cast<std::size_t>(r)];
                    const int f = row_f_of[static_cast<std::size_t>(r)];
                    const double sw = match_level
                                          ? solver.m_sqrt_c[static_cast<std::size_t>(r)]
                                          : 1.0;
                    for (int b = 0; b < B; ++b) {
                        const Eigen::VectorXd& zwf = zw_all[static_cast<std::size_t>(B + b)];
                        const Eigen::VectorXd& zff = zf_all[static_cast<std::size_t>(B + b)];
                        const Eigen::VectorXd& zwp = zw_all[static_cast<std::size_t>(2 * B + b)];
                        const Eigen::VectorXd& zfp = zf_all[static_cast<std::size_t>(2 * B + b)];
                        const double W2 = sw * (zwf[w] + zff[f]);
                        const double W1 = sw * (zwp[w] + zfp[f]);
                        Bfe[static_cast<Eigen::Index>(r)] += W2 * W2;
                        Bpe[static_cast<Eigen::Index>(r)] += W1 * W1;
                        Bcov[static_cast<Eigen::Index>(r)] += W1 * W2;
                    }
                }
            }
        } else {
        for (int d = 0; d < p; ++d) {
            // --- (Pii, Mii) draw over working rows -------------------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
            for (std::int64_t b = 0; b < n_blocks; ++b) {
                const std::uint64_t bits =
                    stream_seed(options.seed, static_cast<std::uint64_t>(3 * d),
                                static_cast<std::uint64_t>(b));
                const std::size_t base = static_cast<std::size_t>(b) * 64;
                const std::size_t hi = std::min(base + 64, R);
                for (std::size_t r = base; r < hi; ++r) {
                    rvec[r] = ((bits >> (r - base)) & 1ULL) ? 1 : -1;
                }
            }
            tw.setZero();
            tf.setZero();
            if (match_level) {
                for (std::size_t m = 0; m < M; ++m) {
                    const double v = solver.m_sqrt_c[m] * static_cast<double>(rvec[m]);
                    tw[sb.m_w[m]] += v;
                    tf[sb.m_f[m]] += v;
                }
            } else {
                for (std::size_t k = 0; k < n_kept; ++k) {
                    const double v = static_cast<double>(rvec[k]);
                    tw[sb.row_w[k]] += v;
                    tf[sb.row_f[k]] += v;
                }
            }
            if (!solver.solve_K(tw, tf, zw, zf, scr_w, scr_f, cg_iters, true)) {
                solve_failed = true;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
            for (std::int64_t r = 0; r < static_cast<std::int64_t>(R); ++r) {
                const int w = row_w_of[static_cast<std::size_t>(r)];
                const int f = row_f_of[static_cast<std::size_t>(r)];
                const double sw = match_level
                                      ? solver.m_sqrt_c[static_cast<std::size_t>(r)]
                                      : 1.0;
                const double Z = sw * (zw[w] + zf[f]);
                const double diff = static_cast<double>(rvec[static_cast<std::size_t>(r)]) - Z;
                const double z2 = Z * Z;
                const double m2 = diff * diff;
                Pacc[r] += z2 / pd;
                Psq[r] += z2 * z2 / pd;
                Macc[r] += m2 / pd;
                Msq[r] += m2 * m2 / pd;
                PM[r] += z2 * m2 / pd;
            }

            // --- shared centered person-year draw for the fe/pe solves -----
            // v_i = g_i/sqrt(p) - mean(g/sqrt(p)); accumulate A'v per block
            // without materializing person-year vectors: per-match Rademacher
            // sums (parallel), then a sequential binning pass so the result
            // is independent of the thread count.
            long long tot_i = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team) reduction(+ : tot_i)
#endif
            for (std::int64_t m = 0; m < static_cast<std::int64_t>(M); ++m) {
                msum[m] = rademacher_sum(
                    stream_seed(options.seed, static_cast<std::uint64_t>(3 * d + 1),
                                static_cast<std::uint64_t>(m)),
                    sb.m_cnt[static_cast<std::size_t>(m)]);
                if (rademacher_tot_int_exact) {
                    tot_i += static_cast<long long>(msum[m]);
                }
            }
            double tot = static_cast<double>(tot_i);
            if (!rademacher_tot_int_exact) {
                tot = 0.0;
                for (std::size_t m = 0; m < M; ++m) {
                    tot += msum[static_cast<Eigen::Index>(m)];
                }
            }
            const double vbar = (tot / n_py) * scale;
            tw.setZero();
            tf.setZero();
            for (std::size_t m = 0; m < M; ++m) {
                const double su = msum[static_cast<Eigen::Index>(m)] * scale -
                                  vbar * static_cast<double>(sb.m_cnt[m]);
                tw[sb.m_w[m]] += su;
                tf[sb.m_f[m]] += su;
            }
            // fe solve: t = (0, F'v); pe solve: t = (D'v, 0).
            {
                Eigen::VectorXd tw0 = Eigen::VectorXd::Zero(N);
                if (!solver.solve_K(tw0, tf, zw_fe, zf_fe, scr_w, scr_f, cg_iters, true)) {
                    solve_failed = true;
                }
                Eigen::VectorXd tf0 = Eigen::VectorXd::Zero(J);
                if (!solver.solve_K(tw, tf0, zw_pe, zf_pe, scr_w, scr_f, cg_iters, true)) {
                    solve_failed = true;
                }
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (par_rows && solver.team > 1) num_threads(solver.team)
#endif
            for (std::int64_t r = 0; r < static_cast<std::int64_t>(R); ++r) {
                const int w = row_w_of[static_cast<std::size_t>(r)];
                const int f = row_f_of[static_cast<std::size_t>(r)];
                const double sw = match_level
                                      ? solver.m_sqrt_c[static_cast<std::size_t>(r)]
                                      : 1.0;
                const double W2 = sw * (zw_fe[w] + zf_fe[f]);
                const double W1 = sw * (zw_pe[w] + zf_pe[f]);
                Bfe[r] += W2 * W2;
                Bpe[r] += W1 * W1;
                Bcov[r] += W1 * W2;
            }
        }
        }
        // Ratio estimator + nonlinearity correction (LeaveOutTwoWay).
        for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(R); ++r) {
            const double denom = Pacc[r] + Macc[r];
            const double Ph = denom > 0.0 ? Pacc[r] / denom : 0.0;
            const double Mh = 1.0 - Ph;
            Pii[r] = Ph;
            Mii[r] = Mh;
            if (Mh > 0.0) {
                const double Vi =
                    (Mh * Mh * Psq[r] + Ph * Ph * Msq[r] - 2.0 * Mh * Ph * PM[r]) / pd;
                const double Bi =
                    (Mh * Psq[r] - Ph * Msq[r] + 2.0 * (Mh - Ph) * PM[r]) / pd;
                corr[r] = 1.0 - Vi / (Mh * Mh) + Bi / Mh;
            }
        }
    }
    prof_.mark("leverages");
    res.solver_iterations = cg_iters + solver.cuda_iters;
    if (solve_failed) {
        res.converged = false;
        res.notes += "leverage solves did not all converge. ";
    }

    // ---- sigma_i ------------------------------------------------------------
    Eigen::VectorXd sigma(static_cast<Eigen::Index>(R));
    for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(R); ++r) {
        if (match_level && row_stayer[static_cast<std::size_t>(r)]) {
            sigma[r] = 0.0;  // replaced below
            continue;
        }
        const double m = Mii[r];
        if (m <= 1e-12) {
            res.converged = false;
            res.notes += "leverage ~1 on a non-stayer row; sample is not leave-out connected. ";
            sigma[r] = 0.0;
            continue;
        }
        // sigma_row_product pins the multiplication order the shipped
        // binaries realize (see XHDFE_AKM_STRICT_FN): bit-stable across
        // builds where plain source is re-associated unpredictably.
        sigma[r] = sigma_row_product(yt[r] - mean_yt, corr[r], eta[r], m);
    }
    if (match_level) {
        // Stayer sigma via the person-year formula (LeaveOutTwoWay
        // sigma_for_stayers): Pii = 1/T_i, collapsed back to match means.
        const double mean_ypy = has_fw ? kw.dot(ystar) / n_py : ystar.mean();
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
        for (std::size_t k = 0; k < n_kept; ++k) {
            const int m = sb.row_match[k];
            if (!row_stayer[static_cast<std::size_t>(m)]) {
                continue;
            }
            const double T = solver.Dw[sb.row_w[k]];
            const double mii = 1.0 - 1.0 / T;
            const double e =
                ystar[static_cast<Eigen::Index>(k)] - alpha[sb.row_w[k]] - psi[sb.row_f[k]];
            acc[m] += (has_fw ? kw[static_cast<Eigen::Index>(k)] : 1.0) *
                      (ystar[static_cast<Eigen::Index>(k)] - mean_ypy) * (e / mii);
        }
        for (std::size_t m = 0; m < M; ++m) {
            if (row_stayer[m]) {
                sigma[static_cast<Eigen::Index>(m)] =
                    acc[static_cast<Eigen::Index>(m)] / static_cast<double>(sb.m_cnt[m]);
            }
        }
    }

    // ---- Components ----------------------------------------------------------
    const double dof = n_py - 1.0;
    double psi_bar = 0.0, alpha_bar = 0.0;
    for (int f = 0; f < J; ++f) {
        psi_bar += solver.Df[f] * psi[f];
    }
    for (int w = 0; w < N; ++w) {
        alpha_bar += solver.Dw[w] * alpha[w];
    }
    psi_bar /= n_py;
    alpha_bar /= n_py;
    double var_psi_pi = 0.0, var_alpha_pi = 0.0, cov_pi = 0.0;
    for (int f = 0; f < J; ++f) {
        const double dv = psi[f] - psi_bar;
        var_psi_pi += solver.Df[f] * dv * dv;
    }
    for (int w = 0; w < N; ++w) {
        const double dv = alpha[w] - alpha_bar;
        var_alpha_pi += solver.Dw[w] * dv * dv;
    }
    for (std::size_t m = 0; m < M; ++m) {
        cov_pi += solver.m_c[m] * (alpha[sb.m_w[m]] - alpha_bar) * (psi[sb.m_f[m]] - psi_bar);
    }
    var_psi_pi /= dof;
    var_alpha_pi /= dof;
    cov_pi /= dof;
    res.plugin.var_psi = var_psi_pi;
    res.plugin.var_alpha = var_alpha_pi;
    res.plugin.cov_alpha_psi = cov_pi;

    double sum_bfe = 0.0, sum_bpe = 0.0, sum_bcov = 0.0;
    double sum_bfe_sig = 0.0, sum_bpe_sig = 0.0, sum_bcov_sig = 0.0;
    for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(R); ++r) {
        sum_bfe += Bfe[r];
        sum_bpe += Bpe[r];
        sum_bcov += Bcov[r];
        sum_bfe_sig += Bfe[r] * sigma[r];
        sum_bpe_sig += Bpe[r] * sigma[r];
        sum_bcov_sig += Bcov[r] * sigma[r];
    }
    res.kss.var_psi = var_psi_pi - sum_bfe_sig / dof;
    res.kss.var_alpha = var_alpha_pi - sum_bpe_sig / dof;
    res.kss.cov_alpha_psi = cov_pi - sum_bcov_sig / dof;

    // AGSU / homoskedastic (pytwoway 'ho' convention at the person-year level):
    // sigma2_ho = RSS_py / (n_py - (N + J - 1)).
    {
        double rss_py = 0.0;
        for (std::size_t k = 0; k < n_kept; ++k) {
            const double e =
                ystar[static_cast<Eigen::Index>(k)] - alpha[sb.row_w[k]] - psi[sb.row_f[k]];
            rss_py += (has_fw ? kw[static_cast<Eigen::Index>(k)] : 1.0) * e * e;
        }
        const double dof_ho = n_py - static_cast<double>(N + J - 1);
        res.sigma2_ho = dof_ho > 0.0 ? rss_py / dof_ho : 0.0;
        res.agsu.var_psi = var_psi_pi - res.sigma2_ho * sum_bfe / dof;
        res.agsu.var_alpha = var_alpha_pi - res.sigma2_ho * sum_bpe / dof;
        res.agsu.cov_alpha_psi = cov_pi - res.sigma2_ho * sum_bcov / dof;
    }


    // ---- Component standard errors (KSS high-rank case) --------------------
    // Replicates leave_out_COMPLETE: person-year block-leave-out
    // representation with per-match constant Lambda blocks, binned
    // sigma-tilde (llr_fit mode 4), W = By - 0.5(Lambda_B eta_h + xi), and
    // V = (4 sum W^2 sigma~ - Var_sim)/n^2. Oracle stayer conventions at
    // match level: p = 1/T, b_fe = b_cov = 0, b_pe = one representative
    // solve per tenure class; eta_h on stayer blocks uses the consistent
    // minimum-norm solution (their within-match OLS residuals sum to zero).
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (options.compute_se) {
        const std::size_t n_py_sz = n_kept;
        const double NTd = n_py;
        const int se_block = akm_se_block_env();
        // per-match bases with SE conventions
        std::vector<double> bp(M), bfe(M), bpe(M), bcov(M);
        std::vector<char> m_stayer(M, 0);
        {
            std::vector<int> worker_deg(static_cast<std::size_t>(N), 0);
            for (std::size_t m = 0; m < M; ++m) {
                ++worker_deg[static_cast<std::size_t>(sb.m_w[m])];
            }
            for (std::size_t m = 0; m < M; ++m) {
                const double c = static_cast<double>(sb.m_cnt[m]);
                const double div = match_level ? c : 1.0;
                // row index of this match's values
                const Eigen::Index r0 = match_level
                    ? static_cast<Eigen::Index>(m)
                    : -1;
                if (match_level) {
                    bp[m] = Pii[r0] / div;
                    bfe[m] = Bfe[r0] / div;
                    bpe[m] = Bpe[r0] / div;
                    bcov[m] = Bcov[r0] / div;
                }
                m_stayer[m] = worker_deg[static_cast<std::size_t>(sb.m_w[m])] < 2 ? 1 : 0;
            }
            if (!match_level) {
                // per-obs rows carry the unweighted bases already
                std::vector<char> seen(M, 0);
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const int m = sb.row_match[k];
                    if (!seen[static_cast<std::size_t>(m)]) {
                        seen[static_cast<std::size_t>(m)] = 1;
                        bp[static_cast<std::size_t>(m)] = Pii[static_cast<Eigen::Index>(k)];
                        bfe[static_cast<std::size_t>(m)] = Bfe[static_cast<Eigen::Index>(k)];
                        bpe[static_cast<std::size_t>(m)] = Bpe[static_cast<Eigen::Index>(k)];
                        bcov[static_cast<std::size_t>(m)] = Bcov[static_cast<Eigen::Index>(k)];
                    }
                }
            }
            // oracle stayer conventions
            std::unordered_map<long long, double> rep_bpe;  // tenure class -> base_pe
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const int m = sb.row_match[k];
                if (!m_stayer[static_cast<std::size_t>(m)]) {
                    continue;
                }
                const long long T = sb.m_cnt[static_cast<std::size_t>(m)];
                if (T > 1 && rep_bpe.find(T) == rep_bpe.end()) {
                    rep_bpe.emplace(T, bpe[static_cast<std::size_t>(m)]);
                }
            }
            for (std::size_t m = 0; m < M; ++m) {
                if (!m_stayer[m]) {
                    continue;
                }
                bp[m] = 1.0 / static_cast<double>(solver.Dw[sb.m_w[m]]);
                bfe[m] = 0.0;
                bcov[m] = 0.0;
                auto it = rep_bpe.find(sb.m_cnt[m]);
                bpe[m] = it != rep_bpe.end() ? it->second : 0.0;
            }
        }
        const bool has_stayers = res.sample.n_stayers > 0;
        const bool pe_identified = !(match_level && has_stayers);

        // person-year eta and block leave-out eta_h
        Eigen::VectorXd eta_py(static_cast<Eigen::Index>(n_py_sz));
        Eigen::VectorXd sum_eta_m = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
        for (std::size_t k = 0; k < n_py_sz; ++k) {
            eta_py[static_cast<Eigen::Index>(k)] =
                ystar[static_cast<Eigen::Index>(k)] - alpha[sb.row_w[k]] - psi[sb.row_f[k]];
            sum_eta_m[sb.row_match[k]] += eta_py[static_cast<Eigen::Index>(k)];
        }
        // block factor: for cluster c (match at match level, obs otherwise):
        // eta_h = eta + p/(1 - T p) (1'eta) within the block; stayer blocks
        // at match level use the consistent solution eta_h = eta.
        auto block_gain = [&](int m) -> double {
            const double p_m = bp[static_cast<std::size_t>(m)];
            const double T_m = match_level ? static_cast<double>(sb.m_cnt[static_cast<std::size_t>(m)]) : 1.0;
            const double den = 1.0 - T_m * p_m;
            if (den <= 1e-12) {
                return 0.0;  // singular stayer block: minimum-norm solution
            }
            return p_m / den;
        };
        Eigen::VectorXd eta_h(static_cast<Eigen::Index>(n_py_sz));
        if (match_level) {
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const int m = sb.row_match[k];
                eta_h[static_cast<Eigen::Index>(k)] =
                    eta_py[static_cast<Eigen::Index>(k)] +
                    block_gain(m) * sum_eta_m[m];
            }
        } else {
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const int m = sb.row_match[k];
                const double den = 1.0 - bp[static_cast<std::size_t>(m)];
                eta_h[static_cast<Eigen::Index>(k)] =
                    eta_py[static_cast<Eigen::Index>(k)] / (den > 1e-12 ? den : 1.0);
            }
        }

        // sigma_i raw and binned sigma-tilde per component
        std::vector<double> sig_raw(n_py_sz);
        for (std::size_t k = 0; k < n_py_sz; ++k) {
            sig_raw[k] = ystar[static_cast<Eigen::Index>(k)] *
                         eta_h[static_cast<Eigen::Index>(k)];
        }
        // float-snapped binning keys: quantile-cut decisions become invariant
        // to last-ulp codegen differences across front-end builds (the bin
        // MEANS still use full-precision sigma_i).
        const auto snap = [](double x) {
            return static_cast<double>(static_cast<float>(x));
        };
        std::vector<double> p_obs(n_py_sz);
        for (std::size_t k = 0; k < n_py_sz; ++k) {
            p_obs[k] = snap(bp[static_cast<std::size_t>(sb.row_match[k])]);
        }
        const std::vector<int> gP = group_equally(p_obs, options.se_sigma_grid);
        // llr_fit mode 0: MATLAB 'lowess' surface fit of sigma_i on
        // (Pii, Bii), normalized predictors, span hBest = NT^(-1/3)
        // (k = ceil(hBest*NT) nearest neighbours), tricube weights, local
        // LINEAR fit, evaluated at the data points; NaN -> sigma_i.
        auto lowess_sigma = [&](const std::vector<double>& bases) {
            const std::size_t n = n_py_sz;
            std::vector<double> px(n), bx(n);
            for (std::size_t k = 0; k < n; ++k) {
                px[k] = bp[static_cast<std::size_t>(sb.row_match[k])];
                bx[k] = bases[static_cast<std::size_t>(sb.row_match[k])];
            }
            // z-normalize (MATLAB Normalize','on'; std with N-1)
            const auto zscore = [&](std::vector<double>& v) {
                double mu = 0.0;
                for (double x : v) mu += x;
                mu /= static_cast<double>(n);
                double s2 = 0.0;
                for (double x : v) s2 += (x - mu) * (x - mu);
                const double sd =
                    n > 1 ? std::sqrt(s2 / static_cast<double>(n - 1)) : 1.0;
                const double inv = sd > 0.0 ? 1.0 / sd : 0.0;
                for (double& x : v) x = (x - mu) * inv;
            };
            zscore(px);
            zscore(bx);
            const double span = std::pow(static_cast<double>(n), -1.0 / 3.0);
            std::size_t kk = static_cast<std::size_t>(
                std::ceil(span * static_cast<double>(n)));
            if (kk < 3) kk = 3;
            if (kk > n) kk = n;
            std::vector<double> out(n);
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                std::vector<std::pair<double, std::size_t>> dist(n);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(n); ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    for (std::size_t j = 0; j < n; ++j) {
                        const double dx = px[j] - px[i];
                        const double dy = bx[j] - bx[i];
                        dist[j] = {dx * dx + dy * dy, j};
                    }
                    std::nth_element(dist.begin(), dist.begin() + (kk - 1),
                                     dist.end());
                    const double dmax2 = dist[kk - 1].first;
                    const double dmax = std::sqrt(dmax2);
                    // weighted local linear fit on the kk neighbours
                    double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
                    double b0 = 0, b1 = 0, b2 = 0;
                    for (std::size_t t = 0; t < kk; ++t) {
                        const std::size_t j = dist[t].second;
                        const double d = std::sqrt(dist[t].first);
                        double wgt = 1.0;
                        if (dmax > 0.0) {
                            const double u = d / dmax;
                            const double c = 1.0 - u * u * u;
                            wgt = c * c * c;
                            if (wgt < 0.0) wgt = 0.0;
                        }
                        const double x1 = px[j] - px[i];
                        const double x2 = bx[j] - bx[i];
                        a00 += wgt;
                        a01 += wgt * x1;
                        a02 += wgt * x2;
                        a11 += wgt * x1 * x1;
                        a12 += wgt * x1 * x2;
                        a22 += wgt * x2 * x2;
                        b0 += wgt * sig_raw[j];
                        b1 += wgt * x1 * sig_raw[j];
                        b2 += wgt * x2 * sig_raw[j];
                    }
                    // solve the 3x3 system; centered predictors -> the
                    // intercept is the prediction at point i
                    Eigen::Matrix3d A;
                    A << a00, a01, a02, a01, a11, a12, a02, a12, a22;
                    Eigen::Vector3d rhs(b0, b1, b2);
                    Eigen::Vector3d sol;
                    bool ok = false;
                    {
                        Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
                        if (lu.isInvertible()) {
                            sol = lu.solve(rhs);
                            ok = true;
                        }
                    }
                    double pred;
                    if (ok) {
                        pred = sol[0];
                    } else if (a00 > 0.0) {
                        pred = b0 / a00;  // singular: weighted mean
                    } else {
                        pred = sig_raw[i];
                    }
                    // finite-magnitude guard (fast-math: no isnan)
                    if (!(pred > -1e300 && pred < 1e300)) {
                        pred = sig_raw[i];
                    }
                    out[i] = pred;
                }
            }
            return out;
        };
        auto binned_sigma = [&](const std::vector<double>& bases) {
            std::vector<double> b_obs(n_py_sz);
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                b_obs[k] = snap(bases[static_cast<std::size_t>(sb.row_match[k])]);
            }
            const std::vector<int> gB = group_equally(b_obs, options.se_sigma_grid);
            std::unordered_map<long long, std::pair<double, long long>> cells;
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const long long key =
                    static_cast<long long>(gB[k]) * 1000003LL + gP[k];
                auto& cell = cells[key];
                cell.first += sig_raw[k];
                ++cell.second;
            }
            std::vector<double> out(n_py_sz);
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const long long key =
                    static_cast<long long>(gB[k]) * 1000003LL + gP[k];
                const auto& cell = cells[key];
                out[k] = cell.first / static_cast<double>(cell.second);
            }
            return out;
        };

        // component machinery
        Eigen::VectorXd tw_se(N), tf_se(J), zw_se(N), zf_se(J), scw(N), scf(J);
        double psi_bar_se = 0.0, alpha_bar_se = 0.0;
        for (int f2 = 0; f2 < J; ++f2) psi_bar_se += solver.Df[f2] * psi[f2];
        for (int w2 = 0; w2 < N; ++w2) alpha_bar_se += solver.Dw[w2] * alpha[w2];
        psi_bar_se /= NTd;
        alpha_bar_se /= NTd;

        // ---- Weak-identification machinery (eigen_diagnostics) ---------
        // Pencil (A_q, K) quantities in the grounded coefficient space:
        // K z and A_q z applied matrix-free over the match structure.
        const bool do_ci = options.eigen_diagnostics;
        auto apply_Kmat = [&](const Eigen::VectorXd& zw, const Eigen::VectorXd& zf,
                              Eigen::VectorXd& ow, Eigen::VectorXd& of) {
            for (int w2 = 0; w2 < N; ++w2) ow[w2] = solver.Dw[w2] * zw[w2];
            for (int f2 = 0; f2 < J; ++f2) of[f2] = solver.Df[f2] * zf[f2];
            for (std::size_t m2 = 0; m2 < M; ++m2) {
                const double c = solver.m_c[m2];
                ow[sb.m_w[m2]] += c * zf[sb.m_f[m2]];
                of[sb.m_f[m2]] += c * zw[sb.m_w[m2]];
            }
            of[J - 1] = 0.0;
        };
        auto apply_Aq = [&](int comp, const Eigen::VectorXd& zw,
                            const Eigen::VectorXd& zf, Eigen::VectorXd& ow,
                            Eigen::VectorXd& of) {
            ow.setZero();
            of.setZero();
            if (comp == 0) {
                double mean_f = 0.0;
                for (int f2 = 0; f2 < J; ++f2) mean_f += solver.Df[f2] * zf[f2];
                mean_f /= NTd;
                for (int f2 = 0; f2 < J; ++f2)
                    of[f2] = solver.Df[f2] * (zf[f2] - mean_f);
            } else if (comp == 2) {
                double mean_w = 0.0;
                for (int w2 = 0; w2 < N; ++w2) mean_w += solver.Dw[w2] * zw[w2];
                mean_w /= NTd;
                for (int w2 = 0; w2 < N; ++w2)
                    ow[w2] = solver.Dw[w2] * (zw[w2] - mean_w);
            } else {
                double mean_w = 0.0, mean_f = 0.0;
                for (int w2 = 0; w2 < N; ++w2) mean_w += solver.Dw[w2] * zw[w2];
                for (int f2 = 0; f2 < J; ++f2) mean_f += solver.Df[f2] * zf[f2];
                mean_w /= NTd;
                mean_f /= NTd;
                for (std::size_t m2 = 0; m2 < M; ++m2) {
                    const double c = solver.m_c[m2];
                    ow[sb.m_w[m2]] += 0.5 * c * (zf[sb.m_f[m2]] - mean_f);
                    of[sb.m_f[m2]] += 0.5 * c * (zw[sb.m_w[m2]] - mean_w);
                }
            }
            of[J - 1] = 0.0;
        };
        // Top-3 eigenpairs of the pencil (A_q, K) — eigs(Adot, Sxxdot, 3)
        // in eigAux.m — via blocked subspace iteration with Rayleigh-Ritz
        // in the K metric (deterministic start vectors).
        auto pencil_eigs = [&](int comp, double lam[3], Eigen::VectorXd& q1w,
                               Eigen::VectorXd& q1f) {
            constexpr int NB = 3;
            std::vector<Eigen::VectorXd> Vw(NB, Eigen::VectorXd(N));
            std::vector<Eigen::VectorXd> Vf(NB, Eigen::VectorXd(J));
            for (int b = 0; b < NB; ++b) {
                std::uint64_t st = stream_seed(options.seed ^ 0xE16E5EEDULL,
                                               static_cast<std::uint64_t>(comp),
                                               static_cast<std::uint64_t>(b));
                for (int w2 = 0; w2 < N; ++w2) Vw[b][w2] = normal_from_stream(st);
                for (int f2 = 0; f2 < J; ++f2) Vf[b][f2] = normal_from_stream(st);
                Vf[b][J - 1] = 0.0;
            }
            Eigen::VectorXd aw(N), af(J), kw(N), kf(J);
            // batched-lane storage (se_block >= 1): the NB solves of one
            // subspace iteration are independent, so they go through the
            // batched multi-RHS solver in a single call; the A_q
            // applications and everything after the solves keep the exact
            // per-lane sequential sequences. Iterations stay sequential.
            std::vector<Eigen::VectorXd> pe_aw, pe_af;
            std::vector<const Eigen::VectorXd*> pe_twp, pe_tfp;
            std::vector<char> pe_ok;
            if (se_block >= 1) {
                pe_aw.assign(NB, Eigen::VectorXd(N));
                pe_af.assign(NB, Eigen::VectorXd(J));
                pe_twp.resize(NB);
                pe_tfp.resize(NB);
                for (int b = 0; b < NB; ++b) {
                    pe_twp[static_cast<std::size_t>(b)] =
                        &pe_aw[static_cast<std::size_t>(b)];
                    pe_tfp[static_cast<std::size_t>(b)] =
                        &pe_af[static_cast<std::size_t>(b)];
                }
            }
            double prev[3] = {0.0, 0.0, 0.0};
            lam[0] = lam[1] = lam[2] = 0.0;
            for (int it = 0; it < 2000; ++it) {
                // Z_b = K^{-1} A_q V_b
                if (se_block >= 1) {
                    [&]() XHDFE_AKM_OUTLINE {
                        for (int b = 0; b < NB; ++b) {
                            apply_Aq(comp, Vw[b], Vf[b],
                                     pe_aw[static_cast<std::size_t>(b)],
                                     pe_af[static_cast<std::size_t>(b)]);
                        }
                        solver.solve_K_multi(pe_twp, pe_tfp, Vw, Vf, pe_ok,
                                             cg_iters, true);
                        for (int b = 0; b < NB; ++b) {
                            if (!pe_ok[static_cast<std::size_t>(b)]) {
                                res.converged = false;
                            }
                        }
                    }();
                } else {
                for (int b = 0; b < NB; ++b) {
                    apply_Aq(comp, Vw[b], Vf[b], aw, af);
                    if (!solver.solve_K(aw, af, Vw[b], Vf[b], scw, scf, cg_iters,
                                        true)) {
                        res.converged = false;
                    }
                }
                }
                // Rayleigh-Ritz in the pencil: Ah = V'A V, Kh = V'K V
                Eigen::Matrix3d Ah, Kh;
                std::vector<Eigen::VectorXd> Aw(NB, Eigen::VectorXd(N));
                std::vector<Eigen::VectorXd> Af(NB, Eigen::VectorXd(J));
                std::vector<Eigen::VectorXd> Kw(NB, Eigen::VectorXd(N));
                std::vector<Eigen::VectorXd> Kf(NB, Eigen::VectorXd(J));
                for (int b = 0; b < NB; ++b) {
                    apply_Aq(comp, Vw[b], Vf[b], Aw[b], Af[b]);
                    apply_Kmat(Vw[b], Vf[b], Kw[b], Kf[b]);
                }
                for (int a = 0; a < NB; ++a) {
                    for (int b = 0; b < NB; ++b) {
                        Ah(a, b) = Vw[a].dot(Aw[b]) + Vf[a].dot(Af[b]);
                        Kh(a, b) = Vw[a].dot(Kw[b]) + Vf[a].dot(Kf[b]);
                    }
                }
                Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3d> ges(
                    Ah, Kh);
                // order by |lambda| descending (eigs 'largestabs')
                int ord[3] = {0, 1, 2};
                const Eigen::Vector3d ev = ges.eigenvalues();
                std::sort(ord, ord + 3, [&](int a, int b) {
                    return std::abs(ev[a]) > std::abs(ev[b]);
                });
                std::vector<Eigen::VectorXd> Nw(NB, Eigen::VectorXd(N));
                std::vector<Eigen::VectorXd> Nf(NB, Eigen::VectorXd(J));
                for (int b = 0; b < NB; ++b) {
                    Nw[b].setZero();
                    Nf[b].setZero();
                    for (int a = 0; a < NB; ++a) {
                        const double cvec = ges.eigenvectors()(a, ord[b]);
                        Nw[b] += cvec * Vw[a];
                        Nf[b] += cvec * Vf[a];
                    }
                    // K-normalize
                    apply_Kmat(Nw[b], Nf[b], kw, kf);
                    const double nrm =
                        std::sqrt(Nw[b].dot(kw) + Nf[b].dot(kf));
                    if (nrm > 0.0) {
                        Nw[b] /= nrm;
                        Nf[b] /= nrm;
                    }
                    lam[b] = ev[ord[b]];
                }
                Vw.swap(Nw);
                Vf.swap(Nf);
                double diff = 0.0;
                for (int b = 0; b < NB; ++b) {
                    diff = std::max(diff, std::abs(lam[b] - prev[b]) /
                                              (1.0 + std::abs(lam[b])));
                    prev[b] = lam[b];
                }
                if (it >= 3 && diff < 1e-11) break;
            }
            q1w = Vw[0];
            q1f = Vf[0];
        };
        // Hutchinson estimate of tr(Atilde^2) (trace_Atilde_sqr.m):
        // mean over draws of ||X K^{-1} A_q K^{-1} X' g||^2, g Rademacher.
        auto trace_atilde_sqr = [&](int comp) {
            const int nsim_tr = options.eig_trace_nsim;
            double acc = 0.0;
            if (se_block >= 1) {
                [&]() XHDFE_AKM_OUTLINE {
                // Two-wave batching: within a block of draws all FIRST
                // solves are independent (batched together), each draw's
                // SECOND solve depends on its own first solve only (A_q of
                // it), so the second solves form another independent batch.
                // Per draw the Rademacher stream, the A_q application and
                // the accumulation keep the exact sequential sequences
                // (ascending draw index), so acc is bit-identical for any
                // block size >= 1 and any thread count.
                const int TB = se_block;
                std::vector<Eigen::VectorXd> t_tw(static_cast<std::size_t>(TB));
                std::vector<Eigen::VectorXd> t_tf(static_cast<std::size_t>(TB));
                std::vector<Eigen::VectorXd> t_c1w(static_cast<std::size_t>(TB));
                std::vector<Eigen::VectorXd> t_c1f(static_cast<std::size_t>(TB));
                std::vector<Eigen::VectorXd> t_aw(static_cast<std::size_t>(TB),
                                                  Eigen::VectorXd(N));
                std::vector<Eigen::VectorXd> t_af(static_cast<std::size_t>(TB),
                                                  Eigen::VectorXd(J));
                std::vector<Eigen::VectorXd> t_c2w(static_cast<std::size_t>(TB));
                std::vector<Eigen::VectorXd> t_c2f(static_cast<std::size_t>(TB));
                std::vector<const Eigen::VectorXd*> twp, tfp;
                std::vector<char> okv;
                for (int s0 = 0; s0 < nsim_tr; s0 += TB) {
                    const int B = std::min(TB, nsim_tr - s0);
                    twp.resize(static_cast<std::size_t>(B));
                    tfp.resize(static_cast<std::size_t>(B));
                    for (int b = 0; b < B; ++b) {
                        const int s2 = s0 + b;
                        std::uint64_t st =
                            stream_seed(options.seed ^ 0x77ACE717ULL,
                                        static_cast<std::uint64_t>(comp),
                                        static_cast<std::uint64_t>(s2));
                        Eigen::VectorXd& tw_b = t_tw[static_cast<std::size_t>(b)];
                        Eigen::VectorXd& tf_b = t_tf[static_cast<std::size_t>(b)];
                        tw_b = Eigen::VectorXd::Zero(N);
                        tf_b = Eigen::VectorXd::Zero(J);
                        for (std::size_t k = 0; k < n_py_sz; ++k) {
                            st = splitmix64(st);
                            const double g = (st >> 63) ? 1.0 : -1.0;
                            tw_b[sb.row_w[k]] += g;
                            tf_b[sb.row_f[k]] += g;
                        }
                        twp[static_cast<std::size_t>(b)] = &tw_b;
                        tfp[static_cast<std::size_t>(b)] = &tf_b;
                        t_c1w[static_cast<std::size_t>(b)].resize(N);
                        t_c1f[static_cast<std::size_t>(b)].resize(J);
                    }
                    solver.solve_K_multi(twp, tfp, t_c1w, t_c1f, okv, cg_iters,
                                         true);
                    for (int b = 0; b < B; ++b) {
                        if (!okv[static_cast<std::size_t>(b)]) {
                            res.converged = false;
                        }
                        apply_Aq(comp, t_c1w[static_cast<std::size_t>(b)],
                                 t_c1f[static_cast<std::size_t>(b)],
                                 t_aw[static_cast<std::size_t>(b)],
                                 t_af[static_cast<std::size_t>(b)]);
                        twp[static_cast<std::size_t>(b)] =
                            &t_aw[static_cast<std::size_t>(b)];
                        tfp[static_cast<std::size_t>(b)] =
                            &t_af[static_cast<std::size_t>(b)];
                        t_c2w[static_cast<std::size_t>(b)].resize(N);
                        t_c2f[static_cast<std::size_t>(b)].resize(J);
                    }
                    solver.solve_K_multi(twp, tfp, t_c2w, t_c2f, okv, cg_iters,
                                         true);
                    for (int b = 0; b < B; ++b) {
                        if (!okv[static_cast<std::size_t>(b)]) {
                            res.converged = false;
                        }
                        const Eigen::VectorXd& c2w_b =
                            t_c2w[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd& c2f_b =
                            t_c2f[static_cast<std::size_t>(b)];
                        double tr_s = 0.0;
                        for (std::size_t k = 0; k < n_py_sz; ++k) {
                            const double v2 = c2w_b[sb.row_w[k]] + c2f_b[sb.row_f[k]];
                            tr_s += v2 * v2;
                        }
                        acc += tr_s;
                    }
                }
                }();
                return acc / static_cast<double>(nsim_tr);
            }
            Eigen::VectorXd tw2(N), tf2(J), c1w(N), c1f(J), aw(N), af(J),
                c2w(N), c2f(J);
            for (int s2 = 0; s2 < nsim_tr; ++s2) {
                std::uint64_t st = stream_seed(options.seed ^ 0x77ACE717ULL,
                                               static_cast<std::uint64_t>(comp),
                                               static_cast<std::uint64_t>(s2));
                tw2.setZero();
                tf2.setZero();
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    st = splitmix64(st);
                    const double g = (st >> 63) ? 1.0 : -1.0;
                    tw2[sb.row_w[k]] += g;
                    tf2[sb.row_f[k]] += g;
                }
                if (!solver.solve_K(tw2, tf2, c1w, c1f, scw, scf, cg_iters,
                                    true)) {
                    res.converged = false;
                }
                apply_Aq(comp, c1w, c1f, aw, af);
                if (!solver.solve_K(aw, af, c2w, c2f, scw, scf, cg_iters,
                                    true)) {
                    res.converged = false;
                }
                double tr_s = 0.0;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const double v2 = c2w[sb.row_w[k]] + c2f[sb.row_f[k]];
                    tr_s += v2 * v2;
                }
                acc += tr_s;
            }
            return acc / static_cast<double>(nsim_tr);
        };

        auto run_component = [&](int comp, const std::vector<double>& bases,
                                 double& theta_out, double& se_out) {
            // theta (leave_out_COMPLETE convention)
            double corr_sum = 0.0;
            Eigen::VectorXd sum_y_m = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                sum_y_m[sb.row_match[k]] += ystar[static_cast<Eigen::Index>(k)];
            }
            for (std::size_t m = 0; m < M; ++m) {
                const double b_m = bases[m];
                if (b_m == 0.0) continue;
                const double T_m = match_level ? static_cast<double>(sb.m_cnt[m]) : 1.0;
                double s_eta_h;
                if (match_level) {
                    s_eta_h = sum_eta_m[static_cast<Eigen::Index>(m)] *
                              (1.0 + T_m * block_gain(static_cast<int>(m)));
                    corr_sum += b_m * sum_y_m[static_cast<Eigen::Index>(m)] * s_eta_h;
                } else {
                    // handled per obs below
                }
            }
            if (!match_level) {
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    corr_sum += bases[static_cast<std::size_t>(sb.row_match[k])] *
                                ystar[static_cast<Eigen::Index>(k)] *
                                eta_h[static_cast<Eigen::Index>(k)];
                }
            }
            double plug = 0.0;
            if (comp == 0) plug = var_psi_pi;
            if (comp == 1) plug = cov_pi;
            if (comp == 2) plug = var_alpha_pi;
            theta_out = plug - corr_sum / NTd;

            // sigma-tilde for this component
            const std::vector<double> sig_t = options.se_sigma_lowess
                                                  ? lowess_sigma(bases)
                                                  : binned_sigma(bases);
            prof_.mark("se_sigma_fit");

            // A_b and By
            tw_se.setZero();
            tf_se.setZero();
            if (comp == 0) {
                for (int f2 = 0; f2 < J; ++f2)
                    tf_se[f2] = solver.Df[f2] * (psi[f2] - psi_bar_se);
            } else if (comp == 2) {
                for (int w2 = 0; w2 < N; ++w2)
                    tw_se[w2] = solver.Dw[w2] * (alpha[w2] - alpha_bar_se);
            } else {
                // A_b = [0.5 D'(fe - mean fe); 0.5 S'F'(pe - mean pe)]
                for (std::size_t m = 0; m < M; ++m) {
                    const double c = solver.m_c[m];
                    tw_se[sb.m_w[m]] += 0.5 * c * (psi[sb.m_f[m]] - psi_bar_se);
                    tf_se[sb.m_f[m]] += 0.5 * c * (alpha[sb.m_w[m]] - alpha_bar_se);
                }
            }
            Eigen::VectorXd By(static_cast<Eigen::Index>(n_py_sz));

            // xi: u = Lambda_B y (block), ydx = (I-Lambda_P)^{-1} u,
            // xi = ydx - X (K^{-1} X' ydx). ydx does not depend on the A_b
            // solve, so with se_block >= 1 the two independent solves go
            // through the batched multi-RHS solver as one two-lane call
            // (per lane identical to the sequential solve_K sequence).
            Eigen::VectorXd ydx(static_cast<Eigen::Index>(n_py_sz));
            if (match_level) {
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const int m = sb.row_match[k];
                    const double u = bases[static_cast<std::size_t>(m)] *
                                     sum_y_m[static_cast<Eigen::Index>(m)];
                    const double T_m = static_cast<double>(sb.m_cnt[static_cast<std::size_t>(m)]);
                    const double den = 1.0 - T_m * bp[static_cast<std::size_t>(m)];
                    ydx[static_cast<Eigen::Index>(k)] =
                        (bases[static_cast<std::size_t>(m)] == 0.0)
                            ? 0.0
                            : u / (den > 1e-12 ? den : 1.0);
                }
            } else {
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const int m = sb.row_match[k];
                    const double den = 1.0 - bp[static_cast<std::size_t>(m)];
                    ydx[static_cast<Eigen::Index>(k)] =
                        bases[static_cast<std::size_t>(m)] *
                        ystar[static_cast<Eigen::Index>(k)] /
                        (den > 1e-12 ? den : 1.0);
                }
            }
            if (se_block >= 1) {
                [&]() XHDFE_AKM_OUTLINE {
                std::vector<Eigen::VectorXd> p_tw(2), p_tf(2), p_zw(2), p_zf(2);
                p_tw[0] = tw_se;
                p_tf[0] = tf_se;
                p_tw[1] = Eigen::VectorXd::Zero(N);
                p_tf[1] = Eigen::VectorXd::Zero(J);
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    p_tw[1][sb.row_w[k]] += ydx[static_cast<Eigen::Index>(k)];
                    p_tf[1][sb.row_f[k]] += ydx[static_cast<Eigen::Index>(k)];
                }
                for (int c = 0; c < 2; ++c) {
                    p_zw[static_cast<std::size_t>(c)].resize(N);
                    p_zf[static_cast<std::size_t>(c)].resize(J);
                }
                std::vector<const Eigen::VectorXd*> twp{&p_tw[0], &p_tw[1]};
                std::vector<const Eigen::VectorXd*> tfp{&p_tf[0], &p_tf[1]};
                std::vector<char> okv;
                solver.solve_K_multi(twp, tfp, p_zw, p_zf, okv, cg_iters, true);
                if (!okv[0]) {
                    res.converged = false;
                    res.notes += "SE solve (A_b) did not converge. ";
                }
                if (!okv[1]) {
                    res.converged = false;
                    res.notes += "SE solve (xi) did not converge. ";
                }
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    By[static_cast<Eigen::Index>(k)] =
                        p_zw[0][sb.row_w[k]] + p_zf[0][sb.row_f[k]];
                }
                // the SUM loop and the sims below read the xi solution from
                // zw_se/zf_se exactly like the sequential path
                zw_se.swap(p_zw[1]);
                zf_se.swap(p_zf[1]);
                }();
            } else {
                if (!solver.solve_K(tw_se, tf_se, zw_se, zf_se, scw, scf, cg_iters, true)) {
                    res.converged = false;
                    res.notes += "SE solve (A_b) did not converge. ";
                }
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    By[static_cast<Eigen::Index>(k)] =
                        zw_se[sb.row_w[k]] + zf_se[sb.row_f[k]];
                }
                tw_se.setZero();
                tf_se.setZero();
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    tw_se[sb.row_w[k]] += ydx[static_cast<Eigen::Index>(k)];
                    tf_se[sb.row_f[k]] += ydx[static_cast<Eigen::Index>(k)];
                }
                if (!solver.solve_K(tw_se, tf_se, zw_se, zf_se, scw, scf, cg_iters, true)) {
                    res.converged = false;
                    res.notes += "SE solve (xi) did not converge. ";
                }
            }
            double SUM = 0.0;
            Eigen::VectorXd sum_etah_m = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                sum_etah_m[sb.row_match[k]] += eta_h[static_cast<Eigen::Index>(k)];
            }
            for (std::size_t k = 0; k < n_py_sz; ++k) {
                const int m = sb.row_match[k];
                const double lam_eta =
                    match_level ? bases[static_cast<std::size_t>(m)] *
                                      sum_etah_m[static_cast<Eigen::Index>(m)]
                                : bases[static_cast<std::size_t>(m)] *
                                      eta_h[static_cast<Eigen::Index>(k)];
                const double xi = ydx[static_cast<Eigen::Index>(k)] -
                                  (zw_se[sb.row_w[k]] + zf_se[sb.row_f[k]]);
                const double W = By[static_cast<Eigen::Index>(k)] -
                                 0.5 * (lam_eta + xi);
                SUM += W * W * sig_t[k];
            }
            prof_.mark("se_ab_xi");

            // ---- q=1 (Andrews-Mikusheva) preparation --------------------
            // eigAux + DO_R1 of leave_out_estimation_two_way: top pencil
            // eigenpairs, x1bar = X q1 (unit norm), Lambda_B2 = Lambda_B -
            // diag(lambda_1 x1bar^2), W_2 and the Sigma_1 pieces.
            double lam3[3] = {0.0, 0.0, 0.0};
            double lam1 = 0.0, trace_est = 0.0, b1 = 0.0;
            double cov11 = 0.0, cov12 = 0.0, sum2 = 0.0;
            Eigen::VectorXd x1bar, W2vec;
            std::vector<double> sims2_re, sims2_im;
            if (do_ci) {
                Eigen::VectorXd q1w(N), q1f(J);
                pencil_eigs(comp, lam3, q1w, q1f);
                prof_.mark("eig_pencil");
                lam1 = lam3[0];
                trace_est = trace_atilde_sqr(comp);
                prof_.mark("eig_trace");
                x1bar.resize(static_cast<Eigen::Index>(n_py_sz));
                double nrm2 = 0.0;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const double v2 = q1w[sb.row_w[k]] + q1f[sb.row_f[k]];
                    x1bar[static_cast<Eigen::Index>(k)] = v2;
                    nrm2 += v2 * v2;
                }
                x1bar /= std::sqrt(nrm2);
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    b1 += x1bar[static_cast<Eigen::Index>(k)] *
                          ystar[static_cast<Eigen::Index>(k)];
                }
                // u = Lambda_B2 y (no longer block-constant: general
                // (I-Lambda_P)^{-1} block form u + gain (1'u))
                Eigen::VectorXd u(static_cast<Eigen::Index>(n_py_sz));
                Eigen::VectorXd sum_u_m =
                    Eigen::VectorXd::Zero(static_cast<Eigen::Index>(M));
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const int m = sb.row_match[k];
                    const double x2 = x1bar[static_cast<Eigen::Index>(k)] *
                                      x1bar[static_cast<Eigen::Index>(k)];
                    const double blk =
                        match_level
                            ? bases[static_cast<std::size_t>(m)] *
                                  sum_y_m[static_cast<Eigen::Index>(m)]
                            : bases[static_cast<std::size_t>(m)] *
                                  ystar[static_cast<Eigen::Index>(k)];
                    u[static_cast<Eigen::Index>(k)] =
                        blk - lam1 * x2 * ystar[static_cast<Eigen::Index>(k)];
                    sum_u_m[m] += u[static_cast<Eigen::Index>(k)];
                }
                Eigen::VectorXd ydx2(static_cast<Eigen::Index>(n_py_sz));
                if (match_level) {
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m = sb.row_match[k];
                        ydx2[static_cast<Eigen::Index>(k)] =
                            u[static_cast<Eigen::Index>(k)] +
                            block_gain(m) * sum_u_m[m];
                    }
                } else {
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m = sb.row_match[k];
                        const double den =
                            1.0 - bp[static_cast<std::size_t>(m)];
                        ydx2[static_cast<Eigen::Index>(k)] =
                            u[static_cast<Eigen::Index>(k)] /
                            (den > 1e-12 ? den : 1.0);
                    }
                }
                tw_se.setZero();
                tf_se.setZero();
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    tw_se[sb.row_w[k]] += ydx2[static_cast<Eigen::Index>(k)];
                    tf_se[sb.row_f[k]] += ydx2[static_cast<Eigen::Index>(k)];
                }
                if (!solver.solve_K(tw_se, tf_se, zw_se, zf_se, scw, scf,
                                    cg_iters, true)) {
                    res.converged = false;
                    res.notes += "CI solve (xi2) did not converge. ";
                }
                W2vec.resize(static_cast<Eigen::Index>(n_py_sz));
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const int m = sb.row_match[k];
                    const double x2 = x1bar[static_cast<Eigen::Index>(k)] *
                                      x1bar[static_cast<Eigen::Index>(k)];
                    const double lam_eta2 =
                        (match_level
                             ? bases[static_cast<std::size_t>(m)] *
                                   sum_etah_m[static_cast<Eigen::Index>(m)]
                             : bases[static_cast<std::size_t>(m)] *
                                   eta_h[static_cast<Eigen::Index>(k)]) -
                        lam1 * x2 * eta_h[static_cast<Eigen::Index>(k)];
                    const double xi2 =
                        ydx2[static_cast<Eigen::Index>(k)] -
                        (zw_se[sb.row_w[k]] + zf_se[sb.row_f[k]]);
                    const double first_piece =
                        By[static_cast<Eigen::Index>(k)] -
                        lam1 * x1bar[static_cast<Eigen::Index>(k)] * b1;
                    const double W2 = first_piece - 0.5 * (lam_eta2 + xi2);
                    W2vec[static_cast<Eigen::Index>(k)] = W2;
                    cov11 += x2 * sig_t[k];
                    cov12 += x1bar[static_cast<Eigen::Index>(k)] * sig_t[k] * W2;
                    sum2 += W2 * W2 * sig_t[k];
                }
                cov12 = 2.0 * cov12 / NTd;
                sims2_re.assign(static_cast<std::size_t>(options.se_nsim), 0.0);
                sims2_im.assign(static_cast<std::size_t>(options.se_nsim), 0.0);
                prof_.mark("cov_r1");
            }

            // simulations for the quadratic part. Oracle-faithful complex
            // semantics: negative sigma-tilde gives v = n*sqrt(sigma) with an
            // imaginary part (MATLAB proceeds in complex arithmetic and
            // var() of the complex sims is var(Re) + var(Im)); we expand the
            // bilinear forms in (v_re, v_im) with real solves.
            const int nsim = options.se_nsim;
            std::vector<double> sims_re(static_cast<std::size_t>(nsim));
            std::vector<double> sims_im(static_cast<std::size_t>(nsim));
            Eigen::VectorXd vre(static_cast<Eigen::Index>(n_py_sz));
            Eigen::VectorXd vim(static_cast<Eigen::Index>(n_py_sz));
            Eigen::VectorXd zw_re(N), zf_re(J), zw_im(N), zf_im(J);
            Eigen::VectorXd sum_m_a(static_cast<Eigen::Index>(M)),
                sum_m_b(static_cast<Eigen::Index>(M));
            bool any_imag_global = false;
            // bilinear pieces given the two solves:
            const auto quad_sub = [&](const Eigen::VectorXd& zwa, const Eigen::VectorXd& zfa,
                                      const Eigen::VectorXd& zwb, const Eigen::VectorXd& zfb) {
                double s1 = 0, s2a = 0, s2b = 0, s3 = 0, s4a = 0, s4b = 0;
                if (comp == 0) {
                    for (int f2 = 0; f2 < J; ++f2) {
                        s1 += solver.Df[f2] * zfa[f2] * zfb[f2];
                        s2a += solver.Df[f2] * zfa[f2];
                        s2b += solver.Df[f2] * zfb[f2];
                    }
                    return s1 - s2a * s2b / NTd;
                }
                if (comp == 2) {
                    for (int w2 = 0; w2 < N; ++w2) {
                        s3 += solver.Dw[w2] * zwa[w2] * zwb[w2];
                        s4a += solver.Dw[w2] * zwa[w2];
                        s4b += solver.Dw[w2] * zwb[w2];
                    }
                    return s3 - s4a * s4b / NTd;
                }
                // non-symmetric R(a, b) = sum centered(D zw_a) centered(F zf_b)
                double cross = 0.0;
                for (std::size_t m2 = 0; m2 < M; ++m2) {
                    cross += solver.m_c[m2] * zwa[sb.m_w[m2]] * zfb[sb.m_f[m2]];
                }
                for (int w2 = 0; w2 < N; ++w2) {
                    s4a += solver.Dw[w2] * zwa[w2];
                }
                for (int f2 = 0; f2 < J; ++f2) {
                    s2b += solver.Df[f2] * zfb[f2];
                }
                (void)s2a;
                (void)s4b;
                return cross - s4a * s2b / NTd;
            };
            // dot part, non-symmetric: G(a, b) = a' Lambda_B (I-Lambda_P)^{-1} (b - X z_b)
            const auto quad_dot = [&](const Eigen::VectorXd& va, const Eigen::VectorXd& vb,
                                      const Eigen::VectorXd& zwb, const Eigen::VectorXd& zfb) {
                double dot = 0.0;
                if (match_level) {
                    sum_m_a.setZero();
                    sum_m_b.setZero();
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m2 = sb.row_match[k];
                        sum_m_a[m2] += va[static_cast<Eigen::Index>(k)];
                        sum_m_b[m2] += vb[static_cast<Eigen::Index>(k)] -
                                       (zwb[sb.row_w[k]] + zfb[sb.row_f[k]]);
                    }
                    for (std::size_t m2 = 0; m2 < M; ++m2) {
                        if (bases[m2] == 0.0) continue;
                        const double T_m = static_cast<double>(sb.m_cnt[m2]);
                        const double gain = 1.0 + T_m * block_gain(static_cast<int>(m2));
                        dot += bases[m2] * sum_m_a[static_cast<Eigen::Index>(m2)] *
                               sum_m_b[static_cast<Eigen::Index>(m2)] * gain;
                    }
                } else {
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m2 = sb.row_match[k];
                        if (bases[static_cast<std::size_t>(m2)] == 0.0) continue;
                        const double den = 1.0 - bp[static_cast<std::size_t>(m2)];
                        const double rb = (vb[static_cast<Eigen::Index>(k)] -
                                           (zwb[sb.row_w[k]] + zfb[sb.row_f[k]])) /
                                          (den > 1e-12 ? den : 1.0);
                        dot += bases[static_cast<std::size_t>(m2)] *
                               va[static_cast<Eigen::Index>(k)] * rb;
                    }
                }
                return dot;
            };
            // (I-Lambda_P)^{-1}(v - X z) per observation (general block
            // form; needed for the diagonal part of Lambda_B2 in sims2)
            Eigen::VectorXd auxfull_re, auxfull_im, sum_r_m;
            if (do_ci) {
                auxfull_re.resize(static_cast<Eigen::Index>(n_py_sz));
                auxfull_im.resize(static_cast<Eigen::Index>(n_py_sz));
                sum_r_m.resize(static_cast<Eigen::Index>(M));
            }
            const auto build_auxfull = [&](const Eigen::VectorXd& vb,
                                           const Eigen::VectorXd& zwb,
                                           const Eigen::VectorXd& zfb,
                                           Eigen::VectorXd& out) {
                if (match_level) {
                    sum_r_m.setZero();
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const double r = vb[static_cast<Eigen::Index>(k)] -
                                         (zwb[sb.row_w[k]] + zfb[sb.row_f[k]]);
                        out[static_cast<Eigen::Index>(k)] = r;
                        sum_r_m[sb.row_match[k]] += r;
                    }
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m2 = sb.row_match[k];
                        out[static_cast<Eigen::Index>(k)] +=
                            block_gain(m2) * sum_r_m[m2];
                    }
                } else {
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        const int m2 = sb.row_match[k];
                        const double den =
                            1.0 - bp[static_cast<std::size_t>(m2)];
                        out[static_cast<Eigen::Index>(k)] =
                            (vb[static_cast<Eigen::Index>(k)] -
                             (zwb[sb.row_w[k]] + zfb[sb.row_f[k]])) /
                            (den > 1e-12 ? den : 1.0);
                    }
                }
            };
            const auto diag2 = [&](const Eigen::VectorXd& va,
                                   const Eigen::VectorXd& auxb) {
                double acc = 0.0;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    acc += x1bar[static_cast<Eigen::Index>(k)] *
                           x1bar[static_cast<Eigen::Index>(k)] *
                           va[static_cast<Eigen::Index>(k)] *
                           auxb[static_cast<Eigen::Index>(k)];
                }
                return acc;
            };
            // Sim-block batching (se_block, default 8; XHDFE_AKM_SE_BLOCK):
            // the nsim draws are independent with deterministic per-sim
            // seeds, so a block's re (and, on the complex-sigma path, im)
            // solves all go through the batched multi-RHS solver in one
            // call; per sim the draw stream, the scatters and every
            // post-solve scalar/vector reduction keep the exact sequential
            // sequences in ascending sim order, so results are identical for
            // ANY block size >= 1 and ANY thread count (2.14.0 JLA
            // guarantee). Extra memory is the per-lane person-year draw
            // vectors: se_block * n * 8 bytes (2x when the component has
            // negative sigma-tilde); reduce XHDFE_AKM_SE_BLOCK if tight.
            // XHDFE_AKM_SE_BLOCK=0 keeps the sequential loop verbatim.
            if (se_block >= 1) {
                [&]() XHDFE_AKM_OUTLINE {
                const int SB = se_block;
                bool comp_has_neg = false;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    if (sig_t[k] < 0.0) {
                        comp_has_neg = true;
                        break;
                    }
                }
                std::vector<Eigen::VectorXd> vre_l(static_cast<std::size_t>(SB));
                std::vector<Eigen::VectorXd> vim_l(static_cast<std::size_t>(SB));
                std::vector<Eigen::VectorXd> s_tw(static_cast<std::size_t>(2 * SB));
                std::vector<Eigen::VectorXd> s_tf(static_cast<std::size_t>(2 * SB));
                std::vector<Eigen::VectorXd> s_zw(static_cast<std::size_t>(2 * SB));
                std::vector<Eigen::VectorXd> s_zf(static_cast<std::size_t>(2 * SB));
                std::vector<char> imag_l(static_cast<std::size_t>(SB));
                std::vector<int> lane_im(static_cast<std::size_t>(SB));
                std::vector<const Eigen::VectorXd*> twp, tfp;
                std::vector<char> okv;
                for (int s0 = 0; s0 < nsim; s0 += SB) {
                    const int B = std::min(SB, nsim - s0);
                    int nl = B;
                    // --- per-sim draws + scatters (legacy sequences) -------
                    for (int b = 0; b < B; ++b) {
                        const int sidx = s0 + b;
                        std::uint64_t st =
                            stream_seed(options.seed ^ 0x5EEDC0FFEEULL,
                                        static_cast<std::uint64_t>(comp),
                                        static_cast<std::uint64_t>(sidx));
                        bool any_imag = false;
                        Eigen::VectorXd& vre_b = vre_l[static_cast<std::size_t>(b)];
                        vre_b.resize(static_cast<Eigen::Index>(n_py_sz));
                        if (comp_has_neg) {
                            Eigen::VectorXd& vim_b =
                                vim_l[static_cast<std::size_t>(b)];
                            vim_b.resize(static_cast<Eigen::Index>(n_py_sz));
                            for (std::size_t k = 0; k < n_py_sz; ++k) {
                                const double nk = normal_from_stream(st);
                                if (sig_t[k] >= 0.0) {
                                    vre_b[static_cast<Eigen::Index>(k)] =
                                        nk * std::sqrt(sig_t[k]);
                                    vim_b[static_cast<Eigen::Index>(k)] = 0.0;
                                } else {
                                    vre_b[static_cast<Eigen::Index>(k)] = 0.0;
                                    vim_b[static_cast<Eigen::Index>(k)] =
                                        nk * std::sqrt(-sig_t[k]);
                                    any_imag = true;
                                }
                            }
                        } else {
                            // no negative sigma-tilde anywhere: the
                            // sequential loop writes vim = 0 and never reads
                            // it; identical vre values, no vim lane needed.
                            for (std::size_t k = 0; k < n_py_sz; ++k) {
                                const double nk = normal_from_stream(st);
                                vre_b[static_cast<Eigen::Index>(k)] =
                                    nk * std::sqrt(sig_t[k]);
                            }
                        }
                        imag_l[static_cast<std::size_t>(b)] = any_imag ? 1 : 0;
                        Eigen::VectorXd& tw_b = s_tw[static_cast<std::size_t>(b)];
                        Eigen::VectorXd& tf_b = s_tf[static_cast<std::size_t>(b)];
                        tw_b = Eigen::VectorXd::Zero(N);
                        tf_b = Eigen::VectorXd::Zero(J);
                        for (std::size_t k = 0; k < n_py_sz; ++k) {
                            tw_b[sb.row_w[k]] += vre_b[static_cast<Eigen::Index>(k)];
                            tf_b[sb.row_f[k]] += vre_b[static_cast<Eigen::Index>(k)];
                        }
                        if (any_imag) {
                            const Eigen::VectorXd& vim_b =
                                vim_l[static_cast<std::size_t>(b)];
                            lane_im[static_cast<std::size_t>(b)] = nl;
                            Eigen::VectorXd& twi = s_tw[static_cast<std::size_t>(nl)];
                            Eigen::VectorXd& tfi = s_tf[static_cast<std::size_t>(nl)];
                            twi = Eigen::VectorXd::Zero(N);
                            tfi = Eigen::VectorXd::Zero(J);
                            for (std::size_t k = 0; k < n_py_sz; ++k) {
                                twi[sb.row_w[k]] += vim_b[static_cast<Eigen::Index>(k)];
                                tfi[sb.row_f[k]] += vim_b[static_cast<Eigen::Index>(k)];
                            }
                            ++nl;
                        } else {
                            lane_im[static_cast<std::size_t>(b)] = -1;
                        }
                    }
                    // --- one batched multi-RHS call for the whole block ----
                    twp.resize(static_cast<std::size_t>(nl));
                    tfp.resize(static_cast<std::size_t>(nl));
                    for (int l = 0; l < nl; ++l) {
                        twp[static_cast<std::size_t>(l)] = &s_tw[static_cast<std::size_t>(l)];
                        tfp[static_cast<std::size_t>(l)] = &s_tf[static_cast<std::size_t>(l)];
                        s_zw[static_cast<std::size_t>(l)].resize(N);
                        s_zf[static_cast<std::size_t>(l)].resize(J);
                    }
                    solver.solve_K_multi(twp, tfp, s_zw, s_zf, okv, cg_iters, true);
                    // --- per-sim post-solve reductions, ascending sidx -----
                    for (int b = 0; b < B; ++b) {
                        const int sidx = s0 + b;
                        const bool any_imag =
                            imag_l[static_cast<std::size_t>(b)] != 0;
                        any_imag_global = any_imag_global || any_imag;
                        if (!okv[static_cast<std::size_t>(b)]) {
                            res.converged = false;
                        }
                        const Eigen::VectorXd& vre_b = vre_l[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd& vim_b = vim_l[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd& zw_re_b = s_zw[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd& zf_re_b = s_zf[static_cast<std::size_t>(b)];
                        const Eigen::VectorXd* zwi = nullptr;
                        const Eigen::VectorXd* zfi = nullptr;
                        if (any_imag) {
                            const int li = lane_im[static_cast<std::size_t>(b)];
                            if (!okv[static_cast<std::size_t>(li)]) {
                                res.converged = false;
                            }
                            zwi = &s_zw[static_cast<std::size_t>(li)];
                            zfi = &s_zf[static_cast<std::size_t>(li)];
                        }
                        // conjugate semantics (MATLAB ' and complex cov):
                        // S_re = [R(re,re)+R(im,im)] - [G(re,re)+G(im,im)]
                        // S_im = [R(re,im)-R(im,re)] - [G(re,im)-G(im,re)]
                        const double Gd_rr = quad_dot(vre_b, vre_b, zw_re_b, zf_re_b);
                        double Gd_ii = 0.0, Gd_ri = 0.0, Gd_ir = 0.0;
                        double S_re = quad_sub(zw_re_b, zf_re_b, zw_re_b, zf_re_b) - Gd_rr;
                        double S_im = 0.0;
                        if (any_imag) {
                            Gd_ii = quad_dot(vim_b, vim_b, *zwi, *zfi);
                            Gd_ri = quad_dot(vre_b, vim_b, *zwi, *zfi);
                            Gd_ir = quad_dot(vim_b, vre_b, zw_re_b, zf_re_b);
                            S_re += quad_sub(*zwi, *zfi, *zwi, *zfi) - Gd_ii;
                            S_im = (quad_sub(zw_re_b, zf_re_b, *zwi, *zfi) -
                                    quad_sub(*zwi, *zfi, zw_re_b, zf_re_b)) -
                                   (Gd_ri - Gd_ir);
                        }
                        sims_re[static_cast<std::size_t>(sidx)] = S_re;
                        sims_im[static_cast<std::size_t>(sidx)] = S_im;
                        if (do_ci) {
                            // aux_SIM2 = -v' Lambda_B2 aux - lambda_1
                            // (v'x1bar)(x1bar'v); G2(a,b) = G(a,b) -
                            // lambda_1 d2(a,b), conjugate expansion as G.
                            build_auxfull(vre_b, zw_re_b, zf_re_b, auxfull_re);
                            const double dx_re = x1bar.dot(vre_b);
                            double s2_re, s2_im;
                            if (any_imag) {
                                build_auxfull(vim_b, *zwi, *zfi, auxfull_im);
                                const double dx_im = x1bar.dot(vim_b);
                                const double G2_rr = Gd_rr - lam1 * diag2(vre_b, auxfull_re);
                                const double G2_ii = Gd_ii - lam1 * diag2(vim_b, auxfull_im);
                                const double G2_ri = Gd_ri - lam1 * diag2(vre_b, auxfull_im);
                                const double G2_ir = Gd_ir - lam1 * diag2(vim_b, auxfull_re);
                                s2_re = -(G2_rr + G2_ii) -
                                        lam1 * (dx_re * dx_re + dx_im * dx_im);
                                s2_im = -(G2_ri - G2_ir);
                            } else {
                                const double G2_rr = Gd_rr - lam1 * diag2(vre_b, auxfull_re);
                                s2_re = -G2_rr - lam1 * dx_re * dx_re;
                                s2_im = 0.0;
                            }
                            sims2_re[static_cast<std::size_t>(sidx)] = s2_re;
                            sims2_im[static_cast<std::size_t>(sidx)] = s2_im;
                        }
                    }
                }
                }();
            } else {
            for (int sidx = 0; sidx < nsim; ++sidx) {
                std::uint64_t st = stream_seed(options.seed ^ 0x5EEDC0FFEEULL,
                                               static_cast<std::uint64_t>(comp),
                                               static_cast<std::uint64_t>(sidx));
                bool any_imag = false;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const double nk = normal_from_stream(st);
                    if (sig_t[k] >= 0.0) {
                        vre[static_cast<Eigen::Index>(k)] = nk * std::sqrt(sig_t[k]);
                        vim[static_cast<Eigen::Index>(k)] = 0.0;
                    } else {
                        vre[static_cast<Eigen::Index>(k)] = 0.0;
                        vim[static_cast<Eigen::Index>(k)] = nk * std::sqrt(-sig_t[k]);
                        any_imag = true;
                    }
                }
                any_imag_global = any_imag_global || any_imag;
                tw_se.setZero();
                tf_se.setZero();
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    tw_se[sb.row_w[k]] += vre[static_cast<Eigen::Index>(k)];
                    tf_se[sb.row_f[k]] += vre[static_cast<Eigen::Index>(k)];
                }
                if (!solver.solve_K(tw_se, tf_se, zw_re, zf_re, scw, scf, cg_iters, true)) {
                    res.converged = false;
                }
                if (any_imag) {
                    tw_se.setZero();
                    tf_se.setZero();
                    for (std::size_t k = 0; k < n_py_sz; ++k) {
                        tw_se[sb.row_w[k]] += vim[static_cast<Eigen::Index>(k)];
                        tf_se[sb.row_f[k]] += vim[static_cast<Eigen::Index>(k)];
                    }
                    if (!solver.solve_K(tw_se, tf_se, zw_im, zf_im, scw, scf, cg_iters, true)) {
                        res.converged = false;
                    }
                } else {
                    zw_im.setZero();
                    zf_im.setZero();
                }
                // conjugate semantics (MATLAB ' and complex cov):
                // S_re = [R(re,re)+R(im,im)] - [G(re,re)+G(im,im)]
                // S_im = [R(re,im)-R(im,re)] - [G(re,im)-G(im,re)]
                const double Gd_rr = quad_dot(vre, vre, zw_re, zf_re);
                double Gd_ii = 0.0, Gd_ri = 0.0, Gd_ir = 0.0;
                double S_re = quad_sub(zw_re, zf_re, zw_re, zf_re) - Gd_rr;
                double S_im = 0.0;
                if (any_imag) {
                    Gd_ii = quad_dot(vim, vim, zw_im, zf_im);
                    Gd_ri = quad_dot(vre, vim, zw_im, zf_im);
                    Gd_ir = quad_dot(vim, vre, zw_re, zf_re);
                    S_re += quad_sub(zw_im, zf_im, zw_im, zf_im) - Gd_ii;
                    S_im = (quad_sub(zw_re, zf_re, zw_im, zf_im) -
                            quad_sub(zw_im, zf_im, zw_re, zf_re)) -
                           (Gd_ri - Gd_ir);
                }
                sims_re[static_cast<std::size_t>(sidx)] = S_re;
                sims_im[static_cast<std::size_t>(sidx)] = S_im;
                if (do_ci) {
                    // aux_SIM2 = -v' Lambda_B2 aux - lambda_1 (v'x1bar)(x1bar'v)
                    // (the constant bias_part drops in the variance);
                    // G2(a,b) = G(a,b) - lambda_1 d2(a,b) with the same
                    // conjugate expansion as G.
                    build_auxfull(vre, zw_re, zf_re, auxfull_re);
                    const double dx_re = x1bar.dot(vre);
                    double s2_re, s2_im;
                    if (any_imag) {
                        build_auxfull(vim, zw_im, zf_im, auxfull_im);
                        const double dx_im = x1bar.dot(vim);
                        const double G2_rr = Gd_rr - lam1 * diag2(vre, auxfull_re);
                        const double G2_ii = Gd_ii - lam1 * diag2(vim, auxfull_im);
                        const double G2_ri = Gd_ri - lam1 * diag2(vre, auxfull_im);
                        const double G2_ir = Gd_ir - lam1 * diag2(vim, auxfull_re);
                        s2_re = -(G2_rr + G2_ii) -
                                lam1 * (dx_re * dx_re + dx_im * dx_im);
                        s2_im = -(G2_ri - G2_ir);
                    } else {
                        const double G2_rr = Gd_rr - lam1 * diag2(vre, auxfull_re);
                        s2_re = -G2_rr - lam1 * dx_re * dx_re;
                        s2_im = 0.0;
                    }
                    sims2_re[static_cast<std::size_t>(sidx)] = s2_re;
                    sims2_im[static_cast<std::size_t>(sidx)] = s2_im;
                }
            }
            }
            (void)any_imag_global;
            prof_.mark("se_sims");
            double mre = 0.0, mim = 0.0;
            for (int t2 = 0; t2 < nsim; ++t2) {
                mre += sims_re[static_cast<std::size_t>(t2)];
                mim += sims_im[static_cast<std::size_t>(t2)];
            }
            mre /= nsim;
            mim /= nsim;
            double svar = 0.0;
            for (int t2 = 0; t2 < nsim; ++t2) {
                const double dre = sims_re[static_cast<std::size_t>(t2)] - mre;
                const double dim = sims_im[static_cast<std::size_t>(t2)] - mim;
                svar += dre * dre + dim * dim;
            }
            svar /= static_cast<double>(nsim - 1);
            se_out = std::sqrt(std::max(0.0, (4.0 * SUM - svar))) / NTd;

            if (do_ci) {
                double m2re = 0.0, m2im = 0.0;
                for (int t2 = 0; t2 < nsim; ++t2) {
                    m2re += sims2_re[static_cast<std::size_t>(t2)];
                    m2im += sims2_im[static_cast<std::size_t>(t2)];
                }
                m2re /= nsim;
                m2im /= nsim;
                double second_bit = 0.0;
                for (int t2 = 0; t2 < nsim; ++t2) {
                    const double dre =
                        sims2_re[static_cast<std::size_t>(t2)] - m2re;
                    const double dim =
                        sims2_im[static_cast<std::size_t>(t2)] - m2im;
                    second_bit += dre * dre + dim * dim;
                }
                second_bit /= static_cast<double>(nsim - 1);
                const double cov22 =
                    (4.0 * sum2 - second_bit) / (NTd * NTd);
                res.eig_lambda1[comp] = lam1;
                res.eig_share1[comp] =
                    trace_est > 0.0 ? lam3[0] * lam3[0] / trace_est : 0.0;
                res.eig_share2[comp] =
                    trace_est > 0.0 ? lam3[1] * lam3[1] / trace_est : 0.0;
                res.eig_share3[comp] =
                    trace_est > 0.0 ? lam3[2] * lam3[2] / trace_est : 0.0;
                double mx = 0.0;
                for (std::size_t k = 0; k < n_py_sz; ++k) {
                    const double x2 = x1bar[static_cast<Eigen::Index>(k)] *
                                      x1bar[static_cast<Eigen::Index>(k)];
                    if (x2 > mx) mx = x2;
                }
                res.lindeberg_max_x1bar_sq[comp] = mx;
                res.b_1[comp] = b1;
                res.cov_r1_11[comp] = cov11;
                res.cov_r1_12[comp] = cov12;
                res.cov_r1_22[comp] = cov22;
                res.gamma_sq[comp] =
                    (lam1 * lam1 / (NTd * NTd)) * cov11 * cov11 / cov22;
                res.f_stat[comp] = b1 * b1 / cov11;
                res.theta_1[comp] =
                    theta_out - (lam1 / NTd) * (b1 * b1 - cov11);
                am_confidence_interval(NTd, lam1, res.gamma_sq[comp], cov11,
                                       cov12, cov22, b1, res.theta_1[comp],
                                       res.ci_lb[comp], res.ci_ub[comp],
                                       res.curvature[comp]);
            }
        };


        run_component(0, bfe, res.theta_c_var_psi, res.se_var_psi);
        run_component(1, bcov, res.theta_c_cov_alpha_psi, res.se_cov_alpha_psi);
        if (pe_identified) {
            run_component(2, bpe, res.theta_c_var_alpha, res.se_var_alpha);
        } else {
            res.theta_c_var_alpha = kNaN;
            res.se_var_alpha = kNaN;
            if (do_ci) {
                res.eig_lambda1[2] = kNaN;
                res.b_1[2] = kNaN;
                res.cov_r1_11[2] = kNaN;
                res.cov_r1_12[2] = kNaN;
                res.cov_r1_22[2] = kNaN;
                res.eig_share1[2] = kNaN;
                res.eig_share2[2] = kNaN;
                res.eig_share3[2] = kNaN;
                res.lindeberg_max_x1bar_sq[2] = kNaN;
                res.gamma_sq[2] = kNaN;
                res.f_stat[2] = kNaN;
                res.theta_1[2] = kNaN;
                res.ci_lb[2] = kNaN;
                res.ci_ub[2] = kNaN;
                res.curvature[2] = kNaN;
            }
            res.notes += "var(alpha) SE not identified at match level with stayers (oracle rule). ";
        }
        res.solver_iterations = cg_iters + solver.cuda_iters;
    }

    // ---- Row-level outputs -----------------------------------------------
    res.row_worker.resize(static_cast<Eigen::Index>(R));
    res.row_firm.resize(static_cast<Eigen::Index>(R));
    for (std::size_t r = 0; r < R; ++r) {
        res.row_worker[static_cast<Eigen::Index>(r)] =
            sb.worker_orig[static_cast<std::size_t>(row_w_of[r])];
        res.row_firm[static_cast<Eigen::Index>(r)] =
            sb.firm_orig[static_cast<std::size_t>(row_f_of[r])];
    }
    {
        double mx = 0.0, sum = 0.0;
        long long cnt = 0;
        for (std::size_t r = 0; r < R; ++r) {
            if (match_level && row_stayer[r]) {
                continue;
            }
            mx = std::max(mx, Pii[static_cast<Eigen::Index>(r)]);
            sum += Pii[static_cast<Eigen::Index>(r)];
            ++cnt;
        }
        res.max_pii = mx;
        res.mean_pii = cnt > 0 ? sum / static_cast<double>(cnt) : 0.0;
    }

    // ---- KSS lincom (Proposition 1): firm effects projected on [1, Z] ----
    if (Z != nullptr && Z->cols() > 0) {
        if (Z->rows() != y.size()) {
            throw std::runtime_error("Z must have the same number of rows as y");
        }
        if (match_level && solver.m_sqrt_c.empty()) solver.ensure_sqrt_c();
        const int r = static_cast<int>(Z->cols());
        Eigen::MatrixXd Zt(static_cast<Eigen::Index>(n_kept), r + 1);
        Zt.col(0).setOnes();
        for (std::size_t k = 0; k < n_kept; ++k) {
            Zt.row(static_cast<Eigen::Index>(k)).tail(r) = Z->row(sb.rows[k]);
        }
        // person-year firm effect (level shift only moves the constant)
        Eigen::VectorXd wy(static_cast<Eigen::Index>(n_kept));
        for (std::size_t k = 0; k < n_kept; ++k) {
            wy[static_cast<Eigen::Index>(k)] = psi[sb.row_f[k]];
        }
        const Eigen::MatrixXd ZtZ = Zt.transpose() * Zt;
        const Eigen::LDLT<Eigen::MatrixXd> zz(ZtZ);
        const Eigen::VectorXd num = zz.solve(Zt.transpose() * wy);
        res.lincom_coef.resize(r);
        res.lincom_se_kss.resize(r);
        res.lincom_se_white.resize(r);
        res.lincom_t.resize(r);
        const int se_block = akm_se_block_env();
        if (se_block >= 1) {
            [&]() XHDFE_AKM_OUTLINE {
            // Batch the r independent column solves through the multi-RHS
            // solver in one call (tw is identically zero -> nullptr lanes,
            // exactly the JLA fe-lane convention); the per-column scalar
            // reductions keep the sequential order (ascending qq).
            std::vector<Eigen::VectorXd> l_tf(static_cast<std::size_t>(r));
            std::vector<Eigen::VectorXd> l_zw(static_cast<std::size_t>(r));
            std::vector<Eigen::VectorXd> l_zf(static_cast<std::size_t>(r));
            for (int qq = 0; qq < r; ++qq) {
                Eigen::VectorXd eq = Eigen::VectorXd::Zero(r + 1);
                eq[qq + 1] = 1.0;
                const Eigen::VectorXd vw = Zt * zz.solve(eq);   // n_py vector
                Eigen::VectorXd& tf_q = l_tf[static_cast<std::size_t>(qq)];
                tf_q = Eigen::VectorXd::Zero(J);
                for (std::size_t k = 0; k < n_kept; ++k) {
                    tf_q[sb.row_f[k]] += vw[static_cast<Eigen::Index>(k)];
                }
                l_zw[static_cast<std::size_t>(qq)].resize(N);
                l_zf[static_cast<std::size_t>(qq)].resize(J);
            }
            std::vector<const Eigen::VectorXd*> twp(static_cast<std::size_t>(r),
                                                    nullptr);
            std::vector<const Eigen::VectorXd*> tfp(static_cast<std::size_t>(r));
            for (int qq = 0; qq < r; ++qq) {
                tfp[static_cast<std::size_t>(qq)] = &l_tf[static_cast<std::size_t>(qq)];
            }
            std::vector<char> okv;
            solver.solve_K_multi(twp, tfp, l_zw, l_zf, okv, cg_iters, true);
            for (int qq = 0; qq < r; ++qq) {
                if (!okv[static_cast<std::size_t>(qq)]) {
                    res.converged = false;
                    res.notes += "lincom solve did not converge. ";
                }
                const Eigen::VectorXd& zw_l = l_zw[static_cast<std::size_t>(qq)];
                const Eigen::VectorXd& zf_l = l_zf[static_cast<std::size_t>(qq)];
                double den_kss = 0.0, den_white = 0.0;
                for (std::size_t m = 0; m < R; ++m) {
                    const int w = row_w_of[m];
                    const int f = row_f_of[m];
                    const double sw = match_level ? solver.m_sqrt_c[m] : 1.0;
                    const double xr = sw * (zw_l[w] + zf_l[f]);
                    den_kss += sigma[static_cast<Eigen::Index>(m)] * xr * xr;
                    den_white += eta[static_cast<Eigen::Index>(m)] *
                                 eta[static_cast<Eigen::Index>(m)] * xr * xr;
                }
                res.lincom_coef[qq] = num[qq + 1];
                res.lincom_se_kss[qq] = std::sqrt(den_kss);
                res.lincom_se_white[qq] = std::sqrt(den_white);
                res.lincom_t[qq] = num[qq + 1] / std::sqrt(den_kss);
            }
            }();
        } else {
        Eigen::VectorXd tw0 = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd tf(J), zw_l(N), zf_l(J), scr_w(N), scr_f(J);
        for (int qq = 0; qq < r; ++qq) {
            Eigen::VectorXd eq = Eigen::VectorXd::Zero(r + 1);
            eq[qq + 1] = 1.0;
            const Eigen::VectorXd vw = Zt * zz.solve(eq);   // n_py vector
            tf.setZero();
            for (std::size_t k = 0; k < n_kept; ++k) {
                tf[sb.row_f[k]] += vw[static_cast<Eigen::Index>(k)];
            }
            if (!solver.solve_K(tw0, tf, zw_l, zf_l, scr_w, scr_f, cg_iters, true)) {
                res.converged = false;
                res.notes += "lincom solve did not converge. ";
            }
            double den_kss = 0.0, den_white = 0.0;
            for (std::size_t m = 0; m < R; ++m) {
                const int w = row_w_of[m];
                const int f = row_f_of[m];
                const double sw = match_level ? solver.m_sqrt_c[m] : 1.0;
                const double xr = sw * (zw_l[w] + zf_l[f]);
                den_kss += sigma[static_cast<Eigen::Index>(m)] * xr * xr;
                den_white += eta[static_cast<Eigen::Index>(m)] *
                             eta[static_cast<Eigen::Index>(m)] * xr * xr;
            }
            res.lincom_coef[qq] = num[qq + 1];
            res.lincom_se_kss[qq] = std::sqrt(den_kss);
            res.lincom_se_white[qq] = std::sqrt(den_white);
            res.lincom_t[qq] = num[qq + 1] / std::sqrt(den_kss);
        }
        }
        res.solver_iterations = cg_iters + solver.cuda_iters;
        prof_.mark("lincom");
    }

    // No internal consumer remains after lincom. Move these O(R) arrays into
    // the result instead of copying them (hundreds of MB at AKM scale).
    res.pii = std::move(Pii);
    res.sigma_i = std::move(sigma);
    res.row_weight = std::move(row_wgt);

    // Observation-level effects, psi centered to zero person-year mean.
    res.alpha.resize(static_cast<Eigen::Index>(n_kept));
    res.psi.resize(static_cast<Eigen::Index>(n_kept));
    for (std::size_t k = 0; k < n_kept; ++k) {
        res.alpha[static_cast<Eigen::Index>(k)] = alpha[sb.row_w[k]] + psi_bar;
        res.psi[static_cast<Eigen::Index>(k)] = psi[sb.row_f[k]] - psi_bar;
    }

    if (!guarded_finite(res.kss.var_psi) || !guarded_finite(res.kss.var_alpha) ||
        !guarded_finite(res.kss.cov_alpha_psi)) {
        res.converged = false;
        res.notes += "non-finite corrected components. ";
    }

    return res;
}

}  // namespace akm

namespace gelbach {

namespace {

Eigen::MatrixXd cluster_meat(const Eigen::MatrixXd& Z, const Eigen::VectorXi* codes,
                             int n_clusters) {
    if (codes == nullptr) {
        return Z.transpose() * Z;
    }
    Eigen::MatrixXd sums = Eigen::MatrixXd::Zero(n_clusters, Z.cols());
    for (Eigen::Index i = 0; i < Z.rows(); ++i) {
        sums.row((*codes)[i]) += Z.row(i);
    }
    return sums.transpose() * sums;
}

// Post-recovery normalization of per-dimension FE contribution vectors,
// replicating the v11 Component-style default: shift the per-mobility-
// component mean of FE2 into FE1, then center FE1 (and dims >= 2) globally;
// a single dimension is just centered. Shared by the fast warm path and the
// legacy-path certification cross-check so both compare in one convention.
void normalize_fe_effects_component_style(std::vector<Eigen::VectorXd>& fe,
                                          const std::vector<Eigen::VectorXi>& fes_k,
                                          const Eigen::VectorXd* wk_ptr) {
    if (fe.empty()) {
        return;
    }
    const Eigen::Index nk = fe[0].size();
    // Neumaier-compensated weighted mean (matches v11 weighted_mean).
    auto wmean = [&](const Eigen::VectorXd& v) -> double {
        double sum = 0.0, comp = 0.0, wsum = 0.0, wcomp = 0.0;
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            const double wi = wk_ptr ? (*wk_ptr)[i] : 1.0;
            const double term = v[i] * wi;
            const double t = sum + term;
            comp += (std::abs(sum) >= std::abs(term)) ? ((sum - t) + term)
                                                      : ((term - t) + sum);
            sum = t;
            const double tw = wsum + wi;
            wcomp += (std::abs(wsum) >= std::abs(wi)) ? ((wsum - tw) + wi)
                                                      : ((wi - tw) + wsum);
            wsum = tw;
        }
        const double denom = wsum + wcomp;
        return denom > 0.0 ? (sum + comp) / denom : 0.0;
    };
    const std::size_t rdims = fe.size();
    if (rdims == 1) {
        fe[0].array() -= wmean(fe[0]);
    } else if (rdims >= 2) {
        // Compact raw level codes first (direct-index when dense, hash otherwise).
        auto compact_ids = [nk](const Eigen::VectorXi& f, std::vector<int>& out) -> int {
            out.resize(static_cast<std::size_t>(nk));
            long long mn = std::numeric_limits<long long>::max();
            long long mx = std::numeric_limits<long long>::min();
            for (Eigen::Index i = 0; i < nk; ++i) {
                mn = std::min<long long>(mn, f[i]);
                mx = std::max<long long>(mx, f[i]);
            }
            const long long range = (nk > 0) ? (mx - mn + 1) : 0;
            int next = 0;
            if (range > 0 && range <= 4LL * nk + 1024) {
                std::vector<int> lut(static_cast<std::size_t>(range), -1);
                for (Eigen::Index i = 0; i < nk; ++i) {
                    int& slot = lut[static_cast<std::size_t>(f[i] - mn)];
                    if (slot < 0) {
                        slot = next++;
                    }
                    out[static_cast<std::size_t>(i)] = slot;
                }
            } else {
                std::unordered_map<int, int> lut;
                lut.reserve(static_cast<std::size_t>(nk) / 4 + 16);
                for (Eigen::Index i = 0; i < nk; ++i) {
                    auto ins = lut.emplace(f[i], next);
                    if (ins.second) {
                        ++next;
                    }
                    out[static_cast<std::size_t>(i)] = ins.first->second;
                }
            }
            return next;
        };
        std::vector<int> c0, c1;
        const int K0 = compact_ids(fes_k[0], c0);
        const int K1 = compact_ids(fes_k[1], c1);
        // Union-find over the bipartite (fe0, fe1) level graph.
        std::vector<int> parent(static_cast<std::size_t>(K0) +
                                static_cast<std::size_t>(K1));
        for (std::size_t i = 0; i < parent.size(); ++i) {
            parent[i] = static_cast<int>(i);
        }
        auto find_root = [&parent](int a) -> int {
            while (parent[static_cast<std::size_t>(a)] != a) {
                parent[static_cast<std::size_t>(a)] =
                    parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
                a = parent[static_cast<std::size_t>(a)];
            }
            return a;
        };
        for (Eigen::Index i = 0; i < nk; ++i) {
            const int ra = find_root(c0[static_cast<std::size_t>(i)]);
            const int rb = find_root(K0 + c1[static_cast<std::size_t>(i)]);
            if (ra != rb) {
                parent[static_cast<std::size_t>(rb)] = ra;
            }
        }
        std::vector<int> comp_id(parent.size(), -1);
        int num_components = 0;
        Eigen::VectorXi components(nk);
        for (Eigen::Index i = 0; i < nk; ++i) {
            const int r = find_root(c0[static_cast<std::size_t>(i)]);
            int& cid = comp_id[static_cast<std::size_t>(r)];
            if (cid < 0) {
                cid = num_components++;
            }
            components[i] = cid;
        }
        std::vector<double> comp_sum(static_cast<std::size_t>(num_components), 0.0);
        std::vector<double> comp_w(static_cast<std::size_t>(num_components), 0.0);
        const double* fe1_ptr = fe[1].data();
        if (wk_ptr) {
            const double* w = wk_ptr->data();
            for (Eigen::Index i = 0; i < nk; ++i) {
                const std::size_t c = static_cast<std::size_t>(components[i]);
                comp_sum[c] += fe1_ptr[i] * w[i];
                comp_w[c] += w[i];
            }
        } else {
            for (Eigen::Index i = 0; i < nk; ++i) {
                const std::size_t c = static_cast<std::size_t>(components[i]);
                comp_sum[c] += fe1_ptr[i];
                comp_w[c] += 1.0;
            }
        }
        for (Eigen::Index i = 0; i < nk; ++i) {
            const std::size_t c = static_cast<std::size_t>(components[i]);
            const double shift = comp_w[c] > 0.0 ? comp_sum[c] / comp_w[c] : 0.0;
            fe[1](i) -= shift;
            fe[0](i) += shift;
        }
        fe[0].array() -= wmean(fe[0]);
        for (std::size_t d = 2; d < rdims; ++d) {
            fe[d].array() -= wmean(fe[d]);
        }
    }
}

}  // namespace

GelbachResult decompose(const Eigen::VectorXd& y_in,
                        const Eigen::MatrixXd& X1_in,
                        const Eigen::MatrixXd& X2_in,
                        const std::vector<int>& x2_group_sizes,
                        const std::vector<Eigen::VectorXi>& fes_in,
                        const Eigen::VectorXi* cluster,
                        const GelbachOptions& options,
                        const Eigen::VectorXd* weights,
                        bool freq_weights) {
    akm::AkmPhaseProfiler gprof_;
    const Eigen::Index n0 = y_in.size();
    const int p = static_cast<int>(X1_in.cols());
    const int q = static_cast<int>(X2_in.cols());
    if (X1_in.rows() != n0 || (q > 0 && X2_in.rows() != n0)) {
        throw std::runtime_error("gelbach: X1/X2 must have the same rows as y");
    }
    {
        int acc = 0;
        for (int g : x2_group_sizes) acc += g;
        if (acc != q) throw std::runtime_error("gelbach: group sizes do not sum to X2 columns");
    }
    for (const auto& f : fes_in) {
        if (f.size() != n0) throw std::runtime_error("gelbach: fe length mismatch");
    }
    if (options.vce == GelbachVce::Cluster && cluster == nullptr) {
        throw std::runtime_error("gelbach: cluster vce requires cluster ids");
    }
    if (x2_group_sizes.empty() && fes_in.empty()) {
        throw std::runtime_error("gelbach: provide at least one x2 group or FE dimension");
    }

    GelbachResult res;
    const int G = static_cast<int>(x2_group_sizes.size() + fes_in.size());
    const int k1 = p + 1;

    // ---- full model ------------------------------------------------------
    Eigen::MatrixXd X_full(n0, p + q);
    X_full.leftCols(p) = X1_in;
    if (q > 0) X_full.rightCols(q) = X2_in;
    if (weights != nullptr && weights->size() != n0) {
        throw std::runtime_error("gelbach: weights length mismatch");
    }
    // Full fit strategy: retain_fixed_effects makes the absorber ineligible
    // for the accelerated (auto-MLSMR) methods, so the 7+-column joint fit
    // runs several times slower. Instead: (a) run the joint fit FAST without
    // retention; (b) when FE blocks are needed, recover the per-observation
    // FE contributions with a second single-column fit on the partialled
    // outcome y - X b_full — the same recovery code path and normalization
    // the retained fit uses, at ~1/(p+q) of the absorption cost.
    // XHDFE_GELBACH_FAST_FIT=0 restores the single retained fit.
    bool gel_fast_fit = true;
    if (const char* gf = std::getenv("XHDFE_GELBACH_FAST_FIT")) {
        gel_fast_fit = !(gf[0] == '0' && gf[1] == '\0');
    }
    HdfeOptions ho;
    ho.se_type = StandardErrorType::Homoskedastic;
    ho.retain_fixed_effects = !fes_in.empty() && !gel_fast_fit;
    ho.num_threads = options.num_threads;
    ho.weights_are_frequencies = freq_weights;
    v11::HdfeRegressorV11 reg(ho);
    reg.fit(y_in, X_full, fes_in, weights);
    gprof_.mark("gel_full_fit");
    // (b) separate FE recovery on the kept sample (fast-fit mode only).
    // A retained 1-column fit would spend ~95% of its time in non-accelerated
    // GS sweeps whose alphas get discarded when the hybrid check falls back to
    // the map recovery — so call the map recovery directly on partial and
    // replicate the v11 post-recovery normalization (Component style default).
    std::vector<Eigen::VectorXd> fe_effects_sep;
    Eigen::MatrixXd W_pre;  // demeaned [x1, x2] reused by the sandwich VCE below
    std::unique_ptr<v11::HdfeRegressorV11> reg_ret;  // reference-fallback fit
    if (gel_fast_fit && !fes_in.empty()) {
        const Eigen::VectorXi& keep0 = reg.results().sample_index;
        const Eigen::Index nk = keep0.size();
        Eigen::VectorXd partial(nk);
        const Eigen::VectorXd& cf = reg.results().coefficients;
        const int ncf = static_cast<int>(cf.size()) - (cf.size() == p + q + 1 ? 1 : 0);
        std::vector<Eigen::VectorXi> fes_k(fes_in.size());
        for (std::size_t d = 0; d < fes_in.size(); ++d) {
            fes_k[d].resize(nk);
        }
        Eigen::VectorXd w_k;
        if (weights != nullptr) {
            w_k.resize(nk);
        }
#pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < nk; ++i) {
            const Eigen::Index r0 = keep0[i];
            double acc = y_in[r0];
            for (int c = 0; c < ncf && c < p + q; ++c) {
                acc -= X_full(r0, c) * cf[c];
            }
            partial[i] = acc;
            for (std::size_t d = 0; d < fes_in.size(); ++d) {
                fes_k[d][i] = fes_in[d][r0];
            }
            if (weights != nullptr) {
                w_k[i] = (*weights)[r0];
            }
        }
        const Eigen::VectorXd* wk_ptr = weights != nullptr ? &w_k : nullptr;
        // Warm start: one MLSMR solve on partial (~8 Krylov iterations) gives
        // near-converged per-level alphas; the exact map recovery then only
        // polishes the small residual (1-2 sweeps) under its usual stopping
        // rule. From-scratch GS recovery on this graph needs ~70 sweeps.
        // Fail-closed: any non-convergence falls back to from-scratch recovery.
        bool warm_recovery = !fes_k.empty();
        if (const char* wr = std::getenv("XHDFE_GELBACH_WARM_RECOVERY")) {
            warm_recovery = warm_recovery && !(wr[0] == '0' && wr[1] == '\0');
        }
        bool warm_done = false;
        // For sandwich VCEs the within transform needs demeaned [x1, x2] as
        // well — feed those columns to the SAME MLSMR call so one
        // preconditioner build and one batched multi-RHS solve serve both the
        // FE recovery warm start (y-slot alphas) and the within matrix W.
        const bool want_W =
            options.vce != GelbachVce::Unadjusted && (p + q) > 0;
        if (warm_recovery) {
            HdfeOptions mo = ho;
            mo.absorption_method = AbsorptionMethod::Mlsmr;
            mo.retain_fixed_effects = true;
            // No ridge: with the default krylov_lambda the recovered alphas
            // carry a lambda*A^{-1}*alpha bias that explodes in the slow
            // directions of ill-conditioned FE graphs (observed 1e-5-level
            // per-dim splits) while the GS gradient stays ~lambda*alpha/counts
            // — invisible to any polish-side certificate. Solving the exact
            // normal equations removes the bias at the source.
            mo.krylov_lambda = 0.0;
            Eigen::MatrixXd xin(nk, want_W ? (p + q) : 0);
            if (want_W) {
#pragma omp parallel for schedule(static)
                for (Eigen::Index i = 0; i < nk; ++i) {
                    xin.row(i) = X_full.row(keep0[i]);
                }
            }
            detail::AbsorptionResult ares =
                detail::absorb_fixed_effects_mlsmr(partial, xin, fes_k, wk_ptr, mo);
            const std::size_t rd = fes_k.size();
            if (ares.converged && ares.fe_means.size() == rd &&
                ares.fe_group_ids.size() == rd && ares.fe_levels.size() == rd) {
                std::vector<Eigen::VectorXd> warm(rd);
                Eigen::VectorXd resid = partial;
                for (std::size_t d = 0; d < rd; ++d) {
                    warm[d].resize(nk);
                    const std::vector<int>& gid = ares.fe_group_ids[d];
                    const Eigen::VectorXd& lev = ares.fe_means[d];
#pragma omp parallel for schedule(static)
                    for (Eigen::Index i = 0; i < nk; ++i) {
                        const double v = lev[gid[static_cast<std::size_t>(i)]];
                        warm[d][i] = v;
                        resid[i] -= v;
                    }
                }
                // Convergence gate (fail-closed): accept the warm path only
                // when the polish CONVERGES at the estimation tolerance within
                // a few sweeps; anything else falls back to the full retained
                // fit (bitwise the reference path). Note on ill-conditioned FE
                // graphs: the accepted warm solution certifies a ~tol gradient
                // (tighter than the legacy retained path's fe_tolerance
                // criterion), so the per-dimension FE split can differ from
                // pre-2.14.1 output at the fe_tolerance/mu_min level there —
                // the warm answer is the more-converged one (see the zigzag
                // precedent). XHDFE_GELBACH_FAST_FIT=0 restores legacy output.
                HdfeOptions po = ho;
                po.fe_tolerance = ho.tol > 0.0 ? ho.tol : 1e-8;
                detail::FeRecoveryResult rec = detail::recover_fixed_effects_group_ids(
                    resid, ares.fe_group_ids, ares.fe_levels, wk_ptr, po,
                    ares.fe_weight_sums.empty() ? nullptr : &ares.fe_weight_sums);
                if (gprof_.on) {
                    std::fprintf(stderr,
                                 "[akm-profile]   gel gate: mlsmr_it=%d mlsmr_ok=%d "
                                 "polish_it=%d polish_delta=%.3e polish_ok=%d\n",
                                 ares.iterations, ares.converged ? 1 : 0,
                                 rec.iterations, rec.max_delta,
                                 rec.converged ? 1 : 0);
                }
                if (rec.converged && rec.contributions.size() == rd &&
                    rec.iterations <= 3) {
                    fe_effects_sep = std::move(warm);
                    for (std::size_t d = 0; d < rd; ++d) {
                        fe_effects_sep[d].noalias() += rec.contributions[d];
                    }
                    warm_done = true;
                    if (want_W) {
                        W_pre = std::move(ares.X_tilde);
                    }
                }
            }
        }
        if (!warm_done) {
            // Reference fallback: rerun the FULL retained fit on the original
            // inputs (the exact pre-fast-path code) so every downstream
            // quantity — coefficients, covariance, residuals, FE effects —
            // is bitwise the reference on graphs the warm start cannot
            // certify. CAUTION (adversarial audit G1, 09jul2026): the
            // retained fit's own per-dimension FE split carries a WEAKER
            // certificate than the fast path (GS map recovery; its
            // update-size stopping rule is blind to slow graph modes and can
            // return a materially wrong split with converged=1 on
            // ill-conditioned FE graphs). The numbers are kept bitwise for
            // A/B reproducibility, but the split is cross-checked against
            // one MLSMR exact-normal-equations solve further down and
            // flagged loudly when it fails.
            HdfeOptions hr = ho;
            hr.retain_fixed_effects = true;
            reg_ret.reset(new v11::HdfeRegressorV11(hr));
            reg_ret->fit(y_in, X_full, fes_in, weights);
            gel_fast_fit = false;
            gprof_.mark("gel_fe_recovery");
        }
        // Post-recovery normalization: warm path only (the retained-fit
        // fallback normalizes inside v11).
        if (warm_done) {
            normalize_fe_effects_component_style(fe_effects_sep, fes_k, wk_ptr);
            gprof_.mark("gel_fe_recovery");
        }  // warm_done normalization
    }
    // regu: the regressor every downstream quantity reads from — the fast fit
    // normally, or the reference-fallback retained fit when the gate rejected.
    const v11::HdfeRegressorV11& regu = reg_ret ? *reg_ret : reg;
    if (!regu.results().converged) {
        res.converged = false;
        res.notes += "full-model absorption did not converge. ";
    }
    const Eigen::VectorXd coefs = regu.results().coefficients;
    const bool has_cons = coefs.size() == p + q + 1;
    res.b_full = coefs.head(p);
    const Eigen::VectorXd b2 = coefs.segment(p, q);
    const Eigen::MatrixXd V_hom = regu.results().covariance;
    const Eigen::VectorXd e = regu.results().residuals;
    const double df_full = regu.results().df_resid;
    res.df_full = df_full;

    // restrict to the kept sample (singleton dropping)
    const Eigen::VectorXi& keep = regu.results().sample_index;
    const Eigen::Index n = keep.size();
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X1(n, p);
    Eigen::MatrixXd X2(n, q);
    std::vector<Eigen::VectorXi> fes(fes_in.size());
    Eigen::VectorXi ccodes;
    int n_clusters = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        y[i] = y_in[keep[i]];
        X1.row(i) = X1_in.row(keep[i]);
        if (q > 0) X2.row(i) = X2_in.row(keep[i]);
    }
    for (std::size_t d = 0; d < fes_in.size(); ++d) {
        fes[d].resize(n);
        for (Eigen::Index i = 0; i < n; ++i) fes[d][i] = fes_in[d][keep[i]];
    }
    Eigen::VectorXd w_raw;
    if (weights != nullptr) {
        w_raw.resize(n);
        for (Eigen::Index i2 = 0; i2 < n; ++i2) w_raw[i2] = (*weights)[keep[i2]];
    }
    // Legacy / gate-reject fallback certification (adversarial audit G1,
    // 09jul2026): the retained fit recovers the per-dimension FE split with
    // GS map sweeps whose update-size stopping rule is blind to the slow
    // modes of ill-conditioned FE graphs — it can return a materially wrong
    // split (sign flips observed) while reporting converged=1 and a clean
    // identity gap. Cross-check the split against one MLSMR solve of the
    // exact normal equations (the only certificate that held in the audits)
    // plus the same gated polish the fast path uses. The reported NUMBERS
    // are never changed here (XHDFE_GELBACH_FAST_FIT=0 stays bitwise
    // A/B-reproducible); a failed cross-check sets converged=false and says
    // so in notes, and an unavailable cross-check is noted softly.
    if (!gel_fast_fit && !fes_in.empty() && n > 0) {
        const auto& fe_eff = regu.results().fe_effects;
        if (fe_eff.size() == fes_in.size() &&
            fe_eff[0].size() == n) {
            Eigen::VectorXd partial = e;
            for (const auto& fd : fe_eff) partial += fd;
            const Eigen::VectorXd* wk_cert = weights != nullptr ? &w_raw : nullptr;
            HdfeOptions mo = ho;
            mo.absorption_method = AbsorptionMethod::Mlsmr;
            mo.retain_fixed_effects = true;
            mo.krylov_lambda = 0.0;
            Eigen::MatrixXd xin(n, 0);
            detail::AbsorptionResult ares =
                detail::absorb_fixed_effects_mlsmr(partial, xin, fes, wk_cert, mo);
            bool check_ran = false;
            const std::size_t rd = fes.size();
            if (ares.converged && ares.fe_means.size() == rd &&
                ares.fe_group_ids.size() == rd && ares.fe_levels.size() == rd) {
                std::vector<Eigen::VectorXd> cand(rd);
                Eigen::VectorXd resid = partial;
                for (std::size_t d = 0; d < rd; ++d) {
                    cand[d].resize(n);
                    const std::vector<int>& gid = ares.fe_group_ids[d];
                    const Eigen::VectorXd& lev = ares.fe_means[d];
                    for (Eigen::Index i = 0; i < n; ++i) {
                        const double v = lev[gid[static_cast<std::size_t>(i)]];
                        cand[d][i] = v;
                        resid[i] -= v;
                    }
                }
                HdfeOptions po = ho;
                po.fe_tolerance = ho.tol > 0.0 ? ho.tol : 1e-8;
                detail::FeRecoveryResult rec = detail::recover_fixed_effects_group_ids(
                    resid, ares.fe_group_ids, ares.fe_levels, wk_cert, po,
                    ares.fe_weight_sums.empty() ? nullptr : &ares.fe_weight_sums);
                if (rec.converged && rec.contributions.size() == rd) {
                    for (std::size_t d = 0; d < rd; ++d) {
                        cand[d].noalias() += rec.contributions[d];
                    }
                    normalize_fe_effects_component_style(cand, fes, wk_cert);
                    double rel = 0.0;
                    for (std::size_t d = 0; d < rd; ++d) {
                        const double scale =
                            std::max(1.0, fe_eff[d].cwiseAbs().maxCoeff());
                        rel = std::max(
                            rel,
                            (cand[d] - fe_eff[d]).cwiseAbs().maxCoeff() / scale);
                    }
                    check_ran = true;
                    if (rel > 1e-3) {
                        res.converged = false;
                        res.notes +=
                            "per-dimension FE split failed the exact-normal-"
                            "equations cross-check on this graph "
                            "(ill-conditioned); FE-block deltas from this "
                            "legacy/fallback path are unreliable — use the "
                            "default fast path. ";
                    }
                }
            }
            if (!check_ran) {
                res.notes +=
                    "per-dimension FE split could not be independently "
                    "certified on this graph. ";
            }
        }
    }
    const Eigen::VectorXi* ccodes_ptr = nullptr;
    if (cluster != nullptr) {
        if (cluster->size() != n0) throw std::runtime_error("gelbach: cluster length mismatch");
        // vce is the single source of truth: cluster ids supplied with a
        // non-cluster vce are ignored (previously the meat and multipliers
        // silently switched to the clustered estimator while the caller had
        // asked for — and the output was labelled — robust/unadjusted).
        if (options.vce == GelbachVce::Cluster) {
            std::unordered_map<int, int> lev;
            ccodes.resize(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = lev.find((*cluster)[keep[i]]);
                if (it == lev.end()) it = lev.emplace((*cluster)[keep[i]], n_clusters++).first;
                ccodes[i] = it->second;
            }
            ccodes_ptr = &ccodes;
        }
    }
    res.n_obs = static_cast<long long>(n);

    // weights: wq = moment weight (aweights normalized to sum n), sf = score
    // factor for the sandwich meat, n_eff = effective observation count
    // (aweights: n; fweights: sum w). Matches b1x2 / Stata _robust
    // conventions exactly (validated). fweights score factor: sqrt(w) is
    // correct only for the ROBUST meat, where each row's outer product then
    // carries weight w (a row stands for w identical observations); under
    // clustering the meat sums score rows within a cluster BEFORE the outer
    // product, so a row standing for w copies must contribute w*z, not
    // sqrt(w)*z — scoring sqrt(w) there understates the SEs (~sqrt(w-bar)x;
    // fw-x-cluster audit finding F2, 09jul2026, fixed to match the
    // row-expanded definition and b1x2 on expanded data).
    Eigen::VectorXd wq = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd sf = Eigen::VectorXd::Ones(n);
    double n_eff = static_cast<double>(n);
    if (weights != nullptr) {
        if (freq_weights) {
            wq = w_raw;
            sf = options.vce == GelbachVce::Cluster ? w_raw
                                                    : w_raw.cwiseSqrt();
            n_eff = w_raw.sum();
        } else {
            wq = w_raw * (static_cast<double>(n) / w_raw.sum());
            sf = wq;
        }
    }

    // ---- base model (X1 + constant, no FEs) ------------------------------
    HdfeOptions hb;
    hb.se_type = StandardErrorType::Homoskedastic;
    hb.weights_are_frequencies = freq_weights;
    v11::HdfeRegressorV11 breg(hb);
    breg.fit(y, X1, {}, weights != nullptr ? &w_raw : nullptr);
    gprof_.mark("gel_base_reg");
    const Eigen::VectorXd bco = breg.results().coefficients;
    res.b_base = bco.head(p);

    // ---- aux regressions of each H_g on [x1, 1] ---------------------------
    Eigen::MatrixXd X1t(n, k1);
    X1t.leftCols(p) = X1;
    X1t.col(p).setOnes();
    const Eigen::MatrixXd X1tw = wq.asDiagonal() * X1t;
    const Eigen::MatrixXd P =
        (X1t.transpose() * X1tw).ldlt().solve(Eigen::MatrixXd::Identity(k1, k1));

    std::vector<Eigen::VectorXd> H(G), vres(G), dvec(G);
    std::vector<Eigen::MatrixXd> gam(G);   // k1 x q_g; empty for FE blocks
    std::vector<std::pair<int, int>> span(G, {-1, -1});
    {
        int cursor = 0;
        int gi = 0;
        for (int gsz : x2_group_sizes) {
            H[gi] = X2.middleCols(cursor, gsz) * b2.segment(cursor, gsz);
            gam[gi] = P * (X1tw.transpose() * X2.middleCols(cursor, gsz));
            span[gi] = {cursor, cursor + gsz};
            cursor += gsz;
            ++gi;
        }
        for (std::size_t d = 0; d < fes.size(); ++d, ++gi) {
            H[gi] = gel_fast_fit ? fe_effects_sep[d] : regu.results().fe_effects[d];
        }
    }
    for (int g = 0; g < G; ++g) {
        dvec[g] = P * (X1tw.transpose() * H[g]);
        vres[g] = H[g] - X1t * dvec[g];
    }
    bool any_fe_block = !fes.empty();
    if (any_fe_block) {
        res.notes += "absorbed FE blocks use the aux-regression (gamma0) variance. ";
    }

    // ---- covariance --------------------------------------------------------
    Eigen::MatrixXd fullcov = Eigen::MatrixXd::Zero(G * k1, G * k1);
    if (options.vce == GelbachVce::Unadjusted) {
        Eigen::MatrixXd R(n, G + 1);
        R.col(0) = e;
        for (int g = 0; g < G; ++g) R.col(g + 1) = vres[g];
        const Eigen::RowVectorXd wmean = (wq.transpose() * R) / wq.sum();
        R.rowwise() -= wmean;
        const Eigen::MatrixXd Omega =
            (R.transpose() * (wq.asDiagonal() * R)) / df_full;
        const double s2u_w =
            (wq.array() * e.array().square()).sum() / df_full;
        const double vscale =
            regu.results().sigma2 > 0.0 ? s2u_w / regu.results().sigma2 : 1.0;
        for (int g = 0; g < G; ++g) {
            for (int h = 0; h < G; ++h) {
                Eigen::MatrixXd block = Omega(g + 1, h + 1) * P;
                if (!options.gamma0 && gam[g].size() > 0 && gam[h].size() > 0) {
                    block += vscale * gam[g] *
                             V_hom.block(p + span[g].first, p + span[h].first,
                                         span[g].second - span[g].first,
                                         span[h].second - span[h].first) *
                             gam[h].transpose();
                }
                fullcov.block(g * k1, h * k1, k1, k1) = block;
            }
        }
    } else {
        // b1x2's stacked-system sandwich (multipliers validated vs b1x2).
        Eigen::MatrixXd W;
        int Kx;
        if (!fes.empty()) {
            // within representation: absorb the FEs out of [x1, x2]. All
            // Kx columns go through ONE multi-column absorption (the same
            // partial_out building block the xfe command uses), sharing the
            // FE index build and one joint iteration schedule, instead of
            // one full absorber fit per column. XHDFE_GELBACH_WITHIN_BATCH=0
            // restores the per-column loop (its per-column stopping rule can
            // differ from the joint one at the solver-tolerance level).
            Kx = p + q;
            W.resize(n, Kx);
            bool within_batch = true;
            if (const char* wb = std::getenv("XHDFE_GELBACH_WITHIN_BATCH")) {
                within_batch = !(wb[0] == '0' && wb[1] == '\0');
            }
            HdfeOptions hw;
            hw.se_type = StandardErrorType::Homoskedastic;
            hw.fit_intercept = false;
            hw.drop_singletons = false;
            hw.num_threads = options.num_threads;
            hw.weights_are_frequencies = freq_weights;
            if (within_batch && W_pre.rows() == n && W_pre.cols() == Kx) {
                // Demeaned [x1, x2] already produced by the shared MLSMR
                // absorption in the FE-recovery warm start above.
                W = std::move(W_pre);
                gprof_.mark("gel_within_batch");
            } else if (within_batch) {
                Eigen::MatrixXd X1X2(n, Kx);
                X1X2.leftCols(p) = X1;
                X1X2.rightCols(q) = X2;
                v11::HdfeRegressorV11 wreg(hw);
                const detail::AbsorptionResult ab = wreg.partial_out(
                    X1X2.col(0), X1X2.rightCols(Kx - 1), fes,
                    weights != nullptr ? &w_raw : nullptr);
                W.col(0) = ab.y_tilde;
                W.rightCols(Kx - 1) = ab.X_tilde;
        gprof_.mark("gel_within_batch");
            } else {
                Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(n, 1);
                for (int c = 0; c < Kx; ++c) {
                    const Eigen::VectorXd col = c < p ? X1.col(c) : X2.col(c - p);
                    v11::HdfeRegressorV11 wreg(hw);
                    wreg.fit(col, ones, fes, weights != nullptr ? &w_raw : nullptr);
                    W.col(c) = wreg.results().residuals;
                }
            }
        } else {
            Kx = p + q + 1;
            W.resize(n, Kx);
            W.leftCols(p) = X1;
            if (q > 0) W.middleCols(p, q) = X2;
            W.col(Kx - 1).setOnes();
        }
        const Eigen::MatrixXd Sw =
            (W.transpose() * (wq.asDiagonal() * W))
                .ldlt()
                .solve(Eigen::MatrixXd::Identity(Kx, Kx));
        double q_big, q_vu;
        const double nd = n_eff;
        if (ccodes_ptr == nullptr) {
            q_big = nd / (nd - 1.0);
            q_vu = nd / df_full;
        } else {
            const double gc = static_cast<double>(n_clusters);
            q_big = gc / (gc - 1.0);
            q_vu = (nd - 1.0) / df_full * gc / (gc - 1.0);
        }
        const Eigen::VectorXd se_score = sf.cwiseProduct(e);
        Eigen::MatrixXd Z(n, Kx + G * k1);
        Z.leftCols(Kx) = W.array().colwise() * se_score.array();
        for (int g = 0; g < G; ++g) {
            Z.middleCols(Kx + g * k1, k1) =
                X1t.array().colwise() * sf.cwiseProduct(vres[g]).array();
        }
        const Eigen::MatrixXd M = cluster_meat(Z, ccodes_ptr, n_clusters);
        gprof_.mark("gel_meat");
        Eigen::MatrixXd bread = Eigen::MatrixXd::Zero(Kx + G * k1, Kx + G * k1);
        bread.topLeftCorner(Kx, Kx) = Sw;
        for (int g = 0; g < G; ++g) {
            bread.block(Kx + g * k1, Kx + g * k1, k1, k1) = P;
        }
        const Eigen::MatrixXd C = q_big * (bread * M * bread);
        const Eigen::MatrixXd vu =
            q_vu * (Sw * M.topLeftCorner(Kx, Kx) * Sw);
        for (int g = 0; g < G; ++g) {
            for (int h = 0; h < G; ++h) {
                Eigen::MatrixXd block = C.block(Kx + g * k1, Kx + h * k1, k1, k1);
                if (!options.gamma0 && gam[g].size() > 0 && gam[h].size() > 0) {
                    const int s0 = span[g].first, sq = span[g].second - span[g].first;
                    const int t0 = span[h].first, tq = span[h].second - span[h].first;
                    block += gam[g] * vu.block(p + s0, p + t0, sq, tq) * gam[h].transpose();
                    if (!options.cov0) {
                        const Eigen::MatrixXd cr =
                            gam[g] * C.block(p + s0, Kx + h * k1, sq, k1);
                        const Eigen::MatrixXd crT =
                            (gam[h] * C.block(p + t0, Kx + g * k1, tq, k1)).transpose();
                        block += cr + crT;
                    }
                }
                fullcov.block(g * k1, h * k1, k1, k1) = block;
            }
        }
    }

    // ---- totals and identity ----------------------------------------------
    res.delta.resize(k1, G);
    for (int g = 0; g < G; ++g) res.delta.col(g) = dvec[g];
    res.cov = fullcov;
    res.total = Eigen::VectorXd::Zero(k1);
    for (int g = 0; g < G; ++g) res.total += dvec[g];
    res.total_cov = Eigen::MatrixXd::Zero(k1, k1);
    for (int g = 0; g < G; ++g) {
        for (int h = 0; h < G; ++h) {
            res.total_cov += fullcov.block(g * k1, h * k1, k1, k1);
        }
    }
    Eigen::VectorXd b_base_c(k1), b_full_c(k1);
    b_base_c.head(p) = res.b_base;
    b_base_c[p] = bco.size() == p + 1 ? bco[p] : 0.0;
    b_full_c.head(p) = res.b_full;
    b_full_c[p] = has_cons ? coefs[p + q] : 0.0;
    res.identity_gap = (b_base_c - b_full_c - res.total).cwiseAbs().maxCoeff();
    const bool gap_finite =
        res.identity_gap > -1e300 && res.identity_gap < 1e300;
    if (!gap_finite ||
        res.identity_gap >
            1e-6 * std::max(1.0, res.b_base.cwiseAbs().maxCoeff())) {
        res.converged = false;
        res.notes += "identity gap above tolerance. ";
    }
    return res;
}

}  // namespace gelbach
}  // namespace hdfe
