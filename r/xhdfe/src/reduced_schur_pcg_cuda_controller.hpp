#ifndef HDFE_REDUCED_SCHUR_PCG_CUDA_CONTROLLER_HPP
#define HDFE_REDUCED_SCHUR_PCG_CUDA_CONTROLLER_HPP
// Mechanical copy of the CPU controller. The products S*p (iterative) and S*x
// (the residual replacement) go to the adapter, which keeps the replacement on
// the CPU unless package 3-lite is in force; the formed residual and the
// stopping rules stay exact.
#include "reduced_schur_pcg.hpp"
namespace hdfe::reduced_cuda_controller {
namespace rp = hdfe::reduced_pcg;
using namespace rp;
using namespace rp::internal;
// Trace-only stamps of solve() for the CUDA tier's attribution line
// (event=attribution); solve() reads no clock for them when its caller passes
// none. A WorkStamps, declared after the workspace is allocated, stamps that
// moment and, destroyed first on every exit, the start of the teardown of the
// workspace and the geometry; the caller stamps its end when solve() returns.
struct SolveStamps {
    std::chrono::steady_clock::time_point prepare_from{}, prepare_to{}, work_allocated{}, teardown_from{};
};
struct WorkStamps {
    SolveStamps* stamps;
    explicit WorkStamps(SolveStamps* s) : stamps(s) {
        if (stamps != nullptr) stamps->work_allocated = std::chrono::steady_clock::now();
    }
    ~WorkStamps() {
        if (stamps != nullptr) stamps->teardown_from = std::chrono::steady_clock::now();
    }
    WorkStamps(const WorkStamps&) = delete;
    WorkStamps& operator=(const WorkStamps&) = delete;
};
// The page-locking of the operator's operand and result rows for a whole solve
// (package 2): p and q of run_group are adjacent slices of the arena (x, r, p, q,
// T), so ONE registration covers both; when the adapter also computes the
// replacement S*x on the device (package 3-lite), the one registration covers
// x, r, p and q. Declared after Work's allocations, it is destroyed before Work
// is freed, on every exit (normal, decline, exception), and the operator drains
// its stream before it releases the registration.
template <class IterativeOperator>
struct HostPin {
    IterativeOperator& op;
    HostPin(IterativeOperator& o, double* begin, std::size_t count) : op(o) { op.pin_host(begin, count); }
    ~HostPin() { op.unpin_host(); }
    HostPin(const HostPin&) = delete;
    HostPin& operator=(const HostPin&) = delete;
};
template <int W, class IterativeOperator>
inline Decline run_group(const Problem& p, const Config& cfg, const Geometry& g, const Team& team,
                         Work& work, int col0, int width, Report& rep, IterativeOperator& iterative_operator) {
    const int m = g.m, L = g.L, n = p.n;
    double* gv = work.g.data();
    double* x = work.arena.data();
    double* r = x + static_cast<std::size_t>(m) * W;
    double* pv = r + static_cast<std::size_t>(m) * W;
    double* q = pv + static_cast<std::size_t>(m) * W;
    double* T = q + static_cast<std::size_t>(m) * W;
    double* vbar = work.vbar.data();
    double* corr = work.corr.data();
    const double* minv = g.minv.data();
    const double* diag = g.diag.data();
    const double* col[W];
    double* res[W];
    double origin[W];
    for (int w = 0; w < W; ++w) {
        col[w] = w < width ? p.columns[col0 + w] : nullptr;
        res[w] = w < width ? p.residuals[col0 + w] : nullptr;
        origin[w] = w < width ? col[w][0] : 0.0;
    }

    // RHS pass, by segment: the level means, the deviations scattered into
    // private copies of g (in the arena, before x, r, p, q and T use it), and
    // the per-column sums of the stopping rule and of the certificate scale.
    auto t0 = std::chrono::steady_clock::now();
    const int rhs_parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1,
                                               std::max(1, (4 * m + L) / std::max(1, m))));
    const std::vector<int> rhs_bounds = balanced_bounds(g.row_ptr.data(), L, rhs_parts);
    const int rparts = static_cast<int>(rhs_bounds.size()) - 1;
    const std::size_t mw = static_cast<std::size_t>(m) * W;
    double* stats = work.stats.data();
    team.run(rparts, [&](int b) {
        double* acc = x + static_cast<std::size_t>(b) * mw;
        std::fill(acc, acc + mw, 0.0);
        double st[kStats * W];
        for (int s = 0; s < kStats * W; ++s) st[s] = 0.0;
        // The level mean as hi + delta, so the deviations sum to zero at the
        // precision of the deviations, not of the mean (rhs_segment).
        for (int l = rhs_bounds[static_cast<std::size_t>(b)]; l < rhs_bounds[static_cast<std::size_t>(b) + 1U]; ++l)
            rhs_segment<W>(p, g, l, col, width, origin, vbar + static_cast<std::size_t>(l) * W,
                           corr + static_cast<std::size_t>(l) * W, acc, st);
        std::memcpy(stats + static_cast<std::size_t>(b) * kStats * W, st, sizeof(st));
    });
    {
        const int units = std::max(1, (m + kVectorUnit - 1) / kVectorUnit);
        team.run(units, [&](int unit) {
            const std::size_t i0 = static_cast<std::size_t>(unit) * kVectorUnit * W;
            const std::size_t i1 = std::min(mw, static_cast<std::size_t>(unit + 1) * kVectorUnit * W);
            for (std::size_t i = i0; i < i1; ++i) {
                double sum = 0.0;
                for (int b = 0; b < rparts; ++b) sum += x[static_cast<std::size_t>(b) * mw + i];
                gv[i] = sum;
            }
        });
    }
    Lane lane[W];
    int k = 0;
    bool iterating = false;
    std::chrono::steady_clock::time_point t_iterate{};
    // The reports of this group's columns, on every exit: a declined attempt
    // still shows how far each column got (iterations, tau, chat).
    auto report_lanes = [&]() {
        for (int w = 0; w < width; ++w) {
            const Lane& ln = lane[w];
            ColumnReport& out = rep.columns[static_cast<std::size_t>(col0 + w)];
            out.stop = ln.active ? Stop::None : ln.stop;
            out.iterations = ln.active ? k : ln.iterations;
            out.first_tau_pass = ln.first_tau_pass;
            out.formed_checks = ln.formed_checks;
            const bool trivial = ln.stop == Stop::Zero || ln.stop == Stop::Span;
            out.tau = trivial ? 0.0 : ln.tau;
            out.chat = trivial ? 0.0 : ln.chat;
            const double s = trivial ? ln.wnorm : (ln.formed ? ln.s_formed : std::sqrt(std::max(0.0, ln.s2)));
            out.s_rel = ln.vnorm > 0.0 ? s / ln.vnorm : 0.0;
            rep.iterations = std::max(rep.iterations, out.iterations);
        }
    };
    auto finish = [&](Decline decline) {
        if (iterating) rep.iterate_seconds += seconds_since(t_iterate);
        report_lanes();
        return decline;
    };
    {
        long double tot[kStats * W];
        for (int s = 0; s < kStats * W; ++s) tot[s] = 0.0L;
        for (int b = 0; b < rparts; ++b)
            for (int s = 0; s < kStats * W; ++s)
                tot[s] += stats[static_cast<std::size_t>(b) * kStats * W + static_cast<std::size_t>(s)];
        // Denominator of the certificate (certify_absorption_result,
        // fe_absorption.cpp:24812-24858): sqrt(K n) times the column norm, the
        // centred one in the dominant-origin case of reghdfe-comparable.
        const long double op_norm = std::sqrt(static_cast<long double>(p.k) * static_cast<long double>(n));
        for (int w = 0; w < width; ++w) {
            Lane& ln = lane[w];
            ln.column = col0 + w;
            const long double raw = tot[0 * W + w];
            const long double sum = tot[2 * W + w];
            long double norm_sq = raw;
            if (cfg.dominant_origin && n > 0 && sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * raw) {
                const long double s1 = tot[3 * W + w];
                const long double centered = tot[4 * W + w] - s1 * (s1 / n);
                if (centered > 0.0L && centered < raw / 1024.0L) norm_sq = centered;
            }
            ln.v2 = static_cast<double>(raw);
            ln.vnorm = std::sqrt(ln.v2);
            ln.w2 = static_cast<double>(tot[1 * W + w]);
            ln.wnorm = std::sqrt(ln.w2);
            ln.denom = static_cast<double>(op_norm * std::sqrt(norm_sq));
            // A non-finite input or sum is never read as a zero or a span.
            if (!finite(ln.v2) || !finite(ln.w2) || !finite(static_cast<double>(sum)) ||
                !finite(static_cast<double>(tot[4 * W + w]))) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            if (!(raw > 0.0L)) {
                ln.stop = Stop::Zero;
            } else if (ln.wnorm <= cfg.btol * ln.vnorm) {
                ln.stop = Stop::Span;
            } else {
                ln.active = true;
            }
        }
    }
    rep.rhs_seconds += seconds_since(t0);

    // Lockstep PCG from x = 0: r = g, p = M+ r; the lanes never mix.
    t_iterate = std::chrono::steady_clock::now();
    iterating = true;
    bool live[W];
    for (int w = 0; w < W; ++w) live[w] = lane[w].active;
    std::fill(x, x + mw, 0.0);
    std::fill(q, q + mw, 0.0);
    double tot[8 * W];
    lane_sums<W, 2>(m, team, work, tot, [&](int i, double* acc) {
        const std::size_t base = static_cast<std::size_t>(i) * W;
        for (int w = 0; w < W; ++w) {
            const double rv = live[w] ? gv[base + w] : 0.0;
            r[base + w] = rv;
            pv[base + w] = minv[i] * rv;
            acc[0 * W + w] += minv[i] * rv * rv;
            acc[1 * W + w] += rv * rv;
        }
    });
    for (int w = 0; w < width; ++w) {
        Lane& ln = lane[w];
        if (!ln.active) continue;
        ln.rho = tot[0 * W + w];
        ln.s2 = ln.w2;
        if (!finite(ln.rho) || !finite(tot[1 * W + w])) {
            rep.decline_column = ln.column;
            return finish(Decline::NonFinite);
        }
        ln.chat = std::sqrt(tot[1 * W + w]) / ln.denom;
        if (!(ln.rho > 0.0)) {
            // tau = 0: no preconditioned residual left (whatever remains lies
            // on null levels, which no x can change). The certificate judges.
            ln.tau = 0.0;
            ln.first_tau_pass = 0;
            if (cfg.chat_stop > 0.0) {
                // D10: r = g is formed exactly here, so chat needs no other
                // confirmation; above the target the column stays active and
                // the first step declines (Breakdown: nothing to reduce).
                if (ln.chat <= cfg.chat_stop) {
                    ln.stop = Stop::Chat;
                    ln.active = false;
                }
            } else {
                ln.stop = Stop::Test2;
                ln.active = false;
            }
        }
    }

    // Stop measures of one lane. The compatible-system exit is LSQR's test1
    // in the coordinates of the preconditioned reduced operator: PCG with
    // M = diag(S) is CG on B^'B^, B^ = (I - P1) D2 M^(-1/2), in y = M^(1/2) x;
    // for min ||w - B^ y|| the test reads ||w - B^ y|| <= btol ||w|| +
    // atol ||B^|| ||y||, with ||B^|| estimated by A^ and ||y|| = sqrt(x'Mx)
    // (MLSMR applies the same rule to its own operator, fe_absorption.cpp:
    // 22319-22324). ||w|| <= ||v||, so it is never looser than the original norm.
    const int max_iter = std::max(1, cfg.max_iter);
    auto evaluate = [&](Lane& ln, double rho, double rr, double xmx) {
        const double s = ln.formed ? ln.s_formed : (ln.s2 > 0.0 ? std::sqrt(ln.s2) : 0.0);
        const double a = std::sqrt(std::max(0.0, ln.a2));
        if (!(rho > 0.0)) ln.tau = 0.0;
        else ln.tau = (s > 0.0 && a > 0.0) ? std::sqrt(rho) / (a * s) : kInf;
        ln.chat = std::sqrt(std::max(0.0, rr)) / ln.denom;
        const bool test2 = ln.tau <= cfg.atol;
        const bool test1 = (ln.formed || ln.s2 > 0.0) &&
                           s <= cfg.btol * ln.wnorm + cfg.atol * a * std::sqrt(std::max(0.0, xmx));
        if (test2 && ln.first_tau_pass < 0) ln.first_tau_pass = k;
        const bool companion_ok = !cfg.companion || ln.chat <= cfg.trigger;
        if (cfg.chat_stop > 0.0) {
            // D10: the target replaces test 2; test 1 stays.
            const bool test_chat = ln.chat <= cfg.chat_stop;
            ln.stop = companion_ok ? (test_chat ? Stop::Chat : (test1 ? Stop::Test1 : Stop::None)) : Stop::None;
        } else {
            ln.stop = companion_ok ? (test2 ? Stop::Test2 : (test1 ? Stop::Test1 : Stop::None)) : Stop::None;
        }
        return ln.stop != Stop::None;
    };
    auto any_active = [&]() {
        for (int w = 0; w < W; ++w)
            if (lane[w].active) return true;
        return false;
    };
    while (any_active()) {
        if (k >= max_iter) {
            for (int w = 0; w < W; ++w)
                if (lane[w].active) {
                    rep.decline_column = lane[w].column;
                    break;
                }
            return finish(Decline::Ceiling);
        }
        iterative_operator.template apply<W>(g, team, pv, q, T, work);
        lane_sums<W, 1>(m, team, work, tot, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) acc[w] += pv[base + w] * q[base + w];
        });
        double alpha[W];
        double rho_old[W];
        for (int w = 0; w < W; ++w) {
            alpha[w] = 0.0;
            rho_old[w] = lane[w].rho;
            Lane& ln = lane[w];
            if (!ln.active) continue;
            // Curvature is checked before the division.
            const double sigma = cfg.test_zero_curvature_at == k + 1 ? 0.0 : tot[w];
            if (!finite(sigma)) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            if (!(sigma > 0.0)) {
                rep.decline_column = ln.column;
                return finish(Decline::Breakdown);
            }
            alpha[w] = ln.rho / sigma;
        }
        lane_sums<W, 3>(m, team, work, tot, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) {
                const double xv = x[base + w] + alpha[w] * pv[base + w];
                const double rv = r[base + w] - alpha[w] * q[base + w];
                x[base + w] = xv;
                r[base + w] = rv;
                acc[0 * W + w] += minv[i] * rv * rv;
                acc[1 * W + w] += rv * rv;
                acc[2 * W + w] += diag[i] * xv * xv;
            }
        });
        ++k;
        bool candidate[W];
        bool enter_formed[W];
        bool any_candidate = false, any_enter = false;
        for (int w = 0; w < W; ++w) {
            candidate[w] = false;
            enter_formed[w] = false;
            Lane& ln = lane[w];
            if (!ln.active) continue;
            const double rho_new = tot[0 * W + w];
            // Trace of the CG Lanczos matrix T_k: 1/alpha_i + beta_(i-1)/alpha_(i-1).
            ln.a2 += 1.0 / alpha[w] + (k > 1 ? ln.beta / ln.alpha_prev : 0.0);
            if (!ln.formed) ln.s2 -= alpha[w] * rho_old[w];  // energy identity
            if (!finite(rho_new) || !finite(ln.s2) || !finite(ln.a2) || !finite(alpha[w])) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            ln.beta = rho_new / rho_old[w];
            ln.alpha_prev = alpha[w];
            ln.rho = rho_new;
            if (!ln.formed && !(ln.s2 > cfg.floor_factor * kEps * ln.w2)) {
                enter_formed[w] = true;
                any_enter = true;
            }
            double rr = tot[1 * W + w];
            if (cfg.test_recurrence_chat_at == k) {
                const double forced = cfg.test_recurrence_chat_value * ln.denom;
                rr = forced * forced;
            }
            candidate[w] = evaluate(ln, rho_new, rr, tot[2 * W + w]);
            any_candidate = any_candidate || candidate[w];
        }
        const bool periodic = k % std::max(1, cfg.refresh_period) == 0;
        const bool refresh = any_candidate || periodic || any_enter;
        if (refresh) {
            // Residual replacement: r := g - S x. The search direction is kept
            // (no restart), so the recurrence and the Lanczos trace continue.
            iterative_operator.template apply_replacement<W>(g, team, x, q, T, work);
            bool act[W];
            for (int w = 0; w < W; ++w) act[w] = lane[w].active;
            double fresh[5 * W];
            lane_sums<W, 5>(m, team, work, fresh, [&](int i, double* acc) {
                const std::size_t base = static_cast<std::size_t>(i) * W;
                for (int w = 0; w < W; ++w) {
                    if (!act[w]) continue;
                    const double rv = gv[base + w] - q[base + w];
                    r[base + w] = rv;
                    const double xv = x[base + w];
                    acc[0 * W + w] += minv[i] * rv * rv;
                    acc[1 * W + w] += rv * rv;
                    acc[2 * W + w] += diag[i] * xv * xv;
                    acc[3 * W + w] += xv * gv[base + w];
                    acc[4 * W + w] += xv * rv;
                }
            });
            bool formed_mask[W];
            bool any_formed = false;
            for (int w = 0; w < W; ++w) {
                formed_mask[w] = false;
                Lane& ln = lane[w];
                if (!ln.active) continue;
                const double rho_new = fresh[0 * W + w];
                if (!finite(rho_new)) {
                    rep.decline_column = ln.column;
                    return finish(Decline::NonFinite);
                }
                ln.beta = rho_new / rho_old[w];
                ln.rho = rho_new;
                if (!ln.formed) {
                    ln.s2 = ln.w2 - (fresh[3 * W + w] + fresh[4 * W + w]);
                    if (!(ln.s2 > cfg.floor_factor * kEps * ln.w2)) enter_formed[w] = true;
                }
                // Near the span of the effects both energy formulas cancel:
                // the exit is decided on the formed residual of the lane.
                if (ln.formed || enter_formed[w]) {
                    formed_mask[w] = true;
                    any_formed = true;
                }
            }
            if (any_formed) {
                double sq[W];
                formed_residual_sq<W>(p, g, team, work, x, vbar, corr, T, col, formed_mask, sq);
                for (int w = 0; w < W; ++w) {
                    if (!formed_mask[w]) continue;
                    Lane& ln = lane[w];
                    if (!finite(sq[w])) {
                        rep.decline_column = ln.column;
                        return finish(Decline::NonFinite);
                    }
                    if (!ln.formed && !(cfg.chat_stop > 0.0)) {
                        // A new measure of tau: the progress window starts
                        // again (with D10 the window reads chat, whose scale
                        // does not change here).
                        ln.best = kInf;
                        ln.have_prev = false;
                    }
                    ln.formed = true;
                    ln.s_formed = std::sqrt(std::max(0.0, sq[w]));
                    ++ln.formed_checks;
                }
            }
            for (int w = 0; w < W; ++w) {
                Lane& ln = lane[w];
                if (!ln.active) continue;
                candidate[w] = evaluate(ln, ln.rho, fresh[1 * W + w], fresh[2 * W + w]);
            }
        }
        for (int w = 0; w < W; ++w) {
            Lane& ln = lane[w];
            if (!ln.active) continue;
            if (cfg.on_iteration != nullptr)
                cfg.on_iteration(cfg.iteration_context, ln.column, k, ln.tau, ln.chat);
            if (candidate[w] && refresh) {
                ln.active = false;
                ln.iterations = k;
                continue;
            }
            ln.stop = Stop::None;
            // Progress: distance to the stop (1 = passes); stall and projection
            // are read at the periodic replacements only.
            const double measure = cfg.chat_stop > 0.0 ? ln.chat / cfg.chat_stop
                                   : cfg.companion       ? std::max(ln.tau / cfg.atol, ln.chat / cfg.trigger)
                                                         : ln.tau / cfg.atol;
            if (measure < ln.best) ln.best = measure;
            const int window = std::max(1, cfg.stall_window);
            if (periodic && k % window == 0) {
                if (ln.have_prev) {
                    if (!(ln.best <= 0.5 * ln.best_prev)) {
                        rep.decline_column = ln.column;
                        return finish(Decline::Stall);
                    }
                    const double rate = std::pow(ln.best / ln.best_prev, 1.0 / window);
                    if (ln.best > 1.0 && rate > 0.0 && rate < 1.0) {
                        const double projected = k + std::log(1.0 / ln.best) / std::log(rate);
                        if (projected > max_iter) {
                            rep.decline_column = ln.column;
                            return finish(Decline::Projected);
                        }
                    }
                }
                ln.best_prev = ln.best;
                ln.have_prev = true;
            }
        }
        bool still[W];
        for (int w = 0; w < W; ++w) still[w] = lane[w].active;
        lane_sums<W, 1>(m, team, work, tot, [&](int i, double* acc) {
            static_cast<void>(acc);
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w)
                pv[base + w] = still[w] ? minv[i] * r[base + w] + lane[w].beta * pv[base + w] : 0.0;
        });
        if (cfg.force_decline_iterate) return finish(Decline::Forced);
    }
    rep.iterate_seconds += seconds_since(t_iterate);
    iterating = false;

    // a1 = mean over each level of e of (v - D2 x) = hi + (delta - mean(D2 x)),
    // on every segment; hi stays in vbar, the small part replaces delta.
    t0 = std::chrono::steady_clock::now();
    const int all_parts = static_cast<int>(g.all_bounds.size()) - 1;
    team.run(all_parts, [&](int b) {
        for (int l = g.all_bounds[static_cast<std::size_t>(b)]; l < g.all_bounds[static_cast<std::size_t>(b) + 1U]; ++l) {
            if (rows_of(g, l) == 0) continue;
            double t[W];
            for (int w = 0; w < W; ++w) t[w] = 0.0;
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                double u[W];
                gather_cell<W>(g, x, c, u);
                const double mc = g.mult != nullptr ? static_cast<double>(g.mult[c]) : 1.0;
                for (int w = 0; w < W; ++w) t[w] += mc * u[w];
            }
            const double inv = g.seg_inv[static_cast<std::size_t>(l)];
            double* small = corr + static_cast<std::size_t>(l) * W;
            for (int w = 0; w < W; ++w) small[w] -= t[w] * inv;
        }
    });
    rep.recover_seconds += seconds_since(t0);

    // Residuals in row order: (v - hi) - ((delta - mean(D2 x)) + D2 x).
    t0 = std::chrono::steady_clock::now();
    const int parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1, n));
    team.run(parts, [&](int b) {
        const int i0 = static_cast<int>(part_begin(n, b, parts));
        const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
        residual_rows<W>(p, g, i0, i1, x, vbar, corr, col, res, width, nullptr, nullptr);
    });
    rep.residual_seconds += seconds_since(t0);

    report_lanes();
    return Decline::None;
}

template<class IterativeOperator>
inline Report solve(const Problem& p, const Config& cfg, IterativeOperator& iterative_operator,
                    SolveStamps* stamps = nullptr) {
    Report rep;
    rep.columns.assign(static_cast<std::size_t>(std::max(0, p.cols)), ColumnReport{});
    bool valid = p.n > 0 && p.k >= 2 && p.k <= 5 && p.cols > 0 && p.codes != nullptr &&
                 p.levels != nullptr && p.columns != nullptr && p.residuals != nullptr &&
                 p.eliminated >= 0 && p.eliminated < p.k;
    for (int b = 0; valid && b < p.k; ++b) valid = p.levels[b] >= 1 && p.codes[b] != nullptr;
    if (!valid) {
        rep.decline = Decline::Input;
        return rep;
    }
    const internal::Team team(cfg.threads, cfg.hooks, 8 * std::max(1, cfg.blocks));
    const int groups = (p.cols + 7) / 8;
    const int per_group = (p.cols + groups - 1) / groups;
    const int lanes = internal::lanes_for(per_group);
    rep.groups = groups;
    rep.lanes = lanes;
    try {
        auto t0 = std::chrono::steady_clock::now();
        internal::Geometry g;
        const Decline geometry = internal::build_geometry(p, cfg, team, lanes, g, &rep.bytes);
        rep.geometry_seconds = internal::seconds_since(t0);
        rep.sorted = g.sorted;
        rep.merged = g.merged;
        rep.segments = g.L;
        rep.operator_segments = g.L_op;
        rep.cells = g.C;
        for (int j = 0; j < g.nb && geometry == Decline::None; ++j) rep.large_effects += g.large[static_cast<std::size_t>(j)];
        if (geometry != Decline::None) {
            rep.decline = geometry;
            return rep;
        }
        if (cfg.force_decline_geometry) {
            rep.decline = Decline::Forced;
            return rep;
        }
        if (stamps != nullptr) stamps->prepare_from = std::chrono::steady_clock::now();
        iterative_operator.prepare(g, cfg, lanes, rep.bytes);
        if (stamps != nullptr) stamps->prepare_to = std::chrono::steady_clock::now();
        internal::Work work;
        const std::size_t W = static_cast<std::size_t>(lanes);
        const std::size_t m = static_cast<std::size_t>(g.m), L = static_cast<std::size_t>(g.L);
        const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;
        work.g.allocate(m * W);
        work.arena.allocate(W * (4U * m + L));
        work.vbar.allocate(L * W);
        work.corr.allocate(L * W);
        work.priv.allocate(static_cast<std::size_t>(op_parts) * static_cast<std::size_t>(g.small_levels) * W);
        work.partials.resize(static_cast<std::size_t>(g.nb));
        for (int j = 0; j < g.nb; ++j)
            if (g.large[static_cast<std::size_t>(j)])
                work.partials[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(g.tr[static_cast<std::size_t>(j)].items) * W);
        work.units.allocate((m / internal::kVectorUnit + 1U) * 8U * W);
        work.stats.allocate((g.row_bounds.size() + 1U) * internal::kStats * W);
        // p and q: run_group's third and fourth slices of the arena (x, r, p, q, T);
        // from x when the replacement product also runs on the device.
        const std::size_t pin_from = iterative_operator.replacement_on_device() ? 0U : 2U * m * W;
        const HostPin<IterativeOperator> pinned(iterative_operator, work.arena.data() + pin_from, 4U * m * W - pin_from);
        const WorkStamps work_stamps(stamps);
        for (int group = 0; group < groups; ++group) {
            const int col0 = group * per_group;
            const int width = std::min(per_group, p.cols - col0);
            iterative_operator.begin_group(width);  // the real width, before each group (the last may be narrower)
            Decline d = Decline::None;
            switch (lanes) {
                case 1: d = run_group<1>(p, cfg, g, team, work, col0, width, rep, iterative_operator); break;
                case 2: d = run_group<2>(p, cfg, g, team, work, col0, width, rep, iterative_operator); break;
                case 4: d = run_group<4>(p, cfg, g, team, work, col0, width, rep, iterative_operator); break;
                default: d = run_group<8>(p, cfg, g, team, work, col0, width, rep, iterative_operator); break;
            }
            if (d != Decline::None) {
                rep.decline = d;
                return rep;
            }
        }
        rep.solved = true;
    } catch (const std::bad_alloc&) {
        rep.solved = false;
        rep.decline = Decline::Memory;
    }
    return rep;
}

} // namespace hdfe::reduced_cuda_controller
#endif
