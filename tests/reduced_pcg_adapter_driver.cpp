// Driver of the R1 adapter (src/reduced_schur_pcg.cpp) on synthetic data,
// linked against its object with a stub certify_absorption_result that
// computes the certificate measure for unit weights (raw ids, the
// dominant-origin scale, the limit 8 min(tol, 1e-9)). Gate A only: it checks
// the adapter's plumbing (code compaction with both paths, acceptance, the
// final-line scope, trace levels, forced stages, the observation fold), not
// the production certificate. Usage: driver accept|decline|off|scope; the
// environment selects the switch, the trace and the forced stage.

#include "reduced_schur_pcg_entry.hpp"

#include "hdfe/parallel_work_observer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace hdfe {
namespace detail {

void certify_absorption_result(const Eigen::Ref<const Eigen::VectorXd>& y, const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<Eigen::VectorXi>& fes, const Eigen::VectorXd* weights,
                               const HdfeOptions& options, const std::vector<HeterogeneousSlopeTerm>& slopes,
                               AbsorptionResult& result, const GroupIndividualStructure* group_individual) {
    (void)weights;
    (void)slopes;
    (void)group_individual;
    const int n = static_cast<int>(y.size());
    const int cols = 1 + static_cast<int>(X.cols());
    double worst = 0.0;
    for (int c = 0; c < cols; ++c) {
        const double* raw = c == 0 ? y.data() : X.col(c - 1).data();
        const double* res = c == 0 ? result.y_tilde.data() : result.X_tilde.col(c - 1).data();
        long double num = 0.0L;
        for (const auto& fe : fes) {
            std::map<int, long double> sums;
            for (int i = 0; i < n; ++i) sums[fe[i]] += res[i];
            for (const auto& entry : sums) num += entry.second * entry.second;
        }
        long double sq = 0.0L, sum = 0.0L;
        for (int i = 0; i < n; ++i) {
            sq += static_cast<long double>(raw[i]) * raw[i];
            sum += raw[i];
        }
        long double norm_sq = sq;
        if (sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * sq) {
            long double shifted = 0.0L;
            for (int i = 0; i < n; ++i) shifted += static_cast<long double>(raw[i]) - raw[0];
            const long double mean = shifted / n;
            long double centered = 0.0L;
            for (int i = 0; i < n; ++i) {
                const long double d = static_cast<long double>(raw[i]) - raw[0] - mean;
                centered += d * d;
            }
            if (centered > 0.0L && centered < sq / 1024.0L) norm_sq = centered;
        }
        const long double scale = std::sqrt(static_cast<long double>(fes.size()) * n) * std::sqrt(norm_sq);
        worst = std::max(worst, scale > 0.0L ? static_cast<double>(std::sqrt(num) / scale) : 0.0);
    }
    result.abs_residual_rel = worst;
    const double tol = std::min(options.tol, 1.0e-9);
    result.precision_certified = result.converged && worst <= std::max(8.0 * tol, 64.0 * 2.220446049250313e-16);
}

// The attempt's entry since phase 2 of the Stata-path plan: the same stub, which
// leaves the tier's codes alone (the taking over is tested with the real
// certificate by tests/shared_fe_coding_contract.py).
void certify_tier_absorption(const Eigen::Ref<const Eigen::VectorXd>& y, const Eigen::Ref<const Eigen::MatrixXd>& X,
                             const std::vector<Eigen::VectorXi>& fes, const HdfeOptions& options,
                             AbsorptionResult& result, std::vector<TierFeCodes>& tier_codes) {
    (void)tier_codes;
    certify_absorption_result(y, X, fes, nullptr, options, {}, result, nullptr);
}

}  // namespace detail
}  // namespace hdfe

namespace {

struct Data {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
};

// Raw ids that are not compact: workers 3, 6, 9, ... (dense remap), firms
// spread over a range wider than 2n (sort fallback), years 1990..2009.
Data make_data() {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<int> worker, firm, year;
    std::vector<double> y, x1, x2;
    for (int u = 0; u < 3000; ++u) {
        const int rows = 2 + static_cast<int>(rng() % 6), home = static_cast<int>(rng() % 300);
        const int start = static_cast<int>(rng() % 12);
        for (int r = 0; r < rows; ++r) {
            const int f = (rng() % 100) < 20 ? static_cast<int>(rng() % 300) : home;
            worker.push_back(3 * u + 3);
            firm.push_back(1000003 * (f + 1));
            year.push_back(1990 + start + r);
            const double a = normal(rng);
            x1.push_back(a);
            x2.push_back(2.0e3 + normal(rng));
            y.push_back(0.5 * a + 0.01 * f + 0.02 * (start + r) + 0.3 * normal(rng));
        }
    }
    Data d;
    const int n = static_cast<int>(y.size());
    d.y = Eigen::Map<Eigen::VectorXd>(y.data(), n);
    d.X.resize(n, 2);
    for (int i = 0; i < n; ++i) {
        d.X(i, 0) = x1[static_cast<std::size_t>(i)];
        d.X(i, 1) = x2[static_cast<std::size_t>(i)];
    }
    for (const auto* codes : {&worker, &firm, &year}) {
        Eigen::VectorXi v(n);
        for (int i = 0; i < n; ++i) v[i] = (*codes)[static_cast<std::size_t>(i)];
        d.fes.push_back(v);
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "accept";
    const Data d = make_data();
    hdfe::HdfeOptions options;
    options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
    options.tol = 1e-8;
    options.max_iter = 100000;
    options.num_threads = 4;
    options.ordinary_krylov_parity_floor = true;
    hdfe::detail::ParallelWorkObserver observer;
    options.parallel_observer = &observer;
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            ++failures;
            std::printf("FAIL %s\n", what);
        }
    };
    if (mode == "off") {
        check(!hdfe::detail::reduced_pcg_enabled(), "switch read as on");
    } else if (mode == "scope") {
        hdfe::detail::ReducedPcgFitScope scope;  // never reaches the tier
    } else {
        std::optional<hdfe::detail::AbsorptionResult> result;
        hdfe::detail::ReducedPcgOutcome outcome;
        {
            hdfe::detail::ReducedPcgFitScope scope;
            {
                hdfe::detail::ReducedPcgFitScope nested;  // a nested fit shares the line
                result = hdfe::detail::absorb_reduced_pcg(d.y, d.X, d.fes, options, 1.0e-11, &outcome);
            }
        }
        // 2.32: a value if and only if the outcome says accepted.
        check(result.has_value() == (outcome.status == 1), "value and outcome disagree");
        const char* stage = std::getenv("XHDFE_TEST_REDUCED_PCG_DECLINE");
        if (stage != nullptr && std::strcmp(stage, "bad_alloc") == 0)
            check(outcome.status == 2 && std::strcmp(outcome.reason, "memory") == 0 && outcome.iterations == 0,
                  "injected bad_alloc not declined as memory");
        std::printf("outcome: status=%d reason=%s iterations=%d\n", outcome.status, outcome.reason,
                    outcome.iterations);
        if (mode == "accept") {
            check(result.has_value(), "not accepted");
            if (result) {
                check(outcome.status == 1 && std::strcmp(outcome.reason, "none") == 0 &&
                          outcome.iterations == result->iterations,
                      "accepted outcome");
                const std::vector<int> expected = {3000, 300, static_cast<int>(std::set<int>(d.fes[2].data(), d.fes[2].data() + d.fes[2].size()).size())};
                std::set<int> firms(d.fes[1].data(), d.fes[1].data() + d.fes[1].size());
                check(result->fe_levels[0] == expected[0], "worker levels");
                check(result->fe_levels[1] == static_cast<int>(firms.size()), "firm levels");
                check(result->fe_levels[2] == expected[2], "year levels");
                check(result->iterations > 0 && result->converged && result->precision_certified, "result flags");
                check(result->abs_residual_rel <= 1.0e-11, "rel above the trigger");
                check(result->y_tilde.size() == d.y.size() && result->X_tilde.cols() == d.X.cols(), "shapes");
                int team = 1, workers = 1;
                hdfe::detail::reduced_pcg_fold_observation(team, workers);
                check(team == 4 && workers >= 1, "observation fold");
                int again = 1, again_workers = 1;
                hdfe::detail::reduced_pcg_fold_observation(again, again_workers);
                check(again == 1, "fold consumed once");
                check(observer.max_team_size() == 1, "fit observer touched by the attempt");
                std::printf("accepted: iterations=%d rel=%.3e levels=%d,%d,%d team=%d\n", result->iterations,
                            result->abs_residual_rel, result->fe_levels[0], result->fe_levels[1],
                            result->fe_levels[2], team);
            }
        } else {
            check(!result.has_value(), "declined attempt returned a result");
            check(outcome.status == 2 && outcome.reason[0] != '\0', "declined outcome");
            int team = 1, workers = 1;
            hdfe::detail::reduced_pcg_fold_observation(team, workers);
            check(team == 1, "declined attempt folded into the observation");
            check(observer.max_team_size() == 1, "fit observer touched by the attempt");
        }
    }
    std::printf("%s\n", failures == 0 ? "driver PASS" : "driver FAIL");
    return failures == 0 ? 0 : 1;
}
