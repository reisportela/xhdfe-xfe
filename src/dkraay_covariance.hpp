#ifndef HDFE_DKRAAY_COVARIANCE_HPP
#define HDFE_DKRAAY_COVARIANCE_HPP

#include "ols.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe { namespace detail {

// Volatile additions preserve compensation under the release fast-math flags.
// long double additionally extends the product range on platforms providing it.
struct DkraaySum {
    long double value = 0.0L, correction = 0.0L;
    void add(long double x) {
        volatile long double adjusted = x - correction;
        volatile long double next = value + adjusted;
        volatile long double delta = next - value;
        correction = delta - adjusted;
        value = next;
    }
};

// The time IDs must preserve chronological order. As in reghdfe 6.14.1,
// lags refer to successive OBSERVED periods in the final estimation sample.
// Point estimates, residuals, absorption and FE recovery are never modified.
inline void apply_dkraay_covariance(
    HdfeResults& result, const OlsResult& ols,
    const Eigen::Ref<const Eigen::MatrixXd>& scores,
    const Eigen::Ref<const Eigen::MatrixXd>& original_X,
    const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>* clusters,
    const std::vector<int>& kept, bool recovered_intercept,
    const HdfeOptions& options, const Eigen::MatrixXd* score_low = nullptr) {
    if (options.se_type != StandardErrorType::Cluster || !clusters || clusters->size() != 1)
        throw std::runtime_error("Driscoll-Kraay requires exactly one ordered time vector");
    const int n = static_cast<int>(scores.rows());
    const int p = static_cast<int>(scores.cols());
    const bool add_constant = recovered_intercept && options.fit_intercept;
    const int q = p + (add_constant ? 1 : 0);
    const auto& time = clusters->front();
    if (time.size() != n || ols.residuals.size() != n ||
        (p && (ols.xtx_inv.rows() != p || ols.xtx_inv.cols() != p)) ||
        (score_low && (score_low->rows() != n || score_low->cols() != p)))
        throw std::runtime_error("Driscoll-Kraay score/bread dimensions do not agree");

    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return time[a] < time[b] || (time[a] == time[b] && a < b);
    });
    std::vector<int> starts;
    for (int i = 0; i < n; ++i)
        if (i == 0 || time[order[i]] != time[order[i-1]]) starts.push_back(i);
    const int T = static_cast<int>(starts.size());
    starts.push_back(n);
    if (T != result.num_clusters)
        throw std::runtime_error("Driscoll-Kraay time count disagrees with final sample");
    // Existing saturated/one-cluster handling marks inference unavailable.
    if (T < 2 || result.df_resid_unadj <= 0.0) return;
    const int requested = options.dkraay_bandwidth == 0
        ? static_cast<int>(std::floor(4 * std::pow(T / 100.0, 2.0 / 9.0))) + 1
        : options.dkraay_bandwidth;
    const int bandwidth = std::min(T, requested);
    if (bandwidth < 1) throw std::runtime_error("Driscoll-Kraay bandwidth must be positive");

    double scale = ols.cov_scale * result.cluster_scale;
    if (options.stats_style == StatsStyle::Reghdfe) {
        // reghdfe DK uses N-df_m-df_a, without cluster's extra nested-constant
        // subtraction. df_a already honours the requested nesting adjustment.
        const double denominator = result.nobs_effective - result.df_m - result.df_a;
        if (!(denominator > 0.0)) return;
        scale = options.ssc_k_adj ? (result.nobs_effective - 1.0) / denominator : 1.0;
        if (options.ssc_g_adj) scale *= static_cast<double>(T) / (T - 1.0);
        result.df_resid = options.ssc_t_df > 0.0 ? options.ssc_t_df
            : std::min(denominator, static_cast<double>(T - 1));
    }
    if (!ieee_finite(scale) || !(scale > 0.0))
        throw std::runtime_error("Driscoll-Kraay finite-sample scale is invalid");
    // Keep the same meaning as cluster_scale: multiplier on the core VCE scale.
    result.cluster_scale = scale / ols.cov_scale;
    if (q == 0) return;

    using Matrix = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix h = Matrix::Zero(T, q);
    const int team = std::max(1, options.num_threads);
    if (options.parallel_observer) options.parallel_observer->begin_region(team);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
    for (int t = 0; t < T; ++t) {
        if (options.parallel_observer) options.parallel_observer->observe_work();
        for (int j = 0; j < q; ++j) {
            DkraaySum sum;
            for (int pos = starts[t]; pos < starts[t+1]; ++pos) {
                const int i = order[pos];
                const long double x = j < p ? static_cast<long double>(scores(i,j)) +
                    (score_low ? static_cast<long double>((*score_low)(i,j)) : 0.0L) : 1.0L;
                const long double w = weights ? static_cast<long double>((*weights)[i]) : 1.0L;
                sum.add(w * static_cast<long double>(ols.residuals[i]) * x);
            }
            h(t,j) = sum.value;
        }
    }
    if (options.parallel_observer) options.parallel_observer->end_region();

    // Apply the bread before forming squares, avoiding large unscaled meats.
    Matrix influence = Matrix::Zero(T, q);
    std::vector<long double> means(static_cast<std::size_t>(p), 0.0L);
    DkraaySum sum_w;
    if (add_constant) {
        for (int i = 0; i < n; ++i) sum_w.add(weights ? (*weights)[i] : 1.0L);
        for (int j = 0; j < p; ++j) {
            DkraaySum sum;
            for (int i = 0; i < n; ++i)
                sum.add(static_cast<long double>(original_X(i,kept[j])) *
                        (weights ? static_cast<long double>((*weights)[i]) : 1.0L));
            means[j] = sum.value / sum_w.value;
        }
    }
    if (options.parallel_observer) options.parallel_observer->begin_region(team);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
    for (int t = 0; t < T; ++t) {
        if (options.parallel_observer) options.parallel_observer->observe_work();
        for (int j = 0; j < p; ++j) {
            DkraaySum sum;
            for (int k = 0; k < p; ++k)
                sum.add(static_cast<long double>(ols.xtx_inv(j,k)) * h(t,k));
            influence(t,j) = sum.value;
        }
        if (add_constant) {
            DkraaySum sum;
            sum.add(h(t,p) / sum_w.value);
            for (int j = 0; j < p; ++j) sum.add(-means[j] * influence(t,j));
            influence(t,p) = sum.value;
        }
    }
    if (options.parallel_observer) options.parallel_observer->end_region();

    // Bartlett identity: M = (1/B) sum_s v_s v_s', where v_s is the sum
    // over a B-period window, zero-padded at BOTH ends. O(T*q^2), independent
    // of B, and a Gram matrix by construction (no eigenvalue repair needed).
    const Eigen::Index windows = static_cast<Eigen::Index>(T) + bandwidth - 1;
    Matrix sums(windows, q);
    for (int j = 0; j < q; ++j) {
        DkraaySum rolling;
        for (Eigen::Index s = 0; s < windows; ++s) {
            if (s < T) rolling.add(influence(s,j));
            if (s >= bandwidth) rolling.add(-influence(s-bandwidth,j));
            sums(s,j) = rolling.value;
        }
    }
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(q, q);
    if (options.parallel_observer) options.parallel_observer->begin_region(team);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
    for (int j = 0; j < q; ++j) {
        if (options.parallel_observer) options.parallel_observer->observe_work();
        for (int k = 0; k <= j; ++k) {
            DkraaySum sum;
            for (Eigen::Index s = 0; s < windows; ++s) sum.add(sums(s,j) * sums(s,k));
            const double value = static_cast<double>(sum.value * (static_cast<long double>(scale) / bandwidth));
            covariance(j,k) = covariance(k,j) = value;
        }
    }
    if (options.parallel_observer) options.parallel_observer->end_region();
    if (!ieee_all_finite(covariance))
        throw std::runtime_error("Driscoll-Kraay covariance exceeds the output arithmetic range");

    std::vector<int> output = kept;
    if (options.fit_intercept) output.push_back(static_cast<int>(result.coefficients.size()) - 1);
    if (static_cast<int>(output.size()) != q)
        throw std::runtime_error("Driscoll-Kraay coefficient mapping is inconsistent");
    result.covariance.setZero();
    result.std_errors.setConstant(std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < q; ++j) {
        for (int k = 0; k < q; ++k) result.covariance(output[j],output[k]) = covariance(j,k);
        result.std_errors[output[j]] = std::sqrt(std::max(0.0, covariance(j,j)));
    }
}

}} // namespace hdfe::detail
#endif
