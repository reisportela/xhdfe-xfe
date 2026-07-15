#include "hdfe/hdfe_regressor_v11.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "fe_absorption.hpp"
#include "fe_absorption_cuda.hpp"
#include "iv.hpp"
#include "ols.hpp"

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe {
namespace v11 {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440084436210485;
constexpr int kMaxSingletonIterations = 100;

struct KahanSum {
    long double sum = 0.0L;
    long double c = 0.0L;
    void add(long double value) {
        const long double y = value - c;
        const long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

bool savefe_profile_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char* raw = std::getenv("XHDFE_PROFILE_SAVEFE");
    if (!raw || *raw == '\0' || *raw == '0') {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

void savefe_profile_log(const std::string& msg) {
    if (!savefe_profile_enabled()) {
        return;
    }
    std::cerr << "savefe_profile " << msg << '\n';
}

bool cpu_profile_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char* raw = std::getenv("XHDFE_PROFILE_CPU");
    if (!raw || *raw == '\0' || *raw == '0') {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

void cpu_profile_log_elapsed(
    const char* label,
    const std::chrono::steady_clock::time_point& start) {
    if (!cpu_profile_enabled()) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cerr << "cpu_profile label=" << label << " ms=" << elapsed.count() << '\n';
}

class SavefeTimer {
public:
    explicit SavefeTimer(const char* label)
        : label_(label), enabled_(savefe_profile_enabled()) {
        if (enabled_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~SavefeTimer() {
        if (!enabled_) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start_;
        std::cerr << "savefe_profile label=" << label_
                  << " ms=" << elapsed.count() << '\n';
    }

private:
    const char* label_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};


double weighted_sum_of_squares(const Eigen::Ref<const Eigen::VectorXd>& values,
                               const Eigen::VectorXd* weights) {
    const int n = static_cast<int>(values.size());
    if (n == 0) {
        return 0.0;
    }
    const double* v = values.data();
    KahanSum sum;
    if (!weights) {
        for (int i = 0; i < n; ++i) {
            const long double vi = static_cast<long double>(v[i]);
            sum.add(vi * vi);
        }
        return static_cast<double>(sum.sum);
    }
    const double* w = weights->data();
    for (int i = 0; i < n; ++i) {
        const long double vi = static_cast<long double>(v[i]);
        sum.add(vi * vi * static_cast<long double>(w[i]));
    }
    return static_cast<double>(sum.sum);
}

double weighted_mean(const Eigen::Ref<const Eigen::VectorXd>& values,
                     const Eigen::VectorXd* weights,
                     double* sum_weights_out = nullptr) {
    const int n = static_cast<int>(values.size());
    if (n == 0) {
        if (sum_weights_out) {
            *sum_weights_out = 0.0;
        }
        return 0.0;
    }
    const double* v = values.data();
    if (!weights) {
        if (sum_weights_out) {
            *sum_weights_out = static_cast<double>(n);
        }
        KahanSum sum;
        for (int i = 0; i < n; ++i) {
            sum.add(static_cast<long double>(v[i]));
        }
        return static_cast<double>(sum.sum / static_cast<long double>(n));
    }
    const double* w = weights->data();
    KahanSum sum_w;
    KahanSum sum_yw;
    for (int i = 0; i < n; ++i) {
        const long double wi = static_cast<long double>(w[i]);
        sum_w.add(wi);
        sum_yw.add(static_cast<long double>(v[i]) * wi);
    }
    const long double denom = sum_w.sum;
    if (!(denom > 0.0L)) {
        throw std::runtime_error("Weights must sum to a positive value");
    }
    if (sum_weights_out) {
        *sum_weights_out = static_cast<double>(denom);
    }
    return static_cast<double>(sum_yw.sum / denom);
}

enum class FeNormalizeStyle { Reghdfe, Component };

FeNormalizeStyle resolve_fe_normalize_style(StatsStyle stats_style) {
    (void)stats_style;
    const char* raw = std::getenv("XHDFE_FE_NORMALIZE");
    if (!raw || !*raw) {
        return FeNormalizeStyle::Component;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "reghdfe" || value == "mean" || value == "mean_only") {
        return FeNormalizeStyle::Reghdfe;
    }
    if (value == "component" || value == "components") {
        return FeNormalizeStyle::Component;
    }
    return FeNormalizeStyle::Component;
}

struct FeDiagStats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double absmax = 0.0;
};

FeDiagStats summarize_vector(const Eigen::Ref<const Eigen::VectorXd>& values,
                             const Eigen::VectorXd* weights) {
    FeDiagStats stats;
    const int n = static_cast<int>(values.size());
    if (n == 0) {
        return stats;
    }
    const double* v = values.data();
    double min_val = v[0];
    double max_val = v[0];
    double abs_max = std::abs(v[0]);
    for (int i = 1; i < n; ++i) {
        const double val = v[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        abs_max = std::max(abs_max, std::abs(val));
    }
    stats.mean = weighted_mean(values, weights);
    stats.min = min_val;
    stats.max = max_val;
    stats.absmax = abs_max;
    return stats;
}

double max_abs_diff(const Eigen::Ref<const Eigen::VectorXd>& a,
                    const Eigen::Ref<const Eigen::VectorXd>& b) {
    const int n = static_cast<int>(a.size());
    if (n == 0 || b.size() != n) {
        return 0.0;
    }
    const double* ap = a.data();
    const double* bp = b.data();
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(ap[i] - bp[i]));
    }
    return max_diff;
}

void maybe_write_fe_diagnostics(const Eigen::Ref<const Eigen::VectorXd>& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const Eigen::VectorXd& coefficients,
                                const std::vector<Eigen::VectorXd>& fe_effects,
                                bool fit_intercept,
                                const Eigen::VectorXd* weights,
                                const Eigen::Ref<const Eigen::VectorXd>& residuals,
                                int abs_iter,
                                int fe_recovery_iter,
                                bool fe_recovery_converged,
                                double fe_recovery_max_delta,
                                const char* fe_recovery_method) {
    const char* path = std::getenv("XHDFE_FE_DIAG_FILE");
    if (!path || *path == '\0') {
        return;
    }
    if (fe_effects.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    out << std::setprecision(17);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto stamp =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

    const int n = static_cast<int>(y.size());
    const int total_coefs = static_cast<int>(coefficients.size());
    const int slope_cols = fit_intercept ? std::max(0, total_coefs - 1) : total_coefs;
    Eigen::VectorXd fitted = Eigen::VectorXd::Zero(n);
    if (slope_cols > 0) {
        fitted.noalias() = X * coefficients.head(slope_cols);
    }
    if (fit_intercept && total_coefs > slope_cols) {
        fitted.array() += coefficients(slope_cols);
    }
    Eigen::VectorXd partial = y - fitted;

    Eigen::VectorXd fe_total = Eigen::VectorXd::Zero(n);
    for (const auto& fe_vec : fe_effects) {
        if (fe_vec.size() == n) {
            fe_total.noalias() += fe_vec;
        }
    }

    Eigen::VectorXd residual_calc = partial - fe_total;
    const double resid_diff = max_abs_diff(residual_calc, residuals);

    const FeDiagStats partial_stats = summarize_vector(partial, weights);
    const FeDiagStats fe_total_stats = summarize_vector(fe_total, weights);
    const FeDiagStats resid_stats = summarize_vector(residual_calc, weights);

    out << "run_id_ms=" << stamp
        << " nobs=" << n
        << " nfe=" << fe_effects.size()
        << " fit_intercept=" << (fit_intercept ? 1 : 0)
        << " weighted=" << (weights ? 1 : 0)
        << " abs_iter=" << abs_iter
        << " fe_recovery_iter=" << fe_recovery_iter
        << " fe_recovery_converged=" << (fe_recovery_converged ? 1 : 0)
        << " fe_recovery_max_delta=" << fe_recovery_max_delta
        << " fe_recovery_method=" << (fe_recovery_method ? fe_recovery_method : "unknown")
        << '\n';
    out << "partial mean=" << partial_stats.mean
        << " min=" << partial_stats.min
        << " max=" << partial_stats.max
        << " absmax=" << partial_stats.absmax
        << '\n';
    out << "fe_total mean=" << fe_total_stats.mean
        << " min=" << fe_total_stats.min
        << " max=" << fe_total_stats.max
        << " absmax=" << fe_total_stats.absmax
        << '\n';
    out << "residual mean=" << resid_stats.mean
        << " min=" << resid_stats.min
        << " max=" << resid_stats.max
        << " absmax=" << resid_stats.absmax
        << '\n';
    out << "residual_diff_absmax=" << resid_diff << '\n';
    for (std::size_t d = 0; d < fe_effects.size(); ++d) {
        const FeDiagStats fe_stats = summarize_vector(fe_effects[d], weights);
        out << "fe_dim=" << d
            << " mean=" << fe_stats.mean
            << " min=" << fe_stats.min
            << " max=" << fe_stats.max
            << " absmax=" << fe_stats.absmax
            << '\n';
    }
    out << "----" << '\n';
}

double compute_total_sum_of_squares(const Eigen::Ref<const Eigen::VectorXd>& y,
                                    const Eigen::VectorXd* weights,
                                    double* sum_weights_out = nullptr) {
    const int n = static_cast<int>(y.size());
    if (n == 0) {
        if (sum_weights_out) {
            *sum_weights_out = 0.0;
        }
        return 0.0;
    }
    const double mean = weighted_mean(y, weights, sum_weights_out);
    const double* y_ptr = y.data();
    KahanSum sum;
    if (!weights) {
        for (int i = 0; i < n; ++i) {
            const long double diff = static_cast<long double>(y_ptr[i] - mean);
            sum.add(diff * diff);
        }
        return static_cast<double>(sum.sum);
    }
    const double* w = weights->data();
    for (int i = 0; i < n; ++i) {
        const long double diff = static_cast<long double>(y_ptr[i] - mean);
        sum.add(diff * diff * static_cast<long double>(w[i]));
    }
    return static_cast<double>(sum.sum);
}

void normalize_nonfrequency_weighted_fit_stats(HdfeResults& results,
                                               bool weights_are_frequencies,
                                               const Eigen::VectorXd* weights,
                                               double sum_weights_hint) {
    if (!weights || weights_are_frequencies) {
        return;
    }
    const double nobs = static_cast<double>(results.nobs);
    if (!(nobs > 0.0)) {
        return;
    }
    double sum_w = sum_weights_hint;
    if (!(sum_w > 0.0) || !std::isfinite(sum_w)) {
        sum_w = weights->sum();
    }
    if (!(sum_w > 0.0) || !std::isfinite(sum_w)) {
        return;
    }
    if (std::abs(sum_w - nobs) <= 1e-12 * std::max(1.0, nobs)) {
        return;
    }
    const double scale = nobs / sum_w;
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return;
    }
    results.rss *= scale;
    results.tss *= scale;
    results.tss_within *= scale;
    results.sigma2 *= scale;
}

Eigen::MatrixXd maybe_add_intercept(const Eigen::Ref<const Eigen::MatrixXd>& X, bool fit_intercept) {
    if (!fit_intercept) {
        return X;
    }
    Eigen::MatrixXd design(X.rows(), X.cols() + 1);
    if (X.size() > 0) {
        design.block(0, 0, X.rows(), X.cols()) = X;
    }
    design.col(X.cols()).setOnes();
    return design;
}

Eigen::MatrixXd append_matrix(const Eigen::Ref<const Eigen::MatrixXd>& left,
                              const Eigen::MatrixXd* right) {
    if (!right || right->cols() == 0) {
        return left;
    }
    if (right->rows() != left.rows()) {
        throw std::runtime_error("Instrument matrix must have the same number of rows as X");
    }
    Eigen::MatrixXd combined(left.rows(), left.cols() + right->cols());
    if (left.cols() > 0) {
        combined.block(0, 0, left.rows(), left.cols()) = left;
    }
    combined.block(0, left.cols(), right->rows(), right->cols()) = *right;
    return combined;
}

bool fix_psd(Eigen::MatrixXd& V) {
    if (V.rows() == 0 || V.cols() == 0) {
        return false;
    }
    V = 0.5 * (V + V.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
    if (es.info() != Eigen::Success) {
        return false;
    }
    Eigen::VectorXd lambda = es.eigenvalues();
    const double min_lambda = lambda.minCoeff();
    if (min_lambda >= 0.0) {
        return false;
    }
    lambda = lambda.cwiseMax(0.0);
    V = es.eigenvectors() * lambda.asDiagonal() * es.eigenvectors().transpose();
    return true;
}

struct ClusterInterceptMeat {
    Eigen::VectorXd cross;
    double alpha = 0.0;
    double u_t = 0.0;
    int num_clusters = 0;
};

ClusterInterceptMeat compute_cluster_intercept_meat(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& residuals,
    const Eigen::VectorXd* weights,
    const Eigen::Ref<const Eigen::VectorXd>& c,
    const Eigen::VectorXi& clusters) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    if (clusters.size() != n || residuals.size() != n || c.size() != n) {
        throw std::runtime_error("Intercept meat inputs must have matching lengths");
    }

    ClusterInterceptMeat out;
    out.cross = Eigen::VectorXd::Zero(p);
    out.alpha = 0.0;
    out.u_t = 0.0;
    out.num_clusters = 0;
    if (n == 0) {
        return out;
    }

    int min_id = clusters(0);
    int max_id = clusters(0);
    for (int i = 1; i < n; ++i) {
        const int v = clusters(i);
        min_id = std::min(min_id, v);
        max_id = std::max(max_id, v);
    }
    const long long range_ll =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    const bool dense_ok =
        min_id >= 0 && range_ll > 0 && range_ll <= kDenseRangeCap &&
        range_ll <= static_cast<long long>(n) * 2LL;
    const int range = dense_ok ? static_cast<int>(range_ll) : 0;

    std::vector<double> aggregated_s;
    std::vector<double> aggregated_t;
    std::vector<double> aggregated_u;
    int next_cluster = 0;
    if (dense_ok) {
        aggregated_s.assign(static_cast<std::size_t>(range) * static_cast<std::size_t>(p), 0.0);
        aggregated_t.assign(static_cast<std::size_t>(range), 0.0);
        aggregated_u.assign(static_cast<std::size_t>(range), 0.0);
        std::vector<uint8_t> seen(static_cast<std::size_t>(range), 0);
        for (int i = 0; i < n; ++i) {
            const int idx = clusters(i) - min_id;
            if (!seen[static_cast<std::size_t>(idx)]) {
                seen[static_cast<std::size_t>(idx)] = 1;
                ++next_cluster;
            }
            const double wi = weights ? (*weights)(i) : 1.0;
            const double ui = residuals(i);
            const double score_u = wi * ui;
            if (p > 0) {
                double* score = aggregated_s.data() +
                                static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                for (int j = 0; j < p; ++j) {
                    score[static_cast<std::size_t>(j)] += X(i, j) * score_u;
                }
            }
            aggregated_t[static_cast<std::size_t>(idx)] += c(i) * ui;
            aggregated_u[static_cast<std::size_t>(idx)] += score_u;
        }
        for (int g = 0; g < range; ++g) {
            if (!seen[static_cast<std::size_t>(g)]) {
                continue;
            }
            const double ug = aggregated_u[static_cast<std::size_t>(g)];
            const double tg = aggregated_t[static_cast<std::size_t>(g)];
            out.alpha += tg * tg;
            out.u_t += ug * tg;
            if (p > 0) {
                const Eigen::Map<const Eigen::VectorXd> score(
                    aggregated_s.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
                out.cross.noalias() += score * tg;
            }
        }
        out.num_clusters = next_cluster;
        return out;
    }

    std::unordered_map<int, int> cluster_map;
    cluster_map.reserve(static_cast<std::size_t>(n));
    aggregated_s.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(p) /
                         static_cast<std::size_t>(8));
    aggregated_t.reserve(static_cast<std::size_t>(n) / static_cast<std::size_t>(8));
    aggregated_u.reserve(static_cast<std::size_t>(n) / static_cast<std::size_t>(8));
    for (int i = 0; i < n; ++i) {
        const int raw_id = clusters(i);
        auto it = cluster_map.find(raw_id);
        int cluster_idx;
        if (it == cluster_map.end()) {
            cluster_idx = next_cluster;
            ++next_cluster;
            cluster_map.emplace(raw_id, cluster_idx);
            aggregated_s.resize(static_cast<std::size_t>(next_cluster) * static_cast<std::size_t>(p),
                                0.0);
            aggregated_t.resize(static_cast<std::size_t>(next_cluster), 0.0);
            aggregated_u.resize(static_cast<std::size_t>(next_cluster), 0.0);
        } else {
            cluster_idx = it->second;
        }
        const double wi = weights ? (*weights)(i) : 1.0;
        const double ui = residuals(i);
        const double score_u = wi * ui;
        if (p > 0) {
            double* score = aggregated_s.data() +
                            static_cast<std::size_t>(cluster_idx) * static_cast<std::size_t>(p);
            for (int j = 0; j < p; ++j) {
                score[static_cast<std::size_t>(j)] += X(i, j) * score_u;
            }
        }
        aggregated_t[static_cast<std::size_t>(cluster_idx)] += c(i) * ui;
        aggregated_u[static_cast<std::size_t>(cluster_idx)] += score_u;
    }
    for (int g = 0; g < next_cluster; ++g) {
        const double ug = aggregated_u[static_cast<std::size_t>(g)];
        const double tg = aggregated_t[static_cast<std::size_t>(g)];
        out.alpha += tg * tg;
        out.u_t += ug * tg;
        if (p > 0) {
            const Eigen::Map<const Eigen::VectorXd> score(
                aggregated_s.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
            out.cross.noalias() += score * tg;
        }
    }
    out.num_clusters = next_cluster;
    return out;
}

Eigen::VectorXi combine_clusters(const std::vector<Eigen::VectorXi>& clusters,
                                 const std::vector<int>& dims);

ClusterInterceptMeat compute_multiway_intercept_meat(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& residuals,
    const Eigen::VectorXd* weights,
    const Eigen::Ref<const Eigen::VectorXd>& c,
    const std::vector<Eigen::VectorXi>& clusters,
    ClusterDofMethod g_df,
    bool g_adj,
    double df_resid) {
    const int m = static_cast<int>(clusters.size());
    if (m <= 0) {
        throw std::runtime_error("At least one cluster dimension is required");
    }
    if (m > 20) {
        throw std::runtime_error("Multi-way clustering supports up to 20 cluster dimensions");
    }
    const int n = static_cast<int>(residuals.size());
    for (const auto& cvec : clusters) {
        if (cvec.size() != n) {
            throw std::runtime_error("Cluster vector length must equal the number of observations");
        }
    }

    int min_clusters = std::numeric_limits<int>::max();
    for (const auto& cvec : clusters) {
        std::unordered_map<int, int> uniq;
        uniq.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            uniq.emplace(cvec(i), 1);
        }
        min_clusters = std::min(min_clusters, static_cast<int>(uniq.size()));
    }

    ClusterInterceptMeat total;
    total.cross = Eigen::VectorXd::Zero(X.cols());
    total.alpha = 0.0;
    total.u_t = 0.0;
    total.num_clusters = (min_clusters == std::numeric_limits<int>::max()) ? 0 : min_clusters;

    const std::uint64_t max_mask = (static_cast<std::uint64_t>(1) << m);
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(m));
    for (std::uint64_t mask = 1; mask < max_mask; ++mask) {
        dims.clear();
        for (int j = 0; j < m; ++j) {
            if (mask & (static_cast<std::uint64_t>(1) << j)) {
                dims.push_back(j);
            }
        }
        Eigen::VectorXi combined = combine_clusters(clusters, dims);
        ClusterInterceptMeat meat =
            compute_cluster_intercept_meat(X, residuals, weights, c, combined);
        double scale = 1.0;
        if (g_df == ClusterDofMethod::Conventional && g_adj && meat.num_clusters > 1) {
            scale = static_cast<double>(meat.num_clusters) /
                    static_cast<double>(meat.num_clusters - 1);
        }
        const double sign = (dims.size() % 2 == 1) ? 1.0 : -1.0;
        total.cross.noalias() += sign * scale * meat.cross;
        total.alpha += sign * scale * meat.alpha;
        total.u_t += sign * scale * meat.u_t;
    }

    double scale = 1.0;
    if (df_resid > 0.0) {
        scale *= (static_cast<double>(n) - 1.0) / df_resid;
    }
    if (g_df == ClusterDofMethod::Min && g_adj && min_clusters > 1) {
        scale *= static_cast<double>(min_clusters) /
                 static_cast<double>(min_clusters - 1);
    }
    total.cross *= scale;
    total.alpha *= scale;
    total.u_t *= scale;
    return total;
}

bool update_intercept_covariance(
    HdfeResults& results,
    const Eigen::Ref<const Eigen::MatrixXd>& X_used,
    const Eigen::Ref<const Eigen::MatrixXd>& X_tilde_used,
    const Eigen::Ref<const Eigen::VectorXd>& residuals,
    const Eigen::VectorXd* weights,
    const std::vector<int>& kept_slope_cols,
    const detail::OlsResult& ols_result,
    StandardErrorType se_type,
    const std::vector<Eigen::VectorXi>* clusters,
    ClusterDofMethod g_df,
    bool g_adj,
    double sigma2,
    double robust_scale,
    double cluster_ratio,
    bool weights_are_frequencies) {
    const int intercept_idx = static_cast<int>(results.coefficients.size()) - 1;
    if (intercept_idx < 0) {
        return false;
    }
    const int slope_cols_full = intercept_idx;
    if (slope_cols_full < 0 ||
        results.std_errors.size() != results.coefficients.size() ||
        results.covariance.rows() != results.covariance.cols() ||
        results.covariance.rows() != results.coefficients.size()) {
        return false;
    }

    const int n = static_cast<int>(X_tilde_used.rows());
    if (residuals.size() != n) {
        return false;
    }
    const int k_active = static_cast<int>(kept_slope_cols.size());
    if (k_active != X_tilde_used.cols()) {
        return false;
    }

    double sum_w = 0.0;
    if (weights) {
        sum_w = weights->sum();
    } else {
        sum_w = static_cast<double>(n);
    }
    if (!(sum_w > 0.0)) {
        return false;
    }

    Eigen::VectorXd xbar = Eigen::VectorXd::Zero(k_active);
    if (k_active > 0) {
        if (!weights) {
            for (int pos = 0; pos < k_active; ++pos) {
                const int col = kept_slope_cols[static_cast<std::size_t>(pos)];
                xbar(pos) = X_used.col(col).mean();
            }
        } else {
            for (int pos = 0; pos < k_active; ++pos) {
                const int col = kept_slope_cols[static_cast<std::size_t>(pos)];
                xbar(pos) = X_used.col(col).dot(*weights) / sum_w;
            }
        }
    }

    Eigen::VectorXd cov_beta_alpha = Eigen::VectorXd::Zero(k_active);
    Eigen::VectorXd B = Eigen::VectorXd::Zero(k_active);
    long double var_alpha = 0.0L;
    if (se_type == StandardErrorType::Homoskedastic) {
        if (k_active > 0) {
            if (ols_result.xtx_inv.rows() != k_active || ols_result.xtx_inv.cols() != k_active) {
                return false;
            }
            const Eigen::MatrixXd cov_beta = sigma2 * ols_result.xtx_inv;
            var_alpha = static_cast<long double>(sigma2 / sum_w);
            const double quad = xbar.dot(cov_beta * xbar);
            var_alpha += static_cast<long double>(quad);
            cov_beta_alpha = -cov_beta * xbar;
        } else {
            var_alpha = static_cast<long double>(sigma2 / sum_w);
        }
    } else {
        bool used_cached_cluster_intercept = false;
        if (se_type == StandardErrorType::Cluster && clusters && clusters->size() == 1 &&
            !weights && ols_result.cluster_ux.size() == k_active &&
            std::isfinite(ols_result.cluster_u2) &&
            ols_result.cov_scale > 0.0 && std::isfinite(ols_result.cov_scale) &&
            ols_result.covariance.rows() == k_active &&
            ols_result.covariance.cols() == k_active) {
            const double inv_n = 1.0 / static_cast<double>(n);
            const double scale = ols_result.cov_scale * cluster_ratio;
            const Eigen::MatrixXd cov_unscaled =
                ols_result.covariance / ols_result.cov_scale;
            if (k_active > 0) {
                if (ols_result.xtx_inv.rows() != k_active ||
                    ols_result.xtx_inv.cols() != k_active) {
                    return false;
                }
                B = ols_result.xtx_inv * xbar;
                const Eigen::VectorXd cov_xbar = cov_unscaled * xbar;
                cov_beta_alpha =
                    inv_n * (ols_result.xtx_inv * ols_result.cluster_ux) - cov_xbar;
                const double alpha_base =
                    inv_n * inv_n * ols_result.cluster_u2 -
                    2.0 * inv_n * B.dot(ols_result.cluster_ux) +
                    xbar.dot(cov_xbar);
                var_alpha = static_cast<long double>(alpha_base);
            } else {
                var_alpha =
                    static_cast<long double>(inv_n * inv_n * ols_result.cluster_u2);
            }
            cov_beta_alpha *= scale;
            var_alpha *= static_cast<long double>(scale);
            used_cached_cluster_intercept = true;
        }

        if (!used_cached_cluster_intercept) {
        Eigen::VectorXd c_vec = Eigen::VectorXd::Zero(n);
        if (k_active > 0) {
            if (ols_result.xtx_inv.rows() != k_active || ols_result.xtx_inv.cols() != k_active) {
                return false;
            }
            B = ols_result.xtx_inv * xbar;
            const Eigen::VectorXd r = X_tilde_used * B;
            if (weights) {
                const double inv_sum_w = 1.0 / sum_w;
                c_vec = weights->array() * (inv_sum_w - r.array());
            } else {
                const double inv_n = 1.0 / static_cast<double>(n);
                c_vec = Eigen::VectorXd::Constant(n, inv_n) - r;
            }
        } else {
            if (weights) {
                c_vec = weights->array() * (1.0 / sum_w);
            } else {
                const double inv_n = 1.0 / static_cast<double>(n);
                c_vec = Eigen::VectorXd::Constant(n, inv_n);
            }
        }

        if (se_type == StandardErrorType::Robust) {
            const Eigen::ArrayXd u = residuals.array();
            const Eigen::ArrayXd c = c_vec.array();
            if (weights && weights_are_frequencies) {
                // fweight (replication): c_vec already carries one factor of w, so the
                // robust intercept variance is Sum (c*u)^2 / w and the cov meat is
                // Sum c*u^2*x (linear w), matching Stata/areg/reghdfe. aweight/pweight
                // keep the w^2 form in the else-branch.
                const Eigen::ArrayXd wv = weights->array();
                var_alpha = static_cast<long double>(((c * u).square() / wv).sum());
                if (k_active > 0) {
                    const Eigen::VectorXd vec =
                        X_tilde_used.transpose() * (u.square() * c).matrix();
                    cov_beta_alpha = ols_result.xtx_inv * vec;
                }
            } else {
                var_alpha = static_cast<long double>((c * u).square().sum());
                if (k_active > 0) {
                    Eigen::ArrayXd w = Eigen::ArrayXd::Ones(n);
                    if (weights) {
                        w = weights->array();
                    }
                    const Eigen::VectorXd vec =
                        X_tilde_used.transpose() * (w * u.square() * c).matrix();
                    cov_beta_alpha = ols_result.xtx_inv * vec;
                }
            }
            cov_beta_alpha *= robust_scale;
            var_alpha *= static_cast<long double>(robust_scale);
        } else if (se_type == StandardErrorType::Cluster && clusters && !clusters->empty()) {
            if (clusters->size() == 1) {
                const ClusterInterceptMeat meat = compute_cluster_intercept_meat(
                    X_tilde_used, residuals, weights, c_vec, (*clusters)[0]);
                if (k_active > 0) {
                    cov_beta_alpha = ols_result.xtx_inv * meat.cross;
                }
                var_alpha = static_cast<long double>(meat.alpha);
                double scale = ols_result.cov_scale * cluster_ratio;
                cov_beta_alpha *= scale;
                var_alpha *= static_cast<long double>(scale);
            } else {
                const ClusterInterceptMeat meat = compute_multiway_intercept_meat(
                    X_tilde_used, residuals, weights, c_vec, *clusters, g_df, g_adj,
                    ols_result.df_resid);
                if (k_active > 0) {
                    cov_beta_alpha = ols_result.xtx_inv * meat.cross;
                }
                var_alpha = static_cast<long double>(meat.alpha);
                cov_beta_alpha *= cluster_ratio;
                var_alpha *= static_cast<long double>(cluster_ratio);
            }
        } else {
            return false;
        }
        }
    }

    double var_alpha_d = static_cast<double>(var_alpha);
    if (!std::isfinite(var_alpha_d) || var_alpha_d < 0.0) {
        var_alpha_d = 0.0;
    }
    bool used_multiway_fallback = false;
    if (se_type == StandardErrorType::Cluster && clusters && clusters->size() > 1 &&
        k_active > 0 && (!(var_alpha_d > 0.0) || !std::isfinite(var_alpha_d))) {
        Eigen::MatrixXd Vtmp = results.covariance;
        for (int j = 0; j < slope_cols_full; ++j) {
            Vtmp(intercept_idx, j) = 0.0;
            Vtmp(j, intercept_idx) = 0.0;
        }
        Vtmp(intercept_idx, intercept_idx) = var_alpha_d;
        for (int pos = 0; pos < k_active; ++pos) {
            const int col = kept_slope_cols[static_cast<std::size_t>(pos)];
            if (col >= 0 && col < slope_cols_full) {
                Vtmp(intercept_idx, col) = cov_beta_alpha(pos);
                Vtmp(col, intercept_idx) = cov_beta_alpha(pos);
            }
        }

        if (fix_psd(Vtmp)) {
            const double var_psd = Vtmp(intercept_idx, intercept_idx);
            if (std::isfinite(var_psd) && var_psd > 0.0) {
                var_alpha_d = var_psd;
                used_multiway_fallback = true;
                for (int pos = 0; pos < k_active; ++pos) {
                    const int col = kept_slope_cols[static_cast<std::size_t>(pos)];
                    if (col >= 0 && col < slope_cols_full) {
                        cov_beta_alpha(pos) = Vtmp(intercept_idx, col);
                    }
                }
            }
        }

        if (!(var_alpha_d > 0.0) || !std::isfinite(var_alpha_d)) {
            const Eigen::MatrixXd cov_beta =
                results.covariance.topLeftCorner(slope_cols_full, slope_cols_full);
            const Eigen::VectorXd cov_fb = -cov_beta * xbar;
            const double var_fb = xbar.dot(cov_beta * xbar);
            if (std::isfinite(var_fb) && var_fb > 0.0) {
                cov_beta_alpha = cov_fb;
                var_alpha_d = var_fb;
                used_multiway_fallback = true;
            }
        }
    }
    if (used_multiway_fallback) {
        const Eigen::MatrixXd cov_beta =
            results.covariance.topLeftCorner(slope_cols_full, slope_cols_full);
        const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(cov_beta);
        const Eigen::VectorXd z = cod.solve(cov_beta_alpha);
        const double schur_lb = cov_beta_alpha.dot(z);
        if (std::isfinite(schur_lb) && var_alpha_d < schur_lb) {
            var_alpha_d = schur_lb;
        }
    }

    for (int j = 0; j < slope_cols_full; ++j) {
        results.covariance(intercept_idx, j) = 0.0;
        results.covariance(j, intercept_idx) = 0.0;
    }
    results.covariance(intercept_idx, intercept_idx) = var_alpha_d;
    for (int pos = 0; pos < k_active; ++pos) {
        const int col = kept_slope_cols[static_cast<std::size_t>(pos)];
        if (col >= 0 && col < slope_cols_full) {
            results.covariance(intercept_idx, col) = cov_beta_alpha(pos);
            results.covariance(col, intercept_idx) = cov_beta_alpha(pos);
        }
    }
    results.std_errors(intercept_idx) =
        (var_alpha_d >= 0.0)
            ? std::sqrt(var_alpha_d)
            : std::numeric_limits<double>::quiet_NaN();
    return true;
}

std::vector<int> sanitize_endogenous_idx(const std::vector<int>& indices, int num_cols) {
    std::vector<int> cleaned = indices;
    std::sort(cleaned.begin(), cleaned.end());
    cleaned.erase(std::unique(cleaned.begin(), cleaned.end()), cleaned.end());
    for (const int idx : cleaned) {
        if (idx < 0 || idx >= num_cols) {
            throw std::runtime_error("Endogenous index out of bounds");
        }
    }
    return cleaned;
}

Eigen::MatrixXd select_columns(const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<int>& columns) {
    Eigen::MatrixXd out(X.rows(), static_cast<int>(columns.size()));
    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        out.col(i) = X.col(columns[i]);
    }
    return out;
}

Eigen::MatrixXd drop_columns(const Eigen::Ref<const Eigen::MatrixXd>& X,
                             const std::vector<int>& columns) {
    if (columns.empty()) {
        return X;
    }
    std::vector<bool> is_endog(static_cast<std::size_t>(X.cols()), false);
    for (const int idx : columns) {
        is_endog[static_cast<std::size_t>(idx)] = true;
    }
    const int keep = static_cast<int>(X.cols()) - static_cast<int>(columns.size());
    Eigen::MatrixXd out(X.rows(), keep);
    int cursor = 0;
    for (int j = 0; j < X.cols(); ++j) {
        if (!is_endog[static_cast<std::size_t>(j)]) {
            out.col(cursor++) = X.col(j);
        }
    }
    return out;
}

enum class FeLookupMode { Dense, DenseMap, Sparse };

struct FeLookup {
    FeLookupMode mode = FeLookupMode::Sparse;
    int min_id = 0;
    int num_groups = 0;
    std::vector<int> dense_map;
    std::unordered_map<int, int> mapping;

    int index(int raw) const {
        if (mode == FeLookupMode::Dense) {
            return raw - min_id;
        }
        if (mode == FeLookupMode::DenseMap) {
            const int idx = raw - min_id;
            if (idx < 0 || idx >= static_cast<int>(dense_map.size())) {
                throw std::runtime_error("FE lookup missing key");
            }
            const int mapped = dense_map[static_cast<std::size_t>(idx)];
            if (mapped < 0) {
                throw std::runtime_error("FE lookup missing key");
            }
            return mapped;
        }
        auto it = mapping.find(raw);
        if (it == mapping.end()) {
            throw std::runtime_error("FE lookup missing key");
        }
        return it->second;
    }
};

FeLookup build_lookup(const Eigen::VectorXi& raw_ids) {
    FeLookup lookup;
    if (raw_ids.size() == 0) {
        return lookup;
    }

    int min_id = raw_ids(0);
    int max_id = raw_ids(0);
    for (int i = 1; i < raw_ids.size(); ++i) {
        const int v = raw_ids(i);
        if (v < min_id) {
            min_id = v;
        } else if (v > max_id) {
            max_id = v;
        }
    }
    const long long range =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    if (min_id >= 0 && range > 0 && range <= kDenseRangeCap &&
        range <= static_cast<long long>(raw_ids.size()) * 2LL) {
        lookup.mode = FeLookupMode::Dense;
        lookup.min_id = min_id;
        lookup.num_groups = static_cast<int>(range);
        return lookup;
    }

    lookup.mode = FeLookupMode::Sparse;
    lookup.min_id = 0;
    std::unordered_map<int, int> mapping;
    constexpr std::size_t kMaxReserve = 5000000ULL;
    const long long approx_range =
        range > 0 && range <= static_cast<long long>(kMaxReserve) ? range : 0LL;
    const std::size_t reserve =
        approx_range > 0 ? static_cast<std::size_t>(approx_range)
                         : std::min<std::size_t>(static_cast<std::size_t>(raw_ids.size()),
                                                 kMaxReserve);
    if (reserve > 0) {
        mapping.reserve(reserve);
    }

    int next_id = 0;
    for (int i = 0; i < raw_ids.size(); ++i) {
        const int key = raw_ids(i);
        auto it = mapping.find(key);
        if (it == mapping.end()) {
            mapping.emplace(key, next_id);
            ++next_id;
        }
    }
    lookup.mapping = std::move(mapping);
    lookup.num_groups = next_id;
    return lookup;
}

FeLookup build_lookup_compact(const Eigen::VectorXi& raw_ids, int expected_levels) {
    FeLookup lookup;
    if (raw_ids.size() == 0) {
        return lookup;
    }

    int min_id = raw_ids(0);
    int max_id = raw_ids(0);
    for (int i = 1; i < raw_ids.size(); ++i) {
        const int v = raw_ids(i);
        if (v < min_id) {
            min_id = v;
        } else if (v > max_id) {
            max_id = v;
        }
    }
    const long long range =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    if (min_id >= 0 && range > 0 && range <= kDenseRangeCap &&
        range <= static_cast<long long>(raw_ids.size()) * 2LL) {
        if (expected_levels > 0 && expected_levels == range) {
            lookup.mode = FeLookupMode::Dense;
            lookup.min_id = min_id;
            lookup.num_groups = static_cast<int>(range);
            return lookup;
        }
        lookup.mode = FeLookupMode::DenseMap;
        lookup.min_id = min_id;
        lookup.dense_map.assign(static_cast<std::size_t>(range), -1);
        int next_id = 0;
        for (int i = 0; i < raw_ids.size(); ++i) {
            const int idx = raw_ids(i) - min_id;
            int& slot = lookup.dense_map[static_cast<std::size_t>(idx)];
            if (slot < 0) {
                slot = next_id++;
            }
        }
        lookup.num_groups = next_id;
        return lookup;
    }

    lookup.mode = FeLookupMode::Sparse;
    lookup.min_id = 0;
    std::unordered_map<int, int> mapping;
    constexpr std::size_t kMaxReserve = 5000000ULL;
    const std::size_t reserve =
        std::min<std::size_t>(static_cast<std::size_t>(raw_ids.size()), kMaxReserve);
    if (reserve > 0) {
        mapping.reserve(reserve);
    }
    int next_id = 0;
    for (int i = 0; i < raw_ids.size(); ++i) {
        const int key = raw_ids(i);
        auto it = mapping.find(key);
        if (it == mapping.end()) {
            mapping.emplace(key, next_id);
            ++next_id;
        }
    }
    lookup.mapping = std::move(mapping);
    lookup.num_groups = next_id;
    return lookup;
}

struct FeIndexerLite {
    std::vector<int> group_ids;
    int num_groups = 0;
};

FeIndexerLite build_indexer_lite(const Eigen::VectorXi& raw_ids) {
    FeIndexerLite idx;
    idx.group_ids.resize(static_cast<std::size_t>(raw_ids.size()));
    if (raw_ids.size() == 0) {
        idx.num_groups = 0;
        return idx;
    }

    int min_id = raw_ids(0);
    int max_id = raw_ids(0);
    for (int i = 1; i < raw_ids.size(); ++i) {
        const int v = raw_ids(i);
        if (v < min_id) {
            min_id = v;
        } else if (v > max_id) {
            max_id = v;
        }
    }

    const long long range =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    if (min_id >= 0 && range > 0 && range <= kDenseRangeCap &&
        range <= static_cast<long long>(raw_ids.size()) * 2LL) {
        std::vector<int> dense_map(static_cast<std::size_t>(range), -1);
        int next_id = 0;
        for (int i = 0; i < raw_ids.size(); ++i) {
            const int mapped = raw_ids(i) - min_id;
            int& slot = dense_map[static_cast<std::size_t>(mapped)];
            if (slot < 0) {
                slot = next_id++;
            }
            idx.group_ids[static_cast<std::size_t>(i)] = slot;
        }
        idx.num_groups = next_id;
        return idx;
    }

    std::unordered_map<int, int> mapping;
    constexpr std::size_t kMaxReserve = 5000000ULL;
    const std::size_t reserve =
        range > 0 && range <= static_cast<long long>(kMaxReserve)
            ? static_cast<std::size_t>(range)
            : std::min<std::size_t>(static_cast<std::size_t>(raw_ids.size()), kMaxReserve);
    if (reserve > 0) {
        mapping.reserve(reserve);
    }

    int next_id = 0;
    for (int i = 0; i < raw_ids.size(); ++i) {
        const int key = raw_ids(i);
        auto it = mapping.find(key);
        int normalized = 0;
        if (it == mapping.end()) {
            normalized = next_id;
            mapping.emplace(key, next_id);
            ++next_id;
        } else {
            normalized = it->second;
        }
        idx.group_ids[static_cast<std::size_t>(i)] = normalized;
    }
    idx.num_groups = next_id;
    return idx;
}

int count_unique_ids(const Eigen::VectorXi& ids) {
    std::unordered_map<int, int> uniq;
    uniq.reserve(static_cast<std::size_t>(ids.size()));
    for (int i = 0; i < ids.size(); ++i) {
        uniq.emplace(ids(i), 1);
    }
    return static_cast<int>(uniq.size());
}

Eigen::VectorXi combine_clusters(const std::vector<Eigen::VectorXi>& clusters,
                                 const std::vector<int>& dims) {
    if (dims.empty()) {
        throw std::runtime_error("Cannot combine an empty set of cluster dimensions");
    }
    const int n = clusters[static_cast<std::size_t>(dims[0])].size();
    for (const int d : dims) {
        if (clusters[static_cast<std::size_t>(d)].size() != n) {
            throw std::runtime_error("All cluster vectors must have the same length");
        }
    }
    if (dims.size() == 1) {
        return clusters[static_cast<std::size_t>(dims[0])];
    }

    Eigen::VectorXi current = clusters[static_cast<std::size_t>(dims[0])];
    for (std::size_t pos = 1; pos < dims.size(); ++pos) {
        const Eigen::VectorXi& next = clusters[static_cast<std::size_t>(dims[pos])];
        std::unordered_map<std::uint64_t, int> map;
        map.reserve(static_cast<std::size_t>(n));
        Eigen::VectorXi combined(n);
        int next_id = 0;
        for (int i = 0; i < n; ++i) {
            const std::uint32_t a = static_cast<std::uint32_t>(current(i));
            const std::uint32_t b = static_cast<std::uint32_t>(next(i));
            const std::uint64_t key = (static_cast<std::uint64_t>(a) << 32) | b;
            auto it = map.find(key);
            if (it == map.end()) {
                map.emplace(key, next_id);
                combined(i) = next_id;
                ++next_id;
            } else {
                combined(i) = it->second;
            }
        }
        current = std::move(combined);
    }
    return current;
}

std::vector<int> compute_cluster_counts(const std::vector<Eigen::VectorXi>& clusters) {
    std::vector<int> counts;
    counts.reserve(clusters.size());
    for (const auto& c : clusters) {
        counts.push_back(count_unique_ids(c));
    }
    return counts;
}

std::vector<int> compute_cluster_combo_counts(const std::vector<Eigen::VectorXi>& clusters) {
    std::vector<int> counts;
    const int m = static_cast<int>(clusters.size());
    if (m <= 0) {
        return counts;
    }
    if (m == 1) {
        counts.reserve(1);
        counts.push_back(count_unique_ids(clusters[0]));
        return counts;
    }
    const std::uint64_t max_mask = (static_cast<std::uint64_t>(1) << m);
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(m));
    counts.reserve(static_cast<std::size_t>(max_mask - 1));
    for (std::uint64_t mask = 1; mask < max_mask; ++mask) {
        dims.clear();
        for (int j = 0; j < m; ++j) {
            if (mask & (static_cast<std::uint64_t>(1) << j)) {
                dims.push_back(j);
            }
        }
        const Eigen::VectorXi combined = combine_clusters(clusters, dims);
        counts.push_back(count_unique_ids(combined));
    }
    return counts;
}

struct FeDofInfo {
    std::vector<int> levels;
    std::vector<int> redundant;
    std::vector<int> num_coefs;
    std::vector<int> inexact;
    int df_a = 0;
    int df_a_levels = 0;
    int df_a_exact = 0;
};

struct UnionFind {
    std::vector<int> parent;
    std::vector<uint8_t> rank;

    explicit UnionFind(int n) : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0) {
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
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return;
        }
        const std::size_t rra = static_cast<std::size_t>(ra);
        const std::size_t rrb = static_cast<std::size_t>(rb);
        if (rank[rra] < rank[rrb]) {
            parent[rra] = rb;
        } else if (rank[rra] > rank[rrb]) {
            parent[rrb] = ra;
        } else {
            parent[rrb] = ra;
            rank[rra] += 1;
        }
    }
};

constexpr const char* kMobilityProfileDefaultPath = "xhdfe_mobility_profile.txt";
constexpr std::uint64_t kFnvOffset64 = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime64 = 1099511628211ULL;

struct MobilityProfileConfig {
    std::string path;
    std::string mode;  // off, auto, read, write
    bool allow_auto_write = false;
};

struct AbsorptionCacheConfig {
    std::string path;
    std::string mode;  // off, auto, read, write
    bool allow_auto_write = false;
    bool mode_set = false;
};

struct FeStructureCacheConfig {
    std::string path;
    std::string mode;  // off, auto, read, write
    bool allow_auto_write = false;
};

struct MobilityProfile {
    std::uint64_t signature = 0;
    std::uint64_t signature_sample = 0;
    bool has_signature_sample = false;
    std::uint64_t signature_canon = 0;
    std::uint64_t signature_sample_canon = 0;
    bool has_signature_canon = false;
    bool has_signature_sample_canon = false;
    int nobs = 0;
    int nobs_full = 0;
    int num_singletons = 0;
    int nfe = 0;
    std::vector<int> fe_levels;
    int num_components = 0;
    int largest_component = 0;
    double largest_component_share = 0.0;
    std::string mobility_class;
    std::vector<int> sweep_order;
    bool suggest_symmetric = false;
    AbsorptionMethod suggest_method = AbsorptionMethod::Auto;
    bool suggest_use_sparse = false;
    bool suggest_use_krylov = false;
};

struct MobilityHint {
    std::vector<int> sweep_order;
    bool force_symmetric = false;
    AbsorptionMethod preferred_method = AbsorptionMethod::Auto;
    bool has_method = false;
    bool use_sparse = false;
    bool use_krylov = false;
};

std::string to_lower_ascii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

std::string trim_ascii(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

bool gpu_backend_env_requested() {
    const char* raw = std::getenv("XHDFE_GPU_BACKEND");
    if (!raw || *raw == '\0') {
        return false;
    }
    const std::string value = to_lower_ascii(trim_ascii(raw));
    return value == "cuda" || value == "metal";
}

std::optional<bool> read_env_bool(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || *raw == '\0') {
        return std::nullopt;
    }
    const std::string value = to_lower_ascii(trim_ascii(raw));
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return std::nullopt;
}

int read_env_int(const char* name, int default_value, int min_value) {
    const char* raw = std::getenv(name);
    if (!raw || *raw == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed < static_cast<long>(min_value)) {
        return default_value;
    }
    if (parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(parsed);
}

double read_env_double(const char* name,
                       double default_value,
                       double min_value,
                       double max_value) {
    const char* raw = std::getenv(name);
    if (!raw || *raw == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(parsed) || parsed < min_value ||
        parsed > max_value) {
        return default_value;
    }
    return parsed;
}

bool file_exists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path);
    return static_cast<bool>(in);
}

std::uint64_t fnv1a_update(std::uint64_t hash, std::uint64_t value) {
    hash ^= value;
    hash *= kFnvPrime64;
    return hash;
}

std::uint64_t hash_fe_signature(const std::vector<Eigen::VectorXi>& fes,
                                bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(fes.size()));
    const std::uint64_t nobs = fes.empty() ? 0ULL : static_cast<std::uint64_t>(fes[0].size());
    hash = fnv1a_update(hash, nobs);
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    for (const auto& fe : fes) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(fe.size()));
        for (int i = 0; i < fe.size(); ++i) {
            const std::int64_t v = static_cast<std::int64_t>(fe(i));
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
        }
    }
    return hash;
}

std::uint64_t hash_fe_signature_sample(const std::vector<Eigen::VectorXi>& fes,
                                       bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(fes.size()));
    const int nobs = fes.empty() ? 0 : static_cast<int>(fes[0].size());
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(nobs));
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    if (nobs <= 0) {
        return hash;
    }
    const int max_samples = 4096;
    const int sample_count = std::min(nobs, max_samples);
    const int stride = std::max(1, nobs / sample_count);
    std::vector<int> sample_idx;
    sample_idx.reserve(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < nobs && static_cast<int>(sample_idx.size()) < sample_count; i += stride) {
        sample_idx.push_back(i);
    }
    for (const auto& fe : fes) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(fe.size()));
        for (int idx : sample_idx) {
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(idx));
            const std::int64_t v = static_cast<std::int64_t>(fe(idx));
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
        }
    }
    return hash;
}

std::uint64_t hash_fe_signature_filtered(const std::vector<Eigen::VectorXi>& fes,
                                         const std::vector<int>* kept_idx,
                                         bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(fes.size()));
    const std::uint64_t nobs = kept_idx
                                   ? static_cast<std::uint64_t>(kept_idx->size())
                                   : (fes.empty() ? 0ULL
                                                  : static_cast<std::uint64_t>(fes[0].size()));
    hash = fnv1a_update(hash, nobs);
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    for (const auto& fe : fes) {
        hash = fnv1a_update(hash, nobs);
        if (kept_idx) {
            for (int idx : *kept_idx) {
                const std::int64_t v = static_cast<std::int64_t>(fe(idx));
                hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
            }
        } else {
            for (int i = 0; i < fe.size(); ++i) {
                const std::int64_t v = static_cast<std::int64_t>(fe(i));
                hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
            }
        }
    }
    return hash;
}

std::uint64_t hash_fe_signature_sample_filtered(const std::vector<Eigen::VectorXi>& fes,
                                                const std::vector<int>* kept_idx,
                                                bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(fes.size()));
    const int nobs = kept_idx
                         ? static_cast<int>(kept_idx->size())
                         : (fes.empty() ? 0 : static_cast<int>(fes[0].size()));
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(nobs));
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    if (nobs <= 0) {
        return hash;
    }
    const int max_samples = 4096;
    const int sample_count = std::min(nobs, max_samples);
    const int stride = std::max(1, nobs / sample_count);
    std::vector<int> sample_idx;
    sample_idx.reserve(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < nobs && static_cast<int>(sample_idx.size()) < sample_count; i += stride) {
        sample_idx.push_back(i);
    }
    for (const auto& fe : fes) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(nobs));
        for (int pos : sample_idx) {
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(pos));
            const int raw_idx = kept_idx ? (*kept_idx)[static_cast<std::size_t>(pos)] : pos;
            const std::int64_t v = static_cast<std::int64_t>(fe(raw_idx));
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
        }
    }
    return hash;
}

std::uint64_t hash_fe_signature_from_indexers(const std::vector<FeIndexerLite>& indexers,
                                              int nobs,
                                              bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(indexers.size()));
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(std::max(0, nobs)));
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    for (const auto& idx : indexers) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(idx.group_ids.size()));
        for (int v : idx.group_ids) {
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
        }
    }
    return hash;
}

std::uint64_t hash_fe_signature_sample_from_indexers(const std::vector<FeIndexerLite>& indexers,
                                                     int nobs,
                                                     bool drop_singletons) {
    std::uint64_t hash = kFnvOffset64;
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(indexers.size()));
    hash = fnv1a_update(hash, static_cast<std::uint64_t>(std::max(0, nobs)));
    hash = fnv1a_update(hash, drop_singletons ? 1ULL : 0ULL);
    if (nobs <= 0) {
        return hash;
    }
    const int max_samples = 4096;
    const int sample_count = std::min(nobs, max_samples);
    const int stride = std::max(1, nobs / sample_count);
    std::vector<int> sample_idx;
    sample_idx.reserve(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < nobs && static_cast<int>(sample_idx.size()) < sample_count; i += stride) {
        sample_idx.push_back(i);
    }
    for (const auto& idx : indexers) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(idx.group_ids.size()));
        for (int pos : sample_idx) {
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(pos));
            const std::int64_t v =
                static_cast<std::int64_t>(idx.group_ids[static_cast<std::size_t>(pos)]);
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(v));
        }
    }
    return hash;
}

std::uint64_t hash_fe_structure_cache_signature(const std::vector<Eigen::VectorXi>& fes,
                                                const Eigen::VectorXd* weights,
                                                bool drop_singletons,
                                                bool weights_are_frequencies) {
    std::uint64_t hash = hash_fe_signature(fes, drop_singletons);
    hash = fnv1a_update(hash, weights_are_frequencies ? 1ULL : 0ULL);
    if (drop_singletons && weights_are_frequencies && weights) {
        hash = fnv1a_update(hash, static_cast<std::uint64_t>(weights->size()));
        for (Eigen::Index i = 0; i < weights->size(); ++i) {
            const double w_raw = (*weights)(i);
            if (!std::isfinite(w_raw) || !(w_raw > 0.0)) {
                // Include a sentinel so invalid weights cannot accidentally reuse a cache entry.
                hash = fnv1a_update(hash, 0ULL);
                continue;
            }
            const std::int64_t w = static_cast<std::int64_t>(std::llround(w_raw));
            hash = fnv1a_update(hash, static_cast<std::uint64_t>(w));
        }
    }
    return hash;
}

constexpr const char* kAbsorptionCacheMagic = "xhdfe_absorption_cache_v1";
constexpr std::size_t kAbsorptionCacheMagicSize = 32;
constexpr std::uint64_t kAbsorptionCacheSalt = 0x9e3779b97f4a7c15ULL;

struct AbsorptionCacheKey {
    std::uint64_t sig1 = 0;
    std::uint64_t sig2 = 0;
};

struct AbsorptionCacheRecord {
    AbsorptionMethod method = AbsorptionMethod::Auto;
    int design_cols = 0;
    int nobs = 0;
    int cols = 0;
    int iterations = 0;
    bool converged = true;
    std::vector<int> fe_levels;
    std::vector<int> sweep_order;
    Eigen::VectorXd y_tilde;
    Eigen::MatrixXd X_tilde;
};

std::uint64_t hash_double_bits(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void hash_update_double(std::uint64_t& hash, double value) {
    hash = fnv1a_update(hash, hash_double_bits(value));
}

AbsorptionCacheKey hash_absorption_signature(const Eigen::Ref<const Eigen::VectorXd>& y,
                                             const Eigen::Ref<const Eigen::MatrixXd>& X,
                                             const std::vector<Eigen::VectorXi>& fes,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options) {
    std::uint64_t h1 = kFnvOffset64;
    std::uint64_t h2 = kFnvOffset64 ^ kAbsorptionCacheSalt;
    auto update = [&](std::uint64_t value) {
        h1 = fnv1a_update(h1, value);
        h2 = fnv1a_update(h2, value + kAbsorptionCacheSalt);
    };

    update(static_cast<std::uint64_t>(y.size()));
    update(static_cast<std::uint64_t>(X.rows()));
    update(static_cast<std::uint64_t>(X.cols()));
    update(static_cast<std::uint64_t>(fes.size()));
    update(weights ? 1ULL : 0ULL);
    update(options.fit_intercept ? 1ULL : 0ULL);
    update(options.drop_singletons ? 1ULL : 0ULL);
    update(static_cast<std::uint64_t>(options.max_iter));
    update(static_cast<std::uint64_t>(static_cast<int>(options.tolerance_mode)));
    hash_update_double(h1, options.tol);
    hash_update_double(h2, options.tol);

    // Solver-selection knobs that change the absorption transform or its
    // convergence path. Including them guarantees a cache entry computed under
    // one solver configuration is never reused for a different configuration
    // (e.g. forced method A vs B, symmetric sweep, Krylov/sparse, custom sweep
    // order). Computed once per fit(), outside all iteration loops.
    update(static_cast<std::uint64_t>(static_cast<int>(options.absorption_method)));
    update(static_cast<std::uint64_t>(static_cast<int>(options.convergence_criterion)));
    update(options.symmetric_sweep ? 1ULL : 0ULL);
    update(options.use_sparse_solver ? 1ULL : 0ULL);
    update(options.use_krylov ? 1ULL : 0ULL);
    update(static_cast<std::uint64_t>(static_cast<std::uint32_t>(options.convergence_check_interval)));
    hash_update_double(h1, options.krylov_lambda);
    hash_update_double(h2, options.krylov_lambda);
    hash_update_double(h1, options.jacobi_relaxation);
    hash_update_double(h2, options.jacobi_relaxation);
    hash_update_double(h1, options.sparse_threshold);
    hash_update_double(h2, options.sparse_threshold);
    update(static_cast<std::uint64_t>(options.sweep_order_override.size()));
    for (int so : options.sweep_order_override) {
        update(static_cast<std::uint64_t>(static_cast<std::int64_t>(so)));
    }

    for (const auto& fe : fes) {
        update(static_cast<std::uint64_t>(fe.size()));
        for (int i = 0; i < fe.size(); ++i) {
            const std::int64_t v = static_cast<std::int64_t>(fe(i));
            update(static_cast<std::uint64_t>(v));
        }
    }

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        hash_update_double(h1, y(i));
        hash_update_double(h2, y(i));
    }
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const double v = X(i, j);
            hash_update_double(h1, v);
            hash_update_double(h2, v);
        }
    }
    if (weights) {
        for (Eigen::Index i = 0; i < weights->size(); ++i) {
            const double v = (*weights)(i);
            hash_update_double(h1, v);
            hash_update_double(h2, v);
        }
    }

    AbsorptionCacheKey key;
    key.sig1 = h1;
    key.sig2 = h2;
    return key;
}

std::string absorption_cache_path(const std::string& profile_path) {
    if (profile_path.empty()) {
        return {};
    }
    return profile_path + ".absorption_cache";
}

template <typename T>
bool write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_binary(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

bool write_int_vector(std::ofstream& out, const std::vector<int>& values) {
    const std::int32_t count = static_cast<std::int32_t>(values.size());
    if (!write_binary(out, count)) {
        return false;
    }
    for (int v : values) {
        const std::int32_t entry = static_cast<std::int32_t>(v);
        if (!write_binary(out, entry)) {
            return false;
        }
    }
    return true;
}

bool read_int_vector(std::ifstream& in, std::vector<int>& values) {
    std::int32_t count = 0;
    if (!read_binary(in, count)) {
        return false;
    }
    if (count < 0) {
        return false;
    }
    values.assign(static_cast<std::size_t>(count), 0);
    for (std::int32_t i = 0; i < count; ++i) {
        std::int32_t entry = 0;
        if (!read_binary(in, entry)) {
            return false;
        }
        values[static_cast<std::size_t>(i)] = static_cast<int>(entry);
    }
    return true;
}

bool write_absorption_cache(const std::string& path,
                            const AbsorptionCacheKey& key,
                            const Eigen::VectorXd& y_tilde,
                            const Eigen::MatrixXd& X_tilde,
                            const std::vector<int>& fe_levels,
                            const std::vector<int>& sweep_order,
                            int design_cols,
                            AbsorptionMethod method,
                            int iterations,
                            bool converged) {
    if (path.empty()) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    char magic[kAbsorptionCacheMagicSize] = {};
    std::strncpy(magic, kAbsorptionCacheMagic, kAbsorptionCacheMagicSize - 1);
    out.write(magic, sizeof(magic));
    if (!write_binary(out, key.sig1) || !write_binary(out, key.sig2)) {
        return false;
    }

    const std::int32_t nobs = static_cast<std::int32_t>(y_tilde.size());
    const std::int32_t cols = static_cast<std::int32_t>(X_tilde.cols());
    const std::int32_t design_cols_out = static_cast<std::int32_t>(design_cols);
    const std::int32_t method_out = static_cast<std::int32_t>(method);
    const std::int32_t iterations_out = static_cast<std::int32_t>(iterations);
    const std::int32_t converged_out = converged ? 1 : 0;
    if (!write_binary(out, nobs) || !write_binary(out, cols) ||
        !write_binary(out, design_cols_out) || !write_binary(out, method_out) ||
        !write_binary(out, iterations_out) || !write_binary(out, converged_out)) {
        return false;
    }
    if (!write_int_vector(out, fe_levels) || !write_int_vector(out, sweep_order)) {
        return false;
    }
    if (!write_binary(out, y_tilde.size())) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(y_tilde.data()),
              static_cast<std::streamsize>(y_tilde.size() * sizeof(double)));
    if (!out) {
        return false;
    }

    const std::size_t x_size =
        static_cast<std::size_t>(X_tilde.size());
    if (!write_binary(out, x_size)) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(X_tilde.data()),
              static_cast<std::streamsize>(x_size * sizeof(double)));
    return static_cast<bool>(out);
}

bool read_absorption_cache(const std::string& path,
                           const AbsorptionCacheKey& key,
                           AbsorptionCacheRecord& record) {
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[kAbsorptionCacheMagicSize] = {};
    in.read(magic, sizeof(magic));
    if (!in) {
        return false;
    }
    if (std::strncmp(magic, kAbsorptionCacheMagic, std::strlen(kAbsorptionCacheMagic)) != 0) {
        return false;
    }

    std::uint64_t sig1 = 0;
    std::uint64_t sig2 = 0;
    if (!read_binary(in, sig1) || !read_binary(in, sig2)) {
        return false;
    }
    if (sig1 != key.sig1 || sig2 != key.sig2) {
        return false;
    }

    std::int32_t nobs = 0;
    std::int32_t cols = 0;
    std::int32_t design_cols = 0;
    std::int32_t method = 0;
    std::int32_t iterations = 0;
    std::int32_t converged = 0;
    if (!read_binary(in, nobs) || !read_binary(in, cols) ||
        !read_binary(in, design_cols) || !read_binary(in, method) ||
        !read_binary(in, iterations) || !read_binary(in, converged)) {
        return false;
    }
    if (nobs < 0 || cols < 0 || design_cols < 0) {
        return false;
    }

    if (!read_int_vector(in, record.fe_levels) || !read_int_vector(in, record.sweep_order)) {
        return false;
    }

    Eigen::Index y_size = 0;
    if (!read_binary(in, y_size) || y_size < 0) {
        return false;
    }
    if (y_size != static_cast<Eigen::Index>(nobs)) {
        return false;
    }
    record.y_tilde.resize(y_size);
    in.read(reinterpret_cast<char*>(record.y_tilde.data()),
            static_cast<std::streamsize>(static_cast<std::size_t>(y_size) * sizeof(double)));
    if (!in) {
        return false;
    }

    std::size_t x_size = 0;
    if (!read_binary(in, x_size)) {
        return false;
    }
    record.X_tilde.resize(nobs, cols);
    const std::size_t expected = static_cast<std::size_t>(nobs) *
                                 static_cast<std::size_t>(cols);
    if (x_size != expected) {
        return false;
    }
    in.read(reinterpret_cast<char*>(record.X_tilde.data()),
            static_cast<std::streamsize>(x_size * sizeof(double)));
    if (!in) {
        return false;
    }

    record.nobs = nobs;
    record.cols = cols;
    record.design_cols = design_cols;
    record.method = static_cast<AbsorptionMethod>(method);
    record.iterations = iterations;
    record.converged = (converged != 0);
    return true;
}

constexpr const char* kFeStructureCacheMagic = "xhdfe_fe_structure_cache_v1";
constexpr std::size_t kFeStructureCacheMagicSize = 32;

struct FeStructureCache {
    std::uint64_t signature = 0;
    int nobs_full = 0;
    int nobs = 0;
    bool drop_singletons = false;
    std::vector<int> kept_idx;
    std::vector<int> num_groups;
    std::vector<Eigen::VectorXi> fe_group_ids;
};

bool write_fe_structure_cache(const std::string& path,
                              std::uint64_t signature,
                              int nobs_full,
                              bool drop_singletons,
                              const std::vector<int>& kept_idx,
                              const std::vector<int>& num_groups,
                              const std::vector<Eigen::VectorXi>& fe_group_ids) {
    if (path.empty()) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    char magic[kFeStructureCacheMagicSize] = {};
    std::strncpy(magic, kFeStructureCacheMagic, kFeStructureCacheMagicSize - 1);
    out.write(magic, sizeof(magic));
    if (!write_binary(out, signature)) {
        return false;
    }

    const std::int32_t nobs_full_out = static_cast<std::int32_t>(nobs_full);
    const std::int32_t nobs_out =
        fe_group_ids.empty() ? nobs_full_out : static_cast<std::int32_t>(fe_group_ids[0].size());
    const std::int32_t nfe_out = static_cast<std::int32_t>(fe_group_ids.size());
    const std::int32_t drop_out = drop_singletons ? 1 : 0;
    if (!write_binary(out, nobs_full_out) || !write_binary(out, nobs_out) ||
        !write_binary(out, nfe_out) || !write_binary(out, drop_out)) {
        return false;
    }

    const std::int32_t keep_count = static_cast<std::int32_t>(kept_idx.size());
    if (!write_binary(out, keep_count)) {
        return false;
    }
    for (int idx : kept_idx) {
        const std::int32_t entry = static_cast<std::int32_t>(idx);
        if (!write_binary(out, entry)) {
            return false;
        }
    }

    for (std::size_t d = 0; d < fe_group_ids.size(); ++d) {
        const std::int32_t groups = static_cast<std::int32_t>(num_groups[d]);
        const std::int32_t len = static_cast<std::int32_t>(fe_group_ids[d].size());
        if (!write_binary(out, groups) || !write_binary(out, len)) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(fe_group_ids[d].data()),
                  static_cast<std::streamsize>(len * sizeof(int)));
        if (!out) {
            return false;
        }
    }
    return static_cast<bool>(out);
}

bool read_fe_structure_cache(const std::string& path,
                             std::uint64_t expected_signature,
                             int expected_nobs_full,
                             bool drop_singletons,
                             int expected_nfe,
                             FeStructureCache& cache) {
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[kFeStructureCacheMagicSize] = {};
    in.read(magic, sizeof(magic));
    if (!in) {
        return false;
    }
    if (std::strncmp(magic, kFeStructureCacheMagic, std::strlen(kFeStructureCacheMagic)) != 0) {
        return false;
    }

    std::uint64_t signature = 0;
    if (!read_binary(in, signature)) {
        return false;
    }
    if (signature != expected_signature) {
        return false;
    }

    std::int32_t nobs_full = 0;
    std::int32_t nobs = 0;
    std::int32_t nfe = 0;
    std::int32_t drop_flag = 0;
    if (!read_binary(in, nobs_full) || !read_binary(in, nobs) ||
        !read_binary(in, nfe) || !read_binary(in, drop_flag)) {
        return false;
    }
    if (nobs_full < 0 || nobs < 0 || nfe < 0) {
        return false;
    }
    if (nobs_full != expected_nobs_full || nfe != expected_nfe) {
        return false;
    }
    if ((drop_flag != 0) != drop_singletons) {
        return false;
    }

    std::int32_t keep_count = 0;
    if (!read_binary(in, keep_count) || keep_count < 0) {
        return false;
    }
    std::vector<int> kept_idx;
    kept_idx.resize(static_cast<std::size_t>(keep_count));
    for (std::int32_t i = 0; i < keep_count; ++i) {
        std::int32_t entry = 0;
        if (!read_binary(in, entry)) {
            return false;
        }
        kept_idx[static_cast<std::size_t>(i)] = static_cast<int>(entry);
    }

    if (nobs == nobs_full && keep_count != 0) {
        return false;
    }
    if (nobs != nobs_full && keep_count != nobs) {
        return false;
    }

    std::vector<int> num_groups;
    std::vector<Eigen::VectorXi> fe_group_ids;
    num_groups.resize(static_cast<std::size_t>(nfe));
    fe_group_ids.resize(static_cast<std::size_t>(nfe));
    for (int d = 0; d < nfe; ++d) {
        std::int32_t groups = 0;
        std::int32_t len = 0;
        if (!read_binary(in, groups) || !read_binary(in, len)) {
            return false;
        }
        if (groups < 0 || len != nobs) {
            return false;
        }
        num_groups[static_cast<std::size_t>(d)] = static_cast<int>(groups);
        fe_group_ids[static_cast<std::size_t>(d)].resize(len);
        in.read(reinterpret_cast<char*>(fe_group_ids[static_cast<std::size_t>(d)].data()),
                static_cast<std::streamsize>(len * sizeof(int)));
        if (!in) {
            return false;
        }
    }

    cache.signature = signature;
    cache.nobs_full = nobs_full;
    cache.nobs = nobs;
    cache.drop_singletons = drop_singletons;
    cache.kept_idx = std::move(kept_idx);
    cache.num_groups = std::move(num_groups);
    cache.fe_group_ids = std::move(fe_group_ids);
    return true;
}

double combined_norm_sq(const Eigen::VectorXd& y, const Eigen::MatrixXd& X) {
    return y.squaredNorm() + X.squaredNorm();
}

struct PilotSample {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    std::optional<Eigen::VectorXd> weights;
};

PilotSample build_pilot_sample(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<Eigen::VectorXi>& fes,
                               const Eigen::VectorXd* weights,
                               int max_rows) {
    PilotSample sample;
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    if (n <= 0 || max_rows <= 0) {
        return sample;
    }
    const int sample_count = std::min(n, max_rows);
    const int stride = std::max(1, n / sample_count);
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < n && static_cast<int>(indices.size()) < sample_count; i += stride) {
        indices.push_back(i);
    }
    sample.y.resize(static_cast<int>(indices.size()));
    sample.X.resize(static_cast<int>(indices.size()), cols);
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        const int idx = indices[static_cast<std::size_t>(i)];
        sample.y(i) = y(idx);
        if (cols > 0) {
            sample.X.row(i) = X.row(idx);
        }
    }
    sample.fes.reserve(fes.size());
    for (const auto& fe : fes) {
        Eigen::VectorXi fe_sample(static_cast<int>(indices.size()));
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            fe_sample(i) = fe(indices[static_cast<std::size_t>(i)]);
        }
        sample.fes.push_back(std::move(fe_sample));
    }
    if (weights) {
        Eigen::VectorXd w_sample(static_cast<int>(indices.size()));
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            w_sample(i) = (*weights)(indices[static_cast<std::size_t>(i)]);
        }
        sample.weights = std::move(w_sample);
    }
    return sample;
}

struct PilotScore {
    AbsorptionMethod method = AbsorptionMethod::Auto;
    double elapsed_ms = 0.0;
    double norm_ratio = 1.0;
    double score = std::numeric_limits<double>::infinity();
};

PilotScore run_pilot_candidate(const PilotSample& sample,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               int iterations) {
    PilotScore result;
    result.method = method;
    if (sample.y.size() == 0) {
        return result;
    }
    const double norm_before = combined_norm_sq(sample.y, sample.X);

    HdfeOptions pilot_opts = options;
    pilot_opts.max_iter = std::max(1, iterations);
    pilot_opts.tol = 0.0;
    pilot_opts.convergence_check_interval = pilot_opts.max_iter;
    pilot_opts.retain_fixed_effects = false;
    pilot_opts.save_groupvar = false;
    pilot_opts.use_sparse_solver = false;
    pilot_opts.use_krylov = false;

    const Eigen::VectorXd* w_ptr = sample.weights ? &(*sample.weights) : nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const detail::AbsorptionResult absorption =
        detail::absorb_fixed_effects_v6(sample.y, sample.X, sample.fes, w_ptr, pilot_opts, method);
    const auto t1 = std::chrono::steady_clock::now();

    const double norm_after = combined_norm_sq(absorption.y_tilde, absorption.X_tilde);
    const double ratio = (norm_before > 0.0) ? (norm_after / norm_before) : 1.0;
    const double reduction = std::max(0.0, 1.0 - ratio);
    const double elapsed =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.elapsed_ms = elapsed;
    result.norm_ratio = ratio;
    if (reduction > 1e-9) {
        result.score = elapsed / reduction;
    }
    return result;
}

std::optional<AbsorptionMethod> select_method_pilot(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options) {
    const int dims = static_cast<int>(fes.size());
    if (dims <= 1) {
        return AbsorptionMethod::GaussSeidel;
    }
    constexpr int kPilotMaxRows = 50000;
    constexpr int kPilotIterations = 2;
    PilotSample sample = build_pilot_sample(y, X, fes, weights, kPilotMaxRows);
    if (sample.y.size() == 0) {
        return std::nullopt;
    }

    std::vector<AbsorptionMethod> candidates;
    candidates.reserve(3);
    candidates.push_back(AbsorptionMethod::GaussSeidel);
    if (dims > 1) {
        candidates.push_back(AbsorptionMethod::SymmetricGaussSeidel);
    }
    if (options.num_threads >= 2 && dims > 1) {
        candidates.push_back(AbsorptionMethod::Jacobi);
    }

    std::vector<PilotScore> scores;
    scores.reserve(candidates.size());
    PilotScore best;
    bool have_score = false;
    for (const auto method : candidates) {
        PilotScore score = run_pilot_candidate(sample, options, method, kPilotIterations);
        scores.push_back(score);
        if (std::isfinite(score.score)) {
            if (!have_score || score.score < best.score) {
                best = score;
                have_score = true;
            }
        }
    }
    if (have_score) {
        return best.method;
    }
    AbsorptionMethod fastest = AbsorptionMethod::Auto;
    double fastest_time = std::numeric_limits<double>::infinity();
    for (const auto& score : scores) {
        if (score.elapsed_ms > 0.0 && score.elapsed_ms < fastest_time) {
            fastest_time = score.elapsed_ms;
            fastest = score.method;
        }
    }
    if (fastest != AbsorptionMethod::Auto) {
        return fastest;
    }
    return std::nullopt;
}

std::string method_name(AbsorptionMethod method) {
    switch (method) {
        case AbsorptionMethod::GaussSeidel:
            return "gauss-seidel";
        case AbsorptionMethod::SymmetricGaussSeidel:
            return "symmetric-gauss-seidel";
        case AbsorptionMethod::Jacobi:
            return "jacobi";
        case AbsorptionMethod::Schwarz:
            return "schwarz";
        case AbsorptionMethod::Lsmr:
            return "lsmr";
        case AbsorptionMethod::Mlsmr:
            return "mlsmr";
        case AbsorptionMethod::AutoMlsmr:
            return "auto-mlsmr";
        case AbsorptionMethod::Auto:
        default:
            return "auto";
    }
}


// ---- Auto-MLSMR convergence-rate probe (ported from Tiago branch 24jun2026) ----
// Decides MLSMR vs sweep by sampling <=200k rows and measuring per-sweep FE-mean
// contraction. Used by absorptionmethod(auto-mlsmr) AND the default-auto promotion.
struct AutoMlsmrProbeSample {
    std::vector<FeIndexerLite> indexers;
    Eigen::VectorXd weights;
    std::vector<int> original_rows;
    int n = 0;
    int total_levels = 0;
    bool weighted = false;
};

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double deterministic_probe_value(int original_row) {
    const std::uint64_t bits =
        splitmix64(static_cast<std::uint64_t>(original_row) + 0x6a09e667f3bcc909ULL);
    const double unit =
        static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
    return unit - 0.5;
}

AutoMlsmrProbeSample build_auto_mlsmr_probe_sample(
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    int max_rows) {
    AutoMlsmrProbeSample sample;
    if (fes.empty() || max_rows <= 0) {
        return sample;
    }
    const int n = static_cast<int>(fes[0].size());
    if (n <= 0) {
        return sample;
    }
    const int sample_count = std::min(n, max_rows);
    sample.original_rows.reserve(static_cast<std::size_t>(sample_count));
    if (sample_count == n) {
        for (int i = 0; i < n; ++i) {
            sample.original_rows.push_back(i);
        }
    } else {
        // Preserve local FE connectivity in sorted panels. A thin stride sample
        // can turn long worker spells into near-singletons and make the MAP
        // convergence probe falsely optimistic.
        const int block_count = std::min(4, sample_count);
        const int base_block = sample_count / block_count;
        const int remainder = sample_count % block_count;
        int added = 0;
        for (int b = 0; b < block_count; ++b) {
            const int block_size = base_block + (b < remainder ? 1 : 0);
            if (block_size <= 0) {
                continue;
            }
            const int max_start = std::max(0, n - block_size);
            const int start = block_count == 1
                                  ? 0
                                  : static_cast<int>(
                                        std::llround(static_cast<double>(max_start) *
                                                     static_cast<double>(b) /
                                                     static_cast<double>(block_count - 1)));
            for (int j = 0; j < block_size; ++j) {
                sample.original_rows.push_back(start + j);
                ++added;
            }
        }
        if (added < sample_count) {
            for (int i = 0; i < n && added < sample_count; ++i) {
                sample.original_rows.push_back(i);
                ++added;
            }
        }
        std::sort(sample.original_rows.begin(), sample.original_rows.end());
        sample.original_rows.erase(
            std::unique(sample.original_rows.begin(), sample.original_rows.end()),
            sample.original_rows.end());
        const std::vector<int> existing_rows = sample.original_rows;
        for (int i = 0; i < n &&
             static_cast<int>(sample.original_rows.size()) < sample_count;
             ++i) {
            if (!std::binary_search(existing_rows.begin(),
                                    existing_rows.end(),
                                    i)) {
                sample.original_rows.push_back(i);
            }
        }
        std::sort(sample.original_rows.begin(), sample.original_rows.end());
    }
    sample.n = static_cast<int>(sample.original_rows.size());
    if (sample.n <= 0) {
        return sample;
    }

    sample.indexers.reserve(fes.size());
    for (const auto& fe : fes) {
        Eigen::VectorXi fe_sample(sample.n);
        for (int i = 0; i < sample.n; ++i) {
            fe_sample(i) = fe(sample.original_rows[static_cast<std::size_t>(i)]);
        }
        sample.indexers.push_back(build_indexer_lite(fe_sample));
        sample.total_levels += sample.indexers.back().num_groups;
    }

    if (weights) {
        sample.weighted = true;
        sample.weights.resize(sample.n);
        for (int i = 0; i < sample.n; ++i) {
            sample.weights(i) = (*weights)(sample.original_rows[static_cast<std::size_t>(i)]);
        }
    }
    return sample;
}

Eigen::VectorXd make_auto_mlsmr_probe_vector(const AutoMlsmrProbeSample& sample) {
    Eigen::VectorXd v(sample.n);
    long double weighted_sum = 0.0L;
    long double weight_total = 0.0L;
    for (int i = 0; i < sample.n; ++i) {
        const double value =
            deterministic_probe_value(sample.original_rows[static_cast<std::size_t>(i)]);
        v(i) = value;
        const double weight = sample.weighted ? sample.weights(i) : 1.0;
        weighted_sum += static_cast<long double>(weight) * value;
        weight_total += static_cast<long double>(weight);
    }
    if (weight_total > 0.0L) {
        const double mean = static_cast<double>(weighted_sum / weight_total);
        v.array() -= mean;
    }
    return v;
}

void apply_probe_demean_dimension(Eigen::VectorXd& v,
                                  const FeIndexerLite& idx,
                                  const Eigen::VectorXd* weights) {
    if (idx.num_groups <= 0) {
        return;
    }
    std::vector<double> sums(static_cast<std::size_t>(idx.num_groups), 0.0);
    std::vector<double> denoms(static_cast<std::size_t>(idx.num_groups), 0.0);
    const int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        const double weight = weights ? (*weights)(i) : 1.0;
        sums[static_cast<std::size_t>(g)] += weight * v(i);
        denoms[static_cast<std::size_t>(g)] += weight;
    }
    for (int g = 0; g < idx.num_groups; ++g) {
        const double denom = denoms[static_cast<std::size_t>(g)];
        if (denom > 0.0) {
            sums[static_cast<std::size_t>(g)] /= denom;
        }
    }
    for (int i = 0; i < n; ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        v(i) -= sums[static_cast<std::size_t>(g)];
    }
}

void apply_probe_sweep(Eigen::VectorXd& v,
                       const AutoMlsmrProbeSample& sample,
                       AbsorptionMethod sweep_method) {
    const Eigen::VectorXd* weights = sample.weighted ? &sample.weights : nullptr;
    for (const auto& idx : sample.indexers) {
        apply_probe_demean_dimension(v, idx, weights);
    }
    if (sweep_method == AbsorptionMethod::SymmetricGaussSeidel &&
        sample.indexers.size() > 1) {
        for (std::size_t pos = sample.indexers.size(); pos-- > 0;) {
            apply_probe_demean_dimension(v, sample.indexers[pos], weights);
        }
    }
}

double probe_fe_mean_norm(const Eigen::VectorXd& v,
                          const AutoMlsmrProbeSample& sample) {
    const Eigen::VectorXd* weights = sample.weighted ? &sample.weights : nullptr;
    long double total = 0.0L;
    long double denom_total = 0.0L;
    const int n = static_cast<int>(v.size());
    for (const auto& idx : sample.indexers) {
        if (idx.num_groups <= 0) {
            continue;
        }
        std::vector<double> sums(static_cast<std::size_t>(idx.num_groups), 0.0);
        std::vector<double> denoms(static_cast<std::size_t>(idx.num_groups), 0.0);
        for (int i = 0; i < n; ++i) {
            const int g = idx.group_ids[static_cast<std::size_t>(i)];
            const double weight = weights ? (*weights)(i) : 1.0;
            sums[static_cast<std::size_t>(g)] += weight * v(i);
            denoms[static_cast<std::size_t>(g)] += weight;
        }
        for (int g = 0; g < idx.num_groups; ++g) {
            const double denom = denoms[static_cast<std::size_t>(g)];
            if (denom <= 0.0) {
                continue;
            }
            const double mean = sums[static_cast<std::size_t>(g)] / denom;
            total += static_cast<long double>(denom) * mean * mean;
            denom_total += static_cast<long double>(denom);
        }
    }
    if (denom_total <= 0.0L) {
        return 0.0;
    }
    return std::sqrt(static_cast<double>(total / denom_total));
}

AbsorptionMethod select_auto_mlsmr_method(const std::vector<Eigen::VectorXi>& fes,
                                          const Eigen::VectorXd* weights,
                                          const HdfeOptions& options,
                                          int rhs_count,
                                          AbsorptionMethod sweep_fallback) {
    const bool trace = read_env_bool("XHDFE_AUTO_MLSMR_TRACE").value_or(false);
    auto trace_fallback = [&](const std::string& reason,
                              int n,
                              int sample_n,
                              int sample_levels,
                              double rho,
                              double threshold) {
        if (!trace) {
            return;
        }
        std::cerr << "auto_mlsmr"
                  << " n=" << n
                  << " sample_n=" << sample_n
                  << " sample_levels=" << sample_levels
                  << " rhs=" << rhs_count
                  << " fallback=" << method_name(sweep_fallback)
                  << " rho=" << rho
                  << " threshold=" << threshold
                  << " reason=" << reason
                  << " selected=" << method_name(sweep_fallback)
                  << '\n';
    };
    if (fes.size() < 2 || options.retain_fixed_effects || options.save_groupvar ||
        options.use_sparse_solver || options.use_krylov) {
        trace_fallback("ineligible_options", static_cast<int>(fes.empty() ? 0 : fes[0].size()),
                       0, 0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN());
        return sweep_fallback;
    }
    const int n = static_cast<int>(fes.empty() ? 0 : fes[0].size());
    const int min_rows =
        read_env_int("XHDFE_AUTO_MLSMR_MIN_ROWS", 200000, 1);
    const int min_rhs =
        read_env_int("XHDFE_AUTO_MLSMR_MIN_RHS", 2, 1);
    if (n < min_rows || rhs_count < min_rhs) {
        trace_fallback(n < min_rows ? "too_few_rows" : "too_few_rhs",
                       n, 0, 0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN());
        return sweep_fallback;
    }

    const int probe_rows =
        read_env_int("XHDFE_AUTO_MLSMR_PROBE_ROWS", 200000, 1000);
    const int probe_sweeps =
        read_env_int("XHDFE_AUTO_MLSMR_PROBE_SWEEPS", 4, 2);
    const int min_levels =
        read_env_int("XHDFE_AUTO_MLSMR_MIN_LEVELS", 4096, 1);
    const bool parity_grade_tolerance =
        options.tolerance_mode == ToleranceMode::ReghdfeComparable ||
        options.tolerance_mode == ToleranceMode::StrictResidual;
    const double default_rho_threshold = parity_grade_tolerance ? 0.645 : 0.80;
    const double rho_threshold =
        read_env_double("XHDFE_AUTO_MLSMR_RHO_THRESHOLD", default_rho_threshold, 0.0, 1.0);
    const bool fast_tolerance =
        options.tolerance_mode == ToleranceMode::XhdfeFast;
    const int fast_min_rows =
        read_env_int("XHDFE_AUTO_MLSMR_FAST_MIN_ROWS", 1000000, 1);
    const int fast_min_rhs =
        read_env_int("XHDFE_AUTO_MLSMR_FAST_MIN_RHS", 8, 1);
    const int fast_min_fes =
        read_env_int("XHDFE_AUTO_MLSMR_FAST_MIN_FES", 3, 1);
    const double fast_rho_min =
        read_env_double("XHDFE_AUTO_MLSMR_FAST_RHO_MIN", 0.60, 0.0, 1.0);
    const double fast_rho_max =
        read_env_double("XHDFE_AUTO_MLSMR_FAST_RHO_MAX", 0.675, 0.0, 1.0);
    const int parity_small_n_rows =
        read_env_int("XHDFE_AUTO_MLSMR_PARITY_SMALL_N_ROWS", 1000000, 1);
    const int parity_small_n_min_rhs =
        read_env_int("XHDFE_AUTO_MLSMR_PARITY_SMALL_N_MIN_RHS", 4, 1);
    const double parity_small_n_rho_threshold =
        read_env_double("XHDFE_AUTO_MLSMR_PARITY_SMALL_N_RHO_THRESHOLD", 0.735, 0.0, 1.0);

    const AutoMlsmrProbeSample sample =
        build_auto_mlsmr_probe_sample(fes, weights, probe_rows);
    if (sample.n <= 0 || sample.total_levels < min_levels) {
        trace_fallback(sample.n <= 0 ? "empty_sample" : "too_few_levels",
                       n, sample.n, sample.total_levels,
                       std::numeric_limits<double>::quiet_NaN(), rho_threshold);
        return sweep_fallback;
    }

    Eigen::VectorXd probe = make_auto_mlsmr_probe_vector(sample);
    apply_probe_sweep(probe, sample, sweep_fallback);
    const double first_violation = probe_fe_mean_norm(probe, sample);
    if (!(first_violation > 0.0) || !std::isfinite(first_violation)) {
        trace_fallback("degenerate_first_violation", n, sample.n, sample.total_levels,
                       std::numeric_limits<double>::quiet_NaN(), rho_threshold);
        return sweep_fallback;
    }
    for (int sweep = 1; sweep < probe_sweeps; ++sweep) {
        apply_probe_sweep(probe, sample, sweep_fallback);
    }
    const double last_violation = probe_fe_mean_norm(probe, sample);
    if (!(last_violation >= 0.0) || !std::isfinite(last_violation)) {
        trace_fallback("invalid_last_violation", n, sample.n, sample.total_levels,
                       std::numeric_limits<double>::quiet_NaN(), rho_threshold);
        return sweep_fallback;
    }
    const double ratio = last_violation / std::max(first_violation, 1.0e-300);
    const double per_sweep_rho =
        std::pow(std::max(0.0, ratio), 1.0 / static_cast<double>(probe_sweeps - 1));
    // Fast mode keeps the conservative rho threshold, but lets large,
    // many-RHS, multi-way designs enter the moderate-rho band where MLSMR won
    // the base-DGP difficult 1M/10M runs without catching Section 5 fast.
    const bool fast_large_rhs_band =
        fast_tolerance &&
        n >= fast_min_rows &&
        rhs_count >= fast_min_rhs &&
        static_cast<int>(fes.size()) >= fast_min_fes &&
        per_sweep_rho >= fast_rho_min &&
        per_sweep_rho <= fast_rho_max;
    const bool parity_small_low_rhs =
        parity_grade_tolerance &&
        n < parity_small_n_rows &&
        rhs_count < parity_small_n_min_rhs;
    const double effective_rho_threshold =
        parity_small_low_rhs
            ? std::max(rho_threshold, parity_small_n_rho_threshold)
            : rho_threshold;
    const bool choose_mlsmr = per_sweep_rho >= effective_rho_threshold || fast_large_rhs_band;
    if (trace) {
        std::cerr << "auto_mlsmr"
                  << " n=" << n
                  << " sample_n=" << sample.n
                  << " sample_levels=" << sample.total_levels
                  << " rhs=" << rhs_count
                  << " fallback=" << method_name(sweep_fallback)
                  << " rho=" << per_sweep_rho
                  << " threshold=" << rho_threshold
                  << " effective_threshold=" << effective_rho_threshold
                  << " tolerance_mode="
                  << (parity_grade_tolerance ? "parity" : "xhdfe-fast")
                  << " fast_large_rhs_band=" << (fast_large_rhs_band ? 1 : 0)
                  << " parity_small_low_rhs=" << (parity_small_low_rhs ? 1 : 0)
                  << " selected=" << (choose_mlsmr ? "mlsmr" : method_name(sweep_fallback))
                  << '\n';
    }
    return choose_mlsmr ? AbsorptionMethod::Mlsmr : sweep_fallback;
}

AbsorptionMethod auto_mlsmr_sweep_fallback(std::size_t num_fes,
                                           bool symmetric_sweep) {
    if (num_fes <= 1) {
        return AbsorptionMethod::GaussSeidel;
    }
    if (symmetric_sweep) {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    return AbsorptionMethod::GaussSeidel;
}

std::optional<AbsorptionMethod> parse_method_hint(const std::string& raw) {
    const std::string name = to_lower_ascii(raw);
    if (name == "gauss-seidel" || name == "gauss_seidel" || name == "gs") {
        return AbsorptionMethod::GaussSeidel;
    }
    if (name == "symmetric-gauss-seidel" || name == "symmetric_gauss_seidel" ||
        name == "sym" || name == "symgs") {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    if (name == "jacobi") {
        return AbsorptionMethod::Jacobi;
    }
    if (name == "auto") {
        return AbsorptionMethod::Auto;
    }
    return std::nullopt;
}

bool parse_uint64(const std::string& raw, std::uint64_t* out) {
    if (!out) {
        return false;
    }
    try {
        std::size_t idx = 0;
        const unsigned long long val = std::stoull(raw, &idx, 0);
        if (idx != raw.size()) {
            return false;
        }
        *out = static_cast<std::uint64_t>(val);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_int(const std::string& raw, int* out) {
    if (!out) {
        return false;
    }
    try {
        std::size_t idx = 0;
        const int val = std::stoi(raw, &idx);
        if (idx != raw.size()) {
            return false;
        }
        *out = val;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& raw, double* out) {
    if (!out) {
        return false;
    }
    try {
        std::size_t idx = 0;
        const double val = std::stod(raw, &idx);
        if (idx != raw.size()) {
            return false;
        }
        *out = val;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_int_list(const std::string& raw, std::vector<int>& out) {
    out.clear();
    std::string token;
    auto flush = [&]() -> bool {
        if (token.empty()) {
            return true;
        }
        try {
            std::size_t idx = 0;
            const int val = std::stoi(token, &idx);
            if (idx != token.size()) {
                out.clear();
                return false;
            }
            out.push_back(val);
            token.clear();
            return true;
        } catch (...) {
            out.clear();
            return false;
        }
    };
    for (char c : raw) {
        if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            if (!flush()) {
                return false;
            }
        } else {
            token.push_back(c);
        }
    }
    return flush();
}

std::optional<AbsorptionMethod> select_two_fe_auto_fastpath(
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    bool gpu_requested,
    bool has_instruments) {
    (void)gpu_requested;
    if (read_env_bool("XHDFE_AUTO_2FE_SHAPE").value_or(true) == false) {
        return std::nullopt;
    }
    if (fes.size() != 2 || weights != nullptr || has_instruments ||
        options.retain_fixed_effects || options.save_groupvar ||
        options.absorption_method != AbsorptionMethod::Auto ||
        options.symmetric_sweep || options.use_sparse_solver || options.use_krylov) {
        return std::nullopt;
    }

    const int n = static_cast<int>(fes[0].size());
    // Small two-FE panels benchmark faster with the single forward sweep path and
    // retain coefficient/SE parity on the Sergio suite.
    if (n < 50000) {
        return AbsorptionMethod::GaussSeidel;
    }
    // Larger 2-FE panels were benchmark-sensitive; keep them on the established
    // symmetric auto path unless the user explicitly requests another method.
    return std::nullopt;
}

MobilityProfileConfig load_mobility_profile_config() {
    MobilityProfileConfig cfg;
    const char* env_path = std::getenv("XHDFE_MOBILITY_PROFILE");
    const char* env_mode = std::getenv("XHDFE_MOBILITY_MODE");
    const bool have_env_path = (env_path && *env_path != '\0');
    const bool have_env_mode = (env_mode && *env_mode != '\0');
    if (have_env_path) {
        cfg.path = env_path;
    } else {
        cfg.path = kMobilityProfileDefaultPath;
    }
    cfg.allow_auto_write = have_env_path || have_env_mode;

    if (have_env_mode) {
        cfg.mode = to_lower_ascii(env_mode);
    } else if (have_env_path) {
        cfg.mode = "auto";
    } else if (file_exists(cfg.path)) {
        cfg.mode = "auto";
    } else {
        cfg.mode = "off";
    }

    if (cfg.mode != "off" && cfg.mode != "auto" &&
        cfg.mode != "read" && cfg.mode != "write") {
        cfg.mode = "off";
    }
    if (cfg.path.empty()) {
        cfg.path = kMobilityProfileDefaultPath;
    }
    return cfg;
}

AbsorptionCacheConfig load_absorption_cache_config() {
    AbsorptionCacheConfig cfg;
    const char* env_path = std::getenv("XHDFE_ABSORPTION_CACHE");
    const char* env_mode = std::getenv("XHDFE_ABSORPTION_CACHE_MODE");
    const bool have_env_path = (env_path && *env_path != '\0');
    const bool have_env_mode = (env_mode && *env_mode != '\0');
    if (have_env_path) {
        cfg.path = env_path;
    }
    cfg.allow_auto_write = have_env_path || have_env_mode;
    cfg.mode_set = have_env_mode;

    if (have_env_mode) {
        cfg.mode = to_lower_ascii(env_mode);
    } else if (have_env_path) {
        cfg.mode = "auto";
    } else {
        cfg.mode = "off";
    }

    if (cfg.mode != "off" && cfg.mode != "auto" &&
        cfg.mode != "read" && cfg.mode != "write") {
        cfg.mode = "off";
    }
    return cfg;
}

FeStructureCacheConfig load_fe_structure_cache_config() {
    FeStructureCacheConfig cfg;
    const char* env_path = std::getenv("XHDFE_FE_STRUCTURE_CACHE");
    const char* env_mode = std::getenv("XHDFE_FE_STRUCTURE_MODE");
    const bool have_env_path = (env_path && *env_path != '\0');
    const bool have_env_mode = (env_mode && *env_mode != '\0');
    if (have_env_path) {
        cfg.path = env_path;
    }
    cfg.allow_auto_write = have_env_path || have_env_mode;

    if (have_env_mode) {
        cfg.mode = to_lower_ascii(env_mode);
    } else if (have_env_path) {
        cfg.mode = "auto";
    } else {
        cfg.mode = "off";
    }

    if (cfg.mode != "off" && cfg.mode != "auto" &&
        cfg.mode != "read" && cfg.mode != "write") {
        cfg.mode = "off";
    }
    if (cfg.path.empty()) {
        cfg.mode = "off";
    }
    return cfg;
}

bool load_mobility_profile(const std::string& path, MobilityProfile& profile) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    bool have_signature = false;
    bool have_signature_sample = false;
    bool have_signature_canon = false;
    bool have_signature_sample_canon = false;
    std::string line;
    while (std::getline(in, line)) {
        const std::string trimmed = trim_ascii(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (trimmed.rfind("xhdfe_mobility_profile_v", 0) == 0) {
            continue;
        }
        const std::size_t eq = trimmed.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = trim_ascii(trimmed.substr(0, eq));
        const std::string value = trim_ascii(trimmed.substr(eq + 1));
        if (key == "signature") {
            std::uint64_t sig = 0;
            if (parse_uint64(value, &sig)) {
                profile.signature = sig;
                have_signature = true;
            }
            continue;
        }
        if (key == "signature_sample") {
            std::uint64_t sig = 0;
            if (parse_uint64(value, &sig)) {
                profile.signature_sample = sig;
                profile.has_signature_sample = true;
                have_signature_sample = true;
            }
            continue;
        }
        if (key == "signature_canon") {
            std::uint64_t sig = 0;
            if (parse_uint64(value, &sig)) {
                profile.signature_canon = sig;
                profile.has_signature_canon = true;
                have_signature_canon = true;
            }
            continue;
        }
        if (key == "signature_sample_canon") {
            std::uint64_t sig = 0;
            if (parse_uint64(value, &sig)) {
                profile.signature_sample_canon = sig;
                profile.has_signature_sample_canon = true;
                have_signature_sample_canon = true;
            }
            continue;
        }
        if (key == "n_obs") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.nobs = val;
            }
            continue;
        }
        if (key == "n_obs_full") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.nobs_full = val;
            }
            continue;
        }
        if (key == "num_singletons") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.num_singletons = val;
            }
            continue;
        }
        if (key == "n_fe") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.nfe = val;
            }
            continue;
        }
        if (key == "fe_levels") {
            parse_int_list(value, profile.fe_levels);
            continue;
        }
        if (key == "num_components") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.num_components = val;
            }
            continue;
        }
        if (key == "largest_component") {
            int val = 0;
            if (parse_int(value, &val)) {
                profile.largest_component = val;
            }
            continue;
        }
        if (key == "largest_component_share") {
            double val = 0.0;
            if (parse_double(value, &val)) {
                profile.largest_component_share = val;
            }
            continue;
        }
        if (key == "mobility_class") {
            profile.mobility_class = value;
            continue;
        }
        if (key == "sweep_order") {
            parse_int_list(value, profile.sweep_order);
            continue;
        }
        if (key == "suggest_symmetric") {
            const std::string val = to_lower_ascii(value);
            profile.suggest_symmetric = (val == "1" || val == "true" || val == "yes");
            continue;
        }
        if (key == "suggest_method") {
            if (auto parsed = parse_method_hint(value)) {
                profile.suggest_method = *parsed;
            }
            continue;
        }
        if (key == "suggest_sparse") {
            const std::string val = to_lower_ascii(value);
            profile.suggest_use_sparse = (val == "1" || val == "true" || val == "yes");
            continue;
        }
        if (key == "suggest_krylov") {
            const std::string val = to_lower_ascii(value);
            profile.suggest_use_krylov = (val == "1" || val == "true" || val == "yes");
            continue;
        }
    }
    return have_signature || have_signature_sample ||
           have_signature_canon || have_signature_sample_canon;
}

bool write_mobility_profile(const std::string& path, const MobilityProfile& profile) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "xhdfe_mobility_profile_v1\n";
    out << "signature=0x" << std::hex << profile.signature << std::dec << "\n";
    if (profile.has_signature_sample) {
        out << "signature_sample=0x" << std::hex << profile.signature_sample << std::dec << "\n";
    }
    if (profile.has_signature_canon) {
        out << "signature_canon=0x" << std::hex << profile.signature_canon << std::dec << "\n";
    }
    if (profile.has_signature_sample_canon) {
        out << "signature_sample_canon=0x" << std::hex << profile.signature_sample_canon << std::dec << "\n";
    }
    out << "n_obs=" << profile.nobs << "\n";
    out << "n_obs_full=" << profile.nobs_full << "\n";
    out << "num_singletons=" << profile.num_singletons << "\n";
    out << "n_fe=" << profile.nfe << "\n";
    out << "fe_levels=";
    for (std::size_t i = 0; i < profile.fe_levels.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << profile.fe_levels[i];
    }
    out << "\n";
    out << "num_components=" << profile.num_components << "\n";
    out << "largest_component=" << profile.largest_component << "\n";
    out << "largest_component_share=" << profile.largest_component_share << "\n";
    out << "mobility_class=" << profile.mobility_class << "\n";
    out << "sweep_order=";
    for (std::size_t i = 0; i < profile.sweep_order.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << profile.sweep_order[i];
    }
    out << "\n";
    out << "suggest_symmetric=" << (profile.suggest_symmetric ? 1 : 0) << "\n";
    out << "suggest_method=" << method_name(profile.suggest_method) << "\n";
    out << "suggest_sparse=" << (profile.suggest_use_sparse ? 1 : 0) << "\n";
    out << "suggest_krylov=" << (profile.suggest_use_krylov ? 1 : 0) << "\n";
    return true;
}

std::string classify_mobility(double largest_share, int num_components) {
    if (num_components <= 1) {
        return (largest_share >= 0.95) ? "high" : "moderate";
    }
    if (largest_share >= 0.9) {
        return "high";
    }
    if (largest_share >= 0.7) {
        return "moderate";
    }
    return "low";
}

MobilityProfile compute_mobility_profile(const std::vector<Eigen::VectorXi>& fes,
                                         int nobs_full,
                                         int num_singletons,
                                         bool drop_singletons,
                                         const std::vector<int>& sweep_order_used) {
    MobilityProfile profile;
    profile.nobs = fes.empty() ? 0 : static_cast<int>(fes[0].size());
    profile.nobs_full = nobs_full;
    profile.num_singletons = num_singletons;
    profile.nfe = static_cast<int>(fes.size());
    profile.sweep_order = sweep_order_used;
    profile.signature = hash_fe_signature(fes, drop_singletons);
    profile.signature_sample = hash_fe_signature_sample(fes, drop_singletons);
    profile.has_signature_sample = true;

    std::vector<FeIndexerLite> indexers;
    indexers.reserve(fes.size());
    profile.fe_levels.reserve(fes.size());
    for (const auto& fe : fes) {
        indexers.push_back(build_indexer_lite(fe));
        profile.fe_levels.push_back(indexers.back().num_groups);
    }
    profile.signature_canon =
        hash_fe_signature_from_indexers(indexers, profile.nobs, drop_singletons);
    profile.signature_sample_canon =
        hash_fe_signature_sample_from_indexers(indexers, profile.nobs, drop_singletons);
    profile.has_signature_canon = true;
    profile.has_signature_sample_canon = true;

    if (fes.empty() || profile.nobs == 0) {
        profile.num_components = (profile.nobs > 0) ? 1 : 0;
        profile.largest_component = profile.nobs;
        profile.largest_component_share = (profile.nobs > 0) ? 1.0 : 0.0;
        profile.mobility_class = "none";
        profile.suggest_symmetric = false;
        profile.suggest_method = AbsorptionMethod::GaussSeidel;
        return profile;
    }

    if (fes.size() == 1) {
        const int groups = indexers[0].num_groups;
        std::vector<int> counts(static_cast<std::size_t>(groups), 0);
        for (int i = 0; i < profile.nobs; ++i) {
            const int g = indexers[0].group_ids[static_cast<std::size_t>(i)];
            if (g >= 0 && g < groups) {
                counts[static_cast<std::size_t>(g)] += 1;
            }
        }
        int largest = 0;
        for (int c : counts) {
            largest = std::max(largest, c);
        }
        profile.num_components = groups;
        profile.largest_component = largest;
        profile.largest_component_share =
            profile.nobs > 0 ? static_cast<double>(largest) / static_cast<double>(profile.nobs) : 0.0;
        profile.mobility_class = classify_mobility(profile.largest_component_share, profile.num_components);
        profile.suggest_symmetric = false;
        profile.suggest_method = AbsorptionMethod::GaussSeidel;
        return profile;
    }

    int total_nodes = 0;
    std::vector<int> offsets;
    offsets.reserve(indexers.size());
    for (const auto& idx : indexers) {
        offsets.push_back(total_nodes);
        total_nodes += idx.num_groups;
    }
    if (total_nodes <= 0) {
        profile.num_components = 0;
        profile.largest_component = 0;
        profile.largest_component_share = 0.0;
        profile.mobility_class = "none";
        profile.suggest_symmetric = false;
        profile.suggest_method = AbsorptionMethod::GaussSeidel;
        return profile;
    }

    UnionFind uf(total_nodes);
    for (int i = 0; i < profile.nobs; ++i) {
        const int root0 = offsets[0] + indexers[0].group_ids[static_cast<std::size_t>(i)];
        for (std::size_t d = 1; d < indexers.size(); ++d) {
            const int node = offsets[d] + indexers[d].group_ids[static_cast<std::size_t>(i)];
            uf.unite(root0, node);
        }
    }

    std::vector<int> counts(static_cast<std::size_t>(total_nodes), 0);
    for (int i = 0; i < profile.nobs; ++i) {
        const int root =
            uf.find(offsets[0] + indexers[0].group_ids[static_cast<std::size_t>(i)]);
        counts[static_cast<std::size_t>(root)] += 1;
    }

    int num_components = 0;
    int largest = 0;
    for (int c : counts) {
        if (c > 0) {
            ++num_components;
            largest = std::max(largest, c);
        }
    }

    profile.num_components = num_components;
    profile.largest_component = largest;
    profile.largest_component_share =
        profile.nobs > 0 ? static_cast<double>(largest) / static_cast<double>(profile.nobs) : 0.0;
    profile.mobility_class = classify_mobility(profile.largest_component_share, profile.num_components);
    profile.suggest_symmetric =
        (profile.nfe >= 2) && (profile.largest_component_share < 0.9 || profile.num_components > 1);
    profile.suggest_method = profile.suggest_symmetric ? AbsorptionMethod::SymmetricGaussSeidel
                                                       : AbsorptionMethod::GaussSeidel;
    return profile;
}

MobilityHint build_mobility_hint(const MobilityProfile& profile) {
    MobilityHint hint;
    hint.sweep_order = profile.sweep_order;
    if (profile.suggest_method != AbsorptionMethod::Auto) {
        hint.preferred_method = profile.suggest_method;
        hint.has_method = true;
        if (profile.suggest_method == AbsorptionMethod::SymmetricGaussSeidel) {
            hint.force_symmetric = true;
        }
    }
    hint.use_sparse = profile.suggest_use_sparse;
    hint.use_krylov = profile.suggest_use_krylov;
    return hint;
}

int count_bipartite_components(const Eigen::VectorXi& base_raw,
                               const Eigen::VectorXi& other_raw,
                               const FeLookup& base,
                               const FeLookup& other,
                               Eigen::VectorXi* edge_components) {
    if (base.num_groups <= 0 || other.num_groups <= 0) {
        return 0;
    }
    if (base_raw.size() != other_raw.size()) {
        throw std::runtime_error("Bipartite FE vectors must have the same length");
    }
    const int total_nodes = base.num_groups + other.num_groups;
    UnionFind uf(total_nodes);
    for (int i = 0; i < base_raw.size(); ++i) {
        const int a = base.index(base_raw(i));
        const int b = other.index(other_raw(i));
        uf.unite(a, base.num_groups + b);
    }

    if (!edge_components) {
        int components = 0;
        for (int node = 0; node < total_nodes; ++node) {
            if (uf.find(node) == node) {
                ++components;
            }
        }
        return components;
    }

    std::vector<int> root_to_component(static_cast<std::size_t>(total_nodes), -1);
    int components = 0;
    for (int node = 0; node < total_nodes; ++node) {
        const int root = uf.find(node);
        if (root_to_component[static_cast<std::size_t>(root)] < 0) {
            root_to_component[static_cast<std::size_t>(root)] = components++;
        }
    }

    edge_components->resize(base_raw.size());
    for (int i = 0; i < base_raw.size(); ++i) {
        const int a = base.index(base_raw(i));
        const int root = uf.find(a);
        const int comp = root_to_component[static_cast<std::size_t>(root)];
        (*edge_components)(i) = comp + 1;  // 1-based, like Stata's groupvar()
    }
    return components;
}

bool fe_is_function_of_refined_fe(const Eigen::VectorXi& coarse,
                                  const Eigen::VectorXi& refined) {
    if (coarse.size() != refined.size()) {
        return false;
    }
    const int n = coarse.size();
    if (n == 0) {
        return true;
    }

    int min_id = refined(0);
    int max_id = refined(0);
    for (int i = 1; i < n; ++i) {
        const int v = refined(i);
        min_id = std::min(min_id, v);
        max_id = std::max(max_id, v);
    }

    const long long range_ll =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    const bool dense_ok =
        min_id >= 0 && range_ll > 0 && range_ll <= kDenseRangeCap &&
        range_ll <= static_cast<long long>(n) * 2LL;
    if (dense_ok) {
        const int range = static_cast<int>(range_ll);
        std::vector<int> mapping(static_cast<std::size_t>(range), 0);
        std::vector<uint8_t> seen(static_cast<std::size_t>(range), 0);
        for (int i = 0; i < n; ++i) {
            const int idx = refined(i) - min_id;
            const int coarse_id = coarse(i);
            if (!seen[static_cast<std::size_t>(idx)]) {
                seen[static_cast<std::size_t>(idx)] = 1;
                mapping[static_cast<std::size_t>(idx)] = coarse_id;
            } else if (mapping[static_cast<std::size_t>(idx)] != coarse_id) {
                return false;
            }
        }
        return true;
    }

    std::unordered_map<int, int> mapping;
    mapping.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int refined_id = refined(i);
        const int coarse_id = coarse(i);
        auto it = mapping.find(refined_id);
        if (it == mapping.end()) {
            mapping.emplace(refined_id, coarse_id);
        } else if (it->second != coarse_id) {
            return false;
        }
    }
    return true;
}

FeDofInfo compute_fe_dof_reghdfe(const std::vector<Eigen::VectorXi>& fes,
                                 const std::vector<int>& fe_levels,
                                 DofAdjustmentMethod method,
                                 Eigen::VectorXi* groupvar,
                                 int threads) {
    FeDofInfo info;
    if (fes.empty()) {
        return info;
    }
    const int dims = static_cast<int>(fes.size());
    info.levels = fe_levels;
    info.redundant.resize(static_cast<std::size_t>(dims), 0);
    info.num_coefs.resize(static_cast<std::size_t>(dims), 0);
    info.inexact.resize(static_cast<std::size_t>(dims), 0);

    std::vector<FeLookup> lookups;
    lookups.reserve(static_cast<std::size_t>(dims));
    for (int d = 0; d < dims; ++d) {
        const int expected =
            (d < static_cast<int>(fe_levels.size())) ? fe_levels[static_cast<std::size_t>(d)] : -1;
        lookups.push_back(build_lookup_compact(fes[static_cast<std::size_t>(d)], expected));
    }
    if (info.levels.size() != static_cast<std::size_t>(dims)) {
        info.levels.assign(static_cast<std::size_t>(dims), 0);
        for (int d = 0; d < dims; ++d) {
            info.levels[static_cast<std::size_t>(d)] = lookups[static_cast<std::size_t>(d)].num_groups;
        }
    }

    if (groupvar) {
        groupvar->resize(0);
    }
    if ((method == DofAdjustmentMethod::All || method == DofAdjustmentMethod::FirstPair ||
         method == DofAdjustmentMethod::Pairwise) &&
        dims < 2 && groupvar) {
        groupvar->resize(0);
    }
    if (groupvar && method == DofAdjustmentMethod::None) {
        throw std::runtime_error("groupvar() requires dofadjustments(all/firstpair/pairwise)");
    }

    if (!groupvar && method != DofAdjustmentMethod::None && dims >= 2) {
        std::vector<uint8_t> coarsened(static_cast<std::size_t>(dims), 0);
        bool any_coarsened = false;
        for (int d = 0; d < dims; ++d) {
            const int coarse_levels = info.levels[static_cast<std::size_t>(d)];
            for (int e = 0; e < dims; ++e) {
                if (d == e) {
                    continue;
                }
                const int refined_levels = info.levels[static_cast<std::size_t>(e)];
                if (refined_levels <= coarse_levels) {
                    continue;
                }
                if (fe_is_function_of_refined_fe(fes[static_cast<std::size_t>(d)],
                                                 fes[static_cast<std::size_t>(e)])) {
                    coarsened[static_cast<std::size_t>(d)] = 1;
                    any_coarsened = true;
                    break;
                }
            }
        }

        if (any_coarsened) {
            std::vector<Eigen::VectorXi> effective_fes;
            std::vector<int> effective_levels;
            std::vector<int> effective_map;
            effective_fes.reserve(static_cast<std::size_t>(dims));
            effective_levels.reserve(static_cast<std::size_t>(dims));
            effective_map.reserve(static_cast<std::size_t>(dims));
            for (int d = 0; d < dims; ++d) {
                if (coarsened[static_cast<std::size_t>(d)]) {
                    continue;
                }
                effective_fes.push_back(fes[static_cast<std::size_t>(d)]);
                effective_levels.push_back(info.levels[static_cast<std::size_t>(d)]);
                effective_map.push_back(d);
            }

            if (!effective_fes.empty() && static_cast<int>(effective_fes.size()) < dims) {
                FeDofInfo effective =
                    compute_fe_dof_reghdfe(effective_fes, effective_levels, method, nullptr, threads);
                for (int d = 0; d < dims; ++d) {
                    if (coarsened[static_cast<std::size_t>(d)]) {
                        info.redundant[static_cast<std::size_t>(d)] =
                            info.levels[static_cast<std::size_t>(d)];
                        info.num_coefs[static_cast<std::size_t>(d)] = 0;
                        info.inexact[static_cast<std::size_t>(d)] = 0;
                    }
                }
                for (std::size_t pos = 0; pos < effective_map.size(); ++pos) {
                    const int orig = effective_map[pos];
                    info.redundant[static_cast<std::size_t>(orig)] = effective.redundant[pos];
                    info.num_coefs[static_cast<std::size_t>(orig)] = effective.num_coefs[pos];
                    info.inexact[static_cast<std::size_t>(orig)] = effective.inexact[pos];
                }

                int df_a_levels = 0;
                int df_a_exact = 0;
                for (int d = 0; d < dims; ++d) {
                    df_a_levels += info.levels[static_cast<std::size_t>(d)];
                    df_a_exact += info.num_coefs[static_cast<std::size_t>(d)];
                }
                info.df_a_levels = df_a_levels;
                info.df_a_exact = df_a_exact;
                info.df_a = df_a_exact;
                return info;
            }
        }
    }

    if (method == DofAdjustmentMethod::None) {
        for (int d = 0; d < dims; ++d) {
            info.redundant[static_cast<std::size_t>(d)] = 0;
            info.num_coefs[static_cast<std::size_t>(d)] = std::max(0, info.levels[static_cast<std::size_t>(d)]);
        }
    } else if (method == DofAdjustmentMethod::FirstPair) {
        info.redundant[0] = 0;
        info.num_coefs[0] = std::max(0, info.levels[0]);
        if (dims >= 2) {
            Eigen::VectorXi tmp;
            Eigen::VectorXi* out_ptr = groupvar ? &tmp : nullptr;
            const int mobility_groups =
                count_bipartite_components(fes[0], fes[1], lookups[0], lookups[1], out_ptr);
            info.redundant[1] = mobility_groups;
            info.num_coefs[1] = std::max(0, info.levels[1] - mobility_groups);
            if (groupvar) {
                *groupvar = std::move(tmp);
            }
        }
        for (int d = 2; d < dims; ++d) {
            info.redundant[static_cast<std::size_t>(d)] = 0;
            info.num_coefs[static_cast<std::size_t>(d)] =
                std::max(0, info.levels[static_cast<std::size_t>(d)]);
        }
    } else {
        // Default: pairwise mobility-group checks (reghdfe "all" and "pairwise").
        info.redundant[0] = 0;
        info.num_coefs[0] = std::max(0, info.levels[0]);

        std::vector<std::vector<int>> pair_groups(
            static_cast<std::size_t>(dims), std::vector<int>(static_cast<std::size_t>(dims), 0));
        if (dims >= 2 && groupvar) {
            Eigen::VectorXi tmp;
            const int groups =
                count_bipartite_components(fes[0], fes[1], lookups[0], lookups[1], &tmp);
            pair_groups[0][1] = groups;
            *groupvar = std::move(tmp);
        }

        struct PairIdx {
            int a = 0;
            int b = 0;
        };
        std::vector<PairIdx> pairs;
        pairs.reserve(static_cast<std::size_t>(dims * (dims - 1) / 2));
        for (int a = 0; a < dims; ++a) {
            for (int b = a + 1; b < dims; ++b) {
                if (groupvar && a == 0 && b == 1) {
                    continue;
                }
                pairs.push_back({a, b});
            }
        }

        int use_threads = 1;
#ifdef HDFE_USE_OPENMP
        use_threads = std::max(1, threads);
#endif
        if (!pairs.empty() && use_threads > 1) {
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(use_threads)
#endif
            for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
                const int a = pairs[idx].a;
                const int b = pairs[idx].b;
                const int groups = count_bipartite_components(
                    fes[static_cast<std::size_t>(a)],
                    fes[static_cast<std::size_t>(b)],
                    lookups[static_cast<std::size_t>(a)],
                    lookups[static_cast<std::size_t>(b)],
                    nullptr);
                pair_groups[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = groups;
            }
        } else {
            for (const auto& pair : pairs) {
                const int groups = count_bipartite_components(
                    fes[static_cast<std::size_t>(pair.a)],
                    fes[static_cast<std::size_t>(pair.b)],
                    lookups[static_cast<std::size_t>(pair.a)],
                    lookups[static_cast<std::size_t>(pair.b)],
                    nullptr);
                pair_groups[static_cast<std::size_t>(pair.a)][static_cast<std::size_t>(pair.b)] = groups;
            }
        }

        for (int d = 1; d < dims; ++d) {
            int max_groups = 0;
            for (int j = 0; j < d; ++j) {
                const int groups = pair_groups[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
                max_groups = std::max(max_groups, groups);
            }
            info.redundant[static_cast<std::size_t>(d)] = max_groups;
            info.num_coefs[static_cast<std::size_t>(d)] =
                std::max(0, info.levels[static_cast<std::size_t>(d)] - max_groups);
        }
    }

    int df_a_levels = 0;
    int df_a_exact = 0;
    for (int d = 0; d < dims; ++d) {
        df_a_levels += info.levels[static_cast<std::size_t>(d)];
        df_a_exact += info.num_coefs[static_cast<std::size_t>(d)];
    }
    info.df_a_levels = df_a_levels;
    info.df_a_exact = df_a_exact;
    info.df_a = df_a_exact;
    return info;
}

std::vector<uint8_t> compute_keep_mask_drop_singletons(const std::vector<Eigen::VectorXi>& fes,
                                                       int max_iter,
                                                       int* dropped_out,
                                                       int num_threads_hint = 0) {
    if (dropped_out) {
        *dropped_out = 0;
    }
    if (fes.empty()) {
        return {};
    }
    const int n = static_cast<int>(fes[0].size());
    for (const auto& fe : fes) {
        if (fe.size() != n) {
            throw std::runtime_error("Fixed-effect vectors must have the same length");
        }
    }
    if (n == 0) {
        return {};
    }

    // Threshold below which the original serial scan is kept: the parallel
    // path only pays off for large inputs, and small runs must not regress.
    constexpr int kParallelScanMinObs = 4194304;
    int scan_threads = 1;
#ifdef HDFE_USE_OPENMP
    scan_threads = num_threads_hint > 0
                       ? std::min(num_threads_hint, omp_get_max_threads())
                       : omp_get_max_threads();
    scan_threads = std::max(1, scan_threads);
#else
    (void)num_threads_hint;
#endif

    std::vector<FeIndexerLite> indexers(fes.size());
#ifdef HDFE_USE_OPENMP
    if (scan_threads > 1 && fes.size() > 1 && n >= kParallelScanMinObs) {
        // Each dimension's dense-rank mapping is independent; building them
        // concurrently keeps the per-dimension first-appearance order (and
        // therefore the result) bit-identical to the serial build.
        const int dim_threads =
            std::min<int>(scan_threads, static_cast<int>(fes.size()));
#pragma omp parallel for schedule(dynamic, 1) num_threads(dim_threads)
        for (std::ptrdiff_t d = 0; d < static_cast<std::ptrdiff_t>(fes.size()); ++d) {
            indexers[static_cast<std::size_t>(d)] =
                build_indexer_lite(fes[static_cast<std::size_t>(d)]);
        }
    } else
#endif
    {
        for (std::size_t d = 0; d < fes.size(); ++d) {
            indexers[d] = build_indexer_lite(fes[d]);
        }
    }

    std::vector<std::vector<int>> counts(indexers.size());
    std::vector<std::vector<int>> active_xor(indexers.size());
    for (std::size_t d = 0; d < indexers.size(); ++d) {
        const int groups = indexers[d].num_groups;
        counts[d].assign(static_cast<std::size_t>(groups), 0);
        active_xor[d].assign(static_cast<std::size_t>(groups), 0);
    }

#ifdef HDFE_USE_OPENMP
    if (scan_threads > 1 && n >= kParallelScanMinObs) {
        // Integer add and xor are associative and commutative, so any
        // aggregation order produces bit-identical counts/xor values. These
        // are transient scratch arrays, not solver state. Low-cardinality
        // dimensions must NOT use atomics (every thread would hammer the same
        // few cache lines); they get per-thread private arrays merged
        // serially. High-cardinality dimensions scatter over enough distinct
        // addresses that atomic contention is negligible.
        constexpr int kAtomicMinGroups = 65536;
        for (std::size_t d = 0; d < indexers.size(); ++d) {
            const int groups = indexers[d].num_groups;
            const int* __restrict gid = indexers[d].group_ids.data();
            int* __restrict cnt = counts[d].data();
            int* __restrict axr = active_xor[d].data();
            if (groups >= kAtomicMinGroups) {
#pragma omp parallel for schedule(static) num_threads(scan_threads)
                for (int i = 0; i < n; ++i) {
                    const int g = gid[i];
#pragma omp atomic
                    cnt[g] += 1;
#pragma omp atomic
                    axr[g] ^= i;
                }
            } else {
                std::vector<std::vector<int>> cnt_tls(
                    static_cast<std::size_t>(scan_threads));
                std::vector<std::vector<int>> axr_tls(
                    static_cast<std::size_t>(scan_threads));
#pragma omp parallel num_threads(scan_threads)
                {
                    const int tid = omp_get_thread_num();
                    std::vector<int>& c = cnt_tls[static_cast<std::size_t>(tid)];
                    std::vector<int>& a = axr_tls[static_cast<std::size_t>(tid)];
                    c.assign(static_cast<std::size_t>(groups), 0);
                    a.assign(static_cast<std::size_t>(groups), 0);
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        const int g = gid[i];
                        c[static_cast<std::size_t>(g)] += 1;
                        a[static_cast<std::size_t>(g)] ^= i;
                    }
                }
                for (int t = 0; t < scan_threads; ++t) {
                    const std::vector<int>& c = cnt_tls[static_cast<std::size_t>(t)];
                    const std::vector<int>& a = axr_tls[static_cast<std::size_t>(t)];
                    if (c.empty()) {
                        continue;
                    }
                    for (int g = 0; g < groups; ++g) {
                        cnt[g] += c[static_cast<std::size_t>(g)];
                        axr[g] ^= a[static_cast<std::size_t>(g)];
                    }
                }
            }
        }
    } else
#endif
    {
        for (int i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < indexers.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                counts[d][static_cast<std::size_t>(g)] += 1;
                active_xor[d][static_cast<std::size_t>(g)] ^= i;
            }
        }
    }

    std::vector<int> current;
    std::vector<int> next;
    std::vector<int> to_drop;
    current.reserve(static_cast<std::size_t>(n / 32 + 1024));
    for (std::size_t d = 0; d < indexers.size(); ++d) {
        const std::vector<int>& dim_counts = counts[d];
        const std::vector<int>& dim_xor = active_xor[d];
        for (std::size_t g = 0; g < dim_counts.size(); ++g) {
            if (dim_counts[g] == 1) {
                current.push_back(dim_xor[g]);
            }
        }
    }

    std::vector<uint8_t> keep(static_cast<std::size_t>(n), 1);
    int dropped_total = 0;
    const int iters = std::max(1, max_iter);
    for (int iter = 0; iter < iters; ++iter) {
        if (current.empty()) {
            break;
        }
        to_drop.clear();
        to_drop.reserve(current.size());
        for (const int i : current) {
            if (!keep[static_cast<std::size_t>(i)]) {
                continue;
            }
            bool singleton = false;
            for (std::size_t d = 0; d < fes.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                if (counts[d][static_cast<std::size_t>(g)] == 1) {
                    singleton = true;
                    break;
                }
            }
            if (singleton) {
                keep[static_cast<std::size_t>(i)] = 0;
                to_drop.push_back(i);
            }
        }

        if (to_drop.empty()) {
            break;
        }
        dropped_total += static_cast<int>(to_drop.size());

        for (const int i : to_drop) {
            for (std::size_t d = 0; d < fes.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                int& count = counts[d][static_cast<std::size_t>(g)];
                count -= 1;
                active_xor[d][static_cast<std::size_t>(g)] ^= i;
            }
        }

        next.clear();
        next.reserve(to_drop.size());
        for (const int i : to_drop) {
            for (std::size_t d = 0; d < fes.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                if (counts[d][static_cast<std::size_t>(g)] == 1) {
                    next.push_back(active_xor[d][static_cast<std::size_t>(g)]);
                }
            }
        }
        current.swap(next);
    }

    if (dropped_out) {
        *dropped_out = dropped_total;
    }
    return keep;
}

std::vector<uint8_t> compute_keep_mask_drop_singletons_fweights(
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    int max_iter,
    int* dropped_rows_out,
    double* dropped_weighted_out) {
    if (dropped_rows_out) {
        *dropped_rows_out = 0;
    }
    if (dropped_weighted_out) {
        *dropped_weighted_out = 0.0;
    }
    if (fes.empty()) {
        return {};
    }
    const int n = static_cast<int>(fes[0].size());
    for (const auto& fe : fes) {
        if (fe.size() != n) {
            throw std::runtime_error("Fixed-effect vectors must have the same length");
        }
    }
    if (weights.size() != n) {
        throw std::runtime_error("Weights must have the same length as fixed-effect vectors");
    }
    if (n == 0) {
        return {};
    }

    std::vector<FeIndexerLite> indexers;
    indexers.reserve(fes.size());
    for (const auto& fe : fes) {
        indexers.push_back(build_indexer_lite(fe));
    }

    std::vector<std::vector<std::int64_t>> counts(indexers.size());
    std::vector<std::vector<int>> stamps(indexers.size());
    for (std::size_t d = 0; d < indexers.size(); ++d) {
        const int groups = indexers[d].num_groups;
        counts[d].assign(static_cast<std::size_t>(groups), 0);
        stamps[d].assign(static_cast<std::size_t>(groups), 0);
    }

    std::vector<uint8_t> keep(static_cast<std::size_t>(n), 1);
    int dropped_rows_total = 0;
    double dropped_w_total = 0.0;
    const int iters = std::max(1, max_iter);
    int stamp = 1;
    for (int iter = 0; iter < iters; ++iter) {
        if (stamp == std::numeric_limits<int>::max()) {
            for (auto& s : stamps) {
                std::fill(s.begin(), s.end(), 0);
            }
            stamp = 1;
        }
        ++stamp;

        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<std::size_t>(i)]) {
                continue;
            }
            const double w_raw = weights(i);
            // Stata fweights are strictly positive integers. Treat non-finite or non-positive
            // weights as invalid here rather than silently miscounting.
            if (!(w_raw > 0.0) || !std::isfinite(w_raw)) {
                throw std::runtime_error("Invalid fweight encountered (must be positive and finite)");
            }
            const std::int64_t w = static_cast<std::int64_t>(std::llround(w_raw));
            if (w <= 0) {
                throw std::runtime_error("Invalid fweight encountered (must be >= 1)");
            }
            for (std::size_t d = 0; d < fes.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                int& seen = stamps[d][static_cast<std::size_t>(g)];
                std::int64_t& count = counts[d][static_cast<std::size_t>(g)];
                if (seen != stamp) {
                    seen = stamp;
                    count = w;
                } else {
                    count += w;
                }
            }
        }

        int dropped_rows_iter = 0;
        double dropped_w_iter = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!keep[static_cast<std::size_t>(i)]) {
                continue;
            }
            bool singleton = false;
            for (std::size_t d = 0; d < fes.size(); ++d) {
                const int g = indexers[d].group_ids[static_cast<std::size_t>(i)];
                if (counts[d][static_cast<std::size_t>(g)] == 1) {
                    singleton = true;
                    break;
                }
            }
            if (singleton) {
                keep[static_cast<std::size_t>(i)] = 0;
                ++dropped_rows_iter;
                dropped_w_iter += weights(i);
            }
        }
        dropped_rows_total += dropped_rows_iter;
        dropped_w_total += dropped_w_iter;
        if (dropped_rows_iter == 0) {
            break;
        }
    }

    if (dropped_rows_out) {
        *dropped_rows_out = dropped_rows_total;
    }
    if (dropped_weighted_out) {
        *dropped_weighted_out = dropped_w_total;
    }
    return keep;
}

std::vector<int> build_keep_indices(const std::vector<uint8_t>& keep) {
    std::vector<int> out;
    out.reserve(keep.size());
    for (std::size_t i = 0; i < keep.size(); ++i) {
        if (keep[i]) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

Eigen::VectorXd filter_vector(const Eigen::Ref<const Eigen::VectorXd>& v,
                              const std::vector<int>& idx) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(idx.size()));
    for (std::size_t r = 0; r < idx.size(); ++r) {
        out(static_cast<Eigen::Index>(r)) = v(idx[r]);
    }
    return out;
}

Eigen::VectorXi filter_vector(const Eigen::Ref<const Eigen::VectorXi>& v,
                              const std::vector<int>& idx) {
    Eigen::VectorXi out(static_cast<Eigen::Index>(idx.size()));
    for (std::size_t r = 0; r < idx.size(); ++r) {
        out(static_cast<Eigen::Index>(r)) = v(idx[r]);
    }
    return out;
}

Eigen::MatrixXd filter_rows(const Eigen::Ref<const Eigen::MatrixXd>& m,
                            const std::vector<int>& idx) {
    Eigen::MatrixXd out(static_cast<Eigen::Index>(idx.size()), m.cols());
    for (std::size_t r = 0; r < idx.size(); ++r) {
        out.row(static_cast<Eigen::Index>(r)) = m.row(idx[r]);
    }
    return out;
}

std::vector<detail::HeterogeneousSlopeTerm> filter_slope_terms(
    const std::vector<detail::HeterogeneousSlopeTerm>& slopes,
    const std::vector<int>& idx) {
    std::vector<detail::HeterogeneousSlopeTerm> out;
    out.reserve(slopes.size());
    for (const auto& slope : slopes) {
        detail::HeterogeneousSlopeTerm filtered;
        filtered.fe_index = slope.fe_index;
        filtered.include_intercept = slope.include_intercept;
        filtered.values = filter_vector(slope.values, idx);
        out.push_back(std::move(filtered));
    }
    return out;
}

struct MixedSavefeRecoveryResult {
    std::vector<Eigen::VectorXd> contributions;
    std::vector<Eigen::VectorXd> save_effects;
    std::vector<int> save_alpha_slot_by_dim;
    int iterations = 0;
    double max_delta = 0.0;
    bool converged = true;
};

struct MixedSavefeDim {
    FeIndexerLite indexer;
    const detail::HeterogeneousSlopeTerm* slope = nullptr;
    Eigen::VectorXd alpha;
    Eigen::VectorXd beta;
    std::vector<double> sw;
    std::vector<double> swz;
    std::vector<double> swzz;
    std::vector<double> rhs0;
    std::vector<double> rhs1;
};

std::vector<std::size_t> resolve_savefe_sweep_order(
    std::size_t dims,
    const std::vector<int>& override_order) {
    std::vector<std::size_t> order;
    order.reserve(dims);
    if (override_order.size() == dims && dims > 0) {
        std::vector<uint8_t> seen(dims, 0);
        bool valid = true;
        for (const int v : override_order) {
            if (v < 0 || v >= static_cast<int>(dims)) {
                valid = false;
                break;
            }
            const std::size_t idx = static_cast<std::size_t>(v);
            if (seen[idx]) {
                valid = false;
                break;
            }
            seen[idx] = 1;
        }
        if (valid) {
            for (const int v : override_order) {
                order.push_back(static_cast<std::size_t>(v));
            }
            return order;
        }
    }
    order.resize(dims);
    std::iota(order.begin(), order.end(), 0);
    return order;
}

MixedSavefeRecoveryResult recover_mixed_savefe_effects(
    const Eigen::VectorXd& partial,
    const std::vector<Eigen::VectorXi>& fes,
    const std::vector<detail::HeterogeneousSlopeTerm>& slopes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options) {
    const int n = static_cast<int>(partial.size());
    if (n == 0) {
        throw std::runtime_error("Partial residual vector must be non-empty");
    }
    const std::size_t dims = fes.size();
    MixedSavefeRecoveryResult result;
    result.contributions.resize(dims);
    result.save_alpha_slot_by_dim.assign(dims, -1);
    if (dims == 0) {
        result.converged = true;
        return result;
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("Weights must have the same length as the outcome");
    }

    std::vector<const detail::HeterogeneousSlopeTerm*> slope_by_dim(dims, nullptr);
    for (const auto& slope : slopes) {
        if (slope.fe_index < 0 || slope.fe_index >= static_cast<int>(dims)) {
            throw std::runtime_error("Heterogeneous slope FE index out of range");
        }
        const std::size_t dim = static_cast<std::size_t>(slope.fe_index);
        if (slope_by_dim[dim] != nullptr) {
            throw std::runtime_error("Only one heterogeneous slope per absorbed FE is supported");
        }
        if (slope.values.size() != n) {
            throw std::runtime_error("Heterogeneous slope variable must match FE length");
        }
        slope_by_dim[dim] = &slope;
    }

    std::vector<MixedSavefeDim> work;
    work.resize(dims);
    {
        SavefeTimer timer("savefe_mixed_recover_setup");
        for (std::size_t dim = 0; dim < dims; ++dim) {
            if (fes[dim].size() != n) {
                throw std::runtime_error("Each fixed-effect vector must match the length of y");
            }
            MixedSavefeDim& wdim = work[dim];
            wdim.indexer = build_indexer_lite(fes[dim]);
            wdim.slope = slope_by_dim[dim];
            const int groups = wdim.indexer.num_groups;
            wdim.alpha = Eigen::VectorXd::Zero(groups);
            if (wdim.slope != nullptr) {
                wdim.beta = Eigen::VectorXd::Zero(groups);
            }
            wdim.sw.assign(static_cast<std::size_t>(groups), 0.0);
            wdim.rhs0.assign(static_cast<std::size_t>(groups), 0.0);
            if (wdim.slope != nullptr) {
                wdim.swz.assign(static_cast<std::size_t>(groups), 0.0);
                wdim.swzz.assign(static_cast<std::size_t>(groups), 0.0);
                wdim.rhs1.assign(static_cast<std::size_t>(groups), 0.0);
            }
            for (int i = 0; i < n; ++i) {
                const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
                const double wi = weights ? (*weights)(i) : 1.0;
                wdim.sw[static_cast<std::size_t>(g)] += wi;
                if (wdim.slope != nullptr) {
                    const double z = wdim.slope->values(i);
                    wdim.swz[static_cast<std::size_t>(g)] += wi * z;
                    wdim.swzz[static_cast<std::size_t>(g)] += wi * z * z;
                }
            }
        }
    }

    const std::vector<std::size_t> order =
        resolve_savefe_sweep_order(dims, options.sweep_order_override);
    const bool use_symmetric = options.symmetric_sweep && order.size() > 1;
    const double fe_tol = options.fe_tolerance > 0.0 ? options.fe_tolerance : options.tol;
    const int max_iter = std::max(1, options.max_iter);
    Eigen::VectorXd residual = partial;
    double last_max_delta = 0.0;
    bool converged = false;

    auto run_dim = [&](std::size_t dim) -> double {
        MixedSavefeDim& wdim = work[dim];
        const int groups = wdim.indexer.num_groups;
        std::fill(wdim.rhs0.begin(), wdim.rhs0.end(), 0.0);
        if (wdim.slope != nullptr) {
            std::fill(wdim.rhs1.begin(), wdim.rhs1.end(), 0.0);
        }
        for (int i = 0; i < n; ++i) {
            const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
            const double wi = weights ? (*weights)(i) : 1.0;
            const double ri = residual(i);
            wdim.rhs0[static_cast<std::size_t>(g)] += wi * ri;
            if (wdim.slope != nullptr) {
                wdim.rhs1[static_cast<std::size_t>(g)] += wi * wdim.slope->values(i) * ri;
            }
        }

        double max_delta = 0.0;
        constexpr double kSolveTol = 1e-12;
        if (wdim.slope == nullptr) {
            for (int g = 0; g < groups; ++g) {
                const double denom = wdim.sw[static_cast<std::size_t>(g)];
                const double delta =
                    denom > 0.0 ? wdim.rhs0[static_cast<std::size_t>(g)] / denom : 0.0;
                wdim.rhs0[static_cast<std::size_t>(g)] = delta;
                wdim.alpha(g) += delta;
                max_delta = std::max(max_delta, std::abs(delta));
            }
            for (int i = 0; i < n; ++i) {
                const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
                residual(i) -= wdim.rhs0[static_cast<std::size_t>(g)];
            }
            return max_delta;
        }

        if (wdim.slope->include_intercept) {
            for (int g = 0; g < groups; ++g) {
                const std::size_t gs = static_cast<std::size_t>(g);
                const double sw = wdim.sw[gs];
                const double swz = wdim.swz[gs];
                const double swzz = wdim.swzz[gs];
                const double nr = wdim.rhs0[gs];
                const double nz = wdim.rhs1[gs];
                const double det = sw * swzz - swz * swz;
                const double scale = std::max(1.0, std::abs(sw * swzz) + std::abs(swz * swz));
                double delta_a = 0.0;
                double delta_b = 0.0;
                if (det > kSolveTol * scale) {
                    delta_a = (nr * swzz - nz * swz) / det;
                    delta_b = (sw * nz - swz * nr) / det;
                } else if (sw > 0.0) {
                    delta_a = nr / sw;
                }
                wdim.rhs0[gs] = delta_a;
                wdim.rhs1[gs] = delta_b;
                wdim.alpha(g) += delta_a;
                wdim.beta(g) += delta_b;
                max_delta = std::max(max_delta, std::abs(delta_a));
                max_delta = std::max(max_delta, std::abs(delta_b));
            }
            for (int i = 0; i < n; ++i) {
                const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
                const std::size_t gs = static_cast<std::size_t>(g);
                residual(i) -= wdim.rhs0[gs] + wdim.rhs1[gs] * wdim.slope->values(i);
            }
            return max_delta;
        }

        for (int g = 0; g < groups; ++g) {
            const std::size_t gs = static_cast<std::size_t>(g);
            const double denom = wdim.swzz[gs];
            const double scale = std::max(1.0, std::abs(denom));
            const double delta_b = denom > kSolveTol * scale ? wdim.rhs1[gs] / denom : 0.0;
            wdim.rhs1[gs] = delta_b;
            wdim.beta(g) += delta_b;
            max_delta = std::max(max_delta, std::abs(delta_b));
        }
        for (int i = 0; i < n; ++i) {
            const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
            residual(i) -= wdim.rhs1[static_cast<std::size_t>(g)] * wdim.slope->values(i);
        }
        return max_delta;
    };

    {
        SavefeTimer timer("savefe_mixed_recover_loop");
        for (int iter = 0; iter < max_iter; ++iter) {
            double max_delta = 0.0;
            for (const std::size_t dim : order) {
                max_delta = std::max(max_delta, run_dim(dim));
            }
            if (use_symmetric) {
                for (std::size_t pos = order.size(); pos-- > 0;) {
                    max_delta = std::max(max_delta, run_dim(order[pos]));
                }
            }
            last_max_delta = max_delta;
            if (fe_tol <= 0.0 || max_delta <= fe_tol) {
                result.iterations = iter + 1;
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
        result.iterations = max_iter;
    }
    result.max_delta = last_max_delta;
    result.converged = converged;

    {
        SavefeTimer timer("savefe_mixed_recover_expand");
        for (std::size_t dim = 0; dim < dims; ++dim) {
            MixedSavefeDim& wdim = work[dim];
            result.contributions[dim] = Eigen::VectorXd::Zero(n);
            const bool has_slope = wdim.slope != nullptr;
            const bool has_intercept = !has_slope || wdim.slope->include_intercept;
            int alpha_slot = -1;
            int beta_slot = -1;
            if (has_intercept) {
                alpha_slot = static_cast<int>(result.save_effects.size());
                result.save_alpha_slot_by_dim[dim] = alpha_slot;
                result.save_effects.emplace_back(Eigen::VectorXd::Zero(n));
            }
            if (has_slope) {
                beta_slot = static_cast<int>(result.save_effects.size());
                result.save_effects.emplace_back(Eigen::VectorXd::Zero(n));
            }

            for (int i = 0; i < n; ++i) {
                const int g = wdim.indexer.group_ids[static_cast<std::size_t>(i)];
                double contribution = 0.0;
                if (has_intercept) {
                    const double a = wdim.alpha(g);
                    contribution += a;
                    result.save_effects[static_cast<std::size_t>(alpha_slot)](i) = a;
                }
                if (has_slope) {
                    const double b = wdim.beta(g);
                    result.save_effects[static_cast<std::size_t>(beta_slot)](i) = b;
                    contribution += b * wdim.slope->values(i);
                }
                result.contributions[dim](i) = contribution;
            }
        }
    }

    if (savefe_profile_enabled()) {
        std::ostringstream oss;
        oss << "event=mixed_fe_recovery_end iterations=" << result.iterations
            << " converged=" << (result.converged ? 1 : 0)
            << " max_delta=" << result.max_delta
            << " saved_effects=" << result.save_effects.size();
        savefe_profile_log(oss.str());
    }
    return result;
}

void center_mixed_savefe_effects(std::vector<Eigen::VectorXd>& contributions,
                                 std::vector<Eigen::VectorXd>& save_effects,
                                 const std::vector<int>& save_alpha_slot_by_dim,
                                 const Eigen::VectorXd* weights) {
    const std::size_t dims = contributions.size();
    if (save_alpha_slot_by_dim.size() != dims) {
        return;
    }
    for (std::size_t dim = 0; dim < dims; ++dim) {
        const int slot = save_alpha_slot_by_dim[dim];
        if (slot < 0 || slot >= static_cast<int>(save_effects.size())) {
            continue;
        }
        const double mean = weighted_mean(contributions[dim], weights);
        contributions[dim].array() -= mean;
        save_effects[static_cast<std::size_t>(slot)].array() -= mean;
    }
}

int heterogeneous_slope_rank(const Eigen::VectorXi& fe,
                             const detail::HeterogeneousSlopeTerm& slope,
                             const Eigen::VectorXd* weights) {
    if (fe.size() != slope.values.size()) {
        throw std::runtime_error("Heterogeneous slope variable must match FE length");
    }
    if (weights && weights->size() != fe.size()) {
        throw std::runtime_error("Weights must match heterogeneous slope length");
    }
    const FeIndexerLite idx = build_indexer_lite(fe);
    std::vector<double> sw(static_cast<std::size_t>(idx.num_groups), 0.0);
    std::vector<double> sz(static_cast<std::size_t>(idx.num_groups), 0.0);
    std::vector<double> szz(static_cast<std::size_t>(idx.num_groups), 0.0);
    for (int i = 0; i < fe.size(); ++i) {
        const int g = idx.group_ids[static_cast<std::size_t>(i)];
        const double w = weights ? (*weights)(i) : 1.0;
        const double z = slope.values(i);
        sw[static_cast<std::size_t>(g)] += w;
        sz[static_cast<std::size_t>(g)] += w * z;
        szz[static_cast<std::size_t>(g)] += w * z * z;
    }
    constexpr double kDetTol = 1e-12;
    int rank = 0;
    for (int g = 0; g < idx.num_groups; ++g) {
        const double wg = sw[static_cast<std::size_t>(g)];
        const double zg = sz[static_cast<std::size_t>(g)];
        const double zzg = szz[static_cast<std::size_t>(g)];
        if (slope.include_intercept) {
            if (wg > 0.0) {
                const double det = wg * zzg - zg * zg;
                const double scale = std::max(1.0, wg * zzg);
                rank += (det > kDetTol * scale) ? 2 : 1;
            }
        } else {
            const double scale = std::max(1.0, zzg);
            if (zzg > kDetTol * scale) {
                ++rank;
            }
        }
    }
    return rank;
}

void apply_heterogeneous_slope_dof(
    FeDofInfo& dof,
    const std::vector<Eigen::VectorXi>& fes,
    const std::vector<detail::HeterogeneousSlopeTerm>& slopes,
    const Eigen::VectorXd* weights,
    const std::vector<uint8_t>& nested_flags) {
    if (slopes.empty()) {
        return;
    }
    for (const auto& slope : slopes) {
        const int d = slope.fe_index;
        if (d < 0 || d >= static_cast<int>(fes.size()) ||
            d >= static_cast<int>(dof.levels.size())) {
            throw std::runtime_error("Heterogeneous slope FE index out of range");
        }
        const int rank = heterogeneous_slope_rank(fes[static_cast<std::size_t>(d)], slope, weights);
        const bool nested = d < static_cast<int>(nested_flags.size()) &&
                            nested_flags[static_cast<std::size_t>(d)] != 0;
        if (slope.include_intercept) {
            const int base_levels = dof.levels[static_cast<std::size_t>(d)];
            const int extra = std::max(0, rank - dof.levels[static_cast<std::size_t>(d)]);
            dof.levels[static_cast<std::size_t>(d)] += extra;
            if (nested) {
                dof.redundant[static_cast<std::size_t>(d)] = base_levels;
                dof.num_coefs[static_cast<std::size_t>(d)] = extra;
            } else {
                dof.num_coefs[static_cast<std::size_t>(d)] += extra;
            }
        } else {
            dof.levels[static_cast<std::size_t>(d)] = rank;
            dof.redundant[static_cast<std::size_t>(d)] = 0;
            dof.num_coefs[static_cast<std::size_t>(d)] = rank;
        }
    }
}

int nested_levels_for_heterogeneous_dim(
    std::size_t dim,
    const std::vector<detail::HeterogeneousSlopeTerm>& slopes,
    const std::vector<int>& base_levels,
    const std::vector<int>& dof_levels) {
    for (const auto& slope : slopes) {
        if (slope.fe_index != static_cast<int>(dim)) {
            continue;
        }
        if (!slope.include_intercept) {
            return 0;
        }
        if (dim < base_levels.size()) {
            return base_levels[dim];
        }
        return dim < dof_levels.size() ? dof_levels[dim] : 0;
    }
    return dim < dof_levels.size() ? dof_levels[dim] : 0;
}

int nested_coefs_for_heterogeneous_dim(
    std::size_t dim,
    const std::vector<detail::HeterogeneousSlopeTerm>& slopes,
    const std::vector<int>& dof_num_coefs) {
    for (const auto& slope : slopes) {
        if (slope.fe_index == static_cast<int>(dim)) {
            return 0;
        }
    }
    return dim < dof_num_coefs.size() ? dof_num_coefs[dim] : 0;
}

constexpr double kGroupConstantTol = 1e-12;

bool approx_equal(double a, double b) {
    const double scale = 1.0 + std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= kGroupConstantTol * scale;
}

int find_identical_fe(const Eigen::VectorXi& needle,
                      const std::vector<Eigen::VectorXi>& haystack) {
    for (int d = 0; d < static_cast<int>(haystack.size()); ++d) {
        if (haystack[static_cast<std::size_t>(d)].size() != needle.size()) {
            continue;
        }
        if ((haystack[static_cast<std::size_t>(d)].array() == needle.array()).all()) {
            return d;
        }
    }
    return -1;
}

std::vector<uint8_t> compute_rank_collinearity_mask(
    const Eigen::Ref<const Eigen::MatrixXd>& transformed_X,
    const std::vector<int>& kept_slope_cols,
    int slope_cols,
    bool has_intercept,
    const Eigen::VectorXd* weights,
    double rank_collinear_tol,
    int threads,
    const std::vector<int>& collinear_priority) {
    std::vector<uint8_t> drop_mask(static_cast<std::size_t>(std::max(0, slope_cols)), 0);
    if (kept_slope_cols.empty() || slope_cols <= 0) {
        return drop_mask;
    }

    std::vector<int> ordered_cols = kept_slope_cols;
    if (!collinear_priority.empty()) {
        std::stable_sort(ordered_cols.begin(), ordered_cols.end(),
                         [&](int a, int b) {
                             const int pa =
                                 (a >= 0 && a < static_cast<int>(collinear_priority.size()))
                                     ? collinear_priority[static_cast<std::size_t>(a)]
                                     : 0;
                             const int pb =
                                 (b >= 0 && b < static_cast<int>(collinear_priority.size()))
                                     ? collinear_priority[static_cast<std::size_t>(b)]
                                     : 0;
                             if (pa != pb) {
                                 return pa > pb;
                             }
                             return a < b;
                         });
    }

    const int p = static_cast<int>(ordered_cols.size());
    Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd sum_xx = Eigen::MatrixXd::Zero(p, p);
    double sum_w = 0.0;

    std::vector<const double*> col_ptrs(static_cast<std::size_t>(p));
    const Eigen::Index n = transformed_X.rows();
    const Eigen::Index stride = transformed_X.rows();
    for (int j = 0; j < p; ++j) {
        const int col = ordered_cols[static_cast<std::size_t>(j)];
        col_ptrs[static_cast<std::size_t>(j)] =
            transformed_X.data() + static_cast<Eigen::Index>(col) * stride;
    }

    int use_threads = 1;
#ifdef HDFE_USE_OPENMP
    use_threads = std::max(1, threads);
#endif
    if (use_threads > 1) {
#ifdef HDFE_USE_OPENMP
        std::vector<Eigen::VectorXd> sum_x_tls(static_cast<std::size_t>(use_threads),
                                               Eigen::VectorXd::Zero(p));
        std::vector<Eigen::MatrixXd> sum_xx_tls(static_cast<std::size_t>(use_threads),
                                                Eigen::MatrixXd::Zero(p, p));
        std::vector<double> sum_w_tls(static_cast<std::size_t>(use_threads), 0.0);
#pragma omp parallel num_threads(use_threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::VectorXd& sum_x_local = sum_x_tls[static_cast<std::size_t>(tid)];
            Eigen::MatrixXd& sum_xx_local = sum_xx_tls[static_cast<std::size_t>(tid)];
            double sum_w_local = 0.0;
            std::vector<double> xvals(static_cast<std::size_t>(p), 0.0);
#pragma omp for schedule(static)
            for (Eigen::Index i = 0; i < n; ++i) {
                const double w = weights ? weights->data()[i] : 1.0;
                if (weights) {
                    sum_w_local += w;
                }
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                    sum_x_local(j) += w * xvals[j];
                }
                for (int j = 0; j < p; ++j) {
                    for (int k = 0; k <= j; ++k) {
                        sum_xx_local(j, k) += w * xvals[j] * xvals[k];
                    }
                }
            }
            sum_w_tls[static_cast<std::size_t>(tid)] = sum_w_local;
        }

        for (int t = 0; t < use_threads; ++t) {
            sum_x.noalias() += sum_x_tls[static_cast<std::size_t>(t)];
            sum_xx.noalias() += sum_xx_tls[static_cast<std::size_t>(t)];
            sum_w += sum_w_tls[static_cast<std::size_t>(t)];
        }
#endif
    } else {
        std::vector<double> xvals(static_cast<std::size_t>(p), 0.0);
        if (weights) {
            const double* w_ptr = weights->data();
            for (Eigen::Index i = 0; i < n; ++i) {
                const double w = w_ptr[i];
                sum_w += w;
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                    sum_x(j) += w * xvals[j];
                }
                for (int j = 0; j < p; ++j) {
                    for (int k = 0; k <= j; ++k) {
                        sum_xx(j, k) += w * xvals[j] * xvals[k];
                    }
                }
            }
        } else {
            for (Eigen::Index i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                    sum_x(j) += xvals[j];
                }
                for (int j = 0; j < p; ++j) {
                    for (int k = 0; k <= j; ++k) {
                        sum_xx(j, k) += xvals[j] * xvals[k];
                    }
                }
            }
        }
    }

    if (!weights) {
        sum_w = static_cast<double>(n);
    }
    if (has_intercept && weights && !(sum_w > 0.0)) {
        throw std::runtime_error("Weights must sum to a positive value");
    }

    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd means;
    if (has_intercept) {
        if (!weights && !(sum_w > 0.0)) {
            sum_w = 1.0;
        }
        means = sum_x / sum_w;
    }
    for (int j = 0; j < p; ++j) {
        for (int k = 0; k <= j; ++k) {
            double value = sum_xx(j, k);
            if (has_intercept) {
                value -= means(j) * sum_x(k) + means(k) * sum_x(j) -
                         means(j) * means(k) * sum_w;
            }
            gram(j, k) = value;
            gram(k, j) = value;
        }
    }

    Eigen::MatrixXd gram_keep(0, 0);
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    std::vector<int> kept_positions;
    kept_positions.reserve(static_cast<std::size_t>(p));

    for (int pos = 0; pos < p; ++pos) {
        const int col_idx = ordered_cols[static_cast<std::size_t>(pos)];
        const double base_var = gram(pos, pos);
        if (!std::isfinite(base_var) || base_var <= 0.0) {
            drop_mask[static_cast<std::size_t>(col_idx)] = 1;
            continue;
        }

        double resid_var = base_var;
        if (!kept_positions.empty()) {
            Eigen::VectorXd g(kept_positions.size());
            for (std::size_t i = 0; i < kept_positions.size(); ++i) {
                g(static_cast<Eigen::Index>(i)) = gram(kept_positions[i], pos);
            }
            const Eigen::VectorXd alpha = ldlt.solve(g);
            resid_var = base_var - g.dot(alpha);
        }
        if (!std::isfinite(resid_var)) {
            drop_mask[static_cast<std::size_t>(col_idx)] = 1;
            continue;
        }
        if (resid_var < 0.0) {
            resid_var = 0.0;
        }
        const double base_norm = std::sqrt(base_var);
        const double resid_norm = std::sqrt(resid_var);
        if (!(base_norm > 0.0) || resid_norm <= rank_collinear_tol * base_norm) {
            drop_mask[static_cast<std::size_t>(col_idx)] = 1;
            continue;
        }

        const int old = gram_keep.rows();
        gram_keep.conservativeResize(old + 1, old + 1);
        for (int i = 0; i < old; ++i) {
            const double v = gram(kept_positions[static_cast<std::size_t>(i)], pos);
            gram_keep(i, old) = v;
            gram_keep(old, i) = v;
        }
        gram_keep(old, old) = base_var;
        kept_positions.push_back(pos);
        ldlt.compute(gram_keep);
        if (ldlt.info() != Eigen::Success) {
            drop_mask[static_cast<std::size_t>(col_idx)] = 1;
            kept_positions.pop_back();
            gram_keep.conservativeResize(old, old);
            if (old > 0) {
                ldlt.compute(gram_keep);
            }
        }
    }

    return drop_mask;
}

struct CrossproductResult {
    Eigen::MatrixXd xtx;
    Eigen::VectorXd xty;
    Eigen::VectorXd sum_x;
    double sum_w = 0.0;
};

CrossproductResult compute_crossproducts_selected(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<int>& slope_cols,
    int intercept_col,
    const Eigen::VectorXd* weights,
    bool center_slopes,
    int threads) {
    CrossproductResult out;
    std::vector<int> cols = slope_cols;
    if (intercept_col >= 0) {
        cols.push_back(intercept_col);
    }
    const int p = static_cast<int>(cols.size());
    const int k = static_cast<int>(slope_cols.size());
    out.xtx = Eigen::MatrixXd::Zero(p, p);
    out.xty = Eigen::VectorXd::Zero(p);
    if (center_slopes) {
        out.sum_x = Eigen::VectorXd::Zero(k);
    }
    out.sum_w = 0.0;
    if (p == 0) {
        return out;
    }

    std::vector<const double*> col_ptrs(static_cast<std::size_t>(p));
    const Eigen::Index n = X.rows();
    const Eigen::Index stride = X.rows();
    for (int j = 0; j < p; ++j) {
        col_ptrs[static_cast<std::size_t>(j)] =
            X.data() + static_cast<Eigen::Index>(cols[static_cast<std::size_t>(j)]) * stride;
    }

    int use_threads = 1;
#ifdef HDFE_USE_OPENMP
    use_threads = std::max(1, threads);
#endif
    if (use_threads > 1) {
#ifdef HDFE_USE_OPENMP
        std::vector<Eigen::MatrixXd> xtx_tls(static_cast<std::size_t>(use_threads),
                                             Eigen::MatrixXd::Zero(p, p));
        std::vector<Eigen::VectorXd> xty_tls(static_cast<std::size_t>(use_threads),
                                             Eigen::VectorXd::Zero(p));
        std::vector<Eigen::VectorXd> sum_x_tls;
        if (center_slopes) {
            sum_x_tls.assign(static_cast<std::size_t>(use_threads), Eigen::VectorXd::Zero(k));
        }
        std::vector<double> sum_w_tls(static_cast<std::size_t>(use_threads), 0.0);

#pragma omp parallel num_threads(use_threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::MatrixXd& xtx_local = xtx_tls[static_cast<std::size_t>(tid)];
            Eigen::VectorXd& xty_local = xty_tls[static_cast<std::size_t>(tid)];
            Eigen::VectorXd* sum_x_local =
                center_slopes ? &sum_x_tls[static_cast<std::size_t>(tid)] : nullptr;
            double sum_w_local = 0.0;
            std::vector<double> xvals(static_cast<std::size_t>(p), 0.0);
#pragma omp for schedule(static)
            for (Eigen::Index i = 0; i < n; ++i) {
                const double w = weights ? weights->data()[i] : 1.0;
                if (center_slopes && weights) {
                    sum_w_local += w;
                }
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                }
                for (int j = 0; j < p; ++j) {
                    xty_local(j) += w * xvals[j] * y(i);
                    for (int l = 0; l <= j; ++l) {
                        xtx_local(j, l) += w * xvals[j] * xvals[l];
                    }
                }
                if (sum_x_local) {
                    for (int j = 0; j < k; ++j) {
                        (*sum_x_local)(j) += w * xvals[j];
                    }
                }
            }
            sum_w_tls[static_cast<std::size_t>(tid)] = sum_w_local;
        }

        for (int t = 0; t < use_threads; ++t) {
            out.xtx.noalias() += xtx_tls[static_cast<std::size_t>(t)];
            out.xty.noalias() += xty_tls[static_cast<std::size_t>(t)];
            if (center_slopes) {
                out.sum_x.noalias() += sum_x_tls[static_cast<std::size_t>(t)];
                out.sum_w += sum_w_tls[static_cast<std::size_t>(t)];
            }
        }
#endif
    } else {
        std::vector<double> xvals(static_cast<std::size_t>(p), 0.0);
        if (weights) {
            const double* w_ptr = weights->data();
            for (Eigen::Index i = 0; i < n; ++i) {
                const double w = w_ptr[i];
                if (center_slopes) {
                    out.sum_w += w;
                }
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                }
                for (int j = 0; j < p; ++j) {
                    out.xty(j) += w * xvals[j] * y(i);
                    for (int l = 0; l <= j; ++l) {
                        out.xtx(j, l) += w * xvals[j] * xvals[l];
                    }
                }
                if (center_slopes) {
                    for (int j = 0; j < k; ++j) {
                        out.sum_x(j) += w * xvals[j];
                    }
                }
            }
        } else {
            for (Eigen::Index i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) {
                    xvals[j] = col_ptrs[static_cast<std::size_t>(j)][i];
                }
                for (int j = 0; j < p; ++j) {
                    out.xty(j) += xvals[j] * y(i);
                    for (int l = 0; l <= j; ++l) {
                        out.xtx(j, l) += xvals[j] * xvals[l];
                    }
                }
                if (center_slopes) {
                    for (int j = 0; j < k; ++j) {
                        out.sum_x(j) += xvals[j];
                    }
                }
            }
        }
    }

    if (!weights && center_slopes) {
        out.sum_w = static_cast<double>(n);
    }
    if (center_slopes && weights && !(out.sum_w > 0.0)) {
        throw std::runtime_error("Weights must sum to a positive value");
    }
    for (int j = 0; j < p; ++j) {
        for (int l = j + 1; l < p; ++l) {
            out.xtx(j, l) = out.xtx(l, j);
        }
    }
    return out;
}

std::vector<uint8_t> compute_rank_collinearity_mask_from_gram(
    const Eigen::Ref<const Eigen::MatrixXd>& gram,
    const std::vector<int>& kept_slope_cols,
    double rank_collinear_tol,
    const std::vector<int>& collinear_priority) {
    const int p = static_cast<int>(gram.rows());
    std::vector<uint8_t> drop_mask(static_cast<std::size_t>(std::max(0, p)), 0);
    if (p <= 0) {
        return drop_mask;
    }

    std::vector<int> ordered_positions(static_cast<std::size_t>(p));
    std::iota(ordered_positions.begin(), ordered_positions.end(), 0);
    if (!collinear_priority.empty()) {
        std::stable_sort(ordered_positions.begin(), ordered_positions.end(),
                         [&](int a, int b) {
                             const int pa =
                                 (a >= 0 && a < static_cast<int>(kept_slope_cols.size()))
                                     ? kept_slope_cols[static_cast<std::size_t>(a)]
                                     : -1;
                             const int pb =
                                 (b >= 0 && b < static_cast<int>(kept_slope_cols.size()))
                                     ? kept_slope_cols[static_cast<std::size_t>(b)]
                                     : -1;
                             const int pri_a =
                                 (pa >= 0 && pa < static_cast<int>(collinear_priority.size()))
                                     ? collinear_priority[static_cast<std::size_t>(pa)]
                                     : 0;
                             const int pri_b =
                                 (pb >= 0 && pb < static_cast<int>(collinear_priority.size()))
                                     ? collinear_priority[static_cast<std::size_t>(pb)]
                                     : 0;
                             if (pri_a != pri_b) {
                                 return pri_a > pri_b;
                             }
                             return a < b;
                         });
    }

    Eigen::MatrixXd gram_keep(0, 0);
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    std::vector<int> kept_positions;
    kept_positions.reserve(static_cast<std::size_t>(p));

    for (const int pos : ordered_positions) {
        const double base_var = gram(pos, pos);
        if (!std::isfinite(base_var) || base_var <= 0.0) {
            drop_mask[static_cast<std::size_t>(pos)] = 1;
            continue;
        }
        double resid_var = base_var;
        if (!kept_positions.empty()) {
            Eigen::VectorXd g(kept_positions.size());
            for (std::size_t i = 0; i < kept_positions.size(); ++i) {
                g(static_cast<Eigen::Index>(i)) = gram(kept_positions[i], pos);
            }
            const Eigen::VectorXd alpha = ldlt.solve(g);
            resid_var = base_var - g.dot(alpha);
        }
        if (!std::isfinite(resid_var)) {
            drop_mask[static_cast<std::size_t>(pos)] = 1;
            continue;
        }
        if (resid_var < 0.0) {
            resid_var = 0.0;
        }
        const double base_norm = std::sqrt(base_var);
        const double resid_norm = std::sqrt(resid_var);
        if (!(base_norm > 0.0) || resid_norm <= rank_collinear_tol * base_norm) {
            drop_mask[static_cast<std::size_t>(pos)] = 1;
            continue;
        }

        const int old = gram_keep.rows();
        gram_keep.conservativeResize(old + 1, old + 1);
        for (int i = 0; i < old; ++i) {
            const double v = gram(kept_positions[static_cast<std::size_t>(i)], pos);
            gram_keep(i, old) = v;
            gram_keep(old, i) = v;
        }
        gram_keep(old, old) = base_var;
        kept_positions.push_back(pos);
        ldlt.compute(gram_keep);
        if (ldlt.info() != Eigen::Success) {
            drop_mask[static_cast<std::size_t>(pos)] = 1;
            kept_positions.pop_back();
            gram_keep.conservativeResize(old, old);
            if (old > 0) {
                ldlt.compute(gram_keep);
            }
        }
    }

    return drop_mask;
}

struct GroupCollapsedData {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> standard_fes;
    std::optional<Eigen::VectorXd> weights;
    std::optional<std::vector<Eigen::VectorXi>> clusters;
    hdfe::detail::GroupIndividualStructure gi;
    Eigen::VectorXi group_values;
    Eigen::VectorXi individual_values;
    // For mapping results back to the long-format input: rep_row[g] is the 0-based row index in
    // the original (y, X, fes, group_ids, individual_ids) arrays corresponding to group g.
    std::vector<int> rep_row;
};

GroupCollapsedData collapse_group_long_format(const Eigen::VectorXd& y,
                                              const Eigen::MatrixXd& X,
                                              const std::vector<Eigen::VectorXi>& fes,
                                              const Eigen::VectorXi& group_ids,
                                              const Eigen::VectorXi* individual_ids,
                                              int individual_fe_index,
                                              GroupAggregation aggregation,
                                              const Eigen::VectorXd* weights,
                                              const std::vector<Eigen::VectorXi>* clusters) {
    if (y.size() == 0) {
        throw std::runtime_error("Outcome vector must be non-empty");
    }
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (group_ids.size() != n) {
        throw std::runtime_error("group_ids must have the same length as y");
    }
    if (individual_ids && individual_ids->size() != n) {
        throw std::runtime_error("individual_ids must have the same length as y");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("weights must have the same length as y");
    }
    if (clusters) {
        for (const auto& c : *clusters) {
            if (c.size() != n) {
                throw std::runtime_error("clusters must have the same length as y");
            }
        }
    }
    for (const auto& fe : fes) {
        if (fe.size() != n) {
            throw std::runtime_error("Each fixed-effect vector must be length n");
        }
    }
    if (individual_ids && individual_fe_index < 0) {
        throw std::runtime_error("individual_ids must also be included in fes");
    }

    // Stata typically passes dense integer IDs (via egen group()).
    // For very large n (e.g., 100M edge rows), avoiding std::unordered_map node allocations is
    // critical. Prefer a dense-range mapping when the ID range is manageable; otherwise fall back
    // to unordered_map.
    constexpr std::int64_t kMaxDenseIdRange = 50000000LL;  // 50M ints ~= 200MB of RAM.

    bool use_dense_group_map = false;
    int group_min = 0;
    std::vector<int> group_dense;
    std::vector<int> group_index;  // only used in the unordered_map fallback
    std::vector<int> rep_row;
    std::vector<int> group_values;

    int group_max = group_ids(0);
    group_min = group_ids(0);
    for (int i = 1; i < n; ++i) {
        const int v = group_ids(i);
        group_min = std::min(group_min, v);
        group_max = std::max(group_max, v);
    }
    const std::int64_t group_range =
        static_cast<std::int64_t>(group_max) - static_cast<std::int64_t>(group_min) + 1LL;
    if (group_range > 0 && group_range <= kMaxDenseIdRange &&
        group_range <= static_cast<std::int64_t>(n)) {
        use_dense_group_map = true;
        group_dense.assign(static_cast<std::size_t>(group_range), -1);
        rep_row.reserve(static_cast<std::size_t>(std::min<std::int64_t>(n, group_range)));
        group_values.reserve(rep_row.capacity());
        for (int i = 0; i < n; ++i) {
            const int raw = group_ids(i);
            const std::size_t slot = static_cast<std::size_t>(raw - group_min);
            int& idx = group_dense[slot];
            if (idx < 0) {
                idx = static_cast<int>(rep_row.size());
                rep_row.push_back(i);
                group_values.push_back(raw);
            }
        }
    } else {
        std::unordered_map<int, int> group_map;
        group_map.reserve(static_cast<std::size_t>(n));
        group_index.assign(static_cast<std::size_t>(n), 0);
        rep_row.reserve(static_cast<std::size_t>(n));
        group_values.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int raw = group_ids(i);
            auto it = group_map.find(raw);
            if (it == group_map.end()) {
                const int g = static_cast<int>(rep_row.size());
                group_map.emplace(raw, g);
                rep_row.push_back(i);
                group_values.push_back(raw);
                group_index[static_cast<std::size_t>(i)] = g;
            } else {
                group_index[static_cast<std::size_t>(i)] = it->second;
            }
        }
    }
    const int G = static_cast<int>(rep_row.size());
    if (G == 0) {
        throw std::runtime_error("No groups found");
    }

    auto group_of_row = [&](int i) -> int {
        if (use_dense_group_map) {
            const int raw = group_ids(i);
            const std::size_t slot = static_cast<std::size_t>(raw - group_min);
            return group_dense[slot];
        }
        return group_index[static_cast<std::size_t>(i)];
    };

    for (int i = 0; i < n; ++i) {
        const int g = group_of_row(i);
        const int r = rep_row[static_cast<std::size_t>(g)];
        if (!approx_equal(y(i), y(r))) {
            throw std::runtime_error("y is not constant within group_ids");
        }
        for (int j = 0; j < X.cols(); ++j) {
            if (!approx_equal(X(i, j), X(r, j))) {
                throw std::runtime_error("X is not constant within group_ids");
            }
        }
        if (weights && !approx_equal((*weights)(i), (*weights)(r))) {
            throw std::runtime_error("weights are not constant within group_ids");
        }
        if (clusters) {
            for (const auto& c : *clusters) {
                if (c(i) != c(r)) {
                    throw std::runtime_error("clusters are not constant within group_ids");
                }
            }
        }
        for (int d = 0; d < static_cast<int>(fes.size()); ++d) {
            if (individual_ids && d == individual_fe_index) {
                continue;
            }
            if (fes[static_cast<std::size_t>(d)](i) != fes[static_cast<std::size_t>(d)](r)) {
                throw std::runtime_error("Fixed effects are not constant within group_ids");
            }
        }
    }

    GroupCollapsedData out;
    out.y.resize(G);
    out.X.resize(G, X.cols());
    out.group_values = Eigen::VectorXi::Zero(G);
    if (weights) {
        out.weights = Eigen::VectorXd::Zero(G);
    }
    if (clusters) {
        out.clusters = std::vector<Eigen::VectorXi>{};
        out.clusters->reserve(clusters->size());
        for (std::size_t j = 0; j < clusters->size(); ++j) {
            out.clusters->emplace_back(Eigen::VectorXi::Zero(G));
        }
    }

    out.standard_fes.clear();
    out.standard_fes.reserve(fes.size());
    for (int d = 0; d < static_cast<int>(fes.size()); ++d) {
        if (individual_ids && d == individual_fe_index) {
            continue;
        }
        out.standard_fes.emplace_back(Eigen::VectorXi::Zero(G));
    }

    for (int g = 0; g < G; ++g) {
        const int r = rep_row[static_cast<std::size_t>(g)];
        out.group_values(g) = group_values[static_cast<std::size_t>(g)];
        out.y(g) = y(r);
        out.X.row(g) = X.row(r);
        if (out.weights) {
            (*out.weights)(g) = (*weights)(r);
        }
        if (out.clusters) {
            for (std::size_t j = 0; j < out.clusters->size(); ++j) {
                (*out.clusters)[j](g) = (*clusters)[j](r);
            }
        }
        int cursor = 0;
        for (int d = 0; d < static_cast<int>(fes.size()); ++d) {
            if (individual_ids && d == individual_fe_index) {
                continue;
            }
            out.standard_fes[static_cast<std::size_t>(cursor)](g) =
                fes[static_cast<std::size_t>(d)](r);
            ++cursor;
        }
    }

    out.rep_row = std::move(rep_row);

    if (!individual_ids) {
        return out;
    }

    bool use_dense_indiv_map = false;
    int indiv_min = 0;
    std::vector<int> indiv_dense;
    std::vector<int> indiv_index;  // only used in the unordered_map fallback
    std::vector<int> indiv_values;
    int I = 0;

    int indiv_max = (*individual_ids)(0);
    indiv_min = (*individual_ids)(0);
    for (int i = 1; i < n; ++i) {
        const int v = (*individual_ids)(i);
        indiv_min = std::min(indiv_min, v);
        indiv_max = std::max(indiv_max, v);
    }
    const std::int64_t indiv_range =
        static_cast<std::int64_t>(indiv_max) - static_cast<std::int64_t>(indiv_min) + 1LL;
    if (indiv_range > 0 && indiv_range <= kMaxDenseIdRange &&
        indiv_range <= static_cast<std::int64_t>(n)) {
        use_dense_indiv_map = true;
        indiv_dense.assign(static_cast<std::size_t>(indiv_range), -1);
        indiv_values.reserve(static_cast<std::size_t>(std::min<std::int64_t>(n, indiv_range)));
        for (int i = 0; i < n; ++i) {
            const int raw = (*individual_ids)(i);
            const std::size_t slot = static_cast<std::size_t>(raw - indiv_min);
            int& idx = indiv_dense[slot];
            if (idx < 0) {
                idx = I++;
                indiv_values.push_back(raw);
            }
        }
    } else {
        std::unordered_map<int, int> indiv_map;
        indiv_map.reserve(static_cast<std::size_t>(n));
        indiv_index.assign(static_cast<std::size_t>(n), 0);
        indiv_values.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int raw = (*individual_ids)(i);
            auto it = indiv_map.find(raw);
            if (it == indiv_map.end()) {
                indiv_map.emplace(raw, I);
                indiv_values.push_back(raw);
                indiv_index[static_cast<std::size_t>(i)] = I;
                ++I;
            } else {
                indiv_index[static_cast<std::size_t>(i)] = it->second;
            }
        }
    }
    if (I == 0) {
        throw std::runtime_error("No individuals found");
    }
    out.individual_values = Eigen::VectorXi::Zero(I);
    for (int i = 0; i < I; ++i) {
        out.individual_values(i) = indiv_values[static_cast<std::size_t>(i)];
    }

    auto indiv_of_row = [&](int i) -> int {
        if (use_dense_indiv_map) {
            const int raw = (*individual_ids)(i);
            const std::size_t slot = static_cast<std::size_t>(raw - indiv_min);
            return indiv_dense[slot];
        }
        return indiv_index[static_cast<std::size_t>(i)];
    };

    std::vector<uint64_t> edges_by_group(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const uint64_t g = static_cast<uint64_t>(group_of_row(i));
        const uint64_t ind = static_cast<uint64_t>(indiv_of_row(i));
        edges_by_group[static_cast<std::size_t>(i)] = (g << 32) | (ind & 0xffffffffULL);
    }

    // Free mapping structures early to reduce peak memory use for huge n.
    if (!group_index.empty()) {
        std::vector<int>().swap(group_index);
    }
    if (!group_dense.empty()) {
        std::vector<int>().swap(group_dense);
    }
    if (!indiv_index.empty()) {
        std::vector<int>().swap(indiv_index);
    }
    if (!indiv_dense.empty()) {
        std::vector<int>().swap(indiv_dense);
    }

    std::sort(edges_by_group.begin(), edges_by_group.end());
    for (std::size_t i = 1; i < edges_by_group.size(); ++i) {
        if (edges_by_group[i] == edges_by_group[i - 1]) {
            throw std::runtime_error(
                "group_ids and individual_ids do not uniquely identify the observations");
        }
    }

    out.gi.num_groups = G;
    out.gi.num_individuals = I;
    out.gi.group_ptr.assign(static_cast<std::size_t>(G) + 1, 0);
    out.gi.group_individual.assign(static_cast<std::size_t>(n), 0);
    for (const uint64_t key : edges_by_group) {
        const int g = static_cast<int>(key >> 32);
        out.gi.group_ptr[static_cast<std::size_t>(g + 1)] += 1;
    }
    for (int g = 0; g < G; ++g) {
        out.gi.group_ptr[static_cast<std::size_t>(g + 1)] +=
            out.gi.group_ptr[static_cast<std::size_t>(g)];
    }
    std::vector<int> group_cursor = out.gi.group_ptr;
    for (const uint64_t key : edges_by_group) {
        const int g = static_cast<int>(key >> 32);
        const int ind = static_cast<int>(key & 0xffffffffULL);
        out.gi.group_individual[static_cast<std::size_t>(group_cursor[static_cast<std::size_t>(g)]++)] =
            ind;
    }

    out.gi.individual_ptr.assign(static_cast<std::size_t>(I) + 1, 0);
    out.gi.individual_group.assign(static_cast<std::size_t>(n), 0);
    for (const uint64_t key : edges_by_group) {
        const int ind = static_cast<int>(key & 0xffffffffULL);
        out.gi.individual_ptr[static_cast<std::size_t>(ind + 1)] += 1;
    }
    for (int i = 0; i < I; ++i) {
        out.gi.individual_ptr[static_cast<std::size_t>(i + 1)] +=
            out.gi.individual_ptr[static_cast<std::size_t>(i)];
    }
    std::vector<int> indiv_cursor = out.gi.individual_ptr;
    for (const uint64_t key : edges_by_group) {
        const int g = static_cast<int>(key >> 32);
        const int ind = static_cast<int>(key & 0xffffffffULL);
        out.gi.individual_group[static_cast<std::size_t>(indiv_cursor[static_cast<std::size_t>(ind)]++)] =
            g;
    }

    out.gi.group_scale.assign(static_cast<std::size_t>(G), 1.0);
    if (aggregation == GroupAggregation::Mean) {
        for (int g = 0; g < G; ++g) {
            const int size =
                out.gi.group_ptr[static_cast<std::size_t>(g + 1)] -
                out.gi.group_ptr[static_cast<std::size_t>(g)];
            if (size <= 0) {
                throw std::runtime_error("Invalid group with zero individuals");
            }
            out.gi.group_scale[static_cast<std::size_t>(g)] = 1.0 / static_cast<double>(size);
        }
    }

    return out;
}

struct GroupSingletonDropResult {
    std::vector<uint8_t> keep_groups;
    std::vector<int> individual_degrees;
    int dropped_groups = 0;
};

GroupSingletonDropResult drop_singletons_group_individual(
    const std::vector<Eigen::VectorXi>& standard_fes,
    const hdfe::detail::GroupIndividualStructure& gi) {
    const int G = gi.num_groups;
    const int I = gi.num_individuals;
    GroupSingletonDropResult res;
    res.keep_groups.assign(static_cast<std::size_t>(G), 1);
    res.individual_degrees.assign(static_cast<std::size_t>(I), 0);

    if (G <= 0 || I <= 0) {
        return res;
    }
    if (static_cast<int>(gi.group_ptr.size()) != G + 1 ||
        static_cast<int>(gi.individual_ptr.size()) != I + 1) {
        throw std::runtime_error("Invalid group/individual structure");
    }

    for (int i = 0; i < I; ++i) {
        res.individual_degrees[static_cast<std::size_t>(i)] =
            gi.individual_ptr[static_cast<std::size_t>(i + 1)] -
            gi.individual_ptr[static_cast<std::size_t>(i)];
    }

    struct FeLevelIndex {
        FeLookup lookup;
        std::vector<int> level_id;
        std::vector<std::vector<int>> groups_by_level;
        std::vector<int> counts;
    };

    std::vector<FeLevelIndex> feinfo;
    feinfo.reserve(standard_fes.size());
    for (const auto& fe : standard_fes) {
        if (fe.size() != G) {
            throw std::runtime_error("Standard fixed effects must be group-level (length #groups)");
        }
        FeLevelIndex info;
        info.lookup = build_lookup(fe);
        info.level_id.assign(static_cast<std::size_t>(G), 0);
        info.groups_by_level.assign(static_cast<std::size_t>(info.lookup.num_groups), {});
        for (int g = 0; g < G; ++g) {
            const int lid = info.lookup.index(fe(g));
            info.level_id[static_cast<std::size_t>(g)] = lid;
            info.groups_by_level[static_cast<std::size_t>(lid)].push_back(g);
        }
        info.counts.assign(static_cast<std::size_t>(info.lookup.num_groups), 0);
        for (int lid = 0; lid < info.lookup.num_groups; ++lid) {
            info.counts[static_cast<std::size_t>(lid)] =
                static_cast<int>(info.groups_by_level[static_cast<std::size_t>(lid)].size());
        }
        feinfo.push_back(std::move(info));
    }

    std::vector<uint8_t> in_queue(static_cast<std::size_t>(G), 0);
    std::deque<int> queue;
    auto enqueue_group = [&](int g) {
        if (g < 0 || g >= G) {
            return;
        }
        if (!res.keep_groups[static_cast<std::size_t>(g)] ||
            in_queue[static_cast<std::size_t>(g)]) {
            return;
        }
        in_queue[static_cast<std::size_t>(g)] = 1;
        queue.push_back(g);
    };

    for (int g = 0; g < G; ++g) {
        bool singleton = false;
        for (const auto& info : feinfo) {
            const int lid = info.level_id[static_cast<std::size_t>(g)];
            if (info.counts[static_cast<std::size_t>(lid)] == 1) {
                singleton = true;
                break;
            }
        }
        if (singleton) {
            enqueue_group(g);
        }
    }

    for (int i = 0; i < I; ++i) {
        if (res.individual_degrees[static_cast<std::size_t>(i)] == 1) {
            const int only =
                gi.individual_group[static_cast<std::size_t>(gi.individual_ptr[static_cast<std::size_t>(i)])];
            enqueue_group(only);
        }
    }

    while (!queue.empty()) {
        const int g = queue.front();
        queue.pop_front();
        in_queue[static_cast<std::size_t>(g)] = 0;
        if (!res.keep_groups[static_cast<std::size_t>(g)]) {
            continue;
        }

        res.keep_groups[static_cast<std::size_t>(g)] = 0;
        res.dropped_groups += 1;

        for (auto& info : feinfo) {
            const int lid = info.level_id[static_cast<std::size_t>(g)];
            int& count = info.counts[static_cast<std::size_t>(lid)];
            if (count <= 0) {
                continue;
            }
            count -= 1;
            if (count == 1) {
                int remaining = -1;
                for (const int gg : info.groups_by_level[static_cast<std::size_t>(lid)]) {
                    if (res.keep_groups[static_cast<std::size_t>(gg)]) {
                        remaining = gg;
                        break;
                    }
                }
                enqueue_group(remaining);
            }
        }

        const int begin = gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = gi.group_ptr[static_cast<std::size_t>(g + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int i = gi.group_individual[static_cast<std::size_t>(pos)];
            int& deg = res.individual_degrees[static_cast<std::size_t>(i)];
            if (deg <= 0) {
                continue;
            }
            deg -= 1;
            if (deg == 1) {
                int remaining = -1;
                const int ibegin = gi.individual_ptr[static_cast<std::size_t>(i)];
                const int iend = gi.individual_ptr[static_cast<std::size_t>(i + 1)];
                for (int k = ibegin; k < iend; ++k) {
                    const int gg = gi.individual_group[static_cast<std::size_t>(k)];
                    if (res.keep_groups[static_cast<std::size_t>(gg)]) {
                        remaining = gg;
                        break;
                    }
                }
                enqueue_group(remaining);
            }
        }
    }

    return res;
}

GroupCollapsedData filter_group_collapsed(const GroupCollapsedData& data,
                                          const std::vector<uint8_t>& keep_groups,
                                          const std::vector<int>& individual_degrees,
                                          GroupAggregation aggregation) {
    const int G = data.gi.num_groups;
    const int I = data.gi.num_individuals;
    if (static_cast<int>(keep_groups.size()) != G) {
        throw std::runtime_error("Invalid keep mask for group filtering");
    }
    if (static_cast<int>(individual_degrees.size()) != I) {
        throw std::runtime_error("Invalid individual degree vector for group filtering");
    }

    std::vector<int> group_new(static_cast<std::size_t>(G), -1);
    int new_G = 0;
    for (int g = 0; g < G; ++g) {
        if (keep_groups[static_cast<std::size_t>(g)]) {
            group_new[static_cast<std::size_t>(g)] = new_G++;
        }
    }

    std::vector<int> ind_new(static_cast<std::size_t>(I), -1);
    int new_I = 0;
    for (int i = 0; i < I; ++i) {
        if (individual_degrees[static_cast<std::size_t>(i)] > 0) {
            ind_new[static_cast<std::size_t>(i)] = new_I++;
        }
    }

    GroupCollapsedData out;
    out.y = Eigen::VectorXd::Zero(new_G);
    out.X = Eigen::MatrixXd::Zero(new_G, data.X.cols());
    out.group_values = Eigen::VectorXi::Zero(new_G);
    if (data.individual_values.size() > 0) {
        out.individual_values = Eigen::VectorXi::Zero(new_I);
    }
    out.standard_fes.clear();
    out.standard_fes.reserve(data.standard_fes.size());
    for (const auto& fe : data.standard_fes) {
        (void)fe;
        out.standard_fes.emplace_back(Eigen::VectorXi::Zero(new_G));
    }
    if (data.weights) {
        out.weights = Eigen::VectorXd::Zero(new_G);
    }
    if (data.clusters) {
        out.clusters = std::vector<Eigen::VectorXi>{};
        out.clusters->reserve(data.clusters->size());
        for (std::size_t j = 0; j < data.clusters->size(); ++j) {
            out.clusters->emplace_back(Eigen::VectorXi::Zero(new_G));
        }
    }
    if (!data.rep_row.empty()) {
        if (static_cast<int>(data.rep_row.size()) != G) {
            throw std::runtime_error("Invalid rep_row size for group filtering");
        }
        out.rep_row.assign(static_cast<std::size_t>(new_G), 0);
    }

    for (int g = 0; g < G; ++g) {
        const int ng = group_new[static_cast<std::size_t>(g)];
        if (ng < 0) {
            continue;
        }
        out.y(ng) = data.y(g);
        out.X.row(ng) = data.X.row(g);
        if (data.group_values.size() == G) {
            out.group_values(ng) = data.group_values(g);
        }
        if (!out.rep_row.empty()) {
            out.rep_row[static_cast<std::size_t>(ng)] = data.rep_row[static_cast<std::size_t>(g)];
        }
        if (out.weights) {
            (*out.weights)(ng) = (*data.weights)(g);
        }
        if (out.clusters) {
            for (std::size_t j = 0; j < out.clusters->size(); ++j) {
                (*out.clusters)[j](ng) = (*data.clusters)[j](g);
            }
        }
        for (std::size_t d = 0; d < out.standard_fes.size(); ++d) {
            out.standard_fes[d](ng) = data.standard_fes[d](g);
        }
    }
    if (out.individual_values.size() == new_I && data.individual_values.size() == I) {
        for (int i = 0; i < I; ++i) {
            const int ni = ind_new[static_cast<std::size_t>(i)];
            if (ni >= 0) {
                out.individual_values(ni) = data.individual_values(i);
            }
        }
    }

    out.gi.num_groups = new_G;
    out.gi.num_individuals = new_I;
    out.gi.group_ptr.assign(static_cast<std::size_t>(new_G) + 1, 0);
    for (int g = 0; g < G; ++g) {
        const int ng = group_new[static_cast<std::size_t>(g)];
        if (ng < 0) {
            continue;
        }
        const int begin = data.gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = data.gi.group_ptr[static_cast<std::size_t>(g + 1)];
        int count = 0;
        for (int pos = begin; pos < end; ++pos) {
            const int ni = ind_new[static_cast<std::size_t>(
                data.gi.group_individual[static_cast<std::size_t>(pos)])];
            if (ni >= 0) {
                ++count;
            }
        }
        out.gi.group_ptr[static_cast<std::size_t>(ng + 1)] = count;
    }
    for (int g = 0; g < new_G; ++g) {
        out.gi.group_ptr[static_cast<std::size_t>(g + 1)] +=
            out.gi.group_ptr[static_cast<std::size_t>(g)];
    }
    out.gi.group_individual.assign(static_cast<std::size_t>(out.gi.group_ptr.back()), 0);
    std::vector<int> cursor = out.gi.group_ptr;
    for (int g = 0; g < G; ++g) {
        const int ng = group_new[static_cast<std::size_t>(g)];
        if (ng < 0) {
            continue;
        }
        const int begin = data.gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = data.gi.group_ptr[static_cast<std::size_t>(g + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int ni = ind_new[static_cast<std::size_t>(
                data.gi.group_individual[static_cast<std::size_t>(pos)])];
            if (ni < 0) {
                continue;
            }
            out.gi.group_individual[static_cast<std::size_t>(cursor[static_cast<std::size_t>(ng)]++)] =
                ni;
        }
    }

    out.gi.individual_ptr.assign(static_cast<std::size_t>(new_I) + 1, 0);
    for (int g = 0; g < new_G; ++g) {
        const int begin = out.gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = out.gi.group_ptr[static_cast<std::size_t>(g + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int ni = out.gi.group_individual[static_cast<std::size_t>(pos)];
            out.gi.individual_ptr[static_cast<std::size_t>(ni + 1)] += 1;
        }
    }
    for (int i = 0; i < new_I; ++i) {
        out.gi.individual_ptr[static_cast<std::size_t>(i + 1)] +=
            out.gi.individual_ptr[static_cast<std::size_t>(i)];
    }
    out.gi.individual_group.assign(static_cast<std::size_t>(out.gi.group_ptr.back()), 0);
    std::vector<int> icursor = out.gi.individual_ptr;
    for (int g = 0; g < new_G; ++g) {
        const int begin = out.gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = out.gi.group_ptr[static_cast<std::size_t>(g + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int ni = out.gi.group_individual[static_cast<std::size_t>(pos)];
            out.gi.individual_group[static_cast<std::size_t>(icursor[static_cast<std::size_t>(ni)]++)] =
                g;
        }
    }

    out.gi.group_scale.assign(static_cast<std::size_t>(new_G), 1.0);
    if (aggregation == GroupAggregation::Mean) {
        for (int g = 0; g < new_G; ++g) {
            const int size =
                out.gi.group_ptr[static_cast<std::size_t>(g + 1)] -
                out.gi.group_ptr[static_cast<std::size_t>(g)];
            if (size <= 0) {
                throw std::runtime_error("Invalid group with zero individuals after filtering");
            }
            out.gi.group_scale[static_cast<std::size_t>(g)] = 1.0 / static_cast<double>(size);
        }
    }

    return out;
}

struct FeIndexMap {
    Eigen::VectorXi level_index;  // length G: 0..L-1
    Eigen::VectorXi raw_values;   // length L: raw ids for each level
};

FeIndexMap build_fe_index_map(const Eigen::VectorXi& raw_ids) {
    const int n = static_cast<int>(raw_ids.size());
    if (n == 0) {
        return FeIndexMap{};
    }
    std::unordered_map<int, int> map;
    map.reserve(static_cast<std::size_t>(n));
    std::vector<int> raw_values;
    raw_values.reserve(static_cast<std::size_t>(n));
    Eigen::VectorXi idx(n);
    int next = 0;
    for (int i = 0; i < n; ++i) {
        const int raw = raw_ids(i);
        auto it = map.find(raw);
        if (it == map.end()) {
            map.emplace(raw, next);
            raw_values.push_back(raw);
            idx(i) = next;
            ++next;
        } else {
            idx(i) = it->second;
        }
    }

    Eigen::VectorXi levels(next);
    for (int i = 0; i < next; ++i) {
        levels(i) = raw_values[static_cast<std::size_t>(i)];
    }
    FeIndexMap out;
    out.level_index = std::move(idx);
    out.raw_values = std::move(levels);
    return out;
}

void gi_group_sum(const hdfe::detail::GroupIndividualStructure& gi,
                  const Eigen::VectorXd& x,
                  Eigen::VectorXd& out) {
    const int G = gi.num_groups;
    if (G <= 0) {
        out.resize(0);
        return;
    }
    if (x.size() != gi.num_individuals) {
        throw std::runtime_error("gi_group_sum: x must be length num_individuals");
    }
    if (static_cast<int>(gi.group_ptr.size()) != G + 1) {
        throw std::runtime_error("gi_group_sum: invalid group_ptr size");
    }
    if (static_cast<int>(gi.group_scale.size()) != G) {
        throw std::runtime_error("gi_group_sum: invalid group_scale size");
    }
    out.setZero(G);
    for (int g = 0; g < G; ++g) {
        const int begin = gi.group_ptr[static_cast<std::size_t>(g)];
        const int end = gi.group_ptr[static_cast<std::size_t>(g + 1)];
        double sum = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int i = gi.group_individual[static_cast<std::size_t>(pos)];
            sum += x(i);
        }
        out(g) = gi.group_scale[static_cast<std::size_t>(g)] * sum;
    }
}

void gi_individual_sum(const hdfe::detail::GroupIndividualStructure& gi,
                       const Eigen::VectorXd& x,
                       Eigen::VectorXd& out) {
    const int I = gi.num_individuals;
    if (I <= 0) {
        out.resize(0);
        return;
    }
    if (x.size() != gi.num_groups) {
        throw std::runtime_error("gi_individual_sum: x must be length num_groups");
    }
    if (static_cast<int>(gi.individual_ptr.size()) != I + 1) {
        throw std::runtime_error("gi_individual_sum: invalid individual_ptr size");
    }
    if (static_cast<int>(gi.group_scale.size()) != gi.num_groups) {
        throw std::runtime_error("gi_individual_sum: invalid group_scale size");
    }
    out.setZero(I);
    for (int i = 0; i < I; ++i) {
        const int begin = gi.individual_ptr[static_cast<std::size_t>(i)];
        const int end = gi.individual_ptr[static_cast<std::size_t>(i + 1)];
        double sum = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int g = gi.individual_group[static_cast<std::size_t>(pos)];
            sum += gi.group_scale[static_cast<std::size_t>(g)] * x(g);
        }
        out(i) = sum;
    }
}

Eigen::VectorXd gi_preconditioner(const hdfe::detail::GroupIndividualStructure& gi,
                                  const Eigen::VectorXd* weights) {
    const int I = gi.num_individuals;
    const int G = gi.num_groups;
    if (I <= 0 || G <= 0) {
        return Eigen::VectorXd{};
    }
    if (weights && weights->size() != G) {
        throw std::runtime_error("gi_preconditioner: weights must be length num_groups");
    }
    if (static_cast<int>(gi.individual_ptr.size()) != I + 1) {
        throw std::runtime_error("gi_preconditioner: invalid individual_ptr size");
    }
    Eigen::VectorXd diag = Eigen::VectorXd::Ones(I);
    for (int i = 0; i < I; ++i) {
        const int begin = gi.individual_ptr[static_cast<std::size_t>(i)];
        const int end = gi.individual_ptr[static_cast<std::size_t>(i + 1)];
        double acc = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int g = gi.individual_group[static_cast<std::size_t>(pos)];
            const double scale = gi.group_scale[static_cast<std::size_t>(g)];
            const double w = weights ? (*weights)(g) : 1.0;
            acc += w * scale * scale;
        }
        diag(i) = acc > 0.0 ? acc : 1.0;
    }
    return diag;
}

Eigen::VectorXd accel_a1(const Eigen::VectorXd& x3,
                         const Eigen::VectorXd& x2,
                         const Eigen::VectorXd& x1,
                         int accel,
                         double a1p1,
                         double a2p1,
                         int a2p2) {
    if (accel == 1) {
        return x3 + a1p1 * (x2 - x1);
    }
    if (accel != 2) {
        return x3;
    }

    const Eigen::ArrayXd x3a = x3.array();
    const Eigen::ArrayXd x2a = x2.array();
    const Eigen::ArrayXd x1a = x1.array();
    const Eigen::ArrayXd d1 = x2a - x1a;
    const Eigen::ArrayXd d2 = x3a - x2a;
    const Eigen::ArrayXd c1 =
        (x3a > x2a).cast<double>() * (x2a > x1a).cast<double>() *
        ((x3a + x1a) < (2.0 * x2a)).cast<double>();
    const Eigen::ArrayXd c2 =
        (x3a < x2a).cast<double>() * (x2a < x1a).cast<double>() *
        ((x3a + x1a) > (2.0 * x2a)).cast<double>();
    const Eigen::ArrayXd ratio = d2 / d1;
    const Eigen::ArrayXd dum =
        x2a - ((d2 * d1) / (d2 - d1)) * (1.0 - ratio.pow(static_cast<double>(a2p2)));
    const Eigen::ArrayXd mask =
        ((c1 + c2) > 0.5).cast<double>() * (d1.abs() > a2p1).cast<double>() *
        ((d2 - d1).abs() > a2p1).cast<double>();
    return (mask > 0.5).select(dum, x3a).matrix();
}

void update_standard_fe(const Eigen::VectorXd& residual,
                        const FeIndexMap& map,
                        const Eigen::VectorXd* weights,
                        Eigen::VectorXd& level_values,
                        Eigen::VectorXd& group_values) {
    const int G = static_cast<int>(residual.size());
    if (map.level_index.size() != G) {
        throw std::runtime_error("update_standard_fe: FE index length mismatch");
    }
    if (weights && weights->size() != G) {
        throw std::runtime_error("update_standard_fe: weights length mismatch");
    }
    const int L = static_cast<int>(map.raw_values.size());
    level_values.setZero(L);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(L);
    for (int g = 0; g < G; ++g) {
        const int lid = map.level_index(g);
        const double w = weights ? (*weights)(g) : 1.0;
        level_values(lid) += w * residual(g);
        counts(lid) += w;
    }
    for (int lid = 0; lid < L; ++lid) {
        const double denom = counts(lid);
        if (denom > 0.0) {
            level_values(lid) /= denom;
        } else {
            level_values(lid) = 0.0;
        }
    }
    group_values.resize(G);
    for (int g = 0; g < G; ++g) {
        group_values(g) = level_values(map.level_index(g));
    }
}

Eigen::VectorXd solve_a1(Eigen::VectorXd x,
                         const hdfe::detail::GroupIndividualStructure& gi,
                         const Eigen::VectorXd& dd,
                         const Eigen::VectorXd* weights,
                         const Eigen::VectorXd& precond,
                         int max_iter,
                         double tol,
                         int verbose) {
    const int I = gi.num_individuals;
    const int G = gi.num_groups;
    if (dd.size() != G) {
        throw std::runtime_error("solve_a1: dd must be length num_groups");
    }
    if (weights && weights->size() != G) {
        throw std::runtime_error("solve_a1: weights must be length num_groups");
    }
    if (precond.size() != I) {
        throw std::runtime_error("solve_a1: preconditioner must be length num_individuals");
    }
    if (x.size() != I) {
        throw std::runtime_error("solve_a1: initial x must be length num_individuals");
    }

    Eigen::VectorXd tmp_group(G);
    Eigen::VectorXd ax(I);
    gi_group_sum(gi, x, tmp_group);
    if (weights) {
        tmp_group.array() *= weights->array();
    }
    gi_individual_sum(gi, tmp_group, ax);

    Eigen::VectorXd rhs_group = dd;
    if (weights) {
        rhs_group.array() *= weights->array();
    }
    Eigen::VectorXd b(I);
    gi_individual_sum(gi, rhs_group, b);

    Eigen::VectorXd r = b - ax;
    Eigen::VectorXd z = r.array() / precond.array();
    Eigen::VectorXd p = z;
    double rsold = r.dot(z);

    Eigen::VectorXd q(I);
    int it = 0;
    while (std::sqrt(rsold) > tol && it < max_iter) {
        gi_group_sum(gi, p, tmp_group);
        if (weights) {
            tmp_group.array() *= weights->array();
        }
        gi_individual_sum(gi, tmp_group, q);
        const double denom = p.dot(q);
        if (!std::isfinite(denom) || denom == 0.0) {
            break;
        }
        const double alpha = rsold / denom;
        x.noalias() += alpha * p;
        r.noalias() -= alpha * q;
        z = r.array() / precond.array();
        const double rsnew = r.dot(z);
        const double beta = rsnew / rsold;
        p = z + beta * p;
        rsold = rsnew;
        ++it;
        if (verbose > 1) {
            std::cout << "subiterations (" << it << ") dif=" << std::sqrt(rsold) << "\n";
        }
    }
    if (verbose > 0) {
        std::cout << "subiterations (" << it << ") tol=" << tol << "\n";
    }
    return x;
}

double weighted_mse(const Eigen::VectorXd& r, const Eigen::VectorXd* weights) {
    if (!weights) {
        return r.size() > 0 ? r.squaredNorm() / static_cast<double>(r.size()) : 0.0;
    }
    const double sumw = weights->sum();
    if (sumw <= 0.0) {
        throw std::runtime_error("Weights must sum to a positive value");
    }
    return (r.array().square() * weights->array()).sum() / sumw;
}

int find_matching_fe_index(const Eigen::VectorXi& clusters,
                           const std::vector<Eigen::VectorXi>& fes) {
    for (std::size_t d = 0; d < fes.size(); ++d) {
        const auto& fe = fes[d];
        if (fe.size() != clusters.size()) {
            continue;
        }
        bool same = true;
        for (int i = 0; i < clusters.size(); ++i) {
            if (clusters(i) != fe(i)) {
                same = false;
                break;
            }
        }
        if (same) {
            return static_cast<int>(d);
        }
    }
    return -1;
}

bool fe_nested_within_cluster(const Eigen::VectorXi& fe,
                              const Eigen::VectorXi& clusters) {
    if (fe.size() != clusters.size()) {
        return false;
    }
    const int n = fe.size();
    if (n == 0) {
        return true;
    }
    if (fe.data() == clusters.data()) {
        return true;
    }
    if (fe(0) == clusters(0) && fe(n / 2) == clusters(n / 2) &&
        fe(n - 1) == clusters(n - 1)) {
        bool identical = true;
        for (int i = 0; i < n; ++i) {
            if (fe(i) != clusters(i)) {
                identical = false;
                break;
            }
        }
        if (identical) {
            return true;
        }
    }
    int min_id = fe(0);
    int max_id = fe(0);
    for (int i = 1; i < n; ++i) {
        const int v = fe(i);
        min_id = std::min(min_id, v);
        max_id = std::max(max_id, v);
    }
    const long long range_ll =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    const bool dense_ok =
        min_id >= 0 && range_ll > 0 && range_ll <= kDenseRangeCap &&
        range_ll <= static_cast<long long>(n) * 2LL;
    if (dense_ok) {
        const int range = static_cast<int>(range_ll);
        std::vector<int> mapping(static_cast<std::size_t>(range), 0);
        std::vector<uint8_t> seen(static_cast<std::size_t>(range), 0);
        for (int i = 0; i < n; ++i) {
            const int idx = fe(i) - min_id;
            const int cl_id = clusters(i);
            if (!seen[static_cast<std::size_t>(idx)]) {
                seen[static_cast<std::size_t>(idx)] = 1;
                mapping[static_cast<std::size_t>(idx)] = cl_id;
            } else if (mapping[static_cast<std::size_t>(idx)] != cl_id) {
                return false;
            }
        }
        return true;
    }

    std::unordered_map<int, int> mapping;
    mapping.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int fe_id = fe(i);
        const int cl_id = clusters(i);
        auto it = mapping.find(fe_id);
        if (it == mapping.end()) {
            mapping.emplace(fe_id, cl_id);
        } else if (it->second != cl_id) {
            return false;
        }
    }
    return true;
}

double normal_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::runtime_error("normal_ppf requires p in (0, 1)");
    }
    // Peter J. Acklam's inverse normal CDF approximation.
    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 = 2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 = 1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 = 2.506628277459239e+00;

    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 = 1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 = 6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;

    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 = 4.374664141464968e+00;
    constexpr double c6 = 2.938163982698783e+00;

    constexpr double d1 = 7.784695709041462e-03;
    constexpr double d2 = 3.224671290700398e-01;
    constexpr double d3 = 2.445134137142996e+00;
    constexpr double d4 = 3.754408661907416e+00;

    constexpr double plow = 0.02425;
    constexpr double phigh = 1.0 - plow;

    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p <= phigh) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

double normal_critical_value(double level_percent) {
    if (!(level_percent > 0.0 && level_percent < 100.0)) {
        throw std::runtime_error("level must be between 0 and 100");
    }
    const double p = 0.5 + (level_percent / 200.0);
    return normal_ppf(p);
}

double betacf(double a, double b, double x) {
    constexpr int kMaxIterations = 200;
    constexpr double kEps = 3e-14;
    constexpr double kFpMin = 1e-30;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) {
        d = kFpMin;
    }
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIterations; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) {
            d = kFpMin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) {
            c = kFpMin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) {
            d = kFpMin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) {
            c = kFpMin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) <= kEps) {
            break;
        }
    }
    return h;
}

double regularized_incomplete_beta(double a, double b, double x) {
    if (!(x >= 0.0 && x <= 1.0)) {
        throw std::runtime_error("regularized_incomplete_beta requires x in [0, 1]");
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }

    const double log_bt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                          a * std::log(x) + b * std::log1p(-x);
    const double bt = std::exp(log_bt);

    const double threshold = (a + 1.0) / (a + b + 2.0);
    if (x < threshold) {
        return bt * betacf(a, b, x) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double student_t_cdf(double t, double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(t)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (t == 0.0) {
        return 0.5;
    }
    if (df > 1e7) {
        return 0.5 * std::erfc(-t * kInvSqrt2);
    }
    const double x = df / (df + t * t);
    const double ib = regularized_incomplete_beta(df / 2.0, 0.5, x);
    if (t > 0.0) {
        return 1.0 - 0.5 * ib;
    }
    return 0.5 * ib;
}

double student_t_tail(double t_abs, double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(t_abs >= 0.0) || !std::isfinite(t_abs)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (df > 1e7) {
        return 0.5 * std::erfc(t_abs * kInvSqrt2);
    }
    const double x = df / (df + t_abs * t_abs);
    return 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x);
}

double student_t_inv_cdf(double p, double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(p > 0.0 && p < 1.0) || !std::isfinite(p)) {
        throw std::runtime_error("student_t_inv_cdf requires p in (0, 1)");
    }
    if (p == 0.5) {
        return 0.0;
    }
    if (p < 0.5) {
        return -student_t_inv_cdf(1.0 - p, df);
    }

    double low = 0.0;
    double high = 1.0;
    while (student_t_cdf(high, df) < p) {
        high *= 2.0;
        if (high > 1e12) {
            break;
        }
    }

    for (int iter = 0; iter < 120; ++iter) {
        const double mid = 0.5 * (low + high);
        const double cdf_mid = student_t_cdf(mid, df);
        if (!std::isfinite(cdf_mid)) {
            break;
        }
        if (cdf_mid < p) {
            low = mid;
        } else {
            high = mid;
        }
        if (std::abs(high - low) <= 1e-13 * (1.0 + high + low)) {
            break;
        }
    }
    return 0.5 * (low + high);
}

double student_t_critical_value(double level_percent, double df) {
    if (!(level_percent > 0.0 && level_percent < 100.0)) {
        throw std::runtime_error("level must be between 0 and 100");
    }
    if (!(df > 0.0) || !std::isfinite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double p = 0.5 + (level_percent / 200.0);
    return student_t_inv_cdf(p, df);
}

// When the residual degrees of freedom are zero or negative (saturated /
// over-specified model), reghdfe reports missing standard errors, t, p-values,
// confidence intervals and F-statistic, while still surfacing the (possibly
// negative) df_r. Mirror that behaviour here: convert covariance diagonals for
// non-omitted coefficients to a sentinel value below -kVDiagZeroEps so the
// Stata display produces sqrt(V[j,j]) = missing, and mark the inferential
// vectors as NaN. Off-diagonals are cleaned to 0 so `ereturn post` accepts
// the matrix.
void mark_invalid_inference_for_saturated(hdfe::HdfeResults& results) {
    constexpr double kVDiagSentinel = -1.0e-10;  // below the 1e-12 fix-to-zero threshold
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int n_coef = static_cast<int>(results.coefficients.size());
    for (int j = 0; j < n_coef; ++j) {
        const bool is_omitted =
            (j < static_cast<int>(results.omitted_reason.size()) &&
             results.omitted_reason[j] != 0);
        if (is_omitted) {
            continue;
        }
        const double vjj = results.covariance(j, j);
        if (!(std::isfinite(vjj) && vjj <= kVDiagSentinel)) {
            results.covariance(j, j) = kVDiagSentinel;
        }
        if (j < results.std_errors.size()) {
            results.std_errors(j) = nan;
        }
        if (j < results.tvalues.size()) {
            results.tvalues(j) = nan;
        }
        if (j < results.pvalues.size()) {
            results.pvalues(j) = nan;
        }
        if (j < results.conf_int.rows()) {
            results.conf_int(j, 0) = nan;
            results.conf_int(j, 1) = nan;
        }
    }
    for (int i = 0; i < n_coef; ++i) {
        for (int j = 0; j < n_coef; ++j) {
            if (i != j && !std::isfinite(results.covariance(i, j))) {
                results.covariance(i, j) = 0.0;
            }
        }
    }
}

void recompute_inference(Eigen::VectorXd& coefficients,
                         Eigen::VectorXd& std_errors,
                         Eigen::VectorXd& tvalues,
                         Eigen::VectorXd& pvalues,
                         Eigen::MatrixXd& conf_int,
                         double level_percent,
                         double df_resid) {
    const int p = static_cast<int>(coefficients.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    tvalues.resize(p);
    pvalues.resize(p);
    conf_int.resize(p, 2);
    const bool df_ok = (df_resid > 0.0) && std::isfinite(df_resid);
    const double crit = df_ok ? student_t_critical_value(level_percent, df_resid) : nan;
    for (int j = 0; j < p; ++j) {
        const double se = std_errors(j);
        if (!std::isfinite(se) || se <= 0.0) {
            tvalues(j) = nan;
            pvalues(j) = nan;
            conf_int(j, 0) = nan;
            conf_int(j, 1) = nan;
            continue;
        }
        const double t = coefficients(j) / se;
        tvalues(j) = t;
        const double abs_t = std::abs(t);
        if (df_ok && std::isfinite(abs_t)) {
            const double tail = student_t_tail(abs_t, df_resid);
            pvalues(j) = 2.0 * tail;
        } else {
            pvalues(j) = nan;
        }
        const double half = crit * se;
        conf_int(j, 0) = coefficients(j) - half;
        conf_int(j, 1) = coefficients(j) + half;
    }
}

}  // namespace

HdfeRegressorV11::HdfeRegressorV11(HdfeOptions options, ThreadingOptions threading)
    : options_(options), threading_(threading) {
    if (!options_.symmetric_sweep) {
        options_.symmetric_sweep = threading_.symmetric_sweep;
    }
}

int HdfeRegressorV11::resolve_threads(int n_rows, int num_fes) const {
#ifdef HDFE_USE_OPENMP
    if (n_rows < threading_.min_parallel_rows) {
        return 1;
    }

    auto cap_for_problem_size = [&](int requested) {
        int capped = std::max(1, requested);
        if (!options_.retain_fixed_effects && num_fes > 0) {
            int problem_cap = 16;
            if (n_rows < 100000) {
                problem_cap = 1;
            } else if (n_rows < 250000) {
                problem_cap = 8;
            }
            capped = std::min(capped, problem_cap);
        }
        return std::max(1, capped);
    };

    int threads_cap = 1;
    if (options_.num_threads > 0) {
        threads_cap = std::max(1, options_.num_threads);
        if (threading_.max_threads > 0) {
            threads_cap = std::min(threads_cap, threading_.max_threads);
        }
        threads_cap = cap_for_problem_size(threads_cap);
        return threads_cap;
    } else {
        threads_cap = threading_.default_threads;
        if (threads_cap <= 0) {
            threads_cap = omp_get_max_threads();
            if (threads_cap <= 1 && std::getenv("OMP_NUM_THREADS") == nullptr) {
                const unsigned hc = std::thread::hardware_concurrency();
                if (hc > 0) {
                    threads_cap = static_cast<int>(hc);
                }
            }
        }
        if (threads_cap <= 0) {
            threads_cap = 1;
        }
        threads_cap = std::max(1, threads_cap);
        if (threading_.target_rows_per_thread > 0) {
            const int by_rows =
                std::max(1, n_rows / std::max(1, threading_.target_rows_per_thread));
            threads_cap = std::min(threads_cap, by_rows);
        }
        if (threading_.max_threads > 0) {
            threads_cap = std::min(threads_cap, threading_.max_threads);
        }
        threads_cap = cap_for_problem_size(threads_cap);
    }

    if (threads_cap <= 1) {
        return 1;
    }

    return threads_cap;
#else
    (void)n_rows;
    (void)num_fes;
    return options_.num_threads > 0 ? options_.num_threads : 1;
#endif
}

AbsorptionMethod HdfeRegressorV11::select_method(std::size_t num_fes) const {
    if (options_.absorption_method != AbsorptionMethod::Auto) {
        return options_.absorption_method;
    }
    if (num_fes <= 1) {
        return AbsorptionMethod::GaussSeidel;
    }
    if (options_.symmetric_sweep) {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    return AbsorptionMethod::GaussSeidel;
}

void HdfeRegressorV11::apply_common_postprocessing(const Eigen::Ref<const Eigen::VectorXd>& y,
                                                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                                                  const Eigen::VectorXd* weights,
                                                  const std::vector<int>& fe_levels,
                                                  const detail::OlsResult& ols_result) {
    results_.coefficients = ols_result.coefficients;
    results_.std_errors = ols_result.std_errors;
    results_.tvalues = ols_result.tvalues;
    results_.pvalues = ols_result.pvalues;
    results_.conf_int = ols_result.conf_int;
    results_.covariance = ols_result.covariance;
    results_.residuals = ols_result.residuals;
    results_.omitted_reason.clear();
    results_.nobs = static_cast<int>(y.size());
    results_.nobs_full = results_.nobs;
    results_.num_singletons = 0;
    results_.nobs_effective = static_cast<double>(results_.nobs);
    results_.nobs_full_effective = results_.nobs_effective;
    results_.num_singletons_effective = 0.0;
    results_.sample_index.resize(0);
    results_.df_resid = ols_result.df_resid;
    results_.df_resid_unadj = ols_result.df_resid;
    results_.df_m = 0.0;
    results_.df_a = 0.0;
    results_.df_a_levels = 0.0;
    results_.df_a_exact = 0.0;
    results_.df_a_nested = 0.0;
    results_.r2 = ols_result.r2;
    results_.r2_within = ols_result.r2_within;
    results_.sigma2 = ols_result.sigma2;
    results_.rss = ols_result.rss;
    results_.tss = ols_result.tss;
    results_.tss_within = ols_result.within_tss;
    results_.fe_num_levels = fe_levels;
    results_.fe_base_levels = fe_levels;
    results_.fe_redundant.clear();
    results_.fe_num_coefs.clear();
    results_.fe_inexact.clear();
    results_.fe_nested.clear();
    results_.groupvar.resize(0);
    results_.fe_effects.clear();
    results_.fe_save_effects.clear();
    results_.fe_recovery_iterations = 0;
    results_.fe_recovery_max_delta = 0.0;
    results_.fe_recovery_converged = true;
    results_.num_clusters = ols_result.num_clusters;
    results_.cluster_counts.clear();
    results_.cluster_combo_counts.clear();
    results_.cluster_scale = 1.0;
}

void HdfeRegressorV11::fit(const Eigen::Ref<const Eigen::VectorXd>& y,
                           const Eigen::Ref<const Eigen::MatrixXd>& X,
                           const std::vector<Eigen::VectorXi>& fes,
                           const Eigen::VectorXd* weights,
                           const std::vector<Eigen::VectorXi>* clusters,
                           const Eigen::MatrixXd* instruments,
                           const std::vector<int>& endogenous_idx,
                           const std::vector<detail::HeterogeneousSlopeTerm>* slopes) {
    const auto fit_outer_t0 = std::chrono::steady_clock::now();
    if (y.size() == 0) {
        throw std::runtime_error("Outcome vector must be non-empty");
    }
    if (X.rows() != y.size()) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (weights && weights->size() != y.size()) {
        throw std::runtime_error("Weights must have the same length as y");
    }
    gpu_used_ = false;
    gpu_status_code_ = 0;
    gpu_attempted_ = false;
    gpu_absorption_converged_ = false;
    gpu_absorption_iterations_ = 0;
    if (clusters) {
        for (const auto& c : *clusters) {
            if (c.size() != y.size()) {
                throw std::runtime_error("Clusters must have the same length as y");
            }
        }
    }
    const std::vector<detail::HeterogeneousSlopeTerm> empty_slopes;
    const std::vector<detail::HeterogeneousSlopeTerm>* slopes_in =
        slopes ? slopes : &empty_slopes;
    const bool has_slopes = slopes_in && !slopes_in->empty();
    if (has_slopes && fes.empty()) {
        throw std::runtime_error("Heterogeneous slopes require at least one absorbed FE");
    }
    if (has_slopes) {
        for (const auto& slope : *slopes_in) {
            if (slope.fe_index < 0 || slope.fe_index >= static_cast<int>(fes.size())) {
                throw std::runtime_error("Heterogeneous slope FE index out of range");
            }
            if (slope.values.size() != y.size()) {
                throw std::runtime_error("Heterogeneous slope variable must have the same length as y");
            }
        }
    }

    const bool wants_iv = !endogenous_idx.empty();
    const bool has_instruments = instruments && instruments->cols() > 0;
    if (wants_iv && !has_instruments) {
        throw std::runtime_error("Endogenous regressors supplied without instruments");
    }
    if (has_instruments && !wants_iv) {
        throw std::runtime_error(
            "Instrument matrix supplied but no endogenous indices were provided");
    }

    const bool has_fes = !fes.empty();
    const bool drop_intercept = options_.fit_intercept && has_fes && !wants_iv && !has_instruments;
    const MobilityProfileConfig mobility_cfg = load_mobility_profile_config();
    const AbsorptionCacheConfig abs_cache_cfg = load_absorption_cache_config();
    const FeStructureCacheConfig fe_cache_cfg = load_fe_structure_cache_config();
    const int nobs_full = static_cast<int>(y.size());

    const double nobs_full_effective =
        (options_.weights_are_frequencies && weights) ? weights->sum()
                                                      : static_cast<double>(nobs_full);

    int singletons_dropped_rows = 0;
    double singletons_dropped_effective = 0.0;
    std::vector<int> kept_idx;
    std::optional<Eigen::VectorXd> y_work;
    std::optional<Eigen::MatrixXd> X_work;
    std::optional<std::vector<Eigen::VectorXi>> fes_work;
    std::optional<Eigen::VectorXd> weights_work;
    std::optional<std::vector<Eigen::VectorXi>> clusters_work;
    std::optional<Eigen::MatrixXd> instruments_work;
    std::optional<std::vector<detail::HeterogeneousSlopeTerm>> slopes_work;

    const Eigen::MatrixXd* inst_ptr = has_instruments ? instruments : nullptr;

    FeStructureCache fe_cache;
    bool fe_cache_hit = false;
    std::uint64_t fe_cache_signature = 0;
    bool fe_cache_signature_ready = false;
    auto ensure_fe_cache_signature = [&]() {
        if (!fe_cache_signature_ready) {
            fe_cache_signature = hash_fe_structure_cache_signature(
                fes, weights, options_.drop_singletons, options_.weights_are_frequencies);
            fe_cache_signature_ready = true;
        }
    };

    auto run_fit = [&](const Eigen::Ref<const Eigen::VectorXd>& y_use,
                       const Eigen::Ref<const Eigen::MatrixXd>& X_use,
                       const std::vector<Eigen::VectorXi>& fes_use,
                       const Eigen::VectorXd* w_ptr,
                       const std::vector<Eigen::VectorXi>* c_ptr,
                       const Eigen::MatrixXd* inst_use,
                       const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use) {
        if (y_use.size() == 0) {
            throw std::runtime_error(
                "no observations remain for estimation (empty input or every row "
                "dropped as a singleton)");
        }
        const auto fit_t0 = std::chrono::steady_clock::now();
        const int threads = resolve_threads(static_cast<int>(y_use.size()),
                                            static_cast<int>(fes_use.size()));
        threads_used_ = threads;
#ifdef HDFE_USE_OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, threads));
        Eigen::setNbThreads(std::max(1, threads));
#endif

        HdfeOptions tuned = options_;
        tuned.from_auto = (options_.absorption_method == AbsorptionMethod::Auto);
        const std::optional<bool> env_krylov = read_env_bool("XHDFE_USE_KRYLOV");
        tuned.num_threads = threads;
        const std::vector<detail::HeterogeneousSlopeTerm>& slope_terms =
            slopes_use ? *slopes_use : empty_slopes;
        const bool has_slopes_use = !slope_terms.empty();

        MobilityHint mobility_hint;
        bool have_mobility_hint = false;
        bool mobility_profile_match = false;
        MobilityProfile profile;
        bool have_profile = false;
        if (!has_slopes_use && (mobility_cfg.mode == "read" || mobility_cfg.mode == "auto")) {
            if (load_mobility_profile(mobility_cfg.path, profile)) {
                have_profile = true;
            }
        }
        if (have_profile) {
            bool match = false;
            if (profile.signature != 0 || profile.signature_canon != 0) {
                const std::uint64_t sig = hash_fe_signature(fes_use, options_.drop_singletons);
                if (profile.signature != 0 && profile.signature == sig) {
                    match = true;
                }
                if (!match && profile.signature_canon != 0 &&
                    profile.signature_canon == sig) {
                    match = true;
                }
            }
            if (!match && fe_cache_hit &&
                profile.signature != 0) {
                const std::vector<int>* kept_ptr =
                    (options_.drop_singletons && !kept_idx.empty()) ? &kept_idx : nullptr;
                const std::uint64_t sig_raw =
                    hash_fe_signature_filtered(fes, kept_ptr, options_.drop_singletons);
                match = (profile.signature == sig_raw);
            }
            if (match) {
                mobility_hint = build_mobility_hint(profile);
                have_mobility_hint = true;
                mobility_profile_match = true;
            }
        }
        if (have_mobility_hint && !mobility_hint.sweep_order.empty()) {
            tuned.sweep_order_override = mobility_hint.sweep_order;
        }
        if (have_mobility_hint && mobility_hint.force_symmetric &&
            options_.absorption_method == AbsorptionMethod::Auto &&
            !options_.symmetric_sweep) {
            tuned.symmetric_sweep = true;
        }

        bool allow_profile_write =
            !has_slopes_use &&
            (mobility_cfg.mode == "write" ||
             (mobility_cfg.mode == "auto" && mobility_cfg.allow_auto_write && !mobility_profile_match));
        bool allow_cache_read = false;
        bool allow_cache_write = false;
        std::string cache_path;
        if (abs_cache_cfg.mode_set) {
            if (abs_cache_cfg.mode != "off") {
                if (!abs_cache_cfg.path.empty()) {
                    cache_path = abs_cache_cfg.path;
                } else if (mobility_cfg.mode != "off") {
                    cache_path = absorption_cache_path(mobility_cfg.path);
                }
                allow_cache_read =
                    abs_cache_cfg.mode == "read" || abs_cache_cfg.mode == "auto";
                allow_cache_write =
                    abs_cache_cfg.mode == "write" ||
                    (abs_cache_cfg.mode == "auto" && abs_cache_cfg.allow_auto_write);
                if (cache_path.empty()) {
                    allow_cache_read = false;
                    allow_cache_write = false;
                }
            }
        } else if (abs_cache_cfg.mode != "off" && !abs_cache_cfg.path.empty()) {
            allow_cache_read =
                abs_cache_cfg.mode == "read" || abs_cache_cfg.mode == "auto";
            allow_cache_write =
                abs_cache_cfg.mode == "write" ||
                (abs_cache_cfg.mode == "auto" && abs_cache_cfg.allow_auto_write);
            cache_path = abs_cache_cfg.path;
        } else {
            allow_cache_read =
                mobility_cfg.mode == "read" || mobility_cfg.mode == "auto";
            allow_cache_write =
                mobility_cfg.mode == "write" ||
                (mobility_cfg.mode == "auto" && mobility_cfg.allow_auto_write);
            cache_path = absorption_cache_path(mobility_cfg.path);
        }
        if (has_slopes_use) {
            allow_cache_read = false;
            allow_cache_write = false;
            cache_path.clear();
        }
        AbsorptionCacheKey abs_cache_key;
        bool abs_cache_key_ready = false;
        bool cache_hit = false;

        if (have_mobility_hint && options_.absorption_method == AbsorptionMethod::Auto &&
            !options_.use_krylov && !options_.use_sparse_solver) {
            if (mobility_hint.use_krylov) {
                tuned.use_krylov = true;
                tuned.use_sparse_solver = false;
            } else if (mobility_hint.use_sparse) {
                tuned.use_sparse_solver = true;
                tuned.use_krylov = false;
            } else {
                tuned.use_krylov = false;
                tuned.use_sparse_solver = false;
            }
        }
        if (env_krylov.has_value()) {
            tuned.use_krylov = *env_krylov;
            if (tuned.use_krylov) {
                tuned.use_sparse_solver = false;
            }
        }

        bool benchmark_methods =
            allow_profile_write && !have_mobility_hint &&
            options_.absorption_method == AbsorptionMethod::Auto &&
            !options_.symmetric_sweep && !options_.use_sparse_solver &&
            !options_.use_krylov && fes_use.size() > 1;

        AbsorptionMethod preferred_method = AbsorptionMethod::Auto;
        if (have_mobility_hint && mobility_hint.has_method) {
            preferred_method = mobility_hint.preferred_method;
        }
        AbsorptionMethod selected_method = AbsorptionMethod::Auto;
        if (options_.absorption_method != AbsorptionMethod::Auto) {
            selected_method = options_.absorption_method;
        } else if (options_.symmetric_sweep) {
            selected_method = AbsorptionMethod::SymmetricGaussSeidel;
        } else if (preferred_method != AbsorptionMethod::Auto) {
            selected_method = preferred_method;
        }
        // ---- Auto-MLSMR selection (CPU-only, standard FEs) -----------------------
        // Plain absorptionmethod(auto) now uses the same selector as the explicit
        // absorptionmethod(auto-mlsmr) alias. The selector probes standard-FE designs
        // and resolves to MLSMR only for large, slow-converging cases; otherwise it
        // resolves to the sweep fallback. Resolving the fallback here also prevents a
        // second survey by the adaptive Schwarz gate after the selector has already
        // decided to stay on a sweep. GPU, savefe, and heterogeneous-slope cases cannot
        // use MLSMR, so they resolve directly to the sweep fallback.
        {
            const bool explicit_auto_mlsmr =
                (options_.absorption_method == AbsorptionMethod::AutoMlsmr);
            const bool default_auto_promote =
                (options_.absorption_method == AbsorptionMethod::Auto) &&
                (selected_method == AbsorptionMethod::Auto);
            // GUARD: 4+ FE default-auto stays on the full-data gate/MAP path. The
            // 200k-sample rho probe over-promotes huge, well-connected 4-way
            // graphs to MLSMR (simulated_panel 173M: full-data 7 GS iters but
            // sample rho=0.79 -> MLSMR ~46x). Explicit absorptionmethod(auto-mlsmr)
            // is uncapped. Tune the cap with XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES.
            const int default_promote_max_fes =
                read_env_int("XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES", 3, 1);
            const bool default_auto_promote_eff =
                default_auto_promote &&
                static_cast<int>(fes_use.size()) <= default_promote_max_fes;
            const bool use_auto_mlsmr_selector =
                explicit_auto_mlsmr || default_auto_promote_eff;
            if (use_auto_mlsmr_selector && !has_slopes_use &&
                !options_.retain_fixed_effects && !gpu_backend_env_requested()) {
                const AbsorptionMethod sweep_fb =
                    auto_mlsmr_sweep_fallback(fes_use.size(), options_.symmetric_sweep);
                const int rhs_count = static_cast<int>(X_use.cols());
                const AbsorptionMethod picked = select_auto_mlsmr_method(
                    fes_use, w_ptr, options_, rhs_count, sweep_fb);
                if (picked == AbsorptionMethod::Mlsmr) {
                    selected_method = AbsorptionMethod::Mlsmr;
                } else {
                    selected_method = sweep_fb;
                }
                tuned.from_auto = false;
                benchmark_methods = false;
            } else if (use_auto_mlsmr_selector) {
                // GPU-requested or ineligible auto selector -> normal sweep selector.
                selected_method =
                    auto_mlsmr_sweep_fallback(fes_use.size(), options_.symmetric_sweep);
                tuned.from_auto = false;
                benchmark_methods = false;
            }
        }
        if (selected_method == AbsorptionMethod::Auto && fes_use.size() == 2 &&
            y_use.size() < 50000) {
            const bool has_inst_use = inst_use && inst_use->cols() > 0;
            if (const auto fast_method = select_two_fe_auto_fastpath(
                    fes_use, w_ptr, options_, gpu_backend_env_requested(), has_inst_use)) {
                selected_method = *fast_method;
            }
        }

        detail::AbsorptionResult absorption;
        const auto absorption_t0 = std::chrono::steady_clock::now();
        bool absorption_ready = false;
        auto ensure_cache_key = [&](const Eigen::Ref<const Eigen::MatrixXd>& design) {
            if (abs_cache_key_ready) {
                return;
            }
            abs_cache_key = hash_absorption_signature(y_use, design, fes_use, w_ptr, options_);
            abs_cache_key_ready = true;
        };
        auto try_load_cache = [&](const Eigen::Ref<const Eigen::MatrixXd>& design,
                                  int design_cols) -> bool {
            if (has_slopes_use || !allow_cache_read || cache_path.empty()) {
                return false;
            }
            ensure_cache_key(design);
            AbsorptionCacheRecord cached;
            if (!read_absorption_cache(cache_path, abs_cache_key, cached)) {
                return false;
            }
            if (!cached.converged) {
                return false;
            }
            if (cached.nobs != y_use.size() ||
                cached.cols != design.cols() ||
                cached.design_cols != design_cols) {
                return false;
            }
            if (static_cast<int>(cached.fe_levels.size()) !=
                static_cast<int>(fes_use.size())) {
                return false;
            }
            absorption.y_tilde = std::move(cached.y_tilde);
            absorption.X_tilde = std::move(cached.X_tilde);
            absorption.fe_levels = std::move(cached.fe_levels);
            absorption.sweep_order_used = std::move(cached.sweep_order);
            absorption.iterations = cached.iterations;
            absorption.converged = cached.converged;
            if (gpu_backend_env_requested()) {
                absorption.gpu_status_code = 5;
                absorption.gpu_attempted = false;
                absorption.gpu_absorption_converged = false;
                absorption.gpu_absorption_iterations = 0;
            }
            method_used_ = cached.method;
            cache_hit = true;
            return true;
        };
        auto benchmark_absorption = [&](const Eigen::Ref<const Eigen::MatrixXd>& design,
                                        AbsorptionMethod& best_method,
                                        bool& best_use_sparse,
                                        bool& best_use_krylov,
                                        detail::AbsorptionResult& best_absorption) -> bool {
            struct AbsorptionCandidate {
                AbsorptionMethod method;
                bool use_sparse;
                bool use_krylov;
                bool symmetric;
            };

            std::vector<AbsorptionCandidate> candidates;
            candidates.reserve(5);
            candidates.push_back({AbsorptionMethod::GaussSeidel, false, false, false});
            if (fes_use.size() > 1) {
                candidates.push_back({AbsorptionMethod::SymmetricGaussSeidel, false, false, true});
            }
            if (fes_use.size() > 1 && threads > 1) {
                candidates.push_back({AbsorptionMethod::Jacobi, false, false, false});
            }
            candidates.push_back({AbsorptionMethod::GaussSeidel, true, false, false});
            candidates.push_back({AbsorptionMethod::GaussSeidel, false, true, false});

            double best_time = std::numeric_limits<double>::infinity();
            bool have_best = false;
            for (const auto& cand : candidates) {
                HdfeOptions bench_opts = tuned;
                bench_opts.absorption_method = cand.method;
                bench_opts.symmetric_sweep = cand.symmetric;
                bench_opts.use_sparse_solver = cand.use_sparse;
                bench_opts.use_krylov = cand.use_krylov;
                if (bench_opts.use_krylov) {
                    bench_opts.use_sparse_solver = false;
                }

                const auto t0 = std::chrono::steady_clock::now();
                detail::AbsorptionResult res =
                    detail::absorb_fixed_effects_v6(y_use, design, fes_use, w_ptr, bench_opts,
                                                    cand.method, slope_terms);
                const auto t1 = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();

                if (!res.converged) {
                    continue;
                }
                if (elapsed < best_time) {
                    best_time = elapsed;
                    best_method = cand.method;
                    best_use_sparse = cand.use_sparse;
                    best_use_krylov = cand.use_krylov;
                    best_absorption = std::move(res);
                    have_best = true;
                }
            }
            return have_best;
        };

        int design_cols = 0;
        if (drop_intercept && !inst_use) {
            design_cols = static_cast<int>(X_use.cols());
            if (!absorption_ready && try_load_cache(X_use, design_cols)) {
                absorption_ready = true;
                allow_profile_write = false;
            }
            if (!absorption_ready && benchmark_methods) {
                AbsorptionMethod best_method = AbsorptionMethod::Auto;
                bool best_use_sparse = false;
                bool best_use_krylov = false;
                if (benchmark_absorption(X_use, best_method, best_use_sparse, best_use_krylov, absorption)) {
                    method_used_ = best_method;
                    tuned.absorption_method = best_method;
                    tuned.symmetric_sweep = (best_method == AbsorptionMethod::SymmetricGaussSeidel);
                    tuned.use_sparse_solver = best_use_sparse;
                    tuned.use_krylov = best_use_krylov;
                    if (tuned.use_krylov) {
                        tuned.use_sparse_solver = false;
                    }
                    absorption_ready = true;
                }
            }
            if (!absorption_ready) {
                if (selected_method == AbsorptionMethod::Auto) {
                    selected_method = select_method(fes_use.size());
                }
                method_used_ = selected_method;
                if (has_slopes_use && method_used_ == AbsorptionMethod::Jacobi) {
                    method_used_ = fes_use.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                                       : AbsorptionMethod::GaussSeidel;
                }
                if (tuned.symmetric_sweep &&
                    options_.absorption_method == AbsorptionMethod::Auto &&
                    method_used_ != AbsorptionMethod::SymmetricGaussSeidel &&
                    fes_use.size() > 1) {
                    method_used_ = AbsorptionMethod::SymmetricGaussSeidel;
                }
                tuned.absorption_method = method_used_;
                absorption = detail::absorb_fixed_effects_v6(y_use, X_use, fes_use, w_ptr, tuned,
                                                             method_used_, slope_terms);
                if (absorption.schwarz_used) {
                    method_used_ = AbsorptionMethod::Schwarz;  // adaptive gate diverted to Schwarz
                } else if (absorption.mlsmr_used) {
                    method_used_ = AbsorptionMethod::Mlsmr;     // adaptive gate diverted to MLSMR
                }
            }
            if (allow_cache_write && !cache_hit && absorption.converged && !cache_path.empty()) {
                ensure_cache_key(X_use);
                write_absorption_cache(cache_path, abs_cache_key, absorption.y_tilde,
                                       absorption.X_tilde, absorption.fe_levels,
                                       absorption.sweep_order_used, design_cols, method_used_,
                                       absorption.iterations, absorption.converged);
            }
        } else {
            if (drop_intercept && inst_use && inst_use->cols() > 0) {
                design_cols = static_cast<int>(X_use.cols());
                const Eigen::MatrixXd combined = append_matrix(X_use, inst_use);
                if (!absorption_ready && try_load_cache(combined, design_cols)) {
                    absorption_ready = true;
                    allow_profile_write = false;
                }
                if (!absorption_ready && benchmark_methods) {
                    AbsorptionMethod best_method = AbsorptionMethod::Auto;
                    bool best_use_sparse = false;
                    bool best_use_krylov = false;
                    if (benchmark_absorption(combined, best_method, best_use_sparse, best_use_krylov,
                                             absorption)) {
                        method_used_ = best_method;
                        tuned.absorption_method = best_method;
                        tuned.symmetric_sweep = (best_method == AbsorptionMethod::SymmetricGaussSeidel);
                        tuned.use_sparse_solver = best_use_sparse;
                        tuned.use_krylov = best_use_krylov;
                        if (tuned.use_krylov) {
                            tuned.use_sparse_solver = false;
                        }
                        absorption_ready = true;
                    }
                }
                if (!absorption_ready) {
                    if (selected_method == AbsorptionMethod::Auto) {
                        selected_method = select_method(fes_use.size());
                    }
                    method_used_ = selected_method;
                    if (has_slopes_use && method_used_ == AbsorptionMethod::Jacobi) {
                        method_used_ = fes_use.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                                           : AbsorptionMethod::GaussSeidel;
                    }
                    if (tuned.symmetric_sweep &&
                        options_.absorption_method == AbsorptionMethod::Auto &&
                        method_used_ != AbsorptionMethod::SymmetricGaussSeidel &&
                        fes_use.size() > 1) {
                        method_used_ = AbsorptionMethod::SymmetricGaussSeidel;
                    }
                    tuned.absorption_method = method_used_;
                    absorption = detail::absorb_fixed_effects_v6(y_use, combined, fes_use, w_ptr, tuned,
                                                                 method_used_, slope_terms);
                    if (absorption.schwarz_used) {
                        method_used_ = AbsorptionMethod::Schwarz;
                    } else if (absorption.mlsmr_used) {
                        method_used_ = AbsorptionMethod::Mlsmr;
                    }
                }
                if (allow_cache_write && !cache_hit && absorption.converged && !cache_path.empty()) {
                    ensure_cache_key(combined);
                    write_absorption_cache(cache_path, abs_cache_key, absorption.y_tilde,
                                           absorption.X_tilde, absorption.fe_levels,
                                           absorption.sweep_order_used, design_cols, method_used_,
                                           absorption.iterations, absorption.converged);
                }
            } else {
                Eigen::MatrixXd design =
                    drop_intercept ? Eigen::MatrixXd(X_use)
                                   : maybe_add_intercept(X_use, options_.fit_intercept);
                design_cols = static_cast<int>(design.cols());
                if (!inst_use || inst_use->cols() == 0) {
                    if (!absorption_ready && try_load_cache(design, design_cols)) {
                        absorption_ready = true;
                        allow_profile_write = false;
                    }
                    if (!absorption_ready && benchmark_methods) {
                        AbsorptionMethod best_method = AbsorptionMethod::Auto;
                        bool best_use_sparse = false;
                        bool best_use_krylov = false;
                        if (benchmark_absorption(design, best_method, best_use_sparse, best_use_krylov,
                                                 absorption)) {
                            method_used_ = best_method;
                            tuned.absorption_method = best_method;
                            tuned.symmetric_sweep =
                                (best_method == AbsorptionMethod::SymmetricGaussSeidel);
                            tuned.use_sparse_solver = best_use_sparse;
                            tuned.use_krylov = best_use_krylov;
                            if (tuned.use_krylov) {
                                tuned.use_sparse_solver = false;
                            }
                            absorption_ready = true;
                        }
                    }
                    if (!absorption_ready) {
                        if (selected_method == AbsorptionMethod::Auto) {
                            selected_method = select_method(fes_use.size());
                        }
                        method_used_ = selected_method;
                        if (has_slopes_use && method_used_ == AbsorptionMethod::Jacobi) {
                            method_used_ = fes_use.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                                               : AbsorptionMethod::GaussSeidel;
                        }
                        if (tuned.symmetric_sweep &&
                            options_.absorption_method == AbsorptionMethod::Auto &&
                            method_used_ != AbsorptionMethod::SymmetricGaussSeidel &&
                            fes_use.size() > 1) {
                            method_used_ = AbsorptionMethod::SymmetricGaussSeidel;
                        }
                        tuned.absorption_method = method_used_;
                        absorption = detail::absorb_fixed_effects_v6(y_use, design, fes_use, w_ptr, tuned,
                                                                     method_used_, slope_terms);
                        if (absorption.schwarz_used) {
                            method_used_ = AbsorptionMethod::Schwarz;
                        } else if (absorption.mlsmr_used) {
                            method_used_ = AbsorptionMethod::Mlsmr;
                        }
                    }
                    if (allow_cache_write && !cache_hit && absorption.converged && !cache_path.empty()) {
                        ensure_cache_key(design);
                        write_absorption_cache(cache_path, abs_cache_key, absorption.y_tilde,
                                               absorption.X_tilde, absorption.fe_levels,
                                               absorption.sweep_order_used, design_cols, method_used_,
                                               absorption.iterations, absorption.converged);
                    }
                } else {
                    const Eigen::MatrixXd combined = append_matrix(design, inst_use);
                    if (!absorption_ready && try_load_cache(combined, design_cols)) {
                        absorption_ready = true;
                        allow_profile_write = false;
                    }
                    if (!absorption_ready && benchmark_methods) {
                        AbsorptionMethod best_method = AbsorptionMethod::Auto;
                        bool best_use_sparse = false;
                        bool best_use_krylov = false;
                        if (benchmark_absorption(combined, best_method, best_use_sparse, best_use_krylov,
                                                 absorption)) {
                            method_used_ = best_method;
                            tuned.absorption_method = best_method;
                            tuned.symmetric_sweep =
                                (best_method == AbsorptionMethod::SymmetricGaussSeidel);
                            tuned.use_sparse_solver = best_use_sparse;
                            tuned.use_krylov = best_use_krylov;
                            if (tuned.use_krylov) {
                                tuned.use_sparse_solver = false;
                            }
                            absorption_ready = true;
                        }
                    }
                    if (!absorption_ready) {
                        if (selected_method == AbsorptionMethod::Auto) {
                            selected_method = select_method(fes_use.size());
                        }
                        method_used_ = selected_method;
                        if (has_slopes_use && method_used_ == AbsorptionMethod::Jacobi) {
                            method_used_ = fes_use.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                                               : AbsorptionMethod::GaussSeidel;
                        }
                        if (tuned.symmetric_sweep &&
                            options_.absorption_method == AbsorptionMethod::Auto &&
                            method_used_ != AbsorptionMethod::SymmetricGaussSeidel &&
                            fes_use.size() > 1) {
                            method_used_ = AbsorptionMethod::SymmetricGaussSeidel;
                        }
                        tuned.absorption_method = method_used_;
                        absorption = detail::absorb_fixed_effects_v6(y_use, combined, fes_use, w_ptr, tuned,
                                                                     method_used_, slope_terms);
                        if (absorption.schwarz_used) {
                            method_used_ = AbsorptionMethod::Schwarz;
                        } else if (absorption.mlsmr_used) {
                            method_used_ = AbsorptionMethod::Mlsmr;
                        }
                    }
                    if (allow_cache_write && !cache_hit && absorption.converged && !cache_path.empty()) {
                        ensure_cache_key(combined);
                        write_absorption_cache(cache_path, abs_cache_key, absorption.y_tilde,
                                               absorption.X_tilde, absorption.fe_levels,
                                               absorption.sweep_order_used, design_cols, method_used_,
                                               absorption.iterations, absorption.converged);
                    }
                }
            }
        }

        cpu_profile_log_elapsed("absorption", absorption_t0);
        gpu_used_ = absorption.gpu_used;
        gpu_status_code_ = absorption.gpu_status_code;
        gpu_attempted_ = absorption.gpu_attempted;
        gpu_absorption_converged_ = absorption.gpu_absorption_converged;
        gpu_absorption_iterations_ = absorption.gpu_absorption_iterations;
        if (!absorption.converged) {
            if (absorption.gpu_status_code == 2) {
                throw std::runtime_error("Requested GPU backend was unavailable during HDFE absorption");
            }
            if (absorption.gpu_status_code == 3) {
                throw std::runtime_error("Requested GPU backend did not converge during HDFE absorption");
            }
            if (absorption.gpu_status_code == 4) {
                throw std::runtime_error("Requested GPU backend failed during HDFE absorption");
            }
            if (absorption.gpu_status_code == 5) {
                throw std::runtime_error("Requested GPU backend was bypassed by CPU cache/profile state");
            }
            // Pure CPU non-convergence (no explicit-GPU request): preserve the reference
            // best-effort behaviour and surface the status through converged_/e(converged)
            // rather than hard-failing the command. Explicit-GPU failures above still throw,
            // since the contract forbids a silent CPU result under a requested GPU backend.
        }
        const auto stats_t0 = std::chrono::steady_clock::now();
        double sum_weights_for_stats = static_cast<double>(y_use.size());
        // Stata/reghdfe convention: the total SS is centered only when the
        // fitted model contains an intercept somewhere — either as an actual
        // regressor column, or absorbed through at least one FE level
        // dimension. Slope-only absorption (absorb(fe#c.z)) and noconstant
        // fits have no intercept, so the uncentered TSS is used (this is what
        // regress, noconstant and reghdfe report).
        bool any_absorbed_level = false;
        if (!fes_use.empty()) {
            std::vector<char> slope_only_dim(fes_use.size(), 0);
            for (const auto& s : slope_terms) {
                if (!s.include_intercept && s.fe_index >= 0 &&
                    s.fe_index < static_cast<int>(fes_use.size())) {
                    slope_only_dim[static_cast<std::size_t>(s.fe_index)] = 1;
                }
            }
            for (std::size_t d = 0; d < fes_use.size(); ++d) {
                if (!slope_only_dim[d]) {
                    any_absorbed_level = true;
                    break;
                }
            }
        }
        const bool model_has_intercept =
            (options_.fit_intercept && !drop_intercept) || any_absorbed_level;
        double total_tss = 0.0;
        if (model_has_intercept) {
            total_tss = compute_total_sum_of_squares(y_use, w_ptr, &sum_weights_for_stats);
        } else {
            total_tss = weighted_sum_of_squares(y_use, w_ptr);
            sum_weights_for_stats =
                w_ptr ? w_ptr->sum() : static_cast<double>(y_use.size());
        }
        const double within_tss = fes_use.empty()
                                      ? total_tss
                                      : weighted_sum_of_squares(absorption.y_tilde, w_ptr);
        cpu_profile_log_elapsed("fit_stats", stats_t0);
        if (read_env_bool("XHDFE_DEBUG_SOLVER").value_or(false)) {
            std::cerr << "xhdfe solver: use_krylov=" << (tuned.use_krylov ? 1 : 0)
                      << " use_sparse=" << (tuned.use_sparse_solver ? 1 : 0)
                      << " method=" << static_cast<int>(method_used_)
                      << " iterations=" << absorption.iterations << "\n";
        }

        auto transformed_X = absorption.X_tilde.leftCols(design_cols);
        const int transformed_full_cols = static_cast<int>(transformed_X.cols());
        const bool transformed_has_intercept =
            (options_.fit_intercept && !drop_intercept && transformed_full_cols > 0);
        const int transformed_slope_cols =
            transformed_has_intercept ? (transformed_full_cols - 1) : transformed_full_cols;

        // Stata/reghdfe-style collinearity with absorbed fixed effects:
        // after partialling-out, regressors with near-zero variation are omitted.
        constexpr double kFeCollinearTol = 1e-9;
        const double rank_collinear_tol = std::max(1e-6, tuned.tol * 10.0);
        std::vector<int> kept_slope_cols;
        std::vector<int> omitted_slope_cols;
        std::vector<int> omitted_fe_slope_cols;
        std::vector<int> omitted_rank_slope_cols;
        kept_slope_cols.reserve(static_cast<std::size_t>(std::max(0, transformed_slope_cols)));
        omitted_slope_cols.reserve(static_cast<std::size_t>(std::max(0, transformed_slope_cols)));
        const auto collinearity_t0 = std::chrono::steady_clock::now();
        for (int j = 0; j < transformed_slope_cols; ++j) {
            const double raw_ss = X_use.col(j).squaredNorm();
            const double tilde_ss = transformed_X.col(j).squaredNorm();
            const bool finite = std::isfinite(raw_ss) && std::isfinite(tilde_ss);
            const bool collinear =
                finite && (raw_ss <= 0.0 ? (tilde_ss <= kFeCollinearTol)
                                         : (tilde_ss <= kFeCollinearTol * raw_ss));
            if (collinear) {
                omitted_fe_slope_cols.push_back(j);
                omitted_slope_cols.push_back(j);
            } else {
                kept_slope_cols.push_back(j);
            }
        }

        Eigen::MatrixXd xtx_used;
        Eigen::VectorXd xty_used;
        bool have_precomputed = false;

        // Detect remaining linear dependencies among kept slopes (including intercept collinearity).
        const bool use_precomputed = (!wants_iv && tuned.se_type != StandardErrorType::Cluster);
        if (use_precomputed) {
            const int k_initial = static_cast<int>(kept_slope_cols.size());
            const int intercept_col = transformed_has_intercept ? (transformed_full_cols - 1) : -1;
            const CrossproductResult cp = compute_crossproducts_selected(
                absorption.y_tilde, transformed_X, kept_slope_cols, intercept_col, w_ptr,
                options_.fit_intercept, tuned.num_threads);

            std::vector<int> keep_positions;
            keep_positions.reserve(kept_slope_cols.size());
            if (k_initial > 0) {
                Eigen::MatrixXd gram = cp.xtx.topLeftCorner(k_initial, k_initial);
                if (options_.fit_intercept) {
                    double denom = cp.sum_w;
                    if (!(denom > 0.0)) {
                        denom = 1.0;
                    }
                    gram.noalias() -= (cp.sum_x * cp.sum_x.transpose()) / denom;
                }
                const std::vector<uint8_t> drop_mask =
                    compute_rank_collinearity_mask_from_gram(
                        gram, kept_slope_cols, rank_collinear_tol, options_.collinear_priority);

                std::vector<int> new_kept;
                std::vector<int> new_omitted;
                new_kept.reserve(kept_slope_cols.size());
                new_omitted.reserve(kept_slope_cols.size());
                for (std::size_t pos = 0; pos < kept_slope_cols.size(); ++pos) {
                    const int idx = kept_slope_cols[pos];
                    if (drop_mask[static_cast<std::size_t>(pos)]) {
                        new_omitted.push_back(idx);
                    } else {
                        new_kept.push_back(idx);
                        keep_positions.push_back(static_cast<int>(pos));
                    }
                }
                kept_slope_cols.swap(new_kept);
                omitted_rank_slope_cols.insert(omitted_rank_slope_cols.end(),
                                               new_omitted.begin(), new_omitted.end());
                omitted_slope_cols.insert(omitted_slope_cols.end(),
                                          new_omitted.begin(), new_omitted.end());
            }

            std::vector<int> xtx_cols = keep_positions;
            if (intercept_col >= 0) {
                xtx_cols.push_back(k_initial);
            }
            const int p_used = static_cast<int>(xtx_cols.size());
            xtx_used.resize(p_used, p_used);
            xty_used.resize(p_used);
            for (int i = 0; i < p_used; ++i) {
                const int col_i = xtx_cols[static_cast<std::size_t>(i)];
                xty_used(i) = cp.xty(col_i);
                for (int j = 0; j < p_used; ++j) {
                    const int col_j = xtx_cols[static_cast<std::size_t>(j)];
                    xtx_used(i, j) = cp.xtx(col_i, col_j);
                }
            }
            have_precomputed = true;
        } else if (!kept_slope_cols.empty()) {
            const std::vector<uint8_t> drop_mask = compute_rank_collinearity_mask(
                transformed_X, kept_slope_cols, transformed_slope_cols, options_.fit_intercept,
                w_ptr, rank_collinear_tol, tuned.num_threads, options_.collinear_priority);

            std::vector<int> new_kept;
            std::vector<int> new_omitted;
            new_kept.reserve(kept_slope_cols.size());
            new_omitted.reserve(kept_slope_cols.size());
            for (const int idx : kept_slope_cols) {
                if (idx >= 0 && idx < transformed_slope_cols &&
                    drop_mask[static_cast<std::size_t>(idx)]) {
                    new_omitted.push_back(idx);
                } else {
                    new_kept.push_back(idx);
                }
            }
            kept_slope_cols.swap(new_kept);
            omitted_rank_slope_cols.insert(omitted_rank_slope_cols.end(),
                                           new_omitted.begin(), new_omitted.end());
            omitted_slope_cols.insert(omitted_slope_cols.end(),
                                      new_omitted.begin(), new_omitted.end());
        }
        cpu_profile_log_elapsed("collinearity", collinearity_t0);

        // Build the design matrix actually used in OLS/IV:
        // (kept slopes) [+ intercept column if present].
        const auto design_t0 = std::chrono::steady_clock::now();
        const int design_used_cols = static_cast<int>(kept_slope_cols.size()) +
                                     (transformed_has_intercept ? 1 : 0);
        bool design_identity = (design_used_cols == transformed_full_cols);
        if (design_identity) {
            for (std::size_t j = 0; j < kept_slope_cols.size(); ++j) {
                if (kept_slope_cols[j] != static_cast<int>(j)) {
                    design_identity = false;
                    break;
                }
            }
        }
        Eigen::MatrixXd transformed_X_used_storage;
        if (!design_identity) {
            transformed_X_used_storage.resize(transformed_X.rows(), design_used_cols);
            for (int out_j = 0; out_j < static_cast<int>(kept_slope_cols.size()); ++out_j) {
                transformed_X_used_storage.col(out_j) =
                    transformed_X.col(kept_slope_cols[static_cast<std::size_t>(out_j)]);
            }
            if (transformed_has_intercept) {
                transformed_X_used_storage.col(static_cast<int>(kept_slope_cols.size())) =
                    transformed_X.col(transformed_full_cols - 1);
            }
        }
        // When no column was dropped or reordered, the copy above would
        // reproduce transformed_X element-for-element, so alias it instead.
        // transformed_X views absorption.X_tilde, which is only read for the
        // remainder of this fit.
        const Eigen::Ref<const Eigen::MatrixXd> transformed_X_used =
            design_identity ? Eigen::Ref<const Eigen::MatrixXd>(transformed_X)
                            : Eigen::Ref<const Eigen::MatrixXd>(transformed_X_used_storage);
        cpu_profile_log_elapsed("design_copy", design_t0);

        const int df_m_effective = static_cast<int>(kept_slope_cols.size());
        const double nobs_effective =
            (tuned.weights_are_frequencies && w_ptr) ? w_ptr->sum()
                                                     : static_cast<double>(y_use.size());

        detail::OlsResult ols_result;
        const auto ols_t0 = std::chrono::steady_clock::now();
        if (!wants_iv) {
            if (transformed_X_used.cols() == 0) {
                // Degenerate case: all regressors were collinear after partialling-out.
                // Return an intercept-only fit downstream (intercept computed later when drop_intercept=true).
                detail::OlsResult empty;
                empty.coefficients.resize(0);
                empty.std_errors.resize(0);
                empty.tvalues.resize(0);
                empty.pvalues.resize(0);
                empty.conf_int.resize(0, 2);
                empty.residuals = absorption.y_tilde;
                empty.df_resid = nobs_effective;
                empty.rss = weighted_sum_of_squares(empty.residuals, w_ptr);
                empty.tss = total_tss;
                empty.within_tss = within_tss;
                empty.r2 = (total_tss > 0.0) ? 1.0 - empty.rss / total_tss : 1.0;
                empty.r2_within = (within_tss > 0.0) ? 1.0 - empty.rss / within_tss : 1.0;
                empty.sigma2 = 0.0;
                empty.nobs = static_cast<int>(absorption.y_tilde.size());
                ols_result = std::move(empty);
            } else if (tuned.se_type == StandardErrorType::Cluster) {
                if (!c_ptr || c_ptr->empty()) {
                    throw std::runtime_error(
                        "Cluster-robust errors requested but no cluster variables were provided");
                }
                if (c_ptr->size() == 1) {
                    ols_result = detail::run_ols(absorption.y_tilde, transformed_X_used, w_ptr, &(*c_ptr)[0],
                                                 tuned.se_type, total_tss, within_tss, nobs_effective);
                } else {
                    ols_result = detail::run_ols_multiway(absorption.y_tilde, transformed_X_used, w_ptr, c_ptr,
                                                         tuned.se_type, total_tss, within_tss,
                                                         tuned.ssc_g_df, tuned.ssc_g_adj, nobs_effective);
                }
            } else if (have_precomputed &&
                       xtx_used.rows() == transformed_X_used.cols() &&
                       xty_used.size() == transformed_X_used.cols()) {
                ols_result = detail::run_ols_fast_from_xtx(absorption.y_tilde, transformed_X_used, w_ptr,
                                                           tuned.se_type, total_tss, within_tss,
                                                           xtx_used, xty_used, nobs_effective,
                                                           tuned.weights_are_frequencies);
            } else {
                ols_result = detail::run_ols(absorption.y_tilde, transformed_X_used, w_ptr, nullptr, tuned.se_type,
                                             total_tss, within_tss, nobs_effective,
                                             tuned.weights_are_frequencies);
            }
        } else {
            if (!inst_use || inst_use->cols() == 0) {
                throw std::runtime_error("IV requested but no instruments were provided");
            }
            Eigen::MatrixXd instrument_slice = absorption.X_tilde.rightCols(inst_use->cols());
            std::vector<int> cleaned =
                sanitize_endogenous_idx(endogenous_idx, transformed_slope_cols);
            if (cleaned.empty()) {
                throw std::runtime_error(
                    "At least one endogenous index is required for IV estimation");
            }
            std::vector<int> slope_map(static_cast<std::size_t>(transformed_slope_cols), -1);
            for (int pos = 0; pos < static_cast<int>(kept_slope_cols.size()); ++pos) {
                slope_map[static_cast<std::size_t>(kept_slope_cols[static_cast<std::size_t>(pos)])] = pos;
            }
            std::vector<int> active_endog_cols;
            active_endog_cols.reserve(cleaned.size());
            for (const int idx : cleaned) {
                if (idx < 0 || idx >= transformed_slope_cols) {
                    throw std::runtime_error("Endogenous index out of bounds");
                }
                const int mapped = slope_map[static_cast<std::size_t>(idx)];
                if (mapped >= 0) {
                    active_endog_cols.push_back(mapped);
                }
            }
            if (active_endog_cols.empty()) {
                throw std::runtime_error(
                    "All endogenous regressors are collinear after partialling-out");
            }
            Eigen::MatrixXd X_endog = select_columns(transformed_X_used, active_endog_cols);
            Eigen::MatrixXd X_exog = drop_columns(transformed_X_used, active_endog_cols);
            Eigen::MatrixXd instrument_matrix(instrument_slice.rows(),
                                              X_exog.cols() + instrument_slice.cols());
            int cursor = 0;
            if (X_exog.cols() > 0) {
                instrument_matrix.block(0, 0, X_exog.rows(), X_exog.cols()) = X_exog;
                cursor = X_exog.cols();
            }
            if (instrument_slice.cols() > 0) {
                instrument_matrix.block(0, cursor, instrument_slice.rows(),
                                        instrument_slice.cols()) = instrument_slice;
            }
            if (instrument_matrix.cols() == 0) {
                throw std::runtime_error(
                    "Instrument matrix is empty after combining exogenous regressors and user-supplied instruments");
            }
            Eigen::MatrixXd projected =
                detail::project_endogenous(instrument_matrix, X_endog,
                                           static_cast<int>(X_exog.cols()), w_ptr);
            Eigen::MatrixXd second_stage = transformed_X_used;
            for (int idx = 0; idx < static_cast<int>(active_endog_cols.size()); ++idx) {
                second_stage.col(active_endog_cols[static_cast<std::size_t>(idx)]) = projected.col(idx);
            }
            // 2SLS inference must use residuals from the ACTUAL regressors
            // (u = y - X b), not the projected second-stage design; the solve
            // itself stays on the instrumented design.
            const Eigen::MatrixXd iv_actual_X = transformed_X_used;
            if (tuned.se_type == StandardErrorType::Cluster) {
                if (!c_ptr || c_ptr->empty()) {
                    throw std::runtime_error(
                        "Cluster-robust errors requested but no cluster variables were provided");
                }
                if (c_ptr->size() == 1) {
                    ols_result = detail::run_ols(absorption.y_tilde, second_stage, w_ptr, &(*c_ptr)[0],
                                                 tuned.se_type, total_tss, within_tss, nobs_effective,
                                                 false, &iv_actual_X);
                } else {
                    ols_result = detail::run_ols_multiway(absorption.y_tilde, second_stage, w_ptr, c_ptr,
                                                         tuned.se_type, total_tss, within_tss,
                                                         tuned.ssc_g_df, tuned.ssc_g_adj, nobs_effective,
                                                         &iv_actual_X);
                }
            } else {
                ols_result = detail::run_ols(absorption.y_tilde, second_stage, w_ptr, nullptr, tuned.se_type,
                                             total_tss, within_tss, nobs_effective,
                                             tuned.weights_are_frequencies, &iv_actual_X);
            }
        }
        cpu_profile_log_elapsed("ols", ols_t0);

        const auto post_t0 = std::chrono::steady_clock::now();
        auto post_phase_t0 = std::chrono::steady_clock::now();
        apply_common_postprocessing(y_use, X_use, w_ptr, absorption.fe_levels, ols_result);
        cpu_profile_log_elapsed("post_common", post_phase_t0);
        results_.nobs_effective = nobs_effective;
        results_.nobs_full_effective = nobs_full_effective;
        results_.num_singletons_effective = std::max(0.0, nobs_full_effective - nobs_effective);
        results_.vcv_psd_fixed = false;
        results_.num_iterations = absorption.iterations;
        results_.converged = absorption.converged;
        if (allow_profile_write) {
            MobilityProfile profile = compute_mobility_profile(
                fes_use, nobs_full, singletons_dropped_rows, options_.drop_singletons,
                absorption.sweep_order_used);
            if (options_.absorption_method == AbsorptionMethod::Auto &&
                !options_.symmetric_sweep) {
                profile.suggest_method = method_used_;
                profile.suggest_symmetric =
                    (method_used_ == AbsorptionMethod::SymmetricGaussSeidel);
            }
            profile.suggest_use_sparse = tuned.use_sparse_solver;
            profile.suggest_use_krylov = tuned.use_krylov;
            write_mobility_profile(mobility_cfg.path, profile);
        }

        // Expand omitted regressors back to the original column layout so Python sees a stable shape.
        // Omitted slopes get coef=0 and inference entries set to NaN (Stata-style "(omitted)").
        if ((!omitted_slope_cols.empty()) &&
            results_.coefficients.size() == static_cast<int>(kept_slope_cols.size()) +
                                                 (transformed_has_intercept ? 1 : 0)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const int full_cols = transformed_full_cols;
            Eigen::VectorXd coef_full = Eigen::VectorXd::Zero(full_cols);
            Eigen::VectorXd se_full = Eigen::VectorXd::Constant(full_cols, nan);
            Eigen::VectorXd t_full = Eigen::VectorXd::Constant(full_cols, nan);
            Eigen::VectorXd p_full = Eigen::VectorXd::Constant(full_cols, nan);
            Eigen::MatrixXd ci_full = Eigen::MatrixXd::Constant(full_cols, 2, nan);
            Eigen::MatrixXd cov_full = Eigen::MatrixXd::Zero(full_cols, full_cols);
            Eigen::MatrixXd cov_used = results_.covariance;

            // Fill kept slopes.
            for (int pos = 0; pos < static_cast<int>(kept_slope_cols.size()); ++pos) {
                const int j = kept_slope_cols[static_cast<std::size_t>(pos)];
                coef_full(j) = results_.coefficients(pos);
                se_full(j) = results_.std_errors(pos);
                t_full(j) = results_.tvalues(pos);
                p_full(j) = results_.pvalues(pos);
                ci_full(j, 0) = results_.conf_int(pos, 0);
                ci_full(j, 1) = results_.conf_int(pos, 1);

                for (int pos2 = 0; pos2 < static_cast<int>(kept_slope_cols.size()); ++pos2) {
                    const int k = kept_slope_cols[static_cast<std::size_t>(pos2)];
                    if (cov_used.rows() > pos && cov_used.cols() > pos2) {
                        cov_full(j, k) = cov_used(pos, pos2);
                    }
                }
            }
            // Fill intercept if it was part of the transformed design.
            if (transformed_has_intercept) {
                const int intercept_full_idx = full_cols - 1;
                const int intercept_pos = static_cast<int>(kept_slope_cols.size());
                coef_full(intercept_full_idx) = results_.coefficients(intercept_pos);
                se_full(intercept_full_idx) = results_.std_errors(intercept_pos);
                t_full(intercept_full_idx) = results_.tvalues(intercept_pos);
                p_full(intercept_full_idx) = results_.pvalues(intercept_pos);
                ci_full(intercept_full_idx, 0) = results_.conf_int(intercept_pos, 0);
                ci_full(intercept_full_idx, 1) = results_.conf_int(intercept_pos, 1);

                if (cov_used.rows() > intercept_pos && cov_used.cols() > intercept_pos) {
                    cov_full(intercept_full_idx, intercept_full_idx) =
                        cov_used(intercept_pos, intercept_pos);
                }
                for (int pos = 0; pos < static_cast<int>(kept_slope_cols.size()); ++pos) {
                    const int j = kept_slope_cols[static_cast<std::size_t>(pos)];
                    if (cov_used.rows() > pos && cov_used.cols() > intercept_pos) {
                        cov_full(j, intercept_full_idx) = cov_used(pos, intercept_pos);
                    }
                    if (cov_used.rows() > intercept_pos && cov_used.cols() > pos) {
                        cov_full(intercept_full_idx, j) = cov_used(intercept_pos, pos);
                    }
                }
            }

            results_.coefficients = std::move(coef_full);
            results_.std_errors = std::move(se_full);
            results_.tvalues = std::move(t_full);
            results_.pvalues = std::move(p_full);
            results_.conf_int = std::move(ci_full);
            results_.covariance = std::move(cov_full);
        }

        if (drop_intercept) {
            post_phase_t0 = std::chrono::steady_clock::now();
            const int slope_cols = static_cast<int>(results_.coefficients.size());
            const double y_mean = weighted_mean(y_use, w_ptr);
            const double denom = w_ptr ? w_ptr->sum() : static_cast<double>(y_use.size());
            Eigen::VectorXd x_means = Eigen::VectorXd::Zero(slope_cols);
            if (slope_cols > 0) {
                if (!w_ptr) {
                    for (int j = 0; j < slope_cols; ++j) {
                        x_means(j) = X_use.col(j).mean();
                    }
                } else if (denom > 0.0) {
                    for (int j = 0; j < slope_cols; ++j) {
                        x_means(j) = X_use.col(j).dot(*w_ptr) / denom;
                    }
                }
            }
            const double intercept = y_mean - x_means.dot(results_.coefficients);
            const int out_cols = slope_cols + 1;
            results_.coefficients.conservativeResize(out_cols);
            results_.coefficients(out_cols - 1) = intercept;

            const double nan = std::numeric_limits<double>::quiet_NaN();
            results_.std_errors.conservativeResize(out_cols);
            results_.std_errors(out_cols - 1) = nan;
            results_.tvalues.conservativeResize(out_cols);
            results_.tvalues(out_cols - 1) = nan;
            results_.pvalues.conservativeResize(out_cols);
            results_.pvalues(out_cols - 1) = nan;
            Eigen::MatrixXd conf_new = Eigen::MatrixXd::Constant(out_cols, 2, nan);
            conf_new.topRows(slope_cols) = results_.conf_int;
            results_.conf_int = std::move(conf_new);

            Eigen::MatrixXd cov_new = Eigen::MatrixXd::Constant(out_cols, out_cols, nan);
            cov_new.topLeftCorner(slope_cols, slope_cols) = results_.covariance;
            results_.covariance = std::move(cov_new);
            cpu_profile_log_elapsed("post_intercept", post_phase_t0);

        }

        const char* fe_recovery_method = "map";
        bool used_mixed_savefe = false;
        if (options_.retain_fixed_effects && has_fes) {
            const bool profile_savefe = savefe_profile_enabled();
            const int slope_cols =
                options_.fit_intercept ? static_cast<int>(results_.coefficients.size()) - 1
                                       : static_cast<int>(results_.coefficients.size());
            Eigen::VectorXd partial;
            {
                SavefeTimer timer("savefe_partial");
                partial = y_use;
                if (slope_cols > 0) {
                    partial.noalias() -= X_use * results_.coefficients.head(slope_cols);
                }
            }
            if (!slope_terms.empty()) {
                HdfeOptions fe_opts = tuned;
                if (!absorption.sweep_order_used.empty()) {
                    fe_opts.sweep_order_override = absorption.sweep_order_used;
                }
                MixedSavefeRecoveryResult recovered;
                {
                    SavefeTimer timer("savefe_recover_mixed_map");
                    recovered = recover_mixed_savefe_effects(partial, fes_use, slope_terms,
                                                             w_ptr, fe_opts);
                }
                center_mixed_savefe_effects(recovered.contributions, recovered.save_effects,
                                            recovered.save_alpha_slot_by_dim, w_ptr);
                results_.fe_effects = std::move(recovered.contributions);
                results_.fe_save_effects = std::move(recovered.save_effects);
                results_.fe_recovery_iterations = recovered.iterations;
                results_.fe_recovery_max_delta = recovered.max_delta;
                results_.fe_recovery_converged = recovered.converged;
                fe_recovery_method = "map_mixed";
                used_mixed_savefe = true;
                if (profile_savefe) {
                    savefe_profile_log("event=fe_recovery_mixed method=map_mixed");
                }
            } else {
            const bool have_group_ids =
                !absorption.fe_group_ids.empty() &&
                absorption.fe_group_ids.size() == fes_use.size();
            bool used_cached = false;
            bool have_cached = false;
            Eigen::VectorXd cached_residual;
            const bool can_try_hybrid =
                options_.fe_recovery_method == FeRecoveryMethod::Hybrid &&
                have_group_ids &&
                absorption.fe_alpha_y.size() == fes_use.size() &&
                absorption.fe_alpha_X.size() == fes_use.size();
            if (profile_savefe && !can_try_hybrid) {
                if (options_.fe_recovery_method != FeRecoveryMethod::Hybrid) {
                    savefe_profile_log("event=hybrid_skip reason=method");
                }
                if (!have_group_ids) {
                    savefe_profile_log("event=hybrid_skip reason=missing_group_ids");
                }
                if (slope_cols > options_.savefe_fastpath_max_cols) {
                    std::ostringstream oss;
                    oss << "event=hybrid_skip reason=cols_gt_max cols=" << slope_cols
                        << " max=" << options_.savefe_fastpath_max_cols;
                    savefe_profile_log(oss.str());
                }
                if (absorption.fe_alpha_y.size() != fes_use.size() ||
                    absorption.fe_alpha_X.size() != fes_use.size()) {
                    savefe_profile_log("event=hybrid_skip reason=missing_alpha_buffers");
                }
            }
            if (can_try_hybrid) {
                const std::size_t dims = fes_use.size();
                const int nobs = static_cast<int>(partial.size());
                bool sizes_ok = true;
                for (std::size_t dim = 0; dim < dims; ++dim) {
                    if (static_cast<int>(absorption.fe_group_ids[dim].size()) != nobs) {
                        sizes_ok = false;
                        break;
                    }
                    const auto& ay = absorption.fe_alpha_y[dim];
                    const auto& aX = absorption.fe_alpha_X[dim];
                    if (aX.rows() != ay.size() || aX.cols() != slope_cols) {
                        sizes_ok = false;
                        break;
                    }
                }
                if (sizes_ok) {
                    {
                        SavefeTimer timer("savefe_hybrid_map");
                        results_.fe_effects.resize(dims);
                        const Eigen::VectorXd beta =
                            slope_cols > 0 ? results_.coefficients.head(slope_cols)
                                           : Eigen::VectorXd();
                        for (std::size_t dim = 0; dim < dims; ++dim) {
                            const auto& gid = absorption.fe_group_ids[dim];
                            const auto& ay = absorption.fe_alpha_y[dim];
                            const auto& aX = absorption.fe_alpha_X[dim];
                            Eigen::VectorXd alpha_d = ay;
                            if (slope_cols > 0) {
                                alpha_d.noalias() -= aX * beta;
                            }
                            Eigen::VectorXd fe_vec(nobs);
                            for (int i = 0; i < nobs; ++i) {
                                fe_vec[i] = alpha_d[gid[static_cast<std::size_t>(i)]];
                            }
                            results_.fe_effects[dim] = std::move(fe_vec);
                        }
                    }
                    const double fe_tol =
                        options_.fe_tolerance > 0.0 ? options_.fe_tolerance : options_.tol;
                    Eigen::VectorXd residual = partial;
                    for (const auto& fe_vec : results_.fe_effects) {
                        residual.noalias() -= fe_vec;
                    }
                    cached_residual = residual;
                    have_cached = true;
                    double max_delta = 0.0;
                    HdfeOptions fe_check_opts = tuned;
                    if (!absorption.sweep_order_used.empty()) {
                        fe_check_opts.sweep_order_override = absorption.sweep_order_used;
                    }
                    {
                        SavefeTimer timer("savefe_hybrid_check");
                        if (fe_tol > 0.0) {
                            bool checked = false;
                            if (absorption.gpu_used &&
                                absorption.fe_weight_sums.size() == dims &&
                                absorption.fe_group_ids.size() == dims) {
                                std::vector<detail::GpuFeInput> fe_inputs;
                                fe_inputs.reserve(dims);
                                bool inputs_ok = true;
                                for (std::size_t dim = 0; dim < dims; ++dim) {
                                    const auto& gid = absorption.fe_group_ids[dim];
                                    const auto& ws = absorption.fe_weight_sums[dim];
                                    if (static_cast<int>(gid.size()) != nobs || ws.size() == 0) {
                                        inputs_ok = false;
                                        break;
                                    }
                                    detail::GpuFeInput input;
                                    input.group_ids = gid.data();
                                    input.num_groups = static_cast<int>(ws.size());
                                    input.num_levels_present =
                                        dim < absorption.fe_levels.size()
                                            ? absorption.fe_levels[dim]
                                            : input.num_groups;
                                    input.weight_sums = ws.data();
                                    fe_inputs.push_back(input);
                                }
                                if (inputs_ok) {
                                    std::vector<std::size_t> check_order;
                                    check_order.reserve(fe_check_opts.sweep_order_override.size());
                                    for (const int dim : fe_check_opts.sweep_order_override) {
                                        if (dim < 0) {
                                            inputs_ok = false;
                                            break;
                                        }
                                        check_order.push_back(static_cast<std::size_t>(dim));
                                    }
                                    if (inputs_ok) {
                                        checked = detail::fe_recovery_max_delta_cuda_cached(
                                            residual, fe_inputs, w_ptr, check_order, max_delta);
                                        if (!checked) {
                                            checked = detail::fe_recovery_max_delta_cuda(
                                                residual, fe_inputs, w_ptr, check_order,
                                                max_delta);
                                        }
                                    }
                                }
                            }
                            if (!checked) {
                                Eigen::VectorXd residual_check = residual;
                                max_delta =
                                    detail::fe_recovery_max_delta(
                                        residual_check,
                                        absorption.fe_group_ids,
                                        absorption.fe_levels,
                                        w_ptr,
                                        fe_check_opts,
                                        absorption.fe_weight_sums.empty()
                                            ? nullptr
                                            : &absorption.fe_weight_sums);
                            }
                        }
                    }
                    if (fe_tol <= 0.0 || max_delta <= fe_tol) {
                        results_.fe_recovery_iterations = 0;
                        results_.fe_recovery_max_delta = max_delta;
                        results_.fe_recovery_converged = true;
                        fe_recovery_method = "hybrid_cached";
                        used_cached = true;
                        if (profile_savefe) {
                            std::ostringstream oss;
                            oss << "event=hybrid_cached max_delta=" << max_delta
                                << " fe_tol=" << fe_tol;
                            savefe_profile_log(oss.str());
                        }
                    } else if (profile_savefe) {
                        std::ostringstream oss;
                        oss << "event=hybrid_skip reason=max_delta max_delta=" << max_delta
                            << " fe_tol=" << fe_tol;
                        savefe_profile_log(oss.str());
                    }
                } else if (profile_savefe) {
                    savefe_profile_log("event=hybrid_skip reason=size_mismatch");
                }
            }
            if (!used_cached) {
                HdfeOptions fe_opts = tuned;
                if (!absorption.sweep_order_used.empty()) {
                    fe_opts.sweep_order_override = absorption.sweep_order_used;
                }
                const Eigen::VectorXd& fe_partial = have_cached ? cached_residual : partial;
                detail::FeRecoveryResult recovered;
                {
                    SavefeTimer timer("savefe_recover_map");
                    recovered = have_group_ids
                        ? detail::recover_fixed_effects_group_ids(
                              fe_partial, absorption.fe_group_ids, absorption.fe_levels, w_ptr,
                              fe_opts,
                              absorption.fe_weight_sums.empty()
                                  ? nullptr
                                  : &absorption.fe_weight_sums)
                        : detail::recover_fixed_effects(fe_partial, fes_use, w_ptr, fe_opts);
                }
                if (have_cached &&
                    recovered.contributions.size() == results_.fe_effects.size()) {
                    for (std::size_t dim = 0; dim < results_.fe_effects.size(); ++dim) {
                        results_.fe_effects[dim].noalias() += recovered.contributions[dim];
                    }
                } else {
                    results_.fe_effects = recovered.contributions;
                }
                results_.fe_recovery_iterations = recovered.iterations;
                results_.fe_recovery_max_delta = recovered.max_delta;
                results_.fe_recovery_converged = recovered.converged;
                fe_recovery_method = "map";
                if (profile_savefe) {
                    savefe_profile_log("event=fe_recovery_fallback method=map");
                }
            }
            }
        }

        if (!used_mixed_savefe && !results_.fe_effects.empty()) {
            const std::size_t dims = results_.fe_effects.size();
            const FeNormalizeStyle normalize_style =
                resolve_fe_normalize_style(options_.stats_style);
            if (dims == 1) {
                const double mean = weighted_mean(results_.fe_effects[0], w_ptr);
                results_.fe_effects[0].array() -= mean;
            } else if (dims >= 2) {
                if (normalize_style == FeNormalizeStyle::Reghdfe) {
                    for (std::size_t d = 0; d < dims; ++d) {
                        const double mean = weighted_mean(results_.fe_effects[d], w_ptr);
                        results_.fe_effects[d].array() -= mean;
                    }
                } else {
                    // Component normalization: center FE2 within each mobility component.
                    const Eigen::VectorXi& fe0 = fes_use[0];
                    const Eigen::VectorXi& fe1 = fes_use[1];
                    const int expected0 =
                        !absorption.fe_levels.empty() ? absorption.fe_levels[0] : -1;
                    const int expected1 =
                        absorption.fe_levels.size() > 1 ? absorption.fe_levels[1] : -1;
                    const FeLookup lookup0 = build_lookup_compact(fe0, expected0);
                    const FeLookup lookup1 = build_lookup_compact(fe1, expected1);
                    Eigen::VectorXi components;
                    const int num_components =
                        count_bipartite_components(fe0, fe1, lookup0, lookup1, &components);
                    if (num_components > 0) {
                        std::vector<double> comp_sum(static_cast<std::size_t>(num_components), 0.0);
                        std::vector<double> comp_w(static_cast<std::size_t>(num_components), 0.0);
                        const double* fe1_ptr = results_.fe_effects[1].data();
                        if (w_ptr) {
                            const double* w = w_ptr->data();
                            for (int i = 0; i < components.size(); ++i) {
                                const int c = components(i) - 1;
                                comp_sum[static_cast<std::size_t>(c)] += fe1_ptr[i] * w[i];
                                comp_w[static_cast<std::size_t>(c)] += w[i];
                            }
                        } else {
                            for (int i = 0; i < components.size(); ++i) {
                                const int c = components(i) - 1;
                                comp_sum[static_cast<std::size_t>(c)] += fe1_ptr[i];
                                comp_w[static_cast<std::size_t>(c)] += 1.0;
                            }
                        }
                        for (int i = 0; i < components.size(); ++i) {
                            const int c = components(i) - 1;
                            const double denom = comp_w[static_cast<std::size_t>(c)];
                            const double shift =
                                denom > 0.0 ? comp_sum[static_cast<std::size_t>(c)] / denom : 0.0;
                            results_.fe_effects[1](i) -= shift;
                            results_.fe_effects[0](i) += shift;
                        }
                    }
                    const double mean0 = weighted_mean(results_.fe_effects[0], w_ptr);
                    results_.fe_effects[0].array() -= mean0;
                    for (std::size_t d = 2; d < dims; ++d) {
                        const double mean = weighted_mean(results_.fe_effects[d], w_ptr);
                        results_.fe_effects[d].array() -= mean;
                    }
                }
            }
        }

        if (!results_.fe_effects.empty()) {
            Eigen::VectorXd fe_total = Eigen::VectorXd::Zero(static_cast<int>(y_use.size()));
            for (const auto& fe_vec : results_.fe_effects) {
                fe_total.noalias() += fe_vec;
            }
            if (options_.fit_intercept) {
                const int intercept_idx =
                    static_cast<int>(results_.coefficients.size()) - 1;
                const int slope_cols = intercept_idx;
                Eigen::VectorXd linear_predictor =
                    Eigen::VectorXd::Zero(static_cast<int>(y_use.size()));
                if (slope_cols > 0) {
                    linear_predictor.noalias() = X_use * results_.coefficients.head(slope_cols);
                }
                linear_predictor.noalias() += fe_total;
                const double intercept = weighted_mean(y_use - linear_predictor, w_ptr);
                results_.coefficients(intercept_idx) = intercept;
                results_.residuals = y_use - (linear_predictor.array() + intercept).matrix();
            } else {
                Eigen::VectorXd fitted_full = X_use * results_.coefficients;
                fitted_full.noalias() += fe_total;
                results_.residuals = y_use - fitted_full;
            }
        }

        if (options_.retain_fixed_effects && !results_.fe_effects.empty()) {
            maybe_write_fe_diagnostics(y_use, X_use, results_.coefficients, results_.fe_effects,
                                       options_.fit_intercept, w_ptr, results_.residuals,
                                       absorption.iterations, results_.fe_recovery_iterations,
                                       results_.fe_recovery_converged,
                                       results_.fe_recovery_max_delta, fe_recovery_method);
        }

        const bool needs_refined_residuals =
            options_.refine_stored_residuals && has_fes && !options_.retain_fixed_effects &&
            results_.residuals.size() == y_use.size();
        if (needs_refined_residuals) {
            const int coef_size = static_cast<int>(results_.coefficients.size());
            const int expected_slope_cols =
                options_.fit_intercept ? std::max(0, coef_size - 1) : coef_size;
            if (expected_slope_cols == static_cast<int>(X_use.cols())) {
                Eigen::VectorXd partial = y_use;
                if (expected_slope_cols > 0) {
                    partial.noalias() -= X_use * results_.coefficients.head(expected_slope_cols);
                }
                if (options_.fit_intercept && coef_size > expected_slope_cols) {
                    partial.array() -= results_.coefficients(coef_size - 1);
                }

                HdfeOptions resid_opts = tuned;
                resid_opts.retain_fixed_effects = false;
                if (!absorption.sweep_order_used.empty()) {
                    resid_opts.sweep_order_override = absorption.sweep_order_used;
                }

                Eigen::MatrixXd empty_design(static_cast<int>(y_use.size()), 0);
                detail::AbsorptionResult resid_absorption =
                    detail::absorb_fixed_effects_v6(partial,
                                                    empty_design,
                                                    fes_use,
                                                    w_ptr,
                                                    resid_opts,
                                                    method_used_,
                                                    slope_terms);
                if (resid_absorption.converged &&
                    resid_absorption.y_tilde.size() == partial.size()) {
                    results_.residuals = std::move(resid_absorption.y_tilde);
                }
            }
        }

        const int omit_cols = static_cast<int>(results_.coefficients.size());
        results_.omitted_reason.assign(static_cast<std::size_t>(omit_cols), 0);
        for (const int idx : omitted_fe_slope_cols) {
            if (idx >= 0 && idx < omit_cols) {
                results_.omitted_reason[static_cast<std::size_t>(idx)] = 1;
            }
        }
        for (const int idx : omitted_rank_slope_cols) {
            if (idx >= 0 && idx < omit_cols &&
                results_.omitted_reason[static_cast<std::size_t>(idx)] == 0) {
                results_.omitted_reason[static_cast<std::size_t>(idx)] = 2;
            }
        }

        // Degrees-of-freedom adjustments to match reghdfe / fixest SSC.
        const double nobs = results_.nobs_effective;
        const int df_m = df_m_effective;  // excludes constant
        results_.df_m = static_cast<double>(df_m);
        const int df_intercept = (options_.fit_intercept && !drop_intercept) ? 1 : 0;
        const bool use_reghdfe_stats = (options_.stats_style == StatsStyle::Reghdfe);
        const int df_intercept_dof =
            use_reghdfe_stats ? (has_fes ? 0 : df_intercept) : df_intercept;

        int df_a_levels = 0;
        int df_a_exact = 0;
        int nested_dof_levels = 0;
        int nested_dof_coefs = 0;
        if (has_fes) {
            post_phase_t0 = std::chrono::steady_clock::now();
            Eigen::VectorXi groupvar;
            Eigen::VectorXi* groupvar_ptr = options_.save_groupvar ? &groupvar : nullptr;
            const std::size_t dof_dims = fes_use.size();
            std::vector<uint8_t> nested_flags(dof_dims, 0);
            std::vector<uint8_t> slope_only_flags(dof_dims, 0);
            for (const auto& slope : slope_terms) {
                if (!slope.include_intercept &&
                    slope.fe_index >= 0 &&
                    slope.fe_index < static_cast<int>(dof_dims)) {
                    slope_only_flags[static_cast<std::size_t>(slope.fe_index)] = 1;
                }
            }
            if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
                !c_ptr->empty()) {
                for (const auto& cl : *c_ptr) {
                    for (std::size_t d = 0; d < dof_dims; ++d) {
                        if (nested_flags[d]) {
                            continue;
                        }
                        if (fe_nested_within_cluster(fes_use[d], cl)) {
                            nested_flags[d] = 1;
                        }
                    }
                }
            }
            bool any_nested = false;
            if (options_.dof_method != DofAdjustmentMethod::None) {
                for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                    if (nested_flags[d]) {
                        any_nested = true;
                        break;
                    }
                }
            }

            FeDofInfo dof;
            const bool can_skip_full_dof =
                any_nested && groupvar_ptr == nullptr &&
                absorption.fe_levels.size() == dof_dims &&
                options_.dof_method != DofAdjustmentMethod::None;
            if (can_skip_full_dof) {
                dof.levels = absorption.fe_levels;
                dof.redundant.assign(dof_dims, 0);
                dof.num_coefs.assign(dof_dims, 0);
                dof.inexact.assign(dof_dims, 0);

                std::vector<Eigen::VectorXi> fes_nonnested;
                std::vector<int> levels_nonnested;
                std::vector<int> map_nonnested;
                fes_nonnested.reserve(dof_dims);
                levels_nonnested.reserve(dof_dims);
                map_nonnested.reserve(dof_dims);
                for (std::size_t d = 0; d < dof_dims; ++d) {
                    if (nested_flags[d]) {
                        continue;
                    }
                    fes_nonnested.push_back(fes_use[d]);
                    levels_nonnested.push_back(dof.levels[d]);
                    map_nonnested.push_back(static_cast<int>(d));
                }
                if (!fes_nonnested.empty()) {
                    const FeDofInfo dof_nonnested =
                        compute_fe_dof_reghdfe(fes_nonnested, levels_nonnested,
                                               options_.dof_method, nullptr, tuned.num_threads);
                    for (std::size_t pos = 0; pos < map_nonnested.size(); ++pos) {
                        const int orig = map_nonnested[pos];
                        dof.redundant[static_cast<std::size_t>(orig)] =
                            dof_nonnested.redundant[pos];
                    }
                }
            } else {
                dof = compute_fe_dof_reghdfe(fes_use, absorption.fe_levels,
                                             options_.dof_method, groupvar_ptr,
                                             tuned.num_threads);

                // If some fixed effects are nested within the clustering variable, reghdfe does not
                // use them to compute mobility-group redundancies for the remaining dimensions.
                if (any_nested && options_.dof_method != DofAdjustmentMethod::None) {
                    std::vector<Eigen::VectorXi> fes_nonnested;
                    std::vector<int> levels_nonnested;
                    std::vector<int> map_nonnested;
                    fes_nonnested.reserve(dof_dims);
                    levels_nonnested.reserve(dof_dims);
                    map_nonnested.reserve(dof_dims);
                    for (std::size_t d = 0; d < dof_dims; ++d) {
                        if (nested_flags[d]) {
                            continue;
                        }
                        fes_nonnested.push_back(fes_use[d]);
                        levels_nonnested.push_back(dof.levels[d]);
                        map_nonnested.push_back(static_cast<int>(d));
                    }
                    if (!fes_nonnested.empty()) {
                        const FeDofInfo dof_nonnested =
                            compute_fe_dof_reghdfe(fes_nonnested, levels_nonnested,
                                                   options_.dof_method, nullptr, tuned.num_threads);
                        for (std::size_t pos = 0; pos < map_nonnested.size(); ++pos) {
                            const int orig = map_nonnested[pos];
                            dof.redundant[static_cast<std::size_t>(orig)] =
                                dof_nonnested.redundant[pos];
                        }
                    }
                }
            }
            int nested_total = 0;
            if (options_.dof_method != DofAdjustmentMethod::None) {
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    if (nested_flags[d]) {
                        nested_total += dof.levels[d];
                        dof.redundant[d] = dof.levels[d];
                        dof.num_coefs[d] = 0;
                    }
                }
                std::vector<int> intercept_index;
                intercept_index.reserve(dof.levels.size());
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    if (!nested_flags[d] && !slope_only_flags[d]) {
                        intercept_index.push_back(static_cast<int>(d));
                    }
                }
                if (!intercept_index.empty()) {
                    bool skip_next = (nested_total == 0);
                    for (const int idx : intercept_index) {
                        if (dof.redundant[static_cast<std::size_t>(idx)] >=
                            dof.levels[static_cast<std::size_t>(idx)]) {
                            continue;
                        }
                        if (skip_next) {
                            skip_next = false;
                            continue;
                        }
                        if (dof.redundant[static_cast<std::size_t>(idx)] < 1) {
                            dof.redundant[static_cast<std::size_t>(idx)] = 1;
                        }
                    }
                }
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    dof.num_coefs[d] =
                        std::max(0, dof.levels[d] - dof.redundant[d]);
                }
                dof.inexact.assign(dof.levels.size(), 0);
                if (options_.dof_method == DofAdjustmentMethod::Pairwise ||
                    options_.dof_method == DofAdjustmentMethod::All ||
                    options_.dof_method == DofAdjustmentMethod::FirstPair) {
                    int intercept_count = 0;
                    for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                        if (nested_flags[d] || slope_only_flags[d]) {
                            continue;
                        }
                        ++intercept_count;
                        if (intercept_count > 2) {
                            dof.inexact[d] = 1;
                        }
                    }
                }
            } else {
                dof.inexact.assign(dof.levels.size(), 0);
            }
            apply_heterogeneous_slope_dof(dof, fes_use, slope_terms, w_ptr, nested_flags);
            df_a_levels = 0;
            df_a_exact = 0;
            for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                df_a_levels += dof.levels[d];
                df_a_exact += dof.num_coefs[d];
            }
            dof.df_a_levels = df_a_levels;
            dof.df_a_exact = df_a_exact;
            dof.df_a = df_a_exact;
            results_.df_a_levels = static_cast<double>(df_a_levels);
            results_.df_a_exact = static_cast<double>(df_a_exact);
            results_.fe_num_levels = dof.levels;
            results_.fe_base_levels = absorption.fe_levels;
            results_.fe_redundant = dof.redundant;
            results_.fe_num_coefs = dof.num_coefs;
            results_.fe_inexact = dof.inexact;
            results_.fe_nested.assign(dof.levels.size(), 0);
            if (groupvar_ptr) {
                results_.groupvar = std::move(groupvar);
            }
            if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
                !c_ptr->empty()) {
                for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                    if (nested_flags[d]) {
                        nested_dof_coefs += nested_coefs_for_heterogeneous_dim(
                            d, slope_terms, dof.num_coefs);
                        nested_dof_levels += nested_levels_for_heterogeneous_dim(
                            d, slope_terms, absorption.fe_levels, dof.levels);
                        results_.fe_nested[d] = 1;
                    }
                }
            }
            cpu_profile_log_elapsed("post_dof", post_phase_t0);
        }

        const int df_a_report = df_a_exact;
        const int nested_report = nested_dof_levels;
        results_.df_a = static_cast<double>(df_a_report);
        results_.df_a_nested = static_cast<double>(nested_report);

        int df_a_used = df_a_report;
        int nested_dof_used = nested_dof_levels;
        int nested_adj = (nested_dof_levels > 0) ? 1 : 0;
        if (!use_reghdfe_stats) {
            df_a_used = 0;
            nested_dof_used = 0;
            const int df_a_base = tuned.ssc_k_exact ? df_a_exact : df_a_levels;
            const int nested_base = tuned.ssc_k_exact ? nested_dof_coefs : nested_dof_levels;
            nested_adj = (nested_base > 0) ? 1 : 0;
            if (tuned.ssc_k_fixef == FixefDofMethod::None) {
                df_a_used = 0;
                nested_dof_used = 0;
            } else {
                if (tuned.ssc_k_fixef == FixefDofMethod::Nonnested &&
                    tuned.se_type == StandardErrorType::Cluster) {
                    df_a_used = std::max(0, df_a_base - nested_base);
                    nested_dof_used = 0;
                } else {
                    df_a_used = df_a_base;
                    nested_dof_used = nested_base;
                }
            }
        }

        // Use absorbed DoF (after redundancy) for RMSE/Adj R2 residual df.
        // Keep the raw (possibly non-positive) df_r so the saturated case can
        // be handled reghdfe-style (negative e(df_r), missing inference).
        const double df_r_raw =
            nobs - static_cast<double>(df_m) - static_cast<double>(df_intercept_dof) -
            static_cast<double>(df_a_report);
        const double df_r_unadj = std::max(0.0, df_r_raw);
        results_.df_resid_unadj = df_r_raw;
        if (df_r_unadj > 0.0) {
            const double sigma2_old = results_.sigma2;
            const double sigma2_new = results_.rss / df_r_unadj;
            results_.sigma2 = sigma2_new;
            if (tuned.se_type == StandardErrorType::Homoskedastic && sigma2_old > 0.0) {
                const double ratio = sigma2_new / sigma2_old;
                results_.covariance *= ratio;
                results_.std_errors *= std::sqrt(ratio);
            }
        } else {
            results_.sigma2 = 0.0;
        }
        const bool saturated_model = !(df_r_raw > 0.0);

        double df_r_model =
            nobs - static_cast<double>(df_m) - static_cast<double>(df_intercept_dof) -
            static_cast<double>(df_a_used);
        df_r_model = std::max(0.0, df_r_model);

        // Cluster diagnostics (counts).
        post_phase_t0 = std::chrono::steady_clock::now();
        if (c_ptr && !c_ptr->empty()) {
            results_.num_clusters = ols_result.num_clusters;
            if (c_ptr->size() == 1) {
                results_.cluster_counts.assign(1, results_.num_clusters);
                results_.cluster_combo_counts.assign(1, results_.num_clusters);
            } else {
                results_.cluster_counts = compute_cluster_counts(*c_ptr);
                results_.cluster_combo_counts = compute_cluster_combo_counts(*c_ptr);
            }
        } else {
            results_.cluster_counts.clear();
            results_.cluster_combo_counts.clear();
            results_.num_clusters = 0;
        }
        cpu_profile_log_elapsed("post_cluster_counts", post_phase_t0);

        double robust_scale = 1.0;
        double cluster_ratio = 1.0;
        if (tuned.se_type == StandardErrorType::Robust && df_r_model > 0.0) {
            if (tuned.ssc_k_adj) {
                robust_scale = nobs / df_r_model;
                const double scale = std::sqrt(robust_scale);
                results_.std_errors *= scale;
                results_.covariance *= robust_scale;
            }
            results_.df_resid = tuned.ssc_t_df > 0.0 ? tuned.ssc_t_df : df_r_model;
        } else if (tuned.se_type == StandardErrorType::Cluster && df_m > 0 && c_ptr && !c_ptr->empty()) {
            const int G = ols_result.num_clusters;
            if (use_reghdfe_stats) {
                const double df_r_unclust = df_r_model;
                const double df_r_cluster = static_cast<double>(std::max(0, G - 1));
                results_.df_resid = std::min(df_r_unclust, df_r_cluster);
                double ratio = 1.0;
                if (tuned.ssc_k_adj) {
                    const double denom = std::max(
                        0.0, nobs - static_cast<double>(df_m) - static_cast<double>(df_a_report) -
                                 static_cast<double>(nested_adj));
                    if (denom > 0.0) {
                        ratio *= ols_result.df_resid / denom;
                    }
                } else {
                    const double denom = std::max(1.0, nobs - 1.0);
                    ratio *= ols_result.df_resid / denom;
                }
                if (c_ptr->size() == 1 && !tuned.ssc_g_adj && G > 1) {
                    ratio *= static_cast<double>(G - 1) / static_cast<double>(G);
                }
                if (ratio > 0.0 && std::isfinite(ratio)) {
                    const double scale = std::sqrt(ratio);
                    results_.std_errors *= scale;
                    results_.covariance *= ratio;
                    cluster_ratio = ratio;
                }
                results_.cluster_scale = ratio;
                if (tuned.ssc_t_df > 0.0) {
                    results_.df_resid = tuned.ssc_t_df;
                }
            } else {
                results_.df_resid = static_cast<double>(std::max(0, G - 1));
                double ratio = 1.0;
                if (tuned.ssc_k_adj) {
                    const double df_denom =
                        std::max(0.0, df_r_model + static_cast<double>(nested_dof_used) -
                                          static_cast<double>(nested_adj));
                    if (df_denom > 0.0) {
                        ratio *= ols_result.df_resid / df_denom;
                    }
                } else {
                    const double denom = std::max(1.0, nobs - 1.0);
                    ratio *= ols_result.df_resid / denom;
                }
                if (c_ptr->size() == 1 && !tuned.ssc_g_adj && G > 1) {
                    ratio *= static_cast<double>(G - 1) / static_cast<double>(G);
                }
                if (ratio > 0.0 && std::isfinite(ratio)) {
                    const double scale = std::sqrt(ratio);
                    results_.std_errors *= scale;
                    results_.covariance *= ratio;
                    cluster_ratio = ratio;
                }
                results_.cluster_scale = ratio;
                if (tuned.ssc_t_df > 0.0) {
                    results_.df_resid = tuned.ssc_t_df;
                }
            }
        } else if (df_r_model > 0.0) {
            results_.df_resid = tuned.ssc_t_df > 0.0 ? tuned.ssc_t_df : df_r_model;
        }
        if (saturated_model) {
            results_.df_resid = df_r_raw;
        }

        bool intercept_cov_updated = false;
        if (drop_intercept && options_.fit_intercept && !saturated_model) {
            post_phase_t0 = std::chrono::steady_clock::now();
            intercept_cov_updated = update_intercept_covariance(
                results_, X_use, transformed_X_used, ols_result.residuals, w_ptr,
                kept_slope_cols, ols_result, tuned.se_type, c_ptr, tuned.ssc_g_df,
                tuned.ssc_g_adj, results_.sigma2, robust_scale, cluster_ratio,
                tuned.weights_are_frequencies);
            cpu_profile_log_elapsed("post_intercept_cov", post_phase_t0);

            // Multiway clustering: rebuild the whole covariance the way
            // reghdfe does for report_constant — extended partitioned-inverse
            // bread over [X_tilde, 1] and cluster scores on the means-restored
            // regressors. The recovered-intercept formula above matches one-
            // way clustering but not the inclusion-exclusion CGM sum, and the
            // PSD fix below is extremely sensitive to the _cons row/col input.
            if (tuned.se_type == StandardErrorType::Cluster && c_ptr &&
                c_ptr->size() > 1) {
                const int cov_cols = static_cast<int>(results_.covariance.cols());
                const int slope_cols_aug = cov_cols - 1;
                if (slope_cols_aug > 0 &&
                    slope_cols_aug == static_cast<int>(transformed_X_used.cols()) &&
                    slope_cols_aug == static_cast<int>(ols_result.xtx_inv.cols()) &&
                    slope_cols_aug == static_cast<int>(kept_slope_cols.size())) {
                    const int n_rows = static_cast<int>(transformed_X_used.rows());
                    Eigen::RowVectorXd means_x(slope_cols_aug);
                    for (int j = 0; j < slope_cols_aug; ++j) {
                        means_x(j) = weighted_mean(
                            X_use.col(kept_slope_cols[static_cast<std::size_t>(j)]),
                            w_ptr);
                    }
                    // reghdfe scores: demeaned X with the original means added
                    // back, plus the constant column.
                    Eigen::MatrixXd scores(n_rows, cov_cols);
                    scores.leftCols(slope_cols_aug) =
                        transformed_X_used.rowwise() + means_x;
                    scores.col(slope_cols_aug).setOnes();
                    // Extended bread (partitioned inverse; reghdfe_extend_b_and_xx).
                    const double n_bread =
                        (tuned.weights_are_frequencies && w_ptr)
                            ? w_ptr->sum()
                            : static_cast<double>(n_rows);
                    Eigen::MatrixXd bread(cov_cols, cov_cols);
                    bread.topLeftCorner(slope_cols_aug, slope_cols_aug) =
                        ols_result.xtx_inv;
                    Eigen::RowVectorXd side = -means_x * ols_result.xtx_inv;
                    bread.block(slope_cols_aug, 0, 1, slope_cols_aug) = side;
                    bread.block(0, slope_cols_aug, slope_cols_aug, 1) =
                        side.transpose();
                    bread(slope_cols_aug, slope_cols_aug) =
                        1.0 / n_bread +
                        (means_x * ols_result.xtx_inv * means_x.transpose())(0, 0);
                    Eigen::MatrixXd cov_aug = detail::multiway_cluster_sandwich(
                        bread, scores, ols_result.residuals, w_ptr, *c_ptr);
                    double num = 0.0;
                    double den = 0.0;
                    for (int j = 0; j < slope_cols_aug; ++j) {
                        num += results_.covariance(j, j);
                        den += cov_aug(j, j);
                    }
                    const double rescale = (den > 0.0) ? (num / den) : 0.0;
                    if (rescale > 0.0 && std::isfinite(rescale) && cov_aug.allFinite()) {
                        results_.covariance = cov_aug * rescale;
                        results_.std_errors =
                            results_.covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
                        intercept_cov_updated = true;
                    }
                }
            }
        }

        if (tuned.se_type == StandardErrorType::Cluster && c_ptr && c_ptr->size() > 1) {
            const int cov_cols = static_cast<int>(results_.covariance.cols());
            const int slope_cols =
                (options_.fit_intercept && cov_cols > 0) ? (cov_cols - 1) : cov_cols;

            // reghdfe (reghdfe_fix_psd with report_constant): clamp negative
            // eigenvalues on the FULL matrix (including the recovered _cons
            // row/col), then re-fix the slope block separately so slope
            // inference matches the constant-free correction. reghdfe runs the
            // clamp on STANDARDIZED data (each regressor divided by its raw
            // sample stdev), and the clamp does not commute with per-column
            // rescaling, so the same metric is used here (the congruence
            // preserves the eigenvalue signs, hence whether a fix happens).
            Eigen::VectorXd psd_scale = Eigen::VectorXd::Ones(std::max(cov_cols, 0));
            if (cov_cols > 0 &&
                slope_cols <= static_cast<int>(kept_slope_cols.size())) {
                const double n_sd = static_cast<double>(X_use.rows());
                for (int j = 0; j < slope_cols; ++j) {
                    const auto col = X_use.col(kept_slope_cols[static_cast<std::size_t>(j)]);
                    const double s = col.sum();
                    const double ss = col.squaredNorm();
                    double sd = 0.0;
                    if (n_sd > 1.0) {
                        sd = std::sqrt(std::max(0.0, (ss - s * s / n_sd) / (n_sd - 1.0)));
                    }
                    psd_scale(j) = std::max(sd, 1.0e-3);
                }
            }
            auto fix_psd_scaled = [&](Eigen::MatrixXd& V,
                                      const Eigen::Ref<const Eigen::VectorXd>& d) -> bool {
                Eigen::MatrixXd Vs = d.asDiagonal() * V * d.asDiagonal();
                if (!fix_psd(Vs)) {
                    return false;
                }
                V = d.cwiseInverse().asDiagonal() * Vs * d.cwiseInverse().asDiagonal();
                return true;
            };

            const bool full_psd_ready = options_.fit_intercept && cov_cols > 0 &&
                                        intercept_cov_updated &&
                                        results_.covariance.allFinite();
            if (full_psd_ready) {
                Eigen::MatrixXd cov_full = results_.covariance;
                Eigen::MatrixXd cov_block =
                    cov_full.topLeftCorner(slope_cols, slope_cols);
                if (fix_psd_scaled(cov_full, psd_scale)) {
                    if (slope_cols > 0) {
                        (void)fix_psd_scaled(cov_block, psd_scale.head(slope_cols));
                        cov_full.topLeftCorner(slope_cols, slope_cols) = cov_block;
                    }
                    results_.covariance = cov_full;
                    results_.std_errors = cov_full.diagonal().cwiseMax(0.0).cwiseSqrt();
                    results_.vcv_psd_fixed = true;
                }
            } else if (slope_cols > 0) {
                Eigen::MatrixXd cov_block =
                    results_.covariance.topLeftCorner(slope_cols, slope_cols);
                if (fix_psd_scaled(cov_block, psd_scale.head(slope_cols))) {
                    results_.covariance.topLeftCorner(slope_cols, slope_cols) = cov_block;
                    results_.std_errors.head(slope_cols) =
                        cov_block.diagonal().cwiseMax(0.0).cwiseSqrt();
                    results_.vcv_psd_fixed = true;
                }
            }
        }

        post_phase_t0 = std::chrono::steady_clock::now();
        recompute_inference(results_.coefficients, results_.std_errors, results_.tvalues,
                            results_.pvalues, results_.conf_int, tuned.level, results_.df_resid);
        normalize_nonfrequency_weighted_fit_stats(
            results_, tuned.weights_are_frequencies, w_ptr, sum_weights_for_stats);
        if (saturated_model) {
            mark_invalid_inference_for_saturated(results_);
        }
        cpu_profile_log_elapsed("post_inference", post_phase_t0);
        cpu_profile_log_elapsed("postprocess", post_t0);
        cpu_profile_log_elapsed("fit_inner_total", fit_t0);

#ifdef HDFE_USE_OPENMP
        Eigen::setNbThreads(1);
#endif
    };

    if (!has_slopes && fe_cache_cfg.mode != "off" && has_fes) {
        const auto fe_cache_read_t0 = std::chrono::steady_clock::now();
        const bool allow_read =
            (fe_cache_cfg.mode == "read" || fe_cache_cfg.mode == "auto");
        if (allow_read) {
            ensure_fe_cache_signature();
            fe_cache_hit =
                read_fe_structure_cache(fe_cache_cfg.path, fe_cache_signature, nobs_full,
                                        options_.drop_singletons,
                                        static_cast<int>(fes.size()), fe_cache);
        }
        cpu_profile_log_elapsed("fe_cache_read", fe_cache_read_t0);
    }

    if (fe_cache_hit) {
        if (options_.drop_singletons && fe_cache.nobs < nobs_full) {
            const auto filter_t0 = std::chrono::steady_clock::now();
            kept_idx = std::move(fe_cache.kept_idx);
            singletons_dropped_rows = nobs_full - fe_cache.nobs;
            y_work = filter_vector(y, kept_idx);
            X_work = filter_rows(X, kept_idx);
            fes_work = std::move(fe_cache.fe_group_ids);

            const Eigen::VectorXd* w_ptr = nullptr;
            if (weights) {
                weights_work = filter_vector(*weights, kept_idx);
                w_ptr = &(*weights_work);
            }

            const std::vector<Eigen::VectorXi>* c_ptr = clusters;
            if (clusters && !clusters->empty()) {
                std::vector<Eigen::VectorXi> clusters_tmp;
                clusters_tmp.reserve(clusters->size());
                for (const auto& c : *clusters) {
                    clusters_tmp.push_back(filter_vector(c, kept_idx));
                }
                clusters_work = std::move(clusters_tmp);
                c_ptr = &(*clusters_work);
            }

            const Eigen::MatrixXd* inst_use = inst_ptr;
            if (inst_ptr) {
                instruments_work = filter_rows(*inst_ptr, kept_idx);
                inst_use = &(*instruments_work);
            }
            const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use = slopes_in;
            if (has_slopes) {
                slopes_work = filter_slope_terms(*slopes_in, kept_idx);
                slopes_use = &(*slopes_work);
            }
            cpu_profile_log_elapsed("singleton_filter_cache", filter_t0);

            run_fit(*y_work, *X_work, *fes_work, w_ptr, c_ptr, inst_use, slopes_use);

            if (options_.weights_are_frequencies && weights_work) {
                singletons_dropped_effective =
                    std::max(0.0, nobs_full_effective - weights_work->sum());
            } else {
                singletons_dropped_effective = static_cast<double>(singletons_dropped_rows);
            }
        } else {
            fes_work = std::move(fe_cache.fe_group_ids);
            run_fit(y, X, *fes_work, weights, clusters, inst_ptr, slopes_in);
        }
    } else if (options_.drop_singletons && has_fes) {
        const auto singleton_t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> keep;
        if (options_.weights_are_frequencies && weights) {
            keep = compute_keep_mask_drop_singletons_fweights(
                fes, *weights, kMaxSingletonIterations, &singletons_dropped_rows,
                &singletons_dropped_effective);
        } else {
            keep = compute_keep_mask_drop_singletons(fes, kMaxSingletonIterations,
                                                     &singletons_dropped_rows,
                                                     options_.num_threads);
            singletons_dropped_effective = static_cast<double>(singletons_dropped_rows);
        }
        cpu_profile_log_elapsed("singleton_scan", singleton_t0);

        if (singletons_dropped_rows > 0) {
            const auto filter_t0 = std::chrono::steady_clock::now();
            kept_idx = build_keep_indices(keep);
            y_work = filter_vector(y, kept_idx);
            X_work = filter_rows(X, kept_idx);

            std::vector<Eigen::VectorXi> fes_tmp;
            fes_tmp.reserve(fes.size());
            for (const auto& fe : fes) {
                fes_tmp.push_back(filter_vector(fe, kept_idx));
            }
            fes_work = std::move(fes_tmp);

            const Eigen::VectorXd* w_ptr = nullptr;
            if (weights) {
                weights_work = filter_vector(*weights, kept_idx);
                w_ptr = &(*weights_work);
            }

            const std::vector<Eigen::VectorXi>* c_ptr = clusters;
            if (clusters && !clusters->empty()) {
                std::vector<Eigen::VectorXi> clusters_tmp;
                clusters_tmp.reserve(clusters->size());
                for (const auto& c : *clusters) {
                    clusters_tmp.push_back(filter_vector(c, kept_idx));
                }
                clusters_work = std::move(clusters_tmp);
                c_ptr = &(*clusters_work);
            }

            const Eigen::MatrixXd* inst_use = inst_ptr;
            if (inst_ptr) {
                instruments_work = filter_rows(*inst_ptr, kept_idx);
                inst_use = &(*instruments_work);
            }
            const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use = slopes_in;
            if (has_slopes) {
                slopes_work = filter_slope_terms(*slopes_in, kept_idx);
                slopes_use = &(*slopes_work);
            }
            cpu_profile_log_elapsed("singleton_filter", filter_t0);

            run_fit(*y_work, *X_work, *fes_work, w_ptr, c_ptr, inst_use, slopes_use);
        } else {
            run_fit(y, X, fes, weights, clusters, inst_ptr, slopes_in);
        }
    } else {
        run_fit(y, X, fes, weights, clusters, inst_ptr, slopes_in);
    }
    const bool allow_fe_cache_write =
        fe_cache_cfg.mode == "write" ||
        (fe_cache_cfg.mode == "auto" && fe_cache_cfg.allow_auto_write);
    if (!has_slopes && !fe_cache_hit && allow_fe_cache_write && has_fes && !fe_cache_cfg.path.empty()) {
        ensure_fe_cache_signature();
        const std::vector<Eigen::VectorXi>* fes_used =
            fes_work ? &(*fes_work) : &fes;
        if (!fes_used->empty()) {
            std::vector<Eigen::VectorXi> fe_group_ids;
            std::vector<int> num_groups;
            fe_group_ids.reserve(fes_used->size());
            num_groups.reserve(fes_used->size());
            for (const auto& fe : *fes_used) {
                FeIndexerLite idx = build_indexer_lite(fe);
                Eigen::VectorXi ids(static_cast<int>(idx.group_ids.size()));
                for (int i = 0; i < ids.size(); ++i) {
                    ids(i) = idx.group_ids[static_cast<std::size_t>(i)];
                }
                fe_group_ids.push_back(std::move(ids));
                num_groups.push_back(idx.num_groups);
            }
            write_fe_structure_cache(fe_cache_cfg.path, fe_cache_signature, nobs_full,
                                     options_.drop_singletons, kept_idx, num_groups,
                                     fe_group_ids);
        }
    }

    results_.nobs_full = static_cast<int>(y.size());
    results_.num_singletons = singletons_dropped_rows;
    const auto sample_index_t0 = std::chrono::steady_clock::now();
    results_.sample_index.resize(results_.nobs);
    if (!kept_idx.empty()) {
        for (int i = 0; i < results_.nobs; ++i) {
            results_.sample_index(i) = kept_idx[static_cast<std::size_t>(i)];
        }
    } else {
        for (int i = 0; i < results_.nobs; ++i) {
            results_.sample_index(i) = i;
        }
    }
    cpu_profile_log_elapsed("sample_index", sample_index_t0);
    cpu_profile_log_elapsed("fit_total", fit_outer_t0);

}

detail::AbsorptionResult HdfeRegressorV11::partial_out(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>* clusters,
    const std::vector<detail::HeterogeneousSlopeTerm>* slopes) {
    if (y.size() == 0) {
        throw std::runtime_error("Outcome vector must be non-empty");
    }
    if (X.rows() != y.size()) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (weights && weights->size() != y.size()) {
        throw std::runtime_error("Weights must have the same length as y");
    }
    gpu_used_ = false;
    gpu_status_code_ = 0;
    gpu_attempted_ = false;
    gpu_absorption_converged_ = false;
    gpu_absorption_iterations_ = 0;
    if (clusters) {
        for (const auto& c : *clusters) {
            if (c.size() != y.size()) {
                throw std::runtime_error("Clusters must have the same length as y");
            }
        }
    }
    const std::vector<detail::HeterogeneousSlopeTerm> empty_slopes;
    const std::vector<detail::HeterogeneousSlopeTerm>* slopes_in =
        slopes ? slopes : &empty_slopes;
    const bool has_slopes = slopes_in && !slopes_in->empty();
    if (has_slopes && fes.empty()) {
        throw std::runtime_error("Heterogeneous slopes require at least one absorbed FE");
    }
    if (has_slopes) {
        for (const auto& slope : *slopes_in) {
            if (slope.fe_index < 0 || slope.fe_index >= static_cast<int>(fes.size())) {
                throw std::runtime_error("Heterogeneous slope FE index out of range");
            }
            if (slope.values.size() != y.size()) {
                throw std::runtime_error("Heterogeneous slope variable must have the same length as y");
            }
        }
    }

    const bool has_fes = !fes.empty();
    const FeStructureCacheConfig fe_cache_cfg = load_fe_structure_cache_config();
    const int nobs_full = static_cast<int>(y.size());
    const double nobs_full_effective =
        (options_.weights_are_frequencies && weights) ? weights->sum()
                                                      : static_cast<double>(nobs_full);

    int singletons_dropped_rows = 0;
    double singletons_dropped_effective = 0.0;
    std::vector<int> kept_idx;
    std::optional<Eigen::VectorXd> y_work;
    std::optional<Eigen::MatrixXd> X_work;
    std::optional<std::vector<Eigen::VectorXi>> fes_work;
    std::optional<Eigen::VectorXd> weights_work;
    std::optional<std::vector<Eigen::VectorXi>> clusters_work;
    std::optional<std::vector<detail::HeterogeneousSlopeTerm>> slopes_work;

    FeStructureCache fe_cache;
    bool fe_cache_hit = false;
    std::uint64_t fe_cache_signature = 0;
    bool fe_cache_signature_ready = false;
    auto ensure_fe_cache_signature = [&]() {
        if (!fe_cache_signature_ready) {
            fe_cache_signature = hash_fe_structure_cache_signature(
                fes, weights, options_.drop_singletons, options_.weights_are_frequencies);
            fe_cache_signature_ready = true;
        }
    };

    auto choose_method_used = [&](std::size_t dims,
                                  int threads,
                                  const HdfeOptions& tuned) -> AbsorptionMethod {
        if (tuned.absorption_method != AbsorptionMethod::Auto) {
            return tuned.absorption_method;
        }
        if (dims <= 1) {
            return AbsorptionMethod::GaussSeidel;
        }
        if (dims == 2) {
            return AbsorptionMethod::SymmetricGaussSeidel;
        }
        if (threads >= static_cast<int>(2 * dims)) {
            return AbsorptionMethod::Jacobi;
        }
        if (tuned.symmetric_sweep && dims > 1) {
            return AbsorptionMethod::SymmetricGaussSeidel;
        }
        return AbsorptionMethod::GaussSeidel;
    };

    auto run_partial = [&](const Eigen::Ref<const Eigen::VectorXd>& y_use,
                           const Eigen::Ref<const Eigen::MatrixXd>& X_use,
                           const std::vector<Eigen::VectorXi>& fes_use,
                           const Eigen::VectorXd* w_ptr,
                           const std::vector<Eigen::VectorXi>* c_ptr,
                           const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use) -> detail::AbsorptionResult {
        const int threads = resolve_threads(static_cast<int>(y_use.size()),
                                            static_cast<int>(fes_use.size()));
        threads_used_ = threads;
#ifdef HDFE_USE_OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, threads));
        Eigen::setNbThreads(std::max(1, threads));
#endif

        HdfeOptions tuned = options_;
        tuned.num_threads = threads;
        const std::vector<detail::HeterogeneousSlopeTerm>& slope_terms =
            slopes_use ? *slopes_use : empty_slopes;

        detail::AbsorptionResult absorption =
            detail::absorb_fixed_effects_v6(y_use, X_use, fes_use, w_ptr, tuned,
                                            AbsorptionMethod::Auto, slope_terms);
        method_used_ = choose_method_used(fes_use.size(), threads, tuned);
        if (!slope_terms.empty() && method_used_ == AbsorptionMethod::Jacobi) {
            method_used_ = fes_use.size() > 1 ? AbsorptionMethod::SymmetricGaussSeidel
                                               : AbsorptionMethod::GaussSeidel;
        }
        gpu_used_ = absorption.gpu_used;
        gpu_status_code_ = absorption.gpu_status_code;
        gpu_attempted_ = absorption.gpu_attempted;
        gpu_absorption_converged_ = absorption.gpu_absorption_converged;
        gpu_absorption_iterations_ = absorption.gpu_absorption_iterations;

        // Populate a minimal results_ object for Stata/Python consumers that expect e(df_a),
        // singleton counts, and convergence diagnostics.
        results_.coefficients.resize(0);
        results_.std_errors.resize(0);
        results_.tvalues.resize(0);
        results_.pvalues.resize(0);
        results_.conf_int.resize(0, 0);
        results_.covariance.resize(0, 0);
        results_.residuals.resize(0);
        results_.omitted_reason.clear();
        results_.nobs = static_cast<int>(y_use.size());
        results_.nobs_full = results_.nobs;
        results_.num_singletons = 0;
        results_.nobs_effective =
            (options_.weights_are_frequencies && w_ptr) ? w_ptr->sum()
                                                        : static_cast<double>(results_.nobs);
        results_.nobs_full_effective = results_.nobs_effective;
        results_.num_singletons_effective = 0.0;
        results_.sample_index.resize(0);
        results_.df_resid = 0.0;
        results_.df_resid_unadj = 0.0;
        results_.df_m = 0.0;
        results_.df_a = 0.0;
        results_.df_a_levels = 0.0;
        results_.df_a_exact = 0.0;
        results_.df_a_nested = 0.0;
        results_.r2 = 0.0;
        results_.r2_within = 0.0;
        results_.sigma2 = 0.0;
        results_.rss = 0.0;
        results_.tss = 0.0;
        results_.tss_within = 0.0;
        results_.fe_num_levels = absorption.fe_levels;
        results_.fe_base_levels = absorption.fe_levels;
        results_.fe_redundant.clear();
        results_.fe_num_coefs.clear();
        results_.fe_inexact.clear();
        results_.fe_nested.clear();
        results_.num_iterations = absorption.iterations;
        results_.groupvar.resize(0);
        results_.fe_effects.clear();
        results_.fe_save_effects.clear();
        results_.fe_recovery_iterations = 0;
        results_.fe_recovery_max_delta = 0.0;
        results_.fe_recovery_converged = true;
        results_.converged = absorption.converged;
        results_.num_clusters = 0;
        results_.cluster_counts.clear();
        results_.cluster_combo_counts.clear();
        results_.cluster_scale = 1.0;

        // Degrees-of-freedom adjustments (hdfe-style; mirrors the reghdfe logic in fit()).
        int df_a_levels = 0;
        int df_a_exact = 0;
        int nested_dof_levels = 0;
        int nested_dof_coefs = 0;
        if (!fes_use.empty()) {
            Eigen::VectorXi groupvar;
            Eigen::VectorXi* groupvar_ptr = options_.save_groupvar ? &groupvar : nullptr;
            FeDofInfo dof =
                compute_fe_dof_reghdfe(fes_use, absorption.fe_levels, options_.dof_method, groupvar_ptr,
                                       tuned.num_threads);
            std::vector<uint8_t> nested_flags(dof.levels.size(), 0);
            std::vector<uint8_t> slope_only_flags(dof.levels.size(), 0);
            for (const auto& slope : slope_terms) {
                if (!slope.include_intercept &&
                    slope.fe_index >= 0 &&
                    slope.fe_index < static_cast<int>(slope_only_flags.size())) {
                    slope_only_flags[static_cast<std::size_t>(slope.fe_index)] = 1;
                }
            }
            if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
                !c_ptr->empty()) {
                for (const auto& cl : *c_ptr) {
                    for (std::size_t d = 0; d < fes_use.size(); ++d) {
                        if (nested_flags[d]) {
                            continue;
                        }
                        if (fe_nested_within_cluster(fes_use[d], cl)) {
                            nested_flags[d] = 1;
                        }
                    }
                }
            }

            // If some fixed effects are nested within the clustering variable, reghdfe does not
            // use them to compute mobility-group redundancies for the remaining dimensions.
            if (options_.dof_method != DofAdjustmentMethod::None) {
                bool any_nested = false;
                for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                    if (nested_flags[d]) {
                        any_nested = true;
                        break;
                    }
                }
                if (any_nested) {
                    std::vector<Eigen::VectorXi> fes_nonnested;
                    std::vector<int> levels_nonnested;
                    std::vector<int> map_nonnested;
                    fes_nonnested.reserve(fes_use.size());
                    levels_nonnested.reserve(fes_use.size());
                    map_nonnested.reserve(fes_use.size());
                    for (std::size_t d = 0; d < fes_use.size(); ++d) {
                        if (nested_flags[d]) {
                            continue;
                        }
                        fes_nonnested.push_back(fes_use[d]);
                        levels_nonnested.push_back(dof.levels[d]);
                        map_nonnested.push_back(static_cast<int>(d));
                    }
                    if (!fes_nonnested.empty()) {
                        const FeDofInfo dof_nonnested =
                            compute_fe_dof_reghdfe(fes_nonnested, levels_nonnested,
                                                   options_.dof_method, nullptr, tuned.num_threads);
                        for (std::size_t pos = 0; pos < map_nonnested.size(); ++pos) {
                            const int orig = map_nonnested[pos];
                            dof.redundant[static_cast<std::size_t>(orig)] =
                                dof_nonnested.redundant[pos];
                        }
                    }
                }
            }
            int nested_total = 0;
            if (options_.dof_method != DofAdjustmentMethod::None) {
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    if (nested_flags[d]) {
                        nested_total += dof.levels[d];
                        dof.redundant[d] = dof.levels[d];
                        dof.num_coefs[d] = 0;
                    }
                }
                std::vector<int> intercept_index;
                intercept_index.reserve(dof.levels.size());
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    if (!nested_flags[d] && !slope_only_flags[d]) {
                        intercept_index.push_back(static_cast<int>(d));
                    }
                }
                if (!intercept_index.empty()) {
                    bool skip_next = (nested_total == 0);
                    for (const int idx : intercept_index) {
                        if (skip_next) {
                            skip_next = false;
                            continue;
                        }
                        if (dof.redundant[static_cast<std::size_t>(idx)] < 1) {
                            dof.redundant[static_cast<std::size_t>(idx)] = 1;
                        }
                    }
                }
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    dof.num_coefs[d] =
                        std::max(0, dof.levels[d] - dof.redundant[d]);
                }
                dof.inexact.assign(dof.levels.size(), 0);
                if (options_.dof_method == DofAdjustmentMethod::Pairwise ||
                    options_.dof_method == DofAdjustmentMethod::All ||
                    options_.dof_method == DofAdjustmentMethod::FirstPair) {
                    int intercept_count = 0;
                    for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                        if (nested_flags[d] || slope_only_flags[d]) {
                            continue;
                        }
                        ++intercept_count;
                        if (intercept_count > 2) {
                            dof.inexact[d] = 1;
                        }
                    }
                }
            } else {
                dof.inexact.assign(dof.levels.size(), 0);
            }
            apply_heterogeneous_slope_dof(dof, fes_use, slope_terms, w_ptr, nested_flags);

            df_a_levels = 0;
            df_a_exact = 0;
            for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                df_a_levels += dof.levels[d];
                df_a_exact += dof.num_coefs[d];
            }
            dof.df_a_levels = df_a_levels;
            dof.df_a_exact = df_a_exact;
            dof.df_a = df_a_exact;
            results_.df_a_levels = static_cast<double>(df_a_levels);
            results_.df_a_exact = static_cast<double>(df_a_exact);
            results_.fe_num_levels = dof.levels;
            results_.fe_base_levels = absorption.fe_levels;
            results_.fe_redundant = dof.redundant;
            results_.fe_num_coefs = dof.num_coefs;
            results_.fe_inexact = dof.inexact;
            results_.fe_nested.assign(dof.levels.size(), 0);
            if (groupvar_ptr) {
                results_.groupvar = std::move(groupvar);
            }
            if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
                !c_ptr->empty()) {
                for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                    if (nested_flags[d]) {
                        nested_dof_coefs += nested_coefs_for_heterogeneous_dim(
                            d, slope_terms, dof.num_coefs);
                        nested_dof_levels += nested_levels_for_heterogeneous_dim(
                            d, slope_terms, absorption.fe_levels, dof.levels);
                        results_.fe_nested[d] = 1;
                    }
                }
            }
        }

        const int df_a_report = df_a_exact;
        const int nested_report = nested_dof_levels;
        results_.df_a = static_cast<double>(df_a_report);
        results_.df_a_nested = static_cast<double>(nested_report);

#ifdef HDFE_USE_OPENMP
        Eigen::setNbThreads(1);
#endif
        return absorption;
    };

    if (!has_slopes && fe_cache_cfg.mode != "off" && has_fes) {
        const bool allow_read =
            (fe_cache_cfg.mode == "read" || fe_cache_cfg.mode == "auto");
        if (allow_read) {
            ensure_fe_cache_signature();
            fe_cache_hit =
                read_fe_structure_cache(fe_cache_cfg.path, fe_cache_signature, nobs_full,
                                        options_.drop_singletons,
                                        static_cast<int>(fes.size()), fe_cache);
        }
    }

    detail::AbsorptionResult absorption;
    if (fe_cache_hit) {
        if (options_.drop_singletons && fe_cache.nobs < nobs_full) {
            kept_idx = std::move(fe_cache.kept_idx);
            singletons_dropped_rows = nobs_full - fe_cache.nobs;
            y_work = filter_vector(y, kept_idx);
            X_work = filter_rows(X, kept_idx);
            fes_work = std::move(fe_cache.fe_group_ids);

            const Eigen::VectorXd* w_ptr = nullptr;
            if (weights) {
                weights_work = filter_vector(*weights, kept_idx);
                w_ptr = &(*weights_work);
            }

            const std::vector<Eigen::VectorXi>* c_ptr = clusters;
            if (clusters && !clusters->empty()) {
                std::vector<Eigen::VectorXi> clusters_tmp;
                clusters_tmp.reserve(clusters->size());
                for (const auto& c : *clusters) {
                    clusters_tmp.push_back(filter_vector(c, kept_idx));
                }
                clusters_work = std::move(clusters_tmp);
                c_ptr = &(*clusters_work);
            }

            const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use = slopes_in;
            if (has_slopes) {
                slopes_work = filter_slope_terms(*slopes_in, kept_idx);
                slopes_use = &(*slopes_work);
            }
            absorption = run_partial(*y_work, *X_work, *fes_work, w_ptr, c_ptr, slopes_use);
            if (options_.weights_are_frequencies && weights_work) {
                singletons_dropped_effective =
                    std::max(0.0, nobs_full_effective - weights_work->sum());
            } else {
                singletons_dropped_effective = static_cast<double>(singletons_dropped_rows);
            }
        } else {
            fes_work = std::move(fe_cache.fe_group_ids);
            absorption = run_partial(y, X, *fes_work, weights, clusters, slopes_in);
        }
    } else if (options_.drop_singletons && has_fes) {
        std::vector<uint8_t> keep;
        if (options_.weights_are_frequencies && weights) {
            keep = compute_keep_mask_drop_singletons_fweights(
                fes, *weights, kMaxSingletonIterations, &singletons_dropped_rows,
                &singletons_dropped_effective);
        } else {
            keep = compute_keep_mask_drop_singletons(fes, kMaxSingletonIterations,
                                                     &singletons_dropped_rows,
                                                     options_.num_threads);
            singletons_dropped_effective = static_cast<double>(singletons_dropped_rows);
        }
        if (singletons_dropped_rows > 0) {
            kept_idx = build_keep_indices(keep);
            y_work = filter_vector(y, kept_idx);
            X_work = filter_rows(X, kept_idx);

            std::vector<Eigen::VectorXi> fes_tmp;
            fes_tmp.reserve(fes.size());
            for (const auto& fe : fes) {
                fes_tmp.push_back(filter_vector(fe, kept_idx));
            }
            fes_work = std::move(fes_tmp);

            const Eigen::VectorXd* w_ptr = nullptr;
            if (weights) {
                weights_work = filter_vector(*weights, kept_idx);
                w_ptr = &(*weights_work);
            }

            const std::vector<Eigen::VectorXi>* c_ptr = clusters;
            if (clusters && !clusters->empty()) {
                std::vector<Eigen::VectorXi> clusters_tmp;
                clusters_tmp.reserve(clusters->size());
                for (const auto& c : *clusters) {
                    clusters_tmp.push_back(filter_vector(c, kept_idx));
                }
                clusters_work = std::move(clusters_tmp);
                c_ptr = &(*clusters_work);
            }

            const std::vector<detail::HeterogeneousSlopeTerm>* slopes_use = slopes_in;
            if (has_slopes) {
                slopes_work = filter_slope_terms(*slopes_in, kept_idx);
                slopes_use = &(*slopes_work);
            }
            absorption = run_partial(*y_work, *X_work, *fes_work, w_ptr, c_ptr, slopes_use);
        } else {
            absorption = run_partial(y, X, fes, weights, clusters, slopes_in);
        }
    } else {
        absorption = run_partial(y, X, fes, weights, clusters, slopes_in);
    }

    const bool allow_fe_cache_write =
        fe_cache_cfg.mode == "write" ||
        (fe_cache_cfg.mode == "auto" && fe_cache_cfg.allow_auto_write);
    if (!has_slopes && !fe_cache_hit && allow_fe_cache_write && has_fes && !fe_cache_cfg.path.empty()) {
        ensure_fe_cache_signature();
        const std::vector<Eigen::VectorXi>* fes_used =
            fes_work ? &(*fes_work) : &fes;
        if (!fes_used->empty()) {
            std::vector<Eigen::VectorXi> fe_group_ids;
            std::vector<int> num_groups;
            fe_group_ids.reserve(fes_used->size());
            num_groups.reserve(fes_used->size());
            for (const auto& fe : *fes_used) {
                FeIndexerLite idx = build_indexer_lite(fe);
                Eigen::VectorXi ids(static_cast<int>(idx.group_ids.size()));
                for (int i = 0; i < ids.size(); ++i) {
                    ids(i) = idx.group_ids[static_cast<std::size_t>(i)];
                }
                fe_group_ids.push_back(std::move(ids));
                num_groups.push_back(idx.num_groups);
            }
            write_fe_structure_cache(fe_cache_cfg.path, fe_cache_signature, nobs_full,
                                     options_.drop_singletons, kept_idx, num_groups,
                                     fe_group_ids);
        }
    }

    results_.nobs_full = nobs_full;
    results_.num_singletons = singletons_dropped_rows;
    const double nobs_effective =
        (options_.weights_are_frequencies && weights)
            ? (weights_work ? weights_work->sum() : weights->sum())
            : static_cast<double>(results_.nobs);
    results_.nobs_effective = nobs_effective;
    results_.nobs_full_effective = nobs_full_effective;
    results_.num_singletons_effective =
        options_.weights_are_frequencies
            ? std::max(0.0, nobs_full_effective - nobs_effective)
            : singletons_dropped_effective;
    results_.sample_index.resize(results_.nobs);
    if (!kept_idx.empty()) {
        for (int i = 0; i < results_.nobs; ++i) {
            results_.sample_index(i) = kept_idx[static_cast<std::size_t>(i)];
        }
    } else {
        for (int i = 0; i < results_.nobs; ++i) {
            results_.sample_index(i) = i;
        }
    }

    return absorption;
}

void HdfeRegressorV11::fit_grouped(const Eigen::Ref<const Eigen::VectorXd>& y,
                                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                                  const std::vector<Eigen::VectorXi>& fes,
                                  const Eigen::VectorXi& group_ids,
                                  const Eigen::VectorXi* individual_ids,
                                  GroupAggregation aggregation,
                                  const Eigen::VectorXd* weights,
                                  const std::vector<Eigen::VectorXi>* clusters) {
    gpu_used_ = false;
    gpu_status_code_ = 0;
    gpu_attempted_ = false;
    gpu_absorption_converged_ = false;
    gpu_absorption_iterations_ = 0;
    if (!individual_ids) {
        GroupCollapsedData collapsed =
            collapse_group_long_format(y, X, fes, group_ids, nullptr, -1, aggregation, weights, clusters);
        const Eigen::VectorXd* w_ptr = collapsed.weights ? &(*collapsed.weights) : nullptr;
        const std::vector<Eigen::VectorXi>* c_ptr = collapsed.clusters ? &(*collapsed.clusters) : nullptr;
        fit(collapsed.y, collapsed.X, collapsed.standard_fes, w_ptr, c_ptr, nullptr, {});
        results_.sample_index.resize(0);
        return;
    }

    const MobilityProfileConfig mobility_cfg = load_mobility_profile_config();

    if (fes.empty()) {
        throw std::runtime_error("fes must include the individual_ids when using group/individual");
    }
    const int individual_fe_index = find_identical_fe(*individual_ids, fes);
    if (individual_fe_index < 0) {
        throw std::runtime_error("individual_ids must also be included in fes");
    }

    if (options_.retain_fixed_effects) {
        throw std::runtime_error("retain_fes is not supported with group/individual fixed effects");
    }

    GroupCollapsedData collapsed = collapse_group_long_format(
        y, X, fes, group_ids, individual_ids, individual_fe_index, aggregation, weights, clusters);
    const int collapsed_full = static_cast<int>(collapsed.y.size());
    const double nobs_full_effective =
        (options_.weights_are_frequencies && collapsed.weights) ? collapsed.weights->sum()
                                                                : static_cast<double>(collapsed_full);
    int group_singletons_dropped = 0;

    if (options_.drop_singletons) {
        const GroupSingletonDropResult dropped =
            drop_singletons_group_individual(collapsed.standard_fes, collapsed.gi);
        group_singletons_dropped = dropped.dropped_groups;
        if (dropped.dropped_groups > 0) {
            collapsed =
                filter_group_collapsed(collapsed, dropped.keep_groups, dropped.individual_degrees, aggregation);
        }
    }

    const Eigen::VectorXd* w_ptr = collapsed.weights ? &(*collapsed.weights) : nullptr;
    const std::vector<Eigen::VectorXi>* c_ptr = collapsed.clusters ? &(*collapsed.clusters) : nullptr;

    const bool has_fes = true;
    const bool drop_intercept = options_.fit_intercept && has_fes;

    Eigen::VectorXd y_work = collapsed.y;
    Eigen::MatrixXd X_work = collapsed.X;
    std::vector<Eigen::VectorXi> standard_fes_work = collapsed.standard_fes;
    hdfe::detail::GroupIndividualStructure gi_work = std::move(collapsed.gi);

    Eigen::MatrixXd design =
        drop_intercept ? X_work : maybe_add_intercept(X_work, options_.fit_intercept);
    const int design_cols = static_cast<int>(design.cols());

    const int threads = resolve_threads(static_cast<int>(y_work.size()),
                                        static_cast<int>(standard_fes_work.size() + 1));
    threads_used_ = threads;
#ifdef HDFE_USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, threads));
    Eigen::setNbThreads(std::max(1, threads));
#endif

    HdfeOptions tuned = options_;
    tuned.num_threads = threads;

    MobilityHint mobility_hint;
    bool have_mobility_hint = false;
    bool mobility_profile_match = false;
    MobilityProfile profile;
    bool have_profile = false;
    if (mobility_cfg.mode == "read" || mobility_cfg.mode == "auto") {
        if (load_mobility_profile(mobility_cfg.path, profile)) {
            have_profile = true;
        }
    }
    if (have_profile) {
        bool match = false;
        if (profile.signature != 0 || profile.signature_canon != 0) {
            const std::uint64_t sig =
                hash_fe_signature(standard_fes_work, options_.drop_singletons);
            if (profile.signature != 0 && profile.signature == sig) {
                match = true;
            }
            if (!match && profile.signature_canon != 0 &&
                profile.signature_canon == sig) {
                match = true;
            }
        }
        if (match) {
            mobility_hint = build_mobility_hint(profile);
            have_mobility_hint = true;
            mobility_profile_match = true;
        }
    }
    if (have_mobility_hint && !mobility_hint.sweep_order.empty()) {
        tuned.sweep_order_override = mobility_hint.sweep_order;
    }
    if (have_mobility_hint && mobility_hint.force_symmetric &&
        options_.absorption_method == AbsorptionMethod::Auto &&
        !options_.symmetric_sweep) {
        tuned.symmetric_sweep = true;
    }

    const bool allow_profile_write =
        mobility_cfg.mode == "write" ||
        (mobility_cfg.mode == "auto" && mobility_cfg.allow_auto_write && !mobility_profile_match);

    const bool benchmark_methods =
        allow_profile_write && !have_mobility_hint &&
        options_.absorption_method == AbsorptionMethod::Auto &&
        !options_.symmetric_sweep && !options_.use_sparse_solver &&
        standard_fes_work.size() + 1 > 1;

    AbsorptionMethod preferred_method = AbsorptionMethod::Auto;
    if (have_mobility_hint && mobility_hint.has_method &&
        mobility_hint.preferred_method != AbsorptionMethod::Jacobi) {
        preferred_method = mobility_hint.preferred_method;
    }
    AbsorptionMethod selected_method = AbsorptionMethod::Auto;
    if (options_.absorption_method != AbsorptionMethod::Auto) {
        selected_method = options_.absorption_method;
    } else if (options_.symmetric_sweep) {
        selected_method = AbsorptionMethod::SymmetricGaussSeidel;
    } else if (preferred_method != AbsorptionMethod::Auto) {
        selected_method = preferred_method;
    }
    if (selected_method == AbsorptionMethod::Auto ||
        selected_method == AbsorptionMethod::Jacobi) {
        selected_method = select_method(standard_fes_work.size() + 1);
    }
    method_used_ = selected_method;
    if (tuned.symmetric_sweep && options_.absorption_method == AbsorptionMethod::Auto &&
        method_used_ != AbsorptionMethod::SymmetricGaussSeidel &&
        standard_fes_work.size() + 1 > 1) {
        method_used_ = AbsorptionMethod::SymmetricGaussSeidel;
    }
    tuned.absorption_method = method_used_;

    detail::AbsorptionResult absorption;
    bool absorption_ready = false;
    if (benchmark_methods) {
        std::vector<AbsorptionMethod> candidates;
        candidates.reserve(2);
        candidates.push_back(AbsorptionMethod::GaussSeidel);
        candidates.push_back(AbsorptionMethod::SymmetricGaussSeidel);

        double best_time = std::numeric_limits<double>::infinity();
        AbsorptionMethod best_method = AbsorptionMethod::Auto;
        for (const auto method : candidates) {
            HdfeOptions bench_opts = tuned;
            bench_opts.absorption_method = method;
            bench_opts.symmetric_sweep = (method == AbsorptionMethod::SymmetricGaussSeidel);

            const auto t0 = std::chrono::steady_clock::now();
            detail::AbsorptionResult res =
                detail::absorb_fixed_effects_group_individual(
                    y_work, design, standard_fes_work, gi_work, w_ptr, bench_opts, method);
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (!res.converged) {
                continue;
            }
            if (elapsed < best_time) {
                best_time = elapsed;
                best_method = method;
                absorption = std::move(res);
                absorption_ready = true;
            }
        }
        if (absorption_ready) {
            method_used_ = best_method;
            tuned.absorption_method = best_method;
            tuned.symmetric_sweep = (best_method == AbsorptionMethod::SymmetricGaussSeidel);
        }
    }
    if (!absorption_ready) {
        absorption = detail::absorb_fixed_effects_group_individual(
            y_work, design, standard_fes_work, gi_work, w_ptr, tuned, method_used_);
    }
    gpu_used_ = absorption.gpu_used;
    gpu_status_code_ = absorption.gpu_status_code;
    gpu_attempted_ = absorption.gpu_attempted;
    gpu_absorption_converged_ = absorption.gpu_absorption_converged;
    gpu_absorption_iterations_ = absorption.gpu_absorption_iterations;
    if (!absorption.converged) {
        if (absorption.gpu_status_code == 2) {
            throw std::runtime_error("Requested GPU backend was unavailable during group/individual HDFE absorption");
        }
        if (absorption.gpu_status_code == 3) {
            throw std::runtime_error("Requested GPU backend did not converge during group/individual HDFE absorption");
        }
        if (absorption.gpu_status_code == 4) {
            throw std::runtime_error("Requested GPU backend failed during group/individual HDFE absorption");
        }
        // Pure CPU non-convergence: preserve reference best-effort behaviour (see fit()).
    }

    double sum_weights_for_stats = static_cast<double>(y_work.size());
    const double total_tss =
        compute_total_sum_of_squares(y_work, w_ptr, &sum_weights_for_stats);
    const double within_tss = weighted_sum_of_squares(absorption.y_tilde, w_ptr);

    Eigen::MatrixXd transformed_X = absorption.X_tilde.leftCols(design_cols);
    const int transformed_full_cols = static_cast<int>(transformed_X.cols());
    const bool transformed_has_intercept =
        (options_.fit_intercept && !drop_intercept && transformed_full_cols > 0);
    const int transformed_slope_cols =
        transformed_has_intercept ? (transformed_full_cols - 1) : transformed_full_cols;

    // Drop collinear slopes after partialling-out.
    constexpr double kFeCollinearTol = 1e-9;
    const double rank_collinear_tol = std::max(1e-6, tuned.tol * 10.0);
    std::vector<int> kept_slope_cols;
    std::vector<int> omitted_slope_cols;
    std::vector<int> omitted_fe_slope_cols;
    std::vector<int> omitted_rank_slope_cols;
    kept_slope_cols.reserve(static_cast<std::size_t>(std::max(0, transformed_slope_cols)));
    omitted_slope_cols.reserve(static_cast<std::size_t>(std::max(0, transformed_slope_cols)));
    for (int j = 0; j < transformed_slope_cols; ++j) {
        const double raw_ss = X_work.col(j).squaredNorm();
        const double tilde_ss = transformed_X.col(j).squaredNorm();
        const bool finite = std::isfinite(raw_ss) && std::isfinite(tilde_ss);
        const bool collinear =
            finite && (raw_ss <= 0.0 ? (tilde_ss <= kFeCollinearTol)
                                     : (tilde_ss <= kFeCollinearTol * raw_ss));
        if (collinear) {
            omitted_fe_slope_cols.push_back(j);
            omitted_slope_cols.push_back(j);
        } else {
            kept_slope_cols.push_back(j);
        }
    }

    Eigen::MatrixXd xtx_used;
    Eigen::VectorXd xty_used;
    bool have_precomputed = false;

    const bool use_precomputed = (tuned.se_type != StandardErrorType::Cluster);
    if (use_precomputed) {
        const int k_initial = static_cast<int>(kept_slope_cols.size());
        const int intercept_col = transformed_has_intercept ? (transformed_full_cols - 1) : -1;
        const CrossproductResult cp = compute_crossproducts_selected(
            absorption.y_tilde, transformed_X, kept_slope_cols, intercept_col, w_ptr,
            transformed_has_intercept, tuned.num_threads);

        std::vector<int> keep_positions;
        keep_positions.reserve(kept_slope_cols.size());
        if (k_initial > 0) {
            Eigen::MatrixXd gram = cp.xtx.topLeftCorner(k_initial, k_initial);
            if (transformed_has_intercept) {
                double denom = cp.sum_w;
                if (!(denom > 0.0)) {
                    denom = 1.0;
                }
                gram.noalias() -= (cp.sum_x * cp.sum_x.transpose()) / denom;
            }
            const std::vector<uint8_t> drop_mask =
                compute_rank_collinearity_mask_from_gram(
                    gram, kept_slope_cols, rank_collinear_tol, options_.collinear_priority);

            std::vector<int> new_kept;
            std::vector<int> new_omitted;
            new_kept.reserve(kept_slope_cols.size());
            new_omitted.reserve(kept_slope_cols.size());
            for (std::size_t pos = 0; pos < kept_slope_cols.size(); ++pos) {
                const int idx = kept_slope_cols[pos];
                if (drop_mask[static_cast<std::size_t>(pos)]) {
                    new_omitted.push_back(idx);
                } else {
                    new_kept.push_back(idx);
                    keep_positions.push_back(static_cast<int>(pos));
                }
            }
            kept_slope_cols.swap(new_kept);
            omitted_rank_slope_cols.insert(omitted_rank_slope_cols.end(),
                                           new_omitted.begin(), new_omitted.end());
            omitted_slope_cols.insert(omitted_slope_cols.end(),
                                      new_omitted.begin(), new_omitted.end());
        }

        std::vector<int> xtx_cols = keep_positions;
        if (intercept_col >= 0) {
            xtx_cols.push_back(k_initial);
        }
        const int p_used = static_cast<int>(xtx_cols.size());
        xtx_used.resize(p_used, p_used);
        xty_used.resize(p_used);
        for (int i = 0; i < p_used; ++i) {
            const int col_i = xtx_cols[static_cast<std::size_t>(i)];
            xty_used(i) = cp.xty(col_i);
            for (int j = 0; j < p_used; ++j) {
                const int col_j = xtx_cols[static_cast<std::size_t>(j)];
                xtx_used(i, j) = cp.xtx(col_i, col_j);
            }
        }
        have_precomputed = true;
    } else if (!kept_slope_cols.empty()) {
        const std::vector<uint8_t> drop_mask = compute_rank_collinearity_mask(
            transformed_X, kept_slope_cols, transformed_slope_cols, transformed_has_intercept,
            w_ptr, rank_collinear_tol, tuned.num_threads, options_.collinear_priority);

        std::vector<int> new_kept;
        std::vector<int> new_omitted;
        new_kept.reserve(kept_slope_cols.size());
        new_omitted.reserve(kept_slope_cols.size());
        for (const int idx : kept_slope_cols) {
            if (idx >= 0 && idx < transformed_slope_cols &&
                drop_mask[static_cast<std::size_t>(idx)]) {
                new_omitted.push_back(idx);
            } else {
                new_kept.push_back(idx);
            }
        }
        kept_slope_cols.swap(new_kept);
        omitted_rank_slope_cols.insert(omitted_rank_slope_cols.end(),
                                       new_omitted.begin(), new_omitted.end());
        omitted_slope_cols.insert(omitted_slope_cols.end(),
                                  new_omitted.begin(), new_omitted.end());
    }

    Eigen::MatrixXd transformed_X_used;
    transformed_X_used.resize(transformed_X.rows(),
                              static_cast<int>(kept_slope_cols.size()) +
                                  (transformed_has_intercept ? 1 : 0));
    for (int out_j = 0; out_j < static_cast<int>(kept_slope_cols.size()); ++out_j) {
        transformed_X_used.col(out_j) =
            transformed_X.col(kept_slope_cols[static_cast<std::size_t>(out_j)]);
    }
    if (transformed_has_intercept) {
        transformed_X_used.col(static_cast<int>(kept_slope_cols.size())) =
            transformed_X.col(transformed_full_cols - 1);
    }
    const int df_m_effective = static_cast<int>(kept_slope_cols.size());

    detail::OlsResult ols_result;
    if (transformed_X_used.cols() == 0) {
        detail::OlsResult empty;
        empty.coefficients.resize(0);
        empty.std_errors.resize(0);
        empty.tvalues.resize(0);
        empty.pvalues.resize(0);
        empty.conf_int.resize(0, 2);
        empty.residuals = absorption.y_tilde;
        empty.df_resid = static_cast<double>(absorption.y_tilde.size());
        empty.rss = weighted_sum_of_squares(empty.residuals, w_ptr);
        empty.tss = total_tss;
        empty.within_tss = within_tss;
        empty.r2 = (total_tss > 0.0) ? 1.0 - empty.rss / total_tss : 1.0;
        empty.r2_within = (within_tss > 0.0) ? 1.0 - empty.rss / within_tss : 1.0;
        empty.sigma2 = 0.0;
        empty.nobs = static_cast<int>(absorption.y_tilde.size());
        ols_result = std::move(empty);
    } else if (tuned.se_type == StandardErrorType::Cluster) {
        if (!c_ptr || c_ptr->empty()) {
            throw std::runtime_error(
                "Cluster-robust errors requested but no cluster variables were provided");
        }
        if (c_ptr->size() == 1) {
            ols_result = detail::run_ols(absorption.y_tilde, transformed_X_used, w_ptr, &(*c_ptr)[0],
                                         tuned.se_type, total_tss, within_tss);
        } else {
            ols_result = detail::run_ols_multiway(absorption.y_tilde, transformed_X_used, w_ptr, c_ptr,
                                                 tuned.se_type, total_tss, within_tss,
                                                 tuned.ssc_g_df, tuned.ssc_g_adj);
        }
    } else if (have_precomputed &&
               xtx_used.rows() == transformed_X_used.cols() &&
               xty_used.size() == transformed_X_used.cols()) {
        ols_result = detail::run_ols_fast_from_xtx(absorption.y_tilde, transformed_X_used, w_ptr,
                                                   tuned.se_type, total_tss, within_tss,
                                                   xtx_used, xty_used, -1.0,
                                                   tuned.weights_are_frequencies);
    } else {
        ols_result = detail::run_ols(absorption.y_tilde, transformed_X_used, w_ptr, nullptr, tuned.se_type,
                                     total_tss, within_tss, -1.0,
                                     tuned.weights_are_frequencies);
    }

    apply_common_postprocessing(y_work, X_work, w_ptr, absorption.fe_levels, ols_result);
    const double nobs_effective =
        (options_.weights_are_frequencies && w_ptr) ? w_ptr->sum()
                                                    : static_cast<double>(y_work.size());
    results_.nobs_effective = nobs_effective;
    results_.nobs_full_effective = nobs_full_effective;
    results_.num_singletons_effective = std::max(0.0, nobs_full_effective - nobs_effective);
    results_.vcv_psd_fixed = false;
    results_.nobs_full = collapsed_full;
    results_.num_singletons = group_singletons_dropped;
    results_.num_iterations = absorption.iterations;
    results_.converged = absorption.converged;
    // For group/individual mode, map each collapsed group observation back to a representative
    // row in the original long-format input.
    if (static_cast<int>(collapsed.rep_row.size()) != results_.nobs) {
        throw std::runtime_error("Unexpected rep_row size in grouped regression");
    }
    results_.sample_index.resize(results_.nobs);
    for (int i = 0; i < results_.nobs; ++i) {
        results_.sample_index(i) = collapsed.rep_row[static_cast<std::size_t>(i)];
    }
    if (allow_profile_write) {
        MobilityProfile profile = compute_mobility_profile(
            standard_fes_work, collapsed_full, group_singletons_dropped,
            options_.drop_singletons, absorption.sweep_order_used);
        if (options_.absorption_method == AbsorptionMethod::Auto &&
            !options_.symmetric_sweep) {
            profile.suggest_method = method_used_;
            profile.suggest_symmetric =
                (method_used_ == AbsorptionMethod::SymmetricGaussSeidel);
        }
        profile.suggest_use_sparse = tuned.use_sparse_solver;
        profile.suggest_use_krylov = tuned.use_krylov;
        write_mobility_profile(mobility_cfg.path, profile);
    }

    if ((!omitted_slope_cols.empty()) &&
        results_.coefficients.size() == static_cast<int>(kept_slope_cols.size()) +
                                             (transformed_has_intercept ? 1 : 0)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const int full_cols = transformed_full_cols;
        Eigen::VectorXd coef_full = Eigen::VectorXd::Zero(full_cols);
        Eigen::VectorXd se_full = Eigen::VectorXd::Constant(full_cols, nan);
        Eigen::VectorXd t_full = Eigen::VectorXd::Constant(full_cols, nan);
        Eigen::VectorXd p_full = Eigen::VectorXd::Constant(full_cols, nan);
        Eigen::MatrixXd ci_full = Eigen::MatrixXd::Constant(full_cols, 2, nan);
        Eigen::MatrixXd cov_full = Eigen::MatrixXd::Zero(full_cols, full_cols);
        Eigen::MatrixXd cov_used = results_.covariance;

        for (int pos = 0; pos < static_cast<int>(kept_slope_cols.size()); ++pos) {
            const int j = kept_slope_cols[static_cast<std::size_t>(pos)];
            coef_full(j) = results_.coefficients(pos);
            se_full(j) = results_.std_errors(pos);
            t_full(j) = results_.tvalues(pos);
            p_full(j) = results_.pvalues(pos);
            ci_full(j, 0) = results_.conf_int(pos, 0);
            ci_full(j, 1) = results_.conf_int(pos, 1);

            for (int pos2 = 0; pos2 < static_cast<int>(kept_slope_cols.size()); ++pos2) {
                const int k = kept_slope_cols[static_cast<std::size_t>(pos2)];
                if (cov_used.rows() > pos && cov_used.cols() > pos2) {
                    cov_full(j, k) = cov_used(pos, pos2);
                }
            }
        }
        if (transformed_has_intercept) {
            const int intercept_full_idx = full_cols - 1;
            const int intercept_pos = static_cast<int>(kept_slope_cols.size());
            coef_full(intercept_full_idx) = results_.coefficients(intercept_pos);
            se_full(intercept_full_idx) = results_.std_errors(intercept_pos);
            t_full(intercept_full_idx) = results_.tvalues(intercept_pos);
            p_full(intercept_full_idx) = results_.pvalues(intercept_pos);
            ci_full(intercept_full_idx, 0) = results_.conf_int(intercept_pos, 0);
            ci_full(intercept_full_idx, 1) = results_.conf_int(intercept_pos, 1);

            if (cov_used.rows() > intercept_pos && cov_used.cols() > intercept_pos) {
                cov_full(intercept_full_idx, intercept_full_idx) =
                    cov_used(intercept_pos, intercept_pos);
            }
            for (int pos = 0; pos < static_cast<int>(kept_slope_cols.size()); ++pos) {
                const int j = kept_slope_cols[static_cast<std::size_t>(pos)];
                if (cov_used.rows() > pos && cov_used.cols() > intercept_pos) {
                    cov_full(j, intercept_full_idx) = cov_used(pos, intercept_pos);
                }
                if (cov_used.rows() > intercept_pos && cov_used.cols() > pos) {
                    cov_full(intercept_full_idx, j) = cov_used(intercept_pos, pos);
                }
            }
        }

        results_.coefficients = std::move(coef_full);
        results_.std_errors = std::move(se_full);
        results_.tvalues = std::move(t_full);
        results_.pvalues = std::move(p_full);
        results_.conf_int = std::move(ci_full);
        results_.covariance = std::move(cov_full);
    }

    if (drop_intercept) {
        const int slope_cols = static_cast<int>(results_.coefficients.size());
        const double y_mean = weighted_mean(y_work, w_ptr);
        const double denom = w_ptr ? w_ptr->sum() : static_cast<double>(y_work.size());
        Eigen::VectorXd x_means = Eigen::VectorXd::Zero(slope_cols);
        if (slope_cols > 0) {
            if (!w_ptr) {
                for (int j = 0; j < slope_cols; ++j) {
                    x_means(j) = X_work.col(j).mean();
                }
            } else if (denom > 0.0) {
                for (int j = 0; j < slope_cols; ++j) {
                    x_means(j) = X_work.col(j).dot(*w_ptr) / denom;
                }
            }
        }
        const double intercept = y_mean - x_means.dot(results_.coefficients);

        const int out_cols = slope_cols + 1;
        results_.coefficients.conservativeResize(out_cols);
        results_.coefficients(out_cols - 1) = intercept;

        const double nan = std::numeric_limits<double>::quiet_NaN();
        results_.std_errors.conservativeResize(out_cols);
        results_.std_errors(out_cols - 1) = nan;
        results_.tvalues.conservativeResize(out_cols);
        results_.tvalues(out_cols - 1) = nan;
        results_.pvalues.conservativeResize(out_cols);
        results_.pvalues(out_cols - 1) = nan;
        Eigen::MatrixXd conf_new = Eigen::MatrixXd::Constant(out_cols, 2, nan);
        conf_new.topRows(slope_cols) = results_.conf_int;
        results_.conf_int = std::move(conf_new);

        Eigen::MatrixXd cov_new = Eigen::MatrixXd::Constant(out_cols, out_cols, nan);
        cov_new.topLeftCorner(slope_cols, slope_cols) = results_.covariance;
        results_.covariance = std::move(cov_new);
    }

    const int omit_cols = static_cast<int>(results_.coefficients.size());
    results_.omitted_reason.assign(static_cast<std::size_t>(omit_cols), 0);
    for (const int idx : omitted_fe_slope_cols) {
        if (idx >= 0 && idx < omit_cols) {
            results_.omitted_reason[static_cast<std::size_t>(idx)] = 1;
        }
    }
    for (const int idx : omitted_rank_slope_cols) {
        if (idx >= 0 && idx < omit_cols &&
            results_.omitted_reason[static_cast<std::size_t>(idx)] == 0) {
            results_.omitted_reason[static_cast<std::size_t>(idx)] = 2;
        }
    }

    // Degrees-of-freedom adjustments (extend the standard FE DoF with the individual FE count).
    const int nobs = results_.nobs;
    const int df_m = df_m_effective;  // excludes constant
    results_.df_m = static_cast<double>(df_m);

    int df_a_levels = gi_work.num_individuals;
    int df_a_exact = gi_work.num_individuals;
    int nested_dof_levels = 0;
    int nested_dof_coefs = 0;
    if (!standard_fes_work.empty()) {
        std::vector<int> standard_levels;
        if (absorption.fe_levels.size() >= 1) {
            standard_levels.assign(absorption.fe_levels.begin(),
                                   absorption.fe_levels.end() - 1);
        }
        Eigen::VectorXi groupvar;
        Eigen::VectorXi* groupvar_ptr = options_.save_groupvar ? &groupvar : nullptr;
        FeDofInfo dof =
            compute_fe_dof_reghdfe(standard_fes_work, standard_levels, options_.dof_method, groupvar_ptr,
                                   tuned.num_threads);
        std::vector<uint8_t> nested_flags(dof.levels.size(), 0);
        if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
            !c_ptr->empty()) {
            for (const auto& cl : *c_ptr) {
                for (std::size_t d = 0; d < standard_fes_work.size(); ++d) {
                    if (nested_flags[d]) {
                        continue;
                    }
                    if (fe_nested_within_cluster(standard_fes_work[d], cl)) {
                        nested_flags[d] = 1;
                    }
                }
            }
        }

        // If some fixed effects are nested within the clustering variable, reghdfe does not
        // use them to compute mobility-group redundancies for the remaining dimensions.
        if (options_.dof_method != DofAdjustmentMethod::None) {
            bool any_nested = false;
            for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                if (nested_flags[d]) {
                    any_nested = true;
                    break;
                }
            }
            if (any_nested) {
                std::vector<Eigen::VectorXi> fes_nonnested;
                std::vector<int> levels_nonnested;
                std::vector<int> map_nonnested;
                fes_nonnested.reserve(standard_fes_work.size());
                levels_nonnested.reserve(standard_fes_work.size());
                map_nonnested.reserve(standard_fes_work.size());
                for (std::size_t d = 0; d < standard_fes_work.size(); ++d) {
                    if (nested_flags[d]) {
                        continue;
                    }
                    fes_nonnested.push_back(standard_fes_work[d]);
                    levels_nonnested.push_back(dof.levels[d]);
                    map_nonnested.push_back(static_cast<int>(d));
                }
                if (!fes_nonnested.empty()) {
                    const FeDofInfo dof_nonnested =
                        compute_fe_dof_reghdfe(fes_nonnested, levels_nonnested,
                                               options_.dof_method, nullptr, tuned.num_threads);
                    for (std::size_t pos = 0; pos < map_nonnested.size(); ++pos) {
                        const int orig = map_nonnested[pos];
                        dof.redundant[static_cast<std::size_t>(orig)] =
                            dof_nonnested.redundant[pos];
                    }
                }
            }
        }
        int nested_total = 0;
        if (options_.dof_method != DofAdjustmentMethod::None) {
            for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                if (nested_flags[d]) {
                    nested_total += dof.levels[d];
                    dof.redundant[d] = dof.levels[d];
                    dof.num_coefs[d] = 0;
                }
            }
            std::vector<int> intercept_index;
            intercept_index.reserve(dof.levels.size());
            for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                if (!nested_flags[d]) {
                    intercept_index.push_back(static_cast<int>(d));
                }
            }
            if (!intercept_index.empty()) {
                bool skip_next = (nested_total == 0);
                for (const int idx : intercept_index) {
                    if (skip_next) {
                        skip_next = false;
                        continue;
                    }
                    if (dof.redundant[static_cast<std::size_t>(idx)] < 1) {
                        dof.redundant[static_cast<std::size_t>(idx)] = 1;
                    }
                }
            }
            for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                dof.num_coefs[d] =
                    std::max(0, dof.levels[d] - dof.redundant[d]);
            }
            dof.inexact.assign(dof.levels.size(), 0);
            if (options_.dof_method == DofAdjustmentMethod::Pairwise ||
                options_.dof_method == DofAdjustmentMethod::All ||
                options_.dof_method == DofAdjustmentMethod::FirstPair) {
                int intercept_count = 0;
                for (std::size_t d = 0; d < dof.levels.size(); ++d) {
                    if (nested_flags[d]) {
                        continue;
                    }
                    ++intercept_count;
                    if (intercept_count > 2) {
                        dof.inexact[d] = 1;
                    }
                }
            }
        } else {
            dof.inexact.assign(dof.levels.size(), 0);
        }
        dof.df_a_levels = 0;
        dof.df_a_exact = 0;
        for (std::size_t d = 0; d < dof.levels.size(); ++d) {
            dof.df_a_levels += dof.levels[d];
            dof.df_a_exact += dof.num_coefs[d];
        }
        dof.df_a = dof.df_a_exact;
        df_a_levels += dof.df_a_levels;
        df_a_exact += dof.df_a_exact;
        if (groupvar_ptr) {
            results_.groupvar = std::move(groupvar);
        }
        if (options_.dof_adjust_clusters && tuned.se_type == StandardErrorType::Cluster && c_ptr &&
            !c_ptr->empty()) {
            for (std::size_t d = 0; d < nested_flags.size(); ++d) {
                if (nested_flags[d]) {
                    nested_dof_coefs += dof.num_coefs[d];
                    nested_dof_levels += dof.levels[d];
                }
            }
        }
        results_.fe_num_levels = dof.levels;
        results_.fe_base_levels = dof.levels;
        results_.fe_redundant = dof.redundant;
        results_.fe_num_coefs = dof.num_coefs;
        results_.fe_inexact = dof.inexact;
        results_.fe_nested.assign(dof.levels.size(), 0);
        for (std::size_t d = 0; d < nested_flags.size(); ++d) {
            if (nested_flags[d]) {
                results_.fe_nested[d] = 1;
            }
        }
    } else {
        // No standard FEs: apply_common_postprocessing seeded fe_num_levels
        // with absorption.fe_levels (which includes the individual level count).
        // Reset so we can rebuild with just the individual row below.
        results_.fe_num_levels.clear();
        results_.fe_base_levels.clear();
        results_.fe_redundant.clear();
        results_.fe_num_coefs.clear();
        results_.fe_inexact.clear();
        results_.fe_nested.clear();
    }
    // Append the individual FE row so the Stata ado can display it in the
    // "Absorbed degrees of freedom" footnote, matching reghdfe's layout.
    // The individual FE is placed last in absorb_ordered on the ado side.
    results_.fe_num_levels.push_back(gi_work.num_individuals);
    results_.fe_base_levels.push_back(gi_work.num_individuals);
    results_.fe_redundant.push_back(0);
    results_.fe_num_coefs.push_back(gi_work.num_individuals);
    // Mark as inexact ("?" in the footnote): redundancy is approximate for
    // the individual FE in group/individual mode.
    results_.fe_inexact.push_back(1);
    results_.fe_nested.push_back(0);

    results_.df_a_levels = static_cast<double>(df_a_levels);
    results_.df_a_exact = static_cast<double>(df_a_exact);

    const int df_a_report = df_a_exact;
    const int nested_report = nested_dof_levels;
    results_.df_a = static_cast<double>(df_a_report);
    results_.df_a_nested = static_cast<double>(nested_report);

    const bool use_reghdfe_stats = (options_.stats_style == StatsStyle::Reghdfe);
    int df_a_used = df_a_report;
    int nested_dof_used = nested_dof_levels;
    int nested_adj = (nested_dof_levels > 0) ? 1 : 0;
    if (!use_reghdfe_stats) {
        df_a_used = 0;
        nested_dof_used = 0;
        const int df_a_base = tuned.ssc_k_exact ? df_a_exact : df_a_levels;
        const int nested_base = tuned.ssc_k_exact ? nested_dof_coefs : nested_dof_levels;
        nested_adj = (nested_base > 0) ? 1 : 0;
        if (tuned.ssc_k_fixef == FixefDofMethod::None) {
            df_a_used = 0;
            nested_dof_used = 0;
        } else {
            if (tuned.ssc_k_fixef == FixefDofMethod::Nonnested &&
                tuned.se_type == StandardErrorType::Cluster) {
                df_a_used = std::max(0, df_a_base - nested_base);
                nested_dof_used = 0;
            } else {
                df_a_used = df_a_base;
                nested_dof_used = nested_base;
            }
        }
    }

    // Use absorbed DoF (after redundancy) for RMSE/Adj R2 residual df.
    // Preserve the raw (possibly non-positive) df_r for reghdfe-compatible
    // reporting: reghdfe surfaces a negative e(df_r) and missing inference
    // when the model is saturated or over-specified.
    const int df_r_raw = nobs - df_m - df_a_report;
    const int df_r_unadj = std::max(0, df_r_raw);
    results_.df_resid_unadj = static_cast<double>(df_r_raw);
    if (df_r_unadj > 0) {
        const double sigma2_old = results_.sigma2;
        const double sigma2_new = results_.rss / static_cast<double>(df_r_unadj);
        results_.sigma2 = sigma2_new;
        if (tuned.se_type == StandardErrorType::Homoskedastic && sigma2_old > 0.0) {
            const double ratio = sigma2_new / sigma2_old;
            results_.covariance *= ratio;
            results_.std_errors *= std::sqrt(ratio);
        }
    } else {
        results_.sigma2 = 0.0;
    }
    const bool saturated_model = (df_r_raw <= 0);

    int df_r_model = std::max(0, nobs - df_m - df_a_used);

    if (c_ptr && !c_ptr->empty()) {
        results_.num_clusters = ols_result.num_clusters;
        if (c_ptr->size() == 1) {
            results_.cluster_counts.assign(1, results_.num_clusters);
            results_.cluster_combo_counts.assign(1, results_.num_clusters);
        } else {
            results_.cluster_counts = compute_cluster_counts(*c_ptr);
            results_.cluster_combo_counts = compute_cluster_combo_counts(*c_ptr);
        }
    } else {
        results_.cluster_counts.clear();
        results_.cluster_combo_counts.clear();
        results_.num_clusters = 0;
    }

    double robust_scale = 1.0;
    double cluster_ratio = 1.0;
    if (tuned.se_type == StandardErrorType::Robust && df_r_model > 0) {
        if (tuned.ssc_k_adj) {
            robust_scale = static_cast<double>(nobs) / static_cast<double>(df_r_model);
            const double scale = std::sqrt(robust_scale);
            results_.std_errors *= scale;
            results_.covariance *= robust_scale;
        }
        results_.df_resid = tuned.ssc_t_df > 0.0 ? tuned.ssc_t_df : static_cast<double>(df_r_model);
    } else if (tuned.se_type == StandardErrorType::Cluster && df_m > 0 && c_ptr && !c_ptr->empty()) {
        const int Gc = ols_result.num_clusters;
        if (use_reghdfe_stats) {
            const int df_r_unclust = df_r_model;
            const int df_r_cluster = std::max(0, Gc - 1);
            results_.df_resid = static_cast<double>(std::min(df_r_unclust, df_r_cluster));
            double ratio = 1.0;
            if (tuned.ssc_k_adj) {
                const int denom = std::max(0, nobs - df_m - df_a_report - nested_adj);
                if (denom > 0) {
                    ratio *= ols_result.df_resid / static_cast<double>(denom);
                }
            } else {
                const double denom = std::max(1.0, static_cast<double>(nobs) - 1.0);
                ratio *= ols_result.df_resid / denom;
            }
            if (c_ptr->size() == 1 && !tuned.ssc_g_adj && Gc > 1) {
                ratio *= static_cast<double>(Gc - 1) / static_cast<double>(Gc);
            }
            if (ratio > 0.0 && std::isfinite(ratio)) {
                const double scale = std::sqrt(ratio);
                results_.std_errors *= scale;
                results_.covariance *= ratio;
                cluster_ratio = ratio;
            }
            results_.cluster_scale = ratio;
            if (tuned.ssc_t_df > 0.0) {
                results_.df_resid = tuned.ssc_t_df;
            }
        } else {
            results_.df_resid = static_cast<double>(std::max(0, Gc - 1));
            double ratio = 1.0;
            if (tuned.ssc_k_adj) {
                const int df_denom = std::max(0, df_r_model + nested_dof_used - nested_adj);
                if (df_denom > 0) {
                    ratio *= ols_result.df_resid / static_cast<double>(df_denom);
                }
            } else {
                const double denom = std::max(1.0, static_cast<double>(nobs) - 1.0);
                ratio *= ols_result.df_resid / denom;
            }
            if (c_ptr->size() == 1 && !tuned.ssc_g_adj && Gc > 1) {
                ratio *= static_cast<double>(Gc - 1) / static_cast<double>(Gc);
            }
            if (ratio > 0.0 && std::isfinite(ratio)) {
                const double scale = std::sqrt(ratio);
                results_.std_errors *= scale;
                results_.covariance *= ratio;
                cluster_ratio = ratio;
            }
            results_.cluster_scale = ratio;
            if (tuned.ssc_t_df > 0.0) {
                results_.df_resid = tuned.ssc_t_df;
            }
        }
    } else if (df_r_model > 0) {
        results_.df_resid = tuned.ssc_t_df > 0.0 ? tuned.ssc_t_df : static_cast<double>(df_r_model);
    }
    if (saturated_model) {
        // Report the raw (possibly negative) residual df_r, matching reghdfe.
        results_.df_resid = static_cast<double>(df_r_raw);
    }

    bool intercept_cov_updated = false;
    if (drop_intercept && options_.fit_intercept && !saturated_model) {
        intercept_cov_updated = update_intercept_covariance(
            results_, X_work, transformed_X_used, ols_result.residuals, w_ptr,
            kept_slope_cols, ols_result, tuned.se_type, c_ptr, tuned.ssc_g_df,
            tuned.ssc_g_adj, results_.sigma2, robust_scale, cluster_ratio,
                tuned.weights_are_frequencies);

        // Multiway clustering: rebuild from the extended bread and means-
        // restored scores (see the standard path; reghdfe semantics).
        if (tuned.se_type == StandardErrorType::Cluster && c_ptr &&
            c_ptr->size() > 1) {
            const int cov_cols = static_cast<int>(results_.covariance.cols());
            const int slope_cols_aug = cov_cols - 1;
            if (slope_cols_aug > 0 &&
                slope_cols_aug == static_cast<int>(transformed_X_used.cols()) &&
                slope_cols_aug == static_cast<int>(ols_result.xtx_inv.cols()) &&
                slope_cols_aug == static_cast<int>(kept_slope_cols.size())) {
                const int n_rows = static_cast<int>(transformed_X_used.rows());
                Eigen::RowVectorXd means_x(slope_cols_aug);
                for (int j = 0; j < slope_cols_aug; ++j) {
                    means_x(j) = weighted_mean(
                        X_work.col(kept_slope_cols[static_cast<std::size_t>(j)]),
                        w_ptr);
                }
                Eigen::MatrixXd scores(n_rows, cov_cols);
                scores.leftCols(slope_cols_aug) =
                    transformed_X_used.rowwise() + means_x;
                scores.col(slope_cols_aug).setOnes();
                const double n_bread =
                    (tuned.weights_are_frequencies && w_ptr)
                        ? w_ptr->sum()
                        : static_cast<double>(n_rows);
                Eigen::MatrixXd bread(cov_cols, cov_cols);
                bread.topLeftCorner(slope_cols_aug, slope_cols_aug) =
                    ols_result.xtx_inv;
                Eigen::RowVectorXd side = -means_x * ols_result.xtx_inv;
                bread.block(slope_cols_aug, 0, 1, slope_cols_aug) = side;
                bread.block(0, slope_cols_aug, slope_cols_aug, 1) =
                    side.transpose();
                bread(slope_cols_aug, slope_cols_aug) =
                    1.0 / n_bread +
                    (means_x * ols_result.xtx_inv * means_x.transpose())(0, 0);
                Eigen::MatrixXd cov_aug = detail::multiway_cluster_sandwich(
                    bread, scores, ols_result.residuals, w_ptr, *c_ptr);
                double num = 0.0;
                double den = 0.0;
                for (int j = 0; j < slope_cols_aug; ++j) {
                    num += results_.covariance(j, j);
                    den += cov_aug(j, j);
                }
                const double rescale = (den > 0.0) ? (num / den) : 0.0;
                if (rescale > 0.0 && std::isfinite(rescale) && cov_aug.allFinite()) {
                    results_.covariance = cov_aug * rescale;
                    results_.std_errors =
                        results_.covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
                    intercept_cov_updated = true;
                }
            }
        }
    }

    if (tuned.se_type == StandardErrorType::Cluster && c_ptr && c_ptr->size() > 1) {
        const int cov_cols = static_cast<int>(results_.covariance.cols());
        const int slope_cols =
            (options_.fit_intercept && cov_cols > 0) ? (cov_cols - 1) : cov_cols;

        // Same Cameron-Gelbach-Miller PSD handling as the standard path (see
        // reghdfe_fix_psd): full-matrix clamp plus slope-block override, in
        // reghdfe's standardized metric.
        Eigen::VectorXd psd_scale = Eigen::VectorXd::Ones(std::max(cov_cols, 0));
        if (cov_cols > 0 &&
            slope_cols <= static_cast<int>(kept_slope_cols.size())) {
            const double n_sd = static_cast<double>(X_work.rows());
            for (int j = 0; j < slope_cols; ++j) {
                const auto col = X_work.col(kept_slope_cols[static_cast<std::size_t>(j)]);
                const double s = col.sum();
                const double ss = col.squaredNorm();
                double sd = 0.0;
                if (n_sd > 1.0) {
                    sd = std::sqrt(std::max(0.0, (ss - s * s / n_sd) / (n_sd - 1.0)));
                }
                psd_scale(j) = std::max(sd, 1.0e-3);
            }
        }
        auto fix_psd_scaled = [&](Eigen::MatrixXd& V,
                                  const Eigen::Ref<const Eigen::VectorXd>& d) -> bool {
            Eigen::MatrixXd Vs = d.asDiagonal() * V * d.asDiagonal();
            if (!fix_psd(Vs)) {
                return false;
            }
            V = d.cwiseInverse().asDiagonal() * Vs * d.cwiseInverse().asDiagonal();
            return true;
        };

        const bool full_psd_ready = options_.fit_intercept && cov_cols > 0 &&
                                    intercept_cov_updated &&
                                    results_.covariance.allFinite();
        if (full_psd_ready) {
            Eigen::MatrixXd cov_full = results_.covariance;
            Eigen::MatrixXd cov_block =
                cov_full.topLeftCorner(slope_cols, slope_cols);
            if (fix_psd_scaled(cov_full, psd_scale)) {
                if (slope_cols > 0) {
                    (void)fix_psd_scaled(cov_block, psd_scale.head(slope_cols));
                    cov_full.topLeftCorner(slope_cols, slope_cols) = cov_block;
                }
                results_.covariance = cov_full;
                results_.std_errors = cov_full.diagonal().cwiseMax(0.0).cwiseSqrt();
                results_.vcv_psd_fixed = true;
            }
        } else if (slope_cols > 0) {
            Eigen::MatrixXd cov_block =
                results_.covariance.topLeftCorner(slope_cols, slope_cols);
            if (fix_psd_scaled(cov_block, psd_scale.head(slope_cols))) {
                results_.covariance.topLeftCorner(slope_cols, slope_cols) = cov_block;
                results_.std_errors.head(slope_cols) =
                    cov_block.diagonal().cwiseMax(0.0).cwiseSqrt();
                results_.vcv_psd_fixed = true;
            }
        }
    }

    recompute_inference(results_.coefficients, results_.std_errors, results_.tvalues,
                        results_.pvalues, results_.conf_int, tuned.level, results_.df_resid);
    normalize_nonfrequency_weighted_fit_stats(
        results_, tuned.weights_are_frequencies, w_ptr, sum_weights_for_stats);
    if (saturated_model) {
        mark_invalid_inference_for_saturated(results_);
    }

#ifdef HDFE_USE_OPENMP
    Eigen::setNbThreads(1);
#endif
}

GroupIndividualFeEstimates HdfeRegressorV11::extract_group_individual_fes(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXi& group_ids,
    const Eigen::VectorXi& individual_ids,
    GroupAggregation aggregation,
    const Eigen::VectorXd* weights,
    const GroupIndividualFeOptions& options) const {
    if (results_.coefficients.size() == 0) {
        throw std::runtime_error("extract_group_individual_fes requires calling fit() first");
    }
    if (y.size() == 0) {
        throw std::runtime_error("Outcome vector must be non-empty");
    }
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (group_ids.size() != n) {
        throw std::runtime_error("group_ids must have the same length as y");
    }
    if (individual_ids.size() != n) {
        throw std::runtime_error("individual_ids must have the same length as y");
    }
    if (weights && weights->size() != n) {
        throw std::runtime_error("weights must have the same length as y");
    }
    if (fes.empty()) {
        throw std::runtime_error("fes must include individual_ids when extracting group/individual effects");
    }

    const int individual_fe_index = find_identical_fe(individual_ids, fes);
    if (individual_fe_index < 0) {
        throw std::runtime_error("individual_ids must also be included in fes");
    }

    GroupCollapsedData collapsed =
        collapse_group_long_format(y, X, fes, group_ids, &individual_ids, individual_fe_index,
                                   aggregation, weights, nullptr);

    if (options_.drop_singletons) {
        const GroupSingletonDropResult dropped =
            drop_singletons_group_individual(collapsed.standard_fes, collapsed.gi);
        if (dropped.dropped_groups > 0) {
            collapsed = filter_group_collapsed(collapsed, dropped.keep_groups,
                                               dropped.individual_degrees, aggregation);
        }
    }

    const Eigen::VectorXd* w_ptr = collapsed.weights ? &(*collapsed.weights) : nullptr;

    const int coef_size = static_cast<int>(results_.coefficients.size());
    const int x_cols = static_cast<int>(collapsed.X.cols());
    const int slope_cols = options_.fit_intercept ? (coef_size - 1) : coef_size;
    if (slope_cols != x_cols) {
        throw std::runtime_error("Coefficient dimension mismatch: X columns differ from last fit()");
    }

    Eigen::VectorXd linear = Eigen::VectorXd::Zero(collapsed.y.size());
    if (slope_cols > 0) {
        linear.noalias() = collapsed.X * results_.coefficients.head(slope_cols);
    }
    // Match reghdfe's predict, d convention: remove only the slope component (exclude _cons),
    // then project onto the FE space.
    const Eigen::VectorXd y_minus_xb = collapsed.y - linear;
    // Stata's predict, d returns the fitted fixed-effect component (excluding the regression residual).
    // Compute it as (y - Xb) - M(y - Xb), where M is the within transformation that partials out
    // the group/individual and standard fixed effects.
    HdfeOptions tuned = options_;
    tuned.num_threads = 1;
    tuned.max_iter = std::max(tuned.max_iter, 20000);
    tuned.tol = std::min(tuned.tol, 1e-15);
    Eigen::MatrixXd empty_X(static_cast<int>(y_minus_xb.size()), 0);
    const detail::AbsorptionResult absorbed =
        detail::absorb_fixed_effects_group_individual(y_minus_xb, empty_X, collapsed.standard_fes,
                                                      collapsed.gi, w_ptr, tuned, method_used_);
    const Eigen::VectorXd d = y_minus_xb - absorbed.y_tilde;

    const int G = collapsed.gi.num_groups;
    const int I = collapsed.gi.num_individuals;
    if (G != d.size()) {
        throw std::runtime_error("Internal error: group count mismatch");
    }
    if (I <= 0) {
        throw std::runtime_error("Group/individual structure must have at least one individual");
    }

    const int K = static_cast<int>(collapsed.standard_fes.size());
    if (K > 3) {
        throw std::runtime_error("extract_group_individual_fes supports up to 3 standard fixed effects (Stata extractfes limit)");
    }

    std::vector<FeIndexMap> maps;
    maps.reserve(static_cast<std::size_t>(K));
    std::vector<Eigen::VectorXd> fe_level_values;
    std::vector<Eigen::VectorXd> fe_group_values;
    fe_level_values.reserve(static_cast<std::size_t>(K));
    fe_group_values.reserve(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
        maps.push_back(build_fe_index_map(collapsed.standard_fes[static_cast<std::size_t>(k)]));
        fe_level_values.emplace_back(Eigen::VectorXd::Zero(maps.back().raw_values.size()));
        fe_group_values.emplace_back(Eigen::VectorXd::Zero(G));
    }

    Eigen::VectorXd a = Eigen::VectorXd::Zero(I);
    Eigen::VectorXd a_prev1 = Eigen::VectorXd::Zero(I);
    Eigen::VectorXd a_prev2 = Eigen::VectorXd::Zero(I);
    Eigen::VectorXd a_long(G);

    const Eigen::VectorXd precond = gi_preconditioner(collapsed.gi, w_ptr);

    double tols = options.tol_start;
    double mse = options.tol_main + 1.0;
    int iter = 0;
    bool converged = false;
    double constant = 0.0;

    while (mse > options.tol_main) {
        gi_group_sum(collapsed.gi, a, a_long);

        if (K == 0) {
            Eigen::VectorXd resid = d - a_long;
            constant = weighted_mean(resid, w_ptr);
        } else {
            for (int k = 0; k < K; ++k) {
                Eigen::VectorXd resid = d - a_long;
                for (int j = 0; j < K; ++j) {
                    if (j == k) {
                        continue;
                    }
                    resid.noalias() -= fe_group_values[static_cast<std::size_t>(j)];
                }
                update_standard_fe(resid,
                                   maps[static_cast<std::size_t>(k)],
                                   w_ptr,
                                   fe_level_values[static_cast<std::size_t>(k)],
                                   fe_group_values[static_cast<std::size_t>(k)]);
            }
        }

        Eigen::VectorXd dd = d;
        if (K == 0) {
            dd.array() -= constant;
        } else {
            for (int k = 0; k < K; ++k) {
                dd.noalias() -= fe_group_values[static_cast<std::size_t>(k)];
            }
        }

        a_prev2 = a_prev1;
        a_prev1 = a;
        a = solve_a1(std::move(a), collapsed.gi, dd, w_ptr, precond,
                     options.max_iter_solver, tols, options.verbose);

        if (mse < options.factor * tols) {
            tols = std::max(tols / 10.0, options.tol_final);
        }

        if (iter > options.start_accel && options.accel > 0 && options.every_accel > 0 &&
            (iter % options.every_accel) == 0) {
            a = accel_a1(a, a_prev1, a_prev2, options.accel, options.a1p1, options.a2p1,
                         options.a2p2);
        }

        gi_group_sum(collapsed.gi, a, a_long);

        Eigen::VectorXd resid = d - a_long;
        if (K == 0) {
            resid.array() -= constant;
        } else {
            for (int k = 0; k < K; ++k) {
                resid.noalias() -= fe_group_values[static_cast<std::size_t>(k)];
            }
        }
        mse = weighted_mse(resid, w_ptr);
        ++iter;
        if (options.verbose > 0) {
            std::cout << "iter: " << iter << " mse=" << mse << "\n";
        }
        if (iter > options.max_iter_main) {
            break;
        }
    }
    converged = (mse <= options.tol_main);

    GroupIndividualFeEstimates out;
    out.individual_ids = collapsed.individual_values;
    out.individual_effects = std::move(a);
    out.fe_level_ids.reserve(static_cast<std::size_t>(K));
    out.fe_level_effects.reserve(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
        out.fe_level_ids.push_back(maps[static_cast<std::size_t>(k)].raw_values);
        out.fe_level_effects.push_back(fe_level_values[static_cast<std::size_t>(k)]);
    }
    out.iterations = iter;
    out.converged = converged;
    out.mse = mse;
    return out;
}

}  // namespace v11
}  // namespace hdfe
