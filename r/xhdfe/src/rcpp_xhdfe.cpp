// Rcpp bindings for the hdfe C++ library (v11).
//
// This file mirrors python/py_hdfe_v11.cpp: the same option strings (with the
// same aliases and case-insensitive parsing), the same fit() input contract,
// and the same result surface, so that the R, Python and Stata packages sit
// on the identical compiled estimator. Additions relative to the Python
// binding (all present in the Stata command): fe_tolerance,
// fe_recovery_method, stats_style, weights_are_frequencies, omitted_reason
// and the effective (weighted) observation counts.
//
// Index conventions at this layer are C++ conventions: fe_index inside
// slope terms and endogenous_idx are 0-based (the R wrapper translates from
// 1-based R indices). sample_index in the result is returned 0-based and
// shifted in the R wrapper.

#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "hdfe/akm_kss.hpp"
#include "hdfe/hdfe_regressor_v11.hpp"

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

hdfe::StandardErrorType parse_se_type(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "homoskedastic" || lower == "classical" || lower == "unadjusted" ||
        lower == "unadj" || lower == "ols" || lower == "iid") {
        return hdfe::StandardErrorType::Homoskedastic;
    }
    if (lower == "robust" || lower == "hc1" || lower == "heteroskedastic") {
        return hdfe::StandardErrorType::Robust;
    }
    if (lower == "cluster" || lower == "clustered") {
        return hdfe::StandardErrorType::Cluster;
    }
    throw std::runtime_error("Unknown standard error type: " + name);
}

hdfe::AbsorptionMethod parse_absorption_method(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "auto") return hdfe::AbsorptionMethod::Auto;
    if (lower == "gs" || lower == "gauss-seidel" || lower == "gauss_seidel") {
        return hdfe::AbsorptionMethod::GaussSeidel;
    }
    if (lower == "sym" || lower == "symmetric" || lower == "symgs" ||
        lower == "symmetric-gauss-seidel" || lower == "symmetric_gauss_seidel") {
        return hdfe::AbsorptionMethod::SymmetricGaussSeidel;
    }
    if (lower == "jacobi") return hdfe::AbsorptionMethod::Jacobi;
    if (lower == "schwarz" || lower == "pcg" || lower == "schwarz-pcg") {
        return hdfe::AbsorptionMethod::Schwarz;
    }
    if (lower == "lsmr" || lower == "plain-lsmr" || lower == "plain_lsmr") {
        return hdfe::AbsorptionMethod::Lsmr;
    }
    if (lower == "mlsmr" || lower == "modified-lsmr" || lower == "modified_lsmr" ||
        lower == "within" || lower == "within-additive" || lower == "within_additive") {
        return hdfe::AbsorptionMethod::Mlsmr;
    }
    if (lower == "auto-mlsmr" || lower == "auto_mlsmr" || lower == "mlsmr-auto" ||
        lower == "mlsmr_auto") {
        return hdfe::AbsorptionMethod::AutoMlsmr;
    }
    throw std::runtime_error("Unknown absorption method: " + name);
}

hdfe::ToleranceMode parse_tolerance_mode(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "xhdfe-fast" || lower == "xhdfe_fast" || lower == "xhdfe" ||
        lower == "fast" || lower == "current") {
        return hdfe::ToleranceMode::XhdfeFast;
    }
    if (lower == "reghdfe-comparable" || lower == "reghdfe_comparable" ||
        lower == "reghdfe" || lower == "comparable") {
        return hdfe::ToleranceMode::ReghdfeComparable;
    }
    if (lower == "strict-residual" || lower == "strict_residual" ||
        lower == "residual-certificate" || lower == "residual_certificate" ||
        lower == "strict") {
        return hdfe::ToleranceMode::StrictResidual;
    }
    throw std::runtime_error("Unknown tolerance mode: " + name);
}

hdfe::ConvergenceCriterion parse_convergence_criterion(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower.empty() || lower == "auto") return hdfe::ConvergenceCriterion::Auto;
    if (lower == "normchange" || lower == "norm_change" || lower == "norm-change" ||
        lower == "norm") {
        return hdfe::ConvergenceCriterion::NormChange;
    }
    if (lower == "reghdfe" || lower == "update" || lower == "reldif") {
        return hdfe::ConvergenceCriterion::Reghdfe;
    }
    if (lower == "both") return hdfe::ConvergenceCriterion::Both;
    throw std::runtime_error("Unknown convergence criterion: " + name);
}

hdfe::FixefDofMethod parse_fixef_method(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "full" || lower == "true" || lower == "yes") {
        return hdfe::FixefDofMethod::Full;
    }
    if (lower == "none" || lower == "false" || lower == "no") {
        return hdfe::FixefDofMethod::None;
    }
    if (lower == "nonnested" || lower == "non-nested" || lower == "non_nested") {
        return hdfe::FixefDofMethod::Nonnested;
    }
    throw std::runtime_error("Unknown K.fixef: " + name);
}

hdfe::ClusterDofMethod parse_cluster_df(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "min") return hdfe::ClusterDofMethod::Min;
    if (lower == "conventional" || lower == "conv") return hdfe::ClusterDofMethod::Conventional;
    throw std::runtime_error("Unknown G.df: " + name);
}

hdfe::FeRecoveryMethod parse_fe_recovery_method(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "hybrid") return hdfe::FeRecoveryMethod::Hybrid;
    if (lower == "map") return hdfe::FeRecoveryMethod::Map;
    throw std::runtime_error("Unknown FE recovery method: " + name);
}

hdfe::StatsStyle parse_stats_style(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "reghdfe") return hdfe::StatsStyle::Reghdfe;
    if (lower == "legacy") return hdfe::StatsStyle::Legacy;
    throw std::runtime_error("Unknown stats style: " + name);
}

hdfe::v11::GroupAggregation parse_group_aggregation(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "mean" || lower == "average" || lower == "avg") {
        return hdfe::v11::GroupAggregation::Mean;
    }
    if (lower == "sum") return hdfe::v11::GroupAggregation::Sum;
    throw std::runtime_error("Unknown aggregation: " + name);
}

void apply_dofadjustments(hdfe::HdfeOptions& opts, const std::vector<std::string>& tokens) {
    opts.dof_method = hdfe::DofAdjustmentMethod::All;
    opts.dof_adjust_clusters = false;
    opts.dof_adjust_continuous = false;
    for (const std::string& raw : tokens) {
        const std::string token = to_lower(raw);
        if (token.empty()) continue;
        if (token == "all") {
            opts.dof_method = hdfe::DofAdjustmentMethod::All;
            opts.dof_adjust_clusters = true;
            opts.dof_adjust_continuous = true;
        } else if (token == "none") {
            opts.dof_method = hdfe::DofAdjustmentMethod::None;
        } else if (token == "firstpair" || token == "first") {
            opts.dof_method = hdfe::DofAdjustmentMethod::FirstPair;
        } else if (token == "pairwise" || token == "pair") {
            opts.dof_method = hdfe::DofAdjustmentMethod::Pairwise;
        } else if (token == "clusters" || token == "cluster") {
            opts.dof_adjust_clusters = true;
        } else if (token == "continuous" || token == "cont") {
            opts.dof_adjust_continuous = true;
        } else {
            throw std::runtime_error("Unknown dofadjustments token: " + raw);
        }
    }
}

const char* method_name(hdfe::AbsorptionMethod method) {
    switch (method) {
        case hdfe::AbsorptionMethod::Auto: return "auto";
        case hdfe::AbsorptionMethod::GaussSeidel: return "gauss-seidel";
        case hdfe::AbsorptionMethod::SymmetricGaussSeidel: return "symmetric-gauss-seidel";
        case hdfe::AbsorptionMethod::Jacobi: return "jacobi";
        case hdfe::AbsorptionMethod::Schwarz: return "schwarz";
        case hdfe::AbsorptionMethod::Lsmr: return "lsmr";
        case hdfe::AbsorptionMethod::Mlsmr: return "mlsmr";
        case hdfe::AbsorptionMethod::AutoMlsmr: return "auto-mlsmr";
    }
    return "unknown";
}

int method_code(hdfe::AbsorptionMethod method) {
    // Same code table as Stata e(absorption_method_used).
    switch (method) {
        case hdfe::AbsorptionMethod::Auto: return 0;
        case hdfe::AbsorptionMethod::GaussSeidel: return 1;
        case hdfe::AbsorptionMethod::SymmetricGaussSeidel: return 2;
        case hdfe::AbsorptionMethod::Jacobi: return 3;
        case hdfe::AbsorptionMethod::Schwarz: return 4;
        case hdfe::AbsorptionMethod::Lsmr: return 5;
        case hdfe::AbsorptionMethod::Mlsmr: return 6;
        case hdfe::AbsorptionMethod::AutoMlsmr: return 7;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// R -> Eigen conversions
// ---------------------------------------------------------------------------

Eigen::VectorXd as_vector_xd(const Rcpp::NumericVector& v) {
    Eigen::VectorXd out(v.size());
    if (v.size() > 0) std::memcpy(out.data(), &v[0], sizeof(double) * v.size());
    return out;
}

Eigen::VectorXi as_vector_xi(const Rcpp::IntegerVector& v, const char* label) {
    Eigen::VectorXi out(v.size());
    for (R_xlen_t i = 0; i < v.size(); ++i) {
        if (v[i] == NA_INTEGER) {
            throw std::runtime_error(std::string(label) + " contains missing values");
        }
        out[static_cast<int>(i)] = v[i];
    }
    return out;
}

Eigen::MatrixXd as_matrix_xd(const Rcpp::NumericMatrix& m) {
    Eigen::MatrixXd out(m.nrow(), m.ncol());
    if (m.size() > 0) std::memcpy(out.data(), &m[0], sizeof(double) * m.size());
    return out;
}

std::vector<Eigen::VectorXi> parse_id_list(const Rcpp::List& ids, R_xlen_t nobs,
                                           const char* label) {
    std::vector<Eigen::VectorXi> out;
    out.reserve(ids.size());
    for (R_xlen_t i = 0; i < ids.size(); ++i) {
        Rcpp::IntegerVector v(ids[i]);
        if (v.size() != nobs) {
            throw std::runtime_error(std::string(label) + " length must match nobs");
        }
        out.push_back(as_vector_xi(v, label));
    }
    return out;
}

std::vector<hdfe::detail::HeterogeneousSlopeTerm> parse_slopes(const Rcpp::List& slopes,
                                                               R_xlen_t nobs,
                                                               std::size_t num_fes) {
    std::vector<hdfe::detail::HeterogeneousSlopeTerm> out;
    out.reserve(slopes.size());
    for (R_xlen_t i = 0; i < slopes.size(); ++i) {
        Rcpp::List item(slopes[i]);
        hdfe::detail::HeterogeneousSlopeTerm term;
        term.fe_index = Rcpp::as<int>(item["fe_index"]);
        Rcpp::NumericVector values(item["values"]);
        if (values.size() != nobs) {
            throw std::runtime_error("slope values length must match nobs");
        }
        term.values = as_vector_xd(values);
        term.include_intercept = item.containsElementNamed("include_intercept")
                                     ? Rcpp::as<bool>(item["include_intercept"])
                                     : false;
        if (term.fe_index < 0 || static_cast<std::size_t>(term.fe_index) >= num_fes) {
            throw std::runtime_error("slope fe_index out of range");
        }
        out.push_back(std::move(term));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Eigen -> R conversions
// ---------------------------------------------------------------------------

Rcpp::NumericVector wrap_vector(const Eigen::VectorXd& v) {
    Rcpp::NumericVector out(v.size());
    if (v.size() > 0) std::memcpy(&out[0], v.data(), sizeof(double) * v.size());
    return out;
}

Rcpp::IntegerVector wrap_vector(const Eigen::VectorXi& v) {
    Rcpp::IntegerVector out(v.size());
    for (int i = 0; i < v.size(); ++i) out[i] = v[i];
    return out;
}

Rcpp::IntegerVector wrap_vector(const std::vector<int>& v) {
    return Rcpp::IntegerVector(v.begin(), v.end());
}

Rcpp::NumericMatrix wrap_matrix(const Eigen::MatrixXd& m) {
    Rcpp::NumericMatrix out(static_cast<int>(m.rows()), static_cast<int>(m.cols()));
    if (m.size() > 0) std::memcpy(&out[0], m.data(), sizeof(double) * m.size());
    return out;
}

Rcpp::List wrap_vector_list(const std::vector<Eigen::VectorXd>& vs) {
    Rcpp::List out(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i) out[i] = wrap_vector(vs[i]);
    return out;
}

Rcpp::List wrap_int_vector_list(const std::vector<Eigen::VectorXi>& vs) {
    Rcpp::List out(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i) out[i] = wrap_vector(vs[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

template <typename T>
bool opt_has(const Rcpp::List& opts, const char* name) {
    return opts.containsElementNamed(name) && !Rf_isNull(opts[name]);
}

void parse_options(const Rcpp::List& opts, hdfe::HdfeOptions& out,
                   hdfe::v11::ThreadingOptions& threading) {
    const auto get_bool = [&](const char* name, bool fallback) {
        return opt_has<bool>(opts, name) ? Rcpp::as<bool>(opts[name]) : fallback;
    };
    const auto get_int = [&](const char* name, int fallback) {
        return opt_has<int>(opts, name) ? Rcpp::as<int>(opts[name]) : fallback;
    };
    const auto get_double = [&](const char* name, double fallback) {
        return opt_has<double>(opts, name) ? Rcpp::as<double>(opts[name]) : fallback;
    };
    const auto get_string = [&](const char* name, const std::string& fallback) {
        return opt_has<std::string>(opts, name) ? Rcpp::as<std::string>(opts[name]) : fallback;
    };

    out.se_type = parse_se_type(get_string("se_type", "unadjusted"));
    out.tol = get_double("tol", 1e-8);
    out.max_iter = get_int("max_iter", 100000);
    out.convergence_check_interval = std::max(1, get_int("check_interval", 1));
    out.convergence_criterion =
        parse_convergence_criterion(get_string("convergence", "auto"));
    out.fit_intercept = get_bool("fit_intercept", true);
    out.num_threads = get_int("num_threads", 0);
    out.drop_singletons = get_bool("drop_singletons", true);
    if (opt_has<bool>(opts, "keepsingletons")) {
        out.drop_singletons = !Rcpp::as<bool>(opts["keepsingletons"]);
    }
    out.retain_fixed_effects = get_bool("retain_fes", false);
    out.symmetric_sweep = get_bool("symmetric_sweep", false);
    out.absorption_method =
        parse_absorption_method(get_string("absorption_method", "auto"));
    out.jacobi_relaxation = get_double("jacobi_relaxation", 0.0);
    out.tolerance_mode =
        parse_tolerance_mode(get_string("tolerance_mode", "reghdfe-comparable"));
    out.use_krylov = false;

    double level = get_double("level", 95.0);
    if (level > 0.0 && level <= 1.0) level *= 100.0;
    if (!(level > 0.0 && level < 100.0)) {
        throw std::runtime_error("level must be between 0 and 100");
    }
    out.level = level;

    if (opt_has<Rcpp::CharacterVector>(opts, "dofadjustments")) {
        std::vector<std::string> tokens;
        Rcpp::CharacterVector raw(opts["dofadjustments"]);
        for (R_xlen_t i = 0; i < raw.size(); ++i) {
            // Each element may itself contain comma/space separated tokens.
            std::string chunk = Rcpp::as<std::string>(raw[i]);
            std::string current;
            for (char c : chunk) {
                if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
                    if (!current.empty()) tokens.push_back(current);
                    current.clear();
                } else {
                    current.push_back(c);
                }
            }
            if (!current.empty()) tokens.push_back(current);
        }
        apply_dofadjustments(out, tokens);
    }

    if (opt_has<bool>(opts, "groupvar")) {
        out.save_groupvar = Rcpp::as<bool>(opts["groupvar"]);
    }

    if (opt_has<bool>(opts, "ssc_k_adj")) out.ssc_k_adj = Rcpp::as<bool>(opts["ssc_k_adj"]);
    if (opt_has<std::string>(opts, "ssc_k_fixef")) {
        out.ssc_k_fixef = parse_fixef_method(Rcpp::as<std::string>(opts["ssc_k_fixef"]));
    }
    if (opt_has<bool>(opts, "ssc_k_exact")) out.ssc_k_exact = Rcpp::as<bool>(opts["ssc_k_exact"]);
    if (opt_has<bool>(opts, "ssc_g_adj")) out.ssc_g_adj = Rcpp::as<bool>(opts["ssc_g_adj"]);
    if (opt_has<std::string>(opts, "ssc_g_df")) {
        out.ssc_g_df = parse_cluster_df(Rcpp::as<std::string>(opts["ssc_g_df"]));
    }
    if (opt_has<double>(opts, "ssc_t_df")) out.ssc_t_df = Rcpp::as<double>(opts["ssc_t_df"]);

    // Extensions beyond the Python binding (all present in the Stata command).
    out.fe_tolerance = get_double("fe_tolerance", 1e-6);
    out.fe_recovery_method =
        parse_fe_recovery_method(get_string("fe_recovery_method", "hybrid"));
    out.stats_style = parse_stats_style(get_string("stats_style", "reghdfe"));
    out.weights_are_frequencies = get_bool("weights_are_frequencies", false);

    const int num_threads = out.num_threads;
    const int default_threads = get_int("default_threads", 0);
    threading.default_threads =
        default_threads > 0 ? default_threads : (num_threads > 0 ? num_threads : 0);
    threading.max_threads = get_int("max_threads", 0);
    threading.min_parallel_rows = get_int("min_parallel_rows", 20000);
    threading.target_rows_per_thread = get_int("target_rows_per_thread", 500000);
    threading.symmetric_sweep = out.symmetric_sweep;
}

Rcpp::List build_results(const hdfe::v11::HdfeRegressorV11& reg) {
    const hdfe::HdfeResults& res = reg.results();
    Rcpp::List out;

    out["coefficients"] = wrap_vector(res.coefficients);
    out["se"] = wrap_vector(res.std_errors);
    out["tvalues"] = wrap_vector(res.tvalues);
    out["pvalues"] = wrap_vector(res.pvalues);
    out["conf_int"] = wrap_matrix(res.conf_int);
    out["covariance"] = wrap_matrix(res.covariance);
    out["residuals"] = wrap_vector(res.residuals);
    out["omitted_reason"] = wrap_vector(res.omitted_reason);

    out["nobs"] = res.nobs;
    out["nobs_full"] = res.nobs_full;
    out["num_singletons"] = res.num_singletons;
    out["nobs_effective"] = res.nobs_effective;
    out["nobs_full_effective"] = res.nobs_full_effective;
    out["num_singletons_effective"] = res.num_singletons_effective;
    out["sample_index0"] = wrap_vector(res.sample_index);

    out["df_resid"] = res.df_resid;
    out["df_resid_unadj"] = res.df_resid_unadj;
    out["df_m"] = res.df_m;
    out["df_a"] = res.df_a;
    out["df_a_levels"] = res.df_a_levels;
    out["df_a_exact"] = res.df_a_exact;
    out["df_a_nested"] = res.df_a_nested;

    out["r2"] = res.r2;
    out["r2_within"] = res.r2_within;
    out["sigma2"] = res.sigma2;
    out["rss"] = res.rss;
    out["tss"] = res.tss;
    out["tss_within"] = res.tss_within;
    out["saturated"] = res.is_saturated();

    out["fe_num_levels"] = wrap_vector(res.fe_num_levels);
    out["fe_base_levels"] = wrap_vector(res.fe_base_levels);
    out["fe_redundant"] = wrap_vector(res.fe_redundant);
    out["fe_num_coefs"] = wrap_vector(res.fe_num_coefs);
    out["fe_inexact"] = wrap_vector(res.fe_inexact);
    out["fe_nested"] = wrap_vector(res.fe_nested);

    out["num_iterations"] = res.num_iterations;
    out["converged"] = res.converged;

    out["groupvar"] = wrap_vector(res.groupvar);
    out["fe_effects"] = wrap_vector_list(res.fe_effects);
    out["fe_recovery_iterations"] = res.fe_recovery_iterations;
    out["fe_recovery_max_delta"] = res.fe_recovery_max_delta;
    out["fe_recovery_converged"] = res.fe_recovery_converged;

    out["num_clusters"] = res.num_clusters;
    out["cluster_counts"] = wrap_vector(res.cluster_counts);
    out["cluster_combo_counts"] = wrap_vector(res.cluster_combo_counts);
    out["cluster_scale"] = res.cluster_scale;
    out["vcv_psd_fixed"] = res.vcv_psd_fixed;

    out["threads_used"] = reg.threads_used();
    out["absorption_method_used"] = std::string(method_name(reg.absorption_method_used()));
    out["absorption_method_code"] = method_code(reg.absorption_method_used());
    out["gpu_used"] = reg.gpu_used();
    out["gpu_status_code"] = reg.gpu_status_code();
    out["gpu_attempted"] = reg.gpu_attempted();
    out["gpu_absorption_converged"] = reg.gpu_absorption_converged();
    out["gpu_absorption_iterations"] = reg.gpu_absorption_iterations();

    return out;
}

}  // namespace

// [[Rcpp::export(name = ".xhdfe_cpp_fit")]]
Rcpp::List xhdfe_cpp_fit(Rcpp::NumericVector y,
                         Rcpp::NumericMatrix X,
                         Rcpp::List fes,
                         SEXP weights,
                         Rcpp::List clusters,
                         SEXP instruments,
                         Rcpp::IntegerVector endogenous_idx,
                         Rcpp::List slopes,
                         SEXP group,
                         SEXP individual,
                         std::string aggregation,
                         Rcpp::List opts) {
    const R_xlen_t n = y.size();
    if (X.nrow() != n) {
        throw std::runtime_error("X must have the same number of rows as y");
    }

    hdfe::HdfeOptions options;
    hdfe::v11::ThreadingOptions threading;
    parse_options(opts, options, threading);

    // Zero-copy views over the R storage (R vectors/matrices are
    // column-major doubles, exactly Eigen's default layout).
    Eigen::Map<const Eigen::VectorXd> y_map(&y[0], n);
    Eigen::Map<const Eigen::MatrixXd> X_map(X.size() > 0 ? &X[0] : nullptr, n, X.ncol());

    std::vector<Eigen::VectorXi> fe_list = parse_id_list(fes, n, "fixed effect");

    Eigen::VectorXd weights_vec;
    const Eigen::VectorXd* weights_ptr = nullptr;
    if (!Rf_isNull(weights)) {
        Rcpp::NumericVector w(weights);
        if (w.size() != n) throw std::runtime_error("weights length must match nobs");
        weights_vec = as_vector_xd(w);
        weights_ptr = &weights_vec;
    }

    std::vector<Eigen::VectorXi> cluster_list = parse_id_list(clusters, n, "cluster");
    const std::vector<Eigen::VectorXi>* clusters_ptr =
        cluster_list.empty() ? nullptr : &cluster_list;

    Eigen::MatrixXd instruments_mat;
    const Eigen::MatrixXd* instruments_ptr = nullptr;
    if (!Rf_isNull(instruments)) {
        Rcpp::NumericMatrix z(instruments);
        if (z.nrow() != n) {
            throw std::runtime_error("instruments must have the same number of rows as y");
        }
        instruments_mat = as_matrix_xd(z);
        instruments_ptr = &instruments_mat;
    }

    std::vector<int> endo_idx(endogenous_idx.begin(), endogenous_idx.end());

    std::vector<hdfe::detail::HeterogeneousSlopeTerm> slope_terms =
        parse_slopes(slopes, n, fe_list.size());
    const std::vector<hdfe::detail::HeterogeneousSlopeTerm>* slopes_ptr =
        slope_terms.empty() ? nullptr : &slope_terms;

    const bool has_group = !Rf_isNull(group);
    if (has_group && !slope_terms.empty()) {
        throw std::runtime_error(
            "heterogeneous slopes are not supported with group()/individual() mode");
    }
    if (has_group && (instruments_ptr != nullptr || !endo_idx.empty())) {
        throw std::runtime_error(
            "IV/instruments are not supported with group()/individual() mode");
    }

    hdfe::v11::HdfeRegressorV11 reg(options, threading);

    if (has_group) {
        Rcpp::IntegerVector g(group);
        if (g.size() != n) throw std::runtime_error("group length must match nobs");
        Eigen::VectorXi group_ids = as_vector_xi(g, "group");
        Eigen::VectorXi individual_ids;
        const Eigen::VectorXi* individual_ptr = nullptr;
        if (!Rf_isNull(individual)) {
            Rcpp::IntegerVector ind(individual);
            if (ind.size() != n) throw std::runtime_error("individual length must match nobs");
            individual_ids = as_vector_xi(ind, "individual");
            individual_ptr = &individual_ids;
        }
        const hdfe::v11::GroupAggregation agg = parse_group_aggregation(aggregation);
        reg.fit_grouped(y_map, X_map, fe_list, group_ids, individual_ptr, agg, weights_ptr,
                        clusters_ptr);
    } else {
        parse_group_aggregation(aggregation);  // validate even when unused
        reg.fit(y_map, X_map, fe_list, weights_ptr, clusters_ptr, instruments_ptr, endo_idx,
                slopes_ptr);
    }

    return build_results(reg);
}

// [[Rcpp::export(name = ".xhdfe_cpp_partial_out")]]
Rcpp::List xhdfe_cpp_partial_out(Rcpp::NumericVector y,
                                 Rcpp::NumericMatrix X,
                                 Rcpp::List fes,
                                 SEXP weights,
                                 Rcpp::List slopes,
                                 Rcpp::List opts) {
    const R_xlen_t n = y.size();
    if (X.nrow() != n) {
        throw std::runtime_error("X must have the same number of rows as y");
    }

    hdfe::HdfeOptions options;
    hdfe::v11::ThreadingOptions threading;
    parse_options(opts, options, threading);

    Eigen::Map<const Eigen::VectorXd> y_map(&y[0], n);
    Eigen::Map<const Eigen::MatrixXd> X_map(X.size() > 0 ? &X[0] : nullptr, n, X.ncol());

    std::vector<Eigen::VectorXi> fe_list = parse_id_list(fes, n, "fixed effect");

    Eigen::VectorXd weights_vec;
    const Eigen::VectorXd* weights_ptr = nullptr;
    if (!Rf_isNull(weights)) {
        Rcpp::NumericVector w(weights);
        if (w.size() != n) throw std::runtime_error("weights length must match nobs");
        weights_vec = as_vector_xd(w);
        weights_ptr = &weights_vec;
    }

    std::vector<hdfe::detail::HeterogeneousSlopeTerm> slope_terms =
        parse_slopes(slopes, n, fe_list.size());
    const std::vector<hdfe::detail::HeterogeneousSlopeTerm>* slopes_ptr =
        slope_terms.empty() ? nullptr : &slope_terms;

    hdfe::v11::HdfeRegressorV11 reg(options, threading);
    hdfe::detail::AbsorptionResult partial =
        reg.partial_out(y_map, X_map, fe_list, weights_ptr, nullptr, slopes_ptr);

    Rcpp::List out;
    out["y_tilde"] = wrap_vector(partial.y_tilde);
    out["X_tilde"] = wrap_matrix(partial.X_tilde);
    // Kept-row map (0-based, identity when no singleton was dropped).
    out["sample_index0"] = wrap_vector(reg.results().sample_index);
    out["num_singletons"] = reg.results().num_singletons;
    out["fe_num_levels"] = wrap_vector(partial.fe_levels);
    out["iterations"] = partial.iterations;
    out["converged"] = partial.converged;
    out["schwarz_used"] = partial.schwarz_used;
    out["mlsmr_used"] = partial.mlsmr_used;
    out["gpu_used"] = partial.gpu_used;
    out["gpu_status_code"] = partial.gpu_status_code;
    out["gpu_attempted"] = partial.gpu_attempted;
    out["threads_used"] = reg.threads_used();
    out["absorption_method_used"] = std::string(method_name(reg.absorption_method_used()));
    return out;
}

// [[Rcpp::export(name = ".xhdfe_cpp_extract_group_fes")]]
Rcpp::List xhdfe_cpp_extract_group_fes(Rcpp::NumericVector y,
                                       Rcpp::NumericMatrix X,
                                       Rcpp::List fes,
                                       Rcpp::IntegerVector group,
                                       Rcpp::IntegerVector individual,
                                       SEXP weights,
                                       std::string aggregation,
                                       Rcpp::List gi_opts,
                                       Rcpp::List opts) {
    const R_xlen_t n = y.size();
    if (X.nrow() != n) {
        throw std::runtime_error("X must have the same number of rows as y");
    }
    if (group.size() != n) throw std::runtime_error("group length must match nobs");
    if (individual.size() != n) throw std::runtime_error("individual length must match nobs");

    hdfe::HdfeOptions options;
    hdfe::v11::ThreadingOptions threading;
    parse_options(opts, options, threading);

    Eigen::Map<const Eigen::VectorXd> y_map(&y[0], n);
    Eigen::Map<const Eigen::MatrixXd> X_map(X.size() > 0 ? &X[0] : nullptr, n, X.ncol());

    std::vector<Eigen::VectorXi> fe_list = parse_id_list(fes, n, "fixed effect");
    if (fe_list.empty()) {
        throw std::runtime_error("fes is required for extract_group_individual_fes");
    }

    Eigen::VectorXi group_ids = as_vector_xi(group, "group");
    Eigen::VectorXi individual_ids = as_vector_xi(individual, "individual");

    Eigen::VectorXd weights_vec;
    const Eigen::VectorXd* weights_ptr = nullptr;
    if (!Rf_isNull(weights)) {
        Rcpp::NumericVector w(weights);
        if (w.size() != n) throw std::runtime_error("weights length must match nobs");
        weights_vec = as_vector_xd(w);
        weights_ptr = &weights_vec;
    }

    hdfe::v11::GroupIndividualFeOptions gi;
    const auto gi_double = [&](const char* name, double fallback) {
        return opt_has<double>(gi_opts, name) ? Rcpp::as<double>(gi_opts[name]) : fallback;
    };
    const auto gi_int = [&](const char* name, int fallback) {
        return opt_has<int>(gi_opts, name) ? Rcpp::as<int>(gi_opts[name]) : fallback;
    };
    gi.tol_main = gi_double("tol_main", 1e-9);
    gi.tol_start = gi_double("tol_start", 1e-3);
    gi.tol_final = gi_double("tol_final", 1e-9);
    gi.max_iter_main = gi_int("max_iter_main", 100000);
    gi.max_iter_solver = gi_int("max_iter_solver", 1000);
    gi.verbose = gi_int("verbose", 0);
    gi.accel = gi_int("accel", 2);
    gi.start_accel = gi_int("start_accel", 5);
    gi.every_accel = gi_int("every_accel", 5);
    gi.factor = gi_double("factor", 1.0);
    gi.a1p1 = gi_double("a1p1", 0.75);
    gi.a2p1 = gi_double("a2p1", 1e-8);
    gi.a2p2 = gi_int("a2p2", 5);

    hdfe::v11::HdfeRegressorV11 reg(options, threading);
    const hdfe::v11::GroupAggregation agg = parse_group_aggregation(aggregation);
    // The extractor consumes state prepared by a grouped fit (the Python
    // binding is used the same way: fit(..., group=..., individual=...)
    // followed by extract_group_individual_fes on the same regressor).
    reg.fit_grouped(y_map, X_map, fe_list, group_ids, &individual_ids, agg,
                    weights_ptr, nullptr);
    const hdfe::v11::GroupIndividualFeEstimates est = reg.extract_group_individual_fes(
        y_map, X_map, fe_list, group_ids, individual_ids, agg, weights_ptr, gi);

    Rcpp::List out;
    out["individual_ids"] = wrap_vector(est.individual_ids);
    out["individual_effects"] = wrap_vector(est.individual_effects);
    out["fe_level_ids"] = wrap_int_vector_list(est.fe_level_ids);
    out["fe_level_effects"] = wrap_vector_list(est.fe_level_effects);
    out["iterations"] = est.iterations;
    out["converged"] = est.converged;
    out["mse"] = est.mse;
    return out;
}

// [[Rcpp::export(name = ".xhdfe_cpp_build_info")]]
Rcpp::List xhdfe_cpp_build_info() {
    Rcpp::List out;
#ifdef HDFE_USE_CUDA
    out["cuda_enabled"] = true;
#else
    out["cuda_enabled"] = false;
#endif
#ifdef HDFE_USE_OPENMP
    out["openmp_enabled"] = true;
#else
    out["openmp_enabled"] = false;
#endif
    out["default_backend"] = std::string(HDFE_GPU_BACKEND_DEFAULT);
#ifdef __VERSION__
    out["compiler"] = std::string("g++ " __VERSION__);
#else
    out["compiler"] = std::string("unknown");
#endif
#ifdef XHDFE_R_MARCH
    out["march"] = std::string(XHDFE_R_MARCH);
#else
    out["march"] = std::string("");
#endif
#ifdef XHDFE_R_CUDA_ARCH
    out["cuda_arch"] = std::string(XHDFE_R_CUDA_ARCH);
#else
    out["cuda_arch"] = std::string("");
#endif
#ifdef __OPTIMIZE__
    out["optimized"] = true;
#else
    out["optimized"] = false;
#endif
#ifdef __FAST_MATH__
    out["fast_math"] = true;
#else
    out["fast_math"] = false;
#endif
    return out;
}

// ---------------------------------------------------------------------------
// AKM + leave-out (KSS) variance decomposition (opt-in module; mirrors the
// Python free functions py_hdfe_v11.akm_leave_out_set / akm_kss).
// ---------------------------------------------------------------------------

namespace {

Eigen::VectorXi as_vector_xi(const Rcpp::IntegerVector& v) {
    Eigen::VectorXi out(v.size());
    for (R_xlen_t k = 0; k < v.size(); ++k) {
        out[static_cast<int>(k)] = v[k];
    }
    return out;
}

hdfe::akm::LeaveOutLevel parse_leave_out_level(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "match" || lower == "matches") return hdfe::akm::LeaveOutLevel::Match;
    if (lower == "obs" || lower == "observation" || lower == "observations") {
        return hdfe::akm::LeaveOutLevel::Observation;
    }
    throw std::runtime_error("Unknown leave_out_level: " + name + " (use 'match' or 'obs')");
}

hdfe::akm::LeverageMethod parse_leverage_method(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "auto") return hdfe::akm::LeverageMethod::Auto;
    if (lower == "exact") return hdfe::akm::LeverageMethod::Exact;
    if (lower == "jla" || lower == "jl") return hdfe::akm::LeverageMethod::Jla;
    throw std::runtime_error("Unknown leverages method: " + name +
                             " (use 'auto', 'exact' or 'jla')");
}

Rcpp::List akm_set_to_list(const hdfe::akm::LeaveOutSetResult& s) {
    Rcpp::LogicalVector keep(static_cast<R_xlen_t>(s.keep.size()));
    for (R_xlen_t k = 0; k < keep.size(); ++k) {
        keep[k] = s.keep[static_cast<std::size_t>(k)] != 0;
    }
    Rcpp::List out;
    out["keep"] = keep;
    out["n_obs_input"] = static_cast<double>(s.n_obs_input);
    out["n_obs_connected"] = static_cast<double>(s.n_obs_connected);
    out["n_obs"] = static_cast<double>(s.n_obs);
    out["n_workers"] = s.n_workers;
    out["n_firms"] = s.n_firms;
    out["n_matches"] = static_cast<double>(s.n_matches);
    out["n_movers"] = s.n_movers;
    out["n_stayers"] = s.n_stayers;
    out["prune_iterations"] = s.prune_iterations;
    return out;
}

Rcpp::List akm_components_to_list(const hdfe::akm::AkmComponents& c,
                                  double var_y) {
    Rcpp::List out;
    out["var_alpha"] = c.var_alpha;
    out["var_psi"] = c.var_psi;
    out["cov_alpha_psi"] = c.cov_alpha_psi;
    // derived summary (pytwoway-style at-a-glance quantities)
    out["corr_alpha_psi"] =
        c.cov_alpha_psi / std::sqrt(c.var_alpha * c.var_psi);
    out["var_alpha_plus_psi"] =
        c.var_alpha + c.var_psi + 2.0 * c.cov_alpha_psi;
    if (var_y > 0.0) {
        out["share_var_alpha"] = c.var_alpha / var_y;
        out["share_var_psi"] = c.var_psi / var_y;
        out["share_2cov"] = 2.0 * c.cov_alpha_psi / var_y;
    }
    return out;
}

}  // namespace

// [[Rcpp::export(name = ".xhdfe_cpp_akm_leave_out_set")]]
Rcpp::List xhdfe_cpp_akm_leave_out_set(Rcpp::IntegerVector worker, Rcpp::IntegerVector firm) {
    if (worker.size() != firm.size()) {
        throw std::runtime_error("worker and firm must have the same length");
    }
    const Eigen::VectorXi w = as_vector_xi(worker);
    const Eigen::VectorXi f = as_vector_xi(firm);
    return akm_set_to_list(hdfe::akm::leave_out_connected_set(w, f));
}

// [[Rcpp::export(name = ".xhdfe_cpp_akm_kss")]]
Rcpp::List xhdfe_cpp_akm_kss(Rcpp::NumericVector y,
                             Rcpp::IntegerVector worker,
                             Rcpp::IntegerVector firm,
                             SEXP X,
                             Rcpp::List opts,
                             SEXP fweights = R_NilValue) {
    const R_xlen_t n = y.size();
    if (worker.size() != n || firm.size() != n) {
        throw std::runtime_error("y, worker and firm must have the same length");
    }
    Eigen::Map<const Eigen::VectorXd> y_map(&y[0], n);
    const Eigen::VectorXd y_vec = y_map;
    const Eigen::VectorXi w = as_vector_xi(worker);
    const Eigen::VectorXi f = as_vector_xi(firm);

    Eigen::MatrixXd X_mat;
    const Eigen::MatrixXd* X_ptr = nullptr;
    if (!Rf_isNull(X)) {
        Rcpp::NumericMatrix Xm(X);
        if (Xm.nrow() != n) {
            throw std::runtime_error("X must have the same number of rows as y");
        }
        if (Xm.ncol() > 0) {
            Eigen::Map<const Eigen::MatrixXd> X_map(&Xm[0], n, Xm.ncol());
            X_mat = X_map;
            X_ptr = &X_mat;
        }
    }

    hdfe::akm::AkmOptions options;
    if (opts.containsElementNamed("leave_out_level")) {
        options.leave_out_level =
            parse_leave_out_level(Rcpp::as<std::string>(opts["leave_out_level"]));
    }
    if (opts.containsElementNamed("leverages")) {
        options.leverage_method =
            parse_leverage_method(Rcpp::as<std::string>(opts["leverages"]));
    }
    if (opts.containsElementNamed("jla_draws")) {
        options.jla_draws = Rcpp::as<int>(opts["jla_draws"]);
    }
    if (opts.containsElementNamed("seed")) {
        options.seed = static_cast<std::uint64_t>(Rcpp::as<double>(opts["seed"]));
    }
    if (opts.containsElementNamed("prune")) {
        options.prune = Rcpp::as<bool>(opts["prune"]);
    }
    if (opts.containsElementNamed("exact_max_rows")) {
        options.exact_max_rows = Rcpp::as<int>(opts["exact_max_rows"]);
    }
    if (opts.containsElementNamed("direct_max_firms")) {
        options.direct_max_firms = Rcpp::as<int>(opts["direct_max_firms"]);
    }
    if (opts.containsElementNamed("direct_max_nnz")) {
        options.direct_max_nnz =
            static_cast<long long>(Rcpp::as<double>(opts["direct_max_nnz"]));
    }
    if (opts.containsElementNamed("cg_tol")) {
        options.cg_tol = Rcpp::as<double>(opts["cg_tol"]);
    }
    if (opts.containsElementNamed("cg_max_iter")) {
        options.cg_max_iter = Rcpp::as<int>(opts["cg_max_iter"]);
    }
    if (opts.containsElementNamed("num_threads")) {
        options.num_threads = Rcpp::as<int>(opts["num_threads"]);
    }
    if (opts.containsElementNamed("fwl_tol")) {
        options.fwl_tol = Rcpp::as<double>(opts["fwl_tol"]);
    }
    if (opts.containsElementNamed("fwl_max_iter")) {
        options.fwl_max_iter = Rcpp::as<int>(opts["fwl_max_iter"]);
    }
    if (opts.containsElementNamed("compute_se")) {
        options.compute_se = Rcpp::as<bool>(opts["compute_se"]);
    }
    if (opts.containsElementNamed("se_nsim")) {
        options.se_nsim = Rcpp::as<int>(opts["se_nsim"]);
    }
    if (opts.containsElementNamed("gpu")) {
        options.use_gpu = Rcpp::as<bool>(opts["gpu"]);
    }
    if (opts.containsElementNamed("eigen_diagnostics")) {
        options.eigen_diagnostics = Rcpp::as<bool>(opts["eigen_diagnostics"]);
        if (options.eigen_diagnostics) {
            options.compute_se = true;
        }
    }
    if (opts.containsElementNamed("eig_trace_nsim")) {
        options.eig_trace_nsim = Rcpp::as<int>(opts["eig_trace_nsim"]);
    }
    if (opts.containsElementNamed("se_sigma_lowess")) {
        options.se_sigma_lowess = Rcpp::as<bool>(opts["se_sigma_lowess"]);
    }

    std::optional<Eigen::VectorXd> fw_vec;
    if (fweights != R_NilValue) {
        Rcpp::NumericVector fwv(fweights);
        if (fwv.size() != n) {
            Rcpp::stop("fweights must have the same length as y");
        }
        fw_vec = Eigen::VectorXd(n);
        for (R_xlen_t i2 = 0; i2 < n; ++i2) (*fw_vec)[i2] = fwv[i2];
    }
    const hdfe::akm::AkmKssResult res = hdfe::akm::akm_kss_decompose(
        y_vec, w, f, X_ptr, options, nullptr, fw_vec ? &(*fw_vec) : nullptr);

    Rcpp::List out;
    out["sample"] = akm_set_to_list(res.sample);
    out["alpha"] = Rcpp::NumericVector(res.alpha.data(), res.alpha.data() + res.alpha.size());
    out["psi"] = Rcpp::NumericVector(res.psi.data(), res.psi.data() + res.psi.size());
    out["beta"] = Rcpp::NumericVector(res.beta.data(), res.beta.data() + res.beta.size());
    out["plugin"] = akm_components_to_list(res.plugin, res.var_y);
    out["agsu"] = akm_components_to_list(res.agsu, res.var_y);
    out["kss"] = akm_components_to_list(res.kss, res.var_y);
    out["var_y"] = res.var_y;
    out["sigma2_ho"] = res.sigma2_ho;
    out["pii"] = Rcpp::NumericVector(res.pii.data(), res.pii.data() + res.pii.size());
    out["sigma_i"] =
        Rcpp::NumericVector(res.sigma_i.data(), res.sigma_i.data() + res.sigma_i.size());
    out["row_worker"] = Rcpp::IntegerVector(res.row_worker.data(),
                                            res.row_worker.data() + res.row_worker.size());
    out["row_firm"] = Rcpp::IntegerVector(res.row_firm.data(),
                                          res.row_firm.data() + res.row_firm.size());
    out["row_weight"] = Rcpp::NumericVector(res.row_weight.data(),
                                            res.row_weight.data() + res.row_weight.size());
    if (options.compute_se) {
        Rcpp::List cse;
        cse["se_var_psi"] = res.se_var_psi;
        cse["se_cov_alpha_psi"] = res.se_cov_alpha_psi;
        cse["se_var_alpha"] = res.se_var_alpha;
        cse["theta_var_psi"] = res.theta_c_var_psi;
        cse["theta_cov_alpha_psi"] = res.theta_c_cov_alpha_psi;
        cse["theta_var_alpha"] = res.theta_c_var_alpha;
        out["component_se"] = cse;
    }
    if (options.eigen_diagnostics) {
        const char* comps[3] = {"var_psi", "cov_alpha_psi", "var_alpha"};
        Rcpp::List wk;
        for (int c = 0; c < 3; ++c) {
            Rcpp::List cd;
            cd["lambda1"] = res.eig_lambda1[c];
            cd["eig_share1"] = res.eig_share1[c];
            cd["eig_share2"] = res.eig_share2[c];
            cd["eig_share3"] = res.eig_share3[c];
            cd["lindeberg_max_x1bar_sq"] = res.lindeberg_max_x1bar_sq[c];
            cd["gamma_sq"] = res.gamma_sq[c];
            cd["f_stat"] = res.f_stat[c];
            cd["theta_1"] = res.theta_1[c];
            cd["ci_lb"] = res.ci_lb[c];
            cd["ci_ub"] = res.ci_ub[c];
            cd["curvature"] = res.curvature[c];
            cd["b_1"] = res.b_1[c];
            cd["cov_r1_11"] = res.cov_r1_11[c];
            cd["cov_r1_12"] = res.cov_r1_12[c];
            cd["cov_r1_22"] = res.cov_r1_22[c];
            wk[comps[c]] = cd;
        }
        out["weak_id"] = wk;
    }
    out["max_pii"] = res.max_pii;
    out["mean_pii"] = res.mean_pii;
    out["n_rows"] = static_cast<double>(res.n_rows);
    out["leverages_exact"] = res.leverages_exact;
    out["gpu_used"] = res.gpu_used;
    out["solver_direct"] = res.solver_direct;
    out["jla_draws_used"] = res.jla_draws_used;
    out["seed"] = static_cast<double>(res.seed_used);
    out["solver_iterations"] = static_cast<double>(res.solver_iterations);
    out["converged"] = res.converged;
    out["notes"] = res.notes;
    return out;
}

// ---------------------------------------------------------------------------
// Gelbach (2016) conditional decomposition, HDFE-aware (M9B). Mirrors the
// Python py_hdfe_v11.gelbach_decompose free function.
// ---------------------------------------------------------------------------

// [[Rcpp::export(name = ".xhdfe_cpp_gelbach")]]
Rcpp::List xhdfe_cpp_gelbach(Rcpp::NumericVector y,
                             Rcpp::NumericMatrix X1,
                             SEXP X2,
                             Rcpp::IntegerVector x2_group_sizes,
                             Rcpp::List fes,
                             SEXP cluster,
                             std::string vce,
                             bool gamma0,
                             bool cov0,
                             int num_threads,
                             SEXP weights,
                             bool fweights) {
    const R_xlen_t n = y.size();
    if (X1.nrow() != n) {
        throw std::runtime_error("x1 must have the same number of rows as y");
    }
    Eigen::Map<const Eigen::VectorXd> y_map(&y[0], n);
    const Eigen::VectorXd y_vec = y_map;
    Eigen::Map<const Eigen::MatrixXd> X1_map(X1.size() > 0 ? &X1[0] : nullptr, n, X1.ncol());
    const Eigen::MatrixXd X1_mat = X1_map;
    Eigen::MatrixXd X2_mat(n, 0);
    if (!Rf_isNull(X2)) {
        Rcpp::NumericMatrix X2m(X2);
        if (X2m.nrow() != n) {
            throw std::runtime_error("x2 must have the same number of rows as y");
        }
        if (X2m.ncol() > 0) {
            Eigen::Map<const Eigen::MatrixXd> X2_map(&X2m[0], n, X2m.ncol());
            X2_mat = X2_map;
        }
    }
    std::vector<int> sizes(x2_group_sizes.begin(), x2_group_sizes.end());
    std::vector<Eigen::VectorXi> fe_list;
    fe_list.reserve(static_cast<std::size_t>(fes.size()));
    for (R_xlen_t d = 0; d < fes.size(); ++d) {
        fe_list.push_back(as_vector_xi(Rcpp::IntegerVector(fes[d])));
    }
    Eigen::VectorXi cl;
    const Eigen::VectorXi* cl_ptr = nullptr;
    if (!Rf_isNull(cluster)) {
        cl = as_vector_xi(Rcpp::IntegerVector(cluster));
        cl_ptr = &cl;
    }
    hdfe::gelbach::GelbachOptions opt;
    {
        const std::string lower = to_lower(vce);
        if (lower == "unadjusted" || lower == "homoskedastic" || lower == "iid") {
            opt.vce = hdfe::gelbach::GelbachVce::Unadjusted;
        } else if (lower == "robust") {
            opt.vce = hdfe::gelbach::GelbachVce::Robust;
        } else if (lower == "cluster") {
            opt.vce = hdfe::gelbach::GelbachVce::Cluster;
        } else {
            throw std::runtime_error("unknown vce: " + vce);
        }
    }
    opt.gamma0 = gamma0;
    opt.cov0 = cov0;
    opt.num_threads = num_threads;
    Eigen::VectorXd w_vec;
    const Eigen::VectorXd* w_ptr = nullptr;
    if (!Rf_isNull(weights)) {
        Rcpp::NumericVector wv(weights);
        if (wv.size() != n) throw std::runtime_error("weights length mismatch");
        Eigen::Map<const Eigen::VectorXd> w_map(&wv[0], n);
        w_vec = w_map;
        w_ptr = &w_vec;
    }
    const hdfe::gelbach::GelbachResult r =
        hdfe::gelbach::decompose(y_vec, X1_mat, X2_mat, sizes, fe_list, cl_ptr, opt,
                                 w_ptr, fweights);
    const auto to_rmat = [](const Eigen::MatrixXd& m) {
        Rcpp::NumericMatrix out(static_cast<int>(m.rows()), static_cast<int>(m.cols()));
        std::copy(m.data(), m.data() + m.size(), out.begin());
        return out;
    };
    Rcpp::List out;
    out["b_base"] = Rcpp::NumericVector(r.b_base.data(), r.b_base.data() + r.b_base.size());
    out["b_full"] = Rcpp::NumericVector(r.b_full.data(), r.b_full.data() + r.b_full.size());
    out["delta"] = to_rmat(r.delta);
    out["cov"] = to_rmat(r.cov);
    out["total"] = Rcpp::NumericVector(r.total.data(), r.total.data() + r.total.size());
    out["total_cov"] = to_rmat(r.total_cov);
    out["identity_gap"] = r.identity_gap;
    out["n_obs"] = static_cast<double>(r.n_obs);
    out["df_full"] = r.df_full;
    out["converged"] = r.converged;
    out["notes"] = r.notes;
    return out;
}
