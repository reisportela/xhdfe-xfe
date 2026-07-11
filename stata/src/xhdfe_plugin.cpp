#include "stplugin.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hdfe/akm_kss.hpp"
#include "hdfe/hdfe_regressor_v11.hpp"

// Parallel sort when libstdc++ parallel mode is available (GCC host compiler
// with OpenMP, as used by the plugin build); otherwise fall back to std::sort.
// The only use is the categorical dense-rank below, whose output is invariant
// to how the sort orders equal keys, so results are bit-for-bit unchanged.
#if defined(__GNUC__) && defined(_OPENMP) && __has_include(<parallel/algorithm>)
#include <parallel/algorithm>
#define XHDFE_PARALLEL_SORT(first, last, cmp) __gnu_parallel::sort((first), (last), (cmp))
#else
#define XHDFE_PARALLEL_SORT(first, last, cmp) std::sort((first), (last), (cmp))
#endif

namespace {

using hdfe::AbsorptionMethod;
using hdfe::ConvergenceCriterion;
using hdfe::DofAdjustmentMethod;
using hdfe::FeRecoveryMethod;
using hdfe::FixefDofMethod;
using hdfe::HdfeOptions;
using hdfe::StandardErrorType;
using hdfe::StatsStyle;
using hdfe::ToleranceMode;
using hdfe::ClusterDofMethod;
using hdfe::detail::HeterogeneousSlopeTerm;
using hdfe::v11::GroupAggregation;
using hdfe::v11::HdfeRegressorV11;
using hdfe::v11::ThreadingOptions;

constexpr double kGroupConstantTol = 1e-12;

[[noreturn]] void throw_with_prefix(const char* prefix, const std::string& msg);

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string normalize_gpu_backend(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "cpu" || name == "cuda" || name == "metal") {
        return name;
    }
    throw_with_prefix("xhdfe plugin: ", "invalid gpu_backend: " + raw);
}

void set_env_value(const std::string& key, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), value.c_str()) != 0) {
        throw_with_prefix("xhdfe plugin: ", "failed to set environment variable " + key);
    }
#else
    if (setenv(key.c_str(), value.c_str(), 1) != 0) {
        throw_with_prefix("xhdfe plugin: ", "failed to set environment variable " + key);
    }
#endif
}

void unset_env_value(const std::string& key) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), "") != 0) {
        throw_with_prefix("xhdfe plugin: ", "failed to unset environment variable " + key);
    }
#else
    if (unsetenv(key.c_str()) != 0) {
        throw_with_prefix("xhdfe plugin: ", "failed to unset environment variable " + key);
    }
#endif
}

struct ScopedEnvVar {
    std::string key;
    std::optional<std::string> previous;
    bool active = false;

    ScopedEnvVar() = default;
    ScopedEnvVar(const std::string& key_in, const std::string& value)
        : key(key_in) {
        const char* prev = std::getenv(key.c_str());
        if (prev) {
            previous = std::string(prev);
        }
        set_env_value(key, value);
        active = true;
    }

    ~ScopedEnvVar() noexcept {
        if (!active) {
            return;
        }
        try {
            if (previous.has_value()) {
                set_env_value(key, *previous);
            } else {
                unset_env_value(key);
            }
        } catch (...) {
            // Avoid throwing from destructor.
        }
    }
};

bool plugin_savefe_profile_enabled() {
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

void plugin_savefe_profile_log(const std::string& msg) {
    if (!plugin_savefe_profile_enabled()) {
        return;
    }
    std::cerr << "savefe_profile " << msg << '\n';
}

class PluginSavefeTimer {
public:
    explicit PluginSavefeTimer(const char* label)
        : label_(label), enabled_(plugin_savefe_profile_enabled()) {
        if (enabled_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~PluginSavefeTimer() {
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

bool plugin_cpu_profile_enabled() {
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

void plugin_cpu_profile_log_elapsed(
    const char* label,
    const std::chrono::steady_clock::time_point& start) {
    if (!plugin_cpu_profile_enabled()) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cerr << "plugin_profile label=" << label << " ms=" << elapsed.count() << '\n';
}

void plugin_cpu_profile_log_elapsed(
    const std::string& label,
    const std::chrono::steady_clock::time_point& start) {
    plugin_cpu_profile_log_elapsed(label.c_str(), start);
}

bool approx_equal(double a, double b) {
    const double scale = 1.0 + std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= kGroupConstantTol * scale;
}

[[noreturn]] void throw_with_prefix(const char* prefix, const std::string& msg) {
    throw std::runtime_error(std::string(prefix) + msg);
}

struct ParsedArgs {
    std::unordered_map<std::string, std::string> kv;

    explicit ParsedArgs(int argc, char* argv[]) {
        kv.reserve(static_cast<std::size_t>(std::max(1, argc)));

        auto parse_item = [&](const std::string& item) {
            if (item.empty()) {
                return;
            }
            const std::size_t eq = item.find('=');
            if (eq == std::string::npos || eq == 0) {
                throw_with_prefix("xhdfe plugin: ", "invalid argument (expected key=value): " + item);
            }
            std::string key = item.substr(0, eq);
            std::string val = item.substr(eq + 1);
            kv.emplace(std::move(key), std::move(val));
        };

        auto parse_packed = [&](std::string packed) {
            if (packed.rfind("cfg=", 0) == 0) {
                packed.erase(0, 4);
            }
            std::string buf;
            buf.reserve(packed.size());
            for (char c : packed) {
                if (c == ';') {
                    parse_item(buf);
                    buf.clear();
                } else {
                    buf.push_back(c);
                }
            }
            parse_item(buf);
        };

        if (argc == 1) {
            const std::string only = argv[0] ? argv[0] : "";
            if (only.find(';') != std::string::npos || only.rfind("cfg=", 0) == 0) {
                parse_packed(only);
                return;
            }
        }

        for (int i = 0; i < argc; ++i) {
            const std::string item = argv[i] ? argv[i] : "";
            parse_item(item);
        }
    }

    bool has(const char* key) const { return kv.find(key) != kv.end(); }

    const std::string& get_required(const char* key) const {
        auto it = kv.find(key);
        if (it == kv.end()) {
            throw_with_prefix("xhdfe plugin: ", std::string("missing required argument: ") + key);
        }
        return it->second;
    }

    std::optional<std::string> get_optional(const char* key) const {
        auto it = kv.find(key);
        if (it == kv.end()) {
            return std::nullopt;
        }
        return it->second;
    }
};

bool parse_bool(const std::string& raw, const char* what) {
    const std::string s = to_lower(raw);
    if (s == "1" || s == "true" || s == "t" || s == "yes" || s == "y") {
        return true;
    }
    if (s == "0" || s == "false" || s == "f" || s == "no" || s == "n") {
        return false;
    }
    throw_with_prefix("xhdfe plugin: ", std::string("invalid boolean for ") + what + ": " + raw);
}

int parse_int(const std::string& raw, const char* what) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0') {
        throw_with_prefix("xhdfe plugin: ", std::string("invalid integer for ") + what + ": " + raw);
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        throw_with_prefix("xhdfe plugin: ", std::string("integer out of range for ") + what + ": " + raw);
    }
    return static_cast<int>(v);
}

double parse_double(const std::string& raw, const char* what) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(raw.c_str(), &end);
    if (errno != 0 || end == raw.c_str() || *end != '\0') {
        throw_with_prefix("xhdfe plugin: ", std::string("invalid numeric for ") + what + ": " + raw);
    }
    return v;
}

std::vector<int> parse_csv_ints_0based(const std::string& raw) {
    std::vector<int> out;
    std::string token;
    auto flush = [&]() {
        if (token.empty()) {
            return;
        }
        out.push_back(parse_int(token, "endogenous_idx"));
        token.clear();
    };
    for (char c : raw) {
        if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            flush();
        } else {
            token.push_back(c);
        }
    }
    flush();
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<int> parse_csv_ints_ordered(const std::string& raw, const char* what) {
    std::vector<int> out;
    std::string token;
    auto flush = [&]() {
        if (token.empty()) {
            return;
        }
        out.push_back(parse_int(token, what));
        token.clear();
    };
    for (char c : raw) {
        if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            flush();
        } else {
            token.push_back(c);
        }
    }
    flush();
    return out;
}

StandardErrorType parse_se_type(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "homoskedastic" || name == "classical" || name == "unadjusted" || name == "unadj" ||
        name == "ols") {
        return StandardErrorType::Homoskedastic;
    }
    if (name == "robust" || name == "hc1" || name == "heteroskedastic") {
        return StandardErrorType::Robust;
    }
    if (name == "cluster" || name == "clustered") {
        return StandardErrorType::Cluster;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown se_type: " + raw);
}

AbsorptionMethod parse_absorption_method(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "auto") {
        return AbsorptionMethod::Auto;
    }
    if (name == "gs" || name == "gauss-seidel" || name == "gauss_seidel") {
        return AbsorptionMethod::GaussSeidel;
    }
    if (name == "sym" || name == "symmetric" || name == "symgs" || name == "symmetric-gauss-seidel" ||
        name == "symmetric_gauss_seidel") {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    if (name == "jacobi") {
        return AbsorptionMethod::Jacobi;
    }
    if (name == "lsmr" || name == "plain-lsmr" || name == "plain_lsmr") {
        return AbsorptionMethod::Lsmr;
    }
    if (name == "mlsmr" || name == "modified-lsmr" || name == "modified_lsmr" ||
        name == "within" || name == "within-additive" || name == "within_additive") {
        return AbsorptionMethod::Mlsmr;
    }
    if (name == "auto-mlsmr" || name == "auto_mlsmr" ||
        name == "mlsmr-auto" || name == "mlsmr_auto") {
        return AbsorptionMethod::AutoMlsmr;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown absorption_method: " + raw);
}

ConvergenceCriterion parse_convergence_criterion(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "auto" || name.empty()) {
        return ConvergenceCriterion::Auto;
    }
    if (name == "normchange" || name == "norm" || name == "norm_change" ||
        name == "current" || name == "xhdfe") {
        return ConvergenceCriterion::NormChange;
    }
    if (name == "reghdfe" || name == "vector" || name == "vectors" ||
        name == "reghdfe-style" || name == "reghdfe_style") {
        return ConvergenceCriterion::Reghdfe;
    }
    if (name == "both") {
        return ConvergenceCriterion::Both;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown convergence criterion: " + raw);
}

FeRecoveryMethod parse_fe_recovery_method(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "map" || name == "ap") {
        return FeRecoveryMethod::Map;
    }
    if (name == "hybrid" || name == "auto") {
        return FeRecoveryMethod::Hybrid;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown fe_recovery_method: " + raw);
}

FixefDofMethod parse_fixef_method(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "full" || name == "true" || name == "yes") {
        return FixefDofMethod::Full;
    }
    if (name == "none" || name == "false" || name == "no") {
        return FixefDofMethod::None;
    }
    if (name == "nonnested" || name == "non-nested" || name == "non_nested") {
        return FixefDofMethod::Nonnested;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown K.fixef: " + raw);
}

ClusterDofMethod parse_cluster_df(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "min") {
        return ClusterDofMethod::Min;
    }
    if (name == "conventional" || name == "conv") {
        return ClusterDofMethod::Conventional;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown G.df: " + raw);
}

StatsStyle parse_stats_style(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "reghdfe" || name == "reghdfe-style" || name == "reghdfe_style") {
        return StatsStyle::Reghdfe;
    }
    if (name == "legacy" || name == "current" || name == "xhdfe") {
        return StatsStyle::Legacy;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown statstyle: " + raw);
}

ToleranceMode parse_tolerance_mode(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "xhdfe-fast" || name == "xhdfe_fast" || name == "xhdfe" ||
        name == "fast" || name == "current") {
        return ToleranceMode::XhdfeFast;
    }
    if (name == "reghdfe-comparable" || name == "reghdfe_comparable" ||
        name == "reghdfe" || name == "comparable") {
        return ToleranceMode::ReghdfeComparable;
    }
    if (name == "strict-residual" || name == "strict_residual" ||
        name == "residual-certificate" || name == "residual_certificate" ||
        name == "strict") {
        return ToleranceMode::StrictResidual;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown tolerance_mode: " + raw);
}

DofAdjustmentMethod parse_dof_method(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "all") {
        return DofAdjustmentMethod::All;
    }
    if (name == "none") {
        return DofAdjustmentMethod::None;
    }
    if (name == "firstpair" || name == "first") {
        return DofAdjustmentMethod::FirstPair;
    }
    if (name == "pairwise" || name == "pair") {
        return DofAdjustmentMethod::Pairwise;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown dof_method: " + raw);
}

struct ParsedDofAdjustments {
    DofAdjustmentMethod method = DofAdjustmentMethod::All;
    bool adjust_clusters = true;
    bool adjust_continuous = true;
};

ParsedDofAdjustments parse_dofadjustments(std::optional<std::string> raw_opt) {
    ParsedDofAdjustments parsed;
    if (!raw_opt) {
        return parsed;
    }

    // When the user explicitly supplies dofadjustments(...), mimic Stata/reghdfe behavior by
    // starting from a blank slate (only the listed items are enabled).
    parsed.method = DofAdjustmentMethod::All;
    parsed.adjust_clusters = false;
    parsed.adjust_continuous = false;

    std::string raw = *raw_opt;
    std::replace(raw.begin(), raw.end(), ',', ' ');
    std::string token;
    auto flush = [&]() {
        if (token.empty()) {
            return;
        }
        const std::string tok = to_lower(token);
        token.clear();
        if (tok == "all") {
            parsed.method = DofAdjustmentMethod::All;
            parsed.adjust_clusters = true;
            parsed.adjust_continuous = true;
            return;
        }
        if (tok == "none") {
            parsed.method = DofAdjustmentMethod::None;
            return;
        }
        if (tok == "firstpair" || tok == "first") {
            parsed.method = DofAdjustmentMethod::FirstPair;
            return;
        }
        if (tok == "pairwise" || tok == "pair") {
            parsed.method = DofAdjustmentMethod::Pairwise;
            return;
        }
        if (tok == "clusters" || tok == "cluster") {
            parsed.adjust_clusters = true;
            return;
        }
        if (tok == "continuous" || tok == "cont") {
            parsed.adjust_continuous = true;
            return;
        }
        throw_with_prefix("xhdfe plugin: ", "unknown dofadjustments token: " + tok);
    };
    for (char c : raw) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            flush();
        } else {
            token.push_back(c);
        }
    }
    flush();
    return parsed;
}

GroupAggregation parse_aggregation(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "mean" || name == "average" || name == "avg") {
        return GroupAggregation::Mean;
    }
    if (name == "sum") {
        return GroupAggregation::Sum;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown aggregation: " + raw);
}

void maybe_save_scalar(const std::optional<std::string>& name, double value) {
    if (!name || name->empty()) {
        return;
    }
    const ST_retcode rc = SF_scal_save(const_cast<char*>(name->c_str()), value);
    if (rc) {
        throw_with_prefix("xhdfe plugin: ", "failed to save scalar: " + *name);
    }
}

void maybe_save_gpu_diagnostics(const std::optional<std::string>& s_gpu_status_code,
                                const std::optional<std::string>& s_gpu_attempted,
                                const std::optional<std::string>& s_gpu_absorption_converged,
                                const std::optional<std::string>& s_gpu_absorption_iterations,
                                const HdfeRegressorV11& reg) {
    maybe_save_scalar(s_gpu_status_code, static_cast<double>(reg.gpu_status_code()));
    maybe_save_scalar(s_gpu_attempted, reg.gpu_attempted() ? 1.0 : 0.0);
    const double gpu_absorption_converged =
        reg.gpu_attempted() ? (reg.gpu_absorption_converged() ? 1.0 : 0.0) : SV_missval;
    const double gpu_absorption_iterations =
        reg.gpu_attempted() ? static_cast<double>(reg.gpu_absorption_iterations()) : SV_missval;
    maybe_save_scalar(s_gpu_absorption_converged,
                      gpu_absorption_converged);
    maybe_save_scalar(s_gpu_absorption_iterations,
                      gpu_absorption_iterations);
}

void store_matrix(const std::string& mat, const Eigen::MatrixXd& m) {
    const int want_r = static_cast<int>(m.rows());
    const int want_c = static_cast<int>(m.cols());
    const int have_r = SF_row(const_cast<char*>(mat.c_str()));
    const int have_c = SF_col(const_cast<char*>(mat.c_str()));
    if (have_r != want_r || have_c != want_c) {
        throw_with_prefix(
            "xhdfe plugin: ",
            "matrix dimension mismatch for " + mat + " (have " + std::to_string(have_r) + "x" +
                std::to_string(have_c) + ", want " + std::to_string(want_r) + "x" +
                std::to_string(want_c) + ")");
    }

    for (int i = 0; i < want_r; ++i) {
        for (int j = 0; j < want_c; ++j) {
            const double val = std::isfinite(m(i, j)) ? m(i, j) : SV_missval;
            const ST_retcode rc =
                SF_mat_store(const_cast<char*>(mat.c_str()), i + 1, j + 1, val);
            if (rc) {
                throw_with_prefix("xhdfe plugin: ", "failed to write matrix " + mat);
            }
        }
    }
}

void store_row_vector(const std::string& mat, const Eigen::VectorXd& v) {
    Eigen::MatrixXd m(1, v.size());
    m.row(0) = v.transpose();
    store_matrix(mat, m);
}

void maybe_store_cluster_diag(const std::optional<std::string>& name,
                              const hdfe::HdfeResults& res) {
    if (!name || name->empty()) {
        return;
    }
    const int want_rows = static_cast<int>(res.cluster_combo_counts.size());
    const int have_rows = SF_row(const_cast<char*>(name->c_str()));
    const int have_cols = SF_col(const_cast<char*>(name->c_str()));
    if (have_rows != want_rows || have_cols != 1) {
        throw_with_prefix("xhdfe plugin: ",
                          "cluster_diag has wrong dimensions (have " +
                              std::to_string(have_rows) + "x" + std::to_string(have_cols) +
                              ", want " + std::to_string(want_rows) + "x1)");
    }
    Eigen::MatrixXd out(want_rows, 1);
    for (int i = 0; i < want_rows; ++i) {
        out(i, 0) = static_cast<double>(res.cluster_combo_counts[static_cast<std::size_t>(i)]);
    }
    store_matrix(*name, out);
}

void maybe_store_dof_table(const std::optional<std::string>& name,
                           const hdfe::HdfeResults& res) {
    if (!name || name->empty()) {
        return;
    }
    const int rows = static_cast<int>(res.fe_num_levels.size());
    if (rows <= 0) {
        return;
    }
    // Column 6 (optional, present when the ado allocates it): distinct FE
    // levels in the estimation sample per dimension. It lets the ado skip the
    // O(N log N) egen-tag recount for heterogeneous-slope dimensions.
    const int have_rows = SF_row(const_cast<char*>(name->c_str()));
    const int have_cols = SF_col(const_cast<char*>(name->c_str()));
    if (have_rows != rows || (have_cols != 5 && have_cols != 6)) {
        throw_with_prefix(
            "xhdfe plugin: ",
            "dof_table has wrong dimensions (have " + std::to_string(have_rows) + "x" +
                std::to_string(have_cols) + ", want " + std::to_string(rows) + "x5 or x6)");
    }

    auto get_val = [&](const std::vector<int>& vec, int idx) -> double {
        if (idx < 0 || idx >= static_cast<int>(vec.size())) {
            return SV_missval;
        }
        return static_cast<double>(vec[static_cast<std::size_t>(idx)]);
    };

    for (int i = 0; i < rows; ++i) {
        const double col1 = get_val(res.fe_num_levels, i);
        const double col2 = get_val(res.fe_redundant, i);
        const double col3 = get_val(res.fe_num_coefs, i);
        const double col4 = get_val(res.fe_inexact, i);
        const double col5 = get_val(res.fe_nested, i);
        const ST_retcode rc1 =
            SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 1, col1);
        const ST_retcode rc2 =
            SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 2, col2);
        const ST_retcode rc3 =
            SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 3, col3);
        const ST_retcode rc4 =
            SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 4, col4);
        const ST_retcode rc5 =
            SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 5, col5);
        ST_retcode rc6 = 0;
        if (have_cols >= 6) {
            rc6 = SF_mat_store(const_cast<char*>(name->c_str()), i + 1, 6,
                               get_val(res.fe_base_levels, i));
        }
        if (rc1 || rc2 || rc3 || rc4 || rc5 || rc6) {
            throw_with_prefix("xhdfe plugin: ", "failed to write matrix dof_table");
        }
    }
}

int to_int_checked(double value, const char* what);

std::vector<int> read_matrix_vector_int(const std::string& name, const char* what) {
    const ST_double missval = SV_missval;
    const int rows = SF_row(const_cast<char*>(name.c_str()));
    const int cols = SF_col(const_cast<char*>(name.c_str()));
    if (rows <= 0 || cols != 1) {
        throw_with_prefix("xhdfe plugin: ",
                          std::string(what) + " has wrong dimensions (have " +
                              std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    std::vector<int> out(static_cast<std::size_t>(rows), 0);
    for (int i = 0; i < rows; ++i) {
        double val = 0.0;
        const ST_retcode rc = SF_mat_el(const_cast<char*>(name.c_str()), i + 1, 1, &val);
        if (rc) {
            throw_with_prefix("xhdfe plugin: ",
                              std::string("failed to read ") + what + " matrix");
        }
        if (!std::isfinite(val) || !(val < missval)) {
            out[static_cast<std::size_t>(i)] = 0;
            continue;
        }
        out[static_cast<std::size_t>(i)] = to_int_checked(val, what);
    }
    return out;
}

void maybe_store_omit_reason(const std::optional<std::string>& name,
                             const hdfe::HdfeResults& res) {
    if (!name || name->empty()) {
        return;
    }
    const int rows = static_cast<int>(res.coefficients.size());
    if (rows <= 0) {
        return;
    }
    const int have_rows = SF_row(const_cast<char*>(name->c_str()));
    const int have_cols = SF_col(const_cast<char*>(name->c_str()));
    if (have_rows != rows || have_cols != 1) {
        throw_with_prefix("xhdfe plugin: ",
                          "omit_reason has wrong dimensions (have " +
                              std::to_string(have_rows) + "x" + std::to_string(have_cols) +
                              ", want " + std::to_string(rows) + "x1)");
    }
    Eigen::MatrixXd out(rows, 1);
    if (res.omitted_reason.empty()) {
        out.setZero();
    } else {
        if (static_cast<int>(res.omitted_reason.size()) != rows) {
            throw_with_prefix("xhdfe plugin: ", "omit_reason size mismatch");
        }
        for (int i = 0; i < rows; ++i) {
            out(i, 0) = static_cast<double>(res.omitted_reason[static_cast<std::size_t>(i)]);
        }
    }
    store_matrix(*name, out);
}

int to_int_checked(double value, const char* what) {
    const double rounded = std::round(value);
    if (!approx_equal(value, rounded)) {
        throw_with_prefix("xhdfe plugin: ", std::string("non-integer value in ") + what);
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        throw_with_prefix("xhdfe plugin: ", std::string("integer out of range in ") + what);
    }
    return static_cast<int>(rounded);
}

struct ObservationMap {
    int in1 = 0;
    int n = 0;
    bool contiguous = false;
    std::vector<int> obs;

    int obs_no(int i) const {
        return contiguous ? in1 + i : obs[static_cast<std::size_t>(i)];
    }
};

ObservationMap selected_observations() {
    const int in1 = SF_in1();
    const int in2 = SF_in2();
    ObservationMap out;
    out.in1 = in1;
    if (in2 < in1) {
        return out;
    }
    out.contiguous = true;
    const int capacity = in2 - in1 + 1;
    for (int j = in1; j <= in2; ++j) {
        if (SF_ifobs(j)) {
            if (out.contiguous && j == in1 + out.n) {
                ++out.n;
                continue;
            }
            if (out.contiguous) {
                out.contiguous = false;
                out.obs.reserve(static_cast<std::size_t>(capacity));
                for (int k = 0; k < out.n; ++k) {
                    out.obs.push_back(in1 + k);
                }
            }
            out.obs.push_back(j);
            ++out.n;
        } else if (out.contiguous) {
            out.contiguous = false;
            out.obs.reserve(static_cast<std::size_t>(capacity));
            for (int k = 0; k < out.n; ++k) {
                out.obs.push_back(in1 + k);
            }
        }
    }
    if (out.contiguous) {
        out.obs.clear();
        out.obs.shrink_to_fit();
    }
    return out;
}

// Helpers for group/individual singleton dropping (mirrors v11 core logic).
struct FeLookup {
    bool dense = false;
    int min_id = 0;
    int num_groups = 0;
    std::unordered_map<int, int> mapping;

    int index(int raw) const {
        if (dense) {
            return raw - min_id;
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
        lookup.dense = true;
        lookup.min_id = min_id;
        lookup.num_groups = static_cast<int>(range);
        return lookup;
    }

    lookup.dense = false;
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
            const int only = gi.individual_group[static_cast<std::size_t>(
                gi.individual_ptr[static_cast<std::size_t>(i)])];
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

hdfe::detail::GroupIndividualStructure build_group_individual_structure(
    const Eigen::VectorXi& group_index,
    const Eigen::VectorXi& individual_index,
    int num_groups,
    int num_individuals,
    GroupAggregation aggregation) {
    if (group_index.size() != individual_index.size()) {
        throw std::runtime_error("group_index and individual_index must have same length");
    }
    const int n = group_index.size();
    std::vector<uint64_t> edges;
    edges.reserve(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        const uint32_t g = static_cast<uint32_t>(group_index(r));
        const uint32_t i = static_cast<uint32_t>(individual_index(r));
        edges.push_back((static_cast<uint64_t>(g) << 32) | static_cast<uint64_t>(i));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    hdfe::detail::GroupIndividualStructure gi;
    gi.num_groups = num_groups;
    gi.num_individuals = num_individuals;
    gi.group_ptr.assign(static_cast<std::size_t>(num_groups) + 1, 0);
    for (const uint64_t key : edges) {
        const int g = static_cast<int>(key >> 32);
        gi.group_ptr[static_cast<std::size_t>(g + 1)] += 1;
    }
    for (int g = 0; g < num_groups; ++g) {
        gi.group_ptr[static_cast<std::size_t>(g + 1)] += gi.group_ptr[static_cast<std::size_t>(g)];
    }
    gi.group_individual.assign(static_cast<std::size_t>(gi.group_ptr.back()), 0);
    std::vector<int> cursor = gi.group_ptr;
    for (const uint64_t key : edges) {
        const int g = static_cast<int>(key >> 32);
        const int i = static_cast<int>(key & 0xffffffffULL);
        gi.group_individual[static_cast<std::size_t>(cursor[static_cast<std::size_t>(g)]++)] = i;
    }

    std::vector<uint64_t> edges_by_indiv;
    edges_by_indiv.reserve(edges.size());
    for (const uint64_t key : edges) {
        const uint32_t g = static_cast<uint32_t>(key >> 32);
        const uint32_t i = static_cast<uint32_t>(key & 0xffffffffULL);
        edges_by_indiv.push_back((static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(g));
    }
    std::sort(edges_by_indiv.begin(), edges_by_indiv.end());
    edges_by_indiv.erase(std::unique(edges_by_indiv.begin(), edges_by_indiv.end()), edges_by_indiv.end());

    gi.individual_ptr.assign(static_cast<std::size_t>(num_individuals) + 1, 0);
    gi.individual_group.assign(edges_by_indiv.size(), 0);
    for (const uint64_t key : edges_by_indiv) {
        const int i = static_cast<int>(key >> 32);
        gi.individual_ptr[static_cast<std::size_t>(i + 1)] += 1;
    }
    for (int i = 0; i < num_individuals; ++i) {
        gi.individual_ptr[static_cast<std::size_t>(i + 1)] +=
            gi.individual_ptr[static_cast<std::size_t>(i)];
    }
    std::vector<int> icursor = gi.individual_ptr;
    for (const uint64_t key : edges_by_indiv) {
        const int i = static_cast<int>(key >> 32);
        const int g = static_cast<int>(key & 0xffffffffULL);
        gi.individual_group[static_cast<std::size_t>(icursor[static_cast<std::size_t>(i)]++)] = g;
    }

    gi.group_scale.assign(static_cast<std::size_t>(num_groups), 1.0);
    if (aggregation == GroupAggregation::Mean) {
        for (int g = 0; g < num_groups; ++g) {
            const int size = gi.group_ptr[static_cast<std::size_t>(g + 1)] -
                             gi.group_ptr[static_cast<std::size_t>(g)];
            if (size <= 0) {
                throw std::runtime_error("Invalid group with zero individuals");
            }
            gi.group_scale[static_cast<std::size_t>(g)] = 1.0 / static_cast<double>(size);
        }
    }
    return gi;
}

}  // namespace


// ---------------------------------------------------------------------------
// AKM + leave-out (KSS) variance decomposition (opt-in task; mirrors the
// Python py_hdfe_v11.akm_kss free function). Reached only when the caller
// passes task=akm_kss (the xhdfeakm ado); the standard xhdfe fit path below
// is untouched.
// ---------------------------------------------------------------------------

hdfe::akm::LeaveOutLevel parse_akm_leave_out_level(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "match" || lower == "matches") {
        return hdfe::akm::LeaveOutLevel::Match;
    }
    if (lower == "obs" || lower == "observation" || lower == "observations") {
        return hdfe::akm::LeaveOutLevel::Observation;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown leave-out level: " + name);
}

hdfe::akm::LeverageMethod parse_akm_leverage_method(const std::string& name) {
    const std::string lower = to_lower(name);
    if (lower == "auto") {
        return hdfe::akm::LeverageMethod::Auto;
    }
    if (lower == "exact") {
        return hdfe::akm::LeverageMethod::Exact;
    }
    if (lower == "jla" || lower == "jl") {
        return hdfe::akm::LeverageMethod::Jla;
    }
    throw_with_prefix("xhdfe plugin: ", "unknown leverages method: " + name);
}

void akm_save_scalar(const char* name, double value) {
    const ST_retcode rc = SF_scal_save(const_cast<char*>(name), value);
    if (rc) {
        throw_with_prefix("xhdfe plugin: ", std::string("failed to save scalar: ") + name);
    }
}

void akm_progress_to_stata(const char* line, void* user);

// Leave-out connected set only (task=akm_leave_out_set; the xhdfeconnected
// ado): computes the KSS leave-one-out sample flag and counts without
// running the decomposition.
ST_retcode run_akm_leave_out_set(const ParsedArgs& args) {
    bool has_fweight = false;
    if (auto val = args.get_optional("has_fweight")) {
        has_fweight = parse_bool(*val, "has_fweight");
    }
    // varlist: worker firm [fweight] keep
    const int expected_vars = 2 + (has_fweight ? 1 : 0) + 1;
    if (SF_nvars() != expected_vars) {
        throw_with_prefix("xhdfe plugin: ",
                          "varlist has wrong length (have " + std::to_string(SF_nvars()) +
                              ", want " + std::to_string(expected_vars) + ")");
    }
    const int idx_worker = 1;
    const int idx_firm = 2;
    const int idx_fw = has_fweight ? 3 : -1;
    const int idx_keep = has_fweight ? 4 : 3;

    const ObservationMap obs = selected_observations();
    const int n = obs.n;
    if (n <= 0) {
        return static_cast<ST_retcode>(2000);
    }
    const ST_double missval = SV_missval;
    Eigen::VectorXi worker(n);
    Eigen::VectorXi firm(n);
    Eigen::VectorXd fw;
    if (has_fweight) {
        fw.resize(n);
    }
    for (int i = 0; i < n; ++i) {
        const int row = obs.obs_no(i);
        ST_double z = 0.0;
        if (SF_vdata(idx_worker, row, &z) || !(z < missval)) {
            throw_with_prefix("xhdfe plugin: ", "failed to read worker id");
        }
        worker[i] = static_cast<int>(std::floor(z + 0.5));
        if (SF_vdata(idx_firm, row, &z) || !(z < missval)) {
            throw_with_prefix("xhdfe plugin: ", "failed to read firm id");
        }
        firm[i] = static_cast<int>(std::floor(z + 0.5));
        if (has_fweight) {
            if (SF_vdata(idx_fw, row, &z) || !(z < missval)) {
                throw_with_prefix("xhdfe plugin: ", "failed to read fweight");
            }
            fw[i] = z;
        }
    }

    hdfe::akm::LeaveOutSetOptions options;
    if (auto val = args.get_optional("num_threads")) {
        options.num_threads = parse_int(*val, "num_threads");
    }
    if (auto val = args.get_optional("use_gpu")) {
        options.use_gpu = parse_bool(*val, "use_gpu");
    }
    if (auto val = args.get_optional("verbose")) {
        options.verbose = parse_int(*val, "verbose");
        if (options.verbose > 0) options.progress = &akm_progress_to_stata;
    }

    const hdfe::akm::LeaveOutSetResult set = hdfe::akm::leave_out_connected_set(
        worker, firm, has_fweight ? &fw : nullptr, options);

    for (int i = 0; i < n; ++i) {
        const int row = obs.obs_no(i);
        SF_vstore(idx_keep, row,
                  set.keep[static_cast<std::size_t>(i)] != 0 ? 1.0 : 0.0);
    }
    akm_save_scalar("__xakm_n_obs", static_cast<double>(set.n_obs));
    akm_save_scalar("__xakm_n_obs_input", static_cast<double>(set.n_obs_input));
    akm_save_scalar("__xakm_n_obs_connected",
                    static_cast<double>(set.n_obs_connected));
    akm_save_scalar("__xakm_n_workers", static_cast<double>(set.n_workers));
    akm_save_scalar("__xakm_n_firms", static_cast<double>(set.n_firms));
    akm_save_scalar("__xakm_n_matches", static_cast<double>(set.n_matches));
    akm_save_scalar("__xakm_n_movers", static_cast<double>(set.n_movers));
    akm_save_scalar("__xakm_n_stayers", static_cast<double>(set.n_stayers));
    akm_save_scalar("__xakm_prune_iterations",
                    static_cast<double>(set.prune_iterations));
    akm_save_scalar("__xakm_threads_used", static_cast<double>(set.threads_used));
    akm_save_scalar("__xakm_gpu_used", set.gpu_used ? 1.0 : 0.0);
    akm_save_scalar("__xakm_gpu_status_code",
                    static_cast<double>(set.gpu_status_code));
    return 0;
}

// Verbose progress sink for the companion tasks: route each core progress
// line to the Stata Results window and force it to become visible before the
// next (potentially long) numerical phase starts.  SF_display() alone may
// leave plugin output queued until `plugin call` returns, which turns useful
// progress into an end-of-command dump.  The SPI's output flush plus GUI poll
// are output-only and this callback is invoked exclusively on Stata's calling
// thread, outside OpenMP regions.
void akm_progress_to_stata(const char* line, void* /*user*/) {
    char buf[560];
    std::snprintf(buf, sizeof buf, "%s\n", line);
    SF_display(buf);
    (_stata_)->spoutflush();
    (void)(_stata_)->pollnow();
}

ST_retcode run_akm_kss(const ParsedArgs& args) {
    const int ncontrols = parse_int(args.get_required("ncontrols"), "ncontrols");
    const bool store_effects = parse_bool(args.get_required("store_effects"), "store_effects");
    bool has_fweight = false;
    if (auto val = args.get_optional("has_fweight")) {
        has_fweight = parse_bool(*val, "has_fweight");
    }

    // varlist: y worker firm [fweight] [controls...] [alpha psi] keep
    const int expected_vars =
        3 + (has_fweight ? 1 : 0) + ncontrols + (store_effects ? 2 : 0) + 1;
    if (SF_nvars() != expected_vars) {
        throw_with_prefix("xhdfe plugin: ",
                          "varlist has wrong length (have " + std::to_string(SF_nvars()) +
                              ", want " + std::to_string(expected_vars) + ")");
    }
    const int idx_y = 1;
    const int idx_worker = 2;
    const int idx_firm = 3;
    const int idx_fw = has_fweight ? 4 : -1;
    const int idx_x_start = has_fweight ? 5 : 4;
    const int idx_alpha = store_effects ? idx_x_start + ncontrols : -1;
    const int idx_psi = store_effects ? idx_alpha + 1 : -1;
    const int idx_keep = idx_x_start + ncontrols + (store_effects ? 2 : 0);

    const ObservationMap obs = selected_observations();
    const int n = obs.n;
    if (n <= 0) {
        return static_cast<ST_retcode>(2000);
    }

    const ST_double missval = SV_missval;
    auto read_numeric = [&](int var_idx, int obs_no) {
        ST_double z = 0.0;
        const ST_retcode rc = SF_vdata(var_idx, obs_no, &z);
        if (rc) {
            throw_with_prefix("xhdfe plugin: ",
                              "failed to read data (rc=" + std::to_string(rc) + ")");
        }
        if (!(z < missval)) {
            throw_with_prefix("xhdfe plugin: ",
                              "unexpected missing value; make sure the sample excludes missings");
        }
        return z;
    };
    auto read_id = [&](int var_idx, int obs_no, const char* label) {
        const double z = read_numeric(var_idx, obs_no);
        const double r = std::floor(z + 0.5);
        if (std::abs(z - r) > 1e-6 ||
            r < static_cast<double>(std::numeric_limits<int>::min()) ||
            r > static_cast<double>(std::numeric_limits<int>::max())) {
            throw_with_prefix("xhdfe plugin: ",
                              std::string(label) + " must contain int32-range integer ids");
        }
        return static_cast<int>(r);
    };

    Eigen::VectorXd y(n);
    Eigen::VectorXi worker(n);
    Eigen::VectorXi firm(n);
    Eigen::VectorXd fw;
    if (has_fweight) {
        fw.resize(n);
    }
    Eigen::MatrixXd X;
    if (ncontrols > 0) {
        X.resize(n, ncontrols);
    }
    for (int i = 0; i < n; ++i) {
        const int row = obs.obs_no(i);
        y[i] = read_numeric(idx_y, row);
        worker[i] = read_id(idx_worker, row, "worker");
        firm[i] = read_id(idx_firm, row, "firm");
        if (has_fweight) {
            fw[i] = read_numeric(idx_fw, row);
        }
        for (int j = 0; j < ncontrols; ++j) {
            X(i, j) = read_numeric(idx_x_start + j, row);
        }
    }

    hdfe::akm::AkmOptions options;
    if (auto val = args.get_optional("leave_out_level")) {
        options.leave_out_level = parse_akm_leave_out_level(*val);
    }
    if (auto val = args.get_optional("leverages")) {
        options.leverage_method = parse_akm_leverage_method(*val);
    }
    if (auto val = args.get_optional("jla_draws")) {
        options.jla_draws = parse_int(*val, "jla_draws");
    }
    if (auto val = args.get_optional("seed")) {
        options.seed = static_cast<std::uint64_t>(parse_double(*val, "seed"));
    }
    if (auto val = args.get_optional("prune")) {
        options.prune = parse_bool(*val, "prune");
    }
    if (auto val = args.get_optional("exact_max_rows")) {
        options.exact_max_rows = parse_int(*val, "exact_max_rows");
    }
    if (auto val = args.get_optional("direct_max_firms")) {
        options.direct_max_firms = parse_int(*val, "direct_max_firms");
    }
    if (auto val = args.get_optional("direct_max_nnz")) {
        options.direct_max_nnz = static_cast<long long>(parse_double(*val, "direct_max_nnz"));
    }
    if (auto val = args.get_optional("cg_tol")) {
        options.cg_tol = parse_double(*val, "cg_tol");
    }
    if (auto val = args.get_optional("cg_max_iter")) {
        options.cg_max_iter = parse_int(*val, "cg_max_iter");
    }
    if (auto val = args.get_optional("num_threads")) {
        options.num_threads = parse_int(*val, "num_threads");
    }
    if (auto val = args.get_optional("fwl_tol")) {
        options.fwl_tol = parse_double(*val, "fwl_tol");
    }
    if (auto val = args.get_optional("fwl_max_iter")) {
        options.fwl_max_iter = parse_int(*val, "fwl_max_iter");
    }
    if (auto val = args.get_optional("compute_se")) {
        options.compute_se = parse_bool(*val, "compute_se");
    }
    if (auto val = args.get_optional("se_nsim")) {
        options.se_nsim = parse_int(*val, "se_nsim");
    }
    if (auto val = args.get_optional("verbose")) {
        options.verbose = parse_bool(*val, "verbose") ? 1 : 0;
        if (options.verbose) {
            options.progress = &akm_progress_to_stata;
        }
    }
    if (auto val = args.get_optional("eigen_diagnostics")) {
        options.eigen_diagnostics = parse_bool(*val, "eigen_diagnostics");
        if (options.eigen_diagnostics) {
            options.compute_se = true;
        }
    }
    if (auto val = args.get_optional("eig_trace_nsim")) {
        options.eig_trace_nsim = parse_int(*val, "eig_trace_nsim");
    }
    if (auto val = args.get_optional("se_sigma_lowess")) {
        options.se_sigma_lowess = parse_bool(*val, "se_sigma_lowess");
    }
    if (auto val = args.get_optional("use_gpu")) {
        options.use_gpu = parse_bool(*val, "use_gpu");
    }

    const hdfe::akm::AkmKssResult res = hdfe::akm::akm_kss_decompose(
        y, worker, firm, ncontrols > 0 ? &X : nullptr, options, nullptr,
        has_fweight ? &fw : nullptr);

    // Scalars under fixed hidden names; the ado copies them into r().
    akm_save_scalar("__xakm_plugin_var_alpha", res.plugin.var_alpha);
    akm_save_scalar("__xakm_plugin_var_psi", res.plugin.var_psi);
    akm_save_scalar("__xakm_plugin_cov", res.plugin.cov_alpha_psi);
    akm_save_scalar("__xakm_agsu_var_alpha", res.agsu.var_alpha);
    akm_save_scalar("__xakm_agsu_var_psi", res.agsu.var_psi);
    akm_save_scalar("__xakm_agsu_cov", res.agsu.cov_alpha_psi);
    akm_save_scalar("__xakm_kss_var_alpha", res.kss.var_alpha);
    akm_save_scalar("__xakm_kss_var_psi", res.kss.var_psi);
    akm_save_scalar("__xakm_kss_cov", res.kss.cov_alpha_psi);
    akm_save_scalar("__xakm_var_y", res.var_y);
    akm_save_scalar("__xakm_sigma2_ho", res.sigma2_ho);
    akm_save_scalar("__xakm_n_obs", static_cast<double>(res.sample.n_obs));
    akm_save_scalar("__xakm_n_obs_input", static_cast<double>(res.sample.n_obs_input));
    akm_save_scalar("__xakm_n_obs_connected",
                    static_cast<double>(res.sample.n_obs_connected));
    akm_save_scalar("__xakm_n_workers", static_cast<double>(res.sample.n_workers));
    akm_save_scalar("__xakm_n_firms", static_cast<double>(res.sample.n_firms));
    akm_save_scalar("__xakm_n_matches", static_cast<double>(res.sample.n_matches));
    akm_save_scalar("__xakm_n_movers", static_cast<double>(res.sample.n_movers));
    akm_save_scalar("__xakm_n_stayers", static_cast<double>(res.sample.n_stayers));
    akm_save_scalar("__xakm_n_rows", static_cast<double>(res.n_rows));
    akm_save_scalar("__xakm_max_pii", res.max_pii);
    akm_save_scalar("__xakm_mean_pii", res.mean_pii);
    akm_save_scalar("__xakm_leverages_exact", res.leverages_exact ? 1.0 : 0.0);
    akm_save_scalar("__xakm_solver_direct", res.solver_direct ? 1.0 : 0.0);
    akm_save_scalar("__xakm_fwl_threads_used", static_cast<double>(res.fwl_threads_used));
    akm_save_scalar("__xakm_threads_used", static_cast<double>(res.threads_used));
    akm_save_scalar("__xakm_jla_draws", static_cast<double>(res.jla_draws_used));
    akm_save_scalar("__xakm_seed", static_cast<double>(res.seed_used));
    akm_save_scalar("__xakm_solver_iterations", static_cast<double>(res.solver_iterations));
    akm_save_scalar("__xakm_converged", res.converged ? 1.0 : 0.0);
    akm_save_scalar("__xakm_gpu_used", res.gpu_used ? 1.0 : 0.0);
    if (options.compute_se) {
        akm_save_scalar("__xakm_se_var_psi", res.se_var_psi);
        akm_save_scalar("__xakm_se_cov", res.se_cov_alpha_psi);
        akm_save_scalar("__xakm_se_var_alpha", res.se_var_alpha);
        akm_save_scalar("__xakm_theta_var_psi", res.theta_c_var_psi);
        akm_save_scalar("__xakm_theta_cov", res.theta_c_cov_alpha_psi);
        akm_save_scalar("__xakm_theta_var_alpha", res.theta_c_var_alpha);
    }
    if (options.eigen_diagnostics) {
        static const char* comp_tag[3] = {"psi", "cov", "alpha"};
        char nm[48];
        for (int c = 0; c < 3; ++c) {
            std::snprintf(nm, sizeof(nm), "__xakm_lambda1_%s", comp_tag[c]);
            akm_save_scalar(nm, res.eig_lambda1[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_eigshare1_%s", comp_tag[c]);
            akm_save_scalar(nm, res.eig_share1[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_lindeberg_%s", comp_tag[c]);
            akm_save_scalar(nm, res.lindeberg_max_x1bar_sq[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_gammasq_%s", comp_tag[c]);
            akm_save_scalar(nm, res.gamma_sq[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_fstat_%s", comp_tag[c]);
            akm_save_scalar(nm, res.f_stat[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_theta1_%s", comp_tag[c]);
            akm_save_scalar(nm, res.theta_1[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_ci_lb_%s", comp_tag[c]);
            akm_save_scalar(nm, res.ci_lb[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_ci_ub_%s", comp_tag[c]);
            akm_save_scalar(nm, res.ci_ub[c]);
            std::snprintf(nm, sizeof(nm), "__xakm_curvature_%s", comp_tag[c]);
            akm_save_scalar(nm, res.curvature[c]);
        }
    }
    if (!res.notes.empty()) {
        SF_macro_save(const_cast<char*>("_xakm_notes"), const_cast<char*>(res.notes.c_str()));
    }

    // Control coefficients into the caller-supplied matrix (1 x k).
    if (ncontrols > 0) {
        if (auto bname = args.get_optional("b")) {
            for (int j = 0; j < res.beta.size(); ++j) {
                SF_mat_store(const_cast<char*>(bname->c_str()), 1, j + 1, res.beta[j]);
            }
        }
    }

    // keep flag over the sample rows; alpha/psi on kept rows, missing elsewhere.
    std::size_t kept_cursor = 0;
    for (int i = 0; i < n; ++i) {
        const int row = obs.obs_no(i);
        const bool kept = res.sample.keep[static_cast<std::size_t>(i)] != 0;
        SF_vstore(idx_keep, row, kept ? 1.0 : 0.0);
        if (store_effects) {
            if (kept) {
                SF_vstore(idx_alpha, row, res.alpha[static_cast<Eigen::Index>(kept_cursor)]);
                SF_vstore(idx_psi, row, res.psi[static_cast<Eigen::Index>(kept_cursor)]);
            } else {
                SF_vstore(idx_alpha, row, missval);
                SF_vstore(idx_psi, row, missval);
            }
        }
        if (kept) {
            ++kept_cursor;
        }
    }
    return 0;
}


ST_retcode run_gelbach(const ParsedArgs& args) {
    const int p = parse_int(args.get_required("p"), "p");
    const int q = parse_int(args.get_required("q"), "q");
    const int nfe = parse_int(args.get_required("nfe"), "nfe");
    const bool has_cluster = parse_bool(args.get_required("has_cluster"), "has_cluster");
    const bool has_weight = parse_bool(args.get_required("has_weight"), "has_weight");
    const bool freq_weights = has_weight &&
        to_lower(args.get_required("weight_type")).rfind("fw", 0) == 0;

    std::vector<int> sizes;
    {
        const std::string raw = args.get_required("group_sizes");
        std::string buf;
        for (char c : raw) {
            if (c == ',') { if (!buf.empty()) sizes.push_back(std::stoi(buf)); buf.clear(); }
            else buf.push_back(c);
        }
        if (!buf.empty()) sizes.push_back(std::stoi(buf));
    }

    const int expected_vars = 1 + p + q + nfe + (has_cluster ? 1 : 0) + (has_weight ? 1 : 0);
    if (SF_nvars() != expected_vars) {
        throw_with_prefix("xhdfe plugin: ",
                          "varlist has wrong length (have " + std::to_string(SF_nvars()) +
                              ", want " + std::to_string(expected_vars) + ")");
    }
    const ObservationMap obs = selected_observations();
    const int n = obs.n;
    if (n <= 0) return static_cast<ST_retcode>(2000);

    const ST_double missval = SV_missval;
    auto read_numeric = [&](int var_idx, int obs_no) {
        ST_double z = 0.0;
        if (SF_vdata(var_idx, obs_no, &z)) {
            throw_with_prefix("xhdfe plugin: ", "failed to read data");
        }
        if (!(z < missval)) {
            throw_with_prefix("xhdfe plugin: ", "unexpected missing value");
        }
        return z;
    };
    auto read_id = [&](int var_idx, int obs_no) {
        const double z = read_numeric(var_idx, obs_no);
        const double r = std::floor(z + 0.5);
        if (std::abs(z - r) > 1e-6) {
            throw_with_prefix("xhdfe plugin: ", "ids must be integers");
        }
        if (r < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
            r > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            throw_with_prefix("xhdfe plugin: ", "ids must fit in int32");
        }
        return static_cast<int>(r);
    };

    Eigen::VectorXd y(n);
    Eigen::MatrixXd X1(n, p);
    Eigen::MatrixXd X2(n, q);
    std::vector<Eigen::VectorXi> fes(static_cast<std::size_t>(nfe));
    for (auto& f : fes) f.resize(n);
    Eigen::VectorXi cl;
    if (has_cluster) cl.resize(n);
    Eigen::VectorXd wvec;
    if (has_weight) wvec.resize(n);
    for (int i = 0; i < n; ++i) {
        const int row = obs.obs_no(i);
        y[i] = read_numeric(1, row);
        for (int c = 0; c < p; ++c) X1(i, c) = read_numeric(2 + c, row);
        for (int c = 0; c < q; ++c) X2(i, c) = read_numeric(2 + p + c, row);
        for (int d = 0; d < nfe; ++d) fes[static_cast<std::size_t>(d)][i] = read_id(2 + p + q + d, row);
        if (has_cluster) cl[i] = read_id(2 + p + q + nfe, row);
        if (has_weight) wvec[i] = read_numeric(2 + p + q + nfe + (has_cluster ? 1 : 0), row);
    }

    hdfe::gelbach::GelbachOptions options;
    {
        const std::string v = to_lower(args.get_required("vce"));
        if (v == "unadjusted") options.vce = hdfe::gelbach::GelbachVce::Unadjusted;
        else if (v == "robust") options.vce = hdfe::gelbach::GelbachVce::Robust;
        else if (v == "cluster") options.vce = hdfe::gelbach::GelbachVce::Cluster;
        else throw_with_prefix("xhdfe plugin: ", "unknown vce: " + v);
    }
    options.gamma0 = parse_bool(args.get_required("gamma0"), "gamma0");
    options.cov0 = parse_bool(args.get_required("cov0"), "cov0");
    options.tol = parse_double(args.get_required("tol"), "tol");
    if (auto val = args.get_optional("num_threads")) {
        options.num_threads = parse_int(*val, "num_threads");
    }
    bool use_gpu = false;
    if (auto val = args.get_optional("use_gpu")) {
        use_gpu = parse_bool(*val, "use_gpu");
    }
    if (auto val = args.get_optional("verbose")) {
        options.verbose = parse_int(*val, "verbose");
        if (options.verbose > 0) options.progress = &akm_progress_to_stata;
    }

    // Use the same scoped backend selection as the main xhdfe task.  The
    // environment is restored on every exit path; no later command inherits
    // this request.
    std::optional<ScopedEnvVar> gpu_env;
    if (use_gpu) {
        gpu_env.emplace("XHDFE_GPU_BACKEND", "cuda");
    }

    const hdfe::gelbach::GelbachResult res = hdfe::gelbach::decompose(
        y, X1, X2, sizes, fes, has_cluster ? &cl : nullptr, options,
        has_weight ? &wvec : nullptr, freq_weights);

    const int G = static_cast<int>(res.delta.cols());
    const int k1 = p + 1;
    const std::string dmat = args.get_required("dmat");
    const std::string semat = args.get_required("semat");
    const std::string totmat = args.get_required("totmat");
    const std::string bbasemat = args.get_required("bbasemat");
    const std::string bfullmat = args.get_required("bfullmat");
    const std::string covmat = args.get_required("covmat");
    const std::string totcovmat = args.get_required("totcovmat");
    for (int g = 0; g < G; ++g) {
        for (int r = 0; r < k1; ++r) {
            SF_mat_store(const_cast<char*>(dmat.c_str()), r + 1, g + 1, res.delta(r, g));
            SF_mat_store(const_cast<char*>(semat.c_str()), r + 1, g + 1,
                         std::sqrt(res.cov(g * k1 + r, g * k1 + r)));
        }
    }
    for (int r = 0; r < k1; ++r) {
        SF_mat_store(const_cast<char*>(totmat.c_str()), r + 1, 1, res.total[r]);
        SF_mat_store(const_cast<char*>(totmat.c_str()), r + 1, 2,
                     std::sqrt(res.total_cov(r, r)));
    }
    for (int c = 0; c < p; ++c) {
        SF_mat_store(const_cast<char*>(bbasemat.c_str()), 1, c + 1, res.b_base[c]);
        SF_mat_store(const_cast<char*>(bfullmat.c_str()), 1, c + 1, res.b_full[c]);
    }
    for (int r = 0; r < G * k1; ++r) {
        for (int c = 0; c < G * k1; ++c) {
            SF_mat_store(const_cast<char*>(covmat.c_str()), r + 1, c + 1,
                         res.cov(r, c));
        }
    }
    for (int r = 0; r < k1; ++r) {
        for (int c = 0; c < k1; ++c) {
            SF_mat_store(const_cast<char*>(totcovmat.c_str()), r + 1, c + 1,
                         res.total_cov(r, c));
        }
    }
    akm_save_scalar("__xgel_identity_gap", res.identity_gap);
    akm_save_scalar("__xgel_n_obs", static_cast<double>(res.n_obs));
    akm_save_scalar("__xgel_df_full", res.df_full);
    akm_save_scalar("__xgel_converged", res.converged ? 1.0 : 0.0);
    akm_save_scalar("__xgel_threads_used", static_cast<double>(res.threads_used));
    akm_save_scalar("__xgel_gpu_used", res.gpu_used ? 1.0 : 0.0);
    akm_save_scalar("__xgel_gpu_status_code",
                    static_cast<double>(use_gpu && nfe == 0 ? 6
                                                           : res.gpu_status_code));
    akm_save_scalar("__xgel_gpu_attempted", res.gpu_attempted ? 1.0 : 0.0);
    akm_save_scalar("__xgel_gpu_absorption_converged",
                    res.gpu_absorption_converged ? 1.0 : 0.0);
    akm_save_scalar("__xgel_gpu_absorption_iterations",
                    static_cast<double>(res.gpu_absorption_iterations));
    if (!res.notes.empty()) {
        SF_macro_save(const_cast<char*>("_xgel_notes"), const_cast<char*>(res.notes.c_str()));
    }
    return 0;
}

STDLL stata_call(int argc, char* argv[]) {
    const auto plugin_total_t0 = std::chrono::steady_clock::now();
    try {
        ParsedArgs args(argc, argv);

        if (auto task = args.get_optional("task")) {
            if (*task == "akm_kss") {
                return run_akm_kss(args);
            }
            if (*task == "akm_leave_out_set") {
                return run_akm_leave_out_set(args);
            }
            if (*task == "gelbach") {
                return run_gelbach(args);
            }
            throw_with_prefix("xhdfe plugin: ", "unknown task: " + *task);
        }

        const std::string bmat = args.get_required("b");
        const std::string Vmat = args.get_required("V");

        const int p = parse_int(args.get_required("p"), "p");
        const int nfe = parse_int(args.get_required("nfe"), "nfe");
        const int nslope =
            args.has("nslope") ? parse_int(args.get_required("nslope"), "nslope") : 0;
        const int nclust = parse_int(args.get_required("nclust"), "nclust");
        const int ninst = parse_int(args.get_required("ninst"), "ninst");

        const bool has_weight = parse_bool(args.get_required("has_weight"), "has_weight");
        const bool group_mode = parse_bool(args.get_required("group_mode"), "group_mode");
        const bool has_individual = parse_bool(args.get_required("has_individual"), "has_individual");

        const bool store_resid = parse_bool(args.get_required("store_resid"), "store_resid");
        const bool store_groupvar = parse_bool(args.get_required("store_groupvar"), "store_groupvar");
        const bool store_fes = parse_bool(args.get_required("store_fes"), "store_fes");
        bool esample_prefilled = false;
        if (auto val = args.get_optional("esample_prefilled")) {
            esample_prefilled = parse_bool(*val, "esample_prefilled");
        }

        HdfeOptions opts;
        opts.se_type = parse_se_type(args.get_required("se_type"));
        opts.tol = parse_double(args.get_required("tol"), "tol");
        opts.max_iter = parse_int(args.get_required("max_iter"), "max_iter");
        opts.fit_intercept = parse_bool(args.get_required("fit_intercept"), "fit_intercept");
        opts.num_threads = parse_int(args.get_required("num_threads"), "num_threads");
        opts.drop_singletons = parse_bool(args.get_required("drop_singletons"), "drop_singletons");
        opts.retain_fixed_effects = parse_bool(args.get_required("retain_fes"), "retain_fes");
        opts.refine_stored_residuals = store_resid;
        opts.symmetric_sweep = parse_bool(args.get_required("symmetric_sweep"), "symmetric_sweep");
        opts.absorption_method = parse_absorption_method(args.get_required("absorption_method"));
        if (auto val = args.get_optional("convergence")) {
            opts.convergence_criterion = parse_convergence_criterion(*val);
        }
        opts.jacobi_relaxation = parse_double(args.get_required("jacobi_relaxation"), "jacobi_relaxation");
        opts.level = parse_double(args.get_required("level"), "level");
        opts.save_groupvar = store_groupvar;
        if (has_weight) {
            if (auto val = args.get_optional("weight_type")) {
                const std::string wt = to_lower(*val);
                if (wt == "fweight" || wt == "fw") {
                    opts.weights_are_frequencies = true;
                } else if (wt == "aweight" || wt == "aw" || wt == "pweight" || wt == "pw" ||
                           wt == "iweight" || wt == "iw") {
                    opts.weights_are_frequencies = false;
                } else if (!wt.empty()) {
                    throw_with_prefix("xhdfe plugin: ", "unknown weight_type: " + *val);
                }
            }
        }
        if (auto val = args.get_optional("fetol")) {
            opts.fe_tolerance = parse_double(*val, "fetol");
        }
        if (auto val = args.get_optional("fe_recovery_method")) {
            opts.fe_recovery_method = parse_fe_recovery_method(*val);
        }
        if (auto val = args.get_optional("tolerance_mode")) {
            opts.tolerance_mode = parse_tolerance_mode(*val);
        }

        const ParsedDofAdjustments dof = parse_dofadjustments(args.get_optional("dofadjustments"));
        opts.dof_method = dof.method;
        opts.dof_adjust_clusters = dof.adjust_clusters;
        opts.dof_adjust_continuous = dof.adjust_continuous;

        if (auto val = args.get_optional("ssc_k_adj")) {
            opts.ssc_k_adj = parse_bool(*val, "ssc_k_adj");
        }
        if (auto val = args.get_optional("ssc_k_fixef")) {
            opts.ssc_k_fixef = parse_fixef_method(*val);
        }
        if (auto val = args.get_optional("ssc_k_exact")) {
            opts.ssc_k_exact = parse_bool(*val, "ssc_k_exact");
        }
        if (auto val = args.get_optional("ssc_g_adj")) {
            opts.ssc_g_adj = parse_bool(*val, "ssc_g_adj");
        }
        if (auto val = args.get_optional("ssc_g_df")) {
            opts.ssc_g_df = parse_cluster_df(*val);
        }
        if (auto val = args.get_optional("ssc_t_df")) {
            opts.ssc_t_df = parse_double(*val, "ssc_t_df");
        }
        if (auto val = args.get_optional("statstyle")) {
            opts.stats_style = parse_stats_style(*val);
        }

        ThreadingOptions threading;
        threading.default_threads = parse_int(args.get_required("default_threads"), "default_threads");
        threading.max_threads = parse_int(args.get_required("max_threads"), "max_threads");
        threading.min_parallel_rows = parse_int(args.get_required("min_parallel_rows"), "min_parallel_rows");
        threading.target_rows_per_thread =
            parse_int(args.get_required("target_rows_per_thread"), "target_rows_per_thread");
        threading.symmetric_sweep = opts.symmetric_sweep;

        std::optional<ScopedEnvVar> gpu_env;
        if (auto val = args.get_optional("gpu_backend")) {
            const std::string backend = normalize_gpu_backend(*val);
            gpu_env.emplace("XHDFE_GPU_BACKEND", backend);
        }
        std::optional<ScopedEnvVar> mobility_profile_env;
        if (auto val = args.get_optional("mobility_profile")) {
            if (!val->empty()) {
                mobility_profile_env.emplace("XHDFE_MOBILITY_PROFILE", *val);
            }
        }
        std::optional<ScopedEnvVar> mobility_mode_env;
        if (auto val = args.get_optional("mobility_profile_mode")) {
            if (!val->empty()) {
                mobility_mode_env.emplace("XHDFE_MOBILITY_MODE", *val);
            }
        }
        std::optional<ScopedEnvVar> absorption_cache_env;
        if (auto val = args.get_optional("absorption_cache")) {
            if (!val->empty()) {
                absorption_cache_env.emplace("XHDFE_ABSORPTION_CACHE", *val);
            }
        }
        std::optional<ScopedEnvVar> absorption_cache_mode_env;
        if (auto val = args.get_optional("absorption_cache_mode")) {
            if (!val->empty()) {
                absorption_cache_mode_env.emplace("XHDFE_ABSORPTION_CACHE_MODE", *val);
            }
        }
        std::optional<ScopedEnvVar> fe_structure_cache_env;
        if (auto val = args.get_optional("fe_structure_cache")) {
            if (!val->empty()) {
                fe_structure_cache_env.emplace("XHDFE_FE_STRUCTURE_CACHE", *val);
            }
        }
        std::optional<ScopedEnvVar> fe_structure_cache_mode_env;
        if (auto val = args.get_optional("fe_structure_cache_mode")) {
            if (!val->empty()) {
                fe_structure_cache_mode_env.emplace("XHDFE_FE_STRUCTURE_MODE", *val);
            }
        }

        if (p < 0 || nfe < 0 || nslope < 0 || nclust < 0 || ninst < 0) {
            throw_with_prefix("xhdfe plugin: ", "negative dimension argument");
        }
        if (opts.se_type == StandardErrorType::Cluster && nclust <= 0) {
            throw_with_prefix("xhdfe plugin: ", "se_type=cluster requires nclust>0");
        }
        if (store_fes && !opts.retain_fixed_effects) {
            throw_with_prefix("xhdfe plugin: ", "store_fes requires retain_fes=1");
        }
        if (group_mode && ninst > 0) {
            throw_with_prefix("xhdfe plugin: ", "group()/individual() mode does not support instruments");
        }
        if (group_mode && nslope > 0) {
            throw_with_prefix("xhdfe plugin: ",
                              "group()/individual() mode does not support heterogeneous slopes");
        }

        int out_fes = store_fes ? nfe : 0;
        if (store_fes) {
            if (auto val = args.get_optional("savefe_out_count")) {
                out_fes = parse_int(*val, "savefe_out_count");
            }
            if (out_fes < 0) {
                throw_with_prefix("xhdfe plugin: ", "savefe_out_count must be nonnegative");
            }
        }
        const int nout = (store_resid ? 1 : 0) + (store_groupvar ? 1 : 0) + out_fes + 1;  // +1 esample

        const int expected_vars = 1 + p + nfe + nslope + nclust + (has_weight ? 1 : 0) + (group_mode ? 1 : 0) +
                                  (has_individual ? 1 : 0) + ninst + nout;
        const int have_vars = SF_nvars();
        if (have_vars != expected_vars) {
            throw_with_prefix(
                "xhdfe plugin: ",
                "varlist has wrong length (have " + std::to_string(have_vars) + ", want " +
                    std::to_string(expected_vars) + ")");
        }

        // varlist indices (1-based).
        const int idx_y = 1;
        const int idx_x_start = idx_y + 1;
        const int idx_fe_start = idx_x_start + p;
        const int idx_slope_start = idx_fe_start + nfe;
        const int idx_clust_start = idx_slope_start + nslope;
        const int idx_weight = idx_clust_start + nclust;  // when has_weight
        const int idx_group = idx_weight + (has_weight ? 1 : 0);
        const int idx_individual = idx_group + (group_mode ? 1 : 0);
        const int idx_inst_start = idx_individual + (has_individual ? 1 : 0);
        const int idx_out_start = idx_inst_start + ninst;

        int out_cursor = idx_out_start;
        const int idx_resid_out = store_resid ? out_cursor++ : -1;
        const int idx_groupvar_out = store_groupvar ? out_cursor++ : -1;
        const int idx_fes_out_start = store_fes ? out_cursor : -1;
        if (store_fes) {
            out_cursor += out_fes;
        }
        const int idx_esample_out = out_cursor;

        const auto select_t0 = std::chrono::steady_clock::now();
        const ObservationMap obs = selected_observations();
        plugin_cpu_profile_log_elapsed("select_observations", select_t0);
        if (plugin_cpu_profile_enabled()) {
            std::cerr << "plugin_profile label=select_observations_mode contiguous="
                      << (obs.contiguous ? 1 : 0)
                      << " n=" << obs.n << '\n';
        }
        const int n = obs.n;
        if (n <= 0) {
            return static_cast<ST_retcode>(2000);  // no observations
        }

        const ST_double missval = SV_missval;
        auto read_numeric = [&](int var_idx, int obs_no) {
            ST_double z = 0.0;
            const ST_retcode rc = SF_vdata(var_idx, obs_no, &z);
            if (rc) {
                throw_with_prefix("xhdfe plugin: ", "failed to read data (rc=" + std::to_string(rc) + ")");
            }
            if (!std::isfinite(z) || !(z < missval)) {
                throw_with_prefix("xhdfe plugin: ", "unexpected missing value; make sure sample excludes missings");
            }
            return z;
        };
        auto exact_int32_id = [](double value, int& out) {
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
        };
        auto read_categorical_numeric = [&](int var_idx, const char* what) {
            Eigen::VectorXi ids(n);
            // Most FE/cluster variables already fit int32 exactly. Fill ids
            // directly in that case and only materialize the raw double column
            // if we encounter sparse/out-of-range values that require dense coding.
            const auto read_raw_t0 = std::chrono::steady_clock::now();
            std::vector<double> raw;
            bool all_int = true;
            for (int i = 0; i < n; ++i) {
                const double z = read_numeric(var_idx, obs.obs_no(i));
                if (all_int) {
                    int id = 0;
                    if (exact_int32_id(z, id)) {
                        ids(i) = id;
                        continue;
                    }
                    all_int = false;
                    raw.resize(static_cast<std::size_t>(n));
                    for (int k = 0; k < i; ++k) {
                        raw[static_cast<std::size_t>(k)] = static_cast<double>(ids(k));
                    }
                }
                raw[static_cast<std::size_t>(i)] = z;
            }
            plugin_cpu_profile_log_elapsed(std::string("read_") + what + "_raw", read_raw_t0);
            if (all_int) {
                // Integer-valued ids in range are passed through verbatim; ids
                // were filled during the single Stata read above.
                const auto code_int_t0 = std::chrono::steady_clock::now();
                plugin_cpu_profile_log_elapsed(std::string("code_") + what + "_raw_int",
                                               code_int_t0);
                return ids;
            }

            // Sparse / out-of-range values: dense-rank by ascending value. The id
            // assigned to each observation equals the number of distinct values
            // below it, which is invariant to how the sort orders equal keys, so
            // this is bit-for-bit identical to the previous std::sort-based code.
            const auto code_dense_t0 = std::chrono::steady_clock::now();
            std::vector<std::pair<double, int>> keyed;
            keyed.resize(static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) {
                keyed[static_cast<std::size_t>(j)] =
                    std::pair<double, int>(raw[static_cast<std::size_t>(j)], j);
            }
            XHDFE_PARALLEL_SORT(keyed.begin(), keyed.end(),
                                [](const std::pair<double, int>& a,
                                   const std::pair<double, int>& b) {
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
                ids(kv.second) = next_id;
            }
            if (next_id < 0) {
                throw_with_prefix("xhdfe plugin: ",
                                  std::string("empty categorical variable: ") + what);
            }
            plugin_cpu_profile_log_elapsed(std::string("code_") + what + "_dense",
                                           code_dense_t0);
            return ids;
        };

        const auto read_total_t0 = std::chrono::steady_clock::now();
        const auto read_yx_t0 = std::chrono::steady_clock::now();
        Eigen::VectorXd y(n);
        Eigen::MatrixXd X(n, p);
        for (int i = 0; i < n; ++i) {
            y(i) = read_numeric(idx_y, obs.obs_no(i));
        }
        for (int j = 0; j < p; ++j) {
            const int var_idx = idx_x_start + j;
            for (int i = 0; i < n; ++i) {
                X(i, j) = read_numeric(var_idx, obs.obs_no(i));
            }
        }
        plugin_cpu_profile_log_elapsed("read_yx", read_yx_t0);

        const auto read_fes_t0 = std::chrono::steady_clock::now();
        std::vector<int> fe_dup_map;
        const std::optional<std::string> fe_dup_map_arg = args.get_optional("fe_dup_map");
        if (fe_dup_map_arg && !fe_dup_map_arg->empty()) {
            fe_dup_map = parse_csv_ints_ordered(*fe_dup_map_arg, "fe_dup_map");
            if (static_cast<int>(fe_dup_map.size()) != nfe) {
                throw_with_prefix("xhdfe plugin: ", "fe_dup_map length must equal nfe");
            }
        }
        std::vector<Eigen::VectorXi> fes;
        fes.reserve(static_cast<std::size_t>(nfe));
        for (int d = 0; d < nfe; ++d) {
            const int dup =
                fe_dup_map.empty() ? -1 : fe_dup_map[static_cast<std::size_t>(d)];
            if (dup >= 0) {
                if (dup >= d) {
                    throw_with_prefix("xhdfe plugin: ", "fe_dup_map entry out of range");
                }
                // Same absorb() id variable as an earlier dimension (e.g. two
                // slope terms on one FE): reuse the coded ids instead of
                // re-reading and re-coding the full column.
                fes.push_back(fes[static_cast<std::size_t>(dup)]);
                continue;
            }
            const int var_idx = idx_fe_start + d;
            Eigen::VectorXi v = read_categorical_numeric(var_idx, "fes");
            fes.push_back(std::move(v));
        }
        plugin_cpu_profile_log_elapsed("read_fes", read_fes_t0);

        std::vector<HeterogeneousSlopeTerm> slopes;
        if (nslope > 0) {
            std::vector<int> slope_fe_map =
                parse_csv_ints_ordered(args.get_required("slope_fe_map"), "slope_fe_map");
            std::vector<int> slope_intercepts =
                parse_csv_ints_ordered(args.get_required("slope_intercepts"), "slope_intercepts");
            if (static_cast<int>(slope_fe_map.size()) != nslope) {
                throw_with_prefix("xhdfe plugin: ",
                                  "slope_fe_map length must equal nslope");
            }
            if (static_cast<int>(slope_intercepts.size()) != nslope) {
                throw_with_prefix("xhdfe plugin: ",
                                  "slope_intercepts length must equal nslope");
            }
            const auto read_slopes_t0 = std::chrono::steady_clock::now();
            slopes.reserve(static_cast<std::size_t>(nslope));
            for (int d = 0; d < nslope; ++d) {
                const int mapped_fe = slope_fe_map[static_cast<std::size_t>(d)];
                if (mapped_fe < 0 || mapped_fe >= nfe) {
                    throw_with_prefix("xhdfe plugin: ",
                                      "slope_fe_map entry out of range");
                }
                const int include_intercept = slope_intercepts[static_cast<std::size_t>(d)];
                if (include_intercept != 0 && include_intercept != 1) {
                    throw_with_prefix("xhdfe plugin: ",
                                      "slope_intercepts entries must be 0 or 1");
                }
                Eigen::VectorXd values(n);
                const int var_idx = idx_slope_start + d;
                for (int i = 0; i < n; ++i) {
                    values(i) = read_numeric(var_idx, obs.obs_no(i));
                }
                HeterogeneousSlopeTerm term;
                term.fe_index = mapped_fe;
                term.include_intercept = include_intercept != 0;
                term.values = std::move(values);
                slopes.push_back(std::move(term));
            }
            plugin_cpu_profile_log_elapsed("read_slopes", read_slopes_t0);
        }

        std::optional<Eigen::VectorXd> weights;
        if (has_weight) {
            const auto read_weights_t0 = std::chrono::steady_clock::now();
            Eigen::VectorXd w(n);
            for (int i = 0; i < n; ++i) {
                w(i) = read_numeric(idx_weight, obs.obs_no(i));
            }
            weights = std::move(w);
            plugin_cpu_profile_log_elapsed("read_weights", read_weights_t0);
        }
        const Eigen::VectorXd* w_ptr = weights ? &(*weights) : nullptr;

        std::optional<std::vector<Eigen::VectorXi>> clusters;
        if (nclust > 0) {
            std::vector<int> cluster_fe_map;
            const std::optional<std::string> cluster_fe_map_arg =
                args.get_optional("cluster_fe_map");
            if (cluster_fe_map_arg && !cluster_fe_map_arg->empty()) {
                cluster_fe_map =
                    parse_csv_ints_ordered(*cluster_fe_map_arg, "cluster_fe_map");
                if (static_cast<int>(cluster_fe_map.size()) != nclust) {
                    throw_with_prefix("xhdfe plugin: ",
                                      "cluster_fe_map length must equal nclust");
                }
            }
            const auto read_clusters_t0 = std::chrono::steady_clock::now();
            std::vector<Eigen::VectorXi> c;
            c.reserve(static_cast<std::size_t>(nclust));
            for (int d = 0; d < nclust; ++d) {
                const int mapped_fe =
                    cluster_fe_map.empty() ? -1 : cluster_fe_map[static_cast<std::size_t>(d)];
                if (mapped_fe >= 0) {
                    if (mapped_fe >= nfe) {
                        throw_with_prefix("xhdfe plugin: ",
                                          "cluster_fe_map entry out of range");
                    }
                    c.push_back(fes[static_cast<std::size_t>(mapped_fe)]);
                    continue;
                }
                const int var_idx = idx_clust_start + d;
                Eigen::VectorXi v = read_categorical_numeric(var_idx, "clusters");
                c.push_back(std::move(v));
            }
            clusters = std::move(c);
            plugin_cpu_profile_log_elapsed("read_clusters", read_clusters_t0);
        }
        const std::vector<Eigen::VectorXi>* c_ptr = clusters ? &(*clusters) : nullptr;

        std::optional<Eigen::MatrixXd> instruments;
        if (ninst > 0) {
            const auto read_instruments_t0 = std::chrono::steady_clock::now();
            Eigen::MatrixXd Z(n, ninst);
            for (int j = 0; j < ninst; ++j) {
                const int var_idx = idx_inst_start + j;
                for (int i = 0; i < n; ++i) {
                    Z(i, j) = read_numeric(var_idx, obs.obs_no(i));
                }
            }
            instruments = std::move(Z);
            plugin_cpu_profile_log_elapsed("read_instruments", read_instruments_t0);
        }
        const Eigen::MatrixXd* inst_ptr = instruments ? &(*instruments) : nullptr;
        plugin_cpu_profile_log_elapsed("read_total", read_total_t0);

        std::vector<int> endogenous_idx;
        if (args.has("endogenous_idx")) {
            endogenous_idx = parse_csv_ints_0based(args.get_required("endogenous_idx"));
        }
        if (ninst > 0 && endogenous_idx.empty()) {
            throw_with_prefix("xhdfe plugin: ", "instruments require endogenous_idx");
        }
        if (!endogenous_idx.empty() && ninst == 0) {
            throw_with_prefix("xhdfe plugin: ", "endogenous_idx requires instruments");
        }

        const std::optional<std::string> s_N = args.get_optional("s_N");
        const std::optional<std::string> s_N_full = args.get_optional("s_N_full");
        const std::optional<std::string> s_num_singletons = args.get_optional("s_num_singletons");
        const std::optional<std::string> s_df_r = args.get_optional("s_df_r");
        const std::optional<std::string> s_df_r_unadj = args.get_optional("s_df_r_unadj");
        const std::optional<std::string> s_df_m = args.get_optional("s_df_m");
        const std::optional<std::string> s_df_a = args.get_optional("s_df_a");
        const std::optional<std::string> s_df_a_levels = args.get_optional("s_df_a_levels");
        const std::optional<std::string> s_df_a_exact = args.get_optional("s_df_a_exact");
        const std::optional<std::string> s_df_a_nested = args.get_optional("s_df_a_nested");
        const std::optional<std::string> s_r2 = args.get_optional("s_r2");
        const std::optional<std::string> s_r2_within = args.get_optional("s_r2_within");
        const std::optional<std::string> s_sigma2 = args.get_optional("s_sigma2");
        const std::optional<std::string> s_rss = args.get_optional("s_rss");
        const std::optional<std::string> s_tss = args.get_optional("s_tss");
        const std::optional<std::string> s_tss_within = args.get_optional("s_tss_within");
        const std::optional<std::string> s_saturated = args.get_optional("s_saturated");
        const std::optional<std::string> s_iterations = args.get_optional("s_iterations");
        const std::optional<std::string> s_converged = args.get_optional("s_converged");
        const std::optional<std::string> s_fe_recovery_converged =
            args.get_optional("s_fe_recovery_converged");
        const std::optional<std::string> s_fe_recovery_iterations =
            args.get_optional("s_fe_recovery_iterations");
        const std::optional<std::string> s_fe_recovery_max_delta =
            args.get_optional("s_fe_recovery_max_delta");
        const std::optional<std::string> s_threads_used = args.get_optional("s_threads_used");
        const std::optional<std::string> s_gpu_used = args.get_optional("s_gpu_used");
        const std::optional<std::string> s_gpu_status_code = args.get_optional("s_gpu_status_code");
        const std::optional<std::string> s_gpu_attempted = args.get_optional("s_gpu_attempted");
        const std::optional<std::string> s_gpu_absorption_converged =
            args.get_optional("s_gpu_absorption_converged");
        const std::optional<std::string> s_gpu_absorption_iterations =
            args.get_optional("s_gpu_absorption_iterations");
        const std::optional<std::string> s_method_used = args.get_optional("s_method_used");
        const std::optional<std::string> s_num_clusters = args.get_optional("s_num_clusters");
        const std::optional<std::string> s_cluster_scale = args.get_optional("s_cluster_scale");
        const std::optional<std::string> s_vcv_psd_fixed = args.get_optional("s_vcv_psd_fixed");
        const std::optional<std::string> cluster_diag = args.get_optional("cluster_diag");
        const std::optional<std::string> dof_table = args.get_optional("dof_table");
        const std::optional<std::string> omit_reason = args.get_optional("omit_reason");
        const std::optional<std::string> omit_priority = args.get_optional("omit_priority");

        if (omit_priority && !omit_priority->empty()) {
            opts.collinear_priority = read_matrix_vector_int(*omit_priority, "omit_priority");
        }
        if (!opts.collinear_priority.empty() &&
            static_cast<int>(opts.collinear_priority.size()) < p) {
            throw_with_prefix("xhdfe plugin: ", "omit_priority length is smaller than p");
        }
        HdfeRegressorV11 reg(opts, threading);

        // Group/individual mode: estimate on collapsed group observations and map results back to a
        // representative row per group (first occurrence).
        if (group_mode) {
            if (opts.retain_fixed_effects && has_individual) {
                throw_with_prefix("xhdfe plugin: ",
                                  "retain_fes is not supported with group/individual fixed effects");
            }
            if (store_fes && has_individual) {
                throw_with_prefix("xhdfe plugin: ",
                                  "store_fes is not supported with group/individual fixed effects");
            }

            Eigen::VectorXi group_ids(n);
            for (int i = 0; i < n; ++i) {
                group_ids(i) =
                    to_int_checked(read_numeric(idx_group, obs.obs_no(i)), "group");
            }

            // Group-only mode: collapse to one row per group and run the standard estimator.
            if (!has_individual) {
                std::unordered_map<int, int> group_map;
                group_map.reserve(static_cast<std::size_t>(n));
                std::vector<int> rep_row;
                rep_row.reserve(static_cast<std::size_t>(n));
                std::vector<int> group_index(static_cast<std::size_t>(n), -1);
                for (int i = 0; i < n; ++i) {
                    const int raw = group_ids(i);
                    auto it = group_map.find(raw);
                    if (it == group_map.end()) {
                        const int g = static_cast<int>(rep_row.size());
                        group_map.emplace(raw, g);
                        rep_row.push_back(i);
                        group_index[static_cast<std::size_t>(i)] = g;
                    } else {
                        group_index[static_cast<std::size_t>(i)] = it->second;
                    }
                }
                const int G = static_cast<int>(rep_row.size());
                if (G <= 0) {
                    throw_with_prefix("xhdfe plugin: ", "no groups found");
                }

                // Verify group-constant requirements (same checks as the C++ backend).
                for (int i = 0; i < n; ++i) {
                    const int g = group_index[static_cast<std::size_t>(i)];
                    const int r = rep_row[static_cast<std::size_t>(g)];
                    if (!approx_equal(y(i), y(r))) {
                        throw_with_prefix("xhdfe plugin: ", "y is not constant within group()");
                    }
                    for (int j = 0; j < p; ++j) {
                        if (!approx_equal(X(i, j), X(r, j))) {
                            throw_with_prefix("xhdfe plugin: ", "X is not constant within group()");
                        }
                    }
                    if (weights && !approx_equal((*weights)(i), (*weights)(r))) {
                        throw_with_prefix("xhdfe plugin: ", "weights are not constant within group()");
                    }
                    if (clusters) {
                        for (const auto& cvec : *clusters) {
                            if (cvec(i) != cvec(r)) {
                                throw_with_prefix("xhdfe plugin: ", "clusters are not constant within group()");
                            }
                        }
                    }
                    for (int d = 0; d < nfe; ++d) {
                        if (fes[static_cast<std::size_t>(d)](i) != fes[static_cast<std::size_t>(d)](r)) {
                            throw_with_prefix("xhdfe plugin: ", "fixed effects are not constant within group()");
                        }
                    }
                }

                Eigen::VectorXd y_g(G);
                Eigen::MatrixXd X_g(G, p);
                std::vector<Eigen::VectorXi> fes_g;
                fes_g.reserve(static_cast<std::size_t>(nfe));
                for (int d = 0; d < nfe; ++d) {
                    fes_g.emplace_back(Eigen::VectorXi::Zero(G));
                }
                std::optional<Eigen::VectorXd> w_g;
                if (weights) {
                    w_g = Eigen::VectorXd::Zero(G);
                }
                std::optional<std::vector<Eigen::VectorXi>> c_g;
                if (clusters) {
                    c_g = std::vector<Eigen::VectorXi>{};
                    c_g->reserve(clusters->size());
                    for (std::size_t j = 0; j < clusters->size(); ++j) {
                        c_g->emplace_back(Eigen::VectorXi::Zero(G));
                    }
                }

                std::vector<int> rep_obs(static_cast<std::size_t>(G), 0);
                for (int g = 0; g < G; ++g) {
                    const int r = rep_row[static_cast<std::size_t>(g)];
                    rep_obs[static_cast<std::size_t>(g)] = obs.obs_no(r);
                    y_g(g) = y(r);
                    X_g.row(g) = X.row(r);
                    for (int d = 0; d < nfe; ++d) {
                        fes_g[static_cast<std::size_t>(d)](g) = fes[static_cast<std::size_t>(d)](r);
                    }
                    if (w_g) {
                        (*w_g)(g) = (*weights)(r);
                    }
                    if (c_g) {
                        for (std::size_t j = 0; j < c_g->size(); ++j) {
                            (*c_g)[j](g) = (*clusters)[j](r);
                        }
                    }
                }

                const Eigen::VectorXd* wptr_g = w_g ? &(*w_g) : nullptr;
                const std::vector<Eigen::VectorXi>* cptr_g = c_g ? &(*c_g) : nullptr;
                reg.fit(y_g, X_g, fes_g, wptr_g, cptr_g, nullptr, {});

                const hdfe::HdfeResults& r = reg.results();

                store_row_vector(bmat, r.coefficients);
                store_matrix(Vmat, r.covariance);

                maybe_save_scalar(s_N, r.nobs_effective);
                maybe_save_scalar(s_N_full, r.nobs_full_effective);
                maybe_save_scalar(s_num_singletons, r.num_singletons_effective);
                maybe_save_scalar(s_df_r, r.df_resid);
                maybe_save_scalar(s_df_r_unadj, r.df_resid_unadj);
                maybe_save_scalar(s_df_m, r.df_m);
                maybe_save_scalar(s_df_a, r.df_a);
                maybe_save_scalar(s_df_a_levels, r.df_a_levels);
                maybe_save_scalar(s_df_a_exact, r.df_a_exact);
                maybe_save_scalar(s_df_a_nested, r.df_a_nested);
                maybe_save_scalar(s_r2, r.r2);
                maybe_save_scalar(s_r2_within, r.r2_within);
                maybe_save_scalar(s_sigma2, r.sigma2);
                maybe_save_scalar(s_rss, r.rss);
                maybe_save_scalar(s_tss, r.tss);
                maybe_save_scalar(s_tss_within, r.tss_within);
                maybe_save_scalar(s_saturated, r.is_saturated() ? 1.0 : 0.0);
                maybe_save_scalar(s_iterations, static_cast<double>(r.num_iterations));
                maybe_save_scalar(s_converged, r.converged ? 1.0 : 0.0);
                maybe_save_scalar(s_fe_recovery_converged, r.fe_recovery_converged ? 1.0 : 0.0);
                maybe_save_scalar(s_fe_recovery_iterations, static_cast<double>(r.fe_recovery_iterations));
                maybe_save_scalar(s_fe_recovery_max_delta, r.fe_recovery_max_delta);
                maybe_save_scalar(s_threads_used, static_cast<double>(reg.threads_used()));
                maybe_save_scalar(s_gpu_used, reg.gpu_used() ? 1.0 : 0.0);
                maybe_save_gpu_diagnostics(s_gpu_status_code, s_gpu_attempted,
                                           s_gpu_absorption_converged,
                                           s_gpu_absorption_iterations, reg);
                maybe_save_scalar(s_method_used, static_cast<double>(static_cast<int>(reg.absorption_method_used())));
                maybe_save_scalar(s_num_clusters, static_cast<double>(r.num_clusters));
                maybe_save_scalar(s_cluster_scale, r.cluster_scale);
                maybe_save_scalar(s_vcv_psd_fixed, r.vcv_psd_fixed ? 1.0 : 0.0);
                maybe_store_cluster_diag(cluster_diag, r);
                maybe_store_dof_table(dof_table, r);
                maybe_store_omit_reason(omit_reason, r);

                if (r.sample_index.size() != r.nobs) {
                    throw_with_prefix("xhdfe plugin: ", "unexpected sample_index size");
                }
                std::vector<const double*> fe_ptrs;
                if (store_fes) {
                    if (static_cast<int>(r.fe_effects.size()) != nfe) {
                        throw_with_prefix("xhdfe plugin: ", "unexpected fe_effects size");
                    }
                    fe_ptrs.reserve(static_cast<std::size_t>(nfe));
                    for (int d = 0; d < nfe; ++d) {
                        fe_ptrs.push_back(r.fe_effects[static_cast<std::size_t>(d)].data());
                    }
                }

                if (store_fes) {
                    std::ostringstream oss;
                    oss << "event=plugin_writeback_begin mode=group_collapsed nobs=" << r.nobs
                        << " nfe=" << nfe;
                    plugin_savefe_profile_log(oss.str());
                }
                std::optional<PluginSavefeTimer> writeback_timer;
                if (store_fes) {
                    writeback_timer.emplace("plugin_writeback_group_collapsed");
                }
                for (int i = 0; i < r.nobs; ++i) {
                    const int gi = r.sample_index(i);
                    if (gi < 0 || gi >= G) {
                        throw_with_prefix("xhdfe plugin: ", "invalid sample_index in grouped regression");
                    }
                    const int obs_no = rep_obs[static_cast<std::size_t>(gi)];
                    if (store_resid) {
                        const ST_retcode rc =
                            SF_vstore(idx_resid_out, obs_no, r.residuals(i));
                        if (rc) {
                            throw_with_prefix("xhdfe plugin: ", "failed to store residuals");
                        }
                    }
                    if (store_groupvar) {
                        const double gv = (r.groupvar.size() == r.nobs)
                                              ? static_cast<double>(r.groupvar(i) + 1)
                                              : SV_missval;
                        const ST_retcode rc = SF_vstore(idx_groupvar_out, obs_no, gv);
                        if (rc) {
                            throw_with_prefix("xhdfe plugin: ", "failed to store groupvar");
                        }
                    }
                    if (store_fes) {
                        for (int d = 0; d < nfe; ++d) {
                            const ST_retcode rc = SF_vstore(idx_fes_out_start + d, obs_no,
                                                            fe_ptrs[static_cast<std::size_t>(d)][i]);
                            if (rc) {
                                throw_with_prefix("xhdfe plugin: ", "failed to store fe_effects");
                            }
                        }
                    }
                    const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 1.0);
                    if (rc) {
                        throw_with_prefix("xhdfe plugin: ", "failed to store e(sample)");
                    }
                }

                return static_cast<ST_retcode>(0);
            }

            // Full group/individual mode.
            Eigen::VectorXi individual_ids(n);
            for (int i = 0; i < n; ++i) {
                individual_ids(i) = to_int_checked(
                    read_numeric(idx_individual, obs.obs_no(i)), "individual");
            }

            const GroupAggregation agg = parse_aggregation(args.get_required("aggregation"));

            reg.fit_grouped(y, X, fes, group_ids, &individual_ids, agg, w_ptr, c_ptr);
            const hdfe::HdfeResults& r = reg.results();

            store_row_vector(bmat, r.coefficients);
            store_matrix(Vmat, r.covariance);

            maybe_save_scalar(s_N, r.nobs_effective);
            maybe_save_scalar(s_N_full, r.nobs_full_effective);
            maybe_save_scalar(s_num_singletons, r.num_singletons_effective);
            maybe_save_scalar(s_df_r, r.df_resid);
            maybe_save_scalar(s_df_r_unadj, r.df_resid_unadj);
            maybe_save_scalar(s_df_m, r.df_m);
            maybe_save_scalar(s_df_a, r.df_a);
            maybe_save_scalar(s_df_a_levels, r.df_a_levels);
            maybe_save_scalar(s_df_a_exact, r.df_a_exact);
            maybe_save_scalar(s_df_a_nested, r.df_a_nested);
            maybe_save_scalar(s_r2, r.r2);
            maybe_save_scalar(s_r2_within, r.r2_within);
            maybe_save_scalar(s_sigma2, r.sigma2);
            maybe_save_scalar(s_rss, r.rss);
            maybe_save_scalar(s_tss, r.tss);
            maybe_save_scalar(s_tss_within, r.tss_within);
            maybe_save_scalar(s_saturated, r.is_saturated() ? 1.0 : 0.0);
            maybe_save_scalar(s_iterations, static_cast<double>(r.num_iterations));
            maybe_save_scalar(s_converged, r.converged ? 1.0 : 0.0);
            maybe_save_scalar(s_fe_recovery_converged, r.fe_recovery_converged ? 1.0 : 0.0);
            maybe_save_scalar(s_fe_recovery_iterations, static_cast<double>(r.fe_recovery_iterations));
            maybe_save_scalar(s_fe_recovery_max_delta, r.fe_recovery_max_delta);
            maybe_save_scalar(s_threads_used, static_cast<double>(reg.threads_used()));
            maybe_save_scalar(s_gpu_used, reg.gpu_used() ? 1.0 : 0.0);
            maybe_save_gpu_diagnostics(s_gpu_status_code, s_gpu_attempted,
                                       s_gpu_absorption_converged,
                                       s_gpu_absorption_iterations, reg);
            maybe_save_scalar(s_method_used, static_cast<double>(static_cast<int>(reg.absorption_method_used())));
            maybe_save_scalar(s_num_clusters, static_cast<double>(r.num_clusters));
            maybe_save_scalar(s_cluster_scale, r.cluster_scale);
            maybe_save_scalar(s_vcv_psd_fixed, r.vcv_psd_fixed ? 1.0 : 0.0);
            maybe_store_cluster_diag(cluster_diag, r);
            maybe_store_dof_table(dof_table, r);
            maybe_store_omit_reason(omit_reason, r);

            if (r.sample_index.size() != r.nobs) {
                throw_with_prefix("xhdfe plugin: ", "unexpected sample_index size in grouped regression");
            }
            for (int i = 0; i < r.nobs; ++i) {
                const int orig = r.sample_index(i);
                if (orig < 0 || orig >= n) {
                    throw_with_prefix("xhdfe plugin: ", "invalid sample_index in grouped regression");
                }
                const int obs_no = obs.obs_no(orig);
                if (store_resid) {
                    const ST_retcode rc =
                        SF_vstore(idx_resid_out, obs_no, r.residuals(i));
                    if (rc) {
                        throw_with_prefix("xhdfe plugin: ", "failed to store residuals");
                    }
                }
                if (store_groupvar) {
                    const double gv = (r.groupvar.size() == r.nobs)
                                          ? static_cast<double>(r.groupvar(i) + 1)
                                          : SV_missval;
                    const ST_retcode rc = SF_vstore(idx_groupvar_out, obs_no, gv);
                    if (rc) {
                        throw_with_prefix("xhdfe plugin: ", "failed to store groupvar");
                    }
                }
                const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 1.0);
                if (rc) {
                    throw_with_prefix("xhdfe plugin: ", "failed to store e(sample)");
                }
            }
            return static_cast<ST_retcode>(0);
        }

        // Standard HDFE mode (observation-level).
        const auto fit_standard_t0 = std::chrono::steady_clock::now();
        reg.fit(y, X, fes, w_ptr, c_ptr, inst_ptr, endogenous_idx,
                slopes.empty() ? nullptr : &slopes);
        plugin_cpu_profile_log_elapsed("fit_standard", fit_standard_t0);
        const hdfe::HdfeResults& r = reg.results();

        store_row_vector(bmat, r.coefficients);
        store_matrix(Vmat, r.covariance);

        maybe_save_scalar(s_N, r.nobs_effective);
        maybe_save_scalar(s_N_full, r.nobs_full_effective);
        maybe_save_scalar(s_num_singletons, r.num_singletons_effective);
        maybe_save_scalar(s_df_r, r.df_resid);
        maybe_save_scalar(s_df_r_unadj, r.df_resid_unadj);
        maybe_save_scalar(s_df_m, r.df_m);
        maybe_save_scalar(s_df_a, r.df_a);
        maybe_save_scalar(s_df_a_levels, r.df_a_levels);
        maybe_save_scalar(s_df_a_exact, r.df_a_exact);
        maybe_save_scalar(s_df_a_nested, r.df_a_nested);
        maybe_save_scalar(s_r2, r.r2);
        maybe_save_scalar(s_r2_within, r.r2_within);
        maybe_save_scalar(s_sigma2, r.sigma2);
        maybe_save_scalar(s_rss, r.rss);
        maybe_save_scalar(s_tss, r.tss);
        maybe_save_scalar(s_tss_within, r.tss_within);
        maybe_save_scalar(s_saturated, r.is_saturated() ? 1.0 : 0.0);
        maybe_save_scalar(s_iterations, static_cast<double>(r.num_iterations));
        maybe_save_scalar(s_converged, r.converged ? 1.0 : 0.0);
        maybe_save_scalar(s_fe_recovery_converged, r.fe_recovery_converged ? 1.0 : 0.0);
        maybe_save_scalar(s_fe_recovery_iterations, static_cast<double>(r.fe_recovery_iterations));
        maybe_save_scalar(s_fe_recovery_max_delta, r.fe_recovery_max_delta);
        maybe_save_scalar(s_threads_used, static_cast<double>(reg.threads_used()));
        maybe_save_scalar(s_gpu_used, reg.gpu_used() ? 1.0 : 0.0);
        maybe_save_gpu_diagnostics(s_gpu_status_code, s_gpu_attempted,
                                   s_gpu_absorption_converged,
                                   s_gpu_absorption_iterations, reg);
        maybe_save_scalar(s_method_used, static_cast<double>(static_cast<int>(reg.absorption_method_used())));
        maybe_save_scalar(s_num_clusters, static_cast<double>(r.num_clusters));
        maybe_save_scalar(s_cluster_scale, r.cluster_scale);
        maybe_save_scalar(s_vcv_psd_fixed, r.vcv_psd_fixed ? 1.0 : 0.0);
        maybe_store_cluster_diag(cluster_diag, r);
        maybe_store_dof_table(dof_table, r);
        maybe_store_omit_reason(omit_reason, r);

        if (r.sample_index.size() != r.nobs) {
            throw_with_prefix("xhdfe plugin: ", "unexpected sample_index size");
        }

        std::vector<uint8_t> keep;
        if (esample_prefilled && r.nobs < n) {
            keep.assign(static_cast<std::size_t>(n), 0);
        }

        if (store_fes) {
            std::ostringstream oss;
            oss << "event=plugin_writeback_begin mode=standard nobs=" << r.nobs
                << " nfe=" << nfe
                << " out_fes=" << out_fes;
            plugin_savefe_profile_log(oss.str());
        }
        std::optional<PluginSavefeTimer> writeback_timer;
        if (store_fes) {
            writeback_timer.emplace("plugin_writeback_standard");
        }
        const auto writeback_standard_t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < r.nobs; ++i) {
            const int orig = r.sample_index(i);
            if (orig < 0 || orig >= n) {
                throw_with_prefix("xhdfe plugin: ", "invalid sample_index");
            }
            if (!keep.empty()) {
                keep[static_cast<std::size_t>(orig)] = 1;
            }
            const int obs_no = obs.obs_no(orig);
            if (store_resid) {
                const ST_retcode rc =
                    SF_vstore(idx_resid_out, obs_no, r.residuals(i));
                if (rc) {
                    throw_with_prefix("xhdfe plugin: ", "failed to store residuals");
                }
            }
            if (store_groupvar) {
                const double gv = (r.groupvar.size() == r.nobs)
                                      ? static_cast<double>(r.groupvar(i) + 1)
                                      : SV_missval;
                const ST_retcode rc = SF_vstore(idx_groupvar_out, obs_no, gv);
                if (rc) {
                    throw_with_prefix("xhdfe plugin: ", "failed to store groupvar");
                }
            }
            if (store_fes) {
                const std::vector<Eigen::VectorXd>& fe_write =
                    r.fe_save_effects.empty() ? r.fe_effects : r.fe_save_effects;
                if (static_cast<int>(fe_write.size()) != out_fes) {
                    throw_with_prefix("xhdfe plugin: ", "unexpected saved FE effect count");
                }
                for (int d = 0; d < out_fes; ++d) {
                    const ST_retcode rc = SF_vstore(idx_fes_out_start + d, obs_no,
                                                    fe_write[static_cast<std::size_t>(d)](i));
                    if (rc) {
                        throw_with_prefix("xhdfe plugin: ", "failed to store saved FE effects");
                    }
                }
            }
            if (!esample_prefilled) {
                const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 1.0);
                if (rc) {
                    throw_with_prefix("xhdfe plugin: ", "failed to store e(sample)");
                }
            }
        }

        // When the ado prefills e(sample)=1 for the candidate sample, clear dropped observations
        // only. This avoids O(nobs) SF_vstore calls on large panels when singletons are dropped.
        if (!keep.empty()) {
            for (int i = 0; i < n; ++i) {
                if (keep[static_cast<std::size_t>(i)]) {
                    continue;
                }
                const int obs_no = obs.obs_no(i);
                const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 0.0);
                if (rc) {
                    throw_with_prefix("xhdfe plugin: ", "failed to clear e(sample)");
                }
            }
        }
        plugin_cpu_profile_log_elapsed("writeback_standard", writeback_standard_t0);
        plugin_cpu_profile_log_elapsed("plugin_total", plugin_total_t0);

        return static_cast<ST_retcode>(0);
    } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("No observations left after dropping singleton observations") !=
            std::string::npos) {
            return static_cast<ST_retcode>(2000);
        }
        if (msg.empty() || msg.back() != '\n') {
            msg.push_back('\n');
        }
        SF_error(const_cast<char*>(msg.c_str()));
        return static_cast<ST_retcode>(198);
    } catch (...) {
        const char* msg = "xhdfe plugin: unknown error\n";
        SF_error(const_cast<char*>(msg));
        return static_cast<ST_retcode>(198);
    }
}
