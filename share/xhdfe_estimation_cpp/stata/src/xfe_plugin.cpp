#include "stplugin.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
using hdfe::DofAdjustmentMethod;
using hdfe::HdfeOptions;
using hdfe::StandardErrorType;
using hdfe::v11::HdfeRegressorV11;
using hdfe::v11::ThreadingOptions;

constexpr const char* kPluginPrefix = "xfe plugin: ";
constexpr double kGroupConstantTol = 1e-12;

[[noreturn]] void throw_with_prefix(const std::string& msg) {
    throw std::runtime_error(std::string(kPluginPrefix) + msg);
}

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
    throw_with_prefix("invalid gpu_backend: " + raw);
}

void set_env_value(const std::string& key, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), value.c_str()) != 0) {
        throw_with_prefix("failed to set environment variable " + key);
    }
#else
    if (setenv(key.c_str(), value.c_str(), 1) != 0) {
        throw_with_prefix("failed to set environment variable " + key);
    }
#endif
}

void unset_env_value(const std::string& key) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), "") != 0) {
        throw_with_prefix("failed to unset environment variable " + key);
    }
#else
    if (unsetenv(key.c_str()) != 0) {
        throw_with_prefix("failed to unset environment variable " + key);
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

bool approx_equal(double a, double b) {
    const double scale = 1.0 + std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= kGroupConstantTol * scale;
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
                throw_with_prefix("invalid argument (expected key=value): " + item);
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

    const std::string& get_required(const char* key) const {
        auto it = kv.find(key);
        if (it == kv.end()) {
            throw_with_prefix(std::string("missing required argument: ") + key);
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
    throw_with_prefix(std::string("invalid boolean for ") + what + ": " + raw);
}

int parse_int(const std::string& raw, const char* what) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0') {
        throw_with_prefix(std::string("invalid integer for ") + what + ": " + raw);
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        throw_with_prefix(std::string("integer out of range for ") + what + ": " + raw);
    }
    return static_cast<int>(v);
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

double parse_double(const std::string& raw, const char* what) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(raw.c_str(), &end);
    if (errno != 0 || end == raw.c_str() || *end != '\0') {
        throw_with_prefix(std::string("invalid numeric for ") + what + ": " + raw);
    }
    return v;
}

AbsorptionMethod parse_absorption_method(const std::string& raw) {
    const std::string name = to_lower(raw);
    if (name == "auto") {
        return AbsorptionMethod::Auto;
    }
    if (name == "gs" || name == "gauss-seidel" || name == "gauss_seidel") {
        return AbsorptionMethod::GaussSeidel;
    }
    if (name == "sym" || name == "symmetric" || name == "symgs" ||
        name == "symmetric-gauss-seidel" || name == "symmetric_gauss_seidel") {
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
    throw_with_prefix("unknown absorption_method: " + raw);
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
        throw_with_prefix("unknown dofadjustments token: " + tok);
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

void maybe_save_scalar(const std::optional<std::string>& name, double value) {
    if (!name || name->empty()) {
        return;
    }
    const ST_retcode rc = SF_scal_save(const_cast<char*>(name->c_str()), value);
    if (rc) {
        throw_with_prefix("failed to save scalar: " + *name);
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
    maybe_save_scalar(s_gpu_absorption_converged, gpu_absorption_converged);
    maybe_save_scalar(s_gpu_absorption_iterations, gpu_absorption_iterations);
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
    const int want_cols = 5;
    const int have_rows = SF_row(const_cast<char*>(name->c_str()));
    const int have_cols = SF_col(const_cast<char*>(name->c_str()));
    if (have_rows != rows || have_cols != want_cols) {
        throw_with_prefix(
            "dof_table has wrong dimensions (have " + std::to_string(have_rows) + "x" +
            std::to_string(have_cols) + ", want " + std::to_string(rows) + "x" +
            std::to_string(want_cols) + ")");
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
        if (rc1 || rc2 || rc3 || rc4 || rc5) {
            throw_with_prefix("failed to write matrix dof_table");
        }
    }
}

int to_int_checked(double value, const char* what) {
    const double rounded = std::round(value);
    if (!approx_equal(value, rounded)) {
        throw_with_prefix(std::string("non-integer value in ") + what);
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        throw_with_prefix(std::string("integer out of range in ") + what);
    }
    return static_cast<int>(rounded);
}

std::vector<int> selected_observations() {
    const int in1 = SF_in1();
    const int in2 = SF_in2();
    if (in2 < in1) {
        return {};
    }
    std::vector<int> obs;
    obs.reserve(static_cast<std::size_t>(in2 - in1 + 1));
    for (int j = in1; j <= in2; ++j) {
        if (SF_ifobs(j)) {
            obs.push_back(j);
        }
    }
    return obs;
}

}  // namespace

STDLL stata_call(int argc, char* argv[]) {
    try {
        ParsedArgs args(argc, argv);

        const int k = parse_int(args.get_required("k"), "k");
        const int nfe = parse_int(args.get_required("nfe"), "nfe");
        const int nclust = parse_int(args.get_required("nclust"), "nclust");
        const bool has_weight = parse_bool(args.get_required("has_weight"), "has_weight");

        const bool store_groupvar = parse_bool(args.get_required("store_groupvar"), "store_groupvar");
        bool esample_prefilled = false;
        if (auto val = args.get_optional("esample_prefilled")) {
            esample_prefilled = parse_bool(*val, "esample_prefilled");
        }

        HdfeOptions opts;
        opts.se_type = (nclust > 0) ? StandardErrorType::Cluster : StandardErrorType::Homoskedastic;
        opts.tol = parse_double(args.get_required("tol"), "tol");
        opts.max_iter = parse_int(args.get_required("max_iter"), "max_iter");
        opts.num_threads = parse_int(args.get_required("num_threads"), "num_threads");
        opts.drop_singletons = parse_bool(args.get_required("drop_singletons"), "drop_singletons");
        opts.symmetric_sweep = parse_bool(args.get_required("symmetric_sweep"), "symmetric_sweep");
        opts.absorption_method = parse_absorption_method(args.get_required("absorption_method"));
        opts.jacobi_relaxation = parse_double(args.get_required("jacobi_relaxation"), "jacobi_relaxation");
        opts.save_groupvar = store_groupvar;
        opts.retain_fixed_effects = false;
        if (has_weight) {
            if (auto val = args.get_optional("weight_type")) {
                const std::string wt = to_lower(*val);
                if (wt == "fweight" || wt == "fw") {
                    opts.weights_are_frequencies = true;
                } else if (wt == "aweight" || wt == "aw" || wt == "pweight" || wt == "pw" ||
                           wt == "iweight" || wt == "iw") {
                    opts.weights_are_frequencies = false;
                } else if (!wt.empty()) {
                    throw_with_prefix("unknown weight_type: " + *val);
                }
            }
        }

        const ParsedDofAdjustments dof = parse_dofadjustments(args.get_optional("dofadjustments"));
        opts.dof_method = dof.method;
        opts.dof_adjust_clusters = dof.adjust_clusters;
        opts.dof_adjust_continuous = dof.adjust_continuous;

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

        if (k <= 0 || nfe < 0 || nclust < 0) {
            throw_with_prefix("invalid xfe dimension arguments");
        }

        // varlist layout:
        //   [1..k]        input vars to partial out (numeric)
        //   [..]          fixed-effect ids (nfe)
        //   [..]          cluster ids (nclust)
        //   [..]          weight (optional)
        //   [..]          output vars (k)
        //   [..]          groupvar (optional)
        //   [last]        esample (always)
        const int expected_vars =
            k + nfe + nclust + (has_weight ? 1 : 0) + k + (store_groupvar ? 1 : 0) + 1;
        const int have_vars = SF_nvars();
        if (have_vars != expected_vars) {
            throw_with_prefix("xfe varlist has wrong length (have " + std::to_string(have_vars) +
                              ", want " + std::to_string(expected_vars) + ")");
        }

        const int idx_in_start = 1;
        const int idx_fe_start = idx_in_start + k;
        const int idx_clust_start = idx_fe_start + nfe;
        const int idx_weight = idx_clust_start + nclust;
        const int idx_out_start = idx_weight + (has_weight ? 1 : 0);
        int cursor = idx_out_start + k;
        const int idx_groupvar_out = store_groupvar ? cursor++ : -1;
        const int idx_esample_out = cursor;

        const std::vector<int> obs = selected_observations();
        const int n = static_cast<int>(obs.size());
        if (n <= 0) {
            return static_cast<ST_retcode>(2000);  // no observations
        }

        const ST_double missval = SV_missval;
        auto read_numeric = [&](int var_idx, int obs_no) {
            ST_double z = 0.0;
            const ST_retcode rc = SF_vdata(var_idx, obs_no, &z);
            if (rc) {
                throw_with_prefix("failed to read data (rc=" + std::to_string(rc) + ")");
            }
            if (!std::isfinite(z) || !(z < missval)) {
                throw_with_prefix("unexpected missing value; make sure sample excludes missings");
            }
            return z;
        };
        auto read_categorical_numeric = [&](int var_idx, const char* what) {
            Eigen::VectorXi ids(n);
            // Read the whole column once (the slow per-cell SF_vdata path is the
            // dominant cost here), tracking whether every value is exactly an int
            // in range. The previous code re-read the entire column a second time
            // whenever a single out-of-range value (e.g. a large worker id stored
            // as double) was encountered.
            std::vector<double> raw(static_cast<std::size_t>(n));
            bool all_int = true;
            for (int i = 0; i < n; ++i) {
                const double z = read_numeric(var_idx, obs[static_cast<std::size_t>(i)]);
                raw[static_cast<std::size_t>(i)] = z;
                if (all_int) {
                    const bool fits =
                        z >= static_cast<double>(std::numeric_limits<int>::min()) &&
                        z <= static_cast<double>(std::numeric_limits<int>::max()) &&
                        static_cast<double>(static_cast<int>(z)) == z;
                    if (!fits) {
                        all_int = false;
                    } else {
                        ids(i) = static_cast<int>(z);
                    }
                }
            }
            if (all_int) {
                // Integer-valued ids in range are passed through verbatim; ids
                // were filled during the single Stata read above.
                return ids;
            }

            // Sparse / out-of-range values: dense-rank by ascending value. The id
            // assigned to each observation equals the number of distinct values
            // below it, which is invariant to how the sort orders equal keys, so
            // this is bit-for-bit identical to the previous std::sort-based code.
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
                throw_with_prefix(std::string("empty categorical variable: ") + what);
            }
            return ids;
        };

        Eigen::VectorXd y(n);
        Eigen::MatrixXd Xmat(n, std::max(0, k - 1));
        for (int i = 0; i < n; ++i) {
            y(i) = read_numeric(idx_in_start, obs[static_cast<std::size_t>(i)]);
        }
        for (int j = 1; j < k; ++j) {
            const int var_idx = idx_in_start + j;
            for (int i = 0; i < n; ++i) {
                Xmat(i, j - 1) = read_numeric(var_idx, obs[static_cast<std::size_t>(i)]);
            }
        }

        std::vector<Eigen::VectorXi> fes;
        fes.reserve(static_cast<std::size_t>(nfe));
        for (int d = 0; d < nfe; ++d) {
            const int var_idx = idx_fe_start + d;
            Eigen::VectorXi v = read_categorical_numeric(var_idx, "fes");
            fes.push_back(std::move(v));
        }

        std::optional<Eigen::VectorXd> weights;
        if (has_weight) {
            Eigen::VectorXd w(n);
            for (int i = 0; i < n; ++i) {
                w(i) = read_numeric(idx_weight, obs[static_cast<std::size_t>(i)]);
            }
            weights = std::move(w);
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
                    throw_with_prefix("cluster_fe_map length must equal nclust");
                }
            }

            std::vector<Eigen::VectorXi> c;
            c.reserve(static_cast<std::size_t>(nclust));
            for (int d = 0; d < nclust; ++d) {
                const int mapped_fe =
                    cluster_fe_map.empty() ? -1 : cluster_fe_map[static_cast<std::size_t>(d)];
                if (mapped_fe >= 0) {
                    if (mapped_fe >= nfe) {
                        throw_with_prefix("cluster_fe_map entry out of range");
                    }
                    c.push_back(fes[static_cast<std::size_t>(mapped_fe)]);
                    continue;
                }
                const int var_idx = idx_clust_start + d;
                Eigen::VectorXi v = read_categorical_numeric(var_idx, "clusters");
                c.push_back(std::move(v));
            }
            clusters = std::move(c);
        }
        const std::vector<Eigen::VectorXi>* c_ptr = clusters ? &(*clusters) : nullptr;

        const std::optional<std::string> s_N = args.get_optional("s_N");
        const std::optional<std::string> s_N_full = args.get_optional("s_N_full");
        const std::optional<std::string> s_num_singletons = args.get_optional("s_num_singletons");
        const std::optional<std::string> s_df_a = args.get_optional("s_df_a");
        const std::optional<std::string> s_df_a_levels = args.get_optional("s_df_a_levels");
        const std::optional<std::string> s_df_a_exact = args.get_optional("s_df_a_exact");
        const std::optional<std::string> s_df_a_nested = args.get_optional("s_df_a_nested");
        const std::optional<std::string> s_iterations = args.get_optional("s_iterations");
        const std::optional<std::string> s_converged = args.get_optional("s_converged");
        const std::optional<std::string> s_threads_used = args.get_optional("s_threads_used");
        const std::optional<std::string> s_gpu_used = args.get_optional("s_gpu_used");
        const std::optional<std::string> s_gpu_status_code =
            args.get_optional("s_gpu_status_code");
        const std::optional<std::string> s_gpu_attempted = args.get_optional("s_gpu_attempted");
        const std::optional<std::string> s_gpu_absorption_converged =
            args.get_optional("s_gpu_absorption_converged");
        const std::optional<std::string> s_gpu_absorption_iterations =
            args.get_optional("s_gpu_absorption_iterations");
        const std::optional<std::string> s_method_used = args.get_optional("s_method_used");
        const std::optional<std::string> dof_table = args.get_optional("dof_table");

        HdfeRegressorV11 reg(opts, threading);
        const hdfe::detail::AbsorptionResult absorption = reg.partial_out(y, Xmat, fes, w_ptr, c_ptr);
        const hdfe::HdfeResults& r = reg.results();

        maybe_save_scalar(s_N, r.nobs_effective);
        maybe_save_scalar(s_N_full, r.nobs_full_effective);
        maybe_save_scalar(s_num_singletons, r.num_singletons_effective);
        maybe_save_scalar(s_df_a, r.df_a);
        maybe_save_scalar(s_df_a_levels, r.df_a_levels);
        maybe_save_scalar(s_df_a_exact, r.df_a_exact);
        maybe_save_scalar(s_df_a_nested, r.df_a_nested);
        maybe_save_scalar(s_iterations, static_cast<double>(r.num_iterations));
        maybe_save_scalar(s_converged, r.converged ? 1.0 : 0.0);
        maybe_save_scalar(s_threads_used, static_cast<double>(reg.threads_used()));
        maybe_save_scalar(s_gpu_used, reg.gpu_used() ? 1.0 : 0.0);
        maybe_save_gpu_diagnostics(s_gpu_status_code, s_gpu_attempted,
                                   s_gpu_absorption_converged,
                                   s_gpu_absorption_iterations, reg);
        maybe_save_scalar(s_method_used, static_cast<double>(static_cast<int>(reg.absorption_method_used())));
        maybe_store_dof_table(dof_table, r);

        if (r.sample_index.size() != r.nobs) {
            throw_with_prefix("unexpected sample_index size");
        }

        std::vector<uint8_t> keep;
        if (esample_prefilled && r.nobs < n) {
            keep.assign(static_cast<std::size_t>(n), 0);
        }

        for (int i = 0; i < r.nobs; ++i) {
            const int orig = r.sample_index(i);
            if (orig < 0 || orig >= n) {
                throw_with_prefix("invalid sample_index");
            }
            if (!keep.empty()) {
                keep[static_cast<std::size_t>(orig)] = 1;
            }
            const int obs_no = obs[static_cast<std::size_t>(orig)];

            // Output column 0 is y_tilde; remaining columns are X_tilde.
            {
                const ST_retcode rc = SF_vstore(idx_out_start, obs_no, absorption.y_tilde(i));
                if (rc) {
                    throw_with_prefix("failed to store xfe output");
                }
            }
            for (int j = 1; j < k; ++j) {
                const ST_retcode rc = SF_vstore(idx_out_start + j, obs_no, absorption.X_tilde(i, j - 1));
                if (rc) {
                    throw_with_prefix("failed to store xfe output");
                }
            }

            if (store_groupvar) {
                const double gv =
                    (r.groupvar.size() == r.nobs) ? static_cast<double>(r.groupvar(i) + 1) : SV_missval;
                const ST_retcode rc = SF_vstore(idx_groupvar_out, obs_no, gv);
                if (rc) {
                    throw_with_prefix("failed to store groupvar");
                }
            }
            if (!esample_prefilled) {
                const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 1.0);
                if (rc) {
                    throw_with_prefix("failed to store e(sample)");
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
                const int obs_no = obs[static_cast<std::size_t>(i)];
                const ST_retcode rc = SF_vstore(idx_esample_out, obs_no, 0.0);
                if (rc) {
                    throw_with_prefix("failed to clear e(sample)");
                }
            }
        }

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
        const char* msg = "xfe plugin: unknown error\n";
        SF_error(const_cast<char*>(msg));
        return static_cast<ST_retcode>(198);
    }
}
