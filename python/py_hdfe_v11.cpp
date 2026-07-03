#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <cstdint>
#include <unordered_map>

#include "hdfe/hdfe_regressor_v11.hpp"

namespace py = pybind11;
using hdfe::AbsorptionMethod;
using hdfe::ConvergenceCriterion;
using hdfe::DofAdjustmentMethod;
using hdfe::FixefDofMethod;
using hdfe::HdfeOptions;
using hdfe::StandardErrorType;
using hdfe::ClusterDofMethod;
using hdfe::ToleranceMode;
using hdfe::detail::HeterogeneousSlopeTerm;
using hdfe::v11::HdfeRegressorV11;
using hdfe::v11::GroupAggregation;
using hdfe::v11::ThreadingOptions;

namespace {

int checked_index_id(std::int64_t value, const char* label) {
    if (value < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " contains an id outside the int32 range");
    }
    return static_cast<int>(value);
}

bool fits_int32(std::int64_t value) {
    return value >= static_cast<std::int64_t>(std::numeric_limits<int>::min()) &&
           value <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
}

template <typename Getter>
Eigen::VectorXi remap_or_cast_int64_ids(ssize_t n, const char* label, Getter&& get_value) {
    Eigen::VectorXi ids(static_cast<int>(n));
    // Common path (covers every in-range int64 id column, e.g. the simulated-panel and QP
    // benchmarks): a SINGLE pass that casts directly and bails to the remap only if it ever
    // meets an out-of-int32 value. The earlier two-pass form (pre-scan + cast) doubled the
    // per-column parse cost on large int64 inputs for no benefit on the dominant in-range
    // case.
    ssize_t i = 0;
    for (; i < n; ++i) {
        const std::int64_t value = get_value(i);
        if (!fits_int32(value)) {
            break;
        }
        ids(static_cast<int>(i)) = static_cast<int>(value);
    }
    if (i == n) {
        return ids;
    }

    // Some id exceeds the int32 range: remap ALL ids of this column to compact [0,next)
    // codes (by first appearance) so distinct labels can never collide via 32-bit
    // truncation. Restart from 0 so the mapping is consistent across the whole column.
    std::unordered_map<std::int64_t, int> levels;
    levels.reserve(static_cast<std::size_t>(std::min<ssize_t>(n, 1'000'000)));
    int next = 0;
    for (ssize_t i = 0; i < n; ++i) {
        const std::int64_t value = get_value(i);
        auto it = levels.find(value);
        if (it == levels.end()) {
            if (next == std::numeric_limits<int>::max()) {
                throw std::runtime_error(std::string(label) + " has too many distinct ids");
            }
            it = levels.emplace(value, next++).first;
        }
        ids(static_cast<int>(i)) = it->second;
    }
    return ids;
}

StandardErrorType parse_se_type(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "homoskedastic" || lower == "classical" || lower == "unadjusted" ||
        lower == "unadj" || lower == "ols") {
        return StandardErrorType::Homoskedastic;
    }
    if (lower == "robust" || lower == "hc1" || lower == "heteroskedastic") {
        return StandardErrorType::Robust;
    }
    if (lower == "cluster" || lower == "clustered") {
        return StandardErrorType::Cluster;
    }
    throw std::runtime_error("Unknown standard error type: " + name);
}

AbsorptionMethod parse_absorption_method(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "auto") {
        return AbsorptionMethod::Auto;
    }
    if (lower == "gs" || lower == "gauss-seidel" || lower == "gauss_seidel") {
        return AbsorptionMethod::GaussSeidel;
    }
    if (lower == "sym" || lower == "symmetric" || lower == "symgs" ||
        lower == "symmetric-gauss-seidel" || lower == "symmetric_gauss_seidel") {
        return AbsorptionMethod::SymmetricGaussSeidel;
    }
    if (lower == "jacobi") {
        return AbsorptionMethod::Jacobi;
    }
    if (lower == "schwarz" || lower == "pcg" || lower == "schwarz-pcg") {
        return AbsorptionMethod::Schwarz;
    }
    if (lower == "lsmr" || lower == "plain-lsmr" || lower == "plain_lsmr") {
        return AbsorptionMethod::Lsmr;
    }
    if (lower == "mlsmr" || lower == "modified-lsmr" || lower == "modified_lsmr" ||
        lower == "within" || lower == "within-additive" || lower == "within_additive") {
        return AbsorptionMethod::Mlsmr;
    }
    if (lower == "auto-mlsmr" || lower == "auto_mlsmr" ||
        lower == "mlsmr-auto" || lower == "mlsmr_auto") {
        return AbsorptionMethod::AutoMlsmr;
    }
    throw std::runtime_error("Unknown absorption method: " + name);
}

ToleranceMode parse_tolerance_mode(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "xhdfe-fast" || lower == "xhdfe_fast" || lower == "xhdfe" ||
        lower == "fast" || lower == "current") {
        return ToleranceMode::XhdfeFast;
    }
    if (lower == "reghdfe-comparable" || lower == "reghdfe_comparable" ||
        lower == "reghdfe" || lower == "comparable") {
        return ToleranceMode::ReghdfeComparable;
    }
    if (lower == "strict-residual" || lower == "strict_residual" ||
        lower == "residual-certificate" || lower == "residual_certificate" ||
        lower == "strict") {
        return ToleranceMode::StrictResidual;
    }
    throw std::runtime_error("Unknown tolerance mode: " + name);
}

ConvergenceCriterion parse_convergence_criterion(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "auto" || lower.empty()) {
        return ConvergenceCriterion::Auto;
    }
    if (lower == "normchange" || lower == "norm_change" || lower == "norm-change" ||
        lower == "norm") {
        return ConvergenceCriterion::NormChange;
    }
    if (lower == "reghdfe" || lower == "update" || lower == "reldif") {
        return ConvergenceCriterion::Reghdfe;
    }
    if (lower == "both") {
        return ConvergenceCriterion::Both;
    }
    throw std::runtime_error("Unknown convergence criterion: " + name);
}

FixefDofMethod parse_fixef_method(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "full" || lower == "true" || lower == "yes") {
        return FixefDofMethod::Full;
    }
    if (lower == "none" || lower == "false" || lower == "no") {
        return FixefDofMethod::None;
    }
    if (lower == "nonnested" || lower == "non-nested" || lower == "non_nested") {
        return FixefDofMethod::Nonnested;
    }
    throw std::runtime_error("Unknown K.fixef: " + name);
}

ClusterDofMethod parse_cluster_df(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "min") {
        return ClusterDofMethod::Min;
    }
    if (lower == "conventional" || lower == "conv") {
        return ClusterDofMethod::Conventional;
    }
    throw std::runtime_error("Unknown G.df: " + name);
}

GroupAggregation parse_group_aggregation(const std::string& name) {
    const std::string lower = [&name]() {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return tmp;
    }();
    if (lower == "mean" || lower == "average" || lower == "avg") {
        return GroupAggregation::Mean;
    }
    if (lower == "sum") {
        return GroupAggregation::Sum;
    }
    throw std::runtime_error("Unknown aggregation: " + name);
}

Eigen::VectorXi parse_index_vector(const py::handle& obj,
                                  const char* label,
                                  ssize_t expected_n) {
    if (expected_n <= 0) {
        return Eigen::VectorXi();
    }
    py::array raw;
    if (py::isinstance<py::array>(obj)) {
        raw = py::reinterpret_borrow<py::array>(obj);
    } else {
        auto converted = py::array_t<std::int64_t, py::array::forcecast>::ensure(obj);
        if (!converted) {
            throw std::runtime_error(std::string(label) + " must be array-like");
        }
        raw = converted;
    }
    if (!raw) {
        throw std::runtime_error(std::string(label) + " must be array-like");
    }
    if (raw.ndim() != 1) {
        throw std::runtime_error(std::string(label) + " must be 1-D");
    }
    if (raw.shape(0) != expected_n) {
        throw std::runtime_error(std::string(label) + " length must match nobs");
    }

    const auto dtype = raw.dtype();
    if (dtype.is(py::dtype::of<std::int32_t>())) {
        auto arr = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>(raw);
        return Eigen::Map<const Eigen::VectorXi>(arr.data(),
                                                static_cast<Eigen::Index>(expected_n));
    }
    if (dtype.is(py::dtype::of<std::int64_t>())) {
        auto arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(raw);
        const std::int64_t* data = arr.data();
        return remap_or_cast_int64_ids(expected_n, label, [data](ssize_t i) {
            return data[i];
        });
    }
    auto arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(raw);
    const std::int64_t* data = arr.data();
    return remap_or_cast_int64_ids(expected_n, label, [data](ssize_t i) {
        return data[i];
    });
}

Eigen::VectorXi parse_index_matrix_column(const py::array& arr,
                                         const char* label,
                                         ssize_t row,
                                         ssize_t col) {
    if (row <= 0) {
        return Eigen::VectorXi();
    }
    const auto dtype = arr.dtype();
    if (dtype.is(py::dtype::of<std::int32_t>())) {
        auto c_arr =
            py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>(arr);
        const std::int32_t* data = c_arr.data();
        const ssize_t stride = arr.shape(1);
        Eigen::VectorXi ids(static_cast<int>(row));
        for (ssize_t i = 0; i < row; ++i) {
            ids(static_cast<int>(i)) = checked_index_id(data[i * stride + col], label);
        }
        return ids;
    }
    if (dtype.is(py::dtype::of<std::int64_t>())) {
        auto c_arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(arr);
        const std::int64_t* data = c_arr.data();
        const ssize_t stride = arr.shape(1);
        return remap_or_cast_int64_ids(row, label, [data, stride, col](ssize_t i) {
            return data[i * stride + col];
        });
    }
    auto c_arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(arr);
    const std::int64_t* data = c_arr.data();
    const ssize_t stride = arr.shape(1);
    return remap_or_cast_int64_ids(row, label, [data, stride, col](ssize_t i) {
        return data[i * stride + col];
    });
}

Eigen::VectorXd parse_double_vector(const py::handle& obj,
                                    const char* label,
                                    ssize_t expected_n) {
    if (expected_n <= 0) {
        return Eigen::VectorXd();
    }
    auto borrowed = py::reinterpret_borrow<py::object>(obj);
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(borrowed);
    if (arr.ndim() != 1) {
        throw std::runtime_error(std::string(label) + " must be 1-D");
    }
    if (arr.shape(0) != expected_n) {
        throw std::runtime_error(std::string(label) + " length must match nobs");
    }
    return Eigen::Map<const Eigen::VectorXd>(arr.data(),
                                            static_cast<Eigen::Index>(expected_n));
}

HeterogeneousSlopeTerm parse_slope_term(const py::handle& obj,
                                        ssize_t expected_n) {
    HeterogeneousSlopeTerm term;
    if (py::isinstance<py::dict>(obj)) {
        py::dict item = py::reinterpret_borrow<py::dict>(obj);
        if (!item.contains("fe_index")) {
            throw std::runtime_error("Each heterogeneous slope dict needs fe_index");
        }
        if (!item.contains("values")) {
            throw std::runtime_error("Each heterogeneous slope dict needs values");
        }
        term.fe_index = py::cast<int>(item["fe_index"]);
        term.values = parse_double_vector(item["values"], "heterogeneous slope values", expected_n);
        term.include_intercept =
            item.contains("include_intercept") ? py::cast<bool>(item["include_intercept"]) : false;
        return term;
    }

    py::sequence item = py::reinterpret_borrow<py::sequence>(obj);
    if (item.size() < 2 || item.size() > 3) {
        throw std::runtime_error(
            "Each heterogeneous slope tuple must be (fe_index, values[, include_intercept])");
    }
    term.fe_index = py::cast<int>(item[0]);
    term.values = parse_double_vector(item[1], "heterogeneous slope values", expected_n);
    term.include_intercept = item.size() == 3 ? py::cast<bool>(item[2]) : false;
    return term;
}

std::optional<std::vector<HeterogeneousSlopeTerm>> parse_slope_terms(py::object slopes_obj,
                                                                     ssize_t expected_n) {
    if (slopes_obj.is_none()) {
        return std::nullopt;
    }
    std::vector<HeterogeneousSlopeTerm> out;
    if (py::isinstance<py::dict>(slopes_obj)) {
        out.push_back(parse_slope_term(slopes_obj, expected_n));
        return out;
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(slopes_obj);
    out.reserve(seq.size());
    for (auto item : seq) {
        out.push_back(parse_slope_term(item, expected_n));
    }
    return out;
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

struct ParsedDofAdjustments {
    DofAdjustmentMethod method = DofAdjustmentMethod::All;
    bool adjust_clusters = true;
    bool adjust_continuous = true;
};

ParsedDofAdjustments parse_dofadjustments(py::object dof_obj) {
    ParsedDofAdjustments parsed;
    if (dof_obj.is_none()) {
        return parsed;
    }

    // When the user explicitly supplies dofadjustments(...), mimic Stata's behavior by
    // starting from a blank slate (only the listed items are enabled).
    parsed.method = DofAdjustmentMethod::All;
    parsed.adjust_clusters = false;
    parsed.adjust_continuous = false;

    std::vector<std::string> tokens;
    if (py::isinstance<py::str>(dof_obj)) {
        std::string raw = dof_obj.cast<std::string>();
        std::replace(raw.begin(), raw.end(), ',', ' ');
        std::istringstream iss(raw);
        std::string tok;
        while (iss >> tok) {
            tokens.push_back(tok);
        }
    } else if (py::isinstance<py::sequence>(dof_obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(dof_obj);
        tokens.reserve(seq.size());
        for (auto item : seq) {
            tokens.push_back(py::cast<std::string>(item));
        }
    } else {
        throw std::runtime_error("dofadjustments must be a string or a sequence of strings");
    }

    auto normalize = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    };

    for (const auto& tok_raw : tokens) {
        const std::string tok = normalize(tok_raw);
        if (tok.empty()) {
            continue;
        }
        if (tok == "all") {
            parsed.method = DofAdjustmentMethod::All;
            parsed.adjust_clusters = true;
            parsed.adjust_continuous = true;
            continue;
        }
        if (tok == "none") {
            parsed.method = DofAdjustmentMethod::None;
            continue;
        }
        if (tok == "firstpair" || tok == "first") {
            parsed.method = DofAdjustmentMethod::FirstPair;
            continue;
        }
        if (tok == "pairwise" || tok == "pair") {
            parsed.method = DofAdjustmentMethod::Pairwise;
            continue;
        }
        if (tok == "clusters" || tok == "cluster") {
            parsed.adjust_clusters = true;
            continue;
        }
        if (tok == "continuous" || tok == "cont") {
            parsed.adjust_continuous = true;
            continue;
        }
        throw std::runtime_error("Unknown dofadjustments token: " + tok_raw);
    }

    return parsed;
}

std::string summarize(const hdfe::HdfeResults& res,
                      int threads_used,
                      AbsorptionMethod method_used) {
    std::ostringstream oss;
    oss << "High-dimensional FE regression (v11)\n";
    oss << "Observations: " << res.nobs << ", Threads: " << threads_used
        << ", Absorption: " << method_name(method_used) << "\n";
    if (res.num_singletons > 0 || res.nobs_full > 0) {
        oss << "N (full): " << res.nobs_full << ", singletons dropped: " << res.num_singletons
            << ", df_a: " << res.df_a << ", df_r: " << res.df_resid << "\n";
    }
    oss << "Within R^2: " << res.r2_within << ", Overall R^2: " << res.r2 << "\n";
    oss << "Converged: " << (res.converged ? "yes" : "no")
        << ", iterations: " << res.num_iterations << "\n";
    if (!res.fe_num_levels.empty()) {
        oss << "Fixed effects: ";
        for (std::size_t i = 0; i < res.fe_num_levels.size(); ++i) {
            oss << res.fe_num_levels[i];
            if (i + 1 < res.fe_num_levels.size()) {
                oss << ", ";
            }
        }
        oss << "\n";
    }
    oss << "\n";
    oss << std::left << std::setw(12) << "Variable" << std::right << std::setw(14) << "Coef"
        << std::setw(14) << "Std.Err" << std::setw(10) << "t" << std::setw(10) << "P>|t|"
        << std::setw(14) << "CI Low" << std::setw(14) << "CI High" << "\n";
    for (int j = 0; j < res.coefficients.size(); ++j) {
        oss << std::left << std::setw(12) << ("b" + std::to_string(j)) << std::right
            << std::setw(14) << res.coefficients(j) << std::setw(14) << res.stderr(j)
            << std::setw(10) << res.tvalues(j) << std::setw(10) << res.pvalues(j)
            << std::setw(14) << res.conf_int(j, 0) << std::setw(14) << res.conf_int(j, 1) << "\n";
    }
    return oss.str();
}

}  // namespace

PYBIND11_MODULE(py_hdfe_v11, m) {
    m.doc() = R"pbdoc(
Python bindings for the hdfe C++ library (v11).

Adds reghdfe-style defaults: singleton dropping + DoF adjustments for robust/cluster inference.
)pbdoc";
    py::module_::import("numpy");
    pybind11::detail::npy_api::get();

    py::enum_<StandardErrorType>(m, "StandardErrorType", py::module_local())
        .value("homoskedastic", StandardErrorType::Homoskedastic)
        .value("robust", StandardErrorType::Robust)
        .value("cluster", StandardErrorType::Cluster);

    py::enum_<AbsorptionMethod>(m, "AbsorptionMethod", py::module_local())
        .value("auto", AbsorptionMethod::Auto)
        .value("gauss_seidel", AbsorptionMethod::GaussSeidel)
        .value("symmetric_gauss_seidel", AbsorptionMethod::SymmetricGaussSeidel)
        .value("jacobi", AbsorptionMethod::Jacobi)
        .value("schwarz", AbsorptionMethod::Schwarz)
        .value("lsmr", AbsorptionMethod::Lsmr)
        .value("mlsmr", AbsorptionMethod::Mlsmr)
        .value("auto_mlsmr", AbsorptionMethod::AutoMlsmr);

    py::enum_<ConvergenceCriterion>(m, "ConvergenceCriterion", py::module_local())
        .value("auto", ConvergenceCriterion::Auto)
        .value("normchange", ConvergenceCriterion::NormChange)
        .value("reghdfe", ConvergenceCriterion::Reghdfe)
        .value("both", ConvergenceCriterion::Both);

	    py::class_<HdfeRegressorV11>(m, "HdfeRegressor")
	        .def(py::init([](const std::string& se_type,
	                         double tol,
	                         int max_iter,
	                         int check_interval,
	                         const std::string& convergence,
	                         bool fit_intercept,
	                         int num_threads,
	                         int default_threads,
	                         int max_threads,
                         int min_parallel_rows,
                         int target_rows_per_thread,
                         bool drop_singletons,
                         bool retain_fes,
                         bool symmetric_sweep,
                         const std::string& absorption_method,
                         double jacobi_relaxation,
                         double level,
                         py::object keepsingletons,
                         py::object dofadjustments,
                         py::object groupvar,
                         py::object ssc_k_adj,
                         py::object ssc_k_fixef,
                         py::object ssc_k_exact,
                         py::object ssc_g_adj,
                         py::object ssc_g_df,
                         py::object ssc_t_df,
                         const std::string& tolerance_mode) {
                 HdfeOptions opts;
                 opts.se_type = parse_se_type(se_type);
                 if (level > 0.0 && level <= 1.0) {
                     level *= 100.0;
                 }
                 if (!(level > 0.0 && level < 100.0)) {
                     throw std::runtime_error("level must be between 0 and 100");
                 }
	                 opts.level = level;
		                 opts.tol = tol;
		                 opts.max_iter = max_iter;
		                 opts.convergence_check_interval = std::max(1, check_interval);
	                 opts.convergence_criterion = parse_convergence_criterion(convergence);
		                 opts.fit_intercept = fit_intercept;
	                 opts.num_threads = num_threads;
                 if (!keepsingletons.is_none()) {
                     opts.drop_singletons = !keepsingletons.cast<bool>();
                 } else {
                     opts.drop_singletons = drop_singletons;
                 }
                 opts.retain_fixed_effects = retain_fes;
                 opts.symmetric_sweep = symmetric_sweep;
                 opts.absorption_method = parse_absorption_method(absorption_method);
                 opts.jacobi_relaxation = jacobi_relaxation;
                 opts.use_krylov = false;

                 const ParsedDofAdjustments dof = parse_dofadjustments(dofadjustments);
                 opts.dof_method = dof.method;
                 opts.dof_adjust_clusters = dof.adjust_clusters;
                 opts.dof_adjust_continuous = dof.adjust_continuous;

                 if (!groupvar.is_none()) {
                     if (py::isinstance<py::bool_>(groupvar)) {
                         opts.save_groupvar = groupvar.cast<bool>();
                     } else {
                         opts.save_groupvar = true;
                     }
                 }

                 if (!ssc_k_adj.is_none()) {
                     opts.ssc_k_adj = ssc_k_adj.cast<bool>();
                 }
                 if (!ssc_k_fixef.is_none()) {
                     opts.ssc_k_fixef = parse_fixef_method(ssc_k_fixef.cast<std::string>());
                 }
                 if (!ssc_k_exact.is_none()) {
                     opts.ssc_k_exact = ssc_k_exact.cast<bool>();
                 }
                 if (!ssc_g_adj.is_none()) {
                     opts.ssc_g_adj = ssc_g_adj.cast<bool>();
                 }
                 if (!ssc_g_df.is_none()) {
                     opts.ssc_g_df = parse_cluster_df(ssc_g_df.cast<std::string>());
                 }
                 if (!ssc_t_df.is_none()) {
                     opts.ssc_t_df = ssc_t_df.cast<double>();
                 }
                 opts.tolerance_mode = parse_tolerance_mode(tolerance_mode);

                 ThreadingOptions threading;
                 threading.default_threads =
                     default_threads > 0 ? default_threads : (num_threads > 0 ? num_threads : 0);
                 threading.max_threads = max_threads;
                 threading.min_parallel_rows = min_parallel_rows;
                 threading.target_rows_per_thread = target_rows_per_thread;
                 threading.symmetric_sweep = symmetric_sweep;
                 return std::make_unique<HdfeRegressorV11>(opts, threading);
	             }),
	             py::arg("se_type") = "unadjusted",
		             py::arg("tol") = 1e-8,
		             py::arg("max_iter") = 100000,
		             py::arg("check_interval") = 1,
	             py::arg("convergence") = "auto",
		             py::arg("fit_intercept") = true,
	             py::arg("num_threads") = 0,
	             py::arg("default_threads") = 0,
             py::arg("max_threads") = 0,
             py::arg("min_parallel_rows") = 20000,
             py::arg("target_rows_per_thread") = 500000,
             py::arg("drop_singletons") = true,
             py::arg("retain_fes") = false,
             py::arg("symmetric_sweep") = false,
             py::arg("absorption_method") = "auto",
             py::arg("jacobi_relaxation") = 0.0,
             py::arg("level") = 95.0,
             py::arg("keepsingletons") = py::none(),
             py::arg("dofadjustments") = py::none(),
             py::arg("groupvar") = py::none(),
             py::arg("ssc_k_adj") = py::none(),
             py::arg("ssc_k_fixef") = py::none(),
             py::arg("ssc_k_exact") = py::none(),
             py::arg("ssc_g_adj") = py::none(),
             py::arg("ssc_g_df") = py::none(),
             py::arg("ssc_t_df") = py::none(),
             py::arg("tolerance_mode") = "reghdfe-comparable")
        .def(
            "fit",
            [](HdfeRegressorV11& self,
               py::array y_obj,
               py::array X_obj,
               py::object fes_obj,
               py::object weights_obj,
               py::object clusters_obj,
               py::object instruments_obj,
               std::vector<int> endogenous,
               py::object group_obj,
               py::object individual_obj,
               const std::string& aggregation,
               py::object slopes_obj) {
                auto y_arr =
                    py::array_t<double, py::array::c_style | py::array::forcecast>(y_obj);
                if (y_arr.ndim() != 1) {
                    throw std::runtime_error("y must be a 1-D array");
                }
                const ssize_t n = y_arr.shape(0);
                Eigen::Map<const Eigen::VectorXd> y_vec(y_arr.data(), n);

                py::array X_arr = py::array_t<double, py::array::forcecast>(X_obj);
                if (X_arr.ndim() != 2 || X_arr.shape(0) != n) {
                    throw std::runtime_error("X must be a 2-D array with matching rows");
                }
                using RowMajorMatrix =
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                const bool x_f_style = (X_arr.flags() & py::array::f_style) != 0;
                const bool x_c_style = (X_arr.flags() & py::array::c_style) != 0;
                if (!x_f_style && !x_c_style) {
                    X_arr =
                        py::array_t<double, py::array::f_style | py::array::forcecast>(X_obj);
                }

                std::vector<Eigen::VectorXi> fes;
                if (!fes_obj.is_none()) {
                    py::sequence seq = py::reinterpret_borrow<py::sequence>(fes_obj);
                    fes.reserve(seq.size());
                    for (auto item : seq) {
                        fes.push_back(parse_index_vector(item, "Each fixed-effect vector", n));
                    }
                }

                std::optional<Eigen::VectorXd> weights_vec;
                if (!weights_obj.is_none()) {
                    auto w_arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
                        weights_obj);
                    if (w_arr.ndim() != 1 || w_arr.shape(0) != n) {
                        throw std::runtime_error("weights must be length n");
                    }
                    Eigen::Map<const Eigen::VectorXd> w_map(w_arr.data(), n);
                    weights_vec = Eigen::VectorXd(w_map);
                }

                std::optional<std::vector<Eigen::VectorXi>> clusters_vec;
                if (!clusters_obj.is_none()) {
                    std::vector<Eigen::VectorXi> out;
                    if (py::isinstance<py::array>(clusters_obj)) {
                        py::array c_arr = py::array(clusters_obj);
                        if (c_arr.ndim() == 1) {
                            if (c_arr.shape(0) != n) {
                                throw std::runtime_error("clusters must be length n");
                            }
                            out.push_back(parse_index_vector(c_arr, "clusters", n));
                        } else if (c_arr.ndim() == 2) {
                            if (c_arr.shape(0) != n) {
                                throw std::runtime_error("clusters must have shape (n, q)");
                            }
                            const ssize_t q = c_arr.shape(1);
                            out.reserve(static_cast<std::size_t>(q));
                            for (ssize_t j = 0; j < q; ++j) {
                                out.push_back(parse_index_matrix_column(c_arr, "clusters", n, j));
                            }
                        } else {
                            throw std::runtime_error("clusters must be 1-D or 2-D");
                        }
                    } else if (py::isinstance<py::sequence>(clusters_obj)) {
                        py::sequence seq = py::reinterpret_borrow<py::sequence>(clusters_obj);
                        out.reserve(seq.size());
                        for (auto item : seq) {
                            out.push_back(parse_index_vector(item, "Each cluster vector", n));
                        }
                    } else {
                        throw std::runtime_error(
                            "clusters must be a numpy array or a sequence of 1-D arrays");
                    }
                    if (!out.empty()) {
                        clusters_vec = std::move(out);
                    }
                }

                std::optional<Eigen::MatrixXd> instruments_mat;
                if (!instruments_obj.is_none()) {
                    py::array inst_arr =
                        py::array_t<double, py::array::forcecast>(instruments_obj);
                    if (inst_arr.ndim() != 2 || inst_arr.shape(0) != n) {
                        throw std::runtime_error(
                            "instruments must be a 2-D array with matching rows");
                    }
                    const bool inst_f_style = (inst_arr.flags() & py::array::f_style) != 0;
                    const bool inst_c_style = (inst_arr.flags() & py::array::c_style) != 0;
                    if (!inst_f_style && !inst_c_style) {
                        inst_arr = py::array_t<double, py::array::f_style | py::array::forcecast>(
                            instruments_obj);
                    }
                    if ((inst_arr.flags() & py::array::f_style) != 0) {
                        Eigen::Map<const Eigen::MatrixXd> z_map(
                            static_cast<const double*>(inst_arr.data()), n,
                            inst_arr.shape(1));
                        instruments_mat = Eigen::MatrixXd(z_map);
                    } else {
                        Eigen::Map<const RowMajorMatrix> z_map(
                            static_cast<const double*>(inst_arr.data()), n,
                            inst_arr.shape(1));
                        instruments_mat = Eigen::MatrixXd(z_map);
                    }
                }

                const Eigen::VectorXd* w_ptr = weights_vec ? &(*weights_vec) : nullptr;
                const std::vector<Eigen::VectorXi>* c_ptr = clusters_vec ? &(*clusters_vec) : nullptr;
                const GroupAggregation agg = parse_group_aggregation(aggregation);
                std::optional<std::vector<HeterogeneousSlopeTerm>> slopes_vec =
                    parse_slope_terms(slopes_obj, n);
                const std::vector<HeterogeneousSlopeTerm>* slopes_ptr =
                    slopes_vec ? &(*slopes_vec) : nullptr;

                std::optional<Eigen::VectorXi> group_vec;
                if (!group_obj.is_none()) {
                    py::array g_arr = py::array(group_obj);
                    if (g_arr.ndim() != 1 || g_arr.shape(0) != n) {
                        throw std::runtime_error("group must be length n");
                    }
                    group_vec = parse_index_vector(g_arr, "group", n);
                }

                std::optional<Eigen::VectorXi> individual_vec;
                if (!individual_obj.is_none()) {
                    py::array i_arr = py::array(individual_obj);
                    if (i_arr.ndim() != 1 || i_arr.shape(0) != n) {
                        throw std::runtime_error("individual must be length n");
                    }
                    individual_vec = parse_index_vector(i_arr, "individual", n);
                }

                if (group_vec) {
                    if (slopes_ptr && !slopes_ptr->empty()) {
                        throw std::runtime_error(
                            "heterogeneous slopes are not supported with group()/individual() mode");
                    }
                    if (!instruments_obj.is_none() || !endogenous.empty()) {
                        throw std::runtime_error(
                            "IV/instruments are not supported with group()/individual() mode");
                    }
                    const Eigen::VectorXi* ind_ptr =
                        individual_vec ? &(*individual_vec) : nullptr;
                    if ((X_arr.flags() & py::array::f_style) != 0) {
                        Eigen::Map<const Eigen::MatrixXd> X_map(
                            static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                        self.fit_grouped(y_vec, X_map, fes, *group_vec, ind_ptr, agg, w_ptr, c_ptr);
                    } else {
                        Eigen::Map<const RowMajorMatrix> X_map(
                            static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                        Eigen::MatrixXd X_mat = X_map;
                        self.fit_grouped(y_vec, X_mat, fes, *group_vec, ind_ptr, agg, w_ptr, c_ptr);
                    }
                } else {
                    const Eigen::MatrixXd* inst_ptr =
                        instruments_mat ? &(*instruments_mat) : nullptr;
                    if ((X_arr.flags() & py::array::f_style) != 0) {
                        Eigen::Map<const Eigen::MatrixXd> X_map(
                            static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                        self.fit(y_vec, X_map, fes, w_ptr, c_ptr, inst_ptr, endogenous, slopes_ptr);
                    } else {
                        Eigen::Map<const RowMajorMatrix> X_map(
                            static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                        Eigen::MatrixXd X_mat = X_map;
                        self.fit(y_vec, X_mat, fes, w_ptr, c_ptr, inst_ptr, endogenous, slopes_ptr);
                    }
                }
            },
            py::arg("y"),
            py::arg("X"),
            py::arg("fes") = py::none(),
            py::arg("weights") = py::none(),
            py::arg("clusters") = py::none(),
            py::arg("instruments") = py::none(),
            py::arg("endogenous_idx") = std::vector<int>{},
            py::arg("group") = py::none(),
            py::arg("individual") = py::none(),
            py::arg("aggregation") = "mean",
            py::arg("slopes") = py::none())
        .def(
            "extract_group_individual_fes",
            [](const HdfeRegressorV11& self,
               py::array y_obj,
               py::array X_obj,
               py::object fes_obj,
               py::array group_obj,
               py::array individual_obj,
               py::object weights_obj,
               const std::string& aggregation,
               double tol_main,
               double tol_start,
               double tol_final,
               int max_iter_main,
               int max_iter_solver,
               int verbose,
               int accel,
               int start_accel,
               int every_accel,
               double factor,
               double a1p1,
               double a2p1,
               int a2p2) {
                auto y_arr =
                    py::array_t<double, py::array::c_style | py::array::forcecast>(y_obj);
                if (y_arr.ndim() != 1) {
                    throw std::runtime_error("y must be a 1-D array");
                }
                const ssize_t n = y_arr.shape(0);
                Eigen::Map<const Eigen::VectorXd> y_vec(y_arr.data(), n);

                py::array X_arr = py::array_t<double, py::array::forcecast>(X_obj);
                if (X_arr.ndim() != 2 || X_arr.shape(0) != n) {
                    throw std::runtime_error("X must be a 2-D array with matching rows");
                }
                using RowMajorMatrix =
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                const bool x_f_style = (X_arr.flags() & py::array::f_style) != 0;
                const bool x_c_style = (X_arr.flags() & py::array::c_style) != 0;
                if (!x_f_style && !x_c_style) {
                    X_arr =
                        py::array_t<double, py::array::f_style | py::array::forcecast>(X_obj);
                }

                py::array g_arr = py::array(group_obj);
                if (g_arr.ndim() != 1 || g_arr.shape(0) != n) {
                    throw std::runtime_error("group must be a 1-D array with matching rows");
                }
                Eigen::VectorXi g_ids = parse_index_vector(g_arr, "group", n);

                py::array i_arr = py::array(individual_obj);
                if (i_arr.ndim() != 1 || i_arr.shape(0) != n) {
                    throw std::runtime_error("individual must be a 1-D array with matching rows");
                }
                Eigen::VectorXi i_ids = parse_index_vector(i_arr, "individual", n);

                if (fes_obj.is_none()) {
                    throw std::runtime_error("fes must be provided for group/individual extraction");
                }
                std::vector<Eigen::VectorXi> fes;
                if (py::isinstance<py::array>(fes_obj)) {
                    py::array fe_arr = py::array(fes_obj);
                    if (fe_arr.ndim() == 1) {
                        if (fe_arr.shape(0) != n) {
                            throw std::runtime_error("fes vector must be length n");
                        }
                        fes.push_back(parse_index_vector(fe_arr, "fes vector", n));
                    } else if (fe_arr.ndim() == 2) {
                        if (fe_arr.shape(0) != n) {
                            throw std::runtime_error("fes matrix must have shape (n, k)");
                        }
                        const int k = static_cast<int>(fe_arr.shape(1));
                        fes.reserve(static_cast<std::size_t>(k));
                        for (int j = 0; j < k; ++j) {
                            fes.push_back(parse_index_matrix_column(fe_arr, "fes matrix", n, j));
                        }
                    } else {
                        throw std::runtime_error("fes must be a 1-D/2-D array or a sequence of arrays");
                    }
                } else if (py::isinstance<py::sequence>(fes_obj)) {
                    py::sequence seq = py::reinterpret_borrow<py::sequence>(fes_obj);
                    fes.reserve(seq.size());
                    for (auto item : seq) {
                        fes.push_back(parse_index_vector(item, "Each FE vector", n));
                    }
                } else {
                    throw std::runtime_error("fes must be a numpy array or a sequence of arrays");
                }

                std::optional<Eigen::VectorXd> weights_vec;
                if (!weights_obj.is_none()) {
                    auto w_arr =
                        py::array_t<double, py::array::c_style | py::array::forcecast>(weights_obj);
                    if (w_arr.ndim() != 1 || w_arr.shape(0) != n) {
                        throw std::runtime_error("weights must be a 1-D array with matching rows");
                    }
                    Eigen::Map<const Eigen::VectorXd> w_map(w_arr.data(), n);
                    weights_vec = Eigen::VectorXd(w_map);
                }
                const Eigen::VectorXd* w_ptr = weights_vec ? &(*weights_vec) : nullptr;

                hdfe::v11::GroupIndividualFeOptions fe_opts;
                fe_opts.tol_main = tol_main;
                fe_opts.tol_start = tol_start;
                fe_opts.tol_final = tol_final;
                fe_opts.max_iter_main = max_iter_main;
                fe_opts.max_iter_solver = max_iter_solver;
                fe_opts.verbose = verbose;
                fe_opts.accel = accel;
                fe_opts.start_accel = start_accel;
                fe_opts.every_accel = every_accel;
                fe_opts.factor = factor;
                fe_opts.a1p1 = a1p1;
                fe_opts.a2p1 = a2p1;
                fe_opts.a2p2 = a2p2;

                const GroupAggregation agg = parse_group_aggregation(aggregation);
                hdfe::v11::GroupIndividualFeEstimates res;
                if ((X_arr.flags() & py::array::f_style) != 0) {
                    Eigen::Map<const Eigen::MatrixXd> X_map(
                        static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                    res = self.extract_group_individual_fes(y_vec, X_map, fes, g_ids, i_ids, agg,
                                                           w_ptr, fe_opts);
                } else {
                    Eigen::Map<const RowMajorMatrix> X_map(
                        static_cast<const double*>(X_arr.data()), n, X_arr.shape(1));
                    Eigen::MatrixXd X_mat(X_map);
                    res = self.extract_group_individual_fes(y_vec, X_mat, fes, g_ids, i_ids, agg,
                                                           w_ptr, fe_opts);
                }

                py::dict out;
                out["individual_ids"] = res.individual_ids;
                out["individual_effects"] = res.individual_effects;
                out["fe_level_ids"] = res.fe_level_ids;
                out["fe_level_effects"] = res.fe_level_effects;
                out["iterations"] = res.iterations;
                out["converged"] = res.converged;
                out["mse"] = res.mse;
                return out;
            },
            py::arg("y"),
            py::arg("X"),
            py::arg("fes"),
            py::arg("group"),
            py::arg("individual"),
            py::arg("weights") = py::none(),
            py::arg("aggregation") = "mean",
            py::arg("tol_main") = 1e-9,
            py::arg("tol_start") = 1e-3,
            py::arg("tol_final") = 1e-9,
            py::arg("max_iter_main") = 100000,
            py::arg("max_iter_solver") = 1000,
            py::arg("verbose") = 0,
            py::arg("accel") = 2,
            py::arg("start_accel") = 5,
            py::arg("every_accel") = 5,
            py::arg("factor") = 1.0,
            py::arg("a1p1") = 0.75,
            py::arg("a2p1") = 1e-8,
            py::arg("a2p2") = 5)
        .def("summary", [](const HdfeRegressorV11& self) {
            return summarize(self.results(), self.threads_used(), self.absorption_method_used());
        })
        .def_property_readonly("coef_", [](const HdfeRegressorV11& self) {
            return self.results().coefficients;
        })
        .def_property_readonly("stderr_", [](const HdfeRegressorV11& self) {
            return self.results().stderr;
        })
        .def_property_readonly("tvalues_", [](const HdfeRegressorV11& self) {
            return self.results().tvalues;
        })
        .def_property_readonly("pvalues_", [](const HdfeRegressorV11& self) {
            return self.results().pvalues;
        })
        .def_property_readonly("conf_int_", [](const HdfeRegressorV11& self) {
            return self.results().conf_int;
        })
        .def_property_readonly("nobs_", [](const HdfeRegressorV11& self) {
            return self.results().nobs;
        })
        .def_property_readonly("nobs_full_", [](const HdfeRegressorV11& self) {
            return self.results().nobs_full;
        })
        .def_property_readonly("num_singletons_", [](const HdfeRegressorV11& self) {
            return self.results().num_singletons;
        })
        .def_property_readonly("sample_index_", [](const HdfeRegressorV11& self) {
            return self.results().sample_index;
        })
        .def_property_readonly("df_resid_", [](const HdfeRegressorV11& self) {
            return self.results().df_resid;
        })
        .def_property_readonly("df_resid_unadj_", [](const HdfeRegressorV11& self) {
            return self.results().df_resid_unadj;
        })
        .def_property_readonly("df_m_", [](const HdfeRegressorV11& self) {
            return self.results().df_m;
        })
        .def_property_readonly("df_a_", [](const HdfeRegressorV11& self) {
            return self.results().df_a;
        })
        .def_property_readonly("df_a_levels_", [](const HdfeRegressorV11& self) {
            return self.results().df_a_levels;
        })
        .def_property_readonly("df_a_exact_", [](const HdfeRegressorV11& self) {
            return self.results().df_a_exact;
        })
        .def_property_readonly("df_a_nested_", [](const HdfeRegressorV11& self) {
            return self.results().df_a_nested;
        })
        .def_property_readonly("r2_", [](const HdfeRegressorV11& self) {
            return self.results().r2;
        })
        .def_property_readonly("r2_within_", [](const HdfeRegressorV11& self) {
            return self.results().r2_within;
        })
        .def_property_readonly("rss_", [](const HdfeRegressorV11& self) {
            return self.results().rss;
        })
        .def_property_readonly("tss_", [](const HdfeRegressorV11& self) {
            return self.results().tss;
        })
        .def_property_readonly("tss_within_", [](const HdfeRegressorV11& self) {
            return self.results().tss_within;
        })
        .def_property_readonly("saturated_", [](const HdfeRegressorV11& self) {
            return self.results().is_saturated();
        })
        .def_property_readonly("covariance_", [](const HdfeRegressorV11& self) {
            return self.results().covariance;
        })
        .def_property_readonly("fe_num_levels_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().fe_num_levels;
                               })
        .def_property_readonly("converged_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().converged;
                               })
        .def_property_readonly("fe_effects_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().fe_effects;
                               })
        .def_property_readonly("fe_recovery_iterations_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().fe_recovery_iterations;
                               })
        .def_property_readonly("fe_recovery_max_delta_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().fe_recovery_max_delta;
                               })
        .def_property_readonly("fe_recovery_converged_",
                              [](const HdfeRegressorV11& self) {
                                   return self.results().fe_recovery_converged;
                               })
        .def_property_readonly("residuals_", [](const HdfeRegressorV11& self) {
            return self.results().residuals;
        })
        .def_property_readonly("num_iterations_", [](const HdfeRegressorV11& self) {
            return self.results().num_iterations;
        })
        .def_property_readonly("groupvar_", [](const HdfeRegressorV11& self) {
            return self.results().groupvar;
        })
        .def_property_readonly("num_clusters_", [](const HdfeRegressorV11& self) {
            return self.results().num_clusters;
        })
        .def_property_readonly("cluster_counts_", [](const HdfeRegressorV11& self) {
            return self.results().cluster_counts;
        })
        .def_property_readonly("cluster_combo_counts_", [](const HdfeRegressorV11& self) {
            return self.results().cluster_combo_counts;
        })
        .def_property_readonly("cluster_scale_", [](const HdfeRegressorV11& self) {
            return self.results().cluster_scale;
        })
        .def_property_readonly("threads_used_", [](const HdfeRegressorV11& self) {
            return self.threads_used();
        })
        .def_property_readonly("gpu_used_", [](const HdfeRegressorV11& self) {
            return self.gpu_used();
        })
        .def_property_readonly("gpu_status_code_", [](const HdfeRegressorV11& self) {
            return self.gpu_status_code();
        })
        .def_property_readonly("gpu_attempted_", [](const HdfeRegressorV11& self) {
            return self.gpu_attempted();
        })
        .def_property_readonly("gpu_absorption_converged_", [](const HdfeRegressorV11& self) {
            return self.gpu_absorption_converged();
        })
        .def_property_readonly("gpu_absorption_iterations_", [](const HdfeRegressorV11& self) {
            return self.gpu_absorption_iterations();
        })
        .def_property_readonly("absorption_method_used", [](const HdfeRegressorV11& self) {
            return self.absorption_method_used();
        });
}
