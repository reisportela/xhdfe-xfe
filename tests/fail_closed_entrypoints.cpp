#include "hdfe/hdfe_regressor_v11.hpp"
#include "fe_absorption.hpp"
#include "fe_absorption_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    if (hdfe::detail::thread_cuda_forward_probe_requested()) {
        std::fprintf(stderr,
                     "FAIL: CUDA forward-probe TLS starts enabled\n");
        return 1;
    }
    {
        hdfe::detail::ScopedCudaForwardProbeRequest outer(true);
        if (!hdfe::detail::thread_cuda_forward_probe_requested()) {
            std::fprintf(stderr,
                         "FAIL: CUDA forward-probe TLS scope did not enable\n");
            return 1;
        }
        {
            hdfe::detail::ScopedCudaForwardProbeRequest inner(false);
            if (hdfe::detail::thread_cuda_forward_probe_requested()) {
                std::fprintf(stderr,
                             "FAIL: nested CUDA forward-probe TLS did not override\n");
                return 1;
            }
        }
        if (!hdfe::detail::thread_cuda_forward_probe_requested()) {
            std::fprintf(stderr,
                         "FAIL: nested CUDA forward-probe TLS did not restore\n");
            return 1;
        }
    }
    if (hdfe::detail::thread_cuda_forward_probe_requested()) {
        std::fprintf(stderr,
                     "FAIL: CUDA forward-probe TLS scope leaked\n");
        return 1;
    }
    try {
        hdfe::detail::ScopedCudaForwardProbeRequest scope(true);
        throw std::runtime_error("forward-probe TLS unwind test");
    } catch (const std::runtime_error&) {
    }
    if (hdfe::detail::thread_cuda_forward_probe_requested()) {
        std::fprintf(stderr,
                     "FAIL: CUDA forward-probe TLS leaked after exception\n");
        return 1;
    }

    constexpr int n = 240;
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = std::sin(0.11 * static_cast<double>(i));
        X(i, 1) = std::cos(0.07 * static_cast<double>(i));
        y(i) = 0.4 * X(i, 0) - 0.7 * X(i, 1) +
               std::sin(0.31 * static_cast<double>(i));
    }

    hdfe::HdfeOptions options;
    options.num_threads = 1;
    options.num_threads_explicit = true;
    options.drop_singletons = false;
    options.tol = 1e-10;
    options.tolerance_mode = hdfe::ToleranceMode::StrictResidual;
    hdfe::v11::HdfeRegressorV11 reg(options);
    const std::vector<Eigen::VectorXi> fes;

    if (std::string(reg.lifecycle_state_name()) != "empty" ||
        reg.generation() != 0 || reg.results().converged ||
        reg.results().precision_certified ||
        reg.results().fe_recovery_converged ||
        reg.results().coefficients.size() != 0 ||
        reg.absorption_method_used() != hdfe::AbsorptionMethod::Auto ||
        reg.threads_used() != 0 || reg.gpu_used()) {
        std::fprintf(stderr,
                     "FAIL: fresh regressor advertised a successful/stale state\n");
        return 1;
    }

    reg.fit(y, X, fes);
    if (!reg.results().converged || !reg.results().precision_certified) {
        std::fprintf(stderr,
                     "FAIL: valid setup fit did not establish a successful prior state\n");
        return 1;
    }
    if (std::string(reg.lifecycle_state_name()) != "standard_ready" ||
        reg.generation() != 1 || !reg.has_estimation_result()) {
        std::fprintf(stderr, "FAIL: successful fit lifecycle state is invalid\n");
        return 1;
    }

    Eigen::VectorXd invalid_y = y;
    invalid_y(7) = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try {
        (void)reg.partial_out(invalid_y, X, fes);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::fprintf(stderr, "FAIL: partial_out accepted a NaN outcome\n");
        return 1;
    }
    if (reg.results().converged || reg.results().precision_certified ||
        reg.results().coefficients.size() != 0 ||
        reg.results().residuals.size() != 0 ||
        reg.absorption_method_used() != hdfe::AbsorptionMethod::Auto ||
        reg.threads_used() != 0 || reg.gpu_used() ||
        std::string(reg.lifecycle_state_name()) != "failed" ||
        reg.generation() != 2) {
        std::fprintf(stderr,
                     "FAIL: partial_out exception exposed stale successful results\n");
        return 1;
    }

    reg.fit(y, X, fes);
    if (std::string(reg.lifecycle_state_name()) != "standard_ready" ||
        reg.generation() != 3 || !reg.results().converged) {
        std::fprintf(stderr, "FAIL: valid fit did not recover after failure\n");
        return 1;
    }
    const hdfe::detail::AbsorptionResult partial = reg.partial_out(y, X, fes);
    if (!partial.converged ||
        std::string(reg.lifecycle_state_name()) != "partial_ready" ||
        reg.generation() != 4 || !reg.has_partial_result() ||
        reg.has_estimation_result() || reg.results().coefficients.size() != 0) {
        std::fprintf(stderr, "FAIL: partial_out lifecycle state is invalid\n");
        return 1;
    }
    Eigen::VectorXd extreme_scale = Eigen::VectorXd::LinSpaced(20, 1.0, 2.0);
    Eigen::VectorXd extreme_y = extreme_scale * 1e100;
    Eigen::MatrixXd extreme_X(20, 1);
    extreme_X.col(0) = extreme_scale * 1e-100;
    bool mid_fit_threw = false;
    try {
        reg.fit(extreme_y, extreme_X, fes);
    } catch (const std::runtime_error&) {
        mid_fit_threw = true;
    }
    if (!mid_fit_threw || std::string(reg.lifecycle_state_name()) != "failed" ||
        reg.generation() != 5 || reg.results().coefficients.size() != 0 ||
        reg.absorption_method_used() != hdfe::AbsorptionMethod::Auto) {
        std::fprintf(stderr, "FAIL: mid-fit failure retained consumable state\n");
        return 1;
    }
    reg.fit(y, X, fes);
    if (std::string(reg.lifecycle_state_name()) != "standard_ready" ||
        reg.generation() != 6 || !reg.results().converged) {
        std::fprintf(stderr, "FAIL: standard fit did not recover after mid failure\n");
        return 1;
    }

    // The group/individual certificate must independently reject a perturbed
    // within vector under non-unit weights. Start from an exact zero residual
    // so the mutation, rather than solver behavior, is the only moving part.
    constexpr int grouped_n = 8;
    Eigen::VectorXd grouped_y(grouped_n);
    Eigen::MatrixXd grouped_X(grouped_n, 1);
    Eigen::VectorXd grouped_weights(grouped_n);
    Eigen::VectorXi grouped_fe(grouped_n);
    for (int row = 0; row < grouped_n; ++row) {
        grouped_y[row] = 1.0 + 0.2 * static_cast<double>(row);
        grouped_X(row, 0) = std::cos(0.3 * static_cast<double>(row));
        grouped_weights[row] = 0.5 + static_cast<double>((row % 4) + 1);
        grouped_fe[row] = row % 3;
    }

    hdfe::detail::GroupIndividualStructure gi;
    gi.num_groups = grouped_n;
    gi.num_individuals = 4;
    gi.group_ptr.resize(grouped_n + 1);
    gi.group_scale.assign(grouped_n, 0.5);
    for (int group = 0; group < grouped_n; ++group) {
        gi.group_ptr[static_cast<std::size_t>(group)] =
            static_cast<int>(gi.group_individual.size());
        gi.group_individual.push_back(group % gi.num_individuals);
        gi.group_individual.push_back((group + 1) % gi.num_individuals);
    }
    gi.group_ptr[static_cast<std::size_t>(grouped_n)] =
        static_cast<int>(gi.group_individual.size());
    gi.individual_ptr.assign(
        static_cast<std::size_t>(gi.num_individuals + 1), 0);
    for (const int individual : gi.group_individual) {
        ++gi.individual_ptr[static_cast<std::size_t>(individual + 1)];
    }
    for (int individual = 0; individual < gi.num_individuals; ++individual) {
        gi.individual_ptr[static_cast<std::size_t>(individual + 1)] +=
            gi.individual_ptr[static_cast<std::size_t>(individual)];
    }
    gi.individual_group.resize(gi.group_individual.size());
    std::vector<int> cursor = gi.individual_ptr;
    for (int group = 0; group < grouped_n; ++group) {
        for (int pos = gi.group_ptr[static_cast<std::size_t>(group)];
             pos < gi.group_ptr[static_cast<std::size_t>(group + 1)]; ++pos) {
            const int individual =
                gi.group_individual[static_cast<std::size_t>(pos)];
            gi.individual_group[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(individual)]++)] = group;
        }
    }

    hdfe::HdfeOptions grouped_options;
    grouped_options.num_threads = 1;
    grouped_options.num_threads_explicit = true;
    grouped_options.tol = 1e-10;
    grouped_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
    const std::vector<Eigen::VectorXi> grouped_fes{grouped_fe};
    hdfe::detail::AbsorptionResult exact;
    exact.y_tilde = Eigen::VectorXd::Zero(grouped_n);
    exact.X_tilde = Eigen::MatrixXd::Zero(grouped_n, 1);
    exact.converged = true;
    if (!hdfe::detail::certify_group_individual_candidate(
            grouped_y, grouped_X, grouped_fes, gi, &grouped_weights,
            grouped_options, exact)) {
        std::fprintf(stderr,
                     "FAIL: canonical group certificate rejected exact residual\n");
        return 1;
    }

    hdfe::detail::AbsorptionResult mutated = exact;
    mutated.y_tilde[0] = 1e-4;
    mutated.converged = true;
    if (hdfe::detail::certify_group_individual_candidate(
            grouped_y, grouped_X, grouped_fes, gi, &grouped_weights,
            grouped_options, mutated) || mutated.precision_certified) {
        std::fprintf(stderr,
                     "FAIL: canonical group certificate accepted perturbation\n");
        return 1;
    }

    hdfe::detail::AbsorptionResult nan_candidate = exact;
    nan_candidate.y_tilde[0] = std::numeric_limits<double>::quiet_NaN();
    nan_candidate.converged = true;
    if (hdfe::detail::certify_group_individual_candidate(
            grouped_y, grouped_X, grouped_fes, gi, &grouped_weights,
            grouped_options, nan_candidate) ||
        nan_candidate.precision_certified) {
        std::fprintf(stderr,
                     "FAIL: canonical group certificate accepted NaN residual\n");
        return 1;
    }

    hdfe::detail::AbsorptionResult inf_candidate = exact;
    inf_candidate.X_tilde(0, 0) =
        std::numeric_limits<double>::infinity();
    inf_candidate.converged = true;
    if (hdfe::detail::certify_group_individual_candidate(
            grouped_y, grouped_X, grouped_fes, gi, &grouped_weights,
            grouped_options, inf_candidate) ||
        inf_candidate.precision_certified) {
        std::fprintf(stderr,
                     "FAIL: canonical group certificate accepted infinite residual\n");
        return 1;
    }

    // Malformed internal CSR/index data must be rejected before any CUDA
    // kernel can dereference it. The CPU-only inline backend also returns
    // false, so this remains a portable CTest gate.
    hdfe::HdfeOptions malformed_options = grouped_options;
    malformed_options.from_auto = true;
    const std::vector<hdfe::detail::GpuFeInput> no_standard_gpu_fes;
    const std::vector<std::size_t> no_sweep_order;
    auto malformed_cuda_accepted = [&](const hdfe::detail::GroupIndividualStructure& bad) {
        hdfe::detail::AbsorptionResult bad_result;
        return hdfe::detail::absorb_fixed_effects_group_individual_cuda(
            grouped_y, grouped_X, no_standard_gpu_fes, bad,
            &grouped_weights, no_sweep_order, malformed_options,
            hdfe::AbsorptionMethod::Auto, bad_result);
    };
    hdfe::detail::GroupIndividualStructure bad_ptr = gi;
    bad_ptr.group_ptr[3] = bad_ptr.group_ptr[2] - 1;
    if (malformed_cuda_accepted(bad_ptr)) {
        std::fprintf(stderr, "FAIL: CUDA GI LSMR accepted non-monotone CSR\n");
        return 1;
    }
    hdfe::detail::GroupIndividualStructure bad_individual = gi;
    bad_individual.group_individual[0] = bad_individual.num_individuals;
    if (malformed_cuda_accepted(bad_individual)) {
        std::fprintf(stderr, "FAIL: CUDA GI LSMR accepted invalid individual index\n");
        return 1;
    }
    hdfe::detail::GroupIndividualStructure bad_group = gi;
    bad_group.individual_group[0] = bad_group.num_groups;
    if (malformed_cuda_accepted(bad_group)) {
        std::fprintf(stderr, "FAIL: CUDA GI LSMR accepted invalid group index\n");
        return 1;
    }

    // A grouped CPU solve that exhausts its iteration budget must not expose
    // best-effort coefficients through any binding. The core exception is the
    // shared fail-closed boundary used by C++, Python, R, and Stata.
    constexpr int chain_groups = 128;
    constexpr int chain_rows = 2 * chain_groups;
    Eigen::VectorXd chain_y(chain_rows);
    Eigen::MatrixXd chain_X(chain_rows, 1);
    Eigen::VectorXi chain_group(chain_rows);
    Eigen::VectorXi chain_individual(chain_rows);
    Eigen::VectorXi chain_standard_fe(chain_rows);
    for (int group = 0; group < chain_groups; ++group) {
        for (int member = 0; member < 2; ++member) {
            const int row = 2 * group + member;
            chain_group[row] = group;
            chain_individual[row] = group + member;
            chain_standard_fe[row] = group % 7;
            chain_X(row, 0) = std::sin(0.17 * static_cast<double>(group));
            chain_y[row] =
                0.6 * chain_X(row, 0) +
                std::cos(0.11 * static_cast<double>(group));
        }
    }
    hdfe::HdfeOptions fail_options;
    fail_options.num_threads = 1;
    fail_options.num_threads_explicit = true;
    fail_options.drop_singletons = false;
    fail_options.max_iter = 1;
    fail_options.tol = 1e-14;
    fail_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
    fail_options.absorption_method = hdfe::AbsorptionMethod::Lsmr;
    hdfe::v11::HdfeRegressorV11 fail_reg(fail_options);
    const std::vector<Eigen::VectorXi> chain_fes{
        chain_individual, chain_standard_fe};

    // A successful grouped fit authorizes extraction only while its full
    // certified state remains current. Auto must re-enter as internal Auto,
    // not turn a reported LSMR method into a public explicit-LSMR request.
    hdfe::HdfeOptions state_options;
    state_options.num_threads = 1;
    state_options.num_threads_explicit = true;
    state_options.drop_singletons = false;
    state_options.max_iter = 2000;
    state_options.tol = 1e-10;
    state_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
    state_options.absorption_method = hdfe::AbsorptionMethod::Auto;
    hdfe::v11::HdfeRegressorV11 state_reg(state_options);
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        state_reg.fit(chain_y, chain_X, chain_fes);
    }
    bool standard_extract_threw = false;
    try {
        (void)state_reg.extract_group_individual_fes(
            chain_y, chain_X, chain_fes, chain_group, chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        standard_extract_threw = true;
    }
    if (!standard_extract_threw) {
        std::fprintf(stderr,
                     "FAIL: extractor accepted a standard-fit state\n");
        return 1;
    }
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        state_reg.fit_grouped(
            chain_y, chain_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    }
    if (!state_reg.results().converged ||
        !state_reg.results().precision_certified ||
        state_reg.absorption_method_used() != hdfe::AbsorptionMethod::Lsmr) {
        std::fprintf(stderr,
                     "FAIL: grouped Auto setup did not establish certified LSMR state\n");
        return 1;
    }
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        const hdfe::v11::GroupIndividualFeEstimates extracted =
            state_reg.extract_group_individual_fes(
                chain_y, chain_X, chain_fes, chain_group, chain_individual,
                hdfe::v11::GroupAggregation::Mean);
        if (!extracted.converged) {
            std::fprintf(stderr,
                         "FAIL: certified grouped Auto state could not extract FEs\n");
            return 1;
        }
    }

    // A valid FE-only grouped fit has no regression coefficients. Extraction
    // authority comes from GroupedReady plus the exact fit signature, not from
    // a non-empty coefficient vector.
    Eigen::MatrixXd chain_no_X(chain_rows, 0);
    hdfe::HdfeOptions fe_only_options = state_options;
    fe_only_options.fit_intercept = false;
    hdfe::v11::HdfeRegressorV11 fe_only_reg(fe_only_options);
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        fe_only_reg.fit_grouped(
            chain_y, chain_no_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean);
        if (std::string(fe_only_reg.lifecycle_state_name()) != "grouped_ready" ||
            !fe_only_reg.results().converged ||
            !fe_only_reg.results().precision_certified ||
            fe_only_reg.results().coefficients.size() != 0) {
            std::fprintf(stderr,
                         "FAIL: FE-only grouped fit did not establish ready state\n");
            return 1;
        }
        const auto extracted = fe_only_reg.extract_group_individual_fes(
            chain_y, chain_no_X, chain_fes, chain_group, chain_individual,
            hdfe::v11::GroupAggregation::Mean);
        if (!extracted.converged || !extracted.individual_effects.allFinite()) {
            std::fprintf(stderr,
                         "FAIL: certified FE-only grouped state could not extract FEs\n");
            return 1;
        }
    }
    Eigen::VectorXd changed_fe_only_y = chain_y;
    changed_fe_only_y[0] += 1e-6;
    bool fe_only_signature_threw = false;
    try {
        (void)fe_only_reg.extract_group_individual_fes(
            changed_fe_only_y, chain_no_X, chain_fes, chain_group,
            chain_individual, hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        fe_only_signature_threw = true;
    }
    if (!fe_only_signature_threw) {
        std::fprintf(stderr,
                     "FAIL: FE-only extraction accepted a mismatched signature\n");
        return 1;
    }
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        fe_only_reg.fit(chain_y, chain_no_X, chain_fes);
    }
    bool fe_only_stale_threw = false;
    try {
        (void)fe_only_reg.extract_group_individual_fes(
            chain_y, chain_no_X, chain_fes, chain_group, chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        fe_only_stale_threw = true;
    }
    if (!fe_only_stale_threw) {
        std::fprintf(stderr,
                     "FAIL: FE-only extraction survived a standard refit\n");
        return 1;
    }

    Eigen::VectorXd chain_weights(chain_rows);
    for (int row = 0; row < chain_rows; ++row) {
        chain_weights[row] = 1.0 + static_cast<double>((row / 2) % 2);
    }
    hdfe::v11::HdfeRegressorV11 signature_reg(state_options);
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        signature_reg.fit_grouped(
            chain_y, chain_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean, &chain_weights);
    }
    auto extraction_succeeds = [&](const hdfe::v11::HdfeRegressorV11& candidate) {
        try {
            const auto extracted = candidate.extract_group_individual_fes(
                chain_y, chain_X, chain_fes, chain_group, chain_individual,
                hdfe::v11::GroupAggregation::Mean, &chain_weights);
            return extracted.converged;
        } catch (const std::runtime_error&) {
            return false;
        }
    };
    auto signature_rejected = [&](const Eigen::VectorXd& supplied_y,
                                  const Eigen::MatrixXd& supplied_X,
                                  const std::vector<Eigen::VectorXi>& supplied_fes,
                                  const Eigen::VectorXi& supplied_group,
                                  const Eigen::VectorXi& supplied_individual,
                                  hdfe::v11::GroupAggregation supplied_aggregation,
                                  const Eigen::VectorXd* supplied_weights) {
        try {
            (void)signature_reg.extract_group_individual_fes(
                supplied_y, supplied_X, supplied_fes, supplied_group,
                supplied_individual, supplied_aggregation, supplied_weights);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    if (!extraction_succeeds(signature_reg)) {
        std::fprintf(stderr, "FAIL: exact grouped signature was rejected\n");
        return 1;
    }
    Eigen::VectorXd changed_y = chain_y;
    changed_y[0] += 1e-6;
    Eigen::MatrixXd changed_X = chain_X;
    changed_X(0, 0) += 1e-6;
    std::vector<Eigen::VectorXi> changed_fes = chain_fes;
    changed_fes[1][0] += 1;
    Eigen::VectorXi changed_group = chain_group;
    changed_group[0] += 1;
    Eigen::VectorXi changed_individual = chain_individual;
    std::vector<Eigen::VectorXi> changed_individual_fes = chain_fes;
    changed_individual[0] = 10000;
    changed_individual_fes[0][0] = 10000;
    Eigen::VectorXd changed_weights = chain_weights;
    changed_weights[0] += 1.0;
    if (!signature_rejected(changed_y, chain_X, chain_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &chain_weights) ||
        !signature_rejected(chain_y, changed_X, chain_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &chain_weights) ||
        !signature_rejected(chain_y, chain_X, changed_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &chain_weights) ||
        !signature_rejected(chain_y, chain_X, chain_fes, changed_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &chain_weights) ||
        !signature_rejected(chain_y, chain_X, changed_individual_fes,
                            chain_group, changed_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &chain_weights) ||
        !signature_rejected(chain_y, chain_X, chain_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean,
                            &changed_weights) ||
        !signature_rejected(chain_y, chain_X, chain_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Sum,
                            &chain_weights) ||
        !signature_rejected(chain_y, chain_X, chain_fes, chain_group,
                            chain_individual,
                            hdfe::v11::GroupAggregation::Mean, nullptr)) {
        std::fprintf(stderr, "FAIL: grouped extraction signature accepted a mutation\n");
        return 1;
    }

    hdfe::v11::HdfeRegressorV11 copied(signature_reg);
    hdfe::v11::HdfeRegressorV11 assigned(state_options);
    assigned = signature_reg;
    if (!extraction_succeeds(copied) || !extraction_succeeds(assigned) ||
        copied.generation() != signature_reg.generation() ||
        assigned.generation() != signature_reg.generation()) {
        std::fprintf(stderr, "FAIL: copy lost grouped lifecycle/signature state\n");
        return 1;
    }
    hdfe::v11::HdfeRegressorV11 moved(std::move(copied));
    if (!extraction_succeeds(moved) ||
        std::string(copied.lifecycle_state_name()) != "empty" ||
        copied.results().coefficients.size() != 0) {
        std::fprintf(stderr, "FAIL: move construction left an unsafe state\n");
        return 1;
    }
    copied.fit(chain_y, chain_X, chain_fes);
    if (std::string(copied.lifecycle_state_name()) != "standard_ready") {
        std::fprintf(stderr, "FAIL: moved-from regressor could not recover\n");
        return 1;
    }
    hdfe::v11::HdfeRegressorV11 move_assigned(state_options);
    move_assigned = std::move(assigned);
    if (!extraction_succeeds(move_assigned) ||
        std::string(assigned.lifecycle_state_name()) != "empty") {
        std::fprintf(stderr, "FAIL: move assignment lost lifecycle state\n");
        return 1;
    }
    const std::uint64_t pre_setter_generation = move_assigned.generation();
    move_assigned.set_weights_are_frequencies(true);
    if (std::string(move_assigned.lifecycle_state_name()) != "empty" ||
        move_assigned.generation() != pre_setter_generation + 1 ||
        move_assigned.results().coefficients.size() != 0 ||
        extraction_succeeds(move_assigned)) {
        std::fprintf(stderr, "FAIL: semantic setter retained ready state\n");
        return 1;
    }

    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        state_reg.fit(chain_y, chain_X, chain_fes);
    }
    bool post_group_standard_extract_threw = false;
    try {
        (void)state_reg.extract_group_individual_fes(
            chain_y, chain_X, chain_fes, chain_group, chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        post_group_standard_extract_threw = true;
    }
    if (!post_group_standard_extract_threw) {
        std::fprintf(stderr,
                     "FAIL: standard refit retained grouped extraction authority\n");
        return 1;
    }
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        state_reg.fit_grouped(
            chain_y, chain_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    }

    bool inner_absorption_threw = false;
    {
        // Metal is a deterministic unavailable backend in this CUDA-first
        // implementation. It forces the extractor's inner absorption to
        // return a failed candidate without changing the prior successful fit.
        hdfe::detail::ScopedGpuBackendOverride force_unavailable(
            hdfe::detail::GpuBackend::Metal);
        try {
            (void)state_reg.extract_group_individual_fes(
                chain_y, chain_X, chain_fes, chain_group, chain_individual,
                hdfe::v11::GroupAggregation::Mean);
        } catch (const std::runtime_error&) {
            inner_absorption_threw = true;
        }
    }
    if (!inner_absorption_threw) {
        std::fprintf(stderr,
                     "FAIL: extractor consumed a failed inner absorption\n");
        return 1;
    }

    Eigen::VectorXd invalid_chain_y = chain_y;
    invalid_chain_y[0] = std::numeric_limits<double>::quiet_NaN();
    bool second_fit_threw = false;
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        try {
            state_reg.fit_grouped(
                invalid_chain_y, chain_X, chain_fes, chain_group,
                &chain_individual, hdfe::v11::GroupAggregation::Mean);
        } catch (const std::runtime_error&) {
            second_fit_threw = true;
        }
    }
    if (!second_fit_threw || state_reg.results().converged ||
        state_reg.results().precision_certified ||
        state_reg.results().coefficients.size() != 0 ||
        state_reg.results().residuals.size() != 0 ||
        state_reg.absorption_method_used() != hdfe::AbsorptionMethod::Auto) {
        std::fprintf(stderr,
                     "FAIL: failed grouped refit exposed stale consumable state\n");
        return 1;
    }
    bool stale_extract_threw = false;
    try {
        (void)state_reg.extract_group_individual_fes(
            chain_y, chain_X, chain_fes, chain_group, chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        stale_extract_threw = true;
    }
    if (!stale_extract_threw) {
        std::fprintf(stderr,
                     "FAIL: extractor accepted state after a failed grouped refit\n");
        return 1;
    }
    {
        hdfe::detail::ScopedGpuBackendOverride force_cpu(
            hdfe::detail::GpuBackend::Cpu);
        state_reg.fit_grouped(
            chain_y, chain_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    }
    if (std::string(state_reg.lifecycle_state_name()) != "grouped_ready" ||
        !state_reg.results().converged ||
        !state_reg.results().precision_certified) {
        std::fprintf(stderr, "FAIL: grouped fit did not recover after failure\n");
        return 1;
    }

    bool grouped_fit_threw = false;
    try {
        fail_reg.fit_grouped(
            chain_y, chain_X, chain_fes, chain_group, &chain_individual,
            hdfe::v11::GroupAggregation::Mean);
    } catch (const std::runtime_error&) {
        grouped_fit_threw = true;
    }
    if (!grouped_fit_threw) {
        std::fprintf(stderr,
                     "FAIL: grouped CPU non-convergence exposed estimates\n");
        return 1;
    }
    if (fail_reg.results().converged ||
        fail_reg.results().precision_certified ||
        fail_reg.results().coefficients.size() != 0 ||
        fail_reg.results().residuals.size() != 0 ||
        fail_reg.absorption_method_used() != hdfe::AbsorptionMethod::Auto ||
        fail_reg.threads_used() != 0 || fail_reg.gpu_used() ||
        std::string(fail_reg.lifecycle_state_name()) != "failed") {
        std::fprintf(stderr,
                     "FAIL: grouped CPU failure exposed a successful result state\n");
        return 1;
    }

    // Reghdfe fit-statistics use df_resid_unadj - df_a_nested for sigma2,
    // without feeding that reporting denominator back into clustered VCV or
    // inferential df_r. Legacy keeps its pre-existing denominator and fallback.
    {
        constexpr int stats_n = 240;
        constexpr int stats_levels = 20;
        Eigen::VectorXd stats_y(stats_n);
        Eigen::MatrixXd stats_X(stats_n, 2);
        Eigen::VectorXi stats_fe(stats_n);
        Eigen::VectorXi stats_cluster(stats_n);
        for (int row = 0; row < stats_n; ++row) {
            stats_fe[row] = row % stats_levels;
            stats_cluster[row] = stats_fe[row];
            stats_X(row, 0) = std::sin(0.17 * static_cast<double>(row));
            stats_X(row, 1) = std::cos(0.13 * static_cast<double>(row));
            stats_y[row] = 0.4 * stats_X(row, 0) - 0.2 * stats_X(row, 1) +
                           0.03 * static_cast<double>(stats_fe[row]) +
                           std::sin(0.31 * static_cast<double>(row));
        }
        const std::vector<Eigen::VectorXi> stats_fes{stats_fe};
        const std::vector<Eigen::VectorXi> stats_clusters{stats_cluster};
        hdfe::HdfeOptions stats_options;
        stats_options.num_threads = 1;
        stats_options.num_threads_explicit = true;
        stats_options.drop_singletons = false;
        stats_options.tol = 1e-10;
        stats_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
        stats_options.se_type = hdfe::StandardErrorType::Cluster;

        hdfe::v11::HdfeRegressorV11 stats_reghdfe(stats_options);
        stats_reghdfe.fit(
            stats_y, stats_X, stats_fes, nullptr, &stats_clusters);
        const auto& reghdfe_result = stats_reghdfe.results();
        const double reghdfe_df =
            reghdfe_result.df_resid_unadj - reghdfe_result.df_a_nested;
        const double reghdfe_sigma2 = reghdfe_result.rss / reghdfe_df;
        if (!(reghdfe_df > 0.0) ||
            std::abs(reghdfe_result.sigma2 - reghdfe_sigma2) >
                1e-12 * std::max(1.0, std::abs(reghdfe_sigma2)) ||
            !(reghdfe_result.df_resid > 0.0) ||
            !reghdfe_result.std_errors.allFinite()) {
            std::fprintf(stderr,
                         "FAIL: regular nested Reghdfe sigma2/VCV contract drifted\n");
            return 1;
        }

        hdfe::HdfeOptions legacy_options = stats_options;
        legacy_options.stats_style = hdfe::StatsStyle::Legacy;
        hdfe::v11::HdfeRegressorV11 stats_legacy(legacy_options);
        stats_legacy.fit(stats_y, stats_X, stats_fes, nullptr, &stats_clusters);
        const auto& legacy_result = stats_legacy.results();
        const double legacy_sigma2 =
            legacy_result.rss / legacy_result.df_resid_unadj;
        if (std::abs(legacy_result.sigma2 - legacy_sigma2) >
            1e-12 * std::max(1.0, std::abs(legacy_sigma2))) {
            std::fprintf(stderr, "FAIL: legacy sigma2 definition changed\n");
            return 1;
        }
    }

    // Small deterministic analogue of synthetic-zigzag: two duplicated end
    // edges leave exactly two within dimensions for two slopes. The canonical
    // reporting denominator is zero while clustered inferential df_r is positive.
    {
        constexpr int zigzag_n = 102;
        Eigen::VectorXd zigzag_y(zigzag_n);
        Eigen::MatrixXd zigzag_X(zigzag_n, 2);
        Eigen::VectorXi zigzag_id1(zigzag_n);
        Eigen::VectorXi zigzag_id2(zigzag_n);
        for (int row = 0; row < zigzag_n; ++row) {
            if (row < 3) {
                zigzag_id1[row] = 0;
                zigzag_id2[row] = (row < 2) ? 0 : 1;
            } else if (row < 99) {
                const int id1 = 1 + (row - 3) / 2;
                zigzag_id1[row] = id1;
                zigzag_id2[row] = id1 + ((row - 3) % 2);
            } else {
                zigzag_id1[row] = 49;
                zigzag_id2[row] = (row == 99) ? 49 : 50;
            }
            zigzag_X(row, 0) = std::sin(0.19 * static_cast<double>(row));
            zigzag_X(row, 1) = std::cos(0.23 * static_cast<double>(row));
            zigzag_y[row] = std::sin(0.37 * static_cast<double>(row)) +
                             static_cast<double>(zigzag_id1[row]) / 7.0 -
                             static_cast<double>(zigzag_id2[row]) / 11.0;
        }
        const std::vector<Eigen::VectorXi> zigzag_fes{zigzag_id1, zigzag_id2};
        const std::vector<Eigen::VectorXi> zigzag_clusters{zigzag_id1};
        hdfe::HdfeOptions zigzag_options;
        zigzag_options.num_threads = 1;
        zigzag_options.num_threads_explicit = true;
        zigzag_options.drop_singletons = false;
        zigzag_options.tol = 1e-10;
        zigzag_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
        zigzag_options.se_type = hdfe::StandardErrorType::Cluster;
        hdfe::v11::HdfeRegressorV11 zigzag_reg(zigzag_options);
        zigzag_reg.fit(
            zigzag_y, zigzag_X, zigzag_fes, nullptr, &zigzag_clusters);
        const auto& zigzag_result = zigzag_reg.results();
        const double zigzag_df =
            zigzag_result.df_resid_unadj - zigzag_result.df_a_nested;
        if (zigzag_df != 0.0 || !(zigzag_result.df_resid > 0.0) ||
            std::abs(zigzag_result.sigma2 - zigzag_result.rss) >
                1e-12 * std::max(1.0, std::abs(zigzag_result.rss)) ||
            !zigzag_result.std_errors.allFinite()) {
            std::fprintf(stderr, "FAIL: zero-df Reghdfe sigma2/VCV contract drifted\n");
            return 1;
        }
    }

    // Group/individual negative-df saturated toy: Reghdfe sigma2 is missing;
    // the legacy path retains its historical zero fallback.
    {
        Eigen::VectorXd saturated_y(4);
        saturated_y << -1.0, -1.0, 1.0, 1.0;
        Eigen::MatrixXd saturated_X(4, 1);
        saturated_X << -1.0, -1.0, 1.0, 1.0;
        Eigen::VectorXi saturated_group(4);
        saturated_group << 0, 0, 1, 1;
        Eigen::VectorXi saturated_individual(4);
        saturated_individual << 0, 1, 1, 2;
        Eigen::VectorXi saturated_constant = Eigen::VectorXi::Zero(4);
        const std::vector<Eigen::VectorXi> saturated_fes{
            saturated_constant, saturated_individual};
        hdfe::HdfeOptions saturated_options;
        saturated_options.num_threads = 1;
        saturated_options.num_threads_explicit = true;
        saturated_options.drop_singletons = false;
        saturated_options.absorption_method = hdfe::AbsorptionMethod::GaussSeidel;
        saturated_options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
        hdfe::v11::HdfeRegressorV11 saturated_reghdfe(saturated_options);
        saturated_reghdfe.fit_grouped(
            saturated_y, saturated_X, saturated_fes, saturated_group,
            &saturated_individual, hdfe::v11::GroupAggregation::Sum);
        const auto& saturated_result = saturated_reghdfe.results();
        if (!(saturated_result.df_resid_unadj - saturated_result.df_a_nested < 0.0) ||
            !std::isnan(saturated_result.sigma2)) {
            std::fprintf(stderr, "FAIL: negative-df grouped Reghdfe sigma2 is not missing\n");
            return 1;
        }

        hdfe::HdfeOptions saturated_legacy_options = saturated_options;
        saturated_legacy_options.stats_style = hdfe::StatsStyle::Legacy;
        hdfe::v11::HdfeRegressorV11 saturated_legacy(saturated_legacy_options);
        saturated_legacy.fit_grouped(
            saturated_y, saturated_X, saturated_fes, saturated_group,
            &saturated_individual, hdfe::v11::GroupAggregation::Sum);
        if (saturated_legacy.results().sigma2 != 0.0) {
            std::fprintf(stderr, "FAIL: negative-df grouped legacy sigma2 changed\n");
            return 1;
        }
    }
    return 0;
}
