// Private N05 partial_out and fit-context identity contract. Root compiles this
// with the exact flags of the identified libxhdfe.a; it is not a product API.

#include "hdfe/hdfe_regressor_v11.hpp"
#include "audit/ordinary_post_ols.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "N05_CPP_CONTRACT_FAIL %s\n", message);
    }
}

template <class Left, class Right>
bool same_dense_bits(const Eigen::MatrixBase<Left>& left,
                     const Eigen::MatrixBase<Right>& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) return false;
    for (Eigen::Index j = 0; j < left.cols(); ++j) {
        for (Eigen::Index i = 0; i < left.rows(); ++i) {
            const auto a = left(i, j);
            const auto b = right(i, j);
            if (std::memcmp(&a, &b, sizeof(a)) != 0) return false;
        }
    }
    return true;
}

bool same_double_bits(double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}

std::string capture_stderr(const std::function<void()>& action) {
    std::fflush(stderr);
    const char* directory = std::getenv("TMPDIR");
    if (!directory || !*directory) {
        throw std::runtime_error("TMPDIR must identify the evidence directory");
    }
    std::string pattern(directory);
    if (pattern.back() != '/') pattern += '/';
    pattern += "n05_cpp_stderr_XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int capture_fd = ::mkstemp(writable.data());
    if (capture_fd < 0) {
        throw std::runtime_error("mkstemp failed while capturing stderr");
    }
    FILE* sink = ::fdopen(capture_fd, "w+");
    if (!sink) {
        ::close(capture_fd);
        throw std::runtime_error("fdopen failed while capturing stderr");
    }
    const int saved = ::dup(STDERR_FILENO);
    if (saved < 0) {
        std::fclose(sink);
        throw std::runtime_error("dup failed while capturing stderr");
    }
    if (::dup2(::fileno(sink), STDERR_FILENO) < 0) {
        ::close(saved);
        std::fclose(sink);
        throw std::runtime_error("dup2 failed while capturing stderr");
    }
    auto restore = [&]() {
        std::fflush(stderr);
        const int status = ::dup2(saved, STDERR_FILENO);
        ::close(saved);
        if (status < 0) throw std::runtime_error("failed to restore stderr");
    };
    try {
        action();
        restore();
    } catch (...) {
        try {
            restore();
        } catch (...) {
        }
        std::fclose(sink);
        throw;
    }
    std::rewind(sink);
    std::string text;
    std::array<char, 4096> buffer{};
    while (const std::size_t count = std::fread(buffer.data(), 1, buffer.size(), sink)) {
        text.append(buffer.data(), count);
    }
    std::fclose(sink);
    return text;
}

std::size_t occurrences(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t at = 0; (at = text.find(needle, at)) != std::string::npos;
         at += needle.size()) {
        ++count;
    }
    return count;
}

std::size_t nonempty_lines(const std::string& text) {
    std::size_t count = 0;
    bool content = false;
    for (char value : text) {
        if (value == '\n') {
            count += content;
            content = false;
        } else if (value != '\r') {
            content = true;
        }
    }
    return count + content;
}

bool contains(const std::string& text, const char* value) {
    return text.find(value) != std::string::npos;
}

void set_environment(const char* name, const char* value) {
    if (::setenv(name, value, 1) != 0) {
        throw std::runtime_error(std::string("setenv failed for ") + name);
    }
}

void unset_environment(const char* name) {
    if (::unsetenv(name) != 0) {
        throw std::runtime_error(std::string("unsetenv failed for ") + name);
    }
}

struct DyadicFixture {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    Eigen::VectorXd expected_y_tilde;
    Eigen::MatrixXd expected_X_tilde;
};

double sign_bit(int row, int bit) {
    return 2.0 * ((row / (1 << bit)) % 2) - 1.0;
}

DyadicFixture make_fixture() {
    constexpr int n = 256;
    DyadicFixture fixture;
    fixture.y.resize(n);
    fixture.X.resize(n, 2);
    fixture.fes.assign(2, Eigen::VectorXi(n));
    fixture.expected_y_tilde.resize(n);
    fixture.expected_X_tilde.resize(n, 2);
    for (int row = 0; row < n; ++row) {
        const int first = row / 64;
        const int second = (row / 16) % 4;
        const double a = sign_bit(row, 0);
        const double b = sign_bit(row, 1);
        const double c = sign_bit(row, 2);
        const double d = sign_bit(row, 3);
        fixture.fes[0][row] = first;
        fixture.fes[1][row] = second;
        fixture.X(row, 0) = a + 0.5 * b + 0.25 * first;
        fixture.X(row, 1) = c - 0.5 * second;
        fixture.expected_X_tilde(row, 0) = a + 0.5 * b;
        fixture.expected_X_tilde(row, 1) = c;
        fixture.expected_y_tilde[row] =
            0.75 * fixture.expected_X_tilde(row, 0) -
            0.5 * fixture.expected_X_tilde(row, 1) + 0.125 * d;
        fixture.y[row] = fixture.X.row(row).dot(Eigen::Vector2d(0.75, -0.5)) +
                         3.0 + 2.0 * first - second + 0.125 * d;
    }
    return fixture;
}

bool same_absorption(const hdfe::detail::AbsorptionResult& left,
                     const hdfe::detail::AbsorptionResult& right) {
    return same_dense_bits(left.y_tilde, right.y_tilde) &&
           same_dense_bits(left.X_tilde, right.X_tilde) &&
           left.fe_levels == right.fe_levels &&
           left.fe_group_ids == right.fe_group_ids &&
           left.sweep_order_used == right.sweep_order_used &&
           left.iterations == right.iterations &&
           left.converged == right.converged &&
           left.precision_certified == right.precision_certified &&
           same_double_bits(left.abs_residual, right.abs_residual) &&
           same_double_bits(left.abs_residual_rel, right.abs_residual_rel) &&
           left.gpu_used == right.gpu_used &&
           left.gpu_status_code == right.gpu_status_code;
}

void identity_contracts() {
    Eigen::MatrixXd valid_matrix(3, 2);
    valid_matrix << 1, 2, 3, 4, 5, 6;
    std::vector<int> valid_columns{0, 2};
    std::uint64_t valid_generation = 7;
    hdfe::detail::OrdinaryPostOlsIdentity valid;
    valid.pin(valid_matrix);
    valid.pin(valid_columns);
    valid.pin_metadata([&]() { return valid_generation; });
    check(valid.matches(), "unchanged identity context was rejected");

    Eigen::MatrixXd changed_content = valid_matrix;
    hdfe::detail::OrdinaryPostOlsIdentity content_identity;
    content_identity.pin(changed_content);
    changed_content(1, 1) += 1.0;
    check(!content_identity.matches(), "changed input content retained certification");

    Eigen::MatrixXd changed_pointer = valid_matrix;
    Eigen::MatrixXd replacement = 2.0 * valid_matrix;
    hdfe::detail::OrdinaryPostOlsIdentity pointer_identity;
    pointer_identity.pin(changed_pointer);
    changed_pointer.swap(replacement);
    check(!pointer_identity.matches(), "moved input buffer retained certification");

    using DynamicStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using StridedMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
                                  0, DynamicStride>;
    std::array<double, 32> storage{};
    for (std::size_t index = 0; index < storage.size(); ++index) {
        storage[index] = static_cast<double>(index + 1);
    }
    alignas(StridedMap) unsigned char map_storage[sizeof(StridedMap)];
    StridedMap* view = new (map_storage) StridedMap(
        storage.data(), 3, 2, DynamicStride(4, 1));
    hdfe::detail::OrdinaryPostOlsIdentity stride_identity;
    stride_identity.pin(*view);
    view->~StridedMap();
    view = new (map_storage) StridedMap(
        storage.data(), 3, 2, DynamicStride(5, 1));
    check(!stride_identity.matches(), "changed Eigen stride retained certification");
    view->~StridedMap();

    std::uint64_t option_bits = 11;
    hdfe::detail::OrdinaryPostOlsIdentity metadata_identity;
    metadata_identity.pin_metadata([&]() { return option_bits; });
    option_bits = 12;
    check(!metadata_identity.matches(), "changed option metadata retained certification");

    std::uint64_t generation = 29;
    hdfe::detail::OrdinaryPostOlsIdentity generation_identity;
    generation_identity.pin_metadata([&]() { return generation; });
    ++generation;
    check(!generation_identity.matches(), "changed fit generation retained certification");
}

}  // namespace

int main() {
    try {
        identity_contracts();
        const std::array<const char*, 9> old_flags{
            "XHDFE_ORDINARY_PRECISION_DIAG",
            "XHDFE_ORDINARY_REFERENCE_TRACE",
            "XHDFE_ORDINARY_SUPPORT_TRACE",
            "XHDFE_ORDINARY_SUPPORT_BATCH",
            "XHDFE_ORDINARY_SUPPORT_BATCH_TRACE",
            "XHDFE_ORDINARY_STRUCTURED_COEFFICIENT",
            "XHDFE_ORDINARY_SANDWICH_PROFILE",
            "XHDFE_ORDINARY_FINAL_INFERENCE",
            "XHDFE_OLS_REFINEMENT_TRACE",
        };
        set_environment("XHDFE_GPU_BACKEND", "cpu");
        set_environment("XHDFE_ABSORPTION_CACHE_MODE", "off");
        set_environment("XHDFE_MOBILITY_MODE", "off");
        unset_environment("XHDFE_CERTIFY");
        for (const char* flag : old_flags) set_environment(flag, "1");

        const DyadicFixture fixture = make_fixture();
        const Eigen::VectorXd original_y = fixture.y;
        const Eigen::MatrixXd original_X = fixture.X;
        const std::vector<Eigen::VectorXi> original_fes = fixture.fes;
        hdfe::HdfeOptions options;
        options.se_type = hdfe::StandardErrorType::Homoskedastic;
        options.tol = 1e-8;
        options.max_iter = 1000;
        options.num_threads = 1;
        options.num_threads_explicit = true;
        options.drop_singletons = false;
        options.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
        hdfe::v11::ThreadingOptions threading;
        threading.default_threads = 1;
        threading.max_threads = 1;

        hdfe::v11::HdfeRegressorV11 disabled(options, threading);
        hdfe::detail::AbsorptionResult disabled_result;
        (void)xhdfe_private_n05_work_count(1);
        const std::string disabled_stderr = capture_stderr([&]() {
            disabled_result = disabled.partial_out(fixture.y, fixture.X, fixture.fes);
        });
        check(disabled_stderr.empty(), "obsolete flags produced disabled-path stderr");
        check(occurrences(disabled_stderr, "XHDFE_N05_AUDIT ") == 0,
              "disabled gate emitted an N05 receipt");
        check(xhdfe_private_n05_work_count(0) == 0,
              "disabled gate performed private N05 work");

        for (const char* flag : old_flags) unset_environment(flag);
        set_environment("XHDFE_CERTIFY", "1");
        hdfe::v11::HdfeRegressorV11 enabled(options, threading);
        hdfe::detail::AbsorptionResult enabled_result;
        (void)xhdfe_private_n05_work_count(1);
        const std::string enabled_stderr = capture_stderr([&]() {
            enabled_result = enabled.partial_out(fixture.y, fixture.X, fixture.fes);
        });
        const unsigned long long enabled_work = xhdfe_private_n05_work_count(0);

        check(nonempty_lines(enabled_stderr) == 1,
              "enabled partial_out did not emit exactly one stderr line");
        check(occurrences(enabled_stderr, "XHDFE_N05_AUDIT ") == 1,
              "enabled partial_out receipt count differs from one");
        check(contains(enabled_stderr, "\"schema\":\"xhdfe-n05-audit-v1\""),
              "partial_out receipt schema is missing");
        check(contains(enabled_stderr, "\"entrypoint\":\"partial_out\""),
              "partial_out receipt has the wrong entrypoint");
        check(contains(enabled_stderr, "\"mode\":\"reghdfe-comparable\""),
              "partial_out receipt has the wrong mode");
        check(contains(enabled_stderr, "\"backend\":\"cpu\""),
              "partial_out receipt has the wrong backend");
        check(contains(enabled_stderr, "\"cache_hit\":false"),
              "partial_out receipt has an impossible cache hit");
        check(contains(enabled_stderr, "\"identity_match\":false"),
              "partial_out receipt falsely claims a post-OLS identity");
        check(contains(enabled_stderr, "\"mutated\":false"),
              "partial_out receipt does not preserve its candidate");
        const bool projection_reported =
            contains(enabled_stderr, "\"projection\":{\"status\":\"PASS\"") ||
            contains(enabled_stderr,
                     "\"projection\":{\"status\":\"BOUND_INCONCLUSIVE\"");
        check(projection_reported, "partial_out projection family was not reported");
        for (const char* family : {"homoskedastic", "sandwich", "full_v"}) {
            const std::string expected = std::string("\"") + family +
                "\":{\"status\":\"UNSUPPORTED\",\"reason\":\"partial_out_no_ols\"}";
            check(contains(enabled_stderr, expected.c_str()),
                  "partial_out inference family lacks its unsupported label");
        }
        check(!contains(enabled_stderr, "ERROR_DEMONSTRATED"),
              "dyadic partial_out reported a demonstrated estimator error");
        check(enabled_work > 0, "enabled partial_out performed no private N05 work");

        check(same_absorption(disabled_result, enabled_result),
              "enabled audit changed the returned partial-out result");
        check(disabled.absorption_method_used() == enabled.absorption_method_used(),
              "enabled audit changed the absorption method");
        check(disabled.gpu_used() == enabled.gpu_used() && !enabled.gpu_used(),
              "CPU partial_out backend changed under audit");
        check(disabled.has_partial_result() && enabled.has_partial_result(),
              "partial_out lifecycle was not retained");
        check(same_dense_bits(disabled.results().sample_index,
                              enabled.results().sample_index),
              "partial_out sample identity changed under audit");
        const double tau = std::min(options.tol, 1e-9);
        const double projection_limit = 8.0 * tau;
        check((disabled_result.X_tilde - fixture.expected_X_tilde).norm() <=
                  projection_limit * std::max(1.0, fixture.expected_X_tilde.norm()),
              "dyadic X projection exceeds the 2.26.2 Comparable contract");
        check((disabled_result.y_tilde - fixture.expected_y_tilde).norm() <=
                  projection_limit * std::max(1.0, fixture.expected_y_tilde.norm()),
              "dyadic y projection exceeds the 2.26.2 Comparable contract");
        check(disabled_result.abs_residual_rel <= projection_limit,
              "dyadic rho exceeds the 2.26.2 Comparable contract");
        check(same_dense_bits(fixture.y, original_y) &&
                  same_dense_bits(fixture.X, original_X),
              "partial_out mutated caller-owned y or X");
        bool fes_unchanged = fixture.fes.size() == original_fes.size();
        for (std::size_t index = 0; fes_unchanged && index < fixture.fes.size(); ++index) {
            fes_unchanged = same_dense_bits(fixture.fes[index], original_fes[index]);
        }
        check(fes_unchanged, "partial_out mutated caller-owned FE labels");
        unset_environment("XHDFE_CERTIFY");
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "N05_CPP_CONTRACT_EXCEPTION %s\n", error.what());
    }
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d}\n",
                failures ? "FAIL" : "PASS", checks, failures);
    return failures != 0;
}
