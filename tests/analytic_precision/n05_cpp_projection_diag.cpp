// Diagnose the sole C++ harness projection discrepancy against N05 and N1.

#include "hdfe/hdfe_regressor_v11.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

double sign_bit(int row, int bit) {
    return 2.0 * ((row / (1 << bit)) % 2) - 1.0;
}

struct Fixture {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    Eigen::VectorXd expected_y;
    Eigen::MatrixXd expected_X;
};

Fixture make_fixture() {
    constexpr int n = 256;
    Fixture value;
    value.y.resize(n);
    value.X.resize(n, 2);
    value.fes.assign(2, Eigen::VectorXi(n));
    value.expected_y.resize(n);
    value.expected_X.resize(n, 2);
    for (int row = 0; row < n; ++row) {
        const int first = row / 64;
        const int second = (row / 16) % 4;
        const double a = sign_bit(row, 0);
        const double b = sign_bit(row, 1);
        const double c = sign_bit(row, 2);
        const double d = sign_bit(row, 3);
        value.fes[0][row] = first;
        value.fes[1][row] = second;
        value.X(row, 0) = a + 0.5 * b + 0.25 * first;
        value.X(row, 1) = c - 0.5 * second;
        value.expected_X(row, 0) = a + 0.5 * b;
        value.expected_X(row, 1) = c;
        value.expected_y[row] =
            0.75 * value.expected_X(row, 0) -
            0.5 * value.expected_X(row, 1) + 0.125 * d;
        value.y[row] = 0.75 * value.X(row, 0) - 0.5 * value.X(row, 1) +
                       3.0 + 2.0 * first - second + 0.125 * d;
    }
    return value;
}

double maximum_fe_sum(const Eigen::VectorXd& value,
                      const std::vector<Eigen::VectorXi>& fes) {
    double maximum = 0.0;
    for (const auto& fe : fes) {
        const int levels = fe.maxCoeff() + 1;
        Eigen::VectorXd sums = Eigen::VectorXd::Zero(levels);
        for (int row = 0; row < value.size(); ++row) sums[fe[row]] += value[row];
        maximum = std::max(maximum, sums.cwiseAbs().maxCoeff());
    }
    return maximum;
}

double maximum_fe_sum(const Eigen::MatrixXd& value,
                      const std::vector<Eigen::VectorXi>& fes) {
    double maximum = 0.0;
    for (int column = 0; column < value.cols(); ++column) {
        const Eigen::VectorXd column_value = value.col(column);
        maximum = std::max(maximum, maximum_fe_sum(column_value, fes));
    }
    return maximum;
}

}  // namespace

int main() {
    try {
        ::unsetenv("XHDFE_CERTIFY");
        ::setenv("XHDFE_GPU_BACKEND", "cpu", 1);
        ::setenv("XHDFE_ABSORPTION_CACHE_MODE", "off", 1);
        ::setenv("XHDFE_MOBILITY_MODE", "off", 1);
        const Fixture fixture = make_fixture();
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
        hdfe::v11::HdfeRegressorV11 model(options, threading);
        const auto result = model.partial_out(fixture.y, fixture.X, fixture.fes);
        const Eigen::VectorXd y_error = result.y_tilde - fixture.expected_y;
        const Eigen::MatrixXd X_error = result.X_tilde - fixture.expected_X;
        Eigen::Index worst = 0;
        y_error.cwiseAbs().maxCoeff(&worst);
        const double y_norm = fixture.expected_y.norm();
        std::cout << std::setprecision(17)
                  << "{\"max_y_error\":" << y_error.cwiseAbs().maxCoeff()
                  << ",\"max_y_error_index\":" << worst
                  << ",\"actual_y\":" << result.y_tilde[worst]
                  << ",\"expected_y\":" << fixture.expected_y[worst]
                  << ",\"y_norm\":" << y_norm
                  << ",\"relative_y_error\":"
                  << (y_norm > 0.0 ? y_error.norm() / y_norm : 0.0)
                  << ",\"max_X_error\":" << X_error.cwiseAbs().maxCoeff()
                  << ",\"max_y_fe_sum\":"
                  << maximum_fe_sum(result.y_tilde, fixture.fes)
                  << ",\"max_X_fe_sum\":"
                  << maximum_fe_sum(result.X_tilde, fixture.fes)
                  << ",\"abs_residual\":" << result.abs_residual
                  << ",\"rho\":" << result.abs_residual_rel
                  << ",\"iterations\":" << result.iterations
                  << ",\"method\":" << static_cast<int>(model.absorption_method_used())
                  << ",\"converged\":" << (result.converged ? "true" : "false")
                  << ",\"precision_certified\":"
                  << (result.precision_certified ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "N05_CPP_PROJECTION_DIAG_EXCEPTION " << error.what() << '\n';
        return 1;
    }
}
