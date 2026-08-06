#include "hdfe/hdfe_regressor_v11.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
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

    reg.fit(y, X, fes);
    if (!reg.results().converged || !reg.results().precision_certified) {
        std::fprintf(stderr,
                     "FAIL: valid setup fit did not establish a successful prior state\n");
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
    if (reg.results().converged || reg.results().precision_certified) {
        std::fprintf(stderr,
                     "FAIL: partial_out exception exposed stale successful results\n");
        return 1;
    }
    return 0;
}
