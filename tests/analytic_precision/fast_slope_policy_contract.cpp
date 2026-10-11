#include "fe_absorption.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

// An approximate projection with a small remaining group intercept. The large
// original FE makes the historical aggregate criterion pass; the within-scaled
// block criterion rejects it. Both candidates differ from the originals only
// in the absorbed span. This tests real certificate arithmetic, not a mock.
int main() {
    using namespace hdfe;
    using namespace hdfe::detail;
    int cases = 0;
    auto require = [](bool ok) { if (!ok) throw std::runtime_error("policy check failed"); };
    for (bool weighted : {false, true}) for (double scale : {0.125, 1.0, 1024.0}) {
        Eigen::VectorXd y(8), z(8), weights(8), exact_y(8);
        Eigen::MatrixXd X(8,1), exact_X(8,1);
        Eigen::VectorXi fe(8);
        for (int i=0; i<8; ++i) {
            fe[i] = i/4;
            const double g = fe[i] ? 1.0 : -1.0;
            const double a = i%2 ? 1.0 : -1.0;
            z[i] = scale*((i/2)%2 ? 1.0 : -1.0);
            weights[i] = 1.0 + fe[i];
            exact_y[i] = 2*a;
            exact_X(i,0) = a;
            y[i] = 1e8*g + 2*a + z[i];
            X(i,0) = 1e8*g + a + 3*z[i];
        }
        HeterogeneousSlopeTerm term;
        term.fe_index=0; term.values=z; term.include_intercept=true;
        std::vector<HeterogeneousSlopeTerm> slopes{term};
        HdfeOptions fast; fast.tol=1e-8; fast.num_threads=2;
        fast.tolerance_mode=ToleranceMode::XhdfeFast;
        auto comparable=fast; comparable.tolerance_mode=ToleranceMode::ReghdfeComparable;
        const auto* w = weighted ? &weights : nullptr;
        auto candidate = [&](double defect) {
            AbsorptionResult r; r.y_tilde=exact_y; r.X_tilde=exact_X;
            for (int i=0;i<8;++i) {
                const double g=fe[i] ? 1.0 : -1.0;
                r.y_tilde[i] += defect*g; r.X_tilde(i,0) += defect*g;
            }
            r.converged=true;
            return r;
        };
        auto approximate=candidate(0.001);
        certify_absorption_result(y,X,{fe},w,fast,slopes,approximate);
        require(approximate.precision_certified && !approximate.slope_block_checked);
        require(approximate.abs_residual_rel <= 8e-8);
        certify_absorption_result(y,X,{fe},w,comparable,slopes,approximate);
        require(!approximate.precision_certified && approximate.slope_block_checked);
        require(approximate.slope_block_residual_rel > 1e-5);
        auto exact=candidate(0.0);
        certify_absorption_result(y,X,{fe},w,comparable,slopes,exact);
        require(exact.precision_certified && exact.slope_block_checked);
        // Recertification must clear the scope, not retain the previous flag.
        certify_absorption_result(y,X,{fe},w,fast,slopes,exact);
        require(exact.precision_certified && !exact.slope_block_checked);
        auto wrong=candidate(1e7);
        certify_absorption_result(y,X,{fe},w,fast,slopes,wrong);
        require(!wrong.precision_certified);
        auto unconverged=candidate(0.0); unconverged.converged=false;
        certify_absorption_result(y,X,{fe},w,fast,slopes,unconverged);
        require(!unconverged.precision_certified);
        for (double value : {std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
            auto invalid=candidate(0.0); invalid.X_tilde(0,0)=value;
            certify_absorption_result(y,X,{fe},w,fast,slopes,invalid);
            require(!invalid.precision_certified);
        }
        ++cases;
    }
    std::cout << "FAST_SLOPE_CERTIFICATE_PASS " << cases << '\n';
}
