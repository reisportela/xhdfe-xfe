#include "fe_absorption.hpp"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    using namespace hdfe;
    using namespace hdfe::detail;
    constexpr int n = 128;
    Eigen::VectorXd y(n), z1(n), z2(n);
    std::vector<Eigen::VectorXi> fes(2, Eigen::VectorXi(n));
    std::vector<HeterogeneousSlopeTerm> slopes(2);
    for (int i=0; i<n; ++i) {
        fes[0][i] = fes[1][i] = i/16;
        z1[i] = (i%8)/8.0;
        z2[i] = ((i*3)%13)/8.0;
        // Exactly represented dyadics; analytic within residual is zero.
        y[i] = 2.0 + 3.0*z1[i] - 2.0*z2[i];
    }
    slopes[0].fe_index=0; slopes[0].include_intercept=true; slopes[0].values=z1;
    slopes[1].fe_index=1; slopes[1].include_intercept=false; slopes[1].values=z2;
    HdfeOptions options;
    options.tol=1e-8; options.num_threads=2; options.num_threads_explicit=true;
    int failures=0;
    for (const auto mode : {ToleranceMode::ReghdfeComparable, ToleranceMode::XhdfeFast}) {
    options.tolerance_mode=mode;
    // Kinds 8 and 9 repeat the regressor and weighted designs with a material
    // error (2^-20): the aggregate certificate must refuse them in both modes.
    for (int kind=0; kind<10; ++kind) {
        const bool regressor = kind==2 || kind==8;
        const bool weighted = kind==5 || kind==9;
        Eigen::MatrixXd X=Eigen::MatrixXd::Ones(n,regressor ? 1 : 0);
        Eigen::VectorXd weights=Eigen::VectorXd::Ones(n);
        AbsorptionResult result;
        result.converged=kind!=3;
        result.y_tilde=std::ldexp(1.0, (kind==1 || kind>=8) ? -20 : (kind==6 ? -42 : (kind==7 ? -27 : -52)))*y;
        result.X_tilde=Eigen::MatrixXd::Zero(n,X.cols());
        if(kind==4) result.y_tilde[0]=std::numeric_limits<double>::quiet_NaN();
        certify_absorption_result(y,X,fes,weighted ? &weights : nullptr,
                                  options,slopes,result);
        // Fast accepts on the historical aggregate certificate alone (2.29.0,
        // PR #13): the exact projections with a regressor (2) or weights (5)
        // are no longer refused by the comparable-only block certificate.
        const bool expected=kind==0 || kind==6 ||
            (mode==ToleranceMode::XhdfeFast && (kind==2 || kind==5 || kind==7));
        const bool pass=result.precision_certified==expected;
        failures+=!pass;
        std::cout << "mode=" << (mode==ToleranceMode::XhdfeFast ? "fast" : "comparable")
                  << " case=" << kind << " expected=" << expected
                  << " certified=" << result.precision_certified
                  << " raw_block=" << result.slope_block_residual_rel
                  << " direct_error=" << result.y_tilde.norm()/y.norm()
                  << " pass=" << pass << '\n';
    }
    }
    return failures ? 1 : 0;
}
