#include "ordinary_precision.hpp"
#include <cmath>
#include <iostream>

int main() {
    using namespace hdfe;
    using namespace hdfe::detail;
    int checked=0;
    for(int n:{16,128,4096}) for(double scale:{std::ldexp(1.,-40),1.,std::ldexp(1.,40)})
    for(bool weighted:{false,true}) for(bool strict:{false,true}) {
        Eigen::VectorXd y(n),a(n),e(n),weights(n);
        Eigen::MatrixXd X(n,1);Eigen::VectorXi fe(n);
        AbsorptionResult candidate;candidate.y_tilde.resize(n);candidate.X_tilde.resize(n,1);
        for(int i=0;i<n;++i) {
            fe[i]=i/4;const double g=fe[i]%2 ? 1 : -1;
            a[i]=i%2 ? 1 : -1;e[i]=(i/2)%2 ? 1 : -1;
            X(i,0)=scale*(std::ldexp(g,45)+a[i]);
            y[i]=scale*(std::ldexp(g,45)+.75*a[i]+.125*e[i]);
            candidate.X_tilde(i,0)=scale*(a[i]+g);
            candidate.y_tilde[i]=scale*(.75*a[i]+2*g+.125*e[i]);
            weights[i]=1+fe[i]%3;
        }
        HdfeOptions options;options.num_threads=2;options.max_iter=1000;
        options.tol=strict ? 1e-12 : 1e-8;
        options.tolerance_mode=strict ? ToleranceMode::StrictResidual : ToleranceMode::ReghdfeComparable;
        const auto* w=weighted ? &weights : nullptr;
        // The original-data Frobenius diagnostic can accept this wrong
        // projection; both differences are nevertheless exactly in the FE span.
        certify_absorption_result(y,X,{fe},w,options,{},candidate);
        if(!candidate.precision_certified) return 1;
        if(check_ordinary_precision({fe},w,options,candidate).passed) return 2;
        auto fixed=enforce_ordinary_precision(y,X,{fe},w,options,
            AbsorptionMethod::GaussSeidel,std::move(candidate),{});
        if(!fixed.converged || !fixed.precision_certified ||
           !check_ordinary_precision({fe},w,options,fixed).passed || fixed.iterations>options.max_iter) return 3;
        if((fixed.X_tilde.col(0)/scale-a).cwiseAbs().maxCoeff()>1e-11 ||
           (fixed.y_tilde/scale-.75*a-.125*e).cwiseAbs().maxCoeff()>1e-11) return 4;
        ++checked;
    }
    std::cout<<"ORDINARY_PRECISION_AND_CONTINUATION_PASS "<<checked<<'\n';
}
