// The represented X is full rank although rounded X'X is semidefinite.
// Eigen LDLT can report isPositive()==true and rcond()==0.5 for rank-one
// [[1,1],[1,1]]. This must route to the existing accurate OLS solve.
#include "ols.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

int main() {
    constexpr int n=128;
    constexpr double delta=0x1p-27;
    Eigen::MatrixXd x(n,2);
    Eigen::VectorXd y(n), error(n);
    for(int i=0;i<n;++i) {
        const double a=2*(i%2)-1;
        const double b=2*((i/2)%2)-1;
        error[i]=2*((i/4)%2)-1;
        x(i,0)=a;
        x(i,1)=a+delta*b;
        y[i]=2*a+3*x(i,1)+error[i];
    }
    // Analytic truth: beta=(2,3), residual=e, RSS=128, df=126.
    // V = [[delta^-2+1,-delta^-2],[-delta^-2,delta^-2]] / 126.
    const double reciprocal_delta2=0x1p54;
    Eigen::MatrixXd reference(2,2);
    reference << (reciprocal_delta2+1.0)/(n-2), -reciprocal_delta2/(n-2),
                 -reciprocal_delta2/(n-2), reciprocal_delta2/(n-2);
    int failed=0;
    for(int threads : {1,2,16}) {
#ifdef HDFE_USE_OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#endif
        for(bool general : {false,true}) {
            hdfe::detail::ParallelWorkObserver observer;
            try {
                const auto result=hdfe::detail::run_ols(y,x,nullptr,nullptr,
                    hdfe::StandardErrorType::Homoskedastic,y.squaredNorm(),y.squaredNorm(),
                    n,false,general ? &x : nullptr,true,&observer);
                double beta_error=std::max(std::abs(result.coefficients[0]-2)/2,
                                           std::abs(result.coefficients[1]-3)/3);
                double covariance_error=0;
                for(int j=0;j<2;++j) for(int k=0;k<2;++k)
                    covariance_error=std::max(covariance_error,
                        std::abs(result.covariance(j,k)-reference(j,k))/
                        std::sqrt(reference(j,j)*reference(k,k)));
                const double residual_error=(result.residuals-error).cwiseAbs().maxCoeff();
                const double rss_error=std::abs(result.rss-n);
                const bool finite=hdfe::detail::ieee_all_finite(result.coefficients) &&
                    hdfe::detail::ieee_all_finite(result.covariance) &&
                    hdfe::detail::ieee_all_finite(result.residuals) &&
                    hdfe::detail::ieee_finite(result.rss);
                const bool pass=finite && beta_error<=1e-9 && covariance_error<=1e-8 &&
                    residual_error<=1e-7 && rss_error<=1e-8 && result.df_resid==n-2;
                failed+=!pass;
                std::cout << std::setprecision(17)
                          << "threads=" << threads << " general=" << general
                          << " beta=" << result.coefficients.transpose()
                          << " beta_error=" << beta_error << " V_error=" << covariance_error
                          << " residual_error=" << residual_error << " RSS_error=" << rss_error
                          << " status=" << (pass ? "PASS" : "FAIL") << '\n';
            } catch(const std::exception& exception) {
                ++failed;
                std::cout << "threads=" << threads << " general=" << general
                          << " status=FAIL error=" << exception.what() << '\n';
            }
        }
    }
    return failed ? 1 : 0;
}
