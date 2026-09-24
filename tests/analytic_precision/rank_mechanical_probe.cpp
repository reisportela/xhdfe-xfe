#include "ols_precision.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

// Independent linear-dependence fixtures; output permits exact comparison
// with the previous identified library under the same compiler/ABI flags.
int main() {
    for (int n : {128, 4099}) {
        Eigen::MatrixXd X(n, 5);
        Eigen::VectorXd weights(n);
        for (int i=0;i<n;++i) {
            const double a=(i&1) ? 1.0 : -1.0;
            const double b=(i&2) ? 1.0 : -1.0;
            const double c=(i&4) ? 1.0 : -1.0;
            X(i,0)=a;X(i,1)=b;X(i,2)=a+b;X(i,3)=a-b;X(i,4)=c;
            weights[i]=1.0+(i%3);
        }
        for (bool center : {false,true}) for (bool weighted : {false,true})
        for (int threads : {1,2,8,16,24,48}) {
#ifdef HDFE_USE_OPENMP
            omp_set_dynamic(0);omp_set_num_threads(threads);
#endif
            for (int fixture=0;fixture<5;++fixture) {
                Eigen::MatrixXd design=X;
                std::vector<int> columns{0,1,2,3,4},priority;
                std::vector<std::uint8_t> expected{0,0,1,1,0};
                if (fixture==1) {
                    priority={0,0,0,1,0};expected={0,1,1,0,0};
                } else if (fixture==2) {
                    columns={4,0,1};expected={0,0,0};
                } else if (fixture==3) {
                    design.col(1)=X.col(0)+std::ldexp(1.0,-24)*X.col(1);
                    columns={0,1,4};expected={0,0,0};
                } else if (fixture==4) {
                    design.col(1)=X.col(0);columns={0,1,4};expected={0,1,0};
                }
                if (center) design.array()+=std::ldexp(1.0,16);
                hdfe::detail::ParallelWorkObserver observer;
                const auto got=hdfe::detail::precise_ols_rank_mask(
                    design,columns,weighted ? &weights : nullptr,center,1e-10,
                    priority,threads,&observer);
                if (got!=expected) {
                    std::cerr<<"independent rank fixture failed "<<n<<' '<<fixture<<' '
                             <<center<<' '<<weighted<<' '<<threads<<'\n';return 1;
                }
                std::cout<<n<<' '<<fixture<<' '<<center<<' '<<weighted<<' '<<threads<<' ';
                for (auto value:got) std::cout<<int(value);
                std::cout<<'\n';
            }
        }
    }
    return 0;
}
