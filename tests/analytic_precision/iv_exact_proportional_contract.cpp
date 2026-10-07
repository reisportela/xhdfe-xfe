#include "iv_input.hpp"
#include <iostream>
#include <iomanip>
#include <limits>

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::stoi(argv[1]) : 1;
    std::uint64_t a,b,c,d;
    while (std::cin >> std::hex >> a >> b >> c >> d) {
        Eigen::MatrixXd z(2,2);
        std::memcpy(&z(0,0),&a,8); std::memcpy(&z(1,0),&c,8);
        std::memcpy(&z(0,1),&b,8); std::memcpy(&z(1,1),&d,8);
        const auto reduced=hdfe::detail::remove_exact_proportional_instruments(z,1,nullptr);
        std::cout << bool(reduced) << '\n';
    }
    Eigen::MatrixXd z(6000,3);
    for (int i=0;i<6000;++i) {
        z(i,0)=.125*(i-3000); z(i,1)=3*z(i,0); z(i,2)=7*z(i,0);
    }
    hdfe::detail::ParallelWorkObserver observer;
    const auto reduced=hdfe::detail::remove_exact_proportional_instruments(z,threads,&observer);
    if (observer.max_team_size() != threads || observer.active_workers() != threads) return 3;
    if (!reduced || reduced->cols()!=1 || *reduced!=z.leftCols(1)) return 1;
    z(5555,1)=std::nextafter(z(5555,1),std::numeric_limits<double>::infinity());
    const auto distinct=hdfe::detail::remove_exact_proportional_instruments(z,threads,&observer);
    if (!distinct || distinct->cols()!=2 || *distinct!=z.leftCols(2)) return 2;
    std::cerr << "PASS late-row guard and observed team " << threads << "\n";
    return 0;
}
