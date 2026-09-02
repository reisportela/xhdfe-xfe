#include <boost/math/distributions/students_t.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iomanip>
#include <iostream>

int main() {
    using Real = boost::multiprecision::cpp_dec_float_100;
    const Real df("100000");
    const boost::math::students_t_distribution<Real> distribution(df);
    const Real t("37.794586197174261");
    const Real probability = 2 * boost::math::cdf(
        boost::math::complement(distribution, t));
    std::cout << std::setprecision(90)
              << "fixed," << t << ',' << probability << '\n';
    for (const char* target_text : {
             "4.940656458412465441765687928682213723650598026143247644255856825006755072702e-324",
             "1e-325"}) {
        const Real target(target_text);
        const Real quantile = boost::math::quantile(
            boost::math::complement(distribution, target / 2));
        std::cout << "target," << target << ',' << quantile << '\n';
    }
}
