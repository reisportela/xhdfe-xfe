#include <boost/math/distributions/students_t.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iomanip>
#include <iostream>

int main() {
    using Real = boost::multiprecision::cpp_dec_float_100;
    for (const char* df_text : {"100000", "1000000"}) {
        const Real df(df_text);
        const boost::math::students_t_distribution<Real> distribution(df);
        for (const char* target_text : {
                 "1e-20", "1e-50", "1e-100", "1e-300"}) {
            const Real target(target_text);
            const Real quantile = boost::math::quantile(
                boost::math::complement(distribution, target / 2));
            const Real achieved = 2 * boost::math::cdf(
                boost::math::complement(distribution, quantile));
            std::cout << df_text << ',' << target_text << ','
                      << std::setprecision(90) << quantile << ','
                      << achieved / target << '\n';
        }
    }
}
