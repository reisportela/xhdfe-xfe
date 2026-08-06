#include "hdfe/ieee_bits.hpp"

#include <cstdio>

#include <Eigen/Core>

int main() {
    volatile double zero = 0.0;
    volatile double one = 1.0;
    const double nan_value = zero / zero;
    const double inf_value = one / zero;
    const double finite_value = 1.25;

    Eigen::Vector3d values;
    values << finite_value, nan_value, inf_value;

    const bool ok =
        hdfe::detail::ieee_finite(finite_value) &&
        !hdfe::detail::ieee_finite(nan_value) &&
        !hdfe::detail::ieee_finite(inf_value) &&
        hdfe::detail::ieee_isnan(nan_value) &&
        !hdfe::detail::ieee_isnan(inf_value) &&
        hdfe::detail::ieee_isinf(inf_value) &&
        !hdfe::detail::ieee_isinf(nan_value) &&
        !hdfe::detail::ieee_all_finite(values.data(), values.size()) &&
        !hdfe::detail::ieee_all_finite(values);
    if (!ok) {
        std::fprintf(stderr,
                     "FAIL: IEEE non-finite guards were optimized out\n");
        return 1;
    }
    return 0;
}
