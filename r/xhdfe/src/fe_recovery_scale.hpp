#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "hdfe/ieee_bits.hpp"

namespace hdfe {
namespace detail {

// Raise only small response scales. Normal and large signals keep the existing
// absolute work target; power-of-two changes of coordinates are exact.
inline int fe_recovery_scale_exponent(const Eigen::Ref<const Eigen::VectorXd>& partial) {
    if (!partial.size()) return 0;
    const double peak = partial.cwiseAbs().maxCoeff();
    const double root_n = std::sqrt(static_cast<double>(partial.size()));
    if (!(peak > 0.0) || !ieee_finite(peak) || peak >= root_n) return 0;
    const double rms = partial.stableNorm() / root_n;
    if (!(rms > 0.0) || rms >= 1.0) return 0;
    int exponent = 0;
    std::frexp(rms, &exponent);
    return exponent - 1;
}

inline void scale_fe_recovery_vector(Eigen::Ref<Eigen::VectorXd> values, int exponent) {
    // Multiplication by a normal, exactly represented power of two has
    // the same rounded finite result as ldexp. Keep special inputs and
    // extreme scale factors on the existing scalar path.
    if (exponent >= -1022 && exponent <= 1023) {
        bool normal_or_zero = true;
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            std::uint64_t bits = 0;
            const double value = values[i];
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
            const std::uint64_t field = bits & UINT64_C(0x7ff0000000000000);
            if (field == UINT64_C(0x7ff0000000000000) ||
                (field == 0 && magnitude != 0)) {
                normal_or_zero = false;
                break;
            }
        }
        if (normal_or_zero) {
            if (exponent != 0) values.array() *= std::ldexp(1.0, exponent);
            return;
        }
    }
    for (Eigen::Index i = 0; i < values.size(); ++i)
        values[i] = std::ldexp(values[i], exponent);
}

}  // namespace detail
}  // namespace hdfe
