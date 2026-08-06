#ifndef HDFE_IEEE_BITS_HPP
#define HDFE_IEEE_BITS_HPP

#include <cstdint>
#include <cstring>
#include <limits>

#include <Eigen/Core>

namespace hdfe {
namespace detail {

namespace ieee_bits_detail {

constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
constexpr std::uint64_t kFractionMask = UINT64_C(0x000fffffffffffff);

inline std::uint64_t bits(double value) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "xhdfe requires 64-bit IEEE-754 doubles");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "xhdfe requires IEEE-754 binary64 semantics");
    std::uint64_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

}  // namespace ieee_bits_detail

// These checks deliberately inspect the IEEE-754 representation. Production
// translation units use -ffast-math, under which std::isfinite/isnan/isinf and
// Eigen's allFinite()/hasNaN() may be constant-folded under a finite-only
// assumption and therefore cannot serve as fail-closed guards.
inline bool ieee_finite(double value) noexcept {
    return (ieee_bits_detail::bits(value) &
            ieee_bits_detail::kExponentMask) !=
           ieee_bits_detail::kExponentMask;
}

inline bool ieee_isnan(double value) noexcept {
    const std::uint64_t raw = ieee_bits_detail::bits(value);
    return (raw & ieee_bits_detail::kExponentMask) ==
               ieee_bits_detail::kExponentMask &&
           (raw & ieee_bits_detail::kFractionMask) != 0;
}

inline bool ieee_isinf(double value) noexcept {
    const std::uint64_t raw = ieee_bits_detail::bits(value);
    return (raw & ieee_bits_detail::kExponentMask) ==
               ieee_bits_detail::kExponentMask &&
           (raw & ieee_bits_detail::kFractionMask) == 0;
}

inline bool ieee_all_finite(const double* values, Eigen::Index size) noexcept {
    if (values == nullptr && size != 0) {
        return false;
    }
    std::uint64_t invalid = 0;
    for (Eigen::Index i = 0; i < size; ++i) {
        invalid |= static_cast<std::uint64_t>(!ieee_finite(values[i]));
    }
    return invalid == 0;
}

template <typename Derived>
inline bool ieee_all_finite(const Eigen::DenseBase<Derived>& values) noexcept {
    std::uint64_t invalid = 0;
    for (Eigen::Index col = 0; col < values.cols(); ++col) {
        for (Eigen::Index row = 0; row < values.rows(); ++row) {
            invalid |= static_cast<std::uint64_t>(
                !ieee_finite(values.derived().coeff(row, col)));
        }
    }
    return invalid == 0;
}

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_IEEE_BITS_HPP
