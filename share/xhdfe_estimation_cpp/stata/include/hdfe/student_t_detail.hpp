#pragma once

#include "hdfe/ieee_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace hdfe::detail::student_t_internal {

using WorkingReal = std::conditional_t<
    (std::numeric_limits<long double>::digits >
     std::numeric_limits<double>::digits),
    long double,
    double>;

constexpr int kTailRelativeToleranceFactor = 16384;
constexpr double kExactTinyTailThreshold =
    1.490116119384765625e-8;  // sqrt(DBL_EPSILON)
constexpr double kExactTinyTailMaxDf = 1e6;
constexpr double kLargeDfInverseTailFloor =
    std::numeric_limits<double>::epsilon() / 2.0;

template <typename Real>
inline bool finite_value(Real value) {
    const Real maximum = std::numeric_limits<Real>::max();
    return value >= -maximum && value <= maximum;
}

template <typename Real>
inline Real tail_relative_scale(Real target) {
    if (!(target >= std::numeric_limits<Real>::min()) ||
        !finite_value(target)) {
        throw std::runtime_error("Student-t tail target is invalid");
    }
    const Real relative_scale =
        Real(kTailRelativeToleranceFactor) *
        static_cast<Real>(std::numeric_limits<double>::epsilon());
    if (!(relative_scale > Real(0)) || !finite_value(relative_scale)) {
        throw std::runtime_error(
            "Student-t relative tail scale is invalid");
    }
    return relative_scale;
}

template <typename Real>
inline void require_relative_tail_postcondition(
    Real observed, Real target, const char* label) {
    const Real relative_error = std::abs(observed / target - Real(1));
    if (!(observed >= Real(0) && observed <= Real(1)) ||
        !finite_value(observed) || !finite_value(relative_error) ||
        relative_error > tail_relative_scale(target)) {
        throw std::runtime_error(label);
    }
}

template <typename Real>
inline Real continued_fraction(Real a, Real b, Real x) {
    static_assert(std::is_floating_point_v<Real>);
    constexpr int kMaxIterations = 10000;
    const Real epsilon = Real(8) * std::numeric_limits<Real>::epsilon();
    const Real fp_min = std::numeric_limits<Real>::min() / epsilon;
    const Real qab = a + b;
    const Real qap = a + Real(1);
    const Real qam = a - Real(1);
    Real c = Real(1);
    Real d = Real(1) - qab * x / qap;
    if (std::abs(d) < fp_min) {
        d = fp_min;
    }
    d = Real(1) / d;
    Real h = d;
    for (int m = 1; m <= kMaxIterations; ++m) {
        const int m2 = 2 * m;
        Real aa = Real(m) * (b - Real(m)) * x /
                  ((qam + Real(m2)) * (a + Real(m2)));
        d = Real(1) + aa * d;
        if (std::abs(d) < fp_min) {
            d = fp_min;
        }
        c = Real(1) + aa / c;
        if (std::abs(c) < fp_min) {
            c = fp_min;
        }
        d = Real(1) / d;
        h *= d * c;

        aa = -(a + Real(m)) * (qab + Real(m)) * x /
             ((a + Real(m2)) * (qap + Real(m2)));
        d = Real(1) + aa * d;
        if (std::abs(d) < fp_min) {
            d = fp_min;
        }
        c = Real(1) + aa / c;
        if (std::abs(c) < fp_min) {
            c = fp_min;
        }
        d = Real(1) / d;
        const Real delta = d * c;
        h *= delta;
        if (std::abs(delta - Real(1)) <= epsilon) {
            if (!(h > Real(0))) {
                throw std::runtime_error(
                    "Student-t incomplete-beta fraction is nonpositive");
            }
            return h;
        }
    }
    throw std::runtime_error(
        "Student-t incomplete-beta continued fraction did not converge");
}

template <typename Real>
inline Real asymptotic_log_gamma_half_ratio(Real z) {
    // log(Gamma(z + 1/2) / Gamma(z)); the odd inverse-power expansion
    // avoids subtracting two O(z log z) lgamma values at large z.
    const Real inverse = Real(1) / z;
    const Real inverse2 = inverse * inverse;
    return Real(0.5) * std::log(z) + inverse * (
        Real(-0.125) + inverse2 * (
            Real(1) / Real(192) + inverse2 * (
                Real(-1) / Real(640) + inverse2 * (
                    Real(17) / Real(14336) +
                    inverse2 * (Real(-31) / Real(18432))))));
}

template <typename Real>
inline Real stable_log_gamma_half_ratio(Real z) {
    if (!(z > Real(0)) || !finite_value(z)) {
        throw std::runtime_error("invalid Student-t gamma-ratio input");
    }
    if (z < Real(16)) {
        return std::lgamma(z + Real(0.5)) - std::lgamma(z);
    }
    return asymptotic_log_gamma_half_ratio(z);
}

inline double exp_binary64_preserving_subnormal(double log_value) {
    constexpr double kLogMinNormal =
        -708.39641853226410622441122813025645;
    constexpr double kLogTwo =
        0.693147180559945309417232121458176568;
    if (log_value >= kLogMinNormal) {
        return std::exp(log_value);
    }
    const double scaled = std::exp(log_value + 1074.0 * kLogTwo);
    if (!(scaled >= 0.0) || !hdfe::detail::ieee_finite(scaled)) {
        throw std::runtime_error(
            "Student-t binary64 subnormal scale is invalid");
    }
    const double rounded = std::floor(scaled + 0.5);
    if (!(rounded > 0.0)) {
        return 0.0;
    }
    constexpr std::uint64_t kMinNormalBits = std::uint64_t{1} << 52;
    if (rounded > static_cast<double>(kMinNormalBits)) {
        throw std::runtime_error(
            "Student-t binary64 subnormal mantissa overflowed");
    }
    const std::uint64_t bits = static_cast<std::uint64_t>(rounded);
    double result = 0.0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename Real>
inline Real exp_probability(Real log_value) {
    if (!finite_value(log_value)) {
        throw std::runtime_error("Student-t log probability is invalid");
    }
    if constexpr (
        std::numeric_limits<Real>::digits ==
        std::numeric_limits<double>::digits) {
        return static_cast<Real>(exp_binary64_preserving_subnormal(
            static_cast<double>(log_value)));
    }
    return std::exp(log_value);
}

template <typename Real>
inline Real incomplete_beta_log_normalizer(Real a, Real b) {
    constexpr Real half = Real(0.5);
    if (a >= Real(16) && b == half) {
        return asymptotic_log_gamma_half_ratio(a) - std::lgamma(half);
    }
    if (b >= Real(16) && a == half) {
        return asymptotic_log_gamma_half_ratio(b) - std::lgamma(half);
    }
    return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
}

template <typename Real>
inline Real regularized_incomplete_beta(Real a, Real b, Real x) {
    if (!(a > Real(0) && b > Real(0) && x >= Real(0) && x <= Real(1))) {
        throw std::runtime_error("invalid Student-t incomplete-beta input");
    }
    if (x == Real(0)) {
        return Real(0);
    }
    if (x == Real(1)) {
        return Real(1);
    }
    const Real log_front = incomplete_beta_log_normalizer(a, b) +
                           a * std::log(x) + b * std::log1p(-x);
    const Real threshold = (a + Real(1)) / (a + b + Real(2));
    Real result;
    if (x < threshold) {
        const Real fraction = continued_fraction(a, b, x);
        result = exp_probability(
            log_front + std::log(fraction) - std::log(a));
    } else {
        const Real fraction = continued_fraction(b, a, Real(1) - x);
        const Real log_complement =
            log_front + std::log(fraction) - std::log(b);
        result = -std::expm1(log_complement);
    }
    if (!(result >= Real(0) && result <= Real(1))) {
        throw std::runtime_error("Student-t incomplete-beta result is invalid");
    }
    return result;
}

template <typename Real>
inline Real regularized_incomplete_beta_power_series(
    Real a, Real b, Real x, Real log_x) {
    constexpr int kMaxSeriesIterations = 1000000;
    const Real epsilon = Real(8) * std::numeric_limits<Real>::epsilon();
    Real term = Real(1);
    Real sum = Real(1);
    Real compensation = Real(0);
    bool converged = false;
    for (int n = 1; n <= kMaxSeriesIterations; ++n) {
        const Real index = Real(n);
        term *= ((a + index - Real(1)) / (a + index)) *
                ((Real(1) - b + index - Real(1)) / index) * x;
        if (!(term >= Real(0)) || !finite_value(term)) {
            throw std::runtime_error(
                "Student-t incomplete-beta series term is invalid");
        }
        const Real corrected = term - compensation;
        const Real next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if (!(sum > Real(0)) || !finite_value(sum)) {
            throw std::runtime_error(
                "Student-t incomplete-beta series sum is invalid");
        }
        if (term <= epsilon * sum) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw std::runtime_error(
            "Student-t incomplete-beta power series did not converge");
    }
    const Real log_result = incomplete_beta_log_normalizer(a, b) +
                            a * log_x - std::log(a) + std::log(sum);
    const Real result = exp_probability(log_result);
    if (!(result >= Real(0) && result <= Real(1)) || !finite_value(result)) {
        throw std::runtime_error(
            "Student-t incomplete-beta series result is invalid");
    }
    return result;
}

template <typename Real>
inline Real asymptotic_quantile_from_normal(Real z, Real df) {
    const Real z2 = z * z;
    const Real z3 = z * z2;
    const Real z5 = z3 * z2;
    const Real z7 = z5 * z2;
    const Real inverse_df = Real(1) / df;
    const Real first = (z3 + z) / Real(4);
    const Real second =
        (Real(5) * z5 + Real(16) * z3 + Real(3) * z) / Real(96);
    const Real third =
        (Real(3) * z7 + Real(19) * z5 + Real(17) * z3 - Real(15) * z) /
        Real(384);
    return z + first * inverse_df + second * inverse_df * inverse_df +
           third * inverse_df * inverse_df * inverse_df;
}

template <typename Real>
inline Real asymptotic_quantile_derivative(Real z, Real df) {
    const Real z2 = z * z;
    const Real z4 = z2 * z2;
    const Real z6 = z4 * z2;
    const Real inverse_df = Real(1) / df;
    const Real first = (Real(3) * z2 + Real(1)) / Real(4);
    const Real second =
        (Real(25) * z4 + Real(48) * z2 + Real(3)) / Real(96);
    const Real third =
        (Real(21) * z6 + Real(95) * z4 + Real(51) * z2 - Real(15)) /
        Real(384);
    return Real(1) + first * inverse_df + second * inverse_df * inverse_df +
           third * inverse_df * inverse_df * inverse_df;
}

template <typename Real>
struct AsymptoticNormalCoordinateResult {
    Real z;
    bool certified;
};

template <typename Real>
inline AsymptoticNormalCoordinateResult<Real>
asymptotic_normal_coordinate(Real t, Real df) {
    Real z = t;
    bool converged = false;
    for (int iteration = 0; iteration < 16; ++iteration) {
        const Real derivative = asymptotic_quantile_derivative(z, df);
        const Real residual = asymptotic_quantile_from_normal(z, df) - t;
        if (!(derivative > Real(0)) || !finite_value(derivative) ||
            !finite_value(residual)) {
            return {z, false};
        }
        const Real delta = residual / derivative;
        const Real next = z - delta;
        if (!finite_value(delta) || !finite_value(next) || next < Real(0)) {
            return {z, false};
        }
        z = next;
        if (std::abs(delta) <=
            Real(4) * std::numeric_limits<Real>::epsilon() *
                (Real(1) + std::abs(z))) {
            converged = true;
            break;
        }
    }
    const bool certified = converged &&
        std::abs(asymptotic_quantile_from_normal(z, df) - t) <=
            Real(64) * std::numeric_limits<Real>::epsilon() *
                (Real(1) + std::abs(t));
    return {z, certified};
}

template <typename Real>
inline Real normal_two_sided_from_coordinate(Real z) {
    constexpr Real inverse_sqrt_two =
        Real(0.707106781186547524400844362104849039L);
    if (z < Real(8)) {
        return std::erfc(z * inverse_sqrt_two);
    }
    const Real inverse_z2 = Real(1) / (z * z);
    Real term = Real(1);
    Real sum = Real(1);
    Real best_sum = sum;
    Real best_term = std::abs(term);
    for (int n = 1; n <= 256; ++n) {
        const Real next =
            -term * Real(2 * n - 1) * inverse_z2;
        if (!finite_value(next)) {
            throw std::runtime_error("normal tail series term is invalid");
        }
        if (std::abs(next) >= best_term) {
            break;
        }
        sum += next;
        if (!(sum > Real(0)) || !finite_value(sum)) {
            throw std::runtime_error("normal tail series sum is invalid");
        }
        best_sum = sum;
        best_term = std::abs(next);
        term = next;
    }
    constexpr Real half_log_two_over_pi =
        Real(-0.225791352644727432363097614947441071L);
    const Real log_probability = half_log_two_over_pi -
        Real(0.5) * z * z - std::log(z) + std::log(best_sum);
    const Real result = exp_probability(log_probability);
    if (!(result >= Real(0) && result <= Real(1)) ||
        !finite_value(result)) {
        throw std::runtime_error("normal tail probability is invalid");
    }
    return result;
}

template <typename Real>
inline Real exact_two_sided_probability_real(Real t_abs, Real df) {
    const Real scaled = t_abs / std::sqrt(df);
    if (scaled > std::sqrt(std::numeric_limits<Real>::max())) {
        return Real(0);
    }
    const Real ratio = scaled * scaled;
    const Real a = df / Real(2);
    if ((a + Real(1)) * ratio <= Real(1.5)) {
        const Real y = ratio / (Real(1) + ratio);
        return Real(1) - regularized_incomplete_beta(Real(0.5), a, y);
    }
    const Real x = Real(1) / (Real(1) + ratio);
    const Real log_x = -std::log1p(ratio);
    try {
        const Real fraction = continued_fraction(a, Real(0.5), x);
        const Real log_front = incomplete_beta_log_normalizer(
            a, Real(0.5)) + a * log_x +
            Real(0.5) * (std::log(ratio) + log_x);
        const Real result = exp_probability(
            log_front + std::log(fraction) - std::log(a));
        if (!(result >= Real(0) && result <= Real(1)) ||
            !finite_value(result)) {
            throw std::runtime_error(
                "Student-t exact incomplete-beta result is invalid");
        }
        return result;
    } catch (const std::runtime_error&) {
        return regularized_incomplete_beta_power_series(
            a, Real(0.5), x, log_x);
    }
}

template <typename Real>
inline Real two_sided_probability_real(Real t_abs, Real df) {
    if (!(t_abs >= Real(0)) || !(df > Real(0)) ||
        !finite_value(t_abs) || !finite_value(df)) {
        throw std::runtime_error("invalid Student-t probability input");
    }
    if (t_abs == Real(0)) {
        return Real(1);
    }
    constexpr Real pi = Real(3.141592653589793238462643383279502884L);
    if (df == Real(1)) {
        const Real result = t_abs > Real(1)
            ? (Real(2) / pi) * std::atan(Real(1) / t_abs)
            : Real(1) - (Real(2) / pi) * std::atan(t_abs);
        if (!(result >= Real(0) && result <= Real(1)) ||
            !finite_value(result)) {
            throw std::runtime_error("Student-t Cauchy probability is invalid");
        }
        return result;
    }
    // Third-order Cornish-Fisher is fast and sufficiently accurate for
    // ordinary inference. In the audited large-parameter transition envelope
    // [1e5, 1e6], tails below sqrt(DBL_EPSILON) use the exact beta form because
    // absolute accuracy no longer controls relative error. Above 1e6 the
    // third-order remainder shrinks rapidly while the exact power series is
    // ill-conditioned, so retain the asymptotic path.
    if (df >= Real(1e5)) {
        const auto coordinate = asymptotic_normal_coordinate(t_abs, df);
        if (!coordinate.certified) {
            const Real exact = exact_two_sided_probability_real(t_abs, df);
            if (!(exact >= Real(0) && exact <= Real(1)) ||
                !finite_value(exact)) {
                throw std::runtime_error(
                    "Student-t exact fallback probability is invalid");
            }
            return exact;
        }
        const Real approximation = normal_two_sided_from_coordinate(
            coordinate.z);
        if (!(approximation >= Real(0) && approximation <= Real(1)) ||
            !finite_value(approximation)) {
            throw std::runtime_error(
                "Student-t asymptotic probability is invalid");
        }
        if (approximation >= static_cast<Real>(kExactTinyTailThreshold) ||
            df > static_cast<Real>(kExactTinyTailMaxDf)) {
            return approximation;
        }
    }
    return exact_two_sided_probability_real(t_abs, df);
}

template <typename Real>
inline Real normal_ppf_start(Real p) {
    constexpr Real a1 = Real(-3.969683028665376e+01);
    constexpr Real a2 = Real(2.209460984245205e+02);
    constexpr Real a3 = Real(-2.759285104469687e+02);
    constexpr Real a4 = Real(1.383577518672690e+02);
    constexpr Real a5 = Real(-3.066479806614716e+01);
    constexpr Real a6 = Real(2.506628277459239e+00);
    constexpr Real b1 = Real(-5.447609879822406e+01);
    constexpr Real b2 = Real(1.615858368580409e+02);
    constexpr Real b3 = Real(-1.556989798598866e+02);
    constexpr Real b4 = Real(6.680131188771972e+01);
    constexpr Real b5 = Real(-1.328068155288572e+01);
    constexpr Real c1 = Real(-7.784894002430293e-03);
    constexpr Real c2 = Real(-3.223964580411365e-01);
    constexpr Real c3 = Real(-2.400758277161838e+00);
    constexpr Real c4 = Real(-2.549732539343734e+00);
    constexpr Real c5 = Real(4.374664141464968e+00);
    constexpr Real c6 = Real(2.938163982698783e+00);
    constexpr Real d1 = Real(7.784695709041462e-03);
    constexpr Real d2 = Real(3.224671290700398e-01);
    constexpr Real d3 = Real(2.445134137142996e+00);
    constexpr Real d4 = Real(3.754408661907416e+00);
    constexpr Real lower = Real(0.02425);
    constexpr Real upper = Real(1) - lower;
    if (p < lower) {
        const Real q = std::sqrt(Real(-2) * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + Real(1));
    }
    if (p <= upper) {
        const Real q = p - Real(0.5);
        const Real r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + Real(1));
    }
    const Real q = std::sqrt(Real(-2) * std::log1p(-p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + Real(1));
}

template <typename Real>
inline Real density(Real t, Real df) {
    constexpr Real pi = Real(3.141592653589793238462643383279502884L);
    const Real a = df / Real(2);
    const Real scaled = t / std::sqrt(df);
    if (scaled > std::sqrt(std::numeric_limits<Real>::max())) {
        return Real(0);
    }
    const Real log_density = stable_log_gamma_half_ratio(a) -
        Real(0.5) * (std::log(df) + std::log(pi)) -
        Real(0.5) * (df + Real(1)) * std::log1p(scaled * scaled);
    const Real result = std::exp(log_density);
    return finite_value(result) && result >= Real(0) ? result : Real(0);
}

template <typename Real>
inline Real normal_start_from_two_sided_target(Real target) {
    if (!(target > Real(0) && target <= Real(1)) || !finite_value(target)) {
        throw std::runtime_error("invalid Student-t two-sided target");
    }
    if (target == Real(1)) {
        return Real(0);
    }
    const Real tail = target / Real(2);
    Real start;
    if (tail > std::numeric_limits<Real>::epsilon()) {
        const Real upper_probability = Real(1) - tail;
        if (!(upper_probability > Real(0.5) &&
              upper_probability < Real(1))) {
            throw std::runtime_error("Student-t normal start is invalid");
        }
        start = normal_ppf_start(upper_probability);
    } else {
        start = std::sqrt(Real(-2) * std::log(tail));
    }
    if (!(start > Real(0)) || !finite_value(start)) {
        throw std::runtime_error("Student-t normal start is non-finite");
    }
    return start;
}

template <typename Real>
inline Real normal_two_sided_probability(Real z) {
    const Real result = normal_two_sided_from_coordinate(z);
    if (!(result >= Real(0) && result <= Real(1)) || !finite_value(result)) {
        throw std::runtime_error("normal two-sided probability is invalid");
    }
    return result;
}

template <typename Real>
inline Real solve_normal_two_sided(Real target) {
    constexpr Real inverse_sqrt_two_pi =
        Real(0.398942280401432677939946059934381868L);
    const Real start = normal_start_from_two_sided_target(target);
    Real low = Real(0);
    Real high = std::max(Real(1), Real(1.1) * start);
    Real high_probability = normal_two_sided_probability(high);
    while (high_probability > target) {
        low = high;
        if (high > std::numeric_limits<Real>::max() / Real(2)) {
            throw std::runtime_error("normal inverse has no finite bracket");
        }
        high *= Real(2);
        if (!finite_value(high)) {
            throw std::runtime_error("normal inverse bracket overflowed");
        }
        high_probability = normal_two_sided_probability(high);
    }
    Real value = std::clamp(start, low, high);
    const Real epsilon = Real(8) * std::numeric_limits<Real>::epsilon();
    bool converged = false;
    for (int iteration = 0; iteration < 256; ++iteration) {
        const Real probability = normal_two_sided_probability(value);
        if (probability > target) {
            low = value;
        } else {
            high = value;
        }
        if (high - low <=
            epsilon * (Real(1) + std::abs(low) + std::abs(high))) {
            converged = true;
            break;
        }
        const Real derivative = Real(2) * inverse_sqrt_two_pi *
                                std::exp(Real(-0.5) * value * value);
        Real candidate = Real(0.5) * (low + high);
        if (derivative > Real(0) && finite_value(derivative)) {
            const Real newton = value + (probability - target) / derivative;
            if (newton > low && newton < high && finite_value(newton)) {
                candidate = newton;
            }
        }
        if (!finite_value(candidate)) {
            throw std::runtime_error("normal inverse iterate is invalid");
        }
        if (!(candidate > low && candidate < high)) {
            const Real next = std::nextafter(low, high);
            if (!(next > low && next <= high) || !finite_value(next)) {
                throw std::runtime_error("normal inverse bracket is invalid");
            }
            // The bracket has reached the platform's effective arithmetic
            // resolution. The strict probability postcondition below remains
            // authoritative.
            converged = true;
            break;
        }
        value = candidate;
    }
    if (!converged) {
        throw std::runtime_error("normal inverse did not converge");
    }
    const Real low_probability = normal_two_sided_probability(low);
    const Real final_high_probability = normal_two_sided_probability(high);
    if (!(low_probability >= target && final_high_probability <= target &&
          low_probability >= final_high_probability)) {
        throw std::runtime_error("normal inverse bracket postcondition failed");
    }
    const Real result = Real(0.5) * (low + high);
    if (!finite_value(result)) {
        throw std::runtime_error("normal inverse postcondition failed");
    }
    require_relative_tail_postcondition(
        normal_two_sided_probability(result), target,
        "normal inverse relative-tail postcondition failed");
    return result;
}

template <typename Real>
inline double nearest_double_quantile(Real value, Real target, Real df) {
    if (!(value >= Real(0)) || !finite_value(value)) {
        throw std::runtime_error("Student-t inverse result is non-finite");
    }
    const double center = static_cast<double>(value);
    if (!hdfe::detail::ieee_finite(center)) {
        throw std::runtime_error("Student-t inverse exceeds binary64 range");
    }
    const auto evaluate = [df](double candidate) {
        if (!(candidate >= 0.0) || !hdfe::detail::ieee_finite(candidate)) {
            throw std::runtime_error(
                "Student-t adjacent-double candidate is invalid");
        }
        const Real probability = two_sided_probability_real(
            static_cast<Real>(candidate), df);
        if (!finite_value(probability)) {
            throw std::runtime_error(
                "Student-t adjacent-double probability is invalid");
        }
        return probability;
    };
    const Real relative_scale = tail_relative_scale(target);
    const auto monotone_within_roundoff = [relative_scale](
        Real left, Real right) {
        if (right == Real(0)) {
            return left >= Real(0);
        }
        const Real ratio = left / right;
        return finite_value(ratio) &&
               ratio >= Real(1) - relative_scale;
    };
    const auto on_or_above_target = [target, relative_scale](Real probability) {
        const Real ratio = probability / target;
        return finite_value(ratio) &&
               ratio >= Real(1) - relative_scale;
    };
    const auto on_or_below_target = [target, relative_scale](Real probability) {
        const Real ratio = probability / target;
        return finite_value(ratio) &&
               ratio <= Real(1) + relative_scale;
    };
    const Real center_probability = evaluate(center);
    double left = center_probability >= target
        ? center
        : std::nextafter(center, -std::numeric_limits<double>::infinity());
    double right = center_probability >= target
        ? std::nextafter(center, std::numeric_limits<double>::infinity())
        : center;
    Real left_probability = evaluate(left);
    Real right_probability = evaluate(right);
    constexpr int kMaxAdjacentSearchSteps = 4096;
    bool bracketed = false;
    for (int step = 0; step < kMaxAdjacentSearchSteps; ++step) {
        if (!monotone_within_roundoff(
                left_probability, right_probability)) {
            throw std::runtime_error(
                "Student-t adjacent-double probabilities are not monotone");
        }
        if (on_or_above_target(left_probability) &&
            on_or_below_target(right_probability)) {
            bracketed = true;
            break;
        }
        if (left_probability > target && right_probability > target) {
            left = right;
            left_probability = right_probability;
            right = std::nextafter(
                right, std::numeric_limits<double>::infinity());
            right_probability = evaluate(right);
        } else if (left_probability < target && right_probability < target) {
            right = left;
            right_probability = left_probability;
            left = std::nextafter(
                left, -std::numeric_limits<double>::infinity());
            left_probability = evaluate(left);
        } else {
            throw std::runtime_error(
                "Student-t adjacent-double bracket is inconsistent");
        }
    }
    if (!bracketed) {
        throw std::runtime_error(
            "Student-t root is not bracketed by adjacent doubles");
    }

    const Real left_error = std::abs(left_probability - target);
    const Real right_error = std::abs(right_probability - target);
    const bool choose_left = left_error <= right_error;
    require_relative_tail_postcondition(
        choose_left ? left_probability : right_probability, target,
        "Student-t inverse relative-tail postcondition failed");
    return choose_left ? left : right;
}

template <typename Real>
inline double positive_quantile_from_two_sided_target(Real target, Real df) {
    if (target == Real(1)) {
        return 0.0;
    }
    if (!(target > Real(0) && target < Real(1)) || !finite_value(target) ||
        !(df > Real(0)) || !finite_value(df)) {
        throw std::runtime_error("Student-t inverse tail target is invalid");
    }
    if (df >= Real(1e5) &&
        target < static_cast<Real>(kLargeDfInverseTailFloor)) {
        throw std::runtime_error(
            "Student-t inverse tail is below the supported public level floor");
    }
    constexpr Real pi = Real(3.141592653589793238462643383279502884L);
    if (df == Real(1)) {
        const Real angle = pi * target / Real(2);
        const Real tangent = std::tan(angle);
        if (!(tangent > Real(0)) || !finite_value(tangent)) {
            throw std::runtime_error("Student-t Cauchy inverse is not representable");
        }
        const Real value = Real(1) / tangent;
        return nearest_double_quantile(value, target, df);
    }
    if (df >= Real(1e5) &&
        target >= static_cast<Real>(kExactTinyTailThreshold)) {
        const Real z = solve_normal_two_sided(target);
        const Real value = asymptotic_quantile_from_normal(z, df);
        if (!finite_value(value) || !(value > Real(0))) {
            throw std::runtime_error("Student-t asymptotic quantile is invalid");
        }
        return nearest_double_quantile(value, target, df);
    }

    Real start = normal_start_from_two_sided_target(target);
    if (df >= Real(1e5)) {
        start = asymptotic_quantile_from_normal(
            solve_normal_two_sided(target), df);
        if (!(start > Real(0)) || !finite_value(start)) {
            throw std::runtime_error(
                "Student-t exact-tail refinement start is invalid");
        }
    }
    Real low = Real(0);
    Real high = std::max(Real(1), Real(1.1) * start);
    Real high_probability = two_sided_probability_real(high, df);
    while (high_probability > target) {
        low = high;
        if (high > std::numeric_limits<Real>::max() / Real(2)) {
            throw std::runtime_error("Student-t inverse has no finite bracket");
        }
        high *= Real(2);
        if (!finite_value(high)) {
            throw std::runtime_error("Student-t inverse bracket overflowed");
        }
        high_probability = two_sided_probability_real(high, df);
    }
    Real value = std::clamp(start, low, high);
    const Real epsilon = Real(8) * std::numeric_limits<Real>::epsilon();
    bool converged = false;
    for (int iteration = 0; iteration < 512; ++iteration) {
        const Real probability = two_sided_probability_real(value, df);
        if (probability > target) {
            low = value;
        } else {
            high = value;
        }
        if (high - low <=
            epsilon * (Real(1) + std::abs(low) + std::abs(high))) {
            converged = true;
            break;
        }
        const Real pdf = density(value, df);
        Real candidate = Real(0.5) * (low + high);
        if (pdf > Real(0) && finite_value(pdf)) {
            const Real newton =
                value + (probability - target) / (Real(2) * pdf);
            if (newton > low && newton < high && finite_value(newton)) {
                candidate = newton;
            }
        }
        if (!finite_value(candidate)) {
            throw std::runtime_error("Student-t inverse iterate is invalid");
        }
        if (!(candidate > low && candidate < high)) {
            const Real next = std::nextafter(low, high);
            if (!(next > low && next <= high) || !finite_value(next)) {
                throw std::runtime_error("Student-t inverse bracket is invalid");
            }
            // nearest_double_quantile performs the final bracket and
            // relative-tail certification after arithmetic stalls here.
            converged = true;
            break;
        }
        value = candidate;
    }
    if (!converged) {
        throw std::runtime_error("Student-t inverse did not converge");
    }
    return nearest_double_quantile(
        Real(0.5) * (low + high), target, df);
}

template <typename Real>
inline double inverse_cdf_with_real(double p_double, double df_double) {
    const Real p = static_cast<Real>(p_double);
    const Real df = static_cast<Real>(df_double);
    if (p_double == 0.5) {
        return 0.0;
    }
    const bool negative = p_double < 0.5;
    const Real target = negative
        ? Real(2) * p
        : Real(2) * (Real(1) - p);
    const double result = positive_quantile_from_two_sided_target(target, df);
    return negative ? -result : result;
}

template <typename Real>
inline double two_sided_probability_with_real(double t_abs, double df) {
    if (!(df > 0.0) || !hdfe::detail::ieee_finite(df) ||
        !(t_abs >= 0.0) || !hdfe::detail::ieee_finite(t_abs)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(two_sided_probability_real(
        static_cast<Real>(t_abs), static_cast<Real>(df)));
}

template <typename Real>
inline double inverse_cdf_checked(double p, double df) {
    if (!(df > 0.0) || !hdfe::detail::ieee_finite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(p > 0.0 && p < 1.0) || !hdfe::detail::ieee_finite(p)) {
        throw std::runtime_error("Student-t inverse requires p in (0, 1)");
    }
    return inverse_cdf_with_real<Real>(p, df);
}

template <typename Real>
inline double inverse_survival_checked(double upper_tail, double df) {
    if (!(df > 0.0) || !hdfe::detail::ieee_finite(df)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(upper_tail > 0.0 && upper_tail <= 0.5) ||
        !hdfe::detail::ieee_finite(upper_tail)) {
        throw std::runtime_error(
            "Student-t inverse survival requires tail in (0, 0.5]");
    }
    return positive_quantile_from_two_sided_target(
        Real(2) * static_cast<Real>(upper_tail), static_cast<Real>(df));
}

inline double two_sided_probability(double t_abs, double df) {
    return two_sided_probability_with_real<WorkingReal>(t_abs, df);
}

inline double inverse_cdf(double p, double df) {
    return inverse_cdf_checked<WorkingReal>(p, df);
}

inline double inverse_survival(double upper_tail, double df) {
    return inverse_survival_checked<WorkingReal>(upper_tail, df);
}

}  // namespace hdfe::detail::student_t_internal
