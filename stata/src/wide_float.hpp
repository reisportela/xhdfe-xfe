#ifndef HDFE_WIDE_FLOAT_HPP
#define HDFE_WIDE_FLOAT_HPP

#include "hdfe/ieee_bits.hpp"
#include <cmath>
#include <limits>

namespace hdfe {
namespace detail {

#if defined(__clang__)
// Clang may discard the low product under fast-math even with volatile
// intermediates. Keep error-free transforms precise, including when inlined.
#pragma float_control(precise, on, push)
#endif

// Protected error-free transforms retain about 106 significand bits without
// a platform-specific quadmath dependency. Volatile operations prevent the
// target's fast-math flags from cancelling the compensation terms.
struct OlsWide {
    double hi=0.0,lo=0.0;
    OlsWide(double value=0.0):hi(value) {}
    static double add(double a,double b) {volatile double v=a+b;return v;}
    static double sub(double a,double b) {volatile double v=a-b;return v;}
    static double mul(double a,double b) {volatile double v=a*b;return v;}
    static double div(double a,double b) {volatile double v=a/b;return v;}
#if defined(__SIZEOF_INT128__)
    // MinGW's software fma loses bits in the product residual. Form that
    // residual from the exact binary64 significands instead. This replaces
    // one arithmetic operation; it adds no precision check or solve work.
    static double exact_product_error(double a,double b,double p) {
        const auto ab=ieee_bits_detail::bits(a), bb=ieee_bits_detail::bits(b);
        const auto pb=ieee_bits_detail::bits(p);
        const int ap=static_cast<int>((ab>>52)&0x7ff), bp=static_cast<int>((bb>>52)&0x7ff);
        const int pp=static_cast<int>((pb>>52)&0x7ff);
        if(pp==0x7ff) return std::fma(a,b,-p);
        // A correctly rounded subnormal product has a residual of at most
        // half the smallest subnormal, which rounds to zero in binary64.
        if(pp==0) return 0.0;
        const int ae=ap ? ap-1075 : -1074, be=bp ? bp-1075 : -1074;
        const int shift=pp-1075-ae-be;
        if(shift<=0) return 0.0; // the exact product fits in 53 bits
        constexpr std::uint64_t mask=UINT64_C(0x000fffffffffffff);
        constexpr std::uint64_t hidden=UINT64_C(1)<<52;
        const std::uint64_t am=(ab&mask)|(ap ? hidden : 0);
        const std::uint64_t bm=(bb&mask)|(bp ? hidden : 0);
        using Integer=unsigned __int128;
        const Integer product=Integer(am)*bm;
        const Integer rounded=Integer((pb&mask)|hidden)<<shift;
        // For a finite normal p=a*b, shift<=53 and the difference has at
        // most 53 bits, so its binary64 conversion is exact.
        double error=product>=rounded ? static_cast<double>(static_cast<std::uint64_t>(product-rounded))
                                     : -static_cast<double>(static_cast<std::uint64_t>(rounded-product));
        if((ab^bb)>>63) error=-error;
        return std::ldexp(error,ae+be);
    }
#endif
    static OlsWide pair(double a,double b) {
        OlsWide r;r.hi=add(a,b);const double v=sub(r.hi,a);
        r.lo=add(sub(a,sub(r.hi,v)),sub(b,v));return r;
    }
    double value() const {return add(hi,lo);}
    bool finite() const {return ieee_finite(hi) && ieee_finite(lo);}
    OlsWide operator-() const {OlsWide r;r.hi=-hi;r.lo=-lo;return r;}
    OlsWide operator+(const OlsWide& b) const {
        auto s=pair(hi,b.hi);const auto t=pair(lo,b.lo);
        s=pair(s.hi,add(s.lo,t.hi));return pair(s.hi,add(s.lo,t.lo));
    }
    OlsWide operator-(const OlsWide& b) const {return *this+(-b);}
    OlsWide operator*(const OlsWide& b) const {
        const double p=mul(hi,b.hi);
#if defined(__MINGW32__) && defined(__SIZEOF_INT128__) && !defined(__FMA__) && !defined(__ARM_FEATURE_FMA) && !defined(__FP_FAST_FMA)
        volatile double product_error=exact_product_error(hi,b.hi,p);
#elif defined(__clang__) && !defined(__FMA__) && !defined(__ARM_FEATURE_FMA) && !defined(__FP_FAST_FMA)
        // Without a target FMA instruction, older Clang can lower even a
        // precise builtin to an unfused product/add. Keep the intended
        // library call explicit; hardware-FMA targets retain their intrinsic.
        static double (*volatile library_fma)(double,double,double)=std::fma;
        volatile double product_error=library_fma(hi,b.hi,-p);
#else
        volatile double product_error=std::fma(hi,b.hi,-p);
#endif
        double e=add(product_error,add(mul(hi,b.lo),mul(lo,b.hi)));
        return pair(p,add(e,mul(lo,b.lo)));
    }
    OlsWide operator/(const OlsWide& b) const {
        OlsWide q(div(hi,b.hi));
        q=q+OlsWide(div((*this-b*q).hi,b.hi));
        return q+OlsWide(div((*this-b*q).hi,b.hi));
    }
    OlsWide& operator+=(const OlsWide& b) {*this=*this+b;return *this;}
    OlsWide& operator-=(const OlsWide& b) {*this=*this-b;return *this;}
};

inline OlsWide& operator*=(OlsWide& a, const OlsWide& b) { a = a * b; return a; }
inline OlsWide& operator/=(OlsWide& a, const OlsWide& b) { a = a / b; return a; }
inline bool operator==(const OlsWide& a, const OlsWide& b) {
    return a.hi == b.hi && a.lo == b.lo;
}
inline bool operator!=(const OlsWide& a, const OlsWide& b) { return !(a == b); }
inline bool operator<(const OlsWide& a, const OlsWide& b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool operator>(const OlsWide& a, const OlsWide& b) { return b < a; }
inline bool operator<=(const OlsWide& a, const OlsWide& b) { return a < b || a == b; }
inline bool operator>=(const OlsWide& a, const OlsWide& b) { return b <= a; }
inline OlsWide abs(const OlsWide& value) { return value < OlsWide(0) ? -value : value; }
inline OlsWide abs2(const OlsWide& value) { return value * value; }
inline OlsWide conj(const OlsWide& value) { return value; }
inline OlsWide real(const OlsWide& value) { return value; }
inline OlsWide imag(const OlsWide&) { return OlsWide(0); }
inline bool isfinite(const OlsWide& value) { return value.finite(); }
inline bool isnan(const OlsWide& value) { return ieee_isnan(value.hi) || ieee_isnan(value.lo); }
inline bool isinf(const OlsWide& value) {
    return !isnan(value) && (ieee_isinf(value.hi) || ieee_isinf(value.lo));
}

inline OlsWide sqrt(const OlsWide& value) {
    if (isnan(value) || value < OlsWide(0))
        return OlsWide(std::numeric_limits<double>::quiet_NaN());
    if (isinf(value) || value == OlsWide(0)) return value;
    // Exact binary scaling keeps the Newton products away from overflow.
    int exponent = 0;
    std::frexp(value.hi, &exponent);
    if (exponent % 2 != 0) --exponent;
    OlsWide scaled;
    scaled.hi = std::ldexp(value.hi, -exponent);
    scaled.lo = std::ldexp(value.lo, -exponent);
    OlsWide root(std::sqrt(scaled.hi));
    root += (scaled - root * root) / (root + root);
    root += (scaled - root * root) / (root + root);
    OlsWide result;
    result.hi = std::ldexp(root.hi, exponent / 2);
    result.lo = std::ldexp(root.lo, exponent / 2);
    return result;
}

#if defined(__clang__)
#pragma float_control(pop)
#endif

} // namespace detail
} // namespace hdfe

namespace std {
template<> class numeric_limits<hdfe::detail::OlsWide> : public numeric_limits<double> {
    using Wide = hdfe::detail::OlsWide;
public:
    static constexpr bool is_iec559 = false;
    static constexpr int digits = 106;
    static constexpr int digits10 = 31;
    static constexpr int max_digits10 = 33;
    static Wide min() noexcept { return Wide(numeric_limits<double>::min()); }
    static Wide max() noexcept { return Wide(numeric_limits<double>::max()); }
    static Wide lowest() noexcept { return Wide(numeric_limits<double>::lowest()); }
    static Wide epsilon() noexcept { return Wide(0x1p-105); }
    static Wide round_error() noexcept { return Wide(0.5); }
    static Wide infinity() noexcept { return Wide(numeric_limits<double>::infinity()); }
    static Wide quiet_NaN() noexcept { return Wide(numeric_limits<double>::quiet_NaN()); }
    static Wide signaling_NaN() noexcept { return Wide(numeric_limits<double>::signaling_NaN()); }
    static Wide denorm_min() noexcept { return Wide(numeric_limits<double>::denorm_min()); }
};
} // namespace std

namespace Eigen {
template<> struct NumTraits<hdfe::detail::OlsWide> : GenericNumTraits<hdfe::detail::OlsWide> {
    using Real = hdfe::detail::OlsWide;
    using NonInteger = Real;
    using Nested = Real;
    using Literal = Real;
    enum { IsComplex = 0, IsInteger = 0, IsSigned = 1,
           RequireInitialization = 1, ReadCost = 2,
           AddCost = HugeCost, MulCost = HugeCost };
    static Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
    static Real dummy_precision() { return epsilon(); }
    static Real highest() { return std::numeric_limits<Real>::max(); }
    static Real lowest() { return std::numeric_limits<Real>::lowest(); }
    static int digits() { return std::numeric_limits<Real>::digits; }
    static int digits10() { return std::numeric_limits<Real>::digits10; }
};
namespace internal {
template<> struct cast_impl<hdfe::detail::OlsWide, double> {
    static double run(const hdfe::detail::OlsWide& value) { return value.value(); }
};
} // namespace internal
} // namespace Eigen

#endif
