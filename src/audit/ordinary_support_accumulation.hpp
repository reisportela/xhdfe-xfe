#ifndef XHDFE_ORDINARY_SUPPORT_ACCUMULATION_HPP
#define XHDFE_ORDINARY_SUPPORT_ACCUMULATION_HPP

#include "hdfe/parallel_work_observer.hpp"
#include "ols_numerical_certificate.hpp"
#include <cstdint>
#include <cstring>

namespace hdfe { namespace detail {

// A bounded exponent window makes unweighted FP64 sums exact in 128 bits.
// The eligibility scan proves the sum of absolute inputs fits before any
// signed addition. Outside this window the caller must use its wider path.
#if defined(__SIZEOF_INT128__)
struct OrdinaryExactGradientWindow {
    using Integer=__int128;
    int shift=0;
    bool eligible=false;
    template<class Getter> void inspect(int n,Getter value,bool enclose_tail=false) {
        eligible=false;
        if (n<=0) return;
        int low=2048,high=0;
        for (int row=0;row<n;++row) {
            const double x=value(row);std::uint64_t bits;std::memcpy(&bits,&x,8);
            const int exponent=static_cast<int>((bits>>52)&2047);
            if (exponent==2047) return;
            const auto mantissa=(bits&0x000fffffffffffffULL)|(exponent ? 0x0010000000000000ULL : 0);
            if (!mantissa) continue;
            const int position=exponent ? exponent-1 : 0;
            low=std::min(low,position);high=std::max(high,position);
        }
        if (low==2048) {eligible=true;shift=0;return;}
        int count_bits=0;for (unsigned count=static_cast<unsigned>(n-1);count;count>>=1) ++count_bits;
        // Strictly fewer than 127 magnitude bits; all subtree partial sums
        // contain coefficients -1,0,+1 on the original observations.
        eligible=high-low+53+count_bits<=126;shift=low;
        if (!eligible && enclose_tail) {
            shift=std::max(0,high+53+count_bits-126);eligible=true;
        }
    }
    Integer encode(double x,bool* omitted=nullptr) const {
        if (omitted) *omitted=false;
        std::uint64_t bits;std::memcpy(&bits,&x,8);
        const int exponent=static_cast<int>((bits>>52)&2047);
        const auto mantissa=(bits&0x000fffffffffffffULL)|(exponent ? 0x0010000000000000ULL : 0);
        if (!mantissa) return 0;
        const int position=(exponent ? exponent-1 : 0)-shift;
        if (position<0) {
            if (!omitted) throw std::runtime_error("Exact gradient tail requires an explicit error enclosure");
            *omitted=true;return 0;
        }
        const Integer result=static_cast<Integer>(static_cast<unsigned __int128>(mantissa)<<position);
        return bits>>63 ? -result : result;
    }
    OlsProofInterval interval(Integer value) const {
        if (!value) return {};
        volatile long double rounded=static_cast<long double>(value);
        volatile long double scaled=std::ldexp(static_cast<long double>(rounded),shift-1074);
        // Conversion and power-of-two scaling are exact or correctly rounded.
        // Two outward neighbours cover both, including subnormal scaling.
        using I=OlsProofInterval;
        // This also encloses a subnormal result if the runtime flushes it.
        const auto minimum=std::numeric_limits<long double>::min();
        if (std::abs(static_cast<long double>(scaled))<minimum) return {-minimum,minimum};
        return {I::down(I::down(scaled)),I::up(I::up(scaled))};
    }
};
#else
struct OrdinaryExactGradientWindow {
    using Integer=long long;
    bool eligible=false;
    template<class Getter> void inspect(int,Getter,bool=false) {}
    Integer encode(double,bool* omitted=nullptr) const {if (omitted) *omitted=false;return 0;}
    OlsProofInterval interval(Integer) const {return {};}
};
#endif

// Positive sums need a relative roundoff enclosure, not directed rounding
// at every row. All operations that matter are materialized by volatile.
// The deliberately conservative gamma uses epsilon=2u and includes a
// normal-minimum allowance for gradual-underflow/flush-to-zero arithmetic.
struct OrdinaryPositiveBound {
    using Real=long double;
    Real sum=0;
    std::uint64_t count=0;
    bool valid=true;
    void add_square_divisor(Real upper,Real divisor) {
        volatile Real square=upper*upper,term=square/divisor,next=sum+term;
        sum=next;++count;
        valid=valid && divisor>=1 && sum>=0 && sum<=std::numeric_limits<Real>::max();
    }
    void add_affine(Real weight,Real y,Real residual,Real a,Real b,Real c) {
        volatile Real q1=y-residual,q2=q1-a,q3=q2+b,q=q3-c;
        volatile Real m1=std::abs(y)+std::abs(residual),m2=m1+std::abs(a);
        volatile Real m3=m2+std::abs(b),m=m3+std::abs(c);
        constexpr Real eps=std::numeric_limits<Real>::epsilon();
        constexpr Real floor=std::numeric_limits<Real>::min();
        volatile Real allowance=32*eps*m+32*floor;
        volatile Real bound=std::abs(q)+allowance;
        volatile Real square=bound*bound,term=weight*square,next=sum+term;
        sum=next;++count;
        valid=valid && weight>0 && weight<=1 && sum>=0 && sum<=std::numeric_limits<Real>::max();
    }
    Real upper() const {
        using I=OlsProofInterval;
        if (!valid) return std::numeric_limits<Real>::infinity();
        constexpr Real eps=std::numeric_limits<Real>::epsilon();
        const Real k=static_cast<Real>(count)+32;
        volatile Real error=k*eps,denominator=1-error;
        if (!(denominator>0)) return std::numeric_limits<Real>::infinity();
        // Each underflowed multiplication contributes at most min-normal;
        // the previous multiplication can then be amplified by its weight.
        // The per-row affine allowance handles cancellation, while this
        // accumulator currently applies only to weights <= 1 (caller gate).
        volatile Real floor=32*k*std::numeric_limits<Real>::min();
        volatile Real numerator=sum+floor,value=numerator/denominator;
        return I::up(I::up(value));
    }
};

}}
#endif
