#ifndef XHDFE_AUDIT_EXACT_BINARY_PRODUCTS_HPP
#define XHDFE_AUDIT_EXACT_BINARY_PRODUCTS_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

// Diagnostic prototype. Integer multiplication/accumulation is independent
// of floating contraction and FTZ. Six finite FP64 factors plus int-sized
// row/column counts fit the full window; narrower windows must fail closed.
template<std::size_t Words=200,int BaseExponent=-6444>
class ExactBinaryProducts {
    using Limbs=std::array<std::uint64_t,Words>;
    Limbs positive_{},negative_{};
    bool valid_=true;
    std::size_t first_=Words,last_=0;
    static unsigned trailing(std::uint64_t value) {
        unsigned count=0;
        while ((value&1)==0) {value>>=1;++count;}
        return count;
    }
    static std::array<std::uint64_t,2> multiply(std::uint64_t a,std::uint64_t b) {
        const std::uint64_t a0=static_cast<std::uint32_t>(a),a1=a>>32;
        const std::uint64_t b0=static_cast<std::uint32_t>(b),b1=b>>32;
        const auto low=a0*b0,middle1=a0*b1,middle2=a1*b0,high=a1*b1;
        const auto middle=(low>>32)+static_cast<std::uint32_t>(middle1)+static_cast<std::uint32_t>(middle2);
        return {(middle<<32)|static_cast<std::uint32_t>(low),high+(middle1>>32)+(middle2>>32)+(middle>>32)};
    }
    void add_word(Limbs& target,std::size_t index,std::uint64_t value) {
        while (value) {
            if (index>=Words) {valid_=false;return;}
            if (index<first_) first_=index;
            if (index>last_) last_=index;
            const auto old=target[index];target[index]+=value;
            value=target[index]<old;++index;
        }
    }
    void add_shifted(Limbs& target,std::uint64_t word,int exponent) {
        if (!word) return;
        const unsigned zeros=trailing(word);word>>=zeros;exponent+=zeros;
        const int shift=exponent-BaseExponent;
        if (shift<0) {valid_=false;return;}
        const auto index=static_cast<std::size_t>(shift/64);const unsigned offset=shift%64;
        add_word(target,index,word<<offset);
        if (offset) add_word(target,index+1,word>>(64-offset));
    }
public:
    template<std::size_t Count>
    void add_product(const std::array<double,Count>& values,bool subtract=false) {
        static_assert(Count>=1 && Count<=6,"supported exact product order");
        std::array<std::uint64_t,6> product{};product[0]=1;
        std::size_t used=1;int exponent=0;bool negative=subtract,zero=false;
        for (double value:values) {
            std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
            const unsigned encoded=static_cast<unsigned>((bits>>52)&2047);
            if (encoded==2047) {valid_=false;return;}
            auto mantissa=(bits&0x000fffffffffffffULL)|(encoded ? 0x0010000000000000ULL : 0);
            if (!mantissa) {zero=true;continue;}
            const unsigned zeros=trailing(mantissa);mantissa>>=zeros;
            exponent+=(encoded ? static_cast<int>(encoded)-1075 : -1074)+static_cast<int>(zeros);
            negative=negative!=(bits>>63);
            if (mantissa==1) continue;
            std::uint64_t carry=0;
            for (std::size_t i=0;i<used;++i) {
                auto parts=multiply(product[i],mantissa);
                const auto before=parts[0];parts[0]+=carry;parts[1]+=parts[0]<before;
                product[i]=parts[0];carry=parts[1];
            }
            if (carry) {
                if (used==product.size()) {valid_=false;return;}
                product[used++]=carry;
            }
        }
        if (zero) return;
        auto& target=negative ? negative_ : positive_;
        for (std::size_t i=0;i<used;++i) add_shifted(target,product[i],exponent+static_cast<int>(64*i));
    }
    void merge(const ExactBinaryProducts& other,bool subtract=false) {
        if (this==&other) {const auto copy=other;merge(copy,subtract);return;}
        valid_=valid_ && other.valid_;
        for (std::size_t i=other.first_;i<Words && i<=other.last_;++i) {
            add_word(positive_,i,subtract ? other.negative_[i] : other.positive_[i]);
            add_word(negative_,i,subtract ? other.positive_[i] : other.negative_[i]);
        }
    }
    void add_scaled(const ExactBinaryProducts& other,double scale,bool subtract=false) {
        if (this==&other) {const auto copy=other;add_scaled(copy,scale,subtract);return;}
        std::uint64_t bits;std::memcpy(&bits,&scale,sizeof(bits));
        const unsigned encoded=static_cast<unsigned>((bits>>52)&2047);
        if (encoded==2047 || !other.valid_) {valid_=false;return;}
        auto mantissa=(bits&0x000fffffffffffffULL)|(encoded ? 0x0010000000000000ULL : 0);
        if (!mantissa) return;
        const unsigned zeros=trailing(mantissa);mantissa>>=zeros;
        const int exponent=(encoded ? static_cast<int>(encoded)-1075 : -1074)+static_cast<int>(zeros);
        const bool negative=subtract!=(bits>>63);
        for (std::size_t i=other.first_;i<Words && i<=other.last_;++i) {
            for (int side=0;side<2;++side) {
                const auto word=side ? other.negative_[i] : other.positive_[i];
                if (!word) continue;
                const auto parts=multiply(word,mantissa);
                auto& target=(negative!=(side!=0)) ? negative_ : positive_;
                const int shift=BaseExponent+static_cast<int>(64*i)+exponent;
                add_shifted(target,parts[0],shift);
                add_shifted(target,parts[1],shift+64);
            }
        }
    }
    bool valid() const {return valid_;}
    bool exactly_zero() const {return valid_ && positive_==negative_;}
    // Return a signed integer magnitude for exact independent reconstruction.
    std::pair<bool,Limbs> magnitude() const {
        bool negative=false;
        for (std::size_t j=Words;j>0;--j) if (positive_[j-1]!=negative_[j-1]) {
            negative=positive_[j-1]<negative_[j-1];break;
        }
        const auto& a=negative ? negative_ : positive_;
        const auto& b=negative ? positive_ : negative_;
        Limbs result{};std::uint64_t borrow=0;
        for (std::size_t i=0;i<Words;++i) {
            const auto first=a[i]-b[i];result[i]=first-borrow;
            borrow=(a[i]<b[i] || first<borrow);
        }
        return {negative,result};
    }
    long double upper_absolute() const {
        const auto infinity=std::numeric_limits<long double>::infinity();
        if (!valid_) return infinity;
        const auto absolute=magnitude().second;
        long double total=0;
        for (std::size_t j=Words;j>0;--j) if (absolute[j-1]) {
            const int exponent=BaseExponent+static_cast<int>(64*(j-1));
            unsigned leading=0;auto word=absolute[j-1];
            while (word>>1) {word>>=1;++leading;}
            long double upper;
            if (exponent+static_cast<int>(leading)<std::numeric_limits<long double>::min_exponent-1) {
                upper=std::numeric_limits<long double>::min();
            } else {
                long double coefficient=static_cast<long double>(absolute[j-1]);
                if (std::numeric_limits<long double>::digits<64) coefficient=std::nextafter(coefficient,infinity);
                volatile long double value=std::ldexp(coefficient,exponent);
                upper=std::nextafter(static_cast<long double>(value),infinity);
            }
            volatile long double next=total+upper;
            total=std::nextafter(static_cast<long double>(next),infinity);
        }
        return total;
    }
    int top_exponent() const {
        const auto absolute=magnitude().second;
        for (std::size_t j=Words;j>0;--j) if (absolute[j-1]) {
            unsigned leading=0;auto word=absolute[j-1];
            while (word>>1) {word>>=1;++leading;}
            return BaseExponent+static_cast<int>(64*(j-1)+leading);
        }
        return std::numeric_limits<int>::min();
    }
    std::pair<long double,long double> interval(int binary_scale=0) const {
        const auto infinity=std::numeric_limits<long double>::infinity();
        if (!valid_) return {-infinity,infinity};
        const auto signed_value=magnitude();
        long double lower=0,upper=0;
        for (std::size_t j=Words;j>0;--j) if (signed_value.second[j-1]) {
            const auto word=signed_value.second[j-1];
            const int exponent=BaseExponent+static_cast<int>(64*(j-1))+binary_scale;
            unsigned leading=0;auto copy=word;
            while (copy>>1) {copy>>=1;++leading;}
            long double lo=0,hi;
            if (exponent+static_cast<int>(leading)<std::numeric_limits<long double>::min_exponent-1) {
                hi=std::numeric_limits<long double>::min();
            } else {
                long double coefficient=static_cast<long double>(word);
                const auto coefficient_lo=std::numeric_limits<long double>::digits<64 ? std::nextafter(coefficient,-infinity) : coefficient;
                const auto coefficient_hi=std::numeric_limits<long double>::digits<64 ? std::nextafter(coefficient,infinity) : coefficient;
                volatile long double a=std::ldexp(coefficient_lo,exponent),b=std::ldexp(coefficient_hi,exponent);
                lo=std::nextafter(static_cast<long double>(a),-infinity);
                hi=std::nextafter(static_cast<long double>(b),infinity);
            }
            volatile long double a=lower+lo,b=upper+hi;
            lower=std::max(0.0L,std::nextafter(static_cast<long double>(a),-infinity));
            upper=std::nextafter(static_cast<long double>(b),infinity);
        }
        return signed_value.first ? std::make_pair(-upper,-lower) : std::make_pair(lower,upper);
    }
    static constexpr int base_exponent=BaseExponent;
};
#endif
