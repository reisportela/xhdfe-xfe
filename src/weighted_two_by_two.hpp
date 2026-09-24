#ifndef HDFE_WEIGHTED_TWO_BY_TWO_HPP
#define HDFE_WEIGHTED_TWO_BY_TWO_HPP

#include "fe_absorption.hpp"
#include "exact_binary_products.hpp"
#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

#if defined(__CUDACC__) || defined(__NVCOMPILER_CUDA__)
#define HDFE_N01_HD __host__ __device__
#else
#define HDFE_N01_HD
#endif

namespace hdfe { namespace detail { namespace weighted_two_by_two {

HDFE_N01_HD inline double add(double a,double b) { volatile double v=a+b;return v; }
HDFE_N01_HD inline double sub(double a,double b) { volatile double v=a-b;return v; }
HDFE_N01_HD inline double mul(double a,double b) { volatile double v=a*b;return v; }
HDFE_N01_HD inline double divide(double a,double b) { volatile double v=a/b;return v; }
HDFE_N01_HD inline std::uint64_t bits(double x) {
#ifdef __CUDA_ARCH__
    return static_cast<std::uint64_t>(__double_as_longlong(x));
#else
    std::uint64_t v;std::memcpy(&v,&x,sizeof(v));return v;
#endif
}
HDFE_N01_HD inline double from_bits(std::uint64_t x) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(static_cast<long long>(x));
#else
    double v;std::memcpy(&v,&x,sizeof(v));return v;
#endif
}
HDFE_N01_HD inline bool nonzero(double x) { return (bits(x)&0x7fffffffffffffffULL)!=0; }
HDFE_N01_HD inline bool finite(double x) { return (bits(x)&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL; }
HDFE_N01_HD inline double absolute(double x) { return from_bits(bits(x)&0x7fffffffffffffffULL); }

struct Pair {
    double hi=0,lo=0;
    HDFE_N01_HD Pair()=default;
    HDFE_N01_HD explicit Pair(double x):hi(x) {}
    HDFE_N01_HD static Pair sum(double a,double b) {
        Pair r;r.hi=add(a,b);const double v=sub(r.hi,a);
        r.lo=add(sub(a,sub(r.hi,v)),sub(b,v));return r;
    }
    HDFE_N01_HD Pair operator-() const { Pair r;r.hi=-hi;r.lo=-lo;return r; }
    HDFE_N01_HD Pair operator+(const Pair& b) const {
        auto s=sum(hi,b.hi);const auto t=sum(lo,b.lo);
        s=sum(s.hi,add(s.lo,t.hi));return sum(s.hi,add(s.lo,t.lo));
    }
    HDFE_N01_HD Pair operator-(const Pair& b) const { return *this+(-b); }
    HDFE_N01_HD Pair operator*(const Pair& b) const {
        const double p=mul(hi,b.hi);
        volatile double err=::fma(hi,b.hi,-p);
        const double e=add(err,add(mul(hi,b.lo),mul(lo,b.hi)));
        return sum(p,add(e,mul(lo,b.lo)));
    }
    HDFE_N01_HD Pair operator/(const Pair& b) const {
        Pair q(divide(hi,b.hi));q=q+Pair(divide((*this-b*q).hi,b.hi));
        return q+Pair(divide((*this-b*q).hi,b.hi));
    }
};

// A binary exponent is kept separate from every normalized FP64 parcel.
// This is arithmetic for the four-cell projection, not an acceptance bound.
struct Term {
    double value=0;
    int exponent=0;
    HDFE_N01_HD static Term make(double x,int extra=0) {
        Term out;const auto raw=bits(x);auto significand=raw&0x000fffffffffffffULL;
        const int field=static_cast<int>((raw>>52)&2047);
        if(field) significand|=0x0010000000000000ULL;
        if(!significand) return out;
        int shift=0;
        while((significand&0x0010000000000000ULL)==0) {significand<<=1;++shift;}
        out.value=mul(static_cast<double>(significand),0x1p-53);
        if(raw>>63) out.value=-out.value;
        out.exponent=(field ? field-1022 : -1021-shift)+extra;
        return out;
    }
};

struct Scaled {
    Pair fraction;
    int exponent=0;
    HDFE_N01_HD static Scaled normalized(Pair p,int e) {
        Scaled out;
        if(!nonzero(p.hi)) {p.hi=p.lo;p.lo=0;}
        if(!nonzero(p.hi)) return out;
        const Term t=Term::make(p.hi);
        out.fraction.hi=t.value;out.fraction.lo=::ldexp(p.lo,-t.exponent);
        out.exponent=e+t.exponent;return out;
    }
    HDFE_N01_HD static Scaled number(double x) {
        const Term t=Term::make(x);Scaled out;out.fraction=Pair(t.value);out.exponent=t.exponent;return out;
    }
    HDFE_N01_HD Scaled operator-() const {Scaled out=*this;out.fraction=-out.fraction;return out;}
    HDFE_N01_HD Scaled operator*(const Scaled& b) const {
        return normalized(fraction*b.fraction,exponent+b.exponent);
    }
    HDFE_N01_HD Scaled operator/(const Scaled& b) const {
        return normalized(fraction/b.fraction,exponent-b.exponent);
    }
};

HDFE_N01_HD inline void two_sum(Term a,Term b,Term& high,Term& low) {
    if(!nonzero(a.value)) {high=b;low=Term{};return;}
    if(!nonzero(b.value)) {high=a;low=Term{};return;}
    if(a.exponent<b.exponent) {const Term t=a;a=b;b=t;}
    const int gap=a.exponent-b.exponent;
    if(gap>60) {high=a;low=b;return;}
    const Pair s=Pair::sum(a.value,::ldexp(b.value,-gap));
    high=Term::make(s.hi,a.exponent);low=Term::make(s.lo,a.exponent);
}

template<int Capacity> struct Expansion {
    Term part[Capacity];int size=0;
    HDFE_N01_HD void push(Term value) {
        if(!nonzero(value.value)) return;
        int used=0;
        for(int i=0;i<size;++i) {
            Term high,low;two_sum(value,part[i],high,low);
            if(nonzero(low.value)) part[used++]=low;
            value=high;
        }
        if(nonzero(value.value)) part[used++]=value;
        size=used;
    }
    HDFE_N01_HD void push(Scaled value) {
        push(Term::make(value.fraction.lo,value.exponent));
        push(Term::make(value.fraction.hi,value.exponent));
    }
    HDFE_N01_HD Scaled finish() const {
        if(!size) return Scaled{};
        const int exponent=part[size-1].exponent;
        Pair result;
        for(int i=0;i<size;++i) {
            const int shift=part[i].exponent-exponent;
            // Nonoverlapping smaller parcels cannot cancel the leading one.
            if(shift>=-160) result=result+Pair(::ldexp(part[i].value,shift));
        }
        return Scaled::normalized(result,exponent);
    }
};

inline Scaled plus(Scaled a,Scaled b) {Expansion<4> sum;sum.push(a);sum.push(b);return sum.finish();}

// Construct the final FP64 bits explicitly, including subnormal results.
// A tiny positive weight is never rounded to zero before its products.
HDFE_N01_HD inline double output(Scaled value) {
    const Term t=Term::make(add(value.fraction.hi,value.fraction.lo),value.exponent);
    if(!nonzero(t.value)) return 0;
    const std::uint64_t raw=bits(t.value),sign=raw&0x8000000000000000ULL;
    const auto significand=(raw&0x000fffffffffffffULL)|0x0010000000000000ULL;
    if(t.exponent>1024) return from_bits(sign|0x7ff0000000000000ULL);
    if(t.exponent>=-1021)
        return from_bits(sign|(static_cast<std::uint64_t>(t.exponent+1022)<<52)|(significand&0x000fffffffffffffULL));
    const int shift=-1021-t.exponent;
    if(shift>53) return from_bits(sign);
    auto rounded=significand>>shift;
    const auto mask=(1ULL<<shift)-1,tail=significand&mask,half=1ULL<<(shift-1);
    if(tail>half || (tail==half && (rounded&1))) ++rounded;
    return from_bits(sign|rounded);
}

inline Scaled from_long_double(long double value) {
    if(value==0) return Scaled{};
    int exponent=0;const long double f=std::frexp(value,&exponent);
    Pair p;p.hi=static_cast<double>(f);
    volatile long double error=f-static_cast<long double>(p.hi);
    p.lo=static_cast<double>(error);return Scaled::normalized(p,exponent);
}

inline Scaled from_exact(const ExactBinaryProducts<>& exact) {
    if(!exact.valid()) throw std::runtime_error("2x2 weighted projection: exact moment range exhausted; no estimates returned");
    if(exact.exactly_zero()) return Scaled{};
    const auto magnitude=exact.magnitude();const int exponent=exact.top_exponent()+1;
    Pair result;
    for(std::size_t k=magnitude.second.size();k>0;--k) {
        const auto word=magnitude.second[k-1];if(!word) continue;
        const int shift=ExactBinaryProducts<>::base_exponent+static_cast<int>(64*(k-1))-exponent;
        if(shift < -192) break;
        result=result+Pair(std::ldexp(static_cast<double>(word>>32),shift+32));
        result=result+Pair(std::ldexp(static_cast<double>(word&0xffffffffULL),shift));
    }
    if(magnitude.first) result=-result;
    return Scaled::normalized(result,exponent);
}

inline long double wide_input(double x) {
    const auto raw=bits(x);const int field=static_cast<int>((raw>>52)&2047);
    if(field || !(raw&0x7fffffffffffffffULL)) return static_cast<long double>(x);
    long double value=std::ldexp(static_cast<long double>(raw&0x000fffffffffffffULL),-1074);
    return raw>>63 ? -value : value;
}

struct NativeSum {
    long double sum=0,correction=0,absolute_sum=0;
    void add_value(long double value) {
        volatile long double next=sum+value;
        const bool larger=std::abs(sum)>=std::abs(value);
        volatile long double first=larger ? sum-next : value-next;
        volatile long double error=larger ? first+value : first+sum;
        volatile long double corrected=correction+error;correction=corrected;sum=next;
        volatile long double absolute_next=absolute_sum+std::abs(value);absolute_sum=absolute_next;
    }
    long double value() const {volatile long double v=sum+correction;return v;}
};

struct Cell {
    double anchor=0;
    Scaled offset,residual;
    double fast[5]{};
    bool fast_range=false;
};

HDFE_N01_HD inline bool ordinary_range(Term value) {
    return !nonzero(value.value) || (value.exponent>=-968 && value.exponent<=1020);
}

inline void prepare_application(Cell& cell) {
    const Term parts[5]={Term::make(-cell.anchor),Term::make(-cell.offset.fraction.hi,cell.offset.exponent),
        Term::make(-cell.offset.fraction.lo,cell.offset.exponent),Term::make(cell.residual.fraction.hi,cell.residual.exponent),
        Term::make(cell.residual.fraction.lo,cell.residual.exponent)};
    cell.fast_range=true;
    for(int k=0;k<5;++k) {
        cell.fast_range=cell.fast_range && ordinary_range(parts[k]);
        Scaled value;value.fraction=Pair(parts[k].value);value.exponent=parts[k].exponent;
        cell.fast[k]=output(value);
    }
}

HDFE_N01_HD inline double apply(double raw,const Cell& cell) {
    if(cell.fast_range && ordinary_range(Term::make(raw))) {
        Pair sum(raw);double maximum=absolute(raw);
        for(int k=0;k<5;++k) {sum=sum+Pair(cell.fast[k]);const double a=absolute(cell.fast[k]);if(a>maximum) maximum=a;}
        const double value=add(sum.hi,sum.lo);
        // At most one bit of cancellation: the ordinary compensated sum is
        // sufficient. This selects arithmetic; it never accepts/rejects a fit.
        if(!nonzero(maximum) || absolute(value)>=mul(maximum,0.5)) return value;
    }
    Expansion<8> sum;
    sum.push(Term::make(raw));sum.push(Term::make(-cell.anchor));
    sum.push(-cell.offset);sum.push(cell.residual);
    return output(sum.finish());
}

struct FeView {
    const int* group_ids=nullptr;
    int num_groups=0;
    int num_levels_present=0;
};

struct Dimension {
    int axis=0;
    int groups=2;
    bool coefficient_owner=true;
    int storage_groups=2;
    std::array<int,2> group_ids{{0,1}};
};

struct Plan {
    int n=0,cols=0,threads=1,chunks=0;
    std::vector<unsigned char> cell;
    std::vector<Cell> values; // rhs-major, four cells per RHS
    std::array<Scaled,4> mass;
    std::array<int,4> first{{-1,-1,-1,-1}};
    std::array<int,2> basis{{0,1}};
    std::vector<Dimension> dimensions{{0,2,true},{1,2,true}};
};

struct Shape {
    int labels[2][2]{};
    std::array<int,4> first{{-1,-1,-1,-1}};
};

inline bool identify(const int* first,const int* second,int n,Shape& shape) {
    if(n<=0 || !first || !second) return false;
    shape=Shape{};shape.labels[0][0]=first[0];shape.labels[1][0]=second[0];
    int count[2]={1,1};
    for(int i=0;i<n;++i) {
        int code=0;
        for(int d=0;d<2;++d) {
            const int id=d ? second[i] : first[i];int slot=0;
            if(id!=shape.labels[d][0]) {
                slot=1;
                if(count[d]==1) {shape.labels[d][1]=id;count[d]=2;}
                else if(id!=shape.labels[d][1]) return false;
            }
            code=2*code+slot;
        }
        if(shape.first[code]<0) shape.first[code]=i;
    }
    return count[0]==2 && count[1]==2 && std::find(shape.first.begin(),shape.first.end(),-1)==shape.first.end();
}

struct IndexedShape {
    Shape shape;
    std::array<int,2> basis{{-1,-1}};
    std::vector<Dimension> dimensions;
};

inline bool identify_indexed(const std::vector<FeView>& views,int n,IndexedShape& out) {
    if(n<=0 || views.size()<2) return false;
    // Indexers already know cardinality. Do not scan a large general FE
    // just to discover that this four-cell algebra cannot apply.
    for(const auto& view:views) {
        const int levels=view.num_levels_present>0 ? view.num_levels_present : view.num_groups;
        if(!view.group_ids || (levels!=1 && levels!=2)) return false;
    }
    IndexedShape candidate;
    for(int d=0;d<static_cast<int>(views.size());++d) {
        const auto& view=views[d];
        const int levels=view.num_levels_present>0 ? view.num_levels_present : view.num_groups;
        if(levels!=2) continue;
        if(candidate.basis[0]<0) {candidate.basis[0]=d;continue;}
        if(identify(views[candidate.basis[0]].group_ids,view.group_ids,n,candidate.shape)) {
            candidate.basis[1]=d;break;
        }
    }
    if(candidate.basis[1]<0) return false;
    std::vector<std::array<int,4>> labels(views.size());
    candidate.dimensions.resize(views.size());
    for(int d=0;d<static_cast<int>(views.size());++d) {
        for(int c=0;c<4;++c) labels[d][c]=views[d].group_ids[candidate.shape.first[c]];
        const auto& id=labels[d];auto& dimension=candidate.dimensions[d];
        if(id[0]==id[1] && id[0]==id[2] && id[0]==id[3]) {
            dimension.axis=-1;dimension.groups=1;
        } else if(id[0]==id[1] && id[2]==id[3] && id[0]!=id[2]) {
            dimension.axis=0;dimension.groups=2;
        } else if(id[0]==id[2] && id[1]==id[3] && id[0]!=id[1]) {
            dimension.axis=1;dimension.groups=2;
        } else return false; // An interaction changes the absorbed space.
        const int levels=views[d].num_levels_present>0 ? views[d].num_levels_present : views[d].num_groups;
        if(dimension.groups!=levels) return false;
        dimension.coefficient_owner=d==candidate.basis[0] || d==candidate.basis[1];
        dimension.storage_groups=views[d].num_groups;
        dimension.group_ids={{id[0],dimension.axis==0 ? id[2] : id[1]}};
        for(int g=0;g<dimension.groups;++g)
            if(dimension.group_ids[g]<0 || dimension.group_ids[g]>=dimension.storage_groups) return false;
    }
    const int* first=views[candidate.basis[0]].group_ids;
    const int* second=views[candidate.basis[1]].group_ids;
    for(int i=0;i<n;++i) {
        const int c=2*(first[i]!=candidate.shape.labels[0][0])+(second[i]!=candidate.shape.labels[1][0]);
        for(std::size_t d=0;d<views.size();++d)
            if(views[d].group_ids[i]!=labels[d][c]) return false;
    }
    out=std::move(candidate);return true;
}

class Region {
    ParallelWorkObserver* observer_;
public:
    Region(ParallelWorkObserver* observer,int threads,bool work):observer_(work ? observer : nullptr) {
        if(observer_) observer_->begin_region(threads);
    }
    ~Region() noexcept(false) {if(observer_) observer_->end_region();}
    void work() const noexcept {if(observer_) observer_->observe_work();}
};

inline bool prepare(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const int* first,const int* second,
    const Eigen::VectorXd* weights,const HdfeOptions& options,Plan& plan) {
    if(!weights || y.size()==0) return false;
    const int n=static_cast<int>(y.size()),cols=static_cast<int>(X.cols()),rhs=cols+1;
    if(X.rows()!=n || weights->size()!=n) throw std::runtime_error("2x2 weighted projection: inconsistent dimensions");
    Shape shape;if(!identify(first,second,n,shape)) return false;
    // Preserve the public weight validation; this local operator is defined
    // only on the already-kept, strictly positive-weight sample.
    for(int i=0;i<n;++i) {
        const auto w=bits((*weights)[i]);
        if((w>>63) || !(w&0x7fffffffffffffffULL) || (w&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)
            throw std::runtime_error("Weights must be finite and positive; no estimates returned");
    }
    plan.n=n;plan.cols=cols;plan.first=shape.first;
    plan.basis={{0,1}};plan.dimensions={{0,2,true},{1,2,true}};
    plan.threads=std::max(1,options.num_threads);plan.chunks=deterministic_parallel_chunk_count(n);
#ifdef HDFE_USE_OPENMP
    if(omp_in_parallel() && options.num_threads>1)
        throw std::runtime_error("2x2 weighted projection cannot honor multiple threads inside an active OpenMP region");
    plan.threads=std::min(plan.threads,runtime_thread_capacity());
    if(options.num_threads>0) omp_set_dynamic(0);
#else
    if(plan.threads>1) throw std::runtime_error("num_threads > 1 requires an xhdfe build with OpenMP support");
#endif
    plan.cell.resize(n);plan.values.resize(static_cast<std::size_t>(rhs)*4);
    auto raw=[&](int i,int j) {return j ? X(i,j-1) : y[i];};
    for(int j=0;j<rhs;++j) for(int c=0;c<4;++c) {
        const double anchor=raw(shape.first[c],j);
        if(!finite(anchor)) throw std::runtime_error("2x2 weighted projection received non-finite data; no estimates returned");
        plan.values[static_cast<std::size_t>(j)*4+c].anchor=anchor;
    }
    constexpr bool native_range=std::numeric_limits<long double>::max_exponent>=8192 &&
        std::numeric_limits<long double>::min_exponent<=-8192;
    const std::size_t stride=static_cast<std::size_t>(rhs+1)*4;
    std::vector<NativeSum> partial(native_range ? static_cast<std::size_t>(plan.chunks)*stride : 0);
    std::atomic<bool> bad_input{false};
    {
        Region observed(options.parallel_observer,plan.threads,true);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(plan.threads)
#endif
        for(int chunk=0;chunk<plan.chunks;++chunk) {
            observed.work();
            const int begin=deterministic_parallel_chunk_begin(n,chunk,plan.chunks);
            const int end=deterministic_parallel_chunk_end(n,chunk,plan.chunks);
            for(int i=begin;i<end;++i) {
                const int c=2*(first[i]!=shape.labels[0][0])+(second[i]!=shape.labels[1][0]);plan.cell[i]=static_cast<unsigned char>(c);
                const long double w=native_range ? wide_input((*weights)[i]) : 0;
                if(native_range) partial[static_cast<std::size_t>(chunk)*stride+c].add_value(w);
                for(int j=0;j<rhs;++j) {
                    const double value=raw(i,j);
                    if(!finite(value)) {bad_input.store(true,std::memory_order_relaxed);continue;}
                    if(native_range) {
                        volatile long double delta=wide_input(value)-wide_input(plan.values[static_cast<std::size_t>(j)*4+c].anchor);
                        volatile long double product=w*delta;
                        partial[static_cast<std::size_t>(chunk)*stride+4+static_cast<std::size_t>(j)*4+c].add_value(product);
                    }
                }
            }
        }
    }
    if(bad_input.load(std::memory_order_relaxed)) throw std::runtime_error("2x2 weighted projection received non-finite data; no estimates returned");
    std::vector<NativeSum> total(native_range ? stride : 0);
    if(native_range) for(std::size_t k=0;k<stride;++k) {
        long double absolute_total=0;
        for(int chunk=0;chunk<plan.chunks;++chunk) {
            const auto& entry=partial[static_cast<std::size_t>(chunk)*stride+k];
            total[k].add_value(entry.sum);total[k].add_value(entry.correction);
            volatile long double next=absolute_total+entry.absolute_sum;absolute_total=next;
        }
        total[k].absolute_sum=absolute_total;
    }
    std::array<ExactBinaryProducts<>,4> exact_mass;bool exact_mass_ready=false;
    auto ensure_exact_mass=[&]() {
        if(!exact_mass_ready) {
            for(int i=0;i<n;++i) exact_mass[plan.cell[i]].add_product(std::array<double,1>{(*weights)[i]});
            exact_mass_ready=true;
        }
    };
    if(!native_range) ensure_exact_mass();
    for(int c=0;c<4;++c) plan.mass[c]=native_range ? from_long_double(total[c].value()) : from_exact(exact_mass[c]);
    for(int j=0;j<rhs;++j) {
        std::array<bool,4> exact{};bool any_exact=false;
        for(int c=0;c<4;++c) {
            if(native_range) {
                const auto& moment=total[4+static_cast<std::size_t>(j)*4+c];
                // Recover FP64 headroom after cancellation; this only selects
                // the arithmetic used to compute the same weighted mean.
                const long double headroom=std::ldexp(1.0L,std::numeric_limits<long double>::digits-std::numeric_limits<double>::digits);
                const long double mass=total[c].value(),offset=moment.value()/mass;
                const long double anchor=wide_input(plan.values[static_cast<std::size_t>(j)*4+c].anchor);
                volatile long double mean=anchor+offset;
                exact[c]=std::abs(moment.value())*headroom<moment.absolute_sum ||
                    (moment.correction!=0 && moment.value()==moment.sum) ||
                    (total[c].correction!=0 && mass==total[c].sum) ||
                    std::abs(mean)*headroom<std::abs(anchor)+std::abs(offset);
                if(!exact[c]) plan.values[static_cast<std::size_t>(j)*4+c].offset=
                    from_long_double(offset);
            } else exact[c]=true;
            any_exact=any_exact || exact[c];
        }
        if(any_exact) {
            ensure_exact_mass();std::array<ExactBinaryProducts<>,4> moments;
            for(int i=0;i<n;++i) {
                const int c=plan.cell[i];if(!exact[c]) continue;
                const double anchor=plan.values[static_cast<std::size_t>(j)*4+c].anchor;
                moments[c].add_product(std::array<double,2>{(*weights)[i],raw(i,j)});
                moments[c].add_product(std::array<double,2>{(*weights)[i],anchor},true);
            }
            for(int c=0;c<4;++c) if(exact[c]) {
                auto& cell=plan.values[static_cast<std::size_t>(j)*4+c];
                // Re-anchor near the mean before converting the centered
                // numerator. This preserves a tiny mean after large terms
                // cancel, and a tiny offset next to a nonzero mean.
                moments[c].add_scaled(exact_mass[c],cell.anchor);
                const Scaled mean=from_exact(moments[c])/from_exact(exact_mass[c]);
                cell.anchor=output(mean);
                if(!finite(cell.anchor)) throw std::runtime_error("2x2 weighted mean exceeds its finite input range; no estimates returned");
                moments[c].add_scaled(exact_mass[c],cell.anchor,true);
                cell.offset=from_exact(moments[c])/from_exact(exact_mass[c]);
            }
        }
    }
    Scaled minimum=plan.mass[0];
    for(int c=1;c<4;++c) if(plan.mass[c].exponent<minimum.exponent ||
        (plan.mass[c].exponent==minimum.exponent && plan.mass[c].fraction.hi<minimum.fraction.hi)) minimum=plan.mass[c];
    std::array<Scaled,4> ratio;Expansion<8> denominator_sum;
    for(int c=0;c<4;++c) {ratio[c]=minimum/plan.mass[c];denominator_sum.push(ratio[c]);}
    const Scaled denominator=denominator_sum.finish();
    for(int j=0;j<rhs;++j) {
        Expansion<16> contrast;
        for(int c=0;c<4;++c) {
            const bool negative=c==1 || c==2;const auto& cell=plan.values[static_cast<std::size_t>(j)*4+c];
            contrast.push(Term::make(negative ? -cell.anchor : cell.anchor));contrast.push(negative ? -cell.offset : cell.offset);
        }
        const Scaled delta=contrast.finish();
        for(int c=0;c<4;++c) {
            auto& cell=plan.values[static_cast<std::size_t>(j)*4+c];
            cell.residual=(ratio[c]*delta)/denominator;
            if(c==1 || c==2) cell.residual=-cell.residual;
            prepare_application(cell);
        }
    }
    return true;
}

inline bool prepare_indexed(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const std::vector<FeView>& views,
    const Eigen::VectorXd* weights,const HdfeOptions& options,Plan& plan) {
    if(!weights || options.max_iter<=0) return false;
    IndexedShape shape;if(!identify_indexed(views,static_cast<int>(y.size()),shape)) return false;
    if(!prepare(y,X,views[shape.basis[0]].group_ids,views[shape.basis[1]].group_ids,weights,options,plan)) return false;
    plan.basis=shape.basis;plan.dimensions=std::move(shape.dimensions);return true;
}

inline std::array<double,4> alphas(const Plan& plan,int rhs) {
    std::array<Scaled,4> fitted;
    for(int c=0;c<4;++c) {
        const auto& cell=plan.values[static_cast<std::size_t>(rhs)*4+c];Expansion<8> sum;
        sum.push(Term::make(cell.anchor));sum.push(cell.offset);sum.push(-cell.residual);fitted[c]=sum.finish();
    }
    Scaled a=plus(fitted[2],-fitted[0]),b=fitted[0],c=fitted[1];
    std::array<double,4> out{{0,output(a),output(b),output(c)}};
    if(!finite(out[1]) || !finite(out[2]) || !finite(out[3])) {
        const Scaled shift=-(a*Scaled::number(0.5));
        out={{output(shift),output(plus(a,shift)),output(plus(b,-shift)),output(plus(c,-shift))}};
    }
    for(double value:out) if(!finite(value))
        throw std::runtime_error("2x2 weighted FE coefficients exceed the finite output range; no estimates returned");
    return out;
}

inline void metadata(const Plan& plan,const HdfeOptions& options,AbsorptionResult& result,bool force_alphas=false) {
    const int dims=static_cast<int>(plan.dimensions.size());
    result.fe_levels.resize(dims);result.sweep_order_used.resize(dims);
    for(int d=0;d<dims;++d) {result.fe_levels[d]=plan.dimensions[d].groups;result.sweep_order_used[d]=d;}
    result.iterations=1;result.converged=true;
    if(!options.retain_fixed_effects && !force_alphas) return;
    result.fe_group_ids.resize(dims);result.fe_weight_sums.resize(dims);
    for(int d=0;d<dims;++d) {
        const auto& dimension=plan.dimensions[d];const int axis=dimension.axis;
        result.fe_group_ids[d].resize(plan.n);result.fe_weight_sums[d]=Eigen::VectorXd::Zero(dimension.storage_groups);
        for(int i=0;i<plan.n;++i) result.fe_group_ids[d][i]=dimension.group_ids[axis<0 ? 0 : (axis ? (plan.cell[i]&1) : (plan.cell[i]>>1))];
        for(int g=0;g<dimension.groups;++g) {
            const Scaled mass=axis<0 ? plus(plus(plan.mass[0],plan.mass[1]),plus(plan.mass[2],plan.mass[3])) :
                (axis ? plus(plan.mass[g],plan.mass[2+g]) : plus(plan.mass[2*g],plan.mass[2*g+1]));
            const double value=output(mass);
            if(!finite(value)) throw std::runtime_error("2x2 fixed-effect weight totals exceed the finite output range; no estimates returned");
            result.fe_weight_sums[d][dimension.group_ids[g]]=value;
        }
    }
    const bool store=force_alphas || (options.fe_recovery_method==FeRecoveryMethod::Hybrid && plan.cols<=options.savefe_fastpath_max_cols);
    if(store) {
        result.fe_alpha_y.resize(dims);result.fe_alpha_X.resize(dims);
        for(int d=0;d<dims;++d) {
            result.fe_alpha_y[d]=Eigen::VectorXd::Zero(plan.dimensions[d].storage_groups);
            result.fe_alpha_X[d]=Eigen::MatrixXd::Zero(plan.dimensions[d].storage_groups,plan.cols);
        }
        for(int j=0;j<=plan.cols;++j) {
            const auto values=alphas(plan,j);
            for(int d=0;d<dims;++d) if(plan.dimensions[d].coefficient_owner) for(int g=0;g<2;++g) {
                const double value=values[2*plan.dimensions[d].axis+g];
                const int id=plan.dimensions[d].group_ids[g];
                if(j) result.fe_alpha_X[d](id,j-1)=value;
                else result.fe_alpha_y[d][id]=value;
            }
        }
    } else {
        const auto values=alphas(plan,0);result.fe_means.resize(dims);
        for(int d=0;d<dims;++d) {
            result.fe_means[d]=Eigen::VectorXd::Zero(plan.dimensions[d].storage_groups);
            if(plan.dimensions[d].coefficient_owner)
                for(int g=0;g<2;++g) result.fe_means[d][plan.dimensions[d].group_ids[g]]=values[2*plan.dimensions[d].axis+g];
        }
    }
}

inline AbsorptionResult apply_cpu(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Plan& plan,const HdfeOptions& options) {
    AbsorptionResult result;result.y_tilde.resize(plan.n);result.X_tilde.resize(plan.n,plan.cols);
    std::atomic<bool> bad_output{false};
    {
        Region observed(options.parallel_observer,plan.threads,true);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(plan.threads)
#endif
        for(int chunk=0;chunk<plan.chunks;++chunk) {
            observed.work();
            for(int i=deterministic_parallel_chunk_begin(plan.n,chunk,plan.chunks);i<deterministic_parallel_chunk_end(plan.n,chunk,plan.chunks);++i) {
                const int c=plan.cell[i];const double within=apply(y[i],plan.values[c]);result.y_tilde[i]=within;
                if(!finite(within)) bad_output.store(true,std::memory_order_relaxed);
                for(int j=0;j<plan.cols;++j) {
                    const double value=apply(X(i,j),plan.values[static_cast<std::size_t>(j+1)*4+c]);result.X_tilde(i,j)=value;
                    if(!finite(value)) bad_output.store(true,std::memory_order_relaxed);
                }
            }
        }
    }
    if(bad_output.load(std::memory_order_relaxed)) throw std::runtime_error("2x2 weighted projection exceeds the finite output range; no estimates returned");
    metadata(plan,options,result);return result;
}

inline bool try_cpu(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,const HdfeOptions& options,AbsorptionResult& result) {
    if(!weights || fes.size()!=2 || options.max_iter<=0) return false;
    for(const auto& fe:fes) if(fe.size()!=y.size())
        throw std::runtime_error("Each fixed-effect vector must match the length of y");
    Plan plan;if(!prepare(y,X,fes[0].data(),fes[1].data(),weights,options,plan)) return false;
    result=apply_cpu(y,X,plan,options);return true;
}

inline bool try_cpu_indexed(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const std::vector<FeView>& views,
    const Eigen::VectorXd* weights,const HdfeOptions& options,AbsorptionResult& result) {
    Plan plan;if(!prepare_indexed(y,X,views,weights,options,plan)) return false;
    result=apply_cpu(y,X,plan,options);return true;
}

inline bool recover_cpu_from_plan(const Eigen::VectorXd& partial,const int* first,const int* second,
    const Eigen::VectorXd* weights,const HdfeOptions& options,const Plan& plan,FeRecoveryResult& result) {
    const int n=static_cast<int>(partial.size());Eigen::MatrixXd empty(n,0);
    const auto alpha=alphas(plan,0);FeRecoveryResult recovered;
    recovered.contributions.resize(plan.dimensions.size());
    for(std::size_t d=0;d<plan.dimensions.size();++d) {
        recovered.contributions[d].resize(n);
        if(!plan.dimensions[d].coefficient_owner) recovered.contributions[d].setZero();
    }
    Eigen::VectorXd residual(n);std::atomic<bool> bad_output{false};
    {
        Region observed(options.parallel_observer,plan.threads,true);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(plan.threads)
#endif
        for(int chunk=0;chunk<plan.chunks;++chunk) {
            observed.work();
            for(int i=deterministic_parallel_chunk_begin(n,chunk,plan.chunks);i<deterministic_parallel_chunk_end(n,chunk,plan.chunks);++i) {
                const int c=plan.cell[i];const double a=alpha[c>>1],b=alpha[2+(c&1)];
                recovered.contributions[plan.basis[0]][i]=a;recovered.contributions[plan.basis[1]][i]=b;
                residual[i]=sub(partial[i],add(a,b));
                if(!finite(residual[i])) bad_output.store(true,std::memory_order_relaxed);
            }
        }
    }
    if(bad_output.load(std::memory_order_relaxed)) throw std::runtime_error("2x2 recovered FE residual exceeds the finite output range; no estimates returned");
    Plan checked;if(!prepare(residual,empty,first,second,weights,options,checked))
        throw std::runtime_error("2x2 FE recovery geometry changed; no estimates returned");
    recovered.max_delta=0;
    for(int d=0;d<2;++d) for(int g=0;g<2;++g) {
        Expansion<8> numerator,mass;
        for(int c=0;c<4;++c) if((d ? (c&1) : (c>>1))==g) {
            const auto& cell=checked.values[c];
            numerator.push(checked.mass[c]*plus(Scaled::number(cell.anchor),cell.offset));mass.push(checked.mass[c]);
        }
        const double mean=absolute(output(numerator.finish()/mass.finish()));
        if(!finite(mean)) throw std::runtime_error("2x2 FE recovery mean exceeds the finite output range; no estimates returned");
        recovered.max_delta=std::max(recovered.max_delta,mean);
    }
    const double tolerance=options.fe_tolerance>0 ? options.fe_tolerance : options.tol;
    recovered.iterations=1;recovered.converged=tolerance<=0 || recovered.max_delta<=tolerance;
    result=std::move(recovered);return true;
}

inline bool try_recover_cpu(const Eigen::VectorXd& partial,const int* first,const int* second,
    const Eigen::VectorXd* weights,const HdfeOptions& options,FeRecoveryResult& result) {
    if(!weights || options.max_iter<=0) return false;
    Eigen::MatrixXd empty(partial.size(),0);Plan plan;
    if(!prepare(partial,empty,first,second,weights,options,plan)) return false;
    return recover_cpu_from_plan(partial,first,second,weights,options,plan,result);
}

inline bool try_recover_cpu(const Eigen::VectorXd& partial,const std::vector<FeView>& views,
    const Eigen::VectorXd* weights,const HdfeOptions& options,FeRecoveryResult& result) {
    Eigen::MatrixXd empty(partial.size(),0);Plan plan;
    if(!prepare_indexed(partial,empty,views,weights,options,plan)) return false;
    return recover_cpu_from_plan(partial,views[plan.basis[0]].group_ids,views[plan.basis[1]].group_ids,
        weights,options,plan,result);
}

}}} // namespace hdfe::detail::weighted_two_by_two
#undef HDFE_N01_HD
#endif
