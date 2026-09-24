#pragma once

#include "weighted_two_by_two.hpp"

namespace hdfe { namespace detail { namespace additive_cell_projection {

using FeView=weighted_two_by_two::FeView;
constexpr int max_levels=8,max_cells=64,max_columns=32;

// The selector consumes from_auto to suppress a second survey. Preserve the
// original public request separately, without re-enabling any selector branch.
inline thread_local bool public_auto_request=false;
class ScopedPublicAutoRequest final {
    bool previous_;
public:
    explicit ScopedPublicAutoRequest(bool automatic) noexcept
        :previous_(public_auto_request) {public_auto_request=automatic;}
    ~ScopedPublicAutoRequest() {public_auto_request=previous_;}
    ScopedPublicAutoRequest(const ScopedPublicAutoRequest&)=delete;
    ScopedPublicAutoRequest& operator=(const ScopedPublicAutoRequest&)=delete;
};

// The lattice window describes represented binary values, not a tolerance.
struct Window {
    int low=1024,high=-1024;
    bool empty() const noexcept {return low>high;}
    int width() const noexcept {return empty() ? 0 : high-low+1;}
    bool append(double value) noexcept {
        const auto raw=weighted_two_by_two::bits(value);
        const int field=static_cast<int>((raw>>52)&2047);
        auto significand=raw&0x000fffffffffffffULL;
        if(field==2047) return false;
        if(field) significand|=0x0010000000000000ULL;
        if(!significand) return true;
        int trailing=0,top=0;
        auto copy=significand;
        while((copy&1)==0) {copy>>=1;++trailing;}
        copy=significand;while(copy>>=1) ++top;
        const int base=field ? field-1075 : -1074;
        low=std::min(low,base+trailing);high=std::max(high,base+top);
        return low>=-900 && high<=900;
    }
};

inline int ceil_log2(unsigned int value) noexcept {
    int bits=0;for(std::uint64_t limit=1;limit<value;limit<<=1) ++bits;return bits;
}

inline bool scope(const HdfeOptions& options,const FeView& first,const FeView& second,
    int dimensions,int columns,const Eigen::VectorXd* weights) noexcept {
    // The projection is exact when it applies, so every tolerance mode may
    // use it (a demonstrated wrong return is corrected in fast mode as well).
    if(!weights || dimensions!=2 || columns<0 || columns>max_columns || options.max_iter<=0 ||
        (options.absorption_method!=AbsorptionMethod::Auto && !public_auto_request) ||
        options.use_krylov || options.use_sparse_solver ||
        options.absorption_method==AbsorptionMethod::Schwarz ||
        options.absorption_method==AbsorptionMethod::Lsmr || options.absorption_method==AbsorptionMethod::Mlsmr)
        return false;
    const int left=first.num_levels_present>0 ? first.num_levels_present : first.num_groups;
    const int right=second.num_levels_present>0 ? second.num_levels_present : second.num_groups;
    return left>=2 && left<=max_levels && right>=2 && right<=max_levels &&
        first.num_groups>=left && first.num_groups<=max_levels &&
        second.num_groups>=right && second.num_groups<=max_levels &&
        (left>2 || right>2);
}

struct ExplicitTeam {
#ifdef HDFE_USE_OPENMP
    int previous;
    bool active;
    explicit ExplicitTeam(bool requested):previous(omp_get_dynamic()),active(requested) {
        if(active) omp_set_dynamic(0);
    }
    ~ExplicitTeam() {if(active) omp_set_dynamic(previous);}
#else
    explicit ExplicitTeam(bool) {}
#endif
};

struct Plan {
    int n=0,cols=0,threads=1,cells=0;
    std::array<int,2> storage{{0,0}},levels{{0,0}};
    std::vector<unsigned char> cell;
    std::vector<double> means; // RHS-major, storage[0]*storage[1] cells per RHS.
    std::vector<double> alphas; // RHS-major, storage[0]+storage[1] original FE slots.
    std::array<double,max_cells> cell_mass{};
};

inline bool native_significand_available() noexcept {
    // Separate volatile stores/loads materialize both operations even with
    // fast-math. Observe the active precision without changing the controlword.
    volatile long double one=1.0L;
    volatile long double unit=std::ldexp(1.0L,1-std::numeric_limits<long double>::digits);
    volatile long double probe=one+unit;
    volatile long double recovered_unit=probe-one;
    return recovered_unit==unit;
}

inline bool exact_difference(double a,double b,double& out) noexcept {
    Window window;if(!window.append(a) || !window.append(b) ||
        window.width()+1>std::numeric_limits<long double>::digits) return false;
    volatile long double difference=static_cast<long double>(a)-static_cast<long double>(b);
    out=static_cast<double>(difference);
    return weighted_two_by_two::finite(out) && static_cast<long double>(out)==difference;
}

inline bool exact_sum_equals(double a,double b,double expected) noexcept {
    Window window;if(!window.append(a) || !window.append(b) ||
        window.width()+1>std::numeric_limits<long double>::digits) return false;
    volatile long double sum=static_cast<long double>(a)+static_cast<long double>(b);
    return sum==static_cast<long double>(expected);
}

inline bool prepare(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const FeView& first,const FeView& second,
    const Eigen::VectorXd* weights,const HdfeOptions& options,Plan& output) {
    // O(1) indexed cardinality/mode gate, before reading rows or allocating.
    if(!scope(options,first,second,2,static_cast<int>(X.cols()),weights)) return false;
    // Hosts may alter x87 precision. Use the significand budget only when
    // the running arithmetic actually retains its lowest advertised bit.
    if(!native_significand_available()) return false;
    const int n=static_cast<int>(y.size()),cols=static_cast<int>(X.cols()),rhs=cols+1;
    if(n<=0 || X.rows()!=n || weights->size()!=n || !first.group_ids || !second.group_ids) return false;
    const int cells=first.num_groups*second.num_groups;
    const int sum_bits=ceil_log2(static_cast<unsigned int>(n));
    std::array<int,max_cells> counts{};
    std::array<double,max_cells> cell_weight{};
    std::array<bool,max_levels> seen_left{},seen_right{};
    std::array<Window,max_columns+1> windows;
    for(int i=0;i<n;++i) {
        const int a=first.group_ids[i],b=second.group_ids[i];
        if(a<0 || a>=first.num_groups || b<0 || b>=second.num_groups) return false;
        const int c=a*second.num_groups+b;const double weight=(*weights)[i];
        Window weight_window;
        if(!(weight>0) || !weight_window.append(weight)) return false;
        if(counts[c] && weighted_two_by_two::bits(weight)!=weighted_two_by_two::bits(cell_weight[c])) return false;
        cell_weight[c]=weight;++counts[c];seen_left[a]=true;seen_right[b]=true;
        for(int j=0;j<rhs;++j) {
            if(!windows[j].append(j ? X(i,j-1) : y[i]) ||
                windows[j].width()+sum_bits>std::numeric_limits<long double>::digits-1) return false;
        }
    }
    std::vector<int> left,right;
    for(int a=0;a<first.num_groups;++a) if(seen_left[a]) left.push_back(a);
    for(int b=0;b<second.num_groups;++b) if(seen_right[b]) right.push_back(b);
    if(static_cast<int>(left.size())!=(first.num_levels_present>0 ? first.num_levels_present : first.num_groups) ||
        static_cast<int>(right.size())!=(second.num_levels_present>0 ? second.num_levels_present : second.num_groups)) return false;
    for(int a:left) for(int b:right) if(!counts[a*second.num_groups+b]) return false;

    const int threads=std::max(1,options.num_threads);
    const std::size_t stride=static_cast<std::size_t>(rhs)*cells;
    std::vector<long double> partial(static_cast<std::size_t>(threads)*stride,0),sums(stride,0);
    std::atomic<bool> arithmetic_available{true};
    {
        ExplicitTeam team(options.num_threads_explicit);
        weighted_two_by_two::Region observed(options.parallel_observer,threads,true);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
            int tid=0;
#ifdef HDFE_USE_OPENMP
            tid=omp_get_thread_num();
#endif
            const bool exact_arithmetic=native_significand_available();
            if(!exact_arithmetic) arithmetic_available.store(false,std::memory_order_relaxed);
            bool worked=false;auto* destination=partial.data()+static_cast<std::size_t>(tid)*stride;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
            for(int i=0;i<n;++i) {
                if(!exact_arithmetic) continue;
                if(!worked) {observed.work();worked=true;}
                const int c=first.group_ids[i]*second.num_groups+second.group_ids[i];
                for(int j=0;j<rhs;++j) {
                    const auto at=static_cast<std::size_t>(j)*cells+c;
                    volatile long double next=destination[at]+static_cast<long double>(j ? X(i,j-1) : y[i]);
                    destination[at]=next;
                }
            }
        }
    }
    if(!arithmetic_available.load(std::memory_order_relaxed)) return false;
    // The global N bound also covers every partial sum and their reduction.
    for(int t=0;t<threads;++t) for(std::size_t at=0;at<stride;++at) {
        volatile long double next=sums[at]+partial[static_cast<std::size_t>(t)*stride+at];sums[at]=next;
    }
    Plan plan;plan.n=n;plan.cols=cols;plan.threads=threads;plan.cells=cells;
    plan.storage={{first.num_groups,second.num_groups}};
    plan.levels={{static_cast<int>(left.size()),static_cast<int>(right.size())}};
    plan.means.assign(stride,0);
    plan.alphas.assign(static_cast<std::size_t>(rhs)*(first.num_groups+second.num_groups),0);
    for(int c=0;c<cells;++c) if(counts[c]) {
        const double count=static_cast<double>(counts[c]);Window count_window;count_window.append(count);
        const long double mass=static_cast<long double>(cell_weight[c])*counts[c];
        plan.cell_mass[c]=static_cast<double>(mass);
        if(!weighted_two_by_two::finite(plan.cell_mass[c])) return false;
        for(int j=0;j<rhs;++j) {
            const auto at=static_cast<std::size_t>(j)*cells+c;
            volatile long double quotient=sums[at]/static_cast<long double>(counts[c]);
            const double mean=static_cast<double>(quotient);Window mean_window;
            if(!mean_window.append(mean) || mean_window.width()+count_window.width()>
                std::numeric_limits<long double>::digits-1) return false;
            // This product is exactly representable under the bit bound above.
            volatile long double recovered=static_cast<long double>(mean)*static_cast<long double>(count);
            if(recovered!=sums[at]) return false;
            plan.means[at]=mean;
            if(!windows[j].append(mean)) return false;
        }
    }
    const int a0=left.front(),b0=right.front(),slots=first.num_groups+second.num_groups;
    for(int j=0;j<rhs;++j) {
        // Each raw-minus-mean result is a binary lattice number requiring at
        // most53 bits. The actual CPU/device subtraction is therefore exact.
        if(windows[j].width()+1>std::numeric_limits<double>::digits) return false;
        auto* alpha=plan.alphas.data()+static_cast<std::size_t>(j)*slots;
        const auto* mean=plan.means.data()+static_cast<std::size_t>(j)*cells;
        for(int a:left) if(!exact_difference(mean[a*second.num_groups+b0],mean[a0*second.num_groups+b0],alpha[a])) return false;
        for(int b:right) alpha[first.num_groups+b]=mean[a0*second.num_groups+b];
        for(int a:left) for(int b:right)
            if(!exact_sum_equals(alpha[a],alpha[first.num_groups+b],mean[a*second.num_groups+b])) return false;
    }
    plan.cell.resize(n);
    for(int i=0;i<n;++i) plan.cell[i]=static_cast<unsigned char>(first.group_ids[i]*second.num_groups+second.group_ids[i]);
    output=std::move(plan);return true;
}

inline void metadata(const Plan& plan,const HdfeOptions& options,AbsorptionResult& result) {
    result.fe_levels={plan.levels[0],plan.levels[1]};result.sweep_order_used={0,1};
    result.iterations=1;result.converged=true;
    if(!options.retain_fixed_effects) return;
    result.fe_group_ids.resize(2);result.fe_weight_sums.resize(2);
    for(int d=0;d<2;++d) {
        result.fe_group_ids[d].resize(plan.n);result.fe_weight_sums[d]=Eigen::VectorXd::Zero(plan.storage[d]);
        for(int i=0;i<plan.n;++i) result.fe_group_ids[d][i]=d ? plan.cell[i]%plan.storage[1] : plan.cell[i]/plan.storage[1];
        for(int g=0;g<plan.storage[d];++g) {
            long double mass=0;
            for(int c=0;c<plan.cells;++c) if((d ? c%plan.storage[1] : c/plan.storage[1])==g) {
                volatile long double next=mass+plan.cell_mass[c];mass=next;
            }
            result.fe_weight_sums[d][g]=static_cast<double>(mass);
        }
    }
    const bool store=options.fe_recovery_method==FeRecoveryMethod::Hybrid && plan.cols<=options.savefe_fastpath_max_cols;
    const int slots=plan.storage[0]+plan.storage[1];
    if(store) {
        result.fe_alpha_y.resize(2);result.fe_alpha_X.resize(2);
        for(int d=0;d<2;++d) {
            result.fe_alpha_y[d].resize(plan.storage[d]);result.fe_alpha_X[d].resize(plan.storage[d],plan.cols);
            const int offset=d ? plan.storage[0] : 0;
            for(int g=0;g<plan.storage[d];++g) {
                result.fe_alpha_y[d][g]=plan.alphas[offset+g];
                for(int j=0;j<plan.cols;++j)
                    result.fe_alpha_X[d](g,j)=plan.alphas[static_cast<std::size_t>(j+1)*slots+offset+g];
            }
        }
    } else {
        result.fe_means.resize(2);
        for(int d=0;d<2;++d) {
            result.fe_means[d].resize(plan.storage[d]);const int offset=d ? plan.storage[0] : 0;
            for(int g=0;g<plan.storage[d];++g) result.fe_means[d][g]=plan.alphas[offset+g];
        }
    }
}

inline bool try_cpu(const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const FeView& first,const FeView& second,
    const Eigen::VectorXd* weights,const HdfeOptions& options,AbsorptionResult& output) {
    Plan plan;if(!prepare(y,X,first,second,weights,options,plan)) return false;
    AbsorptionResult result;result.y_tilde.resize(plan.n);result.X_tilde.resize(plan.n,plan.cols);
    {
        ExplicitTeam team(options.num_threads_explicit);
        weighted_two_by_two::Region observed(options.parallel_observer,plan.threads,true);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(plan.threads)
#endif
        {
            bool worked=false;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
            for(int i=0;i<plan.n;++i) {
                if(!worked) {observed.work();worked=true;}
                const int c=plan.cell[i];
                result.y_tilde[i]=weighted_two_by_two::sub(y[i],plan.means[c]);
                for(int j=0;j<plan.cols;++j)
                    result.X_tilde(i,j)=weighted_two_by_two::sub(X(i,j),plan.means[static_cast<std::size_t>(j+1)*plan.cells+c]);
            }
        }
    }
    metadata(plan,options,result);output=std::move(result);return true;
}

}}} // namespace hdfe::detail::additive_cell_projection
