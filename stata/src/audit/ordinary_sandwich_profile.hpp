#ifndef XHDFE_ORDINARY_SANDWICH_PROFILE_HPP
#define XHDFE_ORDINARY_SANDWICH_PROFILE_HPP

#include "ols_numerical_certificate.hpp"
#include "hdfe/deterministic_parallel.hpp"
#include "ols.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>

namespace hdfe { namespace detail {

inline const char* ordinary_sandwich_profile_scope(const HdfeOptions& options,
    const std::vector<Eigen::VectorXi>* clusters,bool full_rank,int n,int p) noexcept {
    if (n<=0 || p<=0) return "empty_design";
    if (!full_rank) return "rank_subset";
    if (options.se_type==StandardErrorType::Homoskedastic) return "homosk";
    if (options.se_type==StandardErrorType::Cluster && (!clusters || clusters->size()!=1))
        return clusters && clusters->size()>1 ? "multiway" : "missing_cluster";
    if (options.se_type==StandardErrorType::Cluster && clusters->front().size()!=n)
        return "cluster_size";
    return nullptr;
}

// These enclosures describe arithmetic on the supplied candidate values.
// No FE/affine error or requested estimator tolerance enters a rounding floor.
struct OrdinarySandwichSum {
    using Real=long double;using I=OlsProofInterval;
    Real sum=0,absolute=0;
    std::uint64_t count=0;
    bool valid=true;
    void add(Real value) noexcept {
        volatile Real next=sum+value,abs_next=absolute+std::abs(value);
        sum=next;absolute=abs_next;++count;
        valid=valid && std::abs(value)<=std::numeric_limits<Real>::max() &&
            std::abs(sum)<=std::numeric_limits<Real>::max() && absolute<=std::numeric_limits<Real>::max();
    }
    static Real gamma(std::uint64_t operations,Real epsilon) noexcept {
        volatile Real numerator=static_cast<Real>(operations)*epsilon;
        if (!(numerator<1)) return std::numeric_limits<Real>::infinity();
        const Real denominator=I::down(1-numerator);
        if (!(denominator>0)) return std::numeric_limits<Real>::infinity();
        return I::up(numerator/denominator);
    }
    Real absolute_upper() const noexcept {
        const Real denominator=I::down(1-gamma(count,std::numeric_limits<Real>::epsilon()));
        if (!valid || !(denominator>0)) return std::numeric_limits<Real>::infinity();
        return I::up(absolute/denominator);
    }
    Real error(int product_operations=0) const noexcept {
        return I::up(gamma(count+product_operations,std::numeric_limits<Real>::epsilon())*
            absolute_upper()+static_cast<Real>(count+product_operations)*std::numeric_limits<Real>::min());
    }
    I interval(int product_operations=0) const noexcept {
        const Real radius=error(product_operations);
        return {I::down(sum-radius),I::up(sum+radius)};
    }
};

// For nonnegative terms, fl(sum products) >= (1-gamma)*exact_sum-U.
// Nonnegative operands can themselves be bounds. Inflate after the positive
// sum, never by scaling a signed score that may have cancellation.
struct OrdinarySandwichPositiveInflation {
    using Real=long double;using I=OlsProofInterval;
    Real multiplier=std::numeric_limits<Real>::infinity(),underflow=0;
    OrdinarySandwichPositiveInflation(std::uint64_t terms,int products) noexcept {
        const Real gamma=OrdinarySandwichSum::gamma(terms+products,std::numeric_limits<Real>::epsilon());
        const Real denominator=I::down(1-gamma);
        if (denominator>0) multiplier=I::up(1/denominator);
        underflow=I::up((static_cast<Real>(terms)*(products+1)+1)*std::numeric_limits<Real>::min());
    }
    Real upper(Real sum) const noexcept {
        if (!(sum>=0 && sum<=std::numeric_limits<Real>::max() &&
              multiplier<=std::numeric_limits<Real>::max())) return std::numeric_limits<Real>::infinity();
        volatile Real numerator=sum+underflow;
        const Real numerator_upper=I::up(numerator);
        volatile Real inflated=numerator_upper*multiplier;
        return I::up(inflated);
    }
};

// Keep both observers balanced even if a registration or end check fails.
struct OrdinarySandwichObservedRegion {
    ParallelWorkObserver local;
    ParallelWorkObserver* shared=nullptr;
    bool local_open=false,shared_open=false;
    void begin(int team,ParallelWorkObserver* observer) {
        shared=observer;local.begin_region(team);local_open=true;
        if (shared) {shared->begin_region(team);shared_open=true;}
    }
    void observe() noexcept {local.observe_work();if (shared_open) shared->observe_work();}
    void finish() {
        std::exception_ptr failure;
        if (shared_open) {
            shared_open=false;
            try {shared->end_region();} catch (...) {failure=std::current_exception();}
        }
        if (local_open) {
            local_open=false;
            try {local.end_region();} catch (...) {if (!failure) failure=std::current_exception();}
        }
        if (failure) std::rethrow_exception(failure);
    }
    ~OrdinarySandwichObservedRegion() {try {finish();} catch (...) {}}
};

struct OrdinarySandwichProfile {
    bool eligible=false,fp64_floor_valid=true;
    const char* reason="unprepared";
    const char* traversal="rows";
    std::uint64_t cells=0,max_cell_rows=0;
    int team=1,workers=1,chunks=0;
    std::size_t workspace_bytes=0,ordering_bytes=0;
    double planning_seconds=0,kernel_seconds=0,total_seconds=0;
    long double max_cell_residual=0;
    std::vector<long double> score_norm,score_lower,score_upper,column_norm;
    std::vector<long double> max_cell_influence,fp64_score_floor,bread_column_radius;
    std::vector<long double> cell_absolute_score_norm_upper;
};

// Requested only by the opt-in final checker. One matrix per deterministic
// chunk, not one matrix per cluster. Means are those of the original design.
struct OrdinarySandwichFullMoments {
    Eigen::VectorXd mean_points;
    std::vector<OlsProofInterval> means;
    const Eigen::VectorXd* response=nullptr;
    std::vector<OlsProofInterval> covariance;
    std::vector<long double> allowance;
    long double residual_formation_bound=-1;
    bool residual_zero=true,response_matches=true;
};

inline OrdinarySandwichProfile ordinary_sandwich_profile(
    const Eigen::MatrixXd& X,const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>* clusters,const HdfeOptions& options,
    const OlsResult& fit,const OlsNumericalCertificate& numerical,
    OrdinarySandwichFullMoments* full=nullptr) {
    if (!ordinary_audit_enabled()) return {};
    ++ordinary_audit_work_count;
    using Real=long double;using I=OlsProofInterval;using Sum=OrdinarySandwichSum;
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();OrdinarySandwichProfile result;
    const int n=static_cast<int>(X.rows()),p=static_cast<int>(X.cols());
    const bool clustered=options.se_type==StandardErrorType::Cluster;
    const int reported=p+1;
    if (full && (weights || full->mean_points.size()!=p ||
        full->means.size()!=static_cast<std::size_t>(p) ||
        (full->response && full->response->size()!=n))) {
        result.reason="full_moments_scope";return result;
    }
    if (full) {full->residual_zero=true;full->response_matches=true;}
    if (const char* reason=ordinary_sandwich_profile_scope(options,clusters,true,n,p)) {
        result.reason=reason;return result;
    }
    if (numerical.bread.size()!=static_cast<std::size_t>(p)*p) {
        result.reason="bread_unavailable";
        return result;
    }
    if (fit.residuals.size()!=n || fit.xtx_inv.rows()!=p || fit.xtx_inv.cols()!=p ||
        (weights && weights->size()!=n)) {result.reason="input_size";return result;}
    // FP64 inputs/products fit without underflow in the extended accumulator.
    // Platforms lacking that range remain observationally unsupported here.
    if (std::numeric_limits<Real>::max_exponent<8*std::numeric_limits<double>::max_exponent ||
        std::numeric_limits<Real>::min_exponent>8*(std::numeric_limits<double>::min_exponent-53)) {
        result.reason="accumulator_range";return result;
    }
#ifdef HDFE_USE_OPENMP
    if (omp_in_parallel()) {result.reason="nested_parallel_context";return result;}
#endif
    const int chunks=deterministic_parallel_chunk_count(n);
    const int team=std::max(1,options.num_threads);
    result.chunks=chunks;
    std::vector<int> order,counts,starts(static_cast<std::size_t>(chunks)+1);
    const Eigen::VectorXi* ids=clustered ? &clusters->front() : nullptr;
    if (ids) {
        bool sorted=true;int minimum=(*ids)[0],maximum=minimum;
        for (int row=1;row<n;++row) {
            sorted=sorted && (*ids)[row]>=(*ids)[row-1];
            minimum=std::min(minimum,(*ids)[row]);maximum=std::max(maximum,(*ids)[row]);
        }
        result.traversal="contiguous_runs";
        if (!sorted) {
            order.resize(n);
            const std::int64_t range=static_cast<std::int64_t>(maximum)-minimum+1;
            if (range<=n) {
                // One flat counter array; stable scatter retains original row order.
                counts.assign(static_cast<std::size_t>(range),0);
                for (int row=0;row<n;++row) ++counts[static_cast<std::int64_t>((*ids)[row])-minimum];
                int position=0;
                for (int& count:counts) {const int next=position+count;count=position;position=next;}
                for (int row=0;row<n;++row) order[counts[static_cast<std::int64_t>((*ids)[row])-minimum]++]=row;
                result.traversal="stable_counted_rows";
            } else {
                std::iota(order.begin(),order.end(),0);
                std::sort(order.begin(),order.end(),[&](int a,int b) {
                    return (*ids)[a]<(*ids)[b] || ((*ids)[a]==(*ids)[b] && a<b);
                });
                result.traversal="stable_sorted_rows";
            }
        }
    }
    auto row_at=[&](int position) {return order.empty() ? position : order[position];};
    if (!ids) {
        for (int chunk=0;chunk<=chunks;++chunk) starts[chunk]=deterministic_parallel_chunk_begin(n,chunk,chunks);
    } else {
        starts[0]=0;int boundary=1;
        for (int position=1;position<n;++position) {
            if ((*ids)[row_at(position)]==(*ids)[row_at(position-1)]) continue;
            while (boundary<chunks && position>=deterministic_parallel_chunk_begin(n,boundary,chunks))
                starts[boundary++]=position;
        }
        while (boundary<=chunks) starts[boundary++]=n;
    }
    result.ordering_bytes=(order.capacity()+counts.capacity()+starts.capacity())*sizeof(int);

    result.score_norm.resize(p);result.score_lower.resize(p);result.score_upper.resize(p);
    result.column_norm.resize(p);result.max_cell_influence.resize(p);
    result.fp64_score_floor.resize(p);result.bread_column_radius.resize(p);
    result.cell_absolute_score_norm_upper.resize(p);
    for (int j=0;j<p;++j) {
        I radius_squared;
        for (int k=0;k<p;++k) {
            const auto entry=numerical.bread[k*p+j];
            const Real point=fit.xtx_inv(k,j);
            if (!(entry.lo<=entry.hi) || !(std::abs(point)<=std::numeric_limits<double>::max())) {
                result.reason="invalid_bread";return result;
            }
            const Real radius=I::up(std::max(std::abs(entry.lo-point),std::abs(entry.hi-point)));
            radius_squared=radius_squared+I::point(radius)*I::point(radius);
        }
        result.bread_column_radius[j]=I::up(std::sqrt(radius_squared.hi));
    }
    struct Chunk {
        std::vector<Sum> sums,absolute_scores,cell_q_absolute;std::vector<Real> influence,cell_bounds;
        Real residual=0;std::uint64_t cells=0,max_rows=0;
        bool valid=true,fp64_valid=true;
        std::vector<Sum> full_cov,full_error,full_allowance,full_absolute;
        std::vector<Real> full_point,full_radius,full_floor,full_abs;
        Sum cell_u;
        bool residual_zero=true,response_matches=true;
    };
    std::vector<Chunk> partial(chunks);
    for (auto& chunk:partial) {
        chunk.sums.resize(static_cast<std::size_t>(7)*p+2);
        chunk.absolute_scores.resize(p);chunk.cell_q_absolute.resize(p);chunk.influence.resize(p);
        chunk.cell_bounds.resize(static_cast<std::size_t>(4)*p);
        if (full) {
            const auto square=static_cast<std::size_t>(reported)*reported;
            chunk.full_cov.resize(square);chunk.full_error.resize(square);
            chunk.full_allowance.resize(square);chunk.full_absolute.resize(square);
            chunk.full_point.resize(reported);chunk.full_radius.resize(reported);
            chunk.full_floor.resize(reported);chunk.full_abs.resize(reported);
        }
    }
    result.workspace_bytes=result.ordering_bytes+partial.capacity()*sizeof(Chunk)+
        (result.score_norm.capacity()+result.score_lower.capacity()+result.score_upper.capacity()+
         result.column_norm.capacity()+result.max_cell_influence.capacity()+result.fp64_score_floor.capacity()+
         result.bread_column_radius.capacity()+result.cell_absolute_score_norm_upper.capacity())*sizeof(Real)+
        numerical.bread.capacity()*sizeof(I);
    for (const auto& chunk:partial) result.workspace_bytes+=
        (chunk.sums.capacity()+chunk.absolute_scores.capacity()+chunk.cell_q_absolute.capacity())*sizeof(Sum)+
        (chunk.influence.capacity()+chunk.cell_bounds.capacity())*sizeof(Real)+
        (chunk.full_cov.capacity()+chunk.full_error.capacity()+chunk.full_allowance.capacity()+
         chunk.full_absolute.capacity())*sizeof(Sum)+
        (chunk.full_point.capacity()+chunk.full_radius.capacity()+chunk.full_floor.capacity()+
         chunk.full_abs.capacity())*sizeof(Real);
    // For qhat=fl(sum x*b), ahat=fl(sum abs(x*b)), |q-qhat| <= g*ahat/(1-g).
    // Apply the resulting norm bound once per cell, not once per observation.
    const Real q_gamma=Sum::gamma(static_cast<std::uint64_t>(p)+1,std::numeric_limits<Real>::epsilon());
    if (!(q_gamma<1)) {result.reason="dot_rounding_range";return result;}
    const Real q_denominator=I::down(1-q_gamma);
    if (!(q_denominator>0)) {result.reason="dot_rounding_range";return result;}
    const Real q_rounding_multiplier=I::up(q_gamma/q_denominator);
    const OrdinarySandwichPositiveInflation positive_p(p,1),absolute_p(p,0),positive_two(2,1),
        positive_three(3,1),positive_four(4,1);
    const Real dot_gamma_fp64=Sum::gamma(static_cast<std::uint64_t>(p)+2,std::numeric_limits<double>::epsilon());
    const Real dot_fp64_multiplier=I::up(1+dot_gamma_fp64);
    const Real dot_tiny=I::up((static_cast<Real>(p)+1)*std::numeric_limits<Real>::min());
    const Real dot_fp64_tiny=I::up((static_cast<Real>(p)+2)*std::numeric_limits<double>::min());
    OrdinarySandwichObservedRegion observed;
    result.planning_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    const auto kernel_start=Clock::now();
    observed.begin(team,options.parallel_observer);
    result.workspace_bytes+=observed.local.worker_capacity()*sizeof(std::atomic<std::uint64_t>);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(team)
#endif
    {
#ifdef HDFE_USE_OPENMP
#pragma omp single
        result.team=omp_get_num_threads();
#pragma omp for schedule(static)
#endif
        for (int chunk=0;chunk<chunks;++chunk) {
            if (starts[chunk]==starts[chunk+1]) continue;
            observed.observe();auto& part=partial[chunk];
            auto* sums=part.sums.data();auto* cell_y=sums+5*p;auto* cell_q=sums+6*p;
            auto& cell_residual=sums[7*p];auto& cell_design=sums[7*p+1];
            auto normal_fp64=[](Real value) {
                const Real magnitude=std::abs(value);
                return magnitude==0 || (magnitude>=std::numeric_limits<double>::min() &&
                    magnitude<=std::numeric_limits<double>::max());
            };
            for (int begin=starts[chunk];begin<starts[chunk+1];) {
                int end=begin+1;
                if (ids) while (end<starts[chunk+1] && (*ids)[row_at(end)]==(*ids)[row_at(begin)]) ++end;
                bool cell_residual_zero=true;
                for (int position=begin;position<end;++position) {
                    const int row=row_at(position);const Real w=weights ? (*weights)[row] : 1.0L;
                    const Real residual=fit.residuals[row],root=weights ? std::sqrt(w) : 1.0L;
                    const bool frequency_hc=options.weights_are_frequencies && !clustered;
                    const Real omega=frequency_hc ? root : w,alpha=frequency_hc ? 1.0L : w;
                    volatile Real scaled_residual=omega*residual,residual_square=residual*residual;
                    volatile Real residual_energy=alpha*residual_square;
                    cell_residual.add(residual_energy);
                    if (full) {
                        part.cell_u.add(residual);
                        const auto stored=ieee_bits_detail::bits(fit.residuals[row]);
                        const bool zero=(stored&0x7fffffffffffffffULL)==0;
                        cell_residual_zero=cell_residual_zero && zero;
                        part.residual_zero=part.residual_zero && zero;
                        if (full->response) {
                            const auto input=ieee_bits_detail::bits((*full->response)[row]);
                            part.response_matches=part.response_matches &&
                                (stored==input || ((stored|input)&0x7fffffffffffffffULL)==0);
                        } else part.response_matches=false;
                    }
                    part.valid=part.valid && w>0;
                    const Real native_residual=(weights && !frequency_hc) ? root*residual : residual;
                    part.fp64_valid=part.fp64_valid && normal_fp64(native_residual);
                    for (int j=0;j<p;++j) {
                        const Real x=X(row,j);
                        volatile Real term=scaled_residual*x,square=x*x,column_energy=w*square,cell_energy=alpha*square;
                        cell_y[j].add(term);sums[j].add(column_energy);cell_design.add(cell_energy);
                        part.fp64_valid=part.fp64_valid && normal_fp64(root*x) && normal_fp64(term);
                        Real q=0,absolute=0;
                        for (int k=0;k<p;++k) {
                            volatile Real product=static_cast<Real>(X(row,k))*fit.xtx_inv(k,j);
                            volatile Real next=q+product,absolute_next=absolute+std::abs(product);
                            q=next;absolute=absolute_next;
                        }
                        volatile Real q_square=q*q,q_energy=alpha*q_square;
                        volatile Real absolute_square=absolute*absolute,absolute_energy=alpha*absolute_square;
                        cell_q[j].add(q_energy);
                        part.cell_q_absolute[j].add(absolute_energy);
                    }
                }
                const auto cell_rows=static_cast<std::uint64_t>(end-begin);
                // The existing exponent gate prevents intermediate underflow
                // in these two-product energy terms formed from FP64 data.
                const OrdinarySandwichPositiveInflation cell_absolute(cell_rows,0),cell_energy(cell_rows,2),
                    design_energy(cell_rows*static_cast<std::uint64_t>(p),2);
                const Real gamma_y=Sum::gamma(cell_rows+6,std::numeric_limits<Real>::epsilon());
                // Shared by every k and j: six operations cover sqrt-weight
                // factors and products in the conditional FP64 score model.
                const Real gamma_y_fp64=Sum::gamma(cell_rows+6,std::numeric_limits<double>::epsilon());
                const Real y_tiny=I::up(static_cast<Real>(cell_rows+6)*std::numeric_limits<Real>::min());
                const Real y_fp64_tiny=I::up(static_cast<Real>(cell_rows+6)*std::numeric_limits<double>::min());
                const Real design_norm=I::up(std::sqrt(design_energy.upper(cell_design.sum)));
                auto* absolute_y=part.cell_bounds.data();auto* error_y=absolute_y+p;
                auto* absolute_bound_y=error_y+p;auto* fp64_y=absolute_bound_y+p;
                Real y_squared=0;
                for (int k=0;k<p;++k) {
                    absolute_y[k]=cell_absolute.upper(cell_y[k].absolute);
                    volatile Real error_product=gamma_y*absolute_y[k],error_sum=error_product+y_tiny;
                    error_y[k]=positive_two.upper(error_sum);
                    volatile Real absolute_sum=absolute_y[k]+error_y[k];
                    absolute_bound_y[k]=positive_two.upper(absolute_sum);
                    volatile Real fp64_product=gamma_y_fp64*absolute_y[k],fp64_first=fp64_product+error_y[k];
                    volatile Real fp64_sum=fp64_first+y_fp64_tiny;
                    fp64_y[k]=positive_three.upper(fp64_sum);
                    volatile Real upper_sum=std::abs(cell_y[k].sum)+error_y[k];
                    const Real upper=positive_two.upper(upper_sum);
                    volatile Real square=upper*upper,next=y_squared+square;y_squared=next;
                }
                const Real y_norm=I::up(std::sqrt(positive_p.upper(y_squared)));
                for (int j=0;j<p;++j) {
                    // T = Y B. Separate accumulation error from the interval
                    // radius of B; neither includes any FE projection error.
                    Sum dot;Real arithmetic=0,fp64=0,absolute_score=0;
                    for (int k=0;k<p;++k) {
                        const Real bread=fit.xtx_inv(k,j);
                        volatile Real term=cell_y[k].sum*bread;dot.add(term);
                        volatile Real arithmetic_term=std::abs(bread)*error_y[k],arithmetic_next=arithmetic+arithmetic_term;
                        volatile Real absolute_term=numerical.bread[k*p+j].absolute()*absolute_bound_y[k];
                        volatile Real absolute_next=absolute_score+absolute_term;
                        volatile Real fp64_term=std::abs(bread)*fp64_y[k],fp64_next=fp64+fp64_term;
                        arithmetic=arithmetic_next;absolute_score=absolute_next;fp64=fp64_next;
                        part.fp64_valid=part.fp64_valid && normal_fp64(term);
                    }
                    const Real absolute_dot=absolute_p.upper(dot.absolute);
                    volatile Real dot_error_product=q_gamma*absolute_dot,dot_error_sum=dot_error_product+dot_tiny;
                    const Real dot_error=positive_two.upper(dot_error_sum);
                    const Real arithmetic_upper=positive_p.upper(arithmetic),fp64_upper=positive_p.upper(fp64);
                    const Real absolute_score_upper=positive_p.upper(absolute_score);
                    volatile Real bread_error=y_norm*result.bread_column_radius[j];
                    volatile Real error_first=arithmetic_upper+dot_error,error_sum=error_first+bread_error;
                    const Real error=positive_three.upper(error_sum);
                    const Real lower=std::max(0.0L,I::down(std::abs(dot.sum)-error));
                    const Real upper=I::up(std::abs(dot.sum)+error);
                    volatile Real propagated_fp64=fp64_upper*dot_fp64_multiplier,propagated_dot=dot_gamma_fp64*absolute_dot;
                    volatile Real floor_first=propagated_fp64+propagated_dot,floor_second=floor_first+dot_error;
                    volatile Real floor_sum=floor_second+dot_fp64_tiny;
                    const Real floor=positive_four.upper(floor_sum);
                    if (full) {
                        part.full_point[j]=dot.sum;part.full_radius[j]=error;
                        part.full_floor[j]=floor;
                    }
                    volatile Real point_square=dot.sum*dot.sum,lower_square=lower*lower,upper_square=upper*upper,floor_square=floor*floor;
                    sums[p+j].add(point_square);sums[2*p+j].add(lower_square);
                    sums[3*p+j].add(upper_square);sums[4*p+j].add(floor_square);
                    volatile Real absolute_square=absolute_score_upper*absolute_score_upper;
                    part.absolute_scores[j].add(absolute_square);
                    // ||U_g B_j|| <= ||U_g b_j|| + ||U_g||_F ||B_j-b_j||,
                    // with the HC/frequency or cluster metric alpha applied.
                    const Real q_norm=I::up(std::sqrt(cell_energy.upper(cell_q[j].sum)));
                    const Real absolute_norm=I::up(std::sqrt(cell_energy.upper(part.cell_q_absolute[j].sum)));
                    volatile Real q_error=q_rounding_multiplier*absolute_norm,delta=design_norm*result.bread_column_radius[j];
                    volatile Real enlarged_first=q_norm+q_error,enlarged_sum=enlarged_first+delta;
                    const Real enlarged=positive_three.upper(enlarged_sum);
                    volatile Real enlarged_square=enlarged*enlarged;
                    part.influence[j]=std::max(part.influence[j],I::up(enlarged_square));
                }
                if (full) {
                    const I inv_n{I::down(1.0L/n),I::up(1.0L/n)};
                    I intercept=part.cell_u.interval()*inv_n;
                    Sum point;point.add(part.cell_u.sum/static_cast<Real>(n));
                    Real floor=I::up((gamma_y_fp64*part.cell_u.absolute_upper()+y_fp64_tiny)/n);
                    for (int j=0;j<p;++j) {
                        const I score{I::down(part.full_point[j]-part.full_radius[j]),
                                      I::up(part.full_point[j]+part.full_radius[j])};
                        intercept=intercept-full->means[j]*score;
                        point.add(-static_cast<Real>(full->mean_points[j])*part.full_point[j]);
                        floor=I::up(floor+I::up(std::abs(static_cast<Real>(full->mean_points[j]))*part.full_floor[j]));
                    }
                    part.full_point[p]=point.sum;
                    part.full_radius[p]=I::up(std::max(std::abs(point.sum-intercept.lo),std::abs(point.sum-intercept.hi)));
                    part.full_floor[p]=I::up(floor+I::up(dot_gamma_fp64*point.absolute_upper())+dot_fp64_tiny);
                    for (int j=0;j<reported;++j) {
                        // All-zero stored residuals make these candidate scores
                        // exactly zero; do not manufacture an arithmetic floor.
                        if (cell_residual_zero) {
                            part.full_point[j]=0;part.full_radius[j]=0;part.full_floor[j]=0;
                        }
                        part.full_abs[j]=part.full_point[j]==0 && part.full_radius[j]==0 ? 0 :
                            I::up(std::abs(part.full_point[j])+part.full_radius[j]);
                    }
                    for (int j=0;j<reported;++j) for (int k=0;k<reported;++k) {
                        const auto index=static_cast<std::size_t>(j)*reported+k;
                        volatile Real value=part.full_point[j]*part.full_point[k];
                        volatile Real error1=std::abs(part.full_point[j])*part.full_radius[k],
                            error2=part.full_radius[j]*std::abs(part.full_point[k]),
                            error3=part.full_radius[j]*part.full_radius[k],error12=error1+error2,error=error12+error3;
                        volatile Real floor1=part.full_abs[j]*part.full_floor[k],
                            floor2=part.full_floor[j]*part.full_abs[k],
                            floor3=part.full_floor[j]*part.full_floor[k],floor12=floor1+floor2,allowance=floor12+floor3;
                        volatile Real absolute=part.full_abs[j]*part.full_abs[k];
                        part.full_cov[index].add(value);part.full_error[index].add(error);
                        part.full_allowance[index].add(allowance);part.full_absolute[index].add(absolute);
                    }
                    part.valid=part.valid && part.cell_u.valid;part.cell_u=Sum{};
                }
                part.residual=std::max(part.residual,cell_energy.upper(cell_residual.sum));
                for (int k=0;k<2*p+2;++k) part.valid=part.valid && cell_y[k].valid;
                for (const auto& sum:part.cell_q_absolute) part.valid=part.valid && sum.valid;
                ++part.cells;part.max_rows=std::max(part.max_rows,static_cast<std::uint64_t>(end-begin));
                std::fill(cell_y,cell_y+2*p+2,Sum{});
                std::fill(part.cell_q_absolute.begin(),part.cell_q_absolute.end(),Sum{});begin=end;
            }
        }
    }
    observed.finish();result.workers=observed.local.active_workers();
    result.kernel_seconds=std::chrono::duration<double>(Clock::now()-kernel_start).count();
    for (const auto& part:partial) {
        if (!part.valid) {result.reason="invalid_input_or_accumulation";return result;}
        for (const auto& sum:part.sums) if (!sum.valid) {result.reason="accumulation_range";return result;}
        for (const auto& sum:part.absolute_scores) if (!sum.valid) {result.reason="accumulation_range";return result;}
    }
    for (int j=0;j<p;++j) {
        I energy,lower,upper,rounding,absolute_score;Sum point;
        for (const auto& part:partial) {
            result.fp64_floor_valid=result.fp64_floor_valid && part.fp64_valid;
            energy=energy+part.sums[j].interval(2);point.add(part.sums[p+j].sum);
            lower=lower+part.sums[2*p+j].interval(1);upper=upper+part.sums[3*p+j].interval(1);
            rounding=rounding+part.sums[4*p+j].interval(1);
            absolute_score=absolute_score+part.absolute_scores[j].interval(1);
            result.max_cell_influence[j]=std::max(result.max_cell_influence[j],part.influence[j]);
        }
        result.column_norm[j]=I::up(std::sqrt(std::max(0.0L,energy.hi)));
        result.score_norm[j]=std::sqrt(std::max(0.0L,point.sum));
        result.score_lower[j]=std::max(0.0L,I::down(std::sqrt(std::max(0.0L,lower.lo))));
        result.score_upper[j]=I::up(std::sqrt(std::max(0.0L,upper.hi)));
        result.fp64_score_floor[j]=I::up(std::sqrt(std::max(0.0L,rounding.hi)));
        result.cell_absolute_score_norm_upper[j]=I::up(std::sqrt(std::max(0.0L,absolute_score.hi)));
        result.fp64_floor_valid=result.fp64_floor_valid &&
            result.fp64_score_floor[j]<=std::numeric_limits<Real>::max();
        if (!(result.score_upper[j]<=std::numeric_limits<Real>::max()) ||
            !(result.column_norm[j]<=std::numeric_limits<Real>::max()) ||
            !(result.max_cell_influence[j]<=std::numeric_limits<Real>::max()) ||
            !(result.cell_absolute_score_norm_upper[j]<=std::numeric_limits<Real>::max())) {
            result.reason="accumulation_range";return result;
        }
    }
    for (const auto& part:partial) {
        result.cells+=part.cells;result.max_cell_rows=std::max(result.max_cell_rows,part.max_rows);
        result.max_cell_residual=std::max(result.max_cell_residual,part.residual);
    }
    if (full) {
        const auto square=static_cast<std::size_t>(reported)*reported;
        full->covariance.assign(square,I{});full->allowance.assign(square,0);
        const Real meat_gamma=Sum::gamma(result.cells+1,std::numeric_limits<double>::epsilon());
        for (const auto& part:partial) {
            full->residual_zero=full->residual_zero && part.residual_zero;
            full->response_matches=full->response_matches && part.response_matches;
        }
        for (std::size_t index=0;index<square;++index) {
            I point,error,allowance,absolute;
            for (const auto& part:partial) {
                if (!part.full_cov[index].valid || !part.full_error[index].valid ||
                    !part.full_allowance[index].valid || !part.full_absolute[index].valid) {
                    result.reason="full_moments_range";return result;
                }
                point=point+part.full_cov[index].interval(1);
                error=error+part.full_error[index].interval(5);
                allowance=allowance+part.full_allowance[index].interval(5);
                absolute=absolute+part.full_absolute[index].interval(1);
            }
            full->covariance[index]=point+I{-error.hi,error.hi};
            full->allowance[index]=I::up(allowance.hi+I::up(meat_gamma*absolute.hi));
            if (full->residual_zero) {full->covariance[index]={};full->allowance[index]=0;}
            if (!(full->covariance[index].absolute()<=std::numeric_limits<Real>::max()) ||
                !(full->allowance[index]>=0 && full->allowance[index]<=std::numeric_limits<Real>::max())) {
                result.reason="full_moments_range";return result;
            }
        }
        result.workspace_bytes+=square*(sizeof(I)+sizeof(Real));
    }
    if (!(result.max_cell_residual<=std::numeric_limits<Real>::max())) {
        result.reason="residual_energy_range";return result;
    }
    result.eligible=true;result.reason="ok";
    result.total_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    return result;
}

}}
#endif
