// Included immediately after certify_absorption_result while hdfe::detail is
// open. The namespace dance keeps dependency headers at global scope while
// retaining access to this translation unit's unnamed helpers.
} }
#include "ordinary_cycle_reference.hpp"
extern "C"
#if defined(__GNUC__)
__attribute__((visibility("default")))
#endif
unsigned long long xhdfe_private_n05_work_count(int reset) noexcept {
    const auto count=hdfe::detail::ordinary_audit_work_count;
    if (reset) hdfe::detail::ordinary_audit_work_count=0;
    return count;
}
namespace hdfe { namespace detail {

struct OrdinaryPrecisionCheck {
    double relative=0.0;
    double absolute=0.0;
    bool finite=true;
    bool passed=true;
};

static OrdinaryPrecisionCheck ordinary_audit_check_precision(
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const HdfeOptions& options,const AbsorptionResult& result,
    const Eigen::MatrixXd* directions) {
    OrdinaryPrecisionCheck check;
    if(fes.empty()) return check;
    const int n=static_cast<int>(result.y_tilde.size()),raw_rhs=static_cast<int>(result.X_tilde.cols())+1;
    const int rhs=raw_rhs+(directions ? static_cast<int>(directions->cols()) : 0);
    if(!n || result.X_tilde.rows()!=n || !ieee_all_finite(result.y_tilde) ||
       !ieee_all_finite(result.X_tilde)) {check.finite=check.passed=false;return check;}
    for(const auto& fe:fes) if(fe.size()!=n) {check.finite=check.passed=false;return check;}
    if(weights && weights->size()!=n) {check.finite=check.passed=false;return check;}
    const int threads=std::max(1,options.num_threads);
    auto value=[&](int row,int column) {
        if (column<raw_rhs) return column ? result.X_tilde(row,column-1) : result.y_tilde[row];
        GroupForwardSum sum;
        for (int j=0;j<result.X_tilde.cols();++j)
            sum.add(static_cast<long double>((*directions)(j,column-raw_rhs))*result.X_tilde(row,j));
        return static_cast<double>(sum.value());
    };
    std::vector<double> scales(rhs,1.0),rms(rhs);
    const double total=weights ? deterministic_chunked_sum<int>(n,threads,[&](int i) {return (*weights)[i];},options.parallel_observer) : n;
    if(!(total>0) || !ieee_finite(total)) {check.finite=check.passed=false;return check;}
    for(int column=0;column<rhs;++column) {
        double largest=0;
        if(options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(threads) reduction(max:largest)
#endif
        {
            bool observed=false;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
            for(int row=0;row<n;++row) {
                if(!observed && options.parallel_observer) options.parallel_observer->observe_work();
                observed=true;
                largest=std::max(largest,std::abs(value(row,column)));
            }
        }
        if(options.parallel_observer) options.parallel_observer->end_region();
        if(largest>0) scales[column]=std::ldexp(1.0,std::max(-900,std::min(900,std::ilogb(largest))));
        const double squared=deterministic_chunked_sum<int>(n,threads,[&](int row) {
            const double scaled=value(row,column)/scales[column];
            return (weights ? (*weights)[row] : 1.0)*scaled*scaled;
        },options.parallel_observer);
        rms[column]=std::sqrt(squared/total);
        check.finite=check.finite && ieee_finite(rms[column]);
    }
    ParallelWorkObserver certificate_observer;
    certificate_observer.reset();
    for(const auto& fe:fes) {
        const FeIndexer index=build_indexer(fe);
        std::vector<int> ptr(index.num_groups+1,0),order(n);
        for(int group:index.group_ids) ++ptr[group+1];
        std::partial_sum(ptr.begin(),ptr.end(),ptr.begin());
        auto cursor=ptr;
        for(int row=0;row<n;++row) order[cursor[index.group_ids[row]]++]=row;
        const int logical=deterministic_parallel_chunk_count(n);
        const int segments=std::max(1,(logical+index.num_groups-1)/index.num_groups);
        const std::int64_t units=static_cast<std::int64_t>(index.num_groups)*segments;
        std::vector<long double> partial(static_cast<std::size_t>(units)*(rhs+1),0.0L);
        if(options.parallel_observer) options.parallel_observer->begin_region(threads);
        certificate_observer.begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(std::int64_t unit=0;unit<units;++unit) {
            const int group=static_cast<int>(unit/segments),part=static_cast<int>(unit%segments);
            const int size=ptr[group+1]-ptr[group];
            const int first=ptr[group]+static_cast<int>(static_cast<std::int64_t>(size)*part/segments);
            const int last=ptr[group]+static_cast<int>(static_cast<std::int64_t>(size)*(part+1)/segments);
            if(first<last && options.parallel_observer) options.parallel_observer->observe_work();
            if(first<last) certificate_observer.observe_work();
            std::vector<GroupForwardSum> sums(rhs+1);
            for(int position=first;position<last;++position) {
                const int row=order[position];
                const long double weight=weights ? (*weights)[row] : 1.0L;
                sums[rhs].add(weight);
                for(int column=0;column<rhs;++column)
                    sums[column].add(weight*(static_cast<long double>(value(row,column))/scales[column]));
            }
            for(int column=0;column<=rhs;++column)
                partial[static_cast<std::size_t>(unit)*(rhs+1)+column]=sums[column].value();
        }
        if(options.parallel_observer) options.parallel_observer->end_region();
        certificate_observer.end_region();
        for(int group=0;group<index.num_groups;++group) {
            std::vector<GroupForwardSum> sums(rhs+1);
            for(int part=0;part<segments;++part) for(int column=0;column<=rhs;++column)
                sums[column].add(partial[(static_cast<std::size_t>(group)*segments+part)*(rhs+1)+column]);
            const long double weight=sums[rhs].value();
            if(!(weight>0)) continue;
            for(int column=0;column<rhs;++column) {
                const double mean=static_cast<double>(std::abs(sums[column].value()/weight));
                const double relative=rms[column]>0 ? mean/rms[column] : mean>0 ? std::numeric_limits<double>::max() : 0;
                const double absolute=mean*scales[column];
                check.finite=check.finite && ieee_finite(relative) && ieee_finite(absolute);
                check.relative=std::max(check.relative,relative);
                check.absolute=std::max(check.absolute,absolute);
            }
        }
    }
    const double floor=64*std::numeric_limits<double>::epsilon();
    check.passed=check.finite && check.relative<=std::max(effective_absorption_tolerance(options),floor);
    if(options.tolerance_mode==ToleranceMode::StrictResidual)
        check.passed=check.passed && check.absolute<=std::max(options.tol,floor);
    (void)certificate_observer;
    return check;
}

OrdinaryAuditFamily run_ordinary_n2_audit(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const HdfeOptions& options,const AbsorptionResult& result,
    const std::vector<Eigen::VectorXi>* clusters,bool verify_regression) {
    if (!ordinary_audit_enabled()) return {OrdinaryAuditFinding::Unsupported,"audit_disabled"};
    ++ordinary_audit_work_count;
    if (fes.empty()) return {OrdinaryAuditFinding::Unsupported,"no_fixed_effects"};
    if (!result.converged || !result.precision_certified)
        return {OrdinaryAuditFinding::Unsupported,"candidate_not_accepted"};
    // Independent audit work must not replace the fit's useful-worker diagnostics.
    HdfeOptions audit_options=options;audit_options.parallel_observer=nullptr;
    try {
        Eigen::MatrixXd directions;
        std::vector<int> all(X.cols());std::iota(all.begin(),all.end(),0);
        precise_ols_rank_mask(result.X_tilde,all,weights,false,
            64*std::numeric_limits<double>::epsilon(),options.collinear_priority,
            std::max(1,options.num_threads),nullptr,&directions);
        const auto check=ordinary_audit_check_precision(fes,weights,audit_options,result,&directions);
        OrdinaryThreeFeSupportWorkspace workspace;
        check_ordinary_cycle_reference(y,X,fes,weights,audit_options,result,clusters,
            verify_regression,&workspace);
        if (!check.passed)
            return {OrdinaryAuditFinding::BoundInconclusive,"legacy_mean_bound"};
        if (fes.size()==1 && !exact_level_projection(y,X,fes[0],weights,result.y_tilde,result.X_tilde))
            return {OrdinaryAuditFinding::BoundInconclusive,"one_fe_affine_bound_unavailable"};
        // The donor skips unweighted projection-only cycle work. Do not call
        // that shortcut a full projection proof merely because means pass.
        if (!verify_regression && fes.size()>1) {
            const long double limit=std::max(effective_absorption_tolerance(options),
                64*std::numeric_limits<double>::epsilon());
            std::vector<long double> targets(X.cols()+1,0);
            for (int rhs=0;rhs<=X.cols();++rhs) {
                long double norm=0;
                for (int row=0;row<y.size();++row)
                    norm=std::hypot(norm,std::sqrt(weights ? (*weights)[row] : 1.0L)*
                        (rhs ? result.X_tilde(row,rhs-1) : result.y_tilde[row]));
                targets[rhs]=limit*norm;
            }
            bool proved=false;
            if (fes.size()==2) {
                const auto tree=certify_ordinary_tree_space(y,X,fes,weights,result,targets);
                proved=tree.eligible && tree.passed;
            } else if (fes.size()==3) {
                const auto support=certify_ordinary_three_fe_support(y,X,fes,weights,result,targets,
                    &workspace,options.num_threads,nullptr);
                proved=support.eligible && support.passed;
            }
            if (!proved) return {OrdinaryAuditFinding::BoundInconclusive,"projection_only_bound"};
        }
        return {OrdinaryAuditFinding::Passed,"v82_projection_proof"};
    } catch (const OrdinaryReferenceDisagreement& error) {
        const std::string reason=error.what();
        // The historical Fast gap and auxiliary residual/quarter-bounds are
        // not estimator errors. Only separated b/V intervals use this class.
        const bool inference_separation=reason.find("coefficient_separation")!=std::string::npos ||
            reason.find("covariance_separation")!=std::string::npos;
        if (options.tolerance_mode==ToleranceMode::ReghdfeComparable && inference_separation)
            return {OrdinaryAuditFinding::ErrorDemonstrated,reason};
        return {OrdinaryAuditFinding::BoundInconclusive,reason};
    } catch (const std::exception& error) {
        return {OrdinaryAuditFinding::BoundInconclusive,error.what()};
    }
}
