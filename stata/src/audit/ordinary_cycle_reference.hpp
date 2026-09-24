#ifndef XHDFE_ORDINARY_CYCLE_REFERENCE_HPP
#define XHDFE_ORDINARY_CYCLE_REFERENCE_HPP

#include <Eigen/LU>
#include <map>
#include "ordinary_tree_certificate.hpp"
#include "ols_numerical_certificate.hpp"
#include "ordinary_three_fe_support.hpp"
#include "ordinary_certification.hpp"

namespace hdfe { namespace detail {

inline bool exact_ordinary_two_fe_projection(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const AbsorptionResult& result) {
    // Exact normal equations plus an exact affine witness determine the
    // projection even on a large graph. No rank factorization is needed.
    for (const auto& fe:fes)
        if (!exact_level_projection(result.y_tilde,result.X_tilde,fe,weights,result.y_tilde,result.X_tilde)) return false;
    Eigen::VectorXd difference(y.size());
    for (int rhs=0;rhs<=X.cols();++rhs) {
        bool nonzero=false;
        for (int row=0;row<y.size();++row) {
            const double input=rhs ? X(row,rhs-1) : y[row];
            const double residual=rhs ? result.X_tilde(row,rhs-1) : result.y_tilde[row];
            volatile double value=input-residual;
            GroupExactDyadicSum exact;
            exact.add(input);exact.add(residual,1,true);exact.add(value,1,true);
            if (!exact.exactly_zero()) return false;
            difference[row]=value;nonzero=nonzero || value!=0;
        }
        if (nonzero && !exact_two_fe_span(difference,fes[0],fes[1])) return false;
    }
    return true;
}

inline OlsResult ordinary_projection_fit(const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>* clusters,const HdfeOptions& options) {
    const double effective=options.weights_are_frequencies && weights ? weights->sum() : y.size();
    if (options.se_type==StandardErrorType::Cluster && clusters && clusters->size()>1)
        return run_ols_multiway(y,X,weights,clusters,options.se_type,0,0,
            options.stats_style==StatsStyle::Reghdfe ? ClusterDofMethod::Conventional : options.ssc_g_df,
            options.stats_style==StatsStyle::Reghdfe ? true : options.ssc_g_adj,effective,nullptr,
            options.num_threads_explicit,options.parallel_observer);
    return run_ols(y,X,weights,clusters && !clusters->empty() ? &clusters->front() : nullptr,
        options.se_type,0,0,effective,options.weights_are_frequencies,nullptr,
        options.num_threads_explicit,options.parallel_observer);
}

inline void tighten_ordinary_residual_target(const AbsorptionResult& result,
    const Eigen::VectorXd* weights,const HdfeOptions& options,
    std::vector<long double>& targets) {
    const int p=static_cast<int>(result.X_tilde.cols());
    if (!p) return;
    std::vector<int> columns(p),kept;std::iota(columns.begin(),columns.end(),0);
    Eigen::MatrixXd full_basis;
    const auto mask=precise_ols_rank_mask(result.X_tilde,columns,weights,false,
        64*std::numeric_limits<double>::epsilon(),options.collinear_priority,
        std::max(1,options.num_threads),options.parallel_observer,nullptr,&full_basis);
    for (int j=0;j<p;++j) if (!mask[j]) kept.push_back(j);
    if (kept.empty()) return;
    Eigen::MatrixXd X(result.X_tilde.rows(),kept.size());
    for (std::size_t j=0;j<kept.size();++j) X.col(j)=result.X_tilde.col(kept[j]);
    HdfeOptions coefficient_options=options;coefficient_options.se_type=StandardErrorType::Homoskedastic;
    const auto fit=ordinary_projection_fit(result.y_tilde,X,weights,nullptr,coefficient_options);
    Eigen::MatrixXd basis(kept.size(),full_basis.cols());
    for (std::size_t j=0;j<kept.size();++j) basis.row(j)=full_basis.row(kept[j]);
    const auto numerical=certify_ols_numerics(result.y_tilde,X,weights,fit.coefficients,basis,
        std::max(1,options.num_threads),options.parallel_observer);
    const long double coefficient_limit=std::max(effective_absorption_tolerance(options),
        64*std::numeric_limits<double>::epsilon());
    if (numerical.coefficient_error>coefficient_limit/4) {
        std::ostringstream message;
        message << "Ordinary FE coefficient solve could not be certified: error bound="
                << std::scientific << numerical.coefficient_error
                << "; allowed OLS error=" << coefficient_limit/4
                << ". No estimates returned.";
        throw std::runtime_error(message.str());
    }
    const long double tolerance=std::max(options.tol,64*std::numeric_limits<double>::epsilon());
    long double residual_norm=0,response_norm=0,prediction_norm=0;
    for (int row=0;row<X.rows();++row) {
        const long double weight=weights ? (*weights)[row] : 1.0L;
        residual_norm=std::hypot(residual_norm,std::sqrt(weight)*fit.residuals[row]);
        response_norm=std::hypot(response_norm,std::sqrt(weight)*result.y_tilde[row]);
        prediction_norm=std::hypot(prediction_norm,std::sqrt(weight)*
            (static_cast<long double>(result.y_tilde[row])-fit.residuals[row]));
    }
    residual_norm=numerical.residual_norm;
    long double budget=tolerance*residual_norm/16;
    if (options.se_type!=StandardErrorType::Homoskedastic) {
        for (int j=0;j<X.cols();++j) {
            long double score_norm=0,maximum=0;
            for (int row=0;row<X.rows();++row) {
                GroupForwardSum sum;
                for (int k=0;k<X.cols();++k)
                    sum.add(static_cast<long double>(X(row,k))*fit.xtx_inv(k,j));
                const long double q=sum.value(),weight=weights ? (*weights)[row] : 1.0L;
                const long double score_weight=options.weights_are_frequencies ? std::sqrt(weight) : weight;
                score_norm=std::hypot(score_norm,score_weight*q*fit.residuals[row]);
                const long double factor=std::abs(q)*(options.weights_are_frequencies ? 1.0L : std::sqrt(weight));
                maximum=std::max(maximum,factor);
            }
            if (maximum>0) budget=std::min(budget,tolerance*score_norm/(16*X.cols()*maximum));
        }
    }
    // A globally numerical-perfect fit has no meaningful relative noise
    // target below the arithmetic floor. Local low-noise rows in an otherwise
    // noisy regression do not qualify for this exception.
    const long double floor=64*std::numeric_limits<double>::epsilon()*(response_norm+prediction_norm);
    if (residual_norm<=floor) budget=std::max(budget,floor);
    targets[0]=std::min(targets[0],budget/2);
    for (std::size_t j=0;j<kept.size();++j) {
        const long double coefficient=std::abs(static_cast<long double>(fit.coefficients[j]));
        if (coefficient>0)
            targets[kept[j]+1]=std::min(targets[kept[j]+1],budget/(2*kept.size()*coefficient));
    }
    // Projection accuracy must also resolve the coefficients. Let s be a
    // lower bound for sigma_min(sqrt(W) X). If ||E|| <= s/8, perturbing both
    // X and y changes beta by at most
    //   ||e-E beta||/(s-||E||) + ||E|| ||r||/(s-||E||)^2.
    // Residual/noise targets alone miss this amplification when X columns
    // are almost collinear. The fixed margins account for rounding of the
    // verified basis, with an independent bound for the OLS solve itself.
    const long double singular_lower=numerical.singular_lower;
    long double beta_norm=0;
    for (int j=0;j<X.cols();++j) beta_norm=std::hypot(beta_norm,static_cast<long double>(fit.coefficients[j]));
    beta_norm+=numerical.coefficient_error;
    targets[0]=std::min(targets[0],coefficient_limit*singular_lower/16);
    const long double design_budget=std::min(singular_lower/8,
        coefficient_limit*singular_lower/(32*(1+beta_norm+residual_norm/singular_lower)));
    for (int j:kept) targets[j+1]=std::min(targets[j+1],design_budget/std::sqrt(static_cast<long double>(kept.size())));
}

inline void check_ordinary_regression_projection(const AbsorptionResult& result,
    const Eigen::VectorXd& reference_y,const Eigen::MatrixXd& reference_X,
    const Eigen::VectorXd* weights,const std::vector<Eigen::VectorXi>* clusters,
    const HdfeOptions& options,int fe_rank) {
    const int p=static_cast<int>(reference_X.cols());
    if (!p) return;
    std::vector<int> columns(p),kept;std::iota(columns.begin(),columns.end(),0);
    const double rank_limit=64*std::numeric_limits<double>::epsilon();
    const auto reference_mask=precise_ols_rank_mask(reference_X,columns,weights,false,rank_limit,
        options.collinear_priority,std::max(1,options.num_threads),options.parallel_observer);
    const auto actual_mask=precise_ols_rank_mask(result.X_tilde,columns,weights,false,rank_limit,
        options.collinear_priority,std::max(1,options.num_threads),options.parallel_observer);
    if (reference_mask!=actual_mask)
        throw std::runtime_error("Ordinary FE numerical rank differs from the independent projection; no estimates returned");
    for (int j=0;j<p;++j) if (!reference_mask[j]) kept.push_back(j);
    if (kept.empty()) return;
    Eigen::MatrixXd actual_X(reference_X.rows(),kept.size()),selected_X(reference_X.rows(),kept.size());
    for (std::size_t j=0;j<kept.size();++j) {
        actual_X.col(j)=result.X_tilde.col(kept[j]);selected_X.col(j)=reference_X.col(kept[j]);
    }
    const auto actual=ordinary_projection_fit(result.y_tilde,actual_X,weights,clusters,options);
    const auto reference=ordinary_projection_fit(reference_y,selected_X,weights,clusters,options);
    if (!ieee_all_finite(actual.coefficients) || !ieee_all_finite(reference.coefficients))
        throw std::runtime_error("Ordinary FE coefficient reference is not finite; no estimates returned");
    const double coefficient_tolerance=std::max(effective_absorption_tolerance(options),rank_limit);
    const double covariance_tolerance=std::max(options.tol,rank_limit);
    long double actual_noise=0,reference_noise=0,response_norm=0,prediction_norm=0;
    for (int row=0;row<reference_y.size();++row) {
        const long double weight=weights ? (*weights)[row] : 1.0L,root=std::sqrt(weight);
        actual_noise=std::hypot(actual_noise,root*actual.residuals[row]);
        reference_noise=std::hypot(reference_noise,root*reference.residuals[row]);
        response_norm=std::hypot(response_norm,root*reference_y[row]);
        prediction_norm=std::hypot(prediction_norm,root*(static_cast<long double>(reference_y[row])-reference.residuals[row]));
    }
    const long double arithmetic_floor=64*std::numeric_limits<double>::epsilon()*(response_norm+prediction_norm);
    const bool numerical_perfect=actual_noise<=arithmetic_floor && reference_noise<=arithmetic_floor;
    double coefficient_error=0,covariance_error=0;
    const double effective=options.weights_are_frequencies && weights ? weights->sum() : reference_y.size();
    const bool inference=effective>fe_rank+static_cast<int>(kept.size());
    for (int j=0;j<reference.coefficients.size();++j) {
        coefficient_error=std::max(coefficient_error,std::abs(actual.coefficients[j]-reference.coefficients[j])/
            std::max(1.0,std::abs(reference.coefficients[j])));
        if (!inference) continue;
        for (int k=0;k<reference.coefficients.size();++k) {
            if (!ieee_finite(reference.covariance(j,k)) || !ieee_finite(actual.covariance(j,k)))
                throw std::runtime_error("Ordinary FE covariance reference is not finite; no estimates returned");
            long double scale=std::sqrt(std::abs(static_cast<long double>(reference.covariance(j,j))))*
                std::sqrt(std::abs(static_cast<long double>(reference.covariance(k,k))));
            if (numerical_perfect) {
                const long double bread_scale=std::sqrt(std::abs(static_cast<long double>(reference.xtx_inv(j,j))))*
                    std::sqrt(std::abs(static_cast<long double>(reference.xtx_inv(k,k))));
                const long double aggregation=options.se_type==StandardErrorType::Cluster ? effective : 1.0L;
                scale+=arithmetic_floor*arithmetic_floor*bread_scale*aggregation*
                    std::max(1.0,reference.cov_scale)/covariance_tolerance;
            }
            const long double error=std::abs(static_cast<long double>(actual.covariance(j,k))-reference.covariance(j,k));
            const double ratio=scale>0 ? static_cast<double>(error/scale) :
                error==0 ? 0 : std::numeric_limits<double>::infinity();
            covariance_error=std::max(covariance_error,ratio);
        }
    }
    if (!ieee_finite(coefficient_error) || !ieee_finite(covariance_error) ||
        coefficient_error>coefficient_tolerance || covariance_error>covariance_tolerance) {
        std::ostringstream message;
        message << "Ordinary FE regression precision failed the independent projection check: coefficient error="
                << std::scientific << coefficient_error << "; relative covariance error=" << covariance_error
                << "; coefficient tolerance=" << coefficient_tolerance
                << "; covariance tolerance=" << covariance_tolerance
                << ". Projection stopping alone does not establish inference accuracy. No estimates returned.";
        throw std::runtime_error(message.str());
    }
}

// Bounded independent check, not a replacement for the selected absorber.
// Compress parallel edges, choose a maximum-weight forest, and write the
// residual flow in its fundamental cycles. In chord-residual coordinates:
//   (I + C_tree' W_tree^-1 C_tree W_chord) r = C' mean(y).
// Every weight ratio in this system is <= 1. Weak links are never obtained
// by subtracting nearly equal diagonal entries of the normal equations.
enum class SmallClassicalReferenceStatus {Inconclusive,Agrees,Disagrees};

struct SmallClassicalReferenceResult {
    SmallClassicalReferenceStatus status=SmallClassicalReferenceStatus::Inconclusive;
    std::string reason;
    long double magnitude=0;
};

inline SmallClassicalReferenceResult small_verified_classical_reference(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const HdfeOptions& options,const AbsorptionResult& result) {
    if (options.se_type!=StandardErrorType::Homoskedastic || options.retain_fixed_effects) return {};
    using I=OlsProofInterval;using Real=long double;using Sum=ExactBinaryProducts<>;
    auto declined=[](const char* why,Real value=0) {
        return SmallClassicalReferenceResult{
            SmallClassicalReferenceStatus::Inconclusive,why,value};
    };
    auto disagrees=[](const char* why,Real value) {
        return SmallClassicalReferenceResult{
            SmallClassicalReferenceStatus::Disagrees,why,value};
    };
    const int n=static_cast<int>(y.size()),p=static_cast<int>(X.cols());
    if (n<=0 || p>32) return declined("size_scope");
    std::vector<std::map<int,int>> labels(fes.size());std::vector<int> offsets(fes.size());
    int columns=0;
    for (std::size_t d=0;d<fes.size();++d) {
        offsets[d]=columns;
        for (int row=0;row<n;++row) {
            const auto added=labels[d].emplace(fes[d][row],static_cast<int>(labels[d].size()));
            if (added.second && ++columns>64) return declined("fe_column_limit");
        }
    }
    if (static_cast<std::size_t>(n)*(columns+p)>100000) return declined("storage_limit");
    try {
        ExactRankBudget budget;PackedRankAccumulator fe_rank(columns,budget,200000);
        for (int row=0;row<n;++row) if (!weights || (*weights)[row]>0) {
            PackedRankAccumulator::Row equation;
            for (std::size_t d=0;d<fes.size();++d)
                equation.emplace_back(offsets[d]+labels[d].at(fes[d][row]),RankFraction(1));
            fe_rank.append(std::move(equation));
        }
        std::vector<int> all(p),kept;std::iota(all.begin(),all.end(),0);
        const auto mask=precise_ols_rank_mask(result.X_tilde,all,weights,false,
            64*std::numeric_limits<double>::epsilon(),options.collinear_priority,
            std::max(1,options.num_threads),options.parallel_observer);
        for (int j=0;j<p;++j) if (!mask[j]) kept.push_back(j);
        if (kept.empty()) return declined("empty_slope_space");
        const int slopes=static_cast<int>(kept.size()),full=slopes+fe_rank.rank();
        if (full>64) return declined("rank_limit");
        std::vector<int> position(columns,-1);int next=slopes;
        for (int j=0;j<columns;++j) if (fe_rank.is_pivot(j)) position[j]=next++;
        Eigen::MatrixXd design=Eigen::MatrixXd::Zero(n,full),within(n,slopes);
        for (int j=0;j<slopes;++j) {design.col(j)=X.col(kept[j]);within.col(j)=result.X_tilde.col(kept[j]);}
        for (int row=0;row<n;++row) for (std::size_t d=0;d<fes.size();++d) {
            const int column=position[offsets[d]+labels[d].at(fes[d][row])];
            if (column>=0) design(row,column)=1;
        }
        std::vector<int> full_columns(full);std::iota(full_columns.begin(),full_columns.end(),0);
        Eigen::MatrixXd basis;
        const auto full_mask=precise_ols_rank_mask(design,full_columns,weights,false,
            64*std::numeric_limits<double>::epsilon(),{},std::max(1,options.num_threads),
            options.parallel_observer,nullptr,&basis);
        if (std::any_of(full_mask.begin(),full_mask.end(),[](std::uint8_t value){return value!=0;})) return declined("rank");
        const auto reference=precise_classical_reference(y,design,weights,options.weights_are_frequencies,options.parallel_observer);
        const auto proof=certify_ols_numerics(y,design,weights,reference.coefficients,basis,
            std::max(1,options.num_threads),options.parallel_observer,true);
        const Real tolerance=std::max(effective_absorption_tolerance(options),64*std::numeric_limits<double>::epsilon());
        if (proof.coefficient_error>tolerance/4) return declined("reference_coefficient_bound",proof.coefficient_error);
        const auto actual=ordinary_projection_fit(result.y_tilde,within,weights,nullptr,options);
        for (int j=0;j<slopes;++j) {
            const Real center=reference.coefficients[j];
            const Real separation=std::max(0.0L,
                std::abs(Real(actual.coefficients[j])-center)-proof.coefficient_error);
            const Real scale=std::max(1.0L,std::abs(center)+proof.coefficient_error);
            if (separation>tolerance*scale)
                return disagrees("coefficient_separation",separation/scale);
        }
        const double effective=options.weights_are_frequencies && weights ? weights->sum() : n;
        if (effective>full) {
            const I reduction=I::point(proof.normalized_matrix_norm)*I::point(proof.normalized_error)*I::point(proof.normalized_error);
            const I rss{std::max(0.0L,I::down(proof.residual_squared.lo-reduction.hi)),proof.residual_squared.hi};
            const Real denominator=effective-slopes;
            const I inverse_df{I::down(1/denominator),I::up(1/denominator)};
            std::vector<I> covariance(static_cast<std::size_t>(slopes)*slopes);
            for (int j=0;j<slopes;++j) for (int k=0;k<slopes;++k)
                covariance[j*slopes+k]=rss*inverse_df*proof.bread[j*full+k];
            for (int j=0;j<slopes;++j) for (int k=0;k<slopes;++k) {
                const auto value=covariance[j*slopes+k];
                if (!(covariance[j*slopes+j].lo>0) || !(covariance[k*slopes+k].lo>0)) return declined("zero_variance");
                const Real actual_value=actual.covariance(j,k);
                const Real separation=actual_value<value.lo ? value.lo-actual_value :
                    actual_value>value.hi ? actual_value-value.hi : 0.0L;
                const Real scale=I::up(std::sqrt(std::max(0.0L,covariance[j*slopes+j].hi))*
                    std::sqrt(std::max(0.0L,covariance[k*slopes+k].hi)));
                if (!ieee_finite(static_cast<double>(separation)) ||
                    !ieee_finite(static_cast<double>(scale))) return declined("covariance_range");
                if (separation>std::max(options.tol,64*std::numeric_limits<double>::epsilon())*scale)
                    return disagrees("covariance_separation",scale>0 ? separation/scale : separation);
            }
        }
        const Real residual_tolerance=options.tolerance_mode==ToleranceMode::StrictResidual ? options.tol :
            options.tol*std::max(1.0,reference.residuals.cwiseAbs().maxCoeff());
        for (int row=0;row<n;++row) {
            Sum affine;affine.add_product(std::array<double,1>{y[row]});
            affine.add_product(std::array<double,1>{reference.residuals[row]},true);
            Real row_squared=0;
            for (int j=0;j<full;++j) {
                affine.add_product(std::array<double,2>{design(row,j),reference.coefficients[j]},true);
                Sum coordinate;
                for (int k=0;k<full;++k) if (basis(k,j)!=0)
                    coordinate.add_product(std::array<double,2>{design(row,k),basis(k,j)});
                const auto interval=coordinate.interval(-proof.basis_exponents[j]);
                const Real upper=std::max(std::abs(interval.first),std::abs(interval.second));
                row_squared=I::up(row_squared+I::up(upper*upper));
            }
            const I reference_error=I::point(I::up(std::sqrt(row_squared)))*I::point(proof.normalized_error)+I::point(affine.upper_absolute());
            const Real separation=std::max(0.0L,
                std::abs(Real(actual.residuals[row])-reference.residuals[row])-reference_error.hi);
            if (!ieee_finite(static_cast<double>(separation))) return declined("residual_range");
            if (separation>residual_tolerance) return disagrees("residual_separation",separation);
        }
        return {SmallClassicalReferenceStatus::Agrees,"",0};
    } catch (const std::runtime_error& error) {return declined(error.what());}
}

inline bool ordinary_fe_nested(const Eigen::VectorXi& inner,const Eigen::VectorXi& outer) {
    std::unordered_map<int,int> categories;
    for (int row=0;row<inner.size();++row) {
        auto added=categories.emplace(outer[row],inner[row]);
        if (!added.second && added.first->second!=inner[row]) return false;
    }
    return true;
}

inline void check_ordinary_cycle_reference(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes, const Eigen::VectorXd* weights,
    const HdfeOptions& options, const AbsorptionResult& result,
    const std::vector<Eigen::VectorXi>* clusters = nullptr,bool verify_regression = false,
    OrdinaryThreeFeSupportWorkspace* support_workspace = nullptr) {
    const int n=static_cast<int>(y.size()), rhs=static_cast<int>(X.cols())+1;
    auto general_reference=[&]() {
        if (fes.size()==2 && exact_ordinary_two_fe_projection(y,X,fes,weights,result)) return;
        // ||W e|| <= sqrt(max w) ||sqrt(W) e||. Tighten the projection
        // bounds to control each covariance-score column as well. The
        // existing raw-input roundoff floor must not override this target.
        const long double largest_weight=weights ? weights->maxCoeff() : 1.0L;
        const long double limit=std::max(effective_absorption_tolerance(options),
            64*std::numeric_limits<double>::epsilon());
        std::vector<long double> targets(rhs,0.0L);
        for (int column=0;column<rhs;++column) {
            long double score=0.0L;
            for (int row=0;row<n;++row) {
                const long double value=column ? result.X_tilde(row,column-1) : result.y_tilde[row];
                const long double weight=(weights ? static_cast<long double>((*weights)[row]) : 1.0L)/largest_weight;
                volatile long double term=(options.weights_are_frequencies ? std::sqrt(weight) : weight)*value;
                if (term==0 && value!=0 && weight!=0)
                    throw std::runtime_error("Ordinary FE covariance-score target lost numerical range; no estimates returned");
                score=std::hypot(score,static_cast<long double>(term));
            }
            targets[column]=0.25L*limit*std::sqrt(largest_weight)*score;
        }
        if (verify_regression) tighten_ordinary_residual_target(result,weights,options,targets);
        const auto tree_check=certify_ordinary_tree_space(y,X,fes,weights,result,targets);
        if (tree_check.eligible && tree_check.passed) return;
        const auto support_check=certify_ordinary_three_fe_support(
            y,X,fes,weights,result,targets,support_workspace,
            options.num_threads,options.parallel_observer);
        if (support_check.eligible && support_check.passed) return;
        // A distinct positive-weight primary pair needs at least one distinct
        // full pattern. This proves the generic verifier cannot fit its budget
        // before allocating its artificial group structure or pattern hashes.
        if (support_check.positive_primary_pairs>kGroupForwardCellLimit/(fes.size()+2) ||
            support_check.positive_primary_pairs>kGroupForwardCellLimit/static_cast<std::size_t>(rhs)) {
            std::ostringstream message;
            message << "Ordinary FE forward precision could not be established: ";
            if (support_check.eligible)
                message << "sparse support ratio=" << std::scientific << support_check.worst_ratio;
            else message << support_check.unavailable;
            message << "; generic verification exceeds its proven pattern-storage limit. No estimates returned.";
            throw std::runtime_error(message.str());
        }
        // A single redundant individual adds a constant already in the span
        // of these categorical FEs. This is the same projection space, with
        // the existing implicit-basis forward check and its fail-closed limits.
        GroupIndividualStructure constant;
        constant.num_groups=n;constant.num_individuals=1;
        constant.group_ptr.resize(static_cast<std::size_t>(n)+1);
        std::iota(constant.group_ptr.begin(),constant.group_ptr.end(),0);
        constant.group_individual.assign(n,0);constant.group_scale.assign(n,1.0);
        constant.individual_ptr={0,n};constant.individual_group.resize(n);
        std::iota(constant.individual_group.begin(),constant.individual_group.end(),0);
        ScopedGroupForwardWorkspace workspace(false);
        const auto checked=certify_group_forward_space(y,X,fes,constant,weights,options,result,nullptr,&targets);
        if (!checked.eligible || !checked.passed) {
            if (verify_regression) {
                const auto classical=small_verified_classical_reference(
                    y,X,fes,weights,options,result);
                if (classical.status==SmallClassicalReferenceStatus::Agrees) return;
                if (classical.status==SmallClassicalReferenceStatus::Disagrees) {
                    std::ostringstream message;
                    message << "independent bounded classical reference disagrees: "
                            << classical.reason << '=' << std::scientific
                            << classical.magnitude;
                    throw OrdinaryReferenceDisagreement(message.str());
                }
            }
            std::ostringstream message;
            message << "Ordinary FE forward precision could not be established on the supplied FE space: ratio="
                    << std::scientific << checked.worst_ratio << "; tolerance=" << checked.limit
                    << "; " << checked.unavailable << ". No estimates returned.";
            throw std::runtime_error(message.str());
        }
    };
    if (fes.size()>2) {
        // The independent check depends on the FE space, not the number of
        // supplied labels. Constants, duplicate partitions and nested FEs
        // must not bypass a two-FE check. The absorber and reported model
        // retain their original FE list and normalization.
        std::vector<std::size_t> kept;
        for (std::size_t candidate=0;candidate<fes.size();++candidate) {
            bool redundant=false;
            for (std::size_t k=0;k<kept.size();) {
                if (ordinary_fe_nested(fes[candidate],fes[kept[k]])) {redundant=true;break;}
                if (ordinary_fe_nested(fes[kept[k]],fes[candidate])) kept.erase(kept.begin()+k);
                else ++k;
            }
            if (!redundant) kept.push_back(candidate);
        }
        if (!kept.empty() && kept.size()<=2) {
            std::vector<Eigen::VectorXi> effective={fes[kept[0]],fes[kept.size()==1 ? kept[0] : kept[1]]};
            check_ordinary_cycle_reference(y,X,effective,weights,options,result,clusters,verify_regression,support_workspace);
        }
        else if (!kept.empty()) general_reference();
        return;
    }
    if (fes.size()!=2 || (!weights && !verify_regression)) return;
    if (!n || (!verify_regression && weights && weights->minCoeff()==weights->maxCoeff())) return;
    constexpr int pattern_limit=4096, solve_limit=512;
    using Real=long double;
    using Matrix=Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>;
    std::map<std::pair<int,int>,int> pair_ids;
    std::vector<std::vector<int>> members;
    std::vector<std::pair<int,int>> pairs;
    std::map<int,int> left_ids,right_ids;
    for (int i=0;i<n;++i) {
        const auto pair=std::make_pair(fes[0][i],fes[1][i]);
        auto inserted=pair_ids.emplace(pair,static_cast<int>(pairs.size()));
        if (inserted.second) {
            if (pairs.size()==pattern_limit) {general_reference();return;}
            pairs.push_back(pair); members.emplace_back();
            left_ids.emplace(pair.first,static_cast<int>(left_ids.size()));
            right_ids.emplace(pair.second,static_cast<int>(right_ids.size()));
        }
        members[inserted.first->second].push_back(i);
    }
    const int edges=static_cast<int>(pairs.size());
    if (static_cast<std::size_t>(edges)*rhs>8000000U) {general_reference();return;}
    const int nl=static_cast<int>(left_ids.size());
    const int vertices=nl+static_cast<int>(right_ids.size());
    std::vector<std::pair<int,int>> endpoints;
    for (const auto& pair:pairs)
        endpoints.emplace_back(left_ids.at(pair.first),nl+right_ids.at(pair.second));
    const Real max_weight=weights ? weights->maxCoeff() : 1.0L;
    auto weight_at=[&](int row) -> Real {return weights ? (*weights)[row] : 1.0L;};
    std::vector<Real> mass(edges);
    Matrix anchors(edges,rhs),offsets(edges,rhs);
    auto raw=[&](int row,int column) -> Real {return column ? X(row,column-1) : y[row];};
    auto require_finite=[](Real value) {
        if (!ieee_finite(static_cast<double>(value)))
            throw std::runtime_error("Ordinary FE cycle precision reference exceeded its numerical range; no estimates returned");
    };
    for (int e=0;e<edges;++e) {
        GroupForwardSum total;
        for (int i:members[e]) {
            volatile Real scaled=weight_at(i)/max_weight;
            if (!(scaled>0))
                throw std::runtime_error("Ordinary FE cycle weight scaling lost numerical range; no estimates returned");
            total.add(scaled);
        }
        mass[e]=total.value();
        require_finite(mass[e]);
        for (int j=0;j<rhs;++j) {
            anchors(e,j)=raw(members[e][0],j);
            GroupForwardSum sum;
            for (int i:members[e]) {
                volatile Real difference=raw(i,j)-anchors(e,j);
                sum.add((weight_at(i)/max_weight)*difference);
            }
            offsets(e,j)=sum.value()/mass[e];
            require_finite(offsets(e,j));
        }
    }
    std::vector<int> parent(vertices),order(edges),tree,chords;
    std::iota(parent.begin(),parent.end(),0); std::iota(order.begin(),order.end(),0);
    auto root=[&](int node) {
        while (parent[node]!=node) {parent[node]=parent[parent[node]];node=parent[node];}
        return node;
    };
    std::stable_sort(order.begin(),order.end(),[&](int a,int b) {return mass[a]>mass[b];});
    std::vector<std::vector<std::pair<int,int>>> adjacent(vertices);
    for (int e:order) {
        const auto [a,b]=endpoints[e]; const int ra=root(a),rb=root(b);
        if (ra==rb) {
            chords.push_back(e);
        } else {
            parent[ra]=rb;tree.push_back(e);
            adjacent[a].emplace_back(b,e);adjacent[b].emplace_back(a,e);
        }
    }
    const int cycles=static_cast<int>(chords.size());
    const int tree_rank=static_cast<int>(tree.size());
    if (std::min(tree_rank,cycles)>solve_limit || static_cast<std::size_t>(edges)*cycles>8000000U) {
        general_reference();return;
    }
    Eigen::MatrixXi C=Eigen::MatrixXi::Zero(edges,cycles);
    for (int c=0;c<cycles;++c) {
        const int chord=chords[c]; const auto [left,right]=endpoints[chord];
        std::vector<int> before(vertices,-1),via(vertices,-1),queue{right};
        before[right]=right;
        for (std::size_t k=0;k<queue.size() && before[left]<0;++k) {
            const int node=queue[k];
            for (const auto& [next,e]:adjacent[node]) if (before[next]<0) {
                before[next]=node;via[next]=e;queue.push_back(next);
            }
        }
        C(chord,c)=1;
        for (int node=left;node!=right;node=before[node]) {
            const int e=via[node];
            C(e,c)=endpoints[e]==std::make_pair(before[node],node) ? 1 : -1;
        }
    }
    Matrix means=Matrix::Zero(edges,rhs);
    if (cycles>tree_rank && tree_rank>0) {
        // Eliminate the chord residuals instead when the tree system is
        // smaller: (I + Wt^-1 Ct Wc Ct') rt = Wt^-1 Ct Wc C' mean(y).
        // rt is obtained directly; there is no subtraction of large flows
        // when recovering the residuals on strong tree edges.
        Matrix contrasts(cycles,rhs),H=Matrix::Identity(tree_rank,tree_rank),b(tree_rank,rhs);
        for (int c=0;c<cycles;++c) for (int j=0;j<rhs;++j) {
            GroupForwardSum sum;
            for (int e=0;e<edges;++e) if (C(e,c)) sum.add(C(e,c)*anchors(e,j));
            for (int e=0;e<edges;++e) if (C(e,c)) sum.add(C(e,c)*offsets(e,j));
            contrasts(c,j)=sum.value();require_finite(contrasts(c,j));
        }
        for (int i=0;i<tree_rank;++i) {
            const int edge=tree[i];
            for (int j=0;j<tree_rank;++j) {
                GroupForwardSum sum;sum.add(i==j ? 1 : 0);
                for (int c=0;c<cycles;++c) if (C(edge,c) && C(tree[j],c))
                    sum.add(C(edge,c)*C(tree[j],c)*(mass[chords[c]]/mass[edge]));
                H(i,j)=sum.value();require_finite(H(i,j));
            }
            for (int j=0;j<rhs;++j) {
                GroupForwardSum sum;
                for (int c=0;c<cycles;++c) if (C(edge,c))
                    sum.add(C(edge,c)*(mass[chords[c]]/mass[edge])*contrasts(c,j));
                b(i,j)=sum.value();require_finite(b(i,j));
            }
        }
        Eigen::PartialPivLU<Matrix> factor(H);
        Matrix solution=factor.solve(b);
        for (int pass=0;pass<2;++pass) {
            Matrix residual(tree_rank,rhs);
            for (int i=0;i<tree_rank;++i) for (int j=0;j<rhs;++j) {
                GroupForwardSum sum;sum.add(b(i,j));
                for (int k=0;k<tree_rank;++k) sum.add(-H(i,k)*solution(k,j));
                residual(i,j)=sum.value();
            }
            solution+=factor.solve(residual);
        }
        const Matrix inverse=factor.solve(Matrix::Identity(tree_rank,tree_rank));
        Real defect=0;
        for (int i=0;i<tree_rank;++i) {
            GroupForwardSum total;
            for (int j=0;j<tree_rank;++j) {
                GroupForwardSum sum;sum.add(i==j ? 1 : 0);
                for (int k=0;k<tree_rank;++k) sum.add(-inverse(i,k)*H(k,j));
                total.add(std::abs(sum.value()));
            }
            defect=std::max(defect,total.value());
        }
        if (!ieee_finite(static_cast<double>(defect)) || defect>=0.125L)
            throw std::runtime_error("Ordinary FE tree precision reference was numerically unresolved; no estimates returned");
        for (int i=0;i<solution.size();++i) require_finite(solution.data()[i]);
        for (int i=0;i<tree_rank;++i) means.row(tree[i])=solution.row(i);
        for (int c=0;c<cycles;++c) for (int j=0;j<rhs;++j) {
            GroupForwardSum sum;sum.add(contrasts(c,j));
            for (int i=0;i<tree_rank;++i) if (C(tree[i],c)) sum.add(-C(tree[i],c)*solution(i,j));
            means(chords[c],j)=sum.value();require_finite(means(chords[c],j));
        }
    }
    else if (cycles) {
        Matrix H=Matrix::Identity(cycles,cycles),b(cycles,rhs);
        for (int c=0;c<cycles;++c) {
            for (int d=0;d<cycles;++d) {
                GroupForwardSum sum;sum.add(c==d ? 1 : 0);
                for (int e:tree) if (C(e,c) && C(e,d))
                    sum.add(C(e,c)*C(e,d)*(mass[chords[d]]/mass[e]));
                H(c,d)=sum.value();
                require_finite(H(c,d));
            }
            for (int j=0;j<rhs;++j) {
                GroupForwardSum sum;
                for (int e=0;e<edges;++e) if (C(e,c)) sum.add(C(e,c)*anchors(e,j));
                for (int e=0;e<edges;++e) if (C(e,c)) sum.add(C(e,c)*offsets(e,j));
                b(c,j)=sum.value();
                require_finite(b(c,j));
            }
        }
        Eigen::PartialPivLU<Matrix> factor(H);
        Matrix solution=factor.solve(b);
        // Refine in the independently formed cycle system, preserving the
        // small chord signal through the final conversion to observation rows.
        for (int pass=0;pass<2;++pass) {
            Matrix residual(cycles,rhs);
            for (int c=0;c<cycles;++c) for (int j=0;j<rhs;++j) {
                GroupForwardSum sum;sum.add(b(c,j));
                for (int d=0;d<cycles;++d) sum.add(-H(c,d)*solution(d,j));
                residual(c,j)=sum.value();
            }
            solution+=factor.solve(residual);
        }
        const Matrix inverse=factor.solve(Matrix::Identity(cycles,cycles));
        for (int i=0;i<solution.size();++i) require_finite(solution.data()[i]);
        for (int i=0;i<inverse.size();++i) require_finite(inverse.data()[i]);
        Real defect=0;
        for (int c=0;c<cycles;++c) {
            GroupForwardSum row;
            for (int d=0;d<cycles;++d) {
                GroupForwardSum sum;sum.add(c==d ? 1 : 0);
                for (int k=0;k<cycles;++k) sum.add(-inverse(c,k)*H(k,d));
                row.add(std::abs(sum.value()));
            }
            defect=std::max(defect,row.value());
        }
        if (!ieee_finite(static_cast<double>(defect)) || defect>=0.125L)
            throw std::runtime_error("Ordinary FE cycle precision reference was numerically unresolved; no estimates returned");
        for (int c=0;c<cycles;++c) means.row(chords[c])=solution.row(c);
        for (int e:tree) for (int j=0;j<rhs;++j) {
            GroupForwardSum sum;
            for (int c=0;c<cycles;++c) if (C(e,c))
                sum.add(C(e,c)*(mass[chords[c]]/mass[e])*solution(c,j));
            means(e,j)=sum.value();
            require_finite(means(e,j));
        }
    }
    Real worst=0;
    Eigen::VectorXd reference_y;
    Eigen::MatrixXd reference_X;
    if (verify_regression && X.cols()>0) {reference_y.resize(n);reference_X.resize(n,X.cols());}
    for (int j=0;j<rhs;++j) {
        Real error_w=0,reference_w=0,error_w2=0,reference_w2=0;
        Real max_error=0,max_reference=0;
        auto accumulate_norm=[&](Real& norm,Real scale,Real value) {
            volatile Real term=scale*value;
            if (term==0 && scale!=0 && value!=0)
                throw std::runtime_error("Ordinary FE cycle score scaling lost numerical range; no estimates returned");
            norm=std::hypot(norm,static_cast<Real>(term));
            require_finite(norm);
        };
        for (int e=0;e<edges;++e) for (int i:members[e]) {
            GroupForwardSum sum;
            volatile Real centered=raw(i,j)-anchors(e,j);
            sum.add(centered);sum.add(-offsets(e,j));sum.add(means(e,j));
            const Real reference=sum.value();
            require_finite(reference);
            if (reference_y.size()) {
                if (j) reference_X(i,j-1)=static_cast<double>(reference);
                else reference_y[i]=static_cast<double>(reference);
            }
            const Real actual=j ? result.X_tilde(i,j-1) : result.y_tilde[i];
            volatile Real difference=actual-reference;
            const Real w=weight_at(i)/max_weight;
            accumulate_norm(error_w,std::sqrt(w),difference);
            accumulate_norm(reference_w,std::sqrt(w),reference);
            const Real score_weight=options.weights_are_frequencies ? std::sqrt(w) : w;
            accumulate_norm(error_w2,score_weight,difference);accumulate_norm(reference_w2,score_weight,reference);
            max_error=std::max(max_error,std::abs(static_cast<Real>(difference)));
            max_reference=std::max(max_reference,std::abs(reference));
        }
        auto ratio=[](Real error,Real reference) {
            return reference>0 ? error/reference : error==0 ? Real(0) : std::numeric_limits<Real>::infinity();
        };
        worst=std::max({worst,ratio(max_error,max_reference),
            ratio(error_w,reference_w),ratio(error_w2,reference_w2)});
    }
    const double limit=std::max(effective_absorption_tolerance(options),64*std::numeric_limits<double>::epsilon());
    if (!ieee_finite(static_cast<double>(worst)) || worst>limit) {
        std::ostringstream message;
        message << "Ordinary FE cycle projection precision failed: relative discrepancy="
                << std::scientific << static_cast<double>(worst) << "; limit=" << limit
                << "; patterns=" << edges << "; cycles=" << cycles
                << ". The weighted projection or covariance score scale is unresolved; no estimates returned.";
        throw std::runtime_error(message.str());
    }
    if (reference_y.size())
        check_ordinary_regression_projection(result,reference_y,reference_X,weights,clusters,options,
            static_cast<int>(tree.size()));
}

}}
#endif
