// Optional full-model projection inference. No ordinary estimation path
// calls this module. See docs/akm-projection-inference.md for the theorem.
#include "hdfe/akm_projection.hpp"
#include "hdfe/akm_projection_critical.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hdfe { namespace akm { namespace projection {
namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

bool finite_matrix(const MatrixXd& matrix) {
    for(Eigen::Index i=0;i<matrix.size();++i)
        if(!hdfe::detail::ieee_finite(matrix.data()[i])) return false;
    return true;
}

struct Minimum { double lower=0., upper=0., gap=0.; };

// Minimise z'H z+2g'z over ||z||<=radius. The reported lower endpoint is
// the Lagrange-dual bound, with a rounding allowance, not an uncertified
// optimiser iterate. The primal point also checks near-hard cases.
Minimum quadratic_minimum(const MatrixXd& input, const VectorXd& linear,
                          double radius) {
    const int size=static_cast<int>(linear.size());
    if(size==0) return {};
    const MatrixXd matrix=(.5*(input+input.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(matrix);
    if(eig.info()!=Eigen::Success || !finite_matrix(eig.eigenvalues()))
        throw std::runtime_error("worker projection: confidence-region eigensolve failed");
    VectorXd values=eig.eigenvalues(), g=eig.eigenvectors().transpose()*linear;
    const double scale=std::max(values.cwiseAbs().maxCoeff(),g.stableNorm()/radius);
    if(scale==0.) return {};
    if(!hdfe::detail::ieee_finite(scale))
        throw std::runtime_error("worker projection: confidence-region scale is not finite");
    values/=scale; g/=scale;
    const double shift=std::max(0.,-values[0]);
    const VectorXd denominator=values.array()+shift;
    VectorXd point=VectorXd::Zero(size);
    bool pole_linear=false;
    for(int j=0;j<size;++j) {
        if(denominator[j]>0.) point[j]=-g[j]/denominator[j];
        else if(g[j]!=0.) pole_linear=true;
    }
    double extra=0.;
    if(!pole_linear && point.norm()<=radius) {
        if(shift>0.) point[0]=std::sqrt(std::max(0.,radius*radius-point.squaredNorm()));
    } else {
        double lower=0., upper=g.norm()/radius;
        if(!(upper>0.)) throw std::runtime_error("worker projection: invalid secular bracket");
        for(int iteration=0;iteration<256;++iteration) {
            const double middle=lower+(upper-lower)*.5;
            if(middle==lower || middle==upper) break;
            long double squared=0.;
            for(int j=0;j<size;++j) {
                const long double value=static_cast<long double>(g[j])/(denominator[j]+middle);
                squared+=value*value;
            }
            if(squared>static_cast<long double>(radius)*radius) lower=middle;
            else upper=middle;
            if(upper-lower<=8.*std::numeric_limits<double>::epsilon()*upper) break;
        }
        extra=upper;
        for(int j=0;j<size;++j) point[j]=-g[j]/(denominator[j]+extra);
        // At a near-hard pole a numerically tiny linear component can leave
        // the iterate inside the sphere. A feasible boundary completion
        // supplies a primal certificate; the dual bound remains unchanged.
        if(shift>0. && point.squaredNorm()<radius*radius) {
            VectorXd boundary=point;
            const double rest=point.tail(size-1).squaredNorm();
            boundary[0]=(g[0]>0. ? -1.:1.)*std::sqrt(std::max(0.,radius*radius-rest));
            const double first=(values.array()*point.array().square()+2.*g.array()*point.array()).sum();
            const double second=(values.array()*boundary.array().square()+2.*g.array()*boundary.array()).sum();
            if(second<first) point=std::move(boundary);
        }
    }
    long double dual=-static_cast<long double>(shift+extra)*radius*radius;
    for(int j=0;j<size;++j) {
        const double denom=denominator[j]+extra;
        if(denom>0.) dual-=static_cast<long double>(g[j])*g[j]/denom;
        else if(g[j]!=0.) throw std::runtime_error("worker projection: singular secular system");
    }
    const double primal=(values.array()*point.array().square()+2.*g.array()*point.array()).sum();
    const double operation_count=16.*size*size+64.*size+32.;
    const double roundoff=std::numeric_limits<double>::epsilon()/2.;
    const double gamma=operation_count*roundoff/(1.-operation_count*roundoff);
    const double magnitude=radius*radius*values.cwiseAbs().sum()+2.*radius*g.cwiseAbs().sum();
    const double allowance=gamma*magnitude;
    const double residual=(matrix*eig.eigenvectors()-eig.eigenvectors()*eig.eigenvalues().asDiagonal()).stableNorm();
    const double orthogonality=(eig.eigenvectors().transpose()*eig.eigenvectors()-MatrixXd::Identity(size,size)).norm();
    const double eig_allowance=residual*radius*radius+orthogonality*scale*magnitude;
    const double gap=(primal-static_cast<double>(dual))*scale;
    if(!hdfe::detail::ieee_finite(gap) || gap < -16.*(allowance*scale+eig_allowance))
        throw std::runtime_error("worker projection: inconsistent confidence-region primal/dual bounds");
    Minimum result;
    result.lower=static_cast<double>(dual)*scale-allowance*scale-eig_allowance;
    result.upper=primal*scale+allowance*scale+eig_allowance;
    result.gap=std::max(0.,gap);
    return result;
}

struct Range { double lower,upper,gap; };
Range quadratic_range(const VectorXd& centre,const MatrixXd& root,
                      const MatrixXd& form,double critical) {
    const MatrixXd matrix=root.transpose()*form*root;
    const VectorXd linear=root.transpose()*form*centre;
    const double constant=centre.dot(form*centre);
    const auto low=quadratic_minimum(matrix,linear,std::sqrt(critical));
    const auto high=quadratic_minimum(-matrix,-linear,std::sqrt(critical));
    const double allowance=(64.*centre.size()+16.)*std::numeric_limits<double>::epsilon()*
        centre.cwiseAbs().dot(form.cwiseAbs()*centre.cwiseAbs());
    return {constant+low.lower-allowance,constant-high.lower+allowance,std::max(low.gap,high.gap)};
}
} // namespace

AkmProjectionResult worker_projection_inference(
    const VectorXd& y,const std::vector<int>& rows,const VectorXd& adjusted,
    const MatrixXd* controls,const std::vector<int>& control_columns,
    const MatrixXd& control_bread,const std::vector<int>& worker,
    const std::vector<int>& firm,const VectorXd& alpha,const VectorXd& psi,int team) {
    const int n=static_cast<int>(rows.size()),nw=static_cast<int>(alpha.size());
    const int nf=static_cast<int>(psi.size()),jf=nf-1;
    const int pc=static_cast<int>(control_columns.size()),q=jf+pc,dim=2*jf;
    if(n<3 || nw<2 || nf<2 || nf>128)
        throw std::runtime_error("worker projection requires at least two workers and 2 to 128 firms");
    if(adjusted.size()!=n || worker.size()!=rows.size() || firm.size()!=rows.size() ||
       (pc>0 && controls==nullptr) || team<1)
        throw std::runtime_error("internal worker projection input mismatch");
    AkmProjectionResult result;
    result.clusters=nw; result.dimension=dim; result.control_rank=pc;
    result.critical_value=kProjectionChiSquare95[dim];
    result.projection_factor=std::sqrt(result.critical_value)/1.959963984540054;
    VectorXd counts=VectorXd::Zero(nw),fc=VectorXd::Zero(nf),yrel(n);
    std::vector<int> ptr(nw+1,0),order(n);
    double response_max=0.;
    for(int i=0;i<n;++i) {
        counts[worker[i]]+=1.; fc[firm[i]]+=1.; ++ptr[worker[i]+1];
        yrel[i]=y[rows[i]]-y[rows[0]];
        response_max=std::max(response_max,std::abs(yrel[i]));
    }
    if(!hdfe::detail::ieee_finite(response_max))
        throw std::runtime_error("worker projection: outcome range is not finite");
    const int response_exponent=response_max>0. ? std::ilogb(response_max):0;
    for(int w=0;w<nw;++w) {
        if(counts[w]<=0. || counts[w]>=n)
            throw std::runtime_error("worker projection requires nonempty independent clusters");
        ptr[w+1]+=ptr[w];
        result.max_cluster_rows=std::max(result.max_cluster_rows,static_cast<int>(counts[w]));
    }
    std::vector<int> position=ptr;
    for(int i=0;i<n;++i) order[position[worker[i]]++]=i;
    MatrixXd means=MatrixXd::Zero(nw,q),cross=MatrixXd::Zero(jf,pc);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> support_control(nw,pc);
    support_control.setZero();
    for(int i=0;i<n;++i) if(firm[i]<jf) means(worker[i],firm[i])+=1.;
    for(int j=0;j<jf;++j) means.col(j).array()/=counts.array();
    hdfe::detail::ParallelWorkObserver observer;
    observer.reset(); observer.begin_region(team);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
    for(int c=0;c<pc;++c) {
        observer.observe_work();
        for(int i=0;i<n;++i) {
            const double value=(*controls)(rows[i],control_columns[c]);
            means(worker[i],jf+c)+=value;
            if(value!=0.) support_control(worker[i],c)=1;
            if(firm[i]<jf) cross(firm[i],c)+=value;
        }
        means.col(jf+c).array()/=counts.array();
    }
    observer.end_region();
    const MatrixXd weighted=means.array().colwise()*counts.array();
    MatrixXd gf=fc.head(jf).asDiagonal().toDenseMatrix();
    gf.noalias()-=means.leftCols(jf).transpose()*weighted.leftCols(jf);
    Eigen::LLT<MatrixXd> firm_factor(gf);
    if(firm_factor.info()!=Eigen::Success)
        throw std::runtime_error("worker projection: firm contrasts are not identified");
    const MatrixXd invf=firm_factor.solve(MatrixXd::Identity(jf,jf));
    cross.noalias()-=means.leftCols(jf).transpose()*weighted.rightCols(pc);
    MatrixXd joint=MatrixXd::Zero(q,q);
    joint.topLeftCorner(jf,jf)=invf;
    if(pc) {
        const MatrixXd a=invf*cross;
        MatrixXd bread=control_bread;
        if(bread.rows()!=pc || bread.cols()!=pc) {
            // Only a degenerate/subnormal homoskedastic scale needs this
            // fallback. Form residualised controls before their Gram to
            // avoid subtracting large nearly equal cross-products.
            MatrixXd within(n,pc);
            observer.begin_region(team);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
            for(int c=0;c<pc;++c) {
                observer.observe_work();
                const VectorXd worker_part=means.leftCols(jf)*a.col(c);
                for(int i=0;i<n;++i)
                    within(i,c)=(*controls)(rows[i],control_columns[c])-means(worker[i],jf+c)
                        -(firm[i]<jf ? a(firm[i],c):0.)+worker_part[worker[i]];
            }
            observer.end_region();
            const MatrixXd gram=within.transpose()*within;
            Eigen::LLT<MatrixXd> factor(gram);
            if(factor.info()!=Eigen::Success)
                throw std::runtime_error("worker projection: retained control contrasts are not identified");
            bread=factor.solve(MatrixXd::Identity(pc,pc));
        } else result.control_bread_reused=true;
        const MatrixXd ac=a*bread;
        joint.topLeftCorner(jf,jf).noalias()+=ac*a.transpose();
        joint.topRightCorner(jf,pc)=-ac;
        joint.bottomLeftCorner(pc,jf)=-ac.transpose();
        joint.bottomRightCorner(pc,pc)=bread;
    }
    joint=(.5*(joint+joint.transpose())).eval();
    if(!finite_matrix(joint)) throw std::runtime_error("worker projection: non-finite full-model bread");
    VectorXd rootp=(fc.array()/n).sqrt(),householder=rootp;
    householder[nf-1]+=1.;
    const MatrixXd basis=MatrixXd::Identity(nf,nf)-2.*householder*householder.transpose()/householder.squaredNorm();
    const MatrixXd contrast=basis.leftCols(jf).transpose()*rootp.asDiagonal();
    MatrixXd lalpha=MatrixXd::Zero(dim,nw),lz=MatrixXd::Zero(dim,q);
    for(int i=0;i<n;++i) lalpha.col(worker[i]).tail(jf)+=contrast.col(firm[i])/fc[firm[i]];
    lz.topLeftCorner(jf,jf)=contrast.leftCols(jf);
    const MatrixXd target= lz-lalpha*means;
    const MatrixXd target_bread=target*joint;
    MatrixXd design_gram=(lalpha.array().rowwise()/counts.transpose().array()).matrix()*lalpha.transpose();
    design_gram.noalias()+=target_bread*target.transpose();
    design_gram=(.5*(design_gram+design_gram.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<MatrixXd> design_eig(design_gram);
    if(design_eig.info()!=Eigen::Success || !finite_matrix(design_eig.eigenvalues()))
        throw std::runtime_error("worker projection: contrast-design eigensolve failed");
    const double rank_threshold=64.*dim*std::numeric_limits<double>::epsilon()*design_eig.eigenvalues().maxCoeff();
    std::vector<int> positive;
    for(int j=0;j<dim;++j) if(design_eig.eigenvalues()[j]>rank_threshold) positive.push_back(j);
    result.rank_diagnostic=static_cast<int>(positive.size());
    if(result.rank_diagnostic<jf)
        throw std::runtime_error("worker projection: firm-contrast design is numerically rank deficient");
    MatrixXd white(dim,result.rank_diagnostic);
    for(int j=0;j<result.rank_diagnostic;++j)
        white.col(j)=design_eig.eigenvectors().col(positive[j])/std::sqrt(design_eig.eigenvalues()[positive[j]]);
    RowMatrix scores(nw,dim);
    VectorXd trace=VectorXd::Zero(nw),leverage=VectorXd::Zero(nw);
    long double total_long=0.;
    for(int i=0;i<n;++i) {
        // Exact binary scaling also protects Householder QR: Eigen forms
        // squared column norms internally. Restore original units only after
        // all confidence-region calculations have completed.
        yrel[i]=std::scalbn(yrel[i],-response_exponent);
        total_long+=static_cast<long double>(yrel[i]);
    }
    const double total=static_cast<double>(total_long);
    observer.begin_region(team);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(team)
#endif
    for(int w=0;w<nw;++w) {
        observer.observe_work();
        const int count=ptr[w+1]-ptr[w];
        std::vector<int> support;
        for(int j=0;j<jf;++j) if(means(w,j)!=0.) support.push_back(j);
        for(int c=0;c<pc;++c) if(support_control(w,c)) support.push_back(jf+c);
        const int width=static_cast<int>(support.size());
        MatrixXd local_z(count,width),sub_bread(width,width),target_sub(dim,width);
        for(int j=0;j<width;++j) {
            const int original=support[j];
            target_sub.col(j)=target_bread.col(original);
            for(int k=0;k<width;++k) sub_bread(j,k)=joint(original,support[k]);
            for(int r=0;r<count;++r) {
                const int i=order[ptr[w]+r];
                const double value=original<jf ? (firm[i]==original ? 1.:0.)
                    : (*controls)(rows[i],control_columns[original-jf]);
                local_z(r,j)=value-means(w,original);
            }
        }
        const MatrixXd bread_z=sub_bread*local_z.transpose();
        MatrixXd local_k=local_z*target_sub.transpose();
        local_k.rowwise()+=(lalpha.col(w)/counts[w]).transpose();
        trace[w]=(local_k*white).squaredNorm();
        VectorXd residual(count),local(count);
        double cluster_sum=0.;
        for(int r=0;r<count;++r) cluster_sum+=yrel[order[ptr[w]+r]];
        const double centre=(total-cluster_sum)/(n-count);
        for(int r=0;r<count;++r) {
            const int i=order[ptr[w]+r];
            residual[r]=std::scalbn(adjusted[i]-alpha[w]-psi[firm[i]],-response_exponent);
            local[r]=yrel[i]-centre;
            leverage[w]=std::max(leverage[w],1./count+local_z.row(r).dot(bread_z.col(r)));
        }
        for(int degree=0;degree<5;++degree) {
            const double worker_part=local.sum()/count;
            VectorXd next=residual+local_z*(bread_z*local);
            next.array()+=worker_part;
            local=std::move(next);
        }
        scores.row(w)=(local_k.transpose()*local).transpose();
    }
    observer.end_region();
    result.threads_used=observer.max_team_size();
    result.parallel_workers_active=observer.active_workers();
    result.max_cluster_leverage_trace=trace.maxCoeff();
    result.max_row_leverage=leverage.maxCoeff();
    VectorXd alpha_rel=alpha.array()-alpha[0],psi_rel=psi.array()-psi[0];
    for(int w=0;w<nw;++w) alpha_rel[w]=std::scalbn(alpha_rel[w],-response_exponent);
    for(int f=0;f<nf;++f) psi_rel[f]=std::scalbn(psi_rel[f],-response_exponent);
    alpha_rel.array()-=counts.dot(alpha_rel)/n;
    psi_rel.array()-=fc.dot(psi_rel)/n;
    result.contrasts.resize(dim);
    result.contrasts.head(jf)=contrast*psi_rel;
    result.contrasts.tail(jf)=lalpha.bottomRows(jf)*alpha_rel;
    if(!finite_matrix(scores) || !finite_matrix(result.contrasts))
        throw std::runtime_error("worker projection: non-finite cluster scores");
    Eigen::HouseholderQR<MatrixXd> qr(scores);
    const int root_columns=std::min(nw,dim);
    MatrixXd triangular=MatrixXd::Zero(root_columns,dim);
    for(int i=0;i<root_columns;++i) for(int j=i;j<dim;++j) triangular(i,j)=qr.matrixQR()(i,j);
    const MatrixXd root=triangular.transpose();
    result.covariance=root*root.transpose();
    if(root_columns<result.rank_diagnostic)
        throw std::runtime_error("worker projection: fewer worker clusters than identified target contrasts");
    const MatrixXd whitened_root=root.transpose()*white;
    Eigen::JacobiSVD<MatrixXd> variance_svd(whitened_root);
    if(variance_svd.info()!=Eigen::Success || !finite_matrix(variance_svd.singularValues()) ||
       !(variance_svd.singularValues().minCoeff()>64.*dim*std::numeric_limits<double>::epsilon()*variance_svd.singularValues().maxCoeff()))
        throw std::runtime_error("worker projection: cluster covariance is rank deficient or ill-scaled on the target contrasts");
    const double sd_condition=variance_svd.singularValues().maxCoeff()/variance_svd.singularValues().minCoeff();
    result.covariance_condition=sd_condition*sd_condition;
    for(int component=0;component<2;++component) {
        MatrixXd form=MatrixXd::Zero(dim,dim);
        const double normaliser=static_cast<double>(n)/(n-1.);
        if(component==0) form.topLeftCorner(jf,jf).diagonal().setConstant(normaliser);
        else {
            form.topRightCorner(jf,jf).diagonal().setConstant(.5*normaliser);
            form.bottomLeftCorner(jf,jf).diagonal().setConstant(.5*normaliser);
        }
        result.se[component]=2.*(root.transpose()*form*result.contrasts).stableNorm();
        const auto range=quadratic_range(result.contrasts,root,form,result.critical_value);
        result.ci_lb[component]=component==0 ? std::max(0.,range.lower):range.lower;
        result.ci_ub[component]=range.upper;
        if(!hdfe::detail::ieee_finite(result.se[component]) ||
           !hdfe::detail::ieee_finite(result.ci_lb[component]) ||
           !hdfe::detail::ieee_finite(result.ci_ub[component]) ||
           result.ci_lb[component]>result.ci_ub[component])
            throw std::runtime_error("worker projection: non-finite or unordered component bounds");
        result.max_trust_region_gap=std::max(result.max_trust_region_gap,range.gap);
        // The index is dimensionless. Normalise before its fourth-order
        // products so a diagnostic cannot overflow on a valid scaled fit.
        const double diagnostic_scale=std::max(result.contrasts.cwiseAbs().maxCoeff(),root.cwiseAbs().maxCoeff());
        const VectorXd diagnostic_centre=result.contrasts/diagnostic_scale;
        const MatrixXd diagnostic_root=root/diagnostic_scale;
        const MatrixXd gradient_form=form*(diagnostic_root*diagnostic_root.transpose())*form;
        const auto gradient=quadratic_range(diagnostic_centre,diagnostic_root,gradient_form,result.critical_value);
        MatrixXd absolute_form=MatrixXd::Identity(dim,dim)*normaliser*(component==0 ? 1.:.5);
        if(component==0) absolute_form.bottomRightCorner(jf,jf).setZero();
        Eigen::SelfAdjointEigenSolver<MatrixXd> nonlinear(diagnostic_root.transpose()*absolute_form*diagnostic_root);
        if(nonlinear.info()!=Eigen::Success)
            throw std::runtime_error("worker projection: linearisation diagnostic failed");
        result.linearisation_index[component]=gradient.lower>0.
            ? result.critical_value*nonlinear.eigenvalues().maxCoeff()/(2.*std::sqrt(gradient.lower))
            : std::numeric_limits<double>::infinity();
    }
    for(Eigen::Index j=0;j<result.contrasts.size();++j)
        result.contrasts[j]=std::scalbn(result.contrasts[j],response_exponent);
    for(Eigen::Index j=0;j<result.covariance.size();++j)
        result.covariance.data()[j]=std::scalbn(result.covariance.data()[j],2*response_exponent);
    result.max_trust_region_gap=std::scalbn(result.max_trust_region_gap,2*response_exponent);
    for(int component=0;component<2;++component) {
        const bool had_width=result.ci_ub[component]>result.ci_lb[component];
        result.se[component]=std::scalbn(result.se[component],2*response_exponent);
        result.ci_lb[component]=std::scalbn(result.ci_lb[component],2*response_exponent);
        result.ci_ub[component]=std::scalbn(result.ci_ub[component],2*response_exponent);
        if(!hdfe::detail::ieee_finite(result.se[component]) ||
           !hdfe::detail::ieee_finite(result.ci_lb[component]) ||
           !hdfe::detail::ieee_finite(result.ci_ub[component]) ||
           (had_width && !(result.ci_ub[component]>result.ci_lb[component])))
            throw std::runtime_error("worker projection: component interval exceeds the representable outcome scale");
    }
    if(!finite_matrix(result.contrasts) || !finite_matrix(result.covariance) ||
       !hdfe::detail::ieee_finite(result.max_trust_region_gap))
        throw std::runtime_error("worker projection: non-finite rescaled inference output");
    return result;
}
}}} // namespace hdfe::akm::projection
