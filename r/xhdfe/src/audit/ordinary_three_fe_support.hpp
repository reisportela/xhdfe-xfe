#ifndef XHDFE_ORDINARY_THREE_FE_SUPPORT_HPP
#define XHDFE_ORDINARY_THREE_FE_SUPPORT_HPP

#include "ordinary_tree_certificate.hpp"
#include "ols_numerical_certificate.hpp"
#include "ordinary_support_accumulation.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace hdfe { namespace detail {

struct OrdinaryThreeFeSupportCheck : GroupForwardCheck {
    std::size_t positive_primary_pairs=0;
    std::vector<long double> fe_bound,affine_bound;
};

// H09, adapted to the v82 support proof: label translation and compaction are
// representation changes only.  The proof still receives every observation,
// weight and residual in the original order.
struct OrdinaryTranslatedFeView {
    const Eigen::VectorXi& values;
    std::int64_t minimum;
    int operator[](int row) const {
        return minimum==0 ? values[row] :
            static_cast<int>(static_cast<std::int64_t>(values[row])-minimum);
    }
};

struct OrdinaryCompactFeView {
    OrdinaryTranslatedFeView raw;
    const std::array<int,64>* mapping;
    int operator[](int row) const {
        const int id=raw[row];
        return mapping ? (*mapping)[id] : id;
    }
};

struct OrdinaryThreeFeSupportWorkspace {
    using Clock=std::chrono::steady_clock;
    using Evaluator=std::function<OrdinaryThreeFeSupportCheck(
        const Eigen::Ref<const Eigen::VectorXd>&,const Eigen::Ref<const Eigen::MatrixXd>&,
        const std::vector<Eigen::VectorXi>&,const Eigen::VectorXd*,const AbsorptionResult&,
        const std::vector<long double>&,Clock::time_point,bool)>;
    int observations=-1;
    bool had_weights=false;
    std::vector<Eigen::VectorXi> signature_fes;
    Eigen::VectorXd signature_weights;
    Evaluator evaluate;
    std::size_t preparations=0,reuses=0;
    bool matches(int n,const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights) const {
        if (!evaluate || observations!=n || had_weights!=(weights!=nullptr) || signature_fes.size()!=fes.size()) return false;
        for (std::size_t d=0;d<fes.size();++d)
            if (signature_fes[d].size()!=fes[d].size() ||
                (fes[d].size() && std::memcmp(signature_fes[d].data(),fes[d].data(),static_cast<std::size_t>(fes[d].size())*sizeof(int)))) return false;
        return !weights || (signature_weights.size()==weights->size() &&
            (!weights->size() || !std::memcmp(signature_weights.data(),weights->data(),static_cast<std::size_t>(weights->size())*sizeof(double))));
    }
    void clear_geometry() {
        evaluate={};signature_fes.clear();signature_weights.resize(0);observations=-1;
    }
    void bind(int n,const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,Evaluator&& next) {
        observations=n;had_weights=weights!=nullptr;signature_fes=fes;
        if (weights) signature_weights=*weights;else signature_weights.resize(0);
        evaluate=std::move(next);++preparations;
    }
};

// Diagnostic implementation. A forest of primary pairs plus within-pair
// categorical contrasts supplies a full-rank lower information matrix.
// No full FE dummy matrix or large numerical rank factorization is built.
inline OrdinaryThreeFeSupportCheck certify_ordinary_three_fe_support(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const AbsorptionResult& result,const std::vector<long double>& targets,
    OrdinaryThreeFeSupportWorkspace* workspace=nullptr,
    int threads=1,ParallelWorkObserver* observer=nullptr) {
    using I=OlsProofInterval;using Real=long double;
    using Matrix=Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>;
    using Vector=Eigen::Matrix<Real,Eigen::Dynamic,1>;
    OrdinaryThreeFeSupportCheck check;check.passed=false;check.limit=1;
    if (!ordinary_audit_enabled()) return check;
    ++ordinary_audit_work_count;
    const int n=static_cast<int>(y.size()),rhs=static_cast<int>(X.cols())+1;
    if (fes.size()!=3 || n==0 || targets.size()!=static_cast<std::size_t>(rhs)) return check;
    constexpr bool trace=false;
    const auto started=std::chrono::steady_clock::now();
    if (workspace && workspace->matches(n,fes,weights)) {
        ++workspace->reuses;
        if (trace) std::fprintf(stderr,"xhdfe ordinary support: geometry_cache=hit\n");
        return workspace->evaluate(y,X,fes,weights,result,targets,started,trace);
    }
    if (workspace) workspace->clear_geometry();
    std::array<int,3> dimensions={0,1,2},sizes;
    std::array<std::int64_t,3> minimums;
    for (int d=0;d<3;++d) {
        if (fes[d].size()!=n) return check;
        const std::int64_t minimum=static_cast<std::int64_t>(fes[d].minCoeff());
        const std::int64_t maximum=static_cast<std::int64_t>(fes[d].maxCoeff());
        const std::int64_t range=maximum-minimum+1;
        if (range<=0 || range>static_cast<std::int64_t>(n)) return check;
        minimums[d]=minimum;sizes[d]=static_cast<int>(range);
    }
    std::stable_sort(dimensions.begin(),dimensions.end(),[&](int a,int b){return sizes[a]>sizes[b];});
    const OrdinaryTranslatedFeView left{fes[dimensions[0]],minimums[dimensions[0]]};
    const OrdinaryTranslatedFeView right{fes[dimensions[1]],minimums[dimensions[1]]};
    const OrdinaryTranslatedFeView raw_coarse{fes[dimensions[2]],minimums[dimensions[2]]};
    const int nl=sizes[dimensions[0]],raw_nc=sizes[dimensions[2]];
    if (raw_nc<2 || raw_nc>64) return check;
    std::uint64_t present_mask=0;
    const int presence_team=std::max(1,threads);
    if (observer) observer->begin_region(presence_team);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(presence_team) reduction(|:present_mask)
#endif
    {
        bool observed=false;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int row=0;row<n;++row) {
            if (!observed && observer) observer->observe_work();
            observed=true;
            present_mask|=std::uint64_t{1}<<static_cast<unsigned>(raw_coarse[row]);
        }
    }
    if (observer) observer->end_region();
    std::array<int,64> coarse_mapping;coarse_mapping.fill(-1);
    int nc=0;
    for (int raw=0;raw<raw_nc;++raw)
        if (present_mask&(std::uint64_t{1}<<static_cast<unsigned>(raw)))
            coarse_mapping[raw]=nc++;
    const bool coarse_identity=nc==raw_nc;
    const OrdinaryCompactFeView coarse{raw_coarse,coarse_identity ? nullptr : &coarse_mapping};
    const std::int64_t vertices_wide=static_cast<std::int64_t>(nl)+sizes[dimensions[1]];
    if (vertices_wide>std::numeric_limits<int>::max() || nc<2) return check;
    const int vertices=static_cast<int>(vertices_wide);
    auto divide=[](const I& a,const I& b) {
        if (!(b.lo>0)) return I{-std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::infinity()};
        return a*I{I::down(1/b.hi),I::up(1/b.lo)};
    };
    auto midpoint=[](const I& value){return value.lo/2+value.hi/2;};
    struct Edge {int a,b,head,count;I mass;};
    std::unordered_map<std::uint64_t,int> pair_ids;
    std::vector<Edge> edges;std::vector<int> next(n);
    for (int row=0;row<n;++row) {
        const int a=left[row],b=nl+right[row];
        const auto key=(static_cast<std::uint64_t>(a)<<32)|static_cast<std::uint32_t>(b);
        const auto added=pair_ids.emplace(key,static_cast<int>(edges.size()));
        if (added.second) edges.push_back({a,b,-1,0,{}});
        auto& edge=edges[added.first->second];next[row]=edge.head;edge.head=row;
        if (weights) edge.mass=edge.mass+I::point((*weights)[row]);
        else ++edge.count;
    }
    if (!weights) for (auto& edge:edges) edge.mass=I::point(edge.count);
    for (const auto& edge:edges) if (edge.mass.lo>0) ++check.positive_primary_pairs;
    pair_ids.clear();pair_ids.rehash(0);
    std::vector<int> sets(vertices),sorted(edges.size());
    std::iota(sets.begin(),sets.end(),0);std::iota(sorted.begin(),sorted.end(),0);
    auto root=[&](int node) {while (sets[node]!=node) {sets[node]=sets[sets[node]];node=sets[node];}return node;};
    std::stable_sort(sorted.begin(),sorted.end(),[&](int a,int b){return edges[a].mass.lo>edges[b].mass.lo;});
    struct Arc {int node,edge,next;};
    std::vector<int> head(vertices,-1),tree;
    std::vector<Arc> arcs;
    for (int e:sorted) {
        const auto& edge=edges[e];const int a=root(edge.a),b=root(edge.b);
        if (a==b) continue;
        sets[a]=b;tree.push_back(e);
        arcs.push_back({edge.b,e,head[edge.a]});head[edge.a]=static_cast<int>(arcs.size())-1;
        arcs.push_back({edge.a,e,head[edge.b]});head[edge.b]=static_cast<int>(arcs.size())-1;
    }
    std::vector<int> parent(vertices,-1),parent_edge(vertices,-1),traversal,edge_child(edges.size(),-1);
    for (int start=0;start<vertices;++start) if (parent[start]<0) {
        parent[start]=start;const auto begin=traversal.size();traversal.push_back(start);
        for (auto pos=begin;pos<traversal.size();++pos) {
            const int node=traversal[pos];
            for (int p=head[node];p>=0;p=arcs[p].next) if (parent[arcs[p].node]<0) {
                parent[arcs[p].node]=node;parent_edge[arcs[p].node]=arcs[p].edge;
                edge_child[arcs[p].edge]=arcs[p].node;
                traversal.push_back(arcs[p].node);
            }
        }
    }
    // The coarse Schur complement is a Laplacian. Each pair contributes
    // m_j*m_k/m_pair to edge (j,k), so the construction stays positive.
    std::vector<I> conductance(static_cast<std::size_t>(nc)*nc),mass(nc);
    std::vector<int> present,counts(nc,0);
    auto pair_masses=[&](int e) {
        for (int c:present) {mass[c]={};counts[c]=0;}
        present.clear();
        for (int row=edges[e].head;row>=0;row=next[row]) {
            const int c=coarse[row];
            if (!weights) {if (counts[c]++==0) present.push_back(c);}
            else {
                if (mass[c].hi==0) present.push_back(c);
                mass[c]=mass[c]+I::point((*weights)[row]);
            }
        }
        if (!weights) for (int c:present) mass[c]=I::point(counts[c]);
    };
    for (int e:tree) {
        pair_masses(e);
        for (std::size_t j=0;j<present.size();++j) for (std::size_t k=j+1;k<present.size();++k) {
            const int a=std::min(present[j],present[k]),b=std::max(present[j],present[k]);
            auto& value=conductance[a*nc+b];value=value+divide(mass[a]*mass[b],edges[e].mass);
        }
    }
    struct CoarseEdge {int a,b;I mass;};
    std::vector<CoarseEdge> coarse_edges;
    Matrix S=Matrix::Zero(nc-1,nc-1);
    for (int a=0;a<nc;++a) for (int b=a+1;b<nc;++b) {
        const auto value=conductance[a*nc+b];
        if (!(value.lo>0)) continue;
        coarse_edges.push_back({a,b,value});const Real w=midpoint(value);
        if (a) S(a-1,a-1)+=w;
        if (b) S(b-1,b-1)+=w;
        if (a && b) {S(a-1,b-1)-=w;S(b-1,a-1)-=w;}
    }
    std::stable_sort(coarse_edges.begin(),coarse_edges.end(),[](const auto& a,const auto& b){return a.mass.lo>b.mass.lo;});
    sets.resize(nc);std::iota(sets.begin(),sets.end(),0);
    std::vector<std::vector<std::pair<int,int>>> ca(nc);
    int coarse_rank=0;
    for (int e=0;e<static_cast<int>(coarse_edges.size());++e) {
        const auto& edge=coarse_edges[e];const int a=root(edge.a),b=root(edge.b);
        if (a==b) continue;
        sets[a]=b;++coarse_rank;
        ca[edge.a].emplace_back(edge.b,e);ca[edge.b].emplace_back(edge.a,e);
    }
    if (coarse_rank!=nc-1) {
        if (trace) std::fprintf(stderr,"xhdfe ordinary support: unavailable coarse_rank=%d required=%d\n",coarse_rank,nc-1);
        check.unavailable="primary forest has insufficient within-pair coarse variation";return check;
    }
    std::vector<int> cp(nc,-1),ce(nc,-1),ct={0};cp[0]=0;
    for (std::size_t pos=0;pos<ct.size();++pos) for (auto [node,e]:ca[ct[pos]]) if (cp[node]<0) {
        cp[node]=ct[pos];ce[node]=e;ct.push_back(node);
    }
    if (trace) std::fprintf(stderr,"xhdfe ordinary support: n=%d primary_vertices=%d pairs=%zu tree=%zu coarse=%d setup=%.6f\n",
        n,vertices,edges.size(),tree.size(),nc,std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
    check.eligible=true;check.passed=true;
    // Only immutable graph information is retained. Every residual, witness,
    // coefficient target and error bound is evaluated afresh on each call.
    OrdinaryThreeFeSupportWorkspace::Evaluator evaluate=[
        n,nl,vertices,nc,dimensions,minimums,coarse_mapping,coarse_identity,
        divide,midpoint,geometry=check,
        edges=std::move(edges),next=std::move(next),tree=std::move(tree),
        parent=std::move(parent),parent_edge=std::move(parent_edge),edge_child=std::move(edge_child),
        traversal=std::move(traversal),coarse_edges=std::move(coarse_edges),
        cp=std::move(cp),ce=std::move(ce),ct=std::move(ct),S=std::move(S)](
        const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
        const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
        const AbsorptionResult& result,const std::vector<long double>& targets,
        OrdinaryThreeFeSupportWorkspace::Clock::time_point started,bool trace) {
    auto check=geometry;
    const int rhs=static_cast<int>(X.cols())+1;
    check.fe_bound.assign(rhs,std::numeric_limits<Real>::infinity());
    check.affine_bound.assign(rhs,std::numeric_limits<Real>::infinity());
    const OrdinaryTranslatedFeView left{fes[dimensions[0]],minimums[dimensions[0]]};
    const OrdinaryTranslatedFeView right{fes[dimensions[1]],minimums[dimensions[1]]};
    const OrdinaryTranslatedFeView raw_coarse{fes[dimensions[2]],minimums[dimensions[2]]};
    const OrdinaryCompactFeView coarse{raw_coarse,coarse_identity ? nullptr : &coarse_mapping};
    const auto factor=S.ldlt();
    std::vector<I> mass(nc);
    std::vector<int> present,counts(nc,0);
    auto pair_masses=[&](int e) {
        for (int c:present) {mass[c]={};counts[c]=0;}
        present.clear();
        for (int row=edges[e].head;row>=0;row=next[row]) {
            const int c=coarse[row];
            if (!weights) {if (counts[c]++==0) present.push_back(c);}
            else {
                if (mass[c].hi==0) present.push_back(c);
                mass[c]=mass[c]+I::point((*weights)[row]);
            }
        }
        if (!weights) for (int c:present) mass[c]=I::point(counts[c]);
    };
    for (int column=0;column<rhs;++column) {
        auto raw=[&](int row)->Real{return column ? X(row,column-1) : y[row];};
        auto residual=[&](int row)->Real{return column ? result.X_tilde(row,column-1) : result.y_tilde[row];};
        OrdinaryExactGradientWindow window;
        if (!weights) window.inspect(n,[&](int row){return static_cast<double>(residual(row));},true);
        const bool exact=window.eligible;
        using Integer=OrdinaryExactGradientWindow::Integer;
        std::vector<Integer> exact_flow(exact ? vertices : 0),exact_coarse(exact ? nc : 0);
        std::vector<I> flow(exact ? 0 : vertices),gcoarse(nc);
        OrdinaryPositiveBound tail;
        if (exact) {
            for (int row=0;row<n;++row) {
                bool omitted=false;
                const auto value=window.encode(static_cast<double>(residual(row)),&omitted);
                if (omitted) tail.add_square_divisor(std::abs(residual(row)),1);
                exact_flow[left[row]]+=value;exact_flow[nl+right[row]]-=value;
                exact_coarse[coarse[row]]+=value;
            }
            for (int c=0;c<nc;++c) gcoarse[c]=window.interval(exact_coarse[c]);
        } else {
            for (int row=0;row<n;++row) {
                const I value=I::point(weights ? (*weights)[row] : 1)*I::point(residual(row));
                flow[left[row]]=flow[left[row]]+value;flow[nl+right[row]]=flow[nl+right[row]]-value;
                gcoarse[coarse[row]]=gcoarse[coarse[row]]+value;
            }
        }
        Real energy=0;
        OrdinaryPositiveBound primary_energy;
        for (auto p=traversal.rbegin();p!=traversal.rend();++p) if (parent[*p]!=*p) {
            const int e=parent_edge[*p];const I value=exact ? window.interval(exact_flow[*p]) : flow[*p];
            const Real upper=value.absolute();
            if (!weights) primary_energy.add_square_divisor(upper,edges[e].mass.lo);
            else energy=I::up(energy+divide(I::point(upper)*I::point(upper),edges[e].mass).hi);
            if (exact) exact_flow[parent[*p]]+=exact_flow[*p];
            else flow[parent[*p]]=flow[parent[*p]]+value;
        }
        if (!weights) energy=primary_energy.upper();
        auto edge_flow=[&](int e) {
            const int node=edge_child[e];
            const I value=exact ? window.interval(exact_flow[node]) : flow[node];
            return node==edges[e].a ? value : -value;
        };
        Vector h=Vector::Zero(nc-1);
        std::vector<Real> pair_mean(edges.size());
        std::vector<GroupForwardSum> by_level(nc);
        for (int e:tree) {
            pair_masses(e);GroupForwardSum total;
            for (int c:present) by_level[c]={};
            for (int row=edges[e].head;row>=0;row=next[row]) {
                volatile Real delta=raw(row)-residual(row);
                volatile Real value=(weights ? (*weights)[row] : 1)*delta;
                total.add(value);by_level[coarse[row]].add(value);
            }
            const Real m=midpoint(edges[e].mass),mean=total.value()/m;
            pair_mean[e]=mean;
            const auto f=edge_flow(e);
            for (int c:present) {
                if (present.size()==1) gcoarse[c]=gcoarse[c]-f;
                else {
                    const I proportion=divide(mass[c],edges[e].mass);
                    gcoarse[c]=gcoarse[c]-proportion*f;
                    if (c) h[c-1]+=by_level[c].value()-midpoint(mass[c])*mean;
                }
            }
        }
        for (auto p=ct.rbegin();p!=ct.rend();++p) if (*p) {
            const Real upper=gcoarse[*p].absolute();
            energy=I::up(energy+divide(I::point(upper)*I::point(upper),coarse_edges[ce[*p]].mass).hi);
            gcoarse[cp[*p]]=gcoarse[cp[*p]]+gcoarse[*p];
        }
        // For r=r_window+r_tail, ||P_F r|| <= ||P_F r_window||+||r_tail||.
        // The small entries remain in the returned residual and in the affine
        // witness check. They are never silently discarded from the bound.
        if (tail.count) {
            const I norm=I::point(I::up(std::sqrt(energy)))+I::point(I::up(std::sqrt(tail.upper())));
            energy=(norm*norm).hi;
        }
        // This solve creates an arbitrary witness only. Its error is fully
        // enclosed by the all-row affine-defect calculation below.
        const Vector gamma=factor.solve(h);
        std::vector<Real> potential(vertices,0);
        for (int node:traversal) if (parent[node]!=node) {
            const int e=parent_edge[node];pair_masses(e);GroupForwardSum correction;
            for (int c:present) if (c) correction.add(midpoint(mass[c])*gamma[c-1]);
            const Real value=pair_mean[e]-correction.value()/midpoint(edges[e].mass);
            volatile Real next_value=potential[parent[node]]+(node==edges[e].a ? value : -value);
            potential[node]=next_value;
        }
        Real affine=0;
        OrdinaryPositiveBound affine_bound;
        for (int row=0;row<n;++row) {
            if (!weights) {
                affine_bound.add_affine(1,raw(row),residual(row),potential[left[row]],
                    potential[nl+right[row]],coarse[row] ? gamma[coarse[row]-1] : 0);
                continue;
            }
            const I q=I::point(raw(row))-I::point(residual(row))-
                (I::point(potential[left[row]])-I::point(potential[nl+right[row]]))-
                I::point(coarse[row] ? gamma[coarse[row]-1] : 0);
            const Real upper=q.absolute();
            const I term=I::point(weights ? (*weights)[row] : 1)*I::point(upper)*I::point(upper);
            affine=I::up(affine+term.hi);
        }
        if (!weights) affine=affine_bound.upper();
        // Primary and coarse tree energies bound g' (F'WF)^+ g.
        // The witness defect bounds the orthogonal-complement error.
        const Real bound=I::up(std::sqrt(I::up(energy+affine)));
        check.fe_bound[column]=energy==0 ? 0 : I::up(std::sqrt(energy));
        check.affine_bound[column]=affine==0 ? 0 : I::up(std::sqrt(affine));
        const double ratio=targets[column]>0 ? static_cast<double>(I::up(bound/targets[column])) :
            bound==0 ? 0 : std::numeric_limits<double>::infinity();
        if (trace) std::fprintf(stderr,"xhdfe ordinary support: rhs=%d exact_gradient=%d tail_rows=%llu tail_norm=%.9Le gradient=%.9Le affine=%.9Le bound=%.9Le target=%.9Le ratio=%.9g elapsed=%.6f\n",
            column,exact ? 1 : 0,static_cast<unsigned long long>(tail.count),std::sqrt(tail.upper()),std::sqrt(energy),std::sqrt(affine),bound,targets[column],ratio,
            std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
        check.worst_ratio=std::max(check.worst_ratio,ratio);
        check.passed=check.passed && ieee_finite(ratio) && ratio<=1;
        // A single failed RHS rules out this certificate. Later columns
        // cannot change that decision, so do not evaluate them on this path.
        if (!check.passed) return check;
    }
    return check;
    };
    if (workspace) {
        workspace->bind(n,fes,weights,std::move(evaluate));
        return workspace->evaluate(y,X,fes,weights,result,targets,started,trace);
    }
    return evaluate(y,X,fes,weights,result,targets,started,trace);
}

}}
#endif
