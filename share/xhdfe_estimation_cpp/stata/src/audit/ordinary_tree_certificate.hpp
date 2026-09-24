#ifndef XHDFE_ORDINARY_TREE_CERTIFICATE_HPP
#define XHDFE_ORDINARY_TREE_CERTIFICATE_HPP

#include "../group_forward_certificate.hpp"
#include "ordinary_certification.hpp"
#include <numeric>

namespace hdfe { namespace detail {

// Outward intervals also survive the production fast-math flags. Potentials
// below are arbitrary stored witnesses; only evaluating their defect needs
// an enclosure, so rounding along a long tree cannot invalidate the proof.
struct OrdinaryTreeInterval {
    using Real=long double;
    Real lo=0,hi=0;
    static Real up(Real x) { return std::nextafter(x,std::numeric_limits<Real>::infinity()); }
    static Real down(Real x) { return std::nextafter(x,-std::numeric_limits<Real>::infinity()); }
    static OrdinaryTreeInterval point(Real x) { return {x,x}; }
    OrdinaryTreeInterval operator-() const { return {-hi,-lo}; }
    OrdinaryTreeInterval operator+(const OrdinaryTreeInterval& b) const {
        if (b.lo==0 && b.hi==0) return *this;
        if (lo==0 && hi==0) return b;
        volatile Real low=lo+b.lo,high=hi+b.hi;
        return {down(low),up(high)};
    }
    OrdinaryTreeInterval operator-(const OrdinaryTreeInterval& b) const { return *this+(-b); }
    static OrdinaryTreeInterval product(Real a,Real b) {
        if (a==0 || b==0) return {};
        volatile Real value=a*b;
        return {down(value),up(value)};
    }
    Real absolute() const { return std::max(std::abs(lo),std::abs(hi)); }
    static Real add_upper(Real a,Real b) {
        if (a==0) return b;
        if (b==0) return a;
        volatile Real value=a+b;return up(value);
    }
    static Real multiply_upper(Real a,Real b) {
        if (a==0 || b==0) return 0;
        volatile Real value=a*b;return up(value);
    }
};

struct OrdinaryTreeSpaceCheck : GroupForwardCheck {
    std::vector<long double> fe_bound,affine_bound;
};

inline OrdinaryTreeSpaceCheck certify_ordinary_tree_space(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const AbsorptionResult& result,const std::vector<long double>& targets) {
    using I=OrdinaryTreeInterval;using Real=I::Real;
    OrdinaryTreeSpaceCheck check;check.limit=1;check.passed=false;
    if (!ordinary_audit_enabled()) return check;
    ++ordinary_audit_work_count;
    const int n=static_cast<int>(y.size()),rhs=static_cast<int>(X.cols())+1;
    if (fes.size()!=2 || targets.size()!=static_cast<std::size_t>(rhs)) return check;
    struct Edge {int a,b,row;I weight;};
    std::unordered_map<int,int> left,right;
    std::vector<int> a(n),b(n),edge_of(n);
    for (int row=0;row<n;++row) {
        a[row]=left.emplace(fes[0][row],static_cast<int>(left.size())).first->second;
        b[row]=right.emplace(fes[1][row],static_cast<int>(right.size())).first->second;
    }
    const int nl=static_cast<int>(left.size()),vertices=nl+static_cast<int>(right.size());
    std::unordered_map<std::uint64_t,int> ids;
    std::vector<Edge> edges;
    for (int row=0;row<n;++row) {
        b[row]+=nl;
        const std::uint64_t key=(static_cast<std::uint64_t>(a[row])<<32)|static_cast<std::uint32_t>(b[row]);
        auto added=ids.emplace(key,static_cast<int>(edges.size()));
        if (added.second) edges.push_back({a[row],b[row],row,{}});
        edge_of[row]=added.first->second;
        auto& mass=edges[edge_of[row]].weight;
        mass=mass+I::point(weights ? (*weights)[row] : 1.0L);
    }
    std::vector<int> sets(vertices),order(edges.size());
    std::iota(sets.begin(),sets.end(),0);std::iota(order.begin(),order.end(),0);
    auto root=[&](int node) {
        while (sets[node]!=node) {sets[node]=sets[sets[node]];node=sets[node];}
        return node;
    };
    std::stable_sort(order.begin(),order.end(),[&](int x,int z) {return edges[x].weight.lo>edges[z].weight.lo;});
    std::vector<std::vector<std::pair<int,int>>> adjacent(vertices);
    for (int e:order) {
        const int ra=root(edges[e].a),rb=root(edges[e].b);
        if (ra==rb) continue;
        sets[ra]=rb;
        adjacent[edges[e].a].emplace_back(edges[e].b,e);
        adjacent[edges[e].b].emplace_back(edges[e].a,e);
    }
    std::vector<int> parent(vertices,-1),parent_edge(vertices,-1),traversal;
    traversal.reserve(vertices);
    for (int start=0;start<vertices;++start) if (parent[start]<0) {
        parent[start]=start;traversal.push_back(start);
        const auto begin=traversal.size()-1;
        for (std::size_t index=begin;index<traversal.size();++index) {
            const int node=traversal[index];
            for (auto [next,e]:adjacent[node]) if (parent[next]<0) {
                parent[next]=node;parent_edge[next]=e;traversal.push_back(next);
            }
        }
    }
    check.eligible=true;check.passed=true;
    check.fe_bound.resize(rhs);check.affine_bound.resize(rhs);
    auto raw=[&](int row,int column) -> Real {return column ? X(row,column-1) : y[row];};
    auto residual=[&](int row,int column) -> Real {
        return column ? result.X_tilde(row,column-1) : result.y_tilde[row];
    };
    for (int column=0;column<rhs;++column) {
        std::vector<I> flow(vertices);
        for (int row=0;row<n;++row) {
            const auto value=I::product(weights ? (*weights)[row] : 1.0L,residual(row,column));
            flow[a[row]]=flow[a[row]]+value;flow[b[row]]=flow[b[row]]-value;
        }
        Real energy=0;
        for (auto node=traversal.rbegin();node!=traversal.rend();++node) {
            if (parent[*node]==*node) continue;
            const Real lower=edges[parent_edge[*node]].weight.lo;
            if (!(lower>0)) {check.passed=false;check.unavailable="tree weight enclosure is not positive";return check;}
            const Real upper=flow[*node].absolute();
            volatile Real term=I::multiply_upper(upper,upper)/lower;
            energy=I::add_upper(energy,I::up(term));
            flow[parent[*node]]=flow[parent[*node]]+flow[*node];
        }
        std::vector<Real> potential(vertices,0);
        for (int node:traversal) if (parent[node]!=node) {
            const auto& edge=edges[parent_edge[node]];
            volatile Real difference=raw(edge.row,column)-residual(edge.row,column);
            volatile Real next=potential[parent[node]]+(node==edge.a ? difference : -difference);
            potential[node]=next;
        }
        Real affine_squared=0;
        for (int row=0;row<n;++row) {
            const auto defect=I::point(raw(row,column))-I::point(residual(row,column))-
                (I::point(potential[a[row]])-I::point(potential[b[row]]));
            const Real upper=defect.absolute();
            const Real term=I::multiply_upper(weights ? (*weights)[row] : 1.0L,I::multiply_upper(upper,upper));
            affine_squared=I::add_upper(affine_squared,term);
        }
        // D is the oriented two-FE incidence. The tree has the same kernel:
        // M=D_T' W_T D_T <= A=D' W D, hence g' A^+ g <= g' M^+ g.
        // The latter is sum(subtree_flow^2 / tree_weight). The affine defect
        // contributes through the orthogonal complement, giving this bound.
        const Real bound=I::up(std::sqrt(I::add_upper(energy,affine_squared)));
        check.fe_bound[column]=energy==0 ? 0 : I::up(std::sqrt(energy));
        check.affine_bound[column]=affine_squared==0 ? 0 : I::up(std::sqrt(affine_squared));
        const double ratio=targets[column]>0 ? static_cast<double>(bound/targets[column]) :
            bound==0 ? 0 : std::numeric_limits<double>::infinity();
        check.worst_ratio=std::max(check.worst_ratio,ratio);
        check.passed=check.passed && ieee_finite(ratio) && ratio<=1;
    }
    return check;
}

}}
#endif
