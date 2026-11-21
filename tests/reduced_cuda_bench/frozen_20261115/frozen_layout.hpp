#ifndef XHDFE_TESTS_REDUCED_CUDA_BENCH_FROZEN_LAYOUT_HPP
#define XHDFE_TESTS_REDUCED_CUDA_BENCH_FROZEN_LAYOUT_HPP
// FROZEN reference for the bench only: src/reduced_schur_pcg_cuda_layout.hpp as it stood before
// package 2 (see frozen_operator.hpp); only the include guard, the include and the namespace are renamed.

#include "reduced_schur_pcg.hpp"
#include "frozen_operator.hpp"
#include <chrono>
#include <climits>

namespace xhdfe_bench::frozen {

// Trace-only split of make_layout (event=attribution); no clock is read for it
// when the caller passes none.
struct LayoutSplit {
    double zero_seconds = 0.0;   // the value-initialising resize/assign calls
    double copy_seconds = 0.0;   // cells in segment order: segments, codes, multiplicities
    double sort_seconds = 0.0;   // the counting sorts by level (csr_ptr, csr_cell)
    double items_seconds = 0.0;  // segment items and level items
};

inline Layout make_layout(const hdfe::reduced_pcg::internal::Geometry& g, int lanes,
                          std::size_t host_budget, std::size_t host_reserved,
                          LayoutSplit* split = nullptr) {
    using Clock = std::chrono::steady_clock;
    Clock::time_point lap_from = split ? Clock::now() : Clock::time_point{};
    const auto lap = [&](double LayoutSplit::*field) {
        if (!split) return;
        const auto now = Clock::now();
        split->*field += std::chrono::duration<double>(now - lap_from).count();
        lap_from = now;
    };
    constexpr int chunk=1024;
    if(g.C_op<0 || g.C_op>INT_MAX || g.L_op<0 || g.m<0 || g.nb<1 || g.nb>4 || g.m>INT_MAX-g.nb)
        throw Error(ErrorKind::Memory,"CUDA reduced layout: shape capacity");
    const std::size_t C=static_cast<std::size_t>(g.C_op),L=g.L_op,m=g.m,b=g.nb;
    // Conservative pre-allocation bound, including construction cursors and
    // vector capacities. This is host memory; device admission is separate.
    const long double amax=L+C/chunk+1, bmax=m+b*(C/chunk+1);
    const long double bound=4.0L*((2*b+2)*C+4*m+6*amax+6*bmax+8*L+8*b+16)+8.0L*L;
    if(host_reserved>host_budget || bound>static_cast<long double>(host_budget-host_reserved))
        throw Error(ErrorKind::Memory,"CUDA reduced layout: host budget");
    Layout out;out.dims=g.nb;out.levels=g.m;out.segments=g.L_op;out.cells=static_cast<int>(C);out.lanes=lanes;
    out.offsets=g.offset;out.segment_ptr.resize(L+1);out.cell_segment.resize(C);
    out.codes.resize(b*C);if(g.mult)out.mult.resize(C);out.segment_inverse.resize(L);
    lap(&LayoutSplit::zero_seconds);
    int pos=0;
    for(int s=0;s<g.L_op;++s) {
        const int l=g.op_seg[s];out.segment_ptr[s]=pos;out.segment_inverse[s]=g.seg_inv[l];
        for(int c=g.cell_ptr[l];c<g.cell_ptr[l+1];++c,++pos) {
            out.cell_segment[pos]=s;
            for(int j=0;j<g.nb;++j)out.codes[static_cast<std::size_t>(j)*C+pos]=g.cc[j][c];
            if(g.mult)out.mult[pos]=g.mult[c];
        }
    }
    out.segment_ptr[L]=pos;
    if(static_cast<std::size_t>(pos)!=C)throw Error(ErrorKind::Fatal,"CUDA reduced layout: cell count mismatch");
    lap(&LayoutSplit::copy_seconds);
    out.csr_offsets.resize(b);out.csr_ptr.assign(m+b,0);out.csr_cell.resize(b*C);
    std::vector<int> cursor(m);
    lap(&LayoutSplit::zero_seconds);
    for(int j=0;j<g.nb;++j) {
        const int off=g.offset[j],mj=g.offset[j+1]-off,pbase=off+j;out.csr_offsets[j]=pbase;
        for(std::size_t c=0;c<C;++c)++out.csr_ptr[pbase+out.codes[static_cast<std::size_t>(j)*C+c]+1];
        for(int lev=0;lev<mj;++lev)out.csr_ptr[pbase+lev+1]+=out.csr_ptr[pbase+lev];
        for(int lev=0;lev<mj;++lev)cursor[off+lev]=out.csr_ptr[pbase+lev];
        for(std::size_t c=0;c<C;++c) {
            const int lev=out.codes[static_cast<std::size_t>(j)*C+c];
            out.csr_cell[static_cast<std::size_t>(j)*C+cursor[off+lev]++]=static_cast<int>(c);
        }
    }
    lap(&LayoutSplit::sort_seconds);
    out.a_item_ptr.resize(L+1);
    out.a_item_segment.reserve(static_cast<std::size_t>(amax));
    out.a_item_begin.reserve(static_cast<std::size_t>(amax));out.a_item_end.reserve(static_cast<std::size_t>(amax));
    for(int s=0;s<g.L_op;++s) {
        if(out.a_item_begin.size()>INT_MAX)throw Error(ErrorKind::Memory,"CUDA reduced layout: segment item capacity");
        out.a_item_ptr[s]=static_cast<int>(out.a_item_begin.size());
        for(int begin=out.segment_ptr[s];begin<out.segment_ptr[s+1];) {
            const int end=begin+std::min(chunk,out.segment_ptr[s+1]-begin);
            out.a_item_segment.push_back(s);out.a_item_begin.push_back(begin);out.a_item_end.push_back(end);begin=end;
        }
    }
    if(out.a_item_begin.size()>INT_MAX)throw Error(ErrorKind::Memory,"CUDA reduced layout: segment item capacity");
    out.a_item_ptr[L]=static_cast<int>(out.a_item_begin.size());
    out.b_item_ptr.resize(m+1);out.b_item_level.reserve(static_cast<std::size_t>(bmax));
    out.b_item_begin.reserve(static_cast<std::size_t>(bmax));out.b_item_end.reserve(static_cast<std::size_t>(bmax));
    for(int j=0;j<g.nb;++j)for(int lev=g.offset[j];lev<g.offset[j+1];++lev) {
        if(out.b_item_begin.size()>INT_MAX)throw Error(ErrorKind::Memory,"CUDA reduced layout: level item capacity");
        out.b_item_ptr[lev]=static_cast<int>(out.b_item_begin.size());const int pbase=out.csr_offsets[j]+lev-g.offset[j];
        for(int begin=out.csr_ptr[pbase];begin<out.csr_ptr[pbase+1];) {
            const int end=begin+std::min(chunk,out.csr_ptr[pbase+1]-begin);
            out.b_item_level.push_back(lev);out.b_item_begin.push_back(begin);out.b_item_end.push_back(end);begin=end;
        }
    }
    if(out.b_item_begin.size()>INT_MAX)throw Error(ErrorKind::Memory,"CUDA reduced layout: level item capacity");
    out.b_item_ptr[m]=static_cast<int>(out.b_item_begin.size());
    lap(&LayoutSplit::items_seconds);
    return out;
}

} // namespace xhdfe_bench::frozen
#endif
