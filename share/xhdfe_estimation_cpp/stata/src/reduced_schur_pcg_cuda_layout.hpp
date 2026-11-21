#ifndef HDFE_REDUCED_SCHUR_PCG_CUDA_LAYOUT_HPP
#define HDFE_REDUCED_SCHUR_PCG_CUDA_LAYOUT_HPP

#include "reduced_schur_pcg.hpp"
#include "reduced_schur_pcg_cuda.hpp"

namespace hdfe::detail::reduced_cuda {

// The CPU geometry as the operator reads it (package 2, 17 November 2026). Nothing
// is copied on the host: the operator uploads the arrays the geometry already holds
// in the order its kernels need (cells in segment order, the codes, multiplicities,
// 1/rows, and the transposes of the large effects, whose seg holds the ORIGINAL
// segment id), and builds the level-ordered arrays of the other effects on the
// device. The view borrows; the geometry must outlive the operator's set-up.
inline GeometryView view_of(const hdfe::reduced_pcg::internal::Geometry& g) {
    GeometryView v;
    if (g.nb < 1 || g.nb > kMaxDims) throw Error(ErrorKind::Fatal, "CUDA reduced operator: 1 to 4 remaining effects");
    v.nb = g.nb;
    v.m = g.m;
    v.L = g.L;
    v.L_op = g.L_op;
    v.C = g.C;
    v.C_op = g.C_op;
    v.cell_ptr = g.cell_ptr;
    v.op_seg = g.op_seg.data();
    v.seg_inv = g.seg_inv.data();
    v.mult = g.mult;
    for (int j = 0; j < g.nb; ++j) {
        const std::size_t J = static_cast<std::size_t>(j);
        v.cc[j] = g.cc[J];
        DimView& d = v.dims[j];
        d.levels = g.offset[J + 1U] - g.offset[J];
        d.offset = g.offset[J];
        d.transposed = g.large[J] != 0;
        if (d.transposed) {
            const auto& t = g.tr[J];
            d.tr_ptr = t.ptr.data();
            d.tr_seg = t.seg.data();
            for (int jj = 0; jj < g.nb; ++jj) d.tr_other[jj] = jj == j ? nullptr : t.other[static_cast<std::size_t>(jj)].data();
            d.has_mult = t.has_mult;
            d.tr_mult = t.has_mult ? t.mult.data() : nullptr;
        }
    }
    return v;
}

} // namespace hdfe::detail::reduced_cuda
#endif
