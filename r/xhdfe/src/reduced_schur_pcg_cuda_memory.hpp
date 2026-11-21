#ifndef HDFE_REDUCED_SCHUR_PCG_CUDA_MEMORY_HPP
#define HDFE_REDUCED_SCHUR_PCG_CUDA_MEMORY_HPP

#include "reduced_schur_pcg.hpp"
#include "reduced_schur_pcg_cuda.hpp"
#include <initializer_list>
#include <limits>

namespace hdfe::detail::reduced_cuda {

inline std::size_t host_sum(std::initializer_list<std::size_t> terms) {
    std::size_t total=0;
    for(const auto bytes:terms) {
        if(bytes>std::numeric_limits<std::size_t>::max()-total)
            throw Error(ErrorKind::Memory,"CUDA reduced tier: host byte count overflow");
        total+=bytes;
    }
    return total;
}
template<class T> inline std::size_t owned_bytes(const std::vector<T>& v) { return v.capacity()*sizeof(T); }
template<class T> inline std::size_t owned_bytes(const hdfe::reduced_pcg::internal::Buffer<T>& v) { return v.size()*sizeof(T); }

// Resident storage after build_geometry returns. Its scratch is gone; Work
// has not yet been allocated. Borrowed input pointers contribute no storage.
inline std::size_t geometry_resident_bytes(const hdfe::reduced_pcg::internal::Geometry& g) {
    std::size_t bytes=host_sum({sizeof(g),owned_bytes(g.remaining),owned_bytes(g.offset),
        owned_bytes(g.perm),owned_bytes(g.row_ptr),owned_bytes(g.cell_ptr_own),
        owned_bytes(g.cell_code_own),owned_bytes(g.cc),owned_bytes(g.mult_own),
        owned_bytes(g.op_seg),owned_bytes(g.op_bounds),owned_bytes(g.all_bounds),
        owned_bytes(g.row_bounds),owned_bytes(g.large),owned_bytes(g.small_list),
        owned_bytes(g.small_offset),owned_bytes(g.small_target),owned_bytes(g.tr),
        owned_bytes(g.seg_inv),owned_bytes(g.diag),owned_bytes(g.minv)});
    for(const auto& c:g.cell_code_own)bytes=host_sum({bytes,owned_bytes(c)});
    for(const auto& t:g.tr) {
        bytes=host_sum({bytes,owned_bytes(t.ptr),owned_bytes(t.seg),owned_bytes(t.other),
            owned_bytes(t.mult),owned_bytes(t.item_level),owned_bytes(t.item_begin),
            owned_bytes(t.item_end),owned_bytes(t.item_ptr),owned_bytes(t.item_blocks)});
        for(const auto& c:t.other)bytes=host_sum({bytes,owned_bytes(c)});
    }
    return bytes;
}

// Host phases of the CUDA tier (restated for package 2, 17 November 2026):
//  - geometry and iteration: the CPU bound (cpu_peak) covers the geometry's
//    construction and the workspace Work; the adapter and the operator's fixed
//    state come on top. During the iteration the operator holds no host vector;
//    the registration page-locks p and q INSIDE Work (no new bytes; it is
//    reported as page-locked, Admission::page_locked_iteration);
//  - the operator's set-up (prepare(), before Work exists): the resident geometry,
//    the adapter, the operator's state and the set-up's own host structures
//    (setup_peak: the two page-locked upload chunks, the long-segment lists, B's
//    downloaded ptrs, one effect's unit lists at a time, the long-level lists),
//    all freed when the set-up ends.
// Adding the set-up to the CPU bound would charge mutually exclusive allocations
// twice. The device admission is separate (Admission, reduced_schur_pcg_cuda.hpp).
inline std::size_t host_phase_peak(std::size_t cpu_peak,std::size_t geometry,
    std::size_t setup_peak,std::size_t adapter,std::size_t operator_state) {
    return std::max(host_sum({cpu_peak,adapter,operator_state}),
                    host_sum({geometry,setup_peak,adapter,operator_state}));
}

} // namespace hdfe::detail::reduced_cuda
#endif
