#ifndef XHDFE_TESTS_REDUCED_CUDA_BENCH_LEVEL_ORDER_REFERENCE_HPP
#define XHDFE_TESTS_REDUCED_CUDA_BENCH_LEVEL_ORDER_REFERENCE_HPP
// Host reference for the level-ordered arrays of one remaining effect j
// (package 2, stage 2a, 17 November 2026): the operator cells sorted, stably,
// by their code of j. Entries follow (level, operator-segment order, cell
// order), the order of the transposes that build_geometry makes for its large
// effects (src/reduced_schur_pcg.hpp), and carry the same arrays: ptr (levels
// + 1), seg (the eliminated level of the entry's segment), other[jj] (the code
// of every other remaining effect) and mult (merged geometry only).
//
// Two builders. The team-parallel one is a stable counting sort over `parts`
// contiguous ranges of operator segments balanced by cells; `parts` is fixed by
// the caller (a constant), never by the team: private histograms per part,
// prefixes in (level, part) order, a stable scatter per part. The serial one is
// the same sort with one part. A stable sort is unique, so both, and the CPU
// geometry's own transpose where one exists, must agree integer for integer.
// The entry arrays are not zero-filled: the scatter writes every entry once.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "reduced_schur_pcg.hpp"

namespace xhdfe_bench {

struct Ints {  // an int array that is not value-initialised
    std::unique_ptr<int[]> data;
    std::size_t size = 0;
    void allocate(std::size_t count) {
        data.reset(count ? new int[count] : nullptr);
        size = count;
    }
    int* get() noexcept { return data.get(); }
    const int* get() const noexcept { return data.get(); }
};

inline bool same_ints(const int* a, std::size_t na, const int* b, std::size_t nb) {
    return na == nb && (na == 0 || std::equal(a, a + na, b));
}

struct LevelOrder {
    std::vector<int> ptr;
    Ints seg;
    std::vector<Ints> other;  // [jj]; empty for jj == j
    Ints mult;                // merged geometry only
    Ints cell;                // the source cell of each entry (the identity and order of the cells)
};

namespace level_order_detail {

inline void allocate(const hdfe::reduced_pcg::internal::Geometry& g, int j, std::size_t entries, LevelOrder& out) {
    out.seg.allocate(entries);
    out.other.clear();
    out.other.resize(static_cast<std::size_t>(g.nb));
    for (int jj = 0; jj < g.nb; ++jj)
        if (jj != j) out.other[static_cast<std::size_t>(jj)].allocate(entries);
    out.mult.allocate(g.mult != nullptr ? entries : 0);
    out.cell.allocate(entries);
}

// The stable scatter of one range of operator segments, with its own cursors.
inline void scatter(const hdfe::reduced_pcg::internal::Geometry& g, int j, int s0, int s1, int* cursor, LevelOrder& out) {
    const int* code = g.cc[static_cast<std::size_t>(j)];
    int* seg = out.seg.get();
    int* other[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int jj = 0; jj < g.nb; ++jj)
        if (jj != j) other[jj] = out.other[static_cast<std::size_t>(jj)].get();
    int* mult = out.mult.get();
    int* cell = out.cell.get();
    for (int s = s0; s < s1; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
            const std::size_t pos = static_cast<std::size_t>(cursor[code[c]]++);
            seg[pos] = l;
            for (int jj = 0; jj < g.nb; ++jj)
                if (jj != j) other[jj][pos] = g.cc[static_cast<std::size_t>(jj)][c];
            if (mult != nullptr) mult[pos] = g.mult[c];
            cell[pos] = c;
        }
    }
}

inline int levels_of(const hdfe::reduced_pcg::internal::Geometry& g, int j) {
    return g.offset[static_cast<std::size_t>(j) + 1U] - g.offset[static_cast<std::size_t>(j)];
}

}  // namespace level_order_detail

inline LevelOrder build_level_order_serial(const hdfe::reduced_pcg::internal::Geometry& g, int j) {
    const int mj = level_order_detail::levels_of(g, j);
    const int* code = g.cc[static_cast<std::size_t>(j)];
    LevelOrder out;
    out.ptr.assign(static_cast<std::size_t>(mj) + 1U, 0);
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) ++out.ptr[static_cast<std::size_t>(code[c]) + 1U];
    }
    for (int lev = 0; lev < mj; ++lev) out.ptr[static_cast<std::size_t>(lev) + 1U] += out.ptr[static_cast<std::size_t>(lev)];
    level_order_detail::allocate(g, j, static_cast<std::size_t>(out.ptr[static_cast<std::size_t>(mj)]), out);
    std::vector<int> cursor(out.ptr.begin(), out.ptr.end() - 1);
    level_order_detail::scatter(g, j, 0, g.L_op, cursor.data(), out);
    return out;
}

// `run(units, body)` runs body(unit) for every unit in [0, units) on the caller's team.
template <class Run>
LevelOrder build_level_order(const hdfe::reduced_pcg::internal::Geometry& g, int j, int parts, Run&& run) {
    namespace rpi = hdfe::reduced_pcg::internal;
    const int mj = level_order_detail::levels_of(g, j);
    const int* code = g.cc[static_cast<std::size_t>(j)];
    // Contiguous ranges of operator segments balanced by cells (C_op fits an int).
    std::vector<int> prefix(static_cast<std::size_t>(g.L_op) + 1U, 0);
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        prefix[static_cast<std::size_t>(s) + 1U] = prefix[static_cast<std::size_t>(s)] + (g.cell_ptr[l + 1] - g.cell_ptr[l]);
    }
    const std::vector<int> bounds = rpi::balanced_bounds(prefix.data(), g.L_op, std::max(1, parts));
    const int P = static_cast<int>(bounds.size()) - 1;
    std::vector<int> hist(static_cast<std::size_t>(P) * static_cast<std::size_t>(mj), 0);
    run(P, [&](int p) {
        int* h = hist.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(mj);
        for (int s = bounds[static_cast<std::size_t>(p)]; s < bounds[static_cast<std::size_t>(p) + 1U]; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) ++h[code[c]];
        }
    });
    LevelOrder out;
    out.ptr.assign(static_cast<std::size_t>(mj) + 1U, 0);
    const int unit = 4096;
    const int level_units = (mj + unit - 1) / unit;
    run(level_units, [&](int u) {  // totals per level, over the parts in order
        const int l1 = std::min(mj, (u + 1) * unit);
        for (int lev = u * unit; lev < l1; ++lev) {
            int total = 0;
            for (int p = 0; p < P; ++p) total += hist[static_cast<std::size_t>(p) * static_cast<std::size_t>(mj) + static_cast<std::size_t>(lev)];
            out.ptr[static_cast<std::size_t>(lev) + 1U] = total;
        }
    });
    for (int lev = 0; lev < mj; ++lev) out.ptr[static_cast<std::size_t>(lev) + 1U] += out.ptr[static_cast<std::size_t>(lev)];
    run(level_units, [&](int u) {  // cursor of (part, level): the level's start, then the parts before it
        const int l1 = std::min(mj, (u + 1) * unit);
        for (int lev = u * unit; lev < l1; ++lev) {
            int position = out.ptr[static_cast<std::size_t>(lev)];
            for (int p = 0; p < P; ++p) {
                int& slot = hist[static_cast<std::size_t>(p) * static_cast<std::size_t>(mj) + static_cast<std::size_t>(lev)];
                const int count = slot;
                slot = position;
                position += count;
            }
        }
    });
    level_order_detail::allocate(g, j, static_cast<std::size_t>(out.ptr[static_cast<std::size_t>(mj)]), out);
    run(P, [&](int p) {
        level_order_detail::scatter(g, j, bounds[static_cast<std::size_t>(p)], bounds[static_cast<std::size_t>(p) + 1U],
                                    hist.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(mj), out);
    });
    return out;
}

inline bool same_level_order(const LevelOrder& a, const LevelOrder& b) {
    if (a.ptr != b.ptr || !same_ints(a.seg.get(), a.seg.size, b.seg.get(), b.seg.size)) return false;
    if (a.other.size() != b.other.size()) return false;
    for (std::size_t jj = 0; jj < a.other.size(); ++jj)
        if (!same_ints(a.other[jj].get(), a.other[jj].size, b.other[jj].get(), b.other[jj].size)) return false;
    return same_ints(a.mult.get(), a.mult.size, b.mult.get(), b.mult.size) &&
           same_ints(a.cell.get(), a.cell.size, b.cell.get(), b.cell.size);
}

// Against the CPU geometry's transpose of a large effect (build_geometry).
inline bool same_as_transpose(const LevelOrder& a, const hdfe::reduced_pcg::internal::Geometry& g, int j) {
    const auto& t = g.tr[static_cast<std::size_t>(j)];
    const int mj = level_order_detail::levels_of(g, j);
    if (a.ptr.size() != static_cast<std::size_t>(mj) + 1U || t.ptr.size() != a.ptr.size()) return false;
    if (!std::equal(a.ptr.begin(), a.ptr.end(), t.ptr.data())) return false;
    const std::size_t entries = static_cast<std::size_t>(a.ptr.back());
    if (!same_ints(a.seg.get(), a.seg.size, t.seg.data(), entries)) return false;
    for (int jj = 0; jj < g.nb; ++jj) {
        if (jj == j) continue;
        const auto& mine = a.other[static_cast<std::size_t>(jj)];
        if (!same_ints(mine.get(), mine.size, t.other[static_cast<std::size_t>(jj)].data(), entries)) return false;
    }
    if (t.has_mult != (a.mult.size != 0)) return false;
    return !t.has_mult || same_ints(a.mult.get(), a.mult.size, t.mult.data(), entries);
}

}  // namespace xhdfe_bench
#endif
