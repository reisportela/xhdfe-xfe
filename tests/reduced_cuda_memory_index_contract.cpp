// The memory and index contract of the CUDA reduced tier (rewritten for package 2,
// Amendment 2, C6; 17 November 2026), on the host only: no device, no data. It runs the
// same plan, unit and index functions the operator runs (src/reduced_schur_pcg_cuda.hpp)
// against the reviewer's table (round 27):
//  shapes and counts  zero, and counts near INT_MAX and SIZE_MAX; every byte count checked;
//  ownership          every operator segment exactly once (K1 or K1L, never both), every
//                     item and every level entry exactly once, PB slots stable under the
//                     launch order; K1's segment order (17 November) a stable sort by
//                     length whatever the order of its parts;
//  addresses          the K1 placement, private offsets, PA/PB/PL/T rows, host and device
//                     pitches, the upload chunks and their byte ranges, the registered range;
//  boundaries         16/17 (K1's register cache), 255/256/257, tiles of 4096, incomplete last
//                     blocks, trees whose leaves are not a power of two, the stable warp ranks;
//  resources          the admission per phase, its bounds for the built effects, the host
//                     phase peak, and a refusal before anything is allocated.
// The device behaviour (initialisation before every read, C_op = 0, padding, failures in the
// middle of the set-up and of the registration, nothing in flight when memory is freed) is
// tests/reduced_cuda_operator_smoke.cpp.
#include "reduced_schur_pcg_cuda_memory.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rc = hdfe::detail::reduced_cuda;

namespace {

int checks = 0;
void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}
template <class F>
bool refused_as(F&& f, rc::ErrorKind kind) {
    try {
        f();
    } catch (const rc::Error& e) {
        return e.kind() == kind;
    }
    return false;
}

}  // namespace

int main() {
    // --- the host phase peak, and the geometry's resident bytes (kept from 15 November) ---
    expect(rc::host_phase_peak(800, 130, 370, 596, 4) == 1400, "phase peak");
    expect(rc::host_phase_peak(400, 300, 700, 100, 4) == 1104, "set-up peak");
    expect(refused_as([] { rc::host_sum({std::numeric_limits<std::size_t>::max(), 1}); }, rc::ErrorKind::Memory),
           "host byte overflow must decline");
    {
        hdfe::reduced_pcg::internal::Geometry g;
        const auto empty = rc::geometry_resident_bytes(g);
        g.row_ptr.allocate(10);
        g.diag.allocate(11);
        g.offset.reserve(12);
        expect(rc::geometry_resident_bytes(g) == empty + 40 + 88 + 12 * sizeof(int), "resident ownership");
        int borrowed = 0;
        g.cell_ptr = &borrowed;
        g.mult = &borrowed;
        expect(rc::geometry_resident_bytes(g) == empty + 40 + 88 + 12 * sizeof(int), "views counted twice");
    }

    // --- checked byte counts near SIZE_MAX ---
    const std::size_t big = std::numeric_limits<std::size_t>::max();
    expect(refused_as([&] { rc::checked_add(big, 1); }, rc::ErrorKind::Memory), "checked_add overflow");
    expect(refused_as([&] { rc::checked_mul(big / 2 + 1, 2); }, rc::ErrorKind::Memory), "checked_mul overflow");
    expect(rc::checked_mul(big / 2, 2) == big - 1 && rc::checked_mul(0, big) == 0, "checked_mul at the edge");
    expect(refused_as([&] { rc::align_up(big); }, rc::ErrorKind::Memory), "align_up overflow");
    expect(rc::align_up(0) == 0 && rc::align_up(1) == 256 && rc::align_up(256) == 256 && rc::align_up(257) == 512, "align_up");

    // --- the thread organisation at every real width: groups never straddle a warp ---
    for (int w = 1; w <= 8; ++w) {
        const rc::Org o = rc::org_for(w);
        expect(o.ws == w && o.gsz == w && o.gsz * o.gpw <= 32 && o.gsz * (o.gpw + 1) > 32, "groups inside a warp");
        expect(o.groups == (rc::kThreads / 32) * o.gpw, "groups per block");
        std::vector<int> seen(static_cast<std::size_t>(o.groups) * w, 0);
        for (int t = 0; t < rc::kThreads; ++t) {
            int g = 0, lane = 0;
            bool in = false;
            rc::place_of(t, o.gsz, o.gpw, g, lane, in);
            if (!in) continue;
            expect(g >= 0 && g < o.groups && lane >= 0 && lane < w && g / o.gpw == t / 32, "placement inside the warp");
            ++seen[static_cast<std::size_t>(g) * w + lane];
        }
        for (int s : seen) expect(s == 1, "every (group, lane) has one thread");
    }

    // --- K1: every operator segment exactly once over (block, group, i); incomplete last blocks ---
    for (int w = 1; w <= 8; ++w) {
        const int G = rc::org_for(w).groups;
        for (long long L_op : {0LL, 1LL, 511LL, 512LL, 513LL, 4095LL, 4096LL, 4097LL, 100003LL}) {
            const long long blocks = rc::k1_blocks(L_op, w);
            std::vector<int> hits(static_cast<std::size_t>(L_op), 0);
            for (long long b = 0; b < blocks; ++b)
                for (int g = 0; g < G; ++g)
                    for (int i = 0; i < rc::kSegPerGroup; ++i) {
                        const long long s = rc::k1_segment(b, g, i, G);
                        if (s >= L_op) break;  // the kernel breaks too (s grows with i)
                        ++hits[static_cast<std::size_t>(s)];
                    }
            for (int h : hits) expect(h == 1, "K1 ownership");
            expect(blocks == 0 || rc::k1_segment(blocks - 1, 0, 0, G) < L_op, "no empty last block");
        }
        // Near INT_MAX: the grid stays an int and the 64-bit segment index does not wrap.
        const long long blocks = rc::k1_blocks(INT_MAX, w);
        expect(blocks <= INT_MAX && rc::k1_segment(blocks - 1, G - 1, rc::kSegPerGroup - 1, G) >= INT_MAX - 1LL,
               "K1 near INT_MAX");
        expect(rc::k1_segment(blocks, 0, 0, G) >= INT_MAX, "K1 past the last block");
    }

    // --- K1's shared accumulators: groups x S x width doubles, the largest over the widths ---
    expect(rc::k1_smem_max(48, 8) == 98304 && rc::k1_smem_max(0, 8) == 0, "K1 shared memory");
    for (int w = 1; w <= 8; ++w) expect(rc::k1_smem_bytes(26, w) <= rc::k1_smem_max(26, 8), "K1 shared memory by width");

    // --- 255/256/257: K1 and K1L partition the operator segments by the same test ---
    {
        // A view of seven segments of 1, 16, 17, 255, 256, 257 and 1000 cells, one effect of 4 levels.
        const int sizes[] = {1, 16, 17, 255, 256, 257, 1000};
        std::vector<int> cell_ptr = {0}, op_seg, code;
        for (int s = 0; s < 7; ++s) {
            cell_ptr.push_back(cell_ptr.back() + sizes[s]);
            for (int c = 0; c < sizes[s]; ++c) code.push_back(c % 4);
            if (sizes[s] > 1) op_seg.push_back(s);
        }
        const std::vector<double> inv(7, 0.5);
        rc::GeometryView v;
        v.nb = 1; v.m = 4; v.L = 7; v.L_op = static_cast<int>(op_seg.size());
        v.C = cell_ptr.back(); v.C_op = v.C - 1;
        v.cell_ptr = cell_ptr.data(); v.op_seg = op_seg.data(); v.seg_inv = inv.data(); v.cc[0] = code.data();
        v.dims[0].levels = 4;
        rc::Options o;
        o.lanes = 8;
        const rc::Plan p = rc::plan_operator(v, o, 1 << 20);
        expect(p.cls[0] == 'P' && p.S == 4, "a 4-level effect is private");
        expect(p.n_long == 2 && p.pl_items == 2 + 4, "long segments: 257 and 1000 cells, items of 256");
        long long k1 = 0, k1l = 0;
        for (int s : op_seg) (sizes[s] > o.short_seg ? k1l : k1) += 1;
        expect(k1 == 4 && k1l == 2, "K1 takes 16, 17, 255, 256; K1L takes 257, 1000");
        // 16/17: K1's register cache takes a segment of at most kK1Cache cells.
        expect(rc::kK1Cache == 16 && sizes[1] <= rc::kK1Cache && sizes[2] > rc::kK1Cache, "16/17 boundary");
        // K1L: items of each long segment, each owned by one group, slots of PL disjoint.
        long long base = 0;
        for (int s : op_seg) {
            if (sizes[s] <= o.short_seg) continue;
            const int items = (sizes[s] + o.item - 1) / o.item;
            const int G = rc::org_for(5).groups;
            std::vector<int> owner(static_cast<std::size_t>(items), 0);
            for (int g = 0; g < G; ++g)
                for (int it = g; it < items; it += G) ++owner[static_cast<std::size_t>(it)];
            for (int x : owner) expect(x == 1, "K1L item ownership");
            base += items;
        }
        expect(base == p.pl_items, "PL slots");
        // A refusal before anything: the plan's validation.
        std::vector<int> bad_seg = op_seg;
        bad_seg[0] = 7;
        v.op_seg = bad_seg.data();
        expect(refused_as([&] { rc::plan_operator(v, o, 1 << 20); }, rc::ErrorKind::Fatal), "op_seg out of range refused");
        v.op_seg = op_seg.data();
        v.C_op = v.C;
        expect(refused_as([&] { rc::plan_operator(v, o, 1 << 20); }, rc::ErrorKind::Fatal), "C_op inconsistent refused");
        v.C_op = v.C - 1;
        o.lanes = 3;
        expect(refused_as([&] { rc::plan_operator(v, o, 1 << 20); }, rc::ErrorKind::Fatal), "lanes 3 refused");
        o.lanes = 8;
        // The shared-memory limit moves a private effect to the built class.
        const rc::Plan q = rc::plan_operator(v, o, rc::k1_smem_max(4, 8) - 1);
        expect(q.cls[0] == 'B' && q.S == 0, "private only while it fits in shared memory");
        rc::Options none = o;
        none.private_max = 0;
        expect(rc::plan_operator(v, none, 1 << 20).cls[0] == 'B', "private_max 0: built");
    }

    // --- K1's segment order (17 November): a stable sort of the operator segments by
    //     decreasing cells (above short_seg one key), the result of std::stable_sort
    //     whatever the order in which the parts run; L_op of 0, 1, below and above 64 ---
    {
        std::mt19937_64 rng(20261117);
        const auto serial = [](int parts, const std::function<void(int)>& body) {
            for (int part = 0; part < parts; ++part) body(part);
        };
        const auto reverse = [](int parts, const std::function<void(int)>& body) {
            for (int part = parts - 1; part >= 0; --part) body(part);
        };
        for (int L_op : {0, 1, 2, 63, 64, 65, 1000, 100003}) {
            for (int short_seg : {2, 16, 256}) {
                for (int law = 0; law < 3; ++law) {  // lengths: mixed across short_seg; one length; two lengths
                    const int L = L_op + 7;          // segments outside op_seg too
                    std::vector<int> cell_ptr(1, 0);
                    std::uniform_int_distribution<int> mixed(0, short_seg + 40), two(0, 1);
                    for (int l = 0; l < L; ++l) {
                        const int n = law == 0 ? mixed(rng) : law == 1 ? 8 : (two(rng) ? short_seg : short_seg + 1);
                        cell_ptr.push_back(cell_ptr.back() + n);
                    }
                    std::vector<int> op_seg(static_cast<std::size_t>(L));
                    std::iota(op_seg.begin(), op_seg.end(), 0);
                    std::shuffle(op_seg.begin(), op_seg.end(), rng);
                    op_seg.resize(static_cast<std::size_t>(L_op));
                    rc::GeometryView v;
                    v.L = L; v.L_op = L_op; v.cell_ptr = cell_ptr.data(); v.op_seg = op_seg.data();
                    const int top = short_seg + 1;
                    std::vector<int> want(op_seg);
                    std::stable_sort(want.begin(), want.end(), [&](int x, int y) {
                        return std::min(cell_ptr[x + 1] - cell_ptr[x], top) > std::min(cell_ptr[y + 1] - cell_ptr[y], top);
                    });
                    std::vector<int> got_a(static_cast<std::size_t>(L_op), -1), got_b(static_cast<std::size_t>(L_op), -1);
                    rc::k1_segment_order(v, short_seg, got_a.data(), serial);
                    rc::k1_segment_order(v, short_seg, got_b.data(), reverse);
                    expect(got_a == want && got_b == want, "K1's order: a stable sort by decreasing cells, whatever the parts' order");
                    if (law == 1) expect(got_a == op_seg, "K1's order: one length keeps the geometry's order");
                }
            }
        }
        expect(rc::k1_order_parts(0) == 1 && rc::k1_order_parts(1) == 1 && rc::k1_order_parts(63) == 63 &&
                   rc::k1_order_parts(64) == 64 && rc::k1_order_parts(INT_MAX) == 64,
               "K1's order: parts fixed by L_op");
        expect(rc::k1_order_count_bytes(256, 100003) == 64U * 258U * sizeof(long long) &&
                   rc::k1_order_count_bytes(256, 0) == 258U * sizeof(long long),
               "K1's order: count bytes");
        // The parts' bounds near INT_MAX: contiguous, covering, 64-bit products.
        const long long Lmax = INT_MAX;
        long long covered = 0;
        for (int part = 0; part < 64; ++part) {
            const long long s0 = Lmax * part / 64, s1 = Lmax * (part + 1) / 64;
            expect(s0 == covered && s1 >= s0 && s1 <= Lmax, "K1's order: parts contiguous near INT_MAX");
            covered = s1;
        }
        expect(covered == Lmax, "K1's order: parts cover the operator segments near INT_MAX");
    }

    // --- an empty geometry (no segment, no cell): arrays without elements may be null pointers ---
    // (the rational oracle's zero_operator fixtures; refused as invalid before 17 November, 04:05)
    {
        const int cell_ptr0[] = {0}, ptr0[] = {0, 0, 0, 0}, ptr1[] = {0, 0, 0};
        rc::GeometryView v;
        v.nb = 2; v.m = 5; v.L = 0; v.L_op = 0; v.C = 0; v.C_op = 0; v.cell_ptr = cell_ptr0;
        v.dims[0].levels = 3; v.dims[1].levels = 2; v.dims[1].offset = 3;
        rc::Options o;
        o.lanes = 1;
        expect(rc::plan_operator(v, o, 1 << 20).S == 5, "an empty geometry is planned (private)");
        rc::Options none = o;
        none.private_max = 0;
        const rc::Plan b = rc::plan_operator(v, none, 1 << 20);
        expect(b.cls[0] == 'B' && b.cls[1] == 'B' && b.units[0].units == 3 && b.units[1].units == 2, "an empty geometry is planned (built)");
        v.dims[0].transposed = true; v.dims[0].tr_ptr = ptr0;
        v.dims[1].transposed = true; v.dims[1].tr_ptr = ptr1;
        const rc::Plan t = rc::plan_operator(v, none, 1 << 20);
        expect(t.cls[0] == 'T' && t.cls[1] == 'T' && t.units[0].units == 3 && t.units[1].units == 2 && t.entries[0] == 0,
               "empty transposes are planned");
        // One single-cell segment (no operator segment): a cell exists, so its codes must.
        const int cell_ptr1[] = {0, 1};
        v.L = 1; v.C = 1; v.cell_ptr = cell_ptr1;
        v.dims[0].transposed = v.dims[1].transposed = false;
        expect(refused_as([&] { rc::plan_operator(v, o, 1 << 20); }, rc::ErrorKind::Fatal), "a cell without codes refused");
        const int codes0[] = {0}, codes1[] = {1};
        v.cc[0] = codes0; v.cc[1] = codes1;
        expect(rc::plan_operator(v, o, 1 << 20).S == 5, "the same with its codes is planned");
    }

    // --- units: levels of 0, 1, 255, 256, 257, 511, 512, 513, 1024 entries; ownership, slots, order ---
    {
        const int sizes[] = {0, 1, 255, 256, 257, 511, 512, 513, 1024, 0, 3};
        const int levels = 11;
        std::vector<int> ptr(levels + 1, 0);
        for (int l = 0; l < levels; ++l) ptr[l + 1] = ptr[l] + sizes[l];
        const rc::UnitCount c = rc::count_units(ptr.data(), levels, 256, 256);
        // Long: 257 (2), 511 (2), 512 (2), 513 (3), 1024 (4) -> 13 slots; 6 short levels.
        expect(c.long_levels == 5 && c.slots == 13 && c.units == 6 + 13, "unit counts");
        const rc::UnitCount bnd = rc::bound_units(levels, ptr[levels], 256, 256);
        expect(bnd.units >= c.units && bnd.slots >= c.slots && bnd.long_levels >= c.long_levels, "the plan's bounds hold");
        rc::UnitLists u;
        rc::LongLevels longs;
        longs.size_for(static_cast<std::size_t>(c.long_levels));
        const long long slot_base = 7;
        rc::build_units(ptr.data(), levels, 100, 256, 256, slot_base, c, u, longs);
        std::vector<int> owner(static_cast<std::size_t>(ptr[levels]), 0);
        std::vector<int> slot_seen(static_cast<std::size_t>(c.slots), 0);
        for (std::size_t i = 0; i < u.level.size(); ++i) {
            const int lev = u.level[i];
            expect(u.begin[i] >= ptr[lev] && u.end[i] <= ptr[lev + 1] && u.begin[i] <= u.end[i], "unit inside its level");
            expect(u.end[i] - u.begin[i] <= 256, "unit length");
            for (int a = u.begin[i]; a < u.end[i]; ++a) ++owner[static_cast<std::size_t>(a)];
            if (sizes[lev] <= 256) {
                expect(u.slot[i] == -1, "a short level writes its row");
            } else {
                // The logical slot: the level's base plus the item index.
                expect(u.slot[i] >= slot_base && u.slot[i] < slot_base + c.slots, "slot range");
                ++slot_seen[static_cast<std::size_t>(u.slot[i] - slot_base)];
            }
        }
        for (int x : owner) expect(x == 1, "every level entry exactly once");
        for (int x : slot_seen) expect(x == 1, "every slot exactly once");
        // Long-level lists: output row, first slot, items; consistent with the units.
        long long total = 0;
        for (std::size_t l = 0; l < longs.out.size(); ++l) {
            const int lev = longs.out[l] - 100;
            expect(lev >= 0 && lev < levels && sizes[lev] > 256, "long level");
            expect(longs.items[l] == (sizes[lev] + 255) / 256, "items of a long level");
            for (std::size_t i = 0; i < u.level.size(); ++i)
                if (u.level[i] == lev)
                    expect(u.slot[i] == longs.base[l] + (u.begin[i] - ptr[lev]) / 256, "slot = base + item");
            total += longs.items[l];
        }
        expect(total == c.slots && longs.next == longs.out.size(), "long-level totals");
        // The launch order: a permutation, decreasing length, stable among equal lengths.
        std::vector<int> seen(u.order.size(), 0);
        for (std::size_t i = 0; i < u.order.size(); ++i) {
            ++seen[static_cast<std::size_t>(u.order[i])];
            if (i) {
                const int a = u.order[i - 1], b = u.order[i];
                const int la = u.end[a] - u.begin[a], lb = u.end[b] - u.begin[b];
                expect(la > lb || (la == lb && a < b), "order: decreasing length, stable");
            }
        }
        for (int x : seen) expect(x == 1, "order is a permutation");
        // The longest units first (a short level of 256 entries ties with full items).
        const int first = u.order.front();
        expect(u.end[static_cast<std::size_t>(first)] - u.begin[static_cast<std::size_t>(first)] == 256, "the longest units first");
        expect(refused_as([&] { std::vector<int> bad = ptr; bad[3] = bad[2] - 1; rc::count_units(bad.data(), levels, 256, 256); },
                          rc::ErrorKind::Fatal), "decreasing ptr refused");
        expect(refused_as([&] { rc::UnitCount wrong = c; wrong.slots += 1; rc::UnitLists x; rc::LongLevels y;
                                y.size_for(static_cast<std::size_t>(c.long_levels));
                                rc::build_units(ptr.data(), levels, 0, 256, 256, 0, wrong, x, y); }, rc::ErrorKind::Fatal),
               "a unit count that does not match refused");
        // Random level sizes: the bounds of the plan always hold.
        std::mt19937 gen(20261117u);
        for (int trial = 0; trial < 200; ++trial) {
            const int L = 1 + static_cast<int>(gen() % 50);
            std::vector<int> pp(static_cast<std::size_t>(L) + 1, 0);
            for (int l = 0; l < L; ++l) pp[l + 1] = pp[l] + static_cast<int>(gen() % 700);
            for (int sl : {1, 2, 255, 256, 257})
                for (int it : {1, 2, 256}) {
                    const rc::UnitCount cc = rc::count_units(pp.data(), L, sl, it);
                    const rc::UnitCount bb = rc::bound_units(L, pp[L], sl, it);
                    expect(bb.units >= cc.units && bb.slots >= cc.slots && bb.long_levels >= cc.long_levels, "random bounds");
                }
        }
    }

    // --- trees whose leaves are not a power of two: each partial added exactly once ---
    for (int items = 1; items <= 1000; ++items) {  // K1L's in-place pairwise tree on item index
        std::vector<long long> v(static_cast<std::size_t>(items));
        std::iota(v.begin(), v.end(), 1LL);
        for (int stride = 1; stride < items; stride <<= 1)
            for (int i = 0; i + stride < items; i += 2 * stride) v[static_cast<std::size_t>(i)] += v[static_cast<std::size_t>(i + stride)];
        expect(v[0] == static_cast<long long>(items) * (items + 1) / 2, "K1L tree");
    }
    for (long long slots : {0LL, 1LL, 255LL, 256LL, 257LL, 1000LL, 4097LL}) {  // K3: strided, then a halving tree of 256
        std::vector<long long> buf(rc::kThreads, 0);
        for (int t = 0; t < rc::kThreads; ++t)
            for (long long s = t; s < slots; s += rc::kThreads) buf[static_cast<std::size_t>(t)] += s + 1;
        for (int st = rc::kThreads / 2; st > 0; st >>= 1)
            for (int t = 0; t < st; ++t) buf[static_cast<std::size_t>(t)] += buf[static_cast<std::size_t>(t + st)];
        expect(buf[0] == slots * (slots + 1) / 2, "K3 tree");
    }

    // --- tiles of 4096 and the stable warp ranks of the scatter ---
    for (long long C_op : {0LL, 1LL, 31LL, 32LL, 33LL, 4095LL, 4096LL, 4097LL, 8191LL, 8192LL, 8193LL}) {
        const long long tiles = (C_op + rc::kTile - 1) / rc::kTile;
        std::vector<int> hit(static_cast<std::size_t>(C_op), 0);
        for (long long t = 0; t < tiles; ++t) {
            const long long q0 = t * rc::kTile, q1 = std::min(C_op, q0 + rc::kTile);
            expect(q1 > q0, "no empty tile");
            for (long long qb = q0; qb < q1; qb += 32)
                for (int lane = 0; lane < 32; ++lane)
                    if (qb + lane < q1) ++hit[static_cast<std::size_t>(qb + lane)];
        }
        for (int h : hit) expect(h == 1, "every position in one tile, one warp step");
    }
    {
        // The scatter of one tile as the warp does it (ranks from the peers of lower
        // lanes; the highest peer advances the cursor), against a stable counting sort.
        std::mt19937 gen(4096u);
        for (int trial = 0; trial < 100; ++trial) {
            const int levels = 1 + static_cast<int>(gen() % 40), n = 1 + static_cast<int>(gen() % 300);
            std::vector<int> code(static_cast<std::size_t>(n));
            for (int& x : code) x = static_cast<int>(gen() % static_cast<unsigned>(levels));
            std::vector<int> start(static_cast<std::size_t>(levels) + 1, 0);
            for (int x : code) ++start[static_cast<std::size_t>(x) + 1];
            for (int l = 0; l < levels; ++l) start[static_cast<std::size_t>(l) + 1] += start[static_cast<std::size_t>(l)];
            std::vector<int> cursor(start.begin(), start.end() - 1), expected(static_cast<std::size_t>(n)), got(static_cast<std::size_t>(n), -1);
            {
                std::vector<int> cur = cursor;
                for (int q = 0; q < n; ++q) expected[static_cast<std::size_t>(cur[static_cast<std::size_t>(code[static_cast<std::size_t>(q)])]++)] = q;
            }
            for (int qb = 0; qb < n; qb += 32) {
                int pos[32];
                for (int lane = 0; lane < 32 && qb + lane < n; ++lane) {
                    const int me = code[static_cast<std::size_t>(qb + lane)];
                    int lower = 0;
                    for (int o = 0; o < lane; ++o) lower += code[static_cast<std::size_t>(qb + o)] == me;
                    pos[lane] = cursor[static_cast<std::size_t>(me)] + lower;
                }
                for (int lane = 0; lane < 32 && qb + lane < n; ++lane) {  // the highest peer advances
                    const int me = code[static_cast<std::size_t>(qb + lane)];
                    int peers = 0, highest = lane;
                    for (int o = 0; o < 32 && qb + o < n; ++o)
                        if (code[static_cast<std::size_t>(qb + o)] == me) { ++peers; highest = o; }
                    if (lane == highest) cursor[static_cast<std::size_t>(me)] += peers;
                }
                for (int lane = 0; lane < 32 && qb + lane < n; ++lane) got[static_cast<std::size_t>(pos[lane])] = qb + lane;
            }
            expect(got == expected, "stable warp ranks");
        }
    }

    // --- the upload chunks: their byte ranges, the last chunk, the copy ranges of 64 KiB ---
    expect(rc::staging_parts(0) == 1 && rc::staging_parts(65535) == 1 && rc::staging_parts(65536) == 1 &&
               rc::staging_parts(131072) == 2 && rc::staging_parts(std::size_t{8} << 20) == 128,
           "copy ranges fixed by the data");
    for (std::size_t chunk : {std::size_t{8}, std::size_t{8} << 20}) {
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, chunk - 1, chunk, chunk + 1, 3 * chunk + 5}) {
            std::size_t covered = 0, n = 0;
            for (std::size_t off = 0; off < bytes; off += chunk, ++n) {
                const std::size_t len = std::min(chunk, bytes - off);
                expect(len > 0 && len <= chunk && off + len <= bytes, "chunk range");
                std::size_t parts = 0;
                const int P = rc::staging_parts(len);
                for (int part = 0; part < P; ++part) {
                    const std::size_t a0 = len * static_cast<std::size_t>(part) / P, a1 = len * static_cast<std::size_t>(part + 1) / P;
                    expect(a0 <= a1 && a1 <= len, "copy range");
                    parts += a1 - a0;
                }
                expect(parts == len, "copy ranges partition the chunk");
                covered += len;
            }
            expect(covered == bytes && n == (bytes + chunk - 1) / chunk, "chunks cover the upload");
        }
    }

    // --- pitches and rows: host rows of W, device rows of the width; the registered range ---
    for (long long m : {1LL, 610026LL, static_cast<long long>(INT_MAX)}) {
        for (int W : {1, 2, 4, 8})
            for (int w = 1; w <= W; ++w) {
                const std::size_t host_row_end = (static_cast<std::size_t>(m) - 1) * W + (W - 1);
                const std::size_t device_row_end = (static_cast<std::size_t>(m) - 1) * w + (w - 1);
                expect(host_row_end < static_cast<std::size_t>(m) * W && device_row_end < static_cast<std::size_t>(m) * W,
                       "rows inside buffers of m x W");
                // pack: e < m*w reads row e/w, lane e%w of the stride-W rows.
                const long long e = m * w - 1, r = e / w;
                expect(r == m - 1 && r * W + (e - r * w) < m * W, "pack address");
                // p and q: the third and fourth slices of an arena of W (4m + L), one range of 2 m W.
                const std::size_t pq = rc::checked_mul(rc::checked_mul(2U, static_cast<std::size_t>(m)), static_cast<std::size_t>(W));
                expect(rc::checked_mul(pq, sizeof(double)) / sizeof(double) == pq, "registered range bytes");
            }
    }

    // --- the admission: per phase, the bounds for B, near the int limits, refusals ---
    {
        // A view with only counts (admit reads no array): 2^31 - 1 cells, three effects,
        // levels 1e6 (T), 500 (B, tiles), 15 (P); W = 8.
        rc::GeometryView v;
        v.nb = 3; v.m = 1000515; v.L = 20000000; v.L_op = 20000000; v.C = INT_MAX; v.C_op = INT_MAX;
        v.dims[0].levels = 1000000; v.dims[0].transposed = true;
        v.dims[1].levels = 500; v.dims[2].levels = 15;
        rc::Plan p;
        p.cls[0] = 'T'; p.cls[1] = 'B'; p.cls[2] = 'P';
        p.npriv = 1; p.priv_j[0] = 2; p.S = 15;
        p.k1_blocks_max = rc::k1_blocks(v.L_op, 8); p.pa_slots = p.k1_blocks_max;
        p.entries[0] = v.C_op; p.entries[1] = v.C_op;
        p.units[0] = {1100000, 120000, 9000};
        p.units[1] = rc::bound_units(500, v.C_op, 256, 256);
        rc::CubSizes cub;
        cub.scan_segments = 1 << 20;
        cub.scan_levels[1] = 4096;
        rc::Options o;
        const rc::Admission a = rc::admit(v, o, p, cub, false);
        const std::size_t I = sizeof(int), D = sizeof(double), C = static_cast<std::size_t>(INT_MAX);
        expect(a.device[rc::Admission::Resident] ==
                   (20000001U + 2U * 20000000U) * I + 20000000U * D + 3U * C * I + 1000001U * I + 3U * C * I + 15U * I,
               "resident bytes (with K1's segment order)");
        std::size_t order_device = 0, order_host = 0;
        for (int e = 0; e < a.count; ++e) {
            const std::string name = a.entries[e].name;
            if (name == "k1_segment_order") order_device += a.entries[e].bytes;
            if (name == "k1_segment_order_host") order_host += a.entries[e].bytes;
        }
        expect(order_device == 20000000U * I && order_host == 20000000U * I + 64U * 258U * sizeof(long long),
               "K1's segment order: L_op ints on the device; on the host with the counts of 64 parts");
        expect(a.device[rc::Admission::Product] ==
                   4U * 1000515U * 8U * D + 20000000U * 8U * D + static_cast<std::size_t>(p.pa_slots) * 15U * 8U * D,
               "per-product bytes");
        expect(a.device[rc::Admission::Build] == 501U * I + 3U * C * I, "built arrays");
        const rc::ScratchLayout s = rc::scratch_layout(v, p, 1, cub);
        expect(s.tiles == (static_cast<long long>(INT_MAX) + rc::kTile - 1) / rc::kTile &&
                   s.bytes >= 2U * 20000000U * I + 2U * C * I + static_cast<std::size_t>(s.tiles) * 500U * I,
               "tile scratch");
        expect(a.device[rc::Admission::Scratch] == s.bytes, "scratch bytes");
        expect(a.device_setup_peak == a.device[0] + a.device[1] + a.device[2] + a.device[3] &&
                   a.device_iteration_peak == a.device[0] + a.device[1] + a.device[2] + a.device[4] &&
                   a.device_peak == std::max(a.device_setup_peak, a.device_iteration_peak),
               "peaks per phase");
        expect(a.page_locked_setup == 2U * o.chunk_bytes && a.page_locked_iteration == 2U * 1000515U * 8U * D, "page-locked");
        // Package 3-lite: four registered slices (x, r, p, q) when the replacement runs on the device.
        rc::Options o4 = o;
        o4.registered_slices = 4;
        const rc::Admission a4 = rc::admit(v, o4, p, cub, false);
        expect(a4.page_locked_iteration == 4U * 1000515U * 8U * D && a4.host_setup == a.host_setup &&
                   a4.device_peak == a.device_peak,
               "page-locked: four slices, locked not allocated (no host or device byte more)");
        // The radix alternative: its own temporary, as CUB reports it, inside the scratch.
        p.radix[1] = true;
        cub.radix[1] = 12345;
        const rc::ScratchLayout r = rc::scratch_layout(v, p, 1, cub);
        expect(r.vals >= r.keys + 2U * C * I && r.radix >= r.vals + 2U * C * I && r.bytes >= r.radix + 12345 && r.keys >= r.scan &&
                   r.hist == 0 && r.totals == 0,
               "radix scratch: double buffers of keys and values, and CUB's own size");
        // Counts beyond SIZE_MAX are refused before anything is allocated.
        p.pa_slots = std::numeric_limits<long long>::max() / 2;
        expect(refused_as([&] { rc::admit(v, o, p, cub, false); }, rc::ErrorKind::Memory), "admission overflow refused");
    }

    std::cout << "CUDA_MEMORY_INDEX_CONTRACT_PASS " << checks << " checks\n";
}
