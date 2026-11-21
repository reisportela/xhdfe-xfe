// The rational operator oracle of 15 November against the integrated CUDA operator of the
// reduced tier (package 2, Amendment 2, C6; 17 November 2026). Linked with the library's
// operator. It reads the oracle's fixtures (synthetic layouts and operands; their exact
// results and the operation-count bounds, fixed before any output was seen, stay in the
// fixtures' .json files and are graded by tests/reduced_cuda_operator_oracle.py), rebuilds
// from each fixture the view of the CPU geometry the operator reads (the fixture's segments
// are the operator segments, so L = L_op and op_seg is the identity; the segment-order
// codes, multiplicities and 1/rows as given), and runs it in several configurations:
//   P  every effect private (K1's accumulators);
//   B  no private effect, the level arrays built on the device (tiles; radix when forced);
//   T  no private effect, host transposes (K2 over the CPU's arrays);
//   *_items_of_2  short segments, short levels and items of 2 (K1L, its tree, K2's item
//                 partials and K3's long-level tree).
// Every configuration at the fixture's lane count and, for 8-lane fixtures, at the real
// widths 3, 5, 6 and 7 (the operand's padding zeroed, as the controller does; the padding of
// q must come back +0.0), through ONE registration of the adjacent p and q, as solve() does.
// Negative controls: a one-byte budget is a Memory decline; a width of lanes + 1 is Fatal.
// One JSON line per (case, configuration, width).
//   usage: reduced_cuda_operator_oracle <fixture .txt>...
#include "reduced_schur_pcg_cuda.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu = hdfe::detail::reduced_cuda;

namespace {

template <class T>
void read_vector(std::istream& in, std::vector<T>& out) {
    std::size_t n = 0;
    in >> n;
    out.resize(n);
    for (auto& x : out) in >> x;
    if (!in) throw std::runtime_error("invalid synthetic input");
}

struct Fixture {
    int dims = 0, levels = 0, segments = 0, cells = 0, lanes = 0;
    std::vector<int> offsets, segment_ptr, cell_segment, codes, mult, csr_offsets, csr_ptr, csr_cell, a_seg, a_begin, a_end, a_ptr,
        b_level, b_begin, b_end, b_ptr;
    std::vector<double> segment_inverse, x;
};

struct Transposes {  // a host stable sort by code, per effect, in the CPU transpose's form
    std::vector<int> ptr[4], seg[4], mult[4];
    std::vector<int> other[4][4];
};

gpu::GeometryView view(const Fixture& f, std::vector<int>& identity, const Transposes* tr) {
    gpu::GeometryView v;
    v.nb = f.dims; v.m = f.levels; v.L = f.segments; v.L_op = f.segments; v.C = f.cells; v.C_op = f.cells;
    v.cell_ptr = f.segment_ptr.data();
    identity.resize(static_cast<std::size_t>(f.segments));
    std::iota(identity.begin(), identity.end(), 0);
    v.op_seg = identity.data();
    v.seg_inv = f.segment_inverse.data();
    v.mult = f.mult.empty() ? nullptr : f.mult.data();
    for (int j = 0; j < f.dims; ++j) {
        v.cc[j] = f.codes.data() + static_cast<std::size_t>(j) * f.cells;
        v.dims[j].levels = f.offsets[j + 1] - f.offsets[j];
        v.dims[j].offset = f.offsets[j];
        if (tr) {
            v.dims[j].transposed = true;
            v.dims[j].tr_ptr = tr->ptr[j].data();
            v.dims[j].tr_seg = tr->seg[j].data();
            for (int jj = 0; jj < f.dims; ++jj) v.dims[j].tr_other[jj] = jj == j ? nullptr : tr->other[j][jj].data();
            v.dims[j].has_mult = !f.mult.empty();
            v.dims[j].tr_mult = f.mult.empty() ? nullptr : tr->mult[j].data();
        }
    }
    return v;
}

Transposes transposes(const Fixture& f) {
    Transposes t;
    for (int j = 0; j < f.dims; ++j) {
        const int mj = f.offsets[j + 1] - f.offsets[j];
        const int* code = f.codes.data() + static_cast<std::size_t>(j) * f.cells;
        t.ptr[j].assign(static_cast<std::size_t>(mj) + 1, 0);
        for (int c = 0; c < f.cells; ++c) ++t.ptr[j][static_cast<std::size_t>(code[c]) + 1];
        for (int l = 0; l < mj; ++l) t.ptr[j][static_cast<std::size_t>(l) + 1] += t.ptr[j][static_cast<std::size_t>(l)];
        std::vector<int> cur(t.ptr[j].begin(), t.ptr[j].end() - 1);
        t.seg[j].resize(static_cast<std::size_t>(f.cells));
        t.mult[j].resize(f.mult.empty() ? 0 : static_cast<std::size_t>(f.cells));
        for (int jj = 0; jj < f.dims; ++jj)
            if (jj != j) t.other[j][jj].resize(static_cast<std::size_t>(f.cells));
        for (int s = 0; s < f.segments; ++s)
            for (int c = f.segment_ptr[s]; c < f.segment_ptr[s + 1]; ++c) {
                const int pos = cur[static_cast<std::size_t>(code[c])]++;
                t.seg[j][static_cast<std::size_t>(pos)] = s;
                for (int jj = 0; jj < f.dims; ++jj)
                    if (jj != j) t.other[j][jj][static_cast<std::size_t>(pos)] = f.codes[static_cast<std::size_t>(jj) * f.cells + c];
                if (!f.mult.empty()) t.mult[j][static_cast<std::size_t>(pos)] = f.mult[static_cast<std::size_t>(c)];
            }
    }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int arg = 1; arg < argc; ++arg) {
            std::ifstream in(argv[arg]);
            Fixture f;
            in >> f.dims >> f.levels >> f.segments >> f.cells >> f.lanes;
            for (auto* v : {&f.offsets, &f.segment_ptr, &f.cell_segment, &f.codes, &f.mult, &f.csr_offsets, &f.csr_ptr, &f.csr_cell,
                            &f.a_seg, &f.a_begin, &f.a_end, &f.a_ptr, &f.b_level, &f.b_begin, &f.b_end, &f.b_ptr})
                read_vector(in, *v);
            read_vector(in, f.segment_inverse);
            read_vector(in, f.x);
            if (f.x.size() != static_cast<std::size_t>(f.levels) * f.lanes) throw std::runtime_error("operand shape");
            if (static_cast<int>(f.offsets.size()) != f.dims + 1) throw std::runtime_error("offsets");
            const Transposes tr = transposes(f);
            std::vector<int> identity;
            // Negative controls.
            bool budget_declined = false, width_rejected = false;
            {
                gpu::Options o;
                o.lanes = f.lanes;
                o.budget = 1;
                try { gpu::Operator denied(view(f, identity, nullptr), o); }
                catch (const gpu::Error& e) { budget_declined = e.kind() == gpu::ErrorKind::Memory; }
                gpu::Options o2;
                o2.lanes = f.lanes;
                gpu::Operator op(view(f, identity, nullptr), o2);
                try { op.begin_group(f.lanes + 1); }
                catch (const gpu::Error& e) { width_rejected = e.kind() == gpu::ErrorKind::Fatal; }
            }
            struct Config { const char* name; int private_max; bool transposed; gpu::BMethod b; int tiny; };
            const Config configs[] = {
                {"P", 48, false, gpu::BMethod::Auto, 0},
                {"B_tiles", 0, false, gpu::BMethod::Tiles, 0},
                {"B_radix", 0, false, gpu::BMethod::Radix, 0},
                {"T", 0, true, gpu::BMethod::Auto, 0},
                {"P_items_of_2", 48, false, gpu::BMethod::Auto, 2},
                {"B_items_of_2", 0, false, gpu::BMethod::Tiles, 2},
                {"T_items_of_2", 0, true, gpu::BMethod::Auto, 2}};
            for (const Config& c : configs) {
                std::vector<int> widths = {f.lanes};
                if (f.lanes == 8) widths.insert(widths.end(), {3, 5, 6, 7});
                for (int width : widths) {
                    gpu::Options o;
                    o.lanes = f.lanes;
                    o.private_max = c.private_max;
                    o.b_method = c.b;
                    if (c.tiny) { o.short_seg = c.tiny; o.short_level = c.tiny; o.item = c.tiny; }
                    gpu::Operator op(view(f, identity, c.transposed ? &tr : nullptr), o);
                    // p and q adjacent, one registration of their covering range.
                    const std::size_t count = f.x.size();
                    std::vector<double> rows(2 * count);
                    double* p = rows.data();
                    double* q = rows.data() + count;
                    for (int i = 0; i < f.levels; ++i)
                        for (int w = 0; w < f.lanes; ++w)
                            p[static_cast<std::size_t>(i) * f.lanes + w] = w < width ? f.x[static_cast<std::size_t>(i) * f.lanes + w] : 0.0;
                    op.register_host(p, 2 * count);
                    op.begin_group(width);
                    std::fill(q, q + count, 7.7e300);  // finite sentinels: an unwritten entry fails the grader
                    op.apply(p, q);
                    const std::vector<double> first(q, q + count);
                    std::fill(q, q + count, -7.7e300);
                    op.apply(p, q);
                    const bool same = std::memcmp(first.data(), q, count * sizeof(double)) == 0;
                    op.unregister_host();
                    std::string classes;
                    for (int j = 0; j < f.dims; ++j) classes += op.effect_class(j);
                    std::cout << "{\"path\":" << std::quoted(argv[arg]) << ",\"config\":\"" << c.name << "\",\"classes\":\"" << classes
                              << "\",\"width\":" << width << ",\"lanes\":" << f.lanes << ",\"calls\":" << op.statistics().calls
                              << ",\"repeat_bit_identical\":" << (same ? "true" : "false")
                              << ",\"budget_declined\":" << (budget_declined ? "true" : "false")
                              << ",\"illegal_width_rejected\":" << (width_rejected ? "true" : "false") << ",\"q\":[";
                    for (std::size_t i = 0; i < count; ++i) {
                        if (i) std::cout << ',';
                        std::cout << std::setprecision(17) << first[i];
                    }
                    std::cout << "]}\n";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
