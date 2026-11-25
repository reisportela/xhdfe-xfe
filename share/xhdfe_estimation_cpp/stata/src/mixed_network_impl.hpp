// This experimental translation unit shares the production level coding and team.
#include "mixed_network.hpp"
#include "network_device.hpp"
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>
#include <array>
#include <map>
#include <new>
#include <numeric>
#include <unordered_map>
#include <stdexcept>

namespace hdfe::detail {
namespace mixed_network {
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColSparse = Eigen::SparseMatrix<double>;
using RowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Block = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Vec = Eigen::Vector3d;
using Team = hdfe::reduced_pcg::internal::Team;
using Clock = std::chrono::steady_clock;

struct Edge {
    int next;
    std::uint16_t mask = 0;
    Block block;

    Edge(int target, const Block &values) : next(target), block(values) {}
};
struct Link {
    int a, b;
    double strength;
    Block block;
};
struct Match {
    int worker, firm, begin, end;
    Vec mean;
};

struct Data {
    const Eigen::Ref<const Eigen::VectorXd> &y;
    const Eigen::Ref<const Eigen::MatrixXd> &X;
    const Eigen::VectorXd &s;
    const Eigen::VectorXd &h;
    std::vector<int> w, f, t, rows, workers, matches, byfirm, row_match, periods;
    std::vector<Match> match;
    std::vector<Vec> origins;
    Matrix worker_mean, time_rhs, time_gram;
    int n, nw, nf, nt, ny, k;

    double value(int i, int c) const { return c ? X(i, c - 1) : y[i]; }
    Vec basis(int i) const {
        const Vec &a = origins[f[i]];
        return Vec(1., s[i] - a[1], h[i] - a[2]);
    }
};

std::vector<int> offsets(const std::vector<int> &keys, int levels) {
    std::vector<int> p(levels + 1, 0);
    for (int x : keys)
        ++p[x + 1];
    std::partial_sum(p.begin(), p.end(), p.begin());
    return p;
}

Data prepare(const Eigen::Ref<const Eigen::VectorXd> &y, const Eigen::Ref<const Eigen::MatrixXd> &X,
             const std::vector<Eigen::VectorXi> &fes,
             const std::vector<HeterogeneousSlopeTerm> &slopes, const Team &team, int blocks,
             std::vector<int> &levels) {
    const int i0 = slopes[0].fe_index, i1 = slopes[1].fe_index;
    std::vector<int> plain;
    for (int i = 0; i < 4; ++i)
        if (i != i0 && i != i1)
            plain.push_back(i);
    auto first = compact_codes(fes[plain[0]], team, blocks),
         second = compact_codes(fes[plain[1]], team, blocks);
    auto firm = compact_codes(fes[i0], team, blocks);
    if (first.levels < second.levels) {
        std::swap(first, second);
        std::swap(plain[0], plain[1]);
    }
    Data d{y, X, slopes[0].values, slopes[1].values};
    d.n = y.size();
    d.k = X.cols() + 1;
    d.nw = first.levels;
    d.nf = firm.levels;
    d.nt = second.levels;
    if (d.nt > 128 || d.nt < 2 || d.nf < 1 || d.nw < 1 || d.k > 64)
        throw MixedNetworkDecline("geometry");
    levels.assign(4, 0);
    levels[i0] = levels[i1] = d.nf;
    levels[plain[0]] = d.nw;
    levels[plain[1]] = d.nt;
    d.w.assign(first.codes.begin(), first.codes.end());
    d.f.assign(firm.codes.begin(), firm.codes.end());
    d.t.assign(second.codes.begin(), second.codes.end());
    d.workers = offsets(d.w, d.nw);
    d.rows.resize(d.n);
    std::vector<int> next = d.workers;
    for (int i = 0; i < d.n; ++i)
        d.rows[next[d.w[i]]++] = i;
    d.origins.assign(d.nf, Vec::Zero());
    for (int i = d.n - 1; i >= 0; --i)
        d.origins[d.f[i]] = Vec(1., d.s[i], d.h[i]);
    d.matches.assign(d.nw + 1, 0);
    team.run(d.nw, [&](int w) {
        auto begin = d.rows.begin() + d.workers[w], end = d.rows.begin() + d.workers[w + 1];
        std::sort(begin, end,
                  [&](int i, int j) { return d.f[i] < d.f[j] || (d.f[i] == d.f[j] && i < j); });
        int last = -1;
        for (auto row = begin; row != end; ++row)
            if (d.f[*row] != last) {
                last = d.f[*row];
                ++d.matches[w + 1];
            }
    });
    std::int64_t links = 0;
    for (int w = 0; w < d.nw; ++w) {
        const auto m = std::int64_t(d.matches[w + 1]);
        links += m * (m - 1) / 2;
    }
    // Bound explicit graph assembly on unusually mobile input before allocating it.
    if (links > std::min<std::int64_t>(12LL * d.n, 120000000LL))
        throw MixedNetworkDecline("edge_budget");
    std::partial_sum(d.matches.begin(), d.matches.end(), d.matches.begin());
    d.match.resize(d.matches.back());
    d.worker_mean.resize(d.nw, d.k);
    team.run(d.nw, [&](int w) {
        const int b = d.workers[w], e = d.workers[w + 1];
        auto mean = d.worker_mean.row(w);
        mean.setZero();
        int p = b, m = d.matches[w];
        while (p < e) {
            const int b0 = p, f = d.f[d.rows[p]];
            Vec z = Vec::Zero();
            do {
                int i = d.rows[p++];
                z += d.basis(i);
                for (int c = 0; c < d.k; ++c)
                    mean[c] += d.value(i, c) - d.value(d.rows[b], c);
            } while (p < e && d.f[d.rows[p]] == f);
            d.match[m++] = {w, f, b0, p, z / double(p - b0)};
        }
        for (int c = 0; c < d.k; ++c)
            mean[c] = d.value(d.rows[b], c) + mean[c] / double(e - b);
    });
    std::vector<int> order(d.match.size());
    for (size_t m = 0; m < order.size(); ++m)
        order[m] = d.match[m].firm;
    d.byfirm = offsets(order, d.nf);
    next = d.byfirm;
    d.row_match.resize(order.size());
    for (size_t m = 0; m < order.size(); ++m)
        d.row_match[next[order[m]]++] = int(m);

    // Fix one period in every connected component of the worker-period graph.
    std::vector<int> root(d.nt);
    std::iota(root.begin(), root.end(), 0);
    auto find = [&](int a) {
        while (root[a] != a) {
            root[a] = root[root[a]];
            a = root[a];
        }
        return a;
    };
    for (int w = 0; w < d.nw; ++w) {
        int q = d.t[d.rows[d.workers[w]]];
        for (int j = d.workers[w] + 1; j < d.workers[w + 1]; ++j) {
            int a = find(q), b = find(d.t[d.rows[j]]);
            if (a != b)
                root[std::max(a, b)] = std::min(a, b);
        }
    }
    d.ny = 0;
    d.periods.resize(d.nt);
    for (int t = 0; t < d.nt; ++t)
        d.periods[t] = find(t) == t ? -1 : d.ny++;
    if (d.ny == 0)
        throw MixedNetworkDecline("no_period_contrasts");
    const int parts = std::min(d.nw, blocks);
    std::vector<Matrix> yrhs(parts), ygram(parts);
    team.run(parts, [&](int part) {
        auto &rhs = yrhs[part];
        auto &gram = ygram[part];
        rhs = Matrix::Zero(d.ny, d.k);
        gram = Matrix::Zero(d.ny, d.ny);
        std::vector<int> counts(d.ny), used;
        used.reserve(d.ny);
        for (int w = std::int64_t(d.nw) * part / parts; w < std::int64_t(d.nw) * (part + 1) / parts;
             ++w) {
            const int b = d.workers[w], e = d.workers[w + 1];
            used.clear();
            for (int j = b; j < e; ++j) {
                int i = d.rows[j], t = d.periods[d.t[i]];
                if (t < 0)
                    continue;
                if (counts[t]++ == 0)
                    used.push_back(t);
                for (int c = 0; c < d.k; ++c)
                    rhs(t, c) += d.value(i, c) - d.worker_mean(w, c);
            }
            for (int t : used) {
                gram(t, t) += counts[t];
                for (int u : used)
                    gram(t, u) -= double(counts[t]) * counts[u] / double(e - b);
            }
            for (int t : used)
                counts[t] = 0;
        }
    });
    d.time_rhs = Matrix::Zero(d.ny, d.k);
    d.time_gram = Matrix::Zero(d.ny, d.ny);
    for (int p = 0; p < parts; ++p) {
        d.time_rhs += yrhs[p];
        d.time_gram += ygram[p];
    }
    return d;
}

struct System {
    RowSparse H;
    Matrix K, rhs, time_rhs;
    Eigen::LLT<Eigen::MatrixXd> time_factor;
    std::vector<std::array<int, 3>> index;
    std::vector<Vec> scale;
    std::vector<Block> inv_blocks;
    std::vector<Block> local;
    std::vector<Vec> moment_scale;
    std::vector<unsigned char> duplicate;
    Eigen::RowVectorXd response_scale;
    std::vector<Link> links;
    std::vector<int> firm_start;
    int dim = 0;
};

bool enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

// The frozen builder is also the differential oracle for emitted CSR bytes.
void legacy_firm_csr(const System &g, std::vector<Block> &diag,
                     std::vector<std::vector<Edge>> &edges, const Team &team,
                     bool preserve_input, RowSparse &result) {
    auto visit = [&](auto action) {
        for (int f = 0; f < static_cast<int>(diag.size()); ++f) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    int a = g.index[f][i], b = g.index[f][j];
                    if (a >= 0 && b >= 0 && diag[f](i, j) != 0)
                        action(a, b, diag[f](i, j) * g.scale[f][i] * g.scale[f][j]);
                }
            for (const Edge &e : edges[f])
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        int a = g.index[f][i], b = g.index[e.next][j];
                        if (a < 0 || b < 0 || e.block(i, j) == 0)
                            continue;
                        const double v = e.block(i, j) * g.scale[f][i] * g.scale[e.next][j];
                        action(a, b, v);
                        action(b, a, v);
                    }
        }
    };
    std::vector<int> nnz(g.dim, 0);
    visit([&](int a, int, double) { ++nnz[a]; });
    const auto total = std::accumulate(nnz.begin(), nnz.end(), std::int64_t(0));
    if (total >= std::numeric_limits<int>::max())
        throw MixedNetworkDecline("index_budget");
    std::vector<int> ptr(g.dim + 1, 0);
    std::partial_sum(nnz.begin(), nnz.end(), ptr.begin() + 1);
    struct Item {
        int col;
        double val;
    };
    std::vector<Item> entries(total);
    auto cursor = ptr;
    visit([&](int a, int b, double v) { entries[cursor[a]++] = {b, v}; });
    if (!preserve_input) {
        edges.clear();
        diag.clear();
    }
    team.run(g.dim, [&](int i) {
        std::sort(entries.begin() + ptr[i], entries.begin() + ptr[i + 1],
                  [](const Item &a, const Item &b) { return a.col < b.col; });
    });
    result.resize(g.dim, g.dim);
    result.reserve(total);
    for (int i = 0; i < g.dim; ++i) {
        result.startVec(i);
        for (int j = ptr[i]; j < ptr[i + 1]; ++j)
            result.insertBack(i, entries[j].col) = entries[j].val;
    }
    result.finalize();
}

std::uint16_t firm_edge_mask(const std::array<int, 3> &left,
                             const std::array<int, 3> &right, const Block &block) {
    std::uint16_t mask = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (left[i] >= 0 && right[j] >= 0 && block(i, j) != 0)
                mask |= 1u << (3 * i + j);
    return mask;
}

void scale_firm_edge(Block &block, std::uint16_t mask, const Vec &left, const Vec &right) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (mask & (1u << (3 * i + j)))
                block(i, j) = block(i, j) * left[i] * right[j];
}

bool owner_firm_csr(const System &g, std::vector<Block> &diag,
                    std::vector<std::vector<Edge>> &edges, const Team &team,
                    std::uint64_t &workspace_bytes, RowSparse &destination) {
    const int firms = static_cast<int>(diag.size());
    std::vector<std::uint16_t> diag_masks(firms);
    team.run(firms, [&](int f) {
        diag_masks[f] = firm_edge_mask(g.index[f], g.index[f], diag[f]);
        for (Edge &edge : edges[f])
            edge.mask = firm_edge_mask(g.index[f], g.index[edge.next], edge.block);
    });
    std::vector<int> begin(firms + 1, 0);
    for (const auto &list : edges)
        for (const Edge &edge : list)
            if (edge.mask != 0)
                ++begin[edge.next + 1];
    const auto lower_edges = std::accumulate(begin.begin(), begin.end(), std::int64_t(0));
    struct LowerEdge {
        int firm, edge;
    };
    workspace_bytes = std::uint64_t(lower_edges) * sizeof(LowerEdge) +
                      2 * std::uint64_t(begin.size()) * sizeof(int) +
                      std::uint64_t(diag_masks.size()) * sizeof(std::uint16_t) +
                      std::uint64_t(g.dim) * sizeof(int);
    if (workspace_bytes > (1ULL << 30) || lower_edges >= std::numeric_limits<int>::max())
        return false;
    std::partial_sum(begin.begin(), begin.end(), begin.begin());
    std::vector<LowerEdge> incoming(lower_edges);
    {
        auto cursor = begin;
        for (int f = 0; f < firms; ++f)
            for (int j = 0; j < static_cast<int>(edges[f].size()); ++j) {
                const Edge &edge = edges[f][j];
                if (edge.mask != 0)
                    incoming[cursor[edge.next]++] = {f, j};
            }
    }
    // Row indices rise by firm and active local coordinate. Each lower list
    // was appended in source order, and the upper list was already sorted.
    auto visit_owner = [&](int f, auto emit) {
        auto visit_block = [&](int other, const Block &block, std::uint16_t mask, bool lower) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const int at = lower ? 3 * j + i : 3 * i + j;
                    if (mask & (1u << at))
                        emit(i, g.index[other][j], block.data()[at]);
                }
        };
        for (int j = begin[f]; j < begin[f + 1]; ++j) {
            const LowerEdge &ref = incoming[j];
            const Edge &edge = edges[ref.firm][ref.edge];
            visit_block(ref.firm, edge.block, edge.mask, true);
        }
        visit_block(f, diag[f], diag_masks[f], false);
        for (const Edge &edge : edges[f])
            visit_block(edge.next, edge.block, edge.mask, false);
    };
    std::vector<int> nnz(g.dim);
    team.run(firms, [&](int f) {
        visit_owner(f, [&](int row, int, double) { ++nnz[g.index[f][row]]; });
    });
    const auto total = std::accumulate(nnz.begin(), nnz.end(), std::int64_t(0));
    if (total >= std::numeric_limits<int>::max())
        throw MixedNetworkDecline("index_budget");
    RowSparse result(g.dim, g.dim);
    result.resizeNonZeros(total);
    int *outer = result.outerIndexPtr();
    std::partial_sum(nnz.begin(), nnz.end(), outer + 1);
    int *inner = result.innerIndexPtr();
    double *values = result.valuePtr();
    // All allocations precede normalisation. The stored masks retain the
    // original sparsity, including any element which scales to zero.
    team.run(firms, [&](int f) {
        scale_firm_edge(diag[f], diag_masks[f], g.scale[f], g.scale[f]);
        for (Edge &edge : edges[f])
            scale_firm_edge(edge.block, edge.mask, g.scale[f], g.scale[edge.next]);
    });
    team.run(firms, [&](int f) {
        std::array<int, 3> cursor{};
        for (int i = 0; i < 3; ++i)
            if (g.index[f][i] >= 0)
                cursor[i] = outer[g.index[f][i]];
        visit_owner(f, [&](int row, int col, double value) {
            const int at = cursor[row]++;
            inner[at] = col;
            values[at] = value;
        });
    });
    destination.swap(result);
    return true;
}

struct CsrProduct {
    long double value = 0;
    long double allowance = 0;
    int mirror = -1;
};

// Offline only. Compute the product on the represented input blocks and
// scales, without relying on either assembly's FP64 multiplication order.
std::vector<CsrProduct> reference_products(
    const System &g, const std::vector<Block> &diagonal,
    const std::vector<std::vector<Edge>> &edges, const RowSparse &reference) {
    const auto *outer = reference.outerIndexPtr();
    const auto *inner = reference.innerIndexPtr();
    const auto slot = [&](int row, int col) {
        const auto *pos = std::lower_bound(inner + outer[row], inner + outer[row + 1], col);
        if (pos == inner + outer[row + 1] || *pos != col)
            throw std::logic_error("expected entry missing from the reference CSR");
        return static_cast<int>(pos - inner);
    };
    constexpr long double u64 = std::numeric_limits<double>::epsilon() / 2.0L;
    constexpr long double uref = std::numeric_limits<long double>::epsilon() / 2.0L;
    constexpr long double g64 = 2 * u64 / (1 - 2 * u64);
    constexpr long double gref = 2 * uref / (1 - 2 * uref);
    const auto normal = [](long double value) {
        return std::abs(value) >= std::numeric_limits<double>::min() &&
               std::abs(value) <= std::numeric_limits<double>::max();
    };
    std::vector<CsrProduct> products(reference.nonZeros());
    auto add = [&](int f, int other, const Block &block) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const int row = g.index[f][i], col = g.index[other][j];
                if (row < 0 || col < 0 || block(i, j) == 0)
                    continue;
                const long double a = block(i, j), b = g.scale[f][i], c = g.scale[other][j];
                const long double value = (a * b) * c;
                if (!normal(a * b) || !normal(a * c) || !normal(b * c) || !normal(value))
                    throw std::logic_error("CSR fixture outside the normal FP64 product envelope");
                const long double bound = std::nextafter(
                    (g64 + gref) * std::abs(value) / (1 - gref),
                    std::numeric_limits<long double>::infinity());
                const int upper = slot(row, col);
                const int lower = f == other ? -1 : slot(col, row);
                products[upper] = {value, bound, lower};
                if (lower != -1)
                    products[lower] = {value, bound, upper};
            }
    };
    for (int f = 0; f < static_cast<int>(diagonal.size()); ++f) {
        add(f, f, diagonal[f]);
        for (const Edge &edge : edges[f])
            add(f, edge.next, edge.block);
    }
    return products;
}

void verify_firm_csr(const RowSparse &actual, const RowSparse &reference,
                     const std::vector<CsrProduct> &products) {
    const auto rows = actual.rows(), count = actual.nonZeros();
    if (!actual.isCompressed() || rows != reference.rows() || count != reference.nonZeros() ||
        std::memcmp(actual.outerIndexPtr(), reference.outerIndexPtr(), (rows + 1) * sizeof(int)) ||
        (count && std::memcmp(actual.innerIndexPtr(), reference.innerIndexPtr(), count * sizeof(int))))
        throw std::logic_error("mixed network CSR indices differ from the frozen builder");
    const double *new_values = actual.valuePtr(), *old_values = reference.valuePtr();
    long differing = 0;
    long double max_fraction = 0;
    for (Eigen::Index i = 0; i < count; ++i) {
        const CsrProduct &p = products[i];
        const long double error = std::max(std::abs(p.value - new_values[i]),
                                           std::abs(p.value - old_values[i]));
        if (!(p.allowance > 0) || !(error <= p.allowance))
            throw std::logic_error("mixed network CSR value exceeds the two-product envelope");
        if (p.mirror >= 0 &&
            (std::memcmp(new_values + i, new_values + p.mirror, sizeof(double)) ||
             std::memcmp(old_values + i, old_values + p.mirror, sizeof(double))))
            throw std::logic_error("mixed network CSR off-diagonal blocks lost exact symmetry");
        differing += std::memcmp(new_values + i, old_values + i, sizeof(double)) != 0;
        max_fraction = std::max(max_fraction, error / p.allowance);
    }
    std::fprintf(stderr,
                 "mixed_network_csr_compare indices_exact=1 mirrored_exact=1 rows=%ld nnz=%ld "
                 "values_differ=%ld max_allowance_fraction=%.8Lg\n",
                 long(rows), long(count), differing, max_fraction);
}

// Local worker projection adds within-match differences and, for movers,
// the direction of the match mean from the firm origin. Once the within-
// match differences lie on a line, a mover's mean is on it exactly when
// its first row is. Use exact binary64 integer products for this cheap
// exception. Other uncertain near-singular blocks use the established route.
bool same_bounded_integer_direction(const Data &d, int firm) {
    auto small_integer = [](double value) {
        return std::abs(value) <= 0x1p24 && value == std::trunc(value);
    };
    double pivot_s = 0, pivot_h = 0;
    for (int q = d.byfirm[firm]; q < d.byfirm[firm + 1]; ++q) {
        const Match &m = d.match[d.row_match[q]];
        const bool mover = m.end - m.begin < d.workers[m.worker + 1] - d.workers[m.worker];
        const int first = d.rows[m.begin];
        const double base_s = mover ? d.origins[firm][1] : d.s[first];
        const double base_h = mover ? d.origins[firm][2] : d.h[first];
        if (!small_integer(base_s) || !small_integer(base_h))
            return false;
        for (int j = m.begin; j < m.end; ++j) {
            const int row = d.rows[j];
            if (!small_integer(d.s[row]) || !small_integer(d.h[row]))
                return false;
            const double s = d.s[row] - base_s, h = d.h[row] - base_h;
            if (pivot_s == 0 && pivot_h == 0) {
                pivot_s = s;
                pivot_h = h;
            } else if (s * pivot_h != h * pivot_s) {
                return false;
            }
        }
    }
    return pivot_s != 0 || pivot_h != 0;
}

System assemble(const Data &d, const Team &team, int blocks, bool forest,
                bool use_owner, bool& csr_owner_used) {
    if (std::uint64_t(d.nf) * 3 * d.ny * sizeof(double) > 6ULL * 1024 * 1024 * 1024)
        throw MixedNetworkDecline("period_budget");
    Matrix cross(3 * d.nf, d.ny), brhs(3 * d.nf, d.k);
    std::vector<Block> diag(d.nf);
    std::vector<std::vector<Edge>> edges(d.nf);
    std::vector<unsigned char> same(d.nf);
    std::vector<unsigned char> weak_slopes(d.nf);
    std::vector<Vec> moment_scale(d.nf);
    team.run(d.nf, [&](int f) {
        auto kt = cross.middleRows(3 * f, 3);
        auto rhs = brhs.middleRows(3 * f, 3);
        kt.setZero();
        rhs.setZero();
        Block D = Block::Zero();
        Vec raw = Vec::Zero();
        bool dup = true;
        std::unordered_map<int, Block> off;
        for (int q = d.byfirm[f]; q < d.byfirm[f + 1]; ++q) {
            const Match &m = d.match[d.row_match[q]];
            const int n = m.end - m.begin, w = m.worker, b = d.workers[w], e = d.workers[w + 1];
            const double frac = double(n) / (e - b);
            D.noalias() +=
                (double(n) * ((e - b) - n) / double(e - b)) * m.mean * m.mean.transpose();
            for (int j = m.begin; j < m.end; ++j) {
                int i = d.rows[j];
                Vec z = d.basis(i), dif = z - m.mean;
                D.noalias() += dif * dif.transpose();
                dup = dup && (static_cast<long double>(d.s[i]) - d.origins[f][1]) ==
                                 (static_cast<long double>(d.h[i]) - d.origins[f][2]);
                raw[0] += 1.;
                raw[1] += d.s[i] * d.s[i];
                raw[2] += d.h[i] * d.h[i];
                const int t = d.periods[d.t[i]];
                if (t >= 0)
                    kt.col(t) += z;
                for (int c = 0; c < d.k; ++c)
                    rhs.col(c) += z * (d.value(i, c) - d.worker_mean(w, c));
            }
            const Vec v = frac * m.mean;
            for (int j = b; j < e; ++j) {
                int t = d.periods[d.t[d.rows[j]]];
                if (t >= 0)
                    kt.col(t) -= v;
            }
            for (int k = d.matches[w]; k < d.matches[w + 1]; ++k) {
                const Match &h = d.match[k];
                if (h.firm <= f)
                    continue;
                auto it = off.try_emplace(h.firm, Block::Zero()).first;
                it->second.noalias() -= double(h.end - h.begin) * v * h.mean.transpose();
            }
        }
        diag[f] = D;
        same[f] = dup;
        if (!dup && D(1, 1) > 0 && D(2, 2) > 0) {
            const double correlation = (D(1, 2) / std::sqrt(D(1, 1))) / std::sqrt(D(2, 2));
            // The scaled two-column design has condition number at least
            // about 1024 here. Normal equations lose accuracy in this region.
            if (!hdfe::detail::ieee_finite(correlation) ||
                std::abs(correlation) >= 1 - 0x1p-19) {
                weak_slopes[f] = !same_bounded_integer_direction(d, f);
            }
        }
        for (int j = 0; j < 3; ++j)
            moment_scale[f][j] = raw[j] > 0 ? 1 / std::sqrt(raw[j]) : 0.;
        auto &list = edges[f];
        list.reserve(off.size());
        for (const auto &it : off)
            list.push_back({it.first, it.second});
        std::sort(list.begin(), list.end(),
                  [](const Edge &x, const Edge &y) { return x.next < y.next; });
    });
    if (std::find(weak_slopes.begin(), weak_slopes.end(), 1) != weak_slopes.end())
        throw MixedNetworkDecline("ill_conditioned_local_slopes");
    System g;
    g.index.resize(d.nf);
    g.scale.resize(d.nf);
    g.firm_start.resize(d.nf + 1);
    g.duplicate = same;
    g.moment_scale = std::move(moment_scale);
    Matrix sq(blocks, d.k);
    team.run(blocks, [&](int b) {
        auto v = sq.row(b);
        v.setZero();
        for (int i = std::int64_t(d.n) * b / blocks; i < std::int64_t(d.n) * (b + 1) / blocks; ++i)
            for (int c = 0; c < d.k; ++c) {
                const double x = d.value(i, c) - d.worker_mean(d.w[i], c);
                v[c] += x * x;
            }
    });
    g.response_scale = sq.colwise().sum().array().sqrt();
    for (int f = 0; f < d.nf; ++f) {
        g.firm_start[f] = g.dim;
        for (int i = 0; i < 3; ++i) {
            const bool active = diag[f](i, i) > 0 && !(i == 2 && same[f]);
            g.index[f][i] = active ? g.dim++ : -1;
            g.scale[f][i] = active ? 1 / std::sqrt(diag[f](i, i)) : 0.;
        }
    }
    g.firm_start[d.nf] = g.dim;
    g.K.resize(g.dim, d.ny);
    g.rhs.resize(g.dim, d.k);
    g.inv_blocks.resize(d.nf);
    g.local.resize(d.nf);
    team.run(d.nf, [&](int f) {
        Block bb = Block::Identity();
        for (int i = 0; i < 3; ++i) {
            int u = g.index[f][i];
            if (u < 0)
                continue;
            g.K.row(u) = cross.row(3 * f + i) * g.scale[f][i];
            g.rhs.row(u) = brhs.row(3 * f + i) * g.scale[f][i];
            for (int j = 0; j < 3; ++j)
                if (g.index[f][j] >= 0)
                    bb(i, j) = diag[f](i, j) * g.scale[f][i] * g.scale[f][j];
        }
        bb.diagonal().array() += std::sqrt(std::numeric_limits<double>::epsilon());
        g.local[f] = bb;
        g.inv_blocks[f] = bb.ldlt().solve(Block::Identity());
    });
    cross.resize(0, 0);
    brhs.resize(0, 0);
    g.time_factor.compute(Eigen::MatrixXd(d.time_gram));
    if (g.time_factor.info() != Eigen::Success)
        throw MixedNetworkDecline("period_factor");
    g.time_rhs = g.time_factor.matrixL().solve(d.time_rhs);
    int parts = std::max(1, std::min(blocks, g.dim));
    team.run(parts, [&](int b) {
        const int i = std::int64_t(g.dim) * b / parts, end = std::int64_t(g.dim) * (b + 1) / parts;
        Eigen::MatrixXd v = g.K.middleRows(i, end - i).transpose();
        g.time_factor.matrixL().solveInPlace(v);
        g.K.middleRows(i, end - i) = v.transpose();
        g.rhs.middleRows(i, end - i).noalias() -= g.K.middleRows(i, end - i) * g.time_rhs;
    });
    if (forest) {
        for (int f = 0; f < d.nf; ++f)
            for (const Edge &e : edges[f]) {
                Block b = g.scale[f].asDiagonal() * e.block * g.scale[e.next].asDiagonal();
                const double weight = b.squaredNorm();
                if (weight > 0)
                    g.links.push_back({f, e.next, weight, b});
            }
    }
    const auto csr_start = Clock::now();
    const bool compare = use_owner && enabled("XHDFE_EXPERIMENT_NETWORK_CSR_COMPARE");
    RowSparse reference;
    std::vector<CsrProduct> reference_envelope;
    if (compare) {
        if (d.n > 100000 || g.dim > 20000)
            throw std::logic_error("mixed network CSR comparison is limited to synthetic fixtures");
        legacy_firm_csr(g, diag, edges, team, true, reference);
        reference_envelope = reference_products(g, diag, edges, reference);
    }
    bool owner = false;
    std::uint64_t owner_bytes = 0;
    if (use_owner) {
        try {
            owner = owner_firm_csr(g, diag, edges, team, owner_bytes, g.H);
        } catch (const std::bad_alloc &) {
            // The original input is still live if an allocation failed.
        }
    }
    if (compare && !owner)
        throw std::logic_error("mixed network CSR comparison did not run the owner path");
    if (compare)
        verify_firm_csr(g.H, reference, reference_envelope);
    if (!owner)
        legacy_firm_csr(g, diag, edges, team, false, g.H);
    csr_owner_used = owner;
    if (mixed_network_trace_enabled()) {
        const double seconds = std::chrono::duration<double>(Clock::now() - csr_start).count();
        std::fprintf(stderr, "mixed_network_csr owner=%d requested=%d nnz=%ld workspace=%llu seconds=%.5f\n",
                     int(bool(owner)), int(use_owner), long(g.H.nonZeros()),
                     static_cast<unsigned long long>(owner_bytes), seconds);
    }
    return g;
}

struct Forest {
    std::vector<int> parent;
    std::vector<std::vector<int>> children, waves;
    std::vector<Block> to_parent, inverse, upstream;

    void setup(System &g, const Team &team) {
        const int nf = g.index.size();
        parent.assign(nf, -2);
        children.resize(nf);
        to_parent.resize(nf);
        inverse.resize(nf);
        upstream.resize(nf);
        std::vector<int> root(nf), size(nf, 1);
        std::iota(root.begin(), root.end(), 0);
        auto find = [&](int a) {
            while (a != root[a]) {
                root[a] = root[root[a]];
                a = root[a];
            }
            return a;
        };
        std::sort(g.links.begin(), g.links.end(), [](const Link &a, const Link &b) {
            return a.strength != b.strength ? a.strength > b.strength
                                            : std::tie(a.a, a.b) < std::tie(b.a, b.b);
        });
        std::vector<std::vector<Edge>> adj(nf);
        for (const auto &e : g.links) {
            int a = find(e.a), b = find(e.b);
            if (a == b)
                continue;
            if (size[a] < size[b])
                std::swap(a, b);
            root[b] = a;
            size[a] += size[b];
            adj[e.a].push_back({e.b, e.block});
            adj[e.b].push_back({e.a, e.block.transpose()});
        }
        g.links.clear();
        g.links.shrink_to_fit();
        std::vector<int> center(nf, -1);
        for (int i = 0; i < nf; ++i) {
            int a = find(i);
            if (center[a] < 0 || adj[i].size() > adj[center[a]].size())
                center[a] = i;
        }
        waves.emplace_back();
        for (int c : center)
            if (c >= 0) {
                parent[c] = -1;
                waves.front().push_back(c);
            }
        for (size_t w = 0; w < waves.size(); ++w) {
            std::vector<int> next;
            for (int node : waves[w])
                for (const auto &e : adj[node])
                    if (parent[e.next] == -2) {
                        parent[e.next] = node;
                        children[node].push_back(e.next);
                        to_parent[e.next] = e.block.transpose();
                        next.push_back(e.next);
                    }
            if (!next.empty())
                waves.push_back(std::move(next));
        }
        for (auto it = waves.rbegin(); it != waves.rend(); ++it) {
            auto &wave = *it;
            team.run(wave.size(), [&](int i) {
                int f = wave[i];
                Block pivot = g.local[f];
                for (int c : children[f])
                    pivot.noalias() -= to_parent[c].transpose() * upstream[c];
                inverse[f] = pivot.ldlt().solve(Block::Identity());
                if (parent[f] >= 0)
                    upstream[f].noalias() = inverse[f] * to_parent[f];
            });
        }
    }

    void apply(const System &g, const Team &team, const Matrix &r, Matrix &z, Matrix &work) const {
        for (auto it = waves.rbegin(); it != waves.rend(); ++it) {
            const auto &wave = *it;
            team.run(wave.size(), [&](int i) {
                int f = wave[i];
                auto v = work.middleRows(3 * f, 3);
                v.setZero();
                for (int j = 0; j < 3; ++j)
                    if (g.index[f][j] >= 0)
                        v.row(j) = r.row(g.index[f][j]);
                for (int child : children[f])
                    v.noalias() -= to_parent[child].transpose() * work.middleRows(3 * child, 3);
                v = (inverse[f] * v).eval();
            });
        }
        for (const auto &wave : waves)
            team.run(wave.size(), [&](int i) {
                int f = wave[i];
                auto v = work.middleRows(3 * f, 3);
                if (parent[f] >= 0)
                    v.noalias() -= upstream[f] * work.middleRows(3 * parent[f], 3);
                for (int j = 0; j < 3; ++j)
                    if (g.index[f][j] >= 0)
                        z.row(g.index[f][j]) = v.row(j);
            });
    }
};

Matrix cross_product(const Matrix &a, const Matrix &b, const Team &team, int blocks) {
    const int parts = std::max(1, std::min<int>(blocks, (a.rows() + 1023) / 1024));
    std::vector<Matrix> sub(parts);
    team.run(parts, [&](int k) {
        int i = a.rows() * k / parts, e = a.rows() * (k + 1) / parts;
        sub[k].noalias() = a.middleRows(i, e - i).transpose() * b.middleRows(i, e - i);
    });
    Matrix out = Matrix::Zero(a.cols(), b.cols());
    for (const auto &s : sub)
        out += s;
    return out;
}

void multiply(const System &g, const Matrix &p, Matrix &q, const Team &team, int blocks) {
    Matrix kt = cross_product(g.K, p, team, blocks);
    const int nv = p.cols();
    int parts = std::max(1, std::min(g.dim, blocks));
    const int *ptr = g.H.outerIndexPtr();
    const int *cols = g.H.innerIndexPtr();
    const double *val = g.H.valuePtr();
    team.run(parts, [&](int b) {
        int first = std::int64_t(g.dim) * b / parts, last = std::int64_t(g.dim) * (b + 1) / parts;
        q.middleRows(first, last - first).noalias() = -g.K.middleRows(first, last - first) * kt;
        for (int i = first; i < last; ++i) {
            double *dest = q.data() + std::ptrdiff_t(i) * nv;
            for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
                const double *src = p.data() + std::ptrdiff_t(cols[j]) * nv;
                const double x = val[j];
                for (int h = 0; h < nv; ++h)
                    dest[h] += x * src[h];
            }
        }
    });
}

struct Preconditioner {
    Eigen::SimplicialLDLT<ColSparse> factor;
    Eigen::LLT<Eigen::MatrixXd> border;
    Matrix invK, workspace;
    Forest forest;
    MixedNetworkMode mode;

    explicit Preconditioner(MixedNetworkMode m) : mode(m) {}
    static bool admit_symbolic_fill(const ColSparse &h) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> inverse;
        Eigen::AMDOrdering<int> ordering;
        ordering(h, inverse);
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation =
            inverse.inverse();
        ColSparse upper(h.rows(), h.cols());
        upper.selfadjointView<Eigen::Upper>() =
            h.selfadjointView<Eigen::Lower>().twistedBy(permutation);
        std::vector<int> parent(h.rows(), -1), visited(h.rows(), -1), length(h.rows(), 0);
        std::int64_t entries = 0, work = 0;
        for (int k = 0; k < upper.cols(); ++k) {
            visited[k] = k;
            for (ColSparse::InnerIterator v(upper, k); v; ++v) {
                int i = v.row();
                if (i >= k)
                    continue;
                while (visited[i] != k) {
                    if (parent[i] < 0)
                        parent[i] = k;
                    visited[i] = k;
                    work += 2LL * length[i] + 1;
                    ++length[i];
                    ++entries;
                    if (entries > 32000000 || work > 60000000000LL)
                        return false;
                    i = parent[i];
                }
            }
        }
        return true;
    }
    void prepare(System &g, const Team &team, int blocks) {
        if (mode == MixedNetworkMode::Forest) {
            forest.setup(g, team);
            workspace.resize(3 * g.index.size(), g.rhs.cols());
            return;
        }
        if (mode != MixedNetworkMode::Direct)
            return;
        if (g.dim > 200000 || g.H.nonZeros() > 8000000)
            throw MixedNetworkDecline("factor_budget");
        ColSparse h = g.H;
        const double shift = std::sqrt(std::numeric_limits<double>::epsilon());
        h.diagonal().array() += shift;
        if (!admit_symbolic_fill(h))
            throw MixedNetworkDecline("symbolic_fill_budget");
        factor.compute(h);
        if (factor.info() != Eigen::Success || !(factor.vectorD().minCoeff() > 0))
            throw MixedNetworkDecline("firm_factor");
        invK.resize(g.K.rows(), g.K.cols());
        team.run(g.K.cols(), [&](int j) {
            Eigen::VectorXd v = g.K.col(j);
            invK.col(j) = factor.solve(v);
        });
        Matrix c =
            Matrix::Identity(g.K.cols(), g.K.cols()) - cross_product(g.K, invK, team, blocks);
        c = (0.5 * (c + c.transpose())).eval();
        c.diagonal().array() += shift;
        border.compute(Eigen::MatrixXd(c));
        if (border.info() != Eigen::Success)
            throw MixedNetworkDecline("border_factor");
    }
    void apply(const System &g, const Matrix &r, Matrix &z, const Team &team, int blocks) {
        if (mode == MixedNetworkMode::Forest) {
            forest.apply(g, team, r, z, workspace);
            return;
        }
        if (mode == MixedNetworkMode::Direct) {
            team.run(r.cols(), [&](int j) {
                Eigen::VectorXd v = r.col(j);
                z.col(j) = factor.solve(v);
            });
            Matrix k = border.solve(cross_product(g.K, z, team, blocks));
            int parts = std::min(g.dim, blocks);
            team.run(parts, [&](int b) {
                int i = std::int64_t(g.dim) * b / parts, e = std::int64_t(g.dim) * (b + 1) / parts;
                z.middleRows(i, e - i).noalias() += invK.middleRows(i, e - i) * k;
            });
            return;
        }
        team.run(g.index.size(), [&](int f) {
            for (int j = 0; j < 3; ++j) {
                int ix = g.index[f][j];
                if (ix < 0)
                    continue;
                z.row(ix).setZero();
                for (int l = 0; l < 3; ++l)
                    if (g.index[f][l] >= 0)
                        z.row(ix) += g.inv_blocks[f](j, l) * r.row(g.index[f][l]);
            }
        });
    }
};

AbsorptionResult recover(const Data &d, const System &g, const Matrix &state, const Team &team,
                         int blocks, const std::vector<int> &levels) {
    Matrix yr = g.time_rhs - cross_product(g.K, state, team, blocks);
    g.time_factor.matrixU().solveInPlace(yr);
    Matrix coef(3 * d.nf, d.k);
    team.run(d.nf, [&](int f) {
        for (int j = 0; j < 3; ++j) {
            auto dest = coef.row(3 * f + j);
            if (g.index[f][j] < 0)
                dest.setZero();
            else
                dest = state.row(g.index[f][j]) * g.scale[f][j];
        }
    });
    AbsorptionResult res;
    res.fe_levels = levels;
    res.y_tilde.resize(d.n);
    res.X_tilde.resize(d.n, d.k - 1);
    team.run(d.nw, [&](int w) {
        const int b = d.workers[w], e = d.workers[w + 1];
        std::array<double, 64> avg{};
        for (int j = b; j < e; ++j) {
            int i = d.rows[j], f = d.f[i], t = d.periods[d.t[i]];
            Vec z = d.basis(i);
            for (int c = 0; c < d.k; ++c) {
                double pred = (t >= 0 ? yr(t, c) : 0.) + coef(3 * f, c) +
                              z[1] * coef(3 * f + 1, c) + z[2] * coef(3 * f + 2, c);
                avg[c] += pred;
                double v = (d.value(i, c) - d.worker_mean(w, c)) - pred;
                if (c == 0)
                    res.y_tilde[i] = v;
                else
                    res.X_tilde(i, c - 1) = v;
            }
        }
        for (int c = 0; c < d.k; ++c)
            avg[c] /= double(e - b);
        for (int j = b; j < e; ++j) {
            int i = d.rows[j];
            res.y_tilde[i] += avg[0];
            for (int c = 1; c < d.k; ++c)
                res.X_tilde(i, c - 1) += avg[c];
        }
    });
    return res;
}

Eigen::RowVectorXd dots(const Matrix &a, const Matrix &b, const Team &team, int blocks) {
    const int parts = std::min<int>(blocks, a.rows());
    Matrix sums(parts, a.cols());
    team.run(parts, [&](int k) {
        int i = a.rows() * k / parts, e = a.rows() * (k + 1) / parts;
        sums.row(k) =
            (a.middleRows(i, e - i).array() * b.middleRows(i, e - i).array()).colwise().sum();
    });
    return sums.colwise().sum();
}

// A scheduling hint only. Translate the recurrence back to the three raw firm
// moments. Every accepted result is reconstructed and certified on the data.
double moment_hint(const System &g, const Data &d, const double *residual, const Team &team,
                   int blocks) {
    const int parts = std::min(d.nf, blocks);
    std::vector<double> worst(parts);
    team.run(parts, [&](int p) {
        double top = 0.;
        for (int f = std::int64_t(d.nf) * p / parts; f < std::int64_t(d.nf) * (p + 1) / parts;
             ++f) {
            for (int c = 0; c < d.k; ++c) {
                Vec m = Vec::Zero();
                for (int j = 0; j < 3; ++j) {
                    int i = g.index[f][j];
                    if (i >= 0)
                        m[j] = residual[std::ptrdiff_t(i) * d.k + c] / g.scale[f][j];
                }
                if (g.duplicate[f])
                    m[2] = m[1];
                m[1] += d.origins[f][1] * m[0];
                m[2] += d.origins[f][2] * m[0];
                if (g.response_scale[c] > 0) {
                    const double ratio = (m.cwiseProduct(g.moment_scale[f])).cwiseAbs().maxCoeff() /
                                         g.response_scale[c];
                    top = std::max(top, ratio);
                }
            }
        }
        worst[p] = top;
    });
    return *std::max_element(worst.begin(), worst.end());
}

// Match both cluster partitions on the actual fit rows, including a possible
// permutation of category labels. A refinement or a coarsening cannot reuse it.
bool same_partition(const Data &d, const Eigen::VectorXi &cluster, bool workers, const Team &team,
                    int blocks) {
    if (cluster.size() != d.n)
        return false;
    const auto &codes = workers ? d.w : d.f;
    const int count = workers ? d.nw : d.nf;
    std::vector<int> labels(count);
    team.run(count, [&](int g) {
        const int pos = workers ? d.workers[g] : d.match[d.row_match[d.byfirm[g]]].begin;
        labels[g] = cluster[d.rows[pos]];
    });
    std::vector<unsigned char> valid(blocks, 1);
    team.run(blocks, [&](int b) {
        for (int i = std::int64_t(d.n) * b / blocks; i < std::int64_t(d.n) * (b + 1) / blocks; ++i)
            if (labels[codes[i]] != cluster[i]) {
                valid[b] = 0;
                break;
            }
    });
    if (std::find(valid.begin(), valid.end(), 0) != valid.end())
        return false;
    if (!std::is_sorted(labels.begin(), labels.end()))
        std::sort(labels.begin(), labels.end());
    return std::adjacent_find(labels.begin(), labels.end()) == labels.end();
}

void retain_score_geometry(Data &d, const std::vector<Eigen::VectorXi> &clusters,
                           const HdfeOptions &options, const Team &team, int blocks,
                           std::unique_ptr<MixedNetworkScores> &destination) try {
    if (clusters.size() != 2 || d.k < 2 || d.k > 17)
        return;

    const std::uint64_t matches = d.match.size();
    if (matches == static_cast<std::uint64_t>(d.nw) ||
        matches == static_cast<std::uint64_t>(d.nf))
        return;
    constexpr std::uint64_t budget = 1ULL << 30;
    const std::uint64_t geometry_bytes =
        (std::uint64_t(d.n) + d.nw + d.nf + 2 * matches + 3) * sizeof(int);
    const std::uint64_t score_bytes =
        (matches * d.k + 3ULL * blocks * d.k * d.k) * sizeof(double);
    const std::uint64_t partition_bytes = std::uint64_t(std::max(d.nw, d.nf)) * sizeof(int);
    if (geometry_bytes + std::max(score_bytes, partition_bytes) > budget)
        return;

    int worker_dim = 0;
    if (!same_partition(d, clusters[0], true, team, blocks))
        worker_dim = 1;
    if ((worker_dim == 1 && !same_partition(d, clusters[1], true, team, blocks)) ||
        !same_partition(d, clusters[1 - worker_dim], false, team, blocks))
        return;
    auto plan = std::make_unique<MixedNetworkScores>();
    plan->clusters = &clusters;
    plan->counts = {d.nw, d.nf, int(d.match.size())};
    if (worker_dim == 1)
        std::swap(plan->counts[0], plan->counts[1]);
    plan->threads = team.threads();
    plan->observer = options.parallel_observer;
    plan->geometry = std::make_unique<MixedNetworkScores::Geometry>();
    auto &geometry = *plan->geometry;
    geometry.match_rows.resize(d.match.size() + 1);
    team.run(int(d.match.size()), [&](int m) { geometry.match_rows[m] = d.match[m].begin; });
    geometry.match_rows.back() = d.n;
    geometry.rows = std::move(d.rows);
    geometry.worker_matches = std::move(d.matches);
    geometry.firm_matches = std::move(d.byfirm);
    geometry.firm_match_ids = std::move(d.row_match);
    destination = std::move(plan);
} catch (const std::bad_alloc &) {
    destination.reset();
}
} // namespace mixed_network

bool MixedNetworkScores::matches(const Eigen::Ref<const Eigen::MatrixXd> &x,
                                 const Eigen::VectorXd &u,
                                 const std::vector<Eigen::VectorXi> &source,
                                 ClusterDofMethod method, bool adjust) const {
    return columns > 0 && clusters == &source && x.cols() == columns &&
           x.rows() == row_count && u.size() == row_count &&
           x.data() == design_data && u.data() == residual_data &&
           conventional_adjustment == (adjust && method == ClusterDofMethod::Conventional);
}

bool MixedNetworkScores::form(const Eigen::Ref<const Eigen::MatrixXd> &x, const Eigen::VectorXd &u,
                              ClusterDofMethod method, bool adjust) try {
    using namespace mixed_network;
    auto plan = std::move(geometry);
    if (!plan || x.rows() != u.size() || u.size() != Eigen::Index(plan->rows.size()) ||
        x.cols() < 1 || x.cols() > 16)
        return false;
    const int q = x.cols() + 1;
    const int nw = plan->worker_matches.size() - 1;
    const int nf = plan->firm_matches.size() - 1;
    const int nm = plan->match_rows.size() - 1;
    const int blocks = deterministic_parallel_chunk_count(std::max(1, nw));
    rp::Hooks hooks;
    if (observer) {
        hooks.context = observer;
        hooks.begin = observer_begin;
        hooks.observe = observer_observe;
        hooks.end = observer_end;
    }
    Team team(threads, hooks, 8 * blocks);
    Matrix score(nm, q);
    const Matrix zero = Matrix::Zero(q, q);
    std::vector<Matrix> worker_meat(blocks, zero), firm_meat(blocks, zero), pair_meat(blocks, zero);
    std::vector<unsigned char> outside(blocks, 0);
    auto bounded = [](double a) {
        a = std::abs(a);
        return a == 0 || (a >= 0x1p-120 && a <= 0x1p120);
    };
    auto add_outer = [q](Matrix &meat, const double *s) {
        for (int i = 0; i < q; ++i)
            for (int j = 0; j <= i; ++j)
                meat(i, j) += s[i] * s[j];
    };
    team.run(blocks, [&](int part) {
        Matrix &mw = worker_meat[part];
        Matrix &mm = pair_meat[part];
        for (int w = std::int64_t(nw) * part / blocks; w < std::int64_t(nw) * (part + 1) / blocks;
             ++w) {
            std::array<double, 17> sw{};
            for (int m = plan->worker_matches[w]; m < plan->worker_matches[w + 1]; ++m) {
                double *sm = score.data() + std::ptrdiff_t(m) * q;
                std::fill_n(sm, q, 0.);
                for (int j = plan->match_rows[m]; j < plan->match_rows[m + 1]; ++j) {
                    const int row = plan->rows[j];
                    const double e = u[row];
                    if (!bounded(e))
                        outside[part] = 1;
                    for (int c = 0; c < q - 1; ++c) {
                        const double v = x(row, c);
                        if (!bounded(v))
                            outside[part] = 1;
                        sm[c] += v * e;
                    }
                    sm[q - 1] += e;
                }
                for (int c = 0; c < q; ++c)
                    sw[c] += sm[c];
                add_outer(mm, sm);
            }
            add_outer(mw, sw.data());
        }
    });
    if (std::find(outside.begin(), outside.end(), 1) != outside.end())
        return false;
    team.run(blocks, [&](int part) {
        Matrix &mf = firm_meat[part];
        for (int f = std::int64_t(nf) * part / blocks; f < std::int64_t(nf) * (part + 1) / blocks;
             ++f) {
            std::array<double, 17> sf{};
            for (int j = plan->firm_matches[f]; j < plan->firm_matches[f + 1]; ++j) {
                const double *sm = score.data() + std::ptrdiff_t(plan->firm_match_ids[j]) * q;
                for (int c = 0; c < q; ++c)
                    sf[c] += sm[c];
            }
            add_outer(mf, sf.data());
        }
    });
    Matrix wm = Matrix::Zero(q, q), fm = Matrix::Zero(q, q), pm = Matrix::Zero(q, q);
    for (int b = 0; b < blocks; ++b) {
        wm += worker_meat[b];
        fm += firm_meat[b];
        pm += pair_meat[b];
    }
    // Leave suspect zero diagonals to the established exact-score checks.
    // Nonzero scores within the operand envelope have normal squared values.
    if ((wm.diagonal().array() <= 0).any() || (fm.diagonal().array() <= 0).any() ||
        (pm.diagonal().array() <= 0).any())
        return false;
    const bool adjust_components = adjust && method == ClusterDofMethod::Conventional;
    auto factor = [&](int g) {
        return adjust_components && g > 1 ? double(g) / (g - 1) : 1.;
    };
    const Matrix sum = factor(nw) * wm + factor(nf) * fm - factor(nm) * pm;
    if (!hdfe::detail::ieee_all_finite(sum))
        return false;
    raw = sum.selfadjointView<Eigen::Lower>();
    conventional_adjustment = adjust_components;
    row_count = x.rows();
    columns = x.cols();
    design_data = x.data();
    residual_data = u.data();
    if (mixed_network_trace_enabled()) {
        std::fprintf(stderr,
                     "mixed_network_scores width=%d clusters_worker=%d clusters_firm=%d matches=%d\n",
                     q, nw, nf, nm);
    }
    return true;
} catch (const std::bad_alloc &) {
    return false;
}

bool mixed_network_trace_enabled() noexcept {
    return std::getenv("XHDFE_EXPERIMENT_SLOPE_TRACE") != nullptr;
}

MixedNetworkMode mixed_network_mode() noexcept {
    auto on = [](const char *k) {
        const char *v = std::getenv(k);
        return v && std::strcmp(v, "1") == 0;
    };
    if (on("XHDFE_EXPERIMENT_SLOPE_DIRECT"))
        return MixedNetworkMode::Direct;
    if (on("XHDFE_EXPERIMENT_SLOPE_NETWORK"))
        return MixedNetworkMode::Forest;
    if (on("XHDFE_EXPERIMENT_SLOPE_SCHUR"))
        return MixedNetworkMode::Diagonal;
    return MixedNetworkMode::Off;
}

MixedNetworkMode mixed_network_mode(SlopeSolverRequest request) noexcept {
    switch (request) {
    case SlopeSolverRequest::Schur: return MixedNetworkMode::Diagonal;
    case SlopeSolverRequest::Network: return MixedNetworkMode::Forest;
    case SlopeSolverRequest::Direct: return MixedNetworkMode::Direct;
    default: return MixedNetworkMode::Off;
    }
}

SlopeSolverDiagnostics resolve_slope_solver(const HdfeOptions& options, bool& scores, bool& csr_owner) noexcept {
    SlopeSolverDiagnostics resolved;
    const bool explicit_request = options.slope_solver != SlopeSolverRequest::Unset;
    if (explicit_request) {
        resolved.request = options.slope_solver;
        resolved.source = options.slope_solver_from_global ? SlopeSolverSource::Global : SlopeSolverSource::Option;
    } else {
        const MixedNetworkMode legacy = mixed_network_mode();
        switch (legacy) {
        case MixedNetworkMode::Direct: resolved.request = SlopeSolverRequest::Direct; break;
        case MixedNetworkMode::Forest: resolved.request = SlopeSolverRequest::Network; break;
        case MixedNetworkMode::Diagonal: resolved.request = SlopeSolverRequest::Schur; break;
        case MixedNetworkMode::Off: break;
        }
        if (legacy != MixedNetworkMode::Off) resolved.source = SlopeSolverSource::Environment;
    }
    scores = false;
    csr_owner = false;
    if (resolved.request != SlopeSolverRequest::Off) {
        scores = options.slope_scores.value_or(explicit_request || mixed_network::enabled("XHDFE_EXPERIMENT_NETWORK_SCORES"));
        csr_owner = options.slope_csr_owner.value_or(explicit_request || mixed_network::enabled("XHDFE_EXPERIMENT_NETWORK_CSR_OWNER"));
    }
    resolved.reset_outcome();
    return resolved;
}

bool mixed_network_shape(const std::vector<Eigen::VectorXi> &fes,
                         const std::vector<HeterogeneousSlopeTerm> &slopes) {
    if (fes.size() != 4 || slopes.size() != 2 ||
        !(slopes[0].include_intercept || slopes[1].include_intercept))
        return false;
    int a = slopes[0].fe_index, b = slopes[1].fe_index;
    return a >= 0 && b >= 0 && a < 4 && b < 4 && a != b && fes[a].size() == fes[b].size() &&
           (fes[a].array() == fes[b].array()).all();
}

AbsorptionResult absorb_mixed_network(const Eigen::Ref<const Eigen::VectorXd> &y,
                                      const Eigen::Ref<const Eigen::MatrixXd> &X,
                                      const std::vector<Eigen::VectorXi> &fes,
                                      const std::vector<HeterogeneousSlopeTerm> &slopes,
                                      const HdfeOptions &options, MixedNetworkMode mode, bool cuda,
                                      const std::vector<Eigen::VectorXi> *clusters,
                                      std::unique_ptr<MixedNetworkScores> *capture,
                                      bool csr_owner_requested, bool& csr_owner_used) {
    using namespace mixed_network;
    if (cuda && mode != MixedNetworkMode::Diagonal)
        throw MixedNetworkDecline("device_preconditioner");
    const auto start = Clock::now();
    const int blocks = deterministic_parallel_chunk_count(std::max<Eigen::Index>(1, y.size()));
    ParallelWorkObserver private_observer;
    rp::Hooks hooks;
    hooks.context = options.parallel_observer ? options.parallel_observer : &private_observer;
    hooks.begin = observer_begin;
    hooks.observe = observer_observe;
    hooks.end = observer_end;
    Team team(std::max(1, options.num_threads), hooks, 8 * blocks);
    std::vector<int> levels;
    const bool trace = mixed_network_trace_enabled();
    Data data = prepare(y, X, fes, slopes, team, blocks, levels);
    const auto coded = Clock::now();
    System g = assemble(data, team, blocks, mode == MixedNetworkMode::Forest,
                        csr_owner_requested, csr_owner_used);
    const auto assembled = Clock::now();
    if (g.dim == 0)
        throw MixedNetworkDecline("no_firm_contrasts");
    Preconditioner pc(mode);
    pc.prepare(g, team, blocks);
    const auto built = Clock::now();
    const double limit =
        std::max(std::min(options.tol, 1e-9), 64 * std::numeric_limits<double>::epsilon());
    const int budget = std::min(1500, options.max_iter);
    double check_seconds = 0.;
    int checks = 0, last_check = 0;
    double last_hint = std::numeric_limits<double>::infinity();
    AbsorptionResult accepted;
    auto should_check = [&](int it, const double *residual) {
        if (it == budget || mode == MixedNetworkMode::Direct)
            return true;
        const double hint = moment_hint(g, data, residual, team, blocks);
        if ((hint <= limit && (hint <= 0.5 * last_hint || it - last_check >= 32)) ||
            it - last_check >= 256 || !std::isfinite(hint)) {
            last_check = it;
            last_hint = hint;
            return true;
        }
        return false;
    };
    auto verify = [&](int it, const Matrix &state) {
        const auto t = Clock::now();
        auto out = recover(data, g, state, team, blocks, levels);
        out.iterations = it;
        out.converged = true;
        certify_absorption_result(y, X, fes, nullptr, options, slopes, out);
        check_seconds += std::chrono::duration<double>(Clock::now() - t).count();
        ++checks;
        if (trace)
            std::fprintf(stderr, "mixed_network_check iter=%d cert=%d block=%.8g target=%.8g\n", it,
                         int(out.precision_certified), out.slope_block_residual_rel, limit);
        if (!(out.precision_certified && out.slope_block_checked &&
              out.slope_block_residual_rel <= limit))
            return false;
        auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        out.slope_internal_tolerance = options.tol;
        if (cuda) {
            out.gpu_used = true;
            out.gpu_attempted = true;
            out.gpu_absorption_converged = true;
            out.gpu_absorption_iterations = it;
            out.gpu_status_code = 1;
        }
        if (trace) std::fprintf(stderr,
                     "mixed_network_accept mode=%d cuda=%d n=%d workers=%d firms=%d periods=%d "
                     "matches=%zu coeff=%d nnz=%ld iterations=%d passes=%d codes=%.5f "
                     "assembly=%.5f preconditioner=%.5f iterate=%.5f checks=%.5f\n",
                     int(mode), int(cuda), data.n, data.nw, data.nf, data.nt, data.match.size(),
                     g.dim, long(g.H.nonZeros()), it, checks, sec(start, coded),
                     sec(coded, assembled), sec(assembled, built),
                     sec(built, Clock::now()) - check_seconds, check_seconds);
        if (capture && clusters) {
            const auto score_start = Clock::now();
            retain_score_geometry(data, *clusters, options, team, blocks, *capture);
            if (trace) {
                std::fprintf(stderr, "mixed_network_score_geometry retained=%d seconds=%.5f\n",
                             int(bool(*capture)), sec(score_start, Clock::now()));
            }
        }
        accepted = std::move(out);
        return true;
    };
    if (cuda) {
#ifdef HDFE_USE_CUDA
        static_assert(sizeof(Block) == 9 * sizeof(double));
        NetworkDeviceInput input{g.dim,
                                 int(g.H.nonZeros()),
                                 data.ny,
                                 data.k,
                                 data.nf,
                                 budget,
                                 g.H.outerIndexPtr(),
                                 g.H.innerIndexPtr(),
                                 g.index.front().data(),
                                 g.H.valuePtr(),
                                 g.K.data(),
                                 g.inv_blocks.front().data(),
                                 g.rhs.data()};
        struct Call {
            decltype(should_check) &probe;
            decltype(verify) &certify;
            int dim, k;
        };
        Call call{should_check, verify, g.dim, data.k};
        const int done = solve_network_device(
            input, &call,
            [](void *v, int i, const double *r) { return static_cast<Call *>(v)->probe(i, r); },
            [](void *v, int i, const double *x) {
                auto &c = *static_cast<Call *>(v);
                return c.certify(i, Eigen::Map<const Matrix>(x, c.dim, c.k));
            });
        if (done <= 0)
            throw MixedNetworkDecline(done == 0 ? "device_memory" : "original_certificate", -done);
        return accepted;
#else
        throw MixedNetworkDecline("cpu_build");
#endif
    }
    Matrix state = Matrix::Zero(g.dim, data.k);
    Matrix r = g.rhs, p(g.dim, data.k), z(g.dim, data.k), q(g.dim, data.k);
    pc.apply(g, r, z, team, blocks);
    p = z;
    Eigen::RowVectorXd rho = dots(r, z, team, blocks);
    std::vector<unsigned char> live(data.k, 1);
    int spent = 0;
    for (int it = 1; it <= budget; ++it) {
        spent = it;
        multiply(g, p, q, team, blocks);
        auto denom = dots(p, q, team, blocks);
        Eigen::RowVectorXd alpha = Eigen::RowVectorXd::Zero(data.k);
        for (int c = 0; c < data.k; ++c)
            if (live[c]) {
                if (rho[c] > 0 && denom[c] > 0 && std::isfinite(rho[c] / denom[c]))
                    alpha[c] = rho[c] / denom[c];
                else
                    live[c] = 0;
            }
        int parts = std::min(g.dim, blocks);
        team.run(parts, [&](int b) {
            int i = std::int64_t(g.dim) * b / parts, e = std::int64_t(g.dim) * (b + 1) / parts;
            state.middleRows(i, e - i).array() +=
                p.middleRows(i, e - i).array().rowwise() * alpha.array();
            r.middleRows(i, e - i).array() -=
                q.middleRows(i, e - i).array().rowwise() * alpha.array();
        });
        const bool no_lanes =
            std::none_of(live.begin(), live.end(), [](unsigned char v) { return v != 0; });
        const bool probe =
            (mode == MixedNetworkMode::Direct ? (it <= 4 || it % 16 == 0) : (it % 16 == 0)) ||
            it == budget;
        if ((no_lanes || (probe && should_check(it, r.data()))) && verify(it, state))
            return accepted;
        if (no_lanes)
            break;
        pc.apply(g, r, z, team, blocks);
        auto next = dots(r, z, team, blocks);
        Eigen::RowVectorXd beta = Eigen::RowVectorXd::Zero(data.k);
        for (int c = 0; c < data.k; ++c)
            if (live[c]) {
                if (next[c] > 0 && std::isfinite(next[c] / rho[c]))
                    beta[c] = next[c] / rho[c];
                else
                    live[c] = 0;
            }
        team.run(parts, [&](int b) {
            int i = std::int64_t(g.dim) * b / parts, e = std::int64_t(g.dim) * (b + 1) / parts;
            p.middleRows(i, e - i).array() =
                z.middleRows(i, e - i).array() +
                p.middleRows(i, e - i).array().rowwise() * beta.array();
        });
        rho = next;
    }
    throw MixedNetworkDecline("original_certificate", spent);
}
} // namespace hdfe::detail
