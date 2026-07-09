// See include/schwarz_demean.hpp. Standalone (Eigen-free) Schwarz/approx-Cholesky PCG demean,
// generalized to k-way FE: bipartite pair = the two highest-cardinality dimensions, the rest
// diagonal-preconditioned inside the Krylov iteration. Ported from the validated prototype
// (_akm_opt/schwarz_proto/schwarz7.cpp + schwarz8.cpp). Not yet wired into absorb_fixed_effects.

#include "schwarz_demean.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xhdfe {
namespace {

bool finite_bits(double x) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

template <int S>
bool schwarz_impl(double* Y, double* X, int64_t n, int k,
                  const std::vector<const int32_t*>& fe, const std::vector<int>& nlev,
                  double tol, int max_iter, int* iters_out, int collapse_mode) {
    const int ndim = (int)fe.size();
    const int NC = k + 1;
    if (NC > S) return false;

    // --- choose bipartite pair = two largest-cardinality dims; rest are "extras" ---
    std::vector<int> dorder(ndim);
    std::iota(dorder.begin(), dorder.end(), 0);
    std::sort(dorder.begin(), dorder.end(), [&](int a, int b) { return nlev[a] > nlev[b]; });
    const int pa = dorder[0], pb = dorder[1];   // pa = "indiv" side, pb = "firm" side
    std::vector<int> extras(dorder.begin() + 2, dorder.end());
    const int64_t na = nlev[pa], nf = nlev[pb], N = na + nf;
    const int32_t* FA = fe[pa];
    const int32_t* FB = fe[pb];
    // extra-dim coefficient offsets into a flat extra array
    std::vector<int64_t> eoff(extras.size() + 1, 0);
    for (size_t e = 0; e < extras.size(); ++e) eoff[e + 1] = eoff[e] + nlev[extras[e]];
    const int64_t totE = eoff[extras.size()];

    // collapse heuristic: collapse when the smaller bipartite side is large (fill explodes)
    const bool collapse = collapse_mode < 0 ? (std::min(na, nf) > 50000) : (collapse_mode != 0);
    const int rho = 1;

    // --- aggregate pa-pb bipartite edges (weight = #obs) ---
    std::vector<int64_t> key(n);
    for (int64_t i = 0; i < n; ++i) key[i] = (int64_t)FA[i] * nf + FB[i];
    std::sort(key.begin(), key.end());
    std::vector<int32_t> eu, ev; std::vector<double> ew;
    for (int64_t i = 0; i < n;) { int64_t kk = key[i]; int64_t j = i; while (j < n && key[j] == kk) ++j;
        eu.push_back((int)(kk / nf)); ev.push_back((int)(na + kk % nf)); ew.push_back((double)(j - i)); i = j; }
    { std::vector<int64_t>().swap(key); }
    const int64_t M0 = (int64_t)ew.size();

    // --- approx-Cholesky of the bipartite Laplacian (multigraph or collapsed) ---
    std::vector<int32_t> EU, EV; std::vector<double> EW; std::vector<uint8_t> dead;
    EU.reserve(M0 * 2); EV.reserve(M0 * 2); EW.reserve(M0 * 2); dead.reserve(M0 * 2);
    std::vector<std::vector<int64_t>> inc(N);
    for (int64_t e = 0; e < M0; ++e) { EU.push_back(eu[e]); EV.push_back(ev[e]); EW.push_back(ew[e]); dead.push_back(0);
        inc[eu[e]].push_back(e); inc[ev[e]].push_back(e); }
    std::vector<int32_t> order(N); for (int64_t i = 0; i < N; ++i) order[i] = (int32_t)i;
    std::mt19937 rng(12345);
    std::vector<uint32_t> jit(N); for (int64_t i = 0; i < N; ++i) jit[i] = rng();
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        size_t da = inc[a].size(), db = inc[b].size(); if (da != db) return da < db; return jit[a] < jit[b]; });
    std::vector<int32_t> pos(N); for (int64_t p = 0; p < N; ++p) pos[order[p]] = (int32_t)p;
    std::vector<uint8_t> elim(N, 0); std::vector<double> cdiag(N, 1.0);
    std::vector<int64_t> cptr(N + 1, 0);
    std::vector<int32_t> cnbr; std::vector<double> cval; cnbr.reserve(M0 * (2 + rho)); cval.reserve(M0 * (2 + rho));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> sacc(collapse ? N : 0, 0.0);
    std::vector<int32_t> nbu; std::vector<double> nbw, cum;
    for (int64_t p = 0; p < N; ++p) { int v = order[p]; elim[v] = 1; nbu.clear(); nbw.clear();
        if (collapse) {
            for (int64_t e : inc[v]) { if (dead[e]) continue; int u = (EU[e] == v) ? EV[e] : EU[e]; if (elim[u]) continue;
                dead[e] = 1; if (sacc[u] == 0.0) nbu.push_back(u); sacc[u] += EW[e]; }
            for (int32_t u : nbu) { nbw.push_back(sacc[u]); sacc[u] = 0.0; }
        } else {
            for (int64_t e : inc[v]) { if (dead[e]) continue; int u = (EU[e] == v) ? EV[e] : EU[e]; if (elim[u]) continue;
                dead[e] = 1; nbu.push_back(u); nbw.push_back(EW[e]); }
        }
        double d = 0; for (double w : nbw) d += w; cptr[p + 1] = cptr[p];
        if (d <= 0) { cdiag[v] = 1.0; continue; }
        double ld = std::sqrt(d); cdiag[v] = ld;
        for (size_t i = 0; i < nbu.size(); ++i) { cnbr.push_back(nbu[i]); cval.push_back(-nbw[i] / ld); }
        cptr[p + 1] = cptr[p] + (int64_t)nbu.size();
        size_t deg = nbu.size();
        if (deg >= 2) { cum.resize(deg); double c = 0; for (size_t i = 0; i < deg; ++i) { c += nbw[i]; cum[i] = c; }
            for (size_t i = 0; i < deg; ++i) for (int s = 0; s < rho; ++s) {
                double r = U(rng) * d; size_t j = std::lower_bound(cum.begin(), cum.end(), r) - cum.begin();
                if (j >= deg) j = deg - 1; if (j == i) j = (i + 1) % deg; int ui = nbu[i], uj = nbu[j]; if (ui == uj) continue;
                double wij = nbw[i] * 0.5 / rho;
                int64_t ne = (int64_t)EU.size(); EU.push_back(ui); EV.push_back(uj); EW.push_back(wij); dead.push_back(0);
                inc[ui].push_back(ne); inc[uj].push_back(ne); } }
    }
    const int64_t FNNZ = (int64_t)cnbr.size();
    { std::vector<std::vector<int64_t>>().swap(inc); std::vector<int32_t>().swap(EU); std::vector<int32_t>().swap(EV);
      std::vector<double>().swap(EW); std::vector<uint8_t>().swap(dead); std::vector<uint8_t>().swap(elim); std::vector<uint32_t>().swap(jit); }

    // factor transpose + level schedule
    std::vector<int64_t> tptr(N + 1, 0);
    for (int64_t t = 0; t < FNNZ; ++t) tptr[cnbr[t] + 1]++;
    for (int64_t i = 0; i < N; ++i) tptr[i + 1] += tptr[i];
    std::vector<int32_t> tnode(FNNZ); std::vector<double> tval(FNNZ);
    { std::vector<int64_t> cur(tptr.begin(), tptr.end());
      for (int64_t p = 0; p < N; ++p) { int src = order[p];
        for (int64_t t = cptr[p]; t < cptr[p + 1]; ++t) { int v = cnbr[t]; tnode[cur[v]] = src; tval[cur[v]++] = cval[t]; } } }
    std::vector<int32_t> flev(N, 0), blev(N, 0); int nfl = 0, nbl = 0;
    for (int64_t p = 0; p < N; ++p) { int v = order[p]; int L = 0;
        for (int64_t t = tptr[v]; t < tptr[v + 1]; ++t) { int s = tnode[t]; if (flev[s] + 1 > L) L = flev[s] + 1; } flev[v] = L; if (L + 1 > nfl) nfl = L + 1; }
    for (int64_t p = N - 1; p >= 0; --p) { int v = order[p]; int L = 0;
        for (int64_t t = cptr[p]; t < cptr[p + 1]; ++t) { int u = cnbr[t]; if (blev[u] + 1 > L) L = blev[u] + 1; } blev[v] = L; if (L + 1 > nbl) nbl = L + 1; }
    auto bucketize = [&](const std::vector<int32_t>& lev, int nlv, std::vector<int64_t>& lptr, std::vector<int32_t>& lnodes) {
        lptr.assign(nlv + 1, 0); for (int64_t i = 0; i < N; ++i) lptr[lev[i] + 1]++;
        for (int i = 0; i < nlv; ++i) lptr[i + 1] += lptr[i]; lnodes.resize(N);
        std::vector<int64_t> cur(lptr.begin(), lptr.end());
        for (int64_t p = 0; p < N; ++p) { int v = order[p]; lnodes[cur[lev[v]]++] = v; } };
    std::vector<int64_t> fptr, bptr; std::vector<int32_t> fnodes, bnodes;
    bucketize(flev, nfl, fptr, fnodes); bucketize(blev, nbl, bptr, bnodes);

    // diagonal degrees for extra dims
    std::vector<double> degE(totE, 0.0);
    for (size_t e = 0; e < extras.size(); ++e) { const int32_t* fz = fe[extras[e]]; int64_t off = eoff[e];
        for (int64_t i = 0; i < n; ++i) degE[off + fz[i]] += 1.0; }

    // CSR obs-lists per node (pa, pb, each extra) for the race-free gather matvec
    auto build_csr = [&](const int32_t* fz, int64_t lv, std::vector<int64_t>& ptr, std::vector<int32_t>& idx) {
        ptr.assign(lv + 1, 0); for (int64_t i = 0; i < n; ++i) ptr[fz[i] + 1]++;
        for (int64_t g = 0; g < lv; ++g) ptr[g + 1] += ptr[g]; idx.resize(n);
        std::vector<int64_t> cur(ptr.begin(), ptr.end());
        for (int64_t i = 0; i < n; ++i) idx[cur[fz[i]]++] = (int32_t)i; };
    std::vector<int64_t> aptr, bptr2; std::vector<int32_t> aidx, bidx;
    build_csr(FA, na, aptr, aidx); build_csr(FB, nf, bptr2, bidx);
    std::vector<std::vector<int64_t>> eptr(extras.size()); std::vector<std::vector<int32_t>> eidx(extras.size());
    for (size_t e = 0; e < extras.size(); ++e) build_csr(fe[extras[e]], nlev[extras[e]], eptr[e], eidx[e]);
    std::vector<double> du((int64_t)n * NC);

    // --- column values per obs ---
    auto colval = [&](int64_t i, double* vv) { vv[0] = Y[i]; for (int c = 1; c < NC; ++c) vv[c] = X[(int64_t)(c - 1) * n + i]; };

    // --- operators ---
    auto chol_solve = [&](std::vector<double>& z) {
        #pragma omp parallel
        { for (int L = 0; L < nfl; ++L) { _Pragma("omp for schedule(static)")
            for (int64_t ix = fptr[L]; ix < fptr[L + 1]; ++ix) { int v = fnodes[ix]; double acc[S]; double* zv = &z[(int64_t)v * S];
                for (int c = 0; c < NC; ++c) acc[c] = zv[c];
                for (int64_t t = tptr[v]; t < tptr[v + 1]; ++t) { double val = tval[t]; const double* zs = &z[(int64_t)tnode[t] * S]; for (int c = 0; c < NC; ++c) acc[c] -= val * zs[c]; }
                double inv = 1.0 / cdiag[v]; for (int c = 0; c < NC; ++c) zv[c] = acc[c] * inv; } }
          for (int L = 0; L < nbl; ++L) { _Pragma("omp for schedule(static)")
            for (int64_t ix = bptr[L]; ix < bptr[L + 1]; ++ix) { int v = bnodes[ix]; int p = pos[v]; double acc[S]; double* zv = &z[(int64_t)v * S];
                for (int c = 0; c < NC; ++c) acc[c] = zv[c];
                for (int64_t t = cptr[p]; t < cptr[p + 1]; ++t) { double val = cval[t]; const double* zs = &z[(int64_t)cnbr[t] * S]; for (int c = 0; c < NC; ++c) acc[c] -= val * zs[c]; }
                double inv = 1.0 / cdiag[v]; for (int c = 0; c < NC; ++c) zv[c] = acc[c] * inv; } } }
    };
    std::vector<double> zsol((int64_t)N * S);
    auto psolve = [&](std::vector<double>& rN, std::vector<double>& rE) {
        #pragma omp parallel for schedule(static)
        for (int64_t a = 0; a < na; ++a) { double* zs = &zsol[a * S]; const double* rn = &rN[a * S]; for (int c = 0; c < NC; ++c) zs[c] = rn[c]; }
        #pragma omp parallel for schedule(static)
        for (int64_t fm = 0; fm < nf; ++fm) { double* zs = &zsol[(na + fm) * S]; const double* rn = &rN[(na + fm) * S]; for (int c = 0; c < NC; ++c) zs[c] = -rn[c]; }
        chol_solve(zsol);
        #pragma omp parallel for schedule(static)
        for (int64_t a = 0; a < na; ++a) { double* rn = &rN[a * S]; const double* zs = &zsol[a * S]; for (int c = 0; c < NC; ++c) rn[c] = zs[c]; }
        #pragma omp parallel for schedule(static)
        for (int64_t fm = 0; fm < nf; ++fm) { double* rn = &rN[(na + fm) * S]; const double* zs = &zsol[(na + fm) * S]; for (int c = 0; c < NC; ++c) rn[c] = -zs[c]; }
        #pragma omp parallel for schedule(static)
        for (int64_t g = 0; g < totE; ++g) { double inv = 1.0 / degE[g]; double* re = &rE[g * S]; for (int c = 0; c < NC; ++c) re[c] *= inv; }
        for (int c = 0; c < NC; ++c) { rN[na * S + c] = 0.0; }                    // pin firm0
        for (size_t e = 0; e < extras.size(); ++e) for (int c = 0; c < NC; ++c) rE[eoff[e] * S + c] = 0.0;  // pin each extra ref0
    };
    auto matvec = [&](const std::vector<double>& uN, const std::vector<double>& uE,
                      std::vector<double>& oN, std::vector<double>& oE) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) { const double* ua = &uN[(int64_t)FA[i] * S]; const double* uf = &uN[(int64_t)(na + FB[i]) * S]; double* d = &du[i * NC];
            for (int c = 0; c < NC; ++c) d[c] = ua[c] + uf[c];
            for (size_t e = 0; e < extras.size(); ++e) { const double* ue = &uE[(eoff[e] + fe[extras[e]][i]) * S]; for (int c = 0; c < NC; ++c) d[c] += ue[c]; } }
        #pragma omp parallel for schedule(static)
        for (int64_t a = 0; a < na; ++a) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
            for (int64_t j = aptr[a]; j < aptr[a + 1]; ++j) { const double* d = &du[(int64_t)aidx[j] * NC]; for (int c = 0; c < NC; ++c) acc[c] += d[c]; }
            double* o = &oN[a * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; }
        #pragma omp parallel for schedule(static)
        for (int64_t fm = 0; fm < nf; ++fm) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
            for (int64_t j = bptr2[fm]; j < bptr2[fm + 1]; ++j) { const double* d = &du[(int64_t)bidx[j] * NC]; for (int c = 0; c < NC; ++c) acc[c] += d[c]; }
            double* o = &oN[(na + fm) * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; }
        for (size_t e = 0; e < extras.size(); ++e) {
            #pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < nlev[extras[e]]; ++g) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
                for (int64_t j = eptr[e][g]; j < eptr[e][g + 1]; ++j) { const double* d = &du[(int64_t)eidx[e][j] * NC]; for (int c = 0; c < NC; ++c) acc[c] += d[c]; }
                double* o = &oE[(eoff[e] + g) * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; } }
        for (int c = 0; c < NC; ++c) oN[na * S + c] = 0.0;
        for (size_t e = 0; e < extras.size(); ++e) for (int c = 0; c < NC; ++c) oE[eoff[e] * S + c] = 0.0;
    };
    auto dotcol = [&](const std::vector<double>& a, const std::vector<double>& ae,
                      const std::vector<double>& b, const std::vector<double>& be, double* out) {
        double s[S] = {0};
#if defined(_MSC_VER)
        // MSVC's OpenMP does not support array-section reductions (reduction(+ : s[:S]));
        // use per-thread partial buffers instead (numerically equivalent, race-free).
#ifdef _OPENMP
        const int nthreads = std::max(1, omp_get_max_threads());
        std::vector<double> partial(static_cast<std::size_t>(nthreads) * S, 0.0);
#pragma omp parallel
        {
            double* local = partial.data() + static_cast<std::size_t>(omp_get_thread_num()) * S;
#pragma omp for schedule(static)
            for (int64_t v = 0; v < N; ++v) { const double* av = &a[v * S]; const double* bv = &b[v * S]; for (int c = 0; c < NC; ++c) local[c] += av[c] * bv[c]; }
#pragma omp for schedule(static)
            for (int64_t g = 0; g < totE; ++g) { const double* av = &ae[g * S]; const double* bv = &be[g * S]; for (int c = 0; c < NC; ++c) local[c] += av[c] * bv[c]; }
        }
        for (int t = 0; t < nthreads; ++t) {
            const double* local = partial.data() + static_cast<std::size_t>(t) * S;
            for (int c = 0; c < NC; ++c) s[c] += local[c];
        }
#else
        for (int64_t v = 0; v < N; ++v) { const double* av = &a[v * S]; const double* bv = &b[v * S]; for (int c = 0; c < NC; ++c) s[c] += av[c] * bv[c]; }
        for (int64_t g = 0; g < totE; ++g) { const double* av = &ae[g * S]; const double* bv = &be[g * S]; for (int c = 0; c < NC; ++c) s[c] += av[c] * bv[c]; }
#endif
#else
        // Non-MSVC (GCC/Clang): unchanged from upstream.
        #pragma omp parallel for reduction(+ : s[:S]) schedule(static)
        for (int64_t v = 0; v < N; ++v) { const double* av = &a[v * S]; const double* bv = &b[v * S]; for (int c = 0; c < NC; ++c) s[c] += av[c] * bv[c]; }
        for (int64_t g = 0; g < totE; ++g) { const double* av = &ae[g * S]; const double* bv = &be[g * S]; for (int c = 0; c < NC; ++c) s[c] += av[c] * bv[c]; }
#endif
        for (int c = 0; c < NC; ++c) out[c] = s[c];
    };

    // --- block PCG ---
    std::vector<double> bN((int64_t)N * S, 0), bE((int64_t)totE * S, 0), xN((int64_t)N * S, 0), xE((int64_t)totE * S, 0),
        rN((int64_t)N * S), rE((int64_t)totE * S), zN((int64_t)N * S), zE((int64_t)totE * S),
        pN((int64_t)N * S), pE((int64_t)totE * S), ApN((int64_t)N * S), ApE((int64_t)totE * S);
    // rhs = D'v via the CSR gather (parallel)
    #pragma omp parallel for schedule(static)
    for (int64_t a = 0; a < na; ++a) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
        for (int64_t j = aptr[a]; j < aptr[a + 1]; ++j) { double vv[S]; colval(aidx[j], vv); for (int c = 0; c < NC; ++c) acc[c] += vv[c]; }
        double* o = &bN[a * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; }
    #pragma omp parallel for schedule(static)
    for (int64_t fm = 0; fm < nf; ++fm) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
        for (int64_t j = bptr2[fm]; j < bptr2[fm + 1]; ++j) { double vv[S]; colval(bidx[j], vv); for (int c = 0; c < NC; ++c) acc[c] += vv[c]; }
        double* o = &bN[(na + fm) * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; }
    for (size_t e = 0; e < extras.size(); ++e) {
        #pragma omp parallel for schedule(static)
        for (int64_t g = 0; g < nlev[extras[e]]; ++g) { double acc[S]; for (int c = 0; c < NC; ++c) acc[c] = 0;
            for (int64_t j = eptr[e][g]; j < eptr[e][g + 1]; ++j) { double vv[S]; colval(eidx[e][j], vv); for (int c = 0; c < NC; ++c) acc[c] += vv[c]; }
            double* o = &bE[(eoff[e] + g) * S]; for (int c = 0; c < NC; ++c) o[c] = acc[c]; } }
    for (int c = 0; c < NC; ++c) { bN[na * S + c] = 0.0; }
    for (size_t e = 0; e < extras.size(); ++e) for (int c = 0; c < NC; ++c) bE[eoff[e] * S + c] = 0.0;

    constexpr double kTiny = 1e-300;
    double bnorm[S];
    dotcol(bN, bE, bN, bE, bnorm);
    bool active[S] = {false};
    int active_count = 0;
    for (int c = 0; c < NC; ++c) {
        if (!finite_bits(bnorm[c]) || bnorm[c] < 0.0) {
            return false;
        }
        bnorm[c] = std::sqrt(bnorm[c]);
        if (bnorm[c] > kTiny) {
            active[c] = true;
            ++active_count;
        }
    }
    if (active_count == 0) {
        if (iters_out) *iters_out = 0;
        return true;
    }
    // Common case: every RHS column has a non-zero projection. Then the per-column
    // active mask never changes and the hot update loops stay branchless / vectorizable
    // and bit-identical to the un-gated solver. The masked variants below are only taken
    // when a legitimate zero-RHS column must be skipped (kept at its already-solved zero).
    // NOTE: converged columns are NOT frozen mid-solve; like the reference solver we keep
    // iterating every active column until all meet tol in the same iteration, so the
    // returned within transform matches the reference iterate exactly.
    const bool all_active = (active_count == NC);
    rN = bN; rE = bE; zN = rN; zE = rE; psolve(zN, zE);
    pN = zN; pE = zE; double rz[S]; dotcol(rN, rE, zN, zE, rz);
    for (int c = 0; c < NC; ++c) {
        if (active[c] && (!finite_bits(rz[c]) || rz[c] <= kTiny)) {
            return false;
        }
    }
    int it = 0;
    bool converged = false;
    for (it = 1; it <= max_iter; ++it) {
        matvec(pN, pE, ApN, ApE); double pAp[S]; dotcol(pN, pE, ApN, ApE, pAp);
        double a[S] = {0};
        for (int c = 0; c < NC; ++c) {
            if (!active[c]) {
                continue;
            }
            if (!finite_bits(pAp[c]) || pAp[c] <= kTiny) {
                return false;
            }
            a[c] = rz[c] / pAp[c];
            if (!finite_bits(a[c])) {
                return false;
            }
        }
        if (all_active) {
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; ++v) { double* xv = &xN[v * S]; double* rv = &rN[v * S]; const double* pv = &pN[v * S]; const double* av = &ApN[v * S];
                for (int c = 0; c < NC; ++c) { xv[c] += a[c] * pv[c]; rv[c] -= a[c] * av[c]; } }
            #pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < totE; ++g) { double* xv = &xE[g * S]; double* rv = &rE[g * S]; const double* pv = &pE[g * S]; const double* av = &ApE[g * S];
                for (int c = 0; c < NC; ++c) { xv[c] += a[c] * pv[c]; rv[c] -= a[c] * av[c]; } }
        } else {
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; ++v) { double* xv = &xN[v * S]; double* rv = &rN[v * S]; const double* pv = &pN[v * S]; const double* av = &ApN[v * S];
                for (int c = 0; c < NC; ++c) if (active[c]) { xv[c] += a[c] * pv[c]; rv[c] -= a[c] * av[c]; } }
            #pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < totE; ++g) { double* xv = &xE[g * S]; double* rv = &rE[g * S]; const double* pv = &pE[g * S]; const double* av = &ApE[g * S];
                for (int c = 0; c < NC; ++c) if (active[c]) { xv[c] += a[c] * pv[c]; rv[c] -= a[c] * av[c]; } }
        }
        double rn[S]; dotcol(rN, rE, rN, rE, rn);
        int nconv = NC - active_count;
        for (int c = 0; c < NC; ++c) {
            if (!active[c]) {
                continue;
            }
            if (!finite_bits(rn[c]) || rn[c] < -kTiny) {
                return false;
            }
            const double rn_nonneg = rn[c] > 0.0 ? rn[c] : 0.0;
            const double rel = std::sqrt(rn_nonneg) / bnorm[c];
            if (!finite_bits(rel)) {
                return false;
            }
            if (rel < tol) {
                ++nconv;
            }
        }
        if (nconv == NC) {
            converged = true;
            break;
        }
        zN = rN; zE = rE; psolve(zN, zE);
        double rz2[S]; dotcol(rN, rE, zN, zE, rz2); double beta[S] = {0};
        for (int c = 0; c < NC; ++c) {
            if (!active[c]) {
                continue;
            }
            if (!finite_bits(rz2[c]) || rz2[c] <= kTiny ||
                !finite_bits(rz[c]) || rz[c] <= kTiny) {
                return false;
            }
            beta[c] = rz2[c] / rz[c];
            if (!finite_bits(beta[c])) {
                return false;
            }
        }
        if (all_active) {
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; ++v) { double* pv = &pN[v * S]; const double* zv = &zN[v * S]; for (int c = 0; c < NC; ++c) pv[c] = zv[c] + beta[c] * pv[c]; }
            #pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < totE; ++g) { double* pv = &pE[g * S]; const double* zv = &zE[g * S]; for (int c = 0; c < NC; ++c) pv[c] = zv[c] + beta[c] * pv[c]; }
        } else {
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < N; ++v) { double* pv = &pN[v * S]; const double* zv = &zN[v * S]; for (int c = 0; c < NC; ++c) if (active[c]) pv[c] = zv[c] + beta[c] * pv[c]; }
            #pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < totE; ++g) { double* pv = &pE[g * S]; const double* zv = &zE[g * S]; for (int c = 0; c < NC; ++c) if (active[c]) pv[c] = zv[c] + beta[c] * pv[c]; }
        }
        for (int c = 0; c < NC; ++c) rz[c] = rz2[c];
    }
    if (iters_out) *iters_out = it;
    if (!converged) {
        return false;
    }

    // residual = v - D u  -> overwrite Y, X
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) { const double* xa = &xN[(int64_t)FA[i] * S]; const double* xf = &xN[(int64_t)(na + FB[i]) * S];
        double du_[S]; for (int c = 0; c < NC; ++c) du_[c] = xa[c] + xf[c];
        for (size_t e = 0; e < extras.size(); ++e) { const double* xe = &xE[(eoff[e] + fe[extras[e]][i]) * S]; for (int c = 0; c < NC; ++c) du_[c] += xe[c]; }
        Y[i] -= du_[0]; for (int c = 1; c < NC; ++c) X[(int64_t)(c - 1) * n + i] -= du_[c]; }
    (void)FNNZ;
    return true;
}

}  // namespace

bool schwarz_demean_raw(double* Y, double* X, int64_t n, int k,
                        const std::vector<const int32_t*>& fe, const std::vector<int>& nlev,
                        int threads, double tol, int max_iter, int* iters_out, int collapse) {
    if ((int)fe.size() < 2 || (int)nlev.size() != (int)fe.size() || n <= 0 || k < 0) return false;
    const int NC = k + 1;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    if (NC <= 8) return schwarz_impl<8>(Y, X, n, k, fe, nlev, tol, max_iter, iters_out, collapse);
    // A k=10 (NC=11) design — the pyfixest difficult_*_3fe-class benchmark — would
    // otherwise pad every node block to a 16-wide stride and stream 5/16 dead lanes.
    // The 12-wide instantiation runs the identical column arithmetic (`for c < NC`)
    // over a tighter layout, so it is bit-for-bit equivalent to the 16-wide path while
    // touching ~25% less node-array bandwidth in the PCG axpy/copy sweeps.
    if (NC <= 12) return schwarz_impl<12>(Y, X, n, k, fe, nlev, tol, max_iter, iters_out, collapse);
    if (NC <= 16) return schwarz_impl<16>(Y, X, n, k, fe, nlev, tol, max_iter, iters_out, collapse);
    return false;  // too many columns for the current stride
}

}  // namespace xhdfe
