// Phase 5b of the Stata-path plan (DESENHO_FASE5B.md, round 23): the multi-way
// inference called directly, and fits through the public C++ entry. Built against
// the branch's core library with -DCSP_NEW_API (the carrier, hdfe::detail::
// ClusterPasses) and against the frozen one without it; run with
// XHDFE_CLUSTER_SHARED_PASSES unset and "0" (tests/cluster_shared_passes_inference.sh).
// Every "CMP" line must be identical in every arm; every "CHECK" line must end in ok.
// Synthetic data only (fixed LCG and fixed patterns).
//
//   A  run_ols_multiway and multiway_cluster_sandwich with identifiers INT_MIN,
//      INT_MAX and negatives (the public fit refuses negative codes before the OLS,
//      hr:10896), two and three dimensions, with and without weights; with the
//      carrier: the same bits, the counts of every mask against counts made here,
//      the pair against first-appearance codes made here
//   B  a failure in the LAST meat (mask 3: the pair's meat overflows while the
//      single dimensions cancel exactly), followed by the wide route
//   C  a failure in the final sandwich (ols.cpp:1764), followed by the wide route
//      B, C: the carrier is left empty (nothing published, the pair freed)
//   D  consumers given a foreign, incomplete, not-ready or wrong-size carrier, and
//      a producer given a carrier already filled: today's bits, carrier untouched
//   E  fits with absorbed effects: one thread outside any region; inside
//      "#pragma omp parallel" (explicit one thread, and the automatic request);
//      two concurrent fits on separate regressors, equal to the one inside a region
//      (a fit inside a region takes its in-region path today: not equal to outside)
//   F  one cluster list whose CONTENT changes at the same address between fits of
//      one regressor, and changes back
//   G  success, failure, success on one regressor
//   H  the inference on frozen post-absorption inputs: one absorption (on the GPU
//      for a CUDA build) dumped by CSP_DUMP and read by every arm (CSP_FROZEN_INPUT)
#include "ols.hpp"
#include "hdfe/hdfe_regressor_v11.hpp"

#include <omp.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using hdfe::detail::OlsResult;
using hdfe::v11::HdfeRegressorV11;
using Clusters = std::vector<Eigen::VectorXi>;

int failures = 0;

std::uint64_t fnv(const void* data, std::size_t bytes, std::uint64_t h = 1469598103934665603ULL) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <class M>
std::uint64_t hash_of(const M& m, std::uint64_t h = 1469598103934665603ULL) {
    return fnv(m.data(), sizeof(typename M::Scalar) * static_cast<std::size_t>(m.size()), h);
}

void check(const std::string& what, bool ok) {
    std::printf("CHECK %s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

bool switch_off() {
    const char* raw = std::getenv("XHDFE_CLUSTER_SHARED_PASSES");
    return raw && std::strcmp(raw, "0") == 0;
}

struct Lcg {
    std::uint64_t s;
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

std::string ols_line(const char* tag, const OlsResult& r) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s coef %016llx cov %016llx resid %016llx num_clusters %d df %.17g", tag,
                  static_cast<unsigned long long>(hash_of(r.coefficients)),
                  static_cast<unsigned long long>(hash_of(r.covariance)),
                  static_cast<unsigned long long>(hash_of(r.residuals)), r.num_clusters, r.df_resid);
    return buf;
}

// Distinct tuples of the selected dimensions, masks 1 .. 2^m - 1 ascending.
std::vector<int> mask_counts_here(const Clusters& c) {
    const int m = static_cast<int>(c.size());
    std::vector<int> out;
    for (int mask = 1; mask < (1 << m); ++mask) {
        std::set<std::vector<int>> seen;
        for (int i = 0; i < c[0].size(); ++i) {
            std::vector<int> key;
            for (int d = 0; d < m; ++d)
                if (mask >> d & 1) key.push_back(c[static_cast<std::size_t>(d)](i));
            seen.insert(key);
        }
        out.push_back(static_cast<int>(seen.size()));
    }
    return out;
}

// combine_clusters() for {0, 1}: uint32 of each int32, concatenated, ids by first appearance.
Eigen::VectorXi pair_here(const Clusters& c) {
    std::unordered_map<std::uint64_t, int> ids;
    Eigen::VectorXi out(c[0].size());
    for (int i = 0; i < c[0].size(); ++i) {
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(c[0](i))) << 32) |
                                  static_cast<std::uint32_t>(c[1](i));
        out(i) = ids.emplace(key, static_cast<int>(ids.size())).first->second;
    }
    return out;
}

OlsResult ols(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Eigen::VectorXd* w, const Clusters& c,
              void* passes = nullptr) {
#ifdef CSP_NEW_API
    return hdfe::detail::run_ols_multiway(y, X, w, &c, hdfe::StandardErrorType::Cluster, 0.0, 0.0,
                                          hdfe::ClusterDofMethod::Conventional, true, -1.0, nullptr, false,
                                          nullptr, static_cast<hdfe::detail::ClusterPasses*>(passes));
#else
    (void)passes;
    return hdfe::detail::run_ols_multiway(y, X, w, &c, hdfe::StandardErrorType::Cluster, 0.0, 0.0,
                                          hdfe::ClusterDofMethod::Conventional, true, -1.0, nullptr, false,
                                          nullptr);
#endif
}

Eigen::MatrixXd sandwich(const OlsResult& r, const Eigen::MatrixXd& X, const Eigen::VectorXd* w, const Clusters& c,
                         const void* passes = nullptr) {
#ifdef CSP_NEW_API
    return hdfe::detail::multiway_cluster_sandwich(r.xtx_inv, X, r.residuals, w, c, hdfe::ClusterDofMethod::Min,
                                                   true, static_cast<const hdfe::detail::ClusterPasses*>(passes));
#else
    (void)passes;
    return hdfe::detail::multiway_cluster_sandwich(r.xtx_inv, X, r.residuals, w, c, hdfe::ClusterDofMethod::Min,
                                                   true);
#endif
}

// --- A: direct inference with extreme identifiers -------------------------------
void part_a() {
    const int n = 6000;
    Lcg g{20261115ULL};
    Eigen::VectorXd y(n), w(n);
    Eigen::MatrixXd X(n, 2);
    Clusters two(2, Eigen::VectorXi(n)), three(3, Eigen::VectorXi(n));
    const int specials[] = {INT_MIN, INT_MAX, -1, 0, INT_MIN + 1, INT_MAX - 1, -7, 1};
    for (int i = 0; i < n; ++i) {
        X(i, 0) = g.next() - 0.5;
        X(i, 1) = g.next() - 0.5 + 0.1 * (i % 7);
        y(i) = 0.5 * X(i, 0) - 0.25 * X(i, 1) + 0.3 * (g.next() - 0.5) + 0.01 * (i % 13);
        w(i) = 0.5 + g.next();
        // Dimension 0: runs of 6 rows over 1000 ids, a few of them the extremes.
        const int run = i / 6;
        two[0](i) = run % 97 == 0 ? specials[(run / 97) % 8] : run * 3 - 1500;
        // Dimension 1: 40 ids interleaved, negative and extreme ones included.
        two[1](i) = (i % 40 < 8) ? specials[i % 8] : -(i % 40) * 1000003;
        three[0](i) = two[0](i);
        three[1](i) = two[1](i);
        three[2](i) = (i % 11 == 0) ? INT_MIN : (i % 5) - 2;
    }
    struct Case { const char* tag; const Clusters* c; const Eigen::VectorXd* w; };
    const Case cases[] = {{"A_two", &two, nullptr}, {"A_two_w", &two, &w}, {"A_three", &three, nullptr},
                          {"A_three_w", &three, &w}};
    for (const Case& k : cases) {
        try {
            const OlsResult r = ols(y, X, k.w, *k.c);
            std::printf("CMP %s\n", ols_line(k.tag, r).c_str());
            const Eigen::MatrixXd s = sandwich(r, X, k.w, *k.c);
            std::printf("CMP %s sandwich %016llx\n", k.tag, static_cast<unsigned long long>(hash_of(s)));
#ifdef CSP_NEW_API
            hdfe::detail::ClusterPasses passes;
            passes.want_pair = k.c->size() == 2;
            const OlsResult rp = ols(y, X, k.w, *k.c, &passes);
            check(std::string(k.tag) + " producer same bits", ols_line(k.tag, rp) == ols_line(k.tag, r));
            if (switch_off()) {
                check(std::string(k.tag) + " off leaves the carrier empty",
                      !passes.ready && passes.mask_counts.empty() && passes.pair.size() == 0);
            } else {
                check(std::string(k.tag) + " carrier ready from this list", passes.ready && passes.source == k.c);
                check(std::string(k.tag) + " carrier counts = counts here", passes.mask_counts == mask_counts_here(*k.c));
                if (k.c->size() == 2) {
                    const Eigen::VectorXi here = pair_here(*k.c);
                    check(std::string(k.tag) + " carrier pair = first appearance here",
                          passes.pair.size() == here.size() && passes.pair == here);
                } else {
                    check(std::string(k.tag) + " no pair kept for three dimensions", passes.pair.size() == 0);
                }
            }
            const Eigen::MatrixXd sp = sandwich(r, X, k.w, *k.c, &passes);
            check(std::string(k.tag) + " consumer same bits", hash_of(sp) == hash_of(s));
#endif
        } catch (const std::exception& e) {
            std::printf("CMP %s error %s\n", k.tag, e.what());
        }
    }
}

// --- B, C: failures after the pair is built, then the wide route ------------------
// 2 x 2 cells of 8 rows, residuals +S, -S, -S, +S cell by cell (y itself: the
// design's only column is constant, X'y = 0, beta = 0 exactly). The single
// dimensions sum to exactly zero; the pair's scores are +-8 S (powers of two).
void failure_case(const char* tag, double scale_x, double s, bool publishes) {
    const int n = 32;
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Constant(n, 1, scale_x);
    Clusters c(2, Eigen::VectorXi(n));
    for (int i = 0; i < n; ++i) {
        const int a = i / 16, b = (i / 8) % 2;
        c[0](i) = a;
        c[1](i) = b;
        y(i) = (a == b) ? s : -s;
    }
    try {
        const OlsResult r = ols(y, X, nullptr, c);
        std::printf("CMP %s\n", ols_line(tag, r).c_str());
    } catch (const std::exception& e) {
        std::printf("CMP %s error %s\n", tag, e.what());
    }
#ifdef CSP_NEW_API
    hdfe::detail::ClusterPasses passes;
    passes.want_pair = true;
    std::string outcome;
    try {
        outcome = ols_line(tag, ols(y, X, nullptr, c, &passes));
    } catch (const std::exception& e) {
        outcome = std::string("error ") + e.what();
    }
    if (switch_off() || !publishes) {
        check(std::string(tag) + (publishes ? " off leaves the carrier empty" : " nothing published, the pair freed"),
              !passes.ready && passes.source == nullptr && passes.mask_counts.empty() && passes.pair.size() == 0);
    } else {
        // The companion differs in one scale only and passes every check: the same
        // structure publishes counts {2, 2, 4} and a pair of n rows, which pins the
        // failure of its twin to the pair's meat (B) or to the final sandwich (C).
        check(std::string(tag) + " publishes counts {2,2,4} and the pair",
              passes.ready && passes.mask_counts == std::vector<int>{2, 2, 4} && passes.pair.size() == n &&
              passes.pair == pair_here(c));
    }
#endif
}

// --- D: consumers with bad carriers, a producer with a filled one ------------------
void part_d() {
#ifdef CSP_NEW_API
    if (switch_off()) return;
    const int n = 4000;
    Lcg g{77ULL};
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X(n, 2);
    Clusters c(2, Eigen::VectorXi(n));
    for (int i = 0; i < n; ++i) {
        X(i, 0) = g.next();
        X(i, 1) = g.next() + 0.01 * (i % 5);
        y(i) = X(i, 0) - X(i, 1) + g.next();
        c[0](i) = i / 5;
        c[1](i) = static_cast<int>(g.next() * 30) * 101;
    }
    const OlsResult r = ols(y, X, nullptr, c);
    hdfe::detail::ClusterPasses good;
    good.want_pair = true;
    (void)ols(y, X, nullptr, c, &good);
    check("D good carrier produced", good.ready && good.pair.size() == n);
    const Clusters other = c;  // the same content at another address
    hdfe::detail::ClusterPasses foreign = good, incomplete = good, unready = good, wrong = good;
    foreign.source = &other;
    incomplete.mask_counts.pop_back();
    unready.ready = false;
    wrong.pair.conservativeResize(n - 1);
    // Poisoned pairs: one cluster for every row. A consumer that read them would
    // change the bits; the wrong-size pair cannot be poisoned that way, so the driver
    // pins the order of the reconstruction's lines between the two markers below:
    // recount/built four times (no carrier, foreign, incomplete, not ready), then
    // carrier/built (wrong size: the counts only), then carrier/shared (good).
    foreign.pair.setZero();
    incomplete.pair.setZero();
    unready.pair.setZero();
    std::fflush(stdout);
    std::fprintf(stderr, "CSP_PART_D_BEGIN\n");
    const std::uint64_t today = hash_of(sandwich(r, X, nullptr, c));
    check("D foreign carrier, today's bits", hash_of(sandwich(r, X, nullptr, c, &foreign)) == today);
    check("D incomplete carrier, today's bits", hash_of(sandwich(r, X, nullptr, c, &incomplete)) == today);
    check("D not-ready carrier, today's bits", hash_of(sandwich(r, X, nullptr, c, &unready)) == today);
    check("D wrong-size pair, today's bits", hash_of(sandwich(r, X, nullptr, c, &wrong)) == today);
    check("D good carrier, today's bits", hash_of(sandwich(r, X, nullptr, c, &good)) == today);
    std::fprintf(stderr, "CSP_PART_D_END\n");
    hdfe::detail::ClusterPasses poisoned = good;
    poisoned.pair.setZero();
    check("D a poisoned good carrier changes the bits (the pair is read)",
          hash_of(sandwich(r, X, nullptr, c, &poisoned)) != today);
    const std::vector<int> counts = good.mask_counts;
    const Eigen::VectorXi pair = good.pair;
    const OlsResult again = ols(y, X, nullptr, c, &good);
    check("D a filled carrier is not produced again", good.ready && good.mask_counts == counts && good.pair == pair);
    check("D the producer with a filled carrier, today's bits", ols_line("D", again) == ols_line("D", r));
#endif
}

// --- E, F, G: fits through the public entry ------------------------------------------
struct Panel {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    Clusters clusters;
    Eigen::VectorXi region;
};

Panel make_panel() {
    Lcg g{20261116ULL};
    const int workers = 20000, firms = 1500, years = 8;
    std::vector<int> wk, fm, yr;
    for (int w = 0; w < workers; ++w) {
        const int rows = 3 + static_cast<int>(g.next() * 6.0);
        const int home = static_cast<int>(g.next() * firms);
        const int start = static_cast<int>(g.next() * years);
        for (int r = 0; r < rows; ++r) {
            wk.push_back(w);
            fm.push_back(g.next() < 0.15 ? static_cast<int>(g.next() * firms) : home);
            yr.push_back((start + r) % years);
        }
    }
    const int n = static_cast<int>(wk.size());
    const long long K = static_cast<long long>(8.7 * n / firms);
    Panel p;
    p.y.resize(n);
    p.X.resize(n, 2);
    Eigen::VectorXi worker(n), firm(n), year(n), firm_sparse(n);
    p.region.resize(n);
    for (int i = 0; i < n; ++i) {
        worker(i) = wk[static_cast<std::size_t>(i)];
        firm(i) = fm[static_cast<std::size_t>(i)];
        year(i) = yr[static_cast<std::size_t>(i)];
        firm_sparse(i) = static_cast<int>(37 + static_cast<long long>(firm(i)) * K);
        p.region(i) = firm(i) / 30;
        p.X(i, 0) = g.next() - 0.5;
        p.X(i, 1) = g.next() - 0.5 + 0.1 * (firm(i) % 7);
        p.y(i) = 0.5 * p.X(i, 0) - 0.25 * p.X(i, 1) + 0.3 * (g.next() - 0.5) + 0.01 * (firm(i) % 13) +
                 0.001 * (worker(i) % 17) + 0.02 * year(i);
    }
    p.fes = {worker, firm, year};
    p.clusters = {worker, firm_sparse};
    return p;
}

hdfe::HdfeOptions options_for(int threads) {
    hdfe::HdfeOptions o;
    o.se_type = hdfe::StandardErrorType::Cluster;
    o.num_threads = threads;
    return o;
}

std::string fit_line(const char* tag, const HdfeRegressorV11& reg) {
    const hdfe::HdfeResults& r = reg.results();
    std::uint64_t counts = fnv(r.cluster_counts.data(), sizeof(int) * r.cluster_counts.size());
    counts = fnv(r.cluster_combo_counts.data(), sizeof(int) * r.cluster_combo_counts.size(), counts);
    char buf[320];
    std::snprintf(buf, sizeof buf, "%s coef %016llx cov %016llx se %016llx resid %016llx counts %016llx "
                  "num_clusters %d df %.17g", tag,
                  static_cast<unsigned long long>(hash_of(r.coefficients)),
                  static_cast<unsigned long long>(hash_of(r.covariance)),
                  static_cast<unsigned long long>(hash_of(r.std_errors)),
                  static_cast<unsigned long long>(hash_of(r.residuals)),
                  static_cast<unsigned long long>(counts), r.num_clusters, r.df_resid);
    return buf;
}

std::string run(const char* tag, HdfeRegressorV11& reg, const Panel& p, const Clusters& c) {
    try {
        reg.fit(p.y, p.X, p.fes, nullptr, &c, nullptr, {}, nullptr);
        return fit_line(tag, reg) + " threads_used " + std::to_string(reg.threads_used());
    } catch (const std::exception& e) {
        return std::string(tag) + " error " + e.what();
    }
}

// The fit line without its tag and without threads_used (compared across calls).
std::string body(const std::string& line) {
    const auto first = line.find(' ');
    const auto t = line.find(" threads_used");
    return line.substr(first, t == std::string::npos ? std::string::npos : t - first);
}

void part_efg(const Panel& p) {
    HdfeRegressorV11 serial(options_for(1));
    const std::string ref = run("E_serial_t1", serial, p, p.clusters);
    std::printf("CMP %s\n", ref.c_str());
    std::string inside, inside_auto, first, second;
    int active = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            active = omp_in_parallel();
            HdfeRegressorV11 a(options_for(1)), b(options_for(0));
            inside = run("E_inside_t1", a, p, p.clusters);
            inside_auto = run("E_inside_auto", b, p, p.clusters);
        }
    }
    std::printf("CMP %s\nCMP %s\n", inside.c_str(), inside_auto.c_str());
    check("E region active", active == 1);
    // Inside an active region the fit takes its in-region path (today, every arm):
    // its bits are compared across arms (CMP), not with the fit outside.
#pragma omp parallel num_threads(2)
    {
        HdfeRegressorV11 own(options_for(1));
        const std::string line = run("E_concurrent", own, p, p.clusters);
#pragma omp critical
        {
            if (omp_get_thread_num() == 0) first = line; else second = line;
        }
    }
    std::printf("CMP %s\n", first.c_str());
    check("E two concurrent fits = the fit inside a region", body(first) == body(inside) && body(second) == body(inside));

    // F: the same list object, its content changed in place and changed back.
    Clusters list = p.clusters;
    HdfeRegressorV11 reg(options_for(1));
    const std::string f1 = run("F_list_1", reg, p, list);
    const int* address = list[1].data();
    const Eigen::VectorXi saved = list[1];
    list[1] = p.region;  // same size: the same buffer
    check("F the same address", list[1].data() == address);
    const std::string f2 = run("F_list_2", reg, p, list);
    HdfeRegressorV11 fresh(options_for(1));
    const std::string f2_fresh = run("F_fresh_2", fresh, p, list);
    list[1] = saved;
    check("F the same address again", list[1].data() == address);
    const std::string f3 = run("F_list_3", reg, p, list);
    std::printf("CMP %s\nCMP %s\nCMP %s\n", f1.c_str(), f2.c_str(), f3.c_str());
    check("F changed content = a fresh fit on it", body(f2) == body(f2_fresh));
    check("F changed content changes the result", body(f2) != body(f1));
    check("F changed back = the first fit", body(f3) == body(f1));

    // G: success, failure (a negative cluster code: refused), success.
    HdfeRegressorV11 sfs(options_for(1));
    const std::string g1 = run("G_ok_1", sfs, p, p.clusters);
    Clusters bad = p.clusters;
    bad[1](5) = -3;
    const std::string g2 = run("G_fail", sfs, p, bad);
    const std::string g3 = run("G_ok_2", sfs, p, p.clusters);
    std::printf("CMP %s\nCMP %s\nCMP %s\n", g1.c_str(), g2.c_str(), g3.c_str());
    check("G failure in between", g2.find(" error ") != std::string::npos);
    check("G success after the failure = the first", body(g3) == body(g1));
}

// --- H: the inference on frozen post-absorption inputs (round 23) --------------------
// CSP_DUMP=<file>: absorb the panel once through partial_out() (on the GPU when
// XHDFE_GPU_BACKEND=cuda; gpu_used is printed) and write y_tilde, X_tilde and the
// clusters; CSP_FROZEN_INPUT=<file>: every arm reads them and runs the multi-way
// inference, so that the new, off and frozen binaries see the same inputs bit for bit.
int dump_inputs(const Panel& p, const char* path) {
    hdfe::HdfeOptions o = options_for(4);
    o.drop_singletons = false;
    HdfeRegressorV11 reg(o);
    const hdfe::detail::AbsorptionResult a = reg.partial_out(p.y, p.X, p.fes, nullptr, &p.clusters, nullptr);
    const std::int64_t n = a.y_tilde.size(), k = a.X_tilde.cols();
    std::printf("DUMP rows %lld cols %lld gpu_used %d\n", static_cast<long long>(n), static_cast<long long>(k),
                a.gpu_used ? 1 : 0);
    if (n != p.y.size() || a.X_tilde.rows() != n) return 1;
    FILE* f = std::fopen(path, "wb");
    if (!f) return 1;
    std::fwrite(&n, sizeof n, 1, f);
    std::fwrite(&k, sizeof k, 1, f);
    std::fwrite(a.y_tilde.data(), sizeof(double), static_cast<std::size_t>(n), f);
    const Eigen::MatrixXd Xt = a.X_tilde;
    std::fwrite(Xt.data(), sizeof(double), static_cast<std::size_t>(n * k), f);
    for (const auto& c : p.clusters) std::fwrite(c.data(), sizeof(int), static_cast<std::size_t>(n), f);
    return std::fclose(f) == 0 ? 0 : 1;
}

void part_h(const char* path) {
    FILE* f = std::fopen(path, "rb");
    std::int64_t n = 0, k = 0;
    if (!f || std::fread(&n, sizeof n, 1, f) != 1 || std::fread(&k, sizeof k, 1, f) != 1) {
        std::printf("CMP H error unreadable %s\n", path);
        if (f) std::fclose(f);
        return;
    }
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X(n, k);
    Clusters c(2, Eigen::VectorXi(n));
    bool ok = std::fread(y.data(), sizeof(double), static_cast<std::size_t>(n), f) == static_cast<std::size_t>(n) &&
              std::fread(X.data(), sizeof(double), static_cast<std::size_t>(n * k), f) == static_cast<std::size_t>(n * k);
    for (auto& v : c) ok = ok && std::fread(v.data(), sizeof(int), static_cast<std::size_t>(n), f) == static_cast<std::size_t>(n);
    std::fclose(f);
    if (!ok) {
        std::printf("CMP H error short %s\n", path);
        return;
    }
    std::printf("CMP H inputs y %016llx X %016llx\n", static_cast<unsigned long long>(hash_of(y)),
                static_cast<unsigned long long>(hash_of(X)));
    const OlsResult r = ols(y, X, nullptr, c);
    std::printf("CMP %s\n", ols_line("H_frozen_input", r).c_str());
    const Eigen::MatrixXd s = sandwich(r, X, nullptr, c);
    std::printf("CMP H_frozen_input sandwich %016llx\n", static_cast<unsigned long long>(hash_of(s)));
#ifdef CSP_NEW_API
    hdfe::detail::ClusterPasses passes;
    passes.want_pair = true;
    const OlsResult rp = ols(y, X, nullptr, c, &passes);
    check("H producer same bits", ols_line("H_frozen_input", rp) == ols_line("H_frozen_input", r));
    check("H consumer same bits", hash_of(sandwich(r, X, nullptr, c, &passes)) == hash_of(s));
    if (!switch_off()) check("H carrier counts = counts here", passes.ready && passes.mask_counts == mask_counts_here(c));
#endif
}

}  // namespace

int main() {
    if (const char* dump = std::getenv("CSP_DUMP")) return dump_inputs(make_panel(), dump);
    part_a();
    // B: S = 2^509, 8 S = 2^512, the pair's square 2^1024 overflows; rss = 32 S^2 = 2^1023 stays finite.
    // Its companion S = 2^507 keeps the pair's meat finite and passes every check.
    failure_case("B_last_meat", 1.0, std::ldexp(1.0, 509), false);
    failure_case("B_companion", 1.0, std::ldexp(1.0, 507), true);
    // C: x = 2^-20, S = 2^492: every meat in range, then bread 2^35 and meat ~2^952 fail the
    // sandwich's range test (2 * 35 + 952 + 6 >= 1024). Its companion S = 2^480 passes it.
    failure_case("C_final_sandwich", std::ldexp(1.0, -20), std::ldexp(1.0, 492), false);
    failure_case("C_companion", std::ldexp(1.0, -20), std::ldexp(1.0, 480), true);
    part_d();
    part_efg(make_panel());
    if (const char* frozen = std::getenv("CSP_FROZEN_INPUT")) part_h(frozen);
    std::printf("DONE failures %d\n", failures);
    return failures == 0 ? 0 : 1;
}
