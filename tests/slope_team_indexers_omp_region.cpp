// Phase 5a of the Stata-path plan: the identifier builds of the route with
// heterogeneous slopes, called through the public C++ entries, outside and
// inside an active OpenMP region, and through partial_out() (the second caller of
// the slope rank). Built against two core libraries (the branch's and the frozen
// phase chain's) and run with XHDFE_SLOPE_TEAM_INDEXERS unset and "0": the three
// outputs must be identical line for line. Synthetic data only (a fixed LCG).
//
//   c++ ... tests/slope_team_indexers_omp_region.cpp <build>/libxhdfe.a -fopenmp
//   ./slope_team_indexers_omp_region > out.txt
//
// Cases: fit outside any region with an explicit team of 8; partial_out() likewise;
// fit inside "#pragma omp parallel num_threads(2)" + single with the automatic
// thread request (the fit resolves it to one thread there); the same with an
// explicit request of 8 (the fit refuses it there: the error text is printed).
// Every result is printed as an FNV-1a hash of its raw bytes, with threads_used.
#include "hdfe/hdfe_regressor_v11.hpp"
#include "fe_absorption.hpp"

#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace hdfe {
using v11::HdfeRegressorV11;
}

namespace {

std::uint64_t fnv(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <class M>
void print_hash(const char* tag, const char* field, const M& m) {
    std::printf("%s %s %016llx\n", tag, field,
                static_cast<unsigned long long>(
                    fnv(m.data(), sizeof(typename M::Scalar) * static_cast<std::size_t>(m.size()))));
}

struct Panel {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    std::vector<hdfe::detail::HeterogeneousSlopeTerm> slopes;
    std::vector<Eigen::VectorXi> clusters;
};

Panel make_panel() {
    std::uint64_t s = 20261115ULL;
    auto next = [&s]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    };
    const int workers = 200000, firms = 12000, years = 10;
    std::vector<int> wk, fm, yr;
    for (int w = 0; w < workers; ++w) {
        const int rows = 3 + static_cast<int>(next() * 6.0);
        const int home = static_cast<int>(next() * firms);
        const int start = static_cast<int>(next() * 2.0);
        for (int r = 0; r < rows; ++r) {
            wk.push_back(w);
            fm.push_back(next() < 0.15 ? static_cast<int>(next() * firms) : home);
            yr.push_back((start + r) % years);
        }
    }
    const int n = static_cast<int>(wk.size());
    // A raw sparse firm identifier, f -> 37 + f * K with K = floor(8.7 n / firms): a range
    // of about 8.7 n, above 2n (the sparse branches) and within the phase-1 index's
    // budget (12 W <= 8 n, about 42.7 n), so that the team builders take the index.
    const long long K = static_cast<long long>(8.7 * n / firms);
    Panel p;
    p.y.resize(n);
    p.X.resize(n, 2);
    Eigen::VectorXi worker(n), firm(n), year(n);
    Eigen::VectorXd z(n);
    for (int i = 0; i < n; ++i) {
        worker(i) = wk[static_cast<std::size_t>(i)];
        const int f = fm[static_cast<std::size_t>(i)];
        firm(i) = static_cast<int>(37 + static_cast<long long>(f) * K);
        year(i) = yr[static_cast<std::size_t>(i)];
        // Firms 0, 3, 6, ...: a slope variable almost constant within the firm, at the
        // determinant boundary of the slope rank (det near 1e-12 * scale).
        const double eps = (f % 3 == 0) ? (f % 9 == 0 ? 1e-7 : (f % 9 == 3 ? 1e-6 : 1e-5)) : -1.0;
        z(i) = eps > 0 ? 1.0 + eps * (next() - 0.5) : next() * 2.0 - 1.0;
        p.X(i, 0) = next() - 0.5;
        p.X(i, 1) = next() - 0.5 + 0.1 * (f % 7);
        p.y(i) = 0.5 * p.X(i, 0) - 0.25 * p.X(i, 1) + 0.2 * z(i) + 0.3 * (next() - 0.5) +
                 0.01 * (f % 13) + 0.001 * (worker(i) % 17);
    }
    p.fes = {worker, firm, year};
    hdfe::detail::HeterogeneousSlopeTerm term;
    term.fe_index = 1;
    term.values = z;
    term.include_intercept = true;
    p.slopes = {term};
    p.clusters = {worker};
    return p;
}

hdfe::HdfeOptions options_for(int threads) {
    hdfe::HdfeOptions o;
    o.se_type = hdfe::StandardErrorType::Cluster;
    o.tolerance_mode = hdfe::ToleranceMode::XhdfeFast;
    o.num_threads = threads;
    return o;
}

void report_fit(const char* tag, const hdfe::HdfeRegressorV11& reg) {
    const hdfe::HdfeResults& r = reg.results();
    print_hash(tag, "coefficients", r.coefficients);
    print_hash(tag, "covariance", r.covariance);
    print_hash(tag, "residuals", r.residuals);
    std::printf("%s df_a %.17g df_resid %.17g nobs %d threads_used %d\n", tag, r.df_a, r.df_resid,
                r.nobs, reg.threads_used());
}

void run_fit(const char* tag, const Panel& p, int threads) {
    try {
        hdfe::HdfeRegressorV11 reg(options_for(threads));
        reg.fit(p.y, p.X, p.fes, nullptr, &p.clusters, nullptr, {}, &p.slopes);
        report_fit(tag, reg);
    } catch (const std::exception& e) {
        std::printf("%s error %s\n", tag, e.what());
    }
}

}  // namespace

int main() {
    const Panel p = make_panel();
    std::printf("rows %d\n", static_cast<int>(p.y.size()));
    run_fit("fit_outside_t8", p, 8);
    try {
        hdfe::HdfeRegressorV11 reg(options_for(8));
        const hdfe::detail::AbsorptionResult a =
            reg.partial_out(p.y, p.X, p.fes, nullptr, &p.clusters, &p.slopes);
        print_hash("partial_out_t8", "y_tilde", a.y_tilde);
        print_hash("partial_out_t8", "X_tilde", a.X_tilde);
        const hdfe::HdfeResults& r = reg.results();
        std::printf("partial_out_t8 df_a %.17g nobs %d threads_used %d\n", r.df_a, r.nobs,
                    reg.threads_used());
    } catch (const std::exception& e) {
        std::printf("partial_out_t8 error %s\n", e.what());
    }
    int in_region = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            in_region = omp_in_parallel();
            run_fit("fit_inside_region_auto", p, 0);
            run_fit("fit_inside_region_t8", p, 8);
        }
    }
    std::printf("inside_region_active %d\n", in_region);
    return 0;
}
