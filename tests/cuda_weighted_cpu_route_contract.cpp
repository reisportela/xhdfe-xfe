// Package C (18nov2026): the weighted CPU route at the C++ entries (ctest, CUDA build only).
//
// A weighted fit with two or more absorbed effects in reghdfe-comparable mode whose effective backend is CUDA
// runs as the CPU call and publishes GPU status code 7 (Verifications/corrigir_erros_20261117/c/DESENHO_C.md,
// c/RESPOSTA_DESENHO_C.md points 5, 8, 9, 11); the partial-out entry is not routed (c/RESPOSTA_3_C.md). Checked:
//  1. fit, routed (thread request CUDA) against the same fit with the CPU request: coefficients, covariance,
//     standard errors, residuals and RSS bit for bit, iterations, convergence and certificate; code 7 against 0,
//     the other GPU fields as the CPU produced them; with and without zero weights.
//  2. a real absorption-cache hit in fit (the reduced tier's reason reads "cache"): code 7, the estimates bit for
//     bit the CPU fit's; and a weighted partial_out with two effects under a CUDA request stays on the device
//     (gpu_used, code 1, in the struct and in the regressor).
//  3. two threads with their own regressors and their own CUDA requests: thread A runs a long routed fit while
//     thread B repeats device fits; at least three device fits start 200 ms or more after A's call began and end
//     before it ends (timestamps; the margin covers A's entry up to its override), each with gpu_used 1 and
//     status 1, and each thread finds its own request afterwards.
//  4. a refusal raised inside the CPU path of a routed fit (the CPU's explicit Schwarz refusal with weights): the
//     same reason text in both starting states (no thread override, the request in the environment; an outer
//     CUDA override, on an object that already holds a valid result), no estimates left, the thread's override
//     back to what it was, and the next call really on the GPU.
// Synthetic data from fixed seeds; explicit thread counts. Exit 0 PASS, 1 FAIL; a missing device fails.
#include "hdfe/hdfe_regressor_v11.hpp"
#include "fe_absorption.hpp"
#include "fe_absorption_cuda.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using hdfe::detail::AbsorptionResult;
using hdfe::detail::GpuBackend;
using hdfe::detail::ScopedGpuBackendOverride;
using hdfe::v11::HdfeRegressorV11;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    } else {
        std::printf("ok: %s\n", what.c_str());
    }
}

struct Data {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    Eigen::VectorXd w;
};

// A panel with `nfe` effects (levels a, b, c), two regressors, U[0.5, 2) weights; `zeros` sets every tenth
// weight to zero.
Data panel(int n, int nfe, unsigned seed, bool zeros = false) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> z(0.0, 1.0);
    const int levels[3] = {std::max(2, n / 40), 97, 23};
    Data d;
    d.y.resize(n);
    d.X.resize(n, 2);
    d.w.resize(n);
    for (int k = 0; k < nfe; ++k) d.fes.emplace_back(n);
    std::vector<std::vector<double>> effect(nfe);
    for (int k = 0; k < nfe; ++k) {
        effect[k].resize(levels[k]);
        for (double& v : effect[k]) v = z(rng);
    }
    for (int i = 0; i < n; ++i) {
        double fe_sum = 0.0;
        for (int k = 0; k < nfe; ++k) {
            const int id = k == 0 ? i % levels[0] : static_cast<int>(u(rng) * levels[k]) % levels[k];
            d.fes[k](i) = id;
            fe_sum += effect[k][id];
        }
        d.X(i, 0) = z(rng) + 0.3 * fe_sum;
        d.X(i, 1) = z(rng);
        d.y(i) = 0.7 * d.X(i, 0) - 0.4 * d.X(i, 1) + fe_sum + 0.5 * z(rng);
        d.w(i) = (zeros && i % 10 == 3) ? 0.0 : 0.5 + 1.5 * u(rng);
    }
    return d;
}

hdfe::HdfeOptions options(int threads) {
    hdfe::HdfeOptions o;
    o.tol = 1e-8;
    o.num_threads = threads;
    o.tolerance_mode = hdfe::ToleranceMode::ReghdfeComparable;
    o.weights_are_frequencies = false;
    o.weighted_cpu_route = true;
    return o;
}

bool same_bits(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.data(), b.data(), sizeof(double) * static_cast<std::size_t>(a.size())) == 0;
}

void set_env(const char* name, const char* value) {
    if (value) {
        ::setenv(name, value, 1);
    } else {
        ::unsetenv(name);
    }
}

// 1. fit identity and the published code, with and without zero weights.
void fit_identity(bool zeros) {
    const Data d = panel(60000, 3, zeros ? 7u : 5u, zeros);
    const std::string tag = zeros ? " (zero weights)" : "";
    HdfeRegressorV11 reg_routed(options(4)), reg_cpu(options(4));
    {
        ScopedGpuBackendOverride request(GpuBackend::Cuda);
        reg_routed.fit(d.y, d.X, d.fes, &d.w);
        check(hdfe::detail::has_thread_gpu_backend_override() &&
                  hdfe::detail::thread_gpu_backend_override() == GpuBackend::Cuda,
              "the caller's CUDA request is back after the routed fit" + tag);
    }
    {
        ScopedGpuBackendOverride request(GpuBackend::Cpu);
        reg_cpu.fit(d.y, d.X, d.fes, &d.w);
    }
    const hdfe::HdfeResults& r = reg_routed.results();
    const hdfe::HdfeResults& c = reg_cpu.results();
    check(reg_routed.gpu_status_code() == hdfe::detail::kGpuStatusWeightedCpuRoute,
          "the routed fit publishes 7" + tag);
    check(reg_cpu.gpu_status_code() == 0, "the CPU fit publishes 0" + tag);
    check(!reg_routed.gpu_used() && !reg_routed.gpu_attempted() && reg_routed.gpu_absorption_iterations() == 0,
          "routed fit: the other GPU fields are the CPU's (false, zero)" + tag);
    check(same_bits(r.coefficients, c.coefficients) && same_bits(r.covariance, c.covariance) &&
              same_bits(r.std_errors, c.std_errors) && same_bits(r.residuals, c.residuals) &&
              std::memcmp(&r.rss, &c.rss, sizeof(double)) == 0,
          "routed fit: coefficients, covariance, standard errors, residuals and RSS bit for bit the CPU fit's" + tag);
    check(r.num_iterations == c.num_iterations && r.converged == c.converged &&
              r.precision_certified == c.precision_certified && r.converged && r.precision_certified,
          "routed fit: iterations, convergence and certificate equal and accepted" + tag);
    check(r.sample_index.size() == c.sample_index.size() && (r.sample_index.array() == c.sample_index.array()).all(),
          "routed fit: the same sample map" + tag);
}

// 2. a real absorption-cache hit in fit.
void fit_cache_hit(const std::string& dir) {
    const std::string path = dir + "/cwr_fit_cache.bin";
    std::remove(path.c_str());
    set_env("XHDFE_ABSORPTION_CACHE", path.c_str());
    set_env("XHDFE_ABSORPTION_CACHE_MODE", "auto");
    const Data d = panel(40000, 2, 11u);
    HdfeRegressorV11 reg_first(options(4)), reg_hit(options(4)), reg_cpu(options(4));
    {
        ScopedGpuBackendOverride request(GpuBackend::Cuda);
        reg_first.fit(d.y, d.X, d.fes, &d.w);
        reg_hit.fit(d.y, d.X, d.fes, &d.w);
    }
    {
        ScopedGpuBackendOverride request(GpuBackend::Cpu);
        reg_cpu.fit(d.y, d.X, d.fes, &d.w);
    }
    set_env("XHDFE_ABSORPTION_CACHE", nullptr);
    set_env("XHDFE_ABSORPTION_CACHE_MODE", nullptr);
    check(reg_first.reduced_pcg_reason() != "cache", "the first routed fit computed and wrote the entry");
    check(reg_hit.reduced_pcg_reason() == "cache" && reg_cpu.reduced_pcg_reason() == "cache",
          "the second routed fit and the CPU fit hit the entry (reason cache)");
    check(reg_hit.gpu_status_code() == hdfe::detail::kGpuStatusWeightedCpuRoute, "a routed cache hit publishes 7");
    check(reg_cpu.gpu_status_code() == 0, "the CPU cache hit publishes 0");
    check(same_bits(reg_hit.results().coefficients, reg_cpu.results().coefficients) &&
              same_bits(reg_hit.results().covariance, reg_cpu.results().covariance) &&
              same_bits(reg_first.results().coefficients, reg_hit.results().coefficients),
          "the routed hit is the CPU's entry, bit for bit");
    std::remove(path.c_str());
}

// 2b. the partial-out entry is not routed (c/RESPOSTA_3_C.md): a weighted partial_out with two effects under a CUDA
// request stays on the device, in the struct and in the regressor.
void partial_out_on_device() {
    const Data d = panel(60000, 2, 23u);
    HdfeRegressorV11 reg(options(4));
    ScopedGpuBackendOverride request(GpuBackend::Cuda);
    const AbsorptionResult a = reg.partial_out(d.y, d.X, d.fes, &d.w);
    check(a.gpu_used && a.gpu_status_code == 1 && reg.gpu_used() && reg.gpu_status_code() == 1,
          "a weighted partial_out with two effects requested on CUDA stays on the device (code 1, struct and regressor)");
}

// 3. two threads, overlapping, each with its own regressor and request.
void two_threads() {
    const Data big = panel(1500000, 3, 13u);
    const Data small = panel(100000, 2, 17u);
    std::atomic<bool> a_started{false}, a_done{false};
    Clock::time_point a_begin, a_end;
    int a_code = -1;
    bool a_gpu = true, a_request_back = false;
    struct Shot { Clock::time_point begin, end; bool gpu_used; int code; bool cuda_seen; };
    std::vector<Shot> shots;
    bool b_request_back = false;
    std::thread a([&] {
        ScopedGpuBackendOverride request(GpuBackend::Cuda);
        HdfeRegressorV11 reg(options(2));
        a_begin = Clock::now();
        a_started.store(true);
        reg.fit(big.y, big.X, big.fes, &big.w);
        a_end = Clock::now();
        a_done.store(true);
        a_code = reg.gpu_status_code();
        a_gpu = reg.gpu_used();
        a_request_back = hdfe::detail::has_thread_gpu_backend_override() &&
                         hdfe::detail::thread_gpu_backend_override() == GpuBackend::Cuda;
    });
    std::thread b([&] {
        ScopedGpuBackendOverride request(GpuBackend::Cuda);
        while (!a_started.load()) std::this_thread::yield();
        while (!a_done.load() && shots.size() < 200) {
            Shot s;
            s.cuda_seen = hdfe::detail::effective_gpu_backend() == GpuBackend::Cuda;
            HdfeRegressorV11 reg(options(2));
            s.begin = Clock::now();
            reg.fit(small.y, small.X, small.fes, nullptr);
            s.end = Clock::now();
            s.gpu_used = reg.gpu_used();
            s.code = reg.gpu_status_code();
            shots.push_back(s);
        }
        b_request_back = hdfe::detail::has_thread_gpu_backend_override() &&
                         hdfe::detail::thread_gpu_backend_override() == GpuBackend::Cuda;
    });
    a.join();
    b.join();
    const auto margin = std::chrono::milliseconds(200);
    int inside = 0, inside_ok = 0;
    bool all_ok = true;
    for (const Shot& s : shots) {
        const bool ok = s.gpu_used && s.code == 1 && s.cuda_seen;
        all_ok = all_ok && ok;
        if (s.begin > a_begin + margin && s.end < a_end) {
            ++inside;
            inside_ok += ok ? 1 : 0;
        }
    }
    std::printf("two threads: A %.3f s, %zu device fits in B, %d starting 200 ms or more into A's routed call and "
                "ending inside it\n",
                std::chrono::duration<double>(a_end - a_begin).count(), shots.size(), inside);
    check(a_code == hdfe::detail::kGpuStatusWeightedCpuRoute && !a_gpu, "thread A's weighted fit was routed (7)");
    check(inside >= 3 && inside_ok == inside,
          "at least three device fits of thread B ran 200 ms or more into A's routed call, on the GPU with status 1");
    check(all_ok, "every device fit of thread B saw CUDA and used the GPU with status 1");
    check(a_request_back && b_request_back, "each thread finds its own CUDA request after its fits");
}

// 4. a refusal raised inside the CPU path of a routed call, through fit.
void refusal_inside_cpu_path() {
    const Data d = panel(30000, 2, 19u);
    hdfe::HdfeOptions o = options(4);
    o.absorption_method = hdfe::AbsorptionMethod::Schwarz;
    const char* reason = "CPU Schwarz requires no weights";
    // (a) the request in the environment, no thread override.
    set_env("XHDFE_GPU_BACKEND", "cuda");
    {
        HdfeRegressorV11 reg(o);
        bool threw = false;
        std::string message;
        try {
            reg.fit(d.y, d.X, d.fes, &d.w);
        } catch (const std::exception& e) {
            threw = true;
            message = e.what();
        }
        check(threw && message.find(reason) != std::string::npos,
              "(environment) the routed explicit Schwarz with weights raised the CPU's refusal");
        check(!reg.has_estimation_result() && !reg.has_partial_result(), "(environment) no estimates left");
        check(!hdfe::detail::has_thread_gpu_backend_override(), "(environment) no override left on the thread");
        HdfeRegressorV11 next(options(4));
        next.fit(d.y, d.X, d.fes, nullptr);
        check(next.gpu_used() && next.gpu_status_code() == 1, "(environment) the next fit really ran on the GPU");
    }
    set_env("XHDFE_GPU_BACKEND", nullptr);
    // (b) an outer CUDA override, on an object that already holds a valid result (an unweighted CPU Schwarz fit).
    {
        ScopedGpuBackendOverride request(GpuBackend::Cuda);
        HdfeRegressorV11 reg(o);
        bool held = false;
        try {
            ScopedGpuBackendOverride cpu(GpuBackend::Cpu);
            reg.fit(d.y, d.X, d.fes, nullptr);
            held = reg.has_estimation_result();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "the unweighted CPU Schwarz fit failed: %s\n", e.what());
        }
        check(held, "(outer override) the object held a valid result before the routed fit");
        bool threw = false;
        std::string message;
        try {
            reg.fit(d.y, d.X, d.fes, &d.w);
        } catch (const std::exception& e) {
            threw = true;
            message = e.what();
        }
        check(threw && message.find(reason) != std::string::npos,
              "(outer override) the routed explicit Schwarz with weights raised the CPU's refusal");
        check(!reg.has_estimation_result() && !reg.has_partial_result(),
              "(outer override) no estimates left, the earlier result included");
        check(hdfe::detail::has_thread_gpu_backend_override() &&
                  hdfe::detail::thread_gpu_backend_override() == GpuBackend::Cuda,
              "(outer override) the outer CUDA request is back");
        HdfeRegressorV11 next(options(4));
        next.fit(d.y, d.X, d.fes, nullptr);
        check(next.gpu_used() && next.gpu_status_code() == 1, "(outer override) the next fit really ran on the GPU");
    }
}

}  // namespace

int main() {
    if (!hdfe::detail::cuda_backend_available()) {
        std::fprintf(stderr, "FAIL: no CUDA device visible (this ctest needs one)\n");
        return 1;
    }
    for (const char* name : {"XHDFE_GPU_BACKEND", "XHDFE_ABSORPTION_CACHE",
                             "XHDFE_ABSORPTION_CACHE_MODE", "XHDFE_MOBILITY_PROFILE", "XHDFE_MOBILITY_MODE"}) {
        set_env(name, nullptr);
    }
    char dir_template[] = "./cwr_ctest_XXXXXX";
    const char* dir = ::mkdtemp(dir_template);
    if (dir == nullptr) {
        std::fprintf(stderr, "FAIL: no temporary directory\n");
        return 1;
    }
    try {
        fit_identity(false);
        fit_identity(true);
        fit_cache_hit(dir);
        partial_out_on_device();
        refusal_inside_cpu_path();
        two_threads();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    ::rmdir(dir);
    std::printf("%s (%d failure%s)\n", g_failures ? "FAIL" : "PASS", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
