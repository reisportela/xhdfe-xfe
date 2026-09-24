// Native SPI/OpenMP gate. Host has no OpenMP or estimator implementation.
// SPI-loading pattern adapted from xsamplefe/tests/plugin_smoke.cpp.
#if defined(_OPENMP)
#error "Build the probe host without OpenMP; only the tested plugin supplies it"
#endif
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0602
#endif
#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#if defined(_WIN32) && !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef SYSTEM
#if defined(_WIN32)
#define SYSTEM 4
#elif defined(__APPLE__)
#define SYSTEM 3
#elif defined(__linux__)
#define SYSTEM 2
#else
#error "This probe targets Windows, macOS and Linux"
#endif
#endif
#include "stplugin.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

namespace {
constexpr int kRows = 64 * 64 * 32;
constexpr double kMissing = 8.0e307;
using Columns = std::vector<std::vector<double>>;
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> values;
};
struct SpiState {
    Columns data;
    std::map<std::string, double> scalars;
    std::map<std::string, Matrix> matrices;
    std::map<std::string, std::string> macros;
    std::string errors;
    int stop = 0;
} state;
using Call = ST_retcode (*)(int, char**);
using Init = ST_retcode (*)(ST_plugin*);
#ifdef _WIN32
using Library = HMODULE;
#else
using Library = void*;
#endif

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
#ifdef _WIN32
std::wstring to_wide(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         text.data(), static_cast<int>(text.size()), nullptr, 0);
    require(size > 0, "invalid UTF-8 path or argument");
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    require(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                static_cast<int>(text.size()), result.data(), size) == size,
            "failed to convert UTF-8 path or argument");
    return result;
}
std::string to_utf8(const std::wstring& text) {
    if (text.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                         text.data(), static_cast<int>(text.size()),
                                         nullptr, 0, nullptr, nullptr);
    require(size > 0, "invalid UTF-16 path or argument");
    std::string result(static_cast<std::size_t>(size), '\0');
    require(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                                static_cast<int>(text.size()), result.data(), size,
                                nullptr, nullptr) == size,
            "failed to convert UTF-16 path or argument");
    return result;
}
std::string windows_error(DWORD code) {
    wchar_t* buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD size = FormatMessageW(flags, nullptr, code, 0,
                                      reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::wstring message = size && buffer ? std::wstring(buffer, size) : L"unknown Windows error";
    if (buffer) LocalFree(buffer);
    while (!message.empty() && (message.back() == L'\r' || message.back() == L'\n'))
        message.pop_back();
    return to_utf8(message) + " (Win32 " + std::to_string(code) + ')';
}
#endif
int nrows() { return state.data.empty() ? 0 : static_cast<int>(state.data[0].size()); }
int nvars() { return static_cast<int>(state.data.size()); }
int first_row() { return 1; }
ST_boolean selected(int row) { return row >= 1 && row <= nrows(); }
ST_boolean not_string(int) { return 0; }
ST_boolean missing(double value) { return !std::isfinite(value) || value >= kMissing; }
int display(char*) { return 0; }
int error_output(char* message) { state.errors += message; return 0; }
int poll() { return 0; }
int read_value(int column, int row, double* value) {
    if (!value || column < 1 || column > nvars() || row < 1 || row > nrows()) return 198;
    *value = state.data[column - 1][row - 1];
    return 0;
}
int write_value(int column, int row, double value) {
    if (column < 1 || column > nvars() || row < 1 || row > nrows()) return 198;
    state.data[column - 1][row - 1] = value;
    return 0;
}
int scalar_save(char* name, double value) { state.scalars[name] = value; return 0; }
int scalar_read(char* name, double* value) {
    const auto found = state.scalars.find(name);
    if (found == state.scalars.end() || !value) return 111;
    *value = found->second;
    return 0;
}
int matrix_rows(char* name) {
    const auto found = state.matrices.find(name);
    return found == state.matrices.end() ? 0 : found->second.rows;
}
int matrix_cols(char* name) {
    const auto found = state.matrices.find(name);
    return found == state.matrices.end() ? 0 : found->second.cols;
}
int matrix_store(char* name, int row, int col, double value) {
    const auto found = state.matrices.find(name);
    if (found == state.matrices.end()) return 111;
    Matrix& matrix = found->second;
    if (row < 1 || row > matrix.rows || col < 1 || col > matrix.cols) return 503;
    matrix.values[(row - 1) * matrix.cols + col - 1] = value;
    return 0;
}
int matrix_read(char* name, int row, int col, double* value) {
    const auto found = state.matrices.find(name);
    if (found == state.matrices.end() || !value) return 111;
    const Matrix& matrix = found->second;
    if (row < 1 || row > matrix.rows || col < 1 || col > matrix.cols) return 503;
    *value = matrix.values[(row - 1) * matrix.cols + col - 1];
    return 0;
}
int macro_save(char* name, char* value) { state.macros[name] = value; return 0; }
int macro_read(char* name, char* value, int length) {
    const auto found = state.macros.find(name);
    const std::string text = found == state.macros.end() ? "" : found->second;
    if (!value || length <= static_cast<int>(text.size())) return 198;
    std::memcpy(value, text.c_str(), text.size() + 1);
    return 0;
}

ST_plugin spi() {
    ST_plugin api{};
    api.spoutsml = display; api.spoutnosml = display; api.spouterr = error_output;
    api.pollstd = poll; api.pollnow = poll;
    api.safevdata = read_value; api.vdata = read_value;
    api.safestore = write_value; api.store = write_value;
    api.scalsave = scalar_save; api.scalaruse = scalar_read;
    api.rowsof = matrix_rows; api.colsof = matrix_cols;
    api.safematstore = matrix_store; api.matstore = matrix_store;
    api.safematel = matrix_read; api.matel = matrix_read;
    api.macresave = macro_save; api.macuse = macro_read;
    api.nobs = nrows; api.nobs1 = first_row; api.nobs2 = nrows;
    api.nvar = nvars; api.nvars = nvars; api.selobs = selected;
    api.isstr = not_string; api.isstrl = not_string; api.ismissing = missing;
    api.missval = kMissing; api.matsize = 32767; api.stopflag = &state.stop;
    api.major = SD_PLUGINMAJ; api.minor = SD_PLUGINMIN;
    return api;
}

std::string architecture() {
#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unsupported";
#endif
}
std::string native_architecture() {
#ifdef _WIN32
    using IsWow64Process2Call = BOOL (WINAPI *)(HANDLE, USHORT*, USHORT*);
    const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    const auto probe = kernel ? reinterpret_cast<IsWow64Process2Call>(
        GetProcAddress(kernel, "IsWow64Process2")) : nullptr;
    if (probe) {
        USHORT process_machine = IMAGE_FILE_MACHINE_UNKNOWN;
        USHORT native_machine = IMAGE_FILE_MACHINE_UNKNOWN;
        if (!probe(GetCurrentProcess(), &process_machine, &native_machine)) return "unknown";
        if (native_machine == IMAGE_FILE_MACHINE_AMD64) return "x86_64";
        if (native_machine == IMAGE_FILE_MACHINE_ARM64) return "arm64";
        return "unsupported";
    }
    SYSTEM_INFO info{};
    GetNativeSystemInfo(&info);
    if (info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) return "x86_64";
    if (info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64) return "arm64";
    return "unsupported";
#else
    return architecture();
#endif
}
int translated() {
#ifdef _WIN32
    using IsWow64Process2Call = BOOL (WINAPI *)(HANDLE, USHORT*, USHORT*);
    const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    const auto probe = kernel ? reinterpret_cast<IsWow64Process2Call>(
        GetProcAddress(kernel, "IsWow64Process2")) : nullptr;
    if (probe) {
        USHORT process_machine = IMAGE_FILE_MACHINE_UNKNOWN;
        USHORT native_machine = IMAGE_FILE_MACHINE_UNKNOWN;
        if (!probe(GetCurrentProcess(), &process_machine, &native_machine)) return -1;
        return process_machine == IMAGE_FILE_MACHINE_UNKNOWN ? 0 : 1;
    }
    const std::string native = native_architecture();
    return native == "unsupported" || native == "unknown" ? -1 : native == architecture() ? 0 : 1;
#elif defined(__APPLE__)
    int answer = 0;
    std::size_t size = sizeof(answer);
    if (sysctlbyname("sysctl.proc_translated", &answer, &size, nullptr, 0) == 0) return answer;
    return errno == ENOENT ? 0 : -1;
#else
    return 0;
#endif
}
bool absolute_path(const std::string& path) {
#ifdef _WIN32
    const std::wstring wide = to_wide(path);
    const bool drive = wide.size() >= 3 && std::iswalpha(wide[0]) && wide[1] == L':' &&
                       (wide[2] == L'\\' || wide[2] == L'/');
    const bool unc = wide.size() >= 2 && wide[0] == L'\\' && wide[1] == L'\\';
    return drive || unc;
#else
    return !path.empty() && path[0] == '/';
#endif
}

Library open_library(const std::string& path) {
#ifdef _WIN32
    const std::wstring wide = to_wide(path);
    Library library = LoadLibraryW(wide.c_str());
    if (!library) throw std::runtime_error(windows_error(GetLastError()));
    return library;
#else
    Library library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    return library;
#endif
}

void* library_symbol(Library library, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(library, name));
#else
    return dlsym(library, name);
#endif
}

std::string loaded_openmp_runtime(Library library) {
#ifdef _WIN32
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32,
                                               GetCurrentProcessId());
    require(snapshot != INVALID_HANDLE_VALUE,
            "could not enumerate loaded modules: " + windows_error(GetLastError()));
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    std::vector<std::pair<HMODULE, FARPROC>> matches;
    if (FARPROC symbol = GetProcAddress(library, "omp_get_max_threads"))
        matches.emplace_back(library, symbol);
    if (Module32FirstW(snapshot, &entry)) {
        do {
            std::wstring name(entry.szModule);
            std::transform(name.begin(), name.end(), name.begin(),
                           [](wchar_t value) { return std::towlower(value); });
            const bool libgomp = name.rfind(L"libgomp-", 0) == 0 &&
                name.size() >= 4 && name.substr(name.size() - 4) == L".dll";
            FARPROC symbol = libgomp ? GetProcAddress(entry.hModule, "omp_get_max_threads") : nullptr;
            if (symbol) matches.emplace_back(entry.hModule, symbol);
        } while (Module32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    require(matches.size() <= 1, "more than one loaded libgomp runtime exports omp_get_max_threads");
    if (matches.empty()) return {};
    HMODULE owner = nullptr;
    require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(matches[0].second), &owner) != 0 &&
            owner == matches[0].first,
            "loaded libgomp symbol could not be attributed to its module");
    std::vector<wchar_t> path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(owner, path.data(), static_cast<DWORD>(path.size()));
    require(length > 0 && length < path.size(),
            "loaded libgomp path could not be read: " + windows_error(GetLastError()));
    return to_utf8(std::wstring(path.data(), length));
#else
    void* omp_symbol = dlsym(library, "omp_get_max_threads");
    Dl_info info{};
    if (omp_symbol && dladdr(omp_symbol, &info) && info.dli_fname) return info.dli_fname;
    return {};
#endif
}
std::string quote(const std::string& text) {
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 32) result += '?';
        else result += static_cast<char>(c);
    }
    return result + '"';
}

Columns fixture(bool xhdfe) {
    Columns data(xhdfe ? 6 : 9, std::vector<double>(kRows, kMissing));
    for (int i = 0; i < kRows; ++i) {
        const int replicate = i % 32;
        const int fe2 = (i / 32) % 64;
        const int fe1 = i / (32 * 64);
        const double x1 = (replicate & 1) ? 1.0 : -1.0;
        const double x2 = (replicate & 2) ? 1.0 : -1.0;
        const double u = (replicate & 4) ? 0.125 : -0.125;
        data[0][i] = 1.25*x1 - 0.5*x2 + 3.0 + 0.25*fe1 - 0.125*fe2 + u;
        data[1][i] = x1; data[2][i] = x2;
        data[3][i] = fe1; data[4][i] = fe2;
    }
    return data;
}

std::string configuration(bool xhdfe, int threads) {
    std::string result = "cfg=nfe=2;nclust=0;has_weight=0;store_groupvar=0;"
        "tol=1e-8;max_iter=1000;drop_singletons=0;symmetric_sweep=0;"
        "absorption_method=gauss-seidel;jacobi_relaxation=0;gpu_backend=cpu;"
        "default_threads=0;max_threads=0;min_parallel_rows=0;target_rows_per_thread=1;"
        "mobility_profile_mode=off;absorption_cache_mode=off;fe_structure_cache_mode=off;"
        "s_N=probe_N;s_df_a=probe_df_a;s_iterations=probe_iterations;s_converged=probe_converged;"
        "s_threads_requested=probe_requested;s_threads_effective=probe_effective;"
        "s_threads_used=probe_used;s_parallel_workers_active=probe_workers;"
        "s_thread_capacity=probe_capacity;s_openmp_enabled=probe_openmp;"
        "s_thread_limit_code=probe_limit;s_gpu_used=probe_gpu;s_method_used=probe_method;";
    if (xhdfe) result += "b=probe_b;V=probe_V;p=2;nslope=0;ninst=0;group_mode=0;has_individual=0;"
        "store_resid=0;store_fes=0;retain_fes=0;fit_intercept=0;se_type=unadjusted;"
        "tolerance_mode=reghdfe-comparable;level=95;s_df_r=probe_df_r;s_rss=probe_rss;"
        "s_precision_certified=probe_certified;";
    else result += "k=3;";
    return result + "num_threads=" + std::to_string(threads) + ';';
}

struct Snapshot {
    int requested = 0, rc = -1;
    std::map<std::string, double> scalars;
    std::vector<double> numerical;
    std::string error;
    double max_analytic_error = 0;
    bool numeric_ok = false, inputs_unchanged = false;
};
double scalar(const Snapshot& snapshot, const char* name) {
    const auto found = snapshot.scalars.find(std::string("probe_") + name);
    require(found != snapshot.scalars.end() && std::isfinite(found->second),
            std::string("missing/invalid plugin diagnostic: ") + name);
    return found->second;
}
Snapshot run(Call call, bool xhdfe, int threads) {
    state = SpiState{};
    state.data = fixture(xhdfe);
    const Columns original = state.data;
    state.matrices["probe_b"] = Matrix{1, 2, std::vector<double>(2, kMissing)};
    state.matrices["probe_V"] = Matrix{2, 2, std::vector<double>(4, kMissing)};
    std::string config = configuration(xhdfe, threads);
    char* arguments[] = {config.data()};
    Snapshot result;
    result.requested = threads;
    result.rc = call(1, arguments);
    result.scalars = state.scalars;
    result.error = state.errors;
    result.inputs_unchanged = true;
    for (int column = 0; column < 5; ++column)
        result.inputs_unchanged &= state.data[column] == original[column];
    if (result.rc != 0) return result;
    require(result.inputs_unchanged, "plugin mutated an input column");
    require(scalar(result, "N") == kRows && scalar(result, "converged") == 1,
            "plugin did not return the complete converged fixture");
    require(scalar(result, "gpu") == 0, "CPU probe used a GPU");
    require(scalar(result, "method") == 1, "explicit Gauss-Seidel method changed");
    for (double value : state.data.back()) require(value == 1, "sample output is incomplete");
    auto compare = [&](double value, double expected, double tolerance) {
        require(std::isfinite(value), "non-finite numerical output");
        result.max_analytic_error = std::max(result.max_analytic_error, std::abs(value-expected));
        require(std::abs(value-expected) <= tolerance, "analytic fixture mismatch");
        result.numerical.push_back(value);
    };
    if (xhdfe) {
        require(scalar(result, "certified") == 1, "absorption was not certified");
        compare(state.matrices["probe_b"].values[0], 1.25, 1e-10);
        compare(state.matrices["probe_b"].values[1], -0.5, 1e-10);
        const double df = scalar(result, "df_r");
        require(df >= 1 && df <= kRows, "invalid residual degrees of freedom");
        const double variance = 0.015625 / df;
        for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col)
            compare(state.matrices["probe_V"].values[row*2+col], row == col ? variance : 0.0,
                    1e-8 * variance);
        compare(scalar(result, "rss"), kRows * 0.015625, 1e-10 * kRows * 0.015625);
    } else {
        for (int i = 0; i < kRows; ++i) {
            const double u = (i % 32 & 4) ? 0.125 : -0.125;
            compare(state.data[5][i], 1.25*original[1][i]-0.5*original[2][i]+u, 1e-10);
            compare(state.data[6][i], original[1][i], 1e-10);
            compare(state.data[7][i], original[2][i], 1e-10);
        }
    }
    result.numeric_ok = true;
    return result;
}

void print_snapshot(const Snapshot& value) {
    std::cout << "{\"request\":" << value.requested << ",\"rc\":" << value.rc
              << ",\"numeric_ok\":" << (value.numeric_ok ? "true" : "false")
              << ",\"inputs_unchanged\":" << (value.inputs_unchanged ? "true" : "false")
              << ",\"max_analytic_error\":" << value.max_analytic_error
              << ",\"error\":" << quote(value.error);
    const std::pair<const char*, const char*> diagnostics[] = {
        {"threads_requested", "probe_requested"}, {"threads_effective", "probe_effective"},
        {"threads_used", "probe_used"}, {"parallel_workers_active", "probe_workers"},
        {"thread_capacity", "probe_capacity"}, {"openmp_enabled", "probe_openmp"}};
    for (const auto& entry : diagnostics) {
        std::cout << ',' << quote(entry.first) << ':';
        const auto found = value.scalars.find(entry.second);
        if (found != value.scalars.end() && std::isfinite(found->second)) std::cout << found->second;
        else std::cout << "null";
    }
    std::cout << ",\"scalars\":{";
    bool first = true;
    for (const auto& entry : value.scalars) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << quote(entry.first) << ':';
        if (std::isfinite(entry.second)) std::cout << entry.second; else std::cout << "null";
    }
    std::cout << "}}";
}
} // namespace

int probe_main(int argc, char** argv) {
    std::string kind, path, expected_arch, runtime_path, reason;
    const std::string process_arch = architecture();
    const std::string native_arch = native_architecture();
    const int translation = translated();
    bool expect_serial = false, passed = false, serial_rejected = false;
    bool numerical_parity = false;
    double parity_error = 0;
    Snapshot one, two;
    try {
        require(argc == 4 || argc == 5,
                "usage: plugin_openmp_probe xhdfe|xfepout /absolute/plugin arm64|x86_64 [--expect-serial]");
        kind = argv[1]; path = argv[2]; expected_arch = argv[3];
        expect_serial = argc == 5 && std::string(argv[4]) == "--expect-serial";
        require(argc != 5 || expect_serial, "unknown probe option");
        require(kind == "xhdfe" || kind == "xfepout", "unknown plugin kind");
        require(absolute_path(path), "plugin path must be absolute");
        require(process_arch == expected_arch, "probe is executing the wrong architecture");
        require(translation == 0, "translated execution is not native architecture evidence");
        Library library = open_library(path);
        auto initialize = reinterpret_cast<Init>(library_symbol(library, "pginit"));
        auto call = reinterpret_cast<Call>(library_symbol(library, "stata_call"));
        require(initialize && call, "plugin SPI entrypoints are missing");
        runtime_path = loaded_openmp_runtime(library);
        ST_plugin api = spi();
        require(initialize(&api) == SD_PLUGINVER, "plugin SPI version mismatch");
        one = run(call, kind == "xhdfe", 1);
        require(one.rc == 0 && one.numeric_ok, "one-thread numerical control failed: " + one.error);
        require(scalar(one, "requested") == 1 && scalar(one, "effective") == 1 &&
                scalar(one, "used") == 1 && scalar(one, "workers") == 1,
                "one-thread diagnostics are not actual serial work");
        two = run(call, kind == "xhdfe", 2);
        if (two.rc == 0) {
            require(two.numeric_ok && one.numerical.size() == two.numerical.size(), "two-thread numerical control failed");
            require(scalar(one, "df_a") == scalar(two, "df_a"), "absorbed DoF depend on threads");
            if (kind == "xhdfe")
                require(scalar(one, "df_r") == scalar(two, "df_r"), "residual DoF depend on threads");
            for (std::size_t i = 0; i < one.numerical.size(); ++i) {
                const double error = std::abs(one.numerical[i]-two.numerical[i]);
                double tolerance = 1e-10;
                if (kind == "xhdfe" && i >= 2 && i <= 5)
                    tolerance = 1e-8 * std::sqrt(one.numerical[2] * one.numerical[5]);
                if (kind == "xhdfe" && i == 6)
                    tolerance = 1e-10 * std::max(1.0, std::abs(one.numerical[i]));
                parity_error = std::max(parity_error, error);
                require(error <= tolerance, "thread1/thread2 numerical disagreement");
            }
            numerical_parity = true;
        }
        if (expect_serial) {
            require(scalar(one, "openmp") == 0, "negative control is not a serial core build");
            require(two.rc != 0 || (scalar(two, "openmp") == 0 && scalar(two, "used") == 1 &&
                    scalar(two, "workers") == 1), "serial negative unexpectedly reported parallel work");
            serial_rejected = true;
            reason = "serial artifact rejected by the OpenMP release gate";
        } else {
            require(two.rc == 0, "plugin rejected the two-thread request: " + two.error);
            require(scalar(one, "openmp") == 1 && scalar(two, "openmp") == 1, "OpenMP is disabled");
            require(scalar(two, "capacity") >= 2, "runtime capacity is below two workers");
            require(scalar(two, "requested") == 2 && scalar(two, "effective") == 2 &&
                    scalar(two, "used") == 2 && scalar(two, "workers") == 2,
                    "two actual estimator workers were not observed");
            require(!runtime_path.empty(), "loaded OpenMP runtime could not be identified");
            passed = true;
        }
    } catch (const std::exception& error) { reason = error.what(); }
    std::cout << std::setprecision(17)
              << "{\"schema\":\"xhdfe-native-plugin-openmp-v1\",\"status\":"
              << quote(passed ? "PASS" : serial_rejected ? "EXPECTED_SERIAL_REJECTION" : "FAIL")
              << ",\"release_gate_pass\":" << (passed ? "true" : "false")
              << ",\"serial_negative_pass\":" << (serial_rejected ? "true" : "false")
              << ",\"reason\":" << quote(reason) << ",\"plugin_kind\":" << quote(kind)
              << ",\"plugin_path\":" << quote(path) << ",\"process_arch\":" << quote(architecture())
              << ",\"native_arch\":" << quote(native_arch)
              << ",\"expected_arch\":" << quote(expected_arch) << ",\"translated\":" << translation
              << ",\"openmp_runtime_path\":" << quote(runtime_path)
              << ",\"fixture_rows\":" << kRows << ",\"numerical_parity_pass\":" << (numerical_parity ? "true" : "false")
              << ",\"full_numeric_parity_max_error\":" << parity_error
              << ",\"thread1\":";
    print_snapshot(one); std::cout << ",\"thread2\":"; print_snapshot(two);
    std::cout << ",\"scope\":\"Native SPI and estimator work only; not the Stata ado layer or full feature/performance certification\"}\n";
    return passed || serial_rejected ? 0 : 1;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** wide_argv) {
    std::vector<std::string> storage;
    storage.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) storage.push_back(to_utf8(wide_argv[i]));
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& argument : storage) argv.push_back(argument.data());
    return probe_main(argc, argv.data());
}
#else
int main(int argc, char** argv) { return probe_main(argc, argv); }
#endif
