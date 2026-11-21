#ifndef XHDFE_TESTS_REDUCED_CUDA_BENCH_FROZEN_OPERATOR_HPP
#define XHDFE_TESTS_REDUCED_CUDA_BENCH_FROZEN_OPERATOR_HPP
// FROZEN reference for the bench only (package 2, integration, 17 November 2026): the CUDA
// operator of the tier as it stood before package 2 (main 90f8aeda8 plus the stage-0 trace
// timers), copied byte for byte from src/reduced_schur_pcg_cuda.hpp, with only its include
// guard and namespace renamed (hdfe::detail::reduced_cuda -> xhdfe_bench::frozen), so that the
// bench can time and check it in the same run as the integrated operator. Not built into any
// library or plugin; never edit it to follow src/.

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xhdfe_bench::frozen {

enum class ErrorKind { Memory, Unavailable, Fatal };

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    ErrorKind kind() const noexcept { return kind_; }
private:
    ErrorKind kind_;
};

// Only active operator cells are packed. ptr/CSR entries remain in the CPU
// geometry's stable segment/cell order. No Schur matrix is assembled.
struct Layout {
    int dims = 0, levels = 0, segments = 0, cells = 0, lanes = 0;
    std::vector<int> offsets, segment_ptr, cell_segment, codes, mult;
    std::vector<double> segment_inverse;
    std::vector<int> csr_offsets, csr_ptr, csr_cell;
    std::vector<int> a_item_segment, a_item_begin, a_item_end, a_item_ptr;
    std::vector<int> b_item_level, b_item_begin, b_item_end, b_item_ptr;
    std::size_t host_bytes() const;
    std::size_t device_bytes() const;
};

struct Statistics {
    std::size_t host_layout_bytes = 0, device_payload_bytes = 0;
    std::size_t h2d_bytes = 0, d2h_bytes = 0;
    int calls = 0;
    bool timed = false;
    double setup_seconds = 0.0, apply_seconds = 0.0;
    double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
};

// Trace-only splits for the attribution line (event=attribution), filled only
// when a caller passes them. They live with the caller, not in Statistics or
// in the operator's state, so the host state the admission counts is unchanged.
struct SetupSplit {
    double validate_seconds = 0.0;  // validate(layout)
    double device_seconds = 0.0;    // device count, stream, free memory, events
    double alloc_seconds = 0.0;     // every cudaMalloc of the operator
    double copy_seconds = 0.0;      // the layout's host-to-device copies, as issued
    double sync_seconds = 0.0;      // the stream synchronisation that ends the set-up
};
struct KernelSplit {  // CUDA events between the four kernels; their sum is kernel_ms
    double segment_items_ms = 0.0, segment_finish_ms = 0.0, level_items_ms = 0.0, level_finish_ms = 0.0;
};

class Operator {
public:
    Operator(const Layout& layout, std::size_t device_budget, bool profile = false, SetupSplit* setup = nullptr);
    ~Operator();
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    // kernels: trace only, read when the operator profiles.
    void apply(const double* in, double* out, KernelSplit* kernels = nullptr);
    const Statistics& statistics() const noexcept;
    static std::size_t host_state_bytes() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};

bool enabled() noexcept;

} // namespace xhdfe_bench::frozen
#endif
