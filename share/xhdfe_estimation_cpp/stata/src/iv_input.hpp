#ifndef HDFE_IV_INPUT_HPP
#define HDFE_IV_INPUT_HPP

#include <Eigen/Dense>
#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace hdfe::detail {
namespace iv_input_detail {

// Canonical binary64 significand/exponent, including subnormals. No floating
// arithmetic is used: equality must survive fast-math, overflow and underflow.
struct BinaryFactor {
    std::uint64_t significand = 0;
    int exponent = 0;
    bool negative = false;
    bool finite = true;
};
inline BinaryFactor factor(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned encoded = static_cast<unsigned>((bits >> 52) & 2047);
    BinaryFactor result;
    result.finite = encoded != 2047;
    result.negative = (bits >> 63) != 0;
    result.significand = bits & UINT64_C(0x000fffffffffffff);
    if (encoded) result.significand |= UINT64_C(0x0010000000000000);
    result.exponent = encoded ? static_cast<int>(encoded) - 1075 : -1074;
    if (!result.significand) return result;
    // Bounded portable count of trailing zero bits; no compiler-specific type.
    unsigned zeros = 0;
    auto& m = result.significand;
    if ((m & UINT64_C(0xffffffff)) == 0) { m >>= 32; zeros += 32; }
    if ((m & UINT64_C(0xffff)) == 0) { m >>= 16; zeros += 16; }
    if ((m & UINT64_C(0xff)) == 0) { m >>= 8; zeros += 8; }
    if ((m & UINT64_C(0xf)) == 0) { m >>= 4; zeros += 4; }
    if ((m & UINT64_C(0x3)) == 0) { m >>= 2; zeros += 2; }
    if ((m & UINT64_C(0x1)) == 0) { m >>= 1; zeros += 1; }
    result.exponent += static_cast<int>(zeros);
    return result;
}
struct BinaryProduct {
    std::uint64_t high, low;
};
inline BinaryProduct multiply(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t a0 = static_cast<std::uint32_t>(a), a1 = a >> 32;
    const std::uint64_t b0 = static_cast<std::uint32_t>(b), b1 = b >> 32;
    const auto low = a0 * b0, mid1 = a0 * b1, mid2 = a1 * b0;
    const auto middle = (low >> 32) + static_cast<std::uint32_t>(mid1) +
                        static_cast<std::uint32_t>(mid2);
    return {a1 * b1 + (mid1 >> 32) + (mid2 >> 32) + (middle >> 32),
            (middle << 32) | static_cast<std::uint32_t>(low)};
}
inline bool equal_products(const BinaryFactor& a, const BinaryFactor& b,
                           const BinaryFactor& c, const BinaryFactor& d) {
    if (!a.finite || !b.finite || !c.finite || !d.finite) return false;
    const bool left_zero = !a.significand || !b.significand;
    const bool right_zero = !c.significand || !d.significand;
    if (left_zero || right_zero) return left_zero && right_zero;
    if ((a.negative != b.negative) != (c.negative != d.negative) ||
        a.exponent + b.exponent != c.exponent + d.exponent) return false;
    const auto left = multiply(a.significand, b.significand);
    const auto right = multiply(c.significand, d.significand);
    return left.high == right.high && left.low == right.low;
}

inline bool proportional(const Eigen::MatrixXd& instruments, Eigen::Index a,
                         Eigen::Index b, Eigen::Index pivot, int num_threads,
                         ParallelWorkObserver* observer) {
    const auto ap = factor(instruments(pivot, a));
    const auto bp = factor(instruments(pivot, b));
    const auto matches_row = [&](Eigen::Index i) {
        return equal_products(factor(instruments(i, a)), bp,
                              factor(instruments(i, b)), ap);
    };
    const Eigen::Index n = instruments.rows();
    // Most independent pairs stop here; a surviving pair is checked on EVERY
    // row, so a small identifying direction confined to one row is preserved.
    for (Eigen::Index i : {Eigen::Index(0), n / 2, n - 1})
        if (!matches_row(i)) return false;
    int threads = 1;
#ifdef HDFE_USE_OPENMP
    threads = std::max(1, num_threads);
#endif
    const int chunks = deterministic_parallel_chunk_count(n);
    int matches = 1;
    if (observer) observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&:matches)
#endif
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (observer) observer->observe_work();
        const auto begin = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const auto end = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (Eigen::Index i = begin; i < end; ++i) {
            if (!matches_row(i)) { matches = 0; break; }
        }
    }
    if (observer) observer->end_region();
    return matches != 0;
}
} // namespace iv_input_detail

// Preserve the original instrument span before independent FE transformations
// can perturb exact proportionality. Only nonzero, exactly proportional user
// columns are removed. Zero columns and general linear dependencies retain
// their existing treatment; numerical near-collinearity is never an omission.
inline std::optional<Eigen::MatrixXd> remove_exact_proportional_instruments(
    const Eigen::MatrixXd& instruments, int num_threads,
    ParallelWorkObserver* observer) {
    const Eigen::Index n = instruments.rows(), k = instruments.cols();
    if (n == 0 || k < 2) return std::nullopt;
    std::vector<Eigen::Index> retained;
    std::vector<Eigen::Index> pivots;
    retained.reserve(static_cast<std::size_t>(k));
    pivots.reserve(static_cast<std::size_t>(k));
    for (Eigen::Index j = 0; j < k; ++j) {
        Eigen::Index pivot = 0;
        while (pivot < n && iv_input_detail::factor(instruments(pivot, j)).significand == 0)
            ++pivot;
        bool redundant = false;
        if (pivot < n) {
            for (std::size_t s = 0; s < retained.size(); ++s) {
                if (pivots[s] == pivot && iv_input_detail::proportional(
                        instruments, retained[s], j, pivot, num_threads, observer)) {
                    redundant = true;
                    break;
                }
            }
        }
        if (!redundant) { retained.push_back(j); pivots.push_back(pivot); }
    }
    if (retained.size() == static_cast<std::size_t>(k)) return std::nullopt;
    Eigen::MatrixXd reduced(n, static_cast<Eigen::Index>(retained.size()));
    for (Eigen::Index j = 0; j < reduced.cols(); ++j)
        reduced.col(j) = instruments.col(retained[static_cast<std::size_t>(j)]);
    return reduced;
}
} // namespace hdfe::detail
#endif
