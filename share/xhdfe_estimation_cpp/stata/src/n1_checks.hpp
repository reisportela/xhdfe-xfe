#ifndef HDFE_N1_NORMAL_HPP
#define HDFE_N1_NORMAL_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "hdfe/ieee_bits.hpp"

namespace hdfe { namespace detail {

inline thread_local bool n1_capture_active = false;

struct N1Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct N1RefinementBudget {
    int maximum;
    bool used = false;
    int remaining(int completed, const N1Failure& failure) {
        if (used || completed >= maximum)
            throw N1Failure(std::string(failure.what()) +
                (used ? "; the single N1 refinement was exhausted" :
                        "; no absorption iteration budget remains"));
        used = true;
        return maximum - completed;
    }
};

struct ScopedN1Capture {
    bool previous;
    // Per-row evidence is captured only when `enable` is true (audit mode);
    // the default fit keeps the free normal-equation defect check instead.
    explicit ScopedN1Capture(bool enable = true) : previous(n1_capture_active) {
        if (enable) n1_capture_active = true;
    }
    ~ScopedN1Capture() { n1_capture_active = previous; }
    ScopedN1Capture(const ScopedN1Capture&) = delete;
    ScopedN1Capture& operator=(const ScopedN1Capture&) = delete;
};

// Private evidence from the same rows and coefficients used to form u.
// Acceptance is relative to ||score||_W ||u||_W, plus a separate arithmetic
// allowance; the scale of the fitted values never replaces ||u||_W.
struct N1NormalCoordinate {
    long double moment = 0;
    long double score_sq = 0;
    long double absolute_products = 0;
    long double formation_products = 0;
};

struct N1NormalChunk {
    long double residual_sq = 0;
    long double within_y_sq = 0;
    std::vector<long double> within_x_sq;
    std::vector<N1NormalCoordinate> coordinates;
    int rows = 0;
    bool finite = true;

    explicit N1NormalChunk(int p = 0)
        : within_x_sq(p, 0), coordinates(p) {}

    template<class Score, class Actual, class Coefficient>
    void add(double y, double residual, double weight, int p,
             Score score, Actual actual, Coefficient coefficient) {
        ++rows;
        finite = finite && ieee_finite(y) && ieee_finite(residual) &&
                 ieee_finite(weight);
        const long double w = weight, u = residual;
        long double formation = std::abs(static_cast<long double>(y));
        for (int j = 0; j < p; ++j) {
            const double x_value = actual(j);
            finite = finite && ieee_finite(x_value);
            const double b_value = coefficient(j);
            finite = finite && ieee_finite(b_value);
            const long double x = x_value, b = b_value;
            formation += std::abs(x * b);
            within_x_sq[j] += x * x;
        }
        residual_sq += w * u * u;
        within_y_sq += static_cast<long double>(y) * y;
        for (int j = 0; j < p; ++j) {
            const double s_value = score(j);
            finite = finite && ieee_finite(s_value);
            const long double s = s_value, ws = w * s;
            auto& c = coordinates[j];
            c.moment += ws * u;
            c.score_sq += ws * s;
            c.absolute_products += std::abs(ws * u);
            c.formation_products += std::abs(ws) * formation;
        }
    }
};

struct N1NormalEvidence {
    std::vector<N1NormalChunk> chunks;
    int columns = 0;

    N1NormalEvidence(int count, int p) : columns(p) {
        chunks.reserve(count);
        for (int k = 0; k < count; ++k) chunks.emplace_back(p);
    }

    // gamma_k bounds k rounded operations. A bound outside its domain is
    // inconclusive, never an excuse to reject or tighten the solver.
    static long double gamma(long double k) {
        const long double ku = k *
            (std::numeric_limits<double>::epsilon() / 2.0L);
        return ku < 1 ? ku / (1 - ku) :
            std::numeric_limits<long double>::infinity();
    }

    struct Summary {
        long double rss = 0;
        long double y_sq = 0;
        std::vector<long double> x_sq;
    };

    Summary summary() const {
        Summary out;
        out.x_sq.assign(columns, 0);
        for (const auto& chunk : chunks) {
            if (!chunk.finite)
                throw N1Failure("N1: non-finite estimation residual; no estimates returned");
            out.rss += chunk.residual_sq;
            out.y_sq += chunk.within_y_sq;
            for (int j = 0; j < columns; ++j)
                out.x_sq[j] += chunk.within_x_sq[j];
        }
        return out;
    }

    void require(double tolerance) const {
        long double rss = 0;
        int longest = 0;
        std::vector<N1NormalCoordinate> totals(columns);
        for (const auto& chunk : chunks) {
            if (!chunk.finite)
                throw N1Failure(
                    "N1: non-finite estimation residual; no estimates returned");
            rss += chunk.residual_sq;
            longest = std::max(longest, chunk.rows);
            for (int j = 0; j < columns; ++j) {
                totals[j].moment += chunk.coordinates[j].moment;
                totals[j].score_sq += chunk.coordinates[j].score_sq;
                totals[j].absolute_products += chunk.coordinates[j].absolute_products;
                totals[j].formation_products += chunk.coordinates[j].formation_products;
            }
        }
        const long double accumulation = gamma(longest + chunks.size() + 4);
        const long double formation = gamma(2 * columns + 2);
        for (int j = 0; j < columns; ++j) {
            const auto& c = totals[j];
            const long double scale = std::sqrt(c.score_sq) * std::sqrt(rss);
            const long double arithmetic =
                accumulation * c.absolute_products +
                (4 * accumulation + formation) * c.formation_products;
            const long double defect = std::abs(c.moment);
            if (defect > tolerance * scale + arithmetic) {
                std::ostringstream message;
                message << "N1: normal equation failed at column " << j + 1
                        << " (defect=" << defect << ", limit="
                        << tolerance * scale << ", arithmetic=" << arithmetic
                        << "); no estimates returned";
                throw N1Failure(message.str());
            }
        }
    }
};

// Keep the signed group sums already formed by the absorption certificate.
// The three extra columns per moment contain ||d||_W^2, ||Wd||^2 and the
// actual number of summands; no incidence structure or new RHS is constructed.
struct N1FeEvidence {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>;
    struct Block {
        Matrix sums;
        int rhs_count;
        int moment_count;
        int extras;
        int chunks;
    };
    std::vector<Block> blocks;

    void add_block(Matrix sums, int rhs_count, int moment_count,
                   int extras, int chunks) {
        blocks.push_back({std::move(sums), rhs_count, moment_count, extras, chunks});
    }

    void require(const Eigen::VectorXd& beta, const std::vector<int>& columns,
                 const N1NormalEvidence& normal, double tolerance) const {
        if (beta.size() != static_cast<int>(columns.size()) ||
            normal.columns != beta.size())
            throw N1Failure("N1: inconsistent fitted-column evidence; no estimates returned");
        const auto norms = normal.summary();
        long double formation_norm = std::sqrt(norms.y_sq);
        for (int j = 0; j < beta.size(); ++j)
            formation_norm += std::abs(static_cast<long double>(beta[j])) *
                              std::sqrt(norms.x_sq[j]);
        for (std::size_t dim = 0; dim < blocks.size(); ++dim) {
            const auto& block = blocks[dim];
            for (int moment = 0; moment < block.moment_count; ++moment) {
                const int base = moment * block.rhs_count;
                for (int group = 0; group < block.sums.rows(); ++group) {
                    long double value = block.sums(group, base);
                    long double absolute = std::abs(value);
                    for (int j = 0; j < beta.size(); ++j) {
                        if (columns[j] < 0 || columns[j] + 1 >= block.rhs_count)
                            throw N1Failure("N1: invalid absorbed-column evidence; no estimates returned");
                        const long double term = static_cast<long double>(beta[j]) *
                            block.sums(group, base + columns[j] + 1);
                        value -= term;
                        absolute += std::abs(term);
                    }
                    const double diagonal = block.sums(group, block.extras + 3 * moment);
                    const double operator_sq = block.sums(group, block.extras + 3 * moment + 1);
                    const double count = block.sums(group, block.extras + 3 * moment + 2);
                    if (!ieee_finite(static_cast<double>(value)) ||
                        !ieee_finite(diagonal) || !ieee_finite(operator_sq) ||
                        diagonal < 0 || operator_sq < 0)
                        throw N1Failure("N1: non-finite FE moment; no estimates returned");
                    const long double accumulation = N1NormalEvidence::gamma(count + block.chunks + 2);
                    const long double arithmetic =
                        accumulation * std::sqrt(static_cast<long double>(operator_sq)) * formation_norm +
                        N1NormalEvidence::gamma(2 * beta.size() + 2) * absolute;
                    const long double scale = std::sqrt(static_cast<long double>(diagonal)) *
                                              std::sqrt(norms.rss);
                    if (std::abs(value) > tolerance * scale + arithmetic) {
                        std::ostringstream message;
                        message << "N1: FE moment failed at block " << dim + 1
                                << ", group " << group + 1 << ", moment " << moment + 1
                                << " (defect=" << std::abs(value) << ", limit="
                                << tolerance * scale << ", arithmetic=" << arithmetic
                                << "); no estimates returned";
                        throw N1Failure(message.str());
                    }
                }
            }
        }
    }
};

struct N1ReconstructionEvidence {
    long double error_sq = 0;
    long double scale_sq = 0;
    long double arithmetic_sq = 0;
    bool finite = true;

    // `partial` and `fe_total` use the fixed pre-recovery intercept gauge;
    // published intercept/FE shifts cancel before entering this evidence.
    // The scale never includes magnitudes of cancelling individual alphas.
    void add(double partial, double fe_total, double published_u, double ols_u,
             double weight, int operations) {
        finite = finite && ieee_finite(partial) && ieee_finite(fe_total) &&
                 ieee_finite(published_u) && ieee_finite(ols_u);
        const long double error = static_cast<long double>(partial) - fe_total - ols_u;
        const long double identity = static_cast<long double>(partial) - fe_total - published_u;
        const long double magnitude = std::abs(static_cast<long double>(partial)) +
            std::abs(static_cast<long double>(fe_total)) + std::abs(static_cast<long double>(ols_u));
        const long double arithmetic = N1NormalEvidence::gamma(operations) * magnitude;
        const long double excess = std::max(0.0L, std::max(std::abs(error), std::abs(identity)) - arithmetic);
        error_sq += weight * excess * excess;
        scale_sq += weight * (static_cast<long double>(partial) * partial +
                              static_cast<long double>(ols_u) * ols_u);
        arithmetic_sq += weight * arithmetic * arithmetic;
    }

    void require(double tolerance) const {
        if (!finite || std::sqrt(error_sq) > tolerance * std::sqrt(scale_sq)) {
            std::ostringstream message;
            message << "N1: published FE reconstruction failed (defect="
                    << std::sqrt(error_sq) << ", limit="
                    << tolerance * std::sqrt(scale_sq)
                    << "); no estimates returned";
            throw N1Failure(message.str());
        }
    }
};

}} // namespace hdfe::detail
#endif
