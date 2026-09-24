#include "iv.hpp"
#include "wide_float.hpp"
#include <type_traits>
#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/parallel_work_observer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hdfe {
namespace detail {
namespace {

bool finite_bits(double value) noexcept {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value),
                  "xhdfe IV validation requires IEEE-754 binary64");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

bool matrix_finite_bits(const Eigen::MatrixXd& matrix) noexcept {
    std::uint64_t invalid = 0;
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
        invalid |= static_cast<std::uint64_t>(!finite_bits(matrix.data()[i]));
    }
    return invalid == 0;
}

double rank_tolerance(Eigen::Index rows, Eigen::Index cols) noexcept {
    return 16.0 * std::numeric_limits<double>::epsilon() *
           static_cast<double>(std::max<Eigen::Index>({1, rows, cols}));
}

Eigen::VectorXd positive_column_scales(const Eigen::MatrixXd& gram,
                                       const char* label) {
    Eigen::VectorXd scales(gram.rows());
    for (Eigen::Index j = 0; j < gram.rows(); ++j) {
        const double diagonal = gram(j, j);
        if (!finite_bits(diagonal) || !(diagonal > 0.0)) {
            std::ostringstream message;
            message << label << " contains a zero or non-finite column at "
                    << j << " (Gram diagonal " << diagonal << ")";
            throw std::runtime_error(message.str());
        }
        scales[j] = std::sqrt(diagonal);
    }
    return scales;
}

Eigen::MatrixXd normalized_gram(const Eigen::MatrixXd& gram,
                                const Eigen::VectorXd& scales,
                                const char* label) {
    Eigen::MatrixXd normalized = gram;
    for (Eigen::Index j = 0; j < normalized.cols(); ++j) {
        normalized.col(j) /= scales[j];
    }
    for (Eigen::Index i = 0; i < normalized.rows(); ++i) {
        normalized.row(i) /= scales[i];
    }
    normalized = 0.5 * (normalized + normalized.transpose()).eval();
    if (!matrix_finite_bits(normalized)) {
        throw std::runtime_error(std::string(label) +
                                 " produced a non-finite normalized Gram matrix");
    }
    return normalized;
}

bool gram_is_clearly_full_rank(const Eigen::MatrixXd& normalized,
                               double* relative_condition = nullptr) {
    if (normalized.cols() == 0) {
        return true;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(normalized,
                                                         Eigen::EigenvaluesOnly);
    if (eigen.info() != Eigen::Success ||
        !matrix_finite_bits(eigen.eigenvalues())) {
        return false;
    }
    const double ambiguity = 1024.0 * std::numeric_limits<double>::epsilon() *
                             static_cast<double>(std::max<Eigen::Index>(
                                 1, normalized.cols()));
    if (relative_condition) {
        *relative_condition = eigen.eigenvalues().minCoeff() /
                              eigen.eigenvalues().maxCoeff();
    }
    return eigen.eigenvalues().minCoeff() > ambiguity;
}

int direct_normalized_rank(Eigen::MatrixXd matrix,
                           const Eigen::VectorXd& scales,
                           double tolerance) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        matrix.col(j) /= scales[j];
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(matrix);
    qr.setThreshold(tolerance);
    return static_cast<int>(qr.rank());
}

int require_full_rank(const Eigen::MatrixXd& matrix,
                       const Eigen::MatrixXd& gram,
                       const char* label,
                       const std::vector<Eigen::Index>* selected_columns = nullptr) {
    const Eigen::Index checked_columns = selected_columns
                                             ? static_cast<Eigen::Index>(selected_columns->size())
                                             : matrix.cols();
    if (gram.rows() != gram.cols() || gram.cols() != checked_columns) {
        throw std::runtime_error("Internal IV rank-check dimension mismatch");
    }
    if (checked_columns == 0) {
        return 0;
    }
    const Eigen::VectorXd scales = positive_column_scales(gram, label);
    const Eigen::MatrixXd normalized = normalized_gram(gram, scales, label);
    double relative_condition = 0.0;
    if (gram_is_clearly_full_rank(normalized, &relative_condition)) {
        // Fixed arithmetic dispatch: eps(double)^(1/4), not a rank/fit tolerance.
        // A single correction needs a well-resolved Gram; otherwise use QR.
        if (relative_condition <= 0x1p-26) return 2;
        return relative_condition <= 0x1p-13 ? 1 : 0;
    }
    Eigen::MatrixXd direct;
    if (selected_columns) {
        direct.resize(matrix.rows(), checked_columns);
        for (Eigen::Index j = 0; j < checked_columns; ++j) {
            direct.col(j) = matrix.col((*selected_columns)[static_cast<std::size_t>(j)]);
        }
    } else {
        direct = matrix;
    }
    const int rank = direct_normalized_rank(
        std::move(direct), scales, rank_tolerance(matrix.rows(), checked_columns));
    if (rank != checked_columns) {
        // Redundant instruments can still identify every endogenous regressor.
        // Let the orthogonal route check the identified instrument span.
        return 2;
    }
    return 2;
}

Eigen::MatrixXd residualize_for_rank(const Eigen::MatrixXd& matrix,
                                     const Eigen::MatrixXd& exogenous,
                                     double tolerance) {
    if (exogenous.cols() == 0) {
        return matrix;
    }
    Eigen::MatrixXd normalized_exogenous = exogenous;
    for (Eigen::Index j = 0; j < normalized_exogenous.cols(); ++j) {
        const double scale = normalized_exogenous.col(j).stableNorm();
        if (!finite_bits(scale) || !(scale > 0.0)) {
            throw std::runtime_error(
                "Exogenous IV design contains a zero or non-finite column");
        }
        normalized_exogenous.col(j) /= scale;
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(normalized_exogenous);
    qr.setThreshold(tolerance);
    if (qr.rank() != normalized_exogenous.cols()) {
        throw std::runtime_error("Exogenous IV design is rank deficient");
    }
    Eigen::MatrixXd residual = matrix - normalized_exogenous * qr.solve(matrix);
    if (!matrix_finite_bits(residual)) {
        throw std::runtime_error("IV residualization produced NaN or Inf");
    }
    return residual;
}

int preflight_identification(const Eigen::MatrixXd& weighted_instruments,
                              const Eigen::MatrixXd& weighted_endogenous,
                              const Eigen::MatrixXd& ztz,
                              const Eigen::MatrixXd& ztq,
                              int num_exogenous) {
    const Eigen::Index n = weighted_instruments.rows();
    const Eigen::Index k = weighted_instruments.cols();
    const Eigen::Index endogenous_cols = weighted_endogenous.cols();
    if (n == 0) {
        throw std::runtime_error("IV estimation requires at least one observation");
    }
    if (num_exogenous < 0 || num_exogenous > k) {
        throw std::runtime_error("Internal IV error: invalid exogenous column count");
    }
    const Eigen::Index excluded_cols = k - num_exogenous;
    if (excluded_cols < endogenous_cols) {
        throw std::runtime_error(
            "IV model is underidentified: fewer excluded instruments than endogenous regressors");
    }
    if (!matrix_finite_bits(ztz) || !matrix_finite_bits(ztq)) {
        throw std::runtime_error("IV design produced a non-finite cross-product");
    }
    if (k > n) return 2;
    for (Eigen::Index j = num_exogenous; j < k; ++j) {
        if (ztz(j, j) == 0.0) return 2;
    }

    // Defensively ignore exactly zero exogenous columns in the instrument-rank
    // preflight.  The current FE path removes an absorbed constant before
    // 2SLS, but direct callers and older cached designs may still supply one.
    std::vector<Eigen::Index> checked_columns;
    checked_columns.reserve(static_cast<std::size_t>(k));
    for (Eigen::Index j = 0; j < num_exogenous; ++j) {
        const double diagonal = ztz(j, j);
        if (!finite_bits(diagonal) || diagonal < 0.0) {
            throw std::runtime_error("Exogenous IV design has a non-finite norm");
        }
        if (diagonal > 0.0) {
            checked_columns.push_back(j);
        }
    }
    const Eigen::Index effective_num_exogenous =
        static_cast<Eigen::Index>(checked_columns.size());
    for (Eigen::Index j = num_exogenous; j < k; ++j) {
        checked_columns.push_back(j);
    }

    std::optional<Eigen::MatrixXd> compact_ztz;
    std::optional<Eigen::MatrixXd> compact_ztq;
    const Eigen::MatrixXd* identification_ztz = &ztz;
    const Eigen::MatrixXd* identification_ztq = &ztq;
    if (effective_num_exogenous != num_exogenous) {
        const Eigen::Index effective_k =
            static_cast<Eigen::Index>(checked_columns.size());
        compact_ztz.emplace(effective_k, effective_k);
        compact_ztq.emplace(effective_k, endogenous_cols);
        for (Eigen::Index i = 0; i < effective_k; ++i) {
            compact_ztq->row(i) = ztq.row(
                checked_columns[static_cast<std::size_t>(i)]);
            for (Eigen::Index j = 0; j < effective_k; ++j) {
                (*compact_ztz)(i, j) = ztz(
                    checked_columns[static_cast<std::size_t>(i)],
                    checked_columns[static_cast<std::size_t>(j)]);
            }
        }
        identification_ztz = &*compact_ztz;
        identification_ztq = &*compact_ztq;
    }

    const int route = require_full_rank(weighted_instruments, *identification_ztz,
                      "IV instrument matrix",
                      effective_num_exogenous == num_exogenous
                          ? nullptr
                          : &checked_columns);
    if (route == 2) return 2;

    const Eigen::MatrixXd qtq =
        weighted_endogenous.transpose() * weighted_endogenous;
    if (!matrix_finite_bits(qtq)) {
        throw std::runtime_error("IV endogenous design produced a non-finite cross-product");
    }

    Eigen::MatrixXd excluded_gram =
        identification_ztz->bottomRightCorner(excluded_cols, excluded_cols);
    Eigen::MatrixXd endogenous_gram = qtq;
    Eigen::MatrixXd first_stage = identification_ztq->bottomRows(excluded_cols);
    if (effective_num_exogenous > 0) {
        const Eigen::MatrixXd exogenous_gram =
            identification_ztz->topLeftCorner(effective_num_exogenous,
                                               effective_num_exogenous);
        const Eigen::MatrixXd exogenous_excluded =
            identification_ztz->topRightCorner(effective_num_exogenous,
                                                excluded_cols);
        const Eigen::MatrixXd exogenous_endogenous =
            identification_ztq->topRows(effective_num_exogenous);
        Eigen::LDLT<Eigen::MatrixXd> exogenous_solver(exogenous_gram);
        if (exogenous_solver.info() != Eigen::Success) {
            throw std::runtime_error("Exogenous IV design is rank deficient");
        }
        const Eigen::MatrixXd solved_excluded =
            exogenous_solver.solve(exogenous_excluded);
        const Eigen::MatrixXd solved_endogenous =
            exogenous_solver.solve(exogenous_endogenous);
        if (!matrix_finite_bits(solved_excluded) ||
            !matrix_finite_bits(solved_endogenous)) {
            throw std::runtime_error("IV residualization solve produced NaN or Inf");
        }
        excluded_gram.noalias() -=
            exogenous_excluded.transpose() * solved_excluded;
        endogenous_gram.noalias() -=
            exogenous_endogenous.transpose() * solved_endogenous;
        first_stage.noalias() -=
            exogenous_excluded.transpose() * solved_endogenous;
    }

    // A cancelled Schur diagonal is not evidence of an unidentified model.
    for (Eigen::Index j = 0; j < excluded_gram.rows(); ++j) {
        if (!(excluded_gram(j, j) > 0.0)) return 2;
    }
    for (Eigen::Index j = 0; j < endogenous_gram.rows(); ++j) {
        if (!(endogenous_gram(j, j) > 0.0)) return 2;
    }
    const Eigen::VectorXd excluded_scales =
        positive_column_scales(excluded_gram,
                               "FWL-residualized excluded instruments");
    const Eigen::MatrixXd excluded_normalized =
        normalized_gram(excluded_gram, excluded_scales,
                        "FWL-residualized excluded instruments");
    if (!gram_is_clearly_full_rank(excluded_normalized)) {
        const Eigen::MatrixXd exogenous =
            [&]() {
                Eigen::MatrixXd selected(weighted_instruments.rows(),
                                         effective_num_exogenous);
                for (Eigen::Index j = 0; j < effective_num_exogenous; ++j) {
                    selected.col(j) = weighted_instruments.col(
                        checked_columns[static_cast<std::size_t>(j)]);
                }
                return selected;
            }();
        const Eigen::MatrixXd excluded =
            weighted_instruments.rightCols(excluded_cols);
        Eigen::MatrixXd excluded_residual = residualize_for_rank(
            excluded, exogenous, rank_tolerance(n, effective_num_exogenous));
        const int rank = direct_normalized_rank(
            std::move(excluded_residual), excluded_scales,
            rank_tolerance(n, excluded_cols));
        if (rank != excluded_cols) {
            std::ostringstream message;
            message << "FWL-residualized excluded instruments are rank deficient (rank "
                    << rank << " < " << excluded_cols << ")";
            throw std::runtime_error(message.str());
        }
    }

    const Eigen::VectorXd endogenous_scales =
        positive_column_scales(endogenous_gram,
                               "FWL-residualized endogenous regressors");
    for (Eigen::Index i = 0; i < first_stage.rows(); ++i) {
        first_stage.row(i) /= excluded_scales[i];
    }
    for (Eigen::Index j = 0; j < first_stage.cols(); ++j) {
        first_stage.col(j) /= endogenous_scales[j];
    }
    if (!matrix_finite_bits(first_stage)) {
        throw std::runtime_error("IV first-stage relation contains NaN or Inf");
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> first_stage_svd(
        first_stage, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = first_stage_svd.singularValues();
    // The SVD is performed on the small, scale-normalized excluded-by-
    // endogenous relation.  Using the observation count here would turn the
    // numerical rank guard into a sample-size-dependent weak-IV cutoff.
    const double tolerance =
        rank_tolerance(first_stage.rows(), first_stage.cols());
    int useful_rank = 0;
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        useful_rank += singular[i] > tolerance ? 1 : 0;
    }
    if (useful_rank < endogenous_cols) return 2;
    // Small identified first-stage correlations also amplify projection roundoff.
    // Use the same fixed arithmetic dispatch; do not reject a weak instrument.
    if (singular.minCoeff() <= 0x1p-13) return 2;
    return route;
}


void validate_projection(const Eigen::MatrixXd& ztz,
                         const Eigen::MatrixXd& ztq,
                         const Eigen::MatrixXd& gamma,
                         const Eigen::MatrixXd& fitted,
                         Eigen::Index observations) {
    if (!matrix_finite_bits(gamma) || !matrix_finite_bits(fitted)) {
        throw std::runtime_error("IV projection produced NaN or Inf");
    }
    const Eigen::MatrixXd residual = ztq - ztz * gamma;
    if (!matrix_finite_bits(residual)) {
        throw std::runtime_error("IV projection residual contains NaN or Inf");
    }
    const double residual_norm = residual.stableNorm();
    const double scale = ztq.stableNorm() + ztz.stableNorm() * gamma.stableNorm();
    const double limit = 256.0 * std::numeric_limits<double>::epsilon() *
                         static_cast<double>(std::max<Eigen::Index>(
                             {1, observations, ztz.cols()})) *
                         std::max(1.0, scale);
    if (!finite_bits(residual_norm) || residual_norm > limit) {
        throw std::runtime_error("IV projection failed its normal-equation residual check");
    }
}



// One residual correction uses the represented rows rather than subtracting
// rounded cross-products. Only the already-flagged moderate/weak path uses it.
Eigen::MatrixXd refine_projection(const Eigen::MatrixXd& instruments,
                                  const Eigen::MatrixXd& endogenous,
                                  const Eigen::VectorXd* weights,
                                  const Eigen::LDLT<Eigen::MatrixXd>& solver,
                                  Eigen::MatrixXd& gamma,
                                  ParallelWorkObserver* observer) {
    using Real = std::conditional_t<(std::numeric_limits<long double>::digits >
                                    std::numeric_limits<double>::digits), long double, OlsWide>;
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Index n = instruments.rows(), k = instruments.cols();
    const Eigen::Index q = endogenous.cols();
    const Matrix coefficients = gamma.cast<Real>();
    const int chunks = deterministic_parallel_chunk_count(n);
    std::vector<Matrix> defects(static_cast<std::size_t>(chunks), Matrix::Zero(k, q));
    Eigen::MatrixXd fitted(n, q), low(n, q);
    int threads = 1;
#ifdef HDFE_USE_OPENMP
    if (n >= 20000) threads = std::max(1, omp_get_max_threads());
#endif
    if (observer) observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (observer) observer->observe_work();
        Matrix& defect = defects[static_cast<std::size_t>(chunk)];
        const auto begin = deterministic_parallel_chunk_begin(n, chunk, chunks);
        const auto end = deterministic_parallel_chunk_end(n, chunk, chunks);
        for (Eigen::Index i = begin; i < end; ++i) {
            const Real weight = weights ? static_cast<Real>((*weights)[i]) : 1.0L;
            for (Eigen::Index column = 0; column < q; ++column) {
                Real value = 0;
                for (Eigen::Index j = 0; j < k; ++j) {
                    value += static_cast<Real>(instruments(i, j)) * coefficients(j, column);
                }
                fitted(i, column) = Eigen::internal::cast<Real, double>(value);
                low(i, column) = Eigen::internal::cast<Real, double>(value - static_cast<Real>(fitted(i, column)));
                const Real residual = (static_cast<Real>(endogenous(i, column)) - value) * weight;
                for (Eigen::Index j = 0; j < k; ++j) {
                    defect(j, column) += static_cast<Real>(instruments(i, j)) * residual;
                }
            }
        }
    }
    if (observer) observer->end_region();
    Matrix defect = Matrix::Zero(k, q);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        defect += defects[static_cast<std::size_t>(chunk)];
    }
    const Eigen::MatrixXd correction = solver.solve(defect.cast<double>());
    // Keep the low part until the small correction is added. Reconstructing
    // Z*(gamma+correction) would cancel the large instrumental coefficients.
    low.noalias() += instruments * correction;
    fitted += low;
    gamma += correction;
    return fitted;
}

// Resolve unstable first stages on the represented inputs, before weighting
// rounds away small directions. This is a solve, not an extra fit certificate.
Eigen::MatrixXd project_orthogonal(const Eigen::MatrixXd& instruments,
                                    const Eigen::MatrixXd& endogenous,
                                    int num_exogenous,
                                    const Eigen::VectorXd* weights,
                                    Eigen::MatrixXd* projection_low) {
    using Real = OlsWide;
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    const Eigen::Index n = instruments.rows();
    const Eigen::Index q = endogenous.cols();
    Matrix weighted = instruments.cast<Real>();
    Matrix response = endogenous.cast<Real>();
    Vector sqrt_w = Vector::Ones(n);
    if (weights) {
        sqrt_w = weights->cast<Real>().array().sqrt();
        weighted.array().colwise() *= sqrt_w.array();
        response.array().colwise() *= sqrt_w.array();
    }
    std::vector<Eigen::Index> columns;
    Eigen::Index exogenous = 0;
    for (Eigen::Index j = 0; j < instruments.cols(); ++j) {
        const Real norm = weighted.col(j).stableNorm();
        if (norm == 0) continue;
        if (!(norm > 0)) {
            throw std::runtime_error("IV instrument matrix contains a zero or non-finite column");
        }
        columns.push_back(j);
        exogenous += j < num_exogenous ? 1 : 0;
    }
    const Eigen::Index k = static_cast<Eigen::Index>(columns.size());
    const Eigen::Index excluded = k - exogenous;
    if (excluded < q) {
        throw std::runtime_error("IV model is underidentified: insufficient nonzero excluded instruments");
    }
    Matrix normalized(n, k);
    Vector scales(k);
    for (Eigen::Index j = 0; j < k; ++j) {
        normalized.col(j) = weighted.col(columns[static_cast<std::size_t>(j)]);
        scales[j] = normalized.col(j).stableNorm();
        normalized.col(j) /= scales[j];
    }
    Eigen::ColPivHouseholderQR<Matrix> qr(normalized);
    qr.setThreshold(static_cast<Real>(rank_tolerance(n, k)));
    if (qr.rank() != k) {
        // Resolve column redundancy in the arithmetic of this QR. Do not
        // erase a small represented direction just because FP64 called it zero.
        qr.setThreshold(Real(64) * std::numeric_limits<Real>::epsilon() *
                        Real(std::max(n, k)));
    }
    const Eigen::Index span_rank = qr.rank();

    // Identification also requires endogenouses outside the exogenous span.
    // Check before normalizing a FWL tail consisting only of roundoff.
    Matrix identifying(n, exogenous + q);
    identifying.leftCols(exogenous) = normalized.leftCols(exogenous);
    for (Eigen::Index j = 0; j < q; ++j) {
        const Real scale = response.col(j).stableNorm();
        if (!(scale > 0)) {
            throw std::runtime_error("IV first-stage relation is rank deficient");
        }
        identifying.col(exogenous + j) = response.col(j) / scale;
    }
    Eigen::ColPivHouseholderQR<Matrix> identifying_qr(identifying);
    identifying_qr.setThreshold(static_cast<Real>(rank_tolerance(n, exogenous + q)));
    if (identifying_qr.rank() != exogenous + q) {
        throw std::runtime_error("IV first-stage relation is rank deficient");
    }

    Matrix excluded_residual(n, excluded);
    for (Eigen::Index j = 0; j < excluded; ++j)
        excluded_residual.col(j) = weighted.col(columns[static_cast<std::size_t>(exogenous + j)]);
    Matrix endogenous_residual = response;
    Matrix span_basis;
    if (span_rank < k)
        span_basis = qr.householderQ() * Matrix::Identity(n, span_rank);
    if (exogenous) {
        Eigen::ColPivHouseholderQR<Matrix> exogenous_qr(normalized.leftCols(exogenous));
        exogenous_qr.setThreshold(static_cast<Real>(rank_tolerance(n, exogenous)));
        if (exogenous_qr.rank() != exogenous) {
            throw std::runtime_error("Exogenous IV design is rank deficient");
        }
        Matrix coordinates(n, excluded + q);
        coordinates << excluded_residual, endogenous_residual;
        coordinates = (exogenous_qr.householderQ().adjoint() * coordinates).eval();
        excluded_residual = coordinates.bottomRows(n - exogenous).leftCols(excluded);
        endogenous_residual = coordinates.bottomRows(n - exogenous).rightCols(q);
        if (span_rank < k) {
            span_basis = (exogenous_qr.householderQ().adjoint() * span_basis).eval();
            span_basis = span_basis.bottomRows(n - exogenous).eval();
        }
    }
    if (span_rank < k) {
        for (Eigen::Index j = 0; j < q; ++j) {
            const Real norm = endogenous_residual.col(j).stableNorm();
            if (!(norm > 0)) throw std::runtime_error("IV first-stage relation is rank deficient");
            endogenous_residual.col(j) /= norm;
        }
        const Eigen::MatrixXd relation =
            (span_basis.transpose() * endogenous_residual).template cast<double>();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(relation);
        const double threshold = rank_tolerance(span_rank, q);
        Eigen::Index useful_rank = 0;
        for (Eigen::Index j = 0; j < svd.singularValues().size(); ++j)
            useful_rank += svd.singularValues()[j] > threshold ? 1 : 0;
        if (span_rank < exogenous + q || useful_rank < q)
            throw std::runtime_error("IV first-stage relation is rank deficient");
    } else {
    for (Eigen::Index j = 0; j < excluded; ++j) {
        const Real norm = excluded_residual.col(j).stableNorm();
        if (!(norm > 0)) {
            throw std::runtime_error("FWL-residualized excluded instruments are rank deficient");
        }
        excluded_residual.col(j) /= norm;
    }
    Eigen::ColPivHouseholderQR<Matrix> residual_qr(excluded_residual);
    residual_qr.setThreshold(static_cast<Real>(rank_tolerance(n, excluded)));
    if (residual_qr.rank() != excluded) {
        throw std::runtime_error("FWL-residualized excluded instruments are rank deficient");
    }
    for (Eigen::Index j = 0; j < q; ++j) {
        const Real norm = endogenous_residual.col(j).stableNorm();
        if (!(norm > 0)) {
            throw std::runtime_error("FWL-residualized endogenous regressors contain a zero column");
        }
        endogenous_residual.col(j) /= norm;
    }
    const Eigen::MatrixXd relation =
        (excluded_residual.transpose() * endogenous_residual).template cast<double>();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(relation);
    const double tolerance = rank_tolerance(excluded, q);
    Eigen::Index useful_rank = 0;
    for (Eigen::Index j = 0; j < svd.singularValues().size(); ++j) {
        useful_rank += svd.singularValues()[j] > tolerance ? 1 : 0;
    }
    if (useful_rank < q) {
        throw std::runtime_error("IV first-stage relation is rank deficient");
    }
    }

    Matrix coordinates = qr.householderQ().adjoint() * response;
    coordinates.bottomRows(n - span_rank).setZero();
    Matrix fitted = qr.householderQ() * coordinates;
    if (weights) {
        Matrix gamma;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (sqrt_w[i] > 0) {
                fitted.row(i) /= sqrt_w[i];
            } else {
                if (gamma.size() == 0) {
                    gamma = qr.solve(response);
                    for (Eigen::Index j = 0; j < k; ++j) gamma.row(j) /= scales[j];
                }
                for (Eigen::Index j = 0; j < q; ++j) {
                    Real value = 0;
                    for (Eigen::Index c = 0; c < k; ++c) {
                        value += static_cast<Real>(instruments(i, columns[static_cast<std::size_t>(c)])) * gamma(c, j);
                    }
                    fitted(i, j) = value;
                }
            }
        }
    }
    Eigen::MatrixXd result(n, q);
    if (projection_low) projection_low->resize(n, q);
    for (Eigen::Index i = 0; i < fitted.size(); ++i) {
        if (!fitted.data()[i].finite()) {
            throw std::runtime_error("IV projection produced NaN or Inf");
        }
        result.data()[i] = projection_low ? fitted.data()[i].hi : fitted.data()[i].value();
        if (projection_low) projection_low->data()[i] = fitted.data()[i].lo;
    }
    if (!matrix_finite_bits(result)) {
        throw std::runtime_error("IV projection produced NaN or Inf");
    }
    return result;
}

std::optional<Eigen::MatrixXd> remove_exact_instrument_origins(
    const Eigen::MatrixXd& instruments,
    const Eigen::MatrixXd& gram,
    int num_exogenous) {
    const Eigen::Index n = instruments.rows(), k = instruments.cols();
    if (!n || num_exogenous <= 0 || num_exogenous > k) return std::nullopt;
    int constant = -1;
    for (int j = 0; j < num_exogenous; ++j) {
        const double value = instruments(0, j);
        if (value != 0.0 && finite_bits(value) && instruments(n - 1, j) == value &&
            (instruments.col(j).array() == value).all()) {
            constant = j;
            break;
        }
    }
    if (constant < 0 || !(gram(constant, constant) > 0.0)) return std::nullopt;
    std::vector<int> shifted_columns;
    for (int j = 0; j < k; ++j) {
        if (j == constant || instruments(0, j) == 0.0 || !(gram(j, j) > 0.0)) continue;
        const double correlation = std::abs(gram(j, constant)) /
            std::sqrt(gram(j, j)) / std::sqrt(gram(constant, constant));
        // Same arithmetic dispatch band as the existing normalized-Gram route.
        if (finite_bits(correlation) && 1.0 - correlation * correlation <= 0x1p-13)
            shifted_columns.push_back(j);
    }
    if (shifted_columns.empty()) return std::nullopt;
    Eigen::MatrixXd shifted = instruments;
    for (int j : shifted_columns) {
        const OlsWide origin(instruments(0, j));
        for (Eigen::Index i = 0; i < n; ++i) {
            const OlsWide difference = OlsWide(instruments(i, j)) - origin;
            // Preserve the represented input span, including small directions.
            // Inexact shifts keep the original arithmetic path.
            if (!difference.finite() || difference.lo != 0.0) return std::nullopt;
            shifted(i, j) = difference.hi;
        }
    }
    return shifted;
}

}  // namespace

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights,
                                   ParallelWorkObserver* parallel_observer,
                                   Eigen::MatrixXd* projection_low) {
    if (projection_low) projection_low->resize(0, 0);
    if (instrument_matrix.rows() != endogenous.rows()) {
        throw std::runtime_error("Instrument and endogenous matrices must share the same number of rows");
    }
    if (endogenous.cols() == 0) {
        return Eigen::MatrixXd(endogenous.rows(), 0);
    }
    const int n = static_cast<int>(instrument_matrix.rows());
    const int k = static_cast<int>(instrument_matrix.cols());

    const Eigen::MatrixXd* Z = &instrument_matrix;
    const Eigen::MatrixXd* Q = &endogenous;
    Eigen::MatrixXd weighted_Z;
    Eigen::MatrixXd weighted_Q;
    Eigen::VectorXd sqrt_weights;
    if (weights) {
        if (weights->size() != n) {
            throw std::runtime_error("Weights must align with instrument rows");
        }
        weighted_Z = instrument_matrix;
        weighted_Q = endogenous;
        sqrt_weights = weights->array().sqrt();
        for (int j = 0; j < k; ++j) {
            weighted_Z.col(j).array() *= sqrt_weights.array();
        }
        for (int j = 0; j < weighted_Q.cols(); ++j) {
            weighted_Q.col(j).array() *= sqrt_weights.array();
        }
        Z = &weighted_Z;
        Q = &weighted_Q;
    }

    Eigen::MatrixXd ztz = Z->transpose() * *Z;
    Eigen::MatrixXd ztq = Z->transpose() * *Q;
    if (auto shifted = remove_exact_instrument_origins(instrument_matrix, ztz, num_exogenous)) {
        return project_endogenous(*shifted, endogenous, num_exogenous, weights,
                                  parallel_observer, projection_low);
    }
    const int projection_route = preflight_identification(*Z, *Q, ztz, ztq, num_exogenous);
    if (projection_route == 2) {
        return project_orthogonal(instrument_matrix, endogenous, num_exogenous, weights, projection_low);
    }
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(ztz);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize instrument cross-product in 2SLS");
    }
    Eigen::MatrixXd gamma = solver.solve(ztq);
    Eigen::MatrixXd fitted;
    if (projection_route == 1) {
        fitted = refine_projection(instrument_matrix, endogenous, weights, solver, gamma, parallel_observer);
    } else {
        fitted = instrument_matrix * gamma;
    }
    validate_projection(ztz, ztq, gamma, fitted, n);
    return fitted;
}

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instruments,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights,
                                   ParallelWorkObserver* observer) {
    return project_endogenous(instruments, endogenous, num_exogenous, weights, observer, nullptr);
}

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instruments,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights) {
    return project_endogenous(instruments, endogenous, num_exogenous, weights, nullptr);
}

}  // namespace detail
}  // namespace hdfe
