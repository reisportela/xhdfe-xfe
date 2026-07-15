#include "iv.hpp"

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

bool gram_is_clearly_full_rank(const Eigen::MatrixXd& normalized) {
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

void require_full_rank(const Eigen::MatrixXd& matrix,
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
        return;
    }
    const Eigen::VectorXd scales = positive_column_scales(gram, label);
    const Eigen::MatrixXd normalized = normalized_gram(gram, scales, label);
    if (gram_is_clearly_full_rank(normalized)) {
        return;
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
        std::ostringstream message;
        message << label << " is rank deficient (rank " << rank << " < "
                << checked_columns << ")";
        throw std::runtime_error(message.str());
    }
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

void preflight_identification(const Eigen::MatrixXd& weighted_instruments,
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
    if (k > n) {
        throw std::runtime_error("IV instrument matrix has more columns than observations");
    }
    if (!matrix_finite_bits(ztz) || !matrix_finite_bits(ztq)) {
        throw std::runtime_error("IV design produced a non-finite cross-product");
    }

    // An absorbed intercept is retained by the legacy IV path as an exactly
    // zero exogenous column.  It must stay in the LDLT below for zero-diff
    // compatibility, but it carries no identifying information and must not
    // make an otherwise valid model fail preflight.
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

    require_full_rank(weighted_instruments, *identification_ztz,
                      "IV instrument matrix",
                      effective_num_exogenous == num_exogenous
                          ? nullptr
                          : &checked_columns);

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
    if (useful_rank < endogenous_cols) {
        std::ostringstream message;
        message << "IV first-stage relation is rank deficient (rank " << useful_rank
                << " < " << endogenous_cols << ")";
        throw std::runtime_error(message.str());
    }
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

}  // namespace

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights) {
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
    preflight_identification(*Z, *Q, ztz, ztq, num_exogenous);
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(ztz);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize instrument cross-product in 2SLS");
    }
    Eigen::MatrixXd gamma = solver.solve(ztq);
    Eigen::MatrixXd fitted = instrument_matrix * gamma;
    validate_projection(ztz, ztq, gamma, fitted, n);
    return fitted;
}

}  // namespace detail
}  // namespace hdfe
