#pragma once

#include <Eigen/QR>
#include "wide_float.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hdfe {
namespace detail {

// The projection and its coefficient influence vectors use the same QR basis.
// Exact origin shifts remove artificial conditioning from the implicit constant.
class AkmLincomProjection {
    using WideMatrix = Eigen::Matrix<OlsWide, Eigen::Dynamic, Eigen::Dynamic>;
    using DoubleQr = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>;
    using WideQr = Eigen::ColPivHouseholderQR<WideMatrix>;
    std::unique_ptr<DoubleQr> ordinary_;
    std::unique_ptr<WideQr> wide_;
    Eigen::VectorXd scales_;
    Eigen::VectorXd coefficients_;
    std::vector<int> position_;
    Eigen::Index rows_ = 0;
    int rank_ = 0;
    static double as_double(double value) { return value; }
    static double as_double(const OlsWide& value) { return value.value(); }

    template <class Qr>
    void initialize(const Qr& qr, const Eigen::VectorXd& response) {
        using Scalar = typename Qr::Scalar;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        rank_ = static_cast<int>(qr.rank());
        position_.assign(static_cast<std::size_t>(scales_.size()), -1);
        coefficients_.setConstant(scales_.size(), std::numeric_limits<double>::quiet_NaN());
        const Vector projected = qr.householderQ().adjoint()*response.template cast<Scalar>();
        const Vector beta = qr.matrixQR().topLeftCorner(rank_, rank_)
            .template triangularView<Eigen::Upper>().solve(projected.head(rank_));
        for (int j = 0; j < rank_; ++j) {
            const int original = qr.colsPermutation().indices()[j];
            position_[static_cast<std::size_t>(original)] = j;
            coefficients_[original] = as_double(beta[j]/Scalar(scales_[original]));
        }
    }

    bool finite_identified() const {
        for (Eigen::Index j = 0; j < coefficients_.size(); ++j)
            if (position_[static_cast<std::size_t>(j)] >= 0 && !ieee_finite(coefficients_[j]))
                return false;
        return true;
    }

    template <class Qr>
    Eigen::VectorXd influence(const Qr& qr, int column) const {
        using Scalar = typename Qr::Scalar;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        Vector target = Vector::Zero(rank_);
        target[position_[static_cast<std::size_t>(column)]] = Scalar(1.0)/Scalar(scales_[column]);
        const Vector solved = qr.matrixQR().topLeftCorner(rank_, rank_).transpose()
            .template triangularView<Eigen::Lower>().solve(target);
        Vector padded = Vector::Zero(rows_);
        padded.head(rank_) = solved;
        const Vector result = qr.householderQ()*padded;
        return result.template cast<double>();
    }

public:
    AkmLincomProjection(const Eigen::MatrixXd& design, const Eigen::VectorXd& response)
        : rows_(design.rows()) {
        if (rows_ == 0 || response.size() != rows_)
            throw std::runtime_error("lincom projection requires aligned, nonempty inputs");
        const Eigen::Index columns = design.cols();
        scales_ = Eigen::VectorXd::Ones(columns);
        Eigen::VectorXd origins = Eigen::VectorXd::Zero(columns);
        Eigen::MatrixXd normalized = design;
        for (Eigen::Index j = 1; j < columns; ++j) {
            const double low = design.col(j).minCoeff();
            const double high = design.col(j).maxCoeff();
            if ((low > 0.0 && high <= 2.0*low) || (high < 0.0 && low >= 2.0*high))
                origins[j] = design(0,j);  // Sterbenz: every subtraction is exact.
            normalized.col(j).array() -= origins[j];
            const double scale = normalized.col(j).cwiseAbs().maxCoeff();
            if (scale > 0.0) {
                scales_[j] = scale;
                normalized.col(j) /= scale;
            }
        }
        ordinary_ = std::make_unique<DoubleQr>(normalized);
        normalized.resize(0,0);
        const auto diagonal = ordinary_->matrixQR().diagonal().cwiseAbs().eval();
        bool need_wide = ordinary_->rank() < columns ||
            diagonal.minCoeff() <= 1e-4*diagonal.maxCoeff();
        if (!need_wide) {
            initialize(*ordinary_, response);
            need_wide = !finite_identified();
        }
        if (need_wide) {
            ordinary_.reset();
            WideMatrix normalized_wide(rows_, columns);
            for (Eigen::Index j = 0; j < columns; ++j)
                for (Eigen::Index i = 0; i < rows_; ++i)
                    normalized_wide(i,j) = (OlsWide(design(i,j))-OlsWide(origins[j]))/OlsWide(scales_[j]);
            wide_ = std::make_unique<WideQr>(normalized_wide);
            initialize(*wide_, response);
            if (!finite_identified())
                throw std::runtime_error("lincom numeric solve produced nonfinite coefficients; no estimates returned");
        }
    }

    bool identified(int column) const { return position_[static_cast<std::size_t>(column)] >= 0; }
    double coefficient(int column) const { return coefficients_[column]; }
    Eigen::VectorXd weights(int column) const {
        if (!identified(column)) return Eigen::VectorXd::Zero(rows_);
        Eigen::VectorXd result = wide_ ? influence(*wide_, column) : influence(*ordinary_, column);
        for (Eigen::Index i = 0; i < result.size(); ++i)
            if (!ieee_finite(result[i]))
                throw std::runtime_error("lincom influence is not representable; no estimates returned");
        return result;
    }
};

} // namespace detail
} // namespace hdfe
