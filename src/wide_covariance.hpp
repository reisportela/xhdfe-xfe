#ifndef HDFE_WIDE_COVARIANCE_HPP
#define HDFE_WIDE_COVARIANCE_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "wide_float.hpp"
#include <stdexcept>

namespace hdfe::detail {

// Optional output of the existing wide IV solve. Keep the low words until
// recovered-intercept and multiway PSD postprocessing have finished.
struct OlsWideCovariance {
    using Matrix = Eigen::Matrix<OlsWide, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix values;
};

inline bool fix_wide_covariance_psd(
    OlsWideCovariance::Matrix& covariance,
    const Eigen::Ref<const Eigen::VectorXd>& scale) {
    using Matrix = OlsWideCovariance::Matrix;
    const Eigen::Index n = covariance.rows();
    if (n == 0) return false;
    Matrix standardized(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            standardized(i,j) = OlsWide(scale[i]) *
                ((covariance(i,j) + covariance(j,i)) * OlsWide(0.5)) *
                OlsWide(scale[j]);
        }
    }
    Eigen::SelfAdjointEigenSolver<Matrix> solver(standardized);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Multiway covariance PSD projection did not converge; no estimates returned");
    auto eigenvalues = solver.eigenvalues().eval();
    bool changed = false;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (eigenvalues[i] < OlsWide(0.0)) {
            eigenvalues[i] = OlsWide(0.0);
            changed = true;
        }
    }
    if (changed) {
        const Matrix projected = solver.eigenvectors() * eigenvalues.asDiagonal() *
                                 solver.eigenvectors().transpose();
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                covariance(i,j) = (projected(i,j) / OlsWide(scale[i])) /
                                  OlsWide(scale[j]);
    }
    return changed;
}

} // namespace hdfe::detail
#endif
