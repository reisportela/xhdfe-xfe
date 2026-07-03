#include "iv.hpp"

#include <stdexcept>

namespace hdfe {
namespace detail {

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   const Eigen::VectorXd* weights) {
    if (instrument_matrix.rows() != endogenous.rows()) {
        throw std::runtime_error("Instrument and endogenous matrices must share the same number of rows");
    }
    if (endogenous.cols() == 0) {
        return Eigen::MatrixXd(endogenous.rows(), 0);
    }
    const int n = static_cast<int>(instrument_matrix.rows());
    const int k = static_cast<int>(instrument_matrix.cols());

    Eigen::MatrixXd Z = instrument_matrix;
    Eigen::MatrixXd Q = endogenous;
    Eigen::VectorXd sqrt_weights;
    if (weights) {
        if (weights->size() != n) {
            throw std::runtime_error("Weights must align with instrument rows");
        }
        sqrt_weights = weights->array().sqrt();
        for (int j = 0; j < k; ++j) {
            Z.col(j).array() *= sqrt_weights.array();
        }
        for (int j = 0; j < Q.cols(); ++j) {
            Q.col(j).array() *= sqrt_weights.array();
        }
    }

    Eigen::MatrixXd ztz = Z.transpose() * Z;
    Eigen::MatrixXd ztq = Z.transpose() * Q;
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(ztz);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize instrument cross-product in 2SLS");
    }
    Eigen::MatrixXd gamma = solver.solve(ztq);
    Eigen::MatrixXd fitted = instrument_matrix * gamma;
    return fitted;
}

}  // namespace detail
}  // namespace hdfe

