#ifndef HDFE_IV_HPP
#define HDFE_IV_HPP

#include <Eigen/Dense>

namespace hdfe {
namespace detail {

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   const Eigen::VectorXd* weights);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_IV_HPP
