#ifndef HDFE_IV_HPP
#define HDFE_IV_HPP

#include <Eigen/Dense>

namespace hdfe {
namespace detail {

class ParallelWorkObserver;
Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights,
                                   ParallelWorkObserver* parallel_observer,
                                   Eigen::MatrixXd* projection_low);
Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights,
                                   ParallelWorkObserver* parallel_observer);

Eigen::MatrixXd project_endogenous(const Eigen::MatrixXd& instrument_matrix,
                                   const Eigen::MatrixXd& endogenous,
                                   int num_exogenous,
                                   const Eigen::VectorXd* weights);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_IV_HPP
