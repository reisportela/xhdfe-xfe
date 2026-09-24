#ifndef XHDFE_OLS_PRECISION_HPP
#define XHDFE_OLS_PRECISION_HPP

#include "ols.hpp"
#include <cstdint>

namespace hdfe { namespace detail {

OlsResult precise_classical_reference(const Eigen::VectorXd& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Eigen::VectorXd* weights,
    bool frequency,ParallelWorkObserver* observer);

// Preserve the numerical rank policy while avoiding cancellation in centering
// and sequential residual-variance calculations. Output follows columns order.
std::vector<std::uint8_t> precise_ols_rank_mask(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<int>& columns,const Eigen::VectorXd* weights,
    bool center,double tolerance,const std::vector<int>& priority,
    int threads,ParallelWorkObserver* observer,
    Eigen::MatrixXd* directions=nullptr,Eigen::MatrixXd* full_basis=nullptr);

bool exact_two_fe_span(const Eigen::Ref<const Eigen::VectorXd>& values,
    const Eigen::VectorXi& first,const Eigen::VectorXi& second);

bool exact_level_projection(const Eigen::VectorXd& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Eigen::VectorXi& fe,
    const Eigen::VectorXd* weights,const Eigen::VectorXd& within_y,
    const Eigen::Ref<const Eigen::MatrixXd>& within_X);

}}
#endif
