#ifndef HDFE_N1_GRAPH_PRECONDITIONER_HPP
#define HDFE_N1_GRAPH_PRECONDITIONER_HPP
#include "fe_absorption_cuda.hpp"
#include <functional>
namespace hdfe { namespace detail {
using N1GraphPreconditioner =
    std::function<void(const Eigen::MatrixXd&, Eigen::MatrixXd&)>;
N1GraphPreconditioner make_n1_graph_preconditioner(
    const std::vector<GpuFeInput>& inputs, int n, const HdfeOptions& options);
} }
#endif
