#pragma once
#include "fe_absorption.hpp"
#include <stdexcept>

namespace hdfe::detail {
enum class MixedNetworkMode { Off, Diagonal, Direct, Forest };
struct MixedNetworkDecline : std::runtime_error {
    int iterations;
    explicit MixedNetworkDecline(const char *reason, int spent = 0)
        : std::runtime_error(reason), iterations(spent) {}
};
struct MixedNetworkScores {
    struct Geometry {
        std::vector<int> rows, worker_matches, firm_matches, firm_match_ids, match_rows;
    };

    const std::vector<Eigen::VectorXi> *clusters = nullptr;
    std::unique_ptr<Geometry> geometry;
    std::vector<int> counts;
    int threads = 1;
    ParallelWorkObserver *observer = nullptr;
    Eigen::MatrixXd raw;
    const double *design_data = nullptr;
    const double *residual_data = nullptr;
    int row_count = 0;
    int columns = 0;
    bool conventional_adjustment = false;

    bool form(const Eigen::Ref<const Eigen::MatrixXd> &design, const Eigen::VectorXd &residuals,
              ClusterDofMethod method, bool adjust);
    bool matches(const Eigen::Ref<const Eigen::MatrixXd> &design, const Eigen::VectorXd &residuals,
                 const std::vector<Eigen::VectorXi> &source, ClusterDofMethod method,
                 bool adjust) const;
};
MixedNetworkMode mixed_network_mode();
bool mixed_network_shape(const std::vector<Eigen::VectorXi> &fes,
                         const std::vector<HeterogeneousSlopeTerm> &slopes);
AbsorptionResult absorb_mixed_network(const Eigen::Ref<const Eigen::VectorXd> &y,
                                      const Eigen::Ref<const Eigen::MatrixXd> &X,
                                      const std::vector<Eigen::VectorXi> &fes,
                                      const std::vector<HeterogeneousSlopeTerm> &slopes,
                                      const HdfeOptions &options, MixedNetworkMode mode, bool cuda,
                                      const std::vector<Eigen::VectorXi> *clusters,
                                      std::unique_ptr<MixedNetworkScores> *capture);
} // namespace hdfe::detail
