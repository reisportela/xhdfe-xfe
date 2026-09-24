// Permanent 30-case underidentification guard, with a valid companion per case.
// Build with the candidate's iv.hpp and core archive (including its IV helper
// objects). This directly tests project_endogenous, not frontend collinearity
// removal. No timing or backend-use conclusion is implied.
#include "iv.hpp"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
bool finite_bits(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
}

int main(int argc, char** argv) {
    const char* label = argc > 1 ? argv[1] : "unspecified";
    std::cout << "label=" << label
              << " double_digits=" << std::numeric_limits<double>::digits
              << " long_double_digits=" << std::numeric_limits<long double>::digits << '\n';
    int accepted = 0, rejected = 0, unexpected_errors = 0, valid_passed = 0;
    for (int n : {3, 4, 7, 16, 128}) {
        for (int pattern : {0, 1, 2}) {
            for (int weighted : {0, 1}) {
                Eigen::MatrixXd z(n, 2), q(n, 1);
                Eigen::VectorXd weights(n);
                for (int i = 0; i < n; ++i) {
                    const double exogenous = pattern == 0 ? 1.0 :
                        (pattern == 1 ? i + 1.0 : (i % 2 ? -1.0 : 1.0) * (i + 1.0));
                    z(i, 0) = exogenous;
                    z(i, 1) = i == 0 ? 0.0 : 1.0;
                    q(i, 0) = exogenous;
                    weights[i] = 1.0 + i % 3;
                }
                // A valid first-stage direction prevents an always-refuse
                // implementation from satisfying the negative cases below.
                try {
                    const Eigen::MatrixXd valid = z.rightCols(1);
                    const auto fitted = hdfe::detail::project_endogenous(
                        z, valid, 1, weighted ? &weights : nullptr);
                    const double error = (fitted - valid).cwiseAbs().maxCoeff();
                    if (!finite_bits(error) || error > 1e-10) {
                        ++unexpected_errors;
                        std::cout << "VALID_FAIL n=" << n << " pattern=" << pattern
                                  << " weighted=" << weighted << " error=" << error << '\n';
                    } else {
                        ++valid_passed;
                    }
                } catch (const std::exception& error) {
                    ++unexpected_errors;
                    std::cout << "VALID_ERROR " << error.what() << '\n';
                }
                // Q equals the exogenous column exactly in represented data.
                // The excluded instrument is independent; endogenous FWL rank
                // is zero regardless of the strictly positive weights.
                std::cout << "n=" << n << " pattern=" << pattern
                          << " weighted=" << weighted << ' ';
                try {
                    const auto fitted = hdfe::detail::project_endogenous(
                        z, q, 1, weighted ? &weights : nullptr);
                    ++accepted;
                    std::cout << "UNEXPECTED_ACCEPT fitted_q_max_error="
                              << std::setprecision(17)
                              << (fitted - q).cwiseAbs().maxCoeff() << '\n';
                } catch (const std::runtime_error& error) {
                    const std::string message = error.what();
                    const bool identified_reason =
                        message.find("IV first-stage relation is rank deficient") != std::string::npos ||
                        (message.find("FWL-residualized endogenous regressors") != std::string::npos &&
                         message.find("zero") != std::string::npos);
                    if (identified_reason) {
                        ++rejected;
                        std::cout << "REJECT " << message << '\n';
                    } else {
                        ++unexpected_errors;
                        std::cout << "UNEXPECTED_ERROR " << message << '\n';
                    }
                } catch (const std::exception& error) {
                    ++unexpected_errors;
                    std::cout << "UNEXPECTED_ERROR " << error.what() << '\n';
                }
            }
        }
    }
    std::cout << "accepted=" << accepted << " rejected=" << rejected
              << "valid_passed=" << valid_passed << " unexpected_errors=" << unexpected_errors << '\n';
    return accepted == 0 && rejected == 30 && valid_passed == 30 && unexpected_errors == 0 ? 0 : 1;
}
