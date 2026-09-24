// Credit2 residualisation, with independent residuals supplied by the driver.
// Usage: binary INPUT_DIRECTORY N MAX_ITER EXPECT_SUCCESS
#include "fe_absorption.hpp"
#include "hdfe/hdfe_regressor_v11.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const std::string directory = argv[1];
    const int n = std::stoi(argv[2]), budget = std::stoi(argv[3]);
    const bool expect_success = std::stoi(argv[4]) != 0;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> raw(n, 3), reference(n, 3);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ids(n, 2);
    auto read = [&](const char* name, char* values, std::streamsize bytes) {
        std::ifstream input(directory + "/" + name, std::ios::binary);
        input.read(values, bytes);
        if (!input || input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("invalid fixture file size");
    };
    read("matrix.bin", reinterpret_cast<char*>(raw.data()), sizeof(double)*n*3);
    read("reference.bin", reinterpret_cast<char*>(reference.data()), sizeof(double)*n*3);
    read("fes.bin", reinterpret_cast<char*>(ids.data()), sizeof(int)*n*2);
    Eigen::VectorXd y = raw.col(0);
    Eigen::MatrixXd X = raw.rightCols(2);
    std::vector<Eigen::VectorXi> fes = {ids.col(0), ids.col(1)};
    hdfe::HdfeOptions options;
    options.num_threads = 2;
    options.drop_singletons = false;
    options.absorption_method = hdfe::AbsorptionMethod::SymmetricGaussSeidel;
    options.ordinary_krylov_parity_floor = true;
    options.max_iter = budget;
    hdfe::v11::HdfeRegressorV11 model(options);
    try {
        const auto out = model.partial_out(y, X, fes);
        if (!expect_success || !out.converged || !out.precision_certified || out.iterations > budget)
            return 1;
        const Eigen::VectorXd beta = out.X_tilde.colPivHouseholderQr().solve(out.y_tilde);
        const Eigen::VectorXd expected = reference.rightCols(2).colPivHouseholderQr().solve(reference.col(0));
        const double error = ((beta-expected).array().abs() / expected.array().abs().max(1.0)).maxCoeff();
        if (error > 1e-9) return 1;
        std::cout << std::setprecision(17) << "PASS iterations=" << out.iterations
                  << " coefficient_error=" << error << " certificate=" << out.abs_residual_rel << '\n';
    } catch (const std::runtime_error& error) {
        if (expect_success || model.has_partial_result() || model.has_estimation_result()) {
            std::cerr << error.what() << '\n';
            return 1;
        }
        std::cout << "PASS exhausted budget leaves no result: " << error.what() << '\n';
    }
}
