#include "fe_absorption.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
    constexpr int n=16;
    int tested=0;
    for (bool individual_block : {false,true}) {
        hdfe::detail::GroupIndividualStructure gi;
        gi.num_groups=n;gi.num_individuals=individual_block?4:1;
        gi.group_scale.assign(n,1.0);gi.group_ptr.resize(n+1);gi.group_individual.resize(n);
        gi.individual_ptr.resize(gi.num_individuals+1);gi.individual_group.resize(n);
        Eigen::VectorXi ordinary(n);
        for (int row=0;row<n;++row) {
            gi.group_ptr[row]=row;gi.group_individual[row]=individual_block?row/4:0;
            gi.individual_group[row]=row;ordinary[row]=individual_block?0:row/4;
        }
        gi.group_ptr[n]=n;
        for (int j=0;j<=gi.num_individuals;++j) gi.individual_ptr[j]=j*(n/gi.num_individuals);
        for (double shift : {1e-9,3e-9}) for (bool fast : {false,true}) {
            Eigen::VectorXd residual(n),original(n);
            for (int row=0;row<n;++row) {
                residual[row]=(row%2?1.0:-1.0)+(row<4?shift:0.0);
                original[row]=(row%2?1.0:-1.0)+3.0+(row/4);
            }
            hdfe::detail::AbsorptionResult result;
            result.y_tilde=residual;result.X_tilde.resize(n,0);result.converged=true;
            hdfe::HdfeOptions options;options.num_threads=2;options.num_threads_explicit=true;
            options.absorption_method=hdfe::AbsorptionMethod::GaussSeidel;
            options.tolerance_mode=fast?hdfe::ToleranceMode::XhdfeFast:hdfe::ToleranceMode::ReghdfeComparable;
            const Eigen::MatrixXd X(n,0);
            hdfe::detail::certify_absorption_result(original,X,{ordinary},nullptr,options,{},result,&gi);
            // Four disjoint columns and their common column: max=shift/2,
            // RMS=shift/4 to first order. The old RMS passes at shift=3e-9.
            check(result.precision_certified==(fast || shift==1e-9),
                  "maximum column moment was hidden by its RMS");
            ++tested;
        }
    }
    std::cout<<"GROUP_COLUMN_MOMENT_CONTRACT_PASS "<<tested<<" cases\n";
    return 0;
    } catch (const std::exception& error) {
        std::cerr<<"GROUP_COLUMN_MOMENT_CONTRACT_FAIL "<<error.what()<<'\n';
        return 1;
    }
}
