#include "fe_absorption.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace hdfe;
    using namespace hdfe::detail;
    int cases=0;
    for(bool overlap:{false,true}) for(bool mean:{false,true}) for(bool weighted:{false,true})
    for(double magnitude:{1.0,1e12,1e16,1e300}) for(double multiplier:{0.0,1.0,2.0}) {
        const std::vector<std::vector<int>> teams=overlap ?
            std::vector<std::vector<int>>{{0},{1},{2},{0,1},{0,2},{1,2}} :
            std::vector<std::vector<int>>{{0,1},{1,2},{2,3},{0,3}};
        const int n=teams.size(),individuals=overlap ? 3 : 4;
        GroupIndividualStructure gi;
        gi.num_groups=n;gi.num_individuals=individuals;gi.group_ptr={0};
        std::vector<std::vector<int>> columns(individuals);
        std::vector<Eigen::VectorXi> fes;
        if(overlap) fes.push_back(Eigen::VectorXi(n));
        Eigen::VectorXd y(n),weights(n);
        for(int row=0;row<n;++row) {
            for(int id:teams[row]) {gi.group_individual.push_back(id);columns[id].push_back(row);}
            gi.group_ptr.push_back(gi.group_individual.size());
            gi.group_scale.push_back(mean ? 1.0/teams[row].size() : 1.0);
            if(overlap) fes[0][row]=teams[row].size();
            weights[row]=weighted ? std::ldexp(1.0,row) : 1.0;
        }
        gi.individual_ptr={0};
        for(const auto& column:columns) {
            gi.individual_group.insert(gi.individual_group.end(),column.begin(),column.end());
            gi.individual_ptr.push_back(gi.individual_group.size());
        }
        if(overlap) {
            const double value=mean ? .5 : 1;
            y<<0,-value,value,1,-1,0;
        } else y<<1,-1,1,-1;
        y.array()/=weights.array();
        Eigen::MatrixXd X(n,0);
        AbsorptionResult result;
        result.y_tilde=multiplier*y; result.X_tilde=X; result.mlsmr_used=true;
        if(overlap) {
            Eigen::VectorXd standard(2),individual(3);
            standard<<-magnitude,-(mean ? magnitude : 2*magnitude);individual.setConstant(magnitude);
            result.fe_alpha_y={standard,individual};
            result.fe_alpha_X={Eigen::MatrixXd(2,0),Eigen::MatrixXd(3,0)};
        } else {
            Eigen::VectorXd individual(4);individual<<magnitude,-magnitude,magnitude,-magnitude;
            result.fe_alpha_y={individual};result.fe_alpha_X={Eigen::MatrixXd(4,0)};
        }
        HdfeOptions options;options.tol=1e-8;options.num_threads=2;
        const bool accepted=certify_group_individual_candidate(y,X,fes,gi,&weights,options,result);
        if(accepted!=(multiplier==1)) {
            std::cerr<<"Unexpected certificate: "<<overlap<<' '<<mean<<' '<<weighted<<' '
                     <<magnitude<<' '<<multiplier<<' '<<accepted<<'\n';return 1;
        }
        ++cases;
    }
    std::cout<<"AFFINE_SHIFT_AND_WRONG_RESIDUAL_CONTRACT_PASS "<<cases<<'\n';
}
