#include "ols.hpp"
#include "wide_covariance.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using namespace hdfe;
    using namespace hdfe::detail;
    const int n=512;
    int case_id=0;
    for (int p : {2,3,5}) for (int constant : {-1,0,p-1}) {
        if (p==2 && constant>=0) continue;
        for (bool weighted : {false,true}) for (int exponent : {-80,0,80})
        for (bool conventional : {false,true}) {
            Eigen::MatrixXd X(n,p), low=Eigen::MatrixXd::Zero(n,p);
            Eigen::VectorXd y(n), w(n), means(p);
            std::vector<Eigen::VectorXi> clusters(2,Eigen::VectorXi(n));
            const double scale=std::ldexp(1.,exponent);
            for (int j=0;j<p;++j) means[j]=(j%2 ? -.5 : .25);
            for (int i=0;i<n;++i) {
                auto walsh=[&](int bit) { return double(2*((i>>bit)&1)-1); };
                int variable=0;
                y[i]=walsh(5)*scale;
                w[i]=weighted ? 1.+i/64 : 1.;
                for (int j=0;j<p;++j) {
                    if (j==constant) X(i,j)=1.;
                    else {
                        if(variable==0) X(i,j)=2*walsh(0)+(constant>=0 ? 4. : 0.);
                        else if(variable==1) X(i,j)=2*walsh(0)+std::ldexp(walsh(1),-27)-(constant>=0 ? 2. : 0.);
                        else X(i,j)=walsh(variable)+(constant>=0 ? double(variable) : 0.);
                        ++variable;
                    }
                    y[i]+=std::ldexp(double(j+1),-3)*X(i,j)*scale;
                }
                clusters[0][i]=(i*37+i/7)%19-9;
                clusters[1][i]=(i*13+i/11)%23-11;
            }
            Eigen::VectorXd intercept;
            OlsWideCovariance wide;
            auto result=run_ols_multiway_with_score_low(y,X,weighted?&w:nullptr,
                &clusters,StandardErrorType::Cluster,1.,1.,
                conventional?ClusterDofMethod::Conventional:ClusterDofMethod::Min,
                true,n,nullptr,true,nullptr,&low,&means,&intercept,&wide);
            if(wide.values.rows()!=p+1) throw std::runtime_error("wide path was not exercised");
            std::cout<<std::setprecision(17)<<"{\"id\":"<<case_id++
                <<",\"p\":"<<p<<",\"constant\":"<<constant
                <<",\"weighted\":"<<(weighted?"true":"false")
                <<",\"exponent\":"<<exponent<<",\"conventional\":"<<(conventional?"true":"false")
                <<",\"rss\":"<<result.rss<<",\"coef\":[";
            for(int j=0;j<p;++j) std::cout<<(j?",":"")<<result.coefficients[j];
            std::cout<<"],\"covariance\":[";
            for(int i=0;i<=p;++i) {
                std::cout<<(i?",[":"[");
                for(int j=0;j<=p;++j) std::cout<<(j?",":"")<<wide.values(i,j).value();
                std::cout<<"]";
            }
            std::cout<<"],\"output_covariance\":[";
            for(int i=0;i<=p;++i) {
                std::cout<<(i?",[":"[");
                for(int j=0;j<=p;++j) {
                    double value=(i==p)?intercept[j]:(j==p?intercept[i]:result.covariance(i,j));
                    std::cout<<(j?",":"")<<value;
                }
                std::cout<<"]";
            }
            std::cout<<"]}\n";
        }
    }
}
