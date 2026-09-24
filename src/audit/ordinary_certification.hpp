#pragma once

#include "fe_absorption.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hdfe { namespace detail {

// Private ELF test visibility; no binding, option or environment surface.
inline thread_local unsigned long long ordinary_audit_work_count=0;

// Read per fit; the existing absorption-cache signature includes this bit.
inline bool ordinary_audit_enabled() noexcept {
    const char* value=std::getenv("XHDFE_CERTIFY");
    return value && std::strcmp(value,"1")==0;
}

enum class OrdinaryAuditFinding { Passed, ErrorDemonstrated, BoundInconclusive, Unsupported };
struct OrdinaryAuditFamily {
    OrdinaryAuditFinding status=OrdinaryAuditFinding::Unsupported;
    std::string reason="not_requested";
};
class OrdinaryReferenceDisagreement final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class OrdinaryAuditFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline const char* ordinary_audit_status(OrdinaryAuditFinding status) noexcept {
    switch (status) {
    case OrdinaryAuditFinding::Passed: return "PASS";
    case OrdinaryAuditFinding::ErrorDemonstrated: return "ERROR_DEMONSTRATED";
    case OrdinaryAuditFinding::BoundInconclusive: return "BOUND_INCONCLUSIVE";
    case OrdinaryAuditFinding::Unsupported: return "UNSUPPORTED";
    }
    return "UNSUPPORTED";
}
inline std::string ordinary_audit_json_string(const std::string& value) {
    std::string out="\"";
    for (unsigned char c:value) {
        if (c=='"' || c=='\\') {out+='\\';out+=char(c);}
        else if (c=='\n') out+="\\n";
        else if (c=='\r') out+="\\r";
        else if (c=='\t') out+="\\t";
        else if (c<32) {char escaped[7];std::snprintf(escaped,sizeof(escaped),"\\u%04x",c);out+=escaped;}
        else out+=char(c);
    }
    return out+'"';
}
struct OrdinaryAuditReceipt {
    const char* entrypoint="fit";
    const char* mode="xhdfe-fast";
    const char* backend="cpu";
    int iterations=0,n1_refinements=0;
    bool cache_hit=false,identity_match=false;
    std::array<OrdinaryAuditFamily,4> families;
    OrdinaryAuditReceipt(const char* entry,const HdfeOptions& options,
        const AbsorptionResult& result,bool cached=false,int refinements=0)
        :entrypoint(entry),mode(options.tolerance_mode==ToleranceMode::XhdfeFast ? "xhdfe-fast" :
          options.tolerance_mode==ToleranceMode::ReghdfeComparable ? "reghdfe-comparable" : "strict-residual"),
         backend(result.gpu_used ? "cuda" : "cpu"),iterations(result.iterations),
         n1_refinements(refinements),cache_hit(cached) {}
    void unsupported(const char* reason) {
        for (auto& family:families) family={OrdinaryAuditFinding::Unsupported,reason};
    }
    void emit() const {
        // Exactly one line, including inconclusive/unsupported audits. No public result fields.
        std::ostringstream line;
        line<<"XHDFE_N05_AUDIT {\"schema\":\"xhdfe-n05-audit-v1\",\"entrypoint\":"
            <<ordinary_audit_json_string(entrypoint)<<",\"mode\":"<<ordinary_audit_json_string(mode)
            <<",\"backend\":"<<ordinary_audit_json_string(backend)
            <<",\"iterations\":"<<iterations<<",\"cache_hit\":"<<(cache_hit ? "true" : "false")
            <<",\"n1_refinements\":"<<n1_refinements
            <<",\"identity_match\":"<<(identity_match ? "true" : "false")
            <<",\"mutated\":false,\"families\":{";
        constexpr const char* names[]={"projection","homoskedastic","sandwich","full_v"};
        for (std::size_t j=0;j<families.size();++j) {
            if(j) line<<',';
            line<<ordinary_audit_json_string(names[j])<<":{\"status\":"
                <<ordinary_audit_json_string(ordinary_audit_status(families[j].status))
                <<",\"reason\":"<<ordinary_audit_json_string(families[j].reason)<<'}';
        }
        line<<"}}\n";
        const auto text=line.str();std::fwrite(text.data(),1,text.size(),stderr);
        for(const auto& family:families)
            if(family.status==OrdinaryAuditFinding::ErrorDemonstrated)
                throw OrdinaryAuditFailure("XHDFE_CERTIFY N2 ERROR_DEMONSTRATED: "+family.reason+"; no estimates returned");
    }
};

// No ownership transfer or refinement: the accepted N1 candidate is immutable.
OrdinaryAuditFamily run_ordinary_n2_audit(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const HdfeOptions& options,const AbsorptionResult& result,
    const std::vector<Eigen::VectorXi>* clusters=nullptr,bool verify_regression=false);

} }

extern "C" unsigned long long xhdfe_private_n05_work_count(int reset) noexcept;
