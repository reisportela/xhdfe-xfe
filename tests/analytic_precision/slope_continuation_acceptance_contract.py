"""Exercise the real slope continuation and refusal guard with saved solve states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

FIXTURE = r'''
namespace hdfe { namespace detail {
struct Solve { int iterations; bool converged; bool certified; double block; };
std::vector<Solve> script;
std::vector<HdfeOptions> calls;
bool gpu = false;

AbsorptionResult absorb_fixed_effects_v6_mixed(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>&, const Eigen::VectorXd*,
    const HdfeOptions& options, AbsorptionMethod method,
    const std::vector<HeterogeneousSlopeTerm>&) {
    if (calls.size() >= script.size()) throw std::logic_error("unexpected solve");
    if (method != AbsorptionMethod::GaussSeidel)
        throw std::logic_error("changed method");
    const auto& spec = script[calls.size()];
    calls.push_back(options);
    AbsorptionResult result;
    result.iterations = spec.iterations;
    result.converged = spec.converged;
    result.slope_block_residual_rel = spec.block;
    result.abs_residual_rel = 4.5348549629472517e-13;
    result.y_tilde = Eigen::VectorXd::Constant(y.size(), calls.size() + 0.125);
    result.X_tilde = Eigen::MatrixXd::Constant(X.rows(), X.cols(), calls.size() + 0.25);
    result.fe_alpha_y = {Eigen::VectorXd::Constant(2, calls.size() + 0.5)};
    result.sweep_order_used = {1, 0};
    result.gpu_attempted = result.gpu_used = gpu;
    result.gpu_absorption_converged = gpu && spec.converged;
    result.gpu_status_code = gpu ? 1 : 0;
    result.gpu_absorption_iterations = gpu ? spec.iterations : 0;
    return result;
}

// The certificate outputs are scripted evidence; its arithmetic is tested elsewhere.
void certify_absorption_result(
    const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::MatrixXd>&,
    const std::vector<Eigen::VectorXi>&, const Eigen::VectorXd*,
    const HdfeOptions&, const std::vector<HeterogeneousSlopeTerm>&,
    AbsorptionResult& result, const GroupIndividualStructure*) {
    result.precision_certified = script.at(calls.size()-1).certified;
}
}}
'''

SEED_ADAPTER = r'''
namespace hdfe { namespace detail {
static AbsorptionResult absorb_fixed_effects_v6_mixed_impl(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const HdfeOptions& options,AbsorptionMethod method,
    const std::vector<HeterogeneousSlopeTerm>& slopes,const AbsorptionResult* seed) {
    if (seed && (weights || options.retain_fixed_effects || !seed->converged ||
                 seed->gpu_used || seed->gpu_attempted))
        throw std::logic_error("seed outside CPU continuation domain");
    if (!y.array().isConstant(1.0,0.0) || !X.array().isConstant(1.0,0.0))
        throw std::logic_error("continuation replaced original inputs");
    return absorb_fixed_effects_v6_mixed(y,X,fes,weights,options,method,slopes);
}
}}
'''

CONTROLS = r'''
namespace {
int failures = 0, cases = 0;
void check(bool ok, const std::string& label) {
    if (!ok) { std::cerr << "FAIL " << label << '\n'; ++failures; }
}
using namespace hdfe;
using namespace hdfe::detail;

void exercise(const std::string& name, HdfeOptions options,
              std::vector<Solve> states, std::vector<double> tolerances,
              bool accepted, bool device = false) {
    ++cases;
    script = states; calls.clear(); gpu = device;
    const Eigen::VectorXd y = Eigen::VectorXd::Ones(3);
    const Eigen::MatrixXd X = Eigen::MatrixXd::Ones(3, 2);
    std::vector<Eigen::VectorXi> fes{Eigen::VectorXi::Zero(3)};
    std::vector<HeterogeneousSlopeTerm> slopes(1);
    auto result = absorb_fixed_effects_v6(y, X, fes, nullptr, options,
                                        AbsorptionMethod::GaussSeidel, slopes);
    bool admitted = true;
    try { hdfe::v11::require_accepted_absorption(options, result, "fixture"); }
    catch (const std::runtime_error&) { admitted = false; }
    check(admitted == accepted, name+" admission");
    check(calls.size() == states.size(), name+" solve count");
    int total = 0;
    for (std::size_t i=0; i<calls.size(); ++i) {
        check(calls[i].max_iter == options.max_iter-total, name+" remaining budget");
        total += states[i].iterations;
        check(std::abs(calls[i].tol-tolerances.at(i)) <= 1e-14*tolerances.at(i),
              name+" unchanged stage target");
        check(calls[i].tolerance_mode == options.tolerance_mode, name+" mode");
        check(calls[i].from_auto == (i==0 && options.from_auto), name+" routing");
        check(calls[i].retain_fixed_effects == options.retain_fixed_effects,
              name+" retain FE");
    }
    check(result.iterations == total, name+" accumulated iterations");
    check(result.slope_accuracy_retry_stages == int(states.size()-1), name+" stages");
    check(result.y_tilde.array().isConstant(states.size()+0.125, 0), name+" y unchanged");
    check(result.X_tilde.array().isConstant(states.size()+0.25, 0), name+" X unchanged");
    check(result.fe_alpha_y[0].array().isConstant(states.size()+0.5, 0), name+" alphas unchanged");
    check(result.sweep_order_used == std::vector<int>({1,0}), name+" sweep order");
    check(result.abs_residual_rel == 4.5348549629472517e-13, name+" residual diagnostic");
    if (accepted) {
        check(result.converged && result.precision_certified, name+" real final flags");
        check(result.gpu_used == device, name+" backend retained");
        if (device) {
            check(result.gpu_status_code == 1 && result.gpu_absorption_converged,
                  name+" device status retained");
            check(result.gpu_absorption_iterations == total, name+" device iterations");
        }
    } else {
        check(!result.converged || !result.precision_certified, name+" rejected state");
        if (device && states.size()>1)
            check(!result.gpu_used && result.gpu_status_code==3,
                  name+" device failure remains visible");
    }
}
}

int main() {
    HdfeOptions fast;
    fast.tol=1e-8; fast.max_iter=100000; fast.from_auto=true;
    fast.tolerance_mode=ToleranceMode::XhdfeFast;
    fast.convergence_criterion=ConvergenceCriterion::Auto;
    const std::vector<Solve> akm={{30,true,false,1.3525671445292131e-5},
        {62,true,false,2.4754871466158074e-6},
        {122,true,false,1.5704318747310904e-7},
        {146,true,false,9.655684038217385e-8},
        {183,true,true,3.683493605810405e-8}};
    const std::vector<double> ftol={1e-8,1e-10,1e-12,1e-13,1e-15};
    exercise("AKM2 saved states",fast,akm,ftol,true);
    exercise("device state retention",fast,akm,ftol,true,true);
    auto save=fast; save.retain_fixed_effects=true;
    exercise("retained FE state",save,akm,ftol,true);
    for (auto flags : {std::pair<bool,bool>{true,false},{false,true},{false,false}}) {
        auto bad=akm; bad.back().converged=flags.first; bad.back().certified=flags.second;
        if (!flags.second) bad.back().block=9e-8;
        exercise("invalid final flags",fast,bad,ftol,false);
        exercise("invalid device flags",fast,bad,ftol,false,true);
    }
    auto nonfinite=akm;
    nonfinite.back().block=std::numeric_limits<double>::quiet_NaN();
    nonfinite.back().certified=false;
    exercise("uncertified nonfinite",fast,nonfinite,ftol,false);
    exercise("primary already accepted",fast,{{30,true,true,5e-9}},{1e-8},true);
    auto explicit_method=fast; explicit_method.from_auto=false;
    exercise("explicit method unchanged",explicit_method,{{30,true,true,3.68e-8}},
             {1e-8},true);
    auto limited=fast; limited.max_iter=543;
    exercise("certified at budget boundary",limited,akm,ftol,true);
    auto bad_budget=akm; bad_budget.back().certified=false;
    exercise("uncertified at budget boundary",limited,bad_budget,ftol,false);
    auto no_budget=fast; no_budget.max_iter=30;
    exercise("no remaining budget",no_budget,{{30,true,false,1e-5}},{1e-8},false);
    auto comparable=fast; comparable.tolerance_mode=ToleranceMode::ReghdfeComparable;
    const std::vector<Solve> comp={{30,true,false,1e-5},{62,true,false,2e-6},
        {122,true,false,1e-7},{146,true,false,9e-8},
        {183,true,false,9e-9},{200,true,true,3.68e-9}};
    const std::vector<double> ctol={1e-8,1e-10,1e-11,1e-13,1e-14,1e-15};
    exercise("Comparable own certificate",comparable,comp,ctol,true);
    auto wrong_mode=comp; wrong_mode.back().block=3.68e-8;
    wrong_mode.back().certified=false;
    exercise("Fast certificate cannot pass Comparable",comparable,wrong_mode,ctol,false);
    std::cout << "cases=" << cases << " failures=" << failures << '\n';
    return failures != 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='source tree with include/, src/ and bundled Eigen')
    parser.add_argument('--source', type=Path, help='absorption source; defaults to ROOT/src/fe_absorption.cpp')
    parser.add_argument('--cxx', default='c++', help='C++ compiler executable')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.source = (args.source or args.root/'src/fe_absorption.cpp').resolve()
    source = args.source.read_text()
    core = (args.root/'src/hdfe_regressor_v11.cpp').read_text()
    begin = source.index('\nAbsorptionResult absorb_fixed_effects_v6(')
    end = source.index('\n    const int n = static_cast<int>(y.size());', begin)
    wrapper = source[begin:end]+'\nthrow std::logic_error("fixture requires slopes");\n}\n'
    begin = source.index('double effective_absorption_tolerance(')
    end = source.index('\ndouble krylov_parity_tolerance(', begin)
    effective = source[begin:end]
    begin = core.index('void require_accepted_absorption(')
    guard = core[begin:core.index('\nstd::string group_failure_criteria(', begin)]
    args.out.mkdir(parents=True, exist_ok=False)
    harness = args.out/'contract.cpp'
    executable = args.out/'contract'
    harness.write_text('#include <algorithm>\n#include <cmath>\n#include <iostream>\n'
                       '#include <limits>\n#include <sstream>\n#include <stdexcept>\n'
                       '#include "fe_absorption.hpp"\n'+FIXTURE+SEED_ADAPTER+
                       '\nnamespace hdfe { namespace detail {\n'+effective+wrapper+'\n}}\n'
                       'namespace hdfe { namespace v11 {\n'+guard+'\n}}\n'+CONTROLS)
    command = [args.cxx, '-std=c++17', '-O2', '-ffast-math', '-fno-finite-math-only',
               '-fopenmp', '-I'+str(args.root/'include'), '-I'+str(args.root/'r/xhdfe/src/eigen'),
               str(harness), '-o', str(executable)]
    with (args.out/'compile.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=10)
    (args.out/'stdout.log').write_text(run.stdout)
    (args.out/'stderr.log').write_text(run.stderr)
    result = dict(status='PASS' if run.returncode==0 else 'FAIL', returncode=run.returncode,
                  source=str(args.source.resolve()), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                  core_sha256=hashlib.sha256(core.encode()).hexdigest(), command=command,
                  stdout=run.stdout, stderr=run.stderr, new_estimator_fits=0,
                  scope='Actual slope wrapper and refusal guard; scripted solve/certificate outputs. GPU metadata controls do not execute CUDA.')
    with (args.out/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'status':result['status'], 'out':str(args.out), 'stdout':run.stdout}))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
