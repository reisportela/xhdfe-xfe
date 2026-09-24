"""Compile the real refusal guard and check the recorded AKM2 diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BEGIN = 'void require_accepted_absorption('
END = '\nstd::string group_failure_criteria('

CONTROLS = r'''
namespace {
int failures = 0;
void check(bool passed, const char* label) {
    if (!passed) { std::cerr << "FAIL " << label << '\n'; ++failures; }
}
std::string refused(const hdfe::HdfeOptions& options,
                    const hdfe::detail::AbsorptionResult& result) {
    try {
        hdfe::v11::require_accepted_absorption(options, result, "HDFE absorption");
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& text, const char* value) {
    return text.find(value) != std::string::npos;
}
}
int main() {
    hdfe::HdfeOptions options;
    options.tol = 1e-8;
    options.max_iter = 100000;
    options.tolerance_mode = hdfe::ToleranceMode::XhdfeFast;
    hdfe::detail::AbsorptionResult result;
    result.iterations = 543;
    result.converged = false;
    result.precision_certified = false;
    result.abs_residual_rel = 4.534855e-13;
    result.slope_block_residual_rel = 3.683493605810405e-8;
    result.slope_accuracy_retry_stages = 4;
    result.slope_internal_tolerance = 1e-15;
    const std::string akm = refused(options, result);
    check(has(akm, "heterogeneous-slope acceptance criteria not met"), "AKM2 acceptance reason");
    check(!has(akm, "did not converge"), "AKM2 does not mislabel the last solve");
    check(has(akm, "slope block residual=3.683494e-08"), "AKM2 measured block");
    check(has(akm, "block certificate limit=8.000000e-08"), "AKM2 certificate limit");
    check(has(akm, "continuation target=1.000000e-08"), "AKM2 continuation target");
    check(has(akm, "continuation stages=4"), "AKM2 stage count");
    check(has(akm, "last internal tolerance=1.000000e-15"), "AKM2 internal tolerance");
    check(has(akm, "iteration budget remaining"), "AKM2 sequence exhaustion");
    check(!has(akm, "increase max_iter"), "AKM2 no false iteration-limit advice");
    check(has(akm, "No estimates returned."), "AKM2 remains refused");
    std::cout << "AKM2_MESSAGE " << akm << '\n';

    for (const auto mode : {hdfe::ToleranceMode::XhdfeFast,
                            hdfe::ToleranceMode::ReghdfeComparable}) {
        options.tolerance_mode = mode;
        for (int stages : {0, 4}) for (int gpu = 0; gpu <= 5; ++gpu)
            for (int converged = 0; converged < 2; ++converged)
                for (int certified = 0; certified < 2; ++certified) {
                    result.slope_accuracy_retry_stages = stages;
                    result.gpu_status_code = gpu;
                    result.converged = converged;
                    result.precision_certified = certified;
                    const std::string message = refused(options, result);
                    check(message.empty() == (converged && certified), "acceptance truth table");
                    if (!message.empty()) check(has(message, "No estimates returned."), "refusal retained");
                    if (stages == 0 && !converged)
                        check(has(message, ": did not converge"), "ordinary stop unchanged");
                    if (stages == 0 && converged && !certified)
                        check(has(message, ": precision certification failed"), "ordinary certificate unchanged");
                }
    }
    result.gpu_status_code = 3;
    result.converged = result.precision_certified = false;
    result.slope_accuracy_retry_stages = 4;
    const std::string gpu = refused(options, result);
    check(!has(gpu, "did not converge"), "GPU continuation acceptance reason");
    check(has(gpu, "block certificate limit=8.000000e-09"), "Comparable certificate limit");
    check(has(gpu, "continuation target=1.000000e-09"), "Comparable continuation target");
    result.iterations = options.max_iter;
    const std::string exhausted = refused(options, result);
    check(has(exhausted, "The iteration limit was reached"), "actual iteration limit");
    check(!has(exhausted, "iteration budget remaining"), "exhausted budget not misreported");
    std::cout << "CONTROLS failures=" << failures << '\n';
    return failures != 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'src/hdfe_regressor_v11.cpp')
    parser.add_argument('--eigen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cxx', default='/bin/c++')
    args = parser.parse_args()
    source = args.source.read_text()
    assert source.count(BEGIN) == source.count(END) == 1
    guard = source[source.index(BEGIN):source.index(END)]
    args.out.mkdir(parents=True, exist_ok=False)
    harness = args.out/'guard_contract.cpp'
    executable = args.out/'guard_contract'
    text = ('#include <algorithm>\n#include <limits>\n#include <sstream>\n'
            '#include <stdexcept>\n#include <iostream>\n#include "fe_absorption.hpp"\n'
            'namespace hdfe { namespace v11 {\n'+guard+'\n}}\n'+CONTROLS)
    harness.write_text(text)
    command = [args.cxx, '-std=c++17', '-O2', '-ffast-math', '-fno-finite-math-only',
               '-I'+str(ROOT/'include'), '-I'+str(args.eigen), str(harness), '-o', str(executable)]
    with (args.out/'compile.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    (args.out/'stdout.log').write_text(run.stdout)
    (args.out/'stderr.log').write_text(run.stderr)
    receipt = dict(status='PASS' if run.returncode == 0 else 'FAIL', returncode=run.returncode,
                   source=str(args.source.resolve()), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   guard_sha256=hashlib.sha256(guard.encode()).hexdigest(), command=command,
                   scope='Exact compiled guard, recorded AKM2 state, 96 acceptance states and diagnostic controls',
                   estimator_fits=0, tolerance_changes=0)
    with (args.out/'RESULT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], out=str(args.out), returncode=run.returncode)))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
