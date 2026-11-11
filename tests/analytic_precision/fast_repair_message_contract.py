"""P3-fix (C): compile the real refusal guard and check the xhdfe-fast repair message.

Up to the P3-fix candidate, a CUDA repair that converged and passed the public
certificate (relative residual <= 8*tol) but missed the bounded xhdfe-fast repair
target (0.5*tol) was refused as "did not converge" (P3 reproducer: relative residual
5.04e-9 against the target 5e-9, Verifications/perf_fereg_gap_20261106/
RELATORIO_P3_20261108.md). The fit is still refused; the message now says what
failed. This contract compiles require_accepted_absorption from
src/hdfe_regressor_v11.cpp (as absorption_failure_diagnostic_contract.py does) and
checks: the (C) message, the measured residual and the iteration accounting in it, the
cold restarts and the device LSMR when they ran, no estimates, and that the
messages of every other refusal are unchanged (the field defaults to 0). The text
matched here is the C_MESSAGE of cuda_fast_restart_contract.py.
usage: fast_repair_message_contract.py --eigen DIR --out NEW_DIR [--source FILE] [--cxx PATH]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BEGIN = 'void require_accepted_absorption('
END = '\nstd::string group_failure_criteria('
C_MESSAGE = 'missed the bounded xhdfe-fast repair target'

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
    result.iterations = 166;
    result.converged = false;
    result.precision_certified = false;
    result.gpu_status_code = 1;
    result.abs_residual_rel = 5.039979e-09;
    result.fast_repair_target = 5e-09;
    const std::string c = refused(options, result);
    check(has(c, "HDFE absorption: the CUDA repair converged and passed the public certificate "
                 "but missed the bounded xhdfe-fast repair target (relative residual <= "
                 "5.000000e-09)"), "(C) message and target");
    check(!has(c, "did not converge"), "(C) not reported as a non-convergence");
    check(!has(c, "precision certification failed"), "(C) not reported as a certificate failure");
    check(has(c, "iterations=166/100000"), "(C) iteration accounting");
    check(has(c, "requested tolerance=1.000000e-08"), "(C) requested tolerance");
    check(has(c, "relative residual=5.039979e-09"), "(C) measured residual");
    check(has(c, "No estimates returned."), "(C) remains a refusal");
    check(!has(c, "cold restart"), "(C) no restart reported when none ran");
    std::cout << "C_MESSAGE " << c << '\n';
    result.cuda_fast_cg_restarts = 2;
    const std::string cg = refused(options, result);
    check(has(cg, "target (relative residual <= 5.000000e-09) after 2 cold restart(s) of its "
                  "broken CG stage;"), "(C) cold restarts reported");
    check(!has(cg, "device LSMR"), "(C) no LSMR reported when it did not run");
    result.cuda_fast_lsmr_restart = true;
    const std::string lsmr = refused(options, result);
    check(has(lsmr, "after 2 cold restart(s) of its broken CG stage and the device LSMR;"),
          "(C) the device LSMR reported");
    std::cout << "C_MESSAGE_RESTART " << lsmr << '\n';
    result.converged = result.precision_certified = true;
    check(refused(options, result).empty(), "an accepted result is never refused by the field");
    result.converged = result.precision_certified = false;
    result.slope_accuracy_retry_stages = 4;
    check(has(refused(options, result), "heterogeneous-slope acceptance criteria not met"),
          "the slope continuation message keeps its precedence");
    for (const auto mode : {hdfe::ToleranceMode::XhdfeFast, hdfe::ToleranceMode::ReghdfeComparable}) {
        options.tolerance_mode = mode;
        for (int gpu = 0; gpu <= 5; ++gpu)
            for (int converged = 0; converged < 2; ++converged)
                for (int certified = 0; certified < 2; ++certified) {
                    hdfe::detail::AbsorptionResult plain;
                    plain.iterations = 543;
                    plain.gpu_status_code = gpu;
                    plain.converged = converged;
                    plain.precision_certified = certified;
                    const std::string message = refused(options, plain);
                    check(message.empty() == (converged && certified), "acceptance truth table");
                    check(!has(message, "xhdfe-fast repair target"), "default field inert");
                    if (!converged) check(has(message, ": did not converge"), "ordinary stop unchanged");
                    if (converged && !certified)
                        check(has(message, ": precision certification failed"), "certificate unchanged");
                }
    }
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
    assert C_MESSAGE in guard, 'the (C) message is missing from the guard'
    args.out.mkdir(parents=True, exist_ok=False)
    harness = args.out/'fast_repair_message.cpp'
    executable = args.out/'fast_repair_message'
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
                   source=str(args.source.resolve()),
                   source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   guard_sha256=hashlib.sha256(guard.encode()).hexdigest(), command=command,
                   scope='Exact compiled guard: the (C) message, its accounting, restarts, '
                         'and the unchanged acceptance truth table', estimator_fits=0)
    with (args.out/'RESULT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], out=str(args.out), returncode=run.returncode)))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
