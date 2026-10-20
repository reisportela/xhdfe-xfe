"""Run pinned reghdfe scenarios against isolated, explicitly selected packages.

Original sources are immutable. Every generated adaptation and every outcome
is retained. This runner neither installs packages nor modifies HOME/adopath
outside the Stata child session. Compatibility is not a mathematical oracle.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'upstream/reghdfe'
INTERNAL = {'part2/test-parallel-save-load.do', 'part1/cluster-mwc-meat.do',
            'compatibility/simple-test.do'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_source():
    manifest = json.loads((SOURCE / 'SOURCE.json').read_text())
    for row in manifest['files']:
        assert sha(SOURCE / row['path']) == row['sha256'], row['path']
    return manifest


def user_ado_state():
    root = Path.home() / 'ado'
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def adapt(original, engine, relative, adapted, environment):
    if adapted and engine == 'xhdfe' and relative == 'part1/cluster-after-expand.do':
        return (f'include "{environment}"\n'
                f'do "{environment.parent}/cluster_expansion_contract.do"\n')
    lines = []
    for line in original.splitlines(keepends=True):
        if re.match(r'^\s*cls\s*$', line):
            line = re.sub(r'\bcls\b', 'capture cls', line)
        line = line.replace('adopath + "C:/git/reghdfe/test"', 'adopath + ".."')
        if engine == 'xhdfe' and relative not in INTERNAL and not line.lstrip().startswith(('*', '//')):
            line = re.sub(r'\bstoredresults\b', '_core_results', line)
            if 'cscript ' in line:
                line = line.replace('adofile reghdfe', 'adofile xhdfe')
            elif not re.search(r'\bversion\(\s*[35]\s*\)', line, re.I):
                line = re.sub(r'\breghdfe\b(?!\.)', '_up_xhdfe', line)
        lines.append(line)
        if re.match(r'^\s*(noi(sily)?\s+)?cscript\b', line):
            lines.append(f'include "{environment}"\n')
    result = ''.join(lines)
    if adapted and engine == 'xhdfe':
        if relative == 'part1/options.do':
            result = result.replace("\n\tassert `ic' < e(ic)\n",
                "\n\tassert `ic' > 0 & `ic' <= 100000\n"
                "\tassert e(ic) > 0 & e(ic) <= 100000\n"
                "\tif (`ic' >= e(ic)) display \"CORE_STRICT_UPSTREAM_DIFFERENCE ic loose=\" `ic' \" tight=\" e(ic) \" public_contract=cumulative_budget\"\n")
        if relative == 'part1/cluster-equals-robust.do':
            result = result.replace('scalar: N_clustervars N_clust1',
                                    'scalar: N_clustervars N_clust1 N_clust')
        if relative == 'part1/lagged-slope.do':
            result = result.replace('\tnotrim', "\ttrim_cons `K'")
        if relative == 'part1/fvstrip.do':
            result = result.replace('areg price weight, a(foreign)',
                "tempvar oracle_foreign\n\tgen byte `oracle_foreign' = foreign\n"
                "\tareg price weight, a(`oracle_foreign')")
        if relative == 'part1/predict-aweight.do':
            result = result.replace('\tdi e(df_a)', "\t_up_ll_convention `wvar'\n\tdi e(df_a)")
        if relative == 'part1/alphas.do':
            marker = 'reghdfe price weight length, a(bench1=turn bench2=trunk) version(3)'
            result = result.replace(marker, marker + '\n\t_up_twofe_oracle')
    return result


def run(args):
    source = verify_source()
    plan = json.loads((HERE / 'manifest.json').read_text())['reghdfe']['tests']
    selected = [r for r in plan if r['kind'] != 'infrastructure']
    if args.case:
        assert set(args.case) <= {r['test'] for r in selected}, 'unknown or non-runnable case'
        selected = [r for r in selected if r['test'] in args.case]
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep test outputs inside the selected repository')
    output.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    for name in ['xhdfe.ado', 'xhdfe.plugin']:
        assert (package / name).is_file(), name
    dependency_source = args.dependencies.resolve()
    dependencies = output / 'dependencies'
    shutil.copytree(dependency_source, dependencies)
    assert (dependencies / 'ado/plus/r/reghdfe.ado').is_file()
    assert not (dependencies / 'profile.do').exists()
    harness = output / 'harness'
    shutil.copytree(HERE / 'stata', harness)
    tests = output / 'test'
    shutil.copytree(SOURCE / 'test', tests)
    changes = []
    for target in sorted(tests.rglob('*.do')):
        relative = str(target.relative_to(tests))
        old = (SOURCE / 'test' / relative).read_text()
        if relative == 'setup.do':
            new = f'clear all\ndiscard\ninclude "{harness}/env.do"\n'
        else:
            new = adapt(old, args.engine, relative, args.mode == 'adapted', harness / 'env.do')
        target.write_text(new)
        changes.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                       fromfile='upstream/' + relative, tofile=args.mode + '/' + relative))
    (output / 'ADAPTATION.diff').write_text(''.join(changes))
    artifacts = {str(p): sha(p) for p in [package / 'xhdfe.ado', package / 'xhdfe.plugin',
                 HERE / 'upstream.py', HERE / 'manifest.json', *sorted((HERE / 'stata').iterdir())] if p.is_file()}
    write_new(output / 'INPUTS.json', dict(engine=args.engine, backend=args.backend, mode=args.mode,
        source_commit=source['commit'], artifacts=artifacts, explicit_fe_tolerance=args.fe_tolerance,
        dependency_source=str(dependency_source),
        dependencies={str(p.relative_to(dependencies)): sha(p) for p in dependencies.rglob('*') if p.is_file()},
        selected=[r['test'] for r in selected], no_installs=True, no_HOME_override=True))
    before = user_ado_state()
    outcomes = []
    for position, item in enumerate(selected, 1):
        relative = item['test']
        if args.backend == 'cuda' and item['kind'] == 'reference_internal':
            outcomes.append(dict(test=relative, status='NOT_APPLICABLE_REFERENCE_INTERNAL', xhdfe_calls=0))
            continue
        case = output / 'cases' / relative.replace('/', '__').removesuffix('.do')
        (case / 'tmp').mkdir(parents=True)
        call_receipt = case / 'calls.tsv'
        driver = case / 'driver.do'
        driver.write_text(f'include "{harness}/env.do"\nset seed 123456789\n'
            f'findfile xhdfe.ado\nassert "`r(fn)\'" == "{package}/xhdfe.ado"\n'
            f'findfile xhdfe.plugin\nassert "`r(fn)\'" == "{package}/xhdfe.plugin"\n'
            f'cd "{(tests/relative).parent}"\n'
            f'capture noisily do "{tests/relative}"\nlocal suite_rc = _rc\n'
            'display "XHDFE_CORE_RESULT rc=" `suite_rc\'\n'
            'display "XHDFE_CORE_END"\nexit\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
        env.update(XHDFE_UPSTREAM_ROOT=str(harness), XHDFE_UPSTREAM_TESTROOT=str(tests),
            XHDFE_CORE_DEPENDENCIES=str(dependencies), XHDFE_CORE_PACKAGE=str(package),
            XHDFE_UPSTREAM_CALLS=str(call_receipt), XHDFE_UPSTREAM_BACKEND=args.backend,
            XHDFE_UPSTREAM_EQUIVALENT=str(int(args.mode == 'adapted')),
            XHDFE_CORE_DK_CONTRACT=str(int(args.mode == 'adapted' and relative == 'part1/dkraay.do')),
            XHDFE_CORE_REPEAT_CONTRACT=str(int(args.mode == 'adapted' and args.backend == 'cuda'
                and relative in ('part1/singleton-and-fweights.do', 'part1/memory.do'))),
            XHDFE_UPSTREAM_FETOL=args.fe_tolerance, STATATMP=str(case / 'tmp'),
            TMPDIR=str(case / 'tmp'), CUDA_CACHE_PATH=str(case / 'tmp/cuda'),
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', XHDFE_ABSORPTION_CACHE_MODE='off')
        start = time.monotonic()
        with (case / 'process.log').open('x') as stream:
            try:
                proc = subprocess.run([args.stata, '-q', '-b', 'do', str(driver)], cwd=case,
                    env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                process_rc = proc.returncode
            except subprocess.TimeoutExpired:
                process_rc = 124
        log_path = case / 'driver.log'
        log = log_path.read_text(errors='replace') if log_path.exists() else ''
        markers = re.findall(r'^XHDFE_CORE_RESULT rc=(\d+)\s*$', log, re.M)
        rc = int(markers[-1]) if markers else None
        calls = call_receipt.read_text().splitlines() if call_receipt.exists() else []
        status = 'PASS' if process_rc == 0 and rc == 0 and '\nXHDFE_CORE_END\n' in log else 'FAIL'
        if process_rc == 124:
            status = 'TIMEOUT'
        elif rc is None:
            status = 'INCOMPLETE'
        if args.engine == 'xhdfe':
            if item['kind'] == 'reference_internal':
                status = 'REFERENCE_ONLY_' + status
            elif status == 'PASS' and not calls:
                status = 'NO_CANDIDATE_COVERAGE'
        strict_differences = [x for x in log.splitlines() if x.startswith('CORE_STRICT_UPSTREAM_DIFFERENCE ')]
        if status == 'PASS' and strict_differences:
            status = 'PASS_PUBLISHED_CONTRACT'
        if status == 'PASS' and '\nCORE_EXPECTED_DIFFERENCE_CONFIRMED ' in log:
            status = 'PASS_EXPECTED_DIFFERENCE'
        backend_mismatches = [x for x in calls if 'backend_ok=0' in x]
        if backend_mismatches:
            status = 'FAIL_BACKEND'
        outcome = dict(test=relative, kind=item['kind'], engine=args.engine, backend=args.backend,
            mode=args.mode, status=status, rc=rc, process_rc=process_rc,
            xhdfe_calls=len(calls), cuda_calls=sum('route=cuda' in x and 'gpu=1 ' in x for x in calls),
            cpu_only_controls=sum('cpu_only_noabsorb_control' in x for x in calls),
            seconds=time.monotonic()-start, log=str(log_path),
            strict_upstream_differences=strict_differences,
            backend_mismatches=backend_mismatches,
            optional_reference_notices=[x.strip() for x in log.splitlines()
                if ('not installed' in x or 'skipping' in x) and not x.lstrip().startswith('.')])
        outcomes.append(outcome)
        write_new(case / 'RESULT.json', outcome)
        print(position, '/', len(selected), relative, status, 'rc', rc, 'calls', len(calls), flush=True)
    after = user_ado_state()
    verify_source()
    unchanged = before == after and all(sha(Path(p)) == h for p, h in artifacts.items())
    write_new(output / 'RESULT.json', dict(outcomes=outcomes, custody_unchanged=unchanged,
        source_commit=source['commit'], planned_files=len(plan), selected_files=len(selected),
        infrastructure=[r for r in plan if r['kind'] == 'infrastructure'],
        numerical_certification=False, interpretation='Compatibility outcomes; failures require classification.'))
    assert unchanged, 'Source/artifact/user-ado custody changed'
    return int(any(r['status'] in ('FAIL', 'FAIL_BACKEND', 'TIMEOUT', 'INCOMPLETE', 'NO_CANDIDATE_COVERAGE') for r in outcomes))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--dependencies', type=Path, required=True,
                   help='Existing isolated dependency tree with ado/ and aux/; copied, never installed')
    p.add_argument('--engine', choices=('reghdfe', 'xhdfe'), required=True)
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--mode', choices=('literal', 'adapted'), default='adapted')
    p.add_argument('--fe-tolerance', help='Explicit recovery-only test option; adapted default 1e-8, literal unset')
    p.add_argument('--case', action='append')
    p.add_argument('--stata', default=shutil.which('stata-mp'))
    p.add_argument('--timeout', type=int, default=420)
    args = p.parse_args()
    if not args.stata:
        p.error('Stata executable required')
    if args.engine == 'reghdfe' and args.backend != 'cpu':
        p.error('The upstream reference is CPU-only')
    if args.fe_tolerance is None:
        args.fe_tolerance = '1e-8' if args.mode == 'adapted' else ''
    raise SystemExit(run(args))
