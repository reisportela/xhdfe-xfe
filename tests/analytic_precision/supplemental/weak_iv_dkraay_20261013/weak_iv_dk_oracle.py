"""Weak-instrument 2SLS under Driscoll-Kraay with first-stage diagnostics.

Reuses the 11oct2026 independent oracle (explicit WLS/2SLS and lag-by-lag
Bartlett sums, never the candidate's output) on a panel whose excluded
instrument carries a small loading, and adds an explicit first-stage oracle:
the DK Wald statistic of the excluded instrument, b_z^2 / V_zz, its residual
degrees of freedom, its p-value and the partial R^2, graded against e(first).
The structural allowance is predeclared from the conditioning of the 2SLS
score Gram matrix; nothing is loosened after observing the candidate.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, re, subprocess
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / 'dkraay_20261011' / 'independent_oracle.py'
spec = importlib.util.spec_from_file_location('independent_oracle', ORACLE)
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)
oracle = oracle_module.oracle
U = 2.0 ** -53


def weak_fixture(loading):
    rng = np.random.default_rng(20261013)
    ident = np.repeat(np.arange(1, 9), 13)
    time = np.tile(np.arange(1, 14), 8)
    x1, z, noise = rng.normal(size=(3, len(time)))
    x2 = loading * z + .4 * noise + rng.normal(size=len(time))
    y = 1.3 + .7 * x1 - .45 * x2 + rng.normal(size=8)[ident - 1] + rng.normal(size=13)[time - 1] + .6 * noise
    return pd.DataFrame(dict(id=ident, time=time, x1=x1, x2=x2, z=z, y=y, row=np.arange(1, len(time) + 1)))


def score_condition(df, fe):
    """kappa_2 of the 2SLS score Gram matrix on the represented inputs."""
    X = df[['x1', 'x2']].to_numpy()
    Z = df[['x1', 'z']].to_numpy()
    if fe:
        D = np.column_stack([pd.get_dummies(df[f], dtype=float).to_numpy() for f in fe])
        X = X - D @ np.linalg.lstsq(D, X, rcond=None)[0]
        Z = Z - D @ np.linalg.lstsq(D, Z, rcond=None)[0]
    else:
        Z = np.column_stack([Z, np.ones(len(df))])
        X = np.column_stack([X, np.ones(len(df))])
    scores = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return float(np.linalg.cond(scores.T @ scores))


def f_tail(F, df1, df2):
    import mpmath as mp
    x = mp.mpf(df2) / (df2 + df1 * mp.mpf(F))
    return float(mp.betainc(mp.mpf(df2) / 2, mp.mpf(df1) / 2, 0, x, regularized=True))


def cases():
    specs = []
    for loading, label in ((0.05, 'veryweak'), (0.15, 'weak'), (0.9, 'strong')):
        for fe in ([], ['id'], ['id', 'time']):
            specs.append(dict(name=f"{label}_" + ('none' if not fe else '_'.join(fe)), fe=fe, bw=3,
                              x=['x1', 'x2'], iv=True, loading=loading))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path, help='New directory for synthetic fixtures and receipts')
    ap.add_argument('--package', type=Path, default=HERE.parents[3] / 'stata')
    ap.add_argument('--backend', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()
    package = args.package.resolve()
    for name in ['xhdfe.ado', 'xhdfe.plugin', 'xhdfe_estat.ado']:
        if not (package / name).is_file():
            raise FileNotFoundError(package / name)
    run = args.output.resolve()
    if any(c in str(p) for p in [package, run] for c in ['"', '`', '$', '\n', '\r']):
        raise ValueError('Unsupported quoting in a Stata path')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'tmp').mkdir()
    for folder in ['plus', 'personal', 'oldplace']:
        (run / 'ado' / folder).mkdir(parents=True)
    setup = ''.join(f'sysdir set {key} "{run}/ado/{folder}/"\n' for key, folder in
                    [('PLUS', 'plus'), ('PERSONAL', 'personal'), ('OLDPLACE', 'oldplace')])
    setup += f'global S_ADO "BASE;SITE;{package};PERSONAL;PLUS;OLDPLACE"\nset more off\nset linesize 255\n'
    for name in ['xhdfe.ado', 'xhdfe.plugin']:
        setup += f'findfile {name}\nassert "`r(fn)\'" == "{package}/{name}"\n'
    outputs = []
    commands = []
    for spec in cases():
        df = weak_fixture(spec['loading'])
        name = spec['name']
        structural = oracle(df, spec)
        first = oracle(df.assign(y=df.x2), dict(fe=spec['fe'], bw=spec['bw'], x=['x1', 'z']))
        restricted = oracle(df.assign(y=df.x2), dict(fe=spec['fe'], bw=spec['bw'], x=['x1']))
        F = float(first['b'][1] ** 2 / first['V'][1, 1])
        expected = dict(structural=structural, first_F=F, first_df=1, first_df_r=float(first['df_r']),
                        first_p=f_tail(F, 1, float(first['df_r'])),
                        first_pr2=max(0.0, min(1.0, 1 - first['rss'] / restricted['rss'])),
                        kappa=score_condition(df, spec['fe']))
        df['oracle_order'] = np.arange(1, len(df) + 1)
        df.to_stata(run / (name + '.dta'), write_index=False, version=118)
        fe = ' '.join(spec['fe'])
        opts = f'absorb({fe})' if fe else 'noabsorb'
        cmd = (f'xhdfe y x1 x2, {opts} endogenous(x2) instruments(z) vce(dkraay {spec["bw"]}) '
               f'gpubackend({args.backend}) numthreads(2) ffirst residuals(__u)')
        outputs.append((spec, expected))
        commands.append(f'''use "{run / name}.dta", clear
quietly xtset row time
sort oracle_order
capture noisily {cmd}
local fit_rc = _rc
file open receipt using "{run / name}.txt", write text
file write receipt "rc " (`fit_rc') _n
if !`fit_rc' {{
    assert "`e(vce)'" == "dkraay"
    matrix bb = e(b)
    matrix vv = e(V)
    matrix ff = e(first)
    matrix fg = e(first_gpu_used)
    file write receipt "b"
    forvalues j=1/`=colsof(bb)' {{
        file write receipt " " %24.17e (bb[1,`j'])
    }}
    file write receipt _n "V"
    forvalues j=1/`=rowsof(vv)' {{
        forvalues k=1/`=colsof(vv)' {{
            file write receipt " " %24.17e (vv[`j',`k'])
        }}
    }}
    file write receipt _n "first"
    forvalues j=1/5 {{
        file write receipt " " %24.17e (ff[`j',1])
    }}
    file write receipt _n "first_gpu_used " %24.17e (fg[1,1]) " " %24.17e (fg[2,1]) _n
    foreach statistic in N rss df_a df_m df_r N_clust dkraay_bw gpu_used converged {{
        file write receipt "`statistic' " %24.17e (e(`statistic')) _n
    }}
    gen byte __sample = e(sample)
    export delimited __sample __u using "{run / name}_rows.csv", replace
}}
file close receipt
''')
    do = run / 'oracle.do'
    do.write_text(setup + ''.join(commands) + 'di "WEAK_IV_DK_COMPLETE"\nexit\n')
    env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH' and not k.startswith('XHDFE_')}
    env.update(STATATMP=str(run / 'tmp'), TMPDIR=str(run / 'tmp'), CUDA_CACHE_PATH=str(run / 'tmp/cuda'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    proc = subprocess.run(['/usr/local/stata/stata-mp', '-q', '-b', 'do', str(do)], cwd=run, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    (run / 'process.txt').write_bytes(proc.stdout)
    log = (run / 'oracle.log').read_text(errors='replace')
    results = []
    for spec, expected in outputs:
        p = run / (spec['name'] + '.txt')
        result = dict(case=spec['name'], loading=spec['loading'], status='FAIL', kappa=expected['kappa'],
                      oracle_first_F=expected['first_F'])
        if p.exists():
            actual = {line.split()[0]: np.array([float(x) if x != '.' else np.nan for x in line.split()[1:]])
                      for line in p.read_text().splitlines()}
            result['rc'] = int(actual['rc'][0])
            if args.backend == 'cuda' and not spec['fe']:
                # Documented fail-closed refusal: CUDA absorption needs a fixed-effect
                # dimension; the frontend must refuse rather than fall back to CPU.
                refused = 'CUDA absorption requires at least one fixed-effect dimension' in log
                result['expected_refusal'] = True
                result['status'] = 'PASS' if result['rc'] != 0 and refused else 'FAIL'
            elif result['rc'] == 0:
                ref = expected['structural']
                V = actual['V'].reshape(ref['V'].shape)
                # Predeclared allowance: 64 u kappa ||b|| on b and 64 u kappa^2 on the scaled V.
                allowance_b = 64 * U * expected['kappa'] * float(np.linalg.norm(ref['b']))
                allowance_v = 64 * U * expected['kappa'] ** 2
                result['b_error'] = float(np.max(np.abs(actual['b'] - ref['b'])))
                scales = np.sqrt(np.maximum(np.diag(ref['V']), 0))
                norm = np.outer(scales, scales)
                result['V_scaled_error'] = float(np.max(np.abs(V - ref['V']) / np.maximum(norm, 1e-12)))
                result['allowances'] = dict(b=1e-9 + allowance_b, V_scaled=1e-8 + allowance_v)
                result['stats'] = {k: dict(actual=float(actual[k][0]), expected=float(ref[k]))
                                   for k in ['N', 'rss', 'df_a', 'df_m', 'df_r', 'N_clust', 'dkraay_bw']}
                first = actual['first']
                result['first'] = dict(pr2=dict(actual=float(first[0]), expected=expected['first_pr2']),
                                       F=dict(actual=float(first[1]), expected=expected['first_F']),
                                       df=dict(actual=float(first[2]), expected=1.0),
                                       df_r=dict(actual=float(first[3]), expected=expected['first_df_r']),
                                       p=dict(actual=float(first[4]), expected=expected['first_p']))
                first_ok = (abs(first[1] - expected['first_F']) <= 1e-8 * max(1.0, expected['first_F'])
                            and first[2] == 1.0 and first[3] == expected['first_df_r']
                            and abs(first[4] - expected['first_p']) <= 1e-7 * max(1.0, expected['first_p'])
                            and abs(first[0] - expected['first_pr2']) <= 1e-8)
                rows = pd.read_csv(run / (spec['name'] + '_rows.csv'))
                sample_ok = np.array_equal(rows['__sample'].to_numpy(), ref['mask'])
                result['sample_equal'] = bool(sample_ok)
                result['residual_error'] = float(np.max(np.abs(
                    rows.loc[ref['mask'].astype(bool), '__u'].to_numpy() - ref['resid'])))
                stats_ok = all(np.isclose(v['actual'], v['expected'], rtol=1e-9, atol=1e-10)
                               for v in result['stats'].values())
                result['gpu_used'] = float(actual['gpu_used'][0])
                result['first_gpu_used'] = actual['first_gpu_used'].tolist()
                gpu_ok = (args.backend == 'cpu' and result['gpu_used'] == 0) or \
                         (args.backend == 'cuda' and (not spec['fe'] or
                          (result['gpu_used'] == 1 and result['first_gpu_used'][0] == 1)))
                result['gpu_ok'] = bool(gpu_ok)
                result['first_ok'] = bool(first_ok)
                result['status'] = 'PASS' if (result['b_error'] <= result['allowances']['b']
                                             and result['V_scaled_error'] <= result['allowances']['V_scaled']
                                             and result['residual_error'] < 1e-8 and stats_ok and sample_ok
                                             and actual['converged'][0] == 1 and first_ok and gpu_ok) else 'FAIL'
        results.append(result)
    complete = bool(re.search(r'^WEAK_IV_DK_COMPLETE$', log, re.M)) and proc.returncode == 0 and \
        not re.search(r'^r\(\d+\);', log, re.M)
    # Expected CUDA refusals are captured by `capture noisily`; they do not end the do-file.
    receipt = dict(complete=complete, backend=args.backend, cases=results,
                   sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [HERE / 'weak_iv_dk_oracle.py', ORACLE, package / 'xhdfe.ado', package / 'xhdfe.plugin',
                             package / 'xhdfe_estat.ado']})
    (run / 'RESULTS.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for r in results:
        print(r['case'], r['status'], r.get('rc'), 'F=%.4g' % r['oracle_first_F'], 'kappa=%.3g' % r['kappa'],
              r.get('b_error'), r.get('V_scaled_error'), r.get('gpu_used'), flush=True)
    raise SystemExit(0 if complete and all(r['status'] == 'PASS' for r in results) else 1)


if __name__ == '__main__':
    main()
