"""Regrade preserved positive rows and reject identified b/u perturbations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_t01 import assess_t01, _within_score_design


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, action='append', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
before = {}
rows = []
selected = None
for report_path in args.report:
    report_path = report_path.resolve()
    before[str(report_path)] = sha(report_path)
    report = json.loads(report_path.read_text())
    for previous in report['rows']:
        if (previous['engine'] != 'xhdfe' or previous['verdict'] != 'PASS' or
                not previous.get('t01', {}).get('applicable')):
            continue
        attempt = report_path.parent / 'attempts' / previous['id']
        raw_path, job_path = attempt / 'raw.json', attempt / 'job.json'
        before[str(raw_path)] = sha(raw_path)
        before[str(job_path)] = sha(job_path)
        raw, job = json.loads(raw_path.read_text()), json.loads(job_path.read_text())
        fixture = report_path.parent / 'fixtures' / previous['case']
        if not fixture.exists():
            fixture = Path(job['fixture'])
        for name in ('explicit.dta', 'design.npz', 'case.json', 'oracle.json'):
            before[str(fixture / name)] = sha(fixture / name)
        updated = assess_t01(job, raw, fixture, previous)
        rows.append(dict(id=previous['id'], report=str(report_path),
            previous_verdict=previous['t01']['g1_verdict'], verdict=updated['g1_verdict'],
            tau=updated['tau'], fe_moments=updated['metrics']['fe_moments'],
            raw_regressor_moments=updated['metrics']['raw_regressor_moments'],
            within_regressor_moments=updated['metrics']['regressor_moments'],
            checks=updated['checks']))
        if (previous['case'] == 'group_sum' and previous['interface'] == 'native' and
                previous['method'] == 'symmetric-gauss-seidel' and previous['backend'] == 'cpu' and
                previous['precision'] == 'default'):
            selected = job, raw, fixture, previous

assert selected is not None, 'the fixed grouped reproducer is absent'
job, raw, fixture, historical = selected
truth = pd.read_stata(fixture / 'explicit.dta')
truth = truth.loc[truth.expected_sample == 1].set_index('group').loc[raw['groups']]
metadata = json.loads((fixture / 'case.json').read_text())
weights = truth.weight.to_numpy() if metadata.get('weight') else np.ones(len(truth))
X = truth[['x1', 'x2', 'x3']].to_numpy()
D = np.load(fixture / 'design.npz')['D'][raw['groups']]
score = _within_score_design(X, D, weights)
negatives = []

bad_beta = copy.deepcopy(raw)
bad_beta['beta'][0] += .01
assessment = assess_t01(job, bad_beta, fixture, historical)
negatives.append(dict(kind='identified_coefficient_error', verdict=assessment['g1_verdict'],
    passed=not assessment['passed'] and not assessment['checks']['comparable_beta']))

bad_residual = copy.deepcopy(raw)
u = np.asarray(raw['residuals'])
direction = score[:, 0]
perturbed = u + 1e-5 * np.sqrt(weights @ (u * u)) * direction / np.sqrt(weights @ (direction * direction))
bad_residual['residuals'] = perturbed.tolist()
assessment = assess_t01(job, bad_residual, fixture, historical)
negatives.append(dict(kind='identified_residual_score_error', verdict=assessment['g1_verdict'],
    passed=not assessment['passed'] and not assessment['checks']['n1'] and
        assessment['metrics']['fe_moments']['max_normalized'] <= assessment['tau'],
    normal=assessment['metrics']['regressor_moments'], fe=assessment['metrics']['fe_moments']))

unchanged = all(sha(Path(path)) == expected for path, expected in before.items())
fixed = sum(row['previous_verdict'] != 'PASS' and row['verdict'] == 'PASS' for row in rows)
passed = unchanged and fixed == 26 and all(row['verdict'] == 'PASS' for row in rows) and all(
    row['passed'] for row in negatives)
report = dict(classification='evaluation-harness error', passed=passed,
    positive_rows=len(rows), corrected_rows=fixed, preserved_inputs_unchanged=unchanged,
    hashes_before=before, evaluator_sha256=sha(Path(__file__).with_name('evaluate_t01.py')),
    worker_sha256=sha(Path(__file__)), rows=rows, negatives=negatives,
    unchanged_contract='tau, b/V, FE moments and rho; no additional arithmetic allowance',
    scope='offline regrading of original estimates; no estimator rebuild or rerun')
with args.out.open('x') as handle:
    json.dump(report, handle, indent=2, default=lambda value: value.item())
    handle.write('\n')
print(json.dumps(dict(passed=passed, positive_rows=len(rows), corrected_rows=fixed,
                     preserved_inputs_unchanged=unchanged, negatives=negatives)), flush=True)
raise SystemExit(not passed)
