"""Audit a classification-only correction against immutable completed evidence.

No PASS/failure or numeric threshold may change. All execution sources except
the evaluator, all fixture bytes and all active binaries must retain custody.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate import assess, selftest
from run import clean_json, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('campaign', type=Path)
args = parser.parse_args()
campaign = args.campaign.resolve()
manifest = json.loads((campaign/'manifest.json').read_text())
evaluator = Path(__file__).with_name('evaluate.py').resolve()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for path, expected in manifest['files'].items():
    original = Path(path)
    actual = campaign/'source_snapshot'/original.name if original==evaluator else original
    assert sha(actual)==expected, path
for name, files in manifest['fixtures'].items():
    fixture = campaign/'fixtures'/name
    assert all(sha(fixture/file)==expected for file,expected in files.items()), name
    selftest(json.loads((fixture/'oracle.json').read_text()),
             json.loads((fixture/'case.json').read_text()), None, fixture)
target = campaign/'classification_review'
target.mkdir(exist_ok=False)
rows, changes, raw_hashes = [], [], {}
for job in manifest['jobs']:
    directory = campaign/'attempts'/job['id']
    old = json.loads((directory/'assessment.json').read_text())
    raw = json.loads((directory/'raw.json').read_text())
    raw_hashes[job['id']] = sha(directory/'raw.json')
    current = assess(job, raw, campaign/'fixtures'/job['case'])
    current['result'] = {k:v for k,v in raw.items() if k not in ('groups','residuals','recovered_fe')}
    assert (old['verdict']=='PASS') == (current['verdict']=='PASS'), job['id']
    if old['verdict'] != current['verdict']:
        assert old['verdict']=='FALSE_CONVERGENCE'
        assert current['verdict']=='FAIL_SAMPLE_PROVENANCE'
        assert old['metrics']['sample_ok'] is False
        changes.append(dict(id=job['id'], old=old['verdict'], new=current['verdict']))
    rows.append(current)
xhdfe = [r for r in rows if r['engine']=='xhdfe']
defaults = [r for r in xhdfe if r['method']=='auto' and r['precision']=='default'
            and not r.get('savefe') and not r.get('expected_rejection')]
summary = dict(utc=datetime.now(timezone.utc).isoformat(),
               source_manifest_sha256=sha(campaign/'manifest.json'),
               original_evaluator_sha256=manifest['files'][str(evaluator)],
               evaluator_sha256=sha(evaluator), regrade_script_sha256=sha(Path(__file__)),
               numerical_gates_and_pass_set_unchanged=True, changes=changes,
               counts=dict(Counter(r['verdict'] for r in rows)),
               xhdfe_counts=dict(Counter(r['verdict'] for r in xhdfe)),
               default_counts=dict(Counter(r['verdict'] for r in defaults)),
               xhdfe_scoped_certified=all(r['verdict'] in ('PASS','UNSUPPORTED_EXPECTED') for r in xhdfe),
               raw_sha256=raw_hashes, pending=manifest['pending'], rows=rows)
write_json(target/'report.json',summary)
with (target/'evaluate.py').open('xb') as h:h.write(evaluator.read_bytes())
print(json.dumps({k:v for k,v in clean_json(summary).items() if k not in ('rows','raw_sha256','changes')},indent=2))
print('CLASSIFICATION_ONLY_CHANGES',len(changes))
