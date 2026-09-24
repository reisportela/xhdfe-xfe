"""Equal sample counts must not conceal different retained rows or stale data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scratch', type=Path, required=True)
args = parser.parse_args()
args.scratch.mkdir(parents=True, exist_ok=False)
expected = {'matching': 'MATCH', 'different_rows_same_count': 'MISMATCH',
            'tampered_payload': 'MISSING_OR_INVALID_EVIDENCE',
            'missing_receipt': 'MISSING_OR_INVALID_EVIDENCE',
            'refused_fit_with_stale_files': 'FIT_NOT_RETURNED'}
for name in expected:
    for order, role in enumerate(('baseline', 'candidate'), 1):
        folder = args.scratch / 'stata_fixture/raw' / name / 'pair_001' / f'order_{order}_{role}'
        folder.mkdir(parents=True)
        refused = role == 'candidate' and name == 'refused_fit_with_stale_files'
        (folder / 'leg_record.json').write_text(json.dumps(
            {'worker_row': {'success': '0' if refused else '1', 'nobs': '1'}}))
        payload = b'1\n0\n'
        if role == 'candidate' and name in ('different_rows_same_count', 'tampered_payload'):
            payload = b'0\n1\n'
        (folder / 'result_sample.csv').write_bytes(payload)
        if role == 'candidate' and name == 'missing_receipt':
            continue
        digest_input = b'1\n0\n' if name == 'tampered_payload' else payload
        digest = hashlib.sha256(digest_input).hexdigest()
        (folder / 'result_sample.sha256').write_text(digest + '  result_sample.csv\n')

output = args.scratch / 'result.json'
command = [sys.executable, '-B', str(Path(__file__).with_name('compare_stata_samples.py')),
           str(args.scratch), '--output', str(output), '--verify-payload']
run = subprocess.run(command, text=True, capture_output=True)
result = json.loads(output.read_text())
observed = {row['dataset']: row['status'] for row in result['rows']}
assert run.returncode == 1 and observed == expected, (run.returncode, observed, run.stderr)
empty = args.scratch / 'empty'
empty.mkdir()
empty_command = [sys.executable, '-B', str(Path(__file__).with_name('compare_stata_samples.py')),
                 str(empty), '--output', str(empty / 'result.json'), '--verify-payload']
empty_run = subprocess.run(empty_command, text=True, capture_output=True)
assert empty_run.returncode == 1
assert json.loads((empty / 'result.json').read_text())['rows'] == []
print('PASS: matching/mismatched masks, tampering, missing receipts, stale refused fits, and empty coverage')
