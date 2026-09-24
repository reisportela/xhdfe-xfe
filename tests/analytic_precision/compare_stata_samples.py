"""Compare exact Stata e(sample) exports; payload reads run outside timings."""
from pathlib import Path
import argparse
import hashlib
import json
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--verify-payload', action='store_true')
args = parser.parse_args()
rows = []
for pair in sorted(args.root.glob('stata_*/raw/*/pair_*')):
    row = dict(cell=pair.parents[2].name, dataset=pair.parent.name, pair=pair.name)
    folders = {role: next(iter(pair.glob('*' + role)), None)
               for role in ('baseline', 'candidate')}
    checkpoints = {role: folder / 'leg_record.json' if folder else None
                   for role, folder in folders.items()}
    if any(path is None or not path.is_file() for path in checkpoints.values()):
        row['status'] = 'PENDING'
    elif any(json.loads(path.read_text())['worker_row'].get('success') != '1'
             for path in checkpoints.values()):
        row['status'] = 'FIT_NOT_RETURNED'
    else:
        try:
            digests = {}
            for role, folder in folders.items():
                receipt, payload = folder / 'result_sample.sha256', folder / 'result_sample.csv'
                digest = receipt.read_text().split()[0]
                if not re.fullmatch('[0-9a-f]{64}', digest) or not payload.is_file():
                    raise ValueError('Invalid digest receipt or missing sample payload')
                if args.verify_payload:
                    hasher = hashlib.sha256()
                    with payload.open('rb') as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                            hasher.update(chunk)
                    actual = hasher.hexdigest()
                    if actual != digest:
                        raise ValueError('Sample payload does not match its digest')
                digests[role] = digest
            row.update(digests=digests, payload_verified=args.verify_payload,
                       status='MATCH' if len(set(digests.values())) == 1 else 'MISMATCH')
        except (OSError, ValueError, IndexError) as error:
            row.update(status='MISSING_OR_INVALID_EVIDENCE', error=str(error))
    rows.append(row)
result = dict(rows=rows, counts={status: sum(r['status'] == status for r in rows)
              for status in sorted({r['status'] for r in rows})},
              qualification='MATCH proves sample identity only. It does not certify estimates or performance.')
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['counts']))
raise SystemExit(not rows or any(r['status'] in ('MISMATCH', 'MISSING_OR_INVALID_EVIDENCE') for r in rows))
