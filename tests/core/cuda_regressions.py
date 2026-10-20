"""Run permanent Stata CUDA regressions; exit zero is insufficient evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
    env.update(XHDFE_STATA_ADOPATH=str(ROOT/'stata'), OPENBLAS_NUM_THREADS='1')
    cases = [('cuda04_cpu', 'cuda_ordinary_weak_graph_contract.do', ['cpu'],
              'CUDA_ORDINARY_WEAK_GRAPH_PASS cpu'),
             ('cuda04_cuda', 'cuda_ordinary_weak_graph_contract.do', ['cuda'],
              'CUDA_ORDINARY_WEAK_GRAPH_PASS cuda'),
             ('cg03', 'cg_cold_restart_floor.do', [], 'CG_COLD_FLOOR_PASS')]
    artifacts = [ROOT/'stata/xhdfe.ado', ROOT/'stata/xhdfe.plugin']
    artifacts += [ROOT/'tests/stata'/name for _, name, _, _ in cases]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    rows = []
    for name, script, arguments, marker in cases:
        folder = out/name
        folder.mkdir()
        with (folder/'process.log').open('x') as log:
            result = subprocess.run(['stata-mp', '-q', '-b', 'do',
                str(ROOT/'tests/stata'/script), *arguments], cwd=folder, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=420)
        text = (folder/Path(script).with_suffix('.log')).read_text()
        ok = result.returncode == 0 and marker in text.splitlines()
        rows.append(dict(case=name, returncode=result.returncode,
                         marker_found=marker in text.splitlines(), status='PASS' if ok else 'FAIL'))
    changed = [p for p, value in hashes.items()
               if hashlib.sha256(Path(p).read_bytes()).hexdigest() != value]
    passed = not changed and all(row['status'] == 'PASS' for row in rows)
    result = dict(status='PASS' if passed else 'FAIL', rows=rows,
                  artifacts=hashes, changed_inputs=changed)
    with (out/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return not passed


if __name__ == '__main__':
    raise SystemExit(main())
