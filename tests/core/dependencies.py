"""Copy the installed Stata test dependencies into a fresh isolated tree.

No network, package installation, HOME override or write to the source PLUS.
The runner uses only this copy, never the user's PLUS, on its child adopath.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PACKAGES = ('ftools', 'require', 'reghdfe', 'moremata', 'ivreg2', 'ranktest',
            'ivreghdfe', 'estout')
CORE = {'ftools', 'require', 'reghdfe'}


def prepare(plus, output):
    plus = plus.resolve()
    entries = {}
    current, files = None, []
    for line in (plus / 'stata.trk').read_text(errors='strict').splitlines():
        if line.startswith('N '):
            current, files = line[2:].strip().removesuffix('.pkg'), []
        elif line.startswith('f ') and current:
            files.append(line[2:].strip())
        elif line == 'e' and current:
            entries[current] = list(files)
            current = None
    missing = [p for p in PACKAGES if not entries.get(p)]
    if missing:
        raise RuntimeError('Required packages are not installed: ' + ', '.join(missing))
    selected = []
    for package in PACKAGES:
        for relative in entries[package]:
            path = (plus / relative).resolve()
            if not path.is_relative_to(plus) or not path.is_file():
                raise RuntimeError('Missing/unsafe dependency file: ' + relative)
            selected.append((package, relative, path))
    # ftools generates this library locally; some installation manifests omit it.
    compiled = plus / 'l/lftools.mlib'
    if compiled.is_file() and not any(p == compiled for _, _, p in selected):
        selected.append(('ftools', 'l/lftools.mlib', compiled))
    output.mkdir(parents=True, exist_ok=False)
    for directory in ['ado/plus', 'ado/personal', 'aux']:
        (output / directory).mkdir(parents=True)
    rows = []
    seen = {}
    for package, relative, source in selected:
        destination = output / ('ado/plus' if package in CORE else 'aux') / relative
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        if destination in seen:
            assert seen[destination] == original, 'Conflicting installed dependency bytes'
            continue
        seen[destination] = original
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == original
        rows.append(dict(package=package, source=str(source),
                         copied=str(destination.relative_to(output)), sha256=original))
    with (output / 'SOURCE.json').open('x') as stream:
        json.dump(dict(source_plus=str(plus), no_install=True, files=rows), stream, indent=2)
        stream.write('\n')
    return dict(status='PASS', packages=len(PACKAGES), files=len(rows), output=str(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.plus, a.output.resolve())))
