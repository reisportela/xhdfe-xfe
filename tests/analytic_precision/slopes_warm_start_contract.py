"""Manual CPU warm-start probe; reuse the supplied core's compile flags and archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def compile_template(database):
    rows = [row for row in json.loads(database.read_text())
            if Path(row['file']).name == 'fe_absorption.cpp']
    if len(rows) != 1:
        raise ValueError('expected one fe_absorption.cpp entry in compile_commands')
    row = rows[0]
    cwd = Path(row['directory']).resolve()
    original = (cwd / row['file']).resolve()
    tokens = row.get('arguments') or shlex.split(row['command'])
    compiler = str((cwd / tokens[0]).resolve()) if '/' in tokens[0] else shutil.which(tokens[0])
    if not compiler:
        raise ValueError('C++ compiler not found')
    flags, inputs, outputs, index = [], 0, 0, 1
    while index < len(tokens):
        token = tokens[index]
        if token == '-o':
            outputs += 1
            index += 2
        elif token in ('-MF', '-MT', '-MQ'):
            index += 2
        elif token in ('-c', '-MD', '-MMD', '-MP') or token.startswith(('-MF', '-MT', '-MQ')):
            index += 1
        elif (cwd / token).resolve() == original:
            inputs += 1
            index += 1
        elif token in ('-I', '-isystem', '-iquote', '-idirafter', '-include', '-imacros'):
            flags.extend((token, str((cwd / tokens[index + 1]).resolve())))
            index += 2
        elif token.startswith('-I') and len(token) > 2:
            flags.append('-I' + str((cwd / token[2:]).resolve()))
            index += 1
        else:
            flags.append(token)
            index += 1
    if (inputs, outputs) != (1, 1):
        raise ValueError('unexpected compile input/output shape')
    if '-fopenmp' not in flags or '-DHDFE_USE_OPENMP' not in flags:
        raise ValueError('the probe requires a CPU OpenMP compile template')
    if any(flag.startswith('-DHDFE_USE_CUDA') for flag in flags):
        raise ValueError('supply CPU compile_commands and its matching archive')
    return [compiler, *flags], row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='absorption source with the private CPU seed helper')
    parser.add_argument('--archive', type=Path, required=True, help='matching CPU core archive; no whole-archive linking')
    parser.add_argument('--compile-commands', type=Path, required=True)
    parser.add_argument('--include-dir', type=Path, action='append', default=[], help='prepend a header directory; repeat if needed')
    parser.add_argument('--baseline-source', type=Path, help='optional C0 absorption source, using the same headers/archive')
    parser.add_argument('--out', type=Path, required=True, help='new directory for binaries, logs and receipt')
    args = parser.parse_args()
    source, archive, database, out = (path.resolve() for path in
        (args.source, args.archive, args.compile_commands, args.out))
    probe = Path(__file__).with_name('slopes_warm_start_probe.cpp').resolve()
    common, row = compile_template(database)
    common[1:1] = ['-I' + str(path.resolve()) for path in args.include_dir]
    sources = {'candidate': (source, 1)}
    if args.baseline_source is not None:
        sources['baseline'] = (args.baseline_source.resolve(), 0)
    paths = [Path(__file__).resolve(), probe, database, archive, *(path for path, _ in sources.values())]
    before = {str(path): sha(path) for path in paths}
    commands = {name: [*common, '-DXHDFE_SLOPE_PROBE_SOURCE=' + json.dumps(str(path)),
        '-DXHDFE_SLOPE_PROBE_HAS_SEED=' + str(seed), str(probe), str(archive),
        '-pthread', '-Wl,-Map,' + str(out / (name + '.map')), '-o', str(out / name)]
        for name, (path, seed) in sources.items()}
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / 'tmp'
    scratch.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith('XHDFE_')}
    env.update(XHDFE_GPU_BACKEND='cpu', XHDFE_CERTIFY='0', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
        CUDA_CACHE_PATH=str(scratch / 'cuda_cache'), XDG_CACHE_HOME=str(scratch / 'xdg_cache'))
    with (out / 'COMMANDS.json').open('x') as stream:
        json.dump(dict(commands=commands, compile_entry=row, sha256=before), stream, indent=2)
        stream.write('\n')
    for name, command in commands.items():
        with (out / (name + '.build.log')).open('x') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=420)
        with (out / (name + '.cold.txt')).open('x') as log:
            subprocess.run([str(out / name), '--cold'], cwd=out, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
    if args.baseline_source is not None:
        if (out / 'candidate.cold.txt').read_bytes() != (out / 'baseline.cold.txt').read_bytes():
            raise RuntimeError('excluded cold domain differs from C0')
    with (out / 'seed.log').open('x') as log:
        subprocess.run([str(out / 'candidate'), '--seed'], cwd=out, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=180)
    if before != {str(path): sha(path) for path in paths}:
        raise RuntimeError('an input changed during the probe')
    result = dict(status='PASS_CPU_WARM_START_CONTRACT', inputs_unchanged=True, sources=before,
        cold_comparison='PASS' if args.baseline_source is not None else 'NOT_REQUESTED',
        checks=['original_stopping_references', 'CPU_iterate_seed', 'cold_exclusions'],
        scope='Existing focused absorption probe only; no full estimator fit, CUDA execution or performance acceptance.')
    with (out / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
