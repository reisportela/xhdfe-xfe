"""Record the exact plugin bytes, native SPI gate and loaded OpenMP runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--kind', choices=['xhdfe', 'xfepout'], required=True)
    parser.add_argument('--arch', choices=['arm64', 'x86_64'], required=True)
    parser.add_argument('--stage', choices=['thin', 'final'], required=True)
    parser.add_argument('--runtime-path', type=Path)
    parser.add_argument('--expect-serial', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    plugin = args.plugin.resolve(strict=True)
    expected_runtime = args.runtime_path.resolve(strict=True) if args.runtime_path else None
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='xhdfe-native-openmp-receipt-v1', status='FAIL',
        plugin_kind=args.kind, plugin_path=str(plugin), plugin_sha256=sha(plugin),
        probe_path=str(probe), probe_sha256=sha(probe),
        expected_arch=args.arch, stage=args.stage, expect_serial=args.expect_serial,
        expected_runtime_path=str(expected_runtime) if expected_runtime else None,
        expected_runtime_sha256=sha(expected_runtime) if expected_runtime else None,
        controller_platform=sys.platform, controller_arch=platform.machine(),
        runner_sha256=sha(Path(__file__)), timeout_seconds=120,
        release_artifact_gate_pass=False, new_estimator_api=False)
    command = [str(probe), args.kind, str(plugin), args.arch]
    if args.expect_serial:
        command.append('--expect-serial')
    receipt['command'] = command
    env = {key: value for key, value in os.environ.items() if not key.startswith('XHDFE_')}
    for name in ('LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'LD_LIBRARY_PATH',
                 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'):
        env.pop(name, None)
    env.update(XHDFE_GPU_BACKEND='cpu', XHDFE_CERTIFY='0',
               XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off',
               XHDFE_FE_STRUCTURE_MODE='off', OMP_DYNAMIC='FALSE',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        run = subprocess.run(command, cwd=output, env=env, text=True,
                             capture_output=True, timeout=120, check=False)
        (output / 'stdout.log').write_text(run.stdout)
        (output / 'stderr.log').write_text(run.stderr)
        receipt['returncode'] = run.returncode
        # One exact JSON receipt; a truncated, extra, or stale success line fails.
        native = json.loads(run.stdout)
        receipt['native'] = native
        assert run.returncode == 0, native.get('reason', 'native probe failed')
        assert native['plugin_kind'] == args.kind
        assert Path(native['plugin_path']).resolve(strict=True) == plugin
        assert native['process_arch'] == args.arch and native['translated'] == 0
        assert sha(plugin) == receipt['plugin_sha256'], 'plugin changed during execution'
        assert sha(probe) == receipt['probe_sha256'], 'probe changed during execution'
        if args.expect_serial:
            assert native['status'] == 'EXPECTED_SERIAL_REJECTION'
            assert native['serial_negative_pass'] and not native['release_gate_pass']
            receipt['status'] = 'EXPECTED_SERIAL_REJECTION'
        else:
            assert native['status'] == 'PASS' and native['release_gate_pass']
            runtime = Path(native['openmp_runtime_path']).resolve(strict=True)
            receipt['loaded_runtime_path'] = str(runtime)
            receipt['loaded_runtime_sha256'] = sha(runtime)
            if sys.platform == 'darwin':
                assert expected_runtime is not None, 'macOS requires the exact expected runtime path'
                if args.stage == 'final':
                    assert expected_runtime.name == 'xhdfe_libomp.dylib'
                    assert expected_runtime.parent == plugin.parent, 'runtime must be beside installed x plugins'
            if expected_runtime is not None:
                assert runtime == expected_runtime, 'OpenMP resolved outside the expected runtime artifact'
                assert sha(expected_runtime) == receipt['expected_runtime_sha256']
            receipt['runtime_identity_pass'] = True
            receipt['status'] = 'PASS'
            receipt['release_artifact_gate_pass'] = args.stage == 'final'
    except subprocess.TimeoutExpired as error:
        receipt.update(reason='Native probe timed out', timed_out=True)
        if error.stdout:
            (output / 'stdout.log').write_bytes(error.stdout if isinstance(error.stdout, bytes) else error.stdout.encode())
        if error.stderr:
            (output / 'stderr.log').write_bytes(error.stderr if isinstance(error.stderr, bytes) else error.stderr.encode())
    except Exception as error:
        receipt['reason'] = repr(error)
    receipt['plugin_unchanged'] = sha(plugin) == receipt['plugin_sha256']
    receipt['scope'] = ('Exact native plugin SPI/workers/numerical fixture only. '
        'Does not certify Stata ado parsing, all numerical features, performance, or publication.')
    with (output / 'RESULT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], result=str(output / 'RESULT.json'),
                         plugin_sha256=receipt['plugin_sha256'])))
    return 0 if receipt['status'] in ('PASS', 'EXPECTED_SERIAL_REJECTION') else 1


if __name__ == '__main__':
    raise SystemExit(main())
