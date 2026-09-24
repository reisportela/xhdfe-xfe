"""Inspect Mach-O dependencies and bind native OpenMP receipts to release bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct


ARCHES = {0x01000007: 'x86_64', 0x0100000c: 'arm64'}
MINIMUM = {'x86_64': (10, 12, 0), 'arm64': (11, 0, 0)}
RUNTIME = 'xhdfe_libomp.dylib'
LICENSE = 'LLVM-OpenMP-LICENSE.txt'
MANIFEST = 'macos-openmp-manifest.json'
LLVM_REVISION = '87f0227cb60147a26a1eeb4fb06e3b505e9c7261'
DYLIB_LOADS = {0xc, 0x80000018, 0x8000001f, 0x80000023}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Missing/non-regular file: {path}')
    return path


def command_string(data, command, offset):
    require(0 <= offset < len(command), 'Invalid Mach-O string offset')
    stop = command.find(b'\0', offset)
    require(stop >= 0, 'Unterminated Mach-O load-command string')
    return command[offset:stop].decode('utf-8')


def thin_info(data):
    require(len(data) >= 32 and data[:4] == b'\xcf\xfa\xed\xfe', 'Expected a little-endian 64-bit Mach-O slice')
    _, cpu, _, filetype, count, size, _, _ = struct.unpack_from('<8I', data)
    require(cpu in ARCHES, 'Unsupported Mach-O CPU type')
    require(32 + size <= len(data) and count <= size // 8, 'Truncated Mach-O load commands')
    position = 32
    dependencies, rpaths, identities, versions = [], [], [], []
    for _ in range(count):
        require(position + 8 <= 32 + size, 'Missing Mach-O command')
        cmd, length = struct.unpack_from('<II', data, position)
        require(length >= 8 and position + length <= 32 + size, 'Invalid Mach-O command size')
        block = data[position:position+length]
        if cmd in DYLIB_LOADS or cmd in (0xd, 0x8000001c):
            require(length >= 12, 'Truncated Mach-O path command')
            value = command_string(data, block, struct.unpack_from('<I', block, 8)[0])
            if cmd == 0xd: identities.append(value)
            elif cmd == 0x8000001c: rpaths.append(value)
            else: dependencies.append(value)
        elif cmd in (0x24, 0x32):
            require(length >= (16 if cmd == 0x24 else 24), 'Truncated macOS version command')
            if cmd == 0x32:
                require(struct.unpack_from('<I', block, 8)[0] == 1, 'Non-macOS build platform')
            version = struct.unpack_from('<I', block, 8 if cmd == 0x24 else 12)[0]
            versions.append((version >> 16, (version >> 8) & 255, version & 255))
        position += length
    require(position == 32 + size and len(versions) == 1, 'Missing/ambiguous macOS deployment target')
    arch = ARCHES[cpu]
    require(versions[0] <= MINIMUM[arch], f'{arch}: minimum macOS version was raised to {versions[0]}')
    return dict(arch=arch, filetype=filetype, dependencies=dependencies, rpaths=rpaths,
                identities=identities, minimum_macos=list(versions[0]))


def macho(path):
    data = regular(path).read_bytes()
    magic = data[:4]
    formats = {b'\xca\xfe\xba\xbe': ('>', False), b'\xbe\xba\xfe\xca': ('<', False),
               b'\xca\xfe\xba\xbf': ('>', True), b'\xbf\xba\xfe\xca': ('<', True)}
    if magic not in formats:
        record = thin_info(data)
        return {record['arch']: record}
    order, wide = formats[magic]
    require(len(data) >= 8, 'Truncated fat Mach-O header')
    count = struct.unpack_from(order+'I', data, 4)[0]
    width = 32 if wide else 20
    require(1 <= count <= 2 and 8 + count*width <= len(data), 'Invalid fat Mach-O architecture table')
    result, intervals = {}, []
    for index in range(count):
        fields = struct.unpack_from(order+('IIQQII' if wide else 'IIIII'), data, 8+index*width)
        cpu, _, offset, size, alignment = fields[:5]
        require(alignment < 32 and offset % (1 << alignment) == 0, 'Invalid Mach-O slice alignment')
        require(offset >= 8+count*width and size >= 32 and offset+size <= len(data), 'Invalid Mach-O slice bounds')
        require(all(offset+size <= a or offset >= b for a, b in intervals), 'Overlapping Mach-O slices')
        intervals.append((offset, offset+size))
        record = thin_info(data[offset:offset+size])
        require(ARCHES.get(cpu) == record['arch'] and record['arch'] not in result, 'Duplicate/mismatched Mach-O architecture')
        result[record['arch']] = record
    return result


def inspect(path, kind, arch=None):
    records = macho(path)
    if arch:
        require(arch in records, f'Missing {arch} slice: {path}')
    else:
        require(set(records) == set(MINIMUM), f'Universal artifact must contain both architectures: {path}')
    for record in records.values():
        external = [item for item in record['dependencies']
                    if not item.startswith(('/usr/lib/', '/System/Library/'))]
        if kind == 'plugin':
            require(record['filetype'] == 8, 'Stata plugin must be a Mach-O bundle')
            require(external == ['@rpath/'+RUNTIME], 'Plugin must load only its packaged OpenMP runtime')
            require(record['rpaths'] == ['@loader_path'], 'Plugin runtime search path must be @loader_path')
        else:
            require(record['filetype'] == 6, 'OpenMP runtime must be a Mach-O dylib')
            require(record['identities'] == ['@rpath/'+RUNTIME], 'OpenMP install name is not portable')
            require(not external, 'OpenMP runtime has an unpackaged dependency')
            require(all(value == '@loader_path' for value in record['rpaths']), 'Runtime contains a build-machine rpath')
    return records


def files_and_inspection(xhdfe, xfepout, runtime_dir):
    runtime_dir = Path(runtime_dir)
    paths = {'xhdfe.plugin': regular(xhdfe), 'xfepout.plugin': regular(xfepout),
             RUNTIME: regular(runtime_dir/RUNTIME), LICENSE: regular(runtime_dir/LICENSE)}
    require(paths[LICENSE].stat().st_size > 0, 'OpenMP license is empty')
    inspection = {name: inspect(path, 'runtime' if name == RUNTIME else 'plugin')
                  for name, path in paths.items() if name != LICENSE}
    return {name: sha(path) for name, path in paths.items()}, inspection


def native_receipts(receipts, files):
    positive, negative = {}, {}
    for record in receipts:
        require(record.get('schema') == 'xhdfe-native-openmp-receipt-v1', 'Unknown native receipt')
        arch, kind = record.get('expected_arch'), record.get('plugin_kind')
        key = (arch, kind)
        require(arch in MINIMUM and kind in ('xhdfe', 'xfepout'), 'Invalid native receipt target')
        require(record.get('controller_platform') == 'darwin' and record.get('controller_arch') == arch,
                'Native receipt was not produced on the advertised macOS architecture')
        native = record.get('native', {})
        require(native.get('process_arch') == arch and native.get('translated') == 0, 'Native probe was translated or mismatched')
        require(record.get('plugin_unchanged') is True, 'Native test plugin changed during execution')
        if record.get('expect_serial') is True:
            require(record.get('status') == 'EXPECTED_SERIAL_REJECTION' and native.get('serial_negative_pass') is True,
                    'Serial negative control did not reject the artifact')
            require(native.get('thread1', {}).get('scalars', {}).get('probe_openmp') == 0,
                    'Serial control was built with OpenMP')
            require(key not in negative, 'Duplicate serial negative receipt')
            negative[key] = record
            continue
        require(key not in positive, 'Duplicate positive native receipt')
        require(record.get('status') == 'PASS' and record.get('release_artifact_gate_pass') is True and
                record.get('stage') == 'final' and native.get('release_gate_pass') is True,
                'Final universal artifact lacks a successful native test')
        require(record.get('plugin_sha256') == files[kind+'.plugin'], 'Native plugin receipt is for different bytes')
        require(record.get('loaded_runtime_sha256') == files[RUNTIME] and record.get('runtime_identity_pass') is True,
                'Native test loaded a different OpenMP runtime')
        plugin_path = PurePosixPath(record['plugin_path'])
        runtime_path = PurePosixPath(record['loaded_runtime_path'])
        require(runtime_path.name == RUNTIME and runtime_path.parent == plugin_path.parent,
                'Native runtime was not loaded beside the installed plugin')
        for count in (1, 2):
            snapshot = native.get('thread'+str(count), {})
            require(snapshot.get('request') == count and snapshot.get('rc') == 0 and
                    snapshot.get('numeric_ok') is True and snapshot.get('inputs_unchanged') is True,
                    'Native estimator fixture did not pass')
            scalars = snapshot.get('scalars', {})
            require(all(scalars.get('probe_'+field) == count
                        for field in ('requested', 'effective', 'used', 'workers')),
                    'Requested OpenMP workers were not observed in estimator work')
            require(scalars.get('probe_openmp') == 1 and scalars.get('probe_converged') == 1 and
                    scalars.get('probe_gpu') == 0, 'Native fixture backend/convergence mismatch')
        require(native.get('full_numeric_parity_max_error', float('inf')) <= 1e-10,
                'Native thread1/thread2 fixture disagreement')
        positive[key] = record
    targets = {(arch, kind) for arch in MINIMUM for kind in ('xhdfe', 'xfepout')}
    require(set(positive) == set(negative) == targets, 'Both architectures and both plugins need native and serial controls')


def verify(xhdfe, xfepout, runtime_dir):
    runtime_dir = Path(runtime_dir)
    record = json.loads(regular(runtime_dir/MANIFEST).read_text())
    files, inspection = files_and_inspection(xhdfe, xfepout, runtime_dir)
    require(record.get('schema') == 'xhdfe-macos-openmp-v1' and record.get('status') == 'NATIVE_VALIDATED',
            'macOS artifacts have not passed native release validation')
    require(record.get('files') == files and record.get('inspection') == inspection, 'macOS manifest does not match artifact bytes')
    require(record.get('llvm_revision') == LLVM_REVISION, 'Unexpected OpenMP source revision')
    native_receipts(record.get('native_receipts', []), files)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('inspect')
    one.add_argument('--file', type=Path, required=True)
    one.add_argument('--arch', choices=list(MINIMUM), required=True)
    one.add_argument('--kind', choices=['plugin', 'runtime'], required=True)
    for name in ['manifest', 'verify']:
        command = commands.add_parser(name)
        command.add_argument('--xhdfe', type=Path, required=True)
        command.add_argument('--xfepout', type=Path, required=True)
        command.add_argument('--runtime-dir', type=Path, required=True)
        if name == 'manifest':
            command.add_argument('--source-archive', type=Path, required=True)
            command.add_argument('--native-receipt', type=Path, action='append', required=True)
            command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inspect':
        result = inspect(args.file, args.kind, args.arch)
        print(json.dumps(dict(status='PASS', scope='Mach-O dependencies and minimum OS only', slices=result)))
    elif args.command == 'verify':
        verify(args.xhdfe, args.xfepout, args.runtime_dir)
        print('MACOS_OPENMP_PACKAGE_PASS')
    else:
        files, inspection = files_and_inspection(args.xhdfe, args.xfepout, args.runtime_dir)
        records = [json.loads(regular(path).read_text()) for path in args.native_receipt]
        native_receipts(records, files)
        result = dict(schema='xhdfe-macos-openmp-v1', status='NATIVE_VALIDATED', files=files,
                      inspection=inspection, llvm_revision=LLVM_REVISION,
                      source_archive_sha256=sha(regular(args.source_archive)), native_receipts=records,
                      scope='Native SPI/worker and analytic fixture checks; no full Stata ado or performance claim')
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        print('MACOS_OPENMP_MANIFEST_PASS')


if __name__ == '__main__':
    main()
