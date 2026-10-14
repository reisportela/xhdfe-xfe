#!/usr/bin/env python3
"""Offline full-factor equality, frozen-arena comparison and a negative control.

The guard copies exact source bodies into new synthetic-only harness inputs.
It never patches sources or reads real-data graphs. No threshold is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MARKERS = ['bool mlsmr_trace_enabled() {',
           'bool mlsmr_env_enabled(const char* name, bool default_value) {',
           'struct MlsmrSchurEdge {', 'double mlsmr_uniform01(',
           'int mlsmr_sample_weighted_suffix(', 'struct MlsmrAcGraphEdge {',
           'struct MlsmrAcNeighbor {', 'struct MlsmrBucketOrdering {',
           'struct MlsmrApproxCholFactor {']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def body(text, marker):
    start = text.index(marker)
    first = text.index('{', start)
    depth = 0
    state = 'code'
    i = first
    while i < len(text):
        ch = text[i]
        nxt = text[i:i+2]
        if state == 'line':
            if ch == '\n': state = 'code'
        elif state == 'comment':
            if nxt == '*/': state = 'code'; i += 1
        elif state in ('string', 'char'):
            if ch == '\\': i += 1
            elif ch == ('"' if state == 'string' else "'"): state = 'code'
        elif nxt == '//': state = 'line'; i += 1
        elif nxt == '/*': state = 'comment'; i += 1
        elif ch == '"': state = 'string'
        elif ch == "'": state = 'char'
        elif ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                if marker.startswith('struct '):
                    while text[end].isspace(): end += 1
                    if text[end] != ';': raise RuntimeError('struct terminator missing')
                    end += 1
                return text[start:end]
        i += 1
    raise RuntimeError('unterminated source body')

def fragment(source):
    text = source.read_text()
    floor = re.search(r'^constexpr double kMlsmrApproxCholNearZero = [^;]+;$', text, re.M)
    if not floor: raise RuntimeError('unchanged floor declaration missing')
    parts = {marker: body(text, marker) for marker in MARKERS}
    includes = '''#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "hdfe/ieee_bits.hpp"
#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/audit_environment.hpp"
#include "ac_arena_storage.hpp"
#define XHDFE_SECTION_BEGIN() ((void)0)
#define XHDFE_SECTION(id) ((void)0)
namespace hdfe { namespace detail {
'''
    generated = includes + floor.group(0) + '\n' + '\n\n'.join(parts.values()) + '\n} }\n'
    reference = body(parts[MARKERS[-1]], 'static std::shared_ptr<MlsmrApproxCholFactor> build_reference(')
    return generated, hashlib.sha256(reference.encode()).hexdigest()

def identical_files(a, b):
    with a.open('rb') as x, b.open('rb') as y:
        while True:
            p, q = x.read(1 << 20), y.read(1 << 20)
            if p != q: return False
            if not p: return True

def run(command, folder, name, env, timeout=420):
    with (folder/(name+'.stdout')).open('x') as out, (folder/(name+'.stderr')).open('x') as err:
        result = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=timeout)
    return result.returncode, (folder/(name+'.stdout')).read_text()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--frozen-source', type=Path, required=True)
    ap.add_argument('--compiler', default='g++')
    ap.add_argument('--sanitizers', action='store_true')
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise RuntimeError('use a new output directory within this checkout')
    compiler = shutil.which(args.compiler)
    if not compiler: raise RuntimeError('compiler unavailable')
    frozen = args.frozen_source.resolve(strict=True)
    current = ROOT/'src/fe_absorption.cpp'
    current_fragment, current_ref = fragment(current)
    frozen_fragment, frozen_ref = fragment(frozen)
    if current_ref != frozen_ref: raise RuntimeError('reference builder was modified')
    pinned = [current, frozen, ROOT/'src/ac_arena_storage.hpp',
              ROOT/'include/hdfe/audit_environment.hpp', ROOT/'include/hdfe/deterministic_parallel.hpp',
              ROOT/'tests/ac_factor_memory_contract.cpp', Path(__file__).resolve()]
    before = {str(p): sha(p) for p in pinned}
    output.mkdir(parents=True)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-march=native', '-mtune=native',
             '-ffast-math', '-fno-finite-math-only', '-funroll-loops', '-ftree-vectorize',
             '-fomit-frame-pointer', '-fno-signed-zeros', '-fno-trapping-math',
             '-fassociative-math', '-freciprocal-math', '-fopenmp', '-DHDFE_USE_OPENMP']
    if 'clang' in Path(compiler).name:
        flags += ['-ffp-contract=off', '-DXHDFE_FP_CONTRACT_OFF=1']
    if args.sanitizers:
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                  '-fno-omit-frame-pointer']
    env = {k:v for k,v in os.environ.items() if k != 'LD_LIBRARY_PATH' and not k.startswith('XHDFE_')}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    if args.sanitizers: env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
    records = {}
    for arm, contents in [('frozen', frozen_fragment), ('candidate', current_fragment)]:
        folder = output/arm; folder.mkdir()
        generated = folder/'ac_factor_fragment.hpp'
        with generated.open('x') as f: f.write(contents)
        binary = folder/'factor_contract'
        command = [compiler, *flags, '-I'+str(folder), '-I'+str(ROOT/'include'),
                   '-I'+str(ROOT/'src'), '-I'+str(ROOT/'r/xhdfe/src/eigen'),
                   str(ROOT/'tests/ac_factor_memory_contract.cpp'), '-o', str(binary)]
        rc, _ = run(command, folder, 'build', env)
        if rc: raise RuntimeError(f'{arm} compilation failed: rc {rc}')
        archive = folder/'factors.bin'
        rc, text = run([str(binary), str(archive)], folder, 'positive', env)
        if rc or 'AC_FACTOR_MEMORY_CONTRACT_PASS cases=441' not in text:
            raise RuntimeError(f'{arm} positive factor contract failed: rc {rc}')
        records[arm] = dict(build_command=command, fragment_sha256=sha(generated),
                            binary_sha256=sha(binary), archive_sha256=sha(archive), cases=441)
    if not identical_files(output/'frozen/factors.bin', output/'candidate/factors.bin'):
        raise RuntimeError('candidate and frozen full-factor archives differ')
    folder = output/'candidate'
    rc, text = run([str(folder/'factor_contract'), str(folder/'negative.bin'), 'negative'],
                   folder, 'negative', env)
    if rc == 0 or 'AC_FACTOR_NEGATIVE_CONTROL_REJECTED' not in text:
        raise RuntimeError('factor corruption negative control was not rejected')
    after = {str(p):sha(p) for p in pinned}
    if after != before: raise RuntimeError('source custody changed during factor checks')
    report = dict(status='PASS', compiler=compiler, sanitizers=args.sanitizers,
                  positive_cases_per_arm=441, negative_rejected=1,
                  full_factor_archives_byte_identical=True, reference_body_sha256=current_ref,
                  source_hashes=before, arms=records,
                  scope='synthetic factor/ownership guard; not estimator or performance certification')
    with (output/'REPORT.json').open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['status','positive_cases_per_arm','negative_rejected',
                                          'full_factor_archives_byte_identical','sanitizers']}))

if __name__ == '__main__':
    main()
