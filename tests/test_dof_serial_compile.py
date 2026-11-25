#!/usr/bin/env python3
"""Check that the diagnostic serial build still compiles the shared DoF path."""

import argparse
import os
from pathlib import Path
import subprocess


MIRRORS = (
    'src/hdfe_regressor_v11.cpp',
    'stata/src/hdfe_regressor_v11.cpp',
    'r/xhdfe/src/hdfe_regressor_v11.cpp',
    'share/xhdfe_estimation_cpp/src/hdfe_regressor_v11.cpp',
    'share/xhdfe_estimation_cpp/stata/src/hdfe_regressor_v11.cpp',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--eigen', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    eigen = args.eigen or root / 'r/xhdfe/src/eigen'
    reference = (root / MIRRORS[0]).read_bytes()
    if any((root / name).read_bytes() != reference for name in MIRRORS[1:]):
        raise RuntimeError('DoF source mirrors differ')
    flags = ['-std=c++17', '-fsyntax-only', '-DNDEBUG', '-DSYSTEM=STUNIX',
             '-I' + str(root / 'stata/include'), '-I' + str(root / 'stata/tools/_deps'),
             '-I' + str(eigen)]
    sources = [root / 'stata/src' / name for name in ('hdfe_regressor_v11.cpp', 'xhdfe_plugin.cpp', 'xfe_plugin.cpp')]
    subprocess.run([args.cxx, *flags, *map(str, sources)], check=True)


if __name__ == '__main__':
    main()
