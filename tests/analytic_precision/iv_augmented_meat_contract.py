#!/usr/bin/env python3
"""Independent positive/indefinite/singular controls for the actual wide meat helper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir()
    source = (ROOT/'src/ols.cpp').read_text()
    start = source.index('bool wide_augmented_meat_positive_definite(')
    end = source.index('\nOlsResult run_ols_wide(', start)
    helper = source[start:end]
    code = '''#include "wide_float.hpp"
#include <vector>
#include <iostream>
#include <limits>
using hdfe::detail::OlsWide;
'''+helper+'''
int main() {
    int cases; if (!(std::cin>>cases)) return 2;
    for (int test=0;test<cases;++test) {
        int p; std::cin>>p;
        std::vector<OlsWide> meat(p*p),cross(p); OlsWide total;
        for (int i=0;i<=p;++i) for (int j=0;j<=p;++j) {
            double high,low; std::cin>>high>>low;
            OlsWide value=OlsWide::pair(high,low);
            if (i<p && j<p) meat[i*p+j]=value;
            if (i==p && j<p) cross[j]=value;
            if (i==p && j==p) total=value;
        }
        if (!std::cin) return 2;
        std::cout<<wide_augmented_meat_positive_definite(meat,cross,total,p)<<'\\n';
    }
    const std::vector<OlsWide> meat{OlsWide(1)},cross{OlsWide(0)};
    for (double value: {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        if (wide_augmented_meat_positive_definite(meat,cross,OlsWide(value),1)) return 3;
}
'''
    (out/'probe.cpp').write_text(code)
    command = ['g++', '-std=c++17', '-O3', '-march=native', '-DNDEBUG',
               '-I', str(ROOT/'src'), '-I', str(ROOT/'include'),
               '-I', str(ROOT/'r/xhdfe/src/eigen'), str(out/'probe.cpp'), '-o', str(out/'probe')]
    build = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (out/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode == 0
    with mp.workdps(120):
        tiny = mp.mpf(2)**-80
        cases = [
            ('positive', mp.matrix([[2, 0, 1], [0, 3, 1], [1, 1, 4]])),
            ('positive_low_part', mp.matrix([[1, 1, 0], [1, 1+tiny, 0], [0, 0, 1]])),
            ('negative_low_part', mp.matrix([[1, 1, 0], [1, 1-tiny, 0], [0, 0, 1]])),
            ('singular', mp.matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])),
            ('zero', mp.zeros(3)),
            ('negative_slope', mp.diag([-1, 2, 3])),
            ('negative_constant', mp.diag([1, 2, -3])),
            ('positive_slopes_indefinite_full', mp.matrix([[2, 0, 2], [0, 3, 6], [2, 6, 1]])),
        ]
        for exponent in (-40, 40):
            scale = mp.diag([mp.mpf(2)**exponent, mp.mpf(2)**-exponent, mp.mpf(2)**(exponent//2)])
            for name, matrix in cases[:4]:
                cases.append((f'{name}_scaled_{exponent}', scale*matrix*scale))
        records = []
        lines = [str(len(cases))]
        for name, matrix in cases:
            try:
                mp.cholesky(matrix)
                positive = True
            except ValueError:
                positive = False
            records.append(dict(name=name, positive=positive))
            lines.append(str(matrix.rows-1))
            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    high = float(matrix[i,j])
                    low = float(matrix[i,j]-mp.mpf(high))
                    assert mp.mpf(high)+mp.mpf(low) == matrix[i,j]
                    lines.append(f'{high:.17g} {low:.17g}')
    run = subprocess.run([str(out/'probe')], input='\n'.join(lines)+'\n',
                         capture_output=True, text=True, timeout=60)
    (out/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0
    actual = run.stdout.splitlines()
    assert len(actual) == len(records)
    for row, value in zip(records, actual):
        row['status'] = 'PASS' if (value == '1') == row['positive'] else 'FAIL'
    passed = all(row['status'] == 'PASS' for row in records)
    report = dict(status='PASS' if passed else 'FAIL', cases=len(records), results=records,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),
                  nonfinite_rejected=True, command=command,
                  scope='Actual helper, exact represented inputs, MP120 Cholesky; strict positivity without a spectral floor')
    with (out/'receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print('IV_AUGMENTED_MEAT', report['status'], len(records), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
