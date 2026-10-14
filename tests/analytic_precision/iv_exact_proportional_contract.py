#!/usr/bin/env python3
"""Independent rational oracle for exact binary64 proportionality.

Compile iv_exact_proportional_contract.cpp with the production flags; pass
--probe PATH and --out NEW.json. Run with --threads 1,2,8,16,24,48 for OpenMP.
The estimator itself performs no Fraction or arbitrary-precision arithmetic.
"""
from fractions import Fraction
from pathlib import Path
import argparse, hashlib, json, math, random, struct, subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--threads',default='1')
args=parser.parse_args()
if args.out.exists(): raise FileExistsError(args.out)
probe=args.probe.resolve(strict=True)
threads=[int(t) for t in args.threads.split(',')]
rng=random.Random(71903)
def bits(x): return struct.unpack('>Q',struct.pack('>d',x))[0]
def value(b): return struct.unpack('>d',struct.pack('>Q',b))[0]
rows=[]
# Full binary64 exponent range, subnormals and signed zeros.
for i in range(3000):
    row=[value(rng.getrandbits(64)) for _ in range(4)]
    if all(math.isfinite(x) for x in row): rows.append(row)
for exponent in (-1074,-1070,-1022,-600,-200,0,200,600,1000):
    for a,b in ((3,7),(1,2),(-5,3),(1,-1)):
        for row_scale in (0,1,2,3,17,101):
            row=[math.ldexp(float(a),exponent),math.ldexp(float(b),exponent),
                 math.ldexp(float(a*row_scale),exponent),math.ldexp(float(b*row_scale),exponent)]
            if all(math.isfinite(x) for x in row):
                rows.append(row)
                if row[-1] != 0: rows.append(row[:3]+[math.nextafter(row[-1],math.inf)])
for _ in range(1000):
    a=rng.uniform(-10,10);c=rng.uniform(-10,10)
    e=rng.randrange(-400,401)
    rows.append([a,math.ldexp(a,e),c,math.ldexp(c,e)])
rows.extend([[0.,-0.,1.,2.],[0.,1.,0.,2.],[1.,0.,2.,0.],[-0.,0.,-0.,0.]])
expected=[]
for a,b,c,d in rows:
    nonzero_a=(a!=0 or c!=0); nonzero_b=(b!=0 or d!=0)
    expected.append(int(nonzero_a and nonzero_b and Fraction(a)*Fraction(d)==Fraction(b)*Fraction(c)))
data=''.join(' '.join(f'{bits(x):x}' for x in r)+'\n' for r in rows)
for team in threads:
    result=subprocess.run([str(probe),str(team)],input=data,text=True,capture_output=True,check=True)
    actual=list(map(int,result.stdout.splitlines()))
    assert actual==expected,[(i,a,b) for i,(a,b) in enumerate(zip(actual,expected)) if a!=b][:10]
report={'status':'PASS','rational_comparisons_per_team':len(rows),'exact_proportional':sum(expected),
        'retained_distinct_or_zero_columns':len(rows)-sum(expected),'late_row_one_ulp_guard':'PASS',
        'observed_teams':threads,'probe':str(probe),'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest()}
with args.out.open('x') as stream:
    json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report,indent=2))
