"""Exact rational oracle for binary64 product roundoff, including its range."""
import argparse
from fractions import Fraction
import math
import random
import struct
import subprocess


def bits(value):
    return struct.unpack('Q', struct.pack('d', value))[0]


def number(value):
    return struct.unpack('d', struct.pack('Q', value))[0]


p = argparse.ArgumentParser()
p.add_argument('--probe', required=True)
a = p.parse_args()
rng = random.Random(14102026)
cases = [(float.fromhex('0x1.cdeb85d1b930fp+0'), float.fromhex('0x1.e02b32d246cd8p+0'))]
edge = [0., -0., math.ldexp(1.,-1074), -math.ldexp(1.,-1074),
        float.fromhex('0x0.fffffffffffffp-1022'), math.ldexp(1.,-1022),
        math.ldexp(1.,-500), 1., -1., math.nextafter(1.,2.),
        math.ldexp(1.,500), float.fromhex('0x1.fffffffffffffp+1023')]
cases += [(x,y) for x in edge for y in edge if math.isfinite(x*y)]
for i in range(40000):
    if i < 20000:
        x = number(0x3ff0000000000000 | rng.getrandbits(52))
        y = number(0x3ff0000000000000 | rng.getrandbits(52))
    else:
        x = number(rng.randrange(2047)<<52 | rng.getrandbits(52) | rng.getrandbits(1)<<63)
        y = number(rng.randrange(2047)<<52 | rng.getrandbits(52) | rng.getrandbits(1)<<63)
    if math.isfinite(x*y):
        cases.append((x,y))
payload = ''.join(f'{bits(x):x} {bits(y):x}\n' for x,y in cases)
run = subprocess.run([a.probe],input=payload,text=True,capture_output=True,check=True,timeout=120)
rows = run.stdout.splitlines()
assert len(rows)==len(cases)
for row,(x,y) in zip(rows,cases):
    actual_p, actual_error = [number(int(v,16)) for v in row.split()]
    rounded = x*y
    expected = float(Fraction(x)*Fraction(y)-Fraction(rounded))
    assert actual_p==rounded and actual_error==expected, dict(
        x=x.hex(),y=y.hex(),p=actual_p.hex(),expected_p=rounded.hex(),
        error=actual_error.hex(),expected_error=expected.hex())
print(f'WIDE_PRODUCT_RATIONAL_CONTRACT_PASS {len(cases)}')
