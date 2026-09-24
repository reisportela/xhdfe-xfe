"""Check the compiled two-double operations against exact rational arithmetic."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(exist_ok=False)
build=json.loads(args.build.read_text())
record=next(r for r in build['records'] if Path(r['source']).name=='ols.cpp')
source=Path(record['source']).read_text()
start=source.index('struct OlsWide {')
number=source[start:source.index('\n};',start)+3]
header=Path(record['source']).with_name('group_forward_certificate.hpp').read_text()
exact=header[header.index('class GroupExactDyadicSum {'):header.index('\nstruct GroupForwardSum')]
program='''#include "hdfe/ieee_bits.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <xmmintrin.h>
using hdfe::detail::ieee_finite;
'''+number+'\n'+exact+r'''
int main() {
    const unsigned csr=_mm_getcsr();
    for(bool flush:{false,true}) {
        _mm_setcsr(flush ? csr|0x8040 : csr&~0x8040U);
        for(std::uint64_t bits:{1ULL,0xfffffffffffffULL,0x10000000000000ULL}) {
            double value;std::memcpy(&value,&bits,8);
            GroupExactDyadicSum sum;sum.add(value);
            if(sum.exactly_zero()) return 2;
            sum.add(value,1,true);if(!sum.exactly_zero()) return 3;
        }
    }
    _mm_setcsr(csr);
    char op;std::uint64_t words[4];
    while(std::cin>>op>>std::hex>>words[0]>>words[1]>>words[2]>>words[3]) {
        OlsWide a,b,r;std::memcpy(&a.hi,&words[0],8);std::memcpy(&a.lo,&words[1],8);
        std::memcpy(&b.hi,&words[2],8);std::memcpy(&b.lo,&words[3],8);
        if(op=='+')r=a+b;else if(op=='-')r=a-b;else if(op=='*')r=a*b;else r=a/b;
        std::memcpy(&words[0],&r.hi,8);std::memcpy(&words[1],&r.lo,8);
        std::cout<<std::hex<<words[0]<<' '<<words[1]<<'\n';
    }
}
'''
cpp=args.out/'arithmetic.cpp';cpp.write_text(program)
command=record['command'][:record['command'].index('-c')]+[str(cpp),'-o',str(args.out/'arithmetic')]
run=subprocess.run(command,capture_output=True,text=True,timeout=120)
(args.out/'compile.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
rng=random.Random(472026)
cases=[]
for iteration in range(1200):
    exponent=rng.randint(-300,300)
    a=(rng.uniform(-2,2)*2.**exponent,rng.uniform(-1,1)*2.**(exponent-54))
    b=(rng.uniform(.5,2)*2.**rng.randint(-300,300),0.)
    if iteration%3==0:b=(-a[0],rng.choice((-1.,1.))*2.**(exponent-60))
    for op in '+-*/':cases.append((op,a,b))
def word(x):return struct.unpack('>Q',struct.pack('>d',x))[0]
data=''.join(f'{op} '+' '.join(f'{word(x):x}' for x in (*a,*b))+'\n' for op,a,b in cases)
run=subprocess.run([str(args.out/'arithmetic')],input=data,capture_output=True,text=True,timeout=60)
assert run.returncode==0,run.stderr
outputs=run.stdout.splitlines();assert len(outputs)==len(cases)
worst=0.
for (op,a,b),output in zip(cases,outputs):
    A=sum(map(Fraction,a));B=sum(map(Fraction,b))
    gold={'+':lambda:A+B,'-':lambda:A-B,'*':lambda:A*B,'/':lambda:A/B}[op]()
    got=sum(Fraction(struct.unpack('>d',struct.pack('>Q',int(w,16)))[0]) for w in output.split())
    scale=abs(gold) if gold else max(abs(A),abs(B))
    error=abs(got-gold)/scale if scale else abs(got-gold)
    assert error<Fraction(1,2**98),(op,a,b,float(error))
    worst=max(worst,float(error))
receipt=dict(status='PASS',rational_cases=len(cases),integer_zero_cases=12,
             max_relative_error=worst,limit=2.**-98,command=command,
             source_sha256=hashlib.sha256(source.encode()).hexdigest(),
             arithmetic_sha256=hashlib.sha256(number.encode()).hexdigest(),
             scope='Normal-range two-double arithmetic; exact zero also tested with FTZ/DAZ on and off')
(args.out/'report.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
