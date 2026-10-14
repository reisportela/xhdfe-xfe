"""Exact rational projection of the public binary32 CUDA fixture, by FE group."""
from fractions import Fraction as F
from pathlib import Path
import json, math
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--fixture",type=Path,default=Path(__file__).with_name("public_cuda_float_fixture.txt"))
parser.add_argument("--residual",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()

def solve(A,b):
    a=[list(row)+[v] for row,v in zip(A,b)]
    n=len(a)
    for j in range(n):
        k=next(k for k in range(j,n) if a[k][j])
        a[j],a[k]=a[k],a[j]
        pivot=a[j][j]
        a[j]=[v/pivot for v in a[j]]
        for k in range(n):
            if k!=j:
                c=a[k][j]
                a[k]=[v-c*w for v,w in zip(a[k],a[j])]
    return [row[-1] for row in a]

rows=[list(map(float,line.split())) for line in args.fixture.read_text().splitlines() if line.strip()]
observed=list(map(float,args.residual.read_text().splitlines()))
groups={}
for i,(y,z1,z2,g) in enumerate(rows):groups.setdefault(int(g),[]).append(i)
residual=[F(0) for _ in rows]
rank=0
for idx in groups.values():
    A=[[F(1),F(rows[i][1]),F(rows[i][2])] for i in idx]
    y=[F(rows[i][0]) for i in idx]
    if len(idx)<=3:
        G=[[sum(x*z for x,z in zip(a,b)) for b in A] for a in A]
        solve(G,y)  # prove full row rank; projection is exactly the identity
        rank+=len(idx)
    else:
        G=[[sum(a[j]*a[k] for a in A) for k in range(3)] for j in range(3)]
        h=[sum(a[j]*v for a,v in zip(A,y)) for j in range(3)]
        beta=solve(G,h)
        for i,a,v in zip(idx,A,y): residual[i]=v-sum(x*b for x,b in zip(a,beta))
        rank+=3
exact_rss=sum(v*v for v in residual)
observed_rss=math.fsum(v*v for v in observed)
error=[F(got)-ref for got,ref in zip(observed,residual)]
result={'n':len(rows),'exact_rank':rank,'oracle':'Fraction exact projection on represented binary32 inputs',
        'oracle_rss':float(exact_rss),'cuda_rss':observed_rss,
        'relative_rss_error':abs(observed_rss-float(exact_rss))/float(exact_rss),
        'max_residual_error':max(abs(float(x)) for x in error),
        'residual_error_norm':math.sqrt(float(sum(x*x for x in error))),
        'variance_contract_limit':1e-8}
result['variance_contract_pass']=result['relative_rss_error']<=result['variance_contract_limit']
result['conservative_rank_contract']=3*len(groups)
result['conservative_df_contract']=len(rows)-3*len(groups)
result['oracle_constant_variance']=float(exact_rss)/(result['conservative_df_contract']*len(rows))
with args.output.open('x') as dest: dest.write(json.dumps(result,indent=2)+'\n')
print(result)
