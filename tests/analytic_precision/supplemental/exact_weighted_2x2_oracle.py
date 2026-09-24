"""Independent exact-rational WLS and HC1 oracle on the original64 rows."""
from fractions import Fraction as F
from pathlib import Path
import json


def inverse(matrix):
    n=len(matrix)
    work=[row[:]+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=next(row for row in range(column,n) if work[row][column])
        work[column],work[pivot]=work[pivot],work[column]
        scale=work[column][column]
        work[column]=[value/scale for value in work[column]]
        for row in range(n):
            if row==column:continue
            scale=work[row][column]
            work[row]=[a-scale*b for a,b in zip(work[row],work[column])]
    return [row[n:] for row in work]


def product(first,second):
    return [[sum(a*b for a,b in zip(row,column)) for column in zip(*second)] for row in first]


records=[]
for beta in (F(3,4),F.from_float(.7)):
    for exponent in (0,20,40,60,80,120):
        for shift in (-200,0,200):
            epsilon=F(2)**-exponent;common=F(2)**shift
            design=[];response=[];weights=[]
            for cell in range(4):
                worker,firm=cell//2,cell%2
                x=int(cell==1)
                for replicate in range(16):
                    design.append([F(1),F(worker),F(firm),F(x)])
                    response.append(beta*x+firm-worker+F(2*(replicate%2)-1,8))
                    weights.append(common*(1 if worker==firm else epsilon))
            gram=[[sum(w*row[j]*row[k] for w,row in zip(weights,design)) for k in range(4)] for j in range(4)]
            rhs=[[sum(w*row[j]*value for w,row,value in zip(weights,design,response))] for j in range(4)]
            bread=inverse(gram);estimate=[row[0] for row in product(bread,rhs)]
            residual=[value-sum(a*b for a,b in zip(row,estimate)) for row,value in zip(design,response)]
            assert estimate==[F(0),F(-1),F(1),beta]
            assert all(value*value==F(1,64) for value in residual)
            rss=sum(w*u*u for w,u in zip(weights,residual))
            classical=rss/F(60)*bread[3][3]
            meat=[[sum(w*w*u*u*row[j]*row[k] for w,u,row in zip(weights,residual,design))
                   for k in range(4)] for j in range(4)]
            robust=product(product(bread,meat),bread)[3][3]*F(64,60)
            assert classical==(1+epsilon)*(1+1/epsilon)/960
            assert robust==F(1,240)
            records.append(dict(beta=str(beta),weight_exponent=exponent,common_shift=shift,
                                exact_beta=str(estimate[3]),classical_variance=str(classical),robust_variance=str(robust)))
out=Path(__file__).resolve().parent/'exact_weighted_2x2_oracle.json'
with out.open('x') as handle:json.dump(dict(status='EXACT_RATIONAL_ORACLE_PASS',cases=len(records),records=records),handle,indent=2)
print(json.dumps(dict(status='EXACT_RATIONAL_ORACLE_PASS',cases=len(records))))
