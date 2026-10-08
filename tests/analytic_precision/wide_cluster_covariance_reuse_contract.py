"""Offline high-precision reference and baseline parity for storage reuse cases."""
import argparse, json, math
from pathlib import Path
import mpmath as mp

def reference(case, digits):
    with mp.workdps(digits):
        n,p=512,case['p']
        scale=mp.mpf(2)**case['exponent']
        X=[]; weights=[]; residual=[]; labels=[]
        for i in range(n):
            walsh=lambda bit: 2*((i>>bit)&1)-1
            row=[]; variable=0
            for j in range(p):
                if j==case['constant']: value=mp.mpf(1)
                else:
                    if variable==0: value=mp.mpf(2*walsh(0)+(4 if case['constant']>=0 else 0))
                    elif variable==1: value=2*walsh(0)+mp.mpf(2)**-27*walsh(1)-(2 if case['constant']>=0 else 0)
                    else: value=mp.mpf(walsh(variable)+(variable if case['constant']>=0 else 0))
                    variable+=1
                row.append(value)
            X.append(row)
            weights.append(mp.mpf(1+i//64 if case['weighted'] else 1))
            residual.append(scale*walsh(5))
            labels.append(((i*37+i//7)%19-9,(i*13+i//11)%23-11))
        sw=sum(weights)
        bread=mp.inverse(mp.matrix([[mp.fsum(weights[i]*X[i][j]*X[i][k] for i in range(n))
                                     for k in range(p)] for j in range(p)]))
        means=mp.matrix([mp.mpf('-.5') if j%2 else mp.mpf('.25') for j in range(p)])
        # Independent augmented covariance through small-matrix congruence.
        meat=mp.zeros(p+1); counts=[]
        for dims,sign in [((0,),1),((1,),1),((0,1),-1)]:
            groups={}
            for i in range(n):
                key=tuple(labels[i][d] for d in dims)
                score=groups.setdefault(key,[mp.mpf(0)]*(p+1))
                for j in range(p): score[j]+=weights[i]*residual[i]*X[i][j]
                score[p]+=weights[i]*residual[i]
            count=len(groups); counts.append(count)
            factor=mp.mpf(sign)
            if case['conventional']: factor*=mp.mpf(count)/(count-1)
            meat+=factor*mp.matrix([[mp.fsum(row[j]*row[k] for row in groups.values())
                                    for k in range(p+1)] for j in range(p+1)])
        transform=mp.zeros(p+1)
        transform[:p,:p]=bread
        transform[p,:p]=-means.T*bread
        transform[p,p]=1/sw
        adjustment=mp.mpf(n-1)/(n-p)
        if not case['conventional']: adjustment*=mp.mpf(min(counts[:2]))/(min(counts[:2])-1)
        return adjustment*transform*meat*transform.T, sw*scale**2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=[json.loads(s) for s in args.baseline.read_text().splitlines()]
    cand=[json.loads(s) for s in args.candidate.read_text().splitlines()]
    assert len(base)==len(cand)==84
    rows=[]
    for old,new in zip(base,cand):
        ref,rss=reference(new,90); high,_=reference(new,120)
        with mp.workdps(120):
            p=new['p']; denom=[mp.sqrt(abs(high[j,j])) for j in range(p+1)]
            assert all(denom)
            stability=max(abs(ref[j,k]-high[j,k])/(denom[j]*denom[k])
                          for j in range(p+1) for k in range(p+1))
            error=max(abs(mp.mpf(new['output_covariance'][j][k])-high[j,k])/(denom[j]*denom[k])
                      for j in range(p+1) for k in range(p+1))
            beta=[mp.mpf(j+1)/8*mp.mpf(2)**new['exponent'] for j in range(p)]
            beta_error=max(abs(mp.mpf(new['coef'][j])-beta[j])/max(1,abs(beta[j])) for j in range(p))
            rss_error=abs(mp.mpf(new['rss'])/rss-1)
        checks=dict(reference=stability<mp.mpf('1e-60'),covariance=error<=mp.mpf('1e-8'),
                    coefficients=beta_error<=mp.mpf('1e-9'),rss=rss_error<=mp.mpf('1e-9'),
                    exact_baseline_parity=old==new,
                    output_matches_retained=new['covariance']==new['output_covariance'])
        rows.append(dict(id=new['id'],p=p,constant=new['constant'],checks=checks,
                         covariance_scaled_error=float(error),reference_stability=float(stability),
                         beta_scaled_error=float(beta_error),rss_relative_error=float(rss_error)))
    report=dict(status='PASS' if all(all(r['checks'].values()) for r in rows) else 'FAIL',
                cases=len(rows),max_covariance_error=max(r['covariance_scaled_error'] for r in rows),
                limits=dict(covariance=1e-8,coefficients=1e-9,rss_relative=1e-9,reference_stability=1e-60),
                results=rows)
    with args.out.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    return report['status']!='PASS'
if __name__=='__main__': raise SystemExit(main())
