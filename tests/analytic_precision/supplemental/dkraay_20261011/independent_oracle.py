"""Synthetic DK checks against explicit WLS/2SLS and lag-by-lag Bartlett sums.

The oracle does not call either estimator and does not use the candidate's b,
residuals, covariance or degrees of freedom as mathematical inputs.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, subprocess
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent

def fixture():
    rng = np.random.default_rng(20261011)
    ident = np.repeat(np.arange(1, 9), 13)
    time = np.tile(np.arange(1, 14), 8)
    x1, z, noise = rng.normal(size=(3, len(time)))
    x2 = .9*z + .4*noise + rng.normal(size=len(time))
    y = 1.3 + .7*x1 - .45*x2 + rng.normal(size=8)[ident-1] + rng.normal(size=13)[time-1] + .6*noise
    return pd.DataFrame(dict(id=ident,time=time,x1=x1,x2=x2,z=z,y=y,
                             aw=.3+rng.random(len(time))*3,fw=1+(np.arange(len(time))%4),
                             row=np.arange(1,len(time)+1)))

def oracle(df, spec):
    mask = df[['y',*spec['x'],'time',*spec['fe']]].notna().all(axis=1).to_numpy()
    if spec.get('subset'): mask &= df.time.to_numpy() > 3
    if spec.get('unbalanced'): mask &= ((df.id*7+df.time)%5 != 0).to_numpy()
    if spec.get('singletons'): mask &= (df.id <= 8).to_numpy()
    if spec.get('group'): mask &= (~df.duplicated('group')).to_numpy()
    weight = spec.get('weight')
    w = df['fw' if weight=='fw' else 'aw'].to_numpy() if weight else np.ones(len(df))
    mask &= w > 0
    data = df.loc[mask]
    w = w[mask].astype(float)
    y = data.y.to_numpy()
    X = data[spec['x']].to_numpy()
    n = w.sum() if weight=='fw' else len(y)
    sw = np.sqrt(w)
    if spec['fe']:
        if spec.get('group'):
            D = pd.crosstab(df.group,df.id).to_numpy(dtype=float)
            if spec['group']=='mean': D/=D.sum(axis=1)[:,None]
        else:
            D = np.column_stack([pd.get_dummies(data[f],dtype=float).to_numpy() for f in spec['fe']])
        DW = D*sw[:,None]
        Xtilde = X-D@np.linalg.lstsq(DW,X*sw[:,None],rcond=None)[0]
        yt = y-D@np.linalg.lstsq(DW,y*sw,rcond=None)[0]
        rankD = np.linalg.matrix_rank(DW)
        # reghdfe nesting convention for these connected synthetic designs.
        if spec['fe']==['time']: dfa = 0
        elif 'time' in spec['fe']: dfa = len(data.id.unique()) - 1
        else: dfa = rankD
        recovered = spec.get('constant',True)
    else:
        recovered = False
        Xtilde = np.column_stack([X,np.ones(len(y))]) if spec.get('constant',True) else X
        yt = y
        dfa = 1 if spec.get('constant',True) else 0
    score = Xtilde
    if spec.get('iv'):
        Z = data[['x1','z']].to_numpy()
        if spec['fe']:
            Z -= D@np.linalg.lstsq(DW,Z*sw[:,None],rcond=None)[0]
        elif spec.get('constant',True): Z=np.column_stack([Z,np.ones(len(y))])
        score = Z@np.linalg.lstsq(Z*sw[:,None],Xtilde*sw[:,None],rcond=None)[0]
    bread = np.linalg.inv((score*sw[:,None]).T@(score*sw[:,None])) if score.shape[1] else np.empty((0,0))
    beta = bread@(score.T@(w*yt))
    u = yt-Xtilde@beta
    if recovered:
        means = np.average(X,axis=0,weights=w)
        beta = np.r_[beta,np.average(y,weights=w)-means@beta]
        expanded = np.zeros((len(beta),len(beta)))
        expanded[:-1,:-1]=bread
        expanded[-1,-1]=1/w.sum()
        transport = np.eye(len(beta));transport[-1,:-1]=-means
        bread = transport@expanded
        score = np.column_stack([score,np.ones(len(y))])
    levels = np.unique(data.time)
    T = len(levels)
    B = min(T,spec['bw'] or int(np.floor(4*(T/100)**(2/9)))+1)
    h = np.stack([((w*u)[:,None]*score)[data.time.to_numpy()==t].sum(axis=0) for t in levels])
    meat = h.T@h
    for lag in range(1,B):
        cross=h[lag:].T@h[:-lag]
        meat += (1-lag/B)*(cross+cross.T)
    dfm = len(spec['x'])
    denominator = n-dfm-dfa
    scale = (n-1)/denominator*T/(T-1)
    V = scale*bread@meat@bread.T
    return dict(b=beta,V=V,N=n,rss=float(np.dot(w,u*u))*(len(y)/w.sum() if weight and weight!='fw' else 1),
                df_a=dfa,df_m=dfm,df_r=min(denominator,T-1),N_clust=T,dkraay_bw=B,
                mask=mask.astype(int),resid=u)

def cases():
    specs=[]
    for fe in [[],['id'],['id','time'],['time']]:
        for B in [1,0,4,99]:
            specs.append(dict(name=('none' if not fe else '_'.join(fe))+f'_bw{B}',fe=fe,bw=B,x=['x1','x2']))
    for wt in ['aw','pw','iw','fw']:
        for fe in [['id'],['id','time']]:
            specs.append(dict(name=f'{wt}_'+('_'.join(fe)),fe=fe,bw=3,weight=wt,x=['x1','x2']))
    for mode in ['unbalanced','subset','singletons','missing_time','negative_time','gaps','permuted','scaled_weights']:
        specs.append(dict(name=mode,fe=['id'],bw=3,x=['x1','x2'],**{mode:True},
                          **({'weight':'aw'} if mode=='scaled_weights' else {})))
    for fe in [[],['id']]:
        specs.append(dict(name='noconstant_'+('none' if not fe else 'id'),fe=fe,bw=2,x=['x1','x2'],constant=False))
        specs.append(dict(name='iv_'+('none' if not fe else 'id'),fe=fe,bw=3,x=['x1','x2'],iv=True))
    specs.append(dict(name='pure_fe',fe=['id'],bw=3,x=[]))
    for aggregation in ['mean','sum']:
        specs.append(dict(name='group_'+aggregation,fe=['id'],bw=3,x=['x1','x2'],group=aggregation))
    return specs

def group_fixture(aggregation):
    rng=np.random.default_rng(2033)
    ids=np.array([rng.choice(np.arange(1,10),2,replace=False) for _ in range(48)])
    alpha=rng.normal(size=9)
    x1,x2,noise=rng.normal(size=(3,48))
    time=np.arange(48)%12+1
    fe=alpha[ids-1].sum(axis=1)/(2 if aggregation=='mean' else 1)
    y=1.3+.7*x1-.45*x2+fe+noise+np.cos(time)
    return pd.DataFrame(dict(group=np.repeat(np.arange(1,49),2),id=ids.ravel(),
        time=np.repeat(time,2),x1=np.repeat(x1,2),x2=np.repeat(x2,2),y=np.repeat(y,2),row=np.arange(1,97)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path,help='New directory for synthetic fixtures and receipts')
    ap.add_argument('--package',type=Path,default=HERE.parents[3]/'stata')
    ap.add_argument('--backend',default='cpu',choices=['cpu','cuda'])
    args=ap.parse_args()
    package=args.package.resolve()
    for name in ['xhdfe.ado','xhdfe.plugin']:
        if not (package/name).is_file(): raise FileNotFoundError(package/name)
    run=args.output.resolve()
    if any(c in str(p) for p in [package,run] for c in ['"','`','$','\n','\r']):
        raise ValueError('Unsupported quoting in a Stata path')
    run.mkdir(parents=True,exist_ok=False);(run/'tmp').mkdir()
    for folder in ['plus','personal','oldplace']: (run/'ado'/folder).mkdir(parents=True)
    setup=''.join(f'sysdir set {key} "{run}/ado/{folder}/"\n' for key,folder in
                  [('PLUS','plus'),('PERSONAL','personal'),('OLDPLACE','oldplace')])
    setup+=f'global S_ADO "BASE;SITE;{package};PERSONAL;PLUS;OLDPLACE"\nset more off\nset linesize 255\n'
    for name in ['xhdfe.ado','xhdfe.plugin']:
        setup+=f'findfile {name}\nassert "`r(fn)\'" == "{package}/{name}"\n'
    specs=[s for s in cases() if args.backend=='cpu' or (s['fe'] and not s.get('iv'))]
    outputs=[];commands=[]
    for spec in specs:
        df=group_fixture(spec['group']) if spec.get('group') else fixture()
        name=spec['name']
        if spec.get('singletons'):
            extra=df.iloc[[0]].copy();extra['id']=99;extra['time']=100;extra['row']=105
            df=pd.concat([df,extra],ignore_index=True)
        if spec.get('missing_time'): df.loc[df.index%17==0,'time']=np.nan
        if spec.get('negative_time'): df['time']=df.time-1970
        if spec.get('gaps'): df=df.loc[~df.time.isin([5,8,11])].copy()
        if spec.get('permuted'): df=df.sample(frac=1,random_state=3).reset_index(drop=True)
        if spec.get('scaled_weights'): df['aw']*=1e12
        expected=oracle(df,spec)
        # Stata sorts on tsset; restore the source order before estimation.
        df['oracle_order']=np.arange(1,len(df)+1)
        df.to_stata(run/(name+'.dta'),write_index=False,version=118)
        x=' '.join(spec['x']);fe=' '.join(spec['fe'])
        opts=f'absorb({fe})' if fe else 'noabsorb'
        if spec.get('group'): opts+=f' group(group) individual(id) aggregation({spec["group"]})'
        if not spec.get('constant',True): opts+=' noconstant'
        if spec.get('iv'): opts+=' endogenous(x2) instruments(z)'
        wt=spec.get('weight');wstr=f'[{wt}={"fw" if wt=="fw" else "aw"}]' if wt else ''
        cond='if time > 3' if spec.get('subset') else ('if mod(id*7+time,5)!=0' if spec.get('unbalanced') else '')
        bw=str(spec['bw']) if spec['bw'] else ''
        cmd=f'xhdfe y {x} {cond} {wstr}, {opts} vce(dkraay {bw}) gpubackend({args.backend}) numthreads(2) residuals(__u)'
        outputs.append((spec,expected))
        commands.append(f'''use "{run/name}.dta", clear
quietly xtset row time
sort oracle_order
capture noisily {cmd}
local fit_rc = _rc
file open receipt using "{run/name}.txt", write text
file write receipt "rc " (`fit_rc') _n
if !`fit_rc' {{
    assert "`e(vce)'" == "dkraay"
    assert "`e(dkraay_timevar)'" == "time"
    matrix bb = e(b)
    matrix vv = e(V)
    file write receipt "b"
    forvalues j=1/`=colsof(bb)' {{
        file write receipt " " %24.17e (bb[1,`j'])
    }}
    file write receipt _n "V"
    forvalues j=1/`=rowsof(vv)' {{
        forvalues k=1/`=colsof(vv)' {{
            file write receipt " " %24.17e (vv[`j',`k'])
        }}
    }}
    file write receipt _n
    foreach statistic in N rss df_a df_m df_r N_clust dkraay_bw gpu_used converged {{
        file write receipt "`statistic' " %24.17e (e(`statistic')) _n
    }}
    gen byte __sample = e(sample)
    export delimited __sample __u using "{run/name}_rows.csv", replace
}}
file close receipt
''')
    do=run/'oracle.do'
    do.write_text(setup+''.join(commands)+'di "DK_ORACLE_COMPLETE"\nexit\n')
    env={k:v for k,v in os.environ.items() if k!='LD_LIBRARY_PATH' and not k.startswith('XHDFE_')}
    env.update(STATATMP=str(run/'tmp'),TMPDIR=str(run/'tmp'),CUDA_CACHE_PATH=str(run/'tmp/cuda'),
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    proc=subprocess.run(['/usr/local/stata/stata-mp','-q','-b','do',str(do)],cwd=run,env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=420)
    (run/'process.txt').write_bytes(proc.stdout)
    log=(run/'oracle.log').read_text(errors='replace')
    results=[]
    for spec,expected in outputs:
        p=run/(spec['name']+'.txt');result=dict(case=spec['name'],status='FAIL')
        if p.exists():
            actual={line.split()[0]:np.array([float(x) if x!='.' else np.nan for x in line.split()[1:]]) for line in p.read_text().splitlines()}
            result['rc']=int(actual['rc'][0])
            if result['rc']==0:
                V=actual['V'].reshape(expected['V'].shape)
                result['b_error']=float(np.max(np.abs(actual['b']-expected['b'])))
                scales=np.sqrt(np.maximum(np.diag(expected['V']),0));norm=np.outer(scales,scales)
                result['V_scaled_error']=float(np.max(np.abs(V-expected['V'])/np.maximum(norm,1e-12)))
                result['stats']={k:dict(actual=float(actual[k][0]),expected=float(expected[k])) for k in ['N','rss','df_a','df_m','df_r','N_clust','dkraay_bw']}
                rows=pd.read_csv(run/(spec['name']+'_rows.csv'))
                sample_ok=np.array_equal(rows['__sample'].to_numpy(),expected['mask'])
                result['sample_equal']=bool(sample_ok)
                result['residual_error']=float(np.max(np.abs(rows.loc[expected['mask'].astype(bool),'__u'].to_numpy()-expected['resid'])))
                stats_ok=all(np.isclose(v['actual'],v['expected'],rtol=1e-9,atol=1e-10) for v in result['stats'].values())
                result['gpu_used']=float(actual['gpu_used'][0])
                result['status']='PASS' if (result['b_error']<1e-9 and result['V_scaled_error']<1e-8 and result['residual_error']<1e-8 and stats_ok and sample_ok and actual['converged'][0]==1 and (args.backend=='cpu' or actual['gpu_used'][0]==1)) else 'FAIL'
        results.append(result)
    complete=bool(re.search(r'^DK_ORACLE_COMPLETE$',log,re.M)) and proc.returncode==0 and not re.search(r'^r\(\d+\);',log,re.M)
    receipt=dict(complete=complete,backend=args.backend,cases=results,
                 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'independent_oracle.py',package/'xhdfe.ado',package/'xhdfe.plugin']})
    (run/'RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for r in results: print(r['case'],r['status'],r.get('rc'),r.get('b_error'),r.get('V_scaled_error'),flush=True)
    raise SystemExit(0 if complete and all(r['status']=='PASS' for r in results) else 1)
if __name__=='__main__': main()
