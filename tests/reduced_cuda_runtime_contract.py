"""Opt-in CUDA tier guards, failure state and cache interoperability.

Use new output directories and exact candidate/reference modules. Synthetic
inputs only; no build, installation or timing decision is performed here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    spec=importlib.util.spec_from_file_location('py_hdfe_v11',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert Path(module.__file__).resolve()==Path(path).resolve()
    return module

def data():
    rng=np.random.default_rng(2026111502);n=49152
    fes=[np.repeat(np.arange(n//24,dtype=np.int32),24),rng.integers(0,83,n,dtype=np.int32),rng.integers(0,13,n,dtype=np.int32)]
    X=np.asfortranarray(rng.normal(size=(n,3)));y=X@np.array([.5,-.25,.01])+rng.normal(size=n)
    for f in fes:y+=rng.normal(size=int(f.max())+1)[f]
    return y,X,dict(fes=fes,clusters=[fes[0]])

def snapshot(reg):
    arrays={k:np.asarray(getattr(reg,k)) for k in ['coef_','covariance_','stderr_','residuals_','sample_index_','omitted_reason_']}
    digest={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    diagnostics={}
    for name in ['converged_','precision_certified_','gpu_used_','gpu_status_code_','absorption_method_used',
        'reduced_pcg_status_','reduced_pcg_reason_','threads_used_','parallel_workers_active_','lifecycle_state_']:
        value=getattr(reg,name);diagnostics[name]=value if isinstance(value,(str,bool,int,float)) else int(value)
    return dict(digest=digest,diagnostics=diagnostics),arrays

def worker(args):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));mod=load(args.module)
    y,X,fit=data();kind=args.kind
    options=dict(se_type='cluster',num_threads=args.threads,tolerance_mode='reghdfe-comparable',tol=1e-8,max_iter=10000)
    if kind=='slopes' or kind.startswith('group'):
        sys.path.insert(0,str(ROOT/'tests/analytic_precision'));from cases import generate
        case=generate('heterogeneous' if kind=='slopes' else 'group_sum');frame=case.frame
        y=frame.y.to_numpy();X=frame[['x1','x2','x3']].to_numpy();options['se_type']=case.vce
        fit=dict(fes=[frame[f].to_numpy(dtype=np.int64) for f in case.fes])
        if case.slopes:fit['slopes']=[(i,frame[v].to_numpy(),intercept) for i,v,intercept in case.slopes]
        if case.kind=='group_individual':fit.update(group=frame.group.to_numpy(dtype=np.int64),individual=frame.individual.to_numpy(dtype=np.int64),aggregation=case.aggregation)
    reg=mod.HdfeRegressor(**options);record={};arrays={}
    if kind=='fatal_reuse':
        reg.fit(y,X,**fit);before,_=snapshot(reg)
        os.environ['XHDFE_TEST_REDUCED_PCG_CUDA_FATAL']='1'
        try:reg.fit(y,X,**fit)
        except RuntimeError as exc:record['fatal_error']=str(exc)
        else:raise AssertionError('injected fatal error swallowed')
        failed,_=snapshot(reg)
        assert 'injected fatal error' in record['fatal_error']
        assert not reg.converged_ and not reg.precision_certified_ and reg.lifecycle_state_=='failed'
        assert np.asarray(reg.coef_).size==np.asarray(reg.covariance_).size==np.asarray(reg.residuals_).size==0
        os.environ['XHDFE_TEST_REDUCED_PCG_CUDA_FATAL']='0';reg.fit(y,X,**fit)
        after,arrays=snapshot(reg);assert before['digest']==after['digest']
        record.update(result=after,failed=failed)
    else:
        try:reg.fit(y,X,**fit)
        except RuntimeError as exc:
            record['error']=str(exc);record['error_type']=type(exc).__name__
            assert not reg.converged_ and not reg.precision_certified_
            assert np.asarray(reg.coef_).size==np.asarray(reg.covariance_).size==np.asarray(reg.residuals_).size==0
        record['result'],arrays=snapshot(reg)
    record['module']=str(Path(args.module).resolve());record['sha256']=hashlib.sha256(Path(args.module).read_bytes()).hexdigest()
    Path(args.output).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(Path(args.output).with_suffix('.npz'),**arrays)

def main(args):
    out=Path(args.output).resolve();out.mkdir(exist_ok=args.resume);rows=[]
    def run(label,module,kind='plain',threads=4,enabled=True,backend='cuda',extra=None):
        d=out/label
        if args.resume and (d/'result.json').is_file() and (d/'result.npz').is_file():
            rec=json.loads((d/'result.json').read_text())
            assert rec['sha256']==hashlib.sha256(Path(module).read_bytes()).hexdigest()
            rec['_arrays']=str(d/'result.npz');return rec,(d/'stderr.txt').read_text()
        d.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('XHDFE_')}
        env.update(XHDFE_GPU_BACKEND=backend,XHDFE_REDUCED_PCG_CUDA=str(int(enabled)),XHDFE_REDUCED_PCG_TRACE='1',
            XHDFE_MOBILITY_MODE='off',XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_DEBUG_SOLVER='1',
            CUDA_CACHE_PATH=str(out/'cuda_cache'),CUDA_ENABLE_COREDUMP_ON_EXCEPTION='0',OPENBLAS_NUM_THREADS='1')
        if extra:env.update(extra)
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--module',str(module),'--kind',kind,'--threads',str(threads),'--output',str(d/'result')]
        proc=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=420)
        (d/'stdout.txt').write_text(proc.stdout);(d/'stderr.txt').write_text(proc.stderr)
        assert proc.returncode==0,(label,proc.returncode,proc.stderr[-2000:])
        rec=json.loads((d/'result.json').read_text());rec['_arrays']=str(d/'result.npz')
        return rec,proc.stderr
    def same(a,b,label,exact=False):
        assert a.get('error')==b.get('error'),label
        if a['result']['digest']==b['result']['digest']:
            rows.append(dict(label=label,status='PASS',comparison='exact result bytes'));return
        assert not exact,label
        x=np.load(a['_arrays']);y=np.load(b['_arrays'])
        for key in ['sample_index_','omitted_reason_']:assert np.array_equal(x[key],y[key]),label
        use=y['omitted_reason_']==0
        db=float(np.max(np.abs(x['coef_'][use]-y['coef_'][use])/np.maximum(1,np.abs(y['coef_'][use]))))
        V=y['covariance_'][np.ix_(use,use)];sd=np.sqrt(np.maximum(0,np.diag(V)));scale=np.outer(sd,sd)
        delta=np.abs(x['covariance_'][np.ix_(use,use)]-V)
        dv=float(np.max(np.divide(delta,scale,out=np.where(delta==0,0.0,np.inf),where=scale>0)))
        ds=float(np.max(np.abs(x['stderr_'][use]/y['stderr_'][use]-1)))
        dr=float(np.max(np.abs(x['residuals_']-y['residuals_'])))
        assert db<=1e-8 and dv<=1e-8 and ds<=1e-7 and dr<=1e-7,(label,db,dv,ds,dr)
        rows.append(dict(label=label,status='PASS',comparison='existing numerical limits; legacy CUDA reductions need not repeat bits',b=db,V=dv,se=ds,residual=dr))
    def accepted(record,trace):
        d=record['result']['diagnostics']
        assert d['converged_'] and d['precision_certified_'] and d['gpu_used_'] and d['gpu_status_code_']==1
        assert d['absorption_method_used']==8 and d['reduced_pcg_status_']==1
        assert 'reduced_pcg_cuda event=operator calls=0 ' not in trace and 'reduced_pcg_cuda event=operator calls=' in trace
    for threads in [1,2,8,16,24,48]:
        rec,trace=run('threads_'+str(threads),args.module,threads=threads);accepted(rec,trace)
        d=rec['result']['diagnostics'];assert d['threads_used_']==threads
        if threads>1:assert d['parallel_workers_active_']>1
        rows.append(dict(label='threads_'+str(threads),status='PASS',diagnostics=d))
    rec,trace=run('fatal_reuse',args.module,kind='fatal_reuse');accepted(rec,trace)
    rows.append(dict(label='fatal_reuse',status='PASS',failure=rec['failed']))
    legacy,_=run('legacy',args.module,enabled=False)
    for stage in ['geometry','iterate','residuals','certificate','bad_alloc']:
        rec,trace=run('decline_'+stage,args.module,extra={'XHDFE_TEST_REDUCED_PCG_DECLINE':stage})
        assert rec['result']['diagnostics']['reduced_pcg_status_']==2
        same(rec,legacy,'decline_'+str(stage)+'_legacy')
    for kind in ['slopes','group_sum']:
        ref,_=run(kind+'_base',args.base_cuda,kind=kind);rec,trace=run(kind+'_new',args.module,kind=kind)
        assert rec['result']['diagnostics']['reduced_pcg_status_']!=1 and 'reduced_pcg_cuda event=operator' not in trace
        same(rec,ref,kind+'_bypass')
    ref,_=run('absent_base',args.base_cuda,extra={'CUDA_VISIBLE_DEVICES':'-1'})
    rec,trace=run('absent_new',args.module,extra={'CUDA_VISIBLE_DEVICES':'-1'});same(rec,ref,'device_absent')
    for enabled in [False,True]:
        ref,_=run('cpu_base_'+str(enabled),args.base_cpu,backend='cpu',enabled=enabled)
        rec,_=run('cpu_new_'+str(enabled),args.cpu_module,backend='cpu',enabled=enabled);same(rec,ref,'cpu_only_'+str(enabled),exact=True)
    for cache in ['abs','fe','mobility']:
        def env(mode,path):
            if cache=='abs':return dict(XHDFE_ABSORPTION_CACHE_MODE=mode,XHDFE_ABSORPTION_CACHE=str(path))
            if cache=='fe':return dict(XHDFE_FE_STRUCTURE_CACHE=str(path),XHDFE_FE_STRUCTURE_MODE=mode)
            return dict(XHDFE_MOBILITY_MODE=mode,XHDFE_MOBILITY_PROFILE=str(path))
        paths=[out/(cache+'_'+str(i)+'.cache') for i in range(2)]
        writes=[]
        for i,module in enumerate([args.base_cuda,args.module]):
            rec,trace=run(cache+'_write_'+str(i),module,extra=env('write',paths[i]));writes.append(rec)
            assert 'reduced_pcg_cuda event=operator' not in trace
            assert paths[i].is_file()
        same(writes[0],writes[1],cache+'_write_equivalence')
        if cache=='abs':
            # Legacy CUDA deliberately does not load absorption transforms.
            # Its record keys and stored method must still be unchanged.
            headers=[p.read_bytes()[:80] for p in paths]
            assert headers[0][:64]==headers[1][:64]
            assert all(struct.unpack_from('=i',h,76)[0]!=8 for h in headers)
        hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        for i,module in enumerate([args.module,args.base_cuda]):
            rec,trace=run(cache+'_crossread_'+str(i),module,extra=env('read',paths[i]))
            same(rec,writes[i],cache+'_crossread_'+str(i));assert 'reduced_pcg_cuda event=operator' not in trace
            if cache=='abs':assert 'cache_hit=0' in trace
        assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        rows.append(dict(label=cache+'_files_unchanged',status='PASS',hashes=hashes))
    (out/'receipt.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
    print('REDUCED_CUDA_RUNTIME_CONTRACT_PASS',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--output',required=True)
    p.add_argument('--base-cuda');p.add_argument('--base-cpu');p.add_argument('--cpu-module')
    p.add_argument('--worker',action='store_true');p.add_argument('--kind',default='plain');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true',help='Reuse completed immutable-module worker receipts in this synthetic campaign.')
    a=p.parse_args();worker(a) if a.worker else main(a)
