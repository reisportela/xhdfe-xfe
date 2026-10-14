"""IV covariance with relabelled/equal-size distinct cluster partitions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import iv_intercept_covariance_contract as oracle

ap=argparse.ArgumentParser()
ap.add_argument('--module',type=Path,required=True)
ap.add_argument('--backend',choices=['cpu','cuda'],required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
assert not args.output.exists()
os.environ['XHDFE_GPU_BACKEND']=args.backend
os.environ['CUDA_CACHE_DISABLE']='1'
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
class Factory:
    def HdfeRegressor(self,**kwargs):
        self.last=core.HdfeRegressor(**kwargs)
        return self.last
rows=[]
for equivalent in [True,False]:
    for weighted,kind in [(False,'analytic'),(True,'analytic'),(True,'frequency')]:
        case=oracle.fixture(2.**(-27 if equivalent else -15),(0,0),weighted,kind)
        if not equivalent:
            case['cl2']=np.random.default_rng(314).integers(0,17,len(case['y']))
            case['partition']='distinct_equal_cardinality'
        for mode in ['reghdfe-comparable','xhdfe-fast']:
            factory=Factory()
            if not equivalent and not weighted:
                # This held-out raw CGM matrix is genuinely indefinite. Its
                # expected output follows the chosen PSD policy, so comparing
                # the corrected matrix to the raw positive-V oracle is invalid.
                try:
                    oracle.reference(case,'multiway')
                    raise AssertionError('The negative raw-CGM control became positive')
                except ValueError as error:
                    assert 'not positive-definite' in str(error)
                model=factory.HdfeRegressor(num_threads=2,fit_intercept=True,
                    se_type='cluster',tolerance_mode=mode)
                model.fit(case['y'],case['x'],fes=[case['fe']],instruments=case['z'],
                    endogenous_idx=[0,1],clusters=np.column_stack([case['cl1'],case['cl2']]))
                V=np.asarray(model.covariance_)
                scale=np.sqrt(np.maximum(np.diag(V),np.finfo(float).tiny))
                correlation=(V+V.T)/2/np.outer(scale,scale)
                unit=2.**-53
                allowance=256*unit/(1-256*unit)*np.linalg.norm(correlation,2)
                minimum=float(np.linalg.eigvalsh(correlation).min())
                be=float(np.max(abs(model.coef_-[.5,-.25,3.])))
                passed=bool(model.converged_ and np.isfinite(V).all() and
                    (np.diag(V)>=0).all() and minimum>=-allowance and be<=1e-9)
                row=dict(status='PASS' if passed else 'FAIL',mode=mode,
                    grade_scope='independently indefinite raw CGM; corrected PSD and exact b',
                    minimum_correlation_eigenvalue=minimum,PSD_allowance=allowance,b_error=be)
            else:
                row=oracle.evaluate(factory,case,'multiway',mode)
                if not hasattr(factory,'last'):
                    raise AssertionError(row)
            model=factory.last
            counts=list(model.cluster_counts_)
            combinations=list(model.cluster_combo_counts_)
            assert counts==[17,17]
            assert (combinations==[17,17,17])==equivalent
            row.update(cluster_counts=counts,intersection_counts=combinations,
                equivalent_partitions=equivalent,gpu_used=bool(model.gpu_used_))
            if bool(model.gpu_used_)!=(args.backend=='cuda'):
                row['status']='FAIL'
            rows.append(row)
report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,
    module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='MP90/MP120 oracle; no changed tolerance; equal marginal cardinalities do not imply equal partitions')
with args.output.open('x') as f:json.dump(report,f,indent=2)
print('EQUIVALENT_CLUSTER_PARTITION',report['status'],len(rows))
if report['status']!='PASS':raise SystemExit(1)
