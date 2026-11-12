"""Independent full-design contract for optional worker projection inference.

The reference uses a dense SVD of the complete design, not xhdfe's FWL bread
or sparse operators. Data are synthetic. Refusals are recorded separately.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2
from akm_projection_oracle import trust_min


def fixture(kind,workers=80):
    firms=5
    w=np.repeat(np.arange(workers),4)
    f=np.empty(len(w),dtype=int)
    for worker in range(workers):
        first=worker%firms
        last=first if kind=="stayers" and worker%3 else (first+1+worker//firms%(firms-1))%firms
        f[4*worker:4*worker+4]=[first,first,last,last]
    if kind=="unbalanced":
        lengths=2+np.arange(workers)%9
        w=np.repeat(np.arange(workers),lengths)
        f=np.concatenate([np.r_[np.zeros(length-1,dtype=int),1+worker%4]
                          for worker,length in enumerate(lengths)])
    rng=np.random.default_rng(1182151)
    x=rng.normal(size=(len(w),1))*.3+1.3*np.sin(f)[:,None]
    if kind=="none":x=np.empty((len(w),0))
    elif kind=="local":
        local=np.zeros(len(w));local[:2]=[1.,-1.]
        x=np.column_stack([x,local])
    elif kind=="omitted":x=np.column_stack([x,x,np.ones(len(w))])
    elif kind in ("omitted_front","pruned"):x=np.column_stack([np.ones(len(w)),x])
    signal=.3*np.sin(w)+.4*np.cos(f)
    if x.shape[1]:signal+=.5*x[:,1 if kind in ("omitted_front","pruned") else 0]
    error=.2*rng.normal(size=workers)[w]+.2*rng.normal(size=len(w))
    y=signal+error
    if kind=="pruned":
        # A small disconnected component, a singleton and an articulation
        # worker are all excluded. Random order makes rows non-identity.
        w=np.r_[w,800,800,801,801,802,803,803]
        f=np.r_[f,80,81,80,81,0,0,82]
        y=np.r_[y,np.arange(7)+1000.]
        x=np.vstack([x,np.column_stack([np.ones(7),np.arange(7)+100.])])
        order=rng.permutation(len(y));y,w,f,x=y[order],w[order],f[order],x[order]
    return y,w.astype(np.int32),f.astype(np.int32),x


def reference(y,w,f,x):
    n=len(y);nw=int(max(w))+1;nf=int(max(f))+1;j=nf-1;dim=2*j
    design=np.column_stack([np.eye(nw)[w],np.eye(nf)[f,:j],x])
    inverse=np.linalg.pinv(design,rcond=1e-13)
    p=design @ inverse;m=np.eye(n)-p
    probabilities=np.bincount(f,minlength=nf)/n
    contrast=null_space(np.sqrt(probabilities)[None,:]).T @ np.diag(np.sqrt(probabilities))
    selector=np.zeros((dim,design.shape[1]))
    selector[:j,nw:nw+j]=contrast[:,:j]
    worker_average=np.zeros((nf,nw));np.add.at(worker_average,(f,w),1.)
    worker_average/=np.bincount(f)[:,None]
    selector[j:,:nw]=contrast @ worker_average
    influence=selector @ inverse
    centred=y-y[0];eta=m @ centred
    scores=[]
    for worker in range(nw):
        g=np.flatnonzero(w==worker);pg=p[np.ix_(g,g)]
        h=np.eye(len(g));power=h.copy()
        for _ in range(4):power=power @ pg;h+=power
        outside=(centred.sum()-centred[g].sum())/(n-len(g))
        residual=h @ eta[g]+power @ pg @ (centred[g]-outside)
        scores.append(influence[:,g] @ residual)
    scores=np.asarray(scores);_,r=np.linalg.qr(scores,mode="reduced");root=r.T
    k=influence @ centred;cov=root @ root.T;critical=chi2.ppf(.95,dim)
    bounds=[];ses=[]
    for comp in (0,1):
        a=np.zeros((dim,dim))
        if comp==0:a[:j,:j]=np.eye(j)*n/(n-1)
        else:a[:j,j:]=a[j:,:j]=np.eye(j)*n/(n-1)/2
        h=root.T @ a @ root;g=root.T @ a @ k;c=k @ a @ k
        lo=trust_min(h,g,np.sqrt(critical));hi=trust_min(-h,-g,np.sqrt(critical))
        bounds.append([max(0.,c+lo["value"]) if comp==0 else c+lo["value"],c-hi["value"]])
        ses.append(2*np.linalg.norm(root.T @ a @ k))
    return np.asarray(ses),np.asarray(bounds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",action="store_true")
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("py_hdfe_v11",args.module.resolve())
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    rows=[];refusals=[];fits=0
    for kind in ("none","controlled","stayers","local","omitted","omitted_front","pruned","unbalanced"):
        y,w,f,x=fixture(kind)
        kept=w<80 if kind=="pruned" else np.ones(len(y),dtype=bool)
        expected_se,expected_ci=reference(y[kept],w[kept],f[kept],x[kept])
        for level in ("match","obs"):
            previous=None
            for method,seed in (("exact",20261110),("jla",20261110),("jla",20261111)):
                options=dict(leave_out_level=level,leverages=method,seed=seed,
                             sigma_centering="person_year",num_threads=8,gpu=args.gpu)
                if x.shape[1]:options["X"]=x
                point=native.akm_kss(y,w,f,**options)
                result=native.akm_kss(y,w,f,compute_se=True,**options);fits+=2
                assert point["converged"] and result["converged"]
                assert result["sample"]["n_obs"]==sum(kept)
                assert np.array_equal(result["sample"]["keep"],kept)
                for component in ("var_alpha","var_psi","cov_alpha_psi"):
                    assert point["kss"][component]==result["kss"][component]
                p=result["projection_inference"]
                assert p["method"]=="worker_projection" and p["cluster"]=="worker"
                assert p["backend"]=="cpu" and p["simultaneous"]
                assert p["dimension"]==8 and p["clusters"]==80
                if kind in ("omitted_front","pruned"):
                    assert p["control_rank"]==1 and p["control_bread_reused"]
                assert p["threads_used"]==8 and p["parallel_workers_active"]==8
                assert "weak_id" not in result
                se=np.array([result["component_se"]["se_var_psi"],result["component_se"]["se_cov_alpha_psi"]])
                ci=np.array([[p[name]["ci_lb"],p[name]["ci_ub"]] for name in ("var_psi","cov_alpha_psi")])
                se_error=float(np.max(np.abs(se-expected_se)));ci_error=float(np.max(np.abs(ci-expected_ci)))
                assert se_error<1e-9,(kind,level,method,se_error)
                assert ci_error<1e-8,(kind,level,method,ci_error)
                assert result["component_se"]["theta_var_psi"]==result["kss"]["var_psi"]
                assert np.isnan(result["component_se"]["se_var_alpha"])
                if previous is None:previous=(se,ci)
                else:
                    np.testing.assert_allclose(se,previous[0],atol=1e-11,rtol=1e-10)
                    np.testing.assert_allclose(ci,previous[1],atol=1e-10,rtol=1e-10)
                if args.gpu:assert result["gpu_used"] and result["gpu_backend"]=="cuda" and result["gpu_status"]=="used"
                rows.append(dict(kind=kind,level=level,method=method,seed=seed,se_error=se_error,ci_error=ci_error))
        # New inference does not depend on the outcome's arbitrary level or
        # the requested legacy Monte Carlo/AM switches.
        base_options=dict(sigma_centering="person_year",compute_se=True,leverages="exact",num_threads=1,gpu=args.gpu)
        if x.shape[1]:base_options["X"]=x
        first=native.akm_kss(y,w,f,**base_options)
        shifted=native.akm_kss(y+1e6,w,f,eigen_diagnostics=True,se_nsim=7,**base_options);fits+=2
        for name in ("var_psi","cov_alpha_psi"):
            for endpoint in ("ci_lb","ci_ub"):
                assert abs(first["projection_inference"][name][endpoint]-shifted["projection_inference"][name][endpoint])<1e-7
        for key in ("se_var_psi","se_cov_alpha_psi"):
            assert abs(first["component_se"][key]-shifted["component_se"][key])<1e-8
    y,w,f,x=fixture("none")
    for label,extra in (("lowess",dict(se_sigma_lowess=True)),
                        ("lincom",dict(Z=np.sin(f)[:,None])),
                        ("fweights inference",dict(compute_se=True,fweights=np.full(len(y),2.))),
                        ("fweights eigen-only",dict(eigen_diagnostics=True,fweights=np.full(len(y),2.)))):
        try:native.akm_kss(y,w,f,sigma_centering="person_year",**extra)
        except (ValueError,RuntimeError) as error:refusals.append(dict(case=label,message=str(error)))
        else:raise AssertionError("unsupported combination accepted: "+label)
    try:native.akm_kss(np.zeros_like(y),w,f,sigma_centering="person_year",compute_se=True)
    except (ValueError,RuntimeError) as error:
        assert "covariance is rank deficient" in str(error)
        refusals.append(dict(case="zero target covariance",message=str(error)))
    else:raise AssertionError("zero target covariance silently accepted")
    # Arithmetic range is tested offline; it never tightens solver targets.
    range_rows=[]
    baseline=native.akm_kss(y,w,f,sigma_centering="person_year",compute_se=True,leverages="exact",num_threads=1,gpu=args.gpu);fits+=1
    for exponent in (-500,500):
        scale=2.**exponent
        value=native.akm_kss(y*scale,w,f,sigma_centering="person_year",compute_se=True,leverages="exact",num_threads=1,gpu=args.gpu);fits+=1
        for key in ("se_var_psi","se_cov_alpha_psi"):
            np.testing.assert_allclose(value["component_se"][key]/(scale*scale),baseline["component_se"][key],atol=1e-11,rtol=1e-9)
        for name in ("var_psi","cov_alpha_psi"):
            for endpoint in ("ci_lb","ci_ub"):
                np.testing.assert_allclose(value["projection_inference"][name][endpoint]/(scale*scale),baseline["projection_inference"][name][endpoint],atol=1e-10,rtol=1e-9)
        range_rows.append(exponent)
    # A full-rank low-noise covariance must not be rejected merely because
    # forming its normal matrix squares an otherwise representable condition.
    y,w,f,x=fixture("controlled",workers=400)
    mu=.3*np.sin(w)+.4*np.cos(f)+.5*x[:,0]
    low=native.akm_kss(mu+1e-7*(y-mu),w,f,X=x,sigma_centering="person_year",compute_se=True,leverages="exact",num_threads=1,gpu=args.gpu);fits+=1
    assert low["projection_inference"]["rank_diagnostic"]==8
    assert low["projection_inference"]["covariance_condition"]>1e12
    assert 0<low["component_se"]["se_var_psi"]<1e-6
    report=dict(status="PASS_SCOPED_PROJECTION_CONTRACT",fits=fits,rows=rows,
                expected_refusals=refusals,gpu=args.gpu,arithmetic_range_exponents=range_rows,
                low_noise_covariance_condition=low["projection_inference"]["covariance_condition"],
                module=str(args.module.resolve()),module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="Independent full-design inference checks; not performance or package certification")
    with args.output.open("x") as out:json.dump(report,out,indent=2)
    print(json.dumps({"status":report["status"],"fits":fits,"oracle_cases":len(rows),"refusals":len(refusals),
                      "max_se_error":max(r["se_error"] for r in rows),"max_ci_error":max(r["ci_error"] for r in rows)}))


if __name__=="__main__":main()
