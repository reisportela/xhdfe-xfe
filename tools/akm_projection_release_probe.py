"""NumPy-only closed-form inference gate for installed release modules."""
from __future__ import annotations
import numpy as np


def validate_projection(core):
    worker=np.repeat(np.arange(64,dtype=np.int32),4)
    firm=np.tile(np.array([0,0,1,1],dtype=np.int32),64)
    fp=np.where(firm==0,-.5,.5)
    s=1+worker%3
    v=s*np.tile([1.,-1.,1.,-1.],64)
    x=5*fp+v
    y=.3*np.sin(.37*(worker+1))+.7*fp+.35*x+.1*np.sin(.73*np.arange(1,257))
    vv=float(v@v);raw=float(fp@y)/64;gamma=float(v@y)/vv
    fy=np.sum((fp*y).reshape(64,4),axis=1)
    vy=np.sum((v*y).reshape(64,4),axis=1)
    local_vv=4.*(1+np.arange(64)%3)**2
    lf=1/64;lv=local_vv/vv
    geometric=lambda a:1+a+a*a+a**3+a**4
    score=(-.5/64*(geometric(lf)*(fy-raw)+lf**5*fy)
           +2.5*local_vv/vv*(geometric(lv)*(vy/local_vv-gamma)+lv**5*vy/local_vv))
    variance=float(score@score);centre=abs(raw-5*gamma)/2
    radius=np.sqrt(-2*np.log(.05)*variance)
    expected=np.array([2*256/255*centre*np.sqrt(variance),
                       256/255*max(0.,centre-radius)**2,
                       256/255*(centre+radius)**2])
    rows=[]
    for method in ("exact","jla"):
        for exponent in (-300,0,300,500):
            scale=2.**exponent
            point=core.akm_kss(y*scale,worker,firm,X=x[:,None],leverages=method,
                               sigma_centering="person_year",num_threads=2)
            fit=core.akm_kss(y*scale,worker,firm,X=x[:,None],leverages=method,
                             sigma_centering="person_year",num_threads=2,compute_se=True)
            p=fit["projection_inference"]
            actual=np.array([fit["component_se"]["se_var_psi"],p["var_psi"]["ci_lb"],p["var_psi"]["ci_ub"]])/(scale*scale)
            if not point["converged"] or not fit["converged"]:
                raise RuntimeError("AKM projection release fixture did not converge")
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-8)
            for key in ("var_alpha","var_psi","cov_alpha_psi"):
                if fit["kss"][key]!=point["kss"][key]:raise RuntimeError("SE request changed AKM point")
            if (p["threads_used"],p["parallel_workers_active"])!=(2,2):
                raise RuntimeError("AKM projection did not use two real workers")
            if p["backend"]!="cpu" or p["cluster"]!="worker" or p["dimension"]!=2:
                raise RuntimeError("AKM projection metadata mismatch")
            rows.append({"leverages":method,"exponent":exponent,
                         "maximum_normalised_error":float(np.max(np.abs(actual-expected)))})
    return {"status":"PASS","cases":rows,"oracle":"balanced two-firm scalar closed form"}
