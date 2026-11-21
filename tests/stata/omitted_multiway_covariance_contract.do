* Active-column covariance, structural omitted zeros and Stata's inference convention.
* Synthetic data only. Run in a new output directory with XHDFE_STATA_ADOPATH
* pointing to the exact candidate package. Arguments: cpu|cuda and tolerance mode.
version 16
clear all
set more off
set linesize 255
args backend mode
if "`backend'" == "" local backend cpu
if "`mode'" == "" local mode reghdfe-comparable
local package : env XHDFE_STATA_ADOPATH
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert `"`r(fn)'"' == `"`package'/xhdfe.ado"'
findfile xhdfe.plugin
assert `"`r(fn)'"' == `"`package'/xhdfe.plugin"'

mata:
void omc_compare(real rowvector kept) {
    real rowvector b0, b1, bs
    real matrix v0, v1
    real colvector se0, se1
    real scalar db, dv, ds, i, j, difference
    b0=st_matrix("B0"); v0=st_matrix("V0")
    b1=st_matrix("B1"); v1=st_matrix("V1")
    b1=b1[1,kept]; v1=v1[kept,kept]
    bs=abs(b0)
    for (i=1;i<=cols(bs);i++) if (bs[i]<1) bs[i]=1
    db=max(abs(b1-b0):/bs)
    se0=sqrt(diagonal(v0)); se1=sqrt(diagonal(v1))
    dv=0; ds=0
    for (i=1;i<=rows(v0);i++) {
        if (se0[i]>0) ds=max((ds,abs(se1[i]/se0[i]-1)))
        for (j=1;j<=cols(v0);j++) {
            difference=abs(v1[i,j]-v0[i,j])
            if (se0[i]>0 & se0[j]>0) dv=max((dv,difference/se0[i]/se0[j]))
            else assert(difference<=1e-12)
        }
    }
    st_numscalar("omc_db",db); st_numscalar("omc_dv",dv); st_numscalar("omc_ds",ds)
    assert(db<=1e-8 & dv<=1e-8 & ds<=1e-7)
}
end

* A positive-covariance fixture checks reghdfe tight and the omitted r(table)
* convention directly; Python's unavailable-SE NaN convention is separate.
set seed 20261115
set obs 3000
generate long fe = floor(60*runiform())
generate long c1 = floor(8*runiform())
generate long c2 = floor(150*runiform())
generate double x1 = rnormal() + c1/10
generate double x2 = rnormal() - c2/100
generate double zero = 0
generate double y = x1 - 2*x2 + sin(fe) + sin(c1) + cos(c2) + rnormal()
reghdfe y zero x1 x2, absorb(fe) vce(cluster c1 c2) tolerance(1e-12)
matrix BR = e(b)
matrix VR = e(V)
matrix TR = r(table)
assert rowsof(TR)>=6 & colsof(TR)==colsof(BR)
xhdfe y zero x1 x2, absorb(fe) vce(cluster c1 c2) tolerance(1e-8) ///
    tolerancemode(`mode') numthreads(4) gpubackend(`backend')
assert e(converged)==1 & e(precision_certified)==1
assert e(gpu_used)==("`backend'"=="cuda")
if "`backend'"=="cuda" {
    assert "`e(gpu_backend)'"=="cuda" & "`e(gpu_status)'"=="used"
}
matrix B1=e(b)
matrix V1=e(V)
matrix T1=r(table)
assert rowsof(T1)>=6 & colsof(T1)==colsof(B1)
matrix B0=(BR[1,2..4])
matrix V0=(VR[2..4,2..4])
mata: omc_compare((2,3,4))
assert B1[1,1]==0 & V1[1,1]==0
forvalues r=1/6 {
    assert T1[`r',1]==TR[`r',1]
}
matrix list T1

* A sparse worker-firm panel exercises the original crash and PSD rebuild.
clear
set seed 20261116
set obs 20000
generate long worker=_n
generate int length=3+floor(6*runiform())
generate long home=floor(1500*runiform())
expand length
bysort worker: generate int year=mod(_n+worker,8)
generate long firm=cond(runiform()<.15,floor(1500*runiform()),home)
generate long firm_sparse=37+firm*641
generate long region=floor(firm/30)
generate double x1=rnormal()
generate double x2=rnormal()+mod(firm,7)/10
generate double zfirst=0
generate double zlast=0
generate double y=.5*x1-.25*x2+sin(worker)+cos(firm)+year/10+rnormal()
generate double aw=.5+1.5*runiform()
generate long fw=1+floor(3*runiform())
foreach weight in none aw fw {
    local w
    if "`weight'"!="none" local w "[`weight'=`weight']"
    foreach constant in on off {
        local nc
        if "`constant'"=="off" local nc noconstant
        foreach cluster in "firm" "firm region" "worker firm_sparse" {
            quietly xhdfe y x1 x2 `w', absorb(worker firm year) vce(cluster `cluster') ///
                tolerancemode(`mode') numthreads(4) gpubackend(`backend') `nc'
            matrix B0=e(b)
            matrix V0=e(V)
            quietly xhdfe y zfirst x1 x2 zlast `w', absorb(worker firm year) ///
                vce(cluster `cluster') tolerancemode(`mode') numthreads(4) gpubackend(`backend') `nc'
            assert e(converged)==1 & e(precision_certified)==1
            assert e(gpu_used)==("`backend'"=="cuda")
            matrix B1=e(b)
            matrix V1=e(V)
            if "`constant'"=="on" mata: omc_compare((2,3,5))
            else mata: omc_compare((2,3))
            assert B1[1,1]==0 & B1[1,4]==0
            forvalues j=1/`=colsof(B1)' {
                assert V1[1,`j']==0 & V1[`j',1]==0
                assert V1[4,`j']==0 & V1[`j',4]==0
            }
        }
    }
}

* k_active=0: all omitted vectors must retain their valid public dimensions.
quietly xhdfe y zfirst zlast, absorb(worker firm year) vce(cluster worker firm_sparse) ///
    tolerancemode(`mode') numthreads(4) gpubackend(`backend')
matrix B1=e(b)
matrix V1=e(V)
assert colsof(B1)==3 & rowsof(V1)==3 & colsof(V1)==3
assert B1[1,1]==0 & B1[1,2]==0
forvalues j=1/3 {
    assert V1[1,`j']==0 & V1[`j',1]==0 & V1[2,`j']==0 & V1[`j',2]==0
}
display "STATA_OMITTED_MULTIWAY_CONTRACT_PASS `backend' `mode'"
