version 16
args stage backend
clear all
set more off
adopath ++ "`stage'"

set obs 1024
generate double a = 2*mod(_n-1,2)-1
generate double e = 2*mod(floor((_n-1)/4),2)-1
generate double y = .75*a+.125*e
generate double x1 = .
generate double weight = 1+2*mod(floor((_n-1)/8),2)
generate byte fe = 0
foreach offset in 1e6 1e10 1e12 1e14 1e15 {
    replace x1 = `offset'+a
    foreach weight_type in none aw fw {
        local weight_option ""
        if "`weight_type'" != "none" local weight_option "[`weight_type'=weight]"
        quietly xhdfe y x1 `weight_option', absorb(fe) noconstant keepsingletons numthreads(2) gpubackend(`backend') tolerance(1e-12)
        assert abs(_b[x1]-.75)<1e-10
        assert e(converged)==1 & e(precision_certified)==1
        assert e(gpu_used)==("`backend'"=="cuda")
    }
}

clear
set obs 4096
generate double x1 = 2*mod(_n-1,2)-1
generate double x2 = 2*mod(floor((_n-1)/2),2)-1
generate double e = 2*mod(floor((_n-1)/4),2)-1
generate int f1 = mod(floor((_n-1)/8),7)
generate int f2 = mod(floor((_n-1)/56),11)
generate int f3 = mod(floor((_n-1)/616),5)
generate double y = .75*x1-.5*x2+.5*f1-.25*f2+.125*f3+.125*e
generate double age = .
foreach offset in 0 1e6 1e14 {
    replace age = `offset'+3*f1-2*f2+f3
    quietly xhdfe y x1 x2 age, absorb(f1 f2 f3) noconstant keepsingletons numthreads(2) gpubackend(`backend') tolerance(1e-12)
    assert abs(_b[x1]-.75)<1e-10 & abs(_b[x2]+.5)<1e-10 & _b[age]==0
    assert e(converged)==1 & e(precision_certified)==1
    assert e(gpu_used)==("`backend'"=="cuda")
}

clear
set obs 320
generate long group = _n
generate int pattern = floor((_n-1)/8)
generate double x1 = 2*mod(_n-1,2)-1
generate double x2 = 2*mod(floor((_n-1)/2),2)-1
generate double e = 2*mod(floor((_n-1)/4),2)-1
generate byte fe = 0
expand 2+2*mod(pattern,2)
bysort group: generate byte individual = mod(pattern+cond(_n==1,0,cond(_n==2,1,cond(_n==3,3,7))),12)
bysort group: egen double total = total(individual)
bysort group: generate byte size = _N
generate double component = .
generate double y = .
generate double age = .
foreach aggregation in sum mean {
    replace component = total
    if "`aggregation'" == "mean" replace component = total/size
    replace y = .75*x1-.5*x2+component+.125*e
    foreach offset in 0 1e6 1e14 {
        replace age = component+`offset'
        quietly xhdfe y x1 x2 age, absorb(individual fe) group(group) individual(individual) aggregation(`aggregation') dofadjustments(exact) noconstant keepsingletons numthreads(2) gpubackend(`backend') tolerance(1e-12)
        assert abs(_b[x1]-.75)<1e-10 & abs(_b[x2]+.5)<1e-10 & _b[age]==0
        assert e(converged)==1 & e(precision_certified)==1
        assert e(gpu_used)==("`backend'"=="cuda")
    }
}
display "FE_OMISSION_STATA_PASS"
