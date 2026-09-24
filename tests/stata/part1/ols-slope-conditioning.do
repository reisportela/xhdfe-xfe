version 16
args stage backend
clear all
set more off
adopath ++ "`stage'"

foreach n in 1024 8192 {
    clear
    set obs `n'
    generate long group = _n
    generate long fe = floor((_n-1)/16)
    generate long cluster = floor((_n-1)/2)
    generate double x1 = 2*mod(_n-1,2)-1
    generate double b = 2*mod(floor((_n-1)/2),2)-1
    generate double e = 2*mod(floor((_n-1)/4),2)-1
    generate double x2 = .
    generate double y = .
    foreach power in 16 18 19 20 21 {
        scalar delta = 3*2^(-`power')
        replace x2 = x1+delta*b
        replace y = .75*x1-.5*x2+.125*e
        foreach kind in ols ordinary_fe grouped_sum grouped_mean {
            if "`backend'" == "cuda" & "`kind'" == "ols" continue
            preserve
            local options ""
            scalar K = 2
            if "`kind'" == "ordinary_fe" {
                local options "absorb(fe)"
                scalar K = 2+`n'/16
            }
            if inlist("`kind'","grouped_sum","grouped_mean") {
                expand 2
                bysort group: generate byte individual = _n
                local aggregation = subinstr("`kind'","grouped_","",1)
                local options "absorb(individual) group(group) individual(individual) aggregation(`aggregation') dofadjustments(exact)"
                scalar K = 3
            }
            foreach vce in unadjusted robust cluster {
                local variance "`vce'"
                if "`vce'" == "cluster" local variance "cluster cluster"
                quietly xhdfe y x1 x2, `options' noconstant keepsingletons numthreads(2) gpubackend(`backend') tolerance(1e-12) vce(`variance')
                assert abs(_b[x1]-.75)<1e-10 & abs(_b[x2]+.5)<1e-10
                assert abs(e(rss)-`n'/64)<1e-9
                assert e(converged)==1 & e(precision_certified)==1
                assert e(gpu_used)==("`backend'"=="cuda")
                scalar scale = 1/(64*(`n'-K)*delta^2)
                if "`vce'" == "cluster" scalar scale = ((`n'-1)/(`n'-K))*((`n'/2)/(`n'/2-1))/(32*`n'*delta^2)
                matrix expected = (1,-1\-1,1)*scale
                if "`vce'" != "cluster" matrix expected[1,1] = scale*(1+delta^2)
                matrix difference = e(V)-expected
                assert abs(difference[1,1])/scale<1e-8 & abs(difference[2,2])/scale<1e-8
                assert abs(difference[1,2])/scale<1e-8 & abs(difference[2,1])/scale<1e-8
            }
            restore
        }
    }
}
display "SLOPE_CONDITIONING_STATA_PASS"
