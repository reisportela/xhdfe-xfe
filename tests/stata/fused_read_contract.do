* The fused input read, the column readers and the parallel reader must hand
* the estimator the same inputs. This do-file runs a fixed set of fits with
* whichever reader the environment selects (XHDFE_FUSED_READ=0 keeps the column
* readers, XHDFE_PARALLEL_READ=0 the serial fused reader, XHDFE_PARALLEL_READ=1
* forces the parallel reader) and the thread count of its second argument
* (default 1), and writes every result in hexadecimal. run_stata_tests.sh runs
* it once per reader and requires the files of a pair with the same thread
* count to be identical byte for byte, together with the input fingerprints
* the plugin prints under XHDFE_PROFILE_CPU=1.
version 16
clear all
set more off
args label threads
if ("`label'" == "") local label fused
if ("`threads'" == "") local threads 1
global FRC_THREADS `threads'
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe

* frc_fit <handle> <tag> : <command>     the command must succeed
capture program drop frc_fit
program define frc_fit
    gettoken handle 0 : 0
    gettoken tag 0 : 0, parse(" :")
    gettoken colon 0 : 0, parse(":")
    capture noisily `0' numthreads($FRC_THREADS) gpubackend(cpu)
    if (_rc) {
        di as error "`tag': rc=" _rc " from:`0'"
        exit 9
    }
    file write `handle' "`tag' N=" %21x (e(N)) " df_r=" %21x (e(df_r)) " rss=" %21x (e(rss)) _n
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    forvalues j = 1/`k' {
        file write `handle' "`tag' b`j'=" %21x (`b'[1,`j']) _n
        forvalues i = 1/`j' {
            file write `handle' "`tag' V`i'_`j'=" %21x (`V'[`i',`j']) _n
        }
    }
end

* frc_output <handle> <tag> <varlist>    moments of variables the fit created
capture program drop frc_output
program define frc_output
    gettoken handle 0 : 0
    gettoken tag 0 : 0
    foreach v of varlist `0' {
        quietly summarize `v'
        file write `handle' "`tag' `v' n=" %21x (r(N)) " sum=" %21x (r(sum)) ///
            " min=" %21x (r(min)) " max=" %21x (r(max)) _n
    }
end

* frc_error <handle> <tag> : <command>   return code and message text
capture program drop frc_error
program define frc_error
    gettoken handle 0 : 0
    gettoken tag 0 : 0, parse(" :")
    gettoken colon 0 : 0, parse(":")
    tempfile message
    tempname capture_log reader
    quietly log using `"`message'"', text replace name(`capture_log')
    capture noisily `0' numthreads($FRC_THREADS) gpubackend(cpu)
    local rc = _rc
    quietly log close `capture_log'
    if (`rc' == 0) {
        di as error "`tag': expected an error from:`0'"
        exit 9
    }
    file write `handle' "`tag' rc=`rc'" _n
    file open `reader' using `"`message'"', read text
    file read `reader' line
    while (r(eof) == 0) {
        local line = strtrim(`"`line'"')
        * drop the log's own header and the ado's timing lines (profile switch)
        if (`"`line'"' != "" & !regexm(`"`line'"', "^(name|log|log type|opened on|closed on):") ///
            & !regexm(`"`line'"', "^-+$") & !regexm(`"`line'"', "^ado_profile ")) {
            file write `handle' `"`tag' message=`line'"' _n
        }
        file read `reader' line
    }
    file close `reader'
end

tempname OUT
file open `OUT' using "fused_read_contract_`label'.txt", write text replace

* --- worker-year panel -------------------------------------------------------
set seed 20261106
set obs 6000
gen long id = ceil(_n/6)
bysort id: gen int year = 2000 + _n
gen int firm = 1 + mod(id*7 + year*3, 53)
gen double bigid = id + 2^40
gen double halfid = firm + .5
gen str6 sfirm = "f" + string(firm)
gen int cl = 1 + mod(id, 37)
forvalues j = 1/30 {
    gen double x`j' = rnormal()
}
gen double z1 = rnormal() + .5*x1
gen double z2 = rnormal() + .3*x1
gen double w = 1 + runiform()
gen int fw = 1 + mod(_n, 3)
gen double y = .5*x1 - .2*x2 + .1*x3 + .01*id + .02*firm + rnormal()

* identifiers that fit int32
frc_fit `OUT' plain : xhdfe y x1 x2 x3, absorb(id firm)
frc_fit `OUT' one_fe : xhdfe y x1 x2 x3, absorb(id)
frc_fit `OUT' noabsorb : xhdfe y x1 x2 x3, noabsorb
frc_fit `OUT' cluster_on_fe : xhdfe y x1 x2 x3, absorb(id firm) vce(cluster id)
frc_fit `OUT' cluster_other : xhdfe y x1 x2 x3, absorb(id firm) vce(cluster cl)
frc_fit `OUT' cluster_two : xhdfe y x1 x2 x3, absorb(id firm) vce(cluster id cl)
frc_fit `OUT' robust : xhdfe y x1 x2 x3, absorb(id firm) vce(robust)
frc_fit `OUT' wide_x : xhdfe y x1-x30, absorb(id firm)

* identifiers outside exact int32: the column reader keeps them
frc_fit `OUT' big_first : xhdfe y x1 x2 x3, absorb(bigid firm)
frc_fit `OUT' half_second : xhdfe y x1 x2 x3, absorb(id halfid)
frc_fit `OUT' all_wide : xhdfe y x1 x2 x3, absorb(bigid halfid) vce(cluster bigid)
frc_fit `OUT' cluster_wide : xhdfe y x1 x2 x3, absorb(id firm) vce(cluster halfid)
frc_fit `OUT' cluster_mixed : xhdfe y x1 x2 x3, absorb(id firm) vce(cluster cl halfid)

* weights, slopes, instruments, strings, interactions, sample restrictions
frc_fit `OUT' aweight : xhdfe y x1 x2 x3 [aw=w], absorb(id firm)
frc_fit `OUT' fweight : xhdfe y x1 x2 x3 [fw=fw], absorb(id firm)
frc_fit `OUT' pweight : xhdfe y x1 x2 x3 [pw=w], absorb(id firm) vce(cluster id)
frc_fit `OUT' slope : xhdfe y x1 x2 x3, absorb(id firm##c.z1)
frc_fit `OUT' slope_twice : xhdfe y x1 x2 x3, absorb(id firm##c.z1 firm#c.z2)
frc_fit `OUT' slope_wide : xhdfe y x1 x2 x3, absorb(bigid firm##c.z1)
frc_fit `OUT' slope_weight : xhdfe y x1 x2 x3 [aw=w], absorb(id firm#c.z2) vce(cluster cl)
frc_fit `OUT' iv : xhdfe y x1 x2 x3, absorb(id firm) endogenous(x1) instruments(z1 z2)
frc_fit `OUT' iv_cluster : xhdfe y x1 x2 x3 [aw=w], absorb(id firm) endogenous(x1) instruments(z1 z2) vce(cluster cl)
frc_fit `OUT' string_fe : xhdfe y x1 x2 x3, absorb(sfirm id) vce(cluster sfirm)
frc_fit `OUT' interaction : xhdfe y x1 x2 x3, absorb(firm#year id)
frc_fit `OUT' gaps : xhdfe y x1 x2 x3 if mod(_n, 3) != 0 in 50/5900, absorb(id firm)
frc_fit `OUT' condition : xhdfe y x1 x2 x3 if x1 > 0, absorb(id firm) vce(cluster cl)
frc_fit `OUT' fast_mode : xhdfe y x1 x2 x3, absorb(id firm) tolerancemode(xhdfe-fast)

* outputs written back after the fit
frc_fit `OUT' outputs : xhdfe y x1 x2 x3, absorb(id firm) savefes(frc_fe_) residuals(frc_res)
frc_output `OUT' outputs frc_fe_* frc_res
drop frc_fe_* frc_res
frc_fit `OUT' outputs_wide : xhdfe y x1 x2 x3, absorb(bigid firm) savefes(frc_fe_) residuals(frc_res)
frc_output `OUT' outputs_wide frc_fe_* frc_res
drop frc_fe_* frc_res

* --- group() and individual() ------------------------------------------------
clear
set obs 2400
gen long patent = ceil(_n/3)
bysort patent: gen byte slot = _n
gen long inventor = 1 + mod(patent*5 + slot*17, 197)
gen int year = 2000 + mod(patent, 9)
bysort patent: gen double funding = rnormal() if _n == 1
bysort patent: replace funding = funding[1]
bysort patent: gen double citations = funding + .1*year + rnormal() if _n == 1
bysort patent: replace citations = citations[1]
frc_fit `OUT' group : xhdfe citations funding, absorb(year) group(patent)
frc_fit `OUT' group_individual : xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor)
frc_fit `OUT' group_individual_sum : xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor) aggregation(sum)

* invalid individual identifiers: same code and same words from either reader
gen double inventor_negative = inventor
replace inventor_negative = -3 in 7
frc_error `OUT' individual_negative : xhdfe citations funding, absorb(year inventor_negative) group(patent) individual(inventor_negative)
gen double inventor_fraction = inventor
replace inventor_fraction = 4.25 in 2000
frc_error `OUT' individual_fraction : xhdfe citations funding, absorb(year inventor_fraction) group(patent) individual(inventor_fraction)
gen double inventor_huge = inventor
replace inventor_huge = 2^60 in 1200
frc_error `OUT' individual_huge : xhdfe citations funding, absorb(year inventor_huge) group(patent) individual(inventor_huge)

* --- row blocks of the parallel reader ----------------------------------------
* The parallel reader selects and reads in blocks of 16,384 rows, with at most
* one thread per block: samples of exactly one block (serial) and of one row
* more (with the whole range selected and as an in range), several blocks, an
* in range in the middle, a selected prefix followed by exclusions, scattered
* exclusions, an identifier that leaves int32 only in a later block, an empty
* sample and, expanded to five blocks, a full team of four threads. The
* automatic fits at the end check that the threading switches govern the read
* as they govern the fit.
clear
set seed 20261107
set obs 40000
gen long id = ceil(_n/8)
bysort id: gen int year = 2000 + _n
gen int firm = 1 + mod(id*7 + year*3, 97)
gen double lateid = cond(_n > 20000, id + 2^40, id)
forvalues j = 1/3 {
    gen double x`j' = rnormal()
}
gen double y = .5*x1 - .2*x2 + .1*x3 + .01*id + .02*firm + rnormal()
frc_fit `OUT' blocks_all : xhdfe y x1 x2 x3, absorb(id firm)
frc_fit `OUT' block_exact_in : xhdfe y x1 x2 x3 in 1/16384, absorb(id firm)
frc_fit `OUT' block_plus_one_in : xhdfe y x1 x2 x3 in 1/16385, absorb(id firm)
frc_fit `OUT' in_middle : xhdfe y x1 x2 x3 in 9001/31000, absorb(id firm) vce(cluster id)
frc_fit `OUT' prefix_then_gaps : xhdfe y x1 x2 x3 if _n <= 20000 | mod(_n, 5) != 0, absorb(id firm)
frc_fit `OUT' scattered : xhdfe y x1 x2 x3 if mod(_n, 7) != 3, absorb(id firm) vce(cluster firm)
frc_fit `OUT' late_wide : xhdfe y x1 x2 x3, absorb(lateid firm)
frc_error `OUT' empty : xhdfe y x1 x2 x3 if x1 > 100, absorb(id firm)
preserve
quietly keep in 1/16384
frc_fit `OUT' block_exact : xhdfe y x1 x2 x3, absorb(id firm)
restore
preserve
quietly keep in 1/16385
frc_fit `OUT' block_plus_one : xhdfe y x1 x2 x3, absorb(id firm)
restore
preserve
quietly expand 2
frc_fit `OUT' blocks_five : xhdfe y x1 x2 x3, absorb(id firm)
restore
* Automatic team (numthreads(0)) without absorbed fixed effects, so that no
* problem-size cap applies: one thread per 10,000 rows gives four threads and a
* read of three blocks by three; maxthreads(2) gives two; one thread per 500,000
* rows gives one and a serial read.
global FRC_THREADS 0
frc_fit `OUT' auto_rows10k : xhdfe y x1 x2 x3, noabsorb targetrowsperthread(10000)
frc_fit `OUT' auto_rows10k_max2 : xhdfe y x1 x2 x3, noabsorb targetrowsperthread(10000) maxthreads(2)
frc_fit `OUT' auto_rows500k : xhdfe y x1 x2 x3, noabsorb targetrowsperthread(500000)
* e(threads_used) reports the read when the fit is smaller. The fit's 5,000 rows
* are below minparallelrows(20000) and resolve one thread. The sample selection
* scans the whole 40,000-row range, which the same rule resolves to four threads
* (one per 10,000 rows) and the reader runs on three, one per block: in the
* parallel run e(threads_used) and e(parallel_workers_active) are 3; in every
* other run the selection is serial (parallel read switched off, fused read off,
* or 160,000 values below the default threshold) and they are 1.
frc_fit `OUT' auto_in_small : xhdfe y x1 x2 x3 in 1/5000, noabsorb targetrowsperthread(10000)
local selection_team = cond("`label'" == "parallel4", 3, 1)
assert e(threads_effective) == 1
assert e(threads_used) == `selection_team'
assert e(parallel_workers_active) == `selection_team'
global FRC_THREADS `threads'

* group() and individual() over several blocks, then an invalid individual
* identifier only in a later block: same code and same words from every reader
clear
set seed 20261108
set obs 36000
gen long patent = ceil(_n/3)
bysort patent: gen byte slot = _n
gen long inventor = 1 + mod(patent*5 + slot*17, 1997)
gen int year = 2000 + mod(patent, 9)
bysort patent: gen double funding = rnormal() if _n == 1
bysort patent: replace funding = funding[1]
bysort patent: gen double citations = funding + .1*year + rnormal() if _n == 1
bysort patent: replace citations = citations[1]
frc_fit `OUT' group_individual_blocks : xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor)
gen double inventor_late = inventor
replace inventor_late = -3 in 30001
frc_error `OUT' individual_late_negative : xhdfe citations funding, absorb(year inventor_late) group(patent) individual(inventor_late)

* --- alias variables (Stata 18+) -----------------------------------------------
* A variable aliased from another frame (fralias) is read through that frame:
* the plugin keeps its serial reader for any call that receives one, and the
* results agree with every other reader.
if (c(stata_version) >= 18) {
    clear
    set seed 20261109
    set obs 6000
    gen long key = _n
    gen long id = ceil(_n/6)
    bysort id: gen int year = 2000 + _n
    gen int firm = 1 + mod(id*7 + year*3, 53)
    gen double x1 = rnormal()
    gen double y = .5*x1 + .01*id + .02*firm + rnormal()
    capture frame drop frc_other
    frame create frc_other
    frame frc_other {
        quietly set obs 6000
        gen long key = _n
        gen double ax = sin(key) + key/1000
        gen long aid = 1 + mod(key, 211)
    }
    quietly frlink 1:1 key, frame(frc_other)
    fralias add ax aid, from(frc_other)
    frc_fit `OUT' alias_regressor : xhdfe y x1 ax, absorb(id firm)
    frc_fit `OUT' alias_fe : xhdfe y x1 ax, absorb(aid firm)
    clear
    frame drop frc_other
    di "FUSED_READ_CONTRACT_ALIAS ran"
}
else {
    di "FUSED_READ_CONTRACT_ALIAS skipped: Stata `c(stata_version)' has no alias variables"
}

file close `OUT'
di "FUSED_READ_CONTRACT_PASS `label'"
