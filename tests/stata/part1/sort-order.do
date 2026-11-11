noi di as text "xhdfe certification: data order and sort markers are left untouched"

* xhdfe must return the dataset in the order it received it and with the same
* -sortedby- marker, whether the command succeeds or fails. Every scenario runs
* on data in random order (no marker), sorted by the panel and sorted by a
* regressor.
*
* Two facts shape the scenarios. Time-series operators never sort: on a panel
* that is tsset but not in tsset order Stata stops with r(5), and so does
* xhdfe. And the one way a call can touch a sort key is by refilling one of its
* own automatic outputs (__hdfe#__ from absorb(..., savefe), _reghdfe_resid
* from residuals): Stata then re-sorts on exit so that the marker stays true.

capture program drop sortorder_arrange
program define sortorder_arrange
    args order
    if ("`order'" == "random") {
        tempvar u
        quietly gen double `u' = runiform()
        sort `u'
        drop `u'
        local marker : sortedby
        assert "`marker'" == ""
    }
    else if ("`order'" == "panel") {
        sort id year
    }
    else if ("`order'" == "panelx") {
        sort id year x1
    }
    else if ("`order'" == "byx") {
        sort x1
    }
    else {
        di as error "unknown order `order'"
        exit 198
    }
end

* sortorder_check <expected rc> : <command>
* Same observation order and same marker after the command.
capture program drop sortorder_check
program define sortorder_check
    gettoken expect 0 : 0, parse(":")
    gettoken colon 0 : 0, parse(":")
    capture drop __so_position
    quietly gen long __so_position = _n
    local before : sortedby
    capture quietly `0'
    local rc = _rc
    if (`rc' != `expect') {
        di as error "expected rc=`expect', got rc=`rc' from:`0'"
        exit 9
    }
    capture assert __so_position == _n
    if (_rc) {
        di as error "observation order changed by:`0'"
        exit 9
    }
    local after : sortedby
    if ("`after'" != "`before'") {
        di as error "sort marker changed from [`before'] to [`after'] by:`0'"
        exit 9
    }
    drop __so_position
    global SORTORDER_CHECKS = $SORTORDER_CHECKS + 1
end

* sortorder_marker_check : <command>
* Same marker after the command, and the marker is true: a stable re-sort on
* it does not move a single observation.
capture program drop sortorder_marker_check
program define sortorder_marker_check
    gettoken colon 0 : 0, parse(":")
    local before : sortedby
    capture quietly `0'
    if (_rc) {
        di as error "rc=" _rc " from:`0'"
        exit 9
    }
    local after : sortedby
    if ("`after'" != "`before'") {
        di as error "sort marker changed from [`before'] to [`after'] by:`0'"
        exit 9
    }
    tempvar position
    quietly gen long `position' = _n
    sort `before' `position'
    capture assert `position' == _n
    if (_rc) {
        di as error "sort marker [`before'] is not true after:`0'"
        exit 9
    }
    global SORTORDER_CHECKS = $SORTORDER_CHECKS + 1
end

global SORTORDER_CHECKS 0

clear
set seed 20261106
set obs 3000
gen long id = ceil(_n/10)
bysort id: gen int year = 2000 + _n
gen int firm = 1 + mod(id*7 + year*3, 41)
gen str8 sid = "w" + string(id)
gen int cat = mod(_n, 5)
gen double x1 = rnormal()
gen double x2 = rnormal() + .3*x1
gen double z1 = rnormal() + .5*x1
gen double w = 1 + runiform()
gen int fw = 1 + mod(_n, 3)
gen double y = .5*x1 - .2*x2 + .01*id + .1*cat + rnormal()

* --- cross-section commands: no time-series setting on the data -------------
foreach order in random panel byx {
    sortorder_arrange `order'
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm)
    sortorder_check 0 : xhdfe
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) vce(cluster id)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) vce(cluster id firm)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) vce(cluster id#cat)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) vce(robust)
    sortorder_check 0 : xhdfe y x1 x2, absorb(sid year)
    sortorder_check 0 : xhdfe y x1 x2, absorb(sid year) vce(cluster sid)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id#cat firm)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id##c.x2 firm)
    sortorder_check 0 : xhdfe y x1 i.cat c.x1#c.x2, absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2 [aw=w], absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2 [fw=fw], absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2 [pw=w], absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2 if cat != 2 in 100/2900, absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) keepsingletons
    sortorder_check 0 : xhdfe y x1 x2, noabsorb
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) endogenous(x1) instruments(z1)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) savefes(so_fe_) residuals(so_res)
    capture drop so_fe_* so_res
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) tolerancemode(xhdfe-fast) numthreads(2)
    * failed commands must leave the data alone as well
    sortorder_check 198 : xhdfe y x1 x2, absorb(id firm) so_bad_option
    sortorder_check 111 : xhdfe y x1 so_missing_variable, absorb(id firm)
    sortorder_check 2000 : xhdfe y x1 x2 if cat > 99, absorb(id firm)
}

* --- automatic outputs: created on unrelated sort keys ----------------------
foreach order in random panel byx {
    sortorder_arrange `order'
    capture drop __hdfe* _reghdfe_resid
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm, savefe) residuals
    sortorder_check 0 : xhdfe y x1, absorb(id firm, savefe) residuals
}

* --- automatic outputs: the sort key itself is refilled ---------------------
foreach key in "__hdfe1__" "__hdfe1__ __hdfe2__" "_reghdfe_resid" "id _reghdfe_resid" {
    sort `key'
    sortorder_marker_check : xhdfe y x1 x2, absorb(id firm, savefe) residuals
    sort `key'
    sortorder_marker_check : xhdfe y x1, absorb(id firm, savefe) residuals
}
drop __hdfe* _reghdfe_resid

* --- time-series operators on a declared panel ------------------------------
xtset id year
foreach order in panel panelx random byx {
    sortorder_arrange `order'
    * in tsset order the operators work in place; otherwise Stata refuses
    local ts_rc = cond(inlist("`order'", "panel", "panelx"), 0, 5)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm)
    sortorder_check `ts_rc' : xhdfe y L.x1 x2, absorb(id firm)
    sortorder_check `ts_rc' : xhdfe D.y x1 x2, absorb(id firm)
    sortorder_check `ts_rc' : xhdfe y x1 L(0/1).x2, absorb(id firm) vce(cluster id)
    sortorder_check `ts_rc' : xhdfe y x1 x2, absorb(firm#c.L.x2 id)
    sortorder_check `ts_rc' : xhdfe y x1 x2 if L.x1 < ., absorb(id firm)
    sortorder_check 0 : xhdfe y x1 x2, absorb(id firm) endogenous(x1) instruments(z1)
    sortorder_check 0 : xhdfe y x1 x2, absorb(firm) vce(dkraay 2)
}
xtset, clear

* --- group() and individual() on long-format data ---------------------------
clear
set obs 1200
gen long patent = ceil(_n/3)
bysort patent: gen byte slot = _n
gen long inventor = 1 + mod(patent*5 + slot*17, 97)
gen int year = 2000 + mod(patent, 9)
bysort patent: gen double funding = rnormal() if _n == 1
bysort patent: replace funding = funding[1]
bysort patent: gen double citations = funding + .1*year + rnormal() if _n == 1
bysort patent: replace citations = citations[1]
gen long id = patent
gen double x1 = funding
foreach order in random panel byx {
    sortorder_arrange `order'
    sortorder_check 0 : xhdfe citations funding, absorb(year) group(patent)
    sortorder_check 0 : xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor)
    sortorder_check 0 : xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor) aggregation(sum)
}

noi di as text "sort-order checks passed: " as result $SORTORDER_CHECKS
assert $SORTORDER_CHECKS == 3*23 + 3*2 + 4*2 + 4*8 + 3*3
macro drop SORTORDER_CHECKS
