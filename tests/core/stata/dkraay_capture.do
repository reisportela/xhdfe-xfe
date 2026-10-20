* Public auto fixture for the independent observed-period DK oracle.
version 16
clear all
sysuse auto, clear
bysort turn: generate t = _n
tsset turn t
drop if missing(rep78)
ivreghdfe price weight length, absorb(turn t) dkraay(2)
matrix rb = e(b)
matrix rV = e(V)
generate byte reference_sample = e(sample)
_up_xhdfe price weight length, absorb(turn t) vce(dkraay 2)
matrix cb = e(b)
matrix cV = e(V)
generate byte selected = e(sample)
assert selected == reference_sample
file open matrices using "matrices.txt", write text
foreach name in rb rV cb cV {
    file write matrices "`name' " (rowsof(`name')) " " (colsof(`name')) _n
    forvalues i = 1/`=rowsof(`name')' {
        forvalues j = 1/`=colsof(`name')' {
            file write matrices %24.17e (`name'[`i',`j']) " "
        }
        file write matrices _n
    }
}
file close matrices
keep price weight length turn t selected
save "public_auto_fixture.dta"
display "CORE_DK_CAPTURE_PASS"
