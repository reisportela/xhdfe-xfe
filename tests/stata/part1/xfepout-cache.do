* xfepout absorption/profile cache plumbing and documented aliases.
capture program drop xfe_cache_file_receipt
program define xfe_cache_file_receipt, rclass
    syntax , FILE(string) HASHRECEIPT(string) STAMP(string)
    confirm file `"`file'"'
    capture erase `"`hashreceipt'"'
    capture erase `"`stamp'"'
    quietly shell /usr/bin/sha256sum -- "`file'" > "`hashreceipt'"
    quietly shell /usr/bin/stat --format=%y -- "`file'" > "`stamp'"
    confirm file `"`hashreceipt'"'
    confirm file `"`stamp'"'
    file open __xfe_hash using `"`hashreceipt'"', read text
    file read __xfe_hash hash_line
    file close __xfe_hash
    gettoken hash_value hash_rest : hash_line
    if (strlen("`hash_value'") != 64 | !regexm("`hash_value'", "^[0-9a-f]+$")) {
        di as err "xfepout cache test could not obtain a valid SHA-256 receipt"
        exit 459
    }
    file open __xfe_stamp using `"`stamp'"', read text
    file read __xfe_stamp stamp_line
    file close __xfe_stamp
    if (strlen(strtrim(`"`stamp_line'"')) < 30 | strpos(`"`stamp_line'"', ".") == 0) {
        di as err "xfepout cache test could not obtain nanosecond mtime"
        exit 459
    }
    quietly checksum `"`file'"'
    if (r(filelen) <= 0) exit 459
    return local sha256 "`hash_value'"
    return local mtime `"`stamp_line'"'
    return scalar bytes = r(filelen)
end

clear
set seed 20260901
set obs 6000
gen long fe1 = ceil(_n / 30)
gen long fe2 = mod(_n - 1, 37) + 1
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double y = .7*x1 - .3*x2 + fe1/100 + fe2/70 + rnormal()
replace fe1 = 999 if _n == _N
replace fe2 = 999 if _n == _N
sort fe1 fe2

tempfile abs fe mob legacy corrupt
tempfile hash_write stat_write hash_read stat_read hash_auto stat_auto
tempfile hash_miss_before stat_miss_before hash_miss_after stat_miss_after
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(w_) sample(w_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    abscachefile(`"`abs'"') absorptioncachemode(write) ///
    fescache(`"`fe'"') fecachemode(write) ///
    mobilityfile(`"`mob'"') mobilityprofile
assert e(converged) == 1
assert e(gpu_used) == 0
confirm file `"`abs'"'
confirm file `"`fe'"'
confirm file `"`mob'"'
xfe_cache_file_receipt, file(`"`abs'"') ///
    hashreceipt(`"`hash_write'"') stamp(`"`stat_write'"')
local write_sha "`r(sha256)'"
local write_mtime `"`r(mtime)'"'
local write_bytes = r(bytes)

quietly xfepout y x1 x2, absorb(fe1 fe2) generate(r_) sample(r_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    absorptioncache(`"`abs'"') abscachemode(read) ///
    festructurecachefile(`"`fe'"') fescachemode(read) ///
    mobilityprofilefile(`"`mob'"')
assert e(converged) == 1
foreach v in y x1 x2 {
    assert r_`v' == w_`v'
}
xfe_cache_file_receipt, file(`"`abs'"') ///
    hashreceipt(`"`hash_read'"') stamp(`"`stat_read'"')
assert r(bytes) == `write_bytes'
if ("`r(sha256)'" != "`write_sha'" | `"`r(mtime)'"' != `"`write_mtime'"') {
    di as err "xfepout read cache hit rewrote its payload"
    exit 459
}

quietly xfepout y x1 x2, absorb(fe1 fe2) generate(h_) sample(h_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    absorptioncache(`"`abs'"') abscachemode(auto) ///
    fescache(`"`fe'"') fecachemode(read)
assert e(converged) == 1
foreach v in y x1 x2 {
    assert h_`v' == w_`v'
}
xfe_cache_file_receipt, file(`"`abs'"') ///
    hashreceipt(`"`hash_auto'"') stamp(`"`stat_auto'"')
assert r(bytes) == `write_bytes'
if ("`r(sha256)'" != "`write_sha'" | `"`r(mtime)'"' != `"`write_mtime'"') {
    di as err "xfepout auto cache hit rewrote its payload"
    exit 459
}
di as result "XFEPOUT_CACHE_HIT_RECEIPT_PASS"

file open legacy_handle using `"`legacy'"', write text replace
file write legacy_handle "xhdfe_absorption_cache_v2" _n "legacy"
file close legacy_handle
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(a_) sample(a_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    abscachefile(`"`legacy'"') absorptioncachemode(auto)
assert e(converged) == 1
foreach v in y x1 x2 {
    assert a_`v' == w_`v'
}

file open corrupt_handle using `"`corrupt'"', write text replace
file write corrupt_handle "not-a-cache"
file close corrupt_handle
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(c_) sample(c_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    absorptioncache(`"`corrupt'"') abscachemode(read)
assert e(converged) == 1
foreach v in y x1 x2 {
    assert c_`v' == w_`v'
}

xfe_cache_file_receipt, file(`"`abs'"') ///
    hashreceipt(`"`hash_miss_before'"') stamp(`"`stat_miss_before'"')
local miss_before_sha "`r(sha256)'"
local miss_before_mtime `"`r(mtime)'"'
replace y = y + 1e-6 if _n == 1
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(n_) sample(n_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu)
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(s_) sample(s_s) ///
    tolerance(1e-12) maxiter(100000) numthreads(2) gpubackend(cpu) ///
    absorptioncache(`"`abs'"') abscachemode(auto)
assert e(converged) == 1
foreach v in y x1 x2 {
    assert s_`v' == n_`v'
}
xfe_cache_file_receipt, file(`"`abs'"') ///
    hashreceipt(`"`hash_miss_after'"') stamp(`"`stat_miss_after'"')
if ("`r(sha256)'" == "`miss_before_sha'" | ///
    `"`r(mtime)'"' == `"`miss_before_mtime'"') {
    di as err "xfepout signature miss did not rewrite the cache"
    exit 459
}
replace y = y - 1e-6 if _n == 1
di as result "XFEPOUT_CACHE_SIGNATURE_MISS_REWRITE_PASS"

foreach mode in read write auto {
    capture noisily xfepout y x1 x2, absorb(fe1 fe2) generate(q_) ///
        abscachemode(`mode')
    assert _rc == 198
}
capture noisily xfepout y x1 x2, absorb(fe1 fe2) generate(q_) ///
    absorptioncachemode(auto)
assert _rc == 198
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(off_) ///
    abscachemode(off) gpubackend(cpu)
assert e(converged) == 1

local mob_abs `"`mob'.absorption_cache"'
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(mc_) ///
    mobfile(`"`mob'"') abscachemode(write) gpubackend(cpu)
assert e(converged) == 1
confirm file `"`mob_abs'"'
erase `"`mob_abs'"'

capture noisily xfepout y x1 x2, absorb(fe1 fe2) generate(q_) ///
    absorptioncache(`"`abs'"') abscachefile(`"`abs'"')
assert _rc == 198
capture noisily xfepout y x1 x2, absorb(fe1 fe2) generate(q_) ///
    mobfile(`"`mob'"') mobilityfile(`"`mob'"')
assert _rc == 198

program drop xfe_cache_file_receipt
di as result "XFEPOUT_CACHE_PROFILE_PASS"
