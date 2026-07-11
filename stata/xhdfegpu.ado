*! version 2.18.0  11jul2026
*! xhdfegpu: build and install a CUDA (GPU) xhdfe/xfe plugin for this machine.
*!
*! net install ships CPU-only plugins. On a machine with an NVIDIA GPU, run
*! xhdfegpu once to compile a CUDA plugin for the local architecture and
*! install it OVER the CPU plugin (same file name, same adopath location).
*!
*! Sources come from the self-contained offline distribution zip:
*!   1. by default xhdfegpu downloads it from the online site;
*!   2. if there is no internet, download it manually and pass it with zip();
*!   3. or point source() at an already-extracted xhdfe-src directory.
*! The zip vendors Eigen / pybind11 / stplugin, so the build needs no network
*! (only a CUDA toolkit and a C++ compiler on this machine).

program define xhdfegpu, rclass
    version 14.0
    syntax [, ZIP(string) SOURCE(string) ARCH(integer 0) KEEP FORCE ///
              URL(string) ]

    local ziturl "`url'"
    if ("`ziturl'" == "") {
        local ziturl "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata/xhdfe-src.zip"
    }

    * ---- 1. Platform: the GPU build path is Linux (bash build scripts) ------
    if ("`c(os)'" != "Unix") {
        di as err "xhdfegpu: GPU plugin builds are supported on Linux only."
        di as err "On Windows/macOS, build from source per docs/gpu.md, or use the CPU plugin."
        exit 198
    }

    * ---- 2. GPU gate: only proceed if an NVIDIA GPU is present --------------
    tempfile smi
    capture shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits > "`smi'" 2>/dev/null
    local cc ""
    capture file open g using "`smi'", read
    if (!_rc) {
        file read g line
        while (r(eof) == 0 & "`cc'" == "") {
            local line = strtrim("`line'")
            if regexm("`line'", "([0-9]+)\.([0-9]+)") {
                local cc = regexs(1) + regexs(2)
            }
            file read g line
        }
        file close g
    }
    if (`arch' > 0) local cc = `arch'
    if ("`cc'" == "") {
        di as err "xhdfegpu: no NVIDIA GPU detected on this machine (nvidia-smi found no compute capability)."
        di as err "xhdfegpu only builds a CUDA plugin when a GPU is present; the CPU plugin from net install is already active."
        exit 198
    }
    di as txt "xhdfegpu: NVIDIA GPU detected, compute capability " as res "sm_`cc'" as txt "."

    * ---- 3. Toolchain check (via marker files; shell _rc is unreliable) -----
    tempfile which
    capture shell { command -v nvcc; command -v c++ || command -v g++; } > "`which'" 2>/dev/null
    local havecc 0
    local havecxx 0
    tempname wf
    capture file open `wf' using "`which'", read
    if (!_rc) {
        file read `wf' l
        while (r(eof) == 0) {
            if (strpos("`l'", "nvcc") > 0) local havecc 1
            if (regexm("`l'", "(c\+\+|g\+\+)$")) local havecxx 1
            file read `wf' l
        }
        file close `wf'
    }
    if (!`havecc') {
        di as err "xhdfegpu: the CUDA toolkit (nvcc) was not found on PATH."
        di as err "Install the NVIDIA CUDA toolkit (and driver), then re-run xhdfegpu."
        exit 198
    }
    if (!`havecxx') {
        di as err "xhdfegpu: a C++ compiler (g++/c++) was not found on PATH."
        exit 198
    }

    * ---- 4. Obtain the source tree -----------------------------------------
    local work "`c(tmpdir)'/xhdfegpu_build"
    capture shell rm -rf "`work'" 2>/dev/null
    capture mkdir "`work'"
    local srcdir ""

    if ("`source'" != "") {
        * (c) already-extracted source directory
        capture confirm file "`source'/stata/tools/build-plugin.sh"
        if (_rc) {
            di as err "xhdfegpu: source(`source') does not look like an extracted xhdfe-src tree"
            di as err "(expected `source'/stata/tools/build-plugin.sh)."
            exit 198
        }
        local srcdir "`source'"
        di as txt "xhdfegpu: building from the source directory you supplied."
    }
    else {
        local zipfile "`zip'"
        if ("`zipfile'" == "") {
            * (a) try to download the distribution zip from the online site
            local zipfile "`work'/xhdfe-src.zip"
            di as txt "xhdfegpu: downloading the source distribution..."
            capture copy "`ziturl'" "`zipfile'", replace
            if (_rc) {
                * try a shell downloader as a fallback
                capture shell curl -fsSL "`ziturl'" -o "`zipfile'" 2>/dev/null
                if (_rc | !fileexists("`zipfile'")) capture shell wget -q "`ziturl'" -O "`zipfile'" 2>/dev/null
            }
            if (!fileexists("`zipfile'")) {
                di as err "xhdfegpu: could not download the source distribution (no internet, or the site is unreachable)."
                di as txt ""
                di as txt "To build the GPU plugin on a machine without internet access:"
                di as txt `"  1. On any machine, download the self-contained source zip:"'
                di as txt `"       {browse "`ziturl'"}"'
                di as txt `"  2. Copy it to this machine, then re-run pointing xhdfegpu at it:"'
                di as txt `"       . xhdfegpu, zip("/path/to/xhdfe-src.zip")"'
                di as txt `"     (or extract it first and use source("/path/to/xhdfe-src"))."'
                di as txt "The zip is self-contained: no further downloads are needed to build."
                exit 601
            }
        }
        else {
            * (b) local zip supplied
            capture confirm file "`zipfile'"
            if (_rc) {
                di as err "xhdfegpu: zip(`zipfile') not found."
                exit 601
            }
        }
        di as txt "xhdfegpu: unpacking the source distribution..."
        capture shell cd "`work'" && unzip -q -o "`zipfile'" 2>/dev/null
        if (_rc | !fileexists("`work'/xhdfe-src/stata/tools/build-plugin.sh")) {
            di as err "xhdfegpu: could not unpack the source zip (need 'unzip' on PATH)."
            exit 198
        }
        local srcdir "`work'/xhdfe-src"
    }

    * note the source version (for user awareness)
    capture confirm file "`srcdir'/VERSION"
    if (!_rc) {
        tempname vf
        file open `vf' using "`srcdir'/VERSION", read
        file read `vf' zver
        file close `vf'
        di as txt "xhdfegpu: building xhdfe/xfe " as res "`zver'" as txt " from source."
    }

    * ---- 5. Build the CUDA plugins -----------------------------------------
    di as txt "xhdfegpu: compiling the CUDA plugin for sm_`cc' (this takes a few minutes)..."
    tempfile blog
    capture shell cd "`srcdir'" && XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=`cc' bash stata/tools/build-plugin.sh --linux --openmp > "`blog'" 2>&1
    local rc_xhdfe = _rc
    capture shell cd "`srcdir'" && XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=`cc' bash stata/tools/build-xfe-plugin.sh --linux --openmp >> "`blog'" 2>&1
    local rc_xfe = _rc
    if (`rc_xhdfe' | !fileexists("`srcdir'/stata/xhdfe.plugin")) {
        di as err "xhdfegpu: the CUDA build failed. Last lines of the build log:"
        capture type "`blog'", lines(20)
        exit 198
    }

    * ---- 6. Install over the CPU plugin (atomic; only on success) ----------
    quietly findfile xhdfe.plugin
    local dest "`r(fn)'"
    capture copy "`srcdir'/stata/xhdfe.plugin" "`dest'", replace public
    if (_rc) {
        di as err "xhdfegpu: built the plugin but could not write to `dest' (permissions?)."
        exit 603
    }
    local didx 1
    local didxfe 0
    if (`rc_xfe' == 0 & fileexists("`srcdir'/stata/xfe.plugin")) {
        capture findfile xfe.plugin
        if (!_rc) {
            capture copy "`srcdir'/stata/xfe.plugin" "`r(fn)'", replace public
            if (!_rc) local didxfe 1
        }
    }

    if ("`keep'" == "") capture shell rm -rf "`work'" 2>/dev/null

    * ---- 7. Report ---------------------------------------------------------
    di as txt ""
    di as txt "xhdfegpu: {res}CUDA plugin installed{txt} (sm_`cc') over the CPU plugin:"
    di as txt "  xhdfe.plugin -> " as res "`dest'"
    if (`didxfe') di as txt "  xfe.plugin   -> installed"
    di as txt ""
    di as txt "Restart Stata (or type {stata discard}) to load the new plugin, then request the GPU:"
    di as txt `"  . xhdfe y x, absorb(fe1 fe2) gpubackend(cuda)"'
    di as txt "  . display e(gpu_used)      // should be 1"
    return local plugin "`dest'"
    return local arch "sm_`cc'"
    return scalar gpu_ready = 1
end
