#!/usr/bin/env python3
"""Issue #18: the built Python module under the Windows x64 host x87 regime.

Linux/glibc x86-64 only (skips elsewhere).  Two fresh child interpreters run
the same fits: one at the native control word 0x037F (reference, never
perturbed) and one with the word set to 0x027F (the Windows x64 thread
start-up value) by glibc fesetenv BEFORE the module is imported; OpenMP
workers created later inherit it.  The child proves the regime (read-back and
the 11 low significand bits of sqrtl(2) are zero).  Fits:
  * frequency-weighted OLS on 5 rows with df_resid = 138700..138799 and
    50 ... 2e7 (the Windows Stata probes v1/v2; 2.30.1 refused 28 + 4 there);
  * akm_kss with one control (the internal FWL fit) on panels whose FWL df
    fall by one.
Requires zero refusals in both children, the control word still 0x027F after
every call in the perturbed child, and coefficients, covariance, p-values,
confidence intervals and KSS components equal to the reference child within
1e-12 relative (covariance scaled by sqrt(V_jj V_kk)).  Not bit identity: other
long double accumulators (covariance, FWL) legitimately move the last ulp
with the regime; the bit-level and contract checks of the Student-t entry
points themselves are tests/student_t_x87_guard_contract.cpp.  The 2.30.1
module fails this test (28 + 4 refusals).

--native-host runs the same fits in this process without imposing anything,
for hosts whose own start-up regime is the one to test (python.exe on
Windows x64, where a MinGW-built wheel runs at 53 bits); it requires zero
refusals only.
"""
import argparse
import ctypes
import json
import platform
import subprocess
import sys

NS = list(range(138702, 138802)) + [52, 1002, 50002, 99002, 150002, 500002, 1500002,
                                    5000002, 20000002]


def child(module_dir, cw):
    if cw is None:  # --native-host: the host's own regime, nothing imposed or read
        return fits(module_dir, lambda: None)
    libm = ctypes.CDLL("libm.so.6")

    class FEnv(ctypes.Structure):
        _fields_ = [("raw", ctypes.c_ubyte * 32)]

    def word():
        env = FEnv()
        assert libm.fegetenv(ctypes.byref(env)) == 0
        return env, env.raw[0] | (env.raw[1] << 8)

    env, _ = word()
    env.raw[0], env.raw[1] = cw & 0xFF, cw >> 8
    assert libm.fesetenv(ctypes.byref(env)) == 0
    assert word()[1] == cw

    class LD(ctypes.c_longdouble):
        pass

    libm.sqrtl.restype, libm.sqrtl.argtypes = LD, [ctypes.c_longdouble]
    root = libm.sqrtl(2.0)
    low = int.from_bytes(ctypes.string_at(ctypes.addressof(root), 8), "little") & 0x7FF
    assert (low == 0) == ((cw & 0x0300) == 0x0200), "imposed precision not in effect"
    return fits(module_dir, lambda: word()[1] == cw)


def fits(module_dir, word_intact):
    if module_dir:
        sys.path.insert(0, module_dir)
    import numpy as np
    import py_hdfe_v11 as m

    def hexes(values):
        return [float(v).hex() for v in np.ravel(np.asarray(values, dtype=float))]

    out = {"fits": {}, "refusals": [], "cw_violations": 0}
    x = np.array([[-1.5], [0.25], [0.5], [2.0], [3.75]])
    y = np.array([0.7, -0.2, 1.1, 2.9, 3.3])
    for n in NS:
        q, r = divmod(n, 5)
        w = np.array([q + (1 if i < r else 0) for i in range(5)], dtype=float)
        try:
            reg = m.HdfeRegressor(num_threads=1)
            reg.fit(y, x, None, w, fweights=True)
            out["fits"][str(n)] = {"b": hexes(reg.coef_), "V": hexes(reg.covariance_),
                                   "p": hexes(reg.pvalues_), "ci": hexes(reg.conf_int_)}
        except Exception as exc:  # recorded, never swallowed silently
            out["refusals"].append([n, str(exc)])
        if word_intact() is False:
            out["cw_violations"] += 1
    rng = np.random.default_rng(20261111)
    workers, firms = 9200, 20
    spells = rng.integers(8, 25, workers)
    wk, fk, first_rows = [], [], []
    for i in range(workers):
        moves = 1 if rng.random() < 0.3 else int(rng.integers(2, 4))
        cuts = np.sort(rng.choice(np.arange(1, spells[i]), size=moves - 1, replace=False)) if moves > 1 else []
        chosen = []
        for _ in range(moves):
            while True:
                j = int(rng.integers(0, firms))
                if not chosen or j != chosen[-1]:
                    break
            chosen.append(j)
        if moves == 1 and spells[i] >= 10:
            first_rows.append(len(wk))
        bounds = [0, *cuts, spells[i]]
        for s in range(moves):
            wk += [i] * int(bounds[s + 1] - bounds[s])
            fk += [chosen[s]] * int(bounds[s + 1] - bounds[s])
    wk, fk = np.array(wk), np.array(fk)
    xk = rng.normal(0, 1, wk.size)
    yk = rng.normal(0, 1, workers)[wk] + rng.normal(0, 0.4, firms)[fk] + 0.3 * xk + rng.normal(0, 0.5, wk.size)
    for k in range(8):
        keep = np.ones(wk.size, dtype=bool)
        keep[np.array(first_rows[:k], dtype=int)] = False
        try:
            res = m.akm_kss(yk[keep], wk[keep], fk[keep], X=xk[keep, None], leave_out_level="match",
                            leverages="jla", jla_draws=50, seed=20261110, num_threads=2)
            out["fits"][f"akm{k}"] = {"kss": [float(res["kss"][c]).hex()
                                              for c in ("var_psi", "cov_alpha_psi", "var_alpha")],
                                      "b": hexes(res["beta"])}
        except Exception as exc:
            out["refusals"].append([f"akm{k}", str(exc)])
        if word_intact() is False:
            out["cw_violations"] += 1
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="", help="directory holding py_hdfe_v11 (default: installed)")
    ap.add_argument("--child", type=lambda s: int(s, 0))
    ap.add_argument("--native-host", action="store_true")
    args = ap.parse_args()
    if args.child is not None:
        child(args.module_dir, args.child)
        return 0
    if args.native_host:
        out = child(args.module_dir, None)
        assert not out["refusals"], out["refusals"][:5]
        print("STUDENT_T_X87_NATIVE_HOST_PASS", len(out["fits"]))
        return 0
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        print("STUDENT_T_X87_HOST_REGIME_SKIP")
        return 0
    runs = {}
    for cw in (0x037F, 0x027F):
        proc = subprocess.run([sys.executable, __file__, "--module-dir", args.module_dir,
                               "--child", hex(cw)], capture_output=True, text=True, check=True)
        runs[cw] = json.loads(proc.stdout.splitlines()[-1])
    ref, pc53 = runs[0x037F], runs[0x027F]
    assert not ref["refusals"], ref["refusals"][:5]
    assert not pc53["refusals"], pc53["refusals"][:5]
    assert pc53["cw_violations"] == 0 and ref["cw_violations"] == 0
    assert set(ref["fits"]) == set(pc53["fits"])
    worst = 0.0
    for key, want in ref["fits"].items():
        got = pc53["fits"][key]
        for field, values in want.items():
            a = [float.fromhex(v) for v in values]
            b = [float.fromhex(v) for v in got[field]]
            if field == "V":
                k = int(round(len(a) ** 0.5))
                scale = [abs(a[i // k * (k + 1)] * a[i % k * (k + 1)]) ** 0.5 for i in range(len(a))]
            else:
                scale = [max(abs(v), 1e-300) for v in a]
            for x, y, s in zip(a, b, scale):
                err = abs(x - y) / s if x != y else 0.0
                worst = max(worst, err)
                assert err <= 1e-12, f"{key} {field}: {x!r} vs {y!r} under the 53-bit host regime"
    print("STUDENT_T_X87_HOST_REGIME_PASS", len(ref["fits"]), "max_rel_diff=%.3g" % worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
