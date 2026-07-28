# Validation suites

Run these commands from the repository root. The validators use independent
oracles and cross-frontend comparisons that are broader than the ordinary unit
tests.

## Fast contract checks

```bash
bash tests/validation/run_fast.sh
```

The help contract always runs. The retained-sample provenance check also runs
when `XHDFE_PY_MODULE` points to a built `py_hdfe_v11` extension:

```bash
XHDFE_PY_MODULE=build/py_hdfe_v11.so \
  bash tests/validation/run_fast.sh
```

## Core estimator checks

```bash
python tests/validation/VALIDATE_AKM_KSS.py --module-dir build
python tests/validation/VALIDATE_GELBACH.py --module-dir build
python tests/validation/VALIDATE_GELBACH_ADVERSARIAL.py --module-dir build
```

## Extended and cross-frontend checks

The frontend validator requires Stata and R. The remediation coverage suite is
intentionally slow and is reserved for release certification.

```bash
python tests/validation/VALIDATE_GELBACH_FRONTENDS.py --module-dir build
python tests/validation/VALIDATE_GELBACH_PYFIXEST_FEATURES.py --module-dir build
python tests/validation/VALIDATE_GELBACH_REMEDIATION_COVERAGE.py --module-dir build
```
