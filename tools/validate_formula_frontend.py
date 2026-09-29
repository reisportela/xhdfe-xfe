#!/usr/bin/env python3
"""Run the same complete formula contract against source or installed packages."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("test_formula_frontend", "tests/test_formula_frontend.py", 63),
    ("test_maketables_integration", "tests/test_maketables_integration.py", 32),
    ("polars_formula_contract", "tests/analytic_precision/polars_formula_contract.py", 2),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path)
    parser.add_argument("--forbid-prefix", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import xhdfe
    from xhdfe import py_hdfe_v11 as core

    package = Path(xhdfe.__file__).resolve()
    native = Path(core.__file__).resolve()
    for path in (package, native):
        if args.expected_prefix and not path.is_relative_to(args.expected_prefix.resolve()):
            raise RuntimeError(f"unexpected package/native path: {path}")
        if args.forbid_prefix and path.is_relative_to(args.forbid_prefix.resolve()):
            raise RuntimeError(f"source checkout shadowed the installed package: {path}")

    suite = unittest.TestSuite()
    counts = {}
    paths = [
        package, native, package.with_name("_formula.py"),
        package.with_name("_maketables.py"), package.with_name("_version.py"),
        Path(__file__).resolve(),
    ]
    for name, relative, expected in SUITES:
        path = ROOT / relative
        paths.append(path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        selected = unittest.defaultTestLoader.loadTestsFromModule(module)
        counts[name] = selected.countTestCases()
        if counts[name] != expected:
            raise RuntimeError(f"{name}: expected {expected} tests, found {counts[name]}")
        suite.addTests(selected)
    before = {str(path): sha256(path) for path in paths}
    print("FORMULA_INPUTS " + json.dumps({"version": xhdfe.__version__, "sha256": before}), flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == {str(path): sha256(path) for path in paths}
    success = (
        result.wasSuccessful()
        and not result.skipped
        and result.testsRun == sum(counts.values())
        and unchanged
    )
    print("FORMULA_CONTRACT " + json.dumps({
        "status": "PASS" if success else "FAIL",
        "counts": counts,
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": [(str(test), reason) for test, reason in result.skipped],
        "inputs_unchanged": unchanged,
    }), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
