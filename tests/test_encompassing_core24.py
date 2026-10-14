import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ENC = ROOT / "benchmarks" / "encompassing"
REGISTRY = ENC / "registry.json"
MARTA_GATE = ROOT / "Verifications" / "marta_core24_gate_20260828"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS24 = load_module("xhdfe_core24_ab", ENC / "run_xhdfe_core24_ab.py")
HARNESS23 = load_module("xhdfe_core23_ab", ENC / "run_xhdfe_core23_ab.py")
WORKER = load_module("xhdfe_enc_worker", ENC / "enc_worker_py.py")
CERTIFIER = load_module("xhdfe_marta_certifier", MARTA_GATE / "certify_python.py")


def expand_group(groups, name):
    result = []
    for item in groups[name]:
        if "include_group" in item:
            result.extend(expand_group(groups, item["include_group"]))
        else:
            result.append(item)
    return result


class Core24RegistryTest(unittest.TestCase):
    def test_marta_data_and_generator_identities(self):
        expected = {
            ROOT / "benchmarks/data/marta_core24_group_individual_3m.parquet":
                "404259fe2a361bf471853299a8194ac6020de0d13108f153196e741c9f0ef93d",
            ROOT / "benchmarks/data/generate_marta_core24_group_individual_3m.do":
                "8f5dd201898ddbe124d66d51ecc29b3afa8a02e9c362d1c6be7611602dea36c8",
        }
        for path, digest in expected.items():
            self.assertTrue(path.is_file(), path)
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            self.assertEqual(hasher.hexdigest(), digest)

    def test_core24_is_core23_plus_literal_marta_spec(self):
        payload = json.loads(REGISTRY.read_text())
        core23 = expand_group(payload["groups"], "core23")
        core24 = expand_group(payload["groups"], "core24")
        self.assertEqual(len(core23), 23)
        self.assertEqual(len(core24), 24)
        self.assertEqual(core24[:23], core23)

        marta = core24[-1]
        self.assertEqual(marta["name"], "marta_group_individual_3m")
        self.assertEqual(marta["path"], "$MARTA_CORE24")
        self.assertEqual(marta["y"], "lhwage")
        self.assertEqual(marta["x"], ["FTC", "tenure", "tenure2"])
        self.assertEqual(marta["fe"], ["p_ntrab", "ntrab"])
        self.assertEqual(marta["group"], "p_groupid")
        self.assertEqual(marta["individual"], "p_ntrab")
        self.assertEqual(marta["aggregation"], "sum")
        self.assertIs(marta["singleton_drop"], True)
        self.assertIs(marta["compare_intercept"], True)
        self.assertNotIn("cluster", marta)
        self.assertTrue(all("compare_intercept" not in item for item in core23))

    def test_harnesses_resolve_exact_suite_sizes(self):
        core24 = HARNESS24.suite_specs(REGISTRY, "core24", 24)
        core23 = HARNESS24.suite_specs(REGISTRY, "core23", 23)
        self.assertEqual(len(core24), 24)
        self.assertEqual(len(core23), 23)
        self.assertEqual(len(HARNESS23.core23_specs(REGISTRY)), 23)
        self.assertEqual(list(core24)[:23], list(core23))
        self.assertEqual(
            HARNESS24.comparison_terms(core24["marta_group_individual_3m"]),
            ["FTC", "tenure", "tenure2", "_cons"],
        )
        self.assertEqual(
            HARNESS24.comparison_terms(core23["credit"]), ["x1", "x2"]
        )
        expected = ROOT / "benchmarks/data/marta_core24_group_individual_3m.parquet"
        self.assertEqual(
            Path(core24["marta_group_individual_3m"]["_resolved_path"]),
            expected.resolve(),
        )

    def test_worker_path_override_and_stata_list_fields(self):
        override = "/tmp/xhdfe-marta-core24-override.parquet"
        with mock.patch.dict(os.environ, {"XHDFE_MARTA_CORE24_DATA": override}):
            datasets = WORKER.select(WORKER.load_registry(REGISTRY), "core24", "")
        self.assertEqual(len(datasets), 24)
        marta = datasets[-1]
        self.assertEqual(marta["_registry_group"], "core24")
        self.assertEqual(marta["group"], "p_groupid")
        self.assertEqual(marta["path"], override)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            target = Path(tmp) / "marta.csv"
            env = os.environ.copy()
            env["XHDFE_MARTA_CORE24_DATA"] = override
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENC / "enc_worker_py.py"),
                    "--groups", "core24",
                    "--only", "marta_group_individual_3m",
                    "--emit-stata-list", str(target),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with target.open(newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["group"], "core24")
        self.assertEqual(row["cluster"], "")
        self.assertEqual(row["groupvar"], "p_groupid")
        self.assertEqual(row["individual"], "p_ntrab")
        self.assertEqual(row["aggregation"], "sum")
        self.assertEqual(row["singleton_drop"], "1")
        self.assertEqual(row["compare_intercept"], "1")

    def test_worker_reghdfe_derived_statistics_cover_nested_and_boundary_df(self):
        import numpy as np

        class Fit:
            pass

        def fit(**values):
            obj = Fit()
            for name, value in values.items():
                setattr(obj, name, value)
            return obj

        common = {
            "nobs_full_": 0.0,
            "num_singletons_": 0.0,
            "r2_": 0.5,
            "r2_within_": 0.4,
            "coef_": np.array([0.3, -0.2, 1.0]),
            "covariance_": np.diag([0.04, 0.09, 0.01]),
            "saturated_": False,
        }
        cases = {
            "synthetic-complete": {
                "nobs_": 500000.0,
                "df_resid_": 999.0,
                "df_resid_unadj_": 499499.0,
                "df_m_": 2.0,
                "df_a_": 499.0,
                "df_a_nested_": 1000.0,
                "rss_": 4988275053.694966,
                "tss_": 5004987798.903969,
                "tss_within_": 4989545534.675014,
                "expected": (
                    100.03294402465838,
                    0.000340232672298324,
                    0.00025061757460209666,
                ),
            },
            "soccer": {
                "nobs_": 73487.0,
                "df_resid_": 527.0,
                "df_resid_unadj_": 72965.0,
                "df_m_": 2.0,
                "df_a_": 520.0,
                "df_a_nested_": 528.0,
                "rss_": 729707332.9467653,
                "tss_": 740347866.4521056,
                "tss_within_": 729730599.5262115,
                "expected": (
                    100.36773591622313,
                    9.892735222982729e-05,
                    4.274474199483791e-06,
                ),
            },
        }
        for name, case in cases.items():
            expected = case.pop("expected")
            values = dict(common, **case)
            values["nobs_full_"] = values["nobs_"]
            stats = WORKER.xhdfe_fit_statistics(fit(**values), 2)
            with self.subTest(name=name):
                self.assertAlmostEqual(stats["rmse"], expected[0], places=12)
                self.assertAlmostEqual(stats["r2_a"], expected[1], places=14)
                self.assertAlmostEqual(stats["r2_a_within"], expected[2], places=14)
                self.assertTrue(np.isfinite(stats["f_stat"]))
                self.assertEqual(stats["model_has_cons"], 1)
                self.assertEqual(stats["df_r_unadj"], case["df_resid_unadj_"])
                self.assertEqual(stats["df_a_nested"], case["df_a_nested_"])

        zigzag = fit(
            **dict(
                common,
                nobs_=10002.0,
                nobs_full_=10002.0,
                df_resid_=4999.0,
                df_resid_unadj_=5000.0,
                df_m_=2.0,
                df_a_=5000.0,
                df_a_nested_=5000.0,
                rss_=5.054072663991876e-16,
                tss_=102580960.03346725,
                tss_within_=112658.47373177457,
            )
        )
        stats = WORKER.xhdfe_fit_statistics(zigzag, 2)
        self.assertEqual(stats["rmse"], 2.248126478646581e-08)
        self.assertTrue(np.isnan(stats["f_stat"]))
        self.assertTrue(np.isnan(stats["r2_a"]))
        self.assertTrue(np.isnan(stats["r2_a_within"]))

        grouped_saturated = fit(
            **dict(
                common,
                nobs_=2.0,
                nobs_full_=2.0,
                df_resid_=-2.0,
                df_resid_unadj_=-2.0,
                df_m_=1.0,
                df_a_=3.0,
                df_a_nested_=0.0,
                rss_=1e-20,
                tss_=1.0,
                tss_within_=0.5,
                coef_=np.array([0.3, 1.0]),
                covariance_=np.diag([0.04, 0.01]),
                # The current grouped core flag is false even with negative df_r;
                # derived display semantics therefore cannot depend on this flag.
                saturated_=False,
            )
        )
        stats = WORKER.xhdfe_fit_statistics(grouped_saturated, 1)
        self.assertTrue(np.isnan(stats["rmse"]))
        self.assertTrue(np.isnan(stats["f_stat"]))
        self.assertEqual(stats["r2_a"], 1.0)
        self.assertEqual(stats["r2_a_within"], 1.0)

        no_intercept = fit(
            **dict(
                common,
                nobs_=100.0,
                nobs_full_=100.0,
                df_resid_=90.0,
                df_resid_unadj_=90.0,
                df_m_=2.0,
                df_a_=8.0,
                df_a_nested_=0.0,
                rss_=180.0,
                tss_=400.0,
                tss_within_=300.0,
                coef_=np.array([0.3, -0.2]),
                covariance_=np.diag([0.04, 0.09]),
            )
        )
        stats = WORKER.xhdfe_fit_statistics(no_intercept, 2)
        self.assertEqual(stats["model_has_cons"], 0)
        self.assertEqual(stats["r2_a"], 1.0 - (180.0 / 90.0) / (400.0 / 100.0))

    def test_reghdfe_statistics_mirrors_and_frontend_guards_are_aligned(self):
        core_paths = [
            ROOT / "src/hdfe_regressor_v11.cpp",
            ROOT / "stata/src/hdfe_regressor_v11.cpp",
            ROOT / "share/xhdfe_estimation_cpp/src/hdfe_regressor_v11.cpp",
            ROOT / "share/xhdfe_estimation_cpp/stata/src/hdfe_regressor_v11.cpp",
            ROOT / "r/xhdfe/src/hdfe_regressor_v11.cpp",
        ]
        core = core_paths[0].read_bytes()
        self.assertTrue(all(path.read_bytes() == core for path in core_paths[1:]))
        core_text = core.decode()
        self.assertEqual(core_text.count("const double sigma2_df ="), 2)
        self.assertEqual(
            core_text.count(
                "use_reghdfe_stats ? df_r_raw - static_cast<double>(nested_report)"
            ),
            2,
        )
        self.assertEqual(
            core_text.count("use_reghdfe_stats && sigma2_df == 0.0"), 2
        )

        ado = (ROOT / "stata/xhdfe.ado").read_bytes()
        self.assertEqual(
            ado, (ROOT / "share/xhdfe_estimation_cpp/stata/xhdfe.ado").read_bytes()
        )
        ado_text = ado.decode()
        self.assertIn("`used_df_r' - `df_a_nested'", ado_text)
        self.assertNotIn("max(`used_df_r' - `df_a_nested', 0)", ado_text)
        self.assertIn("e(df_r) <= 0", ado_text)

        r_text = (ROOT / "r/xhdfe/R/estimate.R").read_text()
        self.assertIn('reghdfe_stats <- identical(stats_style, "reghdfe")', r_text)
        self.assertIn("!reghdfe_stats || (is.finite(used_df_r)", r_text)

    def test_stata_worker_builds_literal_group_options_and_optional_vce(self):
        source = (ENC / "enc_worker_stata.do").read_text()
        self.assertIn('"group(`gr`i\'\')"', source)
        self.assertIn('"`giopts`i\'\' individual(`ind`i\'\')"', source)
        self.assertIn('"`giopts`i\'\' aggregation(`agg`i\'\')"', source)
        self.assertIn('if ("`cl`i\'\'" != "") local vceopt`i\'', source)
        self.assertIn("absorb(`abs`i'') `giopts`i''", source)
        self.assertIn('(\"_cons\") (_b[_cons]) (_se[_cons])', source)

        py_source = (ENC / "enc_worker_py.py").read_text()
        self.assertIn("group_arg = fac(df[group_var])", py_source)
        self.assertIn("individual_arg = fac(df[individual_var])", py_source)
        self.assertIn('aggregation_arg = d.get("aggregation", "mean")', py_source)
        self.assertIn('se_type="cluster" if cl_vars else "unadjusted"', py_source)
        self.assertIn("group=group_arg, individual=individual_arg", py_source)
        self.assertIn("aggregation=aggregation_arg", py_source)
        self.assertIn('term="_cons"', py_source)

    def test_core24_documentation_is_addable_and_declares_precision_gates(self):
        doc = ENC / "CORE24.md"
        self.assertTrue(doc.is_file())
        text = doc.read_text()
        self.assertIn("LSMR", text)
        self.assertIn("tolerance(1e-14)", text)
        self.assertIn("tolerance(1e-12)", text)
        self.assertIn("A missing value is a failure", text)
        self.assertIn("Eight xhdfe cells", text)
        self.assertIn("run_xhdfe_core24_ab.py", text)
        self.assertIn("oracle_residuals.parquet", text)
        self.assertIn("relative-L2", text)
        readme = (ENC / "README.md").read_text()
        self.assertIn("[`CORE24.md`](CORE24.md)", readme)

    def test_direct_residual_oracle_alignment_is_exact_and_fail_closed(self):
        import numpy as np

        oracle_keys = np.array([10, 20, 30], dtype=np.int64)
        oracle_residuals = np.array([0.5, -1.25, 2.0])
        candidate_keys = np.array([30, 10, 20], dtype=np.int64)
        candidate_residuals = np.array([2.0, 0.5, -1.25])
        result = CERTIFIER.compare_residual_vectors(
            np, candidate_keys, candidate_residuals,
            oracle_keys, oracle_residuals,
        )
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["relative_l2"], 0.0)
        self.assertEqual(result["max_normalized"], 0.0)
        self.assertEqual(result["oracle_rss"], result["candidate_rss"])

        with self.assertRaisesRegex(CERTIFIER.CertificationError, "keys differ"):
            CERTIFIER.compare_residual_vectors(
                np, np.array([10, 20, 31]), candidate_residuals,
                oracle_keys, oracle_residuals,
            )
        with self.assertRaisesRegex(CERTIFIER.CertificationError, "non-finite"):
            CERTIFIER.compare_residual_vectors(
                np, oracle_keys, np.array([0.5, float("nan"), 2.0]),
                oracle_keys, oracle_residuals,
            )

    def test_residual_oracle_sidecar_and_runner_custody_are_mandatory(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            dataset = root / "marta.parquet"
            oracle = root / "comparisons.csv"
            residual = root / "oracle_residuals.parquet"
            dataset.write_bytes(b"dataset")
            oracle.write_text("metric,reference\n")
            residual.write_bytes(b"residual-oracle")
            dataset_hash = CERTIFIER.sha256(dataset)
            oracle_hash = CERTIFIER.sha256(oracle)
            residual_hash = CERTIFIER.sha256(residual)
            oracle_sidecar = root / "comparisons.csv.sha256"
            oracle_sidecar.write_text(f"{oracle_hash}  {oracle}\n")
            sidecar = root / "oracle_residuals.parquet.sha256"
            sidecar.write_text(f"{residual_hash}  {residual}\n")
            (root / "runner.log").write_text(
                f"{dataset_hash}  {dataset}\n"
                f"{oracle_hash}  {oracle}\n"
                f"{residual_hash}  {residual}\n"
            )
            custody = CERTIFIER.verify_oracle_dataset_custody(
                oracle, oracle_hash, dataset, dataset_hash,
                residual, residual_hash,
            )
            self.assertTrue(custody["dataset_hash_recorded"])
            self.assertTrue(custody["oracle_hash_recorded"])
            self.assertTrue(custody["residual_hash_recorded"])
            self.assertEqual(custody["residual_sha256_sidecar"], str(sidecar))

            oracle_sidecar.write_text(f"{'0' * 64}  {oracle}\n")
            with self.assertRaisesRegex(
                CERTIFIER.CertificationError, "differs from its sidecar"
            ):
                CERTIFIER.verify_oracle_dataset_custody(
                    oracle, oracle_hash, dataset, dataset_hash,
                    residual, residual_hash,
                )
            oracle_sidecar.write_text(f"{oracle_hash}  {oracle}\n")
            sidecar.write_text(f"{'0' * 64}  {residual}\n")
            with self.assertRaisesRegex(
                CERTIFIER.CertificationError, "differs from its sidecar"
            ):
                CERTIFIER.verify_oracle_dataset_custody(
                    oracle, oracle_hash, dataset, dataset_hash,
                    residual, residual_hash,
                )

    def test_residual_oracle_parquet_schema_and_finiteness_are_mandatory(self):
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "oracle_residuals.parquet"
            pd.DataFrame({
                "p_groupid": [3, 1, 2],
                "residual": [0.25, -0.5, 1.0],
            }).to_parquet(path, index=False)
            keys, residuals = CERTIFIER.load_residual_oracle(pd, np, path)
            self.assertEqual(keys.tolist(), [3, 1, 2])
            self.assertEqual(residuals.tolist(), [0.25, -0.5, 1.0])

            pd.DataFrame({
                "p_groupid": [1, 2],
                "wrong": [0.0, 1.0],
            }).to_parquet(path, index=False)
            with self.assertRaisesRegex(
                CERTIFIER.CertificationError, "cannot read residual oracle"
            ):
                CERTIFIER.load_residual_oracle(pd, np, path)

            pd.DataFrame({
                "p_groupid": [1, 2],
                "residual": [0.0, float("inf")],
            }).to_parquet(path, index=False)
            with self.assertRaisesRegex(CERTIFIER.CertificationError, "non-finite"):
                CERTIFIER.load_residual_oracle(pd, np, path)

    def test_final_marta_certifiers_require_lsmr_and_residual_artifact(self):
        python_source = (MARTA_GATE / "certify_python.py").read_text()
        stata_source = (MARTA_GATE / "marta_core24_certify.do").read_text()
        runner_source = (MARTA_GATE / "run_gate.sh").read_text()

        self.assertIn('else oracle_path.with_name("oracle_residuals.parquet")',
                      python_source)
        self.assertIn('method == "lsmr" and method_code == 5', python_source)
        self.assertNotIn('method == "gauss-seidel"', python_source)
        self.assertIn("PUBLIC_TOLERANCE = 1e-12", python_source)
        self.assertIn('"residual:relative_l2"', python_source)
        self.assertIn('"residual:max_normalized"', python_source)
        self.assertIn('"residual_oracle_sha256"', python_source)
        self.assertIn("CANONICAL_DATASET_SHA256", python_source)

        self.assertIn("findfile xhdfe.ado", stata_source)
        self.assertIn("findfile xhdfe.plugin", stata_source)
        self.assertIn("set processors `threads_n'", stata_source)
        self.assertGreaterEqual(stata_source.count("vce(unadjusted)"), 3)
        self.assertIn("default and tight reghdfe samples differ", stata_source)
        self.assertEqual(
            stata_source.count("scalar(X_absorption_method_used) != 5"), 2
        )
        self.assertIn("keep p_groupid `resid_ref_g'", stata_source)
        self.assertIn("isid p_groupid", stata_source)
        self.assertIn("compression(zstd)", stata_source)
        self.assertIn("oracle_residual_rows=", stata_source)

        self.assertIn('[[ ! -s "$ARTIFACT" ]]', runner_source)
        self.assertIn('for ARTIFACT in "$COMPARISONS_FILE" "$ORACLE_FILE"',
                      runner_source)
        self.assertIn('sha256sum --check "$SIDECAR"', runner_source)
        self.assertIn("CANONICAL_DATASET_SHA256=404259fe", runner_source)
        self.assertIn('sha256sum --check "$INPUT_MANIFEST"', runner_source)
        self.assertIn("oracle_residual_rows=", runner_source)
        completed = subprocess.run(
            ["bash", "-n", str(MARTA_GATE / "run_gate.sh")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runner_fails_closed_when_stata_omits_residual_oracle(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_stata = fake_bin / "stata-mp"
            fake_stata.write_text(
                "#!/bin/sh\n"
                "printf 'verdict=PASS\\noracle_residual_rows=3\\n' "
                '> "$MARTA_GATE_OUTDIR/summary.txt"\n'
                "printf 'metric,reference,candidate,absdiff,reldiff,comparable,pass,note\\n' "
                '> "$MARTA_GATE_OUTDIR/comparisons.csv"\n'
                "exit 0\n"
            )
            fake_stata.chmod(0o755)
            out = root / "out"
            env = os.environ.copy()
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "MARTA_GATE_BACKEND": "cpu",
                "MARTA_GATE_MODE": "comparable",
                "MARTA_GATE_THREADS": "1",
                "MARTA_GATE_OUTDIR": str(out),
            })
            completed = subprocess.run(
                ["bash", str(MARTA_GATE / "run_gate.sh")],
                cwd=ROOT, env=env, text=True, capture_output=True,
                check=False, timeout=60,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout)
            self.assertIn("missing or empty certification artifact", completed.stdout)
            verdict = (out / "runner_verdict.txt").read_text()
            self.assertIn("verdict=FAIL", verdict)
            self.assertIn("stata_rc=0", verdict + "\n" + completed.stdout)
            self.assertIn("gate_rc=1", verdict)

    def test_marta_leg_fails_when_any_conditionally_defined_stat_is_missing(self):
        record = {
            "dataset": "marta_group_individual_3m",
            "pair": 1,
            "role": "candidate",
            "engine": "cpp",
            "artifact_path": "/tmp/not-used-artifact",
            "package_path": "/tmp/not-used-package",
            "worker_row": {},
        }
        failures = HARNESS24.validate_leg(
            record, "xhdfe_cpp_cpu_comparable", "cpu", "comparable", 1, False
        )
        for field in ("f_stat", "r2_a", "r2_a_within"):
            self.assertTrue(
                any(f"{field} is missing; Marta core24 requires all stats" in f
                    for f in failures),
                failures,
            )

    def test_core23_and_core24_entrypoint_help(self):
        for script, expected in (
            ("run_xhdfe_core23_ab.py", "core23 subset"),
            ("run_xhdfe_core24_ab.py", "core24 subset"),
        ):
            completed = subprocess.run(
                [sys.executable, str(ENC / script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected, completed.stdout)

    def test_core23_wrapper_preserves_practical_import_api(self):
        historical_public = {
            "HarnessError",
            "core23_specs",
            "parse_args",
            "main",
            "execute_leg",
            "validate_leg",
            "pair_contract",
            "build_summary",
            "exact_two_sided_sign_p",
            "signed_adjudication",
            "envelope_compatibility",
        }
        self.assertFalse(
            historical_public - set(vars(HARNESS23)),
            f"missing wrapper API: {sorted(historical_public - set(vars(HARNESS23)))}",
        )
        self.assertEqual(str(inspect.signature(HARNESS23.parse_args)), "()")
        self.assertEqual(str(inspect.signature(HARNESS23.main)), "() -> int")
        self.assertEqual(len(HARNESS23.core23_specs(REGISTRY)), 23)
        self.assertIs(
            HARNESS23.exact_two_sided_sign_p,
            sys.modules[HARNESS23._core24_main.__module__].exact_two_sided_sign_p,
        )

    def test_pair_contract_reports_algorithm_and_absolute_residual_changes(self):
        base_row = {
            "iterations": "24",
            "gpu_absorption_iterations": "0",
            "method": "mlsmr",
            "gpu_backend": "cpu",
            "gpu_status": "not_requested",
            "abs_residual": "0.7883685712042727",
            "abs_residual_rel": "9.071932950220561e-10",
            "rss": "2.9e-19",
            "tss_within": "112658.47",
            "df_m": "2",
            "f_stat": "1.63e31",
            "elapsed_seconds": "1.0",
        }
        candidate_row = dict(base_row)
        candidate_row.update({
            "iterations": "25",
            "method": "lsmr",
            "abs_residual": "0.7884129063330778",
            "abs_residual_rel": "9.093019597620248e-10",
            "f_stat": "6.40e31",
        })
        coefficients = {"x": (1.0, 0.1)}
        common = {
            "dataset": "probe",
            "pair": 1,
            "coefficients": coefficients,
            "leg_contract_failures": [],
            "legacy_baseline_contract": False,
            "background_busy_fraction": 0.0,
            "max_background_diff": 0.03,
        }
        baseline = dict(common, worker_row=base_row, order=1)
        candidate = dict(common, worker_row=candidate_row, order=2)
        metric, failures = HARNESS24.pair_contract(
            baseline, candidate, ["x"], backend="cpu"
        )
        self.assertEqual(failures, [])
        observations = metric["legacy_baseline_observations"]
        self.assertTrue(any("iterations changed" in item for item in observations))
        self.assertTrue(any("method changed" in item for item in observations))
        self.assertTrue(any("abs_residual changed" in item for item in observations))
        self.assertTrue(any("f_stat is non-decisional" in item for item in observations))


if __name__ == "__main__":
    unittest.main()
