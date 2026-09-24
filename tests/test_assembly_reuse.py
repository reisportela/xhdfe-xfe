"""Assembly retries must never reuse changed native inputs or failed jobs."""
import copy
from pathlib import Path
import unittest

from tools.validate_assembly_reuse import validate_inputs


class AssemblyReuseTests(unittest.TestCase):
    def setUp(self):
        self.workflow = dict(env={"XHDFE_RELEASE_BUILD": "1"}, jobs={
            "build-linux": {"if": "old trigger", "steps": [{"run": "compiler --openmp"}]},
            "assemble-release": {"steps": [{"run": "assemble"}]},
            "release": {"if": "tag", "steps": [{"run": "draft"}]},
        })
        names = ["Build Linux (CPU + CUDA fatbin, Python and R)",
                 "Build/import Windows Python wheel (Strawberry MinGW)"]
        names += ["Build and validate Windows plugins in Stata installation layout / " + n
                  for n in ("build-probe-inputs", "native-loader")]
        names += ["Build and validate macOS OpenMP / " + n
                  for n in ("arm", "intel", "assemble", "final-arm", "final-intel", "seal")]
        self.jobs = [dict(name=n, conclusion="success") for n in names]

    def test_packaging_change_can_reuse_successful_native_jobs(self):
        current = copy.deepcopy(self.workflow)
        current["jobs"]["build-linux"]["if"] = "skip on assembly-only"
        current["jobs"]["assemble-release"]["steps"] = [{"run": "correct metadata"}]
        self.assertEqual(len(validate_inputs([".github/workflows/release.yml"],
                         self.workflow, current, self.jobs)), 10)

    def test_native_source_or_flags_or_environment_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            validate_inputs(["src/ols.cpp"], self.workflow, self.workflow, self.jobs)
        current = copy.deepcopy(self.workflow)
        current["jobs"]["build-linux"]["steps"][0]["run"] = "compiler --no-openmp"
        with self.assertRaisesRegex(ValueError, "job changed"):
            validate_inputs([], self.workflow, current, self.jobs)
        current = copy.deepcopy(self.workflow)
        current["env"]["XHDFE_RELEASE_BUILD"] = "0"
        with self.assertRaisesRegex(ValueError, "environment changed"):
            validate_inputs([], self.workflow, current, self.jobs)

    def test_failed_or_missing_native_receipts_cannot_be_reused(self):
        for status in ("failure", "skipped", "cancelled", None):
            jobs = copy.deepcopy(self.jobs)
            jobs[-1]["conclusion"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_inputs([], self.workflow, self.workflow, jobs)
        with self.assertRaises(ValueError):
            validate_inputs([], self.workflow, self.workflow, self.jobs[:-1])

    def test_manual_retry_cannot_publish(self):
        text = (Path(__file__).resolve().parents[1] / ".github/workflows/release.yml").read_text()
        self.assertIn("if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')", text)
        self.assertIn("if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/publish-v')", text)


if __name__ == "__main__":
    unittest.main()
