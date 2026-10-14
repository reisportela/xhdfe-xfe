import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "encompassing" / "run_xhdfe_core23_ab.py"
SPEC = importlib.util.spec_from_file_location("xhdfe_core23_ab", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CudaEnvelopeAdjudicationTest(unittest.TestCase):
    def test_exact_two_sided_sign_test(self):
        self.assertEqual(
            MODULE.exact_two_sided_sign_p([1.0] * 10), 2.0 / 1024.0
        )
        self.assertEqual(
            MODULE.exact_two_sided_sign_p([1.0] * 5 + [-1.0] * 5), 1.0
        )
        self.assertIsNone(MODULE.exact_two_sided_sign_p([0.0] * 10))

    def test_signed_envelope_outcomes(self):
        self.assertEqual(
            MODULE.signed_adjudication([0.5] * 10, 1.0)["outcome"], "PASS"
        )
        self.assertEqual(
            MODULE.signed_adjudication([2.0] * 10, 1.0)["outcome"], "FAIL"
        )
        mixed = MODULE.signed_adjudication([2.0] * 5 + [-2.0] * 5, 1.0)
        self.assertEqual(mixed["outcome"], "INCONCLUSIVE")

    def test_self_control_order_compatibility_is_fail_closed(self):
        self.assertTrue(
            MODULE.envelope_compatibility(1.0, 9.0)[
                "compatible_within_one_order"
            ]
        )
        self.assertFalse(
            MODULE.envelope_compatibility(1.0, 10.0)[
                "compatible_within_one_order"
            ]
        )
        self.assertFalse(
            MODULE.envelope_compatibility(0.0, 1e-16)[
                "compatible_within_one_order"
            ]
        )
        floored = MODULE.envelope_compatibility(
            0.0,
            1e-16,
            MODULE.ENVELOPE_REPRESENTATION_FLOOR,
        )
        self.assertTrue(floored["compatible_within_one_order"])
        self.assertTrue(floored["floor_applied"])
        self.assertFalse(
            MODULE.envelope_compatibility(
                0.0,
                1e-8,
                MODULE.ENVELOPE_REPRESENTATION_FLOOR,
            )["compatible_within_one_order"]
        )
        self.assertIsNone(
            MODULE.envelope_compatibility(None, 1.0)[
                "compatible_within_one_order"
            ]
        )


if __name__ == "__main__":
    unittest.main()
