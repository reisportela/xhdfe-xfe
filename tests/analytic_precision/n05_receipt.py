"""Strict private parser for the single-line N05 audit receipt."""

from __future__ import annotations

import copy
import json
from typing import Iterable, Mapping, Optional


PREFIX = "XHDFE_N05_AUDIT "
SCHEMA = "xhdfe-n05-audit-v1"
FAMILIES = ("projection", "homoskedastic", "sandwich", "full_v")
STATUSES = frozenset(
    ("PASS", "ERROR_DEMONSTRATED", "BOUND_INCONCLUSIVE", "UNSUPPORTED")
)
ENTRYPOINTS = frozenset(("fit", "partial_out", "group_fit"))
MODES = frozenset(("xhdfe-fast", "reghdfe-comparable", "strict-residual"))
BACKENDS = frozenset(("cpu", "cuda"))
TOP_LEVEL_KEYS = frozenset(
    (
        "schema",
        "entrypoint",
        "mode",
        "backend",
        "iterations",
        "cache_hit",
        "n1_refinements",
        "identity_match",
        "mutated",
        "families",
    )
)
FAMILY_KEYS = frozenset(("status", "reason"))
PASS_REASONS = {
    "projection": "v82_projection_proof",
    "homoskedastic": "structured_homoskedastic_retained_covariance",
    "sandwich": "sandwich_candidate_moments",
    "full_v": "full_oneway_covariance_and_recovered_intercept",
}


class N05ReceiptError(ValueError):
    """The N05 stderr receipt is absent, forged, stale, or malformed."""


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_receipt(
    receipt: object,
    *,
    expected_entrypoint: Optional[str],
    expected_mode: Optional[str],
    expected_backend: Optional[str],
) -> dict:
    if not isinstance(receipt, dict):
        raise N05ReceiptError("receipt JSON must be an object")
    keys = frozenset(receipt)
    if keys != TOP_LEVEL_KEYS:
        raise N05ReceiptError(
            f"receipt keys differ: missing={sorted(TOP_LEVEL_KEYS - keys)}, "
            f"extra={sorted(keys - TOP_LEVEL_KEYS)}"
        )
    if receipt["schema"] != SCHEMA:
        raise N05ReceiptError("wrong receipt schema")
    if not isinstance(receipt["entrypoint"], str) or receipt["entrypoint"] not in ENTRYPOINTS:
        raise N05ReceiptError("invalid entrypoint")
    if not isinstance(receipt["mode"], str) or receipt["mode"] not in MODES:
        raise N05ReceiptError("invalid tolerance mode")
    if not isinstance(receipt["backend"], str) or receipt["backend"] not in BACKENDS:
        raise N05ReceiptError("invalid backend")
    if expected_entrypoint is not None and receipt["entrypoint"] != expected_entrypoint:
        raise N05ReceiptError("wrong entrypoint for this call")
    if expected_mode is not None and receipt["mode"] != expected_mode:
        raise N05ReceiptError("wrong tolerance mode for this call")
    if expected_backend is not None and receipt["backend"] != expected_backend:
        raise N05ReceiptError("wrong backend for this call")
    if not _integer(receipt["iterations"]) or receipt["iterations"] < 0:
        raise N05ReceiptError("iterations must be a nonnegative integer")
    if not isinstance(receipt["cache_hit"], bool):
        raise N05ReceiptError("cache_hit must be boolean")
    if not _integer(receipt["n1_refinements"]) or receipt["n1_refinements"] not in (0, 1):
        raise N05ReceiptError("n1_refinements must be 0 or 1")
    if not isinstance(receipt["identity_match"], bool):
        raise N05ReceiptError("identity_match must be boolean")
    if receipt["mutated"] is not False:
        raise N05ReceiptError("audit reports a mutated estimation result")
    families = receipt["families"]
    if not isinstance(families, dict) or frozenset(families) != frozenset(FAMILIES):
        present = sorted(families) if isinstance(families, dict) else []
        raise N05ReceiptError(f"receipt must contain exactly all four families; got {present}")
    for name in FAMILIES:
        family = families[name]
        if not isinstance(family, dict) or frozenset(family) != FAMILY_KEYS:
            raise N05ReceiptError(f"family {name} must contain exactly status and reason")
        if not isinstance(family["status"], str) or family["status"] not in STATUSES:
            raise N05ReceiptError(f"family {name} has an invalid status")
        if not isinstance(family["reason"], str) or not family["reason"]:
            raise N05ReceiptError(f"family {name} reason must be a nonempty string")
        if family["status"] == "PASS" and family["reason"] != PASS_REASONS[name]:
            raise N05ReceiptError(f"family {name} PASS has the wrong reason code")
    if not receipt["identity_match"] and any(
        families[name]["status"] == "PASS"
        for name in ("homoskedastic", "sandwich", "full_v")
    ):
        raise N05ReceiptError("post-OLS PASS is attached to a stale or missing result identity")
    return receipt


def parse_n05_receipt(
    stderr: str,
    *,
    required: bool,
    expected_entrypoint: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_backend: Optional[str] = None,
) -> Optional[dict]:
    """Parse exactly one prefixed JSON line, or require its meaningful absence."""
    if not isinstance(stderr, str):
        raise N05ReceiptError("stderr must be text")
    lines = [line for line in stderr.splitlines() if line.startswith(PREFIX)]
    if not lines:
        if required:
            raise N05ReceiptError("required N05 receipt is absent")
        return None
    if not required:
        raise N05ReceiptError("N05 receipt appeared while the exact selector was disabled")
    if len(lines) != 1:
        raise N05ReceiptError(f"expected one N05 receipt, found {len(lines)}")
    encoded = lines[0][len(PREFIX) :]
    try:
        receipt = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise N05ReceiptError(f"invalid receipt JSON: {error.msg}") from error
    return _validate_receipt(
        receipt,
        expected_entrypoint=expected_entrypoint,
        expected_mode=expected_mode,
        expected_backend=expected_backend,
    )


def require_family_passes(receipt: Mapping[str, object], families: Iterable[str]) -> None:
    """Require actual PASS statuses; bounds and unsupported labels are not proof."""
    failures = []
    for name in families:
        if name not in FAMILIES:
            raise N05ReceiptError(f"unknown proof family {name}")
        family = receipt["families"][name]
        if family["status"] != "PASS":
            failures.append(f"{name}={family['status']}:{family['reason']}")
    if failures:
        raise N05ReceiptError("audit proof did not pass: " + ", ".join(failures))


def _base_receipt() -> dict:
    return {
        "schema": SCHEMA,
        "entrypoint": "fit",
        "mode": "reghdfe-comparable",
        "backend": "cpu",
        "iterations": 3,
        "cache_hit": False,
        "n1_refinements": 0,
        "identity_match": True,
        "mutated": False,
        "families": {
            name: {"status": "PASS", "reason": PASS_REASONS[name]} for name in FAMILIES
        },
    }


def _self_test() -> int:
    checks = 0
    base = _base_receipt()

    def parse(value: dict, **expected: str) -> dict:
        line = PREFIX + json.dumps(value, separators=(",", ":")) + "\n"
        return parse_n05_receipt(line, required=True, **expected)

    parse(
        base,
        expected_entrypoint="fit",
        expected_mode="reghdfe-comparable",
        expected_backend="cpu",
    )
    checks += 1
    assert parse_n05_receipt("ordinary diagnostic\n", required=False) is None
    checks += 1

    bounded = copy.deepcopy(base)
    bounded["families"]["projection"] = {
        "status": "BOUND_INCONCLUSIVE",
        "reason": "bounded_reference",
    }
    parse(bounded)
    try:
        require_family_passes(bounded, ("projection",))
    except N05ReceiptError:
        checks += 1
    else:
        raise AssertionError("BOUND_INCONCLUSIVE counted as proof PASS")

    unsupported = copy.deepcopy(base)
    unsupported["families"]["sandwich"] = {
        "status": "UNSUPPORTED",
        "reason": "multiway",
    }
    parse(unsupported)
    try:
        require_family_passes(unsupported, ("sandwich",))
    except N05ReceiptError:
        checks += 1
    else:
        raise AssertionError("UNSUPPORTED counted as proof PASS")

    projection_only = copy.deepcopy(base)
    projection_only["entrypoint"] = "partial_out"
    projection_only["identity_match"] = False
    for name in ("homoskedastic", "sandwich", "full_v"):
        projection_only["families"][name] = {
            "status": "UNSUPPORTED",
            "reason": "partial_out_no_ols",
        }
    parse(projection_only, expected_entrypoint="partial_out")
    checks += 1

    negatives = []
    stale = copy.deepcopy(base)
    stale["identity_match"] = False
    negatives.append((stale, {}))
    missing = copy.deepcopy(base)
    del missing["families"]["full_v"]
    negatives.append((missing, {}))
    negatives.append((copy.deepcopy(base), {"expected_backend": "cuda"}))
    mutated = copy.deepcopy(base)
    mutated["mutated"] = True
    negatives.append((mutated, {}))
    forged_refinements = copy.deepcopy(base)
    forged_refinements["n1_refinements"] = True
    negatives.append((forged_refinements, {}))
    for value, expected in negatives:
        try:
            parse(value, **expected)
        except N05ReceiptError:
            checks += 1
        else:
            raise AssertionError("malformed or forged receipt was accepted")

    doubled = PREFIX + json.dumps(base) + "\n" + PREFIX + json.dumps(base) + "\n"
    try:
        parse_n05_receipt(doubled, required=True)
    except N05ReceiptError:
        checks += 1
    else:
        raise AssertionError("duplicate receipts were accepted")

    try:
        parse_n05_receipt(PREFIX + json.dumps(base), required=False)
    except N05ReceiptError:
        checks += 1
    else:
        raise AssertionError("disabled-selector receipt was accepted")

    print(json.dumps({"status": "PASS", "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
