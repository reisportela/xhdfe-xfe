# T01 supplemental controls

These files are copied, immutable controls from the post-2.26.2 audit, except
for the three `ordinary_*_contract.py` graders identified in
`../supplemental_cases.json`. Their only T01 change is classificatory: an
informative refusal of a finite, supported, oracle-defined input now fails G1
as `FAIL_VALID_REFUSAL`. Numerical inputs, solver options, tolerances and
oracles are unchanged.

The scripts remain standalone and are not silently added to the historical
1,719-row count. A G1 report must list each separately, with the exact module,
backend, source hash and outcome. `product_probe_2262.json` is evidence, not an
executable oracle and not the same input as the `1e14+a` omission fixture.
