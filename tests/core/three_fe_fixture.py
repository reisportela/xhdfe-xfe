"""Generate synthetic Stata inputs and an independent explicit-dummy oracle."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def prepare(output):
    sys.path.insert(0, str(ROOT / "tests/analytic_precision"))
    from cuda_three_fe_covariance_contract import fixture, oracle
    import numpy as np
    import pandas as pd
    y, x, fes = fixture()
    b, v, reference = oracle(y, x, fes)
    data = output / "three_fe_synthetic.dta"
    oracle_do = output / "three_fe_oracle.do"
    assert not data.exists() and not oracle_do.exists()
    frame = pd.DataFrame(dict(y=y, x1=x[:, 0], x2=x[:, 1], x3=x[:, 2], x4=x[:, 3],
                              worker=fes[0], firm=fes[1], year=fes[2],
                              permutation=np.random.default_rng(2718).permutation(len(y))))
    frame.to_stata(data, write_index=False, version=118)
    matrix_b = ", ".join(format(value, ".17g") for value in b)
    matrix_v = " \\ ".join(", ".join(format(value, ".17g") for value in row) for row in v)
    with oracle_do.open("x") as out:
        out.write(f"matrix oracle_b = ({matrix_b})\nmatrix oracle_V = ({matrix_v})\n")
    with (output / "THREE_FE_FIXTURE.json").open("x") as out:
        json.dump(dict(synthetic=True, seed=20261020, reference=reference,
                       files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [data, oracle_do]}), out, indent=2)
        out.write("\n")
    return data, oracle_do
