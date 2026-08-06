"""Focused tests for the pure-Python AKM convenience layer."""

from unittest.mock import patch

import numpy as np

from xhdfe import akm


def test_subsampling_forwards_shared_execution_controls_to_leave_out_set():
    y = np.arange(6.0)
    worker = np.repeat(np.arange(3), 2)
    firm = np.tile(np.arange(2), 3)
    calls = {}

    def fake_leave_out_set(worker_arg, firm_arg, **kwargs):
        calls["leave_out"] = kwargs
        return {"keep": np.ones(worker_arg.size, dtype=bool)}

    def fake_akm_kss(y_arg, worker_arg, firm_arg, X=None, **kwargs):
        calls["akm"] = kwargs
        return {
            "sample": {"n_obs": y_arg.size, "n_movers": 3},
            "plugin": {},
            "agsu": {},
            "kss": {},
            "converged": True,
        }

    with (
        patch.object(akm, "leave_out_set", fake_leave_out_set),
        patch.object(akm, "akm_kss", fake_akm_kss),
    ):
        records = akm.subsampling_diagnostic(
            y,
            worker,
            firm,
            fractions=(0.0,),
            num_threads=7,
            gpu=True,
            verbose=2,
            leverages="exact",
        )

    assert calls["leave_out"] == {
        "num_threads": 7,
        "gpu": True,
        "verbose": 2,
    }
    assert calls["akm"] == {
        "num_threads": 7,
        "gpu": True,
        "verbose": 2,
        "leverages": "exact",
    }
    assert len(records) == 1


if __name__ == "__main__":
    test_subsampling_forwards_shared_execution_controls_to_leave_out_set()
