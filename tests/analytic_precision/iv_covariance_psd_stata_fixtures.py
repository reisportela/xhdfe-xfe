#!/usr/bin/env python3
"""Export the independent PSD contract's represented inputs and MP120 oracle.

Usage: python iv_covariance_psd_stata_fixtures.py NEW_DIRECTORY
Then run iv_covariance_psd_contract.do cpu|cuda NEW_DIRECTORY in a scratch dir.
"""
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from iv_covariance_psd_contract import cases, reference
import mpmath as mp


def main():
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for index, case in enumerate(cases(), 1):
        ref = reference(case, 120)
        with mp.workdps(120):
            norms = [mp.sqrt(max(abs(ref["raw"][j,j]), abs(ref["V"][j,j]))) for j in range(3)]
        metadata = [case["weight_type"], case["g_df"], int(case["g_adj"]), int(case["k_adj"]),
                    int(ref["fixed"])]
        expected = [mp.nstr(ref["V"][j,k], 85) for j in range(3) for k in range(3)]
        expected += [mp.nstr(value, 85) for value in norms]
        with (directory/f"case_{index:03d}.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["y", "x1", "x2", "z1", "z2", "weight", "fe", "cl1", "cl2",
                             "weight_type", "g_df", "g_adj", "k_adj", "psd_fixed"] +
                            [f"v{j+1}{k+1}" for j in range(3) for k in range(3)] +
                            ["norm1", "norm2", "norm3"])
            for i in range(len(case["y"])):
                writer.writerow([repr(float(case["y"][i])), *map(repr, map(float, case["x"][i])),
                                 *map(repr, map(float, case["z"][i])), case["weights"][i],
                                 int(case["fe"][i]), int(case["cl1"][i]), int(case["cl2"][i])] +
                                (metadata+expected if i == 0 else [""]*17))
    print(f"Exported {index} independently referenced fixtures to {directory}")


if __name__ == "__main__":
    main()
