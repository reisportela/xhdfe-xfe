#!/usr/bin/env python3
"""Best-effort host-state provenance for xhdfe benchmark runs.

The sentinel is deliberately non-gating: host telemetry can be incomplete on
shared machines, but the missing field must be visible in the run artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def record_error(payload: dict[str, Any], source: str, exc: Exception | str) -> None:
    payload.setdefault("errors", {})[source] = str(exc)


def command(args: list[str], timeout: int = 15) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout).stdout.strip()


def meminfo() -> dict[str, int]:
    wanted = {"MemFree", "MemAvailable", "SwapFree", "Cached"}
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB$", line)
        if match and match.group(1) in wanted:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def buddyinfo() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in Path("/proc/buddyinfo").read_text().splitlines():
        tokens = line.replace(",", "").split()
        # "Node 0 zone Normal 12 13 ...": orders start after the zone name.
        if len(tokens) < 5 or tokens[0] != "Node":
            continue
        counts = [int(v) for v in tokens[4:] if v.isdigit()]
        result.append({
            "node": int(tokens[1]),
            "zone": tokens[3],
            "order_ge_9": {str(order): count for order, count in enumerate(counts) if order >= 9},
            "pages_order_ge_9": sum(count * (1 << order) for order, count in enumerate(counts) if order >= 9),
        })
    return result


def nvidia_state(payload: dict[str, Any]) -> None:
    try:
        query = ("driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,"
                 "clocks.sm,clocks.mem,memory.used")
        output = command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
        payload["gpus"] = [dict(zip(query.split(","), [v.strip() for v in line.split(",")]))
                           for line in output.splitlines() if line.strip()]
    except Exception as exc:  # telemetry must never gate a benchmark
        record_error(payload, "nvidia_smi_gpu", exc)
    try:
        output = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                          "--format=csv,noheader,nounits"])
        payload["resident_compute_apps"] = [
            dict(zip(("pid", "process_name", "used_gpu_memory"),
                     [v.strip() for v in line.split(",", 2)]))
            for line in output.splitlines() if line.strip() and "No running processes" not in line
        ]
    except Exception as exc:
        record_error(payload, "nvidia_smi_compute_apps", exc)


def h2d_probe(payload: dict[str, Any]) -> None:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        size = 256 * 1024 * 1024

        def bandwidth(source: Any) -> float:
            destination = torch.empty(size, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            destination.copy_(source, non_blocking=False)
            torch.cuda.synchronize()
            return size / (time.perf_counter() - start) / 1e9

        pageable = torch.empty(size, dtype=torch.uint8)
        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        payload["h2d_256mb_gbps"] = {
            "pageable": bandwidth(pageable),
            "pinned": bandwidth(pinned),
        }
    except Exception as exc:
        record_error(payload, "h2d_256mb", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="directory receiving env_sentinel_<stamp>.json")
    parser.add_argument("--label", default="", help="optional run phase label, e.g. start or end")
    args = parser.parse_args()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    payload: dict[str, Any] = {"schema": 1, "timestamp_utc": stamp, "label": args.label}
    try:
        payload["loadavg"] = list(os.getloadavg())
    except Exception as exc:
        record_error(payload, "loadavg", exc)
    for name, collector in (("meminfo_bytes", meminfo), ("buddyinfo", buddyinfo)):
        try:
            payload[name] = collector()
        except Exception as exc:
            record_error(payload, name, exc)
    try:
        payload["tuned_active"] = command(["tuned-adm", "active"])
    except Exception as exc:
        record_error(payload, "tuned_adm", exc)
    nvidia_state(payload)
    if os.environ.get("XHDFE_SENTINEL_H2D") == "1":
        h2d_probe(payload)

    try:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"env_sentinel_{stamp}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(target)
    except Exception as exc:
        # This is deliberately a success status: a benchmark must proceed even
        # when its optional provenance file cannot be written.
        print(f"env sentinel write error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
