#!/usr/bin/env python3
"""SHA-256 and the framed block graph checked independently with hashlib."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

BLOCK = 1 << 20


def tree_hash(data):
    output = b"xhdfe-cache-sha256-blocks-v1" + struct.pack("<QQ", 0x5848444645544553, 4)
    for node in (17, data, len(data), data[1:]):
        if isinstance(node, int):
            output += b"\0" + struct.pack("<Q", node)
        else:
            chunks = [node[i:i+BLOCK] for i in range(0, len(node), BLOCK)] or [b""]
            output += b"\1" + struct.pack("<QQ", len(node), len(chunks))
            output += b"".join(hashlib.sha256(b"C" + struct.pack("<Q", len(chunk)) + chunk).digest()
                               for chunk in chunks)
    return hashlib.sha256(output).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    vectors = [b"", b"abc", b"a" * 1000000]
    for size in (55, 56, 63, 64, 65, BLOCK-1, BLOCK, BLOCK+1, 2*BLOCK+13):
        vectors.append((bytes(range(256)) * (size // 256 + 1))[:size])
    rows = []
    for case, data in enumerate(vectors):
        path = args.scratch / (str(case) + ".bin")
        path.write_bytes(data)
        expected = [hashlib.sha256(data).hexdigest(), tree_hash(data)]
        for threads in (1, 4):
            run = subprocess.run([str(args.exe.resolve()), str(path.resolve()), str(threads)],
                                 capture_output=True, text=True, timeout=60)
            actual = run.stdout.splitlines()
            rows.append(dict(case=case, size=len(data), threads=threads, actual=actual,
                             status="PASS" if run.returncode == 0 and actual == expected else "FAIL"))
    (args.scratch / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["status"] == "PASS" for row in rows), rows
    print("PASS: 24 SHA-256/framing/thread cases against hashlib")


if __name__ == "__main__":
    main()
