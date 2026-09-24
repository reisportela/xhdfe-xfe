"""Authenticate successful native jobs before an assembly-only diagnostic retry.

This mode cannot create or publish a release. Version tags always rebuild.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

PACKAGING_ONLY = {
    ".github/workflows/release.yml",
    "tools/validate_assembly_reuse.py",
    "tools/validate_corresponding_source_bundle.py",
    "tests/test_assembly_reuse.py",
    "tests/test_windows_stata_static.py",
    "docs/LICOES_RELEASE_2.28.0_20260924.md",
}
NATIVE_PREFIXES = (
    "Build Linux (CPU + CUDA fatbin, Python and R)",
    "Build/import Windows Python wheel",
    "Build and validate Windows plugins in Stata installation layout /",
    "Build and validate macOS OpenMP /",
)


def validate_inputs(changed, old_workflow, current_workflow, jobs):
    unexpected = set(changed) - PACKAGING_ONLY
    if unexpected:
        raise ValueError("non-packaging inputs changed: " + ", ".join(sorted(unexpected)))
    if old_workflow.get("env") != current_workflow.get("env"):
        raise ValueError("build environment changed")
    old_jobs, current_jobs = old_workflow["jobs"], current_workflow["jobs"]
    if set(old_jobs) != set(current_jobs):
        raise ValueError("job set changed")
    for name in old_jobs:
        if name == "assemble-release":
            continue
        old = {k: v for k, v in old_jobs[name].items() if k != "if"}
        new = {k: v for k, v in current_jobs[name].items() if k != "if"}
        if old != new:
            raise ValueError("build or publication job changed: " + name)
    native = [job for job in jobs if job["name"].startswith(NATIVE_PREFIXES)]
    if len(native) != 10 or any(job.get("conclusion") != "success" for job in native):
        raise ValueError("expected all ten successful native build/validation jobs")
    for prefix in NATIVE_PREFIXES:
        if not any(job["name"].startswith(prefix) for job in native):
            raise ValueError("native platform coverage missing: " + prefix)
    return [job["name"] for job in native]


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(output("gh", "api", f"repos/{args.repo}/actions/runs/{args.run_id}"))
    if run["status"] != "completed" or run["path"] != ".github/workflows/release.yml":
        raise ValueError("source run is not a completed release workflow")
    source = run["head_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", source) or output("git", "rev-parse", "HEAD") != args.current_sha:
        raise ValueError("source/current commit identity mismatch")
    subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", source], check=True)
    changed = output("git", "diff", "--name-only", source, args.current_sha).splitlines()
    old = yaml.safe_load(output("git", "show", source + ":.github/workflows/release.yml"))
    current = yaml.safe_load(output("git", "show", args.current_sha + ":.github/workflows/release.yml"))
    jobs = json.loads(output("gh", "api", f"repos/{args.repo}/actions/runs/{args.run_id}/jobs?filter=latest&per_page=100"))["jobs"]
    native = validate_inputs(changed, old, current, jobs)
    receipt = dict(status="PASS", scope="assembly-only diagnostic; not a publishable version-tag build",
                   source_run=args.run_id, source_sha=source, current_sha=args.current_sha,
                   packaging_only_changes=changed, successful_native_jobs=native)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("ASSEMBLY_REUSE_INPUTS_VERIFIED")


if __name__ == "__main__":
    main()
