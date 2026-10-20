"""Reject an OpenMP runtime that can expose private symbols to a Stata host.

Run on macOS for each final slice. The separate Mach-O dependency/receipt
validator remains portable and continues to require native execution evidence.

With ``--plugin`` (repeatable), every OpenMP-family symbol that a plugin imports
(public API, compiler entry points, GNU compatibility, OMPT, and also internal
``__kmp_*`` names) must be exported by the runtime, so a compiler emitting a new
``__kmpc_*`` entry point, or a build that leaks a dependency on an internal
symbol, fails here instead of at load time. ``XHDFE_MACOS_NM`` overrides the
``nm`` binary (``llvm-nm`` accepts the same cctools-style flags), which lets the
logic run against Mach-O files on any host. ``--self-test`` exercises the pure
classification with positive and negative controls and needs no binaries.
"""
import argparse
import fnmatch
import os
from pathlib import Path
import subprocess

# Any imported symbol in these families is an OpenMP dependency that the
# runtime must satisfy. Internal names (___kmp_*, __kmp_*) are recognised so
# that a plugin depending on them is refused: the export list never allows them.
OPENMP_FAMILIES = ("_omp_", "_OMP_", "_ompc_", "_ompt_", "_kmp_", "_kmpc_",
                   "___kmpc_", "___kmp_", "__kmp_", "_GOMP_", "__You_must_link_with_")
REQUIRED = {"_omp_get_max_threads", "_omp_set_num_threads", "___kmpc_fork_call"}


def nm_names(nm, arch, flags, path):
    output = subprocess.check_output([nm, "-arch", arch, flags, str(path)], text=True)
    return {line.strip() for line in output.splitlines() if line.strip()}


def export_patterns(path=None):
    path = path or Path(__file__).with_name("macos_openmp.exports")
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def is_openmp_symbol(name):
    return name.startswith(OPENMP_FAMILIES)


def preview(names, limit=8):
    names = sorted(names)
    shown = ", ".join(names[:limit])
    return f"{len(names)} [{shown}{', ...' if len(names) > limit else ''}]"


def runtime_failures(symbols, patterns):
    """Pure check of a runtime's exported symbol set against the export list."""
    failures = []
    missing = REQUIRED - symbols
    if missing:
        failures.append(f"missing required exports {sorted(missing)}")
    unexpected = [name for name in symbols
                  if not any(fnmatch.fnmatchcase(name, p) for p in patterns)]
    if unexpected:
        failures.append(f"private exports {preview(unexpected)}")
    return failures


def plugin_failures(label, imports, symbols):
    """Pure check: every OpenMP-family import of a plugin is exported by the runtime."""
    openmp_imports = {name for name in imports if is_openmp_symbol(name)}
    uncovered = openmp_imports - symbols
    failures = []
    if not openmp_imports:
        failures.append(f"{label}: no OpenMP imports found")
    if uncovered:
        failures.append(f"{label}: imports missing from the runtime {preview(uncovered)}")
    return failures, len(openmp_imports), len(uncovered)


def check(runtime: Path, arch: str, plugins, nm: str) -> None:
    patterns = export_patterns()
    # -g: external; -U: defined only; -j: just the symbol names.
    symbols = nm_names(nm, arch, "-gUj", runtime)
    failures = runtime_failures(symbols, patterns)
    for plugin in plugins:
        # -u: undefined only; -j: names.
        imports = nm_names(nm, arch, "-uj", plugin)
        plugin_fail, total, uncovered = plugin_failures(plugin.name, imports, symbols)
        failures.extend(plugin_fail)
        print(f"OpenMP import coverage ({arch}): {plugin.name} {total - uncovered}/{total} covered")
    if failures:
        raise SystemExit(f"OpenMP export isolation failed ({arch}): " + "; ".join(failures))
    print(f"OpenMP export isolation OK: {arch}, {len(symbols)} public ABI symbols")


def self_test() -> None:
    patterns = export_patterns()
    public = {"_omp_get_max_threads", "_omp_set_num_threads", "___kmpc_fork_call",
              "___kmpc_barrier", "_GOMP_parallel", "_ompc_set_num_threads", "_kmp_set_defaults"}
    # Positive control: a clean runtime and a plugin whose imports are covered.
    assert runtime_failures(public, patterns) == []
    fails, total, uncovered = plugin_failures("ok.plugin", public | {"_malloc", "__ZN3std3fooEv"}, public)
    assert fails == [] and total == len(public) and uncovered == 0
    # Negative controls: private C++ and internal exports are refused ...
    assert runtime_failures(public | {"__Z16__kmp_suspend_64ILb0ELb1EEviP11kmp_flag_64IXT_EXT0_EE"}, patterns)
    assert runtime_failures(public | {"___kmp_fork_barrier"}, patterns)
    assert runtime_failures(public - {"___kmpc_fork_call"}, patterns)
    # ... and every OpenMP family of imports is classified, including OMPT and internals.
    for name in ("___kmpc_dispatch_init_4", "_ompt_libomp_connect", "___kmp_get_reduce_method",
                 "__kmp_debugging", "_GOMP_barrier", "_ompc_set_dynamic", "_kmp_malloc"):
        fails, total, uncovered = plugin_failures("bad.plugin", public | {name}, public)
        assert total == len(public) + 1 and uncovered == 1 and fails, name
    # Non-OpenMP imports are ignored.
    fails, total, uncovered = plugin_failures("libc.plugin", public | {"_pthread_mutexattr_settype"}, public)
    assert fails == [] and total == len(public)
    print("check_macos_openmp_exports self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--arch", choices=["arm64", "x86_64"])
    parser.add_argument("--plugin", type=Path, action="append", default=[],
                        help="plugin whose OpenMP imports must be exported by the runtime")
    parser.add_argument("--nm", default=os.environ.get("XHDFE_MACOS_NM", "/usr/bin/nm"))
    parser.add_argument("--self-test", action="store_true",
                        help="run the pure positive/negative controls and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.runtime is None or args.arch is None:
        parser.error("--runtime and --arch are required unless --self-test is given")
    else:
        check(args.runtime, args.arch, args.plugin, args.nm)
