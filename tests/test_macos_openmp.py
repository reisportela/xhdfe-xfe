"""Local control tests, never evidence of native macOS/OpenMP execution.

Run with ``python3 -B tests/test_macos_openmp.py --out
provenance/openmp_release_20260913/macos_controls/run_001``.  Every output is
retained in that new directory.  The real builders run only against synthetic
SDKs and mock tools; no estimator is compiled, loaded, or run.
"""

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ROOT / "provenance/openmp_release_20260913/macos_controls"
INPUTS = (
    "tools/validate_macos_openmp.py",
    "stata/tools/macos-openmp.sh",
    "stata/tools/build-plugin.sh",
    "stata/tools/build-xfepout-plugin.sh",
    "stata/tools/cuda-common.sh",
)
CPU = {"x86_64": 0x01000007, "arm64": 0x0100000C}
TARGET = {"x86_64": "x86_64-apple-macos10.12", "arm64": "arm64-apple-macos11"}
RUNTIME = "xhdfe_libomp.dylib"
OUT = None
VALIDATOR = None


def setUpModule():
    if OUT is None:
        raise unittest.SkipTest("Retained control evidence requires the documented --out CLI invocation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data, executable=False):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError(f"Control output escaped its directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o700 if executable else 0o600
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data.encode() if isinstance(data, str) else data)
    return path


def write_json(path, value):
    return write_new(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


# Independent Mach-O fixtures: real headers/load commands, no machine code.
# These constants/encoders deliberately do not use the validator's parser.
def path_command(code, name):
    raw = name.encode() + b"\0"
    header_size = 12 if code == 0x8000001C else 24
    size = (header_size + len(raw) + 7) // 8 * 8
    header = struct.pack("<III", code, size, header_size)
    if header_size == 24:
        header += struct.pack("<III", 0, 0x10000, 0x10000)
    return (header + raw).ljust(size, b"\0")


def thin(arch, kind="plugin", *, minimum=None, dependencies=None, rpaths=None,
         runtime_id="@rpath/xhdfe_libomp.dylib", payload=b"SYNTHETIC CONTROL"):
    minimum = minimum or ((10, 12, 0) if arch == "x86_64" else (11, 0, 0))
    version = minimum[0] << 16 | minimum[1] << 8 | minimum[2]
    if dependencies is None:
        dependencies = ["/usr/lib/libSystem.B.dylib"]
        if kind == "plugin":
            dependencies += ["@rpath/xhdfe_libomp.dylib"]
    if rpaths is None:
        rpaths = ["@loader_path"] if kind == "plugin" else []
    commands = [path_command(0xC, value) for value in dependencies]
    commands += [path_command(0x8000001C, value) for value in rpaths]
    if kind == "runtime" and runtime_id is not None:
        commands.append(path_command(0xD, runtime_id))
    if arch == "x86_64":
        commands.append(struct.pack("<4I", 0x24, 16, version, 0x000F0000))
    else:
        commands.append(struct.pack("<6I", 0x32, 24, 1, version, 0x000F0000, 0))
    contents = b"".join(commands)
    header = struct.pack("<8I", 0xFEEDFACF, CPU[arch], 3 if arch == "x86_64" else 0,
                         8 if kind == "plugin" else 6, len(commands), len(contents), 0, 0)
    return header + contents + payload


def fat(slices, *, wide=False, order=">"):
    width = 32 if wide else 20
    header = struct.pack(order + "II", 0xCAFEBABF if wide else 0xCAFEBABE, len(slices))
    offset = ((8 + width * len(slices) + 4095) // 4096) * 4096
    entries, placements = [], []
    for blob in slices:
        cpu, subtype = struct.unpack_from("<II", blob, 4)
        fields = (cpu, subtype, offset, len(blob), 12)
        entries.append(struct.pack(order + ("IIQQII" if wide else "IIIII"),
                                   *(fields + (0,) if wide else fields)))
        placements.append((offset, blob))
        offset = ((offset + len(blob) + 4095) // 4096) * 4096
    contents = bytearray(header + b"".join(entries))
    for position, blob in placements:
        contents.extend(b"\0" * (position - len(contents)))
        contents.extend(blob)
    return bytes(contents)


def snapshot(count):
    return {
        "request": count, "rc": 0, "numeric_ok": True, "inputs_unchanged": True,
        "scalars": {
            "probe_requested": count, "probe_effective": count,
            "probe_used": count, "probe_workers": count,
            "probe_openmp": 1, "probe_converged": 1, "probe_gpu": 0,
        },
    }


def simulated_receipts(files):
    records = []
    for arch in CPU:
        for kind in ("xhdfe", "xfepout"):
            record = {
                "schema": "xhdfe-native-openmp-receipt-v1",
                "control_fixture": "SIMULATED; NOT NATIVE EXECUTION",
                "controller_platform": "darwin", "controller_arch": arch,
                "expected_arch": arch, "plugin_kind": kind,
                "plugin_path": f"/synthetic/installed/x/{kind}.plugin",
                "plugin_sha256": files[kind + ".plugin"], "plugin_unchanged": True,
                "expect_serial": False, "stage": "final", "status": "PASS",
                "loaded_runtime_path": "/synthetic/installed/x/xhdfe_libomp.dylib",
                "loaded_runtime_sha256": files[RUNTIME], "runtime_identity_pass": True,
                "release_artifact_gate_pass": True,
                "native": {
                    "process_arch": arch, "translated": 0, "release_gate_pass": True,
                    "serial_negative_pass": False, "numerical_parity_pass": True,
                    "full_numeric_parity_max_error": 0.0,
                    "thread1": snapshot(1), "thread2": snapshot(2),
                },
            }
            records.append(record)
            serial = copy.deepcopy(record)
            serial.update(expect_serial=True, stage="thin", status="EXPECTED_SERIAL_REJECTION",
                          release_artifact_gate_pass=False, runtime_identity_pass=False,
                          loaded_runtime_path=None, loaded_runtime_sha256=None,
                          plugin_sha256=hashlib.sha256((arch + kind + "serial").encode()).hexdigest())
            serial["native"].update(release_gate_pass=False, serial_negative_pass=True)
            serial["native"]["thread1"]["scalars"]["probe_openmp"] = 0
            serial["native"]["thread2"] = {"request": 2, "rc": 498, "numeric_ok": False}
            records.append(serial)
    return records


def package(directory):
    files, inspection = {}, {}
    for name, kind in (("xhdfe.plugin", "plugin"), ("xfepout.plugin", "plugin"),
                       (RUNTIME, "runtime")):
        path = write_new(directory / name, fat([
            thin(arch, kind, payload=("SIMULATED " + name).encode()) for arch in CPU
        ]))
        files[name] = sha(path)
        inspection[name] = VALIDATOR.inspect(path, kind)
    license_path = write_new(directory / "LLVM-OpenMP-LICENSE.txt", "Control placeholder, not a shipped license.\n")
    files[license_path.name] = sha(license_path)
    return {
        "schema": "xhdfe-macos-openmp-v1", "status": "NATIVE_VALIDATED",
        "scope": "SYNTHETIC CONTROL ONLY; NO NATIVE VALIDATION CLAIM",
        "llvm_revision": VALIDATOR.LLVM_REVISION, "files": files,
        "inspection": inspection, "native_receipts": simulated_receipts(files),
    }


class MachOControls(unittest.TestCase):
    def setUp(self):
        self.directory = OUT / self.id().split(".")[-1]

    def artifact(self, name, data):
        return write_new(self.directory / name, data)

    def test_thin_slices_and_deployment_commands(self):
        for arch in CPU:
            for kind in ("plugin", "runtime"):
                with self.subTest(arch=arch, kind=kind):
                    path = self.artifact(arch + "_" + kind, thin(arch, kind))
                    info = VALIDATOR.inspect(path, kind, arch)
                    self.assertEqual(set(info), {arch})
                    self.assertEqual(info[arch]["minimum_macos"],
                                     [10, 12, 0] if arch == "x86_64" else [11, 0, 0])

    def test_universal_encodings(self):
        for wide in (False, True):
            for order in (">", "<"):
                for kind in ("plugin", "runtime"):
                    with self.subTest(wide=wide, order=order, kind=kind):
                        data = fat([thin(arch, kind) for arch in CPU], wide=wide, order=order)
                        path = self.artifact(f"fat_{wide}_{ord(order)}_{kind}", data)
                        self.assertEqual(set(VALIDATOR.inspect(path, kind)), set(CPU))

    def test_missing_architecture_rejected(self):
        for arch in CPU:
            with self.subTest(arch=arch):
                path = self.artifact(arch, thin(arch))
                with self.assertRaisesRegex(ValueError, "both architectures"):
                    VALIDATOR.inspect(path, "plugin")
                other = "arm64" if arch == "x86_64" else "x86_64"
                with self.assertRaisesRegex(ValueError, "Missing .* slice"):
                    VALIDATOR.inspect(path, "plugin", other)

    def test_dependency_and_rpath_rejections(self):
        cases = [
            ("homebrew", "plugin", {"dependencies": ["/opt/homebrew/opt/libomp/lib/libomp.dylib"]}),
            ("absolute", "plugin", {"dependencies": ["/private/build/lib/xhdfe_libomp.dylib"]}),
            ("external_rpath", "plugin", {"rpaths": ["@loader_path", "/opt/homebrew/lib"]}),
            ("serial", "plugin", {"dependencies": ["/usr/lib/libSystem.B.dylib"]}),
            ("wrong_id", "runtime", {"runtime_id": "/opt/homebrew/lib/libomp.dylib"}),
            ("missing_id", "runtime", {"runtime_id": None}),
            ("runtime_external", "runtime", {"dependencies": ["/usr/local/lib/libunwind.dylib"]}),
            ("runtime_rpath", "runtime", {"rpaths": ["/private/build/lib"]}),
        ]
        for arch in CPU:
            for name, kind, options in cases:
                with self.subTest(arch=arch, case=name):
                    path = self.artifact(arch + "_" + name, thin(arch, kind, **options))
                    with self.assertRaises(ValueError):
                        VALIDATOR.inspect(path, kind, arch)

    def test_raised_minimum_rejected(self):
        for arch, version in (("x86_64", (10, 13, 0)), ("arm64", (11, 1, 0))):
            for kind in ("plugin", "runtime"):
                with self.subTest(arch=arch, kind=kind):
                    path = self.artifact(arch + kind, thin(arch, kind, minimum=version))
                    with self.assertRaisesRegex(ValueError, "minimum macOS version was raised"):
                        VALIDATOR.inspect(path, kind, arch)

    def test_malformed_headers_rejected(self):
        truncated = thin("arm64")[:40]
        bad_size = bytearray(thin("arm64"))
        struct.pack_into("<I", bad_size, 36, 4)
        mismatch = bytearray(fat([thin("x86_64"), thin("arm64")]))
        struct.pack_into(">I", mismatch, 8, CPU["arm64"])
        for name, data in (("truncated", truncated), ("bad_command", bad_size),
                           ("wrong_fat_cpu", mismatch), ("duplicate", fat([thin("arm64")] * 2))):
            with self.subTest(case=name):
                path = self.artifact(name, data)
                with self.assertRaises(ValueError):
                    VALIDATOR.macho(path)


class ReceiptControls(unittest.TestCase):
    def verify_record(self, name, transform=None):
        directory = OUT / self.id().split(".")[-1] / name
        record = package(directory)
        if transform:
            transform(record)
        write_json(directory / "macos-openmp-manifest.json", record)
        return VALIDATOR.verify(directory / "xhdfe.plugin", directory / "xfepout.plugin", directory)

    def test_complete_eight_receipt_manifest(self):
        record = self.verify_record("eight_simulated_receipts")
        self.assertEqual(len(record["native_receipts"]), 8)
        self.assertEqual(sum(r["expect_serial"] for r in record["native_receipts"]), 4)

    def test_positive_receipt_rejections(self):
        # Every case keeps top-level PASS and mutates one decisive observation.
        changes = [
            ("workers_one", ("native", "thread2", "scalars", "probe_workers"), 1),
            ("used_one", ("native", "thread2", "scalars", "probe_used"), 1),
            ("other_plugin_bytes", ("plugin_sha256",), "0" * 64),
            ("other_runtime_bytes", ("loaded_runtime_sha256",), "0" * 64),
            ("translated", ("native", "translated"), 1),
            ("process_arch", ("native", "process_arch"), "arm64"),
            ("controller_arch", ("controller_arch",), "arm64"),
            ("wrong_platform", ("controller_platform",), "linux"),
            ("thin_not_final", ("stage",), "thin"),
            ("external_runtime", ("loaded_runtime_path",), "/opt/homebrew/lib/xhdfe_libomp.dylib"),
            ("inputs_changed", ("native", "thread2", "inputs_unchanged"), False),
            ("numeric_failure", ("native", "thread2", "numeric_ok"), False),
            ("thread_disagreement", ("native", "full_numeric_parity_max_error"), 1e-4),
            ("serial_as_positive", ("native", "thread2", "scalars", "probe_openmp"), 0),
        ]
        for name, keys, value in changes:
            def mutate(record, keys=keys, value=value):
                target = record["native_receipts"][0]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.verify_record(name, mutate)

    def test_incomplete_and_duplicate_coverage_rejected(self):
        for index in range(8):
            with self.subTest(missing_receipt=index), self.assertRaisesRegex(ValueError, "both plugins"):
                self.verify_record(f"missing_{index}", lambda r, i=index: r["native_receipts"].pop(i))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.verify_record("duplicate", lambda r: r["native_receipts"].append(r["native_receipts"][0]))

    def test_false_serial_control_rejected(self):
        def mutate(record):
            record["native_receipts"][1]["native"]["thread1"]["scalars"]["probe_openmp"] = 1
        with self.assertRaisesRegex(ValueError, "built with OpenMP"):
            self.verify_record("false_serial", mutate)

    def test_artifact_manifest_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match artifact bytes"):
            self.verify_record("wrong_manifest_hash", lambda r: r["files"].update({"xhdfe.plugin": "0" * 64}))


def mock_tool(name, args):
    """Only used by newly created fixture launchers; never invokes a compiler."""
    global OUT
    OUT = Path(os.environ["MACOS_CONTROL_OUT"])
    with Path(os.environ["MACOS_CONTROL_TRACE"]).open("a") as log:
        log.write(json.dumps({"tool": name, "args": args}) + "\n")
    if name == "uname":
        print("Darwin" if args == ["-s"] else os.environ["MACOS_CONTROL_ARCH"])
    elif name in ("curl", "wget", "tar"):
        raise RuntimeError(f"A control attempted a forbidden download/extraction: {name}")
    elif name == "cxx":
        output = Path(args[args.index("-o") + 1])
        is_plugin = "-bundle" in args
        toolchain = output.name.startswith("openmp-toolchain-")
        fail = os.environ.get("MACOS_CONTROL_FAIL", "")
        if (fail == "plugin" and is_plugin) or (fail == "toolchain" and toolchain):
            return 73
        if is_plugin:
            target = args[args.index("-target") + 1]
            arch = target.split("-", 1)[0]
            version = tuple(int(x) for x in target.split("macos", 1)[1].split("."))
            version = (version + (0, 0))[:3]
            dependencies = ["/usr/lib/libSystem.B.dylib"]
            if "-fopenmp" in args and "-DHDFE_USE_OPENMP" in args:
                dependencies.append("@rpath/xhdfe_libomp.dylib")
            paths = [x.removeprefix("-Wl,-rpath,") for x in args if x.startswith("-Wl,-rpath,")]
            write_new(output, thin(arch, minimum=version, dependencies=dependencies,
                                   rpaths=paths, payload=output.name.encode()))
        else:
            # The builder can execute its mock liveness/team checks, not machine code.
            command = [sys.executable, str(Path(__file__).resolve()), "--mock-tool", "executed", str(output)]
            write_new(output, "#!/bin/sh\nexec " + shlex.join(command) + "\n", executable=True)
    elif name == "lipo":
        index = args.index("-output")
        output = Path(args[index + 1])
        inputs = [Path(value) for i, value in enumerate(args)
                  if i not in (index, index + 1) and value != "-create"]
        data = [path.read_bytes() for path in inputs]
        write_new(output, data[0] if len(data) == 1 else fat(data))
    elif name not in ("codesign", "strip", "executed"):
        raise RuntimeError(f"Unknown mock tool: {name}")
    return 0


def builder_case(name, kind, *, arches="x86_64 arm64", mode="--openmp", release=True,
                 fail="", sdk_arches=("x86_64", "arm64"), host_arch="arm64"):
    directory = OUT / "builders" / name
    fixture = directory / "tree"
    for relative in INPUTS:
        write_new(fixture / relative, (OUT / "originals" / relative).read_bytes())
    for relative in ("stata/tools/_deps/stplugin.h", "stata/tools/_deps/stplugin.c",
                     "stata/tools/_deps/eigen-3.4.0.tar.gz",
                     "stata/tools/_build/eigen-3.4.0/Eigen/Dense"):
        write_new(fixture / relative, "SYNTHETIC DEPENDENCY; NEVER COMPILED\n")
    for arch in sdk_arches:
        write_new(fixture / "sdk" / arch / "include/omp.h", "SYNTHETIC HEADER\n")
        write_new(fixture / "sdk" / arch / "lib" / RUNTIME, thin(arch, "runtime"))
    bindir = directory / "mockbin"
    for tool in ("uname", "cxx", "lipo", "codesign", "strip", "curl", "wget", "tar"):
        command = [sys.executable, str(Path(__file__).resolve()), "--mock-tool", tool]
        write_new(bindir / tool, "#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n', executable=True)
    trace = write_new(directory / "calls.jsonl", "")
    temporary = directory / "tmp"
    temporary.mkdir()
    environment = {
        "PATH": str(bindir) + os.pathsep + os.defpath,
        "CXX": str(bindir / "cxx"), "STRIP_BIN": str(bindir / "strip"),
        "XHDFE_OPENMP_ROOT": str(fixture / "sdk"), "XHDFE_MACOS_ARCHS": arches,
        "XHDFE_RELEASE_BUILD": "1" if release else "0", "XHDFE_ENABLE_METAL": "OFF",
        "MACOS_CONTROL_OUT": str(OUT), "MACOS_CONTROL_TRACE": str(trace),
        "MACOS_CONTROL_ARCH": host_arch, "MACOS_CONTROL_FAIL": fail,
        "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(temporary), "LC_ALL": "C",
    }
    builder = "build-plugin.sh" if kind == "xhdfe" else "build-xfepout-plugin.sh"
    command = ["/bin/bash", str(fixture / "stata/tools" / builder), "--linux", mode, "--no-cuda"]
    started = time.monotonic()
    result = subprocess.run(command, cwd=fixture, env=environment, capture_output=True, text=True, timeout=30)
    write_new(directory / "stdout.txt", result.stdout)
    write_new(directory / "stderr.txt", result.stderr)
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    write_json(directory / "RESULT.json", {
        "scope": "BUILD ARGUMENT/CONTROL FLOW MOCK; NO COMPILATION OR NATIVE EXECUTION",
        "command": command, "rc": result.returncode, "elapsed_seconds": time.monotonic() - started,
        "kind": kind, "arches": arches, "mock_host_arch": host_arch, "mode": mode,
        "injected_failure": fail, "sources": {rel: sha(fixture / rel) for rel in INPUTS},
    })
    if any(call["tool"] in ("curl", "wget", "tar") for call in calls):
        raise AssertionError("The builder attempted a download/extraction")
    return fixture, result, calls


class BuilderControls(unittest.TestCase):
    def test_real_builders_pass_correct_architecture_flags(self):
        for kind in ("xhdfe", "xfepout"):
            for arches in ("x86_64 arm64", "arm64", "x86_64"):
                with self.subTest(kind=kind, arches=arches):
                    host = arches.split()[-1]
                    fixture, result, calls = builder_case(kind + "_" + arches.replace(" ", "_"), kind,
                                                         arches=arches, host_arch=host)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    compiles = [c["args"] for c in calls if c["tool"] == "cxx" and "-bundle" in c["args"]]
                    self.assertEqual(len(compiles), len(arches.split()))
                    for args, arch in zip(compiles, arches.split()):
                        self.assertEqual(args[args.index("-target") + 1], TARGET[arch])
                        self.assertIn("-DHDFE_USE_OPENMP", args)
                        self.assertEqual(args[args.index("-Xpreprocessor") + 1], "-fopenmp")
                        self.assertIn(str(fixture / "sdk" / arch / "lib" / RUNTIME), args)
                        self.assertIn("-I" + str(fixture / "sdk" / arch / "include"), args)
                        self.assertIn("-Wl,-rpath,@loader_path", args)
                        self.assertNotIn("-march=native", args)
                        for value in args:
                            if value.startswith("-Wl,-rpath,"):
                                self.assertEqual(value, "-Wl,-rpath,@loader_path")
                    probes = [c["args"] for c in calls if c["tool"] == "cxx" and
                              any("openmp-toolchain-" in a and a.endswith(".cpp") for a in c["args"])]
                    self.assertEqual(len(probes), len(arches.split()))
                    for args, arch in zip(probes, arches.split()):
                        self.assertEqual(args[args.index("-target") + 1], TARGET[arch])
                        self.assertIn("-fopenmp", args)
                    executed = [c["args"][0] for c in calls if c["tool"] == "executed"]
                    self.assertEqual(sum("openmp-toolchain-" in x for x in executed), 1)
                    self.assertTrue(any(x.endswith("openmp-toolchain-" + host) for x in executed))
                    plugin = fixture / "stata" / (kind + ".plugin")
                    runtime = fixture / "stata" / RUNTIME
                    self.assertEqual(set(VALIDATOR.macho(plugin)), set(arches.split()))
                    self.assertEqual(set(VALIDATOR.macho(runtime)), set(arches.split()))
                    self.assertTrue(any(c["tool"] == "codesign" and str(plugin) in c["args"] for c in calls))

    def test_compile_failure_never_retries_serial(self):
        for kind in ("xhdfe", "xfepout"):
            for failure in ("toolchain", "plugin"):
                with self.subTest(kind=kind, failure=failure):
                    fixture, result, calls = builder_case(kind + "_fail_" + failure, kind, fail=failure)
                    self.assertEqual(result.returncode, 73, result.stderr)
                    compiles = [c["args"] for c in calls if c["tool"] == "cxx" and "-bundle" in c["args"]]
                    self.assertEqual(len(compiles), int(failure == "plugin"))
                    self.assertTrue(all("-fopenmp" in args for args in compiles))
                    self.assertFalse(any(c["tool"] == "lipo" for c in calls))
                    self.assertFalse((fixture / "stata" / (kind + ".plugin")).exists())

    def test_release_rejects_serial_before_compilation(self):
        for kind in ("xhdfe", "xfepout"):
            with self.subTest(kind=kind):
                fixture, result, calls = builder_case(kind + "_serial_release", kind, mode="--no-openmp")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("release plugins require OpenMP", result.stderr)
                self.assertFalse(any(c["tool"] in ("cxx", "lipo") for c in calls))
                self.assertFalse((fixture / "stata" / (kind + ".plugin")).exists())

    def test_partial_sdk_cannot_produce_universal(self):
        for kind in ("xhdfe", "xfepout"):
            with self.subTest(kind=kind):
                fixture, result, calls = builder_case(kind + "_partial_sdk", kind, sdk_arches=("x86_64",))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("missing macOS OpenMP header/runtime for arm64", result.stderr)
                self.assertFalse(any(c["tool"] == "lipo" for c in calls))
                self.assertFalse((fixture / "stata" / (kind + ".plugin")).exists())

    def test_serial_diagnostic_stays_outside_release_gate(self):
        for kind in ("xhdfe", "xfepout"):
            with self.subTest(kind=kind):
                fixture, result, calls = builder_case(kind + "_serial_diagnostic", kind,
                                                     mode="--no-openmp", release=False, sdk_arches=())
                self.assertEqual(result.returncode, 0, result.stderr)
                compiles = [c["args"] for c in calls if c["tool"] == "cxx" and "-bundle" in c["args"]]
                self.assertEqual(len(compiles), 2)
                self.assertTrue(all("-fopenmp" not in args for args in compiles))
                self.assertFalse((fixture / "stata" / RUNTIME).exists())
                with self.assertRaisesRegex(ValueError, "packaged OpenMP runtime"):
                    VALIDATOR.inspect(fixture / "stata" / (kind + ".plugin"), "plugin")


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.observations.append({"test": test.id(), "status": "PASS"})

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        self.observations.append({"test": subtest.id(), "status": "FAIL" if error else "PASS",
                                  "error": str(error[1]) if error else None})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.observations.append({"test": test.id(), "status": "FAIL", "error": str(error[1])})

    def addError(self, test, error):
        super().addError(test, error)
        self.observations.append({"test": test.id(), "status": "ERROR", "error": str(error[1])})


def main():
    global OUT, VALIDATOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory under macos_controls")
    args = parser.parse_args()
    OUT = args.out.resolve()
    if not OUT.is_relative_to(ALLOWED.resolve()) or OUT == ALLOWED.resolve():
        parser.error(f"--out must be a new subdirectory of {ALLOWED}")
    for path in (args.out, *args.out.absolute().parents):
        if path.is_symlink():
            parser.error("Output paths may not contain symlinks")
    OUT.mkdir(parents=True, exist_ok=False)
    originals = {}
    for relative in INPUTS:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular source: {source}")
        copied = write_new(OUT / "originals" / relative, source.read_bytes())
        originals[relative] = sha(copied)
    write_new(OUT / "originals/tests/test_macos_openmp.py", Path(__file__).read_bytes())
    write_json(OUT / "SOURCES.json", {"inputs": originals, "test_sha256": sha(__file__)})
    specification = importlib.util.spec_from_file_location("macos_validator_under_test", OUT / "originals" / INPUTS[0])
    VALIDATOR = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(VALIDATOR)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (MachOControls, ReceiptControls, BuilderControls))
    stream = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    write_new(OUT / "TEST_LOG.txt", stream.getvalue())
    unchanged = originals == {relative: sha(ROOT / relative) for relative in INPUTS}
    receipt = {
        "status": "PASS" if result.wasSuccessful() and unchanged else "FAIL",
        "classification": "evaluation-harness controls",
        "scope": "Synthetic Mach-O/receipts and real builder control flow with mock tools only",
        "native_macos_validation": "NOT RUN", "estimator_compilation": "NOT RUN",
        "estimator_fits": 0, "download_attempts_allowed": 0,
        "test_cases": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "observations": result.observations, "production_sources_unchanged": unchanged,
        "input_hashes": originals, "test_sha256": sha(__file__),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUT / "RECEIPT.json", receipt)
    print(stream.getvalue(), end="")
    print(f"{receipt['status']}: local controls only; receipt {OUT / 'RECEIPT.json'}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mock-tool":
        raise SystemExit(mock_tool(sys.argv[2], sys.argv[3:]))
    raise SystemExit(main())
