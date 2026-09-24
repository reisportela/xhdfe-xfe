"""Offline falsification of static Stata provenance and system-only PE closure.

PE imports are simulated here; native loading/parallel work is a separate CI gate.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import windows_stata_linkage as linkage
from tools import validate_python_release_artifacts as pe
from tools import build_corresponding_source_bundle as builder
from tools import validate_corresponding_source_bundle as source_validator
import test_corresponding_source_bundle as source_fixtures


def fixture(root):
    names = sorted(linkage.PLUGIN_NAMES)
    plugins = []
    for name in names:
        path = root / name
        path.write_bytes(name.encode())
        plugins.append(dict(name=name, sha256=linkage.digest(path), size=path.stat().st_size,
                            link_map_sha256="a" * 64))
    archives = []
    for name in ("libgcc.a", "libstdc++.a", "libgomp.a", "libwinpthread.a"):
        mingw = name == "libwinpthread.a"
        archives.append(dict(name=name, source_path="/toolchain/" + name,
            sha256=hashlib.sha256(name.encode()).hexdigest(), size=len(name),
            runtime_package="test-package", runtime_version="1-test",
            source_package="mingw-w64" if mingw else "gcc-mingw-w64",
            source_version="11.0.1-test" if mingw else "13.2.0-test",
            built_using="", provider_id="ubuntu-mingw-w64" if mingw else "ubuntu-mingw-gcc",
            used_by=names))
    ledger = dict(schema_version=1, artifact="xhdfe-xfe-stata-windows-cpu", linkage="static",
        compiler=dict(target="x86_64-w64-mingw32", version="13.2.0"),
        entries=[], static_archives=archives, plugins=plugins)
    closure = dict(format="xhdfe-windows-runtime-closure-v1", runtimes=[], roots=[
        dict(architecture="pei-x86-64", dependencies=["KERNEL32.dll", "msvcrt.dll"],
             member=p["name"], member_sha256=p["sha256"], size=p["size"]) for p in plugins])
    return ledger, closure


class StaticStataTests(unittest.TestCase):
    def test_netinstall_static_layout_and_tampered_plugin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger, closure = fixture(root)
            runtime = root / "empty"
            runtime.mkdir()
            provider = root / "provider.json"
            provider.write_text(json.dumps(ledger))
            graph = root / "closure.json"
            graph.write_text(json.dumps(closure))
            site = root / "site"
            repo = Path(__file__).resolve().parents[1]
            run = subprocess.run(["bash", str(repo / "tools/stage_stata_netinstall_site.sh"), str(site),
                "--windows-xhdfe", str(root / "xhdfe.plugin.windows"),
                "--windows-xfepout", str(root / "xfepout.plugin.windows"),
                "--windows-runtime-dir", str(runtime), "--windows-runtime-provider-ledger", str(provider),
                "--windows-runtime-closure-ledger", str(graph)], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for name in ("xhdfe", "xfepout"):
                manifest = (site / (name + ".pkg")).read_text()
                self.assertIn(f"g WIN64 {name}.win64.plugin {name}.plugin", manifest)
                self.assertNotIn(".dll", manifest)
            (site / "xhdfe.win64.plugin").write_bytes(b"different bytes")
            check = subprocess.run(["bash", str(repo / "tools/validate_stata_package_site.sh"), str(site)],
                                   text=True, capture_output=True)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn("differs from static link evidence", check.stderr)

    def test_pair_hashes_and_each_runtime_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger, closure = fixture(root)
            linkage.validate(ledger, root)
            linkage.validate_closure(ledger, closure)
            for name in ("libgcc.a", "libstdc++.a", "libgomp.a", "libwinpthread.a"):
                bad = copy.deepcopy(ledger)
                bad["static_archives"] = [a for a in bad["static_archives"] if a["name"] != name]
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    linkage.validate(bad)
            (root / "xhdfe.plugin.windows").write_bytes(b"substituted plugin")
            with self.assertRaisesRegex(ValueError, "differs"):
                linkage.validate(ledger, root)

    def test_static_closure_rejects_hidden_dependency_and_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger, closure = fixture(Path(temp))
            bad = copy.deepcopy(closure)
            bad["roots"][0]["dependencies"].append("libgomp-1.dll")
            with self.assertRaisesRegex(ValueError, "non-system"):
                linkage.validate_closure(ledger, bad)
            closure["roots"][0]["member_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hashes disagree"):
                linkage.validate_closure(ledger, closure)

    def test_system_only_inspects_imports_and_does_not_weaken_dynamic_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, closure = fixture(root)
            objdump = root / "objdump"
            objdump.touch()
            kwargs = dict(roots={n: (root / n).read_bytes() for n in linkage.PLUGIN_NAMES},
                runtimes={}, ledger_bytes=json.dumps(closure).encode(), objdump=objdump,
                python_host_dll_names=frozenset(), label="static fixture")
            with patch.object(pe, "_pe_architecture", return_value="pei-x86-64"), patch.object(
                    pe, "_pe_dependencies", return_value=["KERNEL32.dll", "msvcrt.dll"]):
                pe._validate_pe_runtime_ledger(**kwargs, system_only=True)
                with self.assertRaisesRegex(Exception, "no runtime DLLs"):
                    pe._validate_pe_runtime_ledger(**kwargs)
            with patch.object(pe, "_pe_architecture", return_value="pei-x86-64"), patch.object(
                    pe, "_pe_dependencies", return_value=["libgomp-1.dll"]):
                with self.assertRaisesRegex(Exception, "dependencies do not match"):
                    pe._validate_pe_runtime_ledger(**kwargs, system_only=True)

    def test_corresponding_source_static_roundtrip_and_missing_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = source_fixtures.CorrespondingSourceBundleTests()._inputs(root)
            ledger, _ = fixture(root)
            path = root / "static.json"
            path.write_text(json.dumps(ledger))
            args.windows_stata_static_ledger = path
            for field in ("runtime_binary", "runtime_provider", "provider_binary", "metadata"):
                setattr(args, field, [v for v in getattr(args, field) if not v.startswith("windows-stata-")])
            builder.build_archive(args)
            provenance = source_validator.validate(Path(args.output))
            self.assertEqual(provenance["schema_version"], 2)
            self.assertEqual(provenance["windows_stata_static_link"], ledger)
            ledger["static_archives"] = [a for a in ledger["static_archives"] if a["name"] != "libgomp.a"]
            path.write_text(json.dumps(ledger))
            args.output = str(root / "bad.zip")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                builder.build_archive(args)


if __name__ == "__main__":
    unittest.main()
