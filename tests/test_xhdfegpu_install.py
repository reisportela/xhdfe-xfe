"""Protocol and rollback tests for the Stata CUDA installer without a real install."""

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "stata" / "xhdfegpu-install.sh"
VERSION = "2.35.1.20261125"
ADO_DATE = "25nov2026"
OLD = {"xhdfe": b"prior xhdfe\n", "xfepout": b"prior xfepout\n"}


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux installer protocol")
class InstallerProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="xhdfegpu fixture ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source with spaces"
        self.destination = self.root / "isolated plus" / "x"
        self.tmp = self.root / "temporary builds"
        self.bin = self.root / "mock bin"
        for directory in (self.destination, self.tmp, self.bin):
            directory.mkdir(parents=True)
        self.log = self.root / "builder arguments.txt"
        self.request = self.root / "request file.txt"
        self.receipt = self.root / "receipt file.txt"
        self.smoke = self.root / "smoke input.do"
        self.smoke.write_text("* Synthetic installer marker fixture.\n")
        self.fake_stata = self.bin / "fake stata"
        self.fake_stata.write_text(
            '#!/bin/sh\n[ "$1 $2 $3 $4" = "-q -b do smoke.do" ] || exit 3\n'
            '[ -f xhdfe.plugin ] || exit 4\n'
            'if [ "$XHDFEGPU_XFE" = 1 ]; then [ -f xfepout.plugin ] || exit 5; fi\n'
            'if [ "${XHDFEGPU_TEST_FAIL_SMOKE:-}" = 1 ]; then printf "r(9);\\n" > smoke.log; exit 0; fi\n'
            'printf "XHDFEGPU_VALIDATED\\n" > smoke.log\n'
        )
        self.fake_stata.chmod(0o700)
        self.env = {
            "PATH": os.pathsep.join((str(self.bin), "/usr/bin", "/bin")),
            "HOME": str(self.root),
            "LC_ALL": "C",
            "NVCC": shutil.which("false") or "/usr/bin/false",
            "CXX": shutil.which("false") or "/usr/bin/false",
            "XHDFEGPU_TEST_TRACE": str(self.log),
        }
        self.build_tree()
        for name, value in OLD.items():
            path = self.destination / f"{name}.plugin"
            path.write_bytes(value)
            path.chmod(0o640)
            shutil.copyfile(self.source / "stata" / f"{name}.ado", self.destination / f"{name}.ado")
        self.original = self.plugins()

    def build_tree(self):
        files = {
            "CMakeLists.txt": f"project(xhdfe VERSION {VERSION} LANGUAGES CXX)\n",
            "stata/src/synthetic-input.cpp": "// Stub build input.\n",
            "stata/include/synthetic-input.hpp": "// Stub build input.\n",
            "stata/tools/cuda-common.sh": "# Stub build input.\n",
            "stata/tools/_deps/eigen-3.4.0.tar.gz": "stub input\n",
            "stata/tools/_deps/stplugin.c": "stub input\n",
            "stata/tools/_deps/stplugin.h": "stub input\n",
            "tests/ieee_bits_liveness.cpp": "stub input\n",
            "tests/fma_liveness.cpp": "stub input\n",
            "tools/check_verifier_device_fma.sh": "stub input\n",
            "stata/xhdfe.ado": f"*! version 2.35.1 {ADO_DATE}\n",
            "stata/xfepout.ado": f"*! version 1.14.7 {ADO_DATE}\n",
        }
        for name, item in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item)
        for arm, name in (("xhdfe", "build-plugin.sh"), ("xfepout", "build-xfepout-plugin.sh")):
            script = self.source / "stata" / "tools" / name
            script.write_text(
                "#!/bin/sh\nset -eu\n"
                f'printf "%s|%s|%s|%s|%s|%s|%s\\n" "{arm}" "$*" "$XHDFE_ENABLE_CUDA" '
                '"$XHDFE_CUDA_ARCH" "$XHDFE_CUDA_ARCHS" "$XHDFE_ENABLE_MARCH_NATIVE" '
                '"$XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA" >> "$XHDFEGPU_TEST_TRACE"\n'
                f'[ "${{XHDFEGPU_TEST_FAIL_BUILD:-}}" != "{arm}" ] || exit 73\n'
                f'printf "built {arm} %s\\n" "$XHDFE_ENABLE_MARCH_NATIVE" > "stata/{arm}.plugin"\n'
            )

    def plugins(self):
        names = sorted(self.destination.glob("*.plugin"))
        return {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode), p.stat().st_uid, p.stat().st_gid) for p in names}

    def request_lines(self, target, *, archive=None, source=None, both=True, arch="90", force=False):
        xad = self.destination / "xhdfe.ado"
        fad = self.destination / "xfepout.ado"
        return [target, arch, str(source or ("" if archive else self.source)), str(archive or ""), "", str(xad),
                str(xad.with_suffix(".plugin")), str(fad) if both else "", str(fad.with_suffix(".plugin")) if both else "",
                str(self.smoke), str(self.fake_stata), str(self.tmp), "keep", "force" if force else ""]

    def install(self, target="", *, archive=None, source=None, both=True, arch="90", force=False, **extra):
        args = self.request_lines(target, archive=archive, source=source, both=both, arch=arch, force=force)
        self.request.write_text("\n".join(args) + "\n")
        return subprocess.run(["bash", str(INSTALLER), str(self.request), str(self.receipt)],
                              cwd=self.root, env=self.env | extra, capture_output=True, text=True, timeout=20)

    def trace(self):
        return [line.split("|") for line in self.log.read_text().splitlines()]

    def test_invalid_cpu_target_and_architecture_preserve_destination(self):
        cases = (("invalid", "90", "cputarget() must be portable or native"),
                 ("portable", "74", "no usable CUDA architecture"))
        for target, arch, message in cases:
            with self.subTest(target=target, arch=arch):
                result = self.install(target, arch=arch)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.plugins(), self.original)
                self.assertFalse(self.log.exists())

    def test_legacy_environment_precedence_and_explicit_overrides(self):
        for target, march, force, expected in (
            ("", "", "", "portable"),
            ("", "on", "", "portable"),
            ("", "", "ON", "portable"),
            ("", "yes", "true", "portable"),
            ("", "on", "true", "native"),
            ("", "on", "ON", "native"),
            ("portable", "1", "ON", "portable"),
            ("native", "", "", "native"),
            ("native", "off", "no", "native"),
        ):
            with self.subTest(target=target, march=march, force=force):
                result = self.install(target, XHDFE_ENABLE_MARCH_NATIVE=march,
                                      XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA=force,
                                      XHDFE_ENABLE_CUDA="OFF", XHDFE_CUDA_ARCH="86", XHDFE_CUDA_ARCHS="75,89")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.receipt.read_text().splitlines()[1:3], [expected, "90"])
                effective = "on" if expected == "native" else "off"
                choice = "--march-native" if expected == "native" else "--no-march-native"
                self.assertEqual(self.trace()[-2:], [[name, f"--linux --openmp {choice}", "ON", "90", "", effective, effective]
                                                     for name in OLD])
                self.assertEqual({name: self.plugins()[name][1:] for name in self.original},
                                 {name: fields[1:] for name, fields in self.original.items()})

    def test_second_build_failure_does_not_touch_the_installed_pair(self):
        result = self.install("native", XHDFEGPU_TEST_FAIL_BUILD="xfepout")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build failed", result.stderr)
        self.assertEqual([args[0] for args in self.trace()], ["xhdfe", "xfepout"])
        self.assertEqual(self.plugins(), self.original)
        self.assertFalse(list(self.destination.glob(".xhdfegpu-backup.*")))

    def test_stata_error_despite_zero_exit_status_prevents_install(self):
        result = self.install("native", XHDFEGPU_TEST_FAIL_SMOKE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Stata validation failed", result.stderr)
        self.assertEqual(self.plugins(), self.original)
        self.assertFalse(list(self.destination.glob(".xhdfegpu-backup.*")))

    def test_force_keeps_byte_identical_plugins_in_place(self):
        first = self.install("portable")
        self.assertEqual(first.returncode, 0, first.stderr)
        aliases = [self.destination / f"{name}.plugin" for name in OLD]
        inodes = [alias.stat().st_ino for alias in aliases]
        second = self.install("portable", force=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual([alias.stat().st_ino for alias in aliases], inodes)
        self.assertEqual(self.receipt.read_text().splitlines()[5], "")

    def test_second_replacement_failure_restores_first_plugin(self):
        mv = shutil.which("mv")
        self.assertIsNotNone(mv)
        hook = self.bin / "mv"
        hook.write_text(
            '#!/bin/sh\n'
            'if [ "$#" = 4 ] && [ "$1" = -Tf ] && [ "$2" = -- ] && [ "${3##*/}" = candidate.plugin ] '
            '&& [ "${4##*/}" = xfepout.plugin ]; then\n'
            '    cat "$XHDFEGPU_TEST_FIRST" > "$XHDFEGPU_TEST_OBSERVED"\n'
            '    exit 73\nfi\n'
            'exec "$XHDFEGPU_TEST_MV" "$@"\n'
        )
        hook.chmod(0o700)
        observed = self.root / "first plugin observed during second rename.txt"
        result = self.install("native", XHDFEGPU_TEST_FIRST=str(self.destination / "xhdfe.plugin"),
                              XHDFEGPU_TEST_OBSERVED=str(observed), XHDFEGPU_TEST_MV=mv)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(observed.read_bytes(), b"built xhdfe on\n")
        self.assertIn("original plugin files restored", result.stderr + result.stdout)
        self.assertEqual(self.plugins(), self.original)

    def test_offline_zip_with_spaces_and_standalone_xhdfe(self):
        archive = self.root / "offline sources with spaces.zip"
        with ZipFile(archive, "x") as out:
            for path in sorted(self.source.rglob("*")):
                if path.is_file():
                    out.write(path, Path("xhdfe-src") / path.relative_to(self.source))
        for name in ("xfepout.ado", "xfepout.plugin"):
            (self.destination / name).unlink()
        result = self.install("portable", archive=archive, both=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line[0] for line in self.trace()], ["xhdfe"])
        self.assertEqual((self.destination / "xhdfe.plugin").read_bytes(), b"built xhdfe off\n")
        self.assertFalse((self.destination / "xfepout.plugin").exists())
        self.assertEqual(self.receipt.read_text().splitlines()[4], "")

    def test_explicit_source_wins_over_an_unavailable_zip(self):
        missing = self.root / "unused source archive.zip"
        result = self.install("portable", source=self.source, archive=missing)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([args[0] for args in self.trace()], ["xhdfe", "xfepout"])
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
