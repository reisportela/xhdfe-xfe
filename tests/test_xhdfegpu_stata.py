"""Stata package layout and status tests with a disposable PLUS and HOME."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATA = shutil.which(os.environ.get("XHDFE_TEST_STATA", "stata-mp"))
CC = shutil.which("cc")
HELPERS = ("xhdfegpu-install.sh", "xhdfegpu-smoke.do")
DEMOS = ("xhdfe_example_cluster.do", "xhdfe_example_partial_out.do")
MARKER = "XHDFEGPU_STATA_TEST_COMPLETE"
ARTIFACTS = os.environ.get("XHDFE_TEST_PLUGIN_PAIR")


@unittest.skipUnless(STATA and sys.platform.startswith("linux"), "requires Linux Stata/MP")
class StataInstallerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="xhdfegpu stata ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        names = ("home", "stata plus", "personal", "site", "tmp", "work")
        self.home, self.plus, self.personal, self.site, self.tmp, self.work = (
            self.root / name for name in names
        )
        self.package_site = self.root / "package source"
        for directory in (self.home, self.plus, self.personal, self.site, self.tmp, self.work):
            directory.mkdir()
        (self.work / "profile.do").write_text("* Isolated regression test.\n")
        self.env = {
            "HOME": str(self.home),
            "STATATMP": str(self.tmp),
            "PATH": os.defpath,
            "LC_ALL": "C",
        }

    def run_stata(self, commands):
        name = "contract"
        driver = self.work / f"{name}.do"
        driver.write_text(
            "set more off\n"
            f'sysdir set PLUS "{self.plus}"\n'
            f'sysdir set PERSONAL "{self.personal}"\n'
            f'sysdir set SITE "{self.site}"\n'
            f'{commands}\ndisplay "{MARKER}"\n'
        )
        process = subprocess.run([STATA, "-q", "-b", "do", driver.name], cwd=self.work,
                                 env=self.env, capture_output=True, text=True, timeout=90)
        log = (self.work / f"{name}.log").read_text(errors="replace")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn(MARKER, log.splitlines(), log)
        self.assertIsNone(re.search(r"^r\(\d+\);", log, re.MULTILINE), log)

    def stage_site(self, aliases):
        subprocess.run(
            ["bash", str(ROOT / "tools/stage_stata_netinstall_site.sh"), str(self.package_site),
             "--linux-xhdfe", str(aliases[0]), "--linux-xfepout", str(aliases[1])],
            cwd=self.work, env=self.env, check=True, capture_output=True, text=True,
            timeout=30,
        )

    def test_net_install_includes_helpers_and_net_get_only_examples(self):
        # The copies stand in for the mapped binaries in this placement test.
        aliases = []
        for command in ("xhdfe", "xfepout"):
            alias = self.root / f"{command}.plugin"
            alias.write_bytes(f"layout fixture {command}\n".encode())
            aliases.append(alias)
        self.stage_site(aliases)
        self.run_stata(
            f'net install xhdfe, from("{self.package_site}")\n'
            f'net get xhdfe, from("{self.package_site}")\n'
        )
        commands = list(self.plus.rglob("xhdfegpu.ado"))
        self.assertEqual(len(commands), 1)
        for helper in HELPERS:
            self.assertEqual((commands[0].parent / helper).read_bytes(),
                             (ROOT / "stata" / helper).read_bytes())
        downloads = {path.name for path in self.work.iterdir()}
        downloaded_examples = {name for name in downloads if name.endswith(".do")}
        self.assertEqual(downloaded_examples - {"contract.do", "profile.do"}, set(DEMOS))
        self.assertFalse(downloads.intersection(HELPERS))
        for demo in DEMOS:
            self.assertEqual((self.work / demo).read_bytes(), (ROOT / "stata" / demo).read_bytes())
        for alias in aliases:
            [installed] = list(self.plus.rglob(alias.name))
            self.assertEqual(installed.read_bytes(), alias.read_bytes())

    def test_compiled_source_snapshot_installs_directly(self):
        source = ROOT / "stata"
        if not all((source / f"{name}.plugin").is_file() for name in ("xhdfe", "xfepout")):
            self.skipTest("the direct snapshot installation requires both built aliases")
        self.run_stata(f'net install xhdfe, from("{source}")\n')
        [ado] = list(self.plus.rglob("xhdfegpu.ado"))
        for name in (*HELPERS, "xhdfe.plugin", "xfepout.plugin"):
            self.assertEqual((ado.parent / name).read_bytes(), (source / name).read_bytes())

    def test_missing_plugin_returns_missing_capabilities(self):
        fixture = self.root / "source with spaces"
        fixture.mkdir()
        shutil.copyfile(ROOT / "stata" / "xhdfegpu.ado", fixture / "xhdfegpu.ado")
        self.run_stata(
            f'adopath ++ "{fixture}"\n'
            'xhdfegpu, status\n'
            'assert "`r(xhdfe_build_source)\'" == "missing"\n'
            'assert "`r(xfepout_build_source)\'" == "missing"\n'
            'assert "`r(xhdfe_cputarget)\'" == "unrecorded"\n'
            'assert "`r(xfepout_sms)\'" == "unrecorded"\n'
            'assert missing(r(xhdfe_cuda)) & missing(r(xfepout_openmp))\n'
        )

    @unittest.skipUnless(CC, "requires a C compiler for the minimal legacy plugin")
    def test_legacy_plugin_status_and_invalid_install_request(self):
        fixture = self.root / "source with spaces"
        fixture.mkdir()
        shutil.copyfile(ROOT / "stata" / "xhdfegpu.ado", fixture / "xhdfegpu.ado")
        (fixture / "xhdfe.ado").write_text("program xhdfe\nend\n")
        legacy = fixture / "legacy.c"
        legacy.write_text(
            '#include "stplugin.h"\n'
            'STDLL stata_call(int argc, char *argv[]) { return 198; }\n'
        )
        deps = ROOT / "stata" / "tools" / "_deps"
        alias = fixture / "xhdfe.plugin"
        subprocess.run(
            [CC, "-shared", "-fPIC", "-DSYSTEM=OPUNIX", "-I", str(deps),
             str(legacy), str(deps / "stplugin.c"), "-o", str(alias)],
            check=True, capture_output=True, text=True, timeout=15, env=self.env,
        )
        prior = alias.read_bytes()
        bound = fixture / "plugin from earlier in the session.plugin"
        shutil.copyfile(alias, bound)
        self.run_stata(
            f'adopath ++ "{fixture}"\n'
            'xhdfegpu, status\n'
            'assert "`r(xhdfe_build_source)\'" == "by_path"\n'
            'assert "`r(xhdfe_cputarget)\'" == "unrecorded"\n'
            'assert "`r(xhdfe_sms)\'" == "unrecorded"\n'
            'assert missing(r(xhdfe_cuda)) & missing(r(xhdfe_openmp))\n'
            'capture xhdfegpu, cputarget(bad)\n'
            'assert _rc == 198\n'
            'capture xhdfegpu, status cputarget(native)\n'
            'assert _rc == 198\n'
            'macro drop XHDFEGPU_PROBE_xhdfe\n'
            'xhdfegpu, status\n'
            'assert "`r(xhdfe_build_source)\'" == "memory_unrecorded"\n'
            'assert "`r(xhdfe_bound_plugin)\'" == ""\n'
            'assert missing(r(xhdfe_cuda)) & missing(r(xhdfe_openmp))\n'
            f'program __xhdfe_plugin_dispatch, plugin using("{bound}")\n'
            f'global XHDFE_PLUGIN_PATH_INTERNAL "{bound}"\n'
            'xhdfegpu, status\n'
            'assert "`r(xhdfe_build_source)\'" == "memory"\n'
            f'assert "`r(xhdfe_plugin)\'" == "{alias}"\n'
            f'assert "`r(xhdfe_bound_plugin)\'" == "{bound}"\n'
            'assert "`r(xhdfe_build_version)\'" == "unrecorded"\n'
        )
        self.assertEqual(alias.read_bytes(), prior)

    @unittest.skipUnless(ARTIFACTS, "set XHDFE_TEST_PLUGIN_PAIR to check release plugin metadata")
    def test_real_build_status_for_both_plugins(self):
        target = os.environ.get("XHDFE_TEST_CPU_TARGET", "portable")
        sms = os.environ.get("XHDFE_TEST_SMS", "none")
        cuda = 0 if sms == "none" else 1
        version = re.search(r"project\(xhdfe VERSION ([\d.]+) ",
                            (ROOT / "CMakeLists.txt").read_text()).group(1)
        self.assertIn(target, ("native", "portable"))
        self.assertTrue(sms == "none" or re.fullmatch(r"[\d,]+", sms))
        aliases = [Path(ARTIFACTS) / f"{name}.plugin" for name in ("xhdfe", "xfepout")]
        self.stage_site(aliases)
        installed = self.plus / "x"
        commands = [f'net install xhdfe, from("{self.package_site}")']
        for query in ("by_path", "memory"):
            commands.append("xhdfegpu, status")
            for name in ("xhdfe", "xfepout"):
                alias = installed / f"{name}.plugin"
                bound = alias if query == "memory" else ""
                commands.extend((
                    f'assert "`r({name}_plugin)\'" == "{alias}"',
                    f'assert "`r({name}_bound_plugin)\'" == "{bound}"',
                    f'assert "`r({name}_build_source)\'" == "{query}"',
                    f'assert "`r({name}_cputarget)\'" == "{target}"',
                    f'assert "`r({name}_sms)\'" == "{sms}"',
                    f'assert r({name}_cuda) == {cuda} & r({name}_openmp) == 1',
                    f'assert "`r({name}_compiler)\'" != "unrecorded"',
                    f'assert "`r({name}_build_version)\'" == "{version}"',
                ))
        self.run_stata("\n".join(commands))
        for alias in aliases:
            self.assertEqual((installed / alias.name).read_bytes(), alias.read_bytes())


    @unittest.skipUnless(ARTIFACTS, "set XHDFE_TEST_PLUGIN_PAIR for actual Stata fits")
    def test_status_preserves_real_estimations_and_uses_their_plugins(self):
        roles = ("xhdfe", "xfepout")
        target = os.environ.get("XHDFE_TEST_CPU_TARGET", "portable")
        sms = os.environ.get("XHDFE_TEST_SMS", "none")
        cuda = int(sms != "none")
        version = re.search(r"project\(xhdfe VERSION ([\d.]+) ",
                            (ROOT / "CMakeLists.txt").read_text()).group(1)
        aliases = [Path(ARTIFACTS) / f"{role}.plugin" for role in roles]
        self.stage_site(aliases)
        installed = self.plus / "x"
        relocated = self.root / "relocated xfepout"
        relocated.mkdir()
        shutil.copyfile(ROOT / "stata/xfepout.ado", relocated / "xfepout.ado")
        shutil.copyfile(aliases[1], relocated / "xfepout.plugin")

        def status(paths, sources):
            checks = ["xhdfegpu, status"]
            for role, path, source in zip(roles, paths, sources, strict=True):
                bound = path if source == "memory" else ""
                for key, value in (("build_source", source), ("plugin", path), ("bound_plugin", bound),
                                   ("cputarget", target), ("sms", sms), ("build_version", version)):
                    checks.append(f'assert "`r({role}_{key})\'" == "{value}"')
                checks.append(f"assert r({role}_cuda) == {cuda} & r({role}_openmp) == 1")
            return "\n".join(checks)

        first = (installed / "xhdfe.plugin", installed / "xfepout.plugin")
        second = (first[0], relocated / "xfepout.plugin")
        self.run_stata(
            f'net install xhdfe, from("{self.package_site}")\n'
            'sysuse auto, clear\n'
            'generate double _xhdfegpu_buildinfo = weight\n'
            'quietly xhdfe price _xhdfegpu_buildinfo, absorb(rep78) gpubackend(cpu)\n'
            'local coefficient = _b[_xhdfegpu_buildinfo]\n'
            'local n = e(N)\n'
            'generate byte original_sample = e(sample)\n'
            'quietly datasignature\n'
            'local signature "`r(datasignature)\'"\n'
            + status(first, ("memory", "by_path")) + "\n"
            'assert "`e(cmd)\'" == "xhdfe" & e(N) == `n\' & _b[_xhdfegpu_buildinfo] == `coefficient\'\n'
            'assert e(sample) == original_sample\n'
            'quietly datasignature\n'
            'assert "`r(datasignature)\'" == "`signature\'"\n'
            f'adopath ++ "{relocated}"\n'
            'quietly xfepout price weight, absorb(rep78) generate(po_) gpubackend(cpu)\n'
            'local n = e(N)\n'
            'replace original_sample = e(sample)\n'
            'quietly datasignature\n'
            'local signature "`r(datasignature)\'"\n'
            + status(second, ("memory", "memory")) + "\n"
            'assert "`e(cmd)\'" == "xfepout" & e(N) == `n\'\n'
            'assert e(sample) == original_sample\n'
            'quietly datasignature\n'
            'assert "`r(datasignature)\'" == "`signature\'"\n'
            'discard\n'
            + status(second, ("by_path", "by_path")) + "\n"
            'quietly datasignature\n'
            'assert "`r(datasignature)\'" == "`signature\'"\n'
        )


if __name__ == "__main__":
    unittest.main()
