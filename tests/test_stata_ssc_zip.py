"""Verify the complete SSC install site and both plug-in mappings."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from tools.build_stata_ssc_zip import build, validate_plugin_directives


VERSION = "2.34.2.20261122"
FILES = (
    "xhdfe.ado", "xfepout.ado", "xhdfe.sthlp", "xfepout.sthlp", "LICENSE", "NOTICE",
    "xhdfe_example_cluster.do", "xhdfe_example_partial_out.do",
)
HELPERS = ("xhdfegpu-install.sh", "xhdfegpu-smoke.do")
BINARIES = (
    "xhdfe.linux64.plugin", "xhdfe.macos-universal.plugin", "xhdfe.win64.plugin",
    "xfepout.linux64.plugin", "xfepout.macos-universal.plugin", "xfepout.win64.plugin",
)
BASE = ("v 2.34.2\nd SSC test package\n"
        + "".join(f"f {name}\n" for name in FILES)
        + "".join(f"F {name}\n" for name in HELPERS))
MANIFEST = BASE + """G MACARM64 xhdfe_libomp.dylib xhdfe_libomp.dylib
G OSX.ARM64 xhdfe_libomp.dylib xhdfe_libomp.dylib
G MACINTEL64 xhdfe_libomp.dylib xhdfe_libomp.dylib
G OSX.X8664 xhdfe_libomp.dylib xhdfe_libomp.dylib
g LINUX64 xhdfe.linux64.plugin xhdfe.plugin
g LINUX64P xhdfe.linux64.plugin xhdfe.plugin
g MACARM64 xhdfe.macos-universal.plugin xhdfe.plugin
g OSX.ARM64 xhdfe.macos-universal.plugin xhdfe.plugin
g MACINTEL64 xhdfe.macos-universal.plugin xhdfe.plugin
g OSX.X8664 xhdfe.macos-universal.plugin xhdfe.plugin
g WIN64 xhdfe.win64.plugin xhdfe.plugin
g LINUX64 xfepout.linux64.plugin xfepout.plugin
g LINUX64P xfepout.linux64.plugin xfepout.plugin
g MACARM64 xfepout.macos-universal.plugin xfepout.plugin
g OSX.ARM64 xfepout.macos-universal.plugin xfepout.plugin
g MACINTEL64 xfepout.macos-universal.plugin xfepout.plugin
g OSX.X8664 xfepout.macos-universal.plugin xfepout.plugin
g WIN64 xfepout.win64.plugin xfepout.plugin
h xhdfe.plugin
h xfepout.plugin
"""
CORE = (*FILES, *HELPERS, *BINARIES, "xhdfe_libomp.dylib")
AUXILIARY = ("libgomp-1.dll", "windows-stata-runtime-ledger.json", "other.plugin",
             "SampleNote.txt", "samplenote.txt")
TOC = "v 3\np xhdfe Main package\np xfepout Companion package\n"
XFEP_OUT_AUX = "xfepout-only-notice.txt"
XFEP_MANIFEST = (
    f"v 1.14.5\nd SSC companion test package\nf xfepout.ado\nf xfepout.sthlp\nf {XFEP_OUT_AUX}\n"
    + "\n".join(line for line in MANIFEST.splitlines()
                if line.startswith("G ") or line.endswith(" xfepout.plugin"))
    + "\n"
)


class StataSSCZipTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.site = root / "site.zip"
        self.output = root / f"xhdfe-ssc-{VERSION}.zip"

    def stage(self, manifest, xfepout=XFEP_MANIFEST, toc=TOC, omit=(), xhdfe_version="2.34.2"):
        members = {name: name.encode() for name in (*CORE, *AUXILIARY, XFEP_OUT_AUX)}
        members.update({
            "xhdfe.pkg": manifest.encode(),
            "xfepout.pkg": xfepout.encode(),
            "stata.toc": toc.encode(),
            "xhdfe.ado": f"*! version {xhdfe_version} 22nov2026\n".encode(),
            "xfepout.ado": b"*! version 1.14.5 22nov2026\n",
        })
        with ZipFile(self.site, "w") as site:
            for name, data in members.items():
                if name not in omit:
                    site.writestr(f"stata/{name}", data)

    def reject(self, manifest, error, **site_options):
        self.stage(manifest, **site_options)
        with self.assertRaisesRegex(ValueError, error):
            build(self.site, VERSION, self.output)
        self.assertFalse(self.output.exists())

    def test_flat_archive_preserves_release_bytes_and_windows_runtime(self):
        self.stage(MANIFEST + "G WIN64 libgomp-1.dll libgomp-1.dll\n"
                   "g WIN64 windows-stata-runtime-ledger.json\n"
                   "g WIN64 windows-stata-runtime-ledger.json\n")
        build(self.site, VERSION, self.output)
        expected = sorted((*CORE, "libgomp-1.dll", "windows-stata-runtime-ledger.json",
                           XFEP_OUT_AUX, "xhdfe.pkg", "xfepout.pkg", "stata.toc"))
        with ZipFile(self.output) as archive, ZipFile(self.site) as site:
            self.assertEqual(archive.namelist(), expected)
            for name in expected:
                self.assertEqual(archive.read(name), site.read(f"stata/{name}"))

    def test_rejects_suffixed_files_without_installed_aliases(self):
        broken = BASE + "\n".join(f"f {name}" for name in BINARIES) + "\nF xhdfe_libomp.dylib\n"
        self.reject(broken, "Invalid SSC plug-in directive: f xhdfe.linux64.plugin")

    def test_rejects_incomplete_platform_or_required_file_declaration(self):
        for line in (
            "h xfepout.plugin\n",
            "g OSX.ARM64 xfepout.macos-universal.plugin xfepout.plugin\n",
            "G OSX.X8664 xhdfe_libomp.dylib xhdfe_libomp.dylib\n",
        ):
            with self.subTest(line=line.rstrip()):
                self.reject(MANIFEST.replace(line, ""), "Missing SSC plug-in directives")

    def test_rejects_swapped_plugins_and_wrong_runtime_directive(self):
        replacements = (
            ("g WIN64 xhdfe.win64.plugin xhdfe.plugin", "g WIN64 xfepout.win64.plugin xhdfe.plugin"),
            ("G MACARM64 xhdfe_libomp.dylib xhdfe_libomp.dylib", "g MACARM64 xhdfe_libomp.dylib xhdfe_libomp.dylib"),
        )
        for valid, invalid in replacements:
            with self.subTest(line=invalid):
                self.reject(MANIFEST.replace(valid, invalid), "Invalid SSC plug-in directive")

    def test_rejects_duplicate_conflicting_and_plain_directives(self):
        additions = (
            ("g LINUX64 xhdfe.linux64.plugin xhdfe.plugin", "Duplicate SSC plug-in directive"),
            ("g WIN64 other.plugin XHDFE.PLUGIN", "Invalid SSC plug-in directive"),
            ("g WIN64 other.plugin ./xhdfe.plugin", "Invalid SSC plug-in directive"),
            ("h xhdfe.plugin", "Duplicate SSC plug-in directive"),
            ("f xfepout.win64.plugin", "Invalid SSC plug-in directive"),
            ("F xhdfe_libomp.dylib", "Invalid SSC plug-in directive"),
        )
        for line, error in additions:
            with self.subTest(line=line):
                self.reject(MANIFEST + line + "\n", error)

    def test_requires_uppercase_identity_mapping_for_windows_runtime(self):
        for line in (
            "g WIN64 libgomp-1.dll libgomp-1.dll",
            "G WIN64 libgomp-1.dll other.dll",
            "G LINUX64 libgomp-1.dll libgomp-1.dll",
            "F libgomp-1.dll",
        ):
            with self.subTest(line=line):
                self.reject(MANIFEST + line + "\n", "SSC runtime DLL requires an identity G WIN64 mapping")

    def test_editorial_descriptor_can_validate_without_ci_version_line(self):
        editorial = MANIFEST.split("\n", 1)[1]
        validate_plugin_directives(editorial)

    def test_rejects_missing_descriptor_catalogue_or_declared_member(self):
        for member in ("xhdfe.pkg", "xfepout.pkg", "stata.toc", XFEP_OUT_AUX):
            with self.subTest(member=member):
                self.reject(MANIFEST, f"Missing SSC site member: {member}", omit=(member,))

    def test_rejects_missing_help_or_standalone_example(self):
        for member in ("xhdfe.sthlp", "xhdfe_example_cluster.do", "xhdfe_example_partial_out.do"):
            with self.subTest(member=member):
                self.reject(MANIFEST.replace(f"f {member}\n", ""),
                            "Missing install or example entry in xhdfe.pkg")

    def test_rejects_wrong_catalogue_format(self):
        self.reject(MANIFEST, "SSC catalogue must have one v 3 directive",
                    toc=TOC.replace("v 3\n", "v 2\n"))

    def test_rejects_catalogue_that_does_not_match_descriptors(self):
        for toc in (TOC.replace("p xfepout Companion package\n", ""),
                    TOC.replace("p xfepout", "p xhdfe"), TOC + "p other Another package\n"):
            with self.subTest(toc=toc):
                self.reject(MANIFEST, "SSC catalogue must list xhdfe and xfepout exactly once", toc=toc)

    def test_rejects_incomplete_or_conflicting_xfepout_manifest(self):
        cases = (
            (XFEP_MANIFEST.replace("h xfepout.plugin\n", ""), "Missing SSC plug-in directives"),
            (XFEP_MANIFEST.replace("f xfepout.ado\n", ""), "Missing install or example entry in xfepout.pkg"),
            (XFEP_MANIFEST + "g WIN64 other.plugin xfepout.plugin\n", "Invalid SSC plug-in directive"),
        )
        for manifest, error in cases:
            with self.subTest(error=error):
                self.reject(MANIFEST, error, xfepout=manifest)

    def test_xfepout_descriptor_version_must_match_its_own_ado(self):
        self.reject(MANIFEST, "Expected package version 1.14.5",
                    xfepout=XFEP_MANIFEST.replace("v 1.14.5", "v 2.34.2"))

    def test_xhdfe_ado_must_match_requested_release_version(self):
        self.reject(MANIFEST, "Expected version 2.34.2 in xhdfe.ado, found 2.34.1",
                    xhdfe_version="2.34.1")

    def test_rejects_case_collision_between_package_descriptors(self):
        self.reject(MANIFEST + "f SampleNote.txt\n", "Case-colliding SSC archive member names",
                    xfepout=XFEP_MANIFEST + "f samplenote.txt\n")
