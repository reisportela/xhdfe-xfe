"""Falsification controls for the CUDA disassembly gate, without CUDA hardware."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PTX = ".visible .entry _ZN9hdfe_cert6reduceEv() {\nadd.rn.f64 %fd1, %fd2, %fd3;\n}\n"
SASS = "Function : _ZN9hdfe_cert6reduceEv\n/*0000*/ DADD R0, R2, R4;\n"


@unittest.skipUnless(sys.platform.startswith("linux"), "CUDA build gate targets Linux")
class DeviceFmaGateTests(unittest.TestCase):
    def test_missing_tmpdir_does_not_skip_disassembly_checks(self):
        cases = {
            "clean": (PTX, SASS, True),
            "predicated_ptx": (PTX.replace("add.rn", "@%p1 fma.rn"), SASS, False),
            "dfma_sass": (PTX, SASS.replace("DADD", "@P0 DFMA"), False),
            "ffma_sass": (PTX, SASS.replace("DADD", "FFMA"), False),
            "hfma_sass": (PTX, SASS.replace("DADD", "HFMA2"), False),
            "constant_materialization": (PTX, SASS.replace("DADD R0, R2, R4", "HFMA2.MMA R0, -RZ, RZ, 1, 2"), True),
            "division_expansion": (PTX, SASS.replace("reduce", "bracket_decide").replace("DADD", "DFMA"), True),
            "no_ptx": ("", SASS, False),
            "no_sass": (PTX, "", False),
            "wrong_function": (PTX.replace("hdfe_cert", "absorber"), SASS, False),
        }
        with tempfile.TemporaryDirectory(prefix="xhdfe-fma-gate-") as temp:
            root = Path(temp)
            dumper = root / "cuobjdump"
            payload = (f"#!{sys.executable}\nimport os,sys\n"
                       "print(os.environ['FMA_TEST_PTX' if sys.argv[1]=='--dump-ptx' else 'FMA_TEST_SASS'])\n")
            fd = os.open(dumper, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
            obj = root / "verifier.o"
            obj.write_bytes(b"disassembly fixture")
            env = dict(os.environ, CUOBJDUMP=str(dumper))
            env.pop("TMPDIR", None)
            for script in ("tools/check_verifier_device_fma.sh", "r/xhdfe/src/check_verifier_device_fma.sh"):
                for name, (ptx, sass, valid) in cases.items():
                    with self.subTest(script=script, case=name):
                        result = subprocess.run(["bash", str(ROOT / script), str(obj)],
                            env=dict(env, FMA_TEST_PTX=ptx, FMA_TEST_SASS=sass),
                            text=True, capture_output=True, timeout=10)
                        self.assertEqual(result.returncode == 0, valid, result.stdout + result.stderr)
                        if valid:
                            self.assertIn("OK: verifier device code clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
