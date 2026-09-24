"""Compare the macOS-compatible cache writer with the filesystem implementation.

This POSIX control does not replace the native macOS deployment-target build.
Both executables contain the actual production cache-writer class.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <limits.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
"""
MAIN = r"""
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    AtomicAbsorptionCacheFile file(argv[1]);
    const bool opened = static_cast<bool>(file);
    file.write(argv[2], std::strlen(argv[2]));
    const bool committed = std::string(argv[3]) == "commit" && file.commit();
    std::printf("%d %d\n", opened, committed);
}
"""


def run():
    source = (ROOT / "src/hdfe_regressor_v11.cpp").read_text()
    body = source.split("class AtomicAbsorptionCacheFile {", 1)[1].split(
        "\ntemplate <typename T, typename Output>", 1)[0]
    body = "class AtomicAbsorptionCacheFile {" + body
    with tempfile.TemporaryDirectory(prefix="xhdfe-cache-posix-") as temp:
        root = Path(temp)
        binaries = []
        for variant in ("filesystem", "macos"):
            cpp = root / (variant + ".cpp")
            cpp.write_text(PREFIX + "\n#undef __APPLE__\n" +
                           ("#define __APPLE__ 1\n" if variant == "macos" else "") + body + MAIN)
            binary = root / variant
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            binaries.append(binary)
        cases = ("new", "existing", "alias-file", "alias-parent/item", "missing/item",
                 "existing-directory", "sub/../existing", "accent-á", "interrupted")
        observations = []
        for case in cases:
            results = []
            for binary in binaries:
                directory = root / (binary.name + "-" + str(len(observations)))
                directory.mkdir()
                (directory / "sub").mkdir()
                (directory / "existing-directory").mkdir()
                (directory / "existing").write_text("old")
                (directory / "alias-file").symlink_to("existing")
                (directory / "alias-parent").symlink_to("sub", target_is_directory=True)
                mode = "abort" if case == "interrupted" else "commit"
                result = subprocess.run([str(binary), case, "new payload", mode],
                                        cwd=directory, capture_output=True, text=True, check=True)
                assert not list(directory.rglob(".xhdfe-cache-*")), "temporary cache leaked"
                files = {str(p.relative_to(directory)): p.read_bytes().decode()
                         for p in directory.rglob("*") if p.is_file() and not p.is_symlink()}
                results.append((result.stdout, files, (directory / "alias-file").is_symlink()))
            assert results[0] == results[1], (case, results)
            if case == "interrupted":
                assert results[1][1]["existing"] == "old" and "interrupted" not in results[1][1]
            observations.append(case)
        print(json.dumps(dict(status="PASS", cases=observations,
                              classification="missing platform build coverage",
                              native_macos="separate CI gate")))


if __name__ == "__main__":
    run()
