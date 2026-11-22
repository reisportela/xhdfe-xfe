#!/usr/bin/env python3
"""Build the flat SSC submission from the staged Stata net-install ZIP."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import stat
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def package_files(manifest, version):
    names = set()
    versions = []
    for line in manifest.splitlines():
        fields = line.split()
        if not fields or fields[0] in {"*", "d", "h"}:
            continue
        flag = fields[0]
        if flag == "v" and len(fields) == 2:
            versions.append(fields[1])
            continue
        if flag in {"f", "F"} and len(fields) == 2:
            name = fields[1]
        elif flag in {"g", "G"} and len(fields) in {3, 4}:
            name = fields[2]
        else:
            raise ValueError(f"Invalid package directive: {line}")
        if (not re.fullmatch(r"[A-Za-z0-9._+-]+", name) or name in {".", ".."}
                or Path(name).suffix.lower() in {".pkg", ".toc"}):
            raise ValueError(f"Invalid flat SSC component: {name}")
        names.add(name)
    if versions != [version]:
        raise ValueError(f"Expected package version {version}, found {versions}")
    if not names or len({name.casefold() for name in names}) != len(names):
        raise ValueError("Empty or case-colliding SSC component list")
    return sorted(names)


def build(site_zip, version, output):
    match = re.fullmatch(r"(\d+\.\d+\.\d+)\.(\d{8})", version)
    if match is None:
        raise ValueError("Expected the dated xhdfe version")
    day = datetime.strptime(match[2], "%Y%m%d").date()
    timestamp = (day.year, day.month, day.day, 12, 0, 0)
    with ZipFile(site_zip) as site:
        entries = site.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError("Duplicate members in the net-install ZIP")
        names = package_files(site.read("stata/xhdfe.pkg").decode("utf-8"), match[1])
        for name in names:
            entry = site.getinfo(f"stata/{name}")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or (entry.create_system == 3 and mode not in {0, stat.S_IFREG}):
                raise ValueError(f"SSC component is not a regular file: {name}")
        with ZipFile(output, "x") as archive:
            for name in names:
                member = ZipInfo(name, timestamp)
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                member.compress_type = ZIP_DEFLATED
                archive.writestr(member, site.read(f"stata/{name}"), compresslevel=6)
        with ZipFile(output) as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise ValueError("SSC ZIP inventory or CRC mismatch")
            for name in names:
                if archive.read(name) != site.read(f"stata/{name}"):
                    raise ValueError(f"SSC ZIP component differs from the release site: {name}")
    print(f"Built {output.name} with {len(names)} byte-verified components")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_zip", type=Path)
    parser.add_argument("version")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.site_zip, args.version, args.output)


if __name__ == "__main__":
    main()
