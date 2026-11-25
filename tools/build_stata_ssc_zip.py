#!/usr/bin/env python3
"""Build the flat, installable SSC submission from the staged Stata site ZIP."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import stat
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PLATFORMS = {
    "LINUX64": "linux64", "LINUX64P": "linux64",
    "MACARM64": "macos-universal", "OSX.ARM64": "macos-universal",
    "MACINTEL64": "macos-universal", "OSX.X8664": "macos-universal",
    "WIN64": "win64",
}
PLUGINS = ("xhdfe", "xfepout")
MAC_RUNTIME = "xhdfe_libomp.dylib"
EXAMPLES = {"xhdfe_example_cluster.do", "xhdfe_example_partial_out.do"}
PROTECTED = {MAC_RUNTIME, *(f"{plugin}.plugin" for plugin in PLUGINS),
             *(f"{plugin}.{build}.plugin" for plugin in PLUGINS for build in PLATFORMS.values())}


def validate_plugin_directives(manifest, plugins=PLUGINS):
    required = {
        ("g", platform, f"{plugin}.{build}.plugin", f"{plugin}.plugin")
        for platform, build in PLATFORMS.items() for plugin in plugins
    }
    required.update(("G", platform, MAC_RUNTIME, MAC_RUNTIME)
                    for platform, build in PLATFORMS.items() if build == "macos-universal")
    required.update(("h", f"{plugin}.plugin") for plugin in plugins)

    seen = set()
    for line in manifest.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"f", "F", "g", "G", "h"}:
            continue
        flag = fields[0]
        if flag in {"g", "G"}:
            if len(fields) not in {3, 4}:
                raise ValueError(f"Invalid package directive: {line}")
            platform, source = fields[1:3]
            destination = fields[3] if len(fields) == 4 else source
            installed = destination.replace("\\", "/").casefold()
            row = (flag, platform, source, destination)
            names = {source.casefold(), installed.rsplit("/", 1)[-1]}
        else:
            if len(fields) != 2:
                raise ValueError(f"Invalid package directive: {line}")
            row = tuple(fields)
            names = {fields[1].casefold()}
        if not names.isdisjoint(PROTECTED):
            if row not in required:
                raise ValueError(f"Invalid SSC plug-in directive: {line}")
        elif any(name.endswith(".dll") for name in names):
            if len(row) != 4 or row[:2] != ("G", "WIN64") or row[2] != row[3]:
                raise ValueError(f"SSC runtime DLL requires an identity G WIN64 mapping: {line}")
        else:
            continue
        if row in seen:
            raise ValueError(f"Duplicate SSC plug-in directive: {line}")
        seen.add(row)

    missing = required - seen
    if missing:
        raise ValueError("Missing SSC plug-in directives: " + "; ".join(" ".join(row) for row in sorted(missing)))


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


def catalogue_packages(toc):
    versions = [line.split() for line in toc.splitlines() if line.split()[:1] == ["v"]]
    if versions != [["v", "3"]]:
        raise ValueError("SSC catalogue must have one v 3 directive")
    packages = []
    for line in toc.splitlines():
        fields = line.split(maxsplit=2)
        if fields and fields[0] == "p":
            if len(fields) != 3:
                raise ValueError(f"Invalid SSC catalogue entry: {line}")
            packages.append(fields[1])
    if sorted(packages) != sorted(PLUGINS):
        raise ValueError("SSC catalogue must list xhdfe and xfepout exactly once")
    return packages


def site_entry(site, name):
    try:
        entry = site.getinfo(f"stata/{name}")
    except KeyError as error:
        raise ValueError(f"Missing SSC site member: {name}") from error
    mode = stat.S_IFMT(entry.external_attr >> 16)
    if entry.is_dir() or (entry.create_system == 3 and mode not in {0, stat.S_IFREG}):
        raise ValueError(f"SSC component is not a regular file: {name}")
    return entry


def ado_version(site, package):
    name = f"{package}.ado"
    data = site.read(site_entry(site, name))
    version = re.match(rb"\*!\s+version\s+(\d+\.\d+\.\d+)\s", data)
    if version is None:
        raise ValueError(f"Missing version banner in {name}")
    return version[1].decode("ascii")


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
        packages = catalogue_packages(site.read(site_entry(site, "stata.toc")).decode("utf-8"))
        versions = {package: ado_version(site, package) for package in packages}
        if versions["xhdfe"] != match[1]:
            raise ValueError(f"Expected version {match[1]} in xhdfe.ado, found {versions['xhdfe']}")
        names = {"stata.toc"}
        for package in packages:
            descriptor = f"{package}.pkg"
            manifest = site.read(site_entry(site, descriptor)).decode("utf-8")
            plugins = PLUGINS if package == "xhdfe" else (package,)
            components = package_files(manifest, versions[package])
            required = {f"{plugin}.{suffix}" for plugin in plugins for suffix in ("ado", "sthlp")}
            if package == "xhdfe":
                required.update(EXAMPLES)
            missing = required.difference(components)
            if missing:
                raise ValueError(f"Missing install or example entry in {descriptor}: {sorted(missing)}")
            validate_plugin_directives(manifest, plugins)
            names.add(descriptor)
            names.update(components)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Case-colliding SSC archive member names across package descriptors")
        names = sorted(names)
        entries = [site_entry(site, name) for name in names]
        with ZipFile(output, "x") as archive:
            for name, entry in zip(names, entries):
                member = ZipInfo(name, timestamp)
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                member.compress_type = ZIP_DEFLATED
                archive.writestr(member, site.read(entry), compresslevel=6)
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
