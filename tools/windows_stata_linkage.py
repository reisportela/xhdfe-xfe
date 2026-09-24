"""Record static MinGW link inputs and authenticate the final Stata plugin pair."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


PROVIDERS = {"gcc-mingw-w64": "ubuntu-mingw-gcc", "mingw-w64": "ubuntu-mingw-w64"}
PLUGIN_NAMES = {"xhdfe.plugin.windows", "xfepout.plugin.windows"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def validate(ledger, artifact_dir=None, plugin_paths=None):
    if (ledger.get("schema_version") != 1 or ledger.get("linkage") != "static"
            or ledger.get("artifact") != "xhdfe-xfe-stata-windows-cpu"
            or ledger.get("entries") != []
            or ledger.get("compiler", {}).get("target") != "x86_64-w64-mingw32"):
        raise ValueError("invalid static Windows Stata ledger")
    archives = ledger.get("static_archives", [])
    names = {entry["name"] for entry in archives}
    if not {"libgcc.a", "libstdc++.a", "libgomp.a"} <= names or not names & {"libpthread.a", "libwinpthread.a"}:
        raise ValueError("static GCC/OpenMP/winpthreads archive evidence is incomplete")
    if len({entry["source_path"] for entry in archives}) != len(archives):
        raise ValueError("duplicate static archive")
    for entry in archives:
        if (entry.get("provider_id") != PROVIDERS.get(entry.get("source_package"))
                or entry.get("provider_id") is None):
            raise ValueError("unmapped static archive provider")
        for field in ("source_path", "runtime_package", "runtime_version", "source_version"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"missing static archive {field}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) or entry.get("size", 0) <= 0:
            raise ValueError("invalid static archive hash/size")
        if not set(entry.get("used_by", [])) <= PLUGIN_NAMES or not entry["used_by"]:
            raise ValueError("static archive is not linked to the plugin pair")
    plugins = ledger.get("plugins", [])
    if len(plugins) != 2 or {entry["name"] for entry in plugins} != PLUGIN_NAMES:
        raise ValueError("static plugin pair missing")
    for entry in plugins:
        used = {item["name"] for item in archives if entry["name"] in item["used_by"]}
        if not {"libgcc.a", "libstdc++.a", "libgomp.a"} <= used or not used & {"libpthread.a", "libwinpthread.a"}:
            raise ValueError(f"{entry['name']}: static runtime closure incomplete")
        for field in ("sha256", "link_map_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry.get(field, "")):
                raise ValueError(f"invalid plugin {field}")
        if entry.get("size", 0) <= 0:
            raise ValueError("invalid plugin size")
        if artifact_dir is not None or plugin_paths is not None:
            path = plugin_paths[entry["name"]] if plugin_paths is not None else artifact_dir / entry["name"]
            if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
                raise ValueError(f"plugin differs from static link evidence: {path}")


def validate_closure(ledger, closure):
    from validate_python_release_artifacts import _is_windows_host_dll
    validate(ledger)
    if closure.get("format") != "xhdfe-windows-runtime-closure-v1" or closure.get("runtimes") != []:
        raise ValueError("static Stata plugins have a nonempty runtime DLL closure")
    roots = closure.get("roots", [])
    if len(roots) != 2 or {entry["member"] for entry in roots} != PLUGIN_NAMES:
        raise ValueError("static Stata PE root pair missing")
    plugins = {entry["name"]: entry for entry in ledger["plugins"]}
    for entry in roots:
        plugin = plugins[entry["member"]]
        if entry.get("member_sha256") != plugin["sha256"] or entry.get("size") != plugin["size"]:
            raise ValueError("static link and PE closure hashes disagree")
        if not entry.get("dependencies") or any(
                not _is_windows_host_dll(name, frozenset()) for name in entry["dependencies"]):
            raise ValueError("static Stata PE root has a non-system dependency")


def record(artifact_dir, compiler):
    archives = {}
    plugins = []
    for name in sorted(PLUGIN_NAMES):
        plugin = artifact_dir / name
        link_map = artifact_dir / (name + ".map")
        # GNU ld's LOAD records identify the archives actually selected by this link.
        paths = re.findall(r"^LOAD (.+\.a)\s*$", link_map.read_text(), re.MULTILINE)
        if not paths:
            raise ValueError(f"no archive LOAD records in {link_map}")
        for text in paths:
            source = Path(text).resolve(strict=True)
            key = str(source)
            if key not in archives:
                package = output("dpkg-query", "-S", key).split(":", 1)[0]
                fields = output("dpkg-query", "-W",
                    "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\t${Built-Using}\tEND",
                    package).split("\t")
                if len(fields) != 6 or fields[-1] != "END" or fields[2] not in PROVIDERS:
                    raise ValueError(f"unreviewed archive provider: {source}: {fields}")
                archives[key] = dict(name=source.name, source_path=key,
                    sha256=digest(source), size=source.stat().st_size,
                    runtime_package=fields[0], runtime_version=fields[1],
                    source_package=fields[2], source_version=fields[3], built_using=fields[4],
                    provider_id=PROVIDERS[fields[2]], used_by=[])
            if name not in archives[key]["used_by"]:
                archives[key]["used_by"].append(name)
        plugins.append(dict(name=name, sha256=digest(plugin), size=plugin.stat().st_size,
                            link_map_sha256=digest(link_map)))
    ledger = dict(schema_version=1, artifact="xhdfe-xfe-stata-windows-cpu", linkage="static",
        compiler=dict(target=output(compiler, "-dumpmachine"),
                      version=output(compiler, "-dumpfullversion", "-dumpversion")),
        entries=[], static_archives=[archives[key] for key in sorted(archives)], plugins=plugins)
    validate(ledger, artifact_dir)
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["record", "verify"])
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--compiler", default="x86_64-w64-mingw32-g++")
    args = parser.parse_args()
    if args.mode == "record":
        ledger = record(args.artifact_dir, args.compiler)
        args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    else:
        validate(json.loads(args.ledger.read_text()), args.artifact_dir)
    print("WINDOWS_STATA_STATIC_LINKAGE_PASS")


if __name__ == "__main__":
    main()
