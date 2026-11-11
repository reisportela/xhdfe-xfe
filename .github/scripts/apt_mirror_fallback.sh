#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners list the Azure Ubuntu mirror first in
# /etc/apt/apt-mirrors.txt. When that mirror cannot be reached from the runner,
# apt stalls on every index and package: on 11 November 2026 the runners could
# not resolve any ubuntu.com name and four runs waited in apt-get for up to half
# an hour before being cancelled. This probe keeps the runner's list when the
# Azure mirror answers. Otherwise it moves apt to the first archive that does:
# the primary Ubuntu archive, then the kernel.org mirror of it. Packages stay
# signature-verified by apt with the runner's Ubuntu keyring whichever mirror
# serves them. The job fails at once, with the names it could not resolve, when
# no archive answers. Run it after checkout and before the first apt-get.
set -euo pipefail

LIST=/etc/apt/apt-mirrors.txt
if [ ! -f "$LIST" ]; then
    echo "apt mirror list not present on this runner; nothing to do"
    exit 0
fi

CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
probe() {
    curl -fsS --max-time 10 -o /dev/null "$1/dists/$CODENAME/InRelease"
}

azure_ok=1
for attempt in 1 2 3; do
    if ! probe http://azure.archive.ubuntu.com/ubuntu; then
        azure_ok=0
        echo "Azure Ubuntu mirror did not answer (attempt $attempt)"
        break
    fi
done
if [ "$azure_ok" = 1 ]; then
    echo "Azure Ubuntu mirror reachable; apt mirror list unchanged"
    exit 0
fi

for host in azure.archive.ubuntu.com archive.ubuntu.com security.ubuntu.com \
    mirrors.edge.kernel.org launchpad.net pypi.org github.com; do
    echo "$host resolves to: $(getent hosts "$host" | head -n 1 | awk '{print $1}' || true)"
done
for archive in http://archive.ubuntu.com/ubuntu https://mirrors.edge.kernel.org/ubuntu; do
    if probe "$archive"; then
        echo "Switching apt to $archive for this job"
        printf '%s/\n' "$archive" | sudo tee "$LIST" >/dev/null
        exit 0
    fi
    echo "$archive did not answer"
done
echo "ERROR: no Ubuntu archive answers from this runner" >&2
exit 1
