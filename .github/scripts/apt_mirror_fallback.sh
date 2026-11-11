#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners list the Azure Ubuntu mirror first in
# /etc/apt/apt-mirrors.txt. When that mirror cannot be reached from the runner,
# apt stalls on every index and package: on 11 November 2026 four runs waited in
# apt-get for up to half an hour before being cancelled. This probe moves apt to
# the primary Ubuntu archive in that case and leaves the runner untouched when
# the mirror answers. Packages stay signature-verified by apt whichever mirror
# serves them. Run it after checkout and before the first apt-get of a job.
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

echo "Switching apt to the primary Ubuntu archive for this job"
for host in azure.archive.ubuntu.com archive.ubuntu.com security.ubuntu.com; do
    echo "$host resolves to: $(getent hosts "$host" | head -n 1 | awk '{print $1}' || true)"
done
printf 'http://archive.ubuntu.com/ubuntu/\tpriority:1\nhttp://security.ubuntu.com/ubuntu/\tpriority:2\n' \
    | sudo tee "$LIST" >/dev/null
if ! probe http://archive.ubuntu.com/ubuntu; then
    echo "ERROR: neither the Azure mirror nor archive.ubuntu.com answers from this runner" >&2
    exit 1
fi
echo "archive.ubuntu.com reachable"
