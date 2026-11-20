#!/usr/bin/env bash
# Puts ResearchBoost-Agent's numerics/ package (commit f27b8ff) into each judge directory.
# The package is not copied into this repository because ResearchBoost-Agent does not state a license. Every fetched
# file is checked against numerics.sha256, so each judge runs with exactly the package it was frozen with.
#   ./fetch_numerics.sh                    # clones https://github.com/DAGroup-PKU/ResearchBoost-Agent into a temp dir
#   RBA_DIR=<clone> ./fetch_numerics.sh    # uses an existing clone that contains the commit
set -euo pipefail
COMMIT=f27b8ff623238bc5dbcef4ea4015698b5fbb8c69
HERE=$(cd "$(dirname "$0")" && pwd)
if command -v sha256sum > /dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
SRC=${RBA_DIR:-}
if [ -z "$SRC" ]; then
  SRC=$(mktemp -d)
  trap 'rm -rf "$SRC"' EXIT
  git clone --quiet https://github.com/DAGroup-PKU/ResearchBoost-Agent "$SRC"
fi
for j in pi05-rl pi05-rln oft-rl gr00t-rl; do
  rm -rf "$HERE/$j/numerics"
  git -C "$SRC" archive "$COMMIT" numerics | tar -x -C "$HERE/$j"
  (cd "$HERE/$j" && $SHA -c "$HERE/numerics.sha256" > /dev/null) || { echo "$j/numerics: checksum mismatch" >&2; exit 1; }
  echo "$j/numerics: ok ($(wc -l < "$HERE/numerics.sha256") files, ResearchBoost-Agent ${COMMIT:0:7})"
done
