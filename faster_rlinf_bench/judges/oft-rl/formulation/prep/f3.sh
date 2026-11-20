#!/bin/bash
# Baseline retry (F2 baseline: INVALID, the leased GPU did not become idle within 10 min).
set -uo pipefail
O=$1
mkdir -p $O
echo "=== baseline $(date +%T)"; python bench/bench.py baseline --out $O/baseline --repeats 2; echo "=== rc $? $(date +%T)"
echo done > $O/DONE
