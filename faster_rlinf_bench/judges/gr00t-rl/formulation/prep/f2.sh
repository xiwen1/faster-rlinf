#!/bin/bash
# F1 re-measure (F1 baseline on host-e: NUMA node ~39% idle, window spreads 12-24%, batch-8 window P50s 150-625 ms; the
# request is ~90% host time, so host contention moves it): profile + interleaved ref-vs-ref baseline with the window
# lengths set by hand (batch 8: 8 x 25, batch 1: 5 x 30; ~4-5 s windows at the quiet ~150-200 ms / ~100-150 ms).
set -uo pipefail
O=$1
mkdir -p $O
step() { echo "=== $1 $(date +%T)"; shift; "$@"; echo "=== rc $? $(date +%T)"; }
echo "host $(hostname)"
step profile python bench/bench.py profile --tree targets/reference --out $O/profile --iters 5
step baseline python bench/bench.py baseline --out $O/baseline --repeats 2
echo done > $O/DONE
