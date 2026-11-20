#!/bin/bash
# F1 re-measure (the F1 baseline ran on a disturbed host: window spreads 6-54%): profile + interleaved ref-vs-ref
# baseline with the window lengths set from the quiet latencies (batch 8 ~115 ms -> 25 iters, batch 1 ~58 ms -> 40).
set -uo pipefail
O=$1
mkdir -p $O
step() { echo "=== $1 $(date +%T)"; shift; "$@"; echo "=== rc $? $(date +%T)"; }
step profile python bench/bench.py profile --tree targets/reference --out $O/profile --iters 5
step baseline python bench/bench.py baseline --out $O/baseline --repeats 2
echo done > $O/DONE
