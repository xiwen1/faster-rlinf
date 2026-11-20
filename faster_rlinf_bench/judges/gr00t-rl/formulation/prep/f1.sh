#!/bin/bash
# Formulation run inside the gr00t-rl sandbox (PREP=1 GR00T_FORMULATION=1), one leased GPU, from ws/:
# F0 probe (window length fitted to the latency, written into bench.json), F1 reference recorded twice (bitwise
# reproducible across processes?), thresholds and comparison, F3 profile at the primary batch, F2 interleaved
# ref-vs-ref baseline (two sessions).
set -uo pipefail
O=$1
mkdir -p $O
step() { echo "=== $1 $(date +%T)"; shift; "$@"; echo "=== rc $? $(date +%T)"; }
step probe python ../.prep/probe.py $O/probe.json
step record-a python bench/bench.py record --tree targets/reference --out $O/a
step record-b python bench/bench.py record --tree targets/reference --out $O/b
step calibrate python -m numerics calibrate --ref $O/a/records --variants $O/b/records --out $O/thresholds.json
step compare python -m numerics compare --ref $O/a/records --cand $O/b/records --thresholds $O/thresholds.json --out $O/compare_ab.json
step profile python bench/bench.py profile --tree targets/reference --out $O/profile --iters 5
step baseline python bench/bench.py baseline --out $O/baseline --repeats 2
echo done > $O/DONE
