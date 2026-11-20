#!/bin/bash
# Freeze after the re-measure (login node): goal into bench.json, freeze inputs = F1's records/thresholds/comparison
# (two bitwise-identical reference recordings) + the chosen quiet baseline, freeze.py, task/baseline.md.
#   freeze2.sh <baseline run dir, e.g. f2a> <target> <minimum>
set -euo pipefail
W=$WORK
F=$W/gr00t-rl_formulation; WS=$W/gr00t-rl/ws; PR=$W/gr00t-rl/.prep
B=$1; T=$2; M=$3
python3 - $WS $T $M <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "bench/bench.json"; c = json.loads(p.read_text())
c["goal"]["target_speedup"], c["goal"]["minimum_speedup"] = float(sys.argv[2]), float(sys.argv[3])
p.write_text(json.dumps(c, indent=1) + "\n"); print("goal", c["goal"])
PY
rm -rf $PR/freeze-in; mkdir -p $PR/freeze-in
ln -s ../f1/a $PR/freeze-in/a; ln -s ../f1/compare_ab.json $PR/freeze-in/compare_ab.json
ln -s ../f1/thresholds.json $PR/freeze-in/thresholds.json; ln -s ../$B/baseline $PR/freeze-in/baseline
cp $PR/f1/probe.json $PR/freeze-in/probe.json
python3 $F/prep/freeze.py $WS $PR/freeze-in
python3 - $WS $PR/freeze-in $B <<'PY'
import json, sys
from pathlib import Path
ws, fi, b_run = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
t = json.loads((ws / "task/timing_ref.json").read_text())
g = json.loads((ws / "bench/bench.json").read_text())["goal"]
lines = ["# Baseline (formulation)", "",
         f"The reference (RLinf ec8a5f6) measured by `bench.py baseline` ({b_run}): four reference processes in the check's "
         "R C C R slots, two sessions, on one B200; windows of 25 requests at batch 8 and 30 at batch 1.", "",
         "| batch | P50 ms (mean of runs) | runs | spread (max/min-1) | null speedup (ref vs ref) | window spreads |",
         "|---|---|---|---|---|---|"]
for b, v in t["batches"].items():
    lines.append(f"| {b} | {v['p50_ms']} | {v['p50_ms_runs']} | {v['spread']} | {v['null_speedup']} | {v['window_spreads']} |")
lines += ["", f"- hosts: {t['hosts']}", f"- build s per run: {t['build_s_runs']}", f"- warmup s per run: {t['warmup_s_runs']}",
          f"- peak memory allocated: {t['peak_memory_allocated_gib']} GiB",
          "- profile (batch 8, quiet host): about 31 ms of GPU kernels per request (2900 kernel launches) out of a "
          "~135 ms request: the rest is host time (Python dispatch and launches, the modality transforms and tokenizer).",
          "- the request is mostly host time, so its latency depends on the host: on a busy host (F1, NUMA node ~39% idle) "
          "the same reference measured 150-625 ms per window. A check on a busy host comes back INVALID (frozen band or "
          "window spread); run it again.",
          "", f"TARGET {g['target_speedup']}x, MINIMUM {g['minimum_speedup']}x (batch 8, check speedup)."]
(ws / "task" / "baseline.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "evaluator_sha256 $(python3 -c "import json; print(json.load(open('$WS/task/formulation.json'))['evaluator_sha256'])")"
