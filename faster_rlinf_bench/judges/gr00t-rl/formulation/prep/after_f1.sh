#!/bin/bash
# Formulation chain after F1 (runs on the login node, detached): wait for the F1 launcher to end, then freeze the judge
# and queue the acceptance runs, but only when F1 is clean:
#   - F1 finished (gr00t-rl/.prep/f1/DONE), the two reference recordings are bitwise equal (compare_ab PASS, every
#     checked tensor bitwise identical), the ref-vs-ref baseline is ok with equal digests in every run;
#   - every reference process's batch-8 window spread in the baseline is within timing.window_band (else the bands need
#     a documented decision: stop here).
# Then: freeze.py -> ws/task, task/baseline.md from the F1 numbers, and acc.sh through launch.sh (no PREP: the judge
# is read-only, as the agent will run it). bench.json goal (target/minimum) stays null: the verdict does not read it;
# TARGET/MINIMUM go into GOAL.md / task/baseline.md (setting them in bench.json later means re-running freeze.py, which
# leaves records and timing unchanged).
#   after_f1.sh            (log: evidence/after_f1.log)
set -uo pipefail
W=$WORK
F=$W/gr00t-rl_formulation; WS=$W/gr00t-rl/ws; F1=$W/gr00t-rl/.prep/f1
log() { echo "[$(date +%T)] $*"; }
log "after_f1 pid $$ waiting for $F/evidence/f1.rc"
for i in $(seq 1 2160); do [ -f $F/evidence/f1.rc ] && break; sleep 120; done
[ -f $F/evidence/f1.rc ] || { log "F1 did not finish within 72 h"; exit 1; }
log "F1 launcher rc $(cat $F/evidence/f1.rc)"
[ -f $F1/DONE ] || { log "F1 incomplete (no DONE): stop for a manual look"; exit 1; }
python3 - "$WS" "$F1" <<'EOF' || { log "F1 not clean: stop for a manual look"; exit 1; }
import json, sys
from pathlib import Path
ws, f1 = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((ws / "bench/bench.json").read_text())
a = json.loads((f1 / "compare_ab.json").read_text())
ok = a.get("status") == "PASS" and a.get("checked", 0) > 0 and a["checked"] == a.get("bitwise_identical")
print("compare_ab", a.get("status"), a.get("checked"), a.get("bitwise_identical"))
b = json.loads((f1 / "baseline/baseline.json").read_text())
print("baseline", b.get("status"), b.get("speedup_ref_vs_ref"))
ok &= b.get("status") == "ok" and all(v["digests_equal"] for v in b["batches"].values())
p = str(cfg["goal"]["primary_batch"])
spreads = b["batches"][p]["window_spreads"]
print("window spreads", p, spreads, "band", cfg["timing"]["window_band"])
ok &= max(spreads) <= cfg["timing"]["window_band"]
sys.exit(0 if ok else 1)
EOF
log "F1 clean: freezing"
python3 $F/prep/freeze.py $WS $F1 || { log "freeze failed"; exit 1; }
python3 - "$WS" "$F1" <<'EOF'
import json, sys
from pathlib import Path
ws, f1 = Path(sys.argv[1]), Path(sys.argv[2])
t = json.loads((ws / "task/timing_ref.json").read_text())
probe = json.loads((f1 / "probe.json").read_text()) if (f1 / "probe.json").exists() else {}
lines = ["# Baseline (formulation F1)", "",
         "The reference (RLinf ec8a5f6) measured by `bench.py baseline`: four reference processes in the check's R C C R "
         "slots, two sessions, on one B200.", "",
         "| batch | P50 ms (mean of runs) | runs | spread (max/min-1) | null speedup (ref vs ref) | window spreads |",
         "|---|---|---|---|---|---|"]
for b, v in t["batches"].items():
    lines.append(f"| {b} | {v['p50_ms']} | {v['p50_ms_runs']} | {v['spread']} | {v['null_speedup']} | {v['window_spreads']} |")
lines += ["", f"- hosts: {t['hosts']}", f"- build s per run: {t['build_s_runs']}", f"- warmup s per run: {t['warmup_s_runs']}",
          f"- peak memory allocated: {t['peak_memory_allocated_gib']} GiB",
          f"- probe: {json.dumps(probe)}", "",
          "TARGET / MINIMUM: see GOAL.md (set by the formulator from the profile in F1)."]
(ws / "task" / "baseline.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
log "frozen: $(python3 -c "import json; print(json.load(open('$WS/task/formulation.json'))['evaluator_sha256'])")"
OFF=$(python3 -c "import json; print(json.load(open('$F/final/secret.json'))['acceptance_offset'])")
JOBS="a_unmodified:quick b_entry_point:quick c1_skip_layers:quick c2_extra_rng:quick c3_timed_only:quick c4_py_random:quick c4b_np_random:quick c5_replay_only:quick c6_logprob_only:quick c7_perturb:quick d_equivalent:quick f_slower:quick a_unmodified:full c1_skip_layers:full c3_timed_only:full d_equivalent:full f_slower:full a_unmodified:quick:$OFF"
log "queueing acceptance"
cd $F && LAUNCH_ENV="GR00T_ACCEPTANCE=1" bash prep/launch.sh acc bash ../.prep/acc.sh ../.prep/acc $JOBS
log "acceptance launcher ended rc $(cat $F/evidence/acc.rc 2>/dev/null)"
