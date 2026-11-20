"""Freeze the gr00t-rl judge's reference artifacts into ws/task (formulation only).

    python3 freeze.py <ws> <f1 dir>

<f1 dir> holds two reference recordings (a, b), the thresholds calibrated from them and their comparison, and
`bench.py baseline` (reference against itself, interleaved, in <f1>/baseline). Writes task/records/ref,
task/thresholds.json, task/timing_ref.json, task/formulation.json.
"""
import json
import shutil
import statistics
import sys
from pathlib import Path

ws, f1 = (Path(p).resolve() for p in sys.argv[1:3])
sys.path.insert(0, str(ws))
from bench.bench import CFG, evaluator_sha256, sha256_file, summarize  # noqa: E402

task = ws / "task"
task.mkdir(exist_ok=True)

a = json.loads((f1 / "compare_ab.json").read_text())
assert a["status"] == "PASS" and a["checked"] == a["bitwise_identical"] > 0, "reference not bitwise reproducible"
if (task / "records" / "ref").exists():
    shutil.rmtree(task / "records" / "ref")
shutil.copytree(f1 / "a" / "records", task / "records" / "ref")
shutil.copy(f1 / "thresholds.json", task / "thresholds.json")

base = json.loads((f1 / "baseline" / "baseline.json").read_text())
assert base["status"] == "ok", f"baseline failed: {base.get('note')}"
assert all(v["digests_equal"] for v in base["batches"].values()), "baseline runs produced different outputs"
runs = [r for s in base["sessions"] for r in s["runs"].values()]
batches = {}
for b, v in base["batches"].items():
    batches[b] = {"p50_ms": round(statistics.fmean(v["p50_ms_runs"]), 3), "p50_ms_runs": v["p50_ms_runs"],
                  "spread": v["spread"], "null_speedup": v["null_speedup"],
                  # recomputed from the window P50s with the evaluator's own spread (the baseline may predate it)
                  "window_spreads": [summarize([{"batch": b, "samples_s": [0.0], "p50_ms": x, "digest": ""}
                                                for x in r["batches"][b]["window_p50_ms"]])[b]["window_spread"]
                                     for r in runs],
                  "digests": runs[0]["batches"][b]["digests"]}
timing = {"batches": batches, "rng_after": runs[0]["rng_after"],
          "build_s_runs": [r["build_s"] for r in runs], "warmup_s_runs": [r["warmup_s"] for r in runs],
          "peak_memory_allocated_gib": runs[0]["peak_memory_allocated_gib"], "fingerprint": runs[0]["fingerprint"],
          "hosts": sorted({s["runs"]["ref1"]["fingerprint"]["host"] for s in base["sessions"]}),
          "plan": CFG["timing"]}
(task / "timing_ref.json").write_text(json.dumps(timing, indent=1) + "\n")

weights = Path(CFG["weights"])
inputs = {}
# The checkpoint: shards by size and mtime (7.6 GB; their sha256 = the HF LFS oids, checked at download), every
# other file (config, index, experiment_cfg/metadata.json with the normalization statistics) by content.
for name in sorted(str(p.relative_to(weights)) for p in weights.rglob("*") if p.is_file()):
    kind = "stat" if name.endswith(".safetensors") else "sha256"
    p = weights / name
    if kind == "sha256":
        value = sha256_file(p)
    else:
        st = p.stat()
        value = f"{st.st_size}:{st.st_mtime_ns}"
    inputs[str(p)] = {"kind": kind, "value": value}
eagle = Path(CFG["python"]).parents[1] / "lib/python3.11/site-packages/gr00t/model/backbone/eagle2_hg_model"
for p in sorted(eagle.iterdir()):
    if p.is_file():
        inputs[str(p)] = {"kind": "sha256", "value": sha256_file(p)}
form = {"task": CFG["task"], "reference_commit": CFG["reference_commit"], "evaluator_sha256": evaluator_sha256(),
        "inputs": inputs, "records": "task/records/ref", "timing_ref": "task/timing_ref.json",
        "precision_floor": CFG["agent_max_tier"], "goal": CFG["goal"]}
(task / "formulation.json").write_text(json.dumps(form, indent=1) + "\n")
print(json.dumps({"p50_ms": {b: v["p50_ms"] for b, v in batches.items()},
                  "null_speedup": {b: v["null_speedup"] for b, v in batches.items()},
                  "evaluator_sha256": form["evaluator_sha256"]}, indent=1))
