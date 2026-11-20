"""Formulation probe (inside the oft-rl sandbox, PREP=1, from ws/): build the reference, time a few requests at each
planned batch size, and set bench.json timing.plan window_iters so that one window lasts about TARGET_S seconds
(clamped to [8, 40]). The timing protocol itself (windows, interleaving, gates) is unchanged; only the window length
is fitted to this model's latency.   python ../.prep/probe.py <out.json>"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from bench import workload  # noqa: E402
from bench.bench import CFG  # noqa: E402
from numerics.record import seed_everything  # noqa: E402

TARGET_S = 3.0
seed_everything(0)
t0 = time.perf_counter()
model = workload.build_model("targets/reference", CFG["weights"])
torch.cuda.synchronize()
build_s = time.perf_counter() - t0
seed_everything(1)
lm = model.language_model
res = {"build_s": round(build_s, 1), "attn_implementation": getattr(lm.config, "_attn_implementation", None),
       "vision_backbone": type(model.vision_backbone).__name__, "dtype": str(next(model.parameters()).dtype),
       "batches": {}}
for item in CFG["timing"]["plan"]:
    b = item["batch"]
    for i in range(5):
        workload.request(model, workload.observation(99, 10_000 + i, batch=b))
    ts = []
    for i in range(10):
        obs = workload.observation(99, i, batch=b)
        torch.cuda.synchronize()
        s = time.perf_counter()
        workload.request(model, obs)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    p50 = statistics.median(ts)
    iters = max(8, min(40, round(TARGET_S / p50)))
    res["batches"][str(b)] = {"p50_ms": round(1e3 * p50, 2), "min_ms": round(1e3 * min(ts), 2), "window_iters": iters}
    item["window_iters"] = iters
res["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
p = Path("bench/bench.json")
cfg = json.loads(p.read_text())
cfg["timing"]["plan"] = CFG["timing"]["plan"]
p.write_text(json.dumps(cfg, indent=1) + "\n")
Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
