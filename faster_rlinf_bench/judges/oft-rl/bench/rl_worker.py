"""One timing or profiling process of a tree (launched by bench.py; do not time with anything else).

The process builds the model with the tree's own factory (bench/workload.py), seeds the RNGs, and serves warmup
requests at every timed batch size. Then:

  --serve    it answers commands from bench.py, one JSON line each on stdin, with one JSON line on its original
             stdout (everything the model prints goes to the log): {"cmd": "window", "batch": B, "start": i, "n": n}
             serves requests i..i+n-1 of the observation stream at batch B, each timed on the host from the call to
             the end of a device synchronize (so every output, not only the host actions, is complete), and returns
             the latencies and the digest of every output of those requests (taken outside the timed span);
             {"cmd": "finish"} writes the report and exits. bench.py keeps this process stopped (SIGSTOP) while it
             times the other tree, so the two never run at the same time.
  --profile  torch.profiler over --profile-iters requests at batch --profile-batch.

Two processes that serve the same commands from the same seeds produce the same outputs, request for request.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import workload  # noqa: E402
from numerics.record import math_mode, rng_fingerprint, seed_everything  # noqa: E402

WARMUP_STREAM = 1_000_000   # warmup requests use their own observation indices, never timed ones


def compile_counters():
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    return {f"{group}/{name}": int(value) for group in ("stats", "frames", "recompiles", "graph_break")
            for name, value in counters.get(group, {}).items()}


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out = f"unavailable: {e}"
    return out


def stats(samples):
    s = sorted(samples)
    return {"n": len(s), "p50_ms": 1e3 * s[len(s) // 2], "mean_ms": 1e3 * statistics.fmean(s),
            "p95_ms": 1e3 * s[min(len(s) - 1, int(0.95 * len(s)))], "min_ms": 1e3 * s[0], "max_ms": 1e3 * s[-1]}


def timed_request(model, obs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    actions, result = workload.request(model, obs)
    torch.cuda.synchronize()
    return time.perf_counter() - start, actions, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=7)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batches", default="8,1")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--profile-iters", type=int, default=0)
    p.add_argument("--profile-batch", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)   # the model's prints go to the log, not into the protocol
    batches = [int(b) for b in a.batches.split(",")]

    seed_everything(a.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = workload.build_model(a.tree, a.weights)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    seed_everything(a.seed + 1)

    t0 = time.perf_counter()
    first = {}
    for b in batches:
        for i in range(a.warmup):
            dt, _, _ = timed_request(model, workload.observation(a.obs_seed, WARMUP_STREAM + i, batch=b))
            first.setdefault(str(b), round(dt, 4))
    warmup_s = time.perf_counter() - t0
    before = compile_counters()
    fingerprint = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                   "gpu": gpu_info(), "math_mode": math_mode(), "host": platform.node(), "warmup": a.warmup,
                   "batches": batches, "obs_seed": a.obs_seed, "seed": a.seed}
    report = {"argv": sys.argv[1:], "build_s": round(build_s, 3), "warmup_s": round(warmup_s, 3),
              "first_request_s": first, "compile_counters_after_warmup": before, "fingerprint": fingerprint}

    if a.profile_iters:
        from torch.profiler import ProfilerActivity, profile

        obs = [workload.observation(a.obs_seed, i, batch=a.profile_batch) for i in range(a.profile_iters)]
        samples = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for o in obs:
                dt, _, _ = timed_request(model, o)
                samples.append(dt)
        prof.export_chrome_trace(str(out / "trace.json"))
        for sort, name in (("self_cuda_time_total", "ops_by_cuda_time.txt"), ("self_cpu_time_total", "ops_by_cpu_time.txt")):
            (out / name).write_text(prof.key_averages().table(sort_by=sort, row_limit=40) + "\n")
        report.update(requests_profiled=a.profile_iters, profile_batch=a.profile_batch,
                      profiled=stats(samples))
    elif a.serve:
        proto.write(json.dumps({"ready": True, "build_s": report["build_s"], "warmup_s": report["warmup_s"]}) + "\n")
        windows = []
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg["cmd"] == "finish":
                break
            b, start, n = int(msg["batch"]), int(msg["start"]), int(msg["n"])
            obs = [workload.observation(a.obs_seed, start + i, batch=b) for i in range(n)]
            h, samples = hashlib.sha256(), []
            for o in obs:
                dt, actions, result = timed_request(model, o)
                samples.append(dt)
                workload.digest_update(h, workload.flatten(actions, result))
            w = {"batch": b, "start": start, "n": n, "samples_s": samples, "digest": h.hexdigest(),
                 "rng_after": rng_fingerprint(), **stats(samples)}
            windows.append(w)
            proto.write(json.dumps(w) + "\n")
        report["windows"] = windows
    after = compile_counters()
    report.update({
        "rng_after": rng_fingerprint(),
        "compiled_inside_window": {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)
                                   if after.get(k, 0) != before.get(k, 0)},
        "peak_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    })
    (out / "worker.json").write_text(json.dumps(report, indent=1) + "\n")
    if a.serve:
        proto.write(json.dumps({"finished": True}) + "\n")


if __name__ == "__main__":
    main()
