"""Bitwise check of the terminal-inference reuse (rlinf/models/embodiment/repeat_rng.py).

The rollout worker of upstream RLinf, for a chunk after which some env is done, calls
``predict_action_batch`` a second time on the observation it just inferred, only to read the
value. faster-rlinf replaces that second call by its random-number consumption and reuses the
first call's value. This script runs the same request stream both ways and records, per step,
a digest of every output tensor of the first call, the value the terminal request yields, and
the full RNG state afterwards (Python ``random``, numpy, torch CPU, torch CUDA):

  --mode ref   : second call really executed (upstream behaviour; run it on the upstream tree)
  --mode cand  : repeat_rng(model).advance_as_repeat() + first call's value (faster-rlinf tree)

Compare two outputs with ``--compare ref.json cand.json``. Workloads (model construction and
synthetic LIBERO observations) are in faster_rlinf_bench/workloads/<model>/workload.py.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import random
import sys

import numpy as np
import torch


def load_workload(path):
    spec = importlib.util.spec_from_file_location("frl_workload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_digest(t):
    if t is None:
        return None
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()[:24]


def rng_digest():
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()[:24],
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()[:24],
        "torch_cpu": tensor_digest(torch.get_rng_state()),
        "torch_cuda": tensor_digest(torch.cuda.get_rng_state()),
        "cuda_offset": torch.cuda.default_generators[torch.cuda.current_device()].get_offset(),
    }


def run(a):
    wl = load_workload(a.workload)
    model = wl.build_model(a.tree, a.weights)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    steps, fallbacks, reused = [], 0, 0
    for k in range(a.steps):
        obs = wl.observation(a.seed, k)
        actions, result = wl.request(model, obs)
        h = hashlib.sha256()
        wl.digest_update(h, wl.flatten(actions, result))
        rec = {"k": k, "first": h.hexdigest()[:24]}
        if k % 3 != 2:  # a terminal request after this chunk
            if a.mode == "ref":
                _, r2 = wl.request(model, obs)
                value = r2.get("prev_values")
            else:
                from rlinf.models.embodiment.repeat_rng import repeat_rng

                if repeat_rng(model).advance_as_repeat():
                    value = result.get("prev_values")
                    reused += 1
                else:
                    _, r2 = wl.request(model, obs)
                    value = r2.get("prev_values")
                    fallbacks += 1
            rec["terminal_value"] = tensor_digest(value)
        rec["rng"] = rng_digest()
        steps.append(rec)
    out = {"mode": a.mode, "tree": a.tree, "steps": steps, "reused": reused, "fallbacks": fallbacks}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{a.mode}: {len(steps)} steps, reused {reused}, fallbacks {fallbacks} -> {a.out}")


def compare(ref_path, cand_path):
    r, c = json.load(open(ref_path)), json.load(open(cand_path))
    diffs = []
    for sr, sc in zip(r["steps"], c["steps"]):
        if sr != sc:
            diffs.append((sr["k"], {k: (sr.get(k), sc.get(k)) for k in sr if sr.get(k) != sc.get(k)}))
    if len(r["steps"]) != len(c["steps"]):
        diffs.append(("length", (len(r["steps"]), len(c["steps"]))))
    print(f"steps {len(r['steps'])}; candidate reused {c['reused']} terminal calls, fell back {c['fallbacks']}")
    for d in diffs[:10]:
        print("  DIFF", d)
    print("BITWISE", "PASS" if not diffs else f"FAIL ({len(diffs)} steps differ)")
    return 0 if not diffs else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--workload")
    ap.add_argument("--tree")
    ap.add_argument("--weights")
    ap.add_argument("--mode", choices=["ref", "cand"])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a)
