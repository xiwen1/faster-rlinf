"""Compare RLinf timing between a reference run and a candidate run (or several pairs).

Usage: analyze_e2e.py [--skip N] <ref_run> <cand_run> [<ref_run> <cand_run> ...]
Reads <run>/metrics.json (dump_tb.py). Iterations < N (default 1: iteration 0 holds compilation,
CUDA-graph capture and process start-up) are excluded from the steady-state means.
Prints the per-iteration wall time, every time/* stage and the ref/cand ratio.
"""
import json
import os
import statistics as st
import sys

MAIN = [
    "time/step",
    "time/sync_weights",
    "time/generate_rollouts",
    "time/env/env_interact_step",
    "time/env/env/bootstrap_step",
    "time/rollout/predict",
    "time/cal_adv_and_returns",
    "time/actor_training",
]


def load(d):
    with open(os.path.join(d, "metrics.json")) as f:
        return json.load(f)


def series(m, tag):
    return {int(k): v for k, v in m.get(tag, {}).items()}


def main():
    args = sys.argv[1:]
    skip = 1
    if args and args[0] == "--skip":
        skip = int(args[1])
        args = args[2:]
    for i in range(0, len(args), 2):
        R, C = load(args[i]), load(args[i + 1])
        rn, cn = os.path.basename(args[i].rstrip("/")), os.path.basename(args[i + 1].rstrip("/"))
        tags = MAIN + sorted(t for t in set(R) | set(C) if t.startswith("time/") and t not in MAIN)
        steps = sorted(set(series(R, "time/step")) | set(series(C, "time/step")))
        print(f"\n## {rn} (ref) vs {cn} (cand), seconds; steady state = iterations >= {skip}")
        print("| stage | " + " | ".join(f"ref it{s} | cand it{s}" for s in steps) + " | ref steady | cand steady | ref/cand |")
        print("|---" * (2 * len(steps) + 4) + "|")
        for t in tags:
            r, c = series(R, t), series(C, t)
            if not r and not c:
                continue
            cells = []
            for s in steps:
                cells += [f"{r[s]:.1f}" if s in r else "-", f"{c[s]:.1f}" if s in c else "-"]
            rs = [r[s] for s in steps if s >= skip and s in r]
            cs = [c[s] for s in steps if s >= skip and s in c]
            rm = st.mean(rs) if rs else float("nan")
            cm = st.mean(cs) if cs else float("nan")
            ratio = rm / cm if cs and cm else float("nan")
            print(f"| {t} | " + " | ".join(cells) + f" | {rm:.1f} | {cm:.1f} | {ratio:.2f}x |")
        rstep = [v for s, v in series(R, "time/step").items() if s >= skip]
        cstep = [v for s, v in series(C, "time/step").items() if s >= skip]
        if rstep and cstep:
            print(f"\nEND-TO-END steady-state speed-up {rn} -> {cn}: {st.mean(rstep):.1f} s -> {st.mean(cstep):.1f} s = "
                  f"{st.mean(rstep) / st.mean(cstep):.2f}x  (n_ref={len(rstep)}, n_cand={len(cstep)})")


if __name__ == "__main__":
    main()
