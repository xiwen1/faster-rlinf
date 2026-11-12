"""Compare two env_bench.py outputs bit for bit and summarize their timings.

Usage: env_compare.py <ref_out_dir> <cand_out_dir> [--videos]
Exit code 0 iff every recorded digest (reset obs, per-step obs/reward/termination/truncation)
is identical and, with --videos, every saved MP4 decodes to identical frames.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np


def load(d):
    with open(os.path.join(d, "env_bench.json")) as f:
        return json.load(f)


def flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{prefix}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def timing(r):
    eps = r["epochs_data"]
    return {
        "reset_s": [round(e["reset_s"], 3) for e in eps],
        "chunk_ms_mean": [round(1000 * np.mean(e["chunk_step_s"]), 1) for e in eps],
        "chunk_sum_s": [round(e["chunk_step_sum_s"], 2) for e in eps],
        "flush_s": [round(e["flush_s"], 3) for e in eps],
        "epoch_s": [round(e["reset_s"] + e["chunk_step_sum_s"] + e["flush_s"], 2) for e in eps],
    }


def video_frames_digest(path):
    import imageio

    h = hashlib.sha256()
    n = 0
    for fr in imageio.get_reader(path):
        h.update(np.ascontiguousarray(fr).tobytes())
        n += 1
    return n, h.hexdigest()[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--videos", action="store_true")
    a = ap.parse_args()
    r, c = load(a.ref), load(a.cand)
    keys = ("num_envs", "chunk", "chunks_per_epoch", "epochs", "config")
    for k in keys:
        if r[k] != c[k]:
            print(f"setup differs: {k} {r[k]} vs {c[k]}")
            sys.exit(2)
    fr = dict(flat([{"reset": e["reset_obs"], "steps": e["steps"]} for e in r["epochs_data"]]))
    fc = dict(flat([{"reset": e["reset_obs"], "steps": e["steps"]} for e in c["epochs_data"]]))
    bad = [k for k in fr if fr.get(k) != fc.get(k)] + [k for k in fc if k not in fr]
    print(f"compared {len(fr)} digests: {len(bad)} differ")
    for k in bad[:20]:
        print("  DIFF", k, fr.get(k), fc.get(k))
    ok = not bad
    if a.videos:
        vr = sorted(glob.glob(os.path.join(a.ref, "video", "**", "*.mp4"), recursive=True))
        vc = sorted(glob.glob(os.path.join(a.cand, "video", "**", "*.mp4"), recursive=True))
        rel = lambda p, base: os.path.relpath(p, base)  # noqa: E731
        if [rel(p, a.ref) for p in vr] != [rel(p, a.cand) for p in vc]:
            print("video file sets differ:", [rel(p, a.ref) for p in vr], [rel(p, a.cand) for p in vc])
            ok = False
        else:
            for p, q in zip(vr, vc):
                dr, dc = video_frames_digest(p), video_frames_digest(q)
                same = dr == dc
                ok &= same
                print(f"  video {rel(p, a.ref)}: frames {dr[0]} vs {dc[0]} {'identical' if same else 'DIFFERENT'}")
    tr, tc = timing(r), timing(c)
    print("ref :", json.dumps(tr))
    print("cand:", json.dumps(tc))
    er, ec = sum(tr["epoch_s"][1:] or tr["epoch_s"]), sum(tc["epoch_s"][1:] or tc["epoch_s"])
    print(f"epoch wall (excluding epoch 0 when >1 epoch): ref {er:.2f}s cand {ec:.2f}s speed-up {er / ec:.2f}x")
    print("BITWISE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
