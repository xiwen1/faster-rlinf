#!/usr/bin/env python3
"""Check that the published judges are the frozen ones, and make a runnable copy of a judge for this machine.

The judges in this directory are copies of the judges that were frozen before the optimizing agents started (see
README.md). Three things in them belong to the machine they were frozen on:

- bench/bench.json names the Python and the checkpoint by absolute paths. In the published copies these read
  $ASSETS/..., and since bench.json is one of the evaluator files, the published copies no longer hash to the recorded
  evaluator_sha256. frozen.json gives the sha256 of each frozen bench.json, which is all that is needed to recompute
  the recorded hash from the published files (`verify`).
- The judge refuses to run unless the reference tree is at bench.json's reference_commit. That commit is an internal
  snapshot of upstream RLinf 2348ece that is not public (its rlinf/ equals 2348ece's apart from three apxinf files
  that no judge imports).
- Before and after every run the judge checks the frozen inputs in task/formulation.json: checkpoint shards by size
  and modification time, which cannot match on another machine, and the other files by sha256, under the frozen
  machine's paths.

Commands (Python 3.8+, standard library only):

  python3 localize.py verify [JUDGE ...]
      No GPU or checkpoint needed. For each judge (default: all four), after fetch_numerics.sh: numerics/ matches
      numerics.sha256, and the evaluator files hash to the recorded evaluator_sha256 when bench/bench.json is counted
      with its frozen sha256 from frozen.json. A match means that every evaluator file except bench.json is
      byte-identical to the file that was frozen.

  python3 localize.py apply JUDGE --to DIR --reference RLINF --assets ASSETS
      Verifies JUDGE, copies it to DIR (a new directory; keep it outside the tree you will judge, because the judge
      copies the whole candidate tree for every run) and localizes the copy:
        bench.json        python and weights under ASSETS; reference = RLINF, a clean clone of upstream RLinf at
                          2348ece3; reference_commit = 2348ece3
        formulation.json  checkpoint files checked by the sha256 in frozen.json (equal to the LFS sha256 on Hugging
                          Face at the pinned revision) instead of size and mtime; the PaliGemma tokenizer and the
                          GR00T package files looked up where this machine keeps them; evaluator_sha256 recomputed
                          for the copy, with the recorded value kept as frozen_evaluator_sha256
      Every change is listed in DIR/task/localized.json. Then every input is checked; a mismatch exits with status 1.

  python3 localize.py retime DIR BASELINE_JSON
      For a host unlike the one the judge was frozen on: sets the reference P50 per batch size in DIR's
      task/timing_ref.json to the median of BASELINE_JSON (the output of `python3 bench/bench.py baseline --out ...`
      run in DIR), keeping the frozen values as frozen_p50_ms. The benefit gate answers INVALID when the reference's
      P50 is more than 15% off this anchor; the speed-up itself is always reference vs candidate in the same session.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = json.loads((HERE / "frozen.json").read_text())
REF_COMMIT = FROZEN["reference"]["commit"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def evaluator_files(jdir):
    """The files bench.py's evaluator_sha256() hashes, in its order: EVALUATOR_FILES, then numerics/*.py sorted."""
    m = re.search(r"^EVALUATOR_FILES = \((.*?)\)$", (jdir / "bench" / "bench.py").read_text(), re.S | re.M)
    if not m:
        raise SystemExit(f"{jdir}/bench/bench.py: EVALUATOR_FILES not found")
    return re.findall(r'"([^"]+)"', m.group(1)) + [f"numerics/{p.name}"
                                                    for p in sorted((jdir / "numerics").glob("*.py"))]


def evaluator_sha256(jdir, override=None):
    """bench.py's evaluator_sha256(), with the sha256 of some files given instead of read."""
    h = hashlib.sha256()
    for rel in evaluator_files(jdir):
        digest = (override or {}).get(rel) or sha256_file(jdir / rel)
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def judge_dir(name):
    if name not in FROZEN["judges"]:
        raise SystemExit(f"unknown judge {name!r}; one of {', '.join(FROZEN['judges'])}")
    return HERE / name


def verify(name):
    """Problems that keep the published judge `name` from being the frozen one (empty list = it is)."""
    jdir = judge_dir(name)
    expected = {}
    for line in (HERE / "numerics.sha256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split()
            expected[rel] = digest
    present = {f"numerics/{p.name}" for p in (jdir / "numerics").glob("*.py")}
    if not present:
        return ["numerics/ is missing: run fetch_numerics.sh first"]
    problems = [f"{rel}: not the frozen file" for rel, d in expected.items()
                if not (jdir / rel).is_file() or sha256_file(jdir / rel) != d]
    problems += [f"{rel}: not part of the frozen package" for rel in sorted(present - set(expected))]
    recorded = json.loads((jdir / "task" / "formulation.json").read_text())["evaluator_sha256"]
    got = evaluator_sha256(jdir, {"bench/bench.json": FROZEN["judges"][name]["bench_json_sha256"]})
    if got != recorded:
        problems.append(f"evaluator files hash to {got}, recorded {recorded}")
    return problems


def cmd_verify(a):
    failed = False
    for name in a.judges or list(FROZEN["judges"]):
        problems = verify(name)
        recorded = json.loads((judge_dir(name) / "task" / "formulation.json").read_text())["evaluator_sha256"]
        if problems:
            failed = True
            print(f"{name}: NOT the frozen judge")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"{name}: frozen evaluator_sha256 {recorded} reproduced from the published files")
    return 1 if failed else 0


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


def gr00t_package_dir(python):
    code = "import importlib.util as u; print(u.find_spec('gr00t').submodule_search_locations[0])"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{python} cannot find the gr00t package (Isaac-GR00T n1.5-release): {r.stderr.strip()}")
    return Path(r.stdout.strip())


def openpi_cache_dir():
    # as openpi.shared.download.get_cache_dir(); the judge passes HOME and OPENPI_DATA_HOME through to its processes
    return Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()


def fingerprint(p, kind):
    if kind == "sha256":
        return sha256_file(p)
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def cmd_apply(a):
    name, src = a.judge, judge_dir(a.judge)
    problems = verify(name)
    if problems:
        raise SystemExit(f"{name} is not the frozen judge, not localizing it: {problems}")
    to, ref, assets = Path(a.to).resolve(), Path(a.reference).resolve(), Path(a.assets).resolve()
    if to.exists():
        raise SystemExit(f"{to} exists; --to must be a new directory")
    head, dirty = git(ref, "rev-parse", "HEAD"), git(ref, "status", "--porcelain")
    if head != REF_COMMIT or dirty:
        raise SystemExit(f"--reference {ref} must be a clean clone of {FROZEN['reference']['repository']} at "
                         f"{REF_COMMIT} (HEAD: {head or 'not a git checkout'}, uncommitted changes: {bool(dirty)})")
    shutil.copytree(src, to, ignore=shutil.ignore_patterns("out", "targets", "__pycache__", "*.pyc"))
    changes = {"bench/bench.json": {}, "task/formulation.json": {"inputs": {}}}

    bench = json.loads((to / "bench" / "bench.json").read_text())
    for key in ("python", "weights"):
        if not bench[key].startswith("$ASSETS/"):
            raise SystemExit(f"bench.json {key} = {bench[key]!r} does not start with $ASSETS/")
        local = str(assets / bench[key][len("$ASSETS/"):])
        changes["bench/bench.json"][key] = [bench[key], local]
        bench[key] = local
    changes["bench/bench.json"]["reference"] = [bench["reference"], str(ref)]
    changes["bench/bench.json"]["reference_commit"] = [bench["reference_commit"], REF_COMMIT]
    bench["reference"], bench["reference_commit"] = str(ref), REF_COMMIT
    write_json(to / "bench" / "bench.json", bench)
    for key in ("python", "weights"):
        if not Path(bench[key]).exists():
            print(f"warning: bench.json {key} {bench[key]} does not exist", file=sys.stderr)

    form = json.loads((to / "task" / "formulation.json").read_text())
    weights_sha = FROZEN["judges"][name]["weights_sha256"]
    gr00t_dir = None
    inputs = {}
    for key, spec in form["inputs"].items():
        new_spec = dict(spec)
        if key.startswith("$ASSETS/weights/"):
            rel = key[len("$ASSETS/weights/"):]
            local = assets / "weights" / rel
            if spec["kind"] == "stat":
                new_spec = {"kind": "sha256", "value": weights_sha[rel]}
        elif "/site-packages/gr00t/" in key:
            gr00t_dir = gr00t_dir or gr00t_package_dir(bench["python"])
            local = gr00t_dir / key.split("/site-packages/gr00t/", 1)[1]
        elif re.match(r"\$WORK/[^/]+/home/\.cache/openpi/", key):
            local = openpi_cache_dir() / key.split("/.cache/openpi/", 1)[1]
        else:
            raise SystemExit(f"formulation.json: no rule for input {key}")
        inputs[str(local)] = new_spec
        change = {"path": str(local)}
        if new_spec != spec:
            change.update(kind=[spec["kind"], new_spec["kind"]], value=[spec["value"], new_spec["value"]])
        changes["task/formulation.json"]["inputs"][key] = change
    form["inputs"] = inputs
    changes["task/formulation.json"]["reference_commit"] = [form["reference_commit"], REF_COMMIT]
    form["reference_commit"] = REF_COMMIT
    form["frozen_evaluator_sha256"] = form["evaluator_sha256"]
    form["evaluator_sha256"] = evaluator_sha256(to)
    changes["task/formulation.json"]["evaluator_sha256"] = [form["frozen_evaluator_sha256"], form["evaluator_sha256"]]
    write_json(to / "task" / "formulation.json", form)
    write_json(to / "task" / "localized.json", {
        "note": "a copy of the published judge localized for this machine by localize.py; it is not the frozen "
                "original. `localize.py verify` checks the published copy against the recorded hash.",
        "source": f"faster_rlinf_bench/judges/{name}",
        "frozen_evaluator_sha256": form["frozen_evaluator_sha256"],
        "local_evaluator_sha256": form["evaluator_sha256"],
        "changes": changes})

    bad = []
    for key, spec in inputs.items():
        p = Path(key)
        if not p.is_file():
            bad.append(f"missing: {key}")
        elif fingerprint(p, spec["kind"]) != spec["value"]:
            bad.append(f"differs from the frozen input: {key}")
    print(f"{name}: localized copy in {to}; evaluator_sha256 {form['evaluator_sha256']} "
          f"(frozen {form['frozen_evaluator_sha256']})")
    if bad:
        print("inputs that do not match (the judge will refuse to run):")
        for b in bad:
            print(f"  {b}")
        if any("openpi" in b for b in bad):
            print("  the PaliGemma tokenizer is downloaded by openpi on first use from "
                  "gs://big_vision/paligemma_tokenizer.model into $OPENPI_DATA_HOME (default ~/.cache/openpi)")
        if any("/weights/" in b for b in bad):
            ck = FROZEN["judges"][name]["checkpoint"]
            print(f"  checkpoint: huggingface.co/{ck['repo']} at revision {ck['revision']}")
        return 1
    print("every frozen input matches")
    return 0


def cmd_retime(a):
    d = Path(a.dir).resolve()
    base = json.loads(Path(a.baseline).read_text())
    if base.get("status") != "ok":
        raise SystemExit(f"{a.baseline}: status {base.get('status')!r}, not ok")
    tr_path = d / "task" / "timing_ref.json"
    tr = json.loads(tr_path.read_text())
    for b, v in base["batches"].items():
        new = round(statistics.median(v["p50_ms_runs"]), 3)
        tr["batches"][b].setdefault("frozen_p50_ms", tr["batches"][b]["p50_ms"])
        print(f"batch {b}: reference P50 anchor {tr['batches'][b]['frozen_p50_ms']} ms (frozen) -> {new} ms (this host)")
        tr["batches"][b]["p50_ms"] = new
    tr["retimed_from"] = str(Path(a.baseline).resolve())
    write_json(tr_path, tr)
    loc_path = d / "task" / "localized.json"
    if loc_path.exists():
        loc = json.loads(loc_path.read_text())
        loc["changes"]["task/timing_ref.json"] = {b: [v["frozen_p50_ms"], v["p50_ms"]] for b, v in tr["batches"].items()}
        write_json(loc_path, loc)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("judges", nargs="*")
    ap = sub.add_parser("apply")
    ap.add_argument("judge")
    ap.add_argument("--to", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--assets", default=os.environ.get("FRL_ASSETS"), required="FRL_ASSETS" not in os.environ)
    rt = sub.add_parser("retime")
    rt.add_argument("dir")
    rt.add_argument("baseline")
    a = p.parse_args()
    return {"verify": cmd_verify, "apply": cmd_apply, "retime": cmd_retime}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
