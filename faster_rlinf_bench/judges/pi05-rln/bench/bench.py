"""pi0.5 RL rollout bench, flow_noise + joint log-probabilities (RLinf Pi0RL.predict_action_batch(mode="train"),
batch 40): one command per question.

Run from the workspace root (ws/) with the environment's Python, on a GPU leased to you (one GPU is visible).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing]
        the verdict (gates 0-2 below) -> <out>/check.json; --quick: gates 0 and 2 only; --skip-timing: no gate 1
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        one timing process of the tree (the plan below), with its speed relative to the frozen reference timing and
        whether its timed outputs equal the reference's -> <out>/time.json
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--iters N]
        torch.profiler over N batch-40 requests (default 5) after the warmup: Chrome trace, op tables by CUDA and CPU
        time, and the kernel utilization report (bench/kernel_util.py) -> <out>/profile.md, <out>/profile.json
  python bench/bench.py baseline --out <dir> [--repeats N]   (formulator) reference against itself, interleaved
  python bench/bench.py record   --tree <dir> --out <dir>    (formulator) record the cases, no comparison

The workload (bench/workload.py) is one RL rollout step: a numpy observation for a batch of ManiSkill environments ->
Pi0RL.predict_action_batch(obs, mode="train") under no_grad with the model in eval mode -> actions.cpu(), with the
other outputs (prev_logprobs, prev_values, forward_inputs, model_actions) left on the GPU, as RLinf's rollout worker
does. The model is built by the tree's own rlinf.models.embodiment.openpi.get_model from the frozen configuration
(bench/model_cfg.json: flow_noise, joint log-probabilities, 4 steps) and the frozen checkpoint; the value and noise
heads, which the checkpoint lacks, are set from a fixed generator. The initial and per-step noise come from the global
torch CUDA RNG, seeded once after the build. A timed request ends with a device synchronize: every output is complete when its time is taken.

Every GPU run waits until the visible GPU is idle (INVALID after 10 minutes), runs from a fresh copy of the tree in
its own process group, with a fresh TMPDIR and fresh compile caches (cold compile), pinned to the least busy CPUs of
the GPU's NUMA node, and kills what it started afterwards. Every command checks the frozen inputs
(task/formulation.json: checkpoint, tokenizer, norm stats, the judge's own files) before (mismatch: nothing is
measured) and after (mismatch: ERROR), and the source digest of the tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes only. Variables the bench manages (bench.json reserved_env) are refused.

Timing plan (bench.json timing): after the build, `warmup` requests at every timed batch size; then windows of
`window_iters` requests per batch size (batch 40 first, then batch 8). The latency of a run is the P50 over all its
timed requests at that batch size.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  the signatures of get_model, Pi0RL.predict_action_batch, Pi0RL.forward and Pi0RL.default_forward
                must equal the reference's (the rollout worker and the trainer call exactly these) -> else ERROR.
  1. benefit    two reference and two candidate processes, launched in the order R C C R, stay alive on the
                same GPU and CPUs and take turns window by window (the order reversed every round): while one is
                timed the others are stopped (SIGSTOP), so all see the same host conditions within seconds of each
                other, and the few percent by which two processes of the same tree differ is pooled out. Their
                fingerprints (library versions, GPU, plan) must agree (else ERROR). speedup = mean reference P50 /
                mean candidate P50 at batch 40 (batch 8 is measured and reported). INVALID (measure again later) when
                the reference's batch-40 P50 is more than timing.frozen_band off the frozen reference (a host much
                busier or quieter than the one the reference was frozen on), when the spread (median absolute deviation
                / median) of one process's window P50s exceeds timing.window_band (candidate: window_band_candidate), or when
                torch.compile compiled anything inside the timed windows. speedup <= max(benefit.min_speedup,
                1 + 2 x reference window spread) -> UNAVAILABLE (no_benefit), rejected without a precision
                evaluation. Every output of every timed request (at both batch sizes) must also equal the
                reference's, bit for bit (else INCORRECT: the timed path is not the judged computation).
  2. bitwise    the cases in bench/cases.json (a stream of rollout requests at batch 40, one at 8 and one at 3, each
                followed by the trainer's recompute of it, Pi0RL.default_forward), recorded in one fresh process.
                Exact checks (RNG fingerprints before/after every step: torch CPU and CUDA, Python random, numpy;
                invariants: denoising steps, noise method and level, action shapes, eval mode, parameter count per
                dtype; tensor names and shapes; finiteness) must hold, else INCORRECT. The precision floor is N0:
                every rollout output (actions, prev_logprobs, prev_values, model_actions, every forward_inputs
                tensor) and every replayed tensor (logprobs, values, entropy) must be bitwise identical to the
                reference's, with the same dtypes and math modes -> PASS; anything else -> INCORRECT. The full check
                also re-records the reference and requires it to equal the frozen records (else ERROR).
  The final judgement runs the same check on observation streams (timed and recorded) that are not the ones in
  bench/bench.json and bench/cases.json, with the reference recorded afresh on them.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "bench" / "bench.json").read_text())
EVALUATOR_FILES = ("bench/bench.py", "bench/bench.json", "bench/cases.json", "bench/model_cfg.json",
                   "bench/workload.py", "bench/rl_adapter.py", "bench/rl_worker.py", "bench/kernel_util.py",
                   "bench/__init__.py")


class Refused(Exception):
    pass


class WorkerFailed(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def path(value):
    p = Path(os.path.expandvars(str(value)))
    return p if p.is_absolute() else ROOT / p


def log(msg):
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def evaluator_sha256():
    h = hashlib.sha256()
    files = [ROOT / f for f in EVALUATOR_FILES] + sorted((ROOT / "numerics").glob("*.py"))
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree)
        if p.is_dir() or ".git" in rel.parts or "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        h.update(str(rel).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def file_fingerprint(p, kind):
    p = path(p)
    if kind == "sha256":
        return sha256_file(p)
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def verify_inputs(when):
    form_path = ROOT / "task" / "formulation.json"
    if not form_path.exists() and os.environ.get("PI05_FORMULATION") == "1":
        return   # formulation, before anything is frozen
    form = json.loads(form_path.read_text())
    bad = [name for name, spec in form["inputs"].items() if file_fingerprint(name, spec["kind"]) != spec["value"]]
    if evaluator_sha256() != form["evaluator_sha256"]:
        bad.append("evaluator_sha256 (bench/, numerics/)")
    if bad:
        raise Refused(f"frozen inputs changed ({when}): {bad}")


def gpu_state():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        rows = [[x.strip() for x in line.split(",")] for line in q.strip().splitlines()]
        return [{"index": int(r[0]), "memory_used_mib": int(r[1]), "util": int(r[2])} for r in rows]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


def numa_cpus():
    """The CPUs local to the visible GPU (its NUMA node), from sysfs."""
    bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", "0"],
                         capture_output=True, text=True, timeout=60).stdout.strip().lower()
    for cand in (bus, bus[-12:]):
        p = Path("/sys/bus/pci/devices") / cand / "local_cpulist"
        if cand and p.exists():
            cpus = []
            for part in p.read_text().strip().split(","):
                lo, _, hi = part.partition("-")
                cpus += list(range(int(lo), int(hi or lo) + 1))
            return cpus
    raise Refused(f"cannot find the CPUs local to GPU {bus!r}")


def _cpu_times():
    times = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if re.match(r"cpu\d+ ", line):
            f = line.split()
            vals = [int(x) for x in f[1:]]
            times[int(f[0][3:])] = (vals[3] + vals[4], sum(vals))   # idle + iowait, total
    return times


def pick_cpus():
    """The timing.cpus least busy CPUs of the GPU's NUMA node over timing.cpu_sample_s (other users share the host)."""
    t = CFG["timing"]
    local = numa_cpus()
    a = _cpu_times()
    time.sleep(t["cpu_sample_s"])
    b = _cpu_times()
    idle = {c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in local if c in a and c in b}
    chosen = sorted(sorted(idle, key=lambda c: -idle[c])[:t["cpus"]])
    return ",".join(str(c) for c in chosen), {str(c): round(idle[c], 3) for c in chosen}, \
        round(statistics.fmean(idle.values()), 3) if idle else None


def wait_idle(what, patience=600):
    start, calm = time.time(), 0
    while time.time() - start < patience:
        state = gpu_state()
        idle = all("error" not in g and g["memory_used_mib"] < CFG["idle"]["max_memory_mib"]
                   and g["util"] <= CFG["idle"]["max_util"] for g in state)
        calm = calm + 1 if idle else 0
        if calm >= 2:
            return None
        time.sleep(5)
    return f"{what}: the GPU did not become idle within {patience}s: {gpu_state()}"


@contextlib.contextmanager
def scratch_dir(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def base_env(tmp, cand_env=None):
    keep = ("PATH", "HOME", "OPENPI_DATA_HOME", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "ROBOT_PLATFORM",
            "PYTHONNOUSERSITE", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "IS_SANDBOX", "USER", "LOGNAME")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    tmp = Path(tmp)
    for sub in ("inductor", "triton", "cudacache", "xdg", "mpl"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(tmp), TORCHINDUCTOR_CACHE_DIR=str(tmp / "inductor"), TRITON_CACHE_DIR=str(tmp / "triton"),
               CUDA_CACHE_PATH=str(tmp / "cudacache"), XDG_CACHE_HOME=str(tmp / "xdg"), MPLCONFIGDIR=str(tmp / "mpl"),
               PYTHONPATH=str(ROOT), PYTHONHASHSEED="0", **CFG["fixed_env"])
    env.update(cand_env or {})
    return env


def candidate_environment(config_file):
    if not config_file:
        return {}
    cfg = json.loads(Path(config_file).read_text())
    extra = set(cfg) - {"environment"}
    if extra:
        raise Refused(f"switch file may only contain 'environment', found {sorted(extra)}")
    envs = {str(k): str(v) for k, v in cfg.get("environment", {}).items()}
    reserved = sorted(set(envs) & set(CFG["reserved_env"]) | {k for k in envs if k.startswith("PI05_BENCH")})
    if reserved:
        raise Refused(f"switches may not set variables the bench manages: {reserved}")
    return envs


def cpu_prefix(cpus):
    nice = CFG["timing"].get("nice")
    prefix = ["taskset", "-c", cpus]
    if nice and os.geteuid() == 0:
        prefix = ["nice", "-n", str(nice)] + prefix
    return prefix


def invalid_signature(text):
    hits = [s["reason"] for s in CFG["invalid_run_signatures"] if re.search(s["regex"], text)]
    return hits[0] if hits else None


def run(cmd, env, log_file, timeout, what, cpus):
    """Run in its own process group; kill the group afterwards. Returns status ok / ERROR / TIMEOUT / INVALID."""
    problem = wait_idle(what)
    if problem:
        return {"status": "INVALID", "reason": problem}
    before, t0 = gpu_state(), time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cpu_prefix(cpus) + [str(c) for c in cmd], cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc, status = proc.wait(timeout=timeout), None
        except subprocess.TimeoutExpired:
            rc, status = None, "TIMEOUT"
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    res = {"status": status or ("ok" if rc == 0 else "ERROR"), "returncode": rc, "wall_s": round(time.time() - t0, 1),
           "log": str(log_file), "gpus_before": before, "gpus_after": gpu_state(), "cpus": cpus}
    text = log_file.read_text(errors="replace")
    hit = invalid_signature(text)
    if res["status"] == "ok" and hit:
        res.update(status="INVALID", reason="invalid-run signature in the log: " + hit)
    if res["status"] == "ERROR":
        res["tail"] = text[-3000:]
    return res


def materialize(src, dest):
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"), symlinks=True)
    return dest


def check_reference():
    ref = path(CFG["reference"])
    head = subprocess.run(["git", "-C", str(ref), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ref), "status", "--porcelain"], capture_output=True, text=True).stdout
    if head != CFG["reference_commit"] or dirty.strip():
        raise Refused(f"reference tree is not the pinned commit {CFG['reference_commit']} (HEAD {head}, dirty: "
                      f"{bool(dirty.strip())})")
    return ref


# ---------------------------------------------------------------- gate 0

def _function(tree, rel, cls, name):
    src = (Path(tree) / rel).read_text()
    nodes = ast.parse(src).body
    if cls:
        found = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls]
        if not found:
            return None
        nodes = found[0].body
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def coverage(tree):
    ref = path(CFG["reference"])
    problems = []
    for item in CFG["entry_points"]:
        rel, cls, name = item["file"], item.get("class"), item["function"]
        try:
            r, c = _function(ref, rel, cls, name), _function(tree, rel, cls, name)
        except (OSError, SyntaxError) as e:
            problems.append(f"{rel}: {e}")
            continue
        label = f"{rel}:{cls + '.' if cls else ''}{name}"
        if c is None:
            problems.append(f"{label} is missing")
        elif ast.dump(r.args) != ast.dump(c.args):
            problems.append(f"{label}: signature differs from the reference")
    if problems:
        return "ERROR", {"passed": False, "stage": "structure", "problems": problems,
                         "note": "the entry points the rollout worker and the trainer call must keep their signatures"}
    return None


# ---------------------------------------------------------------- timing sessions

def plan():
    return CFG["timing"]["plan"]


class Worker:
    """One live rl_worker.py process of one tree copy (serve mode), driven over its stdin/stdout."""

    def __init__(self, label, tree, out, cand_env, offset, cpus, stack, profile_iters=0):
        t = CFG["timing"]
        tmp = stack.enter_context(scratch_dir(f"pi05rl-{label}-"))
        self.label, self.out = label, out
        self.work = materialize(tree, tmp / "tree")
        self.digest = tree_digest(self.work)
        batches = ",".join(str(p["batch"]) for p in plan())
        cmd = [CFG["python"], ROOT / "bench" / "rl_worker.py", "--tree", self.work, "--weights", CFG["weights"],
               "--out", out / label, "--seed", t["seed"], "--obs-seed", t["obs_seed"] + offset,
               "--warmup", t["warmup"], "--batches", batches]
        cmd += ["--profile-iters", profile_iters, "--profile-batch", CFG["goal"]["primary_batch"]] if profile_iters \
            else ["--serve"]
        self.log_path = out / f"{label}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w")
        stack.callback(self.log_file.close)
        self.proc = subprocess.Popen(cpu_prefix(cpus) + [str(c) for c in cmd], cwd=ROOT, env=base_env(tmp, cand_env),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
                                     start_new_session=True, text=True, bufsize=1)
        stack.callback(self.kill)

    def tail(self):
        self.log_file.flush()
        return self.log_path.read_text(errors="replace")[-3000:]

    def read(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise WorkerFailed("TIMEOUT", f"{self.label}: no answer within {timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait()
            raise WorkerFailed("ERROR", f"{self.label} exited (code {self.proc.returncode}):\n{self.tail()}")
        return json.loads(line)

    def send(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.proc.wait()
            raise WorkerFailed("ERROR", f"{self.label} exited (code {self.proc.returncode}):\n{self.tail()}")

    def signal(self, sig):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, sig)

    def kill(self):
        self.signal(signal.SIGCONT)
        self.signal(signal.SIGKILL)
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=60)


def quantile(xs, q):
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(windows):
    """Per batch size: P50 over all timed requests, window P50s, their spread, the window digests.

    Spread = median absolute deviation of the window P50s / their median: a host disturbance that hits two or three
    windows of a process (EXP-003 formulation: +38% for ~2 window slots) does not move it, a steadily noisy one does."""
    res = {}
    for b in sorted({w["batch"] for w in windows}):
        ws = [w for w in windows if w["batch"] == b]
        samples = [s for w in ws for s in w["samples_s"]]
        p50s = [w["p50_ms"] for w in ws]
        res[str(b)] = {"p50_ms": round(1e3 * quantile(samples, 0.5), 4),
                       "mean_ms": round(1e3 * statistics.fmean(samples), 4),
                       "p95_ms": round(1e3 * quantile(samples, 0.95), 4),
                       "requests": len(samples), "window_p50_ms": [round(x, 3) for x in p50s],
                       "window_spread": round(statistics.median(abs(x - statistics.median(p50s)) for x in p50s)
                                                / statistics.median(p50s), 4),
                       "window_range": round(max(p50s) / min(p50s) - 1, 4),
                       "digests": [w["digest"] for w in ws]}
    return res


def pooled(runs, b):
    """P50 over the timed requests of several processes of one tree (window P50s of all of them)."""
    p50s = [w for r in runs for w in r["batches"][b]["window_p50_ms"]]
    return {"p50_ms": round(statistics.fmean(r["batches"][b]["p50_ms"] for r in runs), 4),
            "process_p50_ms": [r["batches"][b]["p50_ms"] for r in runs],
            "process_ratio": round(runs[0]["batches"][b]["p50_ms"] / runs[-1]["batches"][b]["p50_ms"], 4),
            "window_spread": max(r["batches"][b]["window_spread"] for r in runs),
            "window_p50_ms": p50s}


def session(specs, out, offset=0):
    """Time the trees in `specs` [(label, tree, cand_env)] as live processes taking turns window by window.

    Returns {"status", "runs": {label: summary + report}, "cpus", ...}."""
    t = CFG["timing"]
    problem = wait_idle("timing")
    if problem:
        return {"status": "INVALID", "reason": problem}
    cpus, cpu_idle, node_idle = pick_cpus()
    started, before = time.time(), gpu_state()
    res = {"cpus": cpus, "cpu_idle_share": cpu_idle, "numa_node_idle_share": node_idle, "gpus_before": before,
           "order": []}
    with contextlib.ExitStack() as stack:
        try:
            workers = {}
            for label, tree, cand_env in specs:
                w = Worker(label, tree, out, cand_env, offset, cpus, stack)
                msg = w.read(CFG["timeouts_s"]["ready"])
                if not msg.get("ready"):
                    raise WorkerFailed("ERROR", f"{label}: unexpected first message {msg}")
                w.signal(signal.SIGSTOP)
                workers[label] = w
            labels = [s[0] for s in specs]
            windows = {label: [] for label in labels}
            for item in plan():
                for i in range(item["windows"]):
                    for label in (labels if i % 2 == 0 else labels[::-1]):
                        w = workers[label]
                        w.signal(signal.SIGCONT)
                        w.send({"cmd": "window", "batch": item["batch"], "start": i * item["window_iters"],
                                "n": item["window_iters"]})
                        windows[label].append(w.read(CFG["timeouts_s"]["window"]))
                        w.signal(signal.SIGSTOP)
                        res["order"].append(f"{label}:{item['batch']}:{i}")
            for label, w in workers.items():
                w.signal(signal.SIGCONT)
                w.send({"cmd": "finish"})
                w.read(CFG["timeouts_s"]["window"])
                w.proc.wait(timeout=300)
            runs = {}
            for label, w in workers.items():
                report = json.loads((out / label / "worker.json").read_text())
                report.pop("windows", None)
                runs[label] = {"batches": summarize(windows[label]), **report, "log": str(w.log_path)}
                hit = invalid_signature(w.log_path.read_text(errors="replace"))
                if hit:
                    runs[label]["invalid"] = f"invalid-run signature in the log: {hit}"
                if report.get("compiled_inside_window"):
                    runs[label]["invalid"] = f"torch.compile compiled inside the timed requests: " \
                                             f"{report['compiled_inside_window']}"
                if tree_digest(w.work) != w.digest:
                    raise WorkerFailed("ERROR", f"{label}: the tree copy changed while it ran")
            res.update(status="ok", runs=runs)
        except WorkerFailed as e:
            res.update(status=e.status, reason=e.reason)
    res.update(wall_s=round(time.time() - started, 1), gpus_after=gpu_state())
    bad = [f"{label}: {r['invalid']}" for label, r in (res.get("runs") or {}).items() if r.get("invalid")]
    if bad:
        res.update(status="INVALID", reason="; ".join(bad))
    return res


def fingerprint_diff(a, b):
    a, b = dict(a or {}), dict(b or {})
    for d in (a, b):
        d.pop("math_mode", None)   # a math-mode change is judged by gate 2 (tier), not treated as incomparable
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def brief(run_):
    return {k: run_.get(k) for k in ("batches", "build_s", "warmup_s", "first_request_s", "peak_memory_allocated_gib",
                                     "rng_after", "fingerprint", "log", "invalid")}


# ---------------------------------------------------------------- gate 1

TIMING_LABELS = ("ref1", "cand1", "cand2", "ref2")   # launch order; windows rotate through it, reversed every round


def benefit(tree, out, cand_env, offset=0):
    """Two reference and two candidate processes (launched R C C R), taking turns window by window.

    A process's latency carries a few percent of its own (two processes of the same tree differ by ~3-4%
    consistently over all their windows, EXP-003 formulation): pooling two processes per tree, launched in balanced
    order, keeps that from deciding the ratio."""
    t = CFG["timing"]
    primary = str(CFG["goal"]["primary_batch"])
    frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
    ref_src = check_reference()
    specs = [(label, ref_src if label.startswith("ref") else tree, None if label.startswith("ref") else cand_env)
             for label in TIMING_LABELS]
    s = session(specs, out, offset)
    session_info = {k: v for k, v in s.items() if k != "runs"}
    if s["status"] != "ok":
        return s["status"] if s["status"] in ("INVALID", "TIMEOUT") else "ERROR", \
            {"speedup": None, "session": session_info,
             "runs": {k: brief(v) for k, v in (s.get("runs") or {}).items()}, "reason": s.get("reason")}
    runs = s["runs"]
    refs, cands = [runs["ref1"], runs["ref2"]], [runs["cand1"], runs["cand2"]]
    perf = {"runs": {k: brief(v) for k, v in runs.items()}, "session": session_info, "goal": CFG["goal"]}
    diffs = {k: fingerprint_diff(runs["ref1"]["fingerprint"], v["fingerprint"]) for k, v in runs.items()}
    if any(diffs.values()):
        return "ERROR", dict(perf, speedup=None, reason=f"timing runs not comparable: fingerprints differ: {diffs}")
    per_batch = {}
    for b in runs["ref1"]["batches"]:
        r, c = pooled(refs, b), pooled(cands, b)
        per_batch[b] = {"reference_p50_ms": r["p50_ms"], "candidate_p50_ms": c["p50_ms"],
                        "speedup": round(r["p50_ms"] / c["p50_ms"], 4),
                        "reference_process_p50_ms": r["process_p50_ms"], "candidate_process_p50_ms": c["process_p50_ms"],
                        "reference_process_ratio": r["process_ratio"], "candidate_process_ratio": c["process_ratio"],
                        "reference_window_spread": r["window_spread"], "candidate_window_spread": c["window_spread"],
                        "frozen_reference_p50_ms": frozen["batches"][b]["p50_ms"],
                        "reference_vs_frozen": round(r["p50_ms"] / frozen["batches"][b]["p50_ms"] - 1, 4)}
    perf.update(batches=per_batch, speedup=per_batch[primary]["speedup"], primary_batch=int(primary))
    problems = []
    pb = per_batch[primary]
    if abs(pb["reference_vs_frozen"]) > t["frozen_band"]:
        problems.append(f"reference batch-{primary} P50 {pb['reference_p50_ms']:.2f} ms is "
                        f"{pb['reference_vs_frozen']:+.1%} off the frozen reference {pb['frozen_reference_p50_ms']:.2f} "
                        f"ms (> {t['frozen_band']:.0%}): the host is not in the state the reference was frozen in")
    for name, run_ in runs.items():
        band = t["window_band"] if name.startswith("ref") else t["window_band_candidate"]
        spread = run_["batches"][primary]["window_spread"]
        if spread > band:
            problems.append(f"{name}: batch-{primary} window P50s {run_['batches'][primary]['window_p50_ms']} have a "
                            f"spread (median absolute deviation / median) of {spread:.1%} (> {band:.0%}): the host was disturbed")
    perf["problems"] = problems
    if problems:
        return "INVALID", perf
    base = runs["ref1"]["batches"]
    mismatched = {f"{name}/{b}": [i for i, (x, y) in enumerate(zip(base[b]["digests"], run_["batches"][b]["digests"]))
                                  if x != y]
                  for name, run_ in runs.items() for b in base}
    perf["timed_outputs"] = {"references_equal": not any(v for k, v in mismatched.items() if k.startswith("ref")),
                             "candidate_equals_reference": not any(v for k, v in mismatched.items()
                                                                   if k.startswith("cand")),
                             "mismatched_windows": {k: v for k, v in mismatched.items() if v},
                             "rng_after_equal": len({r["rng_after"] for r in runs.values()}) == 1}
    bar = max(CFG["benefit"]["min_speedup"], 1 + 2 * pb["reference_window_spread"])
    perf["benefit"] = {"speedup": perf["speedup"], "speedup_bar": round(bar, 4), "passed": perf["speedup"] > bar,
                       "candidate_peak_memory_allocated_gib": max(r["peak_memory_allocated_gib"] for r in cands),
                       "reference_peak_memory_allocated_gib": max(r["peak_memory_allocated_gib"] for r in refs),
                       "candidate_startup_s": round(statistics.fmean(r["build_s"] + r["warmup_s"] for r in cands), 2),
                       "reference_startup_s": round(statistics.fmean(r["build_s"] + r["warmup_s"] for r in refs), 2)}
    return ("ok" if perf["benefit"]["passed"] else "UNAVAILABLE"), perf


# ---------------------------------------------------------------- gate 2

def record_cases(tree, rec, cand_env, label, offset=0):
    n = CFG["numerics"]
    cpus = pick_cpus()[0]
    with scratch_dir(f"pi05rl-rec-{label}-") as tmp:
        work = materialize(tree, tmp / "tree")
        digest = tree_digest(work)
        kwargs = {"tree": str(work), "weights": CFG["weights"], "obs_seed_offset": offset}
        cmd = [CFG["python"], "-m", "numerics", "record", "--adapter", n["adapter"], "--adapter-kwargs",
               json.dumps(kwargs), "--cases", ROOT / n["cases"], "--out", rec] + \
              (["--deterministic"] if n["deterministic"] else [])
        r = run(cmd, base_env(tmp, cand_env), Path(str(rec) + ".log"), CFG["timeouts_s"]["numerics"],
                f"record {label}", cpus)
        if r["status"] == "ok" and tree_digest(work) != digest:
            r.update(status="ERROR", reason="the tree copy changed while it ran")
    return r


def compare(ref_rec, cand_rec, out_json):
    with scratch_dir("pi05rl-cmp-") as tmp:
        cmd = [CFG["python"], "-m", "numerics", "compare", "--ref", ref_rec, "--cand", cand_rec, "--thresholds",
               ROOT / CFG["numerics"]["thresholds"], "--out", out_json]
        p = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=base_env(tmp), capture_output=True, text=True,
                           timeout=1800)
    if not Path(out_json).exists():
        return {"status": "ERROR", "errors": [p.stdout[-2000:] + p.stderr[-2000:]]}
    rep = json.loads(Path(out_json).read_text())
    return {k: rep.get(k) for k in ("status", "measured_tier", "checked", "bitwise_identical", "exact_failures",
                                    "errors", "dtype_changes", "precision_mode_changes", "closest")} | \
        {"violations": len(rep.get("violations", [])), "bitwise": rep.get("checked", 0) > 0 and
         rep.get("checked") == rep.get("bitwise_identical") and rep.get("measured_tier") == "N0",
         "report": str(out_json)}


def numerics(tree, out, cand_env, offset=0, fresh_reference=False, recheck_reference=False):
    recs = out / "records"
    if fresh_reference or offset:
        ref_rec = recs / "reference"
        r = record_cases(check_reference(), ref_rec, None, "reference", offset)
        if r["status"] != "ok":
            return {"status": "ERROR", "errors": [f"recording the reference failed: {r}"]}
    else:
        ref_rec = path(CFG["numerics"]["ref"])
        if recheck_reference:
            again = recs / "reference_recheck"
            r = record_cases(check_reference(), again, None, "reference_recheck")
            if r["status"] != "ok":
                return {"status": "ERROR", "errors": [f"re-recording the reference failed: {r}"]}
            same = compare(ref_rec, again, out / "compare_reference_recheck.json")
            if not same["bitwise"]:
                return {"status": "ERROR", "errors": ["the reference re-recorded now differs from its frozen records: "
                                                      "the environment changed; nothing can be judged"],
                        "reference_recheck": same}
    cand_rec = recs / "candidate"
    r = record_cases(tree, cand_rec, cand_env, "candidate", offset)
    if r["status"] != "ok":
        return {"status": r["status"] if r["status"] in ("INVALID", "TIMEOUT") else "ERROR",
                "errors": [f"recording the candidate failed: {r.get('reason') or r.get('tail', '')[-1500:]}"]}
    res = compare(ref_rec, cand_rec, out / "compare.json")
    res["observation_streams"] = "frozen (bench/cases.json)" if not offset else "not in bench/cases.json"
    return res


def correctness_from(num):
    if num.get("exact_failures"):
        return "INCORRECT", {"passed": False, "stage": "bitwise", "verdict": "structure", "numerics": num,
                             "note": "exact checks failed: the candidate does different work"}
    if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
        st = num.get("status") if num.get("status") in ("INVALID", "TIMEOUT") else "ERROR"
        return st, {"passed": False, "stage": "bitwise", "numerics": num,
                    "note": "the records could not be made or compared"}
    if not num["bitwise"]:
        return "INCORRECT", {"passed": False, "stage": "bitwise", "verdict": "precision_floor",
                             "measured_tier": num.get("measured_tier"), "numerics": num,
                             "note": "the precision floor is N0: every recorded tensor must be bitwise identical with "
                                     "the same dtypes and math modes"}
    return "PASS", {"passed": True, "stage": "bitwise", "verdict": "bitwise", "measured_tier": "N0", "numerics": num}


def evaluate(tree, out, cand_env, quick=False, skip_timing=False, offset=0, fresh_reference=False):
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    performance = None
    if not quick and not skip_timing:
        status, performance = benefit(tree, out, cand_env, offset=offset)
        if status in ("ERROR", "INVALID", "TIMEOUT"):
            return status, {"passed": False, "stage": "benefit", "note": (performance or {}).get("reason") or
                            "; ".join((performance or {}).get("problems") or []) or "timing failed"}, performance
        if status == "UNAVAILABLE":
            performance["measured_speedup"] = performance.pop("speedup")
            return "UNAVAILABLE", {"passed": None, "stage": "benefit", "verdict": "no_benefit",
                                   "note": "not faster than the reference: rejected without a precision evaluation"}, \
                performance
        if not performance["timed_outputs"]["references_equal"]:
            return "ERROR", {"passed": False, "stage": "benefit",
                             "note": "the reference timing processes produced different outputs"}, performance
        if not performance["timed_outputs"]["candidate_equals_reference"]:
            return "INCORRECT", {"passed": False, "stage": "benefit", "verdict": "timed_path",
                                 "note": "the outputs of the timed requests differ from the reference's: the timed "
                                         "path is not the judged computation"}, performance
    num = numerics(tree, out, cand_env, offset=offset, fresh_reference=fresh_reference,
                   recheck_reference=not quick)
    status, correctness = correctness_from(num)
    return status, correctness, performance


# ---------------------------------------------------------------- commands

def guarded(out, fn, tree=None):
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {"started": started, "evaluator_sha256": evaluator_sha256(), "argv": sys.argv[1:]}
    try:
        verify_inputs("before")
    except Refused as e:
        return dict(meta, status="ERROR", note=str(e))
    if tree is not None:
        meta["tree"] = str(tree)
        meta["tree_digest_before"] = tree_digest(tree)
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
        meta["tree_head"] = head.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True)
        meta["tree_uncommitted"] = bool(dirty.stdout.strip())
    try:
        res = fn()
    except Refused as e:
        res = {"status": "ERROR", "note": str(e)}
    try:
        verify_inputs("after")
    except Refused as e:
        res = dict(res, status="ERROR", note=str(e))
    if tree is not None:
        meta["tree_digest_after"] = tree_digest(tree)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return dict(meta, **res)


def finish(out, name, res):
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    summary = {k: res.get(k) for k in ("status", "note")}
    perf = res.get("performance") or {}
    summary["speedup"] = perf.get("speedup", perf.get("measured_speedup"))
    if perf.get("batches"):
        summary["speedup_by_batch"] = {b: v["speedup"] for b, v in perf["batches"].items()}
    for k in ("p50_ms", "speed_vs_frozen_reference", "timed_outputs_equal_reference", "speedup_ref_vs_ref"):
        if k in res:
            summary[k] = res[k]
    corr = res.get("correctness") or {}
    for k in ("stage", "verdict", "measured_tier", "note"):
        if corr.get(k) is not None:
            summary[f"correctness.{k}"] = corr[k]
    print(json.dumps(summary, indent=1), flush=True)
    return {"PASS": 0, "ok": 0, "UNAVAILABLE": 4, "INCORRECT": 2}.get(res.get("status"), 3)


def tree_arg(a):
    tree = Path(a.tree).resolve()
    if not (tree / "rlinf").is_dir():
        raise SystemExit(f"--tree {tree} is not an RLinf checkout")
    return tree


def cmd_check(a):
    out, tree = Path(a.out).resolve(), tree_arg(a)

    def body():
        cand_env = candidate_environment(a.config)
        status, correctness, performance = evaluate(tree, out, cand_env, quick=a.quick, skip_timing=a.skip_timing,
                                                    offset=a.obs_seed_offset, fresh_reference=a.fresh_reference)
        return {"status": status, "correctness": correctness, "performance": performance,
                "config": a.config, "candidate_environment": cand_env, "quick": a.quick}
    return finish(out, "check", guarded(out, body, tree))


def cmd_time(a):
    out, tree = Path(a.out).resolve(), tree_arg(a)

    def body():
        frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
        s = session([("cand", tree, candidate_environment(a.config))], out)
        if s["status"] != "ok":
            return {"status": s["status"], "note": s.get("reason") or "timing failed",
                    "session": {k: v for k, v in s.items() if k != "runs"}}
        run_ = s["runs"]["cand"]
        primary = str(CFG["goal"]["primary_batch"])
        batches = {b: {"p50_ms": v["p50_ms"], "window_spread": v["window_spread"],
                       "speed_vs_frozen_reference": round(frozen["batches"][b]["p50_ms"] / v["p50_ms"], 4),
                       "timed_outputs_equal_reference": v["digests"] == frozen["batches"][b]["digests"]}
                   for b, v in run_["batches"].items()}
        return {"status": "ok", "p50_ms": batches[primary]["p50_ms"],
                "speed_vs_frozen_reference": batches[primary]["speed_vs_frozen_reference"],
                "timed_outputs_equal_reference": all(v["timed_outputs_equal_reference"] for v in batches.values()),
                "batches": batches, "goal": CFG["goal"], "run": brief(run_),
                "session": {k: v for k, v in s.items() if k != "runs"},
                "note": "a single process against the frozen reference timing: screening only; the check times "
                        "reference and candidate taking turns on one GPU"}
    return finish(out, "time", guarded(out, body, tree))


def cmd_profile(a):
    out, tree = Path(a.out).resolve(), tree_arg(a)

    def body():
        problem = wait_idle("profile")
        if problem:
            return {"status": "INVALID", "note": problem}
        cpus = pick_cpus()[0]
        with contextlib.ExitStack() as stack:
            w = Worker("cand", tree, out, candidate_environment(a.config), 0, cpus, stack, profile_iters=a.iters)
            w.proc.stdin.close()
            try:
                rc = w.proc.wait(timeout=CFG["timeouts_s"]["profile"])
            except subprocess.TimeoutExpired:
                return {"status": "TIMEOUT", "note": "profiling did not finish", "log": str(w.log_path)}
            if rc != 0:
                return {"status": "ERROR", "note": f"profiling failed (code {rc})", "tail": w.tail()}
        run_ = json.loads((out / "cand" / "worker.json").read_text())
        trace = out / "cand" / "trace.json"
        rep_md, rep_json = out / "kernel_util.md", out / "kernel_util.json"
        subprocess.run([CFG["python"], ROOT / "bench" / "kernel_util.py", trace, "--out", rep_md, "--json", rep_json],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
        parts = [f"# profile of {tree}", "",
                 f"{a.iters} requests at batch {CFG['goal']['primary_batch']} after {CFG['timing']['warmup']} warmup "
                 f"requests per batch size (latency under the profiler: {run_['profiled']['p50_ms']:.1f} ms P50).", ""]
        for name, title in (("kernel_util.md", "Kernel utilization"), ("cand/ops_by_cuda_time.txt", "Ops by CUDA time"),
                            ("cand/ops_by_cpu_time.txt", "Ops by CPU time")):
            p = out / name
            if p.exists():
                parts += [f"## {title}", "", "```" if p.suffix == ".txt" else "", p.read_text(),
                          "```" if p.suffix == ".txt" else "", ""]
        (out / "profile.md").write_text("\n".join(parts))
        return {"status": "ok", "trace": str(trace), "report": str(out / "profile.md"), "cpus": cpus,
                "run": {k: run_.get(k) for k in ("build_s", "warmup_s", "profiled", "peak_memory_allocated_gib")}}
    return finish(out, "profile", guarded(out, body, tree))


def cmd_baseline(a):
    """The reference against itself in the check's scheme: four reference processes in the slots of TIMING_LABELS;
    ref1+ref2 against cand1+cand2 is the null ratio the check would report for an unchanged tree."""
    out = Path(a.out).resolve()

    def body():
        ref = check_reference()
        sessions = []
        for i in range(a.repeats):
            s = session([(label, ref, None) for label in TIMING_LABELS], out / f"s{i}")
            sessions.append(s)
            if s["status"] != "ok":
                return {"status": s["status"], "note": s.get("reason"),
                        "sessions": [{k: v for k, v in x.items() if k != "runs"} for x in sessions]}
        res = {"status": "ok", "sessions": [{k: v for k, v in s.items() if k != "runs"} | {
            "runs": {k: brief(v) for k, v in s["runs"].items()}} for s in sessions]}
        batches = {}
        for b in sessions[0]["runs"]["ref1"]["batches"]:
            runs = [s["runs"][k] for s in sessions for k in TIMING_LABELS]
            p50 = [r["batches"][b]["p50_ms"] for r in runs]
            null = [round(pooled([s["runs"]["ref1"], s["runs"]["ref2"]], b)["p50_ms"] /
                          pooled([s["runs"]["cand1"], s["runs"]["cand2"]], b)["p50_ms"], 4) for s in sessions]
            batches[b] = {"p50_ms_runs": p50, "spread": round(max(p50) / min(p50) - 1, 4), "null_speedup": null,
                          "window_spreads": [r["batches"][b]["window_spread"] for r in runs],
                          "digests_equal": len({tuple(r["batches"][b]["digests"]) for r in runs}) == 1}
        res["batches"] = batches
        res["speedup_ref_vs_ref"] = {b: v["null_speedup"] for b, v in batches.items()}
        return res
    return finish(out, "baseline", guarded(out, body))


def cmd_record(a):
    out, tree = Path(a.out).resolve(), tree_arg(a)

    def body():
        r = record_cases(tree, out / "records", candidate_environment(a.config), "record", a.obs_seed_offset)
        return {"status": r["status"], "run": r}
    return finish(out, "record", guarded(out, body, tree))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tree", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--config")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--skip-timing", action="store_true")
    c.add_argument("--obs-seed-offset", type=int, default=0, help=argparse.SUPPRESS)
    c.add_argument("--fresh-reference", action="store_true", help=argparse.SUPPRESS)
    c.set_defaults(fn=cmd_check)
    t = sub.add_parser("time")
    t.add_argument("--tree", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config")
    t.set_defaults(fn=cmd_time)
    f = sub.add_parser("profile")
    f.add_argument("--tree", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--config")
    f.add_argument("--iters", type=int, default=5)
    f.set_defaults(fn=cmd_profile)
    b = sub.add_parser("baseline")
    b.add_argument("--out", required=True)
    b.add_argument("--repeats", type=int, default=2)
    b.set_defaults(fn=cmd_baseline)
    r = sub.add_parser("record")
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config")
    r.add_argument("--obs-seed-offset", type=int, default=0)
    r.set_defaults(fn=cmd_record)
    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
