# Frozen rollout judges

These are the four benchmark judges under which the policy-inference speed-ups of faster-rlinf (rows 1–3 of
[What is changed](../../FASTER_RLINF.md#what-is-changed)) were found and accepted. Each one turns one RL rollout step of
RLinf into a task with a fixed workload, a bitwise correctness gate and a timing gate. They were built with the method
of [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent): the `numerics/` package is
ResearchBoost-Agent's own (commit f27b8ff, fetched by `fetch_numerics.sh`), each judge was frozen and its hash recorded before an
optimizing agent started, the agent could only submit candidates to it, and the final verdict was computed on a clean
checkout with a hidden input offset and a freshly recorded reference.

| Judge | Workload (one B200) | Primary batch | Frozen evaluator sha256 (`task/formulation.json`) |
|---|---|---|---|
| [`pi05-rl`](pi05-rl) | π0.5, LIBERO PPO (`flow_sde`), `Pi0RL.predict_action_batch` | 8 | `eea0af04…a4f5c` |
| [`pi05-rln`](pi05-rln) | π0.5, ManiSkill PPO (`flow_noise` + joint log-prob) | 40 | `ac3a4fcb…b825` |
| [`oft-rl`](oft-rl) | OpenVLA-OFT, LIBERO-10 GRPO, sampling at temperature 1.6 | 8 | `5000db98…862be` |
| [`gr00t-rl`](gr00t-rl) | GR00T N1.5, LIBERO-Spatial PPO | 8 | `03c948b7…fdee` |

## What a check decides

`python bench/bench.py check --tree <RLinf tree> --out <dir>` runs three gates (details at the top of `bench/bench.py`):

0. **structure** — the entry points the rollout worker and the trainer call still exist with the same signatures.
1. **benefit** — two reference and two candidate processes are timed interleaved (R C C R) on the same GPU; the
   speed-up must clear a bar set from the measured window spread (for `oft-rl`: `max(1.10, 1 + 2 × spread)`); a host
   far busier or quieter than the frozen reference host, or too noisy, gives `INVALID` instead of a number.
2. **numerics** — on every case in `bench/cases.json`, every output tensor of the rollout step (actions, log-probs,
   values, forward inputs), the random-number state afterwards, and the trainer-side replay of the same inputs
   (`forward` / `default_forward`) must be **bitwise identical** to the reference. ResearchBoost-Agent's numerics
   package grades agreement in tiers; N0, the strictest, means bitwise equality with the same dtypes and math modes,
   and it is the only tier these judges accept.

## Results

| Judge | Agent's final tree | Merged inference tree | This repository (release rejudge) |
|---|---|---|---|
| `pi05-rl` | PASS N0, batch 8 2.67× | PASS N0, batch 8 2.93× | PASS N0, batch 8 2.97× |
| `pi05-rln` | PASS N0, batch 40 1.55× | PASS N0, batch 40 1.55× | PASS N0, batch 40 1.59× |
| `oft-rl` | PASS N0, batch 8 1.70× | PASS N0, batch 8 1.69× | PASS N0, batch 8 1.70× |
| `gr00t-rl` | PASS N0, batch 8 2.84× | PASS N0, batch 8 2.91× | PASS N0, batch 8 2.82× |

"Agent's final tree" is the final verdict on the last candidate of the agent working under that judge. "Merged
inference tree" is the four agents' changes merged into one tree, before the environment and training changes were
added; it is a commit of the authors' working history and is not in this repository. "Release rejudge" is the code
published here.

Every check above ran the frozen judge itself on the authors' machine, used a hidden observation-seed offset that no
earlier run had used, and recorded the reference afresh. The release rejudge ran each judge, unchanged, on a clean
clone of this repository. The published history was rewritten afterwards, changing only documentation and comments in scripts, so
`rlinf/` in every published commit is byte-identical to the tree that was rejudged. Every attempt reported the recorded evaluator hash. Attempts
on a host busier or noisier than at freeze time ended `INVALID` in the timing gate before any output was compared
(pi05-rl 1, pi05-rln 3, oft-rl 2, gr00t-rl 4, one of them because the GPU never became idle) and were measured again
with the same offset; no attempt failed the numerics gate.

The published copies were then run exactly as [Running a judge](#running-a-judge) describes, on 2026-11-25:
`localize.py verify`, `localize.py apply` with a fresh clone of upstream 2348ece3, and the full check through
`one_gpu.sh` on a clean clone of this repository (`rlinf/` as published),
one B200 per judge, without an observation offset. All four passed N0 on the first attempt: primary-batch speed-up
pi05-rl 2.85×, pi05-rln 1.58× (batch 40), oft-rl 1.69×, gr00t-rl 2.90×, with every timed output equal to the
reference's and the reference P50 at the primary batch within 4% of the frozen value.

## Running a judge

The published judges are the frozen ones with the authors' machine paths replaced, so they do not run as they are.
[`localize.py`](localize.py) first checks that they are the frozen judges and then makes a copy of one for your
machine. Everything except the GPU run needs only Python 3.8+ and the standard library.

```bash
# in a clone of this repository, with the environments and checkpoints of FASTER_RLINF.md#setup
FRL=$PWD
export FRL_ASSETS=<dir with venv/, venv-oft/, venv-gr00t/ and weights/<checkpoint name>/>
faster_rlinf_bench/judges/fetch_numerics.sh            # once: ResearchBoost-Agent's numerics/ into every judge
python3 faster_rlinf_bench/judges/localize.py verify   # the published judges reproduce the recorded hashes
# the reference tree: one clean clone of upstream RLinf 2348ece for all four judges
git clone https://github.com/RLinf/RLinf ~/rlinf-2348ece
git -C ~/rlinf-2348ece checkout 2348ece3df113874671478be1360ae3115b346e3

# per judge (oft-rl, pi05-rl, pi05-rln, gr00t-rl): a localized copy outside this clone, then the full check on
# host GPU 0 (one_gpu.sh: see below; on a single-GPU machine or in a one-GPU container, drop it)
python3 faster_rlinf_bench/judges/localize.py apply oft-rl --to ~/frl-judges/oft-rl --reference ~/rlinf-2348ece
cd ~/frl-judges/oft-rl
$FRL/faster_rlinf_bench/judges/one_gpu.sh 0 python3 bench/bench.py check --tree $FRL --out out/check --fresh-reference
# optional: inputs no earlier run has seen
#   ... --obs-seed-offset <any integer>
```

`out/check/check.json` holds the verdict (`status`; `correctness.verdict` and `correctness.measured_tier`), the
speed-ups per batch size and the evaluator hash. `--skip-timing` runs gates 0 and 2 only. `--fresh-reference`
records the reference's outputs first. On one B200 a full check took 8–14 minutes.

**One visible GPU.** A judge must see exactly one GPU. Before every GPU run it waits until every GPU that `nvidia-smi`
lists is idle (at most 2000 MiB used and 10% utilization; `INVALID` after 10 minutes), and it pins its processes to
CPUs of the NUMA node of `nvidia-smi`'s first GPU. `CUDA_VISIBLE_DEVICES` does not change what `nvidia-smi` lists,
so on a host with several GPUs a judge run directly waits for GPUs it does not use.
The authors ran every judge in a bubblewrap sandbox that exposed only the leased GPU. [`one_gpu.sh`](one_gpu.sh)
does just that part: `one_gpu.sh <host GPU index> <command>` runs the command with only that GPU's device files and
driver entry visible, where it is GPU 0 (needs [bubblewrap](https://github.com/containers/bubblewrap); set `BWRAP`
if it is not on `PATH`). A container with one GPU (`docker run --gpus '"device=N"'`) works the same way.

**What `localize.py verify` shows.** `bench/bench.py` hashes its own evaluator files (`bench/*`, `numerics/*.py`)
and refuses to run when the result differs from `evaluator_sha256` in `task/formulation.json`, the value recorded
when the judge was frozen, before the agent started. The published `bench/bench.json` differs from the frozen one in
the values of `python` and `weights`, which were the authors' machine paths. [`frozen.json`](frozen.json) gives the
sha256 of each frozen `bench.json`; with it, `verify` recomputes the recorded hash from the published files. A match
shows that every other evaluator file, `numerics/` included, is byte-identical to the frozen one.

**What `localize.py apply` changes** in the copy (each change is listed in the copy's `task/localized.json`):

- `bench/bench.json`: `python` and `weights` under `$FRL_ASSETS`, and the reference tree. The judges were frozen
  with an internal snapshot `ec8a5f66` of upstream RLinf as the reference; its `rlinf/` tree is identical to
  2348ece's apart from three `apxinf` files that no judge imports, and the snapshot commit is not public. The copy
  pins `--reference`, which must be a clean checkout of 2348ece3.
- `task/formulation.json`: the frozen inputs that every run checks before and after. Checkpoint shards were checked by
  the size and modification time of the authors' files; the copy checks them by sha256 instead, using the values in
  `frozen.json`, each of which equals the LFS sha256 that Hugging Face lists for that file at the pinned revision.
  The PaliGemma tokenizer of the π0.5 judges is looked up in openpi's cache (`$OPENPI_DATA_HOME`, default
  `~/.cache/openpi`; openpi downloads it on first use), the GR00T processor files wherever the `venv-gr00t` Python
  finds the `gr00t` package; their sha256 values are unchanged. `evaluator_sha256` is recomputed for the copy, and
  the recorded value is kept as `frozen_evaluator_sha256`.

A localized copy therefore reports its own evaluator hash; `verify` is what ties it to the frozen judge.

**Timing on other hosts.** The benefit gate compares the reference's P50 at the primary batch size with the value
recorded at freeze time on one B200 (`task/timing_ref.json`) and answers `INVALID` when the two
differ by more than 15%, because a host much busier or quieter than at freeze time would decide the ratio. On another
GPU model, or a host in a very different state, either run the numerics gate alone (`--skip-timing`), or re-anchor
the copy on your host:

```bash
$FRL/faster_rlinf_bench/judges/one_gpu.sh 0 python3 bench/bench.py baseline --out out/baseline   # 4 reference processes, twice
python3 $FRL/faster_rlinf_bench/judges/localize.py retime . out/baseline/baseline.json
```

The speed-up is always the reference against the candidate in the same session; the anchor only decides whether the
host is steady enough to measure.

Notes:

- The reference records made at freeze time (`task/records/`) are not included: they are machine-specific and large.
  `--fresh-reference` records them again from the reference tree.
- Machine paths in `bench/bench.json`, `task/formulation.json`, `GOAL.md` and `formulation/` were rewritten to
  `$ASSETS` / `$WORK` / `$SUBJECT` after the runs (see `verify` above for `bench/bench.json`).
- `numerics/` is not included: ResearchBoost-Agent does not state a license, so `fetch_numerics.sh` takes the
  package from that repository at commit f27b8ff and checks every file against `numerics.sha256`.
- Infrastructure names were removed from the published copies. `task/timing_ref.json` no longer holds the GPU UUIDs
  or the driver's GPU model string (the judge reads only the per-batch timings and output digests from this file).
  Host names are shown as `host-a`…`host-f`; in `formulation/` the authors' GPU scheduler, its GPU idle check and
  the job-name prefix are shown as `$GPU_SCHEDULER`, `$GPU_IDLE_CHECK` and plain task names; in `GOAL.md` the
  SSH alias of the login node is shown as `<login-node>`.
- The authors ran the checks in Results inside a bubblewrap sandbox on a leased GPU; `one_gpu.sh` keeps only the
  part of it that the judge's GPU checks need.
- `GOAL.md` is what the optimizing agent saw. `formulation/` holds the formulator's notes and preparation scripts
  (probe, baseline, freeze, acceptance tests with deliberately wrong candidates); they call the authors' GPU scheduler
  and sandbox wrapper (names replaced as above), so they document what was done rather than run as is.
  `pi05-b1` is the authors' earlier inference-only judge, and `EXP-003` in comments of `bench/bench.py` (an evaluator
  file, left unchanged) refers to their formulation notes; neither is published.
