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
   (`forward` / `default_forward`) must be **bitwise identical** to the reference (tier N0).

## Results

| Judge | Agent's final tree (EXP verdict) | Merged inference tree 564f1a5 | This repository (release rejudge) |
|---|---|---|---|
| `pi05-rl` | PASS N0, batch 8 2.67× | PASS N0, batch 8 2.93× | PASS N0, batch 8 2.97× |
| `pi05-rln` | PASS N0, batch 40 1.55× | PASS N0, batch 40 1.55× | PASS N0, batch 40 1.59× |
| `oft-rl` | PASS N0, batch 8 1.70× | PASS N0, batch 8 1.69× | PASS N0, batch 8 1.70× |
| `gr00t-rl` | PASS N0, batch 8 2.84× | PASS N0, batch 8 2.91× | PASS N0, batch 8 2.82× |

Every check above used a hidden observation-seed offset that no earlier run had used, and recorded the reference
afresh. The release rejudge ran each judge, unchanged, on a clean clone of this repository. The published history was rewritten afterwards, changing only documentation and comments in scripts, so
`rlinf/` in every published commit is byte-identical to the tree that was rejudged. Every attempt reported the
recorded evaluator hash. Attempts on a host busier or noisier than at freeze time ended `INVALID` in the timing gate before any
output was compared (pi05-rl 1, pi05-rln 3, oft-rl 2, gr00t-rl 4, one of them because the GPU never became idle) and
were measured again with the same offset; no attempt failed the numerics gate.

## Running a judge

```bash
faster_rlinf_bench/judges/fetch_numerics.sh   # once: ResearchBoost-Agent's numerics/ into every judge
cd faster_rlinf_bench/judges/oft-rl          # or pi05-rl, pi05-rln, gr00t-rl
# 1. the reference: upstream RLinf 2348ece (the judges were frozen on a snapshot whose rlinf/ tree is identical to
#    2348ece apart from three apxinf files that no judge imports)
git clone https://github.com/RLinf/RLinf targets/reference && git -C targets/reference checkout 2348ece
# 2. edit the "python" and "weights" paths in bench/bench.json ($ASSETS = your environments and checkpoints,
#    see FASTER_RLINF.md#setup)
# 3. full check of a tree on one idle GPU; the reference outputs are recorded first
CUDA_VISIBLE_DEVICES=0 python bench/bench.py check --tree <path to faster-rlinf> --out out/check --fresh-reference
# optional: unseen inputs
#   ... --obs-seed-offset <any integer>
```

`out/check/check.json` holds the verdict (`correctness.verdict`, `correctness.measured_tier`), the speed-ups per batch
size and the evaluator hash. `--skip-timing` runs the numerics gate alone.

Notes:

- The reference records made at freeze time (`task/records/`) are not included: they are machine-specific and large.
  `--fresh-reference` records them again from `targets/reference`.
- Machine paths in `bench/bench.json`, `task/formulation.json`, `GOAL.md` and `formulation/` were rewritten to
  `$ASSETS` / `$WORK` / `$SUBJECT` after the runs. Because `bench/bench.json` is one of the evaluator files, the hash
  of these copies differs from the recorded one; the judging logic is unchanged.
- `numerics/` is not included: ResearchBoost-Agent does not state a license, so `fetch_numerics.sh` takes the
  package from that repository at commit f27b8ff and checks every file against `numerics.sha256`.
- Infrastructure names were removed from the published copies. `task/timing_ref.json` no longer holds the GPU UUIDs
  or the driver's GPU model string (the judge reads only the per-batch timings and output digests from this file).
  Host names are shown as `host-a`…`host-f`; in `formulation/` the authors' GPU scheduler, its GPU idle check and
  the job-name prefix are shown as `$GPU_SCHEDULER`, `$GPU_IDLE_CHECK` and plain task names; in `GOAL.md` the
  SSH alias of the login node is shown as `<login-node>`.
- The authors ran every check inside a bubblewrap sandbox on a leased GPU; running it without a sandbox has not been
  tested.
- `GOAL.md` is what the optimizing agent saw. `formulation/` holds the formulator's notes and preparation scripts
  (probe, baseline, freeze, acceptance tests with deliberately wrong candidates); they call the authors' GPU scheduler
  and sandbox wrapper (names replaced as above), so they document what was done rather than run as is.
  `pi05-b1` is the authors' earlier inference-only judge, and `EXP-003` in comments of `bench/bench.py` (an evaluator
  file, left unchanged) refers to their formulation notes; neither is published.
