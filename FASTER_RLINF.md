# faster-rlinf

faster-rlinf is a performance fork of [RLinf](https://github.com/RLinf/RLinf) (Apache-2.0, forked at
upstream commit `2348ece`, the RLinf v0.4 release notes of 2026-10-20). It makes **reinforcement-learning
training of robot policies** in simulation faster — π0.5 (openpi), OpenVLA-OFT and GR00T N1.5 trained with
PPO/GRPO on LIBERO — **without changing what is computed**. Upstream history, license and authorship are
kept; every changed or added file is listed in [MODIFICATIONS.md](MODIFICATIONS.md).

## The problem

One RLinf embodied-RL iteration runs these phases on every GPU, one after another:

1. **weight sync** — the trainer ("actor") copies its weights into the sampling copy ("rollout");
2. **rollout** — for `rollout_epoch` episodes: the simulated environments are reset; then, repeatedly,
   the policy reads the camera images + robot state and outputs a chunk of actions (**policy inference**),
   and the simulator executes the chunk and renders new images (**env step**);
3. **advantages** (negligible);
4. **actor training** — forward, backward and optimizer step over the collected data.

Measured with upstream RLinf on one B200 (π0.5, LIBERO-Object PPO): policy inference ~47% of the
iteration, environment reset ~24%, environment stepping ~28%, training ~12%. (These are RLinf's own timers, kept by
different workers; they partly overlap, so the shares add up to more than 100%.) The inference is
launch-bound (the GPU is busy ~22% of the time at batch 8), the environment does avoidable work and
waits on itself, and a large share of inference calls are exact repeats.

## Rule: same results, bit for bit

Every default optimization here is **bitwise-exact**: given the same inputs and random state it produces
exactly the same bits as upstream — actions, log-probs, values, the random-number state afterwards,
environment observations, rewards, saved videos, synced weights. Rows 1–9 of the table below are checked by a
frozen judge or a script that compares every output bit for bit with the upstream code path, either upstream RLinf
itself or the upstream path kept behind the switch (see [Verification](#verification)). Rows 10 and 11 have no check
of their own: they change where and when data is copied or freed, not what is computed. Every row with a switch can
be turned back to the upstream code path.

Upstream RLinf is not run-to-run deterministic (two runs of the *same* upstream code give different
training metrics: it never seeds Python's `random`, and the environment and some GPU kernels are not
deterministic either). So whole training curves cannot prove or disprove exactness; exactness is checked
component by component, and end-to-end runs are used to measure wall-clock time.

## What is changed

| # | Component | What upstream does | What faster-rlinf does | Switch |
|---|---|---|---|---|
| 1 | Policy inference, π0.5 | eager PyTorch, two blocking host syncs, launch-bound | prefix + every denoising step replayed as CUDA graphs (RNG saved/restored around capture, eager fallback on any failure); camera slots masked for the whole batch skip SigLIP; a camera slot masked out of a sample is not resized (it gets a zero image of the target size; this replaces openpi's `ResizeImages` in every openpi transform pipeline, SFT data loading included) | `RLINF_OPENPI_CUDA_GRAPHS=0`, `RLINF_OPENPI_SKIP_DEAD_SLOTS=0`; the resize change has no switch |
| 2 | Policy inference, OpenVLA-OFT | eager LLM + host image preprocessing | Triton fusions that round exactly where eager rounds; GPU image preprocessing with identical pixels; whole device step (incl. multinomial sampling) as one CUDA graph | `RLINF_OFT_FAST_LLM=0`, `RLINF_OFT_FAST_INPUTS=0`, `RLINF_OFT_CUDA_GRAPHS=0` |
| 3 | Policy inference, GR00T N1.5 | eager backbone + DiT, per-image PIL transform | backbone and denoising loop as CUDA graphs; batched `GR00TTransform`; vision tower overlapped with tokenization | `RLINF_GR00T_CUDA_GRAPHS=0`, `RLINF_GR00T_FAST_TRANSFORM=0` |
| 4 | Terminal value requests | once an env succeeds, every later chunk asks for a **second full inference on the same observation** just to read its value | reuse the first call's value (it depends on the observation only) and advance the RNG exactly as the repeated call would (same `random.randint` draw, same CUDA philox offset) — falls back to the real call if anything is unknown | `rollout.reuse_terminal_inference: false` |
| 5 | LIBERO env step | one pipe round trip + 8-process barrier **per control step**; full robosuite observation pickled; main process extracts images | whole action chunk per round trip; extraction in the env worker; same env calls in the same order | `env.train.fast_vector_env: false` |
| 6 | LIBERO env reset | rebuild + seed each env worker **one after another**, then 15 settle steps each with a barrier and rendering | one parallel command per worker; settle steps render only the last (kept) frame | same as 5 |
| 7 | Reset prefetch | reset happens at the start of every episode, on the critical path | the next reset's request is predicted on a copy of the env state and executed by a spare set of env workers during the current episode; used only if the real request is identical field by field | `env.train.reset_prefetch: false` |
| 8 | Training videos | text overlay + tiling of every frame inside the env step | same frames composed on a background thread | — |
| 9 | Patch weight sync | per-element host scatter of the changed entries into the CPU snapshot; per-element host `index_put` when the rollout is offloaded | dense `where(changed, new, old)` on the GPU copied back; dense patches applied on the GPU and copied back — identical bits incl. ±0.0 and NaN | `PATCH_DENSE_SNAPSHOT_UPDATE=0` |
| 10 | FSDP offload | `.to("cpu")` into fresh pageable memory (synchronous, page faults) | page-locked host buffers (torch caching host allocator), asynchronous copies, one sync per offload | `FSDP_PINNED_OFFLOAD=0` |
| 11 | Rollout offload | — | the rollout fast paths' CUDA graphs are released together with the offloaded weights (they are re-captured after reload anyway) | — |

## How the speed-ups were found

Rows 1–3 (policy inference) were found by optimizing agents (Claude Code) working under frozen judges built with
[ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent). For each model, one RL rollout step became a
benchmark task; its judge — a bitwise (N0) comparison of every output, the random-number state and the trainer-side
replay, plus a timing gate — was frozen and its hash recorded before the agent started. The agent saw only the task,
could only submit candidates to the judge, and the final verdict was computed on a clean checkout with a hidden input
offset. The judges are in [`faster_rlinf_bench/judges/`](faster_rlinf_bench/judges/README.md).

Rows 4–11 came afterwards, from profiling the end-to-end training with the inference already fast (the same kind of
agent, outside a judge loop). Rows 4–9 are checked bit for bit by scripts of their own; rows 10 and 11 have no
dedicated check (see [Verification](#verification)).

## Measured speed-up (real RLinf training, B200)

Protocol: one GPU per run with the per-GPU load of the stock 8-GPU config (8 environments,
`global_batch_size / 8`, everything else stock incl. video saving and `torch_compile`); the upstream and
the faster-rlinf run on two idle GPUs of the **same node at the same time** (shared CPU load);
3 iterations, steady state = iterations 1–2 (iteration 0 includes compilation and graph capture).
Script: [`faster_rlinf_bench/e2e/run_pair.sh`](faster_rlinf_bench/e2e/run_pair.sh). The π0.5 and GR00T rows were
measured on faster-rlinf 3c6a8ea2, before rows 9–11 (weight sync and offload) were added; they were not measured
again afterwards. The OpenVLA-OFT row was measured on 90a6a500, whose `rlinf/` is the published one.
Per-phase times are RLinf's own timers (inference `time/rollout/predict`, env step `time/env/env_interact_step`,
env reset `time/env/env/bootstrap_step`, training `time/actor_training`), kept by different workers; they overlap
and do not add up to the iteration time.

| Model / config | Upstream s/iter | faster-rlinf s/iter | **End-to-end** | inference | env step | env reset | training |
|---|---|---|---|---|---|---|---|
| π0.5, `libero_object_ppo_openpi_pi05` | 331.0 | 151.8 | **2.18×** | 156.1→37.0 s (4.2×) | 94.4→60.9 s (1.55×) | 78.2→0.3 s | 38.6→37.1 s |
| GR00T N1.5, `libero_spatial_ppo_gr00t` | 342.9 | 194.9 | **1.76×** | 104.3→28.2 s (3.7×) | 128.7→87.1 s (1.48×) | 83.0→0.3 s | 59.7→59.7 s |
| OpenVLA-OFT, `libero_10_grpo_openvlaoft` ¹ | 1087.3 | 692.2 | **1.57×** | 190.2→72.1 s (2.6×) | 426.5→270.7 s (1.58×) | 160.5→0.6 s | 262.0→262.9 s |

¹ OpenVLA-OFT offloads the actor and the rollout model every iteration (`enable_offload: True`); weight
sync went 54.6→29.2 s. This row is a pair run with [`faster_rlinf_bench/e2e/run_group.sh`](faster_rlinf_bench/e2e/run_group.sh)
(4 iterations, steady state = iterations 1–3) on two GPUs of one node that did no other work during the
whole run (process list sampled every 30 s).
An earlier pair (faster-rlinf dc463439) measured 1181.1→782.8 s, 1.51×, with a ~10% slower training phase
(315.6→349.3 s): the faster-rlinf GPU of that pair was not idle. The training phase is
GPU-compute-bound (py-spy: backward 65%, forward 12%, loading weights and optimizer ~1%), and on a clean GPU
upstream and faster-rlinf train equally fast (above: 262.0 vs 262.9 s; on one physical GPU an hour apart: 278.9 vs
280.3 s); another pair whose faster-rlinf GPU stopped being idle mid-run measured 674 s instead of
~280 s for the training phase.

Policy inference alone (frozen judges, hidden inputs, batch 8, bit-identical outputs), on this repository: π0.5 2.97×,
OpenVLA-OFT 1.70×, GR00T 2.82× (π0.5 flow-noise config, batch 40: 1.59×); earlier checks of the agents' trees and of
the merged inference tree are in [`faster_rlinf_bench/judges/`](faster_rlinf_bench/judges/README.md#results).

Numbers depend on the host: environment stepping and host-side inference are CPU-bound, and these
nodes were not reserved for the benchmark (median 1-minute load average 31–48 per run, 90th percentile up to 73, sampled every 30 s by
`run_e2e.sh`). Check that no other
process runs on the benchmark GPUs: `run_group.sh` with `FRL_SAMPLE=1` logs every process on the leased
GPUs every 30 s, plus SM clock, power and throttle reasons every 5 s.

### Tried and not adopted

- Moving the uint8→float image normalization of the training step to the GPU: `x/255*2-1` is not
  bit-identical between CPU and CUDA for large tensors (checked exhaustively,
  `faster_rlinf_bench/u8_normalize_check.py`), so it would change the training numerics.
- `rollout.pipeline_stage_num: 2` (overlap env and inference by splitting the 8 envs into 2 stages):
  0.92× on π0.5 — 4-env chunks cost almost as much as 8-env chunks and batch-4 inference almost as much as
  batch 8, so the overlap does not pay. (It also changes inference batch shapes, i.e. is not bitwise.)

## Verification

| Rows | Check | What it compares | Script |
|---|---|---|---|
| 1–3 | Policy inference | frozen judges: every output tensor + RNG fingerprint + trainer-side replay on hidden inputs, upstream RLinf vs candidate | [`faster_rlinf_bench/judges/`](faster_rlinf_bench/judges/README.md) |
| 4 | Terminal value reuse | upstream RLinf doing the real second call vs faster-rlinf reusing it: first-call outputs, terminal values, Python/numpy/torch-CPU/CUDA RNG state after every step | `faster_rlinf_bench/repeat_check.sh` |
| 5–8 | Environment | upstream RLinf vs faster-rlinf on a seeded random action stream, same Hydra config and video wrapper; SHA-256 of every reset/step observation (both cameras, state), reward, termination, truncation; decoded video frames | `faster_rlinf_bench/env_bench.py`, `env_compare.py` |
| 9 | Patch weight sync | in one faster-rlinf tree, the new path against the upstream path kept behind `PATCH_DENSE_SNAPSHOT_UPDATE=0`, on synthetic weights over several versions: patches, sender snapshots and receiver (offloaded) parameters, incl. ±0.0 flips and NaN | `faster_rlinf_bench/patch_check.py` |
| 10–11 | Offload | no dedicated check (they change where and when data is copied or freed); exercised by the OpenVLA-OFT end-to-end run, which offloads the actor and the rollout model every iteration | — |

Results at the time of writing: environment 5392 digests + 4 videos identical; terminal reuse PASS for
π0.5, GR00T and OFT; patch sync PASS; all four inference judges PASS N0 on this repository (rollout-step P50 speed-up π0.5 2.97×, GR00T 2.82×,
OpenVLA-OFT 1.70× at batch 8; π0.5 flow-noise 1.59× at batch 40); the published copies of the judges, run as their
README describes on a clean clone of this repository, PASS N0 as well.

## Setup

faster-rlinf has the dependencies of RLinf 2348ece; install it the way RLinf documents, one environment per model,
for example `bash requirements/install.sh embodied --model openpi --env libero` (likewise `--model openvla-oft`,
`--model gr00t`; add `--no-root` if the system packages are already present and you cannot run `apt`). The numbers
above were measured with CPython 3.11.15, torch 2.11.0+cu130 and flash-attn 2.8.3; the OpenVLA-OFT environment used
the openvla-oft stack (transformers 4.40.1 fork `moojink/transformers-openvla-oft` @ bc339d9, `moojink/openvla-oft`
@ e4287e9), the GR00T environment Isaac-GR00T `n1.5-release` @ 4af2b62 with transformers 4.51.3.

Checkpoints (Hugging Face, revision):

| Model | Checkpoint | Used by |
|---|---|---|
| π0.5 | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` @ `6222623f635769bfc73c9472e29fab9b7fd8e027` | end-to-end π0.5, judge `pi05-rl` |
| π0.5 | `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` @ `eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030` | judge `pi05-rln` |
| OpenVLA-OFT | `Haozhan72/Openvla-oft-SFT-libero10-traj1` @ `679a2cc1504b905e6cd954a46c80a9408a8ba3e0` | end-to-end OFT, judge `oft-rl` |
| GR00T N1.5 | `RLinf/RLinf-Gr00t-SFT-Spatial` @ `73f710e70e7d571f8d828e51e0a428f5a1e0ac22` | end-to-end GR00T, judge `gr00t-rl` |

The scripts in `faster_rlinf_bench/` read `FRL_ASSETS`, a directory with `venv/` (π0.5), `venv-oft/`, `venv-gr00t/`
and `weights/<checkpoint name>/`, for example
`huggingface-cli download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --revision 6222623f635769bfc73c9472e29fab9b7fd8e027 --local-dir $FRL_ASSETS/weights/RLinf-Pi05-LIBERO-130-fullshot-SFT`.
They also use `nvidia-smi` and `bc`, and, with `FRL_SAMPLE=1`, `py-spy` from `$FRL_ASSETS/venv/bin`
(`pip install py-spy`). Downloads go to `huggingface.co` unless `HF_ENDPOINT` is set.

## Reproducing

Run from a clone of this repository; all paths absolute.

```bash
export FRL_ASSETS=<dir with venv/, venv-oft/, venv-gr00t/ and weights/, see Setup>
UP=<clean clone of upstream RLinf at 2348ece3>; ME=$PWD; PY=$FRL_ASSETS/venv/bin/python

# rows 1-3, policy inference: the frozen judges, see faster_rlinf_bench/judges/README.md#running-a-judge

# row 4, terminal value reuse: upstream vs faster-rlinf on one GPU (models: pi05, oft, gr00t)
CUDA_VISIBLE_DEVICES=0 faster_rlinf_bench/repeat_check.sh pi05 $UP $ME $PWD/out/repeat

# rows 5-8, environment: bitwise + timing, upstream vs faster-rlinf (env_run.sh runs inside the tree it is given)
faster_rlinf_bench/env_run.sh $UP $PY bench --out $PWD/out/env-ref  --epochs 4 --digest-all 1 --policy-ms 150
faster_rlinf_bench/env_run.sh $ME $PY bench --out $PWD/out/env-cand --epochs 4 --digest-all 1 --policy-ms 150
$PY faster_rlinf_bench/env_compare.py $PWD/out/env-ref $PWD/out/env-cand --videos

# row 9, patch weight sync: new vs upstream path in this tree, one GPU; prints BITWISE PASS or FAIL
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$ME $PY faster_rlinf_bench/patch_check.py

# end-to-end as in the π0.5 and GR00T rows: 2 GPUs of one node, 3 iterations
# -> faster_rlinf_bench/e2e/runs/pi05-summary.md (FRL_E2E_OUT sets the output root)
CUDA_VISIBLE_DEVICES=0,1 faster_rlinf_bench/e2e/run_pair.sh pi05 $UP $ME pi05 3      # or gr00t

# end-to-end as in the OpenVLA-OFT row: 4 iterations, GPU/process sampling and actor py-spy profiles
CUDA_VISIBLE_DEVICES=0,1 FRL_SAMPLE=1 faster_rlinf_bench/e2e/run_group.sh oft 4 oft ref:$UP cand:$ME
$PY faster_rlinf_bench/e2e/train_breakdown.py faster_rlinf_bench/e2e/runs/oft-group/oft-cand-actor.raw

# an ablation: the same tree with one switch off, here reset prefetch
CUDA_VISIBLE_DEVICES=0,1 faster_rlinf_bench/e2e/run_group.sh oft 4 oftpf cand:$ME nopf:$ME:+env.train.reset_prefetch=false
```

## Known limitations

Found in a review of the code before publication. None of them changes a result in the configurations measured
above; they matter when the code is used outside them. `rlinf/` was left byte-identical to the tree the judges
passed, so these are documented here rather than fixed.

- **Silent fallback.** When capturing or replaying a fast path fails (the CUDA graphs of π0.5, OpenVLA-OFT and
  GR00T), the model logs one warning and runs the upstream eager path for that input shape from then on; the GR00T
  batched transform falls back without logging. A bug in a fast path therefore shows up as lost speed, not as an
  error. The capture code catches `BaseException`, so a Ctrl-C that arrives during a capture becomes such a fallback
  instead of stopping the program.
- **Configurations checked bit for bit.** The π0.5 fast path is used for every noise method, but the judges cover
  `flow_sde` (LIBERO) and `flow_noise` with joint log-prob (ManiSkill) only; other combinations (for example
  `flow_ode`) have not been compared with upstream. The resize change of row 1 relies on a masked camera slot not
  reaching any output, which does not hold for RLT with `rlt_use_mask=False`; no released config uses that
  combination.
- **Hardware and libraries.** Tested on NVIDIA B200 with CUDA only. `openvla_oft/rlinf/fast_llm.py` imports Triton
  when the module is imported, so loading the OpenVLA-OFT model needs Triton even with `RLINF_OFT_FAST_LLM=0`. The
  page-locked FSDP offload (row 10) calls `torch.cuda.synchronize()`; on other accelerators set
  `FSDP_PINNED_OFFLOAD=0`. The GR00T graph path mirrors internals of transformers 4.51.3 and flash-attn 2.x without
  a version check; with other versions set `RLINF_GR00T_CUDA_GRAPHS=0`.
- **Reset prefetch** (row 7) keeps a spare set of env workers alive, which doubles the env processes, their EGL
  contexts and host memory, and errors in the spare set are not caught: they stop the training instead of falling
  back to a normal reset. `env.train.reset_prefetch: false` turns it off.
- **Memory.** The page-locked host buffers of the FSDP offload stay in torch's caching host allocator and are not
  returned to the system.
- **GR00T processor.** The batched transform caches per-shape data in the image processor's `__dict__` under tuple
  keys; afterwards `repr()`, `to_json_string()` and `save_pretrained()` of that processor raise `TypeError`.
- **Graph release** (row 11) frees the rollout model's CUDA graphs by attribute name; the graphs of a DAgger expert
  model are not released.
- **Tests and style.** There are no unit tests for the new code paths; they are checked by the judges and scripts
  under [Verification](#verification), which need GPUs and the checkpoints. The changed files do not pass RLinf's
  ruff configuration (formatting, import order, copyright headers of new files); reformatting them would change no
  syntax tree except the order of imports.

## License and attribution

RLinf is © The RLinf Authors, licensed under Apache-2.0 (see [LICENSE](LICENSE)). The changes in this
fork are released under the same license. The upstream README follows unchanged in
[README.md](README.md) below the faster-rlinf header.
