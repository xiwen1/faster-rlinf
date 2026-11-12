# faster-rlinf

faster-rlinf is a performance fork of [RLinf](https://github.com/RLinf/RLinf) (Apache-2.0, forked at
upstream commit `2348ece`, the RLinf v0.4 release notes of 2026-10-20). It makes **reinforcement-learning
training of robot policies** in simulation faster — π0.5 (openpi), OpenVLA-OFT and GR00T N1.5 trained with
PPO/GRPO on LIBERO — **without changing what is computed**. Upstream history, license and authorship are
kept; every changed file is listed in [MODIFICATIONS.md](MODIFICATIONS.md).

## The problem

One RLinf embodied-RL iteration runs these phases on every GPU, one after another:

1. **weight sync** — the trainer ("actor") copies its weights into the sampling copy ("rollout");
2. **rollout** — for `rollout_epoch` episodes: the simulated environments are reset; then, repeatedly,
   the policy reads the camera images + robot state and outputs a chunk of actions (**policy inference**),
   and the simulator executes the chunk and renders new images (**env step**);
3. **advantages** (negligible);
4. **actor training** — forward, backward and optimizer step over the collected data.

Measured with upstream RLinf on one B200 (π0.5, LIBERO-Object PPO): policy inference ~47% of the
iteration, environment reset ~24%, environment stepping ~28%, training ~12%. The inference is
launch-bound (the GPU is busy ~22% of the time at batch 8), the environment does avoidable work and
waits on itself, and a large share of inference calls are exact repeats.

## Rule: same results, bit for bit

Every default optimization here is **bitwise-exact**: given the same inputs and random state it produces
exactly the same bits as upstream — actions, log-probs, values, the random-number state afterwards,
environment observations, rewards, saved videos, synced weights. Each one is checked by a script that
runs upstream and faster-rlinf side by side and compares every output bit for bit (see
[Verification](#verification)). Each has a switch that restores the upstream code path.

Upstream RLinf is not run-to-run deterministic (two runs of the *same* upstream code give different
training metrics: it never seeds Python's `random`, and the environment and some GPU kernels are not
deterministic either). So whole training curves cannot prove or disprove exactness; exactness is checked
component by component, and end-to-end runs are used to measure wall-clock time.

## What is changed

| # | Component | What upstream does | What faster-rlinf does | Switch |
|---|---|---|---|---|
| 1 | Policy inference, π0.5 | eager PyTorch, two blocking host syncs, launch-bound | prefix + every denoising step replayed as CUDA graphs (RNG saved/restored around capture, eager fallback on any failure); camera slots masked for the whole batch skip SigLIP | `RLINF_OPENPI_CUDA_GRAPHS=0`, `RLINF_OPENPI_SKIP_DEAD_SLOTS=0` |
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

## Measured speed-up (real RLinf training, B200)

Protocol: one GPU per run with the per-GPU load of the stock 8-GPU config (8 environments,
`global_batch_size / 8`, everything else stock incl. video saving and `torch_compile`); the upstream and
the faster-rlinf run on two idle GPUs of the **same node at the same time** (shared CPU load);
3 iterations, steady state = iterations 1–2 (iteration 0 includes compilation and graph capture).
Script: [`faster_rlinf_bench/e2e/run_pair.sh`](faster_rlinf_bench/e2e/run_pair.sh).

| Model / config | Upstream s/iter | faster-rlinf s/iter | **End-to-end** | inference | env step | env reset | training |
|---|---|---|---|---|---|---|---|
| π0.5, `libero_object_ppo_openpi_pi05` | 331.0 | 151.8 | **2.18×** | 156.1→37.0 s (4.2×) | 94.4→60.9 s (1.55×) | 78.2→0.3 s | 38.6→37.1 s |
| GR00T N1.5, `libero_spatial_ppo_gr00t` | 342.9 | 194.9 | **1.76×** | 104.3→28.2 s (3.7×) | 128.7→87.1 s (1.48×) | 83.0→0.3 s | 59.7→59.7 s |
| OpenVLA-OFT, `libero_10_grpo_openvlaoft` | 1181.1 | 782.8 | **1.51×** | 188.7→77.3 s (2.4×) | 404.3→263.8 s (1.53×) | 195.8→0.6 s | 315.6→349.3 s ¹ |

¹ OpenVLA-OFT offloads the actor and the rollout model every iteration (`enable_offload: True`); weight
sync went 80.6→35.3 s. Its training phase was ~10% slower in the faster-rlinf run of both OFT pairs; the
rollout's CUDA graphs stayed resident on the GPU while the rollout weights were offloaded, which is fixed
by row 11 below (measurement pending).

Policy inference alone (frozen judges, hidden inputs, batch 8, bit-identical outputs): π0.5 2.93×,
OpenVLA-OFT 1.69×, GR00T 2.91× (π0.5 flow-noise config, batch 40: 1.55×).

Numbers depend on the host: environment stepping and host-side inference are CPU-bound, and these
nodes were not reserved for the benchmark (load average 30–45 during the runs above).

### Tried and not adopted

- Moving the uint8→float image normalization of the training step to the GPU: `x/255*2-1` is not
  bit-identical between CPU and CUDA for large tensors (checked exhaustively,
  `faster_rlinf_bench/u8_normalize_check.py`), so it would change the training numerics.
- `rollout.pipeline_stage_num: 2` (overlap env and inference by splitting the 8 envs into 2 stages):
  0.92× on π0.5 — 4-env chunks cost almost as much as 8-env chunks and batch-4 inference almost as much as
  batch 8, so the overlap does not pay. (It also changes inference batch shapes, i.e. is not bitwise.)

## Verification

| Check | What it compares | Script |
|---|---|---|
| Policy inference | frozen judges: every output tensor + RNG fingerprint + trainer-side replay on hidden inputs, reference vs candidate | (judge harness, see research notes) |
| Environment | seeded random action stream, same Hydra config and video wrapper; SHA-256 of every reset/step observation (both cameras, state), reward, termination, truncation; decoded video frames | `faster_rlinf_bench/env_bench.py`, `env_compare.py` |
| Terminal value reuse | upstream doing the real second call vs faster-rlinf reusing it: first-call outputs, terminal values, Python/numpy/torch-CPU/CUDA RNG state after every step | `faster_rlinf_bench/repeat_check.sh` |
| Patch weight sync | patches, sender snapshots and receiver (offloaded) parameters, new vs old path, incl. ±0.0 flips and NaN | `faster_rlinf_bench/patch_check.py` |

Results at the time of writing: environment 5392 digests + 4 videos identical; terminal reuse PASS for
π0.5, GR00T and OFT; patch sync PASS; all four inference judges PASS on the merged tree.

## Reproducing

```bash
# environment: bitwise + timing, upstream vs faster-rlinf
faster_rlinf_bench/env_run.sh <upstream-tree> <python> bench --out out/ref  --epochs 4 --digest-all 1 --policy-ms 150
faster_rlinf_bench/env_run.sh <this-tree>     <python> bench --out out/cand --epochs 4 --digest-all 1 --policy-ms 150
python faster_rlinf_bench/env_compare.py out/ref out/cand --videos

# end-to-end, 2 GPUs of one node (CUDA_VISIBLE_DEVICES=a,b), 3 iterations
FRL_ASSETS=<dir with venv/ and weights/> faster_rlinf_bench/e2e/run_pair.sh pi05 <upstream-tree> <this-tree> pi05 3
```

## License and attribution

RLinf is © The RLinf Authors, licensed under Apache-2.0 (see [LICENSE](LICENSE)). The changes in this
fork are released under the same license. The upstream README follows unchanged in
[README.md](README.md) below the faster-rlinf header.
