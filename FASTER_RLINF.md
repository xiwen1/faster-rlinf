# faster-rlinf

faster-rlinf is a performance fork of [RLinf](https://github.com/RLinf/RLinf) (Apache-2.0,
forked at upstream commit `2348ece` = RLinf v0.4 release notes, 2026-10-20). It speeds up
**RL training of robot policies** (VLA models trained with PPO/GRPO in simulation) without
changing what is computed. Upstream history, license and authorship are kept unchanged; every
file changed here is listed in [MODIFICATIONS.md](MODIFICATIONS.md).

## What problem this solves

One RLinf embodied-RL iteration has four phases that run one after another on each GPU:

1. **weight sync** – the actor's weights are copied to the rollout model;
2. **rollout** – simulated environments (LIBERO / MuJoCo) and the policy alternate: the policy
   looks at the camera images and robot state and outputs a chunk of actions, the simulator
   executes them and renders the next images; this repeats for `rollout_epoch` episodes;
3. **advantages** – GAE / group-relative advantages (negligible time);
4. **actor training** – PPO/GRPO forward + backward + optimizer.

Measured on a real run (π0.5, LIBERO-Object PPO, one B200, upstream code): policy inference is
~50% of the iteration, environment stepping ~29%, environment reset ~19%, training ~12%.
The policy inference is launch-bound (the GPU is busy only ~22% of the time at batch 8), and the
environment does avoidable work. faster-rlinf removes that overhead.

## Rule: same results, bit for bit

Every optimization in this fork is either

- **bitwise-exact (tier N0)** – for identical inputs and RNG state, it produces exactly the same
  bits as upstream (actions, log-probs, values, RNG state afterwards, environment observations,
  rewards). These are on by default. They are verified by *frozen judges*: a separate harness that
  records the reference outputs, runs the candidate on hidden inputs, compares every output tensor
  bit-for-bit and measures latency with interleaved reference/candidate processes; or
- **opt-in** – changes scheduling or shapes (so numerics may differ in the last bits, exactly like
  changing the number of GPUs would), off unless enabled in the config. Each is documented with
  what changes.

RLinf itself is not run-to-run deterministic (it never seeds Python `random`, and two runs of
the *same* upstream code already give different metrics), so end-to-end training curves cannot
prove bit-exactness. Exactness is therefore checked at the component level (policy inference,
environment), and end-to-end runs are used to measure wall-clock time.

## What is changed (so far)

### Policy rollout inference (bitwise, verified)

| Model (RLinf config) | What was done | Speed-up of `predict_action_batch` (batch 8, B200) |
|---|---|---|
| π0.5 (openpi), LIBERO PPO `flow_sde` | no blocking host→device copies; prefix + each denoising step captured as CUDA graphs, RNG saved/restored around capture, failure-safe fallback to eager | **2.93×** |
| π0.5, ManiSkill PPO `flow_noise` + `joint_logprob` (batch 40) | same graphs extended to the noise head; camera slots masked out for the whole batch skip SigLIP | **1.55×** |
| OpenVLA-OFT, LIBERO GRPO | Triton fusions that round exactly where eager rounds (RoPE+layout, add+RMSNorm, SiLU·mul); GPU image preprocessing with identical pixels; whole step in one CUDA graph (multinomial sampling inside the graph) | **1.69×** (batch 1: 3.3×) |
| GR00T N1.5, LIBERO-Spatial PPO | backbone and DiT denoising loop as CUDA graphs; batched `GR00TTransform`; vision tower overlapped with tokenization | **2.91×** (batch 1: 4.6×) |

All four were re-judged on the merged tree: PASS at N0 (every output tensor bit-identical on
hidden inputs, including the RNG fingerprint and the training-side log-prob/value replay).

### End-to-end (real RLinf training)

| Config | Upstream s/iter | faster-rlinf s/iter | Speed-up | Notes |
|---|---|---|---|---|
| π0.5 LIBERO-Object PPO, 1 GPU, 8 envs | 477.5 | 364.0 | **1.31×** | rollout-inference changes only; rollout inference 3.28× inside training |

More end-to-end rows (OpenVLA-OFT, GR00T) and the environment/pipeline optimizations are being
added; see the [benchmark section](#reproducing-the-numbers).

## Reproducing the numbers

Benchmark scripts live in [`faster_rlinf_bench/`](faster_rlinf_bench/) (added as they land).

## License and attribution

RLinf is © The RLinf Authors, licensed under Apache-2.0 (see [LICENSE](LICENSE)). The changes in
this fork are released under the same license. The upstream README follows unchanged in
[README.md](README.md) below the faster-rlinf header.
