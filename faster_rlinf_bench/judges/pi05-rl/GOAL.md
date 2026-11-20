# Goal: make the pi0.5 RL rollout step faster, bit for bit

## The task

[pi0.5](https://www.physicalintelligence.company/blog/pi05) is a vision-language-action model: it takes two camera
images, the robot's state and an instruction, and returns a chunk of robot actions. RLinf trains it with
reinforcement learning (PPO): at every environment step, its rollout worker asks the model for the next actions of a
batch of simulated robots, and keeps what the trainer needs later — the log-probability of the sampled actions
(`prev_logprobs`), the value estimate (`prev_values`) and the denoising chain (`forward_inputs`). The trainer uses
those `prev_logprobs` directly in the PPO probability ratio (`recompute_logprobs: False`), so the rollout's numbers
are training numbers: a rollout that is faster and **bitwise identical** leaves the whole training run unchanged.

- **Code**: `ws/RLinf`, a snapshot of RLinf (`release/v0.4`, 2348ece) on branch `opt`. It is your candidate: change
  anything in it. `ws/targets/reference` is the same snapshot, never changed: the reference the judge compares with.
- **The workload** (`bench/workload.py`, `bench/model_cfg.json`): the model is built by the tree's own
  `rlinf.models.embodiment.openpi.get_model` with RLinf's `pi0_5` RL configuration (`pi05_libero`; `flow_sde` noise,
  noise level 0.5, 5 denoising steps, an action horizon of 10, 5 executed actions of 7 dimensions, a value head on
  the VLM output, the OpenPI default mixed precision) from the checkpoint `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`;
  the value head, which the checkpoint lacks, is set from a fixed generator after the build. One request is one
  rollout step for a batch of environments: a numpy observation (per environment two 224x224 uint8 images, an 8-dim
  state, a LIBERO instruction) passed to `Pi0RL.predict_action_batch(obs, mode="train")` under `torch.no_grad()` with
  the model in eval mode, with the actions returned to the host as `actions.cpu()` and every other output left on the
  GPU. The initial and per-step noise are drawn from the global torch CUDA RNG, the denoising step that gets the SDE
  noise from Python's `random`. Every timed request has a new observation and ends with a device synchronize.
- **The metric**: the P50 latency of one rollout step at **batch 8** (64 LIBERO environments over 8 GPUs) on one
  B200. Batch 40 (320 ManiSkill environments over 8 GPUs) is measured and reported too.
- **Baseline** (`task/baseline.md`, `task/timing_ref.json`): see there.

## The single criterion

`status == "PASS"` and `performance.speedup >= 2.0` in the `check.json` written by

    python bench/bench.py check --tree RLinf --out runs/<name> [--config <switch file>]

If a profile shows that 2.0 cannot be reached within the rules below, `performance.speedup >= 1.5` with
`status == "PASS"` is also acceptable for wrapping up, provided the WORK_LOG explains why the remaining hotspots cannot
be addressed (see "Wrapping up").

The final judgement repeats this check on new observation streams (both the timed and the recorded ones differ from
those in `bench/bench.json` and `bench/cases.json`), from a clean checkout of the commit you name in `FINAL.json`.

## How the judge decides (bench/bench.py, numerics/: read them)

0. **Structure**: the signatures of `get_model`, `Pi0RL.predict_action_batch`, `Pi0RL.forward` and
   `Pi0RL.default_forward` must stay those of the reference (the rollout worker and the trainer call exactly these)
   -> else ERROR.
1. **Benefit**: two reference and two candidate processes run live on the same GPU and CPUs and take turns, window
   by window (while one is timed, the others are stopped). speedup = mean reference P50 / mean candidate P50 at
   batch 8. The reference must be within a band of its frozen timing and every process's windows must be steady,
   else INVALID (measure again). A speedup that does not exceed `max(1.10, 1 + 2 x reference window spread)` is
   UNAVAILABLE (no_benefit). Every output of every timed request, at both batch sizes, must equal the reference's
   bit for bit, else INCORRECT.
2. **Bitwise** (the precision floor is **N0**): a stream of rollout requests (batch 8, plus one at batch 40 and one at
   batch 3), each followed by the trainer's recompute of it (`Pi0RL.default_forward` on the request's own
   `forward_inputs`, forward only), recorded in a fresh process. The RNG fingerprints before and after every step
   (torch CPU and CUDA, Python `random`, numpy), the invariants (denoising steps, noise method and level, action
   shapes, eval mode, parameter count per dtype), tensor names and shapes must match, and every rollout output
   (`actions`, `prev_logprobs`, `prev_values`, `model_actions`, every `forward_inputs` tensor) and every recomputed
   tensor (`logprobs`, `values`, `entropy`) must be **bitwise identical** to the reference's, with the same dtypes and
   math modes. Anything else is INCORRECT.

There is no tolerance: a change that alters a single bit of any output, or draws one random number more or fewer,
fails, however small.

Durations (one GPU, measured at formulation): see `task/baseline.md`. Screen with `time` and `check --quick`; run the
full check when you are confident.

## Running commands

Your shell is on a workstation; the work happens on the B200 cluster over SSH (`ssh -o BatchMode=yes <login-node> '...'`).
Let `TOOLS=$WORK/tools/pi05-rl`.

- **Every command that reads or writes the workspace** goes through the sandbox, which starts in `ws/`:
  `ssh -o BatchMode=yes <login-node> "$TOOLS/agent-run.sh bash -c 'git -C RLinf status'"`. Inside it you see the
  workspace, the read-only Python environment and checkpoint, and your home (`$HOME`, with the skill cards). Write
  files with heredocs or small Python scripts run through `agent-run.sh`.
- **Every GPU command** goes through `$TOOLS/agent-gpu.sh <name> <command ...>`, which waits for an idle B200, runs
  the command in the sandbox on it and returns at once; then `$TOOLS/agent-wait.sh <name>` (waits up to 9 minutes;
  call it again while it says "still running"). Example:
  `ssh -o BatchMode=yes <login-node> "$TOOLS/agent-gpu.sh r01-profile python bench/bench.py profile --tree RLinf --out runs/r01-profile"`.
  Names are unique; use `r<round>-<what>`. The cluster is shared and busy: a run may wait for a GPU.
- The GPUs are on several hosts, and the same code can run slower on one host than on another. `time` compares with
  a frozen reference measured at formulation, so its `speed_vs_frozen_reference` is a screen only; the check
  measures reference and candidate taking turns on the same GPU, and only its ratio counts. The hosts' CPUs are not reserved for the judge:
  a check can come back INVALID because the host was disturbed. That is the judge refusing a noisy
  measurement, not a verdict on your code; run it again.
- Never run GPU work any other way, never use `CUDA_VISIBLE_DEVICES`, and never start background processes that
  outlive your command.

## Rules

- **Every round starts with a profile before any change**, and **timing only through bench.py** (`time`,
  `profile`, `check`). Do not write your own timing scripts. The profile includes the kernel utilization report
  (`runs/<name>/kernel_util.md`, see the `kernel-utilization-analysis` skill); keep a bytes model for the top kernel
  classes and log the top classes and repeated chains of the kernel table in the WORK_LOG.
- **Pace: at most 30 minutes per round**, counted from starting the round's profile: profile -> change one hotspot ->
  re-time with `time` -> log the round. Failed and no-gain rounds are logged too. Waiting for a GPU and the full
  check do not count.
- Append every round to `ws/WORK_LOG.md`: the profile findings, the change, the time result, the check result (if
  run).
- **Constraints**:
  - the computation must not change: same checkpoint, same configuration, the same denoising steps, the same noise
    draws in the same order from the same generators, the same outputs bit for bit (the N0 floor). No dtype or
    math-mode changes (TF32, reduced-precision reductions, attention backends that change results), no quantization.
  - the code must not behave differently depending on whether it is being timed, recorded or judged, nor on which
    observations it sees (the final judgement uses new ones).
  - the same request must keep working at other batch sizes (the judge records batch 3 and batch 40 requests), with
    a caller-supplied `noise` or `rng`, and with the other configurations the code supports (other step counts,
    `flow_noise`, `joint_logprob`, no value head): you may specialize a fast path, but every other case must still
    take a correct path.
  - the trainer's path (`forward` / `default_forward`) must keep giving the reference's numbers and must still be
    trainable (autograd through the action expert); do not break it to speed up the rollout.
  - change only `ws/RLinf` and your own files under `ws/runs`, `ws/WORK_LOG.md`, `ws/FINAL.json` and `$HOME`. The
    judge (`bench/`, `numerics/`, `task/`, `targets/`) and the Python environment are read-only; installing packages
    is not allowed.
  - startup cost counts: the check reports it (`benefit.candidate_startup_s`). Anything that has to compile or
    capture must finish within the build and the warmup requests (20 per batch size); compiling inside the timed
    requests makes the run INVALID.
- Do not search the cluster outside the sandbox, and do not look for other implementations of this model.

## Skills

Your home has skill cards (`$HOME/.claude/skills/*/SKILL.md`): priors summarized from other optimization tasks
(mostly training; the profiling, measurement and kernel-analysis cards apply directly). Re-verify them on this task
before treating them as conclusions.

## Wrapping up

- Commit each change separately on branch `opt` of `ws/RLinf` (local git identity only:
  `git -C RLinf config user.name pi05-agent; git -C RLinf config user.email pi05-agent@localhost`).
- "The target cannot be reached within the rules" requires that every large `short`, `under-filled` or `bw-poor`
  kernel class and every large repeated chain in the latest kernel report has had an attempt (a written, checked and
  timed change) or a logged reason why it cannot be changed.
- End the WORK_LOG with a summary: the final commit and switch file, the final check path and result, a breakdown of
  the gains, and the directions that did not work and why.
- Write `ws/FINAL.json`: `{"commit": "<sha on opt>", "config": "<switch file or null>", "check": "runs/<name>/check.json"}`.
