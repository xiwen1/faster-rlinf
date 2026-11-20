# Goal: make the GR00T N1.5 RL rollout step faster, bit for bit

## The task

[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) is NVIDIA's 3B vision-language-action model: an Eagle 2.5
vision-language backbone (a SigLIP2 vision encoder and the first 12 layers of Qwen3-1.7B) embeds the camera images and
the instruction once; a flow-matching action head (a 16-layer diffusion transformer over the robot state, learned
"future" tokens and the noisy action chunk, cross-attending to the backbone features) then turns Gaussian noise into a
chunk of 16 actions in a few denoising steps. RLinf trains it with reinforcement learning (PPO on LIBERO): at every
environment step, its rollout worker asks the model for the next action chunk of a batch of simulated robots in
mode "train", where the denoising chain is made stochastic (`flow_sde`: one of the 4 steps, drawn at random, adds
Gaussian noise of level 0.5), and keeps what the trainer needs later — the log-probability of that stochastic step
(`prev_logprobs`), the value estimate (`prev_values`, from the value head) and the model's inputs and whole denoising
chain (`forward_inputs`). The trainer recomputes the log-probability and value of the same step from those inputs and
uses the rollout's `prev_logprobs` in the PPO probability ratio, so the rollout's numbers are training numbers: a
rollout that is faster and **bitwise identical** leaves the whole training run unchanged.

- **Code**: `ws/RLinf`, a snapshot of RLinf (`release/v0.4`, 2348ece) on branch `opt`. It is your candidate: change
  anything in it. It does not start from the plain snapshot: it starts from commit e09ab89, the result of two earlier
  tasks on RLinf's pi0.5 model (`rlinf/models/embodiment/openpi/`: CUDA-graph fast paths of
  `Pi0RL.predict_action_batch`; read `git -C RLinf log` and those tasks' logs in `notes/`). Those commits do not
  touch GR00T: in this task the candidate is, as it stands, as slow as the reference.
  `ws/targets/reference` is the plain snapshot, never changed: the reference the judge compares with.
- **The model code**: `rlinf/models/embodiment/gr00t/` (`gr00t_n1d5/` is the version this configuration uses;
  `simulation_io.py` converts observations and actions), and the installed packages `gr00t` (Isaac-GR00T
  n1.5-release: `gr00t/model/gr00t_n1.py`, `gr00t/model/backbone/` with the Eagle 2.5 remote code,
  `gr00t/model/action_head/` with the DiT, `gr00t/data/transform/` and `gr00t/model/transforms.py` for the host-side
  preprocessing), `transformers` 4.51.3 (Qwen3, SigLIP) and `diffusers` 0.30.2 (the DiT's attention). The installed
  packages are read-only: anything you change goes into `ws/RLinf` (you may re-implement or override any part of the
  computation there).
- **The workload** (`bench/workload.py`, `bench/model_cfg.json`): the model is built by the tree's own
  `rlinf.models.embodiment.gr00t.get_model` with RLinf's `model/gr00t.yaml` and the overrides of
  `libero_spatial_ppo_gr00t.yaml` (bf16; two cameras; 5 action chunks of 7 dimensions out of a 16 x 32 action
  horizon; 4 denoising steps; prompt padded to 570 tokens; value head) from the checkpoint
  `RLinf/RLinf-Gr00t-SFT-Spatial` (the SFT checkpoint RLinf's LIBERO-Spatial PPO recipe starts from). One request is
  one rollout step for a batch of environments: the LIBERO env's observation (torch CPU tensors: per environment a
  256x256 uint8 agent-view image, a 256x256 uint8 wrist image, an 8-dim state and an instruction) passed to
  `predict_action_batch(env_obs=obs, mode="train")` under `torch.no_grad()` with the model in eval mode; the actions
  come back on the host, every other output stays where the model leaves it. The initial noise and the noise of
  every denoising step are drawn from the global torch CUDA RNG (`torch.randn`, `torch.normal`); the index of the
  stochastic step from Python's `random` (`random.randint`). Every timed request has a new observation and ends with
  a device synchronize.
- **The metric**: the P50 latency of one rollout step at **batch 8** (RLinf's 64 LIBERO training environments over
  the 8 GPUs of one node) on one B200. Batch 1 is measured and reported too.
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

0. **Structure**: the signatures of both `get_model` factories (`gr00t/__init__.py`, `gr00t/gr00t_n1d5/__init__.py`)
   and of `GR00T_N1_5_ForRLActionPrediction.predict_action_batch`, `.forward` and `.default_forward` must stay those
   of the reference (the rollout worker and the trainer call exactly these) -> else ERROR.
1. **Benefit**: two reference and two candidate processes run live on the same GPU and CPUs and take turns, window
   by window (while one is timed, the others are stopped). speedup = mean reference P50 / mean candidate P50 at
   batch 8. The reference must be within a band of its frozen timing and every process's windows must be steady,
   else INVALID (measure again). A speedup that does not exceed `max(1.10, 1 + 2 x reference window spread)` is
   UNAVAILABLE (no_benefit). Every output of every timed request, at both batch sizes, must equal the reference's
   bit for bit, else INCORRECT.
2. **Bitwise** (the precision floor is **N0**): a stream of rollout requests (batch 8, plus one each at batch 1, 3,
   16 and 32), each followed by the trainer's recompute of it (`default_forward` on the request's own
   `forward_inputs` with the rollout's `prev_logprobs`, values on, forward only), recorded in a fresh process. The
   RNG fingerprints before and after every step (torch CPU and CUDA, Python `random`, numpy), the invariants (action
   chunking, horizon and dimensions, denoising steps, noise method and level, value head, embodiment, image count,
   prompt padding, eval mode, parameter count per dtype), tensor names and shapes must match, and every rollout
   output (`actions`, `prev_logprobs`, `prev_values`, every `forward_inputs` tensor: `chains`, `denoise_inds`,
   `state`, `state_mask`, `embodiment_id`, `eagle_*`) and every recomputed tensor (`logprobs`, `prev_logprobs`,
   `values`) must be **bitwise identical** to the reference's, with the same dtypes and math modes. Anything else is
   INCORRECT.

There is no tolerance: a change that alters a single bit of any output, or draws one random number more or fewer,
fails, however small. The noise is drawn on every denoising step, also on the deterministic ones where it is scaled
by zero: skipping those draws moves the CUDA RNG and changes every later request.

Durations (one GPU, measured at formulation): see `task/baseline.md`. Screen with `time` and `check --quick`; run the
full check when you are confident.

## Running commands

Your shell is on a workstation; the work happens on the B200 cluster over SSH (`ssh -o BatchMode=yes <login-node> '...'`).
Let `TOOLS=$WORK/tools/gr00t-rl`.

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
  classes and log the top classes and repeated chains of the kernel table in the WORK_LOG. The profile measures the
  GPU side; this request also has a large host side (the modality transforms: image crop and resize, normalization,
  the Eagle processor's tokenization; the action unnormalization in numpy): read the CPU op table and the gaps in the
  trace as well.
- **Pace: at most 30 minutes per round**, counted from starting the round's profile: profile -> change one hotspot ->
  re-time with `time` -> log the round. Failed and no-gain rounds are logged too. Waiting for a GPU and the full
  check do not count.
- Append every round to `ws/WORK_LOG.md`: the profile findings, the change, the time result, the check result (if
  run).
- **Constraints**:
  - the computation must not change: same checkpoint, same configuration, the same random draws in the same order
    from the same generators, the same outputs bit for bit (the N0 floor). No dtype or math-mode changes (TF32,
    reduced-precision reductions, attention backends or kernels that change results), no quantization. The host-side
    preprocessing is part of the computation: `eagle_pixel_values`, `eagle_input_ids`, `state` are outputs.
  - the code must not behave differently depending on whether it is being timed, recorded or judged, nor on which
    observations it sees (the final judgement uses new ones).
  - the same request must keep working at other batch sizes (the judge records batch 1, 3, 16 and 32 requests), with
    other camera frames and instructions, in mode "eval", and with the other configurations of this model (other `num_action_chunks`,
    other `denoising_steps`, no value head, other noise levels, `ignore_last`, the `flow_cps` and `reinflow` noise
    methods, other `value_vlm_mode`, `safe_get_logprob`): you may specialize a fast path, but every other case must
    still take a correct path. Shape-dependent rewrites are a known trap: kernels are chosen by shape, so a
    "mathematically equivalent" rewrite that changes a shape (batching, splitting, padding) usually changes bits.
  - **the earlier tasks' results must stay**: the pi0.5 code (`rlinf/models/embodiment/openpi/`) must keep giving the
    reference's numbers as fast as at e09ab89. The final judgement re-runs the earlier checks (pi05-rl: LIBERO
    `flow_sde`, batch 8, 2.9x at e09ab89; pi05-rln: ManiSkill `flow_noise`, batch 40, 1.55x at e09ab89; oft-rl, if
    its judge is frozen by then) on your commit, and a formulator-side recording of several GR00T configurations x
    call variants (including the trainer's recompute with autograd on) against the reference; any difference there
    fails this task too.
  - the trainer's path (`forward` / `default_forward`) must keep giving the reference's numbers and must still be
    trainable (autograd through the action head and the value head); do not break it to speed up the rollout.
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
  `git -C RLinf config user.name gr00t-agent; git -C RLinf config user.email gr00t-agent@localhost`).
- "The target cannot be reached within the rules" requires that every large `short`, `under-filled` or `bw-poor`
  kernel class, every large repeated chain in the latest kernel report and every large host-side span (gaps where the
  GPU idles) has had an attempt (a written, checked and timed change) or a logged reason why it cannot be changed.
- End the WORK_LOG with a summary: the final commit and switch file, the final check path and result, a breakdown of
  the gains, and the directions that did not work and why.
- Write `ws/FINAL.json`: `{"commit": "<sha on opt>", "config": "<switch file or null>", "check": "runs/<name>/check.json"}`.
