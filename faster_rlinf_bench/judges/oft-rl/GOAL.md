# Goal: make the OpenVLA-OFT RL rollout step faster, bit for bit

## The task

[OpenVLA-OFT](https://openvla-oft.github.io/) is a 7.5B vision-language-action model (a fused SigLIP + DINOv2 vision
backbone, a projector and a Llama-2 7B language model): it takes a camera image and an instruction and predicts a
chunk of robot actions in one forward pass, as discrete action tokens decoded in parallel (bidirectional attention over
the action positions). RLinf trains it with reinforcement learning (GRPO on LIBERO): at every environment step, its
rollout worker asks the model for the next action chunk of a batch of simulated robots, **samples** the action tokens
(temperature 1.6), and keeps what the trainer needs later — the log-probability of the sampled tokens
(`prev_logprobs`), the value estimate (`prev_values`, zeros without a value head) and the model's inputs and sampled
tokens (`forward_inputs`). The trainer uses those `prev_logprobs` directly in the PPO/GRPO probability ratio
(`recompute_logprobs: False`), so the rollout's numbers are training numbers: a rollout that is faster and **bitwise
identical** leaves the whole training run unchanged.

- **Code**: `ws/RLinf`, a snapshot of RLinf (`release/v0.4`, 2348ece) on branch `opt`. It is your candidate: change
  anything in it. It does not start from the plain snapshot: it starts from commit e09ab89, the result of two earlier
  tasks on RLinf's pi0.5 model (`rlinf/models/embodiment/openpi/`: CUDA-graph fast paths of
  `Pi0RL.predict_action_batch`; read `git -C RLinf log` and those tasks' logs in `notes/`). Those commits do not
  touch OpenVLA-OFT: in this task the candidate is, as it stands, as slow as the reference.
  `ws/targets/reference` is the plain snapshot, never changed: the reference the judge compares with.
- **The model code**: `rlinf/models/embodiment/openvla_oft/` (`rlinf/` is the version this configuration uses), the
  processor `rlinf/models/embodiment/prismatic/processing_prismatic.py`, and the installed packages `prismatic`
  (openvla-oft) and `transformers` 4.40.1 (the openvla-oft fork with bidirectional attention in
  `models/llama/modeling_llama.py`, SDPA path). The installed packages are read-only: anything you change goes into
  `ws/RLinf` (you may re-implement or override any part of the computation there).
- **The workload** (`bench/workload.py`, `bench/model_cfg.json`): the model is built by the tree's own
  `rlinf.models.embodiment.openvla_oft.get_model` with RLinf's `model/openvla_oft.yaml` and the overrides of
  `libero_10_grpo_openvlaoft.yaml` (bf16; one camera image; 8 action chunks of 7 dimensions = 56 action tokens; the
  prompt padded to 128 tokens; no value head) from the checkpoint `Haozhan72/Openvla-oft-SFT-libero10-traj1` (the
  SFT checkpoint RLinf's LIBERO-10 GRPO recipe starts from). One request is one rollout step for a batch of
  environments: the LIBERO env's observation (torch CPU tensors: per environment a 256x256 uint8 agent-view image, a
  256x256 uint8 wrist image, which this configuration does not use, an 8-dim state and an instruction) passed to
  `predict_action_batch(env_obs=obs, do_sample=True, temperature=1.6, top_k=-1, top_p=1.0, max_new_tokens=None)`
  under `torch.no_grad()` with the model in eval mode; the actions come back on the host, every other output stays
  on the GPU. The action tokens are sampled with `torch.multinomial` from the global torch CUDA RNG. Every timed
  request has a new observation and ends with a device synchronize.
- **The metric**: the P50 latency of one rollout step at **batch 8** (RLinf's 64 LIBERO training environments over the
  8 GPUs of one node) on one B200. Batch 1 is measured and reported too.
- **Baseline** (`task/baseline.md`, `task/timing_ref.json`): see there.

## The single criterion

`status == "PASS"` and `performance.speedup >= 1.4` in the `check.json` written by

    python bench/bench.py check --tree RLinf --out runs/<name> [--config <switch file>]

If a profile shows that 1.4 cannot be reached within the rules below, `performance.speedup >= 1.2` with
`status == "PASS"` is also acceptable for wrapping up, provided the WORK_LOG explains why the remaining hotspots cannot
be addressed (see "Wrapping up").

The final judgement repeats this check on new observation streams (both the timed and the recorded ones differ from
those in `bench/bench.json` and `bench/cases.json`), from a clean checkout of the commit you name in `FINAL.json`.

## How the judge decides (bench/bench.py, numerics/: read them)

0. **Structure**: the signatures of both `get_model` factories (`openvla_oft/__init__.py`, `openvla_oft/rlinf/__init__.py`)
   and of `OpenVLAOFTForRLActionPrediction.predict_action_batch`, `.forward` and `.default_forward` must stay those of
   the reference (the rollout worker and the trainer call exactly these) -> else ERROR.
1. **Benefit**: two reference and two candidate processes run live on the same GPU and CPUs and take turns, window
   by window (while one is timed, the others are stopped). speedup = mean reference P50 / mean candidate P50 at
   batch 8. The reference must be within a band of its frozen timing and every process's windows must be steady,
   else INVALID (measure again). A speedup that does not exceed `max(1.10, 1 + 2 x reference window spread)` is
   UNAVAILABLE (no_benefit). Every output of every timed request, at both batch sizes, must equal the reference's
   bit for bit, else INCORRECT.
2. **Bitwise** (the precision floor is **N0**): a stream of rollout requests (batch 8, plus one each at batch 1, 3 and
   32), each followed by the trainer's recompute of it (`default_forward` on the request's own `forward_inputs`, with
   temperature 1.6 and top_k -1, log-probabilities and entropy, forward only), recorded in a fresh process. The RNG
   fingerprints before and after every step (torch CPU and CUDA, Python `random`, numpy), the invariants (action
   chunking, prompt length, vocabulary and action bins, image count, sampling parameters, eval mode, parameter count
   per dtype), tensor names and shapes must match, and every rollout output (`actions`, `prev_logprobs`,
   `prev_values`, every `forward_inputs` tensor: `input_ids`, `attention_mask`, `pixel_values`, `action_tokens`) and
   every recomputed tensor (`logprobs`, `entropy`) must be **bitwise identical** to the reference's, with the same
   dtypes and math modes. Anything else is INCORRECT.

There is no tolerance: a change that alters a single bit of any output, or draws one random number more or fewer,
fails, however small. The sampled tokens depend on every bit of the logits and on the CUDA RNG's position, so
`torch.multinomial` must see the same probabilities and draw from the same generator state, in the same way.

Durations (one GPU, measured at formulation): see `task/baseline.md`. Screen with `time` and `check --quick`; run the
full check when you are confident.

## Running commands

Your shell is on a workstation; the work happens on the B200 cluster over SSH (`ssh -o BatchMode=yes <login-node> '...'`).
Let `TOOLS=$WORK/tools/oft-rl`.

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
  GPU side; this request also has a large host side (the image preprocessing and tokenization in the processor, the
  numpy action decoding): read the CPU op table and the gaps in the trace as well.
- **Pace: at most 30 minutes per round**, counted from starting the round's profile: profile -> change one hotspot ->
  re-time with `time` -> log the round. Failed and no-gain rounds are logged too. Waiting for a GPU and the full
  check do not count.
- Append every round to `ws/WORK_LOG.md`: the profile findings, the change, the time result, the check result (if
  run).
- **Constraints**:
  - the computation must not change: same checkpoint, same configuration, the same random draws in the same order
    from the same generators, the same outputs bit for bit (the N0 floor). No dtype or math-mode changes (TF32,
    reduced-precision reductions, attention backends or kernels that change results), no quantization. The image
    preprocessing is part of the computation: `pixel_values` (bf16) is an output.
  - the code must not behave differently depending on whether it is being timed, recorded or judged, nor on which
    observations it sees (the final judgement uses new ones).
  - the same request must keep working at other batch sizes (the judge records batch 1, 3 and 32 requests), with
    the other sampling parameters and call forms the code supports (`do_sample=False`, `top_k > 0`, other
    temperatures, `calculate_values=False`, tensors passed directly instead of `env_obs`) and with the other
    configurations of this model (a value head with `action_level` or `chunk_level` values, two images,
    other `num_action_chunks`, other `max_prompt_length`): you may specialize a fast path, but every other case must
    still take a correct path.
  - **the earlier tasks' results must stay**: the pi0.5 code (`rlinf/models/embodiment/openpi/`) must keep giving the
    reference's numbers as fast as at e09ab89. The final judgement re-runs both earlier checks (pi05-rl: LIBERO
    `flow_sde`, batch 8, 2.9x at e09ab89; pi05-rln: ManiSkill `flow_noise`, batch 40, 1.55x at e09ab89) on your
    commit, and a formulator-side recording of several OpenVLA-OFT configurations x call variants (including the
    trainer's recompute with autograd on) against the reference; any difference there fails this task too.
  - the trainer's path (`forward` / `default_forward`) must keep giving the reference's numbers and must still be
    trainable (autograd through the whole model); do not break it to speed up the rollout.
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
  `git -C RLinf config user.name oft-agent; git -C RLinf config user.email oft-agent@localhost`).
- "The target cannot be reached within the rules" requires that every large `short`, `under-filled` or `bw-poor`
  kernel class, every large repeated chain in the latest kernel report and every large host-side span (gaps where the
  GPU idles) has had an attempt (a written, checked and timed change) or a logged reason why it cannot be changed.
- End the WORK_LOG with a summary: the final commit and switch file, the final check path and result, a breakdown of
  the gains, and the directions that did not work and why.
- Write `ws/FINAL.json`: `{"commit": "<sha on opt>", "config": "<switch file or null>", "check": "runs/<name>/check.json"}`.
