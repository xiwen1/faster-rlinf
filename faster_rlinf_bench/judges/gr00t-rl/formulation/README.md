# gr00t-rl formulation

Outside the sandbox: the optimizing agent never sees this directory.

- Task: P50 latency of one RL rollout step of RLinf's GR00T N1.5,
  `GR00T_N1_5_ForRLActionPrediction.predict_action_batch(env_obs=obs, mode="train")` (flow_sde, noise level 0.5,
  4 denoising steps, value head) at batch 8 (batch 1 reported), one B200, LIBERO-Spatial PPO configuration.
- Source: the RLinf snapshot ec8a5f6 (`ws/targets/reference`, copied from `oft-rl/ws/targets/reference`, same commit
  as pi05-rln's); candidate `ws/RLinf` branch `opt` cloned from `pi05-rln/ws/RLinf` at e09ab89 (the two pi0.5 tasks'
  result; no GR00T changes: `rlinf/models/embodiment/gr00t` identical to the reference), remote removed, identity
  `gr00t-agent`.
- Environment: `$ASSETS/venv-gr00t` (overlay of the shared venv with the Isaac-GR00T
  n1.5-release stack; see its `OVERLAY.md`). Checkpoint `$ASSETS/weights/RLinf-Gr00t-SFT-Spatial`
  (HF `RLinf/RLinf-Gr00t-SFT-Spatial` @ 73f710e70e7d571f8d828e51e0a428f5a1e0ac22 via hf-mirror; config, index and
  experiment_cfg/metadata.json; no scheduler.pt / trainer_state.json; shard sha256 = the HF LFS oids, checked by
  `prep/pdl_all.sh`, see `evidence/download.log`). Skill cards: `home/.claude/skills` copied from oft-rl (= pi05-rln's).
  `ws/notes/`: the two pi0.5 tasks' WORK_LOGs.
- Tools: `tools/gr00t-rl/{agent-run,agent-gpu,agent-wait}.sh` (oft-rl's, retargeted: sandbox `gr00t-rl`, LOCAL
  `/var/tmp/goal/gr00t-rl`, scheduler job names `g00-*`, read-only mounts of venv, venv-gr00t and the GR00T weights).

## What changed from the oft-rl judge (each deliberate)

| oft-rl | gr00t-rl | why |
|---|---|---|
| `OpenVLAOFTForRLActionPrediction.predict_action_batch(env_obs=obs, do_sample, temperature 1.6, ...)` | `GR00T_N1_5_ForRLActionPrediction.predict_action_batch(env_obs=obs, mode="train")` | `huggingface_worker.predict` passes only `mode` for GR00T |
| replay `default_forward(compute_logprobs, compute_entropy=True, compute_values=False, temperature, top_k)` | replay `default_forward(compute_logprobs=True, compute_entropy=False, compute_values=True, use_cache=False, prev_logprobs=<rollout's>)` | `embodied_fsdp_actor_worker.train_micro_batch` for GR00T: GAE -> values on, entropy_bonus 0, `prev_logprobs` kwarg; GR00T returns `entropy: None` |
| RNG: CUDA (multinomial) | RNG: CUDA (initial noise + per-step noise) and Python `random` (stochastic step index) | flow_sde in `FlowMatchingActionHeadForRLActionPrediction.get_rl_action` |
| no value head | value head (re-initialized by RLinf after loading) set from a fixed generator by parameter name (`action_head.value_head.*`, also `action_head.reinflow_explore_noise_net.*` in the extra configs) | libero_spatial_ppo_gr00t: `add_value_head: True` |
| gate 0: OFT factories, predict_action_batch, forward, default_forward | `gr00t/__init__.py:get_model`, `gr00t/gr00t_n1d5/__init__.py:get_model`, `GR00T_N1_5_ForRLActionPrediction.{predict_action_batch,forward,default_forward}` | what the worker and trainer call |
| primary batch 8, cases [8, 8, 8, 1, 3, 32, 8] | primary batch 8 (+1), cases [8, 8, 8, 1, 3, 16, 32, 8] | 64 LIBERO train envs / 8 GPUs (placement `all`); 16 = libero_goal_async / isaaclab per-rollout-GPU load (64 envs on 4 GPUs); 32 = a large batch; 1 = single env; 3 = odd |
| invariants: OFT action/prompt/vocab | action chunks/horizon/dims, denoising steps, timestep buckets, prompt padding, image count, embodiment, noise method/level, joint_logprob, ignore_last, value head and mode, backbone layers, eval mode (model and modality transform), kwargs, param numel per dtype | |
| frozen inputs: checkpoint files | checkpoint files (shards by size:mtime, everything else incl. experiment_cfg/metadata.json by sha256) + the Eagle 2.5 config/processor/tokenizer files inside venv-gr00t's `gr00t` package (sha256) | the backbone config, processor and tokenizer are read from the package, the normalization statistics from the checkpoint |
| python venv-oft | python venv-gr00t | |
| env OFT_FORMULATION / OFT_BENCH* | GR00T_FORMULATION / GR00T_BENCH* | |

Unchanged: R C C R interleaving with SIGSTOP, MAD window-spread gate (4% reference / 6% candidate), frozen band 15%,
bar max(1.10, 1 + 2 x spread), timed outputs must equal the reference (checked only after the bar), numerics package
byte-identical to pi05-rln's / oft-rl's, hidden offset + fresh reference at the final, HF_HOME writable in the sandbox
home (the Eagle remote code is copied to `$HF_HOME/modules`).

## Steps and evidence (evidence/, prep-state/)

0. Downloads (`prep/download.sh`, `prep/pdl.sh`, `prep/pdl_all.sh`; `evidence/download.log`): the single-connection
   download stalled at ~260 KB/s; 16 parallel range requests; both shard sha256 = the HF LFS oids.
1. CPU smoke on the login node (`evidence/cpu-tests/`; Eagle attention switched to SDPA, no flash-attention CPU kernel):
   reference builds, batch-2 rollout + replay (replay logprobs = rollout prev_logprobs at the stochastic step,
   bitwise), repeat from the same RNG state bitwise equal; extra_configs on 12 configs x 7 calls.
2. F1 (`prep/f1.sh` -> `prep-state/f1`, 2026-11-10 11:49-12:08, host-e gpu2): probe, reference recorded twice
   (120/120 bitwise, `compare_ab.json`), thresholds, profile, baseline. The host was disturbed (NUMA node ~39% idle;
   batch-8 window P50s 150-625 ms, spreads 12-24%): baseline and the probe's window length (8) discarded;
   `prep/after_f1.sh` stopped there as designed ("F1 not clean").
3. Window lengths set by hand (`prep/set_windows.py`): batch 8: 8 x 25, batch 1: 5 x 30 (bench.json).
4. F2a (`prep/f2.sh` via `prep/launch2.sh`, host-restricted -> `prep-state/f2a`, 2026-11-11 01:51-02:04, host-b
   gpu5, NUMA node 99.6% idle): profile (31.5 ms of kernels / request) and baseline (batch 8 P50 100-108 ms, window
   spreads 0.2-0.9%; null speedups 0.995 / 0.999): both sessions ok, frozen. Bands unchanged (4% / 6% / 15%). A
   second-host run (f2b) was queued and stopped before it got a GPU (`evidence/f2b.launch.log`).
5. Freeze (`prep/freeze2.sh f2a 2.0 1.5`: goal into bench.json, `prep-state/freeze-in` = F1 records/thresholds/
   comparison + F2a baseline, `prep/freeze.py`): evaluator_sha256
   03c948b789f3616fefb6b8893aa52d2d4b78051b100b4de171f245478d36fdee; `ws/task/baseline.md` rewritten by hand afterwards
   (not an evaluator file).
6. Acceptance (`prep/make_accept.py` -> `prep-state/accept`; runner `prep/acc.sh` via `prep/launch2.sh`, no PREP:
   `prep-state/accf` = 5 full checks on host-b, `prep-state/accq` = 12 quick checks + a at the hidden
   acceptance_offset): all 18 verdicts as expected (`evidence/acceptance_table.md`, `prep/acc_table.py`).
7. Launchers: `--min-free-mb` raised from 180000 to 181500 (`prep/launch.sh`, `prep/launch2.sh`,
   `tools/gr00t-rl/agent-gpu.sh`, `final/final_launch.sh`), as in oft-rl.
8. Installed: `ws/GOAL.md` (TARGET 2.0, MINIMUM 1.5; staging copy in `staging/GOAL.md`); `gr00t-rl/.prep` moved to
   `prep-state/` (nothing of the formulation is left inside the sandbox).
9. Final judgement material: `final/secret.json` (0600), `final/final_launch.sh setup|check|extra|xj-pi05-rl|
   xj-pi05-rln|xj-oft-rl`, `final/extra_configs.py`, `final/extra_inner.sh`, `final/xjudge_inner.sh` (not yet
   exercised on a GPU).
