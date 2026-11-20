# oft-rl formulation

Outside the sandbox: the optimizing agent never sees this directory.

- Task: P50 latency of one RL rollout step of RLinf's OpenVLA-OFT, `OpenVLAOFTForRLActionPrediction.predict_action_batch(
  env_obs=obs, do_sample=True, temperature=1.6, top_k=-1, top_p=1.0, max_new_tokens=None)` at batch 8 (batch 1
  reported), one B200, LIBERO-10 GRPO configuration.
- Source: the RLinf snapshot ec8a5f6 (`ws/targets/reference`, copied from `pi05-rln/ws/targets/reference`); candidate
  `ws/RLinf` branch `opt` cloned from `pi05-rln/ws/RLinf` at e09ab89 (the two pi0.5 tasks' result; no OFT changes),
  remote removed, identity `oft-agent`.
- Environment: `$ASSETS/venv-oft` (overlay of the shared venv with the openvla-oft stack; see
  its `OVERLAY.md`). Checkpoint `$ASSETS/weights/Openvla-oft-SFT-libero10-traj1`
  (HF `Haozhan72/Openvla-oft-SFT-libero10-traj1` @ 679a2cc1504b905e6cd954a46c80a9408a8ba3e0, curl via hf-mirror;
  shard sha256 = the HF LFS oids, verified 2026-11-10). Skill cards: `home/.claude/skills` copied from pi05-rln.
  `ws/notes/`: the two pi0.5 tasks' WORK_LOGs.
- Tools: `tools/oft-rl/{agent-run,agent-gpu,agent-wait}.sh` (pi05-rln's, retargeted: sandbox `oft-rl`, LOCAL
  `/var/tmp/goal/oft-rl`, scheduler job names `oft-*`, read-only mounts of venv, venv-oft and the OFT weights).

## What changed from the pi05-rln judge (each deliberate)

| pi05-rln | oft-rl | why |
|---|---|---|
| `Pi0RL.predict_action_batch(obs, mode="train")`, pi0.5 ManiSkill flow_noise | `OpenVLAOFTForRLActionPrediction.predict_action_batch(env_obs=obs, **train sampling)`; the rollout worker's kwargs for this model type (do_sample, temperature_train 1.6, top_k -1, top_p 1.0, max_new_tokens None) | `huggingface_worker.predict` passes the train sampling params to OFT (not `mode=`) |
| numpy observation | torch CPU tensors (uint8 256x256 agent-view + wrist image, float32 state, instructions) | OFT's code calls `.permute`/`.unsqueeze` on the images (torch only); the LIBERO env's `_wrap_obs` builds torch tensors (`to_tensor`, float64 -> float32) |
| replay `default_forward` with values | replay `default_forward(..., compute_logprobs=True, compute_entropy=True, compute_values=False, temperature=1.6, top_k=-1)` | GRPO has no value head; the trainer passes temperature/top_k for OFT; entropy included to judge more of the trainer path |
| value + noise heads from a fixed generator | none in the frozen config (value head in the extra configs, same fixed-generator rule) | libero GRPO config: `add_value_head: False` |
| gate 0: openpi get_model, Pi0RL.* | both OFT factories (`openvla_oft/__init__.py`, `openvla_oft/rlinf/__init__.py`), `predict_action_batch`, `forward`, `default_forward` | the worker and trainer call these |
| primary batch 40 (+8), cases 40x5, 8, 3 | primary batch 8 (+1), cases [8, 8, 8, 1, 3, 32, 8] | 64 LIBERO train envs / 8 GPUs (component placement `all`, one node); 1 = a single env; 32 = the ManiSkill PPO OFT per-rollout-GPU load (128 envs on 4 rollout GPUs); 3 = an odd size |
| invariants: steps, noise method... | action dim/chunks, prompt length, vocab size, action bins, unnorm key, image count, patches, value head, eval mode, sampling and trainer kwargs, param numel per dtype | |
| frozen inputs: safetensors stat + norm stats + tokenizer sha | every checkpoint file: shards by size:mtime, all other files by sha256 | the processor, tokenizer, remote config code and normalization stats are read from the checkpoint dir |
| HF_HOME read-only under assets | HF_HOME = `$HOME/.cache/huggingface` (writable) | the checkpoint has `auto_map` remote code that transformers copies into `$HF_HOME/modules` |
| env PI05_FORMULATION / PI05_BENCH* | OFT_FORMULATION / OFT_BENCH*; HF_MODULES_CACHE reserved | |

Unchanged: R C C R interleaving with SIGSTOP, MAD window-spread gate (4% reference / 6% candidate), frozen band 15%,
bar max(1.10, 1 + 2 x spread), timed outputs must equal the reference (checked only after the bar: UNAVAILABLE beats
a timed-path mismatch), numerics package byte-identical to pi05-rln's, hidden offset + fresh reference at the final.

## Steps and evidence (evidence/, prep-state/)

1. CPU smoke on the login node (`evidence/cpu-tests/`): reference builds (sdpa = the fork's bidirectional path),
   batch-2 rollout and replay logprobs equal; extra_configs runs on 3 configs x 6 calls.
2. F1 (`prep/f1.sh` -> `prep-state/f1`, 2026-11-10 11:09-11:38, host-e): probe, reference recorded twice
   (63/63 bitwise, `compare_ab.json`), thresholds, profile, baseline. The baseline host was disturbed (window spreads
   6-54%, P50 120-300 ms): discarded; so was the probe's window length (8 iterations from a disturbed 397 ms P50).
3. Window lengths set by hand from the quiet latencies: batch 8: 8 x 25, batch 1: 5 x 40 (bench.json).
4. F2 (`prep/f2.sh` -> `prep-state/f2`, host-a): quiet profile (kept: ~71.6 ms kernels / request); baseline INVALID
   (the leased GPU was not idle: 20 GB already allocated).
5. F3 (`prep/f3.sh` -> `prep-state/f3`): the first lease (host-c gpu4) ran both sessions after its launcher had
   been replaced (a killed launcher does not stop the remote job): `baseline-host-c.json`, both sessions ok, frozen.
   The second lease (host-e gpu1) ran one ok session, then INVALID (`baseline-host-e-invalid.json`).
6. Freeze (`prep/freeze.py ws prep-state/f3/freeze-in`: F1 records/thresholds + the host-c baseline), goal 1.4 / 1.2
   in bench.json before the freeze: evaluator_sha256 5000db9880fe6c1710581e571920538edeb0fc113ffe40c095bcec3c6e1862be.
7. Acceptance (`prep/make_accept.py`, runner `prep/acc.sh`, results `prep-state/acc` and `acc2`): all 12 candidates
   got the expected verdicts. Three runs were INVALID first (host-c GPUs were
   not idle) and were rerun with `prep/launch2.sh` (host-c excluded).
8. Launchers: `--min-free-mb` raised from 180000 to 181500 (`prep/launch.sh`, `prep/launch2.sh`,
   `tools/oft-rl/agent-gpu.sh`, `final/final_launch.sh`): the old filter kept leasing a GPU holding 2.2 GB of another
   job, which the GPU idle check then refused.
9. Final judgement material: `final/secret.json` (0600), `final/final_launch.sh setup|check|extra|xj-pi05-rl|xj-pi05-rln`,
   `final/extra_configs.py`, `final/extra_inner.sh`, `final/xjudge_inner.sh` (not yet exercised on a GPU).
