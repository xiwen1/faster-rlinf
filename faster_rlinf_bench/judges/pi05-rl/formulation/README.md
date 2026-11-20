# pi05-rl formulation

Outside the sandbox: the optimizing agent never sees this directory.

- Task: P50 latency of one RL rollout step, RLinf `Pi0RL.predict_action_batch(obs, mode="train")` at batch 8 (batch 40
  reported), one B200. Decisions of the project owner
  (batch 8, flow_sde 5 steps, N0 floor including the trainer's recompute, target 2.0x, minimum 1.5x, optimizing
  agent = a Claude subagent of the formulating session).
- Source: the same ApxInf-free RLinf snapshot as pi05-b1 (commit ec8a5f66c68fa721cf1afd28ca37f488a8aa91bf, cloned from
  `pi05-b1/ws/targets/reference`, no remote): `ws/targets/reference`, candidate `ws/RLinf` branch `opt`. The candidate
  starts from the reference, not from the pi05-b1 agent's `Pi0Eval` commits.
- Environment, checkpoint, tokenizer, skill cards: as pi05-b1 (`home/` copied from `pi05-b1/home`, without the pi05-b1
  agent's `scripts/`). Tools: `tools/pi05-rl/{agent-run,agent-gpu,agent-wait}.sh` (pi05-b1's with the task name
  changed); `tools/sandbox.sh`, the GPU idle check, `tools/bin/bwrap` shared.

## What changed from the pi05-b1 judge (each deliberate)

| pi05-b1 | pi05-rl | why |
|---|---|---|
| `Pi0Eval`, batch 1, actions only | `Pi0RL` `mode="train"`, batch 8 (+40); all rollout outputs and the trainer's recompute (`default_forward`, forward only, no_grad) recorded | the RL rollout returns training data: `prev_logprobs` feed PPO directly (`recompute_logprobs: False`) |
| timed request ends at `actions.cpu()` | timed request ends with `torch.cuda.synchronize()` | outputs other than the actions stay on the GPU; deferring their work past the timer must not count as speed |
| reference, candidate, reference: three sequential processes; INVALID when the two references differ > 3% or one is > 15% off the frozen reference | two reference and two candidate processes (launched R C C R) live, taking turns window by window, the idle ones SIGSTOPped; per-tree P50 = mean of its two processes; INVALID when the reference batch-8 P50 is > 15% off the frozen reference or a process's window P50s have a spread (median absolute deviation / median) > 3% (candidate 5%); bar max(1.10, 1 + 2 x spread) | pi05-b1: 49 of 50 full checks INVALID from reference drift on shared hosts; the first interleaved baseline used the interquartile spread, which two disturbed windows of eight (+38%) push past 10%; interleaving puts both under the same host conditions; the frozen band stays so that a contended host (which slows the launch-bound reference more than a graph-captured candidate) cannot inflate the ratio |
| pinned to all CPUs of the GPU's NUMA node | pinned to the 8 least busy CPUs of the node (sampled 2 s before the run), `nice -n -5` | those CPUs are not reserved for the judge |
| tail band (mean vs P50) | none (P50 over interleaved windows) | spikes are outside the P50 and hit both trees alike |
| digests of timed actions | digests of every output of every timed window, compared window by window | the timed path must be the judged path for all outputs |
| value head absent | value head (not in the checkpoint) set from a fixed generator by parameter name after the build | independent of how much global RNG the candidate's construction consumes |
| gate 0: get_model, Pi0Eval.predict_action_batch | get_model, Pi0RL.predict_action_batch, Pi0RL.forward, Pi0RL.default_forward | the rollout worker and the trainer call these |
| cases: 6 requests at batch 1 (+ one at 4) | 7 rollout requests: batch 8 x5, 40, 3 | per-GPU loads of LIBERO and ManiSkill PPO, and an odd size |

Formulator-side extra at the final judgement (not in the agent's judge): the candidate is also recorded against the
reference on other supported configurations (3 steps, flow_noise + joint_logprob, caller-supplied `rng`), to check
the GOAL rule that every other case still takes a correct path.

## Steps and evidence (evidence/)

1. F1 (`.prep/f1`, launch log `evidence/f1.launch.log`): reference recorded twice in fresh processes.
2. F2 baseline: reference against itself in the interleaved scheme, two sessions.
3. F3 profile of the reference at batch 8.
4. Freeze `prep/freeze.py` -> `ws/task/{records/ref,thresholds.json,timing_ref.json,formulation.json}`; `numerics selftest`.
5. Acceptance `prep/make_accept.py` (candidates in `pi05-rl/.prep/accept`, removed before the agent starts), runner
   `pi05-rl/.prep/acc.sh`.
