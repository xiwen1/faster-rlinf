# Modifications to upstream RLinf

This fork is based on RLinf commit `2348ece3df113874671478be1360ae3115b346e3` (Apache-2.0).
As required by section 4(b) of the Apache License 2.0, the files below were changed or added by faster-rlinf
(the added directory `faster_rlinf_bench/judges/` is one row).
`git diff 2348ece HEAD -- <file>` shows each change; `git log 2348ece..HEAD` explains why.

| Status | File |
|---|---|
| added | `FASTER_RLINF.md` |
| modified | `README.md` |
| added | `faster_rlinf_bench/e2e/analyze_e2e.py` |
| added | `faster_rlinf_bench/e2e/dump_tb.py` |
| added | `faster_rlinf_bench/e2e/run_e2e.sh` |
| added | `faster_rlinf_bench/e2e/run_group.sh` |
| added | `faster_rlinf_bench/e2e/run_pair.sh` |
| added | `faster_rlinf_bench/e2e/train_breakdown.py` |
| added | `faster_rlinf_bench/env_bench.py` |
| added | `faster_rlinf_bench/env_compare.py` |
| added | `faster_rlinf_bench/env_run.sh` |
| added | `faster_rlinf_bench/gen_modifications.py` |
| added | `faster_rlinf_bench/judges/` (99 files: the four frozen rollout judges, see its README.md) |
| added | `faster_rlinf_bench/patch_check.py` |
| added | `faster_rlinf_bench/repeat_check.py` |
| added | `faster_rlinf_bench/repeat_check.sh` |
| added | `faster_rlinf_bench/u8_normalize_check.py` |
| added | `faster_rlinf_bench/workloads/gr00t/model_cfg.json` |
| added | `faster_rlinf_bench/workloads/gr00t/workload.py` |
| added | `faster_rlinf_bench/workloads/oft/model_cfg.json` |
| added | `faster_rlinf_bench/workloads/oft/workload.py` |
| added | `faster_rlinf_bench/workloads/pi05/model_cfg.json` |
| added | `faster_rlinf_bench/workloads/pi05/workload.py` |
| modified | `rlinf/envs/sim/libero/libero_env.py` |
| modified | `rlinf/envs/sim/libero/venv.py` |
| modified | `rlinf/envs/wrappers/record_video.py` |
| modified | `rlinf/hybrid_engines/fsdp/strategy/fsdp.py` |
| modified | `rlinf/hybrid_engines/weight_syncer/patch_syncer.py` |
| added | `rlinf/models/embodiment/gr00t/gr00t_n1d5/fast_transform.py` |
| modified | `rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py` |
| added | `rlinf/models/embodiment/gr00t/gr00t_n1d5/rollout_graph.py` |
| modified | `rlinf/models/embodiment/openpi/modules/gemma.py` |
| modified | `rlinf/models/embodiment/openpi/pi0.py` |
| modified | `rlinf/models/embodiment/openpi/tasks/rl.py` |
| modified | `rlinf/models/embodiment/openpi/transforms/pipeline.py` |
| added | `rlinf/models/embodiment/openpi/transforms/resize.py` |
| added | `rlinf/models/embodiment/openvla_oft/rlinf/fast_inputs.py` |
| added | `rlinf/models/embodiment/openvla_oft/rlinf/fast_llm.py` |
| modified | `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` |
| added | `rlinf/models/embodiment/openvla_oft/rlinf/rollout_graph.py` |
| added | `rlinf/models/embodiment/repeat_rng.py` |
| modified | `rlinf/workers/rollout/hf/huggingface_worker.py` |
