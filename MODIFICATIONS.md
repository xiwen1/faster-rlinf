# Modifications to upstream RLinf

This fork is based on RLinf commit `2348ece3df113874671478be1360ae3115b346e3` (Apache-2.0).
As required by section 4(b) of the Apache License 2.0, the files below were changed or added by faster-rlinf.
`git diff 2348ece HEAD -- <file>` shows each change; `git log 2348ece..HEAD` explains why.

| Status | File |
|---|---|
| added | `FASTER_RLINF.md` |
| added | `faster_rlinf_bench/gen_modifications.py` |
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
