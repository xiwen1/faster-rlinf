# Copyright 2026 faster-rlinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""faster-rlinf: reproduce the RNG consumption of a repeated rollout call without running it.

When an environment is done (in LIBERO: once a task succeeds, every later chunk of the
episode), the env worker asks the rollout worker for a "terminal" forward pass on the very
observation the rollout just computed actions for, only to read its state value
(``huggingface_worker._publish_env_part``). The value of these models depends on the
observation only (VLM / backbone features), not on the sampling noise, so the repeated call
returns the value the first call already returned; the rest of its outputs are discarded.
What the repeated call does change is the random state: one ``random.randint(0, hi)`` draw
(flow_sde step choice) and a fixed advance of the CUDA philox offset (noise / multinomial
inside the replayed CUDA graphs).

``RepeatCallRNG`` records, for every graph bank and stochastic-step choice, by how much a
fully graph-replayed call advanced the CUDA offset, and replays exactly that consumption
(``advance_as_repeat``): same Python draw, same CUDA offset afterwards. If anything needed is
unknown it restores the Python RNG and returns False, and the caller runs the real call.
"""

from __future__ import annotations

import random
from typing import Optional

import torch


def _generator(device: torch.device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[index]


class RepeatCallRNG:
    """Per-model helper; see the module docstring."""

    def __init__(self) -> None:
        self.plan = None
        self._device: Optional[torch.device] = None
        self._offset0: Optional[int] = None

    def begin(self, device) -> None:
        """Call at the top of every ``predict_action_batch`` (drops any earlier plan)."""
        self.plan = None
        self._offset0 = None
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        self._device = device
        self._offset0 = _generator(device).get_offset()

    def commit(self, table: dict, chosen, hi: Optional[int]) -> None:
        """The call ran entirely by graph replay; ``table`` is per graph bank.

        ``chosen`` is the stochastic step drawn with ``random.randint(0, hi)`` (None when the
        call draws nothing, ``hi`` None then)."""
        if self._offset0 is None:
            return
        inc = _generator(self._device).get_offset() - self._offset0
        if table.get(chosen, inc) != inc:
            table["inconsistent"] = True
        table[chosen] = inc
        self.plan = (table, hi)

    def advance_as_repeat(self) -> bool:
        """Consume the random numbers a repeat of the last call would consume; False if unsure."""
        plan, self.plan = self.plan, None
        if plan is None:
            return False
        table, hi = plan
        if table.get("inconsistent"):
            return False
        state = random.getstate()
        chosen = random.randint(0, hi) if hi is not None else None
        inc = table.get(chosen)
        if inc is None:
            random.setstate(state)
            return False
        gen = _generator(self._device)
        gen.set_offset(gen.get_offset() + inc)
        return True


def repeat_rng(model) -> RepeatCallRNG:
    state = model.__dict__.get("_faster_rlinf_repeat_rng")
    if state is None:
        state = RepeatCallRNG()
        model.__dict__["_faster_rlinf_repeat_rng"] = state
    return state
