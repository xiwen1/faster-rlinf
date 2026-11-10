# Copyright 2025 The RLinf Authors.
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

"""CUDA-graph replay of the device part of ``OpenVLAOFTForRLActionPrediction.predict_action_batch``.

The device step (placeholder/stop tokens, embeddings, vision backbone + projector, the fused language-model forward of
``fast_llm``, logits masking, temperature / top-k, log-softmax, ``torch.multinomial`` or argmax, log-probabilities and
values) is the reference's op sequence with the same shapes and dtypes; the only rewrites remove host round trips:
``torch.ones(...).to(device)`` H2D copies become device-side ``torch.ones``/``torch.full`` with the same values and
dtypes, slice assignments of a Python scalar become ``fill_`` (same values), and the ``torch.all`` asserts are checked
on the host copies of the same tensors by the caller. The CUDA RNG is drawn inside the graph by the same
``torch.multinomial`` call (graph-safe philox: the replay draws exactly what the eager call draws and advances the
generator by the same offset).

One graph per input signature (batch, prompt length, image shape, sampling parameters, value flag, whether the
attention mask has padding), captured at the first request of the signature after an eager warmup; the CPU / CUDA /
Python / numpy RNG states are saved before and restored after warmup + capture, so building a graph consumes no
random number of the caller. A failed warmup or capture leaves the process usable (pool and generator restored) and
that signature uses the eager path from then on. ``RLINF_OFT_CUDA_GRAPHS=0`` disables the graphs.
"""

import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.vla.constants import STOP_INDEX
from transformers.generation import TopKLogitsWarper

from rlinf.utils.utils import compute_logprobs_from_logits

from . import fast_llm

logger = logging.getLogger(__name__)

# Graph invalidation: a graph reads the parameters / buffers at the addresses seen at capture. Every
# (re)registration of a parameter, buffer or submodule anywhere (attribute assignment, load_state_dict(assign=True),
# module replacement) bumps this epoch; ``.data`` swaps and ``.to()`` change data_ptr. The signature is the
# data_ptr tuple of the model's parameters and buffers, whose list is re-collected only when the epoch moved.
_EPOCH = [0]


def _bump_epoch(*_args):
    _EPOCH[0] += 1


for _register in (
    torch.nn.modules.module.register_module_parameter_registration_hook,
    torch.nn.modules.module.register_module_buffer_registration_hook,
    torch.nn.modules.module.register_module_module_registration_hook,
):
    _register(_bump_epoch)


class GraphCaptureError(RuntimeError):
    pass


def device_step(model, input_ids, attention_mask, pixel_values, sampling, calc_values, all_attended):
    """The device part of predict_action_batch from the model inputs to (idxs, chunk_logprobs, chunk_values)."""
    do_sample, temperature, top_k = sampling
    n_act = model.action_dim * model.num_action_chunks
    B = input_ids.shape[0]
    dev = input_ids.device
    n_prompt_tokens = input_ids.shape[-1] - 1
    n_patches = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()

    # _prepare_input_for_action_prediction (device-side constants, same values and dtypes)
    ids = torch.cat([input_ids, torch.ones((B, n_act), dtype=input_ids.dtype, device=dev)], dim=-1)
    ids = torch.cat([ids, torch.full((B, 1), STOP_INDEX, dtype=input_ids.dtype, device=dev)], dim=-1)
    am = torch.cat(
        [
            attention_mask,
            torch.ones((B, ids.shape[-1] - attention_mask.shape[-1]), dtype=attention_mask.dtype, device=dev),
        ],
        dim=-1,
    )

    # _build_embedding
    ids = ids[:, :-1]
    am = am[:, :-1]
    all_actions_mask = torch.zeros_like(ids, dtype=torch.bool)
    all_actions_mask[:, -n_act:].fill_(True)
    emb = model.get_input_embeddings()(ids)
    emb = emb * (~all_actions_mask.unsqueeze(-1))
    proj = model._process_vision_features(pixel_values, None, use_film=False)
    proj = proj.reshape(emb.shape[0], -1, *proj.shape[2:])
    mm_emb, mm_mask = model._build_multimodal_attention(emb, proj, am)
    pos = mm_mask.cumsum(dim=1) - 1

    all_logits, last_hidden = fast_llm.lm_forward(
        model.language_model, mm_emb, pos, None if all_attended else (mm_mask != 0)
    )
    logits = all_logits[:, n_patches + n_prompt_tokens : n_patches + n_prompt_tokens + n_act, :]
    last_hidden = last_hidden[:, -n_act - 1 : -1]
    lo = model.vocab_size - model.config.n_action_bins
    logits[..., :lo].fill_(-torch.inf)
    logits[..., model.vocab_size :].fill_(-torch.inf)

    if do_sample:
        processed = logits / temperature
        k = min(top_k, processed.size(-1))
        if k > 0:
            processed = TopKLogitsWarper(k)(None, processed)
        logprob = F.log_softmax(processed, dim=-1)
        probs = torch.exp(logprob)
        flat = probs.view(-1, logprob.shape[-1])
        sample = torch.multinomial(flat, num_samples=1, replacement=True)
        idxs = sample.view(logprob.shape[0], logprob.shape[1])
    else:
        processed = logits
        idxs = processed.argmax(dim=-1)

    processed[..., :lo].fill_(-torch.inf)
    processed[..., model.vocab_size :].fill_(-torch.inf)
    chunk_logprobs = compute_logprobs_from_logits(logits=processed, target=idxs)
    if hasattr(model, "value_head") and calc_values:
        chunk_values = model.value_head(last_hidden[:, -n_act])
    else:
        chunk_values = torch.zeros_like(chunk_logprobs[..., :1])
    return idxs, chunk_logprobs, chunk_values


def _recover_failed_capture(device, pool):
    """End pool routing and take the CUDA generator out of capture mode after a capture_end that raised."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        torch._C._cuda_endAllocateToPool(index, pool)
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001
        pass
    reset = torch.cuda.CUDAGraph()
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        reset.capture_begin(pool=torch.cuda.graph_pool_handle())
        reset.capture_end()
    del reset
    torch.cuda.synchronize(device)


class RolloutGraphs:
    MAX_BANKS = 8

    def __init__(self):
        self.banks = {}
        self.failed = set()
        self.pool = None
        self._tensors = None
        self._epoch = None

    def signature(self, model):
        if self._tensors is None or self._epoch != _EPOCH[0]:
            self._tensors = list(model.parameters()) + list(model.buffers())
            self._epoch = _EPOCH[0]
        return tuple([t.data_ptr() for t in self._tensors])

    def _capture(self, device, fn):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        error = None
        out = None
        with torch.cuda.stream(stream):
            try:
                graph.capture_begin(pool=self.pool)
            except BaseException as e:  # noqa: BLE001
                raise GraphCaptureError(f"capture_begin failed: {e!r}") from e
            try:
                out = fn()
            except BaseException as e:  # noqa: BLE001
                error = e
            ended = False
            try:
                graph.capture_end()
                ended = True
            except BaseException as e:  # noqa: BLE001
                error = error if error is not None else e
            if not ended:
                _recover_failed_capture(device, self.pool)
        if error is not None:
            del graph, out
            raise GraphCaptureError(f"graph capture failed: {error!r}") from error
        return graph, out

    def _build(self, model, key, inputs, sampling, calc_values, all_attended, sig):
        device = inputs[0].device
        static = [t.clone() for t in inputs]
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device)
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                out = device_step(model, *static, sampling, calc_values, all_attended)
                del out
            torch.cuda.current_stream(device).wait_stream(side)
            graph, out = self._capture(
                device, lambda: device_step(model, *static, sampling, calc_values, all_attended)
            )
            torch.cuda.synchronize(device)
        except Exception as e:  # noqa: BLE001
            self.pool = None
            torch.cuda.synchronize(device)
            logger.warning("openvla-oft: CUDA-graph rollout disabled for this input signature (eager path): %s", e)
            return None
        finally:
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state, device)
            random.setstate(py_state)
            np.random.set_state(np_state)
        while len(self.banks) >= self.MAX_BANKS:
            self.banks.pop(next(iter(self.banks)))
        bank = {"static": static, "graph": graph, "out": out, "sig": sig}
        self.banks[key] = bank
        return bank

    def run(self, model, input_ids, attention_mask, pixel_values, sampling, calc_values, all_attended):
        """(idxs, chunk_logprobs, chunk_values) of the device step by graph replay, or None (use the eager path)."""
        inputs = (input_ids, attention_mask, pixel_values)
        key = (
            tuple((tuple(t.shape), t.dtype, str(t.device), t.stride()) for t in inputs),
            tuple(sampling),
            bool(calc_values),
            bool(all_attended),
            bool(model.training),
        )
        if key in self.failed:
            return None
        sig = self.signature(model)
        bank = self.banks.get(key)
        if bank is None or bank["sig"] != sig:
            bank = self._build(model, key, inputs, sampling, calc_values, all_attended, sig)
            if bank is None:
                self.failed.add(key)
                return None
        else:
            for s, t in zip(bank["static"], inputs):
                s.copy_(t)
        bank["graph"].replay()
        return tuple(t.clone() for t in bank["out"])
