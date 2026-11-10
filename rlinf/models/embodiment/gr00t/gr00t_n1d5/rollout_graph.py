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

"""CUDA-graph fast path of the GR00T N1.5 rollout device step.

``GR00T_N1_5_ForRLActionPrediction._get_rl_action`` (Eagle backbone, action
head ``get_rl_action``) launches ~2,900 kernels and syncs the host several
times per request (boolean-mask indexing, the flash-attention unpadding of the
Qwen3 layers, host-to-device copies of per-step constants). This module replays
the same computation from CUDA graphs:

* graph "prefix": SigLIP + Qwen3 backbone, ``process_backbone_output``, state
  encoder, the initial noise draw and the value head;
* graph "loop[k]": the denoising loop, one graph per stochastic step ``k`` drawn
  by ``random.randint`` (``None``: every step deterministic, mode "eval").

Every kernel is the eager one on tensors of the eager shapes (same modules,
same ops, same dtypes), and the noise is drawn from the global CUDA generator in
the same order (graph-safe philox offsets), so the outputs are bitwise the eager
ones. The data-dependent parts are moved to the host, where the inputs already
are: the image-token rows (``input_ids == image_token_index``), the "any padding"
test and the unpadding of the flash attention. For the latter the varlen kernel
runs on the padded rows directly, every sample split into three sequences
(left padding, valid tokens, right padding): the valid tokens form the same
sequence as in the unpadded eager call (each thread block of the varlen kernel
reads only its own sequence), and the padding rows, which eager ``pad_input``
fills with zeros, are zero-filled. The Qwen3 ``lm_head`` logits, which nothing
reads, are not computed.

Only verified configurations take it; everything else runs the eager path.
``RLINF_GR00T_CUDA_GRAPHS=0`` disables it.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("RLINF_GR00T_CUDA_GRAPHS", "1") != "0"


class _GraphCaptureError(RuntimeError):
    """A CUDA-graph capture failed; the process state was restored."""


def _recover_failed_capture(device: torch.device, pool) -> None:
    """Undo what a capture whose ``capture_end`` raised leaves behind (allocator
    routing to the pool, CUDA generators in capture mode)."""
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


def _value_token_indices(mode: str, length: int) -> Optional[list[int]]:
    """Positions selected by ``get_value``'s ``prefix_mask`` (None: unknown mode)."""
    if mode == "mean_token":
        return list(range(length))
    if mode == "last_token":
        return [length - 1]
    if mode == "first_token":
        return [0]
    return None


class RolloutGraphs:
    """Graph banks of one model, keyed by the input signature."""

    MAX_BANKS = 8

    def __init__(self, model):
        self.model = model
        self.banks: dict = {}
        self.failed: set = set()
        self.vision_banks: dict = {}
        self.vision_failed: set = set()
        self.ticket = None
        self.pool = None

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def supported(self, mode) -> bool:
        m = self.model
        if not _ENABLED or not torch.cuda.is_available():
            return False
        if m.training or torch.is_grad_enabled():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            device = m.device
        except Exception:  # noqa: BLE001
            return False
        if device.type != "cuda":
            return False
        head = m.action_head
        cfg = head.rl_config
        if cfg.joint_logprob or cfg.noise_anneal:
            return False
        if mode == "train" and cfg.noise_method != "flow_sde":
            return False
        if cfg.add_value_head and _value_token_indices(cfg.value_vlm_mode, 2) is None:
            return False
        if head.training or not head.config.add_pos_embed:
            return False
        backbone = m.backbone
        if backbone.training:
            return False
        eagle = backbone.eagle_model
        if getattr(eagle, "use_llm_lora", False) or eagle.training:
            return False
        lm = eagle.language_model
        if not hasattr(lm, "model") or not hasattr(lm.model, "layers"):
            return False
        lcfg = lm.config
        if lcfg._attn_implementation != "flash_attention_2" or lcfg.use_cache:
            return False
        for layer in lm.model.layers[: lcfg.num_hidden_layers]:
            if layer.self_attn.sliding_window is not None:
                return False
        return True

    # ------------------------------------------------------------------
    # host-side signature of a request
    # ------------------------------------------------------------------
    def _host_inputs(self, normalized_input: dict[str, Any]):
        """CPU tensors of the request plus the host-derived index data, or None
        when the request has a layout this path does not cover."""
        m = self.model
        need = (
            "eagle_input_ids",
            "eagle_attention_mask",
            "eagle_pixel_values",
            "state",
            "embodiment_id",
        )
        for k in need:
            v = normalized_input.get(k)
            if not torch.is_tensor(v) or v.device.type != "cpu":
                return None
        ids = normalized_input["eagle_input_ids"]
        mask = normalized_input["eagle_attention_mask"]
        pix = normalized_input["eagle_pixel_values"]
        state = normalized_input["state"]
        emb = normalized_input["embodiment_id"]
        dtype = m.action_head.dtype
        if pix.dtype != dtype or state.dtype != dtype:
            return None
        if ids.dim() != 2 or mask.shape != ids.shape or emb.dim() != 1:
            return None
        B, N = ids.shape
        if state.shape[0] != B or emb.shape[0] != B or ids.dtype != torch.long:
            return None
        eagle = m.backbone.eagle_model
        selected = ids.reshape(-1) == eagle.image_token_index
        img_idx = torch.nonzero(selected, as_tuple=False).flatten()
        if img_idx.numel() != pix.shape[0] * eagle.num_image_token:
            return None
        # ``0.0 in attention_mask`` of Qwen3Model._update_causal_mask
        has_pad = bool((mask == 0).any())
        cu = pad_rows = None
        if has_pad:
            if bool(((mask != 0) & (mask != 1)).any()):
                return None
            valid = mask != 0
            counts = valid.sum(dim=1)
            first = torch.where(
                counts > 0, valid.to(torch.int8).argmax(dim=1), torch.zeros_like(counts)
            )
            last = first + counts
            pos = torch.arange(N).unsqueeze(0)
            contiguous = ((pos >= first[:, None]) & (pos < last[:, None])) == valid
            if not bool(contiguous.all()):
                return None
            seg = torch.stack([first, counts, N - last], dim=1).reshape(-1)
            cu = torch.nn.functional.pad(
                torch.cumsum(seg, dim=0, dtype=torch.int32), (1, 0)
            )
            pad_rows = ~valid.reshape(-1)
        return {
            "B": B,
            "N": N,
            "ids": ids,
            "pix": pix,
            "state": state,
            "emb": emb,
            "img_idx": img_idx,
            "has_pad": has_pad,
            "cu": cu,
            "pad_rows": pad_rows,
        }

    def _key(self, h, chosen_variants) -> tuple:
        m = self.model
        cfg = m.action_head.rl_config
        return (
            h["B"],
            h["N"],
            tuple(h["pix"].shape),
            tuple(h["state"].shape),
            h["img_idx"].numel(),
            h["has_pad"],
            tuple(chosen_variants),
            m.action_head.num_inference_timesteps,
            m.action_head.action_chunk,
            m.action_head.valid_action_dim,
            bool(cfg.add_value_head),
            str(cfg.value_vlm_mode),
            float(cfg.noise_level),
        )

    # ------------------------------------------------------------------
    # device computation (capturable mirror of the eager code)
    # ------------------------------------------------------------------
    def _attention(self, attn, hidden_states, position_embeddings, st):
        """Qwen3Attention.forward + transformers' flash_attention_forward /
        _flash_attention_forward, with the unpadding precomputed on the host."""
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from transformers import modeling_flash_attention_utils as fa_utils
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        flash_kwargs = {"deterministic": fa_utils.deterministic_g} if fa_utils.flash_241 else {}

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query = query_states.transpose(1, 2)
        key = key_states.transpose(1, 2)
        value = value_states.transpose(1, 2)
        if st["has_pad"]:
            B, N, _, head_dim = key.shape
            q = query.reshape(B * N, -1, head_dim)
            k = key.reshape(B * N, key.shape[2], head_dim)
            v = value.reshape(B * N, value.shape[2], head_dim)
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=st["cu"],
                cu_seqlens_k=st["cu"],
                max_seqlen_q=N,
                max_seqlen_k=N,
                dropout_p=0.0,
                softmax_scale=attn.scaling,
                causal=True,
                **flash_kwargs,
            )
            out.masked_fill_(st["pad_rows"][:, None, None], 0)
            attn_output = out.view(B, N, out.shape[-2], out.shape[-1])
        else:
            attn_output = flash_attn_func(
                query,
                key,
                value,
                0.0,
                softmax_scale=attn.scaling,
                causal=True,
                **flash_kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output)

    def _language_model(self, inputs_embeds, st):
        """Qwen3Model.forward (no cache, hidden states of every layer)."""
        lm = self.model.backbone.eagle_model.language_model.model
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = lm.rotary_emb(hidden_states, position_ids)
        all_hidden_states = []
        for layer in lm.layers[: lm.config.num_hidden_layers]:
            all_hidden_states.append(hidden_states)
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = self._attention(layer.self_attn, hidden_states, position_embeddings, st)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        hidden_states = lm.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        return all_hidden_states

    def _device_prefix(self, st):
        m = self.model
        head = m.action_head
        backbone = m.backbone
        eagle = backbone.eagle_model
        # Eagle2_5_VLForConditionalGeneration.forward
        input_ids = st["ids"]
        input_embeds = eagle.language_model.get_input_embeddings()(input_ids)
        vit_embeds = st["vit"]  # the vision graph's output (extract_feature)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        img_idx = st["img_idx"]
        input_embeds[img_idx] = input_embeds[img_idx] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)
        hidden_states = self._language_model(input_embeds, st)
        # EagleBackbone.forward_eagle
        eagle_features = hidden_states[backbone.select_layer]
        eagle_features = backbone.eagle_linear(eagle_features)
        # get_rl_action
        vl_embs = head.vlln(eagle_features)
        vl_embs = head.vl_self_attention(vl_embs)
        state_features = head.state_encoder(st["state"], st["emb"])
        x_t = torch.randn(
            size=(B, head.config.action_horizon, head.config.action_dim),
            dtype=vl_embs.dtype,
            device=vl_embs.device,
        )
        if head.rl_config.add_value_head:
            values = self._get_value(vl_embs, state_features, st)[:, None]
        else:
            values = torch.zeros((B, 1), device=vl_embs.device, dtype=vl_embs.dtype)
        return {
            "vl_embs": vl_embs,
            "state_features": state_features,
            "x_t": x_t,
            "values": values,
        }

    def _get_value(self, vl_embs, state_features, st):
        """FlowMatchingActionHeadForRLActionPrediction.get_value with the
        boolean ``prefix_mask`` as its (host-known) token indices."""
        head = self.model.action_head
        bsize = vl_embs.shape[0]
        vl_embs_value = vl_embs[:, st["value_idx"], :]
        vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
        state_features_value = state_features.reshape(bsize, -1)
        if head.rl_config.use_vlm_value:
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        return head.value_head(value_embs)[:, 0]

    def _sample_mean_var(self, vl_embs, idx, x_t, embodiment_id, state_features, mode, st):
        """sample_mean_var_val with the step constants created before capture."""
        head = self.model.action_head
        device = vl_embs.device
        denoise_steps = head.num_inference_timesteps
        idx_t = st["step_idx"][idx]
        noise_level = st["noise_level"]
        timesteps_tensor = st["step_buckets"][idx]
        action_features = head.action_encoder(x_t, timesteps_tensor, embodiment_id)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        model_output = head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        model_output = model_output[:, -head.action_horizon :]
        v_t = head.action_decoder(model_output, embodiment_id)
        timesteps = torch.linspace(0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype)
        t_input = timesteps[idx_t]
        delta = timesteps[idx_t + 1] - timesteps[idx_t]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta
            x_t_std = torch.zeros_like(t_input)
        else:  # train, flow_sde
            sigmas = (
                noise_level
                * torch.sqrt(
                    (1 - timesteps) / torch.where(timesteps == 0, timesteps[1], timesteps)
                )[:-1]
            )
            sigma_i = sigmas[idx_t][:, None, None].expand_as(x_t)
            x0_weight = (
                torch.ones_like(t_input)
                - (t_input + delta)
                - sigma_i**2 * delta / (2 * (1 - t_input))
            )
            x1_weight = t_input + delta
            x_t_std = torch.sqrt(delta) * sigma_i
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def _device_loop(self, st, pre, chosen):
        head = self.model.action_head
        vl_embs = pre["vl_embs"]
        state_features = pre["state_features"]
        embodiment_id = st["emb"]
        device = vl_embs.device
        x_t = pre["x_t"]
        chains = [x_t]
        log_probs = []
        for idx in range(head.num_inference_timesteps):
            mode = "train" if chosen is not None and idx == chosen else "eval"
            x_t_mean, x_t_std = self._sample_mean_var(
                vl_embs, idx, x_t, embodiment_id, state_features, mode, st
            )
            x_t = x_t_mean + head.sample_noise(x_t.shape, device) * x_t_std
            log_prob = head.get_logprob_norm(x_t, x_t_mean, x_t_std)
            chains.append(x_t)
            log_probs.append(log_prob)
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[:, :, : head.action_chunk, : head.valid_action_dim]
        return {"x_0": x_t, "chains": chains, "log_probs": log_probs}

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def _capture(self, fn):
        device = self.model.device
        pool = self.pool
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        error = None
        out = None
        with torch.cuda.stream(stream):
            try:
                graph.capture_begin(pool=pool)
            except BaseException as e:  # noqa: BLE001
                raise _GraphCaptureError(f"capture_begin failed: {e!r}") from e
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
                _recover_failed_capture(device, pool)
        if error is not None:
            del graph, out
            raise _GraphCaptureError(f"graph capture failed: {error!r}") from error
        return graph, out

    _INPUTS = ("ids", "state", "emb", "img_idx", "cu", "pad_rows")

    def _static_buffers(self, h, vision):
        m = self.model
        head = m.action_head
        device = m.device
        B = h["B"]
        steps = head.num_inference_timesteps
        st = {"has_pad": h["has_pad"], "vit": vision["out"]}
        host = {}
        for k in self._INPUTS:
            if h.get(k) is not None:
                st[k] = h[k].to(device)
                host[k] = torch.empty(h[k].shape, dtype=h[k].dtype, pin_memory=True)
        # per-step constants, as sample_mean_var_val computes them from the int step
        step_idx, step_buckets = [], []
        for idx in range(steps):
            idx_cpu = torch.tensor(idx).expand(B)
            t_cont = idx_cpu / float(steps)
            step_buckets.append((t_cont * head.num_timestep_buckets).to(torch.int64).to(device))
            step_idx.append(idx_cpu.to(device))
        st["step_idx"] = step_idx
        st["step_buckets"] = step_buckets
        st["noise_level"] = torch.tensor(head.rl_config.noise_level).to(device)
        if head.rl_config.add_value_head:
            # vl_embs has the backbone's token count: the prompt length N
            st["value_idx"] = torch.tensor(
                _value_token_indices(head.rl_config.value_vlm_mode, h["N"]),
                dtype=torch.long,
                device=device,
            )
        return st, host

    def _saved_rng(self):
        device = self.model.device
        return (
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device),
            random.getstate(),
            np.random.get_state(),
        )

    def _restore_rng(self, states):
        cpu_state, cuda_state, py_state, np_state = states
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, self.model.device)
        random.setstate(py_state)
        np.random.set_state(np_state)

    def _build_vision(self, pix, vkey, param_sig):
        """Graph of the vision tower (SigLIP + mlp1) for pixel values of this shape."""
        device = self.model.device
        eagle = self.model.backbone.eagle_model
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        static = pix.to(device)
        host = torch.empty(pix.shape, dtype=pix.dtype, pin_memory=True)
        rng = self._saved_rng()
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                eagle.extract_feature(static)
            torch.cuda.current_stream(device).wait_stream(side)
            graph, out = self._capture(lambda: eagle.extract_feature(static))
            torch.cuda.synchronize(device)
        except Exception as e:  # noqa: BLE001
            self.pool = None
            torch.cuda.synchronize(device)
            logger.warning("gr00t[rl]: vision CUDA graph disabled, using the eager path: %s", e)
            return None
        finally:
            self._restore_rng(rng)
        while len(self.vision_banks) >= self.MAX_BANKS:
            self.vision_banks.pop(next(iter(self.vision_banks)))
        vb = {"static": static, "host": host, "graph": graph, "out": out, "param_sig": param_sig}
        self.vision_banks[vkey] = vb
        return vb

    def _vision_bank(self, pix, param_sig):
        vkey = (tuple(pix.shape), pix.dtype)
        if vkey in self.vision_failed:
            return None
        vb = self.vision_banks.get(vkey)
        if vb is None or vb["param_sig"] != param_sig:
            vb = self._build_vision(pix, vkey, param_sig)
            if vb is None:
                self.vision_failed.add(vkey)
        return vb

    def launch_vision(self, pix) -> None:
        """Start the vision tower on the request's pixel values (called by the host
        transform as soon as they exist; the text is built while it runs)."""
        self.ticket = None
        try:
            if not torch.is_tensor(pix) or pix.device.type != "cpu":
                return
            if pix.dtype != self.model.action_head.dtype or not pix.is_contiguous():
                return
            param_sig = self._param_signature()
            vb = self._vision_bank(pix, param_sig)
            if vb is None:
                return
            vb["host"].copy_(pix)
            vb["static"].copy_(vb["host"], non_blocking=True)
            vb["graph"].replay()
            self.ticket = (pix, vb, param_sig)
        except Exception as e:  # noqa: BLE001
            self.ticket = None
            logger.warning("gr00t[rl]: vision launch failed: %s", e)

    def _build(self, h, variants, key, param_sig, vision):
        device = self.model.device
        st, host = self._static_buffers(h, vision)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        rng = self._saved_rng()
        loops = {}
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for chosen in variants:
                    pre = self._device_prefix(st)
                    self._device_loop(st, pre, chosen)
                    del pre
            torch.cuda.current_stream(device).wait_stream(side)
            prefix_graph, pre = self._capture(lambda: self._device_prefix(st))
            for chosen in variants:
                loops[chosen] = self._capture(
                    lambda chosen=chosen: self._device_loop(st, pre, chosen)
                )
            torch.cuda.synchronize(device)
        except Exception as e:  # noqa: BLE001
            loops.clear()
            self.pool = None
            torch.cuda.synchronize(device)
            logger.warning(
                "gr00t[rl]: CUDA-graph rollout disabled for this input signature, "
                "using the eager path: %s",
                e,
            )
            return None
        finally:
            self._restore_rng(rng)
        while len(self.banks) >= self.MAX_BANKS:
            self.banks.pop(next(iter(self.banks)))
        bank = {
            "static": st,
            "host": host,
            "prefix": (prefix_graph, pre),
            "loops": loops,
            "param_sig": param_sig,
            "vision": vision,
        }
        self.banks[key] = bank
        return bank

    def _param_signature(self) -> tuple:
        return tuple(p.data_ptr() for p in self.model.parameters())

    # ------------------------------------------------------------------
    # one request
    # ------------------------------------------------------------------
    def run(self, normalized_input: dict[str, Any], mode):
        """Returns (actions [float32, device], result) as ``_get_rl_action``,
        or None when this request must take the eager path (no RNG consumed)."""
        ticket, self.ticket = self.ticket, None
        m = self.model
        head = m.action_head
        cfg = head.rl_config
        h = self._host_inputs(normalized_input)
        if h is None:
            return None
        num_steps = head.num_inference_timesteps
        if mode == "train":
            hi = num_steps - 2 if cfg.ignore_last else num_steps - 1
            if hi < 0:
                return None
            variants = list(range(hi + 1))
        else:
            variants = [None]
        key = self._key(h, variants)
        if key in self.failed:
            return None
        pix = h["pix"]
        if ticket is not None and ticket[0] is pix:
            # vision tower already started on these pixel values
            param_sig = ticket[2]
            vision = ticket[1]
        else:
            param_sig = self._param_signature()
            vision = self._vision_bank(pix, param_sig)
            if vision is None:
                return None
            vision["host"].copy_(pix)
            vision["static"].copy_(vision["host"], non_blocking=True)
            vision["graph"].replay()
        bank = self.banks.get(key)
        if bank is None or bank["param_sig"] != param_sig or bank["vision"] is not vision:
            bank = self._build(h, variants, key, param_sig, vision)
            if bank is None:
                self.failed.add(key)
                return None

        B = h["B"]
        # get_rl_action: the stochastic step, drawn as in eager
        if mode == "train":
            chosen = random.randint(0, hi)
            denoise_inds = torch.tensor([chosen] * num_steps)
        else:
            chosen = None
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(B, 1)

        st = bank["static"]
        host = bank["host"]
        for k, buf in host.items():
            buf.copy_(h[k])
            st[k].copy_(buf, non_blocking=True)
        bank["prefix"][0].replay()
        graph, out = bank["loops"][chosen]
        graph.replay()
        pre = bank["prefix"][1]
        x_0 = out["x_0"].clone()
        chains = out["chains"].clone()
        log_probs = out["log_probs"].clone()
        values = pre["values"].clone()

        actions = x_0.float()
        forward_inputs = {
            "chains": chains,
            "denoise_inds": denoise_inds,
            **normalized_input,
        }
        bsize = normalized_input["state"].shape[0]
        forward_inputs["eagle_pixel_values"] = normalized_input["eagle_pixel_values"].reshape(
            bsize, m.image_nums, *normalized_input["eagle_pixel_values"].shape[1:]
        )
        forward_inputs["eagle_image_sizes"] = normalized_input["eagle_image_sizes"].reshape(
            bsize, m.image_nums, *normalized_input["eagle_image_sizes"].shape[1:]
        )
        result = {
            "prev_logprobs": log_probs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
        return actions, result
