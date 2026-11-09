# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import logging
import os
import random
from typing import Any, Literal, Sequence

import einops
import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.env_io import EnvIO
from rlinf.models.embodiment.openpi.modules import gemma
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.pi0 import Pi0, make_attn_mask
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.sampling import rl_sampler
from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs

logger = logging.getLogger(__name__)

# RLINF_OPENPI_CUDA_GRAPHS=0 turns the CUDA-graph rollout fast path off.
_CUDA_GRAPHS = os.environ.get("RLINF_OPENPI_CUDA_GRAPHS", "1") != "0"
# RLINF_OPENPI_SKIP_DEAD_SLOTS=0 keeps image slots that are masked out for every
# sample (padding cameras) in the graph fast path's prefix.
_SKIP_DEAD_SLOTS = os.environ.get("RLINF_OPENPI_SKIP_DEAD_SLOTS", "1") != "0"


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig:
    """Static RL knobs read from ``actor.model.openpi`` in YAML."""

    add_value_head: bool = False
    noise_method: str = "flow_ode"
    noise_level: float = 0.0
    noise_logvar_range: tuple[float, float] = (0.08, 0.16)
    joint_logprob: bool = False
    ignore_last: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    train_expert_only: bool = False
    is_nft: bool = False
    config_name: str = ""


class Pi0RL(EnvIO, Pi0):
    """On-policy flow RL (PPO / GRPO / NFT) inheriting vendored Pi0."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_chunk: int | None = None,
        action_env_dim: int | None = None,
        rl_cfg: Pi0RLConfig,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name or rl_cfg.config_name,
            state_indices=state_indices,
        )
        self.model_action_dim = self.action_dim
        self.rl_cfg = rl_cfg
        self.global_step = 0
        paligemma_width = gemma.get_config(config.paligemma_variant).width
        action_expert_width = gemma.get_config(config.action_expert_variant).width

        if rl_cfg.add_value_head:
            input_dim = (
                paligemma_width if rl_cfg.value_after_vlm else action_expert_width
            )
            hidden = (1024, 512, 256) if rl_cfg.value_after_vlm else (512, 256, 128)
            if any(
                name in rl_cfg.config_name
                for name in ("pi05_maniskill", "pi05_libero", "pi05_droid_polaris")
            ):
                hidden = (1024, 512, 256)
            # Stay fp32 like OpenPI: action/value heads sit outside
            # paligemma_with_expert and are never converted to bf16.
            self.value_head = ValueHead(
                input_dim=input_dim,
                hidden_sizes=hidden,
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if rl_cfg.noise_method == "flow_noise":
            from rlinf.models.embodiment.modules.explore_noise_net import (
                ExploreNoiseNet,
            )

            self.noise_head = ExploreNoiseNet(
                in_dim=action_expert_width,
                out_dim=self.model_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=list(rl_cfg.noise_logvar_range),
                noise_scheduler_type="learn",
            )

        self._mark_fsdp_wrap_names()

    @property
    def _no_split_modules(self) -> list[str] | None:
        names = list(super()._no_split_modules or [])
        if self.rl_cfg.add_value_head:
            names.append("ValueHead")
        if self.rl_cfg.noise_method == "flow_noise":
            names.append("ExploreNoiseNet")
        return names

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def _build_prefix_cache_for_actor(self, observation: Observation):
        """Prefix KV for actor recompute.

        ``train_expert_only`` freezes the VLM, so prefix must not record
        autograd. OpenPI gets this for free: paligemma is a separate module
        with ``requires_grad=False``. Here expert-0/1 share an FSDP Block, so
        a training prefix forward would keep 18 layers of fp32 attention
        (~2.5GiB/layer at micro_batch=128) and OOM. When the VLM is trained,
        prefix stays in the graph.
        """
        if self.rl_cfg.train_expert_only:
            with torch.no_grad():
                return self.build_prefix_cache(observation)
        return self.build_prefix_cache(observation)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = True,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False)
        # Host-side masks: which camera slots are padding for the whole batch.
        dead_slots = _dead_image_slots(processed)
        observation = self._observation_dict_to_device(processed)
        if mode == "eval":
            return self._predict_eval(observation, noise=noise, rng=rng)
        if self._graph_fast_path_ok(noise, rng):
            actions, result = self._predict_train_graphed(
                observation, compute_values=compute_values, dead_slots=dead_slots
            )
        else:
            actions, result = self._predict_train(
                observation, noise=noise, rng=rng, compute_values=compute_values
            )
        # Match OpenPI: stash raw env obs so the actor can re-transform.
        result["forward_inputs"].update(_clone_replay_obs(repacked))
        return actions, result

    def _predict_train(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        rl_cfg = self.rl_cfg
        device = self.device
        observation = preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        num_steps = self.num_steps

        if noise is None:
            noise = torch.randn(
                B,
                self.action_horizon,
                self.model_action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )

        prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(observation)
        vlm_value = None
        if rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            vlm_value = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )

        collect_nft = rl_cfg.is_nft
        if rl_cfg.joint_logprob or collect_nft:
            chosen_per_step = torch.arange(num_steps, device=device)
            denoise_inds = chosen_per_step[None].expand(B, -1).contiguous()
        else:
            hi = num_steps - 2 if rl_cfg.ignore_last else num_steps - 1
            chosen = random.randint(0, max(hi, 0))
            denoise_inds = torch.full(
                (B, num_steps), chosen, device=device, dtype=torch.long
            )

        nft_state = self._init_nft_state(collect_nft, noise, num_steps, device)
        timesteps = rl_sampler.get_timesteps(num_steps, device)
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        chains = [noise]
        log_probs: list[torch.Tensor] = []
        suffix_values: list[torch.Tensor] = []
        x_t = noise

        if rl_cfg.joint_logprob:
            log_probs.append(
                rl_sampler.gaussian_logprob(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
            )

        for idx in range(num_steps):
            method = (
                rl_cfg.noise_method
                if idx == int(denoise_inds[0, idx].item())
                else "flow_ode"
            )
            t_val = float(timesteps[idx].item())
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
            suffix_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)
            noise_std = None
            if method == "flow_noise":
                noise_std = self.noise_head(suffix_act).to(torch.float32)
            idx_step.fill_(idx)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                x_t.to(torch.float32),
                v_t,
                idx_step,
                noise_method=method,
                noise_level=rl_cfg.noise_level,
                num_steps=num_steps,
                noise_std=noise_std,
            )
            x_t_prev = x_t
            step_noise = torch.randn(
                x_t.shape, device=device, dtype=torch.float32, generator=rng
            )
            x_t = x_t_mean + step_noise * x_t_std
            self._update_nft_state(nft_state, idx, x_t_prev, v_t, x_t, method)
            log_probs.append(rl_sampler.gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)
            if compute_values and rl_cfg.add_value_head and not rl_cfg.value_after_vlm:
                suffix_values.append(
                    rl_sampler.value_from_suffix(
                        self.value_head,
                        suffix_act,
                        action_chunk=self.action_chunk,
                        chunk_critic_input=rl_cfg.chunk_critic_input,
                        detach_critic_input=rl_cfg.detach_critic_input,
                    )
                )

        x_0 = x_t
        chains_tensor = torch.stack(chains, dim=1).contiguous()
        log_probs_stacked = torch.stack(log_probs, dim=1)
        log_probs_stacked = log_probs_stacked[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        if rl_cfg.joint_logprob:
            log_probs_picked = log_probs_stacked.mean(dim=1)
        else:
            log_probs_picked = log_probs_stacked[
                torch.arange(B, device=device), denoise_inds[:, 0]
            ]
        log_probs_picked = log_probs_picked.float().contiguous()

        if vlm_value is not None:
            prev_values = vlm_value[:, None].float().contiguous()
        elif suffix_values:
            prev_values = torch.stack(suffix_values, dim=1).mean(dim=1, keepdim=True)
        else:
            prev_values = torch.zeros((B, 1), device=device, dtype=torch.float32)

        actions = self.decode_actions(x_0, observation.state)

        forward_inputs: dict[str, torch.Tensor] = {
            "chains": chains_tensor,
            "denoise_inds": denoise_inds,
            "tokenized_prompt": observation.tokenized_prompt.contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x_0.reshape(B, -1).contiguous(),
        }
        if nft_state is not None:
            forward_inputs.update(nft_state)
            forward_inputs["nft_x0"] = x_0.detach()

        return actions, {
            "prev_logprobs": log_probs_picked,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
            "model_actions": x_0,
        }

    # ------------------------------------------------------------------
    # CUDA-graph fast path of the rollout (mode="train").
    #
    # The device part of ``_predict_train`` (noise draw, prefix, value head,
    # the denoising loop, the log-probabilities) is replayed from CUDA graphs
    # captured once per input signature and per denoise step chosen by
    # ``random.randint``. The graphs launch exactly the kernels of the eager
    # path on the same inputs, and draw the same noise from the global CUDA
    # generator in the same order (graph-safe philox offsets), so every output
    # is bitwise the eager one. With ``joint_logprob`` every step is noised,
    # no step is drawn, and one loop graph serves every request. Only verified
    # configurations take it (no caller noise / generator, no NFT, a noise
    # method of ``_GRAPH_NOISE_METHODS``, eval mode, no grad); everything else
    # runs ``_predict_train``.
    # ------------------------------------------------------------------
    _GRAPH_MAX_BANKS = 8
    # Noise methods verified capturable and bitwise equal to ``_predict_train``.
    # (flow_cps is not: its ``torch.as_tensor(noise_level, device=...)`` is a
    # host-to-device copy, which a CUDA graph cannot capture.)
    # flow_noise: the noise head is plain device ops on the suffix output.
    _GRAPH_NOISE_METHODS = ("flow_sde", "flow_ode", "flow_noise")

    def _graph_fast_path_ok(self, noise, rng) -> bool:
        rl_cfg = self.rl_cfg
        return (
            _CUDA_GRAPHS
            and noise is None
            and rng is None
            and not self.training
            and not torch.is_grad_enabled()
            and not rl_cfg.is_nft
            and rl_cfg.noise_method in self._GRAPH_NOISE_METHODS
            and torch.cuda.is_available()
            and self.device.type == "cuda"
            and not torch.cuda.is_current_stream_capturing()
        )

    @staticmethod
    def _flatten_observation(observation: Observation) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for k, v in observation.images.items():
            flat[f"images/{k}"] = v
        for k, v in observation.image_masks.items():
            flat[f"image_masks/{k}"] = v
        for name in (
            "state",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
            "pcd_xyz",
        ):
            flat[name] = getattr(observation, name)
        return flat

    @staticmethod
    def _unflatten_observation(flat: dict[str, Any]) -> Observation:
        images = {k[len("images/") :]: v for k, v in flat.items() if k.startswith("images/")}
        masks = {
            k[len("image_masks/") :]: v
            for k, v in flat.items()
            if k.startswith("image_masks/")
        }
        return Observation(
            images=images,
            image_masks=masks,
            state=flat["state"],
            tokenized_prompt=flat["tokenized_prompt"],
            tokenized_prompt_mask=flat["tokenized_prompt_mask"],
            token_ar_mask=flat["token_ar_mask"],
            token_loss_mask=flat["token_loss_mask"],
            pcd_xyz=flat["pcd_xyz"],
        )

    def _timesteps_host(self, device) -> list[float]:
        cache = self.__dict__.setdefault("_timesteps_host_cache", {})
        key = (self.num_steps, str(device))
        if key not in cache:
            # The exact fp32 values the eager loop reads with ``.item()``.
            cache[key] = rl_sampler.get_timesteps(self.num_steps, device).tolist()
        return cache[key]

    def _usable_dead_slots(self, observation: Observation, dead_slots) -> tuple:
        """The subset of ``dead_slots`` the prefix may skip (else ``()``)."""
        if not (_SKIP_DEAD_SLOTS and dead_slots):
            return ()
        if self.pcd or observation.tokenized_prompt is None:
            return ()
        names = list(observation.images)
        dead = tuple(n for n in dead_slots if n in observation.images)
        if not dead or len(dead) == len(names):
            return ()
        shapes = {tuple(observation.images[n].shape) for n in names}
        if len(shapes) != 1:  # a skipped slot's token count is taken from a kept one
            return ()
        return dead

    def _prefix_cache_without(self, obs: Observation, dead: tuple):
        """``build_prefix_cache`` without the image slots in ``dead``.

        Every slot in ``dead`` is masked out for every sample: no valid token
        attends to its tokens (their softmax weight is exactly 0, their values
        enter only as 0 * v), and their own outputs are masked out of the
        value head and of the suffix. Dropping them leaves every valid
        token's computation as it was (the dropped key blocks are a multiple of
        32 / 64 positions, so the softmax lanes and the attention-GEMM K blocks
        of the kept keys do not move); the result is verified bitwise against
        ``build_prefix_cache``.

        Returns ``(prefix_out, full_mask, prefix_mask, kv_cache)``: ``prefix_out``
        and ``full_mask`` in the full layout (zeros at the dropped slots) for the
        value head, ``prefix_mask`` / ``kv_cache`` of the kept tokens for the
        suffix.
        """
        image_names = [name for name in IMAGE_KEYS if name in obs.images]
        image_names.extend(name for name in obs.images if name not in IMAGE_KEYS)
        tokens, kept_mask, full_mask, segments = [], [], [], []
        ntok = None
        for name in image_names:
            if name in dead:
                segments.append((name, None))
                continue
            image_tokens, _ = self.img(obs.images[name])
            ntok = image_tokens.shape[1]
            segments.append((name, image_tokens))
        for name, image_tokens in segments:
            mask = einops.repeat(obs.image_masks[name], "b -> b s", s=ntok)
            full_mask.append(mask)
            if image_tokens is not None:
                tokens.append(image_tokens)
                kept_mask.append(mask)
        lang = self.llm.embed(obs.tokenized_prompt)
        tokens.append(lang)
        kept_mask.append(obs.tokenized_prompt_mask)
        full_mask.append(obs.tokenized_prompt_mask)
        tokens = torch.cat(tokens, dim=1)
        kept_mask = torch.cat(kept_mask, dim=1)
        full_mask = torch.cat(full_mask, dim=1)
        ar_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
        attn_mask = make_attn_mask(kept_mask, ar_mask)
        positions = torch.cumsum(kept_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [tokens, None], positions=positions, mask=attn_mask
        )
        out = outputs[0]
        pieces, start = [], 0
        for _, image_tokens in segments:
            if image_tokens is None:
                pieces.append(out.new_zeros(out.shape[0], ntok, out.shape[2]))
            else:
                pieces.append(out[:, start : start + ntok])
                start += ntok
        pieces.append(out[:, start:])
        return torch.cat(pieces, dim=1), full_mask, kept_mask, kv_cache

    def _device_prefix(self, observation: Observation, dead: tuple = ()) -> dict[str, Any]:
        """Device part of ``_predict_train`` before the denoising loop (capturable)."""
        rl_cfg = self.rl_cfg
        B = observation.state.shape[0]
        noise = torch.randn(
            B,
            self.action_horizon,
            self.model_action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if dead:
            prefix_out, value_mask, prefix_mask, kv_cache = self._prefix_cache_without(
                observation, dead
            )
        else:
            prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(observation)
            value_mask = prefix_mask
        vlm_value = None
        if rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            vlm_value = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, value_mask, mode=rl_cfg.value_vlm_mode
            )
        return {
            "noise": noise,
            "prefix_mask": prefix_mask,
            "kv_cache": kv_cache,
            "vlm_value": vlm_value,
        }

    def _device_suffix(
        self,
        observation: Observation,
        pre: dict[str, Any],
        chosen: int,
        compute_values: bool,
        timesteps_host: list[float],
    ) -> dict[str, torch.Tensor]:
        """Denoising loop of ``_predict_train`` (capturable).

        ``chosen`` is the step drawn by ``random.randint``, or ``None`` with
        ``joint_logprob`` (every step uses the noise method, no step is drawn).
        """
        rl_cfg = self.rl_cfg
        device = self.device
        B = observation.state.shape[0]
        num_steps = self.num_steps
        noise, prefix_mask = pre["noise"], pre["prefix_mask"]
        kv_cache, vlm_value = pre["kv_cache"], pre["vlm_value"]
        joint = chosen is None
        if joint:
            chosen_per_step = torch.arange(num_steps, device=device)
            denoise_inds = chosen_per_step[None].expand(B, -1).contiguous()
        else:
            denoise_inds = torch.full(
                (B, num_steps), chosen, device=device, dtype=torch.long
            )
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        chains = [noise]
        log_probs: list[torch.Tensor] = []
        suffix_values: list[torch.Tensor] = []
        x_t = noise
        if joint:
            log_probs.append(
                rl_sampler.gaussian_logprob(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
            )
        for idx in range(num_steps):
            method = rl_cfg.noise_method if (joint or idx == chosen) else "flow_ode"
            t_val = timesteps_host[idx]
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
            suffix_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)
            noise_std = None
            if method == "flow_noise":
                noise_std = self.noise_head(suffix_act).to(torch.float32)
            idx_step.fill_(idx)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                x_t.to(torch.float32),
                v_t,
                idx_step,
                noise_method=method,
                noise_level=rl_cfg.noise_level,
                num_steps=num_steps,
                noise_std=noise_std,
            )
            step_noise = torch.randn(x_t.shape, device=device, dtype=torch.float32)
            x_t = x_t_mean + step_noise * x_t_std
            log_probs.append(rl_sampler.gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)
            if compute_values and rl_cfg.add_value_head and not rl_cfg.value_after_vlm:
                suffix_values.append(
                    rl_sampler.value_from_suffix(
                        self.value_head,
                        suffix_act,
                        action_chunk=self.action_chunk,
                        chunk_critic_input=rl_cfg.chunk_critic_input,
                        detach_critic_input=rl_cfg.detach_critic_input,
                    )
                )

        x_0 = x_t
        chains_tensor = torch.stack(chains, dim=1).contiguous()
        log_probs_stacked = torch.stack(log_probs, dim=1)
        log_probs_stacked = log_probs_stacked[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        if joint:
            log_probs_picked = log_probs_stacked.mean(dim=1)
        else:
            log_probs_picked = log_probs_stacked[
                torch.arange(B, device=device), denoise_inds[:, 0]
            ]
        log_probs_picked = log_probs_picked.float().contiguous()
        if vlm_value is not None:
            prev_values = vlm_value[:, None].float().contiguous()
        elif suffix_values:
            prev_values = torch.stack(suffix_values, dim=1).mean(dim=1, keepdim=True)
        else:
            prev_values = torch.zeros((B, 1), device=device, dtype=torch.float32)
        return {
            "chains": chains_tensor,
            "denoise_inds": denoise_inds,
            "prev_logprobs": log_probs_picked,
            "prev_values": prev_values,
            "x_0": x_0,
        }

    def _param_signature(self) -> tuple:
        return tuple(p.data_ptr() for p in self.parameters())

    def _capture_graph(self, fn):
        """Capture ``fn()`` into a new graph of the shared pool; returns (graph, out).

        On any failure the capture is ended, the caching allocator stops
        routing allocations to the pool, the CUDA generator leaves capture
        mode, the current stream is restored, and ``_GraphCaptureError`` is
        raised: the process stays usable for the eager path.
        """
        device = self.device
        pool = self._graph_pool
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

    def _graph_variants(self, hi: int) -> list:
        """Loop graphs of a bank: one per drawable step, or one (``None``) when joint."""
        return [None] if self.rl_cfg.joint_logprob else list(range(hi + 1))

    def _build_graph_bank(self, flat, compute_values, hi, key, param_sig, dead=()):
        device = self.device
        variants = self._graph_variants(hi)
        static = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in flat.items()
        }
        static_obs = self._unflatten_observation(static)
        timesteps_host = self._timesteps_host(device)
        if getattr(self, "_graph_pool", None) is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        # Warmup and capture must not consume any random number of the caller.
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device)
        py_state = random.getstate()
        np_state = np.random.get_state()
        graphs = {}
        try:
            # Eager warmup of every variant (compiles, library handles).
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for chosen in variants:
                    pre = self._device_prefix(static_obs, dead)
                    self._device_suffix(
                        static_obs, pre, chosen, compute_values, timesteps_host
                    )
                    del pre
            torch.cuda.current_stream(device).wait_stream(side)
            # Two graphs per request: the prefix (shared by every chosen step),
            # then the denoising loop of the chosen step. Launching the second
            # while the first runs hides its launch cost.
            prefix_graph, pre = self._capture_graph(
                lambda: self._device_prefix(static_obs, dead)
            )
            for chosen in variants:
                graphs[chosen] = self._capture_graph(
                    lambda chosen=chosen: self._device_suffix(
                        static_obs, pre, chosen, compute_values, timesteps_host
                    )
                )
            torch.cuda.synchronize(device)
        except Exception as e:  # noqa: BLE001
            # Not graphable (or the warmup failed): drop everything captured
            # for this signature, use a fresh pool for later banks, and let the
            # caller run the eager path for this and later calls.
            graphs.clear()
            prefix_graph = pre = None
            self._graph_pool = None
            torch.cuda.synchronize(device)
            logger.warning(
                "openpi[rl]: CUDA-graph rollout disabled for this input "
                "signature, using the eager path: %s",
                e,
            )
            return None
        finally:
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state, device)
            random.setstate(py_state)
            np.random.set_state(np_state)
        banks = self.__dict__.setdefault("_graph_banks", {})
        while len(banks) >= self._GRAPH_MAX_BANKS:
            banks.pop(next(iter(banks)))
        bank = {
            "static": static,
            "prefix": (prefix_graph, pre),
            "graphs": graphs,
            "param_sig": param_sig,
        }
        banks[key] = bank
        return bank

    def _predict_train_graphed(
        self, observation: Observation, *, compute_values: bool, dead_slots=()
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        rl_cfg = self.rl_cfg
        observation = preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        num_steps = self.num_steps
        hi = max(num_steps - 2 if rl_cfg.ignore_last else num_steps - 1, 0)
        flat = self._flatten_observation(observation)
        dead = self._usable_dead_slots(observation, dead_slots)
        key = (
            bool(compute_values),
            num_steps,
            dead,
            tuple(
                (k, tuple(v.shape), v.dtype, v.device) if torch.is_tensor(v) else (k, v)
                for k, v in flat.items()
            ),
        )
        if any(not torch.is_tensor(v) and v is not None for v in flat.values()):
            return self._predict_train(
                observation, noise=None, rng=None, compute_values=compute_values
            )
        failed = self.__dict__.setdefault("_graph_failed_keys", set())
        if key in failed:
            return self._predict_train(
                observation, noise=None, rng=None, compute_values=compute_values
            )
        param_sig = self._param_signature()
        bank = self.__dict__.get("_graph_banks", {}).get(key)
        if bank is None or bank["param_sig"] != param_sig:
            bank = self._build_graph_bank(
                flat, compute_values, hi, key, param_sig, dead
            )
            if bank is None:
                failed.add(key)
                return self._predict_train(
                    observation, noise=None, rng=None, compute_values=compute_values
                )

        # As in ``_predict_train``: a step is drawn only without joint logprobs.
        chosen = None if rl_cfg.joint_logprob else random.randint(0, hi)
        for k, v in flat.items():
            if v is not None:
                bank["static"][k].copy_(v)
        bank["prefix"][0].replay()
        graph, out = bank["graphs"][chosen]
        graph.replay()
        res = {k: v.clone() for k, v in out.items()}

        x_0 = res["x_0"]
        actions = self.decode_actions(x_0, observation.state)
        forward_inputs: dict[str, torch.Tensor] = {
            "chains": res["chains"],
            "denoise_inds": res["denoise_inds"],
            "tokenized_prompt": observation.tokenized_prompt.contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x_0.reshape(B, -1).contiguous(),
        }
        return actions, {
            "prev_logprobs": res["prev_logprobs"],
            "prev_values": res["prev_values"],
            "forward_inputs": forward_inputs,
            "model_actions": x_0,
        }

    def _init_nft_state(
        self,
        collect: bool,
        x_t: torch.Tensor,
        num_steps: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if not collect:
            return None
        return {
            "nft_step_index": torch.randint(
                0, num_steps, (x_t.shape[0],), device=device
            ),
            "nft_xcur": torch.zeros_like(x_t),
            "nft_v": torch.zeros_like(x_t),
            "nft_xnext": torch.zeros_like(x_t),
            "nft_noise_level": torch.zeros(
                x_t.shape[0], device=device, dtype=x_t.dtype
            ),
        }

    def _update_nft_state(
        self,
        nft_state: dict[str, torch.Tensor] | None,
        idx: int,
        x_t_prev: torch.Tensor,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sample_method: str,
    ) -> None:
        if nft_state is None:
            return
        mask = nft_state["nft_step_index"] == idx
        if not mask.any():
            return
        mask_bc = mask[:, None, None]
        nft_state["nft_xcur"] = torch.where(
            mask_bc, x_t_prev.detach(), nft_state["nft_xcur"]
        )
        nft_state["nft_v"] = torch.where(mask_bc, v_t.detach(), nft_state["nft_v"])
        nft_state["nft_xnext"] = torch.where(
            mask_bc, x_t.detach(), nft_state["nft_xnext"]
        )
        level = 0.0 if sample_method == "flow_ode" else float(self.rl_cfg.noise_level)
        nft_state["nft_noise_level"] = torch.where(
            mask,
            torch.full_like(nft_state["nft_noise_level"], level),
            nft_state["nft_noise_level"],
        )

    def _observation_from_forward_inputs(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> Observation:
        """Rebuild the rollout observation so actor logprobs match sampling.

        Prefer raw ``observation/*`` tensors (OpenPI path): re-run
        ``input_transform`` then ``preprocess_observation``. Fall back to the
        older ``obs_image__*`` layout, assembling cameras in ``IMAGE_KEYS``
        order rather than ``dict`` iteration order.
        """
        has_raw_obs = any(key.startswith("observation/") for key in forward_inputs)
        if has_raw_obs:
            processed = self.input_transform(forward_inputs, transpose=False)
            observation = self._observation_dict_to_device(processed)
        else:
            images: dict[str, torch.Tensor] = {}
            image_masks: dict[str, torch.Tensor] = {}
            for name in IMAGE_KEYS:
                image_key = f"obs_image__{name}"
                if image_key in forward_inputs:
                    images[name] = forward_inputs[image_key]
                mask_key = f"obs_image_mask__{name}"
                if mask_key in forward_inputs:
                    image_masks[name] = forward_inputs[mask_key]
            for key, value in forward_inputs.items():
                if key.startswith("obs_image__"):
                    name = key[len("obs_image__") :]
                    images.setdefault(name, value)
                elif key.startswith("obs_image_mask__"):
                    name = key[len("obs_image_mask__") :]
                    image_masks.setdefault(name, value)
            observation = Observation(
                images=images,
                image_masks=image_masks,
                state=forward_inputs["obs_state"],
                tokenized_prompt=forward_inputs["tokenized_prompt"],
                tokenized_prompt_mask=forward_inputs["tokenized_prompt_mask"],
            )
        return preprocess_observation(observation, train=False)

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return super().sft_forward(**kwargs)
        raise NotImplementedError(
            f"{type(self).__name__} does not support forward_type={forward_type!r}."
        )

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """PPO/GRPO recompute of logprobs and values from the stored chain."""
        rl_cfg = self.rl_cfg
        compute_values = kwargs.get("compute_values", True)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        B = chains.shape[0]
        device = chains.device
        observation = self._observation_from_forward_inputs(forward_inputs)

        prefix_out, prefix_mask, kv_cache = self._build_prefix_cache_for_actor(
            observation
        )

        timesteps = rl_sampler.get_timesteps(self.num_steps, device)
        if rl_cfg.joint_logprob:
            num_recompute = self.num_steps
            initial = rl_sampler.gaussian_logprob(
                chains[:, 0].to(torch.float32),
                torch.zeros_like(chains[:, 0], dtype=torch.float32),
                torch.ones_like(chains[:, 0], dtype=torch.float32),
            )
            step_logprobs = [initial]
            step_entropy = [
                rl_sampler.gaussian_entropy(
                    torch.ones_like(chains[:, 0], dtype=torch.float32)
                )
            ]
        else:
            num_recompute = 1
            step_logprobs = []
            step_entropy = []

        suffix_values = []
        arange_B = torch.arange(B, device=device)
        for idx in range(num_recompute):
            denoise_ind = denoise_inds[:, idx].to(torch.long)
            chains_pre = chains[arange_B, denoise_ind]
            chains_next = chains[arange_B, denoise_ind + 1]
            t_input = timesteps[denoise_ind].to(torch.float32)
            suffix_act = self.run_suffix(
                observation, chains_pre, t_input, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)
            noise_std = None
            if rl_cfg.noise_method == "flow_noise":
                noise_std = self.noise_head(suffix_act).to(torch.float32)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                chains_pre.to(torch.float32),
                v_t,
                denoise_ind,
                noise_method=rl_cfg.noise_method,
                noise_level=rl_cfg.noise_level,
                num_steps=self.num_steps,
                noise_std=noise_std,
            )
            step_logprobs.append(
                rl_sampler.gaussian_logprob(
                    chains_next.to(torch.float32), x_t_mean, x_t_std
                )
            )
            step_entropy.append(rl_sampler.gaussian_entropy(x_t_std))
            if compute_values and rl_cfg.add_value_head and not rl_cfg.value_after_vlm:
                suffix_values.append(
                    rl_sampler.value_from_suffix(
                        self.value_head,
                        suffix_act,
                        action_chunk=self.action_chunk,
                        chunk_critic_input=rl_cfg.chunk_critic_input,
                        detach_critic_input=rl_cfg.detach_critic_input,
                    )
                )

        log_probs = torch.stack(step_logprobs, dim=1)
        log_probs = log_probs[:, :, : self.action_chunk, : self.action_env_dim].float()
        if rl_cfg.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[:, 0]

        if compute_values and rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            values = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )
        elif suffix_values:
            values = torch.stack(suffix_values, dim=1).mean(dim=1)
        else:
            values = torch.zeros(B, device=device, dtype=torch.float32)

        entropy = torch.stack(step_entropy, dim=1)
        entropy = entropy[:, :, : self.action_chunk, : self.action_env_dim]
        if rl_cfg.noise_method == "flow_noise":
            entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        else:
            entropy = torch.zeros((B, 1), device=device, dtype=torch.float32)
        return {
            "logprobs": log_probs.contiguous(),
            "values": values.float(),
            "entropy": entropy.float(),
        }

    def nft_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """Compute velocity v_theta at explicit (x_t, timesteps) for NFT loss."""
        nft_inputs = kwargs["nft_inputs"]
        observation = self._observation_from_forward_inputs(forward_inputs)
        device = self.device
        x_t = nft_inputs["x_t"].to(device)
        t = nft_inputs["timesteps"].to(device)
        if t.dim() > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        prefix_out, prefix_mask, kv_cache = self._build_prefix_cache_for_actor(
            observation
        )
        suffix_act = self.run_suffix(observation, x_t, t, kv_cache, prefix_mask)
        v_theta = self.velocity_from_suffix(suffix_act)
        v_theta = v_theta[:, : self.action_chunk, :]
        result: dict[str, Any] = {"v_theta": v_theta, "x_t": x_t, "timesteps": t}
        if kwargs.get("compute_values") and self.rl_cfg.add_value_head:
            if self.rl_cfg.value_after_vlm:
                result["values"] = rl_sampler.value_from_prefix(
                    self.value_head,
                    prefix_out,
                    prefix_mask,
                    mode=self.rl_cfg.value_vlm_mode,
                )[:, None]
            else:
                result["values"] = rl_sampler.value_from_suffix(
                    self.value_head, suffix_act, action_chunk=self.action_chunk
                )[:, None]
        return result


class _GraphCaptureError(RuntimeError):
    """A CUDA-graph capture failed; the process state was restored."""


def _recover_failed_capture(device: torch.device, pool) -> None:
    """Undo what a capture whose ``capture_end`` raised leaves behind.

    ``CUDAGraph.capture_end`` stops at the failing ``cudaStreamEndCapture``:
    the caching allocator would keep allocating into the graph pool and the
    CUDA generators would stay in capture mode (every later random draw then
    fails or uses the graph offsets). End the pool allocation and run one empty
    capture (in its own pool) whose clean ``capture_end`` takes the generators
    out of capture mode.
    """
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


def _dead_image_slots(processed: dict[str, Any]) -> tuple:
    """Camera slots whose host-side ``image_mask`` is False for every sample."""
    masks = processed.get("image_mask") if isinstance(processed, dict) else None
    if not isinstance(masks, dict):
        return ()
    dead = []
    for name, mask in masks.items():
        if torch.is_tensor(mask):
            if mask.device.type != "cpu" or mask.numel() == 0:
                continue
            if not bool(mask.any()):
                dead.append(name)
        elif isinstance(mask, np.ndarray) and mask.size and not mask.any():
            dead.append(name)
    return tuple(sorted(dead))


def _clone_replay_obs(repacked: dict[str, Any]) -> dict[str, Any]:
    """Clone env-repacked tensors for ``forward_inputs`` (skip the prompt)."""
    cloned: dict[str, Any] = {}
    for key, value in repacked.items():
        if key == "prompt":
            continue
        if torch.is_tensor(value):
            cloned[key] = value.detach().contiguous()
        elif isinstance(value, np.ndarray):
            cloned[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            cloned[key] = value
    return cloned
