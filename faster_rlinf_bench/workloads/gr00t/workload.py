"""The judged workload, shared by the timing worker and the numerics adapter (one code path for both).

One request = one RL rollout step of RLinf's rollout worker (rlinf/workers/rollout/hf/huggingface_worker.py,
`predict`) for a batch of LIBERO environments, as in examples/embodiment/config/libero_spatial_ppo_gr00t.yaml: the
env worker's observation (rlinf/envs/sim/libero/libero_env.py `_wrap_obs`: torch CPU tensors, one 256x256 uint8
agent-view image and one 256x256 uint8 wrist image per env, the 8-dim end-effector state, the task instruction) goes
into `GR00T_N1_5_ForRLActionPrediction.predict_action_batch(env_obs=obs, mode="train")` (the worker passes only
`mode` for GR00T), under `torch.no_grad()` with the model in eval mode.

What one request computes (reference code: rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py and the
installed `gr00t` package, Isaac-GR00T n1.5-release): the state is rounded through bf16; the observation is converted
to GR00T's modality dict and transformed on the host (both images center-cropped to 95% and resized to 224x224, the
state min-max normalized, the instruction and image tokens built by the Eagle 2.5 processor, padded to 570 tokens);
the Eagle 2.5 VL backbone (SigLIP2 vision + the first 12 layers of Qwen3-1.7B, flash-attention 2) embeds images and
text once; the flow-matching action head (a 16-layer DiT over state, future and action tokens, cross-attending to the
backbone features) then runs 4 denoising steps from Gaussian noise. Mode "train" uses the flow_sde noise method: one
step, drawn with Python's `random.randint`, is stochastic (noise level 0.5); every step draws Gaussian noise from the
global torch CUDA RNG (scaled by zero on the deterministic steps). The value head estimates the state value from the
mean backbone feature and the state features. The result: the env-space actions on the host (5 chunks x 7 dims,
unnormalized, gripper binarized), and, as the worker keeps them for the trainer, `prev_logprobs` (per step, chunk and
dim), `prev_values` and `forward_inputs` (the denoising chain, the stochastic step index, every model input).

`replay` is the trainer's side of the same data: `default_forward` (called through `model(forward_inputs=...)` as
rlinf/workers/actor/embodied_fsdp_actor_worker.py does for GR00T: compute_values True for GAE, compute_entropy False,
the rollout's prev_logprobs passed as a keyword) recomputes the log-probability of the stochastic step and the value
from the stored inputs and chain. The judge runs it forward only, under no_grad, on the rollout's own forward_inputs.

The value head is not trained in the SFT checkpoint: RLinf re-initializes it after loading (`_init_weights`, from the
global RNG). `build_model` then sets its weights from a fixed generator by parameter name, so they do not depend on how
much of the global RNG the model's construction consumes.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# LIBERO-Spatial task instructions, as the LIBERO env passes them (`task.language`, i.e. the suite's task names with
# spaces: libero/libero/benchmark/libero_suite_task_map.py + grab_language_from_filename).
PROMPTS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]

# The rollout worker's call for GR00T (huggingface_worker.predict: kwargs = {"mode": mode}).
ROLLOUT_KWARGS = {"mode": "train"}
# The trainer's call (embodied_fsdp_actor_worker.train_micro_batch for GR00T with adv_type gae, entropy_bonus 0).
TRAIN_KWARGS = {"compute_logprobs": True, "compute_entropy": False, "compute_values": True, "use_cache": False}

# Parameters RLinf initializes randomly after loading (the value head; the reinflow noise net of other
# configurations), set deterministically by name.
UNLOADED_PREFIXES = ("action_head.value_head.", "action_head.reinflow_explore_noise_net.")
UNLOADED_SEED = 20261110

IMAGE_SIZE = 256   # LIBERO env init_params camera_heights / camera_widths


def model_cfg(weights):
    from omegaconf import OmegaConf

    cfg = json.loads((HERE / "model_cfg.json").read_text())
    cfg.pop("_comment", None)
    cfg["model_path"] = str(weights)
    return OmegaConf.create(cfg)


def import_tree(tree):
    """Make `tree` (a checkout of RLinf) the one that `import rlinf` resolves to."""
    tree = str(Path(tree).resolve())
    for name in [m for m in sys.modules if m == "rlinf" or m.startswith("rlinf.")]:
        del sys.modules[name]
    sys.path.insert(0, tree)
    import rlinf

    if not str(Path(rlinf.__file__).resolve()).startswith(tree + os.sep):
        raise RuntimeError(f"rlinf resolved to {rlinf.__file__}, not inside {tree}")
    return rlinf


def set_unloaded(model, required):
    """Value-head (and noise-net) weights from a fixed generator, per parameter name (fan-in scaled normal, zero
    biases)."""
    found = 0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if not name.startswith(UNLOADED_PREFIXES):
                continue
            found += 1
            seed = int(hashlib.sha256(f"{UNLOADED_SEED}|{name}".encode()).hexdigest()[:15], 16)
            g = torch.Generator().manual_seed(seed)
            if p.dim() >= 2:
                value = torch.randn(p.shape, generator=g, dtype=torch.float64) * (1.0 / p.shape[1]) ** 0.5
            else:
                value = torch.zeros(p.shape, dtype=torch.float64)
            p.copy_(value.to(dtype=p.dtype, device=p.device))
    if required and not found:
        raise RuntimeError(f"no parameter starts with {UNLOADED_PREFIXES}: the value head is missing or renamed")
    return found


def build_model(tree, weights, cfg=None):
    """The tree's own factory with the frozen configuration (precision bf16, as rlinf.models.get_model passes it);
    returns the eval-mode rollout model on cuda:0."""
    import_tree(tree)
    from rlinf.models.embodiment.gr00t import get_model

    cfg = cfg if cfg is not None else model_cfg(weights)
    model = get_model(cfg, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    set_unloaded(model, required=bool(cfg.rl_head_config.get("add_value_head", False)))
    return model


def _image(rng, height, width):
    """A camera-like frame: a smooth random field (scene-scale structure) plus pixel noise, as uint8 HWC."""
    coarse = rng.random((-(-height // 16), -(-width // 16), 3))
    smooth = np.kron(coarse, np.ones((16, 16, 1)))[:height, :width]
    noise = rng.normal(0.0, 0.08, (height, width, 3))
    return (np.clip(0.15 + 0.7 * smooth + noise, 0.0, 1.0) * 255).astype(np.uint8)


def observation(seed, index, batch=8, size=IMAGE_SIZE):
    """Request `index` of the stream `seed`: a fresh, deterministic batch of env observations (never repeated), in
    the LIBERO env's format (torch CPU tensors; float64 states become float32 as rlinf.envs.utils.to_tensor does).
    `size` is an int (square) or (height, width)."""
    h, w = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng([int(seed), int(index), int(batch)])
    main = np.stack([_image(rng, h, w) for _ in range(batch)])
    wrist = np.stack([_image(rng, h, w) for _ in range(batch)])
    pos = rng.normal([-0.05, 0.0, 1.0], 0.1, (batch, 3))
    axis_angle = rng.normal([3.1, 0.0, 0.0], 0.2, (batch, 3))
    grip = rng.uniform(0.0, 0.04, (batch, 1)) * np.array([1.0, -1.0])
    states = np.concatenate([pos, axis_angle, grip], axis=1).astype(np.float32)
    return {
        "main_images": torch.from_numpy(main),
        "wrist_images": torch.from_numpy(wrist),
        "states": torch.from_numpy(states),
        "task_descriptions": [PROMPTS[int(i)] for i in rng.integers(0, len(PROMPTS), batch)],
    }


def request(model, obs, **kwargs):
    """One rollout step, end to end (the timed and the judged unit): host actions plus the result as the model leaves
    it."""
    with torch.no_grad():
        actions, result = model.predict_action_batch(env_obs=obs, **(kwargs or ROLLOUT_KWARGS))
    if isinstance(actions, np.ndarray):   # the rollout worker converts numpy actions with torch.from_numpy
        actions = torch.from_numpy(actions)
    return actions.detach().cpu(), result


def replay(model, result, **overrides):
    """The trainer's recompute of the rollout (default_forward), forward only."""
    kw = dict(TRAIN_KWARGS, **overrides)
    with torch.no_grad():
        out = model(forward_inputs=dict(result["forward_inputs"]), prev_logprobs=result["prev_logprobs"], **kw)
    return {k: v for k, v in out.items() if torch.is_tensor(v)}


def flatten(actions, result, replayed=None):
    """Every judged tensor of one request, by name."""
    named = {"actions": actions, "prev_logprobs": result["prev_logprobs"], "prev_values": result["prev_values"]}
    for k, v in sorted(result["forward_inputs"].items()):
        if torch.is_tensor(v):
            named[f"forward_inputs.{k}"] = v
    for k, v in sorted((replayed or {}).items()):
        named[f"replay.{k}"] = v
    return named


def digest_update(h, named):
    """Fold the raw bytes (and dtype, shape) of every named tensor into the hash `h`."""
    for k in sorted(named):
        t = named[k].detach().contiguous().cpu()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
