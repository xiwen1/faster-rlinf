"""The judged workload, shared by the timing worker and the numerics adapter (one code path for both).

One request = one RL rollout step of RLinf's rollout worker (rlinf/workers/rollout/hf/huggingface_worker.py,
`predict`) for a batch of LIBERO environments, as in examples/embodiment/config/libero_10_grpo_openvlaoft.yaml:
the env worker's observation (rlinf/envs/sim/libero/libero_env.py `_wrap_obs`: torch CPU tensors, one 256x256 uint8
agent-view image and one 256x256 uint8 wrist image per env, the 8-dim end-effector state, the task instruction) goes
into `OpenVLAOFTForRLActionPrediction.predict_action_batch(env_obs=obs, **sampling)` with the rollout's train
sampling parameters (do_sample, temperature_train 1.6, top_k -1, top_p 1.0, max_new_tokens None), under
`torch.no_grad()` with the model in eval mode. The action tokens are sampled with `torch.multinomial` from the global
torch CUDA RNG; Python's `random` and numpy's global RNG are not drawn from. The env-space actions come back on the
host (the model returns them as a CPU tensor); the other outputs (`prev_logprobs`, `prev_values`, `forward_inputs`)
stay on the GPU, as in the worker, which keeps them for the trainer.

`replay` is the trainer's side of the same data: `default_forward` (called through `model(forward_inputs=...)` with
the rollout's temperature and top_k, as rlinf/workers/actor/embodied_fsdp_actor_worker.py does) recomputes the
log-probabilities (and the entropy) of a rollout from its stored inputs and sampled action tokens. The judge runs it
forward only, under no_grad, on the rollout's own `forward_inputs`.

The configuration has no value head (GRPO). If a value head is configured (other configurations), its weights, which
the SFT checkpoint lacks, are set by `build_model` from a fixed generator by parameter name, so they do not depend on
how much of the global RNG the model's construction consumes.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# LIBERO-10 (LIBERO-Long) task instructions, as the LIBERO env passes them (`task.language`). The model wraps each in
# "In: What action should the robot take to <instruction>?\nOut: " and pads the prompt to max_prompt_length=128.
PROMPTS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]

# The rollout worker's train sampling parameters (huggingface_worker.setup_sample_params with the
# libero_10_grpo_openvlaoft.yaml rollout.sampling_params after rlinf.config.validate_embodied_cfg).
SAMPLING = {"do_sample": True, "temperature": 1.6, "top_k": -1, "top_p": 1.0, "max_new_tokens": None}
# The trainer's kwargs for the same model type (embodied_fsdp_actor_worker: temperature_train, top_k).
TRAIN_KWARGS = {"temperature": 1.6, "top_k": -1}

# Parameters that are not in the checkpoint (randomly initialized heads), set deterministically by name.
UNLOADED_PREFIXES = ("value_head.",)
UNLOADED_SEED = 20261109

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
    """Value-head weights from a fixed generator, per parameter name (fan-in scaled normal, zero biases)."""
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
    from rlinf.models.embodiment.openvla_oft import get_model

    cfg = cfg if cfg is not None else model_cfg(weights)
    model = get_model(cfg, torch_dtype=torch.bfloat16).cuda().eval()
    set_unloaded(model, required=bool(cfg.get("add_value_head", False)))
    return model


def _image(rng, height, width):
    """A camera-like frame: a smooth random field (scene-scale structure) plus pixel noise, as uint8 HWC."""
    coarse = rng.random((height // 16, width // 16, 3))
    smooth = np.kron(coarse, np.ones((16, 16, 1)))
    noise = rng.normal(0.0, 0.08, (height, width, 3))
    return (np.clip(0.15 + 0.7 * smooth + noise, 0.0, 1.0) * 255).astype(np.uint8)


def observation(seed, index, batch=8, size=IMAGE_SIZE):
    """Request `index` of the stream `seed`: a fresh, deterministic batch of env observations (never repeated), in
    the LIBERO env's format (torch CPU tensors; float64 states become float32 as rlinf.envs.utils.to_tensor does)."""
    rng = np.random.default_rng([int(seed), int(index), int(batch)])
    main = np.stack([_image(rng, size, size) for _ in range(batch)])
    wrist = np.stack([_image(rng, size, size) for _ in range(batch)])
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


def request(model, obs):
    """One rollout step, end to end (the timed and the judged unit): host actions plus the GPU-side result."""
    with torch.no_grad():
        actions, result = model.predict_action_batch(env_obs=obs, **SAMPLING)
    if isinstance(actions, np.ndarray):   # the rollout worker converts numpy actions with torch.from_numpy
        actions = torch.from_numpy(actions)
    return actions.detach().cpu(), result


def replay(model, result):
    """The trainer's recompute of the rollout (default_forward), forward only."""
    with torch.no_grad():
        out = model(forward_inputs=dict(result["forward_inputs"]), compute_logprobs=True, compute_entropy=True,
                    compute_values=False, use_cache=False, **TRAIN_KWARGS)
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
