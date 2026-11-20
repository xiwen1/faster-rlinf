"""The judged workload, shared by the timing worker and the numerics adapter (one code path for both).

One request = one RL rollout step of RLinf's rollout worker (rlinf/workers/rollout/hf/huggingface_worker.py) for a
batch of ManiSkill environments (PutOnPlateInScene25Main, as in maniskill_ppo_openpi_pi05.yaml): a numpy observation
dict (one 480x640 uint8 third-view camera image per env, the 8-dim WidowX joint state, an instruction) goes into
`Pi0RL.predict_action_batch(env_obs, mode="train")` under `torch.no_grad()` with the model in eval mode and no
generator (`rng=None`: the initial and per-step noise come from the global torch CUDA RNG; with flow_noise and joint
log-probabilities every denoising step is noised, its standard deviation predicted by the noise head, and Python's
`random` is not drawn from), and the env-space actions come back to the host as `actions.cpu()`. The other outputs (`prev_logprobs`, `prev_values`, `forward_inputs`, `model_actions`) stay on the
GPU, as in the worker, which keeps them for the trainer.

`replay` is the trainer's side of the same data: `Pi0RL.default_forward` (called through `model(forward_inputs=...)`,
as rlinf/workers/actor/embodied_fsdp_actor_worker.py does) recomputes the log-probabilities, values and entropy of a
rollout from its stored denoising chain. The judge runs it forward only, under no_grad, on the rollout's own
`forward_inputs`.

The SFT checkpoint has no value head and no noise head: their weights are set by `build_model` from a fixed generator
(by parameter name), so they do not depend on how much of the global RNG the model's construction consumes.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# ManiSkill PutOnPlateInScene25Main-style instructions ("put <object> on the plate"). All tokenize well under
# max_token_len=200, so the padded prompt shape is the same for every request.
PROMPTS = [
    "put carrot on plate",
    "put eggplant on plate",
    "put spoon on plate",
    "put the green cube on plate",
    "put the yellow cube on plate",
    "put banana on plate",
    "put the coke can on plate",
    "put the pepsi can on plate",
    "put the sponge on plate",
    "put the orange on plate",
    "put the bread on plate",
    "put the lemon on plate",
]

# Parameters that are not in the checkpoint (randomly initialized heads), set deterministically by name.
UNLOADED_PREFIXES = ("value_head.", "noise_head.")
UNLOADED_SEED = 20261107


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


def _set_unloaded(model):
    """Value- and noise-head weights from a fixed generator, per parameter name (fan-in scaled normal, zero biases)."""
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
    if not found:
        raise RuntimeError(f"no parameter starts with {UNLOADED_PREFIXES}: the value/noise heads are missing or renamed")
    return found


def build_model(tree, weights):
    """The tree's own factory with the frozen configuration; returns the eval-mode rollout model on cuda:0."""
    import_tree(tree)
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(model_cfg(weights), torch_dtype=None).cuda().eval()
    _set_unloaded(model)
    return model


def _image(rng, height, width):
    """A camera-like frame: a smooth random field (scene-scale structure) plus pixel noise, as uint8 HWC."""
    coarse = rng.random((height // 16, width // 16, 3))
    smooth = np.kron(coarse, np.ones((16, 16, 1)))
    noise = rng.normal(0.0, 0.08, (height, width, 3))
    return (np.clip(0.15 + 0.7 * smooth + noise, 0.0, 1.0) * 255).astype(np.uint8)


def observation(seed, index, batch=40, height=480, width=640):
    """Request `index` of the stream `seed`: a fresh, deterministic batch of env observations (never repeated)."""
    rng = np.random.default_rng([int(seed), int(index), int(batch)])
    return {
        "main_images": np.stack([_image(rng, height, width) for _ in range(batch)]),
        "states": rng.normal(0.0, 0.5, (batch, 8)).astype(np.float32),
        "task_descriptions": [PROMPTS[int(i)] for i in rng.integers(0, len(PROMPTS), batch)],
    }


def request(model, obs):
    """One rollout step, end to end (the timed and the judged unit): host actions plus the GPU-side result."""
    with torch.no_grad():
        actions, result = model.predict_action_batch(obs, mode="train")
    return actions.detach().cpu(), result


def replay(model, result):
    """The trainer's recompute of the rollout (default_forward), forward only."""
    with torch.no_grad():
        out = model(forward_inputs=dict(result["forward_inputs"]), compute_logprobs=True, compute_entropy=False,
                    compute_values=True, use_cache=False)
    return {k: v for k, v in out.items() if torch.is_tensor(v)}


def flatten(actions, result, replayed=None):
    """Every judged tensor of one request, by name."""
    named = {"actions": actions, "prev_logprobs": result["prev_logprobs"], "prev_values": result["prev_values"],
             "model_actions": result["model_actions"]}
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
