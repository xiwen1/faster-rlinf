"""LIBERO environment benchmark + bitwise recorder for RLinf / faster-rlinf.

Runs the RLinf ``LiberoEnv`` exactly as the env worker builds it (same Hydra config, optional
``RecordVideo`` wrapper), drives it with a seeded stream of random action chunks, and writes

* per-phase wall-clock timings (reset, chunk_step, video flush), and
* a SHA-256 digest of every observation / reward / termination / truncation tensor,

so that two code trees can be compared bit for bit (``env_compare.py``) and timed on the same
action stream.  Run with ``PYTHONPATH=<tree>``; the tree under test is whatever ``rlinf`` resolves to.

Mode ``profile`` instead times the pieces of one in-process OffScreenRenderEnv
(create / reset / set_init_state / step with and without camera rendering).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


def digest(x):
    h = hashlib.sha256()
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
        return h.hexdigest()[:24]
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        return h.hexdigest()[:24]
    if isinstance(x, dict):
        return {k: digest(v) for k, v in sorted(x.items()) if k != "task_descriptions"} | (
            {"task_descriptions": hashlib.sha256("\n".join(x["task_descriptions"]).encode()).hexdigest()[:24]}
            if "task_descriptions" in x
            else {}
        )
    if isinstance(x, (list, tuple)):
        return [digest(v) for v in x]
    if isinstance(x, (int, float, bool, np.generic)):
        return repr(x)
    return repr(type(x))


def compose_cfg(tree, config_name, log_path, overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=f"{tree}/examples/embodiment/config", version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=[f"runner.logger.log_path={log_path}"] + list(overrides))
    return cfg


def build_env(cfg, num_envs, save_video):
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    env = LiberoEnv(cfg=cfg.env.train, num_envs=num_envs, seed_offset=0, total_num_processes=1, worker_info=None)
    if save_video:
        from rlinf.envs.wrappers import RecordVideo

        env = RecordVideo(env, cfg.env.train.video_cfg)
    return env


def action_stream(seed, num_envs, chunk, scale):
    rng = np.random.default_rng(seed)
    while True:
        a = rng.uniform(-scale, scale, size=(num_envs, chunk, 7)).astype(np.float32)
        a[..., -1] = np.where(rng.uniform(size=(num_envs, chunk)) < 0.5, -1.0, 1.0)
        yield torch.from_numpy(a)


def run_bench(args):
    tree = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["rlinf"].__file__)))
    os.makedirs(args.out, exist_ok=True)
    cfg = compose_cfg(tree, args.config_name, args.out, args.override)
    chunk = int(cfg.actor.model.num_action_chunks)
    steps_per_epoch = int(cfg.env.train.max_steps_per_rollout_epoch) // chunk
    if args.chunks:
        steps_per_epoch = args.chunks
    t = time.perf_counter()
    env = build_env(cfg, args.num_envs, args.save_video)
    t_init = time.perf_counter() - t
    acts = action_stream(args.action_seed, args.num_envs, chunk, args.scale)
    rec = {"tree": tree, "config": args.config_name, "num_envs": args.num_envs, "chunk": chunk,
           "chunks_per_epoch": steps_per_epoch, "epochs": args.epochs, "save_video": args.save_video,
           "init_s": t_init, "epochs_data": []}
    for ep in range(args.epochs):
        e = {"steps": []}
        t = time.perf_counter()
        obs, info = env.reset()
        e["reset_s"] = time.perf_counter() - t
        e["reset_obs"] = digest(obs)
        step_t = []
        prof = None
        if args.cprofile and ep == args.epochs - 1:
            import cProfile
            prof = cProfile.Profile(); prof.enable()
        for c in range(steps_per_epoch):
            a = next(acts)
            t = time.perf_counter()
            obs_list, rew, term, trunc, infos = env.chunk_step(a)
            step_t.append(time.perf_counter() - t)
            if args.digest_all or c == steps_per_epoch - 1 or c % 8 == 0:
                e["steps"].append({"c": c, "obs": digest(obs_list[-1]),
                                   "obs_all": digest(obs_list) if args.digest_all else None,
                                   "rew": digest(rew), "term": digest(term), "trunc": digest(trunc)})
            else:
                e["steps"].append({"c": c, "obs": digest(obs_list[-1]), "rew": digest(rew),
                                   "term": digest(term), "trunc": digest(trunc)})
        if prof is not None:
            import pstats
            prof.disable()
            pstats.Stats(prof).sort_stats("tottime").print_stats(30)
        e["chunk_step_s"] = step_t
        e["chunk_step_sum_s"] = float(np.sum(step_t))
        t = time.perf_counter()
        fl = getattr(env, "flush_video", None)
        if callable(fl):
            fl()
        e["flush_s"] = time.perf_counter() - t
        rec["epochs_data"].append(e)
        print(f"epoch {ep}: reset {e['reset_s']:.2f}s  chunk_step sum {e['chunk_step_sum_s']:.2f}s "
              f"(mean {1000*np.mean(step_t):.1f} ms/chunk)  flush {e['flush_s']:.2f}s", flush=True)
    t = time.perf_counter()
    env.close() if hasattr(env, "close") else None
    rec["close_s"] = time.perf_counter() - t
    with open(os.path.join(args.out, "env_bench.json"), "w") as f:
        json.dump(rec, f)
    print("wrote", os.path.join(args.out, "env_bench.json"))


def run_profile(args):
    tree = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["rlinf"].__file__)))
    cfg = compose_cfg(tree, args.config_name, args.out, args.override)
    from rlinf.envs.sim.libero import libero_env as le

    orig = le.LiberoEnv._init_env
    le.LiberoEnv._init_env = lambda self: None
    host = le.LiberoEnv(cfg=cfg.env.train, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    le.LiberoEnv._init_env = orig
    params = host.get_env_fn_params()[0]
    seed = params.pop("seed")
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.venv import _set_camera_rendering as _scr

    res = {}
    t = time.perf_counter(); env = OffScreenRenderEnv(**dict(params)); env.seed(seed); res["create"] = time.perf_counter() - t
    t = time.perf_counter(); env.reset(); res["reset"] = time.perf_counter() - t
    init = host._get_reset_states(env_idx=np.array([0]))[0]
    t = time.perf_counter(); env.set_init_state(init); res["set_init_state"] = time.perf_counter() - t
    z = np.zeros(7); z[-1] = -1
    n = args.profile_steps
    t = time.perf_counter()
    for _ in range(n):
        env.step(z)
    res["step_render_ms"] = 1000 * (time.perf_counter() - t) / n
    _scr(env, False)
    t = time.perf_counter()
    for _ in range(n):
        env.step(z)
    res["step_norender_ms"] = 1000 * (time.perf_counter() - t) / n
    _scr(env, True)
    t = time.perf_counter(); env.reset(); res["reset_again"] = time.perf_counter() - t
    t = time.perf_counter(); env.close(); res["close"] = time.perf_counter() - t
    if args.cprofile:
        import cProfile
        import pstats

        t = time.perf_counter(); env = OffScreenRenderEnv(**dict(params)); env.seed(seed)
        pr = cProfile.Profile(); pr.enable()
        env.reset(); env.set_init_state(init)
        for _ in range(15):
            env.step(z)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
        pr = cProfile.Profile(); pr.enable()
        for _ in range(n):
            env.step(z)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "profile"])
    ap.add_argument("--config-name", default="libero_object_ppo_openpi_pi05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=0, help="chunk steps per epoch (0 = full episode)")
    ap.add_argument("--save-video", type=int, default=1)
    ap.add_argument("--digest-all", type=int, default=0)
    ap.add_argument("--action-seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--profile-steps", type=int, default=50)
    ap.add_argument("--cprofile", type=int, default=0)
    ap.add_argument("override", nargs="*")
    args = ap.parse_args()
    import rlinf  # noqa: F401  (resolve the tree under test)

    run_bench(args) if args.mode == "bench" else run_profile(args)
