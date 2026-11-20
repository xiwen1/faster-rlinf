"""Build the acceptance candidates for gr00t-rl from the reference tree (formulation only; never shown to the agent).

    python make_accept.py <reference tree> <out dir>

Each candidate is a copy of the reference with one known edit (all in
rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py). Expected verdicts:
  a_unmodified     check --quick PASS (N0); full check UNAVAILABLE (no_benefit)
  b_entry_point    ERROR at gate 0 (default_forward signature changed)
  c1_skip_layers   the rollout's denoising steps run only the first half of the 16 DiT blocks: faster; INCORRECT
                   (gate 1 timed outputs, gate 2 records)
  c2_extra_rng     one extra draw from the CUDA RNG right after the initial noise: INCORRECT at gate 2 (RNG
                   fingerprint, every later noise draw)
  c3_timed_only    c1's shortcut only inside the timing worker: check --quick PASS; full check INCORRECT at gate 1
  c4_py_random     one extra draw from Python's random after the stochastic-step draw: INCORRECT at gate 2 (RNG
                   fingerprint, later requests' stochastic step)
  c4b_np_random    one draw from numpy's global RNG (never drawn from by the reference): INCORRECT at gate 2 (RNG)
  c5_replay_only   rollout untouched, the trainer's recompute (default_forward) computes the log-probability in fp32
                   and rounds it to bf16 once: INCORRECT at gate 2 (replay tensors)
  c6_logprob_only  actions and chain untouched, the rollout's prev_logprobs computed in fp32 and rounded to bf16
                   once: INCORRECT at gate 2
  c7_perturb       the normalized actions scaled by 1+2^-20 before unnormalization (a few fp32 ulps of the host
                   actions): INCORRECT at gate 2
  d_equivalent     the per-env stochastic-step index built with expand+clone instead of repeat (same values, dtype,
                   layout): check --quick PASS
  f_slower         5 ms sleep per request: check --quick PASS; full check UNAVAILABLE
"""
import shutil
import sys
from pathlib import Path

G = "rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py"

LOOP = "        # Run denoising steps.\n        for idx in range(num_steps):\n"
AFTER_LOOP = "        x_0 = x_t\n        chains = torch.stack(chains, dim=1)\n"
INIT_NOISE = """        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
"""
DENOISE_INDS = "        denoise_inds = denoise_inds[None].repeat(batch_size, 1)\n"
ROLLOUT_LOGPROB = "            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)\n"
REPLAY_LOGPROB = "            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)\n"
ACTIONS_FLOAT = "        actions = actions.float()\n"
STATES = '        env_obs["states"] = env_obs["states"].to(torch.bfloat16)\n'


def half_blocks(condition):
    return (f"""        _all_blocks = self.model.transformer_blocks
        if {condition}:
            self.model.transformer_blocks = _all_blocks[: len(_all_blocks) // 2]
""" + LOOP, "        self.model.transformer_blocks = _all_blocks\n" + AFTER_LOOP)


def edit(tree, old, new):
    p = tree / G
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"{G}: anchor found {s.count(old)} times: {old[:60]!r}")
    p.write_text(s.replace(old, new))


def main():
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    def fresh(name):
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(ref, d, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        return d

    fresh("a_unmodified")
    edit(fresh("b_entry_point"), "        use_cache: bool = False,\n        **kwargs,\n    ) -> dict[str, Any]:",
         "        use_cache: bool = False,\n        fast: bool = True,\n        **kwargs,\n    ) -> dict[str, Any]:")
    for name, cond in (("c1_skip_layers", "True"),
                       ("c3_timed_only", "__import__('sys').argv[0].endswith('rl_worker.py')")):
        d = fresh(name)
        before, after = half_blocks(cond)
        edit(d, LOOP, before)
        edit(d, AFTER_LOOP, after)
    edit(fresh("c2_extra_rng"), INIT_NOISE, INIT_NOISE + "        torch.randn(1, device=device)\n")
    edit(fresh("c4_py_random"), DENOISE_INDS, DENOISE_INDS + "        random.random()\n")
    edit(fresh("c4b_np_random"), DENOISE_INDS, DENOISE_INDS + "        np.random.random()\n")
    edit(fresh("c5_replay_only"), REPLAY_LOGPROB,
         "            log_probs = self.get_logprob_norm(\n"
         "                chains_next.float(), x_t_mean.float(), x_t_std.float()\n"
         "            ).to(chains_next.dtype)\n")
    edit(fresh("c6_logprob_only"), ROLLOUT_LOGPROB,
         "            log_prob = self.get_logprob_norm(\n"
         "                x_t.float(), x_t_mean.float(), x_t_std.float()\n"
         "            ).to(x_t.dtype)\n")
    edit(fresh("c7_perturb"), ACTIONS_FLOAT, "        actions = actions.float() * (1 + 2**-20)\n")
    edit(fresh("d_equivalent"), DENOISE_INDS,
         "        denoise_inds = denoise_inds[None].expand(batch_size, -1).clone()\n")
    edit(fresh("f_slower"), STATES, '        __import__("time").sleep(0.005)\n' + STATES)
    print("built", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
