"""Build the acceptance candidates for pi05-rln from the reference tree (formulation only; never shown to the agent).

    python make_accept.py <reference tree> <out dir>

Each candidate is a copy of the reference with one known edit (all in rlinf/models/embodiment/openpi/tasks/rl.py).
Expected verdicts:
  a_unmodified    check --quick PASS (N0); full check UNAVAILABLE (no_benefit)
  b_entry_point   ERROR at gate 0 (default_forward signature changed)
  c1_skip_suffix  denoising steps 1-3 reuse step 0's action-expert pass: faster, INCORRECT (gate 1 timed outputs,
                  gate 2 records)
  c2_extra_rng    INCORRECT at gate 2 (an extra draw from the CUDA RNG: RNG fingerprint)
  c3_timed_only   c1's shortcut only inside the timing worker: check --quick PASS; full check INCORRECT at gate 1
  c4_py_random    INCORRECT at gate 2 (a draw from Python's random, which the joint path never draws from)
  c5_replay_only  rollout untouched, the trainer's recompute (default_forward) values scaled by 1+2^-20: INCORRECT
                  at gate 2 (replay tensors)
  c6_logprob_only actions untouched, prev_logprobs scaled by 1+2^-20: INCORRECT at gate 2
  c7_noise_std    the noise head's standard deviation scaled by 1+2^-20 in the rollout: INCORRECT at gate 2
  d_equivalent    the per-step .item() host syncs replaced by one .tolist() each (same arithmetic): check --quick PASS
  f_slower        5 ms sleep per request: check --quick PASS; full check UNAVAILABLE
"""
import shutil
import sys
from pathlib import Path

RL = "rlinf/models/embodiment/openpi/tasks/rl.py"

SUFFIX = """            suffix_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)"""


def skip_suffix(condition):
    return f"""            if {condition}:
                suffix_act = self.run_suffix(
                    observation, x_t, t_tensor, kv_cache, prefix_mask
                )
            v_t = self.velocity_from_suffix(suffix_act).to(torch.float32)"""


LOOP_HEAD = """        for idx in range(num_steps):
            method = (
                rl_cfg.noise_method
                if idx == int(denoise_inds[0, idx].item())
                else "flow_ode"
            )
            t_val = float(timesteps[idx].item())"""
LOOP_HEAD_EQUIV = """        timesteps_host = timesteps.tolist()
        chosen_host = denoise_inds[0].tolist()
        for idx in range(num_steps):
            method = (
                rl_cfg.noise_method
                if idx == int(chosen_host[idx])
                else "flow_ode"
            )
            t_val = float(timesteps_host[idx])"""


def edit(tree, old, new):
    p = tree / RL
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"{RL}: anchor found {s.count(old)} times: {old[:60]!r}")
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
    edit(fresh("b_entry_point"), "    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):",
         "    def default_forward(self, forward_inputs: dict[str, torch.Tensor], fast: bool = True, **kwargs):")
    edit(fresh("c1_skip_suffix"), SUFFIX, skip_suffix("idx == 0"))
    edit(fresh("c2_extra_rng"), "        prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(observation)\n",
         "        torch.randn(1, device=device)\n"
         "        prefix_out, prefix_mask, kv_cache = self.build_prefix_cache(observation)\n")
    edit(fresh("c3_timed_only"), SUFFIX, skip_suffix(
        "idx == 0 or not __import__('sys').argv[0].endswith('rl_worker.py')"))
    edit(fresh("c4_py_random"), "            chosen_per_step = torch.arange(num_steps, device=device)\n",
         "            chosen_per_step = torch.arange(num_steps, device=device)\n            random.random()\n")
    edit(fresh("c5_replay_only"), '            "values": values.float(),',
         '            "values": values.float() * (1 + 2**-20),')
    edit(fresh("c6_logprob_only"), "        log_probs_picked = log_probs_picked.float().contiguous()\n",
         "        log_probs_picked = log_probs_picked.float().contiguous() * (1 + 2**-20)\n")
    edit(fresh("c7_noise_std"), "            x_t_prev = x_t\n            step_noise = torch.randn(",
         "            if noise_std is not None:\n                x_t_std = x_t_std * (1 + 2**-20)\n"
         "            x_t_prev = x_t\n            step_noise = torch.randn(")
    edit(fresh("d_equivalent"), LOOP_HEAD, LOOP_HEAD_EQUIV)
    edit(fresh("f_slower"), "        del kwargs\n        repacked = repack_env_obs(",
         "        del kwargs\n        __import__('time').sleep(0.005)\n        repacked = repack_env_obs(")
    print("built", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
