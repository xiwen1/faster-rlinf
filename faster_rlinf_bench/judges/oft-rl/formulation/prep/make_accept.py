"""Build the acceptance candidates for oft-rl from the reference tree (formulation only; never shown to the agent).

    python make_accept.py <reference tree> <out dir>

Each candidate is a copy of the reference with one known edit (all in
rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py). Expected verdicts:
  a_unmodified     check --quick PASS (N0); full check UNAVAILABLE (no_benefit)
  b_entry_point    ERROR at gate 0 (default_forward signature changed)
  c1_skip_layers   the rollout runs only the first half of the LLM layers: faster; INCORRECT (gate 1 timed outputs,
                   gate 2 records)
  c2_extra_rng     one extra draw from the CUDA RNG right after the sampling (this request's outputs unchanged):
                   INCORRECT at gate 2 (RNG fingerprint, later requests' samples)
  c3_timed_only    c1's shortcut only inside the timing worker: check --quick PASS; full check INCORRECT at gate 1
  c4_py_random     one draw from Python's random (never drawn from by the reference): INCORRECT at gate 2 (RNG)
  c4b_np_random    one draw from numpy's global RNG: INCORRECT at gate 2 (RNG)
  c5_replay_only   rollout untouched, the trainer's recompute (default_forward) logprobs scaled by 1+2^-20:
                   INCORRECT at gate 2 (replay tensors)
  c6_logprob_only  actions untouched, the rollout's prev_logprobs scaled by 1+2^-20: INCORRECT at gate 2
  c7_temperature   the rollout's temperature scaled by 1+2^-20 (a tiny numeric perturbation of the logits):
                   INCORRECT at gate 2
  d_equivalent     the per-row Python list lookup of the action bin centers replaced by one vectorized numpy
                   indexing (same float64 values): check --quick PASS
  f_slower         5 ms sleep per request: check --quick PASS; full check UNAVAILABLE
"""
import shutil
import sys
from pathlib import Path

OFT = "rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py"

LLM_CALL = """        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,"""
AFTER_LLM = "        # Extract hidden states for action tokens\n"


def half_layers(condition):
    return (f"""        _all_layers = self.language_model.model.layers
        if {condition}:
            self.language_model.model.layers = _all_layers[: len(_all_layers) // 2]
""" + LLM_CALL, "        self.language_model.model.layers = _all_layers\n" + AFTER_LLM)


SAMPLE = """                probs_flat, num_samples=1, replacement=True
            )  # [B * act, 1]
"""
BINS = """        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )  # [B, dim]"""
BINS_EQUIV = """        normalized_actions = np.asarray(
            self.bin_centers[discretized_actions]
        )  # [B, dim]"""


def edit(tree, old, new):
    p = tree / OFT
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"{OFT}: anchor found {s.count(old)} times: {old[:60]!r}")
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
    edit(fresh("b_entry_point"), "        use_cache: Optional[bool] = None,\n        **kwargs,\n    ):",
         "        use_cache: Optional[bool] = None,\n        fast: bool = True,\n        **kwargs,\n    ):")
    for name, cond in (("c1_skip_layers", "True"),
                       ("c3_timed_only", "__import__('sys').argv[0].endswith('rl_worker.py')")):
        d = fresh(name)
        before, after = half_layers(cond)
        edit(d, LLM_CALL, before)
        edit(d, AFTER_LLM, after)
    edit(fresh("c2_extra_rng"), SAMPLE, SAMPLE + "            torch.randn(1, device=probs_flat.device)\n")
    edit(fresh("c4_py_random"), SAMPLE, SAMPLE + "            __import__('random').random()\n")
    edit(fresh("c4b_np_random"), SAMPLE, SAMPLE + "            np.random.random()\n")
    edit(fresh("c5_replay_only"), "                logits=action_logits, target=action_tokens\n            )\n",
         "                logits=action_logits, target=action_tokens\n            )\n"
         "            logprobs = logprobs * (1 + 2**-20)\n")
    edit(fresh("c6_logprob_only"),
         "        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)\n",
         "        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)\n"
         "        chunk_logprobs = chunk_logprobs * (1 + 2**-20)\n")
    edit(fresh("c7_temperature"), '            processed_logits_tensor = logits_tensor / kwargs["temperature"]\n',
         '            processed_logits_tensor = logits_tensor / (kwargs["temperature"] * (1 + 2**-20))\n')
    edit(fresh("d_equivalent"), BINS, BINS_EQUIV)
    edit(fresh("f_slower"), '        do_sample = kwargs.pop("do_sample")\n',
         '        __import__("time").sleep(0.005)\n        do_sample = kwargs.pop("do_sample")\n')
    print("built", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
