"""numerics adapter for the OpenVLA-OFT RL rollout task.

One "step" of a case is one rollout request (bench/workload.py: request) followed by the trainer's recompute of it
(workload.replay: default_forward on the request's own forward_inputs). The judge records, per step, every output
of the rollout (host actions, prev_logprobs, prev_values and each forward_inputs tensor: input_ids, attention_mask,
pixel_values, action_tokens), the replayed logprobs and entropy, the RNG fingerprints before and after (torch CPU and
CUDA, Python random, numpy), the math mode, and the invariants below. Forward-call counts are off: a candidate may
legitimately run the model through compiled or captured graphs that do not fire module hooks. Buffers are not
judged: the outputs are.
"""
from collections import defaultdict

from bench import workload
from numerics.adapter import Adapter, Objects
from numerics.record import seed_everything


class OFTRLAdapter(Adapter):
    name = "oft-rl"

    def __init__(self, tree, weights, obs_seed_offset=0):
        self.tree, self.weights, self.offset = tree, weights, int(obs_seed_offset)
        self.model, self.case = None, None

    def configure(self, case):
        self.case = case

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError("no fp64 golden variant: the precision floor is N0 (bitwise)")
        self.model = workload.build_model(self.tree, self.weights)
        # Rollouts start from a defined RNG state whatever the build consumed.
        seed_everything(seed + 1)

    def objects(self):
        return Objects(models={"oft": self.model}, count_forwards=False, capture_outputs=())

    def batch(self, step, rank):
        return workload.observation(self.case["obs_seed"] + self.offset, step, batch=self.case["batch_sizes"][step])

    def step(self, step, batch):
        actions, result = workload.request(self.model, batch)
        replayed = workload.replay(self.model, result)
        return {k: v.detach() for k, v in workload.flatten(actions, result, replayed).items()}

    def invariants(self):
        m = self.model
        vb = m.vision_backbone
        inv = {"action_dim": int(m.action_dim), "num_action_chunks": int(m.num_action_chunks),
               "max_prompt_length": int(m.max_prompt_length), "vocab_size": int(m.vocab_size),
               "n_action_bins": int(m.config.n_action_bins), "unnorm_key": str(m.unnorm_key),
               "num_images_in_input": int(vb.get_num_images_in_input()), "num_patches": int(vb.get_num_patches()),
               "has_value_head": hasattr(m, "value_head"), "training": bool(m.training),
               "sampling": repr(sorted(workload.SAMPLING.items())),
               "train_kwargs": repr(sorted(workload.TRAIN_KWARGS.items()))}
        numel = defaultdict(int)
        for p in m.parameters():
            numel[str(p.dtype)] += p.numel()
        inv.update({f"param_numel/{k}": v for k, v in sorted(numel.items())})
        return inv

    def canonical(self, kind, name, tensor):
        return {} if kind == "buffer" else {name: tensor}

    def save_state(self, path):
        raise NotImplementedError("rollout cases start from the checkpoint; there is no state to save")

    def load_state(self, path):
        raise NotImplementedError
