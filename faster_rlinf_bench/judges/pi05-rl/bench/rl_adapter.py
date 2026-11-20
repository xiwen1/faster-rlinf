"""numerics adapter for the pi0.5 RL rollout task.

One "step" of a case is one rollout request (bench/workload.py: request) followed by the trainer's recompute of it
(workload.replay: Pi0RL.default_forward on the request's own forward_inputs). The judge records, per step, every
output of the rollout (host actions, prev_logprobs, prev_values, model_actions and each forward_inputs tensor), the
replayed logprobs/values/entropy, the RNG fingerprints before and after (torch CPU and CUDA, Python random, numpy),
the math mode, and the invariants below. Forward-call counts are off: a candidate may legitimately run the model
through compiled or captured graphs that do not fire module hooks. Buffers are not judged: the outputs are.
"""
from collections import defaultdict

from bench import workload
from numerics.adapter import Adapter, Objects
from numerics.record import seed_everything


class Pi05RLAdapter(Adapter):
    name = "pi05-rl"

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
        return Objects(models={"pi05": self.model}, count_forwards=False, capture_outputs=())

    def batch(self, step, rank):
        return workload.observation(self.case["obs_seed"] + self.offset, step, batch=self.case["batch_sizes"][step])

    def step(self, step, batch):
        actions, result = workload.request(self.model, batch)
        replayed = workload.replay(self.model, result)
        return {k: v.detach() for k, v in workload.flatten(actions, result, replayed).items()}

    def invariants(self):
        m = self.model
        cfg = m.rl_cfg
        inv = {"num_steps": int(m.num_steps), "action_horizon": int(m.action_horizon),
               "action_dim": int(m.action_dim), "action_chunk": int(m.action_chunk),
               "action_env_dim": int(m.action_env_dim), "training": bool(m.training),
               "noise_method": str(cfg.noise_method), "noise_level": float(cfg.noise_level),
               "joint_logprob": bool(cfg.joint_logprob), "add_value_head": bool(cfg.add_value_head),
               "value_after_vlm": bool(cfg.value_after_vlm), "train_expert_only": bool(cfg.train_expert_only),
               "global_step": int(m.global_step)}
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
