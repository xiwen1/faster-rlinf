"""numerics adapter for the GR00T N1.5 RL rollout task.

One "step" of a case is one rollout request (bench/workload.py: request) followed by the trainer's recompute of it
(workload.replay: default_forward on the request's own forward_inputs, with the rollout's prev_logprobs, values on).
The judge records, per step, every output of the rollout (host actions, prev_logprobs, prev_values and each
forward_inputs tensor: chains, denoise_inds, state, state_mask, embodiment_id and the eagle_* inputs), the replayed
logprobs, prev_logprobs and values, the RNG fingerprints before and after (torch CPU and CUDA, Python random, numpy),
the math mode, and the invariants below. Forward-call counts are off: a candidate may legitimately run the model
through compiled or captured graphs that do not fire module hooks. Buffers are not judged: the outputs are.
"""
from collections import defaultdict

from bench import workload
from numerics.adapter import Adapter, Objects
from numerics.record import seed_everything


class GR00TRLAdapter(Adapter):
    name = "gr00t-rl"

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
        return Objects(models={"gr00t": self.model}, count_forwards=False, capture_outputs=())

    def batch(self, step, rank):
        return workload.observation(self.case["obs_seed"] + self.offset, step, batch=self.case["batch_sizes"][step])

    def step(self, step, batch):
        actions, result = workload.request(self.model, batch)
        replayed = workload.replay(self.model, result)
        return {k: v.detach() for k, v in workload.flatten(actions, result, replayed).items()}

    def invariants(self):
        m = self.model
        ah = m.action_head
        rl = ah.rl_config
        inv = {"output_action_chunks": int(m.output_action_chunks), "action_chunk": int(ah.action_chunk),
               "valid_action_dim": int(m.valid_action_dim), "action_horizon": int(ah.action_horizon),
               "action_dim": int(ah.config.action_dim), "num_inference_timesteps": int(ah.num_inference_timesteps),
               "num_timestep_buckets": int(ah.num_timestep_buckets), "padding_value": int(m.padding_value),
               "image_nums": int(m.image_nums), "embodiment_tag": str(m.embodiment_tag.value),
               "noise_method": str(rl.noise_method), "noise_level": float(rl.noise_level),
               "joint_logprob": bool(rl.joint_logprob), "ignore_last": bool(rl.ignore_last),
               "noise_anneal": bool(rl.noise_anneal), "safe_get_logprob": bool(rl.safe_get_logprob),
               "has_value_head": hasattr(ah, "value_head"), "value_vlm_mode": str(rl.value_vlm_mode),
               "select_layer": int(m.backbone.select_layer),
               "llm_layers": len(m.backbone.eagle_model.language_model.model.layers),
               "training": bool(m.training), "transforms_training": repr([(type(t).__name__, bool(t.training)) for t in m._modality_transform.transforms]),
               "rollout_kwargs": repr(sorted(workload.ROLLOUT_KWARGS.items())),
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
