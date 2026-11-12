"""Bitwise check of the dense snapshot update in the patch weight syncer (CPU snapshot).

Builds the same patches twice, with PATCH_DENSE_SNAPSHOT_UPDATE on and off, over several
versions of a synthetic state dict (dense changes, sparse changes, +-0.0 flips, NaNs), and
compares every patch tensor and the final snapshots bit for bit. Run with PYTHONPATH=<tree>.
"""
import importlib
import os
import sys

import torch


def run(dense):
    os.environ["PATCH_DENSE_SNAPSHOT_UPDATE"] = "1" if dense else "0"
    import rlinf.hybrid_engines.weight_syncer.patch_syncer as ps

    importlib.reload(ps)
    g = torch.Generator(device="cuda").manual_seed(0)
    shapes = {"a.weight": (512, 384), "b.bias": (4096,), "c.w": (16, 32, 8), "d.w": (2048, 1024)}
    state = {k: torch.randn(s, generator=g, device="cuda", dtype=torch.float32) for k, s in shapes.items()}
    keys = list(shapes)
    snap, orig = {}, {}
    for k in keys:
        v2, o = ps.as_coo_2d_view(state[k])
        orig[k] = o
        snap[k] = v2.to("cpu", dtype=torch.bfloat16, copy=True).pin_memory()
    b = ps.PatchBuilder.create(snap, keys, keys, orig, torch.device("cpu"), torch.device("cuda"), True)
    init = {k: v.clone() for k, v in state.items()}
    patches, received = [], []
    for version in range(1, 6):
        for i, k in enumerate(keys):
            t = state[k]
            if (version + i) % 2:
                t.add_(torch.randn(t.shape, generator=g, device="cuda") * 0.05)  # dense change
            else:
                m = torch.rand(t.shape, generator=g, device="cuda") < 0.01  # sparse change
                t[m] += 1.0
            if version == 3:
                t.view(-1)[:7] = torch.tensor([0.0, -0.0, float("nan"), -0.0, 0.0, 1e-30, -1e-30], device="cuda")
            if version == 4:
                t.view(-1)[:7] = -t.view(-1)[:7]
        p = b.create_patch(state, version)
        patches.append([x.cpu().clone() for x in p.tensors()])
        received.append(p)
    # Receiver side: apply the same patches to a host-resident (offloaded) bf16 model.
    import asyncio

    from omegaconf import OmegaConf

    model = torch.nn.Module()
    for k in keys:
        model.register_parameter(
            k.replace(".", "_"),
            torch.nn.Parameter(
                ps.as_coo_2d_view(init[k])[0].to("cpu", torch.bfloat16).reshape(shapes[k]).clone(),
                requires_grad=False,
            ),
        )
    sd_keys = [k.replace(".", "_") for k in keys]
    syncer = ps.PatchWeightSyncer.__new__(ps.PatchWeightSyncer)
    cfg = OmegaConf.create({"type": "patch", "patch": {"snapshot_device": "cpu", "delta_encoding": True, "compression": "none"}})
    syncer = ps.WeightSyncer.create(cfg) if hasattr(ps, "WeightSyncer") else None
    if syncer is None:
        from rlinf.hybrid_engines.weight_syncer import WeightSyncer

        syncer = WeightSyncer.create(cfg)
    syncer.ordered_keys = sd_keys
    syncer.original_shapes = {sk: torch.Size(shapes[k]) for sk, k in zip(sd_keys, keys)}
    for p in received:
        async def recv(p=p):
            return p
        asyncio.run(syncer.apply(model, recv))
    applied = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return patches, {k: v.clone() for k, v in snap.items()}, applied


def bits(t):
    return t.reshape(-1).contiguous().view(torch.uint8) if t.dtype != torch.bool else t


pa, sa, aa = run(True)
pb, sb, ab = run(False)
for k in aa:
    if not torch.equal(bits(aa[k]), bits(ab[k])):
        print("applied", k, "differs")
        sa = None
print("receiver params compared:", len(aa))
ok = len(pa) == len(pb)
for i, (x, y) in enumerate(zip(pa, pb)):
    for u, v in zip(x, y):
        if u.shape != v.shape or u.dtype != v.dtype or not torch.equal(bits(u), bits(v)):
            ok = False
            print("patch", i, "differs", u.shape, v.shape)
for k in (sa or {"__fail__": None}):
    if sa is None or not torch.equal(bits(sa[k]), bits(sb[k])):
        ok = False
        print("snapshot", k, "differs")
print("patches", len(pa), "tensors per patch", [len(x) for x in pa])
print("BITWISE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
