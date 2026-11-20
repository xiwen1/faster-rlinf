# Baseline: the reference rollout step, flow_noise + joint (measured at formulation, 2026-11-09)

All numbers are the unchanged reference (`targets/reference`) on one B200 (`host-a`), with the judge's
own commands. Machine-readable values: `task/timing_ref.json`.

## Latency

Interleaved scheme of the check (two sessions, four reference processes each, 8 windows x 8 requests at batch 40 and
5 windows x 15 requests at batch 8 per process):

| batch | P50 (mean of the 8 process P50s) | process P50s | reference-vs-reference "speedup" of the two sessions |
|---:|---:|---|---|
| 40 | **472.0 ms** | 467.9 - 477.1 ms | 1.006, 0.991 |
| 8 | 217.4 ms | 208.9 - 227.5 ms | 1.051, 1.031 |

- A no-op change measures a speedup of about 0.99 - 1.01 at batch 40. The benefit bar is 1.10.
- Window spread (median absolute deviation of the window P50s / median) at batch 40: 0.6 - 2.7% on this host, which
  was busy (single windows +10 - 15%). The judge refuses a run whose reference spread exceeds 4% (candidate 6%).
- Hosts are shared: a disturbed host makes the check come back INVALID. Run it again.

## Where the time goes (batch 40, `bench.py profile`, 5 requests)

- About 5.95k kernels per request. The GPU is busy for **48%** of the profiled window (1408 ms of kernels in a 2935
  ms window of 5 requests, i.e. ~281 ms of kernels per request; the profiler's own overhead inflates the host side).
  The rest is host-side launch overhead, Python, the input transforms (one 480x640 image per environment) and
  synchronizations.
- 26% of kernel time is in `short` kernel classes, and 47% in `skewed` ones (the same kernel launched both on the
  large prefix and on the small action-expert tensors).
- Largest classes: the prefix (VLM) GEMMs (`nvjet_sm100_tst_128x256...NNT`, `..._256x256...NNT`: ~29% of kernel time),
  elementwise / copy kernels, `triton_poi_fused_gelu_mul_0`, the RoPE triton kernel, softmax, layer norm, the
  small GEMMs of the action expert, value head and noise head, and fp32 layers that run in TF32
  (`cutlass3x ... tf32gemm`, `sm80_xmma_fprop ... tf32`: the reference's own math mode, which must stay).
- The configuration pads the two missing camera views with zero images (`pi05_maniskill`): the prefix holds three
  image slots per environment, two of them masked.
- Full report: run `python bench/bench.py profile --tree targets/reference --out runs/<name>` yourself (round 1 starts
  with a profile anyway).

## Build and memory

- Build ~31 s, warmup (20 requests per batch size) ~43 s; peak allocated memory 11.96 GiB.

## Durations of the judge's commands (one GPU, after it is leased)

| command | duration |
|---|---|
| `check --quick` (structure + bitwise) | ~2 - 3 min |
| `check` (benefit with four processes + bitwise) | ~15 min |
| `profile` | ~2 min |
| `time` (one candidate process vs the frozen reference) | ~3 - 4 min (estimate) |

Waiting for an idle GPU can take much longer than the command itself.
