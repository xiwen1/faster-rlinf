# Baseline: the reference rollout step (measured at formulation, 2026-11-08)

All numbers are the unchanged reference (`targets/reference`) on one B200 (`host-e`), with the judge's
own commands. Machine-readable values: `task/timing_ref.json`.

## Latency

Interleaved scheme of the check (two sessions, four reference processes each, 8 windows x 15 requests at batch 8 and
5 windows x 8 requests at batch 40 per process):

| batch | P50 (mean of the 8 process P50s) | process P50s | reference-vs-reference "speedup" of the two sessions |
|---:|---:|---|---|
| 8 | **217.1 ms** | 214.0 - 221.6 ms | 1.008, 1.021 |
| 40 | 445.5 ms | 440.2 - 460.2 ms | 1.019, 1.008 |

- A no-op change measures a speedup of about 0.98 - 1.02 (two checks of the unchanged reference at formulation:
  0.995 and 0.977 at batch 8). The benefit bar is 1.10.
- Window spread (median absolute deviation of the window P50s / median) of a quiet process: 0.3 - 1.0%.
- Hosts are shared: a disturbed host shows windows +30 - 40% slower; the check then comes back INVALID. Run it again.

## Where the time goes (batch 8, `bench.py profile`, 5 requests)

- About 6.9k kernels per request; the GPU is busy for only **22%** of the request (367 ms of kernels in a 1669 ms
  window of 5 requests). The rest is host-side launch overhead, Python, and synchronizations.
- 40% of kernel time is in `short` kernel classes (median under ~5 us), 10% in `under-filled` ones.
- Largest classes: the prefix (VLM) GEMMs (`nvjet_sm100_tst_128x256...`, `..._256x256...`: ~22% of kernel time),
  elementwise / copy kernels (~25% together), `triton_poi_fused_gelu_mul_0`, the RoPE triton kernel, softmax, layer
  norm, and a family of small GEMMs of the action expert and value head (some of them TF32 `cutlass3x ... tf32gemm`,
  which is the reference's own math mode and must stay).
- Full report: run `python bench/bench.py profile --tree targets/reference --out runs/<name>` yourself (round 1 starts
  with a profile anyway).

## Build and memory

- Build ~30.5 s, warmup (20 requests per batch size) ~30.5 s; peak allocated memory 11.95 GiB.

## Durations of the judge's commands (one GPU, after it is leased)

| command | duration |
|---|---|
| `check --quick` (structure + bitwise) | ~1 - 2 min |
| `check` (benefit with four processes + bitwise) | ~9 - 10 min |
| `profile` | ~2 min |
| `time` (one candidate process vs the frozen reference) | ~3 min (estimate) |

Waiting for an idle GPU can take much longer than the command itself (40 minutes at formulation).
