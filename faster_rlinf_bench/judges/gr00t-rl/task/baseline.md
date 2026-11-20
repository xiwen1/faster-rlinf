# Baseline: the reference GR00T N1.5 rollout step (measured at formulation, 2026-11-10/11)

All numbers are the unchanged reference (`targets/reference`) on one B200, with the judge's own commands.
Machine-readable values: `task/timing_ref.json`.

## Latency

Interleaved scheme of the check (two sessions, four reference processes each, 8 windows x 25 requests at batch 8 and
5 windows x 30 requests at batch 1 per process; `host-b`, a quiet host):

| batch | P50 (mean of the 8 process P50s) | process P50s | reference-vs-reference "speedup" of the two sessions |
|---:|---:|---|---|
| 8 | **103.2 ms** | 100.3 - 107.6 ms | 0.995, 0.999 |
| 1 | 68.4 ms | 67.0 - 71.7 ms | 0.986, 0.966 |

- A no-op change measures a speedup of about 1.00 at batch 8 (acceptance: 0.9995). The benefit bar is 1.10.
- Window spread (median absolute deviation of the window P50s / median) at batch 8: 0.2 - 0.9% on a quiet host. The
  judge refuses a run whose reference spread exceeds 4% (candidate 6%), or whose reference P50 is more than 15% off
  103.2 ms.
- This request is mostly host time (below): on a busy host (NUMA node ~39% idle) the same reference measured 150 - 625
  ms windows at batch 8 and spreads of 12 - 24%. Such a run is refused (INVALID: band or spread). Run it again.
- Startup: build ~26 s, warmup (20 requests at each batch size) ~6.7 s. Peak memory allocated 6.65 GiB.

## Where the time goes (batch 8, `bench.py profile`, 5 requests, quiet host)

- About 2,900 kernels per request, **~31.5 ms of kernels per request**; under the profiler the GPU is busy 25% of the
  window (the profiler inflates the host side). Against the 103 ms P50, the GPU is busy roughly 30% of a request.
- Kernel time by kind (per request): elementwise / copy / cat / index kernels ~13.5 ms (1,665 launches), GEMMs
  (`nvjet_sm100_*`) ~13.3 ms (771 launches), attention (flash-attention 2 in the backbone, cuDNN SDPA in the DiT)
  ~2.4 ms, layer norms ~1.5 ms.
- Host side (~70 ms per request): Python dispatch and kernel launches of the backbone and of the 4 DiT passes
  (`cudaLaunchKernel` + `cuLaunchKernelEx` ~15 ms of CPU per request), the modality transforms on the CPU (crop,
  antialiased bilinear resize ~2.5 ms, normalization, concatenation), the Eagle processor's tokenization, and the
  numpy action unnormalization.

## Targets

TARGET **2.0x**, MINIMUM **1.5x** (batch 8, `performance.speedup` of the check).
