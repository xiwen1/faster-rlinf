# Baseline: the reference OpenVLA-OFT rollout step (measured at formulation, 2026-11-10/11)

All numbers are the unchanged reference (`targets/reference`) on one B200, with the judge's own commands.
Machine-readable values: `task/timing_ref.json`.

## Latency

Interleaved scheme of the check (two sessions, four reference processes each, 8 windows x 25 requests at batch 8 and
5 windows x 40 requests at batch 1 per process; `host-c`):

| batch | P50 (mean of the 8 process P50s) | process P50s | reference-vs-reference "speedup" of the two sessions |
|---:|---:|---|---|
| 8 | **91.2 ms** | 89.2 - 93.1 ms | 1.006, 1.001 |
| 1 | 44.0 ms | 39.9 - 49.9 ms | 0.909, 1.019 |

- A no-op change measures a speedup of about 1.00 at batch 8. The benefit bar is 1.10.
- Window spread (median absolute deviation of the window P50s / median) at batch 8: 0.5 - 3.3%. The judge refuses
  a run whose reference spread exceeds 4% (candidate 6%). Batch 1 is far more sensitive to host load (one busy
  session: 9 - 14%); it is reported, not judged.
- This request has a sizeable CPU part (image preprocessing): on a busy host the same reference measured 120 - 300 ms
  windows at batch 8. Such a run is refused (INVALID: band or spread). Run it again.

## Where the time goes (batch 8, `bench.py profile`, 5 requests, quiet host)

- About 2,355 kernels per request, **~71.6 ms of kernels per request**; under the profiler the GPU is busy 71% of the
  window (the profiler inflates the host side): against the 91.2 ms P50, the GPU is busy roughly 78% of a request.
- Host side (~20 ms per request): the image preprocessing on the CPU (two bicubic antialiased resizes per request,
  ~2.7 ms each, plus the float conversions and normalizations), the tokenizer, the kernel launches, and
  `cudaStreamSynchronize` waits (the host also waits for the GPU: the `.cpu()` of the sampled tokens).
- Kernel time by kind (per request): the large GEMMs of the Llama-2 7B and the vision backbones (`nvjet_sm100_*`)
  ~30 ms; attention (`cudnn ... sdpa ... flash_fprop`) ~5 ms; **elementwise / copy / concatenation kernels ~35 ms**
  (fp32 RMSNorm chains, RoPE including the `rotate_half` concatenations, dtype conversions, the SiLU-gated MLP
  products, the residual adds, the attention-mask expansion).
- Full report: run `python bench/bench.py profile --tree targets/reference --out runs/<name>` yourself (round 1 starts
  with a profile anyway).

## Build and memory

- Build 33 - 76 s (cold page cache: the 15 GB checkpoint), warmup (20 requests per batch size) ~8 s; peak allocated
  memory 17.4 GiB.

## Durations of the judge's commands (one GPU, after it is leased)

| command | duration |
|---|---|
| `check --quick` (structure + bitwise) | ~1 - 2 min |
| `check` (benefit with four processes + bitwise) | ~7 min to a verdict at the benefit gate; ~10 - 12 min with the bitwise gate |
| `profile` | ~2 min |
| `time` (one candidate process vs the frozen reference) | ~3 min (estimate) |

Waiting for an idle GPU can take much longer than the command itself.
