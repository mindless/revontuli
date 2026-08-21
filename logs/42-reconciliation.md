# Task 9 — the "~45 s unexplained gap" does not exist

**Date:** 2026-08-21
**Method:** re-read existing profile logs. No code change, no generation run.

## Result

The denoise phase reconciles exactly. Across **seven** independent runs:

```
wall = encode + command_wait + stream_unhidden_wait + ~0.019 s loop overhead
```

| run | wall | encode | command_wait | unhidden | residual | residual − unhidden |
|---|---|---|---|---|---|---|
| 41-overlap (HEAD) | 140.99 | 0.237 | 140.707 | 0.025 | 0.043 | 0.018 |
| 37-pagecache | 140.70 | 0.246 | 140.209 | 0.219 | 0.240 | 0.021 |
| 39-revert | 145.51 | 0.241 | 144.990 | 0.261 | 0.280 | 0.019 |
| 40-split | 138.98 | 0.249 | 136.216 | 2.495 | 2.514 | 0.019 |
| 36-batched | 151.18 | 0.247 | 145.880 | 5.033 | 5.052 | 0.019 |
| 38-mmap | 219.62 | 0.233 | 150.915 | 68.456 | 68.475 | 0.019 |
| 33-smoke-opt2 | 57.16 | 0.098 | 12.899 | 44.160 | 44.165 | 0.005 |

The ~0.019 s is `h3_gpu_begin` + `pthread_create` + loop bookkeeping over 200
block visits. There is no fourth serialization and nothing to instrument.

## Where the "45 s" came from

It was `141 − 96`: the phase wall minus `stream_read_seconds`. Both inputs were
misused.

1. `stream_read_seconds` (96.0 s) is wall time measured **inside the prefetch
   thread** (`job->seconds`, `h3_dit.c:667,692`). It is concurrent time and must
   never be subtracted from a phase wall. The only part of it that reaches the
   critical path is `stream_wait_seconds` = **0.025 s**.
2. `root-gpu` (10.06 s) is the documented floor. Using it as a GPU total is
   precisely the trap recorded as blocking constraint #1 — and the previous
   session's reconciliation model used it as a total anyway.

## Consequence 1 — optimization B is retired by measurement

The stream is fully hidden behind compute in the full-run regime, so reducing
bytes moved can only reclaim the unhidden 0.025 s.

Demonstrated directly: stream time varies **2.3×** across runs while
`command_wait` stays in a 136–151 s band, and the two are uncorrelated.

| stream read | GiB/s | command_wait |
|---|---|---|
| 96.0 s | 1.502 | 140.707 |
| 121.6 s | 1.186 | 140.209 |
| 123.3 s | 1.171 | 144.990 |
| 124.6 s | 1.158 | **136.216** |
| 142.9 s | 1.010 | 145.880 |
| 221.0 s | 0.653 | 150.915 |

The 124.6 s stream produced a *lower* wait than the 96.0 s stream. Making the
stream 49% faster (142.9 → 96.0) bought nothing end to end — which is also the
already-recorded outcome of optimization A, now explained rather than just
observed.

B was scoped to cut denoise traffic 24% (stream 96 → ~73 s). Still far under the
140.7 s compute wall, so still hidden. **Projected win: 0.025 s.** Do not build it.

Caveat on regime: in `33-smoke-opt2` (15 blocks gate-skipped) compute was 12.9 s
against a 58.1 s stream, and 44.2 s of stream *was* exposed. B would help a
configuration where compute is short and the stream is long. That is not the
shipping configuration.

## Consequence 2 — "GPU compute is 7% of the run" is wrong, and it retired real work

That figure was `root-gpu` summed across phases in 41-overlap:
3.604 (text) + 10.450 (DiT) + 0.445 (audio VAE) + 3.201 (video VAE) = **17.7 s**
of a ~240 s run. It is a sum of *floors*.

The same runs' `command_wait` — which `h3_gpu.h:31` states is the complete
turnaround measurement — totals
15.313 + 141.096 + 0.351 + 5.227 = **162.0 s**, i.e. **67%** of the run.

Evidence that `command_wait` tracks real GPU-side work rather than CPU idle:
moving activations into VRAM took it from 1803.69 s to 140.71 s (12.8×) while
root-gpu went 666.94 → 10.06 s. A counter that responds that sharply to a memory
placement change is measuring GPU work.

So true GPU busy time for the denoise is bounded: `[10.1 s, 140.7 s]`, and ~130 s
of it is MPSGraph child-buffer work that `gpu_seconds` structurally cannot see.

**The matmul rewrite and flash-attention kernel were abandoned on the 7% figure.**
That premise does not survive this reconciliation. The decision should be
reopened — but on a measurement, not on this argument.

## What to measure next

Narrow the `[10.1, 140.7]` bound for the 800 `linear` + 200 `attention` MPSGraph
dispatches in the denoise. `xcrun xctrace` is **not available** on this box
(Command Line Tools only, no full Xcode), so Metal System Trace needs an Xcode
install first. Cheaper alternative: time a standalone MPSGraph BF16 matmul at the
DiT's actual shapes and multiply by dispatch count — `report.md:302-305` already
records 41.7–45.9 GFLOP/s at 2048×3072×3072, which is ~1000× below this card's
peak and is the strongest available hint that the dispatches, not the bus, own
the remaining time.

## Unchanged

Output bit-identity is untouched: nothing was compiled or run.
Reference md5 `886b40d10c5a83fb01393a4c62bdfcbd`.
