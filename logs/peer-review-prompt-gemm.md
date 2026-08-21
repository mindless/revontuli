# Review request: denoise optimization plan for MiniMax H3 on RX 6900 XT eGPU (Metal/MPSGraph)

## Context

MiniMax H3 (18.5B joint audio-video DiT, antirez/h3.c C+Metal port) running on a
2018 Intel Mac mini (i7-8700B, 64 GB RAM) with an AMD RX 6900 XT (gfx1030, 16 GiB
VRAM) in a Thunderbolt 3 enclosure. macOS 15.7, MPSGraph for matmul/SDPA, custom
Metal shaders for elementwise/norm ops. Weights BF16, streamed per-block from SSD
(model is 134 GiB; DiT weights 38.5 GiB re-read and re-uploaded over Thunderbolt
EVERY denoising step — 144 GiB per 4-step generation, double-buffered slots,
hidden behind compute today).

Production workload (a game asset pipeline): 640×832, 39 frames, 4 steps,
first+last-frame conditioned. Joint sequence ≈ 7500 rows (6240 video + 1040 cond
+ text + audio). 50 DiT blocks. HIDDEN=5376, HEADS=56, HEAD_DIM=128, FFN=14336.
Per block 4 GEMMs: qkv [seq,5376]×[5376,21504], out [seq,7168]×[7168,5376],
fc1 [seq,5376]×[5376,28672], fc2 [seq,14336]×[14336,5376].

Current: 39f generation ≈ 1051 s total, of which denoise ≈ 823 s.
22f profile: denoise wall 537 s, measured GPU 359 s (178 s = GPU idle / CPU-side
gaps), Qwen text encoder 140 s (cacheable), DiT load 41.5 s (cacheable).

## New measurements (steady-state, on the actual card, rows=7488)

GEMM microbenchmark, per-block totals (4 GEMMs), 200 block-passes per generation:

| variant | ms/block | s/generation | TFLOP/s |
|---|---|---|---|
| MPSGraph BF16, weight [N,K] transposed in-graph (PRODUCTION) | 2903 | 580 | 2.0 |
| MPSGraph BF16, weight pre-transposed [K,N] | 554 | 111 | 10.4 |
| MPSGraph FP16, in-graph transpose | 210 | 42 | 27.4 |
| MPSGraph FP16, pre-transposed | 177 | 35 | 32.6 |
| MPSGraph FP32, in-graph transpose | 295 | 59 | 19.6 |
| MPSMatrixMultiplication FP16 (transposeRight) | 208 | 42 | 27.8 |
| MPSMatrixMultiplication FP32 (transposeRight) | 293 | 59 | 19.7 |
| h3 portable 16×16 BF16 tile shader | 1912 | 382 | 3.0 |

So MPSGraph's BF16 matmul with an in-graph transpose is the production
bottleneck: 2.0 TFLOP/s on a 23 TFLOPS FP32 / 46 TFLOPS FP16 card. Note FP16 is
barely hurt by the in-graph transpose (42 vs 35 s) while BF16 is destroyed by it
(580 vs 111 s).

SDPA (MPSGraph native, BF16, [1,56,7488,128], with our >2^31 head-split fix):
0.323 s GPU + 0.236 s CPU encode per dispatch × 200 dispatches ≈ 65 s GPU +
47 s CPU encode per generation.

Structural: the streamed-block loop currently does, per block (200×/generation):
encode compute → commit → waitUntilCompleted → join SSD-prefetch thread → new
command buffer. So the GPU idles during each block's CPU-side MPSGraph encode
(~0.2-0.9 s) and every commit round-trips Thunderbolt. That matches the 178 s
wall-vs-GPU gap at 22f.

Also known: macOS GPU watchdog kills long-running command buffers on this eGPU;
we bound our chains (flush every 4 dispatches, ≤2 in flight) and one generation
at a time is a hard rule. maxBufferLength = 3.50 GiB. simdgroup_matrix does not
compile to a working pipeline on gfx1030 (SC compilation failure), so no custom
tensor-tile kernels. MPSMatrixMultiplication FP16/FP32 work fine. BF16→FP16
weight conversion is exact within FP16 range (BF16 has fewer mantissa bits);
activations are BF16 today.

## Planned changes, in order

1. **Software-pipeline the streamed block loop** (bit-identical): stop
   waitUntilCompleted per block; keep 2 blocks in flight (weight slots
   triple-buffered or event-synced so a slot isn't overwritten while its block
   is still executing). Goal: hide CPU encode + Thunderbolt round-trips; denoise
   wall → max(GPU compute, SSD read, TB upload).
2. **FP16 GEMM path** (env-gated until quality-validated): keep the BF16 SSD
   stream + upload unchanged; after each block's weights land in VRAM run a tiny
   BF16→FP16 conversion kernel (~3 ms/block); matmul in FP16 (input cast
   in-graph BF16→FP16, output cast back to BF16). Per the table this takes GEMM
   from 580 s to ~42 s per generation. Fallback if FP16 shows artifacts:
   pre-transposed BF16 (111 s) via a GPU transpose kernel after upload.
3. Expected post-fix denoise ≈ GEMM 42 + SDPA 65 + other ~25 ≈ 130-150 s,
   bounded below by SSD read (144 GiB @ ~1.4-2.6 GiB/s = 55-100 s, page-cache
   assisted on repeat steps) and TB3 upload (144 GiB @ ~2.2 GiB/s ≈ 65 s), all
   overlapped. 39f total ≈ 1051 → ~200 s warm-session, ~400 s cold.
4. Warm daemon mode for the game rail (the interactive REPL already caches DiT
   prep, text embedding, VAE/vision conditioning ≈ 200 s per roll; we drive it
   over stdin).

## Questions for you (rank + be quantitative; correct anything wrong)

1. Risks in the FP16 GEMM plan on gfx1030/MPSGraph: does MPSGraph FP16 matmul
   accumulate in FP32 on this hardware? Known DiT/video-diffusion failure modes
   from FP16 weights+activations at these magnitudes (RMSNorm'd activations,
   AdaLN modulation, SwiGLU FFN, 4-step Euler flow matching)? Our spot checks
   show rel_err ~2e-4 vs FP64 reference on random data. What should we A/B to
   catch real regressions (we have seed-pinned seam/identity-IoU/body-plan
   gates)?
2. The SDPA cost (65 s GPU + 47 s CPU encode per generation) after the GEMM fix
   becomes ~40% of denoise. Options we see: FP16 SDPA (numerics? the scores
   tensor is 5.9 GiB and we already head-split above 2^31 elements), flash-style
   custom kernel (no simdgroup_matrix on this GPU — is a plain-SIMD flash
   attention worth it on RDNA2?), or leave it. Rank.
3. Pipelining depth vs the macOS eGPU watchdog: with ≤2 block-granularity
   command buffers in flight (each ~0.3-0.5 s GPU post-fix), any known footguns
   with MPSGraph's commitAndContinue interleaving across two logical streams, or
   with blit/compute overlap on a single hardware queue over Thunderbolt?
4. Anything cheaper we're missing that doesn't change model outputs
   at all (dispatch layout, MPSGraph compilation options, weight layout,
   MTLHeap, residency, queue priorities)? The 22f reference must stay
   bit-identical for the pipelining-only change.
5. Is re-uploading 38.5 GiB per step over TB3 avoidable at 16 GiB VRAM without
   quality loss? (int8 weights exist upstream for M5 but int8 matmul kernels
   there use tensorOps we don't have; MPSGraph int8 matmul on gfx1030 —
   worth probing?)
