# Why does MiniMax H3 produce corrupt video above ~39 frames on this port?

I need ranked, falsifiable hypotheses with the specific check for each. Correct me
where my reasoning is wrong. Do not restate the brief.

## Setup

`antirez/h3.c` (C + Objective-C, Metal + MPSGraph — **not** PyTorch), patched to run
MiniMax H3 FL2VA on an **AMD Radeon RX 6900 XT eGPU** (`amdgpu_gfx1030`, 16 GiB VRAM,
Thunderbolt 3 at 20 Gb/s single-lane) attached to an Intel Mac mini 2018 (i7-8700B,
64 GiB RAM, macOS 15.7.9). BF16 is stored as `ushort` and accumulated in `float` in
the portable kernels; MPSGraph BF16 matmul is bit-exact on this device. All discrete-GPU
changes are gated on `hasUnifiedMemory` so Apple Silicon stays byte-identical to upstream.

The 22-frame configuration is fully verified: output md5 `886b40d10c5a83fb01393a4c62bdfcbd`,
stable across ~10 perf commits, `make test` → 1769 checks.

## The observation

Frame counts snap to **5 + 17k** (`h3_align_frame_count`, `h3_host.c:15`), bounded to
5..362 (`h3.c:514`). Latent temporal depth is
`h3_video_latent_t(f) = ((f - 5) / 17) * 5 + 2` (`h3_host.c:23`).
Resolution was **identical (608×352) in every run**, steps=4, seed=42, same prompt.

| frames | seconds | latent_t | verdict | total wall | denoise `encode` | mean YAVG |
|---|---|---|---|---|---|---|
| 22 | 0.92 | 7 | **clean** | 4.0 min | 0.245 s | 113.9 |
| 39 | 1.63 | 12 | **clean** | 5.5 min | 0.241 s | 107.6 |
| 107 | 4.46 | 32 | **CORRUPT** — flat brown mush, faint tile seams | 13.2 min | 0.283 s | 64.5 |
| 124 | 5.17 | 37 | **CORRUPT** — dark noise, tile artifacts | 39.2 min | **1582.687 s** | 36.6 |

Untested: 56 (latent_t 17), 73 (22), 90 (27).

## Constraints on the explanation — please respect these

1. **It is not a resource limit.** The 107-frame run was healthy on every counter:
   `encode` 0.283 s, denoise wall 632 s against ~620 s predicted by linear scaling from
   the clean 39-frame run, weight stream 1.486 GiB/s fully hidden (0.003 s unhidden),
   DiT peak 2.289 GiB, video VAE peak 9.365 GiB, exit 0, valid h264+AAC MP4, `ffprobe`
   clean. Only the pixels are wrong. **Timing health does not predict correctness here.**

2. **It is not divisibility by 22.** `h3.c:861` errors below 22 frames with "generation
   requires at least one trained 22-frame decoder chunk", which invites the guess that
   frame counts must be multiples of 22. But 39 is not a multiple of 22 and is clean,
   while 107 is not a multiple of 22 and is corrupt.

3. **Spatial tiling is not the variable.** Resolution is constant across all four runs,
   so the video VAE's spatial tile decomposition is identical — yet tile seams appear
   only in the long ones.

4. **The VAE decodes in temporal chunks.** Video VAE peak memory is **9.365 GiB in all
   four runs** despite 5.6× the frame count, so frames cannot all be resident.

5. **The corruption is not progressive.** At 107 frames, per-frame YAVG is uniformly
   ~65 from frame 0 (67.5, 66.5, 66.6, 65.4, 66.5, 64.8, 63.1, 66.4, 64.0, 61.2, 58.3).
   It is not a drift that accumulates over time — frame 0 is already wrong.

6. **The two corruption modes differ.** 107 is low-contrast flat mush; 124 is
   high-frequency dark noise. That may be one mechanism at two severities, or two bugs.

## The separate 124-frame anomaly

At 124 frames only, `encode` — main-thread wall time between `h3_gpu_begin` and
`[command commit]`, which includes all CPU time inside MPSGraph's
`encodeToCommandBuffer:` — explodes from 0.245 s to **1582.687 s**, i.e. **7.9 s of CPU
per block** over 200 block visits. Simultaneously the weight stream's measured rate
collapses from 1.486 to 0.090 GiB/s (same 144.272 GiB moved, so it is being descheduled,
not doing more work). Host RAM was **not** swapping: 64 GiB total, 0.00 M swap used,
149 k pageouts. GPU busy in that phase was 536 s of a 2217 s wall (24%), versus 74.7% at
107 frames and 99.9% at 39 frames.

Note this anomaly is absent at 107 frames, which is *also* corrupt — so the encode
blowup cannot be the cause of corruption.

## Relevant implementation detail

MPSGraph work is encoded into an `MPSCommandBuffer` wrapping h3's own root command
buffer. MPSGraph calls `commitAndContinue` internally, committing that root and handing
back a fresh one; the code adopts the new root via `gpu.command = command.rootCommandBuffer`.
I recently instrumented the retired roots (800 of them per denoise at 107+ frames, 400 at
22 frames) because their GPU timestamps were never being summed. Mentioning this in case
`commitAndContinue` interacts with cross-block state — e.g. a buffer whose contents are
assumed to persist across a commit boundary.

## What I am asking

1. **Ranked hypotheses** for corruption starting between latent_t 12 and 32, each with
   the cheapest decisive check. I care much more about the check than the theory.
2. Is there a plausible **trained maximum sequence length** in H3's DiT (3D RoPE /
   positional encoding over the temporal axis) that would produce exactly this — wrong
   from frame 0, non-progressive, degrading with length? If so, how would I detect it
   from weights on disk rather than by generating?
3. Would you expect a **numerical** failure (BF16 accumulation, attention softmax range,
   normalization over a longer sequence) to present as uniform low-contrast mush rather
   than NaN/black? What measurement distinguishes numerical blowup from a positional /
   indexing bug?
4. Anything that explains **both** the corruption and the 124-only encode blowup with one
   mechanism, or a solid argument that they are independent.
5. Is bisecting 56/73/90 (~9-11 min per run) worth it before instrumenting, or should I
   dump intermediate latents first? If the latter, what exactly should I dump and what
   statistic should I compute on it?
