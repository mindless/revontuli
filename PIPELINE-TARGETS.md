# Pipeline targets — what TacticaArena needs from this box

## STATUS UPDATE (2026-08-22, optimization session) — targets HIT

The fork gained two engine changes and a daemon affordance. Measured on the
exact workload below (all quality gates passing — see "verified" table):

| roll | before | after (one-shot) | after (warm session) |
|---|---|---|---|
| 22f conditioned (Run A config) | 765 s | **317 s** | **126 s** (measured, REPL seed sweep) |
| 39f conditioned (Run B config) | 1051 s | **391 s** | ~195 s (denoise 156 + VAE decode 35) |
| — denoise share at 39f | ~823 s | **156 s** | 156 s |

1. **Pipelined streamed-block boundaries** (bit-identical, fox-reference md5
   exact): the per-block submit no longer drains the GPU queue; 3 streaming
   slots + per-slot fences keep upload/read/compute overlapped.
2. **`H3_FP16_GEMM=1`** (opt-in env var; the rail should set it): streamed
   DiT weights are rewritten as FP16 in VRAM after upload and qkv/out/fc1
   matmuls run at FP16 rate (26.5 vs 2.0 TFLOP/s measured); fc2 runs in FP32
   compute because its SwiGLU input carries activation outliers past FP16
   range (learned the hard way — black frames). Weights scale by 2⁻³ into
   FP16 with a saturation counter (`h3: FP16 GEMM ... saturated N`, always 0
   so far); inputs scale 2⁻³ at the cast, output rescales 2⁶ — power-of-two,
   zero mantissa cost. Verified on this workload: YAVG 131.2 vs 131.6
   reference, PSNR 39.5 dB / SSIM 0.992 vs the BF16 output at 22f, and the
   game's own gates at 39f: identitydecay PASS (first-vs-last IoU 0.9625 vs
   Run A's 0.9426), bodyplan PASS (drift 0.0167, tol 0.12).
3. **Warm daemon mode — WIRED INTO THE RAIL**: `roll.mjs --seeds 42,43,44`
   drives one live REPL session (spawn `h3 -d MODEL_DIR`, pipe stdin) that
   generates every seed with the DiT prep, text embedding, and anchor
   conditioning paid once; each seed still gets its own run dir, ledger
   entry, and gate artifacts. The REPL gained `!frames-dir DIR|clear` and
   flushes stdout per command (patch 0015) so the rail parses
   `Done -> <path> [<seconds>s]` live. One process = one generation at a
   time, which also satisfies the watchdog rule for free.

CLI contract: UNCHANGED (all flags as below still work; one-shot spawns behave
identically). Additions only: `H3_FP16_GEMM` env, `!frames-dir` REPL command,
`H3_SDPA_SPLIT_THRESHOLD` (pressure lever, leave unset), `H3_FP16_SKIP`
(debug bisect, leave unset), `H3_DEBUG_GPU_GAPS` (GPU idle diagnostics).
The SDPA int32-overflow split engages at this workload's shapes (56·7488²
crosses the gate) and the 39f output passed all gates through it.

Known cold-roll costs that remain (first roll of a session only): Qwen text
encode ~140 s cold / ~38 s with a warm page cache, DiT load ~41 s, VAE
decode ~35 s at 39f. Next lever if more is wanted: precompiled
MPSGraphExecutables (the FP16 graphs re-plan per encode at some shapes —
~65 s of CPU encode at 22f, though ~3 s at 39f).

---

Written 2026-08-22 after the first two production-shaped rolls through
`~/Sites/TacticaArena/workbench/vidgen/roll.mjs` (the game's rail; it invokes `h3`
directly). This is the workload to optimize for. Both runs below are the FL2VA
first/last-frame conditioning path — which **works** (first time it was ever
exercised in this repo) but is absent from `logs/` and the test suite.

## The workload (asset size / duration)

| parameter | value |
|---|---|
| mode | **image-conditioned, always** — `--first-frame` + `--last-frame` (same image for idle loops; a harvest roll uses first only). Text-to-video is never used. |
| canvas | **640×832** (the game's padded anchors; multiples of 32). A 672×896 tier may come later. Shrinking the canvas is NOT a useful lever (measured: resolution barely moves wall time). |
| frames | **39 = the standard clip** (idle loop, idle→walk, walk→idle). 56 for walk loops/flourishes. 22 only for probes. Grid 5+17k, 24 fps. |
| sampler config | 4 steps, seed pinned per class, `--layers 50 --reuse 1 --ssd-streaming` |
| audio | **never wanted.** The game strips it at mux. Anything spent computing audio is pure waste for this workload. |
| volume | ~6 rolls per class (5 clips + 1 harvest) × 25 classes ≈ **150 rolls per wave**, plus free re-rolls (owner rejects liberally). The dominant repeat pattern: **same anchor + same prompt, new seed**. |

## Measured baseline (this 6900 XT, this checkout)

**Run A — 22f, 640×832, 4 steps, first+last conditioned: 765 s total.**

| stage | wall | notes |
|---|---|---|
| Euler denoise (4 steps) | **537 s** (~134 s/step) | ~70% of the roll. The per-generation 144 GiB SSD stream lives here (~97 s at ~1.5 GiB/s). |
| Qwen text encoder | **140 s** | identical prompt re-rolled constantly — cacheable (the REPL already caches it) |
| DiT load | 41.5 s | cacheable across rolls (REPL) |
| video VAE decode | 23.4 s | |
| video VAE encode (2 anchors) | 14.2 s | conditioning-only cost |
| Qwen vision encode (2 anchors) | 4.9 s | conditioning-only cost |
| audio VAE decode | 1.2 s | wasted — see targets |

**Run B — 39f, same config: 1051 s total** (+17 frames ≈ +286 s; no stage profile
captured — the rail passes `--profile` from now on).

## Optimization targets, ranked by our math

1. **Denoise / the 144 GiB-per-generation stream.** Biggest single pool of time.
   (`H3_STREAM_PAGE_CACHE=1` on repeats? 64 GB host RAM. Anything cutting
   per-step weight traffic.)
2. **A scriptable warm-session mode.** The interactive REPL already caches DiT +
   text embedding + conditioning (~200 s of every one-shot roll). The game's rail
   is a one-shot spawn today; a batch/daemon interface it can drive (stdin
   commands are fine) turns a 150-roll wave from ~44 GPU-hours toward ~30 before
   any kernel work. Seed sweeps benefit most: with cached conditioning a re-roll
   pays only denoise + VAE decode.
3. **An audio-off switch.** Audio VAE decode is only 1.2 s, but if audio tokens
   can be dropped from the joint denoise itself, measure what that frees — for
   this workload it is all waste.
4. **Goal number: a 39f conditioned roll at or under ~5 min.** That makes a full
   roster wave ~12.5 GPU-hours instead of ~44.

## Do-not-break (the game's rail depends on these)

- CLI contract used by `roll.mjs`: `h3` run from `src/h3.c` (shader path is
  cwd-relative), `H3_METAL_DEVICE_NAME="Radeon RX 6900 XT"`, flags
  `-d -p -o --width --height --frames --steps --seed --layers --reuse
  --ssd-streaming --first-frame --last-frame --frames-dir --profile`.
  Change the contract if you must — but say so, loudly, in this file.
- **One generation at a time** (GPU watchdog: 2 channel restarts → macOS refuses
  all further submissions). Keep `H3_METAL_FLUSH_EVERY`/`H3_METAL_MAX_INFLIGHT`
  at their discrete-GPU defaults.
- First-frame fit is STRETCH, last-frame is COVER — the game supplies exact-size
  anchors and relies on identity fit. Don't regress it silently.
- The SDPA int32-overflow split (patch 0012) is gated by size, and 640×832×39f
  is a bigger attention shape than anything in `logs/` — verify the gate covers
  these shapes before trusting long-clip output.
- Every measured failure mode still produced a valid, ffprobe-clean mp4.
  Verify optimizations with the YAVG screen (healthy >100; corrupt band 36–65)
  AND a bit/metric A/B on a fixed seed — the game's gates passed Run A at
  seam 0.0155, identity IoU 0.9426, body-plan drift 0.0104; a faster build must
  not regress those numbers on the same seed.
