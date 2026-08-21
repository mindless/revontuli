# Corruption above ~3553 video tokens — investigation state

**Question:** why does MiniMax H3 (antirez/h3.c, Metal/MPSGraph, AMD RX 6900 XT eGPU
`amdgpu_gfx1030`, Intel Mac mini, 16 GiB VRAM) produce corrupt video at longer lengths?

## The data

Video tokens = `(W/16/2) * (H/16/2) * latent_t`, where
`latent_t = ((frames-5)/17)*5 + 2` (`h3_host.c:23`), `H3_VAE_SPATIAL_RATIO 16`
(`h3_host.h:11`), `patch_size [1,2,2]` (DiT config). At 608x352 that is 19x11 = **209
tokens per latent frame**. Same prompt, seed 42, 4 steps in every run.

| config | tok/frame | latent_t | video tokens | verdict | wall |
|---|---|---|---|---|---|
| 22f @608x352 | 209 | 7 | 1463 | **clean** (reference, md5 886b40d1…) | 4.0 min |
| 39f @608x352 | 209 | 12 | 2508 | **clean** | 5.5 min |
| 56f @608x352 | 209 | 17 | **3553** | **clean** | 7.1 min |
| 73f @608x352 | 209 | 22 | 4598 | **RUNNING** | ~9 min |
| 90f @608x352 | 209 | 27 | 5643 | untested | — |
| 107f @608x352 | 209 | 32 | **6688** | **CORRUPT** — flat brown mush, tile seams | 13.2 min |
| 124f @608x352 | 209 | 37 | 7733 | **CORRUPT** — dark noise + `encode` 1582 s | 39.2 min |
| 107f @320x192 | 60 | 32 | **1920** | **clean but degraded** — smear, colour fringing | 4.6 min |

**The decisive result is the last row.** Same 107 frames, same latent_t 32, only
resolution reduced -> clean. So the trigger is **not temporal length**. It tracks
**total token count**, threshold between 3553 and 6688. Note **4096 = 2^12** sits inside
that window; the running 73f (4598) straddles it.

## Ruled out, and how

1. **Short learned temporal positional table** (the top-ranked hypothesis from both
   Gemini 3.1 Pro and GPT-5). **Refuted from the weights, zero generations:** all 535
   FL2VA DiT tensors contain exactly one positional tensor, `rope.inv_freq [16]`.
   `time_embedder` is the diffusion timestep embedder (256 -> 5376), not positional.
   RoPE with on-the-fly frequencies extrapolates by construction — no structural ceiling.
2. **Divisibility by 22 / "trained 22-frame decoder chunk"** (`h3.c:861`). 39 is not a
   multiple of 22 and is clean; 107 is not a multiple of 22 and is corrupt.
3. **A different code path at longer lengths.** Dispatch counts are identical in all
   runs: `submissions=560 direct=628 linear=800 attention=200`. Caveat: identical
   *counts* do not prove identical *kernel variants* — see the NAX note below.
4. **Token reduction / pooling.** Off by default (`configure_token_reduction`,
   `h3_dit.c:378` needs `H3_TOKEN_REDUCTION`), so the `reduced_*` RoPE path is unused.
5. **NAX row-count kernel selection** (`h3_gpu.m:3005-3024`, thresholds at rows 2048 and
   3072). Gated on `gpu.tensorOpsEnabled`, which is M5-only; this device reports Apple
   GPU family 0 / Metal 4 no. Dead code here. **Consequence: with NAX off, all 800
   `linear` and 200 `attention` dispatches are MPSGraph.**
6. **Stale `MPSGraphTensorData` cache.** `h3_gpu_graph_data` (`h3_gpu.m:160`) compares
   `graphDataShape == shape` by pointer and, on mismatch, allocates and returns a FRESH
   data object. Pointer-identity only causes cache misses (a perf cost), never wrong
   contents.
7. **Host memory pressure** for the 124f encode blowup. 64 GiB RAM, 0.00 M swap used,
   149 k pageouts. Not swapping.

## Live hypotheses

- **A. MPSGraph produces wrong results on gfx1030 above some shape threshold** (linear
  or SDPA). Fits identical dispatch counts with wrong output, and fits 124f's encode
  blowup as the same compiler hitting a worse heuristic. Currently the strongest.
- **B. Cross-commit hazard from MPSGraph's `commitAndContinue`.** MPSGraph commits the
  root buffer it is handed and returns a fresh one, which h3 adopts
  (`h3_gpu_adopt_mps_root`). 800 roots retired per denoise at 107f vs 400 at 22f. If an
  intermediate is assumed live across a commit boundary, longer graphs cross it more.
- **C. Stride / index truncation at larger shapes** (GPT-5 #5).

## Ready-made A/B switches (no rebuild needed)

- `H3_DISABLE_HEAD_MAJOR_SDPA=1` — attention layout variant. Tests A for SDPA.
- `H3_REUSE_MPS_COMMAND=0` — fresh `MPSCommandBuffer` per encode. Tests B directly.
- `H3_MPS_GQA`, `H3_DISABLE_GRAPH_DATA_CACHE`, `H3_METAL_FLUSH_EVERY`,
  `H3_METAL_MAX_INFLIGHT`. Full list: `grep -oE 'getenv\("[A-Z0-9_]+"' src/h3.c/h3_gpu.m`

## Tooling notes

- **`xctrace` is now available** (Xcode installed, licence accepted), so Metal System
  Trace is possible. `bin/trace-denoise.sh` exists but its export/parse step has never
  been executed — verify table names against the TOC it dumps.
- `make` works again. Current `h3` is up to date and backed up at
  `/tmp/h3-verified-backup`. **The toolchain switched from Command Line Tools clang to
  Xcode clang**, so any rebuild must be re-verified byte-identical against
  `outputs/h3-egpu-test.mp4`, md5 `886b40d10c5a83fb01393a4c62bdfcbd`.

## Verification protocol — every corrupt file exited 0 and passed ffprobe

    ffprobe -v error -f lavfi "movie=<f>,signalstats" \
      -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0

Healthy mean YAVG 106-120; corrupt 36-65. **YAVG is a screen, not a verdict** — always
extract a grid and look:

    ffmpeg -i <f> -vf "select='not(mod(n\,18))',scale=304:176,tile=3x2" -frames:v 1 g.png

## Peer consult (already done, $0.282)

`logs/review-frames/` — Gemini 3.1 Pro and GPT-5 usable; Kimi K3 returned
`finish_reason: length` with `content: null` (no answer, reasoning trace only, $0.126
wasted). Their top hypothesis is the one refuted in item 1. Their durable contribution
was method: prefer checks that read weights or dump intermediates over checks that need
a generation. Kimi's salvaged trace derived the VAE chunking structure (decode window 7
latents -> 22 pixel frames, 2-latent/5-pixel overlap, advancing 17 -> hence `5+17k`),
which is why the **VAE chunks but the DiT sees the whole sequence**.

---

## UPDATE (Fable session, same day)

**Reframe: the attention scores matrix is the only seq² object in the model.**
Every linear/MLP intermediate is linear in seq (~0.3 GB at 107f — nowhere near a
limit). SDPA internally materializes [1, 56, seq, seq], and that is MPSGraph-internal:
h3's `peak_live_bytes` cannot see it, which is why the DiT phase read 2.289 GiB while
the scores tensor alone wants 4.66 GiB (BF16) at 107f. Candidate thresholds where
56·seq²·bytes crosses a hard limit:

| candidate | breaks at seq | 73f (seq≈5100) | probe verdict |
|---|---|---|---|
| FP32 scores vs 3.5 GiB maxbuf | 4096 exactly | corrupt | — |
| BF16 scores, int32 byte offset | ~4379 | corrupt | — |
| BF16 scores vs 3.5 GiB maxbuf | ~5793 | clean | in play |
| int32 element count | ~6193 | clean | in play |

**73f @608x352 is CLEAN** (YAVG 107.6, grid verified — best-looking clip yet, 3.04 s,
now the longest usable video). Kills the first two rows.

**The standalone probe reproduces a discontinuity** (`src/sdpa_probe.m`,
`logs/52-sdpa-probe.log`): h3's exact SDPA graph (native MPSGraph op, BF16, [1,56,N,128],
MPSCommandBuffer encode, private storage), spot-checked against a double-precision CPU
reference:

    seq=5100  spot_max_err=0.000082   gpu-wall=0.221s
    seq=6200  spot_max_err=0.007345   gpu-wall=0.188s   <- 90x error jump
    seq=6700  spot_max_err=0.007112   gpu-wall=0.255s

With random ±1 inputs true outputs are ~0.013–0.04, so 0.007 absolute = 20–50%
relative error. gpu-wall is NON-monotonic across the cliff (6200 cheaper than 5100
despite 1.5x the FLOPs) — MPSGraph switches execution path above a size threshold and
the new path is wrong. Probe control at seq=1463 (verified 22f shape): err 1.8e-4, ok.
Probe encode stays ~7 ms even at 6700 — the 124f encode blowup does NOT reproduce in
isolation; it needs h3's full memory context. Separate symptom.

**In flight:** `--spike` mode (planted high-logit key per head; a dropped/misweighted
key region fails with O(1) error exactly where planted → maps the broken region) at
5100 + 6700, plus fine sweep 5400–6250 to localize the cliff. `logs/53-sdpa-cliff.log`.

**If confirmed, the fix is a head-split workaround in `h3_gpu_sdpa`:** run SDPA in two
28-head chunks (or four of 14) whenever 56·seq²·2 approaches the limit — halves the
internal scores buffer, mathematically identical output, negligible cost. Then rebuild
(toolchain is now Xcode clang — MUST re-verify 22f bit-identity, md5 886b40d1…) and
re-run 124f for the 5-second deliverable.

## ROOT CAUSE (established, same session)

**MPSGraph's native SDPA silently corrupts attention when its internal scores tensor
[heads, seq, seq] crosses 2^31 ELEMENTS.** Measured to the single integer on the
RX 6900 XT: seq 6192 (56·6192² = 2,147,171,942 < 2^31) is clean at 0/168 spike spots;
seq 6193 (2,147,865,510 > 2^31) breaks. `logs/54-sdpa-headmap.log`.

- Failure signature: broken heads return NEAR-UNIFORM attention — no NaN, no error,
  values in range. Peaked attention becomes a uniform context mixture -> flat mush.
- Damage spreads from BOTH ends of the head range as seq grows (int32 wraparound:
  overflowing high heads corrupt themselves and clobber low heads):
  seq 6193 -> {0}; 6250 -> {0,1,55}; 6700 -> {0..8, 48..55} = 17 heads.
- At 124f (seq≈7995) ~44 of 56 heads break -> "dark noise" vs 107f's 17-head "mush".
  One mechanism, severity gradient. (The 124f encode blowup is still separate.)
- Head-major and row-major layouts fail IDENTICALLY -> no env-knob escape.
- Random-data probing showed only err ~0.007 because uniform-attention output is
  nearly indistinguishable from correct near-uniform softmax on random data. The
  planted-spike probe (28-sigma logit spike per head) unmasked it at err ~1.0.

**Fix, validated then shipped:** split SDPA into head chunks inside the same graph
(slice dim 1 -> SDPA per chunk -> concat) whenever heads·seq² > 1.9e9. Probe-validated
0/168 broken at seq 6193/6700/7733/7995 (`logs/55-sdpa-splitfix.log`); ported to
`h3_gpu_sdpa_graph` (`h3_gpu.m`), covering causal and non-causal paths. Below the gate
the graph is byte-identical to before. `make test` -> ok: 1769 checks.

**Verification in flight:** 22f bit-identity vs md5 886b40d1… (mandatory: toolchain is
now Xcode clang AND the SDPA builder changed), then the 124f (5.17 s) generation.

Uncommitted in src/h3.c: the mps-gpu counter work + this SDPA fix. Owner has not asked
for a commit.

## RESOLVED

22f bit-identity held through toolchain switch + counters + SDPA fix (md5 886b40d1…
exact). The fixed 124f run: denoise encode 1582.7 s -> 0.388 s (the planner blowup was
the same >2^31 tensor — one root cause, all symptoms), denoise wall 36.9 -> 12.0 min,
total 14.2 min. Output: 124 frames / 5.175 s, YAVG 114.2 (highest of any run), grid
verified clean. `outputs/fox-5sec-fixed.mp4`. Every frame length 5..362 is now
expected to work; the 2^31 gate re-engages automatically for any (heads, seq) that
needs it.
