# Peer review: ranked optimizations

First, two corrections to your premises, because they change the ranking:

**Correction 1: Your "59% waiting" is not streaming wait, and the synchronous blit is NOT your biggest problem.** Do the arithmetic: 144 GiB streamed over a 1.76 GB/s link is a hard floor of **82 s**. You measured 130 s of unhidden streaming wait. But the DiT phase has **1803 s of wait**. Streaming explains at most ~130 s of it. Something else is stalling ~1250 s in denoise — and the obvious suspect is the thing you listed last: **AdaLN in Shared memory, 496 MiB/block, multiplied by h3_linear's per-16-row weight reload.** If AdaLN modulation is computed per-token-block with M=256 rows, weight traffic is 496 MiB × 16 = 7.9 GiB per block over a 1.76 GB/s link = 4.5 s/block × 200 block-passes = **~900 s**. At M=1024 it's 3,600 s — so the truth is somewhere in that band depending on how AdaLN is dispatched. This is almost certainly your real 1803 s. Verify in 10 minutes: Instruments/Metal System Trace, or just log wall time inside the AdaLN dispatch per block.

**Correction 2: RDNA2 cache hierarchy.** gfx1030 (Navi 21): 32 KiB L0 vector cache per CU (well, per WGP: 128 KiB L1 per WGP), **4 MiB L2 total** (not "128 KB L2 per..." anything — you may be thinking of GCN), plus **128 MiB Infinity Cache** (L3). The 128 MB IC is actually your friend: a 294 MiB FC1 weight matrix doesn't fit, but a 220 MiB QKV nearly does, and all tile-level reuse within one matmul lands in L2/IC, so your 478 GB/s number understates effective bandwidth for tiled kernels.

**Correction 3: `simdgroup_matrix` / `simdgroup_multiply_accumulate` is Apple-silicon-only in MSL (requires `MTLGPUFamilyApple7+`). It will not compile for gfx1030.** Do not waste a day on it. On AMD Metal you get wave32 (exposed as simdgroups of 32), LDS ("threadgroup memory"), and scalar FMA. That's what you tile with.

**Correction 4: Your MPSGraph measurement of 41.7 GFLOP/s is almost certainly methodology error, not hardware truth.** That's 1/1000 of peak — you're measuring graph compilation, first-run allocation, or wall time including a synchronous round-trip. Re-measure steady-state (encode N=50 iterations in one command buffer, divide). That said, the conclusion "MPSGraph on AMD/x86 is poorly tuned" is *plausibly* right — Apple's x86 MPS backend has been maintenance-mode for years — so the action item below stands regardless.

---

## Ranked list

### 1. Kill the AdaLN Shared-memory catastrophe — est. win 900–1700 s (the largest single item)
**What's happening:** AdaLN output is 96,768 = 5376 × 18 values per block per step — i.e., 18 modulation vectors (6 for self-attn, 6 for cross/MLP structure, ×3 for scale/shift/gate — the standard DiT AdaLN-Zero layout). The *output* is 387 KB/block/step. The 496 MiB/block × 24.2 GiB you retained is the **input weight matrices**, precomputed-and-cached as if they were activations. This is the classic mistake: it's not an activation cache, it's weights, and weights belong on the SSD streaming path or in VRAM.

**Do this, in order of preference:**
- (a) **Fold AdaLN into the existing 2-slot streaming machinery.** 496 MiB/block fits trivially alongside the 770 MiB slot (1.27 GiB/slot total, ×2 slots = 2.5 GiB — fine). Same pread → staging → blit path you already wrote. This is ~50 lines, reuses tested code, and removes 200 × 496 MiB = 97 GiB of Shared-memory reads from the hot loop.
- (b) Even simpler if AdaLN's matmul is M=1 (conditioning vector × matrix, computed once per block per step, not per token): **run it on the CPU with Accelerate/cblas_sgemm.** 2 × 2688 × 96768 = 0.52 GFLOP per block — a 6C Coffee Lake does that in ~30 ms. Free the GPU weights entirely; stream from page cache to a pinned CPU buffer. Zero GPU memory, zero TB traffic except the pread from NVMe (which is 2-3 GB/s and does NOT share the TB3 link — important, see #8).
- Also **free the weight source buffers after precompute** regardless — retaining 24.2 GiB of host allocations is what's pushing your RSS to 34.7 GiB.

**Verify:** diff the modulation vectors before/after against the current build for seed 42, step 0, block 0 (max abs err < 1e-3 in fp32). **Effort: low (a) / very low (b).** Expected: this converts most of the 1803 s wait into either hidden prefetch or 200 × 30 ms of CPU time.

### 2. Async upload pipeline: dedicated blit queue + MTLSharedEvent — est. win ~130 s + full overlap of the 82 s streaming floor
You're right, and you've half-answered your own question. Structure it exactly like this:
- Two `MTLCommandQueue`s: compute (existing) and transfer (new). Metal on AMD/discrete supports concurrent blit and compute engines for host→device copies.
- Per slot, an `MTLSharedEvent` with monotonically increasing values: transfer buffer signals `event=2k+1` after blit completes; compute encoder calls `encodeWaitForEvent:` before dispatching block N+1; compute signals `event=2k+2` when *done with* slot N so the pread for N+2 can reuse the staging buffer safely.
- Remove the inline `commit`+`waitUntilCompleted` entirely. The CPU thread's job is: pread → encode blit → commit → immediately pread next slot. Never block on GPU.
- Raise slots from 2 to **4** (4 × ~1.3 GiB with AdaLN folded in = 5.2 GiB VRAM — you have the headroom; peak live is 1.678 GiB in DiT). Deeper pipeline = prefetch runs 2-3 blocks ahead and absorbs NVMe jitter.

**How much is recoverable:** the 130 s "unhidden wait" drops to ~0; the 82 s TB floor fully hides under 667 s of GPU compute. Net ~130–210 s. **Risk:** event ordering bugs = reading a half-uploaded block → subtle numerical garbage, not a crash. Verify with a checksum of the uploaded weight buffer on the first pass (read back one block, compare CRC against the file) and by confirming bitwise-identical output vs. current build at seed 42. **Effort: medium** (a day, mostly getting the event protocol right).

### 3. Rewrite `h3_linear_bf16` — est. win 300–500 s of GPU time in denoise + ~150 s in VAE
Your numbers convict this kernel independent of memory placement: FC1 at rows=1024 is 316 GFLOP in 0.252 s = **1.25 TFLOP/s on a card with 23 TFLOP/s FP32 peak** (and ~46 TFLOP/s FP16). Even in VRAM you're at 5% of peak. The per-16-row-block weight reload is a secondary sin (at 478 GB/s, 64× reload of 294 MiB costs ~39 ms — tolerable); the primary sin is the 16×16 tile itself: 256 threads each computing one output with no vectorized loads and no register blocking.

**Spec for gfx1030:**
- Output tile **64×64**, threadgroup of 256 threads (8 wave32s — occupancy matters more than elegance on RDNA2; you want ≥4 threadgroups/CU resident to hide LDS latency).
- Each thread computes a **4×4 register sub-tile** (16 accumulators), loading A and B as `float4`/`ushort4`. This gives you the classic 1/8 arithmetic-intensity amplification: 16 FLOPs per 2 loaded values.
- K-block of 16; A-tile 64×16 and B-tile 16×64 staged in threadgroup memory (2 KiB + 2 KiB in BF16 — tiny; RDNA2 has 128 KiB LDS per CU, use double-buffered tiles if you want, but single-buffer + barrier is already 5-10× over current).
- BF16→FP32 convert on load into registers; accumulate FP32. Weight conversion once, not per use.
- **Loop structure: iterate over M *inside* the kernel is wrong for your shapes; iterate threadblocks over (M/64 × N/64) and keep K innermost.** With N=28672 you get 448 threadgroups in N alone — plenty to fill 80 CUs even at M=256.
- Split-K: only worth it if M×N parallelism < ~2× CU count. At M=256, N=21504 you have 4×336=1344 blocks. Not needed. Skip it.

Expected: 8–15 TFLOP/s effective → FC1@1024 goes 0.252 s → 0.025–0.04 s. Across 800 denoise linears + 870 VAE linears, **~400–550 s**. **Risk:** index-math bugs at edges (5376 and 28672 are both divisible by 64 — 5376/64=84, 28672/64=448, 21504/64=336 — you're lucky, no predication needed if you assert divisibility at compile time). Verify: max-rel-err vs. current kernel < 2e-2 in BF16 terms, then end-to-end seed-42 frame diff. **Effort: medium-high** (this is the only genuinely hard item; a solid tiled GEMM is 300-400 lines of MSL and a day of tuning).

### 4. Video VAE: fix placement and tiling — est. win 400–700 s
956 s wall / 455 s GPU = 500 s of stall, and it's the same disease: **all activations are Shared**, so every conv/linear/attention intermediate round-trips TB3. Do:
- Make the VAE's activation buffers **Private** with the same gated allocator you wrote for weights. Peak live is 9.365 GiB of 15.98 — it fits.
- **Re-tile: 3×2 at 256 px for a 608×352 frame means 6 tiles with overlap for a frame that needs 608/256 ≈ 2.4 × 1.4.** You have 6.6 GiB of headroom. Try 2×2 at 320 px or a single 608×352 tile (peak scales ~linearly with tile area: full frame ≈ 9.365 × (608×352)/(6×256×256×overlap) ≈ well under 16 GiB). Fewer tiles = fewer attention re-computes of halo regions and fewer dispatches (1314 direct for 22 frames is absurd). Measure peak with `MTLHeap` currentAllocatedSize as you raise tile size; stop at 13 GiB.
- The 216 VAE attention ops: check what SDPA is actually doing in MPSGraph (see #5). At latent resolution 76×44=3344 tokens/frame, attention is small *unless* the graph materializes the 3344² score matrix in Shared memory — which, given your placement defaults, it does.

**Win reasoning:** halving the stall (250 s) + GPU speedup from Private activations + fewer tiles (~2× on 455 s GPU = ~200 s) ≈ **400-700 s**. **Effort: low-medium.** Verify: decode one frame tiled vs. untiled, PSNR > 40 dB at tile seams.

### 5. Replace MPSGraph on the hot paths with your own kernels, starting with SDPA — est. win 100–300 s, plus it unblocks #4
- Re-measure MPSGraph GEMM properly (steady-state loop) before you commit. If it comes back ≥2 TFLOP/s BF16, keep it for plain linears and only replace SDPA. If it's genuinely sub-100 GFLOP/s steady-state, rip it all out — you already have the GEMM kernel from #3, and MLP/QKV/out are just GEMMs.
- **Write a flash-attention kernel** rather than calling MPSGraph SDPA. DiT shapes: 56 heads × 128 dim, seq len a few thousand tokens — classic flash decode territory. Tile K/V in blocks of 64 in threadgroup memory, online softmax in registers, FP32 accumulation. On AMD without simdgroup_matrix this is still straightforward scalar FMA; 128-dim head = 4 float4 loads per row.
- Same kernel serves the VAE attention (3344 tokens), where the O(n²) score materialization in MPSGraph+Shared is likely a chunk of your 455 s.

**Effort: medium-high. Risk: numerical** — verify softmax rescaling with a reference numpy implementation on random tensors (max err < 1e-4 fp32).

### 6. Text encoder: stop reallocating, and don't run it on the GPU at all — est. win ~100 s
147 s for a one-shot text encoding is offensive. The repeating 250/80 MiB alloc-free per layer over Shared memory means every layer's weights are fetched from host RAM over TB3 *and* the allocator churns. Two options: (a) same resident-private treatment, reused buffers per layer (weights are per-layer same-shape — allocate two buffers and blit each layer in, no per-layer alloc); (b) since this runs once per run and CPU is otherwise idle, run Qwen on CPU/Accelerate entirely — a ~3B-ish encoder in BF16→FP32 on 12 threads does this in ~20-40 s. Either way ~100+ s back. Also check: with `reuse 1`, are you re-encoding text per step? You shouldn't be — cache the embedding.

### 7. Thunderbolt link at 20 Gb/s: cheap to check, bounded upside
`maxTransferRate = 5.0e9` is suspicious — that's the 40 Gb/s figure (5 GB/s ≈ 40 Gb raw minus encoding), suggesting the *port* negotiated 40 but the *link* is degraded, OR Metal is just reporting the controller spec. Your measured 1.76 GB/s ≈ 14 Gb/s effective, consistent with a 20 Gb/s link (~2.2 GB/s theoretical, ~80% efficiency). Fixes in order of likelihood: (1) **the cable** — passive TB3 cables >0.5 m and all cheap cables fall back to 20 Gb/s; buy a certified 40 Gb/s 0.5 m passive or 1-2 m active cable (~$40-60); (2) try the other TB3 port on the mini; (3) check for a daisy-chained device stealing lanes (Route String 3/103 suggests you're behind something). Upside: 1.76 → ~3.2 GB/s, halving the streaming floor 82 → 42 s and improving every Shared-memory straggler you haven't fixed yet. Real but small — do it because it's $50, not because it's strategic.

### 8. mmap + NoCopy staging; keep F_NOCACHE off, reconsidered
- Your DiT file is ~36-38 GiB read 4× (144 GiB total). With 64 GiB RAM and 34.7 GiB RSS, the page cache **can** hold most of it across steps 2-4 **if** you drop F_NOCACHE *and* fix #1/#6 (which free the 24.2 GiB AdaLN retention + text encoder churn that's pressuring the cache). Steps 2-4 then read at ~5-10 GB/s from RAM instead of ~2.8 GB/s NVMe. But note: **NVMe pread does not traverse the TB3 link**, so disk reads aren't your bottleneck — the 1.76 GB/s TB upload is. Page caching mainly helps if you adopt the async pipeline (#2), where it removes jitter.
- Do adopt **`mmap` + `newBufferWithBytesNoCopy`** for the staging buffer: it eliminates the pread memcpy entirely (kernel pages straight into the Metal-visible buffer). You verified the driver accepts non-page-aligned offsets — still, keep the buffer itself page-aligned and only let *weights* sit at arbitrary offsets within it. ~50 lines, saves ~140 GiB of memcpy (≈40-60 s of CPU and, more importantly, removes a serialization point in the prefetch thread).
- **MTLHeap: mostly irrelevant.** You're not hitting allocation-cost pathology; the slot buffers are 4 big allocations per step. A heap buys you suballocation tidiness and faster reuse, nothing measurable. Skip unless you do #4 and want to cap VAE growth. Low priority — you had this ranked too high in your head.

### 9. What you missed
- **Check whether MPSGraph is allocating its intermediates Shared through your code path.** If your placement patch only touched explicit `h3_gpu_tensor` allocations, every MPSGraph intermediate is still Shared. This could be a large silent contributor to both denoise and VAE stall and would move items #4/#5 up the ranking. Audit first (one hour, log `storageMode` of every buffer in a debug build).
- **The GPU watchdog timeouts you hit are a symptom of Shared-memory dispatch latency, and they'll return** if any path still dispatches a multi-hundred-ms kernel over Shared data. Your N=4/M=2 command-buffer bounding *increases* stall (each flush is a sync point where the GPU drains). After #1 and #2, relax to N=16/M=4 and re-measure — you'll recover another few percent.
- **Overlapping the 4 Euler steps is not safe** (sequential dependence), but **overlapping step-level weight prefetch across the step boundary is** — with 4 slots (#2), start loading step k+1's block 0 while step k's block 49 runs. Free ~1-2 s × 4.

## The "fundamentally better strategy" question
No — your instinct to ask is right but the answer is that streaming is already the correct architecture here; it's just not yet *implemented* asynchronously and is being sabotaged by Shared-memory stragglers. The budget math: DiT weights ≈ 38.5 GiB BF16, VRAM = 16 GiB. You could quantize to FP8 (→19 GiB, still doesn't fit) or INT8 with per-channel scales for just the 4 streamed matrices (→19 GiB, still doesn't fit; INT4 would, at real quality risk for a 4-step sampler). Not worth it. The floor is 144 GiB / TB-bandwidth = **82 s of upload per run**, fully hideable under 667 s of compute with a 4-slot pipeline. After #1-#5 you should land at roughly: denoise ~500-700 s, VAE ~250-400 s, text ~40 s, total **~15-20 min vs. your current 51 min**. That, not a different memory strategy, is the realistic 3×.

**Priority order if you only do three things: #1 (AdaLN), #2 (async pipeline), #3 (GEMM). Then re-profile before touching the VAE — the stall attribution will shift.**