Ranked, concrete, quantitative optimizations. Each has: what to change, estimated wall win, risk/checks, effort.

1) Make streaming truly asynchronous and overlapped with compute (separate blit queue + events; triple-buffer the slots)
- What to change:
  - Create a dedicated MTLCommandQueue for blits (uploads).
  - For each slot, allocate: one Private weight buffer per streamed matrix (already done) plus 2–3 Shared staging buffers for read pipelining.
  - Use either:
    - MTLSharedEvent: after blit CB encodes copy and signalEvent(event, value++). Compute CB for the slot issue wait(event, value) before first kernel that consumes the weights; no commit/waitUntilCompleted on CPU.
    - Or MTLFence: if blit and compute go to the same queue, use updateFence on blit and waitForFence on compute. With two queues, use SharedEvent.
  - Increase slot count to 3 for DiT streaming (N=3) to hide SSD jitter and TB latency.
  - Bump in-flight command buffers to 3–4; keep individual dispatches < ~250 ms to avoid the watchdog; but do not flush after every 4 ops if that starves the pipe.
- Estimated win:
  - Your denoise wait is 1803.7 s. With ideal overlap, you should hide nearly all upload+I/O for block N+1 behind compute of block N. Realistically you’ll still pay occasional bubbles and first/last block. Expect 60–75% of 1803 s hidden → 1080–1350 s wall saved.
  - Side effect: lower GPU idle gaps for the video VAE too, if it currently serializes.
- Risk/checks:
  - Race if compute starts before upload completes; events/fences solve this. Verify by adding a checksum kernel on uploaded buffers in debug builds and compare to CPU checksum.
  - Starvation if compute queue drains before a slot is ready; instrument with per-slot timeline logs; ensure upload depth ≥ compute latency variance.
- Effort: 1–2 days. You already have double-buffered slots; wiring events and an extra queue is straightforward.

2) Fix Thunderbolt link to 40 Gb/s (and maintain it)
- What to change:
  - Use a certified 0.5–0.8 m 40 Gb/s TB3/4 cable. Avoid daisy chains. Use the Mac mini’s rear left port (often better signal). Reboot after cabling.
  - Verify in ioreg: Thunderbolt linkSpeed 40Gbps, PCIe upstream width x4, and in your app: MTLDevice.maxTransferRate should rise from 5.0e9 to ~1.0e10 bytes/s (Apple reports per-direction).
- Estimated win:
  - You measured GPU reads from Shared at 1.76 GB/s, which is consistent with a 20 Gb/s link. At 40 Gb/s you’ll see ~2.7–3.1 GB/s sustained to system memory and ~3.2–3.6 GB/s for DMA reads.
  - Any remaining Shared traffic (activations, any missed weights, mmap staging) shrinks by ~1.6–1.8x. Residual “unhidden” upload time and any page-cache hits also shrink.
  - Conservatively 150–300 s wall saved over the whole run after (1) is done; more if you leave significant Shared loads.
- Risk/checks:
  - Some enclosures negotiate 20 Gb/s when chained with high-bandwidth USB-C devices; eliminate chains. Verify again after wake from sleep.
- Effort: 0.5 day.

3) Remove the 24.2 GiB AdaLN Shared residency; replace with compute-on-the-fly or streamed Private micro-params per block
- What to change:
  - Root cause: keeping AdaLN outputs in Shared forces the GPU to read them over TB on every use. At 1.76–3.0 GB/s, that alone can dominate kernels that touch them repeatedly.
  - Determine what “precompute AdaLN” actually stores. With adaln_out_features=96768=18×5376, it is almost certainly the per-block projection of time/context embeddings into 18 modulation vectors of length 5376 (e.g., scales/biases for q,k,v,out, fc1/fc2, etc.). That means you do not need to retain a 2688×96768 matrix per block; you need its product with the single 2688-dim time/context vector for this run.
  - Replace “precompute and store 496 MiB per block” with either:
    a) Compute the 2688×96768 × t vector once per block per step on GPU (one matvec per block → 2×96768×2688 BF16 ops ≈ 0.52 GFLOP per block), store the resulting 18×5376 scalars/biases in a tiny Private buffer (~18×5376×2 bytes ≈ 189 KiB). Then discard the big matrix or never materialize it.
    b) If the 2688×96768 matrix is derived deterministically from base weights and text/time embedding, stream base weights and do the matvec on-the-fly similarly, not the full matrix.
  - Use a single custom matvec kernel (BF16 in, FP32 acc) or MPSMatrixMultiplication for (2688×96768)×(2688×1). Keep the output in Private and pass it by pointer to the DiT block kernels.
- Estimated win:
  - Eliminates 24.2 GiB of Shared reads throughout the denoise. If each DiT block uses a subset of these 18×5376 vectors many times per frame (typical AdaLN usage), you remove repeated TB reads and replace with on-chip/L2/LDS accesses.
  - Expect large speedup in both denoise and video VAE phases that touch AdaLN: 300–600 s wall saved (mostly by reducing GPU stalls waiting on Shared).
  - Frees 24.2 GiB system RAM → more page cache for weights; improves I/O overlap.
- Risk/checks:
  - Must validate the math equivalence. Dump original “precompute” outputs for a few prompts and compare per-channel scales/biases after refactor.
- Effort: 1 day once you locate the precompute site; the kernel is trivial.

4) Rewrite the linear BF16 kernels for RDNA2: larger tiles, LDS reuse across many rows, vectorized loads; optional split-K
- What to change:
  - Current h3_linear_bf16 reloads weight tiles per 16-row slice. On RDNA2 (gfx1030) each CU has 64 KiB LDS and a large L2 (~4 MiB shared, not 128 KiB; your 128 KiB figure is incorrect). Exploit LDS to reuse W across many rows:
    - Tile sizes: For GEMM A[M×K] × W[K×N] → C[M×N], with K=5376, N≈21–29k, M=rows (e.g., 256–1024).
    - Recommended block: TG tile Cb_M×Cb_N = 64×128 or 64×256, K-slice=64.
      - LDS footprint per K-slice: A_s: 64×64×2 B (=8 KiB) if you use BF16 packed, W_s: 64×128×2 B (=16 KiB) or 64×256×2 B (=32 KiB). Total 24–40 KiB, leaving LDS for double-buffering to overlap global loads with MACs.
    - Threadgroup: 256 threads (8 warps/32-thread simdgroups). Each simdgroup computes a 16×64 sub-tile using register blocking (e.g., 16×8 accumulators per thread group).
    - Vectorize global loads with ushort4/ushort8 into LDS; coalesce over N.
    - Convert BF16→f32 once on LDS->register read; keep accum in f32; downcast at store as needed.
    - Iterate K in 64-step chunks; double-buffer LDS with threadgroup_barrier.
    - Optional split-K across simdgroups for very large M when occupancy is low; reduce with atomic adds in FP32 or inter-block reduction buffers; try to avoid unless occupancy suffers.
  - Aim for persistent W-slab reuse across many M rows: process M in strides of 64–128 so each W_s load is reused 4–16× instead of reloaded every 16 rows.
- Estimated win:
  - Your Private-case timings: QKV 5376→21504, rows=1024: 0.199 s. Rough FLOPs ~ 2×M×K×N = 2×1024×5376×21504 ≈ 2371 GFLOP. That implies 2371/0.199 ≈ 11.9 TFLOP/s achieved today. That’s already decent but memory-bound behavior likely limits larger N. With tuned tiling and reuse, 18–24 TFLOP/s is realistic on RX 6900 XT BF16→FP32 accumulate.
  - Expect 1.5–2× speedup on linear-heavy parts of denoise and VAE MLPs. If 800 linear dispatches consumed, say, ~60% of the 666.9 s GPU time in denoise and ~50% of 455.3 s in VAE, you save 200–300 s wall.
- Risk/checks:
  - Watchdog: keep each dispatch < 200–300 ms by slicing M or N loops.
  - Occupancy vs LDS: profile wave occupancy; adjust Cb_N 128↔256 depending on register pressure.
- Effort: 3–5 days. Start with 64×128×64 tile; measure; then tune.

5) Replace MPSGraph SDPA with a custom Flash-Attention-style kernel (or at least MPSMatrix ops)
- What to change:
  - MPSGraph BF16 matmul at 41.7 GFLOP/s is catastrophic on this GPU. Write a fused attention kernel for 56 heads × 128 dim:
    - Tile in blocks of (seq_q × seq_k) that fit LDS; compute softmax in a numerically stable streaming fashion; accumulate V on the fly; avoid writing QK^T to global memory.
    - Use BF16 inputs and FP32 accumulators; vector loads; per-head parallelism mapped to threadgroups.
  - If not ready to write full FA: replace MPSGraph with MPSMatrixMultiplication (even if “deprecated”), which is often faster on AMD via Metal than MPSGraph on Intel Macs; or call your new linear kernel for QK^T and P×V and fuse scaling/softmax partially.
- Estimated win:
  - For your head size (128), Flash-Attention kernels can hit 20–30% of TF16 peak on RDNA2; expect >5 TFLOP/s effective end-to-end attention. If attention is ~25–35% of DiT GPU time (200 attn dispatches), you can save 100–200 s wall.
- Risk/checks:
  - Numerical drift; verify token-wise outputs against MPSGraph within 1e-3 relative for BF16 path.
  - Kernel complexity higher; start with fixed head_dim=128 specialization.
- Effort: 5–7 days for FA v1; 1 day to trial MPSMatrix over MPSGraph.

6) Eliminate per-layer alloc/free in Qwen; use MTLHeaps + resource reuse; move its weights to Private with streaming
- What to change:
  - Pre-create one or two MTLHeaps (StorageModePrivate) sized to hold one layer’s working set (activations scratch + weights for that layer), and reuse across all layers by cycling offsets.
  - For weights, use the same async streaming pattern as DiT: stage Shared→Private per layer, with events.
  - Mark transient buffers as MTLHazardTrackingModeUntracked and manage ordering explicitly (fences/events) to cut driver overhead.
- Estimated win:
  - Qwen wall 147.5 s, GPU 131.2 s. You’re probably paying allocator and TB penalties. Expect 1.5–2× overall on Qwen → 70–100 s saved.
- Risk/checks:
  - Lifetime bugs if you recycle buffers too early; use a slot index tied to command buffer completion handlers.
- Effort: 1–2 days.

7) Retile the video VAE to minimize tiles; push to 1×1 or 1×2 with 16 GiB VRAM
- What to change:
  - Recompute the VAE’s per-frame memory graph (activations+weights). Peak 9.365 GiB suggests you can increase tile to 512 or even full 608×352 in a single pass on 16 GiB VRAM.
  - Reduce tile grid from 3×2 @256 to 2×1 @512 or 1×1 if it fits. This reduces overlaps, halo recomputations, and per-tile kernel launch overhead; it also improves locality and avoids redundant Shared traffic.
- Estimated win:
  - If 956 s is for 3×2 tiling with heavy boundary overhead, moving to 1×1 typically yields 1.5–2.5× speedup in VAE. Expect 300–500 s saved.
- Risk/checks:
  - Watchdog with very large tiles if single dispatch times blow past 2 s; split layers if needed.
- Effort: 0.5–1 day.

8) Switch file I/O to mmap + newBufferWithBytesNoCopy, drop F_NOCACHE, or adopt Metal I/O if supported
- What to change:
  - Replace pread into a staging malloc’d buffer with:
    - int fd = open(...); void* p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, offset_page_aligned); create MTLBuffer with newBufferWithBytesNoCopy(p+offset_in_page, len, StorageModeShared, deallocator).
    - Then Blit copy Shared→Private asynchronously (as in item 1).
  - Remove F_NOCACHE; enable F_RDAHEAD and/or use posix_fadvise(fd, POSIX_FADV_SEQUENTIAL) to let the page cache prefetch. With ~30 GiB free RAM, you can keep a large window of the 61.7 GiB set hot across the 4 passes; combined with 40 Gb TB this matters.
  - If available on your OS/GPU: Metal I/O (MTLIOCommandQueue/MTLIOFileHandle) to DMA file→Private buffer directly without CPU bounce. Check [device supportsIO] and test on AMD; on some Intel Macs this path is not enabled for discrete AMD, so fall back to mmap.
- Estimated win:
  - Removing the memcpy to staging can save a few GB/s of CPU memory traffic; across 144 GiB streamed → tens of seconds (20–60 s) and reduced CPU contention that indirectly reduces wait.
  - With page cache, repeated passes can cut actual SSD reads by 2–3×, further shrinking the 130 s of “unhidden” I/O.
- Risk/checks:
  - bytesNoCopy must keep the mapping alive until the blit finishes; ensure lifetime tied to command buffer completion handler.
  - If page cache thrashes (other apps), performance can regress; make it a runtime flag.
- Effort: 0.5–1 day.

9) Increase compute/transfer overlap within a block: issue next-block uploads earlier; allow >2 in-flight command buffers
- What to change:
  - Move “schedule upload for block N+1” earlier in the block N compute timeline (e.g., after first linear) to give transfer more slack before the critical path consumer.
  - Allow M=3–4 in-flight command buffers per queue; cap total outstanding work using a semaphore; keep dependency via events.
- Estimated win:
  - Smooths bubbles you currently see; 50–100 s less idle across denoise+VAE even after item 1.
- Risk/checks:
  - Memory pressure if too many uploads in flight; throttle by slot count and available VRAM.
- Effort: <1 day.

10) Precision/storage tweaks where safe: compress one-shot or bandwidth-bound tensors
- What to change:
  - Store AdaLN micro-params as FP16 (not BF16); or even INT8 with per-channel scale if used only as scale/bias. Convert to FP32 in registers.
  - If any attention/MLP intermediate weights are read-once and bandwidth-bound, pack to FP16 on disk and convert during upload on GPU with a lightweight kernel; halves PCIe traffic for those tensors.
- Estimated win:
  - If 30–50% of streamed bytes can be halved, you save another 50–100 s wall just from less I/O and blit time.
- Risk/checks:
  - Potential quality drift; A/B outputs with seeded runs and PSNR/LPIPS metrics; make it a flag.
- Effort: 1–2 days for AdaLN; more for model-wide.

11) Replace MPSGraph linear ops with your tuned kernel across the codebase; keep MPSGraph only for control-flow glue
- What to change:
  - Identify all MPSGraph matmuls in Qwen/DiT/VAEs and route them through your optimized BF16 linear op; keep tensor shapes static to allow specialized pipelines (cache MTLComputePipelineState per shape).
- Estimated win:
  - Removes inconsistent MPSGraph perf cliffs on AMD; an extra 50–150 s wall.
- Risk/checks:
  - Shape mismatches; unit-test a representative set of shapes.
- Effort: 2–3 days incremental.

12) Misc Metal hygiene that matters on AMD eGPU
- What to change:
  - Always allocate activations in Private when they are produced/consumed exclusively on GPU; only Shared when CPU reads are required. Use temporary Private buffers (or heap suballocations) and recycle aggressively.
  - Pre-create all pipeline states; avoid runtime compilation cost on the hot path.
  - Set resourceOptions: CPUCacheModeWriteCombined for Shared staging; StorageModeShared for bytesNoCopy; Untracked hazards when you control order.
  - Use argument buffers to reduce setBuffer overhead on large dispatch counts.
- Estimated win:
  - Not huge alone, but can trim 20–40 s and reduce driver overhead that shows up as idle.
- Risk/checks:
  - None, correctness-wise.
- Effort: 1–2 days.

Direct answers to your specific questions and corrections:

- Synchronous blit: Yes, switch to a dedicated blit queue and MTLSharedEvent so block N+1 upload overlaps block N compute. With triple buffering you can hide most of the 1803 s wait; expect 60–75% recovery.
- MTLHeap/triple buffering: Heaps won’t make math faster, but they eliminate allocator jitter and let you recycle big Private ranges safely. Use a heap per slot and suballocate. Triple buffering is recommended to hide I/O variance.
- 16×16 tile kernel and RDNA2 cache: Your L2 figure is incorrect; Navi 21 has multi-megabyte L2 (≈4 MiB), not 128 KiB. Weight reload per 16 rows is still a limiter because you’re capacity/BW bound on L2 and you’re not reusing W sufficiently. Move to 64×128 or 64×256 tiles with LDS double-buffering and reuse W across many M rows; target 18–24 TFLOP/s effective on the RX 6900 XT for BF16-in/fp32-accum GEMMs.
- AdaLN 24.2 GiB: Do not keep the 2688×96768 matrices. Compute the matvec once per block per step and keep only the 18×5376 results in Private, or stream them. That is the “something smarter.” adaln_out_features equals 18 modulation vectors × 5376 hidden channels.
- Video VAE: 956 s is too high. With 9.365 GiB peak you likely can push to 1×1 tiling at 608×352, or at worst 2×1 with 512 tiles. Expect 1.5–2.5× speedup in this phase.
- Thunderbolt: Worth chasing to 40 Gb/s. Realistic sustained Shared read 2.7–3.1 GB/s. It won’t fix bad Shared usage, but it reduces the pain and shortens any remaining unhidden uploads.
- F_NOCACHE: Drop it. Use mmap + bytesNoCopy for the staging Shared buffer to eliminate the CPU memcpy. Let the page cache keep hot windows over the 4 passes; add F_RDAHEAD or posix_fadvise(SEQUENTIAL). If Metal I/O is supported on your stack, use it to DMA file→Private directly.
- MPSGraph vs hand-written: On AMD Metal, MPSGraph is not tuned; your 41.7 GFLOP/s confirms it. Write your own Flash-Attention (best), or at least switch to MPSMatrixMultiplication or your tuned linear kernel for the matmuls. Expect order-of-magnitude improvement for attention.
- Bigger strategy: The right answer is: keep activations and currently-used weights in Private; stream all large weight matrices asynchronously; avoid ever touching Shared in a compute kernel. Recompute cheap, one-shot transforms (AdaLN matvec) instead of storing gigantic intermediates. Increase tile reuse so each streamed byte feeds many MACs. If you must read from Shared, fix TB to 40 Gb/s; but Shared should be staging only.

If you want a minimal-change, high-ROI sequence to implement:
- A) Async blit overlap with triple buffering (1).
- B) Fix TB to 40 Gb/s (2).
- C) Kill AdaLN 24.2 GiB residency; switch to per-block matvec micro-params (3).
- D) Retile video VAE to 1×1 or 2×1 (7).
- E) Swap MPSGraph attention for FA or MPSMatrix (5).
- F) Replace h3_linear_bf16 with the LDS-tiled kernel (4).
- G) mmap + bytesNoCopy and page cache (8).
- H) Heaps and Qwen resource reuse (6).

Combined, expect to cut wall from 3082 s to roughly:
- A+B: −1200 to −1600 s
- C: −300 to −600 s
- D: −300 to −500 s
- E: −100 to −200 s
- F: −200 to −300 s
- G+H+others: −100 to −200 s
Projected wall: 700–1200 s (12–20 min), dominated by GPU compute rather than waits.