Here is the ranked, actionable optimization plan for your pipeline. 

Your fundamental issue is that your CPU, SSD, PCIe bus, and GPU are completely serialized. You are paying the latency penalty of every component back-to-back. 

### 1. Fix Pipeline Stalls: Dual-Queue Async Uploads (Recover ~1000s+)
Your `commit` + `waitUntilCompleted` inline with the streaming read destroys the double-buffering. You are holding the CPU hostage until the blit finishes, preventing the CPU from initiating the SSD read for slot N+1, and starving the compute queue.

**What to do:** 
1. Create two `MTLCommandQueue`s: one for Compute, one for Blit.
2. Create an `MTLEvent` (not `MTLSharedEvent`, which is for cross-process/CPU signaling; `MTLEvent` is lower overhead for intra-device GPU-GPU sync).
3. **Thread 1 (I/O & Blit):** `pread` to Staging Buffer -> Encode Blit to Private Buffer -> Encode `[blitEncoder encodeSignalEvent:syncEvent value:block_id]` -> `commit` (NO `waitUntilCompleted`).
4. **Thread 2 (Compute):** Encode `[computeEncoder encodeWaitForEvent:syncEvent value:block_id]` -> Encode Compute (Linear/Attn) -> `commit`.

**Quantitative Reasoning:** 
You spend 1803s in "wait" during denoise. Streaming 144.27 GB over a 1.76 GB/s link takes ~82 seconds of raw transfer time. Reading it from a decent NVMe SSD takes ~40-60 seconds. Because you serialize, you are racking up 1803s. Fully overlapping I/O and compute means your wall time for denoise will approach `max(compute_time, io_time)`. Since compute is currently 667s, this one change should cut your denoise time from 1934s to ~700-800s.
**Risk/Effort:** Low risk, medium effort. You already have the 2-slot architecture.

### 2. Stream the AdaLN Weights (Saves thrashing, recovers ~150s+)
You are holding 24.2 GiB of AdaLN matrices in `Shared` memory. When the GPU executes the AdaLN kernels, it fetches these weights over Thunderbolt at 1.76 GB/s. 496.1 MiB per block $\times$ 50 blocks $\times$ 4 steps = 99.2 GiB of AdaLN weight traffic pulled over Thunderbolt, likely experiencing severe cache thrashing because it doesn't fit in the GPU's 128MB Infinity Cache.

**What to do:** 
AdaLN weights are just linear projection weights (dim 2688 $\times$ 96768). **Add them to your SSD streaming slots.** 
1. Your current streamed payload per slot is ~770 MiB. Adding the 496 MiB AdaLN matrix brings the slot to ~1.26 GiB. 
2. Two slots = 2.52 GiB of Private VRAM, well within your 15.98 GiB working set.
3. Free the 24.2 GiB host allocation. Stream the AdaLN weights into `Private` memory alongside QKV/FC matrices and execute them from VRAM.

**Quantitative Reasoning:** Removes ~100 GiB of Thunderbolt traffic per run and entirely eliminates the 24.2 GiB host RAM pressure. 
**Risk/Effort:** Low risk, low effort. Just copy-paste your existing linear streaming logic to AdaLN.

### 3. Let the Page Cache Work: Drop `F_NOCACHE` + Use `mmap` (Speeds up Steps 2-4)
You have 64 GiB of RAM. The DiT model weights are ~61 GiB. You are executing 4 steps. With `F_NOCACHE`, you force the OS to fetch 61 GiB from the physical SSD 4 times. 

**What to do:**
Remove `fcntl(F_NOCACHE, 1)`. Use `mmap(MAP_SHARED)` to map the model file into virtual memory. Allocate your staging buffers using `[MTLDevice newBufferWithBytesNoCopy: length: options: deallocator:]` pointing to the mmap'd regions.
*(Note: Apple's AMD driver requires the pointer to be page-aligned (4KB), which mmap guarantees).*

**Quantitative Reasoning:** 
Step 1 will page-fault the model from SSD into host RAM (~1.5 GB/s to 3 GB/s depending on your Mac Mini's SSD). Steps 2, 3, and 4 will read directly from DDR4 RAM. You bypass the CPU `pread` entirely. The blit engine pulls straight from the page cache into VRAM.
**Risk/Effort:** Low risk. If memory pressure gets too high, macOS will just dynamically evict pages back to SSD. 

### 4. Rewrite `h3_linear_bf16`: RDNA2 SIMD-Group Tiling (Recovers ~400s+ GPU compute)
RDNA2 (gfx1030) does **not** have dedicated tensor cores (Matrix Core / MFMA instructions are CDNA-only). However, it has massive vector throughput, 128KB Local Data Share (threadgroup memory) per WGP, and supports Metal's `simdgroup_matrix` which the compiler brilliantly lowers to optimized `v_dot2_f32_bf16` or FMA sequences. 

Your 16x16 tile is starving the ALUs and thrashing L1/L2. 

**What to do:**
1. Block size must be at least 64x64, ideally 128x128. 
2. **Threadgroup Caching:** Threads must cooperatively load a 128x128 block of weights and 128x128 block of inputs into `threadgroup` memory. 
3. **Barrier:** `threadgroup_barrier(mem_flags::mem_threadgroup)`.
4. **SIMD Matrix Ops:** Load from `threadgroup` into `simdgroup_matrix` objects. Multiply using `simdgroup_multiply_accumulate`. 
5. Accumulate in `float`, write out.

**Quantitative Reasoning:** 
Your QKV matrix takes 0.2s for 1024 rows. That's $(1024 \times 5376 \times 21504 \times 2) \approx 236$ GFLOPs. 236 GFLOPs / 0.2s = 1.18 TFLOP/s. 
The 6900 XT is capable of **~46 TFLOP/s** FP16/BF16. You are running at 2.5% of hardware peak because your 16x16 tile lacks arithmetic intensity. Moving to a 128x128 threadgroup-cached tile using `simdgroup_matrix` should easily push you over 15 TFLOP/s, dropping your 667s compute time to under 100s.
**Risk/Effort:** High effort. Writing a highly optimized GPU matmul is tedious. Look at Apple's Metal Performance Shaders sample code for `simdgroup_matrix` to get the boilerplate.

### 5. Video VAE: Fix Weight Residency and Maximize Tiles (Recovers ~400s)
956s for a 22-frame decode is absurd. A 6900 XT should decode that in <10 seconds. 
1. **The weights are in Shared memory.** You state "All activations" and earlier imply only DiT weights were fixed. Ensure VAE Conv2D weights are explicitly loaded as `Private`. 
2. **Tiling overhead:** 3x2 tiling at 256px requires overlapping halos for convolutions. You are doing redundant compute. You have 16 GiB VRAM. Peak usage is 9.36 GiB. You have over 6 GiB free. 
3. **What to change:** Change tiling to 2x1 or 1x1 (full frame if possible). VAE operates on spatial dimensions, so reducing tiles reduces boundary compute and padding logic overhead.

---

## Direct Answers to Your Remaining Questions

**MPSGraph vs Custom SDPA on AMD?**
Ditch MPSGraph for attention. 41.7 GFLOP/s is host-fallback or scalar-fallback speeds. MPSGraph is notoriously unoptimized for AMD GPUs on macOS, heavily favoring Apple Silicon AMX/Neural Engine paths. 
Write a fused FlashAttention kernel. 56 heads $\times$ 128 dim means one head fits entirely in a threadgroup. RDNA2 has 128KB of local memory per WGP. You only need $(128 \times 4 \text{ bytes}) = 512$ bytes per token. A block of 64 tokens takes 32KB. It fits beautifully.

**Thunderbolt: 20 Gb/s vs 40 Gb/s**
You are using a passive Thunderbolt cable longer than 0.8 meters, or your enclosure is plugged into a daisy-chained port. Buy an **Active Thunderbolt 3 or 4 cable (<= 0.8m)**. 
*Reality Check:* 20 Gb/s yields exactly the 1.76 GB/s payload bandwidth you measured (after 8b/10b encoding and PCIe overhead). 40 Gb/s will raise this to ~2.7 GB/s. It is worth fixing, but if you implement optimization #1 (async overlap), the transfer time will hide behind the compute time anyway, making the cable upgrade a moot point for this specific workload.

**`MTLHeap` / Triple Buffering**
Irrelevant here. `MTLHeap` speeds up allocation CPU overhead. You are reusing two static buffers. Triple buffering might offer a slight 2-5% pipeline smoothing over double buffering, but it is not worth the VRAM given you need to fit AdaLN into the slots now. Stick to double buffering.

### The "Fundamentally Better" Strategy?
With 16 GB VRAM and 64 GB host RAM, your current macro-architecture (SSD streaming double-buffer) is actually the correct one. The flaw is entirely in the execution (synchronous blits, unoptimized AMD kernels, ignoring AdaLN weights, and avoiding the OS page cache). 

**Execute in this order:** Dual-Queues (Sync Blit fix) -> Stream AdaLN -> `mmap` -> Matmul Rewrite. You will see a 4x-5x end-to-end speedup.