# Peer review synthesis — optimizing MiniMax H3 on the RX 6900 XT eGPU

Three independent reviews, same brief, run in parallel over OpenRouter
2026-08-21:

| model | id | output |
|---|---|---|
| Kimi K3 | `moonshotai/kimi-k3` | `kimi-k3-v2.md` (15.8 k chars) |
| GPT-5 | `openai/gpt-5` | `gpt-5-v2.md` (18.0 k chars) |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | `gemini-31.md` (8.1 k chars) |

I then verified the contested claims on the machine. **Four of them were wrong,
including one of my own premises that misled two of the three reviewers.**

---

## Verified corrections (measured, not argued)

### 1. `simdgroup_matrix` is NOT usable on gfx1030 — and fails late

Gemini's headline kernel recommendation was to rewrite the matmul with
`simdgroup_matrix` / `simdgroup_multiply_accumulate`. Kimi said those are
`MTLGPUFamilyApple7+` only and would not compile. **Kimi is right, and the
failure mode is worse than a clean rejection** (`logs/24-simdgroup-matrix.txt`):

```
AMD Radeon RX 6900 XT
  supportsFamily(Apple7)   : no
  simdgroup_matrix compiles: YES        <-- the library builds fine
  pipeline builds          : NO
  pipeline error: SC compilation failure
                 There is a call to an undefined label

Intel(R) UHD Graphics 630
  pipeline error: AIR builtin function was called but no definition was found.
```

So MSL accepts the source and `newLibraryWithSource:` succeeds; the AIR builtins
have no implementation in the AMD backend and it only blows up at
`newComputePipelineStateWithFunction:`. Any rewrite must use plain vector FMA
with threadgroup-memory staging and register blocking.

### 2. My "59% of the run is waiting" premise was WRONG

I fed this to all three reviewers and it shaped their top items. `h3_gpu.h`
documents the caveat explicitly:

```c
double command_wait_seconds;
/* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
 * so command_wait_seconds is the complete turnaround measurement. */
double gpu_seconds;
```

`gpu_seconds` accumulates only `GPUEndTime - GPUStartTime` of **root** command
buffers. The denoise phase issued **800 MPSGraph linear + 200 MPSGraph
attention** dispatches, whose child-buffer GPU time is not counted. So
`root-gpu = 666.94 s` is a floor, not the total, and `wait = 1803.69 s` is the
complete turnaround. The GPU is largely **busy but inefficient**, not idle.

Consequence: Gemini's #1 ("cut denoise 1934 s -> 700-800 s") and GPT-5's #1
("1080-1350 s saved") are both overestimates derived from my bad premise. Kimi
independently smelled it — it computed that 144 GiB over a 1.76 GB/s link is a
hard floor of only ~82 s and concluded the streaming could not explain 1803 s.
Kimi reached the right suspicion via the wrong mechanism (see #3).

### 3. The 24.2 GiB of AdaLN weight is NOT read in the hot loop

Kimi's #1 and GPT-5's #3 both assumed the 496 MiB/block AdaLN matrices are read
during denoise, and sized their wins accordingly (Kimi: 900-1700 s). Verified
false. In `h3_dit_schedule.c` the projection is done once per block and the
weight is released immediately:

```c
h3_gpu_tensor *weight = weight_bf16_2d(weights, gpu, weight_name,
                                       BLOCK_OUTPUT, H3_DIT_TIME_DIM, ...);
schedule->blocks[block] = h3_gpu_tensor_new_bf16(
    gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);   /* the RETAINED output */
... h3_gpu_linear_bf16(gpu, schedule->blocks[block], time, weight, bias, ...)
free_tensor(&weight);                                    /* released here */
```

and the header states the intent: *"This intentionally submits one projection at
a time, so a 498 MiB block projection is released before the next is loaded."*

The denoise loop reads `h3_dit_schedule_block(schedule, block)` — the retained
**output**, `time_rows x 96768 x 2 B`, which is single-digit MiB, not 496 MiB.

**But there is still a real bug here.** My allocation instrumentation shows
device-live growing by ~496.9 MiB per block monotonically across all 50
(4.99 GiB -> 26.85 GiB), so on this platform the weight is **not actually being
reclaimed** despite `free_tensor`. Since I separately proved
`currentAllocatedSize` tracks live residency correctly on this driver
(`logs/17-reclaim.txt`: 8 GiB -> 0.000 GiB, three rounds), something is
retaining those buffers. That is 24.2 GiB of host RSS held for no reason.
**Open item, worth chasing — see plan item 1.**

### 4. GPT-5's matmul FLOP arithmetic is off by ~10x

GPT-5 wrote: *"QKV 5376->21504, rows=1024: 0.199 s. FLOPs ~ 2xMxKxN =
2x1024x5376x21504 ~= 2371 GFLOP. That implies 11.9 TFLOP/s achieved today.
That's already decent"* — and therefore de-prioritized the kernel rewrite.

The correct value is `2 x 1024 x 5376 x 21504 = 236.8 GFLOP`, so
**236.8 / 0.199 = 1.19 TFLOP/s**, about **5% of this card's ~23 TFLOP/s FP32
peak**. Kimi (1.25 TFLOP/s on FC1) and Gemini (1.18 TFLOP/s) both got it right.
The kernel rewrite is justified; GPT-5's dismissal is not.

### 5. Cache hierarchy — Kimi's is correct

gfx1030 / Navi 21: 32 KiB L0 per CU, 128 KiB L1 per shader array, **4 MiB L2**,
plus **128 MiB Infinity Cache**. Gemini's "128 MB Infinity Cache" is right;
its "128KB L2" aside is not. This matters: a 220 MiB QKV weight nearly fits in
Infinity Cache, so tile-level reuse is cheaper than the raw 478 GB/s VRAM figure
suggests, which strengthens the case for large tiles.

### 6. Unresolved: is MPSGraph really that slow?

My probe measured MPSGraph BF16 matmul at 2048x3072x3072 = **41.7 GFLOP/s**.
GPT-5 took it at face value ("catastrophic"). Kimi says it is almost certainly
my methodology — I timed a single `runWithMTLCommandQueue` call including graph
construction and first-run allocation. Kimi is probably right; that number is
1/1000 of peak, which is implausible even for a neglected backend. **Must
re-measure steady-state (N iterations, one command buffer) before deciding to
replace MPSGraph.** Not yet verified either way.

---

## Where all three agree (high confidence)

1. The remaining `Shared` allocations are the dominant cost. Everything not yet
   moved to VRAM is read at 1.76 GB/s.
2. `h3_linear_bf16`'s 16x16 tile is far off peak and should be 64x64 or larger
   with register blocking and threadgroup staging.
3. The synchronous `commit` + `waitUntilCompleted` in the streaming upload
   should become a dedicated blit queue plus event-based sync, with 3-4 slots.
4. The video VAE at 956 s for 22 frames is disproportionate and under-tiled
   given 9.365 GiB peak against a 15.98 GiB budget.
5. Free the retained AdaLN memory.
6. `MTLHeap` is irrelevant here (Gemini explicit; others silent). It reduces
   allocation CPU overhead, which is not the problem.

---

## Ranked plan, re-ordered after the corrections

### 1. Find out why the 496 MiB AdaLN weights are not reclaimed — then reclaim them
Effort: low. Win: 24.2 GiB of host RSS (34.70 GB -> ~10 GB).

Upstream already calls `free_tensor(&weight)` per block and device-live still
grows. Instrument `h3_gpu_tensor_free` to log frees >= 64 MiB with the device
counter after, and find the retainer. Prime suspects: an `MPSGraphTensorData`
held by `gpu.linearCache` (since `h3_gpu_linear_bf16` goes through MPSGraph),
or a command buffer not drained.

The payoff is not just RSS. Freeing 24 GiB lets the page cache hold a much
larger slice of the 61.7 GiB DiT, and the run currently re-reads 144.272 GiB
from SSD per generation (200 block loads x 4 steps). That directly attacks the
biggest I/O term, and it is a prerequisite for item 6 being worthwhile.

**Verify:** device-live after precompute should sit near its pre-precompute
value; output must stay bit-identical at seed 42.

### 2. Move activations to Private VRAM
Effort: medium. Win: large, and only GPT-5 raised it.

`h3_gpu_tensor_new_f32/_bf16` return `Shared` buffers, so **every intermediate
tensor round-trips Thunderbolt twice** — written by one kernel at 1.76 GB/s,
read by the next at 1.76 GB/s. Peak live is 1.678 GiB in DiT and 9.365 GiB in
the VAE, both inside the 15.98 GiB budget.

Extend the `_resident` allocator I already wrote, with **lazy staging**: allocate
Private, and only create the host-visible staging buffer on first CPU access.
There are exactly **7 sites** in `h3_gpu.m` that touch `.contents`
(`read_f32`, `read_f32_range`, `read_bf16`, `write_f32`, `write_f32_range`,
`write_bf16`, `write_bf16_range`, plus the `tensor_new` memcpy and the file
read) — a contained interception surface.

**Verify:** bit-identical output at seed 42; Metal API validation on.

### 3. Rewrite `h3_linear_bf16` for RDNA2 — WITHOUT `simdgroup_matrix`
Effort: medium-high (the only genuinely hard item). Win: the kernel is at
~1.19 TFLOP/s of ~23 TFLOP/s peak.

Consensus spec (Kimi and GPT-5 converge; Gemini's tile advice is right even
though its instruction choice is not):

- Output tile **64x64** (Kimi) to **64x128** (GPT-5); 256 threads = 8 wave32 simdgroups.
- Each thread computes a **4x4 register sub-tile** (16 FP32 accumulators) — 16 FLOPs per 2 loaded values.
- K-block 16-64, A and B tiles staged in threadgroup memory (a few KiB; 128 KiB LDS available), optionally double-buffered.
- `ushort4`/`float4` vectorized global loads; BF16->FP32 convert on load; accumulate FP32.
- Iterate threadgroups over (M/64 x N/64), K innermost. **Skip split-K** — at N=21504 there are already 336 blocks in N alone, plenty for 80 CUs.
- The dimensions cooperate: 5376/64 = 84, 21504/64 = 336, 28672/64 = 448 — all exact, so no edge predication.

Kimi estimates 8-15 TFLOP/s achievable, i.e. FC1@1024 rows from 0.252 s to
0.025-0.04 s. Keep each dispatch under ~250 ms to stay clear of the watchdog.

**Verify:** max relative error vs the current kernel under BF16 tolerance, then
end-to-end frame diff at seed 42.

### 4. Asynchronous upload: dedicated blit queue + `MTLSharedEvent`, 3-4 slots
Effort: medium. Win: **~130-210 s, not the ~1000 s two reviewers projected** —
because of correction #2.

Remove the inline `commit`+`waitUntilCompleted` from
`h3_gpu_tensor_read_file_bf16_mode`. Second `MTLCommandQueue` for uploads; per
slot a `MTLSharedEvent` with monotonically increasing values — transfer signals
after the blit, compute calls `encodeWaitForEvent:` before the first kernel that
consumes the slot, and compute signals when done with a slot so its staging
buffer can be reused. Raise slots from 2 to 3-4 (~1 GiB each, ample headroom).

Note `MTLEvent` suffices for same-device GPU-GPU ordering and is cheaper than
`MTLSharedEvent` (Gemini's point); use `MTLSharedEvent` only if CPU-side waits
are needed.

**Verify:** checksum an uploaded block against the file on the first pass;
bit-identical output at seed 42. This is the item where a race yields subtly
wrong numbers rather than a crash, so verify hard.

### 5. Video VAE: fewer, larger tiles + Private activations
Effort: low-medium. Win: it is **31% of runtime** (956 s of 3082 s).

Currently `tiles 3x2 at 256 pixels` for a 608x352 frame, peak 9.365 GiB of
15.98 GiB. 608x352 needs only ~2.4 x 1.4 tiles of 256 px, so the 3x2 grid is
paying halo overlap and redundant convolution on every seam, across 1314 direct
+ 870 linear + 216 attention dispatches. Try 2x2 at 320 px, or a single tile,
raising tile size while watching `currentAllocatedSize` and stopping around
13 GiB. Combine with item 2.

**Verify:** decode one frame tiled vs untiled and check PSNR at the seams.

### 6. Re-measure MPSGraph properly, then decide about SDPA
Effort: low to measure, high to replace.

Re-time MPSGraph steady-state (many iterations in one command buffer, excluding
graph construction) before writing anything. If it comes back in the TFLOP/s
range, keep it for plain linears and only consider a fused attention kernel. If
it really is sub-100 GFLOP/s, then QKV/out/MLP are just GEMMs that item 3
already covers, and a flash-attention kernel becomes worthwhile: 56 heads x 128
dim, one head fits comfortably in a threadgroup, online softmax in registers,
FP32 accumulation.

### 7. Drop `F_NOCACHE`, consider `mmap` + `newBufferWithBytesNoCopy`
Effort: low. Win: conditional on item 1.

The streaming read sets `fcntl(F_NOCACHE, 1)`, forcing all 144.272 GiB per run
to come off the SSD. With 64 GiB RAM this is self-defeating **once RSS drops**;
today 34.70 GB of RSS leaves too little page cache to matter. I already
verified this driver accepts `newBufferWithBytesNoCopy` at non-page-aligned mmap
offsets (`logs/06-nocopy-probe.txt`, 7/7 offsets), so the `pread` copy can be
skipped entirely. Do item 1 first, then measure.

### 8. Chase the 40 Gb/s Thunderbolt link
Effort: 0.5 day. Win: 1.76 -> ~2.7-3.1 GB/s on whatever `Shared` traffic remains.

The enclosure negotiated 20 Gb/s (`maxTransferRate = 5.0e9`), and 1.76 GB/s is
exactly what a 20 Gb/s link yields after encoding and PCIe overhead. Suspects:
a passive cable over 0.8 m, and the fact that this Core X Chroma presents a
**chained pair** of Thunderbolt devices (Route String 3 and 103). Both Gemini
and GPT-5 note that after items 2-4 this becomes largely moot, since the
remaining transfers hide behind compute. Worth doing, lowest priority.

---

## Sequencing

1 -> 2 -> 5 -> 4 -> 6(measure) -> 3 -> 7 -> 8

Items 1, 2 and 5 are the best effort-to-win ratio and reuse machinery that
already exists. Item 3 is the largest single compute win but the most work, and
should be done after 6 tells us whether MPSGraph or the hand-written kernel is
on the critical path. Item 8 last, because it is hardware shopping.
