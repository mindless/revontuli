You are doing an expert performance peer review. I want ACTIONABLE, RANKED optimizations — not encouragement, not a summary of what I told you. Assume I am a competent systems programmer. Be blunt about anything I got wrong.

# Situation

I got MiniMax H3 (text-to-video-and-audio, ~31B param DiT) running locally on hardware nobody targets: a **2018 Intel Mac mini** with an **external AMD GPU**. It works and produces photorealistic output. It is now SLOW and I want to make it fast.

## Hardware (all measured, not assumed)

- Mac mini 2018 (Macmini8,1), Intel Core i7-8700B, 6C/12T, 64 GiB RAM
- macOS 15.7.9 (Sequoia), x86_64. SIP enabled. No CUDA, no Apple Silicon.
- eGPU: **AMD Radeon RX 6900 XT** 16 GB, PCI `1002:73bf`, Metal reports `architecture.name = amdgpu_gfx1030` (RDNA2 / Navi 21)
- Enclosure: Razer Core X Chroma over Thunderbolt 3. **Upstream link negotiated 20 Gb/s, not 40.** `MTLDevice.maxTransferRate = 5.0e9 bytes/s`
- `recommendedMaxWorkingSetSize` = 15.98 GiB. `maxBufferLength` reports 3.50 GiB but is ADVISORY — I verified single buffers up to 16 GiB allocate fine, 24 GiB fails.
- `hasUnifiedMemory = NO`. Driver is Apple's own `AMDRadeonX6000_AMDNavi21GraphicsAccelerator`.
- Second GPU present: Intel UHD 630 (I explicitly pin Metal to the AMD card).

## Runtime

`antirez/h3.c` at commit `8974cc0` — a native Metal implementation of MiniMax H3 in C/Objective-C. Written for Apple Silicon. Key structure:

- It compiles `h3_shaders.metal` at runtime. 56 "portable" kernels plus 27 M5-only kernels behind `#ifdef H3_METAL_HAS_TENSOR`, which is enabled only when the Metal device *name* contains the string "M5". So on my AMD card the portable path is selected automatically and the `bfloat`/`metal_tensor`/TensorOps block is never compiled.
- Portable BF16 kernels store BF16 as `ushort` and convert with bit manipulation, accumulating in `float`. No native MSL `bfloat` needed.
- It also uses MPSGraph for SDPA, some linear ops, and MLP, with `MPSDataTypeBFloat16` and `MPSDataTypeFloat32`.
- `--ssd-streaming` mode: keeps 2 DiT block "slots" resident and preads the next block's weights from SSD while the GPU runs the current one.

## Model dimensions (from FL2VA/transformer/config.json)

```
hidden_size 5376, num_layers 50, num_attention_heads 56, attention_head_dim 128
inner = 56*128 = 7168, ffn_hidden_size 14336
time_embed_dim 2688, adaln_out_features 96768
```

Per DiT block streamed matrices (BF16): qkv 5376->21504 (220.5 MiB), out (77 MiB), fc1 5376->28672 (294 MiB), fc2 (154 MiB). ~770 MiB per slot.

## THE root cause I already found and fixed

h3.c allocated **every** MTLBuffer as `MTLResourceStorageModeShared`. On Apple Silicon that's free. On a discrete eGPU, Shared = host memory reached over Thunderbolt. I measured GPU read bandwidth with a trivial streaming kernel:

| storage mode | bandwidth |
|---|---|
| `Shared` (host over TB3) | **1.76 GB/s** (flat at 256 MiB / 1 GiB / 2 GiB) |
| `Private` (VRAM) | **318–478 GB/s** |

That's a ~270x gap. And `h3_linear_bf16` is a 16x16 tiled matmul that **reloads the weight tile for every 16-row block**, so effective weight traffic is `weight_bytes * ceil(rows/16)`. Timing h3's own kernel:

| shape | rows | weight | Shared | Private |
|---|---|---|---|---|
| QKV 5376->21504 | 256 | 220.5 MiB | 6.584 s | 0.049 s |
| QKV 5376->21504 | 1024 | 220.5 MiB | 23.106 s | 0.199 s |
| FC1 5376->28672 | 256 | 294.0 MiB | 8.503 s | 0.066 s |
| FC1 5376->28672 | 1024 | 294.0 MiB | **GPU TIMEOUT** | 0.252 s |

macOS runs a GPU progress watchdog. A stalled dispatch gets killed:

```
kernel (IOAcceleratorFamily2) checkGPUProgress() - Signaling hardware error on channel 9
kernel (AMDRadeonX6000HWLibs) AMD Error: Restart Channel: 9 ComputeUQ6
kernel Trying to restart GPU (AMD Radeon RX 6900 XT)...
kernel Deny Submissions/ignore app[h3] with 2 GPURestarts in 56 submissions.
h3 (Metal) Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)
```

### Patches I already applied (both gated on `!hasUnifiedMemory`, no-ops on Apple Silicon)

1. `h3_gpu_tensor_new_bf16_resident()`: allocates the payload `MTLResourceStorageModePrivate` plus a host-visible staging buffer. The SSD streaming path `pread`s into staging, then does a blit encoder copy staging -> private, `commit`, `waitUntilCompleted`. Used for the 4 hot-loop matrices in each of the 2 stream slots.
2. Bounded command buffers: `h3_gpu_require_command()` (the single choke point before every dispatch and every MPS op) flushes after N ops and caps in-flight buffers at M. Defaults N=4, M=2 on a non-unified-memory device. Metal runs same-queue command buffers in commit order so results are unchanged.

After those two changes it runs end to end and the output is genuinely photorealistic.

## Current profile — the thing I want you to optimize

Target run: 608x352, 22 frames, 4 steps, 50 DiT layers, reuse 1, `--ssd-streaming`, seed 42. Total **3081.63 s wall (51 min)**.

| phase | wall | GPU time | peak live | dispatches |
|---|---|---|---|---|
| Qwen text encoder | 147.52 s | 131.19 s | 2.730 GiB | 451 direct, 350 linear |
| DiT load | 38.56 s | 19.76 s | 1.678 GiB | 75 direct, 9 linear, 2 attn |
| DiT Euler denoise | **1934.07 s** | 666.94 s | 1.678 GiB | 628 direct, 800 linear, 200 attn |
| audio VAE | 2.63 s | 2.34 s | 0.282 GiB | 340 direct, 136 conv |
| video VAE | **956.27 s** | 455.33 s | 9.365 GiB | 1314 direct, 870 linear, 216 attn; tiled 3x2 at 256 px |

- **GPU time is only 1275.6 s of 3081.63 s wall — 59% of the run is waiting, not computing.**
- Denoise phase: `wait=1803.69 s` vs `root-gpu=666.94 s`.
- SSD streaming moved **144.272 GiB** during denoise (200 block loads = 4 steps x 50 blocks), 130.007 s reported as "unhidden wait".
- Peak host RSS 34.70 GB of 64 GiB. Zero swap.

## What is still `Shared` (host memory) and therefore slow

- The **Qwen text encoder** weights (per-layer, I see repeating 250 MiB and 80 MiB allocations that are created and freed each layer).
- The **AdaLN matrices: 496.1 MiB per DiT block x 50 blocks = 24.2 GiB**, all allocated during a "precompute AdaLN" phase and retained for the whole run. (496.1 MiB = 2688 x 96768 x 2 bytes.)
- All activations.
- Note 24.2 GiB does NOT fit in 16 GiB VRAM, so a naive "make everything Private" is not available.

# What I want from you

Give me a **ranked list of concrete, implementable optimizations**, each with:
1. What to change (specific enough to code from — name the Metal API, the kernel restructuring, the flag)
2. Estimated wall-clock win, reasoned from the numbers above
3. Risk / what could break, and how I'd verify correctness
4. Rough effort

Please specifically address, and correct me if my reasoning is wrong:

- **The synchronous blit.** My staging->VRAM upload does `commit` + `waitUntilCompleted` inline in the streaming read, on the same queue h3 encodes compute into. Should this be a dedicated queue + `MTLSharedEvent`/`MTLFence` so the upload of block N+1 truly overlaps compute of block N? h3 already double-buffers 2 slots, so the structure is there. How much of that 1803 s wait is recoverable?
- **`MTLHeap` / triple buffering.** Would a suballocated `MTLHeap` for the stream slots help, or is that irrelevant here?
- **The 16x16 tile kernel.** Given the weight now lives in VRAM at ~478 GB/s, is `h3_linear_bf16`'s per-16-row-block weight reload still the limiter? RDNA2 has 128 KB L2 per... (correct me). Would a larger tile (32x32, 64x64), `simdgroup_matrix<float, 8, 8>` MFMA-style ops via MSL `simdgroup_multiply_accumulate`, or a split-K formulation be the right move on gfx1030 specifically? What tile size and threadgroup shape would you pick for 5376x28672 BF16-in/float-accum on RDNA2?
- **The AdaLN 24.2 GiB.** It's precomputed once. Is the right move to (a) free the weight matrices after precompute, (b) stream them like the DiT blocks, (c) compute AdaLN on CPU with Accelerate/AMX since it's one-shot, or (d) something smarter? What does "precompute AdaLN" most likely produce, dimension-wise, given `adaln_out_features 96768 = 5376 * 18`?
- **The video VAE: 956 s for 22 frames at 608x352, tiled 3x2 at 256 px, peak 9.365 GiB.** That's 31% of my runtime for decode alone. Tiling choices? Is 9.365 GiB peak leaving headroom I should spend on fewer/larger tiles?
- **Thunderbolt.** The link came up at 20 Gb/s not 40. Worth chasing (cable, port, `Route String 3`/`103` chained enclosure), and what's the realistic ceiling?
- **F_NOCACHE.** The streaming read sets `fcntl(F_NOCACHE, 1)`. With 64 GiB RAM and a 61.7 GiB DiT read 4x per run, should I let the page cache work instead? Or use `mmap` + `newBufferWithBytesNoCopy` (which I verified this driver accepts even at non-page-aligned offsets) to skip the pread copy entirely?
- **MPSGraph vs hand-written kernels** on gfx1030 for SDPA at these shapes (56 heads x 128 dim). I measured MPSGraph BF16 matmul at 2048x3072x3072 = 41.7 GFLOP/s, which seems terrible for a card with ~46 TFLOP/s fp16 peak. Is MPSGraph a bad choice on AMD Metal, and should I write my own flash-attention-style kernel?
- Anything I have MISSED that is bigger than all of the above.

Also: is there a fundamentally better strategy I'm not seeing, given 16 GiB VRAM, 64 GiB host RAM, and a 1.76 GB/s host<->device link? E.g. is the right answer to keep MORE resident in VRAM and accept fewer layers, or to restructure so each block's weights are read once per step instead of once per row-block?

Be specific and quantitative. If you think one of my measurements must be wrong, say which and why.
