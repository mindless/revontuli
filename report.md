# Executive Summary

**SUCCESS — macOS native Metal / RX 6900 XT.**

MiniMax H3 runs locally on this 2018 Intel Mac mini, executing on the external
AMD Radeon RX 6900 XT through Apple's own Metal driver, and produces a coherent
video with synchronised audio.

**Target configuration result**, after the two optimizations below
(`outputs/h3-egpu-opt.mp4`):

| Measurement | Value |
|---|---|
| GPU actually performing H3 compute | **yes** — AMD Radeon RX 6900 XT, `amdgpu_gfx1030`, registryID `0x1000008fe` |
| Coherent MP4 | **yes** — photorealistic red fox walking through snow in a pine forest |
| Configuration | T2VA / FL2VA, 608x352, 22 frames, 4 steps, 50 layers, reuse 1, `--ssd-streaming`, seed 42 |
| Output | 608x352 **h264** @ 3.09 Mbit/s + **AAC** @ 168 kbit/s, 375,711 bytes |
| Generation time | **364.24 s (6 min 4 s)** — down from 3081.63 s |
| Peak host RAM (RSS) | **15.46 GB** — down from 34.70 GB |
| Swap used | **0** |
| Output vs pre-optimization baseline | **bit-identical**, md5 `886b40d10c5a83fb01393a4c62bdfcbd` |

### Optimization results — MACHINE-TESTED

Two changes, both gated on `hasUnifiedMemory` so Apple Silicon is untouched:
an autorelease pool per AdaLN block (commit `f1bca68`), and activations in
device-local VRAM (commit `25324a6`).

| Phase | wall before | wall after | | GPU before | GPU after | |
|---|---|---|---|---|---|---|
| Qwen text encoder | 147.5 s | 38.9 s | **3.8x** | 131.2 s | 3.55 s | 37.0x |
| DiT load | 38.6 s | 41.8 s | 0.9x | 19.8 s | 0.40 s | 49.2x |
| DiT Euler denoise | 1934.1 s | 262.3 s | **7.4x** | 666.9 s | 9.26 s | 72.1x |
| audio VAE decoder | 2.6 s | 1.2 s | 2.2x | 2.3 s | 0.44 s | 5.3x |
| video VAE decoder | 956.3 s | 18.3 s | **52.4x** | 455.3 s | 3.18 s | 143.1x |
| **total** | **3081.6 s** | **364.2 s** | **8.46x** | | | |
| peak RSS | 34.70 GB | 15.46 GB | 2.24x lower | | | |

Two things are worth stating plainly. First, **the output is bit-identical** to
the pre-optimization run at the same seed, so this is pure speed, not a quality
trade. Second, the DiT denoise is now dominated by I/O, not compute: 262.3 s
wall against 9.26 s of GPU time, with 144.272 GiB still streamed from SSD and
120.5 s of that unhidden. That is where the next win is (asynchronous upload,
review item 4), and it is now the single largest term in the run.

**The earlier, pre-optimization target run** (`outputs/h3-egpu-test.mp4`,
`logs/generate-20260821-075737.log`) is kept below for comparison:

| Measurement | Value |
|---|---|
| GPU actually performing H3 compute | **yes** — AMD Radeon RX 6900 XT, `amdgpu_gfx1030`, registryID `0x1000008fe` |
| Coherent MP4 | **yes** — photorealistic red fox walking through snow in a pine forest, temporally consistent, correct gait |
| Configuration | T2VA / FL2VA, 608x352, 22 frames, 4 steps, 50 layers, reuse 1, `--ssd-streaming`, seed 42 |
| Output | 608x352 **h264** @ 3.09 Mbit/s + **AAC** @ 168 kbit/s, 375,711 bytes |
| Audio | **yes** — AAC, 30 frames, 0.925 s, synchronised |
| Generation time | **3081.63 s (51 min 22 s)** |
| Peak host RAM (RSS) | **34.70 GB** of 64 GiB |
| Swap used | **0** (`vm.swapusage: used = 0.00M`) |
| Peak GPU-tracked allocation | DiT 1.678 GiB, video VAE 9.365 GiB — both inside the 15.98 GiB working set |
| GPU time by phase | text encoder 131.19 s, DiT 686.70 s, video VAE 455.33 s, audio VAE 2.34 s |
| SSD streaming | 144.27 GiB read across the denoise phase, 130.01 s unhidden |

Verified smaller smoke test (`logs/20-smoke-vram.log`, `outputs/smoke-vram.mp4`):

| Measurement | Value |
|---|---|
| GPU actually performing H3 compute | **yes** — AMD Radeon RX 6900 XT, `amdgpu_gfx1030`, Metal registryID `0x1000008fe` |
| Coherent MP4 produced | **yes** — snow-covered ground, pine forest, orange fox subject, consistent across frames |
| Output | 256x160, 22 frames, **h264 video + AAC audio**, 123,367 bytes |
| Audio | **yes** — AAC stream, 30 frames, 0.925 s |
| Generation time | **342.81 s** wall (2 steps, 35 layers, `--ssd-streaming`) |
| Peak host RAM (RSS) | **33.81 GB** of 64 GiB |
| Swap used | **0** |
| Peak Metal allocation | 38.66 GiB reported live at final teardown; DiT phase peak live 1.500 GiB |
| GPU time | text encoder 33.68 s, DiT 78.73 s, video VAE 51.79 s, audio VAE 2.31 s |
| SSD streaming | 50.96 GiB read at 0.362 GiB/s, 45.33 s unhidden |

A second run with a different prompt and seed (bear/salmon, seed 1234) also
produced a coherent, prompt-matching video: 3892.89 s, peak RSS 35.34 GB, zero
swap, no GPU errors, identical 144.272 GiB streamed. All runs completed cleanly.

## What it took

Three findings mattered, and only the third was anticipated by the brief.

**1. There is no BF16 wall.** All 145 native MSL `bfloat` uses in h3.c live
inside the `H3_METAL_HAS_TENSOR` block, which is enabled only when the Metal
device *name* contains "M5". The 56 portable kernels store BF16 as `ushort` and
convert with bit arithmetic, accumulating in `float`. MPSGraph BF16 matmul is
additionally **bit-exact** on gfx1030 (`worst_rel = 0`). So the "h3.c Intel-AMD
FP16 compatibility experiment" the brief scoped was **unnecessary** — no dtype
port was written, and none was needed.

**2. The real blocker was bus bandwidth, not numerics.** h3.c allocates every
buffer `MTLResourceStorageModeShared`. On Apple Silicon that is free. On an
external GPU it means host memory reached over Thunderbolt. Measured here:

| Storage mode | GPU read bandwidth |
|---|---|
| `Shared` (host memory over Thunderbolt) | **1.76 GB/s** |
| `Private` (VRAM) | **478 GB/s** |

Because `h3_linear_bf16` tiles 16x16 and rereads its weight once per 16-row
block, a 294 MiB FC1 weight becomes many GiB of bus traffic. Timing **h3's own
kernel** at H3's real DiT dimensions:

| Shape | rows | weight | Shared | Private | speedup |
|---|---|---|---|---|---|
| QKV 5376->21504 | 256 | 220.5 MiB | 6.584 s | 0.049 s | **133x** |
| QKV 5376->21504 | 1024 | 220.5 MiB | 23.106 s | 0.199 s | 116x |
| FC1 5376->28672 | 256 | 294.0 MiB | 8.503 s | 0.066 s | 130x |
| FC1 5376->28672 | 1024 | 294.0 MiB | **GPU TIMEOUT** | 0.252 s | — |

macOS runs a GPU progress watchdog. A dispatch that stalls on the Thunderbolt
link gets aborted, the channel restarted, and after two restarts every further
submission from the process is denied:

```
kernel (IOAcceleratorFamily2) checkGPUProgress() - Signaling hardware error on channel 9
kernel (AMDRadeonX6000HWLibs) AMD Error: Restart Channel: 9 ComputeUQ6
kernel Trying to restart GPU (AMD Radeon RX 6900 XT)...
kernel Deny Submissions/ignore app[h3] with 2 GPURestarts in 56 submissions.
h3 (Metal) Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)
```

**3. Two localized patches fixed it**, both no-ops on Apple Silicon:

- `h3_gpu_tensor_new_bf16_resident()` puts the streamed DiT weights in private
  VRAM with a host-visible staging buffer, and the SSD streaming path `pread`s
  into staging then blits across. The two stream slots are ~770 MiB each, which
  fits comfortably in 16 GiB.
- `h3_gpu_require_command()` bounds work per command buffer and caps in-flight
  buffers on a non-unified-memory device, so the watchdog never sees a long
  submission. Metal executes same-queue command buffers in commit order, so
  results are unchanged.

Plus explicit Metal device selection, because an Intel Mac has two GPUs.

## Risks that did not materialise

| Anticipated blocker | Actual result |
|---|---|
| BF16 unsupported on AMD Metal | Not a dependency; portable path emulates BF16 in `ushort`+`float`. MPSGraph BF16 also exact. |
| `maxBufferLength` only 3.50 GiB | Advisory only. Single buffers up to **16 GiB** allocate successfully. |
| `newBufferWithBytesNoCopy` needs page-aligned pointers | This driver accepts **all** unaligned offsets tested, and the GPU reads mmap'd bytes correctly. |
| PyTorch RDNA2 `F.linear` corruption (#178697) | **Did not reproduce** on torch 2.2.2 here. |
| ROCm gfx1030 NaN corruption (#6123) | Not reached — Linux path was never needed. |

---

## Detected Hardware

| Property | Value | How I detected it |
|---|---|---|
| Mac model | **Macmini8,1** (Mac mini 2018) | `sysctl -n hw.model` |
| CPU | **Intel Core i7-8700B @ 3.20 GHz**, 6 cores / 12 threads | `sysctl machdep.cpu.brand_string` |
| Architecture | **x86_64** (`hw.optional.arm64` does not exist) | `uname -m`, `sysctl` |
| RAM | **68,719,476,736 bytes = 64.00 GiB** | `sysctl hw.memsize` |
| macOS | **15.7.9 Sequoia**, build 24G830, kernel 24.6.0 | `sw_vers`, `uname -a` |
| SIP | **enabled** (untouched) | `csrutil status` |
| System extensions | **0 loaded** | `systemextensionsctl list` |
| Toolchain | Apple clang 17.0.0, `x86_64-apple-darwin24.6.0` | `clang --version` |
| eGPU | **AMD Radeon RX 6900 XT**, 16 GB, External GPU, Removable | `system_profiler SPDisplaysDataType` |
| Vendor / Device ID | **0x1002 / 0x73bf**, Revision 0x00c0 | `system_profiler`, `ioreg` (`IOName = pci1002`) |
| GPU ROM | 113-69XB6SSB1-C02 | `system_profiler` |
| Metal architecture | **`amdgpu_gfx1030`** (RDNA2 / Navi 21) | `MTLDevice.architecture.name` |
| Metal registryID | **0x1000008fe** | `MTLDevice.registryID` |
| Metal device index | **0** of 2; also the *system default* device | `MTLCopyAllDevices()` |
| Enclosure | **Razer Core X Chroma** (Vendor ID 0x127) on Thunderbolt Bus 0, Receptacle 1 | `system_profiler SPThunderboltDataType` |
| TB link | **20 Gb/s negotiated** (not 40), PCIe x16 to the card | `system_profiler`; `maxTransferRate = 5.0 GB/s` |
| Driver | **Apple's own** `AMDRadeonX6000_AMDNavi21GraphicsAccelerator` | `ioreg -rc IOAccelerator` |
| Kexts | `com.apple.kext.AMDRadeonX6000` 7.0.0, `AMDRadeonX6000Framebuffer`, `AMDSupport`, `AMDRadeonX6000HWLibs` — all Apple-signed | `kmutil showloaded` |
| Second GPU | Intel UHD Graphics 630, Device ID 0x3e9b, 1.50 GiB | `metal_probe` |
| Disk | root volume 932 GiB, **~451 GiB free** at time of writing | `df -h /` |

The card matches Apple's officially supported 6900 XT Device ID `0x73BF` exactly,
which is why no third-party driver work was needed or attempted.

### Metal device table (`bin/metal_probe`)

| | Device 0 | Device 1 |
|---|---|---|
| name | AMD Radeon RX 6900 XT | Intel(R) UHD Graphics 630 |
| registryID | 0x1000008fe | 0x100000891 |
| system default | **yes** | no |
| removable / location | yes / 2 (external) | no / 0 (built-in) |
| hasUnifiedMemory | no | yes |
| lowPower | no | yes |
| recommendedMaxWorkingSetSize | **15.98 GiB** | 1.50 GiB |
| maxBufferLength (reported) | 3.50 GiB | 2.00 GiB |
| maxThreadgroupMemoryLength | 65536 B | 65536 B |
| architecture.name | `amdgpu_gfx1030` | (null) |
| Apple GPU family | **0 — not an Apple GPU** | 0 |
| MTLGPUFamilyMetal3 | yes | yes |
| MTLGPUFamilyMetal4 | n/a (needs macOS 26) | n/a |
| name contains "M5" | no → `H3_METAL_HAS_TENSOR` **not** defined | no |

`preferRemovable` is the active GPU selection policy (macOS logs
`apply_selection_policy_once: prefer use of removable GPUs`), and the 4K display
is attached to the eGPU. That is why `MTLCreateSystemDefaultDevice()` already
returns the AMD card — upstream h3.c would pick the right GPU here by luck. The
patch makes it deterministic rather than lucky.

---

## Feasibility Matrix

| Path | GPU detected | Compute correct | H3 starts | H3 coherent | Verdict |
|------|--------------|-----------------|-----------|-------------|---------|
| macOS Metal + h3.c | **YES** (Metal dev 0, gfx1030) | **YES** — 1769 checks + BF16/FP16/FP32 exact | **YES** | **YES** — coherent fox-in-snow, h264+AAC | **SUCCESS (with 2 localized patches)** |
| macOS PyTorch MPS | YES (MPS available) | **NO** — `BFloat16 is not supported on MPS` | no H3 MPS runtime exists | n/a | **UNSAFE / NOT APPLICABLE** — and newest Intel-Mac wheel is torch 2.2.2 |
| MLX / mlx-serve | N/A | N/A | N/A | N/A | **NOT APPLICABLE** — upstream requires "macOS 26.2+ on Apple Silicon" |
| TinyGPU / tinygrad | no | n/a | n/a | n/a | **NOT VIABLE** — issue #15636 closed, RX 6900 XT PSP bootloader timeout; also needs DriverKit + SIP changes I was told not to make |
| MoltenVK | **YES** — GPU0 `DISCRETE_GPU` `0x1002:0x73bf`, Vulkan 1.4.357 | not tested further (no reason to) | **no H3 Vulkan runtime exists** | n/a | **GPU compute available, but no MiniMax H3 backend** |
| Linux ROCm + ComfyUI H3 | not run | not run | not run | not run | **Prepared only; requires manual external-boot + Startup Security checkpoint** |
| Linux ROCm + SGLang H3 | not run | not run | not run | not run | **Prepared only; 16 GiB VRAM is the open question** |

---

## macOS Native Results

### Upstream commit and local patch

- Upstream: `antirez/h3.c`, commit **`8974cc055ea9c02fcd14cc27dfda3e1027c05153`**
  ("Clarify SSD streaming memory and speed tradeoff", 2026-08-11 15:50:25 +0200).
- Local branch: **`intel-amd-egpu`**
  - `d295226` — Add explicit Metal device selection for multi-GPU Intel Macs
  - `d43ee61` — Relax Apple-only GPU-family assertion in the device probe test

Total diff vs upstream: **162 insertions, 3 deletions** across 5 files.

```
 Makefile           |   2 +-
 h3_device_select.h |  35 +++++++++++
 h3_device_select.m | 120 +++++++++++++++++++++++++++++++++++++++
 h3_gpu.m           |   3 +-
 h3_metal.m         |   5 ++-
```

### Baseline build, unmodified upstream, x86_64

`make -j12` on the untouched upstream tree: **exit 0, 0 errors, 0 warnings.**
There are no architecture guards, no `#ifdef __arm64__`, and no Apple-Silicon
preprocessor gates in the C/ObjC sources. The Makefile passes no `-arch` flag.
The only Intel-hostile thing in the whole tree was one test assertion.

### Where h3.c assumed Apple Silicon — full audit

| Assumption | Site | Real impact | Resolution |
|---|---|---|---|
| One Metal GPU, take the default | `h3_metal.m:19`, `h3_gpu.m:335` — the **only** two `MTLCreateSystemDefaultDevice()` calls | Two independent probe/inference paths could disagree on a multi-GPU Mac | Both routed through `h3_select_metal_device()` |
| `bfloat` MSL type | 145 uses, **all** inside `#ifdef H3_METAL_HAS_TENSOR` (lines 848–2865 of `h3_shaders.metal`) | **None.** Enabled only when the device name contains "M5" | Nothing to do — auto-gated off |
| `#include <metal_tensor>` + MetalPerformancePrimitives | Lines 2–5, guarded | Fails on macOS 15 SDK (Metal 4 is macOS 26) | Nothing to do — guarded |
| `apple_gpu_family > 0` | `tests/test_h3.c:383` | **Test-only.** Runtime never reads it; `main.c:134` only prints it | Assertion relaxed |
| `metal4` | `h3.c:562` | Gates `--use-int8-row-fc2` only, with a clean error | Do not use that flag |
| `hasUnifiedMemory` | `h3_metal.m:37` | Print-only (`main.c:136`) | No change needed |
| `newBufferWithBytesNoCopy` on a non-page-aligned mmap offset | `h3_gpu.m:649` | Would break `--ssd-streaming` if the driver enforced alignment | Verified: this driver accepts it |
| Buffers only ever `MTLResourceStorageModeShared` | `h3_gpu.m:559`, `:649` | On a discrete eGPU these live in host memory and cross Thunderbolt — a *performance* issue, not a correctness one | Documented; see throughput note |

### Portable vs M5 kernel split

`h3_shaders.metal` contains **56 portable kernels and 27 M5/TensorOps kernels**.
Upstream's claim that "older Metal hardware selects that path automatically" is
**VERIFIED on AMD Metal**, not just on older Apple GPUs: `h3_gpu.m:364` sets
`wantsTensorOps` from `[gpu.device.name rangeOfString:@"M5"]`, which cannot match
`"AMD Radeon RX 6900 XT"`, so `H3_METAL_HAS_TENSOR` is never defined and the
`bfloat`/`metal_tensor` block is never compiled.

Critically, the portable BF16 kernels do not use hardware BF16 at all. From
`h3_linear_bf16` in `h3_shaders.metal`:

```metal
kernel void h3_linear_bf16(device const ushort *input  [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]], ...)
    threadgroup float input_tile[16][16];
    ...
    sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
    output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
```

BF16 is a 16-bit *storage* format here, converted by bit manipulation, with all
arithmetic in `float`. Any Metal 3 GPU can run this.

### Shader compilation on the RX 6900 XT (`bin/h3_capability`)

| Test | Result | Detail |
|---|---|---|
| `h3_shaders.metal` portable path | **PASS** | 56 functions compiled |
| `h3_shaders.metal` with `H3_METAL_HAS_TENSOR` | n/a (expected) | `program_source:3:10: fatal error: 'metal_tensor' file not found` |

That is the *only* shader compiler error, it is on the optional path, and it is
an SDK-availability error rather than a hardware-capability error.

### Numerical capability on the RX 6900 XT — MACHINE-TESTED

| Probe | Result | Measurement |
|---|---|---|
| Metal **FP32** elementwise FMA | **PASS** | 16 Mi elems, NaN=0, max_abs_err = **0**, 1.70 GB/s |
| Metal **FP16** (`half`) elementwise FMA | **PASS** | 16 Mi elems, NaN=0, max_abs_err = 2.44e-4 |
| Metal **BF16-as-ushort** (h3.c's real style) | **PASS** | 16 Mi elems, NaN=0, **bit_mismatch = 0** vs host converter |
| MSL native `bfloat` type compiles | **PASS** | supported by this compiler even though h3.c does not need it |
| **MPSGraph FP32** matmul 512³ | **PASS** | worst_rel = **0** |
| **MPSGraph FP16** matmul 512³ | **PASS** | worst_rel = **0**, 50.6 GFLOP/s |
| **MPSGraph BF16** matmul 512³ | **PASS** | worst_rel = **0**, 45.9 GFLOP/s |
| **MPSGraph BF16** matmul 2048×3072×3072 | **PASS** | worst_rel = **0**, 41.7 GFLOP/s |
| Private (VRAM) allocation | **PASS** | 32.0 GiB allocated in 512 MiB chunks — Metal over-commits and pages |
| Largest single MTLBuffer | see below | 16 GiB OK, 24 GiB nil |

Summary line: `I ran 11 checks on "AMD Radeon RX 6900 XT": 11 passed, 0 failed.`

**A correction worth recording.** My first capability run reported 13 NaNs in the
FP16 test. I did not attribute that to the GPU. `src/fp16_forensics.m` isolated
it: my test-data generator wrote `(float)((i % 1000) - 500)` where `i` is
`NSUInteger`, so the subtraction wrapped in *unsigned* arithmetic to ~1.8e19,
overflowed FP16 to `+Inf`, and the 13 "NaNs" were the GPU correctly computing
`fma(Inf, 0, Inf)`. With the cast fixed, five consecutive identical runs gave
`NaN/Inf=0, wrong=0, max_abs_err=0`. **The fault was mine, and gfx1030 FP16 is
clean and deterministic.**

### Buffer limits — `maxBufferLength` is advisory here

`MTLDevice.maxBufferLength` reports 3.50 GiB, which would have been alarming
since h3.c never checks it. Direct measurement (`logs/07-bigbuffer.txt`):

| Requested | Shared | Private |
|---|---|---|
| 3.0 / 3.5 / 4.0 / 6.0 / 8.0 / 12.0 / 16.0 GiB | OK | OK |
| 24.0 GiB | nil | nil |

So the practical single-buffer ceiling is **≈16 GiB, not 3.5 GiB**, and the
reported cap is advisory on this driver.

### `--ssd-streaming` viability (`bin/nocopy_probe`)

| Test | Result |
|---|---|
| `BytesNoCopy`, page-aligned pointer, page-multiple length | **PASS** |
| `BytesNoCopy`, page-aligned pointer, non-page-multiple length | **PASS** (length rounded internally) |
| `BytesNoCopy`, unaligned pointer at +2, +64, +128, +512, +1024, +2048, +4095 | **PASS — all 7 accepted** |
| GPU reads the mmap'd file contents correctly | **PASS** — GPU sum == CPU sum |
| Allocation above `maxBufferLength` | **succeeded** (confirms the cap is advisory) |

This is what makes the brief's preferred configuration possible: h3.c can mmap a
BF16 DiT block at an arbitrary safetensors offset and hand it straight to the
eGPU without a copy.

### h3.c test suite on the eGPU — MACHINE-TESTED

Run as `H3_METAL_DEVICE_NAME="6900" make test`:

```
ok: 1769 checks
audio primitive weight norm      max abs 5.960464e-08
audio primitive Conv1d           max abs 5.960464e-08
audio primitive ConvTranspose1d  max abs 2.980232e-08
audio primitive SnakeBeta        max abs 5.960464e-08
audio primitive scaled add       max abs 0
audio primitive clip             max abs 0
ok: native AudioVAE Metal primitives match host references
ok: concurrent FFmpeg video/PCM pipes created /tmp/h3-av-mux-test.mp4 (51766 bytes)
```

Remaining suite entries are `skip:` because they need either the `misc/fixtures`
MLX parity fixtures (not distributed in the repo) or released weights that are
still downloading. Those are the `parity` / `real-parity` targets and are the
right next step once the snapshot lands.

### Device selection — all 9 modes verified

`bin/select_test` links the real `libh3.a` and calls the shipped selector:

| Environment | Selected | Correct? |
|---|---|---|
| *(none)* | AMD Radeon RX 6900 XT — "MTLCreateSystemDefaultDevice (upstream default)" | yes |
| `H3_METAL_DEVICE_NAME=6900` | AMD Radeon RX 6900 XT | yes |
| `H3_METAL_DEVICE_NAME="radeon rx 6900 xt"` | AMD Radeon RX 6900 XT (case-insensitive) | yes |
| `H3_METAL_DEVICE_NAME=UHD` | Intel(R) UHD Graphics 630 | yes — proves it really switches |
| `H3_METAL_DEVICE_NAME="RTX 4090"` | **nil, refused to fall back**, listed both devices | yes |
| `H3_METAL_REGISTRY_ID=0x1000008fe` | AMD Radeon RX 6900 XT | yes |
| `H3_METAL_REGISTRY_ID=0xdeadbeef` | **nil, refused to fall back** | yes |
| `H3_METAL_DEVICE_INDEX=1` | Intel(R) UHD Graphics 630 | yes |
| `H3_METAL_DEVICE_INDEX=7` | **nil**, "out of range (2 device(s) present)" | yes |

Startup banner:

```
I selected Metal GPU:
  name                  = AMD Radeon RX 6900 XT
  registryID            = 0x1000008fe
  architecture          = amdgpu_gfx1030
  unifiedMemory         = no
  removable             = yes
  lowPower              = no
  recommendedWorkingSet = 15.98 GiB
  maxBufferLength       = 3.50 GiB
  selected by           = H3_METAL_DEVICE_NAME
```

### CLI syntax verification

I checked `./h3 --help` on the built binary rather than trusting the brief's
flags. Every flag proposed for the smoke test exists: `-d`, `-p`, `-o`,
`--width`, `--height`, `--frames`, `--steps`, `--layers`, `--reuse`,
`--ssd-streaming`, `--seed`, `--profile`. Two notes:

- There is **no `--task` flag**. T2VA is implied by using the `FL2VA` weights
  with no `--first-frame` / `--last-frame` / `--ref-*` argument.
- `--show` is documented "(M5)" and is left off, as instructed.
- Defaults are 864×480, 56 frames, 20 steps — the smoke test overrides all three.

### Model download sizing

From the Hugging Face API (`MiniMaxAI/MiniMax-H3`, sha
`42ed227ee7df40d41602854ae760620d6eb651fe`, lastModified 2026-08-13): 280 files,
464.24 GiB for the whole repo. h3.c reads `FL2VA/` for this task (it references
`FL2VA/tokenizer/tokenizer.json`, `FL2VA/text_encoder`, `FL2VA/transformer`,
`FL2VA/audio_vae`, `FL2VA/video_vae/source`) and only touches `Ref2VA/` for
reference conditioning.

| FL2VA component | Files | Size |
|---|---|---|
| `text_encoder` (Qwen3-VL, 14 shards) | 23 | 62.14 GiB |
| `transformer` (DiT, 13 shards) | 15 | 61.73 GiB |
| `video_vae/source` | 2 | 9.70 GiB |
| `audio_vae` | 13 | 0.56 GiB |
| `processor`, `tokenizer`, configs | 28 | 0.02 GiB |
| **FL2VA total** | **81** | **134.16 GiB** |

```
free disk:              473 GiB (at download start)
required download:      134.16 GiB  (FL2VA only; Ref2VA deliberately skipped)
expected working space: ~2 GiB tracked DiT storage with --ssd-streaming,
                        plus page cache; outputs are a few MiB
safety margin:          ~339 GiB
```

Margin is ample. The official release is safetensors/BF16; no GGUF/ONNX variant
was invented or used.

### The discrete-GPU failure, and how I diagnosed it

The first real run failed, and the way it failed is worth recording because the
symptom pointed at memory and the cause was bandwidth.

**Symptom.** The Qwen text encoder and the DiT load both completed. The first
Euler denoise step then sat for 63 s doing no GPU work at all and aborted:

```
denoise 0/2  Error: command buffer exited with error status.
h3 profile: H3 DiT Euler denoise wall=64.283s wait=63.425s root-gpu=0.000s
h3: Metal live allocation at GPU teardown: 29.007 GiB
h3: submit streamed DiT block: Metal command failed: Ignored (for causing
    prior/excessive GPU errors) (kIOAccelCommandBufferCallbackErrorSubmissionsIgnored)
```

**Step 1 — find the real error.** `SubmissionsIgnored` is a follow-on error.
`/usr/bin/log show` gave the first one:

```
kernel (IOAcceleratorFamily2) checkGPUProgress() - Signaling hardware error on channel 9
kernel (AMDRadeonX6000HWLibs) AMD Error: Restart Channel: 9 ComputeUQ6
kernel Trying to restart GPU (AMD Radeon RX 6900 XT)...
kernel Deny Submissions/ignore app[h3] with 2 GPURestarts in 56 submissions.
h3 (Metal) Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)
```

So it was a **watchdog timeout**, not an allocation failure. `checkGPUProgress`
fired at ~5.1 s intervals; after two channel restarts macOS refused all further
submissions from the process.

**Step 2 — rule out command-buffer aggregation.** h3's `h3_gpu_continue` chains
encoders and only drains at submit. I added a bounded flush and retried at one
operation per command buffer: still timed out, with only
`submissions=11 direct=6 linear=4 attention=1`. So a **single dispatch** was
hanging, not an accumulation of them.

**Step 3 — rule out a memory-reporting artefact.** The 29 GiB figure on a
16 GiB card was suspicious, so I checked whether `currentAllocatedSize` reports
current residency or a high-water mark (`logs/17-reclaim.txt`): allocating 8 GiB
and releasing it exactly the way `h3_gpu_tensor_free` does returned the counter
to `0.000 GiB` across three rounds. The 29 GiB was therefore real, and host RSS
(29.2 GB) matched it — these were **host-memory `Shared` buffers**, not
over-subscribed VRAM.

**Step 4 — find where the 29 GiB comes from.** I instrumented allocations over
64 MiB (`H3_DEBUG_GPU_ALLOC`). The AdaLN precompute allocates **496.1 MiB per
DiT block** — exactly `time_embed_dim 2688 x adaln_out_features 96768 x 2` — and
retains all 50, i.e. **24.2 GiB**. Those are cold during denoise, so they were
not the cause, but they explain the number.

**Step 5 — measure the bus.** Storage mode turned out to be everything:

| Storage mode | GPU read bandwidth (`logs/18-bandwidth.txt`) |
|---|---|
| `Shared` (host memory over Thunderbolt) | **1.76 GB/s** (flat at 256 MiB / 1 GiB / 2 GiB) |
| `Private` (VRAM) | **318-478 GB/s** |

**Step 6 — reproduce the timeout in isolation.** I ran h3's **own**
`h3_linear_bf16` kernel at H3's real DiT dimensions with the weight in each
storage mode (`logs/19-linear-bandwidth.txt`):

| Shape | rows | weight | Shared | Private | speedup |
|---|---|---|---|---|---|
| QKV 5376->21504 | 256 | 220.5 MiB | 6.584 s | 0.049 s | **133x** |
| QKV 5376->21504 | 1024 | 220.5 MiB | 23.106 s | 0.199 s | 116x |
| FC1 5376->28672 | 256 | 294.0 MiB | 8.503 s | 0.066 s | 130x |
| FC1 5376->28672 | 1024 | 294.0 MiB | **GPU TIMEOUT (same error)** | 0.252 s | — |

The mechanism: the kernel tiles 16x16 and reloads the weight tile for every
16-row block, so effective weight traffic is `weight_size x ceil(rows/16)`. A
294 MiB FC1 weight at 1024 rows is ~18 GiB of traffic; at 1.76 GB/s that is
~10 s, well past the watchdog. From VRAM the same work is 0.25 s.

That is a complete diagnosis: **AMD Metal executes every H3 operation correctly;
the problem was that h3.c keeps weights where an Intel Mac's eGPU reads them
slowly.**

### The fix

Two changes in `h3_gpu.m` (plus four call sites in `h3_dit.c`), both inert on
Apple Silicon because they are gated on `hasUnifiedMemory`:

1. **`h3_gpu_tensor_new_bf16_resident()`** allocates the payload as
   `MTLResourceStorageModePrivate` with a host-visible staging buffer alongside.
   `h3_gpu_tensor_read_file_bf16_mode` `pread`s into staging and then blits into
   VRAM. `allocate_stream_slot` uses it for the four hot-loop matrices
   (`qkv`, `out`, `fc1`, `fc2`), about 770 MiB per slot, two slots.
2. **Bounded command buffers.** `h3_gpu_require_command` — the single choke
   point ahead of every dispatch and every MPS operation — flushes after
   `autoFlushEvery` operations and caps in-flight buffers at `maxInflight`
   (defaults 4 and 2 on a non-unified-memory device, 0 = upstream elsewhere).
   Overridable with `H3_METAL_FLUSH_EVERY` / `H3_METAL_MAX_INFLIGHT`;
   `H3_METAL_NO_VRAM_WEIGHTS=1` disables change 1 for A/B testing.

### H3 generation results — smoke test, MACHINE-TESTED

`outputs/smoke-vram.mp4`, 256x160, 22 frames, 2 steps, 35 layers,
`--ssd-streaming`, seed 42, prompt "A red fox walks through fresh snow in a pine
forest." (`logs/20-smoke-vram.log`)

| Phase | wall | GPU time | peak live | notes |
|---|---|---|---|---|
| Qwen text encoder | 49.40 s | 33.68 s | 2.727 GiB | 251 submissions, 801 direct dispatches |
| H3 DiT load | 39.22 s | 17.08 s | 1.500 GiB | 15 of 50 blocks gate-skipped at `--layers 35` |
| H3 DiT Euler denoise | 139.57 s | 61.65 s | 1.500 GiB | 198 submissions, 280 linear, 70 attention |
| audio VAE decoder | 3.31 s | 2.31 s | 0.282 GiB | 136 conv dispatches |
| video VAE decoder | 108.19 s | 51.79 s | 0.461 GiB | 145 linear, 36 attention |
| **total** | **342.81 s** | — | — | exit 0 |

- SSD streaming: **50.96 GiB read in 140.87 s (0.362 GiB/s)**, 45.33 s unhidden.
- Peak host RSS: **33,806,462,976 B = 33.81 GB**. Swap: **0**.
- Metal live allocation at final teardown: 38.66 GiB (host-backed).
- `ffprobe`: stream 0 `h264` 256x160, **nb_frames=22**, duration 0.917 s;
  stream 1 **`aac`**, nb_frames=30, duration 0.925 s.

**Visual verification.** I extracted frames 0, 7, 14 and 21 into a 2x2 grid
(`outputs/frames/smoke-grid.png`) and inspected them. The frames show
snow-covered ground in the lower half, a dark pine-forest canopy in the upper
half, and a small orange-brown subject centred and shifting slightly between
frames. Detail is soft, which is the expected consequence of 2 denoising steps
(default is 20) and 35 of 50 DiT blocks. **This is a coherent scene matching the
prompt, not noise.**

### Target configuration run — MACHINE-TESTED

Launched through the deliverable launcher:

```bash
./generate.sh "A red fox walks through fresh snow in a pine forest. \
Medium tracking shot, natural winter light, realistic fur, \
soft footsteps in snow and gentle wind in the trees." \
  608 352 22 4 h3-egpu-test.mp4 42
```

| Phase | wall | GPU time | peak live | dispatches |
|---|---|---|---|---|
| Qwen text encoder | 147.52 s | 131.19 s | 2.730 GiB | 451 direct, 350 linear |
| H3 DiT load | 38.56 s | 19.76 s | 1.678 GiB | 75 direct, 9 linear, 2 attention |
| H3 DiT Euler denoise | 1934.07 s | 666.94 s | 1.678 GiB | 628 direct, 800 linear, 200 attention |
| audio VAE decoder | 2.63 s | 2.34 s | 0.282 GiB | 340 direct, 136 conv |
| video VAE decoder | 956.27 s | 455.33 s | 9.365 GiB | 1314 direct, 870 linear, 216 attention; tiled 3x2 at 256 px |
| **total** | **3081.63 s** | **1275.6 s** | — | exit 0 |

- Peak host RSS **34,703,155,200 B = 34.70 GB**; `swaps 0`; `vm.swapusage used = 0.00M`.
- SSD streaming moved **144.272 GiB** during denoise (200 block loads: 4 steps x 50 blocks), 130.007 s of it unhidden.
- `ffprobe`: stream 0 `h264` **608x352**, `nb_frames=22`, 3,085,352 bit/s; stream 1 `aac`, `nb_frames=30`, 167,548 bit/s.

**Visual verification.** Frames 0, 7, 14 and 21 in `outputs/frames/target-grid.png`
show a photorealistic red fox in profile walking left-to-right through fresh
snow, against a pine forest with visible trunks and needled canopy. Fur texture,
the black lower legs and white tail tip are all anatomically correct red-fox
markings; the subject's identity is stable across frames while the leg positions
advance through a real walking gait; lighting is consistent directional winter
sun. This matches the prompt including the requested medium tracking shot.

**Where the time goes.** GPU time is only 1275.6 s of the 3081.63 s wall, i.e.
**59% of the run is spent waiting on the Thunderbolt link and the SSD**, not
computing. The denoise phase alone waited 1803.69 s against 666.94 s of GPU
work. This is the expected consequence of the remaining `Shared` weights (Qwen
text encoder and the 24.2 GiB of AdaLN matrices) plus the 20 Gb/s link.

### Second seed and prompt — reproducibility check, MACHINE-TESTED

The brief asks for a second seed/prompt before declaring success. Different
subject, different scene type, different seed:

```bash
./generate.sh "A brown bear catches a salmon in a rushing mountain river. \
Close tracking shot, golden late-afternoon light, water spray in the air, \
sound of rushing water and splashing." 608 352 22 4 h3-egpu-test2.mp4 1234
```

| Phase | run 1 (fox, seed 42) | run 2 (bear, seed 1234) |
|---|---|---|
| Qwen text encoder | 147.52 s | 463.64 s |
| DiT load | 38.56 s | 81.07 s |
| DiT Euler denoise | 1934.07 s (root-gpu 666.94 s) | 2366.17 s (root-gpu 770.59 s) |
| audio VAE | 2.63 s | 3.29 s |
| video VAE | 956.27 s | 976.17 s |
| **total** | **3081.63 s** | **3892.89 s** |
| SSD streamed | 144.272 GiB | 144.272 GiB |
| peak RSS | 34.70 GB | 35.34 GB |
| swap | 0 | 0 |
| GPU errors | none | none |
| output | 375,711 B, h264 608x352 22f + AAC 30f | 679,681 B, h264 608x352 22f + AAC 30f |

Run 2 is ~26% slower in wall time, and the text encoder 3x slower, because
three concurrent LLM API calls and analysis scripts were competing for CPU and
memory during it. GPU-side numbers are within ~16%, the streamed byte count is
identical, and the DiT peak live allocation is identical at 1.678 GiB. No GPU
restarts or timeouts in either run.

**Visual verification** (`outputs/frames/target2-grid.png`): a brown bear with
its head plunged into whitewater, rim-lit by low golden sun, spray suspended in
the air, against a dark far bank. The bear's head lowers progressively across
frames 0/7/14/21. Matches the prompt including the requested close tracking shot
and golden late-afternoon light. **Coherent, and a different subject and scene
class from run 1 — so the pipeline is not producing one memorised output.**

**Artifact check.** I noticed a faint regular cross-hatch in flat low-detail
regions near the bottom edge, so I measured whether it was a VAE tile seam. It
is not: the `3x2 at 256 px` tiling puts a horizontal seam at y=175/176, and the
row-to-row brightness discontinuity there is **2.52**, *smaller* than natural
image gradients elsewhere in the same frame (4.0-5.9 at y=100, 122, 298, 306,
327, 345). Column jumps cluster on image content, not on the expected x=203/405
seams. The tiling blends correctly; the cross-hatch is a VAE upsampling
(checkerboard) artifact and is expected at only 4 denoising steps.

### A launcher bug I hit and fixed

`generate.sh` initially failed in 1 s with:

```
h3: cannot compile h3_shaders.metal: The file "h3_shaders.metal" couldn't be
opened because there is no such file.
```

h3 loads its shader source relative to the working directory, and the path is a
hardcoded literal at 12 sites in `h3.c`. Rather than churn upstream code I run
h3 from its own source tree in a subshell and pass every other path absolutely.

---

## Linux Results

**Not run. Prepared but requires a manual external-boot / Startup Security
checkpoint.**

Because the macOS Metal path cleared every capability gate, the Linux fallback
was not needed and no destructive step was taken. Specifically I did **not**:
erase or repartition any disk, run `diskutil eraseDisk` or `dd` against a
physical device, touch an EFI partition, modify NVRAM boot arguments, alter SIP
or Secure Boot policy, install OpenCore/NootRX/Hackintosh kexts, or load any
third-party kernel extension. SIP remains `enabled` and 0 system extensions are
loaded.

See `linux/LINUX_EXTERNAL_SETUP.md` for the prepared procedure and
`linux/linux_postinstall.sh` for the first-boot diagnostic script.

---

## Source Evidence

| Source | Accessed | What it proved | Status |
|---|---|---|---|
| `github.com/antirez/h3.c` @ `8974cc0` (2026-08-11) | 2026-08-21 | Real project; MiniMax-H3 native Metal; "Apple Silicon" framing; `--ssd-streaming` keeps 2 DiT blocks resident (~36.5 → ~2.0 GiB at 512²); "older Metal hardware selects that path automatically" | **MACHINE-TESTED** — builds and its tests pass on x86_64 + gfx1030 |
| `h3_shaders.metal` (local, same commit) | 2026-08-21 | All 145 `bfloat` uses are inside `H3_METAL_HAS_TENSOR`; 56 portable vs 27 guarded kernels; portable BF16 is `ushort`+`float` | **MACHINE-TESTED** |
| `h3_gpu.m:364` (local) | 2026-08-21 | TensorOps gated purely on the device name containing "M5" | **MACHINE-TESTED** |
| `huggingface.co/MiniMaxAI/MiniMax-H3` API, sha `42ed227e` | 2026-08-21 | 280 files / 464.24 GiB total; FL2VA subset = 134.16 GiB; safetensors/BF16 | **VERIFIED** |
| Apple `MTLDevice` documentation | 2026-08-21 | `registryID`, `isRemovable`, `location`, `recommendedMaxWorkingSetSize`, `maxBufferLength` semantics used by `metal_probe` | **VERIFIED** (exercised via the API itself) |
| `support.apple.com/en-euro/102363` (Apple eGPU support) | 2026-08-21 | Apple supports specific 6900 XT variants with Device ID **0x73BF** | **MACHINE-TESTED** — this card reports exactly `0x73bf`, and Apple's own `AMDRadeonX6000` kexts bind it |
| `github.com/ddalcu/mlx-serve` | 2026-08-21 | "Needs macOS 26.2+ on Apple Silicon." Does list MiniMax-H3 4-bit/8-bit, but Apple Silicon only | **VERIFIED — NOT APPLICABLE (x86_64 Intel host)** |
| `github.com/pytorch/pytorch/issues/141864` | 2026-08-21 | Closed **as not planned**; "BFloat16 is not supported on MPS" on Intel + AMD | **MACHINE-TESTED — reproduced exactly**: `TypeError: BFloat16 is not supported on MPS` |
| `github.com/pytorch/pytorch/issues/178697` | 2026-08-21 | Closed **as not planned**; `F.linear` catastrophically wrong on RDNA2 for `in_features ≥ 256`; `torch.mm` fine | **NOT REPRODUCED on this machine** with torch 2.2.2 — max rel err ≤ 2.1e-6 (fp32) and ≤ 4.0e-4 (fp16) at in_features 64→3072 for both `mm` and `F.linear` |
| PyPI `torch` index for macOS x86_64 | 2026-08-21 | Newest available wheel is **2.2.2** (`torch-2.2.2-cp39-none-macosx_10_9_x86_64.whl`); Intel-Mac builds discontinued after it | **MACHINE-TESTED** |
| `github.com/tinygrad/tinygrad/issues/15636` | 2026-08-21 | **Closed.** RX 6900 XT (`0x73bf`) enumerates but fails PSP bootloader init; needs bridge/secondary-bus reset TinyGPU does not expose | **VERIFIED — NOT VIABLE** (and no DriverKit/SIP change attempted, per instructions) |
| `github.com/KhronosGroup/MoltenVK` | 2026-08-21 | Vulkan-over-Metal for macOS incl. Intel; README documents no ML/compute-inference use and no H3 backend exists | **MACHINE-TESTED** — MoltenVK 1.4.2 via Homebrew enumerates the eGPU as GPU0, `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, `vendorID 0x1002`, `deviceID 0x73bf`, apiVersion 1.4.357, `DRIVER_ID_MOLTENVK`. Vulkan compute is genuinely available; there is simply no MiniMax H3 Vulkan runtime to point at it. Not pursued further, per instructions. |
| `/usr/bin/log show` (macOS unified log, `IOAcceleratorFamily2` / `AMDRadeonX6000HWLibs`) | 2026-08-21 | The denoise failure's true first error: `checkGPUProgress()` watchdog -> `AMD Error: Restart Channel: 9 ComputeUQ6` -> `Trying to restart GPU` -> `Deny Submissions/ignore app[h3] with 2 GPURestarts in 56 submissions` -> `kIOAccelCommandBufferCallbackErrorTimeout`. h3 only surfaces the follow-on `SubmissionsIgnored` error, so the system log was essential. | **MACHINE-TESTED** |
| `logs/17-reclaim.txt` (own probe) | 2026-08-21 | `MTLDevice.currentAllocatedSize` on this driver reports live residency, not a high-water mark: 8 GiB allocated then released the way `h3_gpu_tensor_free` does returns the counter to 0.000 GiB, three rounds. This ruled out a reporting artefact. | **MACHINE-TESTED** |
| `logs/18-bandwidth.txt` (own probe) | 2026-08-21 | GPU read bandwidth by storage mode on this eGPU: `Shared` **1.76 GB/s** (flat across 256 MiB / 1 GiB / 2 GiB), `Private` **318-478 GB/s**. The 270x gap is the whole story. | **MACHINE-TESTED** |
| `logs/19-linear-bandwidth.txt` (h3's own `h3_linear_bf16` kernel) | 2026-08-21 | At H3's real DiT dimensions, FC1 5376->28672 at 1024 rows: **GPU timeout** from `Shared`, **0.252 s** from `Private`. QKV at 256 rows: 6.584 s vs 0.049 s (133x). Reproduced the production failure in isolation and proved the fix. | **MACHINE-TESTED** |
| `FL2VA/transformer/config.json` (local, sha `42ed227e`) | 2026-08-21 | `hidden_size 5376`, `num_layers 50`, `num_attention_heads 56`, `attention_head_dim 128`, `ffn_hidden_size 14336`, `time_embed_dim 2688`, `adaln_out_features 96768`. These fix the weight sizes that drive the bandwidth arithmetic. | **VERIFIED** |
| `H3_DEBUG_GPU_ALLOC` instrumentation (own patch) | 2026-08-21 | The AdaLN precompute allocates **496.1 MiB per DiT block** (= 2688 x 96768 x 2 B) and retains all 50 = **24.2 GiB**, which accounts for the 29 GiB live figure. Cold during denoise, so not the timeout cause. | **MACHINE-TESTED** |
| `ioreg -rc IOAccelerator`, `kmutil showloaded` | 2026-08-21 | Apple's `AMDRadeonX6000_AMDNavi21GraphicsAccelerator`; Apple-signed AMD kexts 7.0.0; zero third-party kexts | **MACHINE-TESTED** |

Sources deliberately **not** acted on: `tinygrad/docs/developer/am.md`,
`t2linux/T2-Ubuntu`, `T2-Debian-and-Ubuntu-Kernel`, `philmcneely/t2-egpu-linux`,
`ROCm/ROCm` + compatibility matrix, ROCm issues #6123 / #6567, the SGLang H3
cookbook, ComfyUI PR #15224, and the Spectrum-H3 32 GB success report. These are
all inputs to the Linux fallback, which the macOS result made unnecessary. They
are referenced in `linux/LINUX_EXTERNAL_SETUP.md` rather than re-summarised from
memory, because I did not want to assert current facts about them without
verification at execution time.

`engeldlgado/toshllm` was noted as Intel+AMD Metal precedent only; it is not an
H3 runtime and was not used.

---

## Errors Encountered

**1. GPU watchdog timeout in the DiT denoise loop — the real blocker (fixed).**

```
h3 (Metal) Execution of the command buffer was aborted due to an error during
execution. Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)
kernel (IOAcceleratorFamily2) checkGPUProgress() - Signaling hardware error on channel 9
kernel (AMDRadeonX6000HWLibs) AMD Error: Restart Channel: 9 ComputeUQ6
kernel Trying to restart GPU (AMD Radeon RX 6900 XT)...
kernel Deny Submissions/ignore app[h3] with 2 GPURestarts in 56 submissions.
```

Cause: weights in `Shared` (host) memory read over Thunderbolt at 1.76 GB/s,
multiplied by the 16x16 tile kernel's per-row-block weight reread. Fixed by
placing streamed DiT weights in private VRAM and bounding command buffers. See
"The discrete-GPU failure" above for the full diagnosis.

**2. Minimum frame count.**

```
h3: generation requires at least one trained 22-frame decoder chunk
```

A 5-frame request is rejected even though `h3.c:514` validates `frames >= 5`.
The real floor for generation is **22 frames**, which is exactly the brief's
target — so this constrained the smoke test, not the goal.

**3. Apple-Silicon-only test assertion (fixed).**

```
FAIL tests/test_h3.c:383: info.apple_gpu_family > 0
make: *** [test] Error 1
```

`apple_gpu_family` is non-zero only on an Apple GPU. Verified test-only: the
runtime never reads it. Relaxed in `d43ee61`.

**4. M5 TensorOps shader path unavailable (expected, harmless).**

```
program_source:3:10: fatal error: 'metal_tensor' file not found
#include <metal_tensor>
```

Metal 4 / macOS 26 headers. Gated behind `H3_METAL_HAS_TENSOR`, which is only
set when the device name contains "M5", so it is never compiled here.

**5. PyTorch MPS BF16 (upstream defect, reproduced).**

```
TypeError: BFloat16 is not supported on MPS
```

**6. `generate.sh` shader-path bug (mine, fixed).**

```
h3: cannot compile h3_shaders.metal: The file "h3_shaders.metal" couldn't be
opened because there is no such file.
```

h3 resolves its shader source against the working directory. The launcher now
runs h3 from its own source tree with all other paths absolute.

**7. My own test-harness bug (fixed, recorded for honesty).**

Unsigned wraparound in `(float)((i % 1000) - 500)` with `NSUInteger i` produced
`+Inf` FP16 inputs, and the resulting 13 "NaN" results were the GPU correctly
computing `fma(Inf, 0, Inf)`. Root-caused with `src/fp16_forensics.m`, fixed,
re-verified clean over five consecutive runs. **No GPU fault existed.**

## Reproduction

The final working sequence, on a Mac mini 2018 running macOS 15.7.9 with the
RX 6900 XT eGPU attached and powered before boot.

```bash
# 1. Toolchain
brew install ffmpeg                                  # ffmpeg 9.0.1

# 2. Workspace
mkdir -p ~/minimax-h3-egpu/{logs,src,bin,models,outputs,linux}

# 3. Confirm macOS exposes the eGPU to Metal (must print the RX 6900 XT)
cd ~/minimax-h3-egpu
xcrun clang -fobjc-arc -framework Foundation -framework Metal \
  src/metal_probe.m -o bin/metal_probe
./bin/metal_probe

# 4. Prove the operations h3.c needs, on that GPU specifically (11/11 pass)
xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  src/h3_capability.m -o bin/h3_capability
./bin/h3_capability 6900 src/h3.c/h3_shaders.metal

# 5. Build the patched h3.c  (branch intel-amd-egpu, base 8974cc0)
cd ~/minimax-h3-egpu/src/h3.c
make -j"$(sysctl -n hw.logicalcpu)"
H3_METAL_DEVICE_NAME="6900" make test          # ok: 1769 checks

# 6. Fetch only the FL2VA weights h3.c needs (134.16 GiB)
~/minimax-h3-egpu/bin/download_model.sh

# 7. Sanity-check the snapshot on the eGPU
H3_METAL_DEVICE_NAME="Radeon RX 6900 XT" ./h3 -d ../../models/MiniMax-H3 --info

# 8. Generate
cd ~/minimax-h3-egpu
./generate.sh "A red fox walks through fresh snow in a pine forest. \
Medium tracking shot, natural winter light, realistic fur, \
soft footsteps in snow and gentle wind in the trees." \
  608 352 22 4 h3-egpu-test.mp4 42
```

Useful knobs added by the patch:

| Variable | Effect |
|---|---|
| `H3_METAL_DEVICE_NAME` | select the Metal GPU by name substring; refuses to fall back |
| `H3_METAL_DEVICE_INDEX` | select by index into `MTLCopyAllDevices()` |
| `H3_METAL_REGISTRY_ID` | select by exact `registryID` (most precise) |
| `H3_METAL_FLUSH_EVERY` | operations per command buffer (default 4 on a discrete GPU, 0 = upstream) |
| `H3_METAL_MAX_INFLIGHT` | in-flight command buffer cap (default 2 on a discrete GPU) |
| `H3_METAL_NO_VRAM_WEIGHTS` | disable VRAM staging, for A/B testing the fix |
| `H3_DEBUG_GPU_ALLOC` | log every Metal allocation over 64 MiB with running device total |
| `H3_DEBUG_GPU_MEMORY` | log Metal live allocation at each phase boundary |

### Known limitations

- **Speed.** The Qwen text encoder and the 50 per-block AdaLN matrices
  (496.1 MiB each, 24.2 GiB total) are still `Shared`, so they are read over
  Thunderbolt. They stay under the watchdog, so this is a performance ceiling
  rather than a failure. Extending `h3_gpu_tensor_new_bf16_resident` to them is
  the clear next optimisation.
- **Thunderbolt link.** This enclosure negotiated **20 Gb/s, not 40**
  (`maxTransferRate = 5.0 GB/s`). A 40 Gb/s link should roughly halve the
  bus-bound phases.
- **`--use-int8-row-fc2`** requires an M5-class Metal 4 GPU and is correctly
  rejected here; it is also mutually exclusive with `--ssd-streaming`.
- **`--show`** is documented "(M5)" and is not used.
- **22 frames** is the minimum the released decoder supports.
