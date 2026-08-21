Ranked, concrete, implementable optimizations

1) Stop writing/reading activations over Thunderbolt: make all activations Private and reuse a fixed pool
- What to change:
  - Audit every MTLBuffer used for intermediate tensors (layer outputs, attention Q/K/V/out, MLP activations, residuals, VAE feature maps). Allocate them storageModePrivate, not Shared.
  - Pre-allocate a per-phase activation pool from a MTLHeap in Private VRAM sized to the peak live working set of that phase. Then suballocate ping-pong buffers or alias via heap makesBuffer transiently.
  - For ops that today output into a caller-provided buffer, change to write into a pooled Private buffer and pass that to the next op. For linear/MLP/attention, keep only 2 buffers per tensor “stream” (src/dst ping-pong) plus a small workspace.
  - Set resourceOptions hazardTrackingModeUntracked for internal scratch/pool buffers and insert fences between passes: MTLFence encodeSignal/encodeWait or MTLSharedEvent encodeWaitForEvent on the compute queue.
- Estimated win:
  - You are currently taking the round-trip over TB3 for every activation between every kernel. That throttles memory-bound ops to ~1.76 GB/s. Moving activations to Private changes that path to 300–450 GB/s effective device bandwidth.
  - Expect 2–3x on memory-dominated parts:
    - Denoise GPU time: 666.9 s → ~250–350 s (save 300–420 s)
    - Video VAE GPU time: 455.3 s → ~150–220 s (save 235–305 s)
  - Also reduces CPU “wait” because kernels finish much faster and stop stalling the command queue.
- Risk/verification:
  - Risk: VRAM fragmentation if you keep ad-hoc buffers. Use a single MTLHeap and reuse/alias. Verify with MTLDevice.recommendedMaxWorkingSetSize (you’re under 16 GiB) and track heap usage.
  - Hazard bugs if you turn off hazard tracking. Verify by inserting fences at each ping-pong handoff and running with Metal API Validation; fuzz the inflight depth and check for mismatches at op boundaries with checksums on small test shapes.
- Effort: Medium. You already built a choke point for launches; extend that to a simple memory planner and swap buffer ownership.

2) Make streaming uploads truly asynchronous: separate blit queue + SharedEvent, triple buffering
- What to change:
  - Create a dedicated MTLCommandQueue for blits (copyQueue) in addition to computeQueue.
  - Maintain 3 stream slots (triple-buffer): each slot has a staging Shared buffer and a Private VRAM buffer for the 4 matrices and the AdaLN params for that block.
  - For each block N+1:
    - posix pread/mmap into staging[S].
    - On copyQueue: encodeCopy from staging[S] → private[P], then encodeSignalEvent(sharedEvent, value++).
  - On computeQueue (for block N+1 consumption): encodeWaitForEvent(sharedEvent, value) before first kernel using that weight.
  - No waitUntilCompleted on the CPU. Cap inflight by total bytes resident across the 3 slots (< ~4 GiB) rather than “command buffers M=2”.
  - Raise inflight command buffers to 4–6 on computeQueue. Keep each CB small enough to avoid watchdog (you already fixed the worst case).
- Estimated win:
  - Your denoise phase “wait” is 1803.7 s. Only ~130 s is raw SSD time; the rest is you serializing copy and compute via commit+wait on the same queue.
  - Overlap can recover 60–80% of that wait: 1,100–1,450 s saved on denoise wall.
  - Also covers Qwen/Video VAE streaming where present.
- Risk/verification:
  - Incorrect event ordering → use monotonically increasing event values per slot and assert they advance as expected; log