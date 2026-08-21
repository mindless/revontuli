Ranked, falsifiable hypotheses and the cheapest decisive check for each

1) Temporal attention mask is wrong once latent_t > 12 (broadcast/stride bug or window miscompute)
- Why this fits: Wrong from frame 0, non-progressive, presents as low-contrast mush rather than NaNs. Going from T=12 (clean) to T≥17/32 flips a shape/broadcast path. If the mask rows are all -inf (or almost all), softmax collapses and everything averages to near-constant. Tile seams can then show up in decode because latents are near-constant.
- Cheapest decisive check:
  - Instrument a single DiT block, single head (head 0), in the first denoise step:
    - After computing QK/√d and adding the temporal mask, before softmax:
      - For T=12 and T=32 (39 and 107-frame configs), dump for a couple of query time indices (t=0,1,2 and mid t):
        - count_finite = number of finite entries per row
        - min, max of the masked logits
      - Also after softmax, row_sum and Shannon entropy H = -Σ p log p.
    - Expected: For correct masking, count_finite equals the allowed temporal window (often T). If you see count_finite=0 or 1, or row_sum≈0, or entropy≈-inf/very low only at T≥17, you’ve found it. This requires no full video; one forward of one block is enough.

2) Temporal RoPE application is wrong beyond T=12 (wrong indexing, stride, or broadcast of sin/cos on the time axis)
- Why this fits: Immediate corruption from frame 0 when T exceeds threshold; two severities could be two different codegen tile sizes (107 vs 124) hitting different memory patterns. Mush rather than NaNs is typical if Q/K get rotated by garbage angles.
- Cheapest decisive check:
  - Compute on CPU a reference sin/cos table for the time axis of size [T, d_head/2] using the exact RoPE formula you intend to use (same theta).
  - Just before the first attention’s RoPE is applied on GPU, fetch (or mirror on CPU using the same builder) the exact sin/cos buffers the graph will use and md5 them. Also dump a tiny slice: sincos[0:4,0:8] for T=12 and T=32.
  - Additionally, dump norms of Q/K before and after RoPE per time index for head 0: mean/std across channels. If post-RoPE norms for T=32 are wildly different or NaN/Inf vs T=12, or the sin/cos buffers differ from CPU reference, you’ve found the bug.
  - A very cheap A/B: temporarily disable temporal RoPE (apply identity on time only) and run a 1-step denoise. If T=32 becomes “reasonable” instead of mush (even if worse quality), the RoPE path is implicated.

3) Cross-commit lifetime hazard due to MPSGraph commitAndContinue (intermediate buffers assumed to persist across root-command-buffer commit)
- Why this fits: You changed CB adoption; long sequences have larger graphs, more internal partitioning; a tensor used by later ops may cross a commit boundary and get re-used/overwritten. That would produce immediate garbage at T≥17/32 without timing anomalies. The 124-only encode blowup could be a different effect of the same code path (different tiling → more partitions).
- Cheapest decisive check:
  - Force strict synchronization and single-root semantics temporarily:
    - Do NOT adopt the new root CB returned by MPSGraph; instead, for the test, wrap the entire inner denoise step (or even a single DiT block) in a single command buffer; after encodeToCommandBuffer: call waitUntilCompleted before any CPU writes to resources the graph will see later.
    - Alternatively, insert a hard barrier by ending the command buffer after each graph encode and starting a fresh one yourself, carrying over all external MTLBuffers (feeds/fetches) explicitly.
  - If corruption at T=32 disappears under strict CB lifetime/sync (even if slower), you’ve isolated a lifetime hazard.
  - Even cheaper: make the graph’s feeds/fetches use your own long-lived MTLBuffers that you retain across commits (no internal MPSGraph-owned temp used later). If that fixes it, same conclusion.

4) Temporal positional embedding table (learned, not RoPE) is too short (e.g., length 12) and your code indexes beyond it without interpolation/padding
- Why this fits: Corruption starts exactly once you ask for more positions than the table contains. It would be wrong from frame 0 and not progressive. 39 (T=12) is fine; 56+ (T≥17) corrupt. Two severities could arise from different strides of OOB reads at different T.
- Cheapest decisive check (no generation):
  - Inspect weights on disk: list any tensors named like time_embed, pos_embed_t, temporal_pos_embed, temb, etc. Print their shapes. If you find an embedding of shape [L, C] with L=12 (or 32), you must either clamp/index properly or do interpolation.
  - If present and short, patch the code to clamp t indices to L-1 or to repeat the last position, and run a 1-step forward; if that turns mush into “bad but structured”, you’ve confirmed the table overrun.

5) Temporal attention layout/stride bug when T exceeds a tiled kernel’s implicit limit (e.g., 16-bit length/stride truncated, or miscomputed leading dimension)
- Why this fits: AMD path + “portable kernels” for non-matmul ops. Going from T=12 to T≥17 can flip a kernel to a different code path with 16-bit arguments overflowing or wrong bytes-per-row. Flat mush and seams are textbook symptoms of misstrided gathers/scatters, and 124 vs 107 differing aligns with different tile factors.
- Cheapest decisive check:
  - Audit all places where you pass sizes/strides/counts to Metal kernels: assert at runtime that every dimension, stride, and their products fit 32-bit, and that arguments you cast to ushort/uint16 never exceed 65535. Log when they do at T=12 vs T=32.
  - Enable Metal API Validation and runtime shader validation. On macOS 15 you can turn on GPU Validation; if an OOB access occurs, many times it will be flagged.
  - Micro-check: Write a tiny compute that reads/writes a 3D tensor [B, T, N] with the exact strides you use for Q/K/V and verify md5 after a roundtrip copy for T=12 and T=32. If T=32 fails, your strides are wrong.

6) Wrong temporal chunking window or index arithmetic in the VAE decode (time-chunk runner uses frame_count math instead of latent_t math once T>12)
- Why this fits: VAE memory is constant across runs → temporal chunking. If the first chunk’s indices are wrong for T≥17, frame 0 will already be wrong. The seams appearing only in longer runs hint at a decode path problem as well.
- Cheapest decisive check:
  - Bypass the VAE entirely: dump the DiT output latents for T=12 and T=32 at the same denoise step (e.g., after step 0) and compute:
    - Per-time L2 norm across C×H×W: ||z_t||2
    - Mean/std across channels per t
    - NaN/Inf count
  - If T=32 latents already have near-zero std or extreme values compared to T=12, the fault is in the DiT path, not VAE.
  - If DiT latents look statistically similar, decode the same saved latent tensor with a CPU/AS reference VAE (or your AMD VAE with tiny 1-chunk config). If decode now looks fine, the bug is in VAE’s temporal chunk indexing.

7) Numerical softmax/normalization pathology at larger T (underflow/over-suppression from mask scale or wrong sqrt(d) factor)
- Why this fits: Low-contrast mush, no NaNs, consistent from frame 0. If your mask adds very large negatives or you double-apply scale at larger T only, softmax rows go near-uniform or near-zero.
- Cheapest decisive check:
  - For T=12 and T=32, measure for the first attention layer:
    - Distribution of (QK/√d + mask) magnitudes: mean/std/min/max
    - Softmax row-sum and entropy
  - If at T=32 the logits are all ≪ -20 (after mask), you’ve effectively zeroed attention. Also verify exactly one √d is applied.
  - Check LayerNorm outputs’ mean/std at each block entrance/exit. If stats collapse only at larger T, it’s numerical; if they’re normal but outputs still mush, look back to masking/positioning.

8) Trained max temporal latent length exists (e.g., 12 or 32), and your extrapolation (if any) is wrong
- Why this fits: Many video DiTs use max temporal tokens (e.g
