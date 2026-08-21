Here is the diagnosis and ranked hypotheses based on the constraints provided.

### 1. Ranked Hypotheses & Cheapest Decisive Checks

**Hypothesis A: MPSGraph AMD Compiler / Driver Bug (Explains both 107 & 124)**
*Reasoning:* Apple’s MLIR-based MPSGraph compiler for AMD RDNA2 (via Metal) is far less tested than on Apple Silicon. The 1582s CPU `encode` at 124 frames is a classic symptom of an MLIR compiler pass (e.g., loop unrolling or layout optimization) hitting a catastrophic heuristic boundary for a specific tensor shape. The "mush" at 107 frames could be the compiler generating a silently buggy kernel (e.g., wrong threadgroup sizing or shared memory exhaustion) just *before* the shape triggers the total compiler blowup at 124 frames.
*Cheapest Decisive Check:* **Run 107 frames on Apple Silicon (M1/M2/M3) using the identical upstream binary.** If it succeeds there but fails on the eGPU, the bug is isolated to the AMD Metal driver/MPSGraph fallback paths. 

**Hypothesis B: Discrete GPU Memory Sync (`commitAndContinue` barrier failure)**
*Reasoning:* You mentioned gating discrete GPU changes on `hasUnifiedMemory`. On Apple Silicon (UMA), memory visibility between command buffers is guaranteed implicitly. On a discrete eGPU via Thunderbolt, if an intermediate tensor’s memory is written by a block *before* a `commitAndContinue` and read by a block *after*, you must explicitly synchronize it (e.g., `MTLBlitCommandEncoder` sync or ensuring proper `MTLResource` hazard tracking). If hazard tracking drops across the commit boundary, the read might pull stale or zeroed PCIe memory, resulting in uniform garbage across the temporal axis.
*Cheapest Decisive Check:* **Disable `commitAndContinue` entirely for one 107-frame run**, forcing the entire graph into a single root command buffer. If it works, your discrete memory lifecycle or barrier logic across commits is flawed.

**Hypothesis C: Hardcoded Temporal Position Embedding (RoPE) Limit**
*Reasoning:* If the model was trained with a strict maximum temporal latent size (e.g., `t=24` or `t=32`), the C implementation might be reading past the end of a pre-computed RoPE frequency buffer. Reading uninitialized memory for frequencies causes attention logits to randomize globally, polluting frame 0 (due to bidirectional attention).
*Cheapest Decisive Check:* **Read the weights file metadata (do not generate video).** Dump the shape of the temporal position embedding or frequency cache tensor. If its maximum dimension is strictly `< 32`, you have your answer.

**Hypothesis D: Numerical Softmax Smearing (BF16 Accumulation)**
*Reasoning:* If attention is accumulated across `t`, the sum in the softmax denominator scales with sequence length. If a scaling factor (like $1/\sqrt{d}$) is missing or underflowing, or if a BF16 accumulator is accidentally used instead of `float` inside an MPSGraph custom block, the softmax denominator explodes. This forces all attention weights toward $1/N$, averaging all temporal tokens together (producing "flat brown mush").
*Cheapest Decisive Check:* **Dump the statistical variance of the DiT intermediate output after Step 0.** (See Question 5).

### 2. Is there a plausible trained max sequence length?
Yes. Many DiTs (including early Sora-alikes) cap 3D RoPE temporal indices during training. If the model uses learned positional embeddings rather than rotary, this limit is absolute.
*How to check without generating:* Inspect the model weights via `gguf-dump` or Python `safetensors`. Look for a tensor named `pos_embed`, `temp_embed`, `freqs_cis`, or similar. If its shape is `[32, ...]`, `[1, 32, ...]`, or `[24, ...]`, the model structurally cannot exceed that `latent_t` without algorithmic extrapolation (like NTK-aware scaling), which your C port might lack.

### 3. Numerical presentation vs. Indexing presentation
**Your reasoning on numerical presentation is mostly correct.**
A numerical underflow in attention (softmax smearing to $1/N$) or a collapsed RMSNorm variance will mathematically present as uniform, low-contrast mush. Extreme overflow (NaNs) presents as solid black or checkerboard noise.
*How to distinguish:* 
*   **Numerical Smearing:** The statistical variance of the latents will collapse toward zero. The mean will be finite but stable.
*   **Indexing/Positional Bug:** The variance of the latents will remain high (similar to the 22-frame run), but the spatial arrangement of the data is scrambled, or high-frequency noise is injected.

### 4. Single mechanism for both 107 (mush) and 124 (encode blowup)?
It is **highly unlikely** to be a pure C-level algorithmic bug (like an index math error) because an algorithmic math error does not cause an Apple framework (`encodeToCommandBuffer:`) to suddenly burn 7.9 seconds of CPU time per block. 
The single unifying mechanism is **MPSGraph compilation heuristics**. 
MPSGraph builds an MLIR intermediate representation. At `latent_t = 32`, the graph size or tensor dimensions produce a Metal shader that compiles fast but is semantically wrong on AMD hardware (e.g., an MLIR lowering bug specific to `gfx1030`). At `latent_t = 37`, that same dimension crosses a threshold where the compiler decides to aggressively unroll loops, flatten attention matrices, or alter memory layouts, causing the 1582s CPU compile time *and* producing a kernel that yields dark noise (or overflows VRAM internally leading to page thrashing).

### 5. Bisecting vs. Dumping
**Do not bisect 56/73/90.** It costs ~30 minutes of compute and will only tell you *where* the compiler or bounds check fails, not *what* failed.

**Dump intermediate latents instead.** It is decisive and takes exactly one generation step.
1.  **What to dump:** The raw FP32/BF16 latent tensor exiting the DiT *at the end of Step 0* (before it feeds into Step 1 or the VAE). 
2.  **What statistic to compute:** Compute the **Mean**, **Variance (or StdDev)**, **Min**, and **Max** across the spatial/channel dimensions for `frame[0]`, comparing the 22-frame run vs. the 107-frame run.
    *   *If Variance is ~0:* It is numerical softmax smearing.
    *   *If Max/Min are Inf/NaN:* It is numerical overflow.
    *   *If stats are identical but output is wrong:* It is a VAE temporal chunking bug or discrete memory synchronization issue where the DiT output is structurally sound but corrupted during handoff to the VAE.
