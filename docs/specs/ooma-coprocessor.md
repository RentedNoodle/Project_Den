# OMMA Coprocessor — Unified Inference/Cognition Architecture

**Date:** 2026-05-20
**Hardware:** RTX 5070 Ti (GB203-300-A1, SM120), 70 SMs, 16 GB VRAM, 48 MB L2
**Primary ISA:** OMMA.SF.16864 (mxf4nvf4, UE4M3, scale_vec::4X, m16n8k64, ~29 cycles)

---

## 1. Design Principles

- **Pluggable consumer:** Every optimization must improve pure inference performance independently. The companion is a zero-cost consumer riding on harvested cycles, never a dependency.
- **Hot-pluggable runtime:** Any consumer can be injected mid-session without engine restart or kernel recompilation.
- **No-compromise base path:** When no consumers are registered, the inference kernel executes with zero added instructions beyond a single predicated branch-not-taken per tile boundary.

---

## 2. SM Slot Architecture — The Compute Market

Every SM has 64 warp slots. Inference CTAs occupy 8-32. The remaining slots are **advertised slots** managed through a small constant-memory table (128 bytes, replicated to all 70 SMs).

### Slot Table Layout

```
Offset  Size    Field
0       2       slot[0].consumer_id    // 0 = empty, 1 = cognitive, 2 = integrity, ...
2       2       slot[0].tick_budget    // max cycles per invocation
4       4       slot[0].state_ptr      // offset into mapped state buffer
8       2       slot[1].consumer_id
10      2       slot[1].tick_budget
12      4       slot[1].state_ptr
...             ...
```

### Dispatch at Tile Boundary

Inlined at the end of every OMMA tile operation:

```
// Single predicated branch per slot — zero cost when slot is empty
for slot in slots:
    if slot.consumer_id != 0:
        warp = threadIdx.x % WARP_SIZE == 0
        if warp:
            consumer_dispatch(slot.consumer_id, slot.tick_budget, slot.state_ptr)
```

**Cost when empty:** ~4 instructions (load + branch + nop). Zero cycles on hardware with branch prediction.

---

## 3. Hot-Plug Protocol

### Registration (host-side, Rust/C++)

1. Consumer writes `Registration{ type_id, tick_budget, function_ptr_offset }` to mapped memory
2. Registrar performs single `cudaMemcpyAsync` (3 us) to constant memory slot table entry
3. Next tile boundary on every SM: consumer slot becomes active

### Unregistration (two-phase safe)

1. Consumer sets `consumer_id = 0` in its mapped struct
2. Next tile boundary: kernel sees empty slot -> stops dispatching
3. If consumer crashes: watchdog timer zeroes the slot, no stuck state

### No kernel recompilation

Consumer micro-kernels are precompiled `.cubin` fragments embedded in the engine binary. Registration picks an index into a function pointer table — no JIT, no runtime compilation, no CUDA graph rebuild.

---

## 4. Consumer Micro-Kernel ABI

### Signature

```cuda
__device__ void consumer_tick(
    uint32_t  slot_id,       // which slot this SM is servicing
    uint32_t  tick_budget,   // max cycles this invocation
    float*    local_state,   // per-SM persistent state (mapped)
    float*    global_state   // cross-SM merged state (mapped)
);
```

### Tick Contract

1. Read persistent state from `local_state[slot_id]`
2. Execute up to `tick_budget` cycles
3. Write back `local_state[slot_id]`
4. Return

### Cognitive Micro-kernel

```cuda
__device__ void cognitive_tick(
    uint32_t slot_id, uint32_t budget, float* local, float* global
) {
    // 15 landscape tiles per SM (256x256 / 16x16 / 70 SMs ~ 15)
    float4 tiles[15];  // registers, 960 bytes total
    
    // Read current tiles
    for (int i = 0; i < 15; i++)
        tiles[i] = local[slot_id * 15 + i];
    
    // OMMA blend: landscape x PAD_vector -> new landscape
    omma_blend(tiles, global[PAD_OFFSET], budget);
    
    // Write back
    for (int i = 0; i < 15; i++)
        local[slot_id * 15 + i] = tiles[i];
}
```

### Pre-compiled Consumer Cubins

```
consumers/
  cognitive_loop.cubin        # cognitive loop (landscape, neuromod)
  asr_decoder.cubin            # audio->text (VAD-gated)
  tts_encoder.cubin            # text->audio (prosody-modulated)
  spec_draft.cubin             # harvest-driven speculative decoding
  integrity_scout.cubin        # tile integrity verification
  comfyui_aux.cubin            # background tile processing
  kv_defrag.cubin              # background KV cache defragmentation
  kairos_heartbeat.cubin       # 15s KAIROS tick (cross-SM merge + commit)
  widget_renderer.cubin        # widget surface renderer
```

---

## 5. Voice Consumer Types — Dual-Path ASR/TTS

ASR and TTS models are calibrated as `.den` NVFP4 files using the same NULLGLASS tile format and OMMA.SF.16864 instruction. They integrate as consumers in the compute market with model hot-swap governed by voice activity detection.

### Consumer Type Assignments

```
consumer_id = 3 -> ASR Consumer   (audio samples -> text tokens)
consumer_id = 4 -> TTS Consumer   (text tokens -> audio samples)
```

### Model Tier Policy (VRAM-aware)

| Model | VRAM | Condition |
|-------|------|-----------|
| ASR-0.6B | 984 MB | Default (VAD triggered) |
| ASR-1.7B | 2.2 GB | Low-confidence escalation |
| TTS-0.6B | 864 MB | Default voice output |
| TTS-1.7B | 3.3 GB | High-quality mode |

### Dual-Path Concurrent Pipeline

```
+-------------------------------------------------------------+
|  PATH A — Listen (ASR)         PATH B — Speak (TTS)         |
|                                                             |
|  Mic -> DMA audio buffer        LLM generates tokens         |
|    -> VAD consumer gate           -> TTS consumer (harvested) |
|    -> ASR consumer (harvested)    -> Audio samples -> DMA out|
|    -> text tokens -> LLM ctx       -> VoiceMorph: PAD->prosody|
|                                                             |
|  --------- CONCURRENT ------------------------------------  |
|  Both paths active simultaneously on different harvested     |
|  SM cycles. Full duplex.                                    |
+-------------------------------------------------------------+
```

### Voice Activity Detection (VAD Gate)

```cuda
__device__ bool vad_gate(const float* audio_frame, int frame_samples) {
    float energy = 0.0f;
    int zero_crossings = 0;
    for (int i = 0; i < frame_samples; i++) {
        energy += audio_frame[i] * audio_frame[i];
        if (i > 0 && (audio_frame[i] * audio_frame[i-1] < 0))
            zero_crossings++;
    }
    energy /= frame_samples;
    return energy > VAD_ENERGY_THRESHOLD &&
           zero_crossings > VAD_ZCR_MIN &&
           zero_crossings < VAD_ZCR_MAX;
}
```

### TTS VoiceMorph: PAD to Prosody

```cuda
__device__ void tts_apply_prosody(
    float* audio_sample,
    float pleasure,    // -> pitch shift (+/-30%)
    float arousal,     // -> speed (0.8-1.2x)
    float dominance    // -> timbre (+/-20%)
) {
    // Pitch shift: resample with PAD-driven ratio
    // Speed: adjust playback stride
    // Timbre: low-pass filter cutoff from dominance
}
```

---

## 6. Desktop OCR Consumer — Hardware Screen Capture + Real-Time OCR

### Hardware Engine Abuse

| Engine | Designed For | Abused For |
|--------|-------------|------------|
| NVENC | H.264/H.265 video encoding | Framebuffer capture + motion vector extraction |
| NVOF | Optical flow | Pixel-level change detection map |
| OMMA | FP4 tensor matrix multiply | Vision model OCR inference |

### Capture Pipeline

```
Desktop Framebuffer
  -> NVENC capture (hardware, ~50 us)
  -> NVENC bitstream -> extract motion vectors (free with encode)
  -> NVOF pixel-level diff on changed macroblocks
  -> Changed-region bounding boxes
  -> OMMA OCR consumer: only process changed pixels
```

**First frame:** Full desktop OCR (~30ms, 2560x1440 downscaled to 1024x1024 -> vision model)
**Subsequent frames:** NVOF-guided differential, only changed text regions re-OCR'd (~2-4ms)
**60 FPS at ~5% GPU utilization** — the hardware engines run asynchronously.

---

## 7. Prefetch-Execute Tile Pipeline

### Double-Buffered Tile Loading

Two `__shared__` tile buffers per warp (2 x 160 bytes = 320 bytes of SMEM):

```cuda
__shared__ __align__(16) uint8_t tile_buf[2][160];

int ping = 0;
for (int t = 0; t < num_tiles; t++) {
    // Ping-pong: load N+1 while processing N
    if (t + 1 < num_tiles) {
        cp.async_commit_group(tile_buf[!ping], weight_ptr[t + 1], 160);
    }
    // OMMA on current tile
    omma_fma(C_frag, tile_buf[ping], K_operand);
    if (t > 0) cp.async_wait_group(0);
    ping = !ping;
}
```

### SMEM budget

- 320 bytes for tile buffers (0.3% of 99 KB)
- Consumer state: ~1,000 bytes per SM
- Total SMEM overhead: <1.5 KB (safe within 99 KB budget)

---

## 8. Dead-CTA Harvesting

### SM Utilization Profile

Standard inference:
- Dense layers: ~70-80% SM utilization
- Attention layers: ~40-60% (KV-dependent)
- MoE sparse experts: ~30-50%
- **Weight gaps:** ~100% idle for 50-200 cycles

Harvestable cycles across a forward pass: **30-45% of all SM cycles.**

### Hierarchical Merge

At every global sync point:
1. Each SM local-reduces state via warp shuffle (7 instructions)
2. Merge agent SM collects all 70 local states via device-wide reduce (~6 us)
3. Result broadcast back to all SMs

**Piggybacks on existing sync** — no dedicated sync operations.

---

## 9. Scale-Gated Sparse Attention

### Tile Header as Oracle

NULLGLASS tile format (160 bytes):
```
Bytes 0-143:    FP4 weight data
Byte  144:      sfa (UE4M3 scale factor A)  <- attention gate
Byte  145:      sfb (UE4M3 scale factor B)
Bytes 146-159:  metadata
```

`sfa` is a **free attention saliency signal** — encoded during calibration, costs nothing to read.

### Gate Logic

```
For each KV tile in attention window:
    if tile.sfa < THRESHOLD:
        skip tile  // zero OMMA, zero VRAM, zero compute
    else:
        load tile -> B-fragment -> OMMA(Q, K)
```

### Expected Sparsity

- Bottom 30% tiles: sfa < 0.15 -> near-zero information -> safe to skip
- Middle 50% tiles: sfa 0.15-0.6 -> partial information -> threshold-dependent
- Top 20% tiles: sfa > 0.6 -> high information -> always compute

**30-70% attention reduction without quality loss.**

---

## 10. Silent KV Cache — Cross-Layer Register Forwarding

### Core Mechanism

Every OMMA.SF.16864 produces an 8x8xf32 accumulator (C_frag, 256 bytes) that lives in registers. Silent KV Cache **leaves it in registers** for the next layer.

```
Layer N attention: Q x K -> C_frag_N (registers)
  -- Layer boundary --
Layer N+1 attention:
  if (sfa_N_current == sfa_N_prev):
      reuse C_frag_N as-is -> zero OMMA for this tile
  else:
      load tile, OMMA, update C_frag_N
```

### Register Budget

| Component | Registers per SM |
|-----------|-----------------|
| C_frag tiles (12 KB / 256 B) | 48 |
| SM-local state | 30 |
| Tile buffers (2 x 160 B) | 10 |
| Consumer dispatch | 4 |
| **Total** | **92** out of 232 (40% of register budget) |

### KV Compaction at Generation Boundary

At the end of each autoregressive step:
1. All 70 SMs hold KV tiles as C_frags in registers
2. Only the newly generated token's tile is committed to VRAM
3. Remaining C_frags persist across token boundaries

**For token N+1:** Tiles 0..N-1 unchanged -> zero OMMA. Tile N: single OMMA.
**Effective KV cost: O(1) per token, not O(N).**

---

## 11. Cognitive Tensor Delta Rendering

### Principle

The cognitive landscape (256x256 f32 x 24 layers = 1.5 MB) is recomposited every KAIROS tick (15s). Most ticks change <1% of cells.

### Sparse Update

Track dirty tiles at SM granularity:
```
uint16_t dirty_mask[70];  // 1 bit per tile per SM = 140 bytes total

tick():
    for each tile in local_tiles:
        new = blend(tile, pad_vector)
        if new != tile: dirty_mask[sm_id] |= (1 << tile_idx)
        tile = new
    if dirty_mask[sm_id]:
        omma_copy(dirty_tiles, local, global)
        dirty_mask[sm_id] = 0
```

**Expected reduction:** Average 98% reduction in landscape compute.

---

## 12. Tile Header as VLIW — Extended Instruction Set

The NULLGLASS header's 16-byte metadata section (bytes 144-159) carries not just scale factors — it encodes an **execution policy** for how the SM processes this tile.

### Execution Flag Definitions

```
Bit 0: REMEMBER    — tile pinned. Survives eviction.
Bit 1: FORGET      — tile evicted immediately after read.
Bit 2: ROUTE       — after OMMA, forward tile content to a consumer.
Bit 3: MERGE       — tile output spatially merged with neighbor on writeback.
Bit 4: LOCK        — tile pinned in L2/registers. Cannot be evicted.
Bit 5: SPECULATE   — tile is a speculative decoding target.
Bit 6: PROPAGATE   — writing to this tile triggers ripple update on dependents.
Bit 7: OPAQUE      — tile is NOT an OMMA operand. The 128 data bytes encode a
                     consumer micro-kernel. SM reads into registers and executes.
```

### Bit 7 — OPAQUE: The Tile as Instruction

When the OPAQUE flag is set, the tile's 128 bytes of FP4 weight data are reinterpreted as executable code:

```
0x0000  OMMA              — tile is data. Standard processing.
0x0001  WARP_SHUFFLE_REDUCE — cross-warp state merge.
0x0002  MEMCPY_TILE       — tile relocation within L2/VRAM.
0x0003  BARRIER_SYNC      — SM-level sync point.
0x0004  CONSUMER_DISPATCH — launch consumer micro-kernel from tile payload.
0x0005  LANDSCAPE_BLEND   — blend tile into cognitive landscape.
0x0006  C_FRAG_EXTRACT    — dump C_frag to mapped memory.
0x0007  KV_EVICT          — force eviction of target tile.
0x0008  SPECULATE_VERIFY  — verify speculative token via C_frag comparison.
0x0009  CASCADE_PROPAGATE — propagate tile value change to dependent tiles.
```

### Tile Trifecta Protocol

Three tile types processed as one fused tile load:

```
[Slot A] Weight tile        — loaded into B-fragment (standard)
[Slot B] KV tile            — loaded into second B-fragment slot
[Slot C] Landscape tile     — loaded as bias, accumulated into existing C_frag

Processing:
  1. cp.async loads all three tiles in parallel (3 x 160 bytes)
  2. OMMA.SF.16864 folds A and B into C_frag
  3. Landscape tile C is elementwise-added to C_frag (free, in registers)
```

### Four-Tile Fusion

Extend Trifecta to four tiles:

```
[Slot A] Weight tile          — B-fragment 0 (standard weight GEMM)
[Slot B] KV tile              — B-fragment 1 (attention QK^T)
[Slot C] Landscape bias tile  — C_frag bias (cognitive state injection)
[Slot D] Consumer instruction — executed as opcode after OMMA commit
```

### N-Tile Fusion

Once the three/four-tile load works, the pattern generalizes to N tiles. For a 4 KB SMEM allocation -> 25 tiles -> 25 OMMA operations between writebacks. This is a **fully fused layer cascade** — weights, KV, landscape, and post-ops all processed in a single SMEM-resident loop before touching VRAM.

---

## 13. Landscape-Attention Fusion

Every attention tile (QK^T pair) has overlapping spatial coverage with a cognitive landscape tile. Same format (NVFP4, 160 bytes), same size (16x16 cells), same memory space (L2/VRAM tile pool).

**Landscape tiles are loaded as attention bias in the same OMMA stream:**

```
For each KV tile in attention window K[n]:
  1. cp.async: load weight_tile + kv_tile + landscape_tile[n % N_landscape]
  2. OMMA: Q x weight_tile -> C_frag_0
  3. OMMA: C_frag_0 x kv_tile -> C_frag_1
  4. Add landscape_tile values to C_frag_1 (elementwise, in registers)
  5. Softmax on biased C_frag_1 -> attention weights
```

### Physical Layout

Landscape tiles interleave with KV tiles in the same L2 region. The 1.5 MB landscape (~9,600 tiles at 160 bytes) occupies ~3% of the KV cache's L2 budget.

### Zero-Cost Projection

Because landscape tiles are the same format as KV tiles, loading them follows the exact same cp.async path. No extra memory controller cycles.

---

## 14. Harvest-Driven Speculative Decoding

### Principle

Speculative decoding requires a draft model that predicts tokens faster than the target model. The draft runs as a consumer micro-kernel on harvested cycles. Its speculation depth auto-scales with SM availability.

### Draft as Consumer

```
consumer_id = SPEC_DRAFT
tick_budget = harvested_cycles_available

spec_draft_tick(slot, budget, local, global):
    if budget < MIN_DRAFT_COST: return

    depth = clamp(budget / COST_PER_SPEC_TOKEN, 1, MAX_DEPTH)
    for t in 0..depth:
        omma_speculate(draft_model[t], context)
        global.speculative_tokens[t] = draft_output
    global.speculative_depth = depth
```

**Harvest level determines depth naturally:**
```
SM harvest    ->  speculation depth
< 10%         ->  0 tokens (no speculation, pure inference)
10-20%        ->  1 token
20-40%        ->  3 tokens
40%+          ->  5+ tokens (deep speculation tree)
```

### Compound with Prefetch-Execute

The draft model's tiles are prefetched into SMEM during inference gaps. When harvest cycles become available, the data is already in the register file. **Speculation latency = 0 cycles from the draft's perspective.**

---

## 15. GPU Display — Replacing Tauri/WebView

### Current Cost

- Tauri 2.x process: ~200 MB resident
- WebView (WebKit2GTK): ~150 MB
- JS runtime + bridge: ~50 MB
- Total: **~400 MB** overhead

### Vulkan Native Display (Recommended)

Replace the entire Tauri/WebView stack with a direct GPU display path. CUDA allocates surfaces, exported to Vulkan via `cudaGraphicsVulkanRegisterImage`. Vulkan display process reads the shared buffer as a VkImage. Zero host-side copies. Zero CPU involvement. Zero JS.

**Cost:** 0 MB dedicated display overhead. The Vulkan display process shares the GPU buffer; its only CPU cost is ~1 MB for the swapchain + window.

---

## 16. Multi-Tier Memory Architecture — HDC + Gaussian Splats + Hyperbolic + TDA

A four-tier GPU-native memory system synthesized from H-Mem (hybrid tree+graph), CraniMem (gated consolidation), and non-Euclidean geometric primitives.

### Design Overview

```
+------------------------------------------------------------------+
|  Tier 1 — Holographic STM          Tier 2 — Episodic Splats      |
|  (HDC: 10K-bit hypervectors)       (3D Gaussian splat diffusion) |
|  +----------------------+          +--------------------------+  |
|  | XOR binding          |          | Position = scene subject |  |
|  | Popcount bundling    |          | Opacity = vividness(alpha)|  |
|  | CraniMem goal gate   |--decode--| Covar Sigma = specificity |  |
|  | ~128 bytes/tick      |  encode  | Utility = DA+5HT score   |  |
|  +----------+-----------+          +--------------+-----------+  |
|             |                      |              |              |
|  Tier 3 — Hyperbolic LTM             Tier 4 — TDA Checker       |
|  (Exponential volume, L2 resident)   (Persistent homology)      |
|  +----------------------+          +--------------------------+  |
|  | L0-L3 in 48MB L2     |          | Betti-0 = connected       |  |
|  | Deeper on demand     |          | Betti-1 = logical holes   |  |
|  | Texture unit dist    |          | Warp reduce per check     |  |
|  | ~4 KB/node           |          | ~2 us per verification    |  |
|  +----------------------+          +--------------------------+  |
+------------------------------------------------------------------+
```

### Tier 1 — Holographic STM (CraniMem Goal-Gated)

Every concept, entity, emotion, and action is mapped to a 10,000-bit pseudo-random hypervector. The scene is bound (XOR) and bundled (bitwise majority) into a single composite vector.

### Tier 2 — Gaussian Episodic Memory

Each interaction is stored as a 3D semantic Gaussian splat. Parameters encode position (core subject), opacity (vividness, decays over time), covariance (specificity, diffuses outward), and utility score (from neuromodulator state at encoding).

### Tier 3 — Hyperbolic Long-Term Memory

High-utility episodic splats are consolidated into a hyperbolic tree where volume grows exponentially with radius. A 4-deep tree at curvature K=-1 has the same capacity as a flat 2^16-deep tree.

### Tier 4 — Topological Hallucination Checker

Before generating a response, the planned output's logical dependency graph is compared against the prompt's dependency graph using persistent homology. If Betti numbers don't match, a hallucination is detected.

---

## 17. Pluggability Matrix

| Optimization | Gains Sans Consumer | Gains With Consumer |
|--------------|-------------------|-------------------|
| Prefetch-Execute Tile Pipeline | ~5% throughput | ~5% + earlier harvest window |
| Dead-CTA Harvesting | 0% (no consumer) | 30-45% cognitive cycles |
| Scale-Gated Sparse Attention | 30-70% attention reduction | Same |
| Silent KV Cache | O(N) -> O(1) per token | Same + state shares register budget |
| Cognitive Delta Rendering | 0% | 98% landscape compute reduction |
| Tile Header VLIW | 0% | Tiles as executable instructions |
| Landscape-Attention Fusion | 0% | Cognition injected at tile level |
| Harvest Spec Decoding | 0% | Auto-scaling speculation |

**Consumer-less scenarios are not penalized.** Every optimization improves the no-consumer path.
