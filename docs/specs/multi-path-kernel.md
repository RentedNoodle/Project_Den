# Multi-Path NVFP4 Kernel Architecture — Design Spec

**Date:** 2026-05-22 | **Status:** Approved | **Phase:** Design
**Target Hardware:** SM120 (RTX 5070 Ti, 70 SMs, 16 GB, 99 KB SMEM)
**Models:** 4B Dense, 35B MoE, 0.8B/1.7B media

---

## 1. Overview

A unified multi-path dispatch kernel for OMMA-priority NVFP4 inference across all model types (MoE, Dense, Media). Single dispatch function routes to the optimal path per-tile based on NULLGLASS execution policy flags, with per-workload-class tile configurations.

## 2. Multi-Path Dispatch

```
inference_request(model_type, layer_type, tile_header)
  |
  +- Path 0: OMMA_NVFP4  (primary)
  |   +- mxf4nvf4 / scale_vec::4X / UE4M3 / m16n8k64
  |   +- For: 2D weight matrices, SM120 native
  |
  +- Path 1: QMMA_MXFP4  (fallback for FP8-kept tensors)
  |   +- mxf8f6f4 / scale_vec::1X / UE8M0 / m16n8k32
  |   +- For: linear_attn.* projections kept in FP8
  |
  +- Path 2: DP4A_MMQ  (quaternary, no FP4 hardware)
  |   +- INT4 MMQ via dp4a
  |   +- For: future-proofing non-SM120 GPUs
  |
  +- Path 3: CPU_VNNI  (emergency / TDR recovery)
      +- AVX-512 VNNI on CPU
```

Path selection via NULLGLASS tile header bytes 158-159 (execution policy flags). Zero-cost branch — the policy bits are decoded at tile pack time and burned into the .den file.

## 3. Per-Workload Tile Configurations

| Property | MoE (35B) | Dense (4B/9B) | Media (Flux/Wan) |
|----------|-----------|---------------|-------------------|
| OMMA tile | m16n8k64 (K=64 fix) | m16n8k64 (std) | m128n128k64 (tile GEMM) |
| SMEM budget | 99 KB (tight) | 99 KB (comfortable) | 99 KB (large tiles) |
| CTA pattern | 70 CTA ZL-1 persistent | 70 CTA parallel grid | CUDA Graph compiled |
| Expert handling | elastic, hot in L2, cold paged | N/A | N/A |
| L2 partition | expert-aware round-robin | single-stream contiguous | frame-buffer pinned |

## 4. Stolen Techniques by Component

### 4.1 FlashAttention (from sm120-kernels)

Target: 251 TFLOPS on SM120 (beats cuDNN).

- `ldmatrix.x4` for Q, `ldmatrix.x2` for K, `ldmatrix.x2.trans` for V
- Register-resident P (saves 16 KB SMEM per warp)
- `__constant__` TMA descriptors (4.4x speedup at Sq <= 512)
- XOR swizzle for conflict-free shared memory (99->117 TFLOPS jump)
- BN=128 single-stage with mbarrier prefetch (251 registers, 0 spills)
- Auto-dispatch: BN=64 double-staged for Skv <= 2048, BN=128 single-stage for longer

### 4.2 DeltaNet/SSM (from qwen35-thor + qwen3.5-cuda-kernel)

- `gdn_umma_sm110.cu` — fused Gated Delta Network kernel (decay->read->delta->update->output in one launch)
- `deltanet_chunkwise.cu` — chunked prefill for seq_len > 1 (WY parallel scan)
- Inter-layer chaining: residual-add fused with next layer's input RMSNorm (eliminates 63 HBM round-trips/token for 27B, ~40 for 35B)
- Fused post-projection + recurrent

### 4.3 NVFP4 Quantization (from nvfp4.cu + petit-kernel)

- 128-token blocking for quantization throughput
- SA3 64x16 scale swizzle for better L2 cache behavior during attention
- Vectorized float2 loads (8 pairs = 16 elements per thread)
- 8->4 byte PTX packing (two FP4 values per byte)
- Petit-kernel offline weight shuffling: Marlin-style pre-arranged weight layout eliminates GPU-side bit-twiddling during OMMA access

### 4.4 ISA Truth (from sass-king)

- Kernel 16: OMMA/QMMA block-scaled FP4 — exact SASS patterns for fragment layout, register allocation, accumulator chaining
- Kernel 23: FP4 fragment layout probes — interleaved row/row+8 A-register layout
- Kernel 18: Pipelined MMA tile with async copy staging — TMA + mbarrier pipeline patterns
- `ldmatrix` is the single biggest optimization (62% speedup confirmed)

### 4.5 MoE-Specific (from sm120-kernels CUTLASS fix)

- `EffBlk_SF = min(K/SFVectorSize, Blk_SF)` — unblocks K=64 tiles on 99 KB SMEM
- Without this fix, MoE expert layers fall back to slow generic kernels
- Upstream: FlashInfer PR #2786

## 5. Quantization Firewall

| Layer Group | Quantization | Reason |
|-------------|-------------|--------|
| MoE experts (gate/up/down) | NVFP4 OMMA | Dominates size, safe to quantize |
| Self-attention (q/k/v/o) | NVFP4 OMMA | Small, low sensitivity |
| `linear_attn.*` (all SSM proj) | BF16 / FP8 | Degrades accuracy |
| `embed_tokens` | BF16 | Too sensitive |
| `lm_head` | BF16 | Too sensitive |
| `mlp.gate` / `shared_expert_gate` | BF16 | Router precision critical |
| All norms | BF16 / F32 | Precision required |
| `ssm_conv1d.weight` | F32 | Small enough |
| `ssm_a`, `ssm_dt.bias` | F32 | Tiny vectors |

## 6. Baseline Targets

| Model | Format | Size | Tok/s Target | Quality Gate |
|-------|--------|------|-------------|-------------|
| 4B Dense | NVFP4 .den | ~2.6 GB | 200 tok/s | <= 1% regression vs BF16 |
| 35B MoE | NVFP4 .den | ~23 GB | 50 tok/s | <= 2% regression vs BF16 |
| 2B Dense | NVFP4 .den | ~1.5 GB | 250 tok/s | <= 1% regression |

Hardware: single RTX 5070 Ti, 16 GB VRAM, 896 GB/s bandwidth.

## 7. Speculative Decoding Catalog

### 7.1 CATS (Context-Aware Token Selection)

Uses SSM state from the last N DeltaNet layers as a free draft model.

| Property | Value |
|----------|-------|
| Draft model | None — reuses existing SSM state |
| Draft quality | ~70-80% acceptance |
| Memory cost | ~78 MB (BF16) for projection matrix |

Advantages: Zero extra model — the SSM already computes this per token.

### 7.2 MTP (Multi-Token Prediction) — Qwen3.5/3.6 Native

| Property | Value |
|----------|-------|
| MTP layers | 1 |
| Speedup (Dense) | 1.4-2.0x |
| Speedup (MoE) | 1.15-1.25x |
| Acceptance rate | 83% at spec=2, 56.7% at spec=3 |

### 7.3 DFlash — External Draft Model

| Property | Value |
|----------|-------|
| Draft model | Lightweight cross-attention model |
| Acceptance rate | 62-78% position-0, 2.7-4.4 mean accepted tokens |

### 7.4 Built-In Speculative Mechanisms

| Mechanism | Description |
|-----------|------------|
| SSM Draft | Free draft from SSM state, ~70-80% acceptance, zero model overhead |
| Speculative Attention | 4 attention variants tested simultaneously via warp divergence |
| Path Integral Sample | 8 speculative decode trees (paths), phase=log_probability weighted |
| Intentional Divergence | Next-tile speculative preload — one warp preloads while sibling warps continue OMMA |
| RT Tile Predictor | RT core-guided speculative tile prefetching |
| Reservoir Draft | Reservoir sampling-based draft selection |
| Device Decode Loop | Tree depth 8, fan-out 8 |

### 7.5 Unified Speculative Strategy

1. **Free tier** (always on): SSM Draft — zero cost, 70-80% quality
2. **Standard tier**: MTP=3 for Qwen3.5/3.6 native models
3. **Premium tier**: External draft model when VRAM budget allows
4. **Device-side acceleration**: Device decode loop + RT core tile prefetching + path integral sampling

Target: **Combined spec decoding throughput multiplier of 1.5-2.5x** over baseline decode speed.

## 8. Tile Format Evolution — .den V4

| Property | Current NULLGLASS V3 | Proposed V4 |
|----------|---------------------|-------------|
| Tile size | 160 bytes (128B nibbles + 16B scales + 16B header) | TBD (3-lane compact) |
| MMA time padding | Pre-padded | In-register zero-padding |
| VRAM efficiency | Standard | +20% (no pad zeros stored) |
| Register reuse | Per-op | 4 MLA ops per register load |

## 9. SM120 Fragment Layout (Verified)

From sass-king Kernel 23 + kekzl/imp #350 + ac4k_kernel:

- **D-fragment is inverted on SM120**: `frag[0,1]` -> upper row-pair (row+8), `frag[2,3]` -> lower row. NOT the standard row/row+8 assignment used on SM80-90.
- **A-fragment**: 4 uint32 registers, interleaved row/row+8
- **B-fragment**: 2 uint32 registers, `ldmatrix.trans` for column-major K
- **Scale distribution**: Only lanes 0 and 1 carry scale values; lanes 2-3 are ignored
- **CUTLASS #3185 canonical**: SFA/SFB as MMA instruction fields (not data operands), 4 physical scale registers per side via zero-stride layout compaction

## 10. FP4 Flash Attention Variants

| Implementation | Precision | Target | Peak TFLOPS | Key Technique |
|---------------|-----------|--------|-------------|---------------|
| fp4-fused-attention-sm120 | FP4 E2M1 (mxf8f6f4) | RTX 5070 Ti | ~3.4 | Inline PTX, on-the-fly quant |
| BlackFlash | BF16 (HMMA) | RTX 5060 | ~31 | Warp spec, TMA, xor swizzle |
| fp4-attention | FP4 (SageAttention) | Blackwell | N/A | Block quantization |
| flash-attention-fp4 | NVFP4+FP8 (CuTe) | B200 (SM100) | 2018 | Pre-quantized QK, mixed PV |

### 10.1 SM120 FP4 Specifics

Critical findings from implementations tested on exact hardware:
- FP4 container: nibble in bits 5-2 (not 3-0). `kind::mxf8f6f4` — each FP4 occupies 8-bit container (vs `mxf4nvf4` which packs 2 per byte for 2x throughput)
- Only `scale_vec::1X` available on SM120 for FP4 (not 2X or 4X)
- On-the-fly quantization: 66% of SASS instructions just for quant. Pre-quantize.
- Register layout: 4 A-fragment + 2 B-fragment + 4 acc = 10 regs/thread. Row: `lane / 4`.
- Replace division with `val * exp2f((float)(127 - scale))`
- Use FP16 staging to fit within 99 KB SMEM

### 10.2 OMMA Advantage

The `mxf4nvf4` path with `scale_vec::4X` and `UE4M3` — 2 FP4 values per byte, 4 scales per uint32 (65,025 effective scales through superposition) — provides roughly 2x the throughput ceiling of mxf8f6f4 kernels before any optimization.

### 10.3 Mixed Precision Strategy

Pre-quantized QK in FP4 with block scaling, PV path in BF16. Adaptation: pre-quantized FP4 KV cache, quantize Q in separate lightweight kernel, PV in BF16 via OMMA accelerator.

## 11. Implementation Order

1. **4B OMMA attention** — FlashAttention path, prove 200 tok/s.
2. **4B fused SSM** — DeltaNet path.
3. **Dispatch architecture** — Wire Path 0/1/2/3 with per-workload tile configs.
4. **4B full integration** — End-to-end benchmark, compare BF16 vs NVFP4 quality.
5. **35B MoE** — CUTLASS K=64 fix, expert elastic paging, persistent CTA pattern.
6. **Performance tuning** — Inter-layer chaining, register optimization, L2 partition tuning.

## 12. Key Constraints

- SM120 SMEM: 99 KB hard limit per block — `static_assert(SMEM_BYTES <= 99 * 1024)` in every kernel
- CUDA 13.3 required (12.x/13.2 ptxas rejects `sm_120a`/`mxf4nvf4`)
- Register split: 232/40 via `setmaxnreg`, no `--maxrregcount=128` global
- OMMA 3-operand scale format required: `(uint32 sfa, uint16 bid, uint16 tid_sf)`
- No tcgen05, WGMMA, TMEM, TMA multicast — dead on consumer SM120
