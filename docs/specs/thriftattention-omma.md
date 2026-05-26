# ThriftAttention + OMMA.SF.16864 — Selective Mixed-Precision Attention

**Date:** 2026-05-25 | **Source:** ThriftAttention (arxiv 2605.23081, joesharratt1229/ThriftAttention)
**Novel application:** First selective FP4/BF16 attention on consumer Blackwell via OMMA.SF.16864.

---

## Problem

Full-attention layers (8 for 2B, 10 for 4B) are BF16 firewall — 1.5-2.5 GB VRAM. Naive FP4 attention produces 0.247 mean score vs 0.469 for FP16 on long-context benchmarks (Helmet, Ruler, LongBench). The quality gap is 47%.

## Solution

ThriftAttention: 95% of Q.K^T attention blocks run NVFP4 via OMMA.SF.16864, 5% critical blocks stay BF16. Merged via online softmax. Quality recovery: 89-97% of FP16 baseline.

## Block Importance Heuristic

For each (query_block, key_block) pair where block_size=16:

```
score(q_blk, k_blk) = ||q_blk||_2 * ||k_blk||_2
```

Top-k% blocks (k=5% default) promoted to BF16. Norm product is a cheap proxy for attention dot-product magnitude — high-norm blocks dominate softmax output.

Compute: per-block norms via warp reduction on query/key tensors. O(K_blocks * Q_blocks / 32) warps. Under 0.1% of total attention FLOPs.

## Dispatch Architecture

```
For each attention head:
  1. Compute per-block Q/K norms (warp reduce, 16-element blocks)
  2. Sort block pairs by score, select top-5%
  3. Promoted blocks: BF16 attention (existing CUDA path)
  4. Remaining blocks: NVFP4 OMMA.SF.16864 (m16n8k64, UE4M3 scales)
  5. Online softmax merge: combine BF16 and NVFP4 partial softmax outputs
```

## OMMA Integration

NVFP4 attention blocks use the existing OMMA.SF.16864 kernel path:
- A-fragment: query block, 4 registers x 8 E2M1 nibbles
- B-fragment: key block (transposed), 2 registers x 8 E2M1 nibbles
- Scale: UE4M3 per-block from activation statistics
- Accumulator: F32, online softmax normalization
- Tile: m16n8k64, ~29 cycles per MMA instruction

The OMMA kernel provides the base path. ThriftAttention adds a per-block dispatch layer — no new tensor core code.

## Online Softmax Merge

Standard ThriftAttention merge. Let A_FP16 be BF16 attention scores for promoted blocks, A_FP4 be NVFP4 scores for remaining blocks:

```
m_new = max(m_FP16, m_FP4)
s_new = exp(m_FP16 - m_new) * s_FP16 + exp(m_FP4 - m_new) * s_FP4
output = (exp(m_FP16 - m_new) * s_FP16 * V_FP16 + exp(m_FP4 - m_new) * s_FP4 * V_FP4) / s_new
```

## VRAM Impact

| Component | Current (BF16) | ThriftAttention |
|-----------|---------------|-----------------|
| Q/K/V projections | BF16 passthrough | BF16 passthrough |
| Attention weights (Q.K^T) | BF16 compute | 95% NVFP4, 5% BF16 |
| O-projection | BF16 | BF16 |
| Attention VRAM total | 1.5 GB (4B) | ~0.5 GB |
| Quality vs BF16 | 100% | ~94-97% |

## Implementation

1. **Block importance kernel** (den_thrift_scores.cuh): 32-thread warp per 16-element block. Compute ||q||*||k||, atomicMax to global top-k buffer.
2. **Selective dispatch** in pipeline attention path: replace current shortcut with full attention using ThriftAttention dispatch.
3. **Online softmax merge** (den_thrift_merge.cuh): combine partial outputs.
4. **Converter integration**: attention Q/K weights stay BF16 (projections are small). Only the ATTENTION COMPUTE itself uses mixed precision.

## Novelty

- First selective mixed-precision attention on consumer SM120
- First OMMA.SF.16864 attention blocks (currently only used for FFN/GDN)
- Block importance heuristic adapted for 16-element OMMA tile geometry
- Online softmax merge between CUDA-core BF16 and tensor-core NVFP4 paths
