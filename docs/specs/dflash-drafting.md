# DFlash Cross-Attention Drafting — Speculative Decoding Integration

**Date:** 2026-05-25 | **Source:** BeeLlama.cpp, z-lab/dflash, Anbeeld/beellama.cpp
**Application:** Augment CATS 4x speculation with cross-attention drafting.

---

## What DFlash Adds

CATS uses an autoregressive draft model (0.8B) to propose tokens. DFlash uses a **cross-attention draft model** — the drafter cross-attends to recent target model hidden states stored in a ring buffer. This gives richer context than CATS' autoregressive approach, especially for structured output.

## BeeLlama Results

- **4.56x speedup** on structured output (code/JSON), 50-68% acceptance rate
- **Three speculation modes**: DFlash (cross-attention), MTP (multi-token prediction), CopySpec (literal copy)
- **Ring buffer**: last N target hidden states, cross-attended by draft model
- Draft model is tiny — 4 layers, hidden=256, cross-attention only

## Integration with CATS

Current CATS plan: 0.8B autoregressive draft -> 4x speculation.

DFlash augmentation:
```
Target model forward pass
  -> hidden states written to ring buffer (last 16 tokens)
  -> DFlash drafter cross-attends to ring buffer
  -> proposes 4 tokens in parallel
  -> CATS verify pass: accept/reject
  -> mixed speculation mode:
      - code/JSON: DFlash (higher acceptance on structured output)
      - prose/conversation: CATS autoregressive (better for creative text)
      - boilerplate/repetition: CopySpec (literal copy mode)
```

## Ring Buffer Design

Size: 16 tokens x H (2048 for 2B, 2560 for 4B) x sizeof(float) = up to 164 KB.
GPU-resident, zero host involvement. Stored in L2 cache (48 MB available, easily fits).
Updated each target model forward pass — oldest entry evicted, new entry appended.

## Draft Model Architecture

4 transformer layers, hidden=256, FFN=1024, 4 heads, no RoPE (uses relative position bias).
Cross-attention ONLY — no self-attention. Input is learned query embeddings (4 positions).
Output: 4 token predictions in parallel via 4 independent LM heads.

## Implementation Phases

1. **Phase 1**: Add ring buffer to pipeline context, store per-token hidden states
2. **Phase 2**: Implement cross-attention draft kernel (CUDA, 4 layers)
3. **Phase 3**: Integrate with CATS verify path, add mode selection heuristic
4. **Phase 4**: Benchmark vs autoregressive draft, tune acceptance rates

## Novelty

- First cross-attention drafting on consumer Blackwell
- Mixed speculation modes (DFlash/CATS/CopySpec) selected by content classifier
- Ring buffer in L2 cache — zero VRAM pressure, single-digit microsecond latency
- Draft model runs on harvested OMMA cycles
