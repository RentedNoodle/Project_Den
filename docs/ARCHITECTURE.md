# Project Den — Architecture

> [!IMPORTANT]
> **Project Den is under active development and has not been released.** Public availability is limited to the engine source/build work; the integrated companion/application is not a public release. This document is design context, not a release statement. Current status: [README](../README.md).

**Date:** 2026-10-10 | **Status:** In development — private prototype on Qwen3.8-27B v2.1; not released. Blackwell NVFP4/OMMA dispatch is in progress.

> Ground truth for what is built vs designed is the README's current Status section. This document is the engineering view of the pipeline and hardware; where the two disagree, the README wins.

Project Den is a GPU-resident platform for sovereign AI companions. This document describes how it actually fits together — the pipeline, the container, the model stack, and the hardware it runs on. It complements the README's product narrative with the engineering view.

## The Three-Tier Pipeline

Every model that enters Project Den flows through one container format:

```
HF / GGUF (import)
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│  Converter   │ ──► │  .den format │ ──► │  Runtime (Rust + CUDA)   │
│  (Python)    │     │  (container) │     │  ├─ inference engine      │
└─────────────┘     └──────────────┘     │  └─ cognitive companion   │
      ▲                                        │  └─ avatar / world    │
   quantization/calibration                    │     subsystems        │
   format packing                              ▼                      │
                                     OMMA.SF.16864 tensor core        │
                                     (sm_120a consumer Blackwell)     │
                                     └────────────────────────────────┘
```

### Tier 1 — Converter (Python)

Turns external model formats into `.den`. Not a copy tool — a **semantic-preserving transformer**:

- **Import:** HuggingFace safetensors (primary), GGUF (compat), NVFP4 modelopt safetensors (`build_nvfp4_den.py`).
- **Quantization:** AWQ + NVFP4 calibration pipeline (`den_calibrate.py`), WH4 WHT-domain NVFP4 (`quantize_wh4.py`), BF16 passthrough.
- **Refinement:** Abliteration (`den_abliterate_den.py`), expert reduction by importance ranking (`truncate_den_experts.py`).
- **Output:** a single `.den` container — weights, correction metadata, per-tile format tags, execution policies.

### Tier 2 — `.den` Container Format

`.den` is not a model file; it is a **runtime object**. Each layer is decomposed into per-tile sub-slots (32 per layer) carrying their own precision and dispatch policy:

| Precision | Used for | Rule |
|-----------|----------|------|
| F32 | Norms, SSM params, RoPE | Never quantized. SHA256-identical to source. |
| BF16 | Embeddings, lm_head, router gates | Verbatim passthrough. |
| NVFP4 | Weight matrices | Calibration-aware quantization. |
| MXFP6 | K/Q/V projections | Higher precision where FP4 codepoints can't cover the range. |

**NULLGLASS tile (160 bytes)** — the locked low-level container: FP4 weight data, `sfa`/`sfb` scale factors (E2M1 + UE4M3/UE8M0), Hadamard signs, phase tag, ESAB bias cascade, UV correction pointer, execution policy flags. Per-tile format dispatch via `tile[148]` scale-format bits.

### Tier 3 — Runtime (Rust + CUDA)

**Rust (cognition):** 62+ modules — semantic memory, forgetting curves, trust dynamics, Global Workspace Theory, emotional homeostasis (PAD model), PrivateArchive, narrative self, life log. CPU-only, zero VRAM.

**CUDA (engine):** the inference core on sm_120a consumer Blackwell (RTX 5070 Ti, GB203):

- **Four workload kernels**, dispatched by inference stage: `KT_PREFILL` (chunked GEMM + TMA prefetch), `KT_DECODE` (warp-GEMV, register-resident accumulator), `KT_MOE` (grouped GEMM, persistent grid), `KT_MTP` (multi-token-prediction draft head).
- **Five precision paths**, escalating per tile: NVFP4 OMMA (`mxf4nvf4`, ~29 cycles/MMA) → MXFP4 QMMA → DP4A MMQ → CPU AVX-512 → NULLGLASS direct dispatch.
- **Memory hierarchy exploited:** 99 KB SMEM/block, 48 MB L2 (KV cache, expert pinning), 16 GB VRAM budget.

## The Three-Model Cognitive Stack

Cognition is not one model. It is a division of labor across three tiers, each sized to its role:

| Model | Role | Location | Status |
|-------|------|----------|--------|
| **Cortex** (Qwen3.8-27B v2.1, dense) | Authority — generation, verification, memory operations | VRAM | Private development |
| **MTP head** (in-model) | Multi-token prediction — speculative decode | VRAM | Public engine |
| **Claustrum** (0.8B) | Orchestrator — attention monitoring, GWT ignition, memory triggers | CPU | Designed |

The Claustrum is designed to run entirely in CPU L3 cache, never touching VRAM, and to drive the global-workspace ignition signal at 10 Hz — a cognitive layer with a zero-VRAM budget, running on silicon the GPU never needs.

## Compute Model

| Kernel | Workload | Mechanism |
|--------|----------|-----------|
| KT_PREFILL | Prompt processing | Chunked GEMM, TMA K/V prefetch |
| KT_DECODE | Token generation | Warp-GEMV, register-resident accumulator |
| KT_MOE | Expert dispatch | Grouped GEMM, 70-CTA persistent grid |
| KT_MTP | Multi-token prediction | Draft head, speculative decoding |

## Why the Three Tiers

1. **Converter tier owns fidelity** — quantization/calibration correctness is verified numerically (cos > 0.999 vs HF), not trusted.
2. **`.den` tier owns dispatch** — the format carries *how* to run each tile, so the runtime doesn't guess.
3. **Runtime tier owns performance** — SASS-first: every inner loop audited, 0 MOV, tensor-core OMMA where available, CPU spillway when not.

The endgame: a 35B MoE with NVFP4 + expert offloading fits a ~4.7 GB active working set on 16 GB VRAM — the combination that makes the Cortex tier possible on consumer hardware.
