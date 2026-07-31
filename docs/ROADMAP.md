# Project Den — Roadmap

**Date:** 2026-07-31 | **Honest status:** Engine decode-loop wall broken; GPU OMMA dispatch validating. Public release ~3 months out (see README).

This is the real, current state — not the aspirational one. The README's "What's Built vs What's Designed" table is the ground truth at any moment; this roadmap shows how the Designed columns become Built.

## The Nine Circles

Project Den develops in nine circles. Each gates the next.

| # | Circle | Status (2026-07-31) |
|---|--------|---------------------|
| 1 | **Teacher's Arrival** — 35B NVFP4 .den loaded, coherent output | 🔄 **IN PROGRESS.** The multi-day decode-loop wall (`"the the the"`) is **FIXED** — 35B MoE now produces coherent output matching reference. Root cause: a speculative-decoding default (CATS) corrupting recurrent SSM state on draft rejection. GPU OMMA dispatch still validating end-to-end. |
| 2 | **MoE Router Forged** — AIR-MoE routing, expert offloading, L2 cache | Partial. Routing implemented; expert-offload dispatch is the missing bridge. |
| 3 | **Distillation Engine** — teacher→student, gradient-free training | Not started. |
| 4 | **CATS Speculative** — multi-token parallel spec decode | Reverted to safe default (was corrupting recurrent state). Re-enable requires wiring the SSM-state checkpoint/restore machinery. |
| 5 | **Fall of LocalMaxxing** — beat benchmark targets | Partial. OMMA GEMV 459 tok/s demonstrated. |
| 6 | **Cognitive Bridge** — PAD → dispatch feedback loop | **HERE.** Cognitive stack works (25 daemons, 82/82 tests); not wired to the engine. |
| 7 | **TensorLandscape** — composite kernel, VIC hardware blend | Spec only. |
| 8 | **iDream Manifestation** — prompt→3D, <8s per asset | Pipeline spec complete, model loading stubbed. |
| 9 | **Dreya Wakes Up** — full Governor FSM, dream consolidation | Future. |

## The Wall That Was (and Is)

The project was blocked for months on **Circle 1** — the GPU forward pass producing incorrect output. As of 2026-07-31 that specific wall is **broken**:

- **Root cause found:** CATS self-speculative decoding was enabled by default. It batch-drafts K tokens in one forward pass, which evolves the recurrent SSM state (`kv_self.s_l`) through all draft tokens and cannot roll it back on rejection → state corruption → decode loop.
- **Fix:** default speculative type → NONE, plus a safety gate disabling CATS on recurrent models. 35B MoE now outputs coherent English, byte-identical to reference on greedy decode.

**What remains on Circle 1:** the end-to-end GPU OMMA path — NVFP4 dispatch through real tensor cores rather than software dequant. The tile format is proven in silicon (identity cos=1.0, real-tensor verification pending end-to-end). That is the current engineering focus.

## Near-term (post-Circle-1)

1. **Real OMMA dispatch end-to-end** — wire NVFP4 tiles through `OMMA.SF.16864` (not software dequant). Format proven (cos 0.99996 vs HF); dispatch is the gap.
2. **4B NVFP4 coherence** — a separate NVFP4-specific residual remains on the 4B model; isolated from the GDN/MoE fixes.
3. **Expert offloading (Circle 2)** — the three missing pieces in `forward.c`: selective expert upload, VRAM cache, async prefetch. This is what keeps the Cortex tier's active set at ~4.7 GB on 16 GB VRAM.
4. **CATS re-enable** — wire `llama_spec_ckpt_save/restore` into the CLI speculative path (the server already has it) so recurrent state rolls back correctly; then CATS can return for dense models.

## Definition of Done — Circle 1

- [x] CPU forward pass verified (cos > 0.9999 vs HF)
- [x] 35B MoE decode-loop root cause found and fixed
- [x] NVFP4 tile format proven in silicon (identity cos=1.0)
- [ ] Real tensor-core OMMA dispatch producing coherent output end-to-end
- [ ] Public release

Nothing else in the roadmap moves until the last two boxes are ticked.
