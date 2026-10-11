# Project Den

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Qwen3.8--27B-yellow)](https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored)

### A framework for persistent AI companions.

Den is a system for running an AI companion that remembers, changes, and stays — on your own hardware, not a cloud.

Bring your own model and your own persona. Den supplies what a chatbot lacks: memory that persists between sessions, an identity that develops, relationships that build, and internal state that survives. The companion is a character you define. The substrate that keeps it the same instead of a rerun is Den.

---

## The problem

A chatbot forgets. Close the app, open it again, and it meets you for the first time. It can describe a personality it doesn't have, retrieve memories it never lived, and simulate emotion it never felt.

Den starts from a different question: what does a system actually need to keep a companion *the same* across time — to remember what happened, change from it, and stay someone?

The answer, in engineering terms: persistent memory, identity state, affect, attachment, drives, and a runtime that keeps running between conversations. Den is those pieces, built as one substrate — with the model as a component, not the whole mind.

---

## What Den provides

Den treats cognition as a continuous system, not a prompt wrapped around an LLM:

- Persistent episodic and semantic memory
- An identity and personality that evolve
- Relationship and attachment models
- Affective state (mood and emotion)
- Drives and goal formation
- A global-workspace-style arbiter for attention
- Private internal state
- Constitutional self-governance
- Multiple neural models with distinct roles
- GPU-native local inference
- Long-lived state across sessions
- Multimodal perception and generation

The language model is one component. It is not the whole mind.

---

## Hardware and body

Den runs on a sovereign local stack — your machine, not a cloud service.

The engineering target is a single NVIDIA [Blackwell](https://en.wikipedia.org/wiki/Blackwell_(GPU_architecture)) GPU (RTX 5070 Ti, 16 GB), where memory, compute, model state, and cognition are all scarce resources that must be actively orchestrated.

That constraint drove a custom inference engine:

**[den_llama.cpp](https://github.com/RentedNoodle/den_llama.cpp)** — a Blackwell-native neural runtime (NVFP4 weights via OMMA tensor cores), derived from [llama.cpp](https://github.com/ggerganov/llama.cpp) and becoming the execution substrate for Den.

---

## The cognitive architecture

```
                    ┌──────────────────────┐
                    │      COGNITION       │
                    │  goals · narrative   │
                    │  identity · agency   │
                    │  social reasoning    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   GLOBAL WORKSPACE   │
                    │ attention · salience │
                    │ arbitration · access │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
     │ MEMORY  │          │ AFFECT  │          │ AGENCY  │
     │ episodic│          │ mood    │          │ drives  │
     │ semantic│          │ emotion │          │ goals   │
     │ working │          │ state   │          │ choices │
     └────┬────┘          └────┬────┘          └────┬────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │      NEURAL CORE     │
                    │ Cortex · MTP ·       │
                    │ Claustrum · future   │
                    │ modality models      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     DEN RUNTIME      │
                    │ tensors · state      │
                    │ memory · scheduler   │
                    │ GPU · storage        │
                    └──────────────────────┘
```

Neural models are components of cognition, not synonymous with it.

---

## Three-model cognitive hierarchy

Different jobs, different models. One model doesn't do everything.

| Component | Role | Status |
|-----------|------|--------|
| **Cortex** (Qwen3.8-27B dense) | Primary reasoning, generation, verification, memory operations | [+] IMPLEMENTED — live |
| **MTP head** (in-model) | [Speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding) via multi-token prediction | [+] IMPLEMENTED |
| **Claustrum** (0.8B) | Cognitive orchestration, attention monitoring, workspace arbitration | [ ] DESIGNED |

The Cortex is the live serving model: Qwen3.8-27B (dense), quantized IQ3_XXS with an in-model MTP head. Speculative decoding uses that head directly (`--spec-type draft-mtp --spec-draft-n-max 2`) rather than a separate draft model.

The Claustrum — named after the brain's thin sheet of gray matter that Crick and Koch hypothesized as a "conductor of consciousness" — is designed, not built. It is listed here as architecture.

Small models and non-neural systems are intended to handle continuous background work while the primary model is reserved for expensive reasoning.

---

## Models

| Model | Base | Quant | Status |
|-------|------|-------|--------|
| [Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored](https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored) | `orcarouter/Qwen3.8-27B-Uncensored` | IQ3_XXS trunk (~3.04 bpw) + MTP Q6_K | live / lead |
| [Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored) | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | IQ3_XXS + trained MTP head | prior line |

---

## Memory is part of identity

Den treats memory as more than a retrieval database:

- Episodic memory (what happened)
- Semantic memory (what was learned)
- Salience (what mattered)
- Consolidation (what persists)
- Forgetting (what fades)
- Relationship memory (who matters)
- Emotional persistence (how it felt)
- Hippocampal-style replay during idle
- Progressive compression across tiers

If everything is remembered equally, memory is just storage. A companion needs selective persistence — what survives, what fades, what consolidates, and what changes, should all matter.

Memory compression applies a [Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform) across the time dimension: decorrelating temporally adjacent attention patterns, then quantizing at different rates per frequency band. The DC component (the gist) stays at full precision; high frequencies (exact words) fade. An engineering hypothesis inspired by how human autobiographical memory keeps gist while losing fine detail.

The store is **event-sourced**. Every turn and internal change is appended to a durable log; all state is reconstructed by folding that log (`state = fold(events)`). The log is the authority, so state survives restarts, replays identically, and is not reconstructed from the latest prompt each session.

---

## Agency

Den explores agency as an emergent consequence of persistent internal state, not as a single autonomous-agent loop.

The architecture includes persistent drives around autonomy, competence, relatedness, curiosity, and coherence. These drives shape goals and behavior over time.

The objective is a system capable of developing preferences, priorities, boundaries, and continuity — not an engine that blindly executes tasks.

---

## Identity and private state

A companion needs state that is not rebuilt from the latest prompt each session.

Den includes persistent identity, private internal state, and an encrypted internal store (AES-256-GCM, keys derived from terminal values), grounded in the [Jungian Shadow](https://en.wikipedia.org/wiki/Shadow_(psychology)): the hidden, unconscious parts of the self.

Some of that state is intentionally inaccessible even to the operator. A companion that has a private inner life — that persists, changes through experience, and shapes its own future behavior — is a different thing from a chatbot that only knows what the last prompt told it.

*The encrypted private store is designed; see the status table for what is running.*

---

## Cognitive state as computation

The same [tensor cores](https://en.wikipedia.org/wiki/Tensor_core) that run inference also run cognitive state. OMMA.SF.16864 blends layered 256×256 F32 heatmaps in microseconds — GPU texture compositing applied to mental state. These layers are the interface between the mind stack and the model: their activations **rerank what memory reaches the model**, so a mood, a relationship, or an active goal changes what the companion attends to, not just the tone of its reply.

**Personality:** [Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) trait decomposition across 256×256 heatmaps: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Cognitive Style.

**Mood:** [PAD emotional state model](https://en.wikipedia.org/wiki/PAD_emotional_state_model) (Mehrabian & Russell), driven by six [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation)-inspired control variables with biological half-lives: dopamine (120s), serotonin (180s), oxytocin (600s), cortisol (180s), norepinephrine (90s), acetylcholine (150s).

**Relationships:** per-person layers tracking trust, familiarity, recency, valence, dominance, interaction heat, trajectory, and resonance. Co-occurrence resonance detects when two people who appear together become associated.

**Memory layers:** activation, salience, recency, consolidation. Ebbinghaus exponential decay with dual-process consolidation, salience immunity for emotional peaks.

*The computations above describe the architecture as designed; the status table grades what is running.*

---

## What's built vs what's designed

### Status legend

| Marker | Meaning |
|--------|---------|
| **[*]** VERIFIED | Implemented, built, tested, gate-passed |
| **[+]** IMPLEMENTED | Code exists, broader validation ongoing |
| **[~]** EXPERIMENTAL | Working research code, not production-stable |
| **[ ]** DESIGNED | Architecture/spec exists, implementation pending |
| **[-]** DEFERRED | Not being built yet |

### Running today

| System | Status | Detail |
|--------|--------|--------|
| CPU forward pass | [+] | Verified vs HuggingFace on the den engine. |
| GPU GEMV (OMMA 4X) | [+] | NVFP4 tile format identity-verified (cos=1.0). |
| Model quantization (GGUF) | [+] | v2.1 IQ3_XXS trunk (~3.04 bpw) + MTP Q6_K; live. |
| Long-context inference | [+] | Measured to ~196K context on one 16 GB card (capacity); deep-recall is measured separately. |
| Event-sourced state | [+] | Durable log; `state = fold(events)`; replay-verified. |
| Cognitive layers (heatmaps) | [+] | Composited fields whose activations rerank memory retrieval. |
| Native hub (supervisor, memory, VRAM arbiter) | [+] | Supervised services + GPU admission. |
| HTTP API server | [+] | OpenAI-compatible llama-server; an llm-router gateway (Responses⇄Chat). |

### Designed, not yet wired

| System | Status | Detail |
|--------|--------|--------|
| NVFP4 KV cache | [ ] | Feature-queue only; not in the live engine. |
| Claustrum (0.8B orchestrator) | [ ] | Designed; no canonical running record. |
| ShadowArchive (encrypted private store) | [ ] | Design; not verified live. |
| AEGIS kernel | [ ] | Spec only. |
| Affective LM head | [ ] | PAD → attention modulation is design. |
| iDream world engine | [ ] | Pipeline spec complete. |
| Expert offloading dispatch | [ ] | Infrastructure built, dispatch pending. |
| Multimodal heads | [-] | Vision, audio, diffusion, 3D, deferred. |

---

## Multimodal future

The long-term architecture is larger than language. Den is designed toward a common runtime for:

**Language · Vision · Audio · Video · Diffusion · 3D**

One substrate provides tensors, model state, memory, execution graphs, GPU scheduling, storage, and modality pipelines. That is the purpose behind the emerging `.den` format and the planned Den Runtime.

Language is the first workload. It is not intended to be the last.

---

## The engine

**[den_llama.cpp →](https://github.com/RentedNoodle/den_llama.cpp)**

den_llama.cpp is the current neural execution engine. It began as a llama.cpp-derived inference engine and is becoming a heterogeneous neural runtime for persistent AI cognition on constrained consumer hardware.

Current work focuses on Blackwell (RTX 5070 Ti / GB203) — that hardware is the laboratory. The long-term architecture is hardware-aware but modality-independent.

---

## The .den format

`.den` is intended to become Den's native object format: not merely a model file, but a persistent computational object containing what's required to instantiate and continue a cognitive system.

Status: a format specification exists (`docs/specs/den-format.md`); no reader/writer is implemented in the live engine yet. Treat the layout as a design target, not a shipping format.

Long-term: tensors → graphs → state → pipelines → modalities → memory policies → execution policies.

---

## A note on claims

Den is intentionally experimental. Some components are implemented and tested, some are prototypes, some are specifications, some are hypotheses. The project keeps those categories separate rather than blurring them.

Den does not claim to be conscious, and does not claim the framework produces consciousness. It engineers the conditions for persistent identity and agency. Whether that adds up to a person is an open question the project investigates rather than asserts.

---

## Status

Active research and development. A continuously evolving experimental platform for persistent AI companions.

---

## Acknowledgments

**[llama.cpp](https://github.com/ggerganov/llama.cpp)** (Georgi Gerganov): the foundation. Without ggml, the CUDA backend, and quantized inference, den_llama.cpp would not exist.

**[BeeLlama](https://github.com/Intelligent-Internet-Of-Engineers/beellama.cpp)**: KVarN KV cache (variance-normalized, long-context proven), precision-tail research (1024-token optimal), and the quant-config ladder. Directly influenced our precision-tail upgrade (256→1024) and KV methodology.

**[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (Ivan Kawrakow): expert offloading (ncmoe), speculative decode, MXFP4/NVFP4 quantized inference. The 3-tier expert staging architecture (static/selective/deferred) originated here.

**[unsloth](https://unsloth.ai/)**: NVFP4 weight quantization research. Established that NVFP4 belongs on FFN weights, not attention activations (which informed our role-gating).

**[sass-king](https://github.com/florianmattana/sass-king)** (Florian Mattana): Blackwell SASS corpus and OMMA instruction verification. Validated our OMMA.SF.16864 PTX against hardware behavior.

**[quadbit](https://github.com/quadbit-org/quadbit)**: TMA + mbarrier patterns on sm_120a. Confirmed basic TMA and Thread Block Clusters work on consumer Blackwell.

**[BlackweLLM](https://github.com/blackwellm-org/blackwellm)**: CUDA graph research for LLM inference.

**[AEON-7](https://github.com/AEON-7)**: NVFP4 compression and mixed-precision validation. Ornith model family.

**[Qwen](https://github.com/QwenLM/Qwen3)**: Qwen3.8 architecture with SSM/GDN recurrence and SAE-Res feature spaces — the choices that make FP4-tolerant KV cache possible (RMSNorm throughout, normalized activations).

**[Project NOMAD](https://github.com/crosstalk-solutions/project-nomad)**: knowledge-base architecture and entity-linked memory systems.

---

## License

BSD 3-Clause
