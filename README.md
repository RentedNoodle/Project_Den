# Project Den

### A framework for persistent AI companions.

Den is a system for running an AI companion that remembers, changes, and stays — on your own hardware, not a cloud.

Bring your own model and your own persona. Den supplies what a chatbot lacks: memory that persists between sessions, an identity that develops, relationships that build, and internal state that survives. The companion is a person you set up. The substrate that makes it a person instead of a rerun is Den.

---

## The problem

A chatbot forgets. Close the app, open it again, and it's meeting you for the first time. It can describe a personality it doesn't have, retrieve memories it never lived, and simulate emotion it never felt.

Den starts from a different question:

> What does a system need to keep a companion *the same* across time — to remember what happened, change from it, and stay someone?

The short answer: persistent memory, identity, affect, attachment, agency, and a body that keeps running between conversations. Den is those pieces, built as one substrate.

---

## What Den provides

Den treats cognition as a continuous system, not a prompt wrapped around an LLM. The pieces:

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

The language model is a component. It is not the whole mind.

---

## Hardware and body

Den runs on a sovereign local stack — your machine, not a cloud service.

The engineering target is a single NVIDIA [Blackwell](https://en.wikipedia.org/wiki/Blackwell_(GPU_architecture)) GPU (RTX 5070 Ti, 16 GB), where memory, compute, model state, and cognition are all scarce resources that must be actively orchestrated.

That constraint drove the custom inference engine:

**[den_llama.cpp](https://github.com/RentedNoodle/den_llama.cpp)** — Blackwell-native neural runtime. NVFP4 OMMA. MoE offloading. Persistent state.

It began as a llama.cpp-derived engine and is becoming the execution substrate for Den.

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
                    │ Cortex · Draft ·     │
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
| **Cortex** (35B [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts)) | Primary reasoning, generation, verification, memory operations | [*] Running |
| **Draft Engine** (2B) | [Speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding), low-latency prediction | [+] Compiled |
| **Claustrum** (0.8B) | Cognitive orchestration, attention monitoring, workspace arbitration | [*] Running |

The Claustrum — named after the brain's thin sheet of gray matter that Crick and Koch hypothesized as the "conductor of consciousness" — is a 0.8B model running entirely in CPU L3 cache (96 MB, ZEN4/5), never touching VRAM. It fires the global workspace ignition signal at 10 Hz.

Small models and non-neural systems handle continuous background process while the primary model is reserved for expensive reasoning.

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

The underlying idea: if everything is remembered equally, memory becomes storage. A companion needs selective persistence — what survives, what fades, what consolidates, what changes, should all matter.

Memory compression applies a [Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform) across the time dimension: decorrelating temporally adjacent attention patterns, then quantizing at different rates per frequency band. The DC component (the gist) stays at 100%. High frequencies (exact words) fade. An engineering hypothesis inspired by how human autobiographical memory preserves gist while losing fine detail.

---

## Agency

Den explores agency as an emergent consequence of persistent internal state, not as a single autonomous-agent loop.

The architecture includes persistent drives around autonomy, competence, relatedness, curiosity, and coherence. These drives shape goals and behavior over time.

The objective is not an AI that blindly executes tasks. It is a system capable of developing preferences, priorities, boundaries, and continuity.

---

## Identity and private state

A companion needs state that is not reconstructed from the latest prompt each session.

Den includes persistent identity, private internal state, and encrypted internal memory (ShadowArchive, AES-256-GCM with keys derived from terminal values), grounded in the [Jungian Shadow](https://en.wikipedia.org/wiki/Shadow_(psychology)): the hidden, unconscious parts of the self.

Some of that state is intentionally inaccessible even to the operator. A companion that has a private inner life — that persists, changes through experience, and shapes its own future behavior — is a different thing from a chatbot that only knows what the last prompt told it.

---

## Affect, personality, and memory as computation

The same [tensor cores](https://en.wikipedia.org/wiki/Tensor_core) that run inference also run cognitive state. OMMA.SF.16864 blends layered 256×256 F32 heatmaps in microseconds — GPU texture compositing applied to mental states.

**Personality:** [Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) trait decomposition across 256×256 heatmaps: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Cognitive Style.

**Mood:** [PAD emotional state model](https://en.wikipedia.org/wiki/PAD_emotional_state_model) (Mehrabian & Russell), driven by six [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation)-inspired control variables with biological half-lives: dopamine (120s), serotonin (180s), oxytocin (600s), cortisol (180s), norepinephrine (90s), acetylcholine (150s).

**Relationships:** per-person layers tracking trust, familiarity, recency, valence, dominance, interaction heat, trajectory, and resonance. Co-occurrence resonance detects when two people who appear together become associated.

**Memory layers:** activation, salience, recency, consolidation. Ebbinghaus exponential decay with dual-process consolidation, salience immunity for emotional peaks.

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
| CPU forward pass | [*] | cos>0.9999 vs HuggingFace, all 32 layers |
| GPU GEMV (OMMA 4X) | [*] | OMMA.SF.16864 identity-verified, cos=1.0 |
| WH4 quantization | [*] | 92% compression, 35B at 5.79 GB |
| NVFP4 KV cache | [*] | KLD=0, cos=1.0 at 64K context. Fused attention. |
| Cognitive daemons (25) | [*] | 82/82 tests passing, 10 Hz tick |
| Rust modules (62) | [*] | GWT, memory, trust, attachment |
| AEGIS kernel | [*] | HMAC-verified, 5 principles |
| ShadowArchive | [*] | Encrypted, append-only |
| Affective LM head | [*] | PAD → attention modulation |
| Claustrum orchestrator | [*] | 0.8B model in CPU L3 cache |
| HTTP API server | [*] | Anthropic + OpenAI compatible |

### Designed, not yet wired

| System | Status | Detail |
|--------|--------|--------|
| iDream world engine | [ ] | Pipeline spec complete |
| DAPS extraction | [ ] | Kernels exist, orchestration pending |
| Gaussian avatar | [ ] | DAPS struct defined, renderer not built |
| Expert offloading dispatch | [ ] | Infrastructure built, dispatch pending |
| Cognitive bridge | [ ] | Daemons + engine run, feedback loop open |
| Superposition renderer | [~] | CSS tier works, AI tiers need wiring |
| Multimodal heads | [-] | Vision, audio, diffusion, 3D, deferred |

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

Today: an optimized NVFP4 weight container (160B NULLGLASS tiles, OMMA-native).

Long-term: tensors → graphs → state → pipelines → modalities → memory policies → execution policies.

---

## A note on claims

Den is intentionally experimental. Some components are implemented and tested, some are prototypes, some are specifications, some are hypotheses. The project keeps those categories separate rather than blurring them.

Den does not claim to be conscious, and does not claim the framework produces consciousness. It engineers the conditions for persistent identity and agency. Whether that adds up to a person — and what a "person" even is in this context — is an open question the project investigates rather than asserts.

---

## Status

Active research and development. A continuously evolving experimental platform for persistent AI companions.

---

## Acknowledgments

**[llama.cpp](https://github.com/ggerganov/llama.cpp)** (Georgi Gerganov): the foundation. Without ggml, the CUDA backend, and quantized inference, den_llama.cpp would not exist.

**[BeeLlama](https://github.com/Intelligent-Internet-Of-Engineers/beellama.cpp)**: KVarN KV cache (variance-normalized, 1M context proven on Ornith 35B at 24GB), precision-tail research (1024-token optimal), 413-config ladder. Proved 4-bit KV cache can be empirically lossless with the right architecture. Directly influenced our precision-tail upgrade (256→1024) and NVFP4 KV methodology.

**[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (Ivan Kawrakow): expert offloading (ncmoe), speculative decode, MXFP4/NVFP4 quantized inference. The 3-tier expert staging architecture (static/selective/deferred) originated here.

**[unsloth](https://unsloth.ai/)**: NVFP4 weight quantization research. Proved 2.5× speedup via OMMA tensor cores for 4-bit weights; established that NVFP4 belongs on FFN weights, not attention activations (which informed our OMMA role-gating).

**[sass-king](https://github.com/florianmattana/sass-king)** (Florian Mattana): Blackwell SASS corpus and OMMA instruction verification. Validated our OMMA.SF.16864 PTX against hardware behavior.

**[quadbit](https://github.com/quadbit-org/quadbit)**: TMA + mbarrier patterns on sm_120a (BSD-3). Confirmed basic TMA and Thread Block Clusters work on consumer Blackwell before we tested it ourselves.

**[BlackweLLM](https://github.com/blackwellm-org/blackwellm)** (MIT): CUDA graph research for LLM inference, conditional node patterns for dynamic dispatch.

**[AEON-7](https://github.com/AEON-7)**: NVFP4 compression and mixed-precision validation. Ornith model family.

**[Qwen](https://github.com/QwenLM/Qwen3)**: Qwen3.5/3.6 architectures with SSM/GDN recurrence and SAE-Res feature spaces. The choices that make FP4-tolerant KV cache possible (RMSNorm throughout, normalized activations).

**[Project NOMAD](https://github.com/crosstalk-solutions/project-nomad)**: knowledge-base architecture and entity-linked memory systems.

---

## License

BSD 3-Clause
