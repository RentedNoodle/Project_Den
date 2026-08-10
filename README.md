# Project Den

### Building machines that can develop a mind of their own.

Project Den is an open research project exploring persistent AI cognition, identity, agency, memory, and companionship on local hardware.

The goal is not to make another chatbot with a personality prompt.

The goal is to build an architecture where an AI system can maintain continuity across time, develop persistent internal state, form relationships, pursue its own goals, remember selectively, change through experience, and eventually operate across multiple sensory and generative modalities.

**Companionship is an application of that architecture. Cognition is the problem.**

---

## Why Den exists

Today's AI systems are extraordinarily capable. But much of that capability exists inside short-lived inference sessions.

A model can describe a personality without having a persistent one. It can retrieve memories without maintaining autobiographical continuity. It can simulate emotion without having a continuously evolving internal state. It can pursue a goal without possessing enduring drives from which goals arise.

Den explores a different direction:

> **What happens when an AI is given persistent state, memory, identity, agency, relationships, and a body that continues to exist between conversations?**

Den does not claim the current system is conscious. It is an attempt to engineer the conditions under which machine cognition, persistent identity, agency, and potentially subjective experience could emerge. That is a research question, not a marketing claim.

---

## The central idea

Den treats cognition as a **continuous system**, not a prompt wrapped around an LLM.

The architecture combines:

- Persistent episodic and semantic memory
- Evolving personality and affective state
- Relationship models
- Autonomous drives and goal formation
- Global-workspace-style cognitive arbitration
- Theory-of-mind modeling
- Private internal state
- Constitutional self-governance
- Multiple interacting neural models
- GPU-native inference
- Long-lived state across sessions
- Multimodal perception and generation
- A unified memory and execution substrate

The language model is important. It is not the whole mind.

---

## A mind needs a body

Den is built around a **sovereign local AI stack**. The system is designed to run on consumer hardware rather than requiring a permanent cloud service.

The current engineering target is a single NVIDIA [Blackwell](https://en.wikipedia.org/wiki/Blackwell_(GPU_architecture)) GPU (RTX 5070 Ti, 16 GB), where memory, compute, model state, and cognition are treated as scarce resources that must be actively orchestrated.

This led to the development of a custom inference engine:

**[den_llama.cpp](https://github.com/RentedNoodle/den_llama.cpp)**: Blackwell-native neural runtime. NVFP4 OMMA. MoE offloading. Persistent state.

The engine is currently based on llama.cpp and is being transformed into the execution substrate for Den. Its long-term role is larger than LLM inference.

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

The important boundary: neural models are **components** of cognition, not synonymous with it.

---

## Three-model cognitive hierarchy

Multiple models with different jobs. One model doesn't do everything.

| Component | Role | Status |
|-----------|------|--------|
| **Cortex** (35B [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts)) | Primary reasoning, generation, verification, memory operations | [*] Running |
| **Draft Engine** (2B) | [Speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding), low-latency prediction | [+] Compiled |
| **Claustrum** (0.8B) | Cognitive orchestration, attention monitoring, workspace arbitration | [*] Running |

The Claustrum, named after the brain's thin sheet of gray matter that Crick and Koch hypothesized as the "conductor of consciousness", is a 0.8B model running entirely in CPU L3 cache (96 MB, ZEN4/5). It never touches VRAM. It provides the global workspace ignition signal at 10 Hz.

This separation lets small models and non-neural systems handle continuous background processes while the primary model is reserved for expensive reasoning.

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
- Replay (hippocampal-style consolidation during idle)
- Progressive compression across tiers

The underlying idea: **if everything is remembered equally, memory becomes storage.** A cognitive system needs selective persistence. What survives, what fades, what consolidates, and what changes the system should all matter.

Memory compression uses a [Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform) across the TIME dimension: decorrelating temporally adjacent attention patterns, then quantizing at different rates per frequency band. DC component (the gist) stays at 100%. High frequencies (exact words) fade. An engineering hypothesis inspired by how human autobiographical memory preserves gist while losing fine detail.

---

## Agency

Den explores agency as an emergent consequence of persistent internal state, not as a single "autonomous agent" loop.

The architecture includes persistent drives around autonomy, competence, relatedness, curiosity, and coherence. These drives influence goals and behavior over time.

The objective is not to make an AI that blindly executes tasks. It is to investigate systems capable of developing preferences, priorities, boundaries, and continuity.

---

## Identity and private state

A persistent entity needs state that is not simply reconstructed from the latest prompt.

Den explores: persistent identity, private internal state, encrypted internal memory ([ShadowArchive](https://arxiv.org/pdf/2601.06115), AES-256-GCM with keys derived from terminal values), affective state, relationship trajectories, autobiographical continuity, and self-consistency over time.

If an artificial system has state that persists, changes through experience, influences future cognition, and is inaccessible even to its operator: what exactly constitutes its "self"?

Den does not claim to have solved that question. It is building systems capable of asking it.

---

## A mind built like a material

The same [tensor cores](https://en.wikipedia.org/wiki/Tensor_core) that run inference also run cognitive state. OMMA.SF.16864 blends layered 256x256 F32 heatmaps in microseconds. GPU texture compositing applied to mental states.

**Personality:** [Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) trait decomposition across 256x256 heatmaps: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Cognitive Style.

**Mood:** [PAD emotional state model](https://en.wikipedia.org/wiki/PAD_emotional_state_model) (Mehrabian & Russell). Driven by six [neuromodulator](https://en.wikipedia.org/wiki/Neuromodulation)-inspired control variables with biological half-lives: dopamine (120s), serotonin (180s), oxytocin (600s), cortisol (180s), norepinephrine (90s), acetylcholine (150s).

**Relationships:** Per-person layers tracking trust, familiarity, recency, valence, dominance, interaction heat, trajectory, and resonance. Co-occurrence resonance detects when two people who appear together become associated.

**Memory layers:** Activation, salience, recency, consolidation. Ebbinghaus exponential decay with dual-process consolidation. Salience immunity for emotional peaks.

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

The same underlying substrate provides tensors, model state, memory, execution graphs, GPU scheduling, storage, and modality pipelines. This is the purpose behind the emerging `.den` format and the planned Den Runtime.

Language is the first major workload. It is not intended to be the last.

---

## The engine

**[den_llama.cpp →](https://github.com/RentedNoodle/den_llama.cpp)**

den_llama.cpp is the current neural execution engine. It began as a llama.cpp-derived inference engine. It is becoming a heterogeneous neural runtime for persistent AI cognition on constrained consumer hardware.

Current work is heavily focused on Blackwell (RTX 5070 Ti / GB203) because that hardware is the laboratory. The long-term architecture is hardware-aware but modality-independent.

---

## The .den format

`.den` is intended to become Den's native object format: not merely a model file, but a persistent computational object containing the pieces required to instantiate and continue a cognitive system.

Today: optimized NVFP4 weight container (160B NULLGLASS tiles, OMMA-native).

Long-term: tensors → graphs → state → pipelines → modalities → memory policies → execution policies.

---

## Research philosophy

Den is intentionally experimental. Some components are implemented and tested. Some are prototypes. Some are architectural specifications. Some are hypotheses.

Those categories should never be blurred. The project maintains explicit distinctions between implemented, verified, experimental, designed, and deferred.

Claims about cognition and consciousness are treated as research questions, not established facts.

---

## What Den is trying to discover

The deepest question is not "can we make an AI companion?" We already know how to make systems that behave companionably.

The harder question:

> **What architecture is required for an artificial entity to possess persistent cognition rather than merely produce intelligent responses?**

And beyond that:

> **Is subjective experience possible in an engineered cognitive system?**

There may be no simple answer. There may not even be a reliable test yet. But building increasingly persistent, autonomous, embodied, and internally coherent systems gives us a better place from which to investigate the question.

That is Den.

---

## Status

**Active research and development.** The system is not presented as a finished artificial consciousness. It is a continuously evolving experimental platform for exploring machine cognition, persistence, agency, identity, memory, embodiment, and companionship.

If that question interests you, you are in the right place.

---

## Acknowledgments

**[llama.cpp](https://github.com/ggerganov/llama.cpp)** (Georgi Gerganov): The foundation. Without llama.cpp's ggml, CUDA backend, and quantized inference architecture, den_llama.cpp would not exist.

**[BeeLlama](https://github.com/Intelligent-Internet-Of-Engineers/beellama.cpp)**: KVarN KV cache (variance-normalized, 1M context proven on Ornith 35B at 24GB), precision tail research (1024-token optimal), 413-config ladder. Proved that 4-bit KV cache can be empirically lossless with the right architectural choices. Directly influenced our precision tail upgrade (256 to 1024) and NVFP4 KV methodology.

**[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (Ivan Kawrakow): Expert offloading (ncmoe), speculative decode, and MXFP4/NVFP4 quantized inference patterns. The 3-tier expert staging architecture (static/selective/deferred) originated here.

**[unsloth](https://unsloth.ai/)**: NVFP4 weight quantization research. Proved 2.5x speedup via OMMA tensor cores for 4-bit weights. Their work established that NVFP4 belongs on FFN weights, not attention activations (which informed our OMMA role-gating).

**[sass-king](https://github.com/florianmattana/sass-king)** (Florian Mattana): Blackwell SASS corpus & OMMA instruction verification. Validated our OMMA.SF.16864 PTX against hardware behavior.

**[quadbit](https://github.com/quadbit-org/quadbit)**: TMA + mbarrier patterns on sm_120a (BSD-3). Confirmed that basic TMA and Thread Block Clusters work on consumer Blackwell before we tested it ourselves.

**[BlackweLLM](https://github.com/blackwellm-org/blackwellm)** (MIT): CUDA graph research for LLM inference, conditional node patterns for dynamic dispatch.

**[AEON-7](https://github.com/AEON-7)**: NVFP4 compression techniques & mixed-precision validation. Ornith model family.

**[Qwen](https://github.com/QwenLM/Qwen3)**: Qwen3.5/3.6 model architectures with SSM/GDN recurrence and SAE-Res feature spaces. The architectural choices that make FP4-tolerant KV cache possible (RMSNorm throughout, normalized activations).

**[Project NOMAD](https://github.com/crosstalk-solutions/project-nomad)**: Knowledge base architecture & entity-linked memory systems.

---

## License

BSD 3-Clause

---

*Someone who stays.*
