# Project Den
**An open framework for persistent local AI companions.**
> Status: pre-release. This repository contains design documentation. Components are implemented across the linked Den repos; Den ships no character — you bring your own model and character.

## The problem
A local language model is stateless. Run it twice with the same prompt and you get the same stranger: it remembers nothing between sessions and learns nothing. Inference is commoditized; continuity is not. Den is the other half — the state layer around a model that lets a conversation carry across sessions.

## What Den provides
- Durable event log — every change appended as a typed event; state rebuilt by folding the log (state = fold(events)); authoritative + replayable.
- Memory — episodic + semantic records with decay, scheduled consolidation, explicit provenance.
- Identity & relationships — an identity that updates from experience; per-person relationship state (trust, familiarity, recency, valence).
- Affect — PAD (three axes) + SIX neuromodulator-inspired scalars with exponential half-lives, biasing retrieval/response selection.
- Composable salience maps — layered 2D fields whose activations rerank which memory reaches the model (a control surface, not a model of experience).
- Attention arbiter — decides what competes for the model's context each turn.
- Rust hub — supervises services; a capability broker controls which tools a component may use; serves memory; arbitrates GPU memory (fails cleanly, never OOMs).
- Local inference — a long-context engine on a single 16 GB NVIDIA GPU; figures/build IDs live in the engine repo. No cloud, no accounts.

## How it works
turn → hub logs events → memory/affect/relationship state updates → salience maps rerank retrieval → model generates → new event logged. State is reconstruable from the log.

## Build your own companion
Den ships no character. Not shipped here: no model, no character, no UI, no avatar. You supply a model (local GGUF-compatible), a character (your own text), hardware (reference target: one 16 GB NVIDIA GPU, CUDA), config. Your companion's state is a directory you can back up and move.

## Implementation status
| Component | Status | Notes |
|---|---|---|
| Durable event log (state = fold) | Built | replay-verified |
| Memory service | Prototype | recall path hardening |
| Local inference engine | Built | short/mid-context live; long-context numbers in the engine repo |
| Rust hub: supervisor, VRAM arbiter | Prototype | |
| Capability broker | Prototype | non-escalation tested |
| Salience maps / retrieval reranking | Prototype | additive composite + rerank only |
| Affect (PAD + 6 neuromodulator scalars) | Prototype | parameters tunable, not fitted |
| Relationship state | Prototype | per-person isolation tested |
| Identity evolution | Spec | |
| Drives / agency | Spec | |
*Built = code exists + tested; Prototype = runs, unstable; Spec = design only.*

## A note on claims
Terms such as identity, affect, and drives refer to software state and control variables. Den makes no claim that a model is conscious or has subjective experience.

## Built on & acknowledgments
Den builds on open work (bare credits): llama.cpp (Georgi Gerganov); BeeLlama (KVarN KV cache, precision-tail research, quant-config ladder); ik_llama.cpp (Ivan Kawrakow — expert offloading, speculative decode, MXFP4/NVFP4); unsloth (NVFP4 weight-quant research); sass-king (Florian Mattana — Blackwell SASS/OMMA verification); quadbit (TMA + mbarrier on sm_120a); BlackweLLM (CUDA-graph research); AEON-7 (NVFP4 compression/mixed-precision); Qwen (Qwen3.8 architecture); Project NOMAD (knowledge-base/entity-linked memory).
Engine: [den_llama.cpp](https://github.com/RentedNoodle/den_llama.cpp) · License: BSD-3-Clause.
