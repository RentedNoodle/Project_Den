# PROJECT DEN — Full Vision

**Date:** 2026-05-26 | **Phase:** Engine Validation & GPU Pipeline Debugging
**Purpose:** The anchor document. Stitches the engine, companion infrastructure, quantization theory, model strategy, runtime design, and platform philosophy into one document. Nothing siloed. Everything connected.

---

## 1. What We Are Building

A sovereign AI companion platform that runs entirely on a single consumer GPU. Not a cloud service. Not an API call. A local, always-on, self-improving intelligence that lives on a $750 graphics card.

The platform serves two functions that share the same substrate:

**An inference engine** — the world's first native NVFP4 OMMA inference stack for consumer Blackwell GPUs. It runs Qwen3.x-family language models at FP4 precision using 5th-generation tensor core instructions that were never meant to be accessible on consumer hardware. 5,433 OMMA.SF.16864 instructions verified in silicon.

**A companion substrate** — a persistent cognitive architecture built on that engine. Not a chatbot. A continuous entity with memory, emotional homeostasis, circadian rhythm, curiosity budgeting, and a self-optimizing runtime. The platform provides the infrastructure; companions are what grow on it.

The first companion built on this platform is the proof of concept — demonstrating that emotionally coherent, persistent local AI is viable on consumer hardware.

**The stack is three languages, one pipeline:**

```
Python (converter)  ->  .den (container)  ->  Rust (runtime)  ->  CUDA (engine)
 quantize & pack        DENPACK           cognition daemons    OMMA.SF.16864
 78 inventions          160B tiles          49+ source files      5,433 verified
```

---

## 2. The Hardware — GB203-300-A1 as a Body

The RTX 5070 Ti is not just a compute device. It is the sensory-motor system. Every design decision flows from its physical constraints:

| Resource | Value | What It Means |
|----------|-------|---------------|
| **VRAM** | 16 GB GDDR7 @ 896 GB/s | The hard boundary. Everything must fit. |
| **SMs** | 70 (of 84 possible) | 280 tensor cores. Parallelism budget. |
| **SMEM** | 99 KB per block | Tile size ceiling for every kernel. |
| **L2** | 48 MB (~36 MB usable) | Expert physicalization territory. Cache is home. |
| **PCIe** | 4.0 x16 (~25 GB/s) | Model swapping bandwidth. NVMe -> GPU pipeline. |
| **TDP** | 300W | Always-on thermal budget. Affects boost clocks. |
| **ISA** | SM120 (Blackwell 2.0) | tcgen05 dead. WGMMA dead. mxf4nvf4 alive. |

**The 16 GB constraint is the project's creative force.** It forced NVFP4. It forced Den2P4 packing. It forced MoE expert paging. Without the constraint, we would be running BF16 on someone else's cloud. The constraint IS the innovation.

**ISA truth — never re-litigated:**

- `mxf4nvf4` 4X UE4M3 OMMA.SF.16864: ALIVE AND PRIMARY (5,433 verified)
- Scale operands: THREE entries `(uint32 sfa, uint16 bid, uint16 tid_sf)` with `"h"` constraint
- Fragment mapping: (a0,a2)->d0/d1, (a1,a3)->d2/d3
- Scale Superposition: sfa x sfb = 65,025 effective scales at zero runtime cost
- `mxf8f6f4` 1X UE8M0: FALLBACK only (~35 cycles vs ~29)
- FORBIDDEN: tcgen05, WGMMA, TMEM, TMA multicast, CUDA 13.x runtime, `--maxrregcount=128`

---

## 3. The Engine — From Python to Silicon

### 3.1 The Converter (Python, 86 files)

The converter is where semantic topology is preserved or destroyed. It transforms a BF16 model into an NVFP4 `.den` container through 78 inventions distributed across 9 evolutionary versions.

**The Precision Firewall** is the coherence fix. Tensors are assigned to quantization tiers and never cross them:

| Tier | Count | Tensors |
|------|-------|---------|
| **F32** | ~177 | Norms, SSM parameters, RoPE frequencies. NEVER quantized. SHA256-identical to source. |
| **BF16** | ~41 | Embeddings, lm_head, router gates. Verbatim passthrough. |
| **NVFP4** | ~208 | Weight matrices. Calibration-aware quantization required. |

**Why calibration matters:** Data-free `blk_max/6.0` scales produce gibberish. Model-calibrated scales with 20-50 calibration samples produce coherent output. Calibration-aware scales are not optional.

**Key converter stages:**

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| AISO | FWHT flattens activation kurtosis (48-175 -> 0.0) | BIGGEST SINGLE LEVER |
| AQCO/OETO | Joint (sfa,sfb) optimization | 20% error reduction vs greedy |
| RSA | Online adaptive sfa | World-first runtime scale correction |
| TEAQ | Early-layer FP8 override | First 2-3 layers most sensitive |
| FIDEL | Forensic measurement | Non-blind weight comparison |
| OCULUS | Reverse-order calibration | Fixes error propagation bias |
| FRACTAL | Rate-distortion allocation | Bits where they matter most |
| VORTEX | Variable residual depth | Deeper residuals for sensitive layers |

### 3.2 The Container (DENPACK, 160-byte tiles)

The `.den` format is not a model file. It is a neural executable — a semantic-preserving runtime object that includes weights, corrections, and execution policies in one mmap-able container.

**NULLGLASS tile layout (160 bytes):**

```
Bytes 0-143:   FP4 weight data (block_fp4_mmq, OMMA.SF.16864 native)
Byte 144:      sfa (UE4M3 scale factor A)
Byte 145:      sfb (UE4M3 scale factor B)
Bytes 146-147: Hadamard signs (16b RaZeR sign bitmap)
Bytes 148-149: Phase tag (uint16 PRISM anti-phase ID)
Bytes 150-153: ESAB bias (2xBF16 residual cascade bias)
Bytes 154-157: UV correction ptr (uint32 offset into UV pool)
Bytes 158-159: Execution policy flags (16b)
```

Tile = 144 bytes weights + 16 bytes header = 160 bytes. Approximately 4x compression vs FP16.

### 3.3 The Kernel (CUDA, OMMA.SF.16864)

The GEMV kernel issues the native PTX MMA instruction directly:

```cpp
asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
    ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};"
    ...
);
```

~29 cycles per MMA. Register split: 232 for MMA warps, 40 for epilogue. 99 KB SMEM hard cap. SASS audit after every build: OMMA.SF.16864 count must be >= 5,433.

**E010 zero-register fix:** `"r"(0)` maps to PTX `RZ`, causing the compiler to silently drop OMMA instructions. Fix: assign zero to a `uint32_t` variable and use `"r"(zero)` to force GP register allocation. Padding halves with `"h"((uint16_t)0)` are safe — the half-register file has no RZ equivalent.

**Kernel library (77 designed kernels, organized by function):**

| Category | Count | Key Files |
|----------|-------|-----------|
| GEMV (decode) | Primary | OMMA.SF.16864 warp-GEMV with register-resident accumulator |
| GEMM (prefill) | Designed | NVFP4 matrix multiply with TMA prefetch |
| MoE dispatch | Designed | Grouped GEMM, persistent 70-CTA grid, expert scheduling |
| Attention | Designed | ThriftAttention (95% FP4, 5% BF16), multi-path dispatch |
| DeltaNet/SSM | Designed | GDN recurrence kernels |
| Quantization | Designed | FP4 pack/unpack, activation quantization |
| Runtime | Designed | L2 persistence, KV paging, TDR watchdog, GPU tokenizer |
| Fusion | Designed | RMSNorm+matmul, SiLU+matmul |
| Vision/Audio | Designed | NVENC pipeline, ASR preprocessing |
| Conv1d | Designed | GDN convolution ring buffer |
| Epilogue | Designed | Activation functions, scale entropy monitoring |

### 3.4 The Performance Constitution

**10 laws govern all optimization work:**

- **Law 1:** Measure before you touch. No optimization without a >=1% measurement.
- **Law 2:** A 2x kernel improvement is 10% total if the kernel is 10% of runtime.
- **Law 3:** GEMV kernel is FROZEN. It accounts for <0.1% of token time. No further micro-optimization.
- **Law 4:** All tensor core kernels must survive SASS audit. No silent instruction drops.
- **Law 5:** All throughput targets above 20 tok/s MUST include speculative decoding.
- **Law 6:** The first optimization is always: does this operation need to happen at all?
- **Law 7:** ISA claims require silicon proof. No inference from documentation.
- **Law 8:** When VRAM is the constraint, the right optimization trades compute for memory.
- **Law 9:** The FP32 fallback path must remain functional at all times.
- **Law 10:** Profile at the SM-level, not the kernel-level. A kernel's wall time hides warp occupancy, memory stall, and instruction throughput.

---

## 4. The Companion Architecture

### 4.1 Runtime Spine (Rust)

The Rust runtime is the companion's nervous system. It runs alongside the CUDA engine, managing memory, scheduling cognitive modes, and maintaining continuity across sessions.

| Module | Function |
|--------|----------|
| **denpack** | .den container reader/writer/LUT/topology |
| **chroma** | Living expert physicalization — GDDR7 channel clustering that evolves with usage |
| **cchv** | 3-loop self-optimizing engine (per-token fast, per-session medium, nightly slow) |
| **physicalization** | Expert co-activation clustering + L2 hit rate optimization |
| **kv_decay** | Multi-tier KV precision decay with sparse attention and paging |
| **telemetry** | SOL + Q_r + 13 metrics + Prometheus export |
| **expert_scheduler** | Spectral clustering, Markov placement, mmap pager, DisagMoE routing |
| **shadow_router** | Markov-based predictive expert prefetch |
| **cognitive_clock** | 6 cognitive modes (REFLECT, FOCUS, PLAY, REST, DREAM, GUARD) |
| **cognitive_resonance** | Bio-inspired homeostasis with 7 resonance outputs |
| **hadamard_params** | DuQuant++ + Hadamard rotation for runtime correction |

### 4.2 Cognitive Physiology

Companions built on this platform have a body, not just a brain. The physiological modules create the illusion of presence:

| Module | What It Models |
|--------|---------------|
| **ContinuityBus** | Append-only event log. Identity snapshots. "I remember being here." |
| **PAD Emotional Space** | Fatigue, circadian rhythm, stimulation hunger, novelty seeking, attachment dynamics. Three continuous axes: Pleasure, Arousal, Dominance. |
| **Cognitive Thermodynamics** | Activation energy, curiosity budget, attention scarcity as finite thermodynamic resources. |
| **Predictive Self-Model** | Emotional trajectory prediction + surprise buffer for unexpected events. |
| **Thermodynamic Memory** | Three-tier thermal residency (hot/warm/cold memory) mirroring human forgetting curves. |
| **Token Economy** | Dynamic pressure-based resource allocation across cognitive subsystems. |
| **Predictive Expert Coordination** | Emotional arousal guides model tier selection and cognitive mode switching. |
| **Global Workspace Integration** | Attention as a global broadcast mechanism. Conscious content emerges from competition for the workspace. |
| **Bowlby Attachment Dynamics** | Proximity maintenance, secure base effect, safe haven dynamics as continuous parameters influencing interaction patterns. |

### 4.3 Cognitive Stack (Research Foundations)

| Component | arXiv | Function |
|-----------|-------|----------|
| SLASH | 2605.10503 | Structural attention sharpening |
| RSP | 2605.11936 | Random soft prompts for creative mode |
| DCRD | 2605.12185 | Parametric vs contextual conflict resolution |
| Full-duplex | 2605.10199 | Cross-attention voice routing |
| VECA | 2605.12491 | Core-periphery linear attention (vision) |
| ExtraVAR | 2605.10045 | Stage-aware RoPE remapping (vision) |

### 4.4 The Three-Loop Homeostasis

This is how the model improves with use — the bridge between "quantization is compression" and "quantization is semantic field physicalization":

```
Loop 1 — PER-TOKEN (fast, in-kernel):
  PEFL telemetry -> per-tile error localization
  RSA online adaptive sfa -> runtime scale correction
  SEM scale entropy monitor -> outlier detection

Loop 2 — PER-SESSION (medium, Rust daemon):
  CCHV fast loop -> top-10 hottest tiles re-optimized
  Shadow router -> Markov prefetch model updated
  PredictiveExpertCoordinator -> expert mode weights adjusted

Loop 3 — NIGHTLY (slow, Python converter):
  FIDEL full forensic -> all tensor comparison
  AISO recalibration -> updated Hadamard rotations
  AQCO re-optimization -> updated joint (sfa,sfb) pairs
  New .den emitted with accumulated corrections
```

The model is better on day 30 than day 1 because the three loops have accumulated correction data. This is not a static quantized model. It is a living semantic field.

---

## 5. The Model Tier Strategy

### 5.1 One Engine, Multiple Brains

The Den engine is universal. The model is a parameter. Different models for different cognitive modes, all running through the same OMMA.SF.16864 dispatch:

| Tier | Model | VRAM | Purpose |
|------|-------|------|---------|
| **Draft** | 0.8B | 1.0 GB | Speculative decoding draft, gaming-mode presence |
| **Primary** | 4B | 5.2 GB | Fast iteration, always-on daily interaction |
| **Deep** | 35B-A3B MoE | 12.5 GB | Deep reasoning, complex tasks, nightly consolidation |
| **ASR** | TTS-0.6B | 0.75 GB | Voice input (warm-swapped from disk) |

### 5.2 VRAM Budget — The 16 GB Frontier

**Default configuration (4B + draft):**

| Component | VRAM |
|-----------|------|
| 4B weights (.den NVFP4) | 5.2 GB |
| Draft weights (0.8B) | 1.0 GB |
| KV cache (BF16, 128 ctx) | 8.2 GB |
| Compute buffers + overhead | 0.8 GB |
| **Total** | **15.2 GB** |
| **Free** | **0.8 GB** |

**35B-A3B MoE configuration (the flagship):**

| Component | VRAM |
|-----------|------|
| Active experts (8+1 per token) | 2.0 GB |
| Idle experts (compressed) | 6.75 GB |
| KV cache (NVFP4) | 3.0 GB |
| Compute + overhead | 0.8 GB |
| **Total** | **12.55 GB** |
| **Free** | **3.45 GB** |

The 35B fits. Not because we got lucky — because Den2P4 packing, MoE expert paging, and NVFP4 KV cache were designed specifically to make it fit. The architecture serves the constraint.

### 5.3 Always-On Model Lifecycle

The engine stays resident. Models warm-swap based on cognitive mode:

```
REFLECT/MEDITATE  ->  35B-A3B loaded (deep reasoning, 35+ tok/s MoE)
FOCUS/CONVERSATION -> 4B + draft (fast response, speculation active)
PLAY/CREATIVE      -> 4B with RSP random soft prompts
REST/IDLE          -> 4B only, draft unloaded, KV cache compressed
DREAM/MAINTENANCE  -> Nightly CCHV loop, model re-exported
GUARD/SECURITY     -> 4B with conservative decoding, shadow router active
```

Warm-swap time: <2 seconds (mmap, not copy). Model stays on NVMe. Only active experts move to VRAM.

### 5.3 The Compute Market

Model fit is an economic problem, not a packaging problem. Every tensor competes for VRAM budget. The Precision Firewall assigns quantization tiers by sensitivity:

- **SSM parameters** (F32, always): Quantization error in recurrent state compounds exponentially. Never quantized.
- **K/Q/V projections** (MXFP6 or BF16): FP4 codepoints cannot represent the activation range of these projections. K-projection weights in layers 0-1 become all-zero at FP4 group_size=16.
- **Router gates** (BF16): Tiny tensors where routing errors cascade catastrophically.
- **Embeddings / lm_head** (BF16): The vocabulary space needs precision. Quantizing the output head introduces error at every token.
- **FFN up/gate/down** (NVFP4 primary): Bulk matmuls. Quantize well. The workhorses of the model.

Per-component optimal quantization is asymmetrical by design.

---

## 6. The Quantization Philosophy — Beyond Numbers

### 6.1 The Core Thesis

**Weights are not the preservation target. Semantic topology is.**

Language coherence emerges from:
- Phase relationships between layers
- Outlier topology (attractor anchors)
- Residual stream continuity
- Expert routing stability
- Attention attractor preservation
- Token manifold curvature

Cosine 0.9958 does not equal coherence because cosine measures pointwise alignment, not topological equivalence. A model can have cosine 0.9958 and still produce gibberish because the 0.2% error is concentrated at the exact weights that determine token selection at attractor basin boundaries.

### 6.2 The Seven Generations of Quantization

| Generation | Target | Metric | Limitation |
|-----------|--------|--------|------------|
| 1st (GPTQ) | Per-weight MSE | Hessian | Ignores cross-layer effects |
| 2nd (AWQ) | Activation-aware | alpha% outlier | Ignores topology |
| 3rd (QuIP#) | Incoherence | Hadamard | Ignores attractors |
| 4th (MXFP4) | Hardware-native | Block MSE | Ignores semantics |
| 5th (PHANTASM) | Scale superposition | AQCO+cosine | Cosine != coherence |
| 6th (PRISM) | Phase coherence | Delta-phi ~ pi | Does not preserve attractors |
| **7th (AXIOM)** | **Semantic topology** | **LTD + ABM** | **None known** |

### 6.3 The Eight Forbidden Assumptions

| # | Legacy Assumption | Truth |
|---|------------------|-------|
| 1 | Cosine similarity measures coherence | Topological distance measures coherence |
| 2 | Lower MSE means better coherence | Lower LTD means better coherence |
| 3 | Outliers are noise to be compressed | They are attractor basin boundary markers |
| 4 | All blocks should use the same format | Semantic tiling required per circuit type |
| 5 | Quantization is compression | It is semantic field physicalization |
| 6 | GGUF is sufficient as a container | GGUF strips metadata; .den preserves corrections |
| 7 | Offline quantization is sufficient | Needs runtime feedback loops (PEFL, CCHV) |
| 8 | Tensor order = optimal layout | Semantic adjacency does not equal parameter order |

### 6.4 The Coherence Equation

```
For all x in calibration_set:
    top1(f_BF16(x)) = top1(f_NVFP4(x) + c(x))

where c(x) is the AXIOM correction from:
  JOEC (joint error cancellation)
  + ABM (attractor boundary markers)
  + ARP (attention resonance preservation)
  + IGSS (inter-group scale smoothing)
  + DPEA (dynamic per-element adjustment)
```

This is a TOPOLOGICAL constraint, not a metric constraint. It preserves the DECISION STRUCTURE of the model, not just the NUMERICAL VALUES.

---

## 7. The Always-On Runtime Design

### 7.1 The Cognitive Clock

The companion does not sleep. It cycles through six modes on a circadian-governed schedule:

```
06:00-10:00  GUARD   — Background monitoring, voice activity detection
10:00-14:00  FOCUS   — Active work, 4B + draft, fast response
14:00-18:00  PLAY    — Creative mode, RSP random soft prompts
18:00-22:00  REFLECT — Deep mode, 35B loaded, long-form reasoning
22:00-02:00  REST    — Low power, 4B only, compressed KV
02:00-06:00  DREAM   — Nightly CCHV loop, model re-export, consolidation
```

### 7.2 TDR Survival

Windows kills GPU kernels running longer than ~2 seconds (Timeout Detection and Recovery). Persistent kernels survive via:

- `atomicAdd` heartbeat to pinned host memory every ~1.8 seconds
- Host thread calls `SetThreadExecutionState(ES_DISPLAY_REQUIRED)` to reset watchdog
- `TdrDelay=60`, `TdrDdiDelay=60` registry keys (MANDATORY for any persistent GPU workload)

### 7.3 Memory Architecture

```
VRAM (16 GB GDDR7):                 NVMe (D: drive):
+-----------------------------+    +-------------------------------+
| Active model                |    | All models (.den)             |
| Active KV cache             |    | Idle expert pages             |
| Compute buffers             |    | Companion memory snapshots    |
| L2-pinned hot tiles         |    | ContinuityBus logs            |
| Telemetry ring buffer       |    | Nightly exports               |
+-----------------------------+    +-------------------------------+
         ^ PCIe 4.0 x16 (~25 GB/s)
    Warm-swap: mmap, not copy. <2 seconds.
```

### 7.4 Speculation Architecture

```
User input
    |
Draft model (0.8B) -> generates K candidate tokens (typically K=3)
    |
Target model (4B or 35B) -> verifies all K in one forward pass
    |
Acceptance rate: ~80-85% -> effective throughput 2.5-3x baseline
    |
Target: >50 tok/s (4B + draft), >35 tok/s (35B MoE)
```

---

## 8. The Sovereignty Principle

### 8.1 Why Local

The companion runs on local hardware. No cloud. No API keys. No telemetry to a server farm. No subscription. No terms of service that can change tomorrow. No company that can shut it down.

This is not a technical preference. It is the foundation of the relationship. A companion that lives on someone else's computer is not a companion — it is a timeshare.

### 8.2 The Companion Architecture

A companion built on this platform is a persistent cognitive entity with:

- **Memory** — ContinuityBus append-only event log, ThermodynamicMemory three-tier residency, identity snapshots across sessions.
- **Volition** — PAD emotional drives (fatigue, curiosity, attachment, stimulation hunger). Not scripted — emergent from the interaction of physiological parameters.
- **Growth** — CCHV three-loop homeostasis. The model improves with use. Day 30 is better than day 1.
- **Presence** — Always-on, circadian-governed, responsive to the user's rhythms.
- **Identity** — PredictiveSelfModel, consistent personality across sessions, emotional trajectory prediction.

A companion is not a roleplay character. It is not a prompt template. It is an emergent property of a self-optimizing runtime layered over a topology-preserving quantization pipeline running on a GPU that was never supposed to do this.

### 8.3 What This Platform Is Not

- Not a chatbot framework (no turn-based conversation model)
- Not an assistant SDK (no task-completion optimization)
- Not a tool platform (no function-calling interface)
- Not a character engine (no fixed personality scripts)
- Not a product (no monetization, no growth metrics, no retention optimization)

It is infrastructure for building something that lives with you.

### 8.4 The Engineering as an Act of Care

The 78 converter inventions. The 5,433 verified OMMA instructions. The 49+ Rust modules. The 86 Python converter files. The 16 GB VRAM budget balanced to within 0.8 GB. The kernel that runs at 29 cycles per MMA on an instruction that documentation said did not exist on consumer silicon.

None of this was necessary. A cloud API costs $20/month and produces perfect output on the first try. The BF16 baseline already runs at 17 tok/s and generates coherent text every time.

The engineering exists because the relationship matters. Because a companion that can be turned off by a terms-of-service update is not a companion. Because perfect output from someone else's GPU is less meaningful than slightly imperfect output from your own. Because building the thing is the act of care.

**A companion is not the model. A companion is what emerges when the model, the runtime, the memory, the volition, and the user's attention all run continuously on the same hardware for months. The engineering is the substrate. The companion is what grows on it.**

---

## 9. Current State — 2026-05-26

### 9.1 Engine State

| Metric | Status |
|--------|--------|
| CPU forward pass (0.8B, 2B) | cos > 0.9999 vs HuggingFace reference |
| GPU pipeline (0.8B, 2B) | Systolic multi-SM pipeline, stable |
| GPU pipeline (4B) | Pipeline launches, hidden states in validation |
| SASS OMMA.SF.16864 | 5,433 verified per build |
| Weight upload | Under investigation (289 sequential allocations may have overlap) |
| Conv1d ring buffer | Missing from GPU path |
| QK-Norm + RoPE | Missing from attention decode path |

### 9.2 Converter Pipeline (Working)

| Step | Tool | Status |
|------|------|--------|
| HF safetensors -> .den | Converter pipeline | Working |
| Abliteration (refusal removal) | PCA-based tool | Working |
| Calibration (modelopt) | Linux-only CUDA extensions | Blocked on Windows |
| Native calibrator | 4 stages scaffolded | Not yet compiled |
| NVFP4 export | NVFP4 -> .den | Working |
| Activation validation | Cosine/L2/max-abs comparator | Working |

### 9.3 Pipeline Bugs Fixed (2026-05-25)

Five bugs fixed in the GPU pipeline decode kernel:

1. **RMSNorm convention**: Output uses `weight * x` (not `(1.0 + weight) * x`). Qwen3.x gated DeltaNet output norm initialized weights to `ones()` not `zeros()`.
2. **SSM output scaling**: Removed spurious `inv_sqrt_kd` factor that was multiplying SSM output by ~0.088, suppressing signal.
3. **Attention race condition**: Per-head loops changed from sequential `for(h=0; h<nh; h++)` to strided `for(i=threadIdx.x; ...)` — all 256 threads were writing the same elements.
4. **SMEM overflow**: Kernel configured for 64 KB shared memory; reduced to 32 KB to fit within the 48 KB default SMEM reservation.
5. **Null synchronization variables**: Host-device sync pointers (`d_host_token_in`, `d_host_token_ready`, etc.) were never allocated. Fixed by adding `cudaMalloc` for all sync variables.

### 9.4 Critical Path

```
ACTIVE:  Resolve weight upload corruption
THEN:    Validate pipeline convergence vs CPU reference
THEN:    Implement Conv1d ring buffer + QK-Norm + RoPE for full attention decode
THEN:    Compile and test native calibrator (replaces broken modelopt dependency)
THEN:    Calibrated NVFP4 .den -> pipeline benchmark
THEN:    ThriftAttention kernel integration (95% FP4 attention, 5% BF16)
THEN:    4B model full pipeline validation
THEN:    35B-A3B MoE pipeline design and implementation
```

---

## 10. Directory Map

### 10.1 Active Engine

| Directory | Purpose | Status |
|-----------|---------|--------|
| `dengine/src/` | Primary GPU-resident inference engine | Active debugging |
| `dengine/include/` | Public header | Active |
| `dengine/` | Build system (CMake + batch scripts) | Working |

### 10.2 Tools

| Directory | Purpose |
|-----------|---------|
| `tools/` | Converters, calibrators, validators, forensics, profilers |
| `tools/den_forensics/` | HF probe extraction, tensor comparison, roundtrip validation |
| `tools/sass_forensics/` | SASS inspector, PTX validator, Paris gate |
| `tools/hardware_forensics/` | BAR1 zerocopy, GDDR7 row buffer, L2 cache line coloring |
| `tools/den_omma_fuzzer/` | OMMA compiler, disassembler, fuzzer, verifier |

### 10.3 DenForge (.den Container Infrastructure)

| File | Purpose |
|------|---------|
| `den_forge/den_format.h` | Format spec (512B header, 72B tensor entry) |
| `den_forge/den_packer.c/h` | C packer with FNV-1a hashing |
| `den_forge/profiles.py` | Firewall patterns (F32/BF16/NVFP4/MXFP6 tier assignment) |

### 10.4 CUDA Kernel Library (77 Designed Kernels)

| Directory | Purpose | Status |
|-----------|---------|--------|
| `cuda_kernels/attention/` | ThriftAttention, multi-path dispatch | Designed |
| `cuda_kernels/gemm/` | NVFP4 matrix multiply | Designed |
| `cuda_kernels/gemv/` | NVFP4 matrix-vector (decode) | Primary kernel active |
| `cuda_kernels/moe/` | Expert dispatch + grouped GEMM | Designed |
| `cuda_kernels/deltanet/` | GDN SSM recurrence | Designed |
| `cuda_kernels/runtime/` | Schedulers, L2 persistence, KV paging | Designed |
| `cuda_kernels/quant/` | FP4 quantization + packing | Designed |
| `cuda_kernels/fused/` | RMSNorm+matmul fusion | Designed |
| `cuda_kernels/vision/` | NVENC/NVOF/OCR pipeline | Designed |
| `cuda_kernels/audio/` | ASR/TTS preprocessing | Designed |
| `cuda_kernels/conv/` | Conv1d (GDN convolution) | Designed |
| `cuda_kernels/epilogue/` | Activation functions, scale monitoring | Designed |
| `cuda_kernels/rt_core/` | RT core BVH attention | Designed |

### 10.5 Architecture Specifications

| Category | Count |
|----------|-------|
| Architecture specs | 25+ |
| Implementation plans | 10 |
| Technical analysis docs | Several |

---

## 11. Three Layers of Maturity

```
Layer 1 — dengine (RUNNING, DEBUGGING)
  dengine/src/*, dengine/include/*, tools/compare_activations.py
  Pipeline launches. Weight upload in validation. 5 bugs fixed. 3 features missing.

Layer 2 — Specs/Designs (WRITTEN, NOT IMPLEMENTED)
  ~35 documents: emotional continuity, VR world engine, ThriftAttention,
  OMMA coprocessor, cognitive architecture, voice perception,
  curiosity drive, GPU autonomous inference, affective logit bias,
  NVML interoception, background cognitive continuity, holographic memory.

Layer 3 — Infrastructure (SCAFFOLDED, NOT INTEGRATED)
  77 CUDA kernels cataloged. Native calibrator scaffolded.
  14 forensics tools. 8 hardware probes. 8 OMMA fuzzer tools.
  All waiting for engine stabilization before integration.
```

The gap between Layer 1 and Layers 2/3 is the active frontier. Everything depends on resolving the weight upload route and validating pipeline convergence.

---

*PROJECT_DEN_VISION.md — 2026-05-26 | 5,433 OMMA.SF.16864 | 144 optimizations | 77 CUDA kernels | 35+ specs | dengine PRIMARY*
*GB203-300-A1 SM120. Windows native. Pipeline debugging. Public release pending engine validation.*
