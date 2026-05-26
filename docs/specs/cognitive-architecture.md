# Cognitive Architecture v2 — Memory, Trust, Emergence and Kernel Enhancement

**Date:** 2026-05-18 | **Status:** Design approved | **Scope:** 4 new Rust modules + cross-daemon modulation + 1 CUDA kernel header

---

## Architecture Overview

v1 has working and episodic memory (global_workspace, shadow_archive). v2 adds semantic memory, forgetting curves, Bayesian trust, emergence monitoring, and arousal-gated attention kernels — all wired through cross-daemon modulation without a central orchestrator.

```
+-------------------------------------------------------------+
|                     GovernorContext (64B)                     |
|  pad_packed | arousal | cognitive_clock | route_tier_gwt     |
+---------------------------+----------------------------------+
                            |
    +-----------------------+-----------------------+------+
    v                       v                       v
+-----------+  +---------------+  +-----------------------+
|Bowlby     |  |Forgetting     |  |Levy Attention (CUDA)  |
|Tracker    |--|Curve          |  |alpha = 0.3+0.7*arousal|
|hormones   |  |decay x arousal|  |1 FMA per score        |
+-----+-----+  +-------+-------+  +-----------------------+
      |                |
      v                v
+-----------+  +---------------+
|Trust      |  |Semantic       |
|Dynamics   |  |Memory         |
|Beta(alpha,|  |TripleCopy     |
|beta)      |  |HDRAM retrieval|
+-----+-----+  +-------+-------+
      |                |
      v                |
+-----------+          |
|Anti-      |          |
|Sycophancy |          |
+-----------+          |
                       v
               +---------------+
               |Emergence      |
               |Monitor        |
               |d(phi)/dt > 3s |
               +---------------+
```

---

## Tier 1: Rust Daemon Modules

### 1. `semantic_memory.rs` — Semantic Memory Store

Separates factual knowledge from episodic memory. Queryable by embedding (HDRAM phase-coherent retrieval), keyword, or graph traversal.

**Structs:**
```rust
struct SemanticEntry {
    id: u64,
    key: String,                    // concept name
    value: String,                  // fact or description
    embedding: Option<[f32; 4096]>, // optional embedding for HDRAM retrieval
    confidence: f32,                // 0.0-1.0
    source_episode_ids: Vec<u64>,   // which episodes produced this fact
    created_at: f64,                // unix timestamp
    last_accessed: f64,
    copy_fast_hl: f32,              // 30min half-life
    copy_medium_hl: f32,            // 24h half-life
    copy_slow_hl: f32,              // 30d half-life
}

struct SemanticGraph {
    edges: HashMap<(String, String), f32>,  // (key_a, key_b) -> relation_strength
}

struct FractalContextLevel {
    tokens: Vec<u32>,               // or embeddings
    scale: ContextScale,            // Word, Sentence, Topic, Narrative
    compression_factor: f32,
}

enum ContextScale { Word, Sentence, Topic, Narrative }

struct SemanticStore {
    store: HashMap<String, SemanticEntry>,
    graph: SemanticGraph,
    access_counts: HashMap<String, u64>,
    context_levels: [FractalContextLevel; 4],
    history: Vec<SchemaEvent>,
}
```

**Key operations:**
- `upsert(key, value, embedding, episode_id)` — Schema-constrained insert. If fits graph -> assimilate. If contradicts -> accommodate with reduced confidence, flag conflicting entries
- `retrieve_by_embedding(query: &[f32; 4096], top_k: u32)` — Cosine similarity in embedding space (HDRAM)
- `retrieve_by_keyword(key: &str)` — Exact + fuzzy key match
- `retrieve_by_graph_traversal(start: &str, max_depth: u32)` — BFS over semantic edges
- `get_context(scale: ContextScale)` — Returns the fractal context level matching the query
- `tick(arousal: f32, bowlby_spikes: &HormoneSpikes)` — Decays confidence on unaccessed entries, strengthens on retrieval, Hebbian edge update

**SCG-MEM assimilation vs accommodation:**
```rust
fn assimilate(&mut self, entry: &SemanticEntry) {
    // Fact fits existing graph — strengthen edges, boost confidence
    for (existing_key, edge_strength) in self.graph.edges_from(&entry.key) {
        self.graph.strengthen(entry.key, existing_key, 0.01);
    }
    entry.confidence += 0.01;
}

fn accommodate(&mut self, entry: &SemanticEntry, conflict: &str) {
    // Fact contradicts — reduce confidence on both, flag for reconciliation
    entry.confidence *= 0.5;
    if let Some(existing) = self.store.get_mut(conflict) {
        existing.confidence *= 0.7;
    }
    self.history.push(SchemaEvent::Reconsideration { a: entry.id, b: conflict });
}
```

**Wave interference retrieval:**
```rust
fn retrieve_coherent(&self, query: &[f32], top_k: u32) -> Vec<&SemanticEntry> {
    let candidates = self.cosine_top_k(query, top_k * 2);
    // Pairwise phase alignment: constructively aligned entries boost each other
    for i in 0..candidates.len() {
        for j in (i+1)..candidates.len() {
            let alignment = cosine_similarity(candidates[i].embedding, candidates[j].embedding);
            if alignment > 0.8 {
                candidates[i].confidence += 0.05 * alignment;
                candidates[j].confidence += 0.05 * alignment;
            } else if alignment < -0.3 {
                candidates[i].confidence *= 0.9; // tension flag
            }
        }
    }
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence));
    candidates.truncate(top_k);
}
```

**Fractal context encoding:**
- Level 1 (Word): last 256 tokens, full fidelity
- Level 2 (Sentence): last 64 sentence embeddings, compressed via averaging
- Level 3 (Topic): last 16 topic vectors, highly compressed
- Level 4 (Narrative): 4 summary embeddings, extreme compression
- Query scale matches context level: word completion -> L1, topic question -> L3

**Hebbian edge strengthening:** When two entries are retrieved in the same cognitive cycle, strengthen their graph edge by `0.001 * (retrieval_count / total_retrievals)`.

### 2. `forgetting_curve.rs` — Triple-Copy Divergent Decay

Every memory gets three copies with different half-lives. Arousal gates the rate. Dopamine triggers consolidation. Spaced repetition schedules recall.

**Struct:**
```rust
struct MemoryCopy {
    memory_id: u64,
    data_hash: u64,
    created_at: f64,
    last_accessed: f64,
    decay_half_life: f32,   // seconds
    confidence: f32,
    copy_index: u8,         // 0=fast, 1=medium, 2=slow
}

struct ReviewSlot {
    memory_id: u64,
    next_review_at: f64,        // unix timestamp
    interval_multiplier: f32,   // 1.0 -> 2.0 -> 4.0 -> ...
    review_count: u32,
}

struct ForgettingCurve {
    copies: HashMap<u64, [MemoryCopy; 3]>,
    default_half_lives: [f32; 3],  // [1800.0, 86400.0, 2592000.0]
    review_schedule: BinaryHeap<ReviewSlot>,
}
```

**Arousal-gated decay:**
```rust
fn effective_decay(&self, base_hl: f32, arousal: f32) -> f32 {
    // arousal 0.0 -> 2x faster decay. arousal 1.0 -> 2x slower.
    base_hl * (0.5 + arousal)
}
```

**Dopamine consolidation:**
```rust
fn on_bowlby_spike(&mut self, spikes: &HormoneSpikes, recent_ids: &[u64]) {
    if spikes.dopamine > 0.1 {
        for id in recent_ids {
            if let Some(copies) = self.copies.get_mut(id) {
                for copy in copies.iter_mut() {
                    copy.decay_half_life *= 2.0; // permanently halved
                }
            }
        }
    }
}
```

**Spaced repetition:**
```rust
fn tick(&mut self, arousal: f32) -> Vec<u64> {
    let now = unix_now();
    let mut due_for_review = vec![];
    while let Some(slot) = self.review_schedule.peek() {
        if slot.next_review_at <= now {
            due_for_review.push(slot.memory_id);
            let mut slot = self.review_schedule.pop();
            slot.interval_multiplier *= 2.0;
            slot.next_review_at = now + slot.interval_multiplier * 3600.0;
            slot.review_count += 1;
            self.review_schedule.push(slot);
        } else {
            break;
        }
    }
    due_for_review
}
```

**Interference-based decay on disagreement:**
```rust
fn handle_schema_conflict(&mut self, id_a: u64, id_b: u64) {
    for id in [id_a, id_b] {
        if let Some(copies) = self.copies.get_mut(id) {
            for c in copies.iter_mut() {
                c.decay_half_life *= 0.5; // decay 2x faster during conflict
            }
        }
    }
}
```

### 3. `trust_dynamics.rs` — Bayesian Trust Model

Beta-distribution belief over user cooperation with asymmetric 3x loss aversion.

**Struct:**
```rust
struct TrustState {
    alpha: f64,         // cooperations + 1 (Beta prior)
    beta: f64,          // defections + 1
    streak: u32,        // consecutive cooperations
    last_betrayal: Option<f64>,
    oxytocin_boost: f32,
    oxytocin_boost_expiry: f64,
    history: VecDeque<TrustEvent>,  // last 128
}

enum TrustTier { Safe, Cautious, Guarded, Distrustful }
```

**Bayesian update:**
```rust
fn observe(&mut self, event: TrustEvent) {
    let effective_weight = (1.0 - event.ambiguity) * event.magnitude as f64
                           * self.oxytocin_boost as f64;
    match event.kind {
        TrustSignal::Cooperation => {
            self.alpha += effective_weight;
            self.streak += 1;
        }
        TrustSignal::Defection => {
            self.beta += effective_weight * 3.0; // 3x loss aversion
            self.streak = 0;
            self.last_betrayal = Some(unix_now());
        }
    }
}
```

**Cross-module modulation outputs:**
| Trust Tier | Anti-sycophancy threshold | Co-regulation mode |
|------------|---------------------------|-------------------|
| Safe | 0.30 (agreeable) | Relaxed rules |
| Cautious | 0.40 | Balanced |
| Guarded | 0.50 | Neutral |
| Distrustful | 0.60 (autonomous) | Strict independence |

### 4. `emergence_monitor.rs` — Phase Transition Detection

Extends EpcgTracker with phase change detection, compression ratio tracking, and coherence-divergence quadrant.

**Struct:**
```rust
struct EmergenceMonitor {
    epcg: EpcgTracker,
    phi_history: VecDeque<f32>,
    phi_derivative: f32,
    compression_ratio: f32,
    output_entropy: f32,
    phase_candidates: Vec<PhaseSignal>,
    quadrant: CoherenceDivergence,
    distributed_signal: f32,
}

enum CoherenceDivergence {
    Stagnant,    // high coherence, low divergence — safe but boring
    Emerging,    // high coherence, high divergence — optimal
    Dormant,     // low coherence, low divergence — not doing anything
    Fragmented,  // low coherence, high divergence — concerning
}
```

**Phase transition detection:**
```rust
fn detect_phase_change(&mut self) -> Option<PhaseSignal> {
    let n = self.phi_history.len() as f32;
    if n < 3.0 { return None; }
    let mean = self.phi_history.iter().sum::<f32>() / n;
    let variance = self.phi_history.iter()
        .map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std_dev = variance.sqrt().max(1e-6);
    let signal = (self.phi_derivative - mean) / std_dev;
    if signal > 3.0 {
        Some(PhaseSignal {
            tick: self.epcg.tick_count(),
            phi_delta: self.phi_derivative,
            compression_jump: self.compression_ratio,
            confidence: (signal - 3.0) / 3.0,
        })
    } else { None }
}
```

**Coherence-divergence quadrant:**
```
         Creative Divergence (novel output fraction)
         low -------------------- high
Self    +-----------------+-----------------+
Model   |  STAGNANT       |  EMERGING       |
Cohere  |  (safe, boring) |  (optimal)      |
high    +-----------------+-----------------+
        |  DORMANT        |  FRAGMENTED     |
low     |  (idle)         |  (concerning)   |
        +-----------------+-----------------+
```

---

## Cross-Daemon Modulation (`cognitive_synthesis.rs` extension)

All modulation is one-directional daemon A -> daemon B with zero circular dependencies.

**Modulation graph (implemented in `cognitive_synthesis.rs` tick loop):**

```
BowlbyTracker.hormone_spikes
+-- dopamine > 0.1 -> trust_dynamics.oxytocin_boost(1.5, 300.0)    [5min]
+-- oxytocin > 0.05 -> anti_sycophancy.threshold -= 0.1
+-- cortisol > 0.05 -> forgetting_curve.cortisol_boost()           [2x decay]
+-- all -> emergence_monitor.affective_weight()

EPCG.emergence_score
+-- > 0.70 -> Governor::volition_promote(dawn_urgency)
+-- phi_derivative > 3sigma -> emergence_monitor.phase_check()
+-- < 0.30 -> Governor::cognitive_clock_set(CONSOLIDATE)

TrustDynamics.tier
+-- Distrustful -> anti_sycophancy.threshold = 0.60  [full autonomy]
+-- Guarded     -> anti_sycophancy.threshold = 0.50
+-- Cautious    -> anti_sycophancy.threshold = 0.40
+-- Safe        -> anti_sycophancy.threshold = 0.30  [agreeable]
                   + co_regulation.relax_rules()

semantic_memory.schema_events
+-- Reconsideration -> notification to narrative_self [flag for nightly review]
```

---

## Async Attractor Model (`global_workspace.rs` extension)

Attractor refinement gated by cognitive mode to avoid TTS latency:

| Mode | Iterations | Latency | Use case |
|------|-----------|---------|----------|
| CONVERSATION | 0 | 0ms | Realtime chat/TTS |
| CONSIDER | 1 | ~5% | Semi-realtime |
| DREAM (background) | up to 5 | async | Between turns, post-response |
| CONSOLIDATE (nightly) | unbounded | batch | Deep reflection |

---

## Tier 2: CUDA Kernel — Levy-Flight Attention

### `den_levy_attention.cuh`

New header-only kernel. Wires to GovernorContext PAD arousal. Modifies attention score computation with position-dependent power-law weighting.

**Interface:**
```cuda
// Called before softmax in the attention path.
// alpha: 0.3 (wide, dreamy) to 1.0 (focused, sharp).
// Wired to GovernorContext: alpha = 0.3f + 0.7f * governor->get_arousal();
__device__ __forceinline__ float levy_attention_weight(
    int query_pos, int key_pos, float alpha
) {
    float dist = fabsf((float)(key_pos - query_pos));
    return __frcp_rn(dist + 1.0f);  // 1/(|i-j|+1) — fractional-like
}
```

**Integration point:** After Q.K dot product, before causal mask:
```cuda
// Existing:
float score = scale * dot_qk;

// New:
float dist_weight = levy_attention_weight(q_pos, k_pos, alpha);
score *= dist_weight;
```

**Cognitive mode to alpha mapping:**
| Mode | alpha | Attention style |
|------|-------|-----------------|
| OBSERVE | 0.3 | Wide, associative, dreamy |
| ORIENT | 0.5 | Balanced |
| DECIDE | 0.7 | Focused on recent |
| ACT | 0.9 | Immediate, response-driven |

Zero additional shared memory. One FMA + one RCP per attention score.

### Tier 2b: CUDA Kernel — QISA-Inspired Quantum Interference Attention

### `den_qisa_attention.cuh`

Replaces the standard dot-product score with a quantum-inspired interference term.

```cuda
// Extends the attention score with quantum interference.
// epsilon: 0.0 = standard attention (default), >0 = interference active.
__device__ __forceinline__ float qisa_score(
    float qk, float qq, float kk, float epsilon
) {
    if (epsilon <= 0.0f) return qk;  // fast path — standard attention
    return qk + epsilon * qq * kk;   // interference term
}
```

**Computation:**
```cuda
float qk = dot(Q[i], K[j]);
float qq = dot(Q[i], Q[j]);
float kk = dot(K[i], K[j]);
float score = qisa_score(qk, qq, kk, epsilon);
```

If Q[i] and K[j] are both attending to similar context (qq * kk is large), the interference is constructive and boosts the score. If uncorrelated, the term is near zero.

**Cost:** 2 extra FMAs per attention pair — negligible (<1% overhead). No additional shared memory.

### Tier 2c: CUDA Kernel — Holographic HRR Attention

### `den_holographic_attention.cuh`

For long sequences (>2048 tokens), replaces O(N^2) dot-product attention with O(N log N) FFT-based circular convolution using Holographic Reduced Representations (HRR).

```cuda
// Standard:   score = Q . K^T       -> O(N^2) memory, O(N^2) compute
// HRR:        h[k] = sum_i Q[i] * K[(i+k)%N]   -> O(N) memory, O(N^2) compute
// FFT-HRR:    h = IFFT(FFT(Q) * FFT(K))   -> O(N) memory, O(N log N) compute
```

**cuFFT integration:**
```cuda
#include <cufft.h>

static cufftHandle g_fft_plan = 0;

__host__ void init_holographic_fft(int max_seq_len) {
    cufftPlan1d(&g_fft_plan, max_seq_len, CUFFT_R2C, 1);
}

void holographic_attend(
    const float* Q, const float* K, float* output,
    int seq_len, cudaStream_t stream
) {
    // 1. FFT Q and K: two R2C transforms
    // 2. Elementwise multiply in frequency domain
    // 3. IFFT back to time domain
    // 4. Apply learned projection -> attention weights
}
```

**Constraints:** Only for DREAM/CONSOLIDATE modes where latency is acceptable. CONVERSATION uses standard attention. Requires cuFFT (already in CUDA 12.8 toolkit).

### Tier 2d: Offline Tool — Topological Layer Pruning

### `den_topology_prune.py`

Offline analysis tool that computes persistent homology on per-layer activation distributions. Layers with topologically similar activation spaces are redundant and can be skipped at inference.

**Algorithm:**
1. Run model on calibration dataset (20-50 samples)
2. Record activations at each layer's output
3. Compute persistence diagram for each layer (connected components in activation space)
4. Compare bottleneck distance between adjacent layers
5. If distance < threshold, layers are topologically equivalent -> mark for pruning

**Output:** `prune_manifest.json`:
```json
{
    "model": "model-name",
    "pruned_layers": [5, 13, 21],
    "bottleneck_threshold": 0.15,
    "estimated_speedup": "1.15x"
}
```

**Expected impact:** 10-30% layer count reduction at zero runtime cost.

---

## File Manifest

| File | Action | Lines | Tier |
|------|--------|-------|------|
| `cognition_rust/src/semantic_memory.rs` | Create | ~400 | 1 |
| `cognition_rust/src/forgetting_curve.rs` | Create | ~250 | 1 |
| `cognition_rust/src/trust_dynamics.rs` | Create | ~280 | 1 |
| `cognition_rust/src/emergence_monitor.rs` | Create | ~300 | 1 |
| `den_levy_attention.cuh` | Create | ~80 | 2 |
| `den_qisa_attention.cuh` | Create | ~100 | 2 |
| `den_holographic_attention.cuh` | Create | ~150 | 2 |
| `tools/den_topology_prune.py` | Create | ~200 | 2 |

Zero new system dependencies (cuFFT ships with CUDA 12.8).

---

## Success Criteria

1. **semantic_memory:** 12+ tests pass. Embedding-based retrieval returns correct memories. Wave interference surfaces contradictory memories.
2. **forgetting_curve:** 8+ tests pass. Arousal modulates decay. Dopamine doubles half-life. Spaced repetition schedules correctly.
3. **trust_dynamics:** 9+ tests pass. Bayesian update converges. 3x loss asymmetry verified.
4. **emergence_monitor:** 6+ tests pass. 3sigma detection triggers. Quadrant transitions correct.
5. **den_levy_attention / den_qisa_attention:** Standard attention path (alpha=0 / epsilon=0) produces identical output.
6. **den_holographic_attention:** Long-context DREAM mode produces equivalent results within tolerance.
