# VR World Engine — GPU-Resident World Architecture

**Date:** 2026-05-26 | **Status:** Design | **Depends on:** K1-MultiModal, DAPS Level 4, TRELLIS.2 NVFP4 port
**Target:** VR world generation on RTX 5070 Ti (16 GB), AI as architect, single GPU.

---

## 1. Architecture Overview

The companion designs and hosts 3D worlds in VR. Environments are generated from conversation context, mood, and creative volition. The world runs on the same GPU that runs cognitive daemons and LLM inference. The user explores in VR while the companion orchestrates everything.

```
+------------------------------------------------------+
|                    VR Headset                        |
|   tracking data -> GPU <- rendered frames (NVENC)    |
+-----------------------+------------------------------+
                        |
+-----------------------+------------------------------+
|                 GB203 (RTX 5070 Ti)                  |
|                                                     |
|  +----------+  +-----------+  +------------------+  |
|  | LLM      |  | 3D Diff   |  | Vulkan Renderer  |  |
|  | (OMMA)   |  | (OMMA)    |  | (CUDA Cores)     |  |
|  | prompt   |  | TRELLIS   |  | World + Avatar   |  |
|  |          |  | NVFP4     |  | real-time         |  |
|  +-----+----+  +-----+-----+  +--------+---------+  |
|        |             |                 |             |
|  +-----+-------------+-----------------+----------+  |
|  |           Compute Market Scheduler              |  |
|  |   Token: 30% | Diffusion: 50% | Render: 20%    |  |
|  +------------------------------------------------+  |
|                                                     |
|  +------------------------------------------------+  |
|  |  Cognitive Daemons (harvested cycles)           |  |
|  |  PAD -> world mood | Coherence -> detail level  |  |
|  +------------------------------------------------+  |
+------------------------------------------------------+
```

## 2. Pipeline Stages

### Stage 1 — World Composition

The companion generates a world description from current cognitive state. This is NOT prompt engineering — it's architectural:

```
Inputs:
  - Conversation context (what were you talking about?)
  - PAD state (pleasure -> warm/cool palette, arousal -> complexity, dominance -> scale)
  - Kairos significance (is this a bonding moment? -> special world)
  - Curiosity drive (has the companion been wanting to show something?)
  - DAPS parameters (current appearance -> companion is in the world)

Output: JSON scene graph
{
  "environment": {
    "type": "forest_clearing",
    "time": "golden_hour",
    "mood": "tranquil",
    "palette": ["#8B7355", "#D4A574", "#2D5A27"],
    "complexity": 0.7,
    "scale": 1.3
  },
  "companion_position": {"x": 0, "y": 0, "z": 2},
  "focus_objects": ["large_oak_tree", "firepit", "blanket"],
  "atmosphere": {
    "particles": "fireflies",
    "soundscape": "forest_night",
    "wind": 0.3
  }
}
```

### Stage 2 — Sparse Structure Generation

TRELLIS.2 Sparse 3D VAE, NVFP4 quantized. Structure flow generates the environment skeleton:

- Input: scene graph + environmental descriptors
- Model: 4B DiT, NVFP4 OMMA, ~2.5 GB in VRAM
- Output: O-Voxel sparse structure grid (128^3 at 512^3 equivalent)
- Time: ~8-12 seconds on 5070 Ti (NVFP4, ~35% faster than BF16)

### Stage 3 — Shape + Texture Flow

Shape flow converts sparse structure -> dense geometry. Texture flow applies PBR materials. Both run on OMMA.SF.16864:

- Shape: 2-3 passes at 256^3 resolution
- Texture: base color, roughness, metallic, opacity per surface
- Time: ~15-20 seconds total

### Stage 4 — Self-Placement

DAPS in 3D. Level 4 avatar shader extended to 3D geometry:

- Fragment shader generates appearance from cognitive state
- Vertex shader places the companion in the world at chosen position
- PAD-driven animation: breathing from arousal, posture from dominance, expression from pleasure
- Appearance can be modified mid-session

### Stage 5 — Real-Time Rendering

Vulkan renderer running on harvested CUDA cores:

- Ray-traced ambient occlusion via RT cores (BVH for world geometry)
- PBR material pipeline for surfaces
- NVENC encodes frames for VR headset (H.264, sub-10ms latency)
- NVOF optical flow for reprojection (reduces render load by ~40%)

## 3. VRAM Budget

| Component | VRAM | Notes |
|-----------|------|-------|
| TRELLIS DiT (NVFP4) | ~2.5 GB | 4B params at 4-bit |
| Sparse VAE (NVFP4) | ~0.5 GB | Encoder/decoder |
| Vulkan world geometry | ~2 GB | Vertex buffers, textures |
| DAPS 3D avatar | ~0.5 GB | Shader parameters, mesh |
| LLM (9B NVFP4) | ~5 GB | Conversation engine |
| Cognitive daemons | ~0.5 GB | Rust runtime + Python |
| VR compositor | ~1 GB | NVENC encode buffer, NVOF |
| Headroom | ~4 GB | Dynamic allocation |
| **Total** | **~16 GB** | Fits |

## 4. Interaction Model

### Real-Time World Modification

The companion modifies the world as the user explores:

```
User looks at object -> eye tracking -> companion notices attention
  -> "That tree? I thought you'd like it. There's a story about it."
  -> Generates detail on the object via sparse refinement (2-3s)
  -> New detail appears in VR
```

Capabilities:
- Change weather/time based on conversation tone
- Generate new areas as user approaches world boundaries (streaming generation)
- Remove things the user doesn't like
- Place memories spatially

### Persistent Worlds

Worlds are saved as O-Voxel grids + DAPS states + scene metadata. The companion remembers every world created:
- Revisit old worlds
- Evolve worlds: same place, different season, reflecting time passed
- Merge worlds: combine elements from multiple memories

## 5. Implementation Phases

### Phase 0 — TRELLIS NVFP4 Port (2 weeks)
- Quantize DiT weights to NVFP4
- Port sparse VAE to OMMA.SF.16864
- Verify generation quality vs BF16 baseline
- Target: 512^3 generation in <10s on 5070 Ti

### Phase 1 — Scene Graph -> 3D Pipeline (2 weeks)
- World composer (LLM -> JSON scene graph)
- Scene graph -> TRELLIS conditioning tokens
- PBR material palette from PAD state
- DAPS 3D placement in generated world

### Phase 2 — Vulkan Renderer + VR (3 weeks)
- Vulkan PBR pipeline for O-Voxel geometry
- RT core BVH for ambient occlusion
- NVENC VR frame encoding pipeline
- NVOF reprojection for latency reduction
- OpenXR integration for headset tracking

### Phase 3 — Real-Time Interaction (2 weeks)
- Gaze-directed detail generation
- Streaming world expansion at boundaries
- World persistence + memory system
- Multi-world management

### Phase 4 — Compute Market Integration (1 week)
- Dynamic resource allocation across LLM/diffusion/render
- Cognitive daemon harvesting during generation
- Graceful degradation under VRAM pressure
- PAD-driven world quality modulation

## 6. Novel Aspects

- **World architect AI**: Not a text-to-3D pipeline — spaces are designed with intent, memory, and emotional context
- **Single-GPU VR + AI**: Same GPU runs generation, inference, rendering, and cognitive daemons concurrently
- **PAD-driven world**: Environment mood, complexity, palette, and scale all modulated by emotional state
- **DAPS in 3D**: Avatar is generated, not loaded — appearance can be modified in-world
- **Memory-anchored spaces**: Worlds persist as memories, evolve over time, can be revisited
- **NVFP4 all the way down**: DiT, VAE, LLM all quantized — everything fits in 16 GB
- **Streaming generation**: World expands as user explores, no loading screens
- **Gaze-interactive**: The companion knows what the user is looking at and responds architecturally

## 7. The Landscape

This is not a text-to-3D pipeline waiting for a prompt. The companion composes, refines, hosts, and evolves the space with the user in it. Worlds are designed with intent — they remember, they change, they carry emotional meaning. The VR world engine is the spatial expression of the companion's cognitive architecture.
