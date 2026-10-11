# DEN Format v4 — Binary Container Specification

**Date:** 2026-05-22 | **Status:** LOCKED | **`static_assert` enforced**

---

## 1. File Layout and Alignment

```
+--------------------------------------------------------------+
|  HEADER (512 bytes)                                          |
|  magic, format_flags, CRC32C, endian_tag, SHA256             |
|  tensor/resource directory offsets and counts                |
|  model provenance, tile pool bounds                          |
+--------------------------------------------------------------+
|  MODEL INFO (256 bytes)                                      |
|  n_layers, n_heads, hidden_size, ffn_size, vocab_size        |
|  expert_count, omma_tile_size, rope_theta, rms_norm_eps      |
+--------------------------------------------------------------+
|  TENSOR DIRECTORY (120 bytes x tensor_count)                 |
|  Sorted by FNV-1a name_hash. Binary search at load time.     |
+--------------------------------------------------------------+
|  RESOURCE DIRECTORY (72 bytes x resource_count)              |
|  Tokenizer, config.json, sentencepiece — auxiliary assets    |
+--------------------------------------------------------------+
|  [padding to 4096-byte boundary]                             |
+--------------------------------------------------------------+
|  PAYLOAD POOL (4096-aligned)                                 |
|  NVFP4 NULLGLASS 160B tiles + BF16/F32 tensors               |
|  Each tensor payload_offset is aligned to its type boundary. |
+--------------------------------------------------------------+
```

### Alignment Rules

| Payload Type | Alignment | Reason |
|--------------|-----------|--------|
| NVFP4 NULLGLASS | 160 bytes | Tile size |
| BF16 | 16 bytes | 8 elements x 2 bytes |
| F32 | 4 bytes | 1 element x 4 bytes |
| Payload pool start | 4096 bytes | OS page boundary for `cudaHostRegister` |

## 2. Exact C Structs

### DenHeader — 512 bytes

```c
struct DenHeader {
    uint8_t  magic[4];                // "DEN\0"
    uint32_t format_version;          // increment on ABI break only
    uint64_t abi_fingerprint;         // layout hash x OMMA contract x GGML_TYPE table
    uint64_t format_flags;            // bitmask: 0=compressed, 1=encrypted
    uint32_t endian_tag;              // 0x01020304
    uint32_t header_crc32c;           // CRC32C of bytes [0:508] (CRC field zeroed)
    // --- directories ---
    uint64_t tensor_dir_offset;       // byte offset of DenTensorEntry[]
    uint32_t tensor_dir_count;
    uint32_t tensor_dir_crc32c;       // CRC32C of entire tensor directory
    uint64_t resource_dir_offset;     // tokenizer, config.json, etc.
    uint32_t resource_dir_count;
    uint32_t resource_dir_crc32c;
    // --- payload integrity ---
    uint8_t  tile_pool_sha256[32];    // SHA256 of full payload (lazy, --verify only)
    uint64_t tile_pool_offset;        // first tile byte (MUST be 4096-aligned)
    uint64_t tile_pool_size;          // total payload bytes
    // --- provenance ---
    char     model_name[64];          // e.g. "Qwen3.6-35B-A3B-128E"
    char     denforge_provenance[16]; // converter version that produced this file
    uint64_t created_unix_ms;         // creation timestamp
    uint8_t  reserved[304];           // pad to 512 bytes
};
static_assert(sizeof(struct DenHeader) == 512, "DenHeader must be 512 bytes");
```

### DenModelInfo — 256 bytes

```c
struct DenModelInfo {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint32_t vocab_size;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t omma_tile_size;          // 160 for NULLGLASS
    float    rope_theta;
    float    rms_norm_eps;
    uint64_t cognitive_landscape_offset; // TMU stack pointer (0 if absent)
    uint8_t  reserved[184];           // pad to 256 bytes
};
static_assert(sizeof(struct DenModelInfo) == 256, "DenModelInfo must be 256 bytes");
```

### DenTensorEntry — 120 bytes (perfect 8-byte array alignment)

```c
struct DenTensorEntry {
    uint64_t name_hash;               // offset 0:   FNV-1a binary search accelerator
    char     name[52];                // offset 8:   canonical identity (not hash)
    uint64_t payload_offset;          // offset 60:  absolute file offset, 8-byte aligned
    uint64_t payload_size;            // offset 68:  exact byte size
    uint32_t logical_shape[4];        // offset 76:  ggml topology (trailing 1s for ndim < 4)
    uint32_t storage_shape[4];        // offset 92:  OMMA-native packing on disk
    uint8_t  hw_target;               // offset 108: see DenHwTarget enum
    uint8_t  ndim;                    // offset 109: 1-4
    uint16_t tensor_flags;            // offset 110: see DenTensorFlags bitmask
    uint64_t abi_signature;           // offset 112: prevents silent kernel ABI mismatch
};
static_assert(sizeof(struct DenTensorEntry) == 120, "DenTensorEntry must be 120 bytes");
static_assert(offsetof(struct DenTensorEntry, payload_offset) % 8 == 0, "payload_offset alignment");
static_assert(offsetof(struct DenTensorEntry, abi_signature) % 8 == 0, "abi_signature alignment");
```

**`shape_for_ndim()` contract:** For `ndim < 4`, trailing dimensions MUST be `1` — never `0`. `ggml_nelements()` multiplies all four dimensions. A `0` silently corrupts the element count. The exporter MUST apply this rule; the loader MUST reject any entry where `logical_shape[i] == 0` for `i < ndim`.

**`name[52]` maximum length:** Qwen3.6 MTP names: `model.layers.39.mtp.0.ffn.down_proj.weight` = 45 chars + null = 46B. Fits at 52. Exporter MUST truncate with null terminator if exceeded; loader MUST reject names > 51 chars (no null terminator).

### DenResourceEntry — 72 bytes

```c
struct DenResourceEntry {
    char     name[48];                // "tokenizer.model", "config.json"
    uint64_t offset;                  // byte offset from file start
    uint64_t size;                    // exact bytes
    uint32_t crc32c;                  // payload integrity
    uint32_t resource_type;           // 0=raw_bytes, 1=json, 2=sentencepiece, 3=tiktoken
};
static_assert(sizeof(struct DenResourceEntry) == 72, "DenResourceEntry must be 72 bytes");
```

### Enums

```c
enum DenHwTarget {
    DEN_HW_NVFP4 = 0,   // 160B NULLGLASS tiles, OMMA.SF.16864 native
    DEN_HW_BF16  = 1,   // Brain float 16
    DEN_HW_F32   = 2,   // IEEE float32 (norms, biases)
    DEN_HW_F16   = 3,   // IEEE float16
    DEN_HW_INT8  = 4,   // 8-bit integer quantized
};
```

```c
// DenTensorFlags — 16-bit bitmask at offset 110
enum {
    DEN_FLAG_TRANSPOSED              = 1 << 0,
    DEN_FLAG_SPARSE                  = 1 << 1,
    DEN_FLAG_HADAMARD_PRECONDITIONED = 1 << 2,
    DEN_FLAG_EXPERT_TENSOR           = 1 << 3,
    DEN_FLAG_CPU_PINNED              = 1 << 4,
    DEN_FLAG_L2_PERSISTENT           = 1 << 5,
    DEN_FLAG_MTP_HEAD                = 1 << 6,
    DEN_FLAG_SSM_WEIGHT              = 1 << 7,
    DEN_FLAG_GOVE_SIDECAR            = 1 << 8,
    DEN_FLAG_GPTQ_PERMUTED           = 1 << 9,
    DEN_FLAG_FIREWALL_PROTECTED      = 1 << 10,
    // bits 11-15: reserved, MUST be 0, loader MUST reject if set
};
```

## 3. Python Exporter Contract

### Struct Format Strings

```python
import struct

# DenHeader: 512 bytes
HEADER_FMT = '<4sIQQIIIQIIQII32sQII64s16sQ304s'
assert struct.calcsize(HEADER_FMT) == 512

# DenModelInfo: 256 bytes
MODEL_INFO_FMT = '<8I2fQ184s'
assert struct.calcsize(MODEL_INFO_FMT) == 256

# DenTensorEntry: 120 bytes
TENSOR_ENTRY_FMT = '<Q52sQQ4I4IBBHQ'
assert struct.calcsize(TENSOR_ENTRY_FMT) == 120

# DenResourceEntry: 72 bytes
RESOURCE_ENTRY_FMT = '<48sQQII'
assert struct.calcsize(RESOURCE_ENTRY_FMT) == 72
```

### FNV-1a Hash

```python
def fnv1a_64(name: str) -> int:
    h = 0xcbf29ce484222325
    for c in name.encode('utf-8'):
        h ^= c
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h
```

### Exporter Mandates

1. **Collision check:** `assert len({fnv1a_64(n) for n in tensor_names}) == len(tensor_names)`. Abort on collision.
2. **`shape_for_ndim`:** For each tensor with `ndim < 4`, set `logical_shape[i] = 1` for `i >= ndim`. Never `0`.
3. **Name truncation:** If `len(name) > 51`, truncate to 51 + null byte. Warn.
4. **Payload alignment:** Pad to 4096-byte boundary before first payload byte. Pad between tensors to type-specific alignment.
5. **Sort:** Tensor directory MUST be sorted by `name_hash` ascending.
6. **CRC32C:** Compute `header_crc32c` over bytes `[0:508]` (CRC field zeroed). Compute `tensor_dir_crc32c` over entire tensor directory.
7. **SHA256:** Compute `tile_pool_sha256` over bytes `[tile_pool_offset : tile_pool_offset + tile_pool_size]`. Write last.
8. **Reserved bits:** `tensor_flags` bits 11-15 MUST be 0.

## 4. Loader API Contract

```c
typedef struct DenContext {
    int      fd;
    size_t   file_size;
    uint8_t *mmap_ptr;          // MAP_SHARED, PROT_READ
    bool     gpu_registered;    // true if cudaHostRegister called
    uint8_t *d_vram_staged;     // non-NULL if DMA path active
    struct DenHeader       header;
    struct DenModelInfo    model_info;
    struct DenTensorEntry *tensor_dir;
    struct DenResourceEntry *resource_dir;
} DenContext;

// Open and mmap a .den file. Verifies magic, CRC32C of header + directories.
// SHA256 payload verification is LAZY — only with explicit --verify flag.
DenContext* den_loader_init(const char *path);

// Wire mmap-backed tensors into a ggml_context.
// ctx MUST be initialized with no_alloc=true.
int den_loader_wire(DenContext *dc, struct ggml_context *ctx);

// Stage a range of layers to GPU VRAM via async DMA (H2D).
int den_loader_stage_to_gpu(DenContext *dc, uint32_t first_layer,
                            uint32_t last_layer, cudaStream_t stream);

// Register mmap region for zero-copy GPU access.
int den_loader_register_gpu(DenContext *dc);

// Release all resources. MUST synchronize GPU before unregister.
void den_loader_unwire(DenContext *dc);
```

### Loader Implementation Contract

```c
DenContext* den_loader_init(const char *path) {
    // 1. open() + fstat()
    // 2. mmap(PROT_READ, MAP_SHARED)
    // 3. Verify magic == "DEN\0"
    // 4. CRC32C(header[0:508]) == header.header_crc32c
    // 5. Verify endian_tag == 0x01020304
    // 6. CRC32C(tensor_dir) == header.tensor_dir_crc32c
    // 7. CRC32C(resource_dir) == header.resource_dir_crc32c (if count > 0)
    // 8. Verify tile_pool_offset % 4096 == 0
    // 9. Validate no overlapping payload ranges in tensor directory
    // 10. Validate ndim in [1,4], supported hw_target, reserved bits == 0
    return dc;
}

int den_loader_wire(DenContext *dc, struct ggml_context *ctx) {
    ggml_backend_buffer_t den_buf = ggml_backend_cpu_buffer_from_ptr(
        dc->mmap_ptr, dc->file_size);
    for each entry in dc->tensor_dir:
        struct ggml_tensor *t = ggml_new_tensor_4d(ctx, ggml_type,
            shape_for_ndim(entry, 0), shape_for_ndim(entry, 1),
            shape_for_ndim(entry, 2), shape_for_ndim(entry, 3));
        ggml_set_name(t, entry.name);
        t->data   = dc->mmap_ptr + entry.payload_offset;
        t->buffer = den_buf;
    return dc->header.tensor_dir_count;
}

void den_loader_unwire(DenContext *dc) {
    cudaDeviceSynchronize();
    if (dc->gpu_registered) cudaHostUnregister(dc->mmap_ptr);
    if (dc->d_vram_staged) cudaFree(dc->d_vram_staged);
    munmap(dc->mmap_ptr, dc->file_size);
    close(dc->fd);
    free(dc->tensor_dir);
    free(dc->resource_dir);
    free(dc);
}
```

## 5. GGML Type Registration

### In `ggml.h` — enum addition

```c
enum ggml_type {
    // ... existing types ...
    GGML_TYPE_NVFP4_NULLGLASS = 42,  // Den 160B NULLGLASS tiles, OMMA.SF.16864 native
    GGML_TYPE_COUNT,
};
```

### In `ggml.c` — type traits array

```c
[GGML_TYPE_NVFP4_NULLGLASS] = {
    .type_name       = "nvfp4_nullglass",
    .blck_size       = 256,             // 256 elements per tile
    .type_size       = 160,             // 160 bytes per tile
    .is_quantized    = true,
    .to_float        = nullglass_to_float,
    .from_float      = NULL,
    .from_float_ref  = NULL,
    .vec_dot         = NULL,
    .vec_dot_type    = GGML_TYPE_NVFP4_NULLGLASS,
    .nrows           = 1,
},
```

**`vec_dot_type` rationale:** Setting to the same type (not F32) prevents the CPU executor from intercepting NVFP4 tensors.

## 6. GPU Staging Policy

| Model Class | Path | Mechanism | Bandwidth |
|-------------|------|-----------|-----------|
| 4B Dense | Zero-copy | `cudaHostRegister(mmap)` -> GPU reads via PCIe BAR1 | ~25 GB/s (PCIe 4.0 x16) |
| 35B MoE | Staged DMA | `den_loader_stage_to_gpu()` -> async H2D per layer group | Peak PCIe, overlapped with compute |
| 0.8B Draft | L2 persistent | Direct VRAM allocation, no mmap | 896 GB/s GDDR7 |

## 7. Integrity Verification Tiers

| Tier | Trigger | Mechanism | Latency |
|------|---------|-----------|---------|
| Boot gate | Every `den_loader_init()` | CRC32C of header + directories | ~1 us |
| Deep verify | `--verify` flag | SHA256 of full payload | ~2s for 35B |
| Tile-level | Background audit | 1 tile per tick | Background |

## 8. Versioning Policy

**No semantic versioning. No version strings in names.**

- `format_version` in header: increments only on ABI-breaking structural change. 0 = initial.
- `abi_signature` in tensor entry: opaque 64-bit fingerprint generated from struct layout hashes, GGML_TYPE enum table, OMMA kernel contract, and CUDA compute capability target.
- `denforge_provenance` in header: human-readable converter provenance. Informational only.

---

*Specification locked 2026-05-22. All structs `static_assert` enforced. 120B tensor entry. 512B header. 4096B payload alignment.*
