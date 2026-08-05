---
applies_to: soc=all
reason: Block/grid count tuning patterns are universal — but the optimal numBlocks depends on AIV count which differs by chip (56 on a5, 80-96 on a3, 40-48 on a2). Patterns must derive count from `GetBlockNum()` runtime, not hardcode. SIMT-specific thread-count tuning (threadIdx.x bands) is `a5-only` — flag at per-pattern level.
---

# Domain: Thread Utilization & Scheduling
> Patterns for block/thread count tuning, work distribution, and kernel scheduling.
> Load when: Analyzer detects numBlocks assignment, blockDim tuning, or token_num >> core count.

---

## Patterns

### P-P1: Dynamic numBlocks

**Severity**: High

```cpp
// Pooling: saturate all AIV cores
constexpr uint32_t MAX_AIV_CORES = 56;  // 28 AICore x 2 AIV

// SG: each block handles one token
uint32_t fwd_blk = token_num * grid_y;
```

**Distinction**: Pooling shares work by stride -> saturate all cores. SG has each block own one output slice -> numBlocks = number of work items.

---

### P-P4: Dynamic block size

**Severity**: Medium

```cpp
__aicore__ inline uint32_t calc_block_size(int dim, int divisor) {
    int raw = std::min(dim / divisor, 1024);
    return ((std::max(raw, 1) + 31) / 32) * 32;  // round to warp
}
```

**Note**: The function must be annotated with `__aicore__`.

---

### P-P20: Thread utilization rebalancing (Thread Utilization Analysis)

**Severity**: **High** | **Source**: Expert manual optimization E7-1 (2026-03-27) | **Platform**: all AscendC platforms
**General strategy**: [SKILLS_DESIGN.md §6.2](../../../../../docs/design/SKILLS_DESIGN.md) — can be auto-discovered by static analysis; no msprof needed

**Core principle**: In multi-dimensional work decomposition, the number of threads assigned to each dimension should match the actual work. Rebalance when `utilization = min(actual_work, threads) / threads < 50%`.

**Concrete instance (Pooling BRE=emb_dim)**: Set BLOCK_READ_EMB to emb_dim (instead of a fixed 32 or 512) so that iter_emb=1 and the number of index threads is maximized.

**Anti-pattern** (fixed BRE=32):
```cpp
// At dim=9: 512/32=16 index threads; 23/32 emb threads wasted
constexpr int BRE = 32;
// At dim=1: 16 index threads handle 16 edge/tile (severely inefficient)
```

**Correct pattern** (BRE=emb_dim, runtime parameter):
```cpp
int BRE = (emb_dim <= 512) ? static_cast<int>(emb_dim) : 512;
int block_read_indices = THREAD_NUM / BRE;
// At dim=9: 512/9=56 index threads (3.5x more edge/tile)
// At dim=1: 512/1=512 index threads (32x more edge/tile)
```

**A5 measured effect** (61 clusters, fp32, sorted edges):

| dim range | BRE=32 Fwd | BRE=dim Fwd | Speedup |
|---------|:---:|:---:|:---:|
| dim=1~9 (clusters 0-6) | 0.64ms | 0.39ms | **1.65x** |
| dim=33 (clusters 13-16) | 16.51ms | 11.38ms | **1.45x** |
| dim>256 (clusters 52+) | roughly even | roughly even | 1.0x |
| **Total** | 15.43ms | 10.70ms | **1.44x** |

**Notes**:
- BRE=emb_dim requires turning the template parameter into a runtime parameter (cannot instantiate a template for every dim value).
- **Replaces P-P11's BRE selection strategy**: dim <= 512 uses BRE=emb_dim; dim > 512 uses BRE=512.
- Compatible with sorted-edge register accumulation: when iter_emb=1, accum degenerates to a single scalar.

**Migration-time trigger rule (generator MUST check)**:
- For arch22→arch35 migration, check every fixed work-decomposition constant against
  the target AIV count, UB budget, and runtime shape.
- A fixed `BRE=32` wastes work on small embedding dimensions; derive it from
  `emb_dim` and emit runtime dispatch when the shape is dynamic.

#### Runtime-BRE Variant (_rt_vf) implementation pattern

When BRE=emb_dim, the BRE value is unknown at compile time (different callers use different emb_dim). A separate `_rt_vf` variant must be generated:

**Naming convention**:
```
pool_{op}_{dir}_kernel_vf<T, BRE, TI>        — template BRE/TI (compile-time)
pool_{op}_{dir}_sorted_kernel_vf<T, BRE, TI> — sorted variant (template BRE/TI)
pool_{op}_{dir}_sorted_rt_vf<T>              — sorted + runtime BRE/TI
```

**_rt_vf function signature** (BRE/TI + pre-computed params all passed as runtime parameters):
```cpp
template <typename DATA_TYPE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void pool_{op}_{dir}_sorted_rt_vf(
    GM_ADDR ...,
    int BRE, int TI, int block_read_indices,
    int iter_indices_block, int iter_indices_thread, int iter_emb,
    uint32_t block_index, uint32_t total_block_num) {
  // BRE/TI etc. are pre-computed in the host dispatcher and passed in
  // Combine with P-P21 sorted so that iter_emb=1 degenerates to a single scalar accum
}
```

**Host dispatcher pre-computation** (in `{op}_launch_config.h`):
```cpp
inline void compute_pooling_params(int emb_dim, int thread_num,
    int& BRE, int& TI, int& bri, int& iib, int& iit, int& ie) {
  BRE = (emb_dim <= 512) ? emb_dim : 512;
  bri = thread_num / BRE;  // block_read_indices
  TI = bri;                // tile indices = index threads per block
  iib = (work_items + 56 * bri - 1) / (56 * bri);  // iter per block
  iit = 1;                                           // iter per thread (simplified)
  ie = (emb_dim + BRE - 1) / BRE;                   // iter_emb
}
```

**When to use template vs runtime**:
- dim is known and fixed (e.g., SG hidden_dim=256) -> template BRE/TI + `#pragma unroll`
- dim varies at runtime (e.g., Pooling 61 clusters dim 1~512) -> runtime `_rt_vf`
- **Generate both**: template for benchmark/specialization, runtime for production dispatch.

---

### P-P22: Persistent-core dispatch (Persistent Kernel)

**Severity**: High | **Source**: Expert E8-2 (2026-03-28) + A5 measurement | **Platform**: Ascend950PR (56 AIV cores)

**Anti-pattern**: `numBlocks = token_num` (with many tokens, most blocks queue up time-slicing the 56 cores)
```cpp
numBlocks = token_num;  // e.g. 4096 blocks -> 56 cores queue 73 rounds
for (int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
    // handle 1 token
}
```

**Correct pattern**: 56 persistent blocks, each iterating over multiple tokens
```cpp
numBlocks = 56;  // MAX_AIV_CORES
for (uint32_t token = block_index; token < token_num; token += total_block_num) {
    // handle 1 token (same as original logic)
}
```

**Effect**: Eliminates block-scheduling overhead. medium (512 tokens) 3.2x speedup; xlarge (4096 tokens) 1.86x speedup.

**Applicability**:
- token_num >> MAX_AIV_CORES (56), with modest per-token work (scheduling overhead dominates)
- Forward-type kernels (no atomicAdd write conflicts)
- msprof scalar_ratio > 0.2 (significant indirect-addressing/scheduling-related code -> reducing scheduling is effective)
- **Does not apply** to backward (msprof vec_ratio ~= 1.0, compute-bound; scheduling overhead is not the bottleneck)

**Trigger condition (generator MUST check)**:
- `numBlocks = token_num` seen with token_num possibly >> 56 -> suggest generating a persistent variant
- msprof scalar_ratio > 0.2 -> persistent may be effective
