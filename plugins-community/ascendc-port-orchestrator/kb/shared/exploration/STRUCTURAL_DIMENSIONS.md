# Structural Dimensions for AscendC Kernel Optimization

Every AscendC kernel can be described by choices along 5 dimensions. An "exploration" is a change to one dimension.

## D1 — Loop Nesting Order

**What**: Which loop is outermost (token, expert, dim_tile, etc.)
**Cost**: Low (code restructure, 1 compile)

Alternatives (for 3-level loops, max 6 permutations):
```
A) token(outer) → expert → dim_tile       # per-token: current SG default
B) expert(outer) → token → dim_tile       # expert-major: load expert once
C) dim_tile(outer) → token → expert       # dim-major: complete one slice
```
Not all permutations are valid — check data dependencies before implementing.

**Key insight**: Reordering loops changes data reuse patterns. If N work items share the same GM data, making the shared-data loop outermost reduces reads by Nx.

## D2 — Work Granularity

**What**: How much work per core per dispatch
**Cost**: Low-Medium (may need buffer resize)

Alternatives:
```
A) 1 token/core (default)
B) N tokens batched/core (load shared data once, apply to N tokens)
C) 1 expert-run/core (all tokens for one expert)
D) Persistent: each core loops over many tokens (P-P22)
```

**Key insight**: Increasing granularity amortizes dispatch overhead and enables data reuse within a core.

## D3 — Buffer Strategy

**What**: TBuf vs TQue, queue depth, buffer count
**Cost**: Medium (buffer plumbing changes)

Alternatives:
```
A) TBuf × N (manual management)
B) TQue<VECIN, 2> (minimal prefetch)
C) TQue<VECIN, 4> (current best for SG)
D) TQue<VECIN, 8> (deeper prefetch, if UB allows)
E) Manual ping-pong (if TQue has bugs)
F) Double-buffer both input and output
```

**UB budget check is MANDATORY** before changing buffer strategy:
```
UB_USABLE = 192 * 1024  # 192KB on Ascend950PR
total = queue_depth × tile_size × sizeof(type) × num_queues
assert total <= UB_USABLE
```

## D4 — Synchronization

**What**: How pipes are coordinated
**Cost**: Medium (sync primitive swap)

Alternatives:
```
A) PipeBarrier<PIPE_ALL> (baseline, safe but slow)
B) TQue auto-sync (MTE2→VEC, current best)
C) SetFlag/WaitFlag fine-grained
D) TQue + SetFlag hybrid
```

**Warning**: PipeBarrier→TQue migration was the single biggest optimization in E13 (1.6-2.3x). Going backward is an anti-pattern (P-P28).

## D5 — Tiling Parameters

**What**: Tile sizes, block counts, thread counts
**Cost**: Low (constant change, recompile only)
**Note**: D5 sweeps do NOT count toward the 3-structural-change limit

Sweep ranges:
```
TILE_MAX ∈ {1024, 2048, 4096, 8192, 16384}
nblk ∈ {28, 56, 112}      # must be multiple of AI core count (56 on 950PR)
Queue depth ∈ {2, 4, 8}
BRE ∈ {emb_dim, 32, 64, 128, 256, 512}
```

## Total Search Space

- D1: ~3-4 valid permutations
- D2: ~3-4 granularity options
- D3: ~4 buffer configs
- D4: ~3 sync strategies
- D5: ~15 parameter combinations

**Worst case cross-product**: 4 × 4 × 4 × 3 × 15 = 2880. We never search the full space — grounding chains prune to 3-5 candidates.

## Structural Templates

Common kernel templates for reference during exploration:

### Template A: Per-Token Persistent (current SG forward)
```
for token in core_range:
  for dim_tile in tiles:
    accum = 0
    for expert in top_k:
      load expert[dim_tile] → local
      accum += weight * local
    store accum → output[token, dim_tile]
```

### Template B: Expert-Major Batched
```
for expert in expert_range:
  load expert_data → local (once)
  for token in tokens_using_this_expert:
    accum[token] += weight[token] * local
flush all accum → output
```

### Template C: Tiled Token×Expert
```
for token_tile in token_tiles:
  for expert in top_k:
    load expert_data → local
    for token in token_tile:
      accum[token] += weight * local
  flush accum[token_tile] → output
```
