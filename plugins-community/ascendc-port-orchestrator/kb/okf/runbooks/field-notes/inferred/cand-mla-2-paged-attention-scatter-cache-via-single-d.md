---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Paged-attention scatter-cache via single DataCopy per token — block-index decomposition, no per-token loop on cube side"
description: "applies_to: any soc with public AscendC DataCopy + DataCopyPad + DataCopyParams; cann=9.0.0+; op_class=kv_cache_scatter / paged_attention_writeback / block_indexed_cache_update derived-from: cann-sour"
phenomenon: build_failure
signal:
  - "Op needs to write a per-token result tensor into a paged KV cache laid out as (BlockNum, BlockSize, N, D) where each input token has an integer paTokenIndex ∈ ["
confidence: inferred
status: stub
original_id: CAND-MLA-2
timestamp_inferred: true
tags: [candidate, inferred, datacopy, col, datacopypad, cand-mla-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public AscendC DataCopy + DataCopyPad + DataCopyParams; cann=9.0.0+; op_class=kv_cache_scatter / paged_attention_writeback / block_indexed_cache_update`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/service_scatter_cache.h (ScatterCache + ScatterCacheUnAligned + ScatterCacheMultiRows + MaterializeOffsetsWithHeadSize, ~135 lines, public-API only — DataCopy, DataCopyPad, DataCopyParams), attention/mla_prolog/docs/aclnnMlaProlog.md (kvCacheRef shape `(BlockNum, BlockSize, Nkv, Hckv)` and cacheIndex semantic "取值范围需在[0,BlockNum*BlockSize)内")`
`unverified_on: a5_ops (no paged-KV op shipped)`

**Trigger**: Op needs to write a per-token result tensor into a paged KV cache laid out as `(BlockNum, BlockSize, N, D)` where each input token has an integer `paTokenIndex ∈ [0, BlockNum*BlockSize)` selecting its slot, the slot's block-id is `paTokenIndex / BlockSize`, its in-block row is `paTokenIndex % BlockSize`, and the per-row payload is `N*D` bytes (32B-aligned) or `N*D` elements (unaligned). The op may also need to scatter MULTIPLE consecutive token rows that may straddle a page boundary (last few rows of one page + first few rows of the next page). Cache layout may be ND or NZ (FRACTAL_NZ along the head dim).

**Recommendation**: Decompose the scatter into pure address arithmetic + a single `DataCopy` per token (or per row-run when consecutive), avoiding any per-element loop. The three variants follow the same address-decomposition:

  - **Aligned single-row** (`col % 32B == 0`): one `DataCopy` with no params; offset = `paTokenIndex * stride` for ND, or `(paTokenIndex/blockSize)*blockSize*stride + (paTokenIndex%blockSize)*col0` for NZ.
  - **Unaligned single-row** (`col` not 32B-aligned): `DataCopyPad` with `DataCopyParams{1, col*sizeof(T), 0, 0}`; ND only.
  - **Multi-row run** (consecutive rows, may straddle one page boundary): split into a prefix-in-current-page `DataCopy` and a tail-in-next-page `DataCopy`; the split point is computed once from `rowsInCurBatch` vs `row`.

For NZ cache layout: `col0 = ALIGN_BLOCK_SIZE / sizeof(T)` (16 elements for bf16, 32 for int8); the per-page stride uses `DataCopyParams{col/col0, 1, 0, blockSize-1}` so successive C0 chunks land in the correct NZ sub-block. This means a single `DataCopy` writes ALL of a row's `N*D` elements scattered across `col/col0` C0 sub-blocks within one page, using the destination stride to skip `blockSize-1` rows between consecutive C0 chunks.

A negative `paTokenIndex` (= sentinel for "drop this token") MUST early-return — the scatter is silently dropped, which is the correct behavior for variable-length-batch padding tokens. The caller pre-computes negative indices for padding rows.

**Concrete anchor** (verified shape; ND aligned + NZ paged + multi-row spill):
```cpp
// ND aligned variant — one DataCopy per token
if (paTokenIndex < 0) { return; }
DataCopy(cacheGm[paTokenIndex * stride], inputUb, col);

// NZ paged variant — one DataCopy per token, address decomposed to (page, row-in-page)
constexpr uint8_t col0 = 32 / sizeof(T);    // 16 for bf16, 32 for int8
int64_t pageId  = paTokenIndex / blockSize;
int64_t rowInPg = paTokenIndex % blockSize;
int64_t off = pageId * blockSize * stride + rowInPg * col0;
DataCopyParams p{ static_cast<uint16_t>(col / col0), 1, 0,
                  static_cast<uint16_t>(blockSize - 1) };
DataCopy(cacheGm[off], inputUb, p);

// Multi-row run straddling one page boundary
int64_t copyCnt = col * rowsInCurBatch;
DataCopy(cacheGm[cacheOffset], inputUb, copyCnt);
if (rowsInCurBatch != totalRows) {
    DataCopy(cacheGm[nextBatchOffset], inputUb[copyCnt],
             (totalRows - rowsInCurBatch) * col);
}
```

**Why it works**:
- Paged-attention's address decomposition is purely scalar — `pageId = idx/blockSize`, `rowInPage = idx%blockSize` — and these two divisions+modulos are free relative to the DataCopy cost. No per-element loop is needed because the page-internal stride is encoded in `DataCopyParams.dstStride` (NZ) or implicit in the contiguous offset (ND).
- The NZ variant writes one row's `N*D` elements as `col/col0` C0 chunks with `dstStride=blockSize-1`, which the MTE3 engine fuses into one descriptor — same MTE3 cost as a single contiguous DataCopy. The NZ-vs-ND choice is therefore a layout-only tradeoff with no per-token compute cost difference.
- The multi-row-spill variant is the "row-runs collapse, page-runs split" specialization — collapse consecutive same-page rows into one DataCopy, split only at page boundaries (at most one split per row-run). For typical paged-attention page sizes (16-128), the expected number of splits per token-batch is small, and the speedup vs per-token DataCopy is proportional to in-batch token locality.
- Negative-index early-return is the correct semantic for masked / padding tokens; it avoids spurious GM writes that would corrupt unused slots and avoid synthetic "drop = write to scratch then ignore" overhead.

**Determinism**: Each output cache slot has a single owning input token (by the contract of `paTokenIndex` being a one-to-one assignment); no scatter-add. Deterministic by construction when (a) `paTokenIndex` values are unique across the batch (the caller's responsibility — the CANN reference comment says "取值范围需在 [0, BlockNum*BlockSize) 内" but does not enforce uniqueness; uniqueness IS required for determinism), and (b) the DataCopy ordering across tokens does not matter because each write touches disjoint addresses.

**Other instances predicted**:
- KV-cache writeback in paged-attention prefill and decode (MLA reference; vLLM-style block tables)
- Any "scatter rows into a pre-allocated paged buffer indexed by an integer token-to-slot map" op
- Sparse-tensor-style scatter-writes where the row index is dense and pre-computed (NOT general atomic scatter-add — that needs a different pattern)
- IndexPut variants with a single integer dim-0 index (when `accumulate=False`)
- Beam-search KV reorganization where each beam's KV is scattered into a new batch-major layout
- Speculative-decoding "accepted-prefix writeback" where token-level acceptance produces a sparse-but-dense-after-compaction index map

**Risks before promotion**:
- Uniqueness of `paTokenIndex` across the batch is a precondition for determinism — if the caller computes the index map dynamically and two tokens collide, the scatter is non-deterministic. The MLA reference does not check uniqueness; a hardened caller must.
- The NZ stride encoding assumes `col % col0 == 0` (i.e. the head dim is C0-aligned). The unaligned variant exists for ND only — for NZ unaligned, the pattern does NOT apply as-is and must be padded upstream.
- `blockSize` must be in the documented range (MLA: 16-1024, multiple of 16). Smaller `blockSize` values stress the MTE3 descriptor cost; very large `blockSize` values stress UB residency of the prefix-in-page+tail-in-next-page two-DataCopy path. Profile boundary cases (page=16, page=1024) when first shipping.
- A negative `paTokenIndex` is the CANN convention for "drop"; consumers must verify their caller honors it (some PyTorch-side mappings use `-1` for padding, others use `INT64_MIN`).

**Cross-reference**:
- CAND-MLA-1 (latent-prolog skeleton) — produces the tensors that are scattered into the KV cache via this pattern (the K^C and K^R outputs)
- OL-58 (DataCopyPad for tail-byte handling) — supplies the `DataCopyPad` form used in the unaligned single-row variant
- P-P (any DataCopy-stride pattern entries in `patterns/domains/memory_access.md`) — orthogonal; this is the "writeback with paged index" specialization
- a5_ops 1_RotaryMul / 12_KvRmsnormRopeCache (closest existing benchmark ops) — these have the rope+cache shape but NOT the paged scatter; promoting this candidate provides the paged variant they currently lack

**Promote when**: an a5_ops op ships a paged-KV scatter writeback (e.g. a future MLA prolog, paged FlashAttention prefill, or a paged-cache update op), AND the per-token DataCopy cost is verified to dominate over the address-arithmetic cost on the target SoC, AND the multi-row-spill variant is exercised on a workload with non-trivial in-batch locality so the split-vs-collapse decision can be measured.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-MLA-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
