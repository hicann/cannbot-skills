---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "arch22 → arch35 (220x → 351x) hardware delta reference — single-pointer authoritative summary [V351, hardware-arch]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-149
timestamp_inferred: true
tags: [ascendc, ol-149]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/migration/220x到351x架构变更.md (authoritative CANN team summary)`

**Rule**: When porting from V220 (A3) to V351 (A5), the hardware delta is structural — not just "newer silicon with same ISA". This entry is the single-pointer summary; refer back here from any port-related agent brief.

**Total architecture deltas** (5 categories):

### A. New / removed data paths (搬运单元)

| Path | A3 (220x) | A5 (351x) | Migration impact |
|---|---|---|---|
| L1 Buffer → GM | ✓ | ✗ **REMOVED** | Use L1→L0C→Fixpipe→GM OR L1→UB→GM (vector+cube fusion). `DataCopy/DumpTensor` to GM from L1 = compile error |
| GM → L0A Buffer | ✓ | ✗ **REMOVED** | Split: GM→L1→L0A. `LoadData` no longer goes direct |
| GM → L0B Buffer | ✓ | ✗ **REMOVED** | Same — split via L1 |
| UB → L1 Buffer | ✗ | ✓ **NEW** | Direct UB→L1 supported via `DataCopy` (avoid UB→GM→L1 trip) |
| L0C → UB | ✗ | ✓ **NEW** | Single-direction, via `Fixpipe` |
| ND-DMA | ✗ | ✓ **NEW** | Extended `DataCopy` with rich dim+stride config — "多维数据搬运（ISASI）" |
| L0A/L0B init | hardware-direct | ✗ **REMOVED** | `InitConstValue` no longer initializes L0A/L0B directly |
| MicroScaling LoadData | ✗ | ✓ **NEW** | See OL-145 — fp8_e8m0_t scale matrix path |
| L1→L0A transpose | ✓ ZZ→ZN | ✓ DN format only | `LoadDataWithTranspose` semantics changed |

### B. Compute unit changes (计算单元)

| Change | Impact |
|---|---|
| Cube no longer supports `s4` (int4b_t) | Cast int4→int8 before Mmad |
| Cube no longer does L0A ZZ→ZN transpose | Use DN format + recompute L0A addr |
| Vector Core: Membase → **Regbase** architecture | Memory-based VEC APIs slower; Register-based MicroAPI is the fast path (see OL-152 in Batch 3) |
| No native Subnormal support | `LnConfig{algo}` opt-in for software path (OL-147) |
| No 4:2 sparse matmul | Vector-side dense→sparse helper required (rarely used) |

### C. Storage / UB changes (存储单元)

| Aspect | A3 (220x) | A5 (351x) |
|---|---|---|
| UB total | similar capacity | similar |
| UB bank structure | 16 bank groups × 3 banks × 4KB | **8 bank groups × 2 banks × 16KB** |
| L1 Buffer boundary check (`SetLoadDataBoundary`) | ✓ | ✗ **REMOVED** |
| SSBuffer (per-core scalar-accessible storage) | ✗ | ✓ **NEW** — AIC + AIV both via Scalar |

**Bank-conflict implication**: A3 optimization patterns assuming 16 bank groups × 4KB bank size **DO NOT transfer to A5**. A5's wider banks (16KB) reduce per-bank cycle pressure but coarser stride causes different conflict patterns. Avoidance pattern published in CANN docs `算子实践参考/SIMD算子性能优化/内存访问/避免UB的bank冲突/`.

### D. Sync changes (同步)

| Feature | A3 | A5 |
|---|---|---|
| **Mutex** (intra-core async pipe sync, CPU-lock-like) | ✗ | ✓ **NEW** |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | full pairing | **AIV0/AIV1 single-trigger AIC-wait** new mode |
| `LocalMemBar<MemType::UB>` for L2 MicroAPI | ✗ | ✓ used to replace `SetFlag<HardEvent::MTE2_V>+WaitFlag` (see Batch 3 P-P95) |

### E. Other (其它)

| Change | Impact |
|---|---|
| AIPP hardware instruction | ✗ **REMOVED** — software emulated; `SetAippFunctions`/`LoadImageToLocal` slower on A5 |
| UB debug helpers (`CheckLocalMemoryIA`) | ✗ **REMOVED** — registers gone |
| SIMT mode (up to 2048 threads, direct GM access) | ✗ | ✓ **NEW** — see OL-150 in Batch 3 |

### Newly supported data types

`fp8_e4m3fn_t`, `fp8_e5m2_t`, `hifloat8_t`, `fp4x2_e1m2_t`, `fp4x2_e2m1_t`, `fp8_e8m0_t` (scale).

See OL-144 for the type family; OL-145 for MicroScaling path; OL-146 for CastTrait constants.

### Compatibility policy by API tier

| API tier | Cross-arch policy |
|---|---|
| **High-level (高阶) APIs** | ALL cross-arch compatible (vendor primitives like `LayerNorm`, `SoftMax`, `RmsNorm` work on both) |
| **Basic (基础) APIs — compatible subset** | Cross-arch compatible (`Add`, `Mul`, `DataCopy` basic forms) |
| **Basic (基础) APIs — ISASI subset** | NOT compatible (arch-specific — cube `LoadData`, `Mmad` etc.) |
| **Framework APIs** | Cross-arch compatible (software-implemented) |
| **Compiler BuiltIn APIs** | NOT guaranteed compatible |

**Why this matters for our port_a3_to_a5 work**:
- ISASI APIs in A3 source code are exactly the substitution targets — every ISASI call needs review.
- High-level vendor primitives (`AscendC::LayerNorm<T,T>`) are **safe to keep as-is** during L1 mechanical port (this is why ada_layer_norm's vendor-primitive fallback "worked" precision-wise even though perf was 0.38× — vendor primitive is cross-arch correct, just slow vs L2 MicroAPI rewrite).

**Evidence**: PR 103 imports the authoritative CANN doc `migration/220x到351x架构变更.md` verbatim; PR 103 SKILL.md §266-288 + l2-guide §566-588 cross-reference it as the architecture-delta source-of-truth.

**Other instances (predicted)**: every port_a3_to_a5 op-gen, every greenfield A5 kernel (since "what's available on A5 vs A3" is one query).

**Cross-reference**:
- OL-141 (skip-if-upstream-present) — gates whether the port even runs
- OL-142 (NPU_ARCH macros) — for conditional code per arch
- OL-143 (L1/L2/L3 classifier) — uses delta info to decide tier
- OL-144..148 (narrow-float + MicroScaling + Subnormal + SPR) — specific A5-only features in detail
- `kb/okf/runbooks/hardware/target-ascend950pr.md` — existing hardware spec file; should link this OL

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-149（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
