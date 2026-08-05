# Hardware Reference — Internal-Materials Query Queue

> Gaps identified during 2026-04-21 public manual scan (卷 2 REDACTED_INTERNAL_DOC §25 AICore SubSys + table 25-1). These questions could NOT be answered from the public `DavidV100 用户手册 第一卷 功能描述 分卷*.pdf` and require the internal-only ISA manuals / TRM / AscendC software manual, typically scanned via opencode / codex expert agents with access to the REDACTED_INTERNAL / REDACTED_INTERNAL.

Ping the internal agent with one of the prompt templates below when pursuing the respective optimization.

---

## Queue

### IQ-1 [PARTIAL — public doc answered bank conflict rules; bank count/width still unknown] UB bank structure

**Purpose**: op#11 class UB-scalar-bound kernels may be bank-conflict limited. Need bank layout + conflict rules to tune stride patterns.

**Prompt template for internal agent**:
```
搜索内部 DavidV100 / Ascend950PR 资料（ISA manuals, TRM, REDACTED_INTERNAL / REDACTED_INTERNAL）
查找 AIV 的 Unified Buffer (UB, 256KB) 内部架构细节:

1. UB 是否 multi-bank 结构？bank 数量和每 bank 宽度？
2. Bank-conflict rules: 同 cycle 多路访问冲突的 cost / resolution strategy
3. Alignment sweet-spot: 当前 KB 已记录的阈值是 32B / 512B / 16KB (OL-46)。
   是否有 bank-stride 建议值 (e.g. 64B / 128B / 256B 硬件推荐 stride) 可消除 conflict?
4. DataCopyPad 多路并行的 bank 访问模式: 2 concurrent DataCopyPad 是否一定
   用不同 bank 才能并行？
5. VEC `Muls`/`Mul` 等 load-compute-store 指令的 UB 访问模式：是单 bank 顺序
   还是跨 bank 交织？

参考章节:
- 卷 2 REDACTED_INTERNAL_DOC §25 AICore SubSys (AIV 描述)
- §27 Memory 子系统
- DaVinCi AIC V310 ISA User Guide for David VecCore (卷 2 §25.2.2 引用)
- DaVinCi AIC V310 ISA User Guide for Vector Thread SIMD Extension

输出格式: Answer / Details / KB Entry (for
`src/skills/references/hardware/target/ascend950pr.md` §Memory) / Confidence
```

**Would unlock**: op#11 and similar scalar-bound ops might gain from bank-aware stride padding. Potential perf improvement: unquantified until data available.

---

### IQ-2 [RESOLVED 2026-04-21 via hiascend.com AscendC API ref — no internal agent needed]

Answered on the PUBLIC AscendC API ref tree (`/API/ascendcopapi/`, not `/opdevg/`):
- WholeReduceMax / Min / Sum — page 07_0079-0081
- MrgSort — page 07_0232
- BlockReduce — page 07_0082+

Key specs now in `target/ascend950pr.md` §"Sort / Reduce VEC primitive specs":
half/float only (A5); WholeReduce per-iter cap 128(16b)/64(32b); MrgSort 4-way merge,
8B (score,index), max 4095/queue; ReduceOrder modes ORDER_VALUE_INDEX / INDEX_VALUE /
ONLY_VALUE / ONLY_INDEX all supported.

*(Historical template preserved for archival):*

### IQ-2-HISTORICAL MrgSort / WholeReduce hardware primitive specifics (Q7 from 2026-04-21 scan)

**Purpose**: Sort + reduction primitives drive perf of topk / softmax / sort-family ops (op#7, op#9, op#26 historical). Current KB has qualitative pattern notes but lacks quantitative specs.

**Prompt template for internal agent**:
```
搜索内部 DavidV100 / Ascend950PR 资料（ISA User Guide for VecCore + SIMD/SIMT Extension）
查找 VEC sort / reduction 硬件原语细节:

1. MrgSort 硬件指令:
   - max lane width per-row (element count)?
   - 支持 dtype: fp16 / fp32 / bf16 / int16 / int32?
   - Proposal format: KB 中 §patterns/domains/sort.md 提到但未详述 layout
   - Latency / throughput (cycles per merge)?
2. WholeReduceMax / WholeReduceSum:
   - 树形归约深度 (log2(N) 或固定?)
   - 每 AIV 一次 reduce 可吃的 max 元素数 (8? 64? 256?)
   - 支持 index-tracking 版本 (ReduceMax with calcIndex=true) 的 cost?
3. BlockReduce:
   - block 定义是多少 lane (16? 32? 256?)
   - 支持的 reduction 类型 (sum / max / min / prod)
4. Concat + Sort + Extract pattern (KB §sort.md 引用的) 对应的硬件指令序列
5. Bitonic sort vs radix sort 硬件支持 (KB 提到两种都是候选)

参考章节:
- DaVinCi AIC V310 ISA User Guide for David VecCore
- DaVinCi AIC V310 ISA User Guide for Vector Thread SIMD Extension
- 卷 2 REDACTED_INTERNAL_DOC §25.2.2 (ISA 引用)

输出格式: Answer / Details / KB Entry (for
`src/skills/references/target/ascendc/patterns/domains/sort.md` + `reduction_quant.md`) /
Confidence
```

**Would unlock**: Better sort/reduce op perf tuning, informed algorithm choice.

---

### IQ-3 [RESOLVED 2026-04-21 via hiascend.com AscendC API ref]

Answer: **syntactically valid** per public docs.
- `TPosition` (API ref 07_0174): enum includes A1/A2/B1/B2 among others
- `TBuf` (API ref 07_0161): "存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置" — any TPosition is accepted
- `TBufPool`: "管理 Unified Buffer / L1 Buffer 物理内存，主要用于多个 stage 计算中 ... 物理内存不足的场景" — L1 explicitly listed as a TBufPool-managed resource for UB-shortage scenarios

Unverified empirically: bisheng behavior for `TBuf<TPosition::A1>` in a kernel that
contains no Cube ops (silently accept vs warn vs error) — recommend a minimal test
kernel before committing a production op to this path.

KB updated at `target/ascend950pr.md` §"AIC L1 共享空间" under "Language-level syntactically valid".

*(Historical template preserved):*

### IQ-3-HISTORICAL AscendC language-level allowance of L1 as AIV scratch (follow-up to Q1)

**Purpose**: Public manual confirms hardware UB↔L1 DMA path. But whether AscendC language syntax lets a pure-AIV kernel declare `TPosition::A1` / `B1` tensors **as general scratch** (not matmul operands) is not documented publicly.

**Prompt template for internal agent**:
```
搜索内部 AscendC 文档 (语言手册 / API reference / REDACTED_INTERNAL kernel 源码):

1. 纯 AIV kernel (kernel function only uses VEC/SIMD/SIMT, no Cube op)
   是否可以声明 `TBuf<TPosition::A1>` 或类似 A1/B1 position 的张量作为普通 scratch?
2. 若可以，DataCopy / DataCopyPad 在 UB (VECIN/VECOUT/VECCALC) 和 A1/B1
   之间的 API 调用形式?
3. 若不可以，有没有其他 TPosition 允许 AIV 访问 L1 空间 (e.g.
   某个未公开的 TPosition::L1_SCRATCH)?
4. bisheng 编译器对 pure-AIV kernel 里出现 A1/B1 tensor 的 enforcement 行为
   (silently accept? warn? error?)
5. 现有REDACTED_INTERNAL算子 (ops-transformer / catlass / opbase) 里有没有已知的
   "AIV only, 但借用 L1" 的 pattern 案例?

参考:
- AscendC 语言官方文档 (hiascend.com CANN 9.0 docs)
- CANN 源码 `ops-transformer`, `opbase`, `catlass` (内部)
- 华为 FlashAttention 等 mixed AIC+AIV 算子作为参考对照

输出格式: Answer / Details / KB Entry (for
`src/skills/references/hardware/target/ascend950pr.md` §AIC L1 共享空间 —
update "语言层合法性" section) / Confidence
```

**Would unlock**: Conditional — if yes, direct applicability to op#11 Kind-2 rewrite + any future fused op hitting UB budget.

---

## Resolved queries

### 2026-04-21 — IQ-2 (Sort/Reduce) resolved via hiascend.com AscendC API ref
KB delta: `target/ascend950pr.md` §"Sort / Reduce VEC primitive specs" — WholeReduceMax dtypes/limits/ReduceOrder, MrgSort specs.

### 2026-04-21 — IQ-3 (L1 as AIV scratch, language layer) resolved via hiascend.com AscendC API ref
KB delta: `target/ascend950pr.md` §"AIC L1 共享空间" — upgraded "语言层合法性" from "unverified" to "syntactically valid per TPosition + TBuf + TBufPool public API pages". Empirical bisheng compiler test still recommended.

### 2026-04-21 — IQ-1 (UB bank conflict rules) partially resolved via hiascend.com 351x page
KB delta: `target/ascend950pr.md` §"UB bank 结构与 bank-conflict 规则" — 2R+2W per bank group, 2R0W or 1R1W. **Still open**: bank count + per-bank width (shown in page figure only, no text).
