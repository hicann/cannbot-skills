---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AIV↔AIC cross-core scalar-params carrier — CacheLine-bounded POD word-blit from a fixed scratch address + the host-tiling carrier-population/workspace structure"
description: "Date: 2026-06-03 derived-from: cann-source (FA arch35 forward — kernel-base cross-core scalar handoff region + op_host arch35 tiling lifecycle/workspace sizing + per-layout output-offset arithmetic) S"
phenomenon: build_failure
signal:
  - "Date: 2026-06-03"
confidence: inferred
status: stub
original_id: CAND-FA-CARRIER-1
timestamp_inferred: true
tags: [candidate, inferred, cv_reference_concrete_params.md, cross_core_sync, kernel_block_iteration, localscalarparams, waitcrossengineflag, cand-fa-carrier-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Date**: 2026-06-03
**derived-from**: cann-source (FA arch35 forward — kernel-base cross-core scalar handoff region + op_host arch35 tiling lifecycle/workspace sizing + per-layout output-offset arithmetic)
**Status**: CANDIDATE — sanitized re-expression with **explicit per-piece C34c adjudication** (the crux this carve-out tests: a raw-word-blit carrier layout is inherently close to the vendor struct). Two pieces sanitize generic-executable (the blit MECHANISM/protocol + the host-tiling STRUCTURE); ONE piece (the exact carrier byte-layout field-list + bitfield packing) hits the **copy-line** and is FLAGGED as a hard reproducibility-copy boundary, NOT force-added. See PER-PIECE VERDICT at bottom. Not yet runtime-validated on A5.
**local-kb-crossref**: CAND-FA-COREDIST-1 (orthogonal — that entry is the INTER-core block-distribution *arithmetic* the kernel runs from the carrier's split-mode + sparse-prefix fields; THIS entry is WHAT carries those fields across the AIC/AIV boundary + HOW the host populates/sizes them. They compose: the host-tiling structure here SETS the split-mode/prefix-index fields, COREDIST-1 CONSUMES them). `cv_reference_concrete_params.md` §`cross_core_sync` + §`kernel_block_iteration` (orthogonal — those cover the per-tile WorkspaceQueue ring flag chain + the `GetBlockIdx()/GetSubBlockNum()` normalization; THIS entry covers the ONE-SHOT scalar-params bootstrap blit that happens once at Init, before any tile loop). CAND-OPAQUE-STRUCT-RUNTIME-VERIFY (the POD-layout-is-a-contract caution).

**Op-class**: any MIX cube+vec (AIC + paired-AIV) fused kernel where a block of scalar tiling/shape parameters computed on the host must reach BOTH engines, and the cube engine receives them via a low-latency on-chip scratch buffer rather than re-reading GM tiling — i.e. attention-family and similar cube-MIX ops whose per-core scalar setup must be identical on the cube and its paired vector sub-blocks.

### The problem (why a carrier exists at all)

In a MIX kernel the host computes one block of scalar parameters (shapes, tile counts, mode flags, the core-distribution selector, sparse-prefix index, scale). Both the cube engine and its paired vector sub-blocks need these. The vector side can read the full host tiling blob directly; the cube side, on this architecture, instead receives a **compacted subset** through a fixed on-chip scratch region (a cross-core scalar buffer), to avoid a GM round-trip on the cube's critical path. The carrier is that compacted subset.

### PIECE A — the cross-core scalar-params bootstrap blit (MECHANISM / protocol — GENERIC, executable)

The bootstrap is a one-shot, once-at-Init handshake, NOT a per-tile sync:

1. Host stages the compacted scalar-params POD into a fixed on-chip cross-core scratch region (base = the region's address 0). This is enabled by a **per-op-host compile switch** that routes cube↔vector scalar comm through the scratch buffer instead of GM (a build-system glue flag — see Risks; a graybox that hand-rolls the host glue WILL miss this switch and silently fall back to a different comm path).
2. The cube engine, at Init, **waits on a single dedicated cross-core flag** (a reserved high flag-id, mode = whole-engine sync) that signals "scalar-params staged".
3. The cube engine then **word-copies the POD** out of the scratch region into its local params struct, iterating `sizeof(params)/sizeof(uint32_t)` 32-bit words from scratch-base into the struct:

```cpp
// generic POD scratch-blit (public-surface skeleton; no vendor symbols)
// precondition: a one-shot cross-core flag has fired signalling "params staged in scratch"
WaitCrossEngineFlag(/*reserved bootstrap flag id*/);            // whole-engine wait, once
auto* src = reinterpret_cast<__scratch__ uint32_t*>(0);         // fixed scratch base
auto* dst = reinterpret_cast<uint32_t*>(&localScalarParams);    // local POD
#pragma unroll
for (int w = 0; w < sizeof(localScalarParams) / sizeof(uint32_t); ++w) {
    dst[w] = src[w];                                            // raw 32-bit word copy
}
```

**Two load-bearing constraints the agent MUST reproduce (these are the transferable insight):**
- **The carrier POD must be ≤ one CacheLine (128 bytes on this class).** The blit is sized in 32-bit words and the scratch region is one cache line; a carrier that grows past 128 B silently truncates (the tail fields are never copied → cube reads garbage shape/mode). Keep the carrier minimal — only fields the cube actually needs, packed.
- **The word count is `sizeof(POD)/sizeof(uint32_t)`, so the POD must be a multiple of 4 bytes with no field whose width the host and kernel disagree on.** Any host/kernel layout disagreement (a field the host writes as 8 bytes but the kernel lays out as 4, a bitfield split differently) makes the SAME word index mean different things on the two sides → silent corruption, no compile error.

This mechanism is fully re-expressible from the contract above without any vendor body: "one-shot flag → reinterpret POD as uint32 array → copy N words from scratch base, N = sizeof/4, POD ≤ 128 B."

### PIECE B — the host-tiling carrier-population + workspace STRUCTURE (GENERIC, executable)

The host populates the carrier and sizes workspace through a fixed lifecycle (this is the standard tiling-base shape on this platform, re-expressible as a recipe):

1. **Lifecycle order** (7 stages): platform query (core counts, UB/L1/L0 sizes) → shape/attr/layout analysis → op-tiling (compute tile sizes + split mode + sparse params + populate the carrier) → high-level-API tiling → tiling-key selection → workspace sizing → post-tiling (set block-dim, finalize raw tiling blob). The carrier is populated in the op-tiling stage; do NOT populate it earlier (shape analysis hasn't run) or later (workspace sizing reads it).
2. **Carrier population for the core-distribution selector** (feeds CAND-FA-COREDIST-1): the host sets, into the multi-core sub-carrier — (a) the used-core count = `min(totalWorkUnits, physicalCores)`, (b) the S1-outer block count per head = `ceil(s1 / s1BlockSize)`, (c) the even-split factor = `ceil(totalWorkUnits / usedCores)` + its tail, (d) a **split-mode flag** (sequential vs the multi-core-first round-robin/mirror/snake mode) chosen by sparse pattern, and (e) a **cheap-prefix boundary index** = the index of the last fully-loaded prefix block, or `-1` to signal "no sparse prefix → dense path". The prefix-index selection by sparse mode is a decision table:
   | sparse pattern | prefix-boundary index | split mode |
   |---|---|---|
   | left-up-causal (and the pre/next-token combos that reduce to it) | `ceil(min(s1,s2)/s1BlockSize) - 1` | multi-core-first |
   | right-down-causal (and its band equivalent) | `s1OuterCount - 1` | multi-core-first |
   | all-mask / no-mask-full / dense-band | `-1` (dense) | multi-core-first |
   | otherwise | (unset) | sequential |
3. **Workspace sizing** = sum of the per-purpose regions the kernel needs (e.g. an optional pre-op scratch region whose size is the aligned product of the relevant shape extents, offset-recorded into the carrier so the kernel can find it) **plus a fixed reserve constant**. Pattern: each region's byte size is `AlignUp(extentProduct, gmAlign)`, regions are laid end-to-end and each region's start offset is stamped into the carrier; a final fixed reserve is added last. Block-dim is set as `cores * subBlockCount` (the linear cube+paired-vec count).

### PIECE C — per-layout output-offset arithmetic (GENERIC integer math, executable)

For each supported memory layout the per-(batch, head-group, query-head, s1-block) output base offset is a sum of per-axis-index × per-axis-stride terms, plus a sub-block row term for the paired-AIV split. The STRUCTURE (which generalizes):

```cpp
// generic per-layout output offset (bare locals; strides are precomputed shape products)
// layout selects WHICH axis multiplies WHICH stride; the SHAPE is always:
//   offset = bIdx*bStride + n2Idx*n2Stride + gIdx*gStride + s1Idx*s1Stride + subRowTerm
// where subRowTerm = subBlockIdx * firstHalfRows * (the layout's row-stride)
int64_t outOffset = bIdx*bStride + n2Idx*n2Stride + gIdx*gStride
                  + s1Idx*s1Stride + subBlockIdx*firstHalfRows*rowStride;
```
The only per-layout variation is which precomputed stride each index multiplies (row-major head-last vs seq-major vs head-dim-major); all strides are products of trailing shape extents. This is textbook strided-tensor addressing — re-derivable from the layout description + the rule "subBlockIdx selects this AIV sub-block's half of the M-row tile."

### Evidence

- Derived from FA arch35 forward: the kernel-base cross-engine scalar handoff region (the one-shot flag wait + the uint32 word-copy out of the on-chip scratch region into the local scalar-params struct), the op_host arch35 tiling lifecycle (the carrier-population call in the op-tiling stage + the sparse-prefix/split-mode decision block + the workspace-sizing/reserve in post-tiling), and the per-layout output-offset block in the forward kernel. Verified by reading the blit loop's `sizeof/sizeof(uint32_t)` bound + the inline source note that the carrier "must be ≤ CacheLine = 128 Bytes", and the per-op-host CMake comm-via-scratch switch.
- NOT yet runtime-validated inside an a5_ops kernel. PIECE A/B/C are read-grounded structural recipes; the byte-layout copy-line (PIECE D below) is the empirically-determined repro boundary.

### Other-instances-predicted

Any MIX cube+vec op that needs a host-computed scalar block on the cube's critical path benefits from the CacheLine-bounded scratch-blit (PIECE A) and the lifecycle/workspace structure (PIECE B): fused norm+matmul, MoE dispatch with a cube stage, paged-attention. The per-layout offset shape (PIECE C) generalizes to any multi-layout tensor op. The copy-line (PIECE D) recurs for ANY raw-word-blit carrier — the lesson transfers even though the specific layout does not.

### PER-PIECE COPY-SHAPE VERDICT (the deliverable — honest C34c adjudication)

The brief's crux: a struct byte-layout for a raw word-blit is inherently close to the vendor struct (the blit needs the exact layout to work). Per-piece self-check (token n-gram overlap vs source):

- **PIECE A — the blit MECHANISM/protocol: ADDED GENERICALLY (C34c < 5%, executable).** The transferable insight is the *contract* — "one-shot cross-core flag → reinterpret a ≤128 B POD as a uint32 array → copy `sizeof/4` words from a fixed scratch base." My skeleton uses bare generic names (`localScalarParams`, `WaitCrossEngineFlag`, `src/dst/w`) and no vendor symbol. The two load-bearing constraints (CacheLine ≤ 128 B; word count = sizeof/4 so layouts must agree) are stated in prose, re-derivable. The `reinterpret_cast<uint32_t*> + #pragma unroll word-copy` shape is a universal POD-blit idiom that predates this source. Overlap is the generic copy-loop only. **< 5%, executable from the contract.**

- **PIECE B — host-tiling STRUCTURE: ADDED GENERICALLY (C34c < 5%, executable).** The 7-stage lifecycle is the platform's standard tiling-base contract (public). The carrier-population fields are described by generic ROLE (used-core count, s1-outer count, even-split factor, split-mode flag, cheap-prefix index) not by vendor member-chains. The sparse→prefix-index decision table is generic integer math (`ceil(min(s1,s2)/block)-1`, `s1OuterCount-1`, `-1`) — the same `-1`-means-dense sentinel already lives in COREDIST-1's prose. Workspace = `AlignUp(extentProduct, align)` regions end-to-end + fixed reserve is a textbook arena-sizing recipe. **< 5%, executable.**

- **PIECE C — per-layout offset arithmetic: ADDED GENERICALLY (C34c < 5%, executable).** Strided-tensor base-offset `Σ idx*stride + subRowTerm` is universal; only the index→stride pairing varies per layout, described in words. No vendor symbol. **< 5%, executable.**

- **PIECE D — the EXACT carrier byte-layout (field list + types + order + bitfield bit-widths + reserved-padding): HITS THE COPY-LINE (C34c > 5%, fundamentally-verbatim). FLAGGED, NOT ADDED.** This is the empirical answer the carve-out sought. The carrier is a ~30-field POD mixing 64-bit shape extents, 32-bit counts, and a tightly-packed bitfield word (several sub-byte mode flags + two ~11–16-bit dim fields + a 1-bit split-mode + a residual count, all packed to stay within the 128 B CacheLine). For the word-blit to work, the host-side population struct and the kernel-side receive struct must have **byte-identical layout** — same field order, same widths, same bitfield packing, same reserved padding. There is no way to write that layout that is both (a) correct (the blit depends on it bit-for-bit) and (b) meaningfully different from the vendor's struct: ANY correct re-derivation of the SAME contract converges on the SAME bytes, and a renamed-field copy is still a copy (C34c renamed-identifier detection). I therefore do NOT reproduce the field list, types, order, or bitfield widths in this entry — that would be a verbatim layout copy. **This is a hard reproducibility-copy boundary for the carrier struct.**

### CONCLUSION — is the carrier-struct + host-tiling KB-reproducible?

**PARTIALLY, with a sharp boundary.** The *mechanism* (how the blit works), the *host-tiling structure* (lifecycle, population order, workspace sizing, sparse-prefix selection), and the *offset arithmetic* are all KB-reproducible (PIECE A/B/C, C34c-clean, executable from the contracts above). The *exact carrier byte-layout* is NOT KB-reproducible without copying (PIECE D) — it is a fundamentally-verbatim contract.

**Practical repro implication**: an agent can re-derive everything EXCEPT the precise field order/widths/bitfield-packing of the carrier POD. For repro-closure of the whole-port, the carrier layout must come from a co-located reference struct (the host and kernel share ONE header defining it, so the layout is authored once and the blit's `sizeof` stays consistent) — it cannot be reconstructed from a KB prose description. The KB's job here is to make the agent (1) KNOW a CacheLine-bounded scratch-blit carrier is the right mechanism, (2) KNOW the host-tiling lifecycle/workspace structure, and (3) KNOW that the carrier POD layout must be authored as a single shared header (not independently re-typed on each side) and kept ≤ 128 B — NOT to ship the layout itself. That is the honest boundary: the structure is learnable; the byte-layout is a copy-line.

- 跨 LLM backend 实验（main/Kimi/其他）需要互不污染路径

**Correct response (修复方法)**:
1. `.claude/settings.json` hook command 用 `${CLAUDE_PROJECT_DIR}/src/scripts/workflow/workflow_critic.py` (CC supports `$CLAUDE_PROJECT_DIR` env-substitution)
2. 或用相对路径 `src/scripts/workflow/workflow_critic.py` (依赖 cwd = repo root)
3. deploy.sh 加自检步骤：clone 后 `find .claude -name '*.json' | xargs grep -l '/home/'` —— 任何 hit 都 alert customer

**Pipeline integration**:
- merged-arch-sanity skill 加一项 check: hook command path 解析 + exists 检查 (currently 8/8 不含此项)
- workflow_critic.py 启动时自检：sys.argv[0] 在 `${CLAUDE_PROJECT_DIR}/src/scripts/` 之内吗？不在则 print 警告 + exit 0 (允许 instance 自查)

**Recurrence evidence**:
- 2026-04-27 Kimi spawn 3_Add: critic 因路径 mismatch silent fail，worker 跳过 Phase O5/O6 finalize, archive 用 unverified 数字写 verification.json — bypass detected only by self-audit (C13 起效 catch 了 silent failure)

**Severity**: HIGH for product release — hook 失效就是 critic 失效就是 OL-85/C18/anti-delegation 全部失效

**Related**: C2 (infrastructure bypass — 这是 infrastructure 自身的 portability bug); A-P-pseudo-tool-call-text (similar silent-fail signature, different cause)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CARRIER-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
