---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Lift `nblk` to runtime AIV count for V220 portability"
description: "Anti-pattern: pybind11.cpp hardcodes uint32_t nblk = 56; (A5 AIV count) when launching AscendC kernels. Why it matters on V220: Ascend 910C has 80–96 AIVs; Ascend 910B has 40–48. A static nblk = 56 ei"
severity: medium
confidence: single_run
original_id: P-P51
timestamp_inferred: true
tags: [platform_compat, optimization, nblk, aclnncumsum, y_a, y_b, p-p51, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern**: pybind11.cpp hardcodes `uint32_t nblk = 56;` (A5 AIV count) when launching AscendC kernels.

**Why it matters on V220**: Ascend 910C has 80–96 AIVs; Ascend 910B has 40–48. A static `nblk = 56` either underutilizes (910C) or overcommits (910B b3/b4 with 40 AIVs). Compute-light ops (where launch parallelism dominates) lose 15–30% mean perf on big-shape cases.

**Fix recipe** (pybind11.cpp):
```c++
// Lift nblk to a runtime query of current device's AIV count.
// Option A — pure host-side: derive via aclrtGetCurrentNPUInfo or torch_npu's
//   GetCurrentDeviceProperties (specifics depend on CANN version).
// Option B — wrap a tiny "info" kernel that returns GetBlockNum() and use that.
// Option C — fallback static table per SOC variant from a header constant.

uint32_t nblk;
{
    // PoC: read once and cache. Fall back to A5's 56 if query fails.
    static int cached_nblk = 0;
    if (cached_nblk == 0) {
        // ... CANN host API or device-info query ...
        cached_nblk = query_aiv_count_for_active_device();
        if (cached_nblk <= 0) cached_nblk = 56;  // safe default
    }
    nblk = cached_nblk;
}
```

**Quick proxy** (one-line, low effort): bump `nblk = 56` to `nblk = 80` for a3-only builds. Verified: 1_GELU mean perf 0.48x → 0.60x (+16%), Pass A unchanged. Caveat: hurts a3 chips with fewer AIVs and over-spawns blocks for small-shape ops (very small launch overhead, amortizes worse).

**Trade-off vs always-use-max**: with `nblk = max_AIV`, small-shape ops pay launch overhead for blocks that do trivial work. The optimizer should consider per-shape adaptive nblk for compute-light ops where launch dominates.

**Verified-on**: 1_GELU on Ascend910_9382 — mean +16%, median slightly worse for small shapes (launch amortization). 14_Split predicted +30% (large-shape tail offenders dominate mean). To-do: apply across the 4 archived A3 ops and remeasure.

### A-P36: V220 `aclnnCumsum` chip-specific dim-dependent fp16 path (V351 has none)


**Empirical observation** — same probe run on both chips, same fp16 inputs, identical kernel-side approach (movedim-to-last + sequential fp32-acc + RINT):

| shape | dim | V351 (Ascend950PR_957b) `y_a` ≡ `y_b`? | V220 (Ascend910_9382) `y_a` ≡ `y_b`? |
|---|---|---|---|
| `[128,128]` | 0 | **bit-identical** | differ, max=0.125 |
| `[256,256]` | -2 | **bit-identical** | differ, max=0.188 |
| `[1,16,64,64]` | 1 | **bit-identical** | differ, max≈0.05 |
| `[8192,16384]` | -2 | (within fp16 reduction envelope, max=0.25) | **differ, max=1.125** |
| `[1024,2048]` | -2 | **bit-identical** | differ, max=1.125 |

Where `y_a = torch.cumsum(x, dim=dim)` (direct CANN call) and `y_b = torch.cumsum(x.movedim(dim,-1).contiguous(), dim=-1).movedim(-1, dim).contiguous()` (movedim-equivalent path).

**What this means**: V220's `aclnnCumsum` for `dim ≠ ndim-1` takes a **chip-specific optimization path** that V351's CANN does NOT have. V351 just runs the obvious "permute → innermost-cumsum → permute back" algorithm — so any kernel that does the same (e.g. our movedim approach) bit-matches V351's reference. V220 instead launches a SIMD kernel **on the original layout** with BlockDim=48 (msprof confirmed: single kernel `aclnnCumsum_CumsumAiCore_Cumsum`, AIV-only, no multi-stage tree).

**Sub-cases (V220 path)**:
- **scan_len ≤ 48 + numLines small (BlockDim=1)**: pure fp16 vectorized running buffer on original layout. **Reverse-engineered, bit-reproducible** (probe 03/05). Implement with `Add<half>` + sequential row iterations on original layout (NO movedim).
- **scan_len ≥ 64 + numLines large (BlockDim=48)**: V220 multi-AIV path, each AIV processes a column-slice. Algorithm not bit-reproducible from public AscendC primitives after 4 reverse-engineering rounds (probe 04 swept 36 chunk-K × strategy combos; closest reaches max=1.125 vs y_a but never bit-exact). msprof shows aiv_vec_fp16_ratio=0.002 + aiv_vec_fp32_ratio=0.002 (both low, neither pure-fp16 nor pure-fp32).

**Why same algorithm passes on V351 + fails on V220**: V351's CANN does NOT have this dim-dependent optimization, so it accepts kernels that take the obvious movedim path (our kernel + A5's archived `output/npukernelbench/src/kernels/5_Cumsum/` both 51/51 PASS on V351). On V220, the CANN reference itself uses the chip-specific path → kernels that take the movedim path differ from the reference (even though our kernel is **closer to fp64 ground truth** than the V220 reference is).

**Fix recipe (V220 only)**:
1. **For BlockDim=1 paths (small numLines, scan_len ≤ 48)**: implement `ProcessLineFp16VectorRunning` per probe 03 reverse-engineering. Skeleton: keep tensor on original layout, iterate scan dim sequentially, use `Add<half>(running, running, x_row, row_width)` SIMD. Closes case 29-class shapes.
2. **For BlockDim=48 paths (long scans)**: NO known public-AscendC fix. Document as `chip_scope: a3-only` PARTIAL with empirical evidence. Do not label this as a generic precision floor — the gap is V220 chip-specific, not generic CANN reference quirk. V351 with same algorithm passes 51/51.

**Diagnostic protocol** when encountering a similar fp16 non-innermost-dim mismatch on V220:
1. Run sibling-chip probe (V351 via `/a5_op`): if V351 sees `y_a ≡ y_b`, you're hitting A-P36.
2. Run `msprof` on the V220 reference call to confirm single-kernel SIMD launch + BlockDim=48 signature.
3. Document with chip-specific KB entry, not generic waiver.

**Verified-on**: A3 5_Cumsum, Ascend910_9382 (198.51.100.92, container npu-a3), 2026-04-26 — 5 fp16 Pass A failures (cases 5/8/29/38/40 in benchmark JSON) + 19 fp32 edge-dataset failures, all share the V220 chip-specific reduction-order signature.

**Cross-reference**: aog-self-critic C19 (sibling-project cross-check) and C20 (use msprof before declaring "blocked") were added to catalog 2026-04-26 to prevent the next agent from spending 5 probe rounds reverse-engineering blind without first running msprof on the reference.

**Forward question for V220 future ops**: any reduction op (`aclnn{Sum,Mean,Norm,Softmax,...}`) on dim ≠ -1 may exhibit similar V220-vs-V351 divergence. Check sibling chip + msprof BEFORE assuming the kernel is at fault.

## Known Bugs

### TQue depth=1 data race


`TQue<TPosition::VECIN, 1>` exhibits intermittent data races in certain kernels (random result deviation).
**Temporary fix**: Use depth=2 double buffering. Root cause pending confirmation from the CANN team.

See the discussion of TQue in U-P2 in unverified/candidates.md.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P51，convert_patterns_to_okf.py）。confidence 未升格。 -->
