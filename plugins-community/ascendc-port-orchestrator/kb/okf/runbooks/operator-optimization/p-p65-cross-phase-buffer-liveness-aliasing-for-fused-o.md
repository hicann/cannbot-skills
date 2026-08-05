---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cross-phase buffer-liveness aliasing for fused ops (UB budget relief)"
description: "Scenario: a fused op's ProcessRow() has multiple phases (dequant / SwiGLU / quant / ...); each phase has its own scratch buffer. When the UB budget is tight you want to alias to save space, but the ao"
severity: high
confidence: single_run
original_id: P-P65
timestamp_inferred: true
tags: [memory_access, optimization, otherbuf_, tmpbuf_, tmpbuf, p-p65, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Scenario**: a fused op's ProcessRow() has multiple phases (dequant / SwiGLU / quant / ...); each phase has its own scratch buffer. When the UB budget is tight you want to alias to save space, but the aog-kernel-optimizer's global view often picks the wrong target (aliasing onto a still-live buffer) → silent UB overwrite → precision FAIL.

**Method (fused-op liveness graph)**:
1. For each phase, list every UB buffer's live range (which phase reads, which writes; on which line the last read occurs)
2. Find buffers **"dead past phase N"** — i.e. not used after a certain phase reads them, but physically still occupying a slot
3. The slot can be aliased to a later phase's new scratch; no need to InitBuffer a new TBuf
4. Key: alias target must be a buffer **dead within a cross-phase window**; a buffer still alive in the same phase (e.g. the amax scratch tmpBuf still used by DynQuant) cannot be aliased

**Verification step**: precision FULL re-verify (not a single case) — a wrong liveness judgment produces silent errors.

**op#11 example**:
- `otherBuf_` is dead after the last Mul in SwiGLU (line 413)
- Previously, aog-kernel-optimizer Opt4 chose `tmpBuf_` as the alias target → `tmpBuf` is still alive in DynQuant's amax reduction → silent overflow → precision FAIL → REVERT
- aog-fused-optimizer C4 correctly chose `otherBuf_` → precision PASS on first try, 20 KB released
- This is the first empirical win of the fused-op view over the global view

**Anti-pattern (i.e. the trigger for PB-17)**:
- Aliasing a buffer that is **VEC-written near the end of ProcessRow** and **MTE2-written near the start of the next ProcessRow** creates a V→MTE2 cross-row hazard (no sync → silent corruption). See PLATFORM_BUGS PB-17.
- As a heuristic: the "dead range" of an alias target must be fully clear across rows, or explicitly `SetFlag<HardEvent::V_MTE2>` at the end of ProcessRow.

**When to use this pattern**:
- Fused op, UB budget near the limit, depth=2 / tile expansion blocked
- Need to free UB for architectural restructure
- aog-kernel-optimizer has plateaued or Opt-N tried alias but precision FAILs

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P65，convert_patterns_to_okf.py）。confidence 未升格。 -->
