---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Closing a coverage gap can surface failures rooted in the truth-source reference, not the kernel — a suite over-fit to the trivial path masks reference bugs and can invert a correct kernel's verdict"
description: "applies_to: soc=all (verification/methodology — chip-independent); cann=all; bisheng=n/a; op_class=all"
phenomenon: precision_issue
signal:
  - "applies_to: soc=all (verification/methodology — chip-independent); cann=all; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-262
timestamp_inferred: true
tags: [ascendc, ol-262]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all (verification/methodology — chip-independent); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

<!-- applies_to_backend: all -->

**Principle (decision rule)** — applies to ANY op whose reference (CPU truth / `model.py`) has more than one algorithmic branch, where one branch is degenerate/trivial relative to the others:

> A precision PASS achieved over a suite that is skew-loaded toward the trivial branch (e.g. the no-argmax-changing path) is a **weak signal**: it certifies the easy path, not the op. When the suite is later rebalanced to actually exercise the complex branch, the newly-appearing FAILs are **not necessarily kernel bugs** — they can be **truth-source reference bugs** that the skewed suite happened to never trigger. Treat a gap-closure FAIL as a two-hypothesis problem: (a) kernel wrong, (b) reference wrong. Check (b) BEFORE rewriting the kernel — a correct kernel can be made to look wrong by a buggy reference, and "fixing" the kernel to match a buggy reference is the worst outcome (it bakes the reference bug into the deliverable).

**Diagnostic direction (when gap-closure introduces new FAILs):**
1. Confirm the failing cases exercise a branch the prior suite never hit (a coverage delta, NOT a regression of formerly-passing cases).
2. Grep the kernel for the machinery the failing branch needs. If the kernel implements it correctly (e.g. always runs the full algorithm, no shortcut), the reference is the suspect.
3. Trace the reference's branch condition. A common reference-bug shape: a **fallback that can never fire** because its guard is tautologically true against the values it was computed from (e.g. `if x <= k` where `x` was derived only from the first `k` elements → always true → the `else` full-recompute never runs).
4. Fix the reference, re-synthesize truth, re-verify. Only if FAILs persist after the reference is corrected does the kernel become the suspect.

**Reporting-honesty corollary (anti-pattern)**: reporting dtype coverage as "bf16 3/3 PASS" when every bf16 case ran the trivial path is an **overstatement of the validated surface** — the honest form is "bf16 3/3 (trivial-path only) → extended to bf16 N/N (trivial + complex)". Coverage that is trivially satisfiable certifies nothing about the complex path; label it as such.

**Concrete anchor** (the tautological-guard reference bug — the shape to grep for in a suspect reference):

```python
# BUG: topPNum is derived from ONLY the first guess_k elements, so this guard is
# ALWAYS true -> the GuessKFailed full-sort fallback (A3 source `else { SortAll; recompute }`)
# never fires; when the guess genuinely fails, the reference silently truncates at guess_k.
if topPNum <= guess_k:        # tautology: topPNum <= guess_k by construction
    ...                        # uses only the guess_k-window result
# FIX: gate on whether the guess actually SUCCEEDED, matching A3 source `ifFind == 1`:
if mask.any():                 # mask = (cumsum within the guess_k window already >= p)
    ...
else:                          # guess failed -> full sort + recompute topPNum from complete cumsum
    ...
```

**Why this matters**: the natural reflex on a new precision FAIL is to edit the kernel. For multi-branch references that reflex is wrong as often as right — reference and kernel are independent code paths, and a coverage gap hides bugs in either. Editing the kernel to silence a reference-induced FAIL ships a wrong kernel (one that matches a wrong reference). The 2× cost of checking the reference first is negligible next to shipping a deliverable that encodes a reference bug.

**Evidence**:
- top_k_top_p_sample A3→A5 port, kw-4 (2026-06-24, Ascend950PR CANN 9.0.0): the kw-3 handoff suite was 10 cases, 7 of which ran the no-Q argmax path (near-tautological per OL-260) and only 3 exercised the real Q-pipeline, with **zero bf16+Q coverage**. Extending to 16 cases (9 Q-cases: 5 fp16+Q + 4 bf16+Q) immediately surfaced 2 FAILs in Branch C+Q (case 12 bf16+Q, case 14 fp16+Q). Root cause was a **`model.py` `_branch_c` bug**, NOT the kernel: the guard `if topPNum <= guess_k` was tautologically true (topPNum was computed from only `guess_k` elements), so the A3-source `GuessKFailed` full-sort fallback never fired and the reference truncated at 32 elements when the guess failed. The kernel was correct throughout (it always performs the full iterative sort). Fix: gate on `mask.any()` (matching A3 source `ifFind == 1`); after re-synthesizing CPU truth, 16/16 PASS bit-exact. Evidence cross-checked against `workspace/top_k_top_p_sample/model.py:96-131` — the `mask.any()` guard + full-sort `else` fallback are now present and the old tautology is documented in the inline comment (step 2.7 cross-check PASS).

**Other instances (predicted)**:
- Any multi-branch reference where one branch is a degenerate/early-exit path and the suite skews toward it (samplers, MoE routing, masked-attention, any op with an optional-tensor fast path).
- References ported from upstream (A3) whose fallback / `GuessK` / early-exit logic was transliterated with a guard that no longer discriminates — the original C/C++ control flow can hide a tautology that a Python re-expression exposes.
- Any "the kernel regressed when we added harder cases" report — re-classify as a two-hypothesis problem before touching the kernel.

**Cross-ref**: OL-260 (the no-Q argmax trivial path whose dominance caused the skew), OL-218 (emit concrete per-pass counts at finalize — the reporting-completeness facet), OL-85 (anti-overfit for KERNEL code — distinct; here the overfit is in the SUITE/reference, not the kernel), OL-261 (the same op's finalize-path-divergence lesson), P-P57 (the Q-path top-K technique the complex branch uses).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-262（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
