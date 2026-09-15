---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "PyTorch-UB-class ops with non-deterministic reference — detection + verifier-side alt-ref mitigation"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "op whose semantics are PyTorch-UB on a specific input pattern. Concrete examples:"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-90
timestamp_inferred: true
tags: [ascendc, ol-90]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — entry's verifier-side alt-ref mitigation is the CPU-truth realization for PyTorch-UB-class non-deterministic ops. Aligns with OL-97. Do NOT downgrade.
- **Category**: pipeline / preflight / reference-determinism / verifier-policy
- **Loaded by**: orchestrator (Phase O2.5 step 4 followup when `ref_determinism.json` shows non-det), aog-kernel-worker (Phase A when worker hits PARTIAL on a dup-index scatter-class op), aog-precision-probe (Step 1 directive — non-det check on FAILING cases before classifying root cause)
- **Trigger**: op whose semantics are PyTorch-UB on a specific input pattern. Concrete examples:
  1. `torch.index_put_(accumulate=False)` with **duplicate indices** — PyTorch docs explicitly state "behavior is undefined if indices contain duplicate elements".
  2. `torch.scatter_(reduce=None)` (assignment scatter) with **duplicate indices** — same UB clause.
  3. Any inplace operation where the same destination is written by multiple parallel writers without specified ordering.
- **What "PyTorch-UB-class" means here**: the spec doesn't pin down a winner among duplicate writes. Different backends pick winners differently AND the SAME backend may pick different winners on different runs depending on hardware-thread scheduling. CPU torch is usually deterministic (last-wins on Python/sequential schedule). NPU torch is **NOT deterministic** when the parallel scatter contention is high enough.

- **Detection protocol (aog-precision-probe Step 1, mandatory before classifying)**:
  1. Replicate verifier's exact input generation for the failing cases (same `torch.manual_seed(0)`, same case order)
  2. For each failing case, run `torch_npu`'s reference op 5 times on cloned NPU buffers with explicit `torch.npu.synchronize()` between runs and fresh `.npu()` allocations (no aliasing)
  3. Compute pairwise bit-equality across the 5 runs. **If any pair differs → reference is itself non-deterministic on this input pattern**
  4. Save raw dump (per-dup-slot winner k, run_i pairwise diff matrix) to `workspace/{op}/probes/pp_npu_dup_det_5run.json`

- **Rule-search protocol (aog-precision-probe Step 2, also mandatory before classifying — even when Step 1 already shows non-det)**:
  Worker / probe must have tried at least these candidate deterministic rules and reported per-case match rates:
  - R1 last-wins (max k), R2 first-wins (min k)
  - R3/R4 wave-chunk last-wins / first-wins, W ∈ {8, 16, 32, 64, 128, 256, 512}
  - R5 sort-by-(idx, k)-stable last/first
  - R6 AIV-block-parallel: split K into A equal slices, last-wins within slice, max-end-slice-wins; A ∈ {1,2,4,8,16,20,40}
  - R7 round-robin (k % A buckets)

  **Acceptance threshold**: best rule must reach **≥ 99% match on every case AND ≥ 99% mean** before declaring "deterministic rule found, kernel can match". Anything ≤ 90% mean → REQUIREMENT.

- **Verifier-side mitigation candidates** (when REQUIREMENT confirmed; kernel side can't deterministically match a non-deterministic ref):
  1. **Alt-reference hook** (DEBT-049 sub-task B mechanism reuse): when comparing on UB-trigger inputs, fall back to CPU `torch` as alt-reference and accept either NPU-ref or CPU-ref match. Mirrors OL-89 dual-reference framing but specialized for UB-class — CPU torch is the deterministic reference here.
  2. **Case-gen scope contraction**: case generator emits `allow_dup_indices=False` for accumulate=False / reduce=None scatter cases. This is what OL-88's `case_gen.py` `index_range:<N>` unique-by-default already does for one direction; extend to op-spec-aware UB triggers.
  3. **Per-case tolerance loosening with documented rationale**: bump atol to ≥ 2× max-value-magnitude only when (a) `ref_determinism.json` flags non-det AND (b) probe confirms structured-rule search exhausted. NOT a blanket loosening — only on declared UB-trigger cases.
  4. **REPORT-side dual count**: report Pass A "vs NPU ref X/N (UB-trigger Y subset is non-deterministic by spec) + vs CPU ref Z/N" so customer reading the table sees both the conservative count and the deterministic-reference count.

- **What this is NOT**:
  - NOT a license to match by case-specific predicates (forbidden by OL-85 anti-overfitting). Found-rule must be op-level, not case-level.
  - NOT OL-89 dual-reference: OL-89 picks a Python decomposed truth as math ground; here CPU torch IS the deterministic alternative because the math IS undefined — there's no "math truth", only "implementation-defined" winners.
  - NOT a generalization of OL-88 class-1 (PA_BLK race in fused norm+rope+kvcache). OL-88 is about CANN ops with internal scatter race; OL-90 is about user-facing PyTorch-UB on duplicates. Detection is similar (5-run probe), interpretation differs (CANN bug vs spec UB).

- **Cross-reference**:
  - OL-83 (1-ULP boundary mismatch waiver): different — that's about deterministic reference with rounding-boundary inputs; OL-90 is about non-deterministic reference on UB inputs.
  - OL-85 (logic-first anti-overfitting): companion — OL-90 confirmation must NOT lead to case-specific kernel hacks; only verifier-side or case-gen mitigation.
  - OL-88 (ref non-determinism preflight): OL-90 is a special class — OL-88 covers CANN op internal races; OL-90 covers spec-level UB on user-supplied duplicates.
  - OL-89 (Python decomposed truth): inverse direction; OL-89 says "find Python truth", OL-90 says "Python truth doesn't exist for UB ops, use CPU torch as alt-deterministic-ref instead".

- **Evidence**:
  1. **op#19 IndexPut 2026-04-27 (aog-precision-probe pp-2)**:
     - Step 1 (`probes/pp2_npu_dup_det_5run.json`): NPU `torch.index_put_(accumulate=False)` 5-run det check on the 17 failing cases — **3/17 deterministic, 14/17 NON-deterministic**. The 3 det cases are all small-K (K ∈ {50, 64, 128}); all K ≥ 256 cases differ across runs.
     - Step 1b (`probes/pp2_step1b_case33_isolated.py`): isolated 8-run on case[33] (fp16, N=5120, K=2560) with fresh allocations + explicit syncs — confirms non-det, max abs pairwise diff = 4.39, 60/5120 (1.2%) of slots differ run-to-run. Probe-side artifacts ruled out.
     - Step 2 (`probes/pp2_step2_rule_match.json`): R1-R5 rule search across 7 wave sizes — best-on-mean is WF32 (wave-firstwins W=32) at **mean 61.2%, min 31.2%**. Best-on-min is plain last-wins at **mean 55.0%, min 30.6%**. NO rule passes the ≥ 90% mean acceptance bar. Three small-K det cases (0, 6, 15) hit 100% under WF32; everything else is below 75%.
     - Step 3 (`probes/pp2_step3_aiv_rule.json`): R6/R7 AIV-block-parallel + round-robin search (A ∈ {1,2,4,8,16,20,40}) — best is `aiv_firstwins[A=40]` at **mean 55.9%, min 31.6%**. Same conclusion.
     - Verdict (relative to NPU-torch-as-ref): REQUIREMENT. No deterministic kernel can chase a moving target.
     - **Iter 4 follow-up (OL-89 dual-ref, user-prompted, `probes/track2_pyref_compare.py`)**: kw-4 kernel compared against three deterministic Python references on all 17 failing cases — **17/17 bit-exact vs Python first-wins** (manual `for k in range(K-1,-1,-1): x[idx[k]]=v[k]` loop), 0/17 vs CPU torch (which picks last-wins resolution), 1/17 vs NPU torch. The kernel implements a clean Python sub-semantic of the UB; NPU torch implements a different (and non-deterministic) UB sub-semantic. **Conclusion: OL-90 fires AND OL-89 applies — math truth exists as a deterministic Python loop even when CPU torch picks a different UB resolution than the kernel.**
     - Op#19 final result: vs NPU torch 29/46 + vs Python first-wins 46/46 bit-exact (dual-precision report).
  2. **DEBT recommendation**: open follow-up DEBT to wire **Python first-wins alt-reference** into `utils/verification_ascendc.py current_task` for the `accumulate=False ∧ has_duplicate_indices` subset (DEBT-054 — choose the reference that matches the kernel's chosen UB sub-semantic, not CPU torch which picks the other). Mechanism mirrors DEBT-049 sub-task B. With this hook, op#19 published number lifts from 29/46 → 46/46 against the verifier's default (now-Python-first-wins) ref, without changing the kernel.

- **Lesson (process)**: when an op hits OL-90 (PyTorch-UB-class) and the verdict against the active CANN ref is REQUIREMENT, **always also run the OL-89 dual-reference probe** before publishing. The kernel's chosen sub-semantic may match a clean deterministic Python reference even when it doesn't match CPU torch. Op#19 surfaced this in two passes (pp-2 closed at REQUIREMENT relative to NPU; Iter 4 lifted to OL-89 dual-ref 46/46) — the orchestrator's directive had wrongly forbidden OL-89 framing under a misreading of "prerequisite (ref non-det) unmet". Once OL-90 confirms ref non-det on the failing cases, OL-89's prerequisite IS met.

- **Lesson (kw-5, 2026-04-27, choosing the simplest natural Python ref)**: OL-89 dual-reference path requires not just "find a Python ref" but **"find the SIMPLEST and most-natural Python reference that future readers can verify by inspection"**. Three candidate Python references are typically considered:
  1. **Pure Python forward `for k in range(K): x[idx[k]] = v[k]`** — natural, deterministic, what 99% of users assume. **Default choice — pick this unless evidence says otherwise.**
  2. CPU torch API `tensor.index_put_(...)` — looks natural but empirically uses a non-trivial vectorized scatter on large K (kw-5 finding: matches kernel only 39/46 even when kernel is forward last-wins). **Avoid as alt-ref** — it has its own UB resolution.
  3. Manual reverse loop `for k in reversed(range(K)): ...` — deterministic but unconventional first-wins. **Avoid unless explicitly justified** by a deterministic-NPU-behavior probe (and even then, suspect because NPU rarely deterministic at scale).
  Op#19 kw-4 chose option 3 (chasing NPU on small K) — this matched neither CPU torch nor pure Python and required a custom alt-ref. kw-5 switched to option 1 (forward = pure Python natural) — bit-exact 46/46 against the canonical Python reference, customer-inspectable. **The kernel direction choice and the verifier alt-ref choice should both align to option 1 by default**.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-90（category=pipeline / preflight / reference-determinism / verifier-policy，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
