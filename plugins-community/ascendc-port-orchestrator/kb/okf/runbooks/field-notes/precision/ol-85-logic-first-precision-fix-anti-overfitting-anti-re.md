---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Logic-first precision fix — anti-overfitting, anti-reward-hacking (CRITICAL)"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "any time a fix is proposed to close precision-failing cases"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-85
timestamp_inferred: true
tags: [ascendc, ol-85]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process / correctness / precision
- **Loaded by**: aog-kernel-worker (Phase D precision-fix iterations), aog-precision-probe (Step 2-3 bisection), aog-kernel-optimizer (iteration accept/revert), orchestrator (anti-cheat scan on any kernel diff)
- **Trigger**: any time a fix is proposed to close precision-failing cases
- **Lesson**: the primary diagnostic must be **"does our kernel's logic match the reference algorithm's logic?"** — NOT "which data-dependent predicate would make the failing cases pass?"

### Two classes of fix

**LOGIC-LEVEL FIX (acceptable, encouraged)** — modifies kernel's control flow or operation structure to mirror reference logic:
  - reorder ops to match reference sequence (e.g., `x.float()` before split vs after)
  - swap ordering of `round()` vs `clamp()` to match the specific code path in reference docstring
  - flip polarity of computation (e.g., `int_scale/y_max` vs `y_max/int_scale`)
  - fix dtype casting step to match reference's intermediate precision
  - change primitive selection (CAST_RINT vs CAST_ROUND) when docs/reference specify a particular rounding mode
  - correct loop bounds or accumulation range to match reference math

**REWARD-HACK FIX (forbidden)** — adds data-dependent branches that would NOT exist in a clean logic-matching implementation:
  - ❌ `if (col == 1620) { ... }` — case-specific branch
  - ❌ `if (fabs(val - (-0.5)) < epsilon) { val += 1e-7; }` — epsilon perturbation at rounding boundary
  - ❌ `if (batch_idx >= F1 && batch_idx < F2) skip_or_hack();` — range-specific branch
  - ❌ "Skip failing case X" masks in the kernel
  - ❌ Pre-computing which positions will fail the verifier and patching only those

### Rule for each agent

**aog-kernel-worker Phase D** (precision fix iteration):
  - Before proposing ANY fix, diff kernel's control flow against reference's logic step by step
  - If there's a logic divergence, fix the logic divergence (not the symptom)
  - Only invoke primitive-level fixes (rounding mode, reduction order, etc.) when logic alignment is already proven and drift persists
  - If no logic-level fix can close remaining cases WITHOUT data-dependent branches, exit handoff honestly — don't force closure

**aog-precision-probe** (bisection):
  - Hypothesis tree MUST include logic-alignment hypothesis BEFORE primitive-divergence hypotheses
  - `probe_report.md` MUST include §"Reference algorithm vs kernel: step-by-step logic diff" proving (or disproving) alignment
  - OL-83 waiver can ONLY be claimed AFTER logic alignment is verified AND drift is shown to be 1-ULP inherent-primitive (not a logic bug disguised as ref drift)
  - Recommended fix MUST be a logic-level fix or "no logic-level fix available → OL-83 confirmed"

**aog-kernel-optimizer**:
  - Perf optimizations must not break precision, and the changes applied must be understood as preserving the reference logic (not "it happens to still pass because the 50 cases happen to be in the same distribution")
  - Anti-pattern: "set special path for cases where tile_size > N" that applies differently to different shapes (if reference wouldn't do this, it's a hack)

**orchestrator** (reviewing worker/probe/optimizer output):
  - **Anti-cheat scan on kernel diffs**: grep for data-dependent conditionals that aren't in reference:
    - Literal indices (`== 1620`, `[:1620]`, `GetValue(1620)` with hard-coded constant)
    - Row-specific / batch-specific checks (`row_idx == X`)
    - Epsilon comparisons at rounding boundaries (`fabs(x - K) < 1e-N`)
    - "Skip" predicates on the case-index dimension
  - If the diff contains such patterns AND they aren't mirroring reference logic, REJECT the fix and spawn researcher or abort

### Evidence
- 9_TopKTopP probe closed 29/50 → 49/50 via LOGIC fix (P-P60 tie-break direction reselection — mirrors PyTorch stable-ASC semantics, no data-dependent hacks).
- 10_SwigluQuant (2026-04-19): worker hit 5 Phase D iters catching logic bugs (clamp order / scale polarity / UB alignment / Sigmoid composition). All fixes were logic-level. Residual 2-case OL-83 drift is being probed with explicit anti-overfitting constraint.
- User directive 2026-04-19: "现在的 probe 逻辑会不会出现过拟合的情况？他有没有从 reference 算子的逻辑入手而不是紧盯着失败的用例入手来解决问题？...但指导原则还是需要逻辑上保持一致，而不是一堆 if else 来 hack reward。"
- 22_Nonzero ko-1 RESUME (2026-05-03) — **clean-side validation, OL-85 distinguishes USER-FACING-KNOB dispatch from data-dependent hack**: Opt1 dispatched the pybind transpose path on `as_tuple` (an explicit user-facing argument of `torch.nonzero`), NOT on a data-shape predicate. Both `as_tuple={True,False}` paths remain bit-exact to reference. Result: 0.3734× → 1.1506× overall (Δ+208 %), Pass A 50/50 + Pass B 10/10 max_diff=0.0 + det 50/50 preserved. Cross-validated by op#22 kw-1 replay (2026-05-04): 1.14× independent re-measure (Δ ~0.9 % run-to-run noise), structural stability across container restart. Anchors the rule that branching on user-API knobs is acceptable optimization; what OL-85 forbids is branching on data values / case indices / shape-class predicates that mimic the failing-case set.
- selective_scan_full_grad opt-0 (2026-07-14, A5) — **NO_EDIT on a structurally-unmeasurable perf ratio is the anti-reward-hack move**: no vendor `aclnnSelectiveScan`/Mamba/SSM op exists (researcher grep 0 hits) so the perf ratio is structurally undefined; the optimizer correctly declined to touch the kernel because any edit would be unmeasurable, verdict-irrelevant, and risk the precision floor — a perf-side application of "don't change working logic to chase a number you can't honestly measure." Terminal verdict stayed precision-driven PARTIAL_PERSIST, not a perf plateau.

### Related
- OL-83 (torch_npu vs pytorch-native drift): OL-83 is an outcome that can only be claimed AFTER OL-85 logic-alignment verification; without OL-85 rigor, OL-83 becomes a lazy "waiver" for unsolved bugs.
- OL-84 (worker brief attention budget): parallel principle — don't let secondary quality dimensions dilute primary-dimension logic fidelity.
- ALWAYS_LOADED_RULES §5 iron law (literal-first translation): OL-85 extends §5 from Phase B generation to Phase D fix iterations.
- CLAUDE.md "No Workarounds — find and fix root causes" rule.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-85（category=process / correctness / precision，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
