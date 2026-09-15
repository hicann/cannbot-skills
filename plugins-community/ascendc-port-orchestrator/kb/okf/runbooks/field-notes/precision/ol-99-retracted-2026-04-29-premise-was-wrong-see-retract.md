---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "[RETRACTED 2026-04-29 — premise was wrong, see retraction note below]"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "A precision sweep reports MARE >> threshold but MERE << threshold, AND failing elements correspond to reference values < 1e-4"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-99
timestamp_inferred: true
tags: [ascendc, ol-99]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

**Original claim**: "MERE/MARE eps=1e-7 amplifies 1-ULP rounding noise to MARE > 10⁴ — kernel is at hardware floor; fix is verifier-side."

**Why retracted (2026-04-29 same day)**: empirical probe on op#28 showed `torch_npu.Model.forward` (CANN) produces output that is **bit-exact to CPU `torch.Model.forward`** on the same inputs:
```
case 7  (fp32):  Bit-exact at native dtype: True
case 11 (bf16):  Bit-exact at native dtype: True
case 49 (bf16):  Bit-exact at native dtype: True
```
So 0 ULP off CPU IS achievable. Our kernel's 1 ULP off CPU was **NOT** a hardware floor — it was an implementation gap. The premise "1 ULP is the hardware ULP floor for this op" is empirically false.

**Source of error**: I conflated "our kernel is 1 ULP off CPU" with "1 ULP is the hardware floor for this computation". The former is observation; the latter requires proving NO implementation can do better. CANN proves the latter is false.

**What's still partially true**: the MARE eps=1e-7 amplifier IS a real phenomenon — when a kernel produces 1-ULP-different output from CPU, near-zero refs make MARE explode. But the proper response is to FIX the kernel, not declare the standard wrong. Only consider verifier-side mitigation AFTER establishing that no implementation can match CPU bit-exactly (which would require checking CANN behavior first).

**Action items**:
- This entry retained as a teaching example of premature conclusion / motivated reasoning.
- Do NOT cite it as guidance.
- Related OL-100 retracted in same wave.
- Lesson: always probe achievable bar (e.g., `torch.equal(cpu_out, cann_out)`) before claiming a precision floor.
- **Category**: precision / verification
- **Loaded by**: aog-precision-probe (Step 1 — identify whether failure is real bug or eps amplification), Lead/Reviewer (when interpreting precision sweep results)
- **Trigger**: A precision sweep reports MARE >> threshold but MERE << threshold, AND failing elements correspond to reference values < 1e-4

### Lesson

The br_430 / production-skill MERE/MARE formula is:
```
relative_error = |actual - golden| / (|golden| + 1e-7)
MARE = max(relative_error)
PASS: MERE < threshold AND MARE < 10 × threshold
```

The `+ 1e-7` denominator floor (eps=1e-7) is below the granularity of:
- bf16 ULP (~7e-3 at value 1.0)
- fp16 ULP (~1e-3 at value 1.0)
- fp32 sub-normal range (~1.4e-45 normal, but rounding noise at 1e-6 is common)

So when:
- `golden = 0` (or near-zero, < eps)
- `actual = 1 ULP of the dtype` (smallest non-zero representable)
- `relative_error = 1_ULP / 1e-7` → 10² to 10⁴ depending on dtype

This shows up as **MARE catastrophe with MERE benign**.

### Empirical evidence (op#28 MultimodalRopePosComputationWithGridBasedIndexing)

Probe at 2026-04-29 (after linspace endpoint fix):

| Case | dtype | MERE | MARE | Worst element ref | Worst element actual | abs_diff |
|------|-------|------|------|-------------------|---------------------|----------|
| 7 | fp32 | 8.14e-6 | 6.30e-1 | -3.21e-6 | -5.30e-6 | 2.09e-6 |
| 11 | bf16 | 1.07e+0 | 4.35e+3 | 0.0 | 5.57e-4 | 5.57e-4 |
| 49 | bf16 | 4.85e+0 | 2.75e+4 | 0.0 | 2.75e-3 | 2.75e-3 |

Max abs_diff across all elements in case 49: 1.5625e-2 = **exactly 1 bf16 ULP** at value 2.0. The kernel cannot get more accurate than 1 ULP at the dtype's representable precision. Yet MARE reports 27,000.

### How to diagnose: bug vs MARE artifact

When you see `MERE small, MARE huge`:

1. Probe the worst-MARE element:
   ```python
   diff = (cand - ref).abs()
   rel = diff / (ref.abs() + 1e-7)
   worst = rel.argmax()
   print(ref.flatten()[worst], cand.flatten()[worst], diff.flatten()[worst])
   ```
2. **If `|ref| < 1e-4` AND `abs_diff < 1 dtype-ULP`**: MARE artifact. Kernel is correct.
3. If `|ref| > 1e-2` AND `rel > 10×threshold`: real precision bug. Diagnose further.

### Mitigation paths

When the failure is artifact:
- **Verifier-side fix (preferred)**: bump `eps` to dtype-aware floor:
  - fp32: eps = 1e-5 or 1e-6
  - fp16: eps = 1e-3 (= ~ ULP at 1.0)
  - bf16: eps = 1e-2
  This requires changing `utils/verification_ascendc.py` upstream (`Just-it/AscendOpGenAgent`).
- **Kernel-side**: cannot fix — the kernel is already at hardware precision floor.
- **Document**: declare op as "precision PASS at hardware-ULP level; MARE-eps artifact under default 1e-7 — request verifier-side atol floor".

### Anti-pattern

Trying to "fix" the kernel by changing computation order, FMA disable, fp64 promotion, etc. when the bug is actually MARE-eps artifact. **First probe**, then decide.

### Generalizes to

ANY composite op whose reference can produce exact zeros or near-zero values:
- Bilinear interpolation (op#28)
- Embedding gather + reduction (op#19)
- Matrix multiply with sparse / zero-cancelled rows
- Conv with padding regions producing zeros
- Top-k masking that zeros excluded entries

### Cross-reference

- Sweep 2026-04-29: 42/68 ops were Q4 (both ours and CANN partially fail). Many of those Q4 failures fit this profile.
- See `docs/data/sweep_cpu_truth_2026_04_29.tsv` for the data.
- File upstream issue: `Just-it/AscendOpGenAgent`, request dtype-scaled atol floor in MERE/MARE.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-99（category=precision / verification，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
