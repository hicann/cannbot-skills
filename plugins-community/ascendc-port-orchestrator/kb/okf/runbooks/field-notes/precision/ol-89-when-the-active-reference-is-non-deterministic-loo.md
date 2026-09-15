---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "When the active reference is non-deterministic, look for a docstring-embedded Python decomposed reference and use IT as math ground truth"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "ANY of these"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-89
timestamp_inferred: true
tags: [ascendc, ol-89]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ **VALIDATED-CPU — this entry codifies CPU-truth selection rule when the active reference is non-deterministic**. Aligns with OL-97. Do NOT downgrade.
- **Category**: pipeline / reference-selection / math-truth
- **Loaded by**: orchestrator (Phase O2.5 step 3b — when `ref_determinism.json` flags any output non-deterministic), Phase O5 (extra precision validation when Pass A vs default ref shows residuals attributed to OL-88 class-1 race)
- **Trigger**: ANY of these
  1. `ref_determinism.json` shows `non_deterministic_outputs` non-empty (OL-88 class-1 / class-2 detected)
  2. Pass A precision PARTIAL with failures attributable to ref non-det per pp- probe (OL-88 verdict = `requirement`)
  3. Pre-emptively for any benchmark/opgen op whose active `Model.forward()` calls a single `torch_npu.<api>` or aclnn fused op (the active reference is opaque CANN; check if a decomposed Python truth exists)

- **Action — check for docstring-embedded Python reference**:
  ```bash
  # In the selected input-source definition:
  python3 -c "import ast, sys; t=ast.parse(open(sys.argv[1]).read()); \
    [print(n.lineno, n.value.value[:200]) for n in ast.walk(t) \
     if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
     and isinstance(n.value.value, str) and ('def forward' in n.value.value)]" \
    /path/to/benchmark/N_OpName.py
  ```
  If output lists a class docstring containing `def forward(...)` → there's a Python decomposed reference. Common pattern in NPUKernelBench: the Model class docstring shows the "PyTorch native implementation" while the active forward calls torch_npu.

- **Action — use it**:
  1. Extract the docstring's Python `forward()` body into a `ModelPyRef` class (probe template at `output/npukernelbench/src/kernels/12_KvRmsnormRopeCache/probes/python_ref_vs_cand.py`)
  2. Run `cand_kernel(inputs)` and `pyref_model(inputs)` on the failing cases, compare with `atol=0.01` (tolerate fp32-order residual; the docstring ref runs in different fp order than the kernel)
  3. If 100% PASS against pyref: the kernel is mathematically correct; the failures vs CANN ref are 100% CANN's bug (non-det / race / etc.)
  4. Document this in archive as `python_ref_validation.md` (template at op#12)
  5. Update REPORT.md row to show DUAL precision data: "vs CANN ref X/N + vs Python math ground truth Y/N atol=0.01"

- **Why this matters (the customer-reporting principle)**:
  - REPORT.md rows that say "Pass A 35/50" without context understate the kernel's correctness when the 15 failures are ref-side bugs
  - The conservative count (vs default verifier) is what the harness produces out-of-the-box, but "vs math ground truth" is the truth a customer can rely on
  - Both numbers should be in the report; the gap between them is itself diagnostic ("0/50 gap" → ref is solid; "15/50 gap" → ref has known issues, see linked probe report)

- **Action — backward mode**:
  Require an audited CPU forward specification and derive every requested gradient from it. If no CPU truth exists, request one and keep the result scope-incomplete.

- **Action — opgen mode with no explicit Python reference**:
  Worker should refuse to ship if there's no math truth source. Either:
  - Implement a small Python decomposition manually (orchestrator may write `<op>_pyref.py` based on the formula in the source's docstring)
  - Request user provide one (escalate)
  - Document as scope-incomplete and DO NOT claim PASS based on a single-shot ref

- **Anti-pattern (what this avoids)**:
  Reporting Pass A N/M against a CANN reference that has internal race / non-det / cold-buffer leak as the only precision metric. That number is conservative-by-coincidence (ref happens to be wrong this run) — it can flip between sessions and silently hide a kernel-correctness verdict.

- **Cross-reference**:
  - OL-83 (torch_npu vs pytorch-native drift): static algorithm difference. OL-89 is procedural: check FIRST whether the alternative Python reference exists before accepting the active reference's verdict.
  - OL-88 (ref non-determinism preflight): when OL-88 trips, OL-89 is the followup — use the math truth instead of the broken ref.
  - C18 (cheating-by-claim): OL-89 prevents the inverse cheating risk — under-reporting kernel correctness because the verifier picked the wrong reference.

- **Evidence**:
  1. op#12 KvRmsnormRopeCache 2026-04-25 — Pass A vs torch_npu = 35/50 (15 PA_BLK fails attributed to OL-88 class-1). User asked "doesn't benchmark provide python reference for math ground truth?" → orchestrator extracted the docstring Python decomposed reference (model.py lines 12-216), wrote `python_ref_validation.md` probe → 9/9 PA_BLK cases PASS against Python math truth (cache outputs bit-exact, ckv_cache sub-ULP residual). Verdict flipped from "ambiguous (kernel may be wrong)" to "kernel 100% correct, CANN ref is non-deterministic". REPORT row updated to dual precision. Customer-facing data integrity restored.
  2. **op#10 SwigluQuant 2026-04-26 (PARTIAL ops revisit)** — Pass A vs `torch_npu.npu_swiglu_quant` = 50/52 (2 ±0.5 fp32 boundary cases, FMA-grouping diff). Probe iter 1 already extracted the docstring Python decomposition into `pytorch_native_swiglu_quant.py` and verified kernel is bit-exact 52/52 vs that math truth (probe_report.md H1 verdict). At PARTIAL revisit time, OL-89 was applied: REPORT row reframed from "50/52 PARTIAL" (vs-CANN-only conservative count) to "**50/52 vs CANN ref + 52/52 vs Python math truth**". This is OL-89 trigger #3 (pre-emptive for `Model.forward()` that calls a single `torch_npu.<fused>`) — even when CANN ref is deterministic, FMA-grouping diff at rounding boundaries can flip a single int8 1-ULP and the conservative count understates kernel correctness. The 2-case gap is intrinsic to comparing two different legitimate fp32 paths through a quantization rounding boundary; "match aclnn FMA bit-exact" was previously verified unreachable via DEBT-036 adv_api probe.
  3. **op#30 NMS on a3 2026-04-28 (kw-1 first-shot, archived `output/npukernelbench-a3/src/kernels/30_NMS/`)** — Reference is `torch_npu.npu_nms_v4`. OL-88 preflight verified `selected_indices` (output_0) **NON_DETERMINISTIC** (2/3 probe cases differ run-to-run; max_diff ~ 2e9 because positions of indices in the int32 array shift across runs). Pass A vs CANN .json snapshot = **8/31** PASS (kernel produces a different-but-equally-valid tie-break order on most cases). Per OL-89, orchestrator wrote `edge_runner.py` that uses the deterministic Python CPU NMS from `30_NMS.py` docstring (filter → stable argsort desc → greedy IoU) as the reference. **Pass B (PRIMARY GATE) vs Python CPU = 31/31 bit-exact**, with set-equivalence comparison on `selected_indices[:num_selected]` (sort_asc both sides + bit-exact int32 array compare) + bit-exact `num_selected`. REPORT row reads "Pass B 31/31 bit-exact + Pass A 8/31 (informational, OL-88 non-det)". This is the **strongest application of OL-89 to date** — Pass B is the precision gate, NOT a supplementary check; CANN ref is downgraded to "informational" entirely. **Pattern variant for set-output ops**: when output is a SET (not a position-stable array) — e.g. NMS, top-k-on-ties, sparse selection — comparison must be set-equivalence (`sort_asc(kernel.selected[:n]) == sort_asc(ref.selected[:n])`), NOT positional bit-exact. This is a generalizable sub-pattern of OL-89 worth flagging in the `Action — use it` section as: "if the output is a set (not a position-stable array), compare with set-equivalence semantics; the docstring's emit-order is the reference algorithm, not the bit-exact target".
  4. **op#2 GroupedMatmul 2026-04-28 (prose-spec extension; archived `output/npukernelbench/src/kernels/2_GroupedMatmul/`)** — `ref_preflight verdict=FAILED` because A5 torch lacks both `F.grouped_mm` and `torch._grouped_mm`; the active `Model.forward` raises `AttributeError` on every case. The benchmark `.py` had **no docstring-embedded `def forward`** — strict OL-89 criterion did not apply. With explicit user sign-off (Discord 2026-04-28 "go ahead"), orchestrator extended OL-89 from "embedded executable Python" to **"docstring spec precise enough that the decomposition is mechanical (no rounding choices, no contraction-order ambiguity)"**. The benchmark prose spec (`out_g = A_g @ B[g]` per group, concat over G) is unambiguous — orchestrator appended a `def forward` block to a workspace-local source copy (`workspace/2_groupedmatmul/source_with_decomp.py`), re-ran preflight → `verdict=FAILED, alternate.present=true, recommendation=USE_ALTERNATE`. Worker then authored `workspace/2_groupedmatmul/model.py` with the per-group `torch.matmul` loop as the reference. Result: **50/50 Pass A + 16/16 Pass B bit-exact**, det 50/50, median 1.05×, gmean 0.83× (PASS). **Generalized rule**: when (a) active path is unrunnable AND (b) no executable docstring decomp exists BUT (c) the prose spec is mechanical, the orchestrator MAY append a docstring-decomp block to a workspace source copy with explicit user sign-off, then proceed via OL-89. The audit trail (user message + workspace-local-only source modification) is the integrity guarantee — upstream benchmark file is untouched.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-89（category=pipeline / reference-selection / math-truth，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
