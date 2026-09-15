---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A verification script must be data-flow-independent of the verdict artifact it re-derives — emit diagnostics to stderr and ONE summary object to stdout; the caller (not the verifier) materializes the verdict file"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=backward (harness-structural: the gate is a Python static scan in phase_o5, SoC/CANN-independent)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=backward (harness-structural: the gate is a Python static scan in phase_o5, SoC/CANN-independent)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-277
timestamp_inferred: true
tags: [ascendc, ol-277]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; mode=backward (harness-structural: the gate is a Python static scan in phase_o5, SoC/CANN-independent)`
`verified_on: soc=Ascend950PR (selective_scan_full_grad kw-2 2026-06-18 / A5); the gate mechanism itself is SoC/CANN-independent`

**Principle**: a verification runner exists to INDEPENDENTLY re-derive a claim that lives in a verdict file. The moment the verifier reads or writes that verdict file it stops being independent — it can trivially echo the author's own claim (a self-citing cycle). Keep `author ≠ measurer`: the party that WROTE the claim must not be the party that re-derives it from the same file. `phase_o5._verify_runner_independence` enforces this STATICALLY — it scans the worker's `verify_<op>.py` for ANY non-comment reference to the verdict-file name and returns `RUNNER_FAILED` (rolling the FSM back to `await_worker` with `phase_o5_runner_failed`) if found, orthogonally to whatever precision verdict the run produced.

**Concrete anchor (backward-mode fix shape)**: the verify script prints all diagnostics to **STDERR** and emits a **SINGLE** final summary JSON object to **STDOUT**, carrying the top-level fields the O5 re-measure needs (`{tier1_pass, total, status, performance}` plus the full verdict blocks). The **worker** (author) then materializes the verdict file by running the script and persisting that stdout object — so both author and measurer derive from the same independent tool with ZERO verdict-file I/O inside the verifier. Keep stdout to ONE object so `json.loads(whole_stdout)` succeeds; a multi-line stdout with a `{` in an earlier line (a `dtype_summary` or per-case array) makes a tail-scan `_try_parse_json_tail` forward-pick the wrong `{`.

**Evidence**: selective_scan_full_grad kw-2 (2026-06-18, A5 Ascend950PR). The kw-1 spawn's `verify_<op>.py` did `with open(<verdict>, "w")` → tripped `_verify_runner_independence` → `RUNNER_FAILED` (cycle) → FSM rolled back to `await_worker`, a failure orthogonal to the actual precision FAIL and a wasted iteration. kw-2 rewired it to stderr-diagnostics + single-stdout-object + worker-persists-verdict; validated clean on NPU dev2 (parseable summary, stable ×3).

**Other instances (predicted)**: any backward-mode op's verify script; the O5 re-measure / independent-runner gate in migration and opgen modes; and generally any "author writes a claim, an independent runner re-derives it" split — never let the re-deriver touch the artifact it audits.

**Cross-ref**: OL-272 (backward-mode build/deploy — `_resync_workspace_to_container` `push_files` excludes `verify_<op>.py`, a sibling backward carve-out), OL-273 (Pass B verifier schema), the project Independent-Performance-Verification / author≠measurer rule. backend=ascendc (principle generalizes to all backends).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-277（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
