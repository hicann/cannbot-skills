---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A finalize `phase_o5_mismatch` (\"pass claimed but not measured\") with genuinely-correct worker counts is usually a stale interpreter/env pointer for the active container — never reduce counts to \"fix\" it"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (orchestration / finalize O5 re-measure infra)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (orchestration / finalize O5 re-measure infra)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-238
timestamp_inferred: true
tags: [phase_o5_mismatch, ascendc, ol-238]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (orchestration / finalize O5 re-measure infra)`
`verified_on: container a5_iouv2 (A5 Ascend950PR_9579); cann=9.0.0`

**Principle**: When finalize/O5 rolls back a genuinely-passing archive with `phase_o5_mismatch` reading "pass_a claimed but not measured (runner didn't return it)", the runner producing `measured.pass_a=None` is most often an ENVIRONMENT-resolution failure (the O5 re-measure invoked a nonexistent interpreter / wrong path for the active container), NOT a real count disagreement. **Diagnose read-only**: run the worker's own `pass_a_runner.py` via the EXACT O5 `docker_cmd` with the resolved `_resolve_npu_python_bin` path. Exit 127 / "No such file or directory" ⇒ the `NPU_PYTHON_BIN` pointer in `.ascendc_env` is stale for this container (a path copied from a different host). **Fix**: add a target-scoped `{TARGET}_NPU_PYTHON_BIN` override pointing at the container's real python (one where `torch.npu.is_available()`); the resolver checks `{TARGET}_NPU_PYTHON_BIN` BEFORE the generic pointer, so this fixes O5 re-measure fleet-wide on that container without disturbing the generic value. **Do NOT reduce pass counts** — the counts were never the bug. This is a config-pointer baseline-drift correction, analogous to the sanctioned `ASCEND_OPP_PATH` env-var fix, NOT a paper-over.

Concrete anchor: `.ascendc_env` generic `NPU_PYTHON_BIN=/root/miniconda3/envs/py311/bin` (copied from host `npu_dev3`) does not exist in container `a5_iouv2`, whose real interpreter is `/usr/local/python3.11.13/bin` (torch 2.7.1 + torch_npu, NPU available; runner does 225/225 in ~9 s).

## Evidence
- modulate kw-2 (2026-06-21): kw-1 shipped a genuine 225/225 PASS; O5 rolled back with the "claimed but not measured" mismatch. Probe confirmed exit 127 from the stale generic pointer. Added `A5_NPU_PYTHON_BIN=/usr/local/python3.11.13/bin` → `phase_o5.post_verify_for_finalize` VERIFIED, `measured.pass_a {tier1_pass:225, total:225}`, `mismatches:[]`.

## Other instances (predicted)
Any container whose `.ascendc_env` inherited a generic interpreter/tool pointer from a different host (multi-container fleets, fresh-clone agents); any O5 mismatch whose runner stdout is empty/unparseable rather than showing a lower count.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-238（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
