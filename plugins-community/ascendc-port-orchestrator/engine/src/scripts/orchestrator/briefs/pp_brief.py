# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-precision-probe brief construction.

DEBT-076 (probe iter_cap=1 too tight + premature classification) drives the
brief design — must include a "rigid output schema" requirement so probe
writes probe_result.json (codex P1 #58 future), and explicit anti-pattern
warning about declaring `requirement` while clusters remain UNTESTED.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs._common import (
    env_quirks_block,
    AscendCEnv, load_env,
    env_block, hard_floors_block, kb_manifest_block,
    schema_contract_block, fixed_layout_block, safety_block, g7_slug,
    self_introspection_block,
)


def build_probe_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    handoff_from_worker: Optional[str] = None,
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    if env is None:
        env = load_env()
    # P131 follow-up: inherit env.backend when kwarg defaults
    from briefs._common import resolve_backend_from_env
    backend = resolve_backend_from_env(backend, env)
    # Phase 2 plugin-as-param (Q3 main agent): auto-resolve if caller didn't.
    if plugin is None:
        from briefs._common import resolve_plugin_for_brief
        plugin = resolve_plugin_for_brief(env, workspace=workspace, backend_override=backend)
    slug = g7_slug(op, "aog-precision-probe", spawn_index)

    sections = [
        f"{slug} — precision-probe spawn",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _probe_phase_block(handoff_from_worker, iter_cap_remaining, plugin=plugin),
        "",
        _probe_output_schema_block(),
        "",
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}. "
        "CRITICAL per DEBT-076: do NOT classify `requirement` while clusters remain "
        "UNTESTED. If you can't bisect all clusters in iter_cap, classify "
        "`untested-cluster` (a non-terminal verdict that requests another probe iter) "
        "instead of forcing premature `requirement`.",
    ]
    return "\n".join(sections)


def _probe_phase_block(
    handoff_from_worker: Optional[str],
    iter_cap: int,
    *,
    plugin: Optional[object] = None,
) -> str:
    handoff_str = f"\n\n## Handoff from worker\n\n{handoff_from_worker}" if handoff_from_worker else ""
    return f"""# PHASES (aog-precision-probe){handoff_str}

A. **KB Manifest LOAD** — read every section listed above.

B. **Read worker's verification.json + analysis.md + PROGRESS.md** to
   understand the residual signature.

C. **Bisect** — do NOT jump to GM-dump. Run T1-vs-CPU triage first
   (CAND-PP80 methodology, ~5min vs GM-dump's ~150min):
   1. Compute reference torch_npu output, kernel output, AND pure-CPU fp64
      truth on the same input
   2. Pairwise diff: ref-vs-kernel, ref-vs-cpu_truth, kernel-vs-cpu_truth
   3. Decision tree:
      - kernel_vs_cpu flips ≈ ref_vs_cpu flips AND both large → OL-83-amplified
        / OL-110 sub-family (CAND-PP79 / CAND-PP82); ship at fail-floor with
        Tier-2 evidence; classify `requirement`
      - kernel_vs_cpu flips >> ref_vs_cpu flips → real kernel bug; THEN do
        GM-dump if needed; classify `bug` or `convention`
      - Both ≤ 1 flip per row → canonical OL-83 (1-ULP); classify `requirement`

D. **Classify** — write probe_report.md AND emit probe_result.json (rigid
   schema below). DO NOT use freeform Markdown classification — orchestrator
   parses the JSON.

If iter_cap_remaining = {iter_cap} insufficient for all clusters → classify
`untested-cluster` for unfinished clusters (NOT `requirement` — that's
DEBT-076 anti-pattern). Orchestrator will re-spawn probe with clarified
scope."""


def _probe_output_schema_block() -> str:
    return """# REQUIRED OUTPUT — workspace/<op>/probe_result.json

Rigid schema (codex P1 #58):
```json
{
  "classification": "requirement | convention | bug | untested-cluster | deferred",
  "confidence": "verified | partial | hypothesis",
  "next_directive": "<actionable text for kw or null>",
  "untested_clusters": [
    {"cluster_id": "X", "n_cases": N, "signature": "...", "reason_untested": "..."}
  ],
  "artifacts": ["workspace/<op>/probes/probe_pp1_*.py", "..."],
  "summary": "<1-3 sentence narrative for human eyes>"
}
```

Classification rules:
- `requirement`: residual is structural (OL-83/OL-110 fail-floor, ref-side
  non-determinism per OL-88/OL-89). NO actionable kernel fix. Routes to
  finalize PARTIAL with Tier-2 evidence.
- `convention`: a code-side adaptation will close it (e.g. zero-fill, dtype
  emit-cast, OL-83 step-4 API search found something). Routes to await_worker
  with directive in `next_directive`.
- `bug`: real kernel logic error identified. Routes to await_worker with
  fix recipe in `next_directive`.
- `untested-cluster`: SOME clusters bisected, OTHERS deferred due to iter
  budget. Routes back to await_probe with clarified scope. Use this AT LEAST
  ONCE before declaring `requirement` if any cluster remains untested.
- `deferred`: probe attempted but blocked by infra (e.g. NPU unavailable,
  ssh timeout). Routes back to await_probe (different lane / time).

Also write probe_report.md as human-readable evidence. probe_result.json is
authoritative for routing; markdown is for human review."""
