# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-determinism-analyzer brief construction (P0oo 2026-05-06).

Spawned when DET_POLICY=required AND observed_deterministic=false.
The det-analyzer's job: classify the source of non-determinism (HBM
race / atomicAdd order / FMA grouping / vendor non-det) and produce
determinism_report.md WITHOUT editing the kernel — analyzer-only role
per V3.2 contract.

Origin: was Day-2 TODO in agent_dispatch BRIEF_BUILDERS. P0ii closed
fo_brief; this commit closes the last brief gap. Now O1.5 (P0nn)
correctly classifies DET_POLICY, the await_det_analyzer state can
fire and dispatch will succeed.
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


def build_det_analyzer_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    handoff_from_prior: Optional[str] = None,
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    if env is None:
        env = load_env()
    from briefs._common import resolve_backend_from_env
    backend = resolve_backend_from_env(backend, env)
    if plugin is None:
        from briefs._common import resolve_plugin_for_brief
        plugin = resolve_plugin_for_brief(env, workspace=workspace, backend_override=backend)
    slug = g7_slug(op, "aog-determinism-analyzer", spawn_index)

    sections = [
        f"{slug} — det-analyzer spawn (V3.2 DET_POLICY=required)",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _da_phase_block(handoff_from_prior, iter_cap_remaining, plugin=plugin),
        "",
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        _da_exit_handoff_block(),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}.",
    ]
    return "\n".join(sections)


def _da_phase_block(
    handoff_from_prior: Optional[str],
    iter_cap: int,
    *,
    plugin: Optional[object] = None,
) -> str:
    handoff_str = (
        f"\n\n## Handoff from prior agent\n\n{handoff_from_prior}"
        if handoff_from_prior else ""
    )
    return f"""# PHASES (aog-determinism-analyzer){handoff_str}

You are ANALYZER ONLY. Do NOT edit kernel/ files. Do NOT run optimizer.
Your output is `determinism_report.md` — a precise classification of
where non-determinism enters this kernel + what it would take to fix.

A. **KB Manifest LOAD** — pay attention to OL entries on determinism:
   OL-88 (CANN reference non-det), OL-89 (FMA grouping), OL-83
   (mantissa-collision), platform PB entries on HBM race conditions.

B. **Read evidence** — `verification.json.determinism`,
   prior `run_det_check.py` output, kernel/ source structure.

C. **Bisect by hypothesis** — for each candidate non-det source, design
   a minimal probe:
   1. Single-thread atomicAdd-free path: clone the kernel without atomic
      ops, re-run det check. If now deterministic → atomic ops are the
      source.
   2. Dtype downgrade probe: if fp32 / bf16 mixed, force single dtype
      throughout, re-check. Diagnoses FMA grouping.
   3. Reduction-tree shape probe: if reduce uses partial sums in random
      order, force fixed-tree, re-check.
   4. Vendor-side probe: run reference (Model.forward) 3x on identical
      inputs; if reference non-det too, classify OL-88 (vendor-side, not
      our kernel).

D. **Classify** — one of:
   - `kernel-side-fixable`: deterministic path exists in our kernel; fix
     recipe in `next_directive` (worker re-spawns with directive)
   - `kernel-side-tradeoff`: deterministic path exists but loses N% perf;
     user decides via `await_user_decision`
   - `vendor-side-OL88`: reference itself non-det; our kernel can't be
     more deterministic than the truth → classify as `requirement`,
     route to PARTIAL_PERSIST
   - `untested-cluster`: bisection iter cap insufficient; route back to
     await_det_analyzer with clarified scope

E. **Output** `determinism_report.md` with:
   - Hypothesis tested (one per probe)
   - Probe code snippet (so reproduction is concrete)
   - Result (deterministic / non-det / inconclusive)
   - Classification + next_directive (or PARTIAL_PERSIST evidence)

If iter_cap_remaining = 1 (singleton spawn), you have ONE shot. Pick the
single highest-leverage hypothesis (typically atomic ops if present;
else FMA grouping if fp32/bf16 mixed). Don't over-bisect."""


def _da_exit_handoff_block() -> str:
    return """# EXIT HANDOFF OPTIONS

- `→ orchestrator: done — DA_KERNEL_FIXABLE, directive emitted`
  (`determinism_report.md` + `optimization_directive.md` written;
  routes to await_worker for Kind-1 fix)
- `→ orchestrator: done — DA_TRADEOFF_USER_DECISION`
  (deterministic path costs perf; route to await_user_decision)
- `→ orchestrator: PARTIAL_PERSIST — DA_VENDOR_NON_DET (OL-88)`
  (reference itself non-det per probe; finalize PARTIAL with
  `determinism_report.md` as Tier-2 evidence)
- `@aog-precision-probe` (det issue masks a precision issue — bisect
  precision first, then re-run det)

DO NOT:
- Edit kernel/ files (analyzer-only contract)
- Spawn sub-agents
- Skip the bisection step (without a concrete probe, classification is
  speculation — anti-pattern)"""
