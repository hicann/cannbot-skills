# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-fused-optimizer brief construction (P0ii 2026-05-05).

Spawned when ko plateaus on a fused op AND fo has not yet run, OR via V3.3.10
pre-empt rule that escalates to fo BEFORE accepting Kind-2 from ko (since
fo's per-sub-op gap analysis informs the directive better than ko's global
ratio view).

Difference from ko: fo decomposes the fused reference into sub-ops with
known standalone baselines, computes per-sub-op gap, and emits a Kind-2
directive that points at the slowest sub-op for the worker to rewrite.
fo does NOT directly tune kernels — it produces a directive (or a verified
"no further win possible" verdict).

Origin: this brief was missing entirely (`agent_dispatch.BRIEF_BUILDERS`
had a TODO for fused_optimizer). Caught when 14_adaptive_instance_norm_bwd
ko-2 hit a hook deadlock and the orchestrator routed to await_fused_optimizer
per V3.3.10 — agent_dispatch.spawn_for_state then raised
NotImplementedError, killing the orchestrator with exit 3.
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


def build_fused_optimizer_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    directive_text: Optional[str] = None,
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
    slug = g7_slug(op, "aog-fused-optimizer", spawn_index)

    sections = [
        f"{slug} — fused-optimizer spawn (V3.3 fused-op decomposition)",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _fo_phase_block(directive_text, handoff_from_prior, iter_cap_remaining, plugin=plugin),
        "",
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        _fo_exit_handoff_block(),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}.",
    ]
    return "\n".join(sections)


def _fo_phase_block(
    directive_text: Optional[str],
    handoff_from_prior: Optional[str],
    iter_cap: int,
    *,
    plugin: Optional[object] = None,
) -> str:
    handoff_str = (
        f"\n\n## Handoff from prior agent\n\n{handoff_from_prior}"
        if handoff_from_prior else ""
    )
    directive_str = (
        f"\n\n## Directive (from probe / ko)\n\n{directive_text}"
        if directive_text else ""
    )

    return f"""# PHASES (aog-fused-optimizer){handoff_str}{directive_str}

A. **KB Manifest LOAD** — per section above. Pay particular attention to
   any reduction_quant.md / norm_*.md / fusion-related pattern files.

B. **Sub-op decomposition** — read `analysis.md` and `model.py` to identify
   the fused op's constituent sub-ops. For each sub-op:
   - Locate a provenance-tracked completed baseline of the same primitive
     OR a public AscendC API call that implements the primitive standalone.
   - If neither exists, mark the sub-op as "no-baseline" and skip in §C
     (you cannot compute a gap without a reference).

C. **Per-sub-op gap analysis** — for each decomposable sub-op:
   1. msprof on the current fused kernel; extract per-sub-op cycle counts
      (use trace markers if present, region annotations otherwise).
   2. Compare against the standalone baseline's measured cycles.
   3. Record `(sub_op, fused_cycles, baseline_cycles, gap_ratio)` to
      `fused_analysis.md`.

D. **Connection audit** — between adjacent sub-ops in the fused kernel,
   identify wasted work:
   - Buffer liveness: any TBuf/TQue carried across sub-ops that could be
     dropped (reduces register pressure).
   - Pipeline parallelism: any MTE2/VEC overlap opportunity missed.
   - Tile alignment: each sub-op tile-aligned independently OR sharing
     a unified tile? Mismatch is a common gap.

E. **Decision**:
   - **Architectural rewrite proposed**: emit `optimization_directive.md`
     with concrete Kind-2 directive (which sub-op to extract, which API
     to substitute, which buffer to alias). Worker re-spawns to execute.
   - **No actionable gap found**: emit `→ orchestrator: PARTIAL_PERSIST`
     with `fused_analysis.md` as Tier-2 evidence. The fused op is genuinely
     architecturally optimal at this scope.
   - **Bottleneck is in a single sub-op** + standalone baseline shows that
     baseline ALSO can't beat the gap → handoff to `@aog-precision-probe`
     for OL-53/OL-110 classification (this is structural, not architectural).

If the op is NOT actually fused (single-primitive reference, monolithic
source kernel, novel algorithm) → handoff `→ orchestrator: PARTIAL_PERSIST` with
rationale "not decomposable; fo not applicable" + cite the analysis.md
section that established this. fo MUST NOT invent decomposition where
none exists in the reference."""


def _fo_exit_handoff_block() -> str:
    return """# EXIT HANDOFF OPTIONS

- `→ orchestrator: done — FO_DIRECTIVE_EMITTED, target sub-op X.XXx gap`
  (`fused_analysis.md` + `optimization_directive.md` written; routes back
  to await_worker via state machine for Kind-2 execution)
- `→ orchestrator: PARTIAL_PERSIST — FO_NO_ACTIONABLE_GAP, fused-optimal`
  (per-sub-op gap analysis shows current kernel matches sub-op baselines;
  routes to finalize PARTIAL via Tier-2 evidence in fused_analysis.md)
- `→ orchestrator: PARTIAL_PERSIST — FO_NOT_APPLICABLE, op is not fused`
  (reference is monolithic / single-primitive / novel; fo's decomposition
  premise doesn't hold — finalize PARTIAL with rationale)
- `@aog-precision-probe` (per-sub-op gap traces back to a primitive that
  itself can't be optimized — OL-53/OL-110 structural classification needed)

DO NOT:
- Edit kernel/ files directly (fo emits directives, ko/kw execute them)
- Spawn sub-agents
- Skip the standalone-baseline locate step (no baseline = no gap math =
  invented numbers — anti-pattern)"""
