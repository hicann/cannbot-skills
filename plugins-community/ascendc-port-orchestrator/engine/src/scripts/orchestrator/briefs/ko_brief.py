# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-kernel-optimizer brief construction.

V3.8.4 escalation rule: this agent fires when worker self-declared
"done" with perf < 0.6×. Goal: incremental tuning that beats threshold
without precision regression.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs._common import (
    env_quirks_block,
    AscendCEnv, load_env,
    env_block, hard_floors_block, kb_manifest_block,
    schema_contract_block, fixed_layout_block, safety_block, g7_slug,
    self_introspection_block, _detect_forced_architecture,
)


def build_optimizer_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    directive_text: Optional[str] = None,
    handoff_from_worker: Optional[str] = None,
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    if env is None:
        env = load_env()
    # P131 follow-up: inherit env.backend when kwarg defaults (P131 main PR pattern)
    from briefs._common import resolve_backend_from_env
    backend = resolve_backend_from_env(backend, env)
    # Phase 2 (2026-05-19, Q3 main agent answer): self-resolve plugin if caller
    # didn't pre-resolve. Eliminates need for inline `if backend == "X"` in helpers.
    if plugin is None:
        from briefs._common import resolve_plugin_for_brief
        plugin = resolve_plugin_for_brief(env, workspace=workspace, backend_override=backend)
    slug = g7_slug(op, "aog-kernel-optimizer", spawn_index)

    sections = [
        f"{slug} — kernel-optimizer spawn (V3.8.4 perf escalation)",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _optimizer_phase_block(directive_text, handoff_from_worker,
                               iter_cap_remaining, plugin=plugin, workspace=workspace),
        "",
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        _optimizer_exit_handoff_block(),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}.",
    ]
    return "\n".join(sections)


def _forced_architecture_ko_block(workspace: Optional[Path]) -> str:
    """If the op's architecture is FORCED at classification time, return an
    instruction block binding KO (the optimizer) to optimize ONLY WITHIN the
    forced architecture — NOT take the Outcome-B architecture-rewrite path, NOT
    apply the OL-54 reg-based-SIMD lever to SWITCH architecture, and report a
    forced-arch perf ceiling as a valid conclusion rather than silently switch.
    Else "".

    Companion to `kw_brief._forced_architecture_block` (the kw side). Reason:
    even after kw correctly implements the forced architecture and precision
    passes, ko has an Outcome-B architecture-rewrite path + the OL-54
    reg-based-SIMD lever — so ko could silently switch a forced-SIMT op back to
    SIMD for perf, defeating the forced-architecture mandate (and the owner's
    SIMT-vs-SIMD comparison). Architecture switch under forced_arch is
    owner-approval-only, out of ko's autonomy.

    Empty for non-forced ops so their briefs stay byte-identical (ko keeps the
    Outcome-B / OL-54 levers as before).
    """
    forced = _detect_forced_architecture(workspace)
    if forced is None:
        return ""
    pipeline_focus = (
        "SIMT-pipeline / per-thread occupancy / regbase-under-SIMT"
        if forced == "SIMT"
        else "VEC pipeline / TQue depth / regbase-under-SIMD"
    )
    return (
        f"# ARCHITECTURE IS FIXED — {forced} (forced at classification; ko optimizes WITHIN it only)\n"
        "\n"
        f"This op's architecture is **FORCED to {forced}** by classification "
        "(`op_classification.json` carries a forced-architecture marker). kw "
        f"implemented {forced} and precision has passed. For PERFORMANCE you "
        f"may tune ONLY WITHIN {forced}:\n"
        f"- **Optimize within {forced}**: tiling / UB layout / "
        f"{pipeline_focus} "
        "/ DataCopy batching / launch overhead.\n"
        f"- **Do NOT take the Outcome-B architecture-rewrite path.** Rewriting "
        f"{forced} into the other architecture is FORBIDDEN under forced_arch.\n"
        "- **Do NOT apply the OL-54 reg-based-SIMD lever to SWITCH architecture.** "
        f"(Reg-based tuning WITHIN {forced} is fine; using it to flip {forced}→"
        "the-other-architecture is the forbidden switch.)\n"
        f"- **If you conclude the forced {forced} architecture is architecturally "
        "slower than the alternative, REPORT that as the forced-arch perf CEILING** "
        "(`KO_PERF_PLATEAU` with msprof evidence) — this is a VALID, WANTED "
        "conclusion: it IS the SIMT-vs-SIMD comparison datapoint the owner wants. "
        "Do NOT silently switch to recover the gap.\n"
        f"- **Architecture switch is owner-approval-only** — out of ko's autonomy "
        "under forced_arch. Surface the recommendation in optimization_log.md; do "
        "NOT execute the switch yourself."
    )


def _optimizer_phase_block(
    directive_text: Optional[str],
    handoff_from_worker: Optional[str],
    iter_cap: int,
    *,
    plugin: Optional[object] = None,
    workspace: Optional[Path] = None,
) -> str:
    handoff_str = (
        f"\n\n## Handoff from worker\n\n{handoff_from_worker}" if handoff_from_worker else ""
    )
    directive_str = f"\n\n## Directive (from prior probe / fo)\n\n{directive_text}" if directive_text else ""

    # Forced-architecture honor (2026-06-16): bind ko to optimize within the
    # forced arch only (no Outcome-B rewrite / no OL-54 arch-switch). Prepended
    # ABOVE the phases so ko sees the constraint before any tuning guidance.
    # Empty for non-forced ops (brief stays byte-identical).
    _forced_block = _forced_architecture_ko_block(workspace)
    _forced_prefix = (_forced_block + "\n\n") if _forced_block else ""

    return _forced_prefix + f"""# PHASES (aog-kernel-optimizer){handoff_str}{directive_str}

**SCOPE — optimize performance without weakening migration provenance.** The arch35
implementation must remain independently authored from arch22 interfaces and algorithm
semantics. Never read or copy a CANN arch35 implementation, and never accept renamed or
lightly rearranged target-source blocks as a port. If optimization exposes suspicious
source similarity, stop with an integrity blocker for harness review. Self-containment
(`no #include "arch35/..."` and no aclnn/aclop delegation) is necessary but not sufficient;
the provenance/similarity gates remain authoritative while you measure and tune device time.

A. **KB Manifest LOAD** — per section above.

B. **Read worker's verification.json + perf data** to identify bottleneck.
   Use msprof if perf data insufficient. Classify the gap:
   - HBM-bandwidth bound (multi-pass kernel hits structural ceiling per OL-53)
   - VEC-compute bound (Reg-based candidate per OL-54 IF symbol-existence
     verified — see CAND-PP82 anti-pattern; pre-edit grep
     /data/cann_b103/cann-9.0.0/x86_64-linux/asc/include/ for any Reg::*
     primitive you propose to use)
   - Launch overhead (small-shape cases dominated by aclrtLaunchKernel)
   - Scalar-pipe bound on A5 (OL-54 + P-REG-1 + ascend950pr.md §Reg-based
     MUST be cited if you declare structural ceiling)

C. **Apply tuning** in workspace/<op>/kernel/. Each iter:
   1. Edit ONE axis (don't combine)
   2. deploy_to_npu_lane.sh --lane <LANE> --build
   3. benchmark.py → preserve precision floors
   4. performance.py on ALL benchmark cases — independent re-measure
      (CLAUDE.md rule). Use the exact command:
      `python3 utils/performance.py --output_dir current_task --warmup 5 --repeats 50 --output current_task/perf.json --markdown current_task/perf.md`
      Update verification.json.performance with the new multi-case ratio (mean across all cases).
   5. msprof if perf metric ambiguous
   6. Compare against baseline → KEEP or REVERT

D. **Exit** when perf ≥ 0.6× threshold (mean speedup across ALL cases)
   OR iter_cap exhausted with PLATEAU.

If the bottleneck is architectural (e.g. multi-pass kernel can't beat
single-pass CANN reference) → classify as STRUCTURAL_CEILING + handoff to
fused_optimizer (V3.7.12 fo→ko escalation reverse path: ko-found-arch-issue
→ fo Kind-2 directive).

If precision regresses on ANY iter → REVERT immediately + log in
optimization_log.md."""


def _optimizer_exit_handoff_block() -> str:
    return """# EXIT HANDOFF OPTIONS

- `→ orchestrator: done — KO_OPTIMIZATION_LANDED, perf X.XXx (was Y.YYx)`
  (perf ≥ 0.6× achieved, all floors preserved)
- `→ orchestrator: done — KO_PERF_PLATEAU, best X.XXx after N iters`
  (cap exhausted, structural ceiling identified WITH msprof + reg-based
  citation evidence per V3.8.4 / SC5)
- `@aog-fused-optimizer` (architectural bottleneck identified — Kind-2
  rewrite needed; provide compelling msprof evidence in
  optimization_log.md)
- `@aog-precision-probe` (any iter triggered precision regression that
  REVERT didn't recover — investigate)
- `→ orchestrator: PARTIAL_PERSIST` (Tier-2 evidence on residual; routes
  via schema_norm fail-strict requiring probe_report.md OR pass_b two-tier
  evidence)

DO NOT write `→ orchestrator: done` if perf < 0.6× and no STRUCTURAL_CEILING
evidence — V3.8.4 schema_norm REJECTS this path."""
