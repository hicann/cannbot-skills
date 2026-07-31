# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-cann-learner brief construction (Phase 1b, 2026-05-20).

Thin string-adapter — delegates ALL spawn config + gate enforcement to
`cann_learn.mode5_runner.run_mode5` (the carve-out's canonical entry).
This file holds NO business logic: no precondition checks (mode5_runner
gate_check_preconditions owns those), no scanner re-validation
(revalidate_post_agent owns), no sealed-dir setup (setup_sealed_dir
owns), no .kb_promotion_pending marker writing (run_mode5 owns).

Per design doc KB_DESIGN_NOTES.md#cann-learn-on-research-gap-design-2026-05-20 §3.5:
> `cl_brief.py` is a thin string-adapter that delegates spawn config
> to `mode5_runner.spawn_aog_cann_learner` — it does NOT duplicate the
> gate-precondition logic or the sealed-output / scanner-validation
> pipeline. Same anti-pattern PR #57 §6.5 P1 surfaced for other brief
> builders (duplicating phase-block content in the brief when the
> SKILL.md already owns the procedure).

The brief is the AGENT-facing input — what the aog-cann-learner Claude
sub-agent sees when spawned. The hard rules (sealed output, scanners,
identifier denylist, KB-overlap check) are enforced ORCHESTRATOR-SIDE
by mode5_runner BEFORE the spawn (gate) and AFTER the spawn (revalidate).
The brief reminds the agent of the rules so it doesn't try to write
outside sealed/ or invoke forbidden tools — but the orchestrator scanners
are the authoritative gate.

Phase 1b ships the brief + dispatch hook. A supported plugin opts in through
its method gate; the migration plugin enables it and makes the route reachable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs._common import (
    AscendCEnv, load_env,
    env_block, env_quirks_block, hard_floors_block, kb_manifest_block,
    schema_contract_block, safety_block, g7_slug,
    self_introspection_block,
)


def build_cann_learner_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    handoff_from_prior_agent: Optional[str] = None,
    directive_text: Optional[str] = None,
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    """Compose the aog-cann-learner agent's input brief.

    Kwargs mirror other brief builders for agent_dispatch.spawn_for_state
    dispatch consistency. cann_learner doesn't use directive_text /
    backend / plugin (gate evaluation already happened upstream), but
    accepts them for signature symmetry.
    """
    if env is None:
        env = load_env()
    slug = g7_slug(op, "aog-cann-learner", spawn_index)

    sections = [
        f"{slug} — aog-cann-learner spawn (auto-trigger from research-gap)",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _cl_phase_block(op, handoff_from_prior_agent),
        "",
        _cl_output_block(op),
        "",
        schema_contract_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}. cann_learner is "
        f"ONE-SHOT (iter_cap=1 per op-gen run). After this spawn, the orchestrator routes "
        f"back to await_researcher (if cann_learn_done) or finalize (if cann_learn_empty / "
        f"cann_learn_blocked). Don't expect a second iter — make this one count.",
    ]
    return "\n".join(sections)


def _cl_phase_block(op: str, handoff: Optional[str]) -> str:
    handoff_str = (
        f"\n\n## Handoff from researcher (the gap that triggered us)\n\n{handoff}"
        if handoff else ""
    )
    return f"""# PHASES (aog-cann-learner)

You are the CANN-learner subagent. The orchestrator's researcher (`aog-researcher`)
emitted `research_blocked` or `research_partial` — KB had no actionable pattern
for this op. The carve-out's plugin gate `should_auto_cann_learn_on_gap` returned
True, so you're auto-invoked to extract patterns from CANN reference source.{handoff_str}

## Procedure (canonical — owned by mode5_runner + the carve-out SKILL.md)

`cann_learn/mode5_runner.py` already ran the gate preconditions (researcher iter
≥ 1, ref_runnable RUNNABLE, CANN is BETTER than ours, hook preflight clean) and
set up your sealed working dir. Your job:

1. Read CANN reference source for `{op}` from `module_path` (orchestrator-injected
   absolute path; safe to read).
2. Extract generalized patterns — NOT verbatim kernel code. Focus on:
   - Algorithm-level scheduling (tile order, fusion shape, sync points)
   - Numerical-algorithm choices (online softmax recipe, Kahan summation, etc.)
   - Cross-core dispatch primitives (cube + vec coordination patterns)
3. Write candidates to `workspace/{op}/.cann_learn_sealed_<run_id>/candidates/*.md`
   (sealed dir; mode 0700; agent has write access).
4. Write public summary to `workspace/{op}/cann_learn_summary.json` (JSON-only,
   no prose, no identifiers, no paths — schema in `cann_learn/summary_schema.py`).
5. After return, orchestrator-side scanners re-validate (C34a identifier denylist,
   C34b compile-gate, C34c copy-shape, C35 KB-overlap). Candidates that pass land
   in `patterns/unverified/candidates.md` with `.kb_promotion_pending` marker.

## Strict rules (enforced by hooks G11/G12/SC10 — your writes will be REJECTED
if you violate)

- Sealed dir `workspace/{op}/.cann_learn_sealed_<run_id>/` is your ONLY writable
  area for CANN-derived content
- Public artifacts you MAY produce: `cann_learn_summary.json` (JSON only, no
  prose), `patterns/unverified/candidates.md` (append only)
- Forbidden writes: anywhere else in workspace, ANY source file in src/, ANY
  `.md` outside `patterns/unverified/`
- Forbidden tools: nohup, Agent (no sub-spawn), direct network egress

## Output contract

- `sealed_files`: list of paths under sealed dir
- `summary_path`: path to public JSON summary
- `candidate_paths`: paths to new candidates in patterns/unverified/
- `cann_files_read`: paths to CANN source you read (for scanner audit)"""


def _cl_output_block(op: str) -> str:
    return f"""# REQUIRED OUTPUT (mode5_runner expects these in your return)

- `sealed_files: list[Path]` — every file you wrote under
  `workspace/{op}/.cann_learn_sealed_<run_id>/`
- `summary_path: Path` — pointer to `workspace/{op}/cann_learn_summary.json`
- `candidate_paths: list[Path]` — new entries appended to
  `patterns/unverified/candidates.md` (one path per candidate)
- `cann_files_read: list[Path]` — every CANN source file you read (scanner
  cross-checks these against the identifier denylist; transparency is the gate)
- `metadata_fix_proposals_count: int` (optional) — count of existing KB entries
  you flagged for metadata-fix rather than new entry

## EXIT HANDOFF (one of, exactly)

- `→ orchestrator: cann_learn_done` — ≥1 candidate landed cleanly through
  scanner gates. State machine routes to await_researcher; researcher iter 2
  reads enriched KB.
- `→ orchestrator: cann_learn_empty` — no extractable patterns (CANN op
  missing OR all candidates rejected by C34/C35 scanners). State machine
  routes to finalize PARTIAL.
- `→ orchestrator: cann_learn_blocked` — gated by precondition failure
  (CANN not better than ours / ref not RUNNABLE / .candidates.md.lock held).
  State machine routes to finalize PARTIAL with gate-failure as evidence.

DO NOT improvise free-form handoffs (`done` / `pattern_found` / etc.) —
schema_norm will REJECT them and route you back to await_cann_learn (where
iter_cap=1 means you'd hit the cap immediately, abort the op-gen run)."""
