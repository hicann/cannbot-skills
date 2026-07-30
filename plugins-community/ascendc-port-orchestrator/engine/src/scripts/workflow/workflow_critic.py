#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Workflow critic — mechanical checker for /ascendc-op-gen state machine.

Reads the plugin's workflows/opgen_state_machine.yaml and validates current workspace
state against the phase the orchestrator is trying to advance to.

Designed to be invoked as a Claude Code hook:

    PreToolUse on Agent  → --mode pre_agent_spawn (blocks skipped phases)
    PostToolUse on Agent → --mode post_agent_return (validates agent's output)
    PreToolUse on Edit   → --mode pre_tool_edit (enforces G1 kernel-edit isolation)
    PreToolUse on Bash   → --mode pre_commit_sync (SKILL↔YAML drift)

Rejections exit 2 with structured stderr (rule ID + what's wrong + fix).

Design principles:
  1. Mechanical checks only. No LLM judgment. File existence + grep + JSON schema.
  2. Adversarial default: when a check can't be evaluated confidently, REJECT.
  3. Every rejection cites rule ID from the YAML for traceability.
  4. Critic rejection must force orchestrator to re-run, not just log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from workflow_critic_common import (  # shared leaf (2026-07-05)
    REPO_ROOT, YAML_PATH, Rejection, reject_and_exit,
    load_state_machine,
)
from workflow_critic_validators import (  # gate-check cluster (2026-07-05)
    check_global_invariants, check_phase_O4_agent_loop, validate_phase_o4_state, check_phase_O5_post_verify,
)

try:
    import yaml as _yaml  # noqa: F401 - availability gate for the YAML-backed critic
except ImportError:
    sys.stderr.write("workflow_critic: PyYAML not installed. `pip3 install pyyaml`\n")
    sys.exit(0)  # don't block if critic itself is broken; that's a tooling bug


# ---------------------------------------------------------------------------
# Repo & YAML config locations
# ---------------------------------------------------------------------------
# 2026-04-27 fix: workflow_critic was relocated from src/scripts/workflow/ (3
# levels deep) to src/scripts/ (2 levels deep) during an earlier merge
# merge, but `parents[3]` was not adjusted — REPO_ROOT silently resolved to the
# parent of a3_ops, so `docs/workflow/opgen_state_machine.yaml` could never be
# found and `find_active_workspace()` always returned None. Net effect: every
# Agent / Edit / Bash hook fail-opened (exit 0) — G1, G7, Phase-O preconditions
# all silently bypassed in the merged_skills deployment. Mirror the robust
# walk-up pattern that state_machine.py already uses.
WORKSPACE_ROOT = REPO_ROOT / "workspace"
PLUGIN_ROOT = YAML_PATH.parent.parent
ENTRY_SKILL_PATHS = (
    PLUGIN_ROOT / "skills" / "ascendc-cross-gen-port" / "SKILL.md",
    PLUGIN_ROOT / "skills" / "ascendc-backward-gen" / "SKILL.md",
)


# ---------------------------------------------------------------------------
# Rejection machinery
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# State machine loader
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Workspace detection
# ---------------------------------------------------------------------------
def _resolve_workspace_from_slug(op_slug: str) -> Path | None:
    """V3.8.1 (DEBT-068, 2026-05-03) + V3.8.2 (DEBT-071, 2026-05-04):
    resolve workspace dir from G7 slug.

    The G7 agent name is `{op_slug}-{agent_code}-{N}`; `op_slug` is the per-op
    handle. Find the workspace dir whose name matches this slug, in order:

      1. exact: workspace/{op_slug}
      2. numbered prefix: workspace/<N>_{op_slug} (case-insensitive,
         hyphen/underscore-tolerant)

    V3.8.2 DROPPED priority-3 substring fallback. The substring match was
    permissive enough to mis-route abbreviated slugs (e.g. `adain2dbwd` →
    `workspace/adain2d_bwd_v31` instead of `workspace/14_adaptive_instance_norm_bwd`),
    which then fails workflow_critic O2_5 because the wrong workspace lacks
    input_gen.py. Strict matching is safer; spawning agent with a slug that
    doesn't priority-0 or priority-1 match a workspace falls back to the
    legacy mtime heuristic (caller's responsibility to use a clean slug).

    Returns the most-recently-modified PROGRESS.md's parent if multiple match.
    Returns None if no candidate has a PROGRESS.md.
    """
    if not op_slug or not WORKSPACE_ROOT.exists():
        return None
    slug_norm = op_slug.lower().replace("-", "").replace("_", "")
    candidates = []
    for d in WORKSPACE_ROOT.iterdir():
        if not d.is_dir():
            continue
        if not (d / "PROGRESS.md").exists():
            continue
        name_norm = d.name.lower().replace("-", "").replace("_", "")
        # Strip numeric prefix (e.g. "12kvrms" → "kvrms")
        name_stripped = re.sub(r"^\d+", "", name_norm)
        priority = None
        if name_norm == slug_norm:
            priority = 0
        elif name_stripped == slug_norm:
            priority = 1
        # V3.8.2: priority-2 substring fallback REMOVED — too permissive,
        # caused mis-routing on abbreviated slugs (DEBT-071).
        if priority is not None:
            mtime = (d / "PROGRESS.md").stat().st_mtime
            candidates.append((priority, -mtime, d))
    if not candidates:
        return None
    candidates.sort()  # priority asc, then most-recent mtime
    return candidates[0][2]


def find_active_workspace(agent_name: str | None = None) -> Path | None:
    """Resolve the workspace directory the next critic check should target.

    Resolution order (V3.8.1 — DEBT-068):
      1. CLAUDE_ACTIVE_WORKSPACE env var (explicit override)
      2. G7 slug from caller-supplied agent_name (multi-op-safe)
      3. Most-recent PROGRESS.md mtime within last 4h (legacy heuristic;
         WRONG for multi-op concurrent flows when N>1 ops are active —
         see DEBT-068 root cause)
    """
    env = os.environ.get("CLAUDE_ACTIVE_WORKSPACE")
    if env:
        p = Path(env)
        if (p / "PROGRESS.md").exists():
            return p
        p2 = REPO_ROOT / env
        if (p2 / "PROGRESS.md").exists():
            return p2

    # V3.8.1: prefer G7-slug-driven resolution when the caller has the agent_name.
    if agent_name:
        m = _AGENT_NAME_PATTERN.match(agent_name.strip())
        if m:
            slug_token = m.group(1)
            # slug_token = "{op_slug}-{code}-{N}"; op_slug is the leading part.
            op_slug = re.sub(r"-(?:kw|pp|ko|fo|ar|da|bs|cl)-\d+$", "", slug_token)
            ws = _resolve_workspace_from_slug(op_slug)
            if ws is not None:
                return ws

    # Most recent PROGRESS.md under workspace/ modified in last 4h
    if not WORKSPACE_ROOT.exists():
        return None
    candidates = []
    import time
    now = time.time()
    for prog in WORKSPACE_ROOT.rglob("PROGRESS.md"):
        age = now - prog.stat().st_mtime
        if age < 4 * 3600:
            candidates.append((age, prog.parent))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Exception-file check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Artifact checks (mechanical)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Phase detection from workspace state
# ---------------------------------------------------------------------------
def detect_current_phase(ws: Path) -> str:
    """Heuristic: infer the latest COMPLETED phase from workspace artifacts."""
    if not (ws / "PROGRESS.md").exists():
        return "pre_O1"
    prog = (ws / "PROGRESS.md").read_text()

    # Look for phase markers (most recent wins)
    if "→ orchestrator: done" in prog:
        if (ws / "verification.json").exists():
            return "O5_post_verify"  # or O6 if archived
    if (ws / "verification.json").exists():
        return "O4_agent_loop"
    if (ws / "kernel").exists():
        return "O4_agent_loop_in_progress"
    if (ws / "analysis.md").exists():
        return "O4_agent_loop_started"
    if (ws / "edge_dataset.pt").exists():
        return "O3_progress_init"
    if (ws / "edge_inputs.pt").exists() or (ws / "reference_dataset.pt").exists():
        return "O2_5_reference_provider"
    if "Determinism Policy:" in prog:
        return "O1_5_det_policy"
    return "O1_parse_args"


# ---------------------------------------------------------------------------
# Mode-specific checks
# ---------------------------------------------------------------------------

# Tokens that won't work on V220 (a3/a2). If any kernel file under
# workspace/{op}/kernel/ contains these AND the active TARGET is a3 or a2, the
# build either won't compile (SIMT primitives) or will register-fail at runtime
# (KERNEL_TASK_TYPE_DEFAULT). Better to fail fast at workflow_critic (G7-target).


# ---------------------------------------------------------------------------
# Phase O4 state-machine validator (V3.3, 2026-04-23)
#
# Reads phase_o4_states section of YAML and validates that the agent spawn
# sequence in a workspace matches the declared state machine. This replaces
# hard-coded routing logic in SKILL.md prose.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Mode-specific entry points
# ---------------------------------------------------------------------------
_AGENT_NAME_PATTERN = re.compile(r"^([a-z][a-z0-9_]*-(?:kw|pp|ko|fo|ar|da|bs|cl)-\d+)\b")


def _extract_agent_name(tool_input: dict) -> str | None:
    """Pull the per-spawn agent name from `tool_input`.

    Stock Claude Code's Agent tool schema does NOT expose `name=` as a public
    parameter (only description / isolation / model / prompt / run_in_background
    / subagent_type). Confirmed on 2.1.119/2.1.120 (2026-04-25). The hook
    therefore reads the convention slug `{op_slug}-{code}-{N}` from the leading
    portion of the `description` field instead. `name` is still tried first in
    case a future CC version (or SDK-driven invocation) adds it as a real param.
    """
    n = tool_input.get("name")
    if isinstance(n, str) and n.strip():
        return n.strip()
    desc = tool_input.get("description")
    if not isinstance(desc, str):
        return None
    m = _AGENT_NAME_PATTERN.match(desc.strip())
    return m.group(1) if m else None


def mode_pre_agent_spawn(agent_type: str | None, agent_name: str | None = None) -> None:
    """Before orchestrator spawns an agent, verify preconditions satisfied.

    V3.3 (2026-04-23): also enforce Rule G7 — every Agent() spawn MUST include
    a name slug `{op_slug}-{agent_code}-{spawn_index}`. Originally checked the
    `name=` parameter; V3.3.1 (2026-04-25) reads the slug from the leading
    portion of `description=` since stock Claude Code Agent has no `name`
    parameter. Without an addressable slug, PROGRESS.md / state_transitions
    references are ambiguous when multiple agents of the same type have run.

    V3.8.1 (DEBT-068, 2026-05-03): pass agent_name to find_active_workspace
    so multi-op concurrent flows resolve the workspace from the G7 slug
    (`kvrms-ko-1` → `workspace/12_kvrms`) instead of the freshness heuristic
    that picks the wrong workspace when multiple ops are active.
    """
    ws = find_active_workspace(agent_name=agent_name)
    if ws is None:
        # No active workspace yet — nothing to check (we're in Phase O1)
        return

    load_state_machine()
    rejections: list[Rejection] = []

    # G7: all op-gen agent spawns must be named
    AGENT_TYPES_REQUIRING_NAME = {
        "aog-kernel-worker", "aog-precision-probe", "aog-kernel-optimizer",
        "aog-fused-optimizer", "aog-researcher", "aog-determinism-analyzer",
        "aog-cann-learner",
    }
    if agent_type in AGENT_TYPES_REQUIRING_NAME and not agent_name:
        rejections.append(Rejection(
            rule_id="G7",
            description="Agent spawn without addressable name slug — needed for PROGRESS.md / state_transitions disambiguation",
            expected="Agent(subagent_type='...', description='{op_slug}-{agent_code}-{spawn_index} <free text>', ...)",
            actual=f"spawn of '{agent_type}' has neither a `name` field nor a leading slug in `description` "
                   "matching `{op_slug}-(kw|pp|ko|fo|ar|da|cl)-{N}`",
            fix="Stock Claude Code Agent has no `name` parameter; encode the slug as the FIRST token of "
                "`description`. Agent codes: kw/pp/ko/fo/ar/da/cl. "
                "Example: Agent(subagent_type='aog-kernel-worker', "
                "description='kvcachebwd-kw-1 cold-start', prompt=...). "
                "See the bundled workflow specification's Agent naming convention.",
        ))
        # Fail-fast: if naming is missing, block the spawn before running further precondition checks
        reject_and_exit(f"before spawning {agent_type or 'agent'}", rejections)

    # Global invariants always checked
    check_global_invariants(ws, rejections)

    # SC4: eager-merge-pending gate (V3.7.4, 2026-05-02).
    # Reject any Agent spawn when a RECENTLY-ACTIVE workspace has an unmerged
    # knowledge_update.md (>50 chars body, no .kb_merged marker, mtime < 4h).
    # Forces orchestrator to invoke aog-knowledge-maintain Skill BEFORE spawning
    # the next agent — matches the eager-merge protocol in /aog-op-batch SKILL.md §B5.
    # Freshness gate (4h, matches find_active_workspace heuristic) prevents historical
    # workspace pollution from blocking present-session work.
    # Caught when orchestrator was running the multi-op batch (2026-05-02): 2 ops
    # returned with knowledge_update.md ready, orchestrator spawned next agents
    # without merging. User asked "did you update kb in eager mode?" — gap was real.
    import time as _sc4_time
    workspace_root = WORKSPACE_ROOT
    pending_ops = []
    now_ts = _sc4_time.time()
    if workspace_root.exists():
        for ku_path in workspace_root.glob("*/knowledge_update.md"):
            ws_dir = ku_path.parent
            # Skip if this is the workspace we're spawning IN (it's still in flight)
            if ws_dir == ws:
                continue
            # Skip if .kb_merged marker present
            if (ws_dir / ".kb_merged").exists():
                continue
            # Freshness: skip workspaces whose knowledge_update.md is older than 4h
            # (likely historical, never got merged in some prior session — out of
            # scope for this session's eager-merge gate)
            try:
                ku_age_sec = now_ts - ku_path.stat().st_mtime
            except OSError:
                continue
            if ku_age_sec > 4 * 3600:
                continue
            try:
                text = ku_path.read_text(errors="replace")
            except Exception:
                continue
            body_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            body_chars = sum(len(ln) for ln in body_lines)
            if body_chars >= 50:
                pending_ops.append(ws_dir.name)
    if pending_ops:
        rejections.append(Rejection(
            rule_id="SC4",
            description=f"Eager-merge pending: {len(pending_ops)} op workspace(s) have unmerged knowledge_update.md but no .kb_merged marker",
            expected="Run aog-knowledge-maintain Skill on each pending workspace BEFORE spawning next agent (per /aog-op-batch SKILL.md §B5 eager-mode protocol)",
            actual=f"pending: {', '.join(pending_ops)}",
            fix=(
                "For each pending op, run:\n"
                "  python3 src/scripts/workflow/post_agent_return.py --workspace workspace/<op> --handoff '<...>'\n"
                "Then if exit code 3 (KB merge required), invoke:\n"
                "  Skill(name='aog-knowledge-maintain', args='knowledge_update_path=workspace/<op>/knowledge_update.md mode=auto')\n"
                "Marker .kb_merged will be dropped automatically. Then retry this Agent spawn."
            ),
        ))
        # Fail-fast: don't run further checks until eager-merge gap is closed
        reject_and_exit(f"before spawning {agent_type or 'agent'}", rejections)

    # SC5: KB-Manifest-completeness gate for STRUCTURAL_CEILING / PERF_PLATEAU
    # verdicts on A5 ops (V3.7.10, 2026-05-03).
    #
    # Rule: when fused_analysis.md / optimization_log.md / probe_report.md
    # exists in the active workspace AND contains a verdict containing
    # 'STRUCTURAL_CEILING' / 'PERF_PLATEAU' / 'no actionable Kind-1 path' /
    # 'EARLY_EXIT_NO_ACTIONABLE' / 'CONFIRM_STRUCTURAL_CEILING', AND the op's
    # target is 'a5' (Ascend950PR), AND the verdict's evidence implies
    # scalar-pipe-bound bottleneck (mention of 'aiv_scl_ratio' >= 0.3 OR
    # 'scalar' in bottleneck description), THEN the file's `## KB Manifest`
    # / `LOADED` block MUST cite reg-based evaluation (one of: 'OL-54',
    # 'P-REG-1', 'reg-based', 'RegTensor', '__simd_vf__', 'Reg::').
    #
    # This rule was added 2026-05-03 after op#9 9_TopKTopP fo-1 declared
    # "146× analytical structural ceiling" without ever evaluating the A5
    # reg-based primitive (OL-54 + ascend950pr.md §Reg-based were in the
    # KB but the agent's KB Manifest LOADED block didn't include them —
    # soft-prompt compliance failure). User pushback: "knowledge base 完全
    # 没有这方面的指引… pipeline 出了问题，导致已有的知识不能被贯彻". Hard
    # enforcement closes the gap.
    sc5_files = [
        ("fused_analysis.md", "fused-optimizer"),
        ("optimization_log.md", "kernel-optimizer"),
        ("probe_report.md", "precision-probe"),
    ]
    sc5_ceiling_verdicts = re.compile(
        r"\b(STRUCTURAL_CEILING|CONFIRM_STRUCTURAL_CEILING|PERF_PLATEAU|"
        r"EARLY_EXIT_NO_ACTIONABLE_KIND[12]?_PATH|no actionable Kind-1 path|"
        r"DIRECTIVE_NON_VIABLE_HANDOFF_FOR_REVISION|"
        r"PARTIAL_PERF_STRUCTURAL_CEILING|PARTIAL_PERF_HYPOTHESIS_DISPROVEN)\b",
        re.IGNORECASE,
    )
    sc5_regbased_citation = re.compile(
        r"\b(OL-54|P-REG-1|reg-based|reg_based|RegTensor|__simd_vf__|Reg::|"
        r"reg vector compute|asc_vf_call|LoadAlign|StoreAlign)\b",
        re.IGNORECASE,
    )
    sc5_scalar_bound_signal = re.compile(
        r"(aiv_scl(ar)?_ratio[^\n]*[0]\.[3-9]|scalar.{0,15}(pipe|bound|bottleneck)|"
        r"scalar-pipe[- ]bound|scalar 2-pointer|scalar GetValue|scalar SetValue)",
        re.IGNORECASE,
    )

    # Determine if op is on a5 target (defaults to a5; reads workspace/.ascendc_env if available)
    sc5_target_a5 = True  # default
    env_file = workspace_root / ".ascendc_env"
    if env_file.exists():
        try:
            env_text = env_file.read_text()
            tm = re.search(r"^TARGET=(\S+)", env_text, re.MULTILINE)
            if tm:
                sc5_target_a5 = tm.group(1).strip().lower() == "a5"
        except Exception:
            pass

    # SC8: fused_analysis.md cites tuning candidate without producing
    # routing-relevant directive file (V3.7.12, 2026-05-03).
    #
    # Rule: when fused_analysis.md verdict cites "RECOMMEND_KO" / "KIND2_DIRECTIVE for CB-N" /
    # "tuning candidate" AND `optimization_directive.md` AND `optimization_directive_ko_*.md`
    # are BOTH absent, block next spawn. State machine routes based on files, not
    # prose; if fo recommends ko-spawn but doesn't write the trigger file, ko
    # never fires (op#12 fo-1 2026-05-03 evidence — recommendation dropped).
    #
    # P0acr (2026-05-10): exempt aog-cann-learner spawns from SC8. cann-learner
    # is a KB-side meta agent (per P0x v2 carve-out) — it does NOT enter the
    # fused-optimizer → kernel-optimizer state-machine flow. SC8's premise
    # (recommendation will dangle if ko doesn't fire) does not apply to KB
    # learning spawns. Blocking cann-learner via SC8 re-introduces the
    # over-permission pattern user called out 2026-05-10 ("0-interaction skill
    # 变成必须用户指导才能继续").
    if agent_type == "aog-cann-learner":
        sc8_fused_path = None  # skip SC8 entirely
    else:
        sc8_fused_path = ws / "fused_analysis.md"
    if sc8_fused_path is not None and sc8_fused_path.exists():
        try:
            sc8_content = sc8_fused_path.read_text(errors="replace")
            sc8_has_tuning_recommendation = bool(re.search(
                r"(RECOMMEND_KO|KIND2_DIRECTIVE|tuning candidate|CB-\d+\s+\(highest impact\)|CB-\d+:\s+(vectorize|Reg-based|scalar)|RECOMMEND.*kernel-optimizer|aog-kernel-optimizer.*incremental)",
                sc8_content, re.IGNORECASE,
            ))
            sc8_has_directive = (ws / "optimization_directive.md").exists()
            sc8_has_ko_directive = bool(list(ws.glob("optimization_directive_ko_*.md")))
            if sc8_has_tuning_recommendation and not sc8_has_directive and not sc8_has_ko_directive:
                rejections.append(Rejection(
                    rule_id="SC8",
                    description=(
                        "fused_analysis.md recommends incremental tuning (CB-N / RECOMMEND_KO) "
                        "but no optimization_directive_ko_*.md produced — state machine routes "
                        "by file, not prose"
                    ),
                    expected=(
                        "If fo identified a tuning candidate, write "
                        "workspace/<op>/optimization_directive_ko_<N>.md with CB-N fix spec "
                        "(or workspace/<op>/optimization_directive.md if Kind-2 architectural). "
                        "State machine V3.7.12 await_fused_optimizer → await_optimizer requires "
                        "the file artifact."
                    ),
                    actual=(
                        f"workspace/{ws.name}/fused_analysis.md cites tuning candidate but "
                        f"neither optimization_directive.md nor optimization_directive_ko_*.md "
                        f"exists. ko spawn won't trigger."
                    ),
                    fix=(
                        "Either: (a) update fused_analysis.md to remove tuning recommendation "
                        "if not actionable, OR (b) re-spawn fused-optimizer with directive: "
                        "MUST produce optimization_directive_ko_<N>.md when verdict cites "
                        "RECOMMEND_KO / KIND2_DIRECTIVE for CB-N. See aog-fused-optimizer.md "
                        "V3.7.12 file-driven routing section."
                    ),
                ))
        except Exception:
            pass
    if any(r.rule_id == "SC8" for r in rejections):
        reject_and_exit(f"before spawning {agent_type or 'agent'}", rejections)

    if sc5_target_a5:
        for fname, role in sc5_files:
            fpath = ws / fname
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text(errors="replace")
            except Exception:
                continue
            if not sc5_ceiling_verdicts.search(content):
                continue
            if not sc5_scalar_bound_signal.search(content):
                # ceiling claim but not scalar-bound — SC5 doesn't apply (different bottleneck class)
                continue
            if sc5_regbased_citation.search(content):
                continue
            # Found: A5 + scalar-bound + ceiling verdict + NO reg-based citation
            rejections.append(Rejection(
                rule_id="SC5",
                description=(
                    f"{role} declared scalar-pipe-bound ceiling verdict without "
                    f"evaluating reg-based applicability — A5-specific lever missed"
                ),
                expected=(
                    f"workspace/<op>/{fname} must cite OL-54 / P-REG-1 / reg-based / "
                    f"RegTensor / __simd_vf__ / Reg:: in `## KB Manifest LOADED` block "
                    f"or in the verdict explanation when target=a5 AND scalar-pipe is "
                    f"the dominant bottleneck"
                ),
                actual=(
                    f"workspace/{ws.name}/{fname} contains scalar-bound ceiling verdict "
                    f"but NONE of the reg-based citations were found"
                ),
                fix=(
                    f"1. Re-read KB_INDEX.md §By Symptom row for 'scalar-pipe-bound on A5'\n"
                    f"2. Read OPERATIONAL_KNOWLEDGE.md §OL-54 (reg-based SIMD VERIFIED on A5)\n"
                    f"3. Read patterns/unverified/candidates.md §P-REG-1\n"
                    f"4. Read hardware/target/ascend950pr.md §Reg-based vs Mem-based SIMD\n"
                    f"5. Add 'Reg-based applicable: yes/no/needs_probe' line to {fname} "
                    f"with rationale referencing OL-54 + ascend950pr.md\n"
                    f"6. If reg-based IS applicable, write optimization_directive_<next>.md\n"
                    f"   for reg-based rewrite path BEFORE accepting the ceiling verdict"
                ),
            ))
        if any(r.rule_id == "SC5" for r in rejections):
            reject_and_exit(f"before spawning {agent_type or 'agent'}", rejections)

    # Phase-specific: figure out which phase the agent spawn is starting
    # Heuristic: if aog-kernel-worker being spawned for first time, we're entering O4
    # → preconditions O1..O3 must be complete, which means O2.5 artifacts required

    if agent_type in ("aog-kernel-worker", "aog-precision-probe", "aog-kernel-optimizer",
                      "aog-researcher", "aog-fused-optimizer", "aog-determinism-analyzer"):
        # V3.3 (2026-04-23): Phase O4 state machine validation — iter caps + total spawn cap.
        # Reads phase_o4_states section of opgen_state_machine.yaml dynamically.
        # Previously iter caps were enforced only by SKILL.md prose hints — not machine-checked.
        validate_phase_o4_state(ws, agent_type, rejections)

    if rejections:
        reject_and_exit(f"before spawning {agent_type or 'agent'}", rejections)


def mode_post_agent_return(agent_type: str | None, agent_name: str | None = None,
                           run_in_background: bool = False) -> None:
    """After agent returns, validate artifacts + no prohibited side effects.

    DEBT-034 note (2026-04-23): PostToolUse:Agent fires when the Agent tool call
    completes, which for `run_in_background=true` means "spawn confirmed", NOT
    "agent work done". At that instant no artifacts exist yet — firing phase
    O5 checks is a false positive. Additionally, some O5 checks are designed to
    catch "aog-kernel-worker finished with residual but probe was never run"; firing
    those same checks after the probe ITSELF returns creates a chicken-and-egg
    (probe hasn't yet written probe_report.md by the first PostToolUse fire).

    V3.8.2 (DEBT-072, 2026-05-04): when `run_in_background=true`, the Agent tool
    return is just a spawn-ack (no agent work has happened). PostToolUse should
    skip ALL O5 checks in that case — the proper trigger is the task-notification
    arrival when the agent actually completes (orchestrator manually invokes
    post_agent_return.py at that point). Without this skip, every background
    spawn produces 3 spurious O5.I3 rejections against the pre-existing stale
    workspace state.

    V3.8.2 (DEBT-068+071): pass agent_name through for G7-slug workspace
    resolution (multi-op concurrent flows).

    Filters applied:
      - run_in_background=true: skip ALL O5 checks (agent hasn't run yet).
      - agent_type == 'aog-precision-probe': skip O5.C2 (probe is the agent that
        creates probe_report.md; requiring it to already exist is circular).
      - agent_type in ('aog-kernel-optimizer', 'aog-fused-optimizer', 'aog-researcher',
        'aog-determinism-analyzer'): skip O5.C2 (none of these are responsible for
        driving aog-precision-probe dispatch; only aog-kernel-worker ↔ orchestrator is).
    """
    if run_in_background:
        # V3.8.2: spawn-ack, not completion. Skip all post-checks.
        return
    ws = find_active_workspace(agent_name=agent_name)
    if ws is None:
        return
    rejections: list[Rejection] = []
    check_global_invariants(ws, rejections)

    # For aog-kernel-worker: check Phase O4 expectations
    if agent_type == "aog-kernel-worker":
        check_phase_O4_agent_loop(ws, rejections)

    # For verification-completing runs: check Phase O5.
    # But skip when the returning agent is NOT aog-kernel-worker — other agents
    # don't own the aog-precision-probe dispatch decision.
    if (ws / "verification.json").exists() and agent_type in (None, "aog-kernel-worker"):
        check_phase_O5_post_verify(ws, rejections)

    if rejections:
        reject_and_exit(f"after {agent_type or 'agent'} returned", rejections)


def mode_pre_tool_edit(target_path: str | None) -> None:
    """Before Edit tool runs, enforce G1 kernel-edit isolation.

    G1 goal: prevent the orchestrator (top-level CC) from hand-editing kernel source
    and bypassing aog-kernel-worker's Stop-hook (analysis.md consistency, pybind purity,
    static check, verification.json schema, anti-overfitting).

    Historical implementation (broken 2026-04-23): checked `CLAUDE_SPAWNED_AGENT` env
    var, assuming Agent tool propagates it to the spawned subagent. It does NOT —
    the env var is inherited from the parent process (orchestrator), so legitimate
    aog-kernel-worker / aog-kernel-optimizer writes get rejected. op#17 worker succeeded
    only because it ran before this critic rule existed; op#24 worker was blocked
    and had to return text instead of writing files.

    Current implementation (2026-04-23 fix): use marker files only. Orchestrator is
    responsible for creating `.kernel_worker_active` (or `.optimizer_active`) in the
    workspace dir IMMEDIATELY BEFORE spawning the worker/optimizer Agent, and
    removing it after the spawn returns. This is an explicit signal — the
    orchestrator never writes kernels itself, so creating the marker can't accidentally
    enable its own writes; the narrow window during which the marker exists is
    exactly when the spawned agent needs write permission. Outside that window, any
    Edit on workspace/*/kernel/* is rejected (catching direct-orchestrator writes).
    """
    if not target_path:
        return
    p = Path(target_path)
    try:
        rel = p.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return  # outside repo, not our concern
    parts = rel.parts
    # Match workspace/*/kernel/*
    if len(parts) >= 3 and parts[0] == "workspace" and parts[2] == "kernel":
        workspace_dir = REPO_ROOT / parts[0] / parts[1]  # workspace/<op>/
        optimizer_active = (workspace_dir / ".optimizer_active").exists()
        worker_active = (workspace_dir / ".kernel_worker_active").exists()
        if not (worker_active or optimizer_active):
            sys.stderr.write(
                f"❌ workflow_critic: REJECTED Edit on {target_path}\n"
                f"  - rule G1: Orchestrator must NOT Edit workspace/*/kernel/*\n"
                f"    expected: kernel edits only via aog-kernel-worker or aog-kernel-optimizer agent\n"
                f"    actual:   no marker file present in {workspace_dir}/\n"
                f"    fix (if you are orchestrator): touch {workspace_dir}/.kernel_worker_active\n"
                f"              immediately before spawning aog-kernel-worker, rm after it returns\n"
                f"              (or .optimizer_active for aog-kernel-optimizer).\n"
                f"    fix (if you are the spawned worker and see this): orchestrator forgot to\n"
                f"              create the marker — ask orchestrator to `touch` the marker then retry.\n"
            )
            sys.exit(2)


def _absolutize_staged_paths(git_root: Path, names: list[str]) -> set[Path]:
    """Map git's repo-relative staged names to stable absolute paths.

    ``git -C <plugin>/engine diff --name-only`` still emits paths relative to
    the containing git worktree.  In the cannbot monorepo that means a full
    ``plugins-community/ascendc-port-orchestrator/...`` prefix; in a standalone
    plugin checkout it means ``workflows/...`` or ``engine/output/...``.  Path
    identity, rather than string-prefix guessing, supports both layouts.
    """
    root = git_root.resolve()
    return {
        (Path(name) if Path(name).is_absolute() else root / name).resolve()
        for name in names
        if name
    }


def _read_staged_paths() -> set[Path]:
    """Return staged paths as absolute filesystem paths, or raise on git errors."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("git executable is unavailable")
    root_result = subprocess.run(
        [git_executable, "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    diff_result = subprocess.run(
        [git_executable, "-C", str(REPO_ROOT), "diff", "--cached", "--no-renames", "--name-only"],
        capture_output=True, text=True, check=True,
    )
    root_text = root_result.stdout.strip()
    if not root_text:
        raise RuntimeError("git rev-parse returned an empty worktree root")
    git_root = Path(root_text)
    return _absolutize_staged_paths(git_root, diff_result.stdout.splitlines())


def mode_pre_commit_sync() -> None:
    """Before git commit, check SKILL.md ↔ YAML sync, and enforce aog-self-critic
    artifact presence for any op-gen archive being committed (V3.3.2)."""
    try:
        staged = _read_staged_paths()
    except Exception:
        return  # fail open

    yaml_path = YAML_PATH.resolve()
    entry_paths = tuple(path.resolve() for path in ENTRY_SKILL_PATHS)
    yaml_changed = yaml_path in staged
    changed_skills = [path for path in entry_paths if path in staged]
    skill_changed = bool(changed_skills)
    if skill_changed != yaml_changed:
        yaml_rel = YAML_PATH.relative_to(PLUGIN_ROOT).as_posix()
        skill_rels = [path.relative_to(PLUGIN_ROOT).as_posix() for path in ENTRY_SKILL_PATHS]
        changed_rels = [path.relative_to(PLUGIN_ROOT.resolve()).as_posix()
                        for path in changed_skills]
        expected = f"{yaml_rel} and at least one of {skill_rels}, or neither"
        actual = (f"entry skill(s) {changed_rels} without {yaml_rel}"
                  if skill_changed else f"{yaml_rel} without an entry skill")
        sys.stderr.write(
            f"❌ workflow_critic: REJECTED git commit\n"
            f"  - rule DRIFT: entry Skill and workflow YAML must change together\n"
            f"    expected: {expected}\n"
            f"    actual:   {actual}\n"
            f"    fix:      update the matching workflow/entry documentation in the same commit\n"
        )
        sys.exit(2)

    # V3.3.2 (2026-04-25): SC1 aog-self-critic-required gate.
    # Any commit that adds/modifies a kernel archive at
    # output/<project>/src/kernels/<op>/ MUST include or have already committed
    # a corresponding self_critic_report.md. Caught op#12 + op#25 retroactively;
    # going forward, this gate prevents commit of an op archive without the
    # mandatory C1-C18 audit (specifically C18 = delegation cheating scan).
    archive_paths = set()
    for path in staged:
        try:
            rel = path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            continue
        # Match: output/<project>/src/kernels/<op>/<file>
        parts = rel.parts
        if (len(parts) >= 5 and parts[0] == "output" and parts[2] == "src"
                and parts[3] == "kernels"):
            op_root = Path(*parts[:5])  # output/<project>/src/kernels/<op>
            archive_paths.add(op_root)
    missing = []
    for ar in archive_paths:
        ar_path = REPO_ROOT / ar
        if not ar_path.is_dir():
            continue
        # Accept at archive root OR at .harness/ subdir (2026-05-15 archive
        # layout reorg moved harness internals into .harness/ — customer files
        # stay at root; self_critic_report.md is harness-internal).
        sc_report = ar_path / "self_critic_report.md"
        sc_report_harness = ar_path / ".harness" / "self_critic_report.md"
        if not (sc_report.exists() or sc_report_harness.exists()):
            missing.append(ar.as_posix())
    if missing:
        sys.stderr.write(
            "❌ workflow_critic: REJECTED git commit\n"
            "  - rule SC1 (V3.3.2): kernel archive committed without self_critic_report.md\n"
            "    expected: each op archive at output/<project>/src/kernels/<op>/ contains self_critic_report.md\n"
            "    actual:   missing in:\n"
            + "".join(f"              - {ar}/self_critic_report.md\n" for ar in missing)
            + "    fix: invoke /aog-self-critic skill on the workspace before archiving:\n"
              "         Skill(name='aog-self-critic', args='audit op#X delegation+claims+residuals')\n"
              "         Skill writes workspace/{op}/self_critic_report.md → copy into archive\n"
              "         (also runs scan_delegation_cheating.py — catches torch_npu/aclnn fallback in cand wrapper).\n"
              "         If you are committing a fix to an existing archive, the file must already be there OR be in this commit.\n"
              "         Triggered by op#12 + op#25 delegation cheats slipping through 2 sessions undetected (2026-04-25).\n"
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Hook-input parsing (Claude Code passes JSON via stdin)
# ---------------------------------------------------------------------------
def read_hook_input() -> dict:
    """Parse Claude Code hook JSON input from stdin.
    Format (PreToolUse/PostToolUse):
        {"tool_name": "Agent|Edit|Bash|...", "tool_input": {...}, ...}
    Returns empty dict if stdin is empty or not JSON (runs via CLI for testing).
    """
    if sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # --mode is optional: if hook input has tool_name, infer mode
    ap.add_argument("--mode",
                    choices=["pre_agent_spawn", "post_agent_return", "pre_tool_edit",
                              "pre_commit_sync", "auto"],
                    default="auto")
    ap.add_argument("--agent-type", help="agent being spawned (override)")
    ap.add_argument("--target-path", help="Edit target path (override)")
    ap.add_argument("--hook-event", help="hook event (PreToolUse / PostToolUse / Stop) — from env")
    args = ap.parse_args()

    # Parse Claude Code hook stdin input (if present)
    hook = read_hook_input()
    tool_name = hook.get("tool_name") or ""
    tool_input = hook.get("tool_input") or {}
    hook_event = args.hook_event or hook.get("hook_event_name") or ""

    # Resolve mode: explicit flag > inferred from hook context > error
    mode = args.mode
    if mode == "auto":
        if tool_name == "Agent" and hook_event.lower().startswith("pre"):
            mode = "pre_agent_spawn"
        elif tool_name == "Agent" and hook_event.lower().startswith("post"):
            mode = "post_agent_return"
        elif tool_name in ("Edit", "Write", "MultiEdit") and hook_event.lower().startswith("pre"):
            mode = "pre_tool_edit"
        elif tool_name == "Bash" and hook_event.lower().startswith("pre"):
            cmd = tool_input.get("command", "")
            if "git commit" in cmd:
                mode = "pre_commit_sync"
            else:
                return  # irrelevant bash command — fail open
        else:
            return  # can't infer — fail open

    if mode == "pre_agent_spawn":
        agent_type = args.agent_type or tool_input.get("subagent_type")
        agent_name = _extract_agent_name(tool_input)  # G7: read from name= or description=
        mode_pre_agent_spawn(agent_type, agent_name)
    elif mode == "post_agent_return":
        agent_type = args.agent_type or tool_input.get("subagent_type")
        agent_name = _extract_agent_name(tool_input)  # V3.8.2: G7 slug for workspace resolution
        run_bg = bool(tool_input.get("run_in_background", False))
        mode_post_agent_return(agent_type, agent_name=agent_name, run_in_background=run_bg)
    elif mode == "pre_tool_edit":
        target = args.target_path or tool_input.get("file_path") or tool_input.get("path")
        mode_pre_tool_edit(target)
    elif mode == "pre_commit_sync":
        mode_pre_commit_sync()


if __name__ == "__main__":
    main()
