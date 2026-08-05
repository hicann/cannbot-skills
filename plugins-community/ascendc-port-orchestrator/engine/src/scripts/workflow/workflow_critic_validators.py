#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""workflow_critic_validators — the check_*/validate gate-check cluster, extracted from
workflow_critic.py (behavior-neutral, 2026-07-05). Imports shared symbols from the LEAF
(workflow_critic_common), NOT from workflow_critic (which is a __main__ script-hook)."""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

from workflow_critic_common import (
    Rejection, load_state_machine,
    read_active_target, SIMT_PRIMITIVES, _C_COMMENT_RE,
)


def check_file_exists(path: Path, rule_id: str, description: str) -> Rejection | None:
    if path.exists():
        return None
    return Rejection(
        rule_id=rule_id, description=description,
        expected=f"{path} exists",
        actual=f"{path} MISSING",
        fix=f"produce {path.name} per phase spec",
    )


def check_grep_must_contain(path: Path, patterns: list[str], rule_id: str) -> Rejection | None:
    if not path.exists():
        return None  # caller already checked existence
    content = path.read_text()
    missing = [p for p in patterns if not re.search(p, content, re.MULTILINE)]
    if not missing:
        return None
    return Rejection(
        rule_id=rule_id, description=f"required patterns in {path.name}",
        expected=f"file contains: {missing}",
        actual=f"file does NOT contain: {missing}",
        fix=f"add the required content to {path.name} or fix the generator",
    )


def check_grep_must_not_contain(path: Path, patterns: list[str], rule_id: str) -> Rejection | None:
    if not path.exists():
        return None
    content = path.read_text()
    hits = [p for p in patterns if re.search(p, content, re.MULTILINE)]
    if not hits:
        return None
    return Rejection(
        rule_id=rule_id, description=f"forbidden patterns in {path.name}",
        expected=f"file must NOT contain: {hits}",
        actual=f"file contains: {hits}",
        fix=f"remove forbidden patterns from {path.name}",
    )


def check_json_keys(path: Path, keys: list[str], rule_id: str) -> Rejection | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as e:
        return Rejection(
            rule_id=rule_id, description=f"{path.name} valid JSON",
            expected="parseable JSON",
            actual=f"JSON parse error: {e}",
            fix=f"fix JSON syntax in {path.name}",
        )
    missing = [k for k in keys if k not in data]
    if not missing:
        return None
    return Rejection(
        rule_id=rule_id, description=f"required keys in {path.name}",
        expected=f"keys present: {keys}",
        actual=f"missing keys: {missing}",
        fix=f"add missing keys to {path.name}",
    )


def _strip_c_comments(src: str) -> str:
    """Remove C/C++ comments so substring scans don't false-positive on
    explanatory comments like `// KERNEL_TASK_TYPE_DEFAULT removed for V220`.
    """
    return _C_COMMENT_RE.sub('', src)


def check_target_simt_compat(ws: Path, rejections: list[Rejection]) -> None:
    """G7-target (V3.4): when TARGET ∈ {a3, a2}, kernel/* MUST NOT contain SIMT primitives.

    Run every PreToolUse on Edit and at agent-spawn time. The aog-kernel-worker.md
    Phase 0 instructs the agent to refuse such writes; this critic is a backstop
    in case the agent forgets, OR an Edit hits the workspace from somewhere else.

    Comments are stripped before the substring scan so explanatory mentions like
    `// KERNEL_TASK_TYPE_DEFAULT removed — arch35-only` do not trigger a false
    positive (V3.4.1, 2026-04-25 — caught on op gelu-kw-1).
    """
    target, platform_simt = read_active_target(ws)
    if platform_simt:
        return  # a5 supports SIMT; nothing to enforce here
    kernel_dir = ws / "kernel"
    if not kernel_dir.is_dir():
        return
    for f in kernel_dir.glob("*"):
        if f.suffix not in (".h", ".cpp", ".cc", ".hpp"):
            continue
        try:
            content = _strip_c_comments(f.read_text())
        except Exception:
            continue
        for token in SIMT_PRIMITIVES:
            if token in content:
                rejections.append(Rejection(
                    rule_id="G7-target",
                    description=f"SIMT primitive '{token}' on TARGET={target} (V220 / arch22 has no SIMT path)",
                    expected=f"kernel/* must NOT use {token} when TARGET={target}",
                    actual=f"{f.relative_to(ws)} contains '{token}' (outside comments)",
                    fix="rewrite using SIMD pattern (TPipe / TQue / DataCopy + UB-scratchpad reduction). "
                        "See merged_skills/_kb/references/hardware/INDEX.md §Capability matrix and "
                        "merged_skills/_kb/references/target/ascendc/patterns/domains/scatter_add.md §a3/a2 catalogue gap.",
                ))
                break  # one rejection per file is enough


def check_global_invariants(ws: Path, rejections: list[Rejection]) -> None:
    """Apply global_invariants to workspace."""
    # G7-target (V3.4): SIMT-rejection on a3/a2 targets (BACKSTOP — aog-kernel-worker
    # already enforces this at agent level)
    check_target_simt_compat(ws, rejections)

    # G8 (P0aaj, 2026-05-06): taxonomy coverage gate — warn when an op resolves
    # to is_untagged_fallback=True despite having source files. C34 origin:
    # silent KB-load regression for newly named ops. If source exists,
    # auto-tag inference SHOULD have produced ≥1 tag; empty tags means the
    # source-scan signature catalog has a hole.
    check_taxonomy_coverage(ws, rejections)

    # G2: no pytorch_native_*.py in workspace (only allowed in archive)
    for f in ws.glob("pytorch_native_*.py"):
        rejections.append(Rejection(
            rule_id="G2", description="No pytorch_native_<op>.py in workspace",
            expected="workspace has no pytorch_native_*.py files",
            actual=f"found {f}",
            fix="delete file; use fresh source-NPU truth for migration or fp64 autograd truth for backward generation",
        ))

    # G3 (RETIRED 2026-04-26 per ascendc-op-gen SKILL.md "soft guidance"; reaffirmed
    # 2026-05-01 per user "停止这种针对词不看上下问的拦截"). Substring banning produces
    # too many false positives on legitimate technical narrative ("bit drift" for
    # accumulation-order discussions, "drift" in feedback-rule names like "C1 priority
    # drift", "oracle" as a verb meta-discussion). Semantic concern (importing
    # /aog-regression-check vocabulary semantics into new-port context) is now caught
    # by aog-self-critic C6 (jargon creep) which has full-context reasoning.
    #
    # Block intentionally left as no-op. Removing the block entirely would change rule
    # numbering for downstream rule-ID consumers; keeping the slot reserves "G3" for
    # potential future re-implementation as a context-aware check.
    pass


def check_taxonomy_coverage(ws: Path, rejections: list[Rejection]) -> None:
    """G8 (P0aak, 2026-05-07): catch missing op_classification.json for ops with source files.

    C34 origin: name-keyed `OP_TAGS` dict caused silent KB-load regression
    for newly named ops. v3 fix: `phase_o17_classify` runs `/aog-op-classify`
    skill in subprocess, writes `workspace/<op>/op_classification.json`. Brief
    construction reads that JSON to extend KB load.

    This gate fires when:
        - workspace has source files (model.py / kernel/*.h) AND
        - op_classification.json is missing AND
        - op_classification.error is missing (i.e. classification was never attempted,
          not just failed)
    Signal: Phase O1.7 didn't run, OR source files arrived after classification cache.
    Worker brief will only get DEFAULT_KB_SECTIONS (less than full coverage).
    """
    op = ws.name
    has_source = any([
        (ws / "model.py").exists(),
        (ws / "model_new_ascendc.py").exists(),
        any((ws / "kernel").glob("*.h")) if (ws / "kernel").exists() else False,
        any((ws / "kernel").glob("*.cpp")) if (ws / "kernel").exists() else False,
    ])
    if not has_source:
        return  # no source to classify — fine
    cls_json = ws / "op_classification.json"
    cls_error = ws / "op_classification.error"
    if cls_json.exists() or cls_error.exists():
        return  # classification attempted (success or recorded failure)
    rejections.append(Rejection(
        rule_id="G8",
        description=("op_classification.json missing despite source files — "
                     "Phase O1.7 (/aog-op-classify) likely didn't run"),
        expected=f"workspace/{op}/op_classification.json (or .error if skill failed)",
        actual=f"neither op_classification.json nor op_classification.error in {ws}",
        fix=(f"Run Phase O1.7 manually: python3 -c 'import sys; "
             f"sys.path.insert(0, \"src/scripts/orchestrator\"); "
             f"import phase_o17_classify; "
             f"print(phase_o17_classify.classify(__import__(\"pathlib\").Path(\"{ws}\")))'. "
             f"Without classification, worker brief gets DEFAULT_KB_SECTIONS only — "
             f"silent KB-load regression (C34, P0aak)."),
    ))


def check_phase_O4_agent_loop(ws: Path, rejections: list[Rejection]) -> None:
    """Phase O4: worker loop invariants."""
    # G1: orchestrator must not have edited kernel/* directly
    # Heuristic: if git log shows kernel/ files modified by commits whose message
    # lacks 'aog-kernel-worker' / 'worker' mention, flag as suspicious.
    # For v0 we only check: if kernel/ files exist but no analysis.md, skipped Phase A
    kernel_dir = ws / "kernel"
    if kernel_dir.exists():
        analysis = ws / "analysis.md"
        if not analysis.exists():
            rejections.append(Rejection(
                rule_id="O4.pre1",
                description="kernel/ exists but analysis.md missing — worker skipped Phase A",
                expected=f"{ws}/analysis.md produced by aog-kernel-worker Phase A",
                actual="analysis.md absent",
                fix="worker must produce analysis.md in Phase A before generating kernel (or Edit was done outside worker — G1 violation)",
            ))


def validate_phase_o4_state(ws: Path, agent_spawning: str | None, rejections: list[Rejection]) -> None:
    """Pre-spawn check: is spawning `agent_spawning` a valid transition from current state?

    State is inferred from state_transitions.jsonl (primary, V3.3+) or PROGRESS.md
    (fallback). Iter counts accumulated from log.

    V3.3 (2026-04-23): when state_transitions.jsonl exists, run state_machine.verify
    first — this catches illegal historical transitions (orchestrator bypassing YAML).
    """
    if agent_spawning is None:
        return

    # V3.3 addition: verify state_transitions.jsonl history is legal per YAML
    sm_script = Path(__file__).resolve().parent / "state_machine.py"
    log_file = ws / "state_transitions.jsonl"
    if log_file.exists() and sm_script.exists():
        result = subprocess.run(
            [sys.executable, str(sm_script), "verify", "--workspace", str(ws)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Parse violations from stderr
            for line in result.stderr.splitlines():
                if line.strip().startswith("-"):
                    rejections.append(Rejection(
                        rule_id="O4.state_log",
                        description="state_transitions.jsonl violates YAML phase_o4_states",
                        expected="all logged transitions declared in YAML exit_transitions + iter caps respected",
                        actual=line.strip(),
                        fix="Orchestrator must invoke `state_machine.py next` to decide transitions, not hand-route. Check SKILL.md §Phase O4 for mandatory invocation protocol.",
                    ))

    sm = load_state_machine()
    states = {s["id"]: s for s in sm.get("phase_o4_states", [])}
    if not states:
        return  # no spec → no check (fail-open)

    # Map agent_spawning → expected state id
    agent_to_state = {
        "aog-kernel-worker": "await_worker",
        "aog-precision-probe": "await_probe",
        "aog-kernel-optimizer": "await_optimizer",
        "aog-fused-optimizer": "await_fused_optimizer",
        "aog-researcher": "await_researcher",
        "aog-determinism-analyzer": "await_det_analyzer",
    }
    target_state = agent_to_state.get(agent_spawning)
    if target_state is None or target_state not in states:
        return

    # V3.3 DEBT-045 fix (2026-04-23): verify that current_state allows spawning
    # this agent (target_state). Previously only iter_cap was checked, so an
    # orchestrator could call Agent(aog-kernel-optimizer) without a handoff line
    # that authorized the transition, and critic would silently accept.
    #
    # current_state derivation:
    #   - If state_transitions.jsonl exists and is non-empty: last entry's to_state
    #   - Else: phase_o4_initial_state from YAML (typically await_worker)
    #
    # Valid spawn iff:
    #   - target_state == current_state (iter within same state, capped separately), OR
    #   - target_state appears as `goto:` in current_state's exit_transitions list
    #
    # This closes the DEBT-045 hole: a newly-initialized workspace allows only
    # the initial agent (aog-kernel-worker). Off-state spawns are rejected with a
    # clear message pointing the orchestrator at `state_machine.py next`.
    # V3.3 DEBT-045 fix (2026-04-23): initial_state is mode-dependent.
    #   migration / backward → await_worker (generate from source contract)
    #   optimize            → await_optimizer (tune already-DONE kernel)
    # MODE is parsed from workspace/{op}/PROGRESS.md header line "Mode: <name>".
    mode = None
    progress = ws / "PROGRESS.md"
    if progress.exists():
        for line in progress.read_text().split("\n")[:15]:  # header only
            s = line.strip()
            if s.lower().startswith("mode:"):
                mode = s.split(":", 1)[1].strip().split()[0].lower()
                break
    mode_map = sm.get("phase_o4_initial_state_by_mode", {}) or {}
    initial_state = mode_map.get(mode) or sm.get("phase_o4_initial_state", "await_worker")

    log_has_entries = False
    log_tail_state = None
    if log_file.exists():
        try:
            lines = [
                line for line in log_file.read_text().splitlines() if line.strip()
            ]
            if lines:
                log_has_entries = True
                import json as _json
                last = _json.loads(lines[-1])
                log_tail_state = last.get("to_state") or last.get("next_state")
        except Exception:
            pass

    # Case A — terminal state in log: no further spawns
    if log_tail_state in ("finalize", "abort"):
        rejections.append(Rejection(
            rule_id="O4.state.terminal",
            description=f"spawn {agent_spawning} blocked — last state in log is terminal ({log_tail_state})",
            expected="non-terminal current state",
            actual=f"state_transitions.jsonl tail state = {log_tail_state}",
            fix=(
                f"This workspace's state machine has already reached {log_tail_state}. "
                f"Spawning {agent_spawning} would re-open a completed op. If a new pass "
                f"is intended, reset workspace (remove state_transitions.jsonl) + call "
                f"`state_machine.py next` from {initial_state}."
            ),
        ))
        return

    # Case B — empty log: only the initial-state's agent may spawn (bootstrap)
    if not log_has_entries:
        if target_state != initial_state:
            rejections.append(Rejection(
                rule_id="O4.state.no_log",
                description=(
                    f"spawn {agent_spawning} blocked — state_transitions.jsonl is empty "
                    f"(fresh workspace)"
                ),
                expected=(
                    f"initial spawn must enter initial_state={initial_state} "
                    f"(agent={states[initial_state].get('agent')})"
                ),
                actual=f"attempting to spawn {agent_spawning} → {target_state}",
                fix=(
                    f"On a fresh workspace the only legal first spawn is the initial "
                    f"state's agent ({states[initial_state].get('agent')}). "
                    f"Spawning {agent_spawning} directly bypasses the state machine. "
                    f"If this workspace does have prior state that was lost, recover "
                    f"state_transitions.jsonl or call `state_machine.py next` explicitly "
                    f"to record the transition you believe authorizes this spawn. "
                    f"(DEBT-045 fix, 2026-04-23)"
                ),
            ))
            return
        # fall through — initial-state spawn is allowed with no further check

    # Case C — log has entries: target_state must match log tail OR be a declared goto
    else:
        current_state = log_tail_state if log_tail_state in states else initial_state
        cur_spec = states.get(current_state, {})
        allowed_gotos = set()
        for trans in cur_spec.get("exit_transitions", []):
            goto = trans.get("goto")
            if goto:
                allowed_gotos.add(goto)
        allowed_gotos.add(current_state)  # self-transition (iter loop) always allowed up to cap

        if target_state not in allowed_gotos:
            rejections.append(Rejection(
                rule_id="O4.state.transition",
                description=(
                    f"spawn {agent_spawning} (→ {target_state}) is not a legal transition "
                    f"from current_state {current_state}"
                ),
                expected=(
                    f"target_state ∈ {sorted(allowed_gotos)} given "
                    f"current_state={current_state}"
                ),
                actual=f"spawning {agent_spawning} would set target_state={target_state}",
                fix=(
                    "Orchestrator must call `python3 src/scripts/workflow/state_machine.py "
                    "next --workspace <ws> --handoff <handoff-line>` first — the script "
                    "will either authorize the transition (appending to "
                    "state_transitions.jsonl) or reject it with a YAML-grounded reason. "
                    "(DEBT-045 fix, 2026-04-23)"
                ),
            ))
            return

    # Count spawns per agent. Prefer state_transitions.jsonl when present
    # (authoritative — each entry is one programmatic transition). Fall back to
    # a PROGRESS.md heuristic that counts spawn-start markers only (not every
    # phase entry, which would over-count after DEBT-046 added per-phase logs).
    iter_counts: dict[str, int] = {}
    if log_has_entries:
        # Count to_state transitions in state_transitions.jsonl (excludes "finalize"/"abort").
        try:
            import json as _json
            for line in log_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                to_s = entry.get("to_state") or entry.get("next_state")
                # Map state id → iter_counter from YAML
                for sid, sspec in states.items():
                    if sid == to_s:
                        counter = sspec.get("iter_counter", sid)
                        iter_counts[counter] = iter_counts.get(counter, 0) + 1
                        break
        except Exception:
            pass

    if not iter_counts:
        # PROGRESS.md heuristic — match spawn-start markers, not every phase log.
        # A aog-kernel-worker spawn writes ONE "Phase A" line (its first action);
        # aog-kernel-optimizer writes ONE "spawn" or "Opt0 baseline" line; probe/researcher/
        # det-analyzer write "Iter 0" or similar first-iter marker.
        # To be conservative and robust across all agents, count distinct agent sections
        # that include the substring "(Phase A)" for worker, or the bare agent name on
        # a spawn-init line like "aog-kernel-optimizer — spawn" / "Opt0 baseline" for optimizer.
        progress = ws / "PROGRESS.md"
        if progress.exists():
            prog_text = progress.read_text()
            import re as _re
            # aog-kernel-worker spawns: one "(Phase A)" per spawn
            n_worker = len(_re.findall(r"^### \[[^\]]+\] aog-kernel-worker \(Phase A\)",
                                        prog_text, _re.M))
            if n_worker:
                iter_counts["worker"] = n_worker
            # aog-kernel-optimizer / aog-fused-optimizer / probe / researcher / det-analyzer
            # spawns: one "— spawn" OR "Opt0 baseline" OR "Iter 0" entry per spawn
            for a_name, s_id in agent_to_state.items():
                if a_name == "aog-kernel-worker":
                    continue  # handled above
                counter = states.get(s_id, {}).get("iter_counter", s_id)
                pat = rf"^### \[[^\]]+\] {_re.escape(a_name)}.*(?:spawn|— spawn|Opt0 baseline|Iter 0)"
                n = len(_re.findall(pat, prog_text, _re.M))
                if n:
                    iter_counts[counter] = n

    # iter_cap check
    target_spec = states[target_state]
    counter = target_spec.get("iter_counter", target_state)
    cap = target_spec.get("iter_cap", 999)
    current = iter_counts.get(counter, 0)
    if current >= cap:
        rejections.append(Rejection(
            rule_id="O4.state.iter_cap",
            description=f"iter cap exceeded for state {target_state}",
            expected=f"iter.{counter} < {cap}",
            actual=f"iter.{counter} = {current} (would become {current+1})",
            fix=f"State {target_state} has iter_cap={cap}; further spawns require state transition (check phase_o4_states exit_transitions in opgen_state_machine.yaml).",
        ))

    # Total spawn cap
    total_cap = sm.get("phase_o4_total_spawn_cap", 15)
    total_spawned = sum(iter_counts.values())
    if total_spawned >= total_cap:
        rejections.append(Rejection(
            rule_id="O4.state.total_cap",
            description="total spawn cap exceeded",
            expected=f"total agent spawns < {total_cap}",
            actual=f"total = {total_spawned}",
            fix="Hard safety fuse — orchestrator should declare ABORT or finalize rather than further spawns.",
        ))


def _evaluate_conditional_phases(ws: Path, vj_data: dict, rejections: list[Rejection]) -> None:
    """Read YAML `conditional_phases` section and evaluate each entry against current
    workspace state. Each entry declares its own trigger conditions + required artifact.

    This is the YAML-wins migration of what used to be hard-coded Python checks (O5.C3
    aog-fused-optimizer escalation, 2026-04-23). Future conditional components should be
    added as YAML entries rather than new Python functions — keeps state machine spec
    declarative and co-located with other workflow rules.

    Supported trigger primitives:
      - path_exists: <relative-path-with-{op}-placeholder>
      - analysis_md_contains_any: [str, ...]  (case-insensitive substring match)
      - verification_json_perf_below_threshold: true|false
        (evaluates data.performance.ratio or .sum_ratio against .threshold or 0.6 default)

    Grammar is small on purpose. When a new trigger is needed, add a primitive here
    and a small test. Don't grow the grammar speculatively.
    """
    sm = load_state_machine()
    entries = sm.get("conditional_phases", [])
    if not entries:
        return

    perf = vj_data.get("performance", {})
    perf_ratio = perf.get("ratio") or perf.get("sum_ratio")
    perf_threshold = perf.get("threshold", 0.6)
    try:
        perf_below_target = (
            perf_ratio is not None and float(perf_ratio) < float(perf_threshold)
        )
    except Exception:
        perf_below_target = False

    analysis_md = ws / "analysis.md"
    analysis_lower = analysis_md.read_text().lower() if analysis_md.exists() else ""

    def _eval_trigger(trigger: dict) -> bool:
        """AND semantics across all_of; returns True iff every condition evaluates True."""
        conds = trigger.get("all_of", [])
        for c in conds:
            (kind, arg), = c.items()
            if kind == "path_exists":
                path = ws / arg.replace("workspace/{op}/", "")
                if not path.exists():
                    return False
            elif kind == "analysis_md_contains_any":
                if not any(kw.lower() in analysis_lower for kw in arg):
                    return False
            elif kind == "verification_json_perf_below_threshold":
                if bool(arg) != perf_below_target:
                    return False
            else:
                sys.stderr.write(f"workflow_critic: unknown conditional trigger primitive '{kind}'\n")
                return False
        return True

    for entry in entries:
        try:
            if not _eval_trigger(entry.get("trigger", {})):
                continue
            req_artifact = entry.get("required_artifact", "")
            req_path = ws / req_artifact.replace("workspace/{op}/", "")
            if req_path.exists():
                continue
            rule_id = entry.get("rule_id", entry.get("id", "conditional"))
            desc = entry.get("description", f"conditional phase {entry.get('id')} triggered but required artifact missing")
            soft = bool(entry.get("soft_fail", False))
            fix = entry.get("fix", "spawn the declared agent or document why conditional phase N/A")
            if soft:
                # Warn-only: emit to stderr, don't push to rejections (which would block).
                sys.stderr.write(f"⚠️ workflow_critic: conditional phase warning [{rule_id}] {desc} — {fix}\n")
            else:
                rejections.append(Rejection(
                    rule_id=rule_id, description=desc,
                    expected=f"{req_artifact} exists after conditional phase '{entry.get('id')}' triggered",
                    actual=f"{req_artifact} missing",
                    fix=fix,
                ))
        except Exception as e:
            sys.stderr.write(f"workflow_critic: conditional_phases entry {entry.get('id')!r} eval error: {e}\n")


def check_phase_O5_post_verify(ws: Path, rejections: list[Rejection]) -> None:
    """Phase O5: verification.json schema + invariants."""
    vjson = ws / "verification.json"
    r = check_file_exists(vjson, "O5.art1", "verification.json required for Phase O5")
    if r:
        rejections.append(r)
        return

    r = check_json_keys(vjson, ["precision", "determinism", "performance"], "O5.inv1")
    if r:
        rejections.append(r)
        return

    # V3.3.2 (2026-04-25): aog-self-critic skill output is a mandatory Phase O5
    # artifact, BUT we don't enforce its presence here at every worker return —
    # the orchestrator needs a window to invoke the skill after worker returns
    # but before commit. Enforcement of self_critic_report.md presence happens
    # in `mode_pre_commit_sync` (pre-commit gate) instead — see SC1 there.

    try:
        with vjson.open() as f:
            data = json.load(f)
    except Exception:
        return
    prec = data.get("precision", {})

    # O5.C1 (2026-04-22, caught by user on op#16): precision.status MUST be in canonical
    # set. Worker inventing "PASS_WITH_KNOWN_LIMIT" (or similar non-standard strings) is a
    # reward-hacking surface — claims "PASS" semantics without meeting the bar and without
    # escalation. Hard reject.
    CANONICAL_STATUSES = {"PASS", "PASS_WITHIN_TOLERANCE", "PARTIAL_PASS", "PARTIAL", "FAIL"}
    status = prec.get("status")
    if status is not None and status not in CANONICAL_STATUSES:
        rejections.append(Rejection(
            rule_id="O5.C1",
            description="precision.status must be in canonical set",
            expected=f"precision.status ∈ {sorted(CANONICAL_STATUSES)}",
            actual=f"precision.status = {status!r} (non-canonical)",
            fix=(
                "worker must not invent status strings. Use PARTIAL_PASS for N<M with residual failures "
                "(and escalate to aog-precision-probe) or PASS only for full match. Non-canonical statuses "
                "are reward-hacking."
            ),
        ))

    # O5.C2 (2026-04-22, caught by user on op#16): if precision has residual failures
    # (i.e. status=PARTIAL_PASS/PARTIAL OR pass<total), a probe_report.md is REQUIRED to
    # classify the residual — orchestrator must have spawned aog-precision-probe before
    # archiving. Blocks the "worker says done with partial, orchestrator just accepts"
    # anti-pattern. Exception: probe_report.md exists AND contains classification.
    precision_partial = (
        status in ("PARTIAL_PASS", "PARTIAL")
        or (status == "PASS" and prec.get("pass", 0) != prec.get("total", 0))
        or prec.get("failing_cases")
        or prec.get("failing_edge_cases")
    )
    if precision_partial:
        probe = ws / "probe_report.md"
        if not probe.exists():
            rejections.append(Rejection(
                rule_id="O5.C2",
                description="precision partial/residual but no probe_report.md — probe not dispatched",
                expected="probe_report.md exists with §Recommendation (convention/requirement classification)",
                actual="probe_report.md absent",
                fix="orchestrator must spawn aog-precision-probe BEFORE accepting worker's done handoff when precision has any residual failure. Probe classifies residual as convention (fixable via Kind-1 respawn) or requirement (logged as architectural DEBT with evidence).",
            ))

    # O5.C4 (2026-05-04, DS session): when precision.status == PASS or PASS_WITHIN_TOLERANCE,
    # performance.ratio MUST be populated (not null, not 0, not missing). Probe agents and
    # Kind-1 fix workers often achieve PASS but skip perf measurement — the orchestrator
    # must re-measure independently per Phase O5. This gate blocks archive commits that
    # claim PASS without perf data.
    perf = data.get("performance", {})
    perf_ratio = perf.get("ratio") or perf.get("sum_ratio")
    if status in ("PASS", "PASS_WITHIN_TOLERANCE") and (perf_ratio is None or perf_ratio == 0):
        rejections.append(Rejection(
            rule_id="O5.C4",
            description="precision PASS but performance.ratio missing/null/zero",
            expected="performance.ratio must be populated (≥ 0.01) from independent Phase O5 re-measurement with EC-33 methodology",
            actual=f"performance.ratio = {perf_ratio!r}",
            fix="Run Phase O5: deploy kernel, run utils/performance.py --output_dir current_task --warmup 1 --repeats 2, write ratio to verification.json. Do NOT accept worker self-reported perf without independent re-measurement.",
        ))

    if precision_partial:
        # O5.probe_requirement_evidence (V3.4.2, 2026-04-26 — aog-self-critic C20):
        # If probe verdict=requirement (i.e. proposes waiver/PARTIAL acceptance),
        # there MUST be observational evidence in probes/probe_outputs/. Specifically:
        # at least one of {msprof_on_reference, sibling_chip_*, codex_query_response}
        # files must exist. Prior failure mode: probe declared "requirement" based
        # on CPU-side reverse-engineering simulations only, never running msprof on
        # the actual CANN reference call. User correction 2026-04-26 5_Cumsum.
        try:
            probe_text = probe.read_text()
        except Exception:
            probe_text = ""
        has_requirement_verdict = bool(re.search(
            r"(?im)^[\s\-\*]*Type\s*:\s*\*{0,2}\s*requirement\b", probe_text
        ))
        if has_requirement_verdict:
            outputs_dir = ws / "probes" / "probe_outputs"
            evidence_files = []
            if outputs_dir.is_dir():
                for f in outputs_dir.iterdir():
                    n = f.name.lower()
                    if any(t in n for t in ("msprof", "sibling", "codex", "opencode", "hardware_probe")):
                        evidence_files.append(f.name)
            if not evidence_files:
                rejections.append(Rejection(
                    rule_id="O5.probe_requirement_evidence",
                    description="probe verdict=requirement but no observational evidence (msprof / sibling-chip / codex query)",
                    expected="probes/probe_outputs/ contains at least one file matching {msprof*, sibling*, codex*, opencode*, hardware_probe*}",
                    actual=f"probes/probe_outputs/ contents: {sorted(p.name for p in outputs_dir.iterdir()) if outputs_dir.is_dir() else 'directory absent'}",
                    fix=(
                        "Per src/agents/aog-precision-probe.md §Step 4 (V3.4.2), a `requirement` "
                        "classification REQUIRES at least one of: "
                        "(a) `msprof_on_reference.{json,csv}` — runs msprof on the CANN ref call to capture "
                        "actual kernel/BlockDim/vec-ratio (reveals whether the alleged 'unidentified algorithm' "
                        "is actually a single SIMD kernel); "
                        "(b) `sibling_chip_<chip>.json` — same probe run on the sibling-target chip "
                        "(catches C19 chip-specific behavior); "
                        "(c) `codex_query_response.md` — public-doc search confirmation that no "
                        "documented AscendC API would solve the gap. "
                        "Without this evidence, the verdict is a guess labeled as a verdict (aog-self-critic C20)."
                    ),
                ))

    if prec.get("status") == "PASS":
        passed = prec.get("pass", 0)
        total = prec.get("total", 0)
        if passed != total:
            # Allow OL-83 waiver if probe_report.md has logic-diff section
            probe = ws / "probe_report.md"
            has_diff = probe.exists() and "step-by-step logic diff" in probe.read_text()
            if not has_diff:
                rejections.append(Rejection(
                    rule_id="O5.I2",
                    description="status=PASS requires pass==total OR probe_report.md §logic-diff waiver",
                    expected=f"pass == total (got {passed}/{total})",
                    actual=f"pass={passed} < total={total} and no probe_report.md §logic-diff",
                    fix="either fix kernel to full PASS, set status to PARTIAL_PASS, or produce probe_report.md §'Reference algorithm vs kernel: step-by-step logic diff'",
                ))

    # YAML-driven conditional_phases evaluation (2026-04-23, V3.3).
    # Previously O5.C3 was hard-coded here in Python. User flagged violation of
    # SKILL.md top-level "YAML wins" principle — Python code shouldn't encode
    # conditional-phase logic. Moved to opgen_state_machine.yaml `conditional_phases`
    # section; this function evaluates declaratively.
    _evaluate_conditional_phases(ws, data, rejections)

    # If edge_dataset.pt exists (Phase O2.5 two-source), verification.json must report edge results
    if (ws / "edge_dataset.pt").exists():
        edge_keys = ["pass_benchmark", "pass_edge", "total_edge"]
        edge_missing = [k for k in edge_keys if k not in prec]
        if edge_missing:
            # Also accept alternate keys pass_edge_int8 (op#11 style) since int8 is primary output
            alt_ok = "pass_edge_int8" in prec
            if not alt_ok:
                rejections.append(Rejection(
                    rule_id="O5.I3",
                    description="edge_dataset.pt exists but verification.json missing edge results",
                    expected=f"precision has keys: {edge_keys} (or pass_edge_int8 for ops with tuple outputs)",
                    actual=f"precision missing: {edge_missing}",
                    fix="worker Phase D must run BOTH benchmark_json (Pass A) and edge_dataset (Pass B); write results to verification.json",
                ))
