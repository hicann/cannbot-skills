# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O4 state machine — programmatic driver (V3.3, 2026-04-23).

User's architectural directive 2026-04-23: critic was a passive guardrail; orchestrator
made decisions by reading SKILL.md prose. If orchestrator read prose wrong, critic could
only catch specific rule violations — it couldn't catch "wrong state transition chosen
for this context". Solution: make orchestrator call THIS module at every decision point.
Decisions become programmatic + logged + critic-auditable.

Usage (orchestrator at every Phase O4 decision point):

    # Right after spawning an agent + reading its handoff:
    python3 src/scripts/workflow/state_machine.py next \\
        --workspace workspace/{op} \\
        --current-state await_worker \\
        --handoff "@aog-precision-probe: precision stuck"

    # Prints JSON: {"next_state": "await_probe", "rationale": "...", "matched_transition": {...}}
    # AND appends to workspace/{op}/state_transitions.jsonl

Critic side:

    python3 src/scripts/workflow/state_machine.py verify \\
        --workspace workspace/{op}

    # Reads state_transitions.jsonl, checks each logged transition is declared in YAML.
    # exit 0 if all legal; exit 2 + details if any illegal.

Design contract:
- Log is APPEND-ONLY. Orchestrator never rewrites history. Past decisions stay in record
  even if later proven wrong — easier to debug.
- Log entry is JSON (one per line) with: ts, from_state, to_state, trigger handoff,
  matched_transition_index, iter_counts_snapshot, workspace_state_snapshot (selected
  fields that influence transitions), rationale.
- State inference: `get_current_state(ws)` reads PROGRESS.md for most recent "spawn" +
  "handoff" markers, OR falls back to reading the last entry of state_transitions.jsonl.
- iter counters: derived by counting log entries where to_state matches each agent's
  state. Never stored separately (single source of truth = the log).

YAML fields used:
- phase_o4_states[]: id, iter_cap, iter_counter, exit_transitions (ordered list)
- phase_o4_total_spawn_cap (global)
- phase_o4_initial_state (fallback when log empty + no PROGRESS signal)

Condition DSL (evaluated against workspace snapshot):
- handoff_match: str — PREFIX of handoff line (handoff.startswith(arg))
- handoff_contains: str — SUBSTRING of handoff line (arg in handoff). Use when the
  discriminating token is NOT at the line start — e.g. the canonical arrow form
  `→ orchestrator: done — KO_PERF_PLATEAU` carries the verdict token as a suffix,
  so handoff_match (prefix-only) can never see it. (Added 2026-06-03; before this
  the YAML used `handoff_match: "KO_PERF_PLATEAU"` which silently never fired and
  let the plateau verdict fall through to redundant optimizer respawns.)
- verification_precision_status_in / _not_in: [str, ...]
- verification_det_policy_satisfied: bool
- verification_perf_below_threshold: bool
- iter_below_cap: str — iter counter name
- path_exists: str — relative to workspace (supports {op} placeholder, stripped)
- analysis_md_contains_any: [str, ...] — lowercased substring
- probe_report_has_actionable_fix: bool — heuristic: "Recommendation" section present
  AND non-empty (or explicit flag set)
- probe_classification_in: [str, ...] — match probe_report.md §Classification
  `Type: <verdict>` against allow-list (V3.3.1, 2026-04-25). Verdict in
  {requirement, convention} per aog-precision-probe SOP.
- det_report_decision_in: [RECOMMEND_ARCHITECTURAL, ...]
- trajectory_sigs_stable / trajectory_mad_tight / trajectory_mad_loose: bool (last 3
  Phase-D iters from PROGRESS; conservative: if not enough history, return False)
- always: true
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys
from typing import Any, Optional

from fsm_conditions import (
    _parse_worker_signal,
    _perf_ratio_and_threshold,
    _resolved_precision_status,
    _trajectory_stats,
    eval_condition,
)

try:
    import yaml  # PyYAML
except ImportError:
    sys.stderr.write("state_machine: PyYAML required (pip install pyyaml)\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve()
# FSM relocated 2026-07-05 to <plugin_root>/workflows/opgen_state_machine.yaml (cannbot
# convention: workflows/ holds the real workflow payload, corresponding to
# workflows/development-guide.md). This file lives at
# <plugin_root>/engine/src/scripts/workflow/state_machine.py → plugin_root = parents[4].


def _find_fsm_yaml(start: pathlib.Path) -> pathlib.Path:
    for candidate in [start.parent, *start.parents]:
        cand = candidate / "workflows" / "opgen_state_machine.yaml"
        if cand.exists():
            return cand
    return start.parents[4] / "workflows" / "opgen_state_machine.yaml"  # deterministic fallback


_PROJECT_ROOT = _HERE.parents[3]  # engine/ (kept for any engine-relative use)
YAML_PATH = _find_fsm_yaml(_HERE)

TRANSITIONS_FILENAME = "state_transitions.jsonl"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_state_machine() -> dict[str, Any]:
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"state machine YAML missing: {YAML_PATH}")
    with YAML_PATH.open() as f:
        return yaml.safe_load(f)


def get_state_spec(sm: dict, state_id: str) -> dict | None:
    for s in sm.get("phase_o4_states", []):
        if s["id"] == state_id:
            return s
    return None


# ---------------------------------------------------------------------------
# Transition log (append-only JSONL)
# ---------------------------------------------------------------------------
def log_path(ws: pathlib.Path) -> pathlib.Path:
    return ws / TRANSITIONS_FILENAME


def read_log(ws: pathlib.Path) -> list[dict]:
    p = log_path(ws)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_log(ws: pathlib.Path, entry: dict) -> None:
    p = log_path(ws)
    with p.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def iter_counts_from_log(ws: pathlib.Path, sm: dict) -> dict[str, int]:
    """Count how many times each state has been entered (iter counter basis).

    NODE-5 (2026-05-28): entries with `rollback_kind == "infra"` are SKIPPED
    so infra-rollback re-entries (SCP timeout, JSON parse fail, oversized
    .pt, verifier env issue — i.e. the kernel didn't change) don't consume
    the iter_below_cap budget. Empirical anchor: FA arch22 run `bjfp4fi3e`
    consumed all 4 probe-iters across 3 infra-rollback cycles even though the
    kernel was correct on probe-1 (10/10 PASS); NODE-4 raised cap 4→8 as a
    stopgap, NODE-5 is the cleaner fix that distinguishes infra-rollback
    from algorithm rollback.

    rollback_kind values:
      - "infra" — re-entry is free, count skipped
      - "algorithm" or absent — counted normally (default behavior)
    """
    counts: dict[str, int] = {}
    log = read_log(ws)
    for entry in log:
        to_state = entry.get("to_state")
        if not to_state:
            continue
        # NODE-5: skip infra-rollback re-entries from iter counting.
        if entry.get("rollback_kind") == "infra":
            continue
        spec = get_state_spec(sm, to_state)
        if spec is None:
            continue
        counter = spec.get("iter_counter", to_state)
        counts[counter] = counts.get(counter, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Workspace snapshot — fields that influence transitions
# ---------------------------------------------------------------------------
def snapshot(ws: pathlib.Path) -> dict:
    """Collect the verification.json + analysis.md markers + probe/det report hints
    that transition conditions evaluate against.
    """
    snap: dict[str, Any] = {
        "workspace": str(ws),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
    }

    vj = ws / "verification.json"
    if vj.exists():
        try:
            snap["verification"] = json.loads(vj.read_text())
        except Exception:
            snap["verification"] = None

    am = ws / "analysis.md"
    snap["analysis_md_lower"] = am.read_text().lower() if am.exists() else ""

    # V3.8.5 / DEBT-076 / DEBT-077 (2026-05-04): probe_result.json is the
    # authoritative routing input under the Python orchestrator. Fall back
    # to probe_report.md regex parsing when JSON absent (legacy LLM-orchestrator
    # path + agents that haven't been retrofitted yet).
    pres = ws / "probe_result.json"
    pr = ws / "probe_report.md"
    snap["probe_classification"] = None
    snap["probe_report_has_actionable_fix"] = False
    if pres.exists():
        try:
            data = json.loads(pres.read_text())
            cls = data.get("classification")
            if isinstance(cls, str):
                snap["probe_classification"] = cls.lower()
            # actionable fix = `next_directive` non-null/non-empty
            nd = data.get("next_directive")
            snap["probe_report_has_actionable_fix"] = bool(nd) and isinstance(nd, str) and len(nd.strip()) > 0
        except Exception:
            # malformed JSON — fall through to markdown parsing
            pass
    if pr.exists():
        txt = pr.read_text()
        m = re.search(r"(?im)^#+\s*Recommendation\b.*?(?=^#+\s|\Z)", txt, re.DOTALL)
        # JSON wins; only set from markdown if JSON didn't already set it
        if not snap["probe_report_has_actionable_fix"]:
            snap["probe_report_has_actionable_fix"] = bool(m and len(m.group(0).strip()) > 40)
        if snap["probe_classification"] is None:
            cm = re.search(r"(?im)^[\s\-\*]*Type\s*:\s*\*{0,2}\s*(requirement|convention)\b",
                           txt)
            snap["probe_classification"] = cm.group(1).lower() if cm else None

    dr = ws / "determinism_report.md"
    if dr.exists():
        dtxt = dr.read_text()
        # Extract RECOMMEND_* token from report
        m = re.search(r"\bRECOMMEND_(ARCHITECTURAL|FIX|ACCEPT_NONDET)\b", dtxt)
        snap["det_report_decision"] = f"RECOMMEND_{m.group(1)}" if m else None
    else:
        snap["det_report_decision"] = None

    # V3.8.5 / DEBT-077 #59 (2026-05-04): user_decision.md for await_user_decision
    # state. The expected format names a target such as `await_worker` in the
    # `next_state` field and may carry free text in a `reason` field.
    # Parse `next_state:` value into snapshot["user_decision_target"].
    ud = ws / "user_decision.md"
    snap["user_decision_target"] = None
    if ud.exists():
        utxt = ud.read_text()
        m = re.search(r"(?im)^[\s\-\*]*next_state\s*:\s*([a-z_]+)\s*$", utxt)
        if m:
            snap["user_decision_target"] = m.group(1).lower()

    # 2026-05-20 (structural-rewrite escalation, S5 cold-start gap): surface op_class +
    # derived complexity so eval_condition's `plugin_method` primitive can resolve
    # them before dispatching to the active plugin. Uses the canonical resolver
    # `schema_norm._detect_op_class(workspace, vj)` (same one ko_escalation_threshold
    # uses per main agent's 2026-05-20 pointer) for op_class, plus an FA-class
    # complexity heuristic (FA always L4 per OL-156 structural signature) for
    # op_complexity. Empirical catch: S5 cold-start on 3_FusionAttention 2026-05-20
    # — kw-1 emitted the sentinel + analysis.md cited the expected route, but
    # composite condition fell through to abort because op_class wasn't surfaced.
    snap["op_taxonomy"] = {"class": "unknown", "complexity": "unknown"}
    try:
        # Import locally to avoid making state_machine depend on the orchestrator
        # package at module load (state_machine is also imported by tests / other
        # bare callers that may not have the orchestrator package on PYTHONPATH).
        import sys as _sys
        import pathlib as _pl
        _orch_dir = _pl.Path(__file__).resolve().parent.parent / "orchestrator"
        if str(_orch_dir) not in _sys.path:
            _sys.path.insert(0, str(_orch_dir))
        import schema_norm as _sn  # type: ignore
        op_class_upper = getattr(_sn, "_detect_op_class")(ws, snap.get("verification") or {})
        snap["op_taxonomy"]["class"] = op_class_upper
        # FA-class complexity derivation: substring match on the uppercase
        # multi-tag string from _detect_op_class. {"FUSED", "SOFTMAX"} ⊆ tags
        # is the canonical FA structural signature → L4 per OL-156.
        if "FUSED" in op_class_upper and "SOFTMAX" in op_class_upper:
            snap["op_taxonomy"]["complexity"] = "L4"
    except Exception:
        pass  # snap retains default unknowns; plugin gate evaluates False

    return snap


# ---------------------------------------------------------------------------
# Current state inference
# ---------------------------------------------------------------------------
def get_current_state(ws: pathlib.Path, sm: dict) -> str:
    """Determine what state we're currently IN (just-exited, about to decide next).

    P0o (2026-05-05): empty state log + non-empty PROGRESS.md — extract last
    handoff line, run state machine forward from initial state to derive
    actual current state. Without this, every cold-start reset to
    `await_worker` and a worker that previously emitted `await_user_decision`
    (or any other handoff) gets ignored.

    P0bb / P0cc (2026-05-05): when bootstrap fires (log empty), persist the
    inference to state_transitions.jsonl BEFORE returning. Otherwise this
    function is impure-read: subsequent calls re-bootstrap from PROGRESS.md
    and can return a different answer if PROGRESS.md changed mid-session
    (e.g. agent appended a new handoff). Persisting locks the answer and
    creates an audit trail.

    Order:
      1. log non-empty → tail's to_state (canonical, no side effects)
      2. log empty + PROGRESS.md tail has a handoff routing to non-abort
         target → write `init→initial` and `initial→target` synthetic
         transitions, return target
      3. log empty + no usable handoff → write `init→initial`, return initial
    """
    log = read_log(ws)
    if log:
        return log[-1]["to_state"]

    # Bootstrap fallback: extract handoff from PROGRESS.md tail
    initial = sm.get("phase_o4_initial_state", "await_worker")
    handoff = _extract_handoff_from_progress(ws)
    if handoff:
        try:
            decision = next_state(ws, initial, handoff)
            target = decision.get("next_state")
            if target and target not in ("abort",):
                _persist_bootstrap_chain(ws, initial, target, handoff, decision)
                return target
        except Exception:
            pass  # any failure — fall through to default

    # Truly fresh workspace (empty log + no usable PROGRESS handoff) — return
    # initial state without persisting. Spurious `init→initial` entries on
    # fresh workspaces would confuse inspection tools (diagnose, scan).
    return initial


def _persist_bootstrap_chain(
    ws: pathlib.Path, initial_state: str, target: str,
    handoff: str, decision: dict,
) -> None:
    """Write the full bootstrap chain: `init→initial` then
    `initial→target` with the PROGRESS.md handoff. Used when bootstrap
    successfully routed from initial state to a non-abort target."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    append_log(ws, {
        "ts": ts,
        "from_state": "init",
        "to_state": initial_state,
        "handoff": "",
        "matched_transition_index": -1,
        "rationale": "bootstrap canonicalization (P0bb): empty log, PROGRESS.md handoff present",
        "iter_counts_snapshot": {},
    })
    append_log(ws, {
        "ts": ts,
        "from_state": initial_state,
        "to_state": target,
        "handoff": handoff,
        "matched_transition_index": decision.get("matched_transition_index", -1),
        "rationale": "bootstrap canonicalization (P0bb): " + (
            decision.get("rationale", "PROGRESS.md handoff routed to target")
        ),
        "iter_counts_snapshot": {},
    })


def _extract_handoff_from_progress(ws: pathlib.Path) -> str | None:
    """Read PROGRESS.md tail, find the most recent canonical handoff line.

    Looks for lines starting with `→ orchestrator:` or `@aog-` that match
    one of the canonical exit-handoff forms. Returns the full line, or None.
    """
    progress = ws / "PROGRESS.md"
    if not progress.exists():
        return None
    try:
        text = progress.read_text(errors="replace")
    except Exception:
        return None

    # P0w (2026-05-05): scope handoff search to the LAST "EXIT" or
    # "Exit handoff" section to avoid picking up historical canonical-shaped
    # lines from earlier session content (e.g. mid-document scaffolded
    # examples or prior-session output).
    #
    # Origin: op#28 multimodal_rope 2026-05-05 quota-resume. After
    # --cold-start cleared workspace state, kw-1 ran and wrote PROGRESS.md
    # with a HISTORICAL "→ orchestrator: done" line (template scaffold from
    # prior pre-T1/T2 session) plus a fresh Phase A entry. Worker died at
    # quota mid-iter with no fresh exit handoff. P0o re-bootstrapped from
    # PROGRESS.md tail, matched the historical "done" via last-line-of-
    # canonical-prefix logic, routed to finalize incorrectly.
    #
    # Fix: scope candidate search to lines AFTER the last "### EXIT" or
    # "## EXIT" marker. If no EXIT marker, take only the last 30 lines
    # (still respects "most recent wins" but limits historical false-match).
    lines = text.splitlines()
    scope_start = 0
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if (s.startswith("### EXIT")
                or s.startswith("## EXIT")
                or s.startswith("# EXIT")
                or s == "EXIT"
                or s.startswith("**EXIT**")
                or s.startswith("**Exit**")):
            scope_start = i + 1
            break
    if scope_start == 0:
        # No EXIT marker — per worker brief: "Write your handoff line to
        # PROGRESS.md tail ONLY". Look ONLY at the last non-empty line.
        # If it's canonical, take it. Otherwise return None (let orchestrator
        # use phase_o4_initial_state fallback). This is stricter than a
        # last-N-lines window which still risks historical false-match in
        # short files (op#28 scenario).
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                scope_start = i
                break
        else:
            return None  # all lines empty

    candidates = []
    for line in lines[scope_start:]:
        s = line.strip()
        if not s:
            continue
        # Strip common markdown decoration
        if s.startswith("- "):
            s = s[2:].strip()
        if s.startswith("**Exit handoff**:"):
            s = s.split(":", 1)[1].strip().strip("`")
        # Match canonical exit-handoff forms
        if (s.startswith("→ orchestrator:")
                or s.startswith("@aog-precision-probe")
                or s.startswith("@aog-kernel-optimizer")
                or s.startswith("@aog-fused-optimizer")
                or s.startswith("@aog-determinism-analyzer")
                or s.startswith("@aog-researcher")):
            candidates.append(s)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Next-state decision
# ---------------------------------------------------------------------------
def next_state(
    ws: pathlib.Path,
    current_state: str,
    handoff: str,
    plugin: Any = None,
    runtime_kwargs: Optional[dict] = None,
) -> dict:
    """Return {next_state, matched_transition, rationale} per YAML exit_transitions.

    Added 2026-05-20: optional `plugin` parameter for structural-rewrite escalation
    routing primitive (`plugin_method` in YAML). Backwards-compatible: existing
    callers that don't pass `plugin` get pre-2026-05-20 behavior — any
    `plugin_method` conditions in YAML evaluate to False, so legacy YAML
    stanzas (which don't reference `plugin_method`) keep working unchanged.

    Added 2026-05-27 (force-switch generalization, per owner direction):
    optional `runtime_kwargs` dict — per-invocation values that the
    `plugin_method` primitive can forward to plugin methods by name (via the
    YAML `forward_kwargs` list on the primitive). Generic mechanism, no
    feature-specific evaluator branches. None = empty dict (no kwargs
    forwarded).
    """
    sm = load_state_machine()
    spec = get_state_spec(sm, current_state)
    if spec is None:
        return {"error": f"state '{current_state}' not in YAML phase_o4_states",
                "next_state": None}

    snap = snapshot(ws)
    iter_counts = iter_counts_from_log(ws, sm)
    ctx = {
        "handoff": handoff, "snapshot": snap, "iter_counts": iter_counts,
        "ws": ws, "sm": sm, "plugin": plugin,
        "current_state": current_state,
        "runtime_kwargs": runtime_kwargs or {},
    }

    for idx, trans in enumerate(spec.get("exit_transitions", [])):
        cond = trans.get("condition", {})
        if eval_condition(cond, ctx):
            target = trans.get("goto")
            # V3.8.5 / DEBT-077 #59: __from_user_decision__ is a magic token —
            # resolve from user_decision.md `next_state:` value at runtime.
            if target == "__from_user_decision__":
                target = snap.get("user_decision_target")
                if target is None:
                    return {
                        "error": "user_decision_target_in condition matched but "
                                 "user_decision_target is None — should not happen",
                        "next_state": None,
                        "from_state": current_state,
                    }
            return {
                "next_state": target,
                "matched_transition_index": idx,
                "rationale": trans.get("rationale", "matched condition"),
                "from_state": current_state,
                "handoff": handoff,
                "iter_counts_snapshot": iter_counts,
            }

    # No match — this is a contract violation (states should have always-true fallback)
    return {
        "error": f"no transition matched from state '{current_state}' (handoff={handoff!r})",
        "next_state": None,
        "from_state": current_state,
    }


# ---------------------------------------------------------------------------
# Verify — replay log, check each transition against YAML
# ---------------------------------------------------------------------------
def verify_log(ws: pathlib.Path) -> list[str]:
    """Return list of violation strings. Empty = clean."""
    sm = load_state_machine()
    log = read_log(ws)
    violations: list[str] = []

    total_spawns = 0
    total_cap = sm.get("phase_o4_total_spawn_cap", 15)
    running_iter_counts: dict[str, int] = {}

    for i, entry in enumerate(log):
        from_s = entry.get("from_state")
        to_s = entry.get("to_state")
        from_spec = get_state_spec(sm, from_s) if from_s else None
        to_spec = get_state_spec(sm, to_s) if to_s else None
        if to_spec is None:
            violations.append(
                f"log line {i+1}: to_state '{to_s}' not in YAML phase_o4_states")
            continue

        # Is to_state in from_state's exit_transitions goto list?
        if from_spec is not None:
            declared_targets = {t.get("goto") for t in from_spec.get("exit_transitions", [])}
            if to_s not in declared_targets:
                violations.append(
                    f"log line {i+1}: transition {from_s} → {to_s} not declared "
                    f"(allowed targets: {sorted(declared_targets)})")

        # Iter cap check — running count
        counter = to_spec.get("iter_counter", to_s)
        running_iter_counts[counter] = running_iter_counts.get(counter, 0) + 1
        cap = to_spec.get("iter_cap", 999)
        if running_iter_counts[counter] > cap:
            violations.append(
                f"log line {i+1}: iter cap exceeded for counter '{counter}': "
                f"now {running_iter_counts[counter]} > cap {cap}")

        total_spawns += 1
        if total_spawns > total_cap:
            violations.append(
                f"log line {i+1}: total spawn cap exceeded: {total_spawns} > {total_cap}")

    return violations


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def _cmd_next(args):
    ws = pathlib.Path(args.workspace).resolve()
    if not ws.exists():
        sys.exit(f"workspace not found: {ws}")

    if args.current_state:
        current = args.current_state
    else:
        current = get_current_state(ws, load_state_machine())

    result = next_state(ws, current, args.handoff or "")
    if "error" in result:
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(2)

    # Append to log unless --dry-run
    if not args.dry_run:
        log_entry = {
            "ts": _dt.datetime.utcnow().isoformat() + "Z",
            "from_state": result["from_state"],
            "to_state": result["next_state"],
            "handoff": args.handoff or "",
            "matched_transition_index": result["matched_transition_index"],
            "rationale": result["rationale"],
            "iter_counts_snapshot": result["iter_counts_snapshot"],
        }
        append_log(ws, log_entry)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_verify(args):
    ws = pathlib.Path(args.workspace).resolve()
    if not ws.exists():
        sys.exit(f"workspace not found: {ws}")

    violations = verify_log(ws)
    if not violations:
        sys.stderr.write(f"✓ state_machine.verify: {ws.name} log clean\n")
        sys.exit(0)

    sys.stderr.write(f"❌ state_machine.verify: {ws.name} log has violations:\n")
    for v in violations:
        sys.stderr.write(f"  - {v}\n")
    sys.exit(2)


def _cmd_current(args):
    ws = pathlib.Path(args.workspace).resolve()
    sm = load_state_machine()
    cur = get_current_state(ws, sm)
    iter_counts = iter_counts_from_log(ws, sm)
    print(json.dumps({
        "current_state": cur,
        "iter_counts": iter_counts,
        "log_length": len(read_log(ws)),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Phase O4 state machine driver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_next = sub.add_parser("next", help="Decide next state given current + handoff")
    p_next.add_argument("--workspace", required=True)
    p_next.add_argument("--current-state", help="override detected current state")
    p_next.add_argument("--handoff", default="", help="handoff line from last agent")
    p_next.add_argument("--dry-run", action="store_true",
                         help="don't append to state_transitions.jsonl")
    p_next.set_defaults(func=_cmd_next)

    p_verify = sub.add_parser("verify", help="Verify state_transitions.jsonl conforms to YAML")
    p_verify.add_argument("--workspace", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_cur = sub.add_parser("current", help="Report current state + iter counts")
    p_cur.add_argument("--workspace", required=True)
    p_cur.set_defaults(func=_cmd_current)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
