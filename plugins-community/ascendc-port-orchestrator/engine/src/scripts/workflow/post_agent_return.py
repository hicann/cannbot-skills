#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""post_agent_return.py — eager-merge enforcer for /aog-op-batch + /ascendc-op-gen.

Invoked by the orchestrator IMMEDIATELY after every Agent task-notification,
BEFORE any other tool call. Performs the 3 mandatory steps that the SKILL.md
§B5 eager-mode protocol requires:

  1. Record state_machine.py next (writes workspace/<op>/state_transitions.jsonl)
  2. KB merge: if workspace/<op>/knowledge_update.md exists with body > 50 chars,
     emit a `Skill(name="aog-knowledge-maintain", args=...)` invocation request
     for orchestrator to execute (this script can't invoke skills directly).
  3. Drop workspace/.kb_updated_<TS>.flag with summary for in-flight workers.

Exit codes:
  0  — All steps completed (or skipped because their preconditions not met).
       stdout JSON tells orchestrator what to do next.
  2  — Hard failure (workspace path doesn't exist, state_machine.py crashed).
  3  — KB merge needed: orchestrator MUST invoke the Skill before next Agent spawn.

Usage:
  python3 src/scripts/workflow/post_agent_return.py \\
      --workspace workspace/<op> \\
      --handoff "<exact handoff line agent wrote>"

Also runs in --dry-run mode to plan without committing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_handoff_from_progress(ws: pathlib.Path) -> str | None:
    """Read PROGRESS.md, return the last-line handoff signal.

    Looks for the canonical exit lines emitted by aog-kernel-worker /
    aog-precision-probe / aog-kernel-optimizer / etc:
      "→ orchestrator: ..."
      "@aog-precision-probe: ..."
      "@aog-kernel-optimizer: ..."
      "@aog-fused-optimizer: ..."
      "@aog-researcher: ..."
      "@aog-determinism-analyzer: ..."
      "@orchestrator: ..."

    DEBT-103 (PROGRESS.md path 收尾): also accept the arrow form `→ aog-X:`
    for inter-agent handoffs and normalize it to the canonical `@aog-X:`
    form (mirrors orchestrator.extract_canonical_handoff's _ARROW_TO_AT_FORM
    on the stdout path) so downstream state_machine handoff_match — which
    keys on the @-form for inter-agent handoffs — hits. Without this, a
    worker writing `→ aog-kernel-optimizer:` in PROGRESS.md was silently
    unmatched (caught by a LayerNorm E2E run).
    """
    progress = ws / "PROGRESS.md"
    if not progress.exists():
        return None
    text = progress.read_text(errors="replace")
    handoff_re = re.compile(
        r"^(→ orchestrator:"
        r"|@aog-precision-probe:|@aog-kernel-optimizer:|@aog-fused-optimizer:"
        r"|@aog-researcher:|@aog-determinism-analyzer:|@orchestrator:"
        r"|→ aog-precision-probe:|→ aog-kernel-optimizer:|→ aog-fused-optimizer:"
        r"|→ aog-researcher:|→ aog-determinism-analyzer:)\s*",
        re.MULTILINE,
    )
    matches = list(handoff_re.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    end = text.find("\n", last.end())
    if end == -1:
        end = len(text)
    line = text[last.start():end].strip()
    # Normalize arrow-form inter-agent handoff `→ aog-X` → `@aog-X` so the
    # state_machine handoff_match (keyed on @-form) hits. The orchestrator-form
    # `→ orchestrator:` is left as-is (its canonical form IS the arrow form).
    if line.startswith("→ aog-"):
        line = "@" + line[len("→ "):]
    return line


def run_state_machine_next(ws: pathlib.Path, handoff: str, dry_run: bool) -> dict[str, Any]:
    """Invoke state_machine.py next. Returns parsed JSON or error dict."""
    cmd = [
        "python3",
        str(PROJECT_ROOT / "src" / "scripts" / "workflow" / "state_machine.py"),
        "next",
        "--workspace",
        str(ws),
        "--handoff",
        handoff,
    ]
    if dry_run:
        cmd.append("--dry-run")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"error": "state_machine.py timed out after 30s"}
    if result.returncode != 0:
        return {
            "error": f"state_machine.py exit {result.returncode}",
            "stderr": result.stderr[-2000:],
        }
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"state_machine.py stdout not JSON: {e}", "stdout": result.stdout[:1000]}


def check_knowledge_update(ws: pathlib.Path) -> dict[str, Any]:
    """Inspect workspace/<op>/knowledge_update.md for non-trivial content."""
    ku = ws / "knowledge_update.md"
    if not ku.exists():
        return {"present": False, "reason": "knowledge_update.md absent"}
    text = ku.read_text(errors="replace")
    # Strip comment lines and empty lines for body-size check
    body_lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    body_chars = sum(len(ln) for ln in body_lines)
    if body_chars < 50:
        return {"present": True, "trivial": True, "body_chars": body_chars,
                "reason": f"knowledge_update.md body only {body_chars} chars (<50 threshold)"}
    return {
        "present": True,
        "trivial": False,
        "body_chars": body_chars,
        "path": str(ku.relative_to(PROJECT_ROOT)),
    }


def check_kb_merged_marker(ws: pathlib.Path) -> bool:
    """Has the merge already happened? Marker may live in workspace OR in archive.

    Beyond mere existence, validate the marker contents against disk:
    parse `entries=`, grep each NEW-entry ID in the listed
    `merged_into=` KB files. If a claimed entry is missing on disk, the
    marker is LYING — emit a warning and return False so the
    orchestrator triggers eager-merge again on the next agent return.

    Empirical anchor: 2026-05-17 audit of 4 markers found 3 of 4
    claimed entries that did not exist in any of the listed KB files.
    Skill `/aog-knowledge-maintain` wrote markers without doing the
    edits. See docs/design/KB_DESIGN_NOTES.md#kb-marker-verifier-design.
    """
    if not (ws / ".kb_merged").exists():
        return False
    try:
        from kb_marker_verifier import verify_marker
    except ImportError:
        # Verifier module sits alongside us in src/scripts/workflow/;
        # if it is somehow unimportable, fail closed and request a merge.
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        try:
            from kb_marker_verifier import verify_marker  # type: ignore
        except ImportError:
            return False
    try:
        from kb_tiering.adapters.cannbot_c import resolve_c_root
    except ImportError:
        scripts_root = pathlib.Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(scripts_root))
        try:
            from kb_tiering.adapters.cannbot_c import resolve_c_root
        except ImportError:
            return False
    rep = verify_marker(
        ws,
        project_root=PROJECT_ROOT,
        expected_c_root=resolve_c_root().resolve(),
    )
    if rep.verdict == "OK":
        return True
    if rep.verdict == "MISSING_ENTRIES":
        print(
            f"[post_agent_return] WARNING: .kb_merged marker invalid for "
            f"{ws.name}: claims entries {rep.missing!r} not present in any "
            f"merged_into= KB file. Treating as unmerged so the orchestrator "
            f"re-attempts the KB merge.",
            file=sys.stderr,
        )
        return False
    return False


def drop_kb_updated_flag(
    workspace_root: pathlib.Path,
    op: str,
    state_result: dict[str, Any],
    ku_check: dict[str, Any],
    kb_merge_required: bool,
    dry_run: bool,
) -> str | None:
    """Drop workspace/.kb_updated_<TS>.flag for in-flight worker pickup."""
    ts = _utc_compact()
    flag_path = workspace_root / f".kb_updated_{ts}.flag"
    payload = {
        "ts": ts,
        "op": op,
        "next_state": state_result.get("next_state"),
        "from_state": state_result.get("from_state"),
        "kb_merge_required": kb_merge_required,
        "kb_merge_path": ku_check.get("path") if kb_merge_required else None,
        "summary": (
            f"agent returned for {op}; state {state_result.get('from_state')} "
            f"→ {state_result.get('next_state')}"
        ),
    }
    if dry_run:
        return None
    flag_path.write_text(json.dumps(payload, indent=2))
    return str(flag_path.relative_to(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True, help="workspace/<op>/ for the returned agent")
    ap.add_argument(
        "--handoff",
        help="Handoff line from agent (override PROGRESS.md tail). If omitted, parses from PROGRESS.md.",
    )
    ap.add_argument("--dry-run", action="store_true", help="plan only, no writes")
    args = ap.parse_args()

    ws = pathlib.Path(args.workspace).resolve()
    if not ws.exists():
        print(json.dumps({"error": f"workspace path does not exist: {ws}"}), file=sys.stderr)
        return 2

    op = ws.name
    workspace_root = ws.parent  # i.e. .../workspace/

    handoff = args.handoff or parse_handoff_from_progress(ws)
    if not handoff:
        # Edge: agent crashed before writing handoff. Record as "unknown".
        handoff = "@orchestrator: agent returned without handoff line"

    # Step 1: state_machine.py next
    state_result = run_state_machine_next(ws, handoff, args.dry_run)
    if "error" in state_result:
        print(json.dumps({"error": "state_machine_failed", "detail": state_result}), file=sys.stderr)
        return 2

    # Step 2: KB merge check
    ku_check = check_knowledge_update(ws)
    already_merged = check_kb_merged_marker(ws)
    kb_merge_required = (
        ku_check.get("present", False)
        and not ku_check.get("trivial", False)
        and not already_merged
    )

    # Step 3: drop in-flight pickup flag
    flag_rel = drop_kb_updated_flag(
        workspace_root, op, state_result, ku_check, kb_merge_required, args.dry_run
    )

    # Step 3.5: Update per-op .opgen_state.json (V3.7.9, 2026-05-03)
    # Auto-records last_handoff + iter_history + current_state_machine_state.
    # Best-effort: if state_file.py is missing or .opgen_state.json absent, log + skip.
    state_file_update = {"updated": False, "reason": ""}
    if not args.dry_run:
        try:
            sf_path = ws / ".opgen_state.json"
            if sf_path.is_file():
                # extract from_agent slug from handoff (best-effort) or fall back to "unknown"
                from_agent_match = re.search(r"@?([\w-]+(?:-(?:kw|ko|pp|ar|fo|da|bs)-\d+))", handoff)
                from_agent = from_agent_match.group(1) if from_agent_match else "unknown"
                # extract to_agent (the @<thing> at start of handoff, if present)
                to_agent_match = re.match(r"(@?[\w-]+):", handoff)
                to_agent = to_agent_match.group(1) if to_agent_match else "@orchestrator"
                # Build inline-update commands for last_handoff + current_state_machine_state
                handoff_obj = json.dumps({
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "summary": handoff[:500],
                    "ts": _utc_iso(),
                })
                state_file_py = PROJECT_ROOT / "src" / "scripts" / "workflow" / "state_file.py"
                if state_file_py.is_file():
                    subprocess.run(
                        [sys.executable, str(state_file_py), "per-op", "update",
                         "--op", op, "--workspace", str(ws),
                         "--field", "last_handoff", "--value", handoff_obj],
                        check=False, capture_output=True,
                    )
                    next_sm = state_result.get("next_state", "")
                    if next_sm:
                        subprocess.run(
                            [sys.executable, str(state_file_py), "per-op", "update",
                             "--op", op, "--workspace", str(ws),
                             "--field", "current_state_machine_state",
                             "--value", json.dumps(next_sm)],
                            check=False, capture_output=True,
                        )
                    state_file_update = {"updated": True, "reason": "ok"}
                else:
                    state_file_update["reason"] = "state_file.py missing (V3.7.8 or earlier?)"
            else:
                state_file_update["reason"] = ".opgen_state.json absent (skill did not run Phase O0.5?)"
        except Exception as e:
            state_file_update["reason"] = f"exception: {type(e).__name__}: {e}"

    # Compose decision report for orchestrator
    report = {
        "op": op,
        "workspace": str(ws.relative_to(PROJECT_ROOT)),
        "handoff": handoff,
        "state_machine": {
            "from_state": state_result.get("from_state"),
            "next_state": state_result.get("next_state"),
            "rationale": state_result.get("rationale"),
        },
        "kb_merge": {
            "required": kb_merge_required,
            "knowledge_update_path": ku_check.get("path") if kb_merge_required else None,
            "skill_invocation_args": (
                f"knowledge_update_path={ku_check.get('path')} mode=auto"
                if kb_merge_required
                else None
            ),
            "reason_skipped": (
                ku_check.get("reason")
                if not kb_merge_required and ku_check.get("present")
                else ("already merged" if already_merged else None)
            ),
        },
        "kb_updated_flag": flag_rel,
        "opgen_state_file_update": state_file_update,
        "next_action": (
            "INVOKE_KB_SKILL_NOW"
            if kb_merge_required
            else "PROCEED_TO_NEXT_STATE_DISPATCH"
        ),
        "next_state_dispatch": (
            f"Spawn agent for state {state_result.get('next_state')} per state_machine.yaml"
            if state_result.get("next_state") not in {"finalize", "abort"}
            else f"State {state_result.get('next_state')} reached; run Phase O5/O6 in main"
        ),
        "ts": _utc_iso(),
    }

    print(json.dumps(report, indent=2))

    # Exit code 3 signals "kb merge needed" — orchestrator MUST invoke the skill before
    # spawning next Agent. Any other Agent spawn before this happens is a discipline gap
    # the workflow_critic will REJECT (rule SC4 in workflow_critic.py).
    return 3 if kb_merge_required else 0


if __name__ == "__main__":
    sys.exit(main())
