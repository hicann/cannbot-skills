# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Self-critic invocation — wraps aog-self-critic Skill at YAML triggers.

Codex E review noted that Python orchestrator should fire self-critic by
EXPLICIT TRIGGERS, not LLM-discretion. Triggers configured in design doc:
- pre_phase_o4_first_spawn
- post_iter_cap_warning (current iter at iter_cap - 1)
- pre_finalize
- pre_commit

This module fires the right trigger for the current orchestrator state.

P135.S9 (2026-05-18): trigger-specific subset filtering (LLM evaluates fewer
items but still loads full SKILL.md → only the *evaluation* shrinks).

Q1 slim self-critic (2026-05-21): INLINE catalog extraction. Instead of
"Invoke the /aog-self-critic skill" (which causes claude --print to auto-load
the full 1485-LOC SKILL.md), build the prompt with ONLY the trigger-subset
items inlined. Token cost drops 64-86% per fire matching subset cardinality
(measured per-trigger average ~76%). Per main agent 2026-05-21 Q1 vote.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import state_executor

from backends import get_backend

# Funnel (backend-decoupling): self-critic skill invocation goes through the pluggable Backend,
# not a hardcoded `claude --print`. Default skill shape → dispatch's default params reproduce it
# byte-identical; main's `env=resolve_spawn_env` at the cc_backend chokepoint then covers this site too.
_backend = get_backend()


def _default_prespawn_critic_timeout_sec() -> int:
    """Pre-spawn critic timeout.

    Claude Code can tolerate the historical long timeout because its skill
    path streams predictably. opencode skill calls may sit silent behind a
    provider queue; keep that fail-open gate short enough that it cannot block
    the kernel-worker E2E for half an hour.
    """
    explicit = os.environ.get("AOG_PRESPAWN_CRITIC_TIMEOUT_SEC")
    if explicit is not None:
        return int(explicit)
    if getattr(_backend, "name", "") == "opencode":
        return int(os.environ.get("AOG_OPENCODE_PRESPAWN_CRITIC_TIMEOUT_SEC", "180"))
    return 1800


# Pre-spawn critic skill call timeout (env-overridable).
PRESPAWN_CRITIC_TIMEOUT_SEC = _default_prespawn_critic_timeout_sec()


CRITIC_TRIGGERS = {
    # Trigger ID → human-readable invocation context
    "pre_phase_o4_first_spawn": "before any agent spawn, first spawn of session only",
    "post_iter_cap_warning": "current iter is at iter_cap - 1 for any agent type",
    "pre_finalize": "next_state about to become finalize",
    "pre_commit": "before git commit",
}


# P135.S9 (2026-05-18): trigger-specific catalog subsets cut per-fire token
# cost 60-87% vs full-catalog load. Map per SKILL.md "Trigger-specific
# catalog subsets" section. Keep in sync with SKILL.md.
_TRIGGER_CATALOG_SUBSETS_ASCENDC = {
    "pre_phase_o4_first_spawn": [
        "C2", "C5", "C11", "C17", "C18", "C19", "C20", "C21", "C22", "C27", "C28", "C29", "C31",
    ],
    "post_iter_cap_warning": ["C1", "C7", "C13", "C25"],
    "pre_finalize": ["C13", "C14", "C18", "C23", "C26", "C28", "C30"],
    "pre_commit": ["C8", "C13", "C23", "C24", "C30"],
}

# Backward-compat alias — earlier code references the original name.
_TRIGGER_CATALOG_SUBSETS = _TRIGGER_CATALOG_SUBSETS_ASCENDC


def _get_subset_for(trigger: str, backend: str = "ascendc") -> list[str]:
    """Return the AscendC catalog subset for ``trigger``."""
    if backend != "ascendc":
        raise ValueError(f"unsupported critic backend: {backend!r}")
    return _TRIGGER_CATALOG_SUBSETS_ASCENDC.get(trigger, [])


# Q1 slim self-critic: paths to the SKILL.md source files.
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[4]
_SKILL_MD_PATHS = {
    "ascendc": _PLUGIN_ROOT / "skills" / "aog-self-critic" / "SKILL.md",
}

# H3 item heading pattern: "### C13:" or "### C-INFRA-RETRY-WITHOUT-CAP:"
# Item runs until next H3 (### prefix), H2 (## prefix), or EOF.
_ITEM_HEADING_RE = re.compile(r"^### (C[\w-]*?):", flags=re.MULTILINE)


def _load_skill_md(backend: str) -> Optional[str]:
    """Read SKILL.md content from disk, or None if missing. Cached per process
    via lru_cache wrapping (see _skill_md_cached below)."""
    skill_path = _SKILL_MD_PATHS.get(backend)
    if skill_path is None or not skill_path.is_file():
        return None
    try:
        return skill_path.read_text()
    except Exception:
        return None


def _extract_preamble(skill_md: str) -> str:
    """Extract everything BEFORE the first catalog item (### C1 / ### T1 / etc).

    Includes frontmatter, anti-pressure protocol reminder, When-to-invoke,
    Contract, trigger-subset map, and any other framing prose. The LLM needs
    this to understand HOW to evaluate items.
    """
    m = _ITEM_HEADING_RE.search(skill_md)
    if m is None:
        return skill_md  # no catalog items found; return all
    return skill_md[: m.start()].rstrip() + "\n"


def _extract_items(skill_md: str, item_ids: list[str]) -> dict[str, str]:
    """Extract the body of specific catalog items by ID.

    Args:
        skill_md: full SKILL.md content
        item_ids: list like ["C13", "C14", "C-INFRA-RETRY-WITHOUT-CAP"]

    Returns:
        dict mapping id → item body (including the `### Cn: ...` heading).
        Missing items are silently omitted (caller can detect via len()).
    """
    items: dict[str, str] = {}
    # Find all item headings + their positions
    matches = list(_ITEM_HEADING_RE.finditer(skill_md))
    for i, m in enumerate(matches):
        item_id = m.group(1)
        if item_id not in item_ids:
            continue
        start = m.start()
        # End of this item = start of next item, OR start of next ## section, OR EOF
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            # Find next ## heading after this item, or EOF
            next_h2 = re.search(r"^## ", skill_md[start + len(m.group(0)):], flags=re.MULTILINE)
            if next_h2:
                end = start + len(m.group(0)) + next_h2.start()
            else:
                end = len(skill_md)
        items[item_id] = skill_md[start:end].rstrip() + "\n"
    return items


def _build_slim_catalog(backend: str, subset: list[str]) -> Optional[str]:
    """Build the slim catalog string for the given backend + trigger subset.

    Returns the full inline content (preamble + selected items) or None if
    SKILL.md not readable (caller falls back to /skill invocation).
    """
    skill_md = _load_skill_md(backend)
    if not skill_md:
        return None
    preamble = _extract_preamble(skill_md)
    items_by_id = _extract_items(skill_md, subset)
    # Preserve subset order (deterministic prompt structure)
    body_parts: list[str] = []
    for item_id in subset:
        if item_id in items_by_id:
            body_parts.append(items_by_id[item_id])
    if not body_parts:
        return None  # subset items not found in SKILL.md → bail to fallback
    body = "\n".join(body_parts)
    return f"{preamble}\n\n---\n\n{body}\n"


def _resolve_backend(workspace: Path) -> str:
    """Return the only supported critic backend."""
    return "ascendc"


def fire_critic(
    workspace: Path,
    trigger: str,
    *,
    timeout_sec: int = PRESPAWN_CRITIC_TIMEOUT_SEC,
    backend: str | None = None,
) -> dict:
    """Fire aog-self-critic Skill with structured context.

    Args:
        workspace: workspace path
        trigger: one of CRITIC_TRIGGERS keys
        timeout_sec: claude --print timeout (default 900s = 15min)
        backend: ``ascendc`` or ``None``. Other backends are unsupported.

    Returns dict with success + report path + log entry.
    """
    if trigger not in CRITIC_TRIGGERS:
        raise ValueError(f"unknown trigger {trigger!r}; valid: {list(CRITIC_TRIGGERS.keys())}")

    # P135.S9b (task #19): auto-detect backend if not explicit
    if backend is None:
        backend = _resolve_backend(workspace)
    if backend != "ascendc":
        raise ValueError(f"unsupported critic backend: {backend!r}")

    snap = state_executor.snapshot(workspace)
    context = {
        "trigger": trigger,
        "trigger_description": CRITIC_TRIGGERS[trigger],
        "op": snap.op,
        "current_state": snap.current_state,
        "iter_counts": snap.iter_counts,
        "iter_caps": snap.iter_caps,
        "last_handoff_excerpt": snap.last_handoff[:500],
    }

    # P135.S9 (2026-05-18): per-trigger catalog subset cuts LLM token cost
    # 60-87% vs full-catalog load. Subset is the relevant slice from SKILL.md
    # "Trigger-specific catalog subsets" section.
    subset = _get_subset_for(trigger, backend=backend)

    skill_name = "aog-self-critic"

    # Q1 slim self-critic (2026-05-21): try INLINE catalog. Read SKILL.md
    # once, extract just the (preamble + subset items), build prompt with
    # ONLY those bytes — no /skill invocation. Cuts per-fire token cost
    # 64-86% matching subset cardinality (76% average).
    # Falls back to /skill invocation if extraction fails (SKILL.md missing
    # or no subset items match — e.g. unknown trigger or backend).
    slim_catalog = _build_slim_catalog(backend, subset) if subset else None
    if slim_catalog:
        # Inline-catalog prompt: no /skill auto-load, just the slim content +
        # context + instructions. Bypasses claude --print's skill resolver.
        prompt = (
            f"You are running the {skill_name} audit at trigger={trigger!r} "
            f"backend={backend!r}. The relevant catalog subset is inlined below "
            f"— do NOT load any external skill file; evaluate ONLY these items.\n\n"
            f"=== CATALOG (slim subset for trigger={trigger!r}) ===\n\n"
            f"{slim_catalog}\n\n"
            f"=== ORCHESTRATOR CONTEXT ===\n\n"
            f"{json.dumps(context, indent=2)}\n\n"
            f"=== TASK ===\n\n"
            f"Evaluate against the current orchestrator state and last ~5 turns "
            f"of activity in workspace {workspace}. Apply ONLY the items in the "
            f"slim catalog above ({', '.join(subset)}). "
            f"Write a self_critic_report.md in the workspace. "
            f"Exit with the verdict (PASS / WARN / BLOCK)."
        )
    elif subset:
        # Fallback path 1: subset exists but slim extraction failed — invoke
        # /skill and filter via instruction. Preserves prior P135.S9 behavior.
        subset_clause = (
            f" **Apply ONLY these catalog items for this trigger**: "
            f"{', '.join(subset)}. Do NOT evaluate items outside this subset — "
            f"they are not relevant to the `{trigger}` decision point and "
            f"evaluating them wastes tokens. See SKILL.md "
            f"'Trigger-specific catalog subsets' section for rationale."
        )
        prompt = (
            f"Invoke the /{skill_name} skill with trigger={trigger!r} backend={backend!r}. "
            f"Context: {json.dumps(context, indent=2)}. "
            f"Evaluate against the current orchestrator state and last ~5 turns "
            f"of activity in workspace {workspace}.{subset_clause} "
            f"Write a self_critic_report.md in the workspace. Exit with the "
            f"verdict (PASS / WARN / BLOCK)."
        )
    else:
        # Fallback path 2: no subset (unknown trigger) — full catalog load via
        # /skill invocation. Preserves prior behavior.
        subset_clause = (
            " Run all checks against the current orchestrator state and last "
            "~5 turns of activity. Identify any catalog items in the skill's "
            "SKILL.md that match (full range — currently C1 through the most "
            "recently added entry)."
        )
        prompt = (
            f"Invoke the /{skill_name} skill with trigger={trigger!r} backend={backend!r}. "
            f"Context: {json.dumps(context, indent=2)}. "
            f"Evaluate against the current orchestrator state and last ~5 turns "
            f"of activity in workspace {workspace}.{subset_clause} "
            f"Write a self_critic_report.md in the workspace. Exit with the "
            f"verdict (PASS / WARN / BLOCK)."
        )

    # bypassPermissions per Day 4 finding (P0f) — acceptEdits denies all Bash
    # which the self-critic skill needs (grep, file scans, git log etc.). This is
    # the DEFAULT skill shape, so Backend.dispatch(kind="skill") reproduces the
    # exact `claude --print --output-format json --permission-mode bypassPermissions <prompt>` cmd.
    _env = _backend.dispatch(skill_name, prompt, kind="skill", timeout=timeout_sec)
    _raw = _env.raw_envelope
    if _raw.get("not_found"):
        # faithful: original had no FileNotFoundError guard → propagate as before
        raise FileNotFoundError(_raw.get("stderr") or "claude CLI not found")
    if _raw.get("timed_out"):
        success = False
        exit_code = -1
        stdout_tail = "(critic timed out)"
        timed_out = True
    else:
        exit_code = _raw.get("returncode")
        success = not _env.is_error  # == returncode == 0
        stdout_tail = (_env.output_text or "")[-2000:]
        timed_out = False

    log_entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger": trigger,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "success": success,
        "timed_out": timed_out,
        "stdout_tail": stdout_tail,
    }
    log_path = workspace / ".critic_invoke_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    report_path = workspace / "self_critic_report.md"

    return {
        "success": success,
        "trigger": trigger,
        "report_path": str(report_path) if report_path.exists() else None,
        "log_entry": log_entry,
    }


def should_fire_iter_cap_warning(workspace: Path, state: str) -> bool:
    """Check whether post_iter_cap_warning trigger applies."""
    if state in state_executor.TERMINAL_STATES:
        return False
    cap = state_executor.iter_cap(state, workspace=workspace)
    counter = state[len("await_"):] if state.startswith("await_") else state
    count = state_executor.iter_count(workspace, counter)
    return count >= cap - 1


# CLI smoke
if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--trigger", required=True, choices=list(CRITIC_TRIGGERS.keys()))
    args = ap.parse_args()
    result = fire_critic(args.workspace, args.trigger)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)
