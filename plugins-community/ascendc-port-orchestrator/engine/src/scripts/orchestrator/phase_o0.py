# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Phase O0 hook integrity gate (P0pp 2026-05-06).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 7.

Scope:
    Verify hooks (workflow_critic, anti-pressure load, deploy scripts)
    are reachable + sane BEFORE the orchestrator starts spawning agents.
    Without this, orchestrator could enter Phase O4 with broken hooks
    and silently allow forbidden operations (G1/G7/G11/G12 violations).

This is a lower-damage gap than O2.5 / O5 — broken hooks are usually
caught downstream — but explicit pre-check fails loud rather than
relying on the first violation to surface the bug.
"""
from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent

try:
    from kb_paths import kb_root as _kb_root
except ImportError:  # pragma: no cover — fallback if orchestrator/ not on sys.path
    def _kb_root() -> Path:
        return _PROJECT_ROOT.parent / "kb"

# Engine-relative infra files (resolved against _PROJECT_ROOT == engine/).
REQUIRED_FILES = (
    "src/scripts/workflow/workflow_critic.py",
    "src/scripts/workflow/state_machine.py",
)

# KB files (2026-07-05: relocated to <plugin_root>/kb/, resolved via kb_root()).
REQUIRED_KB_FILES = (
    "shared/ANTI_PRESSURE_PROTOCOLS.md",
    "KB_INDEX.md",
    "target/ascendc/OPERATIONAL_KNOWLEDGE.md",
)

# Workflow FSM (2026-07-05: relocated to <plugin_root>/workflows/ per cannbot convention —
# workflows/ holds the real workflow payload; resolved against _PROJECT_ROOT.parent/"workflows").
REQUIRED_WORKFLOW_FILES = (
    "opgen_state_machine.yaml",
)

REQUIRED_DEPLOY_SCRIPTS = (
    "src/scripts/deploy_to_npu.sh",
    "src/scripts/deploy_to_npu_lane.sh",
)


@dataclass
class O0Report:
    verdict: str  # "READY" | "DEGRADED" | "BLOCKED"
    missing_files: list[str] = field(default_factory=list)
    missing_scripts: list[str] = field(default_factory=list)
    hook_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


def _check_hook_registration() -> tuple[str, Path, list[str]]:
    """Load the cannbot registration checker from the engine script root."""
    checker_path = _PROJECT_ROOT / "src" / "scripts" / "preflight_install_hooks.py"
    spec = importlib.util.spec_from_file_location(
        "ascendc_port_hook_registration", checker_path
    )
    if spec is None or spec.loader is None:
        return "unknown", checker_path, ["cannot load hook registration checker"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_current_registration()


def check_hook_integrity(workspace: Path = None) -> O0Report:
    """Verify required hook + KB + deploy infrastructure is present.

    Returns:
        READY: everything in place
        DEGRADED: deploy scripts missing (orchestrator works, deploys
            won't — fail the first time worker tries to build)
        BLOCKED: critical KB / hook files missing — refuse to spawn
            anything; symptoms would be silent quality regression
    """
    rep = O0Report(verdict="READY")

    for rel in REQUIRED_FILES:
        if not (_PROJECT_ROOT / rel).exists():
            rep.missing_files.append(rel)

    _kb = _kb_root()
    for rel in REQUIRED_KB_FILES:
        if not (_kb / rel).exists():
            rep.missing_files.append(f"kb/{rel}")

    _wf = _PROJECT_ROOT.parent / "workflows"
    for rel in REQUIRED_WORKFLOW_FILES:
        if not (_wf / rel).exists():
            rep.missing_files.append(f"workflows/{rel}")

    for rel in REQUIRED_DEPLOY_SCRIPTS:
        if not (_PROJECT_ROOT / rel).exists():
            rep.missing_scripts.append(rel)

    try:
        _, _, rep.hook_errors = _check_hook_registration()
    except Exception as exc:  # fail closed: an unreadable checker is not an armed install
        rep.hook_errors.append(f"hook registration check failed: {exc}")

    if rep.missing_files or rep.hook_errors:
        rep.verdict = "BLOCKED"
        rep.summary = (
            f"Phase O0 BLOCKED: {len(rep.missing_files)} critical infra file(s) "
            f"missing, {len(rep.hook_errors)} hook registration error(s) — "
            "orchestrator would start without an active safety net"
        )
    elif rep.missing_scripts:
        rep.verdict = "DEGRADED"
        rep.summary = (
            f"Phase O0 DEGRADED: {len(rep.missing_scripts)} deploy script(s) "
            f"missing — workers will fail at build time"
        )
    else:
        rep.summary = "Phase O0 READY: all hooks + KB + deploy scripts present"

    return rep


def format_block_message(rep: O0Report) -> str:
    """Format user-facing block message."""
    lines = [f"[orchestrator] {rep.summary}"]
    if rep.missing_files:
        lines.append("[orchestrator] Missing critical files:")
        lines.extend(f"[orchestrator]   - {f}" for f in rep.missing_files)
    if rep.missing_scripts:
        lines.append("[orchestrator] Missing deploy scripts:")
        lines.extend(f"[orchestrator]   - {s}" for s in rep.missing_scripts)
    if rep.hook_errors:
        lines.append("[orchestrator] Hook registration errors:")
        lines.extend(f"[orchestrator]   - {e}" for e in rep.hook_errors)
        lines.append(
            "[orchestrator] Re-run this plugin's init.sh (or reinstall/enable the "
            "marketplace plugin), restart Claude Code, then invoke the current entry Skill again."
        )
    if rep.warnings:
        lines.append("[orchestrator] Warnings:")
        lines.extend(f"[orchestrator]   - {w}" for w in rep.warnings)
    return "\n".join(lines)
