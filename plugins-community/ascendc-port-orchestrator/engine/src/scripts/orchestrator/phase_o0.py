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
import shutil
import sys
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


def _active_backend_name() -> str:
    """Canonical name of the harness backend this run will dispatch through."""
    raw = (os.environ.get("AOG_HARNESS_BACKEND") or "claude_code").strip().lower()
    return raw.replace("-", "_")


def _opencode_probe_script(adapter, tool: str, path: str) -> str:
    """The JS the canary runs: drive the adapter's real tool hook and report the verdict."""
    import json  # noqa: PLC0415

    return (
        'import { A5OpsHooksPlugin } from %s;\n'
        'const h = await A5OpsHooksPlugin({directory: process.cwd()}, '
        '{projectRoot: %s});\n'
        'let blocked = false;\n'
        'try { await h["tool.execute.before"]('
        '{tool: %s, sessionID: "o0probe", callID: "c0"}, '
        '{args: {filePath: %s, content: "probe"}}); } catch (e) { blocked = true; }\n'
        'console.log(blocked ? "DENIED" : "ALLOWED");\n'
    ) % (
        json.dumps(str(adapter)),
        json.dumps(str(_PROJECT_ROOT)),
        json.dumps(tool),
        json.dumps(path),
    )


def _run_opencode_probe(adapter, agent_type: str, *, tool: str, path: str) -> tuple[bool, str]:
    """Run one canary. Returns (was_denied, error). A non-empty error means UNKNOWN, not allowed."""
    import subprocess  # noqa: PLC0415

    env = dict(os.environ)
    if agent_type:
        env["AOG_HOOK_AGENT_TYPE"] = agent_type
        env["AOG_HOOK_AGENT_ID"] = f"opencode:{agent_type}"
    else:
        env.pop("AOG_HOOK_AGENT_TYPE", None)
        env.pop("AOG_HOOK_AGENT_ID", None)
    # Same runtime order the installer uses. opencode ships as a bun-compiled binary, so a
    # machine can perfectly well have `bun` and no `node`; hard-coding node here while the
    # installer accepted either meant such a machine installed with a green
    # `hooks_verified_live` and then failed this gate on every single run — a success that
    # could never be acted on.
    runtime = shutil.which("node") or shutil.which("bun")
    if not runtime:
        return False, "neither node nor bun found — cannot prove the opencode safety net is armed"
    try:
        res = subprocess.run(
            [runtime, "--input-type=module", "-e", _opencode_probe_script(adapter, tool, path)],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=60, env=env,
        )
    except FileNotFoundError:
        return False, f"{runtime} vanished — cannot prove the opencode safety net is armed"
    except subprocess.TimeoutExpired:
        return False, "opencode safety-net canary timed out"
    if res.returncode != 0:
        return False, f"canary failed to run: {res.stderr.strip()[:200]}"
    return "DENIED" in res.stdout, ""


def _opencode_plugin_registration_errors(cfg: dict) -> list[str]:
    errors: list[str] = []
    plugins = cfg.get("plugin") or []
    if not any(isinstance(p, list) and p and str(p[0]).endswith("a5_ops_hooks.mjs")
               for p in plugins):
        errors.append(
            "opencode config does not register a5_ops_hooks.mjs — sub-agents would run "
            "with no safety net"
        )
    if not _plugin_registration_has_project_root(plugins):
        errors.append(
            "opencode plugin registration carries no projectRoot — the adapter cannot "
            "locate the canonical checkers"
        )
    return errors


def _plugin_registration_has_project_root(plugins) -> bool:
    """One plugin entry must be a list whose options dict carries a truthy projectRoot."""
    return any(
        isinstance(entry, list) and len(entry) > 1
        and (entry[1] or {}).get("projectRoot")
        for entry in plugins
    )


def _opencode_safety_net_errors(adapter) -> list[str]:
    """Run the read and write deny/allow pairs. Returns the errors they found.

    Each pair must DISCRIMINATE, not merely deny. A guard that always throws — the exact shape
    produced by a missing checker script, since python3 then exits non-zero and the adapter
    rethrows — would satisfy a deny-only probe while proving nothing. So every check below runs
    the same operation twice and requires opposite verdicts. This mirrors the Claude Code
    liveness proof in init.sh, which is deliberately a deny/allow pair.

    The READ pair's agent type is deliberately NOT one of the adapter's KERNEL_AUTHOR_AGENTS:
    those take the adapter's inline JS access guard, which refuses output/ reads on its own, so
    a canary using them stays green even when the canonical Python checker is missing entirely.
    A non-kernel-author agent routes the decision through output_read_guard.py alone.

    The WRITE pair exists because the read pair exercises output_read_guard only; the
    generated-code rules are a different code path, and a canary that never writes cannot tell
    whether they are live.
    """
    probes = (
        ("read_deny", "aog-precision-probe", "read",
         str(_PROJECT_ROOT / "output" / "_o0probe" / "verification.json")),
        ("read_allow", "", "read",
         str(_PROJECT_ROOT / "output" / "_o0probe" / "verification.json")),
        ("write_deny", "aog-kernel-worker", "write",
         str(_PROJECT_ROOT / "workspace" / "_o0probe" / "op_host" / "probe.cpp")),
        ("write_allow", "aog-kernel-worker", "write",
         str(_PROJECT_ROOT / "workspace" / "_o0probe" / "probe_notes.md")),
    )
    verdicts: dict = {}
    for key, agent_type, tool, path in probes:
        denied, err = _run_opencode_probe(adapter, agent_type, tool=tool, path=path)
        if err:
            return [err]
        verdicts[key] = denied

    errors: list[str] = []
    if not verdicts["write_deny"]:
        errors.append(
            "opencode safety net did NOT refuse a kernel author writing op_host/ scaffold — "
            "the generated-code rules are not reaching the write path"
        )
    if verdicts["write_allow"]:
        errors.append(
            "opencode safety net refused an ordinary kernel-author write — the write-side "
            "guard is not discriminating (an always-deny guard proves nothing)"
        )
    if not verdicts["read_deny"]:
        errors.append(
            "opencode safety net did NOT refuse an answer-bearing output/ read by a "
            "dispatched sub-agent — sub-agents would run unguarded"
        )
    if verdicts["read_allow"]:
        errors.append(
            "opencode safety net refused the MAIN agent too — it is not discriminating "
            "(an always-deny guard, e.g. a missing/broken canonical checker, proves nothing)"
        )
    return errors


def _check_opencode_registration() -> tuple[str, Path, list[str]]:
    """Prove the opencode safety net is armed — by BEHAVIOUR, not by counting files.

    The Claude Code checker validates a settings.json/hooks.json declaration, which does
    not exist for opencode: its adapter is registered through the process-private
    OPENCODE_CONFIG_CONTENT the backend injects at dispatch time. So verify the two things
    that actually decide whether a sub-agent runs guarded:

      1. the backend really emits a `plugin` registration carrying `projectRoot`;
      2. the adapter, driven with a payload the canonical guard MUST refuse, does refuse —
         i.e. adapter -> canonical Python checker -> denial actually works end to end.

    (2) exercises the real adapter and the real canonical checker. It does NOT prove that
    opencode itself loaded the adapter in a given session; that is the installer canary's
    job and is reported separately.
    """
    adapter = _PROJECT_ROOT / "src" / "opencode" / "a5_ops_hooks.mjs"
    if not adapter.is_file():
        return "opencode", adapter, [f"opencode hook adapter missing: {adapter}"]

    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "src" / "scripts" / "orchestrator"))
        from backends.opencode_backend import OpencodeBackend  # noqa: PLC0415

        cfg = OpencodeBackend.opencode_config()
    except Exception as exc:
        return "opencode", adapter, [f"cannot build opencode config: {exc!r}"]

    errors = _opencode_plugin_registration_errors(cfg)
    errors.extend(_opencode_safety_net_errors(adapter))
    return "opencode", adapter, errors


def _check_hook_registration() -> tuple[str, Path, list[str]]:
    """Load the cannbot registration checker from the engine script root."""
    if _active_backend_name() == "opencode":
        return _check_opencode_registration()
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
