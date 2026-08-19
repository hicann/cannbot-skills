# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Agent dispatch — combines brief construction + harness-backend spawn.

Single entry point for orchestrator.py main loop:

    result = dispatch.spawn_for_state(op, workspace, state, lane, ...)
    # → backend Envelope (is_error / output_text / raw_envelope)

Internally:
1. Resolve agent type from state via state_executor.next_agent
2. Build G7 slug
3. Construct brief via briefs/<agent>_brief.py
4. Spawn via the active harness backend (backends.registry.get_backend)
5. Persist raw envelope to workspace/<op>/.cc_envelope_log.jsonl (codex #6)
6. Return Envelope (or raise on backend/transport failure)

Codex C5 spike confirmed subagent transport from Python is feasible. This
module is the production-ready wrapper.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import agent_transport
import state_executor
# Harness-decoupling: agent spawn goes through the Backend (CC plugin), not agent_transport directly.
# spawn_for_state still owns brief + graybox/bwrap sandbox_prefix build (orchestrator-canonical, inv#2);
# it passes the OPAQUE prefix to the backend, which verbatim-prepends it.
from backends import get_backend
from backends.base import Envelope
from briefs._common import g7_slug, load_env
from briefs.kw_brief import build_worker_brief
from briefs.pp_brief import build_probe_brief
from source_arch import verify_source_stage
from briefs.ko_brief import build_optimizer_brief
from briefs.ar_brief import build_researcher_brief
from briefs.fo_brief import build_fused_optimizer_brief
from briefs.da_brief import build_det_analyzer_brief
from briefs.cl_brief import build_cann_learner_brief


_STOP_GATE_LOG = logging.getLogger(__name__)

_backend = get_backend()


# Map agent type → brief builder.
# Original 6: P0oo 2026-05-06.
BRIEF_BUILDERS = {
    "aog-kernel-worker": build_worker_brief,
    "aog-precision-probe": build_probe_brief,
    "aog-kernel-optimizer": build_optimizer_brief,
    "aog-researcher": build_researcher_brief,
    "aog-fused-optimizer": build_fused_optimizer_brief,
    "aog-determinism-analyzer": build_det_analyzer_brief,
    "aog-cann-learner": build_cann_learner_brief,
}


_ACTIVE_AGENT_MARKERS = {
    "aog-kernel-worker": ".kernel_worker_active",
    "aog-kernel-optimizer": ".optimizer_active",
}


@contextmanager
def _active_agent_marker(workspace: Path, agent_type: str):
    """Mark kernel-authoring agents active for workflow_critic G1.

    The critic is backend-agnostic: it allows workspace/kernel writes only
    while the orchestrator-owned marker exists. Claude Code historically got
    this through its host-hook path; Codex/opencode must get the same lifecycle
    from the shared Python dispatcher.
    """
    marker_name = _ACTIVE_AGENT_MARKERS.get(agent_type)
    if marker_name is None:
        yield
        return
    marker = workspace / marker_name
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "agent_type": agent_type,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }) + "\n")
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def spawn_for_state(
    op: str,
    workspace: Path,
    state: str,
    *,
    lane: int,
    spawn_index: int,
    timeout_sec: int = 3600,
    background: bool = False,
    output_file: Optional[Path] = None,
    directive_text: Optional[str] = None,
    handoff_from_prior: Optional[str] = None,
):
    """Build brief + spawn agent for `state`.

    Args:
        op: workspace dir name
        workspace: workspace dir path
        state: YAML state (await_worker / await_probe / etc.)
        lane: NPU id 0/1/2
        spawn_index: 1-based counter (used for G7 slug)
        timeout_sec: foreground timeout (default 1h)
        background: if True, returns Popen; caller manages lifecycle
        output_file: required when background=True
        directive_text: passed to brief builder for respawns
        handoff_from_prior: handoff text from prior agent

    Returns:
        - foreground: AgentResult
        - background: subprocess.Popen
    """
    agent_type = state_executor.next_agent(state)
    if agent_type is None:
        raise ValueError(f"state {state!r} is terminal or unknown — no agent to spawn")

    builder = BRIEF_BUILDERS.get(agent_type)
    if builder is None:
        raise NotImplementedError(
            f"brief builder for {agent_type!r} not yet implemented (Day 2 task)"
        )

    env = load_env()

    # Task #48 (2026-05-13): overlay workspace-level opgen_mode +
    # port_a3_source from .opgen_state.json onto env. The .ascendc_env
    # file is a static config (target / host / paths); per-op mode lives
    # in workspace/.opgen_state.json (written by _cmd_port_a3 in
    # orchestrator.py). Without this overlay, port_a3_to_a5 workspaces
    # get a non-migration brief and emit the wrong artifact layout.
    state_file = workspace / ".opgen_state.json"
    # gap#2 graybox airtight isolation (a-fs): a workspace whose .opgen_state.json sets
    # `graybox_sandbox: true` spawns the agent inside a bwrap mount-namespace (only KB +
    # copied-in arch22 + toolchain + workspace bound; cann/output absent = airtight). The
    # runner (kw-graybox) sets the flag. Default-absent → no change.
    graybox_cfg = {"enabled": False, "arch22_dir": None}
    state_data: dict = {}
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text())
            if not isinstance(state_data, dict):
                raise TypeError("durable state is not a JSON object")
        except Exception as exc:
            raise RuntimeError(
                f"refusing agent spawn with unreadable durable state: {exc}"
            ) from exc

    ws_mode = state_data.get("opgen_mode")
    if ws_mode and ws_mode != env.opgen_mode:
        env.opgen_mode = ws_mode
    if ws_mode == "port_a3_to_a5":
        valid_stage, stage_reason, _stage_manifest = verify_source_stage(
            workspace, state_data
        )
        if not valid_stage:
            raise RuntimeError(
                "refusing migration agent spawn with invalid source-only snapshot: "
                f"{stage_reason}"
            )
        # These values are security invariants, not caller-controlled options.
        env.port_a3_source = state_data["port_a3_source"]
        graybox_cfg["enabled"] = True
        graybox_cfg["arch22_dir"] = state_data["graybox_arch22_dir"]
    elif env.opgen_mode == "port_a3_to_a5":
        raise RuntimeError(
            "refusing migration agent spawn without a verified migration state"
        )

    # Resolve the scoped plugin once and pass it to each brief builder.  A
    # detection failure is not equivalent to "no workflow owns this
    # workspace": swallowing it would silently build a generic brief and
    # bypass migration/backward-specific safeguards.
    from plugins import detect_plugin
    plugin = detect_plugin(workspace)

    # iter_cap_remaining for budget hint to agent. iter_cap honors any
    # workspace .cap_bumps.jsonl (V3.8.5 / DEBT-077 #61) — agent sees the
    # effective cap including user-explicit bumps.
    iter_cap = state_executor.iter_cap(state, workspace=workspace, plugin=plugin)
    iter_count = state_executor.iter_count(workspace, _state_to_counter(state))
    cap_remaining = iter_cap - iter_count

    # Build brief — each builder accepts kwargs subset.
    # The plugin is passed to every builder
    # so paradigm-native phase block dispatch works without any inline
    # `if backend == "X"` branches.
    if agent_type == "aog-kernel-worker":
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            directive_text=directive_text,
            handoff_from_prior_agent=handoff_from_prior,
            plugin=plugin,
        )
    elif agent_type == "aog-precision-probe":
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            handoff_from_worker=handoff_from_prior,
            plugin=plugin,
        )
    elif agent_type == "aog-kernel-optimizer":
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            directive_text=directive_text,
            handoff_from_worker=handoff_from_prior,
            plugin=plugin,
        )
    elif agent_type == "aog-researcher":
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            handoff_from_prior_agent=handoff_from_prior,
            directive_text=directive_text,
            plugin=plugin,
        )
    elif agent_type == "aog-fused-optimizer":
        # P0ii (2026-05-05): fo brief signature mirrors ko (directive_text +
        # handoff_from_prior). Spawned when ko plateaus on a fused op or via
        # V3.3.10 pre-empt to inform the Kind-2 directive.
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            directive_text=directive_text,
            handoff_from_prior=handoff_from_prior,
            plugin=plugin,
        )
    elif agent_type == "aog-determinism-analyzer":
        # P0oo (2026-05-06): da brief — analyzer-only role (no kernel edit).
        # Fires when DET_POLICY=required AND observed_deterministic=false.
        # P0nn classified DET_POLICY in O1.5; this gate now actually fires.
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            handoff_from_prior=handoff_from_prior,
            plugin=plugin,
        )
    elif agent_type == "aog-cann-learner":
        # Migration-only research recovery. The runner re-validates its sealed
        # inputs and scanners; this brief carries the bounded FSM handoff.
        brief = builder(
            op, workspace,
            lane=lane, spawn_index=spawn_index,
            iter_cap_remaining=cap_remaining, env=env,
            handoff_from_prior_agent=handoff_from_prior,
            directive_text=directive_text,
            plugin=plugin,
        )
    else:
        raise NotImplementedError(f"brief signature for {agent_type!r} not handled in dispatch")

    # Resolve subagent settings file (--settings flag to the CC harness CLI).
    # If configured, the agent spawn inherits that settings file's model / hooks.
    # If not configured, no --settings flag → the harness uses default settings.json.
    extra_args = _build_extra_args(env, agent_type)

    # gap#2 airtight graybox seal (a-fs): build the bwrap mount-namespace prefix when this
    # workspace is graybox-enabled. FAIL-LOUD if requested but unbuildable — NEVER silently
    # spawn unsandboxed (that would defeat the seal = the exact cheat path the seal closes).
    sandbox_prefix: Optional[list[str]] = None
    if graybox_cfg["enabled"]:
        import sys as _sys
        _wf = Path(__file__).resolve().parents[1] / "workflow"
        if str(_wf) not in _sys.path:
            _sys.path.insert(0, str(_wf))
        import graybox_sandbox as _gs
        if not _gs.isolation_available():
            raise RuntimeError(
                "migration authoring requires strict platform isolation, but no "
                "supported backend is available (Linux: bwrap; macOS: "
                "sandbox-exec) — refusing to spawn unsandboxed"
            )
        repo_root = Path(__file__).resolve().parents[3]  # engine/
        # 2026-07-05: KB relocated to <plugin_root>/kb/ (repo_root.parent == plugin_root).
        kb_dir = repo_root.parent / "kb"
        allow_ro, allow_rw = _gs.graybox_allow_set(
            workspace, kb_dir=kb_dir, arch22_dir=graybox_cfg["arch22_dir"]
        )
        _gs.write_construction_manifest(
            workspace, allow_ro, allow_rw,
            inner_cmd=_backend_manifest_cmd(agent_type),
        )
        # An empty inner command yields an opaque prefix; agent_transport appends
        # the backend argv inside the selected platform sandbox.
        # Network access is excluded from the migration authoring boundary.
        sandbox_prefix = _gs.build_isolated_cmd(
            [], allow_ro=allow_ro, allow_rw=allow_rw, workdir=workspace
        )

    if background:
        if output_file is None:
            raise ValueError("output_file required when background=True")
        if agent_type in _ACTIVE_AGENT_MARKERS:
            raise NotImplementedError(
                f"background dispatch for {agent_type} needs explicit active-marker cleanup"
            )
        return _backend.dispatch(
            agent_type, brief, kind="agent", mode="background",
            output_file=output_file, extra_args=extra_args, sandbox_prefix=sandbox_prefix,
        )
    else:
        tee = workspace / f".cc_stream_log_{agent_type}_{spawn_index}.jsonl"
        progress_cb = _make_progress_printer(agent_type, spawn_index)
        # The marker describes THIS dispatch, so the previous one's verdict is dropped before
        # the agent runs. Without this the first stop-gate failure for an agent type wedged
        # the workspace permanently: `_post_spawn_transition` checks the file after every
        # later spawn, so each retry was killed on sight by the stale reason no matter what
        # its own gate said — including the retry that fixed the artifacts. Clearing it here
        # rather than in `_run_stop_gate` also covers the dispatch that raises before the
        # gate is ever reached.
        _clear_stop_gate_marker(workspace, agent_type)
        with _active_agent_marker(workspace, agent_type):
            result = _backend.dispatch(
                agent_type, brief, kind="agent", mode="streaming",
                tee_path=tee, timeout=timeout_sec,
                progress_callback=progress_cb,
                extra_args=extra_args,
                silence_timeout=None,
                sandbox_prefix=sandbox_prefix,
            )
        _run_stop_gate(workspace, agent_type, result)
        persist_envelope(workspace, agent_type, result, spawn_index=spawn_index, brief=brief)
        return result


# Claude Code fires the agent stop gates from its own SubagentStop hook. No other harness
# has that event, but under this dispatch model the ORCHESTRATOR owns the completion: the
# gated agents (see STOP_GATES in hooks/agent-gate-dispatch.py) are spawned exclusively
# here, one process each, so the gate can be invoked where the process is reaped instead of
# requiring a harness-side hook.
#
# Semantics differ from CC in one way that must not be papered over: CC's SubagentStop can
# block the agent from exiting so the SAME agent fixes its artifacts, whereas here the
# process is already gone. So a failed gate cannot be "repaired in place" — it is turned
# into a FAILED dispatch, which is the honest representation and keeps the FSM from
# treating unvalidated artifacts as a completed spawn.
_STOP_GATE_MARKER = ".agent_gate_stop_failed"


def _run_stop_gate(workspace, agent_type: str, result) -> None:
    if _backend.name == "claude_code":
        return  # already covered by the harness hook; running it here would double-fire
    gate = Path(__file__).resolve().parents[4] / "hooks" / "agent-gate-dispatch.py"
    if not gate.is_file():
        _mark_stop_gate_failure(
            workspace, agent_type, result,
            f"agent gate dispatcher missing at {gate}; a gate we cannot run is not a gate "
            "that passed",
        )
        return
    # cwd MUST be the ENGINE root, not the op workspace. hooks/v3/_common.sh resolves
    # `${WORKSPACE_ROOT:-workspace}` RELATIVE to cwd and returns "" when that directory is
    # absent — and check_worker.sh treats "" as "nothing to check" and exits 0. Running from
    # inside workspace/<op> therefore looks for workspace/<op>/workspace, finds nothing, and
    # every stop gate passes unconditionally. That early return happens BEFORE the
    # CLAUDE_ACTIVE_WORKSPACE branch, so the env var alone cannot rescue it; Claude Code
    # never hit this because its hooks run with the engine as cwd.
    engine_root = Path(__file__).resolve().parents[3]
    payload = json.dumps({
        "hook_event_name": "SubagentStop",
        "agent_type": agent_type,
        "agent_id": f"{_backend.name}:{agent_type}",
        "cwd": str(engine_root),
    })
    gate_env = dict(os.environ)
    gate_env["CLAUDE_ACTIVE_WORKSPACE"] = str(workspace)
    gate_env["ASCENDC_WORKSPACE"] = str(workspace)
    try:
        proc = subprocess.run(
            [sys.executable, str(gate), "stop"],
            input=payload, text=True, capture_output=True,
            cwd=str(engine_root), env=gate_env, timeout=120,
        )
    except Exception as exc:  # a gate we cannot run is not a gate that passed
        _mark_stop_gate_failure(workspace, agent_type, result, f"stop gate could not run: {exc!r}")
        return
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:1500]
        _mark_stop_gate_failure(
            workspace, agent_type, result,
            f"stop gate rejected {agent_type} (rc={proc.returncode}): {detail}",
        )


def _clear_stop_gate_marker(workspace, agent_type: str) -> None:
    # missing_ok=True avoids a bare except: an absent marker is the normal case.
    (Path(workspace) / f"{_STOP_GATE_MARKER}_{agent_type}").unlink(missing_ok=True)


def _mark_stop_gate_failure(workspace, agent_type: str, result, reason: str) -> None:
    _STOP_GATE_LOG.error("[stop-gate] %s", reason)
    try:
        (Path(workspace) / f"{_STOP_GATE_MARKER}_{agent_type}").write_text(reason + "\n")
    except OSError as exc:
        # The verdict is already logged; a marker we cannot persist must not be swallowed.
        _STOP_GATE_LOG.warning("[stop-gate] cannot persist failure marker: %s", exc)
    if result is not None:
        result.is_error = True
        result.output_text = f"{reason}\n\n{result.output_text or ''}"


def _make_progress_printer(agent_type: str, spawn_index: int):
    """Build a closure for spawn_agent_streaming progress_callback.

    Prints one terse line per tool_use (so user sees live phase activity)
    and the first line of any assistant-text block (for thinking-out-loud
    progress). Truncates aggressively to keep the log readable.
    """
    prefix = f"  [{agent_type}-{spawn_index}]"

    def _cb(event: dict) -> None:
        etype = event.get("type")
        if etype != "assistant":
            return
        msg = event.get("message", {}) or {}
        for blk in msg.get("content") or []:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "tool_use":
                tname = blk.get("name", "?")
                inp = blk.get("input") or {}
                # Build a short description of the tool call
                if tname == "Bash":
                    cmd = (inp.get("command") or "")[:100]
                    desc = (inp.get("description") or "")[:60]
                    print(f"{prefix} Bash: {desc or cmd}")
                elif tname == "Read":
                    fp = inp.get("file_path") or ""
                    # Trim to last 2 path components for readability
                    parts = Path(fp).parts
                    short = "/".join(parts[-2:]) if len(parts) >= 2 else fp
                    print(f"{prefix} Read: {short}")
                elif tname == "Edit":
                    fp = inp.get("file_path") or ""
                    print(f"{prefix} Edit: {Path(fp).name}")
                elif tname == "Write":
                    fp = inp.get("file_path") or ""
                    print(f"{prefix} Write: {Path(fp).name}")
                elif tname in ("Grep", "Glob"):
                    pat = inp.get("pattern") or inp.get("query") or ""
                    print(f"{prefix} {tname}: {pat[:80]}")
                elif tname == "Skill":
                    skill = inp.get("skill") or "?"
                    print(f"{prefix} Skill: {skill}")
                elif tname == "Agent":
                    sub = inp.get("subagent_type") or "?"
                    print(f"{prefix} Agent: {sub}")
                else:
                    print(f"{prefix} {tname}")
            elif btype == "text":
                text = (blk.get("text") or "").strip()
                if not text:
                    continue
                first = text.split("\n", 1)[0]
                if first:
                    print(f"{prefix} | {first[:140]}")

    return _cb


def _build_extra_args(env, agent_type: str) -> list[str] | None:
    """Return backend-specific extra args for agent dispatch.

    Reads env.subagent_settings (parsed from .ascendc_env SUBAGENT_SETTINGS_* keys).
    Currently only Claude Code supports these settings files via `--settings`.
    Other harness backends must not receive Claude-only argv.
    """
    if getattr(_backend, "name", "") != "claude_code":
        return None
    settings_file = env.get_subagent_settings(agent_type) if env else None
    if settings_file:
        # Resolve relative paths against the project root
        resolved = Path(settings_file)
        if not resolved.is_absolute():
            # Resolve relative to repo root (same dir as workspace/)
            _repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            resolved = _repo_root / resolved
        return ["--settings", str(resolved)]
    return None


def _backend_manifest_cmd(agent_type: str) -> list[str]:
    """Best-effort command shape for graybox construction manifests."""
    backend_name = getattr(_backend, "name", "")
    if backend_name == "claude_code":
        return [os.environ.get("CLAUDE_BIN", "claude"), "--agent", agent_type]
    if backend_name == "codex":
        return [os.environ.get("AOG_CODEX_BIN", "codex"), "exec", f"agent:{agent_type}"]
    if backend_name == "opencode":
        return [os.environ.get("AOG_OPENCODE_BIN", "opencode"), "run", f"agent:{agent_type}"]
    return [backend_name or "unknown-backend", f"agent:{agent_type}"]


def persist_envelope(
    workspace: Path,
    agent_type: str,
    result: Envelope,
    *,
    spawn_index: int,
    brief: Optional[str] = None,
) -> None:
    """Append the raw claude envelope to workspace/.cc_envelope_log.jsonl.

    Codex review #6: orchestrator did not log raw envelopes, making forensic
    debugging impossible. This is the post-spike production fix.
    """
    log = workspace / ".cc_envelope_log.jsonl"
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "agent_type": agent_type,
        "spawn_index": spawn_index,
        "success": result.success,
        "is_error": result.is_error,
        "duration_ms": result.duration_ms,
        "cost_usd": result.cost_usd,
        "session_id": result.session_id,
        "terminal_reason": result.terminal_reason,
        "permission_denials": result.raw_envelope.get("permission_denials", []),
        "num_turns": result.raw_envelope.get("num_turns"),
        "stop_reason": result.raw_envelope.get("stop_reason"),
        # The full agent text output may be large; keep but truncated
        "output_text_tail_2k": result.output_text[-2000:] if result.output_text else "",
        "brief_head_2k": (brief[:2000] if brief else None),
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _state_to_counter(state: str) -> str:
    """Map state to its iter_counter name from YAML.

    Defaults to last word of state if not explicitly mapped.
    e.g. await_worker → worker, await_optimizer → optimizer.
    """
    if state.startswith("await_"):
        return state[len("await_"):]
    return state


# ---------------------------------------------------------------------------
# CLI for smoke-testing brief construction
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="agent_dispatch — preview a brief without spawning")
    ap.add_argument("--op", required=True)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--state", default="await_worker",
                    help="YAML state name (default: await_worker)")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--spawn-index", type=int, default=1)
    ap.add_argument("--directive", default=None)
    ap.add_argument("--print-only", action="store_true",
                    help="just print the brief, don't spawn")
    args = ap.parse_args()

    if not args.print_only:
        print("(--print-only required for CLI smoke; spawning a worker is expensive)")
        import sys
        sys.exit(2)

    agent_type = state_executor.next_agent(args.state)
    builder = BRIEF_BUILDERS.get(agent_type)
    if builder is None:
        print(f"agent_type {agent_type!r}: brief builder not yet implemented")
        sys.exit(2)
    env = load_env()
    iter_cap = state_executor.iter_cap(args.state, workspace=args.workspace)
    iter_count = state_executor.iter_count(args.workspace, _state_to_counter(args.state))
    brief = builder(
        args.op, args.workspace,
        lane=args.lane,
        spawn_index=args.spawn_index,
        iter_cap_remaining=iter_cap - iter_count,
        env=env,
        directive_text=args.directive,
    )
    print(brief)
