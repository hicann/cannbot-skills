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
import signal
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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
from source_arch import (
    verify_generation_source_stage,
    verify_npubench_generation_source_stage,
)
from briefs.ko_brief import build_optimizer_brief
from briefs.ar_brief import build_researcher_brief
from briefs.fo_brief import build_fused_optimizer_brief
from briefs.da_brief import build_det_analyzer_brief
from briefs.cl_brief import build_cann_learner_brief


_STOP_GATE_LOG = logging.getLogger(__name__)


TILELANG2ASCENDC_SOURCE_KIND = "port-aclnn-tilelang2ascendc"


def _tilelang2ascendc_source_api():
    """Return the independent TileLang2AscendC source-stage verifier."""
    try:
        from tilelang2ascendc_source import verify_tilelang2ascendc_source_stage
    except ImportError as exc:
        raise RuntimeError(
            "port-aclnn-tilelang2ascendc requires the tilelang2ascendc_source staging adapter"
        ) from exc
    return verify_tilelang2ascendc_source_stage


def _is_tilelang2ascendc_source_state(state: dict) -> bool:
    port_source = state.get("port_source")
    return (
        state.get("source_kind") == TILELANG2ASCENDC_SOURCE_KIND
        or (
            isinstance(port_source, dict)
            and port_source.get("kind") == TILELANG2ASCENDC_SOURCE_KIND
        )
    )


def _tilelang2ascendc_stage_mismatch(port_source: dict, state: dict) -> bool:
    """Report whether staged metadata disagrees with the durable port_source.

    Split out of the caller so the comparison stays a single named predicate
    instead of one oversized boolean expression.  Evaluation order (and the
    short-circuit on the first mismatch) is unchanged.
    """
    paired = (
        ("stage_root", "port_a3_source"),
        ("manifest", "source_stage_manifest"),
        ("digest", "source_stage_digest"),
        ("file_count", "source_stage_file_count"),
    )
    for source_key, state_key in paired:
        if port_source.get(source_key) != state.get(state_key):
            return True
    pinned = (
        ("source_arch", "arch35"),
        ("target_arch", "arch35"),
        ("root_layout", "model_and_kernel"),
    )
    for key, expected in pinned:
        if port_source.get(key) != expected:
            return True
    return state.get("graybox_source_dir") != state.get("port_a3_source")


def _verify_tilelang2ascendc_source_binding(workspace: Path, state: dict):
    """Validate the durable TileLang2AscendC identity before graybox spawn."""
    port_source = state.get("port_source")
    if state.get("source_kind") != TILELANG2ASCENDC_SOURCE_KIND:
        return False, "TileLang2AscendC source_kind is missing or mismatched", None
    if not isinstance(port_source, dict) or port_source.get("kind") != TILELANG2ASCENDC_SOURCE_KIND:
        return False, "TileLang2AscendC port_source.kind is missing or mismatched", None
    if state.get("source_arch") != "arch35" or state.get("target_arch") != "arch35":
        return False, "TileLang2AscendC route must persist source_arch=target_arch=arch35", None
    required = {
        "port_a3_source": "stage root",
        "source_stage_manifest": "stage manifest",
        "source_stage_digest": "stage digest",
        "source_stage_file_count": "stage file count",
        "graybox_source_dir": "graybox source bind",
    }
    for field, label in required.items():
        value = state.get(field)
        if value is None or value == "":
            return False, f"TileLang2AscendC {label} is missing", None
    if _tilelang2ascendc_stage_mismatch(port_source, state):
        return False, "TileLang2AscendC stage metadata disagrees with durable port_source", None
    try:
        return _tilelang2ascendc_source_api()(workspace, state)
    except Exception as exc:
        return False, f"TileLang2AscendC source-stage verifier unavailable or failed: {exc}", None

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

# NPUKernelBench's frozen task is its functional oracle.  A migration worker
# can still receive the staged arch22 source as implementation provenance, but
# it must not inherit an A3 connection or a transient private-source pointer
# that could turn the old live-A3 route into an implicit fallback.
_NPUBENCH_STRIPPED_ENV_NAMES = frozenset(
    {
        "CANNBOT_PORT_A3_BUILD_SOURCE",
        "PORT_A3_REFERENCE_SOURCE",
        "PORT_A3_SOURCE",
        # A child can otherwise redirect load_env() to an unrelated host-side
        # configuration file after the dispatcher has already selected the
        # frozen provider.
        "ASCENDC_ENV_PATH",
        "ASCENDC_ENV_FILE",
    }
)


def _npubench_sanitized_prefix(sandbox_prefix: list[str]) -> list[str]:
    """Append a child-only A3 environment scrubber to a sandbox prefix.

    The backend transport otherwise inherits this orchestrator process's
    environment.  Keep auth/toolchain variables intact, but remove every A3
    setting and the private live-source pointers before the harness process is
    exec'd.  ``/usr/bin/env -u`` works after both supported sandbox prefixes:
    bwrap's terminal ``--`` and sandbox-exec's shell wrapper both treat it as
    the command to execute.  No parent environment is mutated.
    """
    names = sorted(
        name
        for name in os.environ
        if name.startswith("A3_") or name in _NPUBENCH_STRIPPED_ENV_NAMES
    )
    command = ["/usr/bin/env"]
    for name in names:
        command.extend(["-u", name])
    return [*sandbox_prefix, *command]


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
    # agent_transport owns the child-process signal/reap handler.  Install it
    # before wrapping the signals so our wrapper can clear the marker and then
    # delegate to the transport handler (which terminates the child process
    # group and re-raises the original signal).  Without this ordering, a
    # SIGINT/SIGTERM delivered to the foreground launcher can kill Python
    # before the context-manager finally block runs, leaving a false
    # ``.kernel_worker_active`` marker that wedges the next resume.
    # Bind the transport hook inside the function body (not at module import)
    # so tests that replace it on the transport module still see their
    # replacement, and so a backend without the hook raises ImportError here.
    try:
        from agent_transport import _install_cleanup_handlers as install_cleanup_handlers

        install_cleanup_handlers()
    except (ImportError, AttributeError, OSError, ValueError) as error:
        # Backends/tests that do not expose the transport cleanup hook still
        # get the normal finally-based marker lifecycle.
        _STOP_GATE_LOG.debug(
            "transport cleanup hook unavailable; using marker fallback: %s",
            error,
        )
    previous_handlers: dict[int, object] = {}

    def _clear_then_delegate(signum: int, frame) -> None:
        marker.unlink(missing_ok=True)
        previous = previous_handlers.get(signum, signal.SIG_DFL)
        if previous is signal.SIG_IGN:
            return
        if callable(previous):
            previous(signum, frame)
            return
        signal.signal(signum, previous)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _clear_then_delegate)
        except (OSError, ValueError):
            # Signal registration is only available in the main thread; the
            # context-manager finally block remains the fallback elsewhere.
            previous_handlers.pop(signum, None)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError) as error:
                _STOP_GATE_LOG.debug(
                    "could not restore signal handler for %s: %s", signum, error
                )


def _read_dispatch_state(state_file: Path) -> dict:
    """Read the durable dispatch state, rejecting malformed or linked files."""
    if not state_file.exists():
        return {}
    try:
        if state_file.is_symlink():
            raise RuntimeError("durable state must not be a symlink")
        state_data = json.loads(state_file.read_text())
        if not isinstance(state_data, dict):
            raise TypeError("durable state is not a JSON object")
        return state_data
    except Exception as exc:
        raise RuntimeError(
            f"refusing agent spawn with unreadable durable state: {exc}"
        ) from exc


def _verify_migration_source_stage(workspace: Path, state_data: dict, reference) -> bool:
    """Verify the immutable source stage selected by a migration workspace."""
    tilelang_source = _is_tilelang2ascendc_source_state(state_data)
    if tilelang_source:
        source_stage_verifier = _verify_tilelang2ascendc_source_binding
    else:
        source_stage_verifier = (
            verify_npubench_generation_source_stage
            if reference.get("source") == "npubench"
            else verify_generation_source_stage
        )
    valid_stage, stage_reason, _stage_manifest = source_stage_verifier(
        workspace, state_data
    )
    if not valid_stage:
        raise RuntimeError(
            "refusing migration agent spawn with invalid source-only snapshot: "
            f"{stage_reason}"
        )
    return tilelang_source


def _bind_migration_source(env, state_data: dict, tilelang_source: bool) -> None:
    """Bind the already verified target-format source into the child environment."""
    if tilelang_source:
        tilelang_stage = state_data.get("port_a3_source")
        if not isinstance(tilelang_stage, str) or not tilelang_stage:
            raise RuntimeError("migration workspace is missing a verified target-format source stage")
        env.port_a3_source = tilelang_stage
        return

    legacy_source = state_data.get("port_source")
    legacy_stage = state_data.get("port_a3_source")
    if not legacy_stage and isinstance(legacy_source, dict):
        legacy_stage = legacy_source.get("stage_root")
    if not isinstance(legacy_stage, str) or not legacy_stage:
        raise RuntimeError("migration workspace is missing a verified port source stage")
    env.port_a3_source = legacy_stage


def _verify_npubench_bundle(workspace: Path, reference) -> None:
    """Verify the frozen NPUKernelBench task bundle before a worker spawn."""
    try:
        from npubench.npubench_inputs import verify_npubench_stage

        valid_bundle, bundle_reason, _bundle_manifest = verify_npubench_stage(
            workspace, reference
        )
    except Exception as exc:
        raise RuntimeError(
            "refusing NPUKernelBench migration spawn with unreadable "
            f"frozen task bundle: {type(exc).__name__}: {exc}"
        ) from exc
    if not valid_bundle:
        raise RuntimeError(
            "refusing NPUKernelBench migration spawn with invalid frozen "
            f"task bundle: {bundle_reason}"
        )


def _build_migration_graybox_config(
    workspace: Path,
    state_file: Path,
    state_data: dict,
    env,
) -> dict:
    """Build the graybox configuration for an authenticated migration state."""
    try:
        from reference_source import NPUBENCH, resolve_reference_binding

        reference = resolve_reference_binding(state_data)
    except Exception as exc:
        raise RuntimeError(
            f"refusing migration agent spawn with invalid reference binding: {exc}"
        ) from exc

    tilelang_source = _verify_migration_source_stage(workspace, state_data, reference)
    _bind_migration_source(env, state_data, tilelang_source)
    legacy_source = state_data.get("port_source")
    graybox_cfg = {
        "enabled": True,
        "source_kind": TILELANG2ASCENDC_SOURCE_KIND if tilelang_source else None,
        "arch22_dir": state_data.get("graybox_source_dir") or state_data.get("graybox_arch22_dir") or (
            state_data.get("port_a3_source")
            or (legacy_source.get("stage_root") if isinstance(legacy_source, dict) else None)
        ),
        "npubench_bundle": None,
        "npubench_state": None,
    }
    if not graybox_cfg.get("arch22_dir"):
        raise RuntimeError("migration workspace has no graybox source stage")
    if reference.get("source") == NPUBENCH:
        _verify_npubench_bundle(workspace, reference)
        bundle_sha256 = reference.get("bundle_sha256")
        graybox_cfg["npubench_bundle"] = (
            workspace / "reference_inputs" / "npubench" / bundle_sha256
        )
        graybox_cfg["npubench_state"] = state_file
    return graybox_cfg


def _resolve_dispatch_config(workspace: Path, env) -> dict:
    """Overlay durable state onto the static environment and resolve graybox mode."""
    state_file = workspace / ".opgen_state.json"
    state_data = _read_dispatch_state(state_file)
    graybox_cfg = {
        "enabled": False,
        "arch22_dir": None,
        "source_kind": None,
        "npubench_bundle": None,
        "npubench_state": None,
    }
    ws_mode = state_data.get("opgen_mode")
    if ws_mode and ws_mode != env.opgen_mode:
        env.opgen_mode = ws_mode
    if ws_mode == "port_a3_to_a5":
        return _build_migration_graybox_config(workspace, state_file, state_data, env)
    if env.opgen_mode == "port_a3_to_a5":
        raise RuntimeError("refusing migration agent spawn without a verified migration state")
    return graybox_cfg


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
        background: if True, returns Popen for legacy non-graybox dispatch;
            isolated graybox dispatch is intentionally rejected because its
            staged plugin runtime must live until the worker exits
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

    # Task #48 (2026-05-13): overlay workspace-level opgen_mode and the
    # authenticated source binding from .opgen_state.json onto the static env.
    # Keeping this policy in one helper also keeps the dispatch entry point shallow.
    graybox_cfg = _resolve_dispatch_config(workspace, env)

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
    brief = _build_agent_brief(
        builder,
        op,
        workspace,
        agent_type,
        lane=lane,
        spawn_index=spawn_index,
        cap_remaining=cap_remaining,
        env=env,
        directive_text=directive_text,
        handoff_from_prior=handoff_from_prior,
        plugin=plugin,
    )

    # Resolve subagent settings file (--settings flag to the CC harness CLI).
    # If configured, the agent spawn inherits that settings file's model / hooks.
    # If not configured, no --settings flag → the harness uses default settings.json.
    extra_args = _build_extra_args(env, agent_type)

    # gap#2 airtight graybox seal (a-fs): build the bwrap mount-namespace prefix when this
    # workspace is graybox-enabled. FAIL-LOUD if requested but unbuildable — NEVER silently
    # spawn unsandboxed (that would defeat the seal = the exact cheat path the seal closes).
    sandbox_prefix: Optional[list[str]] = None
    graybox_allow_ro: list[tuple[str, str]] | None = None
    graybox_sandbox_module = None
    graybox_construction_manifest = None
    graybox_construction_manifest_sha256 = None
    graybox_dispatch = _prepare_graybox_dispatch(
        workspace, agent_type, graybox_cfg, extra_args
    )
    sandbox_prefix = graybox_dispatch.sandbox_prefix
    graybox_allow_ro = graybox_dispatch.graybox_allow_ro
    graybox_sandbox_module = graybox_dispatch.graybox_sandbox_module
    graybox_construction_manifest = graybox_dispatch.construction_manifest
    graybox_construction_manifest_sha256 = graybox_dispatch.manifest_sha256
    extra_args = graybox_dispatch.extra_args
    if background:
        if graybox_allow_ro is not None:
            # The legacy background contract returns a bare Popen and has no
            # reap/finally callback through which this dispatcher could retain
            # and later remove the read-only plugin snapshot.  Reject this
            # combination explicitly rather than either leaking the snapshot
            # or deleting it while the worker still has it mounted.
            graybox_sandbox_module.cleanup_staged_plugin_runtimes(graybox_allow_ro)
            raise NotImplementedError(
                "background dispatch is unsupported for isolated graybox workers; "
                "the plugin runtime must remain mounted until the worker exits"
            )
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
    try:
        with _active_agent_marker(workspace, agent_type):
            result = _backend.dispatch(
                agent_type, brief, kind="agent", mode="streaming",
                tee_path=tee, timeout=timeout_sec,
                progress_callback=progress_cb,
                extra_args=extra_args,
                silence_timeout=None,
                sandbox_prefix=sandbox_prefix,
            )
        if (
            graybox_sandbox_module is not None
            and graybox_construction_manifest is not None
            and graybox_construction_manifest_sha256 is not None
        ):
            graybox_sandbox_module.verify_construction_manifest(
                graybox_construction_manifest,
                graybox_construction_manifest_sha256,
            )
        _run_stop_gate(workspace, agent_type, result)
        persist_envelope(workspace, agent_type, result, spawn_index=spawn_index, brief=brief)
        return result
    finally:
        if graybox_allow_ro is not None:
            graybox_sandbox_module.cleanup_staged_plugin_runtimes(graybox_allow_ro)


_BRIEF_HANDOFF_KEYS = {
    "aog-kernel-worker": ("directive_text", "handoff_from_prior_agent"),
    "aog-precision-probe": ("handoff_from_worker",),
    "aog-kernel-optimizer": ("directive_text", "handoff_from_worker"),
    "aog-researcher": ("handoff_from_prior_agent", "directive_text"),
    "aog-fused-optimizer": ("directive_text", "handoff_from_prior"),
    "aog-determinism-analyzer": ("handoff_from_prior",),
    "aog-cann-learner": ("handoff_from_prior_agent", "directive_text"),
}


def _build_agent_brief(
    builder,
    op: str,
    workspace: Path,
    agent_type: str,
    *,
    lane: int,
    spawn_index: int,
    cap_remaining: int,
    env,
    directive_text: Optional[str],
    handoff_from_prior: Optional[str],
    plugin,
):
    """Build a role-specific brief without nesting the dispatch branches."""
    handoff_values = {
        "directive_text": directive_text,
        "handoff_from_prior_agent": handoff_from_prior,
        "handoff_from_worker": handoff_from_prior,
        "handoff_from_prior": handoff_from_prior,
    }
    handoff_keys = _BRIEF_HANDOFF_KEYS.get(agent_type)
    if handoff_keys is None:
        raise NotImplementedError(
            f"brief signature for {agent_type!r} not handled in dispatch"
        )
    specific_kwargs = {key: handoff_values.get(key) for key in handoff_keys}
    return builder(
        op,
        workspace,
        lane=lane,
        spawn_index=spawn_index,
        iter_cap_remaining=cap_remaining,
        env=env,
        plugin=plugin,
        **specific_kwargs,
    )


def _prepare_opencode_graybox_runtime(
    repo_root: Path,
    extra_ro: list[Path],
) -> None:
    """Prepare the read-only runtime trees needed by sandboxed OpenCode."""
    if os.environ.get("AOG_HARNESS_BACKEND", "") != "opencode":
        return

    for runtime_name in (".opencode-agents", ".opencode-runtime"):
        runtime_dir = repo_root / "workspace" / runtime_name
        if runtime_dir.is_dir():
            extra_ro.append(runtime_dir)

    try:
        from backends.opencode_backend import OpencodeBackend

        prepare_skills = getattr(OpencodeBackend, "prepare_skills_runtime", None)
        prepare_opencode = getattr(OpencodeBackend, "prepare_opencode_runtime", None)
        if not callable(prepare_skills) or not callable(prepare_opencode):
            raise RuntimeError("opencode runtime preparation API is unavailable")
        prepare_skills(repo_root)
        prepare_opencode(repo_root)
    except Exception:
        logging.getLogger(__name__).exception(
            "opencode graybox runtime prep failed; spawn will run with whatever is on disk"
        )

    skills_dir = repo_root / "workspace" / ".opencode-skills-runtime"
    if skills_dir.is_dir():
        extra_ro.append(skills_dir)


def _graybox_scan_sources(
    workspace: Path,
    source_dir,
    kb_dir: Path,
    repo_root: Path,
    extra_ro: list[Path],
) -> list[Path]:
    """Return task/source roots for the construction-manifest scan."""
    scan_srcs = [workspace, source_dir, kb_dir]
    opencode_skills = (repo_root / "workspace" / ".opencode-skills-runtime").resolve()
    for extra_source in extra_ro:
        if Path(extra_source).resolve() != opencode_skills:
            scan_srcs.append(extra_source)
    return scan_srcs


def _graybox_plugin_args(allow_ro: list[tuple[str, str]]) -> list[str]:
    """Return the isolated plugin argument required by Claude Code."""
    if getattr(_backend, "name", "") != "claude_code":
        return []
    import sys as _sys

    _wf = Path(__file__).resolve().parents[1] / "workflow"
    if str(_wf) not in _sys.path:
        _sys.path.insert(0, str(_wf))
    import graybox_sandbox as _gs

    return [
        "--plugin-dir",
        _gs.plugin_dir_for_isolation_backend(
            allow_ro,
            backend=_gs.isolation_backend(),
        ),
    ]


def _load_graybox_sandbox():
    """Load the isolation backend and fail closed when it is unavailable."""
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
    return _gs


def _graybox_extra_roots(graybox_cfg: dict, repo_root: Path) -> list[Path]:
    """Collect harness-managed read-only roots for a graybox dispatch."""
    extra_ro: list[Path] = []
    _prepare_opencode_graybox_runtime(repo_root, extra_ro)
    bundle = graybox_cfg.get("npubench_bundle")
    if bundle is not None:
        extra_ro.extend([bundle, graybox_cfg.get("npubench_state")])
    return extra_ro


@dataclass(frozen=True)
class _GrayboxDispatch:
    """Named result of graybox setup, including the cleanup handle."""

    sandbox_prefix: Optional[list[str]]
    graybox_allow_ro: list[tuple[str, str]] | None
    graybox_sandbox_module: object | None
    construction_manifest: object | None
    manifest_sha256: object | None
    extra_args: list[str] | None


@dataclass(frozen=True)
class _GrayboxBuildContext:
    """Shared inputs for the graybox mount and manifest builders."""

    workspace: Path
    agent_type: str
    graybox_cfg: dict
    repo_root: Path
    kb_dir: Path
    extra_ro: list[Path]
    graybox_sandbox: object


@dataclass(frozen=True)
class _GrayboxPayload:
    """Built graybox artifacts before the cleanup handle is attached."""

    plugin_arg: list[str]
    construction_manifest: object
    manifest_sha256: object
    sandbox_prefix: list[str]


def _build_graybox_payload(
    context: _GrayboxBuildContext,
    allow_ro,
    allow_rw,
) -> _GrayboxPayload:
    """Build the manifest and isolated command from an established mount set."""
    workspace = context.workspace
    graybox_cfg = context.graybox_cfg
    plugin_arg = _graybox_plugin_args(allow_ro)
    scan_srcs = _graybox_scan_sources(
        workspace,
        graybox_cfg.get("arch22_dir"),
        context.kb_dir,
        context.repo_root,
        context.extra_ro,
    )
    share_model_network = bool(os.environ.get("ANTHROPIC_BASE_URL", "").strip())
    construction_manifest = context.graybox_sandbox.write_construction_manifest(
        workspace,
        allow_ro,
        allow_rw,
        inner_cmd=[*_backend_manifest_cmd(context.agent_type), *plugin_arg],
        source_stage_root=graybox_cfg.get("arch22_dir"),
        scan_srcs=scan_srcs,
        share_net=share_model_network,
    )
    manifest_sha256 = context.graybox_sandbox.construction_manifest_sha256(
        construction_manifest
    )
    sandbox_prefix = context.graybox_sandbox.build_isolated_cmd(
        [],
        allow_ro=allow_ro,
        allow_rw=allow_rw,
        workdir=workspace,
        share_net=share_model_network,
    )
    if graybox_cfg.get("npubench_bundle") is not None:
        sandbox_prefix = _npubench_sanitized_prefix(sandbox_prefix)
    return _GrayboxPayload(
        plugin_arg=plugin_arg,
        construction_manifest=construction_manifest,
        manifest_sha256=manifest_sha256,
        sandbox_prefix=sandbox_prefix,
    )


def _build_graybox_dispatch(
    context: _GrayboxBuildContext,
    extra_args: list[str] | None,
) -> _GrayboxDispatch:
    """Build the isolated dispatch artifacts and retain cleanup state."""
    graybox_allow_ro = None
    try:
        allow_ro, allow_rw = context.graybox_sandbox.graybox_allow_set(
            context.workspace,
            kb_dir=context.kb_dir,
            arch22_dir=context.graybox_cfg.get("arch22_dir"),
            extra_ro=context.extra_ro,
            plugin_dir=context.repo_root.parent,
        )
        graybox_allow_ro = allow_ro
        payload = _build_graybox_payload(context, allow_ro, allow_rw)
        if payload.plugin_arg:
            extra_args = [*(extra_args or []), *payload.plugin_arg]
    except Exception:
        if graybox_allow_ro is not None:
            context.graybox_sandbox.cleanup_staged_plugin_runtimes(graybox_allow_ro)
        raise
    return _GrayboxDispatch(
        sandbox_prefix=payload.sandbox_prefix,
        graybox_allow_ro=graybox_allow_ro,
        graybox_sandbox_module=context.graybox_sandbox,
        construction_manifest=payload.construction_manifest,
        manifest_sha256=payload.manifest_sha256,
        extra_args=extra_args,
    )


def _prepare_graybox_dispatch(
    workspace: Path,
    agent_type: str,
    graybox_cfg: dict,
    extra_args: list[str] | None,
) -> _GrayboxDispatch:
    """Build the isolated dispatch prefix and retain cleanup state."""
    if not graybox_cfg.get("enabled"):
        return _GrayboxDispatch(None, None, None, None, None, extra_args)
    graybox_sandbox = _load_graybox_sandbox()
    repo_root = Path(__file__).resolve().parents[3]
    extra_ro = _graybox_extra_roots(graybox_cfg, repo_root)
    context = _GrayboxBuildContext(
        workspace=workspace,
        agent_type=agent_type,
        graybox_cfg=graybox_cfg,
        repo_root=repo_root,
        kb_dir=repo_root.parent / "kb",
        extra_ro=extra_ro,
        graybox_sandbox=graybox_sandbox,
    )
    return _build_graybox_dispatch(context, extra_args)


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
