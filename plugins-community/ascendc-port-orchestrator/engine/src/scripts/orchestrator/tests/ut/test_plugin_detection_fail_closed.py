# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression coverage for fail-closed supported-workflow detection."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parents[2]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import agent_dispatch  # noqa: E402
import finalize_pipeline  # noqa: E402,F401  (initialize the finalize import cycle)
import finalize_checks_provenance  # noqa: E402
import finalize_checks_structural  # noqa: E402
import finalize_ge_ophost  # noqa: E402
import handoff_audit  # noqa: E402
import phase_o5  # noqa: E402
import phase_o5_runner  # noqa: E402
import phase_o5_verify  # noqa: E402
import plugins  # noqa: E402
import schema_norm  # noqa: E402
import state_executor  # noqa: E402


def _raise_detection_error(_workspace):
    raise OSError("plugin state is unreadable")


def _raise_ambiguity(_workspace):
    raise plugins.PluginAmbiguityError("ambiguous supported workflow")


@pytest.fixture
def broken_detection(monkeypatch):
    monkeypatch.setattr(plugins, "detect_plugin", _raise_detection_error)


def test_agent_dispatch_does_not_build_generic_brief_on_detection_error(
    tmp_path, monkeypatch, broken_detection,
):
    workspace = tmp_path / "grad"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "op": "grad",
        "opgen_mode": "backward",
    }))
    monkeypatch.setattr(
        agent_dispatch.state_executor,
        "next_agent",
        lambda _state: "aog-kernel-worker",
    )
    monkeypatch.setattr(
        agent_dispatch,
        "load_env",
        lambda: SimpleNamespace(opgen_mode="backward"),
    )

    with pytest.raises(OSError, match="plugin state is unreadable"):
        agent_dispatch.spawn_for_state(
            "grad", workspace, "await_worker", lane=0, spawn_index=1,
        )


def test_state_transition_does_not_skip_plugin_condition_on_detection_error(
    tmp_path, broken_detection,
):
    workspace = tmp_path / "migration"
    workspace.mkdir()

    with pytest.raises(OSError, match="plugin state is unreadable"):
        state_executor.next_state(
            workspace,
            "→ orchestrator: research_blocked",
            from_state="await_researcher",
            dry_run=True,
        )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda ws: getattr(phase_o5_runner, '_is_port_a3_mode')(ws),
            id="phase-o5-mode",
        ),
        pytest.param(
            lambda ws: getattr(phase_o5_verify, '_run_canonical_pass_a_local')(
                ws, "op", {}, lane=0,
            ),
            id="phase-o5-local-verify",
        ),
        pytest.param(
            lambda ws: getattr(finalize_checks_provenance, '_check_ge_ophost_raw_cann_copy')(ws),
            id="ge-provenance-gate",
        ),
        pytest.param(
            lambda ws: getattr(finalize_checks_structural, '_check_arch35_wrap_cheat')(ws),
            id="arch35-wrap-gate",
        ),
        pytest.param(
            lambda ws: finalize_ge_ophost.assemble_ge_ophost(ws),
            id="ge-host-assembly",
        ),
        pytest.param(
            lambda ws: getattr(schema_norm, '_resolve_perf_threshold')(ws, {}),
            id="performance-threshold",
        ),
    ],
)
def test_safety_and_verification_paths_propagate_detection_error(
    tmp_path, broken_detection, call,
):
    with pytest.raises(OSError, match="plugin state is unreadable"):
        call(tmp_path)


def test_architecture_gate_converts_ambiguity_to_explicit_rejection(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(plugins, "detect_plugin", _raise_ambiguity)

    reason = getattr(finalize_checks_structural, '_check_architecture_class')(tmp_path)

    assert reason is not None
    assert "SOURCE_ARCH_UNVERIFIED" in reason
    assert "PluginAmbiguityError" in reason


def test_truth_source_converts_ambiguity_to_hard_error(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins, "detect_plugin", _raise_ambiguity)

    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(tmp_path)


def test_freshness_helper_uses_conservative_union_on_ambiguity(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(plugins, "detect_plugin", _raise_ambiguity)

    assert handoff_audit.delegation_cpp_dir_names(tmp_path) == [
        "kernel",
        "op_host",
        "op_kernel",
    ]


def test_only_explicit_fail_closed_boundaries_catch_detection_errors():
    """Keep ordinary RuntimeError inheritance safe as call sites evolve."""
    scripts = _HERE.parents[3]
    caught_sites = set()
    catching_names = {
        "Exception",
        "BaseException",
        "RuntimeError",
        "PluginAmbiguityError",
    }

    for path in scripts.rglob("*.py"):
        if "tests" in path.parts or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {}
        aliases = set()
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
            if isinstance(node, ast.ImportFrom) and node.module == "plugins":
                aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "detect_plugin"
                )

        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in aliases
            ):
                continue
            function_name = "<module>"
            ancestor = call
            while ancestor in parents:
                ancestor = parents[ancestor]
                if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_name = ancestor.name
                    break

            child = call
            while child in parents:
                parent = parents[child]
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    break
                if isinstance(parent, ast.Try) and child in parent.body:
                    handler_names = set()
                    catches_all = False
                    for handler in parent.handlers:
                        if handler.type is None:
                            catches_all = True
                        elif isinstance(handler.type, ast.Name):
                            handler_names.add(handler.type.id)
                        elif isinstance(handler.type, ast.Tuple):
                            handler_names.update(
                                item.id
                                for item in handler.type.elts
                                if isinstance(item, ast.Name)
                            )
                    if catches_all or handler_names & catching_names:
                        caught_sites.add((
                            path.relative_to(scripts).as_posix(),
                            function_name,
                        ))
                child = parent

    assert caught_sites == {
        (
            "orchestrator/finalize_checks_structural.py",
            "_check_architecture_class",
        ),
        ("orchestrator/handoff_audit.py", "delegation_cpp_dir_names"),
        ("orchestrator/phase_o5.py", "expected_truth_source"),
    }
