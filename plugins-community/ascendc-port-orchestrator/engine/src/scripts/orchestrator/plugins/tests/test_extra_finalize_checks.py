# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P87 (2026-05-15) — plugin.extra_finalize_checks() registry test.

Verifies the option-A extensibility path: a plugin can register new
finalize-gate types WITHOUT changes to PluginProtocol or
finalize_pipeline dispatch site.

This is the "well-localized → registry-based" upgrade that lets the
PPT narrative claim "plugin can freely register additional checks",
not just "well-localized extensibility".

Test strategy: install a fake plugin via monkeypatch that overrides
extra_finalize_checks() to return a custom (gate_name, callable).
Run check_finalize_eligibility on a fake workspace and verify the
custom gate fires.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import finalize_pipeline as fp  # noqa: E402
from plugins import detect_plugin, get_plugin  # noqa: E402
from source_arch import stage_source_tree  # noqa: E402


def _seed_pass_workspace(tmp_path: Path, mode: str = "port_a3_to_a5") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    source = tmp_path / "source"
    source_kernel = source / "op_kernel" / "arch22"
    source_kernel.mkdir(parents=True)
    (source_kernel / "x.cpp").write_text(
        "class Source { void Process() {} };\n"
    )
    stage = stage_source_tree(source, ws)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "x",
        "opgen_mode": mode,
        "source_arch": "arch22",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "graybox_arch22_dir": str(stage.root),
        "graybox_sandbox": True,
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
    }))
    (ws / "model.py").write_text("x\n")
    (ws / "model_new_ascendc.py").write_text("x\nif __name__=='__main__': pass\n")
    deployed = ws / "build" / "deploy" / "model.py"
    object_file = ws / "build" / "model.py.o"
    shared_lib = ws / "build" / "libx.so"
    for path in (deployed, object_file, shared_lib):
        path.parent.mkdir(parents=True, exist_ok=True)
    deployed.write_bytes((ws / "model.py").read_bytes())
    object_file.write_bytes(b"object")
    shared_lib.write_bytes(b"shared")
    source_digest = hashlib.sha256((ws / "model.py").read_bytes()).hexdigest()
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS", "total": 8}},
        "performance": {
            "status": "PASS",
            "ratio": 2.0,
            # P0ee methodology_declaration gate (added 2026-05-26): perf.ratio>1
            # must positively declare symmetric measurement. Use the explicit
            # `method_symmetric` token so the gate accepts this fixture's
            # speedup claim and the test's synthetic gate gets a chance to fire.
            "method": "method_symmetric",
        },
        "truth_source": "a3_cann_via_v1_aclnn_direct",
        "build_evidence": {
            "compiled_provenance": {
                "source": "model.py",
                "deployed_source": "build/deploy/model.py",
                "object": "build/model.py.o",
                "shared_lib": "build/libx.so",
                "workspace_source_sha256": source_digest,
                "deploy_source_sha256": source_digest,
                "built_from_source_sha256": source_digest,
                "object_sha256": hashlib.sha256(object_file.read_bytes()).hexdigest(),
                "shared_lib_sha256": hashlib.sha256(shared_lib.read_bytes()).hexdigest(),
            },
        },
    }))
    (ws / "x_runner.cpp").write_text("// runner\n")
    (ws / "knowledge_update.md").write_text(
        "## Context\nFixture.\n\n## Findings\n- Fixture.\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\n- STUB-0\n\n## Anti-patterns avoided\nNone\n"
    )
    # Skip op_host check by writing 3 dummy files (PR4778 minimum)
    op_host = ws / "op_host"
    op_host.mkdir()
    (op_host / "x_def.cpp").write_text("// def\n")
    (op_host / "x_tiling.cpp").write_text("// tiling cpp\n")
    (op_host / "x_tiling.h").write_text("// tiling h\n")
    return ws


def test_scoped_plugins_extra_finalize_contract():
    """Both supported modes expose the extension hook through the protocol."""
    from plugins import all_plugins

    for plugin in all_plugins():
        result = plugin.extra_finalize_checks()
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_plugin_extra_check_fires_via_registry(tmp_path, monkeypatch):
    """The registry path actually works: install a fake extra check
    on the active plugin, run check_finalize_eligibility, verify the
    fake check fires.
    """
    ws = _seed_pass_workspace(tmp_path)
    plug = detect_plugin(ws)
    assert plug is not None and plug.name == "port_a3_to_a5"

    fired = {"calls": 0}

    def fake_check(workspace, vj):
        fired["calls"] += 1
        return "synthetic_p87_check rejected for test purposes"

    monkeypatch.setattr(
        plug, "extra_finalize_checks",
        lambda: [("synthetic_p87_check", fake_check)],
    )
    # Refresh delegation-scan marker so we don't hit that gate first
    (ws / ".delegation_scan_passed").write_text("ok")

    elig = fp.check_finalize_eligibility(ws)
    assert elig["eligible"] is False
    assert elig["gate"] == "synthetic_p87_check"
    assert "synthetic_p87_check rejected" in elig["reason"]
    assert fired["calls"] == 1


def test_plugin_extra_check_accept_passes_through(tmp_path, monkeypatch):
    """If extra check returns None, eligibility continues to remaining
    gates (or passes entirely).
    """
    ws = _seed_pass_workspace(tmp_path)
    plug = detect_plugin(ws)

    monkeypatch.setattr(
        plug, "extra_finalize_checks",
        lambda: [("synthetic_accept", lambda w, v: None)],
    )
    (ws / ".delegation_scan_passed").write_text("ok")

    elig = fp.check_finalize_eligibility(ws)
    # The fake doesn't reject; whether the overall verdict is eligible
    # depends on the other gates (KB_WRITEUP / op_host / etc.) which
    # this fixture may not satisfy. Key assertion: the fake didn't
    # turn into a false-positive reject.
    if elig["eligible"] is False:
        assert elig["gate"] != "synthetic_accept", (
            "synthetic_accept returned None but eligibility cited it as "
            "the rejecting gate — dispatch bug"
        )


def test_plugin_extra_check_exception_caught(tmp_path, monkeypatch):
    """If extra check raises, dispatch must convert to a rejection
    (not crash the entire finalize pipeline).
    """
    ws = _seed_pass_workspace(tmp_path)
    plug = detect_plugin(ws)

    def buggy_check(workspace, vj):
        raise ValueError("synthetic exception for test")

    monkeypatch.setattr(
        plug, "extra_finalize_checks",
        lambda: [("buggy_check", buggy_check)],
    )
    (ws / ".delegation_scan_passed").write_text("ok")

    elig = fp.check_finalize_eligibility(ws)
    assert elig["eligible"] is False
    assert elig["gate"] == "buggy_check"
    assert "ValueError" in elig["reason"] or "synthetic exception" in elig["reason"]


def test_plugin_extra_checks_failure_in_extras_method_caught(tmp_path, monkeypatch):
    """If extra_finalize_checks() itself raises, treat as empty list
    (don't crash). This protects orchestrator from broken plugin code.
    """
    ws = _seed_pass_workspace(tmp_path)
    plug = detect_plugin(ws)

    def buggy_method():
        raise RuntimeError("plugin author bug")

    monkeypatch.setattr(plug, "extra_finalize_checks", buggy_method)
    (ws / ".delegation_scan_passed").write_text("ok")

    # Should NOT raise — exception in extras-method is swallowed.
    elig = fp.check_finalize_eligibility(ws)
    # Verdict depends on other gates; key thing is no uncaught exception.
    assert "eligible" in elig
