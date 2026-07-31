# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Strict migration-sandbox dispatch gating."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))

import graybox_sandbox  # noqa: E402


def _drive_dispatch(
    monkeypatch, tmp_path, *, state_json: str, seed_valid_stage: bool = True
):
    """Drive spawn_for_state with all heavy deps stubbed; return captured spawn kwargs."""
    import agent_dispatch
    import agent_transport
    import state_executor
    import plugins as plugins_mod
    from agent_transport import AgentResult

    captured = {}

    def fake_spawn(agent_type, brief, **kwargs):
        captured["agent_type"] = agent_type
        captured["sandbox_prefix"] = kwargs.get("sandbox_prefix")
        return AgentResult(
            agent_type=agent_type, success=True, is_error=False, output_text="stub",
            duration_ms=0, cost_usd=0.0, session_id="stub", terminal_reason="completed",
            raw_envelope={"type": "result", "subtype": "success", "result": "stub",
                          "session_id": "stub"}, tool_uses=[],
        )

    monkeypatch.setattr(agent_transport, "spawn_agent_streaming", fake_spawn)
    monkeypatch.setattr(agent_dispatch, "persist_envelope", lambda *a, **k: None)
    monkeypatch.setitem(agent_dispatch.BRIEF_BUILDERS, "aog-kernel-worker",
                        lambda op, workspace, **kw: "stub brief")
    monkeypatch.setattr(state_executor, "next_agent", lambda state: "aog-kernel-worker")
    monkeypatch.setattr(state_executor, "iter_cap", lambda *a, **k: 10)
    monkeypatch.setattr(state_executor, "iter_count", lambda *a, **k: 0)
    monkeypatch.setattr(agent_dispatch, "_state_to_counter", lambda state: "worker")
    monkeypatch.setattr(plugins_mod, "detect_plugin", lambda ws: None)

    class _FakeEnv:
        backend = "ascendc"
        opgen_mode = "port_a3_to_a5"
        port_a3_source = None
        target = "a5"

        def get_subagent_settings(self, agent_type):
            return None

    # Patch the name the dispatcher actually calls (imported binding), not _common's.
    monkeypatch.setattr(agent_dispatch, "load_env", lambda: _FakeEnv())

    workspace = tmp_path / "flash_attention_score"
    workspace.mkdir()
    state = json.loads(state_json)
    if seed_valid_stage and state.get("opgen_mode") == "port_a3_to_a5":
        from source_arch import stage_source_tree

        source = tmp_path / "source"
        kernel = source / "op_kernel" / "arch22" / "op.h"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("class Op { void Process() {} };\n")
        stage = stage_source_tree(source, workspace)
        state.update(
            {
                "port_a3_source": str(stage.root),
                "source_stage_manifest": str(stage.manifest),
                "source_stage_digest": stage.digest,
                "graybox_arch22_dir": str(stage.root),
            }
        )
        state.setdefault("graybox_sandbox", True)
        fake_kb = tmp_path / "kb"
        fake_kb.mkdir()
        original_allow_set = graybox_sandbox.graybox_allow_set

        def _allow_set(ws, **kwargs):
            return original_allow_set(
                ws,
                kb_dir=fake_kb,
                arch22_dir=kwargs.get("arch22_dir"),
                toolchain_dirs=[],
            )

        monkeypatch.setattr(graybox_sandbox, "graybox_allow_set", _allow_set)
    (workspace / ".opgen_state.json").write_text(json.dumps(state))

    agent_dispatch.spawn_for_state(
        "flash_attention_score", workspace, "await_worker",
        lane=1, spawn_index=1, timeout_sec=300,
    )
    return captured, workspace


def test_migration_without_mandatory_sandbox_state_fails(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="sandbox is not mandatory"):
        _drive_dispatch(
            monkeypatch,
            tmp_path,
            state_json=(
                '{"opgen_mode": "port_a3_to_a5", '
                '"graybox_sandbox": false}'
            ),
        )


@pytest.mark.skipif(
    not graybox_sandbox.isolation_available(), reason="strict isolation unavailable"
)
def test_migration_builds_and_passes_platform_sandbox_prefix(monkeypatch, tmp_path):
    """graybox_sandbox: true → a bwrap prefix is BUILT and PASSED to the spawn (catches the
    pass-through-forgotten bug) + a construction_manifest.json is emitted asserting airtight.
    """
    captured, workspace = _drive_dispatch(
        monkeypatch, tmp_path,
        state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
    )
    prefix = captured["sandbox_prefix"]
    assert prefix is not None and isinstance(prefix, list) and prefix
    assert prefix[0].endswith(("bwrap", "sandbox-exec"))
    if graybox_sandbox.isolation_backend() == "bwrap":
        assert prefix[-1] == "--"
        assert "--unshare-net" in prefix
    else:
        assert "(deny network*)" in prefix[2]
    # cann / output must NOT appear among the bwrap binds (airtight by construction;
    # the LOAD-BEARING enforcement is assert_no_answer_paths, a path-based guard called
    # inside build_bwrap_cmd that RAISES on any bind under ~/workspace/cann — this is a
    # redundant smoke-check). Boundary-aware: a bare "/workspace/cann" substring
    # false-matches a legitimately-bound workspace under a `cann`-PREFIXED dir (e.g. the
    # cannbot bundle at /home/<u>/workspace/cannbot-skills-exp/...). Match the cann SOURCE
    # only — `cann` followed by `/`, whitespace, or end-of-string — not `cannbot`.
    import re
    joined = " ".join(prefix)
    assert not re.search(r"/workspace/cann(?:[/ ]|$)", joined), joined
    # manifest emitted + asserts airtight
    manifest = workspace / "construction_manifest.json"
    assert manifest.is_file()
    m = json.loads(manifest.read_text())
    assert m["assertions"]["airtight"] is True


def test_migration_without_platform_backend_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", None)
    monkeypatch.setattr(graybox_sandbox, "_SANDBOX_EXEC", None)
    with pytest.raises(RuntimeError, match="strict platform isolation"):
        _drive_dispatch(
            monkeypatch, tmp_path,
            state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
