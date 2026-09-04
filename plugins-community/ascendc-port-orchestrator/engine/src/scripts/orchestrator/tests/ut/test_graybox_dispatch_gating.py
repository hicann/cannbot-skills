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
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))

import graybox_sandbox  # noqa: E402


def _drive_dispatch(
    monkeypatch,
    tmp_path,
    *,
    state_json: str,
    seed_valid_stage: bool = True,
    npubench: bool = False,
    background: bool = False,
    output_file=None,
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
        from reference_source import explicit_a3_live_binding

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
        state.setdefault("reference", explicit_a3_live_binding())
        state.setdefault("graybox_sandbox", True)
        if npubench:
            from npubench.npubench_inputs import stage_npubench_inputs

            task_root = tmp_path / "npu_benchmark" / "level1"
            task_root.mkdir(parents=True)
            task = task_root / "3_Add.py"
            task.write_text(
                "from pathlib import Path\n"
                "def get_input_groups():\n"
                "    return [Path(__file__).with_suffix('.json').read_bytes()]\n"
            )
            task.with_suffix(".json").write_bytes(b'{"shape": [8]}\n')
            state["reference"] = stage_npubench_inputs(
                workspace, npubench_task=task, npubench_root=task_root
            ).state_block()
        fake_kb = tmp_path / "kb"
        fake_kb.mkdir()
        original_allow_set = graybox_sandbox.graybox_allow_set

        def _allow_set(ws, **kwargs):
            return original_allow_set(
                ws,
                kb_dir=fake_kb,
                arch22_dir=kwargs.get("arch22_dir"),
                extra_ro=kwargs.get("extra_ro", ()),
                toolchain_dirs=[],
                plugin_dir=kwargs.get("plugin_dir"),
                plugin_mount=kwargs.get("plugin_mount", graybox_sandbox.DEFAULT_PLUGIN_MOUNT),
            )

        monkeypatch.setattr(graybox_sandbox, "graybox_allow_set", _allow_set)
    (workspace / ".opgen_state.json").write_text(json.dumps(state))

    agent_dispatch.spawn_for_state(
        "flash_attention_score", workspace, "await_worker",
        lane=1, spawn_index=1, timeout_sec=300,
        background=background, output_file=output_file,
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
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
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


@pytest.mark.skipif(
    not graybox_sandbox.isolation_available(), reason="strict isolation unavailable"
)
def test_direct_worker_shares_network_only_for_explicit_model_endpoint(monkeypatch, tmp_path):
    """Kimi/Anthropic-compatible workers need endpoint access, without weakening FS bounds."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.example.invalid/coding/")
    captured, workspace = _drive_dispatch(
        monkeypatch, tmp_path,
        state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
    )
    prefix = captured["sandbox_prefix"]
    if graybox_sandbox.isolation_backend() == "bwrap":
        assert "--unshare-net" not in prefix
    else:
        assert "(deny network*)" not in prefix[2]
    manifest = json.loads((workspace / "construction_manifest.json").read_text())
    assert manifest["network"] == "shared"
    joined = " ".join(prefix)
    assert "/usr/local/Ascend" not in joined


def test_migration_without_platform_backend_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", None)
    monkeypatch.setattr(graybox_sandbox, "_SANDBOX_EXEC", None)
    with pytest.raises(RuntimeError, match="strict platform isolation"):
        _drive_dispatch(
            monkeypatch, tmp_path,
            state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
        )


def test_graybox_background_dispatch_is_rejected_and_cleans_runtime(
    monkeypatch, tmp_path
):
    # Exercise the intentional hard rejection rather than the unavailable-backend
    # gate; no isolated background process can safely outlive this dispatch's
    # ownership of the staged plugin runtime.
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    staged_roots = []
    original_cleanup = graybox_sandbox.cleanup_staged_plugin_runtimes

    def _cleanup(allow_ro, **kwargs):
        staged_roots.extend(graybox_sandbox.staged_plugin_runtime_roots(allow_ro))
        original_cleanup(allow_ro, **kwargs)

    monkeypatch.setattr(graybox_sandbox, "cleanup_staged_plugin_runtimes", _cleanup)

    with pytest.raises(NotImplementedError, match="background dispatch is unsupported"):
        _drive_dispatch(
            monkeypatch,
            tmp_path,
            state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
            background=True,
            output_file=tmp_path / "background.json",
        )

    assert staged_roots
    assert all(not root.exists() for root in staged_roots)


def test_legacy_background_dispatch_still_returns_backend_handle(monkeypatch, tmp_path):
    import agent_dispatch
    import plugins as plugins_mod
    import state_executor

    captured = {}
    handle = object()

    class _Backend:
        @staticmethod
        def dispatch(*args, **kwargs):
            captured.update(kwargs)
            return handle

    class _Env:
        opgen_mode = "backward"
        port_a3_source = None

        @staticmethod
        def get_subagent_settings(_agent_type):
            return None

    monkeypatch.setattr(agent_dispatch, "_backend", _Backend())
    monkeypatch.setattr(agent_dispatch, "load_env", _Env)
    monkeypatch.setattr(plugins_mod, "detect_plugin", lambda _ws: None)
    monkeypatch.setattr(state_executor, "next_agent", lambda _state: "aog-precision-probe")
    monkeypatch.setattr(state_executor, "iter_cap", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(state_executor, "iter_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(agent_dispatch, "_state_to_counter", lambda _state: "probe")
    monkeypatch.setitem(
        agent_dispatch.BRIEF_BUILDERS,
        "aog-precision-probe",
        lambda _op, _workspace, **_kwargs: "legacy brief",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_file = tmp_path / "background.json"
    result = agent_dispatch.spawn_for_state(
        "op",
        workspace,
        "await_probe",
        lane=0,
        spawn_index=1,
        background=True,
        output_file=output_file,
    )

    assert result is handle
    assert captured["mode"] == "background"
    assert captured["output_file"] == output_file


def test_npubench_environment_scrubber_drops_all_live_provider_inputs(monkeypatch):
    """The scrubber must not mutate the parent but removes all child A3 vars."""
    # Bind the protected helper locally rather than reaching through the module
    # attribute, which CodeCheck flags as G.CLS.11 protected-access.
    from agent_dispatch import _npubench_sanitized_prefix

    monkeypatch.setenv("A3_HOST", "198.51.100.70")
    monkeypatch.setenv("A3_PASSWORD", "not-a-real-secret")
    monkeypatch.setenv("CANNBOT_PORT_A3_BUILD_SOURCE", "/private/source")
    monkeypatch.setenv("ASCENDC_ENV_PATH", "/private/a3.env")
    prefix = _npubench_sanitized_prefix(["sandbox", "--"])

    assert prefix[:2] == ["sandbox", "--"]
    assert prefix[2] == "/usr/bin/env"
    for name in (
        "A3_HOST",
        "A3_PASSWORD",
        "CANNBOT_PORT_A3_BUILD_SOURCE",
        "ASCENDC_ENV_PATH",
    ):
        assert any(prefix[index:index + 2] == ["-u", name]
                   for index in range(3, len(prefix) - 1))


@pytest.mark.skipif(
    not graybox_sandbox.isolation_available(), reason="strict isolation unavailable"
)
def test_npubench_dispatch_overlays_task_and_state_read_only(monkeypatch, tmp_path):
    """The worker sees the selected task/state only through RO nested binds."""
    monkeypatch.setenv("A3_HOST", "198.51.100.70")
    monkeypatch.setenv("A3_PASSWORD", "not-a-real-secret")
    monkeypatch.setenv("CANNBOT_PORT_A3_BUILD_SOURCE", "/private/source")
    captured, workspace = _drive_dispatch(
        monkeypatch,
        tmp_path,
        state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
        npubench=True,
    )
    prefix = captured["sandbox_prefix"]
    state_path = workspace / ".opgen_state.json"
    reference = json.loads(state_path.read_text())["reference"]
    bundle = workspace / "reference_inputs" / "npubench" / reference["bundle_sha256"]
    joined = " ".join(prefix)

    assert str(bundle) in joined
    assert str(state_path) in joined
    assert "/usr/bin/env" in prefix
    for name in ("A3_HOST", "A3_PASSWORD", "CANNBOT_PORT_A3_BUILD_SOURCE"):
        assert "-u" in prefix
        assert name in prefix
    if graybox_sandbox.isolation_backend() == "bwrap":
        def _bind_index(flag, path):
            return next(
                index for index in range(len(prefix) - 2)
                if prefix[index:index + 3] == [flag, str(path), str(path)]
            )

        workspace_bind = _bind_index("--bind", workspace)
        bundle_bind = _bind_index("--ro-bind", bundle)
        state_bind = _bind_index("--ro-bind", state_path)
        assert workspace_bind < bundle_bind < state_bind


@pytest.mark.skipif(
    not graybox_sandbox.isolation_available(), reason="strict isolation unavailable"
)
def test_npubench_nested_overlays_are_enforced_in_a_real_child(monkeypatch, tmp_path):
    """The selected task/state stay frozen after the actual backend exec.

    Static argv assertions cannot prove nested RO overlays win over the parent
    workspace RW bind.  Exercise the generated bwrap/Seatbelt command itself:
    the child may write a normal workspace result and forge a sibling bundle,
    but it cannot mutate, rename, chmod, or repoint the selected bundle/state
    and it cannot inherit live-A3 environment variables.
    """
    monkeypatch.setenv("A3_HOST", "198.51.100.70")
    monkeypatch.setenv("A3_PASSWORD", "not-a-real-secret")
    monkeypatch.setenv("CANNBOT_PORT_A3_BUILD_SOURCE", "/private/source")
    captured, workspace = _drive_dispatch(
        monkeypatch,
        tmp_path,
        state_json='{"opgen_mode": "port_a3_to_a5", "graybox_sandbox": true}',
        npubench=True,
    )
    state_path = workspace / ".opgen_state.json"
    reference = json.loads(state_path.read_text())["reference"]
    bundle = workspace / "reference_inputs" / "npubench" / reference["bundle_sha256"]
    task = bundle / "3_Add.py"
    # POSIX shell code intentionally prints no inherited environment values.
    probe = f'''set -eu
for name in A3_HOST A3_PASSWORD CANNBOT_PORT_A3_BUILD_SOURCE; do
  if /usr/bin/env | /usr/bin/grep -q "^${{name}}="; then exit 31; fi
done
printf 'normal workspace write\n' > worker_probe.txt
mkdir -p reference_inputs/npubench/forged
if (printf 'x' >> {task!s}) >/dev/null 2>&1; then exit 32; fi
if (chmod u+w {task!s}) >/dev/null 2>&1; then exit 33; fi
if (mv {task!s} {task!s}.moved) >/dev/null 2>&1; then exit 34; fi
if (printf 'x' >> {state_path!s}) >/dev/null 2>&1; then exit 35; fi
if /usr/bin/grep -q forged {state_path!s}; then exit 36; fi
'''
    completed = subprocess.run(
        [*captured["sandbox_prefix"], "/bin/sh", "-c", probe],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0
    assert (workspace / "worker_probe.txt").read_text() == "normal workspace write\n"
    assert (workspace / "reference_inputs" / "npubench" / "forged").is_dir()
    assert task.is_file()
    assert not task.with_suffix(".py.moved").exists()
    assert "forged" not in state_path.read_text()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
