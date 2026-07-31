# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for briefs/_common.py + agent_dispatch.py."""
from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import _common as bc  # noqa: E402


# ---------------------------------------------------------------------------
# load_env
# ---------------------------------------------------------------------------
def test_load_env_parses_real_ascendc_env(tmp_path):
    env_text = textwrap.dedent("""\
        # Comment line
        A5_HOST=203.0.113.35
        A5_USER=root
        A5_PASSWORD='Dummy#Test$99'
        A5_CONTAINER=npu_dev3
        CANN_PATH=/data/cann_b103/cann-9.0.0
        SOC_VERSION=Ascend950PR_9579
        BENCHMARK_ROOT=/root/AscendOpGenAgent
        LOCAL_BENCHMARK=/local/path
        LOCAL_PROJECT=/proj
        TARGET=a5
        BUILD_ARCHIVE_ENABLED=1
    """)
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(env_text)

    env = bc.load_env(env_file)
    assert env.target == "a5"
    assert env.host == "203.0.113.35"
    assert env.password == "Dummy#Test$99"  # quotes stripped
    assert env.cann_path == "/data/cann_b103/cann-9.0.0"
    assert env.soc_version == "Ascend950PR_9579"
    assert env.archive_project == "generated_ops"
    assert env.build_archive_enabled is True


def test_load_env_migration_mode_maps_archive(tmp_path):
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("TARGET=a5\nA5_HOST=foo\nOPGEN_MODE=port_a3_to_a5\n")
    env = bc.load_env(env_file)
    assert env.target == "a5"
    assert env.archive_project == "a3_to_a5_port"


def test_load_env_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bc.load_env(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# DEBT-227: target-specific SOC_VERSION / CANN_PATH must WIN over a generic
# one (regression for the customer-repro config bug where a stale generic
# SOC_VERSION silently overrode A3_SOC_VERSION → wrong SoC used for the build).
# ---------------------------------------------------------------------------
def test_a3_soc_version_wins_over_conflicting_generic(tmp_path, capsys):
    """TARGET=a3 with a stale generic SOC_VERSION (A5 value): the target-specific
    A3_SOC_VERSION must win, and a loud WARNING must be emitted (never silent).
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(textwrap.dedent("""\
        TARGET=a3
        A3_HOST=198.51.100.92
        A3_CONTAINER=npu-a3
        A3_SOC_VERSION=Ascend910_9392
        A3_CANN_PATH=/usr/local/Ascend/cann
        SOC_VERSION=Ascend950PR_9579
        CANN_PATH=/data/cann_b103/cann-9.0.0
    """))
    env = bc.load_env(env_file)
    # Target-specific wins (was: generic silently overrode → Ascend950PR_9579).
    assert env.soc_version == "Ascend910_9392"
    assert env.cann_path == "/usr/local/Ascend/cann"
    # Fail-loud on conflict: a WARNING names both keys.
    err = capsys.readouterr().err
    assert "SOC_VERSION" in err and "A3_SOC_VERSION" in err
    assert "CANN_PATH" in err and "A3_CANN_PATH" in err


def test_no_warning_when_generic_matches_target_specific(tmp_path, capsys):
    """No conflict warning when the generic key equals the target-specific one."""
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(textwrap.dedent("""\
        TARGET=a3
        A3_HOST=198.51.100.92
        A3_CONTAINER=npu-a3
        A3_SOC_VERSION=Ascend910_9392
        SOC_VERSION=Ascend910_9392
    """))
    env = bc.load_env(env_file)
    assert env.soc_version == "Ascend910_9392"
    assert "WARNING" not in capsys.readouterr().err


def test_generic_soc_version_still_used_as_fallback(tmp_path, capsys):
    """Backward-compat: when NO target-specific key is set, the generic key is
    still honored (and no warning fires).
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(textwrap.dedent("""\
        TARGET=a3
        A3_HOST=198.51.100.92
        A3_CONTAINER=npu-a3
        SOC_VERSION=Ascend910_9392
        CANN_PATH=/usr/local/Ascend/cann
    """))
    env = bc.load_env(env_file)
    assert env.soc_version == "Ascend910_9392"
    assert env.cann_path == "/usr/local/Ascend/cann"
    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# G7 slug builder
# ---------------------------------------------------------------------------
def test_g7_slug_strips_numeric_prefix():
    assert bc.g7_slug("22_Nonzero", "aog-kernel-worker", 1) == "nonzero-kw-1"
    assert bc.g7_slug("9_TopKTopP", "aog-precision-probe", 3) == "topktopp-pp-3"
    assert bc.g7_slug("14_AdaptiveInstanceNormalization2DBackward",
                       "aog-kernel-worker", 2) == "adaptiveinstancenormalization2dbackward-kw-2"


def test_g7_slug_lowercases():
    assert bc.g7_slug("12_KvRmsnormRopeCache", "aog-kernel-optimizer", 1) == "kvrmsnormropecache-ko-1"


def test_g7_slug_each_agent_code():
    op = "test_op"
    assert bc.g7_slug(op, "aog-kernel-worker", 1).endswith("-kw-1")
    assert bc.g7_slug(op, "aog-precision-probe", 1).endswith("-pp-1")
    assert bc.g7_slug(op, "aog-kernel-optimizer", 1).endswith("-ko-1")
    assert bc.g7_slug(op, "aog-fused-optimizer", 1).endswith("-fo-1")
    assert bc.g7_slug(op, "aog-researcher", 1).endswith("-ar-1")
    assert bc.g7_slug(op, "aog-determinism-analyzer", 1).endswith("-da-1")


def test_g7_slug_unknown_agent_raises():
    with pytest.raises(ValueError):
        bc.g7_slug("x", "general-purpose", 1)


# ---------------------------------------------------------------------------
# env_block / hard_floors_block / kb_manifest_block
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env(tmp_path):
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "A5_HOST=198.51.100.35\nA5_USER=root\nA5_PASSWORD='pw'\n"
        "A5_CONTAINER=npu_dev3\nCANN_PATH=/cann\nSOC_VERSION=Ascend950PR_9579\n"
        "BENCHMARK_ROOT=/bench\nLOCAL_BENCHMARK=/lb\nLOCAL_PROJECT=/lp\nTARGET=a5\n"
    )
    return bc.load_env(env_file)


def test_env_block_includes_lane_and_npu_constraint(fake_env, tmp_path):
    workspace = tmp_path / "22_Nonzero"
    workspace.mkdir()
    block = bc.env_block(fake_env, lane=2, op="22_Nonzero", workspace=workspace)
    assert "LANE: 2 (NPU 2 — A5 has 3 NPUs total IDs 0/1/2)" in block
    assert "TARGET: a5" in block
    assert "PLATFORM_SIMT: true" in block
    assert "Ascend950PR_9579" in block
    assert "ARCHIVE_PROJECT: generated_ops" in block
    line = next(line for line in block.splitlines() if line.startswith("DEPLOY_STAGE_DIR:"))
    assert not line.split(":", 1)[1].strip().startswith("/tmp/")


def test_load_env_and_env_block_surface_a3_stage_roots(tmp_path):
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(textwrap.dedent("""\
        TARGET=a3
        A3_HOST=198.51.100.92
        A3_USER=root
        A3_CONTAINER=npu-a3
        A3_CANN_PATH=/usr/local/Ascend/cann
        A3_SOC_VERSION=Ascend910_9382
        SOC_VERSION=Ascend910_9382
        BENCHMARK_ROOT=/root/AscendOpGenAgent
        A3_HOST_HOME=/home/npu_user
        A3_HOST_BACKUP=/data2/npu_user
        A3_DEPLOY_STAGE_HOST=/home/npu_user
        A3_DEPLOY_STAGE_CONTAINER=/home/npu_user
    """))
    env = bc.load_env(env_file)
    workspace = tmp_path / "codex_e2e_add_a3"
    workspace.mkdir()

    block = bc.env_block(env, lane=0, op="codex_e2e_add_a3", workspace=workspace)

    assert env.a3_deploy_stage_host == "/home/npu_user"
    assert "A3_HOST_HOME: /home/npu_user" in block
    assert "A3_DEPLOY_STAGE_HOST: /home/npu_user" in block
    assert "A3_DEPLOY_STAGE_CONTAINER: /home/npu_user" in block
    line = next(line for line in block.splitlines() if line.startswith("DEPLOY_STAGE_DIR:"))
    assert line.split(":", 1)[1].strip().startswith("op_codex_e2e_add_a3_lane0_")


def test_hard_floors_cold_start_emits_goal(tmp_path):
    workspace = tmp_path / "fresh_op"
    workspace.mkdir()
    block = bc.hard_floors_block(workspace)
    assert "cold start" in block.lower()
    assert "0.6×" in block or "0.6x" in block


def test_hard_floors_with_baseline_reads_verification(tmp_path):
    workspace = tmp_path / "op_with_baseline"
    workspace.mkdir()
    (workspace / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_a": {"status": "PASS", "n_pass": 50, "n_total": 50},
                      "pass_b": {"status": "PASS", "n_pass": 10, "n_total": 10}},
        "performance": {"status": "PASS", "ratio": 1.15},
        "determinism": {"policy_satisfied": True, "n_identical": 50, "n_total": 50},
    }))
    block = bc.hard_floors_block(workspace)
    assert "50/50 PASS" in block
    assert "10/10 PASS" in block
    assert "1.15" in block


def test_kb_manifest_block_uses_taxonomy(tmp_path):
    """When workspace/op_classification.json declares tags + KB paths,
    kb_manifest_block surfaces them in the brief. Mirrors the v3 lookup
    contract (P0aak, 2026-05-07): brief is driven by classification JSON,
    not by hardcoded OP_TAGS dict.
    """
    import json as _json
    workspace = tmp_path / "22_Nonzero"
    workspace.mkdir()
    (workspace / "op_classification.json").write_text(_json.dumps({
        "op_class_tags": ["scatter-gather", "reduction"],
        "kb_recommendations": [
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-110"},
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-67"},
        ],
    }))
    block = bc.kb_manifest_block(
        "22_Nonzero", workspace=workspace, force_legacy_kb=True,
    )
    assert "scatter-gather" in block
    assert "reduction" in block
    assert "OL-110" in block
    assert "OL-67" in block
    # Always-loaded defaults
    assert "KB_INDEX.md" in block
    assert "ascend950pr.md" in block


def test_kb_manifest_untagged_op_says_so(tmp_path):
    block = bc.kb_manifest_block("nonexistent_op_xyz", force_legacy_kb=True)
    assert "UNTAGGED" in block
    # Still has default safe set
    assert "KB_INDEX.md" in block
    assert "OPERATIONAL_KNOWLEDGE.md" in block


def test_schema_contract_block_warns_against_aliases():
    block = bc.schema_contract_block()
    # Must explicitly warn against the DEBT-074 aliases workers keep using
    assert "done" in block
    assert "partial_persist" in block
    assert "from_state" in block
    assert "to_state" in block


def test_safety_block_mentions_3_npu_constraint(fake_env):
    block = bc.safety_block(fake_env)
    assert "3 NPUs" in block or "phantom" in block.lower()


# ---------------------------------------------------------------------------
# kw_brief integration — full brief assembly
# ---------------------------------------------------------------------------
def test_kw_brief_cold_start_has_all_sections(tmp_path, fake_env, monkeypatch):
    """Brief assembly with a workspace classification JSON injects the
    LLM-classified tags into the KB MANIFEST section.
    """
    import json as _json
    workspace = tmp_path / "22_Nonzero"
    workspace.mkdir()
    (workspace / "PROGRESS.md").write_text("")
    (workspace / "op_classification.json").write_text(_json.dumps({
        "op_class_tags": ["scatter-gather", "reduction"],
        "kb_recommendations": [
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-67"},
        ],
    }))
    # This test pins the explicit legacy-manifest escape hatch; OKF remains
    # the production default and is covered by the dedicated OKF tests.
    monkeypatch.setenv("ASCENDC_PORT_OKF", "0")
    scoped_env = replace(
        fake_env,
        opgen_mode="port_a3_to_a5",
        port_a3_source="/fixture/arch22/22_Nonzero",
        archive_project="a3_to_a5_port",
    )

    from briefs.kw_brief import build_worker_brief
    brief = build_worker_brief(
        op="22_Nonzero", workspace=workspace,
        lane=0, spawn_index=1,
        iter_cap_remaining=9, env=scoped_env,
    )
    # G7 slug at top
    assert brief.startswith("nonzero-kw-1")
    # All sections present
    assert "OP: 22_Nonzero" in brief
    assert "PASS A" in brief.upper() or "Pass A" in brief
    assert "KB MANIFEST" in brief
    assert "scatter-gather" in brief  # from classification JSON
    assert "OUTPUT SCHEMA CONTRACT" in brief
    assert "NO torch_npu" in brief
    assert "G1 MARKER" in brief
    assert "iter_cap_remaining = 9" in brief


def test_kw_brief_with_directive_includes_it(tmp_path, fake_env):
    workspace = tmp_path / "fresh"
    workspace.mkdir()
    from briefs.kw_brief import build_worker_brief
    scoped_env = replace(
        fake_env,
        opgen_mode="backward",
        archive_project="backward_ops",
    )

    directive = "Apply Direction 4: emit grad_weight in fp32, cast in pybind."
    brief = build_worker_brief(
        op="14_AdaptiveInstanceNormalization2DBackward", workspace=workspace,
        lane=1, spawn_index=2, iter_cap_remaining=8, env=scoped_env,
        directive_text=directive,
    )
    assert directive in brief
    assert "DIRECTIVE FROM PRIOR AGENT" in brief
    # Cold-start phase A instructions should NOT appear
    assert "Source analysis" not in brief


def test_load_env_honors_monkeypatched_default(tmp_path, monkeypatch):
    """DEBT-101: `load_env()` (no-arg) must resolve DEFAULT_ASCENDC_ENV at
    CALL time so test monkeypatches are effective. Function default `= None`
    + sentinel resolution avoids the def-time binding pitfall.

    Regression for the review-flagged class of failure: any caller doing
    `load_env()` previously read the ORIGINAL DEFAULT_ASCENDC_ENV regardless
    of `monkeypatch.setattr(_common.DEFAULT_ASCENDC_ENV, ...)`. This test
    asserts the monkeypatch IS effective for default-arg callers.
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "TARGET=a5\nA5_HOST=patched-host\nA5_USER=root\n"
        "A5_CONTAINER=patched-c\nA5_CANN_PATH=/x\nA5_SOC_VERSION=Y\n"
    )
    from briefs import _common as _bc
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", env_file)
    env = _bc.load_env()  # no-arg — must use monkeypatched default
    assert env.host == "patched-host", (
        "load_env() default-arg call bypassed the monkeypatched DEFAULT_ASCENDC_ENV"
    )
    assert env.container == "patched-c"


# ---------------------------------------------------------------------------
# resolve_backend_from_env (DEBT-161 Batch-3: the P131 idiom shared by all 9 briefs)
# ---------------------------------------------------------------------------
def test_resolve_backend_from_env():
    class _E:
        def __init__(self, backend=None):
            if backend is not None:
                self.backend = backend
    # The engine supports one kernel-authoring backend.
    assert bc.resolve_backend_from_env("ascendc", _E("ascendc")) == "ascendc"
    with pytest.raises(ValueError, match="only the AscendC backend"):
        bc.resolve_backend_from_env("unsupported", _E("ascendc"))
    # env without a .backend attr → default fallback 'ascendc', no inherit
    assert bc.resolve_backend_from_env("ascendc", _E()) == "ascendc"
