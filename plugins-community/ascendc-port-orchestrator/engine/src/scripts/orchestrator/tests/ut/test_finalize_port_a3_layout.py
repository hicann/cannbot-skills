# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""W7 (2026-05-12, ROADMAP §1.5) — finalize ops-nn-mirror layout tests.

DEBT-094 phase 2 (2026-05-15): mode-detection + path-mapping migrated
from finalize_pipeline._is_port_a3_mode + _resolve_port_a3_archive_target
into the port_a3 plugin. Tests now hit the plugin API directly.

Validates:
- port_a3 plugin detect() reads opgen_mode from .opgen_state.json
- port_a3 plugin.resolve_archive_target maps workspace paths to archive paths
- finalize_op + plugin layout produces ops-nn-mirror archive when port mode active
- finalize_op uses output/<plugin.archive_project_subdir>/src/kernels/ as archive root
- End-to-end: synthetic workspace with port-mode artifacts produces
  the expected ops-nn-mirror archive structure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402
import finalize_dispatch as fd  # noqa: E402

# DEBT-201 (2026-07-06): finalize_op moved to finalize_dispatch.py and reads
# _PROJECT_ROOT by bare name there; patch it on finalize_dispatch so the
# default-archive-root path resolves under tmp, not the real repo output dir.
from plugins.port_a3 import PortA3Plugin  # noqa: E402

_PORT_A3 = PortA3Plugin()


# ---------------------------------------------------------------------------
# port_a3 plugin detect() — replaces legacy _is_port_a3_mode
# ---------------------------------------------------------------------------
def test_port_a3_detect_true(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "ctc_loss_v3", "opgen_mode": "port_a3_to_a5"})
    )
    assert _PORT_A3.detect(tmp_path) is True


def test_port_a3_detect_backward_false(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "13_Cat", "opgen_mode": "backward"})
    )
    assert _PORT_A3.detect(tmp_path) is False


def test_port_a3_detect_missing_state_layout_fallback(tmp_path):
    """Missing state file → layout heuristic. Empty workspace = False."""
    assert _PORT_A3.detect(tmp_path) is False


def test_port_a3_detect_malformed_state_layout_fallback(tmp_path):
    (tmp_path / ".opgen_state.json").write_text("garbage {")
    assert _PORT_A3.detect(tmp_path) is False


def test_port_a3_detect_no_opgen_mode_field(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(
        json.dumps({"op": "13_Cat"})
    )
    assert _PORT_A3.detect(tmp_path) is False


# ---------------------------------------------------------------------------
# port_a3 plugin.resolve_archive_target — replaces legacy _resolve_port_a3_archive_target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src,expected", [
    ("kernel/arch35/ctc_loss_v3.h", "op_kernel/arch35/ctc_loss_v3.h"),
    ("kernel/ctc_loss_v3_apt.cpp", "op_kernel/ctc_loss_v3_apt.cpp"),
    ("kernel/arch35/ctc_loss_v3_helper.h", "op_kernel/arch35/ctc_loss_v3_helper.h"),
    ("op_host/config/ascend950/ctc_loss_v3_binary.json", "op_host/config/ascend950/ctc_loss_v3_binary.json"),
    ("op_host/ctc_loss_v3_def.cpp.patch", "op_host/ctc_loss_v3_def.cpp.patch"),
    ("ctc_loss_v3_a5_migration_plan.md", "docs/ctc_loss_v3_a5_migration_plan.md"),
    ("verification.json", "verification.json"),
    ("PROGRESS.md", "PROGRESS.md"),
    ("a3_reference_runnable.json", "a3_reference_runnable.json"),
    ("peer_router.patch", "peer_router.patch"),
    ("analysis.md", "analysis.md"),
    ("knowledge_update.md", "knowledge_update.md"),
])
def test_port_a3_resolve_archive_target(src, expected):
    assert _PORT_A3.resolve_archive_target(src, "ctc_loss_v3") == expected


# ---------------------------------------------------------------------------
# End-to-end finalize_op in port mode
# ---------------------------------------------------------------------------
def make_port_mode_workspace(tmp_path: Path) -> Path:
    """Synthesize a ctc_loss_v3 workspace in port_a3_to_a5 mode."""
    ws = tmp_path / "workspace" / "ctc_loss_v3"
    ws.mkdir(parents=True)

    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "ctc_loss_v3", "opgen_mode": "port_a3_to_a5",
        "target": "a5", "lane": 0, "schema_version": 1,
    }))
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
        "performance": {"status": "PASS", "ratio": 1.05},
    }))
    (ws / "PROGRESS.md").write_text("# ctc_loss_v3 progress\n")
    (ws / "a3_reference_runnable.json").write_text(json.dumps({
        "verdict": "READY", "peer_op_dependencies": ["ctc_loss_v2"],
    }))

    (ws / "kernel" / "arch35").mkdir(parents=True)
    (ws / "kernel" / "arch35" / "ctc_loss_v3.h").write_text("// A5 arch35 kernel")
    (ws / "kernel" / "ctc_loss_v3_apt.cpp").write_text("// A5 entry point")

    (ws / "op_host" / "config" / "ascend950").mkdir(parents=True)
    (ws / "op_host" / "config" / "ascend950" / "ctc_loss_v3_binary.json").write_text(
        '{"op_type": "CTCLossV3"}'
    )
    (ws / "op_host" / "config" / "ascend950" / "ctc_loss_v3_simplified_key.ini").write_text(
        "[CTCLossV3]\ndefault=0\n"
    )
    (ws / "op_host" / "ctc_loss_v3_def.cpp.patch").write_text("--- a/def.cpp\n+++ b/def.cpp\n")

    (ws / "peer_router.patch").write_text(
        "--- a/loss/ctc_loss_v2/op_api/ctc_loss_v2.cpp\n"
        "+++ b/loss/ctc_loss_v2/op_api/ctc_loss_v2.cpp\n"
    )

    (ws / "ctc_loss_v3_a5_migration_plan.md").write_text("# Migration Plan\n")
    (ws / "analysis.md").write_text("# Analysis")
    (ws / "knowledge_update.md").write_text("# KB Update " + "x" * 200)

    return ws


def test_finalize_op_port_mode_produces_ops_nn_mirror(tmp_path, monkeypatch):
    """End-to-end: workspace with port-mode marker produces ops-nn-mirror archive."""
    ws = make_port_mode_workspace(tmp_path)
    archive_root = tmp_path / "output" / "a3_to_a5_port"

    rep = fp.finalize_op("ctc_loss_v3", ws, archive_root=archive_root)
    assert rep.archive_dir is not None
    assert not rep.errors, f"unexpected errors: {rep.errors}"

    archive = rep.archive_dir
    assert (archive / "op_kernel" / "arch35" / "ctc_loss_v3.h").is_file()
    assert (archive / "op_kernel" / "ctc_loss_v3_apt.cpp").is_file()
    assert (archive / "op_host" / "config" / "ascend950" / "ctc_loss_v3_binary.json").is_file()
    assert (archive / "op_host" / "config" / "ascend950" / "ctc_loss_v3_simplified_key.ini").is_file()
    assert (archive / "op_host" / "ctc_loss_v3_def.cpp.patch").is_file()
    assert (archive / "peer_router.patch").is_file()
    assert (archive / "verification.json").is_file()
    assert (archive / "PROGRESS.md").is_file()
    # 2026-05-16: a3_reference_runnable.json + knowledge_update.md are
    # harness-internal → archive/.harness/. analysis.md stays at root
    # (customer-readable kernel design doc).
    assert (archive / ".harness" / "a3_reference_runnable.json").is_file()
    assert (archive / "analysis.md").is_file()
    assert (archive / ".harness" / "knowledge_update.md").is_file()
    assert (archive / "docs" / "ctc_loss_v3_a5_migration_plan.md").is_file()
    assert not (archive / "kernel").exists()


def test_finalize_op_port_mode_uses_default_a3_to_a5_root(tmp_path, monkeypatch):
    """Port mode + no archive_root override → uses plugin's archive_project_subdir."""
    ws = make_port_mode_workspace(tmp_path)
    monkeypatch.setattr(fd, "_PROJECT_ROOT", tmp_path)
    rep = fp.finalize_op("ctc_loss_v3", ws)
    assert rep.archive_dir is not None
    assert "a3_to_a5_port" in str(rep.archive_dir)
    assert "src/kernels" in str(rep.archive_dir)


def test_finalize_op_port_mode_skips_dotfiles(tmp_path, monkeypatch):
    """Dotfiles (.opgen_state.json, .agent_died_at_*, etc.) NOT promoted."""
    ws = make_port_mode_workspace(tmp_path)
    (ws / ".agent_died_at_finalize").write_text("crash log")
    (ws / "kernel" / ".tmp_build_marker").write_text("ignore me")
    archive_root = tmp_path / "output" / "a3_to_a5_port"
    rep = fp.finalize_op("ctc_loss_v3", ws, archive_root=archive_root)
    archive = rep.archive_dir
    assert not (archive / ".opgen_state.json").exists()
    assert not (archive / ".agent_died_at_finalize").exists()
    assert not (archive / "op_kernel" / ".tmp_build_marker").exists()
