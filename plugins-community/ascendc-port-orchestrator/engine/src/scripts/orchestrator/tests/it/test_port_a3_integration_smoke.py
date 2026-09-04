# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""W12+W13 (2026-05-12, ROADMAP §1.5) — taxonomy tag + integration smoke test.

W12 — `a3_to_a5_port` op-class tag in `TAG_KB_SECTIONS`:
- Tag registered with 4 KB section references (W8, W9 OL-131, W10 P-P90, W11)
- Tag → KB-sections lookup returns all 4 entries
- Cross-check: every referenced KB path resolves to an existing file on disk

W13 — end-to-end integration smoke:
- `python -m orchestrator --port-a3 <ops-nn-dir> --plan` exits 0
- op_classification.json is written with a3_to_a5_port tag
- Plan output mentions all the expected phase wiring
- Smoke run on the REAL ctc_loss_v3 ops-nn dir (when available locally)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402
from briefs import op_taxonomy  # noqa: E402

_PROJECT_ROOT = _reorg_paths.REPO_ROOT
_REFERENCES_ROOT = _PROJECT_ROOT.parent / "kb"


# ---------------------------------------------------------------------------
# W12: taxonomy tag registration
# ---------------------------------------------------------------------------
def test_a3_to_a5_port_tag_registered():
    """W12: tag 'a3_to_a5_port' exists in TAG_KB_SECTIONS."""
    assert "a3_to_a5_port" in op_taxonomy.TAG_KB_SECTIONS, (
        f"a3_to_a5_port tag missing; present tags: "
        f"{sorted(op_taxonomy.TAG_KB_SECTIONS.keys())}"
    )


def test_a3_to_a5_port_tag_references_all_4_kb_entries():
    """W12: tag's KB sections list includes all 4 W8-W11 entries."""
    sections = op_taxonomy.TAG_KB_SECTIONS["a3_to_a5_port"]
    assert len(sections) == 4, (
        f"a3_to_a5_port should reference 4 KB entries (W8/W9/W10/W11), "
        f"got {len(sections)}: {sections}"
    )
    # W8: artifact layout
    assert any("ops_nn_layout/ops_nn_a5_artifact_layout.md" in s for s in sections)
    # W9: OL-131 cross-op router
    assert any("OL-131" in s for s in sections)
    # W10: P-P90 platform compat
    assert any("P-P90" in s for s in sections)
    # W11: ascend950pr hardware reference
    assert any("ascend950pr.md" in s for s in sections)


def test_a3_to_a5_port_kb_paths_resolve_to_files():
    """W12 disk-cross-check: every KB path referenced (sans #anchor) exists.

    P88 (2026-05-15) KB reorg: legacy bare names resolve to new layout
    via op_taxonomy.resolve_legacy_kb_path.
    """
    sections = op_taxonomy.TAG_KB_SECTIONS["a3_to_a5_port"]
    missing = []
    for s in sections:
        path = s.split("#", 1)[0]  # strip #anchor
        resolved = op_taxonomy.resolve_legacy_kb_path(path)
        if not (_REFERENCES_ROOT / resolved).is_file():
            missing.append(path)
    assert not missing, (
        f"a3_to_a5_port references missing KB files: {missing} "
        f"(under {_REFERENCES_ROOT})"
    )


# ---------------------------------------------------------------------------
# W13: integration smoke — in-process
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_ctc_loss_v3_dir(tmp_path):
    """Synthesize a ctc_loss_v3-shaped ops-nn op dir."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    (op_dir / "op_host").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "ctc_loss_v3.cpp").write_text(
        "#include \"kernel_operator.h\"\n"
        "extern \"C\" __global__ __aicore__ void ctc_loss_v3() {}\n"
    )
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "examples" / "test_aclnn_ctc_loss_v3.cpp").write_text("// fake")
    (op_dir / "op_host" / "CMakeLists.txt").write_text(
        "add_op_kernel(ctc_loss_v3 DEPENDENCIES ctc_loss_v2 SOURCES x.cpp)\n"
    )
    return op_dir


@pytest.fixture
def fake_env_with_a3(tmp_path, monkeypatch):
    """Patch DEFAULT_ASCENDC_ENV with A3 config."""
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(dedent("""\
        TARGET=a5
        A5_HOST=test-host
        A5_USER=root
        A5_CONTAINER=test-c
        A5_CANN_PATH=/opt/cann
        A5_SOC_VERSION=Ascend950PR_9579
        LOCAL_PROJECT=/tmp/proj
        A3_HOST=198.51.100.70
        A3_USER=root
        A3_CONTAINER=npu-a3
        A3_CANN_PATH=/home/npu_user/cann/cann-9.0.0
        A3_SOC_VERSION=Ascend910_9382
    """))
    from briefs import _common
    monkeypatch.setattr(_common, "DEFAULT_ASCENDC_ENV", env_file)
    return env_file


def test_port_a3_writes_op_classification_with_tag(
    synthetic_ctc_loss_v3_dir, fake_env_with_a3, tmp_path, monkeypatch, capsys
):
    """W12 ↔ W13: --port-a3 path writes op_classification.json with the tag."""
    # Redirect workspace root to tmp
    monkeypatch.setattr(orch, "WORKSPACE_ROOT", tmp_path / "workspace")
    rc = orch._cmd_port_a3(
        port_a3_dir=synthetic_ctc_loss_v3_dir,
        lane=0, plan_only=True, cold_start=False, cap_bumps={},
    )
    assert rc == 0
    ws = tmp_path / "workspace" / "ctc_loss_v3"
    cls_file = ws / "op_classification.json"
    assert cls_file.is_file(), "op_classification.json not written"
    payload = json.loads(cls_file.read_text())
    assert payload["op"] == "ctc_loss_v3"
    assert payload["op_class_tags"] == ["a3_to_a5_port"]
    assert payload["source"] == "cli_flag_port_a3"


def test_port_a3_plan_full_phase_shape(
    synthetic_ctc_loss_v3_dir, fake_env_with_a3, tmp_path, monkeypatch, capsys
):
    """W13: --port-a3 --plan output covers all 6 phase mentions (O0-O6)."""
    monkeypatch.setattr(orch, "WORKSPACE_ROOT", tmp_path / "workspace")
    rc = orch._cmd_port_a3(
        port_a3_dir=synthetic_ctc_loss_v3_dir,
        lane=1, plan_only=True, cold_start=False, cap_bumps={},
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Plan covers phases O0..O6
    for phase_marker in (
        "O0",
        "O1",
        "O2 source sync",
        "O2.5",
        "O3",
        "O4 kw spawn",
        "O5 verify",
        "O6 archive",
    ):
        assert phase_marker in out, f"plan missing {phase_marker!r} marker"
    # Plan mentions the source dir + op name
    assert "ctc_loss_v3" in out
    assert str(synthetic_ctc_loss_v3_dir) in out
    # Plan mentions correct lane
    assert "NPU 1" in out


# ---------------------------------------------------------------------------
# W13: integration smoke — subprocess (true end-to-end)
# ---------------------------------------------------------------------------
def test_orchestrator_port_a3_plan_subprocess_exits_zero(
    synthetic_ctc_loss_v3_dir, tmp_path, monkeypatch
):
    """Exercise the W13 plan-only subprocess end to end.

    Invoking `python -m orchestrator --port-a3 <dir> --plan` must exit 0 and
    emit the plan structure.

    This is the real CLI dispatch path, not in-process. Validates argparse
    + main() routing + _cmd_port_a3 invocation in one shot.
    """
    # Build a custom .ascendc_env in tmp + workspace dir
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(dedent("""\
        TARGET=a5
        A5_HOST=test
        A5_CONTAINER=test
        A5_CANN_PATH=/opt/cann
        A5_SOC_VERSION=Ascend950PR_9579
        LOCAL_PROJECT=/tmp/proj
        A3_HOST=198.51.100.70
        A3_CONTAINER=npu-a3
        A3_CANN_PATH=/home/x/cann
        A3_SOC_VERSION=Ascend910_9382
    """))
    # DEBT-101 follow-up: subprocess inherits no test fixtures (no monkeypatch
    # across fork). Use `ASCENDC_ENV_PATH` env var to point load_env at our
    # tmp .ascendc_env. Removes dependency on the real workspace env's
    # contents (previously this skipped on any agent whose workspace lacked
    # A3_HOST/A3_CONTAINER, making the test silently fail rather than skip;
    # now the override flows the test's own env in).

    result = subprocess.run(
        [sys.executable, "-m", "orchestrator",
         "--port-a3-ops", str(synthetic_ctc_loss_v3_dir), "--plan"],
        capture_output=True, text=True, timeout=30,
        cwd=str(_PROJECT_ROOT / "src" / "scripts" / "orchestrator"),
        env={
            "PYTHONPATH": str(_PROJECT_ROOT / "src" / "scripts" / "orchestrator"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "ASCENDC_ENV_PATH": str(env_file),
        },
    )
    # Tolerate non-zero only if it'\''s the validation failure (e.g. project-
    # specific workspace dir state); but plan-only should be exit 0 for a
    # clean op dir + a clean A3 env file.
    assert result.returncode == 0, (
        f"orchestrator --port-a3 --plan returned {result.returncode}\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    combined = result.stdout + result.stderr
    assert "arch22→arch35" in combined or "ctc_loss_v3" in combined
    assert "port_a3_to_a5" in combined
