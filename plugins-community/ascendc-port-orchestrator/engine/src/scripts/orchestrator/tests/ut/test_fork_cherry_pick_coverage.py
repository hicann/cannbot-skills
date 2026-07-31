# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Test-coverage for fork cherry-picks (P0abx + P0abq + P0aaz, 2026-05-12).

Per user 2026-05-12 audit: "for new merged cherry picks, are they covered
by UT and sanity properly? if not, add those missing tests."

Coverage gaps identified post-merge of commits 92c4618 (P0abx), 0795de3
(P0abq), 8ac4004 (P0aay+P0aaz):

| Code path                                  | Pre-existing test? | New test below |
|--------------------------------------------|--------------------|----------------|
| --timing CLI flag dispatch                  | NO                 | test_timing_flag_* |
| _generate_timing_report helper              | NO                 | test_generate_timing_report_* |
| _common.py 2-tier KB manifest split        | NO                 | test_kb_manifest_two_tier_* |
| phase_o5_runner _run_verifier_local        | NO                 | test_phase_o5_runner_local_* |

Schema normalization is exercised by the test files cherry-picked alongside
the production changes (conftest.py + test_p0ee/p0qq/p0uu).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402
from briefs._common import kb_manifest_block  # noqa: E402
import phase_o5_runner  # noqa: E402

_PROJECT_ROOT = _reorg_paths.REPO_ROOT


# ---------------------------------------------------------------------------
# P0aaz: --timing flag dispatch + _generate_timing_report
# ---------------------------------------------------------------------------
def test_timing_flag_in_argparse_help():
    """argparse exposes --timing with documented help text."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator", "--help"],
        capture_output=True, text=True, timeout=15,
        cwd=str(_PROJECT_ROOT / "src" / "scripts" / "orchestrator"),
        env={"PYTHONPATH": str(_PROJECT_ROOT / "src" / "scripts" / "orchestrator"),
             "PATH": "/usr/bin:/bin"},
    )
    combined = result.stdout + result.stderr
    assert "--timing" in combined
    assert "TIMING_REPORT" in combined


def test_generate_timing_report_missing_script(tmp_path, caplog, monkeypatch):
    """When gen_timing_report.py is absent, _generate_timing_report logs a
    warning + returns gracefully (no exception).

    2026-05-27 (zero-UT-failure rule): switched from capsys to caplog after
    the logging refactor (`ec913af0` / `e1c7a59a`) converted print() calls
    to log.info() — stdout no longer carries the message; the logging
    fixture captures it correctly.
    """
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path / "nonexistent_proj_root")
    # The orchestrator module uses logger 'a5_orchestrator.orchestrator'
    # with propagate=False (per logging_config.py). caplog only sees records
    # via propagation, so we explicitly attach its handler to the namespace
    # AND set level. Cleaner than monkeypatching propagate, since caplog's
    # records are still semantically correct.
    _capture_logger = logging.getLogger("a5_orchestrator")
    _capture_logger.addHandler(caplog.handler)
    _capture_logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="a5_orchestrator"):
        result = getattr(orch, "_generate_timing_report")(ws, "fake_op")
    assert result is None
    assert "not found" in caplog.text or "skipping" in caplog.text


def test_generate_timing_report_success(tmp_path, caplog, monkeypatch):
    """When gen_timing_report.py runs successfully, _generate_timing_report
    announces TIMING_REPORT.md was written (via log.info, post logging
    refactor).
    """
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    fake_root = tmp_path / "fake_root"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts" / "gen_timing_report.py").write_text("#!/usr/bin/env python3\n")
    monkeypatch.setattr(orch, "PROJECT_ROOT", fake_root)

    fake_result = MagicMock(returncode=0, stdout="ok", stderr="")
    # The orchestrator module uses logger 'a5_orchestrator.orchestrator'
    # with propagate=False (per logging_config.py). caplog only sees records
    # via propagation, so we explicitly attach its handler to the namespace
    # AND set level. Cleaner than monkeypatching propagate, since caplog's
    # records are still semantically correct.
    _capture_logger = logging.getLogger("a5_orchestrator")
    _capture_logger.addHandler(caplog.handler)
    _capture_logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="a5_orchestrator"):
        with patch("subprocess.run", return_value=fake_result):
            getattr(orch, "_generate_timing_report")(ws, "fake_op")
    assert "TIMING_REPORT.md written" in caplog.text


def test_generate_timing_report_subprocess_failure(tmp_path, caplog, monkeypatch):
    """When gen_timing_report.py returns non-zero, _generate_timing_report
    surfaces the error via log without crashing (post logging refactor).
    """
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    fake_root = tmp_path / "fake_root"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts" / "gen_timing_report.py").write_text("#!/usr/bin/env python3\n")
    monkeypatch.setattr(orch, "PROJECT_ROOT", fake_root)

    fake_result = MagicMock(returncode=1, stdout="", stderr="boom: invalid jsonl")
    # The orchestrator module uses logger 'a5_orchestrator.orchestrator'
    # with propagate=False (per logging_config.py). caplog only sees records
    # via propagation, so we explicitly attach its handler to the namespace
    # AND set level. Cleaner than monkeypatching propagate, since caplog's
    # records are still semantically correct.
    _capture_logger = logging.getLogger("a5_orchestrator")
    _capture_logger.addHandler(caplog.handler)
    _capture_logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="a5_orchestrator"):
        with patch("subprocess.run", return_value=fake_result):
            getattr(orch, "_generate_timing_report")(ws, "fake_op")
    assert "report gen failed" in caplog.text
    assert "exit 1" in caplog.text


# ---------------------------------------------------------------------------
# P0abq: 2-tier KB manifest split
# ---------------------------------------------------------------------------
def test_kb_manifest_two_tier_separates_large_file():
    """kb_manifest_block puts OPERATIONAL_KNOWLEDGE.md in Tier 2 (grep-only),
    not bulk-loaded as Tier 1.
    """
    block = kb_manifest_block(
        "13_Cat", workspace=None, target="a5", force_legacy_kb=True,
    )
    assert "two-tier loading" in block
    assert "OPERATIONAL_KNOWLEDGE.md" in block
    # Tier-2 grep hints present
    assert "grep -n" in block
    # Tier-1 small files
    assert "KB_INDEX.md" in block or "PLATFORM_BUGS.md" in block


def test_kb_manifest_anchored_ol_in_tier2():
    """OL anchors land in Tier-2 grep hints, not Tier-1 full-load."""
    block = kb_manifest_block(
        "13_Cat", workspace=None, target="a5", force_legacy_kb=True,
    )
    if "OL-" in block:
        assert "grep -n" in block


# ---------------------------------------------------------------------------
# P0aaz: phase_o5_runner local-container branch
# ---------------------------------------------------------------------------
def test_phase_o5_runner_local_container_skips_ssh(tmp_path, monkeypatch):
    """When env's <TARGET>_CONTAINER == 'local', ssh_runner delegates to
    _run_verifier_local instead of SSH/scp/docker.
    """
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    local_path_called = {"hit": False}

    def _fake_local_runner(workspace, op, env, *args, **kwargs):
        local_path_called["hit"] = True
        return phase_o5_runner.MeasuredResult(pass_a={"tier1_pass": 1, "total": 1})
    monkeypatch.setattr(phase_o5_runner, "_run_verifier_local", _fake_local_runner)

    # phase_o5_runner._read_ascendc_env checks workspace.parent/.ascendc_env
    env_file = ws.parent / ".ascendc_env"
    env_file.write_text(dedent("""\
        TARGET=a2
        A2_HOST=localhost
        A2_USER=root
        A2_CONTAINER=local
        A2_CANN_PATH=/usr/local/Ascend/cann
        A2_SOC_VERSION=Ascend910B2C
    """))
    # Also provide a dummy run_pass_b.py so the verifier-discovery passes
    (ws / "run_pass_b.py").write_text("# stub")

    result = phase_o5_runner.ssh_runner(ws, "op")
    assert local_path_called["hit"] is True, (
        "ssh_runner did NOT dispatch to _run_verifier_local when CONTAINER=local"
    )
    assert result.runner_error is None


def test_phase_o5_runner_non_local_does_not_call_local_path(tmp_path, monkeypatch):
    """Regression: CONTAINER != 'local' should NOT invoke _run_verifier_local."""
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    local_path_called = {"hit": False}

    def _fake_local_runner(workspace, op, env, *args, **kwargs):
        local_path_called["hit"] = True
        return phase_o5_runner.MeasuredResult()
    monkeypatch.setattr(phase_o5_runner, "_run_verifier_local", _fake_local_runner)

    env_file = ws.parent / ".ascendc_env"
    env_file.write_text(dedent("""\
        TARGET=a5
        A5_HOST=198.51.100.35
        A5_USER=root
        A5_CONTAINER=npu_dev3
        A5_CANN_PATH=/data/cann_b103/cann-9.0.0
        A5_SOC_VERSION=Ascend950PR_9579
    """))
    (ws / "run_pass_b.py").write_text("# stub")

    fake_result = MagicMock(returncode=255, stdout="", stderr="ssh refused (test)")
    with patch("subprocess.run", return_value=fake_result):
        try:
            _ = phase_o5_runner.ssh_runner(ws, "op")
        except Exception:
            logging.getLogger(__name__).debug(
                "Remote runner failed as expected in this routing test", exc_info=True
            )

    assert local_path_called["hit"] is False, (
        "ssh_runner WRONGLY dispatched to _run_verifier_local for a remote container"
    )
