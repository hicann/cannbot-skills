# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for the orchestrator `--precision-standard` CLI flag
(owner-directed run-control, controllable-harness #4, 2026-07-21).

The flag threads to the port_a3 grader SUBPROCESS as a `--precision-standard`
arg (not via in-process load_and_classify). These tests lock in:
  (a) the resolver contract the transport relies on: cli_value short-circuits
      BEFORE the env↔.ascendc_env conflict check, so the flag wins WITHOUT
      raising even when env would otherwise conflict;
  (b) the pure argv-suffix helper used at both grader argv sites in
      phase_o5_verify._run_canonical_pass_a (hermetic — the real function does
      SSH/docker, so the suffix decision is factored out for unit testing);
  (c) the flag is registered on the orchestrator CLI (`orch --help`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
# orchestrator dir (…/orchestrator) for phase_o5_verify + orchestrator_cli
sys.path.insert(0, str(_HERE.parents[2]))
# src/scripts for precision_eval_port_a3_two_tier
sys.path.insert(0, str(_HERE.parents[3]))

import precision_eval_port_a3_two_tier as pa3  # noqa: E402
# Import phase_o5_runner FIRST to satisfy the phase_o5_runner<->phase_o5_verify
# import ordering (runner re-exports from verify at module tail); then verify is
# already fully initialized in sys.modules. Mirrors the sibling ut tests.
import phase_o5_runner  # noqa: E402,F401
import phase_o5_verify as o5  # noqa: E402


# ---------------------------------------------------------------------------
# (a) resolver: cli_value bypasses the env↔.ascendc_env conflict → flag wins
# ---------------------------------------------------------------------------
def test_cli_value_bypasses_env_conflict(tmp_path):
    """A conflicting PRECISION_STANDARD env must NOT raise when cli_value is
    supplied — cli short-circuits before the conflict check. This is exactly
    why the orch flag uses a distinct transport → `--precision-standard` arg.
    """
    (tmp_path / ".ascendc_env").write_text("PRECISION_STANDARD=ecosystem\n")
    std, src = pa3.resolve_precision_standard(
        tmp_path, cli_value="commercial", env={"PRECISION_STANDARD": "ecosystem"})
    assert (std, src) == ("commercial", "cli")


# ---------------------------------------------------------------------------
# (b) pure argv-suffix helper: env × is_pa3 matrix
# ---------------------------------------------------------------------------
def test_suffix_appended_for_pa3_when_env_set(monkeypatch):
    monkeypatch.setenv("AOG_PRECISION_STANDARD_CLI", "commercial")
    suffix = getattr(o5, "_precision_standard_cli_suffix")(is_pa3=True)
    assert "--precision-standard commercial" in suffix
    # trailing space so it drops cleanly before the `>/dev/null` tail
    assert suffix.endswith(" ")


def test_suffix_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("AOG_PRECISION_STANDARD_CLI", raising=False)
    assert getattr(o5, "_precision_standard_cli_suffix")(is_pa3=True) == ""


def test_suffix_empty_for_non_pa3_even_when_env_set(monkeypatch):
    """A non-migration verifier does not accept the flag, so never append it."""
    monkeypatch.setenv("AOG_PRECISION_STANDARD_CLI", "commercial")
    assert getattr(o5, "_precision_standard_cli_suffix")(is_pa3=False) == ""


# ---------------------------------------------------------------------------
# (c) the flag is registered on the orchestrator CLI (`orch --help`)
# ---------------------------------------------------------------------------
def test_help_lists_precision_standard(monkeypatch, capsys):
    import orchestrator_cli as cli

    class _NoopOrch:
        def _refuse_if_detached(self):
            return None

    monkeypatch.setattr(cli, "_orch", lambda: _NoopOrch())
    monkeypatch.setattr(sys, "argv", ["orch", "--help"])
    with pytest.raises(BaseException) as exc:  # argparse --help exits after printing
        cli.main()
    assert type(exc.value).__name__ == "SystemExit"
    assert "--precision-standard" in capsys.readouterr().out
