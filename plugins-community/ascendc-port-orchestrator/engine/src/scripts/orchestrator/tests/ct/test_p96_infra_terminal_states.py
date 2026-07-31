# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P96 regression — new orchestrator terminal states:
- infra_transient_retry_exhausted
- infra_baseline_violated

See ANTI_PRESSURE_PROTOCOLS.md §P9 + docs/baseline/environment_baseline.yaml
+ workflows/opgen_state_machine.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest
import yaml

_HERE = Path(__file__).resolve()
_REPO = _reorg_paths.REPO_ROOT
_YAML = _REPO.parent / "workflows/opgen_state_machine.yaml"


@pytest.fixture(scope="module")
def state_machine_yaml():
    with _YAML.open() as f:
        return yaml.safe_load(f)


def _get_state(machine: dict, sid: str) -> dict | None:
    for s in machine.get("phase_o4_states", []):
        if s.get("id") == sid:
            return s
    return None


def test_infra_transient_retry_exhausted_exists(state_machine_yaml):
    """Verify infra_transient_retry_exhausted terminal state defined."""
    s = _get_state(state_machine_yaml, "infra_transient_retry_exhausted")
    assert s is not None, "infra_transient_retry_exhausted state missing"
    assert s.get("agent") is None, "must be terminal (no agent)"
    assert s.get("exit_transitions") == [], "must have no exit transitions"


def test_infra_baseline_violated_exists(state_machine_yaml):
    """Verify infra_baseline_violated terminal state defined."""
    s = _get_state(state_machine_yaml, "infra_baseline_violated")
    assert s is not None, "infra_baseline_violated state missing"
    assert s.get("agent") is None, "must be terminal (no agent)"
    assert s.get("exit_transitions") == [], "must have no exit transitions"


def test_infra_terminals_use_exit_code_11(state_machine_yaml):
    """Both INFRA terminals use exit_code 11 (distinct from 10 await_user
    and 2 orphan-detect).
    """
    for sid in ("infra_transient_retry_exhausted", "infra_baseline_violated"):
        s = _get_state(state_machine_yaml, sid)
        assert s.get("exit_code") == 11, (
            f"{sid} should have exit_code=11, got {s.get('exit_code')}"
        )


def test_infra_terminals_share_counter(state_machine_yaml):
    """Both INFRA terminals share `infra_terminal` iter_counter group for
    aggregate budget tracking.
    """
    for sid in ("infra_transient_retry_exhausted", "infra_baseline_violated"):
        s = _get_state(state_machine_yaml, sid)
        assert s.get("iter_counter") == "infra_terminal", (
            f"{sid} iter_counter should be 'infra_terminal'"
        )


def test_terminal_states_distinct_from_abort_and_done(state_machine_yaml):
    """INFRA terminals are SEPARATE from `abort` (kernel-side) and `done`
    (success). The split matters semantically — INFRA = environment issue
    needing preflight/recover, abort = kernel-side failure.
    """
    abort = _get_state(state_machine_yaml, "abort")
    done = _get_state(state_machine_yaml, "done")
    assert abort is not None and done is not None
    # Distinct ids
    ids = {s.get("id") for s in state_machine_yaml.get("phase_o4_states", [])}
    assert "infra_transient_retry_exhausted" in ids
    assert "infra_baseline_violated" in ids
    assert "abort" in ids
    assert "done" in ids
    # abort has no exit_code (uses default), INFRA states use 11
    assert abort.get("exit_code") is None or abort.get("exit_code") != 11
