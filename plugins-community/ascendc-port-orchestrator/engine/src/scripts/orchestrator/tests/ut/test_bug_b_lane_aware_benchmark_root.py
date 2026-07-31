# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Bug B (2026-05-23): _lane_aware_benchmark_root regression tests.

Symptom: lightning_indexer_grad lane 3 worker built `.so` files at
`/home/npu_user/workspace/AscendOpGenAgent_lane3/current_task/kernel/build/`
(per deploy_to_npu_lane.sh's lane 3 export), but phase_o5_runner read
from `/root/AscendOpGenAgent/current_task/kernel/build/` (the lane 0
default from .ascendc_env). O5 saw stale .so from prior lane 0 op
(swi_glu), reported precision FAIL on what was actually LIG.

Fix: phase_o5_runner uses `_lane_aware_benchmark_root` which returns:
  - lane 0  : /home/npu_user/workspace/AscendOpGenAgent_lane0 (DEBT-159 — lane 0
              is now isolated too; the shared /root/AscendOpGenAgent default no
              longer applies to a lane-managed orchestrator run)
  - lane N+ : /home/npu_user/workspace/AscendOpGenAgent_lane{N}
  - a CALLER-EXPLICIT non-shared BENCHMARK_ROOT is honored verbatim (custom layouts)
matching exactly what deploy_to_npu_lane.sh / deploy_to_npu.sh resolve.

DEBT-159 (2026-06-16): lane 0 used to resolve to the SHARED /root/AscendOpGenAgent,
so a lane-0 run collided with any other lane-0 run (and the bare-deploy default)
on `current_task`. The orchestrator ALWAYS passes an explicit lane, so it is a
lane-managed context — lane 0 gets its own `AscendOpGenAgent_lane0` root. An
env-file `BENCHMARK_ROOT=/root/AscendOpGenAgent` (the common .ascendc_env value)
is the shared default and must NOT defeat isolation (we make the step lane-aware,
not mutate the env). A non-shared custom BENCHMARK_ROOT still wins verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from phase_o5_runner import _lane_aware_benchmark_root  # noqa: E402


def test_lane_0_shared_env_root_does_not_defeat_isolation() -> None:
    """DEBT-159: the SHARED env-file default /root/AscendOpGenAgent must NOT be
    honored for a lane-managed orchestrator run — lane 0 isolates to its own
    AscendOpGenAgent_lane0 root (this was the collision bug).
    """
    env = {"BENCHMARK_ROOT": "/root/AscendOpGenAgent"}
    assert _lane_aware_benchmark_root(env, 0) == "/home/npu_user/workspace/AscendOpGenAgent_lane0"


def test_lane_0_default_when_env_missing() -> None:
    """DEBT-159: lane 0 + no BENCHMARK_ROOT in env → isolated lane-0 root."""
    env = {}
    assert _lane_aware_benchmark_root(env, 0) == "/home/npu_user/workspace/AscendOpGenAgent_lane0"


def test_lane_0_honors_custom_benchmark_root() -> None:
    """Lane 0 honors a CALLER-EXPLICIT, non-shared BENCHMARK_ROOT verbatim
    (e.g. for non-standard container layouts). Only the shared /root default is
    overridden by lane isolation.
    """
    env = {"BENCHMARK_ROOT": "/custom/path/AscendOpGenAgent"}
    assert _lane_aware_benchmark_root(env, 0) == "/custom/path/AscendOpGenAgent"


def test_lane_1_overrides_env_with_lane_path() -> None:
    """Lane 1: ignores env's BENCHMARK_ROOT (which is lane-0-configured) and
    uses the lane-1 host path. This is the load-bearing fix — previously
    lane N read from lane 0's container path.
    """
    env = {"BENCHMARK_ROOT": "/root/AscendOpGenAgent"}
    assert _lane_aware_benchmark_root(env, 1) == "/home/npu_user/workspace/AscendOpGenAgent_lane1"


def test_lane_3_matches_lightning_indexer_grad_symptom() -> None:
    """Lane 3 path must match what deploy_to_npu_lane.sh lane 3 exports — same
    symptom case that motivated this fix (LIG worker built at lane 3 path,
    O5 read from lane 0).
    """
    env = {"BENCHMARK_ROOT": "/root/AscendOpGenAgent"}
    assert _lane_aware_benchmark_root(env, 3) == "/home/npu_user/workspace/AscendOpGenAgent_lane3"


def test_lane_2_path() -> None:
    env = {"BENCHMARK_ROOT": "/root/AscendOpGenAgent"}
    assert _lane_aware_benchmark_root(env, 2) == "/home/npu_user/workspace/AscendOpGenAgent_lane2"


def test_lane_n_ignores_custom_env_root() -> None:
    """For lane N+, env's BENCHMARK_ROOT is NOT honored (it's lane-0-scoped).
    Lane N always uses the canonical lane-N path. If a user wants a custom
    lane-N path they need to update deploy_to_npu_lane.sh + this helper
    together (not a per-op env override).
    """
    env = {"BENCHMARK_ROOT": "/some/other/path"}
    assert _lane_aware_benchmark_root(env, 2) == "/home/npu_user/workspace/AscendOpGenAgent_lane2"


def test_lane_high_number() -> None:
    """Sanity for high lane number (e.g. 7-NPU host)."""
    env = {}
    assert _lane_aware_benchmark_root(env, 7) == "/home/npu_user/workspace/AscendOpGenAgent_lane7"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
