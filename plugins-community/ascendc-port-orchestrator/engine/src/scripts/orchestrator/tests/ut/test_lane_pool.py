# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for lane_pool.py (Track C #2)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import lane_pool as lp  # noqa: E402


# ---------------------------------------------------------------------------
# Sample npu-smi outputs
# ---------------------------------------------------------------------------
NPU_SMI_ALL_IDLE = """\
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1                                 Version: 25.7.rc1                              |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | OK            | 215.0       58                0     / 0             |
|        |                  | 0000:61:00.0  | 0           0    / 0          5407  / 131072        |
+========+==================+===============+=====================================================+
| 1      | Ascend950PR      | OK            | 208.8       57                0     / 0             |
|        |                  | 0000:71:00.0  | 0           0    / 0          5903  / 131072        |
+========+==================+===============+=====================================================+
| 2      | Ascend950PR      | OK            | 213.3       59                0     / 0             |
|        |                  | 0000:91:00.0  | 0           0    / 0          4794  / 114688        |
+========+==================+===============+=====================================================+
+---------------------------+---------------+-----------------------------------------------------+
| NPU ID                    | Process id    | Process name             | Process memory(MB)       |
+===========================+===============+=====================================================+
| No running processes found in NPU 0                                                             |
+===========================+===============+=====================================================+
| No running processes found in NPU 1                                                             |
+===========================+===============+=====================================================+
| No running processes found in NPU 2                                                             |
+===========================+===============+=====================================================+
"""

NPU_SMI_LANE1_BUSY = """\
| 0      | Ascend950PR      | OK            | 215.0       58                0     / 0             |
|        |                  | 0000:61:00.0  | 0           0    / 0          5407  / 131072        |
| 1      | Ascend950PR      | OK            | 208.8       57                0     / 0             |
|        |                  | 0000:71:00.0  | 85          0    / 0          5903  / 131072        |
| 2      | Ascend950PR      | OK            | 213.3       59                0     / 0             |
|        |                  | 0000:91:00.0  | 0           0    / 0          4794  / 114688        |
| No running processes found in NPU 0                                                             |
| 1                          | 3728646       | pytest                   | 396                     |
| No running processes found in NPU 2                                                             |
"""


# ---------------------------------------------------------------------------
# parse_npu_smi
# ---------------------------------------------------------------------------
def test_parse_npu_smi_all_idle():
    statuses = lp.parse_npu_smi(NPU_SMI_ALL_IDLE)
    assert set(statuses.keys()) == {0, 1, 2}
    for s in statuses.values():
        assert s.util_pct == 0
        assert s.has_running_process is False


def test_parse_npu_smi_lane1_busy():
    statuses = lp.parse_npu_smi(NPU_SMI_LANE1_BUSY)
    assert statuses[0].util_pct == 0
    assert statuses[1].util_pct == 85
    assert statuses[2].util_pct == 0
    assert statuses[0].has_running_process is False
    assert statuses[1].has_running_process is True
    assert statuses[2].has_running_process is False


def test_parse_npu_smi_skips_phantom_lanes():
    # Imaginary lane 3 entry — should be ignored
    bad = NPU_SMI_ALL_IDLE + (
        "| 3      | Phantom          | OK            | 0           99                0     / 0             |\n"
    )
    statuses = lp.parse_npu_smi(bad)
    assert 3 not in statuses
    assert set(statuses.keys()) == {0, 1, 2}


def test_parse_npu_smi_empty_string():
    assert lp.parse_npu_smi("") == {}


# ---------------------------------------------------------------------------
# probe_idle_npus
# ---------------------------------------------------------------------------
def test_probe_idle_npus_all_idle():
    idle = lp.probe_idle_npus(NPU_SMI_ALL_IDLE)
    assert idle == [0, 1, 2]


def test_probe_idle_npus_skips_busy():
    idle = lp.probe_idle_npus(NPU_SMI_LANE1_BUSY)
    assert idle == [0, 2]


def test_probe_idle_npus_high_util():
    """Even without process, high util means busy."""
    high_util = NPU_SMI_ALL_IDLE.replace(
        "| 0           0    / 0          5407",
        "| 90          0    / 0          5407", 1,
    )
    idle = lp.probe_idle_npus(high_util)
    assert 0 not in idle
    assert 1 in idle and 2 in idle


def test_probe_idle_threshold_relaxed():
    """Threshold=20 lets in 10% util."""
    mid = NPU_SMI_ALL_IDLE.replace(
        "| 0           0    / 0          5407",
        "| 10          0    / 0          5407", 1,
    )
    assert 0 not in lp.probe_idle_npus(mid, util_threshold=5)
    assert 0 in lp.probe_idle_npus(mid, util_threshold=20)


# ---------------------------------------------------------------------------
# Lane state files — allocate/release
# ---------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path):
    return tmp_path / "lanes"


def test_allocate_picks_first_free(root):
    ln = lp.allocate_lane("op_a", root=root)
    assert ln == 0
    s = lp.read_state(0, root=root)
    assert s["state"] == "busy"
    assert s["op"] == "op_a"


def test_allocate_skips_busy(root):
    lp.write_state(0, state="busy", op="op_x", pid=1, root=root)
    ln = lp.allocate_lane("op_b", root=root)
    assert ln == 1


def test_allocate_returns_none_when_all_busy(root):
    for ln in lp.VALID_LANES:
        lp.write_state(ln, state="busy", op=f"op_{ln}", pid=1, root=root)
    ln = lp.allocate_lane("op_new", root=root)
    assert ln is None


def test_allocate_respects_idle_filter(root):
    """idle_lanes=[2] only — even though 0/1 free, allocator picks 2."""
    ln = lp.allocate_lane("op_a", idle_lanes=[2], root=root)
    assert ln == 2


def test_allocate_skips_phantom_lane3(root):
    # idle_lanes=[3] — phantom; shouldn't allocate
    ln = lp.allocate_lane("op_a", idle_lanes=[3], root=root)
    assert ln is None


def test_release_lane(root):
    lp.write_state(0, state="busy", op="op_a", pid=1, root=root)
    lp.release_lane(0, root=root)
    s = lp.read_state(0, root=root)
    assert s["state"] == "free"
    assert s["op"] is None


def test_release_idempotent(root):
    """Releasing an already-free lane is fine."""
    lp.write_state(0, state="free", op=None, pid=None, root=root)
    lp.release_lane(0, root=root)
    s = lp.read_state(0, root=root)
    assert s["state"] == "free"


def test_release_refuses_wrong_op(root):
    lp.write_state(0, state="busy", op="op_a", pid=1, root=root)
    with pytest.raises(ValueError, match="owned by"):
        lp.release_lane(0, op="op_b", root=root)


def test_release_correct_op_succeeds(root):
    lp.write_state(0, state="busy", op="op_a", pid=1, root=root)
    lp.release_lane(0, op="op_a", root=root)
    assert lp.read_state(0, root=root)["state"] == "free"


def test_list_lanes_includes_all_valid(root):
    lanes = lp.list_lanes(root=root)
    assert set(lanes.keys()) == {0, 1, 2}
    # All free by default (no state files)
    for s in lanes.values():
        assert s["state"] == "free"


def test_list_lanes_reports_busy(root):
    lp.write_state(1, state="busy", op="op_x", pid=42, root=root)
    lanes = lp.list_lanes(root=root)
    assert lanes[0]["state"] == "free"
    assert lanes[1]["state"] == "busy"
    assert lanes[1]["op"] == "op_x"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_state_file_rejects_invalid_lane():
    with pytest.raises(ValueError, match="lane 3 not in"):
        getattr(lp, '_state_file')(3)


def test_write_state_rejects_invalid_state(root):
    with pytest.raises(ValueError, match="state must be"):
        lp.write_state(0, state="bogus", op=None, pid=None, root=root)


def test_pid_defaults_to_current_process(root):
    lp.allocate_lane("op_a", root=root)
    s = lp.read_state(0, root=root)
    assert s["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# allocate_lanes / release_lanes (multi-lane all-or-nothing)
# ---------------------------------------------------------------------------
def test_allocate_lanes_picks_n_free(root):
    lanes = lp.allocate_lanes("op_grouped", 2, root=root)
    assert lanes == [0, 1]
    for ln in (0, 1):
        s = lp.read_state(ln, root=root)
        assert s["state"] == "busy"
        assert s["op"] == "op_grouped"
    # lane 2 untouched — no state file written
    assert lp.read_state(2, root=root) is None


def test_allocate_lanes_n1_degenerates_to_single(root):
    assert lp.allocate_lanes("op_a", 1, root=root) == [0]
    assert lp.read_state(0, root=root)["state"] == "busy"


def test_allocate_lanes_n0_returns_empty(root):
    assert lp.allocate_lanes("op_a", 0, root=root) == []


def test_allocate_lanes_respects_idle_filter(root):
    assert lp.allocate_lanes("op_grouped", 2, idle_lanes=[0, 2], root=root) == [0, 2]


def test_allocate_lanes_insufficient_returns_none_and_writes_nothing(root):
    lp.write_state(2, state="busy", op="op_x", pid=1, root=root)
    lanes = lp.allocate_lanes("op_grouped", 3, root=root)
    assert lanes is None
    # all-or-nothing: the free lanes 0/1 were NOT written (no state file)
    assert lp.read_state(0, root=root) is None
    assert lp.read_state(1, root=root) is None


def test_allocate_lanes_skips_busy_mid_candidate(root):
    lp.write_state(1, state="busy", op="op_x", pid=1, root=root)
    lanes = lp.allocate_lanes("op_grouped", 2, root=root)
    assert lanes == [0, 2]
    assert lp.read_state(1, root=root)["op"] == "op_x"  # untouched


def test_allocate_lanes_returns_none_when_all_busy(root):
    for ln in lp.VALID_LANES:
        lp.write_state(ln, state="busy", op=f"op_{ln}", pid=1, root=root)
    assert lp.allocate_lanes("op_grouped", 2, root=root) is None


def test_allocate_lanes_skips_phantom_lane3(root):
    assert lp.allocate_lanes("op_grouped", 2, idle_lanes=[0, 3, 1], root=root) == [0, 1]


def test_release_lanes(root):
    lanes = lp.allocate_lanes("op_grouped", 2, root=root)
    lp.release_lanes(lanes, op="op_grouped", root=root)
    for ln in lanes:
        assert lp.read_state(ln, root=root)["state"] == "free"


def test_release_lanes_refuses_wrong_op(root):
    lanes = lp.allocate_lanes("op_grouped", 2, root=root)
    with pytest.raises(ValueError, match="owned by"):
        lp.release_lanes(lanes, op="op_other", root=root)


# ---------------------------------------------------------------------------
# AOG_LANE_IDS override (8-NPU machines such as Ascend950DT)
# ---------------------------------------------------------------------------
@pytest.fixture
def reload_lane_pool(monkeypatch):
    """Reload lane_pool with a given AOG_LANE_IDS, restoring default after."""
    import importlib

    def _reload(env_value):
        if env_value is None:
            monkeypatch.delenv("AOG_LANE_IDS", raising=False)
        else:
            monkeypatch.setenv("AOG_LANE_IDS", env_value)
        return importlib.reload(lp)

    yield _reload
    monkeypatch.delenv("AOG_LANE_IDS", raising=False)
    importlib.reload(lp)


def test_valid_lanes_default_is_three_npu_box(reload_lane_pool):
    mod = reload_lane_pool(None)
    assert mod.VALID_LANES == (0, 1, 2)


def test_aog_lane_ids_widens_pool(reload_lane_pool, root):
    mod = reload_lane_pool("0,1,2,3,4,5,6,7")
    assert mod.VALID_LANES == tuple(range(8))
    # A lane beyond the default set is now allocatable.
    ln = mod.allocate_lane("op_a", idle_lanes=[7], root=root)
    assert ln == 7
    assert set(mod.list_lanes(root=root).keys()) == set(range(8))


def test_aog_lane_ids_malformed_falls_back_to_default(reload_lane_pool):
    assert reload_lane_pool("0,1,bogus").VALID_LANES == (0, 1, 2)
    assert reload_lane_pool("").VALID_LANES == (0, 1, 2)
    assert reload_lane_pool("0,-1").VALID_LANES == (0, 1, 2)


def test_aog_lane_ids_dedupes_and_sorts(reload_lane_pool):
    assert reload_lane_pool("2, 0, 2,1").VALID_LANES == (0, 1, 2)
