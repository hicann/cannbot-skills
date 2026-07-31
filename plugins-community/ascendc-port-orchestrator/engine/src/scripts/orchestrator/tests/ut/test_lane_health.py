# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""lane_health tests — Ascend950PR npu-smi parsing + health-aware device pick.

Fixture is the verbatim `.171` npu-smi output captured 2026-06-18 during the
GDN recurrent / selective_scan runs: NPU0 + NPU1 wedged by an external tenant
(100% util, high HBM), NPU2 healthy. This is the exact condition that hung the
hardcoded-lane-0 verify.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from lane_health import (  # type: ignore  # noqa: E402
    parse_npu_health,
    pick_healthy_device,
    resolve_healthy_device,
)

# Verbatim .171 (CANN 25.7.rc1) — NPU0/1 wedged, NPU2 healthy.
_NPU_SMI_171 = """\
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1                                 Version: 25.7.rc1                              |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | Alarm         | 205.7       71                0     / 0             |
|        |                  | 0000:71:00.0  | 100         0    / 0          90633 / 114688        |
+========+==================+===============+=====================================================+
| 1      | Ascend950PR      | Alarm         | 207.7       60                0     / 0             |
|        |                  | 0000:61:00.0  | 100         0    / 0          30492 / 114688        |
+========+==================+===============+=====================================================+
| 2      | Ascend950PR      | OK            | 192.7       57                0     / 0             |
|        |                  | 0000:81:00.0  | 0           0    / 0          5413  / 131072        |
+========+==================+===============+=====================================================+
+---------------------------+---------------+-----------------------------------------------------+
| NPU ID                    | Process id    | Process name             | Process memory(MB)       |
+===========================+===============+=====================================================+
| 0                         | 1530715       | python                   | 4724                    |
| 0                         | 1655234       | python                   | 4728                    |
+===========================+===============+=====================================================+
"""


def test_parse_950pr_three_devices():
    h = parse_npu_health(_NPU_SMI_171)
    assert set(h) == {0, 1, 2}, f"expected devices 0,1,2; got {sorted(h)}"
    assert h[0].health == "Alarm" and h[0].util_pct == 100.0
    assert h[0].hbm_used_mb == 90633 and h[0].hbm_total_mb == 114688
    assert h[2].health == "OK" and h[2].util_pct == 0.0
    assert h[2].hbm_used_mb == 5413 and h[2].hbm_total_mb == 131072
    assert abs(h[2].hbm_frac - 5413 / 131072) < 1e-9


def test_process_table_not_parsed_as_devices():
    """The bottom process-list table has `| 0 | <pid> | python | <mem> |` rows
    that are summary-shaped; they must NOT corrupt device 0's health (no chip
    row follows them, so the pending summary is discarded).
    """
    h = parse_npu_health(_NPU_SMI_171)
    # device 0 keeps its real chip-row values, not a process-row artifact
    assert h[0].hbm_total_mb == 114688
    assert len(h) == 3


def test_health_flags():
    h = parse_npu_health(_NPU_SMI_171)
    assert not h[0].is_healthy()   # Alarm + 79% HBM + 100% util
    assert not h[1].is_healthy()   # Alarm + 100% util
    assert h[2].is_healthy()       # OK + 4% HBM + 0% util


def test_pick_reroutes_wedged_lane0_to_healthy_lane2():
    h = parse_npu_health(_NPU_SMI_171)
    dev, reason = pick_healthy_device(0, h)
    assert dev == 2, f"wedged lane 0 should reroute to healthy 2; got {dev} ({reason})"
    assert "rerouted" in reason


def test_pick_keeps_healthy_requested_lane():
    h = parse_npu_health(_NPU_SMI_171)
    dev, reason = pick_healthy_device(2, h)
    assert dev == 2 and "healthy" in reason


def test_pick_fail_open_on_empty_health():
    dev, reason = pick_healthy_device(0, {})
    assert dev == 0 and "no health data" in reason


def test_pick_fail_open_when_no_healthy_device():
    """All devices wedged → keep requested (don't pretend a wedged one is fine,
    don't crash).
    """
    h = parse_npu_health(_NPU_SMI_171)
    del h[2]  # drop the only healthy device
    dev, reason = pick_healthy_device(0, h)
    assert dev == 0 and "NO healthy device" in reason


def test_resolve_fail_open_on_npu_smi_error():
    def _boom(user, host):
        return 1, "", "ssh: connect refused"
    dev, reason = resolve_healthy_device(0, "h", run_npu_smi=_boom)
    assert dev == 0 and "fail-open" in reason


def test_resolve_reroutes_via_injected_smi():
    def _smi(user, host):
        return 0, _NPU_SMI_171, ""
    dev, reason = resolve_healthy_device(0, "h", run_npu_smi=_smi)
    assert dev == 2 and "rerouted" in reason
