# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression (2026-06-18): phase_o5_runner routes ASCEND_RT_VISIBLE_DEVICES
around a wedged lane via _resolve_visible_device, keeping `lane` for the
BENCHMARK_ROOT deploy location.

Surfaced by the .171 NPU0 external-tenant wedge that hung the lane-0 GDN /
selective_scan verifies. _resolve_visible_device is fail-open: returns `lane`
unchanged when the host is unknown / resolution declines.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import phase_o5_runner as por  # type: ignore  # noqa: E402


def test_fail_open_when_no_host(monkeypatch, tmp_path):
    """No A5_HOST → keep requested lane, never even probe npu-smi."""
    called = []
    monkeypatch.setattr(
        por, "resolve_healthy_device",
        lambda *a, **k: (called.append(1), (99, "should not be called"))[1],
    )
    assert getattr(por, '_resolve_visible_device')({"TARGET": "a5"}, tmp_path, 0) == 0
    assert called == [], "must not probe when host unknown (fail-open)"


def test_target_o5_visible_device_override_skips_health_probe(monkeypatch, tmp_path):
    """Single-device containers map physical allocation to logical 0; O5 can
    pin the container-visible device without treating the deploy lane as a
    physical id.
    """
    called = []
    monkeypatch.setattr(
        por, "resolve_healthy_device",
        lambda *a, **k: (called.append(1), (99, "should not be called"))[1],
    )
    env = {"TARGET": "a5", "A5_HOST": "1.2.3.4", "A5_O5_VISIBLE_DEVICE": "0"}
    assert getattr(por, '_resolve_visible_device')(env, tmp_path, 7) == 0
    assert called == [], "explicit O5 visible-device override must skip host probe"


def test_invalid_o5_visible_device_override_falls_back_to_health_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(por, "resolve_healthy_device", lambda lane, host, user: (lane, "healthy"))
    env = {"TARGET": "a5", "A5_HOST": "h", "A5_O5_VISIBLE_DEVICE": "not-an-int"}
    assert getattr(por, '_resolve_visible_device')(env, tmp_path, 2) == 2


def test_reroutes_off_wedged_lane(monkeypatch, tmp_path):
    """Host present + resolver reroutes 0→2 → visible device is 2."""
    seen = {}

    def _fake(lane, host, user):
        seen.update(lane=lane, host=host, user=user)
        return 2, "rerouted to device 2"
    monkeypatch.setattr(por, "resolve_healthy_device", _fake)
    env = {"TARGET": "a5", "A5_HOST": "1.2.3.4", "A5_USER": "root"}
    assert getattr(por, '_resolve_visible_device')(env, tmp_path, 0) == 2
    assert seen == {"lane": 0, "host": "1.2.3.4", "user": "root"}


def test_keeps_healthy_requested_lane(monkeypatch, tmp_path):
    monkeypatch.setattr(por, "resolve_healthy_device", lambda lane, host, user: (lane, "healthy"))
    env = {"TARGET": "a5", "A5_HOST": "h"}
    assert getattr(por, '_resolve_visible_device')(env, tmp_path, 2) == 2
