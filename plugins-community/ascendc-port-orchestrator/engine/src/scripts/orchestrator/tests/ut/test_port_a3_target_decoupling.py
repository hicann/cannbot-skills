# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests — port_a3 ↔ TARGET env decoupling (task#24 item2,
design `PLUGIN_PARADIGM_NOTES.md#port-a3-target-decoupling-design`).

Pins three things:
  (1) AscendCEnv parses dedicated A5 build-target fields from A5_* keys; absent →
      "" (no behavior change for the no-A5_HOST population).
  (2) The phase_o5_runner A5-build resolvers are MODE-GATED on port_a3: A5_HOST
      precedence applies ONLY in port_a3 mode. The load-bearing regression guard
      (main review): a TARGET=a3 + A5_HOST-set agent in a NON-port_a3 mode (e.g.
      back-agent backward build) must STILL resolve {target}_HOST — a blanket
      flip would misroute its build to A5.
  (3) Byte-identical fallback when no A5_HOST is set (old behavior).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
for p in (str(_ORCH_DIR), str(_ORCH_DIR / "briefs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase_o5_runner as o5  # type: ignore
from briefs._common import _build_env_from_kv  # type: ignore


# --- (1) env field parsing -------------------------------------------------

def test_env_parses_a5_fields_when_present():
    kv = {
        "TARGET": "a3", "A3_HOST": "a3.host", "A3_CONTAINER": "a3c",
        "A5_HOST": "a5.host", "A5_CONTAINER": "a5c",
        "A5_CANN_PATH": "/a5/cann", "A5_SOC_VERSION": "Ascend950PR",
    }
    env = _build_env_from_kv(kv, target="a3", backend="ascendc",
                             archive_project="test_archive", _tp_base="A3")
    assert env.a5_host == "a5.host"
    assert env.a5_container == "a5c"
    assert env.a5_cann_path == "/a5/cann"
    assert env.a5_soc_version == "Ascend950PR"
    # generic host resolution UNCHANGED (TARGET=a3 → A3_HOST):
    assert env.host == "a3.host"
    assert env.a3_host == "a3.host"


def test_env_a5_fields_default_empty_when_absent():
    kv = {"TARGET": "a5", "A5_HOST": "a5.host", "A5_CONTAINER": "a5c"}
    # no A3 keys, no A5_CANN_PATH / A5_SOC_VERSION
    env = _build_env_from_kv(kv, target="a5", backend="ascendc",
                             archive_project="test_archive", _tp_base="A5")
    assert env.a5_host == "a5.host"      # parsed
    assert env.a5_cann_path == ""        # absent → ""
    assert env.a5_soc_version == ""


# --- (2) mode-gated build resolvers ----------------------------------------

def _patch_mode(monkeypatch, is_port_a3: bool):
    monkeypatch.setattr(o5, "_is_port_a3_mode", lambda ws: is_port_a3)


def test_build_host_port_a3_prefers_a5_host(monkeypatch, tmp_path):
    _patch_mode(monkeypatch, True)  # port_a3 mode
    env = {"A3_HOST": "a3.host", "A5_HOST": "a5.host"}
    # target=A3 (TARGET=a3 agent), but port_a3 → A5_HOST wins:
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A3") == "a5.host"


def test_build_host_non_port_a3_keeps_target_host_even_with_a5_set(monkeypatch, tmp_path):
    """THE regression guard (main review): a non-port_a3 build (backward/benchmark)
    on a TARGET=a3 agent that has A5_HOST set must STILL build on A3_HOST. A blanket
    A5_HOST-precedence flip would misroute the back-agent's backward build to A5.
    """
    _patch_mode(monkeypatch, False)  # NOT port_a3 (e.g. backward)
    env = {"A3_HOST": "a3.host", "A5_HOST": "a5.host"}
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A3") == "a3.host"  # NOT a5.host


def test_build_host_target_a5_agent_unchanged(monkeypatch, tmp_path):
    """TARGET=a5 agent: {target}==A5==A5_HOST, so the flip is a no-op (port_a3 or not)."""
    env = {"A5_HOST": "a5.host"}
    _patch_mode(monkeypatch, True)
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A5") == "a5.host"
    _patch_mode(monkeypatch, False)
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A5") == "a5.host"


def test_build_cann_path_mode_gated(monkeypatch, tmp_path):
    env = {"A3_CANN_PATH": "/a3/cann", "A5_CANN_PATH": "/a5/cann"}
    _patch_mode(monkeypatch, True)
    assert getattr(o5, '_a5_build_cann_path')(env, tmp_path, "A3") == "/a5/cann"   # port_a3 → A5 CANN
    _patch_mode(monkeypatch, False)
    assert getattr(o5, '_a5_build_cann_path')(env, tmp_path, "A3") == "/a3/cann"   # non-port_a3 → A3 CANN


def test_build_container_mode_gated(monkeypatch, tmp_path):
    env = {"A3_CONTAINER": "a3c", "A5_CONTAINER": "a5c"}
    _patch_mode(monkeypatch, True)
    assert getattr(o5, '_a5_build_container')(env, tmp_path, "A3") == "a5c"
    _patch_mode(monkeypatch, False)
    assert getattr(o5, '_a5_build_container')(env, tmp_path, "A3") == "a3c"


# --- (3) no-A5_HOST fallback = old behavior --------------------------------

def test_build_host_no_a5_host_falls_back(monkeypatch, tmp_path):
    """No A5_HOST configured → {target}_HOST, identical to pre-change."""
    env = {"A3_HOST": "a3.host"}  # no A5_HOST
    _patch_mode(monkeypatch, True)   # even in port_a3 mode, no A5_HOST → fall back
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A3") == "a3.host"
    _patch_mode(monkeypatch, False)
    assert getattr(o5, '_a5_build_host')(env, tmp_path, "A3") == "a3.host"
