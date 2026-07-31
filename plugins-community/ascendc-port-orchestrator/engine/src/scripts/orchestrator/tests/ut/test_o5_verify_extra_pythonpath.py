# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""O5-EXTRA-PYTHONPATH (2026-07-20): the O5 SSH+docker-exec verifier builds
PYTHONPATH from CANN site-packages + inherited `${PYTHONPATH:-}` only (non-login
shell). There is an LD passthrough (`{TARGET}_EXTRA_LD_LIBRARY_PATH`) but no
PYTHONPATH analog, so a verifier needing an OUT-OF-TREE helper package had no
env hook. This adds the
symmetric `{TARGET}_EXTRA_PYTHONPATH` (then generic `EXTRA_PYTHONPATH`)
passthrough, mirroring `_resolve_extra_ld`.

Guards:
  1. resolver precedence / trailing-colon / empty-default (mirror extra_ld)
  2. container-mode setup: extra prepended onto the CANN site-packages PYTHONPATH
     export; byte-identical to today when unset (no empty segment / stray colon)
  3. host-mode export (driven through `_run_verifier`): injects the extra prefix
     when set; unchanged when unset.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

# Import phase_o5_runner FIRST: runner <-> verify have a load-order-sensitive
# circular re-export (runner:1041 imports _resolve_visible_device from verify).
# Loading runner first lets verify finish before that re-export resolves.
import phase_o5_runner  # type: ignore  # noqa: E402,F401
from phase_o5_helpers import _resolve_extra_pythonpath  # type: ignore  # noqa: E402
from phase_o5_verify import _container_npu_python_setup  # type: ignore  # noqa: E402
import phase_o5_verify as pv  # type: ignore  # noqa: E402


# ---- 1. resolver (mirror of the extra_ld resolver contract) ----

def test_target_specific_wins_over_generic():
    env = {
        "A3_EXTRA_PYTHONPATH": "/root/external_verifier",
        "EXTRA_PYTHONPATH": "/some/generic/pp",  # must NOT win for A3
    }
    assert _resolve_extra_pythonpath(env, "A3") == "/root/external_verifier"
    # A5 has no A5-specific key -> generic is used:
    assert _resolve_extra_pythonpath(env, "A5") == "/some/generic/pp"


def test_falls_back_to_generic_when_target_absent():
    assert _resolve_extra_pythonpath({"EXTRA_PYTHONPATH": "/g/pp"}, "A5") == "/g/pp"


def test_empty_when_neither_present():
    assert _resolve_extra_pythonpath({}, "A5") == ""


def test_trailing_colon_stripped():
    assert _resolve_extra_pythonpath({"A5_EXTRA_PYTHONPATH": "/x/pp:"}, "A5") == "/x/pp"


def test_target_specific_empty_string_falls_through_to_generic():
    env = {"A5_EXTRA_PYTHONPATH": "", "EXTRA_PYTHONPATH": "/g/pp"}
    assert _resolve_extra_pythonpath(env, "A5") == "/g/pp"


# ---- 2. container-mode setup string ----

_CANN = "/usr/local/Ascend/cann-9.1.T500"
_PYBIN = "/root/miniconda3/envs/py311/bin"
_LD = "/usr/local/Ascend/8.5.0/x86_64-linux/lib64"


def test_container_setup_injects_extra_pythonpath_when_set():
    _, setup = _container_npu_python_setup(_CANN, _PYBIN, _LD, "/root/external_verifier")
    # extra prefix comes BEFORE the CANN site-packages, which comes before inherited.
    assert (
        "export PYTHONPATH=/root/external_verifier:"
        "/usr/local/Ascend/cann-9.1.T500/python/site-packages:${PYTHONPATH:-}"
    ) in setup


def test_container_setup_byte_identical_when_pythonpath_unset():
    """Default (extra unset) MUST be byte-identical to today — no empty segment,
    no stray leading colon.
    """
    _, setup_default = _container_npu_python_setup(_CANN, _PYBIN, _LD)
    _, setup_empty = _container_npu_python_setup(_CANN, _PYBIN, _LD, "")
    expected = (
        "export PYTHONPATH="
        "/usr/local/Ascend/cann-9.1.T500/python/site-packages:${PYTHONPATH:-}"
    )
    assert expected in setup_default
    assert setup_default == setup_empty
    # negative: no accidental empty leading segment
    assert "export PYTHONPATH=:" not in setup_default


# ---- 3. host-mode export (driven through _run_verifier) ----

def _drive_run_verifier(tmp_path, env):
    """Invoke _run_verifier with subprocess.run mocked; return the ssh command
    list actually dispatched (last element = the remote command string)."""
    ws = tmp_path / "myop"
    ws.mkdir()
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return types.SimpleNamespace(
            returncode=0,
            stdout='{"tier1_pass": 1, "total": 1, "status": "PASS"}',
            stderr="",
        )

    import unittest.mock as _m
    with _m.patch.object(pv.subprocess, "run", side_effect=fake_run):
        getattr(pv, '_run_verifier')(ws, "myop", dict(env), "run_pass_b.py", "pass_b", lane=0)
    return captured["cmd"]


_HOST_ENV = {
    "TARGET": "a3",
    "A3_HOST": "1.2.3.4", "A3_USER": "root", "A3_PASSWORD": "",
    "A3_CANN_PATH": "/usr/local/Ascend/cann-9.1", "A3_CONTAINER": "ctr",
    "A3_NPU_PYTHON_BIN": "/opt/py311/bin",
    "A5_HOST_MODE": "1",  # host-mode is gated on A5_HOST_MODE regardless of target
}


def test_host_mode_export_injects_extra_pythonpath(tmp_path):
    env = dict(_HOST_ENV, A3_EXTRA_PYTHONPATH="/root/external_verifier")
    remote = _drive_run_verifier(tmp_path, env)[-1]
    assert 'PYTHONPATH="/root/external_verifier:${PYTHONPATH:-}"' in remote, \
        f"extra pythonpath not injected into host export: {remote!r}"


def test_host_mode_export_unchanged_when_unset(tmp_path):
    remote = _drive_run_verifier(tmp_path, dict(_HOST_ENV))[-1]
    # byte-identical to the historical host export: bare ${PYTHONPATH:-}, no prefix.
    assert 'PYTHONPATH="${PYTHONPATH:-}"' in remote, \
        f"host export changed when extra unset: {remote!r}"
    assert 'PYTHONPATH=":${PYTHONPATH:-}"' not in remote, \
        f"stray empty PYTHONPATH segment when unset: {remote!r}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
