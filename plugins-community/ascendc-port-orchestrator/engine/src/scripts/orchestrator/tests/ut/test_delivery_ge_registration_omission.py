# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Delivery-promotion GE-registration omission tests (2026-08-30, PR13 WP-A / A.4.1).

The strict TileLang2AscendC delivery profile previously hard-failed promotion
on the GE registration subtree ``op_host/config/**`` (``*_binary.json`` /
``*_simplified_key.ini``) that the kw brief asks the worker to emit but the
direct-launch (non-GE) delivery contract never consumes — a leftover
``2_FFN_evo_simplified_key.ini`` blocked an otherwise-PASS finalize with a
bare exit 7 (2026-08-29).  The whole subtree is now omitted with a WARN;
fail-closed behavior stays reserved for genuinely unknown file types.

Run: cd src/scripts && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
     orchestrator/tests/ut/test_delivery_ge_registration_omission.py -q
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # orchestrator/

from plugins.port_a3 import (  # noqa: E402
    TILELANG_PROFILE_VALID,
    PortA3Plugin,
    _delivery_path_is_archived,
    _delivery_path_rejection,
)

_STRICT = TILELANG_PROFILE_VALID


def test_ge_registration_subtree_is_omitted_not_rejected() -> None:
    for rel in (
        "op_host/config/ascend950/op_x_binary.json",
        "op_host/config/ascend950/op_x_simplified_key.ini",
        "op_host/config/ascend950/op_x.json",
    ):
        assert _delivery_path_rejection(_STRICT, rel) is None, rel
        # Omitted from the archive, not silently delivered.
        assert _delivery_path_is_archived(_STRICT, rel) is False, rel


def test_ge_registration_omission_records_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert (
            _delivery_path_rejection(
                _STRICT, "op_host/config/ascend950/op_x_simplified_key.ini"
            )
            is None
        )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("GE registration" in r.getMessage() for r in warnings)
    assert any("op_host/config/ascend950/op_x_simplified_key.ini" in r.getMessage() for r in warnings)


def test_plugin_profile_hook_matches_module_classifier() -> None:
    """The plugin static hook used by finalize_dispatch must agree."""
    # Bind the strict-delivery classifier once instead of reaching into the
    # plugin's protected attributes at the call site (same pattern as
    # plugins/port_a3/tests/test_port_a3_plugin.py).
    archive_path_rejection = getattr(PortA3Plugin, "_archive_path_rejection_for_profile")
    assert (
        archive_path_rejection(
            _STRICT, "op_host/config/ascend950/op_x_binary.json"
        )
        is None
    )


def test_unknown_op_host_file_still_fails_closed() -> None:
    reason = _delivery_path_rejection(_STRICT, "op_host/op_x_mystery.xyz")
    assert reason is not None and "unrecognized file" in reason


def test_unknown_kernel_file_still_fails_closed() -> None:
    reason = _delivery_path_rejection(_STRICT, "kernel/op_kernel/strange.xyz")
    assert reason is not None and "unrecognized file" in reason


def test_hidden_artifact_under_config_still_fails_closed() -> None:
    reason = _delivery_path_rejection(_STRICT, "op_host/config/.hidden")
    assert reason is not None and "hidden artifact" in reason


def test_binary_artifact_under_config_still_fails_closed() -> None:
    reason = _delivery_path_rejection(_STRICT, "op_host/config/ascend950/op_x.so")
    assert reason is not None and "binary/build artifact" in reason


def test_ge_ophost_delivery_trio_still_retained() -> None:
    """The GE delivery trio at top-level op_host/ stays part of delivery."""
    assert _delivery_path_is_archived(_STRICT, "op_host/op_x_def.cpp") is True
    assert _delivery_path_is_archived(_STRICT, "op_host/ge_host_shim.h") is True
