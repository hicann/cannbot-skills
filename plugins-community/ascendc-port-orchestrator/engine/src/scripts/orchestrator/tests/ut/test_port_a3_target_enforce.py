# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-PORT-A3-TARGET-ENFORCE (2026-06-05) regression.

port_a3_to_a5 mode structurally targets A5. A non-a5 `.ascendc_env target` in
port_a3 mode silently split the worker build-host from the O5 verify-host
(`phase_o5_runner._a5_build_host` returns A5_HOST in port_a3 regardless of
target) → 0/N ImportError mis-read as precision FAIL (cost the FA-A5 gate-a
sprint hours). `enforce_port_a3_target` forces a5 + warns loud.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import enforce_port_a3_target


def test_port_a3_with_a3_target_is_overridden_to_a5_with_loud_warning():
    """The exact gate-a incident: port_a3 + target=a3 → forced a5 + warning."""
    target, warn = enforce_port_a3_target("port_a3_to_a5", "a3")
    assert target == "a5"
    assert warn is not None
    assert "DEBT-PORT-A3-TARGET-ENFORCE" in warn
    assert "a3" in warn  # the offending value is surfaced
    assert "a5" in warn  # the override target is named


def test_port_a3_with_a5_target_is_unchanged_no_warning():
    """The correct config: port_a3 + target=a5 → no-op, no warning."""
    target, warn = enforce_port_a3_target("port_a3_to_a5", "a5")
    assert target == "a5"
    assert warn is None


def test_port_a3_with_other_nona5_target_is_overridden():
    """Any non-a5 target in port_a3 (e.g. a2) is a contradictory config → forced a5."""
    target, warn = enforce_port_a3_target("port_a3_to_a5", "a2")
    assert target == "a5"
    assert warn is not None


def test_non_port_a3_mode_target_is_not_touched():
    """Non-port modes legitimately use other targets and are never overridden."""
    for mode in ("backward", "unsupported"):
        for tgt in ("a3", "a5", "a2"):
            target, warn = enforce_port_a3_target(mode, tgt)
            assert target == tgt, f"{mode}/{tgt} must NOT be overridden"
            assert warn is None, f"{mode}/{tgt} must NOT warn"


if __name__ == "__main__":
    test_port_a3_with_a3_target_is_overridden_to_a5_with_loud_warning()
    test_port_a3_with_a5_target_is_unchanged_no_warning()
    test_port_a3_with_other_nonA5_target_is_overridden()
    test_non_port_a3_mode_target_is_NOT_touched()
    logging.info("all DEBT-PORT-A3-TARGET-ENFORCE tests passed")
