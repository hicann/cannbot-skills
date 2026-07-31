# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-097 regression suite: phase_o5_perf_capture wire-in.

Owner direction 2026-05-27 ("precision_probe 怎么可能占据 O5 的位置 / 独立验证
怎么可能在 independent review 场景就没了") surfaced a pre-existing structural gap:

`phase_o5_perf_capture.measure_op_perf` had ZERO callers across
orchestrator.py / finalize_pipeline.py / plugins/ for 9 days after module
landing (`07a770ed` 2026-05-18). Independent perf re-verify was effectively
absent for ALL paradigms.

Tests below pin:
- Plugin protocol hook `should_run_phase_o5_perf_capture()` default + override.
- orchestrator.py imports phase_o5_perf_capture at module level.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent / "plugins"))

from plugins.base import BasePlugin  # noqa: E402


# ---------------------------------------------------------------------------
# Plugin hook contract
# ---------------------------------------------------------------------------

def test_base_plugin_should_run_phase_o5_perf_capture_default_true():
    """BasePlugin default must opt every paradigm IN (True) — independent
    verification is the default for all paradigms unless a plugin has a
    structural reason to opt out.
    """
    plug = BasePlugin()
    assert plug.should_run_phase_o5_perf_capture() is True


def test_concrete_plugin_can_opt_out():
    """A plugin can opt out by overriding the hook to return False."""

    class OptOutPlugin(BasePlugin):
        name = "opt_out"

        def should_run_phase_o5_perf_capture(self) -> bool:
            return False

    plug = OptOutPlugin()
    assert plug.should_run_phase_o5_perf_capture() is False


def test_backward_plugin_opts_out_of_phase_o5_perf_capture():
    """BackwardPlugin MUST opt out — backward has its own NPU perf path
    (backward_verify_runner runs verify_<op>.py via {TARGET}_NPU_PYTHON_BIN).
    The inherited default (True) makes the orchestrator fall into
    phase_o5_perf_capture, whose subprocess runs the orchestrator's LOCAL
    `python3` (no torch_npu) -> crashes on `import torch_npu` -> perf=N/A.
    Caught 2026-06-19 on the GQA-bwd orch archive run.
    """
    from plugins.backward import BackwardPlugin  # noqa: E402

    assert BackwardPlugin().should_run_phase_o5_perf_capture() is False


# ---------------------------------------------------------------------------
# Wire-in smoke (module import + symbol availability)
# ---------------------------------------------------------------------------

def test_phase_o5_perf_capture_imported_in_orchestrator():
    """orchestrator.py must import phase_o5_perf_capture at module level so
    the finalize-state wire-in does not pay an import cost per finalize.
    """
    import orchestrator  # noqa: F401  re-import safe
    assert hasattr(orchestrator, "phase_o5_perf_capture"), (
        "orchestrator.phase_o5_perf_capture missing — wire-in regressed"
    )
    # Module must expose measure_op_perf (the wire-in target)
    assert callable(orchestrator.phase_o5_perf_capture.measure_op_perf)
