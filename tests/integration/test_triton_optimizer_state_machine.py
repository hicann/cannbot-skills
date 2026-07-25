# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for Triton Optimizer's static-analysis refactors."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = (
    REPO_ROOT
    / "plugins-community/triton-optimizer/skills/triton-npu-optimize/scripts"
)


def _load_module(module_name: str, relative_path: str):
    path = SCRIPTS_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module specification: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_state_machine_persists_active_round(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT / "optimize-state"))
    module = _load_module(
        "triton_optimizer_state_machine",
        "optimize-state/state_manage/state_machine.py",
    )
    state_path = tmp_path / ".triton-agent" / "state.json"

    module.bootstrap_state(state_path, run_id="test", baseline_reused=False)
    module.mark_baseline_passed(state_path)
    result = module.start_round(
        state_path,
        "opt-round-1",
        round_strategy="exploration",
        analysis_policy="pattern_entry",
        reason="regression test",
    )

    state = module.load_state(state_path)
    if result["round"] != "opt-round-1":
        raise RuntimeError("round state did not preserve the active round name")
    if state["current_round"] != 1:
        raise RuntimeError("round state did not preserve the active round number")
    if state["phase"] != module.PHASE_ROUND_ACTIVE:
        raise RuntimeError("round state did not enter the active phase")


def test_binary_profile_summary_preserves_key_signals(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT / "profile-operator"))
    module = _load_module(
        "triton_optimizer_parse_bin",
        "profile-operator/parse_bin.py",
    )
    summary = module.summarize_results(
        [
            {
                "json": {
                    "name": "kernel",
                    "duration": "3.0",
                    "op_type": "vector",
                    "block_dim": 1,
                    "block_detail": {"head_name": ["id"], "row": [{"value": ["0"]}]},
                }
            },
            {"json": {"subblock_detail": [{"block_id": 0, "name": "pipe", "value": "88"}]}},
            {"json": {}},
            {"json": {}},
            {"json": {}},
        ]
    )

    if summary["base_info"]["name"] != "kernel":
        raise RuntimeError("profile summary lost the kernel name")
    if summary["pipe_utilization"]["top_pipe"]["utilization_percent"] != 88.0:
        raise RuntimeError("profile summary lost the pipe utilization value")
