# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hook_runtime.skill_loader import load_skill_script_module


@lru_cache(maxsize=1)
def _optimize_baseline_module():
    return load_skill_script_module(
        "triton-npu-optimize",
        "optimize-state/baseline/check",
    )


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"


def load_baseline_state(workspace: Path) -> Any:
    return _optimize_baseline_module().load_baseline_state(workspace)


def inspect_baseline_artifacts(workspace: Path) -> Any:
    return _optimize_baseline_module().inspect_baseline_artifacts(workspace)


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    return _optimize_baseline_module().baseline_gate_issues(workspace)
