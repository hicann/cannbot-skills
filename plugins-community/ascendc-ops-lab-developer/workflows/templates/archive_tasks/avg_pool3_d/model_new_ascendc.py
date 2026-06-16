#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys
from pathlib import Path
import importlib.util

# Template for: avg_pool3_d
import torch
import torch.nn as nn

try:
    import ascend_kernel
except ImportError:
    pass

# Load shared helpers from sibling model.py
_model_path = Path(__file__).resolve().parent / "model.py"
_spec = importlib.util.spec_from_file_location("_model", _model_path)
_model_module = importlib.util.module_from_spec(_spec)
sys.modules["_model"] = _model_module
_spec.loader.exec_module(_model_module)
resolve_scenario = _model_module.resolve_scenario


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scenario = self._resolve_scenario(x)
        divisor_override = scenario["divisor_override"] or None
        return torch.ops.npu.avg_pool3d(
            x,
            kernel_size=list(scenario["kernel_size"]),
            stride=list(scenario["stride"]),
            padding=list(scenario["padding"]),
            ceil_mode=scenario["ceil_mode"],
            count_include_pad=scenario["count_include_pad"],
            divisor_override=divisor_override,
        )

    def _resolve_scenario(self, x: torch.Tensor):
        return resolve_scenario(x)
