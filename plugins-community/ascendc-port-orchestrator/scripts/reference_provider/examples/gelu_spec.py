# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Example forward spec for ascendc-backward-gen (GELU).

Defines a differentiable PyTorch `forward` + a `BACKWARD_SPEC` dict. Feed this file to the
engine's backward mode:

    cd <plugin>/engine
    bash src/scripts/orch --backward ../scripts/reference_provider/examples/gelu_spec.py

The engine derives the op name as `<basename>_grad` (here: gelu_spec_grad), generates the fp64
autograd backward truth, then generates + builds + verifies a real backward AscendC kernel.
"""
import torch.nn.functional as F


def forward(x):
    # erf-exact GELU (approximate='none'); F.gelu default is erf-exact.
    return F.gelu(x)


BACKWARD_SPEC = {
    "wrt": ["x"],                       # differentiate w.r.t. x
    "inputs": {"x": {"shape": ["N"]}},  # "N" resolves from each case below
    "cases": [{"N": 1024}, {"N": 4096}],
    "dtypes": ["float16", "bfloat16", "float32"],
    "seed": 1234,
}
