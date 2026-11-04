#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NPUGetFloatStatus 算子 Golden 脚本

版权所有 (c) Huawei Technologies Co., Ltd. 2024-2026. 版权所有。
CANN Open Software License...

算子功能：读取 NPU 硬件浮点溢出状态寄存器。
- 输入 addr (float32, ND, shape [8])：接收溢出状态写回的 tensor（side effect）
- 输出 data (float32, ND, shape [8])：固定全零，不承载状态信息

Golden 限制：CPU 环境无法读取 NPU 硬件溢出状态寄存器，
因此 Golden 仅验证输出 tensor（固定全零），
输入 side-effect（写回 1.0）不在 Golden 比对范围内。
"""

import torch

GOLDEN_TEST_SPEC = {
    "npu_get_float_status": "NPUGetFloatStatusTestSpec",
}


def _torch_compute():
    """共享计算核心：输出固定为 [8] 个 float32 零值。

    NPUGetFloatStatus 读取 NPU 硬件浮点溢出状态寄存器：
    - 输出 tensor `data` 固定为 [8] 个 float32 零值（占位输出）
    - 实际溢出状态写回输入 tensor `addr` 全部 8 个元素（side effect，不在 Golden 比对范围）
    """
    return torch.zeros(8, dtype=torch.float32)


class NPUGetFloatStatusTestSpec:
    """Kernel/GEIR 共用 TestSpec。

    Input:  addr  - float32, ND, shape [8]  (overflow status writeback carrier)
    Output: data  - float32, ND, shape [8]  (fixed all zeros)
    """

    def golden(self, addr, **kwargs):
        """Golden 输入为 numpy.ndarray，返回 list。

        输出固定为 [8] 个 float32 零值，与输入值和硬件状态无关。
        """
        result = _torch_compute()
        return [result.detach().cpu().numpy()]

    class ThirdPartyImpl:
        """torch provider 三方实现。

        GEIR 远端派发时接收 torch.Tensor，返回 list[torch.Tensor]。
        输出与输入值无关（固定全零），provider 独立执行，不反调 Golden wrapper。
        """

        def __init__(self, **kwargs):
            pass

        def __call__(self, addr, **kwargs):
            return [_torch_compute()]

    # GEIR 远端派发需要显式 provider，direct class 不可移植
    third_party = {"torch": ThirdPartyImpl}

    # 浮点 dtype 输出按三方 golden 规范使用 cross_check
    # （输出固定全零，实际比对天然 bit-exact，不受精度容差影响）
    tolerance = {
        "float32": {"standard": "cross_check"},
    }
