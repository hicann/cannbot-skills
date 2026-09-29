#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""最小编译探针：仅验证 Host/Kernel 入口，不执行设备计算。

保留一个 Tensor 参数以满足编译入口的张量契约；空 Kernel 不依赖
内存分配、搬运或核信息 API。前端读取 AST，因此必须保留真实源文件。
"""

from cannbotdsl import Tensor, host, kernel


@kernel
class ProbeKernel:
    def __init__(self):
        pass

    def __call__(self, a: Tensor):
        pass


@host
def probe_entry(a: Tensor):
    ProbeKernel()[1](a)
