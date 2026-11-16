#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python prototype and HostCpu implementation of AddCustom."""

import ctypes

from ge.custom_op import (
    OpBackend,
    get_execute_ctx,
    register_kernel,
    register_op,
    register_op_impl,
)
from ge.graph.types import DataType
from ge.runtime import Tensor, TensorDesc


@register_op(op_type="AddCustom")
def add_custom_infer_meta(x: TensorDesc, y: TensorDesc) -> TensorDesc:
    """Infer the output metadata of AddCustom from its two inputs."""

    print("[Python] InferMeta for AddCustom", flush=True)
    if x.shape.origin_shape.dims != y.shape.origin_shape.dims:
        raise ValueError("AddCustom inputs must have the same shape")
    if x.data_type != y.data_type:
        raise ValueError("AddCustom inputs must have the same data type")
    return TensorDesc(x.shape, x.data_type)


def _float32_view(tensor: Tensor):
    """Wrap the host memory of a float32 tensor as a ctypes array."""

    if tensor.data_type != DataType.DT_FLOAT:
        raise ValueError("the sample kernel only supports float32 tensors")
    return (ctypes.c_float * int(tensor.shape_size)).from_address(int(tensor.addr))


@register_op_impl(op_type="AddCustom")
class AddCustomHostKernel:
    """HostCpu kernel of AddCustom, called by constant folding at compile time."""

    @register_kernel(backend=OpBackend.HOST)
    def execute(self, x: Tensor, y: Tensor) -> None:
        print("[Python] HostCpu execute for AddCustom", flush=True)
        x_data = _float32_view(x)
        y_data = _float32_view(y)
        if len(x_data) != len(y_data):
            raise ValueError("AddCustom inputs must have the same element count")
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
        output_data = _float32_view(output)
        for index in range(len(output_data)):
            output_data[index] = x_data[index] + y_data[index]
