#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np

# 注：测试框架按 dunder 变量名 __golden__ 发现 golden 函数映射；
# 本镜像为满足 CodeCheck G.NAM.05/G.FMT.05 改为常规命名并保持 import 置顶，
# 如需接入测试框架运行，请改回 __golden__ 变量名
GOLDEN_KERNEL_MAP = {
    "kernel": {"arg_max_with_value": "arg_max_with_value_golden"},
    "aclnn": {"aclnnMaxDim": "aclnn_max_dim_golden"},
}


def arg_max_with_value_golden(
    x, *, dimension, keep_dims=False, indice_dtype=None, **kwargs
):
    """
    Kernel golden for arg_max_with_value.
    All the parameters follow @arg_max_with_value.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    output_dtypes = kwargs.get("output_dtypes", ["int32", "float32"])
    indice_dtype = output_dtypes[0] if indice_dtype is None else indice_dtype

    x_bak = x
    if "bfloat16" in str(ori_dtype).lower():
        x = x.astype("float32")

    axis = dimension % len(x.shape)
    indices = np.argmax(x, axis=axis).astype(indice_dtype, copy=False)
    values = np.take_along_axis(x_bak, np.expand_dims(indices, axis=axis), axis=axis)
    return indices, values


def aclnn_max_dim_golden(*args, **kwargs):
    """
    ACLNN golden for aclnnMaxDim / arg_max_with_value.
    Called via _call_custom_golden:
      args[0]: input tensor x (torch.Tensor or numpy.ndarray)
      args[1]: dim (scalar int)
      args[2]: keepdim (scalar bool)
      rest:   output tensors / other inputs (ignored)
    """
    x, dim, keepdim = args[0], args[1], args[2]

    x_backup = x
    if isinstance(x, np.ndarray):
        if "bfloat16" in str(x.dtype).lower():
            x = x.astype("float32")
        axis = dim % len(x.shape)
        indices = np.argmax(x, axis=axis)
        values = np.take_along_axis(
            x_backup, np.expand_dims(indices, axis=axis), axis=axis
        )
        if keepdim:
            indices = np.expand_dims(indices, axis=axis)
            values = np.expand_dims(values, axis=axis)
        return values, indices
    else:
        import torch

        values, indices = torch.max(x, dim=dim, keepdim=keepdim)
        return values, indices
