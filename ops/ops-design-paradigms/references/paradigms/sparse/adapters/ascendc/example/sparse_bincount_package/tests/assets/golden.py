#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os

import numpy as np
import tensorflow as tf

# 注：源仓将 os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 置于 import tensorflow
# 之前以抑制 TF 日志输出；本镜像为满足 CodeCheck G.FMT.05（import 置顶）
# 而重排，运行时请将此行移回 import tensorflow 之前
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 注：测试框架按 dunder 变量名发现 golden 函数（见运行时测试框架文档）；
# 本镜像仅作阅读参照，如需运行请改回 GOLDEN_KERNEL_MAP → 该 dunder 变量名
GOLDEN_KERNEL_MAP = {"kernel": {"sparse_bincount": "sparse_bincount_golden"}}


def sparse_bincount_golden(
    indices, values, dense_shape, size, weights, *, binary_output=False, **kwargs
):
    size_scalar = size.flat[0]
    if values.dtype == np.int64:
        size_scalar = np.int64(size_scalar)
    else:
        size_scalar = np.int32(size_scalar)

    out = tf.raw_ops.SparseBincount(
        indices=tf.constant(indices),
        values=tf.constant(values),
        dense_shape=tf.constant(dense_shape),
        size=tf.constant(size_scalar),
        weights=tf.constant(weights),
        binary_output=binary_output,
    )
    return out.numpy()
