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

import numpy as np

golden_registry = {"kernel": {"three_interpolate": "three_interpolate_golden"}}


def three_interpolate_golden(features, idx, weight, **kwargs):
    """
    Golden function for three_interpolate.
    All the parameters (names and order) follow @three_interpolate.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        features: numpy array, shape (B, M, C), dtype float32/float16
        idx: numpy array, shape (B, N, 3), dtype int32/int64
        weight: numpy array, shape (B, N, 3), dtype float32/float16
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor, shape (B, N, C), dtype float32/float16
    """
    b_num, m_num, c_num = features.shape
    _, n_num, _ = idx.shape

    compute_dtype = np.float32
    features_compute = features.astype(compute_dtype)
    weight_compute = weight.astype(compute_dtype)

    y = np.zeros((b_num, n_num, c_num), dtype=compute_dtype)

    # Compute using single expression (w0*f0 + w1*f1 + w2*f2) per element,
    # matching the kernel's computation order exactly.
    # This avoids sequential += accumulation which produces different float32
    # rounding due to the extra (0 + w0*f0) intermediate step.
    for b in range(b_num):
        for n in range(n_num):
            i0 = idx[b, n, 0]
            i1 = idx[b, n, 1]
            i2 = idx[b, n, 2]
            w0 = weight_compute[b, n, 0]
            w1 = weight_compute[b, n, 1]
            w2 = weight_compute[b, n, 2]
            y[b, n, :] = (
                w0 * features_compute[b, i0, :]
                + w1 * features_compute[b, i1, :]
                + w2 * features_compute[b, i2, :]
            )

    # Cast back to original dtype
    return y.astype(features.dtype)
