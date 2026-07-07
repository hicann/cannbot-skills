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

"""Shared config and helpers for int8_matmul_scale block-level and tile-level designs."""

from types import SimpleNamespace

import tilelang


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def kernel_setup(m_size, n_size, k_size):
    """Return common block/tile sizing and grid parameters."""
    cfg = SimpleNamespace()
    cfg.block_m = 128
    cfg.block_n = 256
    cfg.block_k = 64
    cfg.k_l1 = 256
    cfg.m_num = m_size // cfg.block_m
    cfg.n_num = n_size // cfg.block_n
    return cfg
