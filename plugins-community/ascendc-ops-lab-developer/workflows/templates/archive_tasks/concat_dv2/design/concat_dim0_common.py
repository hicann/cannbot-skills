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

"""Shared config and helpers for concat_dv2 block-level and tile-level designs."""

import tilelang


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def select_block_m(total_m: int) -> int:
    for candidate in (64, 32, 16, 8, 4, 2):
        if total_m >= candidate:
            return candidate
    return 2


def kernel_config(total_m):
    """Return (block_m, sub_block_m, block_num) for a given total row count."""
    block_m = select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m
    return block_m, sub_block_m, block_num
