# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""编译期路由分派；路由号仅属于 DSL，不与 AscendC schedule 混用。"""

import cannbotdsl as cb
from sort_radix_more_core import RadixMoreCore
from sort_radix_one_core import NarrowRadixOneCore, RadixOneCore
from sort_common import dtype_bytes
from sort_small_axis_insertion import InsertionSort
from sort_small_axis_two_stage import SmallAxisTwoStage


# 路由编号用于编译特化与导出。
ROUTE_INSERTION = 1
ROUTE_RADIX_ONE_CORE = 2
ROUTE_RADIX_MORE_CORE = 3
ROUTE_TWO_STAGE_INVERSE = 4
ROUTE_TWO_STAGE_GROUPED = 5


def validate_route(route, groups):
    """在 Python/编译期拒绝未知编号，避免静默生成空调用或错误路线。"""
    if route not in (
        ROUTE_INSERTION,
        ROUTE_RADIX_ONE_CORE,
        ROUTE_RADIX_MORE_CORE,
        ROUTE_TWO_STAGE_INVERSE,
        ROUTE_TWO_STAGE_GROUPED,
    ):
        raise ValueError(f"unknown Sort route: {route}")
    if route == ROUTE_RADIX_MORE_CORE and groups < 0:
        raise ValueError("active row capacity must be nonnegative")
    if route == ROUTE_RADIX_ONE_CORE and groups != 1:
        raise ValueError("single-core Radix requires groups == 1")


@cb.host
def sort_entry(
    x: cb.Tensor,
    values: cb.Tensor,
    indices: cb.Tensor,
    ping: cb.Tensor,
    ping_ids: cb.Tensor,
    pong: cb.Tensor,
    pong_ids: cb.Tensor,
    hist: cb.Tensor,
    states: cb.Tensor,
    rows: cb.dtypes.int64,
    length: cb.dtypes.int64,
    inner: cb.dtypes.int64,
    blocks: cb.dtypes.int64,
    route: cb.Constexpr[int],
    width: cb.Constexpr[int],
    groups: cb.Constexpr[int],
    descending: cb.Constexpr[bool],
):
    validate_route(route, groups)
    if route == ROUTE_INSERTION:
        InsertionSort(x.dtype, width, groups, descending)[blocks](
            x, values, indices, rows, length, inner, blocks
        )
    elif route == ROUTE_TWO_STAGE_INVERSE or route == ROUTE_TWO_STAGE_GROUPED:
        SmallAxisTwoStage(
            x.dtype, width, groups, descending, route == ROUTE_TWO_STAGE_INVERSE
        )[blocks](x, values, indices, rows, length, inner, blocks)
    elif route == ROUTE_RADIX_ONE_CORE:
        if dtype_bytes(x.dtype) <= 2:
            NarrowRadixOneCore(x.dtype, width, groups, descending)[blocks](
                x, values, indices, rows, length, inner, blocks
            )
        else:
            RadixOneCore(x.dtype, width, groups, descending)[blocks](
                x, values, indices, rows, length, inner, blocks
            )
    elif route == ROUTE_RADIX_MORE_CORE:
        RadixMoreCore(x.dtype, descending, width, hist.dtype, groups)[blocks](
            x,
            values,
            indices,
            ping,
            ping_ids,
            pong,
            pong_ids,
            hist,
            states,
            rows,
            length,
            inner,
            blocks,
        )
