# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host helpers shared by the two 950 examples; no shape-specialized compilation."""

from dataclasses import dataclass
from math import ceil

import cannbotdsl as cbd
from cannbotdsl import dtypes
import torch


def align16(n):
    return (n + 15) // 16 * 16


def ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class Tile:
    m: int
    n: int
    k: int


def dtype_of(tensor):
    if tensor.dtype == torch.float16:
        return dtypes.float16
    if tensor.dtype == torch.bfloat16:
        return dtypes.bfloat16
    raise ValueError("example supports FP16/BF16 only")


def check_tensors(*tensors):
    first = tensors[0]
    dtype_of(first)
    for t in tensors:
        if t.device.type != "npu" or t.device != first.device:
            raise ValueError("inputs must be on the same NPU")
        if t.dtype != first.dtype or not t.is_contiguous() or t.ndim != 5:
            raise ValueError(
                "inputs must be contiguous, rank 5, and have the same B16 dtype"
            )
        if min(t.shape) <= 0:
            raise ValueError("empty dimensions are outside this example")


def resources(requested=None, *, device=None):
    # The framework resolves launch streams from tensor devices. Querying
    # another current device here would bind the wrong hardware/stream quota.
    if device is not None and torch.npu.current_device() != device.index:
        raise ValueError(
            "set the current NPU device to the input tensor device before prepare()"
        )
    info = cbd.get_platform_info()
    if not info.available or info.cube_core_num <= 0:
        raise RuntimeError("cannot query current-stream AIC quota")
    if "3510" not in str(info.soc_version):
        raise ValueError("this example targets 950 / DAV_3510")
    count = info.cube_core_num if requested is None else requested
    if not isinstance(count, int) or not 0 < count <= info.cube_core_num:
        raise ValueError(f"blocks must be in [1, {info.cube_core_num}]")
    required = dict(l1=512 * 1024, l0a=64 * 1024, l0b=64 * 1024, l0c=256 * 1024)
    if any(cbd.get_mem_size(name) < size for name, size in required.items()):
        raise ValueError("this example uses the 950 memory partition")
    return count


def check_tile(tile):
    if min(tile.m, tile.n, tile.k) <= 0 or any(
        v % 16 for v in (tile.m, tile.n, tile.k)
    ):
        return False
    # A/B each have two slots; C is ONE slot. C's pitch never depends on K.
    return (
        2 * tile.m * tile.k * 2 <= 65536
        and 2 * tile.n * tile.k * 2 <= 65536
        and tile.m * tile.n * 4 <= 262144
    )


def wave_util(tasks, blocks):
    return tasks / (blocks * ceil(tasks / blocks))


class Invocation:
    """Own program/output until the asynchronous launch has completed."""

    def __init__(self, program, args, output, plan):
        self.program, self.args, self.output, self.plan = program, args, output, plan

    def __call__(self):
        self.program(*self.args)
        return self.output

    def close(self):
        torch.npu.synchronize(self.output.device)
        self.program.close()
