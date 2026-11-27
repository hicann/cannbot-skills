# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Conv 1D/2D/3D DSL 设备验收（数值 + 缓存/空输出契约）。

覆盖：快速集（1D 升维、2D 非对称 pad/stride/dilation、3D 真 kd 归约、程序缓存复用、空输出不编译）
与泛化矩阵（31 规格：Co/Ci 非 16 倍数、1x1、大核、stride 2/3、dilation、宽/窄输入、
大 Ci 驻留/流式回退、多 batch、3D）。

设备测试需要可用 NPU、已加载的 CANN 环境，以及包含 CANNBotDSL、PyTorch 和 torch_npu 的 Python 环境。
**NPU 测试不在仓库普通 CI 中执行**；仓库的 test_language_route.py 只检查资料入口与语言路由。

运行（在本目录父级执行）：
    python tests/test_device.py -v
"""

import sys
import unittest
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conv_dsl import (  # noqa: E402
    _PROGRAMS,
    _normalize,
    _search,
    _target,
    close,
    conv1d,
    conv2d,
    conv3d,
)

ATOL = 1e-3
RTOL = 1e-3

_OPS = {1: conv1d, 2: conv2d, 3: conv3d}

QUICK_CASES = (
    # (name, rank, shape, weight_shape, bias, stride, padding, dilation)
    ("1d_base", 1, (1, 4, 9), (6, 4, 3), True, 1, 1, 1),
    ("1d_stride_dil", 1, (2, 8, 20), (12, 8, 3), False, 2, 1, 2),
    ("2d_no_bias_k_chunks", 2, (1, 32, 5, 18), (16, 32, 3, 3), False, 1, 0, 1),
    ("2d_bias_asym", 2, (2, 8, 7, 9), (8, 8, 3, 3), True, (2, 1), (1, 0, 2, 1), (1, 2)),
    ("3d_base", 3, (1, 3, 4, 5, 6), (8, 3, 2, 3, 3), True, 1, 1, 1),
    (
        "3d_stride_dil_pad",
        3,
        (1, 8, 6, 7, 8),
        (6, 8, 2, 3, 3),
        True,
        (2, 1, 1),
        (1, 0, 1, 1, 0, 1),
        (2, 1, 1),
    ),
)

MATRIX_CASES = (
    # (name, rank, shape, weight_shape, bias, stride, padding, dilation)
    ("d1_base", 1, (1, 4, 39), (6, 4, 3), False, 1, 1, 1),
    ("d1_stride_dil", 1, (2, 8, 40), (12, 8, 5), True, 3, 2, 2),
    ("base_3x3", 2, (1, 16, 5, 18), (16, 16, 3, 3), False, 1, 0, 1),
    ("k_chunks_ci32", 2, (1, 32, 5, 18), (16, 32, 3, 3), False, 1, 0, 1),
    (
        "bias_s2_d2_asym",
        2,
        (2, 8, 7, 9),
        (8, 8, 3, 3),
        True,
        (2, 1),
        (1, 0, 2, 1),
        (1, 2),
    ),
    ("tails_co20", 2, (1, 5, 9, 100), (20, 5, 3, 3), True, 1, 1, 1),
    ("ci_lt_c0_co7", 2, (1, 3, 17, 17), (7, 3, 3, 3), False, 1, 1, 1),
    ("pointwise_1x1", 2, (1, 17, 13, 13), (9, 17, 1, 1), True, 1, 0, 1),
    ("k7", 2, (1, 64, 7, 7), (32, 64, 7, 7), False, 1, 3, 1),
    ("k11", 2, (1, 64, 33, 33), (64, 64, 11, 11), False, 1, 5, 1),
    ("wide_960", 2, (1, 32, 7, 960), (32, 32, 3, 3), True, 1, 1, 1),
    ("wide_192", 2, (1, 32, 56, 192), (32, 32, 3, 3), False, 1, 1, 1),
    ("resident_mid", 2, (1, 128, 14, 14), (64, 128, 3, 3), True, 1, 1, 1),
    ("big_ci_512", 2, (1, 512, 14, 14), (128, 512, 3, 3), False, 1, 1, 1),
    ("stream_ci1536", 2, (1, 1536, 6, 6), (32, 1536, 3, 3), False, 1, 1, 1),
    # 流式权重 + Ci 非整除：ct=112、尾 32 通道，覆盖"K 尾块残留上一步权重"回归
    ("stream_ci_tail", 2, (1, 2048, 6, 6), (64, 2048, 3, 3), True, 1, 1, 1),
    ("minimal_1x1x1", 2, (1, 16, 1, 1), (16, 16, 1, 1), False, 1, 0, 1),
    ("height1_1x33", 2, (1, 16, 1, 33), (16, 16, 1, 5), True, 1, (0, 0, 2, 2), 1),
    ("out_1x1", 2, (1, 8, 3, 3), (8, 8, 3, 3), False, 1, 0, 1),
    ("batch3_odd", 2, (3, 12, 11, 11), (13, 12, 3, 3), True, 1, (1, 0, 2, 1), 1),
    (
        "stride_23_dil21",
        2,
        (1, 64, 31, 29),
        (33, 64, 5, 5),
        False,
        (2, 3),
        (2, 1, 1, 0),
        (2, 1),
    ),
    ("tall_narrow", 2, (1, 7, 40, 5), (9, 7, 2, 4), True, 1, (1, 0, 3, 0), (3, 2)),
    ("stride3", 2, (2, 16, 24, 24), (16, 16, 3, 3), False, 3, 0, 1),
    ("chan_heavy", 2, (1, 256, 7, 7), (64, 256, 3, 3), True, 1, 1, 1),
    ("d3_base", 3, (1, 3, 4, 5, 6), (8, 3, 2, 3, 3), True, 1, 1, 1),
    (
        "d3_stride_dil_pad",
        3,
        (1, 8, 6, 7, 8),
        (6, 8, 2, 3, 3),
        True,
        (2, 1, 1),
        (1, 0, 1, 1, 0, 1),
        (2, 1, 1),
    ),
    ("d3_ci_tail", 3, (1, 5, 4, 4, 4), (7, 5, 1, 3, 3), False, 1, 0, 1),
    ("d3_kd3", 3, (1, 4, 4, 5, 5), (8, 4, 3, 3, 3), False, 1, 1, 1),
    ("d3_od1", 3, (1, 16, 1, 6, 6), (16, 16, 1, 3, 3), True, 1, (0, 1, 1), 1),
    ("d3_chan_heavy", 3, (1, 256, 3, 7, 7), (32, 256, 1, 3, 3), False, 1, (0, 1, 1), 1),
    ("d3_wide", 3, (1, 8, 3, 5, 256), (8, 8, 1, 3, 3), True, 1, (0, 1, 1), 1),
)


def _pairs(rank, padding):
    if isinstance(padding, int):
        values = (padding,) * rank
    else:
        values = tuple(padding)
    if len(values) == rank:
        return [(v, v) for v in values]
    return [(values[2 * i], values[2 * i + 1]) for i in range(rank)]


def _golden(x, weight, bias, rank, stride, padding, dilation):
    strides = (stride,) * rank if isinstance(stride, int) else tuple(stride)
    dilations = (dilation,) * rank if isinstance(dilation, int) else tuple(dilation)
    pairs = _pairs(rank, padding)
    pad_args = tuple(v for pair in reversed(pairs) for v in pair)  # F.pad 逆轴序
    padded = torch.nn.functional.pad(x.double(), pad_args)
    functional = {
        1: torch.nn.functional.conv1d,
        2: torch.nn.functional.conv2d,
        3: torch.nn.functional.conv3d,
    }[rank]
    return functional(
        padded,
        weight.double(),
        None if bias is None else bias.double(),
        stride=strides,
        padding=0,
        dilation=dilations,
    )


def _inputs(shape, weight_shape, has_bias, seed):
    torch.manual_seed(seed)
    x = (torch.rand(shape) - 0.5).to(device="npu", dtype=torch.float16)
    weight = (torch.rand(weight_shape) - 0.5).to(device="npu", dtype=torch.float16)
    bias = None
    if has_bias:
        bias = (torch.rand(weight_shape[0]) * 0.25).to(
            device="npu", dtype=torch.float16
        )
    return x, weight, bias


@unittest.skipUnless(
    torch_npu is not None and torch.npu.is_available(), "NPU device required"
)
class ConvDeviceTests(unittest.TestCase):
    """快速集：三条 rank 各两例 + 缓存复用 + 空输出零编译。"""

    @classmethod
    def setUpClass(cls):
        close()

    @classmethod
    def tearDownClass(cls):
        close()

    def _run_case(
        self,
        name,
        rank,
        shape,
        weight_shape,
        has_bias,
        stride,
        padding,
        dilation,
        seed=1545,
    ):
        x, weight, bias = _inputs(shape, weight_shape, has_bias, seed)
        actual = _OPS[rank](
            x, weight, bias, stride=stride, padding=padding, dilation=dilation
        )
        expected = _golden(
            x.cpu(),
            weight.cpu(),
            None if bias is None else bias.cpu(),
            rank,
            stride,
            padding,
            dilation,
        )
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), name)
        self.assertEqual(actual.dtype, torch.float16, name)
        delta = (actual.cpu().double() - expected).abs()
        limit = ATOL + RTOL * expected.abs()
        self.assertTrue(
            torch.isfinite(actual.cpu()).all(), f"{name}: non-finite output"
        )
        self.assertTrue(bool((delta <= limit).all()), f"{name}: tolerance exceeded")
        print(
            f"\n{name:22s} out={tuple(expected.shape)} maxdiff={float(delta.max()):.2e}"
        )

    def test_1d(self):
        self._run_case(*QUICK_CASES[0])
        self._run_case(*QUICK_CASES[1])

    def test_2d(self):
        self._run_case(*QUICK_CASES[2])
        self._run_case(*QUICK_CASES[3])

    def test_3d(self):
        self._run_case(*QUICK_CASES[4])
        self._run_case(*QUICK_CASES[5])

    def test_program_cache_reuse(self):
        self._run_case(*QUICK_CASES[0])
        before = len(_PROGRAMS)
        self._run_case(*QUICK_CASES[0])
        self.assertEqual(
            len(_PROGRAMS), before, "identical spec must reuse the compiled program"
        )

    def test_empty_output_skips_compile(self):
        close()
        x = torch.zeros((1, 4, 1, 2, 2), device="npu", dtype=torch.float16)
        weight = torch.zeros((4, 4, 3, 3, 3), device="npu", dtype=torch.float16)
        output = conv3d(x, weight)
        self.assertEqual(tuple(output.shape), (1, 4, 0, 0, 0))
        self.assertEqual(output.numel(), 0)
        self.assertEqual(len(_PROGRAMS), 0, "empty output must not compile or launch")


@unittest.skipUnless(
    torch_npu is not None and torch.npu.is_available(), "NPU device required"
)
class ConvMatrixTests(unittest.TestCase):
    """31 规格泛化矩阵：逐例打印实际 tiling 选择并核对数值。"""

    @classmethod
    def setUpClass(cls):
        close()

    @classmethod
    def tearDownClass(cls):
        close()

    def test_matrix(self):
        arch, cores, capacities, _ = _target(
            torch.zeros((1, 1, 1), device="npu", dtype=torch.float16)
        )
        self.assertIn(arch, ("dav-3510", "dav3510"))
        for index, (
            name,
            rank,
            shape,
            weight_shape,
            has_bias,
            stride,
            padding,
            dilation,
        ) in enumerate(MATRIX_CASES):
            with self.subTest(case=name):
                x, weight, bias = _inputs(shape, weight_shape, has_bias, 1545 + index)
                problem = _normalize(x, weight, bias, stride, padding, dilation)
                config = _search(problem, capacities, cores)
                actual = _OPS[rank](
                    x, weight, bias, stride=stride, padding=padding, dilation=dilation
                )
                expected = _golden(
                    x.cpu(),
                    weight.cpu(),
                    None if bias is None else bias.cpu(),
                    rank,
                    stride,
                    padding,
                    dilation,
                )
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                delta = (actual.cpu().double() - expected).abs()
                limit = ATOL + RTOL * expected.abs()
                self.assertTrue(
                    torch.isfinite(actual.cpu()).all(), f"{name}: non-finite output"
                )
                self.assertTrue(
                    bool((delta <= limit).all()), f"{name}: tolerance exceeded"
                )
                print(
                    f"\n{name:18s} out={tuple(expected.shape)} tile={config.tile} "
                    f"resident={config.resident} mrows={config.mrows} nsubs={config.nsubs} "
                    f"depth={config.pipe_depth} tasks={config.tasks} "
                    f"maxdiff={float(delta.max()):.2e}"
                )

    def test_empty_output_skips_compile(self):
        close()
        x = torch.zeros((1, 8, 2, 2), device="npu", dtype=torch.float16)
        weight = torch.zeros((8, 8, 3, 3), device="npu", dtype=torch.float16)
        output = conv2d(x, weight)
        self.assertEqual(tuple(output.shape), (1, 8, 0, 0))
        self.assertEqual(len(_PROGRAMS), 0)


if __name__ == "__main__":
    unittest.main()
