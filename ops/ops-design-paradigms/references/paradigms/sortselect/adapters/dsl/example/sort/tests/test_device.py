# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""真实 DSL 编译及 NPU 检查；缺少 SDK 或设备视为失败。"""

import math
import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

from entry import (
    PreparedSort,
    build_program,
    two_stage_batch,
    build_transpose_program,
    SUPPORTED_DTYPES,
)
from sort_common import RADIX_BINS, DEFAULT_RADIX_TILE, SIMT_SCRATCH_BYTES, radix_digit
from sort_host import (
    ROUTE_INSERTION,
    ROUTE_RADIX_ONE_CORE,
    ROUTE_RADIX_MORE_CORE,
    ROUTE_TWO_STAGE_INVERSE,
    ROUTE_TWO_STAGE_GROUPED,
)


@cb.kernel
class ExtractDigits:
    def __call__(self, keys: cb.Tensor, digits: cb.Tensor, count: cb.dtypes.int64):
        for byte in cb.range_constexpr(8):
            with cb.vf(mode="simt", thread=256):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                for i in range(thread, count, 256):
                    digits[i * 8 + byte] = cb.dtypes.uint8(
                        radix_digit(keys[i], cb.dtypes.uint64, byte)
                    )


@cb.host
def extract_digits(keys: cb.Tensor, digits: cb.Tensor, count: cb.dtypes.int64):
    ExtractDigits()[1](keys, digits, count)


def golden(cpu, descending):
    # 使用 Python 整数保留 uint64 精度，兼容 CPU Sort 缺少该类型的环境。
    rows = cpu.reshape(-1, cpu.shape[-1]).tolist()
    # Python sorted 稳定，等值元素按原轴下标的输入次序保留。
    ids = []
    for row in rows:
        ids.append(sorted(range(len(row)), key=row.__getitem__, reverse=descending))
    values = [[row[i] for i in order] for row, order in zip(rows, ids)]
    return (
        torch.tensor(values, dtype=cpu.dtype).reshape(cpu.shape),
        torch.tensor(ids, dtype=torch.int64).reshape(cpu.shape),
    )


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable")

    def assert_result(self, cpu, values, indices, descending, dim=-1):
        normalized = cpu.movedim(dim, -1).contiguous()
        expected_values, expected_ids = golden(normalized, descending)
        expected_values = expected_values.movedim(-1, dim).contiguous()
        expected_ids = expected_ids.movedim(-1, dim).contiguous()
        self.assertTrue(
            torch.equal(
                values.cpu().view(torch.uint8), expected_values.view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(indices.cpu(), expected_ids))

    def check_direct_schedule(self, route, dtype, rows, length, width, groups, inner=1):
        for descending in (False, True):
            with self.subTest(route=route, inner=inner, descending=descending):
                cpu = (torch.arange(rows * length) * 37) % 127 - 63
                if dtype == torch.uint64:
                    cpu = cpu * (2**32 + 1)
                shape = (rows, length) if inner == 1 else (rows // inner, length, inner)
                dim = -1 if inner == 1 else 1
                cpu = cpu.to(dtype).reshape(shape)
                x = cpu.npu()
                values = torch.empty_like(x)
                indices = torch.empty(x.shape, dtype=torch.int64, device=x.device)
                active_rows = min(rows, groups) if groups > 0 else rows
                count = active_rows * length if route == ROUTE_RADIX_MORE_CORE else 1
                bins = (
                    active_rows * ((length + width - 1) // width) * RADIX_BINS
                    if route == ROUTE_RADIX_MORE_CORE
                    else 1
                )
                key_dtype = (
                    dtype
                    if route == ROUTE_RADIX_MORE_CORE
                    else torch.uint64
                    if dtype in (torch.int64, torch.uint64)
                    else torch.uint32
                )
                workspace = [
                    torch.empty(n, dtype=d, device=x.device)
                    for n, d in [
                        (count, key_dtype),
                        (count, torch.int64),
                        (count, key_dtype),
                        (count, torch.int64),
                        (bins, torch.int64),
                        (
                            active_rows
                            * (((length + width - 1) // width) + 1)
                            * RADIX_BINS
                            * cpu.element_size()
                            if route == ROUTE_RADIX_MORE_CORE
                            else 1,
                            torch.int64,
                        ),
                    ]
                ]
                program = build_program(dtype, route, width, groups, descending)
                try:
                    info = cb.get_platform_info(
                        stream=torch.npu.current_stream(x.device)
                    )
                    if not info.available or info.vector_core_num <= 0:
                        raise RuntimeError("vector core capacity unavailable")
                    blocks = min(2, int(info.vector_core_num))
                    for step in range(2):
                        if step:
                            # 部分 CPU 构建不支持 uint64 flip；通过 int64 视图反转以保留位模式。
                            if dtype in (torch.uint16, torch.uint32, torch.uint64):
                                signed = {
                                    torch.uint16: torch.int16,
                                    torch.uint32: torch.int32,
                                    torch.uint64: torch.int64,
                                }[dtype]
                                cpu = cpu.view(signed).flip(dim).view(dtype)
                            else:
                                cpu = cpu.flip(dim)
                            cpu = cpu.contiguous()
                            x.copy_(cpu)
                        program(
                            x.reshape(-1),
                            values.reshape(-1),
                            indices.reshape(-1),
                            *workspace,
                            rows,
                            length,
                            inner,
                            blocks,
                        )
                        torch.npu.synchronize()
                        self.assert_result(cpu, values, indices, descending, dim)
                        self.assertTrue(
                            torch.equal(
                                x.cpu().view(torch.uint8), cpu.view(torch.uint8)
                            )
                        )
                finally:
                    torch.npu.synchronize()
                    program.close()

    def test_unknown_route_rejected_before_compile(self):
        with mock.patch("entry.cb.compile") as compile_program:
            for route in (-1, 0, 6):
                with (
                    self.subTest(route=route),
                    self.assertRaisesRegex(ValueError, "unknown Sort route"),
                ):
                    build_program(torch.float32, route, 1024, 1, False)
            compile_program.assert_not_called()
        # 绕过公开路由校验，直接编译未知编号也必须失败，而不是生成空调用。
        original_compile = cb.compile

        def compile_unknown_route(*args):
            args = list(args)
            args[-4] = 6
            return original_compile(*args)

        with mock.patch("entry.cb.compile", side_effect=compile_unknown_route):
            with self.assertRaisesRegex(ValueError, "unknown Sort route"):
                build_program(torch.float32, ROUTE_INSERTION, 8, 1, False)

    def test_single_core_rejects_multiple_rows(self):
        # 多行需要行内恢复，不能误用仅分配单行 UB 的模板；两个编译入口都应拒绝。
        with mock.patch("entry.cb.compile") as compile_program:
            for groups in (0, 2):
                with (
                    self.subTest(groups=groups),
                    self.assertRaisesRegex(ValueError, "groups == 1"),
                ):
                    build_program(
                        torch.float32, ROUTE_RADIX_ONE_CORE, 256, groups, False
                    )
            compile_program.assert_not_called()
        original_compile = cb.compile

        def compile_multiple_rows(*args):
            args = list(args)
            args[-2] = 2
            return original_compile(*args)

        with mock.patch("entry.cb.compile", side_effect=compile_multiple_rows):
            with self.assertRaisesRegex(ValueError, "groups == 1"):
                build_program(torch.float32, ROUTE_RADIX_ONE_CORE, 256, 1, False)

    def test_ub_capacity_rejected_before_compile(self):
        # 注入比最低预算少一字节的 UB，确认明确拒绝而非除零或交给设备编译失败。
        for dtype, length, usable, message in (
            (torch.float32, 1, 8 * (4 + 4 + 4) - 1, "insertion"),
            (torch.float32, 65536, DEFAULT_RADIX_TILE * 14 + 4096 - 1, "radix"),
            (torch.int64, 65536, DEFAULT_RADIX_TILE * 18 + 4096 - 1, "radix"),
        ):
            x = torch.empty((1, length), dtype=dtype, device="npu")
            with (
                self.subTest(length=length),
                mock.patch(
                    "entry.cb.get_mem_size", return_value=SIMT_SCRATCH_BYTES + usable
                ),
                mock.patch("entry.build_program") as compile_program,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "insufficient UB for " + message
                ):
                    PreparedSort(x)
                compile_program.assert_not_called()
        # 容量恰好满足默认块时必须放行，不能将两波任务的性能偏好当成合法性条件。
        for dtype, tile_bytes in ((torch.float32, 14), (torch.int64, 18)):
            x = torch.empty((1, 65536), dtype=dtype, device="npu")
            with (
                mock.patch(
                    "entry.cb.get_mem_size",
                    return_value=SIMT_SCRATCH_BYTES
                    + DEFAULT_RADIX_TILE * tile_bytes
                    + 4096,
                ),
                mock.patch("entry.build_program", return_value=None) as compile_program,
            ):
                prepared = PreparedSort(x)
                try:
                    self.assertEqual(prepared.route, "radix_more_core")
                    self.assertEqual(
                        compile_program.call_args.args[2], DEFAULT_RADIX_TILE
                    )
                finally:
                    prepared.close()

    def test_all_schedules(self):
        # 真实编译各路线，覆盖尾块、多个 UB 批次和升降序。
        for case in [
            (ROUTE_INSERTION, torch.float32, 19, 8, 8, 16),
            (ROUTE_RADIX_ONE_CORE, torch.float32, 3, 513, 768, 1),
            (ROUTE_RADIX_ONE_CORE, torch.int8, 3, 513, 768, 1),
            (ROUTE_RADIX_ONE_CORE, torch.uint16, 3, 513, 768, 1),
            (ROUTE_TWO_STAGE_INVERSE, torch.float32, 149, 17, 1280, 74),
            (ROUTE_TWO_STAGE_GROUPED, torch.int32, 51, 65, 1792, 25),
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 2, 16385, 1024, 0),
            # 覆盖多个 tile、单元素尾块与宽计数 lookback 直调。
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 1, 1025, 1024, 0),
            (ROUTE_RADIX_MORE_CORE, torch.float32, 1, 8193, 1024, 0),
        ]:
            self.check_direct_schedule(*case)

    def test_strided_schedules(self):
        # 仅测试低层入口；公开包装始终规范化为 inner=1。
        for case in [
            (ROUTE_INSERTION, torch.float32, 57, 8, 8, 16),
            (ROUTE_RADIX_ONE_CORE, torch.float32, 9, 513, 768, 1),
            (ROUTE_RADIX_ONE_CORE, torch.int8, 9, 513, 768, 1),
            (ROUTE_RADIX_ONE_CORE, torch.float16, 9, 513, 768, 1),
            (ROUTE_RADIX_ONE_CORE, torch.uint16, 9, 513, 768, 1),
            (ROUTE_TWO_STAGE_INVERSE, torch.float32, 447, 17, 1280, 74),
            (ROUTE_TWO_STAGE_GROUPED, torch.int32, 153, 65, 1792, 25),
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 6, 16385, 1024, 0),
            # 跨越原布局 outer 边界的批次、尾批一行及 Prefix 分支切换。
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 9, 1025, 1024, 2),
        ]:
            self.check_direct_schedule(*case, inner=3)

    def test_local_lsd_narrow_pass_counts(self):
        # two-stage 仍用 uint32 载体：分别覆盖一轮/两轮 Stage1，
        # 行号分组须从 ping 重新开始，不能继承 Stage1 奇偶侧。
        for dtype in (torch.uint8, torch.uint16):
            self.check_direct_schedule(ROUTE_TWO_STAGE_GROUPED, dtype, 7, 65, 512, 5)

    def test_local_lsd_float_payloads(self):
        # 尾块和重复特殊值；按位检查 NaN payload、零符号与稳定原索引。
        bits = torch.tensor(
            [
                0,
                -2147483648,
                0x7F800000,
                -8388608,
                0x7FC00001,
                0x7FC00002,
                -4194303,
                0x3F800000,
            ],
            dtype=torch.int32,
        ).repeat(65)[:513]
        cpu = bits.view(torch.float32).reshape(1, -1)
        for descending in (False, True):
            expected_values, expected_ids = torch.sort(
                cpu, stable=True, descending=descending
            )
            prepared = PreparedSort(cpu.npu(), descending=descending)
            try:
                self.assertEqual(prepared.route, "radix_one_core")
                values, indices = prepared.run()
                torch.npu.synchronize()
                self.assertTrue(torch.equal(indices.cpu(), expected_ids))
                self.assertTrue(
                    torch.equal(
                        values.cpu().view(torch.int32),
                        expected_values.view(torch.int32),
                    )
                )
            finally:
                torch.npu.synchronize()
                prepared.close()

    def test_dma_key32_extrema(self):
        # DMA 原位构键必须按无符号位模式处理符号位和最大值。
        words = [0, 1, 2**31 - 1, 2**31, 2**31 + 1, 2**32 - 2, 2**32 - 1]
        for dtype in (torch.int32, torch.uint32):
            numbers = [
                word - 2**32 if dtype == torch.int32 and word >= 2**31 else word
                for word in words
            ]
            cpu = torch.tensor((numbers * 147)[:1025], dtype=dtype).reshape(1, -1)
            for descending in (False, True):
                prepared = PreparedSort(cpu.npu(), descending=descending)
                try:
                    self.assertEqual(prepared.route, "radix_one_core")
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    self.assert_result(cpu, values, indices, descending)
                finally:
                    torch.npu.synchronize()
                    prepared.close()

    def test_public_policy_boundaries(self):
        info = cb.get_platform_info(stream=torch.npu.current_stream())
        cores = int(info.vector_core_num)
        cases = []
        # 依次覆盖插入门槛下方、命中插入门槛、命中首个两阶段档位。
        # 无符号类型共用门槛，但单独编译对应类型的 Kernel。
        for dtypes, length, insertion_gate, two_stage_gate in [
            ((torch.int64, torch.uint64), 15, 4, 8),
            ((torch.int32, torch.uint32), 11, 4, 8),
            ((torch.int16, torch.uint16), 7, 4, 8),
            ((torch.int8, torch.uint8), 3, 2, 8),
        ]:
            for dtype in dtypes:
                for per_core, route in [
                    (insertion_gate - 1, "radix_one_core"),
                    (insertion_gate, "insertion"),
                    (two_stage_gate, "two_stage"),
                ]:
                    cases.append((dtype, length, (per_core - 1) * cores + 1, route))
        # 浮点档位保留语言差异：DSL 尚无 MergeSort 回退。
        for dtype, length, rows, route in [
            (torch.float32, 8, 1, "insertion"),
            (torch.float32, 25, cores * 64, "two_stage"),
            (torch.float32, 256, cores * 64, "two_stage"),
            (torch.float16, 33, cores * 64, "two_stage"),
            (torch.bfloat16, 54, cores * 64, "two_stage"),
            (torch.float16, 55, cores * 64, "two_stage"),
        ]:
            cases.append((dtype, length, rows, route))
        for dtype, length, rows, route in cases:
            cpu = (
                ((torch.arange(rows * length) * 37) % 127)
                .to(dtype)
                .reshape(rows, length)
                .view(dtype)
            )
            for descending in (False, True):
                with self.subTest(
                    dtype=dtype, length=length, rows=rows, descending=descending
                ):
                    x = cpu.npu()
                    prepared = PreparedSort(x, descending=descending)
                    try:
                        self.assertEqual(prepared.route, route)
                        values, indices = prepared.run()
                        torch.npu.synchronize()
                        self.assert_result(cpu, values, indices, descending)
                        self.assertTrue(
                            torch.equal(
                                x.cpu().view(torch.uint8), cpu.view(torch.uint8)
                            )
                        )
                    finally:
                        torch.npu.synchronize()
                        prepared.close()

    def test_more_core_single_launch(self):
        """工作区按全部行分配，一次 run 只 launch 一次；重新绑定输入不复用旧结果。"""
        info = cb.get_platform_info(stream=torch.npu.current_stream())
        cores = int(info.vector_core_num)
        length = 16385
        allocations = []
        for rows in (cores + 1, 2 * cores + 3):
            cpu = (
                ((torch.arange(rows * length) * 37) % 127 - 63)
                .float()
                .reshape(rows, length)
            )
            x = cpu.npu()
            prepared = PreparedSort(x)
            try:
                self.assertEqual(prepared.route, "radix_more_core")
                self.assertEqual(prepared.active_rows, rows)
                allocations.append(prepared.workspace_bytes // rows)
                self.assertEqual(prepared.workspace[0].numel(), rows * length)
                self.assertEqual(prepared.workspace[0].dtype, cpu.dtype)
                self.assertEqual(prepared.workspace[2].dtype, cpu.dtype)
                # 公开多核入口使用窄中间区，最终输出继续提供 int64 原轴坐标。
                for slot in (1, 3, 4, 5):
                    self.assertEqual(prepared.workspace[slot].dtype, torch.int32)
                self.assertEqual(prepared.indices.dtype, torch.int64)
                # 以 hist 的有效容量核算实际选择的 tile 数。
                actual_tiles = prepared.workspace[4].numel() // (rows * RADIX_BINS)
                self.assertEqual(
                    prepared.workspace_bytes,
                    rows * length * 16
                    + 4 * rows * RADIX_BINS * (actual_tiles + 4 * (actual_tiles + 1)),
                )
                self.assertEqual(len(prepared.workspace), 6)
                self.assertEqual(
                    prepared.workspace[5].numel(),
                    rows * 4 * (actual_tiles + 1) * RADIX_BINS,
                )
                self.assertEqual(prepared.kernel_count, 1)
                with mock.patch.object(
                    prepared, "program", wraps=prepared.program
                ) as launch:
                    for step in range(2):
                        if step:
                            cpu = cpu.flip(-1).contiguous()
                            x.copy_(cpu)
                        values, indices = prepared.run()
                        torch.npu.synchronize()
                        self.assertEqual(launch.call_count, step + 1)
                        self.assert_result(cpu, values, indices, False)
                        self.assertTrue(torch.equal(x.cpu(), cpu))
            finally:
                torch.npu.synchronize()
                prepared.close()
        self.assertEqual(allocations[0], allocations[1])

    def test_more_core_digit_preparation_boundaries(self):
        """融合统计和首轮 UB 写回覆盖 256 项边界、尾批及同地址重跑。"""
        for dtype in (torch.int8, torch.uint8, torch.float32, torch.uint64):
            for tail in (255, 256, 257):
                self.check_direct_schedule(
                    ROUTE_RADIX_MORE_CORE, dtype, 3, 1024 + tail, 1024, 2
                )
        # 非连续低层路径继续从 GM 取原值，覆盖首轮即末轮和多轮排序。
        for dtype in (torch.uint8, torch.float32, torch.uint64):
            self.check_direct_schedule(
                ROUTE_RADIX_MORE_CORE, dtype, 9, 1025, 1024, 2, inner=3
            )

    def test_more_core_float_payload_reuse(self):
        self.check_float_payload_reuse(3, 32769, "radix_more_core")

    def test_insertion_dma_tail_payload_reuse(self):
        self.check_float_payload_reuse(2051, 7, "insertion")

    def check_float_payload_reuse(self, rows, length, route):
        # 新 SIMD 构键必须将零/NaN 归为同键，同时保留原值位模式和稳定索引。
        patterns = [
            (
                torch.float16,
                torch.int16,
                [
                    0,
                    0x8000,
                    0x7C00,
                    0xFC00,
                    0x7E01,
                    0xFE02,
                    0x7C01,
                    0x7BFF,
                    0xFBFF,
                    1,
                    0x8001,
                    0x3C00,
                    0xBC00,
                ],
            ),
            (
                torch.bfloat16,
                torch.int16,
                [
                    0,
                    0x8000,
                    0x7F80,
                    0xFF80,
                    0x7FC1,
                    0xFFC2,
                    0x7F81,
                    0x7F7F,
                    0xFF7F,
                    1,
                    0x8001,
                    0x3F80,
                    0xBF80,
                ],
            ),
            (
                torch.float32,
                torch.int32,
                [
                    0,
                    0x80000000,
                    0x7F800000,
                    0xFF800000,
                    0x7FC00001,
                    0xFFC00002,
                    0x7F800001,
                    0x7F7FFFFF,
                    0xFF7FFFFF,
                    1,
                    0x80000001,
                    0x3F800000,
                    0xBF800000,
                ],
            ),
        ]
        for dtype, carrier, words in patterns:
            bits = torch.iinfo(carrier).bits
            signed = [w if w < 2 ** (bits - 1) else w - 2**bits for w in words]
            pattern = torch.tensor(signed, dtype=carrier)
            original = (
                pattern.repeat((rows * length + len(words) - 1) // len(words))[
                    : rows * length
                ]
                .reshape(rows, length)
                .view(dtype)
            )
            for descending in (False, True):
                with self.subTest(dtype=dtype, descending=descending):
                    cpu = original.clone()
                    x = cpu.npu()
                    prepared = PreparedSort(x, descending=descending)
                    try:
                        self.assertEqual(prepared.route, route)
                        for step in range(2):
                            if step:
                                cpu = (
                                    cpu.view(carrier).flip(-1).contiguous().view(dtype)
                                )
                                x.copy_(cpu)
                            expected_ids = torch.argsort(
                                cpu.float(), dim=-1, descending=descending, stable=True
                            )
                            # 浮点 CPU gather 会改变部分窄 NaN 的位模式，按整数视图收集。
                            expected_bits = torch.gather(
                                cpu.view(carrier), -1, expected_ids
                            )
                            values, indices = prepared.run()
                            torch.npu.synchronize()
                            self.assertTrue(
                                torch.equal(values.cpu().view(carrier), expected_bits)
                            )
                            self.assertTrue(torch.equal(indices.cpu(), expected_ids))
                            self.assertTrue(
                                torch.equal(x.cpu().view(carrier), cpu.view(carrier))
                            )
                    finally:
                        torch.npu.synchronize()
                        prepared.close()

    def test_narrow_local_capacity(self):
        usable = cb.get_mem_size("ub") - SIMT_SCRATCH_BYTES
        for dtype, per_element in ((torch.uint8, 6), (torch.uint16, 12)):
            maximum = min(65280, (usable - 1024) // per_element // 256 * 256)
            for length, expected_route in (
                (maximum, "radix_one_core"),
                (maximum + 1, "radix_more_core"),
            ):
                with self.subTest(dtype=dtype, length=length):
                    cpu = (
                        ((torch.arange(length) * 37) % 127).to(dtype).reshape(1, length)
                    )
                    prepared = PreparedSort(cpu.npu(), descending=True)
                    try:
                        self.assertEqual(prepared.route, expected_route)
                        values, indices = prepared.run()
                        torch.npu.synchronize()
                        self.assert_result(cpu, values, indices, True)
                    finally:
                        torch.npu.synchronize()
                        prepared.close()

    def test_bounded_workspace_reuse(self):
        cores = int(
            cb.get_platform_info(stream=torch.npu.current_stream()).vector_core_num
        )
        allocations = []
        for rows in (cores + 1, 2 * cores + 3):
            length = 16384
            cpu = (((torch.arange(rows * length) * 37) % 127) - 63).reshape(
                rows, length
            )
            x = cpu.npu()
            # 压低软件目标，验证最小均衡批次；硬件 UB 查询不变。
            with mock.patch("entry.RADIX_WORKSPACE_TARGET_BYTES", 1):
                prepared = PreparedSort(x)
            try:
                self.assertEqual(prepared.route, "radix_more_core")
                self.assertLess(prepared.active_rows, rows)
                self.assertEqual(prepared.kernel_count, 1)
                allocations.append(prepared.workspace_bytes)
                for step in range(2):
                    if step:
                        cpu = cpu.flip(-1).contiguous()
                        x.copy_(cpu)
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    self.assert_result(cpu, values, indices, False)
                    self.assertTrue(torch.equal(x.cpu(), cpu))
            finally:
                torch.npu.synchronize()
                prepared.close()
        self.assertEqual(allocations[0], allocations[1])

    def test_two_stage_row_grouping(self):
        # 255/256/257 行跨过单字节边界；尾批及重复值验证稳定性和奇偶结果缓冲。
        for groups in (255, 256, 257):
            self.check_direct_schedule(
                ROUTE_TWO_STAGE_GROUPED, torch.int32, groups + 2, 3, 1024, groups
            )
        # 公开入口按单字节行号分组，批次受 UB 和行号范围共同约束。
        info = cb.get_platform_info(stream=torch.npu.current_stream())
        cores = int(info.vector_core_num)
        usable = cb.get_mem_size("ub") - SIMT_SCRATCH_BYTES
        length, rows = 255, cores * 64 - 1
        for dtype in (torch.int32, torch.uint64, torch.float32):
            batch = two_stage_batch(dtype, length, rows, cores, usable)
            self.assertGreater(batch, 16)
            self.assertLessEqual(batch, 256)
            cpu = (
                ((torch.arange(rows * length) * 37) % 71)
                .reshape(rows, length)
                .to(dtype)
            )
            for descending in (False, True):
                prepared = PreparedSort(cpu.npu(), descending=descending)
                try:
                    self.assertEqual(prepared.route, "two_stage")
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    self.assert_result(cpu, values, indices, descending)
                finally:
                    torch.npu.synchronize()
                    prepared.close()

    def test_uint64_high_digits(self):
        """逐字节验证跨 32/63 位边界，覆盖高字为负位模式及非零低字。"""
        words = [
            0,
            1,
            2**32 - 1,
            2**32,
            2**32 + 1,
            2**63 - 1,
            2**63,
            2**63 + 1,
            0xFFFFFFFF00000000,
            0xFFFFFFFF00000001,
            0xFFFFFFFFFFFFFFFF,
        ]
        cpu = torch.tensor(words, dtype=torch.uint64)
        output = torch.empty(len(words) * 8, dtype=torch.uint8, device="npu")
        program = cb.compile(
            extract_digits,
            cb.TensorSpec((len(words),), cb.dtypes.uint64),
            cb.TensorSpec((len(words) * 8,), cb.dtypes.uint8),
            cb.dtypes.int64,
        )
        try:
            program(cpu.npu(), output, len(words))
            torch.npu.synchronize()
            expected = torch.tensor(
                [byte for word in words for byte in word.to_bytes(8, "little")],
                dtype=torch.uint8,
            )
            self.assertTrue(torch.equal(output.cpu(), expected))
        finally:
            torch.npu.synchronize()
            program.close()

    def test_int64_uint64_boundaries(self):
        self.check_int64_uint64_boundaries(1025, "radix_one_core")

    def test_more_core_dma_int64_uint64_boundaries(self):
        self.check_int64_uint64_boundaries(8193, "radix_more_core")

    def check_int64_uint64_boundaries(self, length, route):
        words = [
            0,
            1,
            2**32 - 1,
            2**32,
            2**32 + 1,
            2**63 - 1,
            2**63,
            2**63 + 1,
            0xFFFFFFFF00000000,
            0xFFFFFFFF00000001,
            0xFFFFFFFFFFFFFFFF,
        ]
        for dtype in (torch.int64, torch.uint64):
            numbers = [
                word - 2**64 if dtype == torch.int64 and word >= 2**63 else word
                for word in words
            ]
            cpu = torch.tensor(
                (numbers * ((length + len(numbers) - 1) // len(numbers)))[:length],
                dtype=dtype,
            ).reshape(1, -1)
            for descending in (False, True):
                prepared = PreparedSort(cpu.npu(), descending=descending)
                try:
                    self.assertEqual(prepared.route, route)
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    self.assert_result(cpu, values, indices, descending)
                finally:
                    torch.npu.synchronize()
                    prepared.close()

    def test_rectangular_transpose_boundaries(self):
        # NDDMA、紧凑写回、SIMT 回退及长块门槛；同一编译程序复用不同形状。
        shapes = [
            (2047, 17),
            (2048, 17),
            (2049, 17),
            (65535, 16),
            (65536, 16),
            (65537, 16),
            (65537, 17),
            (4095, 7),
            (4096, 8),
            (4097, 9),
            (4097, 16),
            (4097, 17),
            (4097, 33),
            (4097, 64),
            (4097, 65),
        ]
        for dtype in SUPPORTED_DTYPES:
            program = build_transpose_program(dtype)
            try:
                for h, w in shapes:
                    for height, width in ((h, w), (w, h)):
                        with self.subTest(dtype=dtype, height=height, width=width):
                            cpu = (
                                (torch.arange(2 * height * width) % 251)
                                .to(dtype)
                                .reshape(2, height, width)
                            )
                            x = cpu.npu().reshape(-1)
                            output = torch.empty_like(x)
                            program(x, output, 2, height, width, 56)
                            torch.npu.synchronize()
                            expected = cpu.transpose(1, 2).contiguous().reshape(-1)
                            self.assertTrue(
                                torch.equal(
                                    output.cpu().view(torch.uint8),
                                    expected.view(torch.uint8),
                                )
                            )
            finally:
                torch.npu.synchronize()
                program.close()

    def test_transpose_bit_patterns_rebinding(self):
        # 不能用数值相等验证 NaN payload 或 uint64 高半字；原始位模式逐字节比较。
        patterns = {
            torch.float16: torch.tensor(
                [0, 0x8000, 0x7C00, 0xFC00, 0x7C01, 0xFE35, 0x3C00],
                dtype=torch.uint16,
            ),
            torch.bfloat16: torch.tensor(
                [0, 0x8000, 0x7F80, 0xFF80, 0x7F81, 0xFFC5, 0x3F80],
                dtype=torch.uint16,
            ),
            torch.float32: torch.tensor(
                [
                    0,
                    0x80000000,
                    0x7F800000,
                    0xFF800000,
                    0x7F800001,
                    0xFFC12345,
                    0x3F800000,
                ],
                dtype=torch.uint32,
            ),
        }
        wide = torch.tensor(
            [0, 1, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**64 - 1, 0x123456789ABCDEF0],
            dtype=torch.uint64,
        )
        patterns[torch.int64] = wide
        patterns[torch.uint64] = wide
        for dtype, bits in patterns.items():
            pattern = bits.view(dtype)
            with build_transpose_program(dtype) as program:
                # 方块/短行回退、NDDMA 尾块与 512 长块，两个方向及同址输入更新。
                for h, w in ((17, 129), (65, 33), (65537, 9)):
                    for height, width in ((h, w), (w, h)):
                        count = 2 * height * width
                        x = torch.empty(count, dtype=dtype, device="npu")
                        output = torch.empty_like(x)
                        for shift in (0, 1):
                            with self.subTest(
                                dtype=dtype, height=height, width=width, shift=shift
                            ):
                                cpu = (
                                    pattern.roll(shift)
                                    .repeat(
                                        (count + pattern.numel() - 1) // pattern.numel()
                                    )[:count]
                                    .reshape(2, height, width)
                                )
                                x.copy_(cpu.reshape(-1).npu())
                                program(x, output, 2, height, width, 56)
                                torch.npu.synchronize()
                                expected = cpu.transpose(1, 2).contiguous().reshape(-1)
                                self.assertTrue(
                                    torch.equal(
                                        output.cpu().view(torch.uint8),
                                        expected.view(torch.uint8),
                                    )
                                )
                                self.assertTrue(
                                    torch.equal(
                                        x.cpu().view(torch.uint8),
                                        cpu.reshape(-1).view(torch.uint8),
                                    )
                                )

    def test_transpose_wide_binding_contract(self):
        for dtype in (torch.int64, torch.uint64):
            with build_transpose_program(dtype) as program:
                cpu = torch.arange(15).reshape(3, 5).to(dtype)
                x = cpu.npu().reshape(-1)
                output = torch.empty_like(x)
                program(x=x, output=output, outer=1, height=3, width=5, blocks=1)
                torch.npu.synchronize()
                self.assertTrue(torch.equal(output.cpu().reshape(5, 3), cpu.T))
                wrong = torch.empty(15, dtype=torch.int32, device="npu")
                with self.assertRaisesRegex(TypeError, "compiled dtype"):
                    program(wrong, output, 1, 3, 5, 1)
                with self.assertRaisesRegex(TypeError, "compiled dtype"):
                    program(x, wrong, 1, 3, 5, 1)

    def test_non_tail_layout(self):
        for shape in ((3, 65, 5), (2, 127, 129), (2, 128, 129)):
            with self.subTest(shape=shape):
                count = shape[0] * shape[1] * shape[2]
                cpu = ((torch.arange(count) * 37) % 71).reshape(shape).float()
                x = cpu.npu()
                prepared = PreparedSort(x, dim=1)
                try:
                    self.assertTrue(prepared.needs_transpose)
                    self.assertEqual(prepared.kernel_count, 4)
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    expected_values, expected_ids = torch.sort(cpu, dim=1, stable=True)
                    self.assertTrue(torch.equal(values.cpu(), expected_values))
                    self.assertTrue(torch.equal(indices.cpu(), expected_ids))
                    self.assertTrue(torch.equal(x.cpu(), cpu))
                finally:
                    torch.npu.synchronize()
                    prepared.close()

    def test_empty_public_input(self):
        for shape, dim in [((2, 0, 3), 1), ((2, 0, 3), 0), ((0,), -1)]:
            with self.subTest(shape=shape, dim=dim):
                x = torch.empty(shape, device="npu")
                prepared = PreparedSort(x, dim=dim, stable=False)
                try:
                    values, indices = prepared.run()
                    self.assertEqual(tuple(values.shape), shape)
                    self.assertEqual(tuple(indices.shape), shape)
                    self.assertEqual(values.numel(), 0)
                    self.assertEqual(indices.dtype, torch.int64)
                    self.assertEqual(prepared.route, "empty")
                    self.assertIsNone(prepared.program)
                    self.assertEqual(prepared.kernel_count, 0)
                    self.assertEqual(prepared.workspace_bytes, 0)
                finally:
                    prepared.close()

    def test_length_one_public_input(self):
        cores = int(
            cb.get_platform_info(stream=torch.npu.current_stream()).vector_core_num
        )
        patterns = {
            torch.float16: torch.tensor(
                [0, 0x8000, 0x7C01, 0xFE35, 0x7C00, 0xFC00], dtype=torch.uint16
            ).view(torch.float16),
            torch.bfloat16: torch.tensor(
                [0, 0x8000, 0x7F81, 0xFFC5, 0x7F80, 0xFF80], dtype=torch.uint16
            ).view(torch.bfloat16),
            torch.float32: torch.tensor(
                [0, 0x80000000, 0x7F800001, 0xFFC12345, 0x7F800000, 0xFF800000],
                dtype=torch.uint32,
            ).view(torch.float32),
        }
        wide = torch.tensor(
            [0, 2**32, 2**63 - 1, 2**63, 2**64 - 1, 1], dtype=torch.uint64
        )
        patterns[torch.int64] = wide.view(torch.int64)
        patterns[torch.uint64] = wide
        low_concurrency_insertion = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int64,
            torch.uint64,
        }
        for dtype in SUPPORTED_DTYPES:
            pattern = patterns.get(dtype)
            if pattern is None:
                pattern = torch.tensor([0, 1, -1, 127, -128, 42]).to(dtype)
            # 小批量按普通 dtype 门槛选插入或单核；大批量命中两阶段。
            for shape, dim in [((3, 1), -1), ((2, 1, 3), 1), ((cores * 64 + 1, 1), -1)]:
                count = math.prod(shape)
                cpu = pattern.repeat((count + pattern.numel() - 1) // pattern.numel())[
                    :count
                ].reshape(shape)
                expected_route = (
                    "two_stage"
                    if count > 6
                    else "insertion"
                    if dtype in low_concurrency_insertion
                    else "radix_one_core"
                )
                for descending in (False, True):
                    with self.subTest(
                        dtype=dtype, shape=shape, dim=dim, descending=descending
                    ):
                        x = cpu.npu()
                        prepared = PreparedSort(
                            x, dim=dim, descending=descending, stable=False
                        )
                        try:
                            self.assertEqual(prepared.route, expected_route)
                            self.assertFalse(prepared.needs_transpose)
                            self.assertEqual(prepared.kernel_count, 1)
                            values, indices = prepared.run()
                            torch.npu.synchronize()
                            self.assertTrue(
                                torch.equal(
                                    values.cpu().view(torch.uint8),
                                    cpu.view(torch.uint8),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    indices.cpu(), torch.zeros(shape, dtype=torch.int64)
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    x.cpu().view(torch.uint8), cpu.view(torch.uint8)
                                )
                            )
                        finally:
                            torch.npu.synchronize()
                            prepared.close()

    def test_stable_false_reuses_stable_result(self):
        cpu = torch.tensor([[2, 1, 2, 1, 2, 1]], dtype=torch.int32)
        prepared = PreparedSort(cpu.npu(), stable=False)
        try:
            values, indices = prepared.run()
            torch.npu.synchronize()
            self.assert_result(cpu, values, indices, False)
        finally:
            torch.npu.synchronize()
            prepared.close()


if __name__ == "__main__":
    unittest.main()
