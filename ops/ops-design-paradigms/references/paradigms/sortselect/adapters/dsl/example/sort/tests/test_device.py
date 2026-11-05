# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""真实 DSL 编译及 NPU 检查；缺少 SDK 或设备视为失败。"""

import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

from entry import PreparedSort, build_program, two_stage_batch
from sort_common import RADIX_BINS, DEFAULT_RADIX_TILE, SIMT_SCRATCH_BYTES, radix_digit
from sort_host import (
    ROUTE_INSERTION,
    ROUTE_RADIX_ONE_CORE,
    ROUTE_RADIX_MORE_CORE,
    ROUTE_TWO_STAGE_INVERSE,
    ROUTE_TWO_STAGE_PACKED,
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
                count = x.numel() if route == ROUTE_RADIX_MORE_CORE else 1
                bins = (
                    rows * ((length + width - 1) // width) * RADIX_BINS
                    if route == ROUTE_RADIX_MORE_CORE
                    else 1
                )
                key_dtype = (
                    torch.uint64
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
                        (bins, torch.int64),
                        (count, torch.uint32),
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
                            if dtype == torch.uint64:
                                cpu = cpu.view(torch.int64).flip(dim).view(dtype)
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
        for length, usable, message in (
            (1, 8 * (4 + 4 + 4) - 1, "insertion"),
            (65536, DEFAULT_RADIX_TILE * 7 + 1024 - 1, "radix"),
        ):
            x = torch.empty((1, length), dtype=torch.float32, device="npu")
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
        x = torch.empty((1, 65536), dtype=torch.float32, device="npu")
        with (
            mock.patch(
                "entry.cb.get_mem_size",
                return_value=SIMT_SCRATCH_BYTES + DEFAULT_RADIX_TILE * 7 + 1024,
            ),
            mock.patch("entry.build_program", return_value=None) as compile_program,
        ):
            prepared = PreparedSort(x)
            try:
                self.assertEqual(prepared.route, "radix_more_core")
                self.assertEqual(compile_program.call_args.args[2], DEFAULT_RADIX_TILE)
            finally:
                prepared.close()

    def test_all_schedules(self):
        # 真实编译各路线，覆盖尾块、多个 UB 批次和升降序。
        for case in [
            (ROUTE_INSERTION, torch.float32, 19, 8, 8, 16),
            (ROUTE_RADIX_ONE_CORE, torch.float32, 3, 513, 768, 1),
            (ROUTE_TWO_STAGE_INVERSE, torch.float32, 149, 17, 1280, 74),
            (ROUTE_TWO_STAGE_PACKED, torch.int32, 51, 65, 1792, 25),
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 2, 16385, 1024, 0),
        ]:
            self.check_direct_schedule(*case)

    def test_strided_schedules(self):
        # 仅测试低层入口；公开包装始终规范化为 inner=1。
        for case in [
            (ROUTE_INSERTION, torch.float32, 57, 8, 8, 16),
            (ROUTE_RADIX_ONE_CORE, torch.float32, 9, 513, 768, 1),
            (ROUTE_TWO_STAGE_INVERSE, torch.float32, 447, 17, 1280, 74),
            (ROUTE_TWO_STAGE_PACKED, torch.int32, 153, 65, 1792, 25),
            (ROUTE_RADIX_MORE_CORE, torch.uint64, 6, 16385, 1024, 0),
        ]:
            self.check_direct_schedule(*case, inner=3)

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

    def test_packed_key_maximum(self):
        info = cb.get_platform_info(stream=torch.npu.current_stream())
        cores = int(info.vector_core_num)
        usable = cb.get_mem_size("ub") - SIMT_SCRATCH_BYTES
        for dtype, length, batch, per_core in [
            (torch.int32, 256, 16, 16),
            (torch.uint32, 256, 16, 16),
            (torch.int64, 256, 16, 16),
            (torch.uint64, 256, 16, 16),
            (torch.float32, 64, 32, 64),
        ]:
            for rows in (per_core * cores, 2 * per_core * cores - 1):
                self.assertEqual(batch * batch * length - 1, 65535)
                self.assertEqual(
                    two_stage_batch(dtype, length, rows, cores, usable), batch
                )
                # 每批末行的唯一极值在升降序下均可达到 B*N-1 排名，验证最大键 65535。
                cpu = torch.arange(1, length + 1).repeat(rows, 1)
                last_row = batch - 1
                cpu[last_row::batch, 0] = 0
                cpu[last_row::batch, -1] = batch * length + 1
                cpu = cpu.to(dtype)
                for descending in (False, True):
                    with self.subTest(dtype=dtype, rows=rows, descending=descending):
                        x = cpu.npu()
                        prepared = PreparedSort(x, descending=descending)
                        try:
                            self.assertEqual(prepared.route, "two_stage")
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

    def test_uint64_high_digits(self):
        """逐字节验证跨 32/63 位边界的提取，避免 int64 除法截断污染高位。"""
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
            # N 超过整数两阶段路线的轴长上限，命中局部 Radix。
            cpu = torch.tensor((numbers * 94)[:1025], dtype=dtype).reshape(1, -1)
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

    def test_non_tail_layout(self):
        cpu = ((torch.arange(3 * 65 * 5) * 37) % 71).reshape(3, 65, 5).float()
        x = cpu.npu()
        prepared = PreparedSort(x, dim=1)
        try:
            self.assertTrue(prepared.needs_transpose)
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
        for shape, dim in [((3, 1), -1), ((2, 1, 3), 1)]:
            with self.subTest(shape=shape, dim=dim):
                cpu = torch.arange(
                    6 if len(shape) == 3 else 3, dtype=torch.int64
                ).reshape(shape)
                x = cpu.npu()
                prepared = PreparedSort(x, dim=dim, descending=True, stable=False)
                try:
                    self.assertEqual(prepared.route, "insertion")
                    self.assertFalse(prepared.needs_transpose)
                    self.assertEqual(prepared.kernel_count, 1)
                    values, indices = prepared.run()
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(values.cpu(), cpu))
                    self.assertTrue(torch.equal(indices.cpu(), torch.zeros_like(cpu)))
                    self.assertTrue(torch.equal(x.cpu(), cpu))
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
