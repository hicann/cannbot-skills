# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import torch
import unittest

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

import extension_cpp


def reference_muladd(a, b, c):
    return a * b + c


def get_extension(ext_name):
    if ext_name != "extension_cpp":
        raise ValueError(f"unsupported extension: {ext_name}")
    return extension_cpp


def has_npu():
    if torch_npu is None or not hasattr(torch, "npu"):
        return False
    try:
        if not torch.npu.is_available():
            return False
        torch.empty(1, device="npu")
        return True
    except RuntimeError:
        return False


HAS_NPU = has_npu()


def opcheck_test_utils(device):
    if device == "npu":
        return ("test_schema", "test_faketensor")
    return (
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    )


class TestMyMulAdd(unittest.TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device, ext):
        for args in self.sample_inputs(device):
            result = ext.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    def _test_gradients(self, device, ext):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = ext.ops.mymuladd(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_muladd(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)
            torch.testing.assert_close(result, expected)

    def _opcheck(self, device, ext_name):
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        op = getattr(torch.ops, ext_name).mymuladd.default
        for args in samples:
            torch.library.opcheck(op, args, test_utils=opcheck_test_utils(device))

    def test_correctness_cpu(self):
        self._test_correctness("cpu", get_extension("extension_cpp"))

    def test_gradients_cpu(self):
        self._test_gradients("cpu", get_extension("extension_cpp"))

    def test_opcheck_cpu(self):
        self._opcheck("cpu", "extension_cpp")

    @unittest.skipUnless(HAS_NPU, "requires npu")
    def test_correctness_npu(self):
        self._test_correctness("npu", get_extension("extension_cpp"))

    @unittest.skipUnless(HAS_NPU, "requires npu")
    def test_gradients_npu(self):
        self._test_gradients("npu", get_extension("extension_cpp"))

    @unittest.skipUnless(HAS_NPU, "requires npu")
    def test_opcheck_npu(self):
        self._opcheck("npu", "extension_cpp")


class TestMyAddOut(unittest.TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        return [
            [make_tensor(3), make_tensor(3), make_tensor(3)],
            [make_tensor(20), make_tensor(20), make_tensor(20)],
        ]

    def _test_correctness(self, device, ext):
        for args in self.sample_inputs(device):
            result = args[-1]
            ext.ops.myadd_out(*args)
            expected = torch.add(*args[:2])
            torch.testing.assert_close(result, expected)

    def _opcheck(self, device, ext_name):
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        op = getattr(torch.ops, ext_name).myadd_out.default
        for args in samples:
            torch.library.opcheck(op, args, test_utils=opcheck_test_utils(device))

    def test_correctness_cpu(self):
        self._test_correctness("cpu", get_extension("extension_cpp"))

    def test_opcheck_cpu(self):
        self._opcheck("cpu", "extension_cpp")

    @unittest.skipUnless(HAS_NPU, "requires npu")
    def test_correctness_npu(self):
        self._test_correctness("npu", get_extension("extension_cpp"))

    @unittest.skipUnless(HAS_NPU, "requires npu")
    def test_opcheck_npu(self):
        self._opcheck("npu", "extension_cpp")


if __name__ == "__main__":
    unittest.main()
