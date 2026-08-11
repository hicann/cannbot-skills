# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""D4 测试规范规则（OL19-OL22）核心测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def _jit_impl(
    kernel_name: str = "demo_kernel_npu",
    wrapper_name: str | None = None,
    pure_entry_name: str | None = None,
    delegate_name: str | None = None,
) -> str:
    parts = [
        "import pypto",
        "import torch",
        "",
        "@pypto.frontend.jit",
        f"def {kernel_name}(x: pypto.Tensor([128], pypto.DT_FP32),",
        "                    out: pypto.Tensor([128], pypto.DT_FP32)):",
        "    pypto.set_vec_tile_shapes(128)",
        "    out[:] = x",
    ]
    if pure_entry_name:
        parts.extend([
            "", f"def {pure_entry_name}(x):",
            "    return torch.zeros_like(x)",
        ])
    if wrapper_name:
        parts.extend([
            "", f"def {wrapper_name}(x):", "    out = torch.empty_like(x)",
            f"    {kernel_name}(x, out)", "    return out",
        ])
    if delegate_name and wrapper_name:
        parts.extend([
            "", f"def {delegate_name}(x):", f"    return {wrapper_name}(x)",
        ])
    return "\n".join(parts) + "\n"


def _entry_test(module_name: str, entry_name: str, expected: str = "x") -> str:
    return "\n".join([
        "import torch",
        "from numpy.testing import assert_allclose",
        f"from {module_name} import {entry_name}",
        "",
        "def test_level0_basic():",
        "    x = torch.randn(128, dtype=torch.float32)",
        f"    y = {entry_name}(x)",
        f"    assert_allclose(y.numpy(), {expected}.numpy(), atol=1e-3, rtol=1e-3)",
        "",
    ])


def _write_ol60_case(
    op_dir: Path,
    impl: str,
    test: str,
    *,
    module: bool = False,
) -> None:
    if module:
        write_file(op_dir / "test_demo.py", "")
        write_file(op_dir / "modules" / "demo_module1_impl.py", impl)
        write_file(op_dir / "modules" / "test_demo_module1.py", test)
        return
    write_file(op_dir / "demo_impl.py", impl)
    write_file(op_dir / "test_demo.py", test)


def test_ol19_fail_when_no_assert_allclose(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    y = x.float().numpy()
    assert abs(y.max() - y.max()) < 0.01
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL19")
    assert finding.status == "FAIL"


def test_ol19_pass_when_assert_allclose_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL19")
    assert finding.status == "PASS"


def test_ol19_pass_when_detailed_tensor_compare_present(tmp_path: Path):
    """detailed_tensor_compare alone (no assert_allclose) should still PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from detailed_tensor_compare import detailed_tensor_compare

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    y = torch.randn(1024, dtype=torch.bfloat16)
    detailed_tensor_compare(x, y, atol=0.001, rtol=0.001, tensor_name="out0")
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL19")
    assert finding.status == "PASS", finding.message


def test_ol20_fail_when_no_device_id_and_no_set_device(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_fail_when_device_id_but_no_set_device(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import os
import torch
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_fail_when_set_device_but_no_device_id(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch

def test_level0_basic():
    torch.npu.set_device(0)
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_pass_when_both_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import os
import torch
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))

def test_level0_basic():
    torch.npu.set_device(device_id)
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "PASS"


def test_ol21_fail_when_missing_level1(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "FAIL"


def test_ol21_pass_when_both_levels(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "PASS"


def test_ol21_pass_when_l0_l1_naming(tmp_path: Path):
    """test_template.py.tmpl 推荐的 _l0/_l1 命名应通过 OL21。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_demo_l0():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())

def test_demo_l1():
    x = torch.randn(1024, 256, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "PASS", finding.message


def test_ol21_fail_when_only_l0(tmp_path: Path):
    """只有 _l0 没有 _l1 时应 FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_demo_l0():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "FAIL"


def test_ol22_fail_when_no_manual_seed(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL22")
    assert finding.status == "WARN"


def test_ol22_pass_when_manual_seed_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
torch.manual_seed(42)

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL22")
    assert finding.status == "PASS"


# =============================================================================
# OL60 — test_<op>.py entry must transitively reach @pypto.frontend.jit
# =============================================================================


def test_ol59_fail_test_calls_pure_torch_entry(tmp_path: Path):
    """flash_kda pattern: test calls an entry that never reaches @jit -> FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _jit_impl(
        wrapper_name="demo_wrapper", pure_entry_name="pure_torch_entry",
    )
    test = _entry_test(
        "demo_impl", "pure_torch_entry", expected="torch.zeros_like(x)",
    )
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "FAIL", finding.message
    assert "flash_kda" in finding.message


def test_ol59_pass_test_calls_wrapper_that_invokes_jit(tmp_path: Path):
    """Task_0527 canonical pattern: test -> wrapper -> @jit -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _jit_impl(wrapper_name="demo_wrapper")
    test = _entry_test("demo_impl", "demo_wrapper")
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "PASS", finding.message


def test_ol59_fail_when_impl_has_no_jit_function(tmp_path: Path):
    """impl has no @jit at all but test imports and calls something -> FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import torch

def demo_entry(x):
    return x * 2
"""
    test = """import torch
from numpy.testing import assert_allclose
from demo_impl import demo_entry

def test_level0_basic():
    x = torch.randn(128, dtype=torch.float32)
    y = demo_entry(x)
    assert_allclose(y.numpy(), (x * 2).numpy(), atol=1e-3, rtol=1e-3)
"""
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "FAIL", finding.message
    assert "没有任何 @pypto.frontend.jit" in finding.message


def test_ol59_skip_when_no_impl_import(tmp_path: Path):
    """Test does not import anything from *_impl -> SKIP."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _jit_impl(wrapper_name="demo_wrapper")
    test = """import torch
from numpy.testing import assert_allclose

def test_level0_basic():
    x = torch.randn(128, dtype=torch.float32)
    assert_allclose(x.numpy(), x.numpy(), atol=1e-3, rtol=1e-3)
"""
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "SKIP", finding.message


def test_ol59_skip_when_impl_imported_but_not_called(tmp_path: Path):
    """Symbol is imported but never invoked in the test body -> SKIP (dead import)."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _jit_impl(wrapper_name="demo_wrapper")
    test = """import torch
from numpy.testing import assert_allclose
from demo_impl import demo_wrapper  # noqa: F401  imported but not called

def test_level0_basic():
    x = torch.randn(128, dtype=torch.float32)
    assert_allclose(x.numpy(), x.numpy(), atol=1e-3, rtol=1e-3)
"""
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "SKIP", finding.message


def test_ol59_pass_two_level_delegate(tmp_path: Path):
    """test -> demo_entry -> demo_wrapper -> @jit (transitive depth 2) -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = _jit_impl(wrapper_name="demo_wrapper", delegate_name="demo_entry")
    test = _entry_test("demo_impl", "demo_entry")
    _write_ol60_case(op_dir, impl, test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "PASS", finding.message


# -- Stage 5 per-module test coverage (modules/test_<op>_module<k>.py) ------


def test_ol59_pass_when_module_test_reaches_jit(tmp_path: Path):
    """Stage 5: per-module test calls a wrapper that reaches @jit -> PASS."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    module_impl = _jit_impl(
        kernel_name="demo_module1_kernel_npu",
        wrapper_name="demo_module1_wrapper",
    )
    module_test = _entry_test("demo_module1_impl", "demo_module1_wrapper")
    _write_ol60_case(op_dir, module_impl, module_test, module=True)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "PASS", finding.message


def test_ol59_fail_when_module_test_calls_pure_torch_entry(tmp_path: Path):
    """Stage 5: per-module test calls a pure-PyTorch entry (flash_kda pattern) -> FAIL."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    module_impl = _jit_impl(
        kernel_name="demo_module1_kernel_npu",
        pure_entry_name="demo_module1_pure_torch_entry",
    )
    module_test = _entry_test(
        "demo_module1_impl",
        "demo_module1_pure_torch_entry",
        expected="torch.zeros_like(x)",
    )
    _write_ol60_case(op_dir, module_impl, module_test, module=True)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "FAIL", finding.message
    assert "flash_kda" in finding.message


def test_ol59_skip_when_no_integrated_or_module_test(tmp_path: Path):
    """Neither test_<op>.py nor modules/test_<op>_module*.py exists -> SKIP."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # Wipe the baseline integrated test created by build_stateless_op_dir.
    (op_dir / "test_demo.py").unlink()
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "SKIP", finding.message
    assert "未发现" in finding.message


def test_ol59_pass_with_both_integrated_and_module_tests(tmp_path: Path):
    """When both Stage 5 module test and Stage 6 integrated test are present, both must pass."""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "modules").mkdir(exist_ok=True)
    integrated_impl = _jit_impl(wrapper_name="demo_wrapper")
    integrated_test = _entry_test("demo_impl", "demo_wrapper")
    module_impl = _jit_impl(
        kernel_name="demo_module1_kernel_npu",
        wrapper_name="demo_module1_wrapper",
    )
    module_test = _entry_test("demo_module1_impl", "demo_module1_wrapper")
    write_file(op_dir / "demo_impl.py", integrated_impl)
    write_file(op_dir / "test_demo.py", integrated_test)
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    write_file(op_dir / "modules" / "test_demo_module1.py", module_test)
    finding = run_rule(mod, op_dir, "OL60")
    assert finding.status == "PASS", finding.message
