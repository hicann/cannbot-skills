#!/usr/bin/env python3
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

"""补充测试：OL28, OL29, OL37, OL42, OL43 规则覆盖。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# ── OL28: FP32-only API (sigmoid/softmax/sin/cos) 与 dtype 注解一致性 ──


def test_ol28_pass_when_no_fp32_only_ops(tmp_path: Path):
    """默认 fixture 不使用 sigmoid 等 API，应 PASS"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL28")
    assert finding.status in ("PASS", "SKIP")


def test_ol28_warn_on_sigmoid_with_bf16(tmp_path: Path):
    """使用 sigmoid 但 dtype 非 FP32 时应 WARN"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """\
import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP16), y: pypto.Tensor([1024], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(32, 128)
    tmp = pypto.sigmoid(x)
    y[:] = tmp

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL28", stage=5)
    assert finding.status == "WARN"


def test_ol28_warn_on_module_file_with_sigmoid_and_fp16(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 FP32-only API 与 dtype 不一致。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """\
import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([1024], pypto.DT_FP16),
                        y: pypto.Tensor([1024], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(32, 128)
    tmp = pypto.sigmoid(x)
    y[:] = tmp

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL28", stage=5)
    assert finding.status == "WARN"
    assert "modules/demo_module1_impl.py" in finding.message


# ─── OL29: Tensor 注解应包含 DYNAMIC 维度 ───

def test_ol29_pass_with_dynamic(tmp_path: Path):
    """默认 helpers 构建的 fixture 含 DYNAMIC，应 PASS"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL29")
    assert finding.status == "PASS"


def test_ol29_warn_all_static(tmp_path: Path):
    """所有维度为常量时应 WARN"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """\
import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024, 512], pypto.DT_FP32), y: pypto.Tensor([1024, 512], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL29")
    assert finding.status == "WARN"


def test_ol29_warn_on_module_file_with_all_static_dims(tmp_path: Path):
    """模块开发阶段（Stage 5 Phase M_k）即捕获 impl 注解全部为常量维度的违规。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    module_impl = """\
import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([1024, 512], pypto.DT_FP32),
                        y: pypto.Tensor([1024, 512], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    demo_module1_kernel(x, y)
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL29")
    assert finding.status == "WARN"
    assert "modules/demo_module1_impl.py" in finding.message


# ── OL37: design 与 impl 命名一致性 ──

def test_ol37_pass_when_names_overlap(tmp_path: Path):
    """设计文档与 impl 共享标识符时应 PASS 或 INFO。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL37", stage=5)
    assert finding.status in ("PASS", "INFO", "SKIP")


# ─── OL42: NPU 环境下 sim 模式检测 ───

def test_ol42_skip_when_no_npu(tmp_path: Path):
    """OL42 在无 NPU 环境下 SKIP，在有 NPU 环境下检查 sim 硬编码。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL42")
    # 默认 fixture 无 sim 硬编码：有 NPU 时 PASS，无 NPU 时 SKIP
    assert finding.status in ("SKIP", "PASS")


# ─── OL43: DESIGN 声明动态轴 → impl 需有 loop ───

def test_ol43_skip_when_no_dynamic_axes(tmp_path: Path):
    """DESIGN 不声明动态轴时，OL43 应 SKIP 或 PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 默认 DESIGN 模板含 dynamic_axes，先改成无动态轴
    design = """---
schema_version: "1"
op_name: demo
dynamic_axes: []
---
# 计算图
## 计算图
graph
## Tiling 策略
tiling
## 验证方案
verify
"""
    write_file(op_dir / "DESIGN.md", design)
    finding = run_rule(mod, op_dir, "OL43", stage=5)
    assert finding.status in ("SKIP", "PASS")


def test_ol43_fail_on_module_file_without_loop_when_dynamic_axes_declared(tmp_path: Path):
    """DESIGN 声明动态轴 → 每个 module impl 都必须含 pypto.loop。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    integrated = op_dir / "demo_impl.py"
    if integrated.exists():
        integrated.unlink()
    # module impl 无 pypto.loop（DESIGN 仍声明动态轴）
    module_impl = """import pypto
@pypto.frontend.jit
def demo_module1_kernel(x: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32),
                        y: pypto.Tensor([pypto.DYNAMIC], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_module1_wrapper(x, y):
    return None
"""
    write_file(op_dir / "modules" / "demo_module1_impl.py", module_impl)
    finding = run_rule(mod, op_dir, "OL43", stage=5)
    assert finding.status == "FAIL"
    assert "modules/demo_module1_impl.py" in finding.message
