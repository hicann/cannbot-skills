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

"""D1 PyPTO API 存在性检查 (OL55) 单元测试。

OL55 通过对比 AST 中的 ``pypto.<attr>`` 与 ``dir(pypto)`` 阻止 typo
(``pypto.empty`` 等)。本测试用 ``unittest.mock.patch`` 替换实际的
``get_pypto_attrs`` 函数, 避免依赖 lint 环境是否真的装了 pypto。
"""
from pathlib import Path
from unittest.mock import patch

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# 模拟 dir(pypto) 的常用属性子集 (覆盖 build_stateless_op_dir 默认 impl
# 与本测试中合法 case 用到的全部 attr)。
_FAKE_PYPTO_ATTRS = frozenset({
    # 张量与类型注解
    "Tensor", "DataType",
    "DT_FP32", "DT_FP16", "DT_BF16", "DT_INT32", "DT_UINT8",
    "DYNAMIC", "STATIC",
    # 创建 / 转换
    "zeros", "ones", "full", "from_torch", "cast", "clone",
    # 算术与归约
    "add", "sub", "mul", "div", "exp", "log", "amax", "amin", "sum", "matmul",
    # 控制 / 视图
    "loop", "view", "assemble",
    # tile 配置
    "set_vec_tile_shapes", "set_cube_tile_shapes",
    # 子模块 / 顶层
    "frontend", "op", "config", "RunMode",
})


def _patch_attrs(attrs):
    """Patch ``get_pypto_attrs`` 让 OL55 不再真去 import pypto。"""
    return patch(
        "pypto_op_lint.checks.d1_pypto_api.get_pypto_attrs",
        return_value=set(attrs),
    )


# ──────────────────────────────────────────────────────────────────────────────
# impl.py 上的检查
# ──────────────────────────────────────────────────────────────────────────────


def test_ol55_pass_on_valid_impl(tmp_path: Path):
    """build_stateless_op_dir 的默认 impl 全部使用合法 attr → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "PASS", finding.message


def test_ol55_fail_on_pypto_empty_in_impl(tmp_path: Path):
    """impl 中误用 ``pypto.empty`` (不存在) → FAIL 并建议 ``pypto.zeros``。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl_with_typo = """import pypto

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x):
    out = pypto.empty([1024], dtype=pypto.DT_FP32)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl_with_typo)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "FAIL", finding.message
    assert "empty" in finding.message
    # zeros 应作为 typo 替换建议出现
    assert "zeros" in finding.message


def test_ol55_fail_on_pypto_empty_like_with_host_wrapper_guidance(tmp_path: Path):
    """impl 中误用 ``pypto.empty_like`` → FAIL 并提示 host wrapper 使用 torch.empty_like。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl_with_typo = """import pypto

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x):
    out = pypto.empty_like(x)
    return out
"""
    write_file(op_dir / "demo_impl.py", impl_with_typo)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "FAIL", finding.message
    assert "empty_like" in finding.message
    assert "torch.empty_like" in finding.message
    assert "pypto.zeros" in finding.message


def test_ol55_fail_with_no_close_match(tmp_path: Path):
    """完全无关的 attr → FAIL, 即使无近似建议。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32),
                y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = pypto.zxqvbnmpt(x)

def demo_wrapper(x):
    return x
"""
    write_file(op_dir / "demo_impl.py", impl)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "FAIL", finding.message
    assert "zxqvbnmpt" in finding.message


def test_ol55_chained_attribute_only_outer_layer(tmp_path: Path):
    """``pypto.frontend.jit`` 链式属性只检查最外层 ``frontend`` → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # default impl 已包含 @pypto.frontend.jit
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    # frontend 在 fake attrs 中 → 即使 jit 不在最外层集合也应 PASS
    assert finding.status == "PASS", finding.message


def test_ol55_pass_with_alias_import(tmp_path: Path):
    """``import pypto as pt`` 后 ``pt.zeros`` 通过别名解析 → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto as pt

@pt.frontend.jit
def demo_kernel(x: pt.Tensor([1024], pt.DT_FP32),
                y: pt.Tensor([1024], pt.DT_FP32)):
    pt.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x):
    return pt.zeros([1024], dtype=pt.DT_FP32)
"""
    write_file(op_dir / "demo_impl.py", impl)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "PASS", finding.message


def test_ol55_fail_with_alias_import_and_typo(tmp_path: Path):
    """``import pypto as pt`` 后 ``pt.empty`` (typo) → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto as pt

@pt.frontend.jit
def demo_kernel(x: pt.Tensor([1024], pt.DT_FP32),
                y: pt.Tensor([1024], pt.DT_FP32)):
    pt.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x):
    return pt.empty([1024], dtype=pt.DT_FP32)
"""
    write_file(op_dir / "demo_impl.py", impl)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "FAIL", finding.message
    assert "empty" in finding.message


# ──────────────────────────────────────────────────────────────────────────────
# DESIGN.md 上的检查 (仅扫描 Python 代码块)
# ──────────────────────────────────────────────────────────────────────────────


def test_ol55_fail_in_design_md_python_block(tmp_path: Path):
    """DESIGN.md 的 ```python``` 块中 ``pypto.empty`` → FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---

# DESIGN

## Layer K — Host wrapper

```python
import pypto

def demo_wrapper(x):
    y = pypto.empty([1024], dtype=pypto.DT_FP32)
    return y
```
"""
    write_file(op_dir / "DESIGN.md", design)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=4)
    assert finding.status == "FAIL", finding.message
    assert "empty" in finding.message
    assert "zeros" in finding.message


def test_ol55_pass_design_md_with_valid_pseudo_code(tmp_path: Path):
    """DESIGN.md 中只用合法 ``pypto.<attr>`` → PASS。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---

# DESIGN

```python
import pypto

def demo_kernel(x):
    return pypto.amax(x, dim=-1, keepdim=True)
```
"""
    write_file(op_dir / "DESIGN.md", design)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=4)
    assert finding.status == "PASS", finding.message


def test_ol55_ignores_non_python_code_blocks(tmp_path: Path):
    """DESIGN.md 中 ``bash`` / ``yaml`` 块不应被扫描。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---

# DESIGN

```bash
echo "pypto.empty"
```

```yaml
note: pypto.empty
```
"""
    write_file(op_dir / "DESIGN.md", design)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=4)
    # 非 Python 块不会被解析, OL55 不应 FAIL
    assert finding.status == "PASS", finding.message


def test_ol55_ignores_pseudo_code_with_syntax_error(tmp_path: Path):
    """DESIGN.md 的 Python 块如果 ``ast.parse`` 失败 (伪代码片段) → 静默忽略。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---

# DESIGN

```python
# 伪代码: pypto.empty(<some shape>)  -- 不完整, ast.parse 会失败
def f(...):  # 这一行就是 syntax error
    return pypto.empty(...)
```
"""
    write_file(op_dir / "DESIGN.md", design)
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=4)
    # 伪代码块被忽略, 不应触发 FAIL
    assert finding.status == "PASS", finding.message


# ──────────────────────────────────────────────────────────────────────────────
# 环境降级
# ──────────────────────────────────────────────────────────────────────────────


def test_ol55_skip_when_pypto_unavailable(tmp_path: Path):
    """``get_pypto_attrs`` 返回 None (本地无 pypto) → SKIP, 不应 FAIL。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    with patch(
        "pypto_op_lint.checks.d1_pypto_api.get_pypto_attrs",
        return_value=None,
    ):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "SKIP", finding.message


def test_ol55_skip_when_no_target_files(tmp_path: Path):
    """既无 DESIGN.md 也无 impl.py → SKIP。"""
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 移除 default impl 与 DESIGN.md
    (op_dir / "demo_impl.py").unlink()
    (op_dir / "DESIGN.md").unlink()
    with _patch_attrs(_FAKE_PYPTO_ATTRS):
        finding = run_rule(mod, op_dir, "OL55", stage=5)
    assert finding.status == "SKIP", finding.message
