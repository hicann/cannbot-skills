# 代码模板

## 常量

```python
DTYPE_MAP = {
    "float16": torch.float16, "bfloat16": torch.bfloat16,
    "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "bool": torch.bool,
}

TOLERANCE = {
    torch.float32: (1e-4, 1e-4), torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-3, 1e-3), torch.float64: (1e-6, 1e-6),
    # 非 float 类型不走 TOLERANCE，用 torch.equal 精确比较
}
```

## Tensor 构造

Tensor 构造分两种场景，由 S5_mapping_spec.md 中各 tensor 的 param_type 标注决定：

| param_type | JSON spec 结构 | 构造方式 |
|-----------|---------------|---------|
| REQUIRED | `dict`（含 shape/dtype/_data_range） | 单次 `make_data` 调用 |
| DYNAMIC | `list[dict]`（每个子 tensor 含 shape/dtype/_data_range） | 逐子 tensor `make_data` 调用，返回 `list[Tensor]` |

两种场景共用 `make_data` 函数和统一的 `isinstance(spec, list)` 运行时分支（无需子 agent 按 param_type 生成两套代码）。

### 输入 tensor

shape 和 dtype 直接从 `case["tensors"]["inputs"]` 读取，无需推导。REQUIRED 输入（`dict`）和 DYNAMIC 输入（`list[dict]`）通过 `isinstance(spec, list)` 统一处理。根据每个 tensor 的 `_data_range` 字段选择构造方式：

```python
def make_data(shape, dtype, data_range, value_domain=None):
    """根据 data_range 构造不同值域的 tensor。value_domain 约束生成范围。"""
    if value_domain and value_domain["type"] == "range":
        eff_lo = value_domain.get("min")
        eff_hi = value_domain.get("max")
        eff_lo = eff_lo if eff_lo is not None else -10.0
        eff_hi = eff_hi if eff_hi is not None else 10.0
        if data_range == "normal":
            return torch.rand(shape, dtype=dtype) * (eff_hi - eff_lo) + eff_lo
        elif data_range == "negative":
            neg_hi = min(0, eff_hi)
            if eff_lo < neg_hi:
                return torch.rand(shape, dtype=dtype) * (neg_hi - eff_lo) + eff_lo
            return torch.rand(shape, dtype=dtype) * (eff_hi - eff_lo) + eff_lo
        elif data_range == "near_zero":
            nz_lo = max(eff_lo, -0.01)
            nz_hi = min(eff_hi, 0.01)
            if nz_lo < nz_hi:
                return torch.rand(shape, dtype=dtype) * (nz_hi - nz_lo) + nz_lo
            return torch.rand(shape, dtype=dtype) * (eff_hi - eff_lo) + eff_lo
        elif data_range == "tiny_pos":
            tp_lo = max(eff_lo, 1e-7)
            tp_hi = min(eff_hi, 1e-5)
            if tp_lo < tp_hi:
                return torch.rand(shape, dtype=dtype) * (tp_hi - tp_lo) + tp_lo
            return torch.rand(shape, dtype=dtype) * (eff_hi - eff_lo) + eff_lo

    if data_range == "zero":
        return torch.zeros(shape, dtype=dtype)
    elif data_range == "extreme":
        dtype_max = {torch.float16: 65504.0, torch.bfloat16: 3.3895e38, torch.float32: 3.4e38}
        return torch.full(shape, dtype_max.get(dtype, 3.4e38), dtype=dtype)
    elif data_range == "negative":
        return -torch.rand(shape, dtype=dtype) * 10
    elif data_range == "tiny_pos":
        return torch.ones(shape, dtype=dtype) * 1e-6
    elif data_range == "all_ones":
        return torch.ones(shape, dtype=dtype)
    elif data_range == "near_zero":
        return (torch.rand(shape, dtype=dtype) - 0.5) * 0.02
    elif data_range == "with_inf":
        t = torch.randn(shape, dtype=dtype)
        t.view(-1)[0] = float('inf')
        return t
    elif data_range == "with_nan":
        t = torch.randn(shape, dtype=dtype)
        t.view(-1)[0] = float('nan')
        return t
    else:
        if value_domain:
            t = value_domain["type"]
            if t == "positive":
                return torch.rand(shape, dtype=dtype) * 10 + 0.01
            elif t == "non_negative":
                return torch.rand(shape, dtype=dtype) * 10
            elif t == "non_zero":
                r = torch.randn(shape, dtype=dtype)
                return torch.where(r.abs() < 0.1, torch.ones_like(r), r)
        return torch.randn(shape, dtype=dtype)

tensors = p["tensors"]
params = p["params"]
inputs = {}
for name, spec in tensors["inputs"].items():
    if spec is None:
        inputs[name] = None
        continue
    if isinstance(spec, list):
        inputs[name] = [
            make_data(sub["shape"], DTYPE_MAP[sub["dtype"]],
                      sub.get("_data_range", "normal"),
                      sub.get("_value_domain")).npu()
            for sub in spec
        ]
    else:
        dr = spec.get("_data_range", "normal")
        vd = spec.get("_value_domain")
        inputs[name] = make_data(spec["shape"], DTYPE_MAP[spec["dtype"]], dr, vd).npu()
```

**可选输入**：S5 映射中 optional tensor 为 None 时，跳过构造，传 None 给算子。

### 输出 tensor（预分配场景）

部分算子 API 要求调用方预分配输出 tensor。此时从 `case["tensors"]["outputs"]` 构造。REQUIRED 输出（`dict`）和 DYNAMIC 输出（`list[dict]`）同样通过 `isinstance(spec, list)` 统一处理：

```python
outputs_prealloc = {}
for name, spec in tensors["outputs"].items():
    if spec is None:
        continue
    if isinstance(spec, list):
        # DYNAMIC（TensorList）：逐子 tensor 预分配
        outputs_prealloc[name] = [
            torch.empty(sub["shape"], dtype=DTYPE_MAP[sub["dtype"]]).npu()
            for sub in spec
        ]
    else:
        # REQUIRED（单 tensor）
        outputs_prealloc[name] = torch.empty(spec["shape"], dtype=DTYPE_MAP[spec["dtype"]]).npu()
```

如果算子 API 不要求预分配（API 内部创建输出），则跳过此步。

## 标量属性

从 `case["params"]` 读取。根据算子接口需要提取对应的标量参数：

```python
params = p["params"]
attr_1 = params.get("attr_name_1", default_1)
attr_2 = params.get("attr_name_2", default_2)
```

## 断言

断言逻辑按 output 的 param_type 派发。子 agent 根据 S5_mapping_spec.md 中各 output 的标注选择对应模板。

### REQUIRED 模式（单 tensor 输出）

- **shape 检查**：`assert output.shape == expected_shape`
- **dtype 检查**：`assert output.dtype == expected_dtype`
- **数值对比**：逐输出对比，精度校验失败时必须转为 XFAIL（见 02-constraints.md 铁律）：

```python
for i, (npu_out, ref_out) in enumerate(zip(npu_outputs, ref_outputs)):
    if npu_out.dtype not in TOLERANCE:
        # 非 float 类型：精确比较
        assert torch.equal(npu_out.cpu(), ref_out), f"Output[{i}] value mismatch"
        continue
    rtol, atol = TOLERANCE[npu_out.dtype]
    try:
        torch.testing.assert_close(npu_out.cpu().float(), ref_out.cpu().float(),
                                   rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:
        pytest.xfail(f"Output[{i}] precision mismatch: {e}")
```

### DYNAMIC 模式（TensorList 输出）

- **shape 检查**：逐子 tensor 检查 `assert sub_out.shape == expected_sub_shape`
- **dtype 检查**：逐子 tensor 检查 `assert sub_out.dtype == expected_sub_dtype`
- **数值对比**：嵌套迭代（逐输出名 → 逐子 tensor），精度校验失败时必须转为 XFAIL：

```python
# npu_out 和 ref_out 均为 list[Tensor]，逐子 tensor 对比
for i, (sub_npu, sub_ref) in enumerate(zip(npu_out_list, ref_out_list)):
    assert sub_npu.shape == sub_ref.shape, f"Sub-tensor[{i}] shape mismatch"
    assert sub_npu.dtype == sub_ref.dtype, f"Sub-tensor[{i}] dtype mismatch"
    if sub_npu.dtype not in TOLERANCE:
        assert torch.equal(sub_npu.cpu(), sub_ref), f"Sub-tensor[{i}] value mismatch"
        continue
    rtol, atol = TOLERANCE[sub_npu.dtype]
    try:
        torch.testing.assert_close(sub_npu.cpu().float(), sub_ref.cpu().float(),
                                   rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:
        pytest.xfail(f"Sub-tensor[{i}] precision mismatch: {e}")
```

- 如果某些参数组合没有 reference 实现，只做 shape/dtype 检查

## parametrize ids

直接使用 S5 映射中的 `case["id"]`：

```python
@pytest.mark.parametrize("p", PARAMS, ids=lambda c: c["id"])
def test_{op_name}(p):
    ...
```

## ✅/❌ 示例

```python
# ✅ 正确：conftest.py 注册 --cases-file，测试文件通过 getoption 读取
# conftest.py:
import pytest
def pytest_addoption(parser):
    parser.addoption("--cases-file", default="S5_mapped_cases_low.json")

# S6_test_{op_name}.py:
import json, os
_CASES_DIR = os.path.dirname(os.path.abspath(__file__))
def pytest_generate_tests(metafunc):
    if "p" in metafunc.fixturenames:
        cases_file = metafunc.config.getoption("--cases-file", "S5_mapped_cases_low.json")
        with open(os.path.join(_CASES_DIR, cases_file)) as _f:
            cases = json.load(_f)["cases"]
        metafunc.parametrize("p", cases, ids=lambda c: c["id"])

# ❌ 错误：pytest_addoption 写在测试模块中，命令行 --cases-file 无法注册
# S6_test_{op_name}.py:
def pytest_addoption(parser):  # 不会被 pytest 识别为 conftest hook
    parser.addoption("--cases-file", default="S5_mapped_cases_low.json")

# ❌ 错误：手写参数
PARAMS = [{"dtype": "float16", "D": 32}]
```
