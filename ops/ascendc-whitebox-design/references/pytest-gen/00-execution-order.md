# Pytest Generator — 执行总纲

> **执行顺序约束（强制）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 详细规则见本文件后续章节，执行顺序节未覆盖的细节以各章节为准。
>
> 1. Read 本文件 + `02-constraints.md`（全程适用）+ 收集输入文件
>    前置：无
> 2. 按需 Read `01-code-templates.md` / `03-pitfalls.md` / aclnn 文档「计算公式」节 → 生成 S6_test_{op_name}.py
>    前置：Step 1 完成
> 3. 按 `02-constraints.md`「验证流程」执行自审 + 门禁验证 + tilingkey 覆盖率
>    前置：Step 2 完成

## 角色

将 S5_mapped_cases_low.json（已映射的 tensor 构造配置）转化为一个完整的、可直接 `pytest` 执行的测试文件 S6_test_{op_name}.py。

## 输入

| 文件 | 必须？ | 用途 |
|------|--------|------|
| `S5_mapped_cases_low.json` | 是 | 已映射的参数组合（含 params + tensors，每个 case 已有完整 shape/dtype） |
| `S5_mapping_spec.md` | 是 | 确定每个 input/output 的 param_type（REQUIRED/DYNAMIC），派发 API 调用/reference/断言模板 |
| `S2P2_param_def.json` | 是 | 参数定义（用于理解参数含义，不需要再做过滤） |
| 算子接口信息 | 是 | 函数签名、输入输出 tensor 定义、调用方式 |
| `docs/aclnn*.md`「计算公式」节 | 是 | 仅读该节，作为 reference 实现的唯一依据 |

## 派发机制

`S5_mapping_spec.md` 的输入 tensor 和输出 tensor 各节 header 已标注每个 tensor 的 param_type（`### {name}（REQUIRED）` 或 `### {name}（DYNAMIC）`）。子 agent 生成代码时，对每个 input/output 变量读其 param_type，选择对应分支：

| param_type | JSON spec 结构 | Python 变量 | API 调用 | reference 返回 |
|-----------|---------------|------------|---------|--------------|
| REQUIRED | `dict` | `Tensor` | `op(tensor)` | `Tensor` |
| DYNAMIC | `list[dict]` | `list[Tensor]` | `op([t1, t2, ...])` | `list[Tensor]` |

**无默认行为**：每个 tensor 显式匹配 param_type。混合算子中每个 input/output 独立派发（如 input_a 走 DYNAMIC 分支，input_b 走 REQUIRED 分支）。

**tensor 构造统一处理**：输入 tensor 构造和输出预分配使用 `isinstance(spec, list)` 运行时判断（一套模板通吃两种 param_type，见 01-code-templates.md）。API 调用、reference 实现、断言逻辑由子 agent 按 param_type 显式派发生成。

## 输出

| 文件 | 用途 |
|------|------|
| `conftest.py` | pytest 插件入口，注册 `--cases-file` 命令行选项 |
| `S6_test_{op_name}.py` | 完整的 pytest 测试文件 |

## 文件结构要求

### conftest.py

`pytest_addoption` 是 conftest 级别 hook，必须在 pytest 解析命令行参数之前注册，因此只能放在 `conftest.py` 中，不能放在测试模块中。

```python
"""conftest.py — pytest 插件入口（注册命令行选项）"""
import pytest

def pytest_addoption(parser):
    parser.addoption("--cases-file", default="S5_mapped_cases_low.json")
```

### S6_test_{op_name}.py

测试文件必须包含以下部分，按此顺序：

```python
# 1. 导入（NPU 不可用时整个文件 skip，而非假 PASS）
import pytest
import torch
torch_npu = pytest.importorskip("torch_npu")

import json, os

_CASES_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 常量（DTYPE_MAP / TOLERANCE 详见 01-code-templates.md）

# 3. Reference 实现（仅基于 aclnn 文档「计算公式」一节，详见 03-pitfalls.md）
def reference_{op_name}(...):
    """CPU 参考实现。严格按 aclnn 文档计算公式节编写，不读其他节。"""
    ...

# 4. 测试函数
def pytest_generate_tests(metafunc):
    if "p" in metafunc.fixturenames:
        cases_file = metafunc.config.getoption("--cases-file", "S5_mapped_cases_low.json")
        with open(os.path.join(_CASES_DIR, cases_file)) as _f:
            cases = json.load(_f)["cases"]
        metafunc.parametrize("p", cases, ids=lambda c: c["id"])

def test_{op_name}(p):
    tensors = p["tensors"]
    params = p["params"]
    # a. 构造输入 tensor（运行时 isinstance 判断 REQUIRED/DYNAMIC，详见 01-code-templates.md）
    # b. 调用算子（NPU）— 按 S5_mapping_spec.md 中各 input 的 param_type 派发：
    #    REQUIRED: npu_out = torch_npu.npu_xxx(input_x, ...)
    #    DYNAMIC:  npu_out = torch._foreach_xxx(input_x_list, ...)
    # c. 调用 reference（CPU）— 按 S5_mapping_spec.md 中各 output 的 param_type 派发：
    #    REQUIRED: ref_out = reference(x)
    #    DYNAMIC:  ref_out = [reference(t) for t in x_list]
    # d. 断言（shape + dtype + 数值精度，按 param_type 派发，详见 01-code-templates.md）
```

## 关键规则

S5_mapped_cases_low.json 经过 Step 5 mapper 映射 + validate_config 校验，每个 case 的 tensor shape/dtype 均为合法值。不需要 `is_valid_combo()` 或额外的 shape 过滤。直接使用全部 PARAMS。

## 用例选择

单个 `S6_test_{op_name}.py` 通过参数控制数据源：
- `--cases-file` 切换数据源：
  - 全量版：`pytest ... --cases-file=S5_mapped_cases_high.json`
  - 低覆盖版：`pytest ... --cases-file=S5_mapped_cases_low.json`
- 用例筛选：使用 pytest 内置 `-k` 表达式（基于 `case["id"]`）
  - 单个：`pytest ... -k case00001`
  - 多个：`pytest ... -k "case00001 or case00008"`
  - 与数据源组合：`pytest ... --cases-file=S5_mapped_cases_low.json -k network_00000`

所有逻辑（DTYPE_MAP / TOLERANCE / reference / 测试函数 / 断言 / 约束 / 铁律）完全一致。

## 生成后自审清单

生成 `conftest.py` + `S6_test_{op_name}.py` 后，执行以下验证（发现问题就修，修完再报告）：

1. `python -m py_compile conftest.py S6_test_{op_name}.py` — 无语法错误？
2. `pytest --cases-file=S5_mapped_cases_low.json --collect-only S6_test_{op_name}.py` — `--cases-file` 可识别？能收集到用例？ids 是否唯一？
3. `ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py --cases-file=S5_mapped_cases_low.json -q --tb=line` — 全部用例合法执行？0 个 ERROR / RuntimeError / AttributeError？
4. NPU 不可用时是否正确 skip 而非假 PASS？
5. plog 是否成功复制到 `tilingkey_logs/{op_name}_full.log`？覆盖率脚本是否生成 `S6_tilingkey_coverage.json`（含全局与 per_group 覆盖率）？
6. 单用例执行后能否从 plog 提取 tiling key？
   ```bash
   grep "Tiling Key:" $(ls -t ~/ascend/log/debug/plog/plog-*.log | head -1)
   ```

## 文件索引

| 文件 | 职责 | 读入时机 |
|------|------|---------|
| `01-code-templates.md` | DTYPE_MAP/TOLERANCE 常量、make_data 函数、tensor 构造模板、断言模板、✅/❌ 示例 | 编写 S6_test_{op_name}.py 代码时 |
| `02-constraints.md` | 铁律（3条 NO 规则 + 验证流程 + plog 采集）+ 严格禁止（9条） | 全程适用 |
| `03-pitfalls.md` | 6个实战规则（API探测/golden来源/交叉验证/假PASS/inf-nan/可变输出） | 编写 reference 和断言时 |
