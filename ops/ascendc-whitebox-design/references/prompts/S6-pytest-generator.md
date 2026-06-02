# Pytest Generator — 测试文件生成

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。

1. Step 6a：生成 `conftest.py` 和 `S6_test_{op_name}.py`
   前置：`S5_mapped_cases_high.json`、`S2P2_param_def.json`、接口信息、aclnn 文档公式节已就绪
   完成标志：pytest 文件已写入，`--cases-file` 在 `conftest.py` 注册，测试模块可 collect
2. Step 6b：执行 pytest collect/run 并写入结果
   前置：Step 6a 完成
   完成标志：`pytest_collect.json` 和 `pytest_result.json` 已写入
3. Step 6c：提取 tiling key 覆盖率
   前置：Step 6b 完成，plog 可获取；无 NPU 时记录 skipped，不伪造通过
   完成标志：`S6_tilingkey_coverage.json` 已写入

## 角色

你负责将 S5_mapped_cases_high.json（已映射的 tensor 构造配置）转化为一个完整的、可直接 `pytest` 执行的测试文件。

## 输入 / 输出

**输入**：
1. `S5_mapped_cases_high.json` — Step 5 产出的已映射参数组合（含 params + tensors，每个 case 已有完整 shape/dtype）
2. `S2P2_param_def.json` — 参数定义（用于理解参数含义，不需要再做过滤）
3. 算子接口信息 — 函数签名、输入输出 tensor 定义、调用方式
4. 算子 aclnn 文档中的「计算公式」— 仅读取该节，作为 reference 实现的唯一依据

**输出**：
1. `conftest.py` — 注册 `--cases-file` pytest 命令行参数。该文件必须与 `S6_test_{op_name}.py` 位于同一目录。
2. `S6_test_{op_name}.py` — 一个完整的 pytest 文件。

## 文件结构要求

生成的 pytest 文件必须包含以下部分，按此顺序：

```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption("--cases-file", action="store", default="S5_mapped_cases_high.json")
```

```python
# S6_test_{op_name}.py

# 1. 导入（NPU 不可用时整个文件 skip，而非假 PASS）
import pytest
import torch
torch_npu = pytest.importorskip("torch_npu")

# 2. PARAMS 列表（通过 --cases-file 参数指定 JSON 来源）
import json, os
_CASES_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. 常量
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

# 4. Reference 实现（仅基于 aclnn 文档「计算公式」一节）
def reference_{op_name}(...):
    """CPU 参考实现。严格按 aclnn 文档计算公式节编写，不读其他节。"""
    ...

# 5. 测试函数
def pytest_generate_tests(metafunc):
    if "p" in metafunc.fixturenames:
        cases_file = metafunc.config.getoption("--cases-file", "S5_mapped_cases_high.json")
        with open(os.path.join(_CASES_DIR, cases_file)) as _f:
            cases = json.load(_f)["cases"]
        metafunc.parametrize("p", cases, ids=lambda c: c["id"])

def test_{op_name}(p):
    tensors = p["tensors"]
    params = p["params"]
    # a. 构造输入 tensor（shape/dtype 直接从 tensors 取值）
    # b. 调用算子（NPU）
    # c. 调用 reference（CPU）
    # d. 断言（shape + dtype + 数值精度）
```

## 关键规则

S5_mapped_cases_high.json 经过 Step 5 mapper 映射 + validate_config 校验，每个 case 的 tensor shape/dtype 均为合法值。不需要 `is_valid_combo()` 或额外的 shape 过滤。直接使用全部 PARAMS。

### Tensor 构造

#### 输入 tensor

shape 和 dtype 直接从 `case["tensors"]["inputs"]` 读取，无需推导。根据每个 tensor 的 `_data_range` 字段选择构造方式（存在 `tensors.inputs.{name}._data_range`）：

```python
def _random_bool(shape):
    return torch.randint(0, 2, shape, dtype=torch.int8).bool()

def _randn_like(shape, dtype):
    if dtype == torch.bool:
        return _random_bool(shape)
    if not dtype.is_floating_point:
        return torch.randint(-10, 11, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)

def _rand_like(shape, dtype):
    if dtype == torch.bool:
        return _random_bool(shape)
    if not dtype.is_floating_point:
        return torch.randint(0, 11, shape, dtype=dtype)
    return torch.rand(shape, dtype=dtype)

def make_data(shape, dtype, data_range):
    """根据 data_range 构造不同值域的 tensor。未指定时返回随机值。"""
    if data_range == "zero":
        return torch.zeros(shape, dtype=dtype)
    elif data_range == "extreme":
        if dtype == torch.bool:
            return torch.ones(shape, dtype=dtype)
        if not dtype.is_floating_point:
            return torch.full(shape, torch.iinfo(dtype).max, dtype=dtype)
        dtype_max = {torch.float16: 65504.0, torch.bfloat16: 3.3895e38, torch.float32: 3.4e38}
        return torch.full(shape, dtype_max.get(dtype, 3.4e38), dtype=dtype)
    elif data_range == "negative":
        if dtype == torch.bool:
            return torch.zeros(shape, dtype=dtype)
        if not dtype.is_floating_point:
            return torch.randint(-10, 0, shape, dtype=dtype)
        return -torch.rand(shape, dtype=dtype) * 10
    elif data_range == "tiny_pos":
        if dtype == torch.bool:
            return torch.ones(shape, dtype=dtype)
        if not dtype.is_floating_point:
            return torch.ones(shape, dtype=dtype)
        return torch.ones(shape, dtype=dtype) * 1e-6
    elif data_range == "all_ones":
        return torch.ones(shape, dtype=dtype)
    elif data_range == "near_zero":
        if dtype == torch.bool:
            return torch.zeros(shape, dtype=dtype)
        if not dtype.is_floating_point:
            return torch.zeros(shape, dtype=dtype)
        return (torch.rand(shape, dtype=dtype) - 0.5) * 0.02
    elif data_range == "with_inf":
        t = _randn_like(shape, dtype)
        if not dtype.is_floating_point:
            return t
        t.view(-1)[0] = float('inf')
        return t
    elif data_range == "with_nan":
        t = _randn_like(shape, dtype)
        if not dtype.is_floating_point:
            return t
        t.view(-1)[0] = float('nan')
        return t
    else:
        return _randn_like(shape, dtype)

tensors = p["tensors"]
params = p["params"]
inputs = {}
for name, spec in tensors["inputs"].items():
    if spec is None:
        inputs[name] = None  # optional tensor 缺失
        continue
    dr = spec.get("_data_range", "normal")
    inputs[name] = make_data(spec["shape"], DTYPE_MAP[spec["dtype"]], dr).npu()
```

禁止对非浮点 dtype 直接调用 `torch.randn` / `torch.rand`。所有 int*/bool 输入必须通过上面的
`_randn_like` / `_rand_like` 分支构造，避免 tensor 构造阶段在执行算子前报错。

**可选输入**：S5 映射中 optional tensor 为 None 时，跳过构造，传 None 给算子。

#### 输出 tensor（预分配场景）

部分算子 API 要求调用方预分配输出 tensor。此时从 `case["tensors"]["outputs"]` 构造：

```python
outputs_prealloc = {}
for name, spec in tensors["outputs"].items():
    if spec is not None:
        outputs_prealloc[name] = torch.empty(spec["shape"], dtype=DTYPE_MAP[spec["dtype"]]).npu()
```

如果算子 API 不要求预分配（API 内部创建输出），则跳过此步。

### 标量属性

从 `case["params"]` 读取。根据算子接口需要提取对应的标量参数：

```python
params = p["params"]
attr_1 = params.get("attr_name_1", default_1)
attr_2 = params.get("attr_name_2", default_2)
```

### 断言

- **shape 检查**：`assert output.shape == expected_shape`
- **dtype 检查**：`assert output.dtype == expected_dtype`
- **数值对比**：逐输出对比，精度校验失败时必须转为 XFAIL（见约束-铁律）：

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

- 如果某些参数组合没有 reference 实现，只做 shape/dtype 检查

### parametrize ids

直接使用 S5 映射中的 `case["id"]`：

```python
@pytest.mark.parametrize("p", PARAMS, ids=lambda c: c["id"])
def test_{op_name}(p):
    ...
```

## 约束

### 铁律

```
NO PYTEST FILE WITHOUT VERIFYING IT CAN BE COLLECTED AND EXECUTED

NO INLINE PARAMS — PARAMS 必须通过 json.load 从 S5_mapped_cases_high.json 运行时读取，
禁止将参数组合硬编码/内嵌到 pytest 文件中。修改 S5_mapped_cases_high.json 后不应需要重新生成 pytest 文件。

NO MODIFIED TOLERANCE — 精度标准（rtol/atol）禁止修改，这是质量保证的最后一道门槛。
精度不达标的 case 通过 XFAIL 记录偏差信息，保留可追溯性。
```

生成 S6_test_{op_name}.py 后，按以下顺序验证：

0. `python -m py_compile conftest.py` — pytest 参数插件语法检查。失败 = 文件有语法错误，必须修复。
1. `python -m py_compile S6_test_{op_name}.py` — 语法检查。失败 = 文件有语法错误，必须修复。
2. `pytest --collect-only S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json` — 收集检查。失败时区分原因：
   - **SyntaxError / NameError** → 文件有问题，修复后重试
   - **ModuleNotFoundError（torch_npu 等）** → 环境缺依赖，不是文件问题，可以继续
   - **其他** → 具体分析
3. `ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py -q --tb=line` — **执行检查 + 日志采集**。目标：所有用例都能合法运行。判定规则：
   - **合法**：PASSED / XFAIL / SKIPPED
   - **非法 — 必须修复**：
     - **FAILED**（任何类型的失败，包括 assertion）
     - **ERROR**（import 失败、API 不存在、kernel crash 等）
     - **RuntimeError**（API 签名错误、shape 不匹配、Tiling 失败等）
     - **AttributeError**（API 名称错误、模块缺少属性等）
     - **假 PASS**（NPU 不可用时静默通过，未做任何实质验证）
   - 精度不达标的 AssertionError **必须**用 `try/except` 包裹 `assert_close`，捕获后调用 `pytest.xfail(reason=...)`，使结果为 XFAIL 而非 FAILED
   - 修复后必须重新运行确认，直到 0 个 FAILED / ERROR / RuntimeError / AttributeError
4. 复制 plog 并生成 tilingkey 覆盖率报告：
   ```bash
   mkdir -p tests/whitebox/tilingkey_logs/
   PLOG=$(ls -t ~/ascend/log/debug/plog/plog-*.log | head -1)
   cp "$PLOG" tests/whitebox/tilingkey_logs/{op_name}_full.log
   python {skill_scripts}/compute_tilingkey_coverage.py \
     --log-path tests/whitebox/tilingkey_logs/{op_name}_full.log \
     --param-def tests/whitebox/S2P2_param_def.json \
     --output-dir tests/whitebox/
   ```
   输出：`S6_tilingkey_coverage.json`（含全局与 per_group 覆盖率）。

### 严格禁止

1. 禁止手动编写参数组合——必须使用 S5_mapped_cases_high.json 中的全部内容
2. 禁止将 JSON 内容内嵌为 PARAMS 列表——必须使用 json.load 从 S5_mapped_cases_high.json 运行时读取
3. 禁止手动推导 tensor shape/dtype——必须直接从 `case["tensors"]` 读取
4. 禁止假设 NPU 环境一定可用——使用 `pytest.importorskip("torch_npu")` 守护
5. 禁止在测试函数中硬编码具体参数值
6. 禁止声称"文件已生成"但没跑 `pytest --collect-only` 验证
7. 禁止声称"文件已通过验证"但没跑 `pytest S6_test_{op_name}.py` 实际执行验证
8. 禁止因输出模式不同而跳过用例——全量解包返回值，按模式验证对应输出
9. 禁止在 prompt 或代码中硬编码算子特定的 tensor 名称、属性名称或输出名称
10. 禁止在 `S6_test_{op_name}.py` 中定义 `pytest_addoption`；pytest 在加载测试模块前解析命令行参数，`--cases-file` 必须在同目录 `conftest.py` 或插件中注册。

### 实战规则（从踩坑经验沉淀）

#### 先探测 API，再写测试

禁止凭文档或直觉猜测 API 名称和签名。编写测试前，必须执行探测：

```bash
python -c "import torch_npu; print(hasattr(torch_npu, 'npu_xxx'))"
python -c "import torch, torch_npu; torch_npu.npu_xxx(torch.randn(1,2).npu())"
```

在代码中用 `hasattr` 守护：

```python
HAS_NPU_OP = hasattr(torch_npu, "npu_xxx") and torch.npu.is_available()
if not HAS_NPU_OP:
    pytest.skip("npu_xxx not available")
```

#### Golden 函数的唯一来源是 aclnn 文档「计算公式」

1. **只读**算子 `docs/aclnn*.md` 文件中的「计算公式」这一节（通常紧跟在"接口功能"之后，以 `$$` 包裹的公式）
2. **不读**同一文档的其他节（函数原型、参数说明、示例代码等）
3. **不推**不从 kernel 源码反推公式——kernel 是参考实现，不是定义
4. 公式中每个变量的含义必须从公式本身推断，不需要读参数说明表
5. **输出-公式映射表**：编写 reference 前，必须为每个输出 tensor 显式标注其对应的公式表达式。格式（以注释形式声明在 reference 函数上方）：
   ```python
   # 输出-公式映射：
   # - {output_name_1} = {公式表达式}     ← 来自公式 {来源小节}
   # - {output_name_2} = {公式表达式}     ← 来自公式 {来源小节}
   # - {output_name_3} = {公式表达式}     ← 来自公式 {来源小节}（未显式定义，经数值验证确认）
   ```
   如果公式节未显式给出某个输出的表达式，必须通过数值交叉验证（见第 6 条）确定，禁止猜测。
6. **数值语义验证**：当公式节未显式定义某个输出的计算式时，用一个小 case 在 NPU 上运行，列出所有候选表达式并逐个计算 diff，选择 diff 最小的作为正确语义。将验证结论写入输出-公式映射表注释中。

#### Reference 必须用真实 NPU 输出交叉验证

先手动跑一个小 case 对比 NPU 输出和 reference 输出，确认一致后再批量生成测试。关注点：
- **输出 shape**：NPU 返回的 shape 是否与 reference 一致（尤其注意广播、未 expand 的中间结果等）
- **输出 dtype**：每个输出的实际 dtype 是否与预期一致（尤其注意混合精度场景下中间输出与主输出 dtype 不同）
- **输出数量**：API 实际返回几个值（可能不等于接口文档描述）
- **输出语义**：每个输出 tensor 的数值是否与输出-公式映射表中声明的表达式一致。禁止只检查 shape/dtype 而跳过数值语义验证。

#### 禁止假 PASS — 没验证就不能报通过

```python
# ✅ 正确：NPU 不可用时 skip，或做了实质验证才 report pass
if not HAS_NPU_OP:
    pytest.skip("NPU not available")
# ... NPU 对比 ...

# ❌ 错误：NPU 对比放在 if 里，else 分支只做了 CPU 属性检查就放行
if HAS_NPU_OP:
    torch.testing.assert_close(npu_out, ref_out)
# 没有 else → CPU 上跑了个 assert shape == expected 就 PASSED（假 PASS）
```

#### 输出含 inf/nan 时，比较必须用 equal_nan=True

边界值输入（`extreme` 取 dtype 最大值）或多步算子会导致中间计算溢出，输出包含 inf/nan，这是正常测试路径。

```python
# ❌ 错误：默认 equal_nan=False，nan != nan 必然失败
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=rtol, atol=atol)

# ❌ 错误：放任 FAILED 不处理，执行检查会拦截
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=rtol, atol=atol, equal_nan=True)

# ❌ 错误：降低精度标准放行（见铁律）
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=1.0, atol=1.0)
```

正确写法已在"断言"小节给出（try/except + equal_nan=True + pytest.xfail）。

#### 可变输出数量的算子 — 全量解包，按模式验证

部分算子的接口支持多种输出模式（如根据可选输出参数是否为 None 决定返回不同数量的输出）。torch_npu Python API 通常固定返回所有输出 tensor。

**禁止做法**：因为当前模式不需要某个输出就跳过整个用例。

```python
# ❌ 错误：某个模式下不需要某输出，直接跳过
if mode != full_mode:
    pytest.skip("mode not supported")
```

**正确做法**：始终用全量变量承接返回值，然后根据模式只验证该模式应有的输出。

```python
# ✅ 正确：全量解包，按模式验证
outputs = npu_op(input_1, input_2, ...)
npu_out_0 = outputs[0]
npu_out_1 = outputs[1] if len(outputs) > 1 else None
npu_out_2 = outputs[2] if len(outputs) > 2 else None

# shape/dtype 检查：所有模式下都检查的输出
assert npu_out_0.shape == expected_shape_0
assert npu_out_0.dtype == expected_dtype_0

# 按模式验证对应输出
if should_verify_output_1:
    assert npu_out_1.shape == expected_shape_1
if should_verify_output_2:
    validate(npu_out_2, ref_out_2)
```

**原则**：
- API 返回几个 tensor 就解包几个，禁止丢弃或跳过
- Reference 始终按最完整模式计算（因为底层实际执行的是完整路径）
- 不同模式通过控制验证哪些输出来区分，而非控制是否执行用例
- 目标：0 skipped，所有枚举预算都用于实际验证

## ✅/❌ 示例

```python
# ✅ 正确：通过 --cases-file 参数指定 JSON 来源，支持 low/high 切换
# conftest.py
def pytest_addoption(parser):
    parser.addoption("--cases-file", action="store", default="S5_mapped_cases_high.json")

# S6_test_{op_name}.py
import pytest, json, os

_CASES_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def cases(request):
    with open(os.path.join(_CASES_DIR, request.config.getoption("--cases-file"))) as f:
        return json.load(f)["cases"]

# ❌ 错误：手写参数
PARAMS = [{"dtype": "float16", "D": 32}]
```

## 用例选择

单个 `S6_test_{op_name}.py` 通过参数控制数据源：
- `--cases-file` 切换数据源：
  - high 数据源：`pytest ... --cases-file=S5_mapped_cases_high.json`
  - low 数据源：`pytest ... --cases-file=S5_mapped_cases_low.json`
- 用例筛选：使用 pytest 内置 `-k` 表达式（基于 `case["id"]`）
  - 单个：`pytest ... -k case00001`
  - 多个：`pytest ... -k "case00001 or case00008"`
  - 与数据源组合：`pytest ... --cases-file=S5_mapped_cases_high.json -k network_00000`

所有逻辑（DTYPE_MAP / TOLERANCE / reference / 测试函数 / 断言 / 约束 / 铁律）完全一致。

## 生成后自审清单

生成 S6_test_{op_name}.py 后，执行以下验证（发现问题就修，修完再报告）：

1. `python -m py_compile conftest.py` 与 `python -m py_compile S6_test_{op_name}.py` — 无语法错误？
2. `pytest --collect-only S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json` — 能收集到用例？ids 是否唯一？
3. `ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json -q --tb=line` — 全部用例合法执行？0 个 ERROR / RuntimeError / AttributeError？
4. NPU 不可用时是否正确 skip 而非假 PASS？
5. plog 是否成功复制到 `tilingkey_logs/{op_name}_full.log`？覆盖率脚本是否生成 `S6_tilingkey_coverage.json`（含全局与 per_group 覆盖率）？
6. 单用例执行后能否从 plog 提取 tiling key？
   ```bash
   grep "Tiling Key:" $(ls -t ~/ascend/log/debug/plog/plog-*.log | head -1)
   ```
