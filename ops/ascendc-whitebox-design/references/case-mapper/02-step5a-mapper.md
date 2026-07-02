# Step 5a：构造输入输出空间 → 生成 mapper + verifier

> **前置条件**：S5_mapping_spec.md 已生成（Step 5a-pre 完成）
>
> **职责**：读 S5_mapping_spec.md，按 param_type 逐 tensor 构造输入空间，通过 infershape 规则推导输出空间，生成 S5_case_mapper.py + S5_verify_mapper.py。

## 1. 管道概述

输入输出文件清单见 [00-execution-order.md](00-execution-order.md)。

### 1.1 派发机制

S5_mapping_spec.md 的输入 tensor 和输出 tensor 各节已标注每个 tensor 的 param_type（REQUIRED 或 DYNAMIC）。子 agent 生成代码时，对每个 input/output 变量读其 param_type，选择对应分支：

| param_type | 变量结构 | 构造逻辑来源 |
|-----------|---------|------------|
| REQUIRED | 单 tensor | 本文档内联（各节 REQUIRED 分支） |
| DYNAMIC | TensorList（多 tensor） | 06-dynamic-tensorlist.md（按需读取） |

**无默认行为**：每个 tensor 显式匹配 param_type。混合算子中每个 input/output 独立派发（如 input_a 走 DYNAMIC 分支，input_b 走 REQUIRED 分支）。

### 1.2 核心流程

```
每个 case
  ├── 1. dtype 解析（通用）
  ├── 2. shape 参数解析（通用）
  ├── 3. 输入空间构造（按 param_type 派发）
  │     ├── REQUIRED → 本文档 §3
  │     └── DYNAMIC  → 06-dynamic-tensorlist.md
  ├── 4. 输出推导（通用，读 operator_model infershape 规则）
  ├── 5. 属性采样（通用）
  ├── 6. const input 构造（通用）
  └── 7. 组装返回（格式由 param_type 决定）
```

---

## 2. 共享工具函数

以下 5 项嵌入 S5_case_mapper.py 顶部，不随算子或 param_type 变化。

### _prime_factors
`_prime_factors(n: int) -> list[int]` — 质因数分解，升序。n <= 1 返回 []。

### _decompose
`_decompose(num: int, parts: int, strategy: str = "balanced") -> tuple[int, ...]` — 将 num 分解为 parts 个因子。balanced: 因子 >= 1 尽量接近，不变式 `math.prod(result) == num`。nontrivial: 因子 > 1，不足时降级含 1。parts=0 返回 ()。

### _sample_attr
`_sample_attr(case: dict, spec: dict) -> Any` — case 中有 spec["param"] 则直接取，否则按 spec["sampling"].strategy 采样（log_uniform/uniform/choice/default）。无 sampling 取 spec.get("default")。

### _resolve_param
`_resolve_param(case: dict, name: str, default: Any) -> Any` — case.get(name, default)。

### DTYPE_MAP
DT_* 格式 → Python 格式查表。已是 Python 格式时无害透传。

---

## 3. map_case() — 输入空间构造

```python
def map_case(case, rng=None):
    """
    case: 单条 case dict（字段为算子参数名，来自 S2P2_cases.json 或 S2P2_network_cases.json）
    rng: random.Random, seed 由 load_mapped_configs 设置
    返回: {"params": {...}, "tensors": {"inputs": {...}, "outputs": {...}}}
    """
    if rng is None:
        rng = random.Random()

    # ============================================================
    #  以下内容由子 agent 从 S5_mapping_spec.md 翻译生成
    # ============================================================
```

### 步骤 1：dtype 解析（通用）

```python
    dtype_val = DTYPE_MAP.get(case["{dtype_param}"], case["{dtype_param}"])
```

### 步骤 2：shape 参数解析（通用）

```python
    {param_name} = _resolve_param(case, "{param_name}", {default})
```

### 步骤 3：输入空间构造（按 param_type 派发）

遍历 S5_mapping_spec.md 输入 tensor 的每个 input，读其 param_type 标注，选择对应分支。

#### REQUIRED 分支（单 tensor）

翻译自 S5_mapping_spec.md 输入 tensor 中标注为 REQUIRED 的各节。

**ndim 确定**：

```python
    # 独立范围：rng.randint(lo, hi)
    {tensor}_ndim = rng.randint({lo}, {hi})

    # 有 <= 约束：rng.randint(lo, min(hi, {X}_ndim))
    {tensor}_ndim = rng.randint({lo}, min({hi}, {X}_ndim))

    # = 约束：直接赋值
    {tensor}_ndim = {X}_ndim
```

**ndim 解析顺序**（有跨 tensor 依赖时按以下顺序）：

1. 先解析所有独立 tensor（ndim 范围纯数字，无 le/ge/eq 引用其他 tensor）
2. 再解析有 <=/>=/= 约束的 tensor（此时被引用的 tensor 的 ndim 已确定）
3. 最后处理 sync_with tensor（直接赋值）

**input shape 构造**：

```python
    # product + balanced 分解 -> _decompose(val, N)
    {tensor}_shape = _decompose({param}, {N})

    # 直接取值 -> (val,)
    {tensor}_shape = ({param},)

    # sync_with -> 复制目标 shape
    {tensor}_shape = {X}_shape

    # ndim=min 时，跳过 product 组（空 tuple）
    {tensor}_shape = () if {N} == 0 else _decompose({param}, {N})
```

**组装格式**：

```python
    # 单 tensor：值为 dict
    inputs["{name}"] = {"shape": {tensor}_shape, "dtype": dtype_val}
```

#### DYNAMIC 分支（TensorList）

> **按需读取**：06-dynamic-tensorlist.md §2

DYNAMIC 输入值为 `list[dict]`，每个 dict 描述一个子 tensor。构造逻辑（tensor_count、逐子 tensor ndim/shape、列表级约束、组装格式）见 06。

---

## 4. map_case() — 输出推导（通用，不按 param_type 派发）

读 `S2P1_operator_model.json` 的 `outputs[*]`，按 infershape 规则从输入推导输出。规则与 param_type 正交——REQUIRED 输入是 dict，DYNAMIC 输入是 list[dict]，推导逻辑自动适配。

### tensor_count 推导

读 `outputs[*].tensor_count`：
- `derived_from`：输出 count = 被引用输入的 count（DYNAMIC 输入取 `len(inputs["{name}"])`）
- `param` 控制：`rng.randint(min, max)` 生成（同 06 §2.1 模式 A）
- 无 tensor_count 字段（REQUIRED 输出）：单 tensor，无需 count

### shape 推导

读 `outputs[*].shape.rule`：

| rule | 处理方式 |
|------|---------|
| `same_as_input` | 复制对应输入的 shape。REQUIRED: `inputs["{name}"]["shape"]`；DYNAMIC: 逐子 tensor `[s["shape"] for s in inputs["{name}"]]` |
| `derived` | 子 agent 从 `operator_model.outputs[*].shape.expr` 翻译为 Python 表达式。REQUIRED: 单次计算；DYNAMIC: 逐子 tensor 循环计算 |
| `fixed` | 直接赋值 |

derived 规则代码模板：

```python
    # REQUIRED 输出 + derived：单次计算
    {out}_shape = <expr 翻译，引用 inputs["{src}"]["shape"] 等>

    # DYNAMIC 输出 + derived：逐子 tensor 循环计算
    {out}_shapes = [
        <expr 翻译，引用 inputs["{src}"][i]["shape"] 等>
        for i in range({out}_count)
    ]
```

### dtype 推导

读 `outputs[*].dtype.source`：

| source | 处理方式 |
|--------|---------|
| `input` | 复制对应输入的 dtype |
| `fixed` | 直接赋值 |

### 组装格式

输出格式由输出的 param_type 决定（REQUIRED → dict，DYNAMIC → list[dict]），与对应输入格式一致。

```python
    # REQUIRED 输出
    outputs["{name}"] = {"shape": {out}_shape, "dtype": {out}_dtype}
    # DYNAMIC 输出
    outputs["{name}"] = [
        {"shape": {out}_shapes[i], "dtype": {out}_dtypes[i]}
        for i in range({out}_count)
    ]
```

---

## 5. map_case() — 属性 + const input + 组装返回

### 步骤 5：属性采样（通用）

```python
    {attr_name} = _sample_attr(case, {
        "param": "{key}", "type": "{type}",
        "sampling": {"strategy": "{strategy}", ...}, "default": {default}
    })
```

### 步骤 6：const input 值构造（通用）

```python
    {const_input_name} = [{dim_0_val}, {dim_1_val}, ...]
```

### 步骤 7：组装返回

```python
    return {
        "params": {**case, "{attr}": {attr_val}, ..., "{const_input}": {const_input_val}, ...},
        "tensors": {
            "inputs": {
                # REQUIRED: {"name": {"shape": ..., "dtype": ...}}
                # DYNAMIC:  {"name": [{"shape": ..., "dtype": ...}, ...]}
            },
            "outputs": { ... }
        }
    }
```

---

## 6. validate_config()

```python
def validate_config(cfg, case):
    """L1 校验"""
    errors = []
    t = cfg["tensors"]
```

### REQUIRED 分支（单 tensor）

```python
    if not ({lo} <= len(t["inputs"]["{name}"]["shape"]) <= {hi}): errors.append(...)
    if len(t["inputs"]["{a}"]["shape"]) > len(t["inputs"]["{b}"]["shape"]): errors.append(...)
    if t["inputs"]["{name}"]["shape"][{N}] % {K} != 0: errors.append(...)
```

### DYNAMIC 分支（TensorList）

> **按需读取**：06-dynamic-tensorlist.md §3

校验：tensor_count 范围、逐子 tensor ndim 范围、列表级约束（same_dtype_within_list）、sync_with 一致性。

### 输出校验（通用，不按 param_type 派发）

验证推导结果与 infershape 规则一致。按 rule 分节，每节按输出 param_type 选择对应代码。dtype 校验：source=input 则与对应输入 dtype 比对，source=fixed 则与固定值比对。

#### same_as_input

```python
    # REQUIRED 输出：单 dict 直接比较
    if t["outputs"]["{out}"]["shape"] != t["inputs"]["{src}"]["shape"]:
        errors.append(...)
    if t["outputs"]["{out}"]["dtype"] != t["inputs"]["{src}"]["dtype"]:
        errors.append(...)

    # DYNAMIC 输出：逐子 tensor 比较（先验 count，再验逐子 tensor）
    if len(t["outputs"]["{out}"]) != len(t["inputs"]["{src}"]):
        errors.append(...)
    for i in range(len(t["outputs"]["{out}"])):
        if t["outputs"]["{out}"][i]["shape"] != t["inputs"]["{src}"][i]["shape"]:
            errors.append(...)
        if t["outputs"]["{out}"][i]["dtype"] != t["inputs"]["{src}"][i]["dtype"]:
            errors.append(...)
```

#### derived

子 agent 复用 §4 推导时的 expr，对输出 shape 重新计算并与实际输出比对：

```python
    # REQUIRED 输出：单次计算 + 比对
    expected = <expr 翻译，引用 t["inputs"]["{src}"]["shape"] 等>
    if t["outputs"]["{out}"]["shape"] != expected:
        errors.append(...)

    # DYNAMIC 输出：逐子 tensor 计算 + 比对
    for i in range(len(t["outputs"]["{out}"])):
        expected = <expr 翻译，引用 t["inputs"]["{src}"][i]["shape"] 等>
        if t["outputs"]["{out}"][i]["shape"] != expected:
            errors.append(...)
```

#### fixed

```python
    # REQUIRED 输出
    if t["outputs"]["{out}"]["shape"] != {fixed_shape}:
        errors.append(...)

    # DYNAMIC 输出：逐子 tensor
    for i in range(len(t["outputs"]["{out}"])):
        if t["outputs"]["{out}"][i]["shape"] != {fixed_shape}:
            errors.append(...)
```

### 通用校验

```python
    if cfg["params"].get("{attr}", 0) < {min}: errors.append(...)
    return errors
```

---

## 7. check_dim_diversity()

```python
def check_dim_diversity(mapped_cases, rank_range, rank_constraints=None, count_range=None):
    """检查 ndim 覆盖率，返回 missing 列表（WARNING，不阻塞）"""
```

参数：`rank_range` 从 operator_model inputs[*].rank 提取 (min,max)；`rank_constraints` 从 shape.constraints 识别 ndim 依赖，无依赖传 None；`count_range` 从 inputs[*].tensor_count 提取，仅 DYNAMIC 需要。

### REQUIRED 分支

1. sync_with tensor：标注 skipped，不报 MISSING
2. 约束 tensor（max_le）：按依赖 ndim 分组检查
3. 独立 tensor：检查 rank_range 覆盖

### DYNAMIC 分支

> **按需读取**：06-dynamic-tensorlist.md §4

检查 tensor_count 覆盖率 + 逐子 tensor ndim 覆盖率。

---

## 8. 编排代码（每次照搬）

### load_mapped_configs

```python
def load_mapped_configs(cases_file, seed=42, id_prefix="case"):
    """批量映射：加载 JSON -> 逐 case 映射 -> 校验 -> 返回"""
    rng = random.Random(seed)
    with open(cases_file) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        cfg = map_case(case, rng)
        errs = validate_config(cfg, case)
        if errs:
            print(f"[WARN] case {case.get('_group', '?')}: {errs}")
        cfg["id"] = f"{id_prefix}{len(results):05d}"
        results.append(cfg)

    return results
```

### main

```python
def main(cases_file=None, out_file=None, id_prefix="case"):
    import os
    if cases_file is None:
        cases_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S2P2_cases.json")
    if out_file is None:
        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S5_mapped_cases_path.json")

    mapped = load_mapped_configs(cases_file, id_prefix=id_prefix)
    with open(out_file, "w") as f:
        json.dump({"cases": mapped}, f, indent=2)

    print(f"Written {len(mapped)} cases to {out_file}")
```

子 agent 生成 main() 时，须在末尾调用 check_dim_diversity(mapped, rank_range, rank_constraints, count_range)：

- rank_range：从 S2P1_operator_model.json 的 inputs[*].rank 提取各 tensor 的 (min, max)。
- rank_constraints：从 S2P1_operator_model.json 的 inputs[*].shape.constraints 中识别 ndim 依赖关系，硬编码传入。无依赖关系则传 None。
- count_range：从 S2P1_operator_model.json 的 inputs[*].tensor_count 提取 DYNAMIC 输入的 (min, max)。REQUIRED 输入不需要此参数。

---

## 9. S5_verify_mapper.py 结构

4 层验证，0 fail → PASS，≥1 fail → FAIL（最多 3 轮修复）：

| 层 | 职责 | 规则 |
|----|------|------|
| L1 | 调 S5_case_mapper.validate_config() | — |
| L2 | operator_model 交叉验证（outputs key、dtype/rank 范围） | — |
| L3 | source_constraints 交叉验证 | API 限制 → warn，其他 → fail |
| L4 | NPU e2e 调用算子 API | NPU 不可用 → SKIP |

采样：10 random cases，≤20 时全量。

---

## 10. 闸门

| 结果 | 处理 |
|------|------|
| 0 fail | PASS -> 进入 Step 5b |
| >=1 fail | FAIL -> 回到 map_case/validate_config 修正，最多 3 轮 |
| diversity check 有缺失 | WARNING（不阻塞），缺失信息输出到 stderr |
