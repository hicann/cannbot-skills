# DYNAMIC TensorList 构造方案

> **适用条件**：当 input/output 的 param_type 为 DYNAMIC 时，子 agent 按需读取本文件。
>
> **与 02 的关系**：本文件描述 DYNAMIC 专属的构造/校验/diversity 逻辑。通用步骤（dtype 解析、属性采样、const input、编排代码）见 02-step5a-mapper.md，本文件不重复。空 tensor 处理已移至 5c（见 04-step5c-merge-expand.md）。

## 1. 数据来源

DYNAMIC 构造的全部约束来自 `S2P1_operator_model.json` 的对应字段，子 agent 不得从源码重新推断：

| 信息 | operator_model 字段 |
|------|---------------------|
| tensor count 控制方式 | `inputs[*].tensor_count` |
| dtype 合法值 | `inputs[*].dtype.values` |
| ndim 范围 | `inputs[*].rank.min / rank.max` |
| 列表级 shape 约束 | `inputs[*].shape.constraints` |
| 输出 count 推导 | `outputs[*].tensor_count.derived_from` |
| 输出 shape 规则 | `outputs[*].shape.rule` + `shape.input` |
| 输出 dtype 来源 | `outputs[*].dtype.source` + `dtype.input` |
| 输出 rank 来源 | `outputs[*].rank.source` + `rank.input` |

---

## 2. 输入构造

### 2.1 tensor_count 生成

读 `operator_model.inputs[*].tensor_count`，按两种模式处理：

**硬件约束**：Ascend C DYNAMIC TensorList 最多容纳 50 个子 tensor。若 `operator_model` 中 `tensor_count.max > 50`，实际采样上限取 `min(max, 50)`。

**模式 A：param 控制（有 param/min/max 字段）**

count 范围 [min, min(max, 50)]。优先从 case dict 读取（网络用例预设值），否则由 `rng.randint(min, min(max, 50))` 生成。跨 case 覆盖由 §4 diversity check 监督。

```python
    if "{name}_count" in case:
        {name}_count = case["{name}_count"]
    else:
        {name}_count = rng.randint({min}, min({max}, 50))
```

**模式 B：derived_from（有 derived_from 字段）**

```python
    {name}_count = {target}_count  # 直接赋值，不随机
```

### 2.2 逐子 tensor ndim 生成

```python
    {name}_ndims = [rng.randint({rank_min}, {rank_max}) for _ in range({name}_count)]
```

各子 tensor 的 ndim 可独立选择（除非有 same_rank_within_list 约束）。

### 2.3 逐子 tensor shape 构造

约束：各维度 >= 0；单 tensor numel <= 10,000,000；维度间比值 <= 32；totalElements 不超过 tiling 约束上限（从 S5_mapping_spec.md 读取）。

若有 shape 构造参数：用 `_decompose` 分解。若无 shape 构造参数：按以下算法动态生成。

```python
    # {name}，无 shape 构造参数，动态生成
    # 约束：numel <= MAX_NUMEL，各维度 >= 2，维度间比值 <= RATIO_MAX
    MAX_NUMEL = 10_000_000
    RATIO_MAX = 32
    {name}_shapes = []
    for i in range({name}_count):
        ndim = {name}_ndims[i]
        if ndim == 0:
            {name}_shapes.append(())  # 标量 tensor
        else:
            target = rng.randint(2 ** ndim, MAX_NUMEL)
            gm = int(target ** (1.0 / ndim))
            lo = max(2, gm // int(RATIO_MAX ** 0.5))
            hi = max(lo + 1, min(gm * int(RATIO_MAX ** 0.5), MAX_NUMEL))
            dims = [rng.randint(lo, hi) for _ in range(ndim)]
            rng.shuffle(dims)
            prod = math.prod(dims)
            if prod > MAX_NUMEL:
                scale = (MAX_NUMEL / prod) ** (1.0 / ndim)
                dims = [max(2, int(d * scale)) for d in dims]
                while math.prod(dims) > MAX_NUMEL:
                    idx = dims.index(max(dims))
                    dims[idx] = max(2, dims[idx] - 1)
            {name}_shapes.append(tuple(dims))
```

### 2.4 列表级约束处理

读 `operator_model.inputs[*].shape.constraints`，逐条处理：

| constraint type | 含义 | 处理方式 |
|----------------|------|---------|
| `same_dtype_within_list` | 列表内所有子 tensor dtype 相同 | 使用同一个 dtype_val（步骤 1 已解析），天然满足 |
| `same_shape_within_list` | 列表内所有子 tensor shape 相同 | 生成 1 个 shape，所有子 tensor 复用 |
| `sync_with` | 与另一个 DYNAMIC 输入同步 | count/shape/dtype 复制目标输入的值 |

```python
    {name}_count = {target}_count
    {name}_shapes = list({target}_shapes)
    {name}_dtype = {target}_dtype
```

### 2.5 组装格式

DYNAMIC 输入的值为 `list[dict]`，每个 dict 描述一个子 tensor：

```python
    # DYNAMIC 输入组装
    inputs["{name}"] = [
        {"shape": {name}_shapes[i], "dtype": dtype_val}
        for i in range({name}_count)
    ]
```

---

## 3. L1 输入校验

逐 DYNAMIC 输入校验，每条约束翻译为 assert 风格检查：

```python
    # tensor_count 范围（param 模式）
    if not ({min} <= len(t["inputs"]["{name}"]) <= min({max}, 50)):
        errors.append(f"{name} count {len(...)} out of [{min}, {max}]")

    # 逐子 tensor ndim 范围
    for i, sub in enumerate(t["inputs"]["{name}"]):
        if not ({rank_min} <= len(sub["shape"]) <= {rank_max}):
            errors.append(f"{name}[{i}] ndim {len(sub['shape'])} out of [{rank_min}, {rank_max}]")

    # same_dtype_within_list 约束
    dtypes = set(sub["dtype"] for sub in t["inputs"]["{name}"])
    if len(dtypes) > 1:
        errors.append(f"{name} has inconsistent dtypes: {dtypes}")

    # sync_with 约束（若有）
    if len(t["inputs"]["{name}"]) != len(t["inputs"]["{sync_target}"]):
        errors.append(f"{name} count != {sync_target} count")
```

> 输出校验见 02-step5a-mapper.md §6 输出校验节（通用，不按 param_type 派发）。

---

## 4. diversity check

### 4.1 检查项

DYNAMIC 输入需检查两项覆盖率：

| 检查项 | 范围来源 | 覆盖要求 |
|--------|---------|---------|
| tensor_count 覆盖率 | operator_model tensor_count.min / min(max, 50) | min 和 min(max, 50) 至少各出现 1 次 |
| 逐子 tensor ndim 覆盖率 | operator_model rank.min/max | [min, max] 内每个值至少出现 1 次 |

### 4.2 代码模板

```python
    counts = [len(c["tensors"]["inputs"]["{name}"]) for c in mapped_cases]
    count_min, count_max = count_range["{name}"]
    missing_counts = [v for v in range(count_min, count_max + 1) if v not in counts]
    if missing_counts: print(f"[DIVERSITY] {name} tensor_count: MISSING {missing_counts}")

    all_ndims = [len(sub["shape"]) for c in mapped_cases for sub in c["tensors"]["inputs"]["{name}"]]
    rank_min, rank_max = rank_range["{name}"]
    missing_ndims = [v for v in range(rank_min, rank_max + 1) if v not in all_ndims]
    if missing_ndims: print(f"[DIVERSITY] {name} ndim: MISSING {missing_ndims}")
```

sync_with 输入标注 `sync_with {target} (skipped)`，不单独报 MISSING。
