# Step 5：生成 S2P2_gen_cases.py

> **前置条件**：Step 4 已完成，S2P2_param_def.json 已就绪

## S2P2_gen_cases.py 脚本生成规范

### S2P2_gen_cases.py 脚本模板

```python
#!/usr/bin/env python3
"""{op_name} S2P2 -> S2P2_cases.json（统一 dict 解包，无通用引擎依赖）

Usage: python3 S2P2_gen_cases.py
Output: S2P2_cases.json
"""

import json, os, random

random.seed(42)
K = 2   # 每个 per_dtype pair 搭配的 group 级维度值数
OUT = os.path.join(os.path.dirname(__file__), "S2P2_cases.json")

# ── per_dtype 维度定义（每维独立 list）─────────────────────────────
# 格式：[{"{dim}": {v}}, ...]，每维 5 个值
# 多维度通过 compress_per_dtype() 压缩为单 list；单维度直接使用
_DIM_{group}_{dtype_suffix}_{dim_a} = [{"{dim_a}": {v1}}, {"{dim_a}": {v2}}, ..., {"{dim_a}": {v5}}]
_DIM_{group}_{dtype_suffix}_{dim_b} = [{"{dim_b}": {v1}}, {"{dim_b}": {v2}}, ..., {"{dim_b}": {v5}}]

# ── group 级维度定义（每维独立 list）─────────────────────────────
# 格式：[{"{dim}": {v}}, ...]，每维 10 个值
# 多维度通过 compress_group_pool() 压缩为单 POOL；单维度直接赋值给 POOL_
_POOL_dim_{dim_a} = [{"{dim_a}": {v1}}, {"{dim_a}": {v2}}, ..., {"{dim_a}": {v10}}]
_POOL_dim_{dim_b} = [{"{dim_b}": {v1}}, {"{dim_b}": {v2}}, ..., {"{dim_b}": {v10}}]


def compress_per_dtype(dim_dicts):
    """多 per_dtype 维度 → 单 dict 列表。每个维度轮流做主遍历全部值，其余维随机配对（固定 seed 可复现）。单维度直接返回。"""
    if len(dim_dicts) == 1:
        return list(dim_dicts.values())[0]
    results, seen = [], set()
    dim_names = list(dim_dicts.keys())
    rng = random.Random()
    for i, primary_name in enumerate(dim_names):
        rng.seed(hash(primary_name) % 100000 + i * 31)
        for pv in dim_dicts[primary_name]:
            combo = dict(pv)
            for other_name in dim_names:
                if other_name == primary_name:
                    continue
                combo.update(rng.choice(dim_dicts[other_name]))
            key = tuple(sorted(combo.items()))
            if key not in seen:
                seen.add(key)
                results.append(combo)
    return results


def compress_group_pool(dim_dicts):
    """多 group 级维度 → 单 POOL。各维度独立 shuffle（不同 seed），同位配对。单维度直接返回。"""
    if len(dim_dicts) == 1:
        return list(dim_dicts.values())[0]
    rng = random.Random()
    shuffled = {}
    min_len = min(len(v) for v in dim_dicts.values())
    for name, values in dim_dicts.items():
        rng.seed(hash(name) % 100000)
        s = values[:]
        rng.shuffle(s)
        shuffled[name] = s[:min_len]
    results = []
    for i in range(min_len):
        combo = {}
        for name in shuffled:
            combo.update(shuffled[name][i])
        results.append(combo)
    return results


def shuffled_pool(base, seed):
    """返回打乱后的池和位置指针。每个 group 独立 seed。"""
    rng = random.Random(seed)
    p = base[:]
    rng.shuffle(p)
    return p, 0


# ── per_dtype：压缩单 list ──────────────────────────────────────
# 多维度调 compress_per_dtype，单维度直接赋值
{group}_{dtype_suffix_a} = compress_per_dtype({"{dim_a}": _DIM_{group}_{dtype_suffix}_{dim_a}, "{dim_b}": _DIM_{group}_{dtype_suffix}_{dim_b}})
{group}_{dtype_suffix_b} = _DIM_{group}_{dtype_suffix}_{dim_a}  # 单维度直接赋值

# ── group 级 POOL：压缩单 POOL ──────────────────────────────────
POOL_{group} = compress_group_pool({"{dim_a}": _POOL_dim_{dim_a}, "{dim_b}": _POOL_dim_{dim_b}})

# ── group: 多 dtype group（含 pool 抽样）────────────────────────
g_{group_id} = []
pool, pos = shuffled_pool(POOL_{group}, {seed})
for dtype_entries in [
    ("{dtype_a}", {group}_{dtype_suffix_a}),
    ("{dtype_b}", {group}_{dtype_suffix_b}),
]:
    dtype_name, pairs = dtype_entries
    for p in pairs:                          # **p 解包 per_dtype 维度
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_{group}, {seed})
            gp = pool[pos]; pos += 1
            g_{group_id}.append({"_group": "{group_id}", "{dtype}": dtype_name, **p, **gp})

# ── group: 单 dtype group（含 pool 抽样）────────────────────────
g_{group_id} = []
pool, pos = shuffled_pool(POOL_{group}, {seed})
for p in {group}_{dtype_suffix}:              # dtype 写死，仅循环 per_dtype
    for _ in range(K):
        if pos >= len(pool):
            pool, pos = shuffled_pool(POOL_{group}, {seed})
        gp = pool[pos]; pos += 1
        g_{group_id}.append({"_group": "{group_id}", "{dtype}": dtype_name, **p, **gp})

# ── merge & dedup ────────────────────────────────────────────────
all_cases = g_group1 + g_group2 + ... + g_groupN

ROUTING_KEYS = ["{dim_key_a}", "{dim_key_b}"]   # 所有路由维度名

seen = set()
unique = []
for c in all_cases:
    k = tuple([c["_group"], c["{dtype}"]] + [c[key] for key in ROUTING_KEYS])
    if k not in seen:
        seen.add(k)
        unique.append(c)

# 非路由维度默认值（所有 case 统一）
for c in unique:
    c["{attr_name}"] = {default_val}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

# ── summary ──────────────────────────────────────────────────────
from collections import Counter
cnt = Counter(c["_group"] for c in unique)
print(f"Generated {len(unique)} cases -> {OUT}")
for g in sorted(cnt.keys()):
    print(f"  {g}: {cnt[g]}")
```

### 硬性规则

1. **取值硬编码**。每个 group 的每个维度取值以 Python `list` 形式写在脚本中，不从 JSON 读取。取值来源：`S2P2_param_def.json` 的 `per_dtype` 和 group 级维度字段，在 Task D 阶段转换为硬编码的 dict list。

2. **约束隐含在取值中**。所有路由约束（dtype 依赖、对齐要求、范围互斥）在 **Task D Step 3 选值** 时已消化完毕。脚本中不含 `if` 约束过滤逻辑。

3. **维度分别定义，运行时压缩**。per_dtype 和 group 级维度均以 `[{dim: val}, ...]` 格式分别定义，然后通过 `compress_per_dtype()` / `compress_group_pool()` 压缩为等价单 list / 单 POOL。下游 Mode A/B/C 始终消费单 list + 单 POOL，不感知原始维度数。

4. **去重**。硬编码去重元组（或声明 `ROUTING_KEYS` 列表动态构造），格式为 `(_group, {dtype}, routing_dim1, routing_dim2, ...)`。去重键必须包含所有路由维度，不得遗漏。

5. **零外部依赖**。仅使用 Python 标准库（`json`、`os`、`random`、`collections`）。不 import 第三方包。

6. **自执行**。`python3 S2P2_gen_cases.py` 即可产出 `S2P2_cases.json`，无需命令行参数。

7. **非路由维度默认值**。不出现在 group `per_dtype` 中的变量（如其值为常量的标量属性），在 S2P2_gen_cases.py 中统一设默认值并写入每一条 case：
   ```python
   for c in unique:
       c["{attr_name}"] = {default_val}
   ```
   默认值来自 `S2P1_operator_model.json` 中对应 attribute 的 `default` 字段。

### 压缩函数设计

#### `compress_per_dtype(dim_dicts)` — 轮转主维度

多 per_dtype 维度合并为单 dict 列表。单维度直接返回原 list。

- 每个维度轮流作为主维度：遍历其全部值。
- 主维度遍历期间，其余维度各从中随机选一个值配对（`seed = hash(维度名) % 100000 + i * 31`，i 为当前维度索引）。
- 合并后去重。结果与原始笛卡尔积无路径差异，但规模从 O(v^N) 降至 O(N×v)。

#### `compress_group_pool(dim_dicts)` — 洗牌同位配对

多 group 级维度合并为单 POOL。单维度直接返回原 list。

- 各维度独立 shuffle（不同 seed = `hash(维度名) % 100000`）。
- 取各维度打乱后的同位元素组装 dict。
- 结果池大小 = 各维度中最小长度。合并后结果与原始 cartesian 无路径差异。

### dict 列表定义

所有维度取值统一为 `[{key: val}, ...]` 格式，无论单维还是多维：

**per_dtype 维度定义**（每维独立，5 个值）：
```python
_DIM_{group}_{dtype_suffix}_{dim} = [{"{dim}": {v1}}, {"{dim}": {v2}}, {"{dim}": {v3}}, {"{dim}": {v4}}, {"{dim}": {v5}}]
```

**group 级维度定义**（每维独立，10 个值）：
```python
_POOL_dim_{dim} = [{"{dim}": {v1}}, ..., {"{dim}": {v10}}]
```

**per_dtype 压缩后**（消费方看到的单 list）：
```python
{group}_{dtype_suffix} = compress_per_dtype({"{dim_a}": _DIM_{group}_{dtype_suffix}_{dim_a}})
# 多维度：
{group}_{dtype_suffix} = compress_per_dtype({"{dim_a}": _DIM_{group}_{dtype_suffix}_{dim_a}, "{dim_b}": _DIM_{group}_{dtype_suffix}_{dim_b}})
```

**group 级 POOL 压缩后**（消费方看到的单 POOL）：
```python
POOL_{group} = compress_group_pool({"{dim_a}": _POOL_dim_{dim_a}})
# 多维度：
POOL_{group} = compress_group_pool({"{dim_a}": _POOL_dim_{dim_a}, "{dim_b}": _POOL_dim_{dim_b}})
```

### 生成逻辑：统一模式

#### 模式 A：多 dtype group（含 pool 抽样）

```python
g_{group_id} = []
pool, pos = shuffled_pool(POOL_{group}, {seed})
for dtype_entries in [
    ("{dtype_a}", {group}_{dtype_suffix_a}),
    ("{dtype_b}", {group}_{dtype_suffix_b}),
]:
    dtype_name, pairs = dtype_entries
    for p in pairs:                          # **p 解包 per_dtype 维度
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_{group}, {seed})
            gp = pool[pos]; pos += 1
            g_{group_id}.append({"_group": "{group_id}", "{dtype}": dtype_name, **p, **gp})

#### 模式 B：单 dtype group（含 pool 抽样）

```python
g_{group_id} = []
pool, pos = shuffled_pool(POOL_{group}, {seed})
for p in {group}_{dtype_suffix}:              # dtype 写死，仅循环 per_dtype
    for _ in range(K):
        if pos >= len(pool):
            pool, pos = shuffled_pool(POOL_{group}, {seed})
        gp = pool[pos]; pos += 1
        g_{group_id}.append({"_group": "{group_id}", "{dtype}": dtype_name, **p, **gp})
```

#### 实验性模式 C：无 group 级维度

> **实验性路径**：Mode C 描述的场景（无任何 group 级维度）尚未在实际算子中触发。当前已验证的实现全部使用 Mode A 或 Mode B。只有当 `S2P2_param_def.json` 中该 group 明确无 group 级维度时才允许进入 Mode C，否则必须回到 Mode A/B。

当 group 无任何 group 级维度时，K 倍数循环和 pool 均不适用。直接枚举 per_dtype：

```python
g_{group_id} = []
for dtype_entries in [("{dtype_a}", {group}_{dtype_suffix_a}), ("{dtype_b}", {group}_{dtype_suffix_b})]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        g_{group_id}.append({"_group": "{group_id}", "{dtype}": dtype_name, **p})
```

Mode C 生成后必须增加自检断言：

```python
assert not POOL_KEYS_{group_id}, "Mode C requires no group-level dimensions"
assert len(g_{group_id}) == sum(len(pairs) for _, pairs in dtype_entries_all)
```

### 池化抽样规则

- `shuffled_pool(base, seed)` 函数：返回 `(pool, 0)`，其中 pool 是 base 打乱后的副本
- 每个 group 使用独立 `seed`，不同 group 错开切片实现互补覆盖
- `K` 控制规模：有 pool 时 `总 case 数 ≈ per_dtype 条目总数 × K`（各 dtype 的 per_dtype dict 数量之和 × K）；无 pool 时 K 不参与，case 数 = per_dtype dict 总数
- 池耗尽时调用 `shuffled_pool` 用 **相同 seed** 重新打乱（结果与上一轮完全一致），从头消费。效果等价于对打乱后的池做无限循环采样

### 输出格式

`S2P2_cases.json`：

```json
[
  {"_group": "{group_a}", "{dtype}": "{dtype_val}", "{dim1}": {v1}, "{dim2}": {v2}, "{attr_name}": {default_val}},
  {"_group": "{group_b}", "{dtype}": "{dtype_val}", "{dim1}": {v1}, "{attr_name}": {default_val}}
]
```

- 每条 case 必含 `_group`（对应 `S2P2_param_def.json` 的 group `id`）
- 每条 case 必含 `{dtype}`（数据类型）
- 每条 case 必含所有路由维度（通过 `**` 解包写入）
- 每条 case 必含所有非路由维度的默认值

### 生成后自检

`S2P2_gen_cases.py` 写完后，主 Agent 执行以下自检：

**阶段 1：执行**。`python3 S2P2_gen_cases.py` 返回码为 0。

**阶段 2：完整性**。S2P2_cases.json 中：
- 每个 group 的每类 dtype 至少有一条 case
- 所有 case 的去重键元组无重复
- JSON 语法有效

**阶段 3：一致性**：
- S2P2_gen_cases.py 中的取值列表与 S2P2_param_def.json 的 `per_dtype` 完全一致
- `constraint_note` 描述的约束在取值中全部体现（人工核对——constraint_note 为自由文本，不要求自动化校验）
