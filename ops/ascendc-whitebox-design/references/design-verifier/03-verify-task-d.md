# Task D 产出验证：D1 + D2 + D3

校验 `S2P2_param_def.json` + `S2P2_gen_cases.py` + `S2P2_traceability.md` 的真实性与一致性。

---

## D1：推导链真实性

Read `S2P2_traceability.md` 每个 group 的两张表，逐行验证源码引用。

**D1.1 触发条件表**：逐行 Read `tiling 源码位置` 列（±2 行），验证该位置确实包含表中的 `条件` 描述的分支逻辑 → **fail** 行号无关

**D1.2 内部变量推导表**：逐行 Read `计算链` 列首个源码引用（±2 行），验证该行包含计算链中提到的变量名和命名常量。行号偏移 ≤5 行 → pass，记录实际行号；>5 行或变量名不存在 → **fail**

---

## D2：param_def 结构合规

验证 `S2P2_param_def.json` 的结构。

**必填顶层字段**：`platform`(string) · `platform_cores`(int) · `tiling_keys`(int[]) · `dtype_tensors`(array[object]，每个含 `tensor`+`param`) · `groups`(array)

**Group 字段**：
- 必填：`id`(string) · `mode`(string) · `per_dtype`(dict) · `constraint_note`(string)
- 可选：`group_dims`(dict[str, array])，每个数组**必须为 10 值** → 长度不符 → **fail**。无 group 级维度时省略该字段

**constraint_note 一致性**：Read 该 group 的 `S2P2_traceability.md` 表，验证 constraint_note 中每个引用数值都能在该 group 的触发条件表或等价推导表「写入位置 = constraint_note」的行中找到对应 → **warn** 不一致（不阻塞，因 constraint_note 为自由文本）

**per_dtype entry 字段**（`per_dtype[{dtype}][*]`）：
- 必填：`path`(string) · `key`(int)
- 条件必填：路由参数（决定代码分支）为 array，同 path 不同 dtype 下取值相同
- 条件必填：dtype-dependent 维度为 array（仅取值因 dtype 不同时出现）
- 维度取值格式按类区分：
  - 路由参数（`{route_param}`）：flat value 数组，元素数 ≥1（如 `["NCHW"]`、`["NCHW", "NHWC"]` 或 `[0, 1]`）
  - 退化维度：1 元素 flat value 数组（如 `[1]`）
  - compound 维度（`{compound_dim}`）：`list[dict]`（multi-key dict，如 `[{"dimA": v1, "dimB": v2}]`）
  - 单 key 维度（`{dim_name}`）：flat value 数组（如 `[64, 128, 256, 512, 1024]`），5 个标量元素
- 每个 per_dtype[{dtype}] 的数组长度必须 == 该 group 内 reachability==reachable 且属于该 dtype 的 path 数量（04-step4-output.md 4.2 校验项 6 已覆盖此规则）

**跨 dtype 多样性**：同 group 同 `{param}` 名下，各 dtype 条目的 mid 值集合两两不重叠 → 存在重叠且该维度在 (lo, hi) 内候选值充足（候选数 ≥ 3 × dtype 数）→ **warn**

**禁止字段**（任意层级）：`t`、`coverage`、`thresholds`、`anchor_dim`、`per_value`、`alignment`、`constraints`(JSON 格式)、`low_configs`、`desc_rules`

---

## D3：gen_cases.py 静态语义匹配

Read 源码理解意图，验证脚本正确从 `S2P2_param_def.json` 读取数据。**不运行脚本生成 cases**。

**D3.1 语法 + JSON 加载**

- `python3 -m py_compile` 返回 0
- Section 1 顶部存在 `import random` + `random.seed(42)`
- 脚本从 `S2P2_param_def.json` 加载数据（`json.load` 调用存在）
- `DTYPE_PARAM` 从 `_param_def["dtype_tensors"][0]["param"]` 读取
- `_a` 由脚本运行时从 JSON 累加计算（遍历所有 group × 所有 dtype 的 per_dtype entry 总数）
- `_default_cap = max(min(10, 100 // _a), 1)` 计算逻辑正确

任一不符 → **fail**

**D3.2 提取函数正确性**

Read Section 2 的 `extract_entry_dims()` 和 `extract_group_dims()` 函数：

| 函数 | 关键语义点 |
|------|-----------|
| `extract_entry_dims(entry)` | 跳过 `path`/`key`；dict list → 直接使用；flat array → `[{field: v} for v in values]` |
| `extract_group_dims(group)` | 读取 `group["group_dims"]`；list 字段 → `[{field: v} for v in values]` |

函数实现偏离描述 → **fail**

**D3.3 工具函数正确性**

Read Section 2 的 3 个工具函数：

| 函数 | 关键语义点 |
|------|-----------|
| `compress_per_dtype(dim_dicts, cap)` | 循环生成 cap 个组合；每维度独立随机选值（固定 seed）；seen 去重；可选值不足 cap 时以实际为准 |
| `compress_group_pool(dim_dicts)` | 多维度独立 shuffle + 同位配对 → min_len 个 combo；单维度直接返回 |
| `shuffled_pool(base, seed)` | 用独立 Random(seed) 打乱副本；返回 (shuffled, 0) |

函数实现偏离描述 → **fail**

**D3.4 生成循环完整性**

Section 3 的通用循环验证：

- 遍历 `_param_def["groups"]`（覆盖所有 group）
- 对每个 group：`extract_group_dims(group)` → `compress_group_pool()` → `shuffled_pool()`
- 遍历 `group["per_dtype"]` 每个 dtype 的每个 entry
- 对每个 entry：`extract_entry_dims(entry)` → `compress_per_dtype()` → 对每个组合从 pool 抽取 1 项
- 池耗尽时用相同 seed 重新打乱

**seed 唯一性**：不同 group 的 seed 值必须两两不同（`seed = (group_idx + 1) * 100` 满足此要求） → 共享 seed → **fail**

缺失任一环节 → **fail**

**D3.5 case dict 结构**

Section 3 每个 case 组装满足：

- 必含字段：`_group`(group ID) · `{DTYPE_PARAM}`(dtype 名称) · `path`(str) · `key`(int)
- per_dtype 维度通过 `**p` 解包（p 来自 compress_per_dtype 输出）
- group 级维度通过 `**gp` 解包（gp 来自 pool）

任一不符 → **fail**

**D3.6 路径覆盖逻辑可达性**

对 path_list 每个 `reachability == "reachable"` 路径，验证脚本的通用循环逻辑上会遍历该路径：脚本遍历 `group["per_dtype"][dtype]` 中的每个 entry，entry 的 `path` 字段即为路径 ID。只要 param_def 中该 group 的 per_dtype 包含该 path 的 entry，脚本即可达 → 不可达 → **fail**

---

## D 整体判定

任一子项 fail → **fail**；全 pass → **pass**。

## 输出格式

```
| ID | 状态 | verified/total | 备注 |
|----|------|---------------|------|
| D1.1 | pass/fail | v/t | |
| D1.2 | pass/fail | v/t | |
| D2 | pass/fail | —/— | |
| D3.1 | pass/fail | —/— | JSON 加载 + _a 计算 |
| D3.2 | pass/fail | —/— | extract 函数 |
| D3.3 | pass/fail | —/— | 工具函数 |
| D3.4 | pass/fail/warn | v/t | 生成循环 + seed 唯一性 |
| D3.5 | pass/fail | —/— | case dict 结构 |
| D3.6 | pass/fail | v/t | 路径覆盖 |
```
