# Step 4：输出

> **前置条件**：Step 1-3 全部完成

写入主 Agent 指定目录。

## 4.1 S2P2_param_def.json

```json
{
  "platform": "{platform}",
  "platform_cores": NN,
  "tiling_keys": [K1, K2, K3, ...],
  "groups": [
    {
      "id": "{group_id}",
      "mode": "{一句话模式触发描述}",
      "{shared_dim}": [{v1}, ..., {v10}],
      "per_dtype": {
        "{dtype_a}": {"path": "P_a", "key": K_a, "{dim}": [{v1}, {v2}, {v3}, {v4}, {v5}]},
        "{dtype_b}": {"path": "P_b", "key": K_b, "{dim}": [{v1}, {v2}, {v3}, {v4}, {v5}]}
      },
      "constraint_note": "{中文自然语言约束说明}"
    }
  ]
}
```

**顶层字段**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `platform` | string | 是 | 目标平台标识 |
| `platform_cores` | int | 是 | 目标平台核数 |
| `tiling_keys` | array[int] | 是 | 所有 reachable 路径对应的 tiling key（去重） |
| `groups` | array | 是 | 按 tiling mode 分组的参数定义 |

**group 字段**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | group 标识 |
| `mode` | string | 是 | 触发模式一句话描述 |
| `per_dtype` | dict | 是 | 按 dtype 绑定 path/key/取值。key 为 dtype 字符串，value 含 `path`/`key`/及各维度取值列表 |
| `{dim}` | array/null | 否 | 影响 tensor shape 的非路由维度取值列表。条件必填：若维度属于 `input_variables`、映射到 tensor 元素但并非该 group 的路由维度，必须写入 group 级字段，每 group 10 值（Po2/Composite/Prime 平衡）。同一维度名可出现在多个 group 中。无此类维度时为 null |
| `constraint_note` | string | 是 | 约束说明。只能引用 param 维度名和具体数值，禁止引用内部变量名。内部变量→param 的等价关系必须写入 traceability |

**per_dtype 内字段**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | S2P1 path ID |
| `key` | int | 是 | tiling key 值 |
| `{dim_name}` | array | 条件必填 | 该 dtype 下某维度的取值列表。维度名与 tiling 源码中的 params 名一致 |

**禁止字段**：`t`、`coverage`、`thresholds`、`anchor_dim`、`per_value`、`alignment`、`constraints`(JSON 格式)、`low_configs`、`desc_rules`。

## 4.2 S2P1_path_list.json 更新

在原始文件基础上添加/更新：顶层 `op_name`/`platform`/`groups`（列表）；每条路径 `group`/`reachability`。

写回格式：conditions 每个元素一行；纯字符串数组每个元素一行；其他 indent=2。必须先 Read 原文件再修改，保留原始路径数据和 source_constraints。

**强制校验（Bash，必须全部通过才能继续 Step 5）**：

用 Bash + python3 读 S2P1_path_list.json 和 S2P2_param_def.json，检查以下 5 项，任一失败则回到 4.2 修正：

1. **reachability 全覆盖** — 每条 path 的 `reachability` 字段非 null，值属于 `reachable`/`api_dead`/`api_warn`/`dead`/`disputed` 之一
2. **数量等式** — `reachable + api_dead + api_warn + dead + disputed` 数量之和 == paths 总数（含 dead_paths）
3. **reachable 必有 group** — 所有 `reachability == "reachable"` 的 path 的 `group` 字段非 null
4. **groups 列表一致** — S2P1 顶层 `groups` 列表与 S2P2 每个 group 的 `id` 完全一致（顺序无关）
5. **无空 group** — 每个 S2P2 group 的 `id` 在 S2P1 中至少被一条 reachable path 的 `group` 字段引用

校验通过后打印各状态数量和 groups 列表。校验不通过则 exit 1，禁止继续。

## 4.3 S2P2_traceability.md

```markdown
# 参数推导可追溯性报告：{op_name} ({platform})

## Group: {group_id}

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `条件表达式` | tiling.cpp:行号 | 简要说明 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `{internal_var} op {V}` | L行号: `计算表达式` | `{param} op {V′}` | per_dtype.{dtype}.{param}（取值边界） |
| `{internal_var} op {V}` | L行号: `计算表达式` | `{param} op {V′}` | constraint_note（约束文字） |
```

- **触发条件表**：tiling 源码中该 group 对应模式的分支判断条件
- **等价推导表**：Step 3 中所有内部变量条件的回溯过程。每行一条内部变量条件；若一个条件同时影响取值列表和约束文字，拆为两行分别记录「写入位置」
- 若 group 无内部变量回溯，等价推导表注明此情况
- **constraint_note 校验**：生成此表后，逐行核验 Step 3 写入的 constraint_note 中所有数值和条件都能在本表「写入位置 = constraint_note」的行中找到对应，确保无中间变量残留
