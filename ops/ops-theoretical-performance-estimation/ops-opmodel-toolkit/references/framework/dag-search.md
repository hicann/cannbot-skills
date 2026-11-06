# DAG 搜索接口

用于已有固定 DAG 的参数、slots 与阶段方案实验。先验证基线模型，再声明本次输入和搜索范围。
完整可运行命令见[参数寻优](../optimization.md)及[联合搜索示例](../joint-optimization/wheel-commands.md)。

## 输入

```yaml
name: dag_demo
shape: {M: 256, N: 256, K: 128}
phases: C1
tiling: {Mb: 128, Nb: 128, Kb: 128, M_L0: 64, N_L0: 64}
search: {top_k: 3, replay_top: 3}
```

参数名必须属于当前模型。`phases` 可给出阶段方案或 `search`；`tiling` 固定指定参数，
其余参数须具有合法的候选声明。`slots` 可显式给出模型支持的缓冲数量。
所有参数都固定时属于单点评估，不能作为参数搜索结果报告。

## 调用

```text
python -m accel_model.app.cli.dag_run MODEL --config EXPERIMENT.yaml --out OUTPUT
```

输出位于 `OUTPUT/<name>/`。每轮使用新的实验名或输出目录。
阶段与 slots 的联合候选也可用 `joint_search` 接口，完整调用见
[wheel 命令](../joint-optimization/wheel-commands.md#显式联合搜索)。

| 联合搜索输入 | 含义 |
| --- | --- |
| `phase_candidates` | 非空阶段方案列表，每个方案完整覆盖声明的 stage |
| `parameter_candidates` | 非空参数映射列表，须满足模型输入契约 |
| `issue_policies` | 可选值 `default/startup_first/drain_first/startup_drain`；在报告中保留实际选择 |
| `top_k` | 返回结果数量上限 |
| `max_evaluations` | 本轮评价预算 |

## 用户搜索 hooks

`ms_hooks.py` 属于用户模型目录。需要时从[示例](../../assets/examples/hooks/ms_hooks.py)起步。
存在该文件时显式提供 `SEARCH_DIMS`，不要假定仍自动使用 YAML 搜索空间。

```python
SEARCH_DIMS = [
    {'name': 'M_L0', 'role': 'inner', 'tiles': 'Mb',
     'cands': [32, 64, 128], 'desc': 'M 方向计算块'},
]

def phase_filter(phases):
    return len(phases) == 1
```

`name` 是模型参数；`role` 接受 `outer/inner`；`tiles` 是对应母维参数；
`cands` 接受显式列表、`pow2` 或 `{mult: 正整数, max: 上限}`；`desc` 为可选说明。
候选值必须满足母维、对齐、容量及模型约束；实际覆盖以返回记录为准。
`phase_filter(phases) -> bool` 声明用户允许的阶段组合，只用于阶段搜索。
可选 `validate_grid(rng, cfg) -> list` 提供验证输入。

## 输出含义

常用产物为 `RANKED.txt`、候选 `operator.yaml`、`dag_optimal.yaml`、`PLAN.md`、
trace 和统计表。搜索运行还可产生 `search_audit.json`；单点评估不保证存在覆盖记录。
联合搜索返回 `rows/audit/coverage`，候选包含请求参数、生效参数、周期和配置。

- `rows` 长度受 `top_k` 限制，不代表所有尝试数。
- `coverage` 与审计状态用于区分完成评价、失败和预算未覆盖项。
- `invalid_contract/capacity_invalid/dependency_cycle/resource_deadlock` 等失败状态不构成有效性能结果。
- `budget_skipped` 表示未完成评价；部分执行周期不能用于比较。

## 使用限制

只比较输入、硬件、成本及 scope 一致的结果。候选先应用到模型副本，再用完整 case 复评。
搜索结果不保证全局最优；固定 DAG 参数搜索不表示已经探索所有图结构。
通过固定配置复核不表示数学数值验收或设备性能验收完成。
不支持的节点、配置组合或 hooks 应保留错误和最小输入，不能删去真实约束获得更短周期。
