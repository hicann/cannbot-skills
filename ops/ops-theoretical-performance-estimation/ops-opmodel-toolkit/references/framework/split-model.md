# 分核输入与调用接口

分核模型描述完整 case 的任务空间、可拆范围和用户约束。
YAML 示例见[split.yaml](../../assets/examples/matmul/definition/split.yaml)，
运行示例见[参数寻优](../optimization.md)。

## YAML 输入

`kind: split_spec` 下使用 `task_space.axes/locals/variants` 声明任务字段，
`constraints` 声明不可拆单元、顺序和可拆维度，`task_cost` 提供工作量表达式。
任务工作量不等于最终仿真周期；尾块应使用真实尺寸。

实验中指定 `shape/ncores/strategy/search`。`strategy` 的常用字段：

| 字段 | 接受内容 |
| --- | --- |
| `rule` | `greedy/round/contiguous/linear/balanced_greedy/balanced_round` 等当前支持的策略名 |
| `order` | 当前任务空间中已声明的遍历维度 |
| `granularity` | `task` 或 `unit`；不可拆分单元须在模型中声明 |
| `hit_policy` | 缓存访问假设；教程使用 `all_miss` |
| `section` | 模型支持的分段设置 |

策略名是调用选项，需用同一 case 的结果比较。不得依据名字承诺性能收益。
具体策略字段和可接受值以安装版本的帮助和配置校验为准。

## Python hooks

声明式输入不能表达用户任务规则时，在模型目录的 `split_hooks.py` 导出 `SPLIT_SPEC`：

```python
from accel_model.model.split.model import DimensionSpec, SplitConstraints, SplitSpec

def task_space(params):
    return [{'row': row, 'block': block, 'bytes': 32768}
            for row in range(params['rows'])
            for block in range(params['blocks'])]

def task_cost(task, params):
    return task['bytes']

SPLIT_SPEC = SplitSpec(
    task_space=task_space,
    task_cost=task_cost,
    dimensions=DimensionSpec(iter_dims=('row', 'block'),
                             unit_dims=('row',), section_dims=('row',)),
    constraints=SplitConstraints(atom_key=lambda task: (task['row'],),
        ordered_key=lambda task: (task['block'],),
        splittable_dims=('block',), iter_dims=('row', 'block')),
).validate()
```

该片段仅定义示例用户任务，需要配套 workload、DAG 和 case。
`task_space(params)` 返回任务映射列表；`task_cost(task, params)` 返回标量工作量。
维度和约束必须与任务字段一致，不能借由估计工作量省略真实任务。
存在 `split_hooks.py` 时需提供合法 `SPLIT_SPEC`；不需要 hook 时使用 YAML。

## 调用与输出

```text
python -m accel_model.app.cli.split_run MODEL --config EXPERIMENT.yaml --out OUTPUT --verify-simpy
```

输出包括方案 `split_plan.rankN.yaml`、排名、assignment、evaluation 和可读报告。
`--verify-simpy` 的检查状态属于软件复核，不能当作设备验收。
采纳方案时取其 `split` 映射作为 `run_native_case(strategy=...)` 的输入，
示例见[Python API](../python-api.md)及[快速入门](../quickstart.md)。

## 结果限制

核对任务覆盖、各核任务和周期、`feasible/adoptable`、额外归约耗时与结果 scope。
不能把单核或代表性工作量的耗时当作完整算子耗时。
`refine_engine` 可选择 `replay/simpy_independent/simpy_shared`；旧 `simpy` 名称不再接受。
独立逐核结果和共享全核结果的范围不同，不直接混排。
缓存命中是假设或模型结果，不能仅凭 `hit_policy` 宣称设备命中。
