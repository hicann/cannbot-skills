# 候选比较、图搜索与会话接口

以完整模型目录、实验 YAML 和当前版本支持的动作作为输入。
使用命令入口完成实验；生成的候选和检查点按产物保存，不手工拼装或改写其内容。

## 候选比较输入

```yaml
kind: comparison
evaluation: {scope: all_nonempty_cores}
baseline: {name: baseline, path: ../baseline}
candidates:
  - {name: candidate_a, path: ../candidate_a}
cases:
  - {name: default, params: {}, weight: 1}
cache_model: shared_estimated
objective: {metric: weighted_geomean_ratio}
```

路径相对实验 YAML；`params` 填写模型所需的完整 case 输入。
`all_nonempty_cores` 要求模型具有有效的分核和逐核工作量声明。
`native_workload` 是另一种评价范围，不与完整 case 的结果混用。

```text
python -m accel_model.app.cli.graph_search --experiment EXPERIMENT.yaml --output OUTPUT
python -m accel_model.app.cli.graph_search --actions MODEL
python -m accel_model.app.cli.graph_search --help
```

`--actions` 返回当前模型可供尝试的动作及标识。需要应用动作时使用
`--apply-action MODEL --action-id ID --evidence TEXT --expected-gain TEXT --added-cost TEXT --output NEW_MODEL`。
输入必须是对应父模型的动作标识；记录实际变更及实验依据。
动作列表不是完成搜索的证据，动作被接受也不意味着候选通过全部验收。

上例显式选择兼容目标。用户公式与排序使用支持的[外部目标接口](objectives.md)；
`parameter_search` 则要求 `per_case_min_cycles`，不能直接套用 provider。
同一规则跨输入选择见[规则接口](policies.md)，同形状不同数据见[场景接口](runtime-control.md)。

## 有限存储与布局候选

需要声明式接口发现候选时，可使用以下公开调用。输入是用户模型目录、该模型参数及
有限选择域；返回可选项与适用性记录，不表示搜索或验证已经完成。

```python
from accel_model.optimize.graph.global_search.discovery_native import NativeSeed
from accel_model.optimize.graph.global_search.discovery import discover_sites

seed = NativeSeed("work/my_matmul", {})
discovery = {
    "mechanisms": ["reuse", "address_layout"],
    "slots": [1, 2], "alignments": [32],
    "max_changes": 2, "max_sites": 64,
}
sites, applicability = discover_sites(seed, discovery)
print(sites, applicability)
```

`construct_discovered(seed, choices, discovery)` 来自同一 `discovery` 模块，返回
候选与报告；`choices` 只能引用本输入返回的选项。普通模型候选可用
`candidate.materialize(新目录)` 导出，再调用支持的完整 case 与数值验证入口。
空选择保留基线；发现域为空不表示找到优化方案。

| 选择 | 输入与使用限制 |
| --- | --- |
| `reuse/address_layout` | 同 owner 的固定 UB/L1 分配、明确生产者和所有消费者；声明槽数与对齐范围 |
| 有限循环/条件中的存储复用 | 循环和条件由参数确定，访问与版本明确；数据相关运行时分支、无界循环或隐含跨 task 初值不受支持 |
| `residency/operand_residency/accumulator_residency` | 使用受支持模型的显式存储、版本、搬运及累加契约；不同入口的支持域不互换 |
| `physical_layout` | 适用的完整 UB/L1 二维视图可选择转置、行 padding 或显式 NZ；任意部分 NZ 视图及图侧 NZ 自动候选不在此范围 |

有限域与预算以本次输入及报告为准，超限或不支持的组合保留失败原因。
选项被接受、候选实际发生变化、完整执行通过、数值通过是不同状态。
某选择已在基线生效或没有改变模型时，不重复计作新增覆盖。

`discover_sites` 的第二个返回值为适用性诊断列表。`physical_layout` 遇到访问契约
缺失、owner／生产者不满足或局部视图维数不支持时，保留 `unsupported`、program、
可定位时的 allocation 及 reason；无合法选项记录 `not_applicable`。部分程序失败
仍可返回其他程序的合法 `sites`，空候选集合不能单独证明某类布局完全不受支持。

## 输出与采纳

实验报告包含基线、候选、逐 case 评价、排名、覆盖和失败原因；可用字段取决于入口。
`ranking/design_winner` 表示当前成本与范围下的模型实验结果。
`validated_ranking/validated_winner`、`prediction_ranking`、`recommendation` 等状态
分别读取，不能用一个较低周期替代未完成的检查。

候选与基线必须具有可比较的输入、数学语义、硬件、成本、后端和 scope。
来源不明或未知成本的候选不作为有效性能依据。采纳导出目录后重新运行原 case；
模型数值通过不等于设备正确或已获得设备加速。
构造快照绑定源模型、参数和硬件，换输入后从源模型与选择规则重新生成；不要修改
导出文件中的绑定或签名来复用旧快照。多 case 逐个保留基线、候选及实际适用范围。
原有 `optimize.dag.reuse` 分析 CLI 与这里的候选接口不同，其限制见[能力矩阵](../capabilities.md)。

## 搜索会话

`python -m accel_model.app.cli.agent_search --help` 提供
`init/propose/register/evaluate/status/note/retain/defer/close` 等操作。
会话输入包括 `requirements`、带 `params/weight` 的 `cases`、`objective` 和 `budget`。
会话目录位于 `OPMODEL_WORKSPACE` 下，并与候选模型目录分开。
完整可运行流程见[联合搜索示例](../joint-optimization/wheel-commands.md#图动作与搜索会话)。

`register` 保存本次候选；后续编辑原目录不会自动更新已登记候选。
`status` 用于查看尝试、失败、排除项和当前结果；登记或恢复会话本身不算性能评价。
结束时用真实原因调用 `close`，预算耗尽和暂停不等于达到优化目标。

## 使用限制

不同入口接受的实验类型、参数及模型组合不同，先核对帮助及错误信息。
参数搜索不能视为任意结构搜索；未尝试的候选族及预算未覆盖项应如实报告。
存储、控制、物理执行或 Python hooks 不受某动作支持时，保留拒绝原因并选择可用实验。
不能修改检查点签名或删除模型约束来绕过校验。
