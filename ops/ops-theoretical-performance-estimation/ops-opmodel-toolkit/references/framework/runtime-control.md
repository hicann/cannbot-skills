# 场景与有界运行时控制

按控制值来源选择接口。调用前参数使用[统一规则](policies.md)；冻结输入可以影响
受支持的原生 DAG 条件；执行中间结果驱动的控制使用适用图模型的有界声明。
某个后端支持不表示全部算子已接入。静态模型无需增加无关动态检查。

## 场景输入

会话契约的 case 可增加 `scenario`，同 shape 不同数据分别登记：

```yaml
cases:
  - name: zero
    params: {rows: 1, columns: 2}
    scenario:
      inputs: {lhs: [0.0, 0.0], rhs: [1.0, 2.0]}
  - name: nonzero
    params: {rows: 1, columns: 2}
    scenario:
      inputs: {lhs: [3.0, 4.0], rhs: [1.0, 2.0]}
```

输入名、扁平数组长度、shape/dtype 必须匹配模型的语义和数值绑定，默认最多
100000 个元素。也可提供 `generator: {path, function, config, dependencies}` 和
整数 `seed`；用户函数签名为 `generate(*, seed, config) -> dict[str, list]`。
保留实际输入、生成器及依赖身份；恢复使用冻结输入，不重新生成。数据改变后重新
核验评价与路径身份；仅目标变化按[重评分接口](objectives.md)复用有效观察。

原生规则要求恰好一个已有 `validation.execution_cases` 绑定模板；旧的无数值绑定
全核估算入口拒绝场景。受支持的原生条件可写为
`{input: mask, index: {expr: "group * 2 + ptid"}, comparison: ne, threshold: 0}`，
输入决定的 `range` 可用 `{input: lengths, index: ptid}`。它们只读取冻结输入；
`skip_if` 保持跳过当前 body 的含义，不表示支持读取执行中的任意中间值。

## 图模型的控制声明

从 `accel_model.model.graph.model.control_flow_ir` 导入下列公开类型，绑定完整模型
中的任务名与值版本；片段自身不构成可运行模型。

| 类型 | 输入契约 |
| --- | --- |
| `RuntimeValueRef(value, index=0, comparison='ne', threshold=0.0)` | 带版本的值、元素索引、`eq/ne/lt/le/gt/ge` 与有限阈值 |
| `GuardedRegion(name, condition, then_tasks, else_tasks=())` | 两分支任务 tuple 互斥，允许嵌套，不允许交叉 |
| `BoundedLoop(name, iterations, max_iterations)` | 有限个 GuardedRegion；迭代使用不同任务与值版本，首次退出后不再执行后续迭代 |
| `StateSelect(task, condition, true_value, false_value, prefix=())` | 对应 `state_select` 任务；两个输入形状匹配，只读选中来源，prefix 为适用的前置继续条件 |
| `ControlFlow(regions=(), loops=(), selects=(), max_nodes=10000)` | 有限控制声明，各集合使用 tuple；绑定 `SemanticGraph/TaskGraph.control_flow` |

每次动态评价绑定一个实际场景。当前支持完整 GM 条件发布、整指令完成和动态租约；
无界循环、时序竞争，以及固定 storage／SS、部分就绪／源读完、重叠接收或原子写与
运行时分支的组合不在支持域。分支标量成本未校准，数值与路径通过不授予完整成本
或设备资格。状态选择的工作区仍受容量检查，不能因选择假分支而假定无需工作区。

## 输出和验证范围

查看实际结果的 `executed_instruction_count`、`skipped_instruction_count` 及
`runtime_control.path`；跳过记录只说明覆盖，不计为实际计算、搬运或 pipe 服务。
路径对应本次实现、数据、数值模型、条件决策和时间；序列化后回放须使用对应接口，
不手改路径、签名或跳过标记。条件观察可早于指令开始，不能据两者不相等判断错误。

按任务语义核对真／假分支、适用的零次循环、提前退出、嵌套、输出／最终状态、
完成／跳过覆盖及资源归还。单路径通过不代表所有输入，缺覆盖项明确列出。
比较两次 trace 时区分路径或工作量变化与同路径性能变化，不以静态任务数当执行次数。
静态连接图和已声明控制值不能代替这些运行证据。

## 有状态区域的候选输入

有限向量／状态区域可使用
`accel_model.optimize.graph.global_search.construction_regions` 的
`ConstructionRegion` 和 `construct(base, regions, decisions, scenario_path=None, *, vector_costs=())`。
区域声明任务、全部外部 `inputs/outputs`、有序 `states` 版本链、`frozen` 任务、
允许的 AIV `owners`、正整数 `slots` 及 `GM/UB` 的 `memories`；
`owns` 为 `('placement', 'storage', 'issue')`。决策引用该声明的有限选择域，
返回 `CandidateState`，仍需运行验证；不通过编辑执行产物替代模型输入。
冻结边界适用于完整候选，不能由其他改动越过；冻结不保证竞争或完成时间不变。
固定状态区域与运行时分支组合尚未开放，保留拒绝原因和未覆盖范围。
