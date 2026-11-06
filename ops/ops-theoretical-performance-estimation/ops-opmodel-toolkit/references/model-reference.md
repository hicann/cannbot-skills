# 模型配置参考

字段、调用方式与使用限制见[框架参考](framework/index.md)，其中[DAG 语法](framework/dag-reference.md)
覆盖循环、表达式、signals、pool、版本与搬运数量等详细字段。

本页解释[完整示例](../assets/examples/matmul/README.md)使用的字段。不同执行模式的字段
不能随意混用。新模型采用 `definition/` 目录，不在模型根目录再放同名定义文件。

## 语义与输入

`semantic.yaml` 包含 `dimensions`、`tensors`、`operations`、`outputs`、`layouts`、
`final_state_versions`、`numeric`、`tensor_numeric` 和 `transform_sources`。
张量用 `name/shape/dtype/role` 描述；操作输入输出以 `tensor/version` 引用。
示例的 `contraction` 定义 A(M,K) 与 B(K,N) 到 C(M,N) 的矩阵乘。

`workload.yaml` 的关键字段：

| 字段 | 含义 |
| --- | --- |
| `parameters.required/defaults` | 完整 case 可接受的输入及默认值 |
| `derive` | 按声明顺序计算的命名表达式 |
| `assertions` | 必须成立的几何和分块条件，包含 `expr/message` |
| `semantic_bindings` | case 参数到语义维度的映射 |
| `hardware_fields` | 从硬件字段路径读取的参数 |
| `core_projection` | 已分配任务到逐核执行参数的映射 |

示例使用 `owner: framework`、`task_source: split_spec`、`task_key: [m, n]`，
并把任务的 `m_extent/n_extent` 映射为执行参数 `task_M/task_N`。

## 任务空间

`split.yaml` 的 `kind` 为 `split_spec`。`task_space.axes` 定义迭代维度，
`locals` 计算起点和边界，`variants` 按条件生成任务字段。
`constraints` 描述原子任务和可拆维度，`task_cost` 给出分配任务时使用的工作量。
工作量估计不等于最终仿真周期。

```yaml
task_cost: 2 * m_extent * n_extent * K
```

## 执行配置与 DAG

`operator.yaml` 中，`hardware` 选择发行包中的 profile，`params` 是执行参数，
`buffers` 声明 `name/mem/bytes/slots`，`pipeline.phases` 指定阶段组合。
普通模型的 `run.split_strategy` 包含 `rule/order/granularity/hit_policy`，
这里不填写 `ncores`；完整 case 使用所选硬件的核数。

`dag.yaml` 顶层为 `task_graph` 与 `kernels`。示例中的主要指令：

| 指令 | 字段 | 含义 |
| --- | --- | --- |
| `loop` | `id/dim/range/body` | 命名循环及循环体 |
| `move` | `src/dst/size` | 指定 tensor、memory 和元素形状的搬运 |
| `mmad` | `a/b/c/shape/accumulate` | M×K×N 矩阵乘与累加 |
| `vec` | `name/cycles` | 显式向量服务及其周期成本 |

`pool` 取得目标缓冲区；`use_pools` 声明读取的缓冲区。`kernels.dtypes` 指定张量数据类型。
循环所属 pool 的 `scope/owner` 决定生命周期，不能仅改 slots 数量来绕过依赖。

动态指令字段通过 `expr` 映射传入，例如：

```yaml
size: [{expr: 'min(M_L0, task_M - _m * M_L0)'}, {expr: Kb}]
```

表达式支持当前模型参数、算术以及 `min/max/ceil` 等函数。表达式不是任意 Python，
不能在里面导入模块、读文件或调用自己的函数。

## 成本与搜索

`costs.yaml` 使用 `source/coefficients/native/vf/sensitivity`；显式系数说明数值、
单位、来源及校准状态。空系数表表示该模型未提供额外系数，不表示所有操作成本为零。

`optimize/search.yaml` 的 `parameter_space` 包含 `name/role/tiles/cands`。
`role` 为 `outer` 或 `inner`；`tiles` 指定对应母维；`cands` 指定候选值。
实验可以固定某些参数、只搜索其余参数，见[参数寻优](optimization.md)。
`rewrite_grammar`、`mapping_choices`、`split_choices` 等面向图搜索的声明，
不会自动变成 `dag_run` 或 `split_run` 的实验参数。
