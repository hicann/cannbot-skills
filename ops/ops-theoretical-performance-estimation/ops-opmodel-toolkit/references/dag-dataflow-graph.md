# 数据流图

输入是用户模型的 `definition/dag.yaml`、`operator.yaml`，以及需要说明的数学语义。
根据公开声明整理 Mermaid 或 ASCII 图，结果保存到用户项目，例如 `work/dataflow.md`。

## 读取与输出

1. 核对 `task_graph.nodes`、`kernels` 与 `pipeline.phases` 的 stage 名称。
2. 从 move 的 src/dst、mmad 的 a/b/c、vec 的显式读写声明中整理用户计算主链。
3. 根据 `pipeline.dataflow` 的 from/to/output/input 标出跨 stage 边，区分数据和控制。
4. 输出图、输入文件身份、范围及无法确定的连接；分清声明的所有可能路径与本次实际执行路径。

字段说明见[DAG YAML](framework/dag-reference.md)。只有 tensor 名称相似或信号同名，
不足以认定生产者与消费者对应；还需核对 stage 端点、版本和声明。
图表示用户模型的输入关系，不推断工具的执行次序。

存在[运行时控制](framework/runtime-control.md)时标注条件值版本、分支／循环与状态汇合，
只按本次场景和路径证据标明执行或跳过；没有路径证据时保持声明图，不推断实际次数。
跳过节点不计服务量，单条路径不能代表全部输入。

## 验证调用与限制

使用[完整 case 入口](python-api.md)运行模型，读取实际检查状态。
包含物理执行声明时使用[物理模型入口](framework/physical-execution.md)。
当前工具集不提供单独的数据流图生成或静态连接检查命令，图由工作流根据输入整理。

连接核对不证明数值、容量、输出覆盖或运行时正确性。动态 hooks、未说明的 vec 读写、
物理 owner 与版本关系无法从现有输入确认时，在图中保留缺口，不添加猜测边。
实际运行等待的解读见[空泡分析](dag-bubble-opt.md)。
