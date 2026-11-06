# DAG YAML 接口

`dag.yaml` 声明用户模型的计算、搬运、循环、条件与依赖。
以下片段需要配套 `operator.yaml` 的 stage、参数、buffer 和硬件定义。
可运行文件见[矩阵乘 DAG](../../assets/examples/matmul/definition/dag.yaml)。

## 文件结构

```yaml
task_graph:
  nodes: [V0]
  edges: []
  implementation: {}
kernels:
  dtypes: {'*': bf16}
  V0:
    body:
      - vec: {name: compute, cycles: 8, id: compute}
    signals:
      ready: $compute
```

`nodes` 与 `pipeline.phases` 中的 stage 一致。完整语义模型还需填写
`task_graph.implementation` 对数学操作的覆盖关系。
`kernels.dtypes` 是共享 dtype 表，stage 可给出自己的覆盖值；按 tensor 名、memory 名
或 `*` 声明，不能依靠张量名称猜测数据类型。
`body` 和 `then` 必须是 YAML 列表，条目使用 `-`。

## 搬运 move

```yaml
- move:
    id: load_x
    src: {tensor: X_GM, mem: GM}
    dst: {tensor: X_L1, mem: L1}
    size: [{expr: rows}, 64]
    pool: x_l1
```

| 字段 | 输入含义 |
| --- | --- |
| `src/dst.tensor` | 源、目的张量名称 |
| `src/dst.mem` | `GM/L1/L0A/L0B/L0C/UB` 中的合法路由端点 |
| `size` | 实际元素形状，尾块使用有效大小 |
| `id` | 可供 signals 或依赖引用的标签 |
| `pool` | 本操作写入的 buffer 名称 |
| `use_pools` | 本操作读取的 buffer 名称列表 |
| `after` | 必须满足的显式依赖 |
| `tier` | 选定硬件路由支持的档位；不证明设备缓存命中 |
| `aiv_presplit` | 工作量是否已是单 AIV 的实际份额 |

路由须存在于所选硬件。源、目的 dtype 转换时分别核对两端数据量；不要手工缩小
逻辑形状来表示亚字节 dtype。自定义 Python 搬运见[Stage 接口](../new-operator/python-stages.md)。

## 矩阵乘与累加

```yaml
- {mmad: {a: {tensor: A, mem: L0A}, b: {tensor: B, mem: L0B},
          c: {tensor: C, mem: L0C}, shape: [64, 128, 64],
          accumulate: {expr: k > 0}, pool: c, use_pools: [a, b]}}
```

`a/b/c` 的内存分别为 L0A/L0B/L0C；`shape` 顺序为 M、K、N。
K 分块首块初始化，后续块按数学语义声明累加。最终结果须有实际搬出与消费者。
输入 dtype 与累加器 dtype 分别声明，成本口径见[计算 dtype](../hardware.md#计算-dtype)。
`pool/use_pools` 的内存与操作数一致，缓冲大小使用实际 dtype。
`unitflag: true` 是可选同步声明，需确认当前硬件、模型组合与数值模式支持；
不能把启用它当作已校准的设备同步行为。

## 向量操作与成本

```yaml
- vec:
    id: normalize
    name: Normalize
    cycles: {expr: rows * cols * vector_factor}
    aiv_presplit: true
```

`cycles` 是用户提供的服务周期，可以是常量或有来源的公式。
本例假设 rows/cols 已属于单 AIV；未预切分模型按其工作量接口声明分摊。
声明成本不表示已经给出数学实现。外部预测用 `vfmodel/vf_workload`，与 `cycles`
互斥，完整配置见[VF 成本接口](vf-costs.md)。

## 循环与条件

```yaml
- loop:
    id: chunks
    dim: k
    range: {expr: ceil(K / Kb)}
    body:
      - vec: {name: chunk, cycles: 8}
- if:
    expr: {expr: extra_work > 0}
    then:
      - vec: {name: extra, cycles: {expr: extra_work}}
- skip_if:
    expr: {expr: rows == 0}
```

`loop.dim` 是从零开始的索引，内层可使用外层变量；`range` 是次数。
`if` 只在条件成立时包含 `then` 的声明，`skip_if` 跳过当前 body 余下节点。
它们用于给定参数下的模型，不自动执行任意数据相关 Python 分支。
条件省略了生产者时，不能继续引用其不存在的输出。

## 表达式

动态值写为 `{expr: '表达式'}`。可使用已声明参数、循环变量、算术、比较、
布尔运算及 `ceil/min/max` 等受支持函数。它不是任意 Python，不导入模块或读文件。
任务上下文 `group/ptid/task_id/is_last_task/is_last_in_group` 的取值以实际 workload 为准；
不要在完整 case 中伪造这些派生值。

## signals、inputs 与 dataflow

```yaml
# operator.yaml 的 pipeline 片段
pipeline:
  phases: [[Produce], [Consume]]
  tokens: {}
  dataflow:
    - {from: Produce, to: Consume, output: ready, input: ready, offset: 0}
```

```yaml
# dag.yaml 的 kernels 片段
Produce:
  body:
    - vec: {id: made, name: produce, cycles: 8}
  signals: {ready: $made}
Consume:
  inputs: [ready]
  body:
    - vec: {name: consume, cycles: 8, after: [ready]}
```

`$label` 引用当前 stage 的指令标签，signal 名对应 dataflow 的 output/input。
`offset: 0` 表示同一 task，`offset: 1` 表示前一 task；phase 的位置不决定 offset。
跨 group 时使用适用的 `cross_group` 声明，并检查首组/尾组。
可选 `$label?` 只用于确有合法空路径的模型；不能凭空产生数学输出。

## 缓冲与版本

stage 的 `pools` 可声明 `scope`；循环局部 pool 用 `scope: loop, owner: 循环id`，
整 stage 使用 `stage`，组内驻留使用 `group_resident`。
`pool` 和 `use_pools` 必须反映真实写入和全部读取，不通过增加 slots 消除必需依赖。

L0 操作数可用 `tile` 区分同名值；跨 task 驻留使用明确的 `allocation/version`。
固定 L1/UB、区域、L0C 累加和显式物理槽的字段见[存储接口](ub-storage.md)。
固定分配可显式声明逻辑值与字节地址，见[地址视图](ub-storage.md#字节地址视图)；
物理访问的 shape、dtype 和读取边界见[物理接口](physical-execution.md)。
声明形式之间有使用限制，不能混用 `tile` 与同一值的 `allocation/version`。

## 可选等待与控制

`pipeline.issue_domains` 声明 stage 所属的 `AIC/AIV_AGG` 域，参与 stage 通过
`issue_domain` 引用。等待节点使用：

```yaml
- issue_gate:
    id: wait_ready
    waits:
      - {signal: ready, event: done}
```

`waits` 每项给出合法 signal 或本 stage 已定义的 `instruction: $label`，
`event` 为 `done` 或该生产者支持的 `partial`。缺失生产者、无效 partial、重复等待
和循环依赖不可接受。域内 stage 应完整声明归属。
门控不是计算或数据值，不能用它替代数学输出、容量或真实消费者。
布尔控制值的声明见[控制接口](control-state.md)。

## 运行和限制

使用[完整 case 入口](../python-api.md)验证模型，核对实际参数、任务覆盖、检查状态及
[结果范围](runtime-and-trace.md)。语法接受不代表数学正确或已完成设备验证。
添加节点、依赖、版本或硬件配置后，用尾块、多任务、多组等真实输入重新检查。
