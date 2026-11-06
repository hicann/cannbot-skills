# 物理 AIC/AIV 模型接口

在 `operator.yaml` 中声明 `physical_execution`，输入完整 case、任务字段、物理放置、
存储区域和同步要求。可运行合成示例见[物理模型](../../assets/examples/physical/README.md)。
stage 可来自 YAML 或 [Python `build(context)`](../new-operator/python-stages.md)；
Python 使用本 owner 的投影参数与任务范围，执行支持与结构搜索支持分别核对。

## 任务与放置

```yaml
physical_execution:
  invocation_key: [task.call]
  task_key: [task.call, task.owner, task.lane, task.tile]
  programs:
    vector:
      select: task.kind == 'vector'
      placement:
        kind: AIV
        group: {expr: task.owner}
        lane: {expr: task.lane}
      stages: [load, compute, store]
      order_by: [task.call, task.tile]
      buffers: [input, output]
      parameters:
        rows: {expr: task.rows}
      task_drain: false
  execution:
    memory_service: independent
    issue_queue_mode: unbounded
    notifications:
      mode: assumed_zero
      source: synthetic/test-only
```

这是输入片段，引用的 task 字段、stage 和 buffer 必须在模型中存在。
每个任务恰好匹配一个 program；task_key 唯一；group/kind/lane 在所选硬件范围内，
并与任务分配一致。参数投影不能覆盖 ptid/group/task_id 等保留上下文。
`task_metadata` 可保存报告所需的额外任务字段。

表达式使用 `params.* / task.* / hardware.* / loop.* / index.*` 等受支持命名空间，
不执行任意 Python。零通知成本和独立内存服务是本例假设，不能当作硬件承诺。

## 同步和存储

`event_sets` 用 `programs/instructions` 选择事件，可用 `exclude` 排除，
`frontier` 接受 `all/first_per_engine_pipe/last_per_engine_pipe`。
`synchronization` 使用 `scope/arrivals/departures/gates` 描述参与者；
`gates` 可选择 execution、issue 或两者。缺少参与者或出现循环依赖不可接受。

`physical_execution.storage` 描述 UB/L1/GM 分配及其
`memory/programs/physical_bytes/slots/shape/dtype`；可通过 `buffer` 引用对应 buffer。
指令上的 `physical_accesses` 声明 `storage/role/version/region`，按需要指定 owner、
source_region、foreach，也可显式给出访问的 `shape/dtype`。
`boundary: auto` 按本访问支持的读取完成或完整完成边界检查；不能用它省略有效区域和 owner 检查。
地址字段见[存储接口](ub-storage.md#字节地址视图)。生产和消费必须覆盖同一实际区域，
同 owner 下的同名分配不得重复声明。普通存储声明同样需要正确的 owner 绑定。
完整字段组合以[示例 operator](../../assets/examples/physical/definition/operator.yaml)和
[示例 DAG](../../assets/examples/physical/definition/dag.yaml)为起点修改。

## 调用

```text
opmodel MODEL --case CASE.yaml --backend both --output OUTPUT --trace
```

`--backend` 为 `simpy/replay/both`。`--gates` 是当前入口支持的检查名列表，逗号分隔；
不能通过关闭必要检查来接受候选。其他选项先查看当前命令帮助。
也可使用 `run_native_case()`，签名见[Python API](../python-api.md)。

## 产物与含义

| 产物 | 含义 |
| --- | --- |
| `result.json` | 完整 case 周期、validation 与 artifacts |
| `assignment.json/params.json` | 任务分配和实际参数 |
| `Case_statistic.csv` | 完整 case 统计 |
| `Pipe_statistic.csv/Instruction_statistic.csv` | 带 Engine 字段的管线/指令统计 |
| `trace.json` | 请求导出时生成的全物理 engine 时间线 |

产物的 scope 为 `all_physical_engines`。完整 case 周期以 `result.json` 为准，
不同 engine 的周期不能相加。MAC 不直接等于 FLOPs，全级搬运字节不是 HBM 流量。
CSV 中指令服务周期可能重叠，不能据其总和计算端到端耗时。

## 限制

不支持 slot trace 或压缩 trace；每轮选择新目录以保留原结果。
`capacity/assignment_coverage/region_coverage/quantity/runtime/backend_agreement`
分别报告状态；数值模式仅用于支持的模型。配置被接受不等于这些检查全部完成。
示例未校准、未进行设备实验；其通过结果不能作为其他模型或设备性能的证据。
