# 运行结果与时间线接口

输入是模型目录、case 和可选硬件/方案；输出目录每轮独立。
普通完整 case 调用见[Python API](../python-api.md)，物理模型见[物理执行](physical-execution.md)。

## 产物

| 产物 | 用途 |
| --- | --- |
| `result.json` | 总周期、结果范围、检查状态及 `artifacts` 路径 |
| `params.json` | 本次实际参数 |
| `assignment.json` | 本次任务分配 |
| `Pipe_statistic.csv` | 指定范围内的管线统计 |
| `Instruction_statistic.csv` | 指令类别统计 |
| `trace.json` | 完整时间线；按入口请求导出 |

普通完整 case 的 trace/CSV 可能只对应关键核或某个 phase；以 `artifacts` 的 scope 为准。
物理入口的统计含 engine 身份，范围见[物理产物](physical-execution.md)。
`host_runtime` 是工具在主机上的运行时间，不是模型周期。

## 时间线字段

`trace.json` 使用 trace-event JSON。常用字段为 `name/cat/ph/pid/tid/ts/dur/args`，
OpModel 的时间单位是 cycle。使用支持该格式的查看器打开文件；不要从显示单位
推断设备微秒数。转换为微秒需使用本次 `clock_ghz`。

保留 `pid/tid` 与 engine 身份，同名不同核不能合并为一条管线。
执行轨、slot/等待/控制轨含义不同；等待时间不能当作指令服务工作量。
字段不存在表示该产物没有提供该信息，不能据此断言等待或依赖不存在。

压缩 trace 与完整 trace 的信息范围不同；本版本压缩格式不包含独立的
issue/gate 和控制轨。需要分析这些等待时，运行时请求完整 trace。
不要把局部 phase 的时间线套入完整 case 的周期窗口。

## 解读与使用限制

先确认运行身份、输入、硬件和 scope，再对照模型说明观察工作量及空闲时段。
CSV 的指令服务周期和可能包含重叠，不能相加当作端到端延迟。
完整 case 耗时使用 `result.json`；结果分析见[分析指南](../analysis.md)。
时间相邻或重叠只提供实验线索，不能单凭时间线认定某项资源导致等待。
候选必须按相同输入与成本重新运行，设备收益仍需实测。

## 图搜索的等待摘要

支持该输出的图搜索摘要提供 `issue_queue_waits` 和 `issue_gate_waits`。前者记录
命令、task、engine、VF／variant、等待区间、物理深度及释放边界，后者记录门控等待；
缺失字段不补造。`wait_cycles` 保持原统计范围。各指令与资源等待可能重叠，不能
相加作为可消除的关键路径耗时，也不作为新惩罚再次叠加到总周期。

软件 buffer 与物理队列深度分开；固定硬件比较用户方案的分块、生产消费、预取和
发射选择。队列满可能是正常饱和，不能据普通模型的输出推断未建模的事务／信用容量。
