# 控制值与等待接口

用于显式描述模型的布尔控制值及读写参与者。本页接口支持有限静态路径和对称聚合 AIV；
不表示支持任意运行时数据分支、硬件 flag 编号或真实 SS 布局。

需要数据相关有界分支或循环时，另读[运行时控制](runtime-control.md)；不能把本页
`expect` 或 `issue_gate` 当作动态分支接口。

## 输入声明

在 `operator.yaml:pipeline` 中声明：

```yaml
issue_domains:
  cube: {engine: AIC}
  vector: {engine: AIV_AGG}
control_states:
  ready_flag:
    value_type: bool
    slot_shape: [8, 8]
    producer: AIV_AGG
    readers: [AIC, AIV_AGG]
    reader_events: {AIC: ready, AIV_AGG: read_allowed}
    aggregation: symmetric
    notification_cycles: 0
    cost_source: assumed_zero
```

零通知成本是明确假设，不能当作实测。stage 通过 `issue_domain` 引用对应域。

| DAG 节点 | 主要输入 |
| --- | --- |
| `control_publish` | `allocation/version/slot/value`，可附加标签与 after |
| `control_read` | `allocation/version/slot/expect` |
| `control_signal` | `allocation/version/slot/event` |
| `issue_gate` | `waits` 中的控制引用及命名 event |

`allocation` 对应控制声明名；生产者和读者使用同一逻辑版本和槽位。
`value` 可给布尔值或与 AIV 数量一致的布尔列表，各项只接受 bool 或整数 0/1。
`expect` 是用户对当前静态路径的预期；不会自动生成计算分支。
跨 task 时明确版本及适用的 `logical_group`；需要显式写入次序时提供 `position`。

## 运行与输出

作为完整模型运行，按结果中的实际检查状态和完整 trace 核对控制事件。
缺生产者、缺读者、槽位或版本不匹配时修正输入，不能将错误事件改为可选来绕过检查。
控制等待不是数学数据，也不能用于替代数值输出或普通存储的正确性检查。

## 限制

参与者与静态路径须由真实实现或规格确认。模型通过不能证明设备实现具有同样的保护。
物理核同步应使用[物理执行](physical-execution.md)的事件集合与同步声明。
未覆盖的标量成本、通知成本和数据相关分支在报告中保留为限制。
