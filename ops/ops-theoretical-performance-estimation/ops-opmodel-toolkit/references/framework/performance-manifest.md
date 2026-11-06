# 性能产物与检查状态

普通完整 case 和物理执行均可提供性能产物。先查看 `result.json:artifacts`，
读取实际生成的文件，不假定每种入口返回相同结构。

## 输入与调用

模型输入必须声明实际计算、搬运形状、dtype 和缓冲容量。
需要性能预期检查时，将独立推导的预期通过当前入口支持的
`performance_expectations` 或模型 `validation` 声明传入。
普通调用签名见[Python API](../python-api.md)，物理配置见[物理执行](physical-execution.md)。

预期应来自数学规格或独立证据，不能直接复制本次仿真结果充当预期。
未提供预期时，`performance_consistency_report.json` 的 `not_run` 表示没有执行该项检查。

## 输出含义

| 信息 | 解读方式 |
| --- | --- |
| cycle、duration | 同时记录范围和时钟；主机运行时间另计 |
| MAC/操作量 | 保留指标定义，不直接把 MAC 当作 FLOPs |
| source/destination bytes | 分别表示搬运两端的量，转换 dtype 时可不同 |
| scope/owner/engine | 限定统计范围，不把代表核数据当作全部核总量 |
| capacity/quantity/coverage | 各自的检查状态，不互相替代 |
| provenance/成本身份 | 标明参数、测量或预测来源及适用域 |

全级搬运字节合计不是 HBM 流量；管线服务周期和也不是端到端耗时。
比较前确认相同输入、工作量、硬件、成本和 scope。

## Python 搬运数量接口

Python Stage 在 `build(context)` 中通过 `ProgramBuilder.move(**fields)` 声明搬运，
输入与 DAG 的 `move` 字段一致：`src/dst` 指明张量和内存，`size` 使用实际元素形状，
dtype 在 stage 或共享 dtype 表中声明。完整调用见[Python Stage](../new-operator/python-stages.md)。
源、目的 dtype 不同时分别检查产物中的两端字节量；尾块填写有效形状，
不能用缩小形状或改写字节量来调整预测周期。声明及数量检查不替代真实布局或设备流量的验证。

## 限制

周期完成、工作量匹配、容量合法和数学正确分别报告。
失败或缺失检查须保留原因；数量一致不能证明成本已校准或实际设备更快。
