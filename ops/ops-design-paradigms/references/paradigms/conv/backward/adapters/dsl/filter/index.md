# dW：DSL 适配入口

用 x 和 dedy 计算权重梯度 dedw。先读 [通用场景](../../../generic/filter/scene.md) 和 [MN_StreamK tiling](../../../generic/filter/tiling/templates/mn-streamk/tiling.md)，再用本目录把已选计划落实为 DSL。

类型、格式与场景准入以 [scene](../../../generic/filter/scene.md) 为准。本分支说明 MN_StreamK 的普通窗口、切宽、拆核窗口及写回如何绑定到 DSL；模板字段的绑定见 [tiling.md](tiling.md)。

**名字带 StreamK，不代表这里已经把同一输出的 K 分给多个核。** 当前每个输出任务完成全部归约，长 K 只在核内分段，段间按既有协议原子累加到 GM。严格对齐 generic 的 MN_StreamK 时，保留这一归约所有权。独立的 [dw_splitk.py 扩展示例](../example/dw_splitk.py) 演示跨核分片和确定性合并；它有额外 workspace 与第二阶段，不属于该 generic 模板。

| 正在解决的问题 | 阅读文档 | 得到什么 |
|---|---|---|
| 类型、版本与资源声明 | [公共能力](../capability-mapping.md)、[buffer 与类型](../buffer-and-types.md) | 当前 DSL 接口、槽容量和物理地址的约定 |
| Host 计划如何传给 DSL | [tiling.md](tiling.md) | 模板常量、运行时字段、实际启动核数 |
| 每一步用什么接口 | [instruction-mapping.md](instruction-mapping.md) | 两路搬入、Load2D/3D、MMAD、三种输出布局 |
| 如何组织循环 | [kernel-template.md](kernel-template.md) | 输出任务、归约段、父段和计算块的嵌套关系 |
| 如何预取、复用与初始化 | [sync-implementation.md](sync-implementation.md) | RingBuffer 游标、父段生命周期、清零与写回 |
| 可运行样例 | [dW 样例](../example/dw.py)、[切 K 扩展示例](../example/dw_splitk.py)、[运行说明](../example/README.md) | 多 C1、批量写回、动态 shape 与布局回归 |
| 如何判断完成 | [通用验证](../../../generic/filter/verification.md)及下列检查 | 数值、分支和生成指令证据 |

实现至少覆盖普通/拆核窗口、整宽/切宽、三种 dedw 格式、通道尾块和无贡献深度；另取 `baseN` 不能整除完整通道窗口的形状检查写回。每条搬运记录实际生成的指令数和字段，区分“视图/API 可表达”“编译成功”“完整算子通过设备验证”。本目录的代码块是设备接口片段，除明确标注的示意外使用实际 API；没有提供可直接交付的完整 dW 工程。

编译、保存与调用入口见 [kernel-entrance.md](../kernel-entrance.md)。工程生成时补齐 Host 求解器、所有阶段函数及测试，不能把示意函数名当成框架内置函数。
