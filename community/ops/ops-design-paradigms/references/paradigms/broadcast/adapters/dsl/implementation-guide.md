# Broadcast 范式 CANNBotDSL 实现指引

> **层级**：CANNBotDSL 语言适配层叶子文档。通用层算法见 [../../generic/index.md](../../generic/index.md)；谓词真值见 [capability-mapping.md](capability-mapping.md)。
>
> 本指引结合 DSL 公开入口、Host、Kernel 的职责和 Broadcast 通用层模型。具体 API 签名与设备能力以调用方提供的 DSL 资料为准，本文不复制 API 签名。

## 1 数据流骨架 → DSL 三层结构映射

通用层三段式数据流（[../../generic/scene.md](../../generic/scene.md) §3）落到 DSL 的入口/Host/Kernel 三层：

| 通用层阶段 | DSL 落地 | 关键约束 |
|------|------|------|
| 段 1 搬入（随路广播） | Host launcher 分配输出/workspace 后，kernel 内经数据搬运类目 API 完成 GM → 片上 | 随路广播依赖 `CAP_BROADCAST_IN_TRANSFER`（待验证）；为 false 时退化为独立广播步骤，P 值 +1，按 [../../generic/liveness-fusion.md](../../generic/liveness-fusion.md) 重推 buffer 布局 |
| 段 2 计算 | `@kernel` 内寄存器运算（`ops.reg`）逐元素计算 | 链式中间值驻留寄存器依赖 `CAP_REG_CHAIN_FUSION`（待验证）；为 false 时中间值落 Buffer，按通用层存活节点模型重推 P 值 |
| 段 3 搬出 | kernel 内片上 → GM 独立搬出 | 有效行数写入，不把无效尾行写入输出 |

## 2 代码组织（三层职责）

| 层 | 职责 | 边界 |
|------|------|------|
| 公开入口 | 检查公开 dtype / shape / layout / device 参数组合；准备输出并委托 Host/Kernel | 不把设备计算藏在入口或可达辅助函数 |
| Host launcher | 静态配置、输出/workspace 分配、布局归一化、launch 分开组织 | 只做元数据 / 分配 / 零拷贝视图；torch 调用若触发设备算子或复制即越界（见 [capability-mapping.md](capability-mapping.md) §4 规则 1） |
| Kernel | 按真实数据所有权拆阶段方法，避免单个巨型 `__call__` | Broadcast 纯 Vector 路径通常无 Cube/Vector 交接，不因样例结构给纯 Vector 算子加 CrossCore Channel |

## 3 Buffer/Channel 规划

1. buffer 数量按通用层 [../../generic/liveness-fusion.md](../../generic/liveness-fusion.md) 的 P 值推导：Broadcast standard 路径 P = 输入数 + 输出数 + 融合压缩后的中间值数
2. 阶段内中间量用 Buffer（无交接语义不建 Channel）；仅真实生产/消费交接处用 Channel，produce/consume 成对、不越生命周期
3. Channel 深度按在飞任务数与资源预算核定（depth=1/2/3 均为合法样例选择，非规范）
4. 单 buffer 容量按通用层 [../../generic/splitting.md](../../generic/splitting.md) 公式：`perBufBytes = floor(片上容量/P)`，对齐粒度以 [capability-mapping.md](capability-mapping.md) `ALIGN_GRANULARITY` 真值为准（待验证，代入时声明假设）

## 4 shape 归一化与切分的 DSL 落地

1. 补 1 / 去 1 / 归一 / 合轴四步算法（[../../generic/shape-preprocessing.md](../../generic/shape-preprocessing.md)）在 Host launcher 侧用纯元数据操作完成（shape 运算、视图建立），不触发设备计算
2. 片上单切分与多核切分（[../../generic/splitting.md](../../generic/splitting.md)）的切分参数由 Host 计算后传入 kernel，或经 tiler 表达；具体机制以 DSL API 实际支持为准（待验证）
3. 多核切分的核数经架构/平台信息类目 API 获取，严禁写死

## 5 尾块与同步

1. 尾块处理（掩码/有效行）在 kernel 内完成；不把无效尾行写入输出
2. Buffer/Channel 交接遵守 acquire/commit/wait/release 成对语义；跨执行单元事件的配对规则依赖 `SYNC_PRIMITIVE` / `EXEC_UNITS` 真值（待验证），就位前同步设计按 Channel 自带交接语义保守设计并标注假设

## 6 产出与验证义务

每个 DSL 实现决策按如下格式回答（micro-patterns 纪律）：

```
适用前提 → 最小代码位置/伪代码 → 生命周期或同步不变量 → 对应 Unit 测试
```

- 数学语义与形状以当前算子规格为准，Buffer 容量、Channel 深度和同步关系按实际数据流及资源预算推导。
- 引用待验证谓词的结论，按 [capability-mapping.md](capability-mapping.md) §3 声明假设并给出验证义务
