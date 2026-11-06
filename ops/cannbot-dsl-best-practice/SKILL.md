---
name: cannbot-dsl-best-practice
description: "为 CANNBotDSL 算子提供整体实现、局部写法和生产代码最佳实践。当需要职责划分、Host 约束校验、RingBuffer/同步/布局、注册与适配、编译启动、AICPU 负载均衡 metadata、VF 向量计算或命名建议时使用。"
metadata:
  category: implementation
---

# CANNBotDSL 实现最佳实践

API 写法以 CANNBotDSL 0.8.0 为例；遇到版本差异时，参考本 skill 的编码思想，按实际版本的 API 自适应调整。针对当前实现问题提供可应用的写法、选择依据和正反例。参考内容按主题组织，示例使用通用对象名，不要求某种算法、工程目录或特定实现存在。

## 输入

- 算子规格与设计资料：由调用方提供文件或目录，确定数学语义、执行路径和目标设备约束。
- 当前实现代码与需要解决的组织或微模块问题。
- 按需提供当前 Unit、测试用例和 Golden，用于说明本次能力边界与验证方法。

## 输出

针对当前问题给出代码组织或微模块写法建议，写明设计依据、适用前提、需核实的 API/设备事实，以及应执行的相关测试。调用方决定建议的记录位置和是否用于修改代码。

复核设备初始化、后处理或额外设备任务时，结论逐项覆盖：公开入口及 helper 的执行位置与启动次数；输出覆盖、首次写入到读取及实际同步依赖；CPU 初始化或补算公开结果后搬至设备是否绕过约束；独立 Golden 与生产路径的边界；需回归的数值与结构义务。各项说明调整方向或不适用依据，不以数值等价替代设计承诺。

## 参考选择

| 问题或关键词 | 参考 | 解决的问题 |
|---|---|---|
| 整体结构、入口、Host、Kernel、@export、Native 导出、生命周期、CV 融合、preload | [整体实现](references/overall-implementation.md) | 公共骨架、导出、程序管理与 preload 流水 |
| CV 融合编码、C/V 阶段封装、集中调度 | [CV 融合编码示例](references/patterns/cv-fusion.md) | C1→V1→C2→V2 数据交接、错位调度与排空 |
| 命名、常量、重复、注释、docstring、可读性 | [生产代码](references/production-code.md) | 清楚表达数据、单位与计算意图 |
| Host 校验、非法输入、约束拦截、dtype、shape、参数组合 | [Host 约束校验](references/patterns/host-validation.md) | 在分配、编译和启动前拒绝非法元数据 |
| RingBuffer、Buffer、双缓冲、深度、slot | [RingBuffer 与 Buffer](references/patterns/channel-buffer.md) | 存储选择、生命周期和复用 |
| 跨核、同步、事件、屏障、流水线交接 | [同步](references/patterns/synchronization.md) | 写入、可见、读取、复用顺序 |
| VF、寄存器向量、mask、向量归约、cast、pack/unpack、局部编码优化 | [VF 向量计算](references/patterns/vf.md) | 向量操作链、循环与寄存器复用、尾部掩码、类型转换与累计状态 |
| 搬运优化、批量读写、有效视图、冗余初始化与复制 | [搬运与内存访问](references/patterns/memory_optimization.md) | 既定方案内的批量搬运与冗余访存消除 |
| layout、stride、ND/NZ、视图、尾块、有效行 | [布局与尾块](references/patterns/layout-tail.md) | 物理跨度、格式转换与有效范围 |
| workspace、临时存储、资源容量、别名、跨阶段复用 | [Workspace](references/patterns/workspace.md) | 分配、所有权、容量与生命周期 |
| AICPU、负载均衡、metadata、调度协议、记录消费、任务成本、分核、分轮 | [AICPU 负载均衡 Metadata](references/patterns/aicpu-load-balance-metadata.md) | 设备侧调度、成本划分、记录生成与消费 |
| Torch 注册、schema、Meta/Fake、dispatch、torch.ops | [Torch 注册](references/patterns/torch-registration.md) | 注册契约与真实计算一致 |
| aclnn、参数映射、接口兼容、运行时适配 | [aclnn 适配](references/patterns/aclnn-adaptation.md) | 语义转换与接入职责 |
| 编译、程序复用、特化、TensorSpec、launch | [编译与启动](references/patterns/compilation-launch.md) | 编译结果复用与本次输入绑定 |

生成完整算子或设计其公共结构时，必须读取 [整体实现](references/overall-implementation.md)，落实其中的公共骨架与必备 `@export` 入口；生成或修改算子源码时读取 [生产代码](references/production-code.md)，应用代码规范与轻量检查规则。

按问题含义选择内容；多个主题直接相关时组合读取。仅有命名问题时不需要读取全部计算参考，缓冲覆盖问题则同时涉及 RingBuffer 与同步。

## 禁止事项

以下重申 [生产代码](references/production-code.md) 的约束，覆盖公开入口及其 helper、注册与适配层，按实际执行效果判断：

- 禁止通过 torch 额外设备算子补算、初始化、复制或转换结果，包括 `torch.zeros/zeros_like`、`ones/full`、`zero_/fill_` 及间接调用；也不得先在 CPU 初始化或补算公开结果，再搬到设备绕过约束。输出与 workspace 使用 `empty/empty_like` 等未初始化分配，必要初值由 DSL 路径建立。
- 禁止读取尚未写入的输出或 workspace，或在跨核、跨阶段依赖尚未满足时读取、覆盖或复用数据。
- 禁止将会复制或转换数据的 `reshape/contiguous/.to(...)` 等操作当作零拷贝视图；普通输入与 tiling 校验不得通过 `.item()`、`.cpu()` 或设备 Tensor 转布尔值读取设备内容。
- 禁止在正式 Kernel 中保留设备运行时打印，或为验收插入 probe、事件记录、trace 缓冲、诊断计数、输入克隆及同步回读；关闭开关不能替代移除。Host 普通打印及公开计算必需的数据与同步按实际用途区分。
- 禁止以零值、空壳或未写入的分配结果冒充未实现的合法分支；禁止运行与 Native 导出采用不同的主计算路径，或以局部子程序的验证代替完整算子验证。
- 禁止只按 kernel 名称过滤 profiler 结果来隐藏额外设备任务，或以总耗时下降作为违规调用的放行依据。设计所需的多 DSL Kernel 和 metadata 接口允许保留，不为证明已消费而额外启动检查 Kernel。
- 禁止无依据地硬编码数学系数或开发机核数，将未核实的 Python 写法搬入 DSL 前端，或让有符号设备整数直接参与位运算。
- 禁止在函数体、类体或文件中段导入及用动态导入绕过源码布局要求；公开 Python 输入校验不得依赖可被优化关闭的 `assert`。
- 禁止在算子源码中保留开发期次、内部设计坐标、验收与测试痕迹、实验日志、历史比较和已注释的旧代码；不添加冗余实现说明、无意义算术或诊断操作。

独立 Golden 与测试数据构造不属于正式算子路径；允许不下发额外设备算子的 torch 元数据处理、分配、已确认的零拷贝视图与注册。

## 应用原则

1. 先确认当前问题和公开契约，再读取直接相关的参考。建议应指向具体代码位置，说明为什么适合这里。
2. 示例只展示一个核心决策。将其中的维度、类型、资源上限换成当前契约；不要把示例选择推广成硬件默认值。
3. `text` 代码块是结构伪码，其中的操作名不是 DSL API。落地前从当前源码、安装包或官方资料核实签名、同步语义和设备限制；不能把伪码当成已通过编译的代码。
4. 每条建议附带能够暴露该问题的检查，例如跨步输入、尾块或缓冲复用。诊断信息和反例放在对应主题内，不要求独立探针或额外执行流程。
5. 设计规定的设备计算和搬运由 DSL 完成；Host 处理元数据、分配、已确认的零拷贝视图和启动，可使用不下发额外设备算子的 torch。禁止正式路径用 torch 设备小算子补算、初始化或转换。也不得先在 CPU 初始化或补算公开结果，再搬到设备绕过上述约束。按实际调用效果和 msprof 证据验收，不限制设计所需的自定义 DSL Kernel 数量。独立 Golden 与测试代码不承担公开结果的补算。

6. 区分设计强制承诺、允许选项与待定候选；在选项内调整实现，保留已交付行为并回归受影响能力。关键补证失败或改变强制执行域、通路、同步、复杂度等承诺时，向调用方记录失败证据与影响，不以数学等价替代设计完成，不自行修改冻结设计或调度回滚。
