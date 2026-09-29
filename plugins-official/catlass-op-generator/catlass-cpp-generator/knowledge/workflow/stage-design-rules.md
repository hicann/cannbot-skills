---
type: "workflow"
title: "Stage 设计规则"
description: "R01-R21 全量设计规则。"
tags: ["catlass-cpp", "workflow", "design", "stage", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:architecture-separation-source-check", "at": "2026-09-20T00:00:00Z"}]
sources: [{"id": "pr1069-stage-design-rules", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/stage-design-rules.md", "title": "PR1069 fixed source for Linear Attention Stage 设计规则", "kind": "repository"}, {"id": "catlass-arch-separation", "resource": "git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c", "title": "CATLASS fixed source for AtlasA2 and Ascend950 architecture separation", "kind": "repository"}]
consumers: ["catlass-cpp-design"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# CATLASS C++ Stage 设计规则

当前规则版本为 `V1`。03 方案设计阶段必须阅读本文件，内容按 CANNBot 的 CATLASS C++ 工程能力整理。
新算子必须逐条检查 `R01` 至 `R21`；既有算子只重审受改动影响的规则，
但必须在 `docs/design.md` 中记录未受影响规则的依据。

术语：Cube 操作指矩阵乘和必要的矩阵累加，使用 CATLASS `BlockMmad`/Kernel 组件；Vector
操作指逐元素、归约、广播、layout、cast、copy 和 mask；驻留区指数据在 L1 或 UB 中保留并
供后续 Stage 继续使用；`CG` 指一个 AIC 与两个 AIV 一次协同处理的连续 value head 数。

设计开始前必须冻结架构分支，不能把 A5 路径写成跨架构默认规则：[^catlass-arch-separation]

| 架构 | CATLASS 标识 | Vector 执行模型 | 默认跨单元数据路径 | 相关片上容量 |
|---|---|---|---|---|
| Atlas A2/A3 | `CATLASS_ARCH=2201`、`Arch::AtlasA2` | MemBase，使用 `TPipe`/`TQue`/`TBuf` 或等价分段流水 | Cube→Vector 为 `L0C -> GM -> UB`；Vector→Cube 为 `UB -> GM workspace -> L1` | UB 192 KiB、L0C 128 KiB、FixBuf 7 KiB |
| A5/Ascend950 | `CATLASS_ARCH=3510`、`Arch::Ascend950` | 可选 RegBase/VF；是否使用一次 VF 由公式和资源决定 | 组件支持时可用 `L0C -> AIV UB` 或 `AIV UB -> AIC L1` 直连，也可保留 GM 中转基线 | UB 248 KiB、L0C 256 KiB、FixBuf 16 KiB |

一个 AIC 与两个 AIV 的混合 kernel、`GetSubBlockIdx` 分工、CrossCore ready/free、HardEvent、
GM workspace、生命周期、tail 和精度规则是跨架构共享能力，不标记为 A5 专属。A5 直连和 VF
属于目标版本支持时的候选优化，不是所有 A5 实现必须采用的固定路径。

## Stage、任务映射与依赖

### R01 Stage 类型和 AIC/AIV 映射

Stage 划分和执行单元映射分两步完成。先依据数据依赖、计算类型、数据生命周期和精度观察点
划分 Stage，再为每个 Stage 分配 AIC/AIV 任务。一个 AIC/AIV 可以按程序顺序承载多个 Stage；
执行单元相同不改变 Stage 边界，Stage 数量也不由 AIC/AIV 数量推导。只有在依赖、生命周期、
同步和精度观察点都能保持的条件下，才合并相邻 Stage。每个 Stage 只包含 Cube 或 Vector 一类操作。
`CG` 只沿 value-head 轴 `HV` 分组；AIC 按
`r=0..CG-1` 依次完成 Cube 任务。Vector 任务根据每个 AIV 的 UB 峰值、数据依赖、可独立执行的
head 数和负载选择分工：

- **逐 head 分配**：每个 AIV 负责不同的完整 head，例如 `owner=r%2`。完整 head 的工作集能放入
  单个 AIV 的 UB，且有足够独立 head 时，优先评估这种分工。
- **同 head 分片**：两个 AIV 分别负责同一 head 中不重叠的连续区域。独立 head 较少、Vector
  工作量较大或完整工作集超出单个 AIV 容量时，评估这种分工；给出切分维度、对齐后的边界、
  有效区和每片工作集。涉及跨片归约或归一化时，补齐局部结果合并及后继计算的依赖。

在 `docs/design.md` 记录选择依据、各 AIV 的读写范围和尾部处理；共享输入、归约中间量及对齐
开销按实际计入容量。同一种分工以 `CG`、shape 和有效长度参数化；跨 Stage 调整分工时，
写清数据重分配、搬运和同步。两种分工的代码写法见 `kernel-stage-sync-patterns.md`。

### R02 Stage 输入依赖

Cube Stage 的输入来自算子输入或前序 Stage 已就绪的数据。Vector Stage 内允许后续计算依赖
本 Stage 已产生的结果。

跨 Stage 按消费者实际需要的数据范围判断就绪：所需数据全部写入完成并对消费者可见后，即可消费。
完成粒度可以是 head、分片或整组，由数据依赖决定，并与 ready 通知覆盖的范围一致。

### R03 L1/UB 硬件边界

容量计算使用目标 SoC 对算子实际可用的 L1 和 UB 上限，并在 `docs/design.md` 记录 SoC、
参数来源和可用值。L1 服务 Cube 路径，UB 服务 Vector 路径，不混用容量预算。

### R04 Cube 到 Vector 的 UB 驻留

Cube 结果继续由 Vector 使用时，先按目标架构选择数据路径，再按 R01 的分工写入最终 UB 区间：
A2/A3 以 `L0C -> Fixpipe -> GM -> UB` 为默认路径；A5 在目标 CATLASS/CANN 版本的组件明确支持时，
可由 Fixpipe 将完整 head 写入 owner AIV UB，或通过 dual-destination 将两片分别写入两个 AIV UB，
否则仍使用 GM 中转基线。每个 AIV 的驻留份数按该核同时存活的 head 或分片、流水深度及最后消费
时机推导，并给出缓冲区满时的等待和释放安排。
直连路径中每份数据的 Fixpipe 完成后才允许发布对应 `cubeToVec` ready；GM 中转路径必须分别闭环
Fixpipe→GM 和 GM→UB 的核内事件，再发布消费者可见的 ready。采用聚合通知时，所有参与核按协议
发送或等待对应通知（token）。具体目标地址、pipe 和通知粒度按目标平台及
`kernel-stage-sync-patterns.md` 实例化。

### R05 Vector 中间结果驻留

Vector 结果继续由 Vector 使用时，各 AIV 保留自己负责的完整 head 或分片。UB 空间按该核
同时存活的数据、双缓冲（ping/pong）深度、共享输入和临时量分别计算。
后继 Stage 使用相同分工时直接消费驻留结果；分工变化时按详设完成数据交接后消费。

### R06 Vector 到 Cube 的 GM 交接

Vector 结果继续由 Cube 使用时，逐 head 分配由 owner 将完整结果写入该 head 的 GM 区域；
同 head 分片由两个 AIV 分别写入各自的 GM 区间，合起来覆盖完整有效输入。
每个实际写者在 MTE3 完成后使用 `PIPE_MTE3` 语义发布 ready；AIC 等待当前任务所有输入
就绪后再读取。聚合通知按选定协议配齐所有参与核的 token：未负责当前 head 的 AIV 或有效
长度为零的分片跳过数据计算和写回，仍参与该任务约定的通知。
GM 缓冲槽的复用由最后一次读取完成后的 free 或等价消费完成事件保护。A2/A3 使用本规则作为
默认交接；A5 只有在目标组件明确支持 AIV→AIC L1 直连且完成 alias、容量和同步验证时，才可把
`SYNC_L1_HANDOFF` 作为候选替代，GM 路径仍保留为可回退基线。

### R07 Cube 中间结果的 L1 驻留

Cube 结果继续由 Cube 使用时放入 L1 驻留区。按当前 TilingKey 和实际调度，计算从首次写入到
最后一次消费完成期间的最大同时存活数据量，结合各份数据的大小、对齐和地址布局确定驻留份数。
设计中给出生产、各次消费、释放和复用顺序；新数据覆盖槽位前，用相应同步证明旧数据的最后一次
消费已经完成。例如 `CG=4` 时，若调度最多同时保留两份且复用前旧数据均已消费，可采用两槽流水；
若四个 head 的结果全部产生后才开始消费，则需要保留四份。

## 容量、地址与生命周期

### R08 UB 峰值容量

UB 预算必须覆盖当前 Vector Stage 同时存活的全部输入、输出、中间量和跨 Stage 驻留数据，
按峰值而不是单个算子或单个张量计算。

### R09 L1 常驻和地址复用

L1 可以常驻其他张量，也可以在不同 Stage 用同一区域保存不同数据；所有常驻数据统一按 R07 的
最大同时存活数据量规划空间，生命周期结束后按对应同步条件复用。

### R10 GM 搬运次数

同一路径中的每份数据只从 GM 搬入一次。同一原始数据同时被 Cube 和 Vector 使用时，可以
分别搬到 L1 和 UB 各一次，但设计必须把两条路径及字节数记录清楚。对只使用一次的流式
数据，设计阶段应评估关闭本次搬运对 L2 缓存的参与；对跨核交接或后续会复用的数据，按
实际复用关系选择缓存参与方式。缓存策略必须使用目标 SoC 和当前搬运 API 支持的粒度，
并通过同条件性能测试确认收益。

### R11 UB 地址复用

UB 区间允许在不同 Stage 保存不同数据，但只有旧数据的最后一个消费者完成后才能复用；
设计必须标出生产、最后消费和复用位置，并由对应 `MTE3_MTE2`、`MTE3_V` 或 `V_MTE2`
事件保护。事件必须在 kernel 尾部排空后释放，不能只依赖跨核 ready。

### R12 Vector 单任务装载与执行模型

每个 AIV 按当前逻辑任务装入自己负责部分所需的数据，并根据目标架构选择执行模型。A2/A3 的
MemBase 路径按 `TPipe`/`TQue`/`TBuf` 或等价分段流水组织，不要求一次 VF 调用；A5 采用
RegBase/VF 时，只有在完整公式、寄存器压力和 mask 均满足时才合并为一次 VF，不能为满足调用次数
而改变 Stage 语义。UB 预算必须覆盖当前流水或 VF 同时存活的输入、输出、中间结果、scale、gate
和 mask 等小张量；RegBase 还需核对寄存器占用及编译器生成结果。

### R13 无依赖 Stage 的并行存活

没有依赖关系的 Cube Stage 与 Vector Stage 按可能并行执行建模；并行期间仍存活的数据使用
互不重叠的 L1/UB 地址区间，不能按串行峰值复用。

### R14 容量冲突时的 GM 中转

L1/UB 容量不足或生命周期冲突时，可以把中间结果写回 GM，再由后续 Stage 读入。设计必须
说明冲突原因、增加的读写字节数、同步点和预期性能代价。

### R15 最少 Stage

满足数据依赖、容量、生命周期、同步和精度观察点后，优先合并连续同类操作，减少 Stage 数和
Stage 内任务数。合并依据记录在 Stage 依赖表中；同一个 AIC/AIV 顺序执行多个 Stage 时，仍按
各 Stage 的输入、输出和观察点保留逻辑边界。

### R16 Vector 结果复用

Vector 路径会多次使用的结果优先保留在 UB 驻留区，并在生命周期表记录生产、各次消费和释放。

### R17 连续地址与碎片检查

仍有效的 L1/UB 数据保持原地址。地址规划同时计算总剩余空间和最大连续空闲区，二者都满足
下一次分配才算容量合法。

### R18 跨 head 一致性

同一种输入、输出或中间量在不同 head 上执行相同操作并采用相同的地址映射；差异只能由有效 head、tail
或明确的接口语义产生。

### R19 Stage 划分决策顺序

先检查合并方案中同时存活的全部数据、L1/UB/L0 容量和生命周期。容量足够时按真实依赖和
Cube/Vector 边界取最少 Stage；容量不足时优先评估 R14 的 GM 中转。只有额外拆分能解决
GM 中转仍无法解决的正确性、生命周期或性能问题时才增加 Stage。

## 分核与数值稳定性

### R20 沿 HV 分核和 CG 选择

沿 `HV` 轴按 `CG` 形成工作项，并明确 `hv` 到其他 head 轴的映射、共享输入、归约和写冲突。
chunk 间无依赖时 `Nbase` 为本次调用的 chunk 总数；有依赖时 `Nbase` 为 sequence 总数；
`Nwork=Nbase*ceil(HV/CG)`，`blockDim=min(可用 AIC 核数,Nwork)`，使用 grid-stride 分发，
尾部只处理有效 head。候选通常为 `CG=4/3/2/1`；`HV/HK=3` 时优先纳入 `CG=3`。
`CG` 候选与 R01 的 AIV 分工一起评估，先排除容量或依赖无法满足的组合。
各可行候选比较执行波次、尾部负载、容量、搬运、AIC/AIV 关键路径和同步开销；有设备时同条件
实测，无设备时标为暂定，并依次依据正确性、各核的读写权限和地址分配、负载均衡、流水平衡、搬运和同步选择。

### R21 数值稳定性检查

逐项分析高风险运算的计算 dtype、输入和中间值范围、溢出/下溢、`NaN/Inf`、cast 与 mask
时机。代数变换须提供变换前后的值域、误差和溢出风险依据，证明数值稳定性保持或改善；无效位置
在高风险运算前排除或替换为安全值。只有每项高风险运算都有边界证明或保护方案，才能进入算子开发。

## 设计文档写法

公式必须声明张量维度、索引范围、累加 dtype、cast/归一化顺序、mask、边界行为和 state 更新时机。
每个 Stage 至少写出执行单元、操作顺序、输入/输出、
任务映射、地址和容量、ready/free 或 event/flag、最后消费点、精度检查点以及与前后 Stage 的依赖。

开发期 `docs/validation.md` 记录设计与验证的对应表：设计项、代码位置、用例、实际结果和结论。
Stage、内存和同步等设计调整只在 `docs/design.md` 更新并重新评审后才能改代码；
候选实验只记录到验证记录，采用后再回写正式设计。

## CANNBot 设计产物

`docs/design.md` 必须包含：

1. `R01`–`R21` 逐条结论和证据位置。
2. 完整计算图、Stage 顺序、每个 Stage 的 Cube/Vector 类型和依赖。
3. `HV/HK/CG` 映射、`Nbase/Nwork/blockDim`、各 AIV 负责的 head 或分片范围、选择依据和 tail 处理。
4. L1/UB/L0/GM workspace 分区、峰值容量、最大连续空闲区和生命周期表。
5. ready/free、event、flag、barrier、ping/pong 的同步配对，以及 workspace 缓冲槽（slot）的安全复用条件。
6. CATLASS ArchTag、DispatchPolicy、TileShape、BlockMmad、BlockEpilogue、BlockScheduler、
   Kernel、workspace 与合法分支组合。
7. 高风险数值运算的 dtype、值域、保护方式和精度阈值。
8. 每个实际模型场景用例的基线、目标、能否达到目标的分析和固定性能统计口径。

完成条件：21 条规则均有结论，CATLASS 组件能在当前代码版本找到依据，容量和同步配对可验证，
数值稳定性检查通过，设计与 `docs/api.md`、`reference/reference.py` 一致。满足全部条件后进入 04；
其余情况保留在 03 并补齐设计和证据。

[^catlass-arch-separation]: git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c; paths: README.md, include/catlass/arch/arch.hpp, include/catlass/gemm/tile/copy_l0c_to_gm.hpp, include/catlass/gemm/tile/copy_l0c_to_ub.hpp, include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp

[^pr1069-stage-design-rules]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/stage-design-rules.md
