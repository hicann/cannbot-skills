---
type: "workflow"
title: "方案设计参考"
description: "当前两章交付适配、两步设计法、完整详设规则和历史示例。"
tags: ["catlass-cpp", "workflow", "design", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:architecture-separation-source-check", "at": "2026-09-20T00:00:00Z"}]
sources: [{"id": "pr1069-solution-design-reference", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/solution-design-reference.md", "title": "PR1069 fixed source for Linear Attention 方案设计参考", "kind": "repository"}, {"id": "catlass-arch-separation", "resource": "git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c", "title": "CATLASS fixed source for AtlasA2 and Ascend950 architecture separation", "kind": "repository"}]
consumers: ["catlass-cpp-design"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

# 当前工作流适配

设计阶段必须完整读取并执行“PR1069 原始正文”：先完成 Stage 划分，再补齐资源、同步、tiling、
精度、性能、测试、风险以及 R01-R21 规则检查。完整设计推演可以保存为
`docs/design.full.md` 中间产物，用于保留推导过程和评审证据。

最终交付件 `docs/design.md` 的结构权威是 `catlass-cpp-design/templates/design.md.template` 和
`catlass-cpp-design/scripts/validate_design.py`，只保留两个顶层章节：

1. `目标与数学语义`；
2. `Stage 总览与完整详设`，并以 `Workspace 总量` 作为最后一小节。

这是对完整设计结果的结构投影，不是删减设计过程。历史示例第 3 章除 Workspace 总量外的资源分析、
以及第 4～6 章承载的精度、性能、测试、风险和规则检查，仍须在完整推演中完成；其中可执行验证与
实测证据统一进入 `docs/validation.md`，最终 `docs/design.md` 不生成独立的第 3～6 章或 R01-R21
检查表。若历史章节布局与当前交付模板冲突，以本适配说明和当前模板为准；技术规则与分析深度仍以
完整原始正文为准。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。资源布局的表达粒度参考
`flash-linear-attention-npu` 的 `ChunkGdnBwdIntra design.md`：对象名直接标出 `tensor[N]`，
并在同一条目写明 dtype、每份大小、首次生产/搬入和最后消费。生成文档还须区分物理预留份数与
实际激活份数/条件，并给出总大小和释放/复用条件。该表达规则跨架构适用；参考文档中的 A5
Stage、直连数据路径和具体份数不得原样投射到 A2/A3。

参考来源：https://github.com/flashserve/flash-linear-attention-npu/blob/3f4c016adc69e9ff40690e95522def5fde0735cc/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gdn_bwd_intra/docs/design.md

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取当前适配和 PR1069 原始正文；不得把知识内容
或设计中间产物当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、分析项、阈值、失败条件、状态恢复或验收要求；两章限制只适用于最终
`docs/design.md` 的交付结构。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

# CATLASS C++ 方案详设标准参考

本文迁移 FLA `03-solution-design` 及其 `ChunkGdnBwdIntra` 详设文档的设计结构，作为
`catlass-linear-attention-v1` 的本地标准参考。本文规定设计方法、产物和评审通过条件；算子名、公式、
shape、SoC、容量和性能数值按当前 `docs/api.md` 中已确认的需求和目标平台推导。

本文前半部分说明设计方法，后半部分的[完整详设写法示例](#a5ascend950-完整详设写法示例)展示填写后的六章文档。
生成详设时按示例的表达粒度，将当前公式、地址、生命周期和同步写成相互对应的完整方案。
示例中的算法、Stage 数、CG、分片和缓冲数量是该示例的选择；当前算子按自身需求重新推导。

涉及单位下三角矩阵求逆时，同时读取 [`matrix-inverse-patterns.md`](../operator/linear-attention/matrix-inverse-patterns.md)，先完成块结构分解、叶子递推和 Cube MBH 的边界定义，再确定 CATLASS/Cube 组件与地址布局。

## 设计阶段总原则

方案设计必须分两步依次完成：

1. **Stage 划分**：回答计算如何分组、数据放在哪里、依赖和资源是否成立；
2. **具体详设**：在已确认的 Stage 划分上，补齐实现、调度、地址、同步、tiling、精度、性能和测试。

第二步发现依赖、数据落点或资源假设不成立时，必须退回第一步重新划分和评审，通过后继续具体详设。

## 阶段输入与变更边界

进入设计前必须具备：

- 已确认的 `docs/api.md`，包括输入输出、支持范围和性能目标；
- 已实际运行并验收通过的 PyTorch CPU 标杆、基础用例和精度策略；
- 目标 SoC、CANN 版本、CATLASS 版本及工程约束；
- 当前算子的实现、测试、性能分析结果和已有设计（如为特性修改或优化任务）。

特性修改只设计接口确认阶段批准的差异，并列出保持不变的接口、计算规则、支持场景、标杆和回归范围。
如果必须改变公开接口、语义或支持范围，返回 01；如果必须改变标杆或预期结果，返回 02。

优化任务必须先还原当前代码的 Stage、workspace、同步、tiling、模板和任务划分，固定输入、SoC、
构建产物和性能采集方法，确认精度与性能基线后再设计候选方案。候选方案需记录瓶颈依据、预期收益、
资源代价、精度/同步风险和回退方式。

## 第一步：Stage 划分

### 1. 建立完整计算依赖图

把计算展开为不可遗漏的节点和数据边。每个节点至少记录：公式、输入、输出、shape、dtype、有效区、
是否为最终输出，以及 Cube 或 Vector 类型。每条数据边记录生产者、消费者、首次和最后使用 Stage，
以及最终位于 L1、UB 还是 GM。

同时明确 batch、chunk、task、head 之间的映射，以及不同 head 数的比例关系、定长、变长、尾块和补齐处理。
Cube 节点包括矩阵乘及必要的矩阵累加；Vector 节点包括逐元素、归约、广播、layout、cast、copy 和 mask。

### 2. Stage 划分规则

完整规则以本目录的 [`stage-design-rules.md`](stage-design-rules.md) 为准；kernel 的 Stage 组织、
跨核 ready 和核内 event 采用 [`kernel-stage-sync-patterns.md`](kernel-stage-sync-patterns.md) 中的
本地模式，设计文档必须逐条给出 R01～R21
的满足、不适用或兜底结论。设计时至少检查：

- 单个 Stage 不混合 Cube 与 Vector，依赖关系不能被 Stage 边界破坏；
- L1、UB、GM 的容量、连续地址、驻留份数和生命周期有证明；
- Cube/Vector 结果按后继消费者选择 L1、UB 或 GM，重复搬运只作为容量冲突兜底；
- L2 缓存参与策略按 [`stage-design-rules.md` 的 R10](stage-design-rules.md#r10-gm-搬运次数) 检查，并在 `docs/design.md` 记录 API 支持和实测结果；
- 无依赖 Stage 的并行关系、连续同类 Stage 的数量和不同 head 的一致数据落点均有说明；
- Stage 先按依赖、计算类型、生命周期和精度观察点划分，再映射到 AIC/AIV；同一 AIC/AIV
  顺序执行多个 Stage 时，必须保留各 Stage 的逻辑边界和数据交接说明；
- tail、partial、空任务和无效区通过 padding、中性值和 valid mask 处理，不改变主路径语义；
- kernel Stage 必须说明各 AIC/AIV 的负责范围（owner）、任务分组、双缓冲（ping/pong）、
  CrossCore 数据就绪通知、HardEvent 的发送和等待配对，以及 workspace 缓冲槽（slot）的复用；
- 按 R01 选择逐 head 或同 head 分片，给出各 AIV 的数据范围和选择依据；分片的归约合并、
  空片通知以及跨 Stage 分工变化均落实到依赖图和同步伪代码；
- 在满足依赖、容量、生命周期和 head 一致性的前提下，Stage 数量尽可能少。

### 3. 第一步必须产出

1. 完整依赖图：全部公式节点、Cube/Vector 类型、数据边和最终输出；
2. Stage 依赖表：类型、公式、前驱、可并行 Stage、输入来源、输出落点、任务映射和任务数；
3. 跨 Stage 数据表：shape、dtype、生产/消费 Stage、最后消费者、存放位置和份数依据；
4. L1/UB 地址图：offset、大小、对齐、份数、负责该区域的核、首次写入、最后消费、峰值和碎片证明；
5. GM/workspace 表：中间量 offset、大小、对齐、生命周期、复用条件和重复搬运代价；
6. 调度一致性说明：并行关系、连续同类 Stage、task/head 映射和不同 head 的统一处理；
7. Stage 与 AIC/AIV 映射说明：每个 Stage 的 owner、任务范围、执行顺序以及是否复用同一执行单元；
8. R01～R21 逐条规则检查表和证据位置。

### 4. 第一步评审通过条件

- 所有计算节点恰好分配到一个 Stage，Cube/Vector 类型没有混用；
- 所有依赖边都能由合法的前序或同 Stage Vector 结果提供；
- L1/UB 峰值、地址连续性、驻留份数和 GM 回写均有可复核依据；
- Stage 划分依据与 AIC/AIV 映射分别可复核；同一执行单元承载多个 Stage 时，依赖和生命周期边界仍清晰；
- 已解释 Stage 数量为何不能继续减少，连续同类 Stage 为何不能进一步合并或交错；
- 任一无法满足的规则都明确标记为阻塞并在 Stage 划分阶段解决；全部通过后进入具体详设。

## 第二步：具体详设

以第一步已确认的 Stage 表、数据落点和容量布局为固定输入，按
[`design.md.template`](../../skills/catlass-cpp-design/templates/design.md.template) 的以下 6 个章节编写 `docs/design.md`：

1. **目标与数学语义**：SoC、dtype、layout、关键维度、支持场景和性能目标；与 `docs/api.md`
   一致的输入、输出、属性、异常和兼容性摘要；完整公式、计算顺序、中间量、边界行为和
   PyTorch CPU 标杆对应关系。
2. **Stage 总览与完整详设**：完整依赖图、Stage 的先后及并行关系、Cube/Vector 分工和调用图；逐 Stage
   的公式、shape、有效区、地址、GM 搬运、任务映射和边界处理；AIC/AIV 分工、CATLASS 配置、
   Vector API、搬运接口和输出写回；每个 Stage 及其数据交接的同步伪代码，覆盖生产者和消费者、
   数据就绪与可复用通知（ready/free）、CrossCore flag、HardEvent、缓冲槽复用、空任务、尾块和异常路径。
3. **全局资源、Tiling 和 Workspace**：L1、UB、GM/workspace、L0、slot、flag、生命周期和
   同时存活的数据；Host tiling 的模板、任务数、offset、TilingKey、SoC 差异、参数校验和 workspace 计算。
4. **精度、性能和测试计划**：Stage 可观察点、统一精度策略、profiling 指标、目标 shape 和回归范围。
5. **风险、兼容和回退方案**：兼容策略、回退条件、未决问题、交付清单和验证证据。
6. **R01-R21 规则检查表**：每条规则填写满足、不适用或第 14 条兜底结论及证据位置；
   不适用项说明原因，兜底项说明触发条件、替代方案和资源或搬运代价。

两步设计的材料分别落入上述章节：依赖图和 Stage 表放第 2 章，容量与地址布局放第 3 章，
规则评审证据汇总到第 6 章；完成具体详设时在原位置补齐内容。

### 逐 Stage 详设最低要求

每个 Stage 必须单独写清：

- 公式、计算顺序、完整 shape、有效长度和 dtype；
- 输入、中间量、输出的地址区间、大小、对齐、份数、负责该区域的核、保留和释放时机；
- GM 搬入、GM 写回、跨 Stage 驻留和 workspace 复用步骤；
- task、batch、chunk、head、head ratio 映射及不同 head 的一致处理；
- fixed、varlen、tail、padding、空任务和无效区行为；
- 与前后 Stage 的依赖、同步事件、ready/free 通知顺序和可并行关系。

Cube Stage 还必须写明物理 layout、转置、L1/L0A/L0B/L0C、MMAD 累加和 Fixpipe 写出。Vector Stage
必须对应第一步定义的完整输入和执行模型：A2/A3 写清 MemBase 的分段搬运、队列和 pass；A5 选择
RegBase/VF 时写清 VF 调用边界、寄存器中间量和 mask。一次 VF 只能作为满足语义与资源条件后的
A5 候选，不能作为跨架构强制约束；需要调整 pass 或 tile 划分时，返回第一步评审确认。

### 全局资源、tiling 和同步

- 同时给出逐 Stage 和全局同时存活数据的 L1/UB 峰值、最大连续空闲区和碎片证明；
- workspace 每个区域说明生产者、消费者、offset、大小、对齐、生命周期、释放和复用条件；
- 按 `kernel-stage-sync-patterns.md` 检查每条跨核边和核内边：flag/event 必须成对生产、
  消费、排空和释放；采用聚合通知时，未负责当前数据的参与核（non-owner）也必须按角色发送或等待对应通知（token）；
- 显式推导输出 head 与 Q/K head 的 ratio，以及 fixed、varlen、tail 的任务映射；
- Host tiling 校验 shape、属性、cu_seqlens/chunk_indices、尾块和 workspace 上限；
- 编译期模板承载 dtype、固定维度和核心路径选项，运行时 tiling 承载规模、offset、layout 和任务划分；
- 枚举全部可达 TilingKey 及选择条件，平台和模板选择作为实现配置，公开接口遵循 `docs/api.md`；
- 搬运和计算采用批量路径，避免热路径逐元素标量访问；
- tail/partial 优先用 padding、中性值和有效区 mask 保持已设计的 Cube/Vector 主路径；
- 跨 pipe、跨核和 producer-consumer 数据边使用对应的 HardEvent/CrossCore 通知；每个 event/flag
  必须成对生产、消费和复用，并在 kernel 退出前排空；
- 公共组件、调用参数的类型或内存布局、生成模板或运行时实现变化时，列出并测试全部受影响算子。

## 精度与性能详设

精度计划必须为关键中间量、Stage、chunk、head、状态和最终输出定义观察点，写明
累加 dtype、workspace dtype、计算顺序、cast 时机、统一精度策略和阈值依据，并区分结构性错误、数值误差、
padding/无效区和非确定性问题。

如果精度定位表明错误来自 Stage 划分、数据依赖、workspace/片上数据的生命周期、缓冲槽分配、各核分工、同步或 tiling，
按 [`catlass-cpp-test`](../../skills/catlass-cpp-test/SKILL.md) 的失败恢复规则和 [`catlass-cpp-develop`](../../skills/catlass-cpp-develop/SKILL.md) 的状态约定进入 `design_issue` 分支；
更新 `docs/design.md`、规则证据和同步伪代码，重新通过 `validate_design.py` 和设计评审后，
按同一表切换到开发阶段的定向精度修复状态。最终验收使用 full 范围结果。

性能计划先判断主要瓶颈来自计算、搬运还是同步等待，再定义目标 shape、当前基线、目标值、其他形状的
回归范围和模板适用范围。Scalar、MTE、VEC、CUBE 和 AIC/AIV 等待分别指定性能分析证据；
候选优化必须保持既有功能范围和正确实现路径。

## 第二步评审通过条件

- 详设与第一步确认的 Stage 类型、依赖、数据落点、驻留份数和容量布局一致；
- 已确认的输入输出和计算规则、PyTorch CPU 标杆、数学公式、依赖图和逐 Stage 公式一致；
- 每个 L1/UB/GM 区域、workspace、缓冲槽、flag 和 event 都明确唯一的管理方及分配、使用、释放时机；
- 所有支持维度、边界、SoC、TilingKey 和可达分支都有实现及测试计划；
- 特性修改覆盖已确认差异，并为保持不变的既有行为定义回归范围；
- 优化任务记录基线、瓶颈证据、Stage 复核、前后比较口径和预期收益；
- 风险、兼容策略、回退方案和未决问题明确。

两步均评审通过后，以已确认的 Stage 划分和具体详设作为 04 算子开发的实现依据。

## A5/Ascend950 完整详设写法示例

下面用一个 chunk 内门控矩阵乘展示 A5/Ascend950 的完整写法。它具有 Cube → Vector → Cube 依赖、
共享 key head、尾块和 ready/free 闭环，便于同时核对公式、地址和调度。
示例采用本地 `SYNC_GM_QUEUE`，与 R06 的 Vector→Cube 交接一致；Cube→Vector 的 Fixpipe 直达
AIV UB、RegBase/VF 和相应资源预算是 A5 示例内容，不能原样投射到 A2/A3。
片上直连、跨 work 流水和逐 head 分工作为按当前平台评估的其他候选，选择方法见
[`kernel-stage-sync-patterns.md`](kernel-stage-sync-patterns.md)。[^catlass-arch-separation]

A2/A3 生成同类设计时保留公式、Stage 依赖、AIC/AIV 分工和 ready/free 闭环，但必须做以下对等替换：

| 设计项 | A5 示例 | A2/A3 设计要求 |
|---|---|---|
| 架构标识 | `3510`、`Arch::Ascend950` | `2201`、`Arch::AtlasA2` |
| Vector 模型 | RegBase/VF，可在条件满足时一次 VF | MemBase，显式 `TPipe`/`TQue`/`TBuf`、pass 和事件 |
| Cube→Vector | 组件支持时 `L0C -> AIV UB`，可 dual-destination | `L0C -> Fixpipe -> GM -> UB` |
| Vector→Cube | 本例为 GM；A5 可另评估 `AIV UB -> AIC L1` | `UB -> GM workspace -> L1` |
| 组件与资源 | Ascend950 组件；UB 248 KiB、L0C 256 KiB、FixBuf 16 KiB | AtlasA2/`MmadAtlasA2*`/`FixpipeParamsV220`；UB 192 KiB、L0C 128 KiB、FixBuf 7 KiB |

A2/A3 的真实详设必须重新计算 GM 流量、workspace、UB/L0C 容量、事件和性能，不能只替换类型名。

以下容量是手算的设计用量，性能和设备测试状态为待执行。真实算子的第 1、3、5 章需要填写
目标 CANN/CATLASS 版本、组件源码依据、实际可用容量和验证结论。
阅读时关注同一变量如何贯穿公式、资源表、同步伪代码和规则证据。

# catlass_chunk_gated_product 设计（写法示例）

```text
workflow_id: catlass-linear-attention-v1
design_rule_version: V1
sync_pattern: SYNC_GM_QUEUE
pipeline_pattern: PIPE_SERIAL
status: design_example
```

## 1. 目标与数学语义

### 1.1 目标与范围

示例演示 Ascend950 上的 chunk 内计算。输入 q/k/x 为 BF16，g 为 FP32；矩阵乘使用
FP32 累加。`BT=64`、`K=V=128`，`HV=2*HK`，chunk 之间独立。定长支持 `B>=1`、
任意正 T；变长使用 `B=1` 的拼接序列和已校验的 chunk 索引，支持不足 BT 的尾块。
此处的 `HV/HK=2` 是示例范围，生成其他算子时以它自己的 API 支持范围为准。
示例数学有效域为 `abs(q/k/x)<=1`、`g in [-1,1]`、`abs(scale)<=1/sqrt(128)`。
输入生成与标杆验收使用该范围，数值证明见第 5 章；当前算子的值域按自身数学约定确定。

演示模型为 `B=1,T=11264,HK=16,HV=32`。性能目标由当前任务的用户需求给出；本写法示例
仅提供测量方案，实测基线、目标和结果在真实任务中填写。

### 1.2 输入输出和调用方式

示例入口为 `chunk_gated_product(q,k,x,g,scale,cu_seqlens,chunk_indices) -> o`。
q/k 的 shape 为 `[B,T,HK,128]`，x/o 为 `[B,T,HV,128]`，g 为 `[B,T,HV]`，均为连续
token-major 布局。`scale` 是处于 1.1 范围内的有限标量；定长时两个索引参数为空，变长时共同描述有效 chunk。
head 数由 tensor shape 推导。Host 校验完成后，通过 ACL 和 `<<<>>>` 启动 device kernel。

### 1.3 完整数学语义

令 `hk=floor(hv/2)`，chunk 有效长度为 `M<=BT`，`t,s` 为 chunk 内 token 索引。
q/k/x 补齐行填零。矩阵乘用 `@`，其中 `.T` 表示右操作数的转置读取语义。

```text
S_hk[BT,BT] = FP32(q_hk[BT,K]) @ FP32(k_hk[BT,K]).T       # FP32 累加
valid[t,s]  = (0 <= s <= t < M)
Delta_hv[t,s] = FP32(g_hv[t]) - FP32(g_hv[s])             # 仅有效位置求值
D32_hv[t,s]   = scale * S_hk[t,s] * exp2(Delta_hv[t,s])   # 仅有效位置求值
D_hv[BT,BT]   = BF16(D32_hv)，无效位置直接写 0
O32_hv[BT,V]  = FP32(D_hv[BT,BT]) @ FP32(x_hv[BT,V])      # FP32 累加
o_hv[M,V]     = BF16(O32_hv[0:M,:])
```

这里 g 是每 token、每 value head 的标量，Delta 可在 score 之后逐元素计算。当前算子若有
逐通道 gate，先推导其带 shape 的矩阵表达式和数值边界，再确定 Cube 输入。
示例直接计算有效位置的 FP32 差值后执行 exp2；因果及尾块 mask 在 exp2 前判断。
CPU/PyTorch 标杆保持同样的 Delta、D 的 BF16 舍入和两次 FP32 累加顺序。

## 2. Stage 总览与完整详设

### 2.1 依赖图与 Stage 总览

本例先按两次矩阵乘之间的逐元素依赖划分三个 Stage，再映射执行单元。
一个 MixBlock 包含一个 AIC 和两个 AIV。示例选 `CG=2`，每个 work 处理同一 chunk 的
两个连续 value head，它们共享一份 S。两个 AIV 各处理同一 head 的连续 32 行，均遍历 r。
每片计算独立，完整 g 在各片 UB 中保留，分片之间的归约量为零。

```text
hv_begin = group * CG; hv = hv_begin + r; hk = floor(hv/2)
valid_r = min(CG, HV-hv_begin); r in [0,valid_r)
part = 0 or 1; row_begin = part*32; rows = max(0,min(32,M-row_begin))
Nchunk = B*ceil(T/BT)                   # 定长
Nchunk = chunk_indices.shape[0]        # 变长，host 已过滤空 chunk
Nwork = Nchunk*ceil(HV/CG)
blockDim = min(可用 AIC 核数,Nwork)
work_id = block_idx + n*blockDim
chunk_id = floor(work_id/ceil(HV/CG)); group = work_id % ceil(HV/CG)
```

本例的合法 HV 恒为偶数，完整 work 的 `valid_r=2`；示例仍显式写有效 r 和行范围。
采用支持尾 head 组的接口时，按实际 HV/CG 推导尾组范围。模型有 `176*16=2816` 个 work。
CG 和分工的选择依据是演示一次 score 供两次消费的生命周期；性能优劣留待同条件比较。

```text
GM q/k -> S0 Cube -> S 的两个 UB 行片 -> S1 Vector -> GM D -> S2 Cube -> GM o
                                                    GM x ------^
GM g -------------------------------> S1
```

| Stage | 类型 / 执行单元 | 任务与公式 | 输入来源、输出及最后消费者 |
|---|---|---|---|
| S0 Score | Cube / AIC | 每 work 一次 `q_hk @ k_hk.T` | q/k 来自 GM；FP32 S 写两个 AIV UB，保留到 S1 完成最后一个 r |
| S1 Gate | Vector / 两个 AIV | 每 r 各处理 32 行 Delta、gate、mask、cast | S 驻 UB，g 来自 GM；BF16 D 两片写同一 GM slot，S2 MTE2 最后读取 |
| S2 Output | Cube / AIC | 每 r 一次 `D_hv @ x_hv` | D 从 workspace、x 从输入 GM 读取；Fixpipe 直接写最终 o |

同 work 中 S1 依赖 S0，S2 依赖 S1；S1(r+1) 与 S2(r) 可重叠，各自使用独立的 D 槽。
本例按 work 顺序推进，跨 work 流水作为后续候选。S0 与 S2 中间有真实 Vector 数据依赖，
因此保留三个逻辑 Stage；AIC 同时承载 S0、S2，两个 AIV 承载 S1。

### 2.2 Stage 0：Cube，生成共享 Score

公式为 `S_hk[64,64]=FP32(q_hk[64,128]) @ FP32(k_hk[64,128]).T`。
输入 dtype 为 BF16、累加和 UB 结果为 FP32；有效 shape 为 `[M,M]`。
task/head 映射见 2.1，同一 hk 的 S 只生成一份，分别交付两个 AIV 的行片。

地址采用第 3 章的 L1 q/k 区和各 AIV `UB[0,8 KiB)`。q/k 在 L1 为 Cube 可读的
NZ tile，k 以转置语义载入 L0B。L0C 的 64×64 FP32 结果占 16 KiB，使用全局 L0C
槽的前半段；Fixpipe 将前、后 32 行分别写入对应 AIV 的 ND Score 区。

执行顺序：

1. MTE2 搬入当前 q/k；tail 只读 M 行并将物理 tile 的其余行补零。
2. MTE2→MTE1、MTE1→Cube 按本地事件完成交接，MMAD 首次累加清零。
3. 等待上个 work 的两个 `score_free`，确认目标 UB 的生命周期已结束。
4. Cube→Fixpipe 交接后写两个 UB 行片，写入完成才发布各片 `score_ready`。

当 `rows=0` 时，该片跳过实际 Fixpipe 数据写入，仍消费和发布本轮同步通知。
S0 的精度观察点是有效 S；UB 的最后消费者是 S1 的最后一个有效 r，释放由该 AIV 发布。

### 2.3 Stage 1：Vector，生成 Gate 和 D

每个 AIV 执行 1.3 中自己负责的 D 行片公式。输入 shape 为 FP32 `S_part[32,64]`
和 FP32 `g_r[64]`，输出 dtype 为 BF16、shape 为 `D_part[32,64]`。
task 内 r 顺序和映射与 AIC 相同；当前 work 的两个 head 的 g 一次成组搬入，每个 r 使用一次 VF
完成差值、exp2、scale、mask 和 cast。Delta 等逐行保存在寄存器中。

g 的 GM token 步长为 `HV*4 B`。每 token 读取当前两个连续 head 的 8 B，在 UB 中按
32 B 物理行保存；`g_record[64,8]` 共占 2 KiB，VF 用 `g_record[t,r]` 读取有效值。
二维搬运参数为有效行长 8 B、GM 行步长 `HV*4 B`、UB 行步长 32 B、有效行数 M；
由目标版本支持的非对齐搬运与 padding 接口实现，其支持情况列入第 5 章核验项。
地址采用 `UB[0,8 KiB)` 的 Score 驻留区、一个 g_record 区和两个 D 行片槽。
`D_part` 对应 GM 的地址为 `D_base(block_idx,r)+part*4096`，两片连续且不重叠。
Score 和 g_record 的生命周期跨越 r=0、r=1；最后一个 r 的 VF 结束后释放，D 行片在本次 MTE3 读完后释放。
与前后 Stage 的同步分别使用 `score_ready/score_free` 和 `d_ready/d_free`，见 2.5。

执行顺序：

1. MTE2 搬入当前 work 的 g_record，通过 MTE2→V 事件交给 VF；等待 `score_ready` 后消费 S。
2. 等待本地 D 槽上次 MTE3 读完，然后计算当前 r 的 32 行；因果及 tail 无效位置直接置零。
3. 等待 GM D 槽上次 `d_free`；V→MTE3 交接后写自己的物理行片，再从 MTE3 发布 `d_ready`。
4. 最后一个 r 的 VF 消费完 Score 后，从 V 发布 `score_free`。D 的 GM 写回仍由独立事件保护。

`rows=0` 的片跳过 VF 和搬运，但参与 ready/free。S2 根据 M 在 L1 中补齐缺失行，
保证空片的 GM 区域不会成为未初始化输入。精度观察点为 BF16 D、因果零区和尾块边界。

### 2.4 Stage 2：Cube，计算输出

公式为 `O32_hv[64,128]=FP32(D_hv[64,64]) @ FP32(x_hv[64,128])`，再由 Fixpipe
转换为 BF16 输出 o。D/x dtype 为 BF16，累加 dtype 为 FP32，有效输出 shape 为 `[M,128]`。
每个 head 的执行顺序与 2.1 的 r 一致。

地址采用第 3 章 `D_L1[r]`、`X_L1[r]`，两个 r 各占独立槽。AIC 等待两个 AIV 的
`d_ready` 后，MTE2 只读取 D 的 M 行和 x 的 M 行，L1 剩余行补零；随后发布两个 `d_free`。
这里的 free 表示 GM 最后读取完成，L1 与 L0 生命周期由各自本地事件保护。
该同步边依赖两个实际写入方完成，后续 MMAD 读取依赖本核 MTE2→MTE1 交接。

MTE1 将 NZ 输入读入 L0A/L0B，Cube 完成 FP32 MMAD，Fixpipe 把有效 M 行写入
`o[b,chunk_start:chunk_start+M,hv,:]`。完整输出行 256 B，GM 行步长为 `HV*256 B`；
实际写回接口需支持该二维 stride，输出地址按真实 descriptor 计算。
最终 o 的消费者是调用方，kernel 结束前等待全部 Fixpipe 完成。
varlen 的 GM 基址使用所属 sequence 的起点；空 chunk 已在 host 过滤，tail 仅写有效行。

### 2.5 Stage 间同步方案

以下是设计伪代码。`publish/wait` 表示 CrossCore 通知，逻辑键为
`(work_id,part,r)`；物理 flag 由有界槽位循环复用，使用相同的发布/消费次序。
带 pipe 的完成条件对应实际搬运完成；核内 `Local*` 用已核对的 HardEvent 或 Mutex 实现。

| 通知 | 发布方与条件 | 消费方与地址保护 | 每 work 的逻辑通知数 |
|---|---|---|---|
| score_ready[part] | AIC：该片 Fixpipe 写入完成 | AIV：读取 UB Score | 每片 1 次 |
| score_free[part] | AIV：最后一个 r 的 V 读取完成 | AIC：下次覆盖该 UB Score | 每片 1 次 |
| d_ready[part,r] | AIV：当前 D 行片 MTE3 写入完成 | AIC：读取完整 GM D | 每片每 r 1 次 |
| d_free[part,r] | AIC：当前 GM D 的 MTE2 读取完成 | AIV：下次覆盖该 GM 行片 | 每片每 r 1 次 |

每个 GM D 槽及 Score 区初始为空闲，首轮直接使用；后续覆盖时消费上一轮 free。
最后一轮 free 由退出排空过程消费。聚合或广播的物理 flag 数和计数深度按目标 API 映射，
把上表的每个参与者及通知次数保持完整。每片空行时仍参与相同通知闭环。

```text
Stage0AIC(item):
  LocalLoadQK(item)                   # MTE2 -> MTE1 -> Cube；q/k tail 补零
  LocalMmadScore()                    # Cube -> L0C，等待 M_FIX
  if has_previous_score:
    wait score_free[0], score_free[1]
  Fixpipe score -> AIV[part].S for each part with rows > 0
  publish score_ready[part] after PIPE_FIX for both parts
  has_previous_score = true

Stage1AIV(part,item):
  if rows > 0:
    LocalLoadG(item,part)             # 上轮末次 V 读完才复用；成组搬入 g_record 后 MTE2_V
  wait score_ready[part]
  for r in [0,valid_r):
    LocalWaitDOutputFree(r)           # 上次 MTE3 读完，V 才能覆盖 UB D[r]
    if rows > 0:
      OneVF(S_part,g_record[:,r]) -> D_part[r] # 先判断 mask，再 exp2；一次 BF16 cast
    if r is last:
      publish score_free[part] after PIPE_V
    if gm_slot_was_used[r]:
      wait d_free[part,r]
    if rows > 0:
      LocalVectorToStore(r)           # V_MTE3 SetFlag/WaitFlag 或对应 Mutex
      MTE3 D_part[r] -> GM D[r,part]
    publish d_ready[part,r] after PIPE_MTE3
    gm_slot_was_used[r] = true

Stage2AIC(item):
  for r in [0,valid_r):
    wait d_ready[0,r], d_ready[1,r]
    LocalWaitL1Free(r)                # 上次 MTE1 读完，MTE2 才能覆盖
    MTE2 GM D[r,0:M,:], x[r,0:M,:] -> L1 D[r], X[r]; pad rows to BT
    publish d_free[part,r] after PIPE_MTE2 for both parts
    LocalMmadOutput(r)                # MTE2_MTE1 -> MTE1_M -> M_FIX
    Fixpipe L0C[r] -> o[item,hv_begin+r,0:M,:]

Process(block_idx):
  allocate local buffers, events and CrossCore flag mapping
  initialize every local buffer slot as free, with no pending event
  parallel:
    AIC:
      has_previous_score = false
      for item in Schedule(block_idx,Nwork,blockDim):
        Stage0AIC(item)
        Stage2AIC(item)
      wait final score_free[0], score_free[1] if any work executed
      drain issued local events and final Fixpipe
    AIV part in [0,2):
      gm_slot_was_used[0:CG] = false
      for item in same Schedule:
        Stage1AIV(part,item)
      wait final d_free[part,r] for each used r
      drain issued local events and final MTE3
  release events and flag resources after their final consumers
```

`LocalMmad*` 管理同一套物理 L0 缓冲。每个 L0A/B 槽由 MTE1 写、Cube 读；每个
L0C 槽由 Cube 写、Fixpipe 读。前者按 `MTE1_M/M_MTE1`、后者按 `M_FIX/FIX_M`
闭环复用。L1 按 `MTE2_MTE1/MTE1_MTE2` 闭环；g_record 输入区按
`MTE2_V/V_MTE2` 闭环，最后一个 r 的 VF 结束后才交回 MTE2；每个 D 输出槽按 `V_MTE3/MTE3_V` 闭环。
事件分配表以“事件类型、物理槽、生产 pipe、消费 pipe、首次使用、末次排空”为列填写。
不同 MMAD 复用同一地址时由同一资源管理方持有这些事件，组件内部事件也计入目标平台上限。

### 2.6 Kernel 组件与调用

Host main 完成 ACL 初始化、输入输出及 workspace 分配、tiling 和 kernel 直调。
kernel 的 AIC 入口执行 Stage0/2，AIV 入口执行 Stage1，按 2.5 的 Schedule 领取相同 work。
CATLASS Cube 组件承载 BF16 输入、FP32 累加的 `[64,64,128]` 和 `[64,128,64]`
两种 GEMM（顺序均为 M/N/K）；Vector 使用当前平台的 VF、mask、exp2 和 cast。

真实算子的组件表逐项填写“Stage、ArchTag、组件名称、源码相对路径、TileShape、layout、
累加与 Fixpipe 配置、已核对的版本”。本例的组件参数是语义需求，具体类型通过目标版本源码
核验后填写。host/kernel/测试共用已确认的参数顺序和输出约定。

## 3. 全局资源、Tiling 和 Workspace

### 3.1 地址布局和峰值

以下地址是每个 MixBlock 的相对 offset，单位 KiB，所有大块按 512 B 对齐。
`L1_limit/UB_limit/L0*_limit` 表示 precheck 实际确认可供本算子使用的容量，含组件预留扣除。

| 空间与地址 | 内容、shape、dtype、份数 | 首次写入 → 最后消费 → 复用条件 |
|---|---|---|
| AIC L1 [0,16) | q `[64,128]` BF16，1 份 | S0 MTE2 → S0 MTE1 → 对应 MTE1_MTE2 完成 |
| AIC L1 [16,32) | k `[64,128]` BF16，1 份 | 同 q；后续 value head 复用 Score |
| AIC L1 [32,48) | D `[64,64]` BF16，2 份，每份 8 | S2 MTE2 → S2 MTE1 → 每槽读取完成 |
| AIC L1 [48,80) | x `[64,128]` BF16，2 份，每份 16 | 同 D，r 的 x 对应 slot r |
| 各 AIV UB [0,8) | S_part `[32,64]` FP32，1 份 | S0 Fixpipe → S1 最后一个 r 的 VF → score_free |
| 各 AIV UB [8,10) | g_record `[64,8]` FP32，1 份；每行前 2 项有效 | S1 MTE2 → 最后一个 r 的 VF → V_MTE2 |
| 各 AIV UB [10,18) | D_part `[32,64]` BF16，2 份，每份 4 | S1 V → S1 MTE3 → MTE3_V |
| AIC L0A/L0B 各 [0,32) | 两份各 16 KiB 输入槽 | MTE1 → Cube → M_MTE1；S0/S2 分时复用 |
| AIC L0C [0,64) | 两份各 32 KiB FP32 累加槽 | Cube → Fixpipe → FIX_M；S0 使用其中 16 KiB |

最大同时存活预留量为 L1 80 KiB、每个 AIV UB 18 KiB、L0A/B 各 32 KiB、L0C 64 KiB。
g_record 每行有效 8 B、保留 32 B，padding 计入预算；VF 中间计算使用寄存器。
总空闲区与最大连续空闲区分别为 `L1_limit-80`、`UB_limit-18`，因为有效分区连续排列，
内部对齐空隙已计入预留量。S1(r+1) 与 S2(r) 重叠时仍适用同一峰值。
核对生成组件新增的 TBuf、临时区和编译器分配后再确认最终资源通过。

### 3.2 GM workspace 与搬运量

```text
matrix_bytes = 64*64*2 = 8192
D_base(block_idx,r) = workspace + (block_idx*2+r)*matrix_bytes
D_part_base = D_base + part*32*64*2
workspace_bytes = AlignUp(blockDim*2*matrix_bytes,512)
```

每个 AIC 保留两个 8 KiB D 槽；同一槽跨 work 复用，生产者为两个 AIV，消费者为该 AIC。
每片最多写 4 KiB，只有两个 `d_ready` 到达后读取；MTE2 读完后发布 free。
同一物理 GM 行片被覆盖前必须消费 free。各 block_idx 的 workspace 区间完全分离。

完整 work 从 GM 读取 q/k 共 32 KiB、两个 x 共 32 KiB、两个 g 在两个 AIV 各读一份
共 1 KiB；D 中转写 16 KiB、读 16 KiB；最终 o 写 32 KiB。每 work 的显式 GM 流量
合计 129 KiB，其中 D 交接 32 KiB。Score 共享使 q/k 的两次 value-head 消费只需一次搬入。
一次性 q/k/x 输入的缓存参与策略按目标搬运 API 和同条件性能结果选择；D 的跨核可见性与缓存
一致性按 `SYNC_GM_QUEUE` 核验。tail 按有效行减少搬运，仍使用相同的物理槽跨度。

### 3.3 Tiling 与边界

host 从 tensor descriptor 推导 B/T/HK/HV 和 stride，校验维度、dtype、连续布局、
`HV=2*HK`、BT/K/V、scale 的有限性及 1.1 中的范围；变长索引必须有序且落在所属 sequence 内。
kernel 接收 `Nchunk,Nwork,blockDim,stride,workspace_bytes` 及 chunk 映射数据。
示例按定长/变长、完整 chunk/tail 选择分支，G/CG/BT/K/V 是本例编译期配置。
实际工程为全部可达组合分配 TilingKey，并用用例验证分支选择。
Nwork 为零时 host 返回空结果；进入 kernel 的 work 保证 `1<=M<=64`。

## 4. 精度、性能和测试计划

精度分别检查 S、D、O32 和 BF16 o，观察点覆盖每个 chunk/head、因果对角线、上三角零区、
tail 和两个 AIV 分片边界。采用本工作流的统一精度策略，阈值由当前 CPU 标杆校准。
一个 Stage 的定向精度通过后接入下一 Stage，完整实现通过全量验收后采集性能。

| 测试项 | 具体覆盖 | 检查内容 |
|---|---|---|
| token 边界 | T=1/31/32/33/63/64/65、多个 chunk | 空片、分片交界、tail padding 和有效写回 |
| head 和 batch | HK=1/3/16，HV=2*HK；B=1/2 | 相同 hk 的两次消费、奇数 HK、任务余数和输出地址 |
| varlen | 含零长度序列，以及长度 1/33/65 的序列 | 空 chunk 过滤、索引及不同尾块连续执行 |
| 数值 | 1.1 值域内的零输入、边界值、正负混合和不同 scale | Delta/mask/cast 时机及相消误差 |
| 同步 | 多 work 复用相同槽、重复 launch | ready/free 平衡、旧数据覆盖、最终排空 |
| 非法输入 | 维度、head 比例、索引越界、非有限 scale | host 拦截，与 API 一致 |

性能记录模型 shape、构建版本、预热和采样次数，以 `msprof op_summary` 的
`Task Duration(us)` 中位数及分位数比较。关注 S1 exp2、D 的 GM 中转、MTE 等待和 AIC
空闲时间。每次仅改变一个主要候选：CG、AIV 分工、片上交接或跨 work 重叠。
记录计算、容量、搬运和同步代价，收益需超过测量波动。Sanitizer 分别覆盖越界、未初始化、
竞争和同步问题，并确认实际加载带检测能力的 kernel。

## 5. 风险、兼容和回退方案

本例数值边界采用 1.1 中的 `abs(q/k/x)<=1`、`g in [-1,1]`、`abs(scale)<=1/sqrt(128)`。
据此 `abs(S)<=128`、`Delta in [-2,2]`、`exp2(Delta)<=4`，D32 的绝对值小于 46，
BF16 D 的绝对值也可用 46 作保守界，`abs(O32)<=64*46=2944`。
无效位置在 exp2 前排除；浮点累加和 cast 误差通过第 4 章观察点与 CPU 标杆比较。
真实输入范围超出此示例时，按 R21 重新推导范围和误差预算。

资源、CrossCore 模式、A5 远端 UB 写入、g 的 8 B 二维有效搬运和带 stride 的 Fixpipe 支持需由实际
CANN/CATLASS 版本核验；A2/A3 还需核对 GM 中转、MemBase 与 `FixpipeParamsV220`。本例是 A5
写法示范，设备编译、精度、Sanitizer、性能状态均为待执行。
开发前将实际组件和事件分配证据填入 2.5/2.6，并核对第 3 章的容量。
发现容量或生命周期冲突时，按 R14 计算 GM 中转代价；流水候选失败时恢复已验证的基线。
设计调整统一回写六章的对应公式、Stage、地址和伪代码，实验过程保存到 `docs/validation.md`。
涉及公开接口或标杆语义的变化，按主 Skill 恢复到对应阶段。

## 6. R01-R21 规则检查表

下面展示本例静态设计的规则证据写法；“满足”指上述示例假设下的推导结论。
第 5 章列出的组件及设备证据仍须在真实任务中补齐，整体评审通过后才能进入开发阶段。

| 规则 | 结论（满足/不适用/第14条兜底） | 证据位置 |
|---|---|---|
| R01 | 满足 | 2.1：先划分三类依赖段，再映射 AIC/AIV；分片范围明确 |
| R02 | 满足 | 2.1、2.5：S→D→o；消费者等待全部实际输入就绪 |
| R03 | 满足（容量条件见证据） | 3.1：L1/UB/L0 分别核算，最终需满足实际可用上限 |
| R04 | 满足 | 2.2：S 两个 FP32 行片留在各 AIV UB |
| R05 | 满足 | 2.3：Delta 等 VF 中间量留在寄存器，D 留 UB 至 MTE3 消费 |
| R06 | 满足 | 2.3～2.5：两个 D 行片经 GM 和双向通知交接 AIC |
| R07 | 不适用 | 2.1：本例没有直接 Cube→Cube 中间结果依赖 |
| R08 | 满足（容量条件见证据） | 3.1：每个 AIV 18 KiB，包含 Score、带行 padding 的 g_record 和两份 D |
| R09 | 满足 | 3.1：各 L1 槽独立，跨 work 覆盖等待 MTE1 最后读取 |
| R10 | 满足（缓存候选待实测） | 3.2：GM 流量逐项列出，q/k Score 共享，缓存策略单独核验 |
| R11 | 满足 | 2.5、3.1：Score/g/D 按最后消费者分别释放和排空 |
| R12 | 满足 | 2.3：每片成组装入 g_record，Score 驻留，每 r 一次 VF 生成 D |
| R13 | 满足 | 2.1、3.1：S1(r+1) 与 S2(r) 并行使用独立地址 |
| R14 | 不适用 | 3.2：D 的 GM 区为 R06 常规交接，本例未触发容量冲突兜底 |
| R15 | 满足 | 2.1：两次 Cube 之间有 Vector 数据依赖，三个 Stage 各自合并同类操作 |
| R16 | 不适用 | 2.3：本例的 Vector 结果 D 只有一次 Cube 消费；共享 Score 的驻留按 R04 检查 |
| R17 | 满足（容量条件见证据） | 3.1：连续地址和对齐间隙已计入峰值，最大连续空闲区可计算 |
| R18 | 满足 | 2.1、2.3：各 head 使用相同映射与算法，仅有效区参数变化 |
| R19 | 满足 | 2.1、3.1：先验证同时存活容量，再保留依赖要求的最少 Stage |
| R20 | 满足（示例候选） | 2.1、4：沿 HV 分 work、grid-stride 调度，CG/分工性能比较待执行 |
| R21 | 满足（示例值域） | 1.3、5：差值直接求 exp2、有效区先判断、范围和舍入边界明确 |

[^catlass-arch-separation]: git:cann/catlass@0d78a73c192fc04741a4fd57c94080209424775c; paths: README.md, include/catlass/arch/arch.hpp, include/catlass/gemm/tile/copy_l0c_to_gm.hpp, include/catlass/gemm/tile/copy_l0c_to_ub.hpp, include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp

[^pr1069-solution-design-reference]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/references/solution-design-reference.md
