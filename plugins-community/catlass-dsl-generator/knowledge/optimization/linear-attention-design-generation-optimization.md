---
type: CATLASS DSL Optimization Guide
title: Linear Attention 端到端设计、生成与优化
description: CATLASS Python DSL Linear Attention 的两步设计、SIMD、Stage 与同步原则，以及按计算结构选用的局部块、共享操作数和矩阵逆候选。
tags: [catlass-dsl, optimization, design, generation, linear-attention, stage, synchronization, matrix-inverse]
status: stable
generated: {by: process:catlass-dsl-pr1134-fusion, at: '2026-09-23T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-review, at: '2026-09-23T00:00:00+08:00'}
  - {by: process:catlass-dsl-kda-architecture-review, at: '2026-10-11T00:00:00+08:00'}
sources:
  - id: pr1134-design
    resource: https://gitcode.com/cann/cannbot-skills/blob/4f4e76089077fc302c410c7d12c42d85268dca7b/plugins-community/catlass-dsl-generator/knowledge/linear-attention-design/solution-design.md
    title: PR 1134 Linear Attention 两步方案设计
    kind: repository
  - id: pr1134-stage
    resource: https://gitcode.com/cann/cannbot-skills/blob/4f4e76089077fc302c410c7d12c42d85268dca7b/plugins-community/catlass-dsl-generator/knowledge/linear-attention-design/stage-design-rules.md
    title: PR 1134 Linear Attention Stage 设计规则（包含 f53617a6 完整原文）
    kind: repository
  - id: pr1134-sync
    resource: https://gitcode.com/cann/cannbot-skills/blob/4f4e76089077fc302c410c7d12c42d85268dca7b/plugins-community/catlass-dsl-generator/knowledge/linear-attention-design/kernel-stage-sync.md
    title: PR 1134 Linear Attention 任务与同步设计
    kind: repository
  - id: pr1134-inverse
    resource: https://gitcode.com/cann/cannbot-skills/blob/4f4e76089077fc302c410c7d12c42d85268dca7b/plugins-community/catlass-dsl-generator/knowledge/linear-attention-design/matrix-inverse.md
    title: PR 1134 Linear Attention 单位下三角矩阵逆
    kind: repository
operator_families: [linear-attention, gated-delta-rule, kda]
arch: [c310]
---

# 接口与概念

本条目用于 CATLASS Python DSL Linear Attention 算子的设计、代码生成和性能优化，完整融合
两步方案设计、R01-R21 Stage 规则、任务与同步生命周期以及单位下三角矩阵逆模式。[^pr1134-design]
[^pr1134-stage][^pr1134-sync][^pr1134-inverse]

## 适用范围

本文面向 Linear Attention 领域，GDN（`gated-delta-rule`）和 KDA（`kda`）均可直接检索到
本入口；多 family 标注表示共享设计知识，不表示这些算法互为同义词。本文不替代任何子族的
公式、接口或数值契约，也不宣称已覆盖其他子族的算法或性能。

| 知识层次 | 采用条件 |
| --- | --- |
| 两步设计、SIMD、依赖、地址、容量、同步与数值检查 | 按当前计算图和目标 DSL 能力采用；不适用项记录依据 |
| Cube/Vector overlap、双 AIV、任务聚合 | 存在相应执行单元及可并行的独立任务，且资源和同步成立 |
| 共享 RHS、紧凑对角 subtile、group DMA | 分别存在共享操作数、只消费对角块、连续独立数据等前提 |
| packed solve、单位下三角矩阵逆 | 数学契约确实包含对应局部 solve 或逆，不能向其他算子强加 |
| `HV/HK/CG` 分核 | 接口存在相应 head 轴与映射；其他接口按实际独立轴设计并说明不适用规则 |

它是领域入口，不是排他知识源。使用时仍应按问题查询并完整读取同一知识库中的其他条目：

- API、layout、copy、同步和控制流问题读取 `dsl/`；
- 当前 GDN/KDA/Linear Attention 公式和实现样例读取 `operator/linear-attention/`；
- 编译、lowering、运行时和精度错误读取 `debug/`；
- 指标采集和瓶颈判断读取 `profiler/`；
- tiling、buffer、pipeline、MTE、Cube/Vector、负载均衡等候选读取 `optimization/`；
- 项目实测且通过证据门禁的结论读取 `learned/`。

其他知识可以补充或否定候选，但不能覆盖当前任务冻结的公开接口、PyTorch reference、required
workload、单 kernel 契约和实际设备证据。本文中的 C++ 术语只描述必须满足的计算、数据交接和
同步语义；具体实现必须在目标版本 `python/tla_dsl` 源码中找到真实 `tla.*` API。

## SIMD 与 Cube/Vector 并行约束

本条目的算子生成规则是：**不要使用 SIMT 风格，用 SIMD 风格，充分利用 Cube 以及 Cube 和
Vector 之间的并行。**gate、mask、broadcast、layout、归约和局部 solve 使用
`tla.vec.func(mode="simd")` 的向量 tile、lane mask 和寄存器运算，不用逐线程标量循环模拟。
可表达为矩阵乘或矩阵累加的计算交给 Cube；Vector 完成准备与后处理。相关 API、dtype、layout
和执行域仍须对照目标版本核实，不能仅由存在 SIMD/Cube API 推断候选一定更快。

设计需给出 Cube 当前任务与 Vector 相邻任务准备或后处理的重叠区间、私有 slot、数据 ready
条件、最后消费者、free 条件及退出 drain；仅串行调用两类执行单元不能证明已实现重叠。
依赖确实不允许重叠时记录关键路径及限制，继续评估独立任务聚合。容量、同步或编译问题应保存
原始证据并检查合法 SIMD 路径，不静默改用 SIMT。性能接受仍以完整正确性、同配置总时延及
Cube/Vector/MTE profile 为准；该规则不承诺未测量的加速。

# 用法

## 专项检索与跨阶段采用

通用流程只负责查询和采用知识；本文提供 Linear Attention 的适用条件和进一步检索路径。
使用当前 canonical family 查询共享入口，例如 `linear-attention`、`gated-delta-rule` 或 `kda`；
`gdn` 和 `kda-linear-attention` 分别按版本化词表归一为前述已注册 family，保留子族筛选语义。
先读取通用入口，再按当前计算结构选择专项知识。开发或优化时沿用通用 Skill 的查询和记录
协议，计算图、实现条件或版本变化时重新核对采用依据，不另建阶段门禁或状态协议。

```text
读取 knowledge/optimization/index.md，按 linear-attention / design / generation 标签
定位本条目与相关专项条目；完整读取实际采用的 concept，并记录采用路径。
```

入口零命中时核对 family 归一化与项目词表，不能跳过适用约束。把规范化查询、匹配模式、版本、
采用路径与实际适用的 R01-R21、Stage、同步生命周期及矩阵逆约束写入 `DESIGN.md`。
包含 mixed Cube/Vector、重复独立局部块或共享操作数时，继续执行专项查询：

```text
继续读取 knowledge/optimization/ 下 mixed Cube/Vector、AIV/MTE、RegBase、MatMul
专项条目的 index 与正文，按当前计算结构筛选并记录采用路径。
```

按“性能架构路由门禁”选择并 `get` mixed、AIV/MTE、RegBase、MatMul 专项条目。
family 查询只返回入口或缺少相关专项条目时，保留 arch/text/tags，去掉 family 执行
cross-family fallback query，记录初始查询、fallback、score/matched_fields/matched_terms
和采用路径，不能把缺少覆盖解释为没有优化知识。专项查询使用与当前计算结构相符的词，
不强制附加子族 tag；tags 仍限制结果时按公共知识 Skill 的规则显式重查。涉及 Stage、
同步、任务映射或矩阵逆时重审适用规则。

## 设计阶段

1. 固定算子入口、输入输出、shape/dtype/layout、head ratio、fixed/varlen、chunk、tail、padding、
   state 更新、非法输入、数值顺序及 PyTorch CPU reference。
2. 先完成计算依赖图和 Stage 划分，再完成 DSL 实现详设；第二步发现容量、依赖或同步假设不成立时，
   必须退回第一步重新划分。
3. 逐条完成 R01-R21，写出实际任务映射、地址半开区间、峰值容量、最大连续空闲区、ready/free、
   事件/flag 等价能力、空任务、tail、退出排空、数值边界和证据位置。
4. 设计模板、reference/definition/workload、state 和批准门禁仍由现有专用 Skill 及通用 design
   Skill 管理；本文不另建一套机器状态协议。

## 生成阶段

1. 一次 host `run` 只 launch 一个自包含 `@tla.kernel`，多个逻辑 Stage 在该 kernel 内实现。
2. 按已批准 Stage 表逐项实现，同步更新公式、owner、有效区、地址、生命周期和 focused case；
   不得用多个 executor、Torch/vendor 计算或修改 oracle/容差绕过跨 Stage 依赖。
3. 每次只实现或验证一个可复核任务。Stage、分工、slot、数据交接或数值顺序变化时先更新设计并
   重新评审，再修改 kernel。
4. 先通过 reference/focused correctness，再做独立 review、全量 required cases、anti-hack、
   benchmark 和 profiling。

## 优化阶段

1. 从当前正确基线和 profile 判定瓶颈属于 Cube、Vector、MTE、scalar、同步等待或任务尾波。
2. 每轮只改变一个主要候选，例如 CG、逐 head/同 head 分片、片上驻留、GM 中转、slot 深度、
   pack overlap、RHS overlap、布局、搬运粒度或结构零裁剪。
3. 相同接口、workload、SoC、版本、构建和计时口径下比较；局部指标下降但总延迟回退时拒绝候选。
4. 候选必须重新通过受影响正确性、full test、final review、anti-hack、benchmark 和 profiling；
   只有实测支持的结论才能写入 learned 知识。

# 性能架构路由门禁

本文是设计规则入口，不应成为 Linear Attention 优化查询的唯一结果。设计包含 mixed Cube/Vector
与独立局部块、局部 solve 或共享操作数点积等适用结构时，继续查询并完整读取
[Mixed CUBE/VECTOR 局部块算子](mixed-cube-vector-kda.md)。根据实际瓶颈再读取：

- Vector/MTE、连续小块和 task 粒度：`aiv-vector-layout-mte.md`；
- gate、SIMD、归约或 packed solve：`regbase-and-reduction.md`；
- MMAD 几何、共享操作数、对角 subtile 和 L1/L0 布局：`matmul-and-grouped.md`、
  `matmul-memory-and-layout.md`；
- 跨 task overlap、slot 和同步：`pipeline-and-synchronization.md`。

下表是独立局部块的条件候选；只把符合当前计算图和目标平台的轴纳入设计，不把局部块结构
强加给所有 Linear Attention 算子。缺少共享 RHS、局部 solve 或独立 task 时，相应轴不适用：

| 架构轴 | 满足前提时比较的候选 |
| --- | --- |
| task 粒度 | 一个局部块/task；2/4 个连续独立局部块/group |
| Vector owner | AIV0-only；AIV0/AIV1 分别拥有不重叠 group |
| 输入路径 | 每块 DMA；满足连续性时的 group DMA |
| 点积几何 | 相关结果分别 MMAD；共享 RHS 的 M 轴拼接；逐对角紧凑 subtile |
| 后处理 | 单块 solve；多块独立 lane 打包；结构零跨 task 生命周期复用 |
| 流水 | 串行 ready/free；容量和代际证明成立后的 ping/pong/overlap |

对每个候选记录执行乘加量、有效乘加量、GM/MTE 字节、task/flag 次数、AIC/AIV 工作量、片上峰值、
尾组和预期最长流水。存在性能目标时，对符合各自前提的“task 聚合 + 双 AIV”和“共享 RHS/对角 subtile”
方案记录比较依据及可实验范围；缺少对应结构时说明不适用原因，不要求凑齐候选。AIV0-only、每块两次
MMAD、单块 SIMD 后处理和逐 task 全量清零可以保留为正确性回退，但不能因实现简单而成为唯一 baseline。

批准的性能实验范围称为 `architecture_envelope`。其中的 `group_size`、owner 映射、MMAD 几何、
group DMA、结构零和 packed solve 可在不改变公开 ABI、reference、输出所有权和单 kernel 契约时
逐轴实验；每轮仍需记录设计增量、重新检查容量/同步并先过完整正确性。超出该 envelope 或改变
接口/数值顺序时才返回设计阶段重新批准。

候选矩阵还必须标注轴之间的依赖。只有组合后才能改变关键路径的轴应组成 `architecture_package`，
写明最小组成、合法的中间状态、降级边界和整体判伪条件。例如 task grouping 若依赖双 AIV owner、
私有 slot 和 group DMA，不能把只增加串行 group 循环的结果当作完整候选的性能结论。设计至少为主
吞吐包补齐以下实现表：目标版本 DSL 能力/layout 组合、owner 到输出半开区间、UB/L1 物理 slot、
ready/free 代际以及数值 DAG/cast 顺序。未知能力必须在 develop 中用最小编译探针关闭，不能直接
用串行回退替代。

主吞吐包失败时，分别记录“具体实现失败”和“包不可行”。单个 copy/layout API 报错、一次 owner
映射错误或 focused BF16 mismatch 只判当前写法失败；只有公式、容量、目标版本 API 或完整同步证明
排除了所有批准路径，才可判包不可行。性能目标未达且仍有未关闭的主吞吐包时，AIV0-only baseline
只是正确性锚点，不是性能设计完成状态。

## 开发回退与优化交接

首次实现优先落地主吞吐包。task breakdown 保留完整吞吐方案、正确性回退与预批准架构轴。
owner、slot、DMA、MMAD 和 postprocess 按最小组成实现；编码可以分轮，未达到合法中间状态
前不能据其中间性能否定完整包。回退前记录失败组件、精确 API/layout/dtype、原始错误、已查
目标源码/固定示例、合法替代路径及 owner/address/slot/flag/数值不变量与 focused case。
未排除全部合法路径的包保持 `unresolved`，向 optimize 交付台账、最近合法 checkpoint 与
未关闭依赖，不能从 AIV0-only baseline 重新猜测设计已明确的 owner、slot 与同步关系。

## 架构优先实验与预算

性能目标未达时，在局部微调前评估仍可行的 task 聚合 `group_size=2/4`、双 AIV、共享 RHS/
对角 subtile MMAD、group DMA、packed solve 与结构零生命周期复用，或给出静态排除证据。
独立轴可按 task grouping → owner → DMA/MMAD → postprocess 验证；强耦合轴用一个原子
proposal、统一 `axis_id`、整体 Expected 和判伪条件实验。只增加串行 group 循环不能否定
依赖双 AIV/group DMA 的包；SIMT thread 数调整不满足本条目的 SIMD 与 owner 并行要求。

仅在当前设计存在可行且强耦合的主吞吐包时，根据目标、风险和总迭代上限为完整包验证
预留预算；预算随设计确认，不对所有 Linear Attention 算子固定轮次比例。proposal 附能力、地址、
私有 slot、同步代际、数值 DAG 台账。达到迭代上限仍未达目标时，恢复并复测 best，最终性能状态为 `failed`；
逐包列出已排除路径、剩余合法路径和下一 revision 建议，不能用“已尝试双 AIV”等笼统名称
代替包级证据。

# 两步方案设计

只接受当前任务确认的数学公式、接口、PyTorch CPU 标杆和目标平台。PR 中的 64x64 chunk、
三 Stage、CG=2、Ascend950 容量和具体性能数值均是写法示例，不是默认配置。[^pr1134-design]

## 第一步：Stage 划分

列出每个公式节点的输入/输出 shape、dtype、有效域、Cube/Vector 类型；建立包括 head、chunk、
state 的完整依赖图。每条边记录生产者、消费者、首次与末次消费者、数据落点、可见条件和复用条件。
决定每个 Stage 的 owner、tail/空任务、可并行边和最少必要 Stage 数。

第一步至少产出：

- 完整计算依赖图和跨 Stage 数据表；
- Stage 类型、公式、前驱、可并行关系、输入来源、输出落点和任务数；
- task/batch/sequence/chunk/head/head-ratio 映射；
- L1/UB/L0/GM workspace 地址、大小、对齐、份数、首次写入、最后消费和峰值；
- 逐 head 或同 head 分片选择、各参与核读写范围、跨片归约与空片协议；
- R01-R21 的逐条静态证据。

所有节点必须恰好属于一个逻辑 Stage；Cube/Vector 类型、依赖边、容量和生命周期可复核；无法满足的
规则必须在此阶段解决或标记阻塞，不能带入代码生成。

## 第二步：具体详设

以已确认的划分为输入，计算真实的片上/GM 空间峰值和最大连续空闲区；定义每个槽的地址、对齐、
生产者、最后消费者、ready/free、复用和收尾。实例化目标版本已核对的 `tla.*` 接口、tiling、
精度观察点、测试及性能采集。

设计产物必须记录完整公式与 benchmark 三文件摘要、依赖图、Stage 表、跨 Stage 数据表、地址生命周期表、
同步边及伪代码、R01-R21 结论和证据位置。bench 的 definition/workload 格式由现有
DSL Skills 决定；运行全部 reference case 后按 develop 的实施、审查、全量测试、benchmark
与 profiling 流程推进。容量、依赖或同步假设失败时退回第一步；接口或 reference 变化时建立新 run
并重新审批，不能提高容差或改变 workload 隐藏问题。[^pr1134-design]

# Stage、任务映射与依赖

当前规则版本为 `V1`。新算子必须逐条检查 R01-R21；修改既有算子时至少重审受影响规则，并为其他
规则记录保持成立的依据。Cube 指矩阵乘及矩阵累加，Vector 指逐元素、归约、广播、layout、cast、
copy 和 mask。驻留区是供后续 Stage 使用的 L1/UB 数据；原规则中的 `CG` 指一个 AIC 与两个 AIV
一次协同处理的连续 value head 数。对于当前 Python DSL，AIC/AIV 是设计检查维度，必须核对目标
版本是否提供可观察的等价执行能力，不能由术语直接推断运行时实现。
[^pr1134-stage]

## R01 Stage 类型和 AIC/AIV 映射

Stage 划分和执行单元映射分两步完成。先依据数据依赖、计算类型、数据生命周期和精度观察点划分
Stage，再为每个 Stage 分配 AIC/AIV 任务。一个 AIC/AIV 可以按程序顺序承载多个 Stage；执行单元
相同不改变 Stage 边界，Stage 数量也不由 AIC/AIV 数量推导。只有在依赖、生命周期、同步和精度
观察点都能保持的条件下，才合并相邻 Stage。每个 Stage 只包含 Cube 或 Vector 一类操作。

`CG` 只沿 value-head 轴 `HV` 分组；AIC 按 `r=0..CG-1` 依次完成 Cube 任务。Vector 任务根据
每个 AIV 的 UB 峰值、数据依赖、可独立执行的 head 数和负载选择分工：

- **逐 head 分配**：每个 AIV 负责不同的完整 head，例如 `owner=r%2`。完整 head 工作集能放入
  单个 AIV 的 UB，且有足够独立 head 时，优先评估这种分工。
- **同 head 分片**：两个 AIV 分别负责同一 head 中不重叠的连续区域。独立 head 较少、Vector
  工作量较大或完整工作集超出单个 AIV 容量时，给出切分维度、对齐后的边界、有效区和每片工作集；
  涉及跨片归约或归一化时，补齐局部结果、partial 的合并方和以及后继计算依赖。

设计必须在 `DESIGN.md`（或仓库约定的等价设计文件）记录选择依据、每个 AIV 的读写范围和尾部
处理；共享输入、归约中间量和对齐开销计入容量。同一种分工以 `CG`、shape 和有效长度参数化；
跨 Stage 调整分工时，写清数据重分配、搬运和同步。逐 head 与同 head 分片的同步写法必须结合
`kernel-stage-sync.md` 以及目标 DSL 源码实例化。

## R02 Stage 输入依赖

Cube Stage 的输入来自算子输入或前序 Stage 已就绪的数据；Vector Stage 内允许后续计算依赖本 Stage
已产生结果。跨 Stage 按消费者实际需要的数据范围判断就绪：全部写入且对消费者可见后方可消费。
完成粒度可以是 head、分片或整组，由依赖决定，并与 ready 通知范围一致。

## R03 L1/UB 硬件边界

容量计算使用目标 SoC 对算子实际可用的 L1 和 UB 上限，并记录 SoC、参数来源和可用值。L1 服务
Cube 路径，UB 服务 Vector 路径，不混用容量预算。

## R04 Cube 到 Vector 的 UB 驻留

Cube 结果继续由 Vector 使用时，逐 head 模式写入 owner 的完整结果区，同 head 分片模式写入各 AIV
分片区。驻留份数按同时存活的 head/分片、流水深度和最后消费时机推导，并给出满缓冲等待和释放。
每份数据的 C++ Fixpipe（`PIPE_FIX`）或 DSL 等价写入完成后才发布 `cubeToVec` ready；聚合通知的
所有参与方必须按协议发送或等待 token。目标地址、完成 pipe 和通知粒度须按目标平台和真实 DSL API
实例化，找不到等价能力时标记待验证/阻塞。

## R05 Vector 中间结果驻留

Vector 结果继续由 Vector 使用时，各 AIV 保留自己的完整 head 或分片。UB 预算分别计入同时存活数据、
ping/pong 深度、共享输入和临时量。后继 Stage 分工相同时直接消费；分工变化时先完成已设计的数据交接。

## R06 Vector 到 Cube 的 GM 交接

逐 head 模式由 owner 将完整结果写入该 head 的 GM 区；同 head 分片由多个写者写互不重叠的 GM 区间，
合起来覆盖完整有效输入。每个实际写者在 C++ MTE3（`PIPE_MTE3`）或 DSL 等价 store 完成后发布
ready，Cube 消费者等待当前任务所有输入。聚合协议必须配齐 token：未负责当前 head 的 AIV
（non-owner）或有效长度为零（`validCount=0`）的空片可跳过数据计算和写回，但仍参加该任务已确认
的通知。GM slot 只能在最后一次读取完成后的 free 或等价消费完成事件到达后复用。

## R07 Cube 中间结果的 L1 驻留

Cube 结果继续由 Cube 使用时放入 L1 驻留区。按当前 TilingKey 和实际调度计算从首次写入到最后消费
期间的最大同时存活量，以大小、对齐和地址布局确定份数。记录生产、各次消费、释放和复用；覆盖前
证明旧数据最后消费完成。例如 `CG=4` 时，若最多同时保留两份且复用前均已消费，可以两槽流水；若
四个 head 全部生成后再消费，则需要四份。

# 容量、地址与生命周期

## R08 UB 峰值容量

UB 预算覆盖 Vector Stage 同时存活的全部输入、输出、中间量和跨 Stage 驻留数据，按峰值而非单张量计算。

## R09 L1 常驻和地址复用

L1 可常驻其他张量或在不同 Stage 保存不同数据；全部常驻数据按 R07 的最大同时存活量统一规划，
生命周期结束且同步条件满足后才能复用。

## R10 GM 搬运次数

同一路径每份数据只从 GM 搬入一次。同一原始数据同时被 Cube 和 Vector 使用时可分别搬到 L1 和 UB，
但必须记录两条路径和字节数。一次性流数据应评估是否关闭本次搬运对 L2 缓存的参与；跨核交接或
后续复用数据按真实复用关系选择缓存策略。缓存粒度必须由目标 SoC 和当前搬运 API 支持，并通过
同条件性能测试确认收益。

## R11 UB 地址复用

UB 区间允许在不同 Stage 保存不同数据，但只有旧数据最后一个消费者完成后才能复用。记录生产、最后
消费和覆盖点，并以原规则要求的 `MTE3_MTE2`、`MTE3_V` 或 `V_MTE2` 事件，或目标 DSL 的等价局部
事件/互斥能力保护；事件在 kernel 末尾排空并释放，不能只依赖跨核 ready。找不到等价事件时必须标记
阻塞，不得将跨核 ready 当作本核地址复用证明。

## R12 Vector 单任务装载与计算

每个 AIV 一次装入当前逻辑任务中负责部分所需的全部数据，并用一次原规则所称 VF 或目标 DSL
支持的等价完整 vector/tile 计算处理。UB 预算计入同时存在的输入、输出、中间结果、scale、gate
和 mask。原 C++“一次 VF”要求不能未经源码核对直接写成 Python API；即使 DSL 没有 VF 名称，也必须
保留单任务数据完整性、计算覆盖范围和中间量生命周期约束。

## R13 无依赖 Stage 的并行存活

无依赖 Cube/Vector Stage 按可能并行执行建模；并行期间仍存活的数据使用不重叠地址，不能按串行峰值复用。

## R14 容量冲突时的 GM 中转

L1/UB 容量不足或生命周期冲突时，可在同一个 kernel 的合法路径内把中间结果写回 GM 再读入。设计必须
说明冲突、增加的读写字节、同步点和预期代价；不能拆成多个 executor 规避单 kernel 契约。

## R15 最少 Stage

满足依赖、容量、生命周期、同步和精度观察点后，优先合并连续同类操作，减少 Stage 和 Stage 内任务数。
合并依据记录在依赖表；同一执行单元承载多个 Stage 时仍保留逻辑输入、输出和观察点。

## R16 Vector 结果复用

Vector 路径多次使用的结果优先保留在 UB，并在生命周期表记录生产、各次消费和释放。

## R17 连续地址与碎片检查

仍有效的 L1/UB 数据保持原地址。地址规划同时计算总剩余空间和最大连续空闲区，二者都满足下一次分配
才算容量合法。

## R18 跨 head 一致性

同类输入、输出和中间量在不同 head 执行相同操作并采用相同地址映射；差异只能来自有效 head、tail
或冻结接口语义。

## R19 Stage 划分决策顺序

先检查合并方案中同时存活的数据、L1/UB/L0 容量和生命周期。容量足够时按依赖和 Cube/Vector 边界
取最少 Stage；不足时先评估 R14 GM 中转；仅当额外拆分能解决中转仍无法解决的正确性、生命周期或
性能问题时增加 Stage。

# 分核与数值稳定性

## R20 沿 HV 分核和 CG 选择

沿 `HV` 按 `CG` 形成工作项，明确 `hv` 到其他 head 轴的映射、共享输入、归约和写冲突。chunk 间
无依赖时 `Nbase` 是本次调用的 chunk 总数；有依赖时 `Nbase` 是 sequence 总数；
`Nwork=Nbase*ceil(HV/CG)`，`blockDim=min(可用 AIC 核数,Nwork)`，grid-stride 分发且尾部只处理
有效 head。候选通常为 `CG=4/3/2/1`，`HV/HK=3` 时优先纳入 `CG=3`。CG 与 R01 分工共同评估，
先排除容量或依赖不成立的组合；比较执行波次、尾部负载、容量、搬运、AIC/AIV 关键路径和同步。
有设备时同条件实测；无设备时标为暂定，并按正确性、读写权限与地址、负载均衡、流水、搬运和同步选择。

## R21 数值稳定性检查

逐项分析高风险运算的计算 dtype、输入/中间值范围、溢出/下溢、`NaN/Inf`、cast 和 mask 时机。
代数变换提供前后值域、误差和溢出依据；无效位置在高风险运算前排除或替换为安全值。每项高风险
运算都有边界证明或保护方案后才能进入生成阶段。[^pr1134-stage]

## 设计文档写法与必备产物

公式必须声明张量维度、索引范围、累加 dtype、cast/归一化顺序、mask、边界行为和 state 更新时机。
每个 Stage 至少写出执行单元、操作顺序、输入/输出、任务映射、地址和容量、ready/free 或
event/flag、最后消费点、精度检查点以及与前后 Stage 的依赖。

开发期验证记录必须维护设计与验证的对应表：设计项、代码位置、用例、实际结果和结论。Stage、内存、
同步等设计调整应先更新设计并重新评审，再修改代码；候选实验进入验证记录，采用后才回写正式设计。

设计文档（通常为 `docs/design.md` 或当前 DSL 工作流约定的 `DESIGN.md`）必须包含：

1. R01-R21 逐条结论和证据位置；
2. 完整计算图、Stage 顺序、每个 Stage 的 Cube/Vector 类型和依赖；
3. `HV/HK/CG`、`Nbase/Nwork/blockDim`、每个 AIV 的完整 head 或分片范围、选择依据和 tail；
4. L1/UB/L0/GM workspace 分区、峰值容量、最大连续空闲区和生命周期表；
5. ready/free、event、flag、barrier、ping/pong 的配对和 workspace slot 安全复用条件；
6. 目标 DSL 中经源码核对的 ArchTag、DispatchPolicy、TileShape、矩阵乘、epilogue、scheduler、
   kernel、workspace 或其真实等价能力及合法分支；原 C++ 的 `BlockMmad`、`BlockEpilogue`、
   `BlockScheduler`、`Kernel` 等组件名找不到 DSL 等价物时标记阻塞；
7. 高风险数值运算的 dtype、值域、保护方式和精度阈值；
8. 每个实际模型场景的基线、目标、可达性分析和固定性能统计口径。

开发期验证记录（通常为 `docs/validation.md` 或当前 DSL 工作流约定的验证记录）必须维护设计与验证的
对应表：设计项、代码位置、用例、实际结果和结论。Stage、内存和同步等设计调整应先更新设计并重新
评审，再修改代码；候选实验只记录到验证记录，采用后再回写正式设计。

完成条件是 21 条规则均有结论，所用 DSL 组件在当前源码版本有依据，容量和同步配对可验证，数值稳定性
检查通过，设计与公开接口及 `reference.py` 一致；否则停留在设计阶段补齐证据。[^pr1134-stage]

# 任务、同步与地址生命周期

Stage 是逻辑计算边界，不等于一次 launch。一个 DSL kernel 内每条边至少写清：

```text
producer -> visible/ready -> consumer -> last use/free -> next producer
```

跨核数据 ready 与本核物理 slot 可覆盖是不同条件，不能用单个 ready 替代 free。[^pr1134-sync]

先确认 `(sequence, chunk, head)` 是否独立以及 head 是否共享输入。每个任务给出完整 head owner 或不重叠
的 `[begin,end)` 分片；分片边界符合真实 layout/搬运对齐。跨片归约增加 partial、合并方和独立依赖边。
`validCount=0` 时跳过数据计算和写入，但已进入协同任务的参与者仍按经验证协议完成本轮通知；padding
和无效区在后继读取前初始化。

每条边记录数据粒度、写者、所有读者、实际地址区间/slot、通知参与者、首次空闲状态、生产/消费次数和
末轮排空。所有读者和本核在途读写完成后才能 free。token 轮转前证明旧 token 已消费且 flag 不溢出；
ping/pong 只解决本地重叠，不自动建立跨核可见性。

同步伪代码必须覆盖初始化、首轮、稳态、tail、空任务、slot 回绕和最终 drain；对每个
`(task, head, slot, generation)` 列出 ready/free 发送与等待次数，证明无死锁、未初始化读取和重叠写入。
workspace offset、字节和复用按 tiling 计算，L1/UB 驻留与 GM 中转分别记录流量。

先用串行路径建立正确性/同步基线，再比较片上交接、GM 中转、逐 head/分片、pack overlap 和 RHS overlap。
候选需在同一接口、workload 和 SoC 下通过编译、全量精度、tail、重复 launch、同步检查、benchmark 和
profiling。C++ `CrossCore*Flag`、`HardEvent`、`PIPE_MTE3`、`Fixpipe` 只定义所需完成语义；实际
`tla.*` 调用与 pipe 范围必须由目标版本源码证明。[^pr1134-sync]

# 单位下三角矩阵逆

仅当数学契约确实需要 `A=(I+L)^(-1)` 且 `L` 严格下三角时使用，不能给所有 Linear Attention 算子
强加矩阵逆。chunk=64、两个 32x32 叶子的数学示例为：[^pr1134-inverse]

```text
I+L = [I+L00, 0; L10, I+L11]
XR = (I+L00)^(-1); XL = (I+L11)^(-1)
A  = [XR, 0; -XL @ L10 @ XR, XL]
```

它等价于：

```text
LeafRight = [XR, 0; 0, 0]
LeafLeft  = [0, 0; 0, XL]
Y = I + LeafLeft @ (-L)
A = LeafLeft + Y @ LeafRight
```

其他 chunk 大小和分块数必须重新推导，不能将 32/64 固化为公开接口。先按 Torch CPU reference 顺序确定
`L`、safe gate/clip、mask、cast、有效区和 FP32 累加；严格上三角和 padding 在 exp、逆、矩阵乘等
高风险运算前初始化为安全值。

叶子可逐行递推：

```text
X[0,0] = 1
X[i,:] = I[i,:] - sum(j=0..i-1, L[i,j] * X[j,:])
```

每行写后、后续行读取前证明可见性。两个叶子打包、Vector 递推和跨块 Cube 乘法只是实现候选，必须从
目标 Python DSL 源码验证 layout、归约、矩阵乘和交接能力。设计记录叶子、跨块积和最终 A 的 dtype、
地址、owner、最后消费者和同步。

精度至少观察：

1. 构造 `L` 的严格下三角、上三角零区、gate clip 后有限性；
2. 两个叶子 `X00/X11` 的单位对角和下三角；
3. 跨块中间结果 `Y`；
4. 最终 `A` 的单位对角、下三角、上三角零区和 tail；
5. 若算法包含 `KKT=k'@k'^T`，单独检查其 FP32 结果。

异常时分别排查叶子递推、块 layout、矩阵乘累加初始化、mask/cast 顺序、同步和地址复用。没有源码或
设备证据时不能宣称 C++ NZ、Fixpipe、`M_FIX` 等能力可由 Python DSL 直接调用。[^pr1134-inverse]

# 候选策略

- **Stage 候选**：在依赖、容量、生命周期、同步和精度观察点成立后取最少 Stage；连续同类操作优先合并。
- **任务候选**：联合比较 `CG`、逐 head 与同 head 分片；包含尾波、空片、跨片归约和 head-ratio 成本。
- **数据路径候选**：比较 UB/L1 驻留、kernel 内 GM 中转和重新搬运，记录字节、slot、可见性及缓存策略。
- **流水候选**：先串行，再评估 ping/pong、pack overlap、RHS overlap；没有完整 ready/free 代际图不实现。
- **布局候选**：查询 `dsl/` 与其他 `optimization/` 条目，比较 ND/NZ 或当前 DSL 支持的真实 layout、
  连续 load/store、MTE 粒度、结构零 tile 裁剪和 cache key。
- **矩阵逆候选**：仅在契约需要时比较逐行叶子递推、块乘组合、打包方式和中间精度；各检查点单独验收。
- **数值候选**：clip、mask、exp、累加和 cast 顺序必须与 reference 一致；代数变换先证明值域与误差。
- **知识组合**：本文不禁止读取其他设计、算子、DSL、调试、profiling 或优化文档；任何候选均应查询最相关
  条目并以当前源码与设备结果裁决。
- **架构优先**：性能目标未达且 `architecture_envelope` 仍有可行候选时，先实验 task 聚合、双 AIV、
  共享 RHS/对角 MMAD、group DMA 和结构零生命周期，再做线程数、guard 或地址表达式等局部微调。

# 代码模式

```text
operator_family: linear-attention
workflow: design -> approved generation -> correctness/review -> benchmark/profile -> optimize

stage:
  id: S<n>
  kind: Cube | Vector
  formula: <shape, dtype, valid domain, accumulation and cast order>
  owner: <task/head or aligned [begin,end) shard>
  input: <source and ready condition>
  output: <location, half-open address range and layout>
  lifetime: <producer -> consumers -> last use -> free/reuse>
  synchronization: <verified tla.* API, participants, token count and drain>
  evidence: <source path, focused cases and profile metric>

optimization_axis: <one falsifiable change>
hypothesis: <expected total-latency effect and affected profile counters>
acceptance: full correctness + final review + anti-hack + benchmark + profiling
```

# 约束

- 公开接口、数学语义、reference、required workload 和精度策略冻结后才生成 candidate；变化需新 run 和重新批准。
- 一个 solution 只声明一个自包含 `@tla.kernel`；wrapper 不做计算，不以多个 executor 或 GM 临时张量串联 Stage。
- R01-R21 每条都要有满足、不适用或待验证结论及证据，尤其不能省略 R01-R07 的任务映射和依赖。
- C++ 名称不是 DSL API。目标版本源码找不到等价能力时保留阻塞，不编造实现。
- 空任务、tail、non-owner、slot 回绕和退出排空必须保持通知平衡与地址安全。
- 总剩余容量和最大连续空闲区同时满足才算可分配；跨 Stage/跨任务复用前证明最后消费者完成。
- 其他知识条目与本文并行使用；冲突时按固定接口、目标版本源码、实际测试和 profiling 证据裁决。

# 失败表现

- 先按执行单元数量划 Stage，遗漏真实依赖、生命周期或精度边界。
- 只写“逐 head/分片”，未给每个 owner 的有效范围、对齐、归约、空片和 tail。
- producer 发布 ready 时数据尚未可见，或用 ready 代替地址复用所需的 free。
- 仅按串行阶段估算容量，忽略并行 Stage、ping/pong、共享输入和临时量的同时存活。
- non-owner/空片跳过协议导致 token 不平衡、死锁、旧数据或 flag 溢出。
- 把 C++ `HardEvent`、`CrossCore*Flag`、Fixpipe/NZ 等名称写成不存在的 Python DSL API。
- 矩阵逆的上三角/padding 未在高风险运算前置零，或叶子递推、块 layout、cast 顺序与 reference 不一致。
- 优化只改善局部 MTE/Vector/Cube 指标，却使总延迟、边界正确性或其他 workload 回退。
- 只读取本文而拒绝相关 API、算子、debug、profiler 或其他 optimization 知识，导致候选缺少版本依据。
- 当前计算结构需要专项知识，却在 family 查询只返回本文后跳过相应 mixed、AIV/MTE、RegBase 或 MatMul 条目。
- 存在可并行局部块或共享操作数，却未经比较就冻结 AIV0-only、两个独立 MMAD 或逐 task 全量清零。

# 验证方法

1. **设计完整性**：公式、计算图、Stage 表、R01-R21、地址/容量/生命周期、同步伪代码、数值边界和测试计划齐全。
2. **源码依据**：每个 `tla.*` API、layout、copy、同步和编译期行为可定位到目标版本源码；未知项保持待验证。
3. **reference 门禁**：所有非 performance required cases 实际运行并冻结 reference/definition/workload SHA-256。
4. **结构门禁**：单个自包含 kernel、无 host/Torch candidate 计算、无多 kernel dispatch，anti-hack 明确 passed。
5. **正确性门禁**：覆盖最小/典型/非对齐、fixed/varlen、tail、padding、head ratio、state、非法输入、
   多任务 slot 复用和重复 launch；矩阵逆按各中间检查点验收。
6. **同步门禁**：逐边核对 producer/consumer、ready/free 数量、首次状态、空片、回绕和最终 drain；检查越界、
   未初始化、竞争、死锁和旧数据覆盖。
7. **性能门禁**：同环境采集总延迟及 Cube/Vector/MTE/scalar/等待/尾波指标；只接受超过测量波动且无回归的候选。
8. **知识门禁**：查询并记录采用的其他知识条目；只有当前 kernel SHA 绑定的 correctness/profiling 证据支持的结论
   才写入 learned。[^pr1134-design][^pr1134-stage][^pr1134-sync][^pr1134-inverse]

[^pr1134-design]: PR 1134 head `4f4e7608` 中的两步方案设计，定义 Stage 划分、具体详设和状态门禁。
[^pr1134-stage]: PR 1134 head `4f4e7608` 中的完整 R01-R21；其 Stage、任务映射与依赖源自固定提交 `f53617a6`，本条目未删减规则语义。
[^pr1134-sync]: PR 1134 head `4f4e7608` 中的任务、ready/free、slot 与地址生命周期设计。
[^pr1134-inverse]: PR 1134 head `4f4e7608` 中的单位下三角块化求逆数学模式及 DSL 验证边界。
