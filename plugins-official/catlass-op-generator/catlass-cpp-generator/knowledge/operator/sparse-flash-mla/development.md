---
type: operator
title: SparseFlashMLA 分阶段开发与工程集成
description: 计算阶段、动态调度、组件约束、公开主核与 metadata 接线、SDK 工具准备和正式部署。
tags:
- catlass-cpp
- sparse-flash-mla
- stage
- sync
- slot
- topk
- implementation
- swa
- engineering
- arch22
- metadata
- readability
- workflow
- performance
status: draft
operator_families:
- sparse-flash-mla
architectures:
- atlas-a2-a3
- ascend910b
generated:
  by: swa-general-final
verified: []
sources:
- id: stages-computation-contract
  resource: knowledge/operator/sparse-flash-mla/computation.md
  title: 各Stage的计算契约
  kind: contract
- id: engineering-swa-domain-contract
  resource: knowledge/operator/sparse-flash-mla/interface.md#swa-contract
  title: arch22 范围、golden 语义和新增实现契约
  kind: contract
- id: engineering-swa-runtime-schedule
  resource: knowledge/operator/sparse-flash-mla/metadata.md
  title: 任务区间、生产消费与公开 metadata 表
  kind: contract
- id: engineering-swa-pipeline-knowledge
  resource: knowledge/operator/sparse-flash-mla/performance.md#strategy
  title: 一般 SWA 的参数寻址、分支与持久流水
  kind: repository
- id: engineering-swa-diagnostic-contract
  resource: knowledge/operator/sparse-flash-mla/validation.md#suite
  title: 诊断 ABI、完整输出及域内验证
  kind: contract
- id: implementation-swa-pipeline
  resource: knowledge/operator/sparse-flash-mla/pipeline.md#structure
  title: SWA QK/PV 与片上流水结构
  kind: contract
- id: implementation-swa-sync
  resource: knowledge/operator/sparse-flash-mla/pipeline.md#protocol
  title: SWA 跨任务握手协议
  kind: contract
---

<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

本页按主题合并知识；下列目录进入各主题的起点，各通用章节下保留完整细节。
通过知识工具 get 时使用文件路径，不把章节锚点传给 get。

- [计算阶段与开发分支](#stages)
- [动态工程与正式部署](#engineering)
- [实现起点与工作量检查](#implementation)
- [公共 CANN 构建工具准备](#sdk-build)

# 接口与概念

<a id="stages"></a>

<a id="stages--接口与概念"></a>

## 计算阶段与开发分支

适用于设计路由已经识别出的“多路 KV 选择、共同 softmax、K=V、可选 sink”计算。先读[计算契约](computation.md)、[完整接口](interface.md#api)及[metadata 协议](metadata.md)，同时读取[用例与验收](validation.md#coverage)。这些知识可直接用于实现，不需要获取原算子仓库。依赖只有目标 CANN、CATLASS 和执行调用者标杆所需的测试运行时。

<a id="engineering"></a>

<a id="engineering--接口与概念"></a>

## 动态工程与正式部署

一般 SWA 是一个由运行时参数驱动的成对算子工程；用户指定的 JSON 和 ID 范围只是性能和回归子集。
完整功能域由目标平台的公开契约、arch22 实际限制和用户已授权的 golden 语义确定。
不能用一个有限测试清单、某个优化 tile 或生成者自选表结构定义支持域。
本页为源码静态审查后的纯知识输入，不含完整算子实现；`verified: []` 不表示设备验证通过。

必须分别记录三个事实层级：**入口是否接受、主核是否实际正确消费、此次生成是否按 golden 修正**。
某字段有公开槽位、host 未拒绝、metadata 能读取，均不能单独证明主核数值支持。
arch35/A5 的任意 head 或窗口能力不得并入本页的 arch22 范围。[^engineering-swa-domain-contract]

<a id="engineering--arch22-支持包络与源码差异"></a>
## arch22 支持包络与源码差异

| 维度 | 当前源码静态证据 | 新工程的处理原则 |
|---|---|---|
| 模式 | 有 ori KV，无 cmp KV/压缩索引及 ori sparse 时选择 SWA | 完整保留参数槽；压缩或原始稀疏计算不伪装为 SWA |
| dtype / D / head | 非 A5 的 `CheckFeatureShape` 要求 D512、N2=1、N1/G 为 1..128 的 2 次幂；Q/KV 同 FP16 或 BF16 | 这些是源限制，不是测试子集常量；不能扩写为任意 head/D 或 FP32 主输入 |
| mask / window / ratio | arch22 main 要求 ori mask4、cmp mask3、左127、右0；SWA cmp_ratio=1 | 128-token 窗口有源依据；无压缩的缺省属性适配与非法显式值分开 |
| 布局 | `GetKvLayout` 接受 BSND+BSND、TND+TND、BSND+PA_BBND、TND+PA_BBND | 四种配对都属于完整域；不能把 PA 当未实现的可选优化 |
| batch / 长度 | main 要求 B>0，TND 总 T1>0；调度按运行时 B 和每 batch 长度循环 | 没有来自 1024-word 表的 112/116 等小 batch 限制；整数溢出、可寻址容量与设备内存仍需校验，不能宣称数学意义的无限 batch |
| sink | `GetSinks` 拒绝 nullptr，`CheckSingleParaSinks` 要求非空 FP32[N1] | 必须支持实际非零 sink；nullable sink 是额外数学/API 扩展，不能称原 arch22 已支持 |
| LSE | infer shape 给 FP32：BSND `[B,N2,S1,G]`，TND `[N2,T1,G]`，关闭为 `[0]` | 两个公开输出及开关必须完整接线；golden 的轴序转换属于测试适配 |
| PA | 原始页表 INT32[B,max_pages]、seqused_ori_kv 必需；页长为16倍数且在[16,1024] | 非恒等、非单调物理页号、跨页窗口、首尾部分页均用真实页表寻址 |
| 非连续 KV | host 获取 stride0；Cube 的 BSND batch 和 PA page 路径消费 stride0；TND 路径仍用连续 token 地址 | 只把已贯通的 stride 能力称为源支持；本次要求的其他合法原生 stride 必须真正贯通或显式整理并计时 |
| TND effective length | metadata 优先 seqused，缺省取 cu 差值；arch22 main 的 `GetActualSeqLenQ/KV` 直接取 cu 差值 | 新工程按 golden 支持 `seqused < cu差值` 时属于明确修正；必须保留 storage base 与 effective length 两套值 |
| 空工作与 padding | 源中有 invalid-prefix 清 O/LSE 和零任务排空分支；常规 BSND 初始化仅清 O 且受 needInit 控制 | 不能由此宣称所有 padding/整批零工作已正确；新工程为所有应零的 O/LSE 元素明确指定写者并实测 |

长度、零行和流水必须分别核对调度、Cube 搬运和 Vector 输出的实际行为；三者共同确定实现能力。[^engineering-swa-domain-contract]

metadata 可使用按运行时 batch 大小分配的规划临时区，公开输出只保存每 core 游标，
所以固定表长不构成小 batch 上限。metadata 独立入口允许的边界也不总等于 main：例如其标量
batch 校验允许0，而 main 要求正 batch；应分别记录入口契约与成对可调用域。[^engineering-swa-runtime-schedule]

还要避免把新工程自行缩窄的校验冒充原实现边界：

- arch22 对非空 topk_length 明确拒绝，但保留空 Tensor/None 的区别；不能把所有可选槽都按“非空即错”处理。
- SWA 中某些 cmp 附属槽不参与数学；例如 main 的合法 INT32[B] cmp_residual_kv 可通过校验，SWA kernel 不消费它；cmp block table 的校验也只在压缩模式执行。新工程应逐槽记录 source 行为、忽略原因和校验，不因性能子集都传 None 就缩小原有可调用域。
- `topk_value_mode=1` 是公开兼容值；已归纳的入口行为中，对该值的显式特性检查位于 A5 分支，不能把它描述成 arch22 同一检查已执行，也不能仅凭漏检宣布其他值有语义支持。
- `CheckContiguous` 的注释声称检查页内连续，但实际只比较选定的一个 stride 轴；这不是任意其他轴合法的证明。普通布局的 stride0 检查与 Cube 可读取 batch stride 的能力也须分开。
- 源 `InitOutputSingleCore` 仅清 O；已审查 arch22 SWA 路径中 needInit 默认 false，未见在该路径赋真。LSE 有效行和显式无效前缀有写出，不足以证明剩余 padding 也写出。应记录为静态覆盖疑点，不能把缺写结果当作 golden 语义。

<a id="implementation"></a>

<a id="implementation--接口与概念"></a>

## 实现起点与工作量检查

本页只补普通 SWA 的特化决策。interface、reference、design、implementation、validation 的阶段门禁和 CATLASS 通用组件用法仍由工作流知识负责。已研究的硬件/数据域为 910B3、D512、N2=1、127/0 窗口、FP16/BF16；本轮验证用例由调用者提供。下表是容量与性能推导样本，不是按 case id 派发的代码。

# 用法

<a id="stages--用法"></a>

## 计算阶段与开发分支

普通 SWA 的 design/implementation/validation 先完整读取[逐事件跨任务协议](pipeline.md#protocol)、[混合核结构与局部流水](pipeline.md#structure)、[源码和设备验收](validation.md#pipeline-validation)及[通用工程交付](development.md#engineering)。最终实现必须能把实际调用、slot、ready/free、UB复用、workspace 所有权和尾部排空逐一映射到这些知识；通用schema PASS或四个串行kernel的数值PASS不代表流水已完成。普通无索引SWA跳过稀疏V0，PA可选直接L1或有界pack；不能把metadata当V0。
摘要或schema PASS不能替代正文阅读、源码语义审查和设备结果。

开发顺序和恢复路径统一由本仓库[五阶段流程](workflow.md#routing--五阶段与产物)及
`docs/workflow.json` 控制。
计算 Stage S0–S6 是 kernel 数据流，不是 workflow 的阶段；04 阶段逐段构建运行，05 阶段全量验收。
将各分支实现位置、实际命令、用例 ID、结果和待办写入 `docs/validation.md`；未运行写 NOT_RUN，
只编译成功写 BUILT。新增 tile/slot/分支使受影响旧记录失效。

<a id="stages--算子算法"></a>
## 算子算法

所有合法输入都必须落到一个可说明的路径；不是要求把笛卡尔积全部实例化。将逻辑策略和物理kernel分开，在`docs/design.md`列出 `选择条件→ModePolicy/LayoutPolicy/Schedule→实现符号→用例ID`。

| 轴 | 分支与职责 |
|---|---|
| 模式 | SWA：ori窗口；CFA/HCA：ori窗口+cmp连续可见前缀，ratio128；SCFA/CSA：ori窗口+cmp给定索引，ratio4，1≤K≤8192。三者共享S2–S5，只替换选择/装载策略 |
| Q存储 | BSND按batch容量定位；TND按cu_q[b]定位，有效长度与物理容量分开 |
| KV存储 | BSND连续；TND按各自cu定位；PA_BBND按各自页表定位。Q仅BSND/TND，形成4种目标组合 |
| head | G=1专门检查双AIV工作划分和零行子核；G>1按有效M补齐，G可为2/4/8/16/32/64/128 |
| shape | 单/多S2块、K%512尾、M尾、跨页、ragged batch、有效任务少于核数；用运行时有效长度处理，不做“只支持整块”的隐含假设 |
| 输出 | LSE开启/关闭；完整任务直接归一化写回，split任务写partial并由S6归并 |
| 调度 | FA-only为正确性基线；任务不足且KV长时评估split-S2；实现split后独立实例化/选择，不能因FD字段存在就默认启用 |

实现可分 `ModePolicy`、`KvAccessor`、`TaskScheduler`、`OnlineSoftmax`、QK/PV `BlockMmad` 和主 `Kernel`。host只做属性/长度/任务表/资源规划；真实选择、gather、QK、softmax、PV在device。不改成带独立RoPE的普通MLA接口，也不强迫非分页输入改公开布局。

<a id="engineering--用法"></a>

## 动态工程与正式部署

接口阶段先形成 `docs/api.md` 中逐字段的支持矩阵，并把范围拆成：

1. **完整 SWA 功能域**：目标910B的全部已确认合法输入类别，以及用户要求的 golden 修正能力；逐项列来源、布局、长度、stride、页表、sink/LSE、可选槽和边界行为。
2. **性能子集**：调用者指定 JSON 和 ID 范围及其原参数、哈希、逐例阈值和采样口径。ID只用于报告。
3. **正确性探针**：从完整域推导的边界、动态输入、异常和未见 shape 集合；不是用探针数量定义支持范围。

完整域的一个合法参数组合落不到任何已实现分支，即为功能缺口。可以报告已完成子集，但不能把
“完整接口声明+选定用例通过”记作完成一般 SWA。源实现疑点与修正能力在接口阶段按现有用户授权裁决，
不通过偷偷改变 golden、输入或验收范围回避。参考、设计和验证继续使用同一份冻结契约。[^engineering-swa-domain-contract]

工程结构先写入真实 `docs/design.md`，再实现；不把本知识页、模板原文或 read receipt 当作工程详设。
design 的公式、任务、资源、分支、同步和异常出口须能一一映射到最终源文件和符号。

<a id="implementation--用法"></a>

## 实现起点与工作量检查

在普通SWA设计和首次生成前，完整读取[任务流水协议](pipeline.md#protocol)、[结构与片上流水](pipeline.md#structure)、[实现验收](validation.md#pipeline-validation)。下文的小块地址算例用于解释SDK调用范围，不是允许最终停留在四个串行launch；按所选M/N/Dv更新公式，并落实有界slot和跨任务握手。

PV尾块在设计阶段就按[有效K、物理对齐和行距参数映射](pipeline.md#structure--pv-tail)落表：窗口并集宽度进入逻辑归约，物理N只提供P行距；当前组件若支持实际K，优先走实际K链，避免人为对齐K后再引入零步长GM补零。组件需要对齐K时按同节显式初始化规则实现，不能按单个测试K特化。

在写主 kernel 之前，把每个待支持的布局/长度类映射为一组 query-head 矩阵任务，算出任务数、KV 读取次数、workspace 写入次数和 GM 总字节量。先排除明显无法满足冻结逐例时延上限的方案，再做 CATLASS 资源表。`validate_design.py` 通过只表示设计文档结构完整，不能证明带宽、launch 次数或正式算子性能可达。

共享 KV 的行映射为 `r=t*H+h`，其中 `H=N1/N2`。一个完整 head 组跨 `T` 个相邻 query 时，`M=T*H`，共同 KV 并集最多 `128+T-1` 列，每行仍使用自己的因果 mask。首选验证 `M<=128` 的跨 query 方案；`H=128` 时 `T=1`。若切 head，跨 query 的 Q 物理行可能不再是单一 stride，须明确分段搬运或打包，不能把它伪装成连续 RowMajor。

| 工作类 | 起始候选 | 比逐 query/head 小块少了什么 |
|---|---|---|
| H1 长 prefill | T128、M128、并集至多255列 | 避免约128次重复读取相邻窗口 |
| H4/H8 长 prefill | T32/16、M128 | 相邻 query 共享一个 KV 并集 |
| H64/H128 长 prefill | T2/1、M128 | 一次 QK/PV 覆盖完整 head 组，避免把每个 query 再切成多个 M16 任务 |
| 短 decode | T1，M 为实际 head 数 | 不为不存在的多任务并发增加同步和中转 |

上述 M/N 只是待验证候选；具体 N、Cube/Vector 分工、双槽和地址限制由实际 CATLASS 版本及 Stage 资源表决定。metadata 的 256 行游标块不强制主核使用 M256，也不意味着一个 query 必须拆成多个 M16 微任务。每个有效 `(b,t,h)` 必须恰好由一个任务写入；无效前缀和 padding 的零输出另有明确写者。

`T`、`M`、`N` 必须在同一张资源表中确定。完整 head 组取 `M=T*H<=128` 时，物理 `N` 至少覆盖 `min(Lkv,128+T-1)` 列，再按当前 BlockMmad 支持的粒度补齐；每行仍只看自身的至多128列窗口。例如 H1/T128 需要至多255列、可考虑 N256；H64/T2 需要至多129列，不能继续用 N128。score/P 的行步幅均为**物理 N**，不是窗口宽128；`score[row*N+col]`、`P[row*N+col]` 和各 AIV 分段起点必须一致。设计文档里从 M16/N128 改到 M128/N256 后，逐处重算任务解码、L1/L0/UB、GM slot、尾块 mask、DMA stride 和事件槽；结构校验通过不代表这些关系一致。

任务的逻辑 `M/N` 与一次 CATLASS `BlockMmad` 调用的块 `m/n` 是两层尺寸。已分析的 ping-pong 实现将 `operator()` 作为 block-scoped，调用者须保证 `actualShape.m()<=L1TileShape::M` 且 `actualShape.n()<=L1TileShape::N`；`K` 可由组件内部按 L1 K 分块。`BasicMatmul` 先用 scheduler 切 M/N，再把每个 `actualBlockShape` 交给 `BlockMmad`，核末排空相关 pipe。尤其要保留原矩阵的 layout/leading dimension，不能把尾块尺寸当作实际行距。例如逻辑 M64/N2 配 L1 `<16,128,128>` 时，直接调用 `(64,2,512)` 会越过 M16 的 L1A 块，编译通过也不能证明设备可运行。若首个 QK 在 stream 同步超时，先查调用界限及事件排空，再查 softmax/PV；用单 core 重跑不能修复越界块。资源 owner 构造时会初始化 L1A/L1B、L0A/L0B 和 FIX/M 事件，析构时须按实际版本排空；同一硬件事件池中同时创建两个不协调的 owner 会使握手相互污染。目标 checkout 必须重新核对签名、资源约束、调度和排空行为。

普通 SWA 的外层分块要保持真实物理步幅：QK 对 `m0` 每16行、`n0` 每128个 KV 列调用 `shape=(min(16,M-m0),min(128,Nvalid-n0),512)`；Q 偏移 `m0*512`，连续/PA 的 K 偏移 `n0*kv_token_stride`，score 偏移 `m0*score_stride+n0`。PV 对 `m0` 每16行、`d0` 每128个输出维调用 `shape=(min(16,M-m0),min(128,512-d0),Nvalid)`；P 偏移 `m0*p_stride`，V 偏移 `d0`，delta 偏移 `m0*512+d0`。`score_stride`/`p_stride` 仍是物理 N（如256），不能为了单次 tile 改成128。每个调用传完整输入矩阵的 layout/leading dimension，并在核末按当前组件要求排空；尾块只缩小 `actualShape`，不缩小目标行距。这是可审查的地址公式，具体类型签名仍以当前 CATLASS 头文件为准。

以 D512、M128、N256、FP16/BF16 为资源上界样本：Q 为128 KiB，KV 为256 KiB，P 为64 KiB，三者同时保存在 L1 已达448 KiB；512 KiB L1 下不能再假设第二份256 KiB KV 常驻。QK 的 FP32 L0C 为 `M*N*4=128 KiB`，PV 若一次产出 D512 则为256 KiB，超过128 KiB L0C，应验证输出 D 轴分块（例如 Dv256）及两半 Fixpipe/GM 写回。每 AIV 的 UB 需按实际同时存活的 score、归约 scratch、P 和行状态核算；这些数值是设计上界而非当前实现已编译的配置。

<a id="sdk-build"></a>

<a id="sdk-build--用法"></a>

## 公共 CANN 构建工具准备

本节描述独立工程暂存目标 CANN 公共 CMake 工具的方法；需要自动化时在生成工程内实现，
知识库不附带暂存脚本。输入为当前 CANN 根和新的空输出目录，产物为解引用后的 `cmake/`
与完成后才发布的 `stage-manifest.json`，不生成 kernel、host、算子注册代码。

# 代码模式

<a id="stages--代码模式"></a>

## 计算阶段与开发分支

<a id="stages--数据路径与存储层级"></a>
## 数据路径与存储层级

M为一个任务中的query/head行数，N为KV tile容量，D=512。每个tile携带 `(task_id, row_start, row_count, segment, logical_offset, valid_columns, slot, generation, first, last)`；使用这个描述符驱动所有消费者，不能让不同Stage分别猜测“末块”。

| Stage | owner及运算 | 输出/寿命 | 逐段验证 |
|---|---|---|---|
| S0 调度 | 独立metadata入口生成FA/FD表；主kernel解码(b,kv_head,Mblock,S2block) | 任务描述直至所有消费者完成 | 展开任务表，与有效query/head集合比较；0/1/少于核数/多于核数/ragged任务 |
| S1 选择和装载(V0/MTE) | 连续ori窗口，连续cmp前缀或索引cmp；logical ID→物理地址→KV tile | KV含低精度K/V及有效列信息，QK和PV都读完才释放 | dump逻辑ID、槽号、有效mask和KV；先精确比ID，再比KV字节；乱序页/未来ID/-1/重复ID |
| S2 QK(C1) | Q[M,D] × K[N,D]ᵀ，Catlass BlockMmad、FP32累加；有效score乘scale | score[M,N]供S3 | 对同一gather tile作QK标杆；有效区、D方向多次累加、M/N物理尾 |
| S3 softmax(V1) | mask→rowmax→exp→sum；更新共同m/l，生成alpha及cast后的P | P[M,N]至PV；alpha与对应tile绑定至S5，m/l为任务状态 | 比m/l/alpha及P；单块→两块→17块；第二路不能重置m/l或重复sink |
| S4 PV(C2) | cast(P)[M,N] × V[N,D]，Catlass BlockMmad、FP32累加 | delta[M,D]供S5；此后才可释放KV/P | 对已cast P比较PV；不能用FP32 P的结果代替当前数值路径 |
| S5 累加/输出(V2) | u←alpha*u+delta；末块O=cast(u/l)，LSE=m+log(l+epsilon) | u为任务私有状态；输出只写有效行；split时写未归一化partial | 每tile u以及最终O/LSE；输出axis、dtype、关闭LSE的空tensor；新task状态重置 |
| S6 可选FD | 合并同一行各split的(m,l,u)，最终归一化 | partial buffer直到FD读完 | 2/3/多split、空片、非整块边界、sink恰好一次，与FA-only对照 |

SWA/CFA没有稀疏gather，但S1的逻辑接口仍存在。CFA可先用独立ori/cmp tiles；若优化成混合tile，必须分别携带两组地址/有效mask且维护同一状态，重新核验golden块边界与P舍入。按query共享稀疏行的CSA可令Mbase=G；连续路径可从Mbase256的策略起步，但M物理tile须按资源切片，不能据此直接分配256×512的全部UB中间量。

<a id="stages--4-任意topk与地址规则"></a>
## 4. 任意TopK与地址规则

对batch内query t，`p=Lo-Lq+t`；压缩因果可见量 `C=max(0,min(Lc,(p+1)//r))`。按标杆从前`min(K,C)`个索引槽扫描，-1终止，未来ID跳过，重复ID保留。原槽号s、过滤后序号j、物理token地址是3个坐标。

```text
index_row_base = (logical_query * Nkv + kv_head) * K
scan_limit = min(K, C)
for original_slot in [0, scan_limit):
    id = indices[index_row_base + original_slot]
    if id == -1: break
    if 0 <= id < C: append id to selected list  # 顺序不变，重复不去重
for start in range(0, len(selected), N):
    count = min(N, len(selected)-start)
    # 保留跨加载轮的scan cursor；一个被跳过ID不能消耗有效列
```

这是逻辑算法，不要求物化整条selected list。S1可以分段扫描并压紧，只缓存一个tile。若选择保留无效槽而不压紧，必须证明与标杆的tile边界/P舍入兼容并通过数值验收。容量K用于下一行步长，有效count用于这一片计算，二者不能互换。K1537全可见时N512切成512/512/512/1；读取成对索引时第二个槽独立判界。

KV元素偏移（乘dtype字节数后为字节地址）：BSND `((b*S_cap+s)*Nkv+n)*D+d`；TND `((cu_kv[b]+s)*Nkv+n)*D+d`；PA `((table[b,s//page]*page+s%page)*Nkv+n)*D+d`。ori/cmp各用自己的table/page/cu。Q的head n1=n*G+g，M行m对应t=m//G、g=m%G。PA表只映射物理存储，不改变可见ID。

tail统一用`count=max(0,min(N, logical_end-start))`；logical_end来自当前段的真实长度，不能用某核最后一个loop推导全局尾。物理填充KV可为0，但score填充列必须在max/exp前mask=-inf，P相应为0。空tile不能求(-inf)-(-inf)，不读未初始化缓存，不推进有效状态。

<a id="stages--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

先列地址表再写事件。每项记录 `空间、物理shape/stride、字节数、core/subblock/task/slot偏移、writer、reader、最后reader、可复用时刻`。workspace每个段独立对齐；写明总量公式及host分配验证。

| 对象 | 每slot候选量级 | 释放条件 |
|---|---|---|
| gathered KV | N*D*2 bytes（低精度） | C1对K和C2对V的最后读取均完成；QK完成不够 |
| score FP32 | Mphys*Nphys*4 | S3读取完成 |
| P低精度 | Mphys*Nphys*2 | S4读取完成 |
| delta FP32 | Mphys*D*4 | S5读取完成 |
| alpha/m/l | 按物理行对齐的FP32 | alpha等到对应S5；m/l持续到任务结束 |
| u FP32 | Mphys*D*4 | 最终写回或partial写回完成 |

上述是逻辑量级，不能全塞UB。N512、D512时KV一槽已有512KiB；根据L1/UB/L0容量选择GM中转、D子块、M切片和实际槽数。分别计算L1双缓冲、L0A/B/C、每个AIV私有UB峰值，包含API临时区、padding与对齐。实际容量从当前SoC/CANN查询；不得把GM表容量当可用核数。

每条生产消费边都需要两个不同含义的条件：`ready(slot,generation)`证明本轮数据可读；`free(slot,generation)`证明上一轮所有读取完成。记录对应PIPE及event/flag编号、生产/消费参与者和Set/Wait次数。AIC:AIV配比不能靠“一个Set对应一个Wait”猜测；按实际同步模式确认广播/计数行为。G1的无有效行子核也必须按协议参加，或从两端共同移除，不能只跳过一侧。

首次数值实现按本模式已评审的完整预取、CV重叠和跨任务流水设计生成；不能把源码调用顺序视为DMA完成顺序。首版应落实持续资源owner/typed views、跨任务两槽/late free、每行一次FP32倒数，或给出有代码与实测证据的等效替代。单任务和Stage dump用于分层检查同一优化调度，最终交付不降为无重叠串行版本。实现前画出预热/稳态/排空表，后续调整资源/调度时更新。例如其他设计选择3个描述符轮转时，可在迭代i处理S1/S2(i)、S3/S4(i-1)、S5(i-2)，再加2轮排空；实际flag和buffer释放仍须满足依赖，不能把抽象表原样当作硬件安全代码。

跨任务仍在途的描述符保留旧task_id与长度，状态不能被新task初始化覆盖。短任务0/1/2块、多于槽数、连续两个task和最后空batch都检查Set/Wait平衡。有限槽环按 `FREE→WRITING→READY→READING→FREE` 模拟至少两次回绕；加入消费者延迟，验证不会提前覆盖。

<a id="engineering--代码模式"></a>

## 动态工程与正式部署

<a id="engineering--算子算法"></a>
## 算子算法

每 batch 的有效长度记为 `Lq[b], Lk[b]`，存储前缀记为 `qBase[b], kvBase[b]`。
两类变量含义不同。对于有效 query 的本地位置 t：

\[
c=L_k[b]-L_q[b]+t,\quad
J_t=[\max(0,c-127),\min(L_k[b],c+1)).
\]

`t < Lq-Lk` 的无效前缀和 `t >= Lq` 的存储 padding 分别有零输出路径；不能把空行当作仅有 sink 的有效行。
有效行遵守 QK FP32、scale、逐行 mask/max/exp、cast 前 FP32 分母、P 转输入 dtype、PV FP32、
FP32 归一化和最终 O cast；sink 只影响共同分母，LSE 来自同一 max/sum。
golden 的具体流式分片和舍入顺序以冻结数值契约为准，工程分层不得改变它。[^engineering-swa-domain-contract]

<a id="engineering--职责模块与可读命名"></a>
### 职责模块与可读命名

以下是责任边界，不要求逐文件照抄名称。合并小模块可以，混合职责后难以独立审查不可以。

| 模块 | 唯一主要责任 | 不能藏在其中的工作 |
|---|---|---|
| PublicContract / ParameterValidator | 两公开入口全参数、shape/dtype/layout、可选槽、长度和溢出校验 | case_id 白名单、attention 计算、静默改变输入 |
| SequenceDescriptor / LayoutAddress | storage/effective length、native element stride、PA 逻辑页映射 | 改 mask、改值域、改变数值路径 |
| MetadataPlanner / TaskCursor | 覆盖完整任务域、工作量分配、core 起止游标和版本化表 | 按 tensor 数据值预计算 attention、缓存旧长度 |
| TilePolicy / ResourceLayout | 通用及优化分支条件、tile、L1/L0/UB/workspace 峰值和对齐 | 把资源不适配的合法输入直接判非法 |
| CubePipelineOwner | 持久 CATLASS 资源、QK/PV typed views、owner 内槽状态 | host 校验、输出缓存、每 task 重置尚存活的流水状态 |
| SoftmaxStage / OutputStage | mask/sink/exp/分母、P cast、归一化、O/LSE scatter | 重算另一份不一致的任务/窗口地址 |
| KernelScheduler | 从 metadata 恢复任务，组织预热/稳态/排空及空片协议 | 大段混写校验、地址、向量数学、包装注册 |
| FormalAdapter / DiagnosticAdapter | 各自 ABI 到同一规范描述、workspace、stream、构建注册 | 两套数学核、忽略正式 metadata、诊断接口隐藏扩域 |

命名应呈现量纲和生命周期，例如 `queryStorageBase`、`queryValidLength`、`kvTokenStrideElements`、
`pageIndex`、`taskOrdinal`、`scoreReadyEvent`、`numeratorFreeEvent`、`outputTask[slot]`。
字节偏移与元素偏移分开，batch/head/query/tile 的 index/count/end 分开；不要用同一个 `len`、`off`、
`flag1` 跨含义复用。硬件已定义的 M/N/K、L0A 等常用符号可以保留。
容量和 event ID 使用有含义的命名常量及对应设计表，不把偏移魔数散落在计算语句中。

每个函数执行一个能在设计图中命名的步骤。正常缩进、函数/类型定义分行、条件分支显式展开；
不压成一行宏或混合几十个副作用的表达式。注释解释窗口边界、尾块、最后消费者和单位，
不逐句翻译语法。向量热路径可以融合，但融合边界和同步理由仍应看得见。

<a id="engineering--计算-stage-与源码映射"></a>
### 计算 Stage 与源码映射

| Stage | 计算和状态边界 | 实现映射必须记录的内容 |
|---|---|---|
| S0 调度 | metadata 解码、task cursor、batch/head/query 范围、空任务 | 规划函数、消费函数、版本检查、唯一输出写者 |
| S1 选择与装载 | 当前任务 KV 并集、每行边界、Q/KV 地址、页分段/有限 padding | address 模块、CATLASS copy 或 pack 函数、实际 shape/ld |
| S2 QK | `[M,D] @ [D,N] -> FP32[M,N]` | CATLASS 实例、owner/view、score 槽、ready/free |
| S3 softmax | 逐行 mask、sink、FP32 max/exp/sum、P cast、invL/LSE | Vector 函数、行到 head 映射、P 槽、每槽状态 |
| S4 PV | `P_dtype[M,N] @ KV_dtype[N,Dv] -> FP32[M,Dv]` | CATLASS 实例、共享 KV 的最后消费者、numerator 槽 |
| S5 输出 | numerator 归一化、O cast/scatter、LSE、padding/无效行清零 | 输出地址、待输出 task 描述、最终消费者及槽释放 |
| S6 可选归并 | 仅实际选择 split-window 时合并在线状态 | sink 只一次、归并核、完整计时范围与精度证据 |

工程 `docs/design.md` 的映射列填写**本轮真实路径与类/函数/模板名**。同一函数包含多个 Stage 时写明内部边界；
同一 Stage 跨多个文件时列全。变更 tile、槽、事件或通用路径后更新映射及设计再复验。
源码行号可以作为辅助，不能只靠易漂移行号而没有符号。

<a id="engineering--分核策略与基本块切分"></a>
## 分核策略与基本块切分

任务用 `(batch, kvHead, queryStart, queryCount, headStart, headCount)` 描述。
任务内行 r 映射 `t=queryStart+r/headCount`、`h=headStart+r%headCount`。
任务不得跨 batch；query/head 尾块使用实际 count。跨 query 合并 M 时先证明物理行连续，
head 切片不连续时使用显式分段装载或 pack，不能伪造一个 ld。

规划器枚举逻辑任务并验证每个有效 `(b,t,h)` 恰有一个最终写者；无效前缀和 padding 另有覆盖证明。
成本可按实际行数、KV 并集、分页段数和矩阵 tile 估算，不能按容量元素简单平分打断窗口。
少 head 长 prefill、多 head 短序列/decode、任意尾块可有不同 tile 策略，但共同消费相同逻辑描述。

<a id="engineering--metadata-不因固定输出表缩小-batch-域"></a>
### metadata 不因固定输出表缩小 batch 域

metadata 使用按 batch 大小创建的临时规划向量，输出保存每 core 的启用位和半开任务区间；
main 按起止游标跨 batch 推进。固定 INT32[1024] 表并不保存所有 batch 的完整记录，
因此不能推导 `B <= (1024-header)/record_size`。[^engineering-swa-runtime-schedule]

新成对算子可以采用自己的版本化表；与旧表字节互通并非默认要求。一个可行设计为：

1. 表内保存 magic/version、分支/tile、实际 core 数、任务总数和每 core 起止 cursor；每个 cursor 包含恢复该位置所需的 batch/head/query 信息。
2. batch 的 cu/used 值继续来自本次公开长度 tensor；正式和诊断适配确需统一控制区时，在 workspace 中按 B 动态分配，大小和 H2D/整理成本都计入调用。
3. host 规划遍历各 batch 并均衡工作量。core 从分配给自己的起点恢复后单调前进；不要对每个 tile 都从 batch0 扫描到当前 batch，造成 O(B×task_count) 控制开销。
4. 检查表、控制区和索引整数容量；禁用 core、保留字和空计划确定性初始化。main 验证版本并实际消费表，不能只接收一个未使用指针。
5. shape 相同但 used/cu/page/stride 改变时重新使用当前值。host 数组不得被异步 device kernel 直接解引用；其设备控制区与 host staging 的寿命必须覆盖拷贝及消费。

小 batch 内嵌可以是优化路径，容量外必须落入合法通用路径。私有表只容纳116个 batch 却拒绝117，
属于生成缺陷；解决方案是改编码或使用动态控制区，不是把116写入接口规范。

<a id="engineering--数据路径与存储层级"></a>
## 数据路径与存储层级

以逻辑首元素为 base，地址统一使用带溢出检查的64位元素偏移：

| 对象 | 地址关系 |
|---|---|
| BSND Q | `b*qStrideB+t*qStrideT+h*qStrideH+d` |
| TND Q | `(cuQ[b]+t)*qStrideT+h*qStrideH+d` |
| BSND KV | `b*kvStrideB+j*kvStrideT+n*kvStrideH+d` |
| TND KV | `(cuKV[b]+j)*kvStrideT+n*kvStrideH+d` |
| PA KV | `physicalPage=table[b,j/pageSize]`；`physicalPage*pageStride+(j%pageSize)*tokenStride+n*headStride+d` |

上述形式假设 D 连续；支持其他布局必须给出真实地址/搬运方案，否则依据冻结契约拒绝。
Q/O、LSE 和 page table 也要分别核对自己的布局/stride，不能只检查 KV。storage_offset 已包含在逻辑首地址时不能重复加。
PA 首次搬运可能从页中部开始，每段长度取当前页剩余、逻辑剩余和 tile 剩余的最小值。
页表只访问覆盖有效逻辑区间的条目，物理页边界必须合法；随机页号和真实 page stride 参与正确性探针。
多个逻辑页可指向同一只读物理页，既可跨batch共享前缀，也可在同batch重复。Address模块逐逻辑页查表；Planner按逻辑长度计算任务，不能按独占物理页数量估计有效长度或消除重复页。验证器与数据生成器也要保持此区分，独占页随机分配的前提不能反向约束接口。

每个 task 给出 Q/KV/P/score/numerator/invL 的生产者、最后消费者和存储层级。
L1/UB/L0 大小按**同时存活峰值**计算；KV 同时服务 QK/PV 时，以最后一次 PV 读取作为释放依据。
需要 pack、零填充或布局整理时，缓冲区、数据寿命、同步和完整 device 时间都列入设计。
无效 V 槽必须有限初始化，`P=0` 不能消除 `0*NaN`。

<a id="engineering--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

长任务序列采用持久 Cube owner 和 typed QK/PV views；owner 的资源、event 和索引寿命覆盖整段任务流。
每槽保留实际 task 地址、shape、KV 起点及 invL，直至对应输出写完；跨 batch、head 或尾块时不能用当前 task 覆盖待输出 task。
优化路径满足 `score ready -> softmax/P ready -> PV ready -> output/free`，并单独证明最后读到可覆写的反依赖。
late free 只阻塞确实会覆写的资源，不能用无关全局等待使流水串行化。

两侧 AIV 即使某侧0行，也必须完成该任务所需的通知；整个 task 无计算时由同一调度判断共同跳过或执行有界清零。
启动、稳态、至少两次槽回绕、末尾排空以及下一次 launch 的初始状态均列入 token 数量证明。
同一地址 A/B/A/B、跨 shape 交错和全零有效工作须检查无死锁、无旧结果、无漏写。

可读性不能通过牺牲流水实现：调度层调用具名 Stage，Stage 内保持批量 Vector 和 CATLASS 操作。
不为每个 head/row 重复 scalar 造 mask、重复 GM sink 读取或不必要 barrier；sink 可在本次 launch 内片上复用，
但不能跨调用缓存。通用路径同样必须数值正确、同步闭合；优化谓词不成立时回退通用路径，不能缩域。[^engineering-swa-pipeline-knowledge]

<a id="implementation--代码模式"></a>

## 实现起点与工作量检查

<a id="implementation--算子算法"></a>
## 算子算法

数学顺序保持本家族的 `QK → FP32 mask/max/exp/denominator → 低精度 P → FP32 PV → O/LSE`；本页只约束普通 SWA 的任务合并和数据复用，不改变 sink、无效前缀及 golden 输出语义。

<a id="implementation--分核策略与基本块切分"></a>
## 分核策略与基本块切分

对固定的一个 query-head 微任务，若把整个 `[128,512]` 低精度 KV 窗口每次从输入读到 GM pack，单是输入读加 pack 写就至少 `2*128*512*2=262144` 字节；QK 和 PV 从 pack 再读至少增加同量流量。以 H64、T8192、每次只处理16个 head 为例，`8192*4=32768` 个微任务，仅 pack 的读写就达 8 GiB，尚未计入 QK/PV。这种设计在开始编码前就必须重新分组和复用，再按调用者提供的同输入基线计算实际时延门槛。算账时用实际有效窗口和布局，不把上界当成实测。

把 QK、softmax、PV、输出拆成四次全网格设备 launch，并为每个 work 常驻 score/P/delta，只适合给一个小输入定位正确性。以 H64/T2、4096 work 的设计样本估算，若每个 work 的 FP32 score 为128 KiB、P 为64 KiB、FP32 delta 为256 KiB，光三段中间件就约1.75 GiB，加行状态约1.754 GiB（约1.88 GB）；它们还要跨核写回/读回。开始长 prefill 验收前，必须改为同一 work 的 QK→AIV softmax→PV→AIV 输出流水、按 core 复用有界槽，并核实 ready/free 事件与最后消费者排空。短 decode 也不能让四次 launch 和逐行 scalar GM `GetValue/SetValue`、逐行 Exp/Ln 事件成为最终路径；具体时延上限由调用者提供的对应基线计算。AscendC 标量 `expf/logf` 在当前设备编译中不能作为可移植假设；用向量 math 时按行组批量处理，并显式检查 UB/事件占用与分母未 cast、P 已 cast 的顺序。诊断版本和性能候选要分别标识，不能把能编译的串行版本当作达标设计。

<a id="implementation--数据路径与存储层级"></a>
## 数据路径与存储层级

同一 KV 并集至少应在本任务的所有 head 行之间复用，QK/PV 应共享有效 KV 缓冲的生命周期。连续 BSND/TND 或可表达的正序列 stride 优先直接用物理 leading dimension；PA 仅对不能直接表达的物理页段做有界 gather/pack。PA 打包不能按每个 head 重做，也不能先复制整张 KV。长 prefill 再比较跨 query 合并 M64/M128、N 和波次数；短 decode 比较完整 launch/同步成本。所有 pack、清零、metadata 和额外 kernel 均计入本轮完整执行时间。

<a id="implementation--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

已分析的 CATLASS MLA 示例展示过 QK/PV 两次 `BlockMmad`、`CrossCoreFlag` 与 `SetSyncBaseAddr` 的组件链；这里保存的是 SWA 所需的组件边界与流水约束，不保存源码快照。目标构建须在实际 CANN/CATLASS checkout 中核对版本、头文件、示例、ABI、同步基址及资源 owner 生命周期。只借用公开组件的调用方式，SWA 的窗口、sink、metadata、PA 地址与数值顺序仍按本知识包和调用者提供的标杆推导；MLA 示例的数学与 shape 不能直接作为本算子实现。

可先用一个设备探针证明本轮 metadata 表能从 GM 被 AIC/AIV 读取，再实现最小 QK→mask/softmax→PV→O 闭环，并逐步加 LSE、TND、PA 和尾部。设备读表探针及诊断 C ABI 只证明传输和局部接口；主核必须消费同一轮正式 metadata，最终通过已注册的 `torch.ops.cann_ops_transformer.sparse_flash_mla_metadata` 与 `sparse_flash_mla` 调用原 ATK。不能把诊断库、CPU reference 或已有系统算子代替正式包。原 ATK 脚本若能加载预装的同名系统算子，单独运行它只验证该预装包；在本轮正式包注册、安装并记录二进制身份之前，结果一律不归属本轮候选。ATK 执行环境中还须核对 `python`、`ninja`、CANN 环境与候选动态库路径实际一致；加载依赖报错与设备精度失败分开记录。

普通 SWA 的实施检查点：metadata planner 完成本轮目标用例的任务覆盖并能生成本轮版本化表后，就用该表在设备上打通至少一个非零 Q/KV/sink 的 QK→softmax→PV→O/LSE 主核 smoke。此时可以使用诊断 launcher，结果只标 `direct_device`；随后补齐四布局和尾块，再接入已冻结的两个完整公开入口。交付层按[接口页](interface.md#api--用法)确定，不能把简化诊断ABI等同公开PyTorch，也不把完整PyTorch包装等同ACLNN包互通。**以下AICPU说明适用于已要求ACLNN包的分支**：不要把所有时间先花在只含内部 AICPU OpDef 的 metadata 包上；`op_build --aicpu` 生成配置、AArch64 ELF 可加载、`RunCpuKernel` 可解析，均不证明公开 phase1、scheduler 或主核可运行。若生成的内部 AICPU op_api 在 phase1 报 executor/`binary_info_config.json` 错误，先核对当前 SDK 的 CPU-only 路由；公开 metadata 的 ACLNN launcher 应按[动态工程的 AicpuTaskSpace 路径](development.md#engineering--设备编译与正式注册的贯通检查)建立自身 executor，并实测 Compute 被调度与完整1024词回读。这里复用的是成对入口与 AICPU 提交协议，shape/tile 必须按当前参数推导。

<a id="sdk-build--代码模式"></a>

## 公共 CANN 构建工具准备

<a id="sdk-build--公共构建工具暂存算法"></a>
## 公共构建工具暂存算法

1. 解析 CANN 根的真实路径，要求存在；输出与 SDK 不得互相包含，输出本身不得是符号链接或 junction，
   且只能是新目录或空目录。候选使用独立 stage/build/install，避免混入旧对象。
2. 从 `tools/op_project_templates/ascendc/customize/cmake` 和 `ascendc/common/util` 建立源文件清单。
   前者映射到输出 cmake，后者映射到 cmake/util。逐级解析链接，检测文件/目录链接环和越出本 SDK 的目标，
   只接受普通文件与目录；记录原路径、解析路径、大小和 SHA256。
3. 以 common/util 为补齐集合，有效 customize 同路径文件优先。缺失或断链的 customize 工具仅可由本 SDK
   对应 common 目标补齐；无法补齐的路径须失败，不能从旧生成工程的 autogen 取文件。
4. 写入前检查目标路径的文件/目录冲突，并确认至少存在 `cmake/config.cmake`、`func.cmake`、`intf.cmake`、
   `device_task.cmake`、`util/ascendc_gen_options.py`。这些是该模板的检查项，目标 SDK 布局变更时应先核对模板。
5. 复制已解析的普通文件、保留权限，不能复制原相对链接。核对目标哈希等于清单哈希，且源文件在复制期间未变。
   最终输出中不得有剩余链接，所有必需工具仍应存在。
6. 全部完成后以独占创建发布 manifest：schema_version=1、status=COMPLETE、请求/解析后的 SDK 路径、
   输出/cmake 路径、两个来源目录、优先级、必需文件、被补齐条目以及逐文件源/目标路径、字节数和哈希。

# 约束

<a id="stages--约束"></a>

## 计算阶段与开发分支

仅在计算入口确认生成完整共享KV算子后执行本页阶段。将下面的替换写入当前算子工程的docs/design.md 与 docs/validation.md，并保留本页链接供Developer/Reviewer读取；不改仓库通用模板或其他算子的流程。

| 通用工程示例项 | 此算子在生成项目中的具体展开 |
|---|---|
| 单个Kernel编译/运行步骤 | 按五阶段流程展开下文S0–S6，每段有实际运行比较记录；metadata和主算子分别构建并联调 |
| GEMM组件/分支表 | 保留QK/PV的Catlass选型，另列ModePolicy/KvAccessor/TaskScheduler及第2节的实际分支条件 |
| 默认T1–T3及小M/N限制 | 改为本分支用例manifest与覆盖统计；G=1、小query、任意K尾是合法逻辑shape，使用物理补齐和mask处理，审查不能因其非整块而删除用例 |
| 标准FA的BNSD、PAGED及内置FA标杆约定 | 采用已冻结的BSND/TND/PA接口与用户标杆；通用MLA的独立RoPE分量不适用于本计算 |
| 简化Params或样例参数子集 | 完整保留接口页全部参数、可选输入、默认值及两项输出；用参数去向表验证wrapper→host→device接线 |
| 单输出比较模板 | 按01/02阶段固定的精度策略比较O与LSE（关闭时检查空tensor）；可扩展多输出/双标杆适配，复用判定逻辑，不临时放宽阈值 |
| 非GEMM组件缺失 | 将选择、mask、sink、状态更新封装进自定义Block/Tile，内部使用Ascend C向量/搬运API和固定tile内循环；QK/PV仍使用Catlass矩阵乘 |

内部工程名可用含catlass的项目名，两个公开算子符号按接口契约保留，工程名不决定数学路由。完整调用链使用本次生成的两个产物，不借用系统已有metadata实现来完成联调。

`docs/design.md`中的每个Stage填写：`公式、输入/输出及dtype、逻辑/物理shape与stride、owner/任务坐标、buffer字节数和最后消费者、ready/free对应PIPE与事件、检查点和用例ID`。`docs/validation.md`按Stage定向、公开接口/功能、边界/负向、中大规模/压力/复用、性能分别列清单路径、数量、分支覆盖和实际执行命令。预热/稳态/排空图与容量峰值属于此算子详设；其他算子是否需要这些内容由其自己的知识分支决定。

<a id="engineering--约束"></a>

## 动态工程与正式部署

<a id="engineering--行状态清零与vector地址对齐"></a>
### 行状态清零与Vector地址对齐

片上分配基址对齐不代表切片后的指令地址仍对齐。910B的FP32行状态即使只修改一个元素，也不能直接对任意`state[row]`调用要求32B基址对齐的Vector指令：地址增加`row*4`字节，非8倍数的row会破坏对齐。mask只限制哪些元素参与运算，不放宽指令的基地址约束。为每个Vector调用检查“分配基址＋slot偏移＋行偏移”的最终地址，按本次SDK重载核验repeat/stride单位。

LSE、invL和其他连续行状态宜保持对齐基址，用有效/无效行位图执行批量操作。query/head展开后，设本task起始query为`t0`、每query展开`Hg`行、本AIV起始矩阵行为`r0`，实际行数为`R`。在窗口右下对齐语义下，有效行区间为`[clamp((Lq-Lkv-t0)*Hg-r0,0,R), clamp((Lq-t0)*Hg-r0,0,R))`；区间外应零，物理pad行不能写到逻辑输出。按实际Vector处理宽度拆mask，避免移位等于或超过整数位宽；不要把R=64或固定AIV行数写进一般输入合同。无效前缀与存储padding使用同一行映射，切head时Hg取本task的展开数。

UB内对齐处理与GM输出尾部写回分别设计。支持字节长度的DMA可精确写出`R*sizeof(float)`，不代表先前Vector指令可以从任意4B偏移启动；反之也不能为了32B搬运方便覆写下一task或下一batch的LSE。零行AIV不发起非法Vector/DMA，但仍履行共同流水所需的ready/free。定向验证须包含首个非对齐行出现无效前缀、尾部padding、两者同时存在，以及整批零Q/零KV；只测整tile且全部行有效无法触发这种缺陷。

<a id="engineering--tiling与控制结构的对象表示别名和对齐"></a>
### Tiling与控制结构的对象表示、别名和对齐

结构体的字节布局与访问它的C++类型是两个约束。含`uint32_t`、`int64_t`、`uint64_t`、`float`成员的普通参数结构，不是一个`uint32_t[]`；将其地址转换为`uint32_t*`后逐word写入，再按原类型读取成员，不因地址对齐或字节数相等就合法。C++17的类型访问规则不允许用这种整结构别名覆盖不同类型的成员；`reinterpret_cast`本身也不建立数组对象或改变成员类型。真正声明为整数数组的metadata可以按其整数元素访问，不能把这条规则误解为禁止全部整数控制表。

恢复按对象表示传输的参数时，先创建生命周期已开始、自然对齐的本地对象，例如`Params params{}`；确认类型可平凡复制，再通过`unsigned char`等C++17允许访问对象表示的字符/字节类型复制完整的`sizeof(Params)`字节，之后统一按具名成员读取。支持当前地址空间与设备编译的`memcpy`也可用于对象表示复制；不能假定host标准库实现可在device调用。GM源指针保留SDK要求的地址空间修饰，源缓冲至少有完整负载，搬运指令的对齐与尾部访问范围另行核对。C++17中不能仅把原始存储强转成`Params*`就假设对象生命周期已经开始，也不应借用C++20的`std::bit_cast`作为当前工具链前提。

字节复制保证合法的访问方式，不负责修正生产消费双方的ABI。源字节须符合约定类型的有效表示；任意字节串不会因复制成功自动成为合法控制值。共享参数定义并在host和device两侧静态检查`std::is_trivially_copyable_v<T>`；需要用`offsetof`冻结字段布局的传输结构还应检查`std::is_standard_layout_v<T>`。将`sizeof(T)`、`alignof(T)`、关键成员的`offsetof`及标量宽度与协议值逐项核对，嵌套结构同样检查；大小相等不代表偏移相等。按对象表示跨编译端传输还需匹配字节序及浮点表示。若这些条件无法保证，使用明确的线格式逐字段解码和赋值；浮点位模式与整数数值转换不能混用。不要用`#pragma pack`掩盖偏移差异而产生未对齐成员访问。

将参数装入`INT64`属性或控制tensor时，也不把混合类型结构强转成`int64_t*`遍历。可将其对象表示复制到容量足够的真实字节/字容器，或按冻结字段协议打包，再由消费者采用对应方式恢复；若按word计数，须检查整除与余数字节，不能向上取整后越界读取。保留字段显式赋值；结构padding不等于保留字段，不能假定普通成员初始化使padding字节跨编译端稳定，也不以未经规范化的padding作协议比较或身份哈希。

对象表示修正与设备同步分别验证。字符访问不会代替producer完成、DMA就绪、workspace寿命或跨核ready/free；反过来，增加无条件屏障、`volatile`、强制内联或关闭别名优化也不能代替合法的类型访问与布局。参数装载封装为一个具名边界，后续调度和数学只读已恢复的成员；避免一处按word别名读写、另一处按成员访问同一对象。正式与直调入口使用同一参数解释规则，仍由各launcher负责自己的workspace和同步基址。

<a id="engineering--设备编译与正式注册的贯通检查"></a>
### 设备编译与正式注册的贯通检查

先建立本次SDK的接口位置表，避免凭相近头名编写正式包。在9.2.0-beta.2安装中，常用声明对应如下；其他版本仍核验实际头和导出：

| 工程边界 | 公共声明与含义 |
|---|---|
| `OpDef`构建注册 | `include/register/asc/op_def_registry.h`定义`OP_ADD`；只包含`register/op_def.h`并不能取得该宏 |
| ACLNN内部tensor | 目标 CANN 的 `aclTensor/common_types` API 提供 `GetViewShape/GetViewStrides/GetViewOffset/GetData/GetStorageAddr/GetPlacement`；本知识包的 [接口契约](workflow.md#materials--接口参考) 已摘要这些使用边界，构建时只需按目标 SDK 复核 ABI，不要猜测不存在的替代头名 |
| op类型与executor | `opdev/op_dfx.h`定义`OP_TYPE_REGISTER`；`opdev/op_executor.h`和`opdev/make_op_executor.h`提供executor相关声明 |
| 控制数据复制 | `opdev/framework_op.h`声明`op::CopyToNpuSync`，需同时核对当前`libnnopbase.so`导出 |
| AICPU tensor与属性 | `include/aicpu/cpu_context.h`、`cpu_tensor.h`、`cpu_attr_value.h`提供真实类型与方法；状态常量位于`pkg_inc/aicpu_common/context/utils/status.h`，使用时给该目标加入`pkg_inc`搜索路径 |
| tensor位置 | `exe_graph/runtime/tensor_data.h`的`TensorPlacement`包含`kOnDeviceHbm`、`kOnHost`等，不能猜成`kOnDevice` |

只重编正式 host 适配层时，复用当前工程已成功的 CMake 目标或真实 compile/link 命令，保留宏、C++ ABI、依赖顺序与 RPATH，避免重新拼一条不完整的 `g++` 命令。当前 SDK 的静态 `libtiling_api.a` 引入 `DlogRecord`、`CheckLogLevel`，其精确导出位于 `libunified_dlog.so`；`libascendalog.so` 中的 `DlogRecordForC`、`CheckLogLevelForC` 不能满足这些引用。使用实际 `nm -D` 和原链接行确认依赖闭合，再做新进程 `RTLD_NOW` 检查。静态库放在其依赖库之前；不要删掉 `--no-undefined` 来掩盖链接缺口。host 修复应单独比较安装文件，确认 AICPU 与 AICORE 数学二进制是否保持不变后再决定验证范围。

地址适配需区分`GetData()`的逻辑首元素和`GetStorageAddr()`的存储首地址。当前公共`aclCreateTensor`的FP16探针在view offset为3时，二者相差6字节；若使用`GetData()`，不得再次叠加view offset。该host描述符探针只确认接口的偏移含义，不证明device placement、异步寿命或任意view已经合法。正式validator仍按shape/stride/实际placement检查，每一层统一使用相同的基址约定。

模板资源尺寸应保持真正的device编译期表达式。普通host `constexpr`函数在host规划中可用，不代表在device代码取值时必然被常量折叠；跨编译端的调用可能留下未定义符号。共享尺寸关系宜用模板常量、可在当前ASC设备端求值的表达式或明确的设备函数属性，并以实际device链接结果核验。不要为消除链接错误改写容量公式、删掉边界检查或把当前case的尺寸直接写成常数。

共享数学头的device helper须具有当前ASC支持的设备修饰。普通C++ lambda即使位于设备成员函数内，也可能被编译成host调用；涉及GlobalTensor、SetFlag/WaitFlag、DMA的复用过程优先写成明确`__aicore__`成员函数。这类编译错误应修正调用属性，不删掉同步。管线枚举如`PIPE_V`按SDK实际声明使用，不能自行假设属于`AscendC`命名空间。自动同步关闭使用当前工具链公开的`--cce-auto-sync=off`，不虚构类似内部LLVM参数。

正式OPC的include/宏/同步选项需要进入实际device编译，CMake配置成功不足以证明已传入。某些SDK通用helper会将`COMPUTE_UNIT ascend910b`映射为legacy短名`ascend910`，而实际OPC任务使用`ascend910b`，从而整组选项被忽略。核对生成的compile-options清单、shortsoc与真实OPC命令；在工程已限定910B且所有variant共用选项时，可省略该过滤参数以生成`__ALLSOC__`选项组。保留目标平台限制，不能修改SDK全局文件或用缺失include删掉CATLASS依赖。

tiling key与真正生成的binary dispatch保持一一对应。若只有一个注册binary，内部根据运行时dtype/tile选择模板，host设置该binary实际存在的固定key（例如0），不同shape的策略仍由运行时参数决定；不能再用`dtype*variantCount+tile`请求未生成的key。若采用多key编译，则每个host可返回的key都要有对应构建产物和执行证据。固定binary key不等于固定输入shape。

PyTorch接线查询当前安装的公共头：`torch_npu/csrc/framework/OpCommand.h`的`RunOpApi`接收`std::function<int()>`，回调捕获tensor/executor/workspace资源以保持真实寿命；`FormatHelper.h`位于`framework`目录，`utils/OpPreparation.h`提供普通格式输出分配。检查实际physical format与stride，不能将NZ内部存储声明成ND。Torch扩展和op_api可有不同C++标准库ABI，二者应保持C接口边界，不跨库传递不同ABI的`std::string`等对象。头路径、链接和符号均用本轮SDK实证确认。

参数错误也必须跨框架回调可靠传回调用线程。在`RunOpApi`的worker中直接抛出预期的ACLNN参数异常，可能被框架转写成异步`Inner error`，部分同步路径还可能在没有Python GIL时触发异常转换。CANN桥仍只返回C错误码；Torch侧在回调执行线程立即复制可能属于thread-local的错误文本，捕获异常或保存错误状态，并通过拥有完成/可见性保证的每调用状态传回调用者，再在调用线程抛出。可用标准future/packaged_task实现异常传递，不让异常逃出framework callback；不要以未经证明的栈引用或单个共享bool替代完成协议，也不要在另一线程重新读取last-error。框架提交本身的失败正常传播，不等待一个没有被接受的callback。

phase1拒绝后不提交phase2，所有已拥有的描述符按各自API释放；phase2已部分提交时须遵守设备完成后才能释放资源的既有规则。错误转接不能吞掉实际设备故障或把任意异常归为参数错误。正式默认异步配置分别检验main/metadata拒绝、原错误类别/文本，以及多次负例后同一进程再次成功执行合法metadata→main；仅开启`ASCEND_LAUNCH_BLOCKING`通过不能代替此合同。调试开关在正式性能采样前关闭。

当torch_npu自带的ACL/GE头与当前CANN版本不同，不在一个翻译单元中混合两套定义。`FormatHelper`、`NPUFunctions`等头可能间接引入bundled `graph/ge_error_codes.h`、`acl_mdl.h`；再包含当前CANN `opdev`/`acl_rt.h`会引起枚举或结构重复定义。不能删SDK声明、遮蔽include guard或定义重命名宏来压过冲突。将适配分为两个独立编译单元：Torch侧只含其兼容的bundled头、管理Tensor与当前stream；CANN侧只含当前SDK与本轮公开接口声明、创建和销毁aclTensor/数组并调用ACLNN。两侧之间只传C兼容的定长标量、指针、shape/stride数组及opaque句柄，不传C++容器和异常。CANN桥接对象按op_api所需ABI独立编译，Torch对象按实际torch ABI编译；分别记录include搜索路径和编译命令。桥接公开签名在SDK侧逐参数静态校验，资源寿命覆盖异步提交及消费，运行时继续验证实际加载库与当前stream，编译成功不代表正式入口已验收。

正式OPC混核编译还需要保留C++函数的目标核身份。AIC和AIV可以被分别编译后以`ld.lld -r`合并；共享头中含`ASCEND_IS_AIC/AIV`分支的外部模板若在两侧生成同名weak/COMDAT符号，链接可能合并成同一个实现，使AIV执行AIC路径并停等本应由自己发送的通知。为依赖目标核的自有helper采用内部链接、按目标核区分命名空间，或在合理边界保证内联；接口声明本身不能据此改名。核对合并object的符号和调用路径，并用正式入口验证两类执行单元都完成任务。独立直调编译通过不证明OPC链接正确；只有内联后小诊断返回也不证明正式精度或性能通过。诊断宏、打印和强制内联实验不应混入最终性能包。

- 新工程分支应形成“完整域 = 通用路径域 ∪ 优化路径域”，并证明各优先级和覆盖；优化路径只是策略选择，不是公开合法性判断。
- 通用路径与优化路径使用同一有效长度、layout address 和数学契约。正式与诊断包装复用本轮规划和计算逻辑。
- 按当前五阶段工作流保持两章 `docs/design.md` 的完整 Stage/资源/同步证据及 validator 原始结果；额外推导可放 `docs/design.full.md`。模板填空、文件存在和脚本返回0不能代替容量、地址与同步评审。
- 本页不提供源代码模板或固定 tile 最优结论。提炼机制可以，向隔离生成阶段提供完整手写或上一轮工程不可以。
- 静态源缺口不改变用户 golden；修正后的新能力需要独立证据。源码 host 接受但设备未验证的组合不得报告数值 PASS。

<a id="engineering--正式executor中的host控制数据传输"></a>
### 正式executor中的HOST控制数据传输

每batch的`cu/used`整理记录需要设备读取时，先在本次executor中用`AllocHostTensor(shape, DT_INT64)`取得有明确所有权的HOST staging，再把本次记录复制进去。针对当前9.2.0-beta.2的这条调用链，使用已实际导出的`op::CopyToNpuSync(const aclTensor*, aclOpExecutor*)`取得设备tensor，并把它交给同一executor的主核launcher。不能让kernel解引用HOST指针，也不能让异步任务引用phase1结束后销毁的局部vector。

这里的同步选项有实际接线差异：`AllocHostTensor → memcpy → CopyToNpu → AICORE`曾使正式主核提交后一直等待，连没有任何数学与跨核等待的空kernel也不返回；仅去掉前置copy后空kernel正常，保持真实记录并改为`CopyToNpuSync`后，已测的小输入在三种控制参数placement下返回且O/LSE正确。这定位了该提交链的控制传输问题，不代表其他shape、尾块或数学流水已通过。因此生成这类正式控制区时先选用这条已验证的传输路径，再执行完整域验证；不要把`CopyToNpu`与`CopyToNpuSync`当作只影响host耗时的可互换拼写。该结论限定于这条SDK/executor路径，不表示所有异步DMA都不可用。

公开HOST metadata或页表确需转为设备输入时也统一使用明确的控制传输helper，已是DEVICE的数据直接保留。先明确复制后的逻辑表示与消费者的寻址协议；传输helper返回的tensor可能是运输用的一维view，其stride不能直接当作原二维页表的行步长。本次输入值、workspace、host staging和设备目标的寿命均覆盖消费。同步H2D、此前读取控制值的D2H及host规划计入完整调用链；主数学device、控制传输、metadata与带profiler的wall分别报告，不能从端到端边界偷偷去掉整理成本。用同shape但不同used/cu/page内容及跨shape交错调用验证，不能只验证首次静态输入。

对公开INT32 `[B,W]` HOST页表，一种已验证的规范化协议是：按原逻辑首地址及原生行、列stride逐元素读取，在本次executor拥有的连续HOST staging中按`b*W+j`打包；将实际物理存储作为一维INT32 tensor上传，再明确给主核传入逻辑行步长`W`。不能从上传返回view的stride0推断`W`。若逻辑元素数为零而复制API不接受零大小，单独分配一个物理占位元素，逻辑尺寸与可访问条目数仍为零。DEVICE页表继续使用原设备地址和原生stride，不无条件改成`W`；地址的storage_offset已包含在逻辑首指针时不重复叠加。页表验证与打包使用同一份本次调用的值，不能缓存上次内容。

这类传输错误可能只破坏batch1及后续：batch0行起点为零，错误行步长暂时不显现。验收至少覆盖B>1、非恒等页号、HOST连续及带padding的页表、同值DEVICE页表、跨页有效区间，以及同地址不同内容的A/B/A/B调用；完整比较O/LSE并检查非有限值。HOST失败而DEVICE通过时，应检查传输后的逻辑布局和批次偏移，不能仅凭相同数学二进制就将HOST路径判为通过。

metadata的AICPU launcher同样需要完整的控制数据传输设计。公开`cu_q/cu_kv/used_q/used_kv`仍按INT32接口接收；若规划器内部使用INT64快照，应明确这只是内部表示。内部快照可以分别存四组数组，也可以合并为每batch记录；两者都先建立同一executor拥有的HOST staging，按实际数组长度或记录数分配并复制当前值，再通过`CopyToNpuSync`取得设备tensor，最后把设备tensor作为AICPU输入提交。数据源在HOST不改变设备消费者的所有权、复制依赖和完成前寿命要求；小B直接传HOST曾通过，也不能据此依赖隐式内嵌HOST数据路径。AICPU仍读取这些记录并独立生成任务表；host验证规划不能替代AICPU的实际产表，也不能通过系统已有metadata算子绕过本轮实现。缺省数组的presence与内部占位存储分开编码，不能把占位0误解释成用户传入了有效长度0。

不能因为小B的HOST tensor可直接传入AICPU，就依赖同一路径对动态长度一直有效。可重现的边界是四组HOST快照直接交launcher时，大batch在metadata的phase2完成处停等；显式设备快照路径在B=129、HOST/DEVICE两种公开控制输入位置及较广域验证中完成并得到正确O/LSE。B=129的INT64 `[B]`数组占1032字节只是探针大小，不构成已证明的SDK“1024字节阈值”或通用根因，也不是合法batch上限。因此此路径对所有B使用同一所有权/传输规则，不添加B<=128快路径或拒绝更大B来隐藏问题。若选择其他传输方式，须对该SDK、launcher及异步生命周期重新证明其正确性。

合并记录也有独立验证：INT64 `[129,8]` 的8256字节HOST控制表直接交AICPU时，metadata提交后120秒未完成；只把`AddMetadata`的输入改为已有`OnDevice` helper通过`CopyToNpuSync`得到的设备tensor，同一输入的metadata-only完成且全部1024word匹配，随后正式成对A/B/A/B通过。数学核、AICPU实现和任务表规划未改。这支持修复该提交链的控制传输，不证明SDK存在8192字节限额。验证先分离metadata提交、同步和全表检查，再接main；以跨小批量边界的B作探针，不把探针大小写入支持白名单或派发条件。

staging、设备快照、executor、workspace和输出表的寿命覆盖各自消费者完成；phase1返回后不得再引用局部vector存储，phase2提交返回也不能视为设备消费结束。跨调用不缓存旧控制内容。同一轮验证包含独立metadata-only完成与1024word表检查、正式metadata→main、HOST/DEVICE及合法缺省控制组合、B跨越小数组探针大小、同shape不同内容和跨shape交错调用。分别记录phase1、phase2及stream完成点，失败时先定位metadata还是main停等，不把AICPU控制传输停点归因于Cube/Vector同步。完整调用的控制copy与调度成本仍计入端到端边界。

正式调用阻塞时按提交链定位：先核对phase1/phase2/stream同步边界，再在单独诊断包中使用同正式封装的空kernel，单独开关前置控制copy。空用户函数仍会运行OPC生成的workspace初始化；仅看到主核同步超时不足以断言attention内部通知竞争。诊断包明确禁止用于精度和性能验收，修复后回到正常包重新运行完整数学。

<a id="engineering--诊断构建与正常构建的证据边界"></a>
### 诊断构建与正常构建的证据边界

同步基址的入口初始化与退出恢复是两个独立责任。正式OPC wrapper可能在调用用户函数前设置FFTS基址，而在返回后不自动清零；不能因入口归wrapper所有，就将退出恢复也跳过。读取本次实际生成的wrapper及公共SDK，明确谁初始化、谁在所有FIX/MTE3消费者完成后恢复。统一dispatcher中的早退路径同样经过必要排空与恢复；不能在跨核任务完成前清零，也不能在正式入口重新初始化并覆盖wrapper的workspace状态。正常包与插桩包分别验证，源码上存在清零语句不等于设备已执行到它。

在已核对的910B、CANN 9.2.0-beta.2普通OPC路径中，wrapper设置本次FFTS地址，用户函数返回后没有后续同步使用或退出清零；正式入口应在共享计算函数完成FIX/MTE3及全部跨核通知排空后执行`AscendC::SetSyncBaseAddr(0)`。空任务和提前结束分支也汇合到这一退出点，不能以C++函数已返回代替实际drain证明。该规则以未启用exception restart/super-kernel且实际wrapper尾部没有后续依赖为前提；这些模式或其他SDK需重新核对尾部协议，不能无条件在仍将使用基址的包装层之前清零。正常构建的重复/跨shape同步与数值探针、插桩构建的目标核finding分别保留，不能把框架辅助核的告警混称为目标核失败或全调用链已检查。

sanitizer执行命令使用目标Python解释器的绝对路径。在工程中恰有`python/`目录时，工具对裸`python`的程序解析可能选择该目录，打印工具完成且返回0，但用户测试根本未启动。检查测试自己的开始/结束记录、实际执行命令、预期kernel检查数量、数值结果及工具finding，不能只用进程退出码判断成功。无插桩的辅助kernel被跳过时如实列明，不能把主核检查覆盖扩大到全部调用链。

还要核对工具与目标程序的参数边界：某些mssanitizer版本会把目标Python的`-B`等参数继续当作自身选项解析，打印`unrecognized command`却返回0。按本版本help使用参数终止符；若不支持，用无参数可执行wrapper在内部`exec`绝对Python路径和全部测试参数。保留原无效执行记录并标为未检测，修正启动方式后重新执行必要检测；不能把空工具日志或仅有usage输出作为无finding的证据。

遇到越界报告时保留完整地址、访问大小、PC/源码行、当次task和分配生命周期。分别核验实际原始workspace分配范围、SDK系统区与用户区的偏移、每core/slot范围及完整DMA footprint；某个1024B子访问合法还需检查整个搬运。数值正确不能消除finding；写地址在实际分配内也不能直接判定工具误报。工具可能延迟打印，日志出现在host释放消息后本身不能证明设备在释放后写入。先用独立诊断记录事实，再修改已证实的机制，避免盲目扩容、加入cache刷新或改变分配器掩盖问题。

设备trace、额外GM写入、cache维护或改变内联方式都会影响指令排布和流水时序。增加诊断后某case通过，不能直接认定trace中的某条屏障就是修复。保留失败的正常包以及独立诊断包的源码/二进制身份；修复候选必须回到无trace的正常构建，以相同输入重验，再扩大到尾块、不同控制tensor位置和跨shape调用。

诊断record本身也是设备输出，需要证明写者、GM地址、缓存可见性和host读取时机。若某actor的marker缺失，但由它产生的数学结果已正确，不得据此断言actor未运行；先验证诊断写回协议。反汇编工具退出0但逐指令显示`<not available>`时，只能报告object/section/符号事实，不能声称读到了机器指令。某个整型立即数不以连续字节出现在object中也不能证明对应逻辑被删除。证据必须分case、placement、正常/诊断包记录，不能用一份ALL_NONE快照解释DEVICE调用的失败。

正式接口的小规模联调至少包含四个调度形态：单core整tile、单core部分tile、两core各一个整tile、两core且末core为部分tile。按运行时tile策略构造这些形态，检查实际metadata确认命中，不能仅凭S1大小猜测。再覆盖一个core连续多task以及至少两次槽回绕。单core部分tile通过而两core整tile失败时，应优先核对核映射、任务分区、正式控制传输和每core地址；不能继续将故障统称为tail问题。反之，固定核数改变尾行数才有助于分离尾处理缺陷。每次比较同一正式包的完整O/LSE，控制tensor的HOST、DEVICE及合法缺省位置分别记录。小输入验证只降低定位成本，不代替完整域与原测试清单验收。

候选身份同时绑定本轮源码、编译模式、安装根、device object和AICPU库，而不只记录`libcust_opapi.so`的哈希。仅改kernel时host壳的so可能逐字相同；只记录它会把不同数学binary混为一个候选。保留安装产物清单，运行前后复核哈希，固定本轮vendor路径并排除同名旧vendor混入。记录配置与安装状态不能冒充观察到了runtime实际选择；必要时结合loader日志核对。正式性能必须来自无诊断宏的正常包，修改trace选择器时解析同一批原始记录，不重新采样挑选结果。

<a id="engineering--aicpu入口的导出和运行验证"></a>
### AICPU入口的导出和运行验证

自定义metadata使用AICPU `RunCpuKernel`入口时，链接SDK的context静态库需要保留其动态导出。`--whole-archive`能把object带入so，却不能抵消`--exclude-libs,ALL`的隐藏效果；后者可能让生成的so包含实现但`dlsym("RunCpuKernel")`失败。参照当前SDK cpukernel构建规则，仅对确需隐藏的protobuf archive应用`--exclude-libs,<archive-basename>`，不要把context整体隐藏。对已安装的AICPU so实际运行`nm -D --defined-only`确认`RunCpuKernel`导出，同时检查main/metadata四个公开ACLNN符号、AI Core binary和注册名。metadata-only运行必须真正到达AICPU并返回完整表；`get kernel failed`/11003先核对设备日志中的缺失入口和加载的so，不能改数学核或用host计算替代正式metadata来掩盖注册问题。

<a id="engineering--pytorch动态库的实际可链接接口"></a>
### PyTorch动态库的实际可链接接口

成对动态库还要在一个全新进程中验证依赖解析。先手动加载metadata库再加载main库，可能掩盖main的`DT_NEEDED`依赖搜索缺口；公共验收工具先加载main时仍会失败。检查实际ELF的`NEEDED`及`RUNPATH/RPATH`，同目录私有库可使用正确转义的`$ORIGIN`运行路径；诊断时也可仅对该进程添加构建目录到`LD_LIBRARY_PATH`。重新构建后核对实际ELF并从新进程直接加载main，不能只以CMake属性存在或同进程第二次加载成功作为证据。这是加载前的工程故障，不属于设备精度失败。

公共头可早于或晚于实际安装库；编译器能看到声明不等于动态库导出了它。构建adapter前同时查看当前头和`nm -D ... | c++filt`，构建后在目标环境实际import扩展。某些torch_npu组合仅导出`c10_npu::getCurrentNPUStream(DeviceIndex)`，没有头中声明的`getCurrentNPUStreamNoWait`；使用实际可链接的当前stream入口，不能改为默认stream。类似地，`FormatHelper::IsBaseFormatType`和`OpPreparation::apply_tensor_without_format`可能未导出；普通`at::empty`分配后用当前公共`torch_npu/csrc/core/npu/NPUFormat.h`声明的`int64_t get_npu_format(const at::Tensor&)`核对实际物理格式。输入也做相同检查；只允许已声明支持的base格式，不能将NZ指针按ND解释。若选择显式format_cast，必须记录新buffer的寿命并计入调用开销。SDK头的函数签名和返回类型必须实际读取，不靠nm参数列表猜返回类型。

<a id="implementation--约束"></a>

## 实现起点与工作量检查

- 每个编译期 tile 先核对 L1/L0/UB 峰值、Q/KV/P 的真实 stride、所有尾块地址、AIC 与两 AIV 的事件生命周期；通用检查步骤见工作流知识。
- 不按 case ID 写多份 kernel，不用已安装手写主核或 metadata 代调。参数变化只从本次输入和本轮 metadata 推导。
- 性能报告必须保留实际候选包身份、原 ATK 精度结果及每例 5 次预热后 5 个完整设备采样；冻结手写基线只读，不重新运行。

# 失败表现

<a id="stages--失败表现"></a>

## 计算阶段与开发分支

sanitizer编译开关可改变指令布局和时序，不能依赖其“偶然不重叠”。最终用开启/关闭两种构建、多个种子、重复/交错调用验证；出现异常时比对首个错误Stage、真实地址和最后消费者，不能通过关sanitizer或无依据删除等待修复。

<a id="stages--分核策略与基本块切分"></a>
## 分核策略与基本块切分

先将每个完整(b,kv_head,Mblock)作为任务，成本近似有效行数×被访问KV列数×D，加稀疏gather开销；按成本前缀和划连续核区间。不要仅均分batch。metadata和主kernel共享同一个任务枚举规则，空任务要有明确跳过/零输出规则。

低B、decode、G小使完整任务数少时，再评估split-S2：边界按S2 tile切分，各片计算未归一化状态。空片为(m=-inf,l=0,u=0)并在合并时显式跳过；非空片可以不带sink，由最终归并统一加入一次：

```text
m = max(sink, max(m_i for nonempty partial i))
l = exp(sink-m) + sum(exp(m_i-m)*l_i)
u = sum(exp(m_i-m)*u_i)
O = u/l
```

另一合法方案是只在一个partial注入sink，两种方案不可混用。split改变低精度P舍入，需按冻结精度策略比较。不同CTA间不能使用仅核内HardEvent假装全局完成；选择独立归并kernel、或已验证的跨核完成协议，避免同一launch内忙等引发调度死锁。性能计入归并kernel、partial搬运和metadata；收益不足时保留FA-only并报告split候选没有收益；是否达标由[当前最低性能门槛](validation.md#coverage--性能验收与停止条件)判断，不能仅因未超过手写或未采用split就判定失败。

<a id="engineering--失败表现"></a>

## 动态工程与正式部署

输入仅因 B 超过私有记录数、长度不在选定用例、没有某个优化 tile 或启用 PA/LSE 而被拒绝，说明域被实现反向限定。
TND 将 used 当 storage base 会串 batch；用 cu 差值替代更短 used 会让 metadata 和计算分歧。
PA 只在恒等页表通过、stride padding 只在 B=1 通过，均不足以证明真实寻址。
NaN 输出残留、零任务不写 LSE、head 尾行覆盖下一 token，说明输出写者或 padding 设计不完整。
巨型单函数、无单位的偏移/长度、多处重复地址公式和数值魔数会使上述错误无法在 Stage 层定位。

<a id="implementation--失败表现"></a>

## 实现起点与工作量检查

- 设计校验通过却没有主核：先在上述独立 CATLASS 示例和本次 CANN 头文件上固定实际组件签名，完成一个最小设备 QK/PV 闭环；诊断 probe 不能算实现阶段完成。
- metadata ELF/内部 op_api 能加载但原 ATK 仍不能调用：检查公开27参数 metadata 与28参数 main 的 phase1/phase2、两种输出、安装身份和本轮主核；内部双长度输入 OpDef 不是公开 schema。phase1 的 executor 错误不能通过只改注册 JSON 宣称算子完成。
- 长 prefill 编译正确但远慢于基线：先列出微任务数、每任务 KV 读/pack/QK/PV 字节和波次数，再改跨 query/head 复用。
- 短 decode 慢：检查正式 metadata+main 的 launch 和同步，不凭单独 QK kernel 计时判断达标。

<a id="sdk-build--失败表现"></a>

## 公共 CANN 构建工具准备

报出具体缺失、断链、越界目标、冲突或哈希变化，不发布成功标记。失败留下的部分输出用于诊断，
修正原因后使用新的空目录，不静默递归清理 SDK 或覆盖旧候选。暂存成功只证明公共工具完整，
主核、metadata、构建配置和正式加载仍由本轮工程生成并验收。

# 验证方法

<a id="stages--验证方法"></a>

## 计算阶段与开发分支

在一个shell中激活Python环境、source目标CANN，核验compiler、目标SoC、设备健康和Catlass头文件；将实际命令写入生成项目run脚本。分别构建metadata和主kernel，启动新测试进程验证加载的是本轮产物。kernel-only、metadata-only、组合调用都保留可重复命令。正式ACLNN交付还要验证四个导出符号、注册、AICPU包及Python调用，不能以直调demo代替包交付。

测试adapter从[用例清单](validation.md#coverage)生成实际张量，调新metadata再调新kernel、同步、拷回全部输出。每个ACL调用检查返回值；CPU标杆和NPU阶段分开记录起止/耗时，超时写TIMEOUT而非精度FAIL。Stage dump只在调试构建按核隔离，最终性能构建关闭dump后重新验精度。

交付检查：两个真实实现/入口；三模式四布局；完整参数映射；Stage/分支/资源/同步表；可复现构建运行；全量用例与逐例结果；sanitizer和性能原始记录。空kernel、系统已有metadata代调用、只生成计划或只有smoke均不能标完成。

[^stages-computation-contract]: [各Stage的计算契约](computation.md)。

<a id="engineering--验证方法"></a>

## 动态工程与正式部署

在实现前实际执行 task 覆盖、地址边界、资源峰值及同步 token 的 CPU 枚举；保存输入域、随机种子、命令和输出。
在完整域内构造：四种布局、两种 dtype、全部合法 head 档、不同 B/长度、超过私有内嵌容量的 batch、
127/128/129边界、query/head 尾、非恒等 PA 页表、跨页起点、页 stride padding、Lq/Lk 不等、
零有效 batch、整批零有效工作、TND storage/effective 分离、非默认 scale、非零 sink 和 LSE 开关。
source 缺口的修正探针单独标记来源，不能称为原 kernel 已验证能力。

每种可达分支至少一个设备命中；main 和 metadata 各自校验异常输入，metadata-only 与配套调用都实测。
全量 O/LSE、padding、dtype/shape/轴序、输入不被改写、metadata 全字初始化及重复调用都检查。
直接复用ATK公共双标杆比较器时核对实例的实际阈值，不能只传`CaseConfig`就认为已应用JSON中的策略。有的版本构造器另以关键字参数接收`case_config.standard.acc["cv_fused_double_benchmark"]`，仅传case对象仍沿用构造器默认值。读取当前公共签名，显式绑定本次case策略并回读，使用已知通过/失败数据确认比较器生效。公共比较器通过与原ATK CLI整条接线通过分别记录，二者不互相冒充。
性能只在冻结的性能子集计较逐例指标，统计严格使用用户当前口径；不能把额外清零/pack/归并从完整 device 耗时中删除。
最终报告分别列完整域实现覆盖、正确性探针实测覆盖和性能子集结果，未跑与失败均保留真实状态。[^engineering-swa-diagnostic-contract]

[^engineering-swa-domain-contract]: [一般 SWA 契约](interface.md#swa-contract)：完整接口和 arch22 边界在知识中说明；本轮数值细节仍以调用者提供的 golden 为准。
[^engineering-swa-runtime-schedule]: [metadata 协议](metadata.md)：固定公开表、每 core 半开任务区间、启停和生产消费要求。
[^engineering-swa-pipeline-knowledge]: [一般 SWA 分支、地址和流水](performance.md#strategy)：运行时地址、持久资源、尾块和多任务同步。
[^engineering-swa-diagnostic-contract]: [SWA 诊断 ABI 与完整输出验证](validation.md#suite)：诊断扩展与正式接口的区分，以及动态输入和域内探针。


<a id="engineering--空-batch-的独立-metadata-与物理控制存储"></a>
### 空 batch 的独立 metadata 与物理控制存储

独立 metadata 的合法空 batch 要和 main 的 batch 约束分别处理。零元素不等于HOST placement：在已测CANN 9.2.0-beta.2 executor路径中，`AllocTensor(Shape({0,8}), DT_INT64)`返回非空tensor描述符，`GetPlacement()==gert::kOnDeviceHbm`、`IsEmpty()==true`、shape保持`[0,8]`，所报workspace字节数为0。此时没有控制负载需要H2D，可保留executor拥有的空DEVICE描述符；不能先分配HOST tensor再因空而跳过copy，却声称交给AICPU的已是DEVICE输入，也不能因零字节存储地址为空而拒绝合法空描述符。

该探针只证明空DEVICE描述符的分配与placement，不证明AICPU已经成功消费。AICPU仍按逻辑B、dtype、rank/shape、NumElements/DataSize验证，B=0不解引用负载并完整初始化1024word空任务表；分别实测独立metadata的B=0和普通B>0。若目标路径确实无法直接传递空DEVICE描述符，才评估全零最小物理记录的显式替代协议；同步更新host shape、复制字节数和AICPU padding检查，逻辑batch与任务数仍为0。`CopyToNpuSync`拒绝零元素不能自动推出必须padding、接口禁止B=0或可退回HOST输入，更不能靠绕过AICPU、主机伪造任务表通过。

<a id="engineering--正式安装的-elf-身份"></a>
### 正式安装的 ELF 身份

CMake 安装可能删除或改写 host 共享库的 RPATH/RUNPATH。因此构建与安装 opapi 的整文件 SHA256 可以不同：分别记录它们，核对实际安装日志、动态依赖和加载路径，并检查代码/常量/符号及非路径动态表等价。不能把合法的 RPATH 改写报告成算子构建失败，也不能忽略 `.text` 等计算相关段的变化。device object 未经安装变换时仍应逐个核对哈希。新进程直接加载主库，不能先手工加载 metadata 库掩盖缺失的依赖搜索路径。

<a id="engineering--stage-数值证据的保存"></a>
### Stage 数值证据的保存

将同一字节 workspace 的不同 dtype 切片保存到 PyTorch 文件前，为每个导出视图建立独立 CPU 存储（如 `.clone()`）。直接保存共享同一 storage 的 FP16/FP32 视图可能被 `torch.save` 拒绝；这是证据序列化错误，应保留此前数值比较和真实进程退出码，不能写成完整运行通过或据此重新采集性能。中间行状态没有实际导出时，应明确标记为由 score 推导；不能把推导值称为设备观测。


<a id="engineering--手动流水与-tpipe-的隐式事件所有权"></a>
### 手动流水与 TPipe 的隐式事件所有权

事件图必须把 SDK 对象的构造和析构也列入生产者/消费者。当前 arch2201 的 `TPipe::Init()` 会在 AIC 预置 `M_MTE1` 的 0/1/2 号事件，`Destroy()`/析构会消费它们。若 Cube owner 也在这些物理 ID 上预置和消费 token，即使每个任务的手写 Set/Wait 数目相等，整个 kernel 的生命周期仍存在重复 owner；一次小用例通过不覆盖连续调用。

只借 `TPipe::InitBuffer` 取得静态 L1/L0 TBuf、后续完全手工管理事件的实现，可以在缓冲分配完成后、自己的初始 token 发出前调用一次 `pipe.Destroy()`，完成 SDK 到手工 owner 的交接。后续不得再让 TPipe queue/event API 管理这些资源。仍需使用 TPipe 队列的实现应分配不与 SDK 冲突的事件，明确保留其生命周期。交接只在启动时完成，不在每个 tile 增加全流水排空。审查启动、正常退出、空任务退出以及同址 A/B/A/B 连续调用，记录每个物理 `(HardEvent, id)` 的唯一 owner。

<a id="engineering--自定义-aicpu-配置的安装文件名"></a>
### 自定义 AICPU 配置的安装文件名

公开 op_build 和 INI→JSON 工具生成出正确内容，还不足以证明 runtime 会读取该文件。自定义 vendor 的 CPU 注册配置必须放在 `op_impl/cpu/config/cust_aicpu_kernel.json`；`aicpu_kernel.json` 是内置包配置名，不能直接作为自定义目录下的最终文件名，也不能按算子名自拟 JSON 名称。配置目录不包含 `aicpu_kernel` 这一层：`op_impl/cpu/aicpu_kernel/config` 不会被这里的加载器发现。保持官方转换 JSON 的字节内容，将完整路径作为安装规则写入工程，同时安装其 `kernelSo` 指向的库到 `op_impl/cpu/aicpu_kernel/impl`。

若配置被跳过，日志可能尝试 `aclrtBinaryGetFunction(opType)` 并报找不到 metadata 算子名，即使库已正确导出 `RunCpuKernel`。先核对实际加载器查找的完整路径、算子 key、`opKernelLib=CUSTAICPUKernel`、`functionName` 与 `kernelSo`，再修安装布局；不要为此改数值 kernel 或凭空增加别名。可用同字节包的独立安装副本加上正确文件名定位，正式交付时还须修正安装脚本，避免下一次安装复发。

`OP_TYPE_REGISTER` 只建立 op 类型标识，库文件存在或 `RTLD_NOW` 成功也不证明自定义 CPU 注册表命中。当前加载器在初始化时读取 custom JSON，再用精确的 op 类型字符串查 custom registry；命中并成功加载后才标记 custom task。日志中的 new launch interface 也可用于非 custom 路径，不能据此判断自定义注册成功。安装后以仅指向本工程 vendor 的新进程完成一次真实 metadata 第二阶段、等待输出并检查整张表，再调用主算子；保留安装清单和真实加载身份。纯 host 的阶段一成功无法替代这项接线检查。单纯修正安装路径时保持计算二进制不变，不为排查注册问题反复编译或采集性能。

<a id="implementation--验证方法"></a>

## 实现起点与工作量检查

用调用者提供的原 JSON 按真实 case_id 选择目标用例，分别核对长 prefill、短 decode、ragged 与 PA 路径的覆盖和真实执行；先完成正式接口的精度，再用本目录 `performance.md#acceptance` 对 5+5 记录和调用者提供的可比基线逐例判定。此页的任务与流量推导属于设计筛选，不替代任何设备样本。

[^implementation-swa-pipeline]: [SWA QK/PV 与片上流水结构](pipeline.md#structure)。
[^implementation-swa-sync]: [SWA 跨任务握手协议](pipeline.md#protocol)。
