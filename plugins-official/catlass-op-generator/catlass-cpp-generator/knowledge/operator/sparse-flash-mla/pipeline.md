---
type: operator
title: SWA 流水协议、片上结构与状态模型
description: 跨任务双槽、三 actor 握手、QK/PV 片上资源、尾块和模型验证边界。
tags:
- catlass-cpp
- sparse-flash-mla
- swa
- pipeline
- sync
- slot
- CrossCore
- handshake
- L1
- UB
- vector
- prefetch
- pv-tail
- actual-k
- alignment
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
- id: protocol-sdk-sync
  resource: knowledge/operator/sparse-flash-mla/pipeline.md#protocol
  title: SWA 跨核握手约束与 SDK 核对项
  kind: contract
- id: structure-block-mmad-source
  resource: knowledge/operator/sparse-flash-mla/development.md#implementation
  title: SWA 的 BlockMmad 调用边界
  kind: contract
- id: structure-ordinary-contract
  resource: knowledge/operator/sparse-flash-mla/interface.md#swa-contract
  title: 普通 SWA 合法域
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

- [双槽跨任务握手](#protocol)
- [混合核结构与片上复用](#structure)
- [参考模型的使用边界](#model-usage)
- [三 actor 状态模型的构造与检查](#model)

# 接口与概念

<a id="protocol"></a>

<a id="protocol--接口与概念"></a>

## 双槽跨任务握手

本页是普通 SWA 的必读实现知识，配合[结构与片上流水](pipeline.md#structure)。工作流五阶段仍由通用 workflow 提供；这里的 V0/C1/V1/C2/V2 是运行时计算阶段。本文给出工程可翻译的协议，**不是已经通过设备精度/性能的候选代码**。生成者应按实际 SDK 签名完成实现，再提供设备证据。

普通 SWA 的一个 work 是完整 query/head 行组及其窗口并集。C1 计算 QK，V1 计算 mask/softmax 和低精度 P，C2 计算 PV，V2 归一化及输出。每 work 一次 softmax 覆盖整个 union 时，不需要跨 S2 tile 的在线累加；将 union 拆成多个 softmax tile 时必须使用另一套 m/l/alpha/u 生命周期，见结构页，不能原样套本协议。

**V0 不是每次普通 SWA 都要启动的核。** 稀疏索引 gather 才需要对应的 Vector V0。普通连续 BSND/TND 的装载属于 C1/C2 的 MTE2；普通 PA 可由 Cube 按逻辑页段直接装入 L1，或者采用本页后述有界 pack 扩展。独立 metadata 是 S0 调度信息生产端，不是 V0。

要区分三个层次：GEMM 内 MTE2/MTE1/MMAD/FIX 预取、Vector 内搬运/计算双缓冲、C1/V1/C2/V2 跨 work 重叠。`BlockMmad<Pingpong>` 只自动解决其中一部分；四个 kernel 依次 launch 不能叫跨 C/V 流水。

<a id="structure"></a>

<a id="structure--接口与概念"></a>

## 混合核结构与片上复用

本文与[跨任务握手协议](pipeline.md#protocol)一起用于首次生成普通 SWA。通用构建、五阶段流转、API 查询方法仍复用 workflow；本页只补 SWA 的 QK/PV 共享 KV、窗口复用、sink、M/N/D512 和双 AIV 的结构知识。不包含可直接复制的完整业务算子。

推荐的工程职责如下，名称可改，所有权不能含糊：

| 模块 | 持有/负责 | 不应承担 |
|---|---|---|
| MetadataPlanner / Decoder | 公开 metadata 生产、每核区间、同规则恢复 task | 缓存上一输入的 Q/KV/sink 或替用户做数学 |
| TaskInfo | 当前任务的地址、真实 stride、窗口、slot/generation、validRows | 随“当前 i”变化而覆盖在途旧任务 |
| Kernel / Pipeline | 混合 AIC/AIV 入口、持续循环、跨核通知与尾部排空 | 每 work 四次独立 launch |
| CubeOwner | L1/L0 张量、所有 HardEvent、persistent indices、QK/PV typed views | 两个会重置同组事件的 BlockMmad owner 同时占用相同资源 |
| KvAccessor | 连续/真实 stride/PA 页段到片上 tile 的装载 | 改输入布局契约、全量复制 KV、每 head 重新 pack |
| VectorBlock | 每 AIV 的 UB arena、行块 softmax、两槽 invL、输出 stripe | 每个元素 scalar 运算或让 zero-row 子核漏通知 |

# 用法

<a id="protocol--用法"></a>

## 双槽跨任务握手

1. 冻结[普通 SWA 契约](interface.md#swa-contract)、metadata 任务枚举、M/N/Dv 和两 AIV 行映射。
2. 选择一个完整协议。推荐本页的“双数据槽、V1 当前任务、V2 前一任务、U late-free”；不要将其他调度的调用位置或释放条件拼进来。
3. 写下本页各事件到实际 SDK API 的映射、PIPE、初始化次数、正常消费次数和尾部消费次数。核对 CANN 9.2.0-beta.2 实际依赖头及 CATLASS checkout。
4. 在 `docs/design.full.md` 保存地址/事件表和 L=0/1/2/3/7 演算；在 `docs/design.md` 保留确定的方案，在源码旁记录其符号对应。
5. 使用[验收](validation.md#pipeline-validation)检查实际代码及设备行为。仅填完设计模板或知识阅读回执不能判为已实现。

<a id="structure--用法"></a>

## 混合核结构与片上复用

把下文“形状→地址→资源→事件→调用”串成同一实现。声明某项优化已完成时，在 `docs/validation.md` 指向实际源码函数及调用点；只有文档名或 `Pingpong` 类型名不能证明跨阶段实现。资源容量和 tile 候选不是当前候选的性能保证。

长prefill的首次设计就为三个层次分别选定实现：跨C/V双任务槽、Cube局部预取与Q/P保留、Vector输入/输出子块流水。推荐以本页QP4/KV3、K0=128和r32双输入双输出作为容量已经展开的候选，再通过当前SDK核验和设备比较确定；不要在最终工程中将Vector流水长期留成“以后实现”。单矩阵UB、单L0C等较简单版本可用于阶段定位；若保留为交付分支，必须说明它的适用shape及实际完整性能证据。性能失败且这些机制尚缺时，当前轮应至少完成有明确依赖/容量的结构性优化与复验，不能只微调标量循环后把缺项标为已实现。小decode自然可选更简单路径，依据任务量和测量，不要求无用的缓冲或人为制造重叠。

<a id="structure--算子算法"></a>
## 算子算法

普通 SWA 每行至多看 128 个 KV，多个相邻 query/head 行组成一个 M 矩阵以共享窗口并集。设完整 head 数 H，选 T 个 query，`Mactual=Tactual*H`；物理 M<=128 是合适起点。union 最大宽度 `128+Tactual-1`，每行仍有独立窗口。H1/T128 的 N 最多255；H64/T2 最多129；H128/T1 最多128。物理 N 按 SDK 支持粒度补齐，并作为 X/P 的真实行距。

将整个 union 先做完整 C1，再一次 V1，再 C2，即便 C1 内切成 N128 的两次 MMAD，也不等于分了两次在线 softmax。只有真的分开执行多个 V1/C2 tile 时，才需要后文 online 状态。

首/尾/无效 query 的并集边界使用当前 Lq/Lkv 和真实 storage 信息；不要复制固定 S4096 的 KV 基址公式。细节在[工作量与分块](development.md#implementation)和[通用 SWA 性能](performance.md#strategy)。

<a id="structure--分核策略与基本块切分"></a>
## 分核策略与基本块切分

M<=128 的整体任务分配给一个 AIC、配对两个 AIV。若 Mphys128，两 AIV 每侧 R64；若 Mphys32，每侧 R16。行组物理对齐与 validRows 分开，尾侧可能0行但参加协议。M 切得过小会放大 task 数和 KV 重读；短 decode 任务不足时可评估切 head，而不是要求 B 足够大。

基础 Cube 形状可以使用 QK `C[M,N]`、PV `C[M,Dv]`，`Dv=128/256` 均是候选。必须满足当前 L0C 容量。例如 M128/N256 的 QK C 恰为128 KiB，PV 一次 D512 为256 KiB，不能装入128 KiB L0C；分 Dv256 时单 C128 KiB，Dv128 时单 C64 KiB。选择两个 L0C 槽时相应减小块，不能将两个128 KiB槽放进128 KiB。

`BlockMmad::operator()` 的 actual M/N 必须不超过它的 L1 M/N shape；K 可以按实现内循环切分。任务M128配BlockMmad M16时调用者要切M，不能向单块传128。QK B 的 ColumnMajor 是物理 KV 的转置视图，PV B 的 RowMajor 是同一 KV 的正常视图；native token stride始终保留。

<a id="model-usage"></a>

<a id="model-usage--参考模型与工程自建验证"></a>

## 参考模型的使用边界：参考模型与工程自建验证

- [性能评价器](performance.md#acceptance)：读取冻结基线与本轮正式设备样本，按所选case逐例判断。
- [用例读取](validation.md#case-recipes)、[整套诊断](validation.md#diagnostics)、[三actor参考模型](pipeline.md#model)：使用前核对参数及接口边界，不能替代原ATK。
- [构建工具准备](development.md#sdk-build)：用于已声明的隔离构建流程。
- [三actor参考模型](pipeline.md#model)：构造方法与适用范围如下。

<a id="model-usage--普通-swa-双槽参考模型"></a>
## 普通 SWA 双槽参考模型

生成工程可依据[状态模型配方](pipeline.md#model)使用 Python 3.8+ 标准库实现三个独立 actor 的参考检查。它不读取 C++，不执行 SDK 或 NPU，也不包含主算子的数学实现。

在生成工程内按以下配置构造并保存参考检查：

按[参考模型构造](pipeline.md#model)分别执行自检矩阵、L=7且pack开启的配置、L=3且wrong_x_slot的负对照，报告分别写入本次工程的evidence目录，保留预期失败。
输出文件写入自己的工程，不修改冻结 skill。单配置 PASS 返回 0，FAIL 或 INCONCLUSIVE 返回 1；self-test 只有正常配置通过、负对照按预期失败时才返回 0。负对照命令预期退出 1，不能因此删除其日志。`--max-states` 限制探索状态数；搜索未穷尽且没有观察到具体违反时返回 INCONCLUSIVE，不能视作 PASS。

<a id="model-usage--模型覆盖什么"></a>
## 模型覆盖什么

- AIC、AIV0、AIV1 各自按程序顺序推进，广度优先枚举所有可达的三个程序计数器状态及其全部可执行下一步，不要求两 AIV 同步。未完成但无可执行步骤才算死锁；无限调度饥饿不在保证内。
- 通知按 family/slot/sender/receiver 建立概念 FIFO。AIC 发送到两侧，AIC 等待两侧到齐。task/generation 是诊断标签，不声称硬件存储这些字段。
- X/P/U、可选 pack 和每侧 invL 的双槽代次、ready 先于读取、旧 reader 完成后覆盖，以及尾部通知消费。
- 自检覆盖 L=0/1/2/3/7、pack 开关、正常与两种 free 变体；负对照为漏一侧 P、错误 slot、pack 放在 X wait 后、漏尾部 free、跳 X wait。达到搜索上限有单独的 INCONCLUSIVE 对照。

各阶段是**所有读写已完成的原子步**，不是 C++ helper 返回。零行侧保留通知，但这一抽象模型没有 DMA 地址和数值，不能证明其零填充、有效范围或片上资源安全。具体 descriptor ring 也不在模型内，TaskInfo 按不可变参数处理。

删除全部 U_FREE 的 `drop_u_free` 在此抽象下可通过：每侧已有 `V2(k) → V1(k+2) → P_READY(k+2) → C2(k+2)` 的完成链。`free_at_tail_only` 保留发送并在尾部消费全部 free；它隔离检查 U 覆盖与通知残余。二者不证明真实异步代码可以这样改，也不能用来替代 `omit_tail_free` 的残余检测。

<a id="model-usage--如何用于自己的实现"></a>
## 如何用于自己的实现

1. 先按参考程序实现并运行 self-test，保存命令、模型 SHA、完整 JSON 和实际探索范围，确认理解基准协议。
2. 在自己的工程按**最终源码**建立 actor 程序。把每个动作对应到真实函数、PIPE 和完成事件；记录与参考程序的逐项差异。若任务描述符、pack、通知位置或槽数变化，扩展相应状态与检查。
3. 若声称验证异步重叠，将 issue/complete 分开建模并显式保留尚未完成的 reader/writer，而非将函数返回当完成。不能仅把参考模型报告改名为“实际算子模型”。
4. 模型之后仍执行 HardEvent 与地址生命周期审查、零行/尾部/多槽回绕、A/B/A/B、原 ATK、最终完整 5+5 设备验收，见[流水验收](validation.md#pipeline-validation)。

每份 JSON 固定标明 `reference_model_only=true`、`verifies_generated_operator=false`、`verifies_cpp_sdk_or_async_dma=false`、`hardware_verified=false`、`automatic_pipeline_acceptance=false`。只运行未修改参考模型时，最终报告中“实际实现模型”仍须写 NOT_RUN；协议参考检查通过不能替代算子流水通过。

<a id="model"></a>

<a id="model--用法"></a>

## 三 actor 状态模型的构造与检查

把双槽协议建成三个独立有序 actor 的有限状态模型，可在实现前发现握手遗漏、代次错误和排空残余。
模型只是一项推理工具；知识库不附带脚本。生成工程如需自动探索，应按本节构造模型，并保存模型源码、
命令、输入、SHA256、探索范围和结果。实际实现模型必须对应最终 C++ 的函数、PIPE、完成事件和资源表，
不能把未映射源码的参考模型改名为“实际算子验证”。

# 代码模式

<a id="protocol--代码模式"></a>

## 双槽跨任务握手

<a id="protocol--算子算法"></a>
## 算子算法

本页描述整 union 的非 split-S2 路径：

```text
X_i = Q_i K_i^T                     # FP32，C1
m_i = max(valid(X_i*scale), sink)   # V1，sink 不乘 scale
E_i = exp(X_i*scale-m_i)，无效列显式清零
l_i = sum(E_i)+exp(sink-m_i)
P_i = cast_input_dtype(E_i)         # 分母先以 FP32 计算
U_i = P_i V_i                      # FP32，C2
O_i = cast_output_dtype(U_i/l_i)    # V2
LSE_i = m_i+log(l_i+1e-10)          # 按原golden；输出轴/空行按 contract
```

sink 没有 V 项；没有 sink 时不能代入 logit=0。空行/padding 不做 0/0，不读未初始化 X/U。缺省、尾块和极值 mask 语义继续服从 computation 与 performance 的一般 SWA 策略。

<a id="protocol--分核策略与基本块切分"></a>
## 分核策略与基本块切分

metadata 为每个逻辑 AIC 分配同一批 work 的半开区间 `[begin,end)`，L=end-begin。AIC 和其两个 AIV 从相同 metadata 恢复完全一致的任务序列。`i=0..L-1` 是本核局部流水序号，`global_work=begin+i` 是业务地址序号；slot 使用局部 `s=i%2`，generation 为 `i//2`。换 batch、head group、query 尾块时不重置流水序号。

AIC 使用自己的 block index；在本架构混合核入口中 AIV 使用 `logical_core=GetBlockIdx()/GetSubBlockNum()`、`part=GetSubBlockIdx()`。记录实测 `GetSubBlockNum()==2` 及 kernel mixed 类型，不能将 standalone AIV 的启动数等同于 AIC/AIV 配对。以物理 M128 为例两侧分 `[0,64)`、`[64,128)`，各侧 validRows 再裁剪到 Mactual。其他 M 从对齐行块推导；两个子核的物理 UB 是私有的，GM 行写区间不能相交。

零行 AIV 仍执行该 work 的 CrossCore 协议，仅跳过非法 DMA/Vector。只有 L=0 的整个配对组可一致退出；退出仍执行统一 mask/同步基址恢复。为 padding 写零的任务若保留在序列中，其 C1/C2 可不做 MMAD，但仍产生协议定义的 ready，V1/V2 明确初始化零输出，不消费未初始化数据。

每个 in-flight task 描述至少包含 `global_work, local_i, slot, generation, b, q0, head0, Mactual, Nvalid, Nphys, kv_begin, q/kv/out/lse_base, native_stride, first/last`。描述符、GM slot、UB 行状态槽是不同对象：描述符可用三项 ring，也可用每 slot 持久结构；必须保留所有未结束任务的原始信息，不能用当前 task 覆盖 V2 的旧 task 地址。

<a id="protocol--数据路径与存储层级"></a>
## 数据路径与存储层级

score X、P、numerator U 在 GM 中各有两槽，互不别名；两个 AIV 各有两份 invL（及需延迟写出的 m/l/LSE）。令 aligned `SX=Mphys*Nphys*4`、`SP=Mphys*Nphys*sizeof(Input)`、`SU=Mphys*512*4`：

```text
core_base = workspace + core * core_stride
X(i) = core_base + X_offset + (i%2)*SX
P(i) = core_base + P_offset + (i%2)*SP
U(i) = core_base + U_offset + (i%2)*SU
core_stride = aligned(2*SX + 2*SP + 2*SU + optional_regions)
```

各 region 分别按其 DMA/typed layout 要求对齐，偏移用 64 位字节值，typed Tensor 索引再转换为元素数。workspace 总量按**实际分配核数**计算，必须覆盖 launch 核数；不按所有 work 保存一份中间量。M128/N256 时 X/P/U 双槽合计 896 KiB/core，另计行状态、描述符和可选 pack。GM 中转是 A2 的正常路径，优化目标是有限生命周期和重叠，不是宣称所有中间结果都不落 GM。

每槽 writer/reader：X=AIC→两 AIV；P=两 AIV→AIC；U=AIC→两 AIV。最终 O/LSE 写入业务地址，与循环 GM 区分开。UB 能否复用要看核内 MTE2/V/MTE3 完成，CrossCore 本身只建立相应跨核边，不能包办 UB 依赖。

<a id="protocol--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

<a id="protocol--1-四组通知各两槽"></a>
### 1. 四组通知，各两槽

mode2 用于一个 AIC 与配对的两个 AIV。AIC Set 对应两个 AIV 各自 Wait；反向要求两个 AIV 各 Set，再由 AIC Wait 等齐，不是任选一个子核通知。不能将所有调用的数目简单加总后按一对一解释。使用实际 SDK/CATLASS 的配套 API，避免混入其他 mode/其他架构语义。[^protocol-sdk-sync]

本方案只占用 0–7 共八个 CrossCore flag，示意分配如下；确认不与同 kernel 使用的库组件冲突。某些版本允许更多编号，也不据此随意扩充。CrossCore flag 与 HardEvent 的编号空间及含义不同。

| 名称 | slot0/1 | Set 生产者与管线 | Wait 消费者 | 初始状态 | 每任务含义 |
|---|---|---|---|---|---|
| X_READY | 0/1 | AIC，PIPE_FIX，X 的所有 Fixpipe 写出之后 | 两 AIV，各等一次 | 无 ready | 本轮完整 score 可读 |
| P_READY | 2/3 | 两 AIV，各在 PIPE_MTE3，自己的 P 写出之后 | AIC 等齐 | 无 ready | 两半 P 都可供 PV 读取 |
| U_READY | 4/5 | AIC，PIPE_FIX，全部 D stripe 写出之后 | 两 AIV，各等一次 | 无 ready | 本轮完整 numerator 可读 |
| U_FREE | 6/7 | 两 AIV，各在 PIPE_MTE3，V2 输出链完成之后 | AIC 在同槽 PV 覆盖前等齐 | 无 free；前两次用槽通过逻辑跳过等待 | 旧 U 与对应行状态已消费 |

**不预发 X/P/U ready，也不预发 U_FREE。** 首次两槽本来没有旧任务，通过 `j>=2` 分支跳过 free 等待。若改成预装 free credit，就必须连同首轮、复用和尾部计数重写，不能既预发又跳过。

这些 flag 以 FIFO 序号隐含 generation，硬件不认识 task_id。保证同一槽旧通知被消费后才重用；软件记录 generation 用于诊断，不可用它代替实际完成条件。不要依赖计数器堆积无限 ready，也不要假定 CrossCore 是只能容纳一个值的普通布尔变量。[^catlass-sync]

<a id="protocol--2-推荐的完整双槽推进"></a>
### 2. 推荐的完整双槽推进

以下是独立 AIC、AIV0、AIV1 上的**各自指令序列**，不是让 host 顺序执行的调用清单。每个 helper 的完成语义由对应 PIPE event 落实。

```text
AIC:
  initialize persistent CubeOwner once
  for i in 0..L inclusive:
    if i < L:
      C1(i); set X_READY[i%2] from FIX
    if i > 0:
      j = i-1
      wait P_READY[j%2]                 # 两侧 P 已写完
      if j >= 2: wait U_FREE[j%2]       # 旧 generation j-2，放在 PV 前
      C2(j), including all output-D stripes
      set U_READY[j%2] from FIX
  for j in max(0,L-2)..L-1:
    wait U_FREE[j%2]                    # 恰好末 min(L,2) 个任务
  drain persistent CubeOwner; restore/exit

AIV(part=0 or 1):
  initialize private UB and persistent row-state slots once
  for i in 0..L inclusive:
    if i < L:
      wait X_READY[i%2]
      V1(i, part)                       # 保存 invL[i%2]，写本侧 P
      set P_READY[i%2] from MTE3        # 零行侧也参与
    if i > 0:
      j = i-1
      wait U_READY[j%2]
      V2(j, part) using row-state[j%2]  # 读取旧 task 描述符
      set U_FREE[j%2] from MTE3
  drain all UB/event writers; restore/exit
```

`for` 的上界包含 L，最后 i=L 只处理旧任务的 PV/V2，不解码第 L 个业务 work，不发虚假的 X_READY/P_READY。L=0 不创建外层通知；若 owner 已初始化则仍按 SDK 析构其内部资源 token，不能把资源事件与跨任务通知混为一谈。

为什么 V1 放在当前 i？它让 AIC 的 `C1(i)` 和 AIV 的前序/当前 Vector 工作发生交错，并让 `C2(i-1)` 与 `V1(i)` 有重叠机会。源码里两个 slot 却在每 task 内等齐 C1→V1→C2→V2 才进入下一个 task，仍然是串行。

<a id="protocol--3-填充稳态排空的逐长度演算"></a>
### 3. 填充、稳态、排空的逐长度演算

| L | AIC 矩阵指令序列 | AIV 各自阶段序列 | 循环内 U_FREE 消费 | 尾部 U_FREE 消费 |
|---:|---|---|---|---|
| 0 | 无 | 无 | 无 | 无 |
| 1 | C1(0), C2(0) | V1(0), V2(0) | 无 | 0 |
| 2 | C1(0), C1(1), C2(0), C2(1) | V1(0), V1(1), V2(0), V2(1) | 无 | 0,1 |
| 3 | C1(0), C1(1), C2(0), C1(2), C2(1), C2(2) | V1(0), V1(1), V2(0), V1(2), V2(1), V2(2) | PV2 前 free0 | 1,2 |
| 7 | Q0,Q1,PV0,Q2,PV1,…,Q6,PV5,PV6 | S0,S1,O0,S2,O1,…,S6,O5,O6 | PVj 前 free(j-2)，j=2..6 | 5,6 |

每侧 AIV 对每任务产生一次 U_FREE，AIC 正常循环消费 `max(L-2,0)` 次，尾部消费 `min(L,2)` 次，总共 L 次配对通知。X/P/U ready 各 L 次配对。不在析构中再等一遍已经消费的外层 flag。

<a id="protocol--4-四类内存复用的具体证明"></a>
### 4. 四类内存复用的具体证明

| 被覆盖内容 | 旧消费者 | 本协议中的顺序证据 |
|---|---|---|
| X(i%2) | V1(i-2) 读取 score | C1(i) 在 C2(i-2) 之后发起；C2(i-2) 已等到两侧 P_READY，后者位于旧 score 读取和 softmax/P 写回之后。 |
| P(i%2) | C2(i-2) 的 MTE2→L1 | V1(i) 必须等待 C1(i) 的 FIX 通知；持久 CubeOwner 的 L0C/MMAD/FIX 依赖使该 FIX 位于旧 PV 的最终 MMAD 之后，而旧 PV 的 MMAD 又依赖 P 已读入。 |
| U(i%2) | V2(i-2) 读取 numerator | C2(i) 覆盖前显式等待 U_FREE(i-2)。不能仅凭 P_READY 的名称推断旧 U 已读完；若另有 `V2(i-2)→V1(i)→P_READY(i)` 的真实完成链，可单独证明该依赖也保护复用。原子阶段模型具有这条链，实际异步实现仍需核对 MTE2/V/MTE3 完成。 |
| invL/m/l(i%2) | 本 AIV V2(i-2) | 本侧序列在 V1(i) 前执行了 V2(i-2)；UB 核内 V/MTE3 free 进一步保证真正完成后才覆盖。 |

P 的证明依赖**真实的 Cube 资源顺序**。只在 C++ 里调用 PV 然后 QK，但两个对象拥有重叠内存/事件，或取消了 L0C 的 unitFlag/FIX 依赖，就不能引用这条证明。此时必须补正确的 owner 或显式消费者完成条件，不能靠一个 `PIPE_ALL` 的名字猜跨核安全。

`U_FREE` 是 late-free：放在即将覆盖 U 的 C2 前，允许 C1 先写独立 X。挪到 C1 前会减少重叠；若改成下面的“三代阶段位置”还可能形成等待环，不能视为无害修改。

<a id="protocol--5-三代阶段位置与推荐协议不能混搭"></a>
### 5. 三代阶段位置与推荐协议不能混搭

另一种成熟组织是使用三份 RunInfo：第 i 轮 AIC 做 `C1(i),C2(i-1)`，AIV 做 `V1(i-1),V2(i-2)`，增加两轮尾部推进；普通 SWA 跳过稀疏 V0。其 descriptor 下标可为 `i%3,(i+2)%3,(i+1)%3`，而数据槽仍是 `task_i%2`。描述符三槽不意味着 GM 必须三槽。

这个组织的分块消费、片上资源复用和 CrossCore 等待必须作为一套证明。若在 C2(j) 前添加 U_FREE(j-2)，需把产生它的 V2(j-2) 放回三代调度图检查：它在较早推进轮、只依赖已发出的 C2(j-2)，而不依赖等待中的 C2(j)。此处可构造无环协议，但不能直接拿推荐组织的一轮排空及其描述符覆盖位置来实现。实际更换了等待位置时重新画等待图，查是否出现 AIC 等旧 free、AIV 等尚未发出的未来 score 的环；不能仅因 Set/Wait 数量相等就认为安全。

本页推荐组织与这一组织都表达跨任务重叠，但填充/排空长度、free 消费位置和行状态寿命不同。最终设计只能写自己的完整序列，不能并列抄两张表后由实现者猜选哪一行。

去掉空推进后，两种写法的有效 actor 顺序可以相同：AIC为 `Q0,Q1,PV0,Q2,PV1,…`，每个AIV为 `S0,S1,O0,S2,O1,…`。不同的是循环索引、valid描述符推进与尾部形式；三个actor的局部循环i不必同时到达。**不要在每个i追加全核SyncAll来强行对齐**：各逻辑core的L可不同，零任务core还可能已退出，既破坏重叠也可能令全局barrier参与次数不一致。只用已证明的配对数据依赖推进。

<a id="protocol--6-pa-pack-扩展的边界"></a>
### 6. PA pack 扩展的边界

首选通过有界页段 copy 直接构造所需 L1 B tile，此时保持上述四组通知。若使用 AIV pack，增加 per-core 两槽 packed KV、PACK_READY 两个 flag（例如8/9，先确认当前 SDK 范围与冲突）。两 AIV 都须在读取对应 X_READY 之前完成各自 pack 段，并发 PACK_READY；AIC 在 C1 前等齐。沿用原 V1/V2 序列。

pack(i) 重用 packed(i-2) 前，必须证明 C2(i-2) 读取结束：本 AIV 已等过 U_READY(i-2) 并处理 V2(i-2)，所以到下一轮 pack(i) 时旧 PV 完成。C2(i-1) 和 C1(i) 使用不同 slot。**不能**把 pack 放到 `wait X_READY(i)` 之后，否则 AIC 等 pack、AIV 等 QK，会立即死锁。两个 AIV 的 KV 装载分工按 N 列，和输出行分工独立；零 query 行侧仍可能负责非零 KV。

packed GM 在 QK 完成后仍须保留到 PV 读取完成。不能全体 work 先独立 pack，再启动四个串行阶段并声称已按本协议融合。pack 的 GM↔UB 本身也需双缓冲/批量页段，不能逐 token copy 后 wait。

<a id="protocol--7-退出与跨调用状态"></a>
### 7. 退出与跨调用状态

CrossCore 消费完成后仍排空最终发送管线和 owner 的内部 HardEvent。恢复 Vector mask 的高低有效位和 normal mask mode；所有 early-return 都经过同一 epilogue。直调 launcher 从本次运行时取得硬件 sync 地址并传给混合核；正式 OPC wrapper 如果负责初始化就按其实际协议使用，不重复覆盖。完成全部通知前不能清零同步基址。具体 API/责任见[工程知识](development.md#engineering)，不能将上次 launch 的地址/flag 状态缓存在全局。

<a id="structure--代码模式"></a>

## 混合核结构与片上复用

<a id="structure--数据路径与存储层级"></a>
## 数据路径与存储层级

<a id="structure--1-层级与地址"></a>
### 1. 层级与地址

```text
C1: Q(GM) -> L1(Q) -> L0A
    KV(GM/page segments) -> L1(K) -> L0B
    MMAD -> FP32 L0C -> Fixpipe -> X(GM slot)
V1: X(GM slot) -> UB -> FP32 mask/max/exp/sum -> cast P
    P(UB) -> MTE3 -> P(GM slot); invL/m/l 留私有 UB 对应 slot
C2: P(GM slot) -> L1(P) -> L0A
    KV(GM/page segments) -> L1(V) -> L0B
    MMAD -> FP32 L0C -> Fixpipe -> U(GM slot)
V2: U(GM slot) -> UB -> broadcast invL * U -> cast -> O(GM)
```

Q/P 的 A layout 不同 leading dimension：Q 是512，P是Nphys。QK 的 B 是 ColumnMajor `shape=(512,Nactual),ld=kv_token_stride`；PV 的 B 是 RowMajor `shape=(Nactual,Dv),ld=kv_token_stride`，地址增加 d0但 ld不变。U 的输出stripe ld始终512，不能改成Dv。Q/head切片不连续时须按行段搬运，不能伪造连续 RowMajor。

<a id="structure--pv-tail"></a>
<a id="structure--11-pv-通用尾块有效-k对齐尺寸和行距分别传递"></a>
### 1.1 PV 通用尾块：有效 K、对齐尺寸和行距分别传递

本节描述普通 SWA 的 PV 归约维处理，不新增数学模式。`Kvalid` 是当前任务实际 KV 并集宽度，来自本次窗口/metadata；它不等于输入 headDim，也不等于 score/P 的物理行距。手写路径的可复用方法是**保留真实计算长度，按片上格式对齐存储和搬运布局**。生成者根据下面的变量与参数映射自行实现，无需读取手写业务源码。历史诊断及来源身份见性能页的[实测反例](performance.md#strategy--pv-tail-evidence)。[^pv-tail-evidence]

| 量 | 含义/来源 | 使用处 |
|---|---|---|
| `Kvalid` | 本任务有效 KV 并集长度；连续完整 head 组按已定义窗口边界计算 | PV 总归约范围；不得因为对齐而改变有效 KV 集合 |
| `K1cap`, `K0cap` | 本编译变体的 L1/L0 K 切分容量 | 控制循环与容量，不代替最后一片的有效长度 |
| `k1`, `count1` | L1片起点与 `min(K1cap,Kvalid-k1)` | P 当前片有效列数、该片的内层边界 |
| `k0`, `kc` | L1片内起点与 `min(K0cap,count1-k0)` | 当前 V 片有效行数、当前 MMAD 的逻辑 K |
| `kPhys` | 当前片上格式所需的 `align_up(kc,aK)` | L1/L0 layout、LoadData 范围及容量；A2低精度候选通常取 `aK=16`，以选定组件为准 |
| `ldP`, `ldKV`, `ldU` | P 的 GM 物理行距、原生 KV token stride、U 的完整输出行距 | 地址和 DMA leading dimension；不随尾片 `kc`/`dv` 缩小 |
| `d0`, `dv`, `dvPhys` | 输出 D stripe 起点、`min(DvCap,D-d0)` 与对齐宽度 | 当前 stripe 的有效 copy 宽度、物理布局及 FIX；输出行距仍为 `ldU=D` |

`Kvalid`、`kPhys`、`ldP` 即使数值偶尔相同也不能共用变量。P 的 GM 行距可能是整个工作区的 Nphys；它既不必等于 `Kvalid`，也不必等于 `align_up(Kvalid,16)`。各量使用元素数还是字节数要在接口处明确转换。

<a id="structure--通用分片公式与参数映射"></a>
#### 通用分片公式与参数映射

下面伪码只表达长度、地址与首末 K 条件；`Prepare/Load/MMAD` 是职责名称，不是可直接调用的 SDK 接口。实际预取、typed view 和事件继续按本页 owner 协议实现。K0/K1 从资源表确定；要求对齐后的片能够装入对应缓冲，不能只比较有效元素数。

```text
for each output stripe (d0, dv):
  for k1 = 0; k1 < Kvalid; k1 += K1cap:
    count1 = min(K1cap, Kvalid-k1)
    Prepare P(row0, k1): valid_shape=(Mactual,count1), GM_ld=ldP
    for k0 = 0; k0 < count1; k0 += K0cap:
      kGlobal = k1+k0
      kc = min(K0cap, count1-k0)
      kPhys = align_up(kc,aK)
      Load V(kv_begin+kGlobal, d0): valid_shape=(kc,dv), GM_ld=ldKV
      Load P/V to L0 using their actual fractal layouts and kPhys
      MMAD.k = kc                         # 选定组件支持非对齐实际K的路径
      firstK = (kGlobal == 0)
      lastK  = (kGlobal+kc == Kvalid)
  FIX the accumulated U stripe, GM_ld=ldU
```

计数为0时不调用DMA/MMAD；外层任务仍按零工作/零行协议处理，不伪造 ready 或让某个配对子核提前退出。不能将 `count1` 用作每个内层 MMAD 的 K，也不能用 `k0+count1` 判断最后一片。改变 D stripe 不改变 K 的有效范围；同一 MN 输出的全部 K 片累加在同一 L0C 槽。

P 在 L1 跨 D stripe 保留时，沿用 owner 的 resident-key 和最后消费者规则：key 至少包含 task/generation、M片、K片及layout。只有当前容量确实容纳并保护了要复用的 P 片，才可跳过其再次装载；不能仅用 `d0==0` 加载一次，却在多个 K1 片轮转覆盖后继续读取旧片。

| 阶段 | 应传入有效长度 | 应传入物理布局/行距 |
|---|---|---|
| P，GM→L1 | 行 `Mactual`，列 `count1` | `srcDValue=ldP`；目标zN布局使用实际M对齐/分形stride |
| V，GM→L1 | 行 `kc`，列 `dv`；PA再切成有界页段 | `srcDValue=ldKV>0`；目标nZ/zN布局和对齐pitch由选定copy类型确定 |
| P/V，L1→L0 | 从本片对应位置取数据 | LoadData/layout 的 K扩展使用 `kPhys`，M/D同样按各自格式对齐 |
| MMAD | 支持实际K的实现传 `kc` | A/B/C布局、M/N参数满足实际SDK约束；不把 `kPhys` 无条件写入逻辑K |
| U，L0C→GM | 当前M/D有效输出范围 | 完整U的leading dimension，不变成当前尾stripe宽度 |

在当前A2手写式 `LoadData3DParamsV2` 的PV视图中，P作为A时 `kExtension=kPhys`；V转置装入B时 `mExtension=kPhys`、`kExtension=dvPhys`。同一个逻辑K在两侧对应的API字段不同，不能把P的LoadData配置原样套给V。采用CATLASS typed copy时从实际 `LayoutSrc/LayoutDst` 与copy实现确认这个轴映射，不把这些字段当作跨架构固定API。

对于连续原始KV，V源元素偏移是 `kv_base+(kv_begin+kGlobal)*ldKV+d0`；P源偏移是 `P_slot+row0*ldP+k1`（`row0`为task内行起点）。PA的 `kv_begin+kGlobal` 是逻辑token，先按实际页表映射，`ncopy=min(remaining,page_size-in_page)`；不能按 `kPhys` 多读页尾或下一页。多个页段写入同一目标片时按目标分形layout求偏移，不能简单用 `written*dv` 假装ND连续。全部有效页段及必要padding初始化完成后才发该L1片的ready。

<a id="structure--两种-l1-布局不能混用"></a>
#### 两种 L1 布局不能混用

- **CATLASS统一L1片布局**：从实际 `CopyGmToL1B` / `CopyL1ToL0B` 类型取得对应layout，按该layout的 `GetOffset` 定位子片，保留整片的固定pitch。尾片仅收缩有效copy/actual shape；不能自行把目标pitch改成尾片的 `kPhys`，而L0读取仍使用整片pitch。
- **按K0分段打包的手写组织**：每个K0片有自己的分形布局，片起点按固定容量或显式偏移表定位，片内pitch用该片的对齐长度。例如尾片仅1行时，其片内K物理尺寸为16；L0必须使用同一尾片layout读取。不能将这些分段拼接的存储重新解释成一张统一pitch的K1矩阵。

二者都可以实现有效K与物理对齐分离。复制层的pitch、局部起点、LoadData形状和源地址是一整套约定；修改其一后重新证明其余映射。零填充也必须按所选分形layout落在实际padding位置，不能覆盖相邻有效数据。

<a id="structure--padding-的责任与可移植边界"></a>
#### padding 的责任与可移植边界

首选利用**已核实的**ND→NZ/LoadData尾块处理和实际K接口，不在普通PV热路径上人为添加来自GM零源的多行 `srcDValue=0` 搬运。该模式在当前A2组合已有严重性能退化的实测；即使源码只有一次copy调用，也不能当作高效广播。这里不把其它API的零步长语义一概判为非法。[^pv-tail-evidence]

`MMAD.k=kc` 本身不是“所有物理padding都不会被读取”的证明。实现前记录选定copy/MMAD重载：是否自动补齐、补齐哪些维度与范围、是否需要显式设置、实际参与计算的K如何确定。如果硬件/组件会读取并参与对齐区域，P与V对应的padding都必须是已初始化有限值；`P=0` 不能抵消V中的NaN。不要用扩大逻辑K掩盖这些职责，也不要仅因一个随机用例通过就删除初始化。

若选定组件只接受对齐的计算K，单独引入 `Kexec=align_up(Kvalid,aK)`，保持原始 `Kvalid` 不变，并明确初始化 `[Kvalid,Kexec)` 的P/V和其真实读取footprint。可用容量充足、正行距的连续零区按目标layout批量填充，或该层支持的本地初始化。连续零源的最小可读元素跨度为 `(rows-1)*ldZero+cols`（非空时），实际DMA若扩大读取还须额外覆盖；不能只准备一行而使用多行正stride。先核对可写目标的分形地址，不照搬ND线性清零。补零受L1 free/ready事件保护，其代价计入完整主核时延。

例如仅用于推导的 `K1cap=256,K0cap=128,aK=16`：

| `Kvalid` | 各L1片内的实际K0序列 | 对应K物理尺寸 |
|---:|---|---|
| 1 / 15 / 16 / 17 | `[1]` / `[15]` / `[16]` / `[17]` | `[16]` / `[16]` / `[16]` / `[32]` |
| 127 / 128 / 129 | `[127]` / `[128]` / `[128,1]` | `[128]` / `[128]` / `[128,16]` |
| 255 / 256 / 257 | `[128,127]` / `[128,128]` / `[128,128];[1]` | `[128,128]` / `[128,128]` / `[128,128];[16]` |

表由上述 `min`/对齐公式得到，分号代表换L1片；不存在 `case_id`、`K==129`、`H==64` 的特判。实际允许的K范围和tile容量仍由当前普通SWA契约/资源表决定；257仅演示跨L1片公式，不能据此声称原N256 workspace已支持该长度。K切分是同一次PV的累加，不额外分割softmax，也不改变P低精度舍入点。验收见[有效K与物理尾块检查](validation.md#pipeline-validation--pv-tail-validation)。

<a id="structure--2-两种-cube-owner-组织的选择"></a>
### 2. 两种 Cube owner 组织的选择

**A：共享 persistent owner、typed GEMM view、K 双缓冲。** 从 SDK BlockMmad 类型取得 CopyGmToL1A/B、CopyL1ToL0A/B、TileMmad、CopyL0CToGm 和对应 layout 类型，自有 block 组合这些 CATLASS tile primitive；一个 owner 分配并初始化事件。QK/PV view 不构造第二份有事件所有权的实例。首版可采用本方案打通正确的融合流水；容量算例与完整 K 循环按本页公式和当前输入重新推导，不能照搬某个固定shape的N192/BF16常量。

**B：在 A 的基础上保留 Q/P 的 L1 数据，跨输出分块复用。** C1 的同一 M 范围、不同 N block 共用 Q；C2 的同一 M/N 范围、不同输出 d0 共用 P。Q/P各阶段只在第一个输出块装载，最后一个使用者完成 L1→L0 后才释放对应槽。是否切 D/K256 与 L0块大小共同设计。不能只在外面保留 Resource，然后每次调用标准 BlockMmad，误以为它会自动识别相同 GM 地址并不再加载 A。

当前 SDK `block_mmad_pingpong.hpp` 每次 `operator()` 都加载首 A/B tile，内部预取下一 K tile；L1、L0A、L0B是两槽，但 L0C是一个张量。它提供局部预取，不自动提供 Q/P跨调用常驻，也不自动提供跨 C/V task 流水。[^structure-block-mmad-source]

一种可评估的 L1划分是总共7块，每块64 KiB：QP区4块、KV区3块，合计448 KiB，剩余空间仍按SDK实际用途核对。Q M128/D512总128 KiB，可在两个D256片中占两块；P M128/N<=256最多64 KiB。QP槽分配必须同时考虑 C1当前task 与 C2前一task，按具体使用图定义，不能凭4/3两个数字写 `%4`/`%3`。KV三块可以为流水装载提供周转；这并不意味着所有 N256/D512 KV 及其第二份副本都常驻。不要宣称 C1 到 C2 永远不重读 KV，当前组织可以在 C2重新加载 V。

若选择这套7块结构，下面是可重新实现的一种明确映射，而非要求复刻某份源码的迭代变量：

| 区域 | 字节范围 | 逻辑用途 |
|---|---|---|
| QP pair0 | `[0,131072)`，含两个64KiB块 | 一次C1的Q两个K256片，或一次C2的P片 |
| QP pair1 | `[131072,262144)`，含两个64KiB块 | 与pair0轮转，避免紧随的另一阶段过早等待旧MTE1读取 |
| KV slot0/1/2 | `[262144,458752)`，每块65536字节 | QK的K256×N128，或PV的K<=256×Dv128低精度B |

设`aStageOrdinal`仅在**实际发出非空C1或C2矩阵阶段**后加1，该阶段使用`pair=aStageOrdinal%2`，块编号为`2*pair+kPiece`；首使用前只等待将要写的块free。C1的kPiece=0/1都装Q，并跨本次C1的所有N128输出块保留；每片在最后N块的最后一次L1→L0读取后分别释放。C2在Nphys<=256时只需kPiece0存P，跨四个Dv128输出块保留，最后D块最后一次读取后释放；未使用的kPiece1既不消费free也不发ready。零计算任务不能虚构Set/Wait。若扩大到Nphys>256，就不适用这个P单片分配，必须另算。

KV独立使用`bTileOrdinal%3`，每加载一个实际B片推进一次；不因换C1/C2或GM任务槽而归零。下一片装入下一空闲槽，ready在全部页段/补齐copy之后；该片被所有M子块最后一次L1→L0读取完成后释放。该映射当前约束每阶段M<=128，因此只有一个M子块；若将任务扩为M256，B可以供两个M子块复用，但free必须推迟到第二个M子块，QP与C累加器也要重新设计。**GM任务槽、QP阶段pair、KV片槽是三个不同计数器，不能共用`task%2`。**

两槽L0A/B仍独立按实际MMAD推进，不随L1槽数变成三槽。可将K0从64评估为128：M128×K128×2=32KiB的L0A、K128×N128×2=32KiB的L0B，各两槽恰为64KiB。QK D512需4次K128 MMAD；K256 PV需2次。减少指令次数是候选收益，actual K尾/分形copy、参数限制和设备结果均需重验，不能只改模板常量。

L0C若采用两个64KiB输出槽，则每个物理M/N块至多128×128 FP32。设`cOutputOrdinal`对每个完整MN结果推进一次：从首K累加到末K始终使用`c[cOutputOrdinal%2]`，末K与FIX成对完成后才推进。不能按K片翻转C，否则把一个结果拆到两个累加器；也不能给M128×Dv256的128KiB结果仍分配两个64KiB槽。unitFlag/显式事件的生产消费必须针对实际C地址及当前SDK建立，增加第二槽不会自动为copy/MMAD补同步。每个GM输出块仍使用整行X的Nphys或U的D512作为dst stride，不能用Nblock128替代。

<a id="structure--可选的m256逻辑任务物理cube块仍m128"></a>
#### 可选的M256逻辑任务：物理Cube块仍M128

长prefill若M128已正确而吞吐仍受任务/握手/KV重读限制，可比较M256逻辑任务。它是一套独立资源映射，**不能直接把上面的单M块QP阶段pair公式放大两倍**。完整head组H>=2且`T=256/H`时union最多`128+T-1<=255`，仍可Nphys256；H1/T256则最多383列，不属于此N256方案，继续M128或单独设计更宽/流式路径。短decode任务不足时增大M可能更慢，必须按实际任务数选择。

一个可核算的候选是：每task两块M128，Q占QP四个64KiB块，映射`block=2*mBlock+k256Piece`；本C1全部N输出完成才释放。P在N<=256时每M块64KiB，使用QP块0和2，各跨四个Dv128保留，其他QP块没有本阶段ready。此时QP空间用来保存同一task的两个M块，不能同时宣称还存了下一task的完整Q。每侧AIV负责128行，按r32或其它已核算子块处理，invL每task槽也扩大为128行。GM X/P/U尺寸及workspace翻倍，metadata任务步长改为256/H，各侧仍只在完整task自己的P或O写出后发一次外层通知。

为了让两个M块共享B，Cube可采用`output N/D block → K1片 → K0片 → 两M块`的嵌套：一份B的L1片供两M块使用；同一个K0片的L0B装载一次，供M块0和1的MMAD读，**第二个MMAD完成后才释放L0B**。L0A分别装两M块的A；L0C用两个64KiB槽固定对应当前MN结果的mBlock0/1，跨所有K片保持各自累加，最后K后分别FIX。下一个N/D输出块复用这两个C槽前满足各自FIX完成依赖。B的L1free在两个M块全部K0读取完成后发，Q/P的L1free仍须等待该A块的最后N/D输出消费者。

这可以减少每两块M独立任务带来的外层通知及B重复装载，但不会减少真实QK/PV运算量；Vector每侧工作增加、C槽占满、任务数量减半，可能降低短任务并行度。设计要列出L0B的两个reader和C的固定归属，重做协议/容量检查及设备精度和性能。不允许让一个M128 BlockMmad直接吃M256，或只更改metadata而保持AIV行区间、GM行距和分母容量不变。该候选并未随本页发布而自动获得设备验证。

<a id="structure--3-owner-内部-k-预取的事件协议"></a>
### 3. owner 内部 K 预取的事件协议

以 L1 A/B 两槽、L0 A/B两槽为例。对每类事件使用独立范围，创建一次、最终销毁一次；**不得在 QK/PV view切换时把槽号重置为0**。同数字不同 HardEvent方向是不同事件，同方向相同数字不可由两个owner反复初始化。

| 数据边/复用 | 生产完成通知 | 消费/覆盖时等待 | 初始 token |
|---|---|---|---|
| GM→L1 A/B | `MTE2_MTE1`，装载后 | 首次 L1→L0读取前 | 无 |
| L1槽可重新写 | `MTE1_MTE2`，该槽最后一次 L1→L0读取后 | 下一 GM→L1覆盖前 | 每槽1 |
| L1→L0 A/B | `MTE1_M`，A/B载入后 | 对应 MMAD前 | 无 |
| L0A/B可重新写 | `M_MTE1`，对应 MMAD消费后 | 下次L1→L0覆盖前 | 每槽1 |
| L0C MMAD→FIX及再次累加 | 当前SDK完整 unitFlag 或 FIX_M/M_FIX方案 | 对应C消费者/覆盖者 | 按所选方案初始化，不能混用 |

K循环动作：先预取首片→若还有下一片则先向另一空闲L1槽预取→当前片按K0载入L0并MMAD→最后L1读取后返还该L1free→MMAD消费后返还L0free→更新所有slot parity→最终FIX写出。定义 `off` 为当前L1片起点、`count=min(K1,Ktotal-off)` 为整段长度、`kk` 为段内L0片起点、`kc=min(K0,count-kk)` 为本次MMAD的K长度。首K清零仅在 `off==0 && kk==0`；末K条件是 `off+kk+kc==Ktotal`，不能把整段count加在kk之后，也不能每个L1段都重新清零。

在现有 `MmadAtlasA2Pingpong<true>` 中使用SDK配套unitFlag：非末MMAD为0b10，末MMAD及对应L0C→GM为0b11；检查真实头文件后照其完整依赖实现。若选false，使用完整FIX_M/M_FIX事件链。两种都能正确，不能混用，也不能只保留MMAD上的标记却省掉FIX copy的匹配字段。

Q/P常驻优化改变了“最后L1读取”的含义：不是当前K片第一次读完，而是所有N/D输出块的最后一次读取结束。需要独立的 retained/free 状态；若提前发L1free，后续重用的Q/P会被新KV或任务覆盖。设备上常见症状是第一N/D块正确、第二块错误。

<a id="structure--七块l1与双l0c的完整事件分配示例"></a>
#### 七块L1与双L0C的完整事件分配示例

对上文QP4块/KV3块方案，先按当前SDK确认每个HardEvent方向支持的事件ID范围，再给同一owner分配下表；这里最大使用ID6。不要沿用两槽BlockMmad的A0/1、B2/3映射，却把L1扩成7块而不更新事件表。

| 对象 | free方向与ID | ready方向与ID | 本owner构造时 |
|---|---|---|---|
| QP物理块0..3 | MTE1_MTE2，0..3 | MTE2_MTE1，0..3 | 每块仅free Set一次 |
| KV物理槽0..2 | MTE1_MTE2，4..6 | MTE2_MTE1，4..6 | 每槽仅free Set一次 |
| L0A两槽 | M_MTE1，0..1 | 见下方成对MTE1_M | 每槽仅free Set一次 |
| L0B两槽 | M_MTE1，2..3 | 见下方成对MTE1_M | 每槽仅free Set一次 |
| L0C两个64KiB结果槽（显式事件方案） | FIX_M，0..1 | M_FIX，0..1 | 每槽仅free Set一次 |

上述不同方向中的相同ID是不同事件；AIC的HardEvent、每个AIV的HardEvent、外层mode2跨核flag也分别记账。禁止同一AIC上的另一个helper私自用`MTE1_MTE2/0`当临时Fence，与QP块0冲突。若API内部会分配事件，也要核对其所有权，不把表中ID当全局自动预留。

一片保留Q/P的生命周期严格是：`Wait free → MTE2写入 → Set ready → 首次MTE1读取前Wait ready → 多个N/D消费者读取 → 最后MTE1读取后Set free`。中间消费者没有新的MTE2生产，所以不得重复Wait同一次ready，也不得为了“配平”重复Set ready。未用QP块的初始free仍保留，析构消耗它即可；不产生其ready。KV每次实际装载都经历一套完整free/ready，跨阶段保留自己的轮转计数。

L0 A/B可以使用成对ready：等待将被改写的A/B槽free，发出本次L1→L0 copy后`Set MTE1_M/0`，对应MMAD前`Wait MTE1_M/0`；发出MMAD之后，分别在M管线发A/B的free。若M256共享一次L0B给两个MMAD，则第一MMAD后只释放它自己的L0A，L0B在第二个MMAD后释放。每个成对ready只有一次Set和Wait，不额外预置token。

双L0C最容易核验的候选是采用当前SDK的**关闭unitFlag**的tile类型，并显式扩展SDK单槽事件链：对结果槽c，在这个MN结果首K前`Wait FIX_M/c`，所有K片向同一地址MMAD（首K清零、后续累加、unitFlag=0）；末K后`Set M_FIX/c → Wait M_FIX/c → FIX copy（不带unitFlag）→ Set FIX_M/c`。只在该C槽下一次首K覆写时Wait它的free，不在每个K片或FIX之后立即消费free。这样另一C槽可以开始独立结果，仍由M/FIX各自顺序与事件约束真实重叠。若采用unitFlag方案则使用其完整约定，不能再叠加这套显式C ready/free而缺少一方生产者。

单M块时C按完整MN输出推进；M256共享B时C0/C1固定对应当前输出块的两个M128结果，分别在各自首K前等free、末K后FIX。退出顺序是：所有数学与外层U_FREE完成→等待L1七项free（QP4加KV3）、L0A/B四项free、显式C两个free→owner结束。每项恰好消费当前保留的一枚free，包括从未使用的初始化free；所有ready应已由真实消费者消费。L=0也按已构造资源的初始状态排空，或整组在构造前一致跳过，不能构造一半后直接退出。

析构等待全部 L1/L0自由事件和当前C方案的保留token，位置在整个core任务循环及跨核U_FREE排空之后。标准BlockMmad在task作用域构造/析构会反复排空；即便数值正确也会损失跨任务重叠。

<a id="structure--4-vector-ub-存活图与行块计算"></a>
### 4. Vector UB 存活图与行块计算

每侧 AIV用行组 R及列Nphys做批量运算，不用每行重建 TPipe、复制sink、调用一轮 scalar SetValue掩码。一个 M128/N256、R64 的保守 UB 起点：

| UB区 | 最大字节 | 生命期与复用条件 |
|---|---:|---|
| FP32主矩阵 R×max(Nphys,Dv256) | 65536 | V1 score/exp 与 V2输出stripe复用，前一V/MTE3读完才能覆写 |
| 低精度矩阵 R×max(Nphys,Dv256) | 32768 | P cast/output cast复用，前一MTE3完成才能覆写 |
| 归约中间 R×64 FP32 | 16384 | rowmax/rowsum串行复用，结果保存在独立行向量 |
| row向量、两槽invL/m/l、广播临时区 | 按R对齐计算 | 不能与下一task临时归约别名 |
| mask位图/索引、sink常驻、副本、API临时区 | 实际核算 | 整体与API保留区一起小于设备可用UB |

不能再无条件为整张FP32矩阵复制一份64KiB以做双缓冲，192KiB UB会超出或无足够scratch。要双缓冲时缩小行块r，例如16/32，并对两个输入/输出子槽分别核算；或在同一UB区域由事件约束复用。Vector程序可按行块切V1、按Dstripe切V2，跨核ready在完整task的本侧P/U数据满足协议后发，不能处理第一块就通知全P已就绪。

推荐V1向量步骤：

1. 一次/少数连续DMA加载本侧行块X，等待MTE2_V。
2. 向量Muls scale，按每query窗口生成共享列mask再扩展到其head行；不可对每head逐列scalar扫描256次。列索引可片上常驻，随task只更新左右边界。
3. 使用行归约得到m；sink表在launch开始按合法DMA粒度读入UB，按head周期选取并参与max，不为每行GM GetValue。
4. Brcb/其他合法广播使每行m能被批量减去，然后Exp；无效列显式清零，再归约得到FP32 sum，加sink项。
5. 每行一次FP32倒数，保存在当前slot；计算需要的LSE或保存m/l直到V2。P由未除分母的E cast为输入dtype。
6. P按真实Nphys行距写GM。全部本侧P写完后由MTE3发P_READY。零行侧跳计算但发协议通知。

N不是64整数倍的实际有效列时，物理对齐列仍显式mask；M尾、无效prefix、全空行单独处理。有限负哨兵相对行max可能不是负无穷，所以exp后清零无效E才是可靠分母排除，详见performance中的一般SWA极值例。

行广播实现必须核对每条API的mask模式、repeat宽度、32B block stride、256B repeat和每次repeat的实际覆盖。以FP32 mask=8为例，一个活跃block为8元素，repStride=1意味着下一次地址前进8个FP32，不是前进headDim；`src1RepStride=0`反复使用同一组8个值。需要每行一个标量时先构造正确的广播布局。repeatTimes若为uint8不能直接传576；拆成合法子段并根据**该操作的repeat元素步长**移动地址，不能按mask值推断所有布局。一般mask可能跨多个32B块，blkStride也参与寻址。相关API语义查通用SDK知识，本算子必须给出R/N/Dv映射后的实际参数。

对本页常用的FP32 `X[R,Nphys]`、`Nphys=256`、`R<=64`，可具体按列stripe64、repeat按行组织。先把每行标量a[r]用合法Brcb等操作构造成 `A8[r,0..7]=a[r]`。每个列stripe起点c取0/64/128/192，使用连续mask=64、repeat=R：

```text
dstBlkStride=1, src0BlkStride=1, src1BlkStride=0
dstRepStride=Nphys/8=32, src0RepStride=32, src1RepStride=1
dst = X[c], src0 = X[c], src1 = A8[0]
```

第r个repeat中，X前进r*256元素；src1前进r*8元素，8个活跃block都重复读取本行同一个已扩成8份的标量。因此4次Sub可批量完成所有R行减rowmax，4次Mul可批量完成N256或Dv256的逐行乘invL。若物理列数为192则repStride=24、三个stripe；末stripe mask按实际需要且padding另处理。这里`src1BlkStride=0`只复用一个32B block，**不会自动把未扩展的a[r]广播成标量**。FP16/BF16的block元素数不同，不照抄FP32参数；本算法的rowmax/invL运算仍在FP32。

以FP32行归约为例，可以先把Nphys按64列折叠成 `Fold[R,64]`（N256需要3次逐元素Max或Add，每次同时处理R*64元素），再按64元素每行做WholeReduceMax/Sum得到连续行向量。max和sum复用同一Fold但在不同时间；sum只归约已显式清零无效列的E。CANN兼容重载的dstRepStride单位及ReduceOrder按实际头文件核对，不能把二元指令的参数顺序套到Reduce。此方案减少逐行scalar往返，但仍需保护Fold和行结果的存活期。

V2按行块和Dstripe读取U，广播对应slot的invL做Mul，cast并以O真实stride写回。每行不重复计算1/l，更不能对512个元素做512次除法。最终LSE和O尾部只写合法元素，短LSE不通过越界32B写入凑对齐。

<a id="structure--5-可直接翻译为实现的-ub-子块流水"></a>
### 5. 可直接翻译为实现的 UB 子块流水

上面的R64单矩阵是容量保守的正确性起点，并不自动具有MTE2/V/MTE3重叠。若每次`GM→UB`后等输入、每次`UB→GM`后又立即等输出释放，同一个UB区域可以正确复用，但不能在验收表写成“Vector双缓冲流水已实现”。下面给出普通SWA可选的两输入、两输出组织；实际收益仍以测量确认。

对每侧R64行，把V1拆成两个r32行块，Nphys256；V2拆成两个r32行块×两个Dv256列块。输入FP32两槽各`32*256*4=32768`字节，输出FP16/BF16两槽各`32*256*2=16384`字节，共96KiB；共用归约Fold`32*64*4=8192`字节。另加当前rowmax/sum、每任务槽64行invL、Brcb区、sink表、LSE区及SDK scratch，再核对实际UB上限。**invL是任务槽的持久状态，不放进轮转输入槽或临时Fold。** 输入、输出子槽的编号也不是GM任务槽编号。

每种HardEvent的子槽编号分别独占；这里以符号`IN_FREE[s]`等表示，不与实际固定编号混淆：

| 子槽所有权边 | 事件与发出位置 | 消费位置 | 初始/最终计数 |
|---|---|---|---|
| 输入UB可写 | `V_MTE2(IN_FREE[s])`：所有V读取完成，含最后Cast读取 | 对该槽下一次GM→UB之前 | 每槽初始化1次；最终owner排空消费剩余1次 |
| 输入可计算 | `MTE2_V(IN_READY[s])`：该行/列块的DMA之后 | 本块第一条V指令之前 | 初始0；每个实际加载子块恰好一Set一Wait |
| 输出UB可覆盖 | `MTE3_V(OUT_FREE[s])`：该槽最后一笔UB→GM之后 | 下次Cast/Vector写同槽之前 | 每槽初始化1次；最终排空消费剩余1次 |
| 输出可搬运 | `V_MTE3(OUT_READY[s])`：Cast及本块所有V写完 | 本块UB→GM之前 | 初始0；每个实际输出子块一Set一Wait |

输入输出分开，输入free无需等待MTE3：它的最后reader是V；输出free必须来自MTE3。若合并或别名两区域，则这套表不再成立，需增加MTE3→MTE2约束。仅在owner创建时预置free，不能每次V1/V2调用都重新Set同一token。

可用以下提交顺序处理一个阶段的子块列表（V1的列表按行块；V2按行块与Dstripe）。`load(k)`包含等待对应输入free、合法范围DMA、发input-ready；`store(k)`包含等output-ready、合法范围DMA、发output-free：

```text
if 子块数 > 0: load(0)
for k in [0, 子块数):
  if k+1 < 子块数: load(k+1)       # 使用另一输入槽，先排队下一块搬运
  wait IN_READY[k%2]
  V计算当前块，输出/保留本块行状态
  wait OUT_FREE[k%2]
  Cast当前结果到独立输出槽[k%2]
  set IN_FREE[k%2] from V         # 最后一次读取输入之后
  set OUT_READY[k%2] from V
  store(k)
  # 此处不立即wait刚发出的OUT_FREE，推迟到同槽复用/最终排空
```

这给MTE2加载下一输入、V处理当前块、MTE3写上一输出留下重叠机会；是否真正重叠还取决于指令排队和资源。最初两次load各消费一次预置free；再次回绕等待上一reader；最后不加载不存在的块。零子块不产生ready，仍保留初始free供owner统一排空。V1和V2共用owner时，子槽parity可持续计数，也可明确回到已释放槽，但不得只重置编号并遗失尚未消费的事件。

V1子块写`P[taskSlot,row0:row0+r,Nphys]`，并保存对应行的invL到`invL[taskSlot,row0:row0+r]`。两个行块都已排队写完后，才从MTE3发一次本任务`P_READY`；不能每个子块发一次同名外层ready，除非连Cube消费粒度和计数一起改写。V2子块读取`U[taskSlot,row0:row0+r,d0:d0+Dv]`，只使用同taskSlot、同行范围的invL；全部O输出排队后才从MTE3发一次`U_FREE`。短LSE有独立输出区域/事件时可以在P通知后继续写出，LSE完成仍属于该次算子完成条件；若与P/O输出槽复用，必须服从上表的output-free。

<a id="structure--lse异步写出的单独生命期"></a>
#### LSE异步写出的单独生命期

`invL[taskSlot]`的最后reader是V2的Vector广播，`lse[taskSlot]`若直接作为GM写出的源，其最后reader却是MTE3。两者即使同属一代行状态，也不能共享“V2函数已经返回”这个释放条件。特别是`V2(i)`排队写LSE后，下一轮`V1(i+2)`会重用相同taskSlot；在新V1写LSE之前必须确认旧MTE3读取结束。

一种完整方案为两份独立LSE UB和两组free/ready。以下以启用LSE且本侧有输出行的任务为单位，子块只在首次/最后一次执行相应动作：

| 位置 | 动作 | 计数与完成含义 |
|---|---|---|
| VectorOwner构造 | 对两个LSE槽各Set一次`MTE3_V(LSE_FREE[s])` | 只预置free，不预置ready |
| V1第一次写该任务LSE之前 | Wait `LSE_FREE[taskSlot]` | 旧一代MTE3不再读同一UB；一个任务只等一次，非每行块都等 |
| V1各子块 | 将LSE写到当前槽的对应行区间 | 不与invL、归约临时区或别代输出别名 |
| V2输出该任务LSE | Set/Wait `V_MTE3(LSE_READY[taskSlot])`，随后精确长度UB→GM | ready覆盖所有LSE Vector写入，不能短向量越界写32B |
| 该笔UB→GM之后 | Set `MTE3_V(LSE_FREE[taskSlot])` | 返还free；不立即Wait，留到下一同槽写入或退出 |
| VectorOwner排空 | 两槽各Wait一次当前free | 消费最后返还的free或从未使用槽的初始free |

例如输出矩阵已使用`MTE3_V/0,1`及`V_MTE3/0,1`，LSE可在确认SDK范围且无其它owner占用后使用相同方向的2,3。编号属于**本AIV上的同方向事件**，不是全局自动保留。LSE关闭或本侧零行时跳过本任务的LSE Wait/Set，保留构造free供退出；外层P_READY/U_FREE仍按跨核协议发送。若在V1就写LSE，或先复制到受OUT_FREE保护的独立输出缓冲，再由MTE3读取，也可成立，但必须相应改写唯一writer、最后reader、free位置和计数，不能重复写出/重复返还。

`CrossCoreSetFlag<2,PIPE_MTE3>(U_FREE)`在MTE3上排在写出之后，建立的是接收AIC的跨核边；它不会让发出通知的AIV的V/S管线自动等完自己的MTE3。接收AIC尚未走到同槽覆盖、AIV却已开始新V1的交错完全可能发生。因此不能用这个外层发送代替LSE的本地`MTE3_V`回收。O/P矩阵若使用其它输出UB，等待它们的OUT_FREE也不能保护提前被改写的LSE UB。

逐槽记录 `LSE_FREE -> V1 write -> LSE_READY -> MTE3 copy -> LSE_FREE` 的生产/消费链及初始、稳态、尾部计数。free 在首次复用方案中预置；ready 则是运行时由真实 Vector 写入产生，`Set V_MTE3 -> Wait V_MTE3` 无需构造初始化且是合法的 V→MTE3 依赖。它只能证明本次输出源已经写好，不能证明前一次 MTE3 已读完。审查复用时查 copy 后的 `MTE3_V` 返还和下一次 V1 覆写前的消费，不把 ready 与 free 混淆。LSE 关闭时跳过该链但仍排空其它事件；本侧零行时不产生 LSE copy，保留对应初始 free 供退出。

最后r不足32或某侧无有效行时，DMA只搬合法行，Vector对物理对齐行使用已初始化有限中性状态。不能从未写的输入UB尾行归约出NaN，也不能为了凑满32行写越输出。全无效但负责padding清零的块仍有实际输出工作，与完全不拥有行的零子块不同。外层零行AIV仍完成跨核必要通知，具体见协议页。

<a id="structure--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

<a id="structure--1-aiv核内事件不能由crosscore替代"></a>
### 1. AIV核内事件不能由CrossCore替代

| UB边 | 要等待的完成 | 对应操作 |
|---|---|---|
| 复用旧输入UB给MTE2 | 旧V读完：V_MTE2；若复用区曾供MTE3读取，还需MTE3_MTE2 | 之后才能下一GM→UB |
| V读取新输入 | MTE2_V | 之后才能scale/softmax/output |
| MTE3读取cast结果 | V_MTE3 | 之后才能UB→GM |
| V覆盖旧输出UB | MTE3_V | 之前的P/O写出完成 |
| scalar读Vector归约结果 | V_S | 若不得不用GetValue，先等待；主热路径尽量批量留在Vector |
| Vector读scalar写入的UB | S_V | 不可只看源代码先后 |

同一pipeline内的依赖按API规则使用PipeBarrier；不同pipeline按HardEvent生产消费。`SetFlag`应发在真实生产pipe，`WaitFlag`在真实消费者发起前。每个UB ping-pong槽的event互不冲突。记录第一轮free如何初始化、末轮如何消费，不能在每行盲目Set/Wait所有方向造成全串行。

<a id="structure--2-pa-按页段装入l1"></a>
### 2. PA 按页段装入L1

给逻辑起点 k、剩余count：`logical_page=k/page_size`，`in_page=k%page_size`，`ncopy=min(remaining,page_size-in_page)`；读当前batch页表求物理page，用真实page/token/head/dim stride形成源地址，把这段放到tile的对应行/列位置，再推进k和tile偏移。重复页号合法，page_size和KV有效长度来自本次调用。

QK的目标是对应K的分形L1布局，PV的目标是V的分形L1布局；不能仅因物理source都是KV就共用错误的ND→NZ参数。使用SDK当前可验证的TileCopy/DataCopyPad/ND→NZ能力组成KvAccessor；跨页是多段copy，ready在整片完成后发。tail的补齐区置有限零，避免P=0乘NaN。

若选择pack→GM的替代路径，按协议页扩展PACK_READY、有限两槽及双AIV列分工；额外搬运与完整执行时间均入账，不伪装成免费host适配。

<a id="structure--3-拆分-s2-时的在线状态"></a>
### 3. 拆分 S2 时的在线状态

对同一行多个KV tile，记录全局旧m、l、u。当前tile最大值包含其有效score；sink只初始化/注入一次。更新：

```text
m_new = max(m_old, tile_max)
alpha = exp(m_old-m_new)             # 空初态显式处理中性值
E_tile = masked_exp(score-m_new)
l_new = alpha*l_old + sum(E_tile)
delta = cast_input(E_tile) @ V_tile
u_new = alpha*u_old + delta
```

alpha必须和产生delta的tile绑定到V2，不能让下一V1覆盖。m/l在V1更新，u在V2更新，二者属于不同阶段但同一业务task；first/last由TaskInfo明确标记，不能以当前core最后一个循环猜任务结束。最终O=u/l，LSE=m+log(l+1e-10)，epsilon遵从原golden。跨query task开始必须重置状态；跨S2 tile不能重置或重复sink。

**在线状态公式不自动保持本任务的低精度P舍入契约。** 实数/全FP32累加下的重新缩放与`cast_input`不交换：通常`alpha*cast_input(E_old) != cast_input(alpha*E_old)`。因此把旧tile先按旧m转低精度并做PV、随后缩放u，与原golden使用全窗口最终m后一次cast P不等价。普通SWA整union可放入一次V1时保留一次softmax/cast路径。若确需拆S2，必须另行设计符合已冻结舍入顺序的方案，例如先确定全窗口m，再按这个共同m形成各tile的E、FP32分母和低精度P，并核对分母/PV归约顺序；不能仅引用上面在线递推就宣称数值契约不变。若任务明确允许一种不同舍入算法，也须单独记录获准契约并按原精度验收，不能由生成者自行放宽。

如果union最多255且一次V1完整处理，则没有old-u更新分支是合理特化，不因此宣告计算缺失；但仍必须实现跨**不同work**的C/V重叠。不能用“窗口短不用online”来取消任务流水。

<a id="model--代码模式"></a>

## 三 actor 状态模型的构造与检查

<a id="model--参考程序与状态转移"></a>
## 参考程序与状态转移

actor 为 AIC、AIV0、AIV1，允许两侧 Vector 以任意有限相对延迟推进。任务 t 使用
`slot=t mod 2,generation=floor(t/2)`。每个 family/slot/sender/receiver 一条概念 FIFO 通道，
AIC 发送分别到两 AIV，AIC 等待需要两 AIV 的通知都到齐；task/generation 仅为诊断标签，
不能声称实际硬件通知保存这些载荷。

任务数量为 L，参考程序用 i=0..L 的启动/稳态/排空迭代：

| actor | i<L 的新任务 | i>0 的旧任务 j=i−1 |
| --- | --- | --- |
| AIC | 可选 waitPack(i)，C1(i)，setX(i) | waitP(j)；j≥2 时 waitFree(j−2)；C2(j)，setU(j) |
| 每个 AIV | 可选 pack(i)/setPack(i)，waitX(i)，V1(i)，setP(i) | waitU(j)，V2(j)，setFree(j) |

AIC 循环后消费 j=max(0,L−2)..L−1 的剩余 free。首两次 U 使用不等待 free，初始也没有预发 free credit。
零行 AIV 保留协议指令。X/P/U、可选 pack 和各 AIV 行状态是互不混淆的双槽区域。

原子阶段模型假定 C1/C2/V1/V2/pack 的全部读写在一步内完成。状态可用三个程序计数器表示，
已发/已消费的通知由各 actor 进度导出。广度优先枚举每个可达状态中所有可执行 actor 的下一条指令，
记录父状态以构造反例路径。wait 只有所需 FIFO 对应项已到达才可执行；到达 task 不匹配就是代次错误。
未结束但没有可执行动作才是死锁；不证明无限调度饥饿下的活性。

<a id="model--必须检查的读写不变量"></a>
## 必须检查的读写不变量

| 动作 | 必须已经满足的条件 |
| --- | --- |
| C1(t) 覆盖 X | t<2，或两侧 V1(t−2) 已完成 |
| V1(t) 读取 X | 已 waitX(t)，C1(t) 完成而 C1(t+2) 尚未覆盖 |
| V1(t) 写 P/行状态 | t<2，或 C2(t−2) 与本侧 V2(t−2) 已完成 |
| C2(t) 读取 P | 已 waitP(t)，两侧 V1(t) 完成且两侧 V1(t+2) 都未覆盖 |
| C2(t) 写 U | t<2，或两侧 V2(t−2) 已完成 |
| V2(t) 读取 U | 已 waitU(t)，C2(t) 完成而 C2(t+2) 尚未覆盖 |
| V2(t) 读取行状态 | 本侧 V1(t) 完成、本侧 V1(t+2) 未覆盖 |
| pack(t) 写槽 | t<2，或 C2(t−2) 已完成 |
| C1/C2 读取 pack | 已 waitPack(t)，两侧 pack(t) 完成且任一侧 pack(t+2) 都未覆盖 |

每次转移检查不变量；终态还必须所有通知发送数=消费数。保存首个各类错误的状态、下一操作和完整 witness，
以及死锁、终态残余、最大待消费 token 数、状态/转移数。达到状态上限而未找到具体错误时为 INCONCLUSIVE，
不是 PASS。只有搜索穷尽、无不变量错误、无死锁、完整终态成立且无残留通知才可报告参考协议 PASS。

<a id="model--负对照及抽象边界"></a>
### 负对照及抽象边界

| 变体 | 目的与预期 |
| --- | --- |
| none | 正常双槽，复用时等待 free，最后排空 |
| free_at_tail_only | 保留 free 发送，但仅尾部消费全部 free，隔离内存覆盖与通知残余；原子模型可通过 |
| drop_u_free | 同时删除所有 free 发送和等待；原子 V2(t)→V1(t+2)→P_READY(t+2)→C2(t+2) 链可保护 U，模型通过是合法结果 |
| omit_aiv1_p0 | 漏 AIV1 首个 P_READY，应发现死锁 |
| wrong_x_slot | 一侧首个 X wait 用相反 slot，应发现代次不匹配 |
| pack_after_x | pack/setPack 放到 waitX 后，应发现循环等待 |
| omit_tail_free | 保留发送与复用等待，省末尾 free 消费，应发现终态残余 |
| skip_x_wait | 一侧首个 V1 不等待 X，应发现无 ready 读取 |

自检矩阵用 L=0/1/2/3/7、pack 开关、none/free_at_tail_only/drop_u_free，共30组正常配置，
五种负对照取 L=3；另设极小搜索上限检查 INCONCLUSIVE。原子参考程序对应的可达状态数可作实现核对：

| L | 不带 pack | 带 pack |
| --- | ---: | ---: |
| 0 | 1 | 1 |
| 1 | 37 | 46 |
| 2 | 262 | 244 |
| 3 | 625 | 391 |
| 7 | 2689 | 979 |

这些数仅适用于上表程序与状态抽象；改变协议后应重建模型，不能为了符合数量伪造转移或删反例。
探索上限可从200000起按规模调整，达到上限要如实记未穷尽。

# 约束

<a id="protocol--约束"></a>

## 双槽跨任务握手

- 最终长 prefill 路径是混合核、有限槽、跨任务交错。单任务自然退化不必伪造重叠；调试串行开关不是性能交付。
- 上述伪码中的阶段完成由实际 PIPE 依赖实现。不得用空通知、只在一侧通知、host 同步或读取旧输出替代。
- 改 task 顺序、slot 数、Dv、AIV 行映射、资源 owner、PA 路径后，重新证明复用和 token 数，并重验设备精度。保留浮点舍入顺序。
- 不需要读取手写业务源代码来实现本页；合法 API 输入是冻结的 CANN/CATLASS SDK，数学输入是原 golden 和接口。

<a id="structure--约束"></a>

## 混合核结构与片上复用

- 普通 SWA阶段是否实际实现以函数体、调用图、缓冲/事件与设备证据判断；不凭文件名、API名称、注释或构造了两个Tensor判断。
- 所有资源按所选编译变体真实峰值重算；不把L1、L0、UB容量互相挪用，不拿A5路径代替A2。
- 使用CATLASS矩阵/拷贝tile组件组合SWA block是合法专用实现；不能用逐元素手写矩阵乘绕过组件。资源优化不得改变冻结的低精度P舍入点。
- 以上结构是设计输入。没有本轮完整ATK及5+5性能不能标verified或声称0.8x。

<a id="model--约束"></a>

## 三 actor 状态模型的构造与检查

真实异步 DMA/MMAD/FIX 不能直接套用原子阶段完成假设。若要验证重叠，必须把 issue/complete 分开，
显式保留未完成 reader/writer；helper 返回不能当作硬件完成。具体 descriptor ring、地址别名、物理事件号、
数值、UB/L0 容量和编译器行为也不在原子参考模型内，改变这些内容必须扩展状态与检查。
零行侧的通知正确不证明零填充或访问范围正确；drop_u_free 通过不证明硬件可删除相应握手。

# 失败表现

<a id="protocol--失败表现"></a>

## 双槽跨任务握手

| 现象 | 优先检查 |
|---|---|
| 首个非空 task 就挂起 | mixed kernel 类型、逻辑核映射、sync base、mode2 两侧参与、PIPE 生产者是否真实到达 |
| L1/L2 正常，L3 开始错 | slot/generation、旧 U free、P 的跨 GEMM 顺序、invL 被新任务覆盖 |
| 第一次正常，第二次挂起 | 外层 free 多/少等、owner 内部 token 未排空、最后发送未完成、同步基址或 mask 恢复 |
| 仅小 M/尾块挂起 | 零行 AIV 提前 return，或只一侧省略 P_READY/U_FREE |
| PA 首轮挂起 | pack 被放在 score wait 之后，或某侧用 queryRows==0 跳过 KV pack |
| 有两槽但 Cube/Vector 无重叠 | 每 task 内同步到底；资源对象每 task 析构；free 放得过早；不同阶段仍是独立 launch |

<a id="structure--失败表现"></a>

## 混合核结构与片上复用

| 现象 | 优先检查 |
|---|---|
| 仅第二N块或第二D半边错 | Q/P L1被提前释放、C输出行距误用Nblock/Dv、L0C首末K条件 |
| 每task实例化两个GEMM导致挂起 | 两owner的同方向event冲突、构造初始token重复、析构消费不匹配 |
| PV非对齐尾块在长任务中大量变慢 | 先查有效K是否被物理对齐值替换、零步长ND2NZ补零及MTE2计数，再用受控对照检查Q/P重读、mask和归约；不凭结构差异断言主因 |
| QK单独正确，融合随机错 | UB跨pipe复用、X/P/U槽代次、C1FIX之前未保证旧PV读取结束 |
| 稀疏/PA名称齐全但普通SWA慢 | 实际仍独立pack及四次全网格launch，或按head重复KV搬运 |

# 验证方法

<a id="protocol--验证方法"></a>

## 双槽跨任务握手

生成者写出实际三个 actor 的可执行状态模型，至少枚举 L=0/1/2/3/7、两槽多次回绕、一个 AIV 故意延迟、零行侧、PA pack 开关。检查未 ready 读取、旧 reader 未结束就覆盖、错误 generation、非终态三方全阻塞、重复通知未消费和退出残留。阶段视为原子完成的抽象假设要明写；该模型不覆盖硬件 DMA 细节，不能替代设备测试。

可以按[参考模型配方及使用边界](pipeline.md#model-usage)在生成工程内建模。本节模型仅检查参考序列；必须另列最终源码与模型的对应和差异。参考模型PASS不代表实际生成代码已实现该协议。

随后依次跑单 work、跨槽回绕、ragged 尾块、A/B/A/B、调用者选定的原 ATK 用例、最终 release 的完整 5+5 profiler。只有真实最终代码及测量满足才宣告本协议已工程落地。具体产物和父级验收见[流水验收](validation.md#pipeline-validation)。

[^protocol-sdk-sync]: 本知识包只保守记录 mode2 的 0–9 编号语义；具体 SDK 的编号限制、Wait 重载和参与者行为仍须核对目标版本。
[^catlass-sync]: 历史 CATLASS `CrossCoreFlag` 封装将 flag ID 交给 AscendC `CrossCoreSetFlag<MODE, PIPE>` 和 `CrossCoreWaitFlag`；`CrossCoreFlagWithReverse<REVERSE_DEPTH>` 在正向 set/wait 累计到阈值时由消费者反向 set、生产者反向 wait，避免单向连续 set 过多（历史阈值 15）。本页协议是 SWA 所需的握手约束，不能把历史 wrapper 当作目标 SDK 的可编译签名；须核对当前版本的 mode、PIPE、ID 范围、反向 flag 和参与者行为。

<a id="structure--验证方法"></a>

## 混合核结构与片上复用

以真实源码画出每个区间的writer、最后reader和free事件。先验证一个M块的QK/PV actual shape、layout、L0C输出；再核验V1的E/m/l/P、V2的U/invL/O及LSE；最后恢复跨task流水并通过多次回绕、两dtype、四布局、尾部与A/B/A/B。禁止把debug dump运行的正确性自动转移到改变tile/事件后的release。

父级审查按[流水验收表](validation.md#pipeline-validation)检查。对于Q/P常驻和片上重叠，源码可证明装载次数与依赖，实际收益需profiler；不要仅凭静态字节量把瓶颈百分比写成实测。

[^structure-block-mmad-source]: 已分析的 CATLASS 版本中，ping-pong 的 STAGES 由 dispatch 与各层 stage 数决定，资源 owner 只有一个 L0C tensor；每次 block-scoped 调用管理自己的 GM/L1/L0 事件。通用定义和目标版本真实限制必须在本轮 checkout 核对，不能从知识页反推固定 `STAGES=2` 或固定事件 ID。
[^structure-ordinary-contract]: [普通SWA合法域](interface.md#swa-contract)，以及[窗口、sink和舍入契约](computation.md)。

<a id="model--验证方法"></a>

## 三 actor 状态模型的构造与检查

参考模型报告显式保存 `reference_model_only=true`、`verifies_generated_operator=false`、
`verifies_cpp_sdk_or_async_dma=false`、`hardware_verified=false`、`automatic_pipeline_acceptance=false`。
单配置 FAIL 或 INCONCLUSIVE 不返回成功；负对照命令失败属于预期证据，应保留报告。
自检成功要求正常配置通过且每个负对照检测到对应问题，而不是要求全部变体都 PASS。
之后仍进行最终源码事件映射、地址生命周期、零行/尾部/多次回绕、ABAB、原 ATK 和完整5+5设备验收。
