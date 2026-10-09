---
type: operator
title: SWA 分块优化与性能验收
description: 一般 SWA 和连续 prefill 的布局、复用、分核与流水优化，以及5+5采样和逐例性能证据校验。
tags:
- catlass-cpp
- sparse-flash-mla
- swa
- decode
- prefill
- stride
- performance
- pv-tail
- actual-k
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
- id: strategy-swa-contract
  resource: knowledge/operator/sparse-flash-mla/computation.md
  title: 共享KV窗口计算
  kind: contract
- id: prefill-shared-kv-formula
  resource: knowledge/operator/sparse-flash-mla/computation.md
  title: 共享KV与sink的Attention计算定义
  kind: repository
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

- [一般 SWA 分支与性能](#strategy)
- [连续短窗口 prefill 推导](#prefill)
- [性能证据的解析与逐例判断](#acceptance)

# 接口与概念

<a id="strategy"></a>

<a id="strategy--接口与概念"></a>

## 一般 SWA 分支与性能

本页服务一般SWA单模式生成，主算子和配套metadata保持完整接口。输入域以[SWA实际契约](interface.md#swa-contract)及调用者本轮冻结的契约为准；D512、N2=1、query head数1..128的2次幂、FP16/BF16、窗口127/0是已研究的硬件配置，不是按测试文件固化的路由。矩阵大小、有效长度、dtype和布局是运行时输入，不以case id路由，也不按已知shape返回固定结果。调用者指定的验收子集不是实现白名单。实现前先按[工作量门槛](development.md#implementation)核算任务数与 KV 搬运量。

普通域以[跨任务协议](pipeline.md#protocol)与[持久资源结构](pipeline.md#structure)为组织入口。首版就应有长prefill、短prefill/decode和尾块的完整路径；跨任务优化用于确有多个任务的路径，不强迫单任务制造虚假的双槽并发。

<a id="prefill"></a>

<a id="prefill--接口与概念"></a>

## 连续短窗口 prefill 推导

适用特征：BF16 Q/KV、共享 K=V、D512、BSND连续布局、每个KV head对应G=4个query head、因果窗口W=128、每个query head一个动态FP32 sink。以下以B=1、S=4096、一个KV head推导候选实现；这是一条固定形状SWA推导，不代表压缩或索引场景已实现。当前任务若只要求普通SWA，按[一般合法域](interface.md#swa-contract)保留完整公开参数并拒绝域外模式，不因本页提及其它模式扩大任务。输入形状、输出dtype及舍入规则仍以本次冻结契约为准。

本页保存公式、工程不变量和构造方法。开发者从这些约束生成主计算及独立metadata入口，再依据固定版本CATLASS标准组件实现；不依赖已有完整算子工程，也不提供可继承的精度或性能结论。

# 用法

<a id="strategy--用法"></a>

## 一般 SWA 分支与性能

首次实现必须读[普通SWA流水协议](pipeline.md#protocol)和[片上结构](pipeline.md#structure)，把整套调度作为一个方案落实。特别区分CATLASS内部GEMM双缓冲与整个C/V跨任务重叠；两者不能互相代替。最终按[流水验收](validation.md#pipeline-validation)检查实际源码。

先冻结每batch实际Lq/Lk、storage base、原生stride、query/head/kv关系、sink可选性与LSE，然后按下面步骤推导候选。CUDA式block计数、AIC/AIV计数和张量batch不是同一维度。对不同参数采用少量经容量验证的编译期tile变体，host/metadata按参数派发；禁止为每条case写一份相同kernel或将分支缩成某个固定shape并拒绝其余输入。

<a id="prefill--用法"></a>

## 连续短窗口 prefill 推导

先按[计算定义](computation.md)确认窗口、sink和cast位置，再依次完成形状推导、GM地址表、片上容量表、事件所有权表、任务与槽生命周期证明。需要连续 prefill 时，配合[通用流水协议](pipeline.md#protocol)、[片上结构](pipeline.md#structure)和[通用工程交付](development.md#engineering)落实持久共享 owner、跨任务两槽流水和每行一次倒数；不先交付串行版本。阶段探针用于调试，最终仍在完整流水路径上验证。后续性能实验每次只改变一个主因。

CATLASS的标准Pingpong BlockMmad提供布局、TileCopy、TileMmad与K轴分块规则；本页说明如何从该规则建立共享资源方案。以目标SDK实际接口和静态约束为准，不能把某次生成的类、构建命令或测量结果作为模板前提。接口冻结时记录SDK/CANN版本、架构、运行核数、编译选项和计时范围。

<a id="acceptance"></a>

<a id="acceptance--用法"></a>

## 性能证据的解析与逐例判断

性能报告评价是对既有采样证据的离线核对，不启动设备，也不能仅凭 JSON 宣称已检查原始 profiler。
生成工程可按本节实现自己的报告读取/校验入口。输入是调用者给出的基线、显式选定的 case_id、
相应用例文件以及可选的本轮候选报告；输出写新路径，不能覆盖任何输入。
未给候选只返回门槛与 `REFERENCE_ONLY`，不得返回候选 PASS。

# 代码模式

<a id="strategy--代码模式"></a>

## 一般 SWA 分支与性能

<a id="strategy--算子算法"></a>
## 算子算法

batch b的query本地位置t映射到原序列位置 `p=Lk[b]-Lq[b]+t`；可见KV为 `[max(0,p-127),min(Lk[b],p+1))`。prefill时Lq=Lk只是该式的特例，decode和chunked prefill必须保留Lk-Lq。

一任务覆盖同batch的query范围[q0,q0+T)和head范围[h0,h0+H)。逻辑行r对应 `t=q0+r//H, h=h0+r%H`。实际行数Mactual=Tactual*Hactual，pad行不参与输出。若完整head组小且跨query的物理行连续，可将T与H直接合并成矩阵M；head切片跨query时通常不再有单一row stride，须分query搬运或打包，不能让RowMajor越过未选择的head。

本任务全部query的KV并集从 `max(0,Lk-Lq+q0-127)` 到 `min(Lk,Lk-Lq+q0+Tactual)`，宽度最多128+Tactual-1；每行还要应用自己的mask，不能只用并集mask。尾部为满足固定N搬运而左移KV基址时重新计算每行相对列坐标。Lk<N时不能从负地址或越过有效storage加载N行：采用真实actual N和零补片上/暂存区，或分离有界gather。无效列的V若是NaN，P=0也不能防止0*NaN污染PV，必须保证实际矩阵输入的补齐值有限且初始化。

若任务域覆盖storage query以负责清零padding，上式还要处理`q0>=Lq`：取`qe=min(q0+Tactual,Lq)`，`begin=min(Lk,max(0,Lk-Lq+q0-127))`，`end=max(begin,min(Lk,Lk-Lq+qe))`；只有存在有效query且`end>begin`时发起矩阵计算。枚举断言对有效和padding任务均有`0<=begin<=end<=Lk`，不能只验证有效query而留下空任务的越界基址。每行mask依然按自己的有效性计算，整个task有工作不代表每行均有效。

FP16和BF16都保留QK FP32累加、FP32 exp与分母、P转换为当前输入dtype、PV FP32累加、最终转换输出的顺序。不要为了FP16复用而把P先变BF16，也不要在P cast前除分母。sink必须动态参与max和分母，不进入分子。无sink、空行和无效query输出按冻结契约单独处理，不能用一个常量sink=0代替缺失sink。

<a id="strategy--多head的sink数据准备"></a>
### 多head的sink数据准备

`sinks`为全batch共享的FP32 `[N1]`，同一次launch里不随query/task改变。每个AIV只需在launch内把至多128个sink（512字节）由GM加载一次，之后按本任务head组构造行向量；下一次调用重新加载，不能跨调用缓存值。对每AIV的R行，行r取 `sink[h0 + (part*R+r) % H]`。当H>=R，所需值是head组内一段连续数据；当H<R且H整除R，是长度H的周期重复。利用连续DMA、局部复制/向量gather或有界广播生成这段行向量；小于32字节的读取使用合法的短拷贝/padding，不能向GM额外越界读取。

特别核对`H==R`边界：M128/R64且H64时，两侧AIV分别处理两个query，但都使用`sink[0:64]`，第二侧起点是`(64%64)=0`。H128时两侧才分别用`[0:64]`、`[64:128]`。不能写成`if H>=64: sinkCache[part*64]`而漏掉取模；H64的第二侧会读出有效sink表之外，H128却恰好正确。带head切片时起点为`h0+(part*R)%Hactual`，跨该组边界的行向量需分段/周期构造，不能读取未选择的head。测试用非等值sink，并让两query的Q/KV不同，分别比较两侧O/LSE。

不要为每个head再遍历R行逐bit造mask、单独GM GetValue、Duplicate和PipeBarrier：每任务scalar代价为O(H*R)，多head长prefill同时增大任务数与单任务开销。以M128/R64为例，H4/16/32时该做法分别扫描256/1024/2048次，而输出行数一直只有128。N1=32长prefill更容易由这段准备而非矩阵乘限制。若head组与AIV行映射对本core的全部task相同，展开的sink行向量也可在循环外准备一次；ragged/无效行仅修改当前工作副本为有限中性值，不污染后续task的持久sink。若任务切head导致h0改变，只从片上原始sink表重新选取本组，不再次从GM逐head读。

设计时写清 `每任务行数 × 任务数` 与每阶段scalar循环次数。mask按query边界分组、sink按head周期分组，两者的复用维度不同。性能回归应同时覆盖H=1/4/16/32/64/128，防止在H4上低成本的控制代码随head数放大。归因最终以真实设备时长及流水证据确认，不能只凭源码把候选瓶颈宣称已证明。

一种满足32字节拷贝粒度的构造是：将原始sink用`Brcb`扩为每head一个8个FP32的datablock，再按N1个datablock为周期局部复制到能覆盖行起点加R的长度；从head起点切片，以`WholeReduceMax`收集每block内相同的值到连续行向量。每次repeat只逻辑读取一个8元素block，其余block stride为0，下一repeat前进一个block；核对SDK的repeat/stride单位并分段处理上限。这样N1=1/2/4也不会出现16字节以下普通DataCopy长度。也可使用`Gather`从缓存sink按行索引读取；索引单位及offset dtype须按当前SDK核对，byte offset与元素head编号不能混用。两种方案都只产生头向量，不能改变sink logit值或softmax舍入路径。

<a id="strategy--大head任务的有效行与padding控制量"></a>
### 大head任务的有效行与padding控制量

即使列mask已经向量化，每task仍逐行求`r/H`、拼位图、逐行清O，也可能把长prefill变成scalar发射密集路径。完整head组、任务不跨batch时，有效query在task内是一个连续区间，可以一次推导两AIV各自的有效行位图，不逐行做除法：

```text
validQueryBegin = max(q0, max(0,Lq-Lk))
validQueryEnd   = min(q0+Tactual, Lq)
若Lk==0或validQueryEnd<=validQueryBegin：有效行区间为空
否则 task行区间为 [(validQueryBegin-q0)*H, (validQueryEnd-q0)*H)
再与本侧[part*R, min(part*R+R,Mactual))相交，转换为本侧[lo,hi)
位图 = Prefix(hi) & ~Prefix(lo)，Prefix(0)=0、Prefix(64)=UINT64_MAX
```

在做乘法前先裁剪query边界到task范围，使用溢出检查与64位地址量纲；不能做`1ULL<<64`。R>64按多个64bit分片构造。head切片也可按本片Hactual推导，但q/head到物理行映射须继续一致。此mask是“该query有合法attention”的行mask，**不是窗口列mask**；每query的左右列边界仍要单独计算。

把完整有效的任务设为明确fast path：不做逐行invalid检查/清零；首部无效query和尾padding才使用批量行mask处理。FP32 `X[r,N256]`可对连续无效行段用合法count操作或repeat按行Duplicate；短行向量在对齐基址配位图清零。相同task的V1、两个Dv256输出stripe复用同一行有效性，不能各自重新遍历64行算一遍。对H64/R64，常见稳态每侧是一个完整query；H128/R64时是同query的半个head组，二者都应能用常量次数控制操作表达。

PV的 `Kvalid` 取实际窗口并集宽度，物理对齐值仅用于相应布局/容量，P的GM行距另行保留。优先采用支持实际K的copy/MMAD链；若组件必须执行对齐K，则单独定义 `Kexec`，按真实分形footprint初始化padding。完整变量、分片伪码、ND→NZ/LoadData/MMAD参数及PA页段规则集中在[结构页的通用PV尾块处理](pipeline.md#structure--pv-tail)。生成代码用有效长度、tile容量和余数推导，不按某个K、head数或case ID特判。

对于需要显式补零的路径，使用足够容量的连续有限零区与正源行距或经验证的本地初始化；不要把多行 `ld=0/srcDValue=0` 的GM→L1 ND2NZ作为普通SWA尾块的默认优化。一次API调用不保证低代价。新SDK若需评估其它方案，先核对合法性、真实读取范围，再用保持数学/输入一致的同机对照确认，不能由源码调用数推断性能。

<a id="strategy--pv-tail-evidence"></a>
<a id="strategy--pv-尾块的诊断结论与适用边界"></a>
### PV 尾块的诊断结论与适用边界

历史单因素诊断显示：PV 先把有效 K 向上对齐，再用多行零步长 GM→L1 ND2NZ 补零，可能让 MTE2 明显退化；只扩大零源容量不能消除该问题。将零源改为正行距连续有限区域，或直接让组件执行有效 K，在当时的诊断输入上都避免了主要退化。这个结论用于提示 API 选择和 profiler 定位，不是新任务的性能基线，也不证明其它长度、dtype、布局或 SDK 版本具有相同比例收益。

对新的生成工程，先用调用者提供的用例和基线冻结输入与计时范围，再做保持数学一致的单因素对照：记录有效 K、执行 K、补零方式、真实 GM 读取范围、精度、MTE2/MAC/Vector profile 和完整设备时间。PMU 管线指标可重叠，不能相加；诊断直调结果不能替代正式 ATK 和主核验收。未见组合按[尾块验证矩阵](validation.md#pipeline-validation--pv-tail-validation)验证。

<a id="strategy--分核策略与基本块切分"></a>
## 分核策略与基本块切分

候选分支按工作量推导：

| 输入特征 | 设计重点 | 需要比较的候选 |
|---|---|---|
| 长prefill、很多query、较少head | 跨query合并M，复用KV窗口，持续资源与跨任务流水 | M64/128，T=M/H；根据128+T-1选N，不能固定所有head都N192 |
| 多head、较少query或decode | 批次与head维共同分核，避免只有B个任务 | H16/32/64/128和T1/少量T；优先使任务足够又不过度重复KV |
| KV极短，如Lk<=16 | 主要成本可能是launch、矩阵/向量握手与pad | 一次小GEMM或向量短归约，需真实profiling选择，不能据规模直接宣称快 |
| ragged batch或尾query | 任务不能跨batch，实际行数和两侧AIV零行协议 | 每batch独立ceil分片，工作量前缀或分桶派发 |
| 非连续或page KV | 真实stride/gather代价，不能误读或隐形改输入 | CATLASS物理ld直接读；只有不能表达的视图才需要有界pack/gather |

一个可推导的初始候选是长prefill M<=128、T=max(1,M/H)、N=round_up(128+T-1,64)、Dv=256。举例：H=1且T128时N256；H=4且T32时N192；H=16且T8时N192。它是候选而非固定最佳值，特别是N192有较多pad时可对比N160等SDK合法形状。所有变体均重新核对4*M*N和4*M*Dv不超过128KiB L0C，L1 A/B双槽总量不超过512KiB，L0A/B各64KiB及UB存活峰值192KiB；不能复制任何固定样例的绝对偏移。

head极少的decode可由batch分核；B不足时再考虑head或窗口分片。split-window会增加partial归并及数值舍入差异，窗口仅128时不能预设收益。任何额外归并核都必须计入完整计算时间，不能只挑最快主核当速度比。

所有任务覆盖关系先在CPU枚举证明：每个有效(b,t,h)恰好出现一次；无效query的输出初始化另有明确写者；不读取别batch或page；每逻辑AIC对应两AIV共同执行同一任务序列。按task真实工作量均衡，而非按tensor总元素直接均分切断query窗口。

<a id="strategy--数据路径与存储层级"></a>
## 数据路径与存储层级

地址计算始终使用元素stride和64位偏移。BSND `q[b,t,h,d]` 为 `b*sB+t*sT+h*sH+d*sD`；TND把cu_q[b]+t作为第一维，再按真实stride读。KV同理，但query位置偏移Lk-Lq与storage偏移cu不是同一件事。输出stride按实际输出对象，不默认等于输入stride。

若D连续且KV只有序列stride扩大，QK B可用ColumnMajor的物理ld=kv_stride_T、PV B用RowMajor同一ld，避免整张KV contiguous复制。GM到L1 copy的实际行/列数与原ld分别传递。只有该布局不满足CATLASS拷贝约束时才pack；pack仍是本算子的执行部分，容量、同步、时间都必须入账。paged KV首先按逻辑k求page/table entry/块内offset，再读实际block和stride；不能把block table当作纯性能提示忽略。

Query reshape为矩阵A仅在相邻逻辑行等于单一物理ld时成立；例如未切head的连续BSND可以合并T和H，切head后若query stride仍为全N1*D则跨query不连续。写O和LSE时按原输出轴scatter对应行，不能把pad head或pad query写进下一batch。

<a id="strategy--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

长任务序列使用持久Cube owner+typed QK/PV views与两GM槽，保留slot parity、每块实际shape、KV基址及每槽invL直到对应输出消费完成。i时刻同时存在当前任务i和待输出任务i-1，二者可能属于不同batch/头组/尾块；不可用一个覆盖掉的runInfo存其地址。AIC和两个AIV必须能从同一metadata独立恢复对应task信息。

任何M变体先决定每AIV行数。若Mactual不够两侧，不应让零行AIV提前退出而漏发mode2通知：零行侧不做非法DMA/Vector，但仍参加本任务的必要ready/free；整组L=0才共同退出。Vector的R需按API要求对齐后处理pad标量，Brcb repeat和二级归约次数由R推导，不固定64。pad行的inverse不得产生NaN进入有效输出。

行状态的mask/清零使用对齐基址和批量位图，参照[行状态清零与Vector地址对齐](development.md#engineering--行状态清零与vector地址对齐)。逐row偏移后的一元素Vector操作既增加scalar发射开销，也可能违反UB地址对齐；不能用更多同步掩盖指令地址不合法。有效长度、切head与AIV分片共同决定无效行位图，精确GM尾部写回单独处理。

query行分工与KV装载分工独立。例如两个AIV按`[floor(Nactual*part/2),floor(Nactual*(part+1)/2))`搬运KV时，即使某侧`queryRows=0`，它分到的KV段仍可能非空；S1必须完成该段，Cube才能读取完整KV。零query行只能跳过该侧S3/S5的行运算与输出，不能笼统跳过S1或其ready通知。PA尾task应实际覆盖“零query行但非零KV段”的组合。

长prefill的late free仅保护即将覆写的numerator，不阻塞无关score预发射；短分支可使用更简单但闭合的协议，不能为形式统一增加无收益同步。数据清零、pack、QK、mask/softmax、PV、归一化、输出和可选LSE均纳入依赖表。所有launch退出恢复mask并完成发送管线；sync base由相应launcher所有者管理，OPC wrapper已初始化时共享数学函数不重复初始化或在消费者完成前清零。入口初始化不等于退出恢复：普通OPC路径若wrapper不负责退出清零，正式入口应在完整排空后恢复，按[退出恢复的责任与条件](development.md#engineering--诊断构建与正常构建的证据边界)核验。重复调用/跨shape交替都必须通过。

设计模型不能仅统计总set/wait数量相等。至少把一个AIC和两个AIV作为独立actor，枚举可达Stage边界交错，分别检查pack开关、0/1/2/3及多次槽环绕的任务数：未就绪不能读、最后消费者前不能覆写、通知消费的slot/generation正确、非终态不能全部阻塞、退出无残留token。CrossCore的配对到达与核内HardEvent分开建模；不能把mode2通知无依据地当作容量1的布尔flag，也不能假定真实硬件允许无限通知积压。参考模型采用概念FIFO只是顺序诊断，实际峰值和可容纳状态仍按冻结SDK核对。模型需声明把哪段实际硬件管线视为原子完成；它证明抽象协议，不替代MMAD/FIX、DMA完成事件、sanitizer及重复NPU调用证据，详见[协议](pipeline.md#protocol)与[模型说明](pipeline.md#model-usage)。

<a id="strategy--参数装载差异与性能归因"></a>
### 参数装载差异与性能归因

正式与直调路径共用数学源码但结果不同，先核对进入S0的参数值及其恢复方式。按[对象表示、别名和对齐要求](development.md#engineering--tiling与控制结构的对象表示别名和对齐)检查tiling/control生产端与消费端的大小、偏移、位表示及实际访问类型，特别是混合32/64位整数与浮点成员的结构。优化编译可能使不合法的类型访问表现不同；不能仅凭更换复制写法后少量场景通过，就断言编译器存在缺陷或全部错误根因已证实。

比较时固定数学核、编译选项、控制值、输入和设备条件，只改变参数恢复边界，并保留失败与修正候选各自的构建身份。用单核/多核、整块/部分块和连续多任务区分参数解释、分核与尾处理；覆盖控制值变化及跨shape复用。具名参数dump可辅助核对各字段，但诊断会影响代码布局和时序，结果必须回到无trace正常包复验，之后仍需完整精度与5+5性能验收。合法字节复制可能被编译器合并为更宽的装载，不按源码循环次数推断慢或快；也不将全局屏障、降低优化级别或特殊别名编译flag设为默认修复。

<a id="prefill--代码模式"></a>

## 连续短窗口 prefill 推导

<a id="prefill--算子算法"></a>
## 算子算法

<a id="prefill--形状与数值推导"></a>
### 形状与数值推导

先枚举query跨度T，计算M=G*T与窗口并集上界W+T−1。N取SDK允许且Vector尾段能正确处理的补齐宽度；192来自下面三段64列的安排，不是Attention数学要求。联合选择PV输出宽度Dv和L1/L0 K，检查FP32的4MN与4MDv均不超过L0C、A/B双槽容量、UB存活峰值、DMA对齐及AIV整行归属。至少记录一个保留候选和一个淘汰候选及具体约束原因；合法不等于更快。

例如G=4时，T16和T32对应M64和M128，均可用N192，但任务数与片上占用不同；T64对应M256，即便N仍为192，完整QK的L0C需196608B，超过128KiB，不能直接使用本页不切M/N的标准Block。改变T、N或Dv后重新生成布局、资源、metadata及事件生命周期，不只替换常量；不整除S的候选还需真实尾块路径。

逻辑行`r=G*t+h`，t为query位置，h为head。设每任务处理T=32个query，则M=G*T=128。连续窗口并集最多含`W+T-1=159`个KV，按候选tile粒度补齐到N=192；补齐列始终受mask约束。任务起点q0下，S>=N时可取`kv0=min(max(q0-W+1,0),S-N)`，物理列c对应`k=kv0+c`，仅当`max(t-W+1,0)<=k<=t`时有效。尾部基址钳制不改变逻辑窗口。S<N或不足T的query尾块需要另行推导有界搬运和actual shape，不能直接沿用固定形状地址。

对一行的物理score x、最大值m、FP32指数e、分母l和分子u，按以下数值依赖生成各阶段：

| 量 | 公式及精度 |
|---|---|
| x[c] | `FP32_accumulate(Q[t,h,:]*KV[k,:])/sqrt(D)`；非法列赋`-FLT_MAX` |
| m | `max(sink[h], max_c(x[c]))` |
| e[c] | `exp_fp32(x[c]-m)` |
| l | `sum_fp32(e[c])+exp_fp32(sink[h]-m)` |
| P[c] | `BF16_RINT(e[c])` |
| u[d] | `FP32_accumulate(P[:]*KV[:,d])` |
| O[t,h,d] | `BF16_RINT(u[d]*FP32(1/l))` |

D512对应FP32 scale约为`0.044194173824159216`。sink不乘scale，只进入m和l，不增加value项。l在P转换前由FP32指数求和；把除法前移到P的BF16转换前会改变数值路径。QK和PV均覆盖完整D512，PV切分输出D轴不会减少点积维度。倒数在每行计算一次后广播；近似倒数替代FP32除法属于需单独验证的数值变更。

<a id="prefill--从张量地址推导gemm布局"></a>
### 从张量地址推导GEMM布局

矩阵shape和物理leading dimension必须分别列出。以下offset与leading dimension均以元素计，只有转换为GM字节地址时才乘dtype大小。`RowMajor(i,j)`的offset为`i*ld+j`，`ColumnMajor(i,j)`为`i+j*ld`。

| 调用/操作数 | GM起点 | 逻辑shape | 布局与ld | 必须成立的地址关系 |
|---|---|---|---|---|
| QK A | Q的`q0*G*D` | M×D | RowMajor，D | 相邻逻辑行是同一query的下一个head |
| QK B | KV的`kv0*D` | D×N | ColumnMajor，D | B[d,c]就是KV[kv0+c,d]；只换视图，无需GM转置 |
| QK C | 当前score槽 | M×N | RowMajor，N | score行长N |
| PV A | 当前P槽 | M×N | RowMajor，N | K轴为窗口的物理N |
| PV B | KV的`kv0*D+d0` | N×Dv | RowMajor，D | Dv=256是调用宽度，物理ld仍是512 |
| PV C | 当前numerator槽的d0 | M×Dv | RowMajor，D | 两次d0=0/256写入同一M×D矩阵的左右半边 |

候选QK的L1/L0 tile分别为`(M,N,K)=(128,192,256)/(128,192,64)`，actual shape为`(128,192,512)`。PV的L1/L0 tile为`(128,256,192)/(128,256,64)`，每次actual shape为`(128,256,192)`，D轴调用两次。对子矩阵建立tile layout时保留原ld；把PV的ld改成256会把跨行读取和写回都移位。

CATLASS layout的GetOffset接口接收MatrixCoord。该类型与通用MakeCoord推导的tuple不是任意可互换的；混合有符号的0与无符号tile维度可能得到不同的tuple类型，无法隐式转换。坐标先核对范围再按接口显式构造MatrixCoord，不能仅凭二者都含两个数字判断类型兼容。GetTileLayout等其他接口的参数也以各自声明为准。此约束覆盖GM到L1的下一块预取和L1到L0的K切片，不只覆盖最外层矩阵layout；复用typed Copy不改变源layout的GetOffset签名。首轮构建前遍历所有坐标调用，核对两项坐标的实际类型，特别是整数常量0与无符号循环变量混用的位置。

<a id="prefill--vector的mask归约与广播"></a>
### Vector的mask、归约与广播

一个AIV处理R=64行。FP32一条256B向量含64元素，一个32B块含8元素；BF16一个32B块含16元素。repeat描述重复的向量操作，block stride与repeat stride描述32B块间距，mask描述当前repeat中的有效lane。使用手动mask的API时必须显式设置后续所需mask，不能继承前一个masked Duplicate留下的状态。

参数形式不能因操作都属于Vector而互相套用。传统Duplicate在dst与scalarValue之后是mask、repeatTime、dstBlockStride、dstRepeatStride，两个stride是独立标量；它不接受用一个二元stride结构体替代最后两参。三输入逐元素二元API的stride结构、归约API的repeat/mask顺序与Duplicate不同。调用前把当前重载的每个参数与地址公式逐项对应，再检查isSetMask的默认或显式取值。

位图mask还需区分数组顺序和底层函数参数顺序：当前Duplicate的两元素数组按低64位、高64位排列，内部转成SetMask(maskHigh,maskLow)时会交换下标。FP32只有低64个有效lane，因此数组第0项放实际位图、第1项为0；不能把底层SetMask的实参顺序直接写成数组顺序。当前数组重载接受uint64_t*，因此局部位图数组不能声明为const；按API真实参数类型声明数组，不用强制去除const掩盖类型不匹配。用于窗口无效列和逐head sink填充时都要检查这一点。低64位全0会使应写的FP32 lane没有被初始化，随后row max/exp可传播NaN；检查QK score、P和numerator的finite分布，可以区分矩阵计算错误、softmax阶段错误和输出未覆盖。

| 操作 | 从地址公式得到的安排 |
|---|---|
| 192列压到64列 | 对每行u=0..63，按`(x[u] op x[64+u]) op x[128+u]`生成64个中间值；op为max或sum。原行repeat stride=`192*4/32=24`，中间行stride=`64*4/32=8` |
| 64列归约到1列 | 第一级按8个FP32一组归约，R个repeat输出R×8个值；第二级对连续的R×8个值做R/8个repeat，每个32B组对应原来的一行，最终得到R个行标量。不能把第二级repeat误设为R |
| 行标量广播 | 将R个FP32值按每值8份扩展为R个32B块，空间为R×8×4=2048B；Brcb处理8个源标量为一组，repeat为R/8。逐行二元操作让广播源block stride=0、repeat stride=1，即同一行复用一个32B块，下一行推进一块 |
| score减行最大值 | 每行分三个64列stripe，score的repeat stride为24；广播源仍每行推进一块 |
| numerator乘行倒数 | 先把每行Dv=256列搬成紧凑UB矩阵，再分四个64列stripe；紧凑行repeat stride=`256*4/32=32` |

sum的结合顺序影响FP32误差。三段相加后再做两级8宽归约是一种明确的数值顺序；切换WholeReduce或改变树形结构需重新验证精度和耗时。上表是向量索引关系，不要求把每个lane写成标量循环。

实现前逐项映射API参数顺序。当前传统BlockReduceMax/Sum的dst、src之后依次是repeatTime、mask、dstRepStride、srcBlkStride、srcRepStride；不能套用其他向量API先mask后repeat的习惯。对R行、每行64个中间值且R可被8整除的安排，第一级repeat=R、mask=64，第二级repeat=R/8、mask=64；mask取决于FP32向量宽度，不随R改变。R64时对应第一级64/64、第二级8/64。二级若误写repeat=64、mask=8，会按错误源跨度继续读取scratch之外的UB并覆盖其他行状态；一级恰好两个数相同，会掩盖这种错误。无自动mask的模板参数isSetMask=false仅表示mask由调用者预先设置，不改变repeat参数的位置。

窗口mask按query生成并复用于其G个head：第c0列stripe的有效lane为`[clamp(left-c0,0,64),clamp(right-c0,0,64))`，left/right为相对于kv0的半开区间端点。前缀位图在长度64时必须单独取全1，避免整数移位64的未定义行为；无效lane才写负无穷近似值。G行masked Duplicate的repeat stride为N/8=24。不要只对四个head中的第一行施加mask。

sink从每次调用的输入读取，行r使用`sink[r%G]`。若用四个间隔lane mask分别Duplicate四个head的值，虽然lane集合互斥，四次写仍触及同一批32B块；每次写后都要有PIPE_V顺序依赖，不能只在四次写完后加一次屏障。

<a id="prefill--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

<a id="prefill--从标准block推导共享资源所有权"></a>
### 从标准Block推导共享资源所有权

标准Pingpong构造函数同时承担buffer映射、event初始token设置和stage索引初始化，析构函数消费最终释放token。因此两个对象即使使用同一Resource，也不意味着可安全复用：若各自初始化同一event或持有独立stage索引，会破坏单一资源的生命周期。

对于顺序调用的QK/PV，建立一个覆盖整个Cube任务流的资源owner；typed view只选择SDK的布局、copy与mmad类型以及本次shape。owner持有L1 A/B和L0 A/B槽、L0C、event映射及三个持续更新的索引：L1当前槽、L0A当前槽、L0B当前槽。view不初始化或析构token，不在切换QK/PV时重置任何索引。buffer容量取所有view需求的逐项最大值，容量推导见下节。

以两个stage、A事件编号s、B编号s+2为例，按标准Block的生产者/消费者关系推导事件表；同一个编号只有在不同HardEvent方向上才是不同事件：

| 资源/事件方向 | token含义 | 初始化及每轮消费/返还 |
|---|---|---|
| L1，MTE1_MTE2 | 槽已被L1→L0读完，可以覆盖 | owner每槽初始化一次；GM→L1前wait；该L1槽最后一次L1→L0读后set |
| L1，MTE2_MTE1 | GM→L1的数据可读 | 每次搬入后set；首次读取该L1槽前wait；不预设ready token |
| L0 A/B，M_MTE1 | MMAD已读完，可载入下一块 | owner每槽初始化一次；L1→L0前wait；最后一次使用该A/B块的MMAD后set |
| L0 ready，MTE1_M | L0载入完成，可做MMAD | 按标准Block在对应copy后set、MMAD前wait；这一临时事件不跨调用遗留 |
| L0C，unitFlag依赖 | 计算完成与FIX读出/L0C再利用的顺序 | 启用unitFlag时遵守标准Block：非末次MMAD用`0b10`，末次用`0b11`，L0C→GM携带`0b11`，不能仅以函数返回视为完成 |

本候选的unitFlag模式仍按标准Block成对初始化/销毁`FIX_M`的EVENT_ID0；各次operator不消费该保留token。它与unitFlag承接的实际L0C依赖是两件事。若改为非unitFlag模式，必须重新使用完整的FIX_M/M_FIX读写保护协议，不能只删除unitFlag参数。

构造typed view时从SDK类型推导L1/L0 layout、AlignHelper、TileCopy和TileMmad；K分块、预取、首次清零累加器、尾块actual K、最后读者释放和stage推进顺序遵守标准Block。生成时改变的是资源归属与引用关系，不另造矩阵乘法。审核重点为：每个初始free token只有一个owner，每次wait对应唯一set，索引变化与真实使用的物理槽一致，最后由owner统一排空并析构。

<a id="prefill--跨任务两槽流水"></a>
### 跨任务两槽流水

内部K双缓冲不覆盖Cube等待Vector的时间。可另外设置两个GM任务槽和两个UB分母槽。设本核任务数L，局部任务号j，slot=`j mod 2`；任务号、slot、全局query位置分别计算。

| 阶段 | Cube顺序 | 两个AIV各自的顺序 |
|---|---|---|
| 启动i=0 | QK(0)，发score-ready | 等score-ready，softmax(0)，发P-ready |
| 中间i=1..L-1 | QK(i)，发score-ready；等P-ready(i-1)，PV(i-1)，发numerator-ready | 等score-ready(i)，softmax(i)，发P-ready；等numerator-ready(i-1)，归一化/写O(i-1)，发free |
| 排空i=L | 等P-ready(L-1)，PV(L-1)，发numerator-ready；消费最后未取走的free | 等numerator-ready(L-1)，归一化/写O(L-1)，发free |

表中L=0须统一退出，L=1只有启动和排空。覆盖同槽numerator之前，PV(j)在j>=2时还需等待free(j-2)；前两任务没有旧消费者，不等free。最终再消费最后min(2,L)个free。两个AIV都参与同一任务的mode2同步，不能让一个AIV替另一个提前宣布P或输出完成。

| 两槽flag组 | 编号示例 | 发送管线 | 消费端 |
|---|---|---|---|
| score-ready | 0、1 | AIC的PIPE_FIX | 对应的两个AIV |
| P-ready | 2、3 | 各AIV的PIPE_MTE3 | AIC |
| numerator-ready | 4、5 | AIC的PIPE_FIX | 对应的两个AIV |
| free | 6、7 | 各AIV的PIPE_MTE3 | AIC，保护同槽numerator重写 |

同槽score、P、numerator是互不重叠的GM区，复用证明分别成立：

- QK(j)之前的PV(j-2)已等待P-ready(j-2)，所以旧score已由Vector读完，新QK可覆盖score。
- softmax(j)等待QK(j)的FIX通知；该QK在旧PV之后，且共享GEMM的MTE2→MTE1→MMAD→FIX依赖证明旧PV对P的读取已完成，才能覆盖同槽P。仅有C++调用顺序不能代替这条跨pipe完成链。
- QK(j)不改numerator，故可与旧输出归一化重叠；把free等待放在真正覆盖numerator的PV前。若把等待移到QK前，需要重新评估失去的重叠。
- AIV顺序是softmax(i)再输出(i-1)，因此输出(j-2)消费完分母后，softmax(j)才复用同一个分母槽。两份分母在此之前都必须独立存活。

<a id="prefill--本地pipe和跨核通知的边界"></a>
### 本地pipe和跨核通知的边界

跨核ready只证明GM生产者完成，不能自动保护接收端UB重用。GM→UB前需保证旧Vector使用结束，搬入后保证MTE2完成才运行Vector；Vector产物→GM前保证Vector完成，UB改写前保证MTE3已读完。启动时可能连续执行softmax(0)、softmax(1)，同样要保护第一次的UB临时区。每次P写回后保留MTE3→V完成依赖，再复用P所在UB；输出的两次D256搬运之间也保持这些依赖。

同步基址由实际launcher管理：直调入口设置本次launch的地址，退出时由同一入口恢复为0；正式OPC wrapper已设置时，共享数学函数保留该值，不重复设置或清零。两种路径都先完成最终free排空与资源析构，再以成对的发送管线→标量事件等待通知发出：AIC用FIX_S，AIV用MTE3_S，最后交还各自launcher。这两个方向的事件编号不得与其他存活token冲突。外层退出处理同样覆盖metadata解码失败/无任务路径；函数返回或笼统PIPE_ALL不能替代发送完成依赖。已发送通知由接收核通过其自身有效基址读取，不必另加反向确认。

Vector控制状态也属于退出契约。显式或API内部设置过lane mask后，不能假定最后一次算术操作会恢复默认；在所有AIV退出路径统一调用目标SDK的掩码恢复API，核对VECTOR_MASK_0与VECTOR_MASK_1均回到全1，若切换过计数mask模式也要恢复普通模式。FP32运算期间只使用低64个lane，不代表退出时高64位可保留为0。该缺陷可能数值精度和racecheck均通过，却被memcheck报为寄存器未复原；仍需修复并用同源release与instrumented构建重新验证，不能过滤此类目标核告警。

<a id="prefill--分核策略与基本块切分"></a>
## 分核策略与基本块切分

双AIV映射先从运行模式确认每AIC有两个AIV，再令part取0/1。AIC逻辑核号为其block index，AIV逻辑核号为`block index/subblock count`，part来自subblock index；不能把AIV原始block index直接当作workspace核号。

ASCEND_IS_AIC/ASCEND_IS_AIV在当前SDK展开为带constexpr的条件语法片段，用于紧随if的分支。它们不是布尔表达式，不能直接放入三目运算、赋值或普通函数实参；核号折算先赋AIC基准，再在AIV编译分支中调整即可。

| AIV行归属 | 推导 |
|---|---|
| 本任务行 | `part*R .. (part+1)*R-1`，R=M/2=64 |
| 当前query | `q0+floor((part*R+localRow)/G)` |
| 当前head | `(part*R+localRow)%G` |
| score/P槽内元素偏移 | `part*R*N` |
| numerator槽内元素偏移 | `part*R*D` |
| 最终输出起始行 | `q0*G+part*R` |

R可被G整除，所以每个AIV独占16个完整query的四个head，两个AIV覆盖128行且无重叠。若改变G、M或AIV数量，必须重新证明整行归属与同步参与者，不能只修改一个除数。

metadata入口独立生成调度数据，不读取Q/KV内容、不计算attention数值。任务总数在固定形状下为`J=S/T=128`；推广到非整除S时使用ceil并携带真实尾长。对运行时查询的C个AIC，可取本核半开任务区间`[floor(J*c/C),floor(J*(c+1)/C))`，验证所有区间互斥且并集为`[0,J)`。20个AIC时每核6或7任务、最多7波；这是调度算术，不是延时保证。已有足够query任务时，split-S2还会增加归并和GM流量，须有实测收益才能引入。

metadata建议包含magic/version、实际使用核数、任务数、形状/窗口签名以及每核valid/begin/end，保留字段清零。header和record长度按协议定义，表容量至少`headerWords+C*recordWords`；固定表容量不是物理核数。host构表、launch、workspace大小及device解码必须使用同一核数和形状契约。device先验证有符号范围再转换无符号数，检查版本、核数、begin<=end<=J及核号；无任务核在初始化跨核协议前一致退出。非法表应由host拒绝并报告，不能把device不写输出当作成功。

<a id="acceptance--代码模式"></a>

## 性能证据的解析与逐例判断

<a id="acceptance--报告结构和采样语义"></a>
## 报告结构和采样语义

| 字段 | 约束 |
| --- | --- |
| task / scope / units | performance_device / formal_main_kernel / us；直调诊断保持 direct_c_abi，不能自动升级 |
| 基线 schema_version | 整数1 |
| 新 collection | warmup=5、sample_count=5、last_samples=5、statistic=mean_all_5 |
| 历史基线 collection | warmup=20、sample_count=50、last_samples=30、statistic=mean_last_30；仅基线允许 |
| cases | 非空对象数组，每行 id 是唯一非负整数，bool 不能当整数 |
| case 采样字段 | sample_count/last_samples 必须与顶层 collection 一致；重复出现的 task/scope/units/collection 不得矛盾 |
| validated_cases | 若给出，必须等于 cases 的实际条数 |
| 时长 | 有限、严格正的数值，排除 bool、字符串、NaN、Inf、零和负数 |
| 基线身份 | 每例 case_spec_sha256 是64位十六进制，最低速度比为有限正数 |

候选保存5条独立正式 `samples_us`，不含预热，也不舍弃其中任何一条。
基线要求 device_us 和 raw_mean_us；候选至少提供 device_us/mean_us 等均值摘要。
同一行若同时提供 device_us、raw_mean_us、mean_us 或历史 last30_mean_us，均值必须一致，
核对容差为 `rel_tol=1e-8,abs_tol=1e-6`；5样本协议禁止出现 last30_mean_us。
有原始样本时从样本重新计算均值并核对摘要，验收使用重算值，不能借摘要舍入容差把超限值变成通过。
只有冻结的旧50/30基线允许没有内嵌 samples_us，此时明确 `baseline_raw_samples_checked=false`。
候选缺原始5条样本不可采用这一兼容规则。

JSON 读取应拒绝重复对象键。输入报告和原始用例都记录文件 SHA256；规范化完整 spec 的哈希采用
UTF-8、键排序、紧凑分隔符、保留 Unicode、禁止非有限值。ATK 用例从唯一的
`inputs[name=smla_case_spec].range_values` 解码，支持该字段是字符串或对象；直接 manifest 支持对象行。
外层 id 若存在必须等于内层 case_id。显式所选 ID 非空、无重复，基线/候选/用例都不得缺任何一条。

没有 schema_version 的历史基线还需原 case JSON 才能建立当前 ID 与完整 spec 的映射。
先检查 task、validated_cases、唯一 ID、每例50/30、两个摘要一致，再形成带身份哈希的规范化基线，
保留历史 collection，最低速度比缺省采用本任务约定0.8。这个关联不是对历史实际运行输入的独立证明。
已有规范化基线若同时给出用例文件，须复核哈希；候选给出哈希时须与基线一致，缺失则标记
`NOT_AUTOMATICALLY_VERIFIED`，另外核验输入身份，不能把同 ID 当同输入的充分条件。

<a id="acceptance--逐例判断和失败语义"></a>
## 逐例判断和失败语义

对每条选中 case 计算 `limit_us=baseline_us/minimum_speedup`、`speedup=baseline_us/candidate_us`，
候选重算均值 `<=limit_us` 才通过，边界相等通过。最低速度比从冻结基线取值，只有显式用户目标才覆盖。
所有选中 case 分别通过才可报告性能子集 PASS，未选中候选 ID 单独列出，不计入本次覆盖。
硬门槛只决定是否允许结束性能验收，精度、同步和交付验收仍独立；更高性能仍可继续优化。

结果至少保留输入文件与哈希、所选 ID、两侧 collection、baseline/candidate duration、limit、speedup、
原始样本是否核对、输入身份状态及计时边界证据。离线汇总未重读 raw CSV、未观察预热隔离或实际张量时，
不得把报告声明写成实测核验。建议状态/退出语义：PASS 与 REFERENCE_ONLY 为0、性能 FAIL 为1、
缺字段/缺例/数值非法/口径冲突或写文件失败为 INVALID/2；REFERENCE_ONLY 始终不赋予通过资格。

# 约束

<a id="strategy--约束"></a>

## 一般 SWA 分支与性能

<a id="strategy--有限mask哨兵不能代替概率mask"></a>
### 有限mask哨兵不能代替概率mask

若masked score用`-FLT_MAX`参与行最大值，不能仅假设exp后自然下溢为0。合法有限score及sink也可能等于`-FLT_MAX`，此时被mask列的`exp(-FLT_MAX-rowMax)`为1。必须在exp后、FP32分母归约和P cast之前显式将所有窗口外/尾块列置0，或使用经证明等价的排除归约方案；无效query行仍遵守独立零输出语义。不能通过限制scale到测试值来掩盖此问题。

可复现边界：FP16的B=S1=S2=N1=N2=1,D512，Q全1、KV全-1，`scale=FLT_MAX/512`、FP32 sink为`-FLT_MAX`。QK=-512，缩放后score精确为有限的`-FLT_MAX`，有效KV与sink各贡献1，O=-0.5。若N64 tile的63个padding列错误贡献分母，会得到约-1/65。此例检查mask的数学边界，不要求用极值覆盖全部性能集合。数值域、普通随机用例和性能子集需要分开设计。

动态核数查询有真实链接依赖。在 CANN 9.2.0-beta.2 中，`PlatformAscendCManager::platformInitMtx/platformInfo` 及初始化实现位于 `lib64/libtiling_api.a`，仅链接 `libplatform.so` 会出现共享库构建成功、`dlopen` 未定义符号失败。使用该公共类的 host 目标须链接匹配 SDK 的 tiling_api 及其依赖，独立 metadata 库也须具备依赖；不能为了绕过链接失败改成硬编码卡的核数。当前 SDK 的 ASCEND 可执行探针若在链接 `libtiling_api.a` 后报 `undefined symbol: CheckLogLevel`，用 `nm` 核实该符号由同一 CANN 安装的 `libnnopbase.so` 提供，并将 `nnopbase` 加入该目标的链接依赖；修改后重新配置、链接并保留完整日志。构建后先在已初始化环境实际加载两个共享库并查询 plan，再进入完整设备验证；用 `nm`/`ldd` 检查公开符号与未解析依赖。

混合核硬件同步基址在当前SDK公开头`acl/acl_rt.h`由`aclError aclrtGetHardwareSyncAddr(void** addr)`取得。桥接层检查本次调用返回值，将地址作为launch控制参数传递并在设备入口设置`SetSyncBaseAddr`；它不是attention参数、不是用户metadata签名的一部分，不应跨调用缓存。正式OPC路径如由生成wrapper设置，先检查实际wrapper与tiling/entry协议，避免遗漏或重复使用过期地址。优先使用当前公共API，不因旧示例含已废弃的runtime符号而手工声明并硬链接它。

metadata 的任务划分、窗口起点和地址推导凡依赖本次 KV 有效长度、`cu_seqlens_ori_kv`、布局、页大小或页表时，生产端和消费端必须都使用本次值。只绑定 Q 长度的签名不足以证明同形状复用正确。构建固定 Q、交替改变 KV 长度或 PA 映射的 A/B/A 测试：逐次重新运行 metadata→main，验证表的任务边界、输出和 LSE，并检查 A 的两次结果一致；若某项不进签名，需证明它既不改变表字段也不被缓存的控制信息使用。

数学/接口只SWA并不意味着只一个shape。选中性能子集的batch、seq、head、dtype都不得成为数据常量；metadata负责从本次输入派生任务。保持完整API并显式拒绝未实现的CSA/HCA，不代调已安装手写metadata或主核。正式接口与诊断C ABI共用本轮数学与metadata，不出现各自不同的隐藏实现。

默认逐case速度比不低于手写0.8即可满足普通生成任务的性能最低门槛；优化研究的用户指定轮数仍完整执行。[验收口径](validation.md#coverage--性能验收与停止条件)改为5次预热、5次正式采样并取全部5次均值。这里列出的tile和分支在真实设备测试前仅是候选，不把某个样例结果当作其它head/dtype的性能保证。

<a id="prefill--约束"></a>

## 连续短窗口 prefill 推导

<a id="prefill--数据路径与存储层级"></a>
## 数据路径与存储层级

<a id="prefill--按所有view的最大需求分配共享片上空间"></a>
### 按所有view的最大需求分配共享片上空间

标准Pingpong/FullLoadA要求L1与L0的M、N相等，L0 K不大于L1 K；actual M/N不超过实际tile。不能假定L1M256/L0M128自动切M。容量须结合物理layout对齐计算，再对所有typed view取最大值。

| 资源 | 每槽容量公式（BF16 A/B、FP32 C） | 本候选容量 |
|---|---|---|
| L1 A | `max(QK.M*QK.K, PV.M*PV.K)*2` | max(65536,49152)=65536B，双槽 |
| L1 B | `max(QK.N*QK.K, PV.N*PV.K)*2` | max(98304,98304)=98304B，双槽 |
| L0 A/B | 各view的L0 tile字节数不超过各自槽；本候选按硬件容量平分双槽 | A/B分别32768B×2；QK B需24576B，PV B需32768B |
| L0 C | `max(QK.M*QK.N,PV.M*PV.N)*4` | max(98304,131072)=131072B，共享单槽 |

共享L1可把两个A槽放在前段、两个B槽放在其后，各view保留自身tile layout但受共享槽容量约束；合计`2*(65536+98304)=320KiB`。固定AtlasA2容量为L1 512KiB、L0A/B各64KiB、L0C 128KiB、UB 192KiB，仍应以本次SDK/芯片核对。M256/N192的FP32 L0C需192KiB，超过该容量；增大query tile还必须同步扩大窗口并集，不能只改M。

<a id="prefill--gm-workspace与ub生命周期"></a>
### GM workspace与UB生命周期

每GM任务槽依次分配score、P、numerator，分别为`M*N*4`、`M*N*2`、`M*D*4`字节。本候选是98304+49152+262144=409600B；两槽每核819200B。核c槽s的基址为`workspace+c*coreBytes+s*slotBytes`。核stride、slot stride和各段偏移统一以字节定义，typed tensor下标再以元素定义，地址乘加使用足够宽的整数。workspace分配覆盖实际launch的所有核，按最大核数预留也必须记录这一策略；核间区间不可重叠。

UB按存活区间而非变量名分配。单AIV的score需R*N*4=49152B，归约中间矩阵R*64*4=16384B，P需R*N*2=24576B，一级归约scratch需R*8*4=2048B。输出阶段的R*256 FP32紧凑块需65536B，可复用已经结束的score和归约中间区；其BF16输出需32768B，因此P/输出共用区要按32768B预留。两份R*4=256B分母、sink行向量、行最大值、广播区与sink指数/倒数等常驻小区不能被这些复用覆盖。按这些容量建立32B对齐的地址表可控制在192KiB内；具体偏移在生成时推导并检查，禁止凭已有数值偏移猜测存活期。

<a id="prefill--dma单位必须逐参数换算"></a>
### DMA单位必须逐参数换算

以下针对传统32B块单位的DataCopyParams重载；带元素count的DataCopy重载和字节参数的其他重载不能混用。blockCount是行数，blockLen/srcStride/dstStride是32B块数，stride表示本次block末尾到下次block起点的间隙，不是完整行距。

| 搬运 | blockCount | blockLen | srcStride | dstStride |
|---|---:|---:|---:|---:|
| GM numerator的R行D512中读取D256 FP32到紧凑UB | R=64 | `256*4/32=32` | `(512-256)*4/32=32` | 0 |
| 紧凑UB的R行D256 BF16写入GM输出D512的半边 | R=64 | `256*2/32=16` | 0 | `(512-256)*2/32=16` |
| 紧凑score或P连续搬运 | 元素count重载 | count=`R*N` | 不传块步长 | 不传块步长 |

两个D256块的GM起点分别加0或256个目标dtype元素，不能加256字节。DMA对齐、边界、源/目的实际长度与Vector repeat stride分别验证；相同数值32在不同参数中可能代表不同含义。

# 失败表现

<a id="strategy--失败表现"></a>

## 一般 SWA 分支与性能

- decode或chunked prefill精度错：先查Lk-Lq对齐与有效长度，不先改softmax容差。
- 只有stride padding用例错：查物理ld和分query切head地址，不把整个输入强行contiguous掩盖遗漏。
- ragged尾部NaN：检查无效query初始化、零行AIV通知、pad KV的有限值及LSE输出映射。
- 大head短序列慢：核对任务数/空闲核与重复KV搬运，尝试head分核而非只增大M。
- 跨参数扩展后挂起：查每任务描述的双槽寿命、实际R归约次数、owner索引parity和尾部free计数。

<a id="prefill--失败表现"></a>

## 连续短窗口 prefill 推导

| 表现或危险改动 | 优先检查的原因 |
|---|---|
| 首次运行正常，重复或动态输入后错/挂起 | 多owner重复设置event、stage索引在QK/PV切换时重置、未排空token、sink被固化或同步基址提前恢复 |
| 仅第二个D256或跨行结果错 | PV ld误用256、DMA stride误当完整行距、typed offset与字节offset混用 |
| 两个AIV输出重叠或只半边正确 | AIV核号未折算、part行偏移缺失、只有一侧参与flag |
| mask边界/head相关异常 | kv0尾部钳制遗漏、每query四行mask不全、移位64、共享32B块的masked写缺少局部顺序依赖 |
| 归约/广播后成组行错误 | repeat被当元素数、第二级归约repeat错误、Brcb只分配R元素、广播源block/repeat stride混淆 |
| 扩大tile后越界或静态检查失败 | L1/L0 M/N不匹配、L0C超容量、窗口并集未随T增加 |

减少API调用、字节数或任务数并不自动缩短关键路径。FullLoadA可能减少P重复搬运，也可能增加QK/PV调用和同步；WholeReduce、常驻mask、不同输出分组或任务分配都需独立测量。先辨认等待、搬运、计算中的主导成本，再作单因素实验；本页候选形状和流水不构成性能达标承诺。

<a id="prefill--从工作量建立性能假设"></a>
## 从工作量建立性能假设

对本页B=1、一个KV head的因果窗口，有用QK+PV MAC为`2*G*D*sum_t(min(t+1,W))`；无query尾块的密集实现MAC为`2*J*M*N*D`，J=S/T。两者之比刻画补齐计算，不能用实际工作冒充有效工作。若S>=W，窗口长度总和为`S*W-W*(W-1)/2`。本例有效MAC为2114191360，T32/N192候选实际MAC为3221225472，约1.524倍；这是算术事实，不是瓶颈结论。

对于P不驻留、PV分p=D/Dv次且每次重新读取P的实现，每任务显式GM指令字节为`12*M*D+(10+2*p)*M*N+4*N*D`：Q与输出各2MD，numerator写读共8MD，score写读8MN，P写2MN与读2pMN，QK/PV的KV读取共4ND。不计很小的sink/metadata访问；额外拆分或缓存必须按实际指令重新计数。M128/N192/D512/p2时为1523712B/task。该量不等于HBM流量，跨阶段/跨任务cache命中须另取证。

当N不变且没有尾块时，J*M=G*S，增大T不会减少总密集MAC；它减少KV重复指令字节和任务次数，同时改变片上容量、并发粒度和任务波次。等成本任务的波次填充率`J/(C*ceil(J/C))`仅是静态调度指标，不能当作硬件利用率。新增候选先写清它改变哪一项，以及可能恶化的代价。

每次优化记录可证伪的瓶颈假设、所需观测指标和单因素改动。怀疑搬运受限，检查实际内存层级流量、MTE占用及等待；怀疑Cube受限，同时检查有效/实际计算、Cube活动和消费者等待；怀疑Vector或同步受限，检查AIC等P、AIV等score/numerator及流水启动排空。没有相关指标时明确写“瓶颈未确认”，总device时间不能分解成阶段耗时。阶段探针只作诊断，最终效果用同环境完整融合核的release样本判断；缩小N、增大Dv或增加槽数不能只凭某项静态成本下降认定有效。

# 验证方法

<a id="strategy--验证方法"></a>

## 一般 SWA 分支与性能

性能子集保留调用者指定原JSON和ID范围的参数、值域、golden，并逐条记录分支、shape/stride、有效长度、误差和完整device样本。完整功能验收还须从合法域构造未见shape、不同有效长度、同址Q/KV/sink变化及尾部/零行核探针，覆盖所有实现分支。PA和LSE属于一般SWA必需功能，未实现或未运行时保持未完成；指定子集通过不能替代。无sink是诊断ABI可选扩展，不能与正式arch22接口要求非空sink混淆。

所有优化都保留同源编译、全量精度、动态输入、边界及sanitizer证据。严禁把CPU标杆或参考cache放到设备算子forward中；参考cache只能用于测试端，绑定输入字节、golden、dtype和数值路径。

[^strategy-swa-contract]: [共享KV窗口计算](computation.md)。

<a id="prefill--验证方法"></a>

## 连续短窗口 prefill 推导

生成后先静态核对布局地址、容量、metadata覆盖和事件生命周期，再用实际输入执行完整精度、finite、重复一致性及同范围release性能测试。检查窗口起始、稳定区和尾部钳制，四个head、两个AIV及两个D256半块；用不同Q/KV/sink的A/B/A/B序列检查动态依赖。0、1、2、3、7个本核任务分别覆盖空区间、启动、首轮复用及排空，确认每个有效输出恰好写一次、所有set/wait配对。

对本次相同源码单独构建instrumented库，执行动态A/B/A/B的racecheck和memcheck，核对目标核检查次数、ERROR、Warning及明确结束记录。工具进程成功或精度PASS不能替代日志判定；保留实际报告和源码/二进制/输入输出哈希，不过滤告警。instrumented时间不混入release统计。

性能报告给出硬件、实际核数、预热数、样本数、device时间统计和测量范围，metadata host planning与H2D准备单列。只有基线与新实现计时范围一致时才计算加速比。每轮生成独立验证，不引用知识页或历史实现的通过记录作为本轮证据。

[^prefill-shared-kv-formula]: [共享KV与sink的计算定义](computation.md)。

<a id="acceptance--验证方法"></a>

## 性能证据的解析与逐例判断

验收前实际检查原始 trace 的主核归属、5条样本和每次完整计算的组成。新增 pack、清零、归并等必需设备核
必须纳入与手写完整数学计算可比的边界，metadata、控制传输和端到端另记。不能仅修改报告 scope
把不同计时范围变成可比。零工作没有 kernel 记录时用 null 和 NO_DEVICE_KERNEL_RECORDS，不伪造0µs。
