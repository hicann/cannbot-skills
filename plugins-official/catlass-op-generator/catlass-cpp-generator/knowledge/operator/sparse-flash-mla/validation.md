---
type: operator
title: SparseFlashMLA 用例构造与完整验证
description: 基础矩阵、负例、动态诊断 ABI、输入与标杆准备、ABAB、流水、sanitizer、原 ATK 和性能门禁。
tags:
- catlass-cpp
- sparse-flash-mla
- validation
- cases
- precision
- performance
- sanitizer
- swa
- abi
- pipeline
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
- id: coverage-development-contract
  resource: knowledge/operator/sparse-flash-mla/development.md#stages
  title: 待覆盖的Stage和分支
  kind: contract
- id: suite-swa-supplied-golden
  resource: knowledge/operator/sparse-flash-mla/interface.md#api
  title: 原标杆与正式入口映射
  kind: contract
- id: suite-pa-fixture-construction
  resource: knowledge/operator/sparse-flash-mla/validation.md#diagnostics
  title: PA页表与物理容量准备配方
  kind: repository
- id: pipeline-validation-pipeline-spec
  resource: knowledge/operator/sparse-flash-mla/pipeline.md#protocol
  title: SWA三actor双槽协议
  kind: contract
- id: pipeline-validation-structure-spec
  resource: knowledge/operator/sparse-flash-mla/pipeline.md#structure
  title: 片上资源和行块流水
  kind: contract
- id: pipeline-validation-swa-validation
  resource: knowledge/operator/sparse-flash-mla/validation.md#coverage
  title: 原ATK与完整设备5+5验收
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

- [功能覆盖与正式验收](#coverage)
- [动态诊断 ABI 与完整输出](#suite)
- [流水源码与设备验收](#pipeline-validation)
- [用例生成与覆盖审计配方](#case-recipes)
- [动态诊断运行器的实现步骤](#diagnostics)

# 接口与概念

<a id="coverage"></a>

<a id="coverage--接口与概念"></a>

## 功能覆盖与正式验收

与[开发阶段](development.md#stages)配合使用。目标是让每个算法分支、地址边界和流水状态得到实际运行验证；数量本身不是质量证明。保留用户标杆，额外使用独立的逻辑索引oracle核对选择集合。不得因为某个失败难定位而缩小K、G或KV后替代原用例。

<a id="suite"></a>

<a id="suite--接口与概念"></a>

## 动态诊断 ABI 与完整输出

[^suite-swa-supplied-golden]: [原标杆与正式入口映射](interface.md#api)。

本文件冻结本轮可移植验证器与每次独立生成工程之间的诊断 ABI。它不是正式 ACLNN/PyTorch 接口的替代品。生成者必须同时交付正式 main、metadata 入口，注册/构建产物，以及下列直调诊断符号；正式入口按 [完整接口](interface.md#api) 保留可选参数槽、属性和两个返回值，当前 SWA 支持域之外明确拒绝。

<a id="pipeline-validation"></a>

<a id="pipeline-validation--接口与概念"></a>

## 流水源码与设备验收

本页用于普通 SWA design/implementation/validation 的语义检查，不增加工作流阶段。规范化的文档或schema只能证明结构完整；最终算子的五阶段和流水必须从实际编译源码、运行入口、设备结果核验。只实现计算公式与实现高效流水是两个不同检查项。

# 用法

<a id="coverage--用法"></a>

## 功能覆盖与正式验收

普通SWA补充执行[流水与结构验收](validation.md#pipeline-validation)，并读取[完整握手协议](pipeline.md#protocol)和[片上结构](pipeline.md#structure)核对最终实现。该检查属于当前validation，不能新建优化阶段。原ATK数值通过不能替代跨任务流水与逐例0.8x性能验收。

按[用例生成与覆盖审计配方](validation.md#case-recipes)在生成工程中构造逻辑manifest，**不是可直接传给ATK的json**。生成工程须实现adapter：按清单分配张量、布局转换、产生页表和索引、调用两个新算子及用户标杆。ATK适配时将case存入其实际schema，并核对筛选的是数组位置还是case_id，禁止空集合显示PASS。

1. 在本次工程的测试目录生成逻辑 cases manifest，记录 seed、布局、有效长度和覆盖标签。
2. 按[用例配方](validation.md#case-recipes)检查矩阵、实际 full-K、页表和负例构造；通过后才交给运行适配器。
清单包括480条基础正例（3模式×4布局×2dtype×20 profile）、定向语义用例、单因素负例。每条带seed、有效长度、K、G、两组页长、索引构造、sink/scale/LSE、功能/压力/性能分层。压力用例可按ID分片运行，仍属于完整验收；不能只跑functional后声称整个清单通过。性能子集单独标记，从同一批实际精度通过的输入取样。

清单生成不调用NPU，不代替精度验证。静态审计核对清单及full-K条件；结果对账检查逐例执行记录是否齐全，不能证明记录中的运行真实发生。报告中保留实际日志和输出文件，Reviewer抽取每个模式/布局/G分支重跑。

<a id="suite--用法"></a>

## 动态诊断 ABI 与完整输出

生成者在冻结公开契约后读取本页，以相同 descriptor 连接 main 和可独立调用的 metadata。运行器按[动态诊断实现步骤](validation.md#diagnostics)在生成工程中实现；本知识包不附带脚本。只运行本页的直调不能宣布完整正式接口通过。

脚本在`result.json.execution_stage`持久化设备分配、plan、metadata、主核提交、同步及比较进度。超时后先读最后检查点：`submit_main`尚不能证明提交返回，`synchronize_main`表示提交已成功返回而同步未完成。CPU标杆完成也不能证明设备核已经启动；旧工具版本未逐阶段保存时，需用限时进度/调用栈诊断补证。诊断重跑只针对当前失败case，设置有强制终止后备的时间上限，保留后续case为`NOT_RUN`。进度写盘位于精度流程及profiling开始之前，不放进被测launch回调或5次采样循环。

<a id="pipeline-validation--用法"></a>

## 流水源码与设备验收

在最终 `docs/validation.md` 增加“流水与结构验收”表，逐项写：实际编译文件与函数、触发条件、实际资源/事件、验证命令与证据、PASS/FAIL/NOT_RUN。源码改变后使受影响旧证据失效，不能只更新report日期或引用另一个.so的结果。

先按[接口页的交付形式](interface.md#api--用法)记录实际调用层，避免将“正式”一词用于不同目标而混淆验收：

| 实际交付层 | 需要证明的入口链与产物 |
|---|---|
| 原ATK所用完整PyTorch接口 | main 28参数、metadata 27参数的真实`torch.ops`/wrapper接线，两者都调用本轮实现；本轮独立metadata实际产表，CATLASS主核实际消费；原ATK、Fake/Meta及完整公开域按选定契约验证。包装内使用本轮C ABI不自动使完整公开入口降为简化诊断，也不自动证明ACLNN包互通。 |
| 简化descriptor直调诊断 | 只证明该descriptor覆盖的行为。即使对照golden通过，也不能替代未实现的公开参数、默认值、dispatch与原ATK接线。 |
| ACLNN算子包互通 | 在已要求此交付层时，额外证明main/metadata四个ACLNN导出、注册与安装、真实AI Core和AICPU执行；host产表或仅存在同名torch schema不能替代AICPU。具体工程由已有工作流和专用工程页提供。 |

不得因选用诊断host planner而省略独立metadata，也不因专用工程页包含ACLNN经验，就在未要求包互通的任务中擅自改变接口交付范围。报告明确区分`PyTorch公开入口通过`、`descriptor诊断通过`、`ACLNN/AICPU包通过`，并绑定各自实际二进制和计时范围。本文后续“正式入口”指任务已冻结的公开交付层，不单凭`.so`或包目录的名称判定。

<a id="pipeline-validation--算子算法"></a>
## 算子算法

明确普通 SWA 的 V0适用性，C1/V1/C2/V2公式、sink注入位置、P舍入点与LSE写出位置。整union一次softmax时“没有多S2在线u累加”不是漏算；PA独立pack不是稀疏V0；metadata不是数学阶段。附本轮解码的实际M/N、T/H和有效区。

<a id="pipeline-validation--分核策略与基本块切分"></a>
## 分核策略与基本块切分

证明 metadata→main 的每个合法 `(b,t,h)` 写者唯一，两 AIV 的逻辑核/part映射与启动类型匹配，零行侧不会破坏通知。任务枚举和性能分支按shape/layout/dtype，禁止case ID路由。CPU枚举coverage有价值，但不能代替NPU消费本轮metadata。

<a id="case-recipes"></a>

<a id="case-recipes--用法"></a>

## 用例生成与覆盖审计配方

本节定义可由生成工程实现的逻辑用例构造、清单审计和结果对账方法。知识包不附带执行脚本。
先冻结本次支持模式和输入域，再在生成工程内实现需要的适配器；只要求 SWA 时只选 SWA 相关轴，
压缩和稀疏模式的配方保留供相应任务使用。逻辑 manifest 不是 ATK 可直接读取的 JSON。
用例生成、静态覆盖审计、真实设备运行、运行记录完整性检查是四件独立的事。

<a id="diagnostics"></a>

<a id="diagnostics--用法"></a>

## 动态诊断运行器的实现步骤

本节给出生成工程自行实现动态诊断运行器的步骤与结果约束。它只依赖调用者提供的未改动数学标杆、
用例文件、本轮主库/metadata 库和实际设备；不依赖知识包中的可执行脚本，也不读取旧工程的私有缓存。
结构检查、ABI 描述和用例选择可只使用标准库，数值阶段才加载 Torch/NumPy/torch_npu。
输入参数包括 golden、case JSON、显式 case_id 列表或 all-cases、主库、独立 metadata 库、
逻辑 device、round ID、新输出目录、CPU 线程数、数值 seed 和可选 profiling 开关/完整 kernel 选择器。
CPU 线程数取正数，例如8，interop可设1；记录设备可见性环境，不能把逻辑号当物理号。

用例必须明确选择，禁止默取前十条。解析 ATK 外层 inputs 中唯一的 smla_case_spec，或直接的 cases 数组；
支持 range_values 的字符串/对象两种形式，核对内外 ID 一致、唯一、无缺例。逐 ID 与全部用例选择互斥，
选择顺序保留。只检视输入时报告 dtype/layout/batch/head、有效长度、存储长度、stride 类别、LSE 和零工作，
不产生设备 PASS。

# 代码模式

<a id="coverage--代码模式"></a>

## 功能覆盖与正式验收

<a id="coverage--算子算法"></a>
## 算子算法

每batch记录 `Lq,Lo,Lc,residual`；压缩模式默认构造 `Lo=r*Lc+residual, 0≤residual<r`，TND前缀和由真实长度累加，BSND使用每batch最大容量并写seqused。cmp和ori前缀不能混用。TND中相邻batch长度故意相差1/2/7，不能全相等伪装ragged。

CSA不能只把索引tensor宽度K改大：

```text
C_first = max(0, min(Lc, floor((Lo-Lq+1)/4)))
full-K要求 C_first >= K，且前K槽全有效、无-1、ID互异
充分构造：Lc >= K + ceil((Lq-1)/4)，Lo=4*Lc+residual
clipped-K要求 0 < C_first < K，另设C_first=0的语义探针
```

full-K和clipped-K必须同时存在。K集合为 `1,2,15,16,17,127,128,255,256,511,512,513,1023,1024,1025,1537,2048,4096,8191,8192`；full-K的1537必须实际进入第4个512块，8192实际处理16块。数据张量物理容量大并不等于有效选择量大。报告记录实际scan_count/selected_count/有效tile数，不仅记录K标签。

<a id="coverage--数据路径与存储层级"></a>
## 数据路径与存储层级

PA_BBND分别生成ori/cmp页表，覆盖打乱的独占物理页、跨batch共享前缀、同batch重复映射物理页。独占页只是一类构造方式，不能要求物理页数不少于全batch逻辑页数之和，也不能要求页号唯一。页内尾部填入有区分度的哨兵值。页长交叉16/32/64/128/256/512/1024，两组页长可不同。共享物理页在每个逻辑位置仍是一个独立的attention位置，禁止按物理页去重。相同逻辑KV通过BSND/TND/PA分别存放，必须得到相同逻辑选择和数值输出；不能仅使用恒等页表测试。

若标杆附带的随机生成器只会分配独占页，保持其数学forward不改。用足够的临时页数完成其他输入生成，再恢复实际物理页数、显式替换物理KV与页表，从实际量化后的物理数据按页表还原逻辑KV送入标杆。生成参数、实际DUT参数、页表及输入哈希分别记录；不能把临时扩大后的KV传给DUT冒充共享页测试。定向例：页长16、两个batch各129个token，共享前8页、各有1个尾页，18个逻辑页使用10个物理页；另用1个物理页重复映射9次覆盖129个token。交叉Q布局、dtype、LSE和页/表行stride padding。

索引按[稀疏行构造配方](validation.md#case-recipes)确定性构造，adapter与oracle共享数据而非共享设备实现。模式包括：full、clipped、permuted、duplicate、future、early_stop、empty；未来ID必须仍在物理Lc内但≥C，-1前的元素有效，-1后的故意放有效ID以检验终止。重复ID保留权重，不能用set或布尔mask删除。对外宣称支持ori indices/topk_length之前要为其单独启用路径和用例；标杆入口若忽略它们，不能用helper的能力虚报覆盖。

<a id="coverage--分核策略与基本块切分"></a>
## 分核策略与基本块切分

| 类别 | 构造和需要发现的问题 |
|---|---|
| 三模式×四布局×两dtype | SWA、CFA、SCFA；BSND+BSND/TND+TND/BSND+PA_BBND/TND+PA_BBND；fp16/bf16，每个组合20profile |
| query/head/任务数 | Lq=1/2/3/7/15/16/17/31/32/63/64/127/128/129/255/256/257/511/512/513；G=1..128的2幂；B=1/2/4。专设一个核顺序处理多个task、任务少于核数及M跨块尾 |
| 长KV与长流水 | SWA长物理KV检地址；CFA压缩可见量跨512、1024及多块；CSA full-K8192。至少覆盖一次槽环绕、两次环绕和任务切换，不拿长度128的smoke代替 |
| 稀疏语义 | -1在首/中/末槽，重复ID、乱序ID、未来ID跳过、空稀疏列表、K奇数成对读取、索引行跨query/batch |
| 双路与sink | 两路都非空、一组为空、两组空（契约允许时）；非零/正/负sink、不同比例scale；sink只贡献共同分母且仅一次 |
| 数值 | 随机非零、常量、交替符号、近均匀logit、大但有限logit；低精度P cast；LSE开启关闭，逐batch/head统计 |
| padding与布局 | 有效长度不整齐；物理尾填大值检查mask；两组页长不同、随机页表、页边前后1；输出轴次序和原始dtype |
| metadata | FA-only正确覆盖、关闭核全零、保留位清零、上限核数、少任务、多任务、ragged、改变长度后重生成；启用FD后加2/3/多片和空片归并 |
| 并发与复用 | 同一case至少20次，不同长度A/B交替至少20轮；同一workspace复用及合法多stream独立workspace；不同seed；sanitizer on/off |
| 参数拒绝 | 单因素变异K0、8193、>8192非512倍数、q不支持dtype、缺cmp输入、非法layout、非法ratio；主入口与metadata分别验证 |

小规模精确语义用例用于定位；基础矩阵用于功能覆盖；大G/长KV/多batch用于压力和性能。不要要求每个边界都做完整笛卡尔积，但必须输出各分支命中统计。添加新fast path后补阈值两侧和交叉条件，不能只复用旧20profiles。

<a id="coverage--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

按逻辑坐标先恢复每batch的Q和两组KV，再逐query/head列出ori窗口及cmp选择ID，用列表保留重复。数学oracle用 `J=ori⧺cmp`、一次softmax、sink的零value项，验证共同分母；流式oracle按冻结tile顺序、FP32 m/l/u、低精度P实现，验证数值路径。两者应数学相符但不强求低精度逐位一致。数值标准沿用调用者约定或仓内精度规范，在01/02阶段固定，不能出错后临时放宽。

若使用双标杆：先把输入量化到实际输入dtype，分别生成高精度参考和目标数值路径参考；被测结果与两者计算同一组冻结指标。比较器不能让两份参考都来自被测kernel。记录O和LSE的所有元素、shape、dtype，最大/平均/均方误差和失败数量及分区；不能只看一个均值或第一个输出。

标杆有已知不一致时用区分性输入确定当前契约：例如Lo31/Lc7/r4/Lq4/residual0，按Lo计算首query C=7，按Lc*r+residual计算为6。此例属于语义判定，不能自动塞进要求长度关系一致的API正例。标杆TND输出dtype/LSE轴与设备ABI不同的，在wrapper映射表明确转换，不能无说明比较错误轴。

debug检查点按开发页S0–S6：metadata任务集合、S1逻辑ID/KV、S2 score、S3 m/l/alpha/P、S4 delta、S5 u/O/LSE。第一个不同的位置决定修复阶段。检查点本身不能增加全局同步来掩盖原先竞争；精度通过后关闭dump再次运行。

<a id="suite--代码模式"></a>

## 动态诊断 ABI 与完整输出

<a id="suite--算子算法"></a>
## 算子算法

采用自然 64-bit ABI 对齐，禁止 `#pragma pack`；所有 stride 单位都是元素。枚举 `dtype: FP16=0, BF16=1, FP32=2, INT32=3, INT64=4`；`layout: BSND=0, TND=1, PA_BSND=2, PA_TND=3`。后两者仅用于 KV，物理 shape 均为 `[physical_blocks, block_size, N2, D]`，分别配套 BSND/TND query，逻辑 batch→page 映射见 ori_block_table；正式 wrapper 使用文档中的 `PA_BBND` 名称。

```c
#include <stdint.h>
typedef struct {
    void *data;                 /* device address; NULL means absent */
    int32_t dtype;
    int32_t rank;               /* 0 for absent, otherwise 1..4 */
    int64_t shape[4];           /* unused slots zero */
    int64_t stride[4];          /* actual native tensor strides */
} SwaTensorView;                /* sizeof = 80 */
typedef struct {
    const int64_t *data;        /* HOST values; NULL iff count == 0 */
    uint64_t count;
} SwaHostIntArray;              /* sizeof = 16 */
typedef struct {
    uint32_t abi_version;      /* 1 */
    uint32_t struct_bytes;     /* sizeof(SwaDescriptor) = 632 */
    int32_t layout_q;
    int32_t layout_kv;
    int32_t return_lse;        /* 0 or 1 */
    int32_t reserved;          /* 0 */
    int64_t batch;
    int64_t heads_q;
    int64_t heads_kv;
    int64_t head_dim;
    double softmax_scale;
    int64_t ori_mask_mode;
    int64_t ori_win_left;
    int64_t ori_win_right;
    SwaTensorView q;
    SwaTensorView ori_kv;
    SwaTensorView sinks;       /* FP32[N1], nullable; absent means no sink */
    SwaTensorView out;         /* same logical shape and dtype as q */
    SwaTensorView ori_block_table; /* INT32[B,max_pages], nullable without PA */
    SwaTensorView lse;         /* FP32; absent iff return_lse == 0 */
    SwaHostIntArray cu_q;      /* B+1 prefix including initial zero, or absent */
    SwaHostIntArray cu_kv;
    SwaHostIntArray used_q;    /* B effective lengths, or absent */
    SwaHostIntArray used_kv;
} SwaDescriptor;
typedef struct {
    uint32_t abi_version;      /* caller sets 1 */
    uint32_t struct_bytes;     /* caller sets 32 */
    uint32_t core_count;       /* callee queries actual platform */
    uint32_t reserved;         /* 0 */
    uint64_t metadata_words;   /* number of int32 words */
    uint64_t workspace_bytes;  /* zero is legal */
} SwaPlan;

int smla_swa_get_plan(const SwaDescriptor *, SwaPlan *);
int smla_swa_metadata(const SwaDescriptor *, int32_t *host_table,
                      uint64_t table_words);
int smla_swa_launch(const SwaDescriptor *, const void *device_metadata,
                    void *device_workspace, uint64_t workspace_bytes,
                    void *stream);
const char *smla_swa_last_error(void);
```

<a id="suite--数据路径与存储层级"></a>
## 数据路径与存储层级

Main library exports all four symbols. Independently built metadata library exports get_plan, metadata and last_error, using this identical descriptor and plan. Zero return means successful submission/completion appropriate to the entry. Nonzero returns must supply a diagnostic string. Callers retain all descriptor host arrays until each synchronous host call returns; asynchronous device kernels cannot dereference host pointers. get_plan/metadata may inspect tensor metadata but must never read q/KV/sinks device values. The fixed scheduling table encodes the plan and per-core cursors. Per-batch lengths/offsets may reside in a separately sized device control region derived from the current descriptor; include its capacity, upload and lifetime in the workspace/launch contract. Do not require every batch record to fit in the fixed table or derive a private batch limit from it. The main kernel must actually consume this generated metadata and the corresponding current control data.

Metadata initializes every returned word deterministically, including reserved/disabled slots. The caller allocates exactly plan.metadata_words, which cannot exceed the formal 1024-word table, and uploads the produced table before launch. The formal public output remains INT32[1024]. get_plan has no side effects or launches; metadata measures CPU planning separately from H2D. launch consumes caller workspace and current stream and never substitutes cached outputs. All three entries validate ABI version, struct size, layout, dtype, shape, stride and effective lengths independently. No case ID is present or permitted as a dispatch key. `q.data` points at its logical first element; storage offsets are already included in this pointer.

<a id="suite--分核策略与基本块切分"></a>
## 分核策略与基本块切分

The first-ten subset covers BF16/FP16, BSND/BSND and TND/TND, dynamic B, ragged lengths, N1 powers of two 1..128, N2=1, D=512, mask4 and window127/0. Full SWA additionally requires both PA layout pairs and the LSE switch; these branches cannot be rejected merely because the performance subset does not use them. Padded KV native strides must survive CPU→NPU transfer. Nullable sinks are a diagnostic extension and do not remove the formal interface's required sink. Enabled LSE uses formal shapes BSND `[B,N2,S1,N1/N2]` or TND `[N2,T1,N1/N2]`.

<a id="suite--sanitizer与实际分配生命周期"></a>
### Sanitizer与实际分配生命周期

先核对被测主库确实采用同一数学源码的instrumented构建；mssanitizer返回0不保证被包装的Python程序成功，也不保证没有WARNING。同时检查子程序结果、工具日志、实际kernel名称和最终输出。正常release性能不使用instrumented包。

OPC正式包的插桩包含编译与最终链接两个阶段。只向bisheng传`--cce-enable-sanitizer`可能产生`__sanitizer_report_*`调用，却在OPC链接时缺少对应runtime。使用本次公开SDK支持的OPC debug配置，并检查真实生成的OPC子命令和输出object。9.2.0-beta.2的`ascendc_gen_options.py`并不把任意`--op_debug_config=sanitizer`自动归入debug字段；该参数可能被原样写入编译选项。应读取当前parser，保留原有include/架构/优化/同步选项，在独立构建目录按其`custom_opc_options.ini`协议设置`<OpType>@@--op_debug_config=sanitizer`，并核对实际编译命令消费了它；不修改公共SDK或release配置，不以空桩补齐sanitizer符号。安装时使用本次生成的object及匹配json，执行SDK的simplified-key和kernel-config生成步骤；不能把debug object改成release文件名后沿用不匹配的配置。

逐个目标kernel统计工具的开始、结束、无错误结束与finding；数值通过、进程返回0、racecheck无竞争都不能消除memcheck的寄存器恢复或越界告警。正式OPC wrapper的入口初始化与用户代码退出恢复分别审查，核对主路径、零任务、关闭核和提前返回的排空与恢复位置；不要仅验证独立直调入口的epilogue。框架Fill等辅助核告警和未插桩skip完整保留、单独归属，不能把它们算成目标核错误，也不能把目标核的warning归为框架噪声。无明确根因的finding保持未解决，不因性能或数值结果好而标记clean。

对GM越界告警记录本次logical data pointer、storage pointer、storage offset、shape/stride、逻辑字节与实际storage字节，并计算报告指令实际访问的半开区间。不能因为精度通过忽略越界，也不能把合法短LSE写扩大到32B来消除告警。

在当前9.2.0-beta.2与torch_npu组合中，跨case释放后复用相同地址可能使memcheck报告与当前分配不一致的LSE写界限。已观察到合法48B LSE的`[base+16,base+32)`、`[base+32,base+48)`两次16B写被报越界，而独立进程、保持前序分配存活、以及`PYTORCH_NO_NPU_MEMORY_CACHING=1`的原释放生命周期均无告警、同核O/LSE通过。遇到相同现象时保留默认环境原始告警和当前地址证据，在确认目标torch_npu支持该变量后仅对sanitizer诊断进程禁用缓存复测，报告该配置下的真实结果；不能把对照通过写成默认环境也无告警，其他越界仍逐项排查。性能采集恢复正常分配配置，不能混用诊断环境的时间。

实际核数由 get_plan 查询平台；metadata 的任务范围、关闭核及保留位须确定性写出。分块、流水和 metadata 表内部布局由生成工程设计，不能按 case_id 分派，也不能依据某例 shape 把其他 descriptor 字段视为无效。主核实际消费配套表，长度/布局改变后重新产表。

<a id="suite--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

The project diagnostic runner loads the supplied unmodified `golden.py`, requires RUN_MODE=1, calls its gen_data and GeneralizedSFA.forward on CPU, and compares every output element. It preserves the original E cast before PV and denominator accumulated before that cast. BSND invalid/padding rows must be exactly zero as returned by golden; the verifier never zeros DUT results. TND golden returns FP32 containing quantized output values; DUT output retains q dtype and numerical comparison promotes both to FP32 without modifying golden.

The direct diagnostic recipe has its own deterministic structural/value seeds and stride-padding perturbation. They help reproduce A/B/A/B runs on identical device addresses; they do not assert byte-for-byte parity with every caller-provided ATK adapter. Compare its input construction with the supplied ATK script before treating the diagnostic as coverage of the same effective input. All four calls receive NaN-filled output, then full output comparison and byte-hash repeat checks. A fresh per-round output directory may cache only its own CPU input and golden tensors, with byte hashes, golden/tool/case identity and an explicit round ID; no previous private cache is read.

The direct diagnostic uses its own fixed BF16/FP16 elementwise tolerance; all elements must pass and be finite, and output padding is exact zero. This tolerance is not the caller's formal ATK comparator. Read the supplied script's actual thresholds, reference paths and input construction, then run that unmodified script for formal acceptance. The direct tool reports that gate as NOT_RUN and must not weaken its own policy after observing a DUT.

<a id="pipeline-validation--代码模式"></a>

## 流水源码与设备验收

<a id="pipeline-validation--数据路径与存储层级"></a>
## 数据路径与存储层级

| 验收项 | 最终源码必须能定位的证据 | 不足以通过的证据 |
|---|---|---|
| C1/V1/C2/V2真实计算 | 实际调用图及各函数体；输入输出shape/stride与舍入 | 空函数、只写stage名称、CPU模拟 |
| 混合核跨任务流水 | 同一主kernel的AIC/AIV分支、i/i-1序列、至少两代同时存活 | 同一stream依次launch四个全网格kernel |
| 两AIV分工 | core/part映射、不同的行区间和两侧flag参与 | design说two-AIV，源码每AIV仍独立跑全work |
| 持久Cube owner | 构造在任务循环外、共享typed views、事件唯一归属及最终析构 | 只有Resource共用而两个BlockMmad各自管理重叠事件 |
| PV有效K与物理尾块 | `Kvalid/kc/kPhys/ldP/ldKV`逐阶段映射、copy/MMAD实际重载与padding责任、跨K0/K1边界及PA页尾 | 仅改一个K常量；实际shape全部用对齐值；以P=0解释未初始化V；K=129单例通过 |
| L1/K局部预取 | 下一K搬运与当前K计算的buffer/event映射 | 单纯类型名含Pingpong |
| Q/P跨输出块复用 | 装载条件与最后消费者；Q按N、P按D复用 | 每BlockMmad重读A却在文档写“一次装载” |
| 有界GM slot | 每核slot寻址和正确free，workspace与core而非W线性相关 | 全体work各分配一份X/P/U |
| Vector行块处理 | 批量mask/归约/sink广播/一次倒数及UB依赖 | 每row256列scalar循环、逐row GM GetValue |
| Vector子块搬运重叠 | 输入/输出子槽独立，下一块预取，free延迟到同槽复用；P/U外层通知覆盖全部子块 | 有两块UB但每笔输出后立即等待完成，或只按类型名认定预取 |
| LSE与行状态生命周期 | 直接读LSE UB的MTE3完成后，新V1才重写该槽；具体本地free/ready及尾部计数 | 仅有两个LSE槽，或用外层U_FREE发送当作本AIV已经等完MTE3 |
| L0C实际缓冲粒度 | C槽容量、一个MN结果全部K累加的同槽归属、FIX和下一覆盖的完整依赖 | L0A/B有ping-pong就声称L0C也有；按K翻转C；128KiB结果写进64KiB槽 |
| PA可实现路径 | 按页段直接L1或有限pack协议；真实页表/stride | 只预留PA参数、identity页表才正确 |
| 可恢复退出 | 末task与空task排空、事件消费、mask/sync责任 | 仅第一次运行不挂起 |

LSE 事件的源码审计要逐个核对以下动作：采用结构页的独立两槽方案时，构造每槽只预置一次 `LSE_FREE`；V1 第一次写某槽前等待其 free；V2 在所有 LSE Vector 写入之后 Set/Wait `LSE_READY` 再发起精确长度的 GM copy；copy 之后 Set `LSE_FREE`，下一代同槽写入前或 owner 析构时 Wait。`V_MTE3` 的 ready 是运行时现产现消，无需构造时预置；例如 `Set V_MTE3/2 -> Wait V_MTE3/2 -> copy` 可以正确保护本次 V→MTE3 读取，但不证明旧 copy 已结束。真正要查的是 copy 之后的 `MTE3_V` free 及下一次写入前的 Wait，或另外一条覆盖相同最后 reader 的完成链。不能因编号未在构造预置就判 ready 错误，也不能因某个 LSE case 数值通过就跳过复用审计。

任务游标每次跨 batch 时在 `batch < B` 的前提下读取 control 并跳过 storage-empty batch，`local_i`、slot、generation 只对实际 work 递增。核对 producer 的 task count 与 decoder 的边界一致；中间多个空 batch、尾部空 batch 和全部空 batch 分别演算，不能只验证 `used_q=0` 而 storage 仍非空的任务。若发现无界 while，需要先证明合法 producer 是否可达越界状态；不能直接将畸形私有表当作公开合法输入。若提供错误表检测，三方应一致退出或由 host 在 launch 前拒绝，不能单个 AIV 跳过通知造成配对核死锁。

辅助槽的边界以[普通契约](interface.md#swa-contract)逐项表为准。`cmp_block_table` 在 SWA 中未被数学读取，且 `cmp_kv=None`，因此不能要求其页号映射到不存在的 cmp 存储，也不能一概拒绝合法非空辅助表。按该槽公开 rank/dtype/shape 校验，记录未使用，测试合法非空表不改变 O/LSE以及错误 rank/dtype/长度被拒绝。真正被消费的 `ori_block_table` 才需要按本次 ori 物理存储核验所访问页号、覆盖有效长度的容量和实际 stride。不要从 ori 页表复制校验条件后扩大 cmp 辅助槽的限制。

Q/P保留属于重要优化检查项，未实现要明确记录和分析，不能在性能未达标时当作已经完成。替代结构若提供同范围功能及目标性能证据，可据实解释；无实测的“后续可优化”不构成交付。[^pipeline-validation-structure-spec]

资源验收表至少区分`GM task slot`、`QP阶段pair`、`KV片slot`、`L0 A/B slot`、`L0C输出slot`、`Vector输入/输出子slot`和`invL任务slot`。每项分别填写模数、递增事件、跨函数是否保留、首次可写、最后reader、最终排空。它们即使都取模2也不代表同一个生命周期；把所有对象绑定到`task%2`必须逐一证明，不能因为小用例parity恰好恢复就通过。

<a id="pipeline-validation--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

<a id="pipeline-validation--实现前的三个actor模型"></a>
### 实现前的三个actor模型

把AIC、AIV0、AIV1各自视为独立程序，通知携带概念上的(slot,generation)；完成事件可按当前抽象允许的顺序交错。至少验证：

- L=0/1/2/3/7，slot环绕不止一次；令一个AIV明显比另一个慢。
- X/P/U/可选pack的每次读取都匹配本task的完整ready，两半P完成前不能PV。
- 覆写前旧reader完成，invL和TaskInfo也包含在内。
- 不能出现非终态所有actor均阻塞，不能把一个generation通知消费两次，终态无待消费token。
- 保留故障注入：省略一侧P_READY应检测到阻塞；错误slot应发现generation错配。对去掉U_FREE的变体，先核对模型是否允许旧V2的异步读取在新C2写入时尚未完成：允许时应检测到覆盖；若V2被抽象为原子完成，`V2(k)→V1(k+2)→P_READY(k+2)→C2(k+2)`本身已形成保护，删显式free未必产生错误，应如实报告该抽象下free冗余，不制造失败。模型不能宣称覆盖未建模的DMA重叠风险。

模型必须说明每个原子步代表哪段真实DMA/MMAD/FIX完成。C++函数返回并非原子完成；将所有stage顺序原子化可能掩盖真实异步覆盖，只能证明该抽象下的顺序。最终还需HardEvent审查及设备检查。[^pipeline-validation-pipeline-spec]

随附[参考模型](pipeline.md#model)及[使用说明](pipeline.md#model-usage)提供可运行起点。先将输出保存到自己的工程，再按最终源码另建实际实现模型及映射表。未修改参考程序的PASS只能填“参考协议检查”，不能填“实际算子模型通过”；也不能据此把地址范围、HardEvent、异步DMA或硬件验收标为PASS。模型输出保留自身SHA、命令、探索是否完成、负对照和抽象边界。

<a id="pipeline-validation--设备检查顺序"></a>
### 设备检查顺序

1. 单任务非零Q/KV/sink，确认metadata读表和C1→V1→C2→V2数值闭环；输出与中间量按原reference比较。
2. 两任务、三任务、七任务，覆盖首槽/另一槽/回绕、尾部、两AIV中一侧零行、最后一个batch空有效工作。
3. 当前输入与另一个输入A/B/A/B复用相同设备地址；逐次重新metadata，prefill NaN输出，核验全部O/LSE、输入未变及metadata确定性初始化。PA含非identity和重复页号。
4. instrumented/release均用本轮同一数学源码；记录真实告警，不能以工具退出码0当作无warning。
5. 最终候选通过调用者指定的原ATK用例（逐case ID选择，不能取前十个数组元素冒充指定ID），再采集性能。更改最终库、tile、同步或wrapper后不能继承旧二进制的设备结果。

<a id="pipeline-validation--pv-tail-validation"></a>
<a id="pipeline-validation--pv-有效-k-与物理尾块检查"></a>
### PV 有效 K 与物理尾块检查

按[通用尾块规则](pipeline.md#structure--pv-tail)对最终源码列出 `Kvalid`、每片 `kc/kPhys`、P/KV/U行距、真实GM读取区间、L1/L0布局、MMAD实际K以及首末K条件。重点验证以下内容，已有测试能覆盖时复用，不另造固定case分支：

- **参数化边界**：围绕所选 `aK/K0cap/K1cap` 构造前一值、整倍数、后一值和多片尾块，覆盖至少一次跨L1切分；只在本工程声明容量内提交设备任务。超出当前容量的公式枚举不计为硬件支持。K=0走零工作协议，不能发零长度非法DMA。原129例仅是其中一个取样点。
- **覆盖与地址**：K片半开区间不重叠、不漏元素，合计恰为 `Kvalid`；首K清零一次、末K只有一次。改变K余数不改变P的物理行距、KV原生stride或U整行步长；PA包含非identity/重复页及窗口末端落在页尾的情况。
- **padding与复用**：在合法分配的诊断scratch中预填NaN/哨兵，改变有效长度后复用同一L1/GM槽；确认选定copy/LoadData消除了所有会参与计算的无效值，并核对有效输入、相邻输出未变。仅在GM输出预填NaN不能证明L1/L0尾块已初始化；调试插桩的状态按实际运行范围单列。
- **数值与同步**：支持的FP16/BF16均覆盖P舍入、PV累加和完整O/LSE；保留非零非均匀sink。尾片也正确完成L1/L0事件，P跨D复用不被其它K片覆盖；MMAD的K切分不新增softmax分片。
- **性能归因**：非对齐尾块若大幅慢于邻近对齐长度，检查是否多了零步长ND2NZ补零；用同机同输入单因素实验及MTE2/MMAD计数区分搬运与计算。保持既定5+5口径，不将历史单例、PMU时长或简化ABI结果提升为正式全套PASS。实现中不得出现为验证集某个K/head/case ID单独绕过通用公式的分支。

诊断前把“参考入口、输入量化、舍入路径、输出后处理、比较器”写成一行记录。若调用者提供的ATK含`independent_reference(low_precision=True)`低标杆，须核对其低精度累加和输出floor处理；它可能不同于调用者提供的完整given-golden数值路径。不能把`given-golden`的严格逐元素诊断阈值套到ATK低标杆上，再据误差改kernel去拟合floor。原ATK使用其原高/低双标杆与原比较器；严格诊断使用本轮声明的given-golden和预先固定的容差。若先前混用了二者，保留失败记录、核对输入字节与参考路径，再使用正确组合重测；这不构成放宽阈值，也不能跳过原ATK门禁。数值缓存同时绑定实际参考函数和这些选项，不能只用case_id/dtype作缓存键。

原ATK子集通过后仍核对入口条件是否无意缩小声明域。逐项把[普通契约](interface.md#swa-contract)映射到实际host判断：SWA合法非空`cmp_block_table`等保留辅助槽、singleton轴正stride、TND storage/used区分、有限double scale的实际数值路径、metadata无Tensor factory分派。这些不是额外算法模式；不能以“普通SWA没有压缩数学”为由把所有cmp槽非空一律拒绝，也不能把选定用例没有触发的条件视为已经验证。每项标明静态结论及正式/诊断运行范围；诊断descriptor不含某公开槽时，应从正式入口补探针，不伪造descriptor字段。

单设备kernel提交后60秒无返回按workflow记录TIMEOUT，先清理本任务进程及资源，再最小化定位。CPU golden准备慢与NPU等待分开记录，不能将CPU时长误报为kernel死锁。

<a id="case-recipes--代码模式"></a>

## 用例生成与覆盖审计配方

<a id="case-recipes--算子算法"></a>
## 算子算法

<a id="case-recipes--参数化用例矩阵"></a>
### 参数化用例矩阵

以下是可重建原覆盖矩阵的一组确定性配方，不是算子合法输入的白名单。

| 维度 | 候选或构造 |
| --- | --- |
| 模式 | SWA、CFA、SCFA；压缩倍率分别为 1、128、4 |
| Q/KV 布局 | BSND/BSND、TND/TND、BSND/PA_BBND、TND/PA_BBND |
| dtype | FP16、BF16 |
| query 长度 Q[p] | 1,2,3,7,15,16,17,31,32,63,64,127,128,129,255,256,257,511,512,513 |
| 原 KV 长度候选 V[p] | 128,255,511,512,513,1023,1024,1025,4095,4096,4097,8191,8192,8193,16383,16384,32767,32768,65537,131073 |
| 稀疏容量 K[p] | 1,2,15,16,17,127,128,255,256,511,512,513,1023,1024,1025,1537,2048,4096,8191,8192 |
| batch 与 ragged | B 按 1,2,4,2 循环；第 b 个 batch 的 Lq=max(1,Q[p]−δ[b])，δ=0,1,7,2 |
| head 与维度 | G=2^(p mod 8)，Nkv=1，D=512 |
| 残余 | residual[b]=b mod ratio |
| ori/cmp 页长 | ori 按 16,64,128,1024 循环；cmp 按 32,128,512,64 循环 |
| sink/scale/LSE | sink 按 −2,0.5,3 循环；scale=(0.5,1,2)/sqrt(512) 循环；LSE 交替开关 |
| 原始值域 | 默认均匀分布 [−1,1]；不为通过精度而缩窄 |
| 分层 | p≥16 标压力；p=0,7,15,19 可选性能；其余功能 |

profile p 的范围为 0..19。SCFA 令 Lc=K+ceil((Lq−1)/4)+8，Lo=4Lc+residual；
CFA 令 Lc=ceil(max(V[p]+b,Lq)/128)，Lo=128Lc+residual；SWA 令 Lo=max(V[p]+b,Lq)、Lc=0。
非稀疏模式的清单 K=0 是 manifest 表示，适配公开接口时仍须按 SWA/CFA 的可选槽契约转换，
不能把它当成 CSA 的合法 K=0。窗口为 127/0，ori mask=4；压缩模式 cmp mask=3。

按模式、布局、dtype、profile 的固定顺序可得到 480 条基础正例。用稳定 case_id 标识记录，
seed 可按 `1000+p+7919*case_id` 生成；seed 和 ID 只参与数据和报告，不能进入 kernel 分派。

<a id="case-recipes--稀疏行构造与独立选择-oracle"></a>
### 稀疏行构造与独立选择 oracle

每行先计算 `C=max(0,min(Lc,floor((Lo−Lq+t+1)/ratio)))`，有效扫描上限为 `n=min(K,C)`。
容量 K 的行可以初始化为 `row[j]=j mod max(1,C)`，然后按下列模式修改；不能只分配 n 个槽。

| 模式 | 数据构造与要验证的结果 |
| --- | --- |
| full | 前 K 个 ID 有效且互异；首、末 query 都实际选择 K 个 |
| clipped | 0<C<K，扫描槽数和实际选择量小于容量 K |
| permuted | 用 `seed+b*1000003+t*9176` 洗牌前 n 槽，保持同一列表语义 |
| duplicate | n>1 时令 row[1]=row[0]；重复位置必须保留贡献 |
| future | C<Lc 且 n>0 时令 row[0]=C；物理合法但当前不可见，应跳过而非终止 |
| early_stop | 在 n//2 放 −1，后方仍放合法 ID，验证提前终止 |
| empty | 首槽 −1，验证没有选中位置 |

oracle 只扫描前 n 槽，遇 −1 立即终止，只接受 `0≤ID<C`，按原顺序保留重复。
不要调用设备侧索引函数来构造“独立”参考。定向选择例可取 K=1537、G=1、Lq=[17,16]、ratio=4：
clipped 的 Lc=[384,385]，其他模式 Lc=[1600,1601]，Lo 按倍率与 residual 重建。
四布局×两 dtype×六种选择扰动共 48 条，开启 LSE。

<a id="case-recipes--数值调度和负例扩展"></a>
### 数值、调度和负例扩展

数值模式与三模式×两 dtype 交叉，采用 TND/TND、开启 LSE，可形成 30 条：
常量 Q=0.25/KV=−0.5；交替符号 Q 幅度1/KV幅度0.5；有限大值均匀 [−8,8]；
近均匀 Q 幅度0.001/KV幅度1；Q=0/KV幅度1。冻结生成方式，不在结果出来后改变值域。

低占用调度专设 `B=1,Lq=1,Lc=8200,Lo=4*8200,K=8192`，G 取1和128，
与四布局、两 dtype 交叉形成16条压力/性能例，覆盖最大 K 和少任务的组合。

负例从合法 SCFA 数据出发，在张量准备后做一项变异：

| 变异 | 入口 |
| --- | --- |
| K=0、8193、8705，非法 Q layout，ratio=5 | main 和 metadata 分别调用 |
| Q dtype 改 FP32，移除 cmp_kv | 只检查签名包含该参数的 main |

前述七类与四布局、两 dtype 交叉形成56条。K 变异必须同步更新索引容量和 metadata 属性，
保持 ratio=4 及压缩输入存在，避免被适配成 CFA 或被无关形状错误提前拒绝。
完整示例规模为 `480+48+30+16+56=630`；数量用于对账，不构成所有任务必须执行630条的要求。

<a id="case-recipes--数据路径与存储层级"></a>
## 数据路径与存储层级

独占随机页配方对每 batch 计算 ceil(L/page)，将所有物理页 ID 洗牌后分配各行，
页表不足列填 −1。ori/cmp 分开使用 seed+13/seed+29；多页时若随机结果恰为恒等映射，
交换首尾页以保证这个探针确实非恒等。此处页号排列的双射检查只适用于独占页配方，
共享页/重复页用例采用其独立契约，不能被双射检查拒绝。

估算内存时，Q token 容量 BSND 为 B*max(Lq)，TND 为 sum(Lq)；每组 KV 的容量
BSND 为 B*max(Lkv)，TND 为 sum(Lkv)，PA 为 sum(ceil(Lkv/page)*page)。
FP16/BF16 下基础估算为 `(2*Qcapacity*G+KVcapacity)*512*2+Qcapacity*K*4` 字节，
分别覆盖 Q/O、KV 和 INT32 索引；还要另加实际 LSE、sink、页表、metadata、workspace 与对齐开销。

<a id="diagnostics--代码模式"></a>

## 动态诊断运行器的实现步骤

<a id="diagnostics--诊断输入范围与实际数据构造"></a>
## 诊断输入范围与实际数据构造

基础诊断路径支持正向 SWA：四种 Q/KV 布局配对、相同 BF16/FP16 的 Q/KV、B>0、
N1 为1..128的2幂、N2=1、D=512、ori mask4/window127/0、无压缩 KV/K/ratio，scale有限，LSE为布尔值。
物理存储容量为正，但有效长度允许零；TND cu 从0开始、单调、B+1项、末项等于总存储 T，
seqused 非负且不超过各 batch 的存储差值。PA 页长16..1024且为16倍数，至少一个物理页并有有效 KV 长度。
这些检查描述当前诊断器覆盖，不额外缩小已冻结公开接口范围。

长度先分清存储与有效值：TND storage=相邻 cu 差；BSND storage=固定 S1/S2；
effective 优先 seqused，缺省才取 storage。PA 的实际可访问逻辑容量由页表和有效长度决定。
所有 batch 的 `min(effective_q,effective_kv)=0` 才是全零 attention 工作。

从不变 spec 准备 A/B 两份输入：结构 seed 可用 `20260624+1009*case_id`，数值 seed A 可用
`case_id+seed_offset`，B=A+非零 delta（示例1000003）。Python、NumPy、Torch 的结构随机状态显式固定。
调用原 golden.gen_data 的 SWA 入口准备结构，跳过其重复 CPU 输出计算，随后生成量化后的实际输入。
Q/KV/sink 可分别用 CPU generator 的 seed `13*value_seed+1/3/7`，模2^63−1；
Q/KV 取 spec 原值域，sink 取 Q 值域的1/10。该配方用于复现诊断，不能宣称与原 ATK 输入逐字节相同。

PA 先冻结 actual physical pages、页表和有效长度。原生成器若要求独占页，临时生成页数取
`max(actual_pages,sum(ceil(Lkv/page)))`，随后恢复真实物理张量容量、显式页表并重新填值。
物理容量不够独占页且没有显式表时，可按逻辑页号 mod actual_pages 构造共享映射；显式表必须矩形，
每个有效表项为范围内整数，padding表项不参与访问。独占非恒等配方要求至少两页且足够独占容量，
采用递增页号加1模页数；它与显式共享表互斥，不能用这个配方的容量要求拒绝合法单页算子输入。
记录临时页数、实际页数、逻辑页数和实际页表。

KV stride 扰动可在物理第二轴加3个 padding 后取窄 view；页表行可额外加3个 −1 列再取 view，
保留实际 stride。使用真实量化物理 KV 和实际页表逐段还原 CPU 逻辑 KV `[B,N2,max(Lkv),D]`，
最后一页只取有效元素，送入未改动 golden.forward，不能用临时独占输入或裁剪前参考代替。
单页和共享页须在第一次 prepare 前定型，A/B 不得中途改变共享 spec。

<a id="diagnostics--定向域探针配方"></a>
### 定向域探针配方

下面20条是补充子集的配方，ID仅用于报告，不代表合法域边界：

- 两 dtype×四布局的8条：B=3，Q/KV存储129/145，used_q=[127,128,129]、used_kv=[131,145,113]，
  FP16 H2、BF16 H8，scale=0.09375，LSE开启、原生stride扰动；TND总容量387/435，PA页长16/物理页32。
- 3条零工作：Q/KV容量17/33，used_q=[0,17,9]、used_kv=[33,0,23]，再分别把全部 Q 或 KV used置0。
- 1条多batch短decode：B=129、S1=1、S2=129、H1，KV有效长127/128/129循环，检查私有metadata未引入小B上限。
- 两 dtype×两 Q布局×两共享映射的8条：Q/KV有效长129、H2、页长16、LSE开启。
  共享前缀用B2、10物理页，前8页按7..0排列，两batch尾页分别8/9；单物理页用B1、页号0重复9次。
  KV和页表行都做stride扰动，覆盖少物理页与重复逻辑位置。

定向轻量探针可冻结值域 [−0.02,0.02] 用于地址/长度诊断，但不能替代调用者原值域和数值压力用例。
补充 null sink、单页singleton stride、H128 FP16等分支按实际契约另选，不因这20条通过宣称全域通过。

<a id="diagnostics--cpu-标杆输出及精度判断"></a>
## CPU 标杆、输出及精度判断

要求原 golden 的 RUN_MODE=1，在 CPU 调用其 gen_data 与 GeneralizedSFA.forward。构造器和 forward
分别传其实际签名需要的布局、dtype、容量、长度、页参数、scale、mask、window、sink 和可选槽，
保持 P 的低精度 cast 在 PV 前、分母在 cast 前累计。TND golden 输出 FP32但包含量化后的数值，
DUT仍保持 Q dtype；比较时同时提升到FP32，不改写 golden。

O必须与Q同shape、DUT为Q dtype，CPU参考为原约定dtype，均有限。LSE要求FP32：
BSND `[B,N2,S1,G]`；TND golden `[T1,N2,G]` 显式转成设备 `[N2,T1,G]`，保留原始与适配两份。
关闭LSE要求参考不返回LSE，正式接口的空返回槽按其契约检查。

独立诊断阈值在DUT运行前冻结：

| 输出 | atol | rtol | 额外最大绝对误差上限 |
| --- | ---: | ---: | ---: |
| BF16 O | 2^-16 | 2^-7 | 2^-14 |
| FP16 O | 2^-19 | 2^-10 | 2^-17 |
| FP32 LSE | 2e-5 | 2e-5 | 2e-4 |

逐元素同时满足 `abs(a−r)≤atol+rtol*abs(r)` 与最大绝对误差上限，全部有限、错误数=0，
另检查padding精确零。报告元素数、最大误差、RMS、完全相等元素数和失败数；大张量可分块统计但不能抽样。
每batch无效前缀为max(used_q−used_kv,0)，尾padding从used_q到storage_q，两段分别检查；
如果原golden的这些位置非零，报告契约冲突，不能先把DUT擦零再比。这些阈值不等于原ATK双标杆指标。

<a id="diagnostics--abi设备分配与-metadata-核验"></a>
## ABI、设备分配与 metadata 核验

沿用本文件的64位自然对齐 SwaTensorView/HostIntArray/Descriptor/Plan；记录各结构sizeof/offset，
禁止pragma pack。TensorView data是逻辑首元素，stride单位元素，缺省tensor全部零，rank仅1..4。
四个host长度数组保持到同步host调用返回；异步kernel不得读host指针。
主库提供get_plan/metadata/launch/last_error，独立metadata库提供前三者中除launch外的符号。
非零返回读取last_error并记录，get_plan检查版本、字节数、reserved=0、metadata_words在1..1024，
非零工作要求core_count≥1。主库和独立库plan及任务表应一致，正式metadata输出仍为INT32[1024]。

设备为每个非连续输入分配 `span=1+Σ((shape[d]−1)*stride[d])` 个元素，再按真实shape/stride构造view，
空tensor span=0。分配不等于赋值，显式copy全部有效Q/KV/sink/页表/长度，确认完成；
主输出按Q shape/dtype分配，LSE按公开轴序分配，metadata和workspace按plan分配且保持存活。
记录shape、dtype、stride、storage offset、逻辑字节哈希及设备地址。

metadata预热5次、正式host测5次，每次在两种不同哨兵预填之间交替后重新生成，要求每个字节确定，
包括关闭核和保留字段。主库与独立库生成结果一致；H2D另行计时，不能混成metadata device duration。
Q/KV/sink数值变化而调度参数不变时表不应依赖旧输出；调度参数变化则重新产表。

<a id="diagnostics--同地址复用与完整设备采样"></a>
## 同地址复用与完整设备采样

固定同一批设备输入、输出、metadata和workspace地址，按A/B/A/B运行。每次复制当前输入，生成并上传当前表，
O/LSE用NaN预填并同步，再真实launch、同步、比较全输出，检查输入和metadata字节未改变。
要求两次A逐位相同、两次B逐位相同；非全零工作时A/B输入与输出应有区别，全零工作允许相同全零输出。
CPU标杆运行前后也核验输入未变；失败停止该case的性能采集，并保留失败状态。

精度通过后，5次不采集预热，进入profiler执行5次完整调用再同步；进度写盘、CPU参考和哈希准备放在采样外。
每个必需kernel组件使用唯一可区分的真实名字匹配；每个组件恰好5条记录，同一row不能匹配多个组件。
CSV必须能唯一识别Name/Op Name/Kernel Name与微秒Duration列，所有duration有限且严格正。
若存在未计入的必需设备任务则拒绝汇总。按调用次序或trace调用身份将各组件配对后求和，得到5个完整时长，
全部参与均值，同时保存median/min/max、逐组件样本、原始CSV及哈希和规范化样本CSV。

若全零工作没有kernel记录，报告NO_DEVICE_KERNEL_RECORDS、样本为空、均值null，
不推断零延迟；若有清零/DMA等任务则保留实际开销。采样后再次比较完整O/LSE、最终输入和metadata，
要求输出仍与最后一次B精度结果一致，不能只在profiling前通过。

# 约束

<a id="coverage--约束"></a>

## 功能覆盖与正式验收

负例只验证冻结契约中的拒绝条件，不以设备故障代替参数校验。

<a id="suite--约束"></a>

## 动态诊断 ABI 与完整输出

保持原始 golden 文件逐字不改；不得换算输入值域、跳过输出、擦除 DUT padding 或将原 ATK 的比值配置直接当作 allclose 阈值。只有实际运行的非恒等页表用例才构成 PA 覆盖证据，ABI 预留 PA 字段不算。unsupported 组合明确返回参数错误，不得默默落回 case 常量。

<a id="pipeline-validation--约束"></a>

## 流水源码与设备验收

<a id="pipeline-validation--性能必须用最终完整调用"></a>
### 性能必须用最终完整调用

每例5次未采样预热，再5次正式设备profiler采样，**全部5次取均值**。每次main实际需要的pack、清零、归并等device task全部计入；独立metadata另列其已有约定口径。冻结手写基线只读，不重跑。`baseline_us/candidate_us>=0.8`逐例判定，即`candidate_us<=baseline_us/0.8`。[^pipeline-validation-swa-validation]

ACL Event包住多次launch得到的区间可能含提交/阶段等待，不能伪装成与手写单kernel profiler相同口径；原始CSV缺失或只筛了最快组件就保持性能NOT_ACCEPTED。若正式入口与诊断直调并存，标记实际计时路径并核对二进制身份，不能将direct-only结果写成formal通过。

不要为了看起来通过而改变输入、缩小有效范围、读取已有输出、改golden/ATK阈值、剔除慢样本或调整baseline。性能未达标时继续当前轮自修复；报告每次实质性改动及失败原因，不能只给最新成功编译日志。

# 失败表现

<a id="coverage--失败表现"></a>

## 功能覆盖与正式验收

负例从已通过正例复制，每次只改变一个公开参数。K0/超限CSA必须保持cmp输入和索引tensor存在，不能被adapter根据K0推成CFA；同步改变索引容量与metadata cmp_topk，保证主因是非法K。对于两入口都有的参数分别调用，否则metadata先报错并没有证明主API会拒绝。q dtype、cmp数据tensor不在metadata签名内，这类case的targets仅含main；不能编造metadata不存在的参数。K0保持ratio4，因此metadata也应拒绝该组合；K0+ratio128是合法CFA，不能误判。

PASS要求发生预期参数拒绝并命中约定类别/参数；接受 `Invalid_Argument`、`AclNN_Parameter_Error`及实际包装层保留的参数校验语义，不绑死EZ0026/EZ0037这类日志编号。已有明确返回码时保留校验，不能丢弃原有支持的异常语义。NPU初始化失败、import失败、OOM、超时、设备执行/同步错误都不是负例PASS。未报错为FAIL。每条捕获并记录后继续下一条；设备故障需停止本轮并标记剩余NOT_RUN，不伪装全套通过。

<a id="suite--失败表现"></a>

## 动态诊断 ABI 与完整输出

返回非零时读取 last_error；NaN 残留、padding 非零、A/B 地址变化或复用输出、重复输出 hash 不一致均为失败。导入/设备初始化/同步/运行错误写 ERROR；未执行 case 保持 NOT_RUN。真实 profile 缺失、主核样本不为5、duration 非正或非有限均拒绝汇总，不能用主机计时代替。

[^suite-pa-fixture-construction]: [动态诊断配方](validation.md#diagnostics)规定非恒等页表、临时/实际容量与数据准备步骤；执行时核验本轮生成的运行器，配方前置条件不改变[单页与共享页契约](interface.md#swa-contract)。

<a id="pipeline-validation--失败表现"></a>

## 流水源码与设备验收

- design里仍写“先串行，后续融合”，而最终源码只有四个独立计算kernel：结构未完成。
- 文档写M16，最终使用M128；写双AIV，源码没有part；写Q只装一次，实际每N块重装：先按最终代码重写文档并重新验收。
- 精度全过但0.8x未过：保留精度PASS，性能FAIL；审查缺失机制和profiler，不能降级成完成。
- 有握手但末尾挂起：核对外层ready/free和内部HardEvent分别的初始及最终计数，不靠增加无条件Wait试错。
- 生成者需要父级口头解释才能继续：作为专用知识缺口记录，修订文档后由下一位空上下文生成者重新验证。

# 验证方法

<a id="coverage--验证方法"></a>

## 功能覆盖与正式验收

运行器为每条case输出JSONL，最少字段：

```json
{"case_id":0,"status":"PASS","build_id":"本轮二进制标识","phase":"final","log":"logs/0.log","outputs_checked":["attention_out","softmax_lse"],"metadata_checked":true,"branches":["SWA","BSND+BSND","g1","fa_only"]}
```

关闭LSE也检查返回槽为约定空tensor。负例按清单targets记录 `main_rejected:true` 和/或 `metadata_rejected:true`，以及 `error_kind:"parameter"`；不能只以日志中含error判PASS。status取PASS/FAIL/TIMEOUT/NOT_RUN。最终记录必须来自同一构建；sanitizer构建单独归档，不混用两个build_id凑齐覆盖。

按[执行记录对账](validation.md#case-recipes)核对 manifest 与最终 JSONL 的全部 ID、build_id、两个输出槽、metadata 和分支证据。
此检查拒绝空报告、缺例、重复ID、失败、不同build_id及缺少两个输出/metadata的记录。它是完整性检查，仍需审查日志、Stage记录、重复/sanitizer及性能结果。

性能选每模式/布局/dtype的decode和prefill、中长KV、G1和较大G、batch1和batch>1。固定输入/设备/频率/stream及统计范围；当前采样口径为5次未采集预热、5次正式采集，取全部5次均值，不自行增加次数。主数学核、metadata、控制拷贝与可选FD归并从同组完整调用trace分项统计，额外数学工作必须纳入主核可比边界。用户要求复测时使用新输出目录并保留前次结果，不覆盖失败或挑选较快样本。缓存metadata只用于调度输入完全相同且明确声明的对照，不替代完整成对调用验收。

先确认profiler本轮确实产生数据，按实际kernel名和调用序号筛掉预热/其他算子；保留逐次device时长，汇总中位数/分位数与波动、逐case时间变化和分桶几何平均加速比。不能把host异步launch耗时、空结果或只计PV的时间当算子性能。性能目标在03阶段固定；无可比基线时报告测量值，不能写“性能合理/已达标”。低于本页最低性能门槛时，定位到gather、Cube有效利用率、等待、慢核或额外归并后继续迭代；已达到最低门槛且其余验收通过时允许结束，更高性能仍作为优化方向。

[^coverage-development-contract]: [待覆盖的Stage和分支](development.md#stages)。

<a id="coverage--性能验收与停止条件"></a>
## 性能验收与停止条件

优化方向是尽快生成尽可能高性能的正确实现；默认允许结束的最低性能与优化期望分开记录。首次实现仍采用已知有效的分块、资源寿命和流水，不以降低门槛为理由先生成简易串行版本。

对调用者提供的可比手写基线所覆盖的同一用例，本任务最低速度比为 **0.8×**；其他任务以调用时冻结的目标为准：

`speedup_i = handwritten_device_us_i / generated_device_us_i >= 0.8`

等价于 `generated_device_us_i <= handwritten_device_us_i / 0.8`，即允许耗时达到手写的1.25倍，边界相等通过。逐条应用于本次声明范围，不能用全套平均值掩盖不达标用例，也不能把某个固定样例的绝对耗时阈值用于其他case。精度、功能、同步、接口和本次范围内每条性能都通过后，普通生成任务可归档结束；更高性能继续作为优化方向，不要求为追平历史最好结果无限迭代。只有低于最低速度比，或用户另有更严格的硬目标时，才因性能不达而必须继续。用户指定的研究任务、迭代轮数和优化目标仍按该任务执行，不能用0.8×提前截断。

手写基线由调用者提供，不能从 skill 的历史文件或固定服务器路径补齐。先核对选中 case 的完整 spec、目标架构、输入、单位、统计方式和计时边界；本轮已知历史报告采用预热20次、记录50次、取最后30次均值，统计的是正式调用中的完整 `SparseFlashMla` 主核 device duration，不含metadata、host、H2D、其他辅助核或整链时间。新候选采用5次预热、5次正式采集，全部5次取均值；不把它标成历史50/30口径。输入、设备架构和计时边界保持对应，以手写基线/候选均值计算速度比，并显式记录两者采样配置。若需对波动进一步确认，另建结果目录复测，不能只挑最快样本。必须先核对真实profile中的主核归属、完整执行阶段及report/raw均值，不能把host时间、单个PV片段或端到端时间填入该字段。

按[性能证据解析](performance.md#acceptance)计算本次case门槛，无需运行设备：

提供冻结基线、原始用例文件和完整的目标 ID 集合；按[证据规则](performance.md#acceptance)仅计算逐例 limit_us，状态为 REFERENCE_ONLY。
正式采集并核对profile后，候选结果采用 `validated_performance.json` 的格式：顶层 `task=performance_device`，显式 `collection={warmup:5,sample_count:5,last_samples:5,statistic:"mean_all_5"}`，`cases`中每项包含 `id,device_us,raw_mean_us,samples_us`，五条原始样本与两处均值一致。历史50/30报告只能按原采样说明解析，不能伪装新五条采样。生成工程可以更换算子名，但须在汇总前识别本轮完整主核；若实现含必需的归并/补偿设备核，不能只取其中一段冒充完整计算时间，须另冻结与手写完整计算可比的范围。可写 `scope=formal_main_kernel` 明确口径。评价所需的全部case须用重复的 `--case-id` 指定，例如单case：

提供冻结基线、原始用例文件和完整的目标 ID 集合；再提供本轮正式候选报告，按[证据规则](performance.md#acceptance)保存逐例 performance-acceptance.json。
工程的性能评价步骤逐条计算速度比，未达返回1，缺case/异常数值/口径不符返回2；仅查询基线输出 `REFERENCE_ONLY`，不表示新算子已通过。`--baseline`必须显式给出；调用者提供的原始历史报告还须给出 `--case-json`，脚本才可按完整 `smla_case_spec` 为选中 ID 生成身份哈希。已有带哈希的规范化基线可直接读取，若同时传用例会复核身份。`--min-speedup`用于用户明确更改最低门槛。候选提供 `case_spec_sha256` 时脚本核对用例身份；未提供则必须由接口契约和原case人工核对，结果不会冒称已自动核验。不能只凭同一个id认定同一计算。

诊断和正式调用均采用5次预热、5次采样。简化descriptor直调结果只用于其已声明范围，不能替代已冻结公开接口的精度与性能门禁。公开交付层按[接口路由](interface.md#api--用法)决定，见[三种调用层的验收映射](validation.md#pipeline-validation--用法)：要求ACLNN包互通时，必须以真实ACLNN主核和AICPU metadata完成对应门禁；只要求原ATK的完整PyTorch调用时，验证两个完整schema及本轮独立metadata/native CATLASS链，不另行强制新增ACLNN包。不能在同一次已冻结候选失败后偷换交付层来消除失败，也不能把未执行的层标为PASS。诊断不内置固定的65µs绝对上限；只有用户明确给出可比直调硬目标时才增加对应检查。metadata、辅助核、设备任务总和和端到端仍分别报告，其时间不与主核速度比混算；没有另设硬目标时不派生额外停止门槛。新shape或基线外case先建立对应可比基线，不借用别的case的数值。

<a id="suite--验证方法"></a>

## 动态诊断 ABI 与完整输出

按[动态诊断步骤](validation.md#diagnostics)在生成工程中实现运行器，显式传入本轮 golden、case JSON、目标 ID 或全部用例、主库、独立 metadata 库、设备、round ID、新输出目录和完整主核选择器。需要只查精度时显式关闭 profiling；未运行的正式门禁保持 NOT_RUN。
Selection is explicit: repeat `--case-id` for the caller's target IDs or pass `--all-cases`; missing/duplicate IDs are rejected. `--main-kernel` identifies a real profiler kernel name substring. Repeat it only when multiple mandatory device kernels comprise one complete invocation; every named component must occur exactly 5 times. Warm up 5 launches, profile 5, report the mean of all 5 complete-invocation device durations. Keep raw `kernel_details.csv`, a copied raw CSV, normalized samples CSV, SHA256, mean/median/range, all component samples and other device rows. Metadata host planning and its H2D wall time are separate diagnostics.

所有device task必须计入；若CSV存在未被选择器覆盖的任务则停止汇总，避免只挑主核而遗漏pack/清零/归并核。性能采集固定5次未采样预热、5次正式采样，全部5条参与均值，不再丢弃正式样本。每次A/B/A/B同步后、下一次输入覆盖前，检查设备Q/KV/sinks及metadata字节未改变。metadata输出以两种不同哨兵预填再生成，验证保留位也被确定性写出。

输入由此诊断工具按选中 case 的结构和确定性种子生成，随后调用未改动标杆的完整数学；它只对自己的有效元素负责。是否与调用者 ATK 的输入分布及重采样逻辑一致，须读取本轮脚本并核对，不能凭历史样本假定。padding 空洞初始化值不属于逻辑输入；设备侧保持原 stride。

These measurements are labeled `direct_c_abi`, never `formal_main_kernel`. Sampling parity with the baseline does not make direct metrics formal. Original ATK precision, registered formal entry execution and formal 0.8 performance acceptance remain NOT_RUN and require separate real runs; this script never writes a formal accepted performance candidate. A direct precision pass plus a measured direct profile is only `DIRECT_DIAGNOSTICS_PASS`.

Local tests and `--describe-abi` / `--inspect-cases` use only Python's standard library. Torch, NumPy and torch_npu are imported only when numerical/device execution is requested.

<a id="suite--同轮-cpu-标杆复用与断点续跑"></a>
## 同轮 CPU 标杆复用与断点续跑

生成验证运行器时，将CPU输入/标杆准备、设备精度、原ATK验收和性能采集分成可追溯的步骤。同一独立生成轮内，已完成的CPU输入与完整golden可以复用到后续ABAB和性能验证，避免每次启动或每次重复调用都重新计算高标杆。缓存清单须绑定round ID、原case文件和规范化完整spec、全部参数与布局、结构/数值seed、未改动golden及生成/适配工具哈希；逐tensor记录shape、dtype、原生stride、storage offset和逻辑元素字节哈希，并记录缓存文件哈希。cu/used、页表、sink、scale和LSE开关均属于输入身份，不能仅按case ID或shape命中缓存。读取时实际校验文件、身份与tensor描述；只有清单中的路径而未核验文件，不构成可复用证据。工具或输入语义改变时重新核验其依赖，不能只修改缓存版本字段。

缓存只提供CPU输入和完整数学参考。保持量化输入字节、非连续stride及可选槽的缺省语义，TND低标杆继续保存原FP32输出；比较时按既有规则提升DUT数值，不把golden提前转为输入dtype。启用LSE时保留原始轴序和显式适配后的参考。加载参考后仍真实调用当前候选的metadata与main，执行全部O/LSE、padding、shape/dtype、输入未变和metadata全字初始化检查；A/B/A/B仍使用相同设备地址、逐次输出预填、同步后全输出比较与重复哈希检查。DUT输出、PASS行及旧候选身份可存档为历史证据，不能作为CPU参考缓存、代替新launch或改标为另一候选的通过记录。

原双标杆门禁由最终候选的一次真实原ATK运行满足：规定的原始用例和完整输出均按原FP32输入高标杆、量化低标杆及原阈值通过，且实际加载的候选身份已核验。补充ABAB可使用同轮已保存的低标杆，并单独链接该原ATK报告、哈希、覆盖范围和候选身份；不必为每次ABAB重算高标杆，也不能因此声称这些补充输入另做了双标杆比较。原ATK尚未通过、缺例或仅公共比较器通过时，门禁仍未完成。原ATK证据和补充完整输出检查互不替代。候选身份按[工程中的完整安装清单](development.md#engineering--诊断构建与正常构建的证据边界)绑定源码、包装层、构建模式、安装根、device object及AICPU库，不能只比较op-api壳的哈希；最终候选改变后，CPU参考满足前述条件仍可复用，设备精度、原ATK及性能结果必须来自新的最终候选。

按case保存CPU准备完成、每次精度launch同步/比较完成和整组性能采集完成的检查点，记录命令、输入/参考来源、候选身份与原始证据哈希。超时保留当时的失败/未完成报告及后续NOT_RUN，从实际未完成的case/步骤继续；恢复后不能把部分ABAB当成四次全部完成。仅当候选、输入及检查策略均未变化且完成证据可核验时，汇总可引用同一候选已通过的case；生成新的汇总文件，保留来源，不覆盖旧失败。若现有脚本无恢复接口，应先实现和核验明确的恢复入口，不能改写报告伪造完成。CPU阶段超时且没有设备提交记录时，先检查准备/标杆阶段，不据此判断NPU挂起。

性能从最终候选已通过完整精度的原始完整输入采集，每例仍只执行一组5次未采集metadata→main预热和5次正式完整调用，全部5条参与均值，采集后复核完整输出和输入未变。检查点写盘、缓存校验和参考准备放在采样循环之外。候选未变且原始trace完整时，修正selector、序列化或报告排版只离线重解析原样本，不新增采样、丢弃样本或挑更快结果；不足5条的中断组保留未完成，不能与其他组拼接成通过。正式与直调结果继续分别标记，CPU缓存复用不改变原始全输出验收、正式接口或5+5要求。

分配与初始化是两个独立步骤。`allocate_strided` 或 `empty/as_strided` 只提供形状和步长正确的设备存储，不会复制 CPU 输入。复用精度脚本的 allocation helper 时，须显式逐 tensor `copy_`，包括长度、页表和 sink；等待生产者完成，在第一次预热前核对设备有效元素字节与已通过输入完全一致。CPU 输入的哈希相同不能证明设备缓冲区已初始化。把这一核验放在采样循环外，采后再查完整输出、metadata 和输入未变。若采后发现输入未初始化，该组保留为无效尝试并排除性能验收；修正后首次用正确输入采集有必要，不能把错误输入的时间当作该 case 的有效样本或归咎于计算 kernel。

<a id="suite--一般swa域的定向补充"></a>
## 一般SWA域的定向补充

选定回归子集之外，按[定向域探针配方](validation.md#diagnostics)构造20条补充探针，显式选择全部探针运行；不提供 case_id 列表或 all-cases 时应拒绝运行，不能默认只取前十条。其余golden、库、设备、profiler和输出目录按本轮冻结输入指定。`--inspect-cases`可先核对选择集合与有效长度，不能当作运行通过。其中129-batch短decode探针检查私有metadata表是否引入原接口不存在的小batch上限；另有两dtype×两Q布局×共享前缀/单页重复映射的8条PA组合，覆盖LSE、真实页stride和页表行padding。该清单是定向覆盖，不能成为实现shape白名单。

探针覆盖FP16/BF16的四种合法Q/KV布局配对、127/128/129长度边界、Q长于KV的无效前缀、TND的seqused小于cu存储差值、非默认scale、LSE开关及局部/整批零有效工作。PA使用非恒等物理页表和真实页stride：先生成量化物理KV，再按实际页表还原CPU逻辑KV送入原golden；不得让设备读一份物理输入而标杆仍比较重采样前的逻辑输入。启用LSE时先验证golden实际轴序，再把TND `[T1,N2,G]`显式转换到设备`[N2,T1,G]`，同时保存两者。FP32 LSE诊断界限预先固定为atol=2e-5、rtol=2e-5、最大绝对误差2e-4；它独立于原ATK验收，不随DUT结果调整。

`diagnostic_page_table`可显式指定含重复页号的矩形表。物理页数小于逻辑页总数时，工具会为原golden生成器临时提供足够独占页，随后恢复实际物理容量并替换页表；默认构造使用周期页映射，显式表优先。临时/实际页数与页表写入`inputs.*.preparation`，input hash始终针对实际DUT数据。`diagnostic_nonidentity_pages`表示另一种独占且非恒等的构造配方，要求该配方所需容量；它不能与显式表同时使用，也不代表算子的物理页数约束。

**单物理页singleton-stride探针使用单页适用的配方。** 独占非恒等页重排至少需要两个物理页，且容量足以容纳全部独占逻辑页；这个构造前提不能变成算子的最小页数限制。单逻辑页用identity页表`[[0]]`并关闭`diagnostic_nonidentity_pages`；若多个逻辑页复用唯一物理页，应显式重复页号0，并按实际物理KV和页表重建CPU逻辑KV/golden，不把重复映射称为恒等映射。两类探针都可令物理页轴size=1、正stride小于页内跨度，以单独检查合法singleton首轴语义。[^suite-pa-fixture-construction]

容量、页表、有效长度和配方必须在第一次prepare前定型，A/B均从同一不变spec准备。不要先生成A再把共享spec的`block_num1`改成1，却让B继续使用独占非恒等配方；也不要裁掉物理页后继续比较裁剪前的逻辑golden。先在CPU核对临时/实际页数、表项范围与逻辑覆盖，以及两seed都能重复准备，再进入设备验证。若第二次prepare因配方容量不足而失败，保留fixture失败和未完成ABAB状态；这不是kernel拒绝单页的证据。

零有效工作仍有正的物理容量和有效batch，TND总T1也为正。输出中没有任何有效attention时，A/B相等是合法的，但必须验证全部应为零的位置、metadata初始化及输入不被改写；不能要求不同输入必然产生不同全零输出。若实际profile没有任何device kernel记录，只标`NO_DEVICE_KERNEL_RECORDS`且耗时为null，不伪造0µs或无穷速度比；若实现需要清零核，仍完整计入实际设备时间。未支持PA/LSE等组合的生成轮必须保留明确失败/未完成记录，不能用选定子集PASS覆盖该缺口。

<a id="pipeline-validation--验证方法"></a>

## 流水源码与设备验收

最终报告必须给出：读取skill版本/哈希、工程路径、实际source/build/install manifest、原ATK逐例结果、5条设备样本及均值/基线/比例、流水与结构表、仍未解决问题和NOT_RUN范围。父级先核验候选身份与源码结构，再核验原始运行证据；计数或schema检查只是辅助，不得自动判定语义通过。

[^pipeline-validation-pipeline-spec]: [普通SWA握手与排空协议](pipeline.md#protocol)。
[^pipeline-validation-structure-spec]: [普通SWA片上结构与Vector行块流水](pipeline.md#structure)。
[^pipeline-validation-swa-validation]: [功能与性能验收](validation.md#coverage)、[原ATK边界及采样细则](validation.md#suite)。

<a id="case-recipes--验证方法"></a>

## 用例生成与覆盖审计配方

清单审计检查非空、唯一 ID、各 batch 长度数组和 residual 的长度为 B，基础正例 Lq>0、Lo≥Lq，
压缩正例满足 Lo=ratio*Lc+residual。SWA 零有效长度的定向诊断使用另外的构造，不套用基础矩阵的正长度前提。
对 SCFA 首末 query 实算选择集合：full-K 同时满足数量=K、互异数=K；clipped-K 满足0<数量<K。
逐一统计全部20个 K 边界、布局/dtype/数值模式、G1/G128最大 K 调度、负例和选择扰动是否命中。

执行记录必须与清单的 ID 集合相等，拒绝空报告、重复/缺失/额外 ID；全部来自同一非空最终 build_id。
正例要求真实最终日志、两个输出槽检查、metadata 检查，以及 mode、layout 和 G 分支记录；
关闭 LSE 时检查空返回槽。负例按 targets 检查独立拒绝与参数错误类别。FAIL/TIMEOUT/NOT_RUN 均不计入通过。
记录检查不等于设备执行证明，仍核对原始日志、全输出、sanitizer、复用与性能证据。

<a id="diagnostics--验证方法"></a>

## 动态诊断运行器的实现步骤

每case保存spec与哈希、原ATK standard、精度策略、A/B seed和输入/标杆描述、plan、metadata、
四次launch结果、地址/哈希复用检查、原始trace和采后结果。整轮保存源文件与库哈希、软件版本、
设备号/可见性、round ID和每case状态；结束复核来源文件未在运行中改变。
输出目录须新建或为空，报告先写临时文件再原子替换，缓存只允许同轮满足完整身份验证的CPU参考。

独立诊断通过为DIRECT_ACCURACY_PASS或DIRECT_DIAGNOSTICS_PASS；失败可细分
PRECISION_OR_LIFECYCLE_FAIL、POST_PROFILE_FAIL或ERROR。尚未运行的case保留NOT_RUN，异常保留traceback
与最后execution_stage，不能把CPU准备阻塞当NPU死锁。正式原ATK、完整main/metadata接口和0.8性能门禁
在未执行前均保持NOT_RUN；独立阈值、C ABI计时和脚本退出0不赋予这些正式门禁通过资格。
