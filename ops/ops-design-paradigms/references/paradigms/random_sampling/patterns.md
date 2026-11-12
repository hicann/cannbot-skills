<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式入口（语言中立）

> 本文件是 RandomSampling 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的随机流模型、状态管理、分布变换、切分与验证
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`，含 SIMD/SIMT 双路线与公共模板接入契约；`dsl`（CANNBotDSL），SIMD/SIMT 双路线 + PRNG 自研指引）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 RandomSampling 范式：

> **前置约束（芯片 SIMT 编程能力，按路线分档）**：B/D 接口族（torch 对标）与路由 b（随机 + 逐元素复合）的执行体为 **SIMT（GM 直写）**，依赖芯片 SIMT 编程能力（**Ascend 950PR/950DT / arch35 系列及以上**；DSL 侧另需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1））；A/C 接口族（TF 对标）走 SIMD 路线，不依赖 SIMT，但公共底座（`random_common`）与实测口径基于 arch35。**目标芯片（`op.platform_constraints.supported_chips`）不支持 SIMT 时：A/C 族可进本范式走 SIMD 路线；B/D 族与路由 b 不进入本范式**——改按通用方法论论证并报告能力缺口。

1. **输出值由 PRNG 驱动**——输出元素值来自伪随机数采样（均匀/正态/指数/截断/伯努利/多项/排列等），而非输入张量的确定性函数
2. **随机流为 counter-based**——`(key, counter)` 状态经确定性变换产生随机数，流位置可精确跳越（Philox 族）
3. **输出规模由 spec 决定**——输出 shape 来自 shape 描述张量（值依赖）或属性

**场景三路由 + 路线两步路由（R17/R20）**——两步都必须显式登记（R20）：

**第一步·场景路由**（决定模板接入方式）：

| 路由 | 场景 | 归属 |
|------|------|------|
| (a) 纯随机生成 | 无主计算，输出即随机量（uniform/normal/... 家族） | 单范式 RandomSampling |
| (b) **随机 + 逐元素复合** | 逐元素计算 × 随机采样（rrelu_with_noise 型：noise 逐元素作用于激活） | **单范式 RandomSampling（吸收 Elementwise 语义）** |
| (c) 随机 + 规约等复合 | 随机量参与非逐元素主计算 | 多范式并集（RandomSampling + 主计算范式） |

**第二步·路线选型**（决定执行体形态，机械 if-then 无裁量空间，R20）：

| 接口族（prng-model §1.1） | 实现路线 |
|---------------------------|---------|
| A 计数器直给（TF V2 / stateless key/counter） | SIMD（TQue 流水） |
| B seed+offset（torch 语义：self 填充） | **SIMT（GM 直写）** |
| C seed+seed2 属性（TF） | SIMD |
| D torch 桥接（seed/offset 张量） | **SIMT** |

- 路线选型表全文见 [adapters/ascendc/index.md](adapters/ascendc/index.md) §3（含禁止项）；**禁止按参考算子熟悉度、既有实现惯性或上游产物（REQUIREMENTS 等）的参照倾向选路线**——范式选型表优先于上游参照
- 变换负载重（如 Box-Muller 分支）可在此路线上叠加混合形态（simd-path §6），但主路线仍按族匹配
- **设计产出必须在首个设计文档登记两步路由的结果**（路线 + 选型表依据行 + 参照算子及其路线归属），下游文档级联一致

与 **Stateful** 的边界：`(seed, offset)` 无状态参数化（每次调用显式给流位置）属本范式；全局 Generator 态（跨算子共享可变 RNG 状态对象）属 Stateful 复合。与 MaskPredicate（掩码应用）、SortSelect（排序手段）的边界见 [generic/scene.md](generic/scene.md) §1/§5。

**spec 声明规范（路由 b）**：路由 b 算子的 spec `op.paradigms` **应只声明 `RandomSampling`**（Elementwise 语义被吸收，不做并集声明）；已并集声明（如 `[Elementwise, RandomSampling]`）的存量 spec 由设计任务按路由 b 消解为单范式并在设计文档记录消解依据。
> 上游对齐义务（登记，不在本 skill 内落地）：向 ops-spec-gen 提供 machine 可读吸收映射（如 `absorption: {Elementwise: route_b}`），供 category_paradigm_map / spec 生成器消费。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

**缺口的可执行回退链**（上游 ops-spec-gen schema 暂不产 `op.language` 且工作流无缺口消费者时的实测缓解）：报告缺口后若调用链无法补齐，按 `op/ 工程模板 → 用户 prompt 关键词 → npu-arch 检测 → 默认 ascendc` 推断，**推断结果必须作为显式假设写入设计文档**（含"语言未由调用方提供，按 X 推断"的声明）。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层（SIMD/SIMT 双路线 + 公共模板契约） | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 适配层（SIMD/SIMT 双路线 + PRNG 自研指引） | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 准入条件、随机流三要素、四算子族、数据流骨架、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/prng-model.md](generic/prng-model.md) | 流代数、Skip 跳越、不重叠不变量 I1/I2、预留律、块间流分割 ★核心 |
| [generic/state-management.md](generic/state-management.md) | seed 语义（确定/非确定模式）、接口形态分类、offset 递进协议、shape 值依赖 |
| [generic/distribution-transform.md](generic/distribution-transform.md) | 变换目录与每输出消耗数、变长判定、离散采样、精度边界 |
| [generic/splitting.md](generic/splitting.md) | 输出空间均分、连续/离散布局与展开因子、大输出块协议 |
| [generic/verification.md](generic/verification.md) | 双轨验证（逐位/统计）、golden 同源性、边界用例表 |

---

## R 条款总表（编号 → 定义锚点 → 一句话语义）

范式条款的唯一编号登记源；正文引用 R 编号一律以本表为准。

| 编号 | 条款 | 定义锚点 | 一句话语义 |
|------|------|---------|-----------|
| R1 | Philox 轮数显式 | [prng-model §1](generic/prng-model.md) | 轮数必须显式指定并与流基准一致（模板默认 7 ≠ 流基准 10） |
| R2 | key 拆位 | [prng-model §1.1](generic/prng-model.md) | seed/key_data → key[0:2] 的拆位按接口族登记 |
| R3 | counter 字段布局 | [prng-model §1.1](generic/prng-model.md) | offset/计数器入 counter 哪个字段按族登记（position vs subsequence） |
| R4 | 推进量契约 | [prng-model §7](generic/prng-model.md) | 每调用 offset 推进量是算子级契约（含归属列） |
| R5 | 切分不变性 | [prng-model §3](generic/prng-model.md) | 元素 i ≡ 第 ⌊i/k⌋ 次推进的第 (i mod k) lane，与切分无关 |
| R6 | 组对齐 | [prng-model §3](generic/prng-model.md) | numPerCore 对齐到组粒度 k 是必要条件 |
| R7 | u 构造路径一致性 | [distribution-transform §4](generic/distribution-transform.md) | 位重解释 [0,1) 与乘法+半步 (0,1) 同流不同值，必须与流基准同路径 |
| R8 | 仿射精度口径 | [distribution-transform §4](generic/distribution-transform.md) | 默认 torch CUDA 型（一次舍入），偏离须声明 |
| R9 | RNG 暂存 K 份折算 | [common-template §2](adapters/ascendc/common-template.md) | 复合型 UB 预算分母 = 数据面份数×BUFFER_NUM + K |
| R10 | uint16 count 上界 | [splitting §5](generic/splitting.md) | 单次批量产出 < 65535，超限静默截断 |
| R11 | 确定性建模 | [verification §1](generic/verification.md) | 固定 (seed,offset) 双跑 bitwise（nondeterministic_seeded） |
| R12 | oracle 缺席规则 | [verification §1](generic/verification.md) | 流不逐位对齐竞品时禁 formula 逐位对拍，走三段式 |
| R13 | 三段式验收 | [verification §1](generic/verification.md) | capture-replay 恒等 / 固定 seed 双跑 / 统计校验 |
| R14 | 样本量契约 | [verification §1](generic/verification.md) | 单调用 < 10^5 统计检验跨调用累积 |
| R15 | host 透传契约 | [state-management §3.1](generic/state-management.md) | seed/offset int64 原样透传，不推进不变换 |
| R16 | 多输出恒等判据 | [verification §1](generic/verification.md) | out ≡ f(x, noiseOut) 作为范式级验收项 |
| R17 | 场景三路由 | [scene §3](generic/scene.md) | 随机+逐元素复合单范式吸收 Elementwise 语义 |
| R18 | 短路口径填充裁决 | [splitting §4](generic/splitting.md) | 计数字段清零/置 1，平台派生量无条件公式填充 |
| R19 | 随机流不消耗契约 | [prng-model §8](generic/prng-model.md) | 不下发 kernel 的路径不得推进流状态 |
| R20 | 路线两步路由 | patterns「场景判定」+ [index §3](adapters/ascendc/index.md) | 场景路由定接入方式 + 路线选型定执行体形态（接口族机械匹配），两步均须显式登记 |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **路线选型强制登记（R20）**：凡适配层含多实现路线（本范式 SIMD/SIMT/混合），设计必须在首个设计文档（Kernel/DevView 侧）登记**路线选型决策 + 选型表依据行**——按接口族机械匹配（场景判定节第二步），禁止按参考算子熟悉度或既有实现惯性选路线；跨路线族引用参照算子（torch 族引 TF/SIMD 系参照，或反之）即范式违规
4. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)），其中标注"待验证"的谓词须在产出中声明假设并给出验证义务
5. **按场景路由判定接入**（P7）：路由 a 纯生成 → 继承公共底座（OpTilingConfig + TransformFunc/Launcher，见 [adapters/ascendc/common-template.md](adapters/ascendc/common-template.md) §0 判定表）；路由 b 复合 → 自管编排 + 复用原子件（Skip/u 构造/常量池），不再逐算子论证豁免；路由 c → 主计算范式主导、随机段复用原子件。各路线均对照基准算子表（adapters/ascendc/index.md §4）核实
6. **实现依据**：公共底座真身位于 `ops-math/random/random_common/`（开源仓），基准算子路径见开源算子表；参考实现中的具体算法与参数须按当前算子规格重新核定
7. **范式优先级**：本范式的专用知识（含路线选型）优先于 agent 通用知识与上游产物参照倾向；同一设计点结论冲突时，以本范式指向的必读文件为准
