# {{ op.name }} 设计文档

本模板用于从 `spec.yaml` 和 `REQUIREMENTS.md` 生成算子仓详细设计文档。最终产物必须使用简体中文；代码符号、API 名称、dtype、文件名和 YAML key 保留英文。

## 修订记录

生成修订记录表。

要求：
- 初始版本写 `v1.0`。
- 修订内容写“基于 spec.yaml 生成初始方案设计”或更具体的实际变更。
- 修订时间使用当前日期或任务上下文日期。

## 1. 概述

### 1.1 基本信息

用表格列出：

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `{{ op.name }}` |
| 算子类别 | 来自 `op.category` |
| 范式 | 来自 `op.paradigms` |
| 条件范式路由 | 若 `op.paradigm_routing` 存在，按 attribute → cases 列出每个分区的激活范式子集；不存在则写"无" |
| 支持数据类型 | 来自 `inputs[].dtype_set` 与 `dtype_policy.supported_combinations` |
| 目标芯片 | 来自 `op.platform_constraints.supported_chips` 或 REQUIREMENTS 运行环境 |
| 目标架构 | 根据目标芯片映射：Ascend910B/Ascend910_93 -> DAV_2201/arch22；Ascend950DT/Ascend950PR -> DAV_3510/arch35 |
| 编程框架 | AscendC |

### 1.2 算子功能

基于 `op.description`、`math_semantics.formula` 和 REQUIREMENTS 描述算子功能。不要引入 spec 未声明的语义。

### 1.3 数学公式

用 LaTeX 或清晰公式描述 `math_semantics.formula`。说明输入、输出和属性含义。

### 1.4 spec.yaml 一致性映射

必须用表格逐项说明 spec 字段在 DESIGN.md 中的承接位置：

| spec 字段 | 设计承接位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| `op.paradigm_groups` | 章节路径 | 已承接/待补充 | fusion/combination 关系在设计中的承接 |
| `reduction` | 章节路径 | 已承接/待补充 | axis_source 对 Host Tiling / Kernel 的影响 |
| `dtype_policy.supported_combinations` | 章节路径 | 已承接/待补充 | 说明 |
| `outputs[].shape_rule` / `broadcast` | 章节路径 | 已承接/待补充 | 说明 |
| `math_semantics.formula` / `reference_oracle` | 章节路径 | 已承接/待补充 | 说明 |
| `boundary_conditions` / `extreme_inputs` | 章节路径 | 已承接/待补充 | 说明 |
| `numerical_tolerance.per_dtype` | 章节路径 | 已承接/待补充 | 说明 |
| `determinism` | 章节路径 | 已承接/待补充 | 说明 |

若字段缺失或无法用于设计，写明“需回到 spec-generation 修订”。

## 2. 架构设计

### 2.1 逻辑视图

描述 `op_api`、`op_host`、`op_kernel`、`op_graph` 的职责和核心文件。

要求：
- ACLNN 接口参数必须来自 REQUIREMENTS 的接口定义或 `interface_binding`。
- 输入、输出数量和名称必须与 `spec.yaml` 一致。
- 明确 ACLNN 接口生成路径：指定 ACLNNTYPE 配置值（默认推荐 `aclnn_exclude`，手写 L2+L0；若 `aclnn` 则 L2 由构建系统自动生成，L0 由框架运行时处理，无需交付 op_api），以及 op_api/ 下各文件的手写/自动生成划分。

### 2.2 开发视图

给出算子仓目录结构，至少包含：

```text
operators/{{ op.name }}/
├── CMakeLists.txt
├── README.md
├── op_api/
├── op_host/
├── op_kernel/
├── op_graph/
└── tests/
```

按实际目标架构说明 `arch22` 或 `arch35` 目录。

### 2.3 运行视图

必须包含简洁数据流图或步骤列表，明确：
- GM 输入读取
- UB/L1/workspace 使用情况
- 计算步骤
- GM 输出写回

### 2.4 用户视图

说明 ACLNN 调用和图模式调用方式。不得写未在需求中承诺的调用入口。

## 3. 实现方案

### 3.1 技术路径决策

技术路径选型：RegBase / MemBase / SIMT / 混合。说明选型理由及架构约束。

### 3.2 模板划分总览

基于 `dtype_policy`、`outputs[].shape_rule`、`broadcast`、`boundary_conditions`、目标芯片和性能目标规划 TilingKey/模板。

必须包含：
- 模板划分表
- 每个模板的触发条件
- dtype 覆盖
- shape/layout 覆盖
- 该模板存在的原因

若 `op.paradigm_routing` 存在，模板划分表必须增加「范式分区」列，按 paradigm_routing.cases 分行。每个分区对应独立的 Kernel 模板架构（不同的数据流、Buffer 模型和 API 调用序列），TilingKey 须编码分区标识。示例表头：

| 模板 ID | 范式分区 | 激活范式 | 触发条件 | dtype 覆盖 | shape 覆盖 | 原因 |
| --- | --- | --- | --- | --- | --- | --- |

### 3.3 TilingData 结构体

设计 TilingData 字段。字段必须能支撑模板选择、shape 推导、tile 大小、循环次数、边界处理和 workspace/UB 规划。

### 3.4 Host 侧 Tiling 方案

说明 Host 侧如何读取输入 shape/dtype/属性，如何计算 TilingKey、TilingData 和 workspace。

要求：
- shape 和 dtype 规则必须直接对应 `spec.yaml`。
- 不得使用废弃宏 `BEGIN_TILING_DATA_DEF` 或 `TILING_KEY_IS`。
- TilingKey 选择建议使用模板参数和 `ASCENDC_TPL_SEL_PARAM`。
- 若 `op.paradigm_routing` 存在，Host 侧必须读取 `paradigm_routing.attribute` 属性值，映射到对应的范式分区 ID，并写入 TilingKey/TilingData。不同分区的 TilingData 字段可不同。

### 3.5 Kernel 侧计算方案

描述 Kernel 模板实例化、Init、Process、主循环、尾块处理、边界 case 和输出写回。

要求：
- 算法步骤必须足够具体，使开发者能按文档实现。
- 空 Tensor、标量、广播、rank 边界等有效 case 必须与 spec 一致。

### 3.6 API 映射

用表格记录计算步骤到 Ascend C API 的映射：

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | dtype 支持 | 约束说明 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |

不得把未验证 API 写成“已验证”。如果文档未查到，标记“待验证”并说明后续动作。

### 3.7 API 验证记录

记录每个关键 API 的可信来源：

| API 名称 | 官方文档路径或来源 | 通配符搜索结果 | 验证状态 | 备注 |
| --- | --- | --- | --- | --- |

可信来源只包括官方文档和用户明确提供的资料。

### 3.8 数据流设计

说明每个模板的数据搬入、计算、临时缓冲、数据搬出。可使用 Mermaid 或编号步骤。

### 3.9 内存管理

列出 UB、L1、workspace、scalar/register 元数据和 double buffer 决策。

必须包含 Buffer 大小公式，并说明 dtype 对 element size 的影响。

### 3.10 UB 容量验证

用目标架构的 UB 预算验证：

| 平台 | 可用 UB | Buffer 总和公式 | 验证结论 |
| --- | --- | --- | --- |

已知默认值：
- DAV_2201: 总 UB 192 KB，向量可用按 184 KB 估算。
- DAV_3510: 总 UB 248 KB，向量可用按 248 KB 估算。

若算子需要更精确预算，写明需要通过平台文档或运行环境确认。

## 4. 性能优化

### 4.1 并行策略

说明核间切分、block 维度、tile 粒度和负载均衡策略。

### 4.2 流水线设计

说明是否使用 double buffer、CopyIn/Compute/CopyOut 流水和同步策略。

### 4.3 性能目标与基线

承接 REQUIREMENTS 或 spec 中的性能字段。若无明确性能目标，写“需求未给出明确性能基线，后续以功能正确和基础带宽利用率为验收起点”。

## 5. 风险评估

### 5.1 API 风险

列出 API 可用性、dtype 支持、平台差异和参数约束风险。

### 5.2 精度风险

承接 `numerical_tolerance`、`numerical_stability`、`extreme_inputs` 和 `determinism`，说明风险与应对。

### 5.3 性能风险

说明带宽、UB 容量、workspace、broadcast、尾块和多 dtype 对性能的影响。

### 5.4 应对措施

给出可执行的规避或验证动作。

## 6. 交付件清单

列出必须交付的代码、测试和文档：

- `op_api/`
- `op_host/`
- `op_kernel/`
- `op_graph/`
- `tests/ut/`
- `tests/st/`
- `CMakeLists.txt`
- `README.md`
- `docs/TEST.md`
- `docs/PLAN.md`

## 7. 迭代规划

在 DESIGN.md 中给出摘要表，并与独立 `PLAN.md` 保持一致。表格行数根据 PLAN.md frontmatter 的 `iteration_count` 动态生成：

**iteration_count = 1**：

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
| --- | --- | --- | --- | --- |
| 迭代一 | 全功能实现 | 全功能 + 全 dtype + 边界 + 广播 | 全覆盖用例 | L0+L1 全覆盖 |

**iteration_count = 2**：

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
| --- | --- | --- | --- | --- |
| 迭代一 | 骨架搭建 | 单 TilingKey + 单 dtype + 核心路径 | 核心路径用例 | L0 基础用例 |
| 迭代二 | 整合 + 全功能 | 整合穿刺 + 全 dtype + 边界 + 广播 | 全覆盖用例 | L0+L1 全覆盖 |

**iteration_count = 3**：

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
| --- | --- | --- | --- | --- |
| 迭代一 | 骨架搭建 | 单 TilingKey + 单 dtype + 核心路径 | 核心路径用例 | L0 基础用例 |
| 迭代二 | 策略完善 | 多 TilingKey + 单 dtype | Tiling 分支覆盖 | 多 shape 用例 |
| 迭代三 | 规格完整 | 全 dtype + 边界 + 广播 | 全覆盖用例 | 全 dtype + 边界 + 广播 |

## 8. Design Contract

> 以下章节为开发阶段的强制约束，Developer 实现必须逐条符合。

| 约束类别 | DESIGN.md 来源 |
|---------|---------------|
| Programming Model | §3.1 |
| Compute Chain | §3.5, §3.6 |
| API Constraints | §3.6, §3.7 |
| Precision Strategy | §4.2, §5.2 |
| Memory Strategy | §3.9, §3.10 |
