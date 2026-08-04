---
name: ascendc-blaze-best-practice
description: 在 Ascend 950 / DAV_3510 平台上，基于 Blaze/tensor_api 开发 Basic、Batch、Grouped、Quantized、MX 等 MatMul 类算子或相关的融合算子时使用。不适用于纯 Vector 算子或 A2/A3 平台。
---

# Blaze MatMul 算子开发指南

## 路径与文档合同

将 `<project-root>` 解释为最高层项目根，将 `<operator_name>` 解释为 `operators/` 下的算子目录名：

```text
project_root: <project-root>
operators_root: <project-root>/operators/
operator_root: <project-root>/operators/<operator_name>/
blaze_source_root: <project-root>/ops-tensor/
investigation_report: operators/<operator_name>/docs/blaze/blaze-investigation-report.md
design_path: operators/<operator_name>/docs/DESIGN.md
plan_path: operators/<operator_name>/docs/PLAN.md
```

只把 `<project-root>/ops-tensor/` 作为 Blaze 源码事实根；它与 `<project-root>/operators/` 同级。不得回退读取算子目录内、其他项目或历史工程的 ops-tensor。`project_contract_id` 是逻辑合同 ID，不是路径变量。

统一使用以下模板；本文中的 DESIGN、PLAN 分别指其生成的 `DESIGN.md`、`PLAN.md`：

- [Blaze DESIGN 模板](references/kernel-design/blaze-design-template.md)
- [Blaze PLAN 模板](references/kernel-design/blaze-plan-template.md)

Blaze skill 不读取或依赖 `environment.md`、外部 manifest 或其他工作流产物。调用方可以直接传入已经确认的 `target_chip`、`npu_arch` 和可选 `cann_version`。

## 按请求目的路由

| 请求目的 | 执行 |
|---|---|
| 明确要求“开发算子”或等价的从零完整开发 | Step 1 → Step 2 → Step 3 → Step 4 |
| 明确要求算子设计、方案分析并输出设计文档 | Step 2 → Step 3 |
| 咨询、解释、评审、排障、能力查询 | 只读取相关 references |

不要按调用者身份增加模式。不要因发现已有 DESIGN/PLAN 而单独进入 Step 4；direct invoke Developer 直接消费相同 DESIGN/PLAN，Blaze Step 4 只属于本 skill 的完整四步开发流程。

## 四步流程

### Step 1: Project Setup

创建 `operator_root`，在 `blaze_source_root` clone 或更新授权 ops-tensor，递归初始化 submodule，确认抽象版本一致性，并建立根源码、项目 Blaze 副本、项目 tensor_api 副本三个只读区。不要预先实现公式、Tiling、Golden、固定工程或场景 recipe。

→ [Step 1: Project Setup](references/workflow/step1-project-setup.md)

### Step 2: Blaze Investigation

从 `project_root` 和 `operator_name` 派生路径并只读自检当前 `blaze_source_root`；不要依赖 Step 1 文件产物，不要 clone、更新、切换源码或读取场景资料。按需求语义调查候选 Blaze 组装方案、物理数据和 ABI 事实，只生成 Investigation。

→ [Step 2: Blaze Investigation](references/workflow/step2-blaze-investigation.md)

### Step 3: Kernel Design

依据需求和 Investigation 完成逻辑接口、Blaze 官方方案、必要的唯一 custom 场景、最终 ABI/资源/验证合同，并用统一模板生成 DESIGN 和可执行路线的 PLAN。`unsupported` 只生成 DESIGN；一次补充调查后仍缺决定性事实时不生成最终 DESIGN/PLAN。

→ [Step 3: Kernel Design](references/workflow/step3-kernel-design.md)

### Step 4: Implementation

只在完整四步流程中执行当前 `operator_root` 的 DESIGN/PLAN。核对联合门禁，按冻结的第 9、10 章执行；持续更新 PLAN 第 2、4--8 章并只追加第 11 章。不得重新匹配场景、选择路线/候选、改变接口/ABI、切换备选或扩大支持域。

→ [Step 4: Implementation](references/workflow/step4-operator-development.md)

## 路线模型

```text
implementation_route: blaze_native | blaze_custom | unsupported
selected_scenario: <仅 blaze_custom 填写>
unsupported_points: <仅 unsupported 填写>
```

只在 Step 3 决定项目路线。官方 Blaze 覆盖全部 required partitions 时选择 `blaze_native` 且不读取场景注册表；存在证据闭合的 native gap 且场景唯一命中时选择 `blaze_custom`；证据充分且场景零命中或多命中时选择 `unsupported`。

## 扩展场景适配指导

仅在维护本 skill 的扩展能力时读取[场景接入指导](references/scenarios/scenario-extension-guide.md)。在 `references/scenarios/<scenario-id>/` 内维护场景设计和开发资料，并完成索引注册、源码前提、DESIGN/PLAN 合同和唯一匹配约束；普通算子流程不自动读取其他场景。
