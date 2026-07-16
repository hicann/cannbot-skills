# 分段映射

本技能用于 ops-registry-invoke 的 1.3 方案设计。生成过程分成可并行的设计章节和一个独立的迭代计划文档。

## Phase 1: DESIGN.md 分段

运行 `scripts/slice_design_inputs.py` 默认生成以下 bundle。除 `05-plan.md` 外，其他 bundle 的结果都写入 `operators/<op>/.spec-to-design/sections/` 并由 `assemble_design.py` 合并进 `DESIGN.md`。

| Bundle | 输出章节 | 负责 Agent |
| --- | --- | --- |
| `01-overview-contract.md` | `## 修订记录`、`## 1. 概述` | `ascendc-ops-designer`（scene: generate-section-01） |
| `02-architecture.md` | `## 2. 架构设计` | `ascendc-ops-designer`（scene: generate-section-02） |
| `03-implementation.md` | `## 3. 实现方案` | `ascendc-ops-designer`（scene: generate-section-03） |
| `04-quality-plan.md` | `## 4. 性能优化`、`## 5. 风险评估`、`## 6. 交付件清单`、`## 7. 迭代规划`、`## 8. Design Contract` | `ascendc-ops-designer`（scene: generate-section-04） |

并行生成时，每个分段只返回指定章节 markdown，不要包含文档标题。

## Phase 2: PLAN.md

`05-plan.md` 生成完整 `PLAN.md`，首行必须是：

```markdown
# <operator_name> 迭代执行计划
```

结果保存为 `operators/<op>/.spec-to-design/sections/05-plan.md`，`assemble_design.py --plan-output` 会把它写到 `operators/<op>/docs/PLAN.md`。负责 Agent：`ascendc-ops-designer`（scene: generate-section-05）。

## 子任务调度

5 个 bundle 由主 Agent 并行调度对应的 `ascendc-ops-design-*` Agent 完成（必须同一次响应发起）。每个分段的输入、输出、内容约束以各 Agent 定义（`agents/ascendc-ops-design-*.md`）为唯一权威源；Task 调用参数模板见 workflow 的 `resources/task-prompts.md#13c-并行分段生成`。本文件不再维护提示词全文。

## 合并契约

- `DESIGN.md` 章节必须使用模板中的 `##` 标题，标题文字不能改名。
- `PLAN.md` 是独立文档，不能混入 `DESIGN.md` 章节文件。
- `DESIGN.md` 必须保留模板顶层章节顺序。
- `DESIGN.md` 必须包含「spec.yaml 一致性映射」章节或小节。
- 如果某字段无法由 spec 或需求支持，写明“待补充/需回到 spec-generation 修订”，不要补造结论。
- 主 Agent 在组装前必须检查跨章节一致性：dtype、shape、TilingKey、API 验证状态、UB 容量、迭代计划必须一致。
