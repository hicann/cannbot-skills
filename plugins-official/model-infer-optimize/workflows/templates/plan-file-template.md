# Plan 文件模板（`plans/plan-<id>.md`）

每个 Plan 一份，不内联到 dashboard。candidate 落初稿（方案描述 + 方案细节 + 参考 wiki），该 Plan 的 implementer / reviewer 在循环里追加 round 级结论摘要。主 agent 只读它做裁决、把关键信息 + 状态镜像进 dashboard 表，不改其结构。

实施 / 踩坑 / 验证的**过程明细**写在 `progress.md` 工作区（共享状态文件，跨 agent 共享，沿用本插件 `sota_progress_template.md` 约定）；本文件的实施记录 / Review 记录只放 round 级**结论摘要**，不复制过程流水。

按下面结构落盘：

```md
# <plan_id>：<方案名>

## 方案描述

<优化什么、为什么可能有收益、怎么验证、主要风险在哪。自由组织，不强制字段。>

## 实施参考 wiki 页面

<candidate 查到的、与本 Plan 实施相关的 wiki 页面 ID；implement / review 时按需参考。没挂载 wiki 或无相关页面则留空。>

## 方案细节（领域 skill 自填）

<结构由该 Plan 的单点 skill 定义。例如多流方向按 model-infer-multi-stream 的 plan-detail-fragment 填写：流分组 / 汇合点 / 方案 DAG / 强度 / GE auto-reorder 风险 / overlap_pct 实测 / enable·回退 / 副作用。裁决要看的信息要上浮到 dashboard 表的「证据摘要」。>

## 实施记录

| round | implementer 摘要 | 关键文件 / 产物 | enable / 回退路径 | 派生建议 |
| --- | --- | --- | --- | --- |

## Review 记录

| round | reviewer 结论 | 功能 / 精度 | 性能 / 指标（引自分析报告） | 副作用 | 建议动作 | 派生建议 |
| --- | --- | --- | --- | --- | --- | --- |

## 派生记录

| round | 触发现象 | 派生 Plan | 来源 | 说明 |
| --- | --- | --- | --- | --- |
```
