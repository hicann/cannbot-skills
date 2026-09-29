# PR 检视场景（轻量流程）

## 触发
检视 PR、review MR、merge_requests/\d+、pull/\d+

## 依赖托管

PR 上下文拉取（clone/checkout/diff/token）**全权托管 infra 家族 skill**：

- 首选 `infra/gitcode-pr-handler`（PR 处理全流程，含上下文拉取与回评）
- 退化 `infra/gitcode-toolkit` 的 `scripts/fetch_pr_context.py`
  （一站式：clone → checkout PR → merge-base → diff 统计）

**依赖检查**：两者都不存在 → 通知用户"PR 检视依赖 infra 家族的
gitcode-pr-handler / gitcode-toolkit skill，请先安装"，停止执行，
不要临时拼装拉取命令。Token 解析同样交给 infra 家族处理。

---

## 编排

### 阶段0：拉取上下文

调用 infra 家族获取：`work_dir`（PR 源码目录）、`merge_base`、
`changed_files`。

### 阶段1：圈定检视范围

从 changed_files 筛出 DSL 算子文件（`*.py`，排除测试/文档）；无算子
文件 → 告知用户并结束。记录各文件的 diff 变更行区间。

### 阶段2：派发可读性检视子 Agent

同 `workflows/file-review.md` 阶段1，每个变更算子文件派 1 个子 Agent，
**额外传入 diff 变更行区间**：FAIL/SUSPICIOUS 必须锚定区间内，区间外
标 `out_of_diff: true` 只进附录。

### 阶段3：撰写报告

按 `steps/report-write.md` 模板输出
`./{算子名}_pr{number}_review_report.md`，头部注明 PR 链接与
merge_base；out_of_diff 历史问题只进附录"上下文观察"，不进统计。

报告撰写遵循与文件检视相同的三原则（清晰、可读、简洁），完整要求见
`workflows/file-review.md` 阶段2：结论先行、全部问题一张表一行一条、
证据链一行压缩、片段非必要不给、待确认写明确认什么、不写元信息。

### 回评（可选）

回评也交给 infra 家族（gitcode-pr-handler / toolkit 的评论工作流）。
**未经用户明确确认，禁止向 PR 提交任何评论。**

---

## 约束

- 只对 diff 变更代码报告 FAIL/SUSPICIOUS
- 回评必须经用户确认
