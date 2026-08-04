# Task 调用契约

> 各步子 Agent 的调用参数、输入/输出、验收标准。编号与 [SKILL.md 统一流程表](../SKILL.md#统一流程表) 一一对应。
>
> **调用原则**：PM 每阶段首次调度子 Agent 时，必须严格按照本文档指定的角色和 prompt **原样调用**，仅允许替换 prompt 中的 `<算子名>` 项，**严禁干涉实现细节**。

# 阶段 0：开发准备

## 0 开发准备

- **角色**：developer

```md
- 【权限】你只可写 `.cannbot/环境信息.md`，其它写入操作会被 hooks 拦截。
- 【输出】填充模板：`workflow-doc-templates/references/环境信息.md`，写入 `.cannbot/环境信息.md`。
- 【skills】立即加载 `workflow-doc-templates`、`ascendc-env-check`。
- 【验收标准】环境信息文档完整，各检查项有明确结论；Git 凭据位置候选已列出且不含任何凭据明文。
```

## CP0 环境确认

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`）、`.cannbot/环境信息.md`（按用户结论更新「Git 凭据」节），其它写入操作会被 hooks 拦截。
- 【输入】环境信息文档 `.cannbot/环境信息.md`。
- 【输出】环境确认结果（不通过，立即停止当前工作流流程 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp0`、`workflow-doc-templates`。
```

# 阶段 1：需求分析

## 1.1 需求分析

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】对话上下文、仓库设计约束。
- 【输出】需求文档，写入 `.cannbot/<算子名>/1.1-需求分析.md`；格式模板：`workflow-doc-templates/references/1.1-需求分析.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`ascendc-regbase-best-practice`、`ascendc-simt-best-practices`。
- 【验收标准】确认项无遗漏，用户原始需求逐条记录；只出推荐，不代替用户决定架构。
```

## CP1 需求确认

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`），其它写入操作会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】需求确认结果（不通过，打回 1.1 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp1`、`workflow-doc-templates`。
```

# 阶段 2：方案设计（方案线 / 测试线并行）

## 2.1 黑盒测试设计

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】测试方案文档，写入 `.cannbot/<算子名>/2.1-测试方案设计.md`；格式模板：`workflow-doc-templates/references/2.1-测试方案设计.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-test-develop`、`ascendc-st-design`。
- 【验收标准】golden 方案可行，分级用例覆盖充分（正常/边界/异常/特殊值），覆盖矩阵齐备。
```

## CP2.1 测试检查

- **角色**：QA

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】测试方案文档 `.cannbot/<算子名>/2.1-测试方案设计.md`。
- 【输出】测试检查结论（不通过，打回 2.1 / 通过）
- 【skills】立即加载 `workflow-cp2-1`、`workflow-doc-templates`。
```

## 2.2 开发方案设计

- **角色**：architect

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】需求文档 `.cannbot/<算子名>/1.1-需求分析.md`。
- 【输出】开发方案文档，写入 `.cannbot/<算子名>/2.2-开发方案设计.md`；格式模板：`workflow-doc-templates/references/2.2-开发方案设计.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`repo-op-templates`、`repo-build-guide`、`ascendc-tiling-design`。
- 【验收标准】代码架构与需求文档拍板结果一致，Tiling/切分策略可行，关键 API 已验证；不改选已拍板的代码架构。
```

## CP2.2 方案检查

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/questionnaires/`（问卷 json 与同名 `.reply.json`），其它写入操作会被 hooks 拦截。
- 【输入】开发方案文档 `.cannbot/<算子名>/2.2-开发方案设计.md`。
- 【输出】方案检查结果（不通过，按语义归属打回 2.2 或 1.1 / 通过）；问卷与回复落盘 `.cannbot/<算子名>/questionnaires/`。
- 【skills】立即加载 `workflow-cp2-2`、`workflow-doc-templates`。
```

# 阶段 3：代码开发（开发线 / 测试线并行）

## 3.1 算子开发

- **角色**：developer-code

```md
- 【权限】你可写算子代码目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】开发方案文档 `.cannbot/<算子名>/2.2-开发方案设计.md`；被打回时附结构化修改要求。
- 【输出】算子代码：按开发方案实现，编译验证通过。
- 【skills】立即加载 `repo-op-templates`、`repo-coding-rules`、`repo-build-guide`、`repo-knowledge`、`ascendc-direct-invoke-template`、`ascendc-api-best-practices`。
- 【验收标准】编译通过，按方案实现；性能/正确性瓶颈定位后回退给调用方，不自行改 Tiling/切分/接口。
```

## 3.2 测试工程开发

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】测试方案文档 `.cannbot/<算子名>/2.1-测试方案设计.md`。
- 【输出】golden 代码 + 用例表 + 性能采集框架，写 test 目录。
- 【skills】立即加载 `repo-test-develop`、`ops-precision-standard`、`ops-profiling`。
- 【验收标准】golden 可产出期望输出，用例表按分级覆盖，性能采集框架可用。
```

## 3.3 白盒测试补全

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码、测试代码。
- 【输出】白盒用例 + 分支覆盖说明，写 test 目录。
- 【skills】立即加载 `repo-test-develop`、`ascendc-whitebox-design`。
- 【验收标准】声明的执行分支 / tilingkey 覆盖达标（阈值由对应验收 skill 给定），产出分支覆盖说明。
```

## CP3 功能验收

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP3-功能验收报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 测试代码。
- 【输出】功能验收结果（不通过，回退 3.1 / 通过）；验收报告写入 `.cannbot/<算子名>/CP3-功能验收报告.md`，格式模板：`workflow-doc-templates/references/CP3-功能验收报告.md`。
- 【skills】立即加载 `workflow-cp3`、`workflow-doc-templates`。
```

# 阶段 4：性能迭代与验收

## 4.1 性能采集执行

- **角色**：developer-test

```md
- 【权限】你可写 test 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码（功能验收通过后的最终代码）。
- 【输出】性能数据（各 shape/dtype 的耗时、带宽、利用率），落 test 目录采集输出。
- 【skills】立即加载 `repo-test-develop`、`ops-profiling`。
- 【验收标准】性能数据完整覆盖需求关注的 shape/dtype。
```

## 4.2 性能迭代

- **角色**：developer

```md
- 【权限】你可写算子代码目录、`.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 性能数据（4.1 基线）；共享依赖仓 `.cannbot/cann-samples/`。
- 【输出】优化后算子代码 + 性能迭代记录（逐次优化路径与优化效果），写入 `.cannbot/<算子名>/4.2-性能迭代记录.md`。
- 【skills】立即加载 `ascendc-perf-optimize`、`repo-coding-rules`、`repo-build-guide`。
- 【验收标准】适用的优化路径均已尝试并逐次记录效果；全量用例回归验证通过。
```

## CP4 性能验收

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP4-性能验收报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 性能数据。
- 【输出】性能验收结果（不通过，回退 3.1 / 通过）；验收报告写入 `.cannbot/<算子名>/CP4-性能验收报告.md`，格式模板：`workflow-doc-templates/references/CP4-性能验收报告.md`。
- 【skills】立即加载 `workflow-cp4`、`workflow-doc-templates`。
```

# 阶段 5：代码检视

## CP5 代码检视

- **角色**：QA

```md
- 【权限】你可写 `.cannbot/<算子名>/CP5-代码检视报告.md`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部变更文件。
- 【输出】代码检视结果（不通过，回退 3.1 / 通过）；检视报告写入 `.cannbot/<算子名>/CP5-代码检视报告.md`，格式模板：`workflow-doc-templates/references/CP5-代码检视报告.md`。
- 【skills】立即加载 `workflow-cp5`、`workflow-doc-templates`。
```

# 阶段 6：上库准备

## 6.1 文档补全

- **角色**：developer-doc

```md
- 【权限】你可写 doc 目录（仅 md）；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 设计文档（`.cannbot/<算子名>/1.1-需求分析.md`、`.cannbot/<算子名>/2.2-开发方案设计.md`）。
- 【输出】算子文档，写 doc 目录；格式模板：`workflow-doc-templates/references/6.1-算子文档.md`。
- 【skills】立即加载 `workflow-doc-templates`、`repo-knowledge`、`ascendc-docs-gen`。
- 【验收标准】接口、参数、约束、示例齐全。
```

## 6.2 提交 PR

- **角色**：developer-doc

```md
- 【权限】你可写 doc 目录（仅 md）；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部代码 + 文档。
- 【输出】PR：提交 PR，回传 PR 链接。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】PR 描述完整，变更范围清晰；本地遗留问题清零，或已经用户问卷确认接受。
```

## 6.3 CI 流水线

- **角色**：PM（CI 为外部异步流水线，不调度子 Agent）

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】PR 链接（6.2 回传）。
- 【输出】`state.json` 落盘 `pending_ci`（`waiting`）；告知用户 CI 出结果后回传再继续，随后结束本会话。用户回传后置 `reported`，按结果进 6.4/6.5 或 CP6。
- 【skills】立即加载 `gitcode-toolkit`。
- 【验收标准】`pending_ci` 已落盘且用户已被告知；禁止本地轮询死等线上结果。
```

## 6.4 codecheck 修复

- **角色**：developer

```md
- 【权限】你可写代码、test、doc 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】用户导出的 codecheck 报告（路径以下发时给定为准）。
- 【输出】修复后代码（修复可能同时触及代码、测试与文档）。
- 【skills】立即加载 `repo-coding-rules`、`repo-build-guide`。
- 【验收标准】codecheck 问题清零。
```

## 6.5 检视意见修复

- **角色**：developer

```md
- 【权限】你可写代码、test、doc 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】PR 检视意见。
- 【输出】修复后代码：逐条闭环修复检视意见。
- 【skills】立即加载 `repo-coding-rules`、`repo-build-guide`。
- 【验收标准】检视意见逐条闭环。
```

## CP6 CI 通过确认

- **角色**：QA

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】CI 报告 + PR 状态。
- 【输出】CI 通过确认结果（未通过，回退 6.4/6.5 后重触发 CI / 通过）
- 【skills】立即加载 `workflow-cp6`、`workflow-doc-templates`。
```

# 阶段 7：开发总结

## 7.1 开发报告

- **角色**：developer-doc（起草）→ PM（落盘 `.cannbot/<算子名>/`）

```md
- 【权限】你可写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部交付物。
- 【输出】开发报告全文（格式模板：`workflow-doc-templates/references/7.1-开发报告.md`），回传调用方落盘，不自行写入。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】开发过程、交付物清单完整。
```

## 7.2 经验总结

- **角色**：developer-doc（起草）→ PM（落盘 `.cannbot/<算子名>/`）

```md
- 【权限】你可写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】开发过程记录。
- 【输出】经验总结文档全文（格式模板：`workflow-doc-templates/references/7.2-经验总结.md`），回传调用方落盘，不自行写入。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】沉淀有效经验与踩坑记录，含对工作流/领域知识的改进建议。
```

## 任务恢复 prompt

中断时按 `.cannbot/<算子名>/state.json` 的 `current_stage` 恢复到对应子 Agent（详见 [state-schema.md](state-schema.md)）。恢复时读取已产出交付件继续，不重跑已通过阶段。

| 中断阶段 | 恢复角色 | 恢复说明 |
|----------|----------|----------|
| 0 / CP0 | developer / QA | 读环境信息文档继续；CP0 问卷未收口（`pending_questionnaire`）则由 QA 重发问卷收集结论 |
| 1.1 | architect | 读需求文档继续 |
| CP1 | QA | 读需求文档继续；问卷未收口（`pending_questionnaire`）则由 QA 重发问卷收集结论 |
| CP2.1 | QA | 读测试方案文档继续 |
| 2.1 / 2.2 | architect | 读方案文档继续 |
| CP2.2 | QA | 按 `pending_questionnaire` 状态续跑：无该字段则重跑评审；`sent` 则由 QA 重发问卷收集结论；`answered` 则无异议进 3.1、有异议按语义归属回退 |
| 3.1 / 3.2 / 3.3 | developer-code / developer-test | 读代码与测试继续 |
| 4.1 / 4.2 | developer-test / developer | 读性能数据与性能迭代记录继续 |
| CP3 / CP4 / CP5 | QA | 读对应报告继续；回退则从 3.1 重走 |
| CP6 | QA | 读 CI 报告与 PR 状态继续 |
| 6.x | developer-doc / developer | 读 PR 状态继续；6.3 等待态按 `pending_ci` 续跑：`waiting` 则保持等待、提示用户回传 CI 结果，`reported` 则按回传报告进 6.4/6.5 或 CP6 |
| 7.x | developer-doc | 读交付物继续 |
