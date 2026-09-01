# Task 调用契约

> 本插件各步子 Agent 的调用参数、输入/输出、验收标准。编号与 [SKILL.md 内部步骤表](../SKILL.md#内部步骤表) 一一对应。
>
> **调用原则**：PM 每步首次调度子 Agent 时，必须严格按照本文档指定的角色和 prompt **原样调用**，仅允许替换 prompt 中的 `<算子名>` 项，**严禁干涉实现细节**。
> **工具差异（dsh / codex）**：无权限 hook 环境下【权限】行的「会被 hooks 拦截」为提示性约束（子 Agent 依 `workflow-agent-permissions` 自律，无机制拦截）；dsh 派发方式见主工作流 task-prompts 头部说明。


## plugin-pr-submit-1 提交 PR

- **角色**：developer-doc

```md
- 【权限】你可写 doc 目录（仅 md）；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】全部代码 + 文档。
- 【输出】PR：提交 PR，回传 PR 链接。
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】PR 描述完整，变更范围清晰；本地遗留问题清零，或已经用户问卷确认接受。
```

## plugin-pr-submit-2 CI 流水线

- **角色**：PM（CI 为外部异步流水线，不调度子 Agent）

```md
- 【权限】你可写 `.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】PR 链接（plugin-pr-submit-1 回传）。
- 【输出】`state.json` 落盘 `pending_ci`（`waiting`）；告知用户 CI 出结果后回传再继续，随后结束本会话。用户回传后置 `reported`，按结果判定：未全绿或检视意见未闭环则进对应修复步骤，满足完成条件则交付。
- 【skills】立即加载 `gitcode-toolkit`。
- 【验收标准】`pending_ci` 已落盘且用户已被告知；禁止本地轮询死等线上结果。
```

## plugin-pr-submit-3 codecheck 修复

- **角色**：developer

```md
- 【权限】你可写代码、test、doc 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】用户导出的 codecheck 报告（路径以下发时给定为准）。
- 【输出】修复后代码（修复可能同时触及代码、测试与文档）。
- 【skills】立即加载 `repo-coding-rules`、`repo-build-guide`。
- 【验收标准】codecheck 问题清零。
```

## plugin-pr-submit-4 检视意见修复

- **角色**：developer

```md
- 【权限】你可写代码、test、doc 目录；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】PR 检视意见。
- 【输出】修复后代码：逐条闭环修复检视意见。
- 【skills】立即加载 `repo-coding-rules`、`repo-build-guide`。
- 【验收标准】检视意见逐条闭环。
```
