---
name: workflow-agent-permissions
description: 派发写类任务前加载，判断目标角色是否具备对应目录写权限，避免无效派发。
disable-model-invocation: true
---

# 各 Agent 写权限范围

| 角色 | 可写目录 | 文件类型 | 职责提示 |
|------|---------|---------|---------|
| PM | 仅 `.cannbot` | 任意 | 只调度；实质产物派发给子 Agent |
| architect | 仅 `.cannbot` | 任意 | 设计交付件（md） |
| qa | 仅 `.cannbot` | 任意 | 验收报告、问卷 json |
| developer | 代码 + test + doc | 任意 | 跨域综合修复 |
| developer-code | 代码（除 test） | 任意 | 算子实现（kernel + plugin + 工程骨架） |
| developer-test | test | 任意 | 测试工程（评测集 cases 对齐、白盒补充、本地自测脚本） |
| developer-doc | 代码 + test + doc | 仅 `.md` | 文档产出 |

派发非 `.cannbot` 目录的写任务时，必须选择对应可写范围的角色；权限不足会被 hook 拦截并提示上报主 Agent（**dsh / codex 无项目级权限 hook**：默认拦截缺失，规则降级为 prompt 约束——PM 依本 skill 判定避免无效派发，子 Agent 依 prompt 自律，违规写入无机制拦截。**dsh 可选机制升级**：运行 `hooks/dsh/install.sh` 安装部署级 Cordis 守卫插件（`$DSH_HOME/cordis.patch.yml`，对所有 profile 生效），即恢复与 opencode/claude 同语义的机制拦截，仅作用于 cuda2ascend 初始化的工作区）。

## 目录类别的判定方式

目录类别按写入路径（相对项目根）的**目录段**判定，段名需完全相等，命中即归该类：

| 类别 | 命中的目录段名 | 说明 |
|------|----------------|------|
| `intermediate` | `.cannbot` | 流程中间产物区，所有角色可写、不限文件类型 |
| `test` | `test`、`tests` | 测试目录，单复数均识别 |
| `doc` | `doc`、`docs` | 文档目录，单复数均识别 |
| `code` | —（兜底） | 未命中上述段名的其余路径 |

判定按**任意一段**命中，与深度无关：`<工程目录>/tests/<算子名>/` 与顶层 `test/` 同为 `test` 类。段名须完全相等——`test_op/`、`docs_src/` 这类不算命中，仍归 `code`。多类同时命中时按 `intermediate` → `test` → `doc` 次序取第一个。

## 守卫边界

拦截发生在**写类工具**（write / edit / patch / multiedit / notebookedit 等，dsh 为 write / edit）这一层：

- 用 shell 命令写文件（重定向、`cp`、`tee` 等）**不经过守卫**。落盘一律用写类工具，不得用 shell 绕过权限范围——绕过既失去拦截，也失去留痕。
- 权限不足时正确做法是结束任务并向主 Agent 上报，由其改派具备该目录写权限的角色，不是换一种写法把文件落下去。
- 规则表未命中的角色一律拒绝写入（`.cannbot` 除外，该类别对所有角色短路放行）。

## 静默模式问卷拦截

静默模式（`.cannbot/settings.json` 的 `mode=silent`）启用时，permission-guard hook 在机制层拦截问卷工具（opencode 的 `question`/`ask`、claude 的 `AskUserQuestion`；按工具名子串匹配，覆盖带前后缀的同类命名）的发送：任何角色（含 QA）在静默下调用问卷工具都会被阻断并回传「按静默默认决策执行、落盘 `.reply.json`」的提示。这是 prompt 层约束（QA 不发送问卷）的机制兜底；settings.json 的 `mode` 切换为 `interactive` 后立即解除拦截（opencode 每次调用实时读配置，claude 天然每次调用独立进程）。**dsh**：默认无项目级 hook、无机制兜底，仅靠 prompt 约束（task-prompts 的「【静默模式】不发送问卷」追加指令 + QA 自律）；安装部署级守卫（`hooks/dsh/install.sh`）后，`tools/pre-execute` 门会拦截 `ask_user_question` 等问卷工具（按子串匹配，dsh 侧含 `ask`），恢复机制兜底。**codex**：无任何 hook，仅 prompt 约束。

## 规格文件

`hooks/` 下每角色一个 `.js` 文件，文件名即角色名（如 `PM.js`、`developer-code.js`），ESM `export default { categories, exts }` 形式：

- `categories`：可写目录类别集合（取值 `intermediate` / `test` / `doc` / `code`，判定方式见上方「目录类别的判定方式」）
- `exts`：可写文件扩展名（`"*"` 或后缀数组如 `[".md"]`）

init.sh 将 `hooks/` 下的角色文件整体复制到 `.cannbot/permissions/`（缺失才生成、已存在保留）。

## 启动校验

PM 每次会话开始会检查 `.cannbot/permissions/` 是否齐全；缺失或角色文件不足会拒绝执行任务并要求退出当前 CLI 会话重跑 init.sh（详见 `AGENTS.md` 启动检查段）。
