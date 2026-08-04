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

派发非 `.cannbot` 目录的写任务时，必须选择对应可写范围的角色；权限不足会被 hook 拦截并提示上报主 Agent。

## 规格文件

`hooks/` 下每角色一个 `.js` 文件，文件名即角色名（如 `PM.js`、`developer-code.js`），ESM `export default { categories, exts }` 形式：

- `categories`：可写目录类别集合（取值 `intermediate` / `test` / `doc` / `code`，具体定义见 hook 内置）
- `exts`：可写文件扩展名（`"*"` 或后缀数组如 `[".md"]`）

init.sh 将 `hooks/` 下的角色文件整体复制到 `.cannbot/permissions/`（缺失才生成、已存在保留）。

## 启动校验

PM 每次会话开始会检查 `.cannbot/permissions/` 是否齐全；缺失或角色文件不足会拒绝执行任务并要求退出当前 CLI 会话重跑 init.sh（详见 `AGENTS.md` 启动检查段）。
