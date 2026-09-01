# 修改可插拔流程插件（plugin-*）

> 适用于：基类 `skills/plugin-*/`（内置插件），子仓 `agent/skills/plugin-*/`（新增/覆写插件）。
> plugin-* 是可插拔的子流程 skill：frontmatter 声明挂载点，init 注册到 `.cannbot/settings.json` 的 plugins，由 PM 在流程推进到挂载点时触发；自闭环、可单独任务触发。

## 核心概念

plugin-* 与 final / virtual 的关系：

- **编排（final）不感知具体插件**：基类流程表不列插件步骤；PM 只感知注册机制（`.cannbot/settings.json` 的 plugins），在挂载点触发启用的插件。
- **插件是编排层的 mini-workflow**：自带内部步骤表、角色调度、回退与验收，引用角色逻辑名与 `workflow-*` / `repo-*` skill；其内部验收标准可作为嵌套子 skill 携带（如 `plugin-foo/acceptance-bar`），init 会把嵌套子 skill 顶层链接，子仓可按同名 override。
- **可覆写、可新增**：子仓 `agent/skills/plugin-*/` 同名替换基类插件、基类没有的新增；子仓 `agent/AGENTS.md` 存在时链接子仓的 PM 入口（须保留基类 skills 登记基线，见下）。

## frontmatter 元数据契约

```yaml
name: plugin-<name>
description: <触发说明：挂载点触发 + 单独任务触发>
workflow-hook: after:6.1
workflow-stages:
    - plugin-<name>-1
standalone: true
disable-model-invocation: true
```

- `workflow-hook`：必填，`after|before:<基类流程表步骤编号>`，单挂载点；挂载点必须存在于基类统一流程表，init 校验不合法则不注册（warn）。
- `workflow-stages`：必填，插件内部步骤编号（插件格式，如 `plugin-<name>-1`），与 SKILL.md 内部步骤表一致；写入 state.json / 恢复路由用。
- `standalone`：可单独任务触发（自闭环）。
- **frontmatter 内不得带行尾注释**（`# ...`），init 的 frontmatter 解析不剥离行尾注释，带注释的字段值会导致校验失败不注册。

## 插件结构

```
skills/plugin-<name>/
├── SKILL.md          # 路由：输入/输出契约 + 内部步骤表（编号/角色/输入/输出/说明/备注）+ 通用约定
├── references/       # 内部 task-prompts、回退与轮次、恢复规则
└── <子 skill>/       # 可选：插件自带的验收标准等（顶层链接，可被子仓同名 override）
```

## 编写要求

1. **自闭环**：明确输入件/输出件路径、内部通过指标、内部回退（循环有界）、内部验收（执行方与验收方不同实例）。
2. **状态并入主 state.json**：内部步骤编号写入 `completed_stages`；插件特有等待态字段在 state-schema 注明归属。
3. **F6**：插件 SKILL.md 与 references 是运行时文档，不写 override / virtual / 基类 / 逻辑名等机制词汇；引用 skill 一律写「按 `xxx` skill 的指导」。
4. **任务清单**：SKILL.md 通用约定声明「触发后内部步骤全部并入 PM 的 todolist」。
5. **frontmatter 登记**：基类内置插件登记进 `AGENTS.md` 的 `skills:`（init 收集 + PM 可加载）；子仓插件登记进子仓 `agent/AGENTS.md`。
6. **重跑 init**：新增/删除/改名插件后必须重跑 init 使链接与 `.cannbot/settings.json` 的 plugins 生效（L1）。

## 子仓新增/覆写插件

1. 在 `<repo>/agent/skills/plugin-<name>/` 建实现；同名替换基类插件（含其嵌套子 skill），基类没有的新增。
2. 子仓 `agent/AGENTS.md`：存在即替换基类 PM 入口；**须保留基类 `skills:` 登记基线**（否则内置 skill/插件收集不到），再追加子仓插件。
3. 重跑 `<repo>/agent/init.sh`；核对 `.cannbot/settings.json` 的 plugins 已含子仓插件（override 优先）。
4. 执行 [common.md](common.md)。

## 删除插件

1. 删 `skills/plugin-<name>/`，从 `AGENTS.md` 的 `skills:` 移除；重跑 init（settings.json 的 plugins 自动剔除失效条目）。
2. 存量工作区若 `state.json` 存在该插件的在途编号：保持等待 / 提示用户启用插件 / 交用户裁定，在变更说明中写明。
