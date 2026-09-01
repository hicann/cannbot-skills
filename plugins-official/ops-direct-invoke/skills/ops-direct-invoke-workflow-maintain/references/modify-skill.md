# 修改 virtual skill

> 适用于：基类 `skills/repo-*/`、`skills/workflow-*/`（默认实现），子仓 `agent/skills/*`（override 实现）。
> `<level>` 按当前运行环境确定（project/global）。目标工具支持 opencode / claude / codex / dsh / trae。

## 核心概念

virtual skill 是子仓 override 的对象，工作流按逻辑名引用，基类提供默认实现。按逻辑名前缀分两类：

- **`repo-*` 仓库领域知识 skill**：仓库领域知识、代码模板、编码规范、编译指导、测试开发。
- **`workflow-*` 工作流定义 skill**：交付件模板（`workflow-doc-templates`）与各 CP 点验收标准（`workflow-cp*`：验收对象、通过指标、判定方式，由 QA 加载执行）。

skill 覆盖规则：**同名替换、基类没有的新增**（匹配依据是 skill 目录名 + SKILL.md 的 `name:`）。

## 在基类新增/修改默认 skill

1. 改 `skills/<skill-name>/SKILL.md`（遵守 F1「只做路由」：细节放 `references/`，SKILL.md 只路由）。
2. 若涉及仓库领域知识，遵守 F3/F4：优先引用仓内 doc 原文，不全文硬编码。
3. 若该 skill 需被某子 Agent 或 PM 使用，在对应 `agents/<name>.md` 或 `AGENTS.md` 的 `skills:` frontmatter 中登记（否则 init 的 skill 收集第二步取不到）。
4. **契约红线**：不得重命名已发布的 `repo-*` 逻辑名，不得改变其调用约定——否则子仓针对该逻辑名的 override 全部失效（S6）。
5. 自行运行 `bash <PLUGIN_ROOT>/init.sh <level> <tool> <install>`（仅改内容、不增删目录时可跳过）。
6. 执行 [common.md](common.md)。

## 在子仓 override skill（领域知识或验收标准）

1. 在 `<repo>/agent/skills/<logical-name>/` 建实现，**目录名必须等于基类逻辑名**（如 `repo-coding-rules` 或 `workflow-cp3`），SKILL.md 的 `name:` 也须一致（L2）。
2. 保持与基类相同的职责边界与输入输出（里氏替换 S3）：override 只换「怎么做/依据什么」，不改「做什么」。`workflow-cp*` 须保持相同的验收对象与输出格式（通过/结构化修改意见），使 QA 与编排无感替换。
3. 自行运行 `bash <repo>/agent/init.sh <level> <tool>`，它会透传 `--override` 用本仓实现替换基类默认。
4. 验证 `<tool>/skills/<logical-name>` 软链接已指向子仓实现。
5. 执行 [common.md](common.md)。

## 删除 skill

1. 基类删默认实现：删 `skills/<name>/`，并从所有引用它的 `skills:` frontmatter 中移除。删除逻辑名属破坏性变更，按 common.md 契约专项处理。
2. 子仓移除 override：删 `<repo>/agent/skills/<name>/` 后重跑子仓 init，该逻辑名会回落到基类默认实现。
