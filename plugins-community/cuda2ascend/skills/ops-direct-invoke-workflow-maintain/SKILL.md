---
name: ops-direct-invoke-workflow-maintain
description: 工作流维护技能。任何对工作流文件的新增、修改、删除，都必须先触发本技能再执行，包括修改本技能自身。禁止未触发直接修改。触发：任何对工作流文件（基类的 AGENTS.md / agents / skills / init.sh，或算子仓 agent/ 下的 override 实现）的新增、修改、删除
---

# cuda2ascend 工作流维护指南

本 skill 是工作流的**维护 skill**，与工作流本身处于同一层级，允许感知并描述所有模块（基类编排、agents、skills、init）——不受「下级不感知上级」的层级约束。任何对工作流文件的新增、修改、删除，都先触发本 skill 再执行。

## 首次维护必读

第一次维护本工作流时，**先通读 [references/architecture-and-links.md](references/architecture-and-links.md)**，理解当前这套工作流的链接结构：源文件分布在基类仓与算子仓两处、init 如何通过软链接把它们汇合到运行时目录、skill/agent 的 override 如何按逻辑名绑定。不理解链接来源就动手，容易改错层或破坏契约。

## 设计约束（维护红线）

工作流的一切修改都受设计约束约束。原件见 [references/design-constraints.md](references/design-constraints.md)（勿改）。由其派生的**工作流检视条款**见 [references/review-checklist.md](references/review-checklist.md)——每次修改后按条款自查。

约束中与维护最相关的三条，务必牢记：

- **开闭 + 里氏替换**：子仓要扩展某步骤，必须 override 该步骤用到的 virtual skill（`repo-*` 仓库领域知识 / `workflow-*` 工作流定义的模板与验收标准），**不得改动基类编排与子 Agent**。override 后必须保持相同逻辑名、输入输出格式、调用约定。
- **契约向后兼容**：基类对外暴露的契约——virtual 组件的**逻辑名**、各角色的**输入输出格式**、**调用约定**——一经发布即稳定，演进时只增不破坏。改基类前先问：这会不会让已接入仓的 override 失效？
- **`example/init.sh` 分发契约**：`example/init.sh` 已分发到各子仓，基类 `init.sh` 的 CLI 契约一旦发生不兼容变更，必须**发问卷知会用户**（受影响子仓 + 新旧用法对比），由用户决策是否迁移——详见 [references/modify-init.md](references/modify-init.md) 第 5 条与 review-checklist C1–C4。
- **机制优于自然语言**：能用 hook / 脚本 / 权限声明约束的行为，不要写成 prompt 里的自然语言。


## 修改操作指南

按修改对象加载对应参考文档：

| 修改对象 | 参考文档 | 涉及文件 |
|---------|---------|---------|
| 工作流编排 / PM 入口 | [references/modify-workflow.md](references/modify-workflow.md) | `skills/ops-direct-invoke-workflow/`、`AGENTS.md` |
| virtual skill（`repo-*` 领域知识 / `workflow-*` 模板与验收标准，基类默认或子仓 override） | [references/modify-skill.md](references/modify-skill.md) | `skills/repo-*/`、`skills/workflow-*/`、子仓 `agent/skills/*` |
| 三类固定角色（architect / developer / qa） | [references/modify-agent.md](references/modify-agent.md) | `agents/*.md` |
| init 脚本 | [references/modify-init.md](references/modify-init.md) | 基类 `init.sh`、子仓 `agent/init.sh` |

**所有修改完成后**，必须执行 [references/review-checklist.md](references/review-checklist.md) 的检视条款自查，并按 [references/common.md](references/common.md) 做通用收尾。
