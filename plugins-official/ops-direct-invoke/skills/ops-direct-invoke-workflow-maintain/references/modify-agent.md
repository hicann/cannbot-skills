# 修改 Agent（三类固定角色）

> 适用于：基类 `agents/*.md`（architect、developer 系列、qa）。
> `<level> <tool>` 按当前运行环境确定。

## 核心概念

Agent 团队为三类固定角色，全部属 **final**，承载角色身份、行为边界、工具与写权限声明，子仓不覆写：

- **architect**：方案设计（开发方案、测试方案）。
- **developer**（大类，按操作权限分 `developer-code` / `developer-test` / `developer-doc`）：代码、测试、文档的开发与修复。
- **qa**：验收。在各 CP 点加载对应的 `workflow-cp*` Skill 完成判定，产出验收报告与用户确认问卷。

**验收标准不在 agent 层定制**：各 CP 点的验收对象、通过指标、判定方式承载在 `workflow-cp*` Skill 里，子仓通过 override 这些 Skill 插入自己的验收标准（见 [modify-skill.md](modify-skill.md)）。QA 本身是固定执行者，不随仓变化。

## 修改 agent

1. 改 `agents/<name>.md`。
2. 遵守层级感知（H1）：agent 定义中**不得**提及自己处于工作流哪一步、被谁调度。
3. 遵守权限约束（D3）：写权限声明须为完成职责所需的最小集；执行方与验收方须可分离（D1）——QA 不验收自己产出的交付件。
4. 若 agent 用到某 skill，在其 `skills:` frontmatter 登记（init 据此收集 skill）。
5. **正文与 frontmatter 一致性**（L7）：扫描正文中所有 skill 依赖声明（"使用 `xxx`"、"加载 `xxx`"、"引用 `xxx`"、"依据 `xxx`"等），核对每条是否已列入 `skills:` frontmatter。缺失即补，多余即删。
6. **契约红线**：不得重命名已发布的 agent 逻辑名（文件名），编排按此名调度（S6、L2）。
7. 自行运行基类 init 使链接生效；执行 [common.md](common.md)。

## 注意事项

- Agent 全 final，子仓不通过 override agent 定制流程。子仓要改验收标准 → override `workflow-cp*` Skill；要改领域知识 → override `repo-*` Skill。
- 确需新增/删除角色或调整职责边界，属编排改动，走 [modify-workflow.md](modify-workflow.md)。
