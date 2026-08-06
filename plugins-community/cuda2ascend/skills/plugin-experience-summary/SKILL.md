---
name: plugin-experience-summary
description: 算子开发经验总结的可插拔流程插件。触发：工作流推进到挂载点（开发总结阶段开始前）时由 PM 触发；或用户单独要求总结本次开发经验时。
workflow-hook: before:7.1
workflow-stages:
    - plugin-experience-summary-1
    - plugin-experience-summary-2
    - plugin-experience-summary-3
    - plugin-experience-summary-4
standalone: true
disable-model-invocation: true
---

# 经验总结插件

本插件承载**本次开发的经验总结**完整流程，自闭环执行：回顾本次开发并总结本仓开发容易踩坑、而本仓 skill 未说明确的点；梳理本次会话中 hooks 的异常拦截并给出建议修复点；统计本次工作流运行总时长并定位反复重试耗时的地方；汇总产出经验总结文档，最后征询用户是否需要拟一条 issue 反馈到 cannbot-skills 仓。插件步骤使用插件内部编号（`plugin-experience-summary-1` 起），进度并入 `.cannbot/<算子名>/state.json`。

## 输入 / 输出

- **输入**：本次开发全量过程产物——`.cannbot/<算子名>/`（state.json、LOG.md、Issue-问题记录、各阶段交付件、questionnaires/、tmp/）、`.cannbot/环境信息.md`（含「环境补充记录」节）、插件执行期间产生的中间记录。
- **输出**：`.cannbot/<算子名>/插件经验总结.md`（四节：踩坑回顾 / hooks 分析与修复建议 / 耗时统计与重试定位 / 汇总与 issue 征询结论，含环境检查建议）；用户同意后按 `gitcode-issue-gen` 拟 issue 草稿交用户确认。

## 内部步骤表

| 编号 | 流程 | 角色 | 输入 | 输出 | 说明 | 备注 |
|------|------|------|------|------|------|------|
| plugin-experience-summary-1 | 开发回顾与踩坑总结 | developer-doc | 全部过程产物 | 踩坑回顾段落（暂存） | 回顾本次开发全过程；总结本仓开发容易踩坑、而本仓 skill 未说明确的点（对照各 skill 文档定位盲区） | **与 2/3 并行派发**；产出先落 `.cannbot/<算子名>/tmp/经验总结-1-踩坑回顾.md` |
| plugin-experience-summary-2 | hooks 异常拦截分析 | developer-doc | 拦截记录（state/LOG/tmp）+ 两侧 hook 源 | hooks 分析与修复建议段落（暂存） | 梳理本次会话中 permission-guard 与静默问卷拦截的实际拦截/漏拦/误拦记录，对照 `hooks/opencode/` 与 `hooks/claude/` 源给出建议修复点 | **与 1/3 并行派发**；产出落 `.cannbot/<算子名>/tmp/经验总结-2-hooks分析.md` |
| plugin-experience-summary-3 | 耗时统计与重试定位 | developer-doc | state.json 时间戳、LOG.md、回退记录 | 耗时统计段落（暂存） | 统计本次工作流运行总时长（首步启动至收尾，按阶段/CP 分解），总结反复重试耗时的地方（回退轮次、CI 等待、性能迭代等） | **与 1/2 并行派发**；产出落 `.cannbot/<算子名>/tmp/经验总结-3-耗时统计.md` |
| plugin-experience-summary-4 | 汇总与 issue 征询 | PM | 前三段产物 + `.cannbot/环境信息.md` | `.cannbot/<算子名>/插件经验总结.md` + 征询结论 | 整合三段为完整经验总结并落盘；**检查环境信息文档的「环境补充记录」节——非空（曾被探索补充）时在总结中建议提 Issue 补充环境检查内容**；征询用户是否需要拟 issue 反馈到 cannbot-skills 仓——同意则加载 `gitcode-issue-gen` 拟 issue 草稿交用户确认后提交 | 三段缺失或不全时回退对应步骤补齐，往返不超过 2 轮 |

## 通用约定

- **任务清单**：插件触发后，将本步骤表全部步骤并入 PM 的 todolist，随流程推进逐项刷新状态。
- **并行派发**：plugin-experience-summary-1/2/3 互无依赖，PM 一次性并行派发三个子任务（三个 developer-doc 实例，各自只写自己的 tmp 产出）；任一步失败不影响其它步骤，三段全部落盘后进入步骤 4。
- **任务下发**：按 [references/task-prompts.md](references/task-prompts.md) 中的角色、prompt 原样调用，仅替换 `<算子名>`。
- **状态与回退**：每步完成后将进度并入 `.cannbot/<算子名>/state.json`（调度下一步前先落盘）；plugin-experience-summary-4 汇总时发现前三段缺失/不全，回退对应步骤补齐，补齐往返不超过 2 轮，超限暂停并输出当前问题清单请求用户决策。
- **静默模式**：步骤 1–3 按静默行为执行（不输出、不询问）；步骤 4 的 issue 征询为插件必要交互——静默下不单独弹问卷，随任务完成总结一并列出（含拟 issue 的建议选项），由用户在下一次交互中决定。
- **中断恢复**：读取 state.json 对应编号续跑——plugin-experience-summary-1/2/3 中断时以既有 tmp 段落继续或重做（并行派发任一步中断只重跑该步）；plugin-experience-summary-4 中断时以既有总结文件与征询状态续跑。
