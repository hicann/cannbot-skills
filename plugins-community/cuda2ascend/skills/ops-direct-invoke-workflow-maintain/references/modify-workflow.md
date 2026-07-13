# 修改工作流编排 / PM 入口

> 适用于：`skills/ops-direct-invoke-workflow/`（承载整个流程编排）与 `AGENTS.md`（PM 入口）。
> 工作流编排属 **final**，是所有接入仓共享的稳定契约，改动影响面最大，务必谨慎。

## 架构前提

工作流编排**以 skill 承载**：流程逻辑在 `ops-direct-invoke-workflow` skill 里，AGENTS.md 只做入口。改流程 = 改这个 skill，不是改 AGENTS.md。

## 修改前必须想清楚

1. **能不能不改基类？** 若诉求是「某仓要不一样」，答案通常是子仓 override virtual skill（`repo-*` 领域知识 / `workflow-cp*` 验收标准），而**不是**改工作流编排（开闭 O、S2）。只有「所有仓都该变」的通用编排才改。
2. **会不会破坏契约？** 基类对外暴露的 virtual 逻辑名、各角色输入输出格式、调用约定一经发布即稳定（S6）。改动只能新增，不能破坏已接入仓的 override。

## 修改工作流编排 skill（阶段 / CP / 回退关系）

1. 流程内容在 `skills/ops-direct-invoke-workflow/` 下。遵守 F1：SKILL.md 只做路由与总览，细节分层到 references。
2. 新增或调整阶段时，确保每个环节满足 D5（明确交付件 + 可判定通过指标）、D4（循环环节声明最大次数）、D6/D7（状态持久化与可恢复）。
3. 任何 CP 的输入输出格式变更都是契约变更——按 [common.md](common.md) 契约专项处理。
4. 编排只通过逻辑名引用 virtual 组件（D5 依赖倒置），不写死具体实现。
5. 该 skill 属 final，子仓不 override；不要在其中写入与特定仓耦合的领域知识（那属于 `repo-*`）。
6. **同步 README 概览表**：`SKILL.md` 的统一流程表是权威真值源；插件根 `README.md` 的「开发流程概览」只是它的派生概览。凡增删/调整阶段、CP 点或回退关系，必须回头核对 README 概览表是否仍与 SKILL.md 一致（阶段划分、CP 编号、回退指向），不一致则同步更新 README，避免两处漂移。

## 修改 PM（AGENTS.md）

1. PM 只调度不执行——修改时保持这一核心边界：实质开发动作必须派发给执行角色，PM 只在 `.cannbot` 写流程中间文件。
2. AGENTS.md 只做入口，引导加载 `ops-direct-invoke-workflow` skill；**不**在 AGENTS.md 里写具体流程。
3. 若新增了需要 PM 直接加载的 skill，在 AGENTS.md 的 `skills:` frontmatter 登记。
4. 遵守 D2（最小信息）：PM 派发子 Agent 时只传当前任务所需输入。

## 修改编排流程（阶段 / CP / 回退关系）

1. 新增或调整阶段时，确保每个环节满足 D5（明确交付件 + 可判定通过指标）、D4（循环环节声明最大次数）、D6/D7（状态持久化与可恢复）。
2. 任何 CP 的输入输出格式变更都是契约变更——按 [common.md](common.md) 契约专项处理。
3. 编排只通过逻辑名引用 virtual 组件（D5 依赖倒置），不写死具体实现。

## 收尾

自行运行基类 init 使配置生效，执行 [common.md](common.md)，并对照 [review-checklist.md](review-checklist.md) 逐条自查（尤其 S 系列与 D 系列）。
