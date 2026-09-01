---
name: plugin-pr-submit
description: 提交 PR 到上库确认的可插拔流程插件。触发：工作流推进到挂载点（文档补全完成后）时由 PM 触发；或用户单独要求提交 PR、处理 codecheck / CI / 检视意见时。
workflow-hook: after:6.1
workflow-stages:
    - plugin-pr-submit-1
    - plugin-pr-submit-2
    - plugin-pr-submit-3
    - plugin-pr-submit-4
standalone: true
disable-model-invocation: true
---

# 提交 PR 插件

本插件承载**从提交 PR 到上库确认**的完整流程，自闭环执行：提交前遗留清零 → 提交 PR → CI 等待 → codecheck / 检视意见修复 → 完成条件满足后交付。插件步骤使用插件内部编号（`plugin-pr-submit-1` 起），进度并入 `.cannbot/<算子名>/state.json`。

## 输入 / 输出

- **输入**：全部代码 + 文档（工作流场景下为代码检视通过后的交付状态；单独触发时以当前工作区状态为准）。
- **输出**：PR + 上库交付结论。

## 内部步骤表

| 编号 | 流程 | 角色 | 输入 | 输出 | 说明 | 备注 |
|------|------|------|------|------|------|------|
| plugin-pr-submit-1 | 提交 PR | developer-doc | 全部代码 + 文档 | PR | 提交 PR | 提交前 PM 汇总 state.json / Issue-问题记录 / 各子 Agent 回传摘要做遗留问题清零检查；有遗留（不论优先级多低）须发问卷经用户确认接受后方可提交，确认结论记入 state.json |
| plugin-pr-submit-2 | CI 流水线 | CI（外部） | PR | CI 报告 | 触发 CI：编译 + UT + ST | 非调度子 Agent；外部异步流水线——提交 PR 后落盘 `pending_ci` 等待态并结束会话，告知用户 CI 出结果后回传再继续，禁止本地轮询死等 |
| plugin-pr-submit-3 | codecheck 修复 | developer | 用户导出的 codecheck 报告 | 修复后代码 | 修复 codecheck 问题 | 报告由用户从线上导出并传入工作区给出路径，禁止凭空猜测问题清单 |
| plugin-pr-submit-4 | 检视意见修复 | developer | PR 检视意见 | 修复后代码 | 修复 PR 检视意见 | |

## 通用约定

- **完成条件**（上库交付判定）：CI 全绿——模式 A（评测集）三阶段（编译 / 精度 / 性能）全过且 HAP 达标；模式 B（自建测试）`run.sh` 编译 + 精度全过；无失败、无阻塞项——且 PR 检视意见全部闭环（A 类反作弊红线零容忍），验证报告对应最新代码。由 PM 对照 CI 报告与检视意见状态裁定；未满足时回退对应修复步骤（plugin-pr-submit-3 / plugin-pr-submit-4）后重触发 CI。
- **任务清单**：插件触发后，将本步骤表全部步骤并入 PM 的 todolist，随流程推进逐项刷新状态。
- **任务下发**：按 [references/task-prompts.md](references/task-prompts.md) 中的角色、prompt 原样调用，仅替换 `<算子名>`。
- **状态与回退**：每步完成后将进度并入 `.cannbot/<算子名>/state.json`（调度下一步前先落盘）；回退策略、最大轮次、恢复规则见 [references/error-handling.md](references/error-handling.md)。结束会话是 PM 的动作，恢复由 `pending_ci` 驱动。
