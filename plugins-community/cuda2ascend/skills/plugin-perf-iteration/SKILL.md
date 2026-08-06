---
name: plugin-perf-iteration
description: 算子性能迭代优化的可插拔流程插件。触发：工作流推进到挂载点（性能采集执行完成后）时由 PM 触发；或用户单独要求对算子做性能迭代优化时。
workflow-hook: after:4.1
workflow-stages:
    - plugin-perf-iteration-1
    - plugin-perf-iteration-2
standalone: true
disable-model-invocation: true
---

# 性能迭代插件

本插件承载**算子性能迭代优化**的完整流程，自闭环执行：阅读 cann-samples 寻找适用于本算子的性能优化最佳实践，尝试所有适用的优化路径并逐次记录优化效果；全量用例回归验证经 QA 复核通过后交付。插件步骤使用插件内部编号（`plugin-perf-iteration-1` 起），进度并入 `.cannbot/<算子名>/state.json`。

## 输入 / 输出

- **输入**：算子代码 + 基线性能数据（工作流场景下为功能验收通过后的最终代码与性能采集结果；单独触发时以当前算子代码与既有性能数据为准）；共享依赖仓 `.cannbot/cann-samples/`。
- **输出**：优化后算子代码 + 性能迭代记录（`.cannbot/<算子名>/性能迭代记录.md`）+ 最终性能数据。

## 内部步骤表

| 编号 | 流程 | 角色 | 输入 | 输出 | 说明 | 备注 |
|------|------|------|------|------|------|------|
| plugin-perf-iteration-1 | 性能迭代 | developer | 算子代码 + 基线性能数据 | 优化后算子代码 + 性能迭代记录 | 阅读 cann-samples 寻找适用于本算子的性能优化最佳实践，尝试所有适用的优化路径，逐次记录优化效果，路径穷尽即收尾 | 全量用例回归验证通过后交门禁复核 |
| plugin-perf-iteration-2 | 回归门禁复核 | QA | 优化后算子代码 + 性能迭代记录 | 门禁结论 | QA 复核：性能迭代记录逐次齐备（路径 + 效果），全量用例回归结果全过（无失败、无缺失用例） | 全量回归通过方可交付；不通过回退 plugin-perf-iteration-1 |

## 通用约定

- **任务清单**：插件触发后，将本步骤表全部步骤并入 PM 的 todolist，随流程推进逐项刷新状态。
- **任务下发**：按 [references/task-prompts.md](references/task-prompts.md) 中的角色、prompt 原样调用，仅替换 `<算子名>`。
- **状态与回退**：每步完成后将进度并入 `.cannbot/<算子名>/state.json`（调度下一步前先落盘）；迭代执行方（developer）与门禁复核方（QA）为不同实例；plugin-perf-iteration-2 不通过回退 plugin-perf-iteration-1，迭代-复核往返不超过 3 轮，超限暂停并输出当前问题清单请求用户决策。
- **中断恢复**：读取 state.json 对应编号续跑，不重跑已通过步骤——plugin-perf-iteration-1 中断读 `.cannbot/<算子名>/性能迭代记录.md` 继续，plugin-perf-iteration-2 中断复核既有回归结果（`.cannbot/<算子名>/回归验证结果.md`）与迭代记录。
