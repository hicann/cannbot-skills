# Task 调用契约

> 本插件各步子 Agent 的调用参数、输入/输出、验收标准。编号与 [SKILL.md 内部步骤表](../SKILL.md#内部步骤表) 一一对应。
>
> **调用原则**：PM 每步首次调度子 Agent 时，必须严格按照本文档指定的角色和 prompt **原样调用**，仅允许替换 prompt 中的 `<算子名>` 项，**严禁干涉实现细节**。

## plugin-perf-iteration-1 性能迭代

- **角色**：developer

```md
- 【权限】你可写算子代码目录、`.cannbot/<算子名>/`；临时产物写 `.cannbot/<算子名>/tmp/`。禁止写项目根之外的路径（含 `/tmp`），会被 hooks 拦截。
- 【输入】算子代码 + 性能数据（基线）；共享依赖仓 `.cannbot/cann-samples/`。
- 【输出】优化后算子代码 + 性能迭代记录（逐次优化路径与优化效果），写入 `.cannbot/<算子名>/性能迭代记录.md`；全量用例回归验证结果写入 `.cannbot/<算子名>/回归验证结果.md`。
- 【skills】立即加载 `ascendc-perf-optimize`、`repo-coding-rules`、`repo-build-guide`。
- 【架构演进边界】一般不在一开始就用 SIMT 实现——初始实现遵循需求文档已拍板的架构（通常为 SIMD）；仅针对离散的场景（特定 shape/dtype 组合等局部热点），可将部分实现改为 SIMT 实现；采用 SIMT 的场景范围、依据与优化效果必须逐条记入性能迭代记录，供门禁复核。
- 【验收标准】适用的优化路径均已尝试并逐次记录效果；全量用例回归验证通过。
```

## plugin-perf-iteration-2 回归门禁复核

- **角色**：QA

```md
- 【权限】你只可写 `.cannbot/<算子名>/tmp/`，其它写入操作会被 hooks 拦截。
- 【输入】优化后算子代码 + 性能迭代记录 `.cannbot/<算子名>/性能迭代记录.md` + 全量用例回归验证结果 `.cannbot/<算子名>/回归验证结果.md`。
- 【输出】回归门禁结论（不通过，回退 plugin-perf-iteration-1 / 通过）
- 【skills】立即加载 `workflow-doc-templates`。
- 【验收标准】性能迭代记录逐次齐备（路径 + 效果）；若存在 SIMT 例外条目，其场景范围、依据与优化效果须逐条齐备；全量用例回归结果全过（无失败、无缺失用例）。
```
