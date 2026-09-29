---
name: catlass-cpp-test
description: CATLASS C++ Linear Attention 五阶段工作流的 05 算子测试。适用于需要执行全量精度、功能、异常和性能验收，以及失败恢复和性能候选迭代的算子测试场景。
---

# 05 算子测试

进入条件：workflow 状态为 `validation`、`validation_scope=full`。
读取 `workflow/precision-policy.md` 和
`workflow/development-and-validation.md`。

所有阶段复用冻结的 `reference/precision-policy.json` 和本 Skill 唯一
`scripts/compare_precision.py`。全量验收覆盖功能、边界、异常、TilingKey/模板/运行分支及受影响
算子；性能使用统一模型用例和 `msprof op_summary` 的 `Task Duration(us)`。

单用例连续运行 60 秒无返回视为 kernel 超时。清理进程和设备资源后，将现象、Stage、
TilingKey 和 `blockDim` 写入 `docs/validation.md`；该用例不得计为通过，按失败证据进入对应
恢复分支。

失败按 workflow validator 的恢复矩阵一次更新全部状态字段。性能不达标保持
`stage=validation`、`issue_type=performance_optimize`；每轮只改变一个主要变量，候选先过精度
再同条件比较。修改设计假设返回 03，只改实现返回 04。最终候选必须重新执行 fresh 全量验收。

只有 full 精度、功能和要求的性能验收实际通过后，才能设置 `stage=complete`，清空问题字段并
整理验收 README。未达到目标时如实保留未达标状态，不降低门禁。
