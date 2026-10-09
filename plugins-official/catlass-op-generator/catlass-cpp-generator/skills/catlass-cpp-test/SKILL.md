---
name: catlass-cpp-test
description: CATLASS C++ 已注册算法族五阶段工作流的 05 算子测试。用于全量精度、功能、异常和性能验收，以及失败恢复和性能候选迭代。触发条件：workflow 处于 validation 阶段且 validation_scope=full。
---

# 05 算子测试

进入条件：workflow 状态为 `validation`、`validation_scope=full`。
读取 `workflow/precision-policy.md` 和
`workflow/development-and-validation.md`。

`sparse_flash_mla` 另读家族 `validation.md` 与 `performance.md`；普通 SWA
同时应用其中的动态诊断、完整输出和流水验收规则。分别验证独立 metadata、主核与成对调用。
原 ATK 报告的实际 accuracy、LSE 全量比较和最终构建身份才是正式精度证据；
诊断脚本 PASS、进程返回0或仅 wheel 安装均不等同于精度 PASS。
手写基线从用户指定历史 JSON 读取，不重新执行；候选按任务规定的
5 次预热和 5 次正式样本取完整 device 主核均值，逐 case 比较。

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
