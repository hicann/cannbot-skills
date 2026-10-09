---
skill_name: hccl-aicpu-best-practice
---

# AICPU 实现实践能力评测

`evals.json` 的 18 项文本用例检查实现范围、跨层契约、构建前基线、双轨验收及显式/默认选路区分。评测共享契约场景时须同时加载 [共享跨层契约](../aicpu-shared-contract.md) 和相应参考。专项 template 检视脚本另以 `review/scripts/test_review.py` 做离线回归；文本问答及脚本通过均不能替代真实构建和 Checker。

原共享入口的评测说明与历史验证记录保留在 [engine-contract.md](engine-contract.md)。
