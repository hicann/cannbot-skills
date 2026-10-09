---
skill_name: hccl-aicpu-debug
---

# AICPU 调试能力评测

`evals.json` 的两项文本用例检验算法命中证据和故障分类边界。评测运行器加载本 Skill 后给出 prompt，按 `expected_output` 人工核对；这些用例不运行 HCCL 构建或 Checker，不代表真实故障已定位。
