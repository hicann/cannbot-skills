---
skill_name: hccl-aicpu-design
---

# AICPU 设计能力评测

`evals.json` 的三项文本用例检查跨层改动范围、输入缺口、数据流规格前置条件，以及算法设计与 C++ 实现示例的边界。由独立评测运行器加载本 Skill 与必要参考后判断；不执行 HCCL 构建，也不把文本问答视为真实设计验收。
