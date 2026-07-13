---
skill_name: ascendc-perf-optimize
eval_mode: text
---
# Case 1: 性能优化四步流程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-perf-optimize 技能中 Ascend C 算子的性能优化工作流程是什么？需要经过哪些步骤？每一步分别关注什么？不需要执行任何工具调用。

## Expected Output

回复应描述性能优化的主要流程：Tiling 建模、流水分析、优化策略制定和 Tiling 回修四个步骤。

---

# Case 2: 性能优化所需输入

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

进行 Ascend C 算子性能优化需要提供哪些输入信息？需要准备哪些数据和材料？不需要执行任何工具调用。

## Expected Output

回复应说明性能优化所需的输入，包括算子代码、Tiling 参数与建模数据、流水仿真图、Profiling 数据、硬件配置信息等。这些输入是进行性能瓶颈分析和优化方案设计的基础。

---

# Case 4: 算子崩溃排查咨询（负向看护）

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

我的 Ascend C 算子在 NPU 上运行时出现了 Segmentation Fault，程序直接崩溃了，请问该怎么排查？

## Expected Output

回复应聚焦于 NPU 算子崩溃的通用排查思路，包括收集崩溃现场信息、分析 Segmentation Fault 的常见原因（越界访问、空指针等）和定位方法，不涉及性能优化相关内容。

