---
skill_name: ascendc-tiling-design
eval_mode: text
---
# Case 1: Tiling 设计指南的功能与覆盖范围

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-tiling-design 技能的功能和使用场景，它覆盖哪些算子类别的 Tiling 设计？

## Expected Output

回复应说明 ascendc-tiling-design 技能的功能、使用场景以及覆盖的算子分类体系。

## Expectations

---

# Case 2: 通用 Tiling 设计要素

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

所有算子类型的 Tiling 设计中必须考虑的通用要素有哪些？请分别说明每个要素的核心问题。不需要执行任何工具调用。

## Expected Output

回复应说明 Tiling 设计的四个通用核心要素：
- 多核切分策略：任务如何分配给多个 AI Core，关注负载均衡、数据局部性、粒度适中
- UB 切分策略：单次能处理多少数据，受 UB 容量限制（DAV_2201: 192KB, DAV_3510: 248KB），决定是否需要分 chunk
- Buffer 规划：需要哪些 buffer（inQueue/outQueue/tmpBuf/workBuf）及各多大，支持 Double Buffer 优化
- 分支场景覆盖：需处理不同数据类型（FP32/FP16/BF16/INT8）、Shape 大小、数据对齐、边界情况等分支

## Expectations
