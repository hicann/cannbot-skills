---
skill_name: ascendc-simt-best-practices
eval_mode: text
---
# Case 1: SIMT API 分类概述

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

Ascend C SIMT 算子开发中有哪些可用的 API 类别？请列出主要分类及其用途。不需要执行任何工具调用。

## Expected Output

回复应介绍 Ascend C SIMT 算子开发中可用的 API 类别，按功能分类说明各类 API 的用途。

## Expectations

---

# Case 2: SIMT vs SIMD 差异与选型

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

Ascend C 中 SIMT 和 SIMD 两种编程模型有什么区别？分别在什么场景下使用？不需要执行任何工具调用。

## Expected Output

回复应对比 SIMT 和 SIMD 两种编程模型的差异。SIMD 是向量编程模型，适合计算密集、数据访问规律的场景；SIMT 是线程级编程模型，适合索引复杂有分支条件逻辑的场景。

## Expectations
