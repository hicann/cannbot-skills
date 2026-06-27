---
team_name: catlass-op-generator
eval_mode: text
---

# Case 1: Catlass 算子开发核心流程

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: catlass-op-design;catlass-op-develop;catlass-op-perf-tune;ascendc-docs-search

## Prompt

Catlass 算子开发和普通的 Ascend C 算子开发有什么不同？请介绍一下 Catlass 算子开发的核心流程和关键特点。

## Expected Output

回复应覆盖以下要点：
1. Catlass 是 Ascend C 的高阶模板封装，算子工程结构与通用直调一致，catlass 仅决定 op_kernel 内部如何用模板拼装计算 pipeline
2. 核心开发流程包含：环境检查（含 catlass 命名校验）→ 设计（含 catlass 选型）→ 开发 → 审查 → 性能验收等多个阶段
3. 关键区别包括 CMake 需追加 catlass include 路径、op_kernel 使用模板拼装而非手写 API
4. 算子名必须包含 catlass 子串

## Expectations

- [contains] catlass

---

# Case 2: 适用场景判断

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我有一个普通的 Ascend C 向量加法算子，不涉及复杂的矩阵计算，需要用 Catlass 模板来开发吗？什么场景下适合用 Catlass，什么场景不适合？

## Expected Output

回复应说明：
1. Catlass 主要适用于需要模板拼装复杂计算 pipeline 的场景（如矩阵乘法、卷积等），简单的逐元素运算不需要 Catlass
2. 简单算子使用标准 Ascend C 矢量 API 直接开发更合适
3. Catlass 的优势在于提供高阶模板封装，减少手写复杂计算逻辑的工作量
4. 普通向量加法用标准 Ascend C 直调开发即可

## Expectations

---

# Case 3: 信息不足时主动确认

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想开发一个 catlass 算子，帮我看一下需要什么信息，先不要开始执行。

## Expected Output

回复应在开始开发前主动确认必要信息：算子名称（必须含 catlass 子串）、数学定义、数据类型（dtype）、目标 SoC 芯片型号等，而不是在缺少这些信息的情况下直接开始开发流程

## Expectations

