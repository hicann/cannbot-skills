---
skill_name: ascendc-simt-tiling-design
eval_mode: text
---
# Case 1: SIMT vs SIMD Tiling 设计差异

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

SIMT 算子的切分设计与 SIMD 有什么本质差异？SIMT 算子切分不涉及哪些 SIMD 环节？不需要执行任何工具调用。

## Expected Output

回复应说明 SIMT 和 SIMD 在切分设计上的本质差异在于并行粒度不同：SIMD 以向量级（UB 空间规划）为核心，SIMT 以线程级（核数/线程数规划）为核心。SIMT 切分不涉及的 SIMD 环节包括 UB 空间切分、Vector 指令流水线编排等。

## Expectations

---

# Case 2: SIMT Tiling 核心设计要素

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

SIMT 算子 Tiling 设计的核心要素有哪些？核数、线程数和 DCache 分别如何设置？不需要执行任何工具调用。

## Expected Output

回复应说明 SIMT Tiling 的核心设计要素：核数切分（BlockDim）、线程数（ThreadNum）、DCache/UB 空间分配、数据切分方式。核数通过 SetBlockDim(n) 设置，线程数通过 SetThreadNum(n) 设置，DCache 用于数据预取和缓存。

## Expectations
