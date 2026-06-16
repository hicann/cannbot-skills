---
skill_name: catlass-op-perf-tune
eval_mode: text
---
# Case 1: CATLASS Kernel 可调优参数

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

CATLASS 算子 kernel 性能优化时可以调整哪些参数？请逐一说明每个参数的作用。可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 CATLASS kernel 性能优化可调整的主要参数：Kernel 类型（中间 C 落盘方式，杠杆最大）、DispatchPolicy（流水调度方式，如 Pingpong/Preload/PreloadAsyncWithCallback）、L1TileShape/L0TileShape（Buffer 利用率和 K-tile 循环次数）、BlockScheduler/Swizzle（数据访问顺序和 L2 命中率）。应说明不同参数对性能的影响侧重。

## Expectations

---

# Case 2: CATLASS 头号优化方法

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

CATLASS 算子性能优化的"头号优化"是什么？应该遵循什么调优原则？不需要执行任何工具调用。

## Expected Output

回复应说明 CATLASS 算子性能优化的首要优化目标和应遵循的调优原则，包括优化方向和单变量归因等基本原则。

## Expectations
