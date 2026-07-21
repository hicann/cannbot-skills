---
name: dsl-optimization
disable-model-invocation: true
description: Iteratively optimize AscendDSL code for performance — tiling tuning, vectorization, pipeline adjustments. Use after evaluation shows correctness passes but speedup is below target.
---

## What I do

Optimize DSL code through multiple rounds, improving tiling strategies, memory access patterns, and parallelization.

## Workflow

1. Read baseline DSL from `output/dsl/{op_name}_baseline.py`
2. Load optimization thought list from `prompt_template/optimize_pseudo_template.py`
3. Iterate for `n_rounds` (default 3):
   - Call `@general` with:
     - Hardware spec
     - Current DSL code
     - Specific optimization thought for this round
   - Extract optimized code and thoughts
   - Save thoughts to `output/dsl/{op_name}_thought_{i}.txt`
   - Save code to `output/dsl/{op_name}_optimize_{i}.py`
   - Use optimized code as input for next round
4. Continue to the next step in agent workflow

## Output

- Multiple optimized DSL versions
- Optimization thought log for each round
- Incrementally improved code
