---
skill_name: triton-op-designer
---

# Case 1: Triton Ascend 算子算法草图设计

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

请为 Ascend NPU 上的一个 matmul 算子设计算法草图。算子名称是 matmul_example，任务文件路径为 /workspace/tasks/matmul.py，架构为 A2。请使用 UnifiedSketch DSL 格式输出算法草图，用于指导后续的代码生成。

## Expected Output

回复应以 sketch op_name { ... } 格式输出 UnifiedSketch DSL 描述的算法草图，包含算子类型判断（elementwise/reduce/matmul/attention/复合）、并行化策略、数据切分方式、Tile 大小选择等高层设计。应参考 npu-arch 硬件规格文档，考虑 Ascend NPU 的硬件特性（core 级别并行、内存层次、UB 容量和对齐要求）。草图中应包含 @llm_hint 注解标注优化点和权衡决策。回复应说明该草图仅用于指导后续的 triton-op-coding 代码生成，不包含可执行代码。

## Expectations
- [contains] sketch
- [contains] UnifiedSketch
- [contains] npu-arch
- [contains] @llm_hint

---

# Case 2: 设计边界与触发条件

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

triton-op-designer 和 triton-op-coding 的分工是什么？什么情况下应该先用 designer 再使用 coding？什么情况下可以直接使用 coding 而不需要 designer？

## Expected Output

回复应说明 triton-op-designer 仅生成算法草图（UnifiedSketch DSL），不生成可执行代码；而 triton-op-coding 负责将草图或任务描述转换为可执行的 Triton Ascend 代码。当用户需要从任务描述出发设计算法的并行策略和优化方向时，应使用 triton-op-designer 先设计草图，再将草图传给 triton-op-coding 进行实现。如果已经明确知道实现方案，可以直接使用 triton-op-coding 而不需要 designer。设计过程中应加载 references/sketch-design.md 了解 UnifiedSketch DSL 语法规范，并根据算子类型选择最相关的 2 个手写优化案例作为参考。

## Expectations
- [contains] triton-op-coding
- [contains] UnifiedSketch
- [contains] sketch-design.md
- [contains] 手写优化案例
