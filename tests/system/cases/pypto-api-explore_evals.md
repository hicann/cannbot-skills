---
skill_name: pypto-api-explore
---

# Case 1: PyPTO API 探索与可行性分析

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

我想实现一个 Softmax 算子，请帮我做 API 探索和可行性分析。算子需求如下：

算子名称：Softmax
数学公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
输入：x fp16 [1024, 4096]
输出：y fp16 [1024, 4096]

请运行完整工作流，生成 API_REPORT.md。

## Expected Output

回复应调用 pypto-api-explore skill，按工作流完成 API 探索并生成 API_REPORT.md。应覆盖以下要点：
- 使用三个并行 Explore subagent 分别搜索 pypto/docs/zh/、models/ 和 examples/ 目录
- 将 Softmax 分解为 exp、sum（归约）、div 等原子操作，并查找对应的 PyPTO API
- 提供 API 映射表（exp → pypto.exp, sum → pypto.sum, div → pypto.div）
- 检查 API 约束：dtype 支持、shape 范围、sigmoid 仅 FP32 等硬约束
- 进行 Tiling 需求分析：判断为 Vector 类型（不含 matmul），需使用 set_vec_tile_shapes
- 提供参考实现搜索结果
- 进行可行性判断和风险评估
- 最终输出 API_REPORT.md 到当前目录

## Expectations
- [contains] API_REPORT.md
- [contains] Explore subagent
- [contains] pypto.exp
- [contains] pypto.sum
- [contains] set_vec_tile_shapes
- [contains] 约束

---

# Case 2: 使用边界与前置条件

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

pypto-api-explore skill 的使用边界是什么？它和 pypto-op-design（算子设计）有什么不同？什么情况下应该用 API 探索而不是直接开始设计或者编码？

## Expected Output

回复应说明 pypto-api-explore 的适用范围和使用边界：
- 用于算子开发前期的 API 可行性和约束检查阶段
- 输出 API_REPORT.md，包含 API 映射、约束检查、Tiling 需求、参考实现和风险评估
- 不生成算子实现代码（那是 pypto-op-develop 的职责）
- 不生成设计方案（那是 pypto-op-design 的职责）
- 当不确定 PyPTO 是否支持某个操作时使用
- 当需要验证 API 约束（dtype、shape、Tiling）时使用
- 当需要查找参考实现模式时使用
- 输入不足时主动向用户提问补充

## Expectations
