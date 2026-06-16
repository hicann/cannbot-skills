---
skill_name: ascendc-api-best-practices
eval_mode: text
---
# Case 1: DataCopy API 非对齐数据处理

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C 算子开发中，DataCopy API 用于 GM 到 UB 之间的数据搬运。请问当搬运数据不满足 32 字节对齐要求时应该如何处理？DataCopy 和 DataCopyPad 分别适用于什么场景？请介绍对齐要求和填充策略。不需要执行任何工具调用。

## Expected Output

回复应说明 Ascend C DataCopy API 的对齐要求和处理策略：
- DataCopy 仅适用于搬运数据严格 32 字节对齐的场景
- 非对齐场景必须使用 DataCopyPad 进行填充处理
- DataCopyPad 通过自动填充非对齐部分来满足对齐要求
- 黑名单提示：GlobalTensor::SetValue() 和 GetValue() 因效率极低被禁止，替代方案是 DataCopyPad

## Expectations

---

# Case 2: repeatTimes ≤ 255 限制及分批处理策略

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C API 使用中，repeatTimes 参数存在不超过 255 的限制。请问这个限制的出处和原因是什么？当需要处理的循环次数超过 255 时应该如何处理？有哪些推荐的解决方案？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 Ascend C 中 repeatTimes 的限制和处理方法：
- repeatTimes 上限为 255，原因是硬件指令编码中只分配了 8 bit
- 超过 255 时推荐使用外层循环嵌套内层 repeatTimes ≤ 255 的方案
- 也可通过多核并行（SetBlockSplitNum）或 NewPass 分段来处理

## Expectations
