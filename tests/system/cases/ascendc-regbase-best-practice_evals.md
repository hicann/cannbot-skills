---
skill_name: ascendc-regbase-best-practice
eval_mode: text
---
# Case 1: RegBase 开发模式的适用场景

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

RegBase 开发模式适用于什么场景？它和传统的 Membase 模式有什么区别？当代码中出现哪些特征信号时应该使用 RegBase 模式？不需要执行任何工具调用。

## Expected Output

回复应对比 RegBase 和 Membase 两种开发模式的区别。RegBase 操作数通过寄存器直接传递，无需 DataCopy 指令搬移数据，延迟更低，适用于数据量小、时延敏感的场景。

## Expectations

---

# Case 2: RegBase 开发的参考资源

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在进行 RegBase 算子开发时有哪些可用的参考资源？我需要查阅哪些文档来提升开发效率？如果我想参考已有的算子实现，应该从哪里入手？不需要执行任何工具调用。

## Expected Output

回复应分类列举 RegBase 开发可用的参考资源，包括官方 API 参考文档、社区真实算子实现、模板与最佳实践以及测试用例等，供开发者参考学习。

## Expectations
