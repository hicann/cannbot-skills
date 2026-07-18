---
skill_name: tilelang-op-develop
---

# Case 1: 根据设计文档生成算子代码

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

请介绍 tilelang-op-develop 技能如何根据设计文档生成算子代码的核心工作流程，包括从设计文档提取哪些关键信息、如何查找参考实现、以及生成代码后的验证步骤。不需要执行任何工具调用。

## Expected Output

回复应从 design.md 提取关键信息，基于参考实现生成算子代码，并运行验证检查代码正确性。

## Expectations
- [contains] design.md
- [contains] examples/
- [contains] golden
- [contains] 验证

---

# Case 2: 参考来源优先级与冲突处理

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

当设计文档中的伪代码和 examples/ 目录下的参考实现不一致时，应该以哪个为准？如何处理这种冲突？

## Expected Output

回复应以 examples/ 中的参考实现为准，因为参考实现已验证通过，可信度高。当 design.md 与参考实现冲突时，优先参考参考实现，在代码注释中记录差异说明为何偏离，重大差异需询问用户确认。同时强调应查阅 tilelang-api-best-practices 和 tilelang-programming-model-guide 获取 API 用法和 pass_configs 的正确配置。

## Expectations
- [contains] 参考实现
- [contains] design.md
- [contains] tilelang-api-best-practices
- [contains] 冲突处理
