---
skill_name: ascendc-docs-search
eval_mode: text
---
# Case 1: API 文档搜索策略

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-docs-search 技能如何搜索 Ascend C API 文档和示例代码？搜索优先级是什么？本地有哪些可用资源？不需要执行任何工具调用。

## Expected Output

回复应说明搜索策略采用"本地优先、在线兜底"原则，优先在本地 API 文档中搜索（约 1022 个 API 文档），其次查找示例代码，最后在线兜底搜索。

## Expectations

---

# Case 2: 本地可搜索的资源类型

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-docs-search 技能可以搜索哪些本地资源？具体能搜索到哪些类型的技术文档和示例？不需要执行任何工具调用。

## Expected Output

回复应列出该技能可搜索的本地资源类型，包括 API 文档、示例代码、实现参考。同时说明搜索优先级路径。

## Expectations
