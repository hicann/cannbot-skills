---
skill_name: ascendc-code-review
eval_mode: text
---
# Case 1: 代码检视工作流路由

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-code-review 技能在接收到代码检视请求时，如何根据用户输入选择合适的工作流？请介绍该技能支持的主要工作流类型以及各自的路由逻辑。不需要执行任何工具调用。

## Expected Output

回复应说明 ascendc-code-review 的工作流路由机制：
- file-review：全量检视代码、审核代码场景
- quick-review：快速检视、检查问题场景
- pr-review：PR 检视场景（匹配 pr #、pull request 等关键词）
- design-consistency：设计实现一致性检查场景（对照 DESIGN.md）
- 路由逻辑基于用户输入的关键词和意图自动匹配对应工作流，严格按工作流阶段顺序执行，禁止跳步

## Expectations

---

# Case 2: 代码检视检查项与规范

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在使用 ascendc-code-review 技能进行 Ascend C 代码检视时，主要检查哪些方面的内容？有哪些检视规则和规范文档可以参考？检视完成后如何输出结果？不需要执行任何工具调用。

## Expected Output

回复应介绍代码检视的主要检查维度（安全编码、API 最佳实践、性能约束等）、可参考的规范文档，以及检视结果的输出方式。

## Expectations
