---
skill_name: ops-spec-gen
eval_mode: text
---
# Case 1: ops-spec-gen 技能的主要功能与两种模式

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ops-spec-gen 技能的主要功能是什么？它有哪两种运行模式？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 ops-spec-gen 的两个主要模式：生成模式用于创建 spec.yaml 脚手架文件，校验模式用于执行 9 阶段 L0 验证以确保算子规格定义的正确性和完整性。

## Expectations

---

# Case 2: spec.yaml 的 9 阶段 L0 验证

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ops-spec-gen 的 9 阶段 L0 验证具体包含哪些检查项？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 spec.yaml 的 9 阶段 L0 验证具体包含哪些检查项以及每阶段的作用。

## Expectations
