---
team_name: ops-direct-invoke
eval_mode: text
---

# Case 1: 基本算子开发流程问答

## Config
- Max Tokens: 200000
- Timeout: 900

## Prompt

我想开发一个 Ascend C Kernel 直调算子，计算两个向量的逐元素加法。请描述开发这个算子的完整流程和需要关注的关键点。请包含具体的技术内容（API 名称、工具脚本、代码结构），而不仅是流程步骤的名称。

## Expected Output

回复应覆盖以下要点：
1. 环境检查方法（确认 CANN 环境和工具链是否就绪）
2. 算子设计阶段：tiling 策略选择、API 确认
3. Kernel 实现阶段：host 侧和 device 侧的代码结构
4. 代码审查和测试验证方法
5. 性能验收的基本思路

## Expectations
