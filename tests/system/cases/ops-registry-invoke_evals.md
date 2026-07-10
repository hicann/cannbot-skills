---
team_name: ops-registry-invoke
eval_mode: text
---

# Case 1: 自定义算子注册开发流程

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: ops-registry-invoke-workflow;ascendc-code-review;gitcode-toolkit

## Prompt

我想开发一个自定义 Ascend C 算子并注册到算子库中，ops-registry-invoke 团队能做什么？开发流程是怎样的？

## Expected Output

回复应覆盖以下要点：
1. ops-registry-invoke 管理自定义算子的完整开发生命周期：设计 → 开发 → 验证 → 上库
2.  最终产出可注册到算子库的自定义算子工程

## Expectations

- [contains] registry-invoke

---

# Case 2: 注册调用与直调方式的区别

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

算子注册调用和 Kernel 直调有什么区别？我该什么时候用 ops-registry-invoke，什么时候用 ops-direct-invoke？

## Expected Output

回复应说明：
1. 注册调用方式将算子打包为可注册到算子库的标准工程，便于共享和复用
2. 直调方式更轻量，适用于快速验证和调试
3. ops-registry-invoke 适用于需要将算子注册到算子库的场景，ops-direct-invoke 适用于快速开发验证
4. 选择依据取决于是否需要将算子入库供其他应用调用

## Expectations

---

# Case 3: 开发前需要明确的信息

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想开发一个自定义算子，先帮我看看需要准备哪些信息，先别开始开发。

## Expected Output

回复应在开发前主动确认必要信息：算子的数学定义和规格、输入输出数据类型和 shape、目标芯片架构、算子名称等，而不是在缺少这些信息的情况下直接开始开发

## Expectations

