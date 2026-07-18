---
skill_name: torch-npugraph-ex-dfx-triage
eval_mode: text
---

# Case 1: DFX 问题分诊入口流程

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

我的模型在用 npugraph_ex 编译时出了问题，不知道是编译错误还是运行时错误。torch-npugraph-ex-dfx-triage 能帮我做什么？

## Expected Output

回复应说明 dfx-triage 是 npugraph_ex DFX 问题的统一入口，通过 5 步分层日志采集工作流收集信息，然后路由到对应的子 skill（compile-error/runtime-error/accuracy/performance）。它本身不做最终诊断结论，而是负责分诊和路由。

## Expectations
- [contains] 路由


---

# Case 2: 5 步分层采集工作流

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

torch-npugraph-ex-dfx-triage 的 5 步分层采集工作流是什么？每一步的作用是什么？

## Expected Output

回复应说明 5 步工作流：pure eager → backend="eager" → backend="aot_eager" → backend="npugraph_ex" with force_eager=True → 完整 backend="npugraph_ex"。每一步逐步增加编译层，在首次出错的步骤定位问题，然后路由到对应的子 skill。

## Expectations
- [contains] eager
- [contains] aot_eager


---

# Case 3: 问题路由机制

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

torch-npugraph-ex-dfx-triage 采集完信息后，如何判断应该路由到哪个子 skill？

## Expected Output

回复应说明基于哪一步首次失败来路由：编译期失败路由到 compile-error-diagnosis、运行时失败路由到 runtime-error-diagnosis、精度问题路由到对应的精度子 skill、性能问题路由到 performance-diagnosis。应说明 dfx-triage 收集完整信息后不自己下结论。

## Expectations
- [contains] compile-error
- [contains] runtime-error


---

# Case 4: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Distractor skills: torch-npugraph-ex-compile-error-diagnosis;torch-npugraph-ex-runtime-error-diagnosis;torch-npugraph-ex-knowledge;torch-npugraph-ex-performance-diagnosis
- Ascend Platform: A2

## Prompt

我的模型在用 torch.compile + npugraph_ex 时报错了，我不确定是什么类型的问题。帮我排查一下。

## Expected Output

回复应激活 torch-npugraph-ex-dfx-triage skill，说明将通过 5 步分层采集来定位问题并路由到正确的子 skill。即使在多个 npugraph_ex 诊断 skill 共存的环境下，也应正确激活 dfx-triage 作为统一入口。

## Expectations
- [skill_activated] torch-npugraph-ex-dfx-triage
