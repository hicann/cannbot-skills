---
skill_name: torch-npugraph-ex-compile-error-diagnosis
eval_mode: text
---

# Case 1: 图打断（Graph Break）排查

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

torch.compile 时报 Graph Break 错误，这是什么意思？如何定位是哪里打断了图？

## Expected Output

回复应说明 Graph Break 的含义：torch.compile 遇到无法编译的 Python 特性时被迫退出编译模式。应说明如何使用 TORCH_COMPILE_DEBUG=1 和 debug_save 来定位打断位置，以及常见原因（控制流、动态数据结构、不支持的操作等）。

## Expectations
- [contains] Graph Break


---

# Case 2: Meta 推导失败诊断

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

编译时报了 "Meta derivation failed" 错误，这是什么原因？应该如何排查？

## Expected Output

回复应说明 Meta 推导失败的原因：自定义算子缺少 register_fake 实现，或 register_fake 中 shape/dtype 推导逻辑与真实计算不一致。应说明如何检查算子的 Meta 实现以及如何修复。

## Expectations
- [contains] Meta


---

# Case 3: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Distractor skills: torch-npugraph-ex-runtime-error-diagnosis;torch-npugraph-ex-dfx-triage;torch-npugraph-ex-knowledge;torch-npugraph-ex-performance-diagnosis
- Ascend Platform: A2

## Prompt

我在用 torch.compile 编译模型时遇到了 TorchDynamo 阶段的错误，报了 BackendCompilerFailed。请直接激活编译期错误诊断 skill 帮我排查，不需要经过 dfx-triage 路由。

## Expected Output

回复应激活 torch-npugraph-ex-compile-error-diagnosis skill，说明编译期错误的排查方向。即使在多个 npugraph_ex 诊断 skill 共存的环境下，也应正确激活 compile-error-diagnosis。在信息不足时，可先了解用户的编译环境、模型框架及完整报错堆栈等信息再给出具体诊断方案。

## Expectations
- [contains] BackendCompilerFailed
- [skill_activated] torch-npugraph-ex-compile-error-diagnosis
