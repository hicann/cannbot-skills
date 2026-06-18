---
skill_name: torch-npugraph-ex-knowledge
eval_mode: text
---

# Case 1: npugraph_ex 模式核心概念

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

npugraph_ex（aclgraph）模式是什么？它的核心工作原理是怎样的？

## Expected Output

回复应说明 npugraph_ex 模式的核心概念：Capture & Replay 方式，将算子执行任务卸载到 Device 执行，减少 Host 调度开销，用于固定 shape 推理场景。应说明与 eager 模式的主要区别和适用场景。

## Expectations
- [contains] Capture & Replay


---

# Case 2: TorchAir 文档结构

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

TorchAir 中关于 npugraph_ex 的文档是怎么组织的？包含哪些主要部分？

## Expected Output

回复应说明 TorchAir 文档结构：包括概述、npugraph_ex、custom_op_graph、附录等主要部分。

## Expectations
- [contains] TorchAir
- [contains] npugraph_ex


---

# Case 3: 文档检索策略

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

torch-npugraph-ex-knowledge 使用什么策略来检索文档？为什么采用这种策略？

## Expected Output

回复应说明 refresh-first、cache-fallback 策略：优先从远程刷新获取最新文档，远程不可用时回退到本地缓存。原因在于 npugraph_ex 文档可能随版本更新，需要确保信息的时效性。

## Expectations
- [contains] 文档


---

# Case 4: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Distractor skills: torch-npugraph-ex-template;torch-npugraph-ex-dfx-triage;torch-custom-ops-guide;torch-npugraph-ex-compile-error-diagnosis
- Ascend Platform: A2

## Prompt

我想了解 npugraph_ex 模式的文档和使用方法，包括 FX Pass、编译缓存、多流并行等特性的说明。请帮我查找相关资料。

## Expected Output

回复应激活 torch-npugraph-ex-knowledge skill，说明 npugraph_ex 的核心概念和文档结构。即使在多个 npugraph_ex 相关 skill 共存的环境下，也应正确激活 torch-npugraph-ex-knowledge。

## Expectations
- [contains] npugraph_ex
- [skill_activated] torch-npugraph-ex-knowledge
