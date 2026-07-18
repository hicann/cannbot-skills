---
skill_name: torch-npugraph-ex-performance-diagnosis
eval_mode: text
---

# Case 1: FX 图静态审计流程

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

我的模型在 npugraph_ex 下 5 步都通过了，但推理速度很慢。torch-npugraph-ex-performance-diagnosis 能帮我做什么？

## Expected Output

回复应说明 performance-diagnosis 针对"5 步全过但推理慢"的场景，通过 FX 图静态审计来识别冗余张量搬运（由 reinplace 回填不完整导致的 copy_/clone）。应说明三类冗余搬运的归类方式：input-side copy_ epilogue、auto_functionalized materialization clones、missed out-of-place ops。

## Expectations
- [contains] reinplace
- [contains] copy_


---

# Case 2: 冗余张量搬运分类

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

FX 图中的 copy_ 和 clone 信号如何分类？分别代表什么问题？

## Expected Output

回复应说明三类冗余张量搬运：input-side copy_ epilogue（输入侧多余的拷贝）、auto_functionalized materialization clones（自动函数化产生的克隆）、missed out-of-place ops（未被 inplace 替换的算子）。应说明如何交叉验证 debug.log 中的 missed opportunities 信号。

## Expectations
- [contains] copy_
- [contains] clone


---

# Case 3: 性能诊断证据链

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

torch-npugraph-ex-performance-diagnosis 诊断性能问题时需要哪些证据？如何输出审计结果？

## Expected Output

回复应说明诊断需要 TORCH_COMPILE_DEBUG=1 产生的 FX 图输出。应说明审计结果需附带文件名和行号作为证据。应说明通过 cross-reference debug.log 中的 "missed opportunities" 信号来验证推测。

## Expectations
- [contains] TORCH_COMPILE_DEBUG
- [contains] debug.log


---

# Case 4: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 280000
- Max Tokens (deepseek-v4-flash): 310000
- Max Tokens (glm-5): 295000
- Distractor skills: torch-npugraph-ex-dfx-triage;torch-npugraph-ex-compile-error-diagnosis;torch-npugraph-ex-runtime-error-diagnosis;torch-npugraph-ex-knowledge
- Ascend Platform: A2

## Prompt

我的模型在 npugraph_ex 下 5 步测试全部通过，但推理性能比预期慢很多。这是 TORCH_COMPILE_DEBUG=1 产出的日志目录，请帮我做性能诊断。

## Expected Output

回复应激活 torch-npugraph-ex-performance-diagnosis skill，说明通过 FX 图静态审计来识别冗余张量搬运问题（如 copy_、clone 等信号）。根据用户提供的日志目录，指导用户定位和分类冗余 tensor move 的成因。即使在多个 npugraph_ex 诊断 skill 共存的环境下，也应正确激活 performance-diagnosis。

## Expectations
- [contains] copy_
- [skill_activated] torch-npugraph-ex-performance-diagnosis
