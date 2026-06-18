---
skill_name: torch-custom-ops-guide
eval_mode: text
---

# Case 1: 自定义算子入图完整流程

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Ascend Platform: A2

## Prompt

我有一个已在 Eager 模式下使用 torch.library.custom_op 注册并跑通的自定义 PyTorch 算子，想接入 npugraph_ex 图模式。请问完整的接入流程是什么？

## Expected Output

回复应说明当前算子的状态确认（Eager 已有、custom_op 注册）后，下一步即为编写 Meta 推导函数（register_fake）实现 shape/dtype/device 传播，完成后即可支持入图；如遇问题可路由到 dfx-triage 子 skill 进行问题定位。

## Expectations
- [contains] register_fake
- [contains] Meta


---

# Case 2: custom_op vs Library 注册方式

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

torch-custom-ops-guide 支持哪两种算子注册方式？各适用于什么场景？

## Expected Output

回复应说明两种注册方式：torch.library.custom_op（推荐用于单个自定义算子，简洁）和纯 Python torch.library.Library（适用于批量注册或需要更精细控制）。应说明各自的使用场景和选择依据。

## Expectations
- [contains] custom_op
- [contains] Library


---

# Case 3: Meta 推导函数编写要点

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 165000
- Ascend Platform: A2

## Prompt

编写 Meta 推导函数（register_fake）时需要注意哪些要点？它解决了什么问题？

## Expected Output

回复应说明 Meta 推导函数的作用：在不实际执行计算的情况下推导输出的 shape/dtype/device，使 torch.compile 能够构建计算图。应说明编写要点包括正确推导输出张量的元信息、处理动态 shape、以及常见陷阱。

## Expectations
- [contains] register_fake
- [contains] shape


---

# Case 4: 信息不足时主动追问

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 150000
- Max Tokens (glm-5): 135000
- Ascend Platform: A2

## Prompt

我想让我的算子支持图模式。

## Expected Output

回复应主动询问关键信息：算子当前状态（从零开发/Eager已有）、注册方式偏好、是否已有图模式相关报错等。不应在缺乏信息的情况下直接给出具体代码。

## Expectations
- [contains] 算子


---

# Case 5: 正向看护-多 graph skill 环境下正确触发

## Config
- Max Tokens: 180000
- Max Tokens (deepseek-v4-flash): 210000
- Max Tokens (glm-5): 195000
- Distractor skills: torch-npugraph-ex-knowledge;torch-npugraph-ex-dfx-triage;torch-npugraph-ex-template;torch-npugraph-ex-compile-error-diagnosis
- Ascend Platform: A2

## Prompt

我有一个自定义的 FlashAttention 算子，已经在 Eager 模式下跑通了，现在想把它接入 npugraph_ex 图模式。请问需要做什么？

## Expected Output

回复应激活 torch-custom-ops-guide skill，说明自定义算子入图的流程。即使在多个 npugraph_ex 相关 skill 共存的环境下，也应正确激活 torch-custom-ops-guide。在信息不足时，回应用户问题可先询问算子的当前状态（如是否已有 Eager 实现、注册方式等），再给出完整入图指导。

## Expectations
- [contains] Eager
- [skill_activated] torch-custom-ops-guide
