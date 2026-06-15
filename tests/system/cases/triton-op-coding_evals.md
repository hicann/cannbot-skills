---
skill_name: triton-op-coding
---

# Case 1: Triton Ascend 算子代码生成

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

请根据以下任务描述，为一个 Ascend NPU 上的 softmax 算子生成 Triton Ascend 内核代码。算子名称是 softmax，任务文件路径为 /workspace/tasks/softmax.py，架构为 A2。请按照 triton-op-coding skill 的要求生成包含 ModelNew 类的完整内核代码。

## Expected Output

回复应生成包含 import 语句、@triton.jit 装饰的 kernel 函数以及 ModelNew(nn.Module) 类的完整 Python 代码。kernel 函数中应使用 tl.load、tl.store、tl.arange 等 Triton 语言 API 实现核心计算逻辑。ModelNew 类的 forward() 方法中只能进行 buffer 分配、形状操作、元信息查询和 kernel 启动，禁止出现 torch.*/F.* 计算操作或 tensor 方法计算。必须遵守禁止 PyTorch 退化的约束，所有核心计算必须在 @triton.jit kernel 中实现。

## Expectations
- [contains] @triton.jit
- [contains] ModelNew
- [contains] triton_ascend
- [contains] tl.load
- [contains] tl.store

---

# Case 2: 代码生成模式与使用边界

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

triton-op-coding 有哪些代码生成模式？什么情况下应该使用这个 skill，什么情况下应该使用 triton-op-designer？

## Expected Output

回复应说明 triton-op-coding 支持三种模式：首次生成（无历史信息时）、代码修改（有 previous_code + user_requirements 时）、迭代修复（有 verifier_error / conductor_suggestion 时）。该 skill 的触发条件是用户需要根据任务描述生成或迭代修复 Triton Ascend 内核代码。triton-op-designer 负责设计算法草图（UnifiedSketch DSL 格式）而非生成可执行代码，而 triton-op-coding 负责将草图或任务描述转换为可执行的 Triton Ascend 内核代码。如果传入了 sketch 参数，必须以此为基础进行代码实现。

## Expectations
- [contains] 首次生成
- [contains] 迭代修复
- [contains] triton-op-designer
- [contains] sketch
