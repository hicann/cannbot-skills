---
skill_name: triton-task-extractor
---

# Case 1: 从 PyTorch 代码中提取算子任务文件

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

我有一段 PyTorch 代码 /workspace/models/my_op.py，其中包含一个 matmul_with_bias 函数需要优化，shape 信息已知。请使用 triton-task-extractor 将其提取为标准化算子任务文件。

## Expected Output

回复应分析源代码，将待优化算子逻辑包装到 Model.forward() 中，初始化状态放入 Model.__init__()，所有依赖的自定义函数内联到文件中。输出应包含 class Model(nn.Module)、def get_inputs() 和 def get_init_inputs() 三个部分。生成后必须使用本 skill 自带的 scripts/validate_task.py 进行静态检查和运行时检查，验证通过后还需使用 question 工具请求用户确认。应说明输出为单一自包含 Python 文件，只允许 torch/torch.nn/标准库的 import。

## Expectations
- [contains] class Model
- [contains] get_inputs
- [contains] get_init_inputs
- [contains] validate_task.py

---

# Case 2: 单 case 与多 case 模式的选择

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

triton-task-extractor 的两种模式（单 case 和多 case）有什么区别？什么情况下用哪种模式？源文件有多个 shape 需要测试时应该如何处理？

## Expected Output

回复应说明单 case 模式适用于单一自包含 .py 文件，get_inputs 返回单组输入；多 case 模式适用于 .py + 同名 .json 配对，get_input_groups 返回多组输入。判定规则为：如果源 .py 已定义 get_input_groups() 函数，或同目录存在同名 .json 文件，则走多 case 模式，否则走单 case 模式。多 case 模式的核心原则是原样透传、不改写源码，禁止将多 case 源降级为单 case 输出来通过验证。源文件有多 shape 需要测试时应使用多 case 模式。

## Expectations
- [contains] 单 case
- [contains] 多 case
- [contains] get_input_groups
- [contains] .json
