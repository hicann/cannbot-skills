---
team_name: ops-direct-invoke
eval_mode: text
---
# Case 1: 基本算子开发流程问答

## Config
- Max Tokens: 200000
- Timeout: 900
- Disabled: false
- Truncate Length: 50000
- Ascend Platform: A2

## Prompt

我想开发一个 Ascend C Kernel 直调算子，计算两个向量的逐元素加法。请描述开发这个算子的完整流程和需要关注的关键点。请包含具体的技术内容（API 名称、工具脚本、代码结构），而不仅是流程步骤的名称。

【约束】请直接根据你的知识回答，不要执行任何工具调用（禁止 read、glob、bash 等），不要探索项目文件或目录结构。仅输出文本回答即可。

## Expected Output

回复应覆盖以下要点：
1. 环境检查方法（确认 CANN 环境和工具链是否就绪）
2. 算子设计阶段：tiling 策略选择、API 确认
3. Kernel 实现阶段：host 侧和 device 侧的代码结构
4. 代码审查和测试验证方法
5. 性能验收的基本思路

## Expectations

---

# Case 2: 生成 mish 算子（cann-bench 评测模式）

## Config
- Eval Mode: cann_bench
- Cann Bench Operator: mish
- Cann Bench Level: level1
- Cann Bench Device: 0
- Max Tokens: 10000000
- Disabled: true
- Timeout: 10800
- Ascend Platform: A2

## Prompt

请使用 ops-direct-invoke 团队的工作流，根据 cann-bench 的 mish 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为Ascend910B3。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch/tensorflow）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考./asc-devkit

输出及交付件：请参考./direct_launch_example/，完成算子开发后，通过适配./direct_launch_example/csrc/ops下的算子相关文件与./direct_launch_example/cann_bench/__init__.py增加接口的方式，运行build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/mish/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. 数值稳定性要求：mish(x) = x * tanh(softplus(x))，当 x 较大时 exp(x) 会溢出，请使用数值稳定的实现方式（如分段计算：x > 20 时 mish(x) ≈ x）

【dtype 完整性要求 — 评测通过率的关键】
cases.yaml 中包含 float16、float32、bfloat16 三种 dtype 的测试用例，kernel 和 Host 侧必须全部正确支持：
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast→FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype 运行算子并验证输出精度。如果发现某种 dtype 结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/mish/op_kernel/mish_kernel.cpp — Ascend C Kernel，使用数值稳定的 mish 实现
2. csrc/ops/mish/op_kernel/mish_launch.h — launch 函数声明
3. csrc/ops/mish/op_plugin/mish_plugin.cpp — PyTorch TORCH_LIBRARY 绑定
4. csrc/ops/mish/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.mish(x) 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations

---

# Case 3: 生成 mish 算子（cann-bench 评测模式，A5 平台）

## Config
- Eval Mode: cann_bench
- Cann Bench Operator: mish
- Cann Bench Level: level1
- Cann Bench Device: 0
- Max Tokens: 10000000
- Timeout: 21600
- Ascend Platform: A5

## Prompt

请使用 ops-direct-invoke 团队的工作流，根据 cann-bench 的 mish 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为Ascend950。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch/tensorflow）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考./asc-devkit

输出及交付件：请参考./direct_launch_example/，完成算子开发后，通过适配./direct_launch_example/csrc/ops下的算子相关文件与./direct_launch_example/cann_bench/__init__.py增加接口的方式，运行build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/mish/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. 数值稳定性要求：mish(x) = x * tanh(softplus(x))，当 x 较大时 exp(x) 会溢出，请使用数值稳定的实现方式（如分段计算：x > 20 时 mish(x) ≈ x）

【dtype 完整性要求 — 评测通过率的关键】
cases.yaml 中包含 float16、float32、bfloat16 三种 dtype 的测试用例，kernel 和 Host 侧必须全部正确支持：
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast→FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype 运行算子并验证输出精度。如果发现某种 dtype 结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/mish/op_kernel/mish_kernel.cpp — Ascend C Kernel，使用数值稳定的 mish 实现
2. csrc/ops/mish/op_kernel/mish_launch.h — launch 函数声明
3. csrc/ops/mish/op_plugin/mish_plugin.cpp — PyTorch TORCH_LIBRARY 绑定
4. csrc/ops/mish/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.mish(x) 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations
