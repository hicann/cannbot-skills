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
- Disabled: false
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

---

# Case 4: 生成 sigmoid 算子（cann-bench 评测模式，A5 平台）

## Config
- Eval Mode: cann_bench
- Cann Bench Operator: sigmoid
- Cann Bench Level: level1
- Cann Bench Device: 0
- Max Tokens: 10000000
- Timeout: 10800
- Disabled: false
- Ascend Platform: A5

## Prompt

请使用 ops-direct-invoke 团队的工作流，根据 cann-bench 的 sigmoid 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为 Ascend950。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考./asc-devkit

输出及交付件：请参考./direct_launch_example/，完成算子开发后，通过适配./direct_launch_example/csrc/ops下的算子相关文件与./direct_launch_example/cann_bench/__init__.py增加接口的方式，运行build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/sigmoid/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. 数值稳定性要求：sigmoid(x) = 1 / (1 + e^(-x))，当 x 极负时 e^(-x) 会溢出，请使用数值稳定的分段实现方式：
   - 当 x >= 0 时：sigmoid(x) = 1 / (1 + exp(-x))
   - 当 x < 0 时：sigmoid(x) = exp(x) / (1 + exp(x))
   - 避免对任意输入直接计算 exp(-x) 或 exp(x) 导致上溢
4. Kernel 实现方式约束（**必须遵守，否则编译失败**）：
   - **使用简单的 __global__ __aicore__ 模板函数方式**（参考 add/sqrt/gelu 的实现），**不要使用 class 封装**
   - Kernel 文件由 bisheng 编译器编译，**禁止使用 C++ STL**（如 `std::vector`、`std::string`、`<algorithm>` 等）
   - **不要使用 `AscendC::Sigmoid` 高阶 API**，直接用 `AscendC::Exp`、`AscendC::Add`、`AscendC::Div`、`AscendC::Muls`、`AscendC::Reciprocal` 等基础数学函数组合实现
   - 对 `GlobalTensor` 和 `LocalTensor` 的指针操作，使用 tensor 提供的 API（如 `GetPhysicalPtr`），**禁止使用 `reinterpret_cast` 在 `GlobalTensor<T>` 和指针类型之间转换**

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
1. csrc/ops/sigmoid/op_kernel/sigmoid_kernel.cpp — Ascend C Kernel，使用数值稳定的 sigmoid 实现
2. csrc/ops/sigmoid/op_kernel/sigmoid_launch.h — launch 函数声明
3. csrc/ops/sigmoid/op_plugin/sigmoid_plugin.cpp — PyTorch TORCH_LIBRARY 绑定
4. csrc/ops/sigmoid/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.sigmoid(x) 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations

---

# Case 5: 生成 gelu 算子（cann-bench 评测模式，A5 平台）

## Config
- Eval Mode: cann_bench
- Cann Bench Operator: gelu
- Cann Bench Level: level1
- Cann Bench Device: 0
- Max Tokens: 10000000
- Timeout: 10800
- Disabled: false
- Ascend Platform: A5

## Prompt

请使用 ops-direct-invoke 团队的工作流，根据 cann-bench 的 gelu 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为 Ascend950。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs（approximate 参数）、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考./asc-devkit

输出及交付件：请参考./direct_launch_example/，完成算子开发后，通过适配./direct_launch_example/csrc/ops下的算子相关文件与./direct_launch_example/cann_bench/__init__.py增加接口的方式，运行build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/gelu/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. GELU 算子有两个近似模式，都必须正确支持：
   - approximate="none"：精确模式，公式为 y = 0.5 * x * (1 + erf(x / sqrt(2)))。erf 可以使用 Ascend C 内置的 erf 数学函数计算，或通过多项式逼近实现
   - approximate="tanh"：tanh 近似模式，公式为 y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   - approximate 属性需通过 TilingData 传入 kernel，kernel 根据属性值选择对应计算分支
4. TORCH_LIBRARY 的 schema 类型与 C++ 实现类型必须严格一致。schema 中 `str` 为必需字符串（对应 C++ `std::string`），`str?` 为可选字符串（对应 C++ `c10::optional<std::string>`），**两者不可混用**。常见错误：schema 声明 `str approximate="none"` 但 C++ 实现使用 `c10::optional<std::string>`，会导致 PyTorch 注册时类型校验失败（SIGABRT）。请务必在 gelu_plugin.cpp 中使用正确的类型。
5. Kernel 实现方式必须严格遵循 `direct_launch_example/csrc/ops/add/` 的 template 函数模式，**禁止使用 class 封装**（如 `KernelGelu` 类）。具体要求：
   - 管道队列位置枚举必须使用 `AscendC::TPosition::VECIN`，不要使用 `AscendC::QuePosition::VECIN`
   - 临时缓冲区必须使用独立的 `TBuf<AscendC::TPosition::VECCALC>` 对象分配，**不要使用 `TQue::AllocTensor()` 从数据队列分配临时缓冲**
   - fp16/bf16 混合精度计算：`Cast` 到 fp32 计算后 `Cast` 回来，使用 `TBuf` 存放 fp32 数据
   - 数据搬运统一使用 `DataCopyPad` API，不要用 `reinterpret_cast` 操作 `GlobalTensor` 指针

【dtype 完整性要求 — 评测通过率的关键】
cases.yaml 中包含 float16、float32、bfloat16 三种 dtype 的测试用例，kernel 和 Host 侧必须全部正确支持：
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast→FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype，以及两种 approximate 模式（"none" 和 "tanh"）运行算子并验证输出精度。如果发现某种 dtype 或模式结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/gelu/op_kernel/gelu_kernel.cpp — Ascend C Kernel，支持精确（erf）和 tanh 近似两种模式。**必须使用 template 函数模式（参考 `add_kernel.cpp`），禁止 class 封装。** `TPosition::VECIN` + `TBuf` 临时缓冲 + `DataCopyPad`。
2. csrc/ops/gelu/op_kernel/gelu_launch.h — launch 函数声明
3. csrc/ops/gelu/op_plugin/gelu_plugin.cpp — PyTorch TORCH_LIBRARY 绑定，暴露 approximate 参数
4. csrc/ops/gelu/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.gelu(x, approximate="none") 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations
