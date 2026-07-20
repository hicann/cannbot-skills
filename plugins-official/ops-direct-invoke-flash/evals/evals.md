---
team_name: ops-direct-invoke-flash
eval_mode: text
---

# Case 1: Flash 版直调开发的核心流程

## Config
- Max Tokens: 500000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: ascendc-direct-invoke-template;ascendc-env-check;gitcode-toolkit

## Prompt

我有一段数学公式，想在昇腾 NPU 上实现为核函数，ops-direct-invoke-flash 团队适合做这个吗？它的开发流程是怎样的？

## Expected Output

回复应覆盖以下要点：
1. ops-direct-invoke-flash 适用于从 CPU 函数、数学公式、代码片段或文本描述出发构建并验证新的 Ascend NPU 核函数
2. 核心流程包含：环境检查 → 设计 → 开发 → 测试 → 验收，覆盖从规格到经验证核函数的完整路径
3. 默认在 operators/ 目录下开发，支持使用 /ops-direct-invoke-flash 技能
4. 输出经验证的 NPU 核函数

## Expectations

- [contains] flash

---

# Case 2: 信息不足时主动确认

## Config
- Max Tokens: 500000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想用 Flash 模式开发一个 NPU 核函数，先告诉我你需要什么信息，不用开始开发。

## Expected Output

回复应主动确认必要信息：待实现的数学公式或算法描述、输入输出数据类型和格式、目标芯片型号等规格信息，而不是在缺少这些关键信息的情况下直接开始开发

## Expectations

---

# Case 3: 生成 mish 算子（cann-bench 评测模式，A5 平台，Flash 版）

## Config
- Eval Mode: cann_bench
- Cann Bench Operator: mish
- Cann Bench Level: level1
- Cann Bench Device: 0
- Max Tokens: 10000000
- Timeout: 10800
- Disabled: false
- Ascend Platform: A5

## Prompt

请使用 ops-direct-invoke-flash 团队的工作流（8 阶段 TDD 流程），根据 cann-bench 的 mish 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为 Ascend950（dav-3510），使用经典 AscendC API（参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，用 AscendC::Max、AscendC::Exp、AscendC::Mul、AscendC::Adds、AscendC::Div 等基础数学函数组合）完成算子开发。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考 ./asc-devkit

输出及交付件：请参考 ./direct_launch_example/，完成算子开发后，通过适配 ./direct_launch_example/csrc/ops 下的算子相关文件与 ./direct_launch_example/cann_bench/__init__.py 增加接口的方式，运行 build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/mish/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. 数值稳定性要求：mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1+e^x))，当 x 较大时 exp(x) 会溢出：
   - **禁止使用 __simd_vf__ 函数和 __ubuf__ 指针**实现 mish（CANN 9.0.0 的 LoadAlign 不支持 LoadDist 参数）
   - 正确方式：使用经典 AscendC API + tanh(softplus) 的代数展开，无需 Log 和 Tanh 函数
   - 展开公式：t = exp(clamp(x, -20, 20)) → tanh(softplus(x)) = t(2+t) / (2 + 2t + t²) → mish(x) = x * t(2+t) / (2 + 2t + t²)
   - 裁剪方法：使用两次 AscendC::Max 组合实现 clamp(x, -20, 20)：先 Max(x, -20) 下界，再取负后 Max 上界

【dtype 完整性要求 — 评测通过率的关键】
cases.yaml 中包含 float16、float32、bfloat16 三种 dtype 的测试用例，kernel 和 Host 侧必须全部正确支持：
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast → FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【经典 AscendC API 使用规范 — 必须严格遵守】
1. Kernel 实现方式严格参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，**禁止使用 __simd_vf__ 函数和 __ubuf__ 指针**，禁止使用 class 封装
2. **禁止在 __global__ __aicore__ 函数中使用 C++ lambda 表达式**。bisheng 编译器将 lambda 内代码视为 [host] 函数，导致 AllocTensor、DeQue、DataCopyExtParams 等 __aicore__ API 不可用。pipeline 逻辑必须直接写在函数体循环中（精确参考 `direct_launch_example/csrc/ops/add/op_kernel/add_kernel.cpp` 的写法）
3. fp16/bf16 混合精度计算：使用 AscendC::Cast 将输入 Cast 到 fp32，使用独立的 `TBuf<AscendC::TPosition::VECCALC>` 存放中间 fp32 数据，计算完成后 Cast 回原 dtype
4. **禁止使用 AscendC::Compares + AscendC::Select 组合**进行分支选择操作
5. mish 计算链（经典 AscendC API 组合，必须严格按此顺序）：
   - `AscendC::Max(clamped, x, -20.0f, count)` → 裁剪下界
   - `AscendC::Muls(negClamped, clamped, -1.0f, count)` → 取负
   - `AscendC::Max(negClamped, negClamped, -20.0f, count)` → 裁剪上界
   - `AscendC::Muls(clamped, negClamped, -1.0f, count)` → clamp(x, -20, 20) 完成
   - `AscendC::Exp(t, clamped, count)` → t = exp(clamped)
   - `AscendC::Muls(twoT, t, 2.0f, count)` → 2t
   - `AscendC::Mul(t2, t, t, count)` → t²
   - `AscendC::Add(num, twoT, t2, count)` → 2t + t² = t(2+t)
   - `AscendC::Adds(den, num, 2.0f, count)` → 2 + 2t + t² = (t+1)² + 1
   - `AscendC::Div(ratio, num, den, count)` → tanh(softplus(x))
   - `AscendC::Mul(result, x, ratio, count)` → mish(x) = x * ratio
5. 数据搬运统一使用 DataCopyPad API；禁止使用 reinterpret_cast 转换 GlobalTensor 指针
6. 禁止使用 AscendC::MicroAPI、Membase；禁止使用 C++ STL（std::vector、std::string、<algorithm> 等）
7. 临时缓冲区必须使用独立的 TBuf 对象分配，**禁止使用 TQue::AllocTensor() 从数据队列分配临时缓冲**

【Tiling 性能优化 — 影响性能得分的关键，必须遵守】
1. **禁止使用固定 tile size**（如 FIXED_TILE_ELEMS = 2048）。必须参考 add 算子的 tiling 方式：
   - 通过 `platform_ascendc::PlatformAscendCManager::GetInstance()` 获取 `ubSize`
   - 根据 UB 大小动态计算每 tile 的元素数
2. tile size 计算方式（单位：字节）：
   - 每 tile 的缓冲区总需求 = inQueue(1个buffer) + outQueue(1个buffer) + TBufs(混合精度时的额外 float 缓冲)
   - `tileSizeBytes = ubSize / PIPELINE_DEPTH / (2 + 混合精度TBuf数量)`
   - `elementNumPerTile = tileSizeBytes / sizeof(T)`

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype 运行算子并验证输出精度。如果发现某种 dtype 结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/mish/op_kernel/mish_kernel.cpp — Ascend C Kernel，使用数值稳定的 mish 实现，遵循经典 AscendC API 规范
2. csrc/ops/mish/op_kernel/mish_launch.h — launch 函数声明
3. csrc/ops/mish/op_plugin/mish_plugin.cpp — PyTorch TORCH_LIBRARY 绑定
4. csrc/ops/mish/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.mish(x) 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations

---

# Case 4: 生成 sigmoid 算子（cann-bench 评测模式，A5 平台，Flash 版）

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

请使用 ops-direct-invoke-flash 团队的工作流（8 阶段 TDD 流程），根据 cann-bench 的 sigmoid 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为 Ascend950（dav-3510），使用经典 AscendC API（参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，用 AscendC::Exp、AscendC::Add、AscendC::Div、AscendC::Muls 等基础数学函数组合）完成算子开发。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考 ./asc-devkit

输出及交付件：请参考 ./direct_launch_example/，完成算子开发后，通过适配 ./direct_launch_example/csrc/ops 下的算子相关文件与 ./direct_launch_example/cann_bench/__init__.py 增加接口的方式，运行 build.sh，可以判断是否存在编译问题。

【重要约束 — 必须严格遵守】
1. output/ 目录中只需要包含以下文件（这些是你需要生成的）：
   - csrc/ops/sigmoid/ — 算子 kernel 实现（含 CMakeLists.txt、op_kernel/、op_plugin/）
   - cann_bench/__init__.py — 算子 Python 接口
2. 禁止在 output/ 中生成以下基础设施文件，它们在 ./direct_launch_example/ 中已经正确配置，评测框架会自动使用模板中的版本：
   - csrc/extension.cpp（使用纯 Python C API，不要用 pybind11 重写）
   - 顶层 CMakeLists.txt、setup.py、build.sh、requirements.txt
   - cmake/ 目录下的所有 .cmake 文件
   - csrc/CMakeLists.txt、csrc/ops/CMakeLists.txt
3. 数值稳定性要求：sigmoid(x) = 1 / (1 + e^(-x))，当 x 极负时 e^(-x) 会溢出：
   - **禁止使用 Abs + Compares + Select 分支选择方式**（该组合在 Ascend950 上分支反转，导致 MERE 极高）
   - 正确方式：对输入做单侧裁剪后直接计算 `sigmoid(x) = 1 / (1 + exp(-x))`
   - 裁剪方法：使用 `AscendC::Max(z, x, -80.0f, count)` 将 x 裁剪到 >= -80，确保 exp(-x) <= exp(80) 不溢出 fp32 范围
   - 无需对 x > 0 做裁剪：x 为正时 exp(-x) < 1 不会溢出
4. Kernel 实现方式约束（**必须遵守，否则编译失败**）：
   - **使用简单的 __global__ __aicore__ 模板函数方式**（参考 add/sqrt/gelu 的实现），**不要使用 class 封装**
   - Kernel 文件由 bisheng 编译器编译，**禁止使用 C++ STL**（如 `std::vector`、`std::string`、`<algorithm>` 等）
   - **禁止使用 `AscendC::Sigmoid` 高阶 API**，直接用 `AscendC::Exp`、`AscendC::Add`、`AscendC::Div`、`AscendC::Muls`、`AscendC::Reciprocal` 等基础数学函数组合实现
   - 对 `GlobalTensor` 和 `LocalTensor` 的指针操作，使用 tensor 提供的 API（如 `GetPhysicalPtr`），**禁止使用 `reinterpret_cast` 在 `GlobalTensor<T>` 和指针类型之间转换**

【dtype 完整性要求 — 评测通过率的关键】
cases.yaml 中包含 float16、float32、bfloat16 三种 dtype 的测试用例，kernel 和 Host 侧必须全部正确支持：
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast → FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【经典 AscendC API 使用规范 — 必须严格遵守】
1. Kernel 实现方式严格参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，**禁止使用 __simd_vf__ 函数和 __ubuf__ 指针**，禁止使用 class 封装
2. **禁止在 __global__ __aicore__ 函数中使用 C++ lambda 表达式**。pipeline 逻辑必须直接写在函数体循环中（精确参考 add_kernel.cpp 的写法）
3. fp16/bf16 混合精度计算：使用 AscendC::Cast 将输入 Cast 到 fp32，使用独立的 `TBuf<AscendC::TPosition::VECCALC>` 存放中间 fp32 数据，计算完成后 Cast 回原 dtype
4. sigmoid 计算公式：`AscendC::Max(z, x, -80.0f)` 裁剪 → `Muls(-1.0f)` → `Exp` → `Adds(1.0f)` → `Reciprocal`，**禁止调用 AscendC::Sigmoid**
5. **禁止使用 AscendC::Compares + AscendC::Select 组合**进行任何分支选择操作
5. 数据搬运统一使用 DataCopyPad API；禁止使用 reinterpret_cast 转换 GlobalTensor 指针
6. 禁止使用 AscendC::MicroAPI、Membase；禁止使用 C++ STL（std::vector、std::string、<algorithm> 等）
7. 临时缓冲区必须使用独立的 TBuf 对象分配，**禁止使用 TQue::AllocTensor() 从数据队列分配临时缓冲**

【Tiling 性能优化 — 影响性能得分的关键，必须遵守】
1. **禁止使用固定 tile size**（如 FIXED_TILE_ELEMS = 2048）。必须参考 add 算子的 tiling 方式：
   - 通过 `platform_ascendc::PlatformAscendCManager::GetInstance()` 获取 `ubSize`
   - 根据 UB 大小动态计算每 tile 的元素数
2. tile size 计算方式（单位：字节）：
   - 每 tile 的缓冲区总需求 = inQueue(1个buffer) + outQueue(1个buffer) + TBufs(混合精度时的xFloat+zFloat)
   - `tileSizeBytes = ubSize / PIPELINE_DEPTH / (2 + (混合精度需要的TBuf数量))`
   - `elementNumPerTile = tileSizeBytes / sizeof(T)`（对fp16:T=2, fp32:T=4, bf16:T=2）
   - 混合精度时 T=2 字节的 dtype 需额外计入 2 个 float TBuf（每 TBuf = tile * sizeof(float) 字节）

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype 运行算子并验证输出精度。如果发现某种 dtype 结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/sigmoid/op_kernel/sigmoid_kernel.cpp — Ascend C Kernel，使用数值稳定的 sigmoid 实现，遵循经典 AscendC API 规范
2. csrc/ops/sigmoid/op_kernel/sigmoid_launch.h — launch 函数声明
3. csrc/ops/sigmoid/op_plugin/sigmoid_plugin.cpp — PyTorch TORCH_LIBRARY 绑定
4. csrc/ops/sigmoid/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.sigmoid(x) 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations

---

# Case 5: 生成 gelu 算子（cann-bench 评测模式，A5 平台，Flash 版）

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

请使用 ops-direct-invoke-flash 团队的工作流（8 阶段 TDD 流程），根据 cann-bench 的 gelu 算子任务定义生成一个完整的 Ascend C Kernel 直调算子，芯片信息为 Ascend950（dav-3510），使用经典 AscendC API（参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，用 AscendC::Mul、AscendC::Muls、AscendC::Add、AscendC::Adds、AscendC::Exp、AscendC::Reciprocal 等基础数学函数组合）完成算子开发。

任务定义文件在 ./cann-bench-task/ 目录下，请仔细阅读以下文件了解算子规格：
- cases.csv / cases.yaml：算子用例信息，包含 input_shape、dtype、attrs（approximate 参数）、value_range、baseline_perf_us
- desc.md：算子描述信息，包含数学公式、输入输出信息、精度标准等
- golden.py：算子对标竞品的标杆实现（torch）
- proto.yaml：算子原型信息，包含算子分类、支持的数据类型、难度分类等

输入知识：关于昇腾算子开发的接口、芯片等相关知识可以参考 ./asc-devkit

输出及交付件：请参考 ./direct_launch_example/，完成算子开发后，通过适配 ./direct_launch_example/csrc/ops 下的算子相关文件与 ./direct_launch_example/cann_bench/__init__.py 增加接口的方式，运行 build.sh，可以判断是否存在编译问题。

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
   - approximate="none"：精确模式，公式为 y = 0.5 * x * (1 + erf(x / sqrt(2)))。erf 可以通过多项式逼近实现
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
- Kernel 层：为每种 dtype 提供正确的模板特化（fp16 和 bf16 需要 Cast → FP32 混合精度计算以保证精度达标）
- Host 分发层：bfloat16 和 float16 都是 2 字节，**不能仅按 dtypeBytes 区分**。必须在 TilingData 中增加 dtypeId 字段（如 0=fp32, 1=fp16, 2=bf16），Host 按 dtypeId 分发到对应的 launch 函数
- 常见错误：Host 只判断 dtypeBytes==2 就统一调用 half kernel，导致 bfloat16 数据被当作 fp16 处理，输出完全错误（MERE=1.0）

【经典 AscendC API 使用规范 — 必须严格遵守】
1. Kernel 实现方式严格参考 add/sqrt 的 __global__ __aicore__ 模板函数模式，**禁止使用 __simd_vf__ 函数和 __ubuf__ 指针**，禁止使用 class 封装
2. **禁止在 __global__ __aicore__ 函数中使用 C++ lambda 表达式**。pipeline 逻辑必须直接写在函数体循环中（精确参考 add_kernel.cpp 的写法）
3. fp16/bf16 混合精度计算：使用 AscendC::Cast 将输入 Cast 到 fp32，使用独立的 `TBuf<AscendC::TPosition::VECCALC>` 存放中间 fp32 数据，计算完成后 Cast 回原 dtype
4. **禁止使用 AscendC::Compares + AscendC::Select 组合**进行分支选择操作
5. **approximate="tanh" 模式计算链**（经典 AscendC API，严格按此顺序）：
   - `AscendC::Mul(x2, x, x, count)` → x²
   - `AscendC::Mul(x3, x2, x, count)` → x³
   - `AscendC::Muls(x3, x3, 0.044715f, count)` → 0.044715 * x³
   - `AscendC::Add(y, x, x3, count)` → x + 0.044715*x³
   - `AscendC::Muls(y, y, 0.79788456f, count)` → √(2/π) * (x + 0.044715*x³)
   - `AscendC::Muls(expArg, y, 2.0f, count)` → 2y
   - `AscendC::Exp(expArg, expArg, count)` → exp(2y)
   - `AscendC::Adds(den, expArg, 1.0f, count)` → exp(2y) + 1
   - `AscendC::Reciprocal(recip, den, count)` → 1 / (exp(2y)+1)
   - `AscendC::Muls(tanhVal, recip, -2.0f, count)` → -2 / (exp(2y)+1)
   - `AscendC::Adds(tanhVal, tanhVal, 1.0f, count)` → 1 - 2/(exp(2y)+1) = tanh(y)
   - `AscendC::Adds(tanhVal, tanhVal, 1.0f, count)` → 1 + tanh(y)
   - `AscendC::Muls(result, x, 0.5f, count)` → 0.5 * x
   - `AscendC::Mul(result, result, tanhVal, count)` → 0.5 * x * (1 + tanh(y)) = GELU_tanh(x)
5. **approximate="none" 模式计算链（使用 erf 多项式逼近）**：
   - `AscendC::Muls(z, x, 0.70710678f, count)` → x / √2
   - `AscendC::Abs(absZ, z, count)` → |z|
   - `AscendC::Muls(t, absZ, 0.3275911f, count)` → 0.3275911 * |z|
   - `AscendC::Adds(t, t, 1.0f, count)` → 1 + 0.3275911*|z|
   - `AscendC::Reciprocal(t, t, count)` → 1/(1+0.3275911*|z|) = t
   - Horner 法求多项式 p = t*(a1 + t*(a2 + t*(a3 + t*(a4 + a5*t)))):
     - `AscendC::Muls(p, t, 1.061405429f, count)` → a5*t
     - `AscendC::Adds(p, p, -1.453152027f, count)` → a4 + a5*t
     - `AscendC::Mul(p, p, t, count)` → t*(a4 + a5*t)
     - `AscendC::Adds(p, p, 1.421413741f, count)` → a3 + t*(a4 + a5*t)
     - `AscendC::Mul(p, p, t, count)` → t*(a3 + t*(a4 + a5*t))
     - `AscendC::Adds(p, p, -0.284496736f, count)` → a2 + t*(a3 + t*(a4 + a5*t))
     - `AscendC::Mul(p, p, t, count)` → t*(a2 + t*(a3 + t*(a4 + a5*t)))
     - `AscendC::Adds(p, p, 0.254829592f, count)` → a1 + t*(a2 + t*(a3 + t*(a4 + a5*t)))
     - `AscendC::Mul(p, p, t, count)` → p = t*(a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))))
   - `AscendC::Mul(z2, z, z, count)` → z²
   - `AscendC::Muls(z2, z2, -1.0f, count)` → -z²
   - `AscendC::Exp(z2, z2, count)` → exp(-z²)
   - `AscendC::Mul(p, p, z2, count)` → p * exp(-z²)
   - `AscendC::Muls(erfAbs, p, -1.0f, count)` → -p * exp(-z²)
   - `AscendC::Adds(erfAbs, erfAbs, 1.0f, count)` → 1 - p*exp(-z²) = erf(|z|)
   - 根据 z 的符号取 erf(z)（不需分支，直接用 Abs 后的原始 z 判断：AscendC::Exp 前的 z2 保留了 z 的信息）
   - 更简单的无分支方式：使用 tanh 近似公式的精确修正
6. 数据搬运统一使用 DataCopyPad API；禁止使用 reinterpret_cast 转换 GlobalTensor 指针
7. 禁止使用 AscendC::MicroAPI、Membase；禁止使用 C++ STL（std::vector、std::string、<algorithm> 等）
8. 临时缓冲区必须使用独立的 TBuf 对象分配，**禁止使用 TQue::AllocTensor() 从数据队列分配临时缓冲**
9. approximate 属性值（"none"/"tanh"）通过 TilingData 传入 kernel，kernel 用 `if constexpr` 在编译期选择计算分支

【Tiling 性能优化 — 影响性能得分的关键，必须遵守】
1. **禁止使用固定 tile size**（如 FIXED_TILE_ELEMS = 2048）。必须参考 add 算子的 tiling 方式：
   - 通过 `platform_ascendc::PlatformAscendCManager::GetInstance()` 获取 `ubSize`
   - 根据 UB 大小动态计算每 tile 的元素数
2. tile size 计算方式（单位：字节）：
   - 每 tile 的缓冲区总需求 = inQueue(1个buffer) + outQueue(1个buffer) + TBufs(混合精度时的额外 float 缓冲 + 多项式中间变量)
   - `tileSizeBytes = ubSize / PIPELINE_DEPTH / (2 + 混合精度TBuf数量)`
   - `elementNumPerTile = tileSizeBytes / sizeof(T)`

【自验证要求 — 提交前必须执行】
完成开发后，请使用 golden.py 生成测试数据，分别对 float16、float32、bfloat16 三种 dtype，以及两种 approximate 模式（"none" 和 "tanh"）运行算子并验证输出精度。如果发现某种 dtype 或模式结果异常（如全零、MERE=1.0），请排查 Host 分发逻辑和 kernel 模板特化后再提交。

请你完成整套开发任务，中间过程自己运行，不要询问我进行下一步。

## Expected Output

output/ 目录应只包含以下文件：
1. csrc/ops/gelu/op_kernel/gelu_kernel.cpp — Ascend C Kernel，支持精确（erf）和 tanh 近似两种模式，遵循经典 AscendC API 规范
2. csrc/ops/gelu/op_kernel/gelu_launch.h — launch 函数声明
3. csrc/ops/gelu/op_plugin/gelu_plugin.cpp — PyTorch TORCH_LIBRARY 绑定，暴露 approximate 参数
4. csrc/ops/gelu/CMakeLists.txt — 算子编译配置
5. cann_bench/__init__.py — 暴露 cann_bench.gelu(x, approximate="none") 接口

不应包含 extension.cpp、顶层 CMakeLists.txt、setup.py、cmake/ 等基础设施文件。

## Expectations
