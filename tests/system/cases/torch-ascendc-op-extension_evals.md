---
skill_name: torch-ascendc-op-extension
---

# Case 1: Ascend C 算子 TORCH_LIBRARY 对接

## Config
- Max Tokens: 150000
- Ascend Platform: A2

## Prompt

我有一个可编译运行的 Ascend C 直调工程，位于 /workspace/add_custom/，现在想通过 TORCH_LIBRARY 将其对接到 PyTorch，实现通过 torch.ops.npu.add_custom() 来调用。请按 torch-ascendc-op-extension 的改造步骤进行对接。

## Expected Output

回复应按 Step 0 到 Step 6 的步骤逐步执行：Step 0 将 kernel 与 host 拆分（如源工程为单文件 .asc），提取 op_kernel/xxx_tiling.h、op_kernel/xxx_kernel.asc、改造原 .asc 到 op_host/；Step 1 创建 op_extension/ops.h 函数声明；Step 2 创建 op_extension/xxx_torch.cpp（使用 stream(true) 清 queue 模式，禁止 stream(false) 和 zeros_like）；Step 3 创建 op_extension/register.cpp（包含 TORCH_LIBRARY_FRAGMENT、PrivateUse1 实现绑定、Meta 后端注册三部分）；Step 4 更新 CMakeLists.txt 为双目标结构；Step 5 编译验证；Step 6 进行 Python 测试。产出应包括 .so 动态库和完整的目录结构。

## Expectations
- [contains] TORCH_LIBRARY
- [contains] torch.ops.npu
- [contains] stream(true)
- [contains] register.cpp
- [contains] PrivateUse1

---

# Case 2: 前置条件与适用边界

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

torch-ascendc-op-extension 适用的前置条件是什么？什么情况下不应该使用这个 skill？和 ascendc-direct-invoke-template、ascendc-registry-invoke-to-direct-invoke 有什么区别？

## Expected Output

回复应说明前置条件为：已有可编译运行的 Ascend C <<<>>> 直调工程，环境已安装 torch、torch_npu，CANN 环境已配置（ASCEND_HOME_PATH 已设置）。不适用场景为：从零创建工程（应使用 ascendc-direct-invoke-template）以及注册调用转直调（应使用 ascendc-registry-invoke-to-direct-invoke）。即使没有 PyTorch 对接需求，直调工程本身也能独立运行，本 skill 仅负责添加 PyTorch 接入层。应强调 TORCH_LIBRARY 方式使用 CMake 多文件架构，kernel 作为普通 C 函数调用，必须手动管理 stream 同步。

## Expectations
- [contains] 直调工程
- [contains] ASCEND_HOME_PATH
- [contains] ascendc-direct-invoke-template
- [contains] .so
