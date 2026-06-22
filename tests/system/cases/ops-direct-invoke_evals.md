---
team_name: ops-direct-invoke
eval_mode: text
---
# Case 1: 基本算子开发流程问答

## Config
- Max Tokens: 200000
- Timeout: 900
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

# Case 2: 生成一个简单的 Add 向量逐元素加法算子

## Config
- Eval Mode: file_based
- Max Tokens: 5000000
- Timeout: 18000
- Disabled: true
- Ascend Platform: A5

## Prompt

请使用 ops-direct-invoke 团队的工作流，生成一个名为 add_vector 的 Ascend C Kernel 直调算子，实现两个 float16 向量的逐元素加法（z = x + y）。请先生成算子工程目录结构，然后实现 kernel 代码和辅助脚本。

算子规格：
- 算子名称：add_vector
- 数学公式：z[i] = x[i] + y[i]（逐元素加法）
- 输入：x fp16[256], y fp16[256]
- 输出：z fp16[256]
- 目标芯片：Ascend950PR（DAV_3510）
- 编译架构：dav-3510（CMakeLists.txt 中配置 --npu-arch=dav-3510）
- 数据搬运：GM→UB→计算→UB→GM
- 核函数入口：纯向量算子使用 `__global__ __vector__`

请输出完整的算子文件到沙箱中。

## Expected Output

生成的算子工程应包含以下完整内容：

1. **kernel 实现文件**（.asc）：包含完整的 Ascend C 算子逻辑，使用正确的 API（DataCopyPad、Add、LocalTensor、GlobalTensor 等），有正确的核函数入口（纯向量算子使用 `__global__ __vector__`），包含 tiling 参数获取（GetBlockNum、GetBlockIdx），CMakeLists.txt 中 `--npu-arch` 配置为 dav-3510（Ascend950PR）

2. **CMakeLists.txt**：正确配置 Ascend C 算子编译选项，包括正确的 CANN 路径引用和编译目标（--npu-arch=dav-3510）

3. **数据生成脚本**（gen_data.py 或类似）：生成测试用的 float16 输入数据

4. **运行/编译脚本**（run.sh 或类似）：包含编译和运行命令，正确引用核函数

5. **编译与运行验证**：agent 须在沙箱中执行编译和运行操作（通过 bash 工具调用 run.sh 或手动编译运行），确保生成的算子工程可以正常编译链接、在 NPU 上正确执行，且精度验证（verify_result.py）结果为 PASS

整体实现应结构清晰，代码符合 Ascend C 编程规范，API 使用正确（不能编造不存在的 API）。

## Expectations

- [file_exists] operators/add_vector/op_host/add_vector.asc
- [file_list] operators/add_vector/*.sh
- [file_list] operators/add_vector/CMakeLists.txt
- [file_contains] operators/add_vector/CMakeLists.txt : "dav-3510"
- [not-contain] "dav-2201"
- [file_contains] operators/add_vector/op_host/add_vector.asc : "__global__";"add_vector_kernel";"LocalTensor";"DataCopyPad"
