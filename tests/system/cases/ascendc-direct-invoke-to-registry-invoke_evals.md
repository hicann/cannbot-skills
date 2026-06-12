---
skill_name: ascendc-direct-invoke-to-registry-invoke
eval_mode: text
---
# Case 1: 七阶段工作流知识验证

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

ascendc-direct-invoke-to-registry-invoke 这个 skill 的七阶段工作流是什么？每个阶段的产物是什么？为什么不能跳步？

## Expected Output

回复应完整列出七阶段工作流及各阶段产物：
- 阶段 1：读取原 .asc 文件，产物为 OpDef 契约表 + kernel 清单（Input/Output 名称、DataType 数组、Format 数组、Attr 默认值、SOC 系列名、schMode 数）
- 阶段 2：拆分 host/kernel，产物为 op_host/ 四件套（_def.cpp、_tiling.cpp、_infershape.cpp、_tiling.h）+ op_kernel/ 两件套（.h、.cpp）+ config 文件
- 阶段 3：生成 <op>.json，产物为 msopgen 原型 JSON 文件（顶层数组、op 大驼峰、dtype 语义名）
- 阶段 4：生成验证工程，执行 msopgen gen -i <op>.json -c ai_core-<soc> -lan cpp -out <VerifyProjectDir>，产物为验证工程目录
- 阶段 5：编译构建，执行 bash build.sh，产物为 build_out/custom_opp_*.run
- 阶段 6：安装算子包，直接执行 .run 文件，产物为安装日志
- 阶段 7：ACLNN 二进制一致性验证，产物为 example/ 工程 + aclnn 输出 bin + byte-level compare 结果
应说明任一阶段产物缺失即交付失败，不得跳步或合并阶段

## Expectations
- [contains] 七阶段
- [contains] OpDef
- [contains] msopgen
- [contains] custom_opp


---

# Case 2: host 四件套拆分规范

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我有一个 rms_norm.asc 直调工程，需要转成自定义算子工程。host 侧代码应该怎么拆分？具体拆成哪几个文件，每个文件负责什么？

## Expected Output

回复应基于 skill 的 R-056 规则，说明 host 必须严格拆成 4 个文件，禁止合并为单个 .cpp：
- `<op>_tiling.h`：定义 TilingData（使用 BEGIN_TILING_DATA_DEF 宏 + END_TILING_DATA_DEF;），REGISTER_TILING_DATA_CLASS 注册。SDK 官方 tiling 结构（如 RmsNormTiling）用 TILING_DATA_FIELD_DEF_STRUCT 直接嵌入，不得扁平化
- `<op>_tiling.cpp`：实现 TilingFunc 并 IMPL_OP_OPTILING 注册。TilingFunc 五板斧必须齐全：SaveToBuffer + SetDataSize + SetBlockDim + SetTilingKey + GetWorkspaceSizes。host API 只从 tiling/tiling_api.h 获取
- `<op>_def.cpp`：实现 class RmsNorm : public OpDef，包含 Input/Output 定义（DataType 和 Format 数组）、AICore().AddConfig("ascend910b", cfg)、OP_ADD(RmsNorm)
- `<op>_infershape.cpp`：实现 InferShape/InferDataType 并 IMPL_OP_INFERSHAPE 注册。注意使用 gert::InferShapeContext 专用 API，不得混用 TilingContext 的 GetStorageShape

## Expectations
- [contains] tiling.h
- [contains] def.cpp
- [contains] infershape.cpp
- [contains] TilingData
- [contains] OpDef


---

# Case 3: kernel 侧形式适配规则

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

把 kernel 直调代码迁移到自定义算子工程时，kernel 侧允许做哪些修改？有哪些形式适配规则？计算逻辑可以改吗？

## Expected Output

回复应基于 skill 的 R-054 零号原则和 7 类形式适配规则：
- 零号原则：kernel 计算逻辑只搬不改，除 7 类形式适配外不改写计算顺序、API 选择或数学表达
- 7 类形式适配：
  1. kernel 入口签名固定为 extern "C" __global__ __aicore__ void <op>(...)
  2. 函数体第一行 GET_TILING_DATA(tilingData, tiling)
  3. TILING_KEY_IS 使用数字字面量，不用 constexpr 命名常量
  4. device 端禁用 STL（std::min/std::max/std::swap/algorithm/vector 等）
  5. 内建函数加 AscendC:: 前缀（GetBlockIdx/GetBlockNum/PipeBarrier）
  6. C++ 类型用实现名：bfloat16_t（不是 bfloat16）、half
  7. Init 签名模板化接收 TilingData
- 禁止：不得新增辅助函数/lambda、不得改算术表达式（如把 1.0f/sqrt(x) 换成 rsqrt(x)）、不得保留 main/ReadConfig/KernelCall

## Expectations
- [contains] 只搬不改
- [contains] GET_TILING_DATA
- [contains] TILING_KEY_IS
- [contains] AscendC::
- [contains] bfloat16_t


---

# Case 4: msopgen gen 命令与 -lan cpp 参数

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 150000
- Max Tokens (glm-5): 135000
- Ascend Platform: A2

## Prompt

在阶段 4 中，msopgen gen 命令的完整格式是什么？为什么必须带 -lan cpp 参数？如果不带会怎样？生成后需要校验什么？

## Expected Output

回复应基于 skill 的 R-053 规则：
- 完整命令格式：`msopgen gen -i <UserOutputDir>/<op>.json -c ai_core-<soc> -lan cpp -out <VerifyProjectDir>`
- 必须带 -lan cpp 的原因：默认 -lan py 会生成旧版 TBE DSL 结构（tbe/impl/op_info_cfg/*.py），而不是 C++ 的 op_host/ + op_kernel/ 结构
- 生成后结构校验：必须确认生成了 op_host/ + op_kernel/，且没有 tbe/、impl/、op_info_cfg/ 目录
- 后续操作：删除 msopgen 生成的合并 host 文件 <VerifyProjectDir>/op_host/<op>.cpp，再把 UserOutputDir 的 op_host/ 和 op_kernel/ 同步覆盖到验证工程

## Expectations
- [contains] -lan cpp
- [contains] msopgen gen
- [contains] TBE
- [contains] op_host


---

# Case 5: OpDef 契约表与 _def.cpp 规范

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

OpDef 契约表包含哪些字段？_def.cpp 文件应该怎么写？有哪些必须遵守的规则？如果 _def.cpp 写成空壳会怎样？

## Expected Output

回复应基于 skill 的 R-027 和 R-028 规则：
- OpDef 契约表必填字段：Input/Output 名称、param_type、完整 DataType[] 数组、Format[] 数组、Attr 具体 default_value（不得 TODO）、SOC 系列名、schMode 数 N
- _def.cpp 九点自检：class <Op> : public OpDef、构造函数大驼峰、.Input() 计数≥1、.Output() 计数≥1、每个 Input/Output 后跟 .DataType({ge::DT_xxx})（大括号内至少一个具体类型）、.Format({ge::FORMAT_ND})（非空）、DataType 和 Format 数组长度 == schMode 数 N、AICore().AddConfig("ascend910b", cfg) + OpAICoreConfig 六 flag + ExtendCfgInfo("opFile.value","<op>")、末尾 OP_ADD(<Op>)
- 空壳后果：编译能过但运行时 aclnn 报 NotRegistered / GetOpInfoFailed / op proto is empty，是最隐蔽最常见的失败模式
- SOC 用系列名（如 ascend910b），不带尾部型号数字（如 ascend910b4）

## Expectations
- [contains] OpDef
- [contains] OP_ADD
- [contains] DataType
- [contains] schMode


---

# Case 6: 双目录交付与环境降级

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

ascendc-direct-invoke-to-registry-invoke 这个 skill 要求交付哪两个目录？请详细说明它们之间的关系和所有约束条件（包括路径约束、命名约束、保护约束、交付约束）。如果 msopgen 或 CANN 环境不可用应该怎么降级处理？

## Expected Output

回复应基于 skill 的 R-058 规则，详细说明：
- 双目录定义：UserOutputDir（纯净成果：host 四件套 + kernel 两件套 + config + <op>.json）和 VerifyProjectDir（编译证据链：msopgen 生成的工程 + build_out/custom_opp_*.run + example/ 验证代码）
- 路径约束：两者绝对路径不同、不互为父子、VerifyProjectDir 不在 /tmp 或 /var/tmp、VerifyProjectDir 与 UserOutputDir 同级并列
- 命名约束：目录名带时间戳防覆盖
- 保护约束：验证目录在任何阶段都不得 rm -rf，需要重跑时用 mv 备份
- 交付约束：最终交付件清单必须同时包含两个目录的绝对路径
- 环境不可用降级：Windows 或未安装 CANN/msopgen 时，按 build-verify.md § D 输出人工验证手册，明确说明本次未完成编译、安装和二进制一致性验证，禁止伪装执行

## Expectations
- [contains] UserOutputDir
- [contains] VerifyProjectDir
- [contains] rm -rf
- [contains] 降级


---

# Case 7: ACLNN 二进制一致性验证

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 150000
- Max Tokens (glm-5): 135000
- Ascend Platform: A2

## Prompt

阶段 7 的 ACLNN 二进制一致性验证具体怎么做？验证的输入输出从哪来？可以用 rtol/atol 容差吗？

## Expected Output

回复应基于 skill 的 R-060 规则：
- 在 VerifyProjectDir/example/ 下生成 main.cpp 和 CMakeLists.txt
- 把原直调工程的 input/ 和 output/ 复制到 example/input/ 和 example/output/，输入/输出维度、dtype、epsilon、对齐规则等只能从原脚本或配置抽取，不能猜测
- main.cpp 构造与原直调完全一致的 ACLNN 输入参数，接口按算子名拼接为 aclnn<Op>GetWorkspaceSize 和 aclnn<Op>
- 固定编译运行流程：cd example && mkdir -p build && cd build && cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && make && cd bin && ./opapi_test
- ACLNN 输出写入 example/aclnn_output/，逐文件与原直调输出做 cmp -s 或 SHA256 比较
- 禁止使用 rtol/atol 容差放行，必须 byte-level identical，任一字节不同即验证失败

## Expectations
- [contains] aclnn
- [contains] byte-level
- [contains] cmp
- [contains] example


---

# Case 8: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Distractor skills: ascendc-registry-invoke-template;ascendc-direct-invoke-template;ascendc-registry-invoke-to-direct-invoke;ascendc-api-best-practices
- Ascend Platform: A2

## Prompt

我有一个 RmsNorm 的 kernel 直调工程（.asc 文件），想把它改造成 CANN 标准的自定义算子工程，能通过 ACLNN 接口调用。请先告诉我这个转换的完整流程和关键步骤，不需要立即执行。

## Expected Output

回复应正确激活 ascendc-direct-invoke-to-registry-invoke skill，基于其七阶段工作流给出概述：
- 先读取原 .asc 文件，填写 OpDef 契约表（Input/Output、DataType、Format、Attr、SOC、schMode）
- 拆分 host 四件套（_def.cpp、_tiling.cpp、_infershape.cpp、_tiling.h）和 kernel 两件套（.h、.cpp）
- 生成 <op>.json 并用 msopgen gen -lan cpp 创建验证工程
- 编译构建（bash build.sh）、安装 .run、ACLNN 二进制一致性验证
即使在 ascendc-registry-invoke-template、ascendc-direct-invoke-template 等相似 skill 共存的环境下，也应正确选择 ascendc-direct-invoke-to-registry-invoke。

## Expectations
- [contains] msopgen
- [contains] ACLNN

- [skill_activated] ascendc-direct-invoke-to-registry-invoke

---

# Case 9: 常见编译错误与修复

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

在 kernel 直调转自定义算子的过程中，常见的编译错误有哪些？分别对应什么规则和修复方法？

## Expected Output

回复应基于 skill 的 verification-checklist.md 常见红灯表，列出至少 5 个典型错误：
- adv_api/... No such file：host 侧盲目照搬 .asc 的 SDK 私有 include，改为 #include "tiling/tiling_api.h"（R-015）
- double free detected：TILING_DATA_FIELD_DEF_STRUCT 嵌套 SDK tiling 时用了 local 变量 + 拷贝，改为直接把 tiling.fieldName 作 OUT 引用传给 AscendC::GetXxxTilingInfo（R-009）
- TILING_KEY_IS 报 Var can not be processed as numeric：用了 constexpr 命名常量，改为数字字面量（R-020）
- GetStorageShape / SetOutputShape 不存在：InferShape 与 Tiling 混用 shape API，改用 gert::InferShapeContext 专用 API（R-038）
- bfloat16 unknown type：kernel 侧 BF16 类型名应统一用 bfloat16_t（R-023）
- std::min no member：device 端禁用 STL，改用三元运算符（R-021）
- msopgen 生成 tbe/impl/ 而非 op_host/op_kernel/：msopgen gen 未带 -lan cpp（R-053）

## Expectations
- [contains] adv_api
- [contains] double free
- [contains] TILING_KEY_IS
- [contains] bfloat16_t


---

# Case 10: 使用边界-不适用于从零开发

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我想用 ascendc-direct-invoke-to-registry-invoke 这个 skill 从零开发一个新的自定义算子，我没有已有的直调代码。请详细说明：这个 skill 的使用边界是什么？它需要哪些典型输入材料？如果没有这些材料应该用哪些其他 skill 替代？

## Expected Output

回复应明确说明 ascendc-direct-invoke-to-registry-invoke skill 不适用于从零开发新算子。应说明：
- 本 skill 的使用边界：只用于把已有 kernel 直调工程（.asc 文件）迁移为 CANN 标准 msOpGen 自定义算子工程
- 典型输入要求：必须有原 .asc 文件、原 host 调用代码、原输入输出 bin 和生成数据脚本
- 如果没有这些材料，应建议用户使用 ascendc-direct-invoke-template（从零创建直调工程）或 ascendc-registry-invoke-template（从零创建自定义算子工程）等 skill
- 不应在没有可参考的直调代码时直接生成算子代码

## Expectations
- [contains] 不用于
- [contains] 直调

- [not_contains] BEGIN_TILING_DATA_DEF
