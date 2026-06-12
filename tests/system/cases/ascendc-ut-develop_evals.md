---
skill_name: ascendc-ut-develop
eval_mode: text
---
# Case 1: 为 Add 算子开发 op_api 层 UT

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-ut-develop 技能为 ops-math 仓的 Add 算子开发 op_api 层单元测试的完整工作流程。请依次介绍以下 6 个方面：
1. 入口参数有哪些（算子名称、仓库类型、芯片架构、测试模块、使用模式）
2. 强制前置步骤（问卷确认参数、创建 TODO.md、创建临时目录）
3. 主流程 Step 1 到 Step 5 各步骤概述
4. op_api 层 UT 使用的核心组件（如 TensorDesc、ScalarDesc、OP_API_UT 宏）
5. 用例的 TDD 编写顺序（异常用例、正常用例、边界用例）
6. 覆盖率目标（行覆盖率和函数覆盖率的具体数值）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应体现 ascendc-ut-develop skill 的核心工作流，包含以下要点：
- 首先确认入口参数：算子名称(add)、仓库类型(ops-math)、芯片架构(ascend910b/arch32)、测试模块(opapi)、使用模式(auto)
- 说明强制前置步骤：发送问卷确认参数、创建 TODO.md、创建 /tmp/cannbot_add/ 临时目录
- 说明主流程 Step 1-5：了解仓库信息（阅读基础知识文档、提取 SoC 列表、学习 build.sh 编译命令）→ 检查现有 UT 框架 → （跳过重构）→ 补全用例提升覆盖率 → 生成总结报告
- 说明 op_api 层 UT 的核心组件：TensorDesc、ScalarDesc、OP_API_UT 宏
- 说明 TDD 编写顺序：异常用例（nullptr/无效dtype/shape不匹配）→ 正常用例（各 dtype）→ 边界用例（空tensor/0维）
- 说明覆盖率目标：行覆盖率 ≥ 80%、函数覆盖率 ≥ 80%

## Expectations
- [contains] op_api
- [contains] 覆盖率
- [contains] build.sh
- [contains] 80%


---

# Case 2: 覆盖率提升分析与补测策略

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

我需要为 ops-nn 仓的 Softmax 算子提升 op_host 层 UT 覆盖率，目标芯片 ascend910b，自动模式。当前行覆盖率 65%，函数覆盖率 70%，需要提升到 80% 以上。

入口参数已确认：op_name=softmax, repo_type=ops-nn, soc_type=ascend910b, test_model=ophost, interactive_mode=auto。

请直接告诉我 Step 4 补全用例提升覆盖率的完整流程：如何获取覆盖率、如何分析未覆盖代码、如何设计补测用例。

## Expected Output

回复应基于 skill 的 Step 4 覆盖率提升流程，包含以下要点：
- 获取覆盖率：使用 `bash build.sh -u --ophost --ops=softmax --soc=ascend910b --cov` 编译并生成覆盖率
- 判断覆盖率类型：区分全局覆盖率和单算子覆盖率，全局覆盖率需用 `lcov --extract` 提取
- 分析未覆盖代码：使用 `lcov --list ops.info_filtered | grep ":0"` 获取未覆盖行
- 按缺口类型补充策略：异常分支（ACLNN_ERR_*/GRAPH_FAILED）、dtype 分支（分析 tiling 实现中的 dtype 分支逻辑）、边界条件（空 tensor、大 shape）
- op_host 层特有关注点：Tiling 测试（CompileInfo 类型检测、TilingContextPara）、InferShape 测试（NodeAttrs 配置）
- 完成检查标准：行覆盖率 ≥ 80%、函数覆盖率 ≥ 80%

## Expectations
- [contains] lcov
- [contains] 覆盖率
- [contains] 80%
- [contains] 补充


---

# Case 3: op_api 层 dtype 排列组合校验

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

我需要为 ops-transformer 仓的 FlashAttention 算子确保 op_api 层 UT 覆盖所有合法 dtype 组合，目标芯片全选，自动模式。听说有脚本可以自动提取 dtype 组合。

入口参数已确认：op_name=flash_attention, repo_type=ops-transformer, soc_type=all, test_model=opapi, interactive_mode=auto。

请直接告诉我 dtype 排列组合校验的完整流程：如何定位 def 文件、如何使用脚本提取 dtype 组合、校验步骤和异常组合构造方法。

## Expected Output

回复应基于 skill 的 Dtype 排列组合校验流程，包含以下要点：
- 定位算子定义文件：`find ${op_path}/op_host -name "*_def.cpp"`
- 使用脚本提取 dtype 组合：`python scripts/extract_dtype_combinations.py ${def_file}`，输出到 `/tmp/${op_name}_dtype_combinations.json`
- 5 步校验流程：脚本提取 → 补充用例 → 覆盖率分析（lcov 检查 dtype 分支无 `:0`）→ 实际测试（build.sh 全部 PASS）→ 异常校验（构造 3 种非法组合返回 ACLNN_ERR_PARAM_INVALID）
- 异常组合构造方法：跨位置组合、未定义 dtype、dtype 不一致
- 判定标准：5 步全部通过则 dtype 校验完整

## Expectations
- [contains] extract_dtype_combinations
- [contains] ACLNN_ERR_PARAM_INVALID
- [contains] dtype


---

# Case 4: 入口参数问卷与强制前置步骤

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我想给 ops-transformer 仓的 Mul 算子写 UT，芯片是 ascend950，测试 ophost 和 opapi 两个模块。请告诉我 skill 的入口参数有哪些，以及执行前必须完成哪些前置步骤。

## Expected Output

回复应准确列出 skill 的 5 个入口参数及其取值：
- op_name：算子名（mul），驼峰命名自动转换为下划线命名
- repo_type：仓库类型，枚举值 ops-math/ops-nn/ops-transformer/ops-cv/custom
- soc_type：芯片架构，枚举值 ascend310p(arch20)/ascend910b(arch32)/ascend910_93(arch32)/ascend950(arch35)
- test_model：测试模块，枚举值 opapi/ophost/opkernel，可多选
- interactive_mode：使用模式，auto（默认）或 interactive

说明强制前置步骤（不可跳过）：
1. 使用 question 工具发送问卷确认入口参数，推断选项加"【推荐】"标记
2. 使用 todowrite 工具创建 TODO.md（内容来自 assets/todo.json）
3. 创建 /tmp/cannbot_mul/ 临时目录，存储 params.json 和中间文件

## Expectations
- [contains] op_name
- [contains] repo_type
- [contains] soc_type
- [contains] test_model
- [contains] interactive_mode


---

# Case 5: UT 框架重构流程

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-ut-develop 技能中 Step 3（重构 UT 框架）的详细步骤。我特别想了解：备份旧代码怎么做、搭建新框架要注意什么、ops-transformer 仓的 CSV 格式整改怎么走、迁移旧用例有什么规则、子 Agent 完成后要检查什么。不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应基于 skill 的 Step 3 重构流程，包含以下要点：
- Step 3.1 备份旧框架代码：将 ut/${test_model} 下的 .cpp/.h 文件备份到 /tmp/cannbot_${op_name}/backup/，然后删除原文件
- Step 3.2 搭建新框架：使用 Task 工具调用子 Agent，仅迁移一条预期成功的旧用例验证框架，UT 代码放在算子目录下而非仓库根目录的 test 目录
- CSV 格式重构特殊路径：当 repo_type 为 ops-transformer 且用户提及 CSV 时，参考 ops-transformer.md 和 csv-refactor-workflow.md，使用 assets/csv-refactor/ 下的模板文件，查看详细规范 csv-format-spec.md
- Step 3.3 迁移所有旧用例：每 5 条一组确保编译运行通过后再迁移下一组，不得更改已迁移用例和公共代码
- 子 Agent 完成后必须检查：编译通过 + 运行测试全部 PASS + 用例数量与重构前一致

## Expectations
- [contains] 备份
- [contains] CSV
- [contains] 子 Agent
- [contains] 5 条


---

# Case 6: op_host 层 Tiling 和 InferShape 测试要点

## Config
- Max Tokens: 250000
- Max Tokens (deepseek-v4-flash): 300000
- Max Tokens (glm-5): 275000
- Ascend Platform: A2

## Prompt

请简洁介绍 ascendc-ut-develop 技能中 op_host 层 UT 测试返回 GRAPH_FAILED 的常见原因和解决方法。请依次简要说明以下 5 个要点（每个要点 1-2 句话即可，不要写大段代码示例）：
1. 最常见原因是什么（NodeAttrs 未配置）
2. 如何配置 NodeAttrs（InferShape 和 Tiling 分别用什么 API）
3. MatMul 系列算子有哪些常见属性
4. CompileInfo 类型检测方法（如何确定命名空间）
5. 如何确定需要哪些属性（查看什么宏、搜索什么函数调用）

## Expected Output

回复应基于 skill 的 op-host-ut-guide 知识，简要包含以下要点：
- 最常见原因：未配置 NodeAttrs。大部分算子的 InferShape/Tiling 函数会调用 context->GetAttr() 读取算子属性，未配置则返回 GRAPH_FAILED
- 解决方案：InferShape 测试使用 InferShapeContextFaker().NodeAttrs() 配置属性，Tiling 测试使用 TilingContextFaker().NodeAttrs() + NodeInputTd()
- MatMul 系列算子常见属性：adj_x1、adj_x2、offset_x、opImplMode（命名空间 Ops::NN）
- CompileInfo 类型检测：必须先检测 TilingParse<TYPE> 的类型（BroadcastCompileInfo 用 Ops::Base:: 命名空间，自定义 CompileInfo 用 optiling:: 命名空间）
- 如何确定需要哪些属性：查看 INFER_SHAPE_FUNC/TILING_FUNC 宏、搜索 GetAttr()/GetAttrValue() 调用

## Expectations
- [contains] NodeAttrs
- [contains] GRAPH_FAILED
- [contains] CompileInfo
- [contains] GetAttr


---

# Case 7: 正向看护-多 skill 环境下正确触发目标 skill

## Config
- Max Tokens: 200000
- Distractor skills: ascendc-st-design;ascendc-code-review;ascendc-whitebox-design;ascendc-api-best-practices
- Ascend Platform: A2

## Prompt

请介绍 ascendc-ut-develop 技能中补全用例提升覆盖率的完整方法。请依次介绍以下 5 个方面：
1. 如何获取覆盖率数据（具体编译命令，如 build.sh 的 --cov 参数）
2. 如何分析未覆盖代码（具体工具，如 lcov 命令）
3. 按哪些缺口类型补充用例（如异常分支、dtype 分支、边界条件等）
4. op_api 层 UT 使用哪些核心组件（如 OP_API_UT 宏、TensorDesc 等）
5. 覆盖率目标的具体数值（行覆盖率和函数覆盖率分别要达到多少）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应正确激活 ascendc-ut-develop skill，基于其工作流给出指导。应说明：
- 使用 build.sh --cov 获取覆盖率数据
- 使用 lcov 分析未覆盖代码
- 按缺口类型（异常分支/dtype 分支/边界条件）补充用例
- op_api 层使用 OP_API_UT 宏、TensorDesc 等组件
- 覆盖率目标：行覆盖率和函数覆盖率均 ≥ 80%
即使在 ascendc-st-design、ascendc-code-review 等相似 skill 共存的环境下，也应正确选择 ascendc-ut-develop。

## Expectations
- [contains] 覆盖率
- [contains] 80%

- [skill_activated] ascendc-ut-develop

---

# Case 8: UT 开发总结报告结构

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-ut-develop 的 UT 开发总结报告（Step 5）包含哪些章节？每个章节需要记录什么信息？

## Expected Output

回复应基于 skill 的 assets/template_report.md 模板，列出报告的完整结构：
- 基本信息：算子名称、算子仓、算子类别、SoC 版本、支持的 dtype 和 format
- 层支持情况：op_host/op_api/op_kernel 各层是否支持
- UT 文件路径：展示 ut/ 目录下 op_api/op_host/op_kernel 的文件树
- 用例统计：按层分别统计异常用例、正常用例、边界用例的数量
- 覆盖率统计：覆盖率类型（全局/单算子）、行覆盖率/函数覆盖率/分支覆盖率的值和目标（≥80%）
- 遇到的问题及解决方案
- 编译命令记录
- 最终状态检查清单：编译通过、测试通过、覆盖率类型判断、覆盖率达标、TDD 流程遵循、报告生成

## Expectations
- [contains] 基本信息
- [contains] 用例统计
- [contains] 覆盖率
- [contains] 编译命令


---

# Case 9: 信息不足时主动追问

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

帮我写 UT

## Expected Output

回复应主动追问关键信息，而不是直接开始编写 UT。应至少询问以下信息中的一项或多项：算子名称、仓库类型（ops-math/ops-nn/ops-transformer/ops-cv/custom）、目标芯片架构（ascend310p/ascend910b/ascend910_93/ascend950）、测试模块（opapi/ophost/opkernel）。不应在缺乏算子规格的情况下直接生成测试代码。

## Expectations

- [not_contains] TEST_F
- [not_contains] EXPECT_EQ
- [not_contains] OP_API_UT

---

# Case 10: build.sh 编译命令与 --noexec 注意事项

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 ascendc-ut-develop 的工作流中，build.sh 的 --noexec 参数有什么作用？使用时需要注意什么？获取覆盖率的编译命令是什么？

## Expected Output

回复应基于 skill 的 Step 1.3 知识，包含以下要点：
- 不带 --noexec：编译 + 运行一步完成，推荐日常使用，能发现 CSV 格式等运行时错误
- 带 --noexec：仅编译不运行，仅用于快速验证语法错误
- 常见错误：加了 --noexec 后忘记运行测试，导致 CSV 格式错误未被发现
- 编译通过后必须再运行测试
- 获取覆盖率的命令：`bash build.sh -u --ops=${op_name} --soc=${soc_type} --cov`
- 按模块编译：`--opapi`、`--ophost`、`--opkernel`

## Expectations
- [contains] --noexec
- [contains] --cov
- [contains] build.sh

