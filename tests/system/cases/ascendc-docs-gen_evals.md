---
skill_name: ascendc-docs-gen
eval_mode: file_based
---
# Case 1: 为 Add 算子生成需求分析文档

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我需要开发一个 Add 算子，请帮我生成需求分析文档。算子规格如下：

算子名称：Add
数学公式：z = x + alpha * y（逐元素加法）
输入：x fp16 [1024, 4096], y fp16 [1024, 4096]
输出：z fp16 [1024, 4096]
目标芯片：Ascend910B
精度要求：fp16 双千分之一
调用方式：ACLNN 调用

请生成 REQUIREMENTS.md 文件。

## Expected Output

生成的 REQUIREMENTS.md 应严格遵循需求分析模板结构，包含以下关键章节：
- 修订记录表格（版本、修订内容、修订时间、修订人）
- §1 需求背景：需求来源、基线对齐（框架API/论文公式/用户给定公式的勾选框）
- §2 运行环境：服务器型号、芯片号、编译宏架构（DAV_*）
- §3 调用方式表格：列出 ACLNN 调用、torch_npu 单算子、torch.compile 入图、GE 图模式等调用方式的支持情况
- §4 算子规格：基本信息（算子名称、数学公式）、输入输出规格（x/y/z 的 shape 和 dtype）、数据类型支持（fp16/fp32/bfloat16/int8 勾选框）、精度要求
- §5 ACLNN API 接口定义：两段式接口声明（aclnnAddGetWorkspaceSize + aclnnAdd）、参数说明表、约束与限制（类型推导规则、shape 约束、广播规则）
- §8 约束与要求：计算约束

## Expectations
- [contains] ACLNN
- [contains] 修订记录
- [contains] 算子规格

- [file_exists] REQUIREMENTS.md

---

# Case 2: 为 Softmax 算子生成详细设计文档

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

我需要开发一个 Softmax 算子，请帮我生成详细设计文档。算子规格如下：

算子名称：Softmax
数学公式：softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
输入：x fp16 [1024, 4096]
输出：y fp16 [1024, 4096]
目标芯片：Ascend910B，架构 arch22
算子类别：Reduction

请生成 DESIGN.md 文件。

## Expected Output

生成的 DESIGN.md 应严格遵循详细设计模板结构，包含以下关键章节：
- §1 概述：基本信息表（算子名称、算子类别 Reduction、支持数据类型、目标芯片 Ascend910B、目标架构 arch22）、算子功能描述、数学公式
- §2 架构设计：逻辑视图（op_api/op_host/op_kernel/op_graph 四个模块职责表和依赖关系）、开发视图（目录结构树）、运行视图（数据流 GM→UB→GM、执行流程 aclnnGetWorkspaceSize→aclnn 执行）
- §3 实现方案：模板划分总览（TilingKey 机制、模板参数定义表、模板划分表）、TilingData 结构体定义、每个模板包含触发条件、Host 侧 Tiling 代码、Kernel 侧模板实例化代码、API 映射表（6列：计算步骤/Ascend C API/参数签名/平台验证/约束说明/替代方案）、API 验证记录表、数据流设计、内存管理表、UB 容量验证（DAV_2201 184KB 限制）
- §4 性能优化：并行策略、流水线设计
- §5 风险评估：API 风险、精度风险、应对措施
- §6 交付件清单
- §7 迭代规划表（迭代一/二/三的目标和代码开发/UT开发/ST用例）

## Expectations
- [contains] op_api
- [contains] op_host
- [contains] op_kernel
- [contains] TilingData
- [contains] UB

- [file_exists] DESIGN.md

---

# Case 3: 为 Add 算子生成迭代执行计划

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 150000
- Max Tokens (glm-5): 135000
- Ascend Platform: A2

## Prompt

我需要为 Add 算子制定迭代开发计划，请帮我生成迭代执行计划文档。算子规格如下：

算子名称：Add
支持数据类型：fp16, fp32
目标芯片：Ascend910B
模板划分：模板一（fp16 + small_shape）、模板二（fp32 + small_shape）、模板三（fp16 + large_shape 切分）

请生成 PLAN.md 文件。

## Expected Output

生成的 PLAN.md 应严格遵循迭代计划模板结构，包含以下关键章节：
- 迭代一穿刺列表：表格形式（任务类型/TilingKey/Dtype/Memory Strategy），选择主要 dtype（fp16），包含主线 + 多个穿刺任务，注明并行要求（主线 + 穿刺必须同一次响应发起）
- 迭代二整合目标：列出需要整合的 TilingKey
- 迭代二穿刺列表：表格形式验证迭代三任务，包含验证目标
- 迭代三全覆盖目标：覆盖全 dtype（fp16, fp32）、边界 case、广播用例
- 穿刺结果判定表：成功/部分成功/失败的处理方式

## Expectations
- [contains] 迭代一
- [contains] 迭代二
- [contains] 迭代三
- [contains] TilingKey
- [contains] 穿刺

- [file_exists] PLAN.md

---

# Case 4: 为 Add 算子生成 aclnnAPI 接口文档

## Config
- Max Tokens: 250000
- Max Tokens (deepseek-v4-flash): 220000
- Max Tokens (glm-5): 200000
- Ascend Platform: A2

## Prompt

我需要为 Add 算子编写 aclnnAPI 接口文档，请帮我生成。算子规格如下：

算子名称：Add
数学公式：out = self + alpha * other
输入：self（aclTensor*）, other（aclTensor*）, alpha（aclScalar*）
输出：out（aclTensor*）
支持数据类型：FLOAT, FLOAT16, INT32, INT64, BFLOAT16 等
支持产品：Ascend 950PR/950DT, Atlas A2, Atlas A3, Atlas 训练系列
约束：self 和 other 需满足 broadcast 关系，数据类型需满足互推导关系

请生成 aclnnAdd.md 文件。

## Expected Output

生成的 aclnnAdd.md 应严格遵循 aclnnAPI 文档模板结构，包含以下关键章节：
- 产品支持情况表：按固定产品顺序列出（Ascend 950PR/950DT、Atlas A3、Atlas A2、Atlas 200I/500 A2、Atlas 推理系列、Atlas 训练系列），用 √/× 标注支持情况
- 功能说明：接口功能描述、计算公式（out = self + alpha * other）
- 函数原型：两段式接口（aclnnAddGetWorkspaceSize 和 aclnnAdd），每行一个参数，参数对齐
- aclnnAddGetWorkspaceSize 参数说明：8 字段参数表（参数名/输入输出/描述/使用说明/数据类型/数据格式/维度shape/非连续Tensor），使用说明中包含空 Tensor 支持度和数据类型推导规则引用
- 返回值表：包含错误码（如 ACLNN_ERR_PARAM_NULLPTR 161001、ACLNN_ERR_PARAM_INVALID 161002）
- aclnnAdd 第二段接口参数说明：workspace/workspaceSize/executor/stream
- 约束说明：确定性说明（三选一格式）
- 调用示例：标注编译运行参考路径

## Expectations
- [contains] GetWorkspaceSize
- [contains] 产品支持
- [contains] 确定性
- [contains] 161001

- [file_exists] aclnnAdd.md

---

# Case 5: 为 Add 算子生成算子 README

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我需要为 Add 算子编写开源仓的 README 文档，请帮我生成。算子规格如下：

算子名称：Add
数学公式：y = x1 + alpha * x2
输入：x1, x2（支持 FLOAT, FLOAT16, INT32, BFLOAT16 等）
输出：y
可选属性：alpha（默认值 1.0）
支持产品：Ascend 950PR/950DT, Atlas A2, Atlas A3, Atlas 训练系列
调用方式：aclnn 调用 + 图模式调用
约束：无特殊约束

请生成 README.md 文件。

## Expected Output

生成的 README.md 应严格遵循算子 README 模板结构，包含以下关键章节：
- 产品支持情况表：按固定产品顺序列出（含麒麟芯片行），用 √/× 标注
- 功能说明：算子功能一句话描述、计算公式（y = x1 + alpha * x2）
- 参数说明表：5 列表格（参数名/输入输出属性/描述/数据类型/数据格式），包含 x1、x2、alpha（可选属性）、y 四个参数，芯片差异说明（如 Atlas 训练系列不支持 BFLOAT16）
- 约束说明：无特殊约束时写"无"
- 调用说明：表格形式列出 aclnn 调用和图模式调用，链接到 examples 目录的样例代码
- 参考资源（可选）：链接到算子设计文档

## Expectations
- [contains] 产品支持
- [contains] 参数说明
- [contains] 调用说明
- [contains] alpha

- [file_exists] README.md

---

# Case 6: 信息不足时主动追问

## Config
- Eval Mode: text
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

帮我写个算子文档

## Expected Output

回复应主动追问关键信息，而不是直接生成文档。应至少询问以下信息中的一项或多项：算子名称、数学公式/功能描述、输入输出规格（shape/dtype）、目标芯片/架构、需要生成哪种类型的文档（需求分析/详细设计/迭代计划/aclnnAPI/README）。不应在缺乏算子规格的情况下直接输出完整的文档模板内容。

## Expectations

---

# Case 7: 正向看护-多 skill 环境下正确触发目标 skill

## Config
- Eval Mode: text
- Max Tokens: 120000
- Distractor skills: ascendc-st-design;ascendc-tiling-design;ascendc-direct-invoke-template;ascendc-api-best-practices
- Ascend Platform: A2

## Prompt

我需要为 Softmax 算子编写 aclnnAPI 接口文档和算子 README，应该参考什么模板和规范？

## Expected Output

回复应正确激活 ascendc-docs-gen skill，基于其提供的 aclnnAPI 文档模板和算子 README 模板给出指导。应说明 aclnnAPI 文档需要包含产品支持情况、两段式函数原型、8 字段参数说明表、返回值错误码表、约束说明（含确定性说明）等关键结构。应说明算子 README 需要包含产品支持、功能说明、参数说明表、调用说明等。即使在 ascendc-st-design、ascendc-tiling-design 等相似 skill 共存的环境下，也应正确选择 ascendc-docs-gen。

## Expectations
- [skill_activated] ascendc-docs-gen

---

# Case 8: 文档类型体系和命名规范知识验证

## Config
- Eval Mode: text
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

ascendc-docs-gen 支持哪些文档类型？每种文档的命名规范是什么？它们之间有什么关系？

## Expected Output

回复应完整列出 5 种文档类型及其命名规范：
- 需求分析文档：REQUIREMENTS.md
- 详细设计文档：DESIGN.md
- 迭代执行计划：PLAN.md
- aclnnAPI 接口文档：aclnn{OperatorName}.md（如 aclnnAdd.md）
- 算子 README：README.md

应说明文档间的依赖关系：需求分析确认后产出详细设计，详细设计产出迭代计划；aclnnAPI 文档的数据来源于需求文档的算子规格、API 定义和约束部分；算子 README 的数据来源于需求文档、设计文档和代码。应提及文档存放位置（docs/ 目录或算子根目录）。

## Expectations
- [contains] REQUIREMENTS.md
- [contains] DESIGN.md
- [contains] PLAN.md
- [contains] aclnn
- [contains] README.md
