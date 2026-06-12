---
skill_name: pypto-op-design
---
# Case 1: 简单 Vector 算子（Add）设计

## Config
- Eval Mode: file_based
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我想设计一个 Add 算子，算子规格如下：

算子名称：Add
数学公式：z = x + y（逐元素加法）
输入：x fp16 [1024, 4096], y fp16 [1024, 4096]
输出：z fp16 [1024, 4096]

请帮我生成 DESIGN.md

## Expected Output

生成的 DESIGN.md 应覆盖以下要点（§1-§6 共 6 个章节）：
- §1 计算图与精度路由：算子名称 Add、逐元素加法、API 调用序列（PyPTO 逐元素加法 API）、dtype 流转（fp16，无需 cast）
- §2 数据规格：kernel 签名含 x/y/z（fp16，[1024, 4096]），值类型分析
- §3 Tiling 策略：判断为 Vector 类型（不含 matmul），使用 set_vec_tile_shapes，含 tile 参数推导、UB 容量估算与展开检查
- §4 Loop 与数据流：完整伪代码，简单逐元素算子无动态轴、无需动态 pypto.loop，由 Tiling 按 tile 切分处理，含数据搬运与尾块处理
- §5 约束自检清单：约束检查表（API dtype、广播/Shape、值类型、Tiling 时序）与开放问题
- §6 验证方案：Golden 函数为简单的 torch 加法，覆盖典型 shape 配置，含精度容差
- 无残留 {placeholder} 占位符

## Expectations

- [file_exists] DESIGN.md

---

# Case 2: Cube 算子（Matmul）设计

## Config
- Eval Mode: file_based
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

我想设计一个 Matmul 算子，算子规格如下：

算子名称：Matmul
数学公式：C = A × B + bias
输入：A fp16 [512, 4096], B fp16 [4096, 2048], bias fp16 [512, 2048]
输出：C fp16 [512, 2048]
典型配置：
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 核心配置 | 性能 | P0 | 无 | A[512,4096], B[4096,2048], bias[512,2048] | C[512,2048] | 标准 matmul |
算法描述：带 bias 的矩阵乘法，需先执行矩阵乘 A×B，然后与 bias 相加

请帮我生成 DESIGN.md

## Expected Output

生成的 DESIGN.md 应覆盖以下要点（§1-§6 共 6 个章节）：
- §1 计算图与精度路由：算子名称 Matmul、C=A×B+bias、API 调用序列（matmul 相关 Cube API 与加法 API）、dtype 流转
- §2 数据规格：kernel 签名含 A/B/bias/C 四个 tensor，数据格式 fp16
- §3 Tiling 策略：判断为含 matmul 的 cube+vec 混合类型，matmul 阶段使用 set_cube_tile_shapes、bias 加法阶段按需使用 set_vec_tile_shapes，说明 tile 参数推导依据、UB 容量估算与展开检查
- §4 Loop 与数据流：完整伪代码，根据具体 shape 分析 Loop 结构与跨迭代依赖，数据搬运与尾块处理
- §5 约束自检清单：约束检查表（API dtype、广播/Shape、值类型、Tiling 时序）与开放问题
- §6 验证方案：Golden 函数使用 torch.matmul + bias 加法，覆盖典型配置，含精度容差
- 无残留 {placeholder} 占位符

## Expectations

- [file_exists] DESIGN.md

---

# Case 3: 信息不足时主动追问

## Config
- Ascend Platform: A2

## Prompt

帮我设计一个算子

## Expected Output

回复应询问算子名称、数学公式、输入输出规格等关键信息。不应在缺乏算子规格的情况下直接生成设计文档。应至少询问算子名称和数学公式。

## Expectations

- [not_contains] pypto.set_vec_tile_shapes(
- [not_contains] pypto.set_cube_tile_shapes(

---

# Case 4: 正向看护-多 Skill 环境下正确触发 PyPTO 算子设计

## Config
- Max Tokens: 300000
- Max Tokens (deepseek-v4-flash): 350000
- Max Tokens (glm-5): 320000
- Distractor skills: pypto-op-develop;pypto-api-explore;ascendc-st-design;ascendc-tiling-design
- Ascend Platform: A2

## Prompt

请使用 pypto-op-design skill 的算子设计工作流来设计一个 Softmax 算子，算子规格如下：

算子名称：Softmax
数学公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
输入：x fp16 [1024, 4096]
输出：y fp16 [1024, 4096]

请运行完整的算子设计工作流，生成 DESIGN.md，输出到当前目录。

## Expected Output

回复应调用 pypto-op-design skill 并执行算子设计工作流，生成 Softmax 算子的 DESIGN.md。内容应包含：§1 计算图与精度路由（exp、pypto.sum、除法 div 等操作的 PyPTO API 序列与 dtype 流转）、§2 数据规格、§3 Tiling 策略（Vector 类型）、§4 Loop 与数据流（含完整伪代码）、§5 约束自检清单、§6 验证方案。应覆盖 DESIGN.md 的 6 个章节，且最终输出 DESIGN.md 文件到当前目录。

## Expectations

- [skill_activated] pypto-op-design

---

# Case 5: 工作流知识验证

## Config
- Ascend Platform: A2

## Prompt

pypto-op-design 算子设计工作流包含哪些阶段？请详细介绍每个阶段的输入和输出。

## Expected Output

回复应覆盖 pypto-op-design 的核心工作流。该工作流是问题驱动的迭代设计，核心为 4 轮迭代，每轮聚焦一个核心问题，发现矛盾时回溯修正前序决策：
- 第 1 轮 计算图与精度路由：输入算子规格（API 探索报告 / Golden 参考可选），输出 API 调用序列、dtype 流转与 cast 点、排除的备选方案（DESIGN.md §1）
- 第 2 轮 Tiling 推导：输入第 1 轮确定的 API 序列与 tensor 清单，输出 tile 参数、UB 容量估算与展开检查结果（DESIGN.md §3）
- 第 3 轮 Loop、数据流与 SymbolicScalar 分析：输入第 1 轮 API 序列与第 2 轮 tile 配置，输出动态轴标注、完整伪代码、伪代码可行性验证（DESIGN.md §4）
- 第 4 轮 约束交叉验证：输入前 3 轮全部产出，对 API / Tiling / Loop / SymbolicScalar 逐项检查，不通过则回溯到对应轮次修正（DESIGN.md §5）
回复还应说明 DESIGN.md 的输出结构（§1-§6 共 6 个章节）与最终的完成报告。

## Expectations
- [contains] 计算图与精度路由
- [contains] Tiling
- [contains] Loop
- [contains] SymbolicScalar
- [contains] 约束交叉验证

