---
skill_name: pypto-op-design
---

# Case 1: 简单 Vector 算子（Add）设计

## Config
- Eval Mode: file_based
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000

## Prompt

我想设计一个 Add 算子，算子规格如下：

算子名称：Add
数学公式：z = x + y（逐元素加法）
输入：x fp16 [1024, 4096], y fp16 [1024, 4096]
输出：z fp16 [1024, 4096]

请帮我生成 DESIGN.md

## Expected Output

生成的 DESIGN.md 应覆盖以下要点：
- §1 概述：算子名称 Add、数学公式 z=x+y、简单逐元素加法
- §2 API 映射：使用 PyPTO 的逐元素加法 API，映射表每步有明确的 API 名称
- §3 数据规格设计：Input/Output dataclass，包含 x_fp16、y_fp16、z_fp16，shape 均为 [1024, 4096]
- §4 Tiling 策略：应判断为 Vector 类型（不含 matmul 操作），使用 set_vec_tile_shapes，尾轴 32B 对齐
- §5 Loop 结构：应判断不需要 Loop（简单逐元素算子，单次 Tile 可处理），使用场景 A 模板
- §6 验证方案：Golden 函数为简单的 torch 加法，覆盖典型 shape 配置
- §7 性能指标：包含 TileShape 配置和 pass_options
- §8 风险点：包含 Vector TileShape 尾轴对齐等注意事项
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

生成的 DESIGN.md 应覆盖以下要点：
- §1 概述：算子名称 Matmul，数学公式 C=A×B+bias，矩阵乘加偏置
- §2 API 映射：应包含 matmul 相关 API（mm_dequant 或对应的 PyPTO Cube API）和加法 API，映射表完整无空白
- §3 数据规格设计：Input/Output dataclass，包含 A/B/bias/C 四个 tensor，数据格式 fp16
- §4 Tiling 策略：应判断为 Cube 类型（含 matmul），使用 set_cube_tile_shapes，说明 TileShape 设置依据
- §5 Loop 结构：应根据具体 shape 分析是否需要 Loop，说明 Loop 类型选择理由
- §6 验证方案：Golden 函数使用 torch.matmul + bias 加法，覆盖典型配置
- §7 性能指标：Cube TileShape、pass_options、runtime_options
- §8 风险点：包含 Cube tile shape 约束、L1 容量等注意事项
- 无残留 {placeholder} 占位符

## Expectations

- [file_exists] DESIGN.md

---

# Case 3: 信息不足时主动追问

## Config
- Disabled: true

## Prompt

帮我设计一个算子

## Expected Output

回复应询问算子名称、数学公式、输入输出规格等关键信息。不应在缺乏算子规格的情况下直接生成设计文档。应至少询问算子名称和数学公式。

## Expectations

- [not_contains] §1 概述
- [not_contains] set_vec_tile_shapes

---

# Case 4: 正向看护-多 Skill 环境下正确触发 PyPTO 算子设计

## Config
- Max Tokens: 300000
- Max Tokens (deepseek-v4-flash): 350000
- Max Tokens (glm-5): 320000
- Distractor skills: pypto-op-develop;pypto-api-explore;ascendc-st-design;ascendc-tiling-design

## Prompt

请使用 pypto-op-design skill 的算子设计工作流来设计一个 Softmax 算子，算子规格如下：

算子名称：Softmax
数学公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
输入：x fp16 [1024, 4096]
输出：y fp16 [1024, 4096]

请运行完整的算子设计工作流，生成 DESIGN.md，输出到当前目录。

## Expected Output

回复应调用 pypto-op-design skill 并执行算子设计工作流，生成 Softmax 算子的 DESIGN.md。内容应包含：算子概述、API 映射（exp、reduce_sum、除法等操作的 PyPTO API）、数据规格、Tiling 策略（Vector 类型）、Loop 结构分析、验证方案等。应覆盖 DESIGN.md 的 9 个章节，且最终输出 DESIGN.md 文件到当前目录。

## Expectations

- [skill_activated] pypto-op-design

---

# Case 5: 工作流知识验证

## Prompt

pypto-op-design 算子设计工作流包含哪些阶段？请详细介绍每个阶段的输入和输出。

## Expected Output

回复应覆盖 pypto-op-design 的核心工作流阶段，至少包括：
- 输入验证与特征分析阶段：读取算子规格、验证必须字段、判断算子类型（Cube/Vector）
- 信息收集阶段：知识库查询、API 规格验证
- 生成 DESIGN.md 阶段：基于模板生成 9 个章节
- 质量自检阶段：5 项检查表（API 映射、Tiling/Loop 理由、验证覆盖、风险点、占位符）
- 定向回修和输出阶段

## Expectations

- [contains] 输入验证
- [contains] API 映射
- [contains] Tiling
- [contains] Loop
- [contains] 质量自检
