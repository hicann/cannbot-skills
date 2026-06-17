# 评审提示词模板

所有评审都使用 `Agent` 工具。在单条消息中以 `run_in_background=true` 同时启动成对的评审 Agent 以实现并行执行。

**重要：** 在把这些提示词传给 Agent 工具之前，请将 `{OP}` 替换为算子名，将 `{SOURCE_PATH}` 替换为算子源文件路径。Agent 无法访问父会话的变量。如果不存在源文件（公式或文本输入），请从提示词中删除对源文件的引用。

提示词结构：`读取 X 和 Y，校验 A/B/C，将结论写入 Z。`

## 结构化判定格式

所有评审 Agent 对每一项检查都必须使用以下格式：

```
### Check N: [title]
**Verdict:** PASS | FAIL | CONCERN
**Rationale:** one-line explanation

[details if FAIL or CONCERN]
```

在结论文件顶部写入：`**Overall: PASS** | **Overall: NEEDS_REVISION**`（只要有任一项检查为 FAIL，即为 NEEDS_REVISION）。

## 阶段 3：定义文档

在一条消息中同时启动两个 Agent：

### Agent A：数学评审

```
Agent(
  subagent_type="general-purpose",
  run_in_background=true,
  name="math-review",
  description="评审数学正确性",
  prompt="""
    你正处于该算子工程中。正在构建的算子是 `{OP}`。
    读取 `docs/{OP}/{OP}_definition.md` 以及位于 `{SOURCE_PATH}` 的算子源文件（如果存在）。
    校验：
    1. 数学公式正确表达了预期的计算（如有源文件，请与之交叉核对）。
    2. 公式推导在数学上是严谨的。
    3. CPU 参考伪代码计算出的结果与公式一致。
    4. 任何迭代累加模式都已标注并附带精度分析。
    对每一项检查，给出 PASS、FAIL 或 CONCERN，并说明理由。
    将结论写入 `docs/{OP}/plans/review_math.md`。
  """
)
```

### Agent B：语义评审

```
Agent(
  subagent_type="general-purpose",
  run_in_background=true,
  name="semantics-review",
  description="评审语义与边界场景",
  prompt="""
    你正处于该算子工程中。正在构建的算子是 `{OP}`。
    读取 `docs/{OP}/{OP}_definition.md` 以及位于 `{SOURCE_PATH}` 的算子源文件（如果存在）。
    校验：
    1. 输入/输出语义已清晰定义，并与源文件一致（如果提供了源代码）。
    2. 边界场景已识别（NaN、Inf、零值、负值、边界值）。
    3. 数据类型策略有充分依据（如果存在源代码，检查其与源文件类型处理的一致性）。
    对每一项检查，给出 PASS、FAIL 或 CONCERN，并说明理由。
    将结论写入 `docs/{OP}/plans/review_semantics.md`。
  """
)
```

## 阶段 4：设计文档

在一条消息中同时启动两个 Agent：

### Agent A：UB 预算评审

```
Agent(
  subagent_type="general-purpose",
  run_in_background=true,
  name="ub-review",
  description="评审 UB 预算与切分",
  prompt="""
    你正处于该算子工程中。正在构建的算子是 `{OP}`。
    读取 `docs/{OP}/{OP}_design.md`。
    AscendC Cast 支持矩阵：float->int8 不支持；应使用 float->half（CAST_NONE）再 half->int8（CAST_ROUND）。
    校验：
    1. 所有存活缓冲区都已计入 `liveBytesPerElem`。
    2. 切分公式正确，且涵盖了所有缓冲区。
    3. fp16 路径已计入类型转换工作缓冲区。
    4. 任何 UB-to-UB 拷贝路径都记录了字节数、`32B` 对齐、尾块策略以及按位拷贝要求。
    5. 对每个块大小常量，校验 fp16 与 fp32 下均满足 `count * sizeof(T) >= 32`。
    对每一项检查，给出 PASS、FAIL 或 CONCERN，并说明理由。
    将结论写入 `docs/{OP}/plans/review_ub.md`。
  """
)
```

### Agent B：指令序列评审

```
Agent(
  subagent_type="general-purpose",
  run_in_background=true,
  name="instr-review",
  description="评审指令序列",
  prompt="""
    你正处于该算子工程中。正在构建的算子是 `{OP}`。
    读取 `docs/{OP}/{OP}_design.md` 和 `docs/{OP}/{OP}_definition.md`。
    AscendC Cast 支持矩阵：float->int8 不支持；应使用 float->half（CAST_NONE）再 half->int8（CAST_ROUND）。
    校验：
    1. 指令序列计算出的正是定义文档中的正确公式。
    2. 凡是依赖顺序需要的地方都已存在 `PipeBarrier<PIPE_V>()`。
    3. 没有任何缓冲区在写入之前被读取。
    4. 缓冲区复用不会造成 RAW/WAR/WAW 冒险。
    5. 类型转换链符合 AscendC Cast 支持矩阵。
    6. 定义文档中的所有边界场景都已处理，或已明确延后处理并附带理由。
    对每一项检查，给出 PASS、FAIL 或 CONCERN，并说明理由。
    将结论写入 `docs/{OP}/plans/review_instructions.md`。
  """
)
```

### Agent C：Reg API 评审

针对 Ascend950 / `dav-3510` 的设计与实现，额外使用此评审。

```
Agent(
  subagent_type="general-purpose",
  run_in_background=true,
  name="reg-api-review",
  description="评审 Ascend950 Reg API 合规性",
  prompt="""
    你正处于该算子工程中。正在构建的算子是 `{OP}`。
    读取 `docs/{OP}/{OP}_design.md`、`references/reg-api-guide.md` 和 `references/reg-api-patterns.yaml`。
    如果实现已存在，还需读取 `{OP}.asc`。
    校验：
    1. 设计在向量计算、类型转换和规约中使用了 `AscendC::Reg`。
    2. 没有计划或实现 `AscendC::MicroAPI`、Membase、除 `asc_vf_call` 之外的裸 `asc_*` 调用，或经典 AscendC 的计算/类型转换/规约调用。
    3. Reg 封装遵循 `__simd_vf__` + `__aicore__` + `asc_vf_call` 的结构。
    4. 尾块掩码基于元素数量，并应用于存储。
    5. 规约使用 32B 标量槽，且对 Reg 产出的值避免使用 `LocalTensor::GetValue()` 进行标量回读。
    6. 类型转换路径指定了 `CastTrait`、`LoadDist` 和 `StoreDist`，并在需要时包含 B16 的 unpack/pack。
    7. 相互依赖的 Reg 封装调用之间用 `PipeBarrier<PIPE_V>()` 分隔。
    8. 如果本算子指定了 VF 融合上限 `N`（在 `STATE.md`、设计文档或用户请求中），则每个 `__simd_vf__` 函数融合的 VF 计算指令不超过 `N` 条；更长的链应拆分到多个封装中，通过各自独立的 `asc_vf_call` 调用串联。如果未指定上限，则将此项检查报告为 N/A。
    对每一项检查，给出 PASS、FAIL 或 CONCERN，并说明理由。
    将结论写入 `docs/{OP}/plans/review_reg_api.md`。
  """
)
```

## 评审后工作流

在两个评审 Agent 都完成后：

1. 校验每个 `review_*.md` 文件都存在，且至少包含一条 PASS/FAIL/CONCERN 判定。
2. 读取它们的结论文件。
3. 处理所有 FAIL 问题——更新定义/设计文档。
4. 如果问题较为实质，重新运行相应的评审 Agent。
5. 仅在所有评审结论都已解决或确认后再提交。

## 错误处理

- **Agent 超时或无输出：** 重新运行该 Agent 一次。如果再次失败，则手动进行评审。
- **结论文件为空：** Agent 很可能在读取输入文件时遇到了错误。校验文件路径并重新运行。
- **结论相互矛盾：** 同时读取两份结论与源文档，作出判断。将处理方式记录在评审文件中。
- **臆造的问题：** 在采取行动前，将每条 FAIL 结论与实际源文件交叉核对。如果该结论引用了并不存在的代码或约束，则将其丢弃。
