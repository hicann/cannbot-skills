# 规则名称：使能 Iterate 异步接口，减少 AIC/AIV 同步依赖

## 1. 需求场景 (Requirement)
- **业务背景**：算子使用 `Iterate` 系列接口进行迭代计算，且属于 AI Core (AIC) 与 AI Vector (AIV) 混合参与的模式。
- **形状/数据类型上下文**：中大规模矩阵乘配合同步 Vector 操作（`O.MatMul`）。

## 2. 模式描述 (Pattern)
- **优化原理**：在 `Iterate` 或 `IterateAll` 接口中将同步标记设为 `false`（如 `template Iterate<false>()`）。由硬件自动处理消息队列，而非显式的指令级握手。
- **目标**：降低 AIC 与 AIV 之间因消息同步而产生的 Pipeline Stall，缓解主控单元的等待负担。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：强同步模式下，AIC 在每次迭代末尾必须确认 AIV 已处理完消息。异步模式允许 AIC 预先发射下一轮迭代的消息。
- **事实桥接**：
  - 流水重叠 -> AIV 计算时间被 AIC 的后续迭代所覆盖。
  - 减少同步指令 -> 优化标量流水与同步气泡（`S.PipeStall`）。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（整体执行时长）
  - `PipeStall`（流水线同步停顿）
  - `aic_scalar_ratio`（评估同步握手开销）
- **如何解读（定性）**：
  - 观察到 AIC 侧存在大量的 idle 时间段，且对应时刻 AIV 正在处理 `GetTensorC` 之后的数据。
  - 判定 `Iterate` 循环中是否存在显式的同步屏障。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_PIPE_ITERATE_ASYNC/code_snippets/`
- **实施步骤**：
  - 为 `Iterate` 接口添加 `<false>` 模板参数；
  - 确保配套的临时 workspace 能够满足异步数据的存储需求；
  - 检查依赖链，必要时添加自定义屏障以解决竞态。

## 6. 约束与副作用 (Constraints)
- **内存占用**：异步模式可能需要更大的 workspace 空间来缓存尚未处理完的数据切片。
- **调试难度**：异步触发的问题在 Profiling 中表现较为隐晦。
- **适用场景**：`S.PipeStall`, `O.MatMul`, `U.Mix`。

## 7. 验证逻辑 (Verification)
- **验证原则**：AI Core 发射指令流的连贯性。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈下降趋势；
  - `PipeStall`：期望对应迭代段的同步损耗减少。
- **验证方法**：检查甘特图，确认 AIC 的迭代标记不再频繁等待 AIV 的回包。

## 标签
- Domain: `U.Mix`, `O.MatMul`
- Symptom: `S.PipeStall`
- Context: `C.Arch.910B`
