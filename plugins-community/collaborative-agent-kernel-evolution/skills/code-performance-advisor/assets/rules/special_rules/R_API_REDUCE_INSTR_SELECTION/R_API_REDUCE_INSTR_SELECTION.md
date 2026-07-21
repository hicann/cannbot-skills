# 规则名称：规约指令选型：按 shape 组合 BlockReduce/WholeReduce

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要对连续缓冲区进行规约（ReduceSum, ReduceMax）操作。
- **形状/数据类型上下文**：数据量极大，超出了单条指令的最佳处理范围。

## 2. 模式描述 (Pattern)
- **优化原理**：根据待规约数据的 Shape 和硬件最佳 Block 长度，混合使用 `BlockReduceSum`（粗粒度块规约）和 `WholeReduceSum`（精细化全局规约）。
- **目标**：通过分级递进的方式，在保证单条指令吞吐的同时，最小化规约链条的整体深度。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：单纯循环调用全局规约会导致指令等待（Wait）占比过高。分级规约可以将数据初步压缩，加快收敛。
- **事实桥接**：
  - 指令互补 -> 充分利用向量执行单元的流水潜力。
  - 缩短归并路径 -> 提升 `aiv_vec_ratio`。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aiv_vec_ratio`（向量利用率）
  - `Task Duration(us)`（耗时走向）
  - `PipeStall` (观察规约之间的气泡)
- **如何解读（定性）**：
  - 如果规约阶段的耗时在整个算子中呈现异常高峰。
  - 观察到连续的规约指令之间存在较长的同步等待（`PIPE_V` 停顿）。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_API_REDUCE_INSTR_SELECTION/code_snippets/`
- **实施步骤**：
  - 首先使用 `BlockReduceSum` 将大块数据规约到 intermediate 缓冲区；
  - 对结果进一步调用 `WholeReduceSum` 完成最终汇总。

## 6. 约束与副作用 (Constraints)
- **缓冲区需求**：需要准备额外的临时 UB/calcBuf 存储中间结果。
- **同步控制**：必须在不同层级规约间插入 `PipeBarrier<PIPE_V>`。
- **适用场景**：`O.Reduce`, `U.Vector`。

## 7. 验证逻辑 (Verification)
- **验证原则**：规约段指令执行密度的增加。
- **推荐验证项**：
  - `aiv_vec_ratio`：期望提升；
  - `Task Duration(us)`：期望下降。
- **验证方法**：检查甘特图，确认原本支离破碎的规约流变得更加紧凑。

## 标签
- Domain: `U.Vector`, `O.Reduce`
- Symptom: `S.LowVecUtil`
- Context: `C.Arch.910B`
