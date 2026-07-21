# 规则名称：通过L0C Buffer数据暂存实现高效的矩阵乘结果累加

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要多次执行矩阵乘并在空间维度上进行结果累加（如循环多次分片的 MatMul）。
- **形状/数据类型上下文**：涉及对同一输出 Tiling 块的多次 Mmad 覆盖累加。

## 2. 模式描述 (Pattern)
- **优化思路**：利用 L0C 缓存的持久化能力，将前一次矩阵乘的结果保留在 L0C 寄存池中。通过 `Mmad` 接口的 `cmatrixInitVal` 参数实现原地累加，避免将中间值换出到 UB 或 GM。
- **目标**：消除中间结果的 `CO1 -> UB/GM` 搬运以及额外的 Vector 加法计算损耗。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：中间结果的显式同步（搬出再搬入并加法）会增加一倍以上的搬运量，并额外占用 Vector 算力。
- **事实桥接**：
  - 指令内累加 -> 物理上省去了数据流经 GM/UB 的延迟。
  - 并行度优化 -> L0C 容量约束（`C.L0.Capacity`）决定了单次可缓存的矩阵大小。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（观察计算周期与预期差异）
  - `aic_mac_ratio` / `aic_mte2_ratio`（判读计算与搬运的平衡性）
  - `cube_utilization(%)`（算力利用率）
- **如何解读（定性）**：
  - 判定是否存在“频繁计算 -> 频繁搬出 -> 再回写 Add”的特征模式。
  - `aic_mte2_ratio` 呈现异常波动，且存在针对 Output 的冗余 DataCopy 操作。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_L0C_ACCUM_MMAD/code_snippets/`
- **实施步骤**：
  - 设置 `Mmad` 接口的首个操作 `cmatrixInitVal = true`（清零并初始化）；
  - 后续累加操作设置 `cmatrixInitVal = false` 并指向原输出缓冲区；
  - 合理安排 `PipeBarrier<PIPE_M>` 保证写后读同步。

## 6. 约束 与副作用 (Constraints)
- **内存/UB 使用**：受 `C.L0.Capacity` 约束，L0C 空间有限，累加的 tile 大小不能超过硬件最大承载。
- **适用场景**：矩阵乘密集累加。
- **不适用场景**：非矩阵乘计算、或单核内无法完成所有分片累加且需要通过原子操作（Atomic）跨核完成的情形。

## 7. 验证逻辑 (Verification)
- **验证原则**：关注算子循环架构的变化与搬运单元利用率的动态改善。
- **推荐验证项**：
  - `aic_mte2_ratio`：期望呈下降趋势；
  - `cube_utilization(%)`：期望因等待时间减少而提升。
- **验证方法**：对比优化前后 MTE2 的总搬运流量，确认流量显著降低。

## 标签
- Domain: `U.Cube`, `O.MatMul`
- Symptom: `S.MemoryBound`, `S.TransferDominated`
- Context: `C.L0.Capacity`
