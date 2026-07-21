# 规则名称：共享高阶API临时Buffer，减少UB碎片与搬运次数

## 1. 需求场景 (Requirement)
- **业务背景**：算子在使用 SoftMax 等需要临时 Buffer 的高阶 API 时，UB 空间被不必要的临时缓冲区挤占。
- **形状/数据类型上下文**：数据量较大，导致 UB 频繁切分搬运的情形（`C.UB.Capacity` 敏感）。

## 2. 模式描述 (Pattern)
- **优化思路**：共享临时 Buffer 空间，让多个生命周期不重叠的阶段复用同一块临时 Buffer（采用 `MAX(size)` 分配策略）。
- **目标**：提升单次搬运的数据量规模，降低搬运总次数，改善搬运单元的有效带宽利用率。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：UB 临时 Buffer 独立分配会导致可用内存片碎化，迫使 Tiling 粒度减小，从而增加搬运频率和同步开销。
- **事实桥接**：
  - 复用 Buffer -> 指标上表现为更大的单次搬运量。
  - 减少搬运次数 -> 减少了 MTE2 单元的调度开销。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（整体耗时趋势）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（评估搬运占比）
  - `bytes/transfer_count` (若 profiling 支持，或定性观察搬运密度)
- **如何解读（定性）**：
  - 若 `aic_mte2_ratio` 较高且存在频繁的小数据量搬运；
  - 观察到 UB 分配中存在多个高阶 API 的临时 Buffer 初始化（`pipe.InitBuffer`），且生命周期无重叠。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_SHARE_TMPBUF/code_snippets/`
- **实施步骤**：
  - 计算各阶段所需临时 Buffer 的最大值；
  - 使用统一的 `TBuf` 对象进行 `InitBuffer` 分配；
  - 在不同阶段按需从共享 Buffer 中通过 `Get` 接口获取特定类型的 `LocalTensor`。

## 6. 约束与副作用 (Constraints)
- **内存/UB 使用**：需确保生命周期不重叠。
- **适用场景**：`S.MemoryBound`, `S.TransferDominated`, `C.UB.Capacity`。
- **不适用场景**：多个高阶 API 需并行执行或数据持久化依赖的情况。

## 7. 验证逻辑 (Verification)
- **验证原则**：观察搬运密度下降与总耗时改善。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈下降趋势；
  - `aic_mte2_ratio` / `aiv_mte2_ratio`：期望呈下降趋势（搬运在总时间中占比减弱）。
- **验证方法**：通过 Tiling 实验确认单次搬运动作覆盖的数据量级增加。

## 标签
- Domain: `U.Vector`, `O.Activation`, `O.General`
- Symptom: `S.MemoryBound`, `S.TransferDominated`
- Context: `C.UB.Capacity`
