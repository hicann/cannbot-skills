# 规则名称：核间负载均衡，减少拖尾，提升整体并行效率

## 1. 需求场景 (Requirement)
- **业务背景**：算子在大规模核分配场景下执行任务。
- **形状/数据类型上下文**：任务总量无法被物理核数（Block Dim）整除，产生明显的尾块。

## 2. 模式描述 (Pattern)
- **优化原理**：在 Tiling 侧优化任务分配算法（如分两轮不同粒度的分配），使得“尾核”的工作量与其他核尽量接近，或由多核共同分摊尾部负载。
- **目标**：降低负载不均度（Imbalance Ratio），消除因“一核拖累全场”导致的并行瓶颈。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：并行执行的总时长受限于最慢的那个核。如果尾核任务量是常规核的两倍，则系统利用率将减半。
- **事实桥接**：
  - 均匀分配 -> 消除甘特图末端的空隙。
  - 提升并行度 -> 改善 `load_imbalance_ratio`。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（观察最大耗时核与平均核的差距）
  - `Block Dim`（确认参与核数）
  - `load_imbalance_ratio`（若 profiling 预置该指标）
- **如何解读（定性）**：
  - 检查甘特图中各核的时间对齐度。
  - 判定是否存在部分核早早结束而个别核持续工作的“拖尾”现象。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_TILING_CORE_LOAD_BALANCE/code_snippets/`
- **实施步骤**：
  - 计算 `FormerNum` 和 `TailNum`，确保每个核的 Loop 次数差异不超过 1；
  - 针对复杂 Shape，引入基于贪心算法或动态查询表的 Tile 划分。

## 6. 约束与副作用 (Constraints)
- **调度成本**：动态划分可能引入额外的 Host 侧 Tiling 计算开销。
- **适用场景**：`S.LowComputeUtil`, `S.LoadImbalance`。
- **不适用场景**：任务极小，即便完美均衡也无法掩盖启动延时的情况。

## 7. 验证逻辑 (Verification)
- **验证原则**：多核执行尾部的一致性。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈下降趋势；
  - `load_imbalance_ratio`：期望下降。
- **验证方法**：检查甘特图末尾，各核的任务完成标记点应趋于重合。

## 标签
- Domain: `U.DMA`, `O.General`
- Symptom: `S.LowComputeUtil`
- Context: `C.Arch.910B`
