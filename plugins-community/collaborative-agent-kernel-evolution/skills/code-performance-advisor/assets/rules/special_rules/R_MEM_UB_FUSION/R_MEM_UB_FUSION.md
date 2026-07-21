# 规则名称：UB融合：连续vector计算避免中间结果回GM

## 1. 需求场景 (Requirement)
- **业务背景**：算子涉及多次 Vector 计算（如 Exp -> Abs），且前一阶段的输出是后一阶段的输入。
- **形状/数据类型上下文**：计算链较长，且中间结果的数据规模适中，可容纳于 UB（`C.UB.Capacity`）。

## 2. 模式描述 (Pattern)
- **优化思路**：将中间结果暂时保留在 UB（Unified Buffer）中，直接作为下一阶段 Vector 指令的输入，避免“回刷 GM -> 再从 GM 搬回”的无效路径。
- **目标**：显著减少 GM ↔ UB 的往返搬运次数，降低 MTE2/MTE3 负载，缩短端到端耗时。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：不必要的 GM 往返搬运会引入极大的访存延迟和带宽占用。对于 $N$ 段计算链，非融合方式需要 $2N$ 次主存储访问。
- **事实桥接**：
  - 片上复用 -> 消除中间结果的 `DataCopy` 开销。
  - 带宽节省 -> 缓解 `S.TransferDominated` 瓶颈。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（耗时大幅超过计算本身）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（搬运占比极高）
  - `Block Dim`（确认各核负载特征）
- **如何解读（定性）**：
  - 观察 Profiling 中是否存在针对同一块数据的连续“搬出后再搬入”模式；
  - 确认 `S.MemoryBound` 标签生效，且瓶颈在于访存频率而非访存总量。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_UB_FUSION/code_snippets/`
- **实施步骤**：
  - 在 `Process` 循环中，将连续的计算逻辑写在同一个 Tiling 处理片内；
  - 保证计算结果暂存在 `LocalTensor` 中不显式调用 `DataCopy` 到 GM，直到最终结果产生；
  - 配合 `PipeBarrier` 确保不同流水段的同步。

## 6. 约束与副作用 (Constraints)
- **内存/UB 使用**：需权衡计算链长度与 UB 空间上限。若空间不足，仍需通过双缓冲（Double Buffer）或分块写回缓解。
- **适用场景**：`S.MemoryBound`, `U.Vector`。
- **不适用场景**：中间结果需要跨算子持久化或数据量远超 UB 容量的情况。

## 7. 验证逻辑 (Verification)
- **验证原则**：搬运次数减少与性能指数级提升。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈大幅下降趋势；
  - `aic_mte2_ratio`：比例应显著降低。
- **验证方法**：对比优化前后的 Profiling 甘特图，确认无效搬运动员消失。

## 标签
- Domain: `U.Vector`, `O.General`
- Symptom: `S.MemoryBound`, `S.TransferDominated`
- Context: `C.UB.Capacity`
