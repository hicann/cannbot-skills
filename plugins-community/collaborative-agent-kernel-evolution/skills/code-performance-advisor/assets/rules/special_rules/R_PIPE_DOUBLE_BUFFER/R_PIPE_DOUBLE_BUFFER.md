# 规则名称：Double Buffer 并行搬运与计算，隐藏搬运时间

## 1. 需求场景 (Requirement)
- **业务背景**：算子采用分块处理（Tiling）模式，且每一片的执行包含 CopyIn, Compute 和 CopyOut 三个阶段。
- **形状/数据类型上下文**：Tiling 片数（Round）大于 1，且单片数据量适合在片上 UB 分配两个缓冲区。

## 2. 模式描述 (Pattern)
- **优化原理**：通过使能 `TPipe` 的双缓冲（Ping-Pong）机制，使得“下一片的搬入”可以与“当前片的计算”在时间上重叠并行。
- **目标**：隐藏数据的搬运延时，消除计算单元因等待数据而产生的气泡（Bubble）。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：串行执行下，总耗时等于搬运与计算之和。双缓冲使总耗时趋近于 $\max(\text{搬运}, \text{计算})$。
- **事实桥接**：
  - 流水隐藏 -> 降低 `compute_idle_gap`（计算空闲间隔）。
  - 核心利用 -> 提高计算单元（Cube/Vector）的有效占空比。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（整体执行时长）
  - `aic_mte2_ratio` / `aiv_vec_ratio` (观察搬运与计算的比例)
  - `PipeStall` (流水线等待信号)
- **如何解读（定性）**：
  - 在 Profiling 甘特图中，如果 Copy 序列与 Compute 序列完全错开心，且两者之间存在明显的空白等待；
  - 判定 `compute_idle_gap` 呈现周期性气泡特征。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_PIPE_DOUBLE_BUFFER/code_snippets/`
- **实施步骤**：
  - 在 `pipe.InitBuffer` 中将缓冲区个数参数由 1 修改为 2；
  - 循环时确保 `EnQue` 和 `DeQue` 操作能够按乒乓逻辑流转；
  - 调整 Tiling 粒度以适配扩容后的 UB 占用（`C.UB.Capacity`）。

## 6. 约束与副作用 (Constraints)
- **内存开销**：由于分配了 2 倍的缓冲区，如果单片过大可能导致栈溢出或 UB 空间不足（`C.UB.Capacity`）。
- **适用场景**：`S.PipeStall`, `S.LowComputeUtil`。
- **不适用场景**：Tiling 片数过少，或单片搬运开销远小于管理开销的情况。

## 7. 验证逻辑 (Verification)
- **验证原则**：计算单元利用率的稳定提升。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势；
  - `aiv_vec_ratio` / `aic_mac_ratio`：期望在总时长中占比上升（计算变密集）。
- **验证方法**：检查甘特图，确认 Copy 段与 Compute 段在时间轴上出现了层叠部分。

## 标签
- Domain: `U.Vector`, `U.Mix`
- Symptom: `S.PipeStall`, `S.LowComputeUtil`
- Context: `C.UB.Capacity`
