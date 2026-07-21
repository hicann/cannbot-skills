# 规则名称：尽量一次搬运较大的数据块，降低 DMA 启动开销

## 1. 需求场景 (Requirement)
- **业务背景**：算子存在大量的散块搬运或处理高度非连续的数据。
- **形状/数据类型上下文**：单次搬运字节数较小的情况（如小 Tile、分片细碎）。

## 2. 模式描述 (Pattern)
- **优化原理**：合并相邻的搬运任务或同源数据块，使单次 DataCopy 调用覆盖更多连续空间。
- **目标**：降低 DMA 控制器的启动/发射计数（Launch Count），使传输时间更多消耗在物理搬运上而非指令发射上。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：DMA 发射本身具有流水启动延时。如果 $\frac{\text{搬运字节数}}{\text{发射次数}}$ 过低，流水线的启动开销将掩盖实际带宽性能。
- **事实桥接**：
  - 合并任务 -> 提高 `avg_bytes_per_transfer`。
  - 减少 MTE 竞争 -> 降低 `S.DmaOverhead` 占比。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio`（搬运占比）
  - `Task Duration(us)`（耗时走向）
  - `transfer_count`（单核搬运次数，需定性观察）
- **如何解读（定性）**：
  - 观察到 `S.DmaOverhead` 标签明显，且搬运频率极高但数据吞吐无法饱和。
  - 检查代码是否存在针对微小块的高频循环搬运逻辑。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_DMA_LARGE_BLOCK_TRANSFER/code_snippets/`
- **实施步骤**：
  - 优化 Tiling 策略，优先考虑空间局部性大块搬运；
  - 使用 `DataCopy` 的 Stride 功能代替高频微块搬运；
  - 扩大 UB/L1 缓冲区以承载更大的数据块。

## 6. 约束与副作用 (Constraints)
- **局部性降低**：过度合并可能导致 UB 空间过快溢出。
- **流水中断**：过大的块可能阻塞其他流水的并发度，需配合 `PipeBarrier` 进行平衡。
- **适用场景**：`S.DmaOverhead`, `S.MteBusy`。

## 7. 验证逻辑 (Verification)
- **验证原则**：关注单位搬运量的系统级改善。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势；
  - `aic_mte2_ratio`：期望下降。
- **验证方法**：检查甘特图，确认原本密集的 DataCopy 标记变得稀疏但长度增加。

## 标签
- Domain: `U.DMA`, `O.DataCopy`
- Symptom: `S.DmaOverhead`, `S.MteBusy`
- Context: `C.Arch.910B`
