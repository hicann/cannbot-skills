# 规则名称：高效使用搬运 API，减少无效搬运与隐含同步

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要处理非连续（Stride）布局的数据。
- **形状/数据类型上下文**：搬运粒度小或存在规律性的间隔跳转（`S.StridePenalty`）。

## 2. 模式描述 (Pattern)
- **优化原理**：弃用简单的 `for` 循环搬运小块数据，改用 `DataCopy` 中支持 `srcStride`, `dstStride`, `blockLen`, `blockCount` 等参数的高级接口。
- **目标**：将多次小规模搬运合并为一次硬件原生的 Stride 搬运，减少 DMA 调度次数和同步气泡。

## 3. 性能损耗因功链 (Inference / Physics)
- **因果说明**：频繁的 API 调用会引起多次流水空转。硬件 Stride 搬运可以将跳转开销隐藏在传输流水中。
- **事实桥接**：
  - 加速调度 -> 降低 `mte_busy` 但吞吐不高的现象。
  - 减少搬运计数 -> 提升 `avg_bytes_per_transfer`。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio`（数据搬运占比）
  - `Task Duration(us)`（耗时走向）
  - `DmaOverhead`（DMA 同步与其调度开销）
- **如何解读（定性）**：
  - 判定是否存在“单次搬运量极小但频率极高”的特征（针对 Stride 数据）。
  - `mte_busy` 呈现持续性繁忙，但实际外部带宽利用率较低。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_DMA_USE_COPY_API/code_snippets/`
- **实施步骤**：
  - 封装 `DataCopyParams` 结构体，配置正确的 `blockLen` (单位为 32B block) 和 `stride` 参数；
  - 移除原有的显式 `for` 循环搬运代码。

## 6. 约束与副作用 (Constraints)
- **步长步限**：Stride 的物理偏移必须符合硬件寄存器的表示上限。
- **对齐要求**：起始地址仍需遵循硬件对齐约束（`C.Align.256B`）。
- **适用场景**：`S.DmaOverhead`, `S.MteBusy`。

## 7. 验证逻辑 (Verification)
- **验证原则**：搬运次数大幅下降，有效吞吐率提升。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势；
  - `DmaOverhead`：期望大幅改善。
- **验证方法**：检查甘特图，确认原本密集的 `DataCopy` 块被合并为单个长块。

## 标签
- Domain: `U.DMA`, `O.DataCopy`
- Symptom: `S.DmaOverhead`, `S.MteBusy`
- Context: `C.Arch.910B`
