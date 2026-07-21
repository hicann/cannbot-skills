# 规则名称：GM 地址 512B 对齐，提高搬运带宽效率

## 1. 需求场景 (Requirement)
- **业务背景**：算子存在大量的 Global Memory (GM) 与片上存储（UB/L1/L0）之间的交互。
- **形状/数据类型上下文**：搬运长度或 Stride 存在非对齐情形（`C.Align.256B` 敏感）。

## 2. 模式描述 (Pattern)
- **优化原理**：确保所有的搬运起始地址、跨度（Stride）和单次搬运长度均满足 512 字节（或硬件基数 256 字节）对齐。
- **目标**：避免总线事务被拆分（Split Transaction），提升硬件总线的脉冲利用率，降低 MTE 单元的忙等待。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果推导**：非对齐访存会导致一次总线 Burst 无法载入完整数据，甚至引起两次无效传输，从而使有效带宽折损。
- **事实桥接**：
  - 事务对齐 -> 提高带宽利用率峰值。
  - 减少 MTE 等待 -> 提高 `aic_mte2_ratio` 的有效比重。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（搬运占比）
  - `Task Duration(us)`（耗时走向）
  - `hbm_bw_util` (若 profiling 支持：HBM 带宽利用率)
- **如何解读（定性）**：
  - 观察到搬运占比极高，但实际吞吐量远低于硬件理论值。
  - `S.LowComputeUtil` 与 `S.TransferDominated` 同时存在。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_DMA_GM_ALIGN_512B/code_snippets/`
- **实施步骤**：
  - 在 Tiling 时对起始偏移量进行向上或向下圆整。
  - 针对尾块处理，若无法对齐，尽量通过合并小搬运来规避性能崩塌。
  - 为 GM 缓冲区添加必要的 Padding。

## 6. 约束与副作用 (Constraints)
- **存储开销**：Padding 会导致 GM 内存指纹微量增加。
- **业务正确性**：对齐后的计算范围需配合指令掩码（Mask）处理以保证结果正确。
- **适用场景**：`S.TransferDominated`, `U.DMA`。

## 7. 验证逻辑 (Verification)
- **验证原则**：相同数据量下的搬运耗时显著压降。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势；
  - `aic_mte2_ratio`：比例改善。
- **验证方法**：使用 AIPP 或特定性能计数器观察总线 Split 计数下降。

## 标签
- Domain: `U.DMA`, `O.DataCopy`
- Symptom: `S.TransferDominated`, `S.LowComputeUtil`
- Context: `C.Align.256B`
