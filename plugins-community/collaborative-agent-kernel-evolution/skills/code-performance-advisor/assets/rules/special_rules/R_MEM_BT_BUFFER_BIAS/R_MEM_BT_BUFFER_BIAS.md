# 规则名称：BT Buffer：高效计算/累加 bias，减少额外搬运

## 1. 需求场景 (Requirement)
- **业务背景**：算子涉及带偏置（Bias）的矩阵乘法或大块加法操作。
- **形状/数据类型上下文**：偏置数据通常针对矩阵的一维（如 C 维或 N 维），需要对齐（`C.Align.256B`）。

## 2. 模式描述 (Pattern)
- **优化原理**：利用固有的 Bias Table Buffer (BT/C2) 路径，将 Bias 搬运至专用的 BT 缓冲区，并直接传给 `Mmad` 或 `Fixpipe` 等硬件支持的偏置叠加路径。
- **目标**：消除“矩阵乘结果入 UB -> Bias 入 UB -> Vector Add”的额外搬运与计算开销，实现 Bias 处理的硬件加速。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：软加 Bias 需要将庞大的矩阵乘结果在片上进行多轮往返同步，导致 MTE3 到 Vector 的压力骤增。
- **事实桥接**：
  - 硬件流水集成 -> Bias 的累加可以在输出搬运阶段（Fixpipe）或计算末尾阶段同步完成，几乎不增加额外周期。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio`（数据搬运占比）
  - `aiv_vec_ratio` / `aic_mac_ratio`（Vector 指令占比是否因 Bias 偏高）
  - `DmaOverhead`（DMA 同步开销）
- **如何解读（定性）**：
  - 如果发现 Vector 加法指令（`Add`）消耗了明显的计算时长，且其输入来自于 MatMul 的输出。
  - 判定是否存在冗余的中间缓冲区交互。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_BT_BUFFER_BIAS/code_snippets/`
- **实施步骤**：
  - 声明 `QuePosition::C2` (BT) 缓冲区；
  - 将 Bias 数据通过 `DataCopy` 搬运至 C2 缓冲区；
  - 在 `Mmad` 调用中填入 Bias 对应的 `LocalTensor`；
  - 确保偏移地址与对齐遵循 `C.Align.256B` 约束。

## 6. 约束与副作用 (Constraints)
- **内存/UB 使用**：需满足 Bias Table 的硬件对齐与容量约束。
- **适用场景**：`O.MatMul`, `O.Conv` 带 Bias 场景。
- **不适用场景**：非线性偏置叠加或复杂的后置逐元素（Elementwise）逻辑。

## 7. 验证逻辑 (Verification)
- **验证原则**：关注 Vector 指令流水线的净化与带宽利用的简化。
- **推荐验证项**：
  - `aiv_vec_ratio`：期望呈显著下降趋势；
  - `Task Duration(us)`：期望呈现性能提升。
- **验证方法**：检查甘特图，确认原本独立的 Bias 加法段消失，计算与搬运逻辑更为紧凑。

## 标签
- Domain: `U.Cube`, `O.MatMul`
- Symptom: `S.MemoryBound`, `S.DmaOverhead`
- Context: `C.Align.256B`
