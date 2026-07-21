# 规则名称：Welford 在线算法优化均值方差计算

## 1. 需求场景 (Requirement)
- **业务背景**：Norm 类算子（BatchNorm/LayerNorm/RMSNorm）需要计算均值和方差，传统两遍算法需要两次遍历数据，内存访问开销大。
- **形状/数据类型上下文**：适用于 `O.Norm` 算子族，特别是大规模数据场景（Reduce 维度较大）。

## 2. 模式描述 (Pattern)
- **优化思路**：使用 Welford 在线算法单次遍历同时计算均值和方差，减少 50% 内存访问，同时具有更好的数值稳定性。
- **目标**：单次遍历完成统计计算，减少内存访问，提升数值稳定性。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：传统两遍算法需要两次遍历数据，Welford 算法单次遍历即可完成，理论上减少 50% 内存访问。
- **事实桥接**：
  - 单次遍历 -> 减少 GM 带宽占用 -> 降低 MTE 瓶颈
  - 在线更新 -> 数值稳定性好 -> 避免大数吃小数
  - 并行 Welford -> 向量化更新 -> 充分利用 Vector 并行度

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Op Type`（算子类型为 Norm 族）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（搬运单元占比，两遍算法占比高）
  - `Task Duration(us)`（整体耗时）
  - `Input Shapes`（据此推算每 channel 的 Reduce 维度大小）
- **如何解读（定性）**：
  - 算子为 BatchNorm/LayerNorm/RMSNorm 等 Norm 类
  - `aiv_mte2_ratio` 较高，表明内存访问是瓶颈
  - 使用标签 `O.Norm`, `S.MemoryBound` 标注场景
- **形状阈值约束（重要）**：
  - Welford 的收益来自于**减少数据遍历次数**，因此 Reduce 维度越大，效果越显著
  - 经验阈值：`num_elements_per_channel ≥ 8192` 时 Welford 优势明显
  - `num_elements_per_channel < 8192` 时（如本例的 1024），两遍算法与 Welford 性能相近；此时瓶颈更可能来自**标量 GM 访问**（GetValue/SetValue）而非数据遍历次数
  - **建议在 suggest 输出中显式计算 num_elements_per_channel**，并说明预期收益比例
  - 若 `num_elements_per_channel < 4096`，优先考虑向量化标量参数访问（weight/bias 批量加载）而非 Welford

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (batch_norm_v3 Welford 实现)
- **实施步骤（示例性）**：
  1. 初始化 Welford 状态：`count = 0`, `mean = 0`, `M2 = 0`（方差累计量）
  2. 实现 Welford 并行更新函数：
     ```cpp
     __aicore__ inline void WelfordParallelUpdate(
         float& count, LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
         LocalTensor<float>& xTensor, LocalTensor<float>& deltaTensor, const uint32_t& calcMask)
     {
         count += 1;
         Sub(deltaTensor, xTensor, meanTensor, calcMask);  // delta = x - mean
         PipeBarrier<PIPE_V>();
         Muls(xTensor, deltaTensor, 1 / count, calcMask);  // x = delta / count
         PipeBarrier<PIPE_V>();
         Add(meanTensor, meanTensor, xTensor, calcMask);   // mean += delta / count
         Mul(deltaTensor, deltaTensor, deltaTensor, calcMask);  // delta^2
         PipeBarrier<PIPE_V>();
         Muls(deltaTensor, deltaTensor, (count - 1) / count, calcMask);  // delta^2 * (count-1)/count
         PipeBarrier<PIPE_V>();
         Add(m2Tensor, m2Tensor, deltaTensor, calcMask);   // M2 += delta^2 * (count-1)/count
     }
     ```
  3. 单次遍历数据，每次调用 WelfordParallelUpdate 更新统计量
  4. 最终方差为 `variance = M2 / count`
  5. 对于分块计算场景，实现 Welford 归约函数合并多个分块的统计量

## 6. 约束与副作用 (Constraints)
- **算法复杂度**：Welford 算法理解难度较高，需要维护 mean 和 M2 两个中间变量
- **中间变量**：需要额外的 `deltaTensor` 等临时缓冲区
- **适用场景**：`O.Norm` 算子族，`S.MemoryBound`（内存受限场景）
- **不适用场景**：小数据量场景（两遍算法与 Welford 性能差异不明显）

## 7. 验证逻辑 (Verification)
- **验证原则**：内存访问减少，整体性能提升，数值稳定性好
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势（20-50%）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`：期望呈下降趋势（搬运占比减半）
  - `Numerical Stability`：与 PyTorch FP32 baseline 对齐，避免大数吃小数问题
- **验证方法**：
  - 对比两遍算法 vs Welford 算法的性能和精度
  - 使用大规模数据验证数值稳定性
  - 确认方差计算的正确性（特别是边界情况）

## 标签
- Domain: `U.Vector`, `O.Norm`
- Symptom: `S.MemoryBound`, `S.TransferDominated`
- Context: `C.Reduce.LastDim`
