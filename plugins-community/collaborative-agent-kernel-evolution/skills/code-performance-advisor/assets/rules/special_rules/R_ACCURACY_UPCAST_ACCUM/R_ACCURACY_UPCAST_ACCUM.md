# 规则名称：低精度输入升精度累加计算

## 1. 需求场景 (Requirement)
- **业务背景**：涉及累加、方差等统计计算的算子，使用 FP16/BF16 低精度数据时容易出现累积误差和数值溢出，影响模型训练精度。
- **形状/数据类型上下文**：输入数据类型为 `T.FP16` 或 `T.BF16`，算子涉及多次累加操作（如 Norm、Pooling、Reduce 等）。

## 2. 模式描述 (Pattern)
- **优化思路**：输入数据以低精度（FP16/BF16）存储和搬运，但在计算过程中先 Cast 到 FP32 进行累加和统计计算，最后将结果再 Cast 回原精度输出。
- **目标**：保证低精度输入的数值稳定性，避免累加误差和溢出，同时享受低精度带来的内存带宽优势。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：FP16 有效位数仅 10 位，大量数据累加易导致精度丢失；方差计算涉及平方操作，数值范围扩大，低精度容易溢出。FP32 中间计算可避免这些问题。
- **事实桥接**：
  - 低精度输入 -> 减少 GM 带宽占用 -> 降低 MTE 压力
  - FP32 中间计算 -> 避免累积误差 -> 保证数值精度
  - 适当的 RoundMode -> 最小化精度损失

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Data Type`（输入数据类型为 FP16/BF16）
  - `Op Type`（算子类型为 Norm/Pooling/Reduce 等涉及累加的算子）
  - `Accuracy Metrics`（精度指标，观察是否有精度损失）
- **如何解读（定性）**：
  - 算子涉及累加、方差、均值等统计计算
  - 输入为 FP16/BF16 但直接在低精度下计算
  - 训练或推理结果出现精度问题（如 loss 异常、精度下降）
  - 使用标签 `O.Norm`, `O.Pooling`, `O.Reduce` 标注算子族

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d), `case2/` (batch_norm_v3), `case3/` (deep_norm)
- **实施步骤（示例性）**：
  1. 分配 FP32 类型的中间缓冲区（`sumBuf`, `castBuf` 等）
  2. 在模板类中使用 `if constexpr (std::is_same_v<T, float>)` 判断数据类型
  3. 对于 FP16/BF16 输入：
     - 使用 `Cast(fp32Buf, fp16Input, RoundMode::CAST_NONE, len)` 提升精度
     - 在 FP32 域进行累加操作 `Add(sumBuf, sumBuf, fp32Buf, len)`
  4. 计算完成后根据输出类型选择合适的 RoundMode Cast 回低精度：
     - FP16: `Cast(fp16Output, fp32Result, RoundMode::CAST_NONE, len)`
     - BF16: `Cast(bf16Output, fp32Result, RoundMode::CAST_RINT, len)`
  5. 使用 `PipeBarrier<PIPE_V>()` 确保 Cast 操作完成

## 6. 约束与副作用 (Constraints)
- **UB 内存开销**：需要额外的 FP32 缓冲区，占用空间为低精度的 2 倍
- **Cast 指令开销**：额外的类型转换操作增加计算开销（通常占比较小）
- **适用场景**：`O.Norm`, `O.Pooling`, `O.Reduce` 等涉及累加的算子族
- **不适用场景**：逐元素操作（`O.Elementwise`）且无累加的场景，低精度计算即可满足精度要求

## 7. 验证逻辑 (Verification)
- **验证原则**：精度指标改善，数值稳定性提升
- **推荐验证项**：
  - `Accuracy Metrics`：精度指标（如 Top-1/Top-5 准确率）与 FP32 baseline 对齐
  - `Numerical Stability`：避免 NaN/Inf 等异常值
  - `Task Duration(us)`：确认性能未明显下降（Cast 开销可控）
- **验证方法**：
  - 对比直接低精度计算 vs 升精度中间计算的精度指标
  - 使用 PyTorch FP32 实现作为 Golden Reference 验证正确性
  - 确认训练收敛性和推理精度满足业务需求

## 标签
- Domain: `U.Vector`, `O.Norm`, `O.Pooling`, `O.Reduce`
- Symptom: `S.LowVecUtil`
- Context: `T.FP16`, `T.BF16`
