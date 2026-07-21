# 规则名称：Pooling IndexBuffer 预计算与复用

## 1. 需求场景 (Requirement)
- **业务背景**：Pooling 类算子（特别是 Adaptive Pooling）需要根据输出索引计算对应的输入区域索引，涉及大量除法和取模运算，标量运算成为瓶颈。
- **形状/数据类型上下文**：适用于 `O.Pooling` 算子族，特别是 Adaptive Pooling（输入输出尺寸不成整数倍关系）。

## 2. 模式描述 (Pattern)
- **优化思路**：将输入索引计算提前到 Scalar 阶段，预计算一批输出点的索引信息并缓存到 IndexBuffer 中，在后续 ReduceMean 阶段复用，避免重复的除法/取模运算。
- **目标**：减少 Kernel 中复杂的标量运算（除法/取模），提升向量化计算效率。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：除法和取模运算在 Scalar 单元执行慢，IndexBuffer 预计算将这些运算提前完成，后续仅需简单的 GetValue 操作即可获取索引。
- **事实桥接**：
  - 预计算索引 -> 减少运行时除法/取模 -> 降低 Scalar 瓶颈
  - IndexBuffer 复用 -> 避免重复计算 -> 提升 5-15% 性能
  - 批量预计算 -> 利用 Scalar 并行度 -> 隐藏索引计算延迟

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Op Type`（算子类型为 Pooling 族）
  - `aic_scalar_ratio` / `S.ScalarBound`（标量占比高）
  - `S.HighScalarRatio`（标量运算占比高）
  - `Task Duration(us)`（整体耗时）
- **如何解读（定性）**：
  - 算子为 AdaptiveAvgPool/AdaptiveMaxPool 等 Adaptive Pooling 类
  - `aic_scalar_ratio` 较高，表明标量运算是瓶颈
  - 输入输出尺寸不成整数倍，需要大量索引计算
  - 使用标签 `O.Pooling`, `S.ScalarBound` 标注场景

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d IndexBuffer)
- **实施步骤（示例性）**：
  1. 分配 IndexBuffer，包含 6 个 buffer 存储 D/H/W 维度的起止索引：
     ```cpp
     struct IndexBuffer {
         TBuf<QuePosition::VECCALC> startDIndexBuf;
         TBuf<QuePosition::VECCALC> endDIndexBuf;
         TBuf<QuePosition::VECCALC> startHIndexBuf;
         TBuf<QuePosition::VECCALC> endHIndexBuf;
         TBuf<QuePosition::VECCALC> startWIndexBuf;
         TBuf<QuePosition::VECCALC> endWIndexBuf;
     };
     ```
  2. 实现预计算函数，批量计算一批输出点的索引：
     ```cpp
     __aicore__ inline void CalculateIndex(
         IndexBuffer& indexBuf, PoolShape& inputShape, PoolShape& outputShape, int64_t offset, int64_t len)
     {
         LocalTensor<int64_t> startDIndexLocal = indexBuf.startDIndexBuf.Get<int64_t>();
         // ... 获取其他 5 个 buffer
         Index index;
         for (int64_t i = offset, j = 0; i < offset + len; ++i, ++j) {
             OutputOffsetToInputIndex(i, outputShape, inputShape, index);  // 执行除法/取模
             startDIndexLocal.SetValue(j, index.dstart);
             endDIndexLocal.SetValue(j, index.dend);
             // ... 存储其他索引
         }
     }
     ```
  3. 在 ReduceMean 阶段从 IndexBuffer 获取索引：
     ```cpp
     __aicore__ inline void GetIndexFromBuffer(IndexBuffer& indexBuf, int64_t bufIdx, Index& index)
     {
         index.dstart = indexBuf.startDIndexBuf.Get<int64_t>().GetValue(bufIdx);
         index.dend = indexBuf.endDIndexBuf.Get<int64_t>().GetValue(bufIdx);
         // ... 获取其他索引
     }
     ```
  4. 插入 `SToVSync()` 确保索引已计算完成：
     ```cpp
     GetIndexFromBuffer(indexBuf, bufIdx, index);
     SToVSync();  // 确保索引已准备好
     ```

## 6. 约束与副作用 (Constraints)
- **UB 内存开销**：需要额外的 IndexBuffer 存储索引（6 个 int64 buffer）
- **批处理粒度**：`indexBufLen` 需要根据 UB 容量调整
- **适用场景**：`O.Pooling` 算子族，`S.ScalarBound`, `S.HighScalarRatio`（标量瓶颈场景）
- **不适用场景**：规则 Pooling（输入输出尺寸成整数倍），索引计算简单的场景

## 7. 验证逻辑 (Verification)
- **验证原则**：标量占比下降，整体性能提升
- **推荐验证项**：
  - `Task Duration(us)`：期望呈下降趋势（5-15%）
  - `aic_scalar_ratio`：期望呈显著下降趋势
  - `S.HighScalarRatio`：标量占比降低
- **验证方法**：
  - 对比有无 IndexBuffer 的性能数据
  - 使用不同输入输出尺寸组合验证收益
  - 确认 UB 内存使用未超限

## 标签
- Domain: `U.Vector`, `O.Pooling`
- Symptom: `S.ScalarBound`, `S.HighScalarRatio`
- Context: `C.UB.Capacity`
