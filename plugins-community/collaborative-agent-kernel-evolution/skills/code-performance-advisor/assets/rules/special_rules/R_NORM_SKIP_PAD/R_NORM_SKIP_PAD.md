# 规则名称：SkipPad 高效跳过 Padding 区域计算

## 1. 需求场景 (Requirement)
- **业务背景**：Norm 类算子为满足对齐要求，UB 中存储的数据包含 Padding（补0）区域。直接对整个对齐后数据进行减法会影响后续方差计算（Padding 区域的 0 被错误计入）。
- **形状/数据类型上下文**：适用于 `O.Norm` 算子族，数据维度不是 32B 整数倍，存在 Padding 区域。

## 2. 模式描述 (Pattern)
- **优化思路**：利用 Vector 指令的 repeat 和 stride 参数，在单个指令中跳过 Padding 区域进行计算，避免逐元素处理或污染统计结果。
- **目标**：高效跳过 Padding 区域，避免污染统计计算结果，保持向量化执行效率。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：逐元素跳过 Padding 需要大量条件判断和分支；直接计算 Padding 区域会污染方差结果；SkipPad 通过 stride 参数一次性跳过，保持向量化。
- **事实桥接**：
  - 向量化跳过 -> 避免分支和循环 -> 提升 5-10 倍性能
  - 精确计算 -> 避免 Padding 污染 -> 保证统计正确性
  - repeat/stride 参数 -> 硬件级支持 -> 最优执行效率

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Op Type`（算子类型为 Norm 族）
  - `Input Shapes`（输入形状，判断是否需要 Padding）
  - `Data Type`（数据类型，影响对齐粒度）
  - `Correctness`（功能正确性，观察是否有统计偏差）
- **如何解读（定性）**：
  - 算子为 BatchNorm/LayerNorm 等 Norm 类
  - 输入维度不是 32B 对齐，存在 Padding
  - 方差计算结果不准确（被 Padding 区域的 0 污染）
  - 使用标签 `O.Norm`, `C.Align.256B` 标注场景

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (batch_norm_v3 SkipPadSubMean)
- **实施步骤（示例性）**：
  1. 计算对齐后的数据长度和 Padding 大小：
     ```cpp
     uint64_t patternR0Align = (patternR0 + alignNum - 1) / alignNum * alignNum;
     uint64_t paddingSize = patternR0Align - patternR0;
     ```
  2. 计算 repeat 和 stride 参数：
     ```cpp
     uint8_t repStride = patternR0Align / BLOCK_SIZE_FOR_FLOAT32;  // 每个 repeat 跨越的 block 数
     uint32_t lineNum = dimN * dimR1;  // 行数
     ```
  3. 使用 repeat/stride 参数跳过 Padding 进行向量化操作：
     ```cpp
     for (int64_t i = 0; i < r0ForLoopNum; i++) {
         Adds(calcTensor[i * ELEM_PER_REP_FP32],
              calcTensor[i * ELEM_PER_REP_FP32],
              -finalMean,
              ELEM_PER_REP_FP32,  // 每次处理的元素数
              lineNum,            // repeat 次数（行数）
              {1, 1, repStride, repStride});  // stride 参数跳过 Padding
     }
     ```
  4. 处理剩余元素（如果 lineNum 超过 UINT8_MAX_NUM）：
     ```cpp
     int64_t repeatForLoopNum = lineNum / UINT8_MAX_NUM;
     for (int64_t i = 0; i < repeatForLoopNum; i++) {
         Adds(calcTensor[...], calcTensor[...], -finalMean,
              r0ForRemainNum, UINT8_MAX_NUM, {1, 1, repStride, repStride});
     }
     ```

## 6. 约束与副作用 (Constraints)
- **代码复杂度**：需要处理多种边界情况（lineNum 超限、剩余元素等）
- **硬件限制**：repeat 参数最大为 255（UINT8_MAX），需要分层处理
- **适用场景**：`O.Norm` 算子族，数据存在 Padding 的场景
- **不适用场景**：数据已对齐或无 Padding 的场景

## 7. 验证逻辑 (Verification)
- **验证原则**：统计计算正确性，向量化执行效率
- **推荐验证项**：
  - `Correctness`：方差计算结果与 PyTorch baseline 对齐
  - `Task Duration(us)`：相比逐元素处理性能提升 5-10 倍
  - `Vector Utilization`：保持高向量化执行效率
- **验证方法**：
  - 对比 SkipPad vs 逐元素处理 vs 直接计算（含 Padding）的正确性和性能
  - 使用多种 Shape 验证边界情况处理的正确性
  - 确认不同 lineNum 场景下的性能表现

## 标签
- Domain: `U.Vector`, `O.Norm`
- Symptom: `S.LowVecUtil`, `S.ScalarBound`
- Context: `C.Align.256B`
