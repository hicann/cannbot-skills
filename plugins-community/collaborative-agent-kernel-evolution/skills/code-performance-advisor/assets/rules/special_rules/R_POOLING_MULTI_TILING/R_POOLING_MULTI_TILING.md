# 规则名称：Pooling 多策略 UB Tiling 自适应选择

## 1. 需求场景 (Requirement)
- **业务背景**：Pooling 算子的输入形状多样化（Channel 大小、Spatial 维度、下采样比例等变化大），单一 Tiling 策略无法针对所有场景优化 UB 利用率和计算效率。
- **形状/数据类型上下文**：适用于 `O.Pooling` 算子族，特别是输入形状变化范围大的场景。

## 2. 模式描述 (Pattern)
- **优化思路**：根据输入输出形状关系和 UB 容量，设计多种 UB Tiling 策略（如 Split-C、Split-W、Multi-W），在 Tiling 阶段根据形状特征动态选择最优策略。
- **目标**：针对不同输入形状选择最优 Tiling 策略，最大化 UB 利用率和计算效率。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：不同形状特征适合不同的 Tiling 策略，动态选择可避免 UB 浪费或频繁搬运。
- **事实桥接**：
  - 形状自适应 -> 选择最优 Tiling -> 最大化 UB 利用率
  - 策略多样化 -> 覆盖更多场景 -> 性能提升 20-50%
  - Tiling Key 分发 -> 编译期确定策略 -> 零运行时开销

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Op Type`（算子类型为 Pooling 族）
  - `Input Shapes`（输入形状，判断 C/H/W 维度关系）
  - `Output Shapes`（输出形状，判断下采样比例）
  - `aic_mte2_ratio`（搬运占比，单一策略可能搬运频繁）
  - `Task Duration(us)`（整体耗时）
- **如何解读（定性）**：
  - 算子为 AdaptivePooling/MaxPool/AvgPool 等 Pooling 类
  - 输入形状变化范围大（C 从小到大，Spatial 维度变化大）
  - `aic_mte2_ratio` 较高或 UB 利用率低
  - 使用标签 `O.Pooling`, `C.UB.Capacity` 标注场景

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d 三策略 Tiling)
- **实施步骤（示例性）**：
  1. 计算 UB 可用容量和对齐参数：
     ```cpp
     int32_t dataTypeSize = (dataTypeKey == FP32) ? 4 : 2;
     int32_t needCast = (dataTypeKey == FP32) ? 0 : 1;  // 是否需要升精度
     uint64_t alignNum = BLOCK_SIZE / dataTypeSize;
     uint64_t tileLen = ubSize / (2 * dataTypeSize + sizeof(float) * (1 + needCast)) / alignNum * alignNum;
     uint64_t alignC = (dimC + alignNum - 1) / alignNum * alignNum;
     ```
  2. 策略 A: Split-C（Channel 维度分块）适合 C 大但输出空间小：
     ```cpp
     uint64_t doubleC = 2 * alignC;
     if (doubleC > tileLen) {
         mode = MODE_SPLIT_C;
         cTileLength = alignC > tileLen ? tileLen : alignC;
         return;
     }
     ```
  3. 策略 B: Split-W（宽度维度分块）适合空间下采样较大：
     ```cpp
     uint64_t inputTileNum = (ubSize / alignC - dataTypeSize - sizeof(float) * (1 + needCast)) / dataTypeSize;
     if (inputTileNum < maxWindowWLength) {
         mode = MODE_SPLIT_W;
         return;
     }
     ```
  4. 策略 C: Multi-W（多窗口并行）同时计算多个输出窗口最大化数据复用：
     ```cpp
     mode = MODE_MULTI_W;
     uint64_t windowWNum = (ubSize / alignC - sizeof(float) * needCast) /
                           ((maxWindowWLength + 1) * dataTypeSize + sizeof(float));
     windowWNum = windowWNum < outW ? windowWNum : outW;
     ```
  5. 设置 Tiling Key 编码策略和数据类型：
     ```cpp
     uint32_t tilingKey = modeKey * 10 + dataTypeKey;  // 如: 10/11/12, 20/21/22, 30/31/32
     context->SetTilingKey(tilingKey);
     ```
  6. 在 Kernel 入口根据 Tiling Key 分发到不同模板实例

## 6. 约束与副作用 (Constraints)
- **代码复杂度**：需要维护多种 Tiling 策略和对应的 Kernel 实现
- **Tiling 逻辑**：策略选择逻辑复杂，需要精确计算 UB 容量
- **适用场景**：`O.Pooling` 算子族，输入形状变化范围大的场景
- **不适用场景**：输入形状固定或变化范围小的场景

## 7. 验证逻辑 (Verification)
- **验证原则**：不同形状场景下性能均衡提升
- **推荐验证项**：
  - `Task Duration(us)`：期望呈显著下降趋势（20-50%，取决于形状）
  - `UB Utilization`：UB 利用率提升
  - `aic_mte2_ratio`：搬运占比下降（减少频繁搬运）
- **验证方法**：
  - 对比单一策略 vs 多策略的性能数据
  - 使用多种典型 Shape 组合（大 C/小 C，大 Spatial/小 Spatial 等）验证
  - 确认每种策略在其适用场景下都能获得性能提升

## 标签
- Domain: `U.Vector`, `O.Pooling`
- Symptom: `S.LowVecUtil`, `S.MteBusy`
- Context: `C.UB.Capacity`, `C.K.Small`, `C.K.Large`
