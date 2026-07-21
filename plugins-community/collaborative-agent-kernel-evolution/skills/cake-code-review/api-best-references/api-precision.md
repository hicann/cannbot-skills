# 精度转换与混合精度指南

Cast API 使用规范和混合精度计算模式。

---

## 目录

1. [Cast RoundMode 选择](#cast-roundmode-选择)
2. [混合精度计算模式（FP16 输入）](#混合精度计算模式fp16-输入)
3. [完整代码模板](#完整代码模板)

---

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| **int8_t → half** | `CAST_NONE` | 整数→浮点，无精度损失 |
| **half → int8_t** | `CAST_ROUND` | 浮点→整数，需要舍入 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.AllocTensor<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

---

## 完整代码模板

### 内存分配

```cpp
// FP16 模式需要额外的 FP32 buffer
pipe->InitBuffer(inQueueX, 2, tileRows * paddedColsT * sizeof(half));
pipe->InitBuffer(outQueueY, 2, tileRows * paddedColsT * sizeof(half));
pipe->InitBuffer(workBufFp32, 1, paddedColsFp32 * sizeof(float));  // 单行 FP32
pipe->InitBuffer(reduceBuf, 1, reduceBufSize * sizeof(float));
```

### 计算模板

```cpp
__aicore__ inline void ComputeBatchFp16(uint32_t rowsThisTile)
{
    LocalTensor<half> xLocalHalf = inQueueX.DeQue<half>();
    LocalTensor<half> yLocalHalf = outQueueY.AllocTensor<half>();
    LocalTensor<float> xLocal = workBufFp32.AllocTensor<float>();
    LocalTensor<float> tmpReduce = reduceBuf.AllocTensor<float>();

    for (uint32_t r = 0; r < rowsThisTile; r++) {
        LocalTensor<half> rowIn = xLocalHalf[r * paddedColsT];
        LocalTensor<half> rowOut = yLocalHalf[r * paddedColsT];

        // Step 1: half → float（低→高精度）
        AscendC::Cast<float, half>(xLocal, rowIn, AscendC::RoundMode::CAST_NONE, cols);

        // Step 2: 在 FP32 上计算（如 Softmax）
        SoftmaxRowFp32(xLocal, xLocal, tmpReduce);

        // Step 3: float → half（高→低精度）
        AscendC::Cast<half, float>(rowOut, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
    }

    reduceBuf.FreeTensor(tmpReduce);
    workBufFp32.FreeTensor(xLocal);
    outQueueY.EnQue(yLocalHalf);
    inQueueX.FreeTensor(xLocalHalf);
}

__aicore__ inline void SoftmaxRowFp32(
    LocalTensor<float>& input,
    LocalTensor<float>& output,
    LocalTensor<float>& tmpReduce)
{
    AscendC::ReduceMax<float>(tmpReduce, input, tmpReduce, cols, false);
    float maxValue = tmpReduce.GetValue(0);
    
    AscendC::Adds<float>(output, input, -maxValue, cols);
    AscendC::Exp<float>(output, output, cols);
    
    AscendC::ReduceSum<float>(tmpReduce, output, tmpReduce, cols);
    float sumValue = tmpReduce.GetValue(0);
    
    float invSumValue = 1.0f / sumValue;
    AscendC::Muls<float>(output, output, invSumValue, cols);
}
```

### RoundMode 选择摘要

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **int8_t → half** | `CAST_NONE` | 整数→浮点，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，需要舍入 |
| **half → int8_t** | `CAST_ROUND` | 浮点→整数，需要舍入 |

---

## 量化算子精度技巧（DynamicQuant 验证）

### 1. 避免双重舍入（Double-Rounding）

**场景**：FP32 → FP16 → INT8 两段 Cast 路径。

**问题**：若两步都用 `CAST_ROUND`，FP16 中间值已发生一次舍入，INT8 再次舍入会产生错误：
```
FP32 38.4999... → FP16 CAST_ROUND → 38.5 → INT8 CAST_ROUND → 39（错误，应为 38）
```

**正确做法**：先在 FP32 空间显式 Round，再用 `CAST_NONE` 精确转换：
```cpp
// ✅
AscendC::Round(quantFp32, quantFp32, tileLength);                                // 单次舍入
AscendC::Cast(quantFp16, quantFp32, AscendC::RoundMode::CAST_NONE, tileLength); // 精确
AscendC::Cast(outInt8,   quantFp16, AscendC::RoundMode::CAST_NONE, tileLength); // 精确
```

> 对应假设 H26。`AscendC::Round` 是向量取整指令，见 `api-restrictions.md §1.1`。

---

### 2. 元素除以标量：`Div` vs `Muls(1/scale)`

**问题**：`Muls(quantFp32, rowFp32, 1.0f/scale)` 先计算倒数近似（FP32 精度损失），再乘法，与参考实现的 `x/scale` 不等价，引入 1-2 ULP 额外误差。

**正确做法**：
```cpp
// ✅ 直接除法，与 PyTorch x/scale 结果最接近
AscendC::Duplicate(sharedLocal, rowScale, tileLength);
AscendC::Div(quantFp32, rowFp32, sharedLocal, tileLength);
```

---

### 3. ReduceMax `dst ≠ tmpBuffer`（强制约束）

`AscendC::ReduceMax(dst, src, tmpBuffer, count)` 要求 `dst` 和 `tmpBuffer` **不能是同一块内存**：

```cpp
// ❌ 错误：dst == tmpBuffer == sharedLocal
AscendC::ReduceMax(sharedLocal, absLocal, sharedLocal, tileLength);

// ✅ 正确：独立 tmpBuffer
AscendC::ReduceMax(sharedLocal, absLocal, quantFp32, tileLength);
```

> 同规则见 `api-restrictions.md §2`（Reduce API 限制）和假设 H19。

---

### 4. 避免 Magic Number 取整（910B 不适用）

**尝试**：`Adds(quantFp32, quantFp32, 8388608.0f)` + `Subs(...)` 模拟 IEEE 754 截断取整。

**结果**：910B AIV 向量单元对大标量立即数（≥ 2^23）的 `Adds` 行为与 x86/CPU 不同，误差从 5787 急增至 72,923。

**结论**：优先使用 AscendC 文档中明确列出的 `AscendC::Round` 指令，不要移植 CPU Magic Number 技巧。

---

## 双缓冲下 TBuf MTE3 隔离原则

**场景**：使用 `depth≥2` 软件双缓冲，且某 `TBuf` 同时满足：
1. 在 `CopyOut` 函数中通过 `DataCopyPad`（MTE3）读取
2. 在 `Compute` 函数的 VEC 指令中被覆写

**问题根因**：双缓冲时 `CopyIn(i+1)` 提前发出，`Compute(i+1)` 启动时 MTE2 可能已完成，导致 VEC 写操作与 MTE3 读操作并发，形成数据竞争。

```
CopyOutScale(i):  DataCopyPad(scaleGm[i], sharedUb[0])  ← MTE3 in-flight，读 sharedUb[0]
Compute(i+1):     ReduceMax(..., sharedUb[0], ...)       ← VEC 写 sharedUb[0]
                  ↑ 若先于 MTE3 完成，scale_i 被污染
```

**修复**：使用专用 `scaleBuf`，与 `Compute` 的工作 buf 完全隔离：
```cpp
// 新增成员
AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf;

// Init 中
pipe.InitBuffer(scaleBuf, sizeof(float));

// CopyOutScale 只访问 scaleBuf，Compute 从不访问 scaleBuf
__aicore__ inline void CopyOutScale(uint32_t rowIdx)
{
    AscendC::LocalTensor<float> scaleTensor = scaleBuf.Get<float>();
    scaleTensor.SetValue(0, rowScale);
    AscendC::DataCopyPad(scaleGm[rowIdx], scaleTensor,
                         {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
}
```

**规则**：凡 `TBuf` 上调用 `DataCopyPad`（MTE3），且该 `TBuf` 在后续迭代 VEC 指令中会被覆写，**必须独立出专用 buf，禁止与 Compute 共享**。
