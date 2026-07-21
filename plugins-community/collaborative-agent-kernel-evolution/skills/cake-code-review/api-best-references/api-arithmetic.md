# 算术运算 API 优化指南

> **适用场景**：使用算术运算 API（Add/Sub/Mul/Div）时，选择最优实现方式，避免不必要的广播 buffer 和指令开销。

---

## 目录

- [概述](#概述)
- [场景1：标量操作（单行）](#场景1标量操作单行)
  - [方案对比](#方案对比)
  - [API 接口](#api-接口)
  - [完整示例](#完整示例)
- [场景2：广播操作（多行）](#场景2广播操作多行)
  - [方案对比-1](#方案对比-1)
  - [核心原理](#核心原理)
  - [分批处理](#分批处理)
- [性能对比](#性能对比)
- [适用 API](#适用-api)
- [常见错误](#常见错误)

---

## 概述

算术运算 API（Add/Sub/Mul/Div）支持两种使用模式：

| 模式 | API | 适用场景 | Buffer 需求 |
|-----|-----|---------|------------|
| **标量操作** | `Adds/Muls` | 单行处理（Softmax AR 模板） | 32B |
| **广播操作** | `Sub/Div + BinaryRepeatParams` | 多行处理（Softmax ARA 模板） | alignedCols×4 |

**关键优化**：
- 单行：使用 `Adds/Muls` 避免 Duplicate
- 多行：使用 `src1RepStride=0` 避免逐行循环

---

## 场景1：标量操作（单行）

### 方案对比

**问题**：需要对 tensor 每个元素执行 `x - scalar` 或 `x / scalar`

**典型场景**：
- Softmax AR 模板：`x - max_val`（数值稳定）
- Softmax AR 模板：`exp(x) / sum`（归一化）
- LayerNorm：`x - mean`（中心化）
- BatchNorm：`x * gamma + beta`

**方案对比**：

| 方案 | 指令数 | Buffer 需求 | 推荐度 |
|-----|--------|------------|--------|
| Duplicate + Sub | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| Duplicate + Div | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| **Adds(-scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |
| **Muls(1/scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |

### API 接口

**Adds（标量加法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(
    const LocalTensor<T>& dst, 
    const LocalTensor<T>& src, 
    const T& scalarValue, 
    const int32_t& count);

// 功能: dst[i] = src[i] + scalarValue
// 示例: Adds(dst, src, -maxVal, count)  // 减法转加法
```

**Muls（标量乘法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(
    const LocalTensor<T>& dst, 
    const LocalTensor<T>& src, 
    const T& scalarValue, 
    const int32_t& count);

// 功能: dst[i] = src[i] * scalarValue
// 示例: Muls(dst, src, 1.0/sum, count)  // 除法转乘法
```

### 完整示例

#### 优化前（Sub/Div + Duplicate）

```cpp
// Buffer 初始化
uint32_t broadcastBufSize = rLengthAlign * sizeof(T);  // 例如：512B (rLength=128, FP32)
pipe.InitBuffer(broadcastBuf, broadcastBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> broadcastLocal = broadcastBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;
    
    // Step 1: ReduceMax
    ReduceMax<T>(broadcastLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);
    
    // Step 2: Duplicate + Sub（需要广播 buffer）
    T maxVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, maxVal, rLength);  // 指令 1
    Sub<T>(yLocal[rowOffset], xLocal[rowOffset], broadcastLocal, rLength);  // 指令 2
    
    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);
    
    // Step 4: ReduceSum
    ReduceSum<T, true>(broadcastLocal, yLocal[rowOffset], reduceTmpLocal, rLength);
    
    // Step 5: Duplicate + Div（需要广播 buffer）
    T sumVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, sumVal, rLength);  // 指令 3
    Div<T>(yLocal[rowOffset], yLocal[rowOffset], broadcastLocal, rLength);  // 指令 4
}

// 总计：6 条指令/行，需要 broadcastBuf (512B for rLength=128)
```

#### 优化后（Adds/Muls + 标量）

```cpp
// Buffer 初始化（节省 broadcastBuf）
uint32_t scalarBufSize = 32;  // 最小对齐要求，仅需存储 1 个标量
pipe.InitBuffer(scalarBuf, scalarBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> scalarLocal = scalarBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;
    
    // Step 1: ReduceMax
    ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);
    
    // Step 2: Adds（直接标量操作，无需广播）
    T maxVal = scalarLocal.GetValue(0);
    Adds<T>(yLocal[rowOffset], xLocal[rowOffset], -maxVal, rLength);  // 指令 1
    
    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);
    
    // Step 4: ReduceSum
    ReduceSum<T, true>(scalarLocal, yLocal[rowOffset], reduceTmpLocal, rLength);
    
    // Step 5: Muls（除法转乘法，直接标量操作）
    T sumVal = scalarLocal.GetValue(0);
    T invSumVal = (T)1.0 / sumVal;  // CPU 端计算 1/sum
    Muls<T>(yLocal[rowOffset], yLocal[rowOffset], invSumVal, rLength);  // 指令 2
}

// 总计：4 条指令/行，节省 broadcastBuf (480B for rLength=128)
```

---

## 场景2：广播操作（多行）

### 方案对比

**问题**：需要对多行数据执行相同的标量操作（如 `x - max`、`exp / sum`）

**方案对比**：

| 方案 | API 调用 | Buffer 需求 | 推荐度 |
|-----|---------|------------|--------|
| 逐行循环 | R 次 | alignedCols×4 | ⭐⭐ |
| 单次广播（R ≤ 64） | 1 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |
| 分批广播（R > 64） | ceil(R/64) 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |

### 核心原理

**BinaryRepeatParams.src1RepStride=0 实现广播**：

```cpp
struct BinaryRepeatParams {
    uint8_t dstBlkStride;    // 单次迭代内，dst 的 block 步长
    uint8_t src0BlkStride;   // 单次迭代内，src0 的 block 步长
    uint8_t src1BlkStride;   // 单次迭代内，src1 的 block 步长
    uint8_t dstRepStride;    // 相邻迭代间，dst 的 block 步长
    uint8_t src0RepStride;   // 相邻迭代间，src0 的 block 步长
    uint8_t src1RepStride;   // =0 实现广播
};
```

**工作原理**：
- `dstRepStride = alignedCols/8`：每次迭代，dst 前进 `alignedCols` 个元素
- `src0RepStride = alignedCols/8`：每次迭代，src0 前进 `alignedCols` 个元素
- `src1RepStride = 0`：每次迭代，src1 **不前进**，重复读取相同位置

**效果**：
```
迭代 0: dst[0:cols]     = src0[0:cols]     - src1[0:cols]
迭代 1: dst[cols:2cols] = src0[cols:2cols] - src1[0:cols]  ← 重复读取
迭代 2: dst[2cols:3cols]= src0[2cols:3cols]- src1[0:cols]  ← 重复读取
```

### 分批处理

#### 方案1：逐行循环（低效）

```cpp
for (uint32_t r = 0; r < R; r++) {
    Sub(dstLocal[r * alignedCols], srcLocal[r * alignedCols], scalarLocal, alignedCols);
}
// API 调用：R 次
```

#### 方案2：单次广播（高效，R ≤ 64）

```cpp
uint64_t mask = alignedCols;
uint8_t repeatTime = R;

Sub(dstLocal, srcLocal, scalarLocal, mask, repeatTime, 
    {1, 1, 1, alignedCols/8, alignedCols/8, 0});
// API 调用：1 次
// 性能提升：R 倍
```

#### 方案3：分批广播（高效，R > 64）

```cpp
constexpr uint32_t BATCH_SIZE = 64;
uint32_t totalBatches = (R + BATCH_SIZE - 1) / BATCH_SIZE;  // ceil(R/64)

for (uint32_t batch = 0; batch < totalBatches; batch++) {
    uint32_t startRow = batch * BATCH_SIZE;
    uint8_t repeatTime = (startRow + BATCH_SIZE <= R) ? BATCH_SIZE : (R - startRow);
    uint32_t offset = startRow * alignedCols;
    
    Sub(dstLocal[offset], srcLocal[offset], scalarLocal, 
        mask, repeatTime, {1, 1, 1, alignedCols/8, alignedCols/8, 0});
}
// API 调用：ceil(R/64) 次
// 性能提升：约 64 倍
```

---

## 性能对比

### 标量操作（单行）

| 项目 | 优化前 | 优化后 | 改善 |
|-----|--------|--------|------|
| **指令数/行** | 6 条 | 4 条 | **-33%** |
| **Buffer 大小** | 512B (rLength=128) | 32B | **-94%** |
| **UB 节省** | - | ~480B | 可用于更大 rowsPerLoop |

### 广播操作（多行）

| R (行数) | 逐行循环 | 单次广播 | 分批广播 | 性能提升 |
|---------|---------|---------|---------|---------|
| 32 | 32 次 | 1 次 | - | **32×** |
| 64 | 64 次 | 1 次 | - | **64×** |
| 100 | 100 次 | - | 2 次 | **50×** |
| 128 | 128 次 | - | 2 次 | **64×** |
| 200 | 200 次 | - | 4 次 | **50×** |

### 实测示例（Softmax ARA 分支）

**场景**：R=128, alignedCols=64, FP32

| 操作 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| Sub (x-max) | 128 次 | 2 次 | 64× |
| Div (exp/sum) | 128 次 | 2 次 | 64× |
| **总计** | **256 次** | **4 次** | **64×** |

---

## 适用 API

所有支持 `BinaryRepeatParams` 的二元运算 API：

| API | 用途 | 单行优化 | 多行优化 |
|-----|------|---------|---------|
| **Add** | 加法 | Adds | src1RepStride=0 |
| **Sub** | 减法 | Adds(-val) | src1RepStride=0 |
| **Mul** | 乘法 | Muls | src1RepStride=0 |
| **Div** | 除法 | Muls(1/val) | src1RepStride=0 |
| **Max** | 最大值 | - | src1RepStride=0 |
| **Min** | 最小值 | - | src1RepStride=0 |

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|-----|------|---------|
| 编译错误：mask 超限 | `mask > 64` (FP32) | 分批处理或回退循环 |
| 数据错误 | `src1RepStride` 未设置为 0 | 确认参数：`{..., 0}` |
| 部分行正确 | offset 计算错误 | `offset = startRow * alignedCols` |
| 越界崩溃 | repeatTime 计算错误 | 使用三目运算 |
| Buffer 不足 | 使用 Duplicate 方案 | 改用 Adds/Muls |
| dst == tmpBuffer | Reduce API 限制 | 使用不同 buffer |

---

## 检查清单

使用算术运算 API 时，确保：

**标量操作（单行）**：
- [ ] 使用 `Adds(-scalar)` 替代 `Duplicate + Sub`
- [ ] 使用 `Muls(1/scalar)` 替代 `Duplicate + Div`
- [ ] 标量除法转换为乘法（CPU 端计算 1/scalar）

**广播操作（多行）**：
- [ ] alignedCols ≤ 64 (FP32) / ≤ 128 (FP16)
- [ ] 使用 `src1RepStride = 0` 实现广播
- [ ] R > 64 时使用分批处理
- [ ] offset 计算正确：`offset = startRow * alignedCols`

---

## 参考资料

- [BinaryRepeatParams 结构体](../../../asc-devkit/docs/api/context/BinaryRepeatParams.md)
- [Adds API](../../../asc-devkit/docs/api/context/Adds.md)
- [Muls API](../../../asc-devkit/docs/api/context/Muls.md)
- [Sub API](../../../asc-devkit/docs/api/context/Sub.md)
- [Div API](../../../asc-devkit/docs/api/context/Div.md)
