# 流水线同步机制指南

MTE 与 Vector 同步的核心机制。

---

## 目录

1. [核心问题](#核心问题)
2. [解决方案](#解决方案)
3. [两种方案对比](#两种方案对比)
4. [完整流水线模板](#完整流水线模板)
5. [调试技巧](#调试技巧)

---

## 核心问题

**DataCopy/DataCopyPad 是异步 DMA 操作，直接在搬运后的数据上做 Vector 计算可能读到未完成的数据！**

### 硬件架构

```
GM → MTE2 (异步) → UB → Vector (同步) → MTE3 (异步) → GM
```

### 问题场景

```cpp
// ❌ 错误：DataCopyPad 后直接使用数据
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);  // 可能读到未完成搬运的数据！
```

**现象**：输出数据随机、错误

---

## 解决方案

### 方案一：EnQue/DeQue 队列同步（推荐）

**原理**：TQue 的 EnQue/DeQue 机制自动提供硬件同步点。

```cpp
// ✅ 正确：使用 EnQue/DeQue 同步
// Step 1: CopyIn - MTE2 搬运
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);                    // 标记"就绪"

// Step 2: Compute - Vector 计算
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2 完成
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: CopyOut - MTE3 搬运
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 Vector 完成
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyOutParams);
outQueueY.FreeTensor(yOut);
```

**关键点**：
- `EnQue(xLocal)` 标记 buffer 数据就绪
- `DeQue<float>()` 阻塞等待数据就绪
- DeQue 返回后，数据一定已经搬运完成

### 方案二：PipeBarrier 手动同步

```cpp
// ✅ 可用：使用 PipeBarrier 同步
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
AscendC::PipeBarrier<PIPE_ALL>();          // 等待 MTE2 完成

AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);

AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyOutParams);
AscendC::PipeBarrier<PIPE_ALL>();          // 等待 MTE3 完成
```

**缺点**：性能开销大（全流水线停顿），不推荐用于高性能场景

---

## 两种方案对比

| 特性 | EnQue/DeQue | PipeBarrier |
|-----|-------------|-------------|
| 同步粒度 | buffer 级别 | 全流水线 |
| 性能 | 高（支持并行） | 低（串行等待） |
| 代码复杂度 | 需要队列管理 | 简单直接 |
| 推荐程度 | ⭐⭐⭐⭐⭐ | ⭐⭐（仅调试用） |

### EnQue/DeQue 的双重作用

1. **队列管理**：Double Buffer 场景下管理多 buffer 轮转
2. **硬件同步**：提供 MTE ↔ Vector 之间的同步点

```cpp
// EnQue/DeQue 不仅仅是"队列"，更重要的是同步机制
inQueueX.EnQue(xLocal);    // 1. 标记数据就绪  2. 通知硬件可以等待
xLocal = inQueueX.DeQue(); // 1. 阻塞等待就绪  2. 获取可用 buffer
```

---

## 完整流水线模板

```cpp
__aicore__ inline void ProcessTile(uint32_t tileIdx)
{
    // ========== CopyIn 阶段 ==========
    // MTE2: GM → UB（异步）
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopyPad(xLocal, xGm[tileIdx * tileSize], copyParams, padParams);
    inQueueX.EnQue(xLocal);              // 同步点：标记就绪
    
    // ========== Compute 阶段 ==========
    // Vector: UB 计算（同步，需等待 MTE2）
    AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 同步点：等待 MTE2
    AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    AscendC::Adds<float>(yLocal, xIn, 1.0f, tileSize);
    outQueueY.EnQue(yLocal);             // 同步点：标记就绪
    inQueueX.FreeTensor(xIn);
    
    // ========== CopyOut 阶段 ==========
    // MTE3: UB → GM（异步，需等待 Vector）
    AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 同步点：等待 Vector
    AscendC::DataCopyPad(yGm[tileIdx * tileSize], yOut, copyParams);
    outQueueY.FreeTensor(yOut);
}
```

### 流水线时序图

```
时间 →
        
Tile 0:  [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                      ↑ DeQue等待    ↑ DeQue等待
Tile 1:          [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                  ↑ 并行！    ↑ DeQue等待    ↑ DeQue等待

关键：DeQue 阻塞等待上一个阶段的异步操作完成
```

---

## 调试技巧

### 检查缺少 EnQue/DeQue

```cpp
// ❌ 错误：AllocTensor 后直接用
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 错！可能读到未完成搬运的数据

// ✅ 正确：DeQue 后再计算
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

### 临时加 PipeBarrier 调试

```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // 临时加，如果结果正确说明是同步问题
Compute(x);
```

**如果 PipeBarrier 能解决问题，说明是同步问题** → 修复方案：改为 EnQue/DeQue 机制

### 常见误区

| 误区 | 正确理解 |
|-----|---------|
| AllocTensor 后数据就可用 | AllocTensor 只分配内存，不等待搬运 |
| DataCopy 是同步的 | DataCopy 是异步 DMA，立即返回 |
| 不用 EnQue/DeQue 也能正常工作 | 必须用 EnQue/DeQue 或 PipeBarrier 同步 |
| PipeBarrier 性能好 | PipeBarrier 是全流水线停顿，性能差 |

---

## CrossCore 核间同步（分离模式 ISASI）

**适用场景**：分离模式（ISASI）下 AIC（Cube核）与 AIV（Vector核）之间的同步，典型用于 CV 融合算子。

### 函数原型

```cpp
// Set：发送同步信号
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

// Wait：等待同步信号
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

### modeId 取值

| modeId | 含义 | 典型场景 |
|--------|------|---------|
| `0x0`  | 跨 AI Core（核间）同步，所有 AIC 或所有 AIV 互相等待 | 多核广播屏障（需 SetScheduleMode batchmode） |
| `0x1`  | 同一 AI Core 内，AIV 子核之间同步 | AIV0↔AIV1 |
| `0x2`  | 同一 AI Core 内，AIC↔AIV 双向同步 | **CV 融合主流模式**：AIC 做完通知 AIV，AIV 做完通知 AIC |
| `0x4`  | AIC↔AIV 单向独立触发（AIV0/AIV1 各自触发 AIC 等待） | 仅 Ascend 950PR/950DT |

### flagId 约束

- **Atlas A2/A3**：取值范围 0–10，同一 flagId 计数器最多 Set **15 次**
- **不能与 Matmul 高阶 API 混用**：Matmul 内部已使用本接口进行核间同步，混用会导致 flagId 冲突

### CV 融合标准模式（modeId=2）

```cpp
// AIC 侧（Cube 核）：做完矩阵计算后通知 AIV
if (g_coreType == AscendC::AIC) {
    // ...矩阵计算（Matmul 低阶 API）...
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);   // 通知 AIV 可以开始后处理
    AscendC::CrossCoreWaitFlag(0x1);                  // 等待 AIV 后处理完成
}

// AIV 侧（Vector 核）：等待 AIC，做后处理，再通知 AIC
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x0);                  // 等待 AIC 矩阵计算完成
    // ...Vector 后处理（Epilogue）...
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);  // 通知 AIC 后处理完成
}
```

### SetFlag 的 pipe 参数选择

| pipe 类型 | 含义 | 何时使用 |
|-----------|------|---------|
| `PIPE_FIX` | 标量/控制流水 | AIC 侧发送通知（矩阵计算完成） |
| `PIPE_MTE3` | MTE3（UB→GM）流水 | AIV 侧在 DataCopy 后发送通知 |
| `PIPE_V` | Vector 流水 | AIV 侧在纯 Vector 计算后发送通知 |

### 常见错误

| 错误 | 正确做法 |
|------|---------|
| 与 Matmul 高阶 API 同时使用 CrossCoreSetFlag | 改用 Matmul 低阶 API，手动管理 CrossCore 同步 |
| flagId 超出 0-10 范围（A2/A3） | flagId 限定在 0–10 |
| 模式 0 在多流场景不加 SetScheduleMode | 必须开启 batchmode，否则可能死锁 |
| AIC 侧 pipe 用 PIPE_MTE3 | AIC 做完矩阵计算应用 PIPE_FIX |
