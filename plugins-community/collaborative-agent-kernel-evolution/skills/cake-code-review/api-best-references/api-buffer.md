# UB 缓冲区管理指南

TBuf/TQue 选择、Double Buffer 流水线并行、批量搬运模式。

---

## 目录

1. [TBuf vs TQue 选择](#tbuf-vs-tque-选择)
2. [Double Buffer 流水线并行](#double-buffer-流水线并行)
3. [批量搬运 + 逐行计算模式](#批量搬运--逐行计算模式)

---

## TBuf vs TQue 选择

### 选择原则

| 场景 | 推荐类型 | depth | 原因 |
|-----|---------|-------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | 2 | 需要与 Vector 计算并行 |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 1 | 不涉及 MTE 搬运，串行即可 |
| 需要多 buffer 轮转 | `TQue` (depth≥2) | 2+ | Double Buffer 并行 |
| 单 buffer 临时存储 | `TBuf` | 1 | 简单直接 |

### 正确用法

```cpp
// TQue：需要队列管理（MTE 搬运相关）
AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX;  // depth=2 for Double Buffer
pipe->InitBuffer(inQueueX, 2, bufferSize);

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopy(x, xGm, size);
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);

// TBuf：纯计算缓冲区
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

AscendC::LocalTensor<float> work = workBuf.Get<float>();
// 直接使用，无需 EnQue/DeQue
```

---

## Double Buffer 流水线并行

### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

### 硬件原理

- **MTE2**：搬运工，GM → UB
- **Vector**：加工员，计算
- **MTE3**：搬运工，UB → GM

### 时间线对比

**无 Double Buffer（串行）**：
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**有 Double Buffer（并行）**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                 ↑ MTE2与Vector并行！
```

### 实现原则

| Buffer 类型 | 深度 | 原因 |
|------------|------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | 与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | 与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | **1** | 不涉及 MTE 搬运 |

### 正确用法

```cpp
// 1. Init: depth=2 是关键
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf,   1, workSize * sizeof(T));

// 2. Process: 单循环结构，TQue 自动轮转
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 异步搬运
    Compute(i);  // Vector 计算
    CopyOut(i);  // MTE3 异步搬出
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopy(x, xGm[i * tileSize], tileSize);
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopy(yGm[i * tileSize], y, tileSize);
    outQueueY.FreeTensor(y);
}
```

### 为什么能并行？

| 操作 | 特性 |
|-----|------|
| `DataCopy` | 异步 DMA，立即返回 |
| `EnQue` | 非阻塞，标记就绪 |
| `DeQue` | 阻塞，等待就绪 |

### 常见误区

| 误区 | 正确理解 |
|-----|---------|
| 需要手动拆成 Ping/Pong 两套代码 | 单循环 + depth=2，TQue 自动管理 |
| 每行数据需要2块内存 | 队列有2块 buffer 轮流使用 |
| depth 越大越好 | depth=2 通常性价比最高 |
| 所有 buffer 都要 depth=2 | 只有涉及 MTE 搬运的才需要 |

---

## 批量搬运 + 逐行计算模式

### 适用场景

处理多行数据时，批量搬运减少 MTE2/MTE3 调用次数，充分利用带宽。

### 模式结构

```
CopyInBatch(N行) → 逐行计算(N次) → CopyOutBatch(N行)
```

### 代码模板

```cpp
__aicore__ inline void ProcessBatch()
{
    uint32_t totalRowsToProcess = endRow - startRow;
    if (totalRowsToProcess == 0) return;
    
    for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
        uint32_t startLocalRow = tile * tileRows;
        
        // 边界检查：防止 uint32_t 下溢
        if (startLocalRow >= totalRowsToProcess) break;
        
        uint32_t remaining = totalRowsToProcess - startLocalRow;
        uint32_t rowsThisTile = (remaining < tileRows) ? remaining : tileRows;
        
        CopyInBatch(startLocalRow, rowsThisTile);
        ComputeBatch(rowsThisTile);
        CopyOutBatch(startLocalRow, rowsThisTile);
    }
}
```

### Host 侧 Tiling 计算

```cpp
// A2/A3 UB = 192KB
constexpr uint64_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;  // DataCopyPad blockCount 限制

// bytesPerTileRow: double buffer (in*2 + out*2)
uint32_t bytesPerTileRow = paddedColsT * typeSizeBytes * 4;

// tileRows
uint32_t tileRows = (UB_SIZE - overheadBytes) / bytesPerTileRow;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

### 注意事项

1. **tileRows 限制**：DataCopyPad 的 `blockCount` 最大 4095
2. **尾核处理**：`startLocalRow >= totalRowsToProcess` 时提前退出
3. **stride 计算**：UB 侧 stride 单位是 32 字节块，GM 侧是字节
