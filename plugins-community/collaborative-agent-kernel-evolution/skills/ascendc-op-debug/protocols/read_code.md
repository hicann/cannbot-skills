# Protocol: Layer 1 — 代码阅读策略

> 适用于 evidence=code 的 hypothesis（H01-H10, H13）。
> CC 应按以下顺序和重点阅读源文件，定位 bug。

---

## 必读文件清单

### 1. `op_host/{op_name}_custom.cpp` — Tiling / Host 端

| 阅读位置 | 关注点 | 对应假设 |
|---|---|---|
| `InferShapeAndType` | 输出 tensor 最内维 < 8 float？ | H01 |
| workspace 计算块 | 引用当前 D/N 还是 maxD/maxN？ | H02 |
| `SetWorkspaceSize` | 总大小用 coreNumAiv 还是 usedCoreNum？ | H02 |
| dimTile/dimLoop 计算 | 是否动态推导？lastDimTile 是否处理 | H07 |
| `SetBlockDim` | 是否 `min(totalTokens, GetCoreNum())`？ | H08 |
| rowsPerCore 计算 | 是否有 blockPivot 余数分配？ | H09 |

**关键 grep 命令：**
```bash
grep -n "SetDim\|workspace\|coreWs\|dimTile\|dimLoop\|lastDim\|SetBlockDim\|rowsPerCore\|blockPivot" \
     op_host/{op_name}_custom.cpp
```

### 2. `kernel/{op_name}.cpp` — Kernel / Device 端

| 阅读位置 | 关注点 | 对应假设 |
|---|---|---|
| `Init` 函数开头 | `if (blockIdx >= usedCoreNum) return`？ | H08 |
| `Init` 中 tokenStart/End 计算 | blockPivot 分支是否正确？ | H09 |
| `pipe_.InitBuffer` 调用 | 参数是 dimTile 还是 D？ | H06 |
| DataCopy 调用 | 第 1 个参数是 dst（Local），第 2 个是 src？ | H10 |
| EnQue/DeQue 配对 | 每个 EnQue 有对应的 DeQue？ | H10 |
| for-tile 循环外部 | `float sumSq = 0.0f` 在循环外初始化？ | H03 |
| ReduceSum 之后 | `+= scBuf.GetValue(0)` 用 scalar 累加？ | H03 |
| DataCopyPad 到 workspace | dimLoop>1 时才写？wsBase 计算对？ | H04 |
| matmul 索引 | `combVals[j*N+h]` 还是 `combVals[h*N+j]`？ | H05 |
| lastTile 处理 | `tD = (t==dimLoop-1) ? lastDimTile : dimTile`？ | H07 |
| SetFlag/WaitFlag | 数量配对？event_id 是否复用？ | H11, H12 |

**关键 grep 命令：**
```bash
grep -n "InitBuffer\|DataCopy\|EnQue\|DeQue\|PipeBarrier\|SetFlag\|WaitFlag\|\
GetValue(0)\|sumSq\|wsBase\|blockIdx.*usedCore\|tokenStart\|tokenEnd" \
     kernel/{op_name}.cpp
```

---

## 快速诊断检查清单

在开始深入分析前，先快速检查以下 6 个最高频问题：

- [ ] **输出最内维** `N × sizeof(dtype) >= 32`？（H01）
- [ ] **workspace** 用 `maxD/maxN`？用 `coreNumAiv`？（H02）
- [ ] **InitBuffer** 参数是 `dimTile` 不是 `D`？（H06）
- [ ] **DataCopy(dst=Local, src=GM)** 方向正确？（H10）
- [ ] **scalar 累加变量**在 for-tile 循环**外部**初始化？（H03）
- [ ] **空闲核检查** `if (blockIdx >= usedCoreNum) return`？（H08）

---

## DumpTensor 插桩策略（精度 bug 定位）

当 Layer 1 代码审查未能定位时，使用二分插桩：

```cpp
#include "kernel_log.h"

// 在 Process 函数各阶段末尾插入
AscendC::DumpTensor<float>(inputLocal,  0, 16);  // 加载后
AscendC::DumpTensor<float>(tmpBuf,      0, 16);  // 计算中间
AscendC::DumpTensor<float>(outputLocal, 0, 16);  // 写出前
```

**策略**：二分法——注释掉后半段计算，确认前半段正确后，逐步恢复。

**注意**：DumpTensor 只能打印 LocalTensor（UB），需要先 DataCopy 才能查看 GM 数据。生产代码必须删除所有 DumpTensor。
