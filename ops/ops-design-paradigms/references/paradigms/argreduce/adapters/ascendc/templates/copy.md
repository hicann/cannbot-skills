# COPY 场景模板（AscendC）

> **层级**：AscendC 适配层模板。模式 COPY_ONLY（10003），R==1 退化：无选择，values 透传、indices 全 0。完整实现见 [../example/](../example/) 的 `arg_max_with_value_copy.h`。

## 1 Tiling 侧

```cpp
// R == 1：无归约。多核按输出总量切（纯搬运，带宽型）——切分基数 = sumSize = A1×A0
//（对齐工业实现 CalcSplitInfoForCopyOnly：blkFactor 按 sumSize 均衡，非按 A1 单维）
realCoreNum_ = min(sumSize /*= A1×A0*/, coreNum_);
// 小输出单核门槛：总字节 < 64KB 时单核（省多核 launch 开销，工业实现常量）
if (sumSize * eb < 65536) { realCoreNum_ = 1; }
```

## 2 Kernel 侧

```cpp
__aicore__ inline void Process() {
    if (blockIdx >= tiling_->realCoreNum) { return; }
    auto indiceUb = idxQueue.AllocTensor<TI>();
    Duplicate<TI>(indiceUb, 0, count);                 // indices 全 0
    idxQueue.EnQue(indiceUb);
    CopyOutIndices(count);
    if constexpr (withValue) {
        CopyIn(start, count);                          // values 透传（DataCopy 直通）
        CopyOut(start, count);
    }
}
```

## 3 要点

1. **无选择段**：不调用任何 Compare/Select；位置语义恒 0
2. indices 用 `Duplicate` 填充（无 GM 读）；values 走 CopyIn→CopyOut 直通（或 DataCopy GM→GM 等价路径，按框架约束选择）
3. **StoreUnAlign 单次 count ≤ VL**：位置/值写出的非对齐 Store 单次宽度受限，超限按 VL 分块循环写出（工业实现同款约束）
4. 无跨核同步、无 workspace
