# AR 场景模板 — Kernel 骨架（AscendC）

> **层级**：AscendC 适配层模板。模式 AR_CUT_A / AR_GATHER。真实完整实现见 [../example/](../../example/) 的 `arg_max_with_value_ar.h` / `arg_max_with_value_base.h`；本骨架给出结构与关键段落（骨架代码为范式命名，移植时对照 example 还原）。

## 1 类骨架

```cpp
template <typename TX, typename TIdxMid, typename TI, bool withValue, bool isMin>
class ArgReduceAr {
public:
    __aicore__ inline ArgReduceAr& Init(GM_ADDR x, GM_ADDR indices, GM_ADDR values,
                                        GM_ADDR workspace, const ArgReduceTilingData* tiling) {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer((__gm__ TX*)x);
        indicesGm_.SetGlobalBuffer((__gm__ TI*)indices);
        if constexpr (withValue) { valuesGm_.SetGlobalBuffer((__gm__ TX*)values); }
        // pipe buffer 初始化：值流 inQueue（DB 可选）+ 搬出队列
        return *this;
    }

    __aicore__ inline void Process() {
        if (tiling_->realCoreNum == 0) { return; }               // EMPTY 早退
        const uint64_t aPerCore = GetBlockIdx() < usedCoreNum - 1
                                ? tiling_->blkFactor : tiling_->blkTailFactor;
        for (uint64_t loop = 0; loop < aLoopCnt; loop++) {       // A1 外层（cutAPerLoop 批）
            CopyIn(aStart, tiling_->cutAPerLoop);                 // 整行载入（含尾核边界处理）
            Compute(tiling_->cutAPerLoop);                        // ArgMaxV1 逐行选择
            CopyOut(aStart, tiling_->cutAPerLoop);                // 值/位置双通道写出
        }
    }
};
```

## 2 选择段（手法 B：ArgMaxV1 调用）

```cpp
__aicore__ inline void Compute(uint16_t rows) {
    auto srcUb = inQueue.DeQue<TX>();
    auto valUb = outQueue.AllocTensor<TX>();       // 值通道
    auto idxUb = idxQueue.AllocTensor<TIDX>();     // 位置通道（等宽整数：fp16→uint16，输出前 UnPack+直写）
    // dtype 分派（编译期）：fp32 / fp16 / bf16(cast 计算) / int64 → 对应 ArgMaxV1 特化
    ArgMaxV1<TX>(valUb, idxUb, srcUb, rows, tiling_->cutRSize, startR);
    // cutRSize != rSize 时：外层 R 块循环，逐块调用并传块基址 startR，
    // 跨块结果在 UB 内二次 Compare+Select 归并（平局规则同 GE/LE 方向）
    inQueue.FreeTensor(srcUb);
}
```

**语义落地核对**（每算子逐项过）：GE（argmax first-occurrence）/ LE（argmin）；NaN 自比较 OR 掩码；位置流 = 段号×VL 等差生成（`startR` 基址并入）。

## 3 GATHER 特化路径（AR_GATHER）

```cpp
// R < VL 且非 32 倍数：先构造索引数列把 AR 行转成 RA 列，再一次向量比较
Duplicate(idxVec, 0, fullMask);  Add(idxVec, idxVec, one, addMask);   // 0..dimR-1
Gather(raBuf, srcRow, idxVec, ...);                                    // AR→RA 转置
ArgMaxRa(raBuf, ...);                                                  // 手法 F 向量比较
```

## 4 Buffer / DAG 分析

| Buffer | 通道 | 生命周期 | 说明 |
|--------|------|---------|------|
| inQueue（值流） | TX | CopyIn→Compute | DB 可选（行间无依赖） |
| valUb（胜者值） | TX | Compute→CopyOut | withValue=false 时仅寄存器，可不落 UB |
| idxUb（胜者位置） | 等宽整数（fp16→uint16） | Compute→UnPack/直写→CopyOut | 位置通道；uint16 经 UnPack 展开 + 块基址拼接写 int32，无需 Cast |
| tmp（可选） | — | ArgMaxV1 内部 | INDEX_TRACKING_REDUCE scratch（如走手法 A） |

**同步**：核内一条流水（值流 RAW/WAR 按持有法则 trace）；**无跨核同步**（A1 切核独立）。

## 5 移植对照表（example → 范式命名）

| example（工业实现） | 本范式 |
|---|---|
| `ArgMaxWithValueAr` | ArgReduceAr |
| `ArgMaxV1`（base.h 成员） | 手法 B 选择核心（照用） |
| `ProcessWithCutR / WithoutCutR` | R 块循环（§2 注释段） |
| `ArgMaxGatherRa`（base.h） | 手法 C/F Gather 路径（§3） |
| arch35 头文件 `arg_max_with_value_ar.h` | 本模板的完整实例 |
