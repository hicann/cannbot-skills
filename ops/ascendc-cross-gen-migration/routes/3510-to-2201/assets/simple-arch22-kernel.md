# 简单搬运类算子的 arch22 专用 kernel 模板

> 适用：copy / scatter / gather / cache 更新这类“只搬数据、不算或只做轻量计算”的算子，且 A5 侧实现是通用模板/大 kernel。

## 1. 先判断要不要专用 kernel

满足以下任一条，优先写专用 arch22 kernel，而不是复用 A5 通用类：

- A5 实现是带 pipe/模板分支的大类（如 `XxxNormalSiso<...>`），参数化程度高；
- 目标算子只做连续块搬运，没有复杂计算；
- 复用版本能跑通，但 `msprof` 的 Task Duration 明显高于朴素实现（常见 20%–30%）。

## 2. 最小结构

```cpp
// op_kernel/arch22/<op>_arch22.cpp
// 每个 block 处理若干 tile；tile = 连续 [rows, H] 行块
extern "C" __global__ __aicore__ void <op>_arch22(GM_ADDR key, GM_ADDR cache, GM_ADDR slot,
                                                  GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    // 1) 解析本 block 负责的 tile 范围
    // 2) 对每个 tile：DataCopyPad 到 UB，再按 slot 偏移 DataCopyPad 到 cache
    // 3) 不做逐元素标量循环
}
```

要点：

- `DataCopyPad` 处理非 512B 对齐；对齐时用 `DataCopy`；
- block 数按 `min(coreNum, totalTiles)` 设置，tile 太小会增加调度开销；
- 输出地址 = `slot * H * sizeof(T)` 直接算，不要额外查表/中转 buffer；
- 不要引入与功能无关的 workspace 或同步。

## 3. 验收

1. 正确性：与 CPU 参考一致（copy/scatter 应精确相等）；
2. 会话内 `msprof` 采一次 Task Duration，确认与朴素实现同一量级；
3. 若明显偏慢，先怀疑“是否又走了通用模板”，再查对齐与 block 数。

## 4. 小拷贝（≤4KB/token）别上重装备

实测（见 [experiments/spc.md](../experiments/spc.md)）：单 `TBuf` + 两次 `DataCopyPad` + `PipeBarrier` 明显快于 TQue 双缓冲 + 每 token `AllocTensor/EnQue/DeQue` + 批量预取（约 9.9 vs 13.9 µs，同一 case、同机、msprof 中位数）。

- token 很小、token 数不多时，**不要**用 TQue 双缓冲、批量 slot 预取、复杂事件链；
- 单个 `TBuf`（16 KiB 足够）+ `DataCopyPad` 入/出 + `PipeBarrier<PIPE_ALL>` 更贴近硬件；
- 只有确认带宽受限（大 token、长拷贝）时再考虑双缓冲重叠。

## 5. 开工顺序（减少返工）

1. 先定小 POD tiling 结构（只放 usedCoreNum/blockFactor/tailBlockFactor/tokenSize 这类必需字段）；
2. 再写 A2 host tiling，只填这些字段；
3. 注册 tiling class；
4. 写独立 kernel 入口，只 include 本算子 arch22 头；
5. 最后写拷贝循环，一次构建 + 一次直调 + 一次 profile。

不要先复用 A5 大 tiling 结构“跑通再重构”，返工成本通常是首次做对的 2–3 倍。
