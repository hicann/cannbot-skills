---
id: H07
title: Tiling 参数硬编码导致大 shape 错误
symptom: precision_bias
when: large_shape_only
root_cause: tiling_hardcode
evidence: code
escalate_to: null
source: ascendc-debug.md#原则1
---

## triggers
- 固定 shape 测试通过，变化 shape 时结果错误
- loopCount=1 但实际需要多次循环
- lastTile 大小等于 tileLength（边界 tile 未特殊处理）

## read_target
- `op_host/{op_name}_custom.cpp` → 查 tiling 参数计算
  - grep: `tileLength\|loopCount\|dimTile\|lastTile\|blockDim`
- 检查：是否有硬编码的数值（如 `uint32_t dimTile = 2560`）
- 检查：lastTile 是否等于 `D % dimTile`（0 时等于 dimTile）

## code_pattern
```cpp
// ❌ 硬编码 tiling 参数
tilingData->dimTile   = 2560;     // D=5120 时需要 2 个 tile，但 loopCount=1
tilingData->dimLoop   = 1;        // 错误：D=5120 / 2560 = 2
tilingData->lastDimTile = 2560;   // 错误：最后一个 tile 可能更小

// ❌ 未处理 lastTile
tilingData->tileLength = TILE_SIZE;
tilingData->loopCount  = totalLen / TILE_SIZE;  // 丢弃余数
// 没有 lastTileLength，尾部数据丢失
```

## fix_template
```cpp
// ✅ 完全动态计算 tiling 参数
uint32_t dimTile = D;
if (dimTile > MAX_DIM_TILE) dimTile = MAX_DIM_TILE;  // UB 容量上限
dimTile = (dimTile / ALIGN_ELEM) * ALIGN_ELEM;        // 向下对齐
uint32_t dimLoop     = (D + dimTile - 1) / dimTile;
uint32_t lastDimTile = D % dimTile;
if (lastDimTile == 0) lastDimTile = dimTile;          // 整除时 last = full tile

tilingData->dimTile     = dimTile;
tilingData->dimLoop     = dimLoop;
tilingData->lastDimTile = lastDimTile;

// Kernel 端：区分普通 tile 和 last tile
uint32_t tD = (t == dimLoop_ - 1) ? lastDimTile_ : dimTile_;
```

## verify_cmd
- 分别测试 D 整除 dimTile（如 D=5120, dimTile=2560）和不整除（如 D=3000, dimTile=2560）
- 打印 tiling 参数确认：`dimLoop, lastDimTile` 在各 shape 下的值

## notes
- 所有 tiling 参数必须从输入 shape 动态推导，禁止任何硬编码数值
- `lastDimTile = D % dimTile`，当整除时等于 0，需特判为 `dimTile`
- 核间划分的余数处理见 H08
