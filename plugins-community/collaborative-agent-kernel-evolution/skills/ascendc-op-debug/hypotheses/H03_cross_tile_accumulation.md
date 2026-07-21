---
id: H03
title: D-axis 跨 Tile 累加未用 scalar 变量
symptom: precision_bias
when: large_D_only
root_cause: cross_tile_accum
evidence: code
escalate_to: null
source: ascendc-debug.md#案例3,案例6
---

## triggers
- D=2560（单 tile）精度完全正确，D=5120（多 tile）出现系统性精度偏差
- RMSNorm / ReduceSum 类操作的归约结果偏差随 tile 数增多而累积
- matmul 点积（跨 tile 累加）结果与 golden 有持续偏差

## read_target
- `kernel/{op_name}.cpp` → 查 D-tile 循环中的累加逻辑
  - grep: `ReduceSum\|sumSq\|accum\|GetValue(0)`
- 检查：ReduceSum 结果是赋给 buffer 还是 scalar 变量
- 检查：scalar 累加变量是否在 for-tile 循环**外部**初始化为 0

## code_pattern
```cpp
// ❌ 错误1：在 buffer 中原地累加（对齐/初始化问题导致精度错）
for (uint32_t t = 0; t < dimLoop_; ++t) {
    ReduceSum(accumBuf, sqBuf, reduceWork, tD);
    Add(totalBuf, totalBuf, accumBuf, 1);  // 向量 Add 不适合 scalar 累加
}

// ❌ 错误2：scalar 变量在循环内初始化（每 tile 清零，无法跨 tile 累加）
for (uint32_t t = 0; t < dimLoop_; ++t) {
    float sumSq = 0.0f;  // 每次循环都清零！
    ReduceSum(scBuf, sqBuf, reduceWork, tD);
    sumSq += scBuf.GetValue(0);
}
```

## fix_template
```cpp
// ✅ scalar 变量在循环外初始化，循环内累加
float sumSq = 0.0f;   // ← 循环外
for (uint32_t t = 0; t < dimLoop_; ++t) {
    // ... 加载并计算当前 tile 的 sqBuf ...
    ReduceSum(scBuf, sqBuf, reduceWork, tD);
    PipeBarrier<PIPE_V>();
    sumSq += scBuf.GetValue(0);   // ← scalar 累加，天然精确
}
float invRms = rsqrtf(sumSq / D_ + eps_);

// ✅ matmul 点积跨 tile/head 同样用 scalar 累加
float dotProd = 0.0f;
for (uint32_t n = 0; n < N_; ++n) {
    for (uint32_t t = 0; t < dimLoop_; ++t) {
        Mul(sqBuf, xBuf[n * dimTile_], phiChunk, tD);
        ReduceSum(scBuf, sqBuf, reduceWork, tD);
        PipeBarrier<PIPE_V>();
        dotProd += scBuf.GetValue(0);  // ← scalar 跨 tile/head 累加
    }
}
```

## verify_cmd
- 固定单核，分别测 D=2560 和 D=5120，确认大 D 修复后精度达标
- 用 PyTorch `mean(dim=-1)` 对比归约结果，定位第一个偏差出现的 tile

## notes
- ReduceSum 输出是 tile 内的部分和，必须用 C++ scalar 在外层累加
- `scBuf` 的工作区大小至少需要 `tileLength × sizeof(float)` 字节
- 单 tile 优化：`if (dimLoop_ == 1)` 可跳过 workspace 读写，中间结果留在 UB
- 来自 MhcPostFusion 实战：RMSNorm sum-of-squares 和 matmul 点积都有此问题
