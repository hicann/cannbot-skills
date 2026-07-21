---
id: H06
title: UB buffer 按 D 而非 dimTile 分配导致溢出
symptom: crash
when: large_D_only
root_cause: ub_overalloc
evidence: code
escalate_to: null
source: ascendc-debug.md#原则4
---

## triggers
- D=2560 正常，D=5120 运行时崩溃（InitBuffer 相关）
- 错误发生在算子初始化阶段（Init 函数）
- 编译通过但运行时崩溃，无明显精度问题（直接崩）

## read_target
- `kernel/{op_name}.cpp` → 查 `Init` 函数中的 `InitBuffer` 调用
  - grep: `InitBuffer\|pipe_\.Init\|BUFFER_NUM`
- 检查：InitBuffer 的大小参数是否使用了完整 D，而非分 tile 后的 dimTile

## code_pattern
```cpp
// ❌ 按完整 D 分配，D=5120 时 UB 容量不足（910B UB 约 256KB）
__aicore__ inline void Init(..., uint32_t D, ...) {
    pipe_.InitBuffer(bufHOutFp32_, D * sizeof(float));        // D=5120 → 20KB per buf
    pipe_.InitBuffer(bufX2Fp32_,   N * D * sizeof(float));    // N*D=20480 → 80KB
    pipe_.InitBuffer(bufWeightFp32_, D * sizeof(float));      // 再 20KB
    // 三个 buffer 合计 120KB，若有更多 buffer 则 UB 溢出
}
```

## fix_template
```cpp
// ✅ 按 dimTile 分配，保证单 tile 数据 fit UB
// dimTile 在 Host 端计算：dimTile = min(D, UB_CAPACITY / (buffer_count * sizeof(float)))
__aicore__ inline void Init(..., uint32_t dimTile, uint32_t N, ...) {
    // dimTile ≤ 2560 保证所有 buffer 合计 fit UB
    pipe_.InitBuffer(bufHOutFp32_,  dimTile * sizeof(float));
    pipe_.InitBuffer(bufX2Fp32_,    N * dimTile * sizeof(float));
    pipe_.InitBuffer(bufWeightFp32_, dimTile * sizeof(float));
    // D 轴通过 for-tile 循环处理（见 H03）
}

// Host 端动态计算 dimTile：
uint32_t dimTile = D;
if (dimTile > 2560) dimTile = 2560;           // 硬件 UB 上限
dimTile = (dimTile / ALIGN_ELEM) * ALIGN_ELEM; // 向下对齐
uint32_t dimLoop = (D + dimTile - 1) / dimTile;
```

## verify_cmd
- 在 `Init` 函数末尾打印各 buffer 实际分配大小，与 UB 总容量对比
- 分别测 D=2560（单 tile）和 D=5120（多 tile），确认后者不崩溃

## notes
- Ascend 910B 单核 UB 容量约 256KB，多个 buffer 累加不能超限
- dimTile 必须在 Host 端 tiling 时计算好，传入 kernel 的 tiling 参数
- 多 tile 时 D 轴循环处理，配合 H03（scalar 累加）和 H04（workspace 暂存）
