# GROUP_REDUCE 模板（AscendC）

> **层级**：AscendC 适配层模板。模式 GROUP_REDUCE（30001）。两阶段跨核结构；完整实现见 [../example/](../../example/) 的 `arg_max_with_value_group_reduce.h`。

## 1 Tiling 侧

```cpp
// 触发与回退见 mode-mapping §3；命中后：
SetScheduleMode(1);                                  // 全核到齐调度（host 侧必设）
realCoreNum_ = originCoreNum_;                       // 栅栏口径：物理核
blkFactor_   = rSize / realCoreNum_;                 // R 切核（每核 R 段）
isRaSplit_   = (A0 * eb >= 128 || blkFactor_ * eb < VL);   // RA 分支判定（AR 分支 UB 不足 fallback）

// workspace：
if (!isRaSplit_)   // AR 分支（core0 聚合）：[value: core×outAAlign][indice: core×outAAlign]
    workSpaceSize_ = outAAlign * 2 * sizeof(int32_t) * originCoreNum_;   // int64 ×2；outAAlign 按 32B 对齐（工业口径，256B 为可选超集）
else               // RA 分支（全核聚合）：[value: A1×core×nextAAlign128][indice: A1×core×indiceAlign128]
    workSpaceSize_ = A1 * originCoreNum_ * (nextAAlign128 * eb + indiceAlign128 * 4);
```

## 2 Kernel 侧：Phase 1（局部胜者 + workspace）

```cpp
__aicore__ inline void Process() {
    if (blockIdx >= tiling_->realCoreNum) { return; }
    ComputeLocalWinners();                    // 本核 R 段内：复用 AR/ARA 手法（按场景）
    CopyOutWorkSpace();                       // 局部 (winValue, winPosition) 写 workspace
    SyncAll();                                // 跨核栅栏（全部物理核到齐）
    if (!tiling_->isRaSplit) ProcessMultiCoreResult();        // AR 分支：core0 聚合
    else                    ProcessMultiCoreResultAllCore();  // RA 分支：全核并行聚合
}
```

**核基址偏移**：局部位置在写入前加 `blockIdx × blkFactor`（R 段基址），一次施加。

## 3 Phase 2：聚合

```cpp
// AR 分支（core0 聚合）：逐核两两归并，复用手法 B 的 Compare+Select 结构
if (blockIdx == 0) {
    LoadWsBest();                                          // core0 自己的结果
    for (core = 1; core < originCoreNum; core++) {
        Compare<GE>(cmpMask, bestVal, wsVal[core], mask);  // + NaN OR + 平局方向核对
        Select(bestVal, bestVal, wsVal[core], cmpMask);
        Select(bestPos, bestPos, wsPos[core], cmpMask);
    }
    CopyOut();
}

// RA 分支（全核聚合）：每核负责一批输出（gatherBlockSize 分块），读该批全部核的局部结果归并
```

**聚合三要素核对**：基址偏移已施加 / 平局规则参与比较 / 均衡分配无空块核。

## 4 Buffer / 同步

| Buffer | 说明 |
|--------|------|
| 局部值/位置通道 | Phase 1 常驻 |
| workspace（GM） | 双段布局 [value | indice]；Phase 1 写、Phase 2 读 |
| Phase 2 胜者通道 | 聚合段常驻 |

**同步清单**：核内流水（Phase 1 / Phase 2 各自 trace）+ 跨核 `SyncAll`（唯一栅栏）+ Phase 2 读 workspace 前的 WAR 依赖（栅栏已覆盖）。

## 5 移植对照表

| example | 本范式 |
|---|---|
| `ArgMaxWithValueGroupReduce` | ArgReduceGroupReduce |
| `CopyOutWorkSpace / CopyOutWorkSpaceRow` | Phase 1 写（AR/RA 分支布局） |
| `ProcessMultiCoreResult / AllCore` | Phase 2 聚合两形态 |
| `SetScheduleMode(1)`（tiling cpp） | §1 host 侧配套 |
