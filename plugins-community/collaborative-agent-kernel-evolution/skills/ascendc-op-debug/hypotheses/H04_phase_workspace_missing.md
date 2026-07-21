---
id: H04
title: 多 Phase 融合算子跨 Phase 中间 tensor 未写 workspace
symptom: precision_bias
when: large_D_only
root_cause: phase_ws_missing
evidence: code
escalate_to: null
source: ascendc-debug.md#案例3
---

## triggers
- 多 Phase 融合算子（如 RMSNorm→MHC_Post→Matmul→RMSNorm），后续 Phase 结果严重偏差
- D=2560（单 tile）正确，D=5120（多 tile）后面的 Phase 输出完全错误
- Phase B 计算结果（如 x2[N,D]）在 Phase C 中读取时得到错误值

## read_target
- `kernel/{op_name}.cpp` → 查多 Phase 结构，找跨 Phase 使用的中间 tensor
  - grep: `Phase\|phaseA\|phaseB\|workspace\|wsBase`
- 检查：需要跨 Phase 使用的大 tensor（如 x2[N×D]）在多 tile 时是否写入 workspace GM
- 检查：`wsBase_` 的 per-core 偏移计算是否正确：`blockIdx × coreWsFloats`

## code_pattern
```cpp
// ❌ 中间 tensor 只在 UB 中，多 tile 时 Phase B 下一个 tile 会覆盖 x2
// Phase B 中：
for (uint32_t t = 0; t < dimLoop_; ++t) {
    // 计算 x2Fp32，但没有写到 workspace GM
    // 下一个 tile 循环时 x2Fp32 被覆盖
    Mul(x2Fp32, hOutFp32, weightFp32, tD);
}
// Phase C 中：直接用 x2Fp32，但此时只有最后一个 tile 的数据
```

## fix_template
```cpp
// ✅ 多 tile 时将中间 tensor 写入 workspace GM，Phase C 再读回
// Phase B 中：
for (uint32_t t = 0; t < dimLoop_; ++t) {
    uint64_t wsOff = wsBase_ + (uint64_t)n * D_ + tOff;
    Mul(x2Fp32, hOutFp32, weightFp32, tD);
    if (dimLoop_ > 1) {   // 单 tile 时跳过，避免 ~2% 性能损失
        DataCopyPad(workspaceGm_[wsOff], x2Fp32, ...);
        PipeBarrier<PIPE_ALL>();
    }
}

// Phase C 中：
for (uint32_t t = 0; t < dimLoop_; ++t) {
    uint64_t wsOff = wsBase_ + (uint64_t)n * D_ + tOff;
    if (dimLoop_ > 1) {
        DataCopyPad(x2Fp32, workspaceGm_[wsOff], ...);
        PipeBarrier<PIPE_ALL>();
    }
    // 继续使用 x2Fp32...
}

// workspace per-core 偏移：
// wsBase_ = (uint64_t)blockIdx * coreWsFloats_
```

## verify_cmd
- 对每个 Phase 的输出单独验证（借用 piggybacking 技巧，见 H_PIGGYBACKING）
- 顺序：确认 Phase A 正确 → Phase B 正确 → Phase C 正确 → Phase D 正确
- 强制 D=5120 单核测试，排除多核干扰

## notes
- 仅在 `dimLoop_ > 1` 时才需要写 workspace，单 tile 直接在 UB 保留即可
- workspace 地址：`wsBase_ = blockIdx × coreWsFloats`，必须按物理核号索引
- workspace 大小必须按 maxN × maxD 分配（见 H02），避免跨 shape 越界
- piggybacking 调试技巧：借用一个输出 tensor 临时输出中间调试值，无需改接口
