---
id: H17
title: TQue DEPTH=1 多 lifetime 死锁（内外循环共用同一 queue）
symptom: hang
when: loop_body_only
root_cause: tque_depth1_deadlock
evidence: code
escalate_to: null
source: FlashAttentionV2 FA_V2_REPORT2.md
---

## triggers
- kernel 在特定循环迭代处永久挂起（hardware timeout，无报错日志）
- 单次调用正常，循环体内第 2 次以上的调用挂死
- 症状：进程 hang，需要 kill；NPU 报 507034 超时错误

## read_target
- `op_kernel/{op_name}.cpp` → 检查 TQue 的 AllocTensor/FreeTensor 配对
  - grep: `AllocTensor\|FreeTensor\|EnQue\|DeQue`
- 检查是否存在：外层作用域持有 tensor（未 FreeTensor），内层循环对同一 que 调用 AllocTensor

## code_pattern
```cpp
// ❌ 危险模式：queCalcF2 被外层（qF 存活整行）和内层（ktFloat 每次迭代）共用
// DEPTH=1 队列：当 qF 活跃时，内层 AllocTensor 永久等待（死锁！）

TQue<QuePosition::VECIN, 1> queCalcF2;  // DEPTH=1 ← 关键

// 外层：分配 qF，整行计算期间保持活跃
auto qF = queCalcF2.AllocTensor<float>();   // 分配，占用 queCalcF2
Cast(qF, qHalf, RoundMode::CAST_NONE, d);
queCalcF2.EnQue(qF);
auto qFLive = queCalcF2.DeQue<float>();    // qFLive 存活

// 内层循环：
for (int k = 0; k < Skv; k++) {
    auto ktFloat = queCalcF2.AllocTensor<float>();  // ← 永久阻塞！qFLive 未 Free
    // ...
}
queCalcF2.FreeTensor(qFLive);  // 太晚了，已经死锁
```

## fix_template
```cpp
// ✅ 外层 tensor 和内层 tensor 使用不同的 TQue
TQue<QuePosition::VECIN, 1> queCalcF2;  // 专用于外层长生命周期 tensor（qF, pF）
TQue<QuePosition::VECIN, 1> queCalcF3;  // 专用于内层短生命周期 tensor（ktF, vF）

// 外层：
auto qFLive = queCalcF2.AllocTensor<float>();  // 外层 queue
Cast(qFLive, qHalf, RoundMode::CAST_NONE, d);
// qFLive 存活整行计算期间

// 内层循环：
for (int k = 0; k < Skv; k++) {
    auto ktFloat = queCalcF3.AllocTensor<float>();  // 不同 queue，不会死锁
    // ... 使用 ktFloat ...
    queCalcF3.FreeTensor(ktFloat);  // 内层结束立即释放
}

queCalcF2.FreeTensor(qFLive);  // 外层行结束释放
```

## verify_cmd
```bash
# 静态代码审查：检查内外循环是否共用同一 TQue
grep -n "AllocTensor\|FreeTensor" op_kernel/*.cpp | \
    awk -F: '{print $2, $3}' | sort
# 检查：每个 AllocTensor 前是否所在 queue 的所有之前的 tensor 都已 FreeTensor
```

## notes
- DEPTH=1 的 TQue 在任意时刻只能有 1 个活跃 tensor（从 AllocTensor 到 FreeTensor）
- 规则：外层变量和内层变量的 lifetime 有交叉时，必须使用不同 TQue
- 判断 lifetime 交叉：外层 AllocTensor 的 tensor 在内层 loop 结束之后才 FreeTensor → 交叉
- 增大 DEPTH 可以缓解（DEPTH=2 允许 2 个活跃），但会增加 UB 内存用量
