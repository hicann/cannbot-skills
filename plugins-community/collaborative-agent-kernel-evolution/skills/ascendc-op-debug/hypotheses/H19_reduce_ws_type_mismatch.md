---
id: H19
title: ReduceMax/ReduceSum workspace buffer 类型与 src 不匹配
symptom: precision_bias
when: always
root_cause: reduce_ws_type_mismatch
evidence: code
escalate_to: null
source: FlashAttentionV2 FA_V2_REPORT2.md
---

## triggers
- ReduceMax 或 ReduceSum 结果偶发 NaN 或完全错误
- softmax 归一化结果异常（maxVal/sumVal 不正确）
- 仅在 float 计算路径中出现，half 路径正常（因为 workspace 用了 half buffer）

## read_target
- `op_kernel/{op_name}.cpp` → 检查 ReduceMax/ReduceSum 的 workspace 参数类型
  - grep: `ReduceMax\|ReduceSum`
- 检查 workspace tensor 的分配类型是否与 src tensor 相同

## code_pattern
```cpp
// ❌ 危险：src 是 float，但 workspace 用了 half buffer
TQue<QuePosition::VECIN, 1> queCalcH1;  // ← half 类型 queue
pipe.InitBuffer(queCalcH1, 1, 128 * sizeof(half));

auto accF = queCalcF1.DeQue<float>();   // src: float tensor
auto wsH  = queCalcH1.AllocTensor<half>();  // workspace: half ← 类型不匹配！

ReduceMax(maxVal, accF, wsH, Skv);     // wsH 类型错误 → 结果 NaN 或错误
```

## fix_template
```cpp
// ✅ workspace 必须与 src 相同类型（float src → float workspace）

TQue<QuePosition::VECIN, 1> queCalcF3;  // ← float 类型 queue
pipe.InitBuffer(queCalcF3, 1, 128 * sizeof(float));  // float workspace

auto accF  = queCalcF1.DeQue<float>();    // src: float
auto wsF   = queCalcF3.AllocTensor<float>();  // workspace: float ← 匹配！

// ReduceMax(dst, src, workspace, count)
// 注意：src 可能被 workspace 覆盖，需要提前备份 src
ReduceMax(maxVal, accF, wsF, Skv);
queCalcF3.FreeTensor(wsF);
```

## ReduceMax/ReduceSum 完整约束

```cpp
// 签名：
// ReduceMax(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> workLocal, uint32_t count)
// ReduceSum(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> workLocal, uint32_t count)
//
// 约束：
// 1. dst、src、workLocal 三者类型必须相同（T 一致）
// 2. workLocal 大小 >= src 大小（建议 == src 大小）
// 3. src 在 ReduceMax/ReduceSum 后可能被修改（workLocal 可能写入 src 区域）
//    → 需要在调用前备份 src，或之后不再使用 src 的原始值
// 4. dst 是标量结果（1 个元素）
```

## verify_cmd
```bash
# 检查 ReduceMax/ReduceSum 的参数类型是否一致
grep -n -A3 "ReduceMax\|ReduceSum" op_kernel/*.cpp
# 检查：第 3 个参数（workLocal）的类型与第 2 个参数（src）是否相同
```

## notes
- float 路径中最常见：计算用 float 提高精度，但 workspace 忘记换成 float
- 此 bug 无编译报错，运行时只有计算结果错误（softmax 爆炸）
- 修复原则：workspace 类型跟着 src 走，不是跟着"我用什么 buffer"走
