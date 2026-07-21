---
id: H01
title: 多核 GM 写冲突（小输出 tensor < 32B）
symptom: zero_output
when: multicore_only
root_cause: cache_line_conflict
evidence: code
escalate_to: null
source: ascendc-debug.md#案例1
---

## triggers
- 单核运行结果完全正确，多核并行时部分 token 输出值被其他核的数据覆盖
- 输出 tensor 最内维 N 较小（如 N=4，每 token 仅 16 字节）
- 多核时输出全零或出现"串台"（某些 token 的值等于其他 token 的值）

## read_target
- `op_host/{op_name}_custom.cpp` → 查 `InferShapeAndType` 中输出 tensor 最后一维的值
  - grep: `SetDim\|out.*shape\|output.*dim`
- 计算：`最内维元素数 × sizeof(dtype)` 是否 < 32

## code_pattern
```cpp
// ❌ 输出 shape 按实际 N 设置，未 padding
// N=4, 每 token = 4 × 4B = 16B < 32B（一个 cache line）
out->SetDim(0, B);
out->SetDim(1, S);
out->SetDim(2, N);   // N=4 时每 token 仅占半个 cache line

// Kernel 端直接写出，多核同时写不同 token 时发生 cache line 竞争
DataCopyPad(outGm_[token * N], localBuf, ...);
```

## fix_template
```cpp
// ✅ Host 端：最内维 padding 到 ALIGN8（8 × float = 32B）
constexpr int64_t ALIGN8 = 8;
int64_t outCols = (N < ALIGN8) ? ALIGN8 : N;
out->SetDim(2, outCols);  // [B, S, 8] 而非 [B, S, 4]

// ✅ Kernel 端：填充 padding 区域为 0
for (uint32_t n = 0; n < N_; ++n)   sqBuf.SetValue(n, vals[n]);
for (uint32_t n = N_; n < ALIGN8; ++n) sqBuf.SetValue(n, 0.0f);
DataCopyPad(outGm_[token * ALIGN8], sqBuf, ...);

// ✅ PyBind / 调用端：去掉 padding 还原原始维度
if (outCols > N) {
    output = output.narrow(-1, 0, N).contiguous();
}
```

## verify_cmd
- 强制单核（`SetBlockDim(1)`）与多核结果做 diff，确认多核修复后一致
- 检查：Host / Kernel / PyBind 三处 ALIGN8 处理是否一致

## notes
- Ascend 910B GM 写粒度为 32 字节（一个 cache line）
- 每 token 写入数据 < 32B 时，相邻 token 落在同一 cache line，多核后写覆盖先写
- 单核测试永远发现不了此问题，必须多核测试才暴露
- 受影响场景：N 维度极小（N ≤ 7 for float32，N ≤ 15 for float16）的输出 tensor
