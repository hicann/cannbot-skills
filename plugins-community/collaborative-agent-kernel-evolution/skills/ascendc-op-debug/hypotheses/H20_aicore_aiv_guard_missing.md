---
id: H20
title: ASCEND_IS_AIV 守卫缺失 → AIC/AIV 数据竞争
symptom: multicore_mismatch
when: always
root_cause: aicore_aiv_guard_missing
evidence: code
escalate_to: null
source: FA_V2_report.md / CROSS_OP_SUMMARY2.md
---

## triggers
- 输出中大量行完全相同（AIC 子核与 AIV 子核均执行了写操作，产生竞争）
- 精度偶发性错误，单核正常多核失败
- 使用 `SetBlockDim(aivCoreNum)` 调度但未加 AIV 守卫

## read_target
- `op_kernel/{op_name}.cpp` → 搜索 kernel 入口函数
  - grep: `__global__.*__aicore__\|ASCEND_IS_AIV\|ASCEND_IS_AIC`
- 检查 kernel 入口是否存在 `if ASCEND_IS_AIV { ... }` 守卫

## code_pattern
```cpp
// ❌ 危险模式：无守卫，AIC 子核也会执行，产生数据竞争
extern "C" __global__ __aicore__
void my_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    MyOp op;
    op.Init(x, y, t);
    op.Process();  // AIC 和 AIV 都执行 → 输出中大量相同行
}
```

## fix_template
```cpp
// ✅ 正确：用 ASCEND_IS_AIV 守卫，只让 AIV 子核执行
extern "C" __global__ __aicore__
void my_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if ASCEND_IS_AIV {   // ← 注意：无括号，是宏语法
        MyOp op;
        op.Init(x, y, t);
        op.Process();
    }
}
```

## verify_cmd
```bash
# 检查 kernel 入口是否含有守卫
grep -n "ASCEND_IS_AIV\|ASCEND_IS_AIC" op_kernel/*.cpp
# 若输出为空 → 确认缺失守卫
# 期望看到：if ASCEND_IS_AIV { ... }
```

## notes
- 即使 `SetBlockDim(aivCoreNum)` 调度，每个 block 仍包含 AIC+AIV 子核对
- AIC 子核没有守卫时也会执行相同代码 → 双重写入 → 大量相同行输出
- 宏语法：`if ASCEND_IS_AIV { }` 无括号，加括号 `if (ASCEND_IS_AIV)` 会编译报错
