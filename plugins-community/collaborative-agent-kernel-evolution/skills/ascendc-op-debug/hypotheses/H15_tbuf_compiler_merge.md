---
id: H15
title: TBuf 多写被 CANN 编译器合并（只保留最后一次写入）
symptom: compiler_merged_output
when: always
root_cause: tbuf_compiler_merge
evidence: code
escalate_to: null
source: FlashAttentionV2 FA_V2_REPORT2.md
---

## triggers
- 输出内容全部等于"最后一次赋值"的值（如循环中多次 Duplicate+DataCopy，只看到最后一次的常数）
- 所有输出行/列值完全相同（最后一次循环迭代的值覆盖所有位置）
- 用探针法写入 3 个不同常数（11, 22, 33），输出全部等于 33.0

## read_target
- `op_kernel/{op_name}.cpp` → 搜索 TBuf 的反复 Duplicate+DataCopy 模式
  - grep: `TBuf\|Duplicate\|DataCopy`
- 检查是否存在：同一个 TBuf 变量被 Duplicate 多次后 DataCopy 到不同目标

## code_pattern
```cpp
// ❌ 危险模式：CANN 编译器会合并所有写入，只保留最后一次
TBuf<TPosition::VECCALC> tbuf;
pipe.InitBuffer(tbuf, 256, 128 * sizeof(half));

// 循环内：
Duplicate(tbuf.Get<half>(), val1, 128);
DataCopy(dst[offset1], tbuf.Get<half>(), 128);   // ← 被合并，实际不执行
Duplicate(tbuf.Get<half>(), val2, 128);
DataCopy(dst[offset2], tbuf.Get<half>(), 128);   // ← 只有这次生效

// 结果：dst[offset1] 也被写成 val2
```

## fix_template
```cpp
// ✅ 使用 TQue 替代 TBuf，EnQue/DeQue 建立数据依赖，防止编译器合并
TQue<QuePosition::VECIN, 1> que;
pipe.InitBuffer(que, 1, 128 * sizeof(half));

// 循环内：
{
    auto t = que.AllocTensor<half>();
    Duplicate(t, val1, 128);
    que.EnQue(t);
    auto t2 = que.DeQue<half>();
    DataCopy(dst[offset1], t2, 128);
    que.FreeTensor(t2);
}
{
    auto t = que.AllocTensor<half>();
    Duplicate(t, val2, 128);
    que.EnQue(t);
    auto t2 = que.DeQue<half>();
    DataCopy(dst[offset2], t2, 128);
    que.FreeTensor(t2);
}
```

## verify_cmd
```bash
# 探针验证：写 3 个不同已知值，检查输出
# 在 kernel 里写：
#   DataCopy(y[0],   tbuf_with_11, 128)
#   DataCopy(y[128], tbuf_with_22, 128)
#   DataCopy(y[16],  tbuf_with_33, 128)
# Python 侧：
import torch; out = ...npu...
print(out.flatten()[:5])  # 若全为 33.0 → TBuf 合并确认
```

## notes
- TBuf 适用场景：单次初始化、固定常量、不在循环内重复写的 buffer
- TQue 强制场景：循环内需要多次向不同目标写入的任何 buffer
- 此 bug 无任何编译报错或运行时报错，纯静默错误，极难定位
