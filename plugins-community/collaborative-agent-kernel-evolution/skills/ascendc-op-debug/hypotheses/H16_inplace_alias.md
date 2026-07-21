---
id: H16
title: CANN 框架 Inplace 别名（input0 与 output0 共享物理内存）
symptom: precision_bias
when: always
root_cause: inplace_alias
evidence: code
escalate_to: null
source: FlashAttentionV2 FA_V2_REPORT2.md
---

## triggers
- kernel 读 input0 时，读到的是已经写入 output0 的值（读/写顺序污染）
- 在 pybind 侧 `torch.equal(input_tensor, output_tensor)` 为 True（指向同一 storage）
- `Q.clone()` 无效：clone 后仍然出现 input==output 现象

## read_target
- `op_host/{op_name}.cpp` → 检查 InferShape 函数
  - grep: `InferShape\|SetShapeAndDataType\|OUTPUT`
- 检查是否存在：将 input[0] 的 shape 直接赋给 output[0]

## code_pattern
```cpp
// ❌ 触发条件：InferShape 中将 input0 shape 复制给 output0
// CANN 框架推断：input0 和 output0 shape 相同 → 可以 inplace → 映射同一内存

IMPLEMT_COMMON_INFERFUNC(FlashAttentionV2CustomInferShape) {
    // input[0] = query [B, S, H*d]
    // output[0] = y    [B, S, H*d]  ← 与 query 完全相同 → INPLACE 推断触发！
    auto query_shape = op.GetInputDescByName("query").GetShape();
    op.GetOutputDescByName("y").SetShape(query_shape);  // ← 触发点
    return GRAPH_SUCCESS;
}
```

## fix_template
```cpp
// ✅ 方案：新增 out_buf 作为第 4 个"输入"，kernel 写 out_buf 而非 y

// op_host：注册 out_buf 为输入
this->Input("out_buf")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});

// kernel：写 out_buf (param index 3)，不写 y (param index 4)
__gm__ half* out_ptr = (__gm__ half*)out_buf;   // 实际输出写这里

// pybind 侧：
at::Tensor out_buf = at::zeros_like(query);        // 专用输出
at::Tensor result  = at::empty_like(query);         // 占位（被 aliased 无所谓）
EXEC_NPU_CMD(aclnnMyOp, query, key, value, out_buf, head_num, scale, result);
return out_buf;   // 返回 out_buf，不返回 result
```

## verify_cmd
```python
# 验证是否存在 inplace alias
import torch_npu
q = torch.randn(1, 4, 64, dtype=torch.float16).npu()
result = torch.empty_like(q)
# 调用 op 后检查：
print(q.data_ptr() == result.data_ptr())  # True → 存在 inplace alias
```

## notes
- 触发条件：InferShape 中 output[i].shape == input[j].shape，CANN 自动推断可 inplace
- Q.clone() 无效原因：clone 发生在 Python 侧，CANN framework 在 C++ 侧 re-alias
- 此模式适用于所有"输出 shape 与某输入相同"的算子（attention、residual add 等）
- CANN 官方文档称此为"就地操作优化"，属于性能特性而非 bug
