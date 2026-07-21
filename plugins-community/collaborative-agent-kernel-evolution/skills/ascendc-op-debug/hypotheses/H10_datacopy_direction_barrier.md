---
id: H10
title: DataCopy 方向写反 / PipeBarrier 缺失
symptom: zero_output
when: always
root_cause: datacopy_direction
evidence: code
escalate_to: null
source: ascendc-debug.md#调试检查清单
---

## triggers
- 输出全零（DataCopy 方向写反，输出从未写入）
- 计算结果正确但输出不更新（EnQue/DeQue 不配对）
- 向量计算结果与前序数据加载结果相同（计算前未等待 DataCopy 完成）

## read_target
- `kernel/{op_name}.cpp` → 逐行检查 DataCopy 和 pipe barrier
  - grep: `DataCopy\|EnQue\|DeQue\|PipeBarrier\|SetFlag\|WaitFlag`
- 检查 DataCopy：第 1 个参数是 dst，第 2 个是 src（常见写反）
- 检查 pipe 流水：DataCopy GM→Local 后是否有对应的 EnQue/DeQue 等待

## code_pattern
```cpp
// ❌ DataCopy 方向写反：把输出 GM 的数据复制到 local，而非反向
DataCopy(inputGm[offset], inputLocal, tileLength);   // 应为 (inputLocal, inputGm[offset], ...)

// ❌ 缺少 pipe barrier，计算在数据加载完成前就开始
DataCopy(inputLocal, inputGm[offset], tileLength);
// 缺少：inQue_.EnQue(inputLocal) / outQue_.DeQue()
Mul(outputLocal, inputLocal, weightLocal, tileLength);  // 可能读到脏数据
```

## fix_template
```cpp
// ✅ DataCopy 正确方向：dst 在前，src 在后
DataCopy(inputLocal, inputGm[offset], tileLength);   // GM → Local（加载）
DataCopy(outputGm[offset], outputLocal, tileLength); // Local → GM（写出）

// ✅ 完整 pipe 流水（双 buffer 模式）
// 加载阶段
inQue_.EnQue(inputLocal);
// 计算阶段
auto x = inQue_.DeQue<half>();
Mul(tmpLocal, x, weightLocal, tileLength);
inQue_.FreeTensor(x);
// 写出阶段
outQue_.EnQue(outputLocal);
DataCopy(outputGm[offset], outQue_.DeQue<half>(), tileLength);
```

## verify_cmd
- 使用 DumpTensor 在 DataCopy 后立即打印 inputLocal，确认值非零
- 使用 DumpTensor 在 Mul 后打印 outputLocal，确认计算结果正确
- 检查 EnQue/DeQue 调用次数是否配对

## notes
- `DataCopy(dst, src, len)`：记忆法 = 赋值方向，左边是目标
- 输出全零的最常见原因之一：DataCopy(outputLocal, outputGm) 把 GM 写到 Local 而非反向
- PipeBarrier 缺失通常导致偶发性错误，不是稳定全零
