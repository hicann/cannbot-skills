# A4: SetFlag/WaitFlag Event Sync (事件同步保证精度)
## Overview
专家实现在关键计算节点插入了PipeBarrier<PIPE_V>()进行向量管道同步，确保数据转换和计算操作的正确顺序。这种细粒度的同步控制在BF16的Cast-Compute-Cast流程中尤为重要，防止了数据竞争和乱序执行导致的精度问题。PipeBarrier确保前一步操作完全完成后才执行下一步，虽然会引入一定的性能开销，但对于精度敏感的场景是必要的。baseline实现完全没有使用pipeline barrier，在复杂场景下可能出现同步问题。

## When to Use
- Data dependencies across pipes
- 确保计算顺序正确性，防止数据竞争，保证数值精度
- 每步显式同步，确保计算顺序，避免指令重排序导致的精度问题
- 确保数值计算正确性，避免异步执行导致的数据竞争

## Trade-off
- 引入同步开销，可能降低流水线效率
- 增加了同步开销，但精度收益显著
- 同步会引入轻微性能损失，但对于正确性是必需的

**Source operators**: foreach_add_scalar, gemma_rms_norm, modulate

---

## Variant A: Pipeline Barrier同步控制
Source: foreach_add_scalar

专家实现在关键计算节点插入了PipeBarrier<PIPE_V>()进行向量管道同步，确保数据转换和计算操作的正确顺序。这种细粒度的同步控制在BF16的Cast-Compute-Cast流程中尤为重要，防止了数据竞争和乱序执行导致的精度问题。PipeBarrier确保前一步操作完全完成后才执行下一步，虽然会引入一定的性能开销，但对于精度敏感的场景是必要的。baseline实现完全没有使用pipeline barrier，在复杂场景下可能出现同步问题。

**Expert implementation:**
```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
op(float32Tensor[offset], float32Tensor, scalarVal, dataCount);
PipeBarrier<PIPE_V>();
Cast(outLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

Benefit: 确保计算顺序正确性，防止数据竞争，保证数值精度
Trade-off: 引入同步开销，可能降低流水线效率

---

## Variant B: 分阶段 rstd 计算与同步
Source: gemma_rms_norm

为避免精度损失，rstd（reciprocal square root）的计算采用分阶段策略：先计算 mean = reduce_sum(x^2) / num_col（FP32），加 epsilon：mean + epsilon（FP32），开方：sqrt(mean + epsilon)（FP32），求倒数：1 / sqrt(...)（FP32）。每步之间通过 PipeBarrier 确保前一步完成，避免指令重排序导致的精度问题。

**Expert implementation:**
```cpp
// 专家实现 - 分阶段计算
Muls(rstdLocal, rstdLocal, avg_factor, num);   // 乘以平均因子
PipeBarrier<PIPE_V>();
Adds(rstdLocal, rstdLocal, epsilon, num);      // 加 epsilon
PipeBarrier<PIPE_V>();
Sqrt(rstdLocal, rstdLocal, num);               // 开方
Duplicate(reduce_buf_local, ONE, num);
PipeBarrier<PIPE_V>();
Div(rstdLocal, reduce_buf_local, rstdLocal, num);  // 求倒数得到 rstd
PipeBarrier<PIPE_V>();
```

**vs. baseline:**
```cpp
// baseline - 简单计算
float rmsVal = rowMeanSq + this->eps;
float rms = sqrt(rmsVal);
```

Benefit: 每步显式同步，确保计算顺序，避免指令重排序导致的精度问题
Trade-off: 增加了同步开销，但精度收益显著

---

## Variant C: PipeBarrier同步
Source: modulate

专家实现在关键计算步骤间插入PipeBarrier<PIPE_V>()同步指令，确保数据一致性。AscendC采用异步执行模型，Vector单元和Memory单元可并行执行，不加同步可能导致数据竞争。专家实现在类型转换后同步（Cast后插入PipeBarrier确保转换完成）、逐行计算间同步（每行计算后插入PipeBarrier）、最后同步（所有计算完成后同步确保Cast回写完成）。

**Expert implementation:**
```cpp
// 专家实现: 关键位置插入PipeBarrier
Cast(xLocalFp32, xLocal[this->ubLength], RoundMode::CAST_NONE, handleL * this->alignedD);
PipeBarrier<PIPE_V>();
for (int64_t jL = 0; jL < handleL; ++jL) {
    Mul(xLocalFp32[jL * this->alignedD], xLocalFp32[jL * this->alignedD], scaleLocalFp32, opCopyLength);
    PipeBarrier<PIPE_V>();
    Add(yLocalFp32[jL * this->alignedD], xLocalFp32[jL * this->alignedD], shiftLocalFp32, opCopyLength);
}
PipeBarrier<PIPE_V>();
Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, handleL * this->alignedD);
```

**vs. baseline:**
```cpp
// baseline: 无显式同步
AscendC::Muls(yLocal[offset], xLocal[offset], 1.0f, inputD);
AscendC::Mul(yLocal[offset], yLocal[offset], scaleLocal, inputD);
AscendC::Add(yLocal[offset], yLocal[offset], xLocal[offset], inputD);
AscendC::Add(yLocal[offset], yLocal[offset], shiftLocal, inputD);
```

Benefit: 确保数值计算正确性，避免异步执行导致的数据竞争
Trade-off: 同步会引入轻微性能损失，但对于正确性是必需的
