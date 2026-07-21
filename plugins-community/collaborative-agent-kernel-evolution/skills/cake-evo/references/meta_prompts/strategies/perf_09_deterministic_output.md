# P9: Deterministic Output via Workspace (确定性输出)
## Overview
在多核并行场景下，如果多个核同时更新同一权重行，由于执行顺序不确定，最终结果虽然是正确的（原子加），但浮点累加顺序不同可能导致微小差异（非结合性）。这在分布式训练中会导致不同节点的梯度不一致，影响收敛。确定性模式通过反向处理和边界检测解决这一问题：每个核从后向前处理自己的数据块；检测与前一个核的边界索引是否相同；如果相同，当前核跳过该索引，由前一个核处理；通过SyncAll保证全局顺序。

## When to Use
- Training ops needing reproducibility
- 双输出量化性能接近单输出的2倍，而非简单重复的2倍开销
- 保证bit-wise确定性，分布式训练可复现性100%
- 保证多核场景下梯度累加的正确性

## Trade-off
- 增加GM访存，需要额外的Workspace内存
- 约5-10%的性能损失（边界检测和反向遍历的开销）
- 原子操作有一定性能开销，但这是正确性必需的

**Source operators**: add_rms_norm_dynamic_quant, embedding_dense_grad_v2, multi_scale_deformable_attention_grad, multi_scale_deformable_attn_function, rms_norm_grad

---

## Variant A: 双输出量化的Workspace优化
Source: add_rms_norm_dynamic_quant

对于双输出量化场景，专家实现使用Workspace存储中间结果以避免重复计算。在SLICE_D策略中，第一次遍历计算RMS并将归一化结果写入Workspace，第二次遍历从Workspace读取并应用不同的smooth scale进行量化。这种设计避免了RMS计算和权重应用的重复执行。

**Expert implementation:**
```cpp
if (this->oldDouble || (this->outQuant2Flag == 1 || this->outQuant1Flag == 1)) {
    workspaceGm.SetGlobalBuffer((__gm__ float*)(workspace) + 2 * this->blockIdx_ * this->numLastDim);
}
CopyOutSmoothNorm(yLocalFp32, 0, rowGmOffset, elementCount);  // 写入smooth1结果
CopyOutSmoothNorm(zLocalFp32, this->numLastDim, rowGmOffset, elementCount);  // 写入smooth2结果
CopyInSmoothNorm(xLocalFp32, 0, rowGmOffset, elementCount, this->localMax1);  // 读取
```

Benefit: 双输出量化性能接近单输出的2倍，而非简单重复的2倍开销
Trade-off: 增加GM访存，需要额外的Workspace内存

---

## Variant B: 确定性计算模式
Source: embedding_dense_grad_v2

在多核并行场景下，如果多个核同时更新同一权重行，由于执行顺序不确定，最终结果虽然是正确的（原子加），但浮点累加顺序不同可能导致微小差异（非结合性）。这在分布式训练中会导致不同节点的梯度不一致，影响收敛。确定性模式通过反向处理和边界检测解决这一问题：每个核从后向前处理自己的数据块；检测与前一个核的边界索引是否相同；如果相同，当前核跳过该索引，由前一个核处理；通过SyncAll保证全局顺序。

**Expert implementation:**
```cpp
__aicore__ inline bool SubProcess(const uint64_t dimJ)
{
    // 反向处理: 从后向前
    for (int64_t j = processRowNum_ - 1; j >= 0; j--) {
        if (!CopyIn(j, true, dimJ)) {
            // 检测到与前一个核的边界冲突
            isNeedContinueCopy = j == processRowNum_ - 1 ? false : true;
            if (addParam_.lastIndices != INDICE_INIT_PARAM) {
                CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
            }
            break;
        }
        ComputeAndCopyOut(j, 0, dimJ);
    }
    return isNeedContinueCopy;
}

__aicore__ inline bool CopyIn(const uint64_t progress, const bool flag, const uint64_t dimJ)
{
    PIPE_MTE2_S();
    // 检查是否与前一个核的最后一个索引相同
    if ((indiceLocal.GetValue(0) == standIdice_) == flag) {
        return false;  // 冲突，跳过
    }
}
```

Benefit: 保证bit-wise确定性，分布式训练可复现性100%
Trade-off: 约5-10%的性能损失（边界检测和反向遍历的开销）

---

## Variant C: SetAtomicAdd处理多核竞争
Source: multi_scale_deformable_attention_grad

由于多核可能同时写入gradValueGm的同一位置，专家实现使用SetAtomicAdd开启原子累加模式。这是保证梯度计算正确性的关键。在GridSampleCompute函数中，先开启原子模式，执行所有梯度累加操作，然后关闭原子模式。这种设计确保了即使多个核心同时更新同一个value位置的梯度，结果也是正确的。baseline实现没有处理这个问题，在多核场景下可能产生错误结果。

**Expert implementation:**
```cpp
SetAtomicAdd<DTYPE_VALUE>();
for (query = 0; query < thisCycleNum; query++) {
    ComputeGradTogether(distH, distW, w1, w2, w3, w4, attenWeight);
    // 多核安全地累加梯度
    DataCopyPad(gradValueGm[offsetValue + hPtrOffset + wPtrOffset], mid[queryOffset], copyOutParams);
}
SetAtomicNone();
```

**vs. baseline:**
```cpp
// baseline无原子操作
gradValueGm.SetValue(locOffset, gradLocX);
```

Benefit: 保证多核场景下梯度累加的正确性
Trade-off: 原子操作有一定性能开销，但这是正确性必需的

---

## Variant D: 16MB Workspace预留
Source: multi_scale_deformable_attention_grad

专家实现预留了16MB的系统workspace，这是为框架级别的内存管理预留的空间。虽然当前实现可能不直接使用这些workspace，但这种预留确保了算子可以在复杂的图执行环境中正确运行，避免内存冲突。

**Expert implementation:**
```cpp
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;
size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
size_t *currentWorkspace = TilingContext->GetWorkspaceSizes(1);
currentWorkspace[0] = sysWorkspaceSize;
```

**vs. baseline:**
```cpp
// baseline无workspace预留
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = 0;
```

Benefit: 确保在复杂图执行环境中正确运行
Trade-off: 占用一定内存资源

---

## Variant E: 多核并行与任务分配
Source: multi_scale_deformable_attn_function

Host端通过TilingPrepare4MultiScaleDeformableAttnFunction获取硬件平台信息（核心数、UB大小），并据此进行任务分配。InitTask函数实现了基于numQueries的负载均衡分配，确保每个核心处理相近数量的查询。deterministicFlag标志支持确定性计算模式（单核执行），用于调试和结果对比。

**Expert implementation:**
```cpp
// Host端获取硬件信息
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
compileInfo->isInfBase = (socVersion == platform_ascendc::SocVersion::ASCEND310P) ? true : false;

// 确定性模式支持
deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
if (deterministicFlag == 1) {
    coreNum = 1;
}

// Kernel端任务分配
__aicore__ inline void InitTask() {
    uint32_t avgTasks = numQueries_ / coreNum_;
    uint32_t remainTasks = numQueries_ % coreNum_;
    startOffset_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
    endOffset_ = startOffset_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
}
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 实现负载均衡的多核并行，支持确定性模式用于调试
Trade-off: 增加了任务分配逻辑的复杂性

---

## Variant F: 确定性输出支持(Deterministic Output)
Source: rms_norm_grad

专家实现通过fixed_output标志支持确定性输出模式。非确定性模式：使用AtomicAdd直接累加dgamma，性能高但结果可能因调度顺序而异。确定性模式：先将部分结果写入workspace，最后通过同步后的reduce得到确定结果。这种设计允许用户在性能和确定性之间做出选择，满足不同应用场景的需求。

**Expert implementation:**
```cpp
if (isDeterministic_ == 1) {
    CopyDgammaOutWorkspace();
    SyncAll();
    doDeter();
} else {
    CopyDgammaOut();
}
```

Benefit: 满足需要可重复结果的场景（如模型训练调试）；提供性能和确定性的选择
Trade-off: 确定性模式需要额外的workspace内存和同步开销
