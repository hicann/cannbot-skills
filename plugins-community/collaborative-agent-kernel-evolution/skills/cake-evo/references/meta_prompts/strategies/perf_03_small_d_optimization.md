# P3: Small-D Multi-Row Merging (小D多行合并优化)
## Overview
当embedding_dim ≤ 512时，专家实现采用完全不同的算法：索引排序+连续处理。传统方法按原始顺序处理，相同索引分散在不同位置，导致频繁的GM随机访问。小维度优化先将索引排序（使用Sort指令），使相同索引连续，然后合并处理。具体流程：CreateVecIndex生成位置索引[0,1,2,...,n]；Sort对索引值排序，同时重排位置索引；Extract提取排序后的位置；按排序后的顺序处理，相同索引的梯度连续累加。这种优化将随机访问转化为顺序访问，大幅提升内存局部性。embedding_dim=512的阈值选择基于UB容量：512 * sizeof(float) = 2KB，可以在UB中缓存多行。

## When to Use
- Hidden size ≤ 640 or small D
- 对于小batch推理场景，单核性能达到极致，减少GM访问次数
- 针对不同数据规模采用最优切分策略，最大化数据局部性
- 小维度场景（如vocab_size小的embedding）性能提升2-5x

## Trade-off
- 仅适用于特定场景（每核1行），通用性降低
- 增加了Tiling逻辑的复杂度
- 排序需要额外的UB空间和计算（O(n log n)），大维度时不划算

**Source operators**: add_rms_norm_cast, clipped_swiglu, embedding_dense_grad_v2, modulate, rms_norm_grad

---

## Variant A: SingleN极致单核优化
Source: add_rms_norm_cast

SingleN模式针对每核只处理1行数据的场景进行了极致优化。使用单个大的UB缓冲区（MAXBUF = 195584 bytes），将所有中间结果保存在UB中，避免多次GM访问。优化特点包括：预分配大块UB内存，通过偏移量管理不同缓冲区；精心设计的指令流水线，最大化指令级并行；使用ReinterpretCast灵活复用内存。这对于小batch推理场景（如LLM解码阶段），单核性能达到极致。

**Expert implementation:**
```cpp
// SingleN模式极致优化
static constexpr int32_t MAXBUF = 195584;  // (192 - 1) * 1024 byte
Ppipe->InitBuffer(unitBuf, MAXBUF);
LocalTensor<float> ubLocal = unitBuf.Get<float>();
LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
LocalTensor<T> x1Local = xLocal[0];
LocalTensor<T> x2Local = xLocal[ubFactor];
LocalTensor<float> xFp32Local = ubLocal[ubFactor];
LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
```

**vs. baseline:**
```cpp
// 通用处理，无特殊优化
for (uint32_t rowIdx = 0; rowIdx < this->rowsPerCore; rowIdx++) {
    for (uint32_t tileId = 0; tileId < this->nTiles; tileId++) {
        CopyInPass1(rowIdx, tileId);
        float tileSqSum = ComputePass1();
        rowSqSum += tileSqSum;
    }
}
```

Benefit: 对于小batch推理场景，单核性能达到极致，减少GM访问次数
Trade-off: 仅适用于特定场景（每核1行），通用性降低

---

## Variant B: 长短H场景优化
Source: clipped_swiglu

根据H维度（hidden dimension）的大小，专家实现区分两种场景：1)小2H场景（isLongH=0）：H较小，一个batch的数据可以完全放入UB，采用batch内循环策略；2)大2H场景（isLongH=1）：H较大，需要切分H维度，采用dim2H内循环策略。这种区分允许针对不同的数据规模采用最优的切分策略，最大化数据局部性和并行效率。

**Expert implementation:**
```cpp
if (tl_->isLongH == 0) { // 小2H
    int64_t batchSpace = SWI_FACTOR * DTYPE_FACTOR * AlignBytes(dimH_ * sizeof(bfloat16_t));
    int64_t ubMaxBatch = xQueSpace_ / batchSpace;
    loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
} else { // 大2H
    loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
    pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
    pairFrontLoop_ = ubMaxPair_;
}
```

Benefit: 针对不同数据规模采用最优切分策略，最大化数据局部性
Trade-off: 增加了Tiling逻辑的复杂度

---

## Variant C: 小维度排序优化
Source: embedding_dense_grad_v2

当embedding_dim ≤ 512时，专家实现采用完全不同的算法：索引排序+连续处理。传统方法按原始顺序处理，相同索引分散在不同位置，导致频繁的GM随机访问。小维度优化先将索引排序（使用Sort指令），使相同索引连续，然后合并处理。具体流程：CreateVecIndex生成位置索引[0,1,2,...,n]；Sort对索引值排序，同时重排位置索引；Extract提取排序后的位置；按排序后的顺序处理，相同索引的梯度连续累加。这种优化将随机访问转化为顺序访问，大幅提升内存局部性。embedding_dim=512的阈值选择基于UB容量：512 * sizeof(float) = 2KB，可以在UB中缓存多行。

**Expert implementation:**
```cpp
__aicore__ inline void SortIndices(bool formerFlag)
{
    // 1. Cast indices to fp32
    Duplicate<CT>(tmp2Local, -1, idxAlign32);
    Cast(tmp2Local, indicesLocal, RoundMode::CAST_ROUND, idxNum);
    
    // 2. Create position indices
    CreateVecIndex<int32_t>(idxLocal, 0U, idxNum);
    
    // 3. Sort indices
    Sort<float, true>(sortResLocal, tmp2Local, idxULocal, tmpLocal, sortRepeatTimes);
    
    // 4. Extract sorted positions
    Extract(tmp2Local, idxULocal, sortResLocal, sortRepeatTimes);
    
    // 5. Cast back to int
    Cast(indicesLocal, tmp2Local, RoundMode::CAST_ROUND, idxNum);
}
```

Benefit: 小维度场景（如vocab_size小的embedding）性能提升2-5x
Trade-off: 排序需要额外的UB空间和计算（O(n log n)），大维度时不划算

---

## Variant D: 小D维度多行合并处理
Source: modulate

当D维度较小时，单次搬入一行数据无法充分利用UB空间和带宽。专家实现采用多行合并处理策略：判断条件isDSmall=(!useDTiling)&&(DId==0)&&(opCopyLength*sizeof(T)<MIN_DLENGTH)，计算行数handleL=min(loopLEnd-loopLStart,ubLength/alignedD)，批量处理一次CopyIn/Compute/CopyOut处理handleL行数据。这种优化提高UB利用率，减少GM访问次数，提高计算密度。

**Expert implementation:**
```cpp
// 专家实现: 小D维度优化
bool isDSmall = (!this->useDTiling) && (DId == 0) && (opCopyLength * sizeof(T) < MIN_DLENGTH);
int64_t handleL = isDSmall ? min(loopLEnd - loopLStart, this->ubLength / this->alignedD) : 1;
for (int64_t curL = loopLStart; curL < loopLEnd; ++curL) {
    handleL = (curL + handleL > loopLEnd) ? loopLEnd - curL : handleL;
    CopyIn(offset, opCopyLength, handleL);
    // 逐行计算
    for (int64_t jL = 0; jL < handleL; ++jL) {
        Mul(xLocal[jL * this->alignedD], xLocal[jL * this->alignedD], scaleLocal, opCopyLength);
        Add(yLocal[jL * this->alignedD], xLocal[jL * this->alignedD], shiftLocal, opCopyLength);
    }
    CopyOut(offset, opCopyLength, handleL);
    curL += handleL - 1;
}
```

**vs. baseline:**
```cpp
// baseline: 逐行处理
for (int64_t l = 0; l < inputL; l += tileSize) {
    int64_t curTileSize = (l + tileSize > inputL) ? (inputL - l) : tileSize;
    CopyIn(b, l, curTileSize);
    Compute(curTileSize);
    CopyOut(b, l, curTileSize);
}
```

Benefit: 当D<2048时性能提升20-40%，特别是小D大L场景
Trade-off: 需要维护额外的循环逻辑和offset计算

---

## Variant E: 小D优化(Small-D Optimization)
Source: rms_norm_grad

当列维度较小时(<=640)，专家实现采用特殊的优化策略：使用ComputeSmallD函数替代通用的Compute；利用Broadcast指令在UB中构建2D数据布局；通过ReduceSumMultiN一次性处理多行的reduce操作；减少重复的标量操作，提升向量化程度。这种优化显著提升了小hidden size场景（如Transformer中的前馈层）的性能。

**Expert implementation:**
```cpp
if (colValAlign_ > SMALLD_THRESHOLD) {
    // 普通路径
} else {
    CopyIn(loopIdx * ubFactor_, calcLen);
    ComputeSmallD(calcLen, gammaLocal, dgammaLocal);
    CopyOut(loopIdx * ubFactor_, calcLen);
}
```

Benefit: 小hidden size场景性能提升显著；提升向量化程度
Trade-off: 需要维护额外的代码路径；增加代码复杂度
