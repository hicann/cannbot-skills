# Base Code

```cpp
// 普通模板：左矩阵被重复搬运
for (uint64_t mDim = 0; mDim < mTileCnt; ++mDim) {
    for (uint64_t nDim = 0; nDim < nTileCnt; ++nDim) {
        // 每个 nDim 迭代都重新搬运左矩阵
        for (uint64_t kIter = 0; kIter < kL1Iter; ++kIter) {
            Load aL1();  // 左矩阵从 HBM/L2 重复搬运
            Load bL1();  // 右矩阵搬运
            for (uint64_t kL0 = 0; kL0 < kL0Iter; ++kL0) {
                Load aL0();
                Load bL0();
                Mmad();
            }
        }
        Fixpipe();
    }
}
```