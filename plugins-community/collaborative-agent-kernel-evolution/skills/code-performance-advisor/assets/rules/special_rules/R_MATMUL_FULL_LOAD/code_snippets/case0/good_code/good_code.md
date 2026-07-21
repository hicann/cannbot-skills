# Good Code

```cpp
// 不分核全载：左矩阵驻留 L1，M方向不分核
for (uint64_t nDim = 0; nDim < nTileCnt; ++nDim) {
    // 外层一次性搬运左矩阵到 L1
    Load aL1();  // 左矩阵仅搬运 1 次，驻留 L1
    
    for (uint64_t kIter = 0; kIter < kL1Iter; ++kIter) {
        Load bL1();  // 右矩阵正常搬运
        for (uint64_t kL0 = 0; kL0 < kL0Iter; ++kL0) {
            Load aL0();  // 从 L1 读取
            Load bL0();
            Mmad();
        }
    }
    Fixpipe();
}
```