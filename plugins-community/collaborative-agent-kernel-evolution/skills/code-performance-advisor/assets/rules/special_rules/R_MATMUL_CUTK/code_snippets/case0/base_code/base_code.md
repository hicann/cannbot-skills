# Base Code

```cpp
// 普通模板：只切分 M、N 轴
tileNum = Div(m, baseM) * Div(n, baseN);
for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum) {
    // 每个核独立完成整个 K 轴的计算
    for (uint64_t kIter = 0; kIter < kL1Iter; ++iter) {
        CopyInA1();
        CopyInB1();
        for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            CopyInA2();
            CopyInB2();
            Mmad();
        }
    }
    CopyOut();
}
```